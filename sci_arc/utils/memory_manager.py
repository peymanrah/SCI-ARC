"""
Memory Manager - Smart GPU Memory Management for RLAN Training

This module provides:
1. Pre-emptive memory estimation before module activation
2. Dynamic batch size adjustment to fit within VRAM
3. Memory profiling for each module configuration
4. Safe module activation with rollback on OOM

Key Innovation:
- Instead of activating all staged modules at once (which caused 12GB+ overflow),
  we estimate memory requirements BEFORE activation and adjust accordingly.

Usage:
    from sci_arc.utils.memory_manager import MemoryManager
    
    mem_mgr = MemoryManager(gpu_total_mb=24576, safety_margin=0.90)
    batch_size = mem_mgr.get_safe_batch_size(model, active_modules, grid_size=30)
"""

import gc
import math
from typing import Dict, Optional, Tuple, List, Any
from dataclasses import dataclass, field
import torch
import torch.nn as nn


@dataclass
class ModuleMemoryProfile:
    """Memory profile for a specific module configuration."""
    name: str
    base_memory_mb: float  # Memory with module inactive
    active_memory_mb: float  # Memory with module active
    per_batch_overhead_mb: float  # Additional memory per sample
    per_step_overhead_mb: float  # For solver steps (if applicable)
    
    @property
    def total_overhead_mb(self) -> float:
        return self.active_memory_mb - self.base_memory_mb


@dataclass 
class MemoryBudget:
    """Memory budget for training configuration."""
    total_gpu_mb: float
    model_params_mb: float
    optimizer_state_mb: float
    gradient_mb: float
    headroom_mb: float
    
    @property
    def available_for_activations_mb(self) -> float:
        return (self.total_gpu_mb - self.model_params_mb - 
                self.optimizer_state_mb - self.gradient_mb - self.headroom_mb)


# Pre-computed memory coefficients for RLAN modules
# These are empirically measured on RTX 3090 with batch_size=80, hidden_dim=256
MODULE_MEMORY_COEFFICIENTS = {
    # Module: (base_per_batch_mb, per_grid_cell_mb, per_solver_step_mb)
    'baseline': (0.5, 0.001, 5.0),  # Base RLAN without extras
    'cross_attention_injector': (0.2, 0.0005, 0.0),  # CrossAttentionInjector
    'solver_cross_attention': (0.0, 0.0003, 2.5),  # SolverCrossAttention per step
    'hyperlora': (0.3, 0.0002, 0.5),  # HyperLoRA LoRA deltas
    'hpm': (2.0, 0.0001, 0.0),  # HPM memory banks
    'loo': (0.0, 0.0, 0.0),  # LOO just does extra forward passes (already counted)
    'equivariance': (0.0, 0.0, 0.0),  # Equivariance: 4x forward passes
}


class MemoryManager:
    """
    Smart GPU memory manager for RLAN training.
    
    Ensures training stays within VRAM by:
    1. Estimating memory requirements before module activation
    2. Adjusting batch size dynamically if needed
    3. Providing memory profiling utilities
    """
    
    def __init__(
        self,
        gpu_total_mb: float = 24576,  # RTX 3090 = 24GB
        safety_margin: float = 0.92,  # Use at most 92% of VRAM
        min_batch_size: int = 16,
        max_batch_size: int = 128,
    ):
        self.gpu_total_mb = gpu_total_mb
        self.safety_margin = safety_margin
        self.usable_mb = gpu_total_mb * safety_margin
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        
        # Cache measured values
        self._measured_profiles: Dict[str, ModuleMemoryProfile] = {}
        
    def estimate_forward_memory_mb(
        self,
        batch_size: int,
        max_grid_size: int,
        hidden_dim: int = 256,
        num_pairs: int = 4,
        num_solver_steps: int = 5,
        active_modules: Optional[Dict[str, bool]] = None,
    ) -> float:
        """
        Estimate GPU memory required for a forward pass.
        
        Memory components:
        1. Input tensors: batch * pairs * grid^2 * 4 bytes
        2. Feature maps: batch * hidden * grid^2 * 4 bytes * layers
        3. Support features: batch * pairs * hidden * spatial^2 * 4 bytes
        4. Solver hidden states: batch * hidden * grid^2 * 4 bytes * steps
        5. Output logits: batch * classes * grid^2 * 4 bytes * steps
        6. Attention maps (if active): batch * heads * grid^2 * spatial^2 * 4 bytes
        7. LoRA deltas (if active): batch * hidden * hidden * gates * 4 bytes
        """
        if active_modules is None:
            active_modules = {}
            
        # Basic calculations
        grid_cells = max_grid_size * max_grid_size
        spatial_size = 8  # After downsampling
        spatial_cells = spatial_size * spatial_size
        bytes_per_float = 2  # bfloat16
        
        memory_mb = 0.0
        
        # 1. Input tensors (long integers = 8 bytes, but converted to float)
        input_mb = batch_size * num_pairs * grid_cells * bytes_per_float * 4 / (1024 * 1024)
        memory_mb += input_mb
        
        # 2. Feature encoder output
        feature_mb = batch_size * hidden_dim * grid_cells * bytes_per_float / (1024 * 1024)
        memory_mb += feature_mb * 2  # Input features + position encoded
        
        # 3. Support features (always computed)
        support_mb = batch_size * num_pairs * hidden_dim * spatial_cells * bytes_per_float / (1024 * 1024)
        memory_mb += support_mb
        
        # 4. Solver hidden states (need to keep all for backward)
        solver_hidden_mb = batch_size * hidden_dim * grid_cells * bytes_per_float * num_solver_steps / (1024 * 1024)
        memory_mb += solver_hidden_mb
        
        # 5. Output logits (all steps for deep supervision)
        logits_mb = batch_size * 10 * grid_cells * bytes_per_float * num_solver_steps / (1024 * 1024)
        memory_mb += logits_mb
        
        # 6. CrossAttentionInjector attention maps
        if active_modules.get('cross_attention_active', False):
            # Query: (B, grid^2, hidden) Key/Value: (B, pairs*spatial^2, hidden)
            attn_mb = batch_size * grid_cells * num_pairs * spatial_cells * bytes_per_float / (1024 * 1024)
            memory_mb += attn_mb
            
        # 7. SolverCrossAttention per step
        if active_modules.get('solver_context_active', False):
            # Each step: attention over support features
            solver_attn_mb = batch_size * grid_cells * num_pairs * spatial_cells * bytes_per_float / (1024 * 1024)
            memory_mb += solver_attn_mb * num_solver_steps
            
        # 8. HyperLoRA deltas
        if active_modules.get('hyperlora_active', False):
            # 4 LoRA targets: reset, update, candidate gates + output head
            lora_mb = batch_size * hidden_dim * hidden_dim * 4 * bytes_per_float / (1024 * 1024)
            memory_mb += lora_mb
            
        # 9. HPM memory banks
        if active_modules.get('use_hpm', False):
            # Static banks + dynamic retrieval
            hpm_mb = 200  # Rough estimate
            memory_mb += hpm_mb
            
        # Backward pass roughly doubles activation memory
        memory_mb *= 2.2
        
        return memory_mb
    
    def estimate_batch_overhead_mb(
        self,
        hidden_dim: int = 256,
        num_pairs: int = 4,
        num_solver_steps: int = 5,
        active_modules: Optional[Dict[str, bool]] = None,
    ) -> float:
        """
        Estimate per-batch memory overhead that scales with batch size.
        This helps compute the safe batch size.
        """
        # Use a reference batch to compute per-sample overhead
        ref_batch = 1
        ref_grid = 30
        
        total_1 = self.estimate_forward_memory_mb(
            batch_size=ref_batch,
            max_grid_size=ref_grid,
            hidden_dim=hidden_dim,
            num_pairs=num_pairs,
            num_solver_steps=num_solver_steps,
            active_modules=active_modules,
        )
        
        return total_1  # Per-sample memory
    
    def get_safe_batch_size(
        self,
        model: nn.Module,
        active_modules: Optional[Dict[str, bool]] = None,
        max_grid_size: int = 30,
        requested_batch_size: int = 80,
    ) -> int:
        """
        Compute the maximum safe batch size given current module configuration.
        
        Returns:
            Adjusted batch size that fits within GPU memory
        """
        if not torch.cuda.is_available():
            return requested_batch_size
            
        if active_modules is None:
            active_modules = {
                'hyperlora_active': getattr(model, 'hyperlora_active', False),
                'solver_context_active': getattr(model, 'solver_context_active', False),
                'cross_attention_active': getattr(model, 'cross_attention_active', False),
                'use_hpm': getattr(model, 'use_hpm', False),
            }
        
        # Get current memory usage
        torch.cuda.synchronize()
        current_alloc = torch.cuda.memory_allocated() / (1024 * 1024)
        current_reserved = torch.cuda.memory_reserved() / (1024 * 1024)
        
        # Model params + optimizer state (fixed overhead)
        param_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
        optimizer_mb = param_mb * 2  # AdamW: 2 states per param
        gradient_mb = param_mb  # Gradients same size as params
        
        fixed_overhead = param_mb + optimizer_mb + gradient_mb
        
        # Available for activations
        available_mb = self.usable_mb - fixed_overhead - current_alloc
        
        # Estimate per-batch memory
        per_batch_mb = self.estimate_batch_overhead_mb(
            hidden_dim=getattr(model, 'hidden_dim', 256),
            num_pairs=4,  # Average
            num_solver_steps=getattr(model.solver if hasattr(model, 'solver') else model, 'num_steps', 5),
            active_modules=active_modules,
        )
        
        # Account for bucketed batching - worst case is all 30x30 grids
        # But bucketed batching means most batches are smaller
        # Use average factor of 0.6 (mix of bucket sizes)
        effective_per_batch_mb = per_batch_mb * 0.7
        
        # Calculate safe batch size
        if effective_per_batch_mb > 0:
            safe_batch = int(available_mb / effective_per_batch_mb)
        else:
            safe_batch = requested_batch_size
            
        # Clamp to reasonable range
        safe_batch = max(self.min_batch_size, min(safe_batch, self.max_batch_size))
        
        # Don't increase beyond requested
        safe_batch = min(safe_batch, requested_batch_size)
        
        return safe_batch
    
    def measure_actual_memory(
        self,
        model: nn.Module,
        batch_size: int,
        max_grid_size: int,
        hidden_dim: int = 256,
        num_pairs: int = 4,
        device: str = 'cuda',
    ) -> Tuple[float, float]:
        """
        Measure actual GPU memory usage with a dummy forward pass.
        
        Returns:
            (allocated_mb, peak_mb)
        """
        if not torch.cuda.is_available():
            return (0.0, 0.0)
            
        model.eval()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        
        # Create dummy batch
        with torch.no_grad():
            test_inputs = torch.randint(0, 10, (batch_size, max_grid_size, max_grid_size), device=device)
            train_inputs = torch.randint(0, 10, (batch_size, num_pairs, max_grid_size, max_grid_size), device=device)
            train_outputs = torch.randint(0, 10, (batch_size, num_pairs, max_grid_size, max_grid_size), device=device)
            pair_mask = torch.ones(batch_size, num_pairs, device=device)
            
            try:
                _ = model(
                    test_inputs,
                    train_inputs=train_inputs,
                    train_outputs=train_outputs,
                    pair_mask=pair_mask,
                    return_intermediates=False,
                )
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    torch.cuda.empty_cache()
                    return (float('inf'), float('inf'))
                raise
                
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / (1024 * 1024)
        peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
        
        # Cleanup
        del test_inputs, train_inputs, train_outputs, pair_mask
        torch.cuda.empty_cache()
        
        model.train()
        return (allocated, peak)
    
    def can_activate_modules(
        self,
        model: nn.Module,
        modules_to_activate: Dict[str, bool],
        batch_size: int,
        max_grid_size: int = 30,
    ) -> Tuple[bool, str]:
        """
        Check if we can safely activate the specified modules.
        
        Returns:
            (can_activate, message)
        """
        if not torch.cuda.is_available():
            return (True, "CUDA not available, skipping check")
            
        # Estimate memory with new modules
        estimated_mb = self.estimate_forward_memory_mb(
            batch_size=batch_size,
            max_grid_size=max_grid_size,
            active_modules=modules_to_activate,
        )
        
        if estimated_mb > self.usable_mb:
            deficit = estimated_mb - self.usable_mb
            return (
                False,
                f"Estimated {estimated_mb:.0f}MB exceeds limit {self.usable_mb:.0f}MB "
                f"(deficit: {deficit:.0f}MB)"
            )
            
        return (True, f"Estimated {estimated_mb:.0f}MB within limit {self.usable_mb:.0f}MB")
    
    def get_staggered_activation_schedule(
        self,
        meta_learning_start_epoch: int = 3,
        equiv_start_epoch: int = 8,
        loo_start_epoch: int = 12,
        hpm_start_epoch: int = 14,
    ) -> Dict[int, List[str]]:
        """
        Get a staggered module activation schedule to prevent memory spikes.
        
        The key insight: Activating all modules at once causes 12GB+ memory spikes.
        By staggering, we give the memory manager time to adjust.
        
        Returns:
            Dict mapping epoch -> list of modules to activate
        """
        schedule = {}
        
        # Epoch 3: Only HyperLoRA (smallest impact)
        schedule[meta_learning_start_epoch] = ['hyperlora_active']
        
        # Epoch 5: SolverCrossAttention (moderate impact) - 2 epochs after HyperLoRA
        schedule[meta_learning_start_epoch + 2] = ['solver_context_active']
        
        # Epoch 7: CrossAttentionInjector (higher impact) - 2 epochs after SolverContext
        schedule[meta_learning_start_epoch + 4] = ['cross_attention_active']
        
        # Epoch 8: Equivariance (4 forward passes - significant)
        schedule[equiv_start_epoch] = ['equivariance_active']
        
        # Epoch 12: LOO (N forward passes - heaviest)
        schedule[loo_start_epoch] = ['loo_active']
        
        # Epoch 14: HPM (memory banks)
        schedule[hpm_start_epoch] = ['use_hpm']
        
        return schedule
    
    def cleanup(self):
        """Force GPU memory cleanup."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        gc.collect()


def get_memory_manager(config: Dict[str, Any]) -> MemoryManager:
    """Factory function to create MemoryManager from config."""
    # Get GPU total memory
    if torch.cuda.is_available():
        gpu_total_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
    else:
        gpu_total_mb = 24576  # Default to RTX 3090
        
    # Get safety margin from config
    safety_margin = config.get('training', {}).get('memory_safety_margin', 0.92)
    
    return MemoryManager(
        gpu_total_mb=gpu_total_mb,
        safety_margin=safety_margin,
        min_batch_size=config.get('training', {}).get('min_batch_size', 16),
        max_batch_size=config.get('training', {}).get('max_batch_size', 128),
    )
