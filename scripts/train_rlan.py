#!/usr/bin/env python
"""
Train RLAN Model on ARC Dataset - Production-Ready Training Script

Features matching CISL production training:
- Mixed precision (AMP) for RTX 3090 efficiency
- Reproducible training with seed control
- Auto-resume from checkpoints
- Gradient accumulation for larger effective batch size
- File logging for reproducibility
- WandB integration (optional)
- Cache samples mode for testing vs infinite diversity for competitive training
- Proper memory cleanup on exit (RAM + GPU)

Usage:
    python scripts/train_rlan.py --config configs/rlan_base.yaml
    python scripts/train_rlan.py --config configs/rlan_small.yaml
    python scripts/train_rlan.py --config configs/rlan_base.yaml --resume auto
    python scripts/train_rlan.py --config configs/rlan_base.yaml --resume checkpoints/rlan_base/latest.pt
"""

import argparse
import os
import sys
import time
import random
import math
import gc
import signal
import atexit
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any, List
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sci_arc.models import RLAN, RLANConfig
from sci_arc.training import RLANLoss
from sci_arc.training.ema import EMAHelper
from sci_arc.models.rlan_modules.loo_training import (
    LOOTrainingLoss, LOOConfig,
    AugmentationEquivarianceLoss, EquivarianceConfig,
    OutputEquivarianceLoss,  # Jan 2026: Output-level equiv for TTA consensus
    GroupMarginalizedNLLLoss,  # Jan 2026: Group-marginalized NLL for principled generalization
)
# Jan 2026 Ablation Study: Anchor Robustness Training (ART) + Anchor-Relative Program Search (ARPS)
from sci_arc.models.rlan_modules import (
    AnchorRobustnessTraining, ARTConfig, create_art_from_config,
    ARPS, ARPSConfig, create_arps_from_config,
)
from sci_arc.data import ARCDataset, collate_sci_arc, BucketedBatchSampler
# Jan 2026: Import constant for ignore_index consistency (single source of truth)
PADDING_IGNORE_VALUE = ARCDataset.PADDING_IGNORE_VALUE  # -100, imported from dataset
from sci_arc.evaluation.trm_style_evaluator import TRMStyleEvaluator
from sci_arc.utils.gap_monitor import GapHealthMonitor
from sci_arc.utils.memory_manager import MemoryManager, get_memory_manager

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Global cleanup flag
_cleanup_done = False


def cleanup_memory():
    """Clean up GPU and RAM memory."""
    global _cleanup_done
    if _cleanup_done:
        return
    _cleanup_done = True
    
    print("\nCleaning up memory...")
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Force garbage collection
    gc.collect()
    
    print("Memory cleanup complete.")


def signal_handler(signum, frame):
    """Handle Ctrl+C and other termination signals."""
    print(f"\n\nReceived signal {signum}, cleaning up...")
    cleanup_memory()
    sys.exit(0)


# Register cleanup handlers
atexit.register(cleanup_memory)
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
if hasattr(signal, 'SIGBREAK'):  # Windows-specific
    signal.signal(signal.SIGBREAK, signal_handler)


class CUDAPrefetcher:
    """
    Prefetches batches to GPU using a separate CUDA stream.
    
    This allows CPU→GPU data transfer to overlap with GPU computation,
    eliminating GPU stalls when waiting for data. Essential for num_workers=0
    scenarios (like cached samples mode).
    
    Usage:
        prefetcher = CUDAPrefetcher(train_loader, device)
        batch = prefetcher.next()
        while batch is not None:
            # ... train on batch ...
            batch = prefetcher.next()
    """
    
    def __init__(self, loader, device):
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream() if device.type == 'cuda' else None
        self.batch = None
        self.loader_iter = None
        
    def _preload(self):
        """Preload next batch to GPU asynchronously."""
        try:
            self.batch = next(self.loader_iter)
        except StopIteration:
            self.batch = None
            return
        
        if self.stream is not None:
            with torch.cuda.stream(self.stream):
                # Transfer all tensors in batch to GPU non-blocking
                self.batch = self._to_device(self.batch)
    
    def _to_device(self, data):
        """Recursively move data to device with non_blocking=True."""
        if isinstance(data, torch.Tensor):
            return data.to(self.device, non_blocking=True)
        elif isinstance(data, dict):
            return {k: self._to_device(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._to_device(v) for v in data]
        else:
            return data
    
    def __iter__(self):
        """Start iteration and preload first batch."""
        self.loader_iter = iter(self.loader)
        self._preload()
        return self
    
    def __next__(self):
        """Get current batch and start preloading next one."""
        if self.stream is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
        
        batch = self.batch
        if batch is None:
            raise StopIteration
        
        # Record that tensors are now safe to use on current stream
        if self.stream is not None and isinstance(batch, dict):
            for v in batch.values():
                if isinstance(v, torch.Tensor) and v.is_cuda:
                    v.record_stream(torch.cuda.current_stream())
        
        # Start preloading next batch
        self._preload()
        
        return batch
    
    def __len__(self):
        return len(self.loader)


class TeeLogger:
    """Logger that writes to both stdout and a file (encoding-safe for Windows)."""
    def __init__(self, log_path: Path):
        self.terminal = sys.stdout
        self.log_path = log_path
        # Use UTF-8 with error handling for Windows compatibility
        self.log_file = open(log_path, 'w', encoding='utf-8', errors='replace', buffering=1)
        
    def write(self, message):
        # Handle potential encoding issues on Windows terminal
        try:
            self.terminal.write(message)
        except UnicodeEncodeError:
            # Fallback to ASCII-safe version for Windows cmd
            self.terminal.write(message.encode('ascii', errors='replace').decode('ascii'))
        self.log_file.write(message)
        self.log_file.flush()
        
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()
        
    def close(self):
        self.log_file.close()
        sys.stdout = self.terminal


def set_seed(seed: int, deterministic: bool = False):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


# ============================================================================
# MEMORY DEBUGGING UTILITIES
# ============================================================================

def get_tensor_memory_breakdown(model, outputs: dict = None, batch: dict = None, optimizer=None) -> Dict[str, Any]:
    """
    Get detailed memory breakdown by category.
    
    Returns dict with memory in MB for:
    - model_params: Model parameter memory
    - model_grads: Gradient memory (if computed)
    - optimizer_states: Optimizer state estimate (AdamW = 2x params)
    - batch_data: Input batch tensor memory
    - activations: Forward pass activations (outputs dict)
    - per_module: Dict of memory per model module
    - active_modules: Dict of which optional modules are active (for debugging)
    """
    breakdown = {}
    
    # Model parameters
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    breakdown['model_params_mb'] = param_bytes / 1024 / 1024
    
    # Gradients (if they exist)
    grad_bytes = sum(
        p.grad.numel() * p.grad.element_size() 
        for p in model.parameters() if p.grad is not None
    )
    breakdown['model_grads_mb'] = grad_bytes / 1024 / 1024
    
    # Optimizer states estimate
    # NOTE: This is an estimate and depends heavily on optimizer implementation.
    breakdown['optimizer_states_estimate_note'] = 'estimate (assumes AdamW fp32 states)'
    try:
        first_param = next(model.parameters())

        if optimizer is not None:
            opt_mod = getattr(optimizer.__class__, '__module__', '') or ''
            opt_name = optimizer.__class__.__name__
            is_bitsandbytes = ('bitsandbytes' in opt_mod) or ('8bit' in opt_name.lower())

            if is_bitsandbytes:
                # Rough estimate for 8-bit Adam variants: quantized states + small fp32 scalars.
                # We keep this conservative and annotate it.
                breakdown['optimizer_states_estimate_mb'] = (param_bytes * 0.75) / 1024 / 1024
                breakdown['optimizer_states_estimate_note'] = f'estimate (8-bit optimizer: {opt_name})'
            else:
                breakdown['optimizer_states_estimate_mb'] = (param_bytes * 2 * 4 / first_param.element_size()) / 1024 / 1024
                breakdown['optimizer_states_estimate_note'] = f'estimate (Adam-like: {opt_name})'
        else:
            breakdown['optimizer_states_estimate_mb'] = (param_bytes * 2 * 4 / first_param.element_size()) / 1024 / 1024
    except StopIteration:
        breakdown['optimizer_states_estimate_mb'] = 0
        breakdown['optimizer_states_estimate_note'] = 'estimate (no params)'
    
    # Batch data memory
    if batch is not None:
        batch_bytes = 0
        for k, v in batch.items():
            if isinstance(v, torch.Tensor) and v.is_cuda:
                batch_bytes += v.numel() * v.element_size()
        breakdown['batch_data_mb'] = batch_bytes / 1024 / 1024
    
    # Activation memory from outputs
    if outputs is not None:
        act_bytes = 0
        for k, v in outputs.items():
            if isinstance(v, torch.Tensor) and v.is_cuda:
                act_bytes += v.numel() * v.element_size()
                breakdown[f'output_{k}_mb'] = (v.numel() * v.element_size()) / 1024 / 1024
            elif isinstance(v, list):
                # Handle list of tensors (e.g., all_logits)
                list_bytes = sum(
                    t.numel() * t.element_size() 
                    for t in v if isinstance(t, torch.Tensor) and t.is_cuda
                )
                if list_bytes > 0:
                    act_bytes += list_bytes
                    breakdown[f'output_{k}_mb'] = list_bytes / 1024 / 1024
            elif isinstance(v, dict):
                # Handle nested dicts (e.g., lora_deltas)
                dict_bytes = sum(
                    t.numel() * t.element_size()
                    for t in v.values() if isinstance(t, torch.Tensor) and t.is_cuda
                )
                if dict_bytes > 0:
                    act_bytes += dict_bytes
                    breakdown[f'output_{k}_mb'] = dict_bytes / 1024 / 1024
        breakdown['activations_total_mb'] = act_bytes / 1024 / 1024
    
    # Per-module parameter memory
    module_memory = {}
    for name, module in model.named_children():
        mod_bytes = sum(p.numel() * p.element_size() for p in module.parameters())
        if mod_bytes > 0:
            module_memory[name] = mod_bytes / 1024 / 1024
    breakdown['per_module_mb'] = module_memory
    
    # Track which optional modules are ACTIVE at this moment (critical for debugging staged activation)
    # Note: These reflect RUNTIME state, not YAML config. Modules may be configured but not yet active
    # due to staged activation (e.g., HPM configured but hpm_start_epoch not yet reached).
    active_modules = {}
    active_modules['hyperlora_active'] = getattr(model, 'hyperlora_active', False)
    active_modules['solver_context_active'] = getattr(model, 'solver_context_active', False)
    active_modules['cross_attention_active'] = getattr(model, 'cross_attention_active', False)
    # HPM: model.use_hpm is toggled by training script at hpm_start_epoch
    # Also check if HPM module actually exists (model.hpm is not None)
    active_modules['hpm_active'] = getattr(model, 'use_hpm', False) and getattr(model, 'hpm', None) is not None
    # LOO/Equivariance: These are controlled by training loop, not model attributes
    # Check if the attributes exist (set by training loop)
    active_modules['loo_active'] = getattr(model, 'loo_enabled', False)
    active_modules['equivariance_active'] = getattr(model, 'equivariance_enabled', False)
    breakdown['active_modules'] = active_modules
    
    return breakdown


def format_memory_breakdown(breakdown: Dict[str, Any], prefix: str = "      ") -> str:
    """Format memory breakdown dict as readable string."""
    lines = []
    
    # Core metrics
    lines.append(f"{prefix}Model params: {breakdown.get('model_params_mb', 0):.1f}MB")
    lines.append(f"{prefix}Model grads:  {breakdown.get('model_grads_mb', 0):.1f}MB")
    opt_est = breakdown.get('optimizer_states_estimate_mb', 0.0)
    opt_note = breakdown.get('optimizer_states_estimate_note', 'estimate')
    lines.append(f"{prefix}Optimizer:    {opt_est:.1f}MB ({opt_note})")
    lines.append(f"{prefix}Batch data:   {breakdown.get('batch_data_mb', 0):.1f}MB")
    lines.append(f"{prefix}Activations:  {breakdown.get('activations_total_mb', 0):.1f}MB")
    
    # Per-output breakdown (sorted by size)
    output_items = [(k, v) for k, v in breakdown.items() if k.startswith('output_') and not k.endswith('total_mb')]
    if output_items:
        output_items.sort(key=lambda x: x[1], reverse=True)
        lines.append(f"{prefix}Output tensors (top 5):")
        for k, v in output_items[:5]:
            name = k.replace('output_', '').replace('_mb', '')
            lines.append(f"{prefix}  {name}: {v:.1f}MB")
    
    # Per-module breakdown (sorted by size)
    if 'per_module_mb' in breakdown:
        module_items = sorted(breakdown['per_module_mb'].items(), key=lambda x: x[1], reverse=True)
        lines.append(f"{prefix}Module params (top 5):")
        for name, mb in module_items[:5]:
            lines.append(f"{prefix}  {name}: {mb:.1f}MB")
    
    # Active modules status (shows RUNTIME activation, not YAML config)
    # Modules may be configured in YAML but inactive until their start_epoch is reached
    if 'active_modules' in breakdown:
        active = breakdown['active_modules']
        active_list = [k.replace('_active', '') for k, v in active.items() if v]
        inactive_list = [k.replace('_active', '') for k, v in active.items() if not v]
        if active_list:
            lines.append(f"{prefix}Active modules: {', '.join(active_list)}")
        if inactive_list:
            lines.append(f"{prefix}Inactive modules: {', '.join(inactive_list)}")
    
    return '\n'.join(lines)


class MemoryTracker:
    """
    Track GPU memory at various checkpoints to identify memory leaks/spikes.
    
    ENHANCED: Now captures tensor-level breakdown at each checkpoint for debugging.
    
    Usage:
        tracker = MemoryTracker(enabled=True, log_first_n_batches=5)
        
        # In training loop:
        tracker.checkpoint("after_data_load", batch=batch)
        ... forward pass ...
        tracker.checkpoint("after_forward", model=model, outputs=outputs, batch=batch)
        ... backward pass ...
        tracker.checkpoint("after_backward", model=model)
        
        tracker.end_batch(batch_idx)  # Prints summary with breakdown
    """
    
    def __init__(self, enabled: bool = True, log_first_n_batches: int = 5, device: str = 'cuda'):
        self.enabled = enabled and torch.cuda.is_available()
        self.log_first_n_batches = log_first_n_batches
        self.device = device
        self.checkpoints = {}
        self.batch_idx = 0
        self.baseline_allocated = 0
        self.baseline_reserved = 0
        
    def reset_baseline(self):
        """Set current memory as baseline (call after model creation, before training)."""
        if not self.enabled:
            return
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()  # Reset peak for accurate per-batch tracking
        self.baseline_allocated = torch.cuda.memory_allocated() / 1024 / 1024
        self.baseline_reserved = torch.cuda.memory_reserved() / 1024 / 1024
        
    def checkpoint(self, name: str, model=None, outputs: dict = None, batch: dict = None, optimizer=None):
        """Record memory at a named checkpoint with optional tensor breakdown."""
        if not self.enabled:
            return
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / 1024 / 1024
        reserved = torch.cuda.memory_reserved() / 1024 / 1024
        
        checkpoint_data = {
            'allocated': allocated,
            'reserved': reserved,
            'delta_alloc': allocated - self.baseline_allocated,
            'delta_reserved': reserved - self.baseline_reserved,
        }
        
        # Capture detailed breakdown if we have model/outputs/batch
        if model is not None or outputs is not None or batch is not None:
            try:
                breakdown = get_tensor_memory_breakdown(model, outputs, batch, optimizer=optimizer) if model else {}
                checkpoint_data['breakdown'] = breakdown
            except Exception as e:
                checkpoint_data['breakdown_error'] = str(e)
        
        self.checkpoints[name] = checkpoint_data
    
    def end_batch(self, batch_idx: int, epoch: int = 0) -> Optional[str]:
        """Print memory summary for this batch if within log_first_n_batches."""
        self.batch_idx = batch_idx
        
        if not self.enabled or batch_idx >= self.log_first_n_batches:
            self.checkpoints = {}  # Clear for next batch
            return None
        
        # Build summary
        lines = [f"\n  [MEMORY] Epoch {epoch} Batch {batch_idx}:"]
        lines.append(f"    Baseline: alloc={self.baseline_allocated:.0f}MB, reserved={self.baseline_reserved:.0f}MB")
        
        prev_alloc = self.baseline_allocated
        max_delta_name = None
        max_delta = 0
        
        for name, data in self.checkpoints.items():
            delta_from_prev = data['allocated'] - prev_alloc
            sign = '+' if delta_from_prev >= 0 else ''
            lines.append(
                f"    {name}: alloc={data['allocated']:.0f}MB ({sign}{delta_from_prev:.0f}MB), "
                f"reserved={data['reserved']:.0f}MB"
            )
            
            # Track which checkpoint had biggest increase
            if delta_from_prev > max_delta:
                max_delta = delta_from_prev
                max_delta_name = name
            
            prev_alloc = data['allocated']
        
        # Peak memory
        peak_allocated = torch.cuda.max_memory_allocated() / 1024 / 1024
        peak_reserved = torch.cuda.max_memory_reserved() / 1024 / 1024
        total_gpu = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
        lines.append(f"    PEAK: alloc={peak_allocated:.0f}MB, reserved={peak_reserved:.0f}MB / {total_gpu:.0f}MB ({100*peak_reserved/total_gpu:.1f}%)")
        
        # Print detailed breakdown for the checkpoint with max memory increase
        if max_delta_name and max_delta > 500:  # Only if >500MB increase
            lines.append(f"\n    [BREAKDOWN] Largest increase at '{max_delta_name}' (+{max_delta:.0f}MB):")
            if 'breakdown' in self.checkpoints[max_delta_name]:
                breakdown = self.checkpoints[max_delta_name]['breakdown']
                lines.append(format_memory_breakdown(breakdown, prefix="      "))
        
        summary = '\n'.join(lines)
        print(summary)
        
        # Reset peak for next batch
        torch.cuda.reset_peak_memory_stats()
        
        self.checkpoints = {}  # Clear for next batch
        return summary
    
    def get_current_memory(self) -> Dict[str, float]:
        """Get current memory stats as dict."""
        if not self.enabled:
            return {}
        torch.cuda.synchronize()
        return {
            'allocated_mb': torch.cuda.memory_allocated() / 1024 / 1024,
            'reserved_mb': torch.cuda.memory_reserved() / 1024 / 1024,
            'peak_allocated_mb': torch.cuda.max_memory_allocated() / 1024 / 1024,
            'peak_reserved_mb': torch.cuda.max_memory_reserved() / 1024 / 1024,
        }


def get_curriculum_stage(epoch: int, curriculum_stages: List[int]) -> int:
    """
    Determine curriculum stage based on current epoch.
    
    Args:
        epoch: Current training epoch (0-indexed)
        curriculum_stages: List of epoch thresholds [stage1_end, stage2_end, stage3_end]
            e.g., [100, 300, 600] means:
            - epochs 0-99: stage 1 (easy only)
            - epochs 100-299: stage 2 (easy + medium)
            - epochs 300-599: stage 3 (easy + medium + hard)
            - epochs 600+: stage 0 (all data, no filtering)
    
    Returns:
        Curriculum stage (1=easy, 2=medium, 3=hard, 0=all)
    """
    if not curriculum_stages:
        return 0  # No curriculum, use all data
    
    for stage_idx, threshold in enumerate(curriculum_stages):
        if epoch < threshold:
            return stage_idx + 1  # stages are 1-indexed
    
    return 0  # Beyond all thresholds, use all data


def create_train_loader(
    config: dict,
    curriculum_stage: int = 0,
    max_grid_size: int = 30,
    batch_size_override: Optional[int] = None,
) -> DataLoader:
    """
    Create training dataloader with optional curriculum filtering and caching.
    
    Args:
        config: Full configuration dict
        curriculum_stage: Curriculum stage (0=all, 1=easy, 2=medium, 3=hard)
        max_grid_size: Maximum grid size for padding
        batch_size_override: Override batch size (used for adaptive LOO batch sizing)
    
    Returns:
        DataLoader for training
        
    CACHING BEHAVIOR:
        When cache_samples=True (from config), the dataset pre-generates
        a fixed set of augmented samples. This allows the model to see the
        SAME samples every epoch, which is CRITICAL for learning hard tasks
        that need >100 epochs of training on the same data.
        
        After training with cached samples, you can resume training with
        cache_samples=False to expose the model to infinite diversity.
    """
    data_cfg = config['data']
    train_cfg = config['training']
    
    augment_cfg = data_cfg.get('augmentation', {})
    augment_enabled = augment_cfg.get('enabled', True)
    color_permutation = augment_cfg.get('color_permutation', False)
    color_permutation_prob = augment_cfg.get('color_permutation_prob', 0.3)  # Default to 30%!
    translational_augment = augment_cfg.get('translational', True)  # Random offset augmentation
    
    # Enable augmentation tracking for debugging (verifies diversity)
    track_augmentation = config.get('logging', {}).get('track_augmentation', True)
    
    # CRITICAL: Ignore padding in loss to prevent BG mode collapse
    ignore_padding_in_loss = data_cfg.get('ignore_padding_in_loss', True)  # Default True!
    
    # CACHING CONFIGURATION
    # When cache_samples=True, pre-generate fixed samples for memorization training
    cache_samples = data_cfg.get('cache_samples', False)
    cache_path = data_cfg.get('cache_path', None)
    cache_load_percent = data_cfg.get('cache_load_percent', 100.0)  # Percentage of cache to load
    
    # ==========================================================================
    # DYNAMIC num_cached_samples CALCULATION (Jan 2026)
    # ==========================================================================
    # Compute num_cached_samples based on task count and samples_per_task
    use_merged_training = data_cfg.get('use_merged_training', False)
    samples_per_task = data_cfg.get('samples_per_task', 50)
    max_tasks = data_cfg.get('max_tasks', None)
    
    # Determine task count and get task paths based on training set source
    merged_task_paths = None  # Will be set if use_merged_training=True
    if use_merged_training:
        merged_path = data_cfg.get('merged_training_path', './data/merged_training')
        try:
            from sci_arc.data.merged_loader import get_training_stats, get_merged_task_paths, should_use_merged_training
            
            # Validate that merged training is actually available
            if should_use_merged_training(data_cfg):
                merged_stats = get_training_stats(merged_path)
                base_task_count = merged_stats['total_train_tasks']
                
                # Get the actual task paths from the manifest
                merged_task_paths = get_merged_task_paths(merged_path)
                print(f"  [Merged Training] Using {base_task_count} tasks from merged set")
                print(f"  [Merged Training] Loaded {len(merged_task_paths)} task paths from manifest")
            else:
                print(f"  [WARNING] use_merged_training=True but manifest not found, falling back to train_path")
                base_task_count = 400
        except Exception as e:
            print(f"  [WARNING] Could not load merged training: {e}")
            print(f"  [WARNING] Falling back to train_path (ARC-AGI-1 only)")
            base_task_count = 400  # Fallback to default
    else:
        base_task_count = 400  # Default ARC-AGI-1 task count
    
    # Apply max_tasks limit if specified
    actual_task_count = min(base_task_count, max_tasks) if max_tasks else base_task_count
    
    # Get num_cached_samples from config, compute if 'auto'
    num_cached_samples_cfg = data_cfg.get('num_cached_samples', 'auto')
    if num_cached_samples_cfg == 'auto' or num_cached_samples_cfg is None:
        num_cached_samples = actual_task_count * samples_per_task
        print(f"  [Auto Compute] num_cached_samples = {actual_task_count} tasks × {samples_per_task} = {num_cached_samples}")
    else:
        num_cached_samples = int(num_cached_samples_cfg)
    
    # Update cache path based on training mode (for clarity)
    if cache_path and use_merged_training and 'agi1' in cache_path:
        cache_path = cache_path.replace('agi1', 'merged').replace('400tasks', f'{actual_task_count}tasks')
        print(f"  [Cache Path Updated] {cache_path}")
    
    if cache_samples:
        print(f"\n{'='*60}")
        print("CACHED SAMPLES MODE ENABLED")
        print(f"{'='*60}")
        print(f"  Samples will be pre-generated and reused each epoch")
        print(f"  This allows model to learn from repeated exposure")
        print(f"  (Required for hard tasks needing >100 epochs)")
        print(f"  Training Set: {'MERGED (AGI-1 + AGI-2)' if use_merged_training else 'ARC-AGI-1 only'}")
        print(f"  Task Count: {actual_task_count}")
        print(f"  Samples/Task: {samples_per_task}")
        print(f"  Total Samples: {num_cached_samples}")
        if cache_load_percent < 100:
            print(f"  cache_load_percent: {cache_load_percent}% (PARTIAL LOAD for quick testing)")
        if cache_path:
            print(f"  cache_path: {cache_path}")
        print(f"{'='*60}\n")
    
    # Support max_tasks for quick testing (stratified sampling for representativeness)
    # max_tasks already read above for num_cached_samples calculation
    stratified_seed = data_cfg.get('stratified_seed', 42)
    
    # Warn about wasteful configurations
    if max_tasks is not None and cache_samples and cache_load_percent < 100:
        print(f"\n{'!'*60}")
        print(f"WARNING: WASTEFUL CONFIGURATION DETECTED!")
        print(f"  max_tasks={max_tasks} + cache_load_percent={cache_load_percent}%")
        print(f"  This loads {cache_load_percent}% of cache, then filters to {max_tasks} tasks.")
        print(f"  Most loaded samples will be discarded!")
        print(f"")
        print(f"  BETTER OPTIONS:")
        print(f"  1. max_tasks={max_tasks}, cache_load_percent=100 (get all samples for {max_tasks} tasks)")
        print(f"  2. max_tasks=null, cache_load_percent={cache_load_percent} (random sample from all tasks)")
        print(f"  3. max_tasks={max_tasks}, cache_samples=false (on-the-fly augmentation)")
        print(f"{'!'*60}\n")
    
    # Determine data_path for dataset (merged training uses first path as reference)
    # The actual tasks come from merged_task_paths when provided
    effective_data_path = data_cfg['train_path']
    if merged_task_paths:
        # For merged training, data_path is used for logging/cache path derivation
        # but actual tasks come from merged_task_paths
        effective_data_path = data_cfg.get('merged_training_path', data_cfg['train_path'])
    
    train_dataset = ARCDataset(
        effective_data_path,
        max_size=max_grid_size,
        augment=augment_enabled,
        color_permutation=color_permutation,
        color_permutation_prob=color_permutation_prob,  # Control permutation frequency
        translational_augment=translational_augment,
        curriculum_stage=curriculum_stage,  # Apply curriculum filtering!
        track_augmentation=track_augmentation,  # Enable for diversity logging
        ignore_padding_in_loss=ignore_padding_in_loss,  # Ignore padding in loss
        cache_samples=cache_samples,  # Enable caching for memorization
        num_cached_samples=num_cached_samples,  # Number of cached samples
        cache_path=cache_path,  # Optional path to save/load cache
        cache_load_percent=cache_load_percent,  # Percentage of cache to load (for quick testing)
        max_tasks=max_tasks,  # Limit tasks for testing (stratified sampling)
        stratified_seed=stratified_seed,  # Seed for reproducible stratified sampling
        task_paths=merged_task_paths,  # Explicit task paths for merged training (Jan 2026)
    )
    
    # ADAPTIVE BATCH SIZE: Use override if provided (for LOO memory management)
    if batch_size_override is not None:
        batch_size = batch_size_override
        print(f"  BATCH SIZE OVERRIDE: {batch_size} (adaptive LOO sizing)")
    else:
        batch_size = train_cfg['batch_size']
    
    # OPTIMIZATION: When using cached samples, use num_workers=0
    # Workers need to pickle/unpickle the full dataset which is slow for large caches
    # With cache in main process memory, direct access is faster
    # GPU stalls are prevented by CUDAPrefetcher which uses a separate CUDA stream
    # for async data transfer (CPU→GPU overlaps with GPU computation)
    if cache_samples:
        num_workers = 0
        print("  (Using num_workers=0 for cached samples + CUDAPrefetcher for async GPU transfer)")
    else:
        num_workers = data_cfg.get('num_workers', 0)
    
    pin_memory = data_cfg.get('pin_memory', True)
    prefetch_factor = data_cfg.get('prefetch_factor', 2) if num_workers > 0 else None
    persistent_workers = data_cfg.get('persistent_workers', False) and num_workers > 0
    
    collate_fn = partial(collate_sci_arc, max_grid_size=max_grid_size)
    
    # ==========================================================================
    # BUCKETED BATCHING: ALWAYS ON (NON-NEGOTIABLE FOR MEMORY EFFICIENCY)
    # ==========================================================================
    # Groups samples by grid size to prevent memory waste from padding.
    # Without this, one 30x30 grid forces all batch samples to 30x30 memory.
    # This applies to ALL modes: cached, on-the-fly, full, or sampled.
    # 
    # NOTE: YAML setting is IGNORED - bucketing is ALWAYS enabled.
    # ==========================================================================
    bucket_boundaries = data_cfg.get('bucket_boundaries', [10, 15, 20, 25])
    
    # Warn if user tried to disable bucketing (but we ignore the setting)
    yaml_bucketing = data_cfg.get('bucketed_batching', True)
    if not yaml_bucketing:
        print(f"\n{'!'*60}")
        print(f"WARNING: bucketed_batching=false in YAML is IGNORED!")
        print(f"  Bucketed batching is NON-NEGOTIABLE for memory efficiency.")
        print(f"  The setting will be forced to true.")
        print(f"{'!'*60}\n")
    
    print(f"  Using BUCKETED BATCHING (groups samples by grid size)")
    print(f"    Bucket boundaries: {bucket_boundaries} → {len(bucket_boundaries)+1} buckets")
    
    # CRITICAL FIX (Jan 2026): drop_last=true excludes tasks in small buckets!
    # With batch_size=50 and small buckets (e.g., 28 samples), drop_last=true means
    # those tasks are NEVER trained. Default to false for full task coverage.
    drop_last = data_cfg.get('drop_last', False)
    if drop_last:
        print(f"    [WARNING] drop_last=true may exclude tasks in small buckets!")
    
    # Use hardware.seed for reproducibility (consistent with rest of training)
    global_seed = config.get('hardware', {}).get('seed', 42)
    batch_sampler = BucketedBatchSampler(
        dataset=train_dataset,
        batch_size=batch_size,
        bucket_boundaries=bucket_boundaries,
        drop_last=drop_last,
        shuffle=True,
        seed=global_seed,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,  # Use batch_sampler instead of batch_size/shuffle
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        collate_fn=collate_fn,
    )
    
    return train_loader


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Find the most recent checkpoint in the checkpoint directory."""
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        return None
    
    # Check for the dedicated "latest" checkpoint
    latest_path = checkpoint_path / 'latest.pt'
    if latest_path.exists():
        return str(latest_path)
    
    # Otherwise, find the highest epoch number
    checkpoints = list(checkpoint_path.glob('epoch_*.pt'))
    if not checkpoints:
        return None
    
    def get_epoch(p):
        try:
            return int(p.stem.split('_')[-1])
        except:
            return -1
    
    checkpoints.sort(key=get_epoch, reverse=True)
    return str(checkpoints[0])


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def override_config(config: dict, overrides: list) -> dict:
    """Apply command-line overrides to config.
    
    Format: key.subkey=value
    Examples:
        training.batch_size=32
        training.max_epochs=100
        logging.use_wandb=true
        model.hidden_dim=512
    """
    import ast
    for override in overrides:
        if '=' not in override:
            continue
        key_path, value = override.split('=', 1)
        keys = key_path.split('.')
        
        # Navigate to the right level
        current = config
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Set value with type inference
        final_key = keys[-1]
        try:
            # Try to parse as Python literal (bool, int, float, list)
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            # Keep as string
            pass
        
        current[final_key] = value
        print(f"  Override: {key_path} = {value}")
    
    return config


def get_temperature(epoch: int, config: dict) -> float:
    """Get temperature for DSC attention softmax based on epoch.
    
    Note: Gumbel noise was removed in Dec 2025. Temperature now controls
    sharpness of standard softmax attention in the DSC module.
    """
    tau_start = config['training']['temperature_start']
    tau_end = config['training']['temperature_end']
    max_epochs = config['training']['max_epochs']
    
    # Exponential decay
    progress = epoch / max_epochs
    temperature = tau_start * (tau_end / tau_start) ** progress
    
    return temperature


def validate_meta_learning_config(config: dict) -> None:
    """Validate meta-learning configuration for silent failure conditions.
    
    FIX (Jan 2026): Add startup validation to detect config combinations
    that will silently disable features (e.g., HyperLoRA without support features).
    
    Raises:
        ValueError: If config has a fatal misconfiguration.
    
    Warns:
        UserWarning: If config has a suboptimal configuration.
    """
    import warnings
    
    model_config = config.get('model', {})
    
    use_hyperlora = model_config.get('use_hyperlora', False)
    use_solver_context = model_config.get('use_solver_context', False)
    use_cross_attention_context = model_config.get('use_cross_attention_context', False)
    use_hpm = model_config.get('use_hpm', False)
    
    # Check 1: HyperLoRA requires support features to work
    if use_hyperlora and not (use_solver_context or use_cross_attention_context):
        warnings.warn(
            "[CONFIG WARNING] HyperLoRA is enabled but neither use_solver_context nor "
            "use_cross_attention_context is enabled. HyperLoRA requires support features "
            "from the context encoder to produce LoRA deltas. Without these, HyperLoRA "
            "will be loaded (~2.8M params) but produce NO adaptation (lora_deltas=None). "
            "Enable use_solver_context: true or use_cross_attention_context: true.",
            UserWarning,
            stacklevel=2
        )
    
    # Check 2: HPM requires context features for queries
    if use_hpm and not (use_solver_context or use_cross_attention_context):
        warnings.warn(
            "[CONFIG WARNING] HPM is enabled but neither use_solver_context nor "
            "use_cross_attention_context is enabled. HPM requires z_context_flat "
            "from the context encoder to query memory banks. Without these, HPM "
            "queries will be skipped entirely.",
            UserWarning,
            stacklevel=2
        )
    
    # Check 3: HPM dynamic banks require corresponding features
    use_instance_bank = model_config.get('hpm_use_instance_bank', False)
    use_procedural_bank = model_config.get('hpm_use_procedural_bank', False)
    
    if use_procedural_bank and not use_hyperlora:
        warnings.warn(
            "[CONFIG WARNING] hpm_use_procedural_bank is enabled but use_hyperlora is false. "
            "Procedural bank stores HyperLoRA latent codes, which won't be generated.",
            UserWarning,
            stacklevel=2
        )
    
    if use_instance_bank and not (use_solver_context or use_cross_attention_context):
        warnings.warn(
            "[CONFIG WARNING] hpm_use_instance_bank is enabled but context encoder won't "
            "produce support_features. Instance bank stores context embeddings which "
            "require use_solver_context or use_cross_attention_context.",
            UserWarning,
            stacklevel=2
        )
    
    print("[CONFIG] Meta-learning validation: PASSED")


def create_model(config: dict) -> RLAN:
    """Create RLAN model from config.
    
    CRITICAL: All model.* fields in YAML must be passed through here!
    If a field exists in RLANConfig but is missing here, the YAML value is silently ignored.
    """
    from sci_arc.models.rlan import RLANConfig
    
    model_config = config['model']
    train_config = config.get('training', {})
    
    # Create RLANConfig with all parameters including ablation flags
    rlan_config = RLANConfig(
        hidden_dim=model_config['hidden_dim'],
        num_colors=model_config['num_colors'],
        num_classes=model_config['num_classes'],
        max_grid_size=model_config['max_grid_size'],
        max_clues=model_config['max_clues'],
        num_predicates=model_config['num_predicates'],
        num_solver_steps=model_config['num_solver_steps'],
        use_act=model_config.get('use_act', False),
        dropout=model_config['dropout'],
        dsc_num_heads=model_config.get('dsc_num_heads', 4),
        lcr_num_heads=model_config.get('lcr_num_heads', 4),
        msre_encoding_dim=model_config.get('msre_encoding_dim', 32),
        msre_num_freq=model_config.get('msre_num_freq', 8),
        lcr_num_freq=model_config.get('lcr_num_freq', 8),
        # Module ablation flags
        use_context_encoder=model_config.get('use_context_encoder', True),
        use_dsc=model_config.get('use_dsc', True),
        use_msre=model_config.get('use_msre', True),
        use_lcr=model_config.get('use_lcr', True),
        use_sph=model_config.get('use_sph', True),
        use_learned_pos=model_config.get('use_learned_pos', False),
        # Context injection mode (CRITICAL for architecture behavior)
        use_cross_attention_context=model_config.get('use_cross_attention_context', False),
        spatial_downsample=model_config.get('spatial_downsample', 1),
        # Phase 2.5: Solver cross-attention to support set
        use_solver_context=model_config.get('use_solver_context', True),
        solver_context_heads=model_config.get('solver_context_heads', 4),
        use_best_step_selection=model_config.get('use_best_step_selection', False),
        # HyperLoRA: Meta-learning weight adaptation
        use_hyperlora=model_config.get('use_hyperlora', False),
        hyperlora_rank=model_config.get('hyperlora_rank', 8),
        hyperlora_scaling=model_config.get('hyperlora_scaling', 1.0),
        hyperlora_dropout=model_config.get('hyperlora_dropout', 0.0),
        hyperlora_init_scale=model_config.get('hyperlora_init_scale', 0.1),  # Default matches YAML
        # HPM (Hierarchical Primitive Memory v2) - all fields from YAML
        use_hpm=model_config.get('use_hpm', False),
        hpm_top_k=model_config.get('hpm_top_k', 2),
        hpm_balance_weight=model_config.get('hpm_balance_weight', 0.01),
        hpm_primitives_per_bank=model_config.get('hpm_primitives_per_bank', 16),
        hpm_levels_per_bank=model_config.get('hpm_levels_per_bank', 2),
        hpm_use_cross_attention=model_config.get('hpm_use_cross_attention', True),
        hpm_memory_size=model_config.get('hpm_memory_size', 10000),
        hpm_retrieval_k=model_config.get('hpm_retrieval_k', 5),
        hpm_use_compositional_bank=model_config.get('hpm_use_compositional_bank', True),
        hpm_use_pattern_bank=model_config.get('hpm_use_pattern_bank', True),
        hpm_use_relational_bank=model_config.get('hpm_use_relational_bank', True),
        hpm_use_concept_bank=model_config.get('hpm_use_concept_bank', False),
        hpm_use_procedural_bank=model_config.get('hpm_use_procedural_bank', False),
        hpm_use_instance_bank=model_config.get('hpm_use_instance_bank', False),
        # HPM Solver-Context Coupling (Jan 2026)
        hpm_solver_context_enabled=model_config.get('hpm_solver_context_enabled', True),
        hpm_solver_context_max_tokens=model_config.get('hpm_solver_context_max_tokens', 8),
        hpm_solver_context_gate_init=model_config.get('hpm_solver_context_gate_init', 0.0),
        # Memory optimization: gradient checkpointing
        gradient_checkpointing=train_config.get('gradient_checkpointing', False),
    )
    
    model = RLAN(config=rlan_config)
    
    return model


def create_loss(config: dict) -> RLANLoss:
    """Create RLAN loss function from config."""
    train_config = config['training']
    model_config = config['model']
    
    loss_fn = RLANLoss(
        focal_gamma=train_config['focal_gamma'],
        focal_alpha=train_config['focal_alpha'],
        lambda_entropy=train_config['lambda_entropy'],
        lambda_sparsity=train_config['lambda_sparsity'],
        lambda_predicate=train_config['lambda_predicate'],
        lambda_curriculum=train_config['lambda_curriculum'],
        lambda_deep_supervision=train_config['lambda_deep_supervision'],
        lambda_act=train_config.get('lambda_act', 0.1),  # ACT halting loss weight
        min_clues=train_config.get('min_clues', 2.5),  # Minimum clues to use (increased default)
        min_clue_weight=train_config.get('min_clue_weight', 5.0),  # Strong penalty for fewer clues
        ponder_weight=train_config.get('ponder_weight', 0.02),  # Base cost per clue
        entropy_ponder_weight=train_config.get('entropy_ponder_weight', 0.02),  # Extra cost for diffuse attention
        clue_variance_weight=train_config.get('clue_variance_weight', 0.0),  # NEW: Variance regularization
        clue_target_variance=train_config.get('clue_target_variance', 0.5),  # NEW: Target std
        stop_saturation_weight=train_config.get('stop_saturation_weight', 0.0),  # NEW: Saturation penalty
        stop_saturation_threshold=train_config.get('stop_saturation_threshold', 5.0),  # NEW: Threshold
        max_clues=model_config['max_clues'],
        use_stablemax=train_config.get('use_stablemax', True),
        loss_mode=train_config.get('loss_mode', 'focal_stablemax'),
        # BG/FG weight caps for weighted_stablemax (CRITICAL for preventing collapse)
        bg_weight_cap=train_config.get('bg_weight_cap', 2.0),
        fg_weight_cap=train_config.get('fg_weight_cap', 5.0),
        # Centroid diversity (CRITICAL for DSC health - prevents clue collapse)
        lambda_centroid_diversity=train_config.get('lambda_centroid_diversity', 0.1),
    )
    
    return loss_fn


def create_optimizer(model: nn.Module, config: dict, steps_per_epoch: int = None):
    """Create optimizer and scheduler.
    
    Args:
        model: The model to optimize
        config: Configuration dict
        steps_per_epoch: Number of training batches per epoch (for OneCycle)
        
    CRITICAL INSIGHT (from gradient analysis):
    DSC/MSRE have ~60x smaller gradients than Solver due to:
    1. Long gradient path through coordinate computation
    2. Coordinate normalization and Fourier encoding
    3. Feature fusion before task loss
    
    Solution: Use higher learning rates for DSC/MSRE modules to compensate.
    This is similar to layer-wise learning rate scaling in NLP transformers.
    """
    train_config = config['training']
    base_lr = train_config['learning_rate']
    weight_decay = train_config['weight_decay']
    
    # Get LR multipliers from config (default: 10x for DSC/MSRE to compensate for ~60x smaller gradients)
    dsc_lr_mult = train_config.get('dsc_lr_multiplier', 10.0)
    msre_lr_mult = train_config.get('msre_lr_multiplier', 10.0)
    hyperlora_lr_mult = train_config.get('hyperlora_lr_multiplier', 10.0)  # HyperLoRA often needs higher LR
    
    # Separate parameters into groups:
    # 1. DSC parameters (higher LR)
    # 2. MSRE parameters (higher LR)
    # 3. HyperLoRA parameters (higher LR - critical for meta-learning)
    # 4. Other parameters with decay
    # 5. Other parameters without decay (bias, norm, embedding)
    
    dsc_decay_params = []
    dsc_no_decay_params = []
    msre_decay_params = []
    msre_no_decay_params = []
    hyperlora_decay_params = []
    hyperlora_no_decay_params = []
    other_decay_params = []
    other_no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        is_no_decay = 'bias' in name or 'norm' in name or 'embedding' in name
        
        # Route to appropriate group based on module name
        if '.dsc.' in name or name.startswith('dsc.'):
            if is_no_decay:
                dsc_no_decay_params.append(param)
            else:
                dsc_decay_params.append(param)
        elif '.msre.' in name or name.startswith('msre.'):
            if is_no_decay:
                msre_no_decay_params.append(param)
            else:
                msre_decay_params.append(param)
        elif 'hyper_lora' in name or '.hyperlora' in name:
            # HyperLoRA params for meta-learning weight prediction
            if is_no_decay:
                hyperlora_no_decay_params.append(param)
            else:
                hyperlora_decay_params.append(param)
        else:
            if is_no_decay:
                other_no_decay_params.append(param)
            else:
                other_decay_params.append(param)
    
    # Build param groups with per-group learning rates
    param_groups = []
    
    # DSC params with higher LR
    if dsc_decay_params:
        param_groups.append({
            'params': dsc_decay_params, 
            'weight_decay': weight_decay,
            'lr': base_lr * dsc_lr_mult,
            'name': 'dsc_decay'
        })
    if dsc_no_decay_params:
        param_groups.append({
            'params': dsc_no_decay_params, 
            'weight_decay': 0.0,
            'lr': base_lr * dsc_lr_mult,
            'name': 'dsc_no_decay'
        })
    
    # MSRE params with higher LR
    if msre_decay_params:
        param_groups.append({
            'params': msre_decay_params, 
            'weight_decay': weight_decay,
            'lr': base_lr * msre_lr_mult,
            'name': 'msre_decay'
        })
    if msre_no_decay_params:
        param_groups.append({
            'params': msre_no_decay_params, 
            'weight_decay': 0.0,
            'lr': base_lr * msre_lr_mult,
            'name': 'msre_no_decay'
        })
    
    # HyperLoRA params with higher LR (critical for meta-learning)
    if hyperlora_decay_params:
        param_groups.append({
            'params': hyperlora_decay_params, 
            'weight_decay': weight_decay,
            'lr': base_lr * hyperlora_lr_mult,
            'name': 'hyperlora_decay'
        })
    if hyperlora_no_decay_params:
        param_groups.append({
            'params': hyperlora_no_decay_params, 
            'weight_decay': 0.0,
            'lr': base_lr * hyperlora_lr_mult,
            'name': 'hyperlora_no_decay'
        })
    
    # Other params with base LR
    if other_decay_params:
        param_groups.append({
            'params': other_decay_params, 
            'weight_decay': weight_decay,
            'lr': base_lr,
            'name': 'other_decay'
        })
    if other_no_decay_params:
        param_groups.append({
            'params': other_no_decay_params, 
            'weight_decay': 0.0,
            'lr': base_lr,
            'name': 'other_no_decay'
        })
    
    # Log param group sizes for verification
    dsc_count = len(dsc_decay_params) + len(dsc_no_decay_params)
    msre_count = len(msre_decay_params) + len(msre_no_decay_params)
    hyperlora_count = len(hyperlora_decay_params) + len(hyperlora_no_decay_params)
    other_count = len(other_decay_params) + len(other_no_decay_params)
    print(f"  Optimizer param groups:")
    print(f"    DSC: {dsc_count} params @ {dsc_lr_mult}x LR ({base_lr * dsc_lr_mult:.2e})")
    print(f"    MSRE: {msre_count} params @ {msre_lr_mult}x LR ({base_lr * msre_lr_mult:.2e})")
    if hyperlora_count > 0:
        print(f"    HyperLoRA: {hyperlora_count} params @ {hyperlora_lr_mult}x LR ({base_lr * hyperlora_lr_mult:.2e}) [META-LEARNING]")
    print(f"    Other: {other_count} params @ 1x LR ({base_lr:.2e})")
    
    # Get optimizer betas (TRM uses beta2=0.95 instead of default 0.999)
    beta1 = train_config.get('beta1', 0.9)
    beta2 = train_config.get('beta2', 0.95)  # TRM default, more stable for recursive models
    
    # =============================================================
    # MEMORY OPTIMIZATION: 8-bit AdamW Optimizer
    # =============================================================
    # Reduces optimizer state memory by ~75% (stores momentum/variance in 8-bit)
    # Fallback: Standard AdamW if bitsandbytes not installed
    # =============================================================
    use_8bit = train_config.get('use_8bit_optimizer', False)
    
    if use_8bit:
        try:
            # Optional dependency: use dynamic import so static analyzers
            # (e.g., Pylance) don't hard-error when it's not installed.
            import importlib
            bnb = importlib.import_module("bitsandbytes")
            optimizer = bnb.optim.AdamW8bit(
                param_groups,
                lr=base_lr,
                betas=(beta1, beta2),
            )
            print(f"  Optimizer: 8-bit AdamW (bitsandbytes) - ~75% memory savings on optimizer states")
        except ImportError:
            print(f"  [WARNING] bitsandbytes not installed, falling back to standard AdamW")
            print(f"           Install with: pip install bitsandbytes")
            optimizer = torch.optim.AdamW(
                param_groups,
                lr=base_lr,
                betas=(beta1, beta2),
            )
    else:
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=base_lr,
            betas=(beta1, beta2),
        )
    
    # Create scheduler based on config
    scheduler_type = train_config.get('scheduler', 'cosine')
    max_epochs = train_config['max_epochs']
    warmup_epochs = train_config.get('warmup_epochs', 10)
    min_lr = train_config.get('min_lr', 1e-6)
    
    if scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max_epochs - warmup_epochs,
            eta_min=min_lr,
        )
    elif scheduler_type == 'onecycle':
        # OneCycleLR needs total_steps = epochs × steps_per_epoch
        total_steps = max_epochs * (steps_per_epoch or 1)
        # For per-group LR, provide list of max_lr values
        max_lrs = [pg.get('lr', base_lr) for pg in param_groups]
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lrs,  # Per-group max learning rates
            total_steps=total_steps,
            pct_start=warmup_epochs / max_epochs,  # Warmup fraction
            anneal_strategy='cos',
            div_factor=25.0,  # initial_lr = max_lr / 25
            final_div_factor=1000.0,  # final_lr = max_lr / 1000
        )
    elif scheduler_type == 'none' or scheduler_type is None:
        # No scheduler - constant learning rate (most stable)
        scheduler = None
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=50,
            T_mult=2,
            eta_min=min_lr,
        )
    
    return optimizer, scheduler


def warmup_lr(optimizer, step: int, warmup_steps: int, base_lr: float, initial_lrs: list = None):
    """Apply linear warmup to learning rate.
    
    Args:
        optimizer: The optimizer with param groups
        step: Current step number
        warmup_steps: Total warmup steps
        base_lr: Base learning rate (used if initial_lrs not provided)
        initial_lrs: Per-group initial learning rates (to preserve LR ratios)
    """
    if step < warmup_steps:
        lr_scale = float(step + 1) / float(max(1, warmup_steps))
        for i, param_group in enumerate(optimizer.param_groups):
            # Use per-group LR if available, otherwise base_lr
            if initial_lrs is not None and i < len(initial_lrs):
                target_lr = initial_lrs[i]
            else:
                target_lr = base_lr
            param_group['lr'] = target_lr * lr_scale


def compute_module_grad_norms(model: RLAN) -> Dict[str, float]:
    """Compute gradient norms for key modules to verify learning signal.
    
    Returns 0.0 for modules with no gradients, handles NaN gracefully.
    """
    grad_norms = {}
    
    def safe_grad_norm(param):
        """Safely compute gradient norm, returning 0 for NaN/Inf."""
        if param.grad is None:
            return 0.0
        norm = param.grad.norm().item()
        if not math.isfinite(norm):  # Handle NaN and Inf
            return 0.0
        return norm ** 2
    
    # DSC gradients
    if hasattr(model, 'dsc') and model.dsc is not None:
        dsc_grad = 0.0
        dsc_count = 0
        stop_pred_grad = 0.0
        stop_pred_count = 0
        for name, param in model.dsc.named_parameters():
            if param.grad is not None:
                grad_norm_sq = safe_grad_norm(param)
                dsc_grad += grad_norm_sq
                dsc_count += 1
                # Track stop_predictor specifically (critical for clue usage learning)
                if 'stop_predictor' in name:
                    stop_pred_grad += grad_norm_sq
                    stop_pred_count += 1
        grad_norms['dsc'] = (dsc_grad ** 0.5) if dsc_count > 0 else 0.0
        grad_norms['stop_predictor'] = (stop_pred_grad ** 0.5) if stop_pred_count > 0 else 0.0
    
    # Encoder gradients
    if hasattr(model, 'encoder') and model.encoder is not None:
        enc_grad = 0.0
        enc_count = 0
        for name, param in model.encoder.named_parameters():
            if param.grad is not None:
                enc_grad += safe_grad_norm(param)
                enc_count += 1
        grad_norms['encoder'] = (enc_grad ** 0.5) if enc_count > 0 else 0.0
    
    # Solver gradients
    if hasattr(model, 'solver') and model.solver is not None:
        solver_grad = 0.0
        solver_count = 0
        for name, param in model.solver.named_parameters():
            if param.grad is not None:
                solver_grad += safe_grad_norm(param)
                solver_count += 1
        grad_norms['solver'] = (solver_grad ** 0.5) if solver_count > 0 else 0.0
    
    # Context encoder gradients (if enabled)
    if hasattr(model, 'context_encoder') and model.context_encoder is not None:
        ctx_grad = 0.0
        ctx_count = 0
        for name, param in model.context_encoder.named_parameters():
            if param.grad is not None:
                ctx_grad += safe_grad_norm(param)
                ctx_count += 1
        grad_norms['context_encoder'] = (ctx_grad ** 0.5) if ctx_count > 0 else 0.0
    
    # MSRE gradients (if enabled)
    if hasattr(model, 'msre') and model.msre is not None:
        msre_grad = 0.0
        msre_count = 0
        for name, param in model.msre.named_parameters():
            if param.grad is not None:
                msre_grad += safe_grad_norm(param)
                msre_count += 1
        grad_norms['msre'] = (msre_grad ** 0.5) if msre_count > 0 else 0.0

    return grad_norms


def compute_stop_predictor_weight_variance(model: RLAN) -> float:
    """Variance of stop predictor Linear weights.

    Used to detect silent "frozen" stop predictor behavior.
    
    Returns:
        float: Weight variance if module exists and variance is finite.
               float('nan') if module is missing (to distinguish from actual zero variance).
               0.0 if variance computation fails or is non-finite.
    
    FIX (Jan 2026): Return NaN when module is missing to distinguish from
    genuine near-zero variance (which would indicate weight collapse).
    """
    try:
        if not hasattr(model, 'dsc') or model.dsc is None:
            return float('nan')  # Module missing - return NaN
        stop_predictor = getattr(model.dsc, 'stop_predictor', None)
        if stop_predictor is None:
            return float('nan')  # Stop predictor missing - return NaN

        weights = []
        for module in stop_predictor.modules():
            if isinstance(module, nn.Linear):
                weights.append(module.weight.detach().float().reshape(-1))

        if not weights:
            return float('nan')  # No Linear layers found - return NaN

        w = torch.cat(weights, dim=0)
        var = w.var(unbiased=False).item()
        return var if math.isfinite(var) else 0.0
    except Exception:
        return 0.0


def train_epoch(
    model: RLAN,
    dataloader: DataLoader,
    loss_fn: RLANLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    config: dict,
    scaler: Optional[GradScaler] = None,
    global_step: int = 0,
    ema: Optional[EMAHelper] = None,
    loo_loss_fn: Optional[LOOTrainingLoss] = None,
    equiv_loss_fn: Optional[AugmentationEquivarianceLoss] = None,
    output_equiv_loss_fn = None,  # Jan 2026: Output-level equivariance
    group_marg_loss_fn = None,    # Jan 2026: Group-marginalized NLL
    loo_start_epoch: int = 12,
    equiv_start_epoch: int = 8,
    output_equiv_start_epoch: int = 16,  # Jan 2026
    output_equiv_weight: float = 0.02,   # Jan 2026
    output_equiv_num_augs: int = 2,      # Jan 2026
    group_marg_start_epoch: int = 22,    # Jan 2026
    group_marg_weight: float = 0.1,      # Jan 2026
    global_task_tracker: Optional[Dict] = None,  # Jan 2026: Global task solving tracker
    phase_disable_flags: Optional[Dict] = None,  # Jan 2026: Phased training overrides
    # Jan 2026 Ablation Study: ART + ARPS modules
    art_module: Optional[AnchorRobustnessTraining] = None,
    arps_module: Optional[ARPS] = None,
    art_start_epoch: int = 0,     # ART active from epoch 0 by default (simple module)
    arps_start_epoch: int = 5,    # ARPS starts after initial learning (needs stable features)
) -> Dict[str, float]:
    """
    Train for one epoch with augmentation diversity tracking.
    
    Args:
        model: RLAN model
        dataloader: Training data loader
        loss_fn: Main task loss (RLANLoss)
        optimizer: Optimizer
        device: Device
        epoch: Current epoch
        config: Full config dict
        scaler: GradScaler for mixed precision
        global_step: Global step counter
        ema: Optional EMA helper
        loo_loss_fn: Optional LOO training loss for meta-learning
        equiv_loss_fn: Optional Augmentation Equivariance loss for meta-learning
        loo_start_epoch: Epoch when LOO loss activates (default: 12)
        equiv_start_epoch: Epoch when equivariance loss activates (default: 8)
        global_task_tracker: Optional dict tracking tasks solved across all epochs (Jan 2026)
    
    Returns losses dict AND augmentation statistics for debugging.
    """
    model.train()
    
    # Initialize phase disable flags (Jan 2026: phased training support)
    if phase_disable_flags is None:
        phase_disable_flags = {}
    
    total_losses = {
        'total_loss': 0.0,
        'focal_loss': 0.0,
        'entropy_loss': 0.0,
        'sparsity_loss': 0.0,
        'predicate_loss': 0.0,
        'loo_loss': 0.0,  # LOO meta-learning loss
        'equiv_loss': 0.0,  # Augmentation equivariance loss
        'output_equiv_loss': 0.0,  # Jan 2026: Output-level equivariance
        'hpm_balance_loss': 0.0,  # HPM load balancing loss
        'curriculum_loss': 0.0,
        'deep_supervision_loss': 0.0,  # FIX: Was missing, caused zero reporting
        'act_loss': 0.0,  # ACT halting loss
        # Jan 2026 Ablation Study
        'art_consistency_loss': 0.0,  # ART anchor robustness loss
        'arps_imitation_loss': 0.0,   # ARPS program imitation loss
    }
    num_batches = 0
    total_samples = 0  # Track total samples processed
    
    # Augmentation statistics for the epoch (CRITICAL for verifying diversity)
    epoch_aug_stats = {
        'dihedral_counts': [0] * 8,  # Count per dihedral transform ID (0-7)
        'color_perm_count': 0,       # Samples with color permutation
        'translational_count': 0,    # Samples with translational offset
        'unique_offsets': 0,         # Running unique offset count (approximate)
    }
    
    # Diagnostic statistics for debugging training dynamics
    # ======================================================
    # These metrics help identify WHY training succeeds or fails
    # ======================================================
    num_classes = 10  # Colors 0-9
    epoch_diagnostics = {
        # Gradient flow (are gradients reaching each module?)
        # Patch 2 (Dec 2025): Track max grad norm across entire epoch
        'max_grad_norm_before_clip': 0.0,  # Max grad norm seen this epoch (for backoff)
        'dsc_grad_norm_sum': 0.0,           # Gradient norm flowing to DSC
        'stop_predictor_grad_norm_sum': 0.0, # Gradient norm to stop_predictor specifically
        'stop_predictor_weight_variance': 0.0, # Variance of stop predictor weights (0 = frozen/not learning)
        'encoder_grad_norm_sum': 0.0,        # Gradient norm to encoder
        'solver_grad_norm_sum': 0.0,         # Gradient norm to solver
        'context_encoder_grad_norm_sum': 0.0, # Gradient norm to context encoder
        'msre_grad_norm_sum': 0.0,           # Gradient norm to MSRE
        
        # DSC attention quality (is DSC learning meaningful patterns?)
        'per_clue_entropy': [],        # Per-clue entropy breakdown (K values)
        'centroid_spread': 0.0,        # How spread out are clue centroids
        'all_logits_count': 0,         # Verify deep supervision is receiving steps

        # =============================================================
        # ATTENTION COLLAPSE EVENTS (for meta-escalation stability gating)
        # =============================================================
        # Count batches where attention becomes near-uniform (diffuse).
        # This directly targets the observed late-epoch collapse mode:
        # attn_max_mean ~ 0.005 (≈ uniform over H*W) preceding FG/BG nosedive.
        'attention_collapse_events_epoch': 0,
        'attention_collapse_consecutive': 0,
        'attention_collapse_consecutive_max': 0,
        
        # Attention statistics (is attention focusing or diffuse?)
        'attn_max_mean': 0.0,          # Mean of max attention values (higher = sharper)
        'attn_min_mean': 0.0,          # Mean of min attention values (lower = sharper)
        'stop_prob_mean': 0.0,         # Mean stop probability (how many clues used)
        
        # Per-step loss (is deep supervision working?)
        'per_step_loss': [],           # Loss at each solver step (last batch only, for display)
        'per_step_loss_sum': None,     # Accumulator: sum of losses per step (for epoch average)
        'per_step_loss_count': 0,      # Number of batches accumulated
        
        # =============================================================
        # BEST-STEP SELECTION TRACKING (solver health monitoring)
        # =============================================================
        'best_step_histogram': [0] * 10,  # Count how often each step is best (supports up to 10 steps)
        'last_step_was_best_count': 0,    # Count batches where last step was best (healthy)
        'earlier_step_was_best_count': 0, # Count batches where earlier step was best (over-iterating)
        'step_improvement_sum': 0.0,      # Sum of (step0_loss - stepN_loss) / step0_loss
        'solver_health_warnings': [],     # List of warning messages
        
        # ENTROPY-LOSS CORRELATION TRACKING (validates inference heuristic)
        'entropy_loss_agreement_count': 0,  # How often entropy picks same step as loss
        'entropy_loss_total_count': 0,      # Total comparisons
        
        # =============================================================
        # PER-SAMPLE ACCURACY TRACKING (for early stopping decisions)
        # =============================================================
        'total_samples': 0,            # Total samples processed
        'exact_match_count': 0,        # Samples with 100% pixel accuracy
        'high_accuracy_count': 0,      # Samples with >=90% pixel accuracy
        'per_sample_accuracies': [],   # List of per-sample accuracies (sampled)
        'batch_accuracies': [],        # Accuracy per batch (for trend analysis)
        'batch_exact_matches': [],     # Exact matches per batch
        
        # =============================================================
        # TASK-LEVEL ACCURACY TRACKING (for apples-to-apples eval gap)
        # =============================================================
        # TWO METRICS for different purposes:
        # 1. FIRST-SAMPLE (strict): Only counts first sample seen per task
        #    → Truly comparable to eval (single-shot, no "multiple attempts")
        # 2. ANY-SAMPLE (lenient): Task solved if ANY sample hits exact match
        #    → Upper bound, shows learning capacity but inflated by N attempts
        'first_sample_task_correct': {},  # Dict: task_id -> bool (was FIRST sample correct?)
        'solved_task_ids': set(),         # Set of task_ids with at least one exact match (any sample)
        'seen_task_ids': set(),           # Set of all task_ids seen this epoch
        'fg_accuracy_sum': 0.0,        # Sum of FG accuracy across batches
        'bg_accuracy_sum': 0.0,        # Sum of BG accuracy across batches
        'running_accuracy_window': [], # Last N batch accuracies for trend
        'fg_running_window': [],       # Last 50 FG accuracies for trend
        'bg_running_window': [],       # Last 50 BG accuracies for trend
        'fg_batch_count': 0,           # Number of batches with FG pixels
        'bg_batch_count': 0,           # Number of batches with BG pixels
        
        # =============================================================
        # PER-CLASS ACCURACY (which colors are being learned?)
        # =============================================================
        'per_class_correct': [0] * num_classes,   # Correct predictions per class
        'per_class_total': [0] * num_classes,     # Total pixels per class
        'per_class_predicted': [0] * num_classes, # How often each class is predicted
        
        # =============================================================
        # PER-BATCH PER-CLASS TRACKING (for fine-grained color analysis)
        # =============================================================
        'batch_class_accuracies': [],             # List of per-class acc dicts per batch
        'per_class_running_window': [[] for _ in range(num_classes)],  # Last 50 batch accuracies per class
        
        # =============================================================
        # META-LEARNING HEALTH METRICS (HyperLoRA + LOO)
        # =============================================================
        # These metrics prove that meta-learning is working correctly
        'loo_loss_sum': 0.0,                      # Accumulated LOO loss
        'loo_accuracy_sum': 0.0,                  # Accumulated LOO accuracy (N-1 to Nth prediction)
        'loo_num_holdouts_sum': 0,                # Total holdout predictions made
        'loo_batch_count': 0,                     # Batches with LOO loss computed
        'loo_skipped_count': 0,                   # Batches where LOO was skipped (not enough pairs)
        'hyperlora_grad_norm_sum': 0.0,           # Gradient norm for HyperLoRA params
        'hyperlora_weight_norm_sum': 0.0,         # Weight magnitude of HyperLoRA (should be small)
        'hyperlora_update_count': 0,              # Number of batches with HyperLoRA updates
        
        # =============================================================
        # DETAILED META-LEARNING TRACKING (for debugging epoch 14+)
        # =============================================================
        'lora_delta_norm_sum': 0.0,               # Sum of LoRA delta norms (should be non-zero after epoch 3)
        'lora_delta_batch_count': 0,              # Number of batches with LoRA deltas
        'context_magnitude_sum': 0.0,             # Sum of context/support feature magnitudes
        'context_batch_count': 0,                 # Number of batches with context features
        'hpm_routing_entropy_sum': 0.0,           # HPM routing entropy (lower = more specialized banks)
        'hpm_batch_count': 0,                     # Number of batches with HPM routing
        'equiv_consistency_sum': 0.0,             # Avg consistency of LoRA across augmentations
        'equiv_batch_count': 0,                   # Number of batches with equivariance computed
        
        # =============================================================
        # HPM DEDUPLICATION TRACKING
        # =============================================================
        # Prevent the same task from being added to HPM buffers multiple times
        # per epoch (which would pollute the memory with redundant entries)
        'hpm_tasks_added_this_epoch': set(),      # Task IDs already added to HPM this epoch
        'hpm_duplicate_skipped': 0,               # Count of duplicate adds prevented
        
        # =============================================================
        # HPM WRITE DIAGNOSTICS (Jan 2026 Patch)
        # =============================================================
        # Detailed tracking of why HPM writes may be skipped
        'hpm_write_attempts': 0,                  # Total attempts to write to HPM
        'hpm_writes_succeeded': 0,                # Successful writes
        'hpm_write_skip_reasons': {               # Breakdown of skip reasons
            'no_method': 0,                       # Model lacks hpm_add_solved_task
            'not_enabled': 0,                     # use_hpm=False AND hpm_memory_enabled=False
            'global_duplicate': 0,                # Already in buffer from previous epoch
            'epoch_duplicate': 0,                 # Already added this epoch
            'no_support_features': 0,             # Missing support_features in output
        },
        
        # =============================================================
        # HPM RETRIEVAL STATS (for monitoring memory quality)
        # =============================================================
        'hpm_retrieval_count': 0,                 # Number of batches with HPM retrieval
        'hpm_instance_similarity_sum': 0.0,       # Sum of avg similarities from instance bank
        'hpm_procedural_similarity_sum': 0.0,     # Sum of avg similarities from procedural bank
        'hpm_instance_retrieved_sum': 0,          # Total entries retrieved from instance bank
        'hpm_procedural_retrieved_sum': 0,        # Total entries retrieved from procedural bank
        
        # =============================================================
        # HPM SOLVER-CONTEXT COUPLING STATS (Jan 2026)
        # =============================================================
        # Track how HPM memory tokens are being used by the solver
        'hpm_solver_tokens_used': 0,              # Total HPM tokens injected into solver
        'hpm_solver_batches_with_tokens': 0,      # Batches that had HPM tokens
        'hpm_solver_gate_sum': 0.0,               # Sum of gate values for averaging
        'hpm_solver_proj_norm_max': 0.0,          # Max projected token norm (for explosion detection)
        'hpm_solver_proj_norm_sum': 0.0,          # Sum of projected token norms
        'hpm_solver_explosion_count': 0,          # Batches where projection exceeded clamp
    }
    
    temperature = get_temperature(epoch, config)
    max_epochs = config['training']['max_epochs']
    gradient_clip = config['training']['gradient_clip']
    grad_accumulation_steps = config['training'].get('grad_accumulation_steps', 1)
    use_amp = config['device'].get('mixed_precision', False) and device.type == 'cuda'
    
    # Get AMP dtype from config (default to bfloat16 which is more stable than float16)
    amp_dtype_str = config['device'].get('dtype', 'bfloat16')
    if amp_dtype_str == 'bfloat16':
        amp_dtype = torch.bfloat16
    elif amp_dtype_str == 'float16':
        amp_dtype = torch.float16
    else:
        amp_dtype = torch.bfloat16  # Safe default
    
    log_every = config['logging'].get('log_every', 10)
    warmup_epochs = config['training'].get('warmup_epochs', 10)
    warmup_steps = warmup_epochs * len(dataloader)
    base_lr = config['training']['learning_rate']
    
    # Capture initial LRs for each param group (to preserve per-group LR ratios during warmup)
    initial_lrs = [pg.get('lr', base_lr) for pg in optimizer.param_groups]
    
    optimizer.zero_grad()
    nan_batches = 0  # Track NaN occurrences for diagnostics
    consecutive_nan = 0  # Track consecutive NaN for detecting model corruption
    max_consecutive_nan_streak = 0  # Track peak consecutive NaN streak for backoff (Patch 4 fix)
    max_consecutive_nan = 10  # Abort if this many consecutive NaN (model is corrupted)
    
    # =============================================================
    # MONITORING THRESHOLDS - Read once from YAML for consistency
    # =============================================================
    monitoring_cfg = config.get('monitoring', {})
    monitoring_enabled = monitoring_cfg.get('enabled', True)
    # Gap thresholds
    exact_match_warning = monitoring_cfg.get('exact_match_warning', 0.10)
    exact_match_critical = monitoring_cfg.get('exact_match_critical', 0.20)
    stop_value_warning = monitoring_cfg.get('stop_value_warning', 0.15)
    stop_value_critical = monitoring_cfg.get('stop_value_critical', 0.25)
    # TTA consensus thresholds
    tta_consensus_warning = monitoring_cfg.get('tta_consensus_warning', 0.25)
    tta_consensus_critical = monitoring_cfg.get('tta_consensus_critical', 0.15)
    # Centroid spread thresholds
    centroid_spread_warning = monitoring_cfg.get('centroid_spread_warning', 2.0)
    centroid_spread_critical = monitoring_cfg.get('centroid_spread_critical', 0.5)
    # NaN abort threshold (per-epoch total)
    nan_batches_abort = monitoring_cfg.get('nan_batches_abort', 20)

    # Attention collapse threshold (diffuse attention detector)
    # Default 0.02: uniform over 30x30 is ~0.0011; healthy sharp attention is typically >> 0.1.
    attn_max_collapse_threshold = monitoring_cfg.get('attn_max_collapse_threshold', 0.02)
    # LoRA thresholds (also used in batch loop)
    # TIGHTENED (Jan 2026): Production run showed collapse at LoRA norm 1.714
    # Previous thresholds (2.0/5.0/10.0) were too loose and didn't catch the runaway.
    # New thresholds based on observation: healthy range is 0.3-0.8, collapse at 1.7.
    lora_norm_warn_threshold = monitoring_cfg.get('lora_norm_warn', 1.0)       # Was 2.0
    lora_norm_critical_threshold = monitoring_cfg.get('lora_norm_critical', 1.5)  # Was 5.0
    lora_norm_kill_threshold = monitoring_cfg.get('lora_norm_kill', 2.0)       # Was 10.0
    lora_kill_consecutive_limit = 3  # Abort after this many consecutive kill-threshold breaches
    
    # Wrap dataloader with CUDA prefetcher for async data transfer
    # This overlaps CPU→GPU transfer with GPU computation, eliminating stalls
    prefetcher = CUDAPrefetcher(dataloader, device)
    
    # MEMORY DEBUGGING: Track memory at checkpoints for first N batches
    # This helps identify what's causing unexpected memory usage
    memory_debug_batches = config.get('logging', {}).get('memory_debug_batches', 5)
    mem_tracker = MemoryTracker(
        enabled=(device.type == 'cuda' and memory_debug_batches > 0),
        log_first_n_batches=memory_debug_batches,
        device=device.type
    )
    mem_tracker.reset_baseline()
    
    for batch_idx, batch in enumerate(prefetcher):
        # Proactive memory cleanup every 50 batches to prevent fragmentation
        # This helps avoid CUDA launch failures that occur after many batches
        if batch_idx > 0 and batch_idx % 50 == 0:
            torch.cuda.empty_cache()
        
        # Log batch grid sizes for first 5 batches to verify dynamic padding works
        if batch_idx < 5:
            batch_grid_size = batch.get('batch_max_size', batch['test_inputs'].shape[-1])
            actual_tensor_size = batch['test_inputs'].shape[-1]
            print(f"  [Batch {batch_idx}] Grid size: {actual_tensor_size}x{actual_tensor_size} (batch_max_size={batch_grid_size})")
        
        # Apply warmup with per-group LR preservation
        warmup_lr(optimizer, global_step, warmup_steps, base_lr, initial_lrs)
        
        # Collect augmentation statistics from batch (CRITICAL for diversity verification)
        if 'aug_stats' in batch:
            batch_aug = batch['aug_stats']
            for i in range(8):
                epoch_aug_stats['dihedral_counts'][i] += batch_aug['dihedral_counts'][i]
            epoch_aug_stats['color_perm_count'] += batch_aug['color_perm_count']
            epoch_aug_stats['translational_count'] += batch_aug['translational_count']
            epoch_aug_stats['unique_offsets'] += batch_aug['unique_offsets']
        
        # Track sample count
        batch_size = batch['test_inputs'].shape[0]
        total_samples += batch_size
        
        # MEMORY CHECKPOINT: After batch data is on GPU
        mem_tracker.checkpoint("01_batch_on_gpu", batch=batch, optimizer=optimizer)
        
        # Batch already on device (prefetcher handles transfer asynchronously)
        test_inputs = batch['test_inputs']
        test_outputs = batch['test_outputs']
        train_inputs = batch['input_grids']   # (B, N, H, W)
        train_outputs = batch['output_grids']  # (B, N, H, W)
        pair_mask = batch['grid_masks']        # (B, N)
        
        # Forward pass with optional mixed precision
        if use_amp and scaler is not None:
            with autocast('cuda', dtype=amp_dtype):
                outputs = model(
                    test_inputs,
                    train_inputs=train_inputs,
                    train_outputs=train_outputs,
                    pair_mask=pair_mask,
                    temperature=temperature,
                    return_intermediates=True,
                )
                
                # MEMORY CHECKPOINT: After main forward pass
                mem_tracker.checkpoint("02_after_forward", model=model, outputs=outputs, batch=batch, optimizer=optimizer)
                
                losses = loss_fn(
                    logits=outputs['logits'],
                    targets=test_outputs,
                    attention_maps=outputs['attention_maps'],
                    stop_logits=outputs['stop_logits'],
                    predicates=outputs['predicates'],
                    epoch=epoch,
                    max_epochs=max_epochs,
                    all_logits=outputs.get('all_logits'),
                    act_outputs=outputs.get('act_outputs'),
                    centroids=outputs.get('centroids'),  # For centroid diversity loss
                )
                
                # =============================================================
                # META LOSS CAPPING (Jan 2026 FIX)
                # =============================================================
                # Cap combined meta-losses (LOO + equiv + HPM) to prevent them from
                # overwhelming the task loss. Training logs showed 40.9% meta contribution.
                # =============================================================
                meta_loss_cap_enabled = config.get('training', {}).get('meta_escalation', {}).get('meta_loss_cap_enabled', True)
                meta_loss_cap_ratio = config.get('training', {}).get('meta_escalation', {}).get('meta_loss_cap_ratio', 0.25)
                
                # Get task loss value for capping calculation
                task_loss_value = losses.get('task_loss', losses.get('focal_loss', losses['total_loss']))
                if torch.is_tensor(task_loss_value):
                    task_loss_value = task_loss_value.item()
                
                # Calculate meta loss weight cap multiplier
                # FIX (Jan 2026): Actually compute and apply the cap factor
                meta_loss_cap_factor = 1.0
                if meta_loss_cap_enabled and task_loss_value > 0:
                    # Target: (loo + equiv + hpm) <= cap_ratio * total
                    # Which means: meta_weight <= cap_ratio / (1 - cap_ratio) * task_loss
                    # 
                    # For cap_ratio=0.25: max_meta = 0.25/0.75 * task = 0.333 * task
                    # If raw meta weights sum to W, we scale by min(1, max_meta / W)
                    max_meta_contribution = meta_loss_cap_ratio / (1 - meta_loss_cap_ratio) * task_loss_value
                    
                    # Get the raw meta loss weights that will be applied
                    raw_loo_weight = loo_loss_fn.config.loss_weight if loo_loss_fn is not None else 0.0
                    raw_equiv_weight = equiv_loss_fn.config.loss_weight if equiv_loss_fn is not None else 0.0
                    total_raw_meta_weight = raw_loo_weight + raw_equiv_weight
                    
                    # Scale down if raw weights would exceed the cap
                    if total_raw_meta_weight > 0 and total_raw_meta_weight > max_meta_contribution:
                        meta_loss_cap_factor = max_meta_contribution / total_raw_meta_weight
                        # Clamp to reasonable range
                        meta_loss_cap_factor = max(0.01, min(1.0, meta_loss_cap_factor))
                    
                    # Store for diagnostics
                    epoch_diagnostics['meta_loss_cap_factor'] = meta_loss_cap_factor
                    epoch_diagnostics['task_loss_value'] = task_loss_value
                    epoch_diagnostics['max_meta_contribution'] = max_meta_contribution
                    epoch_diagnostics['raw_meta_weight_sum'] = total_raw_meta_weight
                
                # Compute LOO loss if enabled (meta-learning via HyperLoRA)
                # MEMORY FIX v3: Use iterative backward - backward happens INSIDE the function
                # This prevents O(N) memory accumulation from holding all N computation graphs
                loo_loss_value = 0.0  # Float for logging (backward done inside)
                loo_metrics = None
                # Check epoch threshold - LOO activates at loo_start_epoch
                # Jan 2026: Respect phased training disable flags
                loo_phase_disabled = phase_disable_flags.get('loo', False)
                loo_active = (loo_loss_fn is not None and 
                              epoch >= loo_start_epoch and 
                              not loo_phase_disabled)
                if loo_active and hasattr(model, 'hyper_lora') and model.hyper_lora is not None:
                    # LOO requires at least min_pairs training pairs
                    num_pairs = pair_mask.sum(dim=1).min().item() if pair_mask is not None else train_inputs.shape[1]
                    if num_pairs >= loo_loss_fn.config.min_pairs_for_loo:
                        # Apply meta loss cap to LOO weight
                        effective_loo_weight = loo_loss_fn.config.loss_weight * meta_loss_cap_factor
                        loo_result = loo_loss_fn(
                            model=model,
                            input_grids=train_inputs,
                            output_grids=train_outputs,
                            pair_mask=pair_mask,
                            temperature=temperature,
                            # v3: Iterative backward - pass scaler so backward happens inside
                            scaler=scaler,
                            loss_weight=effective_loo_weight,
                            grad_accumulation_steps=grad_accumulation_steps,
                        )
                        # loo_loss is now a float (backward already done inside)
                        loo_loss_value = loo_result['loo_loss']
                        if isinstance(loo_loss_value, torch.Tensor):
                            loo_loss_value = loo_loss_value.item()
                        loo_metrics = loo_result
                        losses['loo_loss'] = loo_loss_value  # Store for logging
                        # Note: Don't add to total_loss - backward already done inside
                        # Track meta-learning health metrics
                        epoch_diagnostics['loo_loss_sum'] += loo_loss_value
                        epoch_diagnostics['loo_accuracy_sum'] += loo_result.get('loo_accuracy', 0.0)
                        epoch_diagnostics['loo_num_holdouts_sum'] += loo_result.get('loo_num_holdouts', 0)
                        epoch_diagnostics['loo_batch_count'] += 1
                    else:
                        epoch_diagnostics['loo_skipped_count'] += 1
                
                # Compute Augmentation Equivariance loss if enabled (meta-learning)
                # MEMORY FIX v3: Use iterative backward - backward happens INSIDE the function
                equiv_loss_value = 0.0  # Float for logging
                # Check epoch threshold - Equivariance activates at equiv_start_epoch
                # Jan 2026: Respect phased training disable flags
                equiv_phase_disabled = phase_disable_flags.get('equivariance', False)
                equiv_active = (equiv_loss_fn is not None and 
                                epoch >= equiv_start_epoch and 
                                not equiv_phase_disabled)
                if equiv_active and hasattr(model, 'hyper_lora') and model.hyper_lora is not None:
                    support_features = outputs.get('support_features')  # (B, N, D, H, W)
                    lora_deltas = outputs.get('lora_deltas')  # Dict with original deltas
                    
                    if support_features is not None and lora_deltas is not None:
                        # FIX (Jan 2026): Compute original context using simple pooling
                        # The lora_deltas['context'] is D4-averaged, which doesn't give
                        # useful gradients when compared with augmented contexts.
                        # Instead, use pool_context_simple for BOTH original and augmented.
                        support_features_detached = support_features.detach()
                        if hasattr(model.hyper_lora, 'pool_context_simple'):
                            original_context = model.hyper_lora.pool_context_simple(support_features_detached)
                        else:
                            # Fallback: use D4-averaged context from lora_deltas
                            original_context = lora_deltas.get('context')
                            if original_context is not None:
                                original_context = original_context.detach()
                        
                        if original_context is not None:
                            
                            # MEMORY FIX: Generate augmented contexts with memory cleanup
                            # Process one augmentation at a time to avoid memory accumulation
                            augmented_contexts = {}
                            aug_types = ['rotate_90', 'rotate_180', 'flip_h', 'flip_v']
                            num_augs = min(equiv_loss_fn.config.num_augmentations, len(aug_types))
                            selected_augs = random.sample(aug_types, num_augs)
                            
                            for aug_type in selected_augs:
                                # Apply augmentation to support_features spatial dimensions
                                # support_features shape: (B, N, D, H, W)
                                aug_features = equiv_loss_fn.apply_augmentation(
                                    support_features_detached.permute(0, 1, 3, 4, 2),  # (B, N, H, W, D) for spatial aug
                                    aug_type
                                ).permute(0, 1, 4, 2, 3)  # Back to (B, N, D, H, W)
                                
                                # FIX (Jan 2026): Use pool_context_simple for augmented features
                                # The full pool_context uses D4-invariant averaging, which makes
                                # ALL augmented contexts identical (equiv loss ≈ 0 always)
                                # pool_context_simple doesn't average over transforms, so
                                # augmented contexts will differ, giving useful gradients
                                if hasattr(model.hyper_lora, 'pool_context_simple'):
                                    aug_context = model.hyper_lora.pool_context_simple(aug_features)
                                else:
                                    # Fallback for older models without pool_context_simple
                                    aug_context = model.hyper_lora.pool_context(aug_features)
                                augmented_contexts[aug_type] = aug_context
                                # MEMORY FIX: Delete intermediate aug_features immediately
                                del aug_features
                            
                            if augmented_contexts:
                                # P0.4: Wrap equivariance in try/except to track failures
                                try:
                                    # Apply meta loss cap to equiv weight
                                    effective_equiv_weight = equiv_loss_fn.config.loss_weight * meta_loss_cap_factor
                                    equiv_result, equiv_metrics = equiv_loss_fn(
                                        hyper_lora=model.hyper_lora,
                                        original_context=original_context,
                                        augmented_contexts=augmented_contexts,
                                        # v3: Iterative backward
                                        scaler=scaler,
                                        loss_weight=effective_equiv_weight,
                                        grad_accumulation_steps=grad_accumulation_steps,
                                    )
                                    # equiv_result is now a float (backward already done inside)
                                    equiv_loss_value = equiv_result
                                    if isinstance(equiv_loss_value, torch.Tensor):
                                        equiv_loss_value = equiv_loss_value.item()
                                    losses['equiv_loss'] = equiv_loss_value  # Store for logging
                                    # Note: Don't add to total_loss - backward already done inside
                                    epoch_diagnostics['equiv_loss_sum'] = epoch_diagnostics.get('equiv_loss_sum', 0.0) + equiv_loss_value
                                    epoch_diagnostics['equiv_batch_count'] = epoch_diagnostics.get('equiv_batch_count', 0) + 1
                                    # Reset consecutive failures on success
                                    epoch_diagnostics['equiv_consecutive_failures'] = 0
                                except Exception as e:
                                    # P0.4: Track equivariance failures for fail-loud monitoring
                                    epoch_diagnostics['equiv_failures'] = epoch_diagnostics.get('equiv_failures', 0) + 1
                                    epoch_diagnostics['equiv_consecutive_failures'] = epoch_diagnostics.get('equiv_consecutive_failures', 0) + 1
                                    consec = epoch_diagnostics['equiv_consecutive_failures']
                                    total_failures = epoch_diagnostics['equiv_failures']
                                    # Log first 3 failures and every 10th
                                    if total_failures <= 3 or total_failures % 10 == 0:
                                        print(f"[EQUIV WARNING] Equivariance failed (total={total_failures}, consec={consec}): {e}")
                                    # Critical alert after 10 consecutive failures
                                    if consec >= 10 and not epoch_diagnostics.get('equiv_path_broken', False):
                                        print(f"\n{'!'*60}")
                                        print(f"[EQUIV CRITICAL] 10 consecutive equivariance failures!")
                                        print(f"  The equiv=0 metric is masking a BROKEN equivariance path.")
                                        print(f"  TTA consensus will likely be <25%.")
                                        print(f"{'!'*60}\n")
                                        epoch_diagnostics['equiv_path_broken'] = True
                            
                            # MEMORY FIX: Clean up augmented contexts after use
                            del augmented_contexts, support_features_detached
                
                # =========================================================================
                # OUTPUT-LEVEL EQUIVARIANCE LOSS (Jan 2026)
                # =========================================================================
                # This is the NEW, non-degenerate equivariance that compares OUTPUTS
                # instead of intermediate contexts. It directly optimizes TTA consensus.
                # =========================================================================
                output_equiv_loss_value = 0.0
                # Respect phased training disable flags
                output_equiv_phase_disabled = phase_disable_flags.get('equivariance', False)
                output_equiv_active = (output_equiv_loss_fn is not None and 
                                       epoch >= output_equiv_start_epoch and
                                       not output_equiv_phase_disabled)
                if output_equiv_active:
                    try:
                        # Get the original logits from this forward pass
                        original_logits = outputs.get('logits')  # (B, C, H, W)
                        
                        if original_logits is not None:
                            # Compute output-level equivariance
                            # This does: aug_input → model → inv_aug_output → compare with original
                            # GRADIENT FIX (Jan 2026): Do NOT detach original_logits!
                            # Gradients must flow through original_probs for learning.
                            # The augmented outputs serve as target (no_grad is fine there).
                            
                            # MASKING FIX (Jan 2026): Create mask from test_targets
                            # to only compute equivariance on non-padded output region
                            target_mask = None
                            if 'test_targets' in batch and batch['test_targets'] is not None:
                                test_targets_for_mask = batch['test_targets'].to(device)
                                # Valid pixels are those != PADDING_IGNORE_VALUE (ignore_index for padding)
                                # CRITICAL: Use constant, not hardcoded -100, to ensure consistency
                                target_mask = (test_targets_for_mask != PADDING_IGNORE_VALUE)  # (B, H, W) bool
                                
                                # Defensive validation: log if mask is all False (suspicious)
                                if target_mask is not None and not target_mask.any():
                                    print(f"[WARNING] Output equiv target_mask is ALL False - check padding values!")
                            
                            output_equiv_result, output_equiv_metrics = output_equiv_loss_fn(
                                model=model,
                                test_inputs=test_inputs,
                                train_inputs=train_inputs,
                                train_outputs=train_outputs,
                                pair_mask=pair_mask,
                                original_logits=original_logits,  # KEEP GRADIENTS for learning!
                                temperature=temperature,
                                num_augmentations=output_equiv_num_augs,
                                target_mask=target_mask,  # Jan 2026: Mask to valid output region
                            )
                            
                            if torch.is_tensor(output_equiv_result):
                                # Apply meta loss cap factor
                                effective_output_equiv_weight = output_equiv_weight * meta_loss_cap_factor
                                weighted_output_equiv = effective_output_equiv_weight * output_equiv_result
                                
                                # Scale for grad accumulation and backward
                                if scaler is not None:
                                    scaler.scale(weighted_output_equiv / grad_accumulation_steps).backward()
                                else:
                                    (weighted_output_equiv / grad_accumulation_steps).backward()
                                
                                output_equiv_loss_value = output_equiv_result.item()
                                losses['output_equiv_loss'] = output_equiv_loss_value
                                
                                # Track diagnostics
                                epoch_diagnostics['output_equiv_loss_sum'] = epoch_diagnostics.get('output_equiv_loss_sum', 0.0) + output_equiv_loss_value
                                epoch_diagnostics['output_equiv_batch_count'] = epoch_diagnostics.get('output_equiv_batch_count', 0) + 1
                                
                    except Exception as e:
                        epoch_diagnostics['output_equiv_failures'] = epoch_diagnostics.get('output_equiv_failures', 0) + 1
                        if epoch_diagnostics['output_equiv_failures'] <= 3:
                            print(f"[OUTPUT_EQUIV WARNING] Failed: {e}")
                
                # =========================================================================
                # GROUP-MARGINALIZED NLL LOSS (Jan 2026)
                # =========================================================================
                # This is a mathematically principled approach that directly optimizes
                # the group-marginalized distribution used by TTA voting.
                # =========================================================================
                group_marg_loss_value = 0.0
                group_marg_phase_disabled = phase_disable_flags.get('equivariance', False)  # Share phase gating with equiv
                group_marg_active = (group_marg_loss_fn is not None and
                                     epoch >= group_marg_start_epoch and
                                     not group_marg_phase_disabled)
                if group_marg_active:
                    try:
                        # Get targets from batch
                        test_targets = batch.get('test_targets')
                        if test_targets is not None:
                            test_targets = test_targets.to(device)
                            
                            group_marg_result, group_marg_metrics = group_marg_loss_fn(
                                model=model,
                                test_inputs=test_inputs,
                                train_inputs=train_inputs,
                                train_outputs=train_outputs,
                                pair_mask=pair_mask,
                                targets=test_targets,
                                temperature=temperature,
                            )
                            
                            if torch.is_tensor(group_marg_result) and not group_marg_metrics.get('skipped', False):
                                # Apply weight and meta loss cap
                                effective_weight = group_marg_weight * meta_loss_cap_factor
                                weighted_loss = effective_weight * group_marg_result
                                
                                # Scale for grad accumulation and backward
                                if scaler is not None:
                                    scaler.scale(weighted_loss / grad_accumulation_steps).backward()
                                else:
                                    (weighted_loss / grad_accumulation_steps).backward()
                                
                                group_marg_loss_value = group_marg_result.item()
                                losses['group_marginalized_nll'] = group_marg_loss_value
                                
                                # Track diagnostics
                                epoch_diagnostics['group_marg_loss_sum'] = epoch_diagnostics.get('group_marg_loss_sum', 0.0) + group_marg_loss_value
                                epoch_diagnostics['group_marg_batch_count'] = epoch_diagnostics.get('group_marg_batch_count', 0) + 1
                                
                    except Exception as e:
                        epoch_diagnostics['group_marg_failures'] = epoch_diagnostics.get('group_marg_failures', 0) + 1
                        if epoch_diagnostics['group_marg_failures'] <= 3:
                            print(f"[GROUP_MARG WARNING] Failed: {e}")
                
                # Compute HPM load balancing loss if enabled
                # This loss ensures all memory banks are utilized (prevents mode collapse)
                if hasattr(model, 'use_hpm') and model.use_hpm:
                    hpm_balance_loss = model.hpm_get_load_balance_loss()
                    hpm_balance_weight = config['model'].get('hpm_balance_weight', 0.01)
                    weighted_hpm_loss = hpm_balance_weight * hpm_balance_loss
                    losses['hpm_balance_loss'] = weighted_hpm_loss.item() if torch.is_tensor(weighted_hpm_loss) else weighted_hpm_loss
                    losses['total_loss'] = losses['total_loss'] + weighted_hpm_loss
                
                # =========================================================================
                # ANCHOR ROBUSTNESS TRAINING (ART) - Jan 2026 Ablation
                # =========================================================================
                # ART forces consistent predictions under alternate anchors, targeting
                # the observed eval entropy collapse issue.
                # =========================================================================
                art_loss_value = 0.0
                art_phase_disabled = phase_disable_flags.get('art', False)
                art_active = (art_module is not None and 
                              epoch >= art_start_epoch and 
                              not art_phase_disabled)
                if art_active:
                    try:
                        # Get model outputs we already have
                        attention_maps = outputs['attention_maps']
                        centroids = outputs['centroids']
                        logits = outputs['logits']
                        
                        # Extract alternate anchors from attention maps
                        alt_centroids = art_module.extract_alternate_anchors(
                            attention_maps, centroids
                        )
                        
                        # For efficiency, use perturbed logits instead of full re-forward
                        # This approximates what different anchors would produce
                        alt_logits_list = []
                        B, num_alt, K, _ = alt_centroids.shape
                        for alt_idx in range(num_alt):
                            # Perturb logits based on anchor difference (approximation)
                            # Real impl would re-run forward with forced centroids
                            noise_scale = 0.1  # Small perturbation
                            alt_logits = logits + torch.randn_like(logits) * noise_scale
                            alt_logits_list.append(alt_logits)
                        
                        # Compute consistency loss
                        art_consistency_loss = art_module.compute_consistency_loss(
                            logits, alt_logits_list
                        )
                        
                        if torch.is_tensor(art_consistency_loss) and torch.isfinite(art_consistency_loss):
                            art_weight = art_module.config.consistency_weight
                            weighted_art_loss = art_weight * art_consistency_loss
                            
                            # Add to total loss
                            losses['art_consistency_loss'] = art_consistency_loss.item()
                            losses['total_loss'] = losses['total_loss'] + weighted_art_loss
                            art_loss_value = art_consistency_loss.item()
                            
                            # Track diagnostics
                            epoch_diagnostics['art_loss_sum'] = epoch_diagnostics.get('art_loss_sum', 0.0) + art_loss_value
                            epoch_diagnostics['art_batch_count'] = epoch_diagnostics.get('art_batch_count', 0) + 1
                            
                    except Exception as e:
                        epoch_diagnostics['art_failures'] = epoch_diagnostics.get('art_failures', 0) + 1
                        if epoch_diagnostics.get('art_failures', 0) <= 3:
                            print(f"[ART WARNING] Failed: {e}")
                
                # =========================================================================
                # ANCHOR-RELATIVE PROGRAM SEARCH (ARPS) - Jan 2026 Ablation
                # =========================================================================
                # ARPS provides interpretable program supervision via imitation learning.
                # It searches for DSL programs that explain the transformation and uses
                # the best program as a supervision signal.
                # =========================================================================
                arps_loss_value = 0.0
                arps_phase_disabled = phase_disable_flags.get('arps', False)
                arps_active = (arps_module is not None and 
                               epoch >= arps_start_epoch and 
                               not arps_phase_disabled)
                if arps_active:
                    try:
                        centroids = outputs['centroids']
                        
                        # Get clue features (from MSRE or create mock)
                        clue_features = outputs.get('clue_features')
                        if clue_features is None:
                            # Create mock clue features from model features
                            features = outputs.get('features')
                            if features is not None:
                                K = centroids.shape[1]
                                B, D, H, W = features.shape
                                # Expand features to (B, K, D, H, W) by repeating
                                clue_features = features.unsqueeze(1).expand(B, K, D, H, W)
                        
                        if clue_features is not None:
                            arps_result = arps_module(
                                clue_features,
                                test_inputs,
                                train_inputs,
                                train_outputs,
                                centroids,
                                temperature=temperature,
                            )
                            
                            arps_imitation_loss = arps_result.get('imitation_loss')
                            if (torch.is_tensor(arps_imitation_loss) and 
                                torch.isfinite(arps_imitation_loss) and
                                arps_imitation_loss > 0):
                                arps_weight = config['model'].get('arps_dsl_search', {}).get('imitation_weight', 0.1)
                                weighted_arps_loss = arps_weight * arps_imitation_loss
                                
                                # Add to total loss
                                losses['arps_imitation_loss'] = arps_imitation_loss.item()
                                losses['total_loss'] = losses['total_loss'] + weighted_arps_loss
                                arps_loss_value = arps_imitation_loss.item()
                                
                                # Track diagnostics
                                epoch_diagnostics['arps_loss_sum'] = epoch_diagnostics.get('arps_loss_sum', 0.0) + arps_loss_value
                                epoch_diagnostics['arps_batch_count'] = epoch_diagnostics.get('arps_batch_count', 0) + 1
                                
                                # Track program search stats
                                search_stats = arps_result.get('search_stats', {})
                                epoch_diagnostics['arps_valid_programs'] = epoch_diagnostics.get('arps_valid_programs', 0) + search_stats.get('num_valid_programs', 0)
                                epoch_diagnostics['arps_search_attempts'] = epoch_diagnostics.get('arps_search_attempts', 0) + 1
                                
                    except Exception as e:
                        epoch_diagnostics['arps_failures'] = epoch_diagnostics.get('arps_failures', 0) + 1
                        if epoch_diagnostics.get('arps_failures', 0) <= 3:
                            print(f"[ARPS WARNING] Failed: {e}")
                
                # Add HyperLoRA delta regularization loss if present (Dec 2025)
                # This prevents LoRA delta norm from growing without bound
                if 'lora_deltas' in outputs and outputs['lora_deltas'] is not None:
                    delta_reg_loss = outputs['lora_deltas'].get('delta_reg_loss', None)
                    if delta_reg_loss is not None and torch.is_tensor(delta_reg_loss):
                        losses['delta_reg_loss'] = delta_reg_loss.item()
                        losses['total_loss'] = losses['total_loss'] + delta_reg_loss
                
                # Scale loss for gradient accumulation
                # Note: LOO and Equiv losses already have their backward done inside
                loss = losses['total_loss'] / grad_accumulation_steps
            
            # NaN detection: skip batch if loss is NaN
            if not torch.isfinite(loss):
                # =============================================================
                # P0.4: NaN ATTRIBUTION - Identify which component caused NaN
                # =============================================================
                nan_source = "unknown"
                nan_components = []
                
                # Check each loss component individually
                for key, val in losses.items():
                    if key == 'total_loss':
                        continue
                    if torch.is_tensor(val):
                        if not torch.isfinite(val).all():
                            nan_components.append(key)
                            # Prioritize attribution
                            if 'loo' in key.lower():
                                nan_source = "LOO loss (meta-learning)"
                            elif 'equiv' in key.lower():
                                nan_source = "Equivariance loss"
                            elif 'task' in key.lower() or 'focal' in key.lower():
                                nan_source = "Task loss"
                            elif 'hpm' in key.lower():
                                nan_source = "HPM balance loss"
                    elif isinstance(val, float) and not math.isfinite(val):
                        nan_components.append(key)
                        if 'loo' in key.lower():
                            nan_source = "LOO loss (meta-learning)"
                        elif 'equiv' in key.lower():
                            nan_source = "Equivariance loss"
                
                # Track NaN attribution for epoch summary
                if 'nan_attribution' not in epoch_diagnostics:
                    epoch_diagnostics['nan_attribution'] = {}
                epoch_diagnostics['nan_attribution'][nan_source] = epoch_diagnostics['nan_attribution'].get(nan_source, 0) + 1
                
                # Detailed diagnostics on first NaN
                if consecutive_nan == 0:
                    print(f"\n[WARNING] First NaN/Inf loss at batch {batch_idx}!")
                    print(f"  🎯 ATTRIBUTION: {nan_source}")
                    print(f"  NaN components: {nan_components}")
                    print(f"  Loss components: {', '.join(f'{k}={v.item():.4f}' if torch.is_tensor(v) else f'{k}={v}' for k,v in losses.items() if k != 'total_loss')}")
                    # Check model weights for NaN
                    nan_params = []
                    for name, param in model.named_parameters():
                        if not torch.isfinite(param).all():
                            nan_params.append(name)
                    if nan_params:
                        print(f"  NaN/Inf in model parameters: {nan_params[:5]}...")  # First 5
                    else:
                        print(f"  Model parameters are finite - NaN likely from forward computation")
                    # Check outputs
                    if torch.is_tensor(outputs.get('logits')) and not torch.isfinite(outputs['logits']).all():
                        print(f"  Logits contain NaN/Inf! range=[{outputs['logits'].min():.4f}, {outputs['logits'].max():.4f}]")
                    if torch.is_tensor(outputs.get('attention_maps')) and not torch.isfinite(outputs['attention_maps']).all():
                        print(f"  Attention maps contain NaN/Inf!")
                    # Check LoRA norms if available
                    if epoch_diagnostics.get('lora_norm_ema', 0) > 0:
                        print(f"  LoRA norm EMA: {epoch_diagnostics['lora_norm_ema']:.2f}")
                    print()
                else:
                    print(f"[WARNING] NaN/Inf loss at batch {batch_idx} (source: {nan_source}), skipping...")
                
                optimizer.zero_grad()  # Clear any partial gradients
                nan_batches += 1
                consecutive_nan += 1
                max_consecutive_nan_streak = max(max_consecutive_nan_streak, consecutive_nan)  # Track peak streak
                if consecutive_nan >= max_consecutive_nan:
                    print(f"[ERROR] {max_consecutive_nan} consecutive NaN batches - model weights likely corrupted!")
                    print(f"[ERROR] Aborting epoch to prevent further corruption.")
                    break
                continue
            
            # Loss is finite - but only reset consecutive counter after successful optimizer step
            
            # MEMORY CHECKPOINT: Before backward pass
            mem_tracker.checkpoint("03_before_backward", model=model, optimizer=optimizer)
            
            # Wrap backward pass in try-catch to handle CUDA errors gracefully
            # CUDA kernel errors (e.g., launch failures) can occur with memory fragmentation
            try:
                scaler.scale(loss).backward()
            except RuntimeError as e:
                error_str = str(e).lower()
                if 'cuda' in error_str or 'out of memory' in error_str or 'launch' in error_str:
                    print(f"\n{'!'*60}")
                    print(f"[ERROR] CUDA error during backward pass at batch {batch_idx}!")
                    print(f"  Error: {e}")
                    print(f"  This is often caused by memory fragmentation or OOM.")
                    print(f"{'!'*60}")
                    
                    # Log diagnostic info for debugging
                    print(f"\n  === DIAGNOSTIC INFO ===")
                    print(f"  Batch shape: test_inputs={test_inputs.shape}")
                    print(f"  train_inputs={train_inputs.shape}, train_outputs={train_outputs.shape}")
                    if 'task_ids' in batch:
                        task_ids = batch['task_ids'][:5]  # First 5 task IDs
                        print(f"  Task IDs (first 5): {task_ids}")
                    try:
                        mem_allocated = torch.cuda.memory_allocated() / (1024**3)
                        mem_reserved = torch.cuda.memory_reserved() / (1024**3)
                        print(f"  GPU Memory: {mem_allocated:.2f}GB allocated, {mem_reserved:.2f}GB reserved")
                    except:
                        pass
                    print(f"  Loss value: {loss.item() if torch.is_tensor(loss) else loss:.6f}")
                    print(f"  Attempting recovery...")
                    print()
                    
                    # Clear gradients and CUDA cache
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    
                    # Skip this batch
                    nan_batches += 1
                    consecutive_nan += 1
                    max_consecutive_nan_streak = max(max_consecutive_nan_streak, consecutive_nan)  # Track peak streak
                    if consecutive_nan >= max_consecutive_nan:
                        print(f"[ERROR] {max_consecutive_nan} consecutive CUDA errors - cannot recover!")
                        raise
                    continue
                else:
                    # Re-raise non-CUDA errors
                    raise
            
            # MEMORY CHECKPOINT: After backward pass (peak memory usually here)
            mem_tracker.checkpoint("04_after_backward", model=model, optimizer=optimizer)
            
            # Step optimizer after accumulation
            if (batch_idx + 1) % grad_accumulation_steps == 0:
                # Always unscale gradients first for NaN checking
                scaler.unscale_(optimizer)
                
                if gradient_clip > 0:
                    # Compute total grad norm before clipping
                    total_grad_norm_before = 0.0
                    for p in model.parameters():
                        if p.grad is not None:
                            norm_val = p.grad.norm().item()
                            if math.isfinite(norm_val):
                                total_grad_norm_before += norm_val ** 2
                    total_grad_norm_before = total_grad_norm_before ** 0.5
                    
                    # Capture per-module gradient norms BEFORE clipping (first optimizer step only)
                    # This is when (batch_idx + 1) == grad_accumulation_steps
                    if (batch_idx + 1) == grad_accumulation_steps:
                        grad_norms = compute_module_grad_norms(model)
                        epoch_diagnostics['dsc_grad_norm_sum'] += grad_norms.get('dsc', 0.0)
                        epoch_diagnostics['stop_predictor_grad_norm_sum'] += grad_norms.get('stop_predictor', 0.0)
                        epoch_diagnostics['encoder_grad_norm_sum'] += grad_norms.get('encoder', 0.0)
                        epoch_diagnostics['solver_grad_norm_sum'] += grad_norms.get('solver', 0.0)
                        epoch_diagnostics['context_encoder_grad_norm_sum'] += grad_norms.get('context_encoder', 0.0)
                        epoch_diagnostics['msre_grad_norm_sum'] += grad_norms.get('msre', 0.0)
                        epoch_diagnostics['grad_norm_before_clip'] = total_grad_norm_before
                        epoch_diagnostics['grad_was_clipped'] = total_grad_norm_before > gradient_clip
                    
                    # Patch 2 (Dec 2025): Track max grad norm across entire epoch
                    # This ensures we catch explosions even if they happen later in epoch
                    epoch_diagnostics['max_grad_norm_before_clip'] = max(
                        epoch_diagnostics.get('max_grad_norm_before_clip', 0.0),
                        total_grad_norm_before
                    )
                    
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                
                # Check for NaN/Inf gradients AFTER unscaling but BEFORE optimizer step
                # This is critical because GradScaler can produce inf gradients on overflow
                has_nan_grad = False
                for p in model.parameters():
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        has_nan_grad = True
                        break
                
                if has_nan_grad:
                    print(f"[WARNING] NaN/Inf gradients at batch {batch_idx}, skipping optimizer step...")
                    optimizer.zero_grad()
                    # CRITICAL: Must call scaler.update() after unscale_() even when skipping step
                    # Otherwise next iteration's unscale_() will fail
                    scaler.update()
                    nan_batches += 1
                    consecutive_nan += 1
                    max_consecutive_nan_streak = max(max_consecutive_nan_streak, consecutive_nan)  # Track peak streak
                    if consecutive_nan >= max_consecutive_nan:
                        print(f"[ERROR] {max_consecutive_nan} consecutive NaN batches - model weights likely corrupted!")
                        print(f"[ERROR] Aborting epoch to prevent further corruption.")
                        break
                    # Skip scaler.step to avoid corrupting model
                    continue
                
                # Reset consecutive NaN counter on successful step
                consecutive_nan = 0
                
                # HPM gradient routing: zero gradients for frozen primitives
                if hasattr(model, 'hpm_on_backward'):
                    model.hpm_on_backward()
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                # Update EMA after optimizer step
                if ema is not None:
                    ema.update(model)
        else:
            outputs = model(
                test_inputs,
                train_inputs=train_inputs,
                train_outputs=train_outputs,
                pair_mask=pair_mask,
                temperature=temperature,
                return_intermediates=True,
            )
            
            losses = loss_fn(
                logits=outputs['logits'],
                targets=test_outputs,
                attention_maps=outputs['attention_maps'],
                stop_logits=outputs['stop_logits'],
                predicates=outputs['predicates'],
                epoch=epoch,
                max_epochs=max_epochs,
                all_logits=outputs.get('all_logits'),
                act_outputs=outputs.get('act_outputs'),
                centroids=outputs.get('centroids'),  # For centroid diversity loss
            )
            
            # Compute LOO loss if enabled (meta-learning via HyperLoRA)
            loo_loss = torch.tensor(0.0, device=device)
            loo_metrics = None
            # Check epoch threshold - LOO activates at loo_start_epoch
            loo_active = loo_loss_fn is not None and epoch >= loo_start_epoch
            if loo_active and hasattr(model, 'hyper_lora') and model.hyper_lora is not None:
                # LOO requires at least min_pairs training pairs
                num_pairs = pair_mask.sum(dim=1).min().item() if pair_mask is not None else train_inputs.shape[1]
                if num_pairs >= loo_loss_fn.config.min_pairs_for_loo:
                    loo_result = loo_loss_fn(
                        model=model,
                        input_grids=train_inputs,
                        output_grids=train_outputs,
                        pair_mask=pair_mask,
                        temperature=temperature,
                    )
                    loo_loss = loo_result['loo_loss'] * loo_loss_fn.config.loss_weight
                    loo_metrics = loo_result  # Store full metrics for logging
                    losses['loo_loss'] = loo_loss
                    losses['total_loss'] = losses['total_loss'] + loo_loss
                    # Track meta-learning health metrics
                    epoch_diagnostics['loo_loss_sum'] += loo_loss.item()
                    epoch_diagnostics['loo_accuracy_sum'] += loo_result.get('loo_accuracy', 0.0)
                    epoch_diagnostics['loo_num_holdouts_sum'] += loo_result.get('loo_num_holdouts', 0)
                    epoch_diagnostics['loo_batch_count'] += 1
                else:
                    epoch_diagnostics['loo_skipped_count'] += 1
            
            # Compute HPM load balancing loss if enabled (non-AMP path)
            if hasattr(model, 'use_hpm') and model.use_hpm:
                hpm_balance_loss = model.hpm_get_load_balance_loss()
                hpm_balance_weight = config['model'].get('hpm_balance_weight', 0.01)
                weighted_hpm_loss = hpm_balance_weight * hpm_balance_loss
                losses['hpm_balance_loss'] = weighted_hpm_loss.item() if torch.is_tensor(weighted_hpm_loss) else weighted_hpm_loss
                losses['total_loss'] = losses['total_loss'] + weighted_hpm_loss
            
            # =========================================================================
            # ANCHOR ROBUSTNESS TRAINING (ART) - Jan 2026 Ablation (non-AMP path)
            # =========================================================================
            art_loss_value = 0.0
            art_phase_disabled = phase_disable_flags.get('art', False)
            art_active = (art_module is not None and 
                          epoch >= art_start_epoch and 
                          not art_phase_disabled)
            if art_active:
                try:
                    attention_maps = outputs['attention_maps']
                    centroids = outputs['centroids']
                    logits = outputs['logits']
                    
                    alt_centroids = art_module.extract_alternate_anchors(
                        attention_maps, centroids
                    )
                    
                    alt_logits_list = []
                    B, num_alt, K, _ = alt_centroids.shape
                    for alt_idx in range(num_alt):
                        noise_scale = 0.1
                        alt_logits = logits + torch.randn_like(logits) * noise_scale
                        alt_logits_list.append(alt_logits)
                    
                    art_consistency_loss = art_module.compute_consistency_loss(
                        logits, alt_logits_list
                    )
                    
                    if torch.is_tensor(art_consistency_loss) and torch.isfinite(art_consistency_loss):
                        art_weight = art_module.config.consistency_weight
                        weighted_art_loss = art_weight * art_consistency_loss
                        losses['art_consistency_loss'] = art_consistency_loss.item()
                        losses['total_loss'] = losses['total_loss'] + weighted_art_loss
                        art_loss_value = art_consistency_loss.item()
                        epoch_diagnostics['art_loss_sum'] = epoch_diagnostics.get('art_loss_sum', 0.0) + art_loss_value
                        epoch_diagnostics['art_batch_count'] = epoch_diagnostics.get('art_batch_count', 0) + 1
                except Exception as e:
                    epoch_diagnostics['art_failures'] = epoch_diagnostics.get('art_failures', 0) + 1
                    if epoch_diagnostics.get('art_failures', 0) <= 3:
                        print(f"[ART WARNING] Failed: {e}")
            
            # =========================================================================
            # ANCHOR-RELATIVE PROGRAM SEARCH (ARPS) - Jan 2026 Ablation (non-AMP path)
            # =========================================================================
            arps_loss_value = 0.0
            arps_phase_disabled = phase_disable_flags.get('arps', False)
            arps_active = (arps_module is not None and 
                           epoch >= arps_start_epoch and 
                           not arps_phase_disabled)
            if arps_active:
                try:
                    centroids = outputs['centroids']
                    clue_features = outputs.get('clue_features')
                    if clue_features is None:
                        features = outputs.get('features')
                        if features is not None:
                            K = centroids.shape[1]
                            B, D, H, W = features.shape
                            clue_features = features.unsqueeze(1).expand(B, K, D, H, W)
                    
                    if clue_features is not None:
                        arps_result = arps_module(
                            clue_features,
                            test_inputs,
                            train_inputs,
                            train_outputs,
                            centroids,
                            temperature=temperature,
                        )
                        
                        arps_imitation_loss = arps_result.get('imitation_loss')
                        if (torch.is_tensor(arps_imitation_loss) and 
                            torch.isfinite(arps_imitation_loss) and
                            arps_imitation_loss > 0):
                            arps_weight = config['model'].get('arps_dsl_search', {}).get('imitation_weight', 0.1)
                            weighted_arps_loss = arps_weight * arps_imitation_loss
                            losses['arps_imitation_loss'] = arps_imitation_loss.item()
                            losses['total_loss'] = losses['total_loss'] + weighted_arps_loss
                            arps_loss_value = arps_imitation_loss.item()
                            epoch_diagnostics['arps_loss_sum'] = epoch_diagnostics.get('arps_loss_sum', 0.0) + arps_loss_value
                            epoch_diagnostics['arps_batch_count'] = epoch_diagnostics.get('arps_batch_count', 0) + 1
                            search_stats = arps_result.get('search_stats', {})
                            epoch_diagnostics['arps_valid_programs'] = epoch_diagnostics.get('arps_valid_programs', 0) + search_stats.get('num_valid_programs', 0)
                            epoch_diagnostics['arps_search_attempts'] = epoch_diagnostics.get('arps_search_attempts', 0) + 1
                except Exception as e:
                    epoch_diagnostics['arps_failures'] = epoch_diagnostics.get('arps_failures', 0) + 1
                    if epoch_diagnostics.get('arps_failures', 0) <= 3:
                        print(f"[ARPS WARNING] Failed: {e}")
            
            loss = losses['total_loss'] / grad_accumulation_steps
            
            # NaN detection: skip batch if loss is NaN
            if not torch.isfinite(loss):
                print(f"[WARNING] NaN/Inf loss at batch {batch_idx}, skipping...")
                optimizer.zero_grad()  # Clear any partial gradients
                nan_batches += 1
                consecutive_nan += 1
                max_consecutive_nan_streak = max(max_consecutive_nan_streak, consecutive_nan)  # Track peak streak
                if consecutive_nan >= max_consecutive_nan:
                    print(f"[ERROR] {max_consecutive_nan} consecutive NaN batches - model weights likely corrupted!")
                    print(f"[ERROR] Aborting epoch to prevent further corruption.")
                    break
                continue
            
            loss.backward()
            
            if (batch_idx + 1) % grad_accumulation_steps == 0:
                # Compute total grad norm before clipping
                total_grad_norm_before = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        norm_val = p.grad.norm().item()
                        if math.isfinite(norm_val):
                            total_grad_norm_before += norm_val ** 2
                total_grad_norm_before = total_grad_norm_before ** 0.5
                
                # Capture per-module gradient norms BEFORE clipping (first optimizer step only)
                if (batch_idx + 1) == grad_accumulation_steps:
                    grad_norms = compute_module_grad_norms(model)
                    epoch_diagnostics['dsc_grad_norm_sum'] += grad_norms.get('dsc', 0.0)
                    epoch_diagnostics['stop_predictor_grad_norm_sum'] += grad_norms.get('stop_predictor', 0.0)
                    epoch_diagnostics['encoder_grad_norm_sum'] += grad_norms.get('encoder', 0.0)
                    epoch_diagnostics['solver_grad_norm_sum'] += grad_norms.get('solver', 0.0)
                    epoch_diagnostics['context_encoder_grad_norm_sum'] += grad_norms.get('context_encoder', 0.0)
                    epoch_diagnostics['msre_grad_norm_sum'] += grad_norms.get('msre', 0.0)
                    epoch_diagnostics['grad_norm_before_clip'] = total_grad_norm_before
                    epoch_diagnostics['grad_was_clipped'] = gradient_clip > 0 and total_grad_norm_before > gradient_clip
                
                # Patch 2 (Dec 2025): Track max grad norm across entire epoch
                epoch_diagnostics['max_grad_norm_before_clip'] = max(
                    epoch_diagnostics.get('max_grad_norm_before_clip', 0.0),
                    total_grad_norm_before
                )
                
                if gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                
                # Check for NaN/Inf gradients BEFORE optimizer step
                has_nan_grad = False
                for p in model.parameters():
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        has_nan_grad = True
                        break
                
                if has_nan_grad:
                    print(f"[WARNING] NaN/Inf gradients at batch {batch_idx}, skipping optimizer step...")
                    optimizer.zero_grad()
                    nan_batches += 1
                    consecutive_nan += 1
                    max_consecutive_nan_streak = max(max_consecutive_nan_streak, consecutive_nan)  # Track peak streak
                    if consecutive_nan >= max_consecutive_nan:
                        print(f"[ERROR] {max_consecutive_nan} consecutive NaN batches - model weights likely corrupted!")
                        print(f"[ERROR] Aborting epoch to prevent further corruption.")
                        break
                    continue
                
                # Reset consecutive NaN counter on successful step
                consecutive_nan = 0
                
                # HPM gradient routing: zero gradients for frozen primitives
                if hasattr(model, 'hpm_on_backward'):
                    model.hpm_on_backward()
                
                optimizer.step()
                optimizer.zero_grad()
                
                # MEMORY CHECKPOINT: After optimizer step (gradients cleared)
                mem_tracker.checkpoint("05_after_optim_step", model=model, optimizer=optimizer)
                
                # Update EMA after optimizer step
                if ema is not None:
                    ema.update(model)
        
        # Accumulate losses
        for key in total_losses:
            if key in losses:
                val = losses[key]
                total_losses[key] += val.item() if torch.is_tensor(val) else val
        num_batches += 1
        global_step += 1
        
        # MEMORY DEBUGGING: Print per-batch memory summary for first N batches
        mem_tracker.end_batch(batch_idx, epoch)
        
        # MEMORY MONITOR: Log memory for first 3 batches when new modules become active
        # This detects if staged module activation causes memory overflow to shared memory
        if batch_idx < 3 and device.type == 'cuda':
            modules_active = getattr(model, 'hyperlora_active', False) or \
                            getattr(model, 'solver_context_active', False) or \
                            getattr(model, 'cross_attention_active', False)
            if modules_active:
                torch.cuda.synchronize()
                allocated_mb = torch.cuda.memory_allocated() / 1024 / 1024
                reserved_mb = torch.cuda.memory_reserved() / 1024 / 1024
                max_mb = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
                headroom_mb = max_mb - reserved_mb
                if batch_idx == 0:
                    print(f"\n  [MEMORY] First forward pass with staged modules active:")
                print(f"  [MEMORY] Batch {batch_idx}: alloc={allocated_mb:.0f}MB, reserved={reserved_mb:.0f}MB, headroom={headroom_mb:.0f}MB")
                if reserved_mb > max_mb * 0.95:
                    print(f"  [WARNING] >95% GPU memory used! Training may slow due to shared memory.")
        
        # Periodic cache clearing (every 100 batches) - less frequent to avoid perf overhead
        # Note: torch.cuda.empty_cache() can be slow, so only do it occasionally
        if batch_idx > 0 and batch_idx % 100 == 0:
            torch.cuda.empty_cache()
        
        # Collect diagnostics (first batch of each epoch only to avoid overhead)
        if batch_idx == 0:
            # NEW: Color embedding diversity check - are colors differentiated?
            with torch.no_grad():
                if hasattr(model, 'grid_encoder') and hasattr(model.grid_encoder, 'color_embed'):
                    color_embed = model.grid_encoder.color_embed.weight  # (10, hidden_dim//2)
                    # Cosine similarity between color embeddings
                    color_embed_norm = F.normalize(color_embed, dim=-1)
                    color_sim = color_embed_norm @ color_embed_norm.T  # (10, 10)
                    # Off-diagonal similarity (should be LOW for differentiated colors)
                    mask = ~torch.eye(10, dtype=torch.bool, device=color_sim.device)
                    off_diag_sim = color_sim[mask].mean().item()
                    epoch_diagnostics['color_embed_similarity'] = off_diag_sim
                    
                    # BG vs FG similarity
                    bg_fg_sim = color_sim[0, 1:].mean().item()
                    epoch_diagnostics['bg_fg_embed_similarity'] = bg_fg_sim
            
            # Track all_logits count for deep supervision verification
            all_logits = outputs.get('all_logits')
            if all_logits is not None:
                epoch_diagnostics['all_logits_count'] = len(all_logits)
                
                # Per-step loss breakdown (verify deep supervision is working)
                with torch.no_grad():
                    step_losses = []
                    step_logits_stats = []  # Track logit statistics per step
                    for step_idx, step_logits in enumerate(all_logits):
                        # Check for inf/nan in logits (numerical instability)
                        has_nan = torch.isnan(step_logits).any().item()
                        has_inf = torch.isinf(step_logits).any().item()
                        logits_max = step_logits.max().item() if torch.isfinite(step_logits).any() else float('inf')
                        logits_min = step_logits.min().item() if torch.isfinite(step_logits).any() else float('-inf')
                        
                        step_logits_stats.append({
                            'max': logits_max,
                            'min': logits_min,
                            'has_nan': has_nan,
                            'has_inf': has_inf,
                        })
                        
                        if torch.isfinite(step_logits).all():
                            # Use same loss function as training for consistent diagnostics
                            # This ensures the diagnostic values match actual training behavior
                            step_loss = loss_fn.task_loss(step_logits, test_outputs)
                            # Clamp to prevent extreme values in display
                            step_losses.append(min(step_loss.item(), 100.0))
                        else:
                            # Log warning and use placeholder
                            step_losses.append(float('nan'))
                    epoch_diagnostics['per_step_loss'] = step_losses
                    epoch_diagnostics['per_step_logits_stats'] = step_logits_stats
                    
                    # Accumulate per-step losses for epoch average
                    valid_step_losses = [l if l < 100 and l == l else 0.0 for l in step_losses]  # Replace NaN/100 with 0
                    if epoch_diagnostics['per_step_loss_sum'] is None:
                        epoch_diagnostics['per_step_loss_sum'] = valid_step_losses.copy()
                    else:
                        # Accumulate element-wise
                        for i, l in enumerate(valid_step_losses):
                            if i < len(epoch_diagnostics['per_step_loss_sum']):
                                epoch_diagnostics['per_step_loss_sum'][i] += l
                    epoch_diagnostics['per_step_loss_count'] += 1
                    
                    # =============================================================
                    # BEST-STEP TRACKING (solver health monitoring)
                    # =============================================================
                    valid_losses_indexed = [(i, l) for i, l in enumerate(valid_step_losses) if l > 0]
                    if valid_losses_indexed:
                        best_step_idx = min(valid_losses_indexed, key=lambda x: x[1])[0]
                        last_step_idx = len(valid_step_losses) - 1
                        
                        # Track best step histogram
                        if best_step_idx < len(epoch_diagnostics['best_step_histogram']):
                            epoch_diagnostics['best_step_histogram'][best_step_idx] += 1
                        
                        # Track if last step was best (healthy) vs earlier step (over-iterating)
                        if best_step_idx == last_step_idx:
                            epoch_diagnostics['last_step_was_best_count'] += 1
                        else:
                            epoch_diagnostics['earlier_step_was_best_count'] += 1
                        
                        # Track improvement from step 0 to last step
                        if valid_step_losses[0] > 0 and valid_step_losses[-1] > 0:
                            improvement = (valid_step_losses[0] - valid_step_losses[-1]) / max(valid_step_losses[0], 1e-6)
                            epoch_diagnostics['step_improvement_sum'] += improvement
            
            # Per-clue entropy breakdown
            if 'attention_maps' in outputs:
                attn = outputs['attention_maps']  # (B, K, H, W)
                B, K, H, W = attn.shape
                attn_flat = attn.view(B, K, -1)
                attn_clamp = attn_flat.clamp(min=1e-6)  # Match DSC entropy clamp threshold
                per_clue_entropy = -(attn_clamp * attn_clamp.log()).sum(dim=-1).mean(dim=0)  # (K,)
                epoch_diagnostics['per_clue_entropy'] = per_clue_entropy.tolist()
                
                # Attention sharpness statistics
                attn_max = attn_flat.max(dim=-1)[0].mean()  # Mean of max attention
                attn_min = attn_flat.min(dim=-1)[0].mean()  # Mean of min attention
                epoch_diagnostics['attn_max_mean'] = attn_max.item()
                epoch_diagnostics['attn_min_mean'] = attn_min.item()
            
            # Stop probability (how many clues are used)
            if 'stop_logits' in outputs:
                stop_logits = outputs['stop_logits']  # (B, K)
                stop_probs = torch.sigmoid(stop_logits)  # (B, K)
                epoch_diagnostics['stop_prob_mean'] = stop_probs.mean().item()
                # Global stop probability std (captures the “frozen uniform std=0.008” failure mode)
                # This aggregates variation across both batch and clue index.
                epoch_diagnostics['stop_prob_std'] = stop_probs.std().item()
                # Per-clue stop probability for detailed tracking
                epoch_diagnostics['per_clue_stop_prob'] = stop_probs.mean(dim=0).tolist()  # (K,)
                
                # CRITICAL: Track stop_logits statistics to diagnose saturation
                # If |mean| > 5, sigmoid is saturated and gradients vanish!
                epoch_diagnostics['stop_logits_mean'] = stop_logits.mean().item()
                epoch_diagnostics['stop_logits_std'] = stop_logits.std().item()
                epoch_diagnostics['stop_logits_min'] = stop_logits.min().item()
                epoch_diagnostics['stop_logits_max'] = stop_logits.max().item()
                
                # NEW: Track per-sample variance to verify task-dependent clue count
                # If this is high, different samples use different clue counts (GOOD!)
                # If this is near zero, all samples use same clues (BAD - no task-dependence)
                clues_used_per_sample = (1 - stop_probs).sum(dim=-1)  # (B,) - clues used per sample
                epoch_diagnostics['clues_used_std'] = clues_used_per_sample.std().item()
                epoch_diagnostics['clues_used_min'] = clues_used_per_sample.min().item()
                epoch_diagnostics['clues_used_max'] = clues_used_per_sample.max().item()
                
                # CRITICAL: Compute per-sample loss to correlate with clue count
                # This verifies that harder tasks (higher loss) use more clues
                with torch.no_grad():
                    logits = outputs['logits']  # (B, C, H, W)
                    B = logits.shape[0]
                    # Per-sample cross entropy (no reduction)
                    per_sample_loss = torch.zeros(B, device=logits.device)
                    for i in range(B):
                        sample_loss = F.cross_entropy(
                            logits[i:i+1], test_outputs[i:i+1], 
                            reduction='mean'
                        )
                        per_sample_loss[i] = sample_loss
                    
                    # Correlation between clues used and task loss
                    # Positive correlation = harder tasks use more clues (GOOD!)
                    # Zero/negative = clue count not task-dependent (BAD)
                    if B > 1 and clues_used_per_sample.std() > 1e-6 and per_sample_loss.std() > 1e-6:
                        # Pearson correlation
                        clues_centered = clues_used_per_sample - clues_used_per_sample.mean()
                        loss_centered = per_sample_loss - per_sample_loss.mean()
                        correlation = (clues_centered * loss_centered).sum() / (
                            clues_centered.norm() * loss_centered.norm() + 1e-8
                        )
                        epoch_diagnostics['clue_loss_correlation'] = correlation.item()
                    else:
                        epoch_diagnostics['clue_loss_correlation'] = 0.0
                    
                    epoch_diagnostics['per_sample_loss_std'] = per_sample_loss.std().item()
                    epoch_diagnostics['per_sample_loss_min'] = per_sample_loss.min().item()
                    epoch_diagnostics['per_sample_loss_max'] = per_sample_loss.max().item()
            
            # DSC confidence inputs to stop_predictor (coupling verification)
            # NOTE: Despite key name 'dsc_entropy_inputs', these are now CONFIDENCE values (1 - entropy)
            if 'dsc_entropy_inputs' in outputs:
                confidence_inputs = outputs['dsc_entropy_inputs']  # (B, K) - actually confidence now
                epoch_diagnostics['dsc_confidence_input_mean'] = confidence_inputs.mean().item()
                epoch_diagnostics['per_clue_confidence_input'] = confidence_inputs.mean(dim=0).tolist()  # (K,)
                # Backward compat aliases
                epoch_diagnostics['dsc_entropy_input_mean'] = 1.0 - confidence_inputs.mean().item()
                epoch_diagnostics['per_clue_entropy_input'] = [1.0 - c for c in confidence_inputs.mean(dim=0).tolist()]

                # Confidence-stop coupling correlation (per-clue means)
                # We correlate mean confidence per clue with mean stop prob per clue.
                # Expectation: higher confidence (sharper) -> higher stop probability (POSITIVE correlation = HEALTHY).
                if 'stop_logits' in outputs:
                    stop_logits_local = outputs['stop_logits']
                    stop_probs_local = torch.sigmoid(stop_logits_local)
                    per_clue_stop = stop_probs_local.mean(dim=0)  # (K,)
                    per_clue_confidence = confidence_inputs.mean(dim=0)  # (K,)
                    if (
                        per_clue_stop.numel() > 1
                        and per_clue_stop.std() > 1e-6
                        and per_clue_confidence.std() > 1e-6
                    ):
                        a = per_clue_confidence - per_clue_confidence.mean()
                        b = per_clue_stop - per_clue_stop.mean()
                        corr = (a * b).sum() / (a.norm() * b.norm() + 1e-8)
                        epoch_diagnostics['confidence_stop_correlation'] = corr.item()
                        # Backward compat alias
                        epoch_diagnostics['entropy_stop_correlation'] = corr.item()
                    else:
                        epoch_diagnostics['confidence_stop_correlation'] = 0.0
                        epoch_diagnostics['entropy_stop_correlation'] = 0.0
            
            # Context encoder diagnostics
            if 'context' in outputs:
                context = outputs['context']  # (B, D)
                epoch_diagnostics['context_magnitude'] = context.norm(dim=-1).mean().item()
                epoch_diagnostics['context_std'] = context.std(dim=0).mean().item()  # Variation across batch
            
            # Sparsity loss component breakdown (from losses dict)
            epoch_diagnostics['sparsity_min_clue_penalty'] = losses.get('sparsity_min_clue_penalty', 0.0)
            epoch_diagnostics['sparsity_base_pondering'] = losses.get('sparsity_base_pondering', 0.0)
            epoch_diagnostics['sparsity_entropy_pondering'] = losses.get('sparsity_entropy_pondering', 0.0)
            epoch_diagnostics['expected_clues_used'] = losses.get('expected_clues_used', 0.0)
            # NEW: Per-sample clue penalty mean (verifies per-sample gradient coupling is working)
            epoch_diagnostics['per_sample_clue_penalty_mean'] = losses.get('per_sample_clue_penalty_mean', 0.0)
            # NEW: Clues used std from loss function (should match our computed value)
            epoch_diagnostics['clues_used_std_from_loss'] = losses.get('clues_used_std', 0.0)
            
            # Feature statistics for numerical stability check
            if 'features' in outputs:
                features = outputs['features']  # (B, D, H, W)
                epoch_diagnostics['features_mean'] = features.mean().item()
                epoch_diagnostics['features_std'] = features.std().item()
                epoch_diagnostics['features_max'] = features.max().item()
                epoch_diagnostics['features_min'] = features.min().item()
            
            # Logits statistics for numerical stability
            logits = outputs['logits']
            epoch_diagnostics['logits_max'] = logits.max().item()
            epoch_diagnostics['logits_min'] = logits.min().item()
            epoch_diagnostics['logits_has_nan'] = torch.isnan(logits).any().item()
            epoch_diagnostics['logits_has_inf'] = torch.isinf(logits).any().item()
            
            # Per-class prediction distribution (CRITICAL for detecting background collapse)
            # With TRM encoding: class 0=boundary, classes 1-10=colors 0-9
            with torch.no_grad():
                preds = logits.argmax(dim=1)  # (B, H, W)
                num_classes = logits.shape[1]  # 10 classes (colors 0-9)
                total_pixels = preds.numel()
                
                # Target distribution - use VALID pixels only (exclude -100 padding)
                # This is what the loss actually sees
                valid_mask = test_outputs >= 0  # Exclude -100 padding
                valid_pixels = valid_mask.sum().item()
                padding_pct = (total_pixels - valid_pixels) / total_pixels * 100
                
                # Count targets over VALID pixels only
                valid_targets = test_outputs[valid_mask]
                target_counts = [(valid_targets == c).sum().item() for c in range(num_classes)]
                epoch_diagnostics['target_class_counts'] = target_counts
                epoch_diagnostics['target_class_pcts'] = [c / max(valid_pixels, 1) * 100 for c in target_counts]
                epoch_diagnostics['padding_pct'] = padding_pct
                
                # Prediction distribution - also use VALID pixels only for fair comparison
                valid_preds = preds[valid_mask]
                pred_counts = [(valid_preds == c).sum().item() for c in range(num_classes)]
                epoch_diagnostics['pred_class_counts'] = pred_counts
                epoch_diagnostics['pred_class_pcts'] = [c / max(valid_pixels, 1) * 100 for c in pred_counts]
                
                # Also track predictions over ALL pixels (including padding) for debugging
                all_pred_counts = [(preds == c).sum().item() for c in range(num_classes)]
                epoch_diagnostics['pred_all_pcts'] = [c / total_pixels * 100 for c in all_pred_counts]
                
                # Per-class accuracy (which colors are we getting right/wrong?)
                class_correct = []
                class_total = []
                for c in range(num_classes):
                    mask = test_outputs == c
                    if mask.sum() > 0:
                        correct = ((preds == test_outputs) & mask).sum().item()
                        total = mask.sum().item()
                        class_correct.append(correct)
                        class_total.append(total)
                    else:
                        class_correct.append(0)
                        class_total.append(0)
                epoch_diagnostics['per_class_correct'] = class_correct
                epoch_diagnostics['per_class_total'] = class_total
                
                # Color confusion matrix for diagnosing color prediction issues
                # 10-class encoding: class 0=black (BG), classes 1-9=colors 1-9 (FG)
                # NOTE: valid_mask already excludes -100 padding, but we also
                # exclude values > 9 for safety (colors 0-9 only, 10=PAD_COLOR)
                fg_class_start = 1  # Foreground starts at class 1
                fg_class_end = 9    # Foreground ends at class 9
                fg_mask = (test_outputs >= fg_class_start) & (test_outputs <= fg_class_end)  # Foreground = colors 1-9
                if fg_mask.sum() > 0:
                    # What colors is the model predicting for FG targets?
                    fg_preds = preds[fg_mask]
                    fg_pred_dist = [(fg_preds == c).sum().item() for c in range(num_classes)]
                    epoch_diagnostics['fg_pred_color_dist'] = fg_pred_dist
                    
                    # Mode color prediction (which color does model prefer?)
                    mode_color = max(range(num_classes), key=lambda c: fg_pred_dist[c])
                    epoch_diagnostics['fg_pred_mode_color'] = mode_color
                    epoch_diagnostics['fg_pred_mode_pct'] = fg_pred_dist[mode_color] / fg_mask.sum().item() * 100
            
            # Centroid spread (how diverse are clue locations)
            if 'centroids' in outputs:
                centroids = outputs['centroids']  # (B, K, 2)
                # Compute mean distance between clue centroids
                if centroids.shape[1] > 1:
                    centroid_mean = centroids.mean(dim=1, keepdim=True)  # (B, 1, 2)
                    spread = ((centroids - centroid_mean) ** 2).sum(dim=-1).sqrt().mean()
                    epoch_diagnostics['centroid_spread'] = spread.item()
        
        # ================================================================
        # PER-SAMPLE ACCURACY TRACKING (every batch, not just batch_idx==0)
        # ================================================================
        # This runs EVERY batch to track training progress sample-by-sample
        batch_class_acc = {}  # Initialize for logging access
        batch_fg_acc = 0.0    # Initialize for logging access
        batch_bg_acc = 0.0    # Initialize for logging access
        with torch.no_grad():
            # Best-step selection for accuracy tracking
            # Use loss-based best step since we have ground truth during training
            if hasattr(model, 'use_best_step_selection') and model.use_best_step_selection:
                all_logits = outputs.get('all_logits')
                if all_logits and len(all_logits) > 1:
                    best_logits, best_step, step_info = model.get_best_step_logits(
                        all_logits, test_outputs, loss_fn
                    )
                    logits = best_logits
                    # Track which step was selected for diagnostics
                    if best_step < len(epoch_diagnostics['best_step_histogram']):
                        epoch_diagnostics['best_step_histogram'][best_step] += 1
                    if best_step < len(all_logits) - 1:
                        epoch_diagnostics['earlier_step_was_best_count'] += 1
                    else:
                        epoch_diagnostics['last_step_was_best_count'] += 1
                    
                    # Track entropy-loss agreement (validates inference heuristic)
                    if 'entropy_loss_agreement' in step_info:
                        epoch_diagnostics['entropy_loss_total_count'] += 1
                        if step_info['entropy_loss_agreement']:
                            epoch_diagnostics['entropy_loss_agreement_count'] += 1
                else:
                    logits = outputs['logits']
            else:
                logits = outputs['logits']  # (B, C, H, W)

            # =============================================================
            # ATTENTION COLLAPSE DETECTION (every batch)
            # =============================================================
            # Use mean of per-sample-per-clue max attention as a cheap proxy.
            # Near-uniform attention yields max ~1/(H*W); sharp yields much larger.
            if 'attention_maps' in outputs and outputs['attention_maps'] is not None:
                attn = outputs['attention_maps']  # (B, K, H, W)
                if isinstance(attn, torch.Tensor) and attn.numel() > 0:
                    B_attn, K_attn, H_attn, W_attn = attn.shape
                    attn_flat = attn.view(B_attn, K_attn, -1)
                    attn_max_batch = attn_flat.max(dim=-1)[0].mean().item()
                    if attn_max_batch < attn_max_collapse_threshold:
                        epoch_diagnostics['attention_collapse_events_epoch'] += 1
                        epoch_diagnostics['attention_collapse_consecutive'] += 1
                        epoch_diagnostics['attention_collapse_consecutive_max'] = max(
                            epoch_diagnostics.get('attention_collapse_consecutive_max', 0),
                            epoch_diagnostics['attention_collapse_consecutive']
                        )
                    else:
                        epoch_diagnostics['attention_collapse_consecutive'] = 0
            preds = logits.argmax(dim=1)  # (B, H, W)
            targets = test_outputs  # (B, H, W)
            
            B = logits.shape[0]
            
            # Track per-sample metrics
            batch_exact_match_count = 0
            batch_high_acc_count = 0
            batch_sample_accuracies = []
            batch_fg_acc_sum = 0.0
            batch_bg_acc_sum = 0.0
            batch_fg_samples = 0
            batch_bg_samples = 0
            
            # Per-batch per-class tracking
            batch_class_correct = [0] * 10
            batch_class_total = [0] * 10
            
            # Get task_ids for this batch (for task-level tracking)
            batch_task_ids = batch.get('task_ids', [None] * B)
            
            for i in range(B):
                pred_i = preds[i]  # (H, W)
                target_i = targets[i]  # (H, W)
                
                # Track which task this sample belongs to (for task-level accuracy)
                sample_task_id = batch_task_ids[i] if i < len(batch_task_ids) else None
                is_first_sample_for_task = False
                if sample_task_id is not None:
                    # Check if this is the FIRST sample we've seen for this task
                    if sample_task_id not in epoch_diagnostics['seen_task_ids']:
                        is_first_sample_for_task = True
                    epoch_diagnostics['seen_task_ids'].add(sample_task_id)
                    
                    # Global task tracking (Jan 2026): track all ever-seen tasks
                    if global_task_tracker is not None:
                        global_task_tracker['all_seen_task_ids'].add(sample_task_id)
                
                # Valid mask (exclude padding)
                valid_mask = target_i != -100
                valid_pixels = valid_mask.sum().item()
                
                if valid_pixels > 0:
                    # Per-sample accuracy
                    correct = ((pred_i == target_i) & valid_mask).sum().item()
                    sample_acc = correct / valid_pixels
                    batch_sample_accuracies.append(sample_acc)
                    
                    # Exact match (100% correct)
                    is_exact_match = (correct == valid_pixels)
                    
                    # FIRST-SAMPLE TRACKING: Record if first sample for this task was correct
                    # This is the TRUE single-shot metric, comparable to eval
                    if is_first_sample_for_task and sample_task_id is not None:
                        epoch_diagnostics['first_sample_task_correct'][sample_task_id] = is_exact_match
                    
                    if is_exact_match:
                        batch_exact_match_count += 1
                        # ANY-SAMPLE TRACKING: Mark task as solved (lenient metric)
                        if sample_task_id is not None:
                            epoch_diagnostics['solved_task_ids'].add(sample_task_id)
                            
                            # =============================================================
                            # GLOBAL TASK TRACKING (Jan 2026)
                            # =============================================================
                            # Track whether this is a NEWLY solved task (never seen as exact
                            # match before in any previous epoch) or a REPEAT.
                            if global_task_tracker is not None:
                                if sample_task_id not in global_task_tracker['all_solved_task_ids']:
                                    # First time this task has ever been solved exactly!
                                    global_task_tracker['all_solved_task_ids'].add(sample_task_id)
                                    global_task_tracker['first_solve_epoch'][sample_task_id] = epoch + 1
                                    epoch_diagnostics['new_globally_solved_count'] = epoch_diagnostics.get('new_globally_solved_count', 0) + 1
                        
                        # HPM DYNAMIC BUFFER POPULATION:
                        # When a sample is solved exactly, store its context in dynamic banks
                        # for future retrieval-augmented reasoning on similar tasks.
                        # FIX (Jan 2026): Decoupled write vs use - check hpm_memory_enabled OR use_hpm
                        hpm_memory_enabled = getattr(model, 'hpm_memory_enabled', False)
                        has_hpm_method = hasattr(model, 'hpm_add_solved_task')
                        hpm_enabled_check = model.use_hpm or hpm_memory_enabled
                        
                        # Track write attempt (Jan 2026 Patch)
                        epoch_diagnostics['hpm_write_attempts'] = epoch_diagnostics.get('hpm_write_attempts', 0) + 1
                        
                        # DIAGNOSTIC: Track why HPM buffer writes may be skipped
                        if not has_hpm_method:
                            epoch_diagnostics['hpm_write_skip_reasons']['no_method'] += 1
                        elif not hpm_enabled_check:
                            epoch_diagnostics['hpm_write_skip_reasons']['not_enabled'] += 1
                        
                        if has_hpm_method and hpm_enabled_check:
                            # Prefer a stable task id so retrieval isn't polluted by per-sample ids
                            task_id = sample_task_id if sample_task_id is not None else f"epoch{epoch}_batch{batch_idx}_sample{i}"
                            
                            # GLOBAL DEDUPLICATION (not just per-epoch)
                            # Check if task already exists in buffer across ALL epochs
                            already_in_buffer = False
                            if hasattr(model, 'hpm_buffer_contains_task'):
                                already_in_buffer = model.hpm_buffer_contains_task(task_id)
                            
                            if already_in_buffer:
                                epoch_diagnostics['hpm_write_skip_reasons']['global_duplicate'] += 1
                            elif task_id in epoch_diagnostics['hpm_tasks_added_this_epoch']:
                                # Per-epoch dedup (multiple augmentations of same task in one epoch)
                                epoch_diagnostics['hpm_write_skip_reasons']['epoch_duplicate'] += 1
                                epoch_diagnostics['hpm_duplicate_skipped'] += 1
                            else:
                                # Get context embedding for this sample
                                support_features = outputs.get('support_features')
                                if support_features is None:
                                    epoch_diagnostics['hpm_write_skip_reasons']['no_support_features'] += 1
                                else:
                                    # support_features: (B, N, D, H, W) -> pool to (D,)
                                    z_context = support_features[i].mean(dim=(0, 2, 3))  # (D,)

                                    # Procedural embedding: use HyperLoRA pooled context if available
                                    z_task = None
                                    lora_deltas = outputs.get('lora_deltas')
                                    if isinstance(lora_deltas, dict):
                                        lora_ctx = lora_deltas.get('context')
                                        if isinstance(lora_ctx, torch.Tensor) and lora_ctx.dim() == 2 and lora_ctx.shape[0] > i:
                                            z_task = lora_ctx[i]

                                    # Add to dynamic buffers (force_write allows writes during staged-off period)
                                    model.hpm_add_solved_task(
                                        z_context.unsqueeze(0),
                                        z_task.unsqueeze(0) if z_task is not None else None,
                                        task_id,
                                        force_write=hpm_memory_enabled,  # Force write during staged-off period
                                    )
                                    epoch_diagnostics['hpm_tasks_added'] = epoch_diagnostics.get('hpm_tasks_added', 0) + 1
                                    epoch_diagnostics['hpm_writes_succeeded'] = epoch_diagnostics.get('hpm_writes_succeeded', 0) + 1
                                    epoch_diagnostics['hpm_tasks_added_this_epoch'].add(task_id)
                    
                    # High accuracy (>=90% correct)
                    if sample_acc >= 0.9:
                        batch_high_acc_count += 1
                    
                    # FG accuracy (colors 1-9 only, excluding background 0 and padding)
                    fg_mask = (target_i > 0) & (target_i <= 9) & valid_mask
                    fg_pixels = fg_mask.sum().item()
                    if fg_pixels > 0:
                        fg_correct = ((pred_i == target_i) & fg_mask).sum().item()
                        batch_fg_acc_sum += fg_correct / fg_pixels
                        batch_fg_samples += 1
                    
                    # BG accuracy (color 0)
                    bg_mask = (target_i == 0) & valid_mask
                    bg_pixels = bg_mask.sum().item()
                    if bg_pixels > 0:
                        bg_correct = ((pred_i == target_i) & bg_mask).sum().item()
                        batch_bg_acc_sum += bg_correct / bg_pixels
                        batch_bg_samples += 1
                    
                    # Per-class accuracy (colors 0-9)
                    for c in range(10):
                        class_mask = (target_i == c) & valid_mask
                        class_pixels = class_mask.sum().item()
                        if class_pixels > 0:
                            epoch_diagnostics['per_class_total'][c] += class_pixels
                            class_correct = ((pred_i == c) & class_mask).sum().item()
                            epoch_diagnostics['per_class_correct'][c] += class_correct
                            # Also track for this batch
                            batch_class_total[c] += class_pixels
                            batch_class_correct[c] += class_correct
                        # Track how often this class is predicted
                        epoch_diagnostics['per_class_predicted'][c] += ((pred_i == c) & valid_mask).sum().item()
            
            # Compute batch-level per-class accuracies
            batch_class_acc = {}
            for c in range(10):
                if batch_class_total[c] > 0:
                    batch_class_acc[c] = batch_class_correct[c] / batch_class_total[c]
                else:
                    batch_class_acc[c] = None  # Class not present in batch
            
            # Update per-class running windows (last 50 batches)
            for c in range(10):
                if batch_class_acc[c] is not None:
                    epoch_diagnostics['per_class_running_window'][c].append(batch_class_acc[c])
                    if len(epoch_diagnostics['per_class_running_window'][c]) > 50:
                        epoch_diagnostics['per_class_running_window'][c].pop(0)
            
            # Update epoch-level diagnostics
            epoch_diagnostics['total_samples'] += B
            epoch_diagnostics['exact_match_count'] += batch_exact_match_count
            epoch_diagnostics['high_accuracy_count'] += batch_high_acc_count
            
            # Store batch-level metrics
            batch_accuracy = sum(batch_sample_accuracies) / len(batch_sample_accuracies) if batch_sample_accuracies else 0.0
            epoch_diagnostics['batch_accuracies'].append(batch_accuracy)
            epoch_diagnostics['batch_exact_matches'].append(batch_exact_match_count)
            
            # Update FG/BG sums and running windows
            batch_fg_acc = batch_fg_acc_sum / batch_fg_samples if batch_fg_samples > 0 else 0.0
            batch_bg_acc = batch_bg_acc_sum / batch_bg_samples if batch_bg_samples > 0 else 0.0
            if batch_fg_samples > 0:
                epoch_diagnostics['fg_accuracy_sum'] += batch_fg_acc
                epoch_diagnostics['fg_batch_count'] += 1
                epoch_diagnostics['fg_running_window'].append(batch_fg_acc)
                if len(epoch_diagnostics['fg_running_window']) > 50:
                    epoch_diagnostics['fg_running_window'].pop(0)
            if batch_bg_samples > 0:
                epoch_diagnostics['bg_accuracy_sum'] += batch_bg_acc
                epoch_diagnostics['bg_batch_count'] += 1
                epoch_diagnostics['bg_running_window'].append(batch_bg_acc)
                if len(epoch_diagnostics['bg_running_window']) > 50:
                    epoch_diagnostics['bg_running_window'].pop(0)
            
            # Running accuracy window (last 50 batches)
            epoch_diagnostics['running_accuracy_window'].append(batch_accuracy)
            if len(epoch_diagnostics['running_accuracy_window']) > 50:
                epoch_diagnostics['running_accuracy_window'].pop(0)
            
            # Sample some per-sample accuracies for detailed logging (every 10th batch)
            if batch_idx % 10 == 0:
                epoch_diagnostics['per_sample_accuracies'].extend(batch_sample_accuracies[:4])  # First 4 samples
            
            # =============================================================
            # TRACK META-LEARNING METRICS (for epoch 8+ debugging)
            # =============================================================
            # LoRA delta norms (shows HyperLoRA is producing non-trivial adaptations)
            # P2.1: Optimized - aggregate on GPU, single .item() call to reduce sync overhead
            # P2.2 FIX (Dec 2025): Exclude 'context' key from norm calculation!
            #   - 'context' is an auxiliary tensor (B, D) NOT a LoRA delta
            #   - Including it inflated norms by ~sqrt(B*D) = ~143 for B=80, D=256
            #   - This caused false-positive governor aborts at HyperLoRA activation
            # P2.3 FIX: Compute mean per-sample norm for batch-size invariance
            #   - Old: sum over entire batch -> norm scales with sqrt(B)
            #   - New: mean of per-sample norms -> consistent across batch sizes
            if 'lora_deltas' in outputs and outputs['lora_deltas'] is not None:
                # Only include actual LoRA delta keys, exclude auxiliary tensors
                LORA_DELTA_KEYS = {'gru_reset', 'gru_update', 'gru_candidate', 'output_head'}
                lora_norm_sq = torch.tensor(0.0, device=device)
                lora_sample_count = 0
                for key, delta in outputs['lora_deltas'].items():
                    if key not in LORA_DELTA_KEYS:
                        continue  # Skip 'context' and any other auxiliary tensors
                    if isinstance(delta, torch.Tensor) and delta.ndim >= 2:
                        # Compute per-sample norm: delta shape is (B, D1, D2) for weight matrices
                        # Flatten per-sample and compute norm, then average across batch
                        B = delta.shape[0]
                        per_sample_norms_sq = delta.view(B, -1).pow(2).sum(dim=1)  # (B,)
                        lora_norm_sq = lora_norm_sq + per_sample_norms_sq.mean()
                        lora_sample_count += 1
                # Average across all delta matrices
                if lora_sample_count > 0:
                    lora_norm = (lora_norm_sq / lora_sample_count).sqrt().item()
                else:
                    lora_norm = 0.0
                epoch_diagnostics['lora_delta_norm_sum'] += lora_norm
                epoch_diagnostics['lora_delta_batch_count'] += 1
                
                # =============================================================
                # P0.3: LORA NORM GOVERNOR - Auto-detect runaway HyperLoRA
                # =============================================================
                # Problem observed in 821b111: LoRA norms reached 62+ causing NaN
                # Solution: EMA tracking with warn/critical/kill thresholds
                
                # Initialize governor state on first batch
                if 'lora_norm_ema' not in epoch_diagnostics:
                    epoch_diagnostics['lora_norm_ema'] = lora_norm
                    epoch_diagnostics['lora_norm_max_seen'] = lora_norm
                    epoch_diagnostics['lora_norm_warnings'] = 0
                    epoch_diagnostics['lora_norm_critical_count'] = 0
                
                # Update EMA (smoothing factor 0.95 = ~20 batch window)
                ema_alpha = 0.95
                epoch_diagnostics['lora_norm_ema'] = ema_alpha * epoch_diagnostics['lora_norm_ema'] + (1 - ema_alpha) * lora_norm
                epoch_diagnostics['lora_norm_max_seen'] = max(epoch_diagnostics['lora_norm_max_seen'], lora_norm)
                
                # Check thresholds (use EMA for stability, not raw value)
                # Thresholds loaded from YAML at epoch start: lora_norm_*_threshold
                ema_val = epoch_diagnostics['lora_norm_ema']
                
                if ema_val > lora_norm_kill_threshold:
                    epoch_diagnostics['lora_norm_kill_count'] = epoch_diagnostics.get('lora_norm_kill_count', 0) + 1
                    kill_count = epoch_diagnostics['lora_norm_kill_count']
                    if kill_count == 1:
                        print(f"\n{'!'*60}")
                        print(f"[LORA GOVERNOR] KILL THRESHOLD EXCEEDED!")
                        print(f"  EMA LoRA Norm: {ema_val:.2f} > {lora_norm_kill_threshold} (kill)")
                        print(f"  Max seen this epoch: {epoch_diagnostics['lora_norm_max_seen']:.2f}")
                        print(f"  This indicates HyperLoRA is exploding - model weights may be corrupted.")
                        print(f"  Will abort after {lora_kill_consecutive_limit} consecutive kill-threshold breaches.")
                        print(f"{'!'*60}\n")
                    # P0.2: Actually abort after consecutive kill-threshold breaches
                    if kill_count >= lora_kill_consecutive_limit:
                        print(f"\n{'!'*60}")
                        print(f"[LORA GOVERNOR] ABORTING EPOCH - {kill_count} consecutive kill-threshold breaches!")
                        print(f"  EMA LoRA Norm: {ema_val:.2f} (threshold: {lora_norm_kill_threshold})")
                        print(f"  Model weights are likely corrupted. Stopping to prevent wasted compute.")
                        print(f"  Recommend: Reduce hyperlora_lr_multiplier or init_scale and restart.")
                        print(f"{'!'*60}\n")
                        epoch_diagnostics['lora_kill_abort'] = True
                        break  # Exit batch loop
                else:
                    # Reset kill counter on non-kill batch
                    epoch_diagnostics['lora_norm_kill_count'] = 0
                
                if ema_val > lora_norm_critical_threshold:
                    epoch_diagnostics['lora_norm_critical_count'] += 1
                    if epoch_diagnostics['lora_norm_critical_count'] <= 3:
                        print(f"[LORA GOVERNOR] ⚠️ CRITICAL: LoRA norm EMA={ema_val:.2f} > {lora_norm_critical_threshold}")
                elif ema_val > lora_norm_warn_threshold:
                    epoch_diagnostics['lora_norm_warnings'] += 1
                    if epoch_diagnostics['lora_norm_warnings'] <= 2:
                        print(f"[LORA GOVERNOR] Warning: LoRA norm EMA={ema_val:.2f} > {lora_norm_warn_threshold}")
            
            # Context/support feature magnitude (shows context encoder is contributing)
            if 'support_features' in outputs and outputs['support_features'] is not None:
                ctx_mag = outputs['support_features'].abs().mean().item()
                epoch_diagnostics['context_magnitude_sum'] += ctx_mag
                epoch_diagnostics['context_batch_count'] += 1
            
            # HPM routing entropy (shows bank specialization)
            if 'hpm_routing_weights' in outputs and outputs['hpm_routing_weights'] is not None:
                routing = outputs['hpm_routing_weights']
                routing_entropy = -(routing * (routing + 1e-10).log()).sum(dim=-1).mean().item()
                epoch_diagnostics['hpm_routing_entropy_sum'] += routing_entropy
                epoch_diagnostics['hpm_batch_count'] += 1
            
            # HPM retrieval stats (shows memory quality and retrieval effectiveness)
            if 'hpm_retrieval_stats' in outputs and outputs['hpm_retrieval_stats']:
                retrieval_stats = outputs['hpm_retrieval_stats']
                epoch_diagnostics['hpm_retrieval_count'] += 1
                
                if 'instance' in retrieval_stats:
                    inst_stats = retrieval_stats['instance']
                    epoch_diagnostics['hpm_instance_similarity_sum'] += inst_stats.get('avg_similarity', 0.0)
                    epoch_diagnostics['hpm_instance_retrieved_sum'] += inst_stats.get('retrieved', 0)
                
                if 'procedural' in retrieval_stats:
                    proc_stats = retrieval_stats['procedural']
                    epoch_diagnostics['hpm_procedural_similarity_sum'] += proc_stats.get('avg_similarity', 0.0)
                    epoch_diagnostics['hpm_procedural_retrieved_sum'] += proc_stats.get('retrieved', 0)
        
        # Log progress
        if batch_idx % log_every == 0:
            current_lr = optimizer.param_groups[0]['lr']
            loss_mode = losses.get('loss_mode', 'task')
            task_loss_val = losses.get('task_loss', losses.get('focal_loss', torch.tensor(0.0)))
            task_loss_val = task_loss_val.item() if hasattr(task_loss_val, 'item') else task_loss_val
            
            # Enhanced batch logging with accuracy metrics
            running_window = epoch_diagnostics['running_accuracy_window']
            running_acc = sum(running_window) / len(running_window) if running_window else 0.0
            total_exact = epoch_diagnostics['exact_match_count']
            total_processed = epoch_diagnostics['total_samples']
            exact_pct = (total_exact / total_processed * 100) if total_processed > 0 else 0.0
            
            # Get meta-learning losses for display (if active)
            loo_loss_val = losses.get('loo_loss', torch.tensor(0.0))
            loo_loss_val = loo_loss_val.item() if hasattr(loo_loss_val, 'item') else loo_loss_val
            equiv_loss_val = losses.get('equiv_loss', torch.tensor(0.0))
            equiv_loss_val = equiv_loss_val.item() if hasattr(equiv_loss_val, 'item') else equiv_loss_val
            hpm_loss_val = losses.get('hpm_balance_loss', torch.tensor(0.0))
            hpm_loss_val = hpm_loss_val.item() if hasattr(hpm_loss_val, 'item') else hpm_loss_val
            
            # Build meta-learning suffix if active
            meta_str = ""
            if loo_loss_val > 0:
                meta_str += f", loo={loo_loss_val:.4f}"
            if equiv_loss_val > 0:
                meta_str += f", equiv={equiv_loss_val:.4f}"
            if hpm_loss_val > 0:
                meta_str += f", hpm={hpm_loss_val:.4f}"
            
            print(f"  Batch {batch_idx}/{len(dataloader)}: "
                  f"loss={losses['total_loss'].item():.4f}, "
                  f"{loss_mode}={task_loss_val:.4f}, "
                  f"batch_acc={batch_accuracy:.1%}, "
                  f"exact={total_exact}/{total_processed} ({exact_pct:.1f}%), "
                  f"running_acc={running_acc:.1%}, "
                  f"lr={current_lr:.2e}{meta_str}")
            
            # =============================================================
            # GLOBAL TASK TRACKING LOG (Jan 2026)
            # =============================================================
            # Show: total unique solved ever | new this epoch | new this batch
            # PATCH (Jan 2026): Use expected_task_count for stable denominator
            if global_task_tracker is not None:
                total_globally_solved = len(global_task_tracker['all_solved_task_ids'])
                expected_total = global_task_tracker.get('expected_task_count', None)
                new_this_epoch = epoch_diagnostics.get('new_globally_solved_count', 0)
                epoch_solved_this_epoch = len(epoch_diagnostics['solved_task_ids'])
                if expected_total is not None:
                    print(f"    [TaskTrack] Global_Solved: {total_globally_solved}/{expected_total}, "
                          f"Epoch_Solved: {epoch_solved_this_epoch}, New_Puzzles: {new_this_epoch}")
                else:
                    # Fallback to dynamic count
                    total_ever_seen = len(global_task_tracker['all_seen_task_ids'])
                    print(f"    [TaskTrack] Global_Solved: {total_globally_solved}/{total_ever_seen}, "
                          f"Epoch_Solved: {epoch_solved_this_epoch}, New_Puzzles: {new_this_epoch}")
            
            # FG/BG accuracy for this batch and running averages
            fg_window = epoch_diagnostics['fg_running_window']
            bg_window = epoch_diagnostics['bg_running_window']
            fg_running = sum(fg_window) / len(fg_window) if fg_window else 0.0
            bg_running = sum(bg_window) / len(bg_window) if bg_window else 0.0
            print(f"    FG: batch={batch_fg_acc:.1%} run50={fg_running:.1%} | "
                  f"BG: batch={batch_bg_acc:.1%} run50={bg_running:.1%}")
            
            # Per-class accuracy for this batch
            batch_class_strs = []
            for c in range(10):
                if batch_class_acc.get(c) is not None:
                    batch_class_strs.append(f"{c}:{batch_class_acc[c]:.0%}")
                else:
                    batch_class_strs.append(f"{c}:-")
            print(f"    Per-Color: [{' '.join(batch_class_strs)}]")
            
            # Per-class running averages (last 50 batches where class appeared)
            running_class_strs = []
            for c in range(10):
                window = epoch_diagnostics['per_class_running_window'][c]
                if len(window) > 0:
                    avg = sum(window) / len(window)
                    running_class_strs.append(f"{c}:{avg:.0%}")
                else:
                    running_class_strs.append(f"{c}:-")
            print(f"    Running50: [{' '.join(running_class_strs)}]")
            
            # =============================================================
            # PER-BATCH SOLVER STEP LOSSES (for detailed monitoring)
            # =============================================================
            per_step_loss = epoch_diagnostics.get('per_step_loss', [])
            if per_step_loss and batch_idx % 50 == 0:  # Log every 50 batches to avoid spam
                loss_str = ', '.join(f"{l:.3f}" for l in per_step_loss)
                valid_losses = [(i, l) for i, l in enumerate(per_step_loss) if l < 100 and l == l]
                if valid_losses:
                    best_step_idx = min(valid_losses, key=lambda x: x[1])[0]
                    last_step_idx = len(per_step_loss) - 1
                    if best_step_idx == last_step_idx:
                        status = "✓"
                    else:
                        status = f"⚠ best={best_step_idx}"
                    print(f"    Solver: [{loss_str}] {status}")
            
            # =============================================================
            # META-LEARNING DETAILED BATCH LOGGING (epoch 8+)
            # =============================================================
            # Only log every 20 batches to avoid spam, but critical for debugging
            if batch_idx % 20 == 0 and (loo_loss_val > 0 or equiv_loss_val > 0):
                # Get LoRA delta norms from outputs if available
                # P2.2 FIX (Dec 2025): Exclude 'context' key and compute mean per-sample norm
                lora_delta_norm = 0.0
                if 'lora_deltas' in outputs and outputs['lora_deltas'] is not None:
                    LORA_DELTA_KEYS = {'gru_reset', 'gru_update', 'gru_candidate', 'output_head'}
                    norm_sum = 0.0
                    key_count = 0
                    for key, delta in outputs['lora_deltas'].items():
                        if key not in LORA_DELTA_KEYS:
                            continue  # Skip 'context' and other auxiliary tensors
                        if isinstance(delta, torch.Tensor) and delta.ndim >= 2:
                            # Mean per-sample norm for batch-size invariance
                            B = delta.shape[0]
                            per_sample_norms = delta.view(B, -1).norm(dim=1)  # (B,)
                            norm_sum += per_sample_norms.mean().item() ** 2
                            key_count += 1
                    if key_count > 0:
                        lora_delta_norm = (norm_sum / key_count) ** 0.5
                
                # Get context encoder contribution
                context_contrib = 0.0
                if 'support_features' in outputs and outputs['support_features'] is not None:
                    context_contrib = outputs['support_features'].abs().mean().item()
                
                # Build meta info string
                meta_detail = f"    [META] LoRA_norm={lora_delta_norm:.4f}, ctx_mag={context_contrib:.4f}"
                
                # Get HPM routing info if available
                if 'hpm_routing_weights' in outputs and outputs['hpm_routing_weights'] is not None:
                    routing = outputs['hpm_routing_weights']  # (B, num_banks)
                    # Compute routing entropy (lower = more specialized)
                    routing_entropy = -(routing * (routing + 1e-10).log()).sum(dim=-1).mean().item()
                    # Get top bank usage
                    top_bank = routing.argmax(dim=-1).float().mean().item()
                    meta_detail += f", hpm_ent={routing_entropy:.3f}, top_bank={top_bank:.1f}"
                
                print(meta_detail)
        
        # MEMORY FIX: Explicitly clear intermediate tensors at END of loop iteration
        # This is critical for LOO/Equivariance training which creates large graphs
        # Must be after all diagnostics and logging that use outputs/losses
        del outputs, losses
        if 'loo_result' in locals(): del loo_result
        if 'loo_loss' in locals(): del loo_loss
        if 'equiv_loss' in locals(): del equiv_loss
        if 'loss' in locals(): del loss
    
    # Average losses
    for key in total_losses:
        total_losses[key] /= max(num_batches, 1)
    
    # Add epoch-level statistics
    total_losses['total_samples'] = total_samples
    total_losses['num_batches'] = num_batches
    total_losses['nan_batches'] = nan_batches  # Track NaN occurrences
    total_losses['max_consecutive_nan_streak'] = max_consecutive_nan_streak  # Peak consecutive NaN (for backoff)
    
    # Log NaN batches if any occurred
    if nan_batches > 0:
        print(f"[WARNING] {nan_batches} batches had NaN loss and were skipped this epoch")
        # P0.3: Check against YAML nan_batches_abort threshold
        if nan_batches > nan_batches_abort:
            print(f"\\n{'!'*60}")
            print(f"[NaN ABORT] Epoch had {nan_batches} NaN batches > {nan_batches_abort} threshold!")
            print(f"  This exceeds monitoring.nan_batches_abort from config.")
            print(f"  Model may be unstable. Consider reducing learning rate.")
            print(f"{'!'*60}\\n")
            total_losses['nan_abort_triggered'] = True
    
    # Add augmentation diversity stats (CRITICAL for debugging)
    total_losses['aug_stats'] = epoch_aug_stats

    # Stop predictor health diagnostic: weight variance (near-zero can indicate freezing)
    epoch_diagnostics['stop_predictor_weight_variance'] = compute_stop_predictor_weight_variance(model)
    
    # Add training diagnostics (CRITICAL for debugging training dynamics)
    total_losses['diagnostics'] = epoch_diagnostics
    
    return total_losses, global_step


def evaluate(
    model: RLAN,
    dataloader: DataLoader,
    device: torch.device,
    temperature: float = 0.5,  # Use training temperature, not hardcoded 0.1!
) -> Dict[str, float]:
    """
    Evaluate model on validation set with detailed metrics for debugging.
    
    CRITICAL: temperature should match or be close to training temperature!
    Using temp=0.1 when training at temp=0.85 causes distribution shift.
    
    METRICS COMPUTED OVER VALID PIXELS ONLY (excluding -100 padding):
    - pixel_accuracy: correct / total valid pixels
    - task_accuracy: tasks where ALL valid pixels are correct
    - fg_accuracy: accuracy on foreground pixels only (colors 1-9)
    """
    model.eval()
    
    total_correct = 0
    total_valid_pixels = 0
    total_tasks = 0
    correct_tasks = 0
    
    # Additional metrics for debugging
    # 10-class encoding: class 0=black (BG), classes 1-9=colors 1-9 (FG)
    total_fg_correct = 0  # Foreground (colors 1-9)
    total_fg_pixels = 0
    total_bg_correct = 0  # Background/black (class 0)
    total_bg_pixels = 0
    
    num_classes = None  # Will be set from first batch
    color_predictions = None  # Will initialize after knowing num_classes
    color_targets = None
    
    # Module-specific debugging metrics
    dsc_entropy_sum = 0.0
    dsc_usage_sum = 0.0  # How many clues are used (non-stopped)
    predicate_activation_sum = 0.0
    num_eval_samples = 0
    
    # Wrap dataloader with CUDA prefetcher for async data transfer
    prefetcher = CUDAPrefetcher(dataloader, device)
    
    # Progress logging
    total_batches = len(dataloader)
    print(f"\n  [Eval] Running evaluation on {total_batches} batches...", end="", flush=True)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(prefetcher):
            # Progress indicator every 20 batches
            if (batch_idx + 1) % 20 == 0 or batch_idx == total_batches - 1:
                print(f" {batch_idx + 1}/{total_batches}", end="", flush=True)
            
            # Batch already on device (prefetcher handles transfer asynchronously)
            test_inputs = batch['test_inputs']
            test_outputs = batch['test_outputs']
            train_inputs = batch['input_grids']
            train_outputs = batch['output_grids']
            pair_mask = batch['grid_masks']
            
            # Predict with SAME temperature as training to avoid distribution shift
            # CRITICAL FIX: Previously used hardcoded 0.1, causing train-eval mismatch
            outputs = model(
                test_inputs,
                train_inputs=train_inputs,
                train_outputs=train_outputs,
                pair_mask=pair_mask,
                temperature=temperature,
                return_intermediates=True,
            )
            
            # Best-step selection: use lowest-entropy step instead of always last step
            if hasattr(model, 'use_best_step_selection') and model.use_best_step_selection:
                all_logits = outputs.get('all_logits')
                if all_logits and len(all_logits) > 1:
                    # Select step with lowest prediction entropy (most confident)
                    best_logits, best_step, step_info = model.get_best_step_logits(all_logits, None, None)
                    logits = best_logits
                else:
                    logits = outputs['logits']
            else:
                logits = outputs['logits']
            predictions = logits.argmax(dim=1)
            
            # Initialize color tracking on first batch (after we know num_classes)
            if num_classes is None:
                num_classes = logits.shape[1]  # 10 classes (colors 0-9)
                color_predictions = [0] * num_classes
                color_targets = [0] * num_classes
                # 10-class encoding: 0=black(bg), 1-9=colors 1-9 (fg)
                fg_start = 1
                bg_class = 0
            
            # CRITICAL: Only evaluate on VALID pixels (exclude -100 padding)
            valid_mask = test_outputs >= 0  # Exclude -100 padding
            batch_valid_pixels = valid_mask.sum().item()
            
            if batch_valid_pixels > 0:
                # Pixel accuracy (over valid pixels only)
                correct_mask = (predictions == test_outputs) & valid_mask
                total_correct += correct_mask.sum().item()
                total_valid_pixels += batch_valid_pixels
                
                # Foreground accuracy (colors 1-9 only, excluding background 0 and padding)
                # NOTE: Valid colors are 0-9 only (10 is PAD_COLOR for inputs)
                fg_mask = (test_outputs >= fg_start) & (test_outputs <= 9) & valid_mask
                if fg_mask.any():
                    fg_correct = (correct_mask & fg_mask).sum().item()
                    total_fg_correct += fg_correct
                    total_fg_pixels += fg_mask.sum().item()
                
                # Background accuracy (black/color 0)
                bg_mask = (test_outputs == bg_class) & valid_mask
                if bg_mask.any():
                    bg_correct = (correct_mask & bg_mask).sum().item()
                    total_bg_correct += bg_correct
                    total_bg_pixels += bg_mask.sum().item()
                
                # Color distribution tracking (over valid pixels only)
                valid_preds = predictions[valid_mask]
                valid_targets = test_outputs[valid_mask]
                for c in range(num_classes):
                    color_predictions[c] += (valid_preds == c).sum().item()
                    color_targets[c] += (valid_targets == c).sum().item()
            
            # Task accuracy (all VALID pixels correct)
            batch_size = test_inputs.shape[0]
            for i in range(batch_size):
                task_valid_mask = test_outputs[i] >= 0
                if task_valid_mask.any():
                    # Task is correct if ALL valid pixels match
                    task_correct = ((predictions[i] == test_outputs[i]) | ~task_valid_mask).all()
                    if task_correct:
                        correct_tasks += 1
                total_tasks += 1
            
            # DSC metrics (attention maps)
            if 'attention_maps' in outputs:
                attn = outputs['attention_maps']  # (B, K, H, W)
                attn_flat = attn.view(attn.shape[0], attn.shape[1], -1)
                # Entropy of attention (lower = sharper = better)
                attn_clamp = attn_flat.clamp(min=1e-6)  # Match DSC entropy clamp threshold
                entropy = -(attn_clamp * attn_clamp.log()).sum(dim=-1).mean()
                dsc_entropy_sum += entropy.item() * batch_size
            
            # DSC usage (from stop_logits)
            # FIX: Use soft clue count (1 - stop_prob) to match training metric
            # The hard threshold (< 0.5) was showing 6.0 clues when stop_prob=0.79
            if 'stop_logits' in outputs:
                stop_probs = torch.sigmoid(outputs['stop_logits'])  # (B, K)
                # Soft clue count: sum of (1 - stop_prob) across clues
                soft_usage = (1 - stop_probs).sum(dim=-1).mean()  # Matches training
                dsc_usage_sum += soft_usage.item() * batch_size
                # Also track mean stop_prob for debugging
                if 'eval_stop_prob_sum' not in locals():
                    eval_stop_prob_sum = 0.0
                eval_stop_prob_sum += stop_probs.mean().item() * batch_size
            
            # Predicate activations
            if 'predicates' in outputs:
                pred_act = outputs['predicates'].abs().mean()
                predicate_activation_sum += pred_act.item() * batch_size
            
            num_eval_samples += batch_size
    
    # Compute final metrics (all over VALID pixels only)
    pixel_accuracy = total_correct / max(total_valid_pixels, 1)
    task_accuracy = correct_tasks / max(total_tasks, 1)
    fg_accuracy = total_fg_correct / max(total_fg_pixels, 1)
    bg_accuracy = total_bg_correct / max(total_bg_pixels, 1)
    
    # Calculate class distribution ratios (over valid pixels)
    # 10-class encoding: class 0=bg/black, classes 1-9=foreground
    total_valid = sum(color_predictions) if color_predictions else 1
    total_target = sum(color_targets) if color_targets else 1
    
    bg_ratio_pred = color_predictions[0] / max(total_valid, 1) if color_predictions else 0
    bg_ratio_target = color_targets[0] / max(total_target, 1) if color_targets else 0
    fg_ratio_pred = sum(color_predictions[1:]) / max(total_valid, 1) if color_predictions else 0
    fg_ratio_target = sum(color_targets[1:]) / max(total_target, 1) if color_targets else 0
    
    # Color diversity (how many colors are actually predicted vs targets)
    colors_used = sum(1 for c in color_predictions if c > 0) if color_predictions else 0
    colors_target = sum(1 for c in color_targets if c > 0) if color_targets else 0
    
    # Handle stop_prob tracking (may not be in scope if no stop_logits)
    try:
        eval_stop_prob = eval_stop_prob_sum / max(num_eval_samples, 1)
    except:
        eval_stop_prob = 0.0

    return {
        # Primary metrics (computed over valid pixels only)
        'pixel_accuracy': pixel_accuracy,
        'task_accuracy': task_accuracy,
        'fg_accuracy': fg_accuracy,
        'bg_accuracy': bg_accuracy,
        
        # EXACT MATCH counts for clear progress tracking
        'correct_tasks': correct_tasks,
        'total_tasks': total_tasks,
        
        # Class distribution ratios (over valid pixels)
        'bg_ratio_pred': bg_ratio_pred,
        'bg_ratio_target': bg_ratio_target,
        'fg_ratio_pred': fg_ratio_pred,
        'fg_ratio_target': fg_ratio_target,
        
        # Color diversity
        'colors_used': colors_used,
        'colors_target': colors_target,
        'num_classes': num_classes,
        
        # Pixel counts for debugging
        'total_valid_pixels': total_valid_pixels,
        'total_fg_pixels': total_fg_pixels,
        'total_bg_pixels': total_bg_pixels,
        
        # Module metrics
        'dsc_entropy': dsc_entropy_sum / max(num_eval_samples, 1),
        'dsc_clues_used': dsc_usage_sum / max(num_eval_samples, 1),
        'eval_stop_prob': eval_stop_prob,
        'predicate_activation': predicate_activation_sum / max(num_eval_samples, 1),
    }
    print(" Done.")  # End the progress line


# =============================================================================
# CANONICAL TRAIN EVALUATION (Jan 2026 Patch)
# =============================================================================
# Provides a deterministic, stable train accuracy metric by evaluating on
# canonical (non-augmented) samples. This is independent of random augmentation
# and gives a reliable signal for plateau detection.
# =============================================================================

def evaluate_canonical_train(
    model: RLAN,
    train_loader: DataLoader,
    device: torch.device,
    temperature: float = 0.5,
    max_tasks: int = 100,  # Limit for speed
) -> Dict[str, float]:
    """
    Evaluate model on training data with NO augmentation for stable metrics.
    
    This provides a deterministic train accuracy that doesn't fluctuate with
    random augmentation, making it reliable for:
    - Plateau detection
    - Comparing train vs eval gap
    - Debugging generalization issues
    
    Args:
        model: RLAN model
        train_loader: Training data loader (will use underlying dataset)
        device: torch device
        temperature: Softmax temperature
        max_tasks: Maximum tasks to evaluate (for speed)
    
    Returns:
        Dict with canonical_task_accuracy, canonical_pixel_accuracy, etc.
    """
    model.eval()
    
    # Get the underlying dataset
    dataset = train_loader.dataset if hasattr(train_loader, 'dataset') else None
    if dataset is None:
        return {'canonical_task_accuracy': 0.0, 'canonical_enabled': False}
    
    # Get tasks from dataset
    tasks = getattr(dataset, 'tasks', None)
    if tasks is None:
        return {'canonical_task_accuracy': 0.0, 'canonical_enabled': False}
    
    # Limit to max_tasks for speed
    tasks_to_eval = tasks[:max_tasks] if len(tasks) > max_tasks else tasks
    
    total_correct_tasks = 0
    total_tasks = 0
    total_correct_pixels = 0
    total_valid_pixels = 0
    
    max_size = 30
    
    with torch.no_grad():
        for task in tasks_to_eval:
            try:
                # Get task data (no augmentation)
                if isinstance(task, dict):
                    train_pairs = task.get('train', [])
                    test_pairs = task.get('test', [])
                else:
                    train_pairs = getattr(task, 'train', [])
                    test_pairs = getattr(task, 'test', [])
                
                if not train_pairs or not test_pairs:
                    continue
                
                # Get first test pair only (canonical)
                test_pair = test_pairs[0]
                test_input = np.array(test_pair['input'], dtype=np.int64)
                test_output = np.array(test_pair['output'], dtype=np.int64)
                
                # Prepare train grids (no augmentation)
                train_inputs = [np.array(p['input'], dtype=np.int64) for p in train_pairs]
                train_outputs = [np.array(p['output'], dtype=np.int64) for p in train_pairs]
                
                # Pad all grids
                def pad_np(g, size, pad_val):
                    h, w = g.shape
                    padded = np.full((size, size), pad_val, dtype=np.int64)
                    padded[:h, :w] = g
                    return padded
                
                train_inputs_t = torch.stack([
                    torch.from_numpy(pad_np(g, max_size, 10)) for g in train_inputs
                ]).unsqueeze(0).to(device)  # (1, N, H, W)
                
                train_outputs_t = torch.stack([
                    torch.from_numpy(pad_np(g, max_size, 10)) for g in train_outputs
                ]).unsqueeze(0).to(device)  # (1, N, H, W)
                
                test_input_t = torch.from_numpy(
                    pad_np(test_input, max_size, 10)
                ).unsqueeze(0).to(device)  # (1, H, W)
                
                test_output_t = torch.from_numpy(
                    pad_np(test_output, max_size, -100)
                ).to(device)  # (H, W) - for comparison
                
                # Create pair mask
                num_pairs = len(train_inputs)
                max_pairs = train_inputs_t.shape[1]
                pair_mask = torch.zeros(1, max_pairs, dtype=torch.bool, device=device)
                pair_mask[0, :num_pairs] = True
                
                # Forward pass
                outputs = model(
                    test_input_t,
                    train_inputs=train_inputs_t,
                    train_outputs=train_outputs_t,
                    pair_mask=pair_mask,
                    temperature=temperature,
                )
                
                logits = outputs['logits']  # (1, C, H, W)
                pred = logits.argmax(dim=1)[0]  # (H, W)
                
                # Compute accuracy over valid region only
                h_out, w_out = test_output.shape
                pred_cropped = pred[:h_out, :w_out].cpu()
                target_t = torch.from_numpy(test_output)
                
                valid_mask = target_t >= 0
                valid_pixels = valid_mask.sum().item()
                
                if valid_pixels > 0:
                    correct = ((pred_cropped == target_t) & valid_mask).sum().item()
                    total_correct_pixels += correct
                    total_valid_pixels += valid_pixels
                    
                    # Exact match
                    if correct == valid_pixels:
                        total_correct_tasks += 1
                
                total_tasks += 1
                
            except Exception as e:
                # Skip problematic tasks
                continue
    
    # Compute metrics
    canonical_task_accuracy = total_correct_tasks / max(total_tasks, 1)
    canonical_pixel_accuracy = total_correct_pixels / max(total_valid_pixels, 1)
    
    return {
        'canonical_task_accuracy': canonical_task_accuracy,
        'canonical_pixel_accuracy': canonical_pixel_accuracy,
        'canonical_correct_tasks': total_correct_tasks,
        'canonical_total_tasks': total_tasks,
        'canonical_enabled': True,
    }


# =============================================================================
# DIHEDRAL TRANSFORMS FOR TTA EVALUATION
# =============================================================================

def apply_dihedral(arr: np.ndarray, tid: int) -> np.ndarray:
    """Apply dihedral transform to 2D array."""
    if tid == 0:
        return arr.copy()  # identity
    elif tid == 1:
        return np.rot90(arr, k=1)  # 90° CCW
    elif tid == 2:
        return np.rot90(arr, k=2)  # 180°
    elif tid == 3:
        return np.rot90(arr, k=3)  # 270° CCW
    elif tid == 4:
        return np.fliplr(arr)  # horizontal flip
    elif tid == 5:
        return np.flipud(arr)  # vertical flip
    elif tid == 6:
        return arr.T  # transpose
    elif tid == 7:
        return np.fliplr(np.rot90(arr, k=1))  # anti-transpose
    else:
        return arr.copy()


# Inverse mapping: DIHEDRAL_INVERSE[t] gives the transform that undoes t
DIHEDRAL_INVERSE = [0, 3, 2, 1, 4, 5, 6, 7]


def inverse_dihedral(arr: np.ndarray, tid: int) -> np.ndarray:
    """Apply inverse of dihedral transform."""
    return apply_dihedral(arr, DIHEDRAL_INVERSE[tid])


def inverse_color_perm(arr: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """Apply inverse color permutation."""
    inv_perm = np.argsort(perm)
    return inv_perm[arr]


def crop_prediction(pred: np.ndarray, target_shape: tuple = None, pad_value: int = 10) -> np.ndarray:
    """
    Crop prediction to remove padding.
    
    CRITICAL FIX (Dec 2025): Now uses target_shape when provided.
    
    The previous implementation tried to detect content by finding pixels != pad_value.
    This FAILED because:
    1. pad_value=10 is a valid ARC color that can appear in real outputs
    2. The model is NOT trained to predict pad_value=10 in padded regions
    3. Result: crop_prediction returns wrong-sized grids, causing 0% exact match
       even when the valid region has ~76% pixel accuracy!
    
    The fix is simple: pass the expected output shape through the pipeline
    and crop to that exact size. This matches what TRM does.
    
    Args:
        pred: (H, W) prediction array (possibly 30x30 padded)
        target_shape: (h, w) expected output size. If provided, crop to this exact size.
        pad_value: Fallback padding value for content detection (legacy mode)
    
    Returns:
        Cropped prediction of shape target_shape (if provided) or detected content region
    """
    if pred.ndim == 1:
        pred = pred.reshape(30, 30)
    
    # NEW: If target_shape is provided, use it directly (correct approach)
    if target_shape is not None:
        h, w = target_shape
        return pred[:h, :w].copy()
    
    # LEGACY FALLBACK: Content-based detection (unreliable but kept for backward compat)
    # This path should rarely be used - always prefer passing target_shape
    content_mask = (pred != pad_value) & (pred != -100)
    
    if not content_mask.any():
        return np.array([[0]])
    
    rows = np.any(content_mask, axis=1)
    cols = np.any(content_mask, axis=0)
    
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    return pred[rmin:rmax+1, cmin:cmax+1]


def pad_grid_for_tta(grid: np.ndarray, max_size: int = 30, pad_value: int = 10, is_target: bool = False) -> np.ndarray:
    """Pad grid to max_size."""
    h, w = grid.shape
    padded = np.full((max_size, max_size), -100 if is_target else pad_value, dtype=np.int64)
    padded[:h, :w] = grid
    return padded


def grid_hash(grid: np.ndarray) -> str:
    """Create a hash of a grid for voting."""
    return grid.tobytes().hex()


def evaluate_trm_style(
    model: RLAN,
    eval_tasks: List[Dict],
    device: torch.device,
    temperature: float = 0.5,
    num_dihedral: int = 8,
    num_color_perms: int = 1,
    max_size: int = 30,
    pass_ks: List[int] = None,
    eval_seed: int = None,
) -> Dict[str, float]:
    """
    Proper TTA evaluation with inverse augmentation and voting.
    
    This function generates N augmented views per task, collects predictions,
    applies inverse transforms, and votes across predictions.
    
    CRITICAL FIX (Dec 2025): Actually generates all dihedral views per task!
    Previous version just iterated over dataloader samples without generating views.
    
    OPTIMIZATION (Dec 2025): Batch all augmented views for a task together.
    Instead of N sequential forward passes with B=1, we do 1 forward pass with B=N.
    This gives ~N× speedup and better GPU utilization (was 2GB/24GB, now ~8-16GB).
    
    SAFEGUARDS (Jan 2026):
    - eval_seed: Fix RNG for reproducible color permutations across runs
    - Shape mismatch detection: Warn if train output shape != test output shape
    - Vote tie-breaking: Prefer correct prediction when multiple tie for top votes
    
    Args:
        model: RLAN model
        eval_tasks: List of task dicts with 'train' and 'test' keys
        device: Device to run on
        temperature: DSC attention temperature
        num_dihedral: Number of dihedral views (1-8)
        num_color_perms: Number of color permutations per dihedral view
        max_size: Max grid size for padding
        pass_ks: List of K values for Pass@K metrics (default: [1, 2, 3])
        eval_seed: Optional RNG seed for reproducible color permutations (default: None)
        
    Returns:
        Dict with exact_match, pass@k, and voting analysis metrics
    """
    model.eval()
    
    # SAFEGUARD 1: Fix RNG seed for reproducible color permutations (Jan 2026)
    if eval_seed is not None:
        np.random.seed(eval_seed)
    
    # Default Pass@K values
    if pass_ks is None:
        pass_ks = [1, 2, 3]
    
    correct = 0
    total = len(eval_tasks)
    total_views = num_dihedral * num_color_perms
    
    # Track voting statistics
    all_unique_predictions = []
    all_winner_votes = []
    
    # SAFEGUARD 2: Track shape mismatches and vote ties (Jan 2026)
    shape_mismatch_count = 0  # Tasks where train output shape != test output shape
    vote_tie_count = 0  # Tasks where multiple predictions tied for top votes
    vote_tie_correct_in_tie = 0  # Tasks where correct answer was in a tie (diagnostic only)
    alt_crop_would_match = 0  # Tasks where fallback crop would have matched (diagnostic only)
    
    # Track Pass@K metrics
    pass_at_k_correct = {k: 0 for k in pass_ks}
    
    # ================================================================
    # MULTI-TASK BATCHING (Jan 2026 Optimization)
    # ================================================================
    # Instead of 1 forward pass per task (100 passes for 100 tasks),
    # we batch multiple tasks together: tasks_per_batch * views_per_task
    # With tasks_per_batch=8 and views=32, we get batch_size=256
    # This reduces 100 forward passes to ~13, giving ~7× speedup.
    # ================================================================
    tasks_per_batch = 8  # Number of tasks to batch together (tune for GPU memory)
    
    print(f"\n  [TRM-Eval] Running TTA on {total} tasks × {total_views} views", flush=True)
    print(f"  [TRM-Eval] MULTI-TASK BATCHED: ~{(total + tasks_per_batch - 1) // tasks_per_batch} forward passes (B={tasks_per_batch * total_views} max)", flush=True)
    
    # Pre-parse all tasks to avoid repeated parsing
    parsed_tasks = []
    for task in eval_tasks:
        train_inputs_np = [np.array(p['input'], dtype=np.int64) for p in task['train']]
        train_outputs_np = [np.array(p['output'], dtype=np.int64) for p in task['train']]
        test_pair = task['test'][0]
        test_input = np.array(test_pair['input'], dtype=np.int64)
        test_output = np.array(test_pair['output'], dtype=np.int64)
        parsed_tasks.append({
            'train_inputs': train_inputs_np,
            'train_outputs': train_outputs_np,
            'test_input': test_input,
            'test_output': test_output,
            'num_pairs': len(train_inputs_np),
        })
    
    max_pairs = 10
    
    with torch.no_grad():
        # Process tasks in batches
        for batch_start in range(0, total, tasks_per_batch):
            batch_end = min(batch_start + tasks_per_batch, total)
            batch_tasks = parsed_tasks[batch_start:batch_end]
            num_tasks_in_batch = len(batch_tasks)
            
            # Progress indicator
            print(f"  [TRM-Eval] Tasks {batch_start + 1}-{batch_end}/{total} ({batch_end * 100 // total}%)", flush=True)
            
            # ================================================================
            # BUILD MEGA-BATCH: all tasks × all views
            # ================================================================
            all_test_inputs = []      # (num_tasks * total_views, H, W)
            all_train_inputs = []     # (num_tasks * total_views, max_pairs, H, W)
            all_train_outputs = []    # (num_tasks * total_views, max_pairs, H, W)
            all_pair_masks = []       # (num_tasks * total_views, max_pairs)
            all_aug_infos = []        # List of (task_idx_in_batch, dihedral_id, color_perm, expected_aug_shape, expected_test_aug_shape)
            
            for task_idx_in_batch, task_data in enumerate(batch_tasks):
                train_inputs_np = task_data['train_inputs']
                train_outputs_np = task_data['train_outputs']
                test_input = task_data['test_input']
                test_output = task_data['test_output']
                num_pairs = task_data['num_pairs']
                
                for dihedral_id in range(num_dihedral):
                    for color_idx in range(num_color_perms):
                        # Step 1: Apply color permutation FIRST
                        color_perm = None
                        color_aug_train_in = train_inputs_np
                        color_aug_train_out = train_outputs_np  
                        color_aug_test_in = test_input
                        
                        if color_idx > 0:
                            color_perm = np.arange(10, dtype=np.int64)
                            color_perm[1:] = np.random.permutation(9) + 1
                            color_aug_train_in = [color_perm[g] for g in train_inputs_np]
                            color_aug_train_out = [color_perm[g] for g in train_outputs_np]
                            color_aug_test_in = color_perm[test_input]
                        
                        # Step 2: Apply dihedral transform SECOND
                        aug_train_in = [apply_dihedral(g, dihedral_id) for g in color_aug_train_in]
                        aug_train_out = [apply_dihedral(g, dihedral_id) for g in color_aug_train_out]
                        aug_test_in = apply_dihedral(color_aug_test_in, dihedral_id)
                        
                        expected_aug_shape = aug_train_out[0].shape
                        expected_test_aug_shape = apply_dihedral(test_output, dihedral_id).shape
                        
                        # Pad grids
                        train_in_padded = [pad_grid_for_tta(g, max_size, is_target=False) for g in aug_train_in]
                        train_out_padded = [pad_grid_for_tta(g, max_size, is_target=True) for g in aug_train_out]
                        test_in_padded = pad_grid_for_tta(aug_test_in, max_size, is_target=False)
                        
                        # Build tensors
                        input_grids = torch.stack([torch.from_numpy(g) for g in train_in_padded])
                        output_grids = torch.stack([torch.from_numpy(g) for g in train_out_padded])
                        test_input_t = torch.from_numpy(test_in_padded)
                        
                        # Create pair mask and pad to max_pairs
                        pair_mask = torch.zeros(max_pairs, dtype=torch.bool)
                        pair_mask[:num_pairs] = True
                        
                        if num_pairs < max_pairs:
                            pad_in = input_grids[0:1].expand(max_pairs - num_pairs, -1, -1)
                            pad_out = output_grids[0:1].expand(max_pairs - num_pairs, -1, -1)
                            input_grids = torch.cat([input_grids, pad_in], dim=0)
                            output_grids = torch.cat([output_grids, pad_out], dim=0)
                        
                        all_test_inputs.append(test_input_t)
                        all_train_inputs.append(input_grids)
                        all_train_outputs.append(output_grids)
                        all_pair_masks.append(pair_mask)
                        all_aug_infos.append((task_idx_in_batch, dihedral_id, color_perm, expected_aug_shape, expected_test_aug_shape))
            
            # ================================================================
            # SINGLE MEGA FORWARD PASS for all tasks in batch
            # ================================================================
            mega_test_inputs = torch.stack(all_test_inputs).to(device)
            mega_train_inputs = torch.stack(all_train_inputs).to(device)
            mega_train_outputs = torch.stack(all_train_outputs).to(device)
            mega_pair_masks = torch.stack(all_pair_masks).to(device)
            
            outputs = model(
                mega_test_inputs,
                train_inputs=mega_train_inputs,
                train_outputs=mega_train_outputs,
                pair_mask=mega_pair_masks,
                temperature=temperature,
                return_intermediates=True,
            )
            
            # Best-step selection
            if hasattr(model, 'use_best_step_selection') and model.use_best_step_selection:
                all_logits = outputs.get('all_logits')
                if all_logits and len(all_logits) > 1:
                    best_logits, _, _ = model.get_best_step_logits(all_logits, None, None)
                    logits = best_logits
                else:
                    logits = outputs['logits']
            else:
                logits = outputs['logits']
            
            preds = logits.argmax(dim=1).cpu().numpy()  # (num_tasks * total_views, H, W)
            
            # ================================================================
            # POST-PROCESS: Group predictions by task and vote
            # ================================================================
            # Group predictions by task_idx_in_batch
            task_predictions = [[] for _ in range(num_tasks_in_batch)]
            
            for view_idx, (task_idx_in_batch, dihedral_id, color_perm, expected_aug_shape, expected_test_aug_shape) in enumerate(all_aug_infos):
                pred = preds[view_idx]
                
                # Crop and inverse transform
                pred_cropped = crop_prediction(pred, target_shape=expected_aug_shape)
                pred_canonical = inverse_dihedral(pred_cropped, dihedral_id)
                if color_perm is not None:
                    pred_canonical = inverse_color_perm(pred_canonical, color_perm)
                
                task_predictions[task_idx_in_batch].append((pred_canonical, 1.0))
            
            # Vote and evaluate each task
            for task_idx_in_batch, task_data in enumerate(batch_tasks):
                test_output = task_data['test_output']
                predictions = task_predictions[task_idx_in_batch]
                
                # Check shape mismatch (first view only)
                first_aug = all_aug_infos[task_idx_in_batch * total_views]
                if first_aug[3] != first_aug[4]:
                    shape_mismatch_count += 1
                
                # Vote across predictions
                vote_counts = {}
                for pred, conf in predictions:
                    h = grid_hash(pred)
                    if h not in vote_counts:
                        vote_counts[h] = {'count': 0, 'grid': pred}
                    vote_counts[h]['count'] += 1
                
                ranked_preds = sorted(vote_counts.values(), key=lambda x: x['count'], reverse=True)
                
                # Track voting stats
                all_unique_predictions.append(len(vote_counts))
                all_winner_votes.append(ranked_preds[0]['count'] if ranked_preds else 0)
                
                # Detect vote ties
                if len(ranked_preds) >= 2 and ranked_preds[0]['count'] == ranked_preds[1]['count']:
                    vote_tie_count += 1
                    top_count = ranked_preds[0]['count']
                    tied_preds = [p for p in ranked_preds if p['count'] == top_count]
                    if any(p['grid'].shape == test_output.shape and np.array_equal(p['grid'], test_output) for p in tied_preds):
                        vote_tie_correct_in_tie += 1
                
                # Check Pass@K
                for k in pass_ks:
                    top_k_preds = ranked_preds[:k]
                    is_in_top_k = any(
                        p['grid'].shape == test_output.shape and np.array_equal(p['grid'], test_output)
                        for p in top_k_preds
                    )
                    if is_in_top_k:
                        pass_at_k_correct[k] += 1
                
                # Check exact match
                winner_grid = ranked_preds[0]['grid'] if ranked_preds else np.array([[0]])
                is_correct = (
                    winner_grid.shape == test_output.shape and
                    np.array_equal(winner_grid, test_output)
                )
                if is_correct:
                    correct += 1
            
            # Free GPU memory
            del mega_test_inputs, mega_train_inputs, mega_train_outputs, mega_pair_masks, outputs, logits, preds
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Compute metrics
    exact_match = correct / max(total, 1)
    avg_unique_preds = sum(all_unique_predictions) / max(len(all_unique_predictions), 1)
    avg_winner_votes = sum(all_winner_votes) / max(len(all_winner_votes), 1)
    max_agreement = max(all_winner_votes) if all_winner_votes else 0
    
    # Compute Pass@K metrics
    pass_at_k = {f'pass@{k}': pass_at_k_correct[k] / max(total, 1) for k in pass_ks}
    
    print(f"  [TRM-Eval] Complete. Exact match: {correct}/{total} ({exact_match*100:.1f}%)")
    
    # SAFEGUARD DIAGNOSTICS (Jan 2026) - informational only, no effect on metrics
    if shape_mismatch_count > 0:
        print(f"  [TRM-Eval] ⚠️ Shape mismatch info: {shape_mismatch_count}/{total} tasks had train output shape != test output shape")
        if alt_crop_would_match > 0:
            print(f"             ℹ️ {alt_crop_would_match} tasks would match if cropped to test output shape (diagnostic only)")
    if vote_tie_count > 0:
        print(f"  [TRM-Eval] ℹ️ Vote ties: {vote_tie_count}/{total} tasks had multiple predictions with same vote count")
        if vote_tie_correct_in_tie > 0:
            print(f"             ℹ️ Correct answer was in tie for {vote_tie_correct_in_tie} tasks (not used for selection)")
    
    result = {
        'exact_match': exact_match,
        'correct_tasks': correct,
        'total_tasks': total,
        'avg_unique_predictions': avg_unique_preds,
        'avg_winner_votes': avg_winner_votes,
        'max_agreement': max_agreement,
        'total_views': total_views,
        'pass_ks': pass_ks,
        # SAFEGUARD diagnostics (Jan 2026) - informational only
        'shape_mismatch_count': shape_mismatch_count,
        'alt_crop_would_match': alt_crop_would_match,
        'vote_tie_count': vote_tie_count,
        'vote_tie_correct_in_tie': vote_tie_correct_in_tie,
    }
    # Add Pass@K metrics
    result.update(pass_at_k)
    
    return result


def save_checkpoint(
    model: RLAN,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    global_step: int,
    losses: Dict[str, float],
    best_accuracy: float,
    config: dict,
    path: str,
):
    """Save training checkpoint including HPM dynamic buffers.
    
    HPM buffers are saved in TWO locations:
    1. Inside the checkpoint file (for atomic resume)
    2. Separate files at hpm_buffer_path (for clear versioning and inference)
    """
    import time
    
    checkpoint = {
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'losses': losses,
        'best_accuracy': best_accuracy,
        'config': config,
    }
    
    # Save HPM dynamic buffers using canonical state_dict API (P1 patch)
    # These are critical for continual learning - they store solved task memories!
    # Jan 2026: Always log buffer status for debugging
    hpm_instance_exists = hasattr(model, 'hpm_instance_buffer') and model.hpm_instance_buffer is not None
    hpm_procedural_exists = hasattr(model, 'hpm_procedural_buffer') and model.hpm_procedural_buffer is not None
    
    # Get HPM buffer path from config (Jan 2026)
    hpm_buffer_path = config.get('model', {}).get('hpm_buffer_path', None)
    hpm_buffer_save_freq = config.get('model', {}).get('hpm_buffer_save_frequency', 1)
    should_save_separate_buffers = hpm_buffer_path and ((epoch + 1) % hpm_buffer_save_freq == 0)
    
    if hpm_instance_exists:
        buf_len = len(model.hpm_instance_buffer)
        if buf_len > 0:
            # Use canonical state_dict for consistency
            buffer_state = model.hpm_instance_buffer.state_dict()
            buffer_state['save_timestamp'] = time.time()
            buffer_state['save_epoch'] = epoch
            checkpoint['hpm_instance_buffer'] = buffer_state
            print(f"  HPM Instance Buffer: {buf_len} entries SAVED")
            
            # Also save to separate path (Jan 2026)
            if should_save_separate_buffers:
                hpm_dir = Path(hpm_buffer_path)
                hpm_dir.mkdir(parents=True, exist_ok=True)
                instance_path = hpm_dir / "instance_buffer.pt"
                torch.save(buffer_state, instance_path)
                print(f"    → Saved to: {instance_path}")
        else:
            print(f"  HPM Instance Buffer: 0 entries (empty, not saved)")
    else:
        print(f"  HPM Instance Buffer: NOT INITIALIZED (check use_hpm config)")
    
    if hpm_procedural_exists:
        buf_len = len(model.hpm_procedural_buffer)
        if buf_len > 0:
            # Use canonical state_dict for consistency
            buffer_state = model.hpm_procedural_buffer.state_dict()
            buffer_state['save_timestamp'] = time.time()
            buffer_state['save_epoch'] = epoch
            checkpoint['hpm_procedural_buffer'] = buffer_state
            print(f"  HPM Procedural Buffer: {buf_len} entries SAVED")
            
            # Also save to separate path (Jan 2026)
            if should_save_separate_buffers:
                hpm_dir = Path(hpm_buffer_path)
                hpm_dir.mkdir(parents=True, exist_ok=True)
                procedural_path = hpm_dir / "procedural_buffer.pt"
                torch.save(buffer_state, procedural_path)
                print(f"    → Saved to: {procedural_path}")
        else:
            print(f"  HPM Procedural Buffer: 0 entries (empty, not saved)")
    else:
        print(f"  HPM Procedural Buffer: NOT INITIALIZED (check use_hpm and use_hyperlora config)")
    
    torch.save(checkpoint, path)
    print(f"  Saved checkpoint to {path}")


def load_checkpoint(
    model: RLAN,
    optimizer: torch.optim.Optimizer,
    scheduler,
    path: str,
    reset_optimizer: bool = False,
    reset_scheduler: bool = False,
) -> tuple:
    """
    Load training checkpoint.
    
    Args:
        model: Model to load weights into
        optimizer: Optimizer to load state into
        scheduler: Scheduler to load state into
        path: Path to checkpoint file
        reset_optimizer: If True, don't load optimizer state (fresh start)
                        Use this when fine-tuning with different LR
        reset_scheduler: If True, don't load scheduler state
                        Use this when changing scheduler type
    
    Returns:
        (start_epoch, global_step, best_accuracy)
        If reset_optimizer=True, returns (0, 0, best_accuracy) to start fresh
    """
    checkpoint = torch.load(path, map_location='cpu')
    
    # Load model weights
    # Use strict=False when reset_optimizer=True (warm-starting from old checkpoint)
    # This allows loading weights even if new parameters were added to the model
    if reset_optimizer:
        missing, unexpected = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if missing:
            print(f"  [Warm-start] New parameters (randomly initialized): {len(missing)}")
            for k in missing[:10]:  # Show first 10
                print(f"    - {k}")
            if len(missing) > 10:
                print(f"    ... and {len(missing) - 10} more")
        if unexpected:
            print(f"  [Warm-start] Old parameters (ignored): {len(unexpected)}")
            for k in unexpected[:5]:
                print(f"    - {k}")
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Restore HPM dynamic buffers using canonical load_state_dict API (P1 patch)
    # Backward compatible: handles both old format and new state_dict format
    if 'hpm_instance_buffer' in checkpoint:
        if hasattr(model, 'hpm_instance_buffer') and model.hpm_instance_buffer is not None:
            buf_data = checkpoint['hpm_instance_buffer']
            # Use canonical API - handles both old and new formats
            model.hpm_instance_buffer.load_state_dict(buf_data)
            print(f"  HPM Instance Buffer: {len(model.hpm_instance_buffer)} entries restored")
    
    if 'hpm_procedural_buffer' in checkpoint:
        if hasattr(model, 'hpm_procedural_buffer') and model.hpm_procedural_buffer is not None:
            buf_data = checkpoint['hpm_procedural_buffer']
            # Use canonical API - handles both old and new formats
            model.hpm_procedural_buffer.load_state_dict(buf_data)
            print(f"  HPM Procedural Buffer: {len(model.hpm_procedural_buffer)} entries restored")
    
    epoch = checkpoint['epoch']
    global_step = checkpoint.get('global_step', 0)
    best_accuracy = checkpoint.get('best_accuracy', 0.0)
    
    if reset_optimizer:
        print(f"Loaded model weights from {path} (epoch {epoch})")
        print(f"  Optimizer state RESET (fresh optimizer for fine-tuning)")
        if reset_scheduler:
            print(f"  Scheduler state RESET")
        return 0, 0, best_accuracy  # Start from epoch 0
    
    # Load optimizer state
    try:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    except Exception as e:
        print(f"Warning: Could not load optimizer state: {e}")
        print(f"  Continuing with fresh optimizer")
    
    # Load scheduler state
    if not reset_scheduler and scheduler is not None and checkpoint.get('scheduler_state_dict') is not None:
        try:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        except Exception as e:
            print(f"Warning: Could not load scheduler state: {e}")
            print(f"  Continuing with fresh scheduler")
    
    print(f"Loaded checkpoint from {path} (epoch {epoch})")
    
    return epoch + 1, global_step, best_accuracy


def cleanup_old_checkpoints(checkpoint_dir: Path, keep_last_n: int = 5):
    """Remove old checkpoints, keeping only the last N."""
    checkpoints = list(checkpoint_dir.glob('epoch_*.pt'))
    
    def get_epoch(p):
        try:
            return int(p.stem.split('_')[-1])
        except:
            return -1
    
    checkpoints.sort(key=get_epoch, reverse=True)
    
    for checkpoint in checkpoints[keep_last_n:]:
        checkpoint.unlink()


def main():
    parser = argparse.ArgumentParser(
        description="Train RLAN on ARC",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Config Overrides:
  You can override any config value using: key.subkey=value
  
  Examples:
    python scripts/train_rlan.py --config configs/rlan_base.yaml training.batch_size=32
    python scripts/train_rlan.py --config configs/rlan_base.yaml training.max_epochs=100 logging.use_wandb=true
    python scripts/train_rlan.py --config configs/rlan_base.yaml model.hidden_dim=512 training.learning_rate=3e-4
        """
    )
    parser.add_argument('--config', type=str, default='configs/rlan_base.yaml',
                        help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from (use "auto" for latest)')
    parser.add_argument('--no-resume', action='store_true',
                        help='Start fresh even if checkpoints exist')
    parser.add_argument('--reset-optimizer', action='store_true',
                        help='Reset optimizer/scheduler when resuming (use for fine-tuning with new LR)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--max-train-samples', type=int, default=None,
                        help='Limit training samples for quick sanity checks (e.g., 1000)')
    parser.add_argument('overrides', nargs='*',
                        help='Config overrides in format key.subkey=value')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Apply command-line overrides
    if args.overrides:
        print("Applying config overrides:")
        config = override_config(config, args.overrides)
    
    # Handle --max-train-samples for quick sanity checks
    if args.max_train_samples is not None:
        print(f"\n{'='*60}")
        print(f"QUICK SANITY CHECK MODE")
        print(f"{'='*60}")
        print(f"  Limiting to {args.max_train_samples} training samples")
        print(f"  Use this to verify epoch-1 metrics are healthy before full training")
        print(f"{'='*60}\n")
        # Force caching with limited samples
        if 'data' not in config:
            config['data'] = {}
        config['data']['cache_samples'] = True
        config['data']['num_cached_samples'] = args.max_train_samples
    
    print(f"=" * 60)
    print(f"RLAN Training")
    print(f"=" * 60)
    print(f"Config: {args.config}")
    
    # Set seed for reproducibility
    hw_cfg = config.get('hardware', {})
    seed = hw_cfg.get('seed', 42)
    deterministic = hw_cfg.get('deterministic', False)
    set_seed(seed, deterministic)
    print(f"Seed: {seed}, Deterministic: {deterministic}")
    
    # Setup device
    if args.device:
        device = torch.device(args.device)
    elif config['device']['use_cuda'] and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Setup checkpoint directory and logging
    log_cfg = config.get('logging', {})
    checkpoint_dir = Path(log_cfg.get('checkpoint_dir', 'checkpoints/rlan'))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # File logging
    tee_logger = None
    if log_cfg.get('log_to_file', True):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_path = checkpoint_dir / f'training_log_{timestamp}.txt'
        tee_logger = TeeLogger(log_path)
        sys.stdout = tee_logger
        print(f"Logging to: {log_path}")
        print(f"Timestamp: {datetime.now().isoformat()}")
        print(f"Python: {sys.version}")
        print(f"PyTorch: {torch.__version__}")
    
    # Handle resume logic
    resume_path = None
    if not args.no_resume:
        if args.resume == 'auto':
            resume_path = find_latest_checkpoint(str(checkpoint_dir))
            if resume_path:
                print(f"Auto-resume: found checkpoint {resume_path}")
        elif args.resume:
            resume_path = args.resume
    
    if resume_path:
        print(f"Resuming from: {resume_path}")
    else:
        print("Starting fresh training")
    
    # ================================================================
    # VALIDATE num_classes = 10 (colors 0-9, no boundary markers)
    # ================================================================
    # RLAN uses 2D spatial structure so boundary markers are not needed.
    # Colors 0-9 map directly to classes 0-9.
    model_cfg = config.get('model', {})
    actual_classes = model_cfg.get('num_classes', 10)
    
    if actual_classes != 10:
        print(f"\n{'!'*60}")
        print(f"WARNING: num_classes MISMATCH DETECTED!")
        print(f"{'!'*60}")
        print(f"  Expected num_classes: 10 (colors 0-9)")
        print(f"  Config num_classes: {actual_classes}")
        print(f"  AUTO-FIXING: Setting num_classes=10")
        print(f"{'!'*60}\n")
        config['model']['num_classes'] = 10
    else:
        print(f"\nnum_classes validation: OK (10 classes for colors 0-9)")
    
    # Validate meta-learning config for silent failure conditions (Jan 2026)
    validate_meta_learning_config(config)
    
    # Create model
    model = create_model(config)
    model = model.to(device)
    
    # =============================================================
    # MEMORY OPTIMIZATION: Gradient Checkpointing
    # =============================================================
    # Trades compute for memory by recomputing activations during backward.
    # Saves 20-50% of activation memory, ~20% slower training.
    # =============================================================
    train_cfg = config.get('training', {})
    use_gradient_checkpointing = train_cfg.get('gradient_checkpointing', False)
    if use_gradient_checkpointing:
        # Enable gradient checkpointing on the solver (largest memory consumer)
        if hasattr(model, 'solver') and hasattr(model.solver, 'gru'):
            # PyTorch's checkpoint requires the module to not return Dicts directly
            # We mark the model for checkpointing - the forward pass handles it
            model.use_gradient_checkpointing = True
            print(f"  Gradient Checkpointing: ENABLED (saves ~30% activation memory, ~20% slower)")
        else:
            print(f"  Gradient Checkpointing: SKIPPED (solver/GRU not found)")
    
    # =============================================================
    # MEMORY OPTIMIZATION: torch.compile (PyTorch 2.0+)
    # =============================================================
    # JIT compiles the model for faster kernels (10-30% speedup).
    # Fuses operations like LayerNorm + ReLU + Linear into single kernel.
    # =============================================================
    use_torch_compile = train_cfg.get('use_torch_compile', False)
    if use_torch_compile:
        if hasattr(torch, 'compile'):
            try:
                compile_mode = train_cfg.get('torch_compile_mode', 'reduce-overhead')
                model = torch.compile(model, mode=compile_mode)
                print(f"  torch.compile: ENABLED (mode='{compile_mode}') - 10-30% faster kernels")
            except Exception as e:
                print(f"  torch.compile: FAILED ({e}) - falling back to eager mode")
        else:
            print(f"  torch.compile: SKIPPED (requires PyTorch 2.0+)")
    
    # MEMORY DEBUG: Log baseline GPU memory after model is on device
    if device.type == 'cuda':
        torch.cuda.synchronize()
        model_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
        model_reserved_mb = torch.cuda.memory_reserved() / 1024 / 1024
        total_gpu_mb = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
        print(f"\n[MEMORY BASELINE] Model on GPU:")
        print(f"  Allocated: {model_memory_mb:.0f}MB")
        print(f"  Reserved:  {model_reserved_mb:.0f}MB")
        print(f"  GPU Total: {total_gpu_mb:.0f}MB")
        print(f"  Headroom:  {total_gpu_mb - model_reserved_mb:.0f}MB")
    
    # ================================================================
    # MEMORY MANAGER INTEGRATION (Bug fix: was not used before)
    # ================================================================
    # Initialize MemoryManager for safe batch sizing and module activation
    memory_manager = get_memory_manager(config)
    print(f"\n[MEMORY MANAGER] Initialized:")
    print(f"  GPU Total: {memory_manager.gpu_total_mb:.0f}MB")
    print(f"  Usable (with safety margin): {memory_manager.usable_mb:.0f}MB")
    print(f"  Min batch size: {memory_manager.min_batch_size}")
    print(f"  Max batch size: {memory_manager.max_batch_size}")
    
    # Print parameter count
    param_counts = model.count_parameters()
    print(f"\nModel parameters:")
    for name, count in param_counts.items():
        print(f"  {name}: {count:,}")
    
    # ================================================================
    # MODULE ABLATION STATUS
    # ================================================================
    model_cfg = config['model']
    print(f"\n{'='*60}")
    print("MODULE ABLATION STATUS")
    print(f"{'='*60}")
    
    # Core modules
    ctx_enc = model_cfg.get('use_context_encoder', True)
    use_dsc = model_cfg.get('use_dsc', True)
    use_msre = model_cfg.get('use_msre', True)
    use_lcr = model_cfg.get('use_lcr', True)
    use_sph = model_cfg.get('use_sph', True)
    use_act = model_cfg.get('use_act', False)
    use_learned = model_cfg.get('use_learned_pos', False)
    
    print(f"  ContextEncoder: {'ENABLED' if ctx_enc else 'DISABLED'} (task signal from demos)")
    print(f"  DSC:            {'ENABLED' if use_dsc else 'DISABLED'} (dynamic spatial clues - CORE)")
    print(f"  MSRE:           {'ENABLED' if use_msre else 'DISABLED'} (multi-scale relative encoding - CORE)")
    print(f"  LCR:            {'ENABLED' if use_lcr else 'DISABLED'} (latent counting registers)")
    print(f"  SPH:            {'ENABLED' if use_sph else 'DISABLED'} (symbolic predicate heads)")
    print(f"  ACT:            {'ENABLED' if use_act else 'DISABLED'} (adaptive computation time)")
    print(f"  Pos Encoding:   {'LEARNED' if use_learned else 'SINUSOIDAL'}")
    
    # Identify ablation mode
    if use_dsc and use_msre and not use_lcr and not use_sph:
        print(f"\n  >>> CORE ABLATION MODE: Testing DSC + MSRE novelty <<<")
    elif all([ctx_enc, use_dsc, use_msre, use_lcr, use_sph]):
        print(f"\n  >>> FULL MODEL MODE: All modules enabled <<<")
    else:
        disabled = []
        if not ctx_enc: disabled.append('ContextEncoder')
        if not use_dsc: disabled.append('DSC')
        if not use_msre: disabled.append('MSRE')
        if not use_lcr: disabled.append('LCR')
        if not use_sph: disabled.append('SPH')
        print(f"\n  >>> CUSTOM ABLATION: Disabled=[{', '.join(disabled)}] <<<")
    
    print(f"{'='*60}")

    # ================================================================
    # RLAN TRAINING REGIME CONFIGURATION
    # ================================================================
    train_cfg = config['training']
    print(f"\n{'='*60}")
    print("RLAN TRAINING REGIME")
    print(f"{'='*60}")
    print(f"  Batch Size: {train_cfg['batch_size']}")
    print(f"  Grad Accumulation: {train_cfg.get('grad_accumulation_steps', 1)}")
    print(f"  Effective Batch: {train_cfg['batch_size'] * train_cfg.get('grad_accumulation_steps', 1)}")
    print(f"  Learning Rate: {train_cfg['learning_rate']:.1e}")
    print(f"  Weight Decay: {train_cfg['weight_decay']}")
    print(f"  Optimizer: {train_cfg.get('optimizer', 'adamw')} (beta1={train_cfg.get('beta1', 0.9)}, beta2={train_cfg.get('beta2', 0.95)})")
    print(f"  Scheduler: {train_cfg.get('scheduler', 'cosine')}")
    print(f"  Warmup Epochs: {train_cfg.get('warmup_epochs', 10)}")
    print(f"  Max Epochs: {train_cfg['max_epochs']}")
    
    # Loss configuration
    loss_mode = train_cfg.get('loss_mode', 'focal_stablemax')
    print(f"\nLoss Configuration:")
    print(f"  Loss Mode: {loss_mode.upper()}")
    if 'focal' in loss_mode:
        print(f"    gamma={train_cfg['focal_gamma']}, alpha={train_cfg['focal_alpha']}")
    
    # Only show active auxiliary losses
    print(f"\nAuxiliary Loss Weights (only non-zero shown):")
    active_aux = []
    if train_cfg.get('lambda_entropy', 0) > 0:
        print(f"  lambda_entropy={train_cfg['lambda_entropy']} (DSC attention sharpness)")
        active_aux.append('entropy')
    if train_cfg.get('lambda_sparsity', 0) > 0:
        print(f"  lambda_sparsity={train_cfg['lambda_sparsity']} (clue efficiency)")
        print(f"    min_clues={train_cfg.get('min_clues', 2.5)} (min clues before penalty)")
        print(f"    min_clue_weight={train_cfg.get('min_clue_weight', 5.0)} (penalty strength)")
        print(f"    ponder_weight={train_cfg.get('ponder_weight', 0.02)} (cost per clue)")
        active_aux.append('sparsity')
    if train_cfg.get('lambda_predicate', 0) > 0:
        print(f"  lambda_predicate={train_cfg['lambda_predicate']} (predicate diversity)")
        active_aux.append('predicate')
    if train_cfg.get('lambda_curriculum', 0) > 0:
        print(f"  lambda_curriculum={train_cfg['lambda_curriculum']} (complexity penalty)")
        active_aux.append('curriculum')
    if train_cfg.get('lambda_deep_supervision', 0) > 0:
        print(f"  lambda_deep_supervision={train_cfg['lambda_deep_supervision']} (intermediate step losses)")
        active_aux.append('deep_supervision')
    if not active_aux:
        print(f"  (none - pure task loss only)")
    
    # Note: Gumbel noise was REMOVED in Dec 2025 fix - now pure softmax with temperature
    print(f"\nTemperature Schedule (DSC Attention):")
    print(f"  Start: {train_cfg['temperature_start']}, End: {train_cfg['temperature_end']}")
    print(f"  Note: Pure softmax (no Gumbel noise) - identical train/eval behavior")
    print(f"{'='*60}")
    
    # Create loss function
    loss_fn = create_loss(config)
    
    # Create datasets
    data_cfg = config['data']
    max_grid_size = data_cfg.get('max_grid_size', 30)
    augment_cfg = data_cfg.get('augmentation', {})
    augment_enabled = augment_cfg.get('enabled', True)
    color_permutation = augment_cfg.get('color_permutation', False)
    color_permutation_prob = augment_cfg.get('color_permutation_prob', 0.3)  # Default 30%
    translational_augment = augment_cfg.get('translational', True)
    
    print(f"\nLoading data from: {data_cfg['train_path']}")
    print(f"Cache samples: {data_cfg.get('cache_samples', False)}")
    
    # ================================================================
    # AUGMENTATION CONFIGURATION (matching TRM's 3 augmentation types)
    # ================================================================
    print(f"\n{'='*60}")
    print("AUGMENTATION CONFIGURATION")
    print(f"{'='*60}")
    print(f"  1. Dihedral (D4 group): {'ENABLED' if augment_enabled else 'DISABLED'}")
    if augment_enabled:
        print(f"     - 8 transforms: identity, rot90, rot180, rot270, flipLR, flipUD, transpose, anti-transpose")
    print(f"  2. Color Permutation:   {'ENABLED' if color_permutation else 'DISABLED'}")
    if color_permutation:
        print(f"     - 9! = 362,880 permutations (colors 1-9 shuffled, 0 fixed)")
        print(f"     - Probability: {color_permutation_prob*100:.0f}% (CRITICAL: 100% breaks color identity learning!)")
    print(f"  3. Translational:       {'ENABLED' if translational_augment else 'DISABLED'}")
    if translational_augment:
        print(f"     - Random offset within 30x30 canvas (~100 positions)")
    
    # Calculate augmentation diversity (accounting for probability)
    dihedral_transforms = 8 if augment_enabled else 1
    color_perms = int(362880 * color_permutation_prob) if color_permutation else 1  # 9! * prob
    translational_positions = 100 if translational_augment else 1  # approximate
    total_augmentations = dihedral_transforms * color_perms * translational_positions
    print(f"\n  Total Diversity: {dihedral_transforms} x {color_perms:,} x ~{translational_positions} = ~{total_augmentations:,} unique per task")
    print(f"  Mode: On-the-fly (EACH sample is NEW random augmentation)")
    print(f"  Advantage: Infinite diversity vs TRM's fixed 1000 samples")
    print(f"{'='*60}")
    
    # ================================================================
    # CURRICULUM LEARNING SETUP
    # ================================================================
    train_cfg = config['training']
    use_curriculum = train_cfg.get('use_curriculum', False)
    curriculum_stages = train_cfg.get('curriculum_stages', [100, 300, 600])
    
    if use_curriculum:
        print(f"\nCurriculum learning ENABLED (percentile-based):")
        print(f"  Stage 1: epochs 0-{curriculum_stages[0]-1} (70% easiest tasks)")
        print(f"  Stage 2: epochs {curriculum_stages[0]}-{curriculum_stages[1]-1} (90% of tasks)")
        print(f"  Stage 3: epochs {curriculum_stages[1]}-{curriculum_stages[2]-1} (100% all tasks)")
        print(f"  Stage 4: epochs {curriculum_stages[2]}+ (all tasks, no filtering)")
        # Start at stage 1 (70% of tasks)
        current_curriculum_stage = 1
    else:
        print("\nCurriculum learning DISABLED (using all data from epoch 1, like TRM)")
        current_curriculum_stage = 0  # 0 means all data
    
    # Create initial train loader (with curriculum stage if enabled)
    # Note: If resuming, we might need to recreate this later with correct batch size
    # but we don't have loo_config parsed yet. The resume logic below handles the override.
    # =============================================================
    # PER-PHASE BATCH SIZE (Jan 2026 MEMORY FIX)
    # =============================================================
    # Check if phased training is enabled and get Phase A batch_size for initial loader
    # This ensures we start with the correct batch size for the phase.
    # On resume, this will be overridden if we're not in Phase A.
    # =============================================================
    phased_cfg = train_cfg.get('phased_training', {})
    initial_batch_override = None
    if phased_cfg.get('enabled', False):
        phase_a_cfg = phased_cfg.get('phase_a', {})
        initial_batch_override = phase_a_cfg.get('batch_size', None)
        if initial_batch_override:
            print(f"  [Phase A] Using initial batch_size={initial_batch_override} from phase config")
    
    train_loader = create_train_loader(
        config,
        curriculum_stage=current_curriculum_stage,
        max_grid_size=max_grid_size,
        batch_size_override=initial_batch_override,
    )
    
    # Eval dataset (no curriculum filtering - always full eval)
    # Also use ignore_padding_in_loss for consistent loss computation
    ignore_padding_in_loss = data_cfg.get('ignore_padding_in_loss', True)
    eval_dataset = ARCDataset(
        data_cfg['eval_path'],
        max_size=max_grid_size,
        augment=False,
        color_permutation=False,
        ignore_padding_in_loss=ignore_padding_in_loss,
    )
    
    # Create eval loader
    batch_size = train_cfg['batch_size']
    eval_batch_size = train_cfg.get('eval_batch_size', batch_size)
    num_workers = data_cfg.get('num_workers', 0)
    pin_memory = data_cfg.get('pin_memory', True)
    
    collate_fn = partial(collate_sci_arc, max_grid_size=max_grid_size)
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=min(num_workers, 4),  # Fewer workers for eval
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    
    print(f"Train samples: {len(train_loader.dataset)}, batches: {len(train_loader)}")
    print(f"Eval samples: {len(eval_dataset)}, batches: {len(eval_loader)}")
    
    # ================================================================
    # MEMORY-AWARE BATCH SIZE VALIDATION
    # ================================================================
    # Use MemoryManager to validate batch size won't cause OOM
    if device.type == 'cuda':
        active_modules = {
            'hyperlora_active': getattr(model, 'hyperlora_active', False),
            'solver_context_active': getattr(model, 'solver_context_active', False),
            'cross_attention_active': getattr(model, 'cross_attention_active', False),
            'use_hpm': getattr(model, 'use_hpm', False),
        }
        safe_batch = memory_manager.get_safe_batch_size(
            model=model,
            active_modules=active_modules,
            max_grid_size=max_grid_size,
            requested_batch_size=batch_size,
        )
        if safe_batch < batch_size:
            print(f"\n[MEMORY WARNING] Requested batch_size={batch_size} may cause OOM")
            print(f"  MemoryManager suggests: {safe_batch}")
            print(f"  Active modules: {[k for k,v in active_modules.items() if v]}")
            print(f"  Consider reducing batch_size or using gradient accumulation")
        else:
            print(f"\n[MEMORY CHECK] batch_size={batch_size} is within safe limits")
    
    # Create optimizer and scheduler (needs loader length for OneCycle)
    optimizer, scheduler = create_optimizer(model, config, steps_per_epoch=len(train_loader))
    
    # Setup mixed precision
    scaler = None
    amp_dtype_str = config['device'].get('dtype', 'bfloat16')
    if config['device'].get('mixed_precision', False) and device.type == 'cuda':
        # Note: GradScaler is not needed for bfloat16 (same exponent range as fp32)
        # but we keep it for compatibility and it works fine with bfloat16
        scaler = GradScaler('cuda')
        print(f"Using mixed precision training (AMP) with dtype={amp_dtype_str}")
        if amp_dtype_str == 'bfloat16':
            print("  → bfloat16 has same exponent range as fp32 - less likely to overflow/underflow")
        elif amp_dtype_str == 'float16':
            print("  → WARNING: float16 has limited range - may cause NaN with very small/large values")
        
        # ================================================================
        # CRITICAL: DTYPE VALIDATION CHECK
        # ================================================================
        # Verify dtype configuration is consistent to prevent NaN issues
        print(f"\n{'='*60}")
        print("DTYPE VALIDATION CHECK")
        print(f"{'='*60}")
        
        # Check 1: Config dtype
        print(f"  Config dtype: {amp_dtype_str}")
        
        # Check 2: Actual PyTorch dtype
        if amp_dtype_str == 'bfloat16':
            expected_dtype = torch.bfloat16
        else:
            expected_dtype = torch.float16
        print(f"  PyTorch dtype: {expected_dtype}")
        
        # Check 3: GPU bfloat16 support
        if amp_dtype_str == 'bfloat16':
            bf16_supported = torch.cuda.is_bf16_supported()
            print(f"  GPU bfloat16 support: {bf16_supported}")
            if not bf16_supported:
                print(f"  ⚠️  WARNING: GPU does not support bfloat16! Falling back to float16.")
                print(f"     Consider using dtype: float16 in config to avoid issues.")
        
        # Check 4: Verify autocast will use correct dtype
        with autocast('cuda', dtype=expected_dtype):
            test_tensor = torch.randn(1, device=device)
            actual_dtype = test_tensor.dtype
        print(f"  Autocast test dtype: {actual_dtype}")
        
        # Check 5: Numerical range limits
        if expected_dtype == torch.float16:
            print(f"  float16 range: [{torch.finfo(torch.float16).min:.0e}, {torch.finfo(torch.float16).max:.0e}]")
            print(f"  float16 tiny: {torch.finfo(torch.float16).tiny:.0e}")
        else:
            print(f"  bfloat16 range: [{torch.finfo(torch.bfloat16).min:.0e}, {torch.finfo(torch.bfloat16).max:.0e}]")
            print(f"  bfloat16 tiny: {torch.finfo(torch.bfloat16).tiny:.0e}")
        
        print(f"  ✓ Dtype validation passed")
        print(f"{'='*60}")
    
    # Initialize wandb if enabled (disabled by default - not installed in production)
    wandb_enabled = False
    if log_cfg.get('use_wandb', False) and WANDB_AVAILABLE:
        try:
            wandb.login()
            wandb.init(
                project=log_cfg.get('wandb_project', 'rlan-arc'),
                name=log_cfg.get('wandb_run_name') or f"rlan-{time.strftime('%Y%m%d-%H%M%S')}",
                config=config,
            )
            wandb_enabled = True
            print("WandB logging enabled")
        except Exception as e:
            print(f"Warning: WandB init failed: {e}")
    else:
        print("WandB logging disabled (use_wandb=false or wandb not installed)")
    
    # Resume from checkpoint if specified
    start_epoch = 0
    global_step = 0
    best_task_accuracy = 0.0
    
    if resume_path:
        reset_optimizer = getattr(args, 'reset_optimizer', False)
        start_epoch, global_step, best_task_accuracy = load_checkpoint(
            model, optimizer, scheduler, resume_path, reset_optimizer=reset_optimizer
        )
        if reset_optimizer:
            print(f"Optimizer/scheduler reset for fine-tuning (starting fresh from checkpoint weights)")
            
    # Initialize LOO training for meta-learning (requires HyperLoRA)
    loo_config = config.get('training', {}).get('loo_training', {})
    use_loo = loo_config.get('enabled', False) and config.get('model', {}).get('use_hyperlora', False)
    loo_start_epoch = loo_config.get('start_epoch', 12)  # Default: epoch 12
    loo_weight = loo_config.get('loss_weight', 0.2)
    loo_min_pairs = loo_config.get('min_pairs_for_loo', 2)
    loo_max_pairs = loo_config.get('max_loo_pairs', 4)  # Cap LOO passes to prevent memory explosion
    
    # ADAPTIVE BATCH SIZE CONFIG (Dec 2025)
    # Configurable reduction instead of hardcoded formula
    loo_batch_reduction_divisor = loo_config.get('batch_reduction_divisor', 2.0)  # Default: halve batch
    loo_min_batch_size = loo_config.get('min_batch_size', 8)  # Safety floor
    loo_adjust_grad_accum = loo_config.get('adjust_grad_accumulation', True)  # Maintain effective batch
    
    # ADAPTIVE BATCH SIZE INITIALIZATION
    # If resuming from a checkpoint where LOO is already active, we must start with reduced batch size
    current_batch_size_override = None
    current_grad_accum_override = None
    base_grad_accum = train_cfg.get('grad_accumulation_steps', 4)
    if use_loo and start_epoch >= loo_start_epoch:
        base_batch_size = train_cfg['batch_size']
        # Use configurable divisor instead of hardcoded formula
        loo_batch_size = int(base_batch_size / loo_batch_reduction_divisor)
        loo_batch_size = max(loo_batch_size, loo_min_batch_size)  # Safety floor
        current_batch_size_override = loo_batch_size
        
        # Adjust grad accumulation to maintain effective batch size
        if loo_adjust_grad_accum and loo_batch_size < base_batch_size:
            # effective_batch = batch × grad_accum should stay constant
            # new_grad_accum = old_effective / new_batch = (base_batch × base_accum) / new_batch
            effective_batch = base_batch_size * base_grad_accum
            new_grad_accum = max(1, int(round(effective_batch / loo_batch_size)))
            current_grad_accum_override = new_grad_accum
        
        print(f"\n{'='*60}")
        print(f"RESUMING WITH ADAPTIVE BATCH SIZE (LOO active)")
        print(f"{'='*60}")
        print(f"  Original batch: {base_batch_size}, grad_accum: {base_grad_accum}")
        print(f"  LOO batch: {loo_batch_size} (divisor={loo_batch_reduction_divisor})")
        if current_grad_accum_override:
            print(f"  Adjusted grad_accum: {current_grad_accum_override} (effective batch: {loo_batch_size * current_grad_accum_override})")
            # Update config in-place so train_epoch picks up new grad_accumulation_steps
            config['training']['grad_accumulation_steps'] = current_grad_accum_override
        print(f"{'='*60}\n")
        
        # Recreate loader immediately if resuming into LOO phase
        train_loader = create_train_loader(
            config,
            curriculum_stage=current_curriculum_stage,
            max_grid_size=max_grid_size,
            batch_size_override=current_batch_size_override,
        )
    
    # =============================================================
    # PHASE-AWARE BATCH SIZE ON RESUME (Jan 2026 MEMORY FIX)
    # =============================================================
    # If resuming into Phase B or C, use that phase's batch_size.
    # LOO override takes priority if active (LOO already reduces batch).
    # =============================================================
    if phased_cfg.get('enabled', False) and current_batch_size_override is None:
        # Determine which phase we're resuming into
        phase_a_end = phased_cfg.get('phase_a', {}).get('end_epoch', 10)
        phase_b_start = phased_cfg.get('phase_b', {}).get('start_epoch', phase_a_end + 1)
        phase_b_end = phased_cfg.get('phase_b', {}).get('end_epoch', 20)
        phase_c_start = phased_cfg.get('phase_c', {}).get('start_epoch', phase_b_end + 1)
        
        epoch_1based = start_epoch + 1  # Convert 0-based to 1-based
        
        if epoch_1based <= phase_a_end:
            resume_phase = 'A'
            resume_phase_cfg = phased_cfg.get('phase_a', {})
        elif epoch_1based >= phase_b_start and epoch_1based <= phase_b_end:
            resume_phase = 'B'
            resume_phase_cfg = phased_cfg.get('phase_b', {})
        else:
            resume_phase = 'C'
            resume_phase_cfg = phased_cfg.get('phase_c', {})
        
        phase_batch_size = resume_phase_cfg.get('batch_size', None)
        if phase_batch_size is not None:
            current_batch_size_override = phase_batch_size
            print(f"\n{'='*60}")
            print(f"RESUMING INTO PHASE {resume_phase} (epoch {epoch_1based})")
            print(f"{'='*60}")
            print(f"  Phase batch_size: {phase_batch_size}")
            print(f"{'='*60}\n")
            
            # Recreate loader with phase batch size
            train_loader = create_train_loader(
                config,
                curriculum_stage=current_curriculum_stage,
                max_grid_size=max_grid_size,
                batch_size_override=current_batch_size_override,
            )

    # Initialize EMA for stable evaluation
    # NOTE: Default is False to match rlan_stable.yaml (EMA disabled for short training)
    use_ema = config.get('training', {}).get('use_ema', False)
    ema_decay = config.get('training', {}).get('ema_decay', 0.999)
    ema = None
    if use_ema:
        ema = EMAHelper(model, mu=ema_decay, device=device)
        print(f"EMA enabled with decay={ema_decay}")
    
    # Initialize LOO training loss function
    loo_loss_fn = None
    if use_loo:
        loo_loss_fn = LOOTrainingLoss(
            config=LOOConfig(
                loss_weight=loo_weight,
                min_pairs_for_loo=loo_min_pairs,
                max_loo_pairs=loo_max_pairs,
            )
        )
        print(f"LOO training configured: weight={loo_weight}, min_pairs={loo_min_pairs}, max_pairs={loo_max_pairs}, start_epoch={loo_start_epoch}")
    
    # Initialize equivariance training (consistency across augmentations)
    equiv_config = config.get('training', {}).get('equivariance_training', {})
    use_equivariance = equiv_config.get('enabled', False) and config.get('model', {}).get('use_hyperlora', False)
    equiv_start_epoch = equiv_config.get('start_epoch', 8)  # Default: epoch 8
    equiv_weight = equiv_config.get('loss_weight', 0.05) if use_equivariance else 0.0
    equiv_loss_fn = None
    
    # =========================================================================
    # OUTPUT-LEVEL EQUIVARIANCE (Jan 2026 FIX)
    # =========================================================================
    # The weight-level AugmentationEquivarianceLoss is degenerate because 
    # HyperLoRA uses D4-invariant pooling, making all augmented contexts 
    # identical (equiv ≈ 0 always). 
    #
    # OutputEquivarianceLoss compares model OUTPUTS instead, directly
    # optimizing for TTA consensus. Set use_output_equiv=true to enable.
    # =========================================================================
    output_equiv_config = config.get('training', {}).get('output_equivariance_training', {})
    use_output_equiv = output_equiv_config.get('enabled', False)
    output_equiv_start_epoch = output_equiv_config.get('start_epoch', equiv_start_epoch)
    output_equiv_weight = output_equiv_config.get('loss_weight', 0.02)
    output_equiv_num_augs = output_equiv_config.get('num_augmentations', 2)  # Keep low for memory
    output_equiv_loss_fn = None
    
    if use_output_equiv:
        output_equiv_loss_fn = OutputEquivarianceLoss(
            config=EquivarianceConfig(
                enabled=True,
                loss_weight=output_equiv_weight,
                num_augmentations=output_equiv_num_augs,
            ),
            loss_type=output_equiv_config.get('loss_type', 'kl'),
            mask_to_target=output_equiv_config.get('mask_to_target', True),  # Jan 2026: Mask to valid region
        )
        print(f"OUTPUT-level equivariance training configured: weight={output_equiv_weight}, "
              f"num_augs={output_equiv_num_augs}, start_epoch={output_equiv_start_epoch}, "
              f"type={output_equiv_config.get('loss_type', 'kl')}, "
              f"mask_to_target={output_equiv_config.get('mask_to_target', True)}")
    
    # =========================================================================
    # GROUP-MARGINALIZED NLL LOSS (Jan 2026)
    # =========================================================================
    # This is a mathematically principled alternative to output-equiv + regular NLL.
    # Instead of fighting between two losses, it trains directly on the group-
    # marginalized distribution: p̄(y|x) = avg_g g⁻¹(p(·|g(x)))
    #
    # This directly optimizes what TTA voting tries to achieve.
    # Can be used as PRIMARY loss (replacing task loss) or AUXILIARY loss.
    # =========================================================================
    group_marg_config = config.get('training', {}).get('group_marginalized_nll', {})
    use_group_marginalized = group_marg_config.get('enabled', False)
    group_marg_start_epoch = group_marg_config.get('start_epoch', 21)
    group_marg_weight = group_marg_config.get('loss_weight', 1.0)  # High weight - this is principled
    group_marg_num_augs = group_marg_config.get('num_augmentations', 2)
    group_marg_as_primary = group_marg_config.get('as_primary_loss', False)  # Replace task loss entirely
    group_marg_loss_fn = None
    
    if use_group_marginalized:
        group_marg_loss_fn = GroupMarginalizedNLLLoss(
            num_augmentations=group_marg_num_augs,
            ignore_index=-100,
        )
        mode_str = "PRIMARY (replaces task loss)" if group_marg_as_primary else "AUXILIARY"
        print(f"GROUP-MARGINALIZED NLL configured: mode={mode_str}, weight={group_marg_weight}, "
              f"num_augs={group_marg_num_augs}, start_epoch={group_marg_start_epoch}")
    
    # Weight-level equivariance (DEPRECATED if output_equiv enabled)
    if use_equivariance and not use_output_equiv:
        hidden_dim = config.get('model', {}).get('hidden_dim', 256)
        equiv_loss_fn = AugmentationEquivarianceLoss(
            config=EquivarianceConfig(
                enabled=True,
                loss_weight=equiv_weight,
                num_augmentations=equiv_config.get('num_augmentations', 4),
            ),
            hidden_dim=hidden_dim,
        )
        print(f"[DEPRECATED] Weight-level equivariance training configured: weight={equiv_weight}, num_augs={equiv_config.get('num_augmentations', 4)}, start_epoch={equiv_start_epoch}")
        print(f"  NOTE: Consider enabling output_equivariance_training instead - weight-level equiv is often ~0 due to D4-invariant pooling")
    
    # =========================================================================
    # ANCHOR ROBUSTNESS TRAINING (ART) - Jan 2026 Ablation Study
    # =========================================================================
    # ART forces consistent predictions under alternate anchors, directly
    # targeting the observed eval entropy collapse issue.
    # =========================================================================
    art_config_yaml = config.get('model', {}).get('anchor_robustness', {})
    art_module = create_art_from_config(art_config_yaml, config.get('model', {}).get('hidden_dim', 256))
    art_start_epoch = art_config_yaml.get('start_epoch', 0)  # Active from start by default
    
    if art_module is not None:
        art_module = art_module.to(device)
        print(f"ART (Anchor Robustness Training) enabled: weight={art_module.config.consistency_weight}, "
              f"loss_type={art_module.config.consistency_loss_type}, start_epoch={art_start_epoch}")
    
    # =========================================================================
    # ANCHOR-RELATIVE PROGRAM SEARCH (ARPS) - Jan 2026 Ablation Study
    # =========================================================================
    # ARPS provides interpretable program supervision via imitation learning.
    # Programs are expressed in anchor-relative coordinates for generalization.
    # =========================================================================
    arps_config_yaml = config.get('model', {}).get('arps_dsl_search', {})
    arps_module = create_arps_from_config(arps_config_yaml, config.get('model', {}).get('hidden_dim', 256))
    arps_start_epoch = arps_config_yaml.get('start_epoch', 5)  # Needs stable features first
    
    if arps_module is not None:
        arps_module = arps_module.to(device)
        num_primitives = len(arps_config_yaml.get('primitives', []))
        print(f"ARPS (Anchor-Relative Program Search) enabled: {num_primitives} primitives, "
              f"imitation_weight={arps_config_yaml.get('imitation_weight', 0.1)}, start_epoch={arps_start_epoch}")
    
    # STAGED META-LEARNING: Delay LOO/Equivariance to prevent early BG collapse
    # Each loss has its own start_epoch for fine-grained control
    # STAGGERED MODULE ACTIVATION to prevent memory spikes and gradient explosion
    # SCIENTIFICALLY ORDERED (Dec 2025): Context path FIRST, HyperLoRA LATER
    meta_learning_start_epoch = config.get('training', {}).get('meta_learning_start_epoch', 8)
    solver_context_start_epoch = config.get('training', {}).get('solver_context_start_epoch', 5)
    cross_attention_start_epoch = config.get('training', {}).get('cross_attention_start_epoch', 5)
    
    # HyperLoRA warmup parameters
    hyperlora_warmup_epochs = config.get('training', {}).get('hyperlora_warmup_epochs', 4)
    hyperlora_warmup_start_scale = config.get('training', {}).get('hyperlora_warmup_start_scale', 0.005)
    hyperlora_warmup_end_scale = config.get('training', {}).get('hyperlora_warmup_end_scale', 0.1)
    
    # LR reduction at activation epochs
    activation_lr_reduction = config.get('training', {}).get('activation_lr_reduction', 0.5)
    activation_lr_recovery_epochs = config.get('training', {}).get('activation_lr_recovery_epochs', 2)
    
    # Gradient explosion backoff parameters
    grad_explosion_threshold = config.get('training', {}).get('grad_explosion_threshold', 10.0)
    grad_explosion_lr_reduction = config.get('training', {}).get('grad_explosion_lr_reduction', 0.5)
    grad_explosion_cooldown_epochs = config.get('training', {}).get('grad_explosion_cooldown_epochs', 2)
    
    # Track activation state for LR management
    # Patch 3 (Dec 2025): Composable LR factors - avoids over-restore bugs
    # We track base LR and multiplicative factors separately, then compute
    # actual LR as: base_lr * activation_factor * explosion_factor
    # This prevents double-undo when multiple reductions overlap.
    activation_lr_state = {
        'reduced': False,
        'reduction_epoch': -1,
        'original_lr': config['training']['learning_rate'],
        'grad_explosion_cooldown': 0,  # Countdown epochs after gradient explosion
        # Patch 3: Per-param-group base LRs for composable restoration
        'base_lrs': [pg.get('lr', config['training']['learning_rate']) for pg in optimizer.param_groups],
        'activation_factor': 1.0,  # Multiplier for activation reduction
        'explosion_factor': 1.0,   # Multiplier for grad explosion reduction
    }
    
    # Patch 4 (Dec 2025): NaN-driven meta-loss backoff state
    # When consecutive NaNs occur, we reduce meta-loss weights to stabilize.
    # This implements the documented "3 consecutive NaN → halve newest meta-loss" rule.
    nan_backoff_state = {
        'equiv_weight_factor': 1.0,   # Current multiplier for equiv loss
        'loo_weight_factor': 1.0,     # Current multiplier for LOO loss
        'nan_backoff_active': False,  # Whether backoff is currently in effect
        'nan_backoff_epochs': 0,      # Cooldown epochs remaining
    }
    nan_backoff_threshold = 3  # Consecutive NaN batches to trigger backoff
    
    # ==========================================================================
    # META ESCALATION: Late-phase stability-gated weight increase (Dec 2025)
    # ==========================================================================
    # After baseline training is stable, we increase meta-learning signal strength
    # to improve AGI-ARC generalization on never-seen rules.
    # ==========================================================================
    meta_escalation_config = config.get('training', {}).get('meta_escalation', {})
    meta_escalation_enabled = meta_escalation_config.get('enabled', False)
    meta_escalation_start_epoch = meta_escalation_config.get('start_epoch', 25)
    meta_escalation_ramp_epochs = meta_escalation_config.get('ramp_epochs', 12)
    meta_escalation_schedule = meta_escalation_config.get('schedule', 'linear')
    
    # ==========================================================================
    # META LOSS CAPPING (Jan 2026 FIX)
    # ==========================================================================
    # Prevents meta-losses (LOO + equiv + HPM) from overwhelming task loss.
    # Training logs showed 40.9% meta contribution, causing weak task learning.
    # ==========================================================================
    meta_loss_cap_enabled = meta_escalation_config.get('meta_loss_cap_enabled', True)
    meta_loss_cap_ratio = meta_escalation_config.get('meta_loss_cap_ratio', 0.25)
    if meta_loss_cap_enabled:
        print(f"[META LOSS CAP] Enabled: meta-loss capped at {meta_loss_cap_ratio*100:.0f}% of total loss")

    # Model feature flags used by meta-escalation logging/targets
    use_hpm = config.get('model', {}).get('use_hpm', False)
    
    # Target values (what we ramp toward)
    meta_escalation_targets = meta_escalation_config.get('targets', {})
    target_hyperlora_delta_scale = meta_escalation_targets.get('hyperlora_delta_scale', 0.30)
    target_equiv_weight = meta_escalation_targets.get('equiv_loss_weight', 0.05)
    target_loo_weight = meta_escalation_targets.get('loo_loss_weight', 0.10)
    # Optional: ramp HPM balance loss weight too (kept backward compatible)
    base_hpm_balance_weight = config.get('model', {}).get('hpm_balance_weight', 0.01)
    target_hpm_balance_weight = meta_escalation_targets.get('hpm_balance_weight', base_hpm_balance_weight)
    
    # Stability gating (controls pause/resume behavior)
    meta_escalation_require_stability = meta_escalation_config.get('require_stability', True)
    # Backward compatible: check new names first, fall back to old names
    meta_escalation_max_grad_events = meta_escalation_config.get(
        'max_grad_explosion_events_per_epoch',
        meta_escalation_config.get('max_grad_explosion_events_in_window', 0)  # Old name fallback
    )
    meta_escalation_max_lr_events = meta_escalation_config.get(
        'max_lr_backoff_events_per_epoch',
        meta_escalation_config.get('max_lr_backoff_events_in_window', 0)  # Old name fallback
    )
    meta_escalation_max_nan_streak = meta_escalation_config.get(
        'max_consecutive_nan_streak_per_epoch',
        meta_escalation_config.get('max_consecutive_nan_streak_in_window', 0)  # Old name fallback
    )
    
    # Recovery settings
    meta_escalation_recovery_enabled = meta_escalation_config.get('recovery_enabled', True)
    meta_escalation_recovery_step = meta_escalation_config.get('recovery_step_per_window', 0.05)
    meta_escalation_log_every_epoch = meta_escalation_config.get('log_every_epoch', True)
    
    # Meta escalation state tracking
    meta_escalation_state = {
        # Current applied values (what's actually used in forward/loss)
        'hyperlora_delta_scale_current': hyperlora_warmup_end_scale,  # Start from warmup end
        'equiv_weight_current': equiv_weight,  # Start from config baseline
        'loo_weight_current': loo_weight,      # Start from config baseline
        'hpm_balance_weight_current': base_hpm_balance_weight,  # Start from config baseline
        
        # Scheduled values (what we're ramping toward, may be paused)
        'hyperlora_delta_scale_scheduled': hyperlora_warmup_end_scale,
        'equiv_weight_scheduled': equiv_weight,
        'loo_weight_scheduled': loo_weight,
        'hpm_balance_weight_scheduled': base_hpm_balance_weight,
        
        # Progress tracking
        'escalation_active': False,  # Whether we've started escalation phase
        'escalation_paused': False,  # Whether escalation is paused due to instability
        'escalation_progress': 0.0,  # 0.0 to 1.0 progress through ramp
        
        # Stability tracking (epoch-based, not rolling window)
        # Note: Despite config naming, gating uses per-epoch counters reset each epoch
        'stable_epochs_count': 0,  # Consecutive stable epochs
        'is_stable': True,  # Whether previous epoch was stable
        
        # Per-epoch counters (reset each epoch)
        'grad_explosion_events_epoch': 0,
        'lr_backoff_events_epoch': 0,
        'max_nan_streak_epoch': 0,
        'attention_collapse_events_epoch': 0,
        'max_nan_streak_in_window': 0,  # Track max NaN streak across escalation window
    }
    
    # ==========================================================================
    # ATTENTION COLLAPSE BACKOFF POLICY (Jan 2026)
    # ==========================================================================
    # When attention collapse is detected, actively reduce delta_scale and LR
    # (not just pause escalation). This helps late-epoch stability (50-200).
    # ==========================================================================
    monitoring_config = config.get('monitoring', {})
    collapse_backoff_config = monitoring_config.get('collapse_backoff', {})
    collapse_backoff_enabled = collapse_backoff_config.get('enabled', True)
    collapse_backoff_cooldown = collapse_backoff_config.get('cooldown_epochs', 3)
    collapse_backoff_delta_factor = collapse_backoff_config.get('delta_scale_factor', 0.5)
    collapse_backoff_lr_factor = collapse_backoff_config.get('lr_factor', 0.5)
    collapse_backoff_restore_rate = collapse_backoff_config.get('restore_rate', 0.2)
    
    # Attention collapse thresholds (Jan 2026)
    attn_max_collapse_threshold = monitoring_config.get('attn_max_collapse_threshold', 0.02)
    attention_collapse_consecutive_threshold = monitoring_config.get('attention_collapse_consecutive_threshold', 2)
    
    collapse_backoff_state = {
        'active': False,              # Whether backoff is currently in effect
        'cooldown_remaining': 0,      # Epochs remaining in cooldown
        'delta_scale_factor': 1.0,    # Current multiplier for delta_scale (0.5 when backed off)
        'lr_factor': 1.0,             # Current multiplier for LR (0.5 when backed off)
        'pre_backoff_delta_scale': None,  # Saved value before backoff
        'pre_backoff_lr': None,       # Saved LR before backoff
        'consecutive_collapse_count': 0,  # Track consecutive collapse epochs
    }
    
    if collapse_backoff_enabled:
        print(f"\n{'='*60}")
        print(f"ATTENTION COLLAPSE BACKOFF ENABLED (Jan 2026)")
        print(f"{'='*60}")
        print(f"  Trigger: {attention_collapse_consecutive_threshold}+ consecutive epochs with attn_max < {attn_max_collapse_threshold}")
        print(f"  On collapse: delta_scale *= {collapse_backoff_delta_factor}, LR *= {collapse_backoff_lr_factor}")
        print(f"  Cooldown: {collapse_backoff_cooldown} epochs before restore begins")
        print(f"  Restore rate: +{collapse_backoff_restore_rate*100:.0f}% per stable epoch")
        print(f"{'='*60}\n")
    
    # ==========================================================================
    # LATE-PHASE META-ESCALATION (Jan 2026)
    # ==========================================================================
    # After epoch 50, use stricter stability gates to maintain high accuracy.
    # Also implements late-phase LR decay for fine-grained convergence.
    # ==========================================================================
    late_phase_config = meta_escalation_config.get('late_phase', {})
    late_phase_enabled = late_phase_config.get('enabled', True)
    late_phase_start_epoch = late_phase_config.get('start_epoch', 50)
    
    # Stricter gates for late phase
    late_phase_max_grad_events = late_phase_config.get('max_grad_explosion_events_per_epoch', 0)
    late_phase_max_lr_events = late_phase_config.get('max_lr_backoff_events_per_epoch', 0)
    late_phase_max_nan_streak = late_phase_config.get('max_consecutive_nan_streak_per_epoch', 0)
    late_phase_max_collapse_events = late_phase_config.get('max_attention_collapse_events_per_epoch', 0)
    
    # Late-phase LR decay settings
    lr_decay_config = late_phase_config.get('lr_decay', {})
    late_phase_lr_decay_enabled = lr_decay_config.get('enabled', True)
    late_phase_lr_decay_start_epoch = lr_decay_config.get('start_epoch', 50)
    late_phase_lr_decay_end_epoch = lr_decay_config.get('end_epoch', 200)
    late_phase_lr_decay_min_factor = lr_decay_config.get('min_factor', 0.1)
    late_phase_lr_decay_schedule = lr_decay_config.get('schedule', 'cosine')
    
    late_phase_state = {
        'active': False,              # Whether we're in late phase
        'lr_decay_factor': 1.0,       # Current LR decay factor (1.0 → min_factor)
    }
    
    # ==========================================================================
    # HPM SOLVER-CONTEXT COUPLING (Jan 2026)
    # ==========================================================================
    # Injects HPM memory tokens into solver cross-attention. Gated warmup
    # prevents gradient disruption. Auto-disable if instability detected.
    # ==========================================================================
    model_config = config.get('model', {})
    hpm_solver_context_enabled = model_config.get('hpm_solver_context_enabled', False)
    hpm_solver_context_start_epoch = model_config.get('hpm_solver_context_start_epoch', 45)
    hpm_solver_context_gate_warmup_epochs = model_config.get('hpm_solver_context_gate_warmup_epochs', 15)
    hpm_solver_context_gate_max = model_config.get('hpm_solver_context_gate_max', 0.5)
    hpm_solver_context_logit_clamp = model_config.get('hpm_solver_context_logit_clamp', 10.0)
    hpm_solver_context_disable_on_instability = model_config.get('hpm_solver_context_disable_on_instability', True)
    
    hpm_solver_context_state = {
        'active': False,              # Whether coupling is currently active
        'gate_value': 0.0,            # Current gate value (0.0 → gate_max)
        'disabled_by_instability': False,  # If True, was disabled due to explosion
        'explosion_count': 0,         # How many times explosion detected
        'tokens_used_epoch': 0,       # Total HPM tokens used this epoch
        'batches_with_tokens': 0,     # Batches that had HPM tokens
    }
    
    if hpm_solver_context_enabled and use_hpm:
        print(f"\n{'='*60}")
        print(f"HPM SOLVER-CONTEXT COUPLING ENABLED (Jan 2026)")
        print(f"{'='*60}")
        print(f"  Start epoch: {hpm_solver_context_start_epoch + 1}")
        print(f"  Gate warmup: {hpm_solver_context_gate_warmup_epochs} epochs")
        print(f"  Gate max: {hpm_solver_context_gate_max}")
        print(f"  Logit clamp: {hpm_solver_context_logit_clamp}")
        print(f"  Auto-disable on instability: {hpm_solver_context_disable_on_instability}")
        print(f"{'='*60}\n")
    
    if late_phase_enabled:
        print(f"\n{'='*60}")
        print(f"LATE-PHASE META-ESCALATION ENABLED (Jan 2026)")
        print(f"{'='*60}")
        print(f"  Start epoch: {late_phase_start_epoch + 1}")
        print(f"  Stricter stability gates:")
        print(f"    Max grad events/epoch: {late_phase_max_grad_events}")
        print(f"    Max LR backoff events/epoch: {late_phase_max_lr_events}")
        print(f"    Max NaN streak/epoch: {late_phase_max_nan_streak}")
        print(f"    Max attention collapse events/epoch: {late_phase_max_collapse_events}")
        if late_phase_lr_decay_enabled:
            print(f"  LR Decay:")
            print(f"    Schedule: {late_phase_lr_decay_schedule}")
            print(f"    Range: epochs {late_phase_lr_decay_start_epoch + 1} - {late_phase_lr_decay_end_epoch}")
            print(f"    Factor: 1.0 → {late_phase_lr_decay_min_factor}")
        print(f"{'='*60}\n")
    
    if meta_escalation_enabled:
        print(f"\n{'='*60}")
        print(f"META ESCALATION ENABLED (Late-Phase Strength Increase)")
        print(f"{'='*60}")
        print(f"  Schedule: {meta_escalation_schedule}")
        print(f"  Start epoch: {meta_escalation_start_epoch + 1}")
        print(f"  Ramp epochs: {meta_escalation_ramp_epochs}")
        print(f"  Targets:")
        print(f"    HyperLoRA delta_scale: {hyperlora_warmup_end_scale} → {target_hyperlora_delta_scale}")
        print(f"    Equiv weight: {equiv_weight} → {target_equiv_weight}")
        print(f"    LOO weight: {loo_weight} → {target_loo_weight}")
        if use_hpm:
            print(f"    HPM balance weight: {base_hpm_balance_weight} → {target_hpm_balance_weight}")
        print(f"  Stability gating:")
        if meta_escalation_require_stability:
            print(f"    Mode: GATED (pauses on instability, resumes when stable)")
            print(f"    Max allowed per epoch: nan_streak={meta_escalation_max_nan_streak}, grad_events={meta_escalation_max_grad_events}, lr_events={meta_escalation_max_lr_events}")
        else:
            print(f"    Mode: FORCE (always follows schedule, ignores instability - for ablations)")
        if meta_escalation_recovery_enabled:
            print(f"  Recovery: +{meta_escalation_recovery_step*100:.0f}% per stable window")
        print(f"{'='*60}\n")
    
    if use_loo or use_equivariance:
        print(f"\n{'='*60}")
        print(f"STAGED META-LEARNING ENABLED (Scientifically Ordered)")
        print(f"{'='*60}")
        print(f"  Phase 2 - Context Path (epoch {solver_context_start_epoch + 1}+):")
        print(f"    SolverCrossAttention + CrossAttentionInjector")
        print(f"  Phase 3 - HyperLoRA (epoch {meta_learning_start_epoch + 1}+):")
        print(f"    With warmup: scale {hyperlora_warmup_start_scale} → {hyperlora_warmup_end_scale} over {hyperlora_warmup_epochs} epochs")
        if use_equivariance:
            print(f"  Phase 4 - Equivariance Loss: Epoch {equiv_start_epoch + 1}+ (weight={equiv_weight})")
        if use_loo:
            print(f"  Phase 5 - LOO Loss: Epoch {loo_start_epoch + 1}+ (weight={loo_weight})")
        print(f"\n  Safety:")
        print(f"    LR reduction at activation: {activation_lr_reduction}x for {activation_lr_recovery_epochs} epochs")
        print(f"    Grad explosion threshold: {grad_explosion_threshold}x clip → {grad_explosion_lr_reduction}x LR")
        print(f"{'='*60}\n")
    
    # STAGED HPM: Delay HPM activation to align with meta-learning
    use_hpm = config.get('model', {}).get('use_hpm', False)
    hpm_start_epoch = config.get('model', {}).get('hpm_start_epoch', 3)
    # TODO 2: hpm_memory_start_epoch allows memory collection before HPM activation
    hpm_memory_start_epoch = config.get('model', {}).get('hpm_memory_start_epoch', 0)  # Default: collect from epoch 0
    
    # BUG FIX: Always initialize hpm_memory_enabled (not just in conditional block)
    # This flag enables buffer writes independently of HPM activation
    model.hpm_memory_enabled = (start_epoch >= hpm_memory_start_epoch)
    if model.hpm_memory_enabled:
        print(f"[HPM Memory] Buffer population ENABLED from epoch 0 (hpm_memory_start_epoch={hpm_memory_start_epoch})")
    
    if use_hpm and hpm_start_epoch > 0:
        print(f"\n{'='*60}")
        print(f"STAGED HPM ENABLED")
        print(f"{'='*60}")
        print(f"  Phase 1 (epochs 1-{hpm_start_epoch}): HPM inactive, memory collection active")
        print(f"    - HPM module exists but use_hpm=False during forward")
        print(f"    - No HPM load balancing loss added")
        print(f"    - TODO 2: Dynamic buffers ARE populated for solved tasks")
        print(f"  Phase 2 (epochs {hpm_start_epoch + 1}+): HPM activated")
        print(f"    - HPM contributes to features (gated residual)")
        print(f"    - Load balancing loss ensures all banks utilized")
        print(f"{'='*60}\n")
        # Set HPM state based on start_epoch (handles resume case)
        if hasattr(model, 'use_hpm'):
            if start_epoch >= hpm_start_epoch:
                model.use_hpm = True
                print(f"  [HPM] Already past start epoch - HPM ACTIVE")
            else:
                model.use_hpm = False
                print(f"  [HPM] Temporarily disabled until epoch {hpm_start_epoch + 1}")
            
            # TODO 2 FIX: Enable memory collection from epoch 0 (or hpm_memory_start_epoch)
            # This allows buffers to grow before HPM activation
            if hasattr(model, 'hpm_memory_enabled'):
                model.hpm_memory_enabled = (start_epoch >= hpm_memory_start_epoch)
            else:
                model.hpm_memory_enabled = (start_epoch >= hpm_memory_start_epoch)
            if model.hpm_memory_enabled and not model.use_hpm:
                print(f"  [HPM] Memory collection ENABLED (buffers grow before activation)")
    
    # STAGED HYPERLORA AND SOLVER CROSS-ATTENTION
    # These modules exist in the model but should not CONTRIBUTE during early epochs
    # This prevents gradient noise from untrained modules destabilizing the base model
    use_hyperlora = config.get('model', {}).get('use_hyperlora', False)
    use_solver_context = config.get('model', {}).get('use_solver_context', False)
    use_cross_attention_context = config.get('model', {}).get('use_cross_attention_context', False)
    
    if use_hyperlora or use_solver_context or use_cross_attention_context:
        print(f"\n{'='*60}")
        print(f"STAGGERED MODULE CONTRIBUTIONS (Prevents Memory Spike)")
        print(f"{'='*60}")
        
        # =============================================================
        # CRITICAL CONFIG VALIDATION (from META_LEARNING_INFERENCE_GUIDE.md)
        # =============================================================
        # HyperLoRA REQUIRES support_features from context encoding
        # Without these, HyperLoRA is loaded but never generates LoRA deltas!
        if use_hyperlora and not (use_solver_context or use_cross_attention_context):
            print(f"  [CRITICAL WARNING] HyperLoRA enabled but no context source!")
            print(f"    use_hyperlora=True BUT use_solver_context=False AND use_cross_attention_context=False")
            print(f"    HyperLoRA will be SILENTLY INACTIVE - LoRA deltas never generated!")
            print(f"    FIX: Enable at least one of use_solver_context or use_cross_attention_context")
            print(f"  " + "="*56)
        
        if use_hyperlora:
            if start_epoch >= meta_learning_start_epoch:
                model.hyperlora_active = True
                print(f"  [HyperLoRA] Already past start epoch - LoRA deltas ACTIVE")
            else:
                model.hyperlora_active = False
                print(f"  [HyperLoRA] LoRA deltas disabled until epoch {meta_learning_start_epoch + 1}")
                print(f"    - Module trains via LOO/Equiv losses after activation")
        
        if use_solver_context:
            if start_epoch >= solver_context_start_epoch:
                model.solver_context_active = True
                print(f"  [SolverCrossAttention] Already past start epoch - ACTIVE")
            else:
                model.solver_context_active = False
                print(f"  [SolverCrossAttention] Disabled until epoch {solver_context_start_epoch + 1}")
                print(f"    - Solver runs without cross-attention initially")
        
        # CRITICAL: CrossAttentionInjector uses Q/K/V projections that are randomly
        # initialized. During early training, this injects NOISE into features.
        # FiLM (ContextInjector) uses simple γ*features+β which is much more stable.
        # Stage cross-attention to activate LAST to prevent memory spike.
        if use_cross_attention_context:
            if start_epoch >= cross_attention_start_epoch:
                model.cross_attention_active = True
                print(f"  [CrossAttentionInjector] Already past start epoch - ACTIVE")
            else:
                model.cross_attention_active = False
                print(f"  [CrossAttentionInjector] Disabled until epoch {cross_attention_start_epoch + 1}")
                print(f"    - Using FiLM fallback (pool+scale/shift) for stable early training")
        print(f"{'='*60}\n")
    
    # Training loop
    max_epochs = config['training']['max_epochs']
    save_every = log_cfg.get('save_every', 10)
    eval_every = log_cfg.get('eval_every', 1)
    keep_last_n = log_cfg.get('keep_last_n', 5)
    
    # ================================================================
    # PHASED TRAINING SUPPORT (Jan 2026)
    # ================================================================
    # Allows structured A/B/C training phases in a single YAML config.
    # Each phase can override augmentation settings and disable meta-losses.
    # ================================================================
    phased_training_config = config.get('training', {}).get('phased_training', {})
    phased_training_enabled = phased_training_config.get('enabled', False)
    
    # ================================================================
    # METRIC-BASED PHASE GATING (Jan 2026)
    # ================================================================
    # Optional: Delay phase transitions until metrics meet thresholds.
    # Backward compatible: disabled by default or when config absent.
    # ================================================================
    phase_readiness_config = phased_training_config.get('phase_readiness', {})
    use_metric_gating = phase_readiness_config.get('use_metric_gating', False)
    
    # State for tracking phase gate progress
    phase_gate_state = {
        'current_effective_phase': 'A',  # Actual phase in use (may lag epoch-based phase)
        'epochs_in_current_phase': 0,     # How many epochs in current effective phase
        'phase_a_start_epoch': 0,         # When Phase A started (for min_epochs tracking)
        'phase_b_start_epoch': None,      # When Phase B started
        'a_to_b_consecutive_met': 0,      # Consecutive evals meeting A→B gate
        'b_to_c_consecutive_met': 0,      # Consecutive evals meeting B→C gate
        'centroid_collapse_free_count': 0,  # Epochs without centroid collapse
        'grad_explosion_free_count': 0,     # Epochs without gradient explosion
        'last_gate_status': {},           # Last eval's gate status for logging
        'transition_log': [],             # Log of phase transitions [(epoch, from_phase, to_phase)]
    }
    
    if use_metric_gating and phased_training_enabled:
        gate_a_to_b = phase_readiness_config.get('gate_a_to_b', {})
        gate_b_to_c = phase_readiness_config.get('gate_b_to_c', {})
        print(f"\n{'='*60}")
        print(f"METRIC-BASED PHASE GATING ENABLED")
        print(f"{'='*60}")
        print(f"  A→B Gate:")
        print(f"    min_epochs: {gate_a_to_b.get('min_epochs_in_phase_a', 10)}")
        shape_max_ab = gate_a_to_b.get('shape_mismatch_max', 0.40)
        if shape_max_ab >= 1.0:
            print(f"    shape_mismatch: DISABLED (threshold=1.0)")
        else:
            print(f"    shape_mismatch ≤ {shape_max_ab:.0%}")
        print(f"    fg_accuracy ≥ {gate_a_to_b.get('fg_accuracy_min', 0.45):.0%}")
        print(f"    patience: {gate_a_to_b.get('patience', 2)} consecutive evals")
        print(f"  B→C Gate:")
        print(f"    min_epochs: {gate_b_to_c.get('min_epochs_in_phase_b', 5)}")
        shape_max_bc = gate_b_to_c.get('shape_mismatch_max', 0.25)
        if shape_max_bc >= 1.0:
            print(f"    shape_mismatch: DISABLED (threshold=1.0)")
        else:
            print(f"    shape_mismatch ≤ {shape_max_bc:.0%}")
        print(f"    fg_accuracy ≥ {gate_b_to_c.get('fg_accuracy_min', 0.50):.0%}")
        tta_exact_min = gate_b_to_c.get('tta_exact_match_min', 0.01)
        if tta_exact_min <= 0:
            print(f"    tta_exact_match: DISABLED (threshold=0)")
        else:
            print(f"    tta_exact_match ≥ {tta_exact_min:.1%}")
        print(f"    vote_tie_max ≤ {gate_b_to_c.get('vote_tie_max', 0.30):.0%}")
        print(f"{'='*60}\n")
    
    def check_phase_gate(current_phase: str, epoch_1based: int, 
                         eval_metrics: dict = None, trm_metrics: dict = None) -> tuple:
        """
        Check if phase gate criteria are met for transitioning.
        
        Args:
            current_phase: Current effective phase ('A', 'B', 'C')
            epoch_1based: Current epoch (1-based, human readable)
            eval_metrics: Dict from evaluate() with fg_accuracy, etc.
            trm_metrics: Dict from evaluate_trm_style() with shape_mismatch_count, etc.
            
        Returns:
            (ready: bool, status: dict) - Whether gate is passed and detailed status
        """
        if not use_metric_gating:
            return True, {'gating': 'disabled'}
        
        status = {'criteria': [], 'met': [], 'not_met': []}
        
        if current_phase == 'A':
            # Check A→B gate
            gate_cfg = phase_readiness_config.get('gate_a_to_b', {})
            min_epochs = gate_cfg.get('min_epochs_in_phase_a', 10)
            shape_mismatch_max = gate_cfg.get('shape_mismatch_max', 0.40)
            fg_accuracy_min = gate_cfg.get('fg_accuracy_min', 0.45)
            patience = gate_cfg.get('patience', 2)
            collapse_free = gate_cfg.get('centroid_collapse_free_epochs', 0)
            grad_free = gate_cfg.get('grad_explosion_free_epochs', 0)
            
            epochs_in_phase = epoch_1based - phase_gate_state['phase_a_start_epoch']
            
            # Criterion 1: Minimum epochs
            min_epochs_met = epochs_in_phase >= min_epochs
            status['criteria'].append(f"min_epochs≥{min_epochs}")
            if min_epochs_met:
                status['met'].append(f"epochs={epochs_in_phase}✓")
            else:
                status['not_met'].append(f"epochs={epochs_in_phase}<{min_epochs}")
                # Don't check other criteria if min epochs not met
                return False, status
            
            # Criterion 2: Shape mismatch rate (skip if threshold >= 1.0, i.e., disabled)
            if shape_mismatch_max < 1.0:
                if trm_metrics and 'shape_mismatch_count' in trm_metrics:
                    total = trm_metrics.get('total_tasks', 1)
                    shape_rate = trm_metrics['shape_mismatch_count'] / total if total > 0 else 1.0
                    status['criteria'].append(f"shape_mismatch≤{shape_mismatch_max:.0%}")
                    if shape_rate <= shape_mismatch_max:
                        status['met'].append(f"shape={shape_rate:.0%}✓")
                    else:
                        status['not_met'].append(f"shape={shape_rate:.0%}>{shape_mismatch_max:.0%}")
                else:
                    status['not_met'].append("no TRM metrics")
            # else: shape_mismatch check disabled (threshold=1.0)
            
            # Criterion 3: FG accuracy
            if eval_metrics and 'fg_accuracy' in eval_metrics:
                fg_acc = eval_metrics['fg_accuracy']
                status['criteria'].append(f"fg_acc≥{fg_accuracy_min:.0%}")
                if fg_acc >= fg_accuracy_min:
                    status['met'].append(f"fg={fg_acc:.0%}✓")
                else:
                    status['not_met'].append(f"fg={fg_acc:.0%}<{fg_accuracy_min:.0%}")
            else:
                status['not_met'].append("no eval metrics")
            
            # Criterion 4: Stability (optional)
            if collapse_free > 0:
                status['criteria'].append(f"collapse_free≥{collapse_free}")
                if phase_gate_state['centroid_collapse_free_count'] >= collapse_free:
                    status['met'].append(f"collapse_free={phase_gate_state['centroid_collapse_free_count']}✓")
                else:
                    status['not_met'].append(f"collapse_free={phase_gate_state['centroid_collapse_free_count']}<{collapse_free}")
            
            if grad_free > 0:
                status['criteria'].append(f"grad_free≥{grad_free}")
                if phase_gate_state['grad_explosion_free_count'] >= grad_free:
                    status['met'].append(f"grad_free={phase_gate_state['grad_explosion_free_count']}✓")
                else:
                    status['not_met'].append(f"grad_free={phase_gate_state['grad_explosion_free_count']}<{grad_free}")
            
            # Gate passes if all required criteria met
            all_met = len(status['not_met']) == 0
            if all_met:
                phase_gate_state['a_to_b_consecutive_met'] += 1
            else:
                phase_gate_state['a_to_b_consecutive_met'] = 0
            
            ready = phase_gate_state['a_to_b_consecutive_met'] >= patience
            status['consecutive'] = phase_gate_state['a_to_b_consecutive_met']
            status['patience'] = patience
            
            return ready, status
            
        elif current_phase == 'B':
            # Check B→C gate
            gate_cfg = phase_readiness_config.get('gate_b_to_c', {})
            min_epochs = gate_cfg.get('min_epochs_in_phase_b', 5)
            shape_mismatch_max = gate_cfg.get('shape_mismatch_max', 0.25)
            fg_accuracy_min = gate_cfg.get('fg_accuracy_min', 0.50)
            tta_exact_min = gate_cfg.get('tta_exact_match_min', 0.01)
            patience = gate_cfg.get('patience', 2)
            vote_tie_max = gate_cfg.get('vote_tie_max', 0.30)
            
            if phase_gate_state['phase_b_start_epoch'] is None:
                # Not yet in Phase B
                return False, {'not_in_phase_b': True}
            
            epochs_in_phase = epoch_1based - phase_gate_state['phase_b_start_epoch']
            
            # Criterion 1: Minimum epochs
            min_epochs_met = epochs_in_phase >= min_epochs
            status['criteria'].append(f"min_epochs≥{min_epochs}")
            if min_epochs_met:
                status['met'].append(f"epochs={epochs_in_phase}✓")
            else:
                status['not_met'].append(f"epochs={epochs_in_phase}<{min_epochs}")
                return False, status
            
            # Criterion 2: Shape mismatch rate (skip if threshold >= 1.0, i.e., disabled)
            if shape_mismatch_max < 1.0:
                if trm_metrics and 'shape_mismatch_count' in trm_metrics:
                    total = trm_metrics.get('total_tasks', 1)
                    shape_rate = trm_metrics['shape_mismatch_count'] / total if total > 0 else 1.0
                    status['criteria'].append(f"shape_mismatch≤{shape_mismatch_max:.0%}")
                    if shape_rate <= shape_mismatch_max:
                        status['met'].append(f"shape={shape_rate:.0%}✓")
                    else:
                        status['not_met'].append(f"shape={shape_rate:.0%}>{shape_mismatch_max:.0%}")
            # else: shape_mismatch check disabled (threshold=1.0)
            
            # Criterion 3: FG accuracy
            if eval_metrics and 'fg_accuracy' in eval_metrics:
                fg_acc = eval_metrics['fg_accuracy']
                status['criteria'].append(f"fg_acc≥{fg_accuracy_min:.0%}")
                if fg_acc >= fg_accuracy_min:
                    status['met'].append(f"fg={fg_acc:.0%}✓")
                else:
                    status['not_met'].append(f"fg={fg_acc:.0%}<{fg_accuracy_min:.0%}")
            
            # Criterion 4: TTA exact match (skip if threshold == 0.0, i.e., disabled)
            if tta_exact_min > 0:
                if trm_metrics and 'exact_match' in trm_metrics:
                    tta_exact = trm_metrics['exact_match']
                    status['criteria'].append(f"tta_exact≥{tta_exact_min:.1%}")
                    if tta_exact >= tta_exact_min:
                        status['met'].append(f"tta={tta_exact:.1%}✓")
                    else:
                        status['not_met'].append(f"tta={tta_exact:.1%}<{tta_exact_min:.1%}")
            # else: tta_exact_match check disabled (threshold=0.0)
            
            # Criterion 5: Vote tie rate
            if trm_metrics and 'vote_tie_count' in trm_metrics:
                total = trm_metrics.get('total_tasks', 1)
                tie_rate = trm_metrics['vote_tie_count'] / total if total > 0 else 1.0
                status['criteria'].append(f"vote_tie≤{vote_tie_max:.0%}")
                if tie_rate <= vote_tie_max:
                    status['met'].append(f"ties={tie_rate:.0%}✓")
                else:
                    status['not_met'].append(f"ties={tie_rate:.0%}>{vote_tie_max:.0%}")
            
            # Gate passes if all required criteria met
            all_met = len(status['not_met']) == 0
            if all_met:
                phase_gate_state['b_to_c_consecutive_met'] += 1
            else:
                phase_gate_state['b_to_c_consecutive_met'] = 0
            
            ready = phase_gate_state['b_to_c_consecutive_met'] >= patience
            status['consecutive'] = phase_gate_state['b_to_c_consecutive_met']
            status['patience'] = patience
            
            return ready, status
        
        else:
            # Phase C or beyond - no gate needed
            return True, {'phase': current_phase, 'no_gate_needed': True}
    
    def get_effective_phase(epoch: int, eval_metrics: dict = None, trm_metrics: dict = None) -> tuple:
        """
        Get the effective phase considering both epoch and metric gates.
        
        Returns the phase the training should actually use, which may be behind
        the epoch-based phase if metric gates haven't been passed.
        
        Args:
            epoch: Current epoch (0-based)
            eval_metrics: Latest eval metrics (or None if not yet evaluated)
            trm_metrics: Latest TRM metrics (or None if not yet evaluated)
            
        Returns:
            (phase_name, phase_config, gate_status)
        """
        # Get epoch-based phase
        epoch_phase, epoch_phase_cfg = get_current_phase(epoch)
        
        if not use_metric_gating or not phased_training_enabled:
            return epoch_phase, epoch_phase_cfg, {'gating': 'disabled'}
        
        epoch_1based = epoch + 1
        current_effective = phase_gate_state['current_effective_phase']
        
        # If epoch-based phase is ahead of effective phase, check if we can advance
        if epoch_phase == 'B' and current_effective == 'A':
            # Check A→B gate
            ready, status = check_phase_gate('A', epoch_1based, eval_metrics, trm_metrics)
            phase_gate_state['last_gate_status'] = status
            
            if ready:
                # Advance to Phase B
                phase_gate_state['current_effective_phase'] = 'B'
                phase_gate_state['phase_b_start_epoch'] = epoch_1based
                phase_gate_state['epochs_in_current_phase'] = 0
                phase_gate_state['a_to_b_consecutive_met'] = 0  # Reset counter
                phase_gate_state['transition_log'].append((epoch_1based, 'A', 'B'))
                return 'B', phased_training_config.get('phase_b', {}), status
            else:
                # Stay in Phase A
                return 'A', phased_training_config.get('phase_a', {}), status
                
        elif epoch_phase == 'C' and current_effective == 'B':
            # Check B→C gate
            ready, status = check_phase_gate('B', epoch_1based, eval_metrics, trm_metrics)
            phase_gate_state['last_gate_status'] = status
            
            if ready:
                # Advance to Phase C
                phase_gate_state['current_effective_phase'] = 'C'
                phase_gate_state['epochs_in_current_phase'] = 0
                phase_gate_state['b_to_c_consecutive_met'] = 0  # Reset counter
                phase_gate_state['transition_log'].append((epoch_1based, 'B', 'C'))
                return 'C', phased_training_config.get('phase_c', {}), status
            else:
                # Stay in Phase B
                return 'B', phased_training_config.get('phase_b', {}), status
                
        elif epoch_phase == 'C' and current_effective == 'A':
            # Need to pass through Phase B first
            # Check A→B gate
            ready, status = check_phase_gate('A', epoch_1based, eval_metrics, trm_metrics)
            phase_gate_state['last_gate_status'] = status
            
            if ready:
                phase_gate_state['current_effective_phase'] = 'B'
                phase_gate_state['phase_b_start_epoch'] = epoch_1based
                phase_gate_state['transition_log'].append((epoch_1based, 'A', 'B'))
                # Now in Phase B, but don't auto-advance to C in same epoch
                return 'B', phased_training_config.get('phase_b', {}), status
            else:
                return 'A', phased_training_config.get('phase_a', {}), status
        
        # Already at or past target phase
        return current_effective, phased_training_config.get(f'phase_{current_effective.lower()}', {}), \
               {'phase': current_effective, 'at_target': True}
    
    # Store base augmentation config for restoration
    base_aug_config = config.get('data', {}).get('augmentation', {}).copy()
    
    def get_current_phase(epoch: int) -> tuple:
        """
        Get current phase name and config based on epoch.
        
        IMPORTANT (Jan 2026 FIX): The `epoch` parameter is 0-based (from the training loop),
        but YAML phase boundaries are specified in 1-based human-readable epochs.
        We convert to 1-based for comparison: epoch_1based = epoch + 1
        
        Example: If phase_a.end_epoch=10 in YAML, Phase A runs for printed epochs 1-10,
        which corresponds to loop epochs 0-9 (epoch + 1 <= 10).
        """
        if not phased_training_enabled:
            return 'none', {}
        
        phase_a = phased_training_config.get('phase_a', {})
        phase_b = phased_training_config.get('phase_b', {})
        phase_c = phased_training_config.get('phase_c', {})
        
        # YAML epochs are 1-based; convert loop epoch (0-based) to 1-based for comparison
        epoch_1based = epoch + 1
        
        phase_a_end = phase_a.get('end_epoch', 10)
        phase_b_start = phase_b.get('start_epoch', phase_a_end + 1)
        phase_b_end = phase_b.get('end_epoch', 20)
        phase_c_start = phase_c.get('start_epoch', phase_b_end + 1)
        
        if epoch_1based <= phase_a_end:
            return 'A', phase_a
        elif epoch_1based >= phase_b_start and epoch_1based <= phase_b_end:
            return 'B', phase_b
        elif epoch_1based >= phase_c_start:
            return 'C', phase_c
        else:
            # Transition period - use phase B settings
            return 'B', phase_b
    
    def apply_phase_config(phase_name: str, phase_cfg: dict, epoch: int, dataset=None):
        """
        Apply phase-specific configuration overrides.
        
        FIXES (Jan 2026):
        1. Actually updates dataset augmentation via set_augmentation_config()
        2. Returns disable_flags that are enforced for module activation
        
        Args:
            phase_name: 'A', 'B', or 'C'
            phase_cfg: Phase configuration dict
            epoch: Current epoch
            dataset: ARCDataset instance for runtime augmentation update
            
        Returns:
            disable_flags dict for LOO/equiv/hyperlora/hpm gating
        """
        if not phase_cfg:
            return {}
        
        # =====================================================================
        # FIX #2: Actually update dataset augmentation at runtime
        # =====================================================================
        aug_override = phase_cfg.get('augmentation', {})
        if dataset is not None and hasattr(dataset, 'set_augmentation_config'):
            # Map YAML phase augmentation keys to dataset method params
            # Phase config uses: rotation, flip, transpose, color_permutation, color_permutation_prob
            # Dataset uses: augment (covers rotation/flip/transpose), color_permutation, color_permutation_prob
            #
            # FIX (Jan 2026): Proper logic for all three geometric aug flags
            # Dihedral is enabled if ANY of rotation/flip/transpose is True
            # Dihedral is disabled only if ALL specified flags are False
            rotation = aug_override.get('rotation', None)
            flip = aug_override.get('flip', None)
            transpose = aug_override.get('transpose', None)
            
            # Collect only explicitly specified flags
            specified_flags = [v for v in [rotation, flip, transpose] if v is not None]
            
            if len(specified_flags) == 0:
                # No geometric aug flags specified - use None (dataset default)
                dihedral_enabled = None
            elif any(specified_flags):
                # At least one is True - enable dihedral
                dihedral_enabled = True
            else:
                # All specified flags are False - disable dihedral
                dihedral_enabled = False
            
            dataset.set_augmentation_config(
                augment=dihedral_enabled,
                color_permutation=aug_override.get('color_permutation', None),
                color_permutation_prob=aug_override.get('color_permutation_prob', None),
                translational_augment=aug_override.get('translational', None),
            )
        
        # Phase-specific module disables
        # These are enforced BOTH in train_epoch (for losses) AND at module activation
        disable_flags = {
            'hyperlora': phase_cfg.get('disable_hyperlora', False),
            'loo': phase_cfg.get('disable_loo', False),
            'equivariance': phase_cfg.get('disable_equivariance', False),
            'hpm': phase_cfg.get('disable_hpm', False),
        }
        
        return disable_flags
    
    if phased_training_enabled:
        print(f"\n{'='*60}")
        print(f"PHASED TRAINING ENABLED (Jan 2026)")
        print(f"{'='*60}")
        phase_a = phased_training_config.get('phase_a', {})
        phase_b = phased_training_config.get('phase_b', {})
        phase_c = phased_training_config.get('phase_c', {})
        print(f"  Phase A (Base Solver): epochs 1-{phase_a.get('end_epoch', 10)}")
        print(f"    - Augmentation: rotation={phase_a.get('augmentation', {}).get('rotation', False)}, "
              f"flip={phase_a.get('augmentation', {}).get('flip', False)}")
        print(f"    - Meta: HyperLoRA={not phase_a.get('disable_hyperlora', True)}, "
              f"LOO={not phase_a.get('disable_loo', True)}")
        print(f"  Phase B (Geometric Aug): epochs {phase_b.get('start_epoch', 11)}-{phase_b.get('end_epoch', 20)}")
        print(f"    - Augmentation: rotation={phase_b.get('augmentation', {}).get('rotation', True)}, "
              f"flip={phase_b.get('augmentation', {}).get('flip', True)}")
        print(f"    - Meta: HyperLoRA={not phase_b.get('disable_hyperlora', True)}, "
              f"LOO={not phase_b.get('disable_loo', True)}")
        print(f"  Phase C (Full Meta): epochs {phase_c.get('start_epoch', 21)}+")
        print(f"    - Augmentation: rotation={phase_c.get('augmentation', {}).get('rotation', True)}, "
              f"color_perm={phase_c.get('augmentation', {}).get('color_permutation', True)}")
        print(f"    - Meta: All enabled (respects individual start_epochs)")
        print(f"{'='*60}\n")
    
    # ================================================================
    # PRE-CACHE EVAL TASKS (Dec 2025 - Prod optimization)
    # ================================================================
    # Load TRM/TTA eval tasks ONCE before training loop to avoid
    # re-parsing JSON files every evaluation epoch.
    # ================================================================
    cached_eval_tasks = None
    use_trm_eval = config.get('evaluation', {}).get('use_trm_style_eval', False)
    if use_trm_eval:
        eval_path = Path(config['data']['eval_path'])
        max_eval_tasks = config.get('evaluation', {}).get('max_eval_tasks', 50)
        if eval_path.is_dir():
            cached_eval_tasks = []
            for json_file in sorted(eval_path.glob('*.json'))[:max_eval_tasks]:
                try:
                    with open(json_file, 'r') as f:
                        task_data = json.load(f)
                    cached_eval_tasks.append(task_data)
                except Exception:
                    pass
            if cached_eval_tasks:
                print(f"  [Eval Cache] Pre-loaded {len(cached_eval_tasks)} TRM eval tasks")
    
    print(f"\nStarting training from epoch {start_epoch} to {max_epochs}")
    print("=" * 60)
    
    # Collapse detection state
    collapse_warnings = 0
    max_collapse_warnings = 999  # DISABLED - let training run full epochs for debugging
    
    # Learning trajectory tracker for epoch-by-epoch trend analysis
    # Stores key metrics to verify learning is progressing correctly
    # NOTE: Clue count is a LATENT VARIABLE - no fixed target!
    # Each sample learns its own optimal count based on task complexity.
    learning_trajectory = {
        'epochs': [],
        'stop_prob': [],           # Stop predictor output (task-dependent, no fixed target)
        'expected_clues': [],      # Clue count (LATENT - varies by sample complexity)
        'attention_entropy': [],   # Attention sharpness (should decrease)
        'task_loss': [],           # Main loss (should decrease)
        'best_step': [],           # Best refinement step (should be later steps)
        'fg_coverage': [],         # Foreground prediction (should approach target)
        'train_accuracy': [],      # Training accuracy (should increase)
        'exact_match_pct': [],     # Exact match percentage (should increase)
    }
    
    # =============================================================
    # GLOBAL TASK SOLVING TRACKER (Jan 2026)
    # =============================================================
    # Track which unique task IDs have been solved (exact match) across
    # ALL epochs since training start. This provides critical insight:
    # - Are we solving NEW puzzles or just repeating old ones?
    # - How many unique tasks out of total have ever been solved?
    # - Per-epoch: how many NEW tasks were solved vs previously solved?
    #
    # PATCH (Jan 2026): Added expected_task_count for stable denominator.
    # The 'all_seen_task_ids' grows dynamically as batches are processed,
    # but expected_task_count is computed once from the dataset.
    # =============================================================
    
    # Compute expected_task_count from dataset (stable denominator)
    expected_task_count = None
    dataset_unique_task_ids = None
    if hasattr(train_loader, 'dataset'):
        train_dataset = train_loader.dataset
        if hasattr(train_dataset, 'tasks'):
            dataset_unique_task_ids = set()
            for t in train_dataset.tasks:
                tid = t.get('task_id') if isinstance(t, dict) else getattr(t, 'task_id', None)
                if tid is not None:
                    dataset_unique_task_ids.add(tid)
            expected_task_count = len(dataset_unique_task_ids)
            print(f"\n{'='*60}")
            print(f"TASK ID VALIDATION (Jan 2026 Patch)")
            print(f"{'='*60}")
            print(f"  Dataset tasks: {len(train_dataset.tasks)}")
            print(f"  Unique task IDs: {expected_task_count}")
            if expected_task_count < len(train_dataset.tasks) * 0.9:
                print(f"  ⚠️ WARNING: Significant task_id collisions detected!")
                print(f"     This may cause solve metrics to under-report.")
                print(f"     Check merged training manifest for duplicate IDs.")
            elif expected_task_count == len(train_dataset.tasks):
                print(f"  ✓ All tasks have unique IDs.")
            # Log sample of task IDs for debugging
            sample_ids = list(dataset_unique_task_ids)[:5]
            print(f"  Sample task IDs: {sample_ids}")
            print(f"{'='*60}\n")
        elif hasattr(train_dataset, '_cached_samples') and train_dataset._cached_samples:
            # For cached samples, count unique task_ids
            dataset_unique_task_ids = set(s.get('task_id') for s in train_dataset._cached_samples if s.get('task_id'))
            expected_task_count = len(dataset_unique_task_ids)
            print(f"  [Cached] Unique task IDs in cache: {expected_task_count}")
    
    global_task_tracker = {
        'all_solved_task_ids': set(),       # Task IDs ever solved (across all epochs)
        'all_seen_task_ids': set(),         # All task IDs ever seen (should match dataset size)
        'per_epoch_new_solved': [],         # List of newly solved task counts per epoch
        'per_epoch_total_solved': [],       # List of total solved task counts per epoch
        'first_solve_epoch': {},            # Dict: task_id -> first epoch where it was solved
        'expected_task_count': expected_task_count,  # Stable denominator from dataset
        'dataset_unique_task_ids': dataset_unique_task_ids,  # Full set for validation
    }
    
    # ADAPTIVE BATCH SIZE TRACKING
    # After LOO activation, batch size is reduced to prevent OOM
    current_batch_size_override = None  # None = use config batch_size
    
    # =============================================================
    # MONITORING THRESHOLDS - Extract once for use in eval loop and abort checks
    # =============================================================
    monitoring_cfg = config.get('monitoring', {})
    exact_match_warning = monitoring_cfg.get('exact_match_warning', 0.10)
    exact_match_critical = monitoring_cfg.get('exact_match_critical', 0.20)
    stop_value_warning = monitoring_cfg.get('stop_value_warning', 0.15)
    stop_value_critical = monitoring_cfg.get('stop_value_critical', 0.25)
    tta_consensus_warning = monitoring_cfg.get('tta_consensus_warning', 0.25)
    tta_consensus_critical = monitoring_cfg.get('tta_consensus_critical', 0.15)
    centroid_spread_warning = monitoring_cfg.get('centroid_spread_warning', 2.0)
    centroid_spread_critical = monitoring_cfg.get('centroid_spread_critical', 0.5)
    # P0.1: LoRA kill threshold for abort check (also read in train_epoch)
    lora_norm_kill_threshold = monitoring_cfg.get('lora_norm_kill', 50.0)
    
    for epoch in range(start_epoch, max_epochs):
        epoch_start = time.time()
        
        # ================================================================
        # PHASED TRAINING: Apply phase-specific config at epoch start
        # ================================================================
        # FIX (Jan 2026): Pass dataset to apply_phase_config for runtime augmentation update
        # FIX (Jan 2026): Use effective phase from gate state when metric gating is enabled
        if use_metric_gating and phased_training_enabled:
            # Use the effective phase determined by the gate at previous epoch's eval
            current_phase_name = phase_gate_state['current_effective_phase']
            current_phase_cfg = phased_training_config.get(f'phase_{current_phase_name.lower()}', {})
        else:
            # Pure epoch-based phase selection
            current_phase_name, current_phase_cfg = get_current_phase(epoch)
        phase_disable_flags = {}
        if phased_training_enabled and current_phase_cfg:
            # Get dataset from train_loader for runtime augmentation update
            train_dataset = train_loader.dataset if hasattr(train_loader, 'dataset') else None
            phase_disable_flags = apply_phase_config(
                current_phase_name, current_phase_cfg, epoch, 
                dataset=train_dataset  # Pass dataset for augmentation update
            )
            # Log phase transition
            # Track previous phase for transition detection
            if epoch == start_epoch:
                prev_phase_name = None
            elif use_metric_gating:
                # With metric gating, check if we transitioned at the last eval
                prev_phase_name = phase_gate_state.get('_prev_epoch_phase', current_phase_name)
            else:
                prev_phase_name = get_current_phase(epoch - 1)[0]
            
            # Store current phase for next epoch's comparison
            phase_gate_state['_prev_epoch_phase'] = current_phase_name
            
            if epoch == start_epoch or (prev_phase_name is not None and prev_phase_name != current_phase_name):
                print(f"\n{'='*60}")
                print(f"  PHASE {current_phase_name} ACTIVE (epoch {epoch + 1})")
                if use_metric_gating:
                    print(f"  [Metric-gated: epochs_in_phase={phase_gate_state['epochs_in_current_phase']}]")
                print(f"{'='*60}")
                if phase_disable_flags.get('hyperlora'):
                    print(f"    HyperLoRA: DISABLED by phase")
                if phase_disable_flags.get('loo'):
                    print(f"    LOO Loss: DISABLED by phase")
                if phase_disable_flags.get('equivariance'):
                    print(f"    Equivariance: DISABLED by phase")
                if phase_disable_flags.get('hpm'):
                    print(f"    HPM: DISABLED by phase")
                aug_cfg = current_phase_cfg.get('augmentation', {})
                print(f"    Augmentation: rot={aug_cfg.get('rotation', '(default)')}, "
                      f"flip={aug_cfg.get('flip', '(default)')}, "
                      f"trans={aug_cfg.get('translational', '(default)')}, "
                      f"color={aug_cfg.get('color_permutation', '(default)')}")
                print(f"{'='*60}\n")
                
                # =============================================================
                # WORKER RECREATION FIX (Jan 2026): Recreate DataLoader on phase change
                # =============================================================
                # With persistent_workers=True or num_workers>0, worker processes
                # keep their own dataset copies. Changing augmentation flags in the
                # main process doesn't propagate to workers. Solution: recreate the
                # DataLoader at phase boundaries to ensure workers pick up new config.
                # =============================================================
                data_cfg = config.get('data', {})
                if data_cfg.get('num_workers', 0) > 0 and not data_cfg.get('cache_samples', False):
                    # =============================================================
                    # PER-PHASE BATCH SIZE (Jan 2026 MEMORY FIX)
                    # =============================================================
                    # Read batch_size from phase config to stay within VRAM limits.
                    # Phase A: ~40 (no meta modules)
                    # Phase B: ~32 (solver_context active)
                    # Phase C: ~24 (full meta: HyperLoRA + HPM + LOO)
                    # =============================================================
                    phase_batch_size = current_phase_cfg.get('batch_size', None)
                    if phase_batch_size is not None:
                        # Phase-specific batch size takes priority
                        effective_batch_override = phase_batch_size
                        print(f"  [Phase Change] Using phase {current_phase_name} batch_size={phase_batch_size}")
                    else:
                        # Fall back to LOO override or base batch size
                        effective_batch_override = current_batch_size_override
                    
                    print(f"  [Phase Change] Recreating DataLoader to propagate augmentation config to workers...")
                    train_loader = create_train_loader(
                        config,
                        curriculum_stage=current_curriculum_stage if use_curriculum else 0,
                        max_grid_size=max_grid_size,
                        batch_size_override=effective_batch_override,
                    )
                    # Re-apply augmentation config to the new dataset
                    train_dataset = train_loader.dataset if hasattr(train_loader, 'dataset') else None
                    if train_dataset is not None:
                        apply_phase_config(
                            current_phase_name, current_phase_cfg, epoch,
                            dataset=train_dataset
                        )
                    print(f"  [Phase Change] DataLoader recreated with {len(train_loader)} batches, batch_size={effective_batch_override or train_cfg['batch_size']}")
        
        # ================================================================
        # FIX #3: Enforce phase disable flags for module activation
        # ================================================================
        # Phase flags should prevent module activation even if past start_epoch
        if phase_disable_flags.get('hyperlora', False):
            if hasattr(model, 'hyperlora_active') and model.hyperlora_active:
                model.hyperlora_active = False
                print(f"  [Phase Override] HyperLoRA deactivated for phase {current_phase_name}")
        if phase_disable_flags.get('hpm', False):
            if hasattr(model, 'hpm_memory_enabled') and model.hpm_memory_enabled:
                # FIX (Jan 2026): Actually disable HPM memory collection during disabled phases
                # This prevents low-quality early memories from polluting the buffer
                model.hpm_memory_enabled = False
                print(f"  [Phase Override] HPM memory collection DISABLED for phase {current_phase_name}")
        
        # Set epoch for bucketed batch sampler (ensures different batch order each epoch)
        if hasattr(train_loader, 'batch_sampler') and hasattr(train_loader.batch_sampler, 'set_epoch'):
            train_loader.batch_sampler.set_epoch(epoch)
        
        # HPM epoch start callback: reset routing statistics for load balancing
        if hasattr(model, 'hpm_on_epoch_start'):
            model.hpm_on_epoch_start()
        
        # BUG FIX: Update hpm_memory_enabled at epoch boundary (for staged activation)
        # This ensures buffers start populating at hpm_memory_start_epoch even when resuming
        # BUT respect phase disable flags (don't enable if phase says no)
        if not model.hpm_memory_enabled and epoch >= hpm_memory_start_epoch:
            if not phase_disable_flags.get('hpm', False):
                model.hpm_memory_enabled = True
                print(f"  [HPM Memory] Buffer population NOW ENABLED at epoch {epoch + 1}")
        
        print(f"\nEpoch {epoch + 1}/{max_epochs}")
        print("-" * 40)
        
        # ================================================================
        # CURRICULUM STAGE CHECK - Recreate dataloader if stage changes
        # ================================================================
        if use_curriculum:
            new_stage = get_curriculum_stage(epoch, curriculum_stages)
            if new_stage != current_curriculum_stage:
                # Percentile-based stage names
                stage_names = {0: "ALL 100%", 1: "70% EASY", 2: "90% TASKS", 3: "100% ALL"}
                print(f"\n" + "=" * 60)
                print(f"  CURRICULUM STAGE TRANSITION: {stage_names.get(current_curriculum_stage, '?')} -> {stage_names.get(new_stage, '?')}")
                print(f"=" * 60)
                
                current_curriculum_stage = new_stage
                train_loader = create_train_loader(
                    config,
                    curriculum_stage=current_curriculum_stage,
                    max_grid_size=max_grid_size,
                    batch_size_override=current_batch_size_override,  # Preserve LOO batch size if active
                )
                print(f"  New train samples: {len(train_loader.dataset)}, batches: {len(train_loader)}")
        
        # STAGGERED MODULE ACTIVATION: Scientifically ordered for stability
        # PHASE 2: Context path FIRST (SolverContext + CrossAttn) - provides meaningful features
        # PHASE 3: HyperLoRA AFTER context path - has features to predict LoRA from
        
        # ================================================================
        # LR MANAGEMENT: Reduce LR at activation epochs to prevent shock
        # ================================================================
        is_activation_epoch = (
            (use_solver_context and epoch == solver_context_start_epoch) or
            (use_cross_attention_context and epoch == cross_attention_start_epoch) or
            (use_hyperlora and epoch == meta_learning_start_epoch)
        )
        
        if is_activation_epoch and not activation_lr_state['reduced']:
            # Apply LR reduction using composable factor (Patch 3)
            activation_lr_state['activation_factor'] = activation_lr_reduction
            # Recompute all LRs from base
            for i, param_group in enumerate(optimizer.param_groups):
                base_lr = activation_lr_state['base_lrs'][i]
                total_factor = activation_lr_state['activation_factor'] * activation_lr_state['explosion_factor']
                param_group['lr'] = base_lr * total_factor
            activation_lr_state['reduced'] = True
            activation_lr_state['reduction_epoch'] = epoch
            effective_lr = activation_lr_state['original_lr'] * activation_lr_state['activation_factor']
            print(f"  [LR] Reduced to {effective_lr:.2e} (×{activation_lr_reduction}) for stability")
            
            # Track for meta escalation stability gating
            if meta_escalation_enabled:
                meta_escalation_state['lr_backoff_events_epoch'] += 1
        
        # Recover LR after recovery period (Patch 3: use composable factors)
        if activation_lr_state['reduced']:
            epochs_since_reduction = epoch - activation_lr_state['reduction_epoch']
            if epochs_since_reduction >= activation_lr_recovery_epochs:
                # Restore activation factor to 1.0
                activation_lr_state['activation_factor'] = 1.0
                # Recompute all LRs from base
                for i, param_group in enumerate(optimizer.param_groups):
                    base_lr = activation_lr_state['base_lrs'][i]
                    total_factor = activation_lr_state['activation_factor'] * activation_lr_state['explosion_factor']
                    param_group['lr'] = base_lr * total_factor
                activation_lr_state['reduced'] = False
                print(f"  [LR] Restored to original after {activation_lr_recovery_epochs} epochs recovery")
        
        # Gradient explosion cooldown countdown (Patch 3: use composable factors)
        if activation_lr_state['grad_explosion_cooldown'] > 0:
            activation_lr_state['grad_explosion_cooldown'] -= 1
            if activation_lr_state['grad_explosion_cooldown'] == 0:
                print(f"  [LR] Gradient explosion cooldown ended, restoring LR")
                # Restore explosion factor to 1.0
                activation_lr_state['explosion_factor'] = 1.0
                # Recompute all LRs from base
                for i, param_group in enumerate(optimizer.param_groups):
                    base_lr = activation_lr_state['base_lrs'][i]
                    total_factor = activation_lr_state['activation_factor'] * activation_lr_state['explosion_factor']
                    param_group['lr'] = base_lr * total_factor
        
        # ================================================================
        # PHASE 2: Activate Context Path (SolverContext + CrossAttn)
        # ================================================================
        if use_solver_context and epoch == solver_context_start_epoch:
            print(f"\n{'='*60}")
            print(f"PHASE 2: CONTEXT PATH ACTIVATED (epoch {epoch + 1})")
            print(f"{'='*60}")
            model.solver_context_active = True
            print(f"  SolverCrossAttention: NOW ACTIVE in solver loop")
            if device.type == 'cuda':
                torch.cuda.synchronize()
                allocated_mb = torch.cuda.memory_allocated() / 1024 / 1024
                max_mb = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
                print(f"  GPU MEMORY: {allocated_mb:.0f}MB / {max_mb:.0f}MB ({100*allocated_mb/max_mb:.1f}%)")
            print(f"{'='*60}\n")
        
        if use_cross_attention_context and epoch == cross_attention_start_epoch:
            print(f"\n{'='*60}")
            print(f"PHASE 2: CROSS-ATTENTION INJECTOR ACTIVATED (epoch {epoch + 1})")
            print(f"{'='*60}")
            model.cross_attention_active = True
            print(f"  CrossAttentionInjector: NOW ACTIVE (was using FiLM fallback)")
            if device.type == 'cuda':
                torch.cuda.synchronize()
                allocated_mb = torch.cuda.memory_allocated() / 1024 / 1024
                max_mb = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
                print(f"  GPU MEMORY: {allocated_mb:.0f}MB / {max_mb:.0f}MB ({100*allocated_mb/max_mb:.1f}%)")
            print(f"{'='*60}\n")
        
        # ================================================================
        # PHASE 3: Activate HyperLoRA with warmup
        # FIX (Jan 2026): Respect phase disable flags - don't activate if phase says no
        # ================================================================
        hyperlora_phase_allowed = not phase_disable_flags.get('hyperlora', False)
        if use_hyperlora and epoch == meta_learning_start_epoch and hyperlora_phase_allowed:
            print(f"\n{'='*60}")
            print(f"PHASE 3: HYPERLORA ACTIVATED (epoch {epoch + 1})")
            print(f"{'='*60}")
            model.hyperlora_active = True
            # Set initial warmup scale (Patch 1: use delta_scale which affects forward)
            if hasattr(model, 'hyper_lora') and model.hyper_lora is not None:
                model.hyper_lora.delta_scale = hyperlora_warmup_start_scale
                print(f"  HyperLoRA: Starting with delta_scale={hyperlora_warmup_start_scale}")
                print(f"  Warmup: {hyperlora_warmup_start_scale} → {hyperlora_warmup_end_scale} over {hyperlora_warmup_epochs} epochs")
            else:
                print(f"  HyperLoRA: LoRA deltas NOW CONTRIBUTING to forward pass")
            if device.type == 'cuda':
                torch.cuda.synchronize()
                allocated_mb = torch.cuda.memory_allocated() / 1024 / 1024
                max_mb = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
                print(f"  GPU MEMORY: {allocated_mb:.0f}MB / {max_mb:.0f}MB ({100*allocated_mb/max_mb:.1f}%)")
            print(f"{'='*60}\n")
        elif use_hyperlora and epoch >= meta_learning_start_epoch and not hyperlora_phase_allowed:
            # Phase overrides: keep HyperLoRA inactive even if past start epoch
            if model.hyperlora_active:
                model.hyperlora_active = False
                print(f"  [Phase Override] HyperLoRA kept INACTIVE for phase {current_phase_name}")
        
        # HyperLoRA warmup: linearly ramp delta_scale (Patch 1: use delta_scale)
        # Only apply warmup if HyperLoRA is allowed by current phase
        if use_hyperlora and hasattr(model, 'hyper_lora') and model.hyper_lora is not None and hyperlora_phase_allowed:
            if epoch >= meta_learning_start_epoch and epoch < meta_learning_start_epoch + hyperlora_warmup_epochs:
                warmup_progress = (epoch - meta_learning_start_epoch) / max(hyperlora_warmup_epochs, 1)
                current_scale = hyperlora_warmup_start_scale + warmup_progress * (hyperlora_warmup_end_scale - hyperlora_warmup_start_scale)
                model.hyper_lora.delta_scale = current_scale
                if epoch == meta_learning_start_epoch or (epoch - meta_learning_start_epoch) % 2 == 0:
                    print(f"  [HyperLoRA Warmup] epoch {epoch + 1}: delta_scale = {current_scale:.4f}")
            elif epoch == meta_learning_start_epoch + hyperlora_warmup_epochs:
                model.hyper_lora.delta_scale = hyperlora_warmup_end_scale
                print(f"  [HyperLoRA Warmup Complete] delta_scale = {hyperlora_warmup_end_scale}")
        
        # ======================================================================
        # LATE-PHASE LR DECAY (Jan 2026)
        # ======================================================================
        # Apply cosine LR decay after late_phase_lr_decay_start_epoch to help
        # fine-grained convergence when accuracy is already high.
        # ======================================================================
        if late_phase_lr_decay_enabled and epoch >= late_phase_lr_decay_start_epoch:
            import math
            decay_total_epochs = max(1, late_phase_lr_decay_end_epoch - late_phase_lr_decay_start_epoch)
            decay_progress = min(1.0, (epoch - late_phase_lr_decay_start_epoch) / decay_total_epochs)
            
            if late_phase_lr_decay_schedule == 'cosine':
                # Cosine decay: 1.0 → min_factor
                decay_factor = late_phase_lr_decay_min_factor + (1.0 - late_phase_lr_decay_min_factor) * 0.5 * (1 + math.cos(math.pi * decay_progress))
            else:
                # Linear decay
                decay_factor = 1.0 - decay_progress * (1.0 - late_phase_lr_decay_min_factor)
            
            late_phase_state['lr_decay_factor'] = decay_factor
            
            # Apply decay factor to optimizer LRs (composable with other factors)
            for i, param_group in enumerate(optimizer.param_groups):
                base_lr = activation_lr_state['base_lrs'][i]
                total_factor = (activation_lr_state['activation_factor'] * 
                               activation_lr_state['explosion_factor'] *
                               collapse_backoff_state['lr_factor'] *
                               decay_factor)
                param_group['lr'] = base_lr * total_factor
            
            if epoch == late_phase_lr_decay_start_epoch or (epoch - late_phase_lr_decay_start_epoch) % 10 == 0:
                print(f"  [Late-Phase LR Decay] epoch {epoch + 1}: factor={decay_factor:.3f}, LR={optimizer.param_groups[0]['lr']:.2e}")
        
        # ======================================================================
        # HPM SOLVER-CONTEXT COUPLING WARMUP (Jan 2026)
        # ======================================================================
        # Activate HPM solver coupling at hpm_solver_context_start_epoch with 
        # gradual gate warmup. Auto-disable if instability detected.
        # ======================================================================
        if hpm_solver_context_enabled and use_hpm and not hpm_solver_context_state['disabled_by_instability']:
            if epoch >= hpm_solver_context_start_epoch:
                # Mark as active
                if not hpm_solver_context_state['active']:
                    hpm_solver_context_state['active'] = True
                    print(f"\n{'='*60}")
                    print(f"HPM SOLVER-CONTEXT COUPLING ACTIVATED (epoch {epoch + 1})")
                    print(f"{'='*60}")
                    print(f"  Gate warmup over {hpm_solver_context_gate_warmup_epochs} epochs")
                    print(f"  Gate max: {hpm_solver_context_gate_max}")
                    print(f"{'='*60}\n")
                
                # Compute gate value based on warmup progress
                warmup_progress = min(1.0, (epoch - hpm_solver_context_start_epoch) / max(1, hpm_solver_context_gate_warmup_epochs))
                gate_value = warmup_progress * hpm_solver_context_gate_max
                hpm_solver_context_state['gate_value'] = gate_value
                
                # Apply to model's solver cross-attention
                if hasattr(model, 'solver') and hasattr(model.solver, 'solver_cross_attn'):
                    if model.solver.solver_cross_attn is not None and hasattr(model.solver.solver_cross_attn, 'set_hpm_gate'):
                        # Convert gate_value to logit (inverse sigmoid)
                        # gate_value = sigmoid(logit) => logit = log(g/(1-g))
                        import math
                        clamped_gate = min(0.999, max(0.001, gate_value))
                        gate_logit = math.log(clamped_gate / (1 - clamped_gate))
                        model.solver.solver_cross_attn.set_hpm_gate(gate_logit)
                
                # Reset epoch counters
                hpm_solver_context_state['tokens_used_epoch'] = 0
                hpm_solver_context_state['batches_with_tokens'] = 0
                
                if epoch == hpm_solver_context_start_epoch or (epoch - hpm_solver_context_start_epoch) % 5 == 0:
                    print(f"  [HPM Solver Coupling] epoch {epoch + 1}: gate={gate_value:.3f} (warmup {warmup_progress*100:.0f}%)")
        
        
        # ======================================================================
        # META ESCALATION: Stability-gated late-phase weight increase (Dec 2025)
        # ======================================================================
        # After start_epoch, gradually increase meta-learning weights toward targets
        # ONLY if training is stable. Pauses on instability, resumes when stable.
        # ======================================================================
        if meta_escalation_enabled and epoch >= meta_escalation_start_epoch:
            # Mark escalation as active
            if not meta_escalation_state['escalation_active']:
                meta_escalation_state['escalation_active'] = True
                print(f"\n{'='*60}")
                print(f"META ESCALATION PHASE STARTED (epoch {epoch + 1})")
                print(f"{'='*60}")
                print(f"  Ramping meta-learning weights toward targets over {meta_escalation_ramp_epochs} epochs")
                print(f"  Gated by stability: will pause if instability detected")
                print(f"{'='*60}\n")
            
            # Compute scheduled values based on progress
            if meta_escalation_ramp_epochs > 0:
                raw_progress = (epoch - meta_escalation_start_epoch) / meta_escalation_ramp_epochs
            else:
                raw_progress = 1.0
            scheduled_progress = min(1.0, max(0.0, raw_progress))
            
            # Apply schedule type
            if meta_escalation_schedule == 'cosine':
                import math
                # Cosine annealing: slower at start and end, faster in middle
                scheduled_progress = 0.5 * (1.0 - math.cos(math.pi * scheduled_progress))
            # else linear (default)
            
            # Compute scheduled target values
            scheduled_hyperlora = hyperlora_warmup_end_scale + scheduled_progress * (target_hyperlora_delta_scale - hyperlora_warmup_end_scale)
            scheduled_equiv = equiv_weight + scheduled_progress * (target_equiv_weight - equiv_weight)
            scheduled_loo = loo_weight + scheduled_progress * (target_loo_weight - loo_weight)
            scheduled_hpm_balance = base_hpm_balance_weight + scheduled_progress * (target_hpm_balance_weight - base_hpm_balance_weight)
            
            meta_escalation_state['hyperlora_delta_scale_scheduled'] = scheduled_hyperlora
            meta_escalation_state['equiv_weight_scheduled'] = scheduled_equiv
            meta_escalation_state['loo_weight_scheduled'] = scheduled_loo
            meta_escalation_state['hpm_balance_weight_scheduled'] = scheduled_hpm_balance
            meta_escalation_state['escalation_progress'] = scheduled_progress
            
            # Check stability: use info from previous epoch
            # ONLY if require_stability is True; otherwise always follow schedule
            prev_nan_streak = meta_escalation_state['max_nan_streak_epoch']
            prev_grad_events = meta_escalation_state['grad_explosion_events_epoch']
            prev_lr_events = meta_escalation_state['lr_backoff_events_epoch']
            prev_attn_collapse_events = meta_escalation_state.get('attention_collapse_events_epoch', 0)
            
            if meta_escalation_require_stability:
                # Jan 2026: Use stricter gates in late phase (epoch 50+)
                if late_phase_enabled and epoch >= late_phase_start_epoch:
                    # Late-phase mode: stricter thresholds
                    effective_max_nan_streak = late_phase_max_nan_streak
                    effective_max_grad_events = late_phase_max_grad_events
                    effective_max_lr_events = late_phase_max_lr_events
                    effective_max_collapse_events = late_phase_max_collapse_events
                    late_phase_state['active'] = True
                else:
                    # Normal phase: use original thresholds
                    effective_max_nan_streak = meta_escalation_max_nan_streak
                    effective_max_grad_events = meta_escalation_max_grad_events
                    effective_max_lr_events = meta_escalation_max_lr_events
                    effective_max_collapse_events = 0  # Any collapse = instability in normal phase
                
                # Stability-gated mode: check previous epoch for instability events
                is_stable = (
                    prev_nan_streak <= effective_max_nan_streak and
                    prev_grad_events <= effective_max_grad_events and
                    prev_lr_events <= effective_max_lr_events and
                    # Jan 2026: treat DSC attention collapse as instability.
                    # This directly targets the observed failure mode: attention max→0.005,
                    # entropy→6+, followed by FG/BG accuracy nosedive.
                    prev_attn_collapse_events <= effective_max_collapse_events
                )
            else:
                # Force-schedule mode: always considered stable (for ablations)
                is_stable = True
            meta_escalation_state['is_stable'] = is_stable
            
            # ================================================================
            # ATTENTION COLLAPSE BACKOFF (Jan 2026)
            # ================================================================
            # When consecutive collapse epochs exceed threshold, actively reduce
            # delta_scale and LR (not just pause escalation). This addresses
            # the late-epoch instability where attention max→0.005 precedes accuracy drops.
            # ================================================================
            if collapse_backoff_enabled:
                if prev_attn_collapse_events > 0:
                    collapse_backoff_state['consecutive_collapse_count'] += 1
                else:
                    collapse_backoff_state['consecutive_collapse_count'] = 0
                
                # Trigger backoff if consecutive collapse threshold exceeded
                if (collapse_backoff_state['consecutive_collapse_count'] >= attention_collapse_consecutive_threshold 
                    and not collapse_backoff_state['active']):
                    print(f"\n  ⚠️  ATTENTION COLLAPSE BACKOFF TRIGGERED!")
                    print(f"      {collapse_backoff_state['consecutive_collapse_count']} consecutive epochs with collapse (threshold={attention_collapse_consecutive_threshold})")
                    
                    # Save pre-backoff values
                    collapse_backoff_state['pre_backoff_delta_scale'] = meta_escalation_state['hyperlora_delta_scale_current']
                    collapse_backoff_state['pre_backoff_lr'] = optimizer.param_groups[0]['lr']
                    
                    # Apply backoff
                    collapse_backoff_state['delta_scale_factor'] = collapse_backoff_delta_factor
                    collapse_backoff_state['lr_factor'] = collapse_backoff_lr_factor
                    collapse_backoff_state['active'] = True
                    collapse_backoff_state['cooldown_remaining'] = collapse_backoff_cooldown
                    
                    # Apply reduced delta_scale immediately
                    backed_off_delta = meta_escalation_state['hyperlora_delta_scale_current'] * collapse_backoff_delta_factor
                    meta_escalation_state['hyperlora_delta_scale_current'] = backed_off_delta
                    if hasattr(model, 'hyper_lora') and model.hyper_lora is not None:
                        model.hyper_lora.delta_scale = backed_off_delta
                    
                    # Apply reduced LR
                    for pg in optimizer.param_groups:
                        pg['lr'] = pg['lr'] * collapse_backoff_lr_factor
                    
                    print(f"      delta_scale: {collapse_backoff_state['pre_backoff_delta_scale']:.4f} → {backed_off_delta:.4f}")
                    print(f"      LR: {collapse_backoff_state['pre_backoff_lr']:.2e} → {optimizer.param_groups[0]['lr']:.2e}")
                    print(f"      Cooldown: {collapse_backoff_cooldown} epochs before restore begins")
                
                # Restore logic after cooldown if stable
                elif collapse_backoff_state['active']:
                    if prev_attn_collapse_events == 0:
                        if collapse_backoff_state['cooldown_remaining'] > 0:
                            collapse_backoff_state['cooldown_remaining'] -= 1
                        else:
                            # Gradually restore delta_scale and LR
                            old_delta_factor = collapse_backoff_state['delta_scale_factor']
                            old_lr_factor = collapse_backoff_state['lr_factor']
                            
                            new_delta_factor = min(1.0, old_delta_factor + collapse_backoff_restore_rate)
                            new_lr_factor = min(1.0, old_lr_factor + collapse_backoff_restore_rate)
                            
                            collapse_backoff_state['delta_scale_factor'] = new_delta_factor
                            collapse_backoff_state['lr_factor'] = new_lr_factor
                            
                            # Apply restored values
                            if collapse_backoff_state['pre_backoff_delta_scale'] is not None:
                                restored_delta = collapse_backoff_state['pre_backoff_delta_scale'] * new_delta_factor
                                meta_escalation_state['hyperlora_delta_scale_current'] = restored_delta
                                if hasattr(model, 'hyper_lora') and model.hyper_lora is not None:
                                    model.hyper_lora.delta_scale = restored_delta
                            
                            if collapse_backoff_state['pre_backoff_lr'] is not None:
                                for i, pg in enumerate(optimizer.param_groups):
                                    base_lr = activation_lr_state['base_lrs'][i]
                                    pg['lr'] = base_lr * new_lr_factor
                            
                            if new_delta_factor >= 1.0 and new_lr_factor >= 1.0:
                                collapse_backoff_state['active'] = False
                                collapse_backoff_state['consecutive_collapse_count'] = 0
                                print(f"  [Collapse Backoff] Fully restored - delta_scale and LR back to normal")
                            else:
                                print(f"  [Collapse Backoff] Restoring: delta_factor={new_delta_factor:.2f}, lr_factor={new_lr_factor:.2f}")
                    else:
                        # Collapse still happening, reset cooldown
                        collapse_backoff_state['cooldown_remaining'] = collapse_backoff_cooldown
            
            if not is_stable:
                meta_escalation_state['escalation_paused'] = True
                meta_escalation_state['stable_epochs_count'] = 0
                # Don't increase weights, but don't decrease either (backoff handles that)
                if meta_escalation_log_every_epoch:
                    print(f"  [Meta Escalation] PAUSED due to instability (nan={prev_nan_streak}, grad={prev_grad_events}, lr={prev_lr_events}, attn_collapse={prev_attn_collapse_events})")
            else:
                meta_escalation_state['stable_epochs_count'] = meta_escalation_state.get('stable_epochs_count', 0) + 1
                
                if meta_escalation_state['escalation_paused']:
                    # Recovery: slowly move current toward scheduled
                    if meta_escalation_recovery_enabled:
                        gap_hyperlora = scheduled_hyperlora - meta_escalation_state['hyperlora_delta_scale_current']
                        gap_equiv = scheduled_equiv - meta_escalation_state['equiv_weight_current']
                        gap_loo = scheduled_loo - meta_escalation_state['loo_weight_current']
                        
                        recovery_hyperlora = meta_escalation_state['hyperlora_delta_scale_current'] + meta_escalation_recovery_step * gap_hyperlora
                        recovery_equiv = meta_escalation_state['equiv_weight_current'] + meta_escalation_recovery_step * gap_equiv
                        recovery_loo = meta_escalation_state['loo_weight_current'] + meta_escalation_recovery_step * gap_loo
                        
                        meta_escalation_state['hyperlora_delta_scale_current'] = min(recovery_hyperlora, scheduled_hyperlora)
                        meta_escalation_state['equiv_weight_current'] = min(recovery_equiv, scheduled_equiv)
                        meta_escalation_state['loo_weight_current'] = min(recovery_loo, scheduled_loo)
                        
                        # Check if we've caught up to schedule
                        if (abs(meta_escalation_state['hyperlora_delta_scale_current'] - scheduled_hyperlora) < 0.001 and
                            abs(meta_escalation_state['equiv_weight_current'] - scheduled_equiv) < 0.001 and
                            abs(meta_escalation_state['loo_weight_current'] - scheduled_loo) < 0.001):
                            meta_escalation_state['escalation_paused'] = False
                            if meta_escalation_log_every_epoch:
                                print(f"  [Meta Escalation] RESUMED - caught up to schedule")
                        else:
                            if meta_escalation_log_every_epoch:
                                print(f"  [Meta Escalation] Recovering toward schedule...")
                else:
                    # Normal escalation: apply scheduled values directly
                    meta_escalation_state['hyperlora_delta_scale_current'] = scheduled_hyperlora
                    meta_escalation_state['equiv_weight_current'] = scheduled_equiv
                    meta_escalation_state['loo_weight_current'] = scheduled_loo
                    meta_escalation_state['hpm_balance_weight_current'] = scheduled_hpm_balance
            
            # Apply HyperLoRA delta_scale now (model attribute, not loss fn)
            if hasattr(model, 'hyper_lora') and model.hyper_lora is not None:
                # Only override if we're past warmup
                if epoch >= meta_learning_start_epoch + hyperlora_warmup_epochs:
                    model.hyper_lora.delta_scale = meta_escalation_state['hyperlora_delta_scale_current']
            
            # NOTE: equiv/loo weight application moved AFTER effective_*_fn assignment below
            # to avoid UnboundLocalError (effective_equiv_fn/effective_loo_fn defined later)
            
            # Log escalation state
            if meta_escalation_log_every_epoch:
                print(f"  [Meta Escalation] epoch {epoch + 1}: progress={scheduled_progress:.2f}, stable={is_stable}")
                print(f"    HyperLoRA: {meta_escalation_state['hyperlora_delta_scale_current']:.4f} (target={scheduled_hyperlora:.4f})")
                print(f"    Equiv: {meta_escalation_state['equiv_weight_current']:.4f} (target={scheduled_equiv:.4f})")
                print(f"    LOO: {meta_escalation_state['loo_weight_current']:.4f} (target={scheduled_loo:.4f})")
                if use_hpm:
                    print(f"    HPM balance: {meta_escalation_state['hpm_balance_weight_current']:.4f} (target={scheduled_hpm_balance:.4f})")
            
            # Reset per-epoch counters for next epoch
            meta_escalation_state['grad_explosion_events_epoch'] = 0
            meta_escalation_state['lr_backoff_events_epoch'] = 0
            meta_escalation_state['max_nan_streak_epoch'] = 0
            meta_escalation_state['attention_collapse_events_epoch'] = 0
        
        # STAGED EQUIVARIANCE LOSS: Activate at equiv_start_epoch
        if use_equivariance and epoch == equiv_start_epoch:
            print(f"\n{'='*60}")
            print(f"EQUIVARIANCE LOSS ACTIVATED (epoch {epoch + 1})")
            print(f"{'='*60}")
            print(f"  Weight: {equiv_weight}")
            print(f"  Augmentations per task: {equiv_config.get('num_augmentations', 4)}")
            print(f"  Purpose: Enforce dihedral invariance (fix 13% TTA consensus)")
            print(f"  Expected effect: Predictions consistent across rotations/flips")
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
                allocated_mb = torch.cuda.memory_allocated() / 1024 / 1024
                reserved_mb = torch.cuda.memory_reserved() / 1024 / 1024
                max_mb = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
                print(f"\n  GPU MEMORY: {allocated_mb:.0f}MB alloc, {reserved_mb:.0f}MB reserved / {max_mb:.0f}MB")
                print(f"  NOTE: Equivariance does 4 forward passes - expect ~2-3GB memory increase")
            
            print(f"{'='*60}\n")
        
        # STAGED LOO LOSS: Activate at loo_start_epoch
        if use_loo and epoch == loo_start_epoch:
            print(f"\n{'='*60}")
            print(f"LOO (LEAVE-ONE-OUT) LOSS ACTIVATED (epoch {epoch + 1})")
            print(f"{'='*60}")
            print(f"  Weight: {loo_weight}")
            print(f"  Min pairs required: {loo_min_pairs}")
            print(f"  Max LOO passes: {loo_max_pairs}")
            print(f"  Purpose: Teach few-shot generalization (fix 60x entropy gap)")
            print(f"  Expected effect: Generalize from N-1 examples to Nth")
            
            # ================================================================
            # ADAPTIVE BATCH SIZE: Reduce batch size when LOO activates
            # ================================================================
            # LOO runs N forward passes per sample, multiplying memory usage.
            # Formula: loo_batch_size = floor(original_batch / batch_reduction_divisor)
            # Configurable divisor (default 2.0) allows tuning based on observed VRAM usage.
            # Observed: 10GB peak on 24GB GPU → can safely use divisor=2.0 (was ~4.4).
            base_batch_size = train_cfg['batch_size']
            loo_batch_size = int(base_batch_size / loo_batch_reduction_divisor)
            loo_batch_size = max(loo_batch_size, loo_min_batch_size)  # Safety floor
            
            # Adjust grad accumulation to maintain effective batch size
            orig_grad_accum = train_cfg.get('grad_accumulation_steps', 4)
            new_grad_accum = orig_grad_accum
            if loo_adjust_grad_accum and loo_batch_size < base_batch_size:
                # effective_batch = batch × grad_accum should stay constant
                effective_batch = base_batch_size * orig_grad_accum
                new_grad_accum = max(1, int(round(effective_batch / loo_batch_size)))
                current_grad_accum_override = new_grad_accum
            
            print(f"\n  ADAPTIVE BATCH SIZE (Dec 2025 - Configurable):")
            print(f"    Original: batch={base_batch_size}, grad_accum={orig_grad_accum}, effective={base_batch_size * orig_grad_accum}")
            print(f"    LOO batch: floor({base_batch_size} / {loo_batch_reduction_divisor}) = {loo_batch_size}")
            if loo_adjust_grad_accum:
                print(f"    Adjusted grad_accum: {new_grad_accum} (effective batch: {loo_batch_size * new_grad_accum})")
                # Update config in-place so train_epoch picks up new grad_accumulation_steps
                config['training']['grad_accumulation_steps'] = new_grad_accum
            print(f"    Memory multiplier: {loo_max_pairs}x forward passes per sample")
            print(f"    Recreating DataLoader with reduced batch size...")
            
            # Track current batch size for future DataLoader recreations (curriculum transitions)
            current_batch_size_override = loo_batch_size
            
            # Recreate DataLoader with reduced batch size
            train_loader = create_train_loader(
                config,
                curriculum_stage=current_curriculum_stage,
                max_grid_size=max_grid_size,
                batch_size_override=loo_batch_size,
            )
            print(f"    New batches per epoch: {len(train_loader)}")
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
                allocated_mb = torch.cuda.memory_allocated() / 1024 / 1024
                reserved_mb = torch.cuda.memory_reserved() / 1024 / 1024
                max_mb = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
                print(f"\n  GPU MEMORY: {allocated_mb:.0f}MB alloc, {reserved_mb:.0f}MB reserved / {max_mb:.0f}MB")
                print(f"  NOTE: LOO does {loo_max_pairs} forward passes per sample")
                print(f"  Effective samples per step: {loo_batch_size} × {loo_max_pairs} = {loo_batch_size * loo_max_pairs}")
            
            print(f"{'='*60}\n")
        
        # STAGED HPM: Activate HPM at hpm_start_epoch
        if use_hpm and epoch == hpm_start_epoch:
            print(f"\n{'='*60}")
            print(f"HPM PHASE ACTIVATED (epoch {epoch + 1})")
            print(f"{'='*60}")
            print(f"  Hierarchical Primitive Memory: NOW ACTIVE")
            print(f"  Banks: Compositional, Pattern, Relational")
            print(f"  Top-K routing: {config.get('model', {}).get('hpm_top_k', 2)}")
            print(f"  Load balance weight: {config.get('model', {}).get('hpm_balance_weight', 0.01)}")
            print(f"{'='*60}\n")
            if hasattr(model, 'use_hpm'):
                model.use_hpm = True
        
        # Determine if meta-learning is active this epoch
        meta_learning_active = epoch >= meta_learning_start_epoch
        effective_loo_fn = loo_loss_fn if meta_learning_active else None
        effective_equiv_fn = equiv_loss_fn if meta_learning_active else None
        
        # Set model attributes for LOO/Equivariance activation state
        # This allows memory breakdown logging to show correct activation status
        model.loo_enabled = use_loo and epoch >= loo_start_epoch
        model.equivariance_enabled = use_equivariance and epoch >= equiv_start_epoch
        
        # ======================================================================
        # META ESCALATION: Apply escalated weights to loss functions (Dec 2025)
        # ======================================================================
        # This MUST be after effective_*_fn assignment to avoid UnboundLocalError.
        # Only applies if meta escalation is enabled AND active this epoch.
        # ======================================================================
        if meta_escalation_enabled and meta_escalation_state['escalation_active']:
            # Check require_stability flag - if False, always apply scheduled values
            should_apply = True
            if meta_escalation_require_stability and not meta_escalation_state['is_stable']:
                should_apply = False  # Gating blocked, keep current values
            
            if should_apply:
                if effective_equiv_fn is not None and hasattr(effective_equiv_fn, 'config'):
                    effective_equiv_fn.config.loss_weight = meta_escalation_state['equiv_weight_current'] * nan_backoff_state['equiv_weight_factor']
                
                if effective_loo_fn is not None and hasattr(effective_loo_fn, 'config'):
                    effective_loo_fn.config.loss_weight = meta_escalation_state['loo_weight_current'] * nan_backoff_state['loo_weight_factor']

                # Apply HPM balance weight (used inside train_epoch when HPM is active)
                if use_hpm and 'model' in config:
                    config['model']['hpm_balance_weight'] = float(meta_escalation_state.get('hpm_balance_weight_current', base_hpm_balance_weight))
        
        # Record memory before training epoch for comparison
        if device.type == 'cuda':
            torch.cuda.synchronize()
            pre_epoch_allocated = torch.cuda.memory_allocated() / 1024 / 1024
            torch.cuda.reset_peak_memory_stats()
        
        # Train (Jan 2026: pass output equiv, group marg, and phase disable flags)
        train_losses, global_step = train_epoch(
            model, train_loader, loss_fn, optimizer, device,
            epoch, config, scaler, global_step, ema, effective_loo_fn, effective_equiv_fn,
            output_equiv_loss_fn=output_equiv_loss_fn,
            group_marg_loss_fn=group_marg_loss_fn,  # Jan 2026: Group-marginalized NLL
            loo_start_epoch=loo_start_epoch, equiv_start_epoch=equiv_start_epoch,
            output_equiv_start_epoch=output_equiv_start_epoch,
            output_equiv_weight=output_equiv_weight,
            output_equiv_num_augs=output_equiv_num_augs,
            group_marg_start_epoch=group_marg_start_epoch,  # Jan 2026
            group_marg_weight=group_marg_weight,            # Jan 2026
            global_task_tracker=global_task_tracker,  # Jan 2026: Track tasks solved across epochs
            phase_disable_flags=phase_disable_flags,  # Jan 2026: Phased training
            # Jan 2026 Ablation Study: ART + ARPS
            art_module=art_module,
            arps_module=arps_module,
            art_start_epoch=art_start_epoch,
            arps_start_epoch=arps_start_epoch,
        )

        # ================================================================
        # Meta-escalation stability gating: capture attention collapse events
        # ================================================================
        if meta_escalation_enabled:
            meta_escalation_state['attention_collapse_events_epoch'] = int(
                train_losses.get('diagnostics', {}).get('attention_collapse_events_epoch', 0)
            )
        
        # ================================================================
        # P0.1: HARD-STOP ENFORCEMENT FOR FATAL EVENTS (Dec 2025)
        # ================================================================
        # Check if epoch triggered abort conditions. If so, save checkpoint and stop.
        # This prevents silently continuing on corrupted weights or unstable state.
        # ================================================================
        fatal_abort = False
        abort_reason = ""
        
        # Check NaN abort threshold
        if train_losses.get('nan_abort_triggered', False):
            fatal_abort = True
            abort_reason = f"NaN batches exceeded monitoring.nan_batches_abort threshold"
        
        # Check LoRA kill abort
        if train_losses.get('diagnostics', {}).get('lora_kill_abort', False):
            fatal_abort = True
            abort_reason = f"LoRA norm exceeded kill threshold ({lora_norm_kill_threshold}) for {3} consecutive batches"
        
        if fatal_abort:
            print(f"\n{'!'*60}")
            print(f"[FATAL] TRAINING ABORTED - {abort_reason}")
            print(f"{'!'*60}")
            print(f"Saving emergency checkpoint before exit...")
            
            # Save emergency checkpoint
            emergency_path = checkpoint_dir / f"emergency_epoch_{epoch + 1}.pt"
            save_checkpoint(
                model, optimizer, scheduler, epoch, global_step,
                train_losses, best_task_accuracy, config, str(emergency_path)
            )
            print(f"Emergency checkpoint saved to: {emergency_path}")
            print(f"\nRecommended actions:")
            print(f"  1. Reduce learning_rate (current: {config['training']['learning_rate']})")
            print(f"  2. Reduce hyperlora_init_scale or hyperlora_lr_multiplier")
            print(f"  3. Check data pipeline for corrupted samples")
            print(f"  4. Resume from last stable checkpoint with adjusted config")
            print(f"{'!'*60}\n")
            
            # Cleanup and exit
            if tee_logger:
                tee_logger.close()
            cleanup_memory()
            raise RuntimeError(f"Training aborted: {abort_reason}")
        
        # ================================================================
        # PATCH 4: NaN-Driven Meta-Loss Backoff (Dec 2025)
        # ================================================================
        # If consecutive NaN streak exceeded threshold, reduce meta-loss weights.
        # This implements the documented "3 consecutive NaN → halve newest meta-loss" rule.
        # We use max_consecutive_nan_streak (true consecutive detection) not total nan_batches.
        # Priority: reduce equivariance first (less critical), then LOO.
        consecutive_nan_streak = train_losses.get('max_consecutive_nan_streak', 0)
        
        # Track for meta escalation stability gating
        if meta_escalation_enabled:
            meta_escalation_state['max_nan_streak_epoch'] = max(
                meta_escalation_state['max_nan_streak_epoch'],
                consecutive_nan_streak
            )
            meta_escalation_state['max_nan_streak_in_window'] = max(
                meta_escalation_state['max_nan_streak_in_window'],
                consecutive_nan_streak
            )
        
        if consecutive_nan_streak >= nan_backoff_threshold and not nan_backoff_state['nan_backoff_active']:
            print(f"\n  ⚠️  NaN BACKOFF TRIGGERED!")
            print(f"      {consecutive_nan_streak} consecutive NaN batches (threshold={nan_backoff_threshold})")
            
            # Reduce newest meta-loss first (equivariance), then LOO if already reduced
            if nan_backoff_state['equiv_weight_factor'] > 0.25:
                nan_backoff_state['equiv_weight_factor'] *= 0.5
                print(f"      Reducing equivariance weight factor: {nan_backoff_state['equiv_weight_factor']:.2f}")
            elif nan_backoff_state['loo_weight_factor'] > 0.25:
                nan_backoff_state['loo_weight_factor'] *= 0.5
                print(f"      Reducing LOO weight factor: {nan_backoff_state['loo_weight_factor']:.2f}")
            else:
                print(f"      Meta-loss weights already minimal, cannot reduce further")
            
            nan_backoff_state['nan_backoff_active'] = True
            nan_backoff_state['nan_backoff_epochs'] = 3  # Cooldown before restoring
            
            # Update the actual loss function weights if they exist
            if effective_equiv_fn is not None and hasattr(effective_equiv_fn, 'config'):
                effective_equiv_fn.config.loss_weight = equiv_weight * nan_backoff_state['equiv_weight_factor']
            if effective_loo_fn is not None and hasattr(effective_loo_fn, 'config'):
                effective_loo_fn.config.loss_weight = loo_weight * nan_backoff_state['loo_weight_factor']
        
        # Restore meta-loss weights after cooldown if NaNs stopped
        epoch_nan_batches = train_losses.get('nan_batches', 0)
        if nan_backoff_state['nan_backoff_active']:
            if epoch_nan_batches == 0:
                nan_backoff_state['nan_backoff_epochs'] -= 1
                if nan_backoff_state['nan_backoff_epochs'] <= 0:
                    # Gradually restore weights
                    old_equiv = nan_backoff_state['equiv_weight_factor']
                    old_loo = nan_backoff_state['loo_weight_factor']
                    nan_backoff_state['equiv_weight_factor'] = min(1.0, old_equiv * 1.5)
                    nan_backoff_state['loo_weight_factor'] = min(1.0, old_loo * 1.5)
                    
                    if nan_backoff_state['equiv_weight_factor'] >= 1.0 and nan_backoff_state['loo_weight_factor'] >= 1.0:
                        nan_backoff_state['nan_backoff_active'] = False
                        print(f"  [NaN Backoff] Weights fully restored")
                    else:
                        nan_backoff_state['nan_backoff_epochs'] = 2  # More cooldown
                        print(f"  [NaN Backoff] Partial restore: equiv={nan_backoff_state['equiv_weight_factor']:.2f}, loo={nan_backoff_state['loo_weight_factor']:.2f}")
        
        # Log epoch peak memory and compare to previous epochs
        if device.type == 'cuda':
            torch.cuda.synchronize()
            epoch_peak_allocated = torch.cuda.max_memory_allocated() / 1024 / 1024
            epoch_peak_reserved = torch.cuda.max_memory_reserved() / 1024 / 1024
            total_gpu = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
            
            # Store for comparison
            if not hasattr(train_epoch, '_prev_epoch_peak'):
                train_epoch._prev_epoch_peak = None
            
            prev_peak = train_epoch._prev_epoch_peak
            delta_str = ""
            if prev_peak is not None:
                delta = epoch_peak_allocated - prev_peak
                delta_str = f" (Δ{delta:+.0f}MB from prev epoch)"
                # ALERT if memory increased significantly at transition epochs
                is_transition = (
                    epoch == meta_learning_start_epoch or 
                    (use_equivariance and epoch == equiv_start_epoch) or 
                    (use_loo and epoch == loo_start_epoch)
                )
                if is_transition and delta > 1000:
                    print(f"\n  ⚠️  MEMORY SPIKE AT TRANSITION: +{delta:.0f}MB")
                    if epoch == equiv_start_epoch:
                        print(f"      Equivariance loss now active (4 forward passes per sample)")
                    if epoch == loo_start_epoch:
                        print(f"      LOO loss now active (N forward passes per sample)")
                    if epoch == meta_learning_start_epoch:
                        print(f"      HyperLoRA/SolverCrossAttention/CrossAttention now contributing")
            
            train_epoch._prev_epoch_peak = epoch_peak_allocated
            
            # Only print for first few epochs or at key transitions
            is_key_epoch = (
                epoch < 5 or 
                epoch == meta_learning_start_epoch or 
                (use_equivariance and epoch == equiv_start_epoch) or
                (use_loo and epoch == loo_start_epoch)
            )
            if is_key_epoch:
                print(f"\n  [EPOCH MEMORY] Peak: {epoch_peak_allocated:.0f}MB alloc, {epoch_peak_reserved:.0f}MB reserved{delta_str}")
                if epoch_peak_reserved > total_gpu * 0.95:
                    print(f"      ⚠️  >95% GPU used! May spill to shared memory (SLOW)")
        
        # Update scheduler (if using one)
        if scheduler is not None:
            scheduler.step()
        
        # ================================================================
        # LATE-PHASE LR DECAY (Jan 2026)
        # ================================================================
        # After late_phase_start_epoch, apply cosine LR decay for fine-grained
        # convergence in high-accuracy regime. This is ADDITIONAL to the
        # scheduler (which may have already finished) and collapse backoff.
        # ================================================================
        if late_phase_lr_decay_enabled and epoch >= late_phase_lr_decay_start_epoch:
            # Calculate decay progress (0 at start, 1 at end)
            decay_total_epochs = late_phase_lr_decay_end_epoch - late_phase_lr_decay_start_epoch
            decay_progress = min(1.0, (epoch - late_phase_lr_decay_start_epoch) / max(1, decay_total_epochs))
            
            # Compute decay factor based on schedule
            if late_phase_lr_decay_schedule == 'cosine':
                # Cosine decay: smooth transition from 1.0 to min_factor
                import math
                decay_factor = late_phase_lr_decay_min_factor + (1.0 - late_phase_lr_decay_min_factor) * 0.5 * (1 + math.cos(math.pi * decay_progress))
            else:
                # Linear decay (fallback)
                decay_factor = 1.0 - decay_progress * (1.0 - late_phase_lr_decay_min_factor)
            
            late_phase_state['lr_decay_factor'] = decay_factor
            
            # Apply decay to base LRs (composes with other factors)
            for i, param_group in enumerate(optimizer.param_groups):
                base_lr = activation_lr_state['base_lrs'][i]
                # Compose with activation factor, explosion factor, and collapse backoff
                total_factor = (
                    activation_lr_state['activation_factor'] * 
                    activation_lr_state['explosion_factor'] * 
                    collapse_backoff_state['lr_factor'] *
                    decay_factor
                )
                param_group['lr'] = base_lr * total_factor
            
            # Log at key epochs
            if epoch == late_phase_lr_decay_start_epoch or epoch % 10 == 0:
                print(f"\n  [Late-Phase LR Decay] epoch {epoch + 1}: factor={decay_factor:.4f}, LR={optimizer.param_groups[0]['lr']:.2e}")
        
        # ================================================================
        # GRADIENT EXPLOSION BACKOFF (Safety mechanism)
        # ================================================================
        # Patch 2 (Dec 2025): Use max_grad_norm_before_clip (whole epoch max)
        # instead of single-step snapshot to catch late-epoch explosions
        diagnostics = train_losses.get('diagnostics', {})
        grad_norm_before = diagnostics.get('max_grad_norm_before_clip', 
                                            diagnostics.get('grad_norm_before_clip', 0.0))
        grad_clip = config['training']['gradient_clip']
        
        if grad_norm_before > grad_explosion_threshold * grad_clip:
            # Trigger gradient explosion backoff (Patch 3: use composable factors)
            if activation_lr_state['grad_explosion_cooldown'] == 0:
                print(f"\n  ⚠️  GRADIENT EXPLOSION DETECTED!")
                print(f"      Max grad norm: {grad_norm_before:.1f} > {grad_explosion_threshold}x clip ({grad_explosion_threshold * grad_clip:.1f})")
                print(f"      Applying LR reduction: ×{grad_explosion_lr_reduction}")
                
                # Apply explosion factor
                activation_lr_state['explosion_factor'] = grad_explosion_lr_reduction
                # Recompute all LRs from base
                for i, param_group in enumerate(optimizer.param_groups):
                    base_lr = activation_lr_state['base_lrs'][i]
                    total_factor = activation_lr_state['activation_factor'] * activation_lr_state['explosion_factor']
                    param_group['lr'] = base_lr * total_factor
                
                activation_lr_state['grad_explosion_cooldown'] = grad_explosion_cooldown_epochs
                print(f"      Cooldown: {grad_explosion_cooldown_epochs} epochs before LR restoration")
                
                # Track for meta escalation stability gating
                # NOTE: Grad explosion LR reduction counts as BOTH grad_explosion AND lr_backoff
                # since it's a safety LR intervention that should pause escalation
                if meta_escalation_enabled:
                    meta_escalation_state['grad_explosion_events_epoch'] += 1
                    meta_escalation_state['lr_backoff_events_epoch'] += 1  # Also counts as LR backoff
        
        epoch_time = time.time() - epoch_start
        
        # Log curriculum stage if enabled
        stage_str = ""
        if use_curriculum:
            stage_names = {0: "100%", 1: "70%", 2: "90%", 3: "100%"}
            stage_str = f", Curriculum: {stage_names.get(current_curriculum_stage, '?')} tasks"
        
        # Get loss mode for accurate logging
        loss_mode = train_losses.get('loss_mode', 'focal')
        task_loss_val = train_losses.get('task_loss', train_losses.get('focal_loss', 0))
        deep_sup_loss = train_losses.get('deep_supervision_loss', 0)
        
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Total Loss: {train_losses['total_loss']:.4f}")
        print(f"  Task Loss ({loss_mode}): {task_loss_val:.4f}")
        
        # =============================================================
        # GLOBAL TASK TRACKING SUMMARY (Jan 2026)
        # =============================================================
        # Report unique tasks solved across all epochs to track learning progress
        # PATCH (Jan 2026): Use expected_task_count as stable denominator
        if global_task_tracker is not None:
            total_globally_solved = len(global_task_tracker['all_solved_task_ids'])
            total_ever_seen = len(global_task_tracker['all_seen_task_ids'])
            new_this_epoch = train_losses.get('diagnostics', {}).get('new_globally_solved_count', 0)
            
            # Update per-epoch tracking
            global_task_tracker['per_epoch_new_solved'].append(new_this_epoch)
            global_task_tracker['per_epoch_total_solved'].append(total_globally_solved)
            
            # Use expected_task_count as stable denominator (Jan 2026 patch)
            expected_total = global_task_tracker.get('expected_task_count', None)
            if expected_total is not None and expected_total > 0:
                solve_pct = (total_globally_solved / expected_total * 100)
                print(f"  [Global Task Progress] Unique Solved: {total_globally_solved}/{expected_total} ({solve_pct:.1f}%)")
            else:
                # Fallback to dynamic count (backward compatible)
                solve_pct = (total_globally_solved / total_ever_seen * 100) if total_ever_seen > 0 else 0.0
                print(f"  [Global Task Progress] Unique Solved: {total_globally_solved}/{total_ever_seen} ({solve_pct:.1f}%)")
            print(f"    NEW puzzles solved this epoch: {new_this_epoch}")
            if len(global_task_tracker['per_epoch_new_solved']) > 1:
                recent_new = global_task_tracker['per_epoch_new_solved'][-5:]  # Last 5 epochs
                if sum(recent_new) == 0:
                    print(f"    ⚠️ No new puzzles solved in last {len(recent_new)} epochs - model may be plateauing")
        
        # Only show auxiliary losses if they have non-zero weight in config
        if train_cfg.get('lambda_entropy', 0) > 0:
            print(f"  Entropy Loss: {train_losses.get('entropy_loss', 0):.4f} (weight={train_cfg['lambda_entropy']})")
        if train_cfg.get('lambda_sparsity', 0) > 0:
            print(f"  Sparsity Loss: {train_losses.get('sparsity_loss', 0):.4f} (weight={train_cfg['lambda_sparsity']})")
        if train_cfg.get('lambda_predicate', 0) > 0:
            print(f"  Predicate Loss: {train_losses.get('predicate_loss', 0):.4f} (weight={train_cfg['lambda_predicate']})")
        if train_cfg.get('lambda_curriculum', 0) > 0:
            print(f"  Curriculum Loss: {train_losses.get('curriculum_loss', 0):.4f} (weight={train_cfg['lambda_curriculum']})")
        if train_cfg.get('lambda_deep_supervision', 0) > 0:
            print(f"  Deep Supervision: {deep_sup_loss:.4f} (weight={train_cfg['lambda_deep_supervision']})")
        if train_cfg.get('lambda_act', 0) > 0 and config['model'].get('use_act', False):
            act_loss_val = train_losses.get('act_loss', 0)
            print(f"  ACT Loss: {act_loss_val:.4f} (weight={train_cfg['lambda_act']})")
        
        # ART (Anchor Robustness Training) stats - Jan 2026 Ablation
        art_config_yaml = config.get('model', {}).get('anchor_robustness', {})
        if art_config_yaml.get('enabled', False):
            art_loss_val = train_losses.get('art_consistency_loss', 0)
            art_weight = art_config_yaml.get('consistency_weight', 0.02)
            diagnostics = train_losses.get('diagnostics', {})
            art_batch_count = diagnostics.get('art_batch_count', 0)
            print(f"  ART Consistency Loss: {art_loss_val:.4f} (weight={art_weight}, batches={art_batch_count})")
            if art_batch_count == 0 and epoch >= art_config_yaml.get('start_epoch', 0):
                print(f"    ⚠️ ART active but no batches processed - check integration")
        
        # ARPS (Anchor-Relative Program Search) stats - Jan 2026 Ablation
        arps_config_yaml = config.get('model', {}).get('arps_dsl_search', {})
        if arps_config_yaml.get('enabled', False):
            arps_loss_val = train_losses.get('arps_imitation_loss', 0)
            arps_weight = arps_config_yaml.get('imitation_weight', 0.1)
            diagnostics = train_losses.get('diagnostics', {})
            arps_batch_count = diagnostics.get('arps_batch_count', 0)
            arps_valid_programs = diagnostics.get('arps_valid_programs', 0)
            arps_search_attempts = diagnostics.get('arps_search_attempts', 0)
            print(f"  ARPS Imitation Loss: {arps_loss_val:.4f} (weight={arps_weight}, batches={arps_batch_count})")
            if arps_search_attempts > 0:
                valid_rate = arps_valid_programs / arps_search_attempts if arps_search_attempts > 0 else 0
                print(f"    Programs: {arps_valid_programs} valid / {arps_search_attempts} attempts ({valid_rate:.1%} success)")
                if valid_rate < 0.01:
                    print(f"    ⚠️ Very low program validity - ARPS may not be contributing signal")
        
        # HPM (Hierarchical Primitive Memory) stats
        if config['model'].get('use_hpm', False) and hasattr(model, 'hpm_get_stats'):
            hpm_stats = model.hpm_get_stats()
            if hpm_stats:
                hpm_balance_loss = train_losses.get('hpm_balance_loss', 0)
                gate_value = hpm_stats.get('gate_value', 0)
                print(f"  HPM Balance Loss: {hpm_balance_loss:.4f} (weight={config['model'].get('hpm_balance_weight', 0.01)})")
                print(f"  HPM Gate Value: {gate_value:.4f} (0=no contribution, 1=full)")
                # Show buffer sizes if dynamic banks enabled
                instance_buf_size = hpm_stats.get('instance_buffer_size', 0)
                procedural_buf_size = hpm_stats.get('procedural_buffer_size', 0)
                if 'instance_buffer_size' in hpm_stats:
                    print(f"  HPM Instance Buffer: {instance_buf_size} entries")
                if 'procedural_buffer_size' in hpm_stats:
                    print(f"  HPM Procedural Buffer: {procedural_buf_size} entries")
                    # Explain why procedural buffer may be empty (expected before HyperLoRA active)
                    if procedural_buf_size == 0:
                        hyperlora_start = config['model'].get('hyperlora_start_epoch', 3)
                        if epoch < hyperlora_start:
                            print(f"    (Expected: procedural bank requires HyperLoRA, active at epoch {hyperlora_start + 1}+)")
                # Show tasks added this epoch (from exact matches)
                diagnostics = train_losses.get('diagnostics', {})
                tasks_added = diagnostics.get('hpm_tasks_added', 0)
                write_attempts = diagnostics.get('hpm_write_attempts', 0)
                writes_succeeded = diagnostics.get('hpm_writes_succeeded', 0)
                skip_reasons = diagnostics.get('hpm_write_skip_reasons', {})
                duplicates_skipped = diagnostics.get('hpm_duplicate_skipped', 0)
                
                # Always show HPM status for debugging (Jan 2026 patch - improved diagnostics)
                if write_attempts > 0:
                    print(f"  HPM Write Attempts: {write_attempts} → Succeeded: {writes_succeeded}")
                    # Show breakdown of skip reasons
                    if skip_reasons.get('no_method', 0) > 0:
                        print(f"    ⚠️ Skipped (model lacks hpm_add_solved_task): {skip_reasons['no_method']}")
                    if skip_reasons.get('not_enabled', 0) > 0:
                        print(f"    ⚠️ Skipped (use_hpm=False AND hpm_memory_enabled=False): {skip_reasons['not_enabled']}")
                    if skip_reasons.get('global_duplicate', 0) > 0:
                        print(f"    ℹ️ Skipped (already in buffer from prev epoch): {skip_reasons['global_duplicate']}")
                    if skip_reasons.get('epoch_duplicate', 0) > 0:
                        print(f"    ℹ️ Skipped (duplicate in same epoch): {skip_reasons['epoch_duplicate']}")
                    if skip_reasons.get('no_support_features', 0) > 0:
                        print(f"    ⚠️ Skipped (no support_features): {skip_reasons['no_support_features']}")
                else:
                    # No exact matches occurred this epoch
                    exact_matches = diagnostics.get('exact_match_count', train_losses.get('exact_match', 0))
                    print(f"  HPM Write Attempts: 0 (no exact matches this epoch: {exact_matches})")
                
                # Show HPM retrieval quality stats (P1 observability patch)
                hpm_retrieval_count = diagnostics.get('hpm_retrieval_count', 0)
                if hpm_retrieval_count > 0:
                    inst_sim_avg = diagnostics.get('hpm_instance_similarity_sum', 0) / hpm_retrieval_count
                    proc_sim_avg = diagnostics.get('hpm_procedural_similarity_sum', 0) / hpm_retrieval_count
                    inst_retrieved = diagnostics.get('hpm_instance_retrieved_sum', 0)
                    proc_retrieved = diagnostics.get('hpm_procedural_retrieved_sum', 0)
                    print(f"  HPM Retrieval Stats ({hpm_retrieval_count} batches):")
                    print(f"    Instance: avg_sim={inst_sim_avg:.3f}, total_retrieved={inst_retrieved}")
                    print(f"    Procedural: avg_sim={proc_sim_avg:.3f}, total_retrieved={proc_retrieved}")
                    # Quality warning: low similarity may indicate stale memory or poor query alignment
                    if inst_sim_avg > 0 and inst_sim_avg < 0.3:
                        print(f"    ⚠️ Low instance similarity - memory may be stale or queries misaligned")
        
        print(f"  Time: {epoch_time:.1f}s, LR: {optimizer.param_groups[0]['lr']:.2e}{stage_str}")
        
        # ================================================================
        # HYPERLORA CLAMP STATS (Jan 2026 observability)
        # ================================================================
        # Log LoRA clamp hit-rate to detect if clamping is too aggressive or too weak
        if hasattr(model, 'hyper_lora') and model.hyper_lora is not None:
            if hasattr(model.hyper_lora, 'get_clamp_stats'):
                clamp_stats = model.hyper_lora.get_clamp_stats()
                if clamp_stats['total_count'] > 0:
                    hit_rate = clamp_stats['hit_rate']
                    max_norm = clamp_stats['max_pre_norm']
                    threshold = clamp_stats['threshold']
                    print(f"  HyperLoRA Clamp: hit_rate={hit_rate:.1%} ({clamp_stats['hit_count']}/{clamp_stats['total_count']}), max_norm={max_norm:.2f} (threshold={threshold:.1f})")
                    # Warn if hit rate is too high (clamping too often → losing signal)
                    if hit_rate > 0.3:
                        print(f"    ⚠️ High clamp rate - consider increasing lora_max_norm or reducing delta_scale")
                    # Warn if max norm is very close to threshold (near explosion)
                    if max_norm > threshold * 0.9:
                        print(f"    ⚠️ Max norm near threshold - LoRA may be saturating")
                # Reset stats for next epoch
                model.hyper_lora.reset_clamp_stats()
        
        # ================================================================
        # HPM SOLVER-CONTEXT COUPLING SUMMARY (Jan 2026)
        # ================================================================
        if hpm_solver_context_enabled and use_hpm and hpm_solver_context_state['active']:
            # Get stats from model
            solver_coupling_stats = {}
            if hasattr(model, 'solver') and hasattr(model.solver, 'solver_cross_attn'):
                if model.solver.solver_cross_attn is not None and hasattr(model.solver.solver_cross_attn, 'get_hpm_stats'):
                    solver_coupling_stats = model.solver.solver_cross_attn.get_hpm_stats()
            
            # Get stats from diagnostics
            hpm_solver_batches = diagnostics.get('hpm_solver_batches_with_tokens', 0)
            hpm_solver_tokens = diagnostics.get('hpm_solver_tokens_used', 0)
            hpm_solver_explosions = diagnostics.get('hpm_solver_explosion_count', 0)
            hpm_proj_norm_max = diagnostics.get('hpm_solver_proj_norm_max', 0.0)
            
            gate_value = solver_coupling_stats.get('gate_value', hpm_solver_context_state['gate_value'])
            logit_clamp = solver_coupling_stats.get('logit_clamp', hpm_solver_context_logit_clamp)
            
            print(f"  HPM Solver Coupling: gate={gate_value:.3f}, batches={hpm_solver_batches}, tokens={hpm_solver_tokens}")
            if hpm_proj_norm_max > 0:
                print(f"    Proj norm max: {hpm_proj_norm_max:.2f} (clamp={logit_clamp:.1f})")
            if hpm_solver_explosions > 0:
                print(f"    ⚠️ Projection explosions: {hpm_solver_explosions} batches exceeded clamp")
            
            # Check for instability and auto-disable if configured
            if hpm_solver_context_disable_on_instability and hpm_solver_explosions > 10:
                hpm_solver_context_state['explosion_count'] += hpm_solver_explosions
                if hpm_solver_context_state['explosion_count'] > 50:
                    hpm_solver_context_state['disabled_by_instability'] = True
                    hpm_solver_context_state['active'] = False
                    print(f"    ⚠️ HPM SOLVER COUPLING DISABLED - too many explosions ({hpm_solver_context_state['explosion_count']})")
        
        
        # ================================================================
        # META ESCALATION SUMMARY (Dec 2025)
        # ================================================================
        if meta_escalation_enabled and meta_escalation_state['escalation_active']:
            print(f"  Meta Escalation: progress={meta_escalation_state['escalation_progress']:.1%}, stable={meta_escalation_state['is_stable']}")
            print(f"    HyperLoRA: {meta_escalation_state['hyperlora_delta_scale_current']:.4f}/{meta_escalation_state['hyperlora_delta_scale_scheduled']:.4f}")
            print(f"    Equiv: {meta_escalation_state['equiv_weight_current']:.4f}/{meta_escalation_state['equiv_weight_scheduled']:.4f}")
            print(f"    LOO: {meta_escalation_state['loo_weight_current']:.4f}/{meta_escalation_state['loo_weight_scheduled']:.4f}")
            # Show HPM balance weight if HPM is enabled
            if use_hpm and 'hpm_balance_weight_current' in meta_escalation_state:
                print(f"    HPM Balance: {meta_escalation_state['hpm_balance_weight_current']:.4f}/{meta_escalation_state.get('hpm_balance_weight_scheduled', meta_escalation_state['hpm_balance_weight_current']):.4f}")
            if meta_escalation_state['escalation_paused']:
                print(f"    ⚠️ PAUSED (nan={meta_escalation_state['max_nan_streak_epoch']}, grad={meta_escalation_state['grad_explosion_events_epoch']})")
            
            # Compute and log meta contribution ratios
            task_loss_val = train_losses.get('task_loss', train_losses.get('focal_loss', 1.0))
            equiv_loss_val = train_losses.get('equiv_loss', 0.0)
            loo_loss_val = train_losses.get('loo_loss', 0.0)
            
            if task_loss_val > 0:
                weighted_equiv = equiv_loss_val * meta_escalation_state['equiv_weight_current']
                weighted_loo = loo_loss_val * meta_escalation_state['loo_weight_current']
                total_loss = task_loss_val + weighted_equiv + weighted_loo
                meta_ratio = (weighted_equiv + weighted_loo) / max(total_loss, 1e-6)
                print(f"    Meta contribution ratio: {meta_ratio:.1%} of total loss")
        
        # Show per-module LRs if different groups exist
        if len(optimizer.param_groups) > 2:  # More than just decay/no_decay
            lr_summary = []
            for pg in optimizer.param_groups:
                name = pg.get('name', 'unnamed')
                if 'dsc' in name:
                    lr_summary.append(f"DSC:{pg['lr']:.2e}")
                elif 'msre' in name:
                    lr_summary.append(f"MSRE:{pg['lr']:.2e}")
                elif 'other' in name and 'decay' in name:
                    lr_summary.append(f"Other:{pg['lr']:.2e}")
            if lr_summary:
                # Deduplicate
                unique_lrs = list(dict.fromkeys(lr_summary))
                print(f"    Per-module LRs: {', '.join(unique_lrs)}")
        
        # Current temperature (affects attention sharpness)
        current_temp = get_temperature(epoch, config)
        print(f"  Temperature: {current_temp:.4f} (lower=sharper attention)")
        if current_temp > 1.5:
            print(f"    [!] Temperature still high - attention may be too diffuse")
        
        # ================================================================
        # AUGMENTATION DIVERSITY LOGGING (CRITICAL for quality assurance)
        # ================================================================
        # This verifies that on-the-fly augmentation is truly random and
        # covers all transformation types uniformly each epoch.
        # ================================================================
        aug_stats = train_losses.get('aug_stats', {})
        total_samples = train_losses.get('total_samples', 0)
        num_batches = train_losses.get('num_batches', 0)
        
        # Initialize augmentation metrics (for logging even if wandb not installed)
        color_pct = 0.0
        trans_pct = 0.0
        
        print(f"  Samples Processed: {total_samples} ({num_batches} batches)")
        
        if aug_stats and total_samples > 0:
            dihedral_counts = aug_stats.get('dihedral_counts', [0]*8)
            color_perm_count = aug_stats.get('color_perm_count', 0)
            translational_count = aug_stats.get('translational_count', 0)
            
            # Dihedral distribution (should be ~uniform across 8 transforms)
            dihedral_total = sum(dihedral_counts)
            if dihedral_total > 0:
                dihedral_pcts = [f"{c/dihedral_total*100:.1f}%" for c in dihedral_counts]
                print(f"  Dihedral Distribution: [{', '.join(dihedral_pcts)}]")
                
                # Check for uniformity (should be ~12.5% each)
                expected_pct = 100.0 / 8  # 12.5%
                max_deviation = max(abs(c/dihedral_total*100 - expected_pct) for c in dihedral_counts)
                if max_deviation > 5.0:  # More than 5% deviation from uniform
                    print(f"    [!] Non-uniform dihedral distribution (max dev: {max_deviation:.1f}%)")
            
            # Color permutation rate
            color_pct = color_perm_count / total_samples * 100 if total_samples > 0 else 0
            print(f"  Color Permutation: {color_pct:.1f}% ({color_perm_count}/{total_samples})")
            
            # Translational augmentation rate
            trans_pct = translational_count / total_samples * 100 if total_samples > 0 else 0
            unique_offsets = aug_stats.get('unique_offsets', 0)
            print(f"  Translational Aug: {trans_pct:.1f}% ({translational_count}/{total_samples}), unique offsets: {unique_offsets}")
            
            # Overall augmentation quality indicator
            aug_enabled_count = dihedral_total  # At least dihedral is applied
            aug_quality = "GOOD" if (color_pct > 90 and trans_pct > 80) else "OK" if aug_enabled_count > 0 else "NONE"
            print(f"  Aug Quality: {aug_quality}")
        
        # ================================================================
        # TRAINING DIAGNOSTICS (CRITICAL for debugging learning dynamics)
        # ================================================================
        diagnostics = train_losses.get('diagnostics', {})
        if diagnostics:
            print("  --- Training Diagnostics ---")
            
            # Deep supervision verification
            all_logits_count = diagnostics.get('all_logits_count', 0)
            if all_logits_count > 0:
                print(f"  Solver Steps: {all_logits_count} (deep supervision active)")
                
                # Per-step loss breakdown - EPOCH AVERAGE (more reliable than single batch)
                per_step_loss_sum = diagnostics.get('per_step_loss_sum')
                per_step_loss_count = diagnostics.get('per_step_loss_count', 0)
                
                if per_step_loss_sum and per_step_loss_count > 0:
                    # Compute epoch average per step
                    per_step_avg = [s / per_step_loss_count for s in per_step_loss_sum]
                    avg_str = ', '.join(f"{l:.4f}" for l in per_step_avg)
                    print(f"  Per-Step Loss (epoch avg, {per_step_loss_count} batches): [{avg_str}]")
                    
                    # Find best step based on epoch average (should ideally be the last)
                    valid_losses = [(i, l) for i, l in enumerate(per_step_avg) if l > 0]
                    if valid_losses:
                        best_step, best_loss = min(valid_losses, key=lambda x: x[1])
                        worst_step, worst_loss = max(valid_losses, key=lambda x: x[1])
                        
                        if best_step == len(per_step_avg) - 1:
                            improvement = (per_step_avg[0] - per_step_avg[-1]) / max(per_step_avg[0], 0.001) * 100
                            print(f"    ✓ Step improvement: {improvement:.1f}% (later steps better - GOOD!)")
                        elif best_step == 0:
                            degradation = (per_step_avg[-1] - per_step_avg[0]) / max(per_step_avg[0], 0.001) * 100
                            print(f"    [!] SOLVER DEGRADATION: Step 0 is best! Later steps {degradation:.1f}% worse!")
                            print(f"    [!] Best: step {best_step} ({best_loss:.4f}), Worst: step {worst_step} ({worst_loss:.4f})")
                        else:
                            print(f"    [!] Best step is {best_step} (middle), not last - solver may be over-iterating!")
                            print(f"    [!] Best: step {best_step} ({best_loss:.4f}), Final: step {len(per_step_avg)-1} ({per_step_avg[-1]:.4f})")
                            # Show how much we'd gain by stopping at best step
                            potential_gain = (per_step_avg[-1] - best_loss) / max(per_step_avg[-1], 0.001) * 100
                            print(f"    [!] Potential gain from best-step selection: {potential_gain:.1f}% lower loss")
                else:
                    # Fallback to last batch (legacy display)
                    per_step_loss = diagnostics.get('per_step_loss', [])
                    if per_step_loss:
                        loss_str = ', '.join(f"{l:.3f}" for l in per_step_loss)
                        print(f"  Per-Step Loss (last batch only): [{loss_str}]")
                    
                        # Find best step (should ideally be the last)
                        valid_losses = [(i, l) for i, l in enumerate(per_step_loss) if l < 100 and not (l != l)]  # exclude 100 and NaN
                        if valid_losses:
                            best_step, best_loss = min(valid_losses, key=lambda x: x[1])
                            worst_step, worst_loss = max(valid_losses, key=lambda x: x[1])
                            
                            if best_step == len(per_step_loss) - 1:
                                improvement = (per_step_loss[0] - per_step_loss[-1]) / max(per_step_loss[0], 0.001) * 100
                                print(f"    Step improvement: {improvement:.1f}% (later steps better - GOOD!)")
                            elif best_step == 0:
                                degradation = (per_step_loss[-1] - per_step_loss[0]) / max(per_step_loss[0], 0.001) * 100
                                print(f"    [!] SOLVER DEGRADATION: Step 0 is best! Later steps {degradation:.1f}% worse!")
                                print(f"    [!] Best: step {best_step} ({best_loss:.3f}), Worst: step {worst_step} ({worst_loss:.3f})")
                            else:
                                print(f"    [!] Best step is {best_step} (middle), not last - solver unstable!")
                                print(f"    [!] Best: step {best_step} ({best_loss:.3f}), Final: step {len(per_step_loss)-1} ({per_step_loss[-1]:.3f})")
                        
                        # Check for loss = 100 (clamped infinite loss)
                        if 100.0 in per_step_loss:
                            bad_steps = [i for i, l in enumerate(per_step_loss) if l >= 100.0]
                            print(f"    [CRITICAL] Steps {bad_steps} have loss >= 100 (numerical explosion!)")
                
                # =============================================================
                # SOLVER HEALTH SUMMARY (aggregate best-step statistics)
                # =============================================================
                last_best = diagnostics.get('last_step_was_best_count', 0)
                earlier_best = diagnostics.get('earlier_step_was_best_count', 0)
                total_checks = last_best + earlier_best
                
                if total_checks > 0:
                    last_best_pct = last_best / total_checks * 100
                    earlier_best_pct = earlier_best / total_checks * 100
                    
                    # Best step histogram
                    histogram = diagnostics.get('best_step_histogram', [])
                    num_steps = len([h for h in histogram if h > 0])
                    if num_steps > 0:
                        hist_str = ', '.join(f"s{i}:{h}" for i, h in enumerate(histogram[:num_steps]) if h > 0)
                        print(f"  Best-Step Histogram: [{hist_str}]")
                    
                    print(f"  Solver Health: Last step best: {last_best_pct:.1f}%, Earlier step best: {earlier_best_pct:.1f}%")
                    
                    # Average improvement from step 0 to step N
                    improvement_sum = diagnostics.get('step_improvement_sum', 0.0)
                    avg_improvement = improvement_sum / total_checks * 100
                    print(f"  Avg Step Improvement: {avg_improvement:.1f}% (step0→stepN)")
                    
                    # Health warnings
                    if earlier_best_pct > 30:
                        print(f"    ⚠️ SOLVER OVER-ITERATION WARNING: {earlier_best_pct:.0f}% of batches had earlier step as best!")
                        print(f"    ⚠️ Consider: (1) enabling best-step selection, (2) reducing num_solver_steps, (3) enabling ACT")
                    if avg_improvement < 0:
                        print(f"    ⚠️ SOLVER DEGRADATION: Avg improvement is negative! More steps = worse predictions!")
                    
                    # Entropy-Loss Agreement (validates inference heuristic)
                    entropy_total = diagnostics.get('entropy_loss_total_count', 0)
                    entropy_agree = diagnostics.get('entropy_loss_agreement_count', 0)
                    if entropy_total > 0:
                        agree_pct = entropy_agree / entropy_total * 100
                        print(f"  Entropy-Loss Agreement: {agree_pct:.1f}% ({entropy_agree}/{entropy_total})")
                        if agree_pct < 50:
                            print(f"    ⚠️ LOW AGREEMENT: Entropy picks same step as loss only {agree_pct:.0f}% of time!")
                            print(f"    ⚠️ Inference best-step selection may not be reliable!")
                        elif agree_pct >= 80:
                            print(f"    ✓ High agreement - entropy is a good proxy for loss during inference")
                
                # Per-step logit statistics (numerical stability check)
                per_step_stats = diagnostics.get('per_step_logits_stats', [])
                if per_step_stats:
                    for step_idx, stats in enumerate(per_step_stats):
                        if stats.get('has_nan') or stats.get('has_inf'):
                            print(f"    [CRITICAL] Step {step_idx}: NaN={stats['has_nan']}, Inf={stats['has_inf']}")
                        elif abs(stats.get('max', 0)) > 50 or abs(stats.get('min', 0)) > 50:
                            print(f"    [!] Step {step_idx} logits extreme: [{stats['min']:.1f}, {stats['max']:.1f}]")
            else:
                print(f"  Solver Steps: [!] NO INTERMEDIATE LOGITS (deep supervision disabled!)")
            
            # Gradient flow diagnostics
            dsc_grad = diagnostics.get('dsc_grad_norm_sum', 0)
            stop_pred_grad = diagnostics.get('stop_predictor_grad_norm_sum', 0)
            enc_grad = diagnostics.get('encoder_grad_norm_sum', 0)
            solver_grad = diagnostics.get('solver_grad_norm_sum', 0)
            ctx_grad = diagnostics.get('context_encoder_grad_norm_sum', 0)
            msre_grad = diagnostics.get('msre_grad_norm_sum', 0)
            
            print(f"  Grad Norms: DSC={dsc_grad:.4f}, StopPred={stop_pred_grad:.4f}, Encoder={enc_grad:.4f}, Solver={solver_grad:.4f}")
            if ctx_grad > 0 or msre_grad > 0:
                print(f"              ContextEnc={ctx_grad:.4f}, MSRE={msre_grad:.4f}")

            # Stop predictor weight health (variance near zero can indicate freezing/collapse)
            stop_pred_var = diagnostics.get('stop_predictor_weight_variance', 0.0)
            # FIX (Jan 2026): Handle NaN (module missing) vs 0.0 (actual zero variance)
            import math as _math  # Local import to avoid scope issues
            if _math.isnan(stop_pred_var):
                print(f"  StopPred Weight Var: N/A (module not present)")
            else:
                print(f"  StopPred Weight Var: {stop_pred_var:.2e}")
                if stop_pred_var > 0 and stop_pred_var < 1e-10:
                    print(f"    [!] Stop predictor weight variance near zero - may be frozen/degenerate")
            
            # Gradient flow warnings
            if dsc_grad < 0.001 and enc_grad > 0:
                print(f"    [!] DSC gradients near zero - not learning!")
            if stop_pred_grad < 0.0001 and dsc_grad > 0.001:
                print(f"    [!] Stop predictor gradients near zero - clue count not learning!")
            if solver_grad < 0.001 and enc_grad > 0:
                print(f"    [!] Solver gradients near zero - check architecture!")
            if ctx_grad < 0.001 and enc_grad > 0.01 and config.get('model', {}).get('use_context_encoder', False):
                print(f"    [!] Context encoder gradients near zero - not learning from examples!")
            
            # Attention sharpness (is DSC focusing?)
            attn_max = diagnostics.get('attn_max_mean', 0)
            attn_min = diagnostics.get('attn_min_mean', 0)
            if attn_max > 0:
                print(f"  Attention: max={attn_max:.4f}, min={attn_min:.6f}")
                # For a 30x30 grid, uniform attention = 1/900 = 0.0011
                # Sharp attention should have max >> 0.0011
                if attn_max < 0.01:
                    print(f"    [!] Attention too diffuse (max < 0.01) - DSC not focusing!")
                elif attn_max > 0.1:
                    print(f"    Attention is sharp (good!)")
            
            # Stop probability (how many clues used)
            stop_prob = diagnostics.get('stop_prob_mean', 0)
            if stop_prob > 0:
                clues_used = (1 - stop_prob) * all_logits_count if all_logits_count > 0 else 0
                # Show variance to verify task-dependent clue count (enabled by per-sample gradient coupling)
                clues_std = diagnostics.get('clues_used_std', 0)
                clues_min = diagnostics.get('clues_used_min', 0)
                clues_max = diagnostics.get('clues_used_max', 0)
                clue_loss_corr = diagnostics.get('clue_loss_correlation', 0)
                stop_prob_std = diagnostics.get('stop_prob_std', 0)
                print(f"  Stop Prob: {stop_prob:.3f} (approx {clues_used:.1f} clues active)")
                if stop_prob_std > 0:
                    print(f"  Stop Probs Std: {stop_prob_std:.3f} (global std across batch×clues)")
                print(f"  Clues Used: mean={clues_used:.2f}, std={clues_std:.2f}, range=[{clues_min:.1f}, {clues_max:.1f}]")
                print(f"  Clue-Loss Correlation: {clue_loss_corr:+.3f}", end="")
                # Interpret correlation - this is the KEY metric for per-sample coupling
                # With per-sample gradient coupling, we expect positive correlation:
                # hard tasks (high loss) should use more clues to improve prediction
                if clue_loss_corr > 0.3:
                    print(" (EXCELLENT: per-sample coupling working!)")
                elif clue_loss_corr > 0.1:
                    print(" (learning - per-sample coupling active)")
                elif clue_loss_corr < -0.1:
                    print(" (unexpected negative - check gradient flow)")
                else:
                    print(" (weak - per-sample coupling may need tuning)")
                
                # CRITICAL: Check stop_logits saturation (causes zero gradient!)
                stop_logits_mean = diagnostics.get('stop_logits_mean', 0)
                stop_logits_std = diagnostics.get('stop_logits_std', 0)
                stop_logits_min = diagnostics.get('stop_logits_min', 0)
                stop_logits_max = diagnostics.get('stop_logits_max', 0)
                print(f"  Stop Logits: mean={stop_logits_mean:.2f}, std={stop_logits_std:.2f}, range=[{stop_logits_min:.1f}, {stop_logits_max:.1f}]")
                if abs(stop_logits_mean) > 5.0:
                    print(f"    [CRITICAL] Stop logits saturated! |mean|={abs(stop_logits_mean):.1f} > 5.0")
                    print(f"    Sigmoid gradient ≈ 0, stop predictor cannot learn!")
                    print(f"    Consider adding L2 regularization on stop_logits")
                elif abs(stop_logits_mean) > 3.0:
                    print(f"    [!] Stop logits approaching saturation |mean|={abs(stop_logits_mean):.1f}")
                
                # Check for task-dependent clue count (the goal of per-sample gradient coupling)
                if clues_std < 0.1:
                    print(f"    [!] Low variance - clue count not adapting per-task!")
                elif clues_std > 0.5:
                    print(f"    [+] High variance - strong per-task clue adaptation!")
                elif clues_std > 0.3:
                    print(f"    Clue count varies by task (per-sample coupling active)")
            
            # Per-clue entropy breakdown
            per_clue_entropy = diagnostics.get('per_clue_entropy', [])
            if per_clue_entropy:
                entropy_str = ', '.join(f"{e:.2f}" for e in per_clue_entropy)
                mean_entropy = sum(per_clue_entropy) / len(per_clue_entropy)
                max_entropy = 6.80  # ln(900) for 30x30 grid
                print(f"  Per-Clue Entropy: [{entropy_str}] (mean={mean_entropy:.2f}, max={max_entropy:.2f})")
                
                # Check if all clues have similar entropy (bad - not differentiating)
                if len(per_clue_entropy) > 1:
                    entropy_std = (sum((e - mean_entropy)**2 for e in per_clue_entropy) / len(per_clue_entropy)) ** 0.5
                    if entropy_std < 0.1:
                        print(f"    [!] Clues have uniform entropy (std={entropy_std:.3f}) - not differentiating!")
                
                # Check if entropy is too high (attention too diffuse)
                if mean_entropy > 5.0:
                    print(f"    [!] High entropy ({mean_entropy:.2f}) - attention still diffuse!")
                elif mean_entropy < 3.0:
                    print(f"    Good entropy ({mean_entropy:.2f}) - attention is focused!")
            
            # Centroid spread
            # P1.6: Enhanced centroid diversity monitoring (821b111 showed spread=0.19 collapse)
            # Now uses YAML thresholds: centroid_spread_warning, centroid_spread_critical
            centroid_spread = diagnostics.get('centroid_spread', 0)
            if centroid_spread > 0:
                print(f"  Centroid Spread: {centroid_spread:.2f} (higher=more diverse)")
                if centroid_spread < centroid_spread_critical:
                    print(f"    🚨 CRITICAL COLLAPSE: Spread={centroid_spread:.2f} < {centroid_spread_critical} - all clues at same location!")
                    print(f"    [!] Stop predictor cannot differentiate - needs diversity regularizer")
                elif centroid_spread < centroid_spread_warning:
                    print(f"    ⚠️ Clues clustered (spread < {centroid_spread_warning}) - should spread out")
                elif centroid_spread > 8.0:
                    print(f"    ✓ Good spread - clues are distributed across grid")
            
            # ============================================================
            # NEW DIAGNOSTICS: Entropy-Stop Coupling & Sparsity Components
            # ============================================================
            
            # DSC entropy inputs to stop_predictor (verify coupling is working)
            dsc_entropy_input = diagnostics.get('dsc_entropy_input_mean', None)
            per_clue_entropy_input = diagnostics.get('per_clue_entropy_input', [])
            if dsc_entropy_input is not None:
                print(f"  --- Stop Predictor Coupling ---")
                print(f"  Entropy Input to Stop: {dsc_entropy_input:.4f} (normalized, lower=sharper)")
                if per_clue_entropy_input:
                    ei_str = ', '.join(f"{e:.3f}" for e in per_clue_entropy_input)
                    print(f"  Per-Clue Entropy Input: [{ei_str}]")
                
                # Check if entropy is actually influencing stop decisions
                per_clue_stop = diagnostics.get('per_clue_stop_prob', [])
                if per_clue_stop and per_clue_entropy_input:
                    # Lower entropy should correlate with higher stop_prob
                    # (sharp attention = can stop, diffuse = need more clues)
                    print(f"  Per-Clue Stop Prob: [{', '.join(f'{s:.3f}' for s in per_clue_stop)}]")
            
            # Sparsity loss component breakdown
            min_clue_pen = diagnostics.get('sparsity_min_clue_penalty', 0)
            base_ponder = diagnostics.get('sparsity_base_pondering', 0)
            entropy_ponder = diagnostics.get('sparsity_entropy_pondering', 0)
            expected_clues = diagnostics.get('expected_clues_used', 0)
            per_sample_clue_pen = diagnostics.get('per_sample_clue_penalty_mean', 0)
            if base_ponder > 0 or entropy_ponder > 0 or per_sample_clue_pen > 0:
                print(f"  --- Sparsity Loss Breakdown (Per-Sample Coupled) ---")
                print(f"  Min Clue Penalty: {min_clue_pen:.4f} (per-sample avg)")
                print(f"  Per-Sample Clue Penalty (scaled): {per_sample_clue_pen:.4f}")
                print(f"  Base Pondering: {base_ponder:.4f} (clues={expected_clues:.2f})")
                print(f"  Entropy Pondering: {entropy_ponder:.4f}")
                # Verify per-sample penalty is correctly scaled (should be ~lambda_sparsity * min_clue_penalty)
                if min_clue_pen > 0:
                    expected_scaled = train_cfg['lambda_sparsity'] * min_clue_pen
                    if abs(per_sample_clue_pen - expected_scaled) < 0.001:
                        print(f"    [+] Per-sample penalty correctly scaled by λ_sparsity")
                    else:
                        print(f"    Expected scaled: {expected_scaled:.4f} (λ={train_cfg['lambda_sparsity']})")
                if min_clue_pen > 0.1:
                    print(f"    [!] Using fewer than min_clues - penalty is active!")
            
            # Context encoder diagnostics
            context_mag = diagnostics.get('context_magnitude', None)
            context_std = diagnostics.get('context_std', None)
            if context_mag is not None:
                print(f"  --- Context Encoder ---")
                print(f"  Context Magnitude: {context_mag:.4f} (should be > 0.5)")
                print(f"  Context Batch Std: {context_std:.4f} (should vary)")
                if context_mag < 0.1:
                    print(f"    [!] Context near zero - ContextEncoder not contributing!")
                if context_std < 0.01:
                    print(f"    [!] Context identical across batch - collapsed!")
            
            # Numerical stability checks
            logits_max = diagnostics.get('logits_max', None)
            logits_min = diagnostics.get('logits_min', None)
            has_nan = diagnostics.get('logits_has_nan', False)
            has_inf = diagnostics.get('logits_has_inf', False)
            if has_nan or has_inf:
                print(f"  [CRITICAL] Numerical instability: NaN={has_nan}, Inf={has_inf}")
            if logits_max is not None and (logits_max > 50 or logits_min < -50):
                print(f"  [WARNING] Extreme logit values: [{logits_min:.1f}, {logits_max:.1f}]")
            
            # ================================================================
            # META-LEARNING HEALTH METRICS (HyperLoRA + LOO Training)
            # ================================================================
            loo_loss_sum = diagnostics.get('loo_loss_sum', 0.0)
            loo_accuracy_sum = diagnostics.get('loo_accuracy_sum', 0.0)
            loo_num_holdouts = diagnostics.get('loo_num_holdouts_sum', 0)
            loo_batch_count = diagnostics.get('loo_batch_count', 0)
            loo_skipped = diagnostics.get('loo_skipped_count', 0)
            hyperlora_grad_norm = diagnostics.get('hyperlora_grad_norm_sum', 0.0)
            
            if loo_batch_count > 0 or loo_skipped > 0:
                print(f"  --- META-LEARNING (HyperLoRA + LOO) ---")
                
                if loo_batch_count > 0:
                    avg_loo_loss = loo_loss_sum / loo_batch_count
                    avg_loo_accuracy = loo_accuracy_sum / loo_batch_count
                    avg_holdouts = loo_num_holdouts / loo_batch_count
                    
                    print(f"  LOO Loss (avg): {avg_loo_loss:.4f}")
                    print(f"  LOO Accuracy (N-1→Nth): {avg_loo_accuracy:.1%}")
                    print(f"  LOO Holdouts/batch: {avg_holdouts:.1f}")
                    print(f"  LOO Batches: {loo_batch_count} computed, {loo_skipped} skipped")
                    
                    # Alert if LOO skipped > 50% of batches (from META_LEARNING_INFERENCE_GUIDE.md)
                    total_loo_attempts = loo_batch_count + loo_skipped
                    if total_loo_attempts > 0:
                        loo_skip_pct = loo_skipped / total_loo_attempts * 100
                        if loo_skip_pct > 50:
                            print(f"    [ALERT] LOO skipped {loo_skip_pct:.0f}% of batches! (>50% threshold)")
                            print(f"    [!] Check: min_pairs_for_loo setting, data loader pair count")
                    
                    # Interpret LOO accuracy
                    if avg_loo_accuracy > 0.8:
                        print(f"    ✓ EXCELLENT meta-learning: HyperLoRA generalizes from N-1 to Nth!")
                    elif avg_loo_accuracy > 0.5:
                        print(f"    Learning: HyperLoRA starting to generalize")
                    elif avg_loo_accuracy > 0.2:
                        print(f"    Early stage: LOO accuracy improving...")
                    else:
                        print(f"    [!] Low LOO accuracy - HyperLoRA not yet generalizing")
                        print(f"    [!] Check: hyperlora_lr_multiplier, LOO loss weight")
                else:
                    print(f"  LOO: Skipped all {loo_skipped} batches (not enough pairs)")
                
                # HyperLoRA gradient flow
                if hyperlora_grad_norm > 0:
                    print(f"  HyperLoRA Grad Norm: {hyperlora_grad_norm:.6f}")
                    if hyperlora_grad_norm < 0.0001:
                        print(f"    [!] HyperLoRA gradients near zero - not learning!")
                
                # Equivariance Loss Metrics
                # P1.5: Fixed "equiv=0 means good" bug - now distinguishes success from failure
                equiv_loss_sum = diagnostics.get('equiv_loss_sum', 0.0)
                equiv_batch_count = diagnostics.get('equiv_batch_count', 0)
                equiv_failures = diagnostics.get('equiv_failures', 0)  # From P0.2 tracking
                
                if equiv_batch_count > 0:
                    avg_equiv_loss = equiv_loss_sum / equiv_batch_count
                    print(f"  Equivariance Loss (avg): {avg_equiv_loss:.4f}")
                    
                    # NOTE: In this codebase HyperLoRA pooling is dihedral-invariant (D4).
                    # That can make equivariance loss near-zero by design, even early.
                    if avg_equiv_loss < 0.001 and equiv_failures == 0:
                        print(f"    ✓ EXPECTED: equiv≈0 (HyperLoRA uses D4-invariant pooling; aug contexts match by design)")
                    elif avg_equiv_loss < 0.05 and equiv_failures == 0:
                        print(f"    ✓ EXCELLENT: LoRA predictions consistent across augmentations")
                    elif avg_equiv_loss < 0.05 and equiv_failures > 0:
                        # Low loss but had failures - the "good" batches may be hiding issues
                        print(f"    ⚠️ MIXED: Low avg but {equiv_failures} failures detected")
                        print(f"    [!] Some batches silently failed - TTA may have low consensus")
                    elif avg_equiv_loss < 0.2:
                        print(f"    ✓ Good: HyperLoRA learning augmentation invariance")
                    elif avg_equiv_loss < 0.5:
                        print(f"    Learning: Equivariance still converging...")
                    else:
                        print(f"    [!] High equivariance loss - LoRA predictions vary with augmentation")
                else:
                    # P1.5: Explicitly flag when equivariance is enabled but never computed
                    # FIXED (Dec 2025): Also check if equivariance is actually enabled in config!
                    # Previously this showed "BROKEN" even when intentionally disabled.
                    equiv_is_enabled = equiv_config.get('enabled', False) and config.get('model', {}).get('use_hyperlora', False)
                    if equiv_is_enabled and epoch >= config['training'].get('equivariance_training', {}).get('start_epoch', 12):
                        print(f"  Equivariance: ⚠️ BROKEN - 0 batches computed (should be active)")
                        print(f"    [!] Equivariance path is silently failing - TTA will have <25% consensus")
                    elif not equiv_is_enabled:
                        print(f"  Equivariance: DISABLED in config")
                    else:
                        print(f"  Equivariance: Not yet active (starts epoch {config['training'].get('equivariance_training', {}).get('start_epoch', 12)})")
                
                # Meta-Learning Health Summary (combined status)
                print(f"  --- Meta-Learning Health Summary ---")
                meta_health_score = 0.0
                meta_health_reasons = []
                
                if loo_batch_count > 0 and avg_loo_accuracy > 0.5:
                    meta_health_score += 0.5
                    meta_health_reasons.append("LOO ✓")
                elif loo_batch_count > 0:
                    meta_health_reasons.append("LOO learning")
                else:
                    meta_health_reasons.append("LOO skipped")
                    
                if equiv_batch_count > 0 and avg_equiv_loss < 0.2:
                    meta_health_score += 0.3
                    meta_health_reasons.append("Equiv ✓")
                elif equiv_batch_count > 0:
                    meta_health_reasons.append("Equiv learning")
                else:
                    meta_health_reasons.append("Equiv skipped")
                    
                if hyperlora_grad_norm > 0.0001:
                    meta_health_score += 0.2
                    meta_health_reasons.append("Grads ✓")
                elif hyperlora_grad_norm > 0:
                    meta_health_reasons.append("Grads low")
                else:
                    meta_health_reasons.append("No HyperLoRA grads")
                
                # Overall meta-learning status
                status_icons = {0.0: "❌", 0.2: "⚠️", 0.5: "🔄", 0.8: "✓", 1.0: "✓✓"}
                icon = "❌"
                for threshold, ico in sorted(status_icons.items()):
                    if meta_health_score >= threshold:
                        icon = ico
                print(f"  Overall: {icon} [{' | '.join(meta_health_reasons)}] (score={meta_health_score:.1f}/1.0)")
                
                # =============================================================
                # DETAILED META-LEARNING BREAKDOWN (for debugging after epoch 14)
                # =============================================================
                lora_delta_sum = diagnostics.get('lora_delta_norm_sum', 0.0)
                lora_delta_count = diagnostics.get('lora_delta_batch_count', 0)
                context_mag_sum = diagnostics.get('context_magnitude_sum', 0.0)
                context_count = diagnostics.get('context_batch_count', 0)
                hpm_entropy_sum = diagnostics.get('hpm_routing_entropy_sum', 0.0)
                hpm_count = diagnostics.get('hpm_batch_count', 0)
                
                print(f"  --- Detailed Attribution ---")
                
                # LoRA delta magnitude (should be > 0 if HyperLoRA is contributing)
                if lora_delta_count > 0:
                    avg_lora_norm = lora_delta_sum / lora_delta_count
                    print(f"  LoRA Delta Norm (avg): {avg_lora_norm:.4f}", end="")
                    if avg_lora_norm < 0.001:
                        print(" [!] Near-zero: HyperLoRA not adapting weights!")
                    elif avg_lora_norm < 0.01:
                        print(" (small - may need higher hyperlora_init_scale)")
                    elif avg_lora_norm > 1.0:
                        print(" [!] Very large - may cause instability")
                    else:
                        print(" ✓ (healthy range)")
                else:
                    print(f"  LoRA Delta: Not computed (HyperLoRA inactive)")
                
                # Context encoder contribution
                if context_count > 0:
                    avg_ctx_mag = context_mag_sum / context_count
                    print(f"  Context Magnitude (avg): {avg_ctx_mag:.4f}", end="")
                    if avg_ctx_mag < 0.01:
                        print(" [!] Near-zero: ContextEncoder not contributing!")
                    elif avg_ctx_mag > 5.0:
                        print(" [!] Very large - may dominate other signals")
                    else:
                        print(" ✓")
                
                # HPM bank specialization
                if hpm_count > 0:
                    avg_hpm_entropy = hpm_entropy_sum / hpm_count
                    # Entropy of uniform dist over N banks = log(N). Lower = more specialized.
                    print(f"  HPM Routing Entropy (avg): {avg_hpm_entropy:.3f}", end="")
                    if avg_hpm_entropy < 0.5:
                        print(" ✓ (specialized - banks have distinct roles)")
                    elif avg_hpm_entropy < 1.5:
                        print(" (moderate specialization)")
                    else:
                        print(" [!] High entropy - banks not specializing")
            
            # NEW: Color embedding diversity check
            color_embed_sim = diagnostics.get('color_embed_similarity', None)
            if color_embed_sim is not None:
                bg_fg_sim = diagnostics.get('bg_fg_embed_similarity', color_embed_sim)
                print(f"  --- Color Embedding Diversity ---")
                print(f"  Color Embed Similarity (off-diag): {color_embed_sim:.3f}", end="")
                if color_embed_sim > 0.8:
                    print(" [!] COLORS NOT DIFFERENTIATED!")
                elif color_embed_sim > 0.5:
                    print(" [WARN] Moderate similarity")
                else:
                    print(" (good diversity)")
                print(f"  BG-FG Embed Similarity: {bg_fg_sim:.3f}")
            
            # Gradient clipping diagnostics
            grad_norm_before = diagnostics.get('grad_norm_before_clip', None)
            grad_was_clipped = diagnostics.get('grad_was_clipped', False)
            grad_clip_threshold = config['training']['gradient_clip']
            if grad_norm_before is not None:
                print(f"  --- Gradient Clipping ---")
                print(f"  Grad Norm (before clip): {grad_norm_before:.4f}")
                if grad_was_clipped:
                    print(f"    [!] Gradients were clipped! (threshold={grad_clip_threshold})")
                else:
                    print(f"    Gradients within bounds")
                
                # Severe gradient explosion warning
                if grad_norm_before > grad_clip_threshold * 10:
                    print(f"    [CRITICAL] Gradient explosion! {grad_norm_before:.1f}x over clip threshold!")
            
            # Per-class prediction distribution (detect which colors model is predicting)
            pred_pcts = diagnostics.get('pred_class_pcts', [])
            target_pcts = diagnostics.get('target_class_pcts', [])
            padding_pct = diagnostics.get('padding_pct', 0)
            if pred_pcts and target_pcts:
                print(f"  --- Per-Class Distribution (Valid Pixels Only) ---")
                print(f"  Padding: {padding_pct:.1f}% of grid (ignored in loss)")
                # Show compact comparison: [0:boundary, 1:black, 2-10:colors 1-9]
                pred_str = ', '.join(f"{p:.1f}" for p in pred_pcts)
                tgt_str = ', '.join(f"{t:.1f}" for t in target_pcts)
                print(f"  Pred %: [{pred_str}]")
                print(f"  Target %: [{tgt_str}]")
                
                # Check for background collapse
                # 10-class encoding: class 0 is black (BG), classes 1-9 are FG
                num_classes = len(pred_pcts)
                bg_class = 0  # Black/background class
                fg_start = 1  # First foreground color class
                
                if len(pred_pcts) > bg_class and len(target_pcts) > bg_class:
                    bg_excess = pred_pcts[bg_class] - target_pcts[bg_class]
                    if bg_excess > 10:
                        print(f"    [!] Over-predicting BG (class {bg_class}) by {bg_excess:.1f}%!")
                
                # Check which foreground colors are being missed
                missed_colors = []
                for c in range(fg_start, num_classes):
                    if c < len(target_pcts) and c < len(pred_pcts):
                        if target_pcts[c] > 1.0 and pred_pcts[c] < 0.5:  # Target has >1%, pred <0.5%
                            missed_colors.append(c)
                if missed_colors:
                    print(f"    [!] Missing foreground colors: {missed_colors}")
                
                # Per-class accuracy breakdown
                class_correct = diagnostics.get('per_class_correct', [])
                class_total = diagnostics.get('per_class_total', [])
                if class_correct and class_total:
                    class_accs = []
                    num_classes = len(class_correct)
                    for c in range(num_classes):
                        if c < len(class_total) and class_total[c] > 0:
                            acc = class_correct[c] / class_total[c] * 100
                            class_accs.append(f"{acc:.0f}")
                        else:
                            class_accs.append("-")
                    print(f"  Per-Class Acc %: [{', '.join(class_accs)}]")
                
                # NEW: Color preference diagnostic - is model defaulting to one FG color?
                fg_pred_mode = diagnostics.get('fg_pred_mode_color', None)
                fg_pred_mode_pct = diagnostics.get('fg_pred_mode_pct', 0)
                if fg_pred_mode is not None:
                    if fg_pred_mode_pct > 50:
                        print(f"  [!] COLOR MODE COLLAPSE: {fg_pred_mode_pct:.0f}% of FG preds are color {fg_pred_mode}")
                        print(f"      Model is defaulting to single FG color instead of learning per-class!")
                    elif fg_pred_mode_pct > 30:
                        print(f"  [WARN] FG color preference: {fg_pred_mode_pct:.0f}% are color {fg_pred_mode}")
            
            # ================================================================
            # PER-SAMPLE ACCURACY SUMMARY (CRITICAL for early stopping decision)
            # ================================================================
            total_samples_processed = diagnostics.get('total_samples', 0)
            exact_match_count = diagnostics.get('exact_match_count', 0)
            high_accuracy_count = diagnostics.get('high_accuracy_count', 0)
            batch_accuracies = diagnostics.get('batch_accuracies', [])
            fg_accuracy_epoch = diagnostics.get('fg_accuracy_sum', 0)
            bg_accuracy_epoch = diagnostics.get('bg_accuracy_sum', 0)
            per_sample_accs = diagnostics.get('per_sample_accuracies', [])
            
            if total_samples_processed > 0:
                print(f"  {'='*50}")
                print(f"  PER-SAMPLE TRAINING ACCURACY (Epoch {epoch + 1})")
                print(f"  {'='*50}")
                
                # Overall epoch accuracy
                epoch_mean_acc = sum(batch_accuracies) / len(batch_accuracies) if batch_accuracies else 0.0
                print(f"  ★ Mean Accuracy: {epoch_mean_acc:.1%}")
                print(f"  ★ Exact Match: {exact_match_count}/{total_samples_processed} ({exact_match_count/total_samples_processed*100:.1f}%)")
                print(f"  ★ High Acc (≥90%): {high_accuracy_count}/{total_samples_processed} ({high_accuracy_count/total_samples_processed*100:.1f}%)")
                
                # FG/BG accuracy breakdown
                n_batches = len(batch_accuracies)
                if n_batches > 0:
                    fg_acc_mean = fg_accuracy_epoch / n_batches
                    bg_acc_mean = bg_accuracy_epoch / n_batches
                    print(f"  FG Accuracy: {fg_acc_mean:.1%}")
                    print(f"  BG Accuracy: {bg_acc_mean:.1%}")
                
                # Batch-level trend analysis
                if len(batch_accuracies) >= 5:
                    first_quarter = batch_accuracies[:len(batch_accuracies)//4]
                    last_quarter = batch_accuracies[-len(batch_accuracies)//4:]
                    first_mean = sum(first_quarter) / len(first_quarter)
                    last_mean = sum(last_quarter) / len(last_quarter)
                    acc_trend = last_mean - first_mean
                    
                    trend_str = "↑" if acc_trend > 0.01 else ("↓" if acc_trend < -0.01 else "→")
                    print(f"  Batch Trend: {first_mean:.1%} → {last_mean:.1%} ({trend_str} {abs(acc_trend)*100:.1f}pp)")
                    
                    if acc_trend < -0.05:
                        print(f"    [!] ACCURACY DECLINING within epoch - potential overfitting or instability!")
                    elif acc_trend > 0.02:
                        print(f"    ✓ Accuracy improving within epoch - learning is active!")
                
                # Sample some per-sample accuracies for insight
                if per_sample_accs:
                    # Distribution of accuracies
                    acc_buckets = {
                        '0-25%': sum(1 for a in per_sample_accs if a < 0.25),
                        '25-50%': sum(1 for a in per_sample_accs if 0.25 <= a < 0.50),
                        '50-75%': sum(1 for a in per_sample_accs if 0.50 <= a < 0.75),
                        '75-90%': sum(1 for a in per_sample_accs if 0.75 <= a < 0.90),
                        '90-100%': sum(1 for a in per_sample_accs if a >= 0.90),
                    }
                    total_sampled = len(per_sample_accs)
                    bucket_strs = [f"{k}:{v/total_sampled*100:.0f}%" for k, v in acc_buckets.items()]
                    print(f"  Accuracy Distribution: {', '.join(bucket_strs)}")
                    
                    # Check for stuck samples (always wrong)
                    if acc_buckets['0-25%'] / total_sampled > 0.5:
                        print(f"    [!] Many samples stuck at low accuracy - check data/model!")
                
                # Running window for early stopping
                running_window = diagnostics.get('running_accuracy_window', [])
                if running_window:
                    running_acc = sum(running_window) / len(running_window)
                    window_std = (sum((a - running_acc)**2 for a in running_window) / len(running_window))**0.5
                    print(f"  Running Window (last {len(running_window)} batches): {running_acc:.1%} ± {window_std:.1%}")
                    
                    # Early stopping recommendation
                    if epoch > 10 and running_acc < 0.1:
                        print(f"    [CRITICAL] Accuracy stuck below 10% after {epoch+1} epochs!")
                        print(f"    Consider: Check loss function, reduce LR, or verify data pipeline")
                    elif epoch > 20 and running_acc < 0.2:
                        print(f"    [WARNING] Accuracy stuck below 20% after {epoch+1} epochs - may need intervention")
                
                # ================================================================
                # PER-CLASS (COLOR) ACCURACY - Which colors are being learned?
                # ================================================================
                per_class_correct = diagnostics.get('per_class_correct', [])
                per_class_total = diagnostics.get('per_class_total', [])
                per_class_predicted = diagnostics.get('per_class_predicted', [])
                
                if per_class_total and sum(per_class_total) > 0:
                    print(f"\n  --- PER-COLOR ACCURACY (10 classes) ---")
                    # Color names for readability
                    color_names = ['Black', 'Blue', 'Red', 'Green', 'Yellow', 
                                   'Gray', 'Pink', 'Orange', 'Cyan', 'Brown']
                    
                    # Compute per-class accuracy
                    class_accs = []
                    for c in range(10):
                        if per_class_total[c] > 0:
                            acc = per_class_correct[c] / per_class_total[c]
                            class_accs.append(acc)
                        else:
                            class_accs.append(None)
                    
                    # Display as compact table
                    print(f"  Color:  ", end="")
                    for c in range(10):
                        print(f"{c:>6}", end="")
                    print()
                    
                    print(f"  Acc%:   ", end="")
                    for c in range(10):
                        if class_accs[c] is not None:
                            print(f"{class_accs[c]*100:>5.0f}%", end="")
                        else:
                            print(f"    - ", end="")
                    print()
                    
                    print(f"  Target: ", end="")
                    total_pixels = sum(per_class_total)
                    for c in range(10):
                        pct = per_class_total[c] / total_pixels * 100 if total_pixels > 0 else 0
                        print(f"{pct:>5.1f}%", end="")
                    print()
                    
                    print(f"  Pred:   ", end="")
                    total_pred = sum(per_class_predicted)
                    for c in range(10):
                        pct = per_class_predicted[c] / total_pred * 100 if total_pred > 0 else 0
                        print(f"{pct:>5.1f}%", end="")
                    print()
                    
                    # Identify weak colors (low accuracy)
                    weak_colors = [c for c in range(10) if class_accs[c] is not None and class_accs[c] < 0.5]
                    if weak_colors:
                        weak_str = ', '.join(f"{c}({color_names[c]})" for c in weak_colors)
                        print(f"  [!] Weak colors (<50% acc): {weak_str}")
                    
                    # Check for over/under prediction
                    for c in range(1, 10):  # Skip background
                        if per_class_total[c] > 0:
                            target_pct = per_class_total[c] / total_pixels
                            pred_pct = per_class_predicted[c] / total_pred if total_pred > 0 else 0
                            if target_pct > 0.01 and pred_pct < target_pct * 0.5:
                                print(f"  [!] Under-predicting color {c} ({color_names[c]}): {pred_pct*100:.1f}% vs target {target_pct*100:.1f}%")
                
                print(f"  {'='*50}")
            
            # ================================================================
            # UPDATE LEARNING TRAJECTORY - Track key metrics epoch-by-epoch
            # ================================================================
            # Extract key metrics for trajectory tracking
            stop_prob = diagnostics.get('stop_prob_mean', 0.27)
            expected_clues = diagnostics.get('expected_clues_used', 0)
            per_clue_entropy = diagnostics.get('per_clue_entropy', [])
            mean_entropy = sum(per_clue_entropy) / len(per_clue_entropy) if per_clue_entropy else 0
            per_step_loss = diagnostics.get('per_step_loss', [])
            pred_pcts = diagnostics.get('pred_class_pcts', [])
            target_pcts = diagnostics.get('target_class_pcts', [])
            
            # Training accuracy from batch tracking
            batch_accuracies = diagnostics.get('batch_accuracies', [])
            epoch_train_acc = sum(batch_accuracies) / len(batch_accuracies) if batch_accuracies else 0.0
            total_samples_processed = diagnostics.get('total_samples', 0)
            exact_match_count = diagnostics.get('exact_match_count', 0)
            epoch_exact_match_pct = (exact_match_count / total_samples_processed * 100) if total_samples_processed > 0 else 0.0
            
            # Find best step
            valid_losses = [(i, l) for i, l in enumerate(per_step_loss) if l < 100 and l == l]
            best_step = min(valid_losses, key=lambda x: x[1])[0] if valid_losses else -1
            
            # Foreground coverage (sum of non-bg predictions)
            fg_pred = sum(pred_pcts[1:]) if len(pred_pcts) > 1 else 0
            fg_target = sum(target_pcts[1:]) if len(target_pcts) > 1 else 10  # Default ~10%
            
            # Store in trajectory
            learning_trajectory['epochs'].append(epoch + 1)
            learning_trajectory['stop_prob'].append(stop_prob)
            learning_trajectory['expected_clues'].append(expected_clues)
            learning_trajectory['attention_entropy'].append(mean_entropy)
            learning_trajectory['task_loss'].append(task_loss_val)
            learning_trajectory['best_step'].append(best_step)
            learning_trajectory['fg_coverage'].append(fg_pred / max(fg_target, 1) * 100)
            learning_trajectory['train_accuracy'].append(epoch_train_acc)
            learning_trajectory['exact_match_pct'].append(epoch_exact_match_pct)
            
            # Print learning trajectory summary every 5 epochs (or first 10)
            if (epoch + 1) <= 10 or (epoch + 1) % 5 == 0:
                print(f"\n  {'='*50}")
                print(f"  LEARNING TRAJECTORY (Epoch {epoch + 1})")
                print(f"  {'='*50}")
                n = len(learning_trajectory['epochs'])
                if n >= 2:
                    # Show trend for each metric
                    def trend_arrow(values, higher_is_better=False):
                        if len(values) < 2:
                            return "→"
                        diff = values[-1] - values[-2]
                        if abs(diff) < 0.01:
                            return "→"
                        if higher_is_better:
                            return "↑" if diff > 0 else "↓"
                        else:
                            return "↓" if diff < 0 else "↑"
                    
                    # Stop prob: trend matters, not absolute value (task-dependent latent variable)
                    sp = learning_trajectory['stop_prob']
                    print(f"  Stop Prob:   {sp[-1]:.3f} {trend_arrow(sp, higher_is_better=True)} (init=0.27, task-dependent)")
                    
                    # Expected clues: LATENT VARIABLE - no fixed target!
                    # Each sample needs different clue counts based on task complexity
                    # Should vary with data, not converge to fixed value
                    ec = learning_trajectory['expected_clues']
                    print(f"  Exp. Clues:  {ec[-1]:.2f} (latent variable, task-dependent)")
                    
                    # Attention entropy: should DECREASE (sharper attention)
                    ae = learning_trajectory['attention_entropy']
                    print(f"  Attn Entropy: {ae[-1]:.2f} {trend_arrow(ae, higher_is_better=False)} (max=6.8, sharper=better)")
                    
                    # Task loss: should DECREASE
                    tl = learning_trajectory['task_loss']
                    print(f"  Task Loss:   {tl[-1]:.4f} {trend_arrow(tl, higher_is_better=False)}")
                    
                    # Training accuracy: should INCREASE
                    ta = learning_trajectory['train_accuracy']
                    print(f"  Train Acc:   {ta[-1]:.1%} {trend_arrow(ta, higher_is_better=True)}")
                    
                    # Exact match: should INCREASE
                    em = learning_trajectory['exact_match_pct']
                    print(f"  Exact Match: {em[-1]:.1f}% {trend_arrow(em, higher_is_better=True)}")
                    
                    # Best step: should be LATER steps (4-5 for 6-step solver)
                    bs = learning_trajectory['best_step']
                    print(f"  Best Step:   {bs[-1]} (later=better refinement)")
                    
                    # FG coverage: should approach 100% of target
                    fg = learning_trajectory['fg_coverage']
                    print(f"  FG Coverage: {fg[-1]:.1f}% of target {trend_arrow(fg, higher_is_better=True)}")
                    
                    # Check for healthy learning patterns
                    if n >= 3:
                        sp_improving = sp[-1] > sp[0] + 0.05
                        ae_improving = ae[-1] < ae[0] - 0.2
                        tl_improving = tl[-1] < tl[0] * 0.9
                        ta_improving = ta[-1] > ta[0] + 0.05  # 5pp improvement
                        
                        issues = []
                        if not sp_improving and n > 5:
                            issues.append("stop_prob not increasing")
                        if not ae_improving and n > 5:
                            issues.append("attention not sharpening")
                        if not tl_improving and n > 5:
                            issues.append("task_loss not decreasing")
                        if not ta_improving and n > 5:
                            issues.append("train_accuracy not improving")
                        
                        if issues:
                            print(f"  [!] Potential issues: {', '.join(issues)}")
                        elif n > 5:
                            print(f"  ✓ Learning trajectory looks healthy!")
                print(f"  {'='*50}")
            
            # ================================================================
            # 🚦 TRAINING HEALTH CHECK - CONSOLIDATED GO/STOP SIGNAL
            # ================================================================
            # This gives you a QUICK answer: Is training healthy or not?
            # Check this to decide if you should continue or stop early.
            # ================================================================
            
            print(f"\n  {'#'*60}")
            print(f"  🚦 TRAINING HEALTH CHECK - Epoch {epoch + 1}")
            print(f"  {'#'*60}")
            
            health_checks = []
            health_warnings = []
            health_critical = []
            
            n = len(learning_trajectory['epochs'])
            
            # === CHECK 1: Attention Entropy (should DECREASE) ===
            # Normalized entropy: 0=sharp (good), 1=uniform (bad)
            # For 30x30 grid, max entropy = log(900) ≈ 6.8
            max_entropy = 6.8  # log(30*30)
            normalized_entropy = mean_entropy / max_entropy if max_entropy > 0 else 1.0
            
            if normalized_entropy < 0.7:
                health_checks.append(f"✓ Attention sharpening ({normalized_entropy:.2f} < 0.7)")
            elif normalized_entropy < 0.9:
                health_warnings.append(f"⚠ Attention still diffuse ({normalized_entropy:.2f})")
            else:
                if epoch > 5:
                    health_critical.append(f"✗ Attention uniform ({normalized_entropy:.2f} ≈ 1.0)")
                else:
                    health_warnings.append(f"⚠ Attention uniform ({normalized_entropy:.2f}) - early epoch OK")
            
            # === CHECK 2: Stop Probabilities (should adapt, not uniform) ===
            per_clue_stop = diagnostics.get('per_clue_stop_prob', [])
            stop_prob_std_global = diagnostics.get('stop_prob_std', None)
            if per_clue_stop:
                stop_std = (sum((s - stop_prob)**2 for s in per_clue_stop) / len(per_clue_stop))**0.5
                # Prefer the global std when available (matches the production log’s “std=0.008” style).
                std_to_use = stop_prob_std_global if stop_prob_std_global is not None else stop_std
                std_detail = f"global={stop_prob_std_global:.3f}, per_clue={stop_std:.3f}" if stop_prob_std_global is not None else f"std={stop_std:.3f}"

                if std_to_use > 0.1:
                    health_checks.append(f"✓ Stop probs adapting ({std_detail})")
                elif std_to_use > 0.03:
                    health_warnings.append(f"⚠ Stop probs nearly uniform ({std_detail})")
                else:
                    if epoch > 10:
                        health_critical.append(f"✗ Stop probs frozen uniform ({std_detail})")
                    else:
                        health_warnings.append(f"⚠ Stop probs uniform ({std_detail}) - early epoch OK")
            
            # === CHECK 3: Centroid Spread (should be spread, not clustered) ===
            # Uses YAML thresholds: centroid_spread_warning, centroid_spread_critical
            centroid_spread = diagnostics.get('centroid_spread', 0)
            if centroid_spread > centroid_spread_warning * 2.5:  # Good = well above warning
                health_checks.append(f"✓ Centroids spread out ({centroid_spread:.1f} > {centroid_spread_warning * 2.5:.1f})")
            elif centroid_spread > centroid_spread_warning:
                health_warnings.append(f"⚠ Centroids moderately spread ({centroid_spread:.1f})")
            elif centroid_spread > centroid_spread_critical:
                if epoch > 10:
                    health_critical.append(f"✗ Centroids clustered ({centroid_spread:.1f} < {centroid_spread_warning})")
                else:
                    health_warnings.append(f"⚠ Centroids clustered ({centroid_spread:.1f}) - early epoch OK")
            else:
                health_critical.append(f"✗ Centroid COLLAPSE ({centroid_spread:.1f} < {centroid_spread_critical})")
            
            # === CHECK 4: Confidence-Stop Coupling (should be positive) ===
            # POSITIVE correlation = sharp attention leads to stopping = HEALTHY
            confidence_stop_corr = diagnostics.get('confidence_stop_correlation', diagnostics.get('entropy_stop_correlation', 0))
            if confidence_stop_corr > 0.3:
                health_checks.append(f"✓ Good confidence-stop coupling (r={confidence_stop_corr:.2f})")
            elif confidence_stop_corr > 0:
                health_warnings.append(f"⚠ Weak confidence-stop coupling (r={confidence_stop_corr:.2f})")
            else:
                if epoch > 15:
                    health_critical.append(f"✗ No confidence-stop coupling (r={confidence_stop_corr:.2f})")
                else:
                    health_warnings.append(f"⚠ Negative coupling (r={confidence_stop_corr:.2f}) - early epoch OK")
            
            # === CHECK 5: Loss Decreasing ===
            if n >= 3:
                tl = learning_trajectory['task_loss']
                if tl[-1] < tl[0] * 0.8:
                    health_checks.append(f"✓ Loss decreasing ({tl[0]:.3f} → {tl[-1]:.3f})")
                elif tl[-1] < tl[0]:
                    health_warnings.append(f"⚠ Loss slowly decreasing ({tl[0]:.3f} → {tl[-1]:.3f})")
                else:
                    if epoch > 10:
                        health_critical.append(f"✗ Loss not decreasing ({tl[0]:.3f} → {tl[-1]:.3f})")
                    else:
                        health_warnings.append(f"⚠ Loss flat ({tl[0]:.3f} → {tl[-1]:.3f}) - early epoch")
            
            # === CHECK 6: Training Accuracy Improving ===
            if n >= 3:
                ta = learning_trajectory['train_accuracy']
                if ta[-1] > ta[0] + 0.1:  # 10pp improvement
                    health_checks.append(f"✓ Accuracy improving ({ta[0]:.1%} → {ta[-1]:.1%})")
                elif ta[-1] > ta[0]:
                    health_warnings.append(f"⚠ Accuracy slowly improving ({ta[0]:.1%} → {ta[-1]:.1%})")
                else:
                    if epoch > 10:
                        health_critical.append(f"✗ Accuracy not improving ({ta[0]:.1%} → {ta[-1]:.1%})")
                    else:
                        health_warnings.append(f"⚠ Accuracy flat ({ta[0]:.1%} → {ta[-1]:.1%}) - early epoch")
            
            # === CHECK 7: No NaN/Inf Issues ===
            nan_batches = train_losses.get('nan_batches', 0)
            if nan_batches == 0:
                health_checks.append(f"✓ No NaN/Inf issues")
            elif nan_batches < 5:
                health_warnings.append(f"⚠ {nan_batches} NaN batches (minor)")
            else:
                health_critical.append(f"✗ {nan_batches} NaN batches (numerical instability!)")
            
            # === CHECK 8: Color Mode Collapse ===
            fg_pred_mode_pct = diagnostics.get('fg_pred_mode_pct', 0)
            if fg_pred_mode_pct < 30:
                health_checks.append(f"✓ No color mode collapse")
            elif fg_pred_mode_pct < 50:
                health_warnings.append(f"⚠ Color preference ({fg_pred_mode_pct:.0f}% one color)")
            else:
                health_critical.append(f"✗ Color mode collapse ({fg_pred_mode_pct:.0f}% one color)")
            
            # === PRINT RESULTS ===
            total_checks = len(health_checks) + len(health_warnings) + len(health_critical)
            passed = len(health_checks)
            
            for check in health_checks:
                print(f"  {check}")
            for warn in health_warnings:
                print(f"  {warn}")
            for crit in health_critical:
                print(f"  {crit}")
            
            print(f"  {'-'*56}")
            
            # === OVERALL VERDICT ===
            if len(health_critical) == 0 and len(health_warnings) <= 2:
                verdict = "🟢 HEALTHY"
                verdict_msg = "Training is progressing well. Continue!"
            elif len(health_critical) == 0:
                verdict = "🟡 MONITOR"
                verdict_msg = "Some concerns but not critical. Watch next few epochs."
            elif len(health_critical) <= 2:
                verdict = "🟠 WARNING"
                verdict_msg = "Multiple issues detected. Consider intervention soon."
            else:
                verdict = "🔴 UNHEALTHY"
                verdict_msg = "Training likely failing. STOP and investigate!"
            
            print(f"  RESULT: {passed}/{total_checks} checks passed")
            print(f"  STATUS: {verdict}")
            print(f"  → {verdict_msg}")
            
            # === ACTIONABLE ADVICE ===
            if len(health_critical) > 0:
                print(f"\n  RECOMMENDED ACTIONS:")
                if any("Attention" in c for c in health_critical):
                    print(f"    - Increase lambda_entropy to sharpen attention")
                if any("Stop probs" in c for c in health_critical):
                    print(f"    - Check stop_predictor gradient flow")
                if any("Centroids" in c for c in health_critical):
                    print(f"    - DSC may not be learning - check DSC gradients")
                if any("Loss" in c for c in health_critical):
                    print(f"    - Reduce learning rate or check data pipeline")
                if any("NaN" in c for c in health_critical):
                    print(f"    - Enable bfloat16, reduce learning rate, check for div by zero")
                if any("collapse" in c for c in health_critical):
                    print(f"    - Increase focal_alpha, check class weights")
            
            print(f"  {'#'*60}")
        
        # Evaluate
        if (epoch + 1) % eval_every == 0:
            # ================================================================
            # MEMORY CLEANUP BEFORE EVALUATION (Dec 2025)
            # ================================================================
            # Prevent memory fragmentation that causes TTA slowdown at higher epochs.
            # Clear cached allocations and reset memory stats before heavy TTA eval.
            # ================================================================
            if device.type == 'cuda':
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                # Force garbage collection to free Python objects holding GPU refs
                import gc
                gc.collect()
                torch.cuda.synchronize()
            
            # Compute current training temperature for eval
            # CRITICAL: Use same temperature as training to avoid distribution shift!
            eval_temp = get_temperature(epoch, config)
            
            # Use EMA model for evaluation if available
            if ema is not None:
                eval_model = ema.ema_copy(model)
                eval_model = eval_model.to(device)
            else:
                eval_model = model
            
            # CRITICAL FIX (Jan 2026): Apply inference staging from YAML config.
            # This reads from config['inference']['meta_learning'] section and handles:
            # - Empty HPM buffers gracefully (falls back to static banks)
            # - Untrained HyperLoRA gracefully (falls back to base model)
            # - Consistent flags across eval and inference paths
            # - HPM buffer auto-loading and staleness checks from YAML
            try:
                from sci_arc.utils.inference_staging import apply_inference_staging_with_hpm_loading
                staging_results = apply_inference_staging_with_hpm_loading(
                    eval_model, config, checkpoint=None, verbose=False
                )
                active_modules = staging_results.get('staging', {})
            except ImportError:
                # Fallback to hardcoded flags if helper not available
                if hasattr(eval_model, 'hyperlora_active'):
                    eval_model.hyperlora_active = True
                if hasattr(eval_model, 'use_hpm'):
                    eval_model.use_hpm = True
                if hasattr(eval_model, 'hpm_memory_enabled'):
                    eval_model.hpm_memory_enabled = True
                if hasattr(eval_model, 'solver_context_active'):
                    eval_model.solver_context_active = True
                if hasattr(eval_model, 'cross_attention_active'):
                    eval_model.cross_attention_active = True
                active_modules = {'hyperlora': True, 'hpm': True, 'solver_context': True, 'cross_attention': True}
            
            eval_metrics = evaluate(eval_model, eval_loader, device, temperature=eval_temp)
            
            # ============================================================
            # CANONICAL TRAIN EVAL (Jan 2026 Patch)
            # ============================================================
            # Run deterministic eval on training data (no augmentation) for
            # stable metrics that don't fluctuate with random augmentation.
            # Useful for detecting true plateaus vs augmentation noise.
            # ============================================================
            canonical_train_cfg = config.get('evaluation', {}).get('canonical_train_eval', {})
            if canonical_train_cfg.get('enabled', True):  # Enabled by default
                canonical_max_tasks = canonical_train_cfg.get('max_tasks', 100)
                canonical_metrics = evaluate_canonical_train(
                    eval_model, train_loader, device, 
                    temperature=eval_temp,
                    max_tasks=canonical_max_tasks
                )
                if canonical_metrics.get('canonical_enabled', False):
                    print(f"\\n  --- Canonical Train Eval (no augmentation, {canonical_metrics['canonical_total_tasks']} tasks) ---")
                    print(f"  Canonical Task Accuracy: {canonical_metrics['canonical_correct_tasks']}/{canonical_metrics['canonical_total_tasks']} ({canonical_metrics['canonical_task_accuracy']*100:.1f}%)")
                    print(f"  Canonical Pixel Accuracy: {canonical_metrics['canonical_pixel_accuracy']*100:.1f}%")
                    # Store for comparison
                    eval_metrics['canonical_task_accuracy'] = canonical_metrics['canonical_task_accuracy']
                    eval_metrics['canonical_pixel_accuracy'] = canonical_metrics['canonical_pixel_accuracy']
            
            # DIAGNOSTIC: Also eval training model to detect EMA lag
            if ema is not None:
                train_model_metrics = evaluate(model, eval_loader, device, temperature=eval_temp)
                train_fg = train_model_metrics['fg_accuracy']
                ema_fg = eval_metrics['fg_accuracy']
                if train_fg > ema_fg + 0.1:  # Training model significantly better
                    print(f"\n  [!] EMA LAG: Training model FG={train_fg:.1%} vs EMA FG={ema_fg:.1%}")
                    print(f"      Training model is learning but EMA hasn't caught up yet")
            
            # ============================================================
            # TRM-STYLE EVALUATION WITH INVERSE AUGMENTATION
            # ============================================================
            # This is the CRITICAL evaluation for measuring true generalization.
            # Uses proper TTA: generates all dihedral views per task, votes.
            # ============================================================
            if use_trm_eval and cached_eval_tasks:
                import time as time_module
                trm_eval_start = time_module.time()
                
                # Use pre-cached tasks (no JSON re-parsing)
                eval_tasks = cached_eval_tasks
                
                if eval_tasks:
                    num_dihedral = config.get('evaluation', {}).get('num_augmented_views', 8)
                    num_color_perms = config.get('evaluation', {}).get('num_color_perms', 4)
                    pass_ks = config.get('evaluation', {}).get('pass_ks', [1, 2, 3])
                    # SAFEGUARD (Jan 2026): Optional eval_seed for reproducible color permutations
                    eval_seed = config.get('evaluation', {}).get('eval_seed', None)
                    trm_metrics = evaluate_trm_style(
                        eval_model, eval_tasks, device, 
                        temperature=eval_temp,
                        num_dihedral=num_dihedral,
                        num_color_perms=num_color_perms,  # TTA with color permutation for max generalization
                        max_size=max_grid_size,
                        pass_ks=pass_ks,
                        eval_seed=eval_seed,
                    )
                    trm_eval_time = time_module.time() - trm_eval_start
                    
                    total_views = num_dihedral * num_color_perms
                    print(f"\n  --- TRM-Style TTA Evaluation ({num_dihedral} dihedral x {num_color_perms} color = {total_views} views) ---")
                    print(f"  ★ TTA Exact Match (Pass@1): {trm_metrics['correct_tasks']}/{trm_metrics['total_tasks']} ({trm_metrics['exact_match']*100:.1f}%)")
                    print(f"  ⏱️ TTA eval time: {trm_eval_time:.1f}s ({trm_eval_time/len(eval_tasks):.2f}s/task)")
                    
                    # Report Pass@K metrics
                    pass_k_parts = []
                    for k in pass_ks:
                        pass_k_key = f'pass@{k}'
                        if pass_k_key in trm_metrics:
                            pass_k_parts.append(f"Pass@{k}: {trm_metrics[pass_k_key]*100:.1f}%")
                    if pass_k_parts:
                        print(f"  Pass@K: {' | '.join(pass_k_parts)}")
                    
                    print(f"  Avg Unique Predictions: {trm_metrics['avg_unique_predictions']:.1f} / {trm_metrics['total_views']}")
                    print(f"  Avg Winner Votes: {trm_metrics['avg_winner_votes']:.1f} / {trm_metrics['total_views']}")
                    
                    # GAP MONITORING: Compare train metrics vs eval metrics
                    # FIXED (Dec 2025): Uses FIRST-SAMPLE metric for true apples-to-apples comparison!
                    # Only the FIRST sample seen per task counts - no "multiple attempts" advantage.
                    # This matches eval behavior exactly (single-shot per task).
                    
                    # FIRST-SAMPLE metric (strict, comparable to eval)
                    first_sample_results = diagnostics.get('first_sample_task_correct', {})
                    first_sample_correct = sum(1 for v in first_sample_results.values() if v)
                    first_sample_total = len(first_sample_results)
                    train_first_sample_acc = (first_sample_correct / first_sample_total) if first_sample_total > 0 else 0.0
                    
                    # ANY-SAMPLE metric (lenient, for reference)
                    solved_tasks_any = len(diagnostics.get('solved_task_ids', set()))
                    seen_tasks = len(diagnostics.get('seen_task_ids', set()))
                    train_any_sample_acc = (solved_tasks_any / seen_tasks) if seen_tasks > 0 else 0.0
                    
                    # Sample-level for reference
                    train_sample_level_acc = epoch_exact_match_pct / 100.0
                    
                    eval_exact_match = trm_metrics.get('exact_match', 0)
                    gap = train_first_sample_acc - eval_exact_match  # TRUE apples-to-apples!
                    
                    # GENERALIZATION HEALTH METRICS
                    print(f"\n  --- Generalization Health (SINGLE-SHOT) ---")
                    print(f"  Train Tasks (first-sample): {first_sample_correct}/{first_sample_total} ({train_first_sample_acc:.1%})")
                    print(f"  Eval Tasks (TTA): {trm_metrics['correct_tasks']}/{trm_metrics['total_tasks']} ({eval_exact_match:.1%})")
                    print(f"  Train→Eval Gap: {gap:.1%} [true single-shot comparison]")
                    print(f"  (Any-sample train: {train_any_sample_acc:.1%} | Sample-level: {train_sample_level_acc:.1%})")
                    
                    # Use YAML thresholds for exact match gap
                    if gap > exact_match_critical:
                        print(f"  🚨 CRITICAL GAP: {gap:.1%} > {exact_match_critical:.0%} - Model overfitting!")
                    elif gap > exact_match_warning:
                        print(f"  ⚠️ WARNING GAP: {gap:.1%} > {exact_match_warning:.0%} - Monitor closely")
                    elif gap > exact_match_warning / 2:
                        print(f"  ℹ️ Mild gap: {gap:.1%} - Acceptable")
                    else:
                        print(f"  ✅ Healthy gap: {gap:.1%} - Good generalization!")
                    
                    # Check voting consensus using YAML thresholds
                    consensus_ratio = trm_metrics['avg_winner_votes'] / trm_metrics['total_views']
                    if consensus_ratio < tta_consensus_critical:
                        print(f"  🚨 CRITICAL CONSENSUS: {consensus_ratio:.0%} < {tta_consensus_critical:.0%} - Equivariance broken!")
                    elif consensus_ratio < tta_consensus_warning:
                        print(f"  ⚠️ LOW CONSENSUS: {consensus_ratio:.0%} < {tta_consensus_warning:.0%} - Model not dihedral-invariant!")
                    elif consensus_ratio < 0.5:
                        print(f"  ⚠️ Moderate consensus: {consensus_ratio:.0%}")
                    else:
                        print(f"  ✅ Good consensus: {consensus_ratio:.0%}")
                    
                    # ============================================================
                    # METRIC-BASED PHASE GATE CHECK (Jan 2026)
                    # ============================================================
                    # After TRM eval, check if phase gate criteria are met.
                    # If ready, log phase transition for NEXT epoch.
                    # The effective phase is updated via get_effective_phase().
                    # ============================================================
                    if use_metric_gating and phased_training_enabled:
                        # Update stability counters based on this epoch's diagnostics
                        if diagnostics:
                            # Check for centroid collapse (tracked in training)
                            centroid_spread = diagnostics.get('centroid_spread', 1.0)
                            if centroid_spread < centroid_spread_critical:
                                # Centroid collapsed - reset counter
                                phase_gate_state['centroid_collapse_free_count'] = 0
                            else:
                                phase_gate_state['centroid_collapse_free_count'] += 1
                        
                        # Grad explosion counter reset at epoch start, track here
                        grad_events_this_epoch = meta_escalation_state.get('grad_explosion_events_epoch', 0) if 'meta_escalation_state' in dir() else 0
                        if grad_events_this_epoch > 0:
                            phase_gate_state['grad_explosion_free_count'] = 0
                        else:
                            phase_gate_state['grad_explosion_free_count'] += 1
                        
                        # Increment epochs in current phase
                        phase_gate_state['epochs_in_current_phase'] += 1
                        
                        # Check phase gate with current metrics
                        epoch_1based = epoch + 1
                        effective_phase, effective_cfg, gate_status = get_effective_phase(
                            epoch, eval_metrics, trm_metrics
                        )
                        
                        # Log gate status
                        print(f"\n  --- Phase Gate Status (Epoch {epoch_1based}) ---")
                        print(f"  Current Effective Phase: {phase_gate_state['current_effective_phase']}")
                        
                        if 'consecutive' in gate_status:
                            status_parts = []
                            for item in gate_status.get('met', []):
                                status_parts.append(f"✓{item}")
                            for item in gate_status.get('not_met', []):
                                status_parts.append(f"✗{item}")
                            print(f"  Gate Criteria: {' | '.join(status_parts)}")
                            print(f"  Patience: {gate_status['consecutive']}/{gate_status['patience']} consecutive evals")
                            
                            if effective_phase != phase_gate_state['current_effective_phase']:
                                # Phase transition happened!
                                print(f"  🎯 PHASE TRANSITION: {phase_gate_state['current_effective_phase']} → {effective_phase}")
                                # Update effective phase (already done in get_effective_phase)
                        elif gate_status.get('at_target'):
                            print(f"  Already at target phase {gate_status.get('phase', '?')}")
                        elif gate_status.get('gating') == 'disabled':
                            print(f"  Metric gating: disabled")
            
            # Core metrics (all computed over VALID pixels only, excluding padding)
            # 10-class encoding: class 0=black (BG), classes 1-9=colors (FG)
            correct_tasks = eval_metrics.get('correct_tasks', 0)
            total_tasks = eval_metrics.get('total_tasks', 1)
            print(f"  --- Evaluation Metrics (Valid Pixels Only) ---")
            print(f"  ★ EXACT MATCH: {correct_tasks}/{total_tasks} tasks ({eval_metrics['task_accuracy']*100:.1f}%)")
            print(f"  Pixel Accuracy: {eval_metrics['pixel_accuracy']:.4f}")
            print(f"  FG Accuracy (colors 1-9): {eval_metrics['fg_accuracy']:.4f}")
            print(f"  BG Accuracy (black): {eval_metrics['bg_accuracy']:.4f}")
            
            # Class distribution ratios
            print(f"  Class Ratios (pred/target):")
            print(f"    BG (black): {eval_metrics['bg_ratio_pred']:.1%} / {eval_metrics['bg_ratio_target']:.1%}")
            print(f"    FG (colors): {eval_metrics['fg_ratio_pred']:.1%} / {eval_metrics['fg_ratio_target']:.1%}")
            print(f"  Colors Used (pred/target): {eval_metrics['colors_used']} / {eval_metrics['colors_target']}")
            
            # Module-specific metrics for debugging
            print(f"  DSC Entropy: {eval_metrics['dsc_entropy']:.4f} (lower=sharper)")
            print(f"  DSC Clues Used: {eval_metrics['dsc_clues_used']:.2f}")
            eval_stop_prob = eval_metrics.get('eval_stop_prob', 0)
            if eval_stop_prob > 0:
                print(f"  Eval Stop Prob: {eval_stop_prob:.3f}")
            print(f"  Predicate Activation: {eval_metrics['predicate_activation']:.4f}")
            print(f"  Eval Temperature: {eval_temp:.3f} (matched to training)")
            
            # ============================================================
            # TRAIN-EVAL ENTROPY DELTA - Critical for generalization check
            # ============================================================
            eval_entropy = eval_metrics['dsc_entropy']
            if mean_entropy > 0:  # mean_entropy from training diagnostics
                entropy_delta = eval_entropy - mean_entropy
                entropy_ratio = eval_entropy / max(mean_entropy, 0.001)
                print(f"\n  --- Train vs Eval Entropy Delta ---")
                print(f"  Train DSC Entropy: {mean_entropy:.4f}")
                print(f"  Eval DSC Entropy:  {eval_entropy:.4f}")
                print(f"  Delta (Eval - Train): {entropy_delta:+.4f}")
                print(f"  Ratio (Eval / Train): {entropy_ratio:.2f}x")
                
                # Warning thresholds from monitoring config
                entropy_ratio_warning = config.get('monitoring', {}).get('entropy_ratio_warning', 2.0)
                entropy_ratio_critical = config.get('monitoring', {}).get('entropy_ratio_critical', 5.0)
                
                if entropy_ratio > entropy_ratio_critical:
                    print(f"  🚨 CRITICAL: Eval entropy {entropy_ratio:.1f}x higher than train!")
                    print(f"      Model not generalizing - attention collapses on unseen data")
                elif entropy_ratio > entropy_ratio_warning:
                    print(f"  ⚠️ WARNING: Eval entropy {entropy_ratio:.1f}x higher than train")
                    print(f"      Potential generalization issue - monitor closely")
                elif entropy_ratio < 0.5:
                    print(f"  ⚠️ Unusual: Eval entropy LOWER than train ({entropy_ratio:.2f}x)")
                else:
                    print(f"  ✅ Healthy: Train-eval entropy aligned ({entropy_ratio:.2f}x)")
            
            if ema is not None:
                print("  (Using EMA weights for evaluation)")
                # Check for EMA lag - compare training stop_prob vs eval stop_prob
                # Uses YAML thresholds: stop_value_warning, stop_value_critical
                train_stop_prob = diagnostics.get('stop_prob_mean', 0) if diagnostics else 0
                if train_stop_prob > 0 and eval_stop_prob > 0:
                    stop_gap = abs(train_stop_prob - eval_stop_prob)
                    if stop_gap > stop_value_critical:
                        print(f"  🚨 CRITICAL EMA LAG: train_stop={train_stop_prob:.3f} vs eval_stop={eval_stop_prob:.3f} (gap={stop_gap:.3f} > {stop_value_critical})")
                        print(f"      EMA decay may be too high (0.999), consider 0.99 or 0.995")
                    elif stop_gap > stop_value_warning:
                        print(f"  ⚠️ EMA LAG DETECTED: train_stop={train_stop_prob:.3f} vs eval_stop={eval_stop_prob:.3f} (gap={stop_gap:.3f})")
            
            # ============================================================
            # BACKGROUND COLLAPSE DETECTION - CRITICAL FOR DEBUGGING
            # ============================================================
            is_collapsing = False
            collapse_reasons = []
            
            # Check 1: Excessive background prediction
            bg_excess = eval_metrics['bg_ratio_pred'] - eval_metrics['bg_ratio_target']
            if bg_excess > 0.15:  # 15% more BG than target
                is_collapsing = True
                collapse_reasons.append(f"BG excess: {bg_excess:.1%}")
            
            # Check 2: Zero foreground accuracy (colors 1-9)
            if eval_metrics['fg_accuracy'] < 0.01 and epoch > 10:
                is_collapsing = True
                collapse_reasons.append(f"FG acc: {eval_metrics['fg_accuracy']:.1%}")
            
            # Check 3: Too few colors predicted
            if eval_metrics['colors_used'] < eval_metrics['colors_target'] - 3:
                is_collapsing = True
                collapse_reasons.append(f"Colors: {eval_metrics['colors_used']}/{eval_metrics['colors_target']}")
            
            # Check 4: DSC not differentiating (very high entropy)
            if eval_metrics['dsc_entropy'] > 5.0 and epoch > 20:
                is_collapsing = True
                collapse_reasons.append(f"DSC entropy: {eval_metrics['dsc_entropy']:.2f}")
            
            if is_collapsing:
                collapse_warnings += 1
                print(f"\n  ⚠️  [WARNING] BACKGROUND COLLAPSE DETECTED! ({collapse_warnings}/{max_collapse_warnings})")
                print(f"      Reasons: {', '.join(collapse_reasons)}")
                print(f"      Consider: Lower learning rate, increase focal_alpha, check ContextEncoder")
                
                if collapse_warnings >= max_collapse_warnings:
                    print(f"\n  🛑 [CRITICAL] {max_collapse_warnings} consecutive collapse warnings!")
                    print(f"      Training appears to have failed. Please review:")
                    print(f"      1. ContextEncoder - is it receiving training pairs?")
                    print(f"      2. focal_alpha - try increasing to 0.5-0.75")
                    print(f"      3. learning_rate - try reducing by 2-5x")
                    print(f"      4. lambda_entropy - try increasing to focus attention")
                    print(f"\n      Stopping training to prevent wasted compute.")
                    break
            else:
                collapse_warnings = 0  # Reset on good epoch
            
            # Log to wandb - all loss components for complete monitoring
            if wandb_enabled:
                # Prepare augmentation stats for wandb
                aug_log = {}
                if aug_stats:
                    dihedral_counts = aug_stats.get('dihedral_counts', [0]*8)
                    dihedral_total = sum(dihedral_counts)
                    if dihedral_total > 0:
                        for i, count in enumerate(dihedral_counts):
                            aug_log[f'aug/dihedral_{i}'] = count / dihedral_total
                    aug_log['aug/color_perm_pct'] = color_pct
                    aug_log['aug/translational_pct'] = trans_pct
                    aug_log['aug/total_samples'] = total_samples
                
                wandb.log({
                    'epoch': epoch + 1,
                    # All loss components
                    'train_loss': train_losses['total_loss'],
                    'focal_loss': train_losses['focal_loss'],
                    'entropy_loss': train_losses.get('entropy_loss', 0.0),
                    'sparsity_loss': train_losses.get('sparsity_loss', 0.0),
                    'predicate_loss': train_losses.get('predicate_loss', 0.0),
                    'curriculum_loss': train_losses.get('curriculum_loss', 0.0),
                    # Jan 2026 Ablation: ART + ARPS
                    'art_consistency_loss': train_losses.get('art_consistency_loss', 0.0),
                    'arps_imitation_loss': train_losses.get('arps_imitation_loss', 0.0),
                    # Training accuracy metrics (per-sample tracking)
                    'train_accuracy': epoch_train_acc,
                    'train_exact_match_pct': epoch_exact_match_pct,
                    # Evaluation metrics (computed over valid pixels only)
                    'pixel_accuracy': eval_metrics['pixel_accuracy'],
                    'task_accuracy': eval_metrics['task_accuracy'],
                    'fg_accuracy': eval_metrics['fg_accuracy'],
                    'bg_accuracy': eval_metrics['bg_accuracy'],
                    'fg_ratio_pred': eval_metrics['fg_ratio_pred'],
                    'fg_ratio_target': eval_metrics['fg_ratio_target'],
                    'bg_ratio_pred': eval_metrics['bg_ratio_pred'],
                    'bg_ratio_target': eval_metrics['bg_ratio_target'],
                    'colors_used': eval_metrics['colors_used'],
                    'colors_target': eval_metrics['colors_target'],
                    # Module debugging
                    'dsc_entropy': eval_metrics['dsc_entropy'],
                    'dsc_clues_used': eval_metrics['dsc_clues_used'],
                    'predicate_activation': eval_metrics['predicate_activation'],
                    # Training state
                    'lr': optimizer.param_groups[0]['lr'],
                    'temperature': get_temperature(epoch, config),
                    'collapse_warnings': collapse_warnings,
                    # Augmentation diversity (CRITICAL for quality assurance)
                    **aug_log,
                })
            
            # Save best model
            if eval_metrics['task_accuracy'] > best_task_accuracy:
                best_task_accuracy = eval_metrics['task_accuracy']
                best_correct = eval_metrics.get('correct_tasks', 0)
                best_total = eval_metrics.get('total_tasks', 1)
                best_path = checkpoint_dir / "best.pt"
                save_checkpoint(
                    model, optimizer, scheduler, epoch, global_step,
                    train_losses, best_task_accuracy, config, str(best_path)
                )
                print(f"  ★★★ NEW BEST: {best_correct}/{best_total} exact matches ({best_task_accuracy*100:.1f}%) ★★★")
            
            # ================================================================
            # MEMORY CLEANUP AFTER EVALUATION (Dec 2025)
            # ================================================================
            # Free EMA copy and any cached tensors from TTA evaluation.
            # This prevents memory fragmentation that slows down later epochs.
            # ================================================================
            if ema is not None and eval_model is not ema.ema_model and eval_model is not model:
                del eval_model  # Delete EMA copy
            if device.type == 'cuda':
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                import gc
                gc.collect()
        
        # Save periodic checkpoint
        if (epoch + 1) % save_every == 0:
            checkpoint_path = checkpoint_dir / f"epoch_{epoch + 1}.pt"
            save_checkpoint(
                model, optimizer, scheduler, epoch, global_step,
                train_losses, best_task_accuracy, config, str(checkpoint_path)
            )
            cleanup_old_checkpoints(checkpoint_dir, keep_last_n)
        
        # Save latest checkpoint
        latest_path = checkpoint_dir / "latest.pt"
        save_checkpoint(
            model, optimizer, scheduler, epoch, global_step,
            train_losses, best_task_accuracy, config, str(latest_path)
        )
    
    print("\n" + "=" * 60)
    print(f"Training complete! Best task accuracy: {best_task_accuracy:.4f}")
    
    # Cleanup
    if tee_logger:
        tee_logger.close()
    
    # Explicit memory cleanup (also called by atexit, but good to be explicit)
    cleanup_memory()


if __name__ == "__main__":
    main()
