"""
Program-Guided Training: Joint Training of RLAN with Primitive Prediction

This module enables end-to-end training of RLAN with NS-TEPS integration:

1. PSEUDO-LABEL GENERATION:
   - Run NS-TEPS on training pairs to discover programs
   - Cache successful program traces as training targets
   - Use these to supervise the PrimitiveHead

2. JOINT LOSS:
   - Pixel prediction loss (standard RLAN)
   - Primitive prediction loss (new, from PrimitiveHead)
   - Consistency loss (primitive predictions match discovered programs)

3. CURRICULUM:
   - Start with pixel-only training
   - Gradually introduce primitive prediction as NS-TEPS finds more programs
   - Full joint training once sufficient pseudo-labels exist

WHY THIS ENABLES GENERALIZATION:
- RLAN learns FEATURES that are useful for program discovery
- Features that predict correct primitives are also features that generalize
- This is a form of "learning to search" - the network learns what to look for

MODULAR DESIGN (Non-negotiable):
- Wraps existing RLAN model without modification
- Can be disabled completely via config
- All new components are in separate files

Author: AI Research Assistant  
Date: January 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import numpy as np
from pathlib import Path
import json
import time
from collections import defaultdict

from .primitive_head import (
    PrimitiveHead, 
    PrimitiveHeadConfig, 
    PrimitiveHeadLoss,
    PRIMITIVE_NAME_TO_ID,
    PRIMITIVE_TYPE_MAPPING,
)
from .ns_teps import NSTEPS, NSTEPSConfig, ProgramTrace


@dataclass
class ProgramGuidedConfig:
    """Configuration for program-guided training."""
    enabled: bool = True
    
    # Primitive Head config
    primitive_head: PrimitiveHeadConfig = field(default_factory=PrimitiveHeadConfig)
    
    # Loss weights (relative to main pixel loss)
    primitive_loss_weight: float = 0.3  # Weight of primitive prediction loss
    consistency_loss_weight: float = 0.1  # Weight of program consistency loss
    
    # Pseudo-label generation
    use_cached_programs: bool = True  # Use pre-computed program cache
    cache_path: str = "program_cache.json"  # Path to program cache
    online_mining: bool = True  # Mine programs during training
    mining_frequency: int = 10  # Mine every N batches
    min_confidence: float = 0.9  # Min match score to accept program
    
    # Curriculum
    warmup_epochs: int = 2  # Epochs before enabling primitive loss
    curriculum_epochs: int = 5  # Epochs to gradually increase primitive weight
    
    # NS-TEPS config for mining
    nsteps_config: NSTEPSConfig = field(default_factory=lambda: NSTEPSConfig(
        max_search_steps=500,  # Faster for online mining
        timeout_seconds=2.0,
        sample_count=200,
    ))


class ProgramCache:
    """
    Cache for discovered programs.
    
    Stores task_id -> (program_trace, confidence_score) mappings.
    Used to avoid re-mining programs every epoch.
    """
    
    def __init__(self, cache_path: Optional[str] = None):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self._by_input_hash: Dict[str, Dict[str, Any]] = {}
        self.cache_path = Path(cache_path) if cache_path else None
        
        if self.cache_path and self.cache_path.exists():
            self.load()
    
    def add(
        self, 
        task_id: str, 
        trace: Union[List[Tuple[str, Dict]], 'ProgramTrace'], 
        confidence: float,
        input_hash: str
    ):
        """Add a discovered program to cache.
        
        Args:
            task_id: Unique identifier for the task
            trace: Either a ProgramTrace object or List[Tuple[str, Dict]]
            confidence: Match confidence score
            input_hash: Hash of input for staleness check
        """
        # Convert ProgramTrace to serializable format
        if hasattr(trace, 'steps'):
            # It's a ProgramTrace object - extract (primitive, params) tuples
            serialized_trace = [
                (step[0].name if hasattr(step[0], 'name') else str(step[0]), step[1])
                for step in trace.steps
            ]
        else:
            # Already in list format
            serialized_trace = [(name, params) for name, params in trace]
        
        self.cache[task_id] = {
            'trace': serialized_trace,
            'confidence': confidence,
            'input_hash': input_hash,
            'timestamp': time.time(),
        }

        # Secondary index for augmented variants / alternate IDs.
        # This lets training fetch pseudo-labels by the (augmented) input hash.
        if input_hash:
            self._by_input_hash[input_hash] = self.cache[task_id]
    
    def get(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get cached program for task."""
        return self.cache.get(task_id)

    def get_by_input_hash(self, input_hash: str) -> Optional[Dict[str, Any]]:
        """Get cached program by input hash (useful for augmented variants)."""
        return self._by_input_hash.get(input_hash)
    
    def has(self, task_id: str) -> bool:
        """Check if task has cached program."""
        return task_id in self.cache
    
    def save(self):
        """Save cache to disk."""
        if self.cache_path:
            try:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.cache_path, 'w', encoding='utf-8') as f:
                    json.dump(self.cache, f, indent=2)
            except Exception as e:
                print(f"[ProgramCache] Warning: Failed to save cache to {self.cache_path}: {e}")
    
    def load(self):
        """Load cache from disk."""
        if self.cache_path and self.cache_path.exists():
            try:
                with open(self.cache_path, 'r', encoding='utf-8') as f:
                    self.cache = json.load(f)

                # Rebuild secondary index.
                self._by_input_hash = {}
                for entry in self.cache.values():
                    ih = entry.get('input_hash')
                    if ih:
                        self._by_input_hash[ih] = entry
                
                print(f"[ProgramCache] Loaded {len(self.cache)} programs from {self.cache_path}")
                if len(self._by_input_hash) > 0:
                    print(f"[ProgramCache] Built input_hash index with {len(self._by_input_hash)} entries")
            except json.JSONDecodeError as e:
                print(f"[ProgramCache] Warning: Cache file corrupted, starting fresh: {e}")
                self.cache = {}
                self._by_input_hash = {}
            except Exception as e:
                print(f"[ProgramCache] Warning: Failed to load cache: {e}")
                self.cache = {}
                self._by_input_hash = {}
        elif self.cache_path:
            print(f"[ProgramCache] Cache file not found: {self.cache_path}")
            print(f"[ProgramCache] Primitive loss will be inactive until cache is built or online_mining is enabled")
    
    def __len__(self):
        return len(self.cache)


class PseudoLabelGenerator:
    """
    Generates training targets from NS-TEPS discovered programs.
    
    Converts program traces to:
    1. Primitive IDs (classification targets)
    2. Object selection masks
    3. Parameter targets
    """
    
    def __init__(self, config: ProgramGuidedConfig, device: str = 'cpu'):
        self.config = config
        self.device = device
        
        # NS-TEPS for online mining
        if config.online_mining:
            self.nsteps = NSTEPS(config.nsteps_config)
        else:
            self.nsteps = None
        
        # Program cache
        self.cache = ProgramCache(
            config.cache_path if config.use_cached_programs else None
        )

    def get_cached_targets(
        self,
        task_id: str,
        input_hash: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Return pseudo-label targets from cache, if available.

        Works even when online mining is disabled.
        """
        if self.cache.has(task_id):
            cached = self.cache.get(task_id)
            return self._trace_to_targets(cached['trace'])

        if input_hash:
            cached = self.cache.get_by_input_hash(input_hash)
            if cached is not None:
                return self._trace_to_targets(cached['trace'])

        return None
    
    def mine_program(
        self,
        train_inputs: List[np.ndarray],
        train_outputs: List[np.ndarray],
        task_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Mine program from training pairs using NS-TEPS.
        
        Args:
            train_inputs: List of input grids
            train_outputs: List of output grids
            task_id: Task identifier for caching
            
        Returns:
            Dict with program trace and targets, or None if no program found
        """
        # Always try cache first (even when online mining is disabled).
        input_hash = None
        try:
            import hashlib
            input_hash = hashlib.sha256(train_inputs[0].tobytes()).hexdigest()[:16]
        except Exception:
            input_hash = None

        cached_targets = self.get_cached_targets(task_id=task_id, input_hash=input_hash)
        if cached_targets is not None:
            return cached_targets

        # If online mining is disabled, we can't discover new programs.
        if self.nsteps is None:
            return None
        
        # Run NS-TEPS search
        try:
            # Use first input as test for consistency check
            result = self.nsteps.search(
                test_input=train_inputs[0],
                train_inputs=train_inputs,
                train_outputs=train_outputs,
            )
            
            # NS-TEPS returns: {success, prediction, trace, confidence, steps_searched}
            # trace is a ProgramTrace object with .steps = [(ObjectPrimitive, params), ...]
            if result.get('success', False) and result.get('trace') is not None:
                trace = result['trace']  # ProgramTrace object
                confidence = result.get('confidence', 1.0)
                
                if confidence >= self.config.min_confidence:
                    # Cache the program (ProgramCache handles ProgramTrace conversion)
                    # Use stable hashlib hash instead of Python's non-deterministic hash()
                    import hashlib
                    grid_bytes = train_inputs[0].tobytes()
                    input_hash = hashlib.sha256(grid_bytes).hexdigest()[:16]
                    self.cache.add(task_id, trace, confidence, input_hash)
                    
                    # Convert ProgramTrace to (name, params) list for _trace_to_targets
                    trace_list = [
                        (step[0].name if hasattr(step[0], 'name') else str(step[0]), step[1])
                        for step in trace.steps
                    ]
                    return self._trace_to_targets(trace_list)
        except Exception as e:
            print(f"[PseudoLabelGenerator] Mining failed for {task_id}: {e}")
        
        return None
    
    def _trace_to_targets(
        self, 
        trace: List[Tuple[str, Dict]]
    ) -> Dict[str, torch.Tensor]:
        """
        Convert program trace to training targets.
        
        Args:
            trace: List of (primitive_name, params) tuples
            
        Returns:
            Dict with primitive_ids, object_mask, param tensors
        """
        # Extract primitive sequence
        primitive_ids = []
        param_discrete = []
        param_continuous = []
        
        for step_idx, (prim_name, params) in enumerate(trace):
            # Get primitive ID
            prim_id = PRIMITIVE_NAME_TO_ID.get(prim_name, 0)
            primitive_ids.append(prim_id)
            
            # Extract parameters (simplified - discrete values)
            step_params = []
            step_cont = []
            for key, value in sorted(params.items()):
                if isinstance(value, int):
                    step_params.append(min(value, 31))  # Cap at vocab size
                    step_cont.append(float(value))
                elif isinstance(value, float):
                    step_params.append(int(value * 10) % 32)
                    step_cont.append(value)
                else:
                    step_params.append(0)
                    step_cont.append(0.0)
            
            # Pad to num_params
            while len(step_params) < 8:
                step_params.append(0)
                step_cont.append(0.0)
            
            param_discrete.append(step_params[:8])
            param_continuous.append(step_cont[:8])
        
        # Pad trace to max length
        while len(primitive_ids) < 3:
            primitive_ids.append(0)
            param_discrete.append([0] * 8)
            param_continuous.append([0.0] * 8)
        
        return {
            'primitive_ids': torch.tensor(primitive_ids[:3], dtype=torch.long),
            'param_discrete': torch.tensor(param_discrete[:3], dtype=torch.long),
            'param_continuous': torch.tensor(param_continuous[:3], dtype=torch.float),
            'trace_length': min(len(trace), 3),
            'has_program': True,
        }
    
    def get_batch_targets(
        self,
        batch_task_ids: List[str],
        batch_train_inputs: List[List[np.ndarray]],
        batch_train_outputs: List[List[np.ndarray]],
    ) -> Dict[str, torch.Tensor]:
        """
        Get pseudo-label targets for a batch.
        
        Args:
            batch_task_ids: List of task IDs
            batch_train_inputs: List of (list of input grids) per task
            batch_train_outputs: List of (list of output grids) per task
            
        Returns:
            Batched target tensors
        """
        batch_size = len(batch_task_ids)
        
        # Initialize batch tensors
        primitive_ids = torch.zeros(batch_size, 3, dtype=torch.long)
        param_discrete = torch.zeros(batch_size, 3, 8, dtype=torch.long)
        param_continuous = torch.zeros(batch_size, 3, 8, dtype=torch.float)
        has_program = torch.zeros(batch_size, dtype=torch.bool)
        
        for i, (task_id, inputs, outputs) in enumerate(
            zip(batch_task_ids, batch_train_inputs, batch_train_outputs)
        ):
            targets = self.mine_program(inputs, outputs, task_id)
            
            if targets is not None:
                primitive_ids[i] = targets['primitive_ids']
                param_discrete[i] = targets['param_discrete']
                param_continuous[i] = targets['param_continuous']
                has_program[i] = True
        
        return {
            'primitive_ids': primitive_ids.to(self.device),
            'param_discrete': param_discrete.to(self.device),
            'param_continuous': param_continuous.to(self.device),
            'has_program': has_program.to(self.device),
        }


class ProgramGuidedRLAN(nn.Module):
    """
    RLAN with Program-Guided Training.
    
    This wrapper adds PrimitiveHead to RLAN and enables joint training.
    The base RLAN module is NOT modified - all additions are modular.
    
    TRAINING FLOW:
    1. Forward pass through base RLAN
    2. Extract solver features
    3. Forward through PrimitiveHead
    4. Compute combined loss (pixel + primitive)
    
    INFERENCE FLOW:
    1. Forward through base RLAN  
    2. Get primitive prior from PrimitiveHead
    3. Run NS-TEPS with neural guidance
    4. Return best prediction
    """
    
    def __init__(
        self,
        base_rlan: nn.Module,
        config: ProgramGuidedConfig,
    ):
        super().__init__()
        self.config = config
        
        # Base RLAN (frozen architecture, trainable weights)
        self.base_rlan = base_rlan
        
        # Get hidden dim from base RLAN
        if hasattr(base_rlan, 'config'):
            hidden_dim = base_rlan.config.hidden_dim
        elif hasattr(base_rlan, 'hidden_dim'):
            hidden_dim = base_rlan.hidden_dim
        else:
            hidden_dim = 256  # Default
        
        # Update PrimitiveHead config with correct hidden_dim
        self.config.primitive_head.hidden_dim = hidden_dim
        
        # Primitive Head (new trainable module)
        if config.enabled:
            self.primitive_head = PrimitiveHead(config.primitive_head)
            self.primitive_head.primitive_embed.init_primitive_types(PRIMITIVE_TYPE_MAPPING)
        else:
            self.primitive_head = None
        
        # Loss function for primitive prediction
        self.primitive_loss_fn = PrimitiveHeadLoss(config.primitive_head)
        
        # Pseudo-label generator
        self.label_generator = PseudoLabelGenerator(config)
        
        # Training state
        self.current_epoch = 0
        self.batch_count = 0
    
    def __getattr__(self, name: str):
        """
        Delegate attribute access to base_rlan for any attributes not found on this wrapper.
        
        This allows transparent access to base RLAN attributes like:
        - use_hpm, use_loo, use_hasr (training flags)
        - dsc, encoder, solver, hpm, etc. (modules)
        - hidden_dim, config, etc. (config values)
        """
        # Avoid infinite recursion: _modules, base_rlan must exist
        if name in ('_modules', 'base_rlan', '_parameters', '_buffers'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        
        # Try to get from base_rlan
        try:
            base_rlan = self._modules.get('base_rlan')
            if base_rlan is not None and hasattr(base_rlan, name):
                return getattr(base_rlan, name)
        except (KeyError, AttributeError):
            pass
        
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    
    def forward(
        self,
        test_input: torch.Tensor,
        train_inputs: Optional[torch.Tensor] = None,
        train_outputs: Optional[torch.Tensor] = None,
        pair_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        return_intermediates: bool = False,
        return_primitive_outputs: bool = False,
        **kwargs,  # Forward any additional args to base RLAN
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass with optional primitive prediction.
        
        Args:
            test_input: (B, H, W) test input grid (integer colors)
            train_inputs: Optional (B, N, H, W) training input grids
            train_outputs: Optional (B, N, H, W) training output grids
            pair_mask: Optional (B, N) mask for valid pairs
            temperature: Temperature for softmax
            return_intermediates: Return RLAN intermediate outputs
            return_primitive_outputs: Return PrimitiveHead outputs
            **kwargs: Additional arguments passed to base RLAN
            
        Returns:
            If return_primitive_outputs:
                Dict with logits, intermediates, and primitive outputs
            Else:
                Logits tensor or intermediates dict
        """
        # Forward through base RLAN with all arguments
        rlan_output = self.base_rlan(
            test_input,
            train_inputs=train_inputs,
            train_outputs=train_outputs,
            pair_mask=pair_mask,
            temperature=temperature,
            return_intermediates=True,  # Always get intermediates for primitive head
            **kwargs,
        )
        
        # Extract solver features for primitive head
        # RLAN returns 'features' (encoder output) not 'solver_features'
        if isinstance(rlan_output, dict):
            logits = rlan_output.get('logits', rlan_output.get('output'))
            # Use 'features' from RLAN output (encoder output with shape B, D, H, W)
            solver_features = rlan_output.get('features', rlan_output.get('solver_features'))
        else:
            logits = rlan_output
            solver_features = None
        
        # Get primitive outputs if enabled and features available
        primitive_outputs = None
        if self.primitive_head is not None and solver_features is not None:
            primitive_outputs = self.primitive_head(
                solver_features,
                return_trace=True
            )
        elif self.primitive_head is not None:
            # Fallback: use encoded features
            if hasattr(self.base_rlan, 'encoder'):
                B = test_input.shape[0]
                # Get features from encoder
                enc_features = self.base_rlan.encoder(test_input)
                primitive_outputs = self.primitive_head(
                    enc_features,
                    return_trace=True
                )
        
        # Return format
        if return_primitive_outputs:
            return {
                'logits': logits,
                'intermediates': rlan_output if isinstance(rlan_output, dict) else None,
                'primitive_outputs': primitive_outputs,
            }
        elif return_intermediates:
            if isinstance(rlan_output, dict):
                rlan_output['primitive_outputs'] = primitive_outputs
                return rlan_output
            return {'logits': logits, 'primitive_outputs': primitive_outputs}
        else:
            return logits
    
    def compute_loss(
        self,
        outputs: Dict[str, Any],
        targets: torch.Tensor,
        primitive_targets: Optional[Dict[str, torch.Tensor]] = None,
        pixel_loss_fn: Optional[nn.Module] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss for joint training.
        
        Args:
            outputs: Output from forward() with return_primitive_outputs=True
            targets: (B, H, W) pixel-level targets
            primitive_targets: Optional targets from PseudoLabelGenerator
            pixel_loss_fn: Loss function for pixel prediction
            
        Returns:
            Dict with loss components and total loss
        """
        logits = outputs['logits']
        primitive_outputs = outputs.get('primitive_outputs')
        
        # Pixel loss
        if pixel_loss_fn is not None:
            pixel_loss = pixel_loss_fn(logits, targets)
            if isinstance(pixel_loss, dict):
                pixel_loss = pixel_loss.get('total', pixel_loss.get('loss'))
        else:
            # Logits: (B, C, H, W) -> (B*H*W, C), targets: (B, H, W) -> (B*H*W)
            B, C, H, W = logits.shape
            logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)
            targets_flat = targets.reshape(-1)  # (B*H*W)
            pixel_loss = F.cross_entropy(logits_flat, targets_flat)
        
        losses = {'pixel_loss': pixel_loss}
        total_loss = pixel_loss
        
        # Primitive loss (if enabled and targets available)
        if (
            primitive_outputs is not None and 
            primitive_targets is not None and
            self.config.enabled
        ):
            # Apply curriculum: gradually increase primitive loss weight
            weight = self._get_primitive_weight()
            
            if weight > 0 and primitive_targets.get('has_program', torch.tensor(False)).any():
                # Prepare targets (use first step of trace)
                prim_targets = {
                    'primitive_ids': primitive_targets['primitive_ids'][:, 0],  # First step
                    'param_discrete': primitive_targets['param_discrete'][:, 0],
                    'param_continuous': primitive_targets['param_continuous'][:, 0],
                }
                mask = primitive_targets['has_program'].float()
                
                prim_loss_dict = self.primitive_loss_fn(
                    primitive_outputs, prim_targets, mask
                )
                
                primitive_loss = prim_loss_dict['total_loss'] * weight
                losses['primitive_loss'] = primitive_loss
                losses['primitive_loss_unweighted'] = prim_loss_dict['total_loss']
                total_loss = total_loss + primitive_loss
        
        losses['total_loss'] = total_loss
        return losses

    def compute_primitive_loss(
        self,
        outputs: Dict[str, Any],
        primitive_targets: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute ONLY the primitive loss (no pixel loss computation).

        This is the preferred API for production training to avoid the extra
        pixel CE compute inside compute_loss().
        """
        primitive_outputs = outputs.get('primitive_outputs')

        if (
            primitive_outputs is None or
            primitive_targets is None or
            not self.config.enabled
        ):
            return {'primitive_loss': torch.tensor(0.0, device=outputs['logits'].device)}

        weight = self._get_primitive_weight()
        if weight <= 0:
            return {
                'primitive_loss': torch.tensor(0.0, device=outputs['logits'].device),
                'primitive_weight': torch.tensor(weight, device=outputs['logits'].device),
            }

        has_program = primitive_targets.get('has_program', torch.tensor(False, device=outputs['logits'].device))
        if not has_program.any():
            return {
                'primitive_loss': torch.tensor(0.0, device=outputs['logits'].device),
                'primitive_weight': torch.tensor(weight, device=outputs['logits'].device),
            }

        prim_targets = {
            'primitive_ids': primitive_targets['primitive_ids'][:, 0],
            'param_discrete': primitive_targets['param_discrete'][:, 0],
            'param_continuous': primitive_targets['param_continuous'][:, 0],
        }
        mask = has_program.float()

        prim_loss_dict = self.primitive_loss_fn(primitive_outputs, prim_targets, mask)
        primitive_loss = prim_loss_dict['total_loss'] * weight

        return {
            'primitive_loss': primitive_loss,
            'primitive_loss_unweighted': prim_loss_dict['total_loss'],
            'primitive_weight': torch.tensor(weight, device=primitive_loss.device),
        }
    
    def _get_primitive_weight(self) -> float:
        """Get current primitive loss weight based on curriculum."""
        if self.current_epoch < self.config.warmup_epochs:
            return 0.0
        
        if self.current_epoch < self.config.warmup_epochs + self.config.curriculum_epochs:
            # Linear ramp-up
            progress = (self.current_epoch - self.config.warmup_epochs) / self.config.curriculum_epochs
            return progress * self.config.primitive_loss_weight
        
        return self.config.primitive_loss_weight
    
    def set_epoch(self, epoch: int):
        """Update current epoch for curriculum."""
        self.current_epoch = epoch
    
    def get_primitive_prior(
        self, 
        test_input: torch.Tensor,
        train_inputs: Optional[torch.Tensor] = None,
        train_outputs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get primitive prior for NS-TEPS search guidance.
        
        Args:
            test_input: Test input grid
            train_inputs: Training input grids
            train_outputs: Training output grids
            
        Returns:
            top_primitive_ids: (B, K) top-k primitive indices
            top_primitive_probs: (B, K) probabilities
        """
        if self.primitive_head is None:
            # Return uniform prior if disabled
            B = test_input.shape[0]
            K = 5
            ids = torch.arange(K).unsqueeze(0).expand(B, -1)
            probs = torch.ones(B, K) / K
            return ids, probs
        
        # Get features through RLAN encoder
        with torch.no_grad():
            outputs = self.forward(
                test_input,
                train_inputs=train_inputs,
                train_outputs=train_outputs,
                return_primitive_outputs=True
            )
        
        prim_outputs = outputs['primitive_outputs']
        if prim_outputs is None:
            B = test_input.shape[0]
            K = 5
            ids = torch.arange(K).unsqueeze(0).expand(B, -1)
            probs = torch.ones(B, K) / K
            return ids, probs
        
        # Get top-k from primitive logits
        logits = prim_outputs['primitive_logits']
        probs = F.softmax(logits / self.config.primitive_head.temperature, dim=-1)
        top_probs, top_ids = torch.topk(probs, k=5, dim=-1)
        
        return top_ids, top_probs
    
    def save_program_cache(self):
        """Save mined programs to disk."""
        self.label_generator.cache.save()
    
    def load_program_cache(self, path: str):
        """Load programs from cache file."""
        self.label_generator.cache = ProgramCache(path)
    
    def count_parameters(self) -> Dict[str, int]:
        """Count parameters, delegating to base RLAN and adding PrimitiveHead.
        
        Returns:
            Dict with parameter counts by module
        """
        # Get base RLAN counts
        if hasattr(self.base_rlan, 'count_parameters'):
            counts = self.base_rlan.count_parameters()
        else:
            # Fallback: compute manually
            counts = {
                'total': sum(p.numel() for p in self.base_rlan.parameters()),
                'trainable': sum(p.numel() for p in self.base_rlan.parameters() if p.requires_grad),
            }
        
        # Add PrimitiveHead counts
        if self.primitive_head is not None:
            prim_total = sum(p.numel() for p in self.primitive_head.parameters())
            prim_trainable = sum(p.numel() for p in self.primitive_head.parameters() if p.requires_grad)
            counts['primitive_head'] = prim_total
            counts['primitive_head_trainable'] = prim_trainable
            counts['total'] = counts.get('total', 0) + prim_total
            counts['trainable'] = counts.get('trainable', 0) + prim_trainable
        
        return counts


def create_program_guided_rlan(
    base_rlan: nn.Module,
    config: Optional[ProgramGuidedConfig] = None,
) -> ProgramGuidedRLAN:
    """
    Factory function to create ProgramGuidedRLAN.
    
    Args:
        base_rlan: Existing RLAN model
        config: Optional config (uses defaults if None)
        
    Returns:
        ProgramGuidedRLAN wrapper
    """
    if config is None:
        config = ProgramGuidedConfig()
    
    return ProgramGuidedRLAN(base_rlan, config)
