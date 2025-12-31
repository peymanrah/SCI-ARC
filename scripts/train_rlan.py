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
)
from sci_arc.data import ARCDataset, collate_sci_arc, BucketedBatchSampler
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

def get_tensor_memory_breakdown(model, outputs: dict = None, batch: dict = None) -> Dict[str, float]:
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
    
    # Optimizer states estimate (AdamW stores m and v = 2x param size in fp32)
    # This is an estimate since we don't have optimizer reference here
    try:
        first_param = next(model.parameters())
        breakdown['optimizer_states_estimate_mb'] = (param_bytes * 2 * 4 / first_param.element_size()) / 1024 / 1024
    except StopIteration:
        breakdown['optimizer_states_estimate_mb'] = 0
    
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
    
    # Track which optional modules are ACTIVE (critical for debugging staged activation)
    active_modules = {}
    active_modules['hyperlora_active'] = getattr(model, 'hyperlora_active', False)
    active_modules['solver_context_active'] = getattr(model, 'solver_context_active', False)
    active_modules['cross_attention_active'] = getattr(model, 'cross_attention_active', False)
    active_modules['use_hpm'] = getattr(model, 'use_hpm', False)
    active_modules['use_loo'] = getattr(model, 'loo_enabled', False)
    active_modules['use_equivariance'] = getattr(model, 'equivariance_enabled', False)
    breakdown['active_modules'] = active_modules
    
    return breakdown


def format_memory_breakdown(breakdown: Dict[str, Any], prefix: str = "      ") -> str:
    """Format memory breakdown dict as readable string."""
    lines = []
    
    # Core metrics
    lines.append(f"{prefix}Model params: {breakdown.get('model_params_mb', 0):.1f}MB")
    lines.append(f"{prefix}Model grads:  {breakdown.get('model_grads_mb', 0):.1f}MB")
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
    
    # Active modules status (CRITICAL for debugging staged activation)
    if 'active_modules' in breakdown:
        active = breakdown['active_modules']
        active_list = [k for k, v in active.items() if v]
        inactive_list = [k for k, v in active.items() if not v]
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
        
    def checkpoint(self, name: str, model=None, outputs: dict = None, batch: dict = None):
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
                breakdown = get_tensor_memory_breakdown(model, outputs, batch) if model else {}
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
    num_cached_samples = data_cfg.get('num_cached_samples', 32000)
    cache_path = data_cfg.get('cache_path', None)
    cache_load_percent = data_cfg.get('cache_load_percent', 100.0)  # Percentage of cache to load
    
    if cache_samples:
        print(f"\n{'='*60}")
        print("CACHED SAMPLES MODE ENABLED")
        print(f"{'='*60}")
        print(f"  Samples will be pre-generated and reused each epoch")
        print(f"  This allows model to learn from repeated exposure")
        print(f"  (Required for hard tasks needing >100 epochs)")
        print(f"  num_cached_samples: {num_cached_samples}")
        if cache_load_percent < 100:
            print(f"  cache_load_percent: {cache_load_percent}% (PARTIAL LOAD for quick testing)")
        if cache_path:
            print(f"  cache_path: {cache_path}")
        print(f"{'='*60}\n")
    
    # Support max_tasks for quick testing (stratified sampling for representativeness)
    max_tasks = data_cfg.get('max_tasks', None)
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
    
    train_dataset = ARCDataset(
        data_cfg['train_path'],
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
    
    # Use hardware.seed for reproducibility (consistent with rest of training)
    global_seed = config.get('hardware', {}).get('seed', 42)
    batch_sampler = BucketedBatchSampler(
        dataset=train_dataset,
        batch_size=batch_size,
        bucket_boundaries=bucket_boundaries,
        drop_last=True,
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
        max_clues=model_config['max_clues'],
        use_stablemax=train_config.get('use_stablemax', True),
        loss_mode=train_config.get('loss_mode', 'focal_stablemax'),
        # BG/FG weight caps for weighted_stablemax (CRITICAL for preventing collapse)
        bg_weight_cap=train_config.get('bg_weight_cap', 2.0),
        fg_weight_cap=train_config.get('fg_weight_cap', 5.0),
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
            import bitsandbytes as bnb
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
    loo_start_epoch: int = 12,
    equiv_start_epoch: int = 8,
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
    
    Returns losses dict AND augmentation statistics for debugging.
    """
    model.train()
    
    total_losses = {
        'total_loss': 0.0,
        'focal_loss': 0.0,
        'entropy_loss': 0.0,
        'sparsity_loss': 0.0,
        'predicate_loss': 0.0,
        'loo_loss': 0.0,  # LOO meta-learning loss
        'equiv_loss': 0.0,  # Augmentation equivariance loss
        'hpm_balance_loss': 0.0,  # HPM load balancing loss
        'curriculum_loss': 0.0,
        'deep_supervision_loss': 0.0,  # FIX: Was missing, caused zero reporting
        'act_loss': 0.0,  # ACT halting loss
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
        'encoder_grad_norm_sum': 0.0,        # Gradient norm to encoder
        'solver_grad_norm_sum': 0.0,         # Gradient norm to solver
        'context_encoder_grad_norm_sum': 0.0, # Gradient norm to context encoder
        'msre_grad_norm_sum': 0.0,           # Gradient norm to MSRE
        
        # DSC attention quality (is DSC learning meaningful patterns?)
        'per_clue_entropy': [],        # Per-clue entropy breakdown (K values)
        'centroid_spread': 0.0,        # How spread out are clue centroids
        'all_logits_count': 0,         # Verify deep supervision is receiving steps
        
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
        mem_tracker.checkpoint("01_batch_on_gpu", batch=batch)
        
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
                mem_tracker.checkpoint("02_after_forward", model=model, outputs=outputs, batch=batch)
                
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
                )
                
                # Compute LOO loss if enabled (meta-learning via HyperLoRA)
                # MEMORY FIX v3: Use iterative backward - backward happens INSIDE the function
                # This prevents O(N) memory accumulation from holding all N computation graphs
                loo_loss_value = 0.0  # Float for logging (backward done inside)
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
                            # v3: Iterative backward - pass scaler so backward happens inside
                            scaler=scaler,
                            loss_weight=loo_loss_fn.config.loss_weight,
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
                equiv_active = equiv_loss_fn is not None and epoch >= equiv_start_epoch
                if equiv_active and hasattr(model, 'hyper_lora') and model.hyper_lora is not None:
                    support_features = outputs.get('support_features')  # (B, N, D, H, W)
                    lora_deltas = outputs.get('lora_deltas')  # Dict with original deltas
                    
                    if support_features is not None and lora_deltas is not None:
                        # Get original context from HyperLoRA output
                        original_context = lora_deltas.get('context')  # (B, D)
                        
                        if original_context is not None:
                            # CRITICAL: Detach to create independent computation graph
                            # This prevents "backward through graph second time" error
                            # The equivariance loss should NOT backprop through the main forward pass
                            original_context = original_context.detach()
                            support_features_detached = support_features.detach()
                            
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
                                
                                # Pool augmented features to get context
                                aug_context = model.hyper_lora.pool_context(aug_features)
                                augmented_contexts[aug_type] = aug_context
                                # MEMORY FIX: Delete intermediate aug_features immediately
                                del aug_features
                            
                            if augmented_contexts:
                                equiv_result, equiv_metrics = equiv_loss_fn(
                                    hyper_lora=model.hyper_lora,
                                    original_context=original_context,
                                    augmented_contexts=augmented_contexts,
                                    # v3: Iterative backward
                                    scaler=scaler,
                                    loss_weight=equiv_loss_fn.config.loss_weight,
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
                            
                            # MEMORY FIX: Clean up augmented contexts after use
                            del augmented_contexts, support_features_detached
                
                # Compute HPM load balancing loss if enabled
                # This loss ensures all memory banks are utilized (prevents mode collapse)
                if hasattr(model, 'use_hpm') and model.use_hpm:
                    hpm_balance_loss = model.hpm_get_load_balance_loss()
                    hpm_balance_weight = config['model'].get('hpm_balance_weight', 0.01)
                    weighted_hpm_loss = hpm_balance_weight * hpm_balance_loss
                    losses['hpm_balance_loss'] = weighted_hpm_loss.item() if torch.is_tensor(weighted_hpm_loss) else weighted_hpm_loss
                    losses['total_loss'] = losses['total_loss'] + weighted_hpm_loss
                
                # Scale loss for gradient accumulation
                # Note: LOO and Equiv losses already have their backward done inside
                loss = losses['total_loss'] / grad_accumulation_steps
            
            # NaN detection: skip batch if loss is NaN
            if not torch.isfinite(loss):
                # Detailed diagnostics on first NaN
                if consecutive_nan == 0:
                    print(f"\n[WARNING] First NaN/Inf loss at batch {batch_idx}!")
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
                    print()
                else:
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
            
            # Loss is finite - but only reset consecutive counter after successful optimizer step
            
            # MEMORY CHECKPOINT: Before backward pass
            mem_tracker.checkpoint("03_before_backward", model=model)
            
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
            mem_tracker.checkpoint("04_after_backward", model=model)
            
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
                mem_tracker.checkpoint("05_after_optim_step", model=model)
                
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
            
            # DSC entropy inputs to stop_predictor (coupling verification)
            if 'dsc_entropy_inputs' in outputs:
                entropy_inputs = outputs['dsc_entropy_inputs']  # (B, K)
                epoch_diagnostics['dsc_entropy_input_mean'] = entropy_inputs.mean().item()
                epoch_diagnostics['per_clue_entropy_input'] = entropy_inputs.mean(dim=0).tolist()  # (K,)
            
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
                    epoch_diagnostics['best_step_histogram'][best_step] = \
                        epoch_diagnostics['best_step_histogram'].get(best_step, 0) + 1
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
            
            for i in range(B):
                pred_i = preds[i]  # (H, W)
                target_i = targets[i]  # (H, W)
                
                # Valid mask (exclude padding)
                valid_mask = target_i != -100
                valid_pixels = valid_mask.sum().item()
                
                if valid_pixels > 0:
                    # Per-sample accuracy
                    correct = ((pred_i == target_i) & valid_mask).sum().item()
                    sample_acc = correct / valid_pixels
                    batch_sample_accuracies.append(sample_acc)
                    
                    # Exact match (100% correct)
                    if correct == valid_pixels:
                        batch_exact_match_count += 1
                        
                        # HPM DYNAMIC BUFFER POPULATION:
                        # When a sample is solved exactly, store its context in dynamic banks
                        # for future retrieval-augmented reasoning on similar tasks.
                        if hasattr(model, 'hpm_add_solved_task') and hasattr(model, 'use_hpm') and model.use_hpm:
                            # Get context embedding for this sample
                            if 'support_features' in outputs:
                                # support_features: (B, N, D, H, W) -> pool to (D,)
                                z_context = outputs['support_features'][i].mean(dim=(0, 2, 3))  # (D,)
                                # Get task embedding if HyperLoRA available
                                z_task = None
                                if 'lora_deltas' in outputs and outputs['lora_deltas'] is not None:
                                    # lora_deltas contains the HyperLoRA output
                                    # We use support_features mean as task embedding
                                    z_task = z_context  # Same context serves as task signature
                                # Add to dynamic buffers
                                task_id = f"epoch{epoch}_batch{batch_idx}_sample{i}"
                                model.hpm_add_solved_task(z_context.unsqueeze(0), z_task.unsqueeze(0) if z_task is not None else None, task_id)
                                epoch_diagnostics['hpm_tasks_added'] = epoch_diagnostics.get('hpm_tasks_added', 0) + 1
                    
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
            if 'lora_deltas' in outputs and outputs['lora_deltas'] is not None:
                lora_norm = 0.0
                for key, delta in outputs['lora_deltas'].items():
                    lora_norm += delta.norm().item() ** 2
                lora_norm = lora_norm ** 0.5
                epoch_diagnostics['lora_delta_norm_sum'] += lora_norm
                epoch_diagnostics['lora_delta_batch_count'] += 1
            
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
                lora_delta_norm = 0.0
                if 'lora_deltas' in outputs and outputs['lora_deltas'] is not None:
                    for key, delta in outputs['lora_deltas'].items():
                        lora_delta_norm += delta.norm().item() ** 2
                    lora_delta_norm = lora_delta_norm ** 0.5
                
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
    
    # Add augmentation diversity stats (CRITICAL for debugging)
    total_losses['aug_stats'] = epoch_aug_stats
    
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


def crop_prediction(pred: np.ndarray, pad_value: int = 10) -> np.ndarray:
    """Crop prediction to remove padding."""
    if pred.ndim == 1:
        pred = pred.reshape(30, 30)
    
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
    
    Args:
        model: RLAN model
        eval_tasks: List of task dicts with 'train' and 'test' keys
        device: Device to run on
        temperature: DSC attention temperature
        num_dihedral: Number of dihedral views (1-8)
        num_color_perms: Number of color permutations per dihedral view
        max_size: Max grid size for padding
        pass_ks: List of K values for Pass@K metrics (default: [1, 2, 3])
        
    Returns:
        Dict with exact_match, pass@k, and voting analysis metrics
    """
    model.eval()
    
    # Default Pass@K values
    if pass_ks is None:
        pass_ks = [1, 2, 3]
    
    correct = 0
    total = len(eval_tasks)
    total_views = num_dihedral * num_color_perms
    
    # Track voting statistics
    all_unique_predictions = []
    all_winner_votes = []
    
    # Track Pass@K metrics
    pass_at_k_correct = {k: 0 for k in pass_ks}
    
    # Progress logging - note we now do 1 forward pass per task (batched views)
    print(f"\n  [TRM-Eval] Running TTA on {total} tasks × {total_views} views", flush=True)
    print(f"  [TRM-Eval] BATCHED: {total} forward passes (B={total_views} each) instead of {total * total_views}", flush=True)
    
    with torch.no_grad():
        for task_idx, task in enumerate(eval_tasks):
            # Progress indicator every 10 tasks
            if (task_idx + 1) % 10 == 0 or task_idx == total - 1:
                print(f"  [TRM-Eval] Task {task_idx + 1}/{total} ({(task_idx + 1) * 100 // total}%)", flush=True)
            
            # Parse task
            train_inputs_np = [np.array(p['input'], dtype=np.int64) for p in task['train']]
            train_outputs_np = [np.array(p['output'], dtype=np.int64) for p in task['train']]
            test_pair = task['test'][0]
            test_input = np.array(test_pair['input'], dtype=np.int64)
            test_output = np.array(test_pair['output'], dtype=np.int64)
            
            num_pairs = len(train_inputs_np)
            max_pairs = 10
            
            # ================================================================
            # BATCHED VIEW PREPARATION: Build all augmented views at once
            # ================================================================
            # We'll create tensors of shape (total_views, ...) and forward all at once
            
            batch_test_inputs = []      # Will be (total_views, H, W)
            batch_train_inputs = []     # Will be (total_views, max_pairs, H, W)
            batch_train_outputs = []    # Will be (total_views, max_pairs, H, W)
            batch_pair_masks = []       # Will be (total_views, max_pairs)
            
            # Store augmentation info for inverse transforms
            aug_infos = []  # List of (dihedral_id, color_perm) tuples
            
            for dihedral_id in range(num_dihedral):
                for color_idx in range(num_color_perms):
                    # CRITICAL ORDER (matching TRM):
                    # Forward: color permutation FIRST, then dihedral
                    # Inverse: inverse_dihedral FIRST, then inverse_color
                    
                    # Step 1: Apply color permutation FIRST (if any)
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
                    
                    # Pad grids
                    train_in_padded = [pad_grid_for_tta(g, max_size, is_target=False) for g in aug_train_in]
                    train_out_padded = [pad_grid_for_tta(g, max_size, is_target=True) for g in aug_train_out]
                    test_in_padded = pad_grid_for_tta(aug_test_in, max_size, is_target=False)
                    
                    # Build tensors for this view
                    input_grids = torch.stack([torch.from_numpy(g) for g in train_in_padded])  # (K, H, W)
                    output_grids = torch.stack([torch.from_numpy(g) for g in train_out_padded])  # (K, H, W)
                    test_input_t = torch.from_numpy(test_in_padded)  # (H, W)
                    
                    # Create pair mask
                    pair_mask = torch.zeros(max_pairs, dtype=torch.bool)
                    pair_mask[:num_pairs] = True
                    
                    # Pad to max_pairs
                    if num_pairs < max_pairs:
                        pad_in = input_grids[0:1].expand(max_pairs - num_pairs, -1, -1)
                        pad_out = output_grids[0:1].expand(max_pairs - num_pairs, -1, -1)
                        input_grids = torch.cat([input_grids, pad_in], dim=0)
                        output_grids = torch.cat([output_grids, pad_out], dim=0)
                    
                    # Collect for batching
                    batch_test_inputs.append(test_input_t)
                    batch_train_inputs.append(input_grids)
                    batch_train_outputs.append(output_grids)
                    batch_pair_masks.append(pair_mask)
                    aug_infos.append((dihedral_id, color_perm))
            
            # ================================================================
            # BATCHED FORWARD PASS: All views in one call
            # ================================================================
            # Stack all views into batch dimension
            batch_test_inputs = torch.stack(batch_test_inputs).to(device)      # (V, H, W)
            batch_train_inputs = torch.stack(batch_train_inputs).to(device)    # (V, K, H, W)
            batch_train_outputs = torch.stack(batch_train_outputs).to(device)  # (V, K, H, W)
            batch_pair_masks = torch.stack(batch_pair_masks).to(device)        # (V, K)
            
            # Single batched forward pass!
            outputs = model(
                batch_test_inputs,
                train_inputs=batch_train_inputs,
                train_outputs=batch_train_outputs,
                pair_mask=batch_pair_masks,
                temperature=temperature,
                return_intermediates=True,
            )
            
            # Best-step selection: use lowest-entropy step for most confident prediction
            if hasattr(model, 'use_best_step_selection') and model.use_best_step_selection:
                all_logits = outputs.get('all_logits')
                if all_logits and len(all_logits) > 1:
                    best_logits, _, _ = model.get_best_step_logits(all_logits, None, None)
                    logits = best_logits
                else:
                    logits = outputs['logits']
            else:
                logits = outputs['logits']  # (V, C, H, W)
            
            # Get predictions for all views
            preds = logits.argmax(dim=1).cpu().numpy()  # (V, H, W)
            
            # ================================================================
            # INVERSE TRANSFORMS AND VOTING (still sequential per view)
            # ================================================================
            predictions = []  # List of (canonical_pred, confidence)
            
            for view_idx, (dihedral_id, color_perm) in enumerate(aug_infos):
                pred = preds[view_idx]  # (H, W)
                
                # Inverse transform to canonical space
                pred_cropped = crop_prediction(pred)
                pred_canonical = inverse_dihedral(pred_cropped, dihedral_id)
                if color_perm is not None:
                    pred_canonical = inverse_color_perm(pred_canonical, color_perm)
                
                predictions.append((pred_canonical, 1.0))  # Confidence fixed at 1.0
            
            # Vote across predictions
            vote_counts = {}  # {hash: {'count': int, 'grid': np.array}}
            
            for pred, conf in predictions:
                h = grid_hash(pred)
                if h not in vote_counts:
                    vote_counts[h] = {'count': 0, 'grid': pred}
                vote_counts[h]['count'] += 1
            
            # Rank predictions by vote count (descending)
            ranked_preds = sorted(vote_counts.values(), key=lambda x: x['count'], reverse=True)
            
            # Track voting stats
            all_unique_predictions.append(len(vote_counts))
            all_winner_votes.append(ranked_preds[0]['count'] if ranked_preds else 0)
            
            # Check Pass@K: is ground truth among top K predictions?
            for k in pass_ks:
                top_k_preds = ranked_preds[:k]
                is_in_top_k = any(
                    p['grid'].shape == test_output.shape and np.array_equal(p['grid'], test_output)
                    for p in top_k_preds
                )
                if is_in_top_k:
                    pass_at_k_correct[k] += 1
            
            # Also check exact match (Pass@1)
            winner_grid = ranked_preds[0]['grid'] if ranked_preds else np.array([[0]])
            is_correct = (
                winner_grid.shape == test_output.shape and
                np.array_equal(winner_grid, test_output)
            )
            
            if is_correct:
                correct += 1
    
    # Compute metrics
    exact_match = correct / max(total, 1)
    avg_unique_preds = sum(all_unique_predictions) / max(len(all_unique_predictions), 1)
    avg_winner_votes = sum(all_winner_votes) / max(len(all_winner_votes), 1)
    max_agreement = max(all_winner_votes) if all_winner_votes else 0
    
    # Compute Pass@K metrics
    pass_at_k = {f'pass@{k}': pass_at_k_correct[k] / max(total, 1) for k in pass_ks}
    
    print(f"  [TRM-Eval] Complete. Exact match: {correct}/{total} ({exact_match*100:.1f}%)")
    
    result = {
        'exact_match': exact_match,
        'correct_tasks': correct,
        'total_tasks': total,
        'avg_unique_predictions': avg_unique_preds,
        'avg_winner_votes': avg_winner_votes,
        'max_agreement': max_agreement,
        'total_views': total_views,
        'pass_ks': pass_ks,
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
    """Save training checkpoint including HPM dynamic buffers."""
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
    
    # Save HPM dynamic buffers (not part of state_dict since they're not nn.Module)
    # These are critical for continual learning - they store solved task memories!
    if hasattr(model, 'hpm_instance_buffer') and model.hpm_instance_buffer is not None:
        if len(model.hpm_instance_buffer) > 0:
            checkpoint['hpm_instance_buffer'] = {
                'd_model': model.hpm_instance_buffer.d_model,
                'max_size': model.hpm_instance_buffer.max_size,
                'keys': list(model.hpm_instance_buffer._keys),
                'values': list(model.hpm_instance_buffer._values),
                'task_ids': list(model.hpm_instance_buffer._task_ids),
            }
            print(f"  HPM Instance Buffer: {len(model.hpm_instance_buffer)} entries saved")
    
    if hasattr(model, 'hpm_procedural_buffer') and model.hpm_procedural_buffer is not None:
        if len(model.hpm_procedural_buffer) > 0:
            checkpoint['hpm_procedural_buffer'] = {
                'd_model': model.hpm_procedural_buffer.d_model,
                'max_size': model.hpm_procedural_buffer.max_size,
                'keys': list(model.hpm_procedural_buffer._keys),
                'values': list(model.hpm_procedural_buffer._values),
                'task_ids': list(model.hpm_procedural_buffer._task_ids),
            }
            print(f"  HPM Procedural Buffer: {len(model.hpm_procedural_buffer)} entries saved")
    
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
    
    # Always load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Restore HPM dynamic buffers (critical for continual learning)
    if 'hpm_instance_buffer' in checkpoint:
        if hasattr(model, 'hpm_instance_buffer') and model.hpm_instance_buffer is not None:
            buf_data = checkpoint['hpm_instance_buffer']
            model.hpm_instance_buffer.clear()
            for key, value, task_id in zip(buf_data['keys'], buf_data['values'], buf_data['task_ids']):
                model.hpm_instance_buffer._keys.append(key)
                model.hpm_instance_buffer._values.append(value)
                model.hpm_instance_buffer._task_ids.append(task_id)
            # Rebuild FAISS index if applicable
            if model.hpm_instance_buffer.use_faiss and model.hpm_instance_buffer._faiss_index is not None:
                model.hpm_instance_buffer._rebuild_faiss_index()
            print(f"  HPM Instance Buffer: {len(model.hpm_instance_buffer)} entries restored")
    
    if 'hpm_procedural_buffer' in checkpoint:
        if hasattr(model, 'hpm_procedural_buffer') and model.hpm_procedural_buffer is not None:
            buf_data = checkpoint['hpm_procedural_buffer']
            model.hpm_procedural_buffer.clear()
            for key, value, task_id in zip(buf_data['keys'], buf_data['values'], buf_data['task_ids']):
                model.hpm_procedural_buffer._keys.append(key)
                model.hpm_procedural_buffer._values.append(value)
                model.hpm_procedural_buffer._task_ids.append(task_id)
            # Rebuild FAISS index if applicable
            if model.hpm_procedural_buffer.use_faiss and model.hpm_procedural_buffer._faiss_index is not None:
                model.hpm_procedural_buffer._rebuild_faiss_index()
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
    train_loader = create_train_loader(
        config,
        curriculum_stage=current_curriculum_stage,
        max_grid_size=max_grid_size,
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
    
    # ADAPTIVE BATCH SIZE INITIALIZATION
    # If resuming from a checkpoint where LOO is already active, we must start with reduced batch size
    current_batch_size_override = None
    if use_loo and start_epoch >= loo_start_epoch:
        base_batch_size = train_cfg['batch_size']
        loo_batch_size = (base_batch_size // loo_max_pairs) - 2
        loo_batch_size = max(loo_batch_size, 4)  # Minimum batch size of 4
        current_batch_size_override = loo_batch_size
        print(f"\n{'='*60}")
        print(f"RESUMING WITH ADAPTIVE BATCH SIZE (LOO active)")
        print(f"{'='*60}")
        print(f"  Original batch: {base_batch_size}")
        print(f"  LOO batch: {loo_batch_size}")
        print(f"{'='*60}\n")
        
        # Recreate loader immediately if resuming into LOO phase
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
    if use_equivariance:
        hidden_dim = config.get('model', {}).get('hidden_dim', 256)
        equiv_loss_fn = AugmentationEquivarianceLoss(
            config=EquivarianceConfig(
                enabled=True,
                loss_weight=equiv_weight,
                num_augmentations=equiv_config.get('num_augmentations', 4),
            ),
            hidden_dim=hidden_dim,
        )
        print(f"Equivariance training configured: weight={equiv_weight}, num_augs={equiv_config.get('num_augmentations', 4)}, start_epoch={equiv_start_epoch}")
    
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
    
    # Target values (what we ramp toward)
    meta_escalation_targets = meta_escalation_config.get('targets', {})
    target_hyperlora_delta_scale = meta_escalation_targets.get('hyperlora_delta_scale', 0.30)
    target_equiv_weight = meta_escalation_targets.get('equiv_loss_weight', 0.05)
    target_loo_weight = meta_escalation_targets.get('loo_loss_weight', 0.10)
    
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
        
        # Scheduled values (what we're ramping toward, may be paused)
        'hyperlora_delta_scale_scheduled': hyperlora_warmup_end_scale,
        'equiv_weight_scheduled': equiv_weight,
        'loo_weight_scheduled': loo_weight,
        
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
    }
    
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
    if use_hpm and hpm_start_epoch > 0:
        print(f"\n{'='*60}")
        print(f"STAGED HPM ENABLED")
        print(f"{'='*60}")
        print(f"  Phase 1 (epochs 1-{hpm_start_epoch}): HPM inactive")
        print(f"    - HPM module exists but use_hpm=False during forward")
        print(f"    - No HPM load balancing loss added")
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
    
    # ADAPTIVE BATCH SIZE TRACKING
    # After LOO activation, batch size is reduced to prevent OOM
    current_batch_size_override = None  # None = use config batch_size
    
    for epoch in range(start_epoch, max_epochs):
        epoch_start = time.time()
        
        # Set epoch for bucketed batch sampler (ensures different batch order each epoch)
        if hasattr(train_loader, 'batch_sampler') and hasattr(train_loader.batch_sampler, 'set_epoch'):
            train_loader.batch_sampler.set_epoch(epoch)
        
        # HPM epoch start callback: reset routing statistics for load balancing
        if hasattr(model, 'hpm_on_epoch_start'):
            model.hpm_on_epoch_start()
        
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
        # ================================================================
        if use_hyperlora and epoch == meta_learning_start_epoch:
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
        
        # HyperLoRA warmup: linearly ramp delta_scale (Patch 1: use delta_scale)
        if use_hyperlora and hasattr(model, 'hyper_lora') and model.hyper_lora is not None:
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
            
            meta_escalation_state['hyperlora_delta_scale_scheduled'] = scheduled_hyperlora
            meta_escalation_state['equiv_weight_scheduled'] = scheduled_equiv
            meta_escalation_state['loo_weight_scheduled'] = scheduled_loo
            meta_escalation_state['escalation_progress'] = scheduled_progress
            
            # Check stability: use info from previous epoch
            # ONLY if require_stability is True; otherwise always follow schedule
            prev_nan_streak = meta_escalation_state['max_nan_streak_epoch']
            prev_grad_events = meta_escalation_state['grad_explosion_events_epoch']
            prev_lr_events = meta_escalation_state['lr_backoff_events_epoch']
            
            if meta_escalation_require_stability:
                # Stability-gated mode: check previous epoch for instability events
                is_stable = (
                    prev_nan_streak <= meta_escalation_max_nan_streak and
                    prev_grad_events <= meta_escalation_max_grad_events and
                    prev_lr_events <= meta_escalation_max_lr_events
                )
            else:
                # Force-schedule mode: always considered stable (for ablations)
                is_stable = True
            meta_escalation_state['is_stable'] = is_stable
            
            if not is_stable:
                meta_escalation_state['escalation_paused'] = True
                meta_escalation_state['stable_epochs_count'] = 0
                # Don't increase weights, but don't decrease either (backoff handles that)
                if meta_escalation_log_every_epoch:
                    print(f"  [Meta Escalation] PAUSED due to instability (nan={prev_nan_streak}, grad={prev_grad_events}, lr={prev_lr_events})")
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
            
            # Reset per-epoch counters for next epoch
            meta_escalation_state['grad_explosion_events_epoch'] = 0
            meta_escalation_state['lr_backoff_events_epoch'] = 0
            meta_escalation_state['max_nan_streak_epoch'] = 0
        
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
            # Formula: loo_batch_size = floor(original_batch / max_loo_pairs) - 2
            # The -2 provides headroom for HPM overhead (~2GB static memory).
            base_batch_size = train_cfg['batch_size']
            loo_batch_size = (base_batch_size // loo_max_pairs) - 2
            loo_batch_size = max(loo_batch_size, 4)  # Minimum batch size of 4
            
            print(f"\n  ADAPTIVE BATCH SIZE:")
            print(f"    Original batch: {base_batch_size}")
            print(f"    LOO batch: floor({base_batch_size} / {loo_max_pairs}) - 2 = {loo_batch_size}")
            print(f"    Memory multiplier: {loo_max_pairs}x forward passes")
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
        
        # Record memory before training epoch for comparison
        if device.type == 'cuda':
            torch.cuda.synchronize()
            pre_epoch_allocated = torch.cuda.memory_allocated() / 1024 / 1024
            torch.cuda.reset_peak_memory_stats()
        
        # Train
        train_losses, global_step = train_epoch(
            model, train_loader, loss_fn, optimizer, device,
            epoch, config, scaler, global_step, ema, effective_loo_fn, effective_equiv_fn,
            loo_start_epoch=loo_start_epoch, equiv_start_epoch=equiv_start_epoch,
        )
        
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
        
        # HPM (Hierarchical Primitive Memory) stats
        if config['model'].get('use_hpm', False) and hasattr(model, 'hpm_get_stats'):
            hpm_stats = model.hpm_get_stats()
            if hpm_stats:
                hpm_balance_loss = train_losses.get('hpm_balance_loss', 0)
                gate_value = hpm_stats.get('gate_value', 0)
                print(f"  HPM Balance Loss: {hpm_balance_loss:.4f} (weight={config['model'].get('hpm_balance_weight', 0.01)})")
                print(f"  HPM Gate Value: {gate_value:.4f} (0=no contribution, 1=full)")
                # Show buffer sizes if dynamic banks enabled
                if 'instance_buffer_size' in hpm_stats:
                    print(f"  HPM Instance Buffer: {hpm_stats['instance_buffer_size']} entries")
                if 'procedural_buffer_size' in hpm_stats:
                    print(f"  HPM Procedural Buffer: {hpm_stats['procedural_buffer_size']} entries")
                # Show tasks added this epoch (from exact matches)
                diagnostics = train_losses.get('diagnostics', {})
                tasks_added = diagnostics.get('hpm_tasks_added', 0)
                if tasks_added > 0:
                    print(f"  HPM Tasks Added (exact matches): {tasks_added}")
        
        print(f"  Time: {epoch_time:.1f}s, LR: {optimizer.param_groups[0]['lr']:.2e}{stage_str}")
        
        # ================================================================
        # META ESCALATION SUMMARY (Dec 2025)
        # ================================================================
        if meta_escalation_enabled and meta_escalation_state['escalation_active']:
            print(f"  Meta Escalation: progress={meta_escalation_state['escalation_progress']:.1%}, stable={meta_escalation_state['is_stable']}")
            print(f"    HyperLoRA: {meta_escalation_state['hyperlora_delta_scale_current']:.4f}/{meta_escalation_state['hyperlora_delta_scale_scheduled']:.4f}")
            print(f"    Equiv: {meta_escalation_state['equiv_weight_current']:.4f}/{meta_escalation_state['equiv_weight_scheduled']:.4f}")
            print(f"    LOO: {meta_escalation_state['loo_weight_current']:.4f}/{meta_escalation_state['loo_weight_scheduled']:.4f}")
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
                print(f"  Stop Prob: {stop_prob:.3f} (approx {clues_used:.1f} clues active)")
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
            centroid_spread = diagnostics.get('centroid_spread', 0)
            if centroid_spread > 0:
                print(f"  Centroid Spread: {centroid_spread:.2f} (higher=more diverse)")
                if centroid_spread < 2.0:
                    print(f"    [!] Clues clustered (spread < 2) - should spread out")
                elif centroid_spread > 8.0:
                    print(f"    Good spread - clues are distributed across grid")
            
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
                equiv_loss_sum = diagnostics.get('equiv_loss_sum', 0.0)
                equiv_batch_count = diagnostics.get('equiv_batch_count', 0)
                if equiv_batch_count > 0:
                    avg_equiv_loss = equiv_loss_sum / equiv_batch_count
                    print(f"  Equivariance Loss (avg): {avg_equiv_loss:.4f}")
                    # Interpret equivariance loss
                    if avg_equiv_loss < 0.05:
                        print(f"    ✓ EXCELLENT: LoRA predictions consistent across augmentations")
                    elif avg_equiv_loss < 0.2:
                        print(f"    ✓ Good: HyperLoRA learning augmentation invariance")
                    elif avg_equiv_loss < 0.5:
                        print(f"    Learning: Equivariance still converging...")
                    else:
                        print(f"    [!] High equivariance loss - LoRA predictions vary with augmentation")
                
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
            if per_clue_stop:
                stop_std = (sum((s - stop_prob)**2 for s in per_clue_stop) / len(per_clue_stop))**0.5
                if stop_std > 0.1:
                    health_checks.append(f"✓ Stop probs adapting (std={stop_std:.2f})")
                elif stop_std > 0.03:
                    health_warnings.append(f"⚠ Stop probs nearly uniform (std={stop_std:.2f})")
                else:
                    if epoch > 10:
                        health_critical.append(f"✗ Stop probs frozen uniform (std={stop_std:.3f})")
                    else:
                        health_warnings.append(f"⚠ Stop probs uniform (std={stop_std:.3f}) - early epoch OK")
            
            # === CHECK 3: Centroid Spread (should be spread, not clustered) ===
            centroid_spread = diagnostics.get('centroid_spread', 0)
            if centroid_spread > 5.0:
                health_checks.append(f"✓ Centroids spread out ({centroid_spread:.1f} > 5)")
            elif centroid_spread > 2.0:
                health_warnings.append(f"⚠ Centroids moderately spread ({centroid_spread:.1f})")
            else:
                if epoch > 10:
                    health_critical.append(f"✗ Centroids clustered ({centroid_spread:.1f} < 2)")
                else:
                    health_warnings.append(f"⚠ Centroids clustered ({centroid_spread:.1f}) - early epoch OK")
            
            # === CHECK 4: Entropy-Stop Coupling (should be positive) ===
            clue_loss_corr = diagnostics.get('clue_loss_correlation', 0)
            if clue_loss_corr > 0.3:
                health_checks.append(f"✓ Good entropy-stop coupling (r={clue_loss_corr:.2f})")
            elif clue_loss_corr > 0:
                health_warnings.append(f"⚠ Weak entropy-stop coupling (r={clue_loss_corr:.2f})")
            else:
                if epoch > 15:
                    health_critical.append(f"✗ No entropy-stop coupling (r={clue_loss_corr:.2f})")
                else:
                    health_warnings.append(f"⚠ Negative coupling (r={clue_loss_corr:.2f}) - early epoch OK")
            
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
            
            eval_metrics = evaluate(eval_model, eval_loader, device, temperature=eval_temp)
            
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
                    trm_metrics = evaluate_trm_style(
                        eval_model, eval_tasks, device, 
                        temperature=eval_temp,
                        num_dihedral=num_dihedral,
                        num_color_perms=num_color_perms,  # TTA with color permutation for max generalization
                        max_size=max_grid_size,
                        pass_ks=pass_ks,
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
                    train_exact_match = epoch_exact_match_pct / 100.0  # Convert to 0-1
                    eval_exact_match = trm_metrics.get('exact_match', 0)
                    gap = train_exact_match - eval_exact_match
                    
                    # GENERALIZATION HEALTH METRICS
                    print(f"\n  --- Generalization Health ---")
                    print(f"  Train Exact Match: {train_exact_match:.1%}")
                    print(f"  Eval Exact Match (TTA): {eval_exact_match:.1%}")
                    print(f"  Delta (Train - Eval): {gap:.1%}")
                    
                    if gap > 0.20:
                        print(f"  🚨 CRITICAL GAP: {gap:.1%} > 20% - Model overfitting!")
                    elif gap > 0.10:
                        print(f"  ⚠️ WARNING GAP: {gap:.1%} > 10% - Monitor closely")
                    elif gap > 0.05:
                        print(f"  ℹ️ Mild gap: {gap:.1%} - Acceptable")
                    else:
                        print(f"  ✅ Healthy gap: {gap:.1%} - Good generalization!")
                    
                    # Check voting consensus
                    consensus_ratio = trm_metrics['avg_winner_votes'] / trm_metrics['total_views']
                    if consensus_ratio < 0.25:  # Less than 25% agreement
                        print(f"  🚨 LOW CONSENSUS: {consensus_ratio:.0%} - Model not dihedral-invariant!")
                    elif consensus_ratio < 0.5:
                        print(f"  ⚠️ Moderate consensus: {consensus_ratio:.0%}")
                    else:
                        print(f"  ✅ Good consensus: {consensus_ratio:.0%}")
            
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
                train_stop_prob = diagnostics.get('stop_prob_mean', 0) if diagnostics else 0
                if train_stop_prob > 0 and eval_stop_prob > 0:
                    ema_diff = abs(train_stop_prob - eval_stop_prob)
                    if ema_diff > 0.2:
                        print(f"  [!] EMA LAG DETECTED: train_stop={train_stop_prob:.3f} vs eval_stop={eval_stop_prob:.3f}")
                        print(f"      EMA decay may be too high (0.999), consider 0.99 or 0.995")
            
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
