#!/usr/bin/env python
"""
Production Ablation Test for RLAN on Real ARC-AGI Data

This test:
1. EXACTLY mimics production code (train_rlan.py)
2. Uses real ARC-AGI tasks at 3 difficulty levels
3. Systematically ablates components to find what blocks learning
4. No artificial limits on grid size - handles full ARC data

Goal: Find the minimal set of components needed for 100% exact match.

Usage:
    python scripts/test_prod_ablation.py --ablation full
    python scripts/test_prod_ablation.py --ablation no-dsc
    python scripts/test_prod_ablation.py --ablation no-msre
    python scripts/test_prod_ablation.py --ablation no-context
    python scripts/test_prod_ablation.py --ablation minimal
"""

import argparse
import json
import os
import sys
import time
import random
import math
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sci_arc.models import RLAN, RLANConfig
from sci_arc.training import RLANLoss
from sci_arc.training.ema import EMAHelper


# =============================================================================
# LOGGING UTILITIES
# =============================================================================

def log(msg: str, level: str = "INFO"):
    """Thread-safe logging with timestamp."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [{level}] {msg}", flush=True)


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# ARC-AGI DATA LOADING
# =============================================================================

# Difficulty categories (computed offline)
SIMPLE_TASKS = [
    '0d3d703e', '25d8a9c8', '25ff71a9', '3618c87e', '3c9b0459',
    '5582e5ca', '6150a2bd', '6e02f1e3', '74dd1130', '794b24be'
]

MEDIUM_TASKS = [
    '007bbfb7', '017c7c7b', '0520fde7', '05269061', '08ed6ac7',
    '0ca9ddb6', '10fcaaa3', '11852cab', '1b2d62fb', '1b60fb0c'
]

HARD_TASKS = [
    '00d62c1b', '025d127b', '045e512c', '05f2a901', '06df4c85',
    '09629e4f', '0962bcdd', '0a938d79', '0b148d64', '0dfd9992'
]


def load_arc_task(task_path: str) -> Dict:
    """Load a single ARC task from JSON."""
    with open(task_path, 'r') as f:
        return json.load(f)


def grid_to_tensor(grid: List[List[int]]) -> torch.Tensor:
    """Convert grid to tensor."""
    return torch.tensor(grid, dtype=torch.long)


def apply_dihedral_transform(grid: torch.Tensor, transform_id: int) -> torch.Tensor:
    """Apply one of 8 dihedral group transforms for augmentation."""
    if transform_id == 0:
        return grid
    elif transform_id == 1:
        return torch.rot90(grid, k=1, dims=(-2, -1))
    elif transform_id == 2:
        return torch.rot90(grid, k=2, dims=(-2, -1))
    elif transform_id == 3:
        return torch.rot90(grid, k=3, dims=(-2, -1))
    elif transform_id == 4:
        return torch.flip(grid, dims=[-1])
    elif transform_id == 5:
        return torch.flip(torch.rot90(grid, k=1, dims=(-2, -1)), dims=[-1])
    elif transform_id == 6:
        return torch.flip(grid, dims=[-2])
    elif transform_id == 7:
        return torch.flip(torch.rot90(grid, k=1, dims=(-2, -1)), dims=[-2])
    return grid


class ARCAGIDataset(Dataset):
    """Dataset for real ARC-AGI tasks with augmentation."""
    
    def __init__(
        self,
        task_ids: List[str],
        data_dir: str,
        augment: bool = True,
        num_augmentations: int = 8,  # All 8 dihedral transforms
    ):
        self.data_dir = Path(data_dir)
        self.augment = augment
        self.num_augmentations = num_augmentations if augment else 1
        
        # Load all tasks
        self.samples = []
        for task_id in task_ids:
            task_path = self.data_dir / f"{task_id}.json"
            if not task_path.exists():
                log(f"Warning: Task {task_id} not found", "WARN")
                continue
            
            task_data = load_arc_task(str(task_path))
            
            # Use first test case
            test = task_data['test'][0]
            train_pairs = task_data['train']
            
            self.samples.append({
                'task_id': task_id,
                'train': train_pairs,
                'test_input': test['input'],
                'test_output': test['output'],
            })
        
        log(f"Loaded {len(self.samples)} tasks, {len(self.samples) * self.num_augmentations} samples with augmentation")
    
    def __len__(self):
        return len(self.samples) * self.num_augmentations
    
    def __getitem__(self, idx: int) -> Dict:
        sample_idx = idx // self.num_augmentations
        aug_idx = idx % self.num_augmentations
        
        sample = self.samples[sample_idx]
        
        # Convert grids to tensors
        train_inputs = [grid_to_tensor(p['input']) for p in sample['train']]
        train_outputs = [grid_to_tensor(p['output']) for p in sample['train']]
        test_input = grid_to_tensor(sample['test_input'])
        test_output = grid_to_tensor(sample['test_output'])
        
        # Apply dihedral augmentation
        if self.augment and aug_idx > 0:
            train_inputs = [apply_dihedral_transform(g, aug_idx) for g in train_inputs]
            train_outputs = [apply_dihedral_transform(g, aug_idx) for g in train_outputs]
            test_input = apply_dihedral_transform(test_input, aug_idx)
            test_output = apply_dihedral_transform(test_output, aug_idx)
        
        return {
            'task_id': sample['task_id'],
            'train_inputs': train_inputs,
            'train_outputs': train_outputs,
            'test_input': test_input,
            'test_output': test_output,
            'num_pairs': len(train_inputs),
            'aug_id': aug_idx,
        }


# CRITICAL: Use -100 for padding in targets so loss ignores them (production behavior)
PADDING_IGNORE_VALUE = -100


def pad_grid(grid: torch.Tensor, target_h: int, target_w: int, pad_value: int = 0) -> torch.Tensor:
    """Pad grid to target size.
    
    CRITICAL: Use pad_value=-100 for targets so loss ignores padding!
    """
    h, w = grid.shape[-2], grid.shape[-1]
    if h >= target_h and w >= target_w:
        return grid[..., :target_h, :target_w]
    
    # Calculate padding
    pad_h = max(0, target_h - h)
    pad_w = max(0, target_w - w)
    
    return F.pad(grid.float(), (0, pad_w, 0, pad_h), value=pad_value).long()


def collate_arc_agi(batch: List[Dict]) -> Dict[str, Any]:
    """
    Collate function that matches production collate_sci_arc EXACTLY.
    
    CRITICAL: No artificial grid size limits - uses actual max sizes from batch.
    This allows handling the full range of ARC tasks.
    """
    batch_size = len(batch)
    
    # Find max dimensions across ALL grids in batch (NO artificial limit)
    max_h, max_w = 0, 0
    max_pairs = 0
    
    for sample in batch:
        # Check train grids
        for g in sample['train_inputs']:
            max_h = max(max_h, g.shape[0])
            max_w = max(max_w, g.shape[1])
        for g in sample['train_outputs']:
            max_h = max(max_h, g.shape[0])
            max_w = max(max_w, g.shape[1])
        # Check test grids
        max_h = max(max_h, sample['test_input'].shape[0], sample['test_output'].shape[0])
        max_w = max(max_w, sample['test_input'].shape[1], sample['test_output'].shape[1])
        max_pairs = max(max_pairs, sample['num_pairs'])
    
    # Use square grid for simplicity (production does this too)
    grid_size = max(max_h, max_w)
    
    # Initialize tensors (matching production format)
    # CRITICAL: Use PADDING_IGNORE_VALUE (-100) for targets so loss ignores padding!
    input_grids = torch.zeros(batch_size, max_pairs, grid_size, grid_size, dtype=torch.long)
    output_grids = torch.full((batch_size, max_pairs, grid_size, grid_size), PADDING_IGNORE_VALUE, dtype=torch.long)
    test_inputs = torch.zeros(batch_size, grid_size, grid_size, dtype=torch.long)
    test_outputs = torch.full((batch_size, grid_size, grid_size), PADDING_IGNORE_VALUE, dtype=torch.long)
    grid_masks = torch.zeros(batch_size, max_pairs, dtype=torch.bool)
    
    task_ids = []
    
    for i, sample in enumerate(batch):
        task_ids.append(sample['task_id'])
        n_pairs = sample['num_pairs']
        
        # Pad and store train grids (inputs use 0 for padding, outputs use -100)
        for j in range(n_pairs):
            input_grids[i, j] = pad_grid(sample['train_inputs'][j], grid_size, grid_size, pad_value=0)
            output_grids[i, j] = pad_grid(sample['train_outputs'][j], grid_size, grid_size, pad_value=PADDING_IGNORE_VALUE)
            grid_masks[i, j] = True
        
        # Pad and store test grids (inputs use 0 for padding, outputs use -100)
        test_inputs[i] = pad_grid(sample['test_input'], grid_size, grid_size, pad_value=0)
        test_outputs[i] = pad_grid(sample['test_output'], grid_size, grid_size, pad_value=PADDING_IGNORE_VALUE)
    
    return {
        'task_ids': task_ids,
        'input_grids': input_grids,      # (B, max_pairs, H, W) - train inputs
        'output_grids': output_grids,    # (B, max_pairs, H, W) - train outputs
        'test_inputs': test_inputs,      # (B, H, W) - what model receives
        'test_outputs': test_outputs,    # (B, H, W) - target
        'grid_masks': grid_masks,        # (B, max_pairs) - valid train pairs
        'grid_size': grid_size,          # Actual grid size used
    }


# =============================================================================
# PRODUCTION TRAINING COMPONENTS (EXACT COPY FROM train_rlan.py)
# =============================================================================

def get_temperature(epoch: int, max_epochs: int, temp_start: float = 1.0, temp_end: float = 0.5) -> float:
    """Get temperature for Gumbel-softmax based on epoch."""
    progress = epoch / max(max_epochs, 1)
    temperature = temp_start * (temp_end / temp_start) ** progress
    return temperature


def warmup_lr(optimizer, step: int, warmup_steps: int, initial_lrs: List[float]):
    """Apply linear warmup to learning rate (production code)."""
    if step < warmup_steps:
        lr_scale = float(step + 1) / float(max(1, warmup_steps))
        for i, param_group in enumerate(optimizer.param_groups):
            if i < len(initial_lrs):
                param_group['lr'] = initial_lrs[i] * lr_scale


def compute_grad_norms(model: nn.Module) -> Dict[str, float]:
    """Compute gradient norms for all model components."""
    grad_norms = {}
    
    module_names = ['encoder', 'feature_proj', 'context_encoder', 'context_injector', 
                    'dsc', 'msre', 'lcr', 'solver']
    
    for module_name in module_names:
        if hasattr(model, module_name):
            module = getattr(model, module_name)
            if module is None:
                continue
            
            total_norm = 0.0
            count = 0
            for param in module.parameters():
                if param.grad is not None:
                    norm_val = param.grad.norm().item()
                    if math.isfinite(norm_val):
                        total_norm += norm_val ** 2
                        count += 1
            
            if count > 0:
                grad_norms[module_name] = (total_norm ** 0.5) / count
    
    return grad_norms


# =============================================================================
# ABLATION CONFIGURATIONS
# =============================================================================

def get_ablation_config(ablation_name: str) -> Dict[str, Any]:
    """Get model and training config for each ablation level."""
    
    # Base config (FULL production)
    base_model = {
        'hidden_dim': 256,
        'num_colors': 10,
        'num_classes': 10,
        'max_grid_size': 30,  # No limit - will be overridden by actual data
        'max_clues': 6,
        'num_predicates': 32,
        'num_solver_steps': 6,
        'use_act': False,
        'dropout': 0.1,
        'dsc_num_heads': 4,
        'lcr_num_heads': 4,
        'msre_encoding_dim': 32,
        'msre_num_freq': 8,
        'lcr_num_freq': 8,
        # Module flags
        'use_context_encoder': True,
        'use_dsc': True,
        'use_msre': True,
        'use_lcr': True,
        'use_sph': True,
        'use_learned_pos': False,
    }
    
    base_training = {
        'learning_rate': 5e-4,
        'weight_decay': 0.01,
        'dsc_lr_multiplier': 10.0,
        'msre_lr_multiplier': 10.0,
        'beta1': 0.9,
        'beta2': 0.95,
        'gradient_clip': 1.0,
        'temperature_start': 1.0,
        'temperature_end': 0.5,
        'warmup_epochs': 10,
        # Loss config
        'focal_gamma': 2.0,
        'focal_alpha': 0.25,
        'lambda_entropy': 0.1,
        'lambda_sparsity': 0.01,
        'lambda_predicate': 0.1,
        'lambda_curriculum': 0.0,
        'lambda_deep_supervision': 0.3,
        'lambda_act': 0.1,
        'min_clues': 2.5,
        'min_clue_weight': 5.0,
        'ponder_weight': 0.02,
        'entropy_ponder_weight': 0.02,
        'use_stablemax': True,
        'loss_mode': 'weighted_stablemax',
        'bg_weight_cap': 2.0,
        'fg_weight_cap': 5.0,
        # EMA
        'use_ema': True,
        'ema_mu': 0.999,
        # Mixed precision (CPU doesn't use it but we track the flag)
        'use_amp': False,
        # Use simple optimizer (no per-group LR)
        'use_simple_optimizer': False,
    }
    
    # Simplified training config (matches working test_arc_diagnostic.py)
    # This removes the extra regularization that can cause instability
    stable_training = {
        **base_training,
        'dsc_lr_multiplier': 1.0,   # No LR boost
        'msre_lr_multiplier': 1.0,  # No LR boost
        'use_ema': False,           # Disable EMA
        'use_simple_optimizer': True,  # Use simple AdamW
        # Simplify loss (match test_arc_diagnostic.py)
        'lambda_predicate': 0.0,    # Disable predicate loss
        'min_clues': 0.0,           # Disable min clue constraint
        'min_clue_weight': 0.0,     # Disable min clue weight
        'ponder_weight': 0.0,       # Disable ponder cost
        'entropy_ponder_weight': 0.0, # Disable entropy ponder cost
    }
    
    configs = {
        # FULL production (all components)
        'full': {
            'name': 'Full Production',
            'model': {**base_model},
            'training': {**base_training},
        },
        
        # Stable config: Like diagnostic but with simplified training that works
        'stable': {
            'name': 'Stable (Core with simple optimizer)',
            'model': {
                **base_model,
                'use_lcr': False,
                'use_sph': False,
            },
            'training': {**stable_training},  # Simple optimizer, no EMA
        },
        
        # Diagnostic config (matches test_arc_diagnostic.py - no LCR, no SPH)
        # This is what we confirmed works without NaN
        'diagnostic': {
            'name': 'Diagnostic (No LCR/SPH)',
            'model': {
                **base_model,
                'use_lcr': False,  # LCR disabled - was causing issues
                'use_sph': False,  # SPH disabled
            },
            'training': {**stable_training},  # Use stable training
        },
        
        # Core: Context + DSC + MSRE + Solver (no LCR/SPH)
        'core': {
            'name': 'Core (Context+DSC+MSRE)',
            'model': {
                **base_model,
                'use_lcr': False,
                'use_sph': False,
            },
            'training': {**stable_training},  # Use stable training
        },
        
        # Remove DSC (Dynamic Spatial Cluing)
        'no-dsc': {
            'name': 'No DSC',
            'model': {**base_model, 'use_dsc': False},
            'training': {**stable_training},
        },
        
        # Remove MSRE (Multi-Scale Reasoning Embeddings)
        'no-msre': {
            'name': 'No MSRE',
            'model': {**base_model, 'use_msre': False},
            'training': {**base_training},
        },
        
        # Remove LCR (Latent Coordinate Reasoning)
        'no-lcr': {
            'name': 'No LCR',
            'model': {**base_model, 'use_lcr': False},
            'training': {**base_training},
        },
        
        # Remove context encoder (no training examples)
        'no-context': {
            'name': 'No Context Encoder',
            'model': {**base_model, 'use_context_encoder': False},
            'training': {**base_training},
        },
        
        # Remove SPH (Spatial Predicate Head)
        'no-sph': {
            'name': 'No SPH',
            'model': {**base_model, 'use_sph': False},
            'training': {**base_training},
        },
        
        # Remove deep supervision
        'no-deep-sup': {
            'name': 'No Deep Supervision',
            'model': {**base_model},
            'training': {**base_training, 'lambda_deep_supervision': 0.0},
        },
        
        # Remove EMA
        'no-ema': {
            'name': 'No EMA',
            'model': {**base_model},
            'training': {**base_training, 'use_ema': False},
        },
        
        # Remove DSC + MSRE + LCR (keep only context + solver)
        'no-reasoning': {
            'name': 'No Reasoning Modules (DSC+MSRE+LCR)',
            'model': {
                **base_model,
                'use_dsc': False,
                'use_msre': False,
                'use_lcr': False,
            },
            'training': {**base_training},
        },
        
        # MINIMAL: Just encoder + solver (no bells and whistles)
        'minimal': {
            'name': 'Minimal (Encoder + Solver only)',
            'model': {
                **base_model,
                'use_context_encoder': False,
                'use_dsc': False,
                'use_msre': False,
                'use_lcr': False,
                'use_sph': False,
            },
            'training': {
                **base_training,
                'use_ema': False,
                'lambda_deep_supervision': 0.0,
                'lambda_entropy': 0.0,
                'lambda_sparsity': 0.0,
                'lambda_predicate': 0.0,
            },
        },
        
        # Simplified: Context + Solver only (no DSC/MSRE/LCR)
        'simple-context': {
            'name': 'Simple Context (Context + Solver)',
            'model': {
                **base_model,
                'use_dsc': False,
                'use_msre': False,
                'use_lcr': False,
                'use_sph': False,
            },
            'training': {
                **base_training,
                'lambda_deep_supervision': 0.0,
                'lambda_entropy': 0.0,
                'lambda_sparsity': 0.0,
                'lambda_predicate': 0.0,
            },
        },
    }
    
    if ablation_name not in configs:
        raise ValueError(f"Unknown ablation: {ablation_name}. Available: {list(configs.keys())}")
    
    return configs[ablation_name]


# =============================================================================
# TRAINING LOOP (PRODUCTION-EXACT)
# =============================================================================

def create_model(config: Dict) -> RLAN:
    """Create RLAN model from config."""
    model_cfg = config['model']
    
    rlan_config = RLANConfig(
        hidden_dim=model_cfg['hidden_dim'],
        num_colors=model_cfg['num_colors'],
        num_classes=model_cfg['num_classes'],
        max_grid_size=model_cfg['max_grid_size'],
        max_clues=model_cfg['max_clues'],
        num_predicates=model_cfg['num_predicates'],
        num_solver_steps=model_cfg['num_solver_steps'],
        use_act=model_cfg.get('use_act', False),
        dropout=model_cfg['dropout'],
        dsc_num_heads=model_cfg.get('dsc_num_heads', 4),
        lcr_num_heads=model_cfg.get('lcr_num_heads', 4),
        msre_encoding_dim=model_cfg.get('msre_encoding_dim', 32),
        msre_num_freq=model_cfg.get('msre_num_freq', 8),
        lcr_num_freq=model_cfg.get('lcr_num_freq', 8),
        use_context_encoder=model_cfg.get('use_context_encoder', True),
        use_dsc=model_cfg.get('use_dsc', True),
        use_msre=model_cfg.get('use_msre', True),
        use_lcr=model_cfg.get('use_lcr', True),
        use_sph=model_cfg.get('use_sph', True),
        use_learned_pos=model_cfg.get('use_learned_pos', False),
    )
    
    return RLAN(config=rlan_config)


def create_loss(config: Dict) -> RLANLoss:
    """Create loss function from config."""
    train_cfg = config['training']
    model_cfg = config['model']
    
    return RLANLoss(
        focal_gamma=train_cfg['focal_gamma'],
        focal_alpha=train_cfg['focal_alpha'],
        lambda_entropy=train_cfg['lambda_entropy'],
        lambda_sparsity=train_cfg['lambda_sparsity'],
        lambda_predicate=train_cfg['lambda_predicate'],
        lambda_curriculum=train_cfg['lambda_curriculum'],
        lambda_deep_supervision=train_cfg['lambda_deep_supervision'],
        lambda_act=train_cfg.get('lambda_act', 0.1),
        min_clues=train_cfg.get('min_clues', 2.5),
        min_clue_weight=train_cfg.get('min_clue_weight', 5.0),
        ponder_weight=train_cfg.get('ponder_weight', 0.02),
        entropy_ponder_weight=train_cfg.get('entropy_ponder_weight', 0.02),
        max_clues=model_cfg['max_clues'],
        use_stablemax=train_cfg.get('use_stablemax', True),
        loss_mode=train_cfg.get('loss_mode', 'focal_stablemax'),
        bg_weight_cap=train_cfg.get('bg_weight_cap', 2.0),
        fg_weight_cap=train_cfg.get('fg_weight_cap', 5.0),
    )


def create_optimizer(model: nn.Module, config: Dict, steps_per_epoch: int) -> Tuple:
    """Create optimizer with optional per-module LR (production code)."""
    train_cfg = config['training']
    base_lr = train_cfg['learning_rate']
    weight_decay = train_cfg['weight_decay']
    
    # Check if we should use simple optimizer (no per-group LR)
    use_simple = train_cfg.get('use_simple_optimizer', False)
    
    if use_simple:
        # Simple optimizer like test_arc_diagnostic.py
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=base_lr,
            weight_decay=weight_decay,
            betas=(train_cfg['beta1'], train_cfg['beta2']),
        )
        # No scheduler for simple case
        scheduler = None
        return optimizer, scheduler
    
    # Otherwise, use per-module LR (production)
    dsc_lr_mult = train_cfg.get('dsc_lr_multiplier', 10.0)
    msre_lr_mult = train_cfg.get('msre_lr_multiplier', 10.0)
    
    # Separate parameters into groups
    dsc_decay, dsc_no_decay = [], []
    msre_decay, msre_no_decay = [], []
    other_decay, other_no_decay = [], []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        is_no_decay = 'bias' in name or 'norm' in name or 'embedding' in name
        
        if '.dsc.' in name or name.startswith('dsc.'):
            if is_no_decay:
                dsc_no_decay.append(param)
            else:
                dsc_decay.append(param)
        elif '.msre.' in name or name.startswith('msre.'):
            if is_no_decay:
                msre_no_decay.append(param)
            else:
                msre_decay.append(param)
        else:
            if is_no_decay:
                other_no_decay.append(param)
            else:
                other_decay.append(param)
    
    param_groups = []
    
    if dsc_decay:
        param_groups.append({'params': dsc_decay, 'weight_decay': weight_decay, 
                            'lr': base_lr * dsc_lr_mult, 'name': 'dsc_decay'})
    if dsc_no_decay:
        param_groups.append({'params': dsc_no_decay, 'weight_decay': 0.0,
                            'lr': base_lr * dsc_lr_mult, 'name': 'dsc_no_decay'})
    if msre_decay:
        param_groups.append({'params': msre_decay, 'weight_decay': weight_decay,
                            'lr': base_lr * msre_lr_mult, 'name': 'msre_decay'})
    if msre_no_decay:
        param_groups.append({'params': msre_no_decay, 'weight_decay': 0.0,
                            'lr': base_lr * msre_lr_mult, 'name': 'msre_no_decay'})
    if other_decay:
        param_groups.append({'params': other_decay, 'weight_decay': weight_decay,
                            'lr': base_lr, 'name': 'other_decay'})
    if other_no_decay:
        param_groups.append({'params': other_no_decay, 'weight_decay': 0.0,
                            'lr': base_lr, 'name': 'other_no_decay'})
    
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=base_lr,
        betas=(train_cfg['beta1'], train_cfg['beta2']),
    )
    
    # CosineAnnealingLR (more stable than OneCycleLR for small batches)
    # OneCycleLR causes NaN with high LR multipliers on small datasets
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=steps_per_epoch * args.epochs,
        eta_min=1e-6,
    )
    
    return optimizer, scheduler


def train_and_evaluate(
    config: Dict,
    dataset: ARCAGIDataset,
    device: torch.device,
    epochs: int = 100,
    log_every: int = 10,
) -> Dict[str, Any]:
    """
    Train model and return results.
    
    Uses EXACT production training loop from train_rlan.py.
    """
    train_cfg = config['training']
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=len(dataset),  # Full batch for small tests
        shuffle=True,
        collate_fn=collate_arc_agi,
    )
    
    # Create model
    model = create_model(config)
    model = model.to(device)
    
    # Create loss
    loss_fn = create_loss(config)
    
    # Log model info
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"Model Parameters: {num_params:,}")
    
    # Create optimizer and scheduler
    steps_per_epoch = len(dataloader)
    optimizer, scheduler = create_optimizer(model, config, steps_per_epoch)
    
    # EMA (production uses this)
    ema = None
    if train_cfg.get('use_ema', True):
        ema = EMAHelper(model, mu=train_cfg.get('ema_mu', 0.999))
    
    # Training tracking
    best_accuracy = 0.0
    best_exact = 0
    results_log = []
    
    # Get batch once for consistent evaluation
    batch = next(iter(dataloader))
    
    for epoch in range(1, epochs + 1):
        model.train()
        
        temperature = get_temperature(epoch, epochs, 
                                      train_cfg['temperature_start'],
                                      train_cfg['temperature_end'])
        
        # Move batch to device
        test_inputs = batch['test_inputs'].to(device)
        test_outputs = batch['test_outputs'].to(device)
        train_inputs = batch['input_grids'].to(device)
        train_outputs = batch['output_grids'].to(device)
        pair_mask = batch['grid_masks'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        
        outputs = model(
            test_inputs,
            train_inputs=train_inputs,
            train_outputs=train_outputs,
            pair_mask=pair_mask,
            temperature=temperature,
            return_intermediates=True,
        )
        
        # Compute loss (production loss function)
        losses = loss_fn(
            logits=outputs['logits'],
            targets=test_outputs,
            attention_maps=outputs['attention_maps'],
            stop_logits=outputs['stop_logits'],
            predicates=outputs['predicates'],
            epoch=epoch,
            max_epochs=epochs,
            all_logits=outputs.get('all_logits'),
            act_outputs=outputs.get('act_outputs'),
        )
        
        loss = losses['total_loss']
        
        # Check for NaN
        if not torch.isfinite(loss):
            log(f"NaN/Inf loss detected at epoch {epoch}, skipping update", "WARN")
            optimizer.zero_grad()
            continue
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping (production)
        if train_cfg['gradient_clip'] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg['gradient_clip'])
        
        # Optimizer step
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        
        # EMA update
        if ema is not None:
            ema.update(model)
        
        # Compute metrics
        with torch.no_grad():
            preds = outputs['logits'].argmax(dim=1)
            
            # Only evaluate on valid (non-padded) pixels
            valid_mask = test_outputs != PADDING_IGNORE_VALUE
            
            # Per-pixel accuracy (only on valid pixels)
            if valid_mask.sum() > 0:
                correct = ((preds == test_outputs) & valid_mask).sum()
                accuracy = (correct.float() / valid_mask.sum().float()).item() * 100
            else:
                accuracy = 0.0
            
            # Exact match (comparing only valid pixels per sample)
            batch_size = test_outputs.shape[0]
            exact_matches = 0
            for i in range(batch_size):
                sample_mask = valid_mask[i]
                if sample_mask.sum() > 0:
                    if torch.all(preds[i][sample_mask] == test_outputs[i][sample_mask]):
                        exact_matches += 1
        
        # Track best
        if accuracy > best_accuracy:
            best_accuracy = accuracy
        if exact_matches > best_exact:
            best_exact = exact_matches
        
        # Log progress
        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            log(f"Epoch {epoch:3d}/{epochs} | Loss: {loss.item():.4f} | "
                f"Acc: {accuracy:.1f}% | Exact: {exact_matches}/{batch_size} | "
                f"Temp: {temperature:.3f}")
            
            # Log gradient norms
            grad_norms = compute_grad_norms(model)
            if grad_norms:
                grad_str = ", ".join([f"{k}={v:.2e}" for k, v in sorted(grad_norms.items())])
                log(f"  Gradients: {grad_str}")
            
            results_log.append({
                'epoch': epoch,
                'loss': loss.item(),
                'accuracy': accuracy,
                'exact_matches': exact_matches,
                'batch_size': batch_size,
            })
        
        # Early stopping on 100% exact match
        if exact_matches == batch_size:
            log(f"*** REACHED 100% EXACT MATCH at epoch {epoch} ***")
            break
    
    return {
        'final_accuracy': accuracy,
        'final_exact': exact_matches,
        'best_accuracy': best_accuracy,
        'best_exact': best_exact,
        'batch_size': batch_size,
        'epochs_to_convergence': epoch,
        'converged': exact_matches == batch_size,
        'log': results_log,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    global args
    parser = argparse.ArgumentParser(description='Production Ablation Test')
    parser.add_argument('--ablation', type=str, default='stable',
                       choices=['full', 'stable', 'diagnostic', 'core', 'no-dsc', 'no-msre', 'no-lcr', 'no-context',
                               'no-sph', 'no-deep-sup', 'no-ema', 'no-reasoning',
                               'minimal', 'simple-context'],
                       help='Ablation configuration to test')
    parser.add_argument('--difficulty', type=str, default='simple',
                       choices=['simple', 'medium', 'hard', 'all'],
                       help='Task difficulty level')
    parser.add_argument('--num-tasks', type=int, default=1,
                       help='Number of tasks to test')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Training epochs')
    parser.add_argument('--log-every', type=int, default=10,
                       help='Log every N epochs')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--no-augment', action='store_true',
                       help='Disable augmentation')
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    log("=" * 70)
    log("PRODUCTION ABLATION TEST")
    log("=" * 70)
    log(f"Device: {device}")
    log(f"Ablation: {args.ablation}")
    log(f"Difficulty: {args.difficulty}")
    log(f"Num Tasks: {args.num_tasks}")
    log(f"Epochs: {args.epochs}")
    log(f"Augmentation: {not args.no_augment}")
    log("")
    
    # Get ablation config
    config = get_ablation_config(args.ablation)
    log(f"Config: {config['name']}")
    
    # Select tasks based on difficulty
    if args.difficulty == 'simple':
        task_ids = SIMPLE_TASKS[:args.num_tasks]
    elif args.difficulty == 'medium':
        task_ids = MEDIUM_TASKS[:args.num_tasks]
    elif args.difficulty == 'hard':
        task_ids = HARD_TASKS[:args.num_tasks]
    else:  # all - mix of difficulties
        task_ids = (SIMPLE_TASKS[:args.num_tasks] + 
                   MEDIUM_TASKS[:args.num_tasks] + 
                   HARD_TASKS[:args.num_tasks])
    
    log(f"Tasks: {task_ids}")
    
    # Create dataset
    data_dir = project_root / 'data' / 'arc-agi' / 'data' / 'training'
    dataset = ARCAGIDataset(
        task_ids=task_ids,
        data_dir=str(data_dir),
        augment=not args.no_augment,
        num_augmentations=8,
    )
    
    # Train and evaluate
    results = train_and_evaluate(
        config=config,
        dataset=dataset,
        device=device,
        epochs=args.epochs,
        log_every=args.log_every,
    )
    
    log("")
    log("=" * 70)
    log("RESULTS")
    log("=" * 70)
    log(f"Ablation: {config['name']}")
    log(f"Tasks: {task_ids}")
    log(f"Final Accuracy: {results['final_accuracy']:.1f}%")
    log(f"Final Exact Match: {results['final_exact']}/{results['batch_size']}")
    log(f"Best Accuracy: {results['best_accuracy']:.1f}%")
    log(f"Best Exact Match: {results['best_exact']}/{results['batch_size']}")
    log(f"Epochs to Convergence: {results['epochs_to_convergence']}")
    log(f"Converged (100%): {results['converged']}")
    
    if results['converged']:
        print("\n✓ Test PASSED - 100% exact match achieved")
    else:
        print(f"\n✗ Test INCOMPLETE - {results['final_exact']}/{results['batch_size']} exact match")
    
    return 0 if results['converged'] else 1


if __name__ == '__main__':
    sys.exit(main())
