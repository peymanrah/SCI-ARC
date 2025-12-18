#!/usr/bin/env python3
"""
RLAN Production-Mimic Test with Real ARC Data
==============================================

This test script EXACTLY mimics the production training pipeline:
- Uses real ARC-AGI data from local directory
- Uses production optimizer with LR multipliers (10x for DSC/MSRE)
- Uses production RLANLoss with all components
- Uses OneCycleLR scheduler with warmup
- Uses gradient accumulation
- Uses EMA model

The differences from production are:
- Runs on CPU instead of GPU
- Uses first N tasks (configurable)

Goal: Achieve 100% accuracy on a subset of ARC tasks.

Usage:
    python scripts/test_prod_mimic.py --num-tasks 10
"""

import sys
import os
import json
import math
import random
from datetime import datetime
from pathlib import Path
from copy import deepcopy
from functools import partial

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import yaml

from sci_arc.models.rlan import RLAN, RLANConfig
from sci_arc.training.rlan_loss import RLANLoss


# =============================================================================
# Local ARC Dataset Loader
# =============================================================================

class LocalARCDataset(Dataset):
    """
    Load ARC tasks from local JSON files.
    Each task has train examples and test examples.
    We flatten all examples into (train_context, test_input, test_output) tuples.
    """
    
    def __init__(
        self,
        data_dir: str,
        max_tasks: int = None,
        max_size: int = 30,
        augment: bool = True,
        num_dihedral: int = 8,
    ):
        self.data_dir = Path(data_dir)
        self.max_size = max_size
        self.augment = augment
        self.num_dihedral = num_dihedral if augment else 1
        
        # Load all task files
        self.tasks = []
        task_files = sorted(self.data_dir.glob("*.json"))
        
        if max_tasks:
            task_files = task_files[:max_tasks]
        
        for task_file in task_files:
            with open(task_file, 'r') as f:
                task_data = json.load(f)
            
            task_name = task_file.stem
            train_examples = task_data.get('train', [])
            test_examples = task_data.get('test', [])
            
            # For each test example, create a training sample
            for test_idx, test_ex in enumerate(test_examples):
                test_input = torch.tensor(test_ex['input'], dtype=torch.long)
                test_output = torch.tensor(test_ex['output'], dtype=torch.long)
                
                # Collect train context
                train_inputs = []
                train_outputs = []
                for train_ex in train_examples:
                    train_inputs.append(torch.tensor(train_ex['input'], dtype=torch.long))
                    train_outputs.append(torch.tensor(train_ex['output'], dtype=torch.long))
                
                self.tasks.append({
                    'name': f"{task_name}_test{test_idx}",
                    'test_input': test_input,
                    'test_output': test_output,
                    'train_inputs': train_inputs,
                    'train_outputs': train_outputs,
                })
        
        print(f"Loaded {len(self.tasks)} samples from {len(task_files)} tasks")
        
        # Dihedral transforms
        self.dihedral_transforms = [
            lambda x: x,                                # Identity
            lambda x: torch.rot90(x, k=1, dims=(0, 1)), # Rotate 90
            lambda x: torch.rot90(x, k=2, dims=(0, 1)), # Rotate 180
            lambda x: torch.rot90(x, k=3, dims=(0, 1)), # Rotate 270
            lambda x: torch.flip(x, dims=[1]),          # Flip H
            lambda x: torch.flip(x, dims=[0]),          # Flip V
            lambda x: torch.flip(torch.rot90(x, k=1, dims=(0, 1)), dims=[1]), # R90+FH
            lambda x: torch.flip(torch.rot90(x, k=1, dims=(0, 1)), dims=[0]), # R90+FV
        ][:self.num_dihedral]
    
    def __len__(self):
        return len(self.tasks) * self.num_dihedral
    
    def _pad_to_size(self, grid, target_h, target_w, pad_value=0):
        """Pad grid to target size."""
        h, w = grid.shape
        if h > target_h or w > target_w:
            # Crop if too large
            grid = grid[:target_h, :target_w]
            h, w = grid.shape
        
        if h == target_h and w == target_w:
            return grid
        
        padded = torch.full((target_h, target_w), pad_value, dtype=grid.dtype)
        padded[:h, :w] = grid
        return padded
    
    def __getitem__(self, idx):
        # Determine which task and which dihedral transform
        task_idx = idx // self.num_dihedral
        dihedral_idx = idx % self.num_dihedral
        
        task = self.tasks[task_idx]
        transform = self.dihedral_transforms[dihedral_idx]
        
        # Apply transform and pad
        test_input = self._pad_to_size(
            transform(task['test_input'].clone()),
            self.max_size, self.max_size
        )
        test_output = self._pad_to_size(
            transform(task['test_output'].clone()),
            self.max_size, self.max_size
        )
        
        # Process train context
        train_inputs = []
        train_outputs = []
        for ti, to in zip(task['train_inputs'], task['train_outputs']):
            train_inputs.append(self._pad_to_size(
                transform(ti.clone()), self.max_size, self.max_size
            ))
            train_outputs.append(self._pad_to_size(
                transform(to.clone()), self.max_size, self.max_size
            ))
        
        # Stack train context
        train_inputs = torch.stack(train_inputs)   # (N, H, W)
        train_outputs = torch.stack(train_outputs) # (N, H, W)
        
        return {
            'test_input': test_input,
            'test_output': test_output,
            'train_inputs': train_inputs,
            'train_outputs': train_outputs,
            'name': f"{task['name']}_d{dihedral_idx}",
        }


def collate_fn(batch, max_train_examples=5):
    """Collate function that handles variable number of train examples."""
    test_inputs = torch.stack([b['test_input'] for b in batch])
    test_outputs = torch.stack([b['test_output'] for b in batch])
    names = [b['name'] for b in batch]
    
    # Handle variable number of train examples by padding/truncating
    max_train = min(max(b['train_inputs'].size(0) for b in batch), max_train_examples)
    H, W = test_inputs.shape[1], test_inputs.shape[2]
    B = len(batch)
    
    train_inputs = torch.zeros(B, max_train, H, W, dtype=torch.long)
    train_outputs = torch.zeros(B, max_train, H, W, dtype=torch.long)
    
    for i, b in enumerate(batch):
        n = min(b['train_inputs'].size(0), max_train)
        train_inputs[i, :n] = b['train_inputs'][:n]
        train_outputs[i, :n] = b['train_outputs'][:n]
    
    return {
        'test_input': test_inputs,
        'test_output': test_outputs,
        'train_inputs': train_inputs,
        'train_outputs': train_outputs,
        'names': names,
    }


# =============================================================================
# Optimizer creation - EXACT copy from train_rlan.py
# =============================================================================

def create_optimizer_exact(model, config, steps_per_epoch=None):
    """Create optimizer and scheduler - EXACT copy from train_rlan.py."""
    train_config = config['training']
    base_lr = train_config['learning_rate']
    weight_decay = train_config['weight_decay']
    
    dsc_lr_mult = train_config.get('dsc_lr_multiplier', 10.0)
    msre_lr_mult = train_config.get('msre_lr_multiplier', 10.0)
    
    # Separate parameters into groups
    dsc_decay_params = []
    dsc_no_decay_params = []
    msre_decay_params = []
    msre_no_decay_params = []
    other_decay_params = []
    other_no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        is_no_decay = 'bias' in name or 'norm' in name or 'embedding' in name
        
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
        else:
            if is_no_decay:
                other_no_decay_params.append(param)
            else:
                other_decay_params.append(param)
    
    param_groups = []
    
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
    
    dsc_count = len(dsc_decay_params) + len(dsc_no_decay_params)
    msre_count = len(msre_decay_params) + len(msre_no_decay_params)
    other_count = len(other_decay_params) + len(other_no_decay_params)
    print(f"  Optimizer param groups:")
    print(f"    DSC: {dsc_count} params @ {dsc_lr_mult}x LR ({base_lr * dsc_lr_mult:.2e})")
    print(f"    MSRE: {msre_count} params @ {msre_lr_mult}x LR ({base_lr * msre_lr_mult:.2e})")
    print(f"    Other: {other_count} params @ 1x LR ({base_lr:.2e})")
    
    beta1 = train_config.get('beta1', 0.9)
    beta2 = train_config.get('beta2', 0.95)
    
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=base_lr,
        betas=(beta1, beta2),
    )
    
    max_epochs = train_config['max_epochs']
    warmup_epochs = train_config.get('warmup_epochs', 10)
    
    steps_per_epoch = steps_per_epoch or 1
    total_steps = max_epochs * steps_per_epoch
    
    max_lrs = [pg.get('lr', base_lr) for pg in param_groups]
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lrs,
        total_steps=total_steps,
        pct_start=min(warmup_epochs / max_epochs, 0.3),
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=1000.0,
    )
    
    return optimizer, scheduler


def get_temperature(epoch, max_epochs, config):
    """Get temperature for Gumbel-softmax based on epoch."""
    tau_start = config['training']['temperature_start']
    tau_end = config['training']['temperature_end']
    
    progress = epoch / max_epochs
    temperature = tau_start * (tau_end / tau_start) ** progress
    
    return max(temperature, 0.5)  # Floor at 0.5 to prevent NaN


def compute_metrics(logits, targets):
    """Compute detailed metrics."""
    B, C, H, W = logits.shape
    preds = logits.argmax(dim=1)
    
    total_correct = 0
    total_pixels = 0
    exact_matches = 0
    
    for b in range(B):
        pred = preds[b]
        target = targets[b]
        
        correct = (pred == target).sum().item()
        total = H * W
        
        total_correct += correct
        total_pixels += total
        
        if correct == total:
            exact_matches += 1
    
    return {
        'accuracy': total_correct / total_pixels * 100,
        'exact_matches': exact_matches,
        'total_samples': B,
        'exact_match_pct': exact_matches / B * 100,
    }


# =============================================================================
# EMA Model
# =============================================================================

class EMAModel:
    """Exponential Moving Average model."""
    
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.original = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)
    
    def apply_shadow(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.original[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
    
    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.original[name])


# =============================================================================
# Main Training Function
# =============================================================================

def train_prod_mimic(max_epochs=200, device='cpu', num_tasks=10, max_size=15):
    """Train RLAN with EXACT production pipeline on real ARC data."""
    
    print("=" * 70)
    print("RLAN Production-Mimic Test with Real ARC Data")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Max Epochs: {max_epochs}")
    print(f"Num Tasks: {num_tasks}")
    print(f"Max Grid Size: {max_size}")
    print()
    
    # Load EXACT production config
    config_path = project_root / 'configs' / 'rlan_minimal.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_cfg = config['model']
    train_cfg = config['training']
    
    # Override for test
    train_cfg['max_epochs'] = max_epochs
    
    print("Configuration (from rlan_minimal.yaml):")
    print(f"  hidden_dim: {model_cfg['hidden_dim']}")
    print(f"  max_clues: {model_cfg['max_clues']}")
    print(f"  num_solver_steps: {model_cfg['num_solver_steps']}")
    print(f"  learning_rate: {train_cfg['learning_rate']}")
    print(f"  weight_decay: {train_cfg['weight_decay']}")
    print(f"  dsc_lr_multiplier: {train_cfg.get('dsc_lr_multiplier', 10.0)}")
    print(f"  temperature_start: {train_cfg['temperature_start']}")
    print(f"  temperature_end: {train_cfg['temperature_end']}")
    print(f"  lambda_sparsity: {train_cfg['lambda_sparsity']}")
    print(f"  lambda_deep_supervision: {train_cfg['lambda_deep_supervision']}")
    print()
    
    # Create dataset from local ARC data
    data_dir = project_root / 'data' / 'arc-agi' / 'data' / 'training'
    
    train_dataset = LocalARCDataset(
        data_dir=str(data_dir),
        max_tasks=num_tasks,
        max_size=max_size,
        augment=True,
        num_dihedral=8,
    )
    
    batch_size = train_cfg.get('batch_size', 8)
    grad_accumulation_steps = train_cfg.get('grad_accumulation_steps', 4)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=partial(collate_fn, max_train_examples=5),
    )
    
    print(f"Dataset: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"Batch size: {batch_size}, Grad accumulation: {grad_accumulation_steps}")
    print()
    
    # Create model
    rlan_config = RLANConfig(
        hidden_dim=model_cfg['hidden_dim'],
        num_colors=model_cfg['num_colors'],
        num_classes=model_cfg['num_classes'],
        max_grid_size=max_size,
        max_clues=model_cfg['max_clues'],
        num_predicates=model_cfg['num_predicates'],
        num_solver_steps=model_cfg['num_solver_steps'],
        dropout=model_cfg['dropout'],
        use_act=model_cfg.get('use_act', False),
        use_context_encoder=model_cfg.get('use_context_encoder', True),
        use_dsc=model_cfg.get('use_dsc', True),
        use_msre=model_cfg.get('use_msre', True),
        use_lcr=model_cfg.get('use_lcr', False),
        use_sph=model_cfg.get('use_sph', False),
        use_learned_pos=model_cfg.get('use_learned_pos', False),
    )
    
    model = RLAN(config=rlan_config).to(device)
    model.train()
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters: {param_count:,}")
    print()
    
    # Create loss function
    loss_fn = RLANLoss(
        focal_gamma=train_cfg['focal_gamma'],
        focal_alpha=train_cfg['focal_alpha'],
        lambda_entropy=train_cfg['lambda_entropy'],
        lambda_sparsity=train_cfg['lambda_sparsity'],
        lambda_deep_supervision=train_cfg['lambda_deep_supervision'],
        loss_mode=train_cfg.get('loss_mode', 'weighted_stablemax'),
        min_clues=train_cfg.get('min_clues', 2.0),
        min_clue_weight=train_cfg.get('min_clue_weight', 5.0),
        ponder_weight=train_cfg.get('ponder_weight', 0.0),
    )
    
    # Create optimizer with LR multipliers
    steps_per_epoch = len(train_loader)
    optimizer, scheduler = create_optimizer_exact(model, config, steps_per_epoch)
    
    # EMA model
    ema = EMAModel(model, decay=train_cfg.get('ema_decay', 0.999))
    
    print()
    print("=" * 70)
    print("Starting Training")
    print("=" * 70)
    
    best_accuracy = 0.0
    best_exact_match = 0
    best_epoch = 0
    
    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_metrics = {'accuracy': 0.0, 'exact_matches': 0, 'total_samples': 0}
        
        temperature = get_temperature(epoch, max_epochs, config)
        
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(train_loader):
            input_grid = batch['test_input'].to(device)
            target_grid = batch['test_output'].to(device)
            train_inputs = batch['train_inputs'].to(device)
            train_outputs = batch['train_outputs'].to(device)
            
            # Forward pass
            outputs = model(
                input_grid=input_grid,
                train_inputs=train_inputs,
                train_outputs=train_outputs,
                temperature=temperature,
                return_intermediates=True,
            )
            
            logits = outputs['logits']
            attention_maps = outputs.get('attention_maps', torch.zeros(1, 6, 12, 12))
            stop_logits = outputs.get('stop_logits', torch.zeros(1, 6))
            predicates = outputs.get('predicates', torch.zeros(1, 32))
            all_logits = outputs.get('all_logits', None)
            
            # Compute loss
            loss_dict = loss_fn(
                logits=logits,
                targets=target_grid,
                attention_maps=attention_maps,
                stop_logits=stop_logits,
                predicates=predicates,
                epoch=epoch,
                max_epochs=max_epochs,
                all_logits=all_logits,
            )
            loss = loss_dict['total_loss']
            
            # Check for NaN
            if torch.isnan(loss):
                print(f"\n⚠️ NaN loss at epoch {epoch+1}, batch {batch_idx+1}")
                print(f"  Temperature: {temperature:.4f}")
                return False
            
            # Scale for gradient accumulation
            loss = loss / grad_accumulation_steps
            loss.backward()
            
            # Gradient accumulation step
            if (batch_idx + 1) % grad_accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                # Update EMA
                ema.update(model)
            
            epoch_loss += loss.item() * grad_accumulation_steps
            
            # Compute metrics
            metrics = compute_metrics(logits, target_grid)
            epoch_metrics['accuracy'] += metrics['accuracy'] * input_grid.size(0)
            epoch_metrics['exact_matches'] += metrics['exact_matches']
            epoch_metrics['total_samples'] += input_grid.size(0)
        
        # Epoch averages
        epoch_loss /= len(train_loader)
        epoch_metrics['accuracy'] /= epoch_metrics['total_samples']
        exact_pct = epoch_metrics['exact_matches'] / epoch_metrics['total_samples'] * 100
        
        # Track best
        if epoch_metrics['accuracy'] > best_accuracy:
            best_accuracy = epoch_metrics['accuracy']
            best_exact_match = epoch_metrics['exact_matches']
            best_epoch = epoch + 1
        
        # Print progress
        lr = scheduler.get_last_lr()[0]
        if (epoch + 1) % 10 == 0 or epoch < 5 or exact_pct >= 50:
            print(f"Epoch {epoch+1:3d}/{max_epochs} | "
                  f"Loss: {epoch_loss:.4f} | "
                  f"Acc: {epoch_metrics['accuracy']:.1f}% | "
                  f"Exact: {epoch_metrics['exact_matches']}/{epoch_metrics['total_samples']} ({exact_pct:.1f}%) | "
                  f"Temp: {temperature:.3f} | "
                  f"LR: {lr:.2e}")
        
        # Early success
        if exact_pct == 100:
            print()
            print("=" * 70)
            print(f"🎉 100% EXACT MATCH ACHIEVED at epoch {epoch+1}!")
            print("=" * 70)
            return True
    
    print()
    print("=" * 70)
    print(f"Training Complete")
    print(f"Best Accuracy: {best_accuracy:.1f}% at epoch {best_epoch}")
    print(f"Best Exact Match: {best_exact_match}/{epoch_metrics['total_samples']}")
    print("=" * 70)
    
    return best_accuracy >= 90.0


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--num-tasks', type=int, default=10, help='Number of ARC tasks to use')
    parser.add_argument('--max-size', type=int, default=15, help='Max grid size (smaller = faster)')
    args = parser.parse_args()
    
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    success = train_prod_mimic(
        max_epochs=args.epochs,
        device=args.device,
        num_tasks=args.num_tasks,
        max_size=args.max_size,
    )
    
    if success:
        print("\n✅ Test PASSED: Production pipeline achieved good accuracy on real ARC data")
        sys.exit(0)
    else:
        print("\n❌ Test FAILED: Could not achieve target accuracy")
        sys.exit(1)
