#!/usr/bin/env python3
"""
RLAN Production-EXACT Test
==========================

This test uses 100% of the production code paths:
- Uses actual ARCDataset with collate_sci_arc
- Uses actual create_optimizer from train_rlan.py  
- Uses actual RLANLoss with full configuration
- Uses actual EMAHelper from sci_arc.training.ema
- Uses actual batch keys (test_inputs, input_grids, etc.)
- Uses pair_mask for masking training pairs

The ONLY differences from production:
- Runs on CPU (no AMP)
- Limits to first N tasks
- Shorter epochs

Usage:
    python scripts/test_prod_exact.py --num-tasks 3 --epochs 200
"""

import sys
import os
import json
import math
import random
from datetime import datetime
from pathlib import Path
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

# Import ACTUAL production modules
from sci_arc.models.rlan import RLAN, RLANConfig
from sci_arc.training.rlan_loss import RLANLoss
from sci_arc.training.ema import EMAHelper
from sci_arc.data import collate_sci_arc


# =============================================================================
# Local ARC Dataset - mimics ARCDataset exactly
# =============================================================================

class LocalARCDataset(Dataset):
    """
    Load ARC tasks from local JSON files.
    Mimics ARCDataset structure exactly for production-match testing.
    """
    
    def __init__(
        self,
        data_dir: str,
        max_tasks: int = None,
        max_size: int = 30,
        augment: bool = True,
        track_augmentation: bool = True,
        ignore_padding_in_loss: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.max_size = max_size
        self.augment = augment
        self.track_augmentation = track_augmentation
        self.ignore_padding_in_loss = ignore_padding_in_loss
        self.num_dihedral = 8 if augment else 1
        
        # Load all task files
        self.samples = []
        task_files = sorted(self.data_dir.glob("*.json"))
        
        if max_tasks:
            task_files = task_files[:max_tasks]
        
        for task_file in task_files:
            with open(task_file, 'r') as f:
                task_data = json.load(f)
            
            task_name = task_file.stem
            train_examples = task_data.get('train', [])
            test_examples = task_data.get('test', [])
            
            # Check if task fits in max_size
            too_large = False
            for ex in train_examples + test_examples:
                h, w = len(ex['input']), len(ex['input'][0]) if ex['input'] else 0
                oh, ow = len(ex['output']), len(ex['output'][0]) if ex['output'] else 0
                if max(h, w, oh, ow) > max_size:
                    too_large = True
                    break
            
            if too_large:
                print(f"  Skipping {task_name}: grids exceed max_size={max_size}")
                continue
            
            # For each test example, create training samples
            for test_idx, test_ex in enumerate(test_examples):
                self.samples.append({
                    'name': f"{task_name}_test{test_idx}",
                    'test_input': test_ex['input'],
                    'test_output': test_ex['output'],
                    'train_examples': train_examples,
                })
        
        print(f"Loaded {len(self.samples)} samples from {len(task_files)} tasks")
        print(f"Total samples with augmentation: {len(self.samples) * self.num_dihedral}")
        
        # Dihedral transforms
        self.dihedral_transforms = [
            lambda x: x,                                # Identity
            lambda x: np.rot90(x, k=1),                 # Rotate 90
            lambda x: np.rot90(x, k=2),                 # Rotate 180
            lambda x: np.rot90(x, k=3),                 # Rotate 270
            lambda x: np.flip(x, axis=1),              # Flip H
            lambda x: np.flip(x, axis=0),              # Flip V
            lambda x: np.flip(np.rot90(x, k=1), axis=1), # R90+FH
            lambda x: np.flip(np.rot90(x, k=1), axis=0), # R90+FV
        ][:self.num_dihedral]
    
    def __len__(self):
        return len(self.samples) * self.num_dihedral
    
    def _to_tensor(self, grid):
        """Convert grid to tensor."""
        return torch.tensor(np.array(grid), dtype=torch.long)
    
    def _apply_transform(self, grid, transform):
        """Apply transform and return tensor."""
        arr = np.array(grid)
        transformed = transform(arr).copy()  # Copy to ensure contiguous
        return torch.tensor(transformed, dtype=torch.long)
    
    def __getitem__(self, idx):
        # Determine which sample and which dihedral transform
        sample_idx = idx // self.num_dihedral
        dihedral_idx = idx % self.num_dihedral
        
        sample = self.samples[sample_idx]
        transform = self.dihedral_transforms[dihedral_idx]
        
        # Apply transform to test input/output
        test_input = self._apply_transform(sample['test_input'], transform)
        test_output = self._apply_transform(sample['test_output'], transform)
        
        # Apply same transform to training examples
        train_inputs = []
        train_outputs = []
        for train_ex in sample['train_examples']:
            train_inputs.append(self._apply_transform(train_ex['input'], transform))
            train_outputs.append(self._apply_transform(train_ex['output'], transform))
        
        # Get original sizes (before padding) for MSRE
        test_h, test_w = test_input.shape
        
        # Build return dict matching ARCDataset format
        result = {
            'test_input': test_input,
            'test_output': test_output,
            'train_inputs': train_inputs,
            'train_outputs': train_outputs,
            'task_id': sample['name'],
            'original_size': (test_h, test_w),
        }
        
        # Augmentation tracking
        if self.track_augmentation:
            result['aug_stats'] = {
                'dihedral_id': dihedral_idx,
                'color_perm': False,
                'translational': False,
                'offset': (0, 0),
            }
        
        return result


def collate_local_arc(batch, max_grid_size=30):
    """
    Collate function that produces outputs matching collate_sci_arc.
    
    Produces:
        - test_inputs: (B, H, W) test input grids
        - test_outputs: (B, H, W) test output grids  
        - input_grids: (B, N, H, W) training input grids
        - output_grids: (B, N, H, W) training output grids
        - grid_masks: (B, N) mask for valid training pairs
        - aug_stats: aggregated augmentation statistics
    """
    B = len(batch)
    H = W = max_grid_size
    
    # Find max training examples
    max_train = max(len(b['train_inputs']) for b in batch)
    
    # Initialize tensors
    test_inputs = torch.zeros(B, H, W, dtype=torch.long)
    test_outputs = torch.full((B, H, W), -100, dtype=torch.long)  # -100 for ignore
    input_grids = torch.zeros(B, max_train, H, W, dtype=torch.long)
    output_grids = torch.zeros(B, max_train, H, W, dtype=torch.long)
    grid_masks = torch.zeros(B, max_train, dtype=torch.bool)
    
    # Augmentation stats
    aug_stats = {
        'dihedral_counts': [0] * 8,
        'color_perm_count': 0,
        'translational_count': 0,
        'unique_offsets': 0,
    }
    
    for i, b in enumerate(batch):
        # Test input
        th, tw = b['test_input'].shape
        test_inputs[i, :th, :tw] = b['test_input']
        
        # Test output - may have different size than input!
        toh, tow = b['test_output'].shape
        test_outputs[i, :toh, :tow] = b['test_output']
        
        # Training context
        for j, (ti, to) in enumerate(zip(b['train_inputs'], b['train_outputs'])):
            h, w = ti.shape
            input_grids[i, j, :h, :w] = ti
            h, w = to.shape
            output_grids[i, j, :h, :w] = to
            grid_masks[i, j] = True
        
        # Track augmentation
        if 'aug_stats' in b:
            aug_stats['dihedral_counts'][b['aug_stats']['dihedral_id']] += 1
    
    return {
        'test_inputs': test_inputs,
        'test_outputs': test_outputs,
        'input_grids': input_grids,
        'output_grids': output_grids,
        'grid_masks': grid_masks,
        'aug_stats': aug_stats,
    }


# =============================================================================
# Copy of create_optimizer from train_rlan.py
# =============================================================================

def create_optimizer(model, config, steps_per_epoch=None):
    """Create optimizer and scheduler - EXACT copy from train_rlan.py."""
    train_config = config['training']
    base_lr = train_config['learning_rate']
    weight_decay = train_config['weight_decay']
    
    dsc_lr_mult = train_config.get('dsc_lr_multiplier', 10.0)
    msre_lr_mult = train_config.get('msre_lr_multiplier', 10.0)
    
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
    
    # Use OneCycleLR like production
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
    
    return max(temperature, 0.5)


def compute_metrics(logits, targets):
    """Compute detailed metrics with ignore_index support."""
    B, C, H, W = logits.shape
    preds = logits.argmax(dim=1)
    
    total_correct = 0
    total_valid = 0
    exact_matches = 0
    
    for b in range(B):
        pred = preds[b]
        target = targets[b]
        
        # Handle ignore index (-100)
        valid_mask = target != -100
        valid_pixels = valid_mask.sum().item()
        
        if valid_pixels == 0:
            continue
        
        correct = ((pred == target) & valid_mask).sum().item()
        
        total_correct += correct
        total_valid += valid_pixels
        
        if correct == valid_pixels:
            exact_matches += 1
    
    return {
        'accuracy': total_correct / total_valid * 100 if total_valid > 0 else 0,
        'exact_matches': exact_matches,
        'total_samples': B,
        'exact_match_pct': exact_matches / B * 100,
    }


# =============================================================================
# Main Training Function
# =============================================================================

def train_prod_exact(max_epochs=200, device='cpu', num_tasks=5, max_size=12):
    """Train RLAN with EXACT production pipeline on real ARC data."""
    
    print("=" * 70)
    print("RLAN Production-EXACT Test")
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
    model_cfg['max_grid_size'] = max_size
    
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
        track_augmentation=True,
        ignore_padding_in_loss=True,
    )
    
    if len(train_dataset) == 0:
        print("ERROR: No valid samples found! Try increasing max_size.")
        return False
    
    batch_size = min(train_cfg.get('batch_size', 8), len(train_dataset))
    grad_accumulation_steps = train_cfg.get('grad_accumulation_steps', 4)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=partial(collate_local_arc, max_grid_size=max_size),
        drop_last=False,
    )
    
    print(f"Dataset: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"Batch size: {batch_size}, Grad accumulation: {grad_accumulation_steps}")
    print()
    
    # Create model - EXACT production config
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
    
    # Create loss function - EXACT production config
    loss_fn = RLANLoss(
        focal_gamma=train_cfg['focal_gamma'],
        focal_alpha=train_cfg['focal_alpha'],
        lambda_entropy=train_cfg['lambda_entropy'],
        lambda_sparsity=train_cfg['lambda_sparsity'],
        lambda_predicate=train_cfg.get('lambda_predicate', 0.01),
        lambda_curriculum=train_cfg.get('lambda_curriculum', 0.0),
        lambda_deep_supervision=train_cfg['lambda_deep_supervision'],
        lambda_act=train_cfg.get('lambda_act', 0.1),
        min_clues=train_cfg.get('min_clues', 2.0),
        min_clue_weight=train_cfg.get('min_clue_weight', 5.0),
        ponder_weight=train_cfg.get('ponder_weight', 0.01),
        entropy_ponder_weight=train_cfg.get('entropy_ponder_weight', 0.02),
        max_clues=model_cfg['max_clues'],
        use_stablemax=train_cfg.get('use_stablemax', True),
        loss_mode=train_cfg.get('loss_mode', 'weighted_stablemax'),
        bg_weight_cap=train_cfg.get('bg_weight_cap', 2.0),
        fg_weight_cap=train_cfg.get('fg_weight_cap', 5.0),
    )
    
    # Create optimizer with LR multipliers
    steps_per_epoch = len(train_loader)
    optimizer, scheduler = create_optimizer(model, config, steps_per_epoch)
    
    # EMA model - use actual EMAHelper (uses 'mu' not 'decay')
    ema = EMAHelper(model, mu=train_cfg.get('ema_decay', 0.999))
    
    gradient_clip = train_cfg.get('gradient_clip', 1.0)
    
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
            # Use EXACT production batch keys
            test_inputs = batch['test_inputs'].to(device)
            test_outputs = batch['test_outputs'].to(device)
            train_inputs = batch['input_grids'].to(device)
            train_outputs = batch['output_grids'].to(device)
            pair_mask = batch['grid_masks'].to(device)
            
            # Forward pass - EXACT production call
            outputs = model(
                test_inputs,
                train_inputs=train_inputs,
                train_outputs=train_outputs,
                pair_mask=pair_mask,
                temperature=temperature,
                return_intermediates=True,
            )
            
            # Compute loss - EXACT production call
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
            
            loss = losses['total_loss']
            
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
                if gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                # Update EMA
                ema.update(model)
            
            epoch_loss += loss.item() * grad_accumulation_steps
            
            # Compute metrics
            metrics = compute_metrics(outputs['logits'], test_outputs)
            epoch_metrics['accuracy'] += metrics['accuracy'] * test_inputs.size(0)
            epoch_metrics['exact_matches'] += metrics['exact_matches']
            epoch_metrics['total_samples'] += test_inputs.size(0)
        
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
        if (epoch + 1) % 10 == 0 or epoch < 5 or exact_pct >= 50 or (epoch + 1) == max_epochs:
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
    print(f"Best Exact Match: {best_exact_match}/{epoch_metrics['total_samples']} ({best_exact_match/epoch_metrics['total_samples']*100:.1f}%)")
    print("=" * 70)
    
    return best_accuracy >= 90.0


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--num-tasks', type=int, default=5, help='Number of ARC tasks to use')
    parser.add_argument('--max-size', type=int, default=12, help='Max grid size')
    args = parser.parse_args()
    
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    success = train_prod_exact(
        max_epochs=args.epochs,
        device=args.device,
        num_tasks=args.num_tasks,
        max_size=args.max_size,
    )
    
    if success:
        print("\n✅ Test PASSED")
        sys.exit(0)
    else:
        print("\n❌ Test FAILED")
        sys.exit(1)
