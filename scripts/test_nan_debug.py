#!/usr/bin/env python3
"""
NaN Debug Test - Mimics Production Config Exactly
==================================================

This test replicates EXACTLY the production training setup to find NaN sources:
- Uses rlan_stable.yaml config (weighted_stablemax, clue regularization)
- Uses batch_size=32 like production
- Tests many batches to find NaN occurrence

Run on CPU to isolate: is NaN from AMP/bfloat16 or from computation?

If NaN occurs on CPU: Bug is in the loss/model computation
If NaN only on GPU with AMP: Bug is in mixed precision handling
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


# =============================================================================
# Local ARC Dataset - mimics ARCDataset exactly
# =============================================================================

class LocalARCDataset(Dataset):
    """Load ARC tasks from local JSON files with full augmentation."""
    
    def __init__(
        self,
        data_dir: str,
        max_tasks: int = None,
        max_size: int = 30,
        augment: bool = True,
        samples_per_task: int = 100,  # Mimic cached samples
    ):
        self.data_dir = Path(data_dir)
        self.max_size = max_size
        self.augment = augment
        self.samples_per_task = samples_per_task
        self.num_dihedral = 8 if augment else 1
        
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
            
            # Check if task fits in max_size
            too_large = False
            for ex in train_examples + test_examples:
                h, w = len(ex['input']), len(ex['input'][0]) if ex['input'] else 0
                oh, ow = len(ex['output']), len(ex['output'][0]) if ex['output'] else 0
                if max(h, w, oh, ow) > max_size:
                    too_large = True
                    break
            
            if too_large:
                continue
            
            # For each test example, create task entry
            for test_idx, test_ex in enumerate(test_examples):
                self.tasks.append({
                    'name': f"{task_name}_test{test_idx}",
                    'test_input': test_ex['input'],
                    'test_output': test_ex['output'],
                    'train_examples': train_examples,
                })
        
        # Create sample indices (task_idx, dihedral_idx, color_perm_seed)
        self.samples = []
        for task_idx in range(len(self.tasks)):
            for _ in range(samples_per_task):
                dihedral_idx = random.randint(0, self.num_dihedral - 1)
                color_perm_seed = random.randint(0, 1000000)
                self.samples.append((task_idx, dihedral_idx, color_perm_seed))
        
        print(f"Loaded {len(self.tasks)} tasks, {len(self.samples)} total samples")
        
        # Dihedral transforms
        self.dihedral_transforms = [
            lambda x: x,
            lambda x: np.rot90(x, k=1),
            lambda x: np.rot90(x, k=2),
            lambda x: np.rot90(x, k=3),
            lambda x: np.flip(x, axis=1),
            lambda x: np.flip(x, axis=0),
            lambda x: np.flip(np.rot90(x, k=1), axis=1),
            lambda x: np.flip(np.rot90(x, k=1), axis=0),
        ]
    
    def __len__(self):
        return len(self.samples)
    
    def _apply_transform(self, grid, transform):
        arr = np.array(grid)
        transformed = transform(arr).copy()
        return torch.tensor(transformed, dtype=torch.long)
    
    def _apply_color_perm(self, grid, perm):
        """Apply color permutation."""
        result = grid.clone()
        for old_color, new_color in enumerate(perm):
            result[grid == old_color] = new_color
        return result
    
    def __getitem__(self, idx):
        task_idx, dihedral_idx, color_seed = self.samples[idx]
        task = self.tasks[task_idx]
        transform = self.dihedral_transforms[dihedral_idx]
        
        # Random color permutation (30% chance like production)
        rng = random.Random(color_seed)
        apply_color_perm = rng.random() < 0.3
        if apply_color_perm:
            perm = list(range(10))
            rng.shuffle(perm)
        else:
            perm = None
        
        # Apply transform to test input/output
        test_input = self._apply_transform(task['test_input'], transform)
        test_output = self._apply_transform(task['test_output'], transform)
        
        if perm:
            test_input = self._apply_color_perm(test_input, perm)
            test_output = self._apply_color_perm(test_output, perm)
        
        # Apply to training examples
        train_inputs = []
        train_outputs = []
        for train_ex in task['train_examples']:
            ti = self._apply_transform(train_ex['input'], transform)
            to = self._apply_transform(train_ex['output'], transform)
            if perm:
                ti = self._apply_color_perm(ti, perm)
                to = self._apply_color_perm(to, perm)
            train_inputs.append(ti)
            train_outputs.append(to)
        
        return {
            'test_input': test_input,
            'test_output': test_output,
            'train_inputs': train_inputs,
            'train_outputs': train_outputs,
            'task_id': task['name'],
        }


def collate_fn(batch, max_grid_size=30):
    """Collate function matching production."""
    B = len(batch)
    H = W = max_grid_size
    max_train = max(len(b['train_inputs']) for b in batch)
    
    test_inputs = torch.zeros(B, H, W, dtype=torch.long)
    test_outputs = torch.full((B, H, W), -100, dtype=torch.long)
    input_grids = torch.zeros(B, max_train, H, W, dtype=torch.long)
    output_grids = torch.zeros(B, max_train, H, W, dtype=torch.long)
    grid_masks = torch.zeros(B, max_train, dtype=torch.bool)
    
    for i, b in enumerate(batch):
        th, tw = b['test_input'].shape
        test_inputs[i, :th, :tw] = b['test_input']
        
        toh, tow = b['test_output'].shape
        test_outputs[i, :toh, :tow] = b['test_output']
        
        for j, (ti, to) in enumerate(zip(b['train_inputs'], b['train_outputs'])):
            h, w = ti.shape
            input_grids[i, j, :h, :w] = ti
            h, w = to.shape
            output_grids[i, j, :h, :w] = to
            grid_masks[i, j] = True
    
    return {
        'test_inputs': test_inputs,
        'test_outputs': test_outputs,
        'input_grids': input_grids,
        'output_grids': output_grids,
        'grid_masks': grid_masks,
    }


def check_tensor_for_nan(name, tensor):
    """Check tensor for NaN/Inf and report."""
    if tensor is None:
        return False
    if not torch.is_tensor(tensor):
        return False
    
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    
    if has_nan or has_inf:
        print(f"  [NaN DEBUG] {name}: nan={has_nan}, inf={has_inf}")
        print(f"    shape={tensor.shape}, dtype={tensor.dtype}")
        print(f"    min={tensor.min().item():.6f}, max={tensor.max().item():.6f}")
        if has_nan:
            nan_count = torch.isnan(tensor).sum().item()
            print(f"    NaN count: {nan_count} / {tensor.numel()}")
        return True
    return False


def main():
    print("=" * 70)
    print("NaN Debug Test - Production Config on CPU")
    print("=" * 70)
    print()
    
    # Load production config
    config_path = project_root / 'configs' / 'rlan_stable.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_cfg = config['model']
    train_cfg = config['training']
    
    device = torch.device('cpu')
    max_size = 15  # Smaller grids for faster CPU testing (prod uses 30)
    batch_size = 8  # Smaller batch for faster CPU testing (prod uses 32)
    
    print(f"Config: rlan_stable.yaml")
    print(f"Device: {device}")
    print(f"Max Grid Size: {max_size} (reduced from 30 for CPU speed)")
    print(f"Batch Size: {batch_size} (reduced from 32 for CPU speed)")
    print(f"Loss Mode: {train_cfg.get('loss_mode', 'weighted_stablemax')}")
    print(f"Clue Regularization: min_clues={train_cfg.get('min_clues', 2.5)}, "
          f"min_clue_weight={train_cfg.get('min_clue_weight', 5.0)}")
    print()
    
    # Create dataset
    data_dir = project_root / 'data' / 'arc-agi' / 'data' / 'training'
    
    dataset = LocalARCDataset(
        data_dir=str(data_dir),
        max_tasks=50,  # Only 50 tasks for faster CPU testing
        max_size=max_size,
        augment=True,
        samples_per_task=20,  # 20 samples per task for faster testing
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=partial(collate_fn, max_grid_size=max_size),
        drop_last=False,
    )
    
    print(f"Dataset: {len(dataset)} samples, {len(dataloader)} batches")
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
    print(f"Model: {param_count:,} parameters")
    print()
    
    # Create loss function - EXACT production config
    loss_fn = RLANLoss(
        focal_gamma=train_cfg.get('focal_gamma', 2.0),
        focal_alpha=train_cfg.get('focal_alpha', 0.75),
        lambda_entropy=train_cfg.get('lambda_entropy', 0.01),
        lambda_sparsity=train_cfg.get('lambda_sparsity', 0.5),
        lambda_predicate=train_cfg.get('lambda_predicate', 0.01),
        lambda_curriculum=train_cfg.get('lambda_curriculum', 0.0),
        lambda_deep_supervision=train_cfg.get('lambda_deep_supervision', 0.0),
        lambda_act=train_cfg.get('lambda_act', 0.0),
        min_clues=train_cfg.get('min_clues', 2.5),
        min_clue_weight=train_cfg.get('min_clue_weight', 5.0),
        ponder_weight=train_cfg.get('ponder_weight', 0.02),
        entropy_ponder_weight=train_cfg.get('entropy_ponder_weight', 0.02),
        max_clues=model_cfg['max_clues'],
        use_stablemax=train_cfg.get('use_stablemax', True),
        loss_mode=train_cfg.get('loss_mode', 'weighted_stablemax'),
        bg_weight_cap=train_cfg.get('bg_weight_cap', 2.0),
        fg_weight_cap=train_cfg.get('fg_weight_cap', 5.0),
    )
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg['learning_rate'],
        weight_decay=train_cfg['weight_decay'],
        betas=(train_cfg.get('beta1', 0.9), train_cfg.get('beta2', 0.95)),
    )
    
    gradient_clip = train_cfg.get('gradient_clip', 1.0)
    
    # Temperature
    temperature = train_cfg['temperature_start']
    
    print("=" * 70)
    print("Starting NaN Debug Training (1 epoch, many batches)")
    print("=" * 70)
    print()
    
    nan_found = False
    nan_batch_idx = -1
    
    for batch_idx, batch in enumerate(dataloader):
        # Move to device
        test_inputs = batch['test_inputs'].to(device)
        test_outputs = batch['test_outputs'].to(device)
        train_inputs = batch['input_grids'].to(device)
        train_outputs = batch['output_grids'].to(device)
        pair_mask = batch['grid_masks'].to(device)
        
        # Check inputs for NaN
        input_nan = (
            check_tensor_for_nan("test_inputs", test_inputs) or
            check_tensor_for_nan("train_inputs", train_inputs)
        )
        
        if input_nan:
            print(f"[!] NaN in INPUT at batch {batch_idx}")
            nan_found = True
            nan_batch_idx = batch_idx
            break
        
        # Forward pass
        try:
            outputs = model(
                test_inputs,
                train_inputs=train_inputs,
                train_outputs=train_outputs,
                pair_mask=pair_mask,
                temperature=temperature,
                return_intermediates=True,
            )
        except Exception as e:
            print(f"[!] Forward pass exception at batch {batch_idx}: {e}")
            nan_found = True
            nan_batch_idx = batch_idx
            break
        
        # Check outputs for NaN and extreme values
        output_nan = (
            check_tensor_for_nan("logits", outputs['logits']) or
            check_tensor_for_nan("attention_maps", outputs['attention_maps']) or
            check_tensor_for_nan("stop_logits", outputs['stop_logits']) or
            check_tensor_for_nan("predicates", outputs['predicates'])
        )
        
        # Check for extreme values that could cause NaN in backward
        logits_max = outputs['logits'].abs().max().item()
        attn_min = outputs['attention_maps'].min().item()
        if logits_max > 100:
            print(f"  [WARNING] batch {batch_idx}: logits max abs = {logits_max:.2f} (may cause NaN)")
        if attn_min < 1e-10:
            print(f"  [WARNING] batch {batch_idx}: attention min = {attn_min:.2e} (may cause log(0))")
        
        if output_nan:
            print(f"[!] NaN in MODEL OUTPUT at batch {batch_idx}")
            nan_found = True
            nan_batch_idx = batch_idx
            break
        
        # Compute loss
        try:
            losses = loss_fn(
                logits=outputs['logits'],
                targets=test_outputs,
                attention_maps=outputs['attention_maps'],
                stop_logits=outputs['stop_logits'],
                predicates=outputs['predicates'],
                epoch=0,
                max_epochs=100,
                all_logits=outputs.get('all_logits'),
                act_outputs=outputs.get('act_outputs'),
            )
        except Exception as e:
            print(f"[!] Loss computation exception at batch {batch_idx}: {e}")
            import traceback
            traceback.print_exc()
            nan_found = True
            nan_batch_idx = batch_idx
            break
        
        # Check loss for NaN
        loss = losses['total_loss']
        
        if not torch.isfinite(loss):
            print(f"[!] NaN/Inf in LOSS at batch {batch_idx}")
            print(f"    total_loss: {loss.item()}")
            print(f"    task_loss: {losses['task_loss'].item()}")
            print(f"    entropy_loss: {losses['entropy_loss'].item()}")
            print(f"    sparsity_loss: {losses['sparsity_loss'].item()}")
            print(f"    predicate_loss: {losses['predicate_loss'].item()}")
            nan_found = True
            nan_batch_idx = batch_idx
            break
        
        # Backward pass with gradient hooks to trace NaN origin
        optimizer.zero_grad()
        
        # Register gradient hooks to find NaN source
        nan_grad_names = []
        def make_hook(name):
            def hook(grad):
                if grad is not None and not torch.isfinite(grad).all():
                    nan_grad_names.append(name)
            return hook
        
        handles = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                h = param.register_hook(make_hook(name))
                handles.append(h)
        
        loss.backward()
        
        # Remove hooks
        for h in handles:
            h.remove()
        
        # Report NaN gradients
        if nan_grad_names:
            print(f"[!] NaN gradients in {len(nan_grad_names)} parameters at batch {batch_idx}:")
            for name in nan_grad_names[:5]:  # Show first 5
                print(f"    - {name}")
            nan_found = True
            nan_batch_idx = batch_idx
            break
        
        # Gradient clipping
        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        
        # Optimizer step
        optimizer.step()
        
        # Check weights for NaN after update
        weight_nan = False
        for name, param in model.named_parameters():
            if not torch.isfinite(param).all():
                print(f"[!] NaN/Inf in WEIGHTS after update: {name}")
                weight_nan = True
                break
        
        if weight_nan:
            print(f"[!] NaN in WEIGHTS at batch {batch_idx}")
            nan_found = True
            nan_batch_idx = batch_idx
            break
        
        # Progress
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx:4d}/{len(dataloader)}: "
                  f"loss={loss.item():.4f}, "
                  f"task={losses['task_loss'].item():.4f}, "
                  f"sparsity={losses['sparsity_loss'].item():.4f}")
        
        # Stop after enough batches to stress test
        if batch_idx >= 300:
            print(f"\nCompleted 300 batches without NaN!")
            break
    
    print()
    print("=" * 70)
    if nan_found:
        print(f"RESULT: NaN FOUND at batch {nan_batch_idx}")
        print("This indicates a BUG in the computation (not AMP related)")
    else:
        print("RESULT: No NaN found in 300 batches on CPU")
        print("If NaN occurs on GPU with AMP, it's likely bfloat16 precision issue")
        print("Consider disabling AMP or using float32")
    print("=" * 70)


if __name__ == "__main__":
    main()
