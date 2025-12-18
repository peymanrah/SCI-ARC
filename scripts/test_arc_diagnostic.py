#!/usr/bin/env python3
"""
ARC Diagnostic Test - Comprehensive Logging
============================================

Tests RLAN on real ARC tasks with extensive diagnostics to find:
1. What prevents learning on real ARC tasks
2. Which modules are essential for training stability
3. Root cause of accuracy plateaus

Key Differences from test_visual_demo_examples.py:
- Uses REAL ARC data (not synthetic examples)
- Provides training context (train_inputs, train_outputs)
- More detailed logging for debugging

Usage:
    python scripts/test_arc_diagnostic.py --num-tasks 2 --epochs 100
"""

import sys
import os
import json
import math
import random
from datetime import datetime
from pathlib import Path
from functools import partial
from collections import defaultdict

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
# Logging Utilities
# =============================================================================

class DiagnosticLogger:
    """Comprehensive logging for debugging training issues."""
    
    def __init__(self, log_path: str = None):
        self.log_path = log_path
        self.logs = []
        self.metrics_history = defaultdict(list)
        
    def log(self, msg: str, level: str = "INFO"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        entry = f"[{timestamp}] [{level}] {msg}"
        self.logs.append(entry)
        print(entry)
        
    def log_tensor(self, name: str, tensor: torch.Tensor):
        """Log tensor statistics."""
        if tensor is None:
            self.log(f"  {name}: None")
            return
        
        if tensor.numel() == 0:
            self.log(f"  {name}: Empty tensor")
            return
            
        stats = {
            'shape': list(tensor.shape),
            'dtype': str(tensor.dtype),
            'min': tensor.min().item(),
            'max': tensor.max().item(),
            'mean': tensor.float().mean().item(),
            'std': tensor.float().std().item() if tensor.numel() > 1 else 0,
        }
        
        # Check for issues
        issues = []
        if torch.isnan(tensor).any():
            issues.append("HAS_NAN")
        if torch.isinf(tensor).any():
            issues.append("HAS_INF")
        
        issue_str = f" ⚠️ {issues}" if issues else ""
        self.log(f"  {name}: shape={stats['shape']}, range=[{stats['min']:.4f}, {stats['max']:.4f}], mean={stats['mean']:.4f}{issue_str}")
        
    def log_gradients(self, model: nn.Module, prefix: str = ""):
        """Log gradient statistics for key modules."""
        grad_stats = {}
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad
                grad_norm = grad.norm().item()
                
                # Group by module
                module_name = name.split('.')[0]
                if module_name not in grad_stats:
                    grad_stats[module_name] = {'norms': [], 'has_nan': False, 'has_inf': False}
                
                grad_stats[module_name]['norms'].append(grad_norm)
                if torch.isnan(grad).any():
                    grad_stats[module_name]['has_nan'] = True
                if torch.isinf(grad).any():
                    grad_stats[module_name]['has_inf'] = True
        
        self.log(f"{prefix}Gradient stats:")
        for module, stats in grad_stats.items():
            norms = stats['norms']
            avg_norm = sum(norms) / len(norms) if norms else 0
            max_norm = max(norms) if norms else 0
            issues = []
            if stats['has_nan']:
                issues.append("NAN")
            if stats['has_inf']:
                issues.append("INF")
            issue_str = f" ⚠️ {issues}" if issues else ""
            self.log(f"    {module}: avg_norm={avg_norm:.4e}, max_norm={max_norm:.4e}{issue_str}")
    
    def log_predictions(self, preds: torch.Tensor, targets: torch.Tensor, sample_idx: int = 0):
        """Log prediction vs target for debugging."""
        pred = preds[sample_idx]
        target = targets[sample_idx]
        
        # Find non-padding region
        valid_mask = target != -100
        if not valid_mask.any():
            self.log(f"  Sample {sample_idx}: No valid pixels")
            return
            
        # Get dimensions
        valid_rows = valid_mask.any(dim=1)
        valid_cols = valid_mask.any(dim=0)
        h = valid_rows.sum().item()
        w = valid_cols.sum().item()
        
        # Compare
        correct = ((pred == target) & valid_mask).sum().item()
        total = valid_mask.sum().item()
        
        self.log(f"  Sample {sample_idx} ({h}x{w}): {correct}/{total} pixels correct ({100*correct/total:.1f}%)")
        
        if h <= 10 and w <= 10:
            # Show the grids
            pred_grid = pred[:h, :w].cpu().numpy()
            target_grid = target[:h, :w].cpu().numpy()
            
            self.log(f"    Target: {target_grid.tolist()}")
            self.log(f"    Pred:   {pred_grid.tolist()}")
    
    def track_metric(self, name: str, value: float):
        self.metrics_history[name].append(value)
        
    def save(self):
        if self.log_path:
            with open(self.log_path, 'w', encoding='utf-8') as f:
                f.write("# ARC Diagnostic Test Log\n\n")
                f.write(f"Generated: {datetime.now().isoformat()}\n\n")
                f.write("## Training Log\n\n```\n")
                for log in self.logs:
                    f.write(log + "\n")
                f.write("```\n")


# =============================================================================
# ARC Dataset - No size limits
# =============================================================================

class ARCDiagnosticDataset(Dataset):
    """Load ARC tasks with no size limits and full diagnostics."""
    
    def __init__(
        self,
        data_dir: str,
        max_tasks: int = None,
        augment: bool = True,
        logger: DiagnosticLogger = None,
    ):
        self.data_dir = Path(data_dir)
        self.augment = augment
        self.num_dihedral = 8 if augment else 1
        self.logger = logger or DiagnosticLogger()
        
        # Load tasks
        self.samples = []
        self.task_info = {}
        
        task_files = sorted(self.data_dir.glob("*.json"))
        if max_tasks:
            task_files = task_files[:max_tasks]
        
        for task_file in task_files:
            with open(task_file, 'r') as f:
                task_data = json.load(f)
            
            task_name = task_file.stem
            train_examples = task_data.get('train', [])
            test_examples = task_data.get('test', [])
            
            # Analyze task
            max_input_h = max_input_w = 0
            max_output_h = max_output_w = 0
            
            for ex in train_examples + test_examples:
                inp = ex['input']
                out = ex['output']
                max_input_h = max(max_input_h, len(inp))
                max_input_w = max(max_input_w, len(inp[0]) if inp else 0)
                max_output_h = max(max_output_h, len(out))
                max_output_w = max(max_output_w, len(out[0]) if out else 0)
            
            self.task_info[task_name] = {
                'train_count': len(train_examples),
                'test_count': len(test_examples),
                'max_input': (max_input_h, max_input_w),
                'max_output': (max_output_h, max_output_w),
            }
            
            self.logger.log(f"Task {task_name}: train={len(train_examples)}, "
                          f"max_in={max_input_h}x{max_input_w}, max_out={max_output_h}x{max_output_w}")
            
            # Create samples for each test example
            for test_idx, test_ex in enumerate(test_examples):
                self.samples.append({
                    'name': f"{task_name}_test{test_idx}",
                    'task_name': task_name,
                    'test_input': test_ex['input'],
                    'test_output': test_ex['output'],
                    'train_examples': train_examples,
                })
        
        self.logger.log(f"Loaded {len(self.samples)} samples from {len(task_files)} tasks")
        self.logger.log(f"Total with augmentation: {len(self.samples) * self.num_dihedral}")
        
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
    
    def _apply_transform(self, grid, transform):
        arr = np.array(grid)
        transformed = np.ascontiguousarray(transform(arr))
        return torch.tensor(transformed, dtype=torch.long)
    
    def __getitem__(self, idx):
        sample_idx = idx // self.num_dihedral
        dihedral_idx = idx % self.num_dihedral
        
        sample = self.samples[sample_idx]
        transform = self.dihedral_transforms[dihedral_idx]
        
        test_input = self._apply_transform(sample['test_input'], transform)
        test_output = self._apply_transform(sample['test_output'], transform)
        
        train_inputs = []
        train_outputs = []
        for train_ex in sample['train_examples']:
            train_inputs.append(self._apply_transform(train_ex['input'], transform))
            train_outputs.append(self._apply_transform(train_ex['output'], transform))
        
        return {
            'test_input': test_input,
            'test_output': test_output,
            'train_inputs': train_inputs,
            'train_outputs': train_outputs,
            'task_name': sample['task_name'],
            'sample_name': sample['name'],
            'dihedral_idx': dihedral_idx,
        }


def collate_arc_diagnostic(batch, pad_value=-100):
    """
    Collate with dynamic grid size (no fixed limit).
    Pads to the maximum size in the batch.
    """
    B = len(batch)
    
    # Find max dimensions in this batch
    max_input_h = max_input_w = 0
    max_output_h = max_output_w = 0
    max_train = 0
    max_train_h = max_train_w = 0
    
    for b in batch:
        ih, iw = b['test_input'].shape
        oh, ow = b['test_output'].shape
        max_input_h = max(max_input_h, ih)
        max_input_w = max(max_input_w, iw)
        max_output_h = max(max_output_h, oh)
        max_output_w = max(max_output_w, ow)
        max_train = max(max_train, len(b['train_inputs']))
        
        for ti in b['train_inputs']:
            h, w = ti.shape
            max_train_h = max(max_train_h, h)
            max_train_w = max(max_train_w, w)
        for to in b['train_outputs']:
            h, w = to.shape
            max_train_h = max(max_train_h, h)
            max_train_w = max(max_train_w, w)
    
    # Use same size for all grids for model compatibility
    H = max(max_input_h, max_output_h, max_train_h)
    W = max(max_input_w, max_output_w, max_train_w)
    
    # Initialize tensors
    test_inputs = torch.zeros(B, H, W, dtype=torch.long)
    test_outputs = torch.full((B, H, W), pad_value, dtype=torch.long)
    input_grids = torch.zeros(B, max_train, H, W, dtype=torch.long)
    output_grids = torch.zeros(B, max_train, H, W, dtype=torch.long)
    grid_masks = torch.zeros(B, max_train, dtype=torch.bool)
    
    metadata = []
    
    for i, b in enumerate(batch):
        # Test input
        ih, iw = b['test_input'].shape
        test_inputs[i, :ih, :iw] = b['test_input']
        
        # Test output
        oh, ow = b['test_output'].shape
        test_outputs[i, :oh, :ow] = b['test_output']
        
        # Training examples
        for j, (ti, to) in enumerate(zip(b['train_inputs'], b['train_outputs'])):
            h, w = ti.shape
            input_grids[i, j, :h, :w] = ti
            h, w = to.shape
            output_grids[i, j, :h, :w] = to
            grid_masks[i, j] = True
        
        metadata.append({
            'task_name': b['task_name'],
            'sample_name': b['sample_name'],
            'dihedral_idx': b['dihedral_idx'],
            'input_size': (ih, iw),
            'output_size': (oh, ow),
        })
    
    return {
        'test_inputs': test_inputs,
        'test_outputs': test_outputs,
        'input_grids': input_grids,
        'output_grids': output_grids,
        'grid_masks': grid_masks,
        'metadata': metadata,
        'grid_size': (H, W),
    }


# =============================================================================
# Metrics
# =============================================================================

def compute_detailed_metrics(logits, targets, metadata=None):
    """Compute detailed per-sample metrics."""
    B, C, H, W = logits.shape
    preds = logits.argmax(dim=1)
    
    results = {
        'total_correct': 0,
        'total_valid': 0,
        'exact_matches': 0,
        'per_sample': [],
    }
    
    for b in range(B):
        pred = preds[b]
        target = targets[b]
        
        valid_mask = target != -100
        valid_pixels = valid_mask.sum().item()
        
        if valid_pixels == 0:
            continue
        
        correct = ((pred == target) & valid_mask).sum().item()
        
        sample_result = {
            'correct': correct,
            'total': valid_pixels,
            'accuracy': correct / valid_pixels * 100,
            'exact_match': correct == valid_pixels,
        }
        
        if metadata and b < len(metadata):
            sample_result.update(metadata[b])
        
        results['per_sample'].append(sample_result)
        results['total_correct'] += correct
        results['total_valid'] += valid_pixels
        
        if correct == valid_pixels:
            results['exact_matches'] += 1
    
    results['accuracy'] = results['total_correct'] / results['total_valid'] * 100 if results['total_valid'] > 0 else 0
    results['exact_match_pct'] = results['exact_matches'] / B * 100
    
    return results


# =============================================================================
# Training
# =============================================================================

def train_diagnostic(
    num_tasks: int = 2,
    max_epochs: int = 100,
    device: str = 'cpu',
    use_train_context: bool = True,
    log_every: int = 10,
):
    """Train RLAN with comprehensive diagnostics."""
    
    logger = DiagnosticLogger(
        log_path=str(project_root / 'docs' / 'arc_diagnostic_log.md')
    )
    
    logger.log("=" * 70)
    logger.log("ARC DIAGNOSTIC TEST")
    logger.log("=" * 70)
    logger.log(f"Device: {device}")
    logger.log(f"Num Tasks: {num_tasks}")
    logger.log(f"Max Epochs: {max_epochs}")
    logger.log(f"Use Train Context: {use_train_context}")
    logger.log("")
    
    # Load config
    config_path = project_root / 'configs' / 'rlan_minimal.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_cfg = config['model']
    train_cfg = config['training']
    
    # Dataset
    data_dir = project_root / 'data' / 'arc-agi' / 'data' / 'training'
    
    dataset = ARCDiagnosticDataset(
        data_dir=str(data_dir),
        max_tasks=num_tasks,
        augment=True,
        logger=logger,
    )
    
    if len(dataset) == 0:
        logger.log("ERROR: No samples loaded!", "ERROR")
        return False
    
    batch_size = min(16, len(dataset))
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_arc_diagnostic,
        drop_last=False,
    )
    
    # Get first batch to determine grid size
    first_batch = next(iter(loader))
    grid_h, grid_w = first_batch['grid_size']
    logger.log(f"Dynamic grid size from batch: {grid_h}x{grid_w}")
    
    # Create model with dynamic grid size
    rlan_config = RLANConfig(
        hidden_dim=model_cfg['hidden_dim'],
        num_colors=model_cfg['num_colors'],
        num_classes=model_cfg['num_classes'],
        max_grid_size=max(grid_h, grid_w),  # Use actual size needed
        max_clues=model_cfg['max_clues'],
        num_predicates=model_cfg['num_predicates'],
        num_solver_steps=model_cfg['num_solver_steps'],
        dropout=model_cfg['dropout'],
        use_act=False,  # Disable ACT for simplicity
        use_context_encoder=model_cfg.get('use_context_encoder', True),
        use_dsc=model_cfg.get('use_dsc', True),
        use_msre=model_cfg.get('use_msre', True),
        use_lcr=False,  # Disable LCR for simplicity
        use_sph=False,
        use_learned_pos=False,
    )
    
    model = RLAN(config=rlan_config).to(device)
    model.train()
    
    param_count = sum(p.numel() for p in model.parameters())
    logger.log(f"Model Parameters: {param_count:,}")
    
    # Loss
    loss_fn = RLANLoss(
        focal_gamma=train_cfg['focal_gamma'],
        focal_alpha=train_cfg['focal_alpha'],
        lambda_entropy=train_cfg['lambda_entropy'],
        lambda_sparsity=train_cfg['lambda_sparsity'],
        lambda_deep_supervision=train_cfg['lambda_deep_supervision'],
        max_clues=model_cfg['max_clues'],
        use_stablemax=True,
        loss_mode='weighted_stablemax',
    )
    
    # Optimizer - simple AdamW like the working test
    lr = train_cfg['learning_rate'] * 5  # Higher LR like working test (0.0005)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=train_cfg['weight_decay'],
        betas=(0.9, 0.95),
    )
    
    logger.log(f"Optimizer: AdamW, LR={lr}")
    logger.log("")
    
    # Temperature schedule
    tau_start = train_cfg['temperature_start']
    tau_end = train_cfg['temperature_end']
    
    def get_temp(epoch):
        progress = epoch / max_epochs
        temp = tau_start * (tau_end / tau_start) ** progress
        return max(temp, 0.5)
    
    logger.log("=" * 70)
    logger.log("STARTING TRAINING")
    logger.log("=" * 70)
    
    best_accuracy = 0.0
    best_exact = 0
    best_epoch = 0
    
    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_metrics = {'accuracy': 0.0, 'exact_matches': 0, 'total': 0}
        
        temperature = get_temp(epoch)
        
        for batch_idx, batch in enumerate(loader):
            test_inputs = batch['test_inputs'].to(device)
            test_outputs = batch['test_outputs'].to(device)
            metadata = batch['metadata']
            
            # Forward pass
            if use_train_context:
                train_inputs = batch['input_grids'].to(device)
                train_outputs = batch['output_grids'].to(device)
                pair_mask = batch['grid_masks'].to(device)
                
                outputs = model(
                    test_inputs,
                    train_inputs=train_inputs,
                    train_outputs=train_outputs,
                    pair_mask=pair_mask,
                    temperature=temperature,
                    return_intermediates=True,
                )
            else:
                # No training context (like visual_demo test)
                outputs = model(
                    test_inputs,
                    temperature=temperature,
                    return_intermediates=True,
                )
            
            # Loss
            losses = loss_fn(
                logits=outputs['logits'],
                targets=test_outputs,
                attention_maps=outputs['attention_maps'],
                stop_logits=outputs['stop_logits'],
                predicates=outputs['predicates'],
                epoch=epoch,
                max_epochs=max_epochs,
                all_logits=outputs.get('all_logits'),
            )
            
            loss = losses['total_loss']
            
            # Check for NaN
            if torch.isnan(loss):
                logger.log(f"⚠️ NaN loss at epoch {epoch+1}, batch {batch_idx+1}", "ERROR")
                logger.log_tensor("logits", outputs['logits'])
                logger.log_tensor("targets", test_outputs)
                return False
            
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Metrics
            metrics = compute_detailed_metrics(outputs['logits'], test_outputs, metadata)
            epoch_metrics['accuracy'] += metrics['accuracy'] * test_inputs.size(0)
            epoch_metrics['exact_matches'] += metrics['exact_matches']
            epoch_metrics['total'] += test_inputs.size(0)
            
            # Detailed logging every N epochs
            if (epoch + 1) % log_every == 0 and batch_idx == 0:
                logger.log(f"\n--- Epoch {epoch+1} Detailed Diagnostics ---")
                logger.log_tensor("logits", outputs['logits'])
                logger.log_tensor("attention_maps", outputs['attention_maps'])
                logger.log_tensor("stop_logits", outputs['stop_logits'])
                
                # Per-sample breakdown
                logger.log("Per-sample accuracy:")
                for i, sr in enumerate(metrics['per_sample'][:4]):  # First 4
                    logger.log(f"  [{sr.get('dihedral_idx', '?')}] {sr.get('task_name', 'unknown')}: "
                              f"{sr['accuracy']:.1f}% ({'✓' if sr['exact_match'] else '✗'})")
                
                # Show prediction for first sample
                logger.log("Sample 0 prediction:")
                logger.log_predictions(outputs['logits'].argmax(dim=1), test_outputs, 0)
                
                logger.log_gradients(model, "  ")
        
        # Epoch summary
        epoch_loss /= len(loader)
        epoch_metrics['accuracy'] /= epoch_metrics['total']
        exact_pct = epoch_metrics['exact_matches'] / epoch_metrics['total'] * 100
        
        # Track best
        if epoch_metrics['accuracy'] > best_accuracy:
            best_accuracy = epoch_metrics['accuracy']
            best_exact = epoch_metrics['exact_matches']
            best_epoch = epoch + 1
        
        logger.track_metric('loss', epoch_loss)
        logger.track_metric('accuracy', epoch_metrics['accuracy'])
        logger.track_metric('exact_pct', exact_pct)
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch < 5 or exact_pct >= 50 or (epoch + 1) == max_epochs:
            logger.log(f"Epoch {epoch+1:3d}/{max_epochs} | "
                      f"Loss: {epoch_loss:.4f} | "
                      f"Acc: {epoch_metrics['accuracy']:.1f}% | "
                      f"Exact: {epoch_metrics['exact_matches']}/{epoch_metrics['total']} ({exact_pct:.1f}%) | "
                      f"Temp: {temperature:.3f}")
        
        # Early success
        if exact_pct == 100:
            logger.log("")
            logger.log("🎉 100% EXACT MATCH ACHIEVED!")
            logger.log(f"   Epoch: {epoch+1}")
            logger.save()
            return True
    
    logger.log("")
    logger.log("=" * 70)
    logger.log("TRAINING COMPLETE")
    logger.log("=" * 70)
    logger.log(f"Best Accuracy: {best_accuracy:.1f}% at epoch {best_epoch}")
    logger.log(f"Best Exact Match: {best_exact}/{epoch_metrics['total']}")
    
    logger.save()
    
    return best_accuracy >= 90.0


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-tasks', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--no-context', action='store_true', 
                       help='Disable training context (like visual demo test)')
    parser.add_argument('--log-every', type=int, default=10)
    args = parser.parse_args()
    
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    success = train_diagnostic(
        num_tasks=args.num_tasks,
        max_epochs=args.epochs,
        device=args.device,
        use_train_context=not args.no_context,
        log_every=args.log_every,
    )
    
    if success:
        print("\n✅ Test PASSED")
        sys.exit(0)
    else:
        print("\n❌ Test FAILED")
        sys.exit(1)
