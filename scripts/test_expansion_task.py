#!/usr/bin/env python3
"""Test grid expansion tasks with the stable config."""

import os
import sys
import json
import torch
import torch.nn as nn
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sci_arc.models.rlan import RLAN
from sci_arc.training.rlan_loss import RLANLoss

# Constants
PADDING_IGNORE_VALUE = -100

def load_task(task_id: str) -> dict:
    """Load a specific ARC task."""
    task_path = project_root / 'data' / 'arc-agi' / 'data' / 'training' / f'{task_id}.json'
    if not task_path.exists():
        raise FileNotFoundError(f"Task not found: {task_path}")
    with open(task_path) as f:
        return json.load(f)

def pad_grid(grid: list, max_h: int, max_w: int, pad_value: int = 0) -> list:
    """Pad grid to max_h x max_w."""
    padded = []
    for row in grid:
        padded_row = list(row) + [pad_value] * (max_w - len(row))
        padded.append(padded_row)
    # Add empty rows
    for _ in range(max_h - len(grid)):
        padded.append([pad_value] * max_w)
    return padded

def prepare_batch(task: dict, device: torch.device):
    """Prepare a batch from task examples - only training examples for this test."""
    examples = task['train']  # Use training examples only
    
    # Find max dimensions across all examples
    max_h, max_w = 0, 0
    for ex in examples:
        max_h = max(max_h, len(ex['input']), len(ex['output']))
        max_w = max(max_w, len(ex['input'][0]), len(ex['output'][0]))
    
    inputs = []
    targets = []
    
    for ex in examples:
        # Pad input with 0 (black)
        inp = pad_grid(ex['input'], max_h, max_w, pad_value=0)
        # Pad target with PADDING_IGNORE_VALUE (-100)
        out = pad_grid(ex['output'], max_h, max_w, pad_value=PADDING_IGNORE_VALUE)
        inputs.append(inp)
        targets.append(out)
    
    # Convert to tensors: (B, H, W) - model expects raw color indices
    inputs_t = torch.tensor(inputs, dtype=torch.long, device=device)
    targets_t = torch.tensor(targets, dtype=torch.long, device=device)
    
    return inputs_t, targets_t, max_h, max_w

def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> tuple:
    """Compute per-pixel and full-grid accuracy."""
    preds = logits.argmax(dim=1)  # (B, H, W)
    
    # Mask for valid (non-padding) pixels
    valid_mask = targets != PADDING_IGNORE_VALUE
    
    # Per-pixel accuracy
    correct_pixels = ((preds == targets) & valid_mask).sum().item()
    total_pixels = valid_mask.sum().item()
    pixel_acc = correct_pixels / total_pixels if total_pixels > 0 else 0.0
    
    # Full grid match - each grid must match exactly
    batch_size = targets.shape[0]
    full_match = 0
    for i in range(batch_size):
        mask_i = valid_mask[i]
        if torch.all(preds[i][mask_i] == targets[i][mask_i]):
            full_match += 1
    grid_acc = full_match / batch_size
    
    return pixel_acc, grid_acc, full_match, batch_size

def train_on_task(task_id: str, max_epochs: int = 100, target_acc: float = 1.0):
    """Train on a single task until 100% accuracy or max epochs."""
    print(f"\n{'='*60}")
    print(f"Training on task: {task_id}")
    print(f"{'='*60}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load task
    task = load_task(task_id)
    print(f"Task has {len(task['train'])} training examples")
    
    # Show grid sizes
    for i, ex in enumerate(task['train']):
        in_h, in_w = len(ex['input']), len(ex['input'][0])
        out_h, out_w = len(ex['output']), len(ex['output'][0])
        print(f"  Example {i}: input {in_h}x{in_w} -> output {out_h}x{out_w}")
    
    # Prepare batch
    inputs, targets, max_h, max_w = prepare_batch(task, device)
    print(f"Padded grid size: {max_h}x{max_w}")
    print(f"Inputs shape: {inputs.shape}, Targets shape: {targets.shape}")
    
    # Count valid target pixels
    valid_pixels = (targets != PADDING_IGNORE_VALUE).sum().item()
    total_pixels = targets.numel()
    print(f"Valid target pixels: {valid_pixels} / {total_pixels} ({100*valid_pixels/total_pixels:.1f}%)")
    
    # Model config - matches production settings
    model_config = {
        'hidden_dim': 128,
        'num_colors': 10,
        'num_classes': 10,
        'max_grid_size': 30,
        'max_clues': 5,
        'num_predicates': 8,
        'num_solver_steps': 6,
        'use_act': False,
        'dropout': 0.1,
    }
    
    model = RLAN(**model_config).to(device)
    model.train()
    
    # Optimizer - stable settings  
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=5e-4,
        weight_decay=0.01
    )
    
    # Loss function - stable settings
    loss_fn = RLANLoss(
        loss_mode='weighted_stablemax',
        bg_weight_cap=2.0,
        fg_weight_cap=5.0,
        lambda_entropy=0.0,
        lambda_sparsity=0.0,
        lambda_predicate=0.0,
        lambda_curriculum=0.0,
        lambda_deep_supervision=0.0,
    )
    
    print("\nTraining...")
    best_grid_acc = 0.0
    
    for epoch in range(max_epochs):
        optimizer.zero_grad()
        
        # Get intermediates for the loss function
        outputs = model(inputs, return_intermediates=True)
        logits = outputs['logits']
        attention_maps = outputs['attention_maps']
        stop_logits = outputs['stop_logits']
        predicates = outputs['predicates']
        
        loss_dict = loss_fn(
            logits, targets, 
            attention_maps, stop_logits, predicates
        )
        loss = loss_dict['total_loss']
        
        if torch.isnan(loss):
            print(f"  Epoch {epoch+1}: NaN detected!")
            return False, best_grid_acc
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        pixel_acc, grid_acc, full_match, batch_size = compute_accuracy(logits, targets)
        best_grid_acc = max(best_grid_acc, grid_acc)
        
        if epoch % 10 == 0 or grid_acc == 1.0:
            print(f"  Epoch {epoch+1}: loss={loss.item():.4f}, pixel_acc={pixel_acc*100:.1f}%, grid_acc={grid_acc*100:.1f}% ({full_match}/{batch_size})")
        
        if grid_acc >= target_acc:
            print(f"\n*** REACHED {grid_acc*100:.0f}% EXACT GRID MATCH at epoch {epoch+1} ***")
            return True, grid_acc
    
    print(f"\nDid not reach {target_acc*100:.0f}% in {max_epochs} epochs. Best: {best_grid_acc*100:.1f}%")
    return False, best_grid_acc


if __name__ == '__main__':
    # Test several expansion tasks - run one at a time
    import sys
    
    expansion_tasks = [
        ('f5b8619d', '6->12, 4x expansion'),  # Moderate
        ('b91ae062', '3->12, 14x expansion'),  # High expansion
        ('007bbfb7', '3->9, 8x expansion (tiling)'),  # The problematic one
    ]
    
    # Take task index from command line, default to 0
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    task_id, desc = expansion_tasks[idx]
    
    print(f"\n{'#'*70}")
    print(f"# TESTING: {task_id} - {desc}")
    print(f"{'#'*70}")
    
    success, best_acc = train_on_task(task_id, max_epochs=300)
    
    status = "SUCCESS" if success else "FAILED"
    print(f"\n\nRESULT: {task_id}: {status} (best={best_acc*100:.1f}%) - {desc}")
