#!/usr/bin/env python3
"""
Test grid expansion tasks WITH ContextEncoder enabled.

This test compares learning with vs without ContextEncoder to see
if the context encoding helps or hurts learning.

Key difference from test_expansion_task.py:
- Provides train_inputs/train_outputs to the model
- Uses the same training examples as demos (task has multiple train pairs)
"""

import os
import sys
import json
import torch
import torch.nn as nn
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sci_arc.models.rlan import RLAN, RLANConfig
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

def prepare_batch_with_context(task: dict, device: torch.device):
    """
    Prepare a batch from task examples WITH context for ContextEncoder.
    
    For each training example:
    - test_input = that example's input
    - test_output = that example's output (target)
    - train_inputs/outputs = ALL training examples (including this one as demo)
    
    This mimics how ARC works: you see demos and predict for a test case.
    """
    examples = task['train']
    
    # Find max dimensions across all examples
    max_h, max_w = 0, 0
    for ex in examples:
        max_h = max(max_h, len(ex['input']), len(ex['output']))
        max_w = max(max_w, len(ex['input'][0]), len(ex['output'][0]))
    
    # Prepare test inputs/outputs (what we predict)
    test_inputs = []
    test_outputs = []
    
    # Prepare training context (demos shown to the model)
    all_train_inputs = []
    all_train_outputs = []
    
    for ex in examples:
        # Test: input with 0 padding, output with -100 padding
        test_inp = pad_grid(ex['input'], max_h, max_w, pad_value=0)
        test_out = pad_grid(ex['output'], max_h, max_w, pad_value=PADDING_IGNORE_VALUE)
        test_inputs.append(test_inp)
        test_outputs.append(test_out)
        
        # Training context: all examples as demos (pad with 0 for inputs)
        train_inp = pad_grid(ex['input'], max_h, max_w, pad_value=0)
        train_out = pad_grid(ex['output'], max_h, max_w, pad_value=0)  # No -100 for context
        all_train_inputs.append(train_inp)
        all_train_outputs.append(train_out)
    
    B = len(examples)
    N = len(examples)  # All examples as context
    
    # Convert to tensors
    test_inputs_t = torch.tensor(test_inputs, dtype=torch.long, device=device)  # (B, H, W)
    test_outputs_t = torch.tensor(test_outputs, dtype=torch.long, device=device)  # (B, H, W)
    
    # Stack training context: each sample sees ALL training examples as demos
    # Shape: (B, N, H, W)
    train_inputs_t = torch.tensor(all_train_inputs, dtype=torch.long, device=device)
    train_outputs_t = torch.tensor(all_train_outputs, dtype=torch.long, device=device)
    
    # Expand so each test sample sees all training examples
    train_inputs_t = train_inputs_t.unsqueeze(0).expand(B, -1, -1, -1)  # (B, N, H, W)
    train_outputs_t = train_outputs_t.unsqueeze(0).expand(B, -1, -1, -1)  # (B, N, H, W)
    
    # Pair mask: all pairs are valid
    pair_mask = torch.ones(B, N, dtype=torch.bool, device=device)
    
    return test_inputs_t, test_outputs_t, train_inputs_t, train_outputs_t, pair_mask, max_h, max_w

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

def train_on_task(task_id: str, use_context_encoder: bool = True, max_epochs: int = 200, target_acc: float = 1.0):
    """Train on a single task with optional ContextEncoder."""
    print(f"\n{'='*60}")
    print(f"Training on task: {task_id}")
    print(f"ContextEncoder: {'ENABLED' if use_context_encoder else 'DISABLED'}")
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
    
    # Prepare batch WITH context
    test_inputs, test_outputs, train_inputs, train_outputs, pair_mask, max_h, max_w = prepare_batch_with_context(task, device)
    print(f"\nPadded grid size: {max_h}x{max_w}")
    print(f"Test inputs shape: {test_inputs.shape}")
    print(f"Train inputs shape: {train_inputs.shape}")
    
    # Count valid target pixels
    valid_pixels = (test_outputs != PADDING_IGNORE_VALUE).sum().item()
    total_pixels = test_outputs.numel()
    print(f"Valid target pixels: {valid_pixels} / {total_pixels} ({100*valid_pixels/total_pixels:.1f}%)")
    
    # Model config - matches production settings but with smaller size for quick testing
    config = RLANConfig(
        hidden_dim=128,
        num_colors=10,
        num_classes=10,
        max_grid_size=30,
        max_clues=5,
        num_predicates=8,
        num_solver_steps=6,
        use_context_encoder=use_context_encoder,  # KEY PARAMETER
        use_dsc=True,
        use_msre=True,
        use_lcr=False,
        use_sph=False,
        dropout=0.1,
    )
    
    model = RLAN(config=config).to(device)
    model.train()
    
    # Optimizer - stable settings  
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=5e-4,
        weight_decay=0.01
    )
    
    # Loss function - stable settings (same as test_expansion_task.py)
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
        
        # Forward pass - WITH or WITHOUT context depending on flag
        if use_context_encoder:
            outputs = model(
                test_inputs,
                train_inputs=train_inputs,
                train_outputs=train_outputs,
                pair_mask=pair_mask,
                return_intermediates=True
            )
        else:
            # Same as original test - no context
            outputs = model(test_inputs, return_intermediates=True)
        
        logits = outputs['logits']
        attention_maps = outputs['attention_maps']
        stop_logits = outputs['stop_logits']
        predicates = outputs['predicates']
        
        loss_dict = loss_fn(
            logits, test_outputs, 
            attention_maps, stop_logits, predicates
        )
        loss = loss_dict['total_loss']
        
        if torch.isnan(loss):
            print(f"  Epoch {epoch+1}: NaN detected!")
            return False, best_grid_acc
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        pixel_acc, grid_acc, full_match, batch_size = compute_accuracy(logits, test_outputs)
        best_grid_acc = max(best_grid_acc, grid_acc)
        
        if epoch % 10 == 0 or grid_acc == 1.0:
            print(f"  Epoch {epoch+1}: loss={loss.item():.4f}, pixel_acc={pixel_acc*100:.1f}%, grid_acc={grid_acc*100:.1f}% ({full_match}/{batch_size})")
        
        if grid_acc >= target_acc:
            print(f"\n*** REACHED {grid_acc*100:.0f}% EXACT GRID MATCH at epoch {epoch+1} ***")
            return True, grid_acc
    
    print(f"\nDid not reach {target_acc*100:.0f}% in {max_epochs} epochs. Best: {best_grid_acc*100:.1f}%")
    return False, best_grid_acc


if __name__ == '__main__':
    import sys
    
    # Test tasks - known to work with the stable config
    expansion_tasks = [
        ('f5b8619d', '6->12, 4x expansion'),  # Moderate
        ('b91ae062', '3->12, 14x expansion'),  # High expansion
        ('007bbfb7', '3->9, 8x expansion (tiling)'),  # The problematic one
    ]
    
    # Take task index from command line, default to 0
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    task_id, desc = expansion_tasks[idx]
    
    print(f"\n{'#'*70}")
    print(f"# COMPARING: {task_id} - {desc}")
    print(f"{'#'*70}")
    
    # Test WITHOUT ContextEncoder (original behavior)
    print("\n" + "="*70)
    print("TEST 1: WITHOUT ContextEncoder (original test)")
    print("="*70)
    success_no_ctx, acc_no_ctx = train_on_task(task_id, use_context_encoder=False, max_epochs=150)
    
    # Test WITH ContextEncoder
    print("\n" + "="*70)
    print("TEST 2: WITH ContextEncoder (production setting)")
    print("="*70)
    success_with_ctx, acc_with_ctx = train_on_task(task_id, use_context_encoder=True, max_epochs=150)
    
    # Summary
    print("\n" + "#"*70)
    print("# COMPARISON SUMMARY")
    print("#"*70)
    print(f"Task: {task_id} - {desc}")
    print(f"  WITHOUT ContextEncoder: {'SUCCESS' if success_no_ctx else 'FAILED'} (best={acc_no_ctx*100:.1f}%)")
    print(f"  WITH ContextEncoder:    {'SUCCESS' if success_with_ctx else 'FAILED'} (best={acc_with_ctx*100:.1f}%)")
    
    if acc_with_ctx > acc_no_ctx:
        print(f"\n  >>> ContextEncoder HELPED learning (+{(acc_with_ctx-acc_no_ctx)*100:.1f}%)")
    elif acc_with_ctx < acc_no_ctx:
        print(f"\n  >>> ContextEncoder HURT learning (-{(acc_no_ctx-acc_with_ctx)*100:.1f}%)")
    else:
        print(f"\n  >>> ContextEncoder had NO EFFECT")
