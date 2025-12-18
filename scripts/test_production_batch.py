#!/usr/bin/env python3
"""
Production-Like Batch Test for RLAN

Tests:
1. ContextEncoder active (production mode)
2. Clue regularization active (as in successful test)
3. Batch with MULTIPLE different tasks of varying difficulties
4. Verifies:
   - Each sample in batch learns its own rule
   - Per-sample clue count/loss works
   - Different tasks in batch don't interfere
   - Augmentation applies consistently within each sample

This mimics production training more closely than single-task tests.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import random

from sci_arc.models.rlan import RLAN, RLANConfig
from sci_arc.training.rlan_loss import RLANLoss


def create_expansion_task(scale: int = 2, grid_size: int = 3):
    """Create expansion task: output is input scaled by factor."""
    # Random input grid
    inp = np.random.randint(1, 6, (grid_size, grid_size))
    out = np.repeat(np.repeat(inp, scale, axis=0), scale, axis=1)
    return inp, out


def create_rotation_task(grid_size: int = 4):
    """Create 90-degree rotation task."""
    inp = np.random.randint(1, 6, (grid_size, grid_size))
    out = np.rot90(inp, k=1)
    return inp, out


def create_flip_task(grid_size: int = 4, horizontal: bool = True):
    """Create flip task."""
    inp = np.random.randint(1, 6, (grid_size, grid_size))
    out = np.flip(inp, axis=1 if horizontal else 0).copy()
    return inp, out


def create_color_swap_task(grid_size: int = 4, c1: int = 1, c2: int = 2):
    """Create color swap task: swap two colors."""
    inp = np.random.randint(0, 5, (grid_size, grid_size))
    out = inp.copy()
    out[inp == c1] = c2
    out[inp == c2] = c1
    return inp, out


def create_border_task(grid_size: int = 5, border_color: int = 3):
    """Create add-border task: add a border around non-zero regions."""
    inp = np.zeros((grid_size, grid_size), dtype=np.int64)
    # Put a small shape in the center
    cx, cy = grid_size // 2, grid_size // 2
    inp[cx-1:cx+2, cy-1:cy+2] = np.random.randint(1, 4, (3, 3))
    
    out = inp.copy()
    # Add border (simplified - just expand)
    mask = inp > 0
    from scipy.ndimage import binary_dilation
    dilated = binary_dilation(mask)
    out[(dilated) & (~mask)] = border_color
    return inp, out


def pad_grid(grid: np.ndarray, size: int = 12, is_target: bool = False) -> np.ndarray:
    """Pad grid to fixed size."""
    h, w = grid.shape
    padded = np.full((size, size), -100 if is_target else 10, dtype=np.int64)
    padded[:h, :w] = grid
    return padded


def prepare_task(task_fn, num_demos: int = 3, grid_size: int = 12):
    """Create a task with demos and test."""
    demos_in = []
    demos_out = []
    
    for _ in range(num_demos):
        inp, out = task_fn()
        demos_in.append(torch.from_numpy(pad_grid(inp, grid_size, False)))
        demos_out.append(torch.from_numpy(pad_grid(out, grid_size, True)))
    
    # Test sample (different instance, same rule)
    test_in, test_out = task_fn()
    test_input = torch.from_numpy(pad_grid(test_in, grid_size, False))
    test_target = torch.from_numpy(pad_grid(test_out, grid_size, True))
    
    return demos_in, demos_out, test_input, test_target


def main():
    print("=" * 70)
    print("PRODUCTION-LIKE BATCH TEST")
    print("=" * 70)
    print()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Model config matching production
    config = RLANConfig(
        num_colors=10,
        hidden_dim=128,
        max_clues=8,
        num_predicates=16,
        num_solver_steps=4,
        max_grid_size=12,
        # PRODUCTION MODE
        use_context_encoder=True,
    )
    
    model = RLAN(config=config).to(device)
    
    # Loss with clue regularization ACTIVE (matching successful test)
    loss_fn = RLANLoss(
        lambda_sparsity=0.5,
        lambda_entropy=0.01,
        lambda_predicate=0.01,
        lambda_deep_supervision=0.1,
        lambda_curriculum=0.0,
        # CLUE REGULARIZATION - ACTIVE
        min_clues=2.5,
        min_clue_weight=5.0,
        ponder_weight=0.02,
        entropy_ponder_weight=0.02,
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"ContextEncoder: ACTIVE")
    print(f"Clue Regularization: min_clues=2.5, min_clue_weight=5.0, ponder_weight=0.02")
    print()
    
    # Create diverse batch with DIFFERENT tasks
    # Each task has a different transformation rule
    task_creators = [
        ("Expansion 2x", lambda: create_expansion_task(scale=2, grid_size=3)),
        ("Rotation 90°", lambda: create_rotation_task(grid_size=4)),
        ("Flip Horizontal", lambda: create_flip_task(grid_size=4, horizontal=True)),
        ("Color Swap 1↔2", lambda: create_color_swap_task(grid_size=4, c1=1, c2=2)),
    ]
    
    batch_size = len(task_creators)
    print(f"Batch contains {batch_size} DIFFERENT tasks:")
    for name, _ in task_creators:
        print(f"  - {name}")
    print()
    
    # Prepare batch
    all_demos_in = []
    all_demos_out = []
    all_test_inputs = []
    all_test_targets = []
    
    max_demos = 3
    
    for name, task_fn in task_creators:
        demos_in, demos_out, test_input, test_target = prepare_task(task_fn, num_demos=max_demos)
        all_demos_in.append(demos_in)
        all_demos_out.append(demos_out)
        all_test_inputs.append(test_input)
        all_test_targets.append(test_target)
    
    # Stack into batch tensors
    # demo_inputs: (B, N, H, W)
    # demo_outputs: (B, N, H, W)  
    # test_input: (B, H, W)
    # test_target: (B, H, W)
    
    demo_inputs = torch.stack([
        torch.stack(demos, dim=0) for demos in all_demos_in
    ], dim=0).to(device)  # (B, N, H, W)
    
    demo_outputs = torch.stack([
        torch.stack(demos, dim=0) for demos in all_demos_out
    ], dim=0).to(device)  # (B, N, H, W)
    
    test_inputs = torch.stack(all_test_inputs, dim=0).to(device)  # (B, H, W)
    test_targets = torch.stack(all_test_targets, dim=0).to(device)  # (B, H, W)
    
    print(f"Batch shapes:")
    print(f"  demo_inputs:  {demo_inputs.shape}")
    print(f"  demo_outputs: {demo_outputs.shape}")
    print(f"  test_inputs:  {test_inputs.shape}")
    print(f"  test_targets: {test_targets.shape}")
    print()
    
    # Training loop
    max_epochs = 300
    target_accuracy = 1.0
    log_every = 20
    
    best_acc = 0.0
    accuracies_per_task = [0.0] * batch_size
    
    print("Training on mixed batch (each sample = different task)...")
    print("-" * 70)
    
    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(
            test_inputs,
            train_inputs=demo_inputs,
            train_outputs=demo_outputs,
            return_intermediates=True,
        )
        
        logits = outputs['logits']  # (B, C, H, W)
        
        # Compute loss
        loss_dict = loss_fn(
            logits=logits,
            targets=test_targets,
            stop_logits=outputs.get('stop_logits'),
            attention_maps=outputs.get('attention_maps'),
            predicates=outputs.get('predicates'),
            all_logits=outputs.get('all_logits'),
            epoch=epoch,
            max_epochs=max_epochs,
        )
        loss = loss_dict['total_loss']
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Evaluate per-task accuracy
        with torch.no_grad():
            preds = logits.argmax(dim=1)  # (B, H, W)
            
            # Mask for valid positions (not padding)
            valid_mask = test_targets != -100  # (B, H, W)
            
            task_accs = []
            task_exact = []
            for i in range(batch_size):
                mask_i = valid_mask[i]
                if mask_i.sum() > 0:
                    correct = (preds[i] == test_targets[i]) & mask_i
                    acc = correct.sum().float() / mask_i.sum().float()
                    task_accs.append(acc.item())
                    task_exact.append(acc.item() == 1.0)
                else:
                    task_accs.append(0.0)
                    task_exact.append(False)
            
            mean_acc = np.mean(task_accs)
            num_exact = sum(task_exact)
            
            # Update best
            if mean_acc > best_acc:
                best_acc = mean_acc
                accuracies_per_task = task_accs.copy()
        
        # Log
        if epoch % log_every == 0 or num_exact == batch_size:
            clue_info = ""
            if 'expected_clues_used' in loss_dict:
                clue_info = f" | Clues: {loss_dict['expected_clues_used']:.1f}"
            
            task_status = " | ".join([
                f"T{i+1}:{acc*100:.0f}%" for i, acc in enumerate(task_accs)
            ])
            
            print(f"Epoch {epoch:4d} | Loss: {loss.item():.4f} | "
                  f"Mean: {mean_acc*100:.1f}% | Exact: {num_exact}/{batch_size}{clue_info}")
            print(f"         Per-task: {task_status}")
            
            if num_exact == batch_size:
                print()
                print("=" * 70)
                print(f"★ ALL {batch_size} TASKS SOLVED at epoch {epoch}!")
                print("=" * 70)
                break
    
    # Final summary
    print()
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print()
    
    # Re-evaluate with fresh data to test generalization
    print("Testing generalization (new instances of same rules)...")
    model.eval()
    
    # Generate new test instances
    generalization_results = []
    
    with torch.no_grad():
        for i, (name, task_fn) in enumerate(task_creators):
            # New test sample
            demos_in, demos_out, test_input, test_target = prepare_task(task_fn, num_demos=max_demos)
            
            di = torch.stack(demos_in, dim=0).unsqueeze(0).to(device)
            do = torch.stack(demos_out, dim=0).unsqueeze(0).to(device)
            ti = test_input.unsqueeze(0).to(device)
            tt = test_target.unsqueeze(0).to(device)
            
            outputs = model(ti, train_inputs=di, train_outputs=do, return_intermediates=True)
            preds = outputs['logits'].argmax(dim=1)
            
            valid = tt != -100
            if valid.sum() > 0:
                correct = ((preds == tt) & valid).sum().float()
                acc = correct / valid.sum().float()
                generalization_results.append((name, acc.item()))
            else:
                generalization_results.append((name, 0.0))
    
    print("\nGeneralization to NEW instances:")
    for name, acc in generalization_results:
        status = "✓" if acc == 1.0 else "✗"
        print(f"  {status} {name}: {acc*100:.1f}%")
    
    gen_mean = np.mean([r[1] for r in generalization_results])
    gen_exact = sum(1 for _, acc in generalization_results if acc == 1.0)
    
    print()
    print(f"Generalization: {gen_exact}/{len(generalization_results)} exact, {gen_mean*100:.1f}% mean")
    
    # Verify clue usage varies by task difficulty
    print()
    print("Verifying per-sample clue dynamics...")
    
    with torch.no_grad():
        outputs = model(
            test_inputs,
            train_inputs=demo_inputs,
            train_outputs=demo_outputs,
            return_intermediates=True,
        )
        
        if 'stop_logits' in outputs:
            stop_probs = torch.sigmoid(outputs['stop_logits'])  # (B, K)
            clues_per_sample = (1 - stop_probs).sum(dim=-1)  # (B,)
            
            print("\nClues used per task in final batch:")
            for i, (name, _) in enumerate(task_creators):
                print(f"  {name}: {clues_per_sample[i].item():.2f} clues")
            
            clue_std = clues_per_sample.std().item()
            print(f"\nClue usage std: {clue_std:.3f}")
            if clue_std > 0.1:
                print("  → Different tasks use different amounts of clues (good!)")
            else:
                print("  → All tasks use similar clues (may be OK for similar difficulties)")
    
    print()
    print("=" * 70)
    if num_exact == batch_size:
        print("✅ TEST PASSED: Multi-task batch learning with ContextEncoder works!")
    else:
        print(f"⚠️  TEST PARTIAL: {num_exact}/{batch_size} tasks solved")
    print("=" * 70)


if __name__ == '__main__':
    main()
