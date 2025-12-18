#!/usr/bin/env python3
"""
Clue Regularization Stability Test

Tests whether clue regularization settings help or hurt learning:
1. Compare training WITH vs WITHOUT clue regularization
2. Multiple random seeds for statistical validity
3. Uses actual RLAN codebase and production-like settings

Goal: Verify clue reg settings are stable before recommending for production.
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


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_expansion_task(scale: int = 2, grid_size: int = 3):
    """Create expansion task: output is input scaled by factor."""
    inp = np.random.randint(1, 6, (grid_size, grid_size))
    out = np.repeat(np.repeat(inp, scale, axis=0), scale, axis=1)
    return inp, out


def create_rotation_task(grid_size: int = 4):
    """Create 90-degree rotation task."""
    inp = np.random.randint(1, 6, (grid_size, grid_size))
    out = np.rot90(inp, k=1)
    return inp, out


def pad_grid(grid: np.ndarray, size: int = 12, is_target: bool = False) -> np.ndarray:
    """Pad grid to fixed size."""
    h, w = grid.shape
    padded = np.full((size, size), -100 if is_target else 10, dtype=np.int64)
    padded[:h, :w] = grid
    return padded


def prepare_batch(task_fns, num_demos: int = 3, grid_size: int = 12, device='cpu'):
    """Prepare a batch with multiple tasks."""
    all_demos_in = []
    all_demos_out = []
    all_test_inputs = []
    all_test_targets = []
    
    for task_fn in task_fns:
        demos_in = []
        demos_out = []
        
        for _ in range(num_demos):
            inp, out = task_fn()
            demos_in.append(torch.from_numpy(pad_grid(inp, grid_size, False)))
            demos_out.append(torch.from_numpy(pad_grid(out, grid_size, True)))
        
        # Test sample
        test_in, test_out = task_fn()
        test_input = torch.from_numpy(pad_grid(test_in, grid_size, False))
        test_target = torch.from_numpy(pad_grid(test_out, grid_size, True))
        
        all_demos_in.append(demos_in)
        all_demos_out.append(demos_out)
        all_test_inputs.append(test_input)
        all_test_targets.append(test_target)
    
    demo_inputs = torch.stack([torch.stack(demos, dim=0) for demos in all_demos_in], dim=0).to(device)
    demo_outputs = torch.stack([torch.stack(demos, dim=0) for demos in all_demos_out], dim=0).to(device)
    test_inputs = torch.stack(all_test_inputs, dim=0).to(device)
    test_targets = torch.stack(all_test_targets, dim=0).to(device)
    
    return demo_inputs, demo_outputs, test_inputs, test_targets


def train_epoch(model, loss_fn, optimizer, demo_inputs, demo_outputs, test_inputs, test_targets, epoch, max_epochs):
    """Train one epoch and return metrics."""
    model.train()
    optimizer.zero_grad()
    
    outputs = model(
        test_inputs,
        train_inputs=demo_inputs,
        train_outputs=demo_outputs,
        return_intermediates=True,
    )
    
    logits = outputs['logits']
    
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
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    # Compute accuracy
    with torch.no_grad():
        preds = logits.argmax(dim=1)
        valid_mask = test_targets != -100
        correct = ((preds == test_targets) & valid_mask).sum().float()
        total = valid_mask.sum().float()
        accuracy = (correct / total).item() if total > 0 else 0.0
        
        # Per-sample exact match
        B = test_targets.shape[0]
        exact_matches = 0
        for i in range(B):
            mask_i = valid_mask[i]
            if mask_i.sum() > 0:
                if ((preds[i] == test_targets[i]) & mask_i).all():
                    exact_matches += 1
    
    return {
        'loss': loss.item(),
        'accuracy': accuracy,
        'exact_matches': exact_matches,
        'total_samples': B,
        'clues_used': loss_dict.get('expected_clues_used', 0.0),
    }


def run_experiment(use_clue_reg: bool, seed: int, max_epochs: int = 150, device='cpu'):
    """Run a single experiment."""
    set_seed(seed)
    
    # Model config - SMALLER for faster testing
    config = RLANConfig(
        num_colors=10,
        hidden_dim=64,  # Reduced
        max_clues=4,    # Reduced
        num_predicates=8,  # Reduced
        num_solver_steps=2,  # Reduced
        max_grid_size=12,
        use_context_encoder=True,
    )
    model = RLAN(config=config).to(device)
    
    # Loss with or without clue regularization
    if use_clue_reg:
        loss_fn = RLANLoss(
            lambda_sparsity=0.5,
            lambda_entropy=0.01,
            lambda_predicate=0.01,
            lambda_deep_supervision=0.1,
            min_clues=2.5,
            min_clue_weight=5.0,
            ponder_weight=0.02,
            entropy_ponder_weight=0.02,
        ).to(device)
    else:
        loss_fn = RLANLoss(
            lambda_sparsity=0.5,
            lambda_entropy=0.01,
            lambda_predicate=0.01,
            lambda_deep_supervision=0.1,
            min_clues=2.0,
            min_clue_weight=0.0,  # OFF
            ponder_weight=0.0,    # OFF
            entropy_ponder_weight=0.0,  # OFF
        ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    
    # Create batch with 4 different tasks
    task_fns = [
        lambda: create_expansion_task(scale=2, grid_size=3),
        lambda: create_expansion_task(scale=2, grid_size=4),
        lambda: create_rotation_task(grid_size=4),
        lambda: create_rotation_task(grid_size=5),
    ]
    
    demo_inputs, demo_outputs, test_inputs, test_targets = prepare_batch(
        task_fns, num_demos=3, grid_size=12, device=device
    )
    
    # Train until convergence or max epochs
    epochs_to_100 = None
    final_accuracy = 0.0
    final_clues = 0.0
    
    for epoch in range(1, max_epochs + 1):
        metrics = train_epoch(
            model, loss_fn, optimizer,
            demo_inputs, demo_outputs, test_inputs, test_targets,
            epoch, max_epochs
        )
        
        final_accuracy = metrics['accuracy']
        final_clues = metrics['clues_used']
        
        if metrics['exact_matches'] == metrics['total_samples'] and epochs_to_100 is None:
            epochs_to_100 = epoch
            break
    
    return {
        'epochs_to_100': epochs_to_100,
        'final_accuracy': final_accuracy,
        'final_clues': final_clues,
        'converged': epochs_to_100 is not None,
    }


def main():
    print("=" * 70)
    print("CLUE REGULARIZATION STABILITY TEST")
    print("=" * 70)
    print()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    num_seeds = 3  # Reduced for faster testing
    max_epochs = 50  # Reduced for faster testing
    
    print(f"\nRunning {num_seeds} seeds per configuration...")
    print("-" * 70)
    
    # Test WITHOUT clue regularization
    print("\n[1] WITHOUT Clue Regularization (min_clue_weight=0, ponder_weight=0)")
    print("-" * 50)
    
    results_without = []
    for seed in range(num_seeds):
        result = run_experiment(use_clue_reg=False, seed=seed, max_epochs=max_epochs, device=device)
        results_without.append(result)
        status = f"Epoch {result['epochs_to_100']}" if result['converged'] else f"FAILED ({result['final_accuracy']*100:.1f}%)"
        print(f"  Seed {seed}: {status}")
    
    # Test WITH clue regularization
    print("\n[2] WITH Clue Regularization (min_clue_weight=5.0, ponder_weight=0.02)")
    print("-" * 50)
    
    results_with = []
    for seed in range(num_seeds):
        result = run_experiment(use_clue_reg=True, seed=seed, max_epochs=max_epochs, device=device)
        results_with.append(result)
        status = f"Epoch {result['epochs_to_100']}" if result['converged'] else f"FAILED ({result['final_accuracy']*100:.1f}%)"
        print(f"  Seed {seed}: {status}, clues={result['final_clues']:.1f}")
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    def summarize(results, name):
        converged = [r for r in results if r['converged']]
        conv_rate = len(converged) / len(results) * 100
        
        if converged:
            epochs = [r['epochs_to_100'] for r in converged]
            mean_epochs = np.mean(epochs)
            std_epochs = np.std(epochs)
            clues = [r['final_clues'] for r in converged]
            mean_clues = np.mean(clues)
        else:
            mean_epochs = float('inf')
            std_epochs = 0
            mean_clues = 0
        
        print(f"\n{name}:")
        print(f"  Convergence rate: {len(converged)}/{len(results)} ({conv_rate:.0f}%)")
        if converged:
            print(f"  Epochs to 100%: {mean_epochs:.1f} ± {std_epochs:.1f}")
            print(f"  Final clues: {mean_clues:.2f}")
        
        return {
            'conv_rate': conv_rate,
            'mean_epochs': mean_epochs,
            'std_epochs': std_epochs,
            'mean_clues': mean_clues,
        }
    
    stats_without = summarize(results_without, "WITHOUT Clue Reg")
    stats_with = summarize(results_with, "WITH Clue Reg")
    
    # Comparison
    print("\n" + "-" * 70)
    print("COMPARISON:")
    print("-" * 70)
    
    if stats_with['conv_rate'] >= stats_without['conv_rate']:
        print("✅ Clue regularization has SAME OR BETTER convergence rate")
    else:
        print("⚠️  Clue regularization has WORSE convergence rate")
    
    if stats_with['mean_epochs'] <= stats_without['mean_epochs'] * 1.2:  # Allow 20% slower
        print("✅ Clue regularization does NOT significantly slow training")
    else:
        print(f"⚠️  Clue regularization is {stats_with['mean_epochs']/stats_without['mean_epochs']:.1f}x slower")
    
    if stats_with['std_epochs'] <= stats_without['std_epochs'] * 1.5:  # Allow 50% more variance
        print("✅ Clue regularization is STABLE (similar variance)")
    else:
        print("⚠️  Clue regularization increases training variance")
    
    print()
    print("=" * 70)
    if stats_with['conv_rate'] >= 80 and stats_with['mean_epochs'] <= stats_without['mean_epochs'] * 1.5:
        print("✅ RECOMMENDATION: Clue regularization is SAFE for production")
        print("   Settings: min_clue_weight=5.0, ponder_weight=0.02, entropy_ponder_weight=0.02")
    else:
        print("⚠️  RECOMMENDATION: Keep clue regularization OFF until further testing")
    print("=" * 70)


if __name__ == '__main__':
    main()
