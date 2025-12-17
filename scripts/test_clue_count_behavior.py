#!/usr/bin/env python3
"""
Test clue count behavior on grids with different complexity levels.

This script verifies that:
1. Clue count varies per-sample (not just batch average)
2. More complex grids should use more clues
3. The gradient signal correctly teaches clue count

Philosophy:
- Clue count is a LATENT variable - model learns it from task loss
- Simple grids: 1-2 clues should suffice
- Complex grids: 4-5 clues needed
- The stop_predictor learns this implicitly through gradient flow
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sci_arc.models.rlan import RLAN, RLANConfig


def create_simple_grid():
    """Simple grid: mostly background, one color region."""
    grid = torch.zeros(1, 6, 6, dtype=torch.long)  # (B, H, W)
    # Just one small region of color 1
    grid[0, 2:4, 2:4] = 1
    return grid


def create_medium_grid():
    """Medium complexity: 2-3 distinct color regions."""
    grid = torch.zeros(1, 6, 6, dtype=torch.long)  # (B, H, W)
    # Region 1: color 1 (top-left)
    grid[0, 0:2, 0:2] = 1
    # Region 2: color 2 (center)
    grid[0, 2:4, 2:4] = 2
    # Region 3: color 3 (bottom-right)
    grid[0, 4:6, 4:6] = 3
    return grid


def create_complex_grid():
    """Complex grid: many colors, scattered pattern."""
    grid = torch.zeros(1, 6, 6, dtype=torch.long)  # (B, H, W)
    # 5 different color regions scattered
    grid[0, 0, 0] = 1
    grid[0, 0, 5] = 2
    grid[0, 2, 2:4] = 3
    grid[0, 3, 1] = 4
    grid[0, 5, 0] = 5
    grid[0, 5, 5] = 6
    grid[0, 1, 3] = 7
    return grid


def create_checkerboard():
    """Very complex: alternating pattern requiring multiple clues."""
    grid = torch.zeros(1, 6, 6, dtype=torch.long)  # (B, H, W)
    for i in range(6):
        for j in range(6):
            if (i + j) % 2 == 0:
                grid[0, i, j] = 1
            else:
                grid[0, i, j] = 2
    return grid


def analyze_clue_usage(model, grid, name):
    """Analyze how many clues the model uses for a grid."""
    model.eval()
    with torch.no_grad():
        outputs = model(grid, return_intermediates=True)
        
        stop_logits = outputs["stop_logits"]  # (B, K)
        stop_probs = torch.sigmoid(stop_logits)  # (B, K)
        
        # Expected clues used = sum of (1 - stop_prob)
        clues_used = (1 - stop_probs).sum(dim=-1)  # (B,)
        
        # Per-clue analysis
        print(f"\n{'='*60}")
        print(f"GRID: {name}")
        print(f"{'='*60}")
        print(f"Grid shape: {grid.shape}, unique colors: {grid.unique().tolist()}")
        print(f"\nPer-clue stop probabilities:")
        for k in range(stop_probs.shape[1]):
            prob = stop_probs[0, k].item()
            usage = 1 - prob
            bar = "#" * int(usage * 20) + "-" * int((1 - usage) * 20)
            print(f"  Clue {k+1}: stop_prob={prob:.3f}, usage={usage:.3f} [{bar}]")
        
        print(f"\nExpected clues used: {clues_used.item():.2f}")
        
        # Analyze attention maps
        attention_maps = outputs["attention_maps"]  # (B, K, H, W)
        print(f"\nAttention entropy per clue:")
        for k in range(attention_maps.shape[1]):
            attn = attention_maps[0, k].flatten()
            attn_clamped = attn.clamp(min=1e-10)
            entropy = -(attn_clamped * torch.log(attn_clamped)).sum().item()
            max_entropy = torch.log(torch.tensor(float(attn.numel()))).item()
            norm_entropy = entropy / max_entropy
            print(f"  Clue {k+1}: entropy={entropy:.3f} (normalized={norm_entropy:.3f})")
        
        return clues_used.item(), stop_probs[0].tolist()


def test_gradient_flow(model, grid, target, name):
    """Test that gradient flows from task loss to stop_predictor."""
    model.train()
    
    # Forward pass with intermediates
    outputs = model(grid, return_intermediates=True)
    logits = outputs["logits"]  # (B, T, C, H, W) or similar
    stop_logits = outputs["stop_logits"]  # (B, K)
    
    # Simple task loss (cross-entropy)
    # Flatten for loss computation
    if logits.dim() == 5:  # (B, T, C, H, W)
        logits_final = logits[:, -1]  # Use final step
    else:
        logits_final = logits
    
    B, C, H, W = logits_final.shape
    logits_flat = logits_final.permute(0, 2, 3, 1).reshape(-1, C)
    target_flat = target.reshape(-1)  # target is (B, H, W)
    
    task_loss = nn.functional.cross_entropy(logits_flat, target_flat)
    
    # Backward pass
    model.zero_grad()
    task_loss.backward()
    
    # Check gradient on stop_predictor
    print(f"\n{'='*60}")
    print(f"GRADIENT FLOW TEST: {name}")
    print(f"{'='*60}")
    print(f"Task loss: {task_loss.item():.4f}")
    
    # Find stop_predictor parameters
    stop_predictor_grad_norm = 0.0
    for name_param, param in model.named_parameters():
        if "stop_predictor" in name_param and param.grad is not None:
            stop_predictor_grad_norm += param.grad.norm().item() ** 2
    stop_predictor_grad_norm = stop_predictor_grad_norm ** 0.5
    
    print(f"Stop predictor gradient norm: {stop_predictor_grad_norm:.6f}")
    
    if stop_predictor_grad_norm > 1e-6:
        print("[OK] Gradient is flowing to stop_predictor!")
    else:
        print("[WARN] No gradient reaching stop_predictor!")
    
    return stop_predictor_grad_norm


def main():
    print("=" * 70)
    print("CLUE COUNT BEHAVIOR TEST")
    print("=" * 70)
    
    # Create model - use config= keyword argument
    config = RLANConfig(
        hidden_dim=64,  # Smaller for testing
        max_clues=5,
        num_solver_steps=3,
    )
    model = RLAN(config=config)  # Pass as keyword argument
    model.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Test different complexity grids
    grids = [
        (create_simple_grid(), "SIMPLE (1 color region)"),
        (create_medium_grid(), "MEDIUM (3 color regions)"),
        (create_complex_grid(), "COMPLEX (7 scattered colors)"),
        (create_checkerboard(), "CHECKERBOARD (alternating pattern)"),
    ]
    
    results = []
    for grid, name in grids:
        clues_used, stop_probs = analyze_clue_usage(model, grid, name)
        results.append((name, clues_used, stop_probs))
    
    # Summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY: CLUE USAGE BY COMPLEXITY")
    print("=" * 70)
    for name, clues_used, stop_probs in results:
        bar = "#" * int(clues_used * 10) + "-" * int((5 - clues_used) * 10)
        print(f"{name:40s}: {clues_used:.2f} clues [{bar}]")
    
    # Gradient flow test
    print("\n" + "=" * 70)
    print("GRADIENT FLOW VERIFICATION")
    print("=" * 70)
    
    model.train()
    for grid, name in grids[:2]:  # Just test first two
        target = grid.clone()  # Use input as target for simple test (B, H, W)
        test_gradient_flow(model, grid, target, name)
    
    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS & RECOMMENDATIONS")
    print("=" * 70)
    print("""
OBSERVATION (from untrained model):
- All grids likely use similar clue counts (model hasn't learned yet)
- Stop probabilities are near initial bias (-1.0 -> sigmoid ~ 0.27)
- Expected: ~3.6 clues used with random initialization

AFTER TRAINING:
- Simple grids SHOULD use fewer clues (~1-2)
- Complex grids SHOULD use more clues (~4-5)
- The variance in clue count is the signal of task-dependency

WHAT TO CHECK IN TRAINING LOGS:
1. Add per-sample stop_prob variance: std(stop_probs) across batch
2. Correlate clue count with task difficulty metrics
3. If variance is near 0, model is using same clues for all tasks (bad!)

POTENTIAL ISSUES IF CLUE COUNT IS UNIFORM:
1. ponder_weight too high -> over-regularized to use few clues
2. Attention not learning -> all clues look the same
3. Stop predictor collapsed -> always predicts same stop_prob
""")


if __name__ == "__main__":
    main()
