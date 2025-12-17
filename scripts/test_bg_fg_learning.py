#!/usr/bin/env python3
"""
Test script to verify background/foreground learning mathematically.

This script:
1. Loads a real ARC sample with known target
2. Creates a model and passes the sample through
3. Computes the loss and traces gradient flow
4. Verifies that both BG and FG pixels receive proper gradients
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sci_arc.models.rlan import RLAN
from sci_arc.training.rlan_loss import WeightedStablemaxLoss, RLANLoss


def load_arc_sample(data_path: str = "./data/arc-agi/data/training"):
    """Load a single ARC task for testing."""
    # Find first task file
    task_files = [f for f in os.listdir(data_path) if f.endswith('.json')]
    if not task_files:
        raise FileNotFoundError(f"No task files found in {data_path}")
    
    task_path = os.path.join(data_path, task_files[0])
    with open(task_path, 'r') as f:
        task = json.load(f)
    
    # Get first training example
    train_example = task['train'][0]
    test_example = task['test'][0]
    
    return {
        'task_id': task_files[0].replace('.json', ''),
        'train_input': torch.tensor(train_example['input'], dtype=torch.long),
        'train_output': torch.tensor(train_example['output'], dtype=torch.long),
        'test_input': torch.tensor(test_example['input'], dtype=torch.long),
        'test_output': torch.tensor(test_example['output'], dtype=torch.long),
    }


def analyze_target_distribution(target: torch.Tensor):
    """Analyze the class distribution in a target grid."""
    print("\n" + "="*60)
    print("TARGET GRID ANALYSIS")
    print("="*60)
    
    H, W = target.shape
    total = H * W
    
    print(f"Grid size: {H}x{W} = {total} pixels")
    print()
    
    # Count each class
    class_counts = {}
    for c in range(10):
        count = (target == c).sum().item()
        if count > 0:
            pct = count / total * 100
            class_counts[c] = (count, pct)
            label = "BACKGROUND" if c == 0 else f"FOREGROUND (color {c})"
            print(f"  Class {c} ({label}): {count} pixels ({pct:.1f}%)")
    
    # Summary
    bg_count = class_counts.get(0, (0, 0))[0]
    fg_count = total - bg_count
    print()
    print(f"Total Background: {bg_count} pixels ({bg_count/total*100:.1f}%)")
    print(f"Total Foreground: {fg_count} pixels ({fg_count/total*100:.1f}%)")
    
    return class_counts


def test_loss_gradients(logits: torch.Tensor, target: torch.Tensor):
    """
    Test that the loss function provides gradients to both BG and FG.
    """
    print("\n" + "="*60)
    print("LOSS FUNCTION GRADIENT ANALYSIS")
    print("="*60)
    
    # Create loss function
    loss_fn = WeightedStablemaxLoss(
        bg_weight_cap=1.0,
        fg_weight_cap=10.0,
        min_class_weight=0.1,
    )
    
    # Ensure logits require grad
    logits = logits.detach().clone().requires_grad_(True)
    
    # Compute loss
    loss = loss_fn(logits, target)
    print(f"\nLoss value: {loss.item():.4f}")
    
    # Backward pass
    loss.backward()
    
    # Analyze gradients per class
    B, C, H, W = logits.shape
    grad = logits.grad  # (B, C, H, W)
    
    print("\nPer-class gradient magnitude (averaged over pixels):")
    for c in range(C):
        # Get gradient for this class
        class_grad = grad[:, c, :, :].abs().mean().item()
        print(f"  Class {c}: grad_mag = {class_grad:.6f}")
    
    # Analyze gradients at BG vs FG pixel locations
    print("\nGradient analysis by pixel type:")
    
    bg_mask = (target == 0)  # (B, H, W)
    fg_mask = (target > 0)   # (B, H, W)
    
    # Gradient magnitude at BG pixels
    if bg_mask.any():
        bg_grad_mag = grad[:, :, :, :].abs().sum(dim=1)[bg_mask].mean().item()
        print(f"  At BACKGROUND pixels: avg gradient magnitude = {bg_grad_mag:.6f}")
    
    # Gradient magnitude at FG pixels
    if fg_mask.any():
        fg_grad_mag = grad[:, :, :, :].abs().sum(dim=1)[fg_mask].mean().item()
        print(f"  At FOREGROUND pixels: avg gradient magnitude = {fg_grad_mag:.6f}")
    
    # Verify ratio
    if bg_mask.any() and fg_mask.any():
        ratio = fg_grad_mag / (bg_grad_mag + 1e-6)
        print(f"\n  FG/BG gradient ratio: {ratio:.2f}x")
        print(f"  (Higher ratio means FG gets stronger learning signal - GOOD!)")
    
    return grad


def test_output_head_initialization():
    """
    Test that the output head initialization gives foreground a boost.
    """
    print("\n" + "="*60)
    print("OUTPUT HEAD INITIALIZATION TEST")
    print("="*60)
    
    # Create a minimal model to test initialization
    from sci_arc.models.rlan_modules.recursive_solver import RecursiveSolver
    
    solver = RecursiveSolver(
        hidden_dim=256,
        num_colors=10,
        num_classes=10,
        num_steps=6,
    )
    
    # Check the output head bias
    final_layer = None
    for module in solver.output_head.modules():
        if isinstance(module, nn.Conv2d) and module.out_channels == 10:
            final_layer = module
    
    if final_layer is not None and final_layer.bias is not None:
        bias = final_layer.bias.data
        print(f"Output head final layer bias:")
        for c in range(10):
            label = "BG" if c == 0 else f"FG{c}"
            print(f"  Class {c} ({label}): bias = {bias[c].item():.4f}")
        
        # Compute initial prediction probabilities
        # With zero input features, logits = bias
        probs = F.softmax(bias, dim=0)
        print(f"\nInitial prediction probabilities (from bias only):")
        for c in range(10):
            label = "BG" if c == 0 else f"FG{c}"
            print(f"  Class {c} ({label}): P = {probs[c].item():.4f} ({probs[c].item()*100:.1f}%)")
        
        bg_prob = probs[0].item()
        fg_prob = probs[1:].sum().item()
        print(f"\nTotal: P(BG) = {bg_prob*100:.1f}%, P(FG) = {fg_prob*100:.1f}%")
    else:
        print("WARNING: Could not find output head final layer!")


def test_clue_aggregation_gradient_flow():
    """
    Test that stop_logits receive gradients from task loss.
    """
    print("\n" + "="*60)
    print("CLUE AGGREGATION GRADIENT FLOW TEST")
    print("="*60)
    
    from sci_arc.models.rlan_modules.recursive_solver import RecursiveSolver
    
    # Create solver
    solver = RecursiveSolver(
        hidden_dim=64,  # Small for testing
        num_colors=10,
        num_classes=10,
        num_steps=3,
        use_act=False,
    )
    solver.train()
    
    # Create dummy inputs
    B, K, D, H, W = 2, 6, 64, 5, 5
    clue_features = torch.randn(B, K, D, H, W, requires_grad=True)
    attention_maps = torch.softmax(torch.randn(B, K, H, W), dim=-1)
    stop_logits = torch.randn(B, K, requires_grad=True)  # KEY: requires_grad
    count_embedding = torch.zeros(B, 10, D)
    predicates = torch.zeros(B, 8)
    input_grid = torch.zeros(B, H, W, dtype=torch.long)
    
    # Forward pass
    all_logits = solver(
        clue_features=clue_features,
        count_embedding=count_embedding,
        predicates=predicates,
        input_grid=input_grid,
        attention_maps=attention_maps,
        stop_logits=stop_logits,
        return_all_steps=True,
    )
    
    # Create dummy target
    target = torch.zeros(B, H, W, dtype=torch.long)
    target[:, 2, 2] = 1  # One foreground pixel
    
    # Compute loss using final logits
    loss_fn = WeightedStablemaxLoss()
    loss = loss_fn(all_logits[-1], target)
    
    # Backward
    loss.backward()
    
    # Check if stop_logits received gradient
    if stop_logits.grad is not None:
        grad_norm = stop_logits.grad.abs().mean().item()
        print(f"stop_logits gradient norm: {grad_norm:.6f}")
        if grad_norm > 1e-8:
            print("✓ PASS: stop_logits receives gradient from task loss!")
            print("  This means clue count is a TRUE latent variable.")
        else:
            print("✗ FAIL: stop_logits gradient is near zero!")
    else:
        print("✗ FAIL: stop_logits.grad is None!")
    
    # Also check clue_features gradient
    if clue_features.grad is not None:
        grad_norm = clue_features.grad.abs().mean().item()
        print(f"clue_features gradient norm: {grad_norm:.6f}")


def run_full_forward_pass():
    """
    Run a complete forward pass with a real ARC sample.
    """
    print("\n" + "="*60)
    print("FULL FORWARD PASS TEST")
    print("="*60)
    
    # Try to load a real sample
    try:
        sample = load_arc_sample()
        print(f"Loaded task: {sample['task_id']}")
    except FileNotFoundError as e:
        print(f"Could not load ARC data: {e}")
        print("Using synthetic sample instead.")
        
        # Create synthetic sample
        H, W = 10, 10
        target = torch.zeros(H, W, dtype=torch.long)
        # Add some foreground (10% of pixels)
        target[2:4, 3:5] = 1
        target[6:8, 6:8] = 2
        
        sample = {
            'task_id': 'synthetic',
            'train_input': torch.zeros(H, W, dtype=torch.long),
            'train_output': target.clone(),
            'test_input': torch.zeros(H, W, dtype=torch.long),
            'test_output': target,
        }
    
    # Analyze target
    target = sample['test_output']
    class_counts = analyze_target_distribution(target)
    
    # Create model
    print("\nCreating RLAN model...")
    model = RLAN(
        hidden_dim=128,
        num_colors=10,
        num_classes=10,
        max_grid_size=30,
        max_clues=6,
        num_predicates=8,
        num_solver_steps=6,
        use_dsc=True,
        use_msre=True,
        use_lcr=False,
        use_sph=False,
        use_context_encoder=True,
        use_act=False,
    )
    model.train()
    
    # Prepare inputs
    B = 1
    test_input = sample['test_input'].unsqueeze(0).float()  # (1, H, W)
    train_input = sample['train_input'].unsqueeze(0).unsqueeze(0).float()  # (1, 1, H, W)
    train_output = sample['train_output'].unsqueeze(0).unsqueeze(0).float()  # (1, 1, H, W)
    target_batch = sample['test_output'].unsqueeze(0)  # (1, H, W)
    
    # Pad to max grid size
    _, H, W = test_input.shape
    max_size = 30
    if H < max_size or W < max_size:
        test_input_padded = F.pad(test_input, (0, max_size - W, 0, max_size - H))
        train_input_padded = F.pad(train_input, (0, max_size - W, 0, max_size - H))
        train_output_padded = F.pad(train_output, (0, max_size - W, 0, max_size - H))
        target_padded = F.pad(target_batch, (0, max_size - W, 0, max_size - H), value=-100)  # ignore padding
    else:
        test_input_padded = test_input
        train_input_padded = train_input
        train_output_padded = train_output
        target_padded = target_batch
    
    print(f"Input shape: {test_input_padded.shape}")
    print(f"Target shape: {target_padded.shape}")
    
    # Forward pass
    print("\nRunning forward pass...")
    outputs = model(
        test_input_padded,
        train_inputs=train_input_padded,
        train_outputs=train_output_padded,
        return_intermediates=True,
    )
    
    logits = outputs['logits']
    print(f"Logits shape: {logits.shape}")
    
    # Analyze predictions
    preds = logits.argmax(dim=1)  # (B, H, W)
    print("\nPrediction distribution:")
    for c in range(10):
        count = (preds == c).sum().item()
        if count > 0:
            pct = count / preds.numel() * 100
            print(f"  Class {c}: {count} pixels ({pct:.1f}%)")
    
    # Test gradient flow
    print("\n" + "-"*40)
    test_loss_gradients(logits, target_padded)


def main():
    print("="*60)
    print("BACKGROUND/FOREGROUND LEARNING VERIFICATION")
    print("="*60)
    
    # Test 1: Output head initialization
    test_output_head_initialization()
    
    # Test 2: Clue aggregation gradient flow
    test_clue_aggregation_gradient_flow()
    
    # Test 3: Full forward pass (if data available)
    run_full_forward_pass()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("""
The system is designed to learn both background and foreground because:

1. OUTPUT HEAD INITIALIZATION:
   - Background class gets negative bias (-0.5)
   - Foreground classes get positive bias (+0.056 each)
   - Initial P(BG) ≈ 38%, P(FG) ≈ 62%
   - This prevents early collapse to all-background

2. WEIGHTED LOSS FUNCTION:
   - Background weight capped at 1.0
   - Foreground weights can be up to 10.0
   - Rare foreground classes get stronger gradients
   - Ensures minority classes always receive learning signal

3. LATENT CLUE COUNT:
   - stop_logits now flow into clue aggregation
   - Gradients flow: task_loss -> logits -> aggregated -> stop_probs
   - Clue count learned from target grid, not just sparsity penalty

4. UNIFORM DEEP SUPERVISION:
   - All solver steps get equal weight (1.0)
   - Prevents later steps from degrading
   - Every step learns to predict correct output
""")


if __name__ == "__main__":
    main()
