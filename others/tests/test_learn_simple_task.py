"""
Test: Can SCI-ARC learn a simple ARC-like transformation?

This test validates the core learning capability by training on a simple,
consistent task to see if:
1. Task loss decreases (model learns)
2. The model produces non-trivial outputs (not all zeros)
3. CISL losses behave as expected

If this test fails, the model architecture or training loop has fundamental issues.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple
import sys
sys.path.insert(0, 'c:/Users/perahmat/Downloads/SCI-ARC')

from sci_arc.models.sci_arc import SCIARC, SCIARCConfig
from sci_arc.training.cisl_loss import CISLLoss


def create_simple_task_batch(batch_size: int = 8, grid_size: int = 5) -> Dict[str, torch.Tensor]:
    """
    Create a batch of simple "flip vertical" tasks.
    
    Rule: output = input flipped vertically
    This is a pure STRUCTURAL transformation (no color dependency).
    """
    # Create random input grids (1-3 colors, mostly background)
    input_grids = torch.zeros(batch_size, 2, grid_size, grid_size, dtype=torch.long)
    output_grids = torch.zeros(batch_size, 2, grid_size, grid_size, dtype=torch.long)
    
    for b in range(batch_size):
        for p in range(2):  # 2 demo pairs
            # Create simple pattern: random colored pixels
            grid = torch.zeros(grid_size, grid_size, dtype=torch.long)
            num_pixels = np.random.randint(2, 6)
            for _ in range(num_pixels):
                r, c = np.random.randint(0, grid_size, 2)
                color = np.random.randint(1, 5)  # Colors 1-4
                grid[r, c] = color
            
            input_grids[b, p] = grid
            output_grids[b, p] = torch.flip(grid, dims=[0])  # Vertical flip
    
    # Test pair (same rule)
    test_input = torch.zeros(batch_size, grid_size, grid_size, dtype=torch.long)
    test_output = torch.zeros(batch_size, grid_size, grid_size, dtype=torch.long)
    
    for b in range(batch_size):
        grid = torch.zeros(grid_size, grid_size, dtype=torch.long)
        num_pixels = np.random.randint(2, 6)
        for _ in range(num_pixels):
            r, c = np.random.randint(0, grid_size, 2)
            color = np.random.randint(1, 5)
            grid[r, c] = color
        test_input[b] = grid
        test_output[b] = torch.flip(grid, dims=[0])
    
    return {
        'input_grids': input_grids,
        'output_grids': output_grids,
        'test_inputs': test_input,
        'test_outputs': test_output,
        'task_id': ['simple_flip'] * batch_size,
        'num_train_pairs': 2,
    }


def test_simple_task_learning():
    """
    Test if SCI-ARC can learn a simple vertical flip task.
    
    Expected: Task loss should decrease over training steps.
    """
    print("\n" + "=" * 70)
    print("TEST: Simple Task Learning (Vertical Flip)")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create small model for quick testing
    config = SCIARCConfig(
        hidden_dim=64,
        num_colors=10,
        max_grid_size=10,
        num_structure_slots=4,
        se_layers=1,
        max_objects=4,
        num_heads=2,
        H_cycles=4,
        L_cycles=2,
        L_layers=1,
        latent_size=32,
    )
    
    model = SCIARC(config).to(device)
    model.train()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # CISL loss
    cisl_loss = CISLLoss(
        consist_weight=0.5,
        content_inv_weight=0.5,
        variance_weight=1.0,
        target_std=0.3,
    ).to(device)
    
    # Training loop
    num_steps = 50
    batch_size = 8
    
    task_losses = []
    cisl_losses = []
    
    print(f"\nTraining for {num_steps} steps...")
    
    for step in range(num_steps):
        optimizer.zero_grad()
        
        # Create batch
        batch = create_simple_task_batch(batch_size=batch_size, grid_size=5)
        
        # Move to device
        input_grids = batch['input_grids'].to(device)
        output_grids = batch['output_grids'].to(device)
        test_inputs = batch['test_inputs'].to(device)
        test_outputs = batch['test_outputs'].to(device)
        
        # Forward pass
        outputs = model(
            input_grids=input_grids,
            output_grids=output_grids,
            test_input=test_inputs,
            test_output=test_outputs,
        )
        
        # Task loss (cross-entropy)
        logits = outputs['logits']  # [B, H, W, C]
        B, H, W, C = logits.shape
        logits_flat = logits.view(B * H * W, C)
        targets_flat = test_outputs.view(B * H * W)
        task_loss = F.cross_entropy(logits_flat, targets_flat)
        
        # CISL loss
        z_struct = outputs['z_struct']
        z_struct_demos = outputs.get('z_struct_demos')
        cisl_result = cisl_loss(z_struct=z_struct, z_struct_demos=z_struct_demos)
        cisl_total = cisl_result['total']
        
        # Total loss
        total_loss = task_loss + 0.1 * cisl_total
        
        # Backward
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        task_losses.append(task_loss.item())
        cisl_losses.append(cisl_total.item())
        
        if step % 10 == 0 or step == num_steps - 1:
            # Check prediction quality
            with torch.no_grad():
                preds = logits.argmax(dim=-1)  # [B, H, W]
                correct = (preds == test_outputs).float().mean().item()
                bg_pred = (preds == 0).float().mean().item()
                
            print(f"  Step {step:3d}: task_loss={task_loss.item():.4f}, "
                  f"cisl={cisl_total.item():.4f}, "
                  f"acc={correct*100:.1f}%, bg_pred={bg_pred*100:.1f}%")
    
    # Analyze learning
    first_10_avg = np.mean(task_losses[:10])
    last_10_avg = np.mean(task_losses[-10:])
    improvement = (first_10_avg - last_10_avg) / first_10_avg * 100
    
    print(f"\n[RESULTS]")
    print(f"  First 10 steps avg task loss: {first_10_avg:.4f}")
    print(f"  Last 10 steps avg task loss: {last_10_avg:.4f}")
    print(f"  Improvement: {improvement:.1f}%")
    
    # Check for background collapse
    with torch.no_grad():
        batch = create_simple_task_batch(batch_size=4, grid_size=5)
        input_grids = batch['input_grids'].to(device)
        output_grids = batch['output_grids'].to(device)
        test_inputs = batch['test_inputs'].to(device)
        test_outputs = batch['test_outputs'].to(device)
        
        outputs = model(
            input_grids=input_grids,
            output_grids=output_grids,
            test_input=test_inputs,
            test_output=test_outputs,
        )
        
        preds = outputs['logits'].argmax(dim=-1)
        
        print(f"\n[SAMPLE PREDICTION]")
        print(f"  Test Input:\n{test_inputs[0].cpu().numpy()}")
        print(f"  Expected Output:\n{test_outputs[0].cpu().numpy()}")
        print(f"  Predicted:\n{preds[0].cpu().numpy()}")
        
        # Count predictions by color
        unique, counts = torch.unique(preds, return_counts=True)
        print(f"\n  Prediction distribution:")
        for u, c in zip(unique.tolist(), counts.tolist()):
            print(f"    Color {u}: {c} pixels ({c/(4*25)*100:.1f}%)")
        
        bg_ratio = (preds == 0).float().mean().item()
        
    # Assertions
    print(f"\n[ASSERTIONS]")
    
    # 1. Task loss should decrease
    if last_10_avg < first_10_avg:
        print("  [✓] Task loss decreased during training")
    else:
        print("  [✗] Task loss did NOT decrease - model is not learning!")
    
    # 2. Should not be 100% background predictions
    if bg_ratio < 0.99:
        print(f"  [✓] Not fully collapsed to background ({bg_ratio*100:.1f}% bg)")
    else:
        print(f"  [✗] Model collapsed to background ({bg_ratio*100:.1f}% bg)")
    
    # 3. CISL losses should be reasonable
    last_cisl = np.mean(cisl_losses[-10:])
    if last_cisl < 1.0:
        print(f"  [✓] CISL loss reasonable ({last_cisl:.4f})")
    else:
        print(f"  [!] CISL loss high ({last_cisl:.4f})")
    
    return {
        'task_loss_improved': last_10_avg < first_10_avg,
        'not_collapsed': bg_ratio < 0.99,
        'final_task_loss': last_10_avg,
        'bg_ratio': bg_ratio,
    }


def test_color_permutation_invariance():
    """
    Test if the current color permutation strategy is appropriate.
    
    Question: Should structure embeddings be identical after color permutation?
    
    For ARC: DEPENDS on the task!
    - "Tile the pattern 3x3" → color INDEPENDENT (permutation invariant)
    - "Change all red to blue" → color DEPENDENT (NOT permutation invariant)
    
    This test checks what the current model does.
    """
    print("\n" + "=" * 70)
    print("TEST: Color Permutation Invariance Analysis")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    config = SCIARCConfig(
        hidden_dim=64,
        num_colors=10,
        max_grid_size=10,
        num_structure_slots=4,
        se_layers=1,
    )
    
    model = SCIARC(config).to(device)
    model.eval()
    
    # Create a simple pattern
    grid = torch.zeros(1, 2, 5, 5, dtype=torch.long, device=device)
    # Simple L shape with color 1
    grid[0, 0, 0, 0] = 1
    grid[0, 0, 1, 0] = 1
    grid[0, 0, 2, 0] = 1
    grid[0, 0, 2, 1] = 1
    grid[0, 0, 2, 2] = 1
    
    # Output is same L (identity for this test)
    output_grid = grid.clone()
    
    # Color permuted version (1 → 3)
    grid_perm = grid.clone()
    grid_perm[grid_perm == 1] = 3
    output_grid_perm = output_grid.clone()
    output_grid_perm[output_grid_perm == 1] = 3
    
    with torch.no_grad():
        # Get structure embeddings for original
        z_struct_orig = model.encode_structure_only(grid, output_grid)
        
        # Get structure embeddings for color-permuted
        z_struct_perm = model.encode_structure_only(grid_perm, output_grid_perm)
        
        # Compare
        z_orig_flat = z_struct_orig.view(1, -1)
        z_perm_flat = z_struct_perm.view(1, -1)
        
        z_orig_norm = F.normalize(z_orig_flat, dim=-1)
        z_perm_norm = F.normalize(z_perm_flat, dim=-1)
        
        cos_sim = (z_orig_norm * z_perm_norm).sum().item()
        l2_dist = (z_orig_flat - z_perm_flat).pow(2).sum().sqrt().item()
        
    print(f"\n[RESULTS]")
    print(f"  Original pattern color: 1")
    print(f"  Permuted pattern color: 3")
    print(f"  Cosine similarity of structure embeddings: {cos_sim:.4f}")
    print(f"  L2 distance: {l2_dist:.4f}")
    
    print(f"\n[INTERPRETATION]")
    if cos_sim > 0.95:
        print("  Model treats color-permuted grids as SIMILAR structure.")
        print("  This is CORRECT for color-independent tasks (tiling, scaling, etc.)")
        print("  This is INCORRECT for color-dependent tasks (color swap, fill, etc.)")
    elif cos_sim < 0.5:
        print("  Model treats color-permuted grids as DIFFERENT structure.")
        print("  This might cause issues for color-independent tasks.")
    else:
        print("  Model shows partial color sensitivity.")
    
    return {
        'cos_sim': cos_sim,
        'l2_dist': l2_dist,
    }


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("SCI-ARC LEARNING CAPABILITY TESTS")
    print("=" * 70)
    
    # Test 1: Can the model learn at all?
    result1 = test_simple_task_learning()
    
    # Test 2: Color permutation behavior
    result2 = test_color_permutation_invariance()
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if result1['task_loss_improved'] and result1['not_collapsed']:
        print("[✓] Model can learn simple tasks")
    else:
        print("[✗] Model CANNOT learn - fundamental issue!")
        if not result1['task_loss_improved']:
            print("    - Task loss not improving")
        if not result1['not_collapsed']:
            print("    - Model collapsed to background predictions")
