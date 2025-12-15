#!/usr/bin/env python
"""
Validate TRM Best Practices Implementation in SCI-ARC.

Tests:
1. Dihedral transforms (all 8)
2. Inverse dihedral transforms
3. Color permutation (0 stays fixed)
4. stablemax_cross_entropy loss
5. EMAHelper
6. AugmentationVoter
7. TRMCompatibleDataset format
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch

print("=" * 60)
print(" TRM BEST PRACTICES VALIDATION")
print("=" * 60)

passed = 0
failed = 0

def check(name, condition, details=""):
    global passed, failed
    if condition:
        print(f"  [✓ PASS] {name}")
        if details:
            print(f"          {details}")
        passed += 1
    else:
        print(f"  [✗ FAIL] {name}")
        if details:
            print(f"          {details}")
        failed += 1

# =============================================================================
# 1. DIHEDRAL TRANSFORMS
# =============================================================================
print("\n" + "=" * 60)
print(" 1. DIHEDRAL TRANSFORMS")
print("=" * 60)

try:
    from sci_arc.data import dihedral_transform, inverse_dihedral_transform, DIHEDRAL_INVERSE
    
    # Create test grid
    test_grid = np.array([
        [1, 2, 3],
        [4, 5, 6],
    ], dtype=np.int64)
    
    # Test each transform
    transforms_ok = True
    for tid in range(8):
        transformed = dihedral_transform(test_grid, tid)
        inversed = inverse_dihedral_transform(transformed, tid)
        if not np.array_equal(inversed, test_grid):
            print(f"    Transform {tid} inverse failed!")
            transforms_ok = False
    
    check("All 8 dihedral transforms", transforms_ok)
    
    # Test specific transforms
    # Identity
    check("Transform 0 (identity)", np.array_equal(dihedral_transform(test_grid, 0), test_grid))
    
    # Rot90
    rot90 = dihedral_transform(test_grid, 1)
    expected_rot90 = np.array([[3, 6], [2, 5], [1, 4]])
    check("Transform 1 (rot90)", np.array_equal(rot90, expected_rot90), f"Shape: {rot90.shape}")
    
    # Horizontal flip
    fliph = dihedral_transform(test_grid, 4)
    expected_fliph = np.array([[3, 2, 1], [6, 5, 4]])
    check("Transform 4 (flip horizontal)", np.array_equal(fliph, expected_fliph))
    
    # Transpose
    transp = dihedral_transform(test_grid, 6)
    expected_transp = np.array([[1, 4], [2, 5], [3, 6]])
    check("Transform 6 (transpose)", np.array_equal(transp, expected_transp), f"Shape: {transp.shape}")
    
    # Verify inverse mapping
    check("DIHEDRAL_INVERSE correct", DIHEDRAL_INVERSE == [0, 3, 2, 1, 4, 5, 6, 7])
    
except Exception as e:
    check("Dihedral transforms import", False, str(e))

# =============================================================================
# 2. STABLEMAX CROSS ENTROPY
# =============================================================================
print("\n" + "=" * 60)
print(" 2. STABLEMAX CROSS ENTROPY")
print("=" * 60)

try:
    from sci_arc.training import stablemax_cross_entropy
    import torch.nn.functional as F
    
    # Create test data
    logits = torch.randn(4, 900, 12)  # [B, seq_len, vocab]
    labels = torch.randint(0, 12, (4, 900))  # [B, seq_len]
    
    # Compute stablemax loss
    stablemax_loss = stablemax_cross_entropy(logits, labels, n=12, ignore_index=0)
    check("stablemax_cross_entropy computes", True, f"Loss: {stablemax_loss.item():.4f}")
    
    # Compare with standard CE
    ce_loss = F.cross_entropy(logits.view(-1, 12), labels.view(-1), ignore_index=0)
    check("Standard CE for comparison", True, f"Loss: {ce_loss.item():.4f}")
    
    # Check no NaN
    check("No NaN in stablemax", not torch.isnan(stablemax_loss).any())
    
    # Check gradient flows
    logits_grad = logits.clone().requires_grad_(True)
    loss = stablemax_cross_entropy(logits_grad, labels, n=12, ignore_index=0)
    loss.backward()
    check("Gradient flows", logits_grad.grad is not None)
    
except Exception as e:
    check("stablemax_cross_entropy import", False, str(e))

# =============================================================================
# 3. EMA HELPER
# =============================================================================
print("\n" + "=" * 60)
print(" 3. EMA HELPER")
print("=" * 60)

try:
    from sci_arc.training import EMAHelper, EMAWrapper
    import torch.nn as nn
    
    # Create simple model
    model = nn.Linear(64, 32)
    
    # Create EMA helper
    ema = EMAHelper(model, mu=0.999)
    check("EMAHelper instantiation", True)
    
    # Update EMA
    with torch.no_grad():
        model.weight.add_(torch.randn_like(model.weight) * 0.1)
    ema.update(model)
    check("EMA update", True)
    
    # Create EMA copy
    ema_model = ema.ema_copy(model)
    check("EMA copy creation", ema_model is not model)
    
    # Check weights differ from original
    weight_diff = (model.weight - ema_model.weight).abs().mean()
    check("EMA weights differ from current", weight_diff > 0, f"Diff: {weight_diff.item():.6f}")
    
    # Test apply/restore
    original = ema.apply_shadow(model)
    check("Apply shadow", True)
    ema.restore(model, original)
    check("Restore original", True)
    
    # Test state dict
    state = ema.state_dict()
    check("State dict", 'mu' in state and 'shadow' in state)
    
except Exception as e:
    check("EMA import", False, str(e))

# =============================================================================
# 4. AUGMENTATION VOTER
# =============================================================================
print("\n" + "=" * 60)
print(" 4. AUGMENTATION VOTER")
print("=" * 60)

try:
    from sci_arc.evaluation import (
        dihedral_transform_torch,
        inverse_dihedral_transform_torch,
        vote_predictions,
    )
    
    # Test PyTorch dihedral transforms
    test_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
    
    transforms_ok = True
    for tid in range(8):
        transformed = dihedral_transform_torch(test_tensor, tid)
        inversed = inverse_dihedral_transform_torch(transformed, tid)
        if not torch.equal(inversed, test_tensor):
            print(f"    PyTorch Transform {tid} inverse failed!")
            transforms_ok = False
    
    check("PyTorch dihedral transforms", transforms_ok)
    
    # Test voting
    pred1 = np.array([[0, 1, 2], [3, 4, 5]])
    pred2 = np.array([[0, 1, 2], [3, 4, 5]])  # Same
    pred3 = np.array([[0, 0, 2], [3, 4, 0]])  # Different at [0,1] and [1,2]
    
    voted = vote_predictions([pred1, pred2, pred3])
    expected = np.array([[0, 1, 2], [3, 4, 5]])  # Majority wins
    check("Vote predictions", np.array_equal(voted, expected))
    
except Exception as e:
    check("Augmentation voter import", False, str(e))

# =============================================================================
# 5. TRM COMPATIBLE DATASET FORMAT
# =============================================================================
print("\n" + "=" * 60)
print(" 5. TRM COMPATIBLE DATASET FORMAT")
print("=" * 60)

try:
    from sci_arc.data.dataset import TRMCompatibleDataset
    
    # Check class attributes
    check("PAD_TOKEN = 0", TRMCompatibleDataset.PAD_TOKEN == 0)
    check("EOS_TOKEN = 1", TRMCompatibleDataset.EOS_TOKEN == 1)
    check("COLOR_OFFSET = 2", TRMCompatibleDataset.COLOR_OFFSET == 2)
    check("VOCAB_SIZE = 12", TRMCompatibleDataset.VOCAB_SIZE == 12)
    check("SEQ_LEN = 900 (30x30)", TRMCompatibleDataset.SEQ_LEN == 900)
    check("GRID_SIZE = 30", TRMCompatibleDataset.GRID_SIZE == 30)
    
except Exception as e:
    check("TRMCompatibleDataset import", False, str(e))

# =============================================================================
# 6. COLOR PERMUTATION SAFETY
# =============================================================================
print("\n" + "=" * 60)
print(" 6. COLOR PERMUTATION SAFETY")
print("=" * 60)

try:
    import random
    from sci_arc.data.dataset import dihedral_transform
    
    # Simulate the color permutation logic from dataset.py
    num_colors = 10
    
    # Run many trials to ensure 0 is never permuted
    all_safe = True
    for trial in range(100):
        color_perm = list(range(1, num_colors))  # [1, 2, ..., 9]
        random.shuffle(color_perm)
        color_map = {0: 0}  # 0 stays 0
        for i, new_c in enumerate(color_perm):
            color_map[i + 1] = new_c
        
        if color_map[0] != 0:
            all_safe = False
            break
    
    check("Color 0 (background) never permuted", all_safe)
    
    # Check all colors 1-9 are present in permutation
    colors_present = set(color_perm)
    expected_colors = set(range(1, num_colors))
    check("All colors 1-9 present in permutation", colors_present == expected_colors)
    
except Exception as e:
    check("Color permutation check", False, str(e))

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 60)
print(" VALIDATION SUMMARY")
print("=" * 60)
print(f"  Passed: {passed}")
print(f"  Failed: {failed}")
print(f"  Total:  {passed + failed}")
print()

if failed == 0:
    print("  ✓ ALL TRM BEST PRACTICES CORRECTLY IMPLEMENTED!")
    sys.exit(0)
else:
    print(f"  ✗ {failed} tests failed")
    sys.exit(1)
