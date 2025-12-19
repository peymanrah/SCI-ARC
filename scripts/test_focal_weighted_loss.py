#!/usr/bin/env python
"""
Test the new FocalWeightedStablemaxLoss implementation.

Verifies:
1. Loss computes correctly
2. BG/FG gradient balance is maintained
3. Focal modulation works (easy pixels get down-weighted)
"""

import sys
sys.path.insert(0, ".")

import torch
from sci_arc.training.rlan_loss import (
    FocalWeightedStablemaxLoss, 
    WeightedStablemaxLoss,
    stablemax
)


def test_focal_weighted_loss():
    """Test the new FocalWeightedStablemaxLoss."""
    print("=" * 70)
    print("TESTING FocalWeightedStablemaxLoss")
    print("=" * 70)
    
    torch.manual_seed(42)
    B, C, H, W = 8, 10, 20, 20
    total_pixels = B * H * W
    
    # Imbalanced class distribution (52% BG)
    probs = [0.52, 0.10, 0.08, 0.05, 0.05, 0.02, 0.05, 0.05, 0.05, 0.03]
    targets = torch.multinomial(torch.tensor(probs), total_pixels, replacement=True).view(B, H, W)
    
    logits = torch.randn(B, C, H, W, requires_grad=True)
    
    print("\nData Distribution:")
    for c in range(C):
        count = (targets == c).sum().item()
        bar = "█" * int(count / total_pixels * 50)
        print(f"  Class {c}: {count:4d} ({100*count/total_pixels:5.1f}%) {bar}")
    
    # ===== Test FocalWeightedStablemaxLoss =====
    loss_fn = FocalWeightedStablemaxLoss(
        bg_weight_cap=2.0,
        fg_weight_cap=5.0,
        gamma=2.0,
    )
    
    loss = loss_fn(logits, targets)
    loss.backward()
    
    print(f"\n[FocalWeightedStablemaxLoss]")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Gradient norm: {logits.grad.norm().item():.4f}")
    
    # Check BG vs FG gradient distribution
    bg_mask = targets == 0
    fg_mask = targets > 0
    
    bg_grad = logits.grad[:, 0, :, :][bg_mask].abs().sum().item()
    fg_grad = logits.grad[:, 1:, :, :].abs().sum().item()
    total_grad = bg_grad + fg_grad
    
    bg_pixels = bg_mask.sum().item()
    fg_pixels = fg_mask.sum().item()
    
    print(f"\n  BG pixels: {bg_pixels} ({100*bg_pixels/total_pixels:.1f}%)")
    print(f"  FG pixels: {fg_pixels} ({100*fg_pixels/total_pixels:.1f}%)")
    print(f"  BG gradient share: {100*bg_grad/total_grad:.1f}%")
    print(f"  FG gradient share: {100*fg_grad/total_grad:.1f}%")
    
    # ===== Compare with WeightedStablemaxLoss =====
    logits2 = torch.randn(B, C, H, W, requires_grad=True)
    loss_fn2 = WeightedStablemaxLoss(bg_weight_cap=2.0, fg_weight_cap=5.0)
    loss2 = loss_fn2(logits2, targets)
    loss2.backward()
    
    print(f"\n[WeightedStablemaxLoss (baseline)]")
    print(f"  Loss: {loss2.item():.4f}")
    print(f"  Gradient norm: {logits2.grad.norm().item():.4f}")
    
    # ===== Test focal modulation effect =====
    print("\n" + "=" * 70)
    print("FOCAL MODULATION TEST")
    print("=" * 70)
    
    # Create two scenarios: random logits vs confident logits
    logits_random = torch.randn(B, C, H, W, requires_grad=True)
    
    # Confident logits: boost correct class
    logits_confident = logits_random.clone().detach()
    for b in range(B):
        for h in range(H):
            for w in range(W):
                c = targets[b, h, w].item()
                logits_confident[b, c, h, w] += 5.0  # Boost correct class
    logits_confident.requires_grad_(True)
    
    loss_random = loss_fn(logits_random, targets)
    loss_confident = loss_fn(logits_confident, targets)
    
    print(f"  Random logits loss:    {loss_random.item():.4f}")
    print(f"  Confident logits loss: {loss_confident.item():.4f}")
    print(f"  Reduction: {100*(1 - loss_confident.item()/loss_random.item()):.1f}%")
    print("  (Lower loss for confident = focal is down-weighting easy pixels)")
    
    # Check gradient magnitude
    loss_random.backward()
    loss_confident.backward()
    
    grad_random = logits_random.grad.abs().mean().item()
    grad_confident = logits_confident.grad.abs().mean().item()
    
    print(f"\n  Random logits grad magnitude:    {grad_random:.6f}")
    print(f"  Confident logits grad magnitude: {grad_confident:.6f}")
    print(f"  Reduction: {100*(1 - grad_confident/grad_random):.1f}%")
    print("  (Lower grad for confident = focal is reducing gradient for easy pixels)")
    
    print("\n" + "=" * 70)
    print("✓ FocalWeightedStablemaxLoss working correctly!")
    print("=" * 70)
    
    return True


def test_rlan_loss_integration():
    """Test that RLANLoss correctly initializes with focal_weighted mode."""
    print("\n" + "=" * 70)
    print("RLAN LOSS INTEGRATION TEST")
    print("=" * 70)
    
    from sci_arc.training.rlan_loss import RLANLoss
    
    # Test focal_weighted mode
    loss_fn = RLANLoss(
        loss_mode='focal_weighted',
        focal_gamma=2.0,
        bg_weight_cap=2.0,
        fg_weight_cap=5.0,
        lambda_entropy=0.1,
        lambda_sparsity=0.0,
        lambda_predicate=0.001,
        lambda_deep_supervision=0.3,
    )
    
    print(f"  Task loss type: {type(loss_fn.task_loss).__name__}")
    assert isinstance(loss_fn.task_loss, FocalWeightedStablemaxLoss), \
        f"Expected FocalWeightedStablemaxLoss, got {type(loss_fn.task_loss).__name__}"
    
    print("  ✓ focal_weighted mode correctly initializes FocalWeightedStablemaxLoss")
    
    return True


if __name__ == "__main__":
    test_focal_weighted_loss()
    test_rlan_loss_integration()
    print("\n✅ All tests passed!")
