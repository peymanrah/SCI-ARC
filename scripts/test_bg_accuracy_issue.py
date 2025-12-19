#!/usr/bin/env python3
"""
Test script to investigate why BG (background/color 0) has only 1% accuracy
despite being 52% of target pixels.

This tests:
1. Weighted loss computation - is BG being underweighted?
2. Accuracy calculation - is the metric correct?
3. Model output distribution - is the model biased against BG?
4. Loss gradient analysis - are BG gradients being suppressed?
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
from sci_arc.training.rlan_loss import WeightedStablemaxLoss, log_stablemax


def test_weighted_loss_weights():
    """Test if weighted loss correctly computes class weights."""
    print("\n" + "="*60)
    print("TEST 1: Weighted Loss Class Weights")
    print("="*60)
    
    # Simulate batch statistics from the log:
    # Target %: [52.3, 8.5, 13.4, 3.4, 8.5, 2.4, 3.1, 0.6, 3.4, 4.5]
    # This means BG (color 0) is 52.3% of valid pixels
    
    loss_fn = WeightedStablemaxLoss(
        bg_weight_cap=2.0,  # From config
        fg_weight_cap=5.0,  # From config
    )
    
    # Create mock targets with the observed distribution
    n_pixels = 10000
    target_dist = [0.523, 0.085, 0.134, 0.034, 0.085, 0.024, 0.031, 0.006, 0.034, 0.045]
    
    targets = []
    for c, pct in enumerate(target_dist):
        targets.extend([c] * int(n_pixels * pct))
    targets = torch.tensor(targets, dtype=torch.long)
    
    # Compute class weights like the loss function does
    class_counts = torch.bincount(targets, minlength=10).float()
    class_freq = class_counts / len(targets)
    raw_weights = 1.0 / (class_freq + 1e-6)
    
    # Apply caps
    weights = torch.zeros_like(raw_weights)
    weights[0] = raw_weights[0].clamp(0.1, 2.0)  # BG cap
    weights[1:] = raw_weights[1:].clamp(0.1, 5.0)  # FG cap
    
    # Normalize
    weights = weights * (10 / (weights.sum() + 1e-6))
    
    print(f"\nTarget distribution: {[f'{p:.1%}' for p in target_dist]}")
    print(f"Raw weights (1/freq): {raw_weights.numpy()}")
    print(f"Capped weights: {weights.numpy()}")
    print(f"\nBG weight: {weights[0]:.3f} (cap=2.0)")
    print(f"FG weights: {weights[1:].numpy()}")
    
    # Analyze the weight ratio
    bg_ratio = weights[0] / weights.mean()
    fg_mean_ratio = weights[1:].mean() / weights.mean()
    print(f"\nBG weight ratio to mean: {bg_ratio:.2f}x")
    print(f"FG mean weight ratio to mean: {fg_mean_ratio:.2f}x")
    
    # THE ISSUE: BG is capped at 2.0 but FG can be 5.0
    # With 52% BG, its inverse freq is ~1.9, so it's not even hitting the cap
    # But rare FG colors (like color 7 at 0.6%) get weight 5.0
    # This creates a 2.5x bias toward FG colors!
    
    print(f"\n⚠️ ISSUE: BG weight ({weights[0]:.2f}) vs max FG weight ({weights[1:].max():.2f})")
    print(f"   Rare FG colors get {weights[1:].max()/weights[0]:.1f}x more gradient than BG!")
    
    return weights


def test_accuracy_calculation():
    """Test if accuracy is calculated correctly for BG vs FG."""
    print("\n" + "="*60)
    print("TEST 2: Accuracy Calculation")
    print("="*60)
    
    # Create mock predictions and targets
    # Simulate observed behavior: model predicts color 9 too much
    B, H, W = 4, 10, 10
    
    # Target: 52% color 0, rest distributed
    targets = torch.zeros(B, H, W, dtype=torch.long)
    targets[:, :5, :] = 0  # Top half is BG (50%)
    targets[:, 5:7, :] = 1  # Some FG
    targets[:, 7:8, :] = 2
    targets[:, 8:9, :] = 3
    targets[:, 9:, :] = 9
    
    # Prediction: Model over-predicts color 9 (as seen in logs)
    # Pred %: [1.2, 13.7, 11.0, 9.4, 11.8, 8.3, 5.1, 5.1, 3.7, 30.6]
    preds = torch.randint(0, 10, (B, H, W))
    # Make it heavily biased toward color 9
    preds[:, :, :3] = 9  # 30% is color 9
    
    # Calculate accuracy as done in train_rlan.py
    valid_mask = targets != -100  # All valid in this test
    
    total_correct = ((preds == targets) & valid_mask).sum().item()
    total_valid = valid_mask.sum().item()
    overall_acc = total_correct / total_valid
    
    # Per-class accuracy
    per_class_acc = []
    per_class_pct = []
    for c in range(10):
        class_mask = (targets == c) & valid_mask
        class_pixels = class_mask.sum().item()
        if class_pixels > 0:
            class_correct = ((preds == c) & class_mask).sum().item()
            per_class_acc.append(class_correct / class_pixels)
            per_class_pct.append(class_pixels / total_valid)
        else:
            per_class_acc.append(0.0)
            per_class_pct.append(0.0)
    
    print(f"\nTarget class distribution: {[f'{p:.1%}' for p in per_class_pct]}")
    print(f"Per-class accuracy: {[f'{a:.1%}' for a in per_class_acc]}")
    print(f"Overall accuracy: {overall_acc:.1%}")
    
    # BG accuracy specifically
    bg_mask = (targets == 0) & valid_mask
    bg_correct = ((preds == 0) & bg_mask).sum().item()
    bg_acc = bg_correct / bg_mask.sum().item() if bg_mask.sum() > 0 else 0
    print(f"\nBG (color 0) accuracy: {bg_acc:.1%}")
    print(f"BG pixels: {bg_mask.sum().item()}/{total_valid} ({bg_mask.sum().item()/total_valid:.1%})")
    
    return per_class_acc


def test_loss_gradient_flow():
    """Test if gradients flow properly to BG predictions."""
    print("\n" + "="*60)
    print("TEST 3: Loss Gradient Flow for BG vs FG")
    print("="*60)
    
    # Create mock logits and targets
    B, C, H, W = 2, 10, 8, 8
    logits = torch.randn(B, C, H, W, requires_grad=True)
    
    # Target: half BG, half FG
    targets = torch.zeros(B, H, W, dtype=torch.long)
    targets[:, :4, :] = 0  # BG
    targets[:, 4:, :] = 3  # FG (color 3)
    
    # Create loss function
    loss_fn = WeightedStablemaxLoss(
        bg_weight_cap=2.0,
        fg_weight_cap=5.0,
    )
    
    # Compute loss
    loss = loss_fn(logits, targets)
    loss.backward()
    
    # Analyze gradients
    grad = logits.grad  # (B, C, H, W)
    
    # Gradient magnitude for BG vs FG regions
    bg_region_grad = grad[:, :, :4, :].abs().mean().item()
    fg_region_grad = grad[:, :, 4:, :].abs().mean().item()
    
    # Gradient magnitude for BG class vs FG class predictions
    bg_class_grad = grad[:, 0, :, :].abs().mean().item()  # Gradients to BG class logit
    fg_class_grad = grad[:, 1:, :, :].abs().mean().item()  # Gradients to FG class logits
    
    print(f"\nGradient magnitude in BG region: {bg_region_grad:.4f}")
    print(f"Gradient magnitude in FG region: {fg_region_grad:.4f}")
    print(f"Ratio (FG/BG region): {fg_region_grad/bg_region_grad:.2f}x")
    
    print(f"\nGradient to BG class logit: {bg_class_grad:.4f}")
    print(f"Gradient to FG class logits (mean): {fg_class_grad:.4f}")
    print(f"Ratio (FG/BG class): {fg_class_grad/bg_class_grad:.2f}x")
    
    if fg_region_grad / bg_region_grad > 2:
        print(f"\n⚠️ ISSUE: FG regions receive {fg_region_grad/bg_region_grad:.1f}x more gradient than BG!")
        print("   This could explain why BG accuracy is so low.")
    
    return bg_region_grad, fg_region_grad


def test_stablemax_behavior():
    """Test stablemax numerical stability and output distribution."""
    print("\n" + "="*60)
    print("TEST 4: Stablemax Behavior")
    print("="*60)
    
    # Test with various logit ranges
    test_cases = [
        ("Normal", torch.randn(4, 10)),
        ("Large positive", torch.randn(4, 10) * 10 + 50),
        ("Large negative", torch.randn(4, 10) * 10 - 50),
        ("Mixed extreme", torch.cat([torch.randn(2, 10) * 100, torch.randn(2, 10) * 0.1])),
    ]
    
    for name, logits in test_cases:
        log_probs = log_stablemax(logits, dim=-1)
        probs = torch.exp(log_probs)
        
        print(f"\n{name}:")
        print(f"  Logits range: [{logits.min():.2f}, {logits.max():.2f}]")
        print(f"  Log probs range: [{log_probs.min():.2f}, {log_probs.max():.2f}]")
        print(f"  Probs sum: {probs.sum(dim=-1).mean():.4f} (should be ~1.0)")
        print(f"  NaN/Inf: {torch.isnan(log_probs).any() or torch.isinf(log_probs).any()}")


def test_model_output_bias():
    """Test if model architecture has inherent bias."""
    print("\n" + "="*60)
    print("TEST 5: Check for Model Output Bias")
    print("="*60)
    
    # This would require loading the actual model
    # For now, analyze what could cause bias
    
    print("\nPotential causes of BG under-prediction:")
    print("1. Weighted loss caps: bg_cap=2.0 < fg_cap=5.0")
    print("   → Rare FG colors get 2.5x more training signal")
    print()
    print("2. Class imbalance: 52% BG vs 48% FG (9 colors)")
    print("   → Each FG color is ~5% on average, gets higher weight")
    print()
    print("3. Color embedding: BG=0 might have weak representation")
    print("   → Check if color embedding for 0 is learned properly")
    print()
    print("4. Solver residual: If solver degrades, may favor mode prediction")
    print("   → Check if color 9 is the default/mode color")


def propose_fixes():
    """Propose fixes for the BG accuracy issue."""
    print("\n" + "="*60)
    print("PROPOSED FIXES")
    print("="*60)
    
    print("""
1. INCREASE BG WEIGHT CAP:
   Current: bg_weight_cap=2.0, fg_weight_cap=5.0
   Proposed: bg_weight_cap=3.0, fg_weight_cap=3.0 (balanced)
   
   Rationale: With 52% BG, inverse frequency gives weight ~1.9
   But rare FG colors (0.6%) get weight 5.0
   This 2.5x imbalance causes BG under-learning.

2. USE BALANCED CLASS WEIGHTS:
   Instead of inverse frequency, use sqrt(inverse frequency)
   This reduces the extreme weighting of rare classes.
   
3. ADD BG-SPECIFIC LOSS TERM:
   lambda_bg * CE(pred_bg, target_bg)
   Force model to learn BG prediction explicitly.

4. CHECK PADDING vs BG CONFUSION:
   Padding = -100, BG = 0
   If model confuses them, it might avoid predicting 0.
   
5. INCREASE TEMPERATURE:
   Higher temperature → softer predictions
   May help model not over-commit to FG colors.
""")


if __name__ == "__main__":
    print("="*60)
    print("BG ACCURACY INVESTIGATION")
    print("="*60)
    print("\nObserved issue from training logs:")
    print("  Per-Class Acc %: [1, 15, 8, 6, 21, 18, 2, 8, 5, 19]")
    print("  BG (color 0) has only 1% accuracy!")
    print("  But BG is 52.3% of target pixels!")
    print()
    print("  Model predicts: [1.2, 13.7, 11.0, 9.4, 11.8, 8.3, 5.1, 5.1, 3.7, 30.6]%")
    print("  Target has:     [52.3, 8.5, 13.4, 3.4, 8.5, 2.4, 3.1, 0.6, 3.4, 4.5]%")
    print()
    print("  → Model predicts BG only 1.2% of time but target has 52.3%!")
    print("  → Model over-predicts color 9 (30.6% vs 4.5% target)")
    
    weights = test_weighted_loss_weights()
    test_accuracy_calculation()
    test_loss_gradient_flow()
    test_stablemax_behavior()
    test_model_output_bias()
    propose_fixes()
    
    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60)
    print("""
The root cause is likely the WEIGHTED LOSS IMBALANCE:
- BG (52% of pixels) gets weight capped at 2.0
- Rare FG colors (0.6% of pixels) get weight 5.0
- This creates up to 2.5x more gradient for FG than BG

The model learns to predict FG colors (higher reward) and 
ignores BG (lower reward), leading to 1% BG accuracy.

IMMEDIATE FIX: Set bg_weight_cap = fg_weight_cap = 3.0
""")
