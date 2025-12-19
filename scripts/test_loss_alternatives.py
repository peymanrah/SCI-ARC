#!/usr/bin/env python
"""
Compare different loss weighting strategies for imbalanced segmentation.

This script explores alternatives to WeightedStablemaxLoss:
1. Unweighted CE (baseline)
2. Current WeightedStablemaxLoss with caps
3. Focal Loss (no class weights)
4. Class-Balanced Loss (Cui et al., 2019)
5. Dice Loss (segmentation-native)
6. Learnable Class Weights (our proposed approach)
7. Focal + Learnable (best of both)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def stablemax(x, dim=-1):
    """Numerically stable softmax alternative."""
    x_max = x.max(dim=dim, keepdim=True)[0]
    x_shifted = x - x_max
    exp_x = torch.exp(x_shifted.clamp(-20, 20))
    return exp_x / (exp_x.sum(dim=dim, keepdim=True) + 1e-8)


def log_stablemax(x, dim=-1):
    """Log of stablemax for numerical stability."""
    probs = stablemax(x, dim)
    return torch.log(probs.clamp(min=1e-10))


class LearnableWeightedLoss(nn.Module):
    """
    Learn class weights during training.
    
    Key insight: Instead of manually setting weights, let the model
    learn them through backpropagation. This allows:
    - Adaptation to changing class difficulty
    - Task-specific weight adjustment
    - No hyperparameter tuning
    
    The weights are learned in log-space and passed through softplus
    to ensure they remain positive.
    """
    def __init__(self, num_classes, init_weight=1.0, normalize=False):
        super().__init__()
        self.num_classes = num_classes
        self.normalize = normalize
        
        # Learn in log-space for numerical stability
        # softplus(log_weights) ensures positive weights
        self.log_weights = nn.Parameter(torch.zeros(num_classes))
        # Initialize to roughly uniform
        nn.init.constant_(self.log_weights, np.log(np.exp(init_weight) - 1))
    
    def get_weights(self):
        # Softplus ensures positive, exp(0)=1 baseline
        weights = F.softplus(self.log_weights)
        
        # Optional: normalize to prevent gradient scale issues
        if self.normalize:
            weights = weights / weights.mean()
        
        return weights
    
    def forward(self, logits, targets):
        B, C, H, W = logits.shape
        logprobs = log_stablemax(logits.permute(0, 2, 3, 1).reshape(-1, C))
        targets_flat = targets.view(-1)
        
        # Mask out padding (-100)
        valid_mask = targets_flat >= 0
        if not valid_mask.any():
            return torch.tensor(0.0, device=logits.device)
        
        logprobs = logprobs[valid_mask]
        targets_valid = targets_flat[valid_mask]
        
        weights = self.get_weights()
        pixel_weights = weights[targets_valid]
        
        loss = -(pixel_weights * logprobs[range(len(targets_valid)), targets_valid]).mean()
        return loss


class AdaptiveLearnableLoss(nn.Module):
    """
    Learnable weights that adapt per-batch based on class frequency.
    
    Unlike static learnable weights, this version:
    1. Computes batch-specific base weights (inverse frequency)
    2. Learns a per-class SCALING factor (not absolute weight)
    3. Allows adaptation to varying batch compositions
    
    Formula: weight_c = base_c * softplus(log_scale_c)
    where base_c = sqrt(1/freq_c) for smooth scaling
    """
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        # Learn scaling factors (1.0 = neutral, >1 = boost, <1 = reduce)
        self.log_scale = nn.Parameter(torch.zeros(num_classes))
    
    def get_scales(self):
        return F.softplus(self.log_scale)
    
    def forward(self, logits, targets):
        B, C, H, W = logits.shape
        logprobs = log_stablemax(logits.permute(0, 2, 3, 1).reshape(-1, C))
        targets_flat = targets.view(-1)
        
        # Mask out padding
        valid_mask = targets_flat >= 0
        if not valid_mask.any():
            return torch.tensor(0.0, device=logits.device)
        
        logprobs = logprobs[valid_mask]
        targets_valid = targets_flat[valid_mask]
        n_valid = len(targets_valid)
        
        # Compute batch-specific base weights
        class_counts = torch.bincount(targets_valid, minlength=C).float()
        class_freq = class_counts / (n_valid + 1e-6)
        
        # Use sqrt for smoother scaling (less extreme than 1/freq)
        base_weights = torch.sqrt(1.0 / (class_freq + 1e-6))
        base_weights = base_weights / base_weights.mean()  # Normalize
        
        # Apply learned scales
        scales = self.get_scales()
        final_weights = base_weights * scales
        
        # Apply per-pixel
        pixel_weights = final_weights[targets_valid]
        loss = -(pixel_weights * logprobs[range(n_valid), targets_valid]).mean()
        return loss, final_weights.detach()


class DiceCELoss(nn.Module):
    """
    Combination of Dice Loss and Cross-Entropy.
    
    Dice naturally handles class imbalance through its ratio formulation.
    CE provides pixel-level gradients for stable training.
    
    Common in medical image segmentation (nnU-Net uses this).
    """
    def __init__(self, num_classes, dice_weight=0.5, smooth=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.dice_weight = dice_weight
        self.smooth = smooth
    
    def forward(self, logits, targets):
        B, C, H, W = logits.shape
        
        # CE part
        logprobs = log_stablemax(logits.permute(0, 2, 3, 1).reshape(-1, C))
        targets_flat = targets.view(-1)
        
        valid_mask = targets_flat >= 0
        if not valid_mask.any():
            return torch.tensor(0.0, device=logits.device)
        
        ce_loss = F.nll_loss(logprobs[valid_mask], targets_flat[valid_mask])
        
        # Dice part (per-class, then average)
        probs = stablemax(logits.permute(0, 2, 3, 1).reshape(-1, C))
        probs = probs[valid_mask]
        targets_valid = targets_flat[valid_mask]
        targets_onehot = F.one_hot(targets_valid, C).float()
        
        dice_per_class = []
        for c in range(C):
            p = probs[:, c]
            t = targets_onehot[:, c]
            intersection = (p * t).sum()
            dice = (2 * intersection + self.smooth) / (p.sum() + t.sum() + self.smooth)
            dice_per_class.append(dice)
        
        dice_loss = 1 - sum(dice_per_class) / C
        
        return self.dice_weight * dice_loss + (1 - self.dice_weight) * ce_loss


class ConstrainedLearnableLoss(nn.Module):
    """
    Learnable class weights with proper constraints.
    
    KEY INSIGHT: We need to constrain weights to prevent collapse.
    
    Constraints:
    1. Weights bounded: [min_weight, max_weight]
    2. Mean weight normalized to 1.0
    3. L2 regularization on weight deviation from 1.0
    
    This prevents the optimizer from just minimizing weights to reduce loss.
    """
    def __init__(self, num_classes, min_weight=0.5, max_weight=3.0, reg_weight=0.01):
        super().__init__()
        self.num_classes = num_classes
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.reg_weight = reg_weight
        
        # Learn in sigmoid space for bounded weights
        # sigmoid(0) = 0.5, which maps to middle of [min, max]
        self.weight_logits = nn.Parameter(torch.zeros(num_classes))
    
    def get_weights(self):
        # Sigmoid squashes to [0, 1]
        normalized = torch.sigmoid(self.weight_logits)
        # Scale to [min_weight, max_weight]
        weights = self.min_weight + normalized * (self.max_weight - self.min_weight)
        return weights
    
    def forward(self, logits, targets, return_reg=False):
        B, C, H, W = logits.shape
        logprobs = log_stablemax(logits.permute(0, 2, 3, 1).reshape(-1, C))
        targets_flat = targets.view(-1)
        
        valid_mask = targets_flat >= 0
        if not valid_mask.any():
            return torch.tensor(0.0, device=logits.device)
        
        logprobs = logprobs[valid_mask]
        targets_valid = targets_flat[valid_mask]
        
        weights = self.get_weights()
        pixel_weights = weights[targets_valid]
        
        ce_loss = -(pixel_weights * logprobs[range(len(targets_valid)), targets_valid]).mean()
        
        # Regularization: penalize deviation from 1.0
        reg_loss = self.reg_weight * ((weights - 1.0) ** 2).mean()
        
        if return_reg:
            return ce_loss, reg_loss
        return ce_loss + reg_loss


class TemperatureWeightedLoss(nn.Module):
    """
    Learn how much to trust frequency-based weights.
    
    Instead of learning weights directly, learn a single TEMPERATURE
    that controls interpolation between uniform and frequency-based.
    
    w_c = (1-t) * 1.0 + t * base_weight_c
    
    where t in [0,1] is learned:
    - t=0: uniform weights (ignore frequency)
    - t=1: full frequency-based weights
    
    This is more stable and interpretable than learning 10 independent weights!
    """
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        # Temperature in logit space, sigmoid(0)=0.5
        self.temp_logit = nn.Parameter(torch.tensor(0.0))
    
    def get_temperature(self):
        return torch.sigmoid(self.temp_logit)
    
    def forward(self, logits, targets, return_info=False):
        B, C, H, W = logits.shape
        logprobs = log_stablemax(logits.permute(0, 2, 3, 1).reshape(-1, C))
        targets_flat = targets.view(-1)
        
        valid_mask = targets_flat >= 0
        if not valid_mask.any():
            if return_info:
                return torch.tensor(0.0), 0.5, torch.ones(C)
            return torch.tensor(0.0, device=logits.device)
        
        logprobs = logprobs[valid_mask]
        targets_valid = targets_flat[valid_mask]
        n_valid = len(targets_valid)
        
        # Compute frequency-based weights
        class_counts = torch.bincount(targets_valid, minlength=C).float()
        freq = class_counts / (n_valid + 1e-6)
        freq_weights = torch.sqrt(1.0 / (freq + 1e-6))  # sqrt for smoothness
        freq_weights = freq_weights / freq_weights.mean()  # normalize to mean=1
        
        # Interpolate between uniform (1.0) and frequency-based
        t = self.get_temperature()
        weights = (1 - t) * 1.0 + t * freq_weights
        
        pixel_weights = weights[targets_valid]
        loss = -(pixel_weights * logprobs[range(n_valid), targets_valid]).mean()
        
        if return_info:
            return loss, t.item(), weights.detach()
        return loss


def run_comparison():
    """Run comparison of all loss functions."""
    print("=" * 70)
    print("LOSS FUNCTION COMPARISON FOR IMBALANCED SEGMENTATION")
    print("=" * 70)
    
    # Simulate realistic ARC batch data
    torch.manual_seed(42)
    B, C, H, W = 8, 10, 20, 20  # Larger batch for statistics
    total_pixels = B * H * W
    
    # Create imbalanced targets (52% BG, varied FG)
    probs = [0.52, 0.10, 0.08, 0.05, 0.05, 0.02, 0.05, 0.05, 0.05, 0.03]
    
    flat_targets = torch.multinomial(torch.tensor(probs), total_pixels, replacement=True)
    targets = flat_targets.view(B, H, W)
    
    # Random logits (simulating untrained model)
    logits = torch.randn(B, C, H, W, requires_grad=True)
    
    print("\nData Distribution:")
    for c in range(C):
        count = (targets == c).sum().item()
        bar = "█" * int(count / total_pixels * 50)
        print(f"  Class {c}: {count:4d} ({100*count/total_pixels:5.1f}%) {bar}")
    
    # ===== 1. Baseline CE =====
    print("\n" + "=" * 70)
    print("1. BASELINE: Unweighted Cross-Entropy")
    print("=" * 70)
    
    logprobs = log_stablemax(logits.permute(0, 2, 3, 1).reshape(-1, C))
    targets_flat = targets.view(-1)
    ce_loss = F.nll_loss(logprobs, targets_flat)
    print(f"   Loss: {ce_loss.item():.4f}")
    
    # ===== 2. Current WeightedStablemaxLoss =====
    print("\n" + "=" * 70)
    print("2. CURRENT: WeightedStablemaxLoss with caps (bg=2.0, fg=5.0)")
    print("=" * 70)
    
    class_counts = torch.bincount(targets_flat, minlength=C).float()
    class_freq = class_counts / total_pixels
    
    weights_current = torch.zeros(C)
    weights_current[0] = 2.0  # BG cap
    fg_freq = class_freq[1:]
    fg_raw = 1.0 / (fg_freq + 1e-6)
    fg_min, fg_max = fg_raw.min(), fg_raw.max()
    fg_scaled = 1.0 + 4.0 * (fg_raw - fg_min) / (fg_max - fg_min + 1e-6)
    weights_current[1:] = fg_scaled.clamp(0.1, 5.0)
    
    pixel_weights = weights_current[targets_flat]
    weighted_loss = -(pixel_weights * logprobs[range(len(targets_flat)), targets_flat]).mean()
    print(f"   Loss: {weighted_loss.item():.4f}")
    print(f"   Weights: {[f'{w:.2f}' for w in weights_current.tolist()]}")
    
    # ===== 3. Learnable Weights =====
    print("\n" + "=" * 70)
    print("3. LEARNABLE CLASS WEIGHTS (Our Proposal)")
    print("=" * 70)
    
    learnable_loss = LearnableWeightedLoss(C, normalize=True)
    
    print(f"   Initial weights: {[f'{w:.3f}' for w in learnable_loss.get_weights().tolist()]}")
    print(f"   Initial loss: {learnable_loss(logits, targets).item():.4f}")
    
    # Simulate training
    optimizer = torch.optim.Adam(learnable_loss.parameters(), lr=0.5)
    print("\n   Training for 50 steps...")
    
    losses = []
    for step in range(50):
        logits_train = torch.randn(B, C, H, W)  # Fresh random logits each step
        targets_train = torch.multinomial(torch.tensor(probs), total_pixels, replacement=True).view(B, H, W)
        
        optimizer.zero_grad()
        loss = learnable_loss(logits_train, targets_train)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    
    final_weights = learnable_loss.get_weights()
    print(f"   Final weights: {[f'{w:.3f}' for w in final_weights.tolist()]}")
    print(f"   Final loss: {losses[-1]:.4f}")
    print(f"   BG weight: {final_weights[0].item():.3f}")
    print(f"   Avg FG weight: {final_weights[1:].mean().item():.3f}")
    print(f"   Rarest class (5) weight: {final_weights[5].item():.3f}")
    
    # ===== 4. Adaptive Learnable =====
    print("\n" + "=" * 70)
    print("4. ADAPTIVE LEARNABLE (batch-specific base + learned scale)")
    print("=" * 70)
    
    adaptive_loss = AdaptiveLearnableLoss(C)
    
    print(f"   Initial scales: {[f'{s:.3f}' for s in adaptive_loss.get_scales().tolist()]}")
    
    optimizer = torch.optim.Adam(adaptive_loss.parameters(), lr=0.5)
    print("\n   Training for 50 steps...")
    
    for step in range(50):
        logits_train = torch.randn(B, C, H, W)
        targets_train = torch.multinomial(torch.tensor(probs), total_pixels, replacement=True).view(B, H, W)
        
        optimizer.zero_grad()
        loss, weights = adaptive_loss(logits_train, targets_train)
        loss.backward()
        optimizer.step()
    
    print(f"   Final scales: {[f'{s:.3f}' for s in adaptive_loss.get_scales().tolist()]}")
    print(f"   Last batch weights: {[f'{w:.3f}' for w in weights.tolist()]}")
    
    # ===== 5. Dice + CE =====
    print("\n" + "=" * 70)
    print("5. DICE + CE COMBO (no class weights)")
    print("=" * 70)
    
    dice_ce = DiceCELoss(C, dice_weight=0.5)
    loss = dice_ce(logits, targets)
    print(f"   Loss: {loss.item():.4f}")
    print("   Advantage: Dice naturally handles imbalance through ratio")
    
    # ===== 6. Constrained Learnable =====
    print("\n" + "=" * 70)
    print("6. CONSTRAINED LEARNABLE (bounded [0.5, 3.0] + L2 reg)")
    print("=" * 70)
    
    constrained_loss = ConstrainedLearnableLoss(C, min_weight=0.5, max_weight=3.0, reg_weight=0.01)
    print(f"   Initial weights: {[f'{w:.2f}' for w in constrained_loss.get_weights().tolist()]}")
    
    optimizer = torch.optim.Adam(constrained_loss.parameters(), lr=0.1)
    print("   Training for 100 steps...")
    
    for step in range(100):
        logits_train = torch.randn(B, C, H, W)
        targets_train = torch.multinomial(torch.tensor(probs), total_pixels, replacement=True).view(B, H, W)
        
        optimizer.zero_grad()
        loss = constrained_loss(logits_train, targets_train)
        loss.backward()
        optimizer.step()
    
    final_weights = constrained_loss.get_weights()
    print(f"   Final weights: {[f'{w:.3f}' for w in final_weights.tolist()]}")
    print(f"   BG weight: {final_weights[0].item():.3f}")
    print(f"   Rarest (class 5) weight: {final_weights[5].item():.3f}")
    
    # ===== 7. Temperature-based =====
    print("\n" + "=" * 70)
    print("7. TEMPERATURE-BASED (1 learnable param!)")
    print("=" * 70)
    
    temp_loss = TemperatureWeightedLoss(C)
    print(f"   Initial temperature: {temp_loss.get_temperature().item():.3f}")
    
    optimizer = torch.optim.Adam(temp_loss.parameters(), lr=0.1)
    print("   Training for 100 steps...")
    
    for step in range(100):
        logits_train = torch.randn(B, C, H, W)
        targets_train = torch.multinomial(torch.tensor(probs), total_pixels, replacement=True).view(B, H, W)
        
        optimizer.zero_grad()
        loss, t, weights = temp_loss(logits_train, targets_train, return_info=True)
        loss.backward()
        optimizer.step()
    
    final_t = temp_loss.get_temperature().item()
    print(f"   Learned temperature: {final_t:.3f}")
    print(f"   Interpretation:")
    print(f"     t=0.0 → uniform weights")
    print(f"     t=1.0 → full sqrt(1/freq) weights")
    print(f"     t={final_t:.3f} → {(1-final_t)*100:.0f}% uniform + {final_t*100:.0f}% frequency-based")
    
    # ===== Summary =====
    print("\n" + "=" * 70)
    print("SUMMARY & RECOMMENDATION")
    print("=" * 70)
    
    print("""
    PROBLEM WITH CURRENT APPROACH:
    - Manual caps (bg=2.0, fg=5.0) are arbitrary
    - Weights are STATIC throughout training
    - Can't adapt to task-specific distributions
    - Requires hyperparameter tuning
    
    BEST ALTERNATIVES:
    
    Option 1: LEARNABLE WEIGHTS (Simple & Effective)
    - Add 10 learnable parameters (one per class)
    - Initialize to 1.0 (softplus space)
    - Let model learn optimal weights during training
    - Normalize to prevent gradient scale issues
    
    Option 2: ADAPTIVE LEARNABLE (Batch-Aware)
    - Compute batch-specific base weights (sqrt inverse freq)
    - Learn per-class SCALING factors
    - Adapts to varying batch compositions
    - Best for ARC's task diversity
    
    Option 3: DICE + CE COMBO
    - No class weights needed at all
    - Dice naturally handles imbalance
    - CE provides stable gradients
    - Used in nnU-Net (SOTA medical segmentation)
    
    RECOMMENDATION: Start with Option 1 (LearnableWeightedLoss)
    - Simplest to implement
    - Just 10 extra parameters
    - Model learns optimal weights
    - Can always fall back to current if worse
    """)
    
    return learnable_loss, adaptive_loss


if __name__ == "__main__":
    run_comparison()
