#!/usr/bin/env python
"""
Class-Balanced Focal Loss - The best approach for imbalanced segmentation.

This combines:
1. Class-Balanced weights (Cui et al. 2019) - theoretically grounded
2. Focal modulation (Lin et al. 2017) - dynamic per-pixel adjustment

No arbitrary caps, no learnable weights that collapse!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def stablemax(x, dim=-1):
    """Numerically stable softmax alternative."""
    x_max = x.max(dim=dim, keepdim=True)[0]
    x_shifted = x - x_max
    exp_x = torch.exp(x_shifted.clamp(-20, 20))
    return exp_x / (exp_x.sum(dim=dim, keepdim=True) + 1e-8)


class ClassBalancedFocalLoss(nn.Module):
    """
    Class-Balanced Focal Loss - combines best of both worlds.
    
    Advantages:
    - CB: No arbitrary caps, theoretically grounded weights
    - Focal: Self-adjusting based on model confidence
    - No learnable weights that can collapse
    
    Args:
        num_classes: Number of classes
        beta: Class-balanced beta parameter (default 0.9999)
        gamma: Focal loss gamma parameter (default 2.0)
        use_stablemax: Whether to use stablemax instead of softmax
    """
    def __init__(self, num_classes=10, beta=0.9999, gamma=2.0, use_stablemax=True):
        super().__init__()
        self.num_classes = num_classes
        self.beta = beta
        self.gamma = gamma
        self.use_stablemax = use_stablemax
    
    def forward(self, logits, targets):
        B, C, H, W = logits.shape
        
        if self.use_stablemax:
            probs = stablemax(logits.permute(0, 2, 3, 1).reshape(-1, C))
        else:
            probs = F.softmax(logits.permute(0, 2, 3, 1).reshape(-1, C), dim=-1)
        
        logprobs = torch.log(probs.clamp(min=1e-10))
        targets_flat = targets.view(-1)
        
        valid_mask = targets_flat >= 0
        if not valid_mask.any():
            return torch.tensor(0.0, device=logits.device)
        
        probs = probs[valid_mask]
        logprobs = logprobs[valid_mask]
        targets_valid = targets_flat[valid_mask]
        n_valid = len(targets_valid)
        
        # Class-balanced weights (Cui et al. 2019)
        class_counts = torch.bincount(targets_valid, minlength=C).float()
        effective_num = 1.0 - torch.pow(self.beta, class_counts)
        cb_weights = (1.0 - self.beta) / (effective_num + 1e-6)
        # Normalize to mean=1 for gradient scale consistency
        cb_weights = cb_weights / (cb_weights.mean() + 1e-6)
        
        # Focal modulation
        p_t = probs[range(n_valid), targets_valid]
        focal_weight = (1 - p_t) ** self.gamma
        
        # Combined weight
        pixel_weights = cb_weights[targets_valid] * focal_weight
        
        # Cross-entropy loss
        loss = -(pixel_weights * logprobs[range(n_valid), targets_valid]).mean()
        
        return loss


def test_cb_focal():
    """Test Class-Balanced Focal Loss."""
    print("=" * 70)
    print("CLASS-BALANCED FOCAL LOSS TEST")
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
    
    # Test the loss
    loss_fn = ClassBalancedFocalLoss(C, beta=0.9999, gamma=2.0)
    loss = loss_fn(logits, targets)
    
    print(f"\nLoss: {loss.item():.4f}")
    
    # Check gradients
    loss.backward()
    print(f"Gradients: norm={logits.grad.norm().item():.4f}")
    
    # Show internal weights (need to peek inside)
    with torch.no_grad():
        targets_flat = targets.view(-1)
        class_counts = torch.bincount(targets_flat, minlength=C).float()
        effective_num = 1.0 - torch.pow(0.9999, class_counts)
        cb_weights = (1.0 - 0.9999) / (effective_num + 1e-6)
        cb_weights = cb_weights / (cb_weights.mean() + 1e-6)
    
    print(f"\nCB Weights: {[f'{w:.2f}' for w in cb_weights.tolist()]}")
    print(f"BG weight: {cb_weights[0].item():.3f}")
    print(f"Rarest (class 5) weight: {cb_weights[5].item():.3f}")
    print(f"BG/Rarest ratio: {cb_weights[0].item() / cb_weights[5].item():.2f}x")
    
    # Compare to current approach
    print("\n" + "=" * 70)
    print("COMPARISON: Current vs CB-Focal")
    print("=" * 70)
    
    # Current approach weights
    class_freq = class_counts / total_pixels
    weights_current = torch.zeros(C)
    weights_current[0] = 2.0  # BG cap
    fg_freq = class_freq[1:]
    fg_raw = 1.0 / (fg_freq + 1e-6)
    fg_min, fg_max = fg_raw.min(), fg_raw.max()
    fg_scaled = 1.0 + 4.0 * (fg_raw - fg_min) / (fg_max - fg_min + 1e-6)
    weights_current[1:] = fg_scaled.clamp(0.1, 5.0)
    
    print("\nCurrent WeightedStablemaxLoss:")
    print(f"  Weights: {[f'{w:.2f}' for w in weights_current.tolist()]}")
    print(f"  Hyperparams: bg_cap=2.0, fg_cap=5.0, min_weight=0.1 (3 arbitrary params)")
    print(f"  Dynamic: NO (same weights for entire epoch)")
    
    print("\nProposed CB-Focal Loss:")
    print(f"  Weights: {[f'{w:.2f}' for w in cb_weights.tolist()]}")
    print(f"  Hyperparams: beta=0.9999, gamma=2.0 (2 well-established params)")
    print(f"  Dynamic: YES (focal adjusts per-pixel based on confidence)")
    
    print("\n" + "=" * 70)
    print("KEY ADVANTAGES OF CB-FOCAL")
    print("=" * 70)
    print("""
    1. NO ARBITRARY CAPS
       - Current: bg_cap=2.0, fg_cap=5.0 chosen by trial-and-error
       - CB-Focal: Weights derived from class statistics
    
    2. THEORETICALLY GROUNDED
       - Based on "effective number of samples" (Cui et al. 2019)
       - Accounts for overlap in sample representation
    
    3. DYNAMIC ADJUSTMENT
       - Focal: Down-weights EASY pixels automatically
       - Model focuses on HARD pixels it gets wrong
       - Adapts as training progresses
    
    4. WELL-ESTABLISHED DEFAULTS
       - beta=0.9999: Standard for large datasets
       - gamma=2.0: Standard focal loss parameter
       - Both have extensive literature support
    
    5. NO COLLAPSE RISK
       - Learnable weights can collapse to minimize loss
       - CB-Focal is computed from data, can't be "gamed"
    """)
    
    print("=" * 70)
    print("RECOMMENDATION: Replace WeightedStablemaxLoss with ClassBalancedFocalLoss")
    print("=" * 70)


class FocalStablemaxLoss(nn.Module):
    """
    Focal Loss with stablemax - keeps current weighting philosophy, adds focal dynamic.
    
    This is the RECOMMENDED approach:
    - Keeps our BG/FG balance philosophy (BG gets cap, FG scaled)
    - ADDS focal modulation for dynamic adjustment
    - Model learns HARD pixels faster
    """
    def __init__(self, num_classes=10, bg_weight_cap=2.0, fg_weight_cap=5.0, 
                 gamma=2.0, use_stablemax=True):
        super().__init__()
        self.num_classes = num_classes
        self.bg_weight_cap = bg_weight_cap
        self.fg_weight_cap = fg_weight_cap
        self.gamma = gamma
        self.use_stablemax = use_stablemax
    
    def forward(self, logits, targets):
        B, C, H, W = logits.shape
        
        if self.use_stablemax:
            probs = stablemax(logits.permute(0, 2, 3, 1).reshape(-1, C))
        else:
            probs = F.softmax(logits.permute(0, 2, 3, 1).reshape(-1, C), dim=-1)
        
        logprobs = torch.log(probs.clamp(min=1e-10))
        targets_flat = targets.view(-1)
        
        valid_mask = targets_flat >= 0
        if not valid_mask.any():
            return torch.tensor(0.0, device=logits.device)
        
        probs = probs[valid_mask]
        logprobs = logprobs[valid_mask]
        targets_valid = targets_flat[valid_mask]
        n_valid = len(targets_valid)
        
        # Current weighting philosophy
        class_counts = torch.bincount(targets_valid, minlength=C).float()
        class_freq = class_counts / n_valid
        
        weights = torch.zeros(C, device=logits.device)
        weights[0] = self.bg_weight_cap  # BG fixed
        
        # FG: Scale inverse-freq to [1.0, fg_cap]
        fg_freq = class_freq[1:]
        fg_raw = 1.0 / (fg_freq + 1e-6)
        fg_min, fg_max = fg_raw.min(), fg_raw.max()
        fg_scaled = 1.0 + (self.fg_weight_cap - 1.0) * (fg_raw - fg_min) / (fg_max - fg_min + 1e-6)
        weights[1:] = fg_scaled.clamp(0.1, self.fg_weight_cap)
        
        # Focal modulation (the NEW part!)
        p_t = probs[range(n_valid), targets_valid]
        focal_weight = (1 - p_t) ** self.gamma
        
        # Combined weight
        pixel_weights = weights[targets_valid] * focal_weight
        
        # Cross-entropy loss
        loss = -(pixel_weights * logprobs[range(n_valid), targets_valid]).mean()
        
        return loss


def test_focal_stablemax():
    """Test our recommended approach: current weights + focal modulation."""
    print("\n" + "=" * 70)
    print("FOCAL STABLEMAX LOSS - RECOMMENDED HYBRID")
    print("=" * 70)
    
    torch.manual_seed(42)
    B, C, H, W = 8, 10, 20, 20
    total_pixels = B * H * W
    
    probs = [0.52, 0.10, 0.08, 0.05, 0.05, 0.02, 0.05, 0.05, 0.05, 0.03]
    targets = torch.multinomial(torch.tensor(probs), total_pixels, replacement=True).view(B, H, W)
    logits = torch.randn(B, C, H, W, requires_grad=True)
    
    loss_fn = FocalStablemaxLoss(C, bg_weight_cap=2.0, fg_weight_cap=5.0, gamma=2.0)
    
    # Initial loss
    loss = loss_fn(logits, targets)
    print(f"Initial loss: {loss.item():.4f}")
    
    # Simulate training - make some predictions better
    print("\nSimulating training (making some pixels easier)...")
    logits_improved = logits.clone().detach()
    # Make BG predictions better (model learns BG first)
    bg_mask = targets == 0
    logits_improved[:, 0, :, :][bg_mask] += 3.0  # Boost BG logit where target is BG
    logits_improved.requires_grad_(True)
    
    loss_after = loss_fn(logits_improved, targets)
    print(f"Loss after improvement: {loss_after.item():.4f}")
    
    # Check what focal did
    with torch.no_grad():
        # Get p_t for BG pixels
        probs_orig = stablemax(logits.permute(0, 2, 3, 1).reshape(-1, C))
        probs_impr = stablemax(logits_improved.permute(0, 2, 3, 1).reshape(-1, C))
        
        targets_flat = targets.view(-1)
        bg_indices = (targets_flat == 0).nonzero()[:, 0]
        
        p_t_orig_bg = probs_orig[bg_indices, 0].mean()
        p_t_impr_bg = probs_impr[bg_indices, 0].mean()
        
        focal_orig = (1 - p_t_orig_bg) ** 2.0
        focal_impr = (1 - p_t_impr_bg) ** 2.0
    
    print(f"\nFocal effect on BG pixels:")
    print(f"  Original p(BG|BG): {p_t_orig_bg:.3f} → focal weight: {focal_orig:.3f}")
    print(f"  Improved p(BG|BG): {p_t_impr_bg:.3f} → focal weight: {focal_impr:.3f}")
    print(f"  Reduction: {100*(1 - focal_impr/focal_orig):.1f}% less focus on easy BG pixels!")
    
    print("\n" + "-" * 70)
    print("KEY INSIGHT: Focal modulation is ADDITIVE to our philosophy")
    print("-" * 70)
    print("""
    What we keep:
    - BG gets 2.0 cap (maintains BG/FG gradient balance)
    - FG scales 1.0-5.0 (rare classes get more attention)
    
    What we add:
    - Focal modulation: (1 - p_t)^gamma per pixel
    - Early training: All pixels hard → all get focal_weight ≈ 1.0
    - Later training: Easy pixels → focal_weight → 0.0
    
    Result:
    - Model naturally focuses on HARD pixels
    - No collapse risk (weights aren't learnable)
    - BG/FG balance maintained
    - Dynamic adaptation throughout training
    """)


if __name__ == "__main__":
    test_cb_focal()
    test_focal_stablemax()
