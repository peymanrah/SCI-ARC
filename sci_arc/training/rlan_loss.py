"""
RLAN Loss Functions

Comprehensive loss module for training RLAN on ARC tasks.

Loss Components:
1. Focal Loss - Primary task loss with class imbalance handling
2. Stablemax Loss - Numerically stable alternative (from TRM)
3. Entropy Regularization - Encourage sharp attention maps
4. Sparsity Regularization - Encourage stopping early when possible
5. Predicate Diversity - Decorrelate predicate activations
6. Curriculum Penalty - Progressive complexity scheduling

The combined loss enables stable training while encouraging:
- Sharp, interpretable attention patterns
- Efficient use of clues (don't use more than needed)
- Diverse predicate representations
- Gradual increase in model capacity usage
"""

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def stablemax(x: torch.Tensor, epsilon: float = 1e-30) -> torch.Tensor:
    """
    Stablemax activation function (from TRM).
    
    More numerically stable than softmax for extreme values.
    s(x) = 1/(1-x) for x < 0, else x + 1
    
    Args:
        x: Input tensor
        epsilon: Small constant for stability
        
    Returns:
        Stablemax activations (always positive)
    """
    # Clamp input to prevent extreme values that could cause overflow
    x = x.clamp(min=-1000, max=1000)
    
    # Handle NaN/Inf inputs by replacing with 0
    x = torch.where(torch.isfinite(x), x, torch.zeros_like(x))
    
    result = torch.where(
        x < 0,
        1 / (1 - x + epsilon),
        x + 1
    )
    
    # Clamp output to ensure positivity and prevent overflow
    result = result.clamp(min=1e-10, max=1e10)
    
    return result


def log_stablemax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Log stablemax for stable cross entropy computation.
    
    Uses log-space computation for numerical stability:
    log(s_i / sum(s)) = log(s_i) - log(sum(s))
    
    Args:
        x: Input logits
        dim: Dimension to normalize over
        
    Returns:
        Log probabilities using stablemax normalization
    """
    # Clamp input first to prevent extreme values
    x = x.clamp(min=-1000, max=1000)
    
    # Handle NaN/Inf inputs by replacing with 0
    x = torch.where(torch.isfinite(x), x, torch.zeros_like(x))
    
    s_x = stablemax(x)
    
    # Clamp stablemax output to ensure positivity
    s_x = s_x.clamp(min=1e-10)
    
    # Use log-space computation for stability
    log_s_x = torch.log(s_x)
    log_sum_s = torch.log(torch.sum(s_x, dim=dim, keepdim=True) + 1e-10)
    log_probs = log_s_x - log_sum_s
    
    # Clamp output to reasonable range
    log_probs = log_probs.clamp(min=-100, max=0)
    
    # Final NaN check - replace any NaN with -100 (equivalent to ~0 probability)
    log_probs = torch.where(torch.isfinite(log_probs), log_probs, torch.full_like(log_probs, -100.0))
    
    return log_probs


class StablemaxCrossEntropy(nn.Module):
    """
    Stablemax Cross Entropy Loss (from TRM).
    
    Uses stablemax instead of softmax for better numerical stability.
    Particularly useful when logits can have extreme values.
    
    Args:
        reduction: 'mean', 'sum', or 'none'
        ignore_index: Label to ignore
    """
    
    def __init__(
        self,
        reduction: str = "mean",
        ignore_index: int = -100,
    ):
        super().__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute stablemax cross entropy loss.
        
        Args:
            logits: Shape (B, C, H, W) unnormalized logits
            targets: Shape (B, H, W) target class indices
            
        Returns:
            loss: Scalar or per-pixel loss
        """
        B, C, H, W = logits.shape
        
        # Flatten for loss computation
        logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)
        targets_flat = targets.reshape(-1)  # (B*H*W,)
        
        # Create mask for valid positions
        valid_mask = targets_flat != self.ignore_index
        
        if not valid_mask.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        # Filter to valid positions
        logits_valid = logits_flat[valid_mask]
        targets_valid = targets_flat[valid_mask]
        
        # Compute log stablemax probabilities (use float64 for precision)
        logprobs = log_stablemax(logits_valid.to(torch.float64), dim=-1)
        
        # Gather log probs for target classes
        prediction_logprobs = torch.gather(
            logprobs,
            index=targets_valid.unsqueeze(-1).to(torch.long),
            dim=-1
        ).squeeze(-1)
        
        # Negative log likelihood
        loss = -prediction_logprobs.to(logits.dtype)
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            full_loss = torch.zeros(B * H * W, device=logits.device, dtype=logits.dtype)
            full_loss[valid_mask] = loss
            return full_loss.view(B, H, W)


class WeightedStablemaxLoss(nn.Module):
    """
    Weighted Cross-Entropy Loss with Stablemax and DYNAMIC class weights.
    
    Unlike focal loss which down-weights "easy" examples (causing vanishing gradients
    when the model is confident), this uses INVERSE FREQUENCY WEIGHTING computed
    per-batch to always give minority classes strong gradients.
    
    For ARC-AGI (10 classes, no boundary markers):
    - Class 0 = Color 0 (black/background, low weight)
    - Classes 1-9 = Colors 1-9 (foreground, high weight)
    
    Args:
        bg_weight_cap: Maximum weight for background class
        fg_weight_cap: Maximum weight for foreground classes  
        min_class_weight: Minimum weight for any class (prevents zero weights)
        reduction: 'mean', 'sum', or 'none'
        ignore_index: Label to ignore (-100 for padding)
    """
    
    def __init__(
        self,
        bg_weight_cap: float = 1.0,
        fg_weight_cap: float = 10.0,
        min_class_weight: float = 0.1,
        reduction: str = "mean",
        ignore_index: int = -100,
    ):
        super().__init__()
        self.bg_weight_cap = bg_weight_cap
        self.fg_weight_cap = fg_weight_cap
        self.min_class_weight = min_class_weight
        self.reduction = reduction
        self.ignore_index = ignore_index
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            logits: Shape (B, C, H, W) unnormalized logits
            targets: Shape (B, H, W) target class indices
            
        Returns:
            loss: Scalar or per-pixel loss
        """
        B, C, H, W = logits.shape
        
        # Flatten for loss computation
        logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)
        targets_flat = targets.reshape(-1)  # (B*H*W,)
        
        # Create mask for valid positions
        valid_mask = targets_flat != self.ignore_index
        
        if not valid_mask.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        # Filter to valid positions
        logits_valid = logits_flat[valid_mask]
        targets_valid = targets_flat[valid_mask]
        n_valid = targets_valid.numel()
        
        # SAFETY CHECK: Clamp target values to valid range [0, C-1]
        # This prevents CUDA scatter/gather index out of bounds errors if
        # targets contain values >= C (e.g., from TRM encoding mismatch)
        if targets_valid.max() >= C or targets_valid.min() < 0:
            # Log warning for debugging (but don't crash)
            import warnings
            warnings.warn(
                f"Target values out of range! min={targets_valid.min().item()}, "
                f"max={targets_valid.max().item()}, expected [0, {C-1}]. "
                f"Check num_classes matches dataset encoding."
            )
            targets_valid = targets_valid.clamp(0, C - 1)
        
        # Compute DYNAMIC class weights based on batch statistics
        # Count each class in this batch
        class_counts = torch.bincount(targets_valid, minlength=C).float()  # (C,)
        
        # Inverse frequency weighting: rarer classes get higher weight
        # Add epsilon to avoid division by zero
        class_freq = class_counts / (n_valid + 1e-6)
        
        # BALANCED GRADIENT APPROACH:
        # - BG (color 0) always gets bg_weight_cap to ensure strong gradient
        # - FG (colors 1-9) use inverse-freq rescaled to [1.0, fg_weight_cap]
        # This gives roughly balanced total gradient between BG and FG categories
        
        weights = torch.zeros(C, device=logits.device, dtype=logits.dtype)
        
        # BG: fixed at bg_weight_cap (strong, not crushed by normalization)
        weights[0] = self.bg_weight_cap
        
        # FG: inverse frequency rescaled to range [1.0, fg_weight_cap]
        if C > 1:
            fg_freq = class_freq[1:]  # FG class frequencies
            fg_raw = 1.0 / (fg_freq + 1e-6)  # Inverse frequency
            fg_min = fg_raw.min()
            fg_max = fg_raw.max()
            # Rescale to [1.0, fg_weight_cap] - more frequent FG=1.0, rarest=fg_cap
            if fg_max > fg_min:
                fg_scaled = 1.0 + (self.fg_weight_cap - 1.0) * (fg_raw - fg_min) / (fg_max - fg_min + 1e-6)
            else:
                # All FG classes have same frequency
                fg_scaled = torch.ones_like(fg_raw) * (1.0 + self.fg_weight_cap) / 2.0
            weights[1:] = fg_scaled.clamp(self.min_class_weight, self.fg_weight_cap)
        
        # Get per-pixel weights
        pixel_weights = weights[targets_valid]
        
        # Compute log stablemax probabilities
        logprobs = log_stablemax(logits_valid.to(torch.float64), dim=-1)
        logprobs = logprobs.to(logits.dtype)
        
        # Gather log probs for target classes
        ce_loss = -torch.gather(logprobs, 1, targets_valid.unsqueeze(1)).squeeze(1)
        ce_loss = ce_loss.clamp(max=100.0)  # Numerical stability
        
        # NaN safety: replace any NaN losses with 0 (skip those pixels)
        nan_mask = ~torch.isfinite(ce_loss)
        if nan_mask.any():
            import warnings
            num_nan = nan_mask.sum().item()
            warnings.warn(f"WeightedStablemaxLoss: {num_nan} NaN values detected in ce_loss, replacing with 0")
            ce_loss = torch.where(nan_mask, torch.zeros_like(ce_loss), ce_loss)
        
        # Apply class weights
        weighted_loss = pixel_weights * ce_loss
        
        # Final NaN check on weighted_loss
        weighted_nan_mask = ~torch.isfinite(weighted_loss)
        if weighted_nan_mask.any():
            import warnings
            num_nan = weighted_nan_mask.sum().item()
            warnings.warn(f"WeightedStablemaxLoss: {num_nan} NaN values in weighted_loss, replacing with 0")
            weighted_loss = torch.where(weighted_nan_mask, torch.zeros_like(weighted_loss), weighted_loss)
        
        if self.reduction == "mean":
            return weighted_loss.mean()
        elif self.reduction == "sum":
            return weighted_loss.sum()
        else:
            full_loss = torch.zeros(B * H * W, device=logits.device)
            full_loss[valid_mask] = weighted_loss
            return full_loss.view(B, H, W)


class FocalWeightedStablemaxLoss(nn.Module):
    """
    Focal Loss + Our Weight Philosophy + Stablemax.
    
    THE RECOMMENDED LOSS for RLAN training.
    
    Combines:
    1. Our weight philosophy: BG=cap, FG scaled 1.0-cap (maintains BG/FG gradient balance)
    2. Focal modulation: (1-p_t)^gamma per pixel (dynamic focus on hard pixels)
    3. Stablemax: Numerically stable alternative to softmax
    
    Benefits over WeightedStablemaxLoss:
    - Same BG/FG gradient balance
    - PLUS: Dynamic adaptation - easy pixels get down-weighted as training progresses
    - Early training: All pixels hard → focal≈1.0 → all get attention
    - Later training: Easy pixels → focal→0 → focus on hard pixels
    
    Args:
        bg_weight_cap: Weight for background class (default 2.0)
        fg_weight_cap: Maximum weight for foreground classes (default 5.0)
        gamma: Focal loss focusing parameter (default 2.0)
        min_class_weight: Minimum weight for any class
        reduction: 'mean', 'sum', or 'none'
        ignore_index: Label to ignore
    """
    
    def __init__(
        self,
        bg_weight_cap: float = 2.0,
        fg_weight_cap: float = 5.0,
        gamma: float = 2.0,
        min_class_weight: float = 0.1,
        reduction: str = "mean",
        ignore_index: int = -100,
    ):
        super().__init__()
        self.bg_weight_cap = bg_weight_cap
        self.fg_weight_cap = fg_weight_cap
        self.gamma = gamma
        self.min_class_weight = min_class_weight
        self.reduction = reduction
        self.ignore_index = ignore_index
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            logits: Shape (B, C, H, W) unnormalized logits
            targets: Shape (B, H, W) target class indices
            
        Returns:
            loss: Scalar or per-pixel loss
        """
        B, C, H, W = logits.shape
        
        # Flatten for loss computation
        logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)
        targets_flat = targets.reshape(-1)  # (B*H*W,)
        
        # Create mask for valid positions
        valid_mask = targets_flat != self.ignore_index
        
        if not valid_mask.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        # Filter to valid positions
        logits_valid = logits_flat[valid_mask]
        targets_valid = targets_flat[valid_mask]
        n_valid = targets_valid.numel()
        
        # SAFETY CHECK: Clamp target values to valid range [0, C-1]
        if targets_valid.max() >= C or targets_valid.min() < 0:
            import warnings
            warnings.warn(
                f"Target values out of range! min={targets_valid.min().item()}, "
                f"max={targets_valid.max().item()}, expected [0, {C-1}]."
            )
            targets_valid = targets_valid.clamp(0, C - 1)
        
        # ========== PART 1: Class weights (our philosophy) ==========
        class_counts = torch.bincount(targets_valid, minlength=C).float()
        class_freq = class_counts / (n_valid + 1e-6)
        
        weights = torch.zeros(C, device=logits.device, dtype=logits.dtype)
        weights[0] = self.bg_weight_cap  # BG fixed
        
        # FG: inverse frequency rescaled to [1.0, fg_weight_cap]
        if C > 1:
            fg_freq = class_freq[1:]
            fg_raw = 1.0 / (fg_freq + 1e-6)
            fg_min = fg_raw.min()
            fg_max = fg_raw.max()
            if fg_max > fg_min:
                fg_scaled = 1.0 + (self.fg_weight_cap - 1.0) * (fg_raw - fg_min) / (fg_max - fg_min + 1e-6)
            else:
                fg_scaled = torch.ones_like(fg_raw) * (1.0 + self.fg_weight_cap) / 2.0
            weights[1:] = fg_scaled.clamp(self.min_class_weight, self.fg_weight_cap)
        
        # ========== PART 2: Stablemax probabilities ==========
        # Use stablemax for numerical stability
        s_x = stablemax(logits_valid.to(torch.float64))
        probs = s_x / s_x.sum(dim=-1, keepdim=True)
        probs = probs.to(logits.dtype).clamp(min=1e-7, max=1.0 - 1e-7)
        
        # ========== PART 3: Focal modulation (dynamic per-pixel) ==========
        p_t = probs[range(n_valid), targets_valid]  # Prob of true class
        focal_weight = (1 - p_t) ** self.gamma  # Down-weight easy pixels
        
        # ========== PART 4: Combined loss ==========
        pixel_weights = weights[targets_valid] * focal_weight
        
        # Log probabilities for CE
        logprobs = log_stablemax(logits_valid.to(torch.float64), dim=-1).to(logits.dtype)
        ce_loss = -torch.gather(logprobs, 1, targets_valid.unsqueeze(1)).squeeze(1)
        ce_loss = ce_loss.clamp(max=100.0)  # Numerical stability
        
        # NaN safety
        nan_mask = ~torch.isfinite(ce_loss)
        if nan_mask.any():
            import warnings
            warnings.warn(f"FocalWeightedStablemaxLoss: {nan_mask.sum().item()} NaN values, replacing with 0")
            ce_loss = torch.where(nan_mask, torch.zeros_like(ce_loss), ce_loss)
        
        weighted_loss = pixel_weights * ce_loss
        
        # Final NaN check
        weighted_nan_mask = ~torch.isfinite(weighted_loss)
        if weighted_nan_mask.any():
            import warnings
            warnings.warn(f"FocalWeightedStablemaxLoss: {weighted_nan_mask.sum().item()} NaN in weighted_loss")
            weighted_loss = torch.where(weighted_nan_mask, torch.zeros_like(weighted_loss), weighted_loss)
        
        if self.reduction == "mean":
            return weighted_loss.mean()
        elif self.reduction == "sum":
            return weighted_loss.sum()
        else:
            full_loss = torch.zeros(B * H * W, device=logits.device)
            full_loss[valid_mask] = weighted_loss
            return full_loss.view(B, H, W)


class FocalStablemaxLoss(nn.Module):
    """
    Focal Loss with Stablemax denominator (LEGACY - use FocalWeightedStablemaxLoss instead).
    
    Combines the class-balancing benefits of Focal Loss
    with the numerical stability of Stablemax.
    
    Best of both worlds for ARC training.
    
    Args:
        gamma: Focusing parameter
        alpha: Class weight for foreground
        reduction: 'mean', 'sum', or 'none'
        ignore_index: Label to ignore
    """
    
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.25,
        reduction: str = "mean",
        ignore_index: int = -100,
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.ignore_index = ignore_index
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute focal stablemax loss.
        """
        B, C, H, W = logits.shape
        
        logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, C)
        targets_flat = targets.reshape(-1)
        
        valid_mask = targets_flat != self.ignore_index
        
        if not valid_mask.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        logits_valid = logits_flat[valid_mask]
        targets_valid = targets_flat[valid_mask]
        
        # Use stablemax for probabilities
        s_x = stablemax(logits_valid.to(torch.float64))
        probs = s_x / s_x.sum(dim=-1, keepdim=True)
        probs = probs.to(logits.dtype)
        
        # Get target probabilities
        target_probs = probs.gather(1, targets_valid.unsqueeze(1)).squeeze(1)
        target_probs = target_probs.clamp(min=1e-7, max=1.0 - 1e-7)
        
        # Focal weight: down-weight easy examples (high confidence)
        # gamma=2.0 means: p=0.9 (easy) -> weight=0.01, p=0.5 (hard) -> weight=0.25
        focal_weight = (1 - target_probs) ** self.gamma
        
        # Alpha weight: CRITICAL for anti-background-collapse
        # Background (class 0) should have LOW weight, foreground (1-9) should have HIGH weight
        # With alpha=0.75: background=0.25, foreground=0.75 (3x more weight on foreground)
        # This forces the model to care more about getting foreground pixels right
        is_background = (targets_valid == 0).float()
        alpha_weight = is_background * (1 - self.alpha) + (1 - is_background) * self.alpha
        
        # Cross entropy using log stablemax
        logprobs = log_stablemax(logits_valid.to(torch.float64), dim=-1).to(logits.dtype)
        ce_loss = -torch.gather(logprobs, 1, targets_valid.unsqueeze(1)).squeeze(1)
        ce_loss = ce_loss.clamp(max=100.0)
        
        # Combined focal loss
        focal_loss = alpha_weight * focal_weight * ce_loss
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            full_loss = torch.zeros(B * H * W, device=logits.device)
            full_loss[valid_mask] = focal_loss
            return full_loss.view(B, H, W)


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in ARC grids.
    
    ARC grids are ~85% background (color 0), leading to trivial
    all-background predictions. Focal Loss down-weights easy examples
    (confident background predictions) to focus on hard examples.
    
    Formula: L_focal = -α(1-p)^γ log(p)
    
    Parameters:
        gamma: Focusing parameter (higher = more focus on hard examples)
        alpha: Class weight for positive class (lower for background)
    """
    
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.25,
        reduction: str = "mean",
        ignore_index: int = -100,
    ):
        """
        Args:
            gamma: Focusing parameter (typically 2.0)
            alpha: Weight for positive/foreground classes
            reduction: 'mean', 'sum', or 'none'
            ignore_index: Label to ignore in loss computation
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.ignore_index = ignore_index
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            logits: Shape (B, C, H, W) unnormalized logits
            targets: Shape (B, H, W) target class indices
            
        Returns:
            loss: Scalar or per-pixel loss depending on reduction
        """
        B, C, H, W = logits.shape
        
        # Flatten for loss computation
        logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)
        targets_flat = targets.reshape(-1)  # (B*H*W,)
        
        # Create mask for valid positions
        valid_mask = targets_flat != self.ignore_index
        
        if not valid_mask.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        # Filter to valid positions
        logits_valid = logits_flat[valid_mask]
        targets_valid = targets_flat[valid_mask]
        
        # Compute softmax probabilities
        probs = F.softmax(logits_valid, dim=-1)
        
        # Get probability of correct class (clamp for numerical stability)
        target_probs = probs.gather(1, targets_valid.unsqueeze(1)).squeeze(1)
        target_probs = target_probs.clamp(min=1e-7, max=1.0 - 1e-7)
        
        # Compute focal weight: (1 - p)^gamma
        focal_weight = (1 - target_probs) ** self.gamma
        
        # Compute alpha weight (lower for background class 0)
        is_background = (targets_valid == 0).float()
        alpha_weight = is_background * (1 - self.alpha) + (1 - is_background) * self.alpha
        
        # Compute cross entropy with label smoothing for stability
        ce_loss = F.cross_entropy(logits_valid, targets_valid, reduction='none')
        
        # Clamp CE loss to prevent extreme values
        ce_loss = ce_loss.clamp(max=100.0)
        
        # Apply focal and alpha weights
        focal_loss = alpha_weight * focal_weight * ce_loss
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            # Reshape back to (B, H, W) with zeros for invalid positions
            full_loss = torch.zeros(B * H * W, device=logits.device)
            full_loss[valid_mask] = focal_loss
            return full_loss.view(B, H, W)


class EntropyRegularization(nn.Module):
    """
    Entropy regularization for attention maps.
    
    Encourages sharp, focused attention rather than diffuse attention.
    Low entropy = sharp (good), High entropy = diffuse (bad)
    
    Formula: L_entropy = Σ_k H(attention_k) where H(p) = -Σ p log(p)
    """
    
    def __init__(self, target_entropy: float = 0.0):
        """
        Args:
            target_entropy: Target entropy value (0 = perfectly sharp)
        """
        super().__init__()
        self.target_entropy = target_entropy
    
    def forward(self, attention_maps: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy regularization loss.
        
        Args:
            attention_maps: Shape (B, K, H, W) attention weights
            
        Returns:
            loss: Scalar entropy loss
        """
        B, K, H, W = attention_maps.shape
        
        # Flatten spatial dimensions
        attention_flat = attention_maps.view(B, K, -1)  # (B, K, H*W)
        
        # Clamp attention for numerical stability
        attention_clamped = attention_flat.clamp(min=1e-10, max=1.0)
        
        # Compute entropy for each attention map
        # H(p) = -Σ p log(p)
        log_attention = torch.log(attention_clamped)
        entropy = -(attention_clamped * log_attention).sum(dim=-1)  # (B, K)
        
        # Average over clues and batch
        mean_entropy = entropy.mean()
        
        # Loss is distance from target entropy
        loss = (mean_entropy - self.target_entropy).abs()
        
        # Clamp to prevent extreme values
        loss = loss.clamp(max=10.0)
        
        return loss


class SparsityRegularization(nn.Module):
    """
    Clue usage regularization with TWO-SIDED penalty + ENTROPY-WEIGHTED pondering.
    
    CRITICAL INSIGHT: The model needs to learn WHEN to stop AND find GOOD clues.
    
    Three-component penalty:
    1. min_clue_penalty: Prevents collapse to 0 clues
    2. pondering_cost: Base cost per clue (encourages efficiency)  
    3. entropy_pondering: Extra cost for low-quality (high entropy) clues
    
    The entropy-weighted component couples attention quality to clue usage:
    - Sharp attention (low entropy) → small extra cost
    - Diffuse attention (high entropy) → large extra cost
    - This pushes model to find GOOD clues, not just any clues
    
    Formula: 
        L = min_clue_penalty + base_pondering + entropy_pondering
        base_pondering = expected_clues_used * ponder_weight
        entropy_pondering = mean_entropy * entropy_ponder_weight
        
    Learning dynamics:
    - Early training: entropy is high, but that's OK (learning to attend)
    - Mid training: entropy drops, pondering cost dominates (learn when to stop)
    - Late training: sharp attention, efficient clue usage (task-optimal)
    """
    
    def __init__(
        self, 
        min_clues: float = 1.0, 
        ponder_weight: float = 0.1,
        min_clue_weight: float = 1.0,
        entropy_ponder_weight: float = 0.05,  # Extra cost for diffuse attention
        epsilon: float = 1e-6
    ):
        """
        Args:
            min_clues: Minimum expected number of clues to use (soft target)
            ponder_weight: Base cost per clue used (ACT-style)
            min_clue_weight: Weight for minimum clue penalty
            entropy_ponder_weight: Extra cost for high-entropy (diffuse) attention
            epsilon: Small constant for numerical stability
        """
        super().__init__()
        self.min_clues = min_clues
        self.ponder_weight = ponder_weight
        self.min_clue_weight = min_clue_weight
        self.entropy_ponder_weight = entropy_ponder_weight
        self.epsilon = epsilon
    
    def forward(
        self, 
        stop_logits: torch.Tensor,
        attention_maps: Optional[torch.Tensor] = None,
        return_per_sample: bool = False,
    ) -> torch.Tensor:
        """
        Compute clue usage regularization loss.
        
        Args:
            stop_logits: Shape (B, K) stop probability logits
            attention_maps: Shape (B, K, H, W) attention weights (optional)
            return_per_sample: If True, return per-sample min_clue_penalty (B,) separately
                              for adding to per-sample task loss. This enables per-task
                              clue count learning where hard tasks use more clues.
            
        Returns:
            If return_per_sample=False: Scalar regularization loss
            If return_per_sample=True: Tuple of (scalar_loss, per_sample_penalty (B,))
        """
        # Convert to probabilities
        stop_probs = torch.sigmoid(stop_logits)  # (B, K)
        
        # Expected clues used = sum of (1 - stop_prob) across clues
        expected_clues_used = (1 - stop_probs).sum(dim=-1)  # (B,)
        
        # 1. Penalty for using fewer than min_clues (hinge loss)
        # Keep per-sample for per-task gradient flow
        min_clue_penalty_per_sample = F.relu(self.min_clues - expected_clues_used)  # (B,)
        min_clue_penalty = min_clue_penalty_per_sample.mean()  # Scalar for backward compat
        
        # 2. Base pondering cost: small cost per clue used
        base_pondering = expected_clues_used.mean() * self.ponder_weight
        
        # 3. Entropy-weighted pondering (if attention maps provided)
        entropy_pondering = torch.tensor(0.0, device=stop_logits.device)
        if attention_maps is not None and self.entropy_ponder_weight > 0:
            B, K, H, W = attention_maps.shape
            
            # Compute per-clue entropy
            attn_flat = attention_maps.view(B, K, -1).clamp(min=1e-10)  # (B, K, H*W)
            per_clue_entropy = -(attn_flat * torch.log(attn_flat)).sum(dim=-1)  # (B, K)
            
            # Normalize by max entropy for stable values [0, 1]
            max_entropy = math.log(H * W + 1e-6)
            per_clue_entropy_norm = per_clue_entropy / max_entropy  # (B, K)
            
            # Weight entropy by clue usage probability
            # Clues that are "used" (low stop prob) contribute more
            clue_usage_weight = 1 - stop_probs  # (B, K)
            weighted_entropy = (per_clue_entropy_norm * clue_usage_weight).sum(dim=-1)  # (B,)
            
            entropy_pondering = weighted_entropy.mean() * self.entropy_ponder_weight
        
        # Combined loss (excluding per-sample penalty which is returned separately if requested)
        # base_pondering and entropy_pondering are batch-level regularization
        total_loss = (
            self.min_clue_weight * min_clue_penalty 
            + base_pondering 
            + entropy_pondering
        )
        
        # Store component breakdown for diagnostics (can be retrieved after forward)
        self._last_components = {
            'min_clue_penalty': min_clue_penalty.item() if torch.is_tensor(min_clue_penalty) else min_clue_penalty,
            'base_pondering': base_pondering.item() if torch.is_tensor(base_pondering) else base_pondering,
            'entropy_pondering': entropy_pondering.item() if torch.is_tensor(entropy_pondering) else entropy_pondering,
            'expected_clues_used': expected_clues_used.mean().item(),
            'stop_prob_mean': stop_probs.mean().item(),
            # NEW: Track per-sample variance to verify task-dependent clue count
            'stop_prob_std': stop_probs.mean(dim=-1).std().item(),  # Std of mean stop_prob per sample
            'clues_used_std': (1 - stop_probs).sum(dim=-1).std().item(),  # Std of clues used per sample
        }
        
        # Return per-sample penalty for per-task gradient flow
        if return_per_sample:
            # Per-sample penalty weighted by min_clue_weight for adding to task loss
            per_sample_penalty = self.min_clue_weight * min_clue_penalty_per_sample  # (B,)
            # Return scalar loss (base + entropy) + per-sample penalty separately
            scalar_loss = base_pondering + entropy_pondering
            return scalar_loss, per_sample_penalty
        
        return total_loss
    
    def get_last_components(self) -> dict:
        """Get breakdown of last sparsity loss computation for diagnostics."""
        return getattr(self, '_last_components', {})


class PredicateDiversityLoss(nn.Module):
    """
    Predicate diversity loss to decorrelate predicates.
    
    Encourages different predicates to capture different properties
    by minimizing their correlation across the batch.
    
    Formula: L_pred = ||P^T P - I||_F^2 / P^2
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, predicates: torch.Tensor) -> torch.Tensor:
        """
        Compute predicate diversity loss.
        
        Args:
            predicates: Shape (B, P) predicate activations
            
        Returns:
            loss: Scalar diversity loss
        """
        B, P = predicates.shape
        
        if B < 2:
            return torch.tensor(0.0, device=predicates.device)
        
        # Center predicates
        predicates_centered = predicates - predicates.mean(dim=0, keepdim=True)
        
        # Compute correlation matrix
        corr = torch.mm(predicates_centered.T, predicates_centered) / (B - 1)
        
        # Normalize by variance
        var = torch.diag(corr) + 1e-6
        std = torch.sqrt(var)
        corr_normalized = corr / (std.unsqueeze(0) * std.unsqueeze(1))
        
        # Off-diagonal elements should be zero
        identity = torch.eye(P, device=predicates.device)
        off_diagonal = corr_normalized * (1 - identity)
        
        # Frobenius norm of off-diagonal
        diversity_loss = (off_diagonal ** 2).sum() / (P * (P - 1) + 1e-6)
        
        # Check for NaN and clamp
        if torch.isnan(diversity_loss):
            return torch.tensor(0.0, device=predicates.device, requires_grad=True)
        
        return diversity_loss.clamp(max=10.0)


class CurriculumPenalty(nn.Module):
    """
    Curriculum-based complexity penalty.
    
    Early in training: Penalize using many clues (force simple solutions)
    Late in training: Allow full complexity
    
    This helps the model learn simple patterns first before complex ones.
    """
    
    def __init__(self, max_clues: int = 5):
        """
        Args:
            max_clues: Maximum number of clues (K)
        """
        super().__init__()
        self.max_clues = max_clues
    
    def forward(
        self,
        stop_logits: torch.Tensor,
        epoch: int,
        max_epochs: int,
    ) -> torch.Tensor:
        """
        Compute curriculum penalty.
        
        Args:
            stop_logits: Shape (B, K) stop probability logits
            epoch: Current epoch number
            max_epochs: Total number of epochs
            
        Returns:
            loss: Scalar curriculum penalty
        """
        B, K = stop_logits.shape
        
        # Compute expected number of active clues
        stop_probs = torch.sigmoid(stop_logits)
        continue_probs = 1 - stop_probs
        
        # Probability of reaching clue k = product of (1 - stop_prob) for all previous
        reach_probs = torch.cumprod(
            torch.cat([torch.ones(B, 1, device=stop_logits.device), continue_probs[:, :-1]], dim=1),
            dim=1
        )
        expected_clues = reach_probs.sum(dim=1).mean()  # Average over batch
        
        # Curriculum schedule: early training penalizes more clues
        # progress goes from 0 to 1 over training
        progress = epoch / max(max_epochs, 1)
        
        # Target number of clues increases over training
        # Start with 1-2 clues, end with full capacity
        target_clues = 1.0 + progress * (self.max_clues - 1)
        
        # Penalty for exceeding target
        excess = F.relu(expected_clues - target_clues)
        
        return excess


class RLANLoss(nn.Module):
    """
    Combined loss for RLAN training.
    
    Combines all loss components with configurable weights:
    - Task Loss: StablemaxCrossEntropy (TRM-style) or FocalStablemaxLoss
    - Entropy Regularization: Sharp attention
    - Sparsity Regularization: Efficient clue usage (two-sided penalty)
    - Predicate Diversity: Decorrelated predicates
    - Curriculum Penalty: Progressive complexity
    
    Loss Mode Options:
    - 'stablemax': Pure stablemax cross-entropy (TRM uses this)
    - 'weighted_stablemax': Inverse frequency weighting (BEST for ARC)
    - 'focal_stablemax': Focal loss + stablemax (original RLAN)
    - 'focal': Standard focal loss with softmax
    """
    
    def __init__(
        self,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        lambda_entropy: float = 0.1,
        lambda_sparsity: float = 0.05,
        lambda_predicate: float = 0.01,
        lambda_curriculum: float = 0.1,
        lambda_deep_supervision: float = 0.5,
        lambda_act: float = 0.1,  # Weight for ACT pondering cost
        min_clues: float = 2.5,  # Minimum clues to use (increased from 1.0)
        min_clue_weight: float = 5.0,  # Strong penalty for using fewer than min_clues
        ponder_weight: float = 0.02,  # Base cost per clue (REDUCED from 0.1)
        entropy_ponder_weight: float = 0.02,  # Extra cost for diffuse attention (REDUCED)
        max_clues: int = 5,
        use_stablemax: bool = True,  # Stablemax for numerical stability
        loss_mode: str = 'focal_stablemax',  # 'stablemax', 'weighted_stablemax', 'focal_weighted', 'focal_stablemax', or 'focal'
        bg_weight_cap: float = 2.0,  # Max weight for BG in weighted losses
        fg_weight_cap: float = 5.0,  # Max weight for FG in weighted losses
    ):
        """
        Args:
            focal_gamma: Focal loss gamma parameter
            focal_alpha: Focal loss alpha parameter
            lambda_entropy: Weight for entropy regularization
            lambda_sparsity: Weight for clue usage regularization
            lambda_predicate: Weight for predicate diversity
            lambda_curriculum: Weight for curriculum penalty
            lambda_deep_supervision: Weight for intermediate step losses
            lambda_act: Weight for ACT halting loss
            min_clues: Minimum expected clues to use (soft target, default=2.5)
            min_clue_weight: Penalty weight for using fewer than min_clues (default=5.0)
            ponder_weight: Base cost per clue used (ACT-style, default=0.02)
            entropy_ponder_weight: Extra cost for high-entropy attention (default=0.02)
            max_clues: Maximum number of clues
            use_stablemax: DEPRECATED - use loss_mode instead
            loss_mode: Loss function mode:
                - 'stablemax': Pure stablemax CE (for numerical stability)
                - 'weighted_stablemax': Inverse frequency weighting
                - 'focal_weighted': RECOMMENDED - our weights + focal modulation
                - 'focal_stablemax': Focal loss + stablemax (legacy)
                - 'focal': Standard focal loss
            focal_gamma: Focal focusing parameter (higher = more focus on hard pixels)
            bg_weight_cap: Max weight for background in weighted losses (default=2.0)
            fg_weight_cap: Max weight for foreground in weighted losses (default=5.0)
                           Ratio of 5:2=2.5x FG emphasis (reduced from 10:1 to prevent collapse)
        """
        super().__init__()
        
        self.loss_mode = loss_mode
        self.ponder_weight = ponder_weight
        
        # Check if we're in minimal mode (all lambdas zero)
        self.minimal_mode = (
            lambda_entropy == 0.0 and 
            lambda_sparsity == 0.0 and 
            lambda_predicate == 0.0 and 
            lambda_curriculum == 0.0 and 
            lambda_deep_supervision == 0.0 and 
            lambda_act == 0.0
        )
        if self.minimal_mode:
            print("  [!] MINIMAL MODE: Only task loss active")
        self.entropy_ponder_weight = entropy_ponder_weight
        
        # Select loss function based on mode
        if loss_mode == 'stablemax':
            # Pure stablemax cross-entropy (numerically stable)
            self.task_loss = StablemaxCrossEntropy()
            print(f"  Loss Mode: STABLEMAX (pure cross-entropy)")
        elif loss_mode == 'weighted_stablemax':
            # Inverse frequency weighted stablemax
            # Use configurable caps to prevent BG/FG collapse
            self.task_loss = WeightedStablemaxLoss(
                bg_weight_cap=bg_weight_cap,
                fg_weight_cap=fg_weight_cap,
                min_class_weight=0.1,
            )
            print(f"  Loss Mode: WEIGHTED_STABLEMAX (inverse frequency, bg_cap={bg_weight_cap}, fg_cap={fg_weight_cap})")
        elif loss_mode == 'focal_weighted':
            # RECOMMENDED: Our weights + focal modulation (BEST for ARC)
            # Combines BG/FG balance with dynamic hard-pixel focusing
            self.task_loss = FocalWeightedStablemaxLoss(
                bg_weight_cap=bg_weight_cap,
                fg_weight_cap=fg_weight_cap,
                gamma=focal_gamma,
                min_class_weight=0.1,
            )
            print(f"  Loss Mode: FOCAL_WEIGHTED (bg_cap={bg_weight_cap}, fg_cap={fg_weight_cap}, gamma={focal_gamma}) [RECOMMENDED]")
        elif loss_mode == 'focal_stablemax':
            # Focal loss with stablemax (original RLAN)
            self.task_loss = FocalStablemaxLoss(gamma=focal_gamma, alpha=focal_alpha)
            print(f"  Loss Mode: FOCAL_STABLEMAX (gamma={focal_gamma}, alpha={focal_alpha})")
        elif loss_mode == 'focal':
            # Standard focal loss with softmax
            self.task_loss = FocalLoss(gamma=focal_gamma, alpha=focal_alpha)
            print(f"  Loss Mode: FOCAL (gamma={focal_gamma}, alpha={focal_alpha})")
        else:
            # Fallback to use_stablemax for backward compatibility
            if use_stablemax:
                self.task_loss = FocalStablemaxLoss(gamma=focal_gamma, alpha=focal_alpha)
            else:
                self.task_loss = FocalLoss(gamma=focal_gamma, alpha=focal_alpha)
        
        # Keep focal_loss as alias for backward compatibility
        self.focal_loss = self.task_loss
        
        self.entropy_reg = EntropyRegularization()
        self.sparsity_reg = SparsityRegularization(
            min_clues=min_clues,
            ponder_weight=ponder_weight,
            min_clue_weight=min_clue_weight,  # From config (strong penalty for < min_clues)
            entropy_ponder_weight=entropy_ponder_weight,  # Extra cost for diffuse attention
        )
        self.predicate_diversity = PredicateDiversityLoss()
        self.curriculum_penalty = CurriculumPenalty(max_clues=max_clues)
        
        # Print sparsity config for debugging
        print(f"  Clue Regularization: min_clues={min_clues}, min_clue_weight={min_clue_weight}, ponder_weight={ponder_weight}, entropy_weight={entropy_ponder_weight}")
        
        self.lambda_entropy = lambda_entropy
        self.lambda_sparsity = lambda_sparsity
        self.lambda_predicate = lambda_predicate
        self.lambda_curriculum = lambda_curriculum
        self.lambda_deep_supervision = lambda_deep_supervision
        self.lambda_act = lambda_act
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        attention_maps: torch.Tensor,
        stop_logits: torch.Tensor,
        predicates: torch.Tensor,
        epoch: int = 0,
        max_epochs: int = 250,
        all_logits: Optional[list] = None,
        act_outputs: Optional[dict] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all RLAN losses.
        
        Args:
            logits: Shape (B, C, H, W) final output logits
            targets: Shape (B, H, W) target class indices
            attention_maps: Shape (B, K, H, W) attention weights
            stop_logits: Shape (B, K) stop probability logits
            predicates: Shape (B, P) predicate activations
            epoch: Current epoch number
            max_epochs: Total number of epochs
            all_logits: Optional list of (B, C, H, W) for deep supervision
            act_outputs: Optional dict with ACT outputs for halt loss computation
                - 'q_halt_logits': List of (B,) Q-values for halting at each step
                - 'q_continue_logits': List of (B,) Q-values for continuing
            
        Returns:
            Dict with keys:
                - total_loss: Combined weighted loss
                - focal_loss: Task loss
                - entropy_loss: Attention sharpness loss
                - sparsity_loss: Clue efficiency loss
                - predicate_loss: Predicate diversity loss
                - curriculum_loss: Complexity penalty
                - deep_supervision_loss: Intermediate step losses (if all_logits provided)
                - act_loss: ACT halting loss (if act_outputs provided)
        """
        # FAST PATH: Minimal mode - just compute task loss
        if self.minimal_mode:
            task_loss = self.task_loss(logits, targets)
            zero = torch.tensor(0.0, device=logits.device)
            return {
                "total_loss": task_loss,
                "task_loss": task_loss,
                "focal_loss": task_loss,  # Backward compatibility
                "entropy_loss": zero,
                "sparsity_loss": zero,
                "predicate_loss": zero,
                "curriculum_loss": zero,
                "deep_supervision_loss": zero,
                "act_loss": zero,
                "loss_mode": self.loss_mode,
                "sparsity_min_clue_penalty": 0.0,
                "sparsity_base_pondering": 0.0,
                "sparsity_entropy_pondering": 0.0,
                "expected_clues_used": 0.0,
                "stop_prob_from_loss": 0.0,
                "clues_used_std": 0.0,
                "per_sample_clue_penalty_mean": 0.0,
            }
        
        # FULL PATH: All auxiliary losses
        # Regularization losses
        entropy_loss = self.entropy_reg(attention_maps)
        
        # Get per-sample clue penalty for per-task gradient flow
        # This enables task-dependent clue count: hard tasks use more clues
        sparsity_scalar, per_sample_clue_penalty = self.sparsity_reg(
            stop_logits, attention_maps, return_per_sample=True
        )
        
        # Scale per-sample penalty by lambda_sparsity to maintain backward compatibility
        # Before: total_loss += lambda_sparsity * (min_clue_weight * penalty + base + entropy)
        # After:  total_loss += (lambda_sparsity * min_clue_weight * penalty) + lambda_sparsity * (base + entropy)
        # The per_sample_clue_penalty already includes min_clue_weight, so just scale by lambda_sparsity
        per_sample_clue_penalty_scaled = self.lambda_sparsity * per_sample_clue_penalty
        
        # Compute per-sample task loss to add clue penalty per-sample
        # This is the key: each sample's clue penalty affects its own gradient
        B, C, H, W = logits.shape
        
        # NaN detection on input logits (helps debug source of NaN)
        if not torch.isfinite(logits).all():
            import warnings
            num_nan = (~torch.isfinite(logits)).sum().item()
            warnings.warn(f"RLANLoss: {num_nan} NaN/Inf values in input logits! Clamping...")
            logits = torch.where(torch.isfinite(logits), logits, torch.zeros_like(logits))
        
        # Temporarily change reduction to get per-pixel loss
        saved_reduction = self.task_loss.reduction
        self.task_loss.reduction = "none"
        per_pixel_task_loss = self.task_loss(logits, targets)  # (B, H, W)
        self.task_loss.reduction = saved_reduction
        
        # NaN check on per_pixel_task_loss
        if not torch.isfinite(per_pixel_task_loss).all():
            import warnings
            num_nan = (~torch.isfinite(per_pixel_task_loss)).sum().item()
            warnings.warn(f"RLANLoss: {num_nan} NaN/Inf in per_pixel_task_loss, replacing with 0")
            per_pixel_task_loss = torch.where(torch.isfinite(per_pixel_task_loss), per_pixel_task_loss, torch.zeros_like(per_pixel_task_loss))
        
        # Per-sample task loss (mean over spatial dims)
        per_sample_task_loss = per_pixel_task_loss.mean(dim=(1, 2))  # (B,)
        
        # Add per-sample clue penalty to per-sample task loss
        # Now the gradient of clue penalty flows per-sample!
        combined_per_sample_loss = per_sample_task_loss + per_sample_clue_penalty_scaled  # (B,)
        
        # Final task loss is mean of combined per-sample losses
        task_loss = combined_per_sample_loss.mean()
        
        # Sparsity loss is just the scalar components (base_pondering, entropy_pondering)
        # min_clue_penalty is already in task_loss via per_sample_clue_penalty_scaled
        sparsity_loss = sparsity_scalar
        predicate_loss = self.predicate_diversity(predicates)
        curriculum_loss = self.curriculum_penalty(stop_logits, epoch, max_epochs)
        
        # Deep supervision loss (if intermediate predictions available)
        if all_logits is not None and len(all_logits) > 1:
            ds_losses = []
            num_steps = len(all_logits)
            for t, step_logits in enumerate(all_logits[:-1]):  # Exclude final (already in task_loss)
                # CHANGED: Use uniform weighting instead of exponential
                # Previous: weight = 0.5 ** (num_steps - 1 - t) gave step 0 only 3% weight
                # This caused solver degradation: step 0 learned to be good but later steps
                # had weak incentive to maintain quality, causing drift.
                # 
                # With uniform weighting, ALL steps are equally penalized for errors,
                # encouraging the GRU to maintain or improve quality at each step.
                weight = 1.0  # Uniform weighting - all steps equally important
                step_loss = self.task_loss(step_logits, targets)
                ds_losses.append(weight * step_loss)
            deep_supervision_loss = sum(ds_losses) / len(ds_losses) if ds_losses else torch.tensor(0.0)
        else:
            deep_supervision_loss = torch.tensor(0.0, device=logits.device)
        
        # ACT loss (if ACT outputs available)
        # Uses per-step correctness as reward signal for Q-learning
        act_loss = torch.tensor(0.0, device=logits.device)
        if act_outputs is not None and 'q_halt_logits' in act_outputs:
            q_halt_list = act_outputs['q_halt_logits']  # List of (B,) tensors per step
            
            if all_logits is not None and len(all_logits) == len(q_halt_list):
                # Compute per-step correctness as reward signal
                # A step is "correct" if its prediction matches the target
                act_losses = []
                for t, (q_halt, step_logits) in enumerate(zip(q_halt_list, all_logits)):
                    # Per-pixel accuracy at this step
                    step_preds = step_logits.argmax(dim=1)  # (B, H, W)
                    step_correct = (step_preds == targets).float().mean(dim=(1, 2))  # (B,)
                    
                    # Q_halt should predict whether halting NOW gives correct answer
                    # BCE loss: q_halt should be high when step is correct
                    step_act_loss = F.binary_cross_entropy_with_logits(
                        q_halt, step_correct, reduction='mean'
                    )
                    act_losses.append(step_act_loss)
                
                act_loss = sum(act_losses) / len(act_losses) if act_losses else act_loss
        
        # Combine losses
        total_loss = (
            task_loss
            + self.lambda_entropy * entropy_loss
            + self.lambda_sparsity * sparsity_loss
            + self.lambda_predicate * predicate_loss
            + self.lambda_curriculum * curriculum_loss
            + self.lambda_deep_supervision * deep_supervision_loss
            + self.lambda_act * act_loss
        )
        
        # Get sparsity loss component breakdown for diagnostics
        sparsity_components = self.sparsity_reg.get_last_components()
        
        return {
            "total_loss": total_loss,
            "task_loss": task_loss,  # Accurate name (stablemax, focal_stablemax, or focal)
            "focal_loss": task_loss,  # Backward compatibility alias
            "entropy_loss": entropy_loss,
            "sparsity_loss": sparsity_loss,
            "predicate_loss": predicate_loss,
            "curriculum_loss": curriculum_loss,
            "deep_supervision_loss": deep_supervision_loss,
            "act_loss": act_loss,
            "loss_mode": self.loss_mode,  # Include mode for logging
            # Sparsity component breakdown (for debugging clue usage)
            "sparsity_min_clue_penalty": sparsity_components.get('min_clue_penalty', 0.0),
            "sparsity_base_pondering": sparsity_components.get('base_pondering', 0.0),
            "sparsity_entropy_pondering": sparsity_components.get('entropy_pondering', 0.0),
            "expected_clues_used": sparsity_components.get('expected_clues_used', 0.0),
            "stop_prob_from_loss": sparsity_components.get('stop_prob_mean', 0.0),
            # Per-sample variance diagnostics (verify per-task clue learning)
            "clues_used_std": sparsity_components.get('clues_used_std', 0.0),
            # Per-sample penalty mean (scaled by lambda_sparsity, should match old behavior)
            "per_sample_clue_penalty_mean": per_sample_clue_penalty_scaled.mean().item(),
        }


# Backward compatibility alias
RLANLossFunction = RLANLoss
