"""
HyperLoRA Module for RLAN

This module implements "Amortized Hyper-LoRA" - a meta-learning approach that
predicts task-specific LoRA weight adaptations from the support set context.

Key Features:
- Predicts LoRA matrices (A, B) from context vector
- Targets Solver GRU and output_head for maximum impact
- Initializes to near-zero for backward compatibility (safe adoption)
- No gradients needed at inference - instant adaptation

Theoretical Motivation:
- TTT updates weights via gradient descent at inference (slow, overfits)
- HyperLoRA PREDICTS the optimal weight update directly (fast, generalizes)
- The HyperGenerator learns: "For this type of context, use these weights"

Architecture:
    support_features (B, N, D, H, W) 
        → pool to (B, D)
        → HyperGenerator MLP
        → predict A matrices (B, in_dim, rank) 
        → predict B matrices (B, rank, out_dim)
        → ΔW = scaling * A @ B

Usage in RLAN:
    1. ContextEncoder produces support_features
    2. HyperLoRA predicts weight deltas from pooled context
    3. Solver applies deltas during forward pass
    4. Training uses LOO loss to teach HyperLoRA what weights work

Paper Reference:
    Inspired by HyperNetworks (Ha et al., 2016) and LoRA (Hu et al., 2021)
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class HyperLoRAConfig:
    """Configuration for HyperLoRA module.
    
    SCIENTIFIC FIX (2025-12-25):
    - Increased init_scale from 0.01 to 0.1 to fix weak initialization coupling.
    - At init_scale=0.01, adaptation effect was only 0.0001 loss delta (negligible).
    - Higher init_scale = stronger initial signal for meta-learning to latch onto.
    
    STABILITY FIX (2026-01-01):
    - Added lora_max_norm to prevent runaway delta growth (P0.1).
    - Based on log analysis: LoRA norm grew 0.793 → 1.714 causing training collapse at epoch 41.
    - FIX: Reduced lora_max_norm from 3.0 → 1.0 to prevent GRU gate saturation.
    - Mathematical basis: When ||ΔW||_F > 1.0 relative to normalized base weights,
      the adaptation term dominates, causing sigmoid saturation → gradient vanishing.
    """
    enabled: bool = True
    rank: int = 8                          # LoRA rank (8-16 recommended)
    hidden_dim: int = 256                  # Model hidden dimension
    context_dim: int = 256                 # Context vector dimension
    scaling: float = 0.1                   # Initial scaling (start small)
    dropout: float = 0.1                   # Dropout on predicted weights
    target_gru: bool = True                # Adapt GRU weights
    target_output_head: bool = True        # Adapt output head weights
    num_gru_gates: int = 3                 # reset, update, candidate (GRU has 3 gates)
    init_scale: float = 0.1               # Stronger init for better adaptation signal (was 0.01)
    lora_max_norm: float = 1.0            # P0.1: Hard clamp per-sample delta L2 norm (was 3.0, caused collapse)


class LoRAPredictor(nn.Module):
    """
    Predicts a single LoRA matrix pair (A, B) from context.
    
    ΔW = scaling * A @ B
    where A: (in_features, rank), B: (rank, out_features)
    
    The prediction is done via:
        context → MLP → [A_flat, B_flat] → reshape
    """
    
    def __init__(
        self,
        context_dim: int,
        in_features: int,
        out_features: int,
        rank: int = 8,
        scaling: float = 0.1,
        init_scale: float = 0.01,
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = scaling
        
        # Predict A matrix: (in_features, rank)
        self.predict_A = nn.Sequential(
            nn.Linear(context_dim, context_dim),
            nn.GELU(),
            nn.Linear(context_dim, in_features * rank),
        )
        
        # Predict B matrix: (rank, out_features)
        self.predict_B = nn.Sequential(
            nn.Linear(context_dim, context_dim),
            nn.GELU(),
            nn.Linear(context_dim, rank * out_features),
        )
        
        # Initialize to near-zero for safe adoption
        # The model starts with ΔW ≈ 0, so behavior is unchanged initially
        self._init_near_zero(init_scale)
    
    def _init_near_zero(self, scale: float):
        """Initialize output layers to near-zero."""
        # A predictor final layer
        nn.init.normal_(self.predict_A[-1].weight, std=scale)
        nn.init.zeros_(self.predict_A[-1].bias)
        
        # B predictor final layer
        nn.init.normal_(self.predict_B[-1].weight, std=scale)
        nn.init.zeros_(self.predict_B[-1].bias)
    
    def forward(self, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict LoRA matrices from context.
        
        Args:
            context: (B, context_dim) pooled task context
            
        Returns:
            A: (B, in_features, rank)
            B: (B, rank, out_features)
        """
        B = context.shape[0]
        
        # Predict A matrix
        A_flat = self.predict_A(context)  # (B, in_features * rank)
        A = A_flat.view(B, self.in_features, self.rank)
        
        # Predict B matrix
        B_flat = self.predict_B(context)  # (B, rank * out_features)
        B_mat = B_flat.view(B, self.rank, self.out_features)
        
        return A, B_mat
    
    def compute_delta_w(self, context: torch.Tensor) -> torch.Tensor:
        """
        Compute the full weight delta ΔW = scaling * A @ B.
        
        Args:
            context: (B, context_dim) pooled task context
            
        Returns:
            delta_w: (B, in_features, out_features)
        """
        A, B = self.forward(context)
        # ΔW = A @ B with scaling
        delta_w = self.scaling * torch.bmm(A, B)  # (B, in_features, out_features)
        return delta_w


class HyperLoRA(nn.Module):
    """
    HyperLoRA: Predicts task-specific LoRA weight adaptations.
    
    Targets:
    1. Solver GRU gates (reset, update, candidate) - ih and hh weights
    2. Solver output_head first conv layer
    
    The module pools support_features to a context vector, then
    predicts LoRA deltas for each target layer.
    
    Usage:
        hyperlora = HyperLoRA(config)
        deltas = hyperlora(support_features)
        # deltas contains weight updates for targeted layers
    """
    
    def __init__(
        self,
        config: Optional[HyperLoRAConfig] = None,
        hidden_dim: int = 256,
        rank: int = 8,
        scaling: float = 0.1,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        if config is not None:
            hidden_dim = config.hidden_dim
            rank = config.rank
            scaling = config.scaling
            dropout = config.dropout
            self.config = config
        else:
            self.config = HyperLoRAConfig(
                hidden_dim=hidden_dim,
                context_dim=hidden_dim,
                rank=rank,
                scaling=scaling,
                dropout=dropout,
            )
        
        self.hidden_dim = hidden_dim
        self.rank = rank
        self.scaling = scaling
        
        # Runtime delta scaling for warmup (Patch 1: Dec 2025)
        # This allows training to gradually ramp up HyperLoRA influence
        # without changing model initialization. Setting delta_scale < 1.0
        # attenuates all predicted weight deltas proportionally.
        # Memory-neutral: single float, no extra VRAM.
        self.delta_scale = 1.0  # Default: no attenuation (backward compatible)
        
        # P0.1: LoRA norm clamping (Jan 2026 stability fix)
        # Hard clamp per-sample LoRA delta L2 norm to prevent runaway growth.
        # Based on log analysis: LoRA norm grew 0.793 → 1.714 causing collapse at epoch 41.
        # Default 1.0 ensures adaptation never dominates base weights.
        # Mathematical invariant: ||ΔW||_F ≤ 1.0 prevents GRU gate saturation.
        self.lora_max_norm = config.lora_max_norm if config and hasattr(config, 'lora_max_norm') else 1.0
        
        # Jan 2026: Clamp hit-rate tracking for diagnostics
        # These counters are reset each epoch by train_rlan.py to compute hit-rate
        self.clamp_hit_count = 0      # Number of samples that hit the clamp
        self.clamp_total_count = 0    # Total number of samples processed
        self.clamp_max_pre_norm = 0.0  # Max pre-clamp norm seen this epoch
        
        # Context pooling: (B, N, D, H, W) → (B, D)
        self.context_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Will be applied per-pair
        )
        
        # Context fusion: aggregate multiple pairs
        self.context_fuse = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        
        # =============================================================
        # LoRA Predictors for Target Layers
        # =============================================================
        
        # GRU has 3 gates (reset, update, candidate)
        # Each gate has input-to-hidden (ih) and hidden-to-hidden (hh) weights
        # 
        # For ConvGRU in RecursiveSolver:
        # - reset_gate: Conv2d(input_dim + hidden_dim, hidden_dim)
        # - update_gate: Conv2d(input_dim + hidden_dim, hidden_dim)
        # - candidate: SwiGLUConv2d or Conv2d
        #
        # We target the equivalent "bias" adjustment via LoRA-style adaptation
        # that modulates the output of each gate
        
        # GRU gate modulation (applied as additive bias-like adjustment)
        # Input: hidden_dim, Output: hidden_dim (for each gate output)
        init_scale = self.config.init_scale
        
        self.gru_reset_lora = LoRAPredictor(
            context_dim=hidden_dim,
            in_features=hidden_dim,
            out_features=hidden_dim,
            rank=rank,
            scaling=scaling,
            init_scale=init_scale,
        )
        
        self.gru_update_lora = LoRAPredictor(
            context_dim=hidden_dim,
            in_features=hidden_dim,
            out_features=hidden_dim,
            rank=rank,
            scaling=scaling,
            init_scale=init_scale,
        )
        
        self.gru_candidate_lora = LoRAPredictor(
            context_dim=hidden_dim,
            in_features=hidden_dim,
            out_features=hidden_dim,
            rank=rank,
            scaling=scaling,
            init_scale=init_scale,
        )
        
        # Output head modulation
        # The output_head is: Conv2d(hidden_dim, hidden_dim) -> GELU -> Conv2d(hidden_dim, num_classes)
        # We target the first conv output
        self.output_head_lora = LoRAPredictor(
            context_dim=hidden_dim,
            in_features=hidden_dim,
            out_features=hidden_dim,
            rank=rank,
            scaling=scaling,
            init_scale=init_scale,
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def pool_context(self, support_features: torch.Tensor) -> torch.Tensor:
        """
        Pool support features to a single context vector with DIHEDRAL INVARIANCE.
        
        This is critical for TTA consensus: the context must be the same
        regardless of how the input is oriented (rotated/reflected).
        
        We achieve this by averaging over all 8 D4 group transformations:
        - 4 rotations: 0°, 90°, 180°, 270°
        - 2 reflections: horizontal, vertical
        
        Args:
            support_features: (B, N, D, H, W) spatial features from ContextEncoder
            
        Returns:
            context: (B, D) dihedral-invariant pooled context vector
        """
        B, N, D, H, W = support_features.shape
        
        # Pool each pair spatially: (B*N, D, H, W) → (B*N, D, 1, 1)
        # Use .reshape() instead of .view() to handle non-contiguous tensors
        # (can happen after augmentation transforms like rotate/flip)
        features_flat = support_features.reshape(B * N, D, H, W)
        
        # DIHEDRAL INVARIANCE: Average context over all D4 transformations
        # This makes HyperLoRA produce the SAME deltas for rotated/flipped inputs
        contexts = []
        
        # Original
        pooled_orig = self.context_pool(features_flat)  # (B*N, D, 1, 1)
        pooled_orig = pooled_orig.reshape(B, N, D).mean(dim=1)  # (B, D)
        contexts.append(self.context_fuse(pooled_orig))
        
        # 90° rotation
        rot90 = torch.rot90(features_flat, k=1, dims=[2, 3])
        pooled_rot90 = self.context_pool(rot90).reshape(B, N, D).mean(dim=1)
        contexts.append(self.context_fuse(pooled_rot90))
        
        # 180° rotation
        rot180 = torch.rot90(features_flat, k=2, dims=[2, 3])
        pooled_rot180 = self.context_pool(rot180).reshape(B, N, D).mean(dim=1)
        contexts.append(self.context_fuse(pooled_rot180))
        
        # 270° rotation
        rot270 = torch.rot90(features_flat, k=3, dims=[2, 3])
        pooled_rot270 = self.context_pool(rot270).reshape(B, N, D).mean(dim=1)
        contexts.append(self.context_fuse(pooled_rot270))
        
        # Horizontal flip
        flip_h = torch.flip(features_flat, dims=[3])
        pooled_flip_h = self.context_pool(flip_h).reshape(B, N, D).mean(dim=1)
        contexts.append(self.context_fuse(pooled_flip_h))
        
        # Vertical flip
        flip_v = torch.flip(features_flat, dims=[2])
        pooled_flip_v = self.context_pool(flip_v).reshape(B, N, D).mean(dim=1)
        contexts.append(self.context_fuse(pooled_flip_v))
        
        # Horizontal flip + 90° rotation (diagonal flip)
        flip_h_rot90 = torch.rot90(flip_h, k=1, dims=[2, 3])
        pooled_flip_h_rot90 = self.context_pool(flip_h_rot90).reshape(B, N, D).mean(dim=1)
        contexts.append(self.context_fuse(pooled_flip_h_rot90))
        
        # Horizontal flip + 270° rotation (anti-diagonal flip)
        flip_h_rot270 = torch.rot90(flip_h, k=3, dims=[2, 3])
        pooled_flip_h_rot270 = self.context_pool(flip_h_rot270).reshape(B, N, D).mean(dim=1)
        contexts.append(self.context_fuse(pooled_flip_h_rot270))
        
        # Average over all 8 transformations for dihedral invariance
        context = torch.stack(contexts, dim=0).mean(dim=0)  # (B, D)
        
        return context
    
    def _clamp_delta_norm(self, delta: torch.Tensor) -> torch.Tensor:
        """
        Clamp per-sample LoRA delta to max L2 norm.
        
        P0.1 Stability Fix (Jan 2026):
        - Based on log analysis: LoRA norm grew 1.27 → 1.80 causing training collapse
        - Hard clamp prevents runaway growth while preserving direction
        - Gradient-friendly: uses scaling rather than hard truncation
        - Tracks hit-rate for diagnostics (Jan 2026 observability patch)
        
        Args:
            delta: (B, D1, D2) LoRA weight delta matrix
            
        Returns:
            Clamped delta with same shape, per-sample norm <= lora_max_norm
        """
        if self.lora_max_norm is None or self.lora_max_norm <= 0:
            return delta
        
        # Compute per-sample L2 norm
        norms = delta.norm(dim=(1, 2), keepdim=True)  # (B, 1, 1)
        
        # Track hit-rate for diagnostics (Jan 2026)
        # Count samples where norm exceeds threshold
        with torch.no_grad():
            batch_size = delta.shape[0]
            hits = (norms.squeeze() > self.lora_max_norm).sum().item()
            self.clamp_hit_count += hits
            self.clamp_total_count += batch_size
            max_norm = norms.max().item()
            if max_norm > self.clamp_max_pre_norm:
                self.clamp_max_pre_norm = max_norm
        
        # Scale down if norm exceeds max (preserve direction)
        scale = torch.clamp(self.lora_max_norm / (norms + 1e-8), max=1.0)
        
        return delta * scale
    
    def reset_clamp_stats(self):
        """Reset clamp hit-rate tracking for new epoch."""
        self.clamp_hit_count = 0
        self.clamp_total_count = 0
        self.clamp_max_pre_norm = 0.0
    
    def get_clamp_stats(self) -> dict:
        """
        Get clamp hit-rate statistics for diagnostics.
        
        Returns:
            dict with:
                - hit_rate: fraction of samples that hit the clamp (0.0-1.0)
                - hit_count: total samples that hit
                - total_count: total samples processed
                - max_pre_norm: maximum pre-clamp norm seen
                - threshold: the clamp threshold
        """
        hit_rate = self.clamp_hit_count / max(1, self.clamp_total_count)
        return {
            'hit_rate': hit_rate,
            'hit_count': self.clamp_hit_count,
            'total_count': self.clamp_total_count,
            'max_pre_norm': self.clamp_max_pre_norm,
            'threshold': self.lora_max_norm,
        }
    
    def forward(
        self,
        support_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict LoRA weight deltas for all target layers.
        
        Args:
            support_features: (B, N, D, H, W) from ContextEncoder
            
        Returns:
            deltas: Dict with keys:
                - 'gru_reset': (B, hidden_dim, hidden_dim)
                - 'gru_update': (B, hidden_dim, hidden_dim)
                - 'gru_candidate': (B, hidden_dim, hidden_dim)
                - 'output_head': (B, hidden_dim, hidden_dim)
                - 'context': (B, D) pooled context for other uses
        """
        # Pool to context vector
        context = self.pool_context(support_features)  # (B, D)
        context = self.dropout(context)
        
        # Predict LoRA deltas for each target
        # Apply delta_scale for warmup (Patch 1: Dec 2025)
        # This multiplicatively attenuates deltas during early training
        scale = self.delta_scale
        
        # P0.1: Apply norm clamping to prevent runaway LoRA growth (Jan 2026)
        deltas = {
            'gru_reset': self._clamp_delta_norm(scale * self.gru_reset_lora.compute_delta_w(context)),
            'gru_update': self._clamp_delta_norm(scale * self.gru_update_lora.compute_delta_w(context)),
            'gru_candidate': self._clamp_delta_norm(scale * self.gru_candidate_lora.compute_delta_w(context)),
            'output_head': self._clamp_delta_norm(scale * self.output_head_lora.compute_delta_w(context)),
            'context': context,  # Keep for other uses
        }
        
        return deltas
    
    def compute_delta_w(self, context: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute LoRA weight deltas from a pre-pooled context vector.
        
        This is used by AugmentationEquivarianceLoss to compare weight predictions
        for original and augmented contexts without re-pooling.
        
        Args:
            context: (B, D) pre-pooled context vector
            
        Returns:
            deltas: Dict with weight deltas for each target layer
        """
        context = self.dropout(context)
        
        # Predict LoRA deltas for each target
        # Apply delta_scale for warmup (Patch 1: Dec 2025)
        scale = self.delta_scale
        
        # P0.1: Apply norm clamping to prevent runaway LoRA growth (Jan 2026)
        deltas = {
            'gru_reset': self._clamp_delta_norm(scale * self.gru_reset_lora.compute_delta_w(context)),
            'gru_update': self._clamp_delta_norm(scale * self.gru_update_lora.compute_delta_w(context)),
            'gru_candidate': self._clamp_delta_norm(scale * self.gru_candidate_lora.compute_delta_w(context)),
            'output_head': self._clamp_delta_norm(scale * self.output_head_lora.compute_delta_w(context)),
        }
        
        return deltas
    
    def get_lora_magnitude(self, deltas: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Compute L2 magnitude of each LoRA delta for monitoring.
        
        Useful for tracking if HyperLoRA is being used (magnitude > 0)
        or collapsing (magnitude → 0).
        """
        magnitudes = {}
        for name, delta in deltas.items():
            if name != 'context':
                magnitudes[f'lora_{name}_magnitude'] = delta.norm(dim=(1, 2)).mean().item()
        return magnitudes
    
    def get_weight_diversity(self, deltas: Dict[str, torch.Tensor]) -> float:
        """
        Compute diversity of predicted weights across batch.
        
        High diversity = different tasks get different weights (good)
        Low diversity = all tasks get same weights (collapsed, bad)
        """
        diversities = []
        for name, delta in deltas.items():
            if name != 'context':
                # Compute std across batch dimension
                batch_std = delta.std(dim=0).mean().item()
                diversities.append(batch_std)
        
        return sum(diversities) / len(diversities) if diversities else 0.0


class LoRAApplicator:
    """
    Helper class to apply LoRA deltas to layer outputs.
    
    This is used in the modified RecursiveSolver to inject
    the predicted weight adaptations.
    
    Usage:
        applicator = LoRAApplicator()
        
        # In GRU forward:
        reset_gate_out = original_reset_gate(combined)
        reset_gate_out = applicator.apply(reset_gate_out, deltas['gru_reset'])
    """
    
    @staticmethod
    def apply_spatial(
        features: torch.Tensor,  # (B, D, H, W)
        delta_w: torch.Tensor,   # (B, D, D) weight delta
    ) -> torch.Tensor:
        """
        Apply LoRA delta to spatial features.
        
        Equivalent to: output = features + features @ delta_w
        where delta_w is applied per-spatial-location.
        
        Args:
            features: (B, D, H, W) input features
            delta_w: (B, D_in, D_out) weight delta
            
        Returns:
            output: (B, D, H, W) modulated features
        """
        B, D, H, W = features.shape
        
        # Reshape: (B, D, H, W) → (B, H*W, D)
        features_flat = features.permute(0, 2, 3, 1).reshape(B, H * W, D)
        
        # Apply: (B, H*W, D) @ (B, D, D) → (B, H*W, D)
        delta_out = torch.bmm(features_flat, delta_w)
        
        # Add residual and reshape back
        output_flat = features_flat + delta_out
        output = output_flat.reshape(B, H, W, D).permute(0, 3, 1, 2)
        
        return output
    
    @staticmethod
    def apply_1d(
        features: torch.Tensor,  # (B, D)
        delta_w: torch.Tensor,   # (B, D, D) weight delta
    ) -> torch.Tensor:
        """
        Apply LoRA delta to 1D features (e.g., context vector).
        
        Args:
            features: (B, D) input features
            delta_w: (B, D_in, D_out) weight delta
            
        Returns:
            output: (B, D) modulated features
        """
        # features: (B, D) → (B, 1, D)
        features_expanded = features.unsqueeze(1)
        
        # Apply: (B, 1, D) @ (B, D, D) → (B, 1, D)
        delta_out = torch.bmm(features_expanded, delta_w)
        
        # Add residual and squeeze
        output = features + delta_out.squeeze(1)
        
        return output


def compute_hyperlora_health_metrics(
    hyper_lora: HyperLoRA,
    deltas: Dict[str, torch.Tensor],
    loo_accuracy: Optional[float] = None,
) -> Dict[str, float]:
    """
    Compute comprehensive health metrics for HyperLoRA during training.
    
    These metrics help diagnose:
    - Is HyperLoRA being used? (magnitude > 0)
    - Is it collapsing? (diversity → 0)
    - Is it overfitting? (LOO accuracy low)
    - Is it too strong? (magnitude very high)
    
    Args:
        hyper_lora: HyperLoRA module
        deltas: Output from hyper_lora.forward()
        loo_accuracy: Optional LOO accuracy from training
        
    Returns:
        Dict of metrics for logging
    """
    metrics = {}
    
    # Magnitude metrics (is LoRA being used?)
    magnitudes = hyper_lora.get_lora_magnitude(deltas)
    metrics.update(magnitudes)
    metrics['lora_total_magnitude'] = sum(magnitudes.values())
    
    # Diversity metric (are different tasks getting different weights?)
    metrics['lora_weight_diversity'] = hyper_lora.get_weight_diversity(deltas)
    
    # Per-layer statistics
    for name, delta in deltas.items():
        if name != 'context':
            # Mean absolute value
            metrics[f'lora_{name}_mean_abs'] = delta.abs().mean().item()
            # Max value (detect exploding weights)
            metrics[f'lora_{name}_max'] = delta.abs().max().item()
            # Sparsity (fraction near zero)
            metrics[f'lora_{name}_sparsity'] = (delta.abs() < 0.01).float().mean().item()
    
    # LOO accuracy if provided
    if loo_accuracy is not None:
        metrics['loo_accuracy'] = loo_accuracy
    
    # Health status flags
    total_mag = metrics['lora_total_magnitude']
    diversity = metrics['lora_weight_diversity']
    
    metrics['lora_health_collapsed'] = float(total_mag < 0.01)  # Near-zero weights
    metrics['lora_health_uniform'] = float(diversity < 0.01)   # Same weights for all tasks
    metrics['lora_health_exploding'] = float(total_mag > 100)  # Very large weights
    metrics['lora_health_ok'] = float(
        0.01 < total_mag < 100 and diversity > 0.01
    )
    
    return metrics
