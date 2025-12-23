"""
Symbolic Predicate Heads (SPH) for RLAN

The SPH module learns binary predicates that can be used for
compositional rule learning. These predicates answer questions like:
- "Is the input symmetric?"
- "Is the grid square?"
- "Does it have a single connected object?"

Key Features:
- P binary predicate outputs via Gumbel-sigmoid
- Temperature-controlled sharpness
- Predicate diversity regularization (decorrelation)
- Interpretable intermediate representation

The predicates can be used to:
- Gate different transformation pathways
- Compose complex rules from simple properties
- Provide interpretable reasoning traces
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def gumbel_sigmoid(
    logits: torch.Tensor,
    temperature: float = 1.0,
    hard: bool = False,
    deterministic: bool = False,
) -> torch.Tensor:
    """
    Sigmoid for binary predicate outputs.
    
    CRITICAL FIX (Dec 2025): Removed Gumbel noise entirely!
    
    Previous behavior:
    - Training: Added Gumbel noise for exploration
    - Eval: No noise (deterministic)
    - Result: Train/eval distribution mismatch!
    
    New behavior:
    - Training AND Eval: Pure sigmoid with temperature scaling
    - Result: Identical train/eval, no generalization gap
    
    Args:
        logits: Shape (...) logits for binary decision
        temperature: Temperature (lower = sharper)
        hard: If True, use straight-through estimator
        deterministic: Ignored - always deterministic now (kept for API compat)
        
    Returns:
        probs: Shape (...) probabilities in (0, 1)
    """
    # Pure sigmoid with temperature scaling - NO Gumbel noise
    # Same behavior for train and eval
    soft = torch.sigmoid(logits / max(temperature, 1e-10))
    
    if hard:
        hard_decision = (soft > 0.5).float()
        return (hard_decision - soft).detach() + soft
    
    return soft


class SymbolicPredicateHeads(nn.Module):
    """
    Symbolic Predicate Heads - learns binary predicates for compositional rules.
    
    Architecture:
        1. Global feature aggregation (spatial pooling)
        2. MLP to compute predicate logits
        3. Gumbel-sigmoid for differentiable binary outputs
    
    The predicates are learned end-to-end and emerge from the task requirements.
    No explicit supervision is provided - the model learns what predicates
    are useful for solving tasks.
    """
    
    def __init__(
        self,
        hidden_dim: int = 128,
        num_predicates: int = 8,
        dropout: float = 0.1,
    ):
        """
        Args:
            hidden_dim: Input feature dimension
            num_predicates: Number of binary predicates (P)
            dropout: Dropout probability
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_predicates = num_predicates
        
        # Global feature aggregation
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        
        # Additional statistics for richer global features
        self.stats_proj = nn.Linear(hidden_dim * 2, hidden_dim)  # mean + std
        
        # Predicate predictor
        self.predicate_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_predicates),
        )
        
        # Optional: Learnable predicate prototypes for interpretability
        self.predicate_names = [
            "is_symmetric",
            "is_square",
            "has_single_object",
            "is_tiled",
            "has_border",
            "is_sparse",
            "has_pattern",
            "needs_fill",
        ][:num_predicates]
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def _compute_global_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute global features from spatial feature map.
        
        Args:
            features: Shape (B, D, H, W)
            
        Returns:
            global_features: Shape (B, D)
        """
        B, D, H, W = features.shape
        
        # Global average pooling
        mean_features = features.mean(dim=(-2, -1))  # (B, D)
        
        # Global std for additional statistics
        std_features = features.std(dim=(-2, -1))  # (B, D)
        
        # Combine mean and std
        combined = torch.cat([mean_features, std_features], dim=-1)  # (B, 2*D)
        global_features = self.stats_proj(combined)  # (B, D)
        global_features = self.layer_norm(global_features)
        
        return global_features
    
    def forward(
        self,
        features: torch.Tensor,
        temperature: float = 1.0,
        hard: bool = False,
    ) -> torch.Tensor:
        """
        Compute binary predicates from features.
        
        Args:
            features: Shape (B, D, H, W) encoded features
            temperature: Gumbel-sigmoid temperature
            hard: If True, use hard binary decisions
            
        Returns:
            predicates: Shape (B, P) predicate values in (0, 1)
        """
        # Get global features
        global_features = self._compute_global_features(features)  # (B, D)
        
        # Predict predicate logits
        logits = self.predicate_head(global_features)  # (B, P)
        
        # Apply Gumbel-sigmoid (deterministic during eval for reproducibility)
        predicates = gumbel_sigmoid(
            logits, 
            temperature=temperature, 
            hard=hard,
            deterministic=not self.training
        )
        
        return predicates
    
    def forward_with_logits(
        self,
        features: torch.Tensor,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning both predicates and logits.
        
        Useful for computing predicate diversity loss.
        
        Args:
            features: Shape (B, D, H, W)
            temperature: Gumbel-sigmoid temperature
            
        Returns:
            predicates: Shape (B, P)
            logits: Shape (B, P)
        """
        global_features = self._compute_global_features(features)
        logits = self.predicate_head(global_features)
        predicates = gumbel_sigmoid(
            logits, 
            temperature=temperature,
            deterministic=not self.training
        )
        
        return predicates, logits
    
    def compute_diversity_loss(self, predicates: torch.Tensor) -> torch.Tensor:
        """
        Compute predicate diversity loss to decorrelate predicates.
        
        Encourages different predicates to capture different properties.
        
        Args:
            predicates: Shape (B, P) predicate activations
            
        Returns:
            loss: Scalar diversity loss (lower is more diverse)
        """
        B, P = predicates.shape
        
        if B < 2:
            return torch.tensor(0.0, device=predicates.device)
        
        # Center predicates
        predicates_centered = predicates - predicates.mean(dim=0, keepdim=True)
        
        # Compute correlation matrix
        # Correlation between predicates across the batch
        corr = torch.mm(predicates_centered.T, predicates_centered) / (B - 1)
        
        # We want off-diagonal elements to be zero (decorrelated)
        # Frobenius norm of (corr - I)
        identity = torch.eye(P, device=predicates.device)
        diversity_loss = torch.norm(corr - identity, p='fro') ** 2 / (P * P)
        
        return diversity_loss
    
    def get_predicate_summary(self, predicates: torch.Tensor) -> dict:
        """
        Get human-readable summary of predicate activations.
        
        Args:
            predicates: Shape (B, P) or (P,) predicate values
            
        Returns:
            summary: Dict mapping predicate names to activation values
        """
        if predicates.dim() == 2:
            predicates = predicates.mean(dim=0)  # Average over batch
        
        summary = {}
        for i, name in enumerate(self.predicate_names):
            if i < predicates.shape[0]:
                summary[name] = predicates[i].item()
        
        return summary
