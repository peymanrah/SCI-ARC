"""
Recursive Solver for RLAN

The Recursive Solver is the output generation module of RLAN. It iteratively
refines its prediction over T steps, conditioning on:
- Clue-relative features (from DSC + MSRE)
- Count embeddings (from LCR)
- Predicate gates (from SPH)
- Input grid (for residual/copy connections)

Key Features:
- ConvGRU for iterative refinement
- Multi-scale feature integration
- Predicate-based gating for conditional computation
- Deep supervision at each step

Architecture:
    For t = 1 to T:
        1. Aggregate clue features
        2. Inject count information
        3. Modulate by predicates
        4. ConvGRU update
        5. Predict logits
"""

import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvGRUCell(nn.Module):
    """
    Convolutional GRU cell for spatial state updates.
    
    Unlike standard GRU which operates on vectors, ConvGRU
    maintains spatial structure while performing recurrent updates.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        kernel_size: int = 3,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2
        
        # Reset gate
        self.reset_gate = nn.Conv2d(
            input_dim + hidden_dim, hidden_dim,
            kernel_size=kernel_size, padding=padding
        )
        
        # Update gate
        self.update_gate = nn.Conv2d(
            input_dim + hidden_dim, hidden_dim,
            kernel_size=kernel_size, padding=padding
        )
        
        # Candidate state
        self.candidate = nn.Conv2d(
            input_dim + hidden_dim, hidden_dim,
            kernel_size=kernel_size, padding=padding
        )
        
        self.norm = nn.GroupNorm(8, hidden_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        h: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Single GRU step.
        
        Args:
            x: Input features (B, input_dim, H, W)
            h: Previous hidden state (B, hidden_dim, H, W) or None
            
        Returns:
            h_new: Updated hidden state (B, hidden_dim, H, W)
        """
        B, _, H, W = x.shape
        
        if h is None:
            h = torch.zeros(B, self.hidden_dim, H, W, device=x.device, dtype=x.dtype)
        
        # Concatenate input and hidden state
        combined = torch.cat([x, h], dim=1)
        
        # Compute gates
        r = torch.sigmoid(self.reset_gate(combined))
        z = torch.sigmoid(self.update_gate(combined))
        
        # Compute candidate with reset gate
        combined_reset = torch.cat([x, r * h], dim=1)
        h_candidate = torch.tanh(self.candidate(combined_reset))
        
        # Update hidden state
        h_new = (1 - z) * h + z * h_candidate
        h_new = self.norm(h_new)
        
        return h_new


class PredicateGating(nn.Module):
    """
    Gate features based on predicate activations.
    
    Allows the model to conditionally activate different
    transformation pathways based on input properties.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_predicates: int,
    ):
        super().__init__()
        
        # Learn how each predicate affects features
        self.gate_proj = nn.Linear(num_predicates, hidden_dim)
        self.scale_proj = nn.Linear(num_predicates, hidden_dim)
        
    def forward(
        self,
        features: torch.Tensor,
        predicates: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply predicate-based gating.
        
        Args:
            features: Shape (B, D, H, W)
            predicates: Shape (B, P)
            
        Returns:
            gated: Shape (B, D, H, W)
        """
        B, D, H, W = features.shape
        
        # Compute gate and scale from predicates
        gate = torch.sigmoid(self.gate_proj(predicates))  # (B, D)
        scale = self.scale_proj(predicates)  # (B, D)
        
        # Apply to features (broadcast over spatial dims)
        gate = gate.view(B, D, 1, 1)
        scale = scale.view(B, D, 1, 1)
        
        gated = features * gate + scale
        
        return gated


class RecursiveSolver(nn.Module):
    """
    Recursive Solver - iterative refinement decoder for RLAN.
    
    Generates output predictions by iteratively refining a hidden state,
    conditioned on clue features, count embeddings, and predicates.
    """
    
    def __init__(
        self,
        hidden_dim: int = 128,
        num_classes: int = 11,
        num_steps: int = 6,
        num_predicates: int = 8,
        num_colors: int = 10,
        dropout: float = 0.1,
    ):
        """
        Args:
            hidden_dim: Feature dimension
            num_classes: Output classes (10 colors + background)
            num_steps: Number of refinement iterations (T)
            num_predicates: Number of predicate inputs
            num_colors: Number of input colors (for count embedding)
            dropout: Dropout probability
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_steps = num_steps
        
        # Clue feature aggregation
        self.clue_aggregator = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            nn.GELU(),
            nn.GroupNorm(8, hidden_dim),
        )
        
        # Count embedding projection
        self.count_proj = nn.Sequential(
            nn.Linear(hidden_dim * num_colors, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        
        # Predicate gating
        self.predicate_gate = PredicateGating(hidden_dim, num_predicates)
        
        # ConvGRU for iterative refinement
        self.gru = ConvGRUCell(hidden_dim * 2, hidden_dim)
        
        # Input grid embedding (for copy/residual connections)
        self.input_embed = nn.Embedding(num_colors + 1, hidden_dim)  # +1 for padding
        
        # Output head (per-step predictions for deep supervision)
        self.output_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.GroupNorm(8, hidden_dim),
            nn.Conv2d(hidden_dim, num_classes, 1),
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def _aggregate_clues(
        self,
        clue_features: torch.Tensor,
        attention_maps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Aggregate features from multiple clues.
        
        Args:
            clue_features: Shape (B, K, D, H, W)
            attention_maps: Optional (B, K, H, W) attention weights
            
        Returns:
            aggregated: Shape (B, D, H, W)
        """
        B, K, D, H, W = clue_features.shape
        
        if attention_maps is not None:
            # Weighted sum using attention maps
            weights = attention_maps.unsqueeze(2)  # (B, K, 1, H, W)
            weighted = clue_features * weights
            aggregated = weighted.sum(dim=1)  # (B, D, H, W)
        else:
            # Simple mean
            aggregated = clue_features.mean(dim=1)  # (B, D, H, W)
        
        aggregated = self.clue_aggregator(aggregated)
        
        return aggregated
    
    def _inject_counts(
        self,
        features: torch.Tensor,
        count_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Inject count information into spatial features.
        
        Args:
            features: Shape (B, D, H, W)
            count_embedding: Shape (B, num_colors, D)
            
        Returns:
            enhanced: Shape (B, D, H, W)
        """
        B, D, H, W = features.shape
        
        # Flatten count embeddings
        count_flat = count_embedding.view(B, -1)  # (B, num_colors * D)
        count_proj = self.count_proj(count_flat)  # (B, D)
        
        # Add to features (broadcast over spatial dims)
        count_spatial = count_proj.view(B, D, 1, 1).expand(-1, -1, H, W)
        enhanced = features + count_spatial
        
        return enhanced
    
    def forward(
        self,
        clue_features: torch.Tensor,
        count_embedding: torch.Tensor,
        predicates: torch.Tensor,
        input_grid: torch.Tensor,
        attention_maps: Optional[torch.Tensor] = None,
        return_all_steps: bool = False,
    ) -> torch.Tensor:
        """
        Generate output through iterative refinement.
        
        Args:
            clue_features: Shape (B, K, D, H, W) from DSC + MSRE
            count_embedding: Shape (B, num_colors, D) from LCR
            predicates: Shape (B, P) from SPH
            input_grid: Shape (B, H, W) original input grid
            attention_maps: Optional (B, K, H, W) clue attention maps
            return_all_steps: If True, return predictions at all steps
            
        Returns:
            If return_all_steps:
                all_logits: List of (B, num_classes, H, W) for each step
            Else:
                logits: Shape (B, num_classes, H, W) final prediction
        """
        B, K, D, H, W = clue_features.shape
        device = clue_features.device
        
        # Aggregate clue features
        aggregated = self._aggregate_clues(clue_features, attention_maps)  # (B, D, H, W)
        
        # Inject count information
        aggregated = self._inject_counts(aggregated, count_embedding)
        
        # Apply predicate gating
        aggregated = self.predicate_gate(aggregated, predicates)
        
        # Embed input grid for residual connections
        input_clamped = input_grid.clamp(0, 10).long()
        input_embed = self.input_embed(input_clamped)  # (B, H, W, D)
        input_embed = input_embed.permute(0, 3, 1, 2)  # (B, D, H, W)
        
        # Initialize hidden state
        h = None
        all_logits = []
        
        # Iterative refinement
        for t in range(self.num_steps):
            # Combine aggregated features with input embedding
            combined = torch.cat([aggregated, input_embed], dim=1)  # (B, 2D, H, W)
            combined = self.dropout(combined)
            
            # GRU update
            h = self.gru(combined, h)
            
            # Predict output
            logits = self.output_head(h)  # (B, num_classes, H, W)
            all_logits.append(logits)
            
            # Optional: Use prediction to update input embedding (feedback)
            if t < self.num_steps - 1:
                pred = logits.argmax(dim=1)  # (B, H, W)
                input_embed = self.input_embed(pred.clamp(0, 10)).permute(0, 3, 1, 2)
        
        if return_all_steps:
            return all_logits
        else:
            return all_logits[-1]
    
    def compute_deep_supervision_loss(
        self,
        all_logits: List[torch.Tensor],
        targets: torch.Tensor,
        loss_fn: nn.Module,
        decay: float = 0.8,
    ) -> torch.Tensor:
        """
        Compute deep supervision loss over all refinement steps.
        
        Later steps are weighted more heavily.
        
        Args:
            all_logits: List of (B, num_classes, H, W) predictions
            targets: Shape (B, H, W) ground truth
            loss_fn: Loss function (e.g., CrossEntropyLoss)
            decay: Weight decay factor (weight_t = decay^(T-t))
            
        Returns:
            total_loss: Scalar weighted loss
        """
        T = len(all_logits)
        total_loss = 0.0
        total_weight = 0.0
        
        for t, logits in enumerate(all_logits):
            weight = decay ** (T - 1 - t)  # Later steps have higher weight
            step_loss = loss_fn(logits, targets)
            total_loss = total_loss + weight * step_loss
            total_weight += weight
        
        return total_loss / total_weight
