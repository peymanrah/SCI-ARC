"""
Dynamic Saliency Controller (DSC) for RLAN

The DSC is the core innovation of RLAN. It iteratively discovers "clue anchors" -
spatial locations that serve as coordinate origins for relative reasoning.

Key Features:
- Learnable query vectors to attend to different spatial patterns
- Gumbel-softmax for differentiable hard attention
- Progressive masking to discover multiple non-overlapping clues
- Stop-token prediction for dynamic clue count
- Temperature annealing for progressive sharpening

Example:
    For a "move object to marker" task:
    - Clue 1: Attends to the marker (red pixel)
    - Clue 2: Attends to the object (grey square)
    - Stop after 2 clues
"""

import math
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def gumbel_softmax_2d(
    logits: torch.Tensor,
    temperature: float = 1.0,
    hard: bool = False,
    dim: int = -1,
) -> torch.Tensor:
    """
    Apply Gumbel-softmax to 2D spatial attention logits.
    
    Args:
        logits: Shape (B, H, W) or (B, K, H, W)
        temperature: Softmax temperature (lower = sharper)
        hard: If True, use straight-through estimator
        dim: Dimension(s) to apply softmax over
        
    Returns:
        Attention weights with same shape as logits
    """
    # Clamp input logits for numerical stability
    logits = logits.clamp(min=-50.0, max=50.0)
    
    # Add Gumbel noise with improved numerical stability
    # Use clamp to prevent log(0)
    uniform = torch.rand_like(logits).clamp(min=1e-10, max=1.0 - 1e-10)
    gumbel_noise = -torch.log(-torch.log(uniform))
    noisy_logits = (logits + gumbel_noise) / max(temperature, 1e-10)
    
    # Clamp noisy logits for softmax stability  
    noisy_logits = noisy_logits.clamp(min=-50.0, max=50.0)
    
    # Flatten spatial dims for softmax
    B = logits.shape[0]
    if logits.dim() == 3:  # (B, H, W)
        H, W = logits.shape[1], logits.shape[2]
        flat = noisy_logits.view(B, -1)
        soft = F.softmax(flat, dim=-1)
        soft = soft.view(B, H, W)
    elif logits.dim() == 4:  # (B, K, H, W)
        K, H, W = logits.shape[1], logits.shape[2], logits.shape[3]
        flat = noisy_logits.view(B, K, -1)
        soft = F.softmax(flat, dim=-1)
        soft = soft.view(B, K, H, W)
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got {logits.dim()}D")
    
    if hard:
        # Straight-through estimator
        if logits.dim() == 3:
            idx = soft.view(B, -1).argmax(dim=-1)
            hard_attn = torch.zeros_like(soft.view(B, -1))
            hard_attn.scatter_(1, idx.unsqueeze(-1), 1.0)
            hard_attn = hard_attn.view(B, H, W)
        else:
            idx = soft.view(B, K, -1).argmax(dim=-1)
            hard_attn = torch.zeros_like(soft.view(B, K, -1))
            hard_attn.scatter_(2, idx.unsqueeze(-1), 1.0)
            hard_attn = hard_attn.view(B, K, H, W)
        return (hard_attn - soft).detach() + soft
    
    return soft


class DynamicSaliencyController(nn.Module):
    """
    Dynamic Saliency Controller - discovers spatial anchors for RLAN.
    
    Architecture:
        1. K learnable query vectors (one per potential clue)
        2. Cross-attention between queries and encoded features
        3. Gumbel-softmax to produce sharp attention maps
        4. Centroid computation via weighted spatial average
        5. Progressive masking to find non-overlapping clues
        6. Stop-token prediction via MLP on attended features
    
    The DSC answers: "Where should I anchor my coordinate system?"
    """
    
    def __init__(
        self,
        hidden_dim: int = 128,
        max_clues: int = 5,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        """
        Args:
            hidden_dim: Feature dimension (D)
            max_clues: Maximum number of clue anchors (K)
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.max_clues = max_clues
        self.num_heads = num_heads
        
        # Learnable clue queries - each query learns to attend to different patterns
        self.clue_queries = nn.Parameter(torch.randn(max_clues, hidden_dim))
        nn.init.xavier_uniform_(self.clue_queries.unsqueeze(0))
        
        # Query projection for attention
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Attention score projection (multi-head to single attention map)
        self.attn_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        # Stop-token predictor
        # Predicts whether to stop after this clue
        # CRITICAL: Initialize with negative bias to default to "continue" not "stop"
        # sigmoid(-2.0) ≈ 0.12, so model starts by using most clues
        # 
        # Input: attended_features (D) + attention_entropy (1) = D+1 dimensions
        # The entropy input creates coupling between attention sharpness and stopping:
        # - Sharp attention (low entropy) → model can stop (found good anchor)
        # - Diffuse attention (high entropy) → need more clues (uncertain)
        self.stop_predictor = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim // 2),  # +1 for attention entropy
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        # Initialize with proper entropy coupling and less aggressive bias
        # CRITICAL: init_bias controls starting clue count
        # -1.0 -> sigmoid=0.27 -> ~4.4 clues, good gradient flow
        # -2.5 -> sigmoid=0.08 -> ~5.5 clues, BUT sigmoid gradient vanishes!
        # 
        # FIX: Changed from -2.5 to -1.0 to prevent gradient vanishing
        # The sigmoid gradient at x: sig(x)*(1-sig(x))
        # At -2.5: 0.076 * 0.924 = 0.07 (weak)
        # At -1.0: 0.27 * 0.73 = 0.20 (3x stronger!)
        self._init_stop_predictor_for_entropy_coupling(init_bias=-1.0)
        
        # Layer norm for stability
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.feature_norm = nn.LayerNorm(hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(hidden_dim)
        
        # Coordinate grids (will be registered as buffers)
        self._init_coord_grids(30)  # Max ARC size
    
    def _init_stop_predictor_for_entropy_coupling(self, init_bias: float = -1.0):
        """
        Initialize stop predictor with proper entropy coupling.
        
        The stop predictor receives [attended_features (D), entropy (1)].
        We want entropy to have strong influence on stopping:
        - Low entropy (sharp attention) → higher stop probability
        - High entropy (diffuse attention) → lower stop probability
        
        Key insight: entropy is normalized to [0,1] and negated, so:
        - Good focus: entropy_norm ≈ -0.8 to -1.0 (sharp → should stop)
        - Poor focus: entropy_norm ≈ -0.2 to -0.4 (diffuse → continue)
        
        The weight on entropy should be ~1-2 to have meaningful impact on logit.
        
        Args:
            init_bias: Initial bias value. Less negative = easier to learn to stop.
                -1.0 → sigmoid ≈ 0.27 (reasonable starting point)
                -2.0 → sigmoid ≈ 0.12 (too hard to overcome)
        """
        layers = [m for m in self.stop_predictor.modules() if isinstance(m, nn.Linear)]
        
        if len(layers) >= 2:
            first_layer, last_layer = layers[0], layers[-1]
            
            # First layer: Give entropy input (last dimension) strong weight
            # This creates real coupling between attention quality and stopping
            in_features = first_layer.in_features
            hidden_dim = in_features - 1  # All but last dim is attended_features
            
            # Initialize with kaiming for feature part, stronger for entropy
            nn.init.kaiming_normal_(first_layer.weight[:, :hidden_dim], mode='fan_in')
            # Entropy weight: stronger initialization (std ≈ 1.5)
            # Positive weight: low entropy (negative after negation) → higher activation
            nn.init.normal_(first_layer.weight[:, hidden_dim:], mean=1.5, std=0.5)
            nn.init.zeros_(first_layer.bias)
            
            # Last layer: reasonable initialization with less aggressive bias
            nn.init.normal_(last_layer.weight, mean=0.0, std=0.1)  # 10x larger than before
            nn.init.constant_(last_layer.bias, init_bias)
        elif len(layers) == 1:
            # Single layer case
            layer = layers[0]
            in_features = layer.in_features
            hidden_dim = in_features - 1
            
            nn.init.kaiming_normal_(layer.weight[:, :hidden_dim], mode='fan_in')
            nn.init.normal_(layer.weight[:, hidden_dim:], mean=1.5, std=0.5)
            nn.init.constant_(layer.bias, init_bias)
    
    def _init_coord_grids(self, max_size: int):
        """Initialize coordinate grids for centroid computation."""
        rows = torch.arange(max_size).float()
        cols = torch.arange(max_size).float()
        row_grid, col_grid = torch.meshgrid(rows, cols, indexing='ij')
        
        # Clone to ensure contiguous memory (needed for checkpoint loading)
        self.register_buffer('row_grid', row_grid.clone())
        self.register_buffer('col_grid', col_grid.clone())
    
    def _compute_centroid(
        self, 
        attention: torch.Tensor, 
        H: int, 
        W: int
    ) -> torch.Tensor:
        """
        Compute soft centroid from attention weights.
        
        Args:
            attention: Shape (B, H, W) attention weights (sum to 1)
            H, W: Grid dimensions
            
        Returns:
            centroids: Shape (B, 2) with (row, col) coordinates
        """
        B = attention.shape[0]
        
        # Get coordinate grids for this size
        row_grid = self.row_grid[:H, :W].unsqueeze(0).expand(B, -1, -1)
        col_grid = self.col_grid[:H, :W].unsqueeze(0).expand(B, -1, -1)
        
        # Weighted average of coordinates
        row_centroid = (attention * row_grid).sum(dim=(-2, -1))
        col_centroid = (attention * col_grid).sum(dim=(-2, -1))
        
        centroids = torch.stack([row_centroid, col_centroid], dim=-1)
        return centroids
    
    def forward(
        self,
        features: torch.Tensor,
        temperature: float = 1.0,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract clue anchors from encoded features.
        
        Args:
            features: Shape (B, D, H, W) encoded grid features
            temperature: Gumbel-softmax temperature (lower = sharper)
            mask: Optional (B, H, W) mask for valid positions
            
        Returns:
            centroids: Shape (B, K, 2) clue centroid coordinates
            attention_maps: Shape (B, K, H, W) soft attention per clue
            stop_logits: Shape (B, K) stop probability logits
        """
        B, D, H, W = features.shape
        K = self.max_clues
        
        # Reshape features: (B, D, H, W) -> (B, H*W, D)
        features_flat = features.permute(0, 2, 3, 1).reshape(B, H * W, D)
        features_flat = self.feature_norm(features_flat)
        
        # Initialize outputs
        all_centroids = []
        all_attention_maps = []
        all_stop_logits = []
        all_entropy_inputs = []  # Track entropy input to stop_predictor for diagnostics
        
        # Cumulative mask for progressive masking
        cumulative_mask = torch.ones(B, H, W, device=features.device)
        if mask is not None:
            cumulative_mask = cumulative_mask * mask
        
        # Process each clue sequentially
        for k in range(K):
            # Get query for this clue
            query = self.clue_queries[k:k+1].expand(B, -1)  # (B, D)
            query = self.query_norm(query)
            
            # Project query and keys
            q = self.query_proj(query)  # (B, D)
            k_proj = self.key_proj(features_flat)  # (B, H*W, D)
            v = self.value_proj(features_flat)  # (B, H*W, D)
            
            # Compute attention scores
            attn_scores = torch.einsum('bd,bnd->bn', q, k_proj) / self.scale  # (B, H*W)
            attn_scores = attn_scores.view(B, H, W)
            
            # Apply cumulative mask (already attended regions have low weight)
            # Clamp mask to prevent log(0) and extreme negative values
            safe_mask = cumulative_mask.clamp(min=1e-6)
            attn_scores = attn_scores + torch.log(safe_mask)
            
            # Clamp attention scores to prevent extreme values
            attn_scores = attn_scores.clamp(min=-50.0, max=50.0)
            
            # Apply Gumbel-softmax for differentiable attention
            attention = gumbel_softmax_2d(attn_scores, temperature=temperature)
            
            # Compute centroid
            centroid = self._compute_centroid(attention, H, W)
            
            # Compute attended features for stop prediction
            attention_flat = attention.view(B, H * W, 1)
            attended_features = (v * attention_flat).sum(dim=1)  # (B, D)
            
            # Compute attention entropy for this clue
            # Low entropy = sharp attention (confident) → can stop
            # High entropy = diffuse attention (uncertain) → need more clues
            # This couples attention quality directly to stopping decision!
            attn_clamped = attention.view(B, -1).clamp(min=1e-10)  # (B, H*W)
            attn_entropy = -(attn_clamped * torch.log(attn_clamped)).sum(dim=-1, keepdim=True)  # (B, 1)
            
            # Normalize entropy to [0, 1] range for stable learning
            # Max entropy for uniform distribution over H*W pixels = log(H*W)
            max_entropy = math.log(H * W + 1e-6)
            attn_entropy_normalized = attn_entropy / max_entropy  # (B, 1)
            
            # Concatenate attended features with normalized entropy
            # The entropy provides a "confidence" signal to the stop predictor
            stop_input = torch.cat([attended_features, attn_entropy_normalized], dim=-1)  # (B, D+1)
            
            # Predict stop probability with entropy-aware input
            stop_logit = self.stop_predictor(stop_input).squeeze(-1)  # (B,)
            
            # Update cumulative mask (reduce weight of attended regions)
            # Use soft masking to allow gradients to flow
            mask_update = 1.0 - 0.9 * attention.detach()
            cumulative_mask = cumulative_mask * mask_update
            # Ensure mask doesn't become too small
            cumulative_mask = cumulative_mask.clamp(min=1e-6)
            
            # Store results
            all_centroids.append(centroid)
            all_attention_maps.append(attention)
            all_stop_logits.append(stop_logit)
            all_entropy_inputs.append(attn_entropy_normalized.squeeze(-1))  # (B,)
        
        # Stack outputs
        centroids = torch.stack(all_centroids, dim=1)  # (B, K, 2)
        attention_maps = torch.stack(all_attention_maps, dim=1)  # (B, K, H, W)
        stop_logits = torch.stack(all_stop_logits, dim=1)  # (B, K)
        
        # Store entropy inputs for diagnostics (can be retrieved after forward)
        entropy_inputs = torch.stack(all_entropy_inputs, dim=1)  # (B, K)
        self._last_entropy_inputs = entropy_inputs.detach()
        
        return centroids, attention_maps, stop_logits
    
    def get_last_entropy_inputs(self) -> torch.Tensor:
        """Get the normalized entropy inputs used in last forward pass for diagnostics."""
        return getattr(self, '_last_entropy_inputs', None)
    
    def get_active_clues(
        self, 
        stop_logits: torch.Tensor, 
        threshold: float = 0.5
    ) -> torch.Tensor:
        """
        Determine which clues are active based on stop predictions.
        
        Args:
            stop_logits: Shape (B, K) stop probability logits
            threshold: Probability threshold for stopping
            
        Returns:
            active_mask: Shape (B, K) boolean mask of active clues
        """
        stop_probs = torch.sigmoid(stop_logits)
        
        # A clue is active if we haven't stopped yet
        # Cumulative product of (1 - stop_prob) gives probability of reaching this clue
        not_stopped = 1.0 - stop_probs
        reach_prob = torch.cumprod(
            torch.cat([torch.ones_like(not_stopped[:, :1]), not_stopped[:, :-1]], dim=1),
            dim=1
        )
        
        # Active if we have high probability of reaching this clue
        active_mask = reach_prob > threshold
        
        return active_mask
