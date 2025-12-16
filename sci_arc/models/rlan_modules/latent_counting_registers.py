"""
Latent Counting Registers (LCR) for RLAN

The LCR module provides non-spatial numerical reasoning capabilities
that pure spatial approaches lack. It answers questions like:
- "How many red pixels are there?"
- "Is blue the majority color?"
- "Are there more pixels of color A than color B?"

Key Features:
- Differentiable soft counting via one-hot color masks
- Per-color count embeddings with Fourier encoding
- Cross-attention to aggregate color-specific features
- Enables counting-based reasoning without explicit supervision
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentCountingRegisters(nn.Module):
    """
    Latent Counting Registers - soft counting for numerical reasoning.
    
    Architecture:
        1. Create one-hot masks for each color
        2. Count pixels per color (differentiable)
        3. Embed counts using Fourier features
        4. Cross-attend to color-masked features
        5. Produce per-color embeddings
    
    The output embeddings encode both:
    - How many pixels of each color exist
    - What those pixels look like (attended features)
    """
    
    def __init__(
        self,
        num_colors: int = 10,
        hidden_dim: int = 128,
        num_freq: int = 8,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        """
        Args:
            num_colors: Number of ARC colors (0-9)
            hidden_dim: Output embedding dimension per color
            num_freq: Frequency bands for count encoding
            num_heads: Attention heads for feature aggregation
            dropout: Dropout probability
        """
        super().__init__()
        
        self.num_colors = num_colors
        self.hidden_dim = hidden_dim
        self.num_freq = num_freq
        self.num_heads = num_heads
        
        # Count embedding: scalar count -> vector
        # Input: count + fourier features
        count_input_dim = 1 + 2 * num_freq  # count + sin/cos
        self.count_encoder = nn.Sequential(
            nn.Linear(count_input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        
        # Learnable color queries (one per color)
        self.color_queries = nn.Parameter(torch.randn(num_colors, hidden_dim))
        nn.init.xavier_uniform_(self.color_queries.unsqueeze(0))
        
        # Cross-attention: color query attends to color-masked features
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # count_embed + attended
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        
        # Precompute frequency bands
        freqs = 2.0 ** torch.linspace(0, num_freq - 1, num_freq)
        self.register_buffer('freqs', freqs)
        
        self.dropout = nn.Dropout(dropout)
    
    def _fourier_encode_count(self, counts: torch.Tensor) -> torch.Tensor:
        """
        Encode counts using Fourier features.
        
        Args:
            counts: Shape (B, C) normalized counts [0, 1]
            
        Returns:
            encoded: Shape (B, C, 1 + 2*num_freq) with [count, sin, cos, ...]
        """
        B, C = counts.shape
        
        # Expand for frequencies: (B, C, 1) * (num_freq,) -> (B, C, num_freq)
        counts_expanded = counts.unsqueeze(-1) * self.freqs * math.pi
        
        sin_features = torch.sin(counts_expanded)
        cos_features = torch.cos(counts_expanded)
        
        # Combine: [count, sin0, cos0, sin1, cos1, ...]
        encoded = torch.cat([
            counts.unsqueeze(-1),
            sin_features,
            cos_features,
        ], dim=-1)
        
        return encoded
    
    def forward(
        self,
        grid: torch.Tensor,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute count embeddings for each color.
        
        Args:
            grid: Shape (B, H, W) input grid with color indices 0-9
            features: Shape (B, D, H, W) encoded features
            mask: Optional (B, H, W) validity mask
            
        Returns:
            count_embedding: Shape (B, num_colors, hidden_dim)
        """
        B, D, H, W = features.shape
        C = self.num_colors
        device = grid.device
        
        # Create one-hot color masks: (B, H, W) -> (B, C, H, W)
        grid_clamped = grid.clamp(0, C - 1).long()
        color_masks = F.one_hot(grid_clamped, num_classes=C)  # (B, H, W, C)
        color_masks = color_masks.permute(0, 3, 1, 2).float()  # (B, C, H, W)
        
        if mask is not None:
            color_masks = color_masks * mask.unsqueeze(1)
        
        # Count pixels per color (normalized by grid area)
        total_pixels = H * W
        if mask is not None:
            total_pixels = mask.sum(dim=(-2, -1), keepdim=True).unsqueeze(1) + 1e-6
        
        counts = color_masks.sum(dim=(-2, -1))  # (B, C)
        counts_normalized = counts / total_pixels.squeeze() if isinstance(total_pixels, torch.Tensor) else counts / total_pixels
        
        # Encode counts
        count_features = self._fourier_encode_count(counts_normalized)  # (B, C, 1 + 2*num_freq)
        count_embed = self.count_encoder(count_features)  # (B, C, hidden_dim)
        
        # Prepare features for cross-attention
        features_flat = features.permute(0, 2, 3, 1).reshape(B, H * W, D)  # (B, H*W, D)
        
        # For each color, attend to its masked features
        color_embeddings = []
        
        for c in range(C):
            # Get mask for this color
            c_mask = color_masks[:, c].reshape(B, H * W)  # (B, H*W)
            
            # Create attention mask (True = ignore)
            # Positions with this color should be attended to
            attn_mask = (c_mask < 0.5)  # (B, H*W)
            
            # Check if any valid positions exist
            has_color = c_mask.sum(dim=-1) > 0  # (B,)
            
            # Get query for this color
            query = self.color_queries[c:c+1].unsqueeze(0).expand(B, -1, -1)  # (B, 1, D)
            query = query + count_embed[:, c:c+1, :]  # Add count information to query
            
            # Cross-attention
            # For samples without this color, attention will produce zeros
            attended, _ = self.cross_attention(
                query=query,
                key=features_flat,
                value=features_flat,
                key_padding_mask=attn_mask,
                need_weights=False,
            )  # (B, 1, D)
            
            # Zero out for samples without this color
            attended = attended * has_color.view(B, 1, 1)
            
            color_embeddings.append(attended.squeeze(1))  # (B, D)
        
        # Stack color embeddings
        attended_features = torch.stack(color_embeddings, dim=1)  # (B, C, D)
        
        # Combine count embedding with attended features
        combined = torch.cat([count_embed, attended_features], dim=-1)  # (B, C, 2*D)
        output = self.output_proj(combined)  # (B, C, D)
        
        return output
    
    def get_count_probs(self, grid: torch.Tensor) -> torch.Tensor:
        """
        Get normalized count distribution (useful for interpretability).
        
        Args:
            grid: Shape (B, H, W) input grid
            
        Returns:
            probs: Shape (B, num_colors) probability of each color
        """
        B, H, W = grid.shape
        C = self.num_colors
        
        grid_clamped = grid.clamp(0, C - 1).long()
        color_masks = F.one_hot(grid_clamped, num_classes=C).float()  # (B, H, W, C)
        
        counts = color_masks.sum(dim=(1, 2))  # (B, C)
        probs = counts / counts.sum(dim=-1, keepdim=True)
        
        return probs
