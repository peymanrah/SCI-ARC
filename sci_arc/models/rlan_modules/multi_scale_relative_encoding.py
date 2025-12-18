"""
Multi-Scale Relative Encoding (MSRE) for RLAN

The MSRE module transforms absolute spatial coordinates into multiple
relative coordinate representations anchored at clue centroids.

Three Coordinate Types:
1. Absolute Offset: Δr = position - centroid (translation equivariant)
2. Normalized Offset: Δr / grid_size (scale invariant)
3. Polar Coordinates: (||Δr||, atan2(Δr)) (rotation aware)

This allows the model to:
- Reason about "3 cells to the right" (absolute)
- Reason about "halfway across the grid" (normalized)
- Reason about "in the diagonal direction" (polar)
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleRelativeEncoding(nn.Module):
    """
    Multi-Scale Relative Encoding - creates relative coordinate features.
    
    For each clue centroid, this module:
    1. Computes relative coordinates for every grid position
    2. Encodes these coordinates using learnable embeddings
    3. Concatenates with original features
    
    The encoding allows the model to understand spatial relationships
    relative to discovered anchors, rather than absolute positions.
    """
    
    def __init__(
        self,
        hidden_dim: int = 128,
        encoding_dim: int = 32,
        max_size: int = 30,
        num_freq: int = 8,
    ):
        """
        Args:
            hidden_dim: Input feature dimension
            encoding_dim: Output encoding dimension per coordinate type
            max_size: Maximum grid dimension
            num_freq: Number of frequency bands for positional encoding
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.encoding_dim = encoding_dim
        self.max_size = max_size
        self.num_freq = num_freq
        
        # Total raw coordinate features:
        # - Absolute offset: 2 (row, col)
        # - Normalized offset: 2 (row, col)
        # - Polar: 2 (radius, angle)
        # Each encoded with sin/cos at num_freq frequencies
        raw_coord_dim = 6  # 2 + 2 + 2
        fourier_dim = raw_coord_dim * num_freq * 2  # sin + cos for each freq
        
        # MLP to project coordinates to encoding
        self.coord_encoder = nn.Sequential(
            nn.Linear(raw_coord_dim + fourier_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, encoding_dim),
            nn.LayerNorm(encoding_dim),
        )
        
        # Feature fusion: combine original features with coordinate encoding
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + encoding_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        
        # Precompute frequency bands for Fourier encoding
        freqs = 2.0 ** torch.linspace(0, num_freq - 1, num_freq)
        self.register_buffer('freqs', freqs)
        
        # Precompute coordinate grids
        self._init_coord_grids(max_size)
    
    def _init_coord_grids(self, max_size: int):
        """Initialize coordinate grids."""
        rows = torch.arange(max_size).float()
        cols = torch.arange(max_size).float()
        row_grid, col_grid = torch.meshgrid(rows, cols, indexing='ij')
        
        # Clone to ensure contiguous memory (needed for checkpoint loading)
        self.register_buffer('row_grid', row_grid.clone())
        self.register_buffer('col_grid', col_grid.clone())
    
    def _fourier_encode(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Apply Fourier positional encoding to coordinates.
        
        Args:
            coords: Shape (..., D) coordinates
            
        Returns:
            encoded: Shape (..., D * num_freq * 2) Fourier features
        """
        # coords: (..., D)
        # freqs: (num_freq,)
        # Output: (..., D, num_freq, 2) -> (..., D * num_freq * 2)
        
        coords_expanded = coords.unsqueeze(-1) * self.freqs * math.pi  # (..., D, num_freq)
        sin_features = torch.sin(coords_expanded)
        cos_features = torch.cos(coords_expanded)
        
        fourier = torch.stack([sin_features, cos_features], dim=-1)  # (..., D, num_freq, 2)
        fourier = fourier.reshape(*coords.shape[:-1], -1)  # (..., D * num_freq * 2)
        
        return fourier
    
    def _compute_relative_coords(
        self,
        centroids: torch.Tensor,
        grid_sizes: Optional[torch.Tensor],
        H: int,
        W: int,
    ) -> torch.Tensor:
        """
        Compute all three coordinate representations.
        
        Args:
            centroids: Shape (B, K, 2) clue centroids (row, col)
            grid_sizes: Shape (B, 2) original grid sizes (h, w) or None
            H, W: Current grid dimensions
            
        Returns:
            coords: Shape (B, K, H, W, 6) with [abs_row, abs_col, norm_row, norm_col, radius, angle]
        """
        B, K, _ = centroids.shape
        device = centroids.device
        
        # Get coordinate grids
        row_grid = self.row_grid[:H, :W]  # (H, W)
        col_grid = self.col_grid[:H, :W]  # (H, W)
        
        # Expand for batch and clues
        row_grid = row_grid.view(1, 1, H, W).expand(B, K, -1, -1)
        col_grid = col_grid.view(1, 1, H, W).expand(B, K, -1, -1)
        
        # Expand centroids for broadcasting
        centroid_row = centroids[:, :, 0].view(B, K, 1, 1)  # (B, K, 1, 1)
        centroid_col = centroids[:, :, 1].view(B, K, 1, 1)  # (B, K, 1, 1)
        
        # 1. Absolute offset
        abs_row = row_grid - centroid_row  # (B, K, H, W)
        abs_col = col_grid - centroid_col  # (B, K, H, W)
        
        # 2. Normalized offset (by max grid dimension for scale invariance)
        # Paper: "divide by max(H,W)" for consistent normalization across aspect ratios
        if grid_sizes is not None:
            max_dim = torch.max(grid_sizes[:, 0], grid_sizes[:, 1]).view(B, 1, 1, 1).float()
        else:
            max_dim = torch.tensor(max(H, W), device=device, dtype=torch.float).view(1, 1, 1, 1)
        
        norm_row = abs_row / (max_dim + 1e-6)
        norm_col = abs_col / (max_dim + 1e-6)
        
        # 3. Log-Polar coordinates (paper formulation)
        # Paper: r = log(sqrt(dx^2 + dy^2) + 1)
        # Log-radius is specifically helpful for multi-scale generalization
        # and "rings / dilation / expansion" behaviors in ARC
        euclidean_dist = torch.sqrt(abs_row ** 2 + abs_col ** 2 + 1e-6)
        log_radius = torch.log(euclidean_dist + 1)  # Paper's log-polar formulation
        
        # Paper: angle = arctan2(j - mu_x, i - mu_y) = arctan2(col_offset, row_offset)
        # Note: atan2(y, x) convention - col is 'x', row is 'y' in image coords
        angle = torch.atan2(abs_col, abs_row)  # Range: [-pi, pi]
        
        # Normalize log-radius by log of max possible distance (grid diagonal)
        if grid_sizes is not None:
            h_size = grid_sizes[:, 0].view(B, 1, 1, 1).float()
            w_size = grid_sizes[:, 1].view(B, 1, 1, 1).float()
            max_dist = torch.sqrt(h_size ** 2 + w_size ** 2)
        else:
            max_dist = torch.sqrt(torch.tensor(H**2 + W**2, device=device, dtype=torch.float))
        log_radius_norm = log_radius / (torch.log(max_dist + 1) + 1e-6)
        
        # Normalize angle to [0, 1]
        angle_norm = (angle + math.pi) / (2 * math.pi)
        
        # Stack all coordinates
        coords = torch.stack([
            abs_row / self.max_size,  # Normalize absolute by max size
            abs_col / self.max_size,
            norm_row,
            norm_col,
            log_radius_norm,  # Changed from linear radius to log-polar (paper)
            angle_norm,
        ], dim=-1)  # (B, K, H, W, 6)
        
        return coords
    
    def forward(
        self,
        features: torch.Tensor,
        centroids: torch.Tensor,
        grid_sizes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute multi-scale relative encoding.
        
        Args:
            features: Shape (B, D, H, W) encoded grid features
            centroids: Shape (B, K, 2) clue centroids
            grid_sizes: Optional (B, 2) original grid sizes
            
        Returns:
            encoded: Shape (B, K, D, H, W) features with relative encoding
        """
        B, D, H, W = features.shape
        K = centroids.shape[1]
        
        # Compute relative coordinates
        rel_coords = self._compute_relative_coords(centroids, grid_sizes, H, W)  # (B, K, H, W, 6)
        
        # Apply Fourier encoding
        fourier_features = self._fourier_encode(rel_coords)  # (B, K, H, W, 6 * num_freq * 2)
        
        # Combine raw and Fourier features
        coord_input = torch.cat([rel_coords, fourier_features], dim=-1)  # (B, K, H, W, total_dim)
        
        # Encode coordinates
        coord_encoding = self.coord_encoder(coord_input)  # (B, K, H, W, encoding_dim)
        
        # Expand features for each clue
        features_expanded = features.unsqueeze(1).expand(-1, K, -1, -1, -1)  # (B, K, D, H, W)
        features_permuted = features_expanded.permute(0, 1, 3, 4, 2)  # (B, K, H, W, D)
        
        # Fuse features with coordinate encoding
        combined = torch.cat([features_permuted, coord_encoding], dim=-1)  # (B, K, H, W, D + encoding_dim)
        fused = self.fusion(combined)  # (B, K, H, W, D)
        
        # Permute back to channel-first format
        output = fused.permute(0, 1, 4, 2, 3)  # (B, K, D, H, W)
        
        return output
