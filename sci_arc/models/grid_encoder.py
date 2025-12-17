"""Grid Encoder for SCI-ARC

Encodes ARC grids (2D integer arrays with 10 colors) into continuous embeddings
suitable for processing by the structural and content encoders.

Key Features:
- Color embedding (10 colors → hidden_dim/2)
- Positional encoding options:
  - Sinusoidal 2D (default, absolute positions)
  - RoPE 2D (TRM-style, relative positions via Q/K rotation)
- Combined and projected to hidden_dim

This is analogous to token embedding in text models, adapted for 2D grids.
"""

import math
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import learned position embedding from positional_encoding module
try:
    from .rlan_modules.positional_encoding import LearnedPositionEmbedding2D
    LEARNED_POS_AVAILABLE = True
except ImportError:
    LEARNED_POS_AVAILABLE = False


class SinusoidalPositionalEncoding2D(nn.Module):
    """
    2D sinusoidal positional encoding for grids.
    
    Encodes (row, col) position using sin/cos functions at different frequencies.
    This allows the model to understand spatial relationships without learning positions.
    
    The encoding dimension is split: half for row position, half for column position.
    Each half uses alternating sin/cos with exponentially increasing frequencies.
    """
    
    def __init__(self, dim: int, max_size: int = 30):
        """
        Args:
            dim: Embedding dimension (will be split between row and col)
            max_size: Maximum grid size (ARC grids are max 30x30)
        """
        super().__init__()
        self.dim = dim
        self.max_size = max_size
        
        # Precompute positional encodings
        pe = self._create_positional_encoding(max_size, dim)
        self.register_buffer('pe', pe)
    
    def _create_positional_encoding(self, max_size: int, dim: int) -> torch.Tensor:
        """
        Create 2D positional encoding tensor.
        
        Returns:
            pe: [max_size, max_size, dim] tensor
        """
        pe = torch.zeros(max_size, max_size, dim)
        
        # Split dim: first half for rows, second half for cols
        dim_half = dim // 2
        
        # Compute division term for frequency scaling
        # Higher indices get lower frequencies
        div_term = torch.exp(torch.arange(0, dim_half, 2).float() * 
                            -(math.log(10000.0) / dim_half))
        
        # Row and column position indices
        row_pos = torch.arange(max_size).unsqueeze(1).float()  # [max_size, 1]
        col_pos = torch.arange(max_size).unsqueeze(0).float()  # [1, max_size]
        
        # Row encoding (first half of dim)
        for i, freq in enumerate(div_term):
            pe[:, :, 2*i] = torch.sin(row_pos * freq).expand(max_size, max_size)
            pe[:, :, 2*i + 1] = torch.cos(row_pos * freq).expand(max_size, max_size)
        
        # Column encoding (second half of dim)
        offset = dim_half
        for i, freq in enumerate(div_term):
            if offset + 2*i + 1 < dim:
                pe[:, :, offset + 2*i] = torch.sin(col_pos.T * freq).expand(max_size, max_size)
                pe[:, :, offset + 2*i + 1] = torch.cos(col_pos.T * freq).expand(max_size, max_size)
        
        return pe
    
    def forward(self, h: int, w: int) -> torch.Tensor:
        """
        Get positional encoding for grid of size h x w.
        
        Args:
            h: Height of grid
            w: Width of grid
            
        Returns:
            pos_encoding: [h, w, dim] positional encoding
        """
        return self.pe[:h, :w, :]


class GridEncoder(nn.Module):
    """
    Encode ARC grids into embeddings suitable for SCI processing.
    
    Architecture:
    1. Color embedding: Each cell color (0-9) → vector
    2. Position embedding: 2D sinusoidal encoding
    3. Combination: Concatenate and project
    
    Key differences from text encoders:
    - 2D positional encoding (not 1D)
    - Color embedding (not token embedding)
    - Per-cell output (not per-token)
    
    TRM-style improvements:
    - Embedding scaling by sqrt(hidden_dim) for stable training
    - Truncated normal initialization
    
    Parameters: ~500K (intentionally small like TRM philosophy)
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_colors: int = 10,
        max_size: int = 30,
        dropout: float = 0.1,
        use_embed_scale: bool = True,  # TRM-style scaling
        use_learned_pos: bool = False,  # Use learned instead of sinusoidal
    ):
        """
        Args:
            hidden_dim: Output embedding dimension
            num_colors: Number of ARC colors (typically 10: 0-9)
            max_size: Maximum grid dimension (ARC max is 30)
            dropout: Dropout probability
            use_embed_scale: Whether to use TRM-style embedding scaling
            use_learned_pos: Use learned positional embeddings instead of sinusoidal
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_colors = num_colors
        self.max_size = max_size
        self.use_embed_scale = use_embed_scale
        self.use_learned_pos = use_learned_pos and LEARNED_POS_AVAILABLE
        
        # TRM-style embedding scaling
        self.embed_scale = math.sqrt(hidden_dim) if use_embed_scale else 1.0
        embed_init_std = 1.0 / self.embed_scale  # Scale init inversely
        
        # Color embedding: each color gets a unique vector
        # Use hidden_dim // 2 to leave room for positional encoding
        self.color_embed = nn.Embedding(num_colors, hidden_dim // 2)
        
        # Positional encoding: sinusoidal (default) or learned
        if self.use_learned_pos:
            # Learned position embeddings (TRM uses this with RoPE)
            self.pos_embed = LearnedPositionEmbedding2D(hidden_dim // 2, max_size, max_size)
            self.pos_type = 'learned'
        else:
            # 2D sinusoidal positional encoding (original RLAN)
            self.pos_embed = SinusoidalPositionalEncoding2D(hidden_dim // 2, max_size)
            self.pos_type = 'sinusoidal'
        
        # Combine color and position
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        # TRM-style truncated normal initialization
        self._init_weights(embed_init_std)
    
    def _init_weights(self, embed_std: float):
        """Initialize weights with TRM-style truncated normal."""
        # Truncated normal for embeddings
        nn.init.trunc_normal_(self.color_embed.weight, std=embed_std, a=-2*embed_std, b=2*embed_std)
        # Standard init for projection
        nn.init.xavier_uniform_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)
    
    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        """
        Encode a grid into embeddings.
        
        Args:
            grid: [B, H, W] integer tensor with values 0-9
        
        Returns:
            embeddings: [B, H, W, hidden_dim] embedding tensor
        """
        B, H, W = grid.shape
        
        # Clamp grid values to valid range (safety)
        grid = grid.clamp(0, self.num_colors - 1)
        
        # Color embedding
        color_emb = self.color_embed(grid)  # [B, H, W, hidden_dim//2]
        
        # Positional encoding
        pos_emb = self.pos_embed(H, W)  # [H, W, hidden_dim//2]
        pos_emb = pos_emb.unsqueeze(0).expand(B, -1, -1, -1)  # [B, H, W, hidden_dim//2]
        
        # Combine color and position
        combined = torch.cat([color_emb, pos_emb], dim=-1)  # [B, H, W, hidden_dim]
        
        # Project and normalize
        output = self.proj(combined)
        output = self.norm(output)
        output = self.dropout(output)
        
        # TRM-style scaling
        if self.use_embed_scale:
            output = output * self.embed_scale
        
        return output
    
    def encode_pair(
        self, 
        input_grid: torch.Tensor, 
        output_grid: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode an (input, output) grid pair.
        
        Convenience method for encoding demo pairs.
        
        Args:
            input_grid: [B, H_in, W_in] input grid
            output_grid: [B, H_out, W_out] output grid
        
        Returns:
            input_emb: [B, H_in, W_in, D] input embeddings
            output_emb: [B, H_out, W_out, D] output embeddings
        """
        input_emb = self.forward(input_grid)
        output_emb = self.forward(output_grid)
        return input_emb, output_emb


class PatchGridEncoder(nn.Module):
    """
    Alternative grid encoder using patch-based approach.
    
    Instead of per-cell embeddings, divides grid into patches and
    encodes each patch. This can be more efficient for larger grids.
    
    Not used by default but provided as an alternative.
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_colors: int = 10,
        patch_size: int = 3,
        max_size: int = 30
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size
        
        # One-hot encode colors, then conv to embed patches
        self.patch_embed = nn.Conv2d(
            num_colors,  # One channel per color (one-hot)
            hidden_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0
        )
        
        # Positional encoding for patches
        max_patches = max_size // patch_size + 1
        self.pos_embed = nn.Parameter(
            torch.randn(1, max_patches, max_patches, hidden_dim) * 0.02
        )
        
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        """
        Encode grid using patches.
        
        Args:
            grid: [B, H, W] integer tensor
        
        Returns:
            embeddings: [B, H//patch, W//patch, hidden_dim]
        """
        B, H, W = grid.shape
        
        # One-hot encode colors
        one_hot = F.one_hot(grid.long(), num_classes=10)  # [B, H, W, 10]
        one_hot = one_hot.permute(0, 3, 1, 2).float()  # [B, 10, H, W]
        
        # Patch embedding via convolution
        patches = self.patch_embed(one_hot)  # [B, D, H//p, W//p]
        patches = patches.permute(0, 2, 3, 1)  # [B, H//p, W//p, D]
        
        # Add positional encoding
        pH, pW = patches.shape[1], patches.shape[2]
        patches = patches + self.pos_embed[:, :pH, :pW, :]
        
        return self.norm(patches)
