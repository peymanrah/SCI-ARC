"""
Positional Encodings for RLAN

Provides multiple positional encoding options:
1. Learned embeddings (current default)
2. RoPE - Rotary Position Embeddings (used by TRM, LLaMA)
3. Hybrid - Learned for absolute, RoPE for relative

RoPE is particularly suited for relative position reasoning,
which aligns well with RLAN's MSRE module philosophy.

Reference:
- RoFormer: Enhanced Transformer with Rotary Position Embedding (Su et al., 2021)
- LLaMA: Open and Efficient Foundation Language Models
"""

import math
import torch
import torch.nn as nn
from typing import Tuple, Optional


class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) for 2D grids.
    
    RoPE encodes relative positions by rotating query/key vectors,
    making attention scores dependent on relative (not absolute) positions.
    
    For 2D grids, we apply RoPE separately for x and y dimensions,
    splitting the embedding dimension in half.
    
    Args:
        dim: Embedding dimension (will be split for x/y)
        max_size: Maximum grid size
        base: Base for frequency computation (default 10000)
    """
    
    def __init__(
        self,
        dim: int,
        max_size: int = 30,
        base: float = 10000.0,
    ):
        super().__init__()
        
        self.dim = dim
        self.max_size = max_size
        self.base = base
        
        # Compute frequencies for half dimension (other half for y)
        half_dim = dim // 4  # Quarter for x, quarter for y, then repeat
        inv_freq = 1.0 / (base ** (torch.arange(0, half_dim, 2).float() / half_dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Precompute sin/cos for max positions
        self._build_cache(max_size)
    
    def _build_cache(self, max_size: int):
        """Precompute sin/cos tables for efficiency."""
        t = torch.arange(max_size, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        
        # Duplicate for rotation: [sin, cos, sin, cos, ...]
        emb = torch.cat((freqs, freqs), dim=-1)
        
        self.register_buffer('cos_cached', emb.cos(), persistent=False)
        self.register_buffer('sin_cached', emb.sin(), persistent=False)
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions_x: Optional[torch.Tensor] = None,
        positions_y: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary embeddings to query and key tensors.
        
        Args:
            q: Query tensor (B, H, L, D) or (B, L, D)
            k: Key tensor (B, H, L, D) or (B, L, D)
            positions_x: X positions (B, L) or None for sequential
            positions_y: Y positions (B, L) or None for sequential
            
        Returns:
            q_rotated: Query with rotary embeddings
            k_rotated: Key with rotary embeddings
        """
        # Get sequence length
        if q.dim() == 4:
            seq_len = q.shape[2]
        else:
            seq_len = q.shape[1]
        
        # Get sin/cos for positions
        cos = self.cos_cached[:seq_len]
        sin = self.sin_cached[:seq_len]
        
        # Apply rotation
        q_rotated = self._apply_rotary(q, cos, sin)
        k_rotated = self._apply_rotary(k, cos, sin)
        
        return q_rotated, k_rotated
    
    def _apply_rotary(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """Apply rotary embedding to a tensor."""
        # Rotate half
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        x_rotated = torch.cat((-x2, x1), dim=-1)
        
        # Apply rotation using cos/sin
        return x * cos + x_rotated * sin


class RotaryPositionEmbedding2D(nn.Module):
    """
    2D Rotary Position Embedding for grid-structured data.
    
    Applies separate rotary embeddings for height and width dimensions,
    allowing attention to understand both vertical and horizontal
    relative positions.
    
    Args:
        dim: Embedding dimension
        max_h: Maximum height
        max_w: Maximum width
        base: Base for frequency computation
    """
    
    def __init__(
        self,
        dim: int,
        max_h: int = 30,
        max_w: int = 30,
        base: float = 10000.0,
    ):
        super().__init__()
        
        assert dim % 2 == 0, "dim must be even for 2D RoPE"
        
        self.dim = dim
        self.max_h = max_h
        self.max_w = max_w
        
        # Half dim for height, half for width
        half_dim = dim // 2
        
        # Separate RoPE for each axis
        self.rope_h = RotaryPositionEmbedding(half_dim, max_h, base)
        self.rope_w = RotaryPositionEmbedding(half_dim, max_w, base)
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        height: int,
        width: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply 2D rotary embeddings.
        
        Args:
            q: Query (B, L, D) where L = H * W
            k: Key (B, L, D)
            height: Grid height
            width: Grid width
            
        Returns:
            q_rotated, k_rotated: Tensors with 2D RoPE applied
        """
        B, L, D = q.shape
        half_d = D // 2
        
        # Split into height and width components
        q_h, q_w = q[..., :half_d], q[..., half_d:]
        k_h, k_w = k[..., :half_d], k[..., half_d:]
        
        # Reshape to 2D for position extraction
        q_h = q_h.view(B, height, width, half_d)
        q_w = q_w.view(B, height, width, half_d)
        k_h = k_h.view(B, height, width, half_d)
        k_w = k_w.view(B, height, width, half_d)
        
        # Apply RoPE along each axis
        # Height: vary along dim 1, constant along dim 2
        q_h_rot = self._apply_axis_rope(q_h, self.rope_h, axis=1)
        k_h_rot = self._apply_axis_rope(k_h, self.rope_h, axis=1)
        
        # Width: vary along dim 2, constant along dim 1
        q_w_rot = self._apply_axis_rope(q_w, self.rope_w, axis=2)
        k_w_rot = self._apply_axis_rope(k_w, self.rope_w, axis=2)
        
        # Combine and flatten back
        q_rot = torch.cat([q_h_rot, q_w_rot], dim=-1).view(B, L, D)
        k_rot = torch.cat([k_h_rot, k_w_rot], dim=-1).view(B, L, D)
        
        return q_rot, k_rot
    
    def _apply_axis_rope(
        self,
        x: torch.Tensor,
        rope: RotaryPositionEmbedding,
        axis: int,
    ) -> torch.Tensor:
        """Apply RoPE along a specific axis."""
        B, H, W, D = x.shape
        
        if axis == 1:  # Height axis
            # Transpose to make height the sequence dim
            x = x.permute(0, 2, 1, 3)  # (B, W, H, D)
            x = x.reshape(B * W, H, D)
            cos = rope.cos_cached[:H]
            sin = rope.sin_cached[:H]
            x = rope._apply_rotary(x, cos, sin)
            x = x.reshape(B, W, H, D).permute(0, 2, 1, 3)
        else:  # Width axis
            x = x.reshape(B * H, W, D)
            cos = rope.cos_cached[:W]
            sin = rope.sin_cached[:W]
            x = rope._apply_rotary(x, cos, sin)
            x = x.reshape(B, H, W, D)
        
        return x


class LearnedPositionEmbedding2D(nn.Module):
    """
    Learned 2D position embeddings.
    
    Separate embeddings for height and width, combined additively.
    Good for capturing absolute position patterns.
    
    Args:
        dim: Embedding dimension
        max_h: Maximum height
        max_w: Maximum width
    """
    
    def __init__(
        self,
        dim: int,
        max_h: int = 30,
        max_w: int = 30,
    ):
        super().__init__()
        
        self.h_embed = nn.Embedding(max_h, dim)
        self.w_embed = nn.Embedding(max_w, dim)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.trunc_normal_(self.h_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.w_embed.weight, std=0.02)
    
    def forward(self, height: int, width: int) -> torch.Tensor:
        """
        Get position embeddings for a grid.
        
        Args:
            height: Grid height
            width: Grid width
            
        Returns:
            pos_embed: (H, W, D) position embeddings
        """
        h_pos = torch.arange(height, device=self.h_embed.weight.device)
        w_pos = torch.arange(width, device=self.w_embed.weight.device)
        
        h_emb = self.h_embed(h_pos)[:, None, :]  # (H, 1, D)
        w_emb = self.w_embed(w_pos)[None, :, :]  # (1, W, D)
        
        # Combine with scaling to preserve variance
        return 0.707106781 * (h_emb + w_emb)  # (H, W, D)


class HybridPositionEncoding(nn.Module):
    """
    Hybrid positional encoding combining learned + RoPE.
    
    - Learned embeddings: Capture absolute grid structure
    - RoPE: Enhance relative position reasoning
    
    This is ideal for RLAN where:
    - GridEncoder needs absolute positions (edges, corners)
    - RecursiveSolver benefits from relative reasoning
    
    Args:
        dim: Embedding dimension
        max_h: Maximum height
        max_w: Maximum width
        use_learned: Whether to use learned embeddings
        use_rope: Whether to use RoPE
    """
    
    def __init__(
        self,
        dim: int,
        max_h: int = 30,
        max_w: int = 30,
        use_learned: bool = True,
        use_rope: bool = True,
    ):
        super().__init__()
        
        self.use_learned = use_learned
        self.use_rope = use_rope
        
        if use_learned:
            self.learned = LearnedPositionEmbedding2D(dim, max_h, max_w)
        
        if use_rope:
            self.rope = RotaryPositionEmbedding2D(dim, max_h, max_w)
    
    def get_absolute(self, height: int, width: int) -> Optional[torch.Tensor]:
        """Get learned absolute position embeddings."""
        if self.use_learned:
            return self.learned(height, width)
        return None
    
    def apply_relative(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        height: int,
        width: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE for relative position encoding."""
        if self.use_rope:
            return self.rope(q, k, height, width)
        return q, k
