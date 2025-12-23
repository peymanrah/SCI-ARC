"""
Context Encoder for RLAN

Encodes training example pairs (input → output) to provide task context
for the model before it attempts to solve the test case.

This is CRITICAL for ARC - the model must understand the transformation
pattern from examples before predicting.

Architecture:
    Train Pairs [(in1, out1), (in2, out2), ...] 
         ↓
    Encode each pair
         ↓
    Cross-attention aggregation
         ↓
    Context vector (B, D)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class PairEncoder(nn.Module):
    """
    Encode a single input-output pair to capture the transformation.
    
    Uses difference-based encoding: what changed from input to output?
    
    IMPORTANT: Uses same embedding structure as GridEncoder to ensure
    consistent color representation across the model:
    - color_embed: hidden_dim // 2 (same as GridEncoder)
    - pos_embed: hidden_dim // 2, combined to hidden_dim
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_colors: int = 10,
        max_size: int = 30,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_colors = num_colors
        
        # Color embedding - MATCH GridEncoder: hidden_dim // 2
        # This ensures consistent color representation across model
        self.color_embed = nn.Embedding(num_colors + 1, hidden_dim // 2)  # +1 for padding
        
        # Positional encoding - MATCH GridEncoder: hidden_dim // 2
        # Combined with color to get full hidden_dim
        self.pos_embed = nn.Parameter(torch.randn(1, max_size, max_size, hidden_dim // 2) * 0.02)
        
        # Project combined color+pos to hidden_dim (matching GridEncoder's proj layer)
        self.embed_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        
        # Encode input
        self.input_encoder = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
        )
        
        # Encode output
        self.output_encoder = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
        )
        
        # Difference encoder - captures what changed
        self.diff_encoder = nn.Sequential(
            nn.Conv2d(hidden_dim * 3, hidden_dim, 1),  # input, output, diff
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
        )
        
        # Pool to single vector
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # Final projection
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
    
    def forward(
        self,
        input_grid: torch.Tensor,  # (B, H, W)
        output_grid: torch.Tensor,  # (B, H, W)
    ) -> torch.Tensor:
        """
        Encode input-output pair.
        
        Returns:
            pair_embedding: (B, D)
        """
        B, H, W = input_grid.shape
        
        # Embed colors: (B, H, W, D//2) - matching GridEncoder
        input_color = self.color_embed(input_grid.clamp(0, self.num_colors))
        output_color = self.color_embed(output_grid.clamp(0, self.num_colors))
        
        # Get positional encoding: (B, H, W, D//2)
        pos = self.pos_embed[:, :H, :W, :].expand(B, -1, -1, -1)
        
        # Combine color + position: (B, H, W, D) - same as GridEncoder
        input_embed = torch.cat([input_color, pos], dim=-1)
        output_embed = torch.cat([output_color, pos], dim=-1)
        
        # Project combined embeddings (matching GridEncoder's proj+norm)
        input_embed = self.embed_proj(input_embed)
        output_embed = self.embed_proj(output_embed)
        
        # Convert to channel-first: (B, D, H, W)
        input_embed = input_embed.permute(0, 3, 1, 2)
        output_embed = output_embed.permute(0, 3, 1, 2)
        
        # Encode each grid
        input_enc = self.input_encoder(input_embed)
        output_enc = self.output_encoder(output_embed)
        
        # Compute difference
        diff_enc = output_enc - input_enc
        
        # Concatenate and encode transformation
        combined = torch.cat([input_enc, output_enc, diff_enc], dim=1)
        pair_features = self.diff_encoder(combined)  # (B, D, H, W)
        
        # Pool and project
        pooled = self.pool(pair_features).squeeze(-1).squeeze(-1)  # (B, D)
        pair_embedding = self.proj(pooled)
        
        return pair_embedding


class SpatialPairEncoder(nn.Module):
    """
    Encode input-output pair while preserving spatial structure.
    
    Returns (B, D, H, W) instead of pooling to (B, D).
    This allows CrossAttentionInjector to attend to specific spatial regions.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_colors: int = 10,
        max_size: int = 30,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_colors = num_colors
        
        # Color embedding - MATCH GridEncoder: hidden_dim // 2
        self.color_embed = nn.Embedding(num_colors + 1, hidden_dim // 2)  # +1 for padding
        
        # Positional encoding - MATCH GridEncoder: hidden_dim // 2
        self.pos_embed = nn.Parameter(torch.randn(1, max_size, max_size, hidden_dim // 2) * 0.02)
        
        # Project combined color+pos to hidden_dim
        self.embed_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        
        # Encode input
        self.input_encoder = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
        )
        
        # Encode output
        self.output_encoder = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
        )
        
        # Difference encoder - captures what changed (NO POOLING)
        self.diff_encoder = nn.Sequential(
            nn.Conv2d(hidden_dim * 3, hidden_dim, 1),  # input, output, diff
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
        )
    
    def forward(
        self,
        input_grid: torch.Tensor,  # (B, H, W)
        output_grid: torch.Tensor,  # (B, H, W)
    ) -> torch.Tensor:
        """
        Encode input-output pair with spatial structure preserved.
        
        Returns:
            pair_features: (B, D, H, W) - NOT pooled!
        """
        B, H, W = input_grid.shape
        
        # Embed colors: (B, H, W, D//2)
        input_color = self.color_embed(input_grid.clamp(0, self.num_colors))
        output_color = self.color_embed(output_grid.clamp(0, self.num_colors))
        
        # Get positional encoding: (B, H, W, D//2)
        pos = self.pos_embed[:, :H, :W, :].expand(B, -1, -1, -1)
        
        # Combine color + position: (B, H, W, D)
        input_embed = torch.cat([input_color, pos], dim=-1)
        output_embed = torch.cat([output_color, pos], dim=-1)
        
        # Project combined embeddings
        input_embed = self.embed_proj(input_embed)
        output_embed = self.embed_proj(output_embed)
        
        # Convert to channel-first: (B, D, H, W)
        input_embed = input_embed.permute(0, 3, 1, 2)
        output_embed = output_embed.permute(0, 3, 1, 2)
        
        # Encode each grid
        input_enc = self.input_encoder(input_embed)
        output_enc = self.output_encoder(output_embed)
        
        # Compute difference
        diff_enc = output_enc - input_enc
        
        # Concatenate and encode transformation
        combined = torch.cat([input_enc, output_enc, diff_enc], dim=1)
        pair_features = self.diff_encoder(combined)  # (B, D, H, W)
        
        # Return spatial features (NO POOLING!)
        return pair_features


class ContextEncoder(nn.Module):
    """
    Encode multiple training pairs into a task context vector.
    
    Uses cross-attention to aggregate information from all pairs.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_colors: int = 10,
        max_size: int = 30,
        max_pairs: int = 5,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_spatial_features: bool = True,  # New: return spatial features
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.max_pairs = max_pairs
        self.use_spatial_features = use_spatial_features
        
        # Pair encoder
        self.pair_encoder = PairEncoder(hidden_dim, num_colors, max_size)
        
        # Learnable query for aggregation (only for compressed mode)
        self.context_query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        
        # Cross-attention to aggregate pairs (only for compressed mode)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Final processing (only for compressed mode)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        
        # Spatial pair encoder for cross-attention mode
        if use_spatial_features:
            self.spatial_pair_encoder = SpatialPairEncoder(
                hidden_dim=hidden_dim,
                num_colors=num_colors,
                max_size=max_size,
            )
    
    def forward(
        self,
        input_grids: torch.Tensor,   # (B, N, H, W) - N pairs
        output_grids: torch.Tensor,  # (B, N, H, W)
        pair_mask: Optional[torch.Tensor] = None,  # (B, N) - valid pairs
    ) -> torch.Tensor:
        """
        Encode all training pairs into context.
        
        Args:
            input_grids: Batch of input grids for all pairs
            output_grids: Batch of output grids for all pairs
            pair_mask: Boolean mask for valid pairs (True = valid)
            
        Returns:
            If use_spatial_features=True:
                support_features: (B, N, D, H, W) spatial features per pair
            If use_spatial_features=False:
                context: (B, D) aggregated context vector (legacy)
        """
        B, N, H, W = input_grids.shape
        
        # NEW: Spatial features mode for CrossAttentionInjector
        if self.use_spatial_features:
            # Encode each pair with spatial structure preserved
            support_features = []
            for i in range(N):
                pair_feat = self.spatial_pair_encoder(
                    input_grids[:, i],   # (B, H, W)
                    output_grids[:, i],  # (B, H, W)
                )  # (B, D, H, W)
                support_features.append(pair_feat)
            
            # Stack: (B, N, D, H, W)
            support_features = torch.stack(support_features, dim=1)
            
            # Apply pair_mask if provided (zero out invalid pairs)
            if pair_mask is not None:
                # Expand mask: (B, N, 1, 1, 1)
                mask_expanded = pair_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                support_features = support_features * mask_expanded.float()
            
            return support_features
        
        # LEGACY: Compressed vector mode for FiLM
        else:
            # Encode each pair
            pair_embeddings = []
            for i in range(N):
                pair_emb = self.pair_encoder(
                    input_grids[:, i],  # (B, H, W)
                    output_grids[:, i],  # (B, H, W)
                )
                pair_embeddings.append(pair_emb)
            
            # Stack: (B, N, D)
            pair_embeddings = torch.stack(pair_embeddings, dim=1)
            
            # Create attention mask (inverted for MultiheadAttention)
            if pair_mask is not None:
                # MultiheadAttention expects True = ignore
                attn_mask = ~pair_mask
            else:
                attn_mask = None
            
            # Expand query: (B, 1, D)
            query = self.context_query.expand(B, -1, -1)
            
            # Cross-attention: query attends to all pair embeddings
            context, _ = self.cross_attn(
                query=query,
                key=pair_embeddings,
                value=pair_embeddings,
                key_padding_mask=attn_mask,
            )
            
            # Remove sequence dimension: (B, D)
            context = context.squeeze(1)
            
            # Final processing
            context = context + self.ffn(context)
            
            return context


class ContextInjector(nn.Module):
    """
    Inject context into spatial features using FiLM conditioning.
    
    FiLM: Feature-wise Linear Modulation
    y = γ(context) * x + β(context)
    
    Scale uses 2*Sigmoid to allow both attenuation AND amplification:
    - Scale in [0, 2]: values < 1 attenuate, values > 1 amplify
    - This is more expressive than pure Sigmoid [0, 1]
    """
    
    def __init__(
        self,
        hidden_dim: int,
        scale_range: float = 2.0,  # Maximum scale factor (default allows [0, 2])
    ):
        super().__init__()
        
        self.scale_range = scale_range
        
        # Scale projection: output in [0, scale_range] to allow amplification
        self.scale_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.shift_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
        )
    
    def forward(
        self,
        features: torch.Tensor,  # (B, D, H, W)
        context: torch.Tensor,   # (B, D)
    ) -> torch.Tensor:
        """
        Modulate features based on context.
        
        Scale is in [0, scale_range] (default [0, 2]):
        - scale < 1: attenuate features
        - scale = 1: identity (no change)
        - scale > 1: amplify features
        
        Returns:
            modulated: (B, D, H, W)
        """
        # Compute modulation parameters
        # Scale uses sigmoid * scale_range for [0, scale_range] output
        scale = torch.sigmoid(self.scale_proj(context)) * self.scale_range  # (B, D)
        shift = self.shift_proj(context)  # (B, D)
        
        # Reshape for broadcasting: (B, D, 1, 1)
        scale = scale.unsqueeze(-1).unsqueeze(-1)
        shift = shift.unsqueeze(-1).unsqueeze(-1)
        
        # Apply FiLM
        modulated = scale * features + shift
        
        return modulated


class CrossAttentionInjector(nn.Module):
    """
    Inject context into spatial features using Cross-Attention instead of FiLM.
    
    This allows DSC to attend to ALL support set pixels directly,
    removing the information bottleneck of compressing to a single vector.
    
    Architecture inspired by TRM's "Direct Support Attention".
    
    Args:
        features: (B, D, H, W) test grid features (query)
        support_features: (B, N, D, H', W') encoded support pairs (key/value)
    
    Returns:
        modulated: (B, D, H, W) features enhanced with support context
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Multi-head cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Layer norm for stability
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Feed-forward network after attention
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        features: torch.Tensor,          # (B, D, H, W) test grid
        support_features: torch.Tensor,  # (B, N, D, H', W') support pairs
    ) -> torch.Tensor:
        """
        Apply cross-attention from test features to support features.
        
        Returns:
            modulated: (B, D, H, W) with residual connection
        """
        B, D, H, W = features.shape
        B_s, N, D_s, H_s, W_s = support_features.shape
        
        assert D == D_s, f"Feature dims must match: {D} != {D_s}"
        
        # Flatten test features to sequence: (B, H*W, D)
        features_flat = features.permute(0, 2, 3, 1).reshape(B, H * W, D)
        
        # Flatten support features: (B, N*H'*W', D)
        support_flat = support_features.permute(0, 1, 3, 4, 2).reshape(B, N * H_s * W_s, D_s)
        
        # Cross-attention: test pixels query support pixels
        # Query: test grid, Key/Value: support set
        attn_output, attn_weights = self.cross_attn(
            query=features_flat,
            key=support_flat,
            value=support_flat,
        )
        
        # Residual connection + norm
        features_flat = self.norm1(features_flat + self.dropout(attn_output))
        
        # FFN with residual
        features_flat = self.norm2(features_flat + self.dropout(self.ffn(features_flat)))
        
        # Reshape back to spatial: (B, D, H, W)
        modulated = features_flat.reshape(B, H, W, D).permute(0, 3, 1, 2)
        
        return modulated
