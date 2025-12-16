"""
Content Encoder for SCI-ARC

Extracts content/object information from grids, ORTHOGONAL to the structural
representation. This ensures the model separately understands:
- WHAT transformation (structural) 
- WHICH objects (content)

Key Components:
1. Object Queries: Learnable queries that attend to grid content
2. OrthogonalProjector: Ensures content representation is orthogonal to structure

The orthogonality constraint is critical for SCI:
- S(x) captures structure (transformation patterns)
- C(x) captures content (objects, colors, shapes)
- S(x) ⊥ C(x) ensures they don't leak into each other
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class OrthogonalProjector(nn.Module):
    """
    Projects content representation orthogonal to structure.
    
    Ensures S(x) ⊥ C(x) which is critical for SCI.
    
    Uses Gram-Schmidt-style orthogonalization:
    C_orth = C - proj_S(C)
    
    where proj_S(C) = (C · S / ||S||²) * S
    
    This is applied per-slot, ensuring each content slot is
    orthogonal to the corresponding structure slot.
    """
    
    def __init__(self, hidden_dim: int):
        """
        Args:
            hidden_dim: Dimension of representations
        """
        super().__init__()
        
        # Optional learned transformation after orthogonalization
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Learnable scale for orthogonal component
        self.ortho_scale = nn.Parameter(torch.ones(1))
    
    def forward(
        self,
        content: torch.Tensor,    # [B, N, D]
        structure: torch.Tensor   # [B, N, D] or [B, 1, D] (broadcast)
    ) -> torch.Tensor:
        """
        Project content orthogonal to structure.
        
        Args:
            content: [B, N, D] content representation
            structure: [B, N, D] or [B, 1, D] structure representation
        
        Returns:
            content_orthogonal: [B, N, D] orthogonalized content
        """
        # Normalize structure for stable projection
        structure_norm = F.normalize(structure, dim=-1, eps=1e-8)
        
        # Compute projection of content onto structure
        # proj_S(C) = (C · S_norm) * S_norm
        dot_product = (content * structure_norm).sum(dim=-1, keepdim=True)
        projection = dot_product * structure_norm
        
        # Subtract projection (Gram-Schmidt orthogonalization)
        content_orthogonal = content - projection
        
        # Scale and project
        content_orthogonal = content_orthogonal * self.ortho_scale
        content_orthogonal = self.proj(content_orthogonal)
        
        return self.norm(content_orthogonal)
    
    @staticmethod
    def orthogonality_loss(content: torch.Tensor, structure: torch.Tensor) -> torch.Tensor:
        """
        Compute orthogonality loss for training.
        
        L_orth = mean(|C · S|)
        
        This should be minimized to enforce orthogonality.
        """
        c_norm = F.normalize(content.mean(dim=1), dim=-1)
        s_norm = F.normalize(structure.mean(dim=1), dim=-1)
        
        # Absolute dot product (should be close to 0)
        ortho_loss = (c_norm * s_norm).sum(dim=-1).abs().mean()
        
        return ortho_loss


class ContentEncoder2D(nn.Module):
    """
    Extract content (objects) from grids, orthogonal to structure.
    
    Adaptation of SCI's Content Encoder for 2D grids:
    - Learns to detect and encode objects in the grid
    - Extracts per-object features (color, shape, size, position)
    - Projects orthogonal to structural representation
    
    Unlike StructuralEncoder which focuses on transformation patterns,
    ContentEncoder focuses on WHAT is in the grid (objects, colors, etc.)
    
    Parameters: ~1M
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        max_objects: int = 16,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Args:
            hidden_dim: Hidden dimension
            max_objects: Maximum number of content slots
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.max_objects = max_objects
        
        # === OBJECT QUERIES ===
        # Learnable queries that attend to grid content (like DETR)
        self.object_queries = nn.Parameter(
            torch.randn(1, max_objects, hidden_dim) * 0.02
        )
        
        # === CONTENT EXTRACTION ===
        # Cross-attention: object queries attend to grid
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Process content features
        self.content_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # === ORTHOGONAL PROJECTOR (Key SCI component) ===
        self.orthogonal_projector = OrthogonalProjector(hidden_dim)
        
        # Output normalization
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        grid_emb: torch.Tensor,      # [B, H, W, D]
        structure_rep: torch.Tensor  # [B, K, D] from StructuralEncoder
    ) -> torch.Tensor:
        """
        Extract content representation orthogonal to structure.
        
        Args:
            grid_emb: [B, H, W, D] encoded grid
            structure_rep: [B, K, D] structural slots from SE
        
        Returns:
            content_slots: [B, max_objects, D] content representation
        """
        B, H, W, D = grid_emb.shape
        
        # Flatten grid for attention
        grid_flat = grid_emb.view(B, H * W, D)  # [B, N, D]
        
        # Object queries attend to grid
        queries = self.object_queries.expand(B, -1, -1)  # [B, M, D]
        
        content_raw, attn_weights = self.cross_attention(
            query=queries,
            key=grid_flat,
            value=grid_flat
        )
        
        # Process content features
        content_features = self.content_mlp(content_raw)
        content_features = content_features + content_raw  # Residual
        
        # === FULL SUBSPACE ORTHOGONALIZATION ===
        # Project content orthogonal to the ENTIRE structure subspace
        # (not just the mean, which could miss slot-aligned content)
        # Formula: C_orth = C - sum_k(proj_Sk(C)) for all K structure slots
        #
        # This ensures content is orthogonal to every individual structure slot,
        # not just the centroid. A content vector could be orthogonal to mean(S)
        # but aligned with individual slots if they cancel out.
        
        content_orthogonal = content_features
        K = structure_rep.size(1)  # Number of structure slots
        
        for k in range(K):
            # Get k-th structure slot for all batches: [B, 1, D]
            s_k = structure_rep[:, k:k+1, :]  # [B, 1, D]
            # Broadcast to match content shape: [B, M, D]
            s_k_broadcast = s_k.expand(-1, self.max_objects, -1)
            
            # Subtract projection onto this slot (Gram-Schmidt step)
            s_k_norm = F.normalize(s_k_broadcast, dim=-1, eps=1e-8)
            dot_product = (content_orthogonal * s_k_norm).sum(dim=-1, keepdim=True)
            projection = dot_product * s_k_norm
            content_orthogonal = content_orthogonal - projection
        
        # Final projection and normalization via the projector
        # Note: We still use the projector for the learned transformation,
        # but pass zeros as structure since we already orthogonalized
        content_orthogonal = self.orthogonal_projector.proj(content_orthogonal)
        content_orthogonal = self.orthogonal_projector.norm(content_orthogonal)
        
        return self.norm(content_orthogonal)
    
    def forward_with_attention(
        self,
        grid_emb: torch.Tensor,
        structure_rep: torch.Tensor
    ):
        """Forward pass that also returns attention weights for visualization."""
        B, H, W, D = grid_emb.shape
        grid_flat = grid_emb.view(B, H * W, D)
        
        queries = self.object_queries.expand(B, -1, -1)
        content_raw, attn_weights = self.cross_attention(
            query=queries,
            key=grid_flat,
            value=grid_flat
        )
        
        content_features = self.content_mlp(content_raw) + content_raw
        
        structure_mean = structure_rep.mean(dim=1, keepdim=True)
        structure_broadcast = structure_mean.expand(-1, self.max_objects, -1)
        
        content_orthogonal = self.orthogonal_projector(
            content_features,
            structure_broadcast
        )
        
        # Reshape attention weights for visualization: [B, M, H, W]
        attn_map = attn_weights.view(B, self.max_objects, H, W)
        
        return self.norm(content_orthogonal), attn_map


class ConnectedComponentContentEncoder(nn.Module):
    """
    Alternative content encoder using explicit connected component detection.
    
    Instead of learning to attend to objects, explicitly detects connected
    components and encodes them.
    
    This is more interpretable but less flexible than learned queries.
    Not used by default.
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        max_objects: int = 16
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.max_objects = max_objects
        
        # Encode object features
        self.object_encoder = nn.Sequential(
            nn.Linear(hidden_dim + 5, hidden_dim),  # +5 for bbox: color, x, y, w, h
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.orthogonal_projector = OrthogonalProjector(hidden_dim)
    
    def _detect_objects(self, grid: torch.Tensor):
        """
        Detect connected components in grid.
        
        Args:
            grid: [H, W] integer grid
        
        Returns:
            objects: List of (color, bbox) tuples
        """
        # Simple flood-fill based detection
        # For production, use scipy.ndimage.label or similar
        objects = []
        visited = torch.zeros_like(grid, dtype=torch.bool)
        
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if not visited[i, j] and grid[i, j] > 0:
                    color = grid[i, j].item()
                    # BFS to find connected component
                    component = [(i, j)]
                    queue = [(i, j)]
                    visited[i, j] = True
                    
                    while queue:
                        ci, cj = queue.pop(0)
                        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            ni, nj = ci + di, cj + dj
                            if (0 <= ni < grid.shape[0] and 
                                0 <= nj < grid.shape[1] and
                                not visited[ni, nj] and
                                grid[ni, nj] == color):
                                visited[ni, nj] = True
                                queue.append((ni, nj))
                                component.append((ni, nj))
                    
                    # Compute bounding box
                    rows = [p[0] for p in component]
                    cols = [p[1] for p in component]
                    bbox = (min(rows), min(cols), max(rows) - min(rows) + 1, max(cols) - min(cols) + 1)
                    
                    objects.append((color, bbox))
        
        return objects[:self.max_objects]
    
    def forward(
        self,
        grid_emb: torch.Tensor,
        grid: torch.Tensor,
        structure_rep: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract content using connected component detection.
        
        Args:
            grid_emb: [B, H, W, D] encoded grid
            grid: [B, H, W] raw grid (for object detection)
            structure_rep: [B, K, D] structural slots
        
        Returns:
            content_slots: [B, max_objects, D]
        """
        B, H, W, D = grid_emb.shape
        
        all_content = []
        
        for b in range(B):
            objects = self._detect_objects(grid[b])
            
            object_embeddings = []
            for color, (y, x, h, w) in objects:
                # Extract object region embedding
                region = grid_emb[b, y:y+h, x:x+w, :]
                region_pool = region.mean(dim=(0, 1))  # [D]
                
                # Add normalized bbox features
                bbox_feat = torch.tensor(
                    [color / 9.0, x / W, y / H, w / W, h / H],
                    device=grid_emb.device
                )
                
                combined = torch.cat([region_pool, bbox_feat])
                obj_emb = self.object_encoder(combined)
                object_embeddings.append(obj_emb)
            
            # Pad to max_objects
            while len(object_embeddings) < self.max_objects:
                object_embeddings.append(torch.zeros(D, device=grid_emb.device))
            
            all_content.append(torch.stack(object_embeddings[:self.max_objects]))
        
        content = torch.stack(all_content)  # [B, M, D]
        
        # Orthogonalize
        structure_broadcast = structure_rep.mean(dim=1, keepdim=True).expand(-1, self.max_objects, -1)
        content_orthogonal = self.orthogonal_projector(content, structure_broadcast)
        
        return content_orthogonal
