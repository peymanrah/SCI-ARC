"""
Structural Encoder for SCI-ARC

The core SCI innovation adapted for 2D grids. Extracts transformation patterns
(WHAT transformation is being applied) while suppressing content-specific information
(WHICH objects are being transformed).

Key Components:
1. AbstractionLayer2D: Learns to identify and preserve structural features
2. Structure Queries: Learnable queries that extract transformation patterns
3. Transformation Encoder: Processes (input, output) difference

The key insight: For tasks with the SAME transformation (e.g., all "rotate 90°" tasks),
the structural representation S(demos) should be SIMILAR regardless of the specific
objects/colors being rotated.

This is enforced during training via Structural Contrastive Loss (SCL).
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding2D(nn.Module):
    """
    2D Positional Encoding for grids.
    
    CRITICAL for spatial reasoning: Without this, the Transformer cannot
    distinguish between positions. It would see a vertical line and a
    horizontal line as identical if they have the same pixels.
    
    Uses learnable embeddings for (x, y) coordinates that are added together.
    This allows the model to learn spatial relationships like "move right"
    or "rotate 90 degrees".
    """
    
    def __init__(self, hidden_dim: int, max_size: int = 32):
        """
        Args:
            hidden_dim: Embedding dimension
            max_size: Maximum grid dimension (ARC grids are up to 30x30)
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_size = max_size
        
        # Separate learnable embeddings for x and y coordinates
        # These are added together to form the full 2D position encoding
        self.x_embed = nn.Embedding(max_size, hidden_dim)
        self.y_embed = nn.Embedding(max_size, hidden_dim)
        
        # Initialize with small values to not dominate initial representations
        nn.init.normal_(self.x_embed.weight, std=0.02)
        nn.init.normal_(self.y_embed.weight, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add 2D positional encodings to grid embeddings.
        
        Args:
            x: [B, H, W, D] grid embeddings
        
        Returns:
            [B, H, W, D] embeddings with positional information
        """
        B, H, W, D = x.shape
        device = x.device
        
        # Create position indices
        y_pos = torch.arange(H, device=device)  # [H]
        x_pos = torch.arange(W, device=device)  # [W]
        
        # Get embeddings
        y_emb = self.y_embed(y_pos)  # [H, D]
        x_emb = self.x_embed(x_pos)  # [W, D]
        
        # Broadcast and add: y_emb[h] + x_emb[w] for each (h, w)
        # [H, 1, D] + [1, W, D] -> [H, W, D]
        pos_emb = y_emb.unsqueeze(1) + x_emb.unsqueeze(0)
        
        # Add to input: [B, H, W, D] + [H, W, D] (broadcasts over batch)
        return x + pos_emb


class AbstractionLayer2D(nn.Module):
    """
    THE KEY SCI INNOVATION adapted for 2D grids.
    
    This layer learns to identify and preserve ONLY structural information,
    while suppressing content-specific features (colors, specific positions).
    
    How it works:
    1. Structural detector: Scores each feature dimension for "structuralness"
       - High score → structural feature (keep)
       - Low score → content feature (suppress)
    2. Soft masking: Apply scores to modulate features
    3. Residual gate: Small bypass to maintain gradient flow
    
    The magic: This is trained end-to-end with SCL, so the network learns
    what constitutes "structure" vs "content" for ARC tasks.
    
    For example, it might learn that:
    - "Objects move right" is structural (high score)
    - "Objects are red" is content (low score)
    """
    
    def __init__(self, d_model: int, hidden_mult: int = 2, dropout: float = 0.1):
        """
        Args:
            d_model: Input/output dimension
            hidden_mult: Hidden layer multiplier
            dropout: Dropout probability
        """
        super().__init__()
        
        # Structural feature detector
        # Outputs a score [0, 1] for each feature dimension
        self.structural_detector = nn.Sequential(
            nn.Linear(d_model, d_model * hidden_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * hidden_mult, d_model),
            nn.Sigmoid()  # [0, 1] structuralness scores
        )
        
        # Initialize detector to output ~0.5 (neutral) initially
        # This prevents over-suppression at the start of training
        nn.init.zeros_(self.structural_detector[-2].bias)
        nn.init.xavier_uniform_(self.structural_detector[-2].weight, gain=0.1)
        
        # Residual gate: Start with higher value to preserve more information initially
        # This helps with gradient flow and prevents representation collapse
        self.residual_gate = nn.Parameter(torch.tensor(0.5))
        # Note: We don't use LayerNorm here as it was causing representation collapse
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply structural abstraction.
        
        Args:
            x: [B, N, D] input embeddings (flattened grid)
        
        Returns:
            abstracted: [B, N, D] with content suppressed
        """
        # Compute structuralness scores for each feature dimension
        # scores[i] ∈ [0, 1]: how "structural" (vs "content") feature i is
        scores = self.structural_detector(x)  # [B, N, D]
        
        # Soft gating with residual bypass:
        # output = x * (scores + residual_gate * (1 - scores))
        #        = x * (scores + residual_gate - residual_gate * scores)
        #        = x * (residual_gate + scores * (1 - residual_gate))
        #
        # When residual_gate=0.5:
        #   - score=1.0 (structural): output = x * 1.0 (full pass)
        #   - score=0.0 (content):    output = x * 0.5 (50% suppression)
        #
        # This ensures gradient flow while still suppressing content features.
        effective_gate = scores + self.residual_gate * (1.0 - scores)
        abstracted = x * effective_gate
        
        # NO NORMALIZATION HERE - it was causing representation collapse!
        # LayerNorm and per-sample std normalization both collapse representations
        # because they normalize away the differences between samples.
        return abstracted
    
    def get_structural_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Get structuralness scores for visualization/analysis."""
        return self.structural_detector(x)


class StructuralEncoder2D(nn.Module):
    """
    Extract transformation structure from (input, output) grid pairs.
    
    Adaptation of SCI's Structural Encoder for 2D grids:
    - AbstractionLayer2D suppresses content-specific features
    - Structure queries attend to transformation patterns
    - Output is invariant to specific objects/colors
    
    The key insight: The DIFFERENCE between input and output encodes the transformation.
    SE should extract WHAT transformation, not WHICH objects.
    
    Architecture:
    1. Flatten and concatenate input/output embeddings
    2. Apply AbstractionLayer2D to suppress content
    3. Encode with transformer layers
    4. Structure queries cross-attend to extract K structural slots
    
    Parameters: ~2M
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_structure_slots: int = 8,  # K in SCI terminology
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_abstraction: bool = True,
        use_difference: bool = False  # DISABLED by default - causes OOM with large grids
    ):
        """
        Args:
            hidden_dim: Hidden dimension
            num_structure_slots: Number of structural pattern slots (K)
            num_layers: Number of transformer encoder layers
            num_heads: Number of attention heads
            dropout: Dropout probability
            use_abstraction: Whether to use AbstractionLayer2D
            use_difference: Whether to add explicit (output - input) difference embedding
                           NOTE: Disabled by default as it triples sequence length
                           and causes OOM on 24GB GPUs with batch_size=192
        """
        super().__init__()
        
        self.num_slots = num_structure_slots
        self.hidden_dim = hidden_dim
        self.use_abstraction = use_abstraction
        self.use_difference = use_difference
        
        # === 2D POSITIONAL ENCODING (Critical for spatial reasoning) ===
        # Without this, Transformer cannot learn spatial relationships
        self.pos_encoder = PositionalEncoding2D(hidden_dim, max_size=32)
        
        # === ABSTRACTION LAYER (Key SCI component) ===
        if use_abstraction:
            self.abstraction_layer = AbstractionLayer2D(hidden_dim, dropout=dropout)
        
        # === STRUCTURE QUERIES ===
        # Learnable queries that extract transformation patterns
        # Similar to DETR object queries, but for transformations
        # Use orthogonal initialization with unit scale to ensure diverse attention patterns
        self.structure_queries = nn.Parameter(
            torch.empty(1, num_structure_slots, hidden_dim)
        )
        # Initialize with orthogonal vectors at full scale for maximum diversity
        nn.init.orthogonal_(self.structure_queries.data.squeeze(0))
        # No scaling down - orthogonal vectors already have unit norm
        
        # === INPUT/OUTPUT ENCODING ===
        # Mark whether embedding comes from input or output
        # If use_difference=True, we have 3 types: input(0), output(1), difference(2)
        self.io_embed = nn.Embedding(3 if use_difference else 2, hidden_dim)
        
        # === DIFFERENCE PROJECTION (NEW) ===
        # Project the difference embedding to same space
        # The difference captures WHERE changes happen, which is the core transformation signal
        if use_difference:
            self.diff_proj = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        
        # === TRANSFORMATION ENCODER ===
        # Process the (input, output) context
        # Use PreLN (norm_first=True) which is more stable and avoids representation collapse
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
            norm_first=True  # PreLN is more stable than PostLN
        )
        self.context_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # === CROSS-ATTENTION ===
        # Structure queries attend to transformation context
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # === QUERY CONDITIONING ===
        # Condition queries on input statistics to ensure diversity across samples
        # This projects global context stats into a query modulation
        self.query_conditioner = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # Mean and std of context
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # === OUTPUT ===
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        # NOTE: LayerNorm removed - it was causing representation collapse
        # by normalizing away differences between samples
    
    def forward(
        self,
        input_emb: torch.Tensor,   # [B, H_in, W_in, D]
        output_emb: torch.Tensor   # [B, H_out, W_out, D]
    ) -> torch.Tensor:
        """
        Extract structural representation from (input, output) pair.
        
        Args:
            input_emb: [B, H_in, W_in, D] encoded input grid
            output_emb: [B, H_out, W_out, D] encoded output grid
        
        Returns:
            structure_slots: [B, K, D] - K structural pattern slots
        """
        B = input_emb.size(0)
        D = self.hidden_dim
        H_in, W_in = input_emb.shape[1], input_emb.shape[2]
        H_out, W_out = output_emb.shape[1], output_emb.shape[2]
        
        # Add 2D positional encodings BEFORE flattening
        # This gives each cell a unique spatial address (x, y)
        input_pos = self.pos_encoder(input_emb)   # [B, H_in, W_in, D]
        output_pos = self.pos_encoder(output_emb) # [B, H_out, W_out, D]
        
        # Flatten grids to sequences
        input_flat = input_pos.view(B, -1, D)   # [B, H_in*W_in, D]
        output_flat = output_pos.view(B, -1, D) # [B, H_out*W_out, D]
        
        # Add input/output indicators
        input_flat = input_flat + self.io_embed.weight[0]
        output_flat = output_flat + self.io_embed.weight[1]
        
        # === DIFFERENCE EMBEDDING (NEW) ===
        # Compute explicit difference where grids overlap
        # This directly encodes WHERE changes happen - the core transformation signal
        if self.use_difference:
            # Use the smaller overlapping region
            H_min, W_min = min(H_in, H_out), min(W_in, W_out)
            
            # Compute difference in embedding space (output - input)
            # This highlights cells that changed
            diff_emb = output_emb[:, :H_min, :W_min, :] - input_emb[:, :H_min, :W_min, :]
            
            # Add positional encoding to difference
            diff_pos = self.pos_encoder(diff_emb)  # [B, H_min, W_min, D]
            
            # Project through MLP to learn what differences matter
            diff_flat = diff_pos.view(B, -1, D)  # [B, H_min*W_min, D]
            diff_flat = self.diff_proj(diff_flat)  # [B, H_min*W_min, D]
            
            # Add difference indicator
            diff_flat = diff_flat + self.io_embed.weight[2]
        
        # Apply AbstractionLayer to suppress content
        if self.use_abstraction:
            input_abs = self.abstraction_layer(input_flat)
            output_abs = self.abstraction_layer(output_flat)
            if self.use_difference:
                diff_abs = self.abstraction_layer(diff_flat)
        else:
            input_abs = input_flat
            output_abs = output_flat
            if self.use_difference:
                diff_abs = diff_flat
        
        # Concatenate: [input | output | difference]
        if self.use_difference:
            context = torch.cat([input_abs, output_abs, diff_abs], dim=1)
        else:
            context = torch.cat([input_abs, output_abs], dim=1)  # [B, N_in + N_out, D]
        
        # Encode transformation context
        context_encoded = self.context_encoder(context)
        
        # Condition structure queries on context statistics for sample-specific diversity
        # This ensures different inputs produce different query modulations
        context_mean = context_encoded.mean(dim=1)  # [B, D]
        context_std = context_encoded.std(dim=1)    # [B, D]
        context_stats = torch.cat([context_mean, context_std], dim=-1)  # [B, 2D]
        query_modulation = self.query_conditioner(context_stats)  # [B, D]
        
        # Structure queries cross-attend to context (with sample-specific modulation)
        queries = self.structure_queries.expand(B, -1, -1)  # [B, K, D]
        queries = queries + query_modulation.unsqueeze(1)   # Add sample-specific offset
        
        structure_slots, attn_weights = self.cross_attention(
            query=queries,
            key=context_encoded,
            value=context_encoded
        )
        
        # Project output (NO LayerNorm - it collapses sample differences!)
        structure_slots = self.output_proj(structure_slots)
        # NOTE: Removed output_norm LayerNorm as it was causing representation collapse
        # by normalizing away the differences between samples
        
        return structure_slots
    
    def forward_with_attention(
        self,
        input_emb: torch.Tensor,
        output_emb: torch.Tensor
    ):
        """Forward pass that also returns attention weights for visualization."""
        B = input_emb.size(0)
        D = self.hidden_dim
        H_in, W_in = input_emb.shape[1], input_emb.shape[2]
        H_out, W_out = output_emb.shape[1], output_emb.shape[2]
        
        # Add 2D positional encodings BEFORE flattening
        input_pos = self.pos_encoder(input_emb)
        output_pos = self.pos_encoder(output_emb)
        
        input_flat = input_pos.view(B, -1, D)
        output_flat = output_pos.view(B, -1, D)
        
        input_flat = input_flat + self.io_embed.weight[0]
        output_flat = output_flat + self.io_embed.weight[1]
        
        # Difference embedding (if enabled)
        if self.use_difference:
            H_min, W_min = min(H_in, H_out), min(W_in, W_out)
            diff_emb = output_emb[:, :H_min, :W_min, :] - input_emb[:, :H_min, :W_min, :]
            diff_pos = self.pos_encoder(diff_emb)
            diff_flat = diff_pos.view(B, -1, D)
            diff_flat = self.diff_proj(diff_flat)
            diff_flat = diff_flat + self.io_embed.weight[2]
        
        if self.use_abstraction:
            input_abs = self.abstraction_layer(input_flat)
            output_abs = self.abstraction_layer(output_flat)
            if self.use_difference:
                diff_abs = self.abstraction_layer(diff_flat)
            structural_scores = self.abstraction_layer.get_structural_scores(
                torch.cat([input_flat, output_flat], dim=1)
            )
        else:
            input_abs = input_flat
            output_abs = output_flat
            if self.use_difference:
                diff_abs = diff_flat
            structural_scores = None
        
        # Concatenate context
        if self.use_difference:
            context = torch.cat([input_abs, output_abs, diff_abs], dim=1)
        else:
            context = torch.cat([input_abs, output_abs], dim=1)
        context_encoded = self.context_encoder(context)
        
        # Condition queries on context statistics
        context_mean = context_encoded.mean(dim=1)
        context_std = context_encoded.std(dim=1)
        context_stats = torch.cat([context_mean, context_std], dim=-1)
        query_modulation = self.query_conditioner(context_stats)
        
        queries = self.structure_queries.expand(B, -1, -1)
        queries = queries + query_modulation.unsqueeze(1)
        
        structure_slots, attn_weights = self.cross_attention(
            query=queries,
            key=context_encoded,
            value=context_encoded
        )
        
        # Project output (NO LayerNorm - it collapses sample differences!)
        structure_slots = self.output_proj(structure_slots)
        
        return structure_slots, attn_weights, structural_scores


class MultiScaleStructuralEncoder(nn.Module):
    """
    Multi-scale structural encoder that processes grids at different resolutions.
    
    This can help capture both local patterns (cell-level) and global patterns
    (grid-level transformations).
    
    Not used by default, but provided as an extension.
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_structure_slots: int = 8,
        num_scales: int = 2
    ):
        super().__init__()
        
        self.encoders = nn.ModuleList([
            StructuralEncoder2D(hidden_dim, num_structure_slots // num_scales)
            for _ in range(num_scales)
        ])
        
        # Pool grids at different scales
        self.pools = nn.ModuleList([
            nn.AvgPool2d(2**i, 2**i) if i > 0 else nn.Identity()
            for i in range(num_scales)
        ])
        
        self.combine = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, input_emb: torch.Tensor, output_emb: torch.Tensor) -> torch.Tensor:
        all_slots = []
        
        for encoder, pool in zip(self.encoders, self.pools):
            # Pool to different scales
            inp_pooled = pool(input_emb.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            out_pooled = pool(output_emb.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            
            slots = encoder(inp_pooled, out_pooled)
            all_slots.append(slots)
        
        # Concatenate slots from all scales
        combined = torch.cat(all_slots, dim=1)
        
        return self.combine(combined)
