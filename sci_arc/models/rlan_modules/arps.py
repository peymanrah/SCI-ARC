"""
Anchor-Relative Program Search (ARPS) for RLAN - Jan 2026 Ablation Study

ARPS is a novel approach that combines RLAN's anchor-relative reasoning with
program synthesis. The key insight: DSL programs expressed in anchor-relative
coordinates generalize better than absolute coordinates.

Why This Works:
1. RLAN's DSC finds anchors (e.g., "the red marker")
2. MSRE computes coordinates relative to anchors
3. ARPS expresses programs in this relative frame:
   - "translate(0, 0)" = move to anchor
   - "reflect_x" = reflect across anchor's x-axis
   
This makes programs invariant to anchor position - they generalize naturally.

Architecture:
1. ProgramProposalHead: Neural network that proposes program tokens
2. DSLPrimitives: Symbolic operations in anchor-relative coords
3. ProgramExecutor: Deterministic execution of programs
4. ProgramVerifier: Checks programs against training demos

Integration:
- Hooks into RLAN after MSRE (has anchor info)
- Adds program proposals as auxiliary output head
- Uses imitation learning from search to train proposal head

Usage:
    arps = ARPS(arps_config, hidden_dim=256)
    
    # During training:
    proposals = arps.propose_programs(clue_features, context)
    executed = arps.execute_programs(proposals, input_grid, centroids)
    verified = arps.verify_programs(executed, train_outputs)
    best_program = arps.select_best(verified)
    imitation_loss = arps.compute_imitation_loss(proposals, best_program)
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any, Callable
from enum import Enum
from collections import deque
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrimitiveType(Enum):
    """DSL primitive operation types."""
    SELECT_COLOR = "select_color"
    SELECT_CONNECTED = "select_connected"
    TRANSLATE = "translate"
    REFLECT_X = "reflect_x"
    REFLECT_Y = "reflect_y"
    ROTATE_90 = "rotate_90"
    ROTATE_180 = "rotate_180"
    PAINT = "paint"
    COPY_PASTE = "copy_paste"
    TILE = "tile"
    CROP = "crop"
    FILL = "fill"
    END = "end"  # End of program token


@dataclass
class ARPSConfig:
    """Configuration for Anchor-Relative Program Search."""
    enabled: bool = True
    use_as_auxiliary: bool = True       # Add to neural head, don't replace
    max_program_length: int = 12        # INCREASED from 8 for complex multi-step tasks
    beam_size: int = 64                 # INCREASED from 32 for better exploration  
    top_k_proposals: int = 8            # INCREASED from 4 for more candidate verification
    
    # Primitives to use
    primitives: List[str] = field(default_factory=lambda: [
        "select_color", "select_connected", "translate",
        "reflect_x", "reflect_y", "rotate_90", "rotate_180",
        "paint", "copy_paste", "tile", "crop", "fill"
    ])
    
    # Verification
    require_demo_exact_match: bool = True
    use_mdl_ranking: bool = True        # Prefer shorter programs
    
    # Training
    search_during_training: bool = True
    imitation_weight: float = 0.1
    
    # Architecture
    hidden_dim: int = 256
    num_heads: int = 4
    dropout: float = 0.1


class DSLPrimitives:
    """
    Anchor-Relative DSL Primitives.
    
    All geometric operations are expressed relative to the anchor point,
    making them invariant to anchor position in the grid.
    """
    
    @staticmethod
    def select_color(
        grid: torch.Tensor,
        color: int,
        anchor: torch.Tensor,
    ) -> torch.Tensor:
        """
        Select pixels of a specific color.
        
        Args:
            grid: (H, W) input grid
            color: Color index (0-9)
            anchor: (2,) anchor position (not used but kept for API consistency)
            
        Returns:
            mask: (H, W) binary mask of selected pixels
        """
        return (grid == color).float()
    
    @staticmethod
    def select_connected(
        grid: torch.Tensor,
        seed_offset: Tuple[int, int],
        anchor: torch.Tensor,
    ) -> torch.Tensor:
        """
        Select connected component at offset from anchor.
        
        OPTIMIZED: Uses deque for O(1) BFS + numpy conversion to avoid GPU sync.
        
        Args:
            grid: (H, W) input grid
            seed_offset: (row_offset, col_offset) relative to anchor
            anchor: (2,) anchor position
            
        Returns:
            mask: (H, W) binary mask of connected component
        """
        H, W = grid.shape
        device = grid.device
        
        # Move to CPU for fast Python operations (avoids GPU sync per pixel)
        grid_cpu = grid.detach().cpu()
        anchor_cpu = anchor.detach().cpu()
        
        # Compute seed position in absolute coords
        seed_row = int(anchor_cpu[0].item()) + seed_offset[0]
        seed_col = int(anchor_cpu[1].item()) + seed_offset[1]
        
        # Clamp to grid bounds
        seed_row = max(0, min(H - 1, seed_row))
        seed_col = max(0, min(W - 1, seed_col))
        
        # Get seed color (CPU access - fast)
        seed_color = grid_cpu[seed_row, seed_col].item()
        
        # BFS with deque (O(1) popleft vs O(n) pop(0))
        mask_cpu = torch.zeros(H, W, dtype=torch.float32)
        visited = [[False] * W for _ in range(H)]  # Python list faster for small grids
        
        queue = deque([(seed_row, seed_col)])
        visited[seed_row][seed_col] = True
        
        while queue:
            r, c = queue.popleft()  # O(1) instead of O(n)
            if grid_cpu[r, c].item() == seed_color:
                mask_cpu[r, c] = 1.0
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W and not visited[nr][nc]:
                        visited[nr][nc] = True
                        queue.append((nr, nc))
        
        return mask_cpu.to(device)
    
    @staticmethod
    def translate(
        selection: torch.Tensor,
        offset: Tuple[int, int],
        anchor: torch.Tensor,
    ) -> torch.Tensor:
        """
        Translate selection by offset (relative to anchor frame).
        
        OPTIMIZED: Vectorized using roll + masking instead of Python loops.
        
        The offset is in anchor-relative coordinates, so (0, 0) means
        "move to anchor position".
        
        Args:
            selection: (H, W) binary selection mask
            offset: (row_offset, col_offset) in anchor-relative coords
            anchor: (2,) anchor position
            
        Returns:
            translated: (H, W) translated mask
        """
        H, W = selection.shape
        device = selection.device
        
        # Move to CPU for computation
        selection_cpu = selection.detach().cpu()
        anchor_cpu = anchor.detach().cpu()
        
        # Find selection centroid using vectorized ops
        # Use > 0 threshold to match original semantics for float masks
        indices = torch.nonzero(selection_cpu > 0)
        if len(indices) == 0:
            return selection
        
        centroid = indices.float().mean(dim=0)
        
        # Compute absolute offset: selection_centroid → (anchor + offset)
        target_row = anchor_cpu[0].item() + offset[0]
        target_col = anchor_cpu[1].item() + offset[1]
        
        abs_offset_row = int(target_row - centroid[0].item())
        abs_offset_col = int(target_col - centroid[1].item())
        
        # VECTORIZED TRANSLATION using index arithmetic
        # Get all nonzero positions
        src_rows = indices[:, 0].long()
        src_cols = indices[:, 1].long()
        
        # Compute destination positions
        dst_rows = (src_rows + abs_offset_row).long()
        dst_cols = (src_cols + abs_offset_col).long()
        
        # Filter to valid bounds
        valid_mask = (dst_rows >= 0) & (dst_rows < H) & (dst_cols >= 0) & (dst_cols < W)
        valid_dst_rows = dst_rows[valid_mask].clamp(0, H - 1)
        valid_dst_cols = dst_cols[valid_mask].clamp(0, W - 1)
        valid_src_rows = src_rows[valid_mask].clamp(0, H - 1)
        valid_src_cols = src_cols[valid_mask].clamp(0, W - 1)
        
        # Create output and scatter
        translated_cpu = torch.zeros(H, W, dtype=selection_cpu.dtype)
        if len(valid_dst_rows) > 0:
            translated_cpu[valid_dst_rows, valid_dst_cols] = selection_cpu[valid_src_rows, valid_src_cols]
        
        return translated_cpu.to(device)
    
    @staticmethod
    def reflect_x(
        selection: torch.Tensor,
        anchor: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reflect selection across horizontal axis through anchor.
        
        OPTIMIZED: Vectorized index computation.
        
        Args:
            selection: (H, W) binary selection mask
            anchor: (2,) anchor position
            
        Returns:
            reflected: (H, W) reflected mask
        """
        H, W = selection.shape
        device = selection.device
        
        # Move to CPU
        selection_cpu = selection.detach().cpu()
        anchor_cpu = anchor.detach().cpu()
        anchor_row = int(anchor_cpu[0].item())
        
        # Get nonzero indices (use > 0 threshold for float mask correctness)
        indices = torch.nonzero(selection_cpu > 0)
        if len(indices) == 0:
            return selection
        
        src_rows = indices[:, 0].long()
        src_cols = indices[:, 1].long()
        
        # Compute reflected positions
        dst_rows = (2 * anchor_row - src_rows).long()
        dst_cols = src_cols  # Column unchanged
        
        # Filter valid
        valid_mask = (dst_rows >= 0) & (dst_rows < H)
        valid_dst_rows = dst_rows[valid_mask].clamp(0, H - 1)
        valid_dst_cols = dst_cols[valid_mask].clamp(0, W - 1)
        valid_src_rows = src_rows[valid_mask].clamp(0, H - 1)
        valid_src_cols = src_cols[valid_mask].clamp(0, W - 1)
        
        result_cpu = torch.zeros(H, W, dtype=selection_cpu.dtype)
        if len(valid_dst_rows) > 0:
            result_cpu[valid_dst_rows, valid_dst_cols] = selection_cpu[valid_src_rows, valid_src_cols]
        
        return result_cpu.to(device)
    
    @staticmethod
    def reflect_y(
        selection: torch.Tensor,
        anchor: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reflect selection across vertical axis through anchor.
        
        OPTIMIZED: Vectorized index computation.
        """
        H, W = selection.shape
        device = selection.device
        
        # Move to CPU
        selection_cpu = selection.detach().cpu()
        anchor_cpu = anchor.detach().cpu()
        anchor_col = int(anchor_cpu[1].item())
        
        # Get nonzero indices (use > 0 threshold for float mask correctness)
        indices = torch.nonzero(selection_cpu > 0)
        if len(indices) == 0:
            return selection
        
        src_rows = indices[:, 0].long()
        src_cols = indices[:, 1].long()
        
        # Compute reflected positions
        dst_rows = src_rows  # Row unchanged
        dst_cols = (2 * anchor_col - src_cols).long()
        
        # Filter valid
        valid_mask = (dst_cols >= 0) & (dst_cols < W)
        valid_dst_rows = dst_rows[valid_mask].clamp(0, H - 1)
        valid_dst_cols = dst_cols[valid_mask].clamp(0, W - 1)
        valid_src_rows = src_rows[valid_mask].clamp(0, H - 1)
        valid_src_cols = src_cols[valid_mask].clamp(0, W - 1)
        
        result_cpu = torch.zeros(H, W, dtype=selection_cpu.dtype)
        if len(valid_dst_rows) > 0:
            result_cpu[valid_dst_rows, valid_dst_cols] = selection_cpu[valid_src_rows, valid_src_cols]
        
        return result_cpu.to(device)
    
    @staticmethod
    def rotate_90(
        selection: torch.Tensor,
        anchor: torch.Tensor,
    ) -> torch.Tensor:
        """
        Rotate selection 90° clockwise around anchor.
        
        OPTIMIZED: Vectorized rotation using index arithmetic.
        """
        H, W = selection.shape
        device = selection.device
        
        # Move to CPU
        selection_cpu = selection.detach().cpu()
        anchor_cpu = anchor.detach().cpu()
        anchor_row = int(anchor_cpu[0].item())
        anchor_col = int(anchor_cpu[1].item())
        
        # Get nonzero indices (use > 0 threshold for float mask correctness)
        indices = torch.nonzero(selection_cpu > 0)
        if len(indices) == 0:
            return selection
        
        src_rows = indices[:, 0].long()
        src_cols = indices[:, 1].long()
        
        # Translate to anchor origin, rotate, translate back
        # Rotate 90° CW: (dr, dc) → (dc, -dr)
        dr = src_rows - anchor_row
        dc = src_cols - anchor_col
        new_dr = dc
        new_dc = -dr
        dst_rows = (anchor_row + new_dr).long()
        dst_cols = (anchor_col + new_dc).long()
        
        # Filter valid
        valid_mask = (dst_rows >= 0) & (dst_rows < H) & (dst_cols >= 0) & (dst_cols < W)
        valid_dst_rows = dst_rows[valid_mask].clamp(0, H - 1)
        valid_dst_cols = dst_cols[valid_mask].clamp(0, W - 1)
        valid_src_rows = src_rows[valid_mask].clamp(0, H - 1)
        valid_src_cols = src_cols[valid_mask].clamp(0, W - 1)
        
        result_cpu = torch.zeros(H, W, dtype=selection_cpu.dtype)
        if len(valid_dst_rows) > 0:
            result_cpu[valid_dst_rows, valid_dst_cols] = selection_cpu[valid_src_rows, valid_src_cols]
        
        return result_cpu.to(device)
    
    @staticmethod
    def rotate_180(
        selection: torch.Tensor,
        anchor: torch.Tensor,
    ) -> torch.Tensor:
        """Rotate selection 180° around anchor."""
        result = DSLPrimitives.rotate_90(selection, anchor)
        result = DSLPrimitives.rotate_90(result, anchor)
        return result
    
    @staticmethod
    def paint(
        grid: torch.Tensor,
        selection: torch.Tensor,
        color: int,
    ) -> torch.Tensor:
        """
        Paint selected pixels with color.
        
        Args:
            grid: (H, W) input grid
            selection: (H, W) binary mask
            color: Color to paint
            
        Returns:
            result: (H, W) painted grid
        """
        result = grid.clone()
        result[selection > 0.5] = color
        return result
    
    @staticmethod
    def fill(
        grid: torch.Tensor,
        selection: torch.Tensor,
        color: int,
    ) -> torch.Tensor:
        """Fill enclosed regions within selection with color."""
        # For simplicity, same as paint
        return DSLPrimitives.paint(grid, selection, color)
    
    @staticmethod
    def copy_paste(
        grid: torch.Tensor,
        selection: torch.Tensor,
        offset: Tuple[int, int],
        anchor: torch.Tensor,
    ) -> torch.Tensor:
        """
        Copy selected region and paste at offset from anchor.
        
        OPTIMIZED: Vectorized index operations on CPU.
        """
        translated_mask = DSLPrimitives.translate(selection, offset, anchor)
        
        H, W = grid.shape
        device = grid.device
        
        # Move to CPU for fast indexing
        grid_cpu = grid.detach().cpu()
        selection_cpu = selection.detach().cpu()
        translated_cpu = translated_mask.detach().cpu()
        result_cpu = grid_cpu.clone()
        
        # Find where to copy from (original selection) and to (translated)
        src_indices = torch.nonzero(selection_cpu > 0.5)
        dst_indices = torch.nonzero(translated_cpu > 0.5)
        
        # Vectorized copy if indices match
        if len(src_indices) == len(dst_indices) and len(src_indices) > 0:
            src_rows, src_cols = src_indices[:, 0].long(), src_indices[:, 1].long()
            dst_rows, dst_cols = dst_indices[:, 0].long(), dst_indices[:, 1].long()
            
            # Filter valid destinations
            valid_mask = (dst_rows >= 0) & (dst_rows < H) & (dst_cols >= 0) & (dst_cols < W)
            valid_src_rows = src_rows[valid_mask].clamp(0, H - 1)
            valid_src_cols = src_cols[valid_mask].clamp(0, W - 1)
            valid_dst_rows = dst_rows[valid_mask].clamp(0, H - 1)
            valid_dst_cols = dst_cols[valid_mask].clamp(0, W - 1)
            
            # Vectorized assignment
            if len(valid_dst_rows) > 0:
                result_cpu[valid_dst_rows, valid_dst_cols] = grid_cpu[valid_src_rows, valid_src_cols]
        
        return result_cpu.to(device)


class ProgramProposalHead(nn.Module):
    """
    Neural network that proposes DSL programs.
    
    Uses a transformer decoder to autoregressively generate program tokens,
    conditioned on the task context and anchor-relative features.
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_primitives: int = 13,  # 12 primitives + END
        max_length: int = 8,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_primitives = num_primitives
        self.max_length = max_length
        
        # Token embedding
        self.token_embed = nn.Embedding(num_primitives + 1, hidden_dim)  # +1 for start token
        
        # Positional embedding for sequence
        # Note: +2 for start token + safety margin (sequence grows to max_length+1 during autoregressive decode)
        self.pos_embed = nn.Embedding(max_length + 2, hidden_dim)
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, num_primitives)
        
        # Argument prediction heads (for primitives that need arguments)
        self.color_head = nn.Linear(hidden_dim, 10)  # Color 0-9
        self.offset_head = nn.Linear(hidden_dim, 2)  # (row, col) offset
    
    def forward(
        self,
        context: torch.Tensor,
        partial_program: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Propose next token(s) given context and partial program.
        
        Args:
            context: (B, D) task context embedding
            partial_program: (B, L) partial program tokens, or None for start
            
        Returns:
            token_logits: (B, num_primitives) next token logits
            color_logits: (B, 10) color argument logits
            offset_pred: (B, 2) offset argument prediction
        """
        B = context.shape[0]
        device = context.device
        
        if partial_program is None:
            # Start with start token
            partial_program = torch.full((B, 1), self.num_primitives, device=device)
        
        L = partial_program.shape[1]
        
        # SAFETY: Clamp sequence length to max_length+1 to prevent position index overflow
        # This can happen if programs exceed expected length during training
        max_pos = self.max_length + 1
        if L > max_pos:
            partial_program = partial_program[:, -max_pos:]  # Keep last max_pos tokens
            L = max_pos
        
        # SAFETY: Clamp token indices to valid range to prevent embedding index overflow
        partial_program = partial_program.clamp(0, self.num_primitives)
        
        # Embed tokens
        token_emb = self.token_embed(partial_program)  # (B, L, D)
        
        # Add positional embeddings
        # SAFETY: Clamp positions to valid embedding range
        positions = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        positions = positions.clamp(0, self.pos_embed.num_embeddings - 1)
        pos_emb = self.pos_embed(positions)
        token_emb = token_emb + pos_emb
        
        # Context as memory for cross-attention
        memory = context.unsqueeze(1)  # (B, 1, D)
        
        # Causal mask for autoregressive decoding
        causal_mask = torch.triu(
            torch.ones(L, L, device=device) * float('-inf'),
            diagonal=1
        )
        
        # Decode
        hidden = self.decoder(token_emb, memory, tgt_mask=causal_mask)
        
        # Get last position output
        last_hidden = hidden[:, -1, :]  # (B, D)
        
        # Predict outputs
        token_logits = self.output_proj(last_hidden)
        color_logits = self.color_head(last_hidden)
        offset_pred = self.offset_head(last_hidden)
        
        return token_logits, color_logits, offset_pred
    
    def sample_program(
        self,
        context: torch.Tensor,
        temperature: float = 1.0,
        top_k: int = 5,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a complete program autoregressively.
        
        Args:
            context: (B, D) task context
            temperature: Sampling temperature
            top_k: Top-K sampling
            
        Returns:
            program: (B, L) program tokens
            colors: (B, L) color arguments
            offsets: (B, L, 2) offset arguments
        """
        B = context.shape[0]
        device = context.device
        
        program = []
        colors = []
        offsets = []
        
        partial = None
        end_token = PrimitiveType.END.value
        
        for step in range(self.max_length):
            token_logits, color_logits, offset_pred = self.forward(context, partial)
            
            # Sample token with temperature and top-k
            if temperature > 0:
                logits = token_logits / temperature
                # SAFETY: Clamp logits to prevent NaN/Inf in softmax
                logits = logits.clamp(-100.0, 100.0)
                # Top-K filtering
                topk_vals, topk_idx = torch.topk(logits, min(top_k, logits.shape[-1]))
                probs = F.softmax(topk_vals, dim=-1)
                # SAFETY: Handle NaN/Inf in probs that can crash multinomial
                if torch.isnan(probs).any() or torch.isinf(probs).any():
                    probs = torch.ones_like(probs) / probs.shape[-1]  # Uniform fallback
                # SAFETY: Ensure probs sum > 0 (multinomial requires this)
                probs = probs + 1e-8
                probs = probs / probs.sum(dim=-1, keepdim=True)
                sampled_idx = torch.multinomial(probs, 1)
                token = topk_idx.gather(-1, sampled_idx).squeeze(-1)
            else:
                token = token_logits.argmax(dim=-1)
            
            # SAFETY: Clamp token to valid primitive range to prevent embedding overflow
            token = token.clamp(0, self.num_primitives - 1)
            
            # Sample arguments
            color = color_logits.argmax(dim=-1)
            # SAFETY: Clamp color to valid range [0, 9] to prevent index overflow
            color = color.clamp(0, 9)
            # SAFETY: Clamp offsets to reasonable grid range to prevent overflow
            offset = offset_pred.clamp(-30.0, 30.0)

            program.append(token)
            colors.append(color)
            offsets.append(offset)
            
            # Update partial program
            if partial is None:
                partial = torch.full((B, 1), self.num_primitives, device=device)
            partial = torch.cat([partial, token.unsqueeze(-1)], dim=-1)
            
            # SAFETY: Stop if partial exceeds max_length to prevent index overflow
            if partial.shape[1] >= self.max_length + 1:
                break
            
            # Check for END tokens (in actual impl, would track per-batch)
        
        program = torch.stack(program, dim=1)
        colors = torch.stack(colors, dim=1)
        offsets = torch.stack(offsets, dim=1)
        
        return program, colors, offsets


class ProgramExecutor:
    """
    Executes DSL programs to produce output grids.
    
    Programs are lists of (primitive, arguments) tuples, executed sequentially.
    All operations are in anchor-relative coordinates.
    """
    
    PRIMITIVE_MAP = {
        "select_color": DSLPrimitives.select_color,
        "select_connected": DSLPrimitives.select_connected,
        "translate": DSLPrimitives.translate,
        "reflect_x": DSLPrimitives.reflect_x,
        "reflect_y": DSLPrimitives.reflect_y,
        "rotate_90": DSLPrimitives.rotate_90,
        "rotate_180": DSLPrimitives.rotate_180,
        "paint": DSLPrimitives.paint,
        "copy_paste": DSLPrimitives.copy_paste,
        "fill": DSLPrimitives.fill,
    }
    
    @staticmethod
    def execute(
        program: List[Tuple[str, Dict[str, Any]]],
        input_grid: torch.Tensor,
        anchor: torch.Tensor,
    ) -> torch.Tensor:
        """
        Execute a program on an input grid.
        
        Args:
            program: List of (primitive_name, args_dict) tuples
            input_grid: (H, W) input grid
            anchor: (2,) anchor position
            
        Returns:
            output_grid: (H, W) result grid
        """
        current_grid = input_grid.clone()
        current_selection = torch.ones_like(input_grid).float()
        
        for primitive_name, args in program:
            if primitive_name == "end":
                break
            
            primitive_fn = ProgramExecutor.PRIMITIVE_MAP.get(primitive_name)
            if primitive_fn is None:
                continue
            
            try:
                if primitive_name == "select_color":
                    current_selection = primitive_fn(current_grid, args.get("color", 0), anchor)
                elif primitive_name == "select_connected":
                    current_selection = primitive_fn(current_grid, args.get("offset", (0, 0)), anchor)
                elif primitive_name == "translate":
                    current_selection = primitive_fn(current_selection, args.get("offset", (0, 0)), anchor)
                elif primitive_name in ["reflect_x", "reflect_y", "rotate_90", "rotate_180"]:
                    current_selection = primitive_fn(current_selection, anchor)
                elif primitive_name in ["paint", "fill"]:
                    current_grid = primitive_fn(current_grid, current_selection, args.get("color", 0))
                elif primitive_name == "copy_paste":
                    current_grid = primitive_fn(current_grid, current_selection, args.get("offset", (0, 0)), anchor)
            except Exception:
                # Skip failed primitives
                pass
        
        return current_grid


class ProgramVerifier:
    """
    Verifies programs against training demonstrations.
    
    A program is valid if it produces exact match outputs for ALL training pairs.
    
    OPTIMIZED: Fail-fast on first mismatch + CPU execution.
    """
    
    @staticmethod
    def verify(
        program: List[Tuple[str, Dict[str, Any]]],
        train_inputs: torch.Tensor,
        train_outputs: torch.Tensor,
        anchors: torch.Tensor,
        pair_mask: Optional[torch.Tensor] = None,
        require_all_match: bool = True,
    ) -> Tuple[bool, float]:
        """
        Verify program against training demos.
        
        OPTIMIZED: Fail-fast when require_all_match=True (default).
        Exits immediately on first mismatch to avoid wasted computation.
        
        Args:
            program: DSL program
            train_inputs: (N, H, W) training input grids
            train_outputs: (N, H, W) training output grids
            anchors: (N, 2) anchor positions for each pair
            pair_mask: (N,) valid pair mask
            require_all_match: If True, return early on first failure
            
        Returns:
            is_valid: True if program solves all demos
            accuracy: Fraction of demos solved
        """
        N = train_inputs.shape[0]
        
        # Move to CPU once for all verification (avoid GPU sync per pair)
        train_inputs_cpu = train_inputs.detach().cpu()
        train_outputs_cpu = train_outputs.detach().cpu()
        anchors_cpu = anchors.detach().cpu()
        pair_mask_cpu = pair_mask.detach().cpu() if pair_mask is not None else None
        
        num_valid = 0
        num_correct = 0
        
        for i in range(N):
            if pair_mask_cpu is not None and not pair_mask_cpu[i]:
                continue
            
            num_valid += 1
            
            # Execute on CPU tensors (fast, no GPU sync)
            predicted = ProgramExecutor.execute(
                program,
                train_inputs_cpu[i],
                anchors_cpu[i],
            )
            
            # Compare on CPU
            if torch.equal(predicted.cpu(), train_outputs_cpu[i]):
                num_correct += 1
            elif require_all_match:
                # FAIL-FAST: No need to check remaining demos
                return False, num_correct / num_valid
        
        is_valid = (num_correct == num_valid) and (num_valid > 0)
        accuracy = num_correct / max(num_valid, 1)
        
        return is_valid, accuracy


class ARPS(nn.Module):
    """
    Anchor-Relative Program Search module for RLAN.
    
    This module provides:
    1. Neural program proposals conditioned on task context
    2. Symbolic program execution and verification
    3. Imitation learning from search to improve proposals
    """
    
    def __init__(self, config: ARPSConfig):
        super().__init__()
        self.config = config
        
        # Build primitive vocabulary
        self.primitives = config.primitives + ["end"]
        self.primitive_to_idx = {p: i for i, p in enumerate(self.primitives)}
        self.idx_to_primitive = {i: p for p, i in self.primitive_to_idx.items()}
        
        # Program proposal head
        self.proposal_head = ProgramProposalHead(
            hidden_dim=config.hidden_dim,
            num_primitives=len(self.primitives),
            max_length=config.max_program_length,
            num_heads=config.num_heads,
            dropout=config.dropout,
        )
        
        # Context pooling for conditioning
        self.context_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
        )
    
    def propose_programs(
        self,
        clue_features: torch.Tensor,
        temperature: float = 1.0,
        num_samples: int = 4,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Propose programs from neural head.
        
        Args:
            clue_features: (B, K, D, H, W) clue-relative features from MSRE
            temperature: Sampling temperature
            num_samples: Number of programs to sample per batch item
            
        Returns:
            proposals: List of proposal dicts with tokens, colors, offsets
        """
        # SAFETY: Validate input shapes to prevent downstream CUDA errors
        if clue_features.dim() != 5:
            raise ValueError(f"[ARPS] clue_features must be 5D (B,K,D,H,W), got {clue_features.dim()}D")
        
        # SAFETY: Check for NaN/Inf in input that would propagate to embeddings
        if torch.isnan(clue_features).any() or torch.isinf(clue_features).any():
            print("[ARPS WARNING] NaN/Inf detected in clue_features, replacing with zeros")
            clue_features = torch.nan_to_num(clue_features, nan=0.0, posinf=0.0, neginf=0.0)
        
        B, K, D, H, W = clue_features.shape
        
        # Pool features to context
        pooled = clue_features.mean(dim=1)  # (B, D, H, W)
        context = self.context_pool(pooled)  # (B, D)
        
        proposals = []
        for _ in range(num_samples):
            program, colors, offsets = self.proposal_head.sample_program(
                context, temperature=temperature
            )
            proposals.append({
                "program": program,
                "colors": colors,
                "offsets": offsets,
            })
        
        return proposals
    
    def decode_program(
        self,
        tokens: torch.Tensor,
        colors: torch.Tensor,
        offsets: torch.Tensor,
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Decode token tensors to executable program.
        
        OPTIMIZED: Move tensors to CPU once to avoid GPU sync per .item() call.
        
        Args:
            tokens: (L,) program token indices
            colors: (L,) color arguments
            offsets: (L, 2) offset arguments
            
        Returns:
            program: List of (primitive_name, args_dict) tuples
        """
        program = []
        
        # Move to CPU once to avoid GPU sync per .item() call
        tokens_cpu = tokens.detach().cpu()
        colors_cpu = colors.detach().cpu()
        offsets_cpu = offsets.detach().cpu()
        
        # SAFETY: Limit loop iterations to prevent runaway
        max_iterations = min(len(tokens_cpu), self.config.max_program_length)
        
        for i in range(max_iterations):
            token_idx = int(tokens_cpu[i].item())
            
            # SAFETY: Skip invalid token indices
            if token_idx < 0 or token_idx >= len(self.idx_to_primitive):
                continue
            
            primitive_name = self.idx_to_primitive[token_idx]
            if primitive_name == "end":
                break
            
            # SAFETY: Clamp color to valid range [0, 9]
            color_val = int(colors_cpu[i].item())
            color_val = max(0, min(9, color_val))
            
            # SAFETY: Clamp offsets to reasonable grid range [-30, 30]
            offset_row = int(offsets_cpu[i, 0].item())
            offset_col = int(offsets_cpu[i, 1].item())
            offset_row = max(-30, min(30, offset_row))
            offset_col = max(-30, min(30, offset_col))
            
            args = {
                "color": color_val,
                "offset": (offset_row, offset_col),
            }
            program.append((primitive_name, args))
        
        return program
    
    def execute_and_verify(
        self,
        proposals: List[Dict[str, torch.Tensor]],
        input_grid: torch.Tensor,
        train_inputs: torch.Tensor,
        train_outputs: torch.Tensor,
        centroids: torch.Tensor,
        pair_mask: Optional[torch.Tensor] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute proposals and verify against training demos.
        
        OPTIMIZED: Runs under inference_mode to avoid autograd overhead.
        All symbolic execution happens on CPU to avoid GPU sync.
        
        Args:
            proposals: List of proposal dicts from propose_programs
            input_grid: (B, H, W) test inputs
            train_inputs: (B, N, H, W) training inputs
            train_outputs: (B, N, H, W) training outputs
            centroids: (B, K, 2) anchor centroids
            pair_mask: (B, N) valid pair mask
            
        Returns:
            results: List of verification results with executed outputs
        """
        B = input_grid.shape[0]
        results = []
        
        # Move all tensors to CPU ONCE at the start (avoids repeated transfers)
        input_grid_cpu = input_grid.detach().cpu()
        train_inputs_cpu = train_inputs.detach().cpu()
        train_outputs_cpu = train_outputs.detach().cpu()
        centroids_cpu = centroids.detach().cpu()
        pair_mask_cpu = pair_mask.detach().cpu() if pair_mask is not None else None
        
        # Run symbolic execution without autograd (huge speedup)
        with torch.inference_mode():
            for b in range(B):
                batch_results = []
                primary_anchor = centroids_cpu[b, 0]  # Use first anchor
                
                # Create anchor tensor for each demo pair (use primary anchor for all)
                N_demos = train_inputs_cpu[b].shape[0]
                demo_anchors = primary_anchor.unsqueeze(0).expand(N_demos, 2)
                
                for proposal in proposals:
                    program = self.decode_program(
                        proposal["program"][b],
                        proposal["colors"][b],
                        proposal["offsets"][b],
                    )
                    
                    # Skip empty programs (fast path)
                    if len(program) == 0:
                        batch_results.append({
                            "program": program,
                            "is_valid": False,
                            "accuracy": 0.0,
                            "predicted": input_grid_cpu[b],
                            "length": 0,
                        })
                        continue
                    
                    try:
                        # Verify on training demos with fail-fast
                        is_valid, accuracy = ProgramVerifier.verify(
                            program,
                            train_inputs_cpu[b],
                            train_outputs_cpu[b],
                            demo_anchors,
                            pair_mask_cpu[b] if pair_mask_cpu is not None else None,
                            require_all_match=self.config.require_demo_exact_match,
                        )
                        
                        # Only execute on test if verification passed (saves compute)
                        if is_valid:
                            predicted = ProgramExecutor.execute(
                                program,
                                input_grid_cpu[b],
                                primary_anchor,
                            )
                        else:
                            # Skip test execution for invalid programs
                            predicted = input_grid_cpu[b]
                        
                        batch_results.append({
                            "program": program,
                            "is_valid": is_valid,
                            "accuracy": accuracy,
                            "predicted": predicted,
                            "length": len(program),
                        })
                    except Exception as e:
                        # Gracefully handle any execution errors
                        print(f"[ARPS WARNING] Program execution failed: {e}")
                        batch_results.append({
                            "program": program,
                            "is_valid": False,
                            "accuracy": 0.0,
                            "predicted": input_grid_cpu[b],
                            "length": len(program),
                        })
                
                results.append(batch_results)
        
        return results
    
    def select_best_program(
        self,
        results: List[List[Dict[str, Any]]],
    ) -> List[Optional[Dict[str, Any]]]:
        """
        Select best program for each batch item.
        
        Ranking criteria:
        1. Valid programs (solve all demos) first
        2. Among valid: prefer shorter (MDL)
        3. Among same length: prefer higher accuracy
        
        Args:
            results: Verification results from execute_and_verify
            
        Returns:
            best: List of best program dicts (None if no valid program)
        """
        best_programs = []
        
        for batch_results in results:
            # Filter valid programs
            valid = [r for r in batch_results if r["is_valid"]]
            
            if not valid:
                best_programs.append(None)
                continue
            
            # Sort by length (MDL), then accuracy
            if self.config.use_mdl_ranking:
                valid.sort(key=lambda x: (x["length"], -x["accuracy"]))
            else:
                valid.sort(key=lambda x: -x["accuracy"])
            
            best_programs.append(valid[0])
        
        return best_programs
    
    def compute_imitation_loss(
        self,
        clue_features: torch.Tensor,
        target_programs: List[Optional[List[Tuple[str, Dict[str, Any]]]]],
    ) -> torch.Tensor:
        """
        Compute imitation learning loss from verified programs.
        
        Args:
            clue_features: (B, K, D, H, W) features for conditioning
            target_programs: List of target programs (None for no valid program)
            
        Returns:
            loss: Scalar imitation loss
        """
        B = clue_features.shape[0]
        device = clue_features.device
        
        # Pool context
        pooled = clue_features.mean(dim=1)
        context = self.context_pool(pooled)
        
        total_loss = 0.0
        num_valid = 0
        
        for b in range(B):
            if target_programs[b] is None:
                continue
            
            program = target_programs[b]
            
            # Convert program to token targets
            tokens = []
            colors = []
            offsets = []
            
            for primitive_name, args in program:
                if primitive_name in self.primitive_to_idx:
                    tokens.append(self.primitive_to_idx[primitive_name])
                    colors.append(args.get("color", 0))
                    offset = args.get("offset", (0, 0))
                    offsets.append([offset[0], offset[1]])
            
            # Add END token
            tokens.append(self.primitive_to_idx["end"])
            colors.append(0)
            offsets.append([0, 0])
            
            if not tokens:
                continue
            
            # SAFETY: Truncate programs to max_length to prevent index overflow
            # Programs longer than max_length-1 (+ END token) would cause pos_embed OOB
            max_tokens = self.config.max_program_length
            if len(tokens) > max_tokens:
                tokens = tokens[:max_tokens]
                colors = colors[:max_tokens]
                offsets = offsets[:max_tokens]
            
            # Convert to tensors
            # SAFETY: Clamp token indices to valid range
            target_tokens = torch.tensor(tokens, device=device).clamp(0, len(self.primitives) - 1)
            target_colors = torch.tensor(colors, device=device).clamp(0, 9)
            target_offsets = torch.tensor(offsets, device=device, dtype=torch.float).clamp(-30.0, 30.0)
            
            # Compute teacher-forcing loss
            partial = None
            step_loss = 0.0
            
            # SAFETY: Limit iterations to prevent sequence overflow
            max_steps = min(len(tokens), self.config.max_program_length)
            for t in range(max_steps):
                token_logits, color_logits, offset_pred = self.proposal_head(
                    context[b:b+1], partial
                )
                
                # Token loss - SAFETY: Clamp target to valid range for cross_entropy
                safe_token_target = target_tokens[t:t+1].clamp(0, token_logits.shape[-1] - 1)
                step_loss += F.cross_entropy(token_logits, safe_token_target)
                
                # Argument losses (only for non-END tokens)
                if t < max_steps - 1:
                    # SAFETY: Clamp color target to valid range
                    safe_color_target = target_colors[t:t+1].clamp(0, color_logits.shape[-1] - 1)
                    step_loss += F.cross_entropy(color_logits, safe_color_target) * 0.5
                    step_loss += F.mse_loss(offset_pred, target_offsets[t:t+1]) * 0.5
                
                # Update partial
                if partial is None:
                    partial = torch.full((1, 1), len(self.primitives), device=device)
                partial = torch.cat([partial, target_tokens[t:t+1].unsqueeze(-1)], dim=-1)
                
                # SAFETY: Stop early if partial would exceed position embedding size
                if partial.shape[1] >= self.config.max_program_length:
                    break
            
            total_loss += step_loss / max(max_steps, 1)
            num_valid += 1
        
        if num_valid == 0:
            return torch.tensor(0.0, device=device)
        
        return total_loss / num_valid
    
    def forward(
        self,
        clue_features: torch.Tensor,
        input_grid: torch.Tensor,
        train_inputs: torch.Tensor,
        train_outputs: torch.Tensor,
        centroids: torch.Tensor,
        pair_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Full ARPS forward pass.
        
        Args:
            clue_features: (B, K, D, H, W) from MSRE
            input_grid: (B, H, W) test inputs
            train_inputs: (B, N, H, W) training inputs
            train_outputs: (B, N, H, W) training outputs
            centroids: (B, K, 2) anchor centroids
            pair_mask: (B, N) valid pair mask
            temperature: Sampling temperature
            
        Returns:
            result dict with:
                - best_programs: Best verified programs
                - predicted_grids: (B, H, W) predictions from best programs
                - imitation_loss: Imitation learning loss
                - search_stats: Search statistics
        """
        B = input_grid.shape[0]
        device = input_grid.device
        
        # 1. Propose programs
        proposals = self.propose_programs(
            clue_features,
            temperature=temperature,
            num_samples=self.config.top_k_proposals,
        )
        
        # 2. Execute and verify
        results = self.execute_and_verify(
            proposals,
            input_grid,
            train_inputs,
            train_outputs,
            centroids,
            pair_mask,
        )
        
        # 3. Select best programs
        best_dicts = self.select_best_program(results)
        best_programs = [d["program"] if d is not None else None for d in best_dicts]
        
        # 4. Get predictions from best programs (move back to original device)
        predicted_grids = []
        for b in range(B):
            if best_dicts[b] is not None:
                pred = best_dicts[b]["predicted"]
                # Ensure on correct device (predictions come from CPU execution)
                if pred.device != device:
                    pred = pred.to(device)
                predicted_grids.append(pred)
            else:
                # No valid program - return input unchanged
                predicted_grids.append(input_grid[b])
        predicted_grids = torch.stack(predicted_grids)
        
        # 5. Compute imitation loss
        imitation_loss = torch.tensor(0.0, device=device)
        if self.training and self.config.search_during_training:
            imitation_loss = self.compute_imitation_loss(clue_features, best_programs)
        
        # 6. Compute stats
        num_valid = sum(1 for d in best_dicts if d is not None)
        avg_length = sum(d["length"] for d in best_dicts if d is not None) / max(num_valid, 1)
        
        return {
            "best_programs": best_programs,
            "predicted_grids": predicted_grids,
            "imitation_loss": imitation_loss,
            "search_stats": {
                "num_valid_programs": num_valid,
                "avg_program_length": avg_length,
                "num_proposals": len(proposals),
            },
        }


def create_arps_from_config(config: dict, hidden_dim: int = 256) -> Optional[ARPS]:
    """
    Factory function to create ARPS module from YAML config.
    
    Args:
        config: Dictionary from YAML config['model']['arps_dsl_search']
        hidden_dim: Hidden dimension for neural components (default: 256)
        
    Returns:
        ARPS module if enabled, None otherwise
    """
    if not config.get('enabled', False):
        return None
    
    # Use hidden_dim from config if specified, otherwise use parameter
    effective_hidden_dim = config.get('hidden_dim', hidden_dim)
    
    arps_config = ARPSConfig(
        enabled=True,
        use_as_auxiliary=config.get('use_as_auxiliary', True),
        max_program_length=config.get('max_program_length', 12),  # Updated default
        beam_size=config.get('beam_size', 64),                     # Updated default
        top_k_proposals=config.get('top_k_proposals', 8),          # Updated default
        primitives=config.get('primitives', [
            "select_color", "select_connected", "translate",
            "reflect_x", "reflect_y", "rotate_90", "rotate_180",
            "paint", "copy_paste", "tile", "crop", "fill"
        ]),
        require_demo_exact_match=config.get('require_demo_exact_match', True),
        use_mdl_ranking=config.get('use_mdl_ranking', True),
        search_during_training=config.get('search_during_training', True),
        imitation_weight=config.get('imitation_weight', 0.1),
        hidden_dim=effective_hidden_dim,
    )
    
    return ARPS(arps_config)
