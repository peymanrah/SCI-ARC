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
    max_program_length: int = 8
    beam_size: int = 32
    top_k_proposals: int = 4
    
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
        
        Args:
            grid: (H, W) input grid
            seed_offset: (row_offset, col_offset) relative to anchor
            anchor: (2,) anchor position
            
        Returns:
            mask: (H, W) binary mask of connected component
        """
        H, W = grid.shape
        device = grid.device
        
        # Compute seed position in absolute coords
        seed_row = int(anchor[0].item()) + seed_offset[0]
        seed_col = int(anchor[1].item()) + seed_offset[1]
        
        # Clamp to grid bounds
        seed_row = max(0, min(H - 1, seed_row))
        seed_col = max(0, min(W - 1, seed_col))
        
        # Get seed color
        seed_color = grid[seed_row, seed_col].item()
        
        # BFS to find connected component
        mask = torch.zeros(H, W, device=device)
        visited = torch.zeros(H, W, dtype=torch.bool, device=device)
        
        queue = [(seed_row, seed_col)]
        visited[seed_row, seed_col] = True
        
        while queue:
            r, c = queue.pop(0)
            if grid[r, c].item() == seed_color:
                mask[r, c] = 1.0
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W and not visited[nr, nc]:
                        visited[nr, nc] = True
                        queue.append((nr, nc))
        
        return mask
    
    @staticmethod
    def translate(
        selection: torch.Tensor,
        offset: Tuple[int, int],
        anchor: torch.Tensor,
    ) -> torch.Tensor:
        """
        Translate selection by offset (relative to anchor frame).
        
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
        
        # Find selection centroid
        indices = torch.nonzero(selection)
        if len(indices) == 0:
            return selection
        
        centroid = indices.float().mean(dim=0)
        
        # Compute absolute offset: selection_centroid → (anchor + offset)
        target_row = anchor[0] + offset[0]
        target_col = anchor[1] + offset[1]
        
        abs_offset_row = int(target_row - centroid[0])
        abs_offset_col = int(target_col - centroid[1])
        
        # Apply translation WITHOUT wrap-around (proper ARC semantics)
        # Pixels that move outside the grid are clipped, not wrapped
        translated = torch.zeros_like(selection)
        
        for r in range(H):
            for c in range(W):
                if selection[r, c] > 0:
                    new_r = r + abs_offset_row
                    new_c = c + abs_offset_col
                    # Only keep pixels that stay within bounds (no wrap-around)
                    if 0 <= new_r < H and 0 <= new_c < W:
                        translated[new_r, new_c] = selection[r, c]
        
        return translated
    
    @staticmethod
    def reflect_x(
        selection: torch.Tensor,
        anchor: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reflect selection across horizontal axis through anchor.
        
        Args:
            selection: (H, W) binary selection mask
            anchor: (2,) anchor position
            
        Returns:
            reflected: (H, W) reflected mask
        """
        H, W = selection.shape
        anchor_row = int(anchor[0].item())
        
        result = torch.zeros_like(selection)
        for r in range(H):
            for c in range(W):
                if selection[r, c] > 0:
                    # Reflect row around anchor_row
                    new_r = 2 * anchor_row - r
                    if 0 <= new_r < H:
                        result[new_r, c] = selection[r, c]
        
        return result
    
    @staticmethod
    def reflect_y(
        selection: torch.Tensor,
        anchor: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reflect selection across vertical axis through anchor.
        """
        H, W = selection.shape
        anchor_col = int(anchor[1].item())
        
        result = torch.zeros_like(selection)
        for r in range(H):
            for c in range(W):
                if selection[r, c] > 0:
                    new_c = 2 * anchor_col - c
                    if 0 <= new_c < W:
                        result[r, new_c] = selection[r, c]
        
        return result
    
    @staticmethod
    def rotate_90(
        selection: torch.Tensor,
        anchor: torch.Tensor,
    ) -> torch.Tensor:
        """
        Rotate selection 90° clockwise around anchor.
        """
        H, W = selection.shape
        anchor_row = int(anchor[0].item())
        anchor_col = int(anchor[1].item())
        
        result = torch.zeros_like(selection)
        for r in range(H):
            for c in range(W):
                if selection[r, c] > 0:
                    # Translate to anchor origin
                    dr = r - anchor_row
                    dc = c - anchor_col
                    # Rotate 90° CW: (dr, dc) → (dc, -dr)
                    new_dr = dc
                    new_dc = -dr
                    # Translate back
                    new_r = anchor_row + new_dr
                    new_c = anchor_col + new_dc
                    if 0 <= new_r < H and 0 <= new_c < W:
                        result[int(new_r), int(new_c)] = selection[r, c]
        
        return result
    
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
        """Copy selected region and paste at offset from anchor."""
        translated_mask = DSLPrimitives.translate(selection, offset, anchor)
        
        # Get colors from selection
        H, W = grid.shape
        result = grid.clone()
        
        # Find where to copy from (original selection) and to (translated)
        src_indices = torch.nonzero(selection > 0.5)
        dst_indices = torch.nonzero(translated_mask > 0.5)
        
        if len(src_indices) == len(dst_indices):
            for i in range(len(src_indices)):
                sr, sc = src_indices[i]
                dr, dc = dst_indices[i]
                if 0 <= dr < H and 0 <= dc < W:
                    result[dr, dc] = grid[sr, sc]
        
        return result


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
        self.pos_embed = nn.Embedding(max_length, hidden_dim)
        
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
        
        # Embed tokens
        token_emb = self.token_embed(partial_program)  # (B, L, D)
        
        # Add positional embeddings
        positions = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
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
                # Top-K filtering
                topk_vals, topk_idx = torch.topk(logits, min(top_k, logits.shape[-1]))
                probs = F.softmax(topk_vals, dim=-1)
                sampled_idx = torch.multinomial(probs, 1)
                token = topk_idx.gather(-1, sampled_idx).squeeze(-1)
            else:
                token = token_logits.argmax(dim=-1)
            
            # Sample arguments
            color = color_logits.argmax(dim=-1)
            offset = offset_pred
            
            program.append(token)
            colors.append(color)
            offsets.append(offset)
            
            # Update partial program
            if partial is None:
                partial = torch.full((B, 1), self.num_primitives, device=device)
            partial = torch.cat([partial, token.unsqueeze(-1)], dim=-1)
            
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
    """
    
    @staticmethod
    def verify(
        program: List[Tuple[str, Dict[str, Any]]],
        train_inputs: torch.Tensor,
        train_outputs: torch.Tensor,
        anchors: torch.Tensor,
        pair_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[bool, float]:
        """
        Verify program against training demos.
        
        Args:
            program: DSL program
            train_inputs: (N, H, W) training input grids
            train_outputs: (N, H, W) training output grids
            anchors: (N, 2) anchor positions for each pair
            pair_mask: (N,) valid pair mask
            
        Returns:
            is_valid: True if program solves all demos
            accuracy: Fraction of demos solved
        """
        N = train_inputs.shape[0]
        
        num_valid = 0
        num_correct = 0
        
        for i in range(N):
            if pair_mask is not None and not pair_mask[i]:
                continue
            
            num_valid += 1
            
            predicted = ProgramExecutor.execute(
                program,
                train_inputs[i],
                anchors[i],
            )
            
            if torch.equal(predicted, train_outputs[i]):
                num_correct += 1
        
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
        
        Args:
            tokens: (L,) program token indices
            colors: (L,) color arguments
            offsets: (L, 2) offset arguments
            
        Returns:
            program: List of (primitive_name, args_dict) tuples
        """
        program = []
        
        for i in range(len(tokens)):
            token_idx = tokens[i].item()
            if token_idx >= len(self.idx_to_primitive):
                continue
            
            primitive_name = self.idx_to_primitive[token_idx]
            if primitive_name == "end":
                break
            
            args = {
                "color": int(colors[i].item()),
                "offset": (int(offsets[i, 0].item()), int(offsets[i, 1].item())),
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
        
        for b in range(B):
            batch_results = []
            primary_anchor = centroids[b, 0]  # Use first anchor
            
            for proposal in proposals:
                program = self.decode_program(
                    proposal["program"][b],
                    proposal["colors"][b],
                    proposal["offsets"][b],
                )
                
                # Verify on training demos
                is_valid, accuracy = ProgramVerifier.verify(
                    program,
                    train_inputs[b],
                    train_outputs[b],
                    centroids[b, :train_inputs.shape[1]],
                    pair_mask[b] if pair_mask is not None else None,
                )
                
                # Execute on test input
                predicted = ProgramExecutor.execute(
                    program,
                    input_grid[b],
                    primary_anchor,
                )
                
                batch_results.append({
                    "program": program,
                    "is_valid": is_valid,
                    "accuracy": accuracy,
                    "predicted": predicted,
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
            
            # Convert to tensors
            target_tokens = torch.tensor(tokens, device=device)
            target_colors = torch.tensor(colors, device=device)
            target_offsets = torch.tensor(offsets, device=device, dtype=torch.float)
            
            # Compute teacher-forcing loss
            partial = None
            step_loss = 0.0
            
            for t in range(len(tokens)):
                token_logits, color_logits, offset_pred = self.proposal_head(
                    context[b:b+1], partial
                )
                
                # Token loss
                step_loss += F.cross_entropy(token_logits, target_tokens[t:t+1])
                
                # Argument losses (only for non-END tokens)
                if t < len(tokens) - 1:
                    step_loss += F.cross_entropy(color_logits, target_colors[t:t+1]) * 0.5
                    step_loss += F.mse_loss(offset_pred, target_offsets[t:t+1]) * 0.5
                
                # Update partial
                if partial is None:
                    partial = torch.full((1, 1), len(self.primitives), device=device)
                partial = torch.cat([partial, target_tokens[t:t+1].unsqueeze(-1)], dim=-1)
            
            total_loss += step_loss / len(tokens)
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
        
        # 4. Get predictions from best programs
        predicted_grids = []
        for b in range(B):
            if best_dicts[b] is not None:
                predicted_grids.append(best_dicts[b]["predicted"])
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


def create_arps_from_config(config: dict) -> Optional[ARPS]:
    """
    Factory function to create ARPS module from YAML config.
    
    Args:
        config: Dictionary from YAML config['model']['arps_dsl_search']
        
    Returns:
        ARPS module if enabled, None otherwise
    """
    if not config.get('enabled', False):
        return None
    
    arps_config = ARPSConfig(
        enabled=True,
        use_as_auxiliary=config.get('use_as_auxiliary', True),
        max_program_length=config.get('max_program_length', 8),
        beam_size=config.get('beam_size', 32),
        top_k_proposals=config.get('top_k_proposals', 4),
        primitives=config.get('primitives', [
            "select_color", "select_connected", "translate",
            "reflect_x", "reflect_y", "rotate_90", "rotate_180",
            "paint", "copy_paste", "tile", "crop", "fill"
        ]),
        require_demo_exact_match=config.get('require_demo_exact_match', True),
        use_mdl_ranking=config.get('use_mdl_ranking', True),
        search_during_training=config.get('search_during_training', True),
        imitation_weight=config.get('imitation_weight', 0.1),
        hidden_dim=config.get('hidden_dim', 256),
    )
    
    return ARPS(arps_config)
