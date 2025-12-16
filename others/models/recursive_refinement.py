"""
Recursive Refinement for SCI-ARC

TRM-style recursive refinement module, conditioned on z_task from the SCI encoders.

Key differences from pure TRM:
1. Conditioned on z_task (task understanding from SCI)
2. This injects explicit structural understanding into the refinement process

The recursive process:
- For k = 1 to K (supervision steps):
    - For n = 1 to N (recursion steps):
        - z = f(x, y, z, z_task)  ← Update latent (conditioned on z_task!)
    - y = g(y, z)  ← Update answer
    - Loss_k = CE(y_k, target)  ← Deep supervision

This is the TRM insight: iterative refinement allows the model to progressively
improve its answer, correcting errors from previous steps.
"""

from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from sci_arc.models.causal_binding import TaskConditioner


@dataclass
class RefinementState:
    """State carried across refinement steps."""
    y: torch.Tensor  # Current answer: [B, H*W, D]
    z: torch.Tensor  # Latent state: [B, L, D]
    step: int        # Current step


class RecursiveRefinement(nn.Module):
    """
    TRM-style recursive refinement, conditioned on z_task from SCI.
    
    Key modification from TRM:
    - Latent update f() is conditioned on z_task
    - This injects the structural understanding into refinement
    
    Architecture follows TRM closely:
    - Lightweight (2 layers only)
    - RMS normalization
    - SwiGLU activations
    - Residual connections
    
    Parameters: ~3M (keeping TRM's tiny philosophy)
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        max_cells: int = 900,  # 30×30
        num_colors: int = 10,
        H_cycles: int = 16,    # Supervision steps (outer loop)
        L_cycles: int = 4,     # Recursion steps per supervision (inner loop)
        L_layers: int = 2,     # Network depth (TRM uses 2)
        latent_size: int = 64, # Latent sequence length
        num_heads: int = 4,
        dropout: float = 0.1,
        deep_supervision: bool = True,
        use_task_conditioning: bool = True
    ):
        """
        Args:
            hidden_dim: Hidden dimension
            max_cells: Maximum output cells (30*30 for ARC)
            num_colors: Number of output colors
            H_cycles: Number of high-level supervision steps
            L_cycles: Number of low-level recursion steps per H-step
            L_layers: Number of transformer layers
            latent_size: Size of latent sequence
            num_heads: Number of attention heads
            dropout: Dropout probability
            deep_supervision: Whether to use deep supervision (loss at each step)
            use_task_conditioning: Whether to condition on z_task
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.max_cells = max_cells
        self.H_cycles = H_cycles
        self.L_cycles = L_cycles
        self.latent_size = latent_size
        self.deep_supervision = deep_supervision
        self.use_task_conditioning = use_task_conditioning
        
        # === INITIAL STATES ===
        # Learnable initial answer and latent (like TRM)
        self.y_init = nn.Parameter(torch.randn(1, max_cells, hidden_dim) * 0.02)
        self.z_init = nn.Parameter(torch.randn(1, latent_size, hidden_dim) * 0.02)
        
        # === TASK CONDITIONING ===
        if use_task_conditioning:
            self.task_conditioner = TaskConditioner(
                task_dim=hidden_dim,
                target_dim=hidden_dim,
                conditioning_type="film"
            )
        
        # === SHARED REASONING MODULE (TRM-style parameter efficiency) ===
        # TRM uses the same L_level for both z_L and z_H updates
        # This dramatically reduces parameter count
        self.latent_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
                activation='gelu'
            ),
            num_layers=L_layers
        )
        
        # Input projection (combine x, y, z)
        self.input_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # === SPATIAL CROSS-ATTENTION (Critical for spatial reasoning) ===
        # Answer y attends to full spatial input x_test_emb
        # This preserves WHERE information, not just WHAT
        self.spatial_cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.spatial_norm = nn.LayerNorm(hidden_dim)
        
        # === ANSWER UPDATE: g(y, z, x_spatial) ===
        # Now includes spatial context from cross-attention
        self.answer_update = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),  # y + z + spatial_context
            nn.SiLU(),  # Changed from GELU to SiLU (like TRM's SwiGLU base)
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # === OUTPUT PROJECTION ===
        self.to_logits = nn.Linear(hidden_dim, num_colors)
        
        # Layer normalization
        self.y_norm = nn.LayerNorm(hidden_dim)
        self.z_norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        x_test_emb: torch.Tensor,  # [B, H*W, D] encoded test input
        z_task: torch.Tensor,       # [B, D] from CausalBinding
        target_shape: Tuple[int, int],
        memory_efficient: bool = True  # Use TRM's H_cycles-1 no_grad pattern
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Recursively refine answer conditioned on task understanding.
        
        Args:
            x_test_emb: [B, N, D] encoded test input (flattened)
            z_task: [B, D] task embedding from SCI encoders
            target_shape: (H_out, W_out) expected output size
            memory_efficient: If True, use TRM's gradient-free early steps
        
        Returns:
            outputs: List of predictions at each H_cycle (for deep supervision)
            final: Final prediction [B, H, W, num_colors]
        """
        B = x_test_emb.size(0)
        H_out, W_out = target_shape
        num_cells = H_out * W_out
        
        # Initialize answer and latent
        y = self.y_init[:, :num_cells, :].expand(B, -1, -1).clone()
        z = self.z_init.expand(B, -1, -1).clone()
        
        # Keep full spatial input for cross-attention (NO POOLING - preserves spatial info)
        # x_test_emb: [B, N_in, D] - full spatial context
        
        outputs = []
        
        # === TRM MEMORY OPTIMIZATION ===
        # Run H_cycles-1 without gradients to save memory
        # Only the last H_cycle needs gradients for backprop
        # NOTE: When memory_efficient=True, deep supervision on early steps is ineffective
        if memory_efficient and self.training and self.H_cycles > 1:
            with torch.no_grad():
                for h in range(self.H_cycles - 1):
                    y, z = self._single_h_cycle(x_test_emb, y, z, z_task, num_cells)
                    logits = self.to_logits(y).view(B, H_out, W_out, -1)
                    outputs.append(logits.detach())
            
            # Last H_cycle with gradients
            y, z = self._single_h_cycle(x_test_emb, y, z, z_task, num_cells)
            logits = self.to_logits(y).view(B, H_out, W_out, -1)
            outputs.append(logits)
        else:
            # Standard mode: all cycles with gradients
            for h in range(self.H_cycles):
                y, z = self._single_h_cycle(x_test_emb, y, z, z_task, num_cells)
                logits = self.to_logits(y).view(B, H_out, W_out, -1)
                outputs.append(logits)
        
        return outputs, outputs[-1]
    
    def _single_h_cycle(
        self,
        x_test_emb: torch.Tensor,  # [B, N_in, D] - FULL spatial input (not pooled!)
        y: torch.Tensor,
        z: torch.Tensor,
        z_task: torch.Tensor,
        num_cells: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Execute a single H-cycle (outer loop iteration).
        
        Extracted for memory-efficient training pattern.
        
        Key improvement: Uses cross-attention to x_test_emb to preserve
        spatial information (WHERE things are, not just WHAT).
        """
        # === INNER RECURSION (L_cycles) ===
        for l in range(self.L_cycles):
            # Pool current answer for latent update signal
            y_pool = y.mean(dim=1)  # [B, D]
            x_pool = x_test_emb.mean(dim=1)  # [B, D] - for latent update only
            
            # Combine inputs for latent update
            combined = torch.cat([x_pool, y_pool], dim=-1)  # [B, 2D]
            update_signal = self.input_proj(combined)  # [B, D]
            
            # Condition on z_task (KEY SCI CONTRIBUTION)
            if self.use_task_conditioning:
                update_signal = self.task_conditioner(
                    update_signal.unsqueeze(1),
                    z_task
                ).squeeze(1)
            
            # Update latent by injecting update signal
            z_input = z + update_signal.unsqueeze(1)
            z = self.latent_encoder(z_input)
            z = self.z_norm(z)
        
        # === SPATIAL CROSS-ATTENTION (Critical fix for spatial reasoning) ===
        # Answer y attends to full spatial input x_test_emb
        # Query: y (current answer cells), Key/Value: x_test_emb (input pixels)
        # This allows each output cell to "look at" relevant input locations
        spatial_context, _ = self.spatial_cross_attention(
            query=y,           # [B, num_cells, D] - where we're predicting
            key=x_test_emb,    # [B, N_in, D] - full spatial input
            value=x_test_emb   # [B, N_in, D]
        )
        spatial_context = self.spatial_norm(spatial_context + y)  # Residual + norm
        
        # === ANSWER UPDATE ===
        # Broadcast z to match y
        z_broadcast = z.mean(dim=1, keepdim=True).expand(-1, num_cells, -1)
        
        # Update answer with y, z, AND spatial context
        update_input = torch.cat([y, z_broadcast, spatial_context], dim=-1)  # [B, N, 3D]
        y_update = self.answer_update(update_input)
        y = y + y_update  # Residual
        y = self.y_norm(y)
        
        return y, z
    
    def forward_with_states(
        self,
        x_test_emb: torch.Tensor,
        z_task: torch.Tensor,
        target_shape: Tuple[int, int]
    ) -> Tuple[List[torch.Tensor], torch.Tensor, List[RefinementState]]:
        """Forward pass that also returns intermediate states for analysis."""
        B = x_test_emb.size(0)
        H_out, W_out = target_shape
        num_cells = H_out * W_out
        
        y = self.y_init[:, :num_cells, :].expand(B, -1, -1).clone()
        z = self.z_init.expand(B, -1, -1).clone()
        # Keep full spatial input (no pooling)
        
        outputs = []
        states = []
        
        for h in range(self.H_cycles):
            for l in range(self.L_cycles):
                y_pool = y.mean(dim=1)
                x_pool = x_test_emb.mean(dim=1)  # Pool only for latent update
                
                combined = torch.cat([x_pool, y_pool], dim=-1)
                update_signal = self.input_proj(combined)
                
                if self.use_task_conditioning:
                    update_signal = self.task_conditioner(
                        update_signal.unsqueeze(1),
                        z_task
                    ).squeeze(1)
                
                z_input = z + update_signal.unsqueeze(1)
                z = self.z_norm(self.latent_encoder(z_input))
            
            # Spatial cross-attention: y attends to x_test_emb
            spatial_context, _ = self.spatial_cross_attention(
                query=y, key=x_test_emb, value=x_test_emb
            )
            spatial_context = self.spatial_norm(spatial_context + y)
            
            z_broadcast = z.mean(dim=1, keepdim=True).expand(-1, num_cells, -1)
            update_input = torch.cat([y, z_broadcast, spatial_context], dim=-1)
            y = self.y_norm(y + self.answer_update(update_input))
            
            logits = self.to_logits(y).view(B, H_out, W_out, -1)
            outputs.append(logits)
            
            states.append(RefinementState(y=y.clone(), z=z.clone(), step=h))
        
        return outputs, outputs[-1], states


class TRMStyleBlock(nn.Module):
    """
    TRM-style transformer block with RMS normalization and SwiGLU.
    
    This follows TRM's architecture more closely for fair comparison.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        expansion: float = 2.67,  # SwiGLU expansion
        dropout: float = 0.1,
        rms_norm_eps: float = 1e-5
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # SwiGLU MLP
        intermediate_dim = int(hidden_dim * expansion)
        self.w1 = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.w2 = nn.Linear(intermediate_dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        
        self.rms_norm_eps = rms_norm_eps
    
    def rms_norm(self, x: torch.Tensor) -> torch.Tensor:
        """RMS normalization (from TRM)."""
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.rms_norm_eps)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with post-norm
        attn_out, _ = self.self_attn(x, x, x)
        x = self.rms_norm(x + attn_out)
        
        # SwiGLU MLP
        gate = F.silu(self.w1(x))
        value = self.w3(x)
        mlp_out = self.w2(gate * value)
        x = self.rms_norm(x + mlp_out)
        
        return x


class RecursiveRefinementTRMStyle(RecursiveRefinement):
    """
    More faithful TRM-style recursive refinement.
    
    Uses TRM's specific architecture choices:
    - RMS normalization
    - SwiGLU activation
    - Specific initialization
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Replace transformer encoder with TRM-style blocks
        hidden_dim = kwargs.get('hidden_dim', 256)
        L_layers = kwargs.get('L_layers', 2)
        
        self.latent_encoder = nn.Sequential(*[
            TRMStyleBlock(hidden_dim)
            for _ in range(L_layers)
        ])
