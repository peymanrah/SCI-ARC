"""
Recursive Solver for RLAN

The Recursive Solver is the output generation module of RLAN. It iteratively
refines its prediction over T steps, conditioning on:
- Clue-relative features (from DSC + MSRE)
- Count embeddings (from LCR)
- Predicate gates (from SPH)
- Input grid (for residual/copy connections)

Key Features:
- ConvGRU for iterative refinement (with optional SwiGLU activation)
- Multi-scale feature integration
- Predicate-based gating for conditional computation
- Deep supervision at each step
- Adaptive Computation Time (ACT) for variable reasoning depth

Architecture:
    For t = 1 to T:
        1. Aggregate clue features
        2. Inject count information
        3. Modulate by predicates
        4. ConvGRU update (with SwiGLU)
        5. Predict logits
        6. (Optional) ACT halt decision

Module Integrations:
    - SwiGLU: Used in FFN for improved gradient flow
    - RoPE-inspired: Position-aware updates (via MSRE)
    - ACT: Adaptive halting for variable reasoning depth
"""

import math
from typing import Optional, Tuple, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

# Import SwiGLU for enhanced FFN
try:
    from .activations import SwiGLU, SwiGLUConv2d
    SWIGLU_AVAILABLE = True
except ImportError:
    SWIGLU_AVAILABLE = False


def _find_multiple(a: int, b: int) -> int:
    """Round up a to the nearest multiple of b."""
    return (-(a // -b)) * b


def soft_clamp(x: torch.Tensor, threshold: float = 8.0, max_val: float = 10.0) -> torch.Tensor:
    """
    Soft clamp for hidden states that preserves gradients at boundaries.
    
    Unlike hard clamp (torch.clamp) which kills gradients at boundaries,
    soft clamp uses tanh compression to maintain gradient flow.
    
    For |x| <= threshold: returns x unchanged (identity)
    For |x| > threshold: smoothly compresses towards max_val using tanh
    
    Args:
        x: Input tensor (hidden states)
        threshold: Values below this are unchanged (default 8.0)
        max_val: Maximum output magnitude (default 10.0)
        
    Returns:
        Soft-clamped tensor with preserved gradients
    """
    scale = (max_val - threshold) / 2
    abs_x = x.abs()
    sign_x = x.sign()
    excess = (abs_x - threshold).clamp(min=0)
    compressed = threshold + (max_val - threshold) * torch.tanh(excess / scale)
    return torch.where(abs_x <= threshold, x, sign_x * compressed)


def soft_clamp_logits(x: torch.Tensor, threshold: float = 1000.0, max_val: float = 2000.0) -> torch.Tensor:
    """
    Soft clamp that allows natural growth up to threshold, then compresses.
    
    This prevents NaN from extreme logits (10K+) while allowing the model
    to express strong confidence (unlike hard clamp at ±50 which caused collapse).
    
    For |x| <= threshold: returns x unchanged (identity)
    For |x| > threshold: smoothly compresses towards max_val using tanh
    
    The function is:
    - Continuous and differentiable everywhere
    - Identity in the normal operating range
    - Bounded to ±max_val to prevent overflow
    
    Args:
        x: Input logits
        threshold: Values below this are unchanged (default 1000)
        max_val: Maximum output magnitude (default 2000)
        
    Returns:
        Soft-clamped logits
    """
    # For values within threshold, return unchanged
    # For values beyond threshold, use tanh to smoothly compress
    # 
    # Formula: sign(x) * (threshold + (max_val - threshold) * tanh((|x| - threshold) / scale))
    # where scale controls how quickly we approach max_val
    
    scale = (max_val - threshold) / 2  # Reach ~max_val when |x| = threshold + 2*scale
    
    abs_x = x.abs()
    sign_x = x.sign()
    
    # Excess beyond threshold
    excess = (abs_x - threshold).clamp(min=0)
    
    # Compress excess using tanh
    compressed = threshold + (max_val - threshold) * torch.tanh(excess / scale)
    
    # Apply only where |x| > threshold
    result = torch.where(abs_x <= threshold, x, sign_x * compressed)
    
    return result


class ConvGRUCell(nn.Module):
    """
    Convolutional GRU cell for spatial state updates.
    
    Unlike standard GRU which operates on vectors, ConvGRU
    maintains spatial structure while performing recurrent updates.
    
    Enhanced with SwiGLU option for better gradient flow (TRM-style).
    
    HyperLoRA Support (Dec 2025):
    Accepts optional lora_deltas dict with task-specific weight modulations.
    When provided, gate outputs are adjusted by: output + LoRA_apply(output, delta)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        kernel_size: int = 3,
        use_swiglu: bool = True,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.use_swiglu = use_swiglu and SWIGLU_AVAILABLE
        padding = kernel_size // 2
        
        # Reset gate
        self.reset_gate = nn.Conv2d(
            input_dim + hidden_dim, hidden_dim,
            kernel_size=kernel_size, padding=padding
        )
        
        # Update gate
        self.update_gate = nn.Conv2d(
            input_dim + hidden_dim, hidden_dim,
            kernel_size=kernel_size, padding=padding
        )
        
        # Candidate state
        if self.use_swiglu:
            # Use SwiGLUConv2d for candidate generation (more expressive)
            # Input is (input_dim + hidden_dim), output is hidden_dim
            self.candidate = SwiGLUConv2d(
                channels=hidden_dim, # SwiGLUConv2d takes channels as input/output
                expansion=2.0,       # Expansion factor
                kernel_size=kernel_size
            )
            # Projection to match SwiGLU input requirement
            self.candidate_proj = nn.Conv2d(
                input_dim + hidden_dim, hidden_dim,
                kernel_size=1
            )
        else:
            # Standard Tanh candidate
            self.candidate = nn.Conv2d(
                input_dim + hidden_dim, hidden_dim,
                kernel_size=kernel_size, padding=padding
            )
        
        self.norm = nn.GroupNorm(8, hidden_dim)
    
    def _apply_lora_spatial(
        self,
        features: torch.Tensor,  # (B, D, H, W)
        delta_w: torch.Tensor,   # (B, D, D)
    ) -> torch.Tensor:
        """
        Compute LoRA delta for spatial features.
        
        Returns ONLY the delta (not features + delta) so the caller can add it:
            output = features + _apply_lora_spatial(features, delta_w)
        
        This implements: y = Wx + dW·x where dW·x is the delta.
        """
        B, D, H, W = features.shape
        # Reshape: (B, D, H, W) → (B, H*W, D)
        features_flat = features.permute(0, 2, 3, 1).reshape(B, H * W, D)
        # Apply: (B, H*W, D) @ (B, D, D) → (B, H*W, D)
        delta_out = torch.bmm(features_flat, delta_w)
        # Reshape delta back to spatial format (no residual addition here)
        delta_spatial = delta_out.reshape(B, H, W, D).permute(0, 3, 1, 2)
        return delta_spatial
    
    def forward(
        self,
        x: torch.Tensor,
        h: Optional[torch.Tensor] = None,
        lora_deltas: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Single GRU step with optional HyperLoRA weight adaptation.
        
        Args:
            x: Input features (B, input_dim, H, W)
            h: Previous hidden state (B, hidden_dim, H, W) or None
            lora_deltas: Optional dict with keys 'gru_reset', 'gru_update', 'gru_candidate'
                        Each value is (B, hidden_dim, hidden_dim) weight delta
            
        Returns:
            h_new: Updated hidden state (B, hidden_dim, H, W)
        """
        B, _, H, W = x.shape
        
        if h is None:
            # Initialize hidden state from input features instead of zeros
            # This gives step 0 meaningful information to work with
            # Use a learnable projection or average pooling
            h = torch.zeros(B, self.hidden_dim, H, W, device=x.device, dtype=x.dtype)
            # Scale initial hidden state to match expected magnitude
            # This helps step 0 have similar dynamics to later steps
        
        # Concatenate input and hidden state
        combined = torch.cat([x, h], dim=1)
        
        # Compute gates
        r_logits = self.reset_gate(combined)
        z_logits = self.update_gate(combined)
        
        # Apply HyperLoRA modulation if provided
        # We apply LoRA to the pre-activation logits to effectively modulate the weights
        # y = (W + dW)x = Wx + dWx. Here we approximate dWx by applying dW to the output of Wx
        if lora_deltas is not None:
            if 'gru_reset' in lora_deltas:
                # Apply delta to logits: logits + delta_W @ logits
                # Note: This is a simplified application where we transform the output space
                r_delta = self._apply_lora_spatial(r_logits, lora_deltas['gru_reset'])
                r_logits = r_logits + r_delta
            if 'gru_update' in lora_deltas:
                z_delta = self._apply_lora_spatial(z_logits, lora_deltas['gru_update'])
                z_logits = z_logits + z_delta
        
        r = torch.sigmoid(r_logits)
        z = torch.sigmoid(z_logits)
        
        # Compute candidate
        combined_reset = torch.cat([x, r * h], dim=1)
        
        if self.use_swiglu:
            # SwiGLU path
            proj = self.candidate_proj(combined_reset)
            # Apply LoRA to projection if needed (before SwiGLU activation)
            if lora_deltas is not None and 'gru_candidate' in lora_deltas:
                proj_delta = self._apply_lora_spatial(proj, lora_deltas['gru_candidate'])
                proj = proj + proj_delta
                
            h_candidate = self.candidate(proj)
            # Bound the candidate to prevent drift in later steps
            # tanh bounds to [-1, 1] which helps stability
            h_candidate = torch.tanh(h_candidate)
        else:
            # Standard Tanh path
            cand_logits = self.candidate(combined_reset)
            
            if lora_deltas is not None and 'gru_candidate' in lora_deltas:
                cand_delta = self._apply_lora_spatial(cand_logits, lora_deltas['gru_candidate'])
                cand_logits = cand_logits + cand_delta
                
            h_candidate = torch.tanh(cand_logits)
        
        # Update hidden state with residual connection for stability
        h_new = (1 - z) * h + z * h_candidate
        
        # SOFT clamp to prevent extreme values while preserving gradients
        # Hard clamp kills gradients at boundaries, soft clamp preserves them
        h_new = soft_clamp(h_new, threshold=8.0, max_val=10.0)
        
        h_new = self.norm(h_new)
        
        return h_new


class PredicateGating(nn.Module):
    """
    Gate features based on predicate activations.
    
    Allows the model to conditionally activate different
    transformation pathways based on input properties.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_predicates: int,
    ):
        super().__init__()
        
        # Learn how each predicate affects features
        self.gate_proj = nn.Linear(num_predicates, hidden_dim)
        self.scale_proj = nn.Linear(num_predicates, hidden_dim)
        
    def forward(
        self,
        features: torch.Tensor,
        predicates: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply predicate-based gating.
        
        Args:
            features: Shape (B, D, H, W)
            predicates: Shape (B, P)
            
        Returns:
            gated: Shape (B, D, H, W)
        """
        B, D, H, W = features.shape
        
        # Compute gate and scale from predicates
        gate = torch.sigmoid(self.gate_proj(predicates))  # (B, D)
        scale = self.scale_proj(predicates)  # (B, D)
        
        # Apply to features (broadcast over spatial dims)
        gate = gate.view(B, D, 1, 1)
        scale = scale.view(B, D, 1, 1)
        
        gated = features * gate + scale
        
        return gated


class SolverCrossAttention(nn.Module):
    """
    Cross-attention module for the Solver to attend to support set features.
    
    This gives the Solver direct access to the examples at EVERY refinement step,
    rather than relying on pre-compressed context from the initial injection.
    
    Phase 2.5: Removes the information bottleneck identified in Phase 2 analysis.
    
    Memory Optimization (Dec 2025):
    - ContextEncoder already downsamples to 8×8 (64 tokens per pair)
    - With 4 pairs: 4 × 64 = 256 keys (very efficient)
    - No additional pooling needed!
    
    Mathematical Role:
    - DSC-injected features = GLOBAL prior ("what is this task?")
    - Solver cross-attention = LOCAL verification ("does my output match?")
    - These are COMPLEMENTARY, not interfering
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        pool_size: int = 8,  # Match ContextEncoder's spatial_downsample
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.pool_size = pool_size
        
        # Adaptive pooling to handle variable input sizes
        # If support features are already 8×8, this is a no-op
        # If they're larger (e.g., from old config), this downsamples
        self.support_pool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        
        # Multi-head cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Layer norms for pre-norm architecture (more stable)
        self.norm_q = nn.LayerNorm(hidden_dim)
        self.norm_kv = nn.LayerNorm(hidden_dim)
        self.norm_out = nn.LayerNorm(hidden_dim)
        
        # Output projection with gating (allows model to ignore context if not helpful)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        hidden_state: torch.Tensor,       # (B, D, H, W) solver's hidden state
        support_features: torch.Tensor,   # (B, N, D, H', W') support pair features (8×8 from ContextEncoder)
    ) -> torch.Tensor:
        """
        Apply cross-attention from solver hidden state to support features.
        
        At each refinement step, the solver can "look back" at the examples to:
        1. Verify its current hypothesis matches the transformation pattern
        2. Correct mistakes by comparing with ground truth examples
        3. Handle multi-step rules where intermediate states need guidance
        
        Memory cost per step:
        - Query: (B, H*W, D) = (8, 900, 256) = 7.4 MB
        - Key/Value: (B, N*64, D) = (8, 256, 256) = 2.1 MB
        - Attention: (B, heads, 900, 256) = 7.4 MB
        - Total: ~17 MB per step × 6 steps = ~100 MB (trivial on 24GB GPU)
        
        Returns:
            enhanced_state: (B, D, H, W) hidden state enhanced with support context
        """
        B, D, H, W = hidden_state.shape
        _, N, D_s, H_s, W_s = support_features.shape
        
        # Flatten hidden state to sequence: (B, H*W, D)
        h_flat = hidden_state.permute(0, 2, 3, 1).reshape(B, H * W, D)
        
        # Pool support features if needed (usually 8×8 already from ContextEncoder)
        # This handles variable input sizes gracefully
        support_pooled = support_features.reshape(B * N, D_s, H_s, W_s)
        support_pooled = self.support_pool(support_pooled)  # (B*N, D, pool_size, pool_size)
        _, _, H_p, W_p = support_pooled.shape
        support_pooled = support_pooled.reshape(B, N, D_s, H_p, W_p)
        
        # Flatten pooled support features: (B, N*36, D)
        support_flat = support_pooled.permute(0, 1, 3, 4, 2).reshape(B, N * H_p * W_p, D_s)
        
        # Pre-norm for stability
        q = self.norm_q(h_flat)
        kv = self.norm_kv(support_flat)
        
        # Cross-attention: hidden state queries support set
        attn_output, _ = self.cross_attn(
            query=q,
            key=kv,
            value=kv,
        )
        
        # Gated residual connection - allows model to skip if not useful
        gate = self.gate(h_flat)
        h_flat = h_flat + gate * self.dropout(attn_output)
        
        # Post-norm
        h_flat = self.norm_out(h_flat)
        
        # Reshape back to spatial: (B, D, H, W)
        enhanced = h_flat.reshape(B, H, W, D).permute(0, 3, 1, 2)
        
        return enhanced


class RecursiveSolver(nn.Module):
    """
    Recursive Solver - iterative refinement decoder for RLAN.
    
    Generates output predictions by iteratively refining a hidden state,
    conditioned on clue features, count embeddings, and predicates.
    
    Now supports Adaptive Computation Time (ACT) for variable steps.
    
    PHASE 2.5: Added SolverCrossAttention to give the solver direct access
    to support set features at every refinement step. This removes the
    information bottleneck where all context had to be compressed into
    the initial feature injection.
    
    ABLATION SUPPORT:
    - When use_lcr=False, count injection is skipped (not just zero input)
    - When use_sph=False, predicate gating is skipped
    - When use_solver_context=False, solver cross-attention is skipped
    - This avoids wasted computation and gradient noise from unused modules
    """
    
    def __init__(
        self,
        hidden_dim: int = 128,
        num_classes: int = 10,
        num_steps: int = 6,
        num_predicates: int = 8,
        num_colors: int = 10,
        dropout: float = 0.1,
        use_act: bool = False,  # Enable Adaptive Computation Time
        use_lcr: bool = True,   # Enable count injection (ablation flag)
        use_sph: bool = True,   # Enable predicate gating (ablation flag)
        use_feedback: bool = False,  # Use prediction feedback (disabled by default - causes gradient issues)
        use_solver_context: bool = True,  # NEW: Enable solver cross-attention to support set
        num_context_heads: int = 4,  # Heads for solver cross-attention
        use_dsc: bool = True,  # Kept for API compatibility but not used for module creation
        gradient_checkpointing: bool = False,  # Enable to reduce memory ~40%
    ):
        """
        Args:
            hidden_dim: Feature dimension
            num_classes: Output classes (10 colors, 0-9)
            num_steps: Number of refinement iterations (T)
            num_predicates: Number of predicate inputs
            num_colors: Number of input colors (for count embedding)
            dropout: Dropout probability
            use_act: Whether to use Adaptive Computation Time
            use_lcr: Whether to use count injection (skip if False)
            use_sph: Whether to use predicate gating (skip if False)
            use_feedback: Whether to use prediction feedback in refinement loop.
                          Disabled by default because argmax is non-differentiable,
                          causing later steps to receive gradient-disconnected inputs.
            use_solver_context: Whether to use cross-attention to support set at each step.
                                Phase 2.5 feature - gives solver direct access to examples.
            num_context_heads: Number of attention heads for solver cross-attention.
            use_dsc: Kept for API compatibility. Mode is determined at runtime.
            gradient_checkpointing: Whether to use gradient checkpointing to reduce memory.
                                   Trades compute for memory by recomputing activations during backward.
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_steps = num_steps
        self.use_act = use_act
        self.use_lcr = use_lcr
        self.use_sph = use_sph
        self.use_feedback = use_feedback
        self.use_solver_context = use_solver_context
        self.use_dsc = use_dsc
        self.gradient_checkpointing = gradient_checkpointing
        
        if gradient_checkpointing:
            print(f"[RecursiveSolver] Gradient checkpointing ENABLED (memory optimization)")
        
        # Clue feature aggregation
        self.clue_aggregator = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            nn.GELU(),
            nn.GroupNorm(8, hidden_dim),
        )
        
        # Count projection for global count embedding (B, num_colors, D) -> (B, D)
        # Always create for backward compatibility - forward handles both modes
        if use_lcr:
            self.count_proj = nn.Sequential(
                nn.Linear(hidden_dim * num_colors, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
            )
        else:
            self.count_proj = None
        
        # Predicate gating (only if SPH enabled)
        if use_sph:
            self.predicate_gate = PredicateGating(hidden_dim, num_predicates)
        else:
            self.predicate_gate = None
        
        # ConvGRU for iterative refinement (with SwiGLU)
        self.gru = ConvGRUCell(hidden_dim * 2, hidden_dim, use_swiglu=True)
        
        # Input grid embedding (for copy/residual connections)
        self.input_embed = nn.Embedding(num_colors + 1, hidden_dim)  # +1 for padding
        
        # Output head (per-step predictions for deep supervision)
        self.output_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.GroupNorm(8, hidden_dim),
            nn.Conv2d(hidden_dim, num_classes, 1),
        )
        
        # Initialize output head to combat background collapse
        # Problem: Model defaults to predicting background (class 0) which is 90% of pixels
        # Solution: Give foreground classes a slight positive bias to encourage exploration
        self._init_output_head_for_balanced_predictions()
        
        # ACT Controller (optional)
        if use_act:
            try:
                from .adaptive_computation import ACTController
                self.act_controller = ACTController(
                    hidden_dim=hidden_dim,
                    max_steps=num_steps,
                    use_q_continue=True
                )
            except ImportError:
                print("Warning: ACTController not found, disabling ACT")
                self.use_act = False
        
        # Phase 2.5: Solver Cross-Attention to Support Set
        # Gives the solver direct access to examples at EVERY refinement step
        if use_solver_context:
            self.solver_cross_attn = SolverCrossAttention(
                hidden_dim=hidden_dim,
                num_heads=num_context_heads,
                dropout=dropout,
            )
            print(f"[RecursiveSolver] Phase 2.5: Solver cross-attention ENABLED ({num_context_heads} heads)")
        else:
            self.solver_cross_attn = None
        
        self.dropout = nn.Dropout(dropout)
    
    def _init_output_head_for_balanced_predictions(self):
        """
        Initialize output head with NEUTRAL initialization.
        
        History of attempts:
        - bg_bias=+2.2 (50% BG): Model collapses to 100% BG immediately
        - bg_bias=-2.0 (1.5% BG): Model over-corrects, ends at 98% BG
        - bg_bias=-0.5 (6% BG): Combined with 10x FG weight, caused FG collapse!
        - bg_bias=0.0 (10% BG): Uniform, let weighted loss guide learning
        
        STABLE APPROACH (Dec 2025): bg_bias=0.0, NEUTRAL initialization
        This works when the architecture is simple (no noisy new modules).
        
        LESSON LEARNED: bg_bias=-0.5 caused instability when combined with
        HyperLoRA, HPM, and SolverCrossAttention - too much gradient noise.
        NEUTRAL initialization is most robust.
        """
        # Get the final Conv2d layer in output_head
        final_layer = None
        for module in self.output_head.modules():
            if isinstance(module, nn.Conv2d) and module.out_channels == self.num_classes:
                final_layer = module
        
        if final_layer is not None and final_layer.bias is not None:
            # NEUTRAL initialization - let the loss function guide learning
            # This is more stable than anti-collapse bias when architecture has
            # multiple modules that add noise (HyperLoRA, HPM, CrossAttention)
            with torch.no_grad():
                final_layer.bias[0] = 0.0  # NEUTRAL for background
                final_layer.bias[1:] = 0.0  # NEUTRAL for foreground classes
            print(f"[RecursiveSolver] Output head initialized: bg_bias=0.0 (NEUTRAL), fg_bias=0.0")
    
    def _aggregate_clues(
        self,
        clue_features: torch.Tensor,
        attention_maps: Optional[torch.Tensor] = None,
        stop_logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Aggregate features from multiple clues.
        
        CRITICAL DESIGN: Uses stop_probs to weight clue contributions with a 
        DETACHED divisor. This ensures:
        1. STABLE OUTPUT MAGNITUDE regardless of clue count (prevents BG collapse)
        2. GRADIENT FLOWS through clue_usage weights (enables clue count learning)
        
        The key insight: We divide by expected_clues.detach() so the forward pass
        produces stable magnitude, but the backward pass gradient still flows
        through the clue_usage weights in the numerator.
        
        Args:
            clue_features: Shape (B, K, D, H, W)
            attention_maps: Optional (B, K, H, W) attention weights
            stop_logits: Optional (B, K) stop probability logits
            
        Returns:
            aggregated: Shape (B, D, H, W)
        """
        B, K, D, H, W = clue_features.shape
        
        # Compute clue usage weights from stop_logits
        # (1 - stop_prob) = probability of using this clue
        if stop_logits is not None:
            stop_probs = torch.sigmoid(stop_logits)  # (B, K)
            clue_usage = 1 - stop_probs  # (B, K) - higher = more used
            
            # Compute expected clue count for normalization
            # DETACH so gradient flows through numerator only
            expected_clues = clue_usage.sum(dim=-1, keepdim=True).detach()  # (B, 1)
            expected_clues = expected_clues.view(B, 1, 1, 1, 1)  # (B, 1, 1, 1, 1)
            
            clue_usage = clue_usage.view(B, K, 1, 1, 1)  # (B, K, 1, 1, 1)
        else:
            clue_usage = torch.ones(B, K, 1, 1, 1, device=clue_features.device)
            expected_clues = torch.tensor(K, device=clue_features.device).view(1, 1, 1, 1, 1)
        
        if attention_maps is not None:
            # Weighted sum using attention maps AND clue usage
            attn_weights = attention_maps.unsqueeze(2)  # (B, K, 1, H, W)
            # Combine attention (spatial) with clue usage (per-clue weight)
            combined_weights = attn_weights * clue_usage  # (B, K, 1, H, W)
            weighted = clue_features * combined_weights
            aggregated = weighted.sum(dim=1)  # (B, D, H, W)
        else:
            # Weighted mean by clue usage only
            weighted = clue_features * clue_usage
            aggregated = weighted.sum(dim=1)  # (B, D, H, W)
        
        # Divide by expected clue count (DETACHED) to stabilize magnitude
        # This ensures output magnitude is ~constant regardless of clue count
        # while gradient still flows through clue_usage in the numerator
        aggregated = aggregated / (expected_clues.squeeze(1) + 1e-6)  # (B, D, H, W)
        
        aggregated = self.clue_aggregator(aggregated)
        
        return aggregated
    
    def _inject_counts(
        self,
        features: torch.Tensor,
        count_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Inject count information into spatial features.
        
        Supports two modes:
        1. Global counts: count_embedding shape (B, num_colors, D) - broadcast to all pixels
        2. Per-clue counts: count_embedding shape (B, K, D) - inject per-clue before aggregation
        
        Args:
            features: Shape (B, D, H, W) OR (B, K, D, H, W) for per-clue mode
            count_embedding: Shape (B, num_colors, D) for global OR (B, K, D) for per-clue
            
        Returns:
            enhanced: Same shape as input features
        """
        # Check if we have per-clue count embedding (B, K, D)
        # vs global count embedding (B, num_colors, D)
        if count_embedding.dim() == 3 and count_embedding.shape[2] == self.hidden_dim:
            # Per-clue count embedding (B, K, D)
            B, K_or_C, D_count = count_embedding.shape
            
            # If features is per-clue (B, K, D, H, W), inject per-clue counts
            if features.dim() == 5:
                # Per-clue injection: each clue k gets its own count embedding
                B, K, D, H, W = features.shape
                # Expand count_embedding to spatial dimensions
                count_spatial = count_embedding.view(B, K, D, 1, 1).expand(-1, -1, -1, H, W)
                enhanced = features + count_spatial
                return enhanced
            else:
                # Features is aggregated (B, D, H, W), sum per-clue counts
                count_sum = count_embedding.sum(dim=1)  # (B, D)
                B, D, H, W = features.shape
                count_spatial = count_sum.view(B, D, 1, 1).expand(-1, -1, H, W)
                enhanced = features + count_spatial
                return enhanced
        
        # Global count embedding (B, num_colors, D) - original behavior
        # Skip if LCR is disabled (count_proj is None)
        if self.count_proj is None:
            return features
            
        B, D, H, W = features.shape
        
        # Flatten count embeddings
        count_flat = count_embedding.view(B, -1)  # (B, num_colors * D)
        count_proj = self.count_proj(count_flat)  # (B, D)
        
        # Add to features (broadcast over spatial dims)
        count_spatial = count_proj.view(B, D, 1, 1).expand(-1, -1, H, W)
        enhanced = features + count_spatial
        
        return enhanced
    
    def _apply_predicate_gating(
        self,
        features: torch.Tensor,
        predicates: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply predicate-based gating if SPH is enabled.
        
        Args:
            features: Shape (B, D, H, W)
            predicates: Shape (B, P)
            
        Returns:
            gated: Shape (B, D, H, W)
        """
        # Skip if SPH is disabled (avoids wasted computation)
        if self.predicate_gate is None:
            return features
            
        return self.predicate_gate(features, predicates)
    
    def _solver_step_for_checkpoint(
        self,
        combined: torch.Tensor,
        h: torch.Tensor,
    ) -> torch.Tensor:
        """
        Minimal solver step for gradient checkpointing - NO optional args.
        
        IMPORTANT: torch.utils.checkpoint does NOT handle:
        - None values (causes silent gradient issues)
        - Dict arguments (not checkpointed correctly)
        
        So we only use this for the basic GRU step, without LoRA or cross-attention.
        Those are applied OUTSIDE the checkpointed region.
        
        Args:
            combined: (B, 2D, H, W) concatenated features
            h: Hidden state from previous step (or zeros if first step)
            
        Returns:
            h_new: Updated hidden state (B, D, H, W)
        """
        # GRU update WITHOUT LoRA (LoRA applied separately)
        h_new = self.gru(combined, h, lora_deltas=None)
        return h_new
    
    def _solver_step(
        self,
        combined: torch.Tensor,
        h: Optional[torch.Tensor],
        support_features: Optional[torch.Tensor],
        lora_deltas: Optional[Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """
        Full solver step with all optional features.
        Used when gradient checkpointing is disabled.
        
        Args:
            combined: (B, 2D, H, W) concatenated features
            h: Optional hidden state from previous step
            support_features: Optional support set for cross-attention
            lora_deltas: Optional HyperLoRA weight deltas
            
        Returns:
            h_new: Updated hidden state (B, D, H, W)
        """
        # GRU update with optional LoRA modulation
        h_new = self.gru(combined, h, lora_deltas=lora_deltas)
        
        # Phase 2.5: Cross-attention to support set at each step
        if self.solver_cross_attn is not None and support_features is not None:
            h_new = self.solver_cross_attn(h_new, support_features)
            
        return h_new
    
    def forward(
        self,
        clue_features: torch.Tensor,
        count_embedding: torch.Tensor,
        predicates: torch.Tensor,
        input_grid: torch.Tensor,
        attention_maps: Optional[torch.Tensor] = None,
        stop_logits: Optional[torch.Tensor] = None,
        support_features: Optional[torch.Tensor] = None,  # Phase 2.5: (B, N, D, H', W')
        lora_deltas: Optional[Dict[str, torch.Tensor]] = None,  # HyperLoRA weight deltas
        return_all_steps: bool = False,
        return_act_outputs: bool = False,
        num_steps_override: Optional[int] = None,  # Override num_steps at inference
    ) -> torch.Tensor:
        """
        Generate output through iterative refinement.
        
        Args:
            clue_features: Shape (B, K, D, H, W) from DSC + MSRE
            count_embedding: Shape (B, num_colors, D) from LCR global OR (B, K, D) from LCR per-clue
            predicates: Shape (B, P) from SPH (ignored if use_sph=False)
            input_grid: Shape (B, H, W) original input grid
            attention_maps: Optional (B, K, H, W) clue attention maps
            stop_logits: Optional (B, K) stop probability logits from DSC
                        Used to weight clue contributions - creates gradient flow
                        from task_loss to stop_predictor for latent clue count learning
            support_features: Optional (B, N, D, H', W') from ContextEncoder
                             Phase 2.5: Allows solver to attend to support set at each step
            lora_deltas: Optional Dict with HyperLoRA weight deltas:
                        - 'gru_reset': (B, D, D) reset gate modulation
                        - 'gru_update': (B, D, D) update gate modulation
                        - 'gru_candidate': (B, D, D) candidate modulation
                        - 'output_head': (B, D, D) output head modulation
            return_all_steps: If True, return predictions at all steps
            return_act_outputs: If True, also return ACT outputs for loss computation
            num_steps_override: If provided, use this many steps instead of self.num_steps.
                               Useful for inference-time experimentation with more/fewer steps.
            
        Returns:
            If return_all_steps:
                all_logits: List of (B, num_classes, H, W) for each step
            Else:
                logits: Shape (B, num_classes, H, W) final prediction
            If return_act_outputs (and use_act=True):
                Returns tuple: (logits_or_all_logits, act_outputs_dict)
        """
        B, K, D, H, W = clue_features.shape
        device = clue_features.device
        
        # Per-clue count injection (paper design: each clue has its local color stats)
        # If count_embedding is (B, K, D), inject into clue_features BEFORE aggregation
        # This ensures clue k's aggregated features include clue k's local counts
        if count_embedding.dim() == 3 and count_embedding.shape[1] == K:
            # Per-clue counts: inject into per-clue features
            clue_features = self._inject_counts(clue_features, count_embedding)
        
        # Aggregate clue features using attention AND stop_probs
        # This creates gradient flow: task_loss -> logits -> aggregated -> stop_probs -> stop_predictor
        # Making clue count a TRUE latent variable learned from target grids
        aggregated = self._aggregate_clues(clue_features, attention_maps, stop_logits)  # (B, D, H, W)
        
        # Inject global count information (if LCR returns global counts)
        # Only applies if count_embedding is (B, num_colors, D) format
        if count_embedding.dim() == 3 and count_embedding.shape[1] != K:
            aggregated = self._inject_counts(aggregated, count_embedding)
        
        # Apply predicate gating (skipped if use_sph=False)
        aggregated = self._apply_predicate_gating(aggregated, predicates)
        
        # Embed input grid for residual connections
        input_clamped = input_grid.clamp(0, 10).long()
        input_embed = self.input_embed(input_clamped)  # (B, H, W, D)
        input_embed = input_embed.permute(0, 3, 1, 2)  # (B, D, H, W)
        
        # Initialize hidden state
        h = None
        all_logits = []
        
        # ACT state initialization
        act_state = None
        act_outputs = {}
        if self.use_act:
            act_state = self.act_controller.init_state(B, device)
        
        # Iterative refinement
        # Store initial hidden state for residual connections
        h_initial = None
        
        # Allow overriding num_steps at inference (e.g., to test more iterations)
        effective_num_steps = num_steps_override if num_steps_override is not None else self.num_steps
        
        for t in range(effective_num_steps):
            # Check if all samples have halted (inference only)
            if self.use_act and not self.training and act_state.halted.all():
                break
                
            # Combine aggregated features with input embedding
            combined = torch.cat([aggregated, input_embed], dim=1)  # (B, 2D, H, W)
            combined = self.dropout(combined)
            
            # Core solver step - GRU + optional LoRA + optional cross-attention
            # Gradient checkpointing trades compute for memory by recomputing activations
            if self.gradient_checkpointing and self.training:
                # CRITICAL: Use minimal checkpoint function (no Dict/None args)
                # Initialize h to zeros if None (first step)
                if h is None:
                    B_size = combined.shape[0]
                    h_for_ckpt = torch.zeros(
                        B_size, self.hidden_dim, combined.shape[2], combined.shape[3],
                        device=combined.device, dtype=combined.dtype
                    )
                else:
                    h_for_ckpt = h
                
                # Checkpointed GRU step (no LoRA - applied separately)
                h_new = torch_checkpoint(
                    self._solver_step_for_checkpoint,
                    combined,
                    h_for_ckpt,
                    use_reentrant=False,
                )
                
                # Apply LoRA OUTSIDE checkpoint (Dict not supported by checkpoint)
                if lora_deltas is not None:
                    # Re-apply GRU with LoRA for correct weight modulation
                    # This is a compromise - we lose some memory savings but get correctness
                    h_new = self.gru(combined, h, lora_deltas=lora_deltas)
                
                # Apply cross-attention OUTSIDE checkpoint
                if self.solver_cross_attn is not None and support_features is not None:
                    h_new = self.solver_cross_attn(h_new, support_features)
            else:
                # Standard forward - all features included
                h_new = self._solver_step(combined, h, support_features, lora_deltas)
            
            # Store initial hidden state for residual connections
            if t == 0:
                h_initial = h_new.clone()
            elif h_initial is not None:
                # Add residual from initial state to prevent degradation
                # This helps later steps maintain the quality of step 0
                # while still allowing refinement
                # Increased from 0.1 to 0.3 to prevent solver degradation
                # Analysis showed step 0 had best loss, later steps regressed
                h_new = 0.7 * h_new + 0.3 * h_initial
            
            # Handle ACT updates
            if self.use_act:
                # Decide whether to halt based on new state
                halt_decision, step_outputs = self.act_controller.should_halt(h_new, act_state)
                
                # Update state (only for non-halted samples)
                # During training we run all steps but mask loss
                # During inference we can actually stop computation
                
                if self.training:
                    h = h_new  # Always update during training
                    
                    # Store ACT outputs for loss
                    for k, v in step_outputs.items():
                        if k not in act_outputs:
                            act_outputs[k] = []
                        act_outputs[k].append(v)
                        
                    # Update ACT state counters
                    act_state.steps += 1
                    act_state.halted = act_state.halted | halt_decision
                else:
                    # Inference: only update non-halted
                    active_mask = ~act_state.halted
                    if h is None:
                        h = h_new
                    else:
                        # Only update active samples
                        h = torch.where(active_mask.view(B, 1, 1, 1), h_new, h)
                    
                    act_state.steps += active_mask.long()
                    act_state.halted = act_state.halted | halt_decision
            else:
                h = h_new
            
            # Apply output_head LoRA modulation if provided (before output_head)
            # LoRA modulates the hidden state h, then output_head produces logits
            h_for_output = h
            if lora_deltas is not None and 'output_head' in lora_deltas:
                output_lora = lora_deltas['output_head']  # (B, D, D)
                # Apply LoRA to h: h' = h + h @ lora_delta (per-pixel)
                # h: (B, D, H, W) -> (B, H*W, D)
                B_size, D_size, H_size, W_size = h.shape
                h_flat = h.permute(0, 2, 3, 1).reshape(B_size, H_size * W_size, D_size)  # (B, H*W, D)
                delta = torch.bmm(h_flat, output_lora)  # (B, H*W, D)
                delta = delta.reshape(B_size, H_size, W_size, D_size).permute(0, 3, 1, 2)  # (B, D, H, W)
                h_for_output = h + delta
            
            # Predict output
            # Use soft_clamp_logits to prevent NaN from extreme values (10K+)
            # while still allowing natural growth (unlike ±50 clamp which caused collapse)
            # - Identity for |logits| <= 1000 (normal training range)
            # - Smooth compression to ±2000 max (prevents overflow)
            logits = soft_clamp_logits(self.output_head(h_for_output))  # (B, num_classes, H, W)
            
            all_logits.append(logits)
            
            # Optional: Use prediction to update input embedding (feedback)
            # CRITICAL FIX (Dec 2025): Removed Gumbel noise!
            # Use soft argmax (differentiable) instead of Gumbel-softmax
            # to maintain train/eval consistency.
            if self.use_feedback and t < effective_num_steps - 1:
                # Use soft argmax for differentiable feedback - NO Gumbel noise
                # This gives same behavior during training and inference
                soft_pred = F.softmax(logits / 1.0, dim=1)  # (B, C, H, W)
                # Weighted sum of embeddings: soft_pred @ embedding_weights
                # input_embed shape: (B, D, H, W)
                # embedding weights shape: (num_classes, D)
                emb_weights = self.input_embed.weight[:self.num_classes]  # (C, D)
                # Einsum: (B, C, H, W) @ (C, D) -> (B, D, H, W)
                input_embed = torch.einsum('bchw,cd->bdhw', soft_pred, emb_weights)
        
        # Prepare return value
        logits_output = all_logits if return_all_steps else all_logits[-1]
        
        # Return ACT outputs if requested (for computing ACT loss during training)
        if return_act_outputs and self.use_act and act_outputs:
            return logits_output, act_outputs
        else:
            return logits_output
    
    def compute_deep_supervision_loss(
        self,
        all_logits: List[torch.Tensor],
        targets: torch.Tensor,
        loss_fn: nn.Module,
        decay: float = 0.8,
    ) -> torch.Tensor:
        """
        Compute deep supervision loss over all refinement steps.
        
        Later steps are weighted more heavily.
        
        Args:
            all_logits: List of (B, num_classes, H, W) predictions
            targets: Shape (B, H, W) ground truth
            loss_fn: Loss function (e.g., CrossEntropyLoss)
            decay: Weight decay factor (weight_t = decay^(T-t))
            
        Returns:
            total_loss: Scalar weighted loss
        """
        T = len(all_logits)
        total_loss = 0.0
        total_weight = 0.0
        
        for t, logits in enumerate(all_logits):
            weight = decay ** (T - 1 - t)  # Later steps have higher weight
            step_loss = loss_fn(logits, targets)
            total_loss = total_loss + weight * step_loss
            total_weight += weight
        
        return total_loss / total_weight

    def select_best_step_by_loss(
        self,
        all_logits: List[torch.Tensor],
        targets: torch.Tensor,
        loss_fn: nn.Module,
    ) -> Tuple[torch.Tensor, int, List[float]]:
        """
        Select the best prediction step based on lowest loss (training mode).
        
        During training, we have ground truth so we can pick the step with
        lowest loss. This prevents over-iteration from hurting accuracy.
        
        Args:
            all_logits: List of (B, num_classes, H, W) predictions from each step
            targets: Shape (B, H, W) ground truth
            loss_fn: Loss function to evaluate each step. If RLANLoss, uses task_loss component.
            
        Returns:
            best_logits: The logits from the step with lowest loss
            best_step: Index of the best step (0-indexed)
            step_losses: List of loss values for each step
        """
        step_losses = []
        with torch.no_grad():
            # Use task_loss component if available (for RLANLoss), else use loss_fn directly
            task_loss_fn = getattr(loss_fn, 'task_loss', loss_fn)
            for logits in all_logits:
                if torch.isfinite(logits).all():
                    loss = task_loss_fn(logits, targets)
                    step_losses.append(loss.item())
                else:
                    step_losses.append(float('inf'))
        
        # Find step with minimum loss
        best_step = min(range(len(step_losses)), key=lambda i: step_losses[i])
        best_logits = all_logits[best_step]
        
        return best_logits, best_step, step_losses

    def select_best_step_by_entropy(
        self,
        all_logits: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, int, List[float]]:
        """
        Select the best prediction step based on lowest entropy (inference mode).
        
        During inference, we don't have ground truth. We use prediction entropy
        as a proxy for confidence - lower entropy = more confident = likely better.
        
        Args:
            all_logits: List of (B, num_classes, H, W) predictions from each step
            
        Returns:
            best_logits: The logits from the step with lowest entropy
            best_step: Index of the best step (0-indexed)
            step_entropies: List of mean entropy values for each step
        """
        step_entropies = []
        with torch.no_grad():
            for logits in all_logits:
                if torch.isfinite(logits).all():
                    # Compute per-pixel entropy, then average
                    probs = F.softmax(logits, dim=1)  # (B, C, H, W)
                    log_probs = F.log_softmax(logits, dim=1)
                    entropy = -(probs * log_probs).sum(dim=1)  # (B, H, W)
                    mean_entropy = entropy.mean().item()
                    step_entropies.append(mean_entropy)
                else:
                    step_entropies.append(float('inf'))
        
        # Find step with minimum entropy (most confident)
        best_step = min(range(len(step_entropies)), key=lambda i: step_entropies[i])
        best_logits = all_logits[best_step]
        
        return best_logits, best_step, step_entropies

    def select_best_step_combined(
        self,
        all_logits: List[torch.Tensor],
        targets: Optional[torch.Tensor] = None,
        loss_fn: Optional[nn.Module] = None,
    ) -> Tuple[torch.Tensor, int, dict]:
        """
        Unified best-step selection: uses loss if targets available, else entropy.
        
        This is the main API for best-step selection.
        
        Args:
            all_logits: List of predictions from each step
            targets: Ground truth (optional, for training)
            loss_fn: Loss function (optional, for training)
            
        Returns:
            best_logits: Best prediction
            best_step: Index of best step
            info: Dict with 'method', 'step_values', 'all_step_values'
                  During training, also includes 'entropy_best_step' and 'entropy_loss_agreement'
        """
        if targets is not None and loss_fn is not None:
            # Training mode: use actual loss
            best_logits, best_step, step_losses = self.select_best_step_by_loss(
                all_logits, targets, loss_fn
            )
            
            # Also compute entropy-based best step to track correlation
            _, entropy_best_step, step_entropies = self.select_best_step_by_entropy(all_logits)
            
            # Track if entropy would have picked the same step as loss
            agreement = (best_step == entropy_best_step)
            
            return best_logits, best_step, {
                'method': 'loss',
                'best_value': step_losses[best_step],
                'all_step_values': step_losses,
                'entropy_best_step': entropy_best_step,
                'entropy_step_values': step_entropies,
                'entropy_loss_agreement': agreement,  # True if entropy would pick same step
            }
        else:
            # Inference mode: use entropy
            best_logits, best_step, step_entropies = self.select_best_step_by_entropy(
                all_logits
            )
            return best_logits, best_step, {
                'method': 'entropy',
                'best_value': step_entropies[best_step],
                'all_step_values': step_entropies,
            }
