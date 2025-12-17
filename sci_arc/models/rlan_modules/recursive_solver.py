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

# Import SwiGLU for enhanced FFN
try:
    from .activations import SwiGLU, SwiGLUConv2d
    SWIGLU_AVAILABLE = True
except ImportError:
    SWIGLU_AVAILABLE = False


def _find_multiple(a: int, b: int) -> int:
    """Round up a to the nearest multiple of b."""
    return (-(a // -b)) * b


class ConvGRUCell(nn.Module):
    """
    Convolutional GRU cell for spatial state updates.
    
    Unlike standard GRU which operates on vectors, ConvGRU
    maintains spatial structure while performing recurrent updates.
    
    Enhanced with SwiGLU option for better gradient flow (TRM-style).
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
    
    def forward(
        self,
        x: torch.Tensor,
        h: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Single GRU step.
        
        Args:
            x: Input features (B, input_dim, H, W)
            h: Previous hidden state (B, hidden_dim, H, W) or None
            
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
        r = torch.sigmoid(self.reset_gate(combined))
        z = torch.sigmoid(self.update_gate(combined))
        
        # Compute candidate
        combined_reset = torch.cat([x, r * h], dim=1)
        
        if self.use_swiglu:
            # SwiGLU path
            proj = self.candidate_proj(combined_reset)
            h_candidate = self.candidate(proj)
            # Bound the candidate to prevent drift in later steps
            # tanh bounds to [-1, 1] which helps stability
            h_candidate = torch.tanh(h_candidate)
        else:
            # Standard Tanh path
            h_candidate = torch.tanh(self.candidate(combined_reset))
        
        # Update hidden state with residual connection for stability
        h_new = (1 - z) * h + z * h_candidate
        
        # Clamp to prevent extreme values that cause loss=100
        h_new = h_new.clamp(-10, 10)
        
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


class RecursiveSolver(nn.Module):
    """
    Recursive Solver - iterative refinement decoder for RLAN.
    
    Generates output predictions by iteratively refining a hidden state,
    conditioned on clue features, count embeddings, and predicates.
    
    Now supports Adaptive Computation Time (ACT) for variable steps.
    
    ABLATION SUPPORT:
    - When use_lcr=False, count injection is skipped (not just zero input)
    - When use_sph=False, predicate gating is skipped
    - This avoids wasted computation and gradient noise from unused modules
    """
    
    def __init__(
        self,
        hidden_dim: int = 128,
        num_classes: int = 11,
        num_steps: int = 6,
        num_predicates: int = 8,
        num_colors: int = 10,
        dropout: float = 0.1,
        use_act: bool = False,  # Enable Adaptive Computation Time
        use_lcr: bool = True,   # Enable count injection (ablation flag)
        use_sph: bool = True,   # Enable predicate gating (ablation flag)
        use_feedback: bool = False,  # Use prediction feedback (disabled by default - causes gradient issues)
    ):
        """
        Args:
            hidden_dim: Feature dimension
            num_classes: Output classes (10 colors + background)
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
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_steps = num_steps
        self.use_act = use_act
        self.use_lcr = use_lcr
        self.use_sph = use_sph
        self.use_feedback = use_feedback
        
        # Clue feature aggregation
        self.clue_aggregator = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            nn.GELU(),
            nn.GroupNorm(8, hidden_dim),
        )
        
        # Count embedding projection (only if LCR enabled)
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
        
        self.dropout = nn.Dropout(dropout)
    
    def _init_output_head_for_balanced_predictions(self):
        """
        Initialize output head to prevent background collapse.
        
        The problem: With 90% background pixels, the model learns to predict
        background everywhere as a "safe" default. Even with weighted loss,
        the initial random predictions favor background due to class imbalance.
        
        Solution: Initialize the final layer bias to give foreground classes
        a small advantage, encouraging the model to explore foreground predictions
        early in training.
        
        This is similar to how object detection models initialize the classification
        head with a prior probability for the positive class.
        """
        # Get the final Conv2d layer in output_head
        final_layer = None
        for module in self.output_head.modules():
            if isinstance(module, nn.Conv2d) and module.out_channels == self.num_classes:
                final_layer = module
        
        if final_layer is not None and final_layer.bias is not None:
            # Initialize bias for balanced starting point
            # 
            # MATH: For softmax with C=10 classes, to get P(bg) = 50%:
            #   P(bg) = exp(bg_bias) / (exp(bg_bias) + 9*exp(fg_bias))
            #   0.5 = exp(bg_bias) / (exp(bg_bias) + 9*exp(0))
            #   0.5 * (exp(bg_bias) + 9) = exp(bg_bias)
            #   4.5 = 0.5 * exp(bg_bias)
            #   bg_bias = ln(9) ≈ 2.2
            #
            # This creates 50-50 start, letting the weighted loss (10x for FG)
            # guide learning without initial bias toward either direction.
            with torch.no_grad():
                import math
                # ln(9) gives P(bg)=50%, P(all_fg)=50% when fg_bias=0
                final_layer.bias[0] = math.log(self.num_classes - 1)  # ≈2.2 for C=10
                final_layer.bias[1:] = 0.0  # Uniform fg classes
    
    def _aggregate_clues(
        self,
        clue_features: torch.Tensor,
        attention_maps: Optional[torch.Tensor] = None,
        stop_logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Aggregate features from multiple clues.
        
        CRITICAL: Uses stop_probs to weight clue contributions WITHOUT normalizing.
        This creates direct gradient flow from task_loss to stop_predictor, making
        the clue count a TRUE latent variable learned from target grids.
        
        The key insight: if we normalize clue_usage to sum to 1, the output is
        the same regardless of how many clues are "used" (only relative weights
        matter). By NOT normalizing, the output magnitude scales with the fraction
        of clues used, giving the solver information about clue count and providing
        gradient signal for learning the optimal number of clues per sample.
        
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
            # DO NOT NORMALIZE! Divide by constant K instead.
            # This preserves clue count information in the aggregation:
            # - Using all K clues: output = mean of all clue_features
            # - Using 1 clue: output = 1/K * that clue's features (smaller magnitude)
            # The solver learns to handle varying magnitudes based on clue count.
            clue_usage = clue_usage.view(B, K, 1, 1, 1)  # (B, K, 1, 1, 1)
        else:
            clue_usage = torch.ones(B, K, 1, 1, 1, device=clue_features.device)
        
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
        
        # Divide by K (constant) to keep magnitude stable across different K values
        aggregated = aggregated / K
        
        aggregated = self.clue_aggregator(aggregated)
        
        return aggregated
    
    def _inject_counts(
        self,
        features: torch.Tensor,
        count_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Inject count information into spatial features.
        
        Args:
            features: Shape (B, D, H, W)
            count_embedding: Shape (B, num_colors, D)
            
        Returns:
            enhanced: Shape (B, D, H, W)
        """
        # Skip if LCR is disabled (avoids wasted computation)
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
    
    def forward(
        self,
        clue_features: torch.Tensor,
        count_embedding: torch.Tensor,
        predicates: torch.Tensor,
        input_grid: torch.Tensor,
        attention_maps: Optional[torch.Tensor] = None,
        stop_logits: Optional[torch.Tensor] = None,
        return_all_steps: bool = False,
        return_act_outputs: bool = False,
    ) -> torch.Tensor:
        """
        Generate output through iterative refinement.
        
        Args:
            clue_features: Shape (B, K, D, H, W) from DSC + MSRE
            count_embedding: Shape (B, num_colors, D) from LCR (ignored if use_lcr=False)
            predicates: Shape (B, P) from SPH (ignored if use_sph=False)
            input_grid: Shape (B, H, W) original input grid
            attention_maps: Optional (B, K, H, W) clue attention maps
            stop_logits: Optional (B, K) stop probability logits from DSC
                        Used to weight clue contributions - creates gradient flow
                        from task_loss to stop_predictor for latent clue count learning
            return_all_steps: If True, return predictions at all steps
            return_act_outputs: If True, also return ACT outputs for loss computation
            
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
        
        # Aggregate clue features using attention AND stop_probs
        # This creates gradient flow: task_loss -> logits -> aggregated -> stop_probs -> stop_predictor
        # Making clue count a TRUE latent variable learned from target grids
        aggregated = self._aggregate_clues(clue_features, attention_maps, stop_logits)  # (B, D, H, W)
        
        # Inject count information (skipped if use_lcr=False)
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
        
        for t in range(self.num_steps):
            # Check if all samples have halted (inference only)
            if self.use_act and not self.training and act_state.halted.all():
                break
                
            # Combine aggregated features with input embedding
            combined = torch.cat([aggregated, input_embed], dim=1)  # (B, 2D, H, W)
            combined = self.dropout(combined)
            
            # GRU update
            h_new = self.gru(combined, h)
            
            # Store initial hidden state for residual connections
            if t == 0:
                h_initial = h_new.clone()
            elif h_initial is not None:
                # Add residual from initial state to prevent degradation
                # This helps later steps maintain the quality of step 0
                # while still allowing refinement
                # Weighted residual: 0.1 * h_initial + 0.9 * h_new
                h_new = 0.9 * h_new + 0.1 * h_initial
            
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
            
            # Predict output
            logits = self.output_head(h)  # (B, num_classes, H, W)
            all_logits.append(logits)
            
            # Optional: Use prediction to update input embedding (feedback)
            # When enabled, uses DIFFERENTIABLE soft feedback during training
            # and hard argmax during inference.
            if self.use_feedback and t < self.num_steps - 1:
                if self.training:
                    # TRAINING: Use Gumbel-Softmax for differentiable discrete samples
                    # This allows gradients to flow through the feedback loop
                    # tau=1.0 gives soft samples; lower tau → sharper (more discrete-like)
                    soft_pred = F.gumbel_softmax(logits, tau=1.0, hard=False, dim=1)  # (B, C, H, W)
                    # Weighted sum of embeddings: soft_pred @ embedding_weights
                    # input_embed shape: (B, D, H, W)
                    # embedding weights shape: (num_classes, D)
                    emb_weights = self.input_embed.weight[:self.num_classes]  # (C, D)
                    # Einsum: (B, C, H, W) @ (C, D) -> (B, D, H, W)
                    input_embed = torch.einsum('bchw,cd->bdhw', soft_pred, emb_weights)
                else:
                    # INFERENCE: Use hard argmax (non-differentiable, but that's OK)
                    pred = logits.argmax(dim=1)  # (B, H, W)
                    input_embed = self.input_embed(pred.clamp(0, 10)).permute(0, 3, 1, 2)
        
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
