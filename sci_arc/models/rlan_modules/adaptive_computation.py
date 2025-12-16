"""
Adaptive Computation Time (ACT) for RLAN

Implements adaptive halting based on TRM's Q-learning approach.
Allows the model to decide when it has done "enough" reasoning
for a given sample, enabling:
- Faster inference on easy samples
- More compute on hard samples
- Learned optimal reasoning depth

Key Concepts:
- Halt Head: Predicts Q-value for halting vs continuing
- Exploration: ε-greedy during training for diverse step counts
- Target Q: Bootstrapped from future predictions

Reference:
- Adaptive Computation Time for Recurrent Neural Networks (Graves, 2016)
- TinyRecursiveModels (ACT V1 implementation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class ACTState:
    """State for Adaptive Computation Time."""
    steps: torch.Tensor          # (B,) number of steps taken
    halted: torch.Tensor         # (B,) whether sample has halted
    accumulated_prob: torch.Tensor  # (B,) accumulated halt probability
    hidden: Optional[torch.Tensor]  # Hidden state for carry-over


class AdaptiveHaltHead(nn.Module):
    """
    Halt prediction head using Q-learning.
    
    Predicts whether halting now gives higher reward than continuing.
    Uses the first spatial position (or global pool) for prediction.
    
    Args:
        hidden_dim: Input feature dimension
        use_q_continue: If True, also predict Q-value for continuing
                       (TRM's original design). If False, use simpler
                       sigmoid halt probability.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        use_q_continue: bool = False,
    ):
        super().__init__()
        
        self.use_q_continue = use_q_continue
        
        # Pool spatial features to single vector
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # Q-head outputs: [Q_halt, Q_continue] or just [halt_logit]
        output_dim = 2 if use_q_continue else 1
        self.q_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )
        
        # Initialize to prefer not halting early (bias towards exploration)
        self._init_weights()
    
    def _init_weights(self):
        """Initialize Q-head to output low values initially."""
        # Makes model explore more steps before learning to halt
        with torch.no_grad():
            self.q_head[-1].weight.zero_()
            self.q_head[-1].bias.fill_(-5.0)
    
    def forward(
        self, 
        features: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Predict halt Q-values.
        
        Args:
            features: Shape (B, D, H, W) spatial features
            
        Returns:
            q_halt: (B,) Q-value for halting
            q_continue: (B,) Q-value for continuing (if use_q_continue)
        """
        # Global pool: (B, D, H, W) -> (B, D)
        pooled = self.pool(features).squeeze(-1).squeeze(-1)
        
        # Predict Q-values
        q = self.q_head(pooled)  # (B, 1) or (B, 2)
        
        if self.use_q_continue:
            return q[:, 0], q[:, 1]
        else:
            return q.squeeze(-1), None


class ACTController(nn.Module):
    """
    Adaptive Computation Time Controller.
    
    Wraps the RecursiveSolver to add adaptive halting.
    During training, uses ε-greedy exploration.
    During inference, can either:
    - Use max steps (for deterministic batched inference)
    - Use learned halting (for efficiency)
    
    Args:
        hidden_dim: Feature dimension
        max_steps: Maximum reasoning steps
        exploration_prob: Probability of random exploration during training
        use_q_continue: Whether to use Q-learning for continue action
    """
    
    def __init__(
        self,
        hidden_dim: int,
        max_steps: int = 16,
        exploration_prob: float = 0.1,
        use_q_continue: bool = False,
    ):
        super().__init__()
        
        self.max_steps = max_steps
        self.exploration_prob = exploration_prob
        self.use_q_continue = use_q_continue
        
        self.halt_head = AdaptiveHaltHead(
            hidden_dim=hidden_dim,
            use_q_continue=use_q_continue,
        )
    
    def init_state(self, batch_size: int, device: torch.device) -> ACTState:
        """Initialize ACT state for new batch."""
        return ACTState(
            steps=torch.zeros(batch_size, dtype=torch.int32, device=device),
            halted=torch.zeros(batch_size, dtype=torch.bool, device=device),
            accumulated_prob=torch.zeros(batch_size, device=device),
            hidden=None,
        )
    
    def should_halt(
        self,
        features: torch.Tensor,
        state: ACTState,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Decide whether to halt based on current features.
        
        Args:
            features: (B, D, H, W) current feature state
            state: Current ACT state
            
        Returns:
            halt_decision: (B,) boolean tensor of which samples to halt
            outputs: Dict with q_halt, q_continue for loss computation
        """
        B = features.shape[0]
        device = features.device
        
        # Predict Q-values
        q_halt, q_continue = self.halt_head(features)
        
        outputs = {"q_halt_logits": q_halt}
        if q_continue is not None:
            outputs["q_continue_logits"] = q_continue
        
        # Check if at max steps
        is_last_step = (state.steps + 1) >= self.max_steps
        
        # Determine halt decision
        if self.training:
            # During training: Q-learning decision with exploration
            if self.use_q_continue:
                halt_decision = is_last_step | (q_halt > q_continue)
            else:
                halt_decision = is_last_step | (q_halt > 0)
            
            # Exploration: random minimum steps
            if self.exploration_prob > 0:
                explore_mask = torch.rand(B, device=device) < self.exploration_prob
                min_steps = torch.randint(2, self.max_steps + 1, (B,), device=device)
                halt_decision = halt_decision & ((state.steps + 1) >= min_steps) | ~explore_mask
        else:
            # During inference: use max steps for deterministic batching
            # (or could use learned halting for efficiency)
            halt_decision = is_last_step
        
        return halt_decision, outputs
    
    def compute_halt_loss(
        self,
        q_halt_logits: torch.Tensor,
        is_correct: torch.Tensor,
        q_continue_logits: Optional[torch.Tensor] = None,
        target_q_continue: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute ACT losses for training.
        
        Args:
            q_halt_logits: (B,) predicted Q for halting
            is_correct: (B,) whether prediction was correct (reward signal)
            q_continue_logits: (B,) predicted Q for continuing
            target_q_continue: (B,) bootstrapped target for continue Q
            
        Returns:
            losses: Dict with halt_loss and optionally continue_loss
        """
        losses = {}
        
        # Q_halt loss: predict whether halting leads to correct answer
        halt_loss = F.binary_cross_entropy_with_logits(
            q_halt_logits, 
            is_correct.float(),
            reduction='mean'
        )
        losses['halt_loss'] = halt_loss
        
        # Q_continue loss: bootstrapped from next step's max Q
        if q_continue_logits is not None and target_q_continue is not None:
            continue_loss = F.binary_cross_entropy_with_logits(
                q_continue_logits,
                target_q_continue,
                reduction='mean'
            )
            losses['continue_loss'] = continue_loss
        
        return losses


class PonderingCost(nn.Module):
    """
    Alternative to Q-learning: Pondering cost regularization.
    
    Adds a cost for each step taken, encouraging the model
    to halt as soon as it has a confident prediction.
    Simpler than Q-learning but less flexible.
    
    Args:
        cost_per_step: Base cost per reasoning step
    """
    
    def __init__(self, cost_per_step: float = 0.01):
        super().__init__()
        self.cost_per_step = cost_per_step
    
    def forward(self, steps: torch.Tensor) -> torch.Tensor:
        """
        Compute pondering cost.
        
        Args:
            steps: (B,) number of steps taken per sample
            
        Returns:
            cost: Scalar pondering cost
        """
        return self.cost_per_step * steps.float().mean()
