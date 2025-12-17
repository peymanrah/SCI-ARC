"""
RLAN Loss Functions

Comprehensive loss module for training RLAN on ARC tasks.

Loss Components:
1. Focal Loss - Primary task loss with class imbalance handling
2. Stablemax Loss - Numerically stable alternative (from TRM)
3. Entropy Regularization - Encourage sharp attention maps
4. Sparsity Regularization - Encourage stopping early when possible
5. Predicate Diversity - Decorrelate predicate activations
6. Curriculum Penalty - Progressive complexity scheduling

The combined loss enables stable training while encouraging:
- Sharp, interpretable attention patterns
- Efficient use of clues (don't use more than needed)
- Diverse predicate representations
- Gradual increase in model capacity usage
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def stablemax(x: torch.Tensor, epsilon: float = 1e-30) -> torch.Tensor:
    """
    Stablemax activation function (from TRM).
    
    More numerically stable than softmax for extreme values.
    s(x) = 1/(1-x) for x < 0, else x + 1
    
    Args:
        x: Input tensor
        epsilon: Small constant for stability
        
    Returns:
        Stablemax activations
    """
    return torch.where(
        x < 0,
        1 / (1 - x + epsilon),
        x + 1
    )


def log_stablemax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Log stablemax for stable cross entropy computation.
    
    Args:
        x: Input logits
        dim: Dimension to normalize over
        
    Returns:
        Log probabilities using stablemax normalization
    """
    s_x = stablemax(x)
    return torch.log(s_x / torch.sum(s_x, dim=dim, keepdim=True))


class StablemaxCrossEntropy(nn.Module):
    """
    Stablemax Cross Entropy Loss (from TRM).
    
    Uses stablemax instead of softmax for better numerical stability.
    Particularly useful when logits can have extreme values.
    
    Args:
        reduction: 'mean', 'sum', or 'none'
        ignore_index: Label to ignore
    """
    
    def __init__(
        self,
        reduction: str = "mean",
        ignore_index: int = -100,
    ):
        super().__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute stablemax cross entropy loss.
        
        Args:
            logits: Shape (B, C, H, W) unnormalized logits
            targets: Shape (B, H, W) target class indices
            
        Returns:
            loss: Scalar or per-pixel loss
        """
        B, C, H, W = logits.shape
        
        # Flatten for loss computation
        logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)
        targets_flat = targets.reshape(-1)  # (B*H*W,)
        
        # Create mask for valid positions
        valid_mask = targets_flat != self.ignore_index
        
        if not valid_mask.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        # Filter to valid positions
        logits_valid = logits_flat[valid_mask]
        targets_valid = targets_flat[valid_mask]
        
        # Compute log stablemax probabilities (use float64 for precision)
        logprobs = log_stablemax(logits_valid.to(torch.float64), dim=-1)
        
        # Gather log probs for target classes
        prediction_logprobs = torch.gather(
            logprobs,
            index=targets_valid.unsqueeze(-1).to(torch.long),
            dim=-1
        ).squeeze(-1)
        
        # Negative log likelihood
        loss = -prediction_logprobs.to(logits.dtype)
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            full_loss = torch.zeros(B * H * W, device=logits.device)
            full_loss[valid_mask] = loss
            return full_loss.view(B, H, W)


class FocalStablemaxLoss(nn.Module):
    """
    Focal Loss with Stablemax denominator.
    
    Combines the class-balancing benefits of Focal Loss
    with the numerical stability of Stablemax.
    
    Best of both worlds for ARC training.
    
    Args:
        gamma: Focusing parameter
        alpha: Class weight for foreground
        reduction: 'mean', 'sum', or 'none'
        ignore_index: Label to ignore
    """
    
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.25,
        reduction: str = "mean",
        ignore_index: int = -100,
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.ignore_index = ignore_index
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute focal stablemax loss.
        """
        B, C, H, W = logits.shape
        
        logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, C)
        targets_flat = targets.reshape(-1)
        
        valid_mask = targets_flat != self.ignore_index
        
        if not valid_mask.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        logits_valid = logits_flat[valid_mask]
        targets_valid = targets_flat[valid_mask]
        
        # Use stablemax for probabilities
        s_x = stablemax(logits_valid.to(torch.float64))
        probs = s_x / s_x.sum(dim=-1, keepdim=True)
        probs = probs.to(logits.dtype)
        
        # Get target probabilities
        target_probs = probs.gather(1, targets_valid.unsqueeze(1)).squeeze(1)
        target_probs = target_probs.clamp(min=1e-7, max=1.0 - 1e-7)
        
        # Focal weight: down-weight easy examples (high confidence)
        # gamma=2.0 means: p=0.9 (easy) -> weight=0.01, p=0.5 (hard) -> weight=0.25
        focal_weight = (1 - target_probs) ** self.gamma
        
        # Alpha weight: CRITICAL for anti-background-collapse
        # Background (class 0) should have LOW weight, foreground (1-9) should have HIGH weight
        # With alpha=0.75: background=0.25, foreground=0.75 (3x more weight on foreground)
        # This forces the model to care more about getting foreground pixels right
        is_background = (targets_valid == 0).float()
        alpha_weight = is_background * (1 - self.alpha) + (1 - is_background) * self.alpha
        
        # Cross entropy using log stablemax
        logprobs = log_stablemax(logits_valid.to(torch.float64), dim=-1).to(logits.dtype)
        ce_loss = -torch.gather(logprobs, 1, targets_valid.unsqueeze(1)).squeeze(1)
        ce_loss = ce_loss.clamp(max=100.0)
        
        # Combined focal loss
        focal_loss = alpha_weight * focal_weight * ce_loss
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            full_loss = torch.zeros(B * H * W, device=logits.device)
            full_loss[valid_mask] = focal_loss
            return full_loss.view(B, H, W)


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in ARC grids.
    
    ARC grids are ~85% background (color 0), leading to trivial
    all-background predictions. Focal Loss down-weights easy examples
    (confident background predictions) to focus on hard examples.
    
    Formula: L_focal = -α(1-p)^γ log(p)
    
    Parameters:
        gamma: Focusing parameter (higher = more focus on hard examples)
        alpha: Class weight for positive class (lower for background)
    """
    
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.25,
        reduction: str = "mean",
        ignore_index: int = -100,
    ):
        """
        Args:
            gamma: Focusing parameter (typically 2.0)
            alpha: Weight for positive/foreground classes
            reduction: 'mean', 'sum', or 'none'
            ignore_index: Label to ignore in loss computation
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.ignore_index = ignore_index
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            logits: Shape (B, C, H, W) unnormalized logits
            targets: Shape (B, H, W) target class indices
            
        Returns:
            loss: Scalar or per-pixel loss depending on reduction
        """
        B, C, H, W = logits.shape
        
        # Flatten for loss computation
        logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)
        targets_flat = targets.reshape(-1)  # (B*H*W,)
        
        # Create mask for valid positions
        valid_mask = targets_flat != self.ignore_index
        
        if not valid_mask.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        # Filter to valid positions
        logits_valid = logits_flat[valid_mask]
        targets_valid = targets_flat[valid_mask]
        
        # Compute softmax probabilities
        probs = F.softmax(logits_valid, dim=-1)
        
        # Get probability of correct class (clamp for numerical stability)
        target_probs = probs.gather(1, targets_valid.unsqueeze(1)).squeeze(1)
        target_probs = target_probs.clamp(min=1e-7, max=1.0 - 1e-7)
        
        # Compute focal weight: (1 - p)^gamma
        focal_weight = (1 - target_probs) ** self.gamma
        
        # Compute alpha weight (lower for background class 0)
        is_background = (targets_valid == 0).float()
        alpha_weight = is_background * (1 - self.alpha) + (1 - is_background) * self.alpha
        
        # Compute cross entropy with label smoothing for stability
        ce_loss = F.cross_entropy(logits_valid, targets_valid, reduction='none')
        
        # Clamp CE loss to prevent extreme values
        ce_loss = ce_loss.clamp(max=100.0)
        
        # Apply focal and alpha weights
        focal_loss = alpha_weight * focal_weight * ce_loss
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            # Reshape back to (B, H, W) with zeros for invalid positions
            full_loss = torch.zeros(B * H * W, device=logits.device)
            full_loss[valid_mask] = focal_loss
            return full_loss.view(B, H, W)


class EntropyRegularization(nn.Module):
    """
    Entropy regularization for attention maps.
    
    Encourages sharp, focused attention rather than diffuse attention.
    Low entropy = sharp (good), High entropy = diffuse (bad)
    
    Formula: L_entropy = Σ_k H(attention_k) where H(p) = -Σ p log(p)
    """
    
    def __init__(self, target_entropy: float = 0.0):
        """
        Args:
            target_entropy: Target entropy value (0 = perfectly sharp)
        """
        super().__init__()
        self.target_entropy = target_entropy
    
    def forward(self, attention_maps: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy regularization loss.
        
        Args:
            attention_maps: Shape (B, K, H, W) attention weights
            
        Returns:
            loss: Scalar entropy loss
        """
        B, K, H, W = attention_maps.shape
        
        # Flatten spatial dimensions
        attention_flat = attention_maps.view(B, K, -1)  # (B, K, H*W)
        
        # Clamp attention for numerical stability
        attention_clamped = attention_flat.clamp(min=1e-10, max=1.0)
        
        # Compute entropy for each attention map
        # H(p) = -Σ p log(p)
        log_attention = torch.log(attention_clamped)
        entropy = -(attention_clamped * log_attention).sum(dim=-1)  # (B, K)
        
        # Average over clues and batch
        mean_entropy = entropy.mean()
        
        # Loss is distance from target entropy
        loss = (mean_entropy - self.target_entropy).abs()
        
        # Clamp to prevent extreme values
        loss = loss.clamp(max=10.0)
        
        return loss


class SparsityRegularization(nn.Module):
    """
    Sparsity regularization for clue usage.
    
    Encourages the model to use fewer clues when possible.
    Rewards high stop probabilities (stopping early).
    
    Formula: L_sparsity = -Σ_k log(p_stop_k + ε)
    """
    
    def __init__(self, epsilon: float = 1e-6):
        """
        Args:
            epsilon: Small constant for numerical stability
        """
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, stop_logits: torch.Tensor) -> torch.Tensor:
        """
        Compute sparsity regularization loss.
        
        Args:
            stop_logits: Shape (B, K) stop probability logits
            
        Returns:
            loss: Scalar sparsity loss
        """
        # Convert to probabilities
        stop_probs = torch.sigmoid(stop_logits)  # (B, K)
        
        # Encourage high stop probabilities (stopping early)
        # -log(p_stop) is minimized when p_stop is high
        # Clamp to prevent numerical instability
        sparsity_loss = -torch.log(stop_probs.clamp(min=self.epsilon, max=1.0)).mean()
        
        # Clamp final loss to reasonable range
        sparsity_loss = sparsity_loss.clamp(max=10.0)
        
        return sparsity_loss


class PredicateDiversityLoss(nn.Module):
    """
    Predicate diversity loss to decorrelate predicates.
    
    Encourages different predicates to capture different properties
    by minimizing their correlation across the batch.
    
    Formula: L_pred = ||P^T P - I||_F^2 / P^2
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, predicates: torch.Tensor) -> torch.Tensor:
        """
        Compute predicate diversity loss.
        
        Args:
            predicates: Shape (B, P) predicate activations
            
        Returns:
            loss: Scalar diversity loss
        """
        B, P = predicates.shape
        
        if B < 2:
            return torch.tensor(0.0, device=predicates.device)
        
        # Center predicates
        predicates_centered = predicates - predicates.mean(dim=0, keepdim=True)
        
        # Compute correlation matrix
        corr = torch.mm(predicates_centered.T, predicates_centered) / (B - 1)
        
        # Normalize by variance
        var = torch.diag(corr) + 1e-6
        std = torch.sqrt(var)
        corr_normalized = corr / (std.unsqueeze(0) * std.unsqueeze(1))
        
        # Off-diagonal elements should be zero
        identity = torch.eye(P, device=predicates.device)
        off_diagonal = corr_normalized * (1 - identity)
        
        # Frobenius norm of off-diagonal
        diversity_loss = (off_diagonal ** 2).sum() / (P * (P - 1) + 1e-6)
        
        # Check for NaN and clamp
        if torch.isnan(diversity_loss):
            return torch.tensor(0.0, device=predicates.device, requires_grad=True)
        
        return diversity_loss.clamp(max=10.0)


class CurriculumPenalty(nn.Module):
    """
    Curriculum-based complexity penalty.
    
    Early in training: Penalize using many clues (force simple solutions)
    Late in training: Allow full complexity
    
    This helps the model learn simple patterns first before complex ones.
    """
    
    def __init__(self, max_clues: int = 5):
        """
        Args:
            max_clues: Maximum number of clues (K)
        """
        super().__init__()
        self.max_clues = max_clues
    
    def forward(
        self,
        stop_logits: torch.Tensor,
        epoch: int,
        max_epochs: int,
    ) -> torch.Tensor:
        """
        Compute curriculum penalty.
        
        Args:
            stop_logits: Shape (B, K) stop probability logits
            epoch: Current epoch number
            max_epochs: Total number of epochs
            
        Returns:
            loss: Scalar curriculum penalty
        """
        B, K = stop_logits.shape
        
        # Compute expected number of active clues
        stop_probs = torch.sigmoid(stop_logits)
        continue_probs = 1 - stop_probs
        
        # Probability of reaching clue k = product of (1 - stop_prob) for all previous
        reach_probs = torch.cumprod(
            torch.cat([torch.ones(B, 1, device=stop_logits.device), continue_probs[:, :-1]], dim=1),
            dim=1
        )
        expected_clues = reach_probs.sum(dim=1).mean()  # Average over batch
        
        # Curriculum schedule: early training penalizes more clues
        # progress goes from 0 to 1 over training
        progress = epoch / max(max_epochs, 1)
        
        # Target number of clues increases over training
        # Start with 1-2 clues, end with full capacity
        target_clues = 1.0 + progress * (self.max_clues - 1)
        
        # Penalty for exceeding target
        excess = F.relu(expected_clues - target_clues)
        
        return excess


class RLANLoss(nn.Module):
    """
    Combined loss for RLAN training.
    
    Combines all loss components with configurable weights:
    - Task Loss: StablemaxCrossEntropy (TRM-style) or FocalStablemaxLoss
    - Entropy Regularization: Sharp attention
    - Sparsity Regularization: Efficient clue usage
    - Predicate Diversity: Decorrelated predicates
    - Curriculum Penalty: Progressive complexity
    
    Loss Mode Options:
    - 'stablemax': Pure stablemax cross-entropy (TRM uses this)
    - 'focal_stablemax': Focal loss + stablemax (original RLAN)
    - 'focal': Standard focal loss with softmax
    """
    
    def __init__(
        self,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        lambda_entropy: float = 0.1,
        lambda_sparsity: float = 0.05,
        lambda_predicate: float = 0.01,
        lambda_curriculum: float = 0.1,
        lambda_deep_supervision: float = 0.5,
        lambda_act: float = 0.1,  # Weight for ACT pondering cost
        max_clues: int = 5,
        use_stablemax: bool = True,  # TRM uses stablemax for numerical stability
        loss_mode: str = 'focal_stablemax',  # 'stablemax', 'focal_stablemax', or 'focal'
    ):
        """
        Args:
            focal_gamma: Focal loss gamma parameter
            focal_alpha: Focal loss alpha parameter
            lambda_entropy: Weight for entropy regularization
            lambda_sparsity: Weight for sparsity regularization
            lambda_predicate: Weight for predicate diversity
            lambda_curriculum: Weight for curriculum penalty
            lambda_deep_supervision: Weight for intermediate step losses
            lambda_act: Weight for ACT halting loss (pondering cost)
            max_clues: Maximum number of clues
            use_stablemax: DEPRECATED - use loss_mode instead
            loss_mode: Loss function mode:
                - 'stablemax': Pure stablemax CE (TRM uses this - RECOMMENDED)
                - 'focal_stablemax': Focal loss + stablemax
                - 'focal': Standard focal loss
        """
        super().__init__()
        
        self.loss_mode = loss_mode
        
        # Select loss function based on mode
        if loss_mode == 'stablemax':
            # Pure stablemax cross-entropy (TRM style - RECOMMENDED)
            self.task_loss = StablemaxCrossEntropy()
            print(f"  Loss Mode: STABLEMAX (TRM-style, pure cross-entropy)")
        elif loss_mode == 'focal_stablemax':
            # Focal loss with stablemax (original RLAN)
            self.task_loss = FocalStablemaxLoss(gamma=focal_gamma, alpha=focal_alpha)
            print(f"  Loss Mode: FOCAL_STABLEMAX (gamma={focal_gamma}, alpha={focal_alpha})")
        elif loss_mode == 'focal':
            # Standard focal loss with softmax
            self.task_loss = FocalLoss(gamma=focal_gamma, alpha=focal_alpha)
            print(f"  Loss Mode: FOCAL (gamma={focal_gamma}, alpha={focal_alpha})")
        else:
            # Fallback to use_stablemax for backward compatibility
            if use_stablemax:
                self.task_loss = FocalStablemaxLoss(gamma=focal_gamma, alpha=focal_alpha)
            else:
                self.task_loss = FocalLoss(gamma=focal_gamma, alpha=focal_alpha)
        
        # Keep focal_loss as alias for backward compatibility
        self.focal_loss = self.task_loss
        
        self.entropy_reg = EntropyRegularization()
        self.sparsity_reg = SparsityRegularization()
        self.predicate_diversity = PredicateDiversityLoss()
        self.curriculum_penalty = CurriculumPenalty(max_clues=max_clues)
        
        self.lambda_entropy = lambda_entropy
        self.lambda_sparsity = lambda_sparsity
        self.lambda_predicate = lambda_predicate
        self.lambda_curriculum = lambda_curriculum
        self.lambda_deep_supervision = lambda_deep_supervision
        self.lambda_act = lambda_act
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        attention_maps: torch.Tensor,
        stop_logits: torch.Tensor,
        predicates: torch.Tensor,
        epoch: int = 0,
        max_epochs: int = 250,
        all_logits: Optional[list] = None,
        act_outputs: Optional[dict] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all RLAN losses.
        
        Args:
            logits: Shape (B, C, H, W) final output logits
            targets: Shape (B, H, W) target class indices
            attention_maps: Shape (B, K, H, W) attention weights
            stop_logits: Shape (B, K) stop probability logits
            predicates: Shape (B, P) predicate activations
            epoch: Current epoch number
            max_epochs: Total number of epochs
            all_logits: Optional list of (B, C, H, W) for deep supervision
            act_outputs: Optional dict with ACT outputs for halt loss computation
                - 'q_halt_logits': List of (B,) Q-values for halting at each step
                - 'q_continue_logits': List of (B,) Q-values for continuing
            
        Returns:
            Dict with keys:
                - total_loss: Combined weighted loss
                - focal_loss: Task loss
                - entropy_loss: Attention sharpness loss
                - sparsity_loss: Clue efficiency loss
                - predicate_loss: Predicate diversity loss
                - curriculum_loss: Complexity penalty
                - deep_supervision_loss: Intermediate step losses (if all_logits provided)
                - act_loss: ACT halting loss (if act_outputs provided)
        """
        # Primary task loss (name reflects actual loss function used)
        task_loss = self.task_loss(logits, targets)
        
        # Regularization losses
        entropy_loss = self.entropy_reg(attention_maps)
        sparsity_loss = self.sparsity_reg(stop_logits)
        predicate_loss = self.predicate_diversity(predicates)
        curriculum_loss = self.curriculum_penalty(stop_logits, epoch, max_epochs)
        
        # Deep supervision loss (if intermediate predictions available)
        if all_logits is not None and len(all_logits) > 1:
            ds_losses = []
            num_steps = len(all_logits)
            for t, step_logits in enumerate(all_logits[:-1]):  # Exclude final (already in task_loss)
                weight = 0.5 ** (num_steps - 1 - t)  # Later steps weighted more
                step_loss = self.task_loss(step_logits, targets)
                ds_losses.append(weight * step_loss)
            deep_supervision_loss = sum(ds_losses) / len(ds_losses) if ds_losses else torch.tensor(0.0)
        else:
            deep_supervision_loss = torch.tensor(0.0, device=logits.device)
        
        # ACT loss (if ACT outputs available)
        # Uses per-step correctness as reward signal for Q-learning
        act_loss = torch.tensor(0.0, device=logits.device)
        if act_outputs is not None and 'q_halt_logits' in act_outputs:
            q_halt_list = act_outputs['q_halt_logits']  # List of (B,) tensors per step
            
            if all_logits is not None and len(all_logits) == len(q_halt_list):
                # Compute per-step correctness as reward signal
                # A step is "correct" if its prediction matches the target
                act_losses = []
                for t, (q_halt, step_logits) in enumerate(zip(q_halt_list, all_logits)):
                    # Per-pixel accuracy at this step
                    step_preds = step_logits.argmax(dim=1)  # (B, H, W)
                    step_correct = (step_preds == targets).float().mean(dim=(1, 2))  # (B,)
                    
                    # Q_halt should predict whether halting NOW gives correct answer
                    # BCE loss: q_halt should be high when step is correct
                    step_act_loss = F.binary_cross_entropy_with_logits(
                        q_halt, step_correct, reduction='mean'
                    )
                    act_losses.append(step_act_loss)
                
                act_loss = sum(act_losses) / len(act_losses) if act_losses else act_loss
        
        # Combine losses
        total_loss = (
            task_loss
            + self.lambda_entropy * entropy_loss
            + self.lambda_sparsity * sparsity_loss
            + self.lambda_predicate * predicate_loss
            + self.lambda_curriculum * curriculum_loss
            + self.lambda_deep_supervision * deep_supervision_loss
            + self.lambda_act * act_loss
        )
        
        return {
            "total_loss": total_loss,
            "task_loss": task_loss,  # Accurate name (stablemax, focal_stablemax, or focal)
            "focal_loss": task_loss,  # Backward compatibility alias
            "entropy_loss": entropy_loss,
            "sparsity_loss": sparsity_loss,
            "predicate_loss": predicate_loss,
            "curriculum_loss": curriculum_loss,
            "deep_supervision_loss": deep_supervision_loss,
            "act_loss": act_loss,
            "loss_mode": self.loss_mode,  # Include mode for logging
        }


# Backward compatibility alias
RLANLossFunction = RLANLoss
