"""
Loss Functions for SCI-ARC

Key losses:
1. Structural Contrastive Loss (SCL): Core SCI innovation
   - Same transformation → similar structural representation
   - Different transformation → different representation

2. Orthogonality Loss: S(x) ⊥ C(x)
   - Structure and content representations should be orthogonal
   - Prevents information leakage

3. Deep Supervision CE Loss: From TRM
   - Apply CE loss at each refinement step
   - Later steps weighted more heavily

4. Stablemax Cross Entropy: From TRM
   - More numerically stable than standard softmax
   - Better for small vocab sizes like ARC (12 tokens)
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def stablemax_cross_entropy(
    logits: torch.Tensor, 
    labels: torch.Tensor, 
    n: int = 12, 
    ignore_index: int = 0
) -> torch.Tensor:
    """
    Stablemax cross entropy loss from TRM.
    
    More numerically stable than standard softmax for small vocab sizes.
    From TRM's models/losses.py.
    
    The stablemax function is defined as:
        stablemax(x)_i = exp(x_i - max(x)) / ((n-1)/n * exp(max(x) - max(x)) + 1/n * sum(exp(x - max(x))))
    
    This is more stable because:
    1. Subtracts max for numerical stability
    2. Uses a modified normalization that's more balanced for small n
    
    Args:
        logits: [*, vocab_size] unnormalized logits
        labels: [*] integer labels
        n: vocab size (default 12 for ARC: PAD + EOS + 10 colors)
        ignore_index: label to ignore in loss computation (default 0 = PAD)
    
    Returns:
        Scalar loss
    """
    # Shift by max for numerical stability
    max_logits = logits.max(dim=-1, keepdim=True).values
    shifted = logits - max_logits
    
    # Stablemax normalization
    # normalizer = (n-1)/n * max + 1/n * logsumexp
    normalizer = ((n - 1) / n) * max_logits + (1 / n) * shifted.logsumexp(dim=-1, keepdim=True)
    
    # Log probabilities
    log_probs = shifted - normalizer
    
    # NLL loss
    return F.nll_loss(
        log_probs.view(-1, log_probs.size(-1)), 
        labels.view(-1), 
        ignore_index=ignore_index
    )


class StructuralContrastiveLoss(nn.Module):
    """
    Structural Contrastive Loss (SCL) adapted for ARC grids.
    
    IDENTICAL to SCI's SCL in principle:
    - Positive pairs: Same transformation rule, different grids
    - Negative pairs: Different transformation rules
    - Objective: Pull positives together, push negatives apart
    
    This is the KEY SCI INNOVATION that should transfer to ARC:
    - "Rotate 90°" tasks should cluster together
    - "Flip horizontal" tasks should cluster together
    - Different transformation types should be separated
    
    Uses InfoNCE loss formulation.
    """
    
    def __init__(
        self, 
        temperature: float = 0.07,
        normalize: bool = True
    ):
        """
        Args:
            temperature: Temperature for softmax scaling
            normalize: Whether to L2-normalize representations
        """
        super().__init__()
        self.temperature = temperature
        self.normalize = normalize
    
    def forward(
        self,
        structure_reps: torch.Tensor,  # [B, K, D]
        transform_labels: torch.Tensor  # [B] which transformation family
    ) -> torch.Tensor:
        """
        Compute SCL loss.
        
        Args:
            structure_reps: Structural representations from SE [B, K, D]
            transform_labels: Integer labels indicating transformation type
                             (e.g., 0=rotate, 1=flip, 2=color_swap, ...)
        
        Returns:
            loss: Scalar SCL loss
        """
        B = structure_reps.size(0)
        
        if B < 2:
            # Need at least 2 samples for contrastive learning
            return torch.tensor(0.0, device=structure_reps.device)
        
        # Pool structure slots to single vector
        z = structure_reps.mean(dim=1)  # [B, D]
        
        # Normalize representations
        if self.normalize:
            z = F.normalize(z, dim=-1)
        
        # Compute pairwise similarity matrix
        sim = torch.mm(z, z.t()) / self.temperature  # [B, B]
        
        # Create masks
        # Diagonal mask (self-similarity - to be excluded)
        mask_diag = torch.eye(B, device=z.device).bool()
        
        # Positive mask: same transform family (excluding self)
        labels_equal = transform_labels.unsqueeze(0) == transform_labels.unsqueeze(1)
        pos_mask = labels_equal & ~mask_diag
        
        # InfoNCE loss - FULLY VECTORIZED (no Python loops)
        # For each anchor, compute: -log(exp(pos) / sum(exp(all)))
        
        # Mask out diagonal for denominator (exclude self-similarity)
        sim_masked = sim.masked_fill(mask_diag, float('-inf'))
        
        # Log-sum-exp over all samples except self for each anchor [B]
        log_sum_exp = torch.logsumexp(sim_masked, dim=1)
        
        # Compute loss matrix: loss[i,j] = -sim[i,j] + log_sum_exp[i]
        # This is the InfoNCE loss if (i,j) is a positive pair
        loss_matrix = -sim + log_sum_exp.unsqueeze(1)  # [B, B]
        
        # Count positive pairs per anchor (for averaging)
        pos_counts = pos_mask.float().sum(dim=1)  # [B]
        
        # Identify anchors that have at least one positive pair
        has_positives = pos_counts > 0  # [B]
        num_valid_anchors = has_positives.float().sum()  # Scalar on GPU
        
        # Sum of losses for positive pairs per anchor
        # Mask non-positive pairs with 0, then sum
        pos_loss_sum = (loss_matrix * pos_mask.float()).sum(dim=1)  # [B]
        
        # Average over positive pairs for each anchor (avoid div by zero)
        # Use where to handle anchors with no positives
        per_anchor_loss = torch.where(
            has_positives,
            pos_loss_sum / pos_counts.clamp(min=1),
            torch.zeros_like(pos_loss_sum)
        )
        
        # Average only over valid anchors (those with positive pairs)
        # Use safe division to handle case with no valid anchors
        total_loss = per_anchor_loss.sum()
        return total_loss / num_valid_anchors.clamp(min=1)


class OrthogonalityLoss(nn.Module):
    """
    Orthogonality loss to enforce S(x) ⊥ C(x).
    
    This ensures structure and content representations don't leak
    into each other - critical for clean separation in SCI.
    
    L_orth = |mean(S) · mean(C)|
    
    Should be close to 0 when orthogonal.
    """
    
    def __init__(self, normalize: bool = True):
        """
        Args:
            normalize: Whether to L2-normalize before computing dot product
        """
        super().__init__()
        self.normalize = normalize
    
    def forward(
        self,
        structure_rep: torch.Tensor,  # [B, K, D]
        content_rep: torch.Tensor     # [B, M, D]
    ) -> torch.Tensor:
        """
        Compute orthogonality loss.
        
        Args:
            structure_rep: [B, K, D] structural representation
            content_rep: [B, M, D] content representation
        
        Returns:
            loss: Scalar orthogonality loss
        """
        # Pool to single vector per sample
        s = structure_rep.mean(dim=1)  # [B, D]
        c = content_rep.mean(dim=1)    # [B, D]
        
        # Normalize
        if self.normalize:
            s = F.normalize(s, dim=-1, eps=1e-8)
            c = F.normalize(c, dim=-1, eps=1e-8)
        
        # Dot product should be 0 for orthogonal vectors
        dot_product = (s * c).sum(dim=-1)  # [B]
        
        # Take absolute value (we want it close to 0, regardless of sign)
        ortho_loss = dot_product.abs().mean()
        
        return ortho_loss


class DeepSupervisionLoss(nn.Module):
    """
    Deep supervision loss from TRM.
    
    Apply CE loss at each refinement step, with later steps weighted more.
    This helps the model learn to progressively improve its answer.
    
    Weight schedule: linear from 1/K to 1 (later steps more important)
    """
    
    def __init__(
        self,
        num_steps: int,
        weight_schedule: str = "linear",
        ignore_index: int = -1
    ):
        """
        Args:
            num_steps: Number of refinement steps (H_cycles)
            weight_schedule: "linear", "exponential", or "uniform"
            ignore_index: Label value to ignore (for padding)
        """
        super().__init__()
        self.num_steps = num_steps
        self.weight_schedule = weight_schedule
        self.ignore_index = ignore_index
        
        # Compute weights
        if weight_schedule == "linear":
            weights = torch.arange(1, num_steps + 1).float() / num_steps
        elif weight_schedule == "exponential":
            weights = torch.pow(2.0, torch.arange(num_steps).float()) / (2 ** num_steps - 1)
        elif weight_schedule == "uniform":
            weights = torch.ones(num_steps) / num_steps
        else:
            raise ValueError(f"Unknown weight schedule: {weight_schedule}")
        
        self.register_buffer('step_weights', weights)
    
    def forward(
        self,
        predictions: List[torch.Tensor],  # List of [B, H, W, C] predictions
        target: torch.Tensor               # [B, H, W] ground truth
    ) -> torch.Tensor:
        """
        Compute deep supervision loss.
        
        Args:
            predictions: List of predictions at each step [B, H, W, C]
            target: Ground truth grid [B, H, W]
        
        Returns:
            loss: Weighted average CE loss
        """
        device = predictions[0].device
        
        total_loss = torch.tensor(0.0, device=device)
        total_weight = torch.tensor(0.0, device=device)
        
        for t, pred in enumerate(predictions):
            # Flatten for CE computation
            B, H, W, C = pred.shape
            pred_flat = pred.reshape(-1, C)  # [B*H*W, C]
            target_flat = target.reshape(-1)  # [B*H*W]
            
            # Use standard CE with ignore_index (avoids CPU-GPU sync from .any())
            step_loss = F.cross_entropy(
                pred_flat,
                target_flat,
                ignore_index=self.ignore_index,
                reduction='mean'
            )
            
            weight = self.step_weights[t]
            total_loss = total_loss + weight * step_loss
            total_weight = total_weight + weight
        
        # Normalize by total weight
        if total_weight > 0:
            return total_loss / total_weight
        else:
            return total_loss


class SCIARCLoss(nn.Module):
    """
    Combined loss for SCI-ARC training.
    
    L_total = L_CE (deep supervision)
            + λ_scl * L_SCL (structural contrastive)
            + λ_orth * L_orth (orthogonality)
    
    This combines the TRM insight (deep supervision) with
    SCI innovations (SCL + orthogonality).
    """
    
    def __init__(
        self,
        H_cycles: int = 16,
        scl_weight: float = 0.1,
        orthogonality_weight: float = 0.01,
        temperature: float = 0.07,
        weight_schedule: str = "linear",
        ignore_index: int = -1
    ):
        """
        Args:
            H_cycles: Number of refinement steps for deep supervision
            scl_weight: Weight for SCL loss
            orthogonality_weight: Weight for orthogonality loss
            temperature: Temperature for SCL
            weight_schedule: Weight schedule for deep supervision
            ignore_index: Label value to ignore
        """
        super().__init__()
        
        self.scl_weight = scl_weight
        self.orth_weight = orthogonality_weight
        
        # Component losses
        self.deep_supervision = DeepSupervisionLoss(
            num_steps=H_cycles,
            weight_schedule=weight_schedule,
            ignore_index=ignore_index
        )
        self.scl = StructuralContrastiveLoss(temperature=temperature)
        self.orthogonality = OrthogonalityLoss()
    
    def forward(
        self,
        predictions: List[torch.Tensor],  # Predictions at each step
        target: torch.Tensor,              # Ground truth
        structure_rep: torch.Tensor,       # For SCL
        content_rep: torch.Tensor,         # For orthogonality
        transform_labels: Optional[torch.Tensor] = None  # Transform family labels
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all losses.
        
        Args:
            predictions: List of [B, H, W, C] predictions
            target: [B, H, W] ground truth
            structure_rep: [B, K, D] structural representation
            content_rep: [B, M, D] content representation
            transform_labels: [B] transformation family labels (for SCL)
        
        Returns:
            Dict with all losses:
                - 'total': Combined loss
                - 'ce': Deep supervision CE loss
                - 'scl': Structural contrastive loss
                - 'orthogonality': Orthogonality loss
        """
        # 1. Deep supervision CE loss
        ce_loss = self.deep_supervision(predictions, target)
        
        # 2. Structural Contrastive Loss
        if transform_labels is not None:
            scl_loss = self.scl(structure_rep, transform_labels)
        else:
            # If no labels, use batch-wise contrastive (all different)
            # This is less effective but still provides regularization
            scl_loss = torch.tensor(0.0, device=structure_rep.device)
        
        # 3. Orthogonality loss
        orth_loss = self.orthogonality(structure_rep, content_rep)
        
        # Combined loss
        total_loss = ce_loss + self.scl_weight * scl_loss + self.orth_weight * orth_loss
        
        return {
            'total': total_loss,
            'ce': ce_loss,
            'scl': scl_loss,
            'orthogonality': orth_loss
        }


class TRMCompatibleLoss(nn.Module):
    """
    Loss compatible with TRM training setup for fair comparison.
    
    Uses TRM's exact loss formulation:
    - Cross-entropy on valid positions
    - Ignore PAD tokens (value 0 in TRM format)
    """
    
    def __init__(self, ignore_index: int = 0):
        super().__init__()
        self.ignore_index = ignore_index
    
    def forward(
        self,
        logits: torch.Tensor,   # [B, seq_len, vocab_size]
        labels: torch.Tensor,   # [B, seq_len]
        structure_rep: Optional[torch.Tensor] = None,
        content_rep: Optional[torch.Tensor] = None,
        transform_labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        TRM-compatible loss computation.
        
        Args:
            logits: [B, seq_len, vocab_size] predictions
            labels: [B, seq_len] targets
            structure_rep: Optional [B, K, D] for SCL
            content_rep: Optional [B, M, D] for orthogonality
            transform_labels: Optional [B] for SCL
        
        Returns:
            Dict with losses
        """
        B, L, V = logits.shape
        
        # Flatten
        logits_flat = logits.view(-1, V)
        labels_flat = labels.view(-1)
        
        # Valid mask
        valid_mask = labels_flat != self.ignore_index
        
        # CE loss
        if valid_mask.any():
            ce_loss = F.cross_entropy(
                logits_flat[valid_mask],
                labels_flat[valid_mask]
            )
        else:
            ce_loss = torch.tensor(0.0, device=logits.device)
        
        # Additional SCI losses if representations provided
        total_loss = ce_loss
        losses = {'ce': ce_loss}
        
        if structure_rep is not None and content_rep is not None:
            # Orthogonality loss
            orth_loss = OrthogonalityLoss()(structure_rep, content_rep)
            total_loss = total_loss + 0.01 * orth_loss
            losses['orthogonality'] = orth_loss
            
            # SCL if labels provided
            if transform_labels is not None:
                scl_loss = StructuralContrastiveLoss()(structure_rep, transform_labels)
                total_loss = total_loss + 0.1 * scl_loss
                losses['scl'] = scl_loss
        
        losses['total'] = total_loss
        
        # Compute metrics
        with torch.no_grad():
            preds = logits_flat.argmax(dim=-1)
            correct = (preds == labels_flat) & valid_mask
            accuracy = correct.sum().float() / valid_mask.sum().float() if valid_mask.any() else torch.tensor(0.0)
            losses['accuracy'] = accuracy
        
        return losses
