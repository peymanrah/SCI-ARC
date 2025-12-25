"""
Content-Invariant Structure Learning (CISL) for SCI-ARC.

CISL is a general-purpose learning objective for AI systems that need to:
- Learn structural patterns invariant to content changes
- Generalize across different instantiations of the same rule
- Prevent representation collapse in few-shot settings

This module implements the "Quad-Loss" approach:

1. L_recon: Reconstruction loss (already exists as task_loss)
2. L_consist: Within-Task Consistency - all demos in a task should have same structure
3. L_content_inv: Content Invariance - structure should be same after content permutation
4. L_var: Batch Variance - prevents collapse to constant zero

Design Philosophy:
- Structure = "what transformation/rule is being applied"
- Content = "what specific values/objects are involved"
- CISL forces: S(task) = S(permute_content(task))

This is superior to standard contrastive learning because:
- Works with few samples (2-4 demos, typical in few-shot learning)
- Explicitly tests structure-content separation
- Variance term prevents the "constant zero" collapse
- Generalizes beyond color to any content permutation

For ARC-AGI: Colors are content, spatial transformations are structure
For Language: Entities are content, syntactic patterns are structure
For Vision: Object identities are content, spatial relationships are structure

NOTE: CICL (Color-Invariant) is the old name. CISL is the general term.
      CICLLoss is still exported for backward compatibility.

Author: SCI-ARC Research Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import random
import numpy as np


class WithinTaskConsistencyLoss(nn.Module):
    """
    Within-Task Consistency Loss.
    
    For a task with K demos, all structure embeddings should be identical
    (since they all represent the same transformation rule).
    
    L_consist = (1/K) * Σ ||z_i - mean(z)||²
    
    This replaces cross-task contrastive learning with a simpler,
    more stable objective that works with few samples.
    """
    
    def __init__(self, normalize: bool = True):
        super().__init__()
        self.normalize = normalize
    
    def forward(
        self,
        structure_embeddings: torch.Tensor,  # [K, D] or [B, K, D]
        mask: Optional[torch.Tensor] = None   # [K] or [B, K] valid slot mask (rarely needed)
    ) -> torch.Tensor:
        """
        Compute within-task consistency loss.
        
        Measures variance across the K structural slots. For a well-trained model,
        all K slots should converge to similar representations for the same task,
        as they all represent aspects of the same transformation.
        
        Args:
            structure_embeddings: Structure slot embeddings
                If [K, D]: Single task with K slots
                If [B, K, D]: Batch of B tasks, each with K slots
            mask: Optional mask for valid slots (rarely needed - all slots are valid)
        
        Returns:
            Scalar consistency loss = mean variance from centroid
        """
        if structure_embeddings.dim() == 2:
            # Single task: [K, D]
            K, D = structure_embeddings.shape
            z = structure_embeddings
            
            if self.normalize:
                z = F.normalize(z, dim=-1)
            
            # Mean embedding
            if mask is not None:
                # Masked mean
                mask_float = mask.float().unsqueeze(-1)  # [K, 1]
                z_mean = (z * mask_float).sum(0) / mask_float.sum(0).clamp(min=1)
            else:
                z_mean = z.mean(dim=0)  # [D]
            
            # Consistency: distance from mean
            diff = z - z_mean.unsqueeze(0)  # [K, D]
            loss = (diff ** 2).sum(-1)  # [K]
            
            if mask is not None:
                loss = (loss * mask.float()).sum() / mask.float().sum().clamp(min=1)
            else:
                loss = loss.mean()
            
            return loss
        
        else:
            # Batch of tasks: [B, K, D]
            B, K, D = structure_embeddings.shape
            z = structure_embeddings
            
            if self.normalize:
                z = F.normalize(z, dim=-1)
            
            # Mean per task
            if mask is not None:
                # Masked mean per task: [B, D]
                mask_float = mask.float().unsqueeze(-1)  # [B, K, 1]
                z_mean = (z * mask_float).sum(1) / mask_float.sum(1).clamp(min=1)
            else:
                z_mean = z.mean(dim=1)  # [B, D]
            
            # Consistency per task
            diff = z - z_mean.unsqueeze(1)  # [B, K, D]
            loss_per_demo = (diff ** 2).sum(-1)  # [B, K]
            
            if mask is not None:
                # Masked average
                loss_per_task = (loss_per_demo * mask.float()).sum(1) / mask.float().sum(1).clamp(min=1)
            else:
                loss_per_task = loss_per_demo.mean(1)  # [B]
            
            return loss_per_task.mean()


class ContentInvarianceLoss(nn.Module):
    """
    Content Invariance Loss (generalized from ColorInvarianceLoss).
    
    The structure embedding should be identical after content permutation.
    This explicitly teaches: "Content is not structure."
    
    For ARC: Color permutation tests this (colors are content, not structure)
    For Language: Entity substitution tests this (entities are content)
    For Graphs: Node relabeling tests this (labels are content)
    
    L_content_inv = ||mean(z_original) - mean(z_content_permuted)||²
    
    This is the key SCI principle: same structure, different content → same embedding.
    """
    
    def __init__(self, normalize: bool = True):
        super().__init__()
        self.normalize = normalize
    
    def forward(
        self,
        z_original: torch.Tensor,      # [B, D] or [B, K, D]
        z_content_permuted: torch.Tensor  # [B, D] or [B, K, D]
    ) -> torch.Tensor:
        """
        Compute content invariance loss.
        
        Args:
            z_original: Structure embedding from original task
            z_content_permuted: Structure embedding from content-permuted task
        
        Returns:
            Scalar invariance loss
        """
        # Handle both [B, D] and [B, K, D] inputs
        if z_original.dim() == 3:
            z_orig = z_original.mean(dim=1)  # [B, D]
            z_perm = z_content_permuted.mean(dim=1)  # [B, D]
        else:
            z_orig = z_original
            z_perm = z_content_permuted
        
        if self.normalize:
            z_orig = F.normalize(z_orig, dim=-1)
            z_perm = F.normalize(z_perm, dim=-1)
        
        # L2 distance
        diff = z_orig - z_perm
        loss = (diff ** 2).sum(-1).mean()
        
        return loss


class BatchVarianceLoss(nn.Module):
    """
    Batch Variance Loss (Anti-Collapse Regularization).
    
    Ensures that structure embeddings are diverse across the batch.
    Prevents the "constant zero" collapse where encoder outputs zeros for everything.
    
    CRITICAL FIX: We compute variance on L2-NORMALIZED embeddings to measure
    DIRECTIONAL diversity, not magnitude. Without this, embeddings can grow
    unboundedly (norm 40→950) while all pointing in the same direction
    (cosine_sim=1.0), causing the variance loss to be 0 (fooled by large magnitudes).
    
    L_var = ReLU(γ - std(normalize(Z_batch)))
    
    where γ is a target standard deviation (e.g., 0.5).
    
    Inspired by VICReg: https://arxiv.org/abs/2105.04906
    """
    
    def __init__(self, target_std: float = 0.5, epsilon: float = 1e-4):
        """
        Args:
            target_std: Target standard deviation for embeddings
            epsilon: Small constant for numerical stability
        """
        super().__init__()
        self.target_std = target_std
        self.epsilon = epsilon
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute variance regularization loss on NORMALIZED embeddings.
        
        Args:
            z: Embeddings [B, D] or [B, K, D]
        
        Returns:
            Scalar variance loss (0 if std >= target_std)
        """
        # Flatten if needed
        if z.dim() == 3:
            z = z.mean(dim=1)  # [B, D]
        
        B, D = z.shape
        
        if B < 2:
            return torch.tensor(0.0, device=z.device)
        
        # CRITICAL: Normalize embeddings to unit sphere BEFORE computing variance
        # This measures DIRECTIONAL diversity, not magnitude
        # Without this, embeddings can collapse to same direction while growing in norm
        z = F.normalize(z, dim=-1)
        
        # Compute std per dimension, then average
        z_centered = z - z.mean(dim=0, keepdim=True)
        std_per_dim = torch.sqrt(z_centered.var(dim=0) + self.epsilon)  # [D]
        avg_std = std_per_dim.mean()
        
        # Hinge loss: penalty only if std is below target
        loss = F.relu(self.target_std - avg_std)
        
        return loss


class CISLLoss(nn.Module):
    """
    Content-Invariant Structure Learning (CISL) Loss.
    
    A general-purpose learning objective for structure-content separation:
    
    L_total = L_recon + λ₁·L_consist + λ₂·L_content_inv + λ₃·L_var
    
    This replaces standard contrastive learning with a more stable,
    theoretically grounded method for few-shot learning regimes.
    
    Applicability:
    - ARC-AGI: Colors are content, spatial transformations are structure
    - Language: Entities are content, syntactic patterns are structure
    - Vision: Object identities are content, spatial relationships are structure
    - Graphs: Node labels are content, topology patterns are structure
    
    Features:
    - Per-batch logging of all loss components
    - Epoch-level statistics aggregation
    - Debug mode for detailed embedding analysis
    """
    
    def __init__(
        self,
        consist_weight: float = 1.0,
        content_inv_weight: float = 1.0,
        variance_weight: float = 1.0,
        target_std: float = 0.5,
        normalize: bool = True,
        debug: bool = False
    ):
        """
        Args:
            consist_weight: Weight for within-task consistency loss (λ₁)
            content_inv_weight: Weight for content invariance loss (λ₂)
            variance_weight: Weight for batch variance loss (λ₃)
            target_std: Target standard deviation for variance loss
            normalize: Whether to normalize embeddings before computing losses
            debug: Enable detailed per-batch logging with embedding statistics
        """
        super().__init__()
        
        self.consist_weight = consist_weight
        self.content_inv_weight = content_inv_weight
        self.variance_weight = variance_weight
        self.debug = debug
        
        self.consistency = WithinTaskConsistencyLoss(normalize=normalize)
        self.content_invariance = ContentInvarianceLoss(normalize=normalize)
        self.variance = BatchVarianceLoss(target_std=target_std)
        
        # Statistics tracking for debugging
        self._batch_count = 0
        self._epoch_stats = self._reset_epoch_stats()
    
    def _reset_epoch_stats(self) -> Dict:
        """Reset epoch-level statistics."""
        return {
            'consist_sum': 0.0,
            'content_inv_sum': 0.0,
            'variance_sum': 0.0,
            'total_sum': 0.0,
            'z_mean_sum': 0.0,
            'z_std_sum': 0.0,
            'z_norm_sum': 0.0,
            'batch_count': 0
        }
    
    def get_epoch_stats(self) -> Dict[str, float]:
        """Get averaged epoch statistics for logging."""
        n = max(self._epoch_stats['batch_count'], 1)
        return {
            'cisl/consist_avg': self._epoch_stats['consist_sum'] / n,
            'cisl/content_inv_avg': self._epoch_stats['content_inv_sum'] / n,
            'cisl/variance_avg': self._epoch_stats['variance_sum'] / n,
            'cisl/total_avg': self._epoch_stats['total_sum'] / n,
            'cisl/z_mean_avg': self._epoch_stats['z_mean_sum'] / n,
            'cisl/z_std_avg': self._epoch_stats['z_std_sum'] / n,
            'cisl/z_norm_avg': self._epoch_stats['z_norm_sum'] / n,
            'cisl/batches_processed': self._epoch_stats['batch_count'],
        }
    
    def reset_epoch_stats(self):
        """Call at start of each epoch to reset statistics."""
        self._epoch_stats = self._reset_epoch_stats()
        self._batch_count = 0
    
    def forward(
        self,
        z_struct: torch.Tensor,                    # [B, K, D] - aggregated
        z_struct_demos: Optional[torch.Tensor] = None,  # [B, P, K, D] - per-demo (for consistency)
        z_struct_content_aug: Optional[torch.Tensor] = None,  # [B, K, D] or [B, D]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all CISL losses with detailed statistics for logging.
        
        Args:
            z_struct: Aggregated structure embeddings [B, K, D]
                      K = number of structural slots (learned query outputs)
            z_struct_demos: Per-demo structure embeddings [B, P, K, D]
                           P = number of demo pairs
                           Used for consistency loss (variance across demos)
            z_struct_content_aug: Structure embeddings from content-permuted task
                                  Should be identical to z_struct if model is content-invariant
        
        Returns:
            Dict with:
                - 'total': Combined CISL loss (scalar)
                - 'consistency': Within-task consistency loss (variance across demos)
                - 'content_inv': Content invariance loss (z_struct vs z_struct_content_aug)
                - 'variance': Batch variance loss (anti-collapse regularization)
                - 'stats': Dict with embedding statistics (z_mean, z_std, z_norm, cos_sim)
        
        Note: The consistency loss measures variance ACROSS DEMO PAIRS (P dimension).
              Each demo in a task should produce similar structure embeddings since
              they all represent the same transformation rule.
        """
        device = z_struct.device
        losses = {}
        stats = {}
        
        # Compute embedding statistics for monitoring
        # Use contiguous().view() or reshape() to handle non-contiguous tensors (e.g., from expand())
        z_flat = z_struct.reshape(-1, z_struct.size(-1))  # Flatten to [N, D]
        stats['z_mean'] = z_flat.mean().item()
        stats['z_std'] = z_flat.std().item()
        stats['z_norm'] = z_flat.norm(dim=-1).mean().item()
        
        # 1. Within-Task Consistency (across demo pairs)
        # All P demos should produce similar structure embeddings since they represent
        # the same transformation - measure variance across the P dimension
        if z_struct_demos is not None and z_struct_demos.dim() == 4:
            # z_struct_demos: [B, P, K, D] - P demo pairs, K structural slots
            B, P, K, D = z_struct_demos.shape
            
            if P > 1:
                # Aggregate across slots first, then measure consistency across demos
                # Each demo's structure = mean of its K slots: [B, P, D]
                z_per_demo = z_struct_demos.mean(dim=2)  # [B, P, D]
                
                # Now measure consistency across the P demos
                L_consist = self.consistency(z_per_demo, mask=None)  # Uses [B, K, D] path with K=P
            else:
                L_consist = torch.tensor(0.0, device=device)
        elif z_struct.dim() == 3 and z_struct.size(1) > 1:
            # Fallback: measure consistency across K slots (old behavior)
            L_consist = self.consistency(z_struct, mask=None)
        else:
            L_consist = torch.tensor(0.0, device=device)
        losses['consistency'] = L_consist
        
        # 2. Content Invariance (only if content-augmented version provided)
        if z_struct_content_aug is not None:
            L_content_inv = self.content_invariance(z_struct, z_struct_content_aug)
            
            # Compute cosine similarity for debugging
            z_orig_flat = z_struct.mean(dim=1) if z_struct.dim() == 3 else z_struct
            z_aug_flat = z_struct_content_aug.mean(dim=1) if z_struct_content_aug.dim() == 3 else z_struct_content_aug
            z_orig_norm = F.normalize(z_orig_flat, dim=-1)
            z_aug_norm = F.normalize(z_aug_flat, dim=-1)
            cos_sim = (z_orig_norm * z_aug_norm).sum(-1).mean().item()
            stats['orig_aug_cos_sim'] = cos_sim
        else:
            L_content_inv = torch.tensor(0.0, device=device)
            stats['orig_aug_cos_sim'] = 0.0
        losses['content_inv'] = L_content_inv
        
        # 3. Batch Variance (anti-collapse)
        L_var = self.variance(z_struct)
        losses['variance'] = L_var
        
        # Total CISL loss
        total = (
            self.consist_weight * L_consist +
            self.content_inv_weight * L_content_inv +
            self.variance_weight * L_var
        )
        losses['total'] = total
        losses['stats'] = stats
        
        # Update epoch statistics
        self._epoch_stats['consist_sum'] += L_consist.item()
        self._epoch_stats['content_inv_sum'] += L_content_inv.item()
        self._epoch_stats['variance_sum'] += L_var.item()
        self._epoch_stats['total_sum'] += total.item()
        self._epoch_stats['z_mean_sum'] += stats['z_mean']
        self._epoch_stats['z_std_sum'] += stats['z_std']
        self._epoch_stats['z_norm_sum'] += stats['z_norm']
        self._epoch_stats['batch_count'] += 1
        self._batch_count += 1
        
        return losses


# Backward compatibility aliases
CICLLoss = CISLLoss  # Old name -> new name
ColorInvarianceLoss = ContentInvarianceLoss  # More general name


def apply_content_permutation(grid: torch.Tensor, perm: Optional[Dict[int, int]] = None) -> Tuple[torch.Tensor, Dict[int, int]]:
    """
    Apply content permutation to a grid (generalized from color permutation).
    
    For ARC, this is color permutation. For other domains, this could be:
    - Entity substitution (NLP)
    - Node relabeling (graphs)
    - Object ID swapping (vision)
    
    Args:
        grid: [H, W] or [B, H, W] integer grid with values 0-9
        perm: Optional existing permutation {old_value: new_value}
              If None, a random permutation is generated
    
    Returns:
        Tuple of (permuted_grid, permutation_dict)
    """
    if perm is None:
        # Generate random permutation for values 1-9 (keep 0=background fixed)
        # Use torch.randperm for reproducibility with torch.manual_seed
        device = grid.device if hasattr(grid, 'device') else 'cpu'
        shuffled_indices = torch.randperm(9, device=device) + 1  # Values 1-9 shuffled
        perm = {0: 0}  # Background stays fixed
        for old_val in range(1, 10):
            perm[old_val] = shuffled_indices[old_val - 1].item()
    
    # Apply permutation
    result = grid.clone()
    for old_v, new_v in perm.items():
        result[grid == old_v] = new_v
    
    return result, perm


# Backward compatibility alias
apply_color_permutation = apply_content_permutation


def apply_content_permutation_batch(
    input_grids: torch.Tensor,   # [B, K, H, W]
    output_grids: torch.Tensor,  # [B, K, H, W]
    test_inputs: torch.Tensor,   # [B, H, W]
    test_outputs: torch.Tensor   # [B, H, W]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply consistent content permutation to all grids in a batch.
    
    CRITICAL: The SAME permutation is applied to all grids in each batch item.
    This preserves the transformation rule while changing the content.
    
    Args:
        input_grids: Batch of input grids [B, K, H, W]
        output_grids: Batch of output grids [B, K, H, W]
        test_inputs: Batch of test inputs [B, H, W]
        test_outputs: Batch of test outputs [B, H, W]
    
    Returns:
        Tuple of permuted (input_grids, output_grids, test_inputs, test_outputs)
    """
    B = input_grids.size(0)
    device = input_grids.device
    
    # Generate one permutation per batch item
    # Use torch.randperm for reproducibility with torch.manual_seed
    perms = []
    for _ in range(B):
        shuffled_indices = torch.randperm(9, device=device) + 1  # Values 1-9 shuffled
        perm = {0: 0}  # Background stays fixed
        for old_val in range(1, 10):
            perm[old_val] = shuffled_indices[old_val - 1].item()
        perms.append(perm)
    
    # Apply permutations
    input_grids_perm = input_grids.clone()
    output_grids_perm = output_grids.clone()
    test_inputs_perm = test_inputs.clone()
    test_outputs_perm = test_outputs.clone()
    
    for b in range(B):
        perm = perms[b]
        for old_v, new_v in perm.items():
            input_grids_perm[b][input_grids[b] == old_v] = new_v
            output_grids_perm[b][output_grids[b] == old_v] = new_v
            test_inputs_perm[b][test_inputs[b] == old_v] = new_v
            test_outputs_perm[b][test_outputs[b] == old_v] = new_v
    
    return input_grids_perm, output_grids_perm, test_inputs_perm, test_outputs_perm


# Backward compatibility alias
apply_color_permutation_batch = apply_content_permutation_batch
