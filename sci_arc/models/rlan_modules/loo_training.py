"""
Leave-One-Out (LOO) Training Loss Module for RLAN HyperLoRA.

This module implements the LOO training paradigm that teaches the model
to generalize from N-1 examples to the Nth example, improving few-shot
generalization capabilities.

Architecture Overview:
- LOOTrainingLoss: Computes loss where one training pair is held out
- AugmentationEquivarianceLoss: Enforces consistent predictions across augmentations
- CombinedMetaLoss: Orchestrates all meta-learning losses

Key Features:
- Efficient batch processing with masking
- Support for variable number of training pairs
- Augmentation-aware consistency regularization
- Detailed health metrics for debugging
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LOOConfig:
    """Configuration for Leave-One-Out training."""
    enabled: bool = True
    loss_weight: float = 0.5  # Weight of LOO loss relative to task loss
    min_pairs_for_loo: int = 2  # Minimum training pairs needed for LOO
    max_loo_pairs: int = 4  # Maximum LOO passes to cap memory (ARC max is ~5)
    use_weighted_holdout: bool = True  # Weight harder holdouts more


@dataclass
class EquivarianceConfig:
    """Configuration for Augmentation Equivariance training."""
    enabled: bool = True
    loss_weight: float = 0.1  # Weight of equivariance loss
    augmentations: List[str] = field(default_factory=lambda: [
        'rotate_90', 'rotate_180', 'rotate_270', 'flip_h', 'flip_v'
    ])
    num_augmentations: int = 4  # Number of random augmentations per task


class LOOTrainingLoss(nn.Module):
    """
    Leave-One-Out Training Loss.
    
    This loss teaches the model to predict held-out examples using
    only the remaining examples. This is key for few-shot generalization.
    
    For each task with N training pairs:
    1. Iterate through each pair as the held-out pair
    2. Pool context from remaining N-1 pairs
    3. Predict held-out pair using the adapted weights
    4. Compute loss on held-out prediction
    
    Args:
        config: LOO configuration
        hidden_dim: Hidden dimension of the model
    """
    
    def __init__(self, config: LOOConfig, hidden_dim: int = 256):
        super().__init__()
        self.config = config
        self.hidden_dim = hidden_dim
        
    def forward(
        self,
        model: nn.Module = None,
        input_grids: torch.Tensor = None,  # (B, N, H, W) - support inputs
        output_grids: torch.Tensor = None,  # (B, N, H, W) - support targets
        pair_mask: torch.Tensor = None,  # (B, N) - which pairs are valid
        temperature: float = 1.0,
        # Iterative backward parameters (v3 memory fix)
        scaler: Any = None,  # GradScaler for AMP - if provided, backward happens inside
        loss_weight: float = 1.0,  # Weight to apply before backward
        grad_accumulation_steps: int = 1,  # Divide loss by this for accumulation
        # Legacy parameters for backward compatibility
        hyper_lora: nn.Module = None,
        rlan: nn.Module = None,
        support_inputs: torch.Tensor = None,
        support_targets: torch.Tensor = None,
        support_features: torch.Tensor = None,
        num_valid_pairs: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Compute LOO loss using the RLAN model.
        
        Can be called in two ways:
        1. Model-based (preferred): model=, input_grids=, output_grids=, pair_mask=
        2. Component-based (legacy): hyper_lora=, rlan=, support_inputs=, etc.
        
        MEMORY OPTIMIZATION (v3 - Iterative Backward):
        If scaler= is provided, backward() is called INSIDE the loop after each
        holdout, and the computation graph is freed immediately. This prevents
        O(N) memory accumulation.
        
        Returns:
            Dict with 'loo_loss' (tensor or float), 'loo_accuracy', 'loo_num_holdouts', etc.
            If scaler is provided, 'loo_loss' is a float (backward already done).
        """
        # Mode 1: Full model interface (used by train_rlan.py)
        if model is not None and input_grids is not None:
            return self._forward_with_model(
                model=model,
                input_grids=input_grids,
                output_grids=output_grids,
                pair_mask=pair_mask,
                temperature=temperature,
                scaler=scaler,
                loss_weight=loss_weight,
                grad_accumulation_steps=grad_accumulation_steps,
            )
        
        # Mode 2: Component-based interface (legacy)
        if hyper_lora is not None and rlan is not None and support_features is not None:
            return self._forward_with_components(
                hyper_lora=hyper_lora,
                rlan=rlan,
                support_inputs=support_inputs,
                support_targets=support_targets,
                support_features=support_features,
                num_valid_pairs=num_valid_pairs,
            )
        
        # Neither mode - return skip
        return {
            'loo_loss': torch.tensor(0.0),
            'loo_accuracy': 0.0,
            'loo_num_holdouts': 0,
            'loo_skipped': True,
            'loo_reason': 'Invalid arguments - need either model= or hyper_lora=+rlan=',
        }
    
    def _forward_with_model(
        self,
        model: nn.Module,
        input_grids: torch.Tensor,  # (B, N, H, W)
        output_grids: torch.Tensor,  # (B, N, H, W)
        pair_mask: torch.Tensor = None,
        temperature: float = 1.0,
        scaler: Any = None,  # GradScaler for iterative backward
        loss_weight: float = 1.0,
        grad_accumulation_steps: int = 1,
    ) -> Dict[str, Any]:
        """
        Compute LOO loss using full RLAN model.
        
        MEMORY-EFFICIENT IMPLEMENTATION (Dec 2025 v3 - Iterative Backward):
        - Encodes support features ONCE and detaches
        - Processes ONE holdout at a time with gradient
        - If scaler provided: backward() immediately, free graph, O(1) memory
        - If no scaler: accumulate losses, return tensor (legacy mode, O(N) memory)
        
        The iterative backward approach is CRITICAL for fitting LOO training
        on a 24GB GPU. Without it, memory usage is O(N) where N is number of
        training pairs, causing 50+GB allocation.
        """
        device = input_grids.device
        B, N, H, W = input_grids.shape
        
        # Use iterative backward if scaler is provided
        use_iterative_backward = scaler is not None
        
        # Check if model has HyperLoRA
        if not hasattr(model, 'hyper_lora') or model.hyper_lora is None:
            return {
                'loo_loss': torch.tensor(0.0, device=device),
                'loo_accuracy': 0.0,
                'loo_num_holdouts': 0,
                'loo_skipped': True,
                'loo_reason': 'Model does not have HyperLoRA',
            }
        
        # Skip if not enough pairs
        if N < self.config.min_pairs_for_loo:
            return {
                'loo_loss': torch.tensor(0.0, device=device),
                'loo_accuracy': 0.0,
                'loo_num_holdouts': 0,
                'loo_skipped': True,
                'loo_reason': f'Not enough pairs ({N} < {self.config.min_pairs_for_loo})',
            }
        
        # Determine valid pairs per sample
        if pair_mask is not None:
            num_valid_pairs = pair_mask.sum(dim=1)  # (B,)
        else:
            num_valid_pairs = torch.full((B,), N, device=device)
        
        # Check if model has ContextEncoder with spatial features enabled
        if not hasattr(model, 'context_encoder') or model.context_encoder is None:
            return {
                'loo_loss': torch.tensor(0.0, device=device),
                'loo_accuracy': 0.0,
                'loo_num_holdouts': 0,
                'loo_skipped': True,
                'loo_reason': 'Model does not have ContextEncoder (required for LOO)',
            }
        
        if not getattr(model.context_encoder, 'use_spatial_features', False):
            return {
                'loo_loss': torch.tensor(0.0, device=device),
                'loo_accuracy': 0.0,
                'loo_num_holdouts': 0,
                'loo_skipped': True,
                'loo_reason': 'ContextEncoder must have use_spatial_features=True for LOO',
            }
        
        # MEMORY FIX: Encode support features ONCE and DETACH
        # This prevents recomputing the same features N times
        with torch.no_grad():
            all_support_features = model.context_encoder(
                input_grids, output_grids, pair_mask
            )  # (B, N, D, H, W)
        
        D = all_support_features.shape[2]
        Hs, Ws = all_support_features.shape[3], all_support_features.shape[4]
        
        # Track metrics
        total_correct = 0
        total_pixels = 0
        num_holdouts = 0
        total_loss_value = 0.0  # Track loss value for logging
        
        # MEMORY FIX v3: Iterative backward OR legacy accumulation
        # If scaler provided: backward immediately, free graph (O(1) memory)
        # If no scaler: accumulate losses in list (O(N) memory, legacy mode)
        holdout_losses = [] if not use_iterative_backward else None
        
        # Cap the number of holdout passes to prevent memory explosion
        # ARC tasks can have up to 5+ pairs, but we limit to max_loo_pairs (default 4)
        # This keeps memory usage predictable
        max_holdouts = min(N, self.config.max_loo_pairs)
        
        for holdout_idx in range(max_holdouts):
            # Create mask for valid holdouts
            valid_holdout_mask = holdout_idx < num_valid_pairs  # (B,)
            
            if not valid_holdout_mask.any():
                continue
            
            # Get remaining indices
            remaining_indices = [i for i in range(N) if i != holdout_idx]
            
            # Get remaining features (detached to avoid graph accumulation)
            remaining_features = all_support_features[:, remaining_indices].clone()  # (B, N-1, D, Hs, Ws)
            remaining_features.requires_grad_(True)  # Re-enable gradients for HyperLoRA
            
            # Predict LoRA deltas from remaining-pair context
            lora_deltas = model.hyper_lora(remaining_features)
            
            # Get held-out input and target
            holdout_input = input_grids[:, holdout_idx]  # (B, H, W)
            holdout_target = output_grids[:, holdout_idx]  # (B, H, W)
            
            # Forward pass with LoRA weights
            logits = model.forward_with_lora(
                holdout_input,
                remaining_features,
                lora_deltas,
            )  # (B, num_classes, H, W)
            
            # Compute loss with masking for invalid samples AND padding pixels
            target_for_loss = holdout_target.clone().long()
            target_for_loss[~valid_holdout_mask] = -100  # Ignore invalid samples
            target_for_loss[holdout_target == 10] = -100  # Ignore padding pixels (PAD_COLOR)
            
            ce_loss = F.cross_entropy(logits, target_for_loss, ignore_index=-100, reduction='mean')
            
            # Track accuracy (no gradients needed)
            with torch.no_grad():
                preds = logits.argmax(dim=1)
                sample_mask = valid_holdout_mask.view(B, 1, 1).expand_as(holdout_target)
                pixel_mask = (holdout_target != -100) & (holdout_target != 10)
                combined_mask = sample_mask & pixel_mask
                
                correct = ((preds == holdout_target) & combined_mask).sum().item()
                total_correct += correct
                total_pixels += combined_mask.sum().item()
            
            num_holdouts += 1
            
            # MEMORY FIX v3: Iterative backward - call backward IMMEDIATELY
            if use_iterative_backward:
                # Scale loss: weight * (loss / N) / grad_accumulation
                # We divide by N here because we're averaging over holdouts
                scaled_loss = (loss_weight * ce_loss / N) / grad_accumulation_steps
                total_loss_value += ce_loss.item()
                
                # Backward immediately - this frees the computation graph!
                scaler.scale(scaled_loss).backward()
                
                # Now we can safely delete everything - graph is freed
                del scaled_loss, ce_loss
            else:
                # Legacy mode: accumulate for later backward
                holdout_losses.append(ce_loss)
                total_loss_value += ce_loss.item()
            
            # MEMORY FIX: Clear intermediate tensors after each holdout
            del logits, lora_deltas, remaining_features, holdout_input, holdout_target
            del target_for_loss, preds, sample_mask, pixel_mask, combined_mask
        
        # Clean up support features
        del all_support_features
        
        # Compute final loss and accuracy
        accuracy = total_correct / (total_pixels + 1e-8) if total_pixels > 0 else 0.0
        
        if use_iterative_backward:
            # Iterative backward mode: backward already done, return float
            avg_loss_value = total_loss_value / num_holdouts if num_holdouts > 0 else 0.0
            
            return {
                'loo_loss': avg_loss_value,  # Float, not tensor (backward already done)
                'loo_accuracy': accuracy,
                'loo_num_holdouts': num_holdouts,
                'loo_skipped': False,
                'loo_iterative_backward': True,  # Signal that backward was done inside
            }
        else:
            # Legacy mode: return tensor for external backward
            if num_holdouts > 0:
                avg_loss = torch.stack(holdout_losses).mean()
            else:
                avg_loss = torch.tensor(0.0, device=device)
            
            # Clear holdout losses list
            del holdout_losses
            
            return {
                'loo_loss': avg_loss,  # Tensor for external backward
                'loo_accuracy': accuracy,
                'loo_num_holdouts': num_holdouts,
                'loo_skipped': False,
                'loo_iterative_backward': False,
            }
    
    def _forward_with_components(
        self,
        hyper_lora: nn.Module,
        rlan: nn.Module,
        support_inputs: torch.Tensor,  # (B, N, C, H, W)
        support_targets: torch.Tensor,  # (B, N, H, W)
        support_features: torch.Tensor,  # (B, N, D, Hs, Ws) - encoder features
        num_valid_pairs: Optional[torch.Tensor] = None,  # (B,) - actual pairs per sample
    ) -> Dict[str, Any]:
        """
        Compute LOO loss.
        
        Args:
            hyper_lora: HyperLoRA module for weight prediction
            rlan: RLAN model for inference
            support_inputs: Support set inputs
            support_targets: Support set targets
            support_features: Encoded support features
            num_valid_pairs: Number of valid pairs per batch sample
            
        Returns:
            Tuple of (loss, metrics_dict)
        """
        B, N, C, H, W = support_inputs.shape
        D = support_features.shape[2]
        Hs, Ws = support_features.shape[3], support_features.shape[4]
        
        # Determine valid pairs per sample
        if num_valid_pairs is None:
            num_valid_pairs = torch.full((B,), N, device=support_inputs.device)
        
        # Skip if not enough pairs for LOO
        if N < self.config.min_pairs_for_loo:
            return torch.tensor(0.0, device=support_inputs.device), {
                'loo_loss': 0.0,
                'loo_skipped': True,
                'loo_reason': f'Not enough pairs ({N} < {self.config.min_pairs_for_loo})'
            }
        
        total_loss = torch.tensor(0.0, device=support_inputs.device)
        total_correct = 0
        total_pixels = 0
        num_holdouts = 0
        
        # Cap the number of holdout passes to prevent memory explosion
        max_holdouts = min(N, self.config.max_loo_pairs)
        
        # For each possible holdout position
        for holdout_idx in range(max_holdouts):
            # Create mask for valid holdouts (position must be < num_valid_pairs)
            valid_holdout_mask = holdout_idx < num_valid_pairs  # (B,)
            
            if not valid_holdout_mask.any():
                continue
            
            # Get indices of remaining pairs (all except holdout)
            remaining_indices = [i for i in range(N) if i != holdout_idx]
            
            # Pool context from remaining pairs only
            # support_features: (B, N, D, Hs, Ws)
            remaining_features = support_features[:, remaining_indices]  # (B, N-1, D, Hs, Ws)
            
            # Predict LoRA deltas from remaining-pair features
            # HyperLoRA expects (B, N, D, H, W) and internally pools to context
            lora_deltas = hyper_lora(remaining_features)
            
            # Get the held-out input
            holdout_input = support_inputs[:, holdout_idx]  # (B, C, H, W)
            # Squeeze channel dim for forward_with_lora which expects (B, H, W)
            holdout_input = holdout_input.squeeze(1)  # (B, H, W)
            holdout_target = support_targets[:, holdout_idx]  # (B, H, W)
            
            # CRITICAL FIX (Dec 2025): Use remaining_features for cross-attention
            # to prevent data leakage. The held-out pair's output must NOT be
            # visible to the model during LOO training, otherwise it can "cheat"
            # by attending to the answer via CrossAttentionInjector.
            #
            # Why this simulates inference correctly:
            # - At inference, we have N training pairs + 1 test INPUT (no test output)
            # - LOO simulates this: N-1 pairs + 1 held-out INPUT (no held-out output)
            # - The model must generalize from patterns, not copy answers
            logits = rlan.forward_with_lora(
                holdout_input,
                remaining_features,  # Only N-1 pairs - no data leakage!
                lora_deltas,
            )  # (B, num_classes, H, W)
            
            # Compute loss on held-out pair (masked by valid holdouts and -100 padding)
            # CRITICAL: Use ignore_index=-100 to handle padded targets properly
            # First create a target tensor with -100 for invalid samples
            target_for_loss = holdout_target.clone()
            # Mark invalid samples' targets as -100 so they're ignored
            target_for_loss[~valid_holdout_mask] = -100
            # CRITICAL FIX: Also mask out PAD_COLOR=10 pixels (spatial padding)
            # support_targets uses 10 for padding, not -100, because they're also
            # used for ContextEncoder which needs to distinguish from black (0)
            target_for_loss[holdout_target == 10] = -100
            
            ce_loss = F.cross_entropy(logits, target_for_loss, ignore_index=-100, reduction='mean')
            total_loss = total_loss + ce_loss
            
            # Track accuracy for metrics (only on valid pixels)
            with torch.no_grad():
                preds = logits.argmax(dim=1)  # (B, H, W)
                # Create pixel-level valid mask: both valid holdout sample AND valid pixel (not -100 or 10)
                sample_mask = valid_holdout_mask.view(B, 1, 1).expand_as(holdout_target)
                pixel_mask = (holdout_target != -100) & (holdout_target != 10)
                combined_mask = sample_mask & pixel_mask
                
                correct = ((preds == holdout_target) & combined_mask).sum().item()
                total_correct += correct
                total_pixels += combined_mask.sum().item()
            
            num_holdouts += 1
        
        # Average loss over holdouts
        if num_holdouts > 0:
            avg_loss = total_loss / num_holdouts
            accuracy = total_correct / (total_pixels + 1e-8)
        else:
            avg_loss = torch.tensor(0.0, device=support_inputs.device)
            accuracy = 0.0
        
        metrics = {
            'loo_loss': avg_loss,  # Keep as tensor for backward compat
            'loo_accuracy': accuracy,
            'loo_num_holdouts': num_holdouts,
            'loo_skipped': False,
        }
        
        return metrics


class AugmentationEquivarianceLoss(nn.Module):
    """
    Augmentation Equivariance Loss.
    
    This loss enforces that the HyperLoRA weight predictions are consistent
    across augmented versions of the same task. The key insight is that
    a rotation/flip of the entire task should produce equivalent weights.
    
    For each task:
    1. Apply augmentation to all input-output pairs
    2. Compute context and predict LoRA weights for original and augmented
    3. Inverse-transform the augmented weights to align with original space
    4. Compute L2 difference between weight predictions
    
    Args:
        config: Equivariance configuration
        hidden_dim: Hidden dimension of the model
    """
    
    def __init__(self, config: EquivarianceConfig, hidden_dim: int = 256):
        super().__init__()
        self.config = config
        self.hidden_dim = hidden_dim
        
    def apply_augmentation(
        self,
        tensor: torch.Tensor,
        aug_type: str,
        inverse: bool = False
    ) -> torch.Tensor:
        """
        Apply or inverse-apply an augmentation.
        
        Args:
            tensor: Input tensor (B, C, H, W) or (B, H, W)
            aug_type: Type of augmentation
            inverse: If True, apply inverse transform
            
        Returns:
            Augmented tensor
        """
        is_3d = tensor.dim() == 3
        if is_3d:
            tensor = tensor.unsqueeze(1)
        
        if aug_type == 'rotate_90':
            if inverse:
                result = torch.rot90(tensor, k=-1, dims=(2, 3))
            else:
                result = torch.rot90(tensor, k=1, dims=(2, 3))
        elif aug_type == 'rotate_180':
            result = torch.rot90(tensor, k=2, dims=(2, 3))
        elif aug_type == 'rotate_270':
            if inverse:
                result = torch.rot90(tensor, k=-3, dims=(2, 3))
            else:
                result = torch.rot90(tensor, k=3, dims=(2, 3))
        elif aug_type == 'flip_h':
            result = torch.flip(tensor, dims=[3])
        elif aug_type == 'flip_v':
            result = torch.flip(tensor, dims=[2])
        else:
            result = tensor
            
        if is_3d:
            result = result.squeeze(1)
            
        return result
    
    def forward(
        self,
        hyper_lora: nn.Module,
        original_context: torch.Tensor,  # (B, D)
        augmented_contexts: Dict[str, torch.Tensor],  # aug_name -> (B, D)
        scaler: Any = None,  # GradScaler for iterative backward
        loss_weight: float = 1.0,
        grad_accumulation_steps: int = 1,
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Compute augmentation equivariance loss.
        
        MEMORY OPTIMIZATION (v3 - Iterative Backward):
        If scaler= is provided, backward() is called INSIDE the loop after each
        augmentation, and the computation graph is freed immediately.
        
        Args:
            hyper_lora: HyperLoRA module for weight prediction
            original_context: Context vector from original task
            augmented_contexts: Context vectors from augmented tasks
            scaler: GradScaler for AMP - if provided, backward happens inside
            loss_weight: Weight to apply before backward
            grad_accumulation_steps: Divide loss by this for accumulation
            
        Returns:
            Tuple of (loss, metrics_dict)
            If scaler is provided, loss is a float (backward already done).
        """
        if not self.config.enabled or len(augmented_contexts) == 0:
            return torch.tensor(0.0, device=original_context.device), {
                'equivariance_loss': 0.0,
                'equivariance_skipped': True,
            }
        
        use_iterative_backward = scaler is not None
        num_augs = len(augmented_contexts)
        
        # MEMORY FIX: Compute original deltas and DETACH for comparison
        # Only the augmented context computation needs gradients
        with torch.no_grad():
            original_deltas = hyper_lora.compute_delta_w(original_context)
            # Detach all original deltas - we just need their values for comparison
            original_deltas = {k: v.detach() for k, v in original_deltas.items()}
        
        # MEMORY FIX v3: Iterative backward OR legacy accumulation
        aug_losses = [] if not use_iterative_backward else None
        per_aug_loss = {}
        total_loss_value = 0.0
        
        for aug_name, aug_context in augmented_contexts.items():
            # Get augmented LoRA predictions (this is where gradients flow)
            aug_deltas = hyper_lora.compute_delta_w(aug_context)
            
            # BUG FIX #8: Use both magnitude AND direction comparison
            # The original norm-only comparison was too weak - weights could be
            # completely different but have the same norm.
            aug_loss = torch.tensor(0.0, device=original_context.device)
            
            for key in original_deltas:
                orig = original_deltas[key]  # (B, out, in) - detached
                aug = aug_deltas[key]  # (B, out, in) - has gradients
                
                # Flatten for comparison
                orig_flat = orig.view(orig.shape[0], -1)  # (B, out*in)
                aug_flat = aug.view(aug.shape[0], -1)  # (B, out*in)
                
                # 1. Magnitude constraint (weak): norms should be similar
                orig_norm = torch.linalg.norm(orig_flat, dim=1)
                aug_norm = torch.linalg.norm(aug_flat, dim=1)
                norm_diff = (orig_norm - aug_norm).pow(2).mean()
                
                # 2. Direction constraint (stronger): cosine similarity should be high
                # Normalize to unit vectors
                orig_unit = orig_flat / (orig_norm.unsqueeze(1) + 1e-8)
                aug_unit = aug_flat / (aug_norm.unsqueeze(1) + 1e-8)
                
                # Cosine similarity: 1.0 = identical direction, 0.0 = orthogonal
                cosine_sim = (orig_unit * aug_unit).sum(dim=1)  # (B,)
                # Loss: 1 - cosine_similarity (want to maximize similarity)
                direction_loss = (1.0 - cosine_sim).mean()
                
                # Combined: weight direction more heavily since it's more informative
                # norm_weight=0.3, direction_weight=0.7
                aug_loss = aug_loss + 0.3 * norm_diff + 0.7 * direction_loss
                
                # MEMORY FIX: Clear intermediate tensors per-key
                del orig, aug, orig_flat, aug_flat, orig_norm, aug_norm
                del orig_unit, aug_unit, cosine_sim, norm_diff, direction_loss
            
            per_aug_loss[aug_name] = aug_loss.item()
            total_loss_value += aug_loss.item()
            
            # MEMORY FIX v3: Iterative backward - call backward IMMEDIATELY
            if use_iterative_backward:
                # Scale loss: weight * (loss / num_augs) / grad_accumulation
                scaled_loss = (loss_weight * aug_loss / num_augs) / grad_accumulation_steps
                
                # Backward immediately - this frees the computation graph!
                scaler.scale(scaled_loss).backward()
                
                # Now we can safely delete - graph is freed
                del scaled_loss, aug_loss
            else:
                # Legacy mode: accumulate for later backward
                aug_losses.append(aug_loss)
            
            # MEMORY FIX: Clear augmented deltas after each aug
            del aug_deltas
        
        # Clean up
        del original_deltas
        
        if use_iterative_backward:
            # Iterative backward mode: return float
            avg_loss_value = total_loss_value / num_augs if num_augs > 0 else 0.0
            
            metrics = {
                'equivariance_loss': avg_loss_value,
                'equivariance_per_aug': per_aug_loss,
                'equivariance_num_augs': len(per_aug_loss),
                'equivariance_skipped': False,
                'equivariance_iterative_backward': True,
            }
            
            return avg_loss_value, metrics  # Float, not tensor
        else:
            # Legacy mode: return tensor
            if aug_losses:
                avg_loss = torch.stack(aug_losses).mean()
            else:
                avg_loss = torch.tensor(0.0, device=original_context.device)
            
            del aug_losses
            
            metrics = {
                'equivariance_loss': avg_loss.item(),
                'equivariance_per_aug': per_aug_loss,
                'equivariance_num_augs': len(per_aug_loss),
                'equivariance_skipped': False,
                'equivariance_iterative_backward': False,
            }
            
            return avg_loss, metrics  # Tensor for external backward


class CombinedMetaLoss(nn.Module):
    """
    Combined Meta-Learning Loss for RLAN HyperLoRA training.
    
    Orchestrates:
    - Standard task loss (cross-entropy on predictions)
    - LOO loss (generalization from N-1 to Nth example)
    - Equivariance loss (consistency across augmentations)
    
    Args:
        loo_config: LOO training configuration
        equiv_config: Equivariance training configuration
        hidden_dim: Hidden dimension of the model
    """
    
    def __init__(
        self,
        loo_config: Optional[LOOConfig] = None,
        equiv_config: Optional[EquivarianceConfig] = None,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.loo_config = loo_config or LOOConfig()
        self.equiv_config = equiv_config or EquivarianceConfig()
        self.hidden_dim = hidden_dim
        
        if self.loo_config.enabled:
            self.loo_loss = LOOTrainingLoss(self.loo_config, hidden_dim)
        else:
            self.loo_loss = None
            
        if self.equiv_config.enabled:
            self.equiv_loss = AugmentationEquivarianceLoss(self.equiv_config, hidden_dim)
        else:
            self.equiv_loss = None
    
    def forward(
        self,
        task_loss: torch.Tensor,
        hyper_lora: Optional[nn.Module] = None,
        rlan: Optional[nn.Module] = None,
        support_inputs: Optional[torch.Tensor] = None,
        support_targets: Optional[torch.Tensor] = None,
        support_features: Optional[torch.Tensor] = None,
        original_context: Optional[torch.Tensor] = None,
        augmented_contexts: Optional[Dict[str, torch.Tensor]] = None,
        num_valid_pairs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute combined meta-learning loss.
        
        Args:
            task_loss: Standard task loss (cross-entropy)
            hyper_lora: HyperLoRA module
            rlan: RLAN model
            support_inputs: Support set inputs for LOO
            support_targets: Support set targets for LOO
            support_features: Encoded support features for LOO
            original_context: Original context for equivariance
            augmented_contexts: Augmented contexts for equivariance
            num_valid_pairs: Number of valid pairs per sample
            
        Returns:
            Tuple of (total_loss, metrics_dict)
        """
        metrics = {
            'task_loss': task_loss.item(),
        }
        
        # Start with task loss
        total_loss = task_loss
        
        # Add LOO loss if enabled and inputs provided
        loo_loss_value = torch.tensor(0.0, device=task_loss.device)
        if (self.loo_loss is not None and 
            hyper_lora is not None and 
            rlan is not None and
            support_inputs is not None and 
            support_features is not None):
            
            loo_loss_value, loo_metrics = self.loo_loss(
                hyper_lora=hyper_lora,
                rlan=rlan,
                support_inputs=support_inputs,
                support_targets=support_targets,
                support_features=support_features,
                num_valid_pairs=num_valid_pairs,
            )
            metrics.update(loo_metrics)
            total_loss = total_loss + self.loo_config.loss_weight * loo_loss_value
        
        # Add equivariance loss if enabled and contexts provided
        equiv_loss_value = torch.tensor(0.0, device=task_loss.device)
        if (self.equiv_loss is not None and 
            hyper_lora is not None and
            original_context is not None and 
            augmented_contexts is not None):
            
            equiv_loss_value, equiv_metrics = self.equiv_loss(
                hyper_lora=hyper_lora,
                original_context=original_context,
                augmented_contexts=augmented_contexts,
            )
            metrics.update(equiv_metrics)
            total_loss = total_loss + self.equiv_config.loss_weight * equiv_loss_value
        
        metrics['total_loss'] = total_loss.item()
        metrics['loo_loss_weighted'] = (self.loo_config.loss_weight * loo_loss_value).item()
        metrics['equiv_loss_weighted'] = (self.equiv_config.loss_weight * equiv_loss_value).item()
        
        return total_loss, metrics


def create_augmented_contexts(
    encoder: nn.Module,
    support_inputs: torch.Tensor,  # (B, N, C, H, W)
    support_targets: torch.Tensor,  # (B, N, H, W)
    augmentations: List[str],
    num_augmentations: int = 4,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Tuple[torch.Tensor, torch.Tensor]]]:
    """
    Create augmented versions of the support set and compute their contexts.
    
    Args:
        encoder: Encoder module for computing features
        support_inputs: Original support inputs
        support_targets: Original support targets
        augmentations: List of possible augmentations
        num_augmentations: Number of augmentations to apply
        
    Returns:
        Tuple of (augmented_contexts, augmented_pairs)
        - augmented_contexts: Dict[aug_name, context_tensor]
        - augmented_pairs: Dict[aug_name, (inputs, targets)]
    """
    import random
    
    B, N, C, H, W = support_inputs.shape
    device = support_inputs.device
    
    # Select random subset of augmentations
    selected_augs = random.sample(
        augmentations, 
        min(num_augmentations, len(augmentations))
    )
    
    augmented_contexts = {}
    augmented_pairs = {}
    aug_equivariance = AugmentationEquivarianceLoss(EquivarianceConfig())
    
    for aug_name in selected_augs:
        # Apply augmentation to all pairs
        aug_inputs = torch.stack([
            aug_equivariance.apply_augmentation(support_inputs[:, i], aug_name)
            for i in range(N)
        ], dim=1)  # (B, N, C, H, W)
        
        aug_targets = torch.stack([
            aug_equivariance.apply_augmentation(support_targets[:, i], aug_name)
            for i in range(N)
        ], dim=1)  # (B, N, H, W)
        
        # Compute features for augmented inputs
        # Flatten for encoder: (B*N, C, H, W)
        aug_inputs_flat = aug_inputs.view(B * N, C, H, W)
        
        with torch.no_grad():
            aug_features = encoder(aug_inputs_flat)  # (B*N, D, Hs, Ws)
        
        # Reshape and pool to get context
        D, Hs, Ws = aug_features.shape[1:]
        aug_features = aug_features.view(B, N, D, Hs, Ws)
        context = aug_features.mean(dim=(1, 3, 4))  # (B, D)
        
        augmented_contexts[aug_name] = context
        augmented_pairs[aug_name] = (aug_inputs, aug_targets)
    
    return augmented_contexts, augmented_pairs


from sci_arc.models.rlan_modules.acw import AugmentedConfidenceWeighting
