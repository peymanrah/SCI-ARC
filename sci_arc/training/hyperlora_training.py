"""
HyperLoRA Training Integration Module.

This module provides training utilities for integrating HyperLoRA
(meta-learning weight adaptation) into the RLAN training loop.

The integration is designed to be backward-compatible:
- When HyperLoRA is disabled, training proceeds as normal
- When enabled, LOO loss and equivariance loss are added

Usage:
    from sci_arc.training.hyperlora_training import HyperLoRATrainer
    
    trainer = HyperLoRATrainer(model, config)
    
    # In training loop:
    if trainer.enabled:
        hyperlora_losses = trainer.compute_losses(
            support_inputs, support_targets, support_features
        )
        total_loss = task_loss + hyperlora_losses['total']
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from sci_arc.training.rlan_loss import log_stablemax


@dataclass
class HyperLoRATrainingConfig:
    """Configuration for HyperLoRA training."""
    enabled: bool = False
    
    # LOO training settings
    loo_enabled: bool = True
    loo_loss_weight: float = 0.5
    min_pairs_for_loo: int = 2
    
    # Equivariance training settings
    equivariance_enabled: bool = True
    equivariance_loss_weight: float = 0.1
    num_augmentations: int = 4
    
    @classmethod
    def from_config(cls, config: dict) -> "HyperLoRATrainingConfig":
        """Create from config dict."""
        model_cfg = config.get('model', {})
        training_cfg = config.get('training', {})
        loo_cfg = training_cfg.get('loo_training', {})
        equiv_cfg = training_cfg.get('equivariance_training', {})
        
        return cls(
            enabled=model_cfg.get('use_hyperlora', False),
            loo_enabled=loo_cfg.get('enabled', True),
            loo_loss_weight=loo_cfg.get('loss_weight', 0.5),
            min_pairs_for_loo=loo_cfg.get('min_pairs_for_loo', 2),
            equivariance_enabled=equiv_cfg.get('enabled', True),
            equivariance_loss_weight=equiv_cfg.get('loss_weight', 0.1),
            num_augmentations=equiv_cfg.get('num_augmentations', 4),
        )


class HyperLoRATrainer:
    """
    Training helper for HyperLoRA integration.
    
    This class wraps the RLAN model and provides:
    1. LOO loss computation
    2. Equivariance loss computation
    3. Health metrics tracking
    4. Automatic integration with existing loss
    """
    
    def __init__(
        self,
        model: nn.Module,  # RLAN model
        config: HyperLoRATrainingConfig,
        device: torch.device = torch.device('cuda'),
    ):
        self.model = model
        self.config = config
        self.device = device
        
        # Check if model has HyperLoRA
        self.enabled = config.enabled and hasattr(model, 'hyper_lora') and model.hyper_lora is not None
        
        # Warn if LOO/equivariance are configured but HyperLoRA is disabled
        if not self.enabled and (config.loo_enabled or config.equivariance_enabled):
            import warnings
            warnings.warn(
                f"[HyperLoRA] Config has loo_enabled={config.loo_enabled}, "
                f"equivariance_enabled={config.equivariance_enabled}, but HyperLoRA is disabled "
                f"(use_hyperlora={config.enabled}). These settings will have no effect.",
                UserWarning
            )
        
        if self.enabled:
            print(f"[HyperLoRA] Training enabled with:")
            print(f"  - LOO loss weight: {config.loo_loss_weight}")
            print(f"  - Equivariance loss weight: {config.equivariance_loss_weight}")
            print(f"  - Min pairs for LOO: {config.min_pairs_for_loo}")
    
    def compute_loo_loss(
        self,
        support_inputs: torch.Tensor,  # (B, N, H, W)
        support_targets: torch.Tensor,  # (B, N, H, W)
        support_features: torch.Tensor,  # (B, N, D, Hs, Ws)
        num_valid_pairs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute Leave-One-Out loss.
        
        For each batch sample with N pairs:
        1. Hold out pair i
        2. Pool context from remaining N-1 pairs
        3. Predict held-out pair using LoRA weights from that context
        4. Compute loss on held-out prediction
        
        Args:
            support_inputs: Support set inputs (B, N, H, W)
            support_targets: Support set targets (B, N, H, W)
            support_features: Encoded support features (B, N, D, Hs, Ws)
            num_valid_pairs: Number of valid pairs per sample (B,)
            
        Returns:
            Tuple of (loss, metrics_dict)
        """
        B, N, H, W = support_inputs.shape
        D = support_features.shape[2]
        
        if N < self.config.min_pairs_for_loo:
            # Not enough pairs for LOO
            return torch.tensor(0.0, device=self.device), {
                'loo_loss': 0.0,
                'loo_accuracy': 0.0,
                'loo_skipped': True,
            }
        
        if num_valid_pairs is None:
            num_valid_pairs = torch.full((B,), N, device=self.device)
        
        total_loss = 0.0
        total_correct = 0
        total_pixels = 0
        num_holdouts = 0
        
        for holdout_idx in range(N):
            # Skip if this holdout is invalid for some samples
            valid_mask = num_valid_pairs > holdout_idx
            if not valid_mask.any():
                continue
            
            # Create mask excluding holdout
            pair_indices = torch.arange(N, device=self.device)
            keep_mask = pair_indices != holdout_idx  # (N,)
            
            # Get context features without holdout: (B, N-1, D, Hs, Ws)
            loo_features = support_features[:, keep_mask]
            
            # Predict LoRA weights from N-1 context
            lora_deltas = self.model.hyper_lora(loo_features)
            
            # Get held-out input and target
            holdout_input = support_inputs[:, holdout_idx]  # (B, H, W)
            holdout_target = support_targets[:, holdout_idx]  # (B, H, W)
            
            # Predict held-out pair using LoRA weights
            logits = self.model.forward_with_lora(
                holdout_input, loo_features, lora_deltas
            )
            
            # Compute loss (only for valid samples) using STABLEMAX for consistency
            # with main training loss (F.cross_entropy uses softmax, not stablemax)
            target_for_loss = holdout_target.clone()
            target_for_loss[~valid_mask] = -100  # Ignore invalid
            
            # Use stablemax-based loss (matching main training loss mode)
            B_curr, C, H_curr, W_curr = logits.shape
            logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)
            targets_flat = target_for_loss.reshape(-1)  # (B*H*W,)
            
            # Mask for valid positions (not -100)
            valid_pixel_mask = targets_flat != -100
            
            if valid_pixel_mask.any():
                logits_valid = logits_flat[valid_pixel_mask]
                targets_valid = targets_flat[valid_pixel_mask]
                
                # Compute log stablemax probabilities
                logprobs = log_stablemax(logits_valid.to(torch.float64), dim=-1)
                
                # Gather log probs for target classes
                prediction_logprobs = torch.gather(
                    logprobs,
                    index=targets_valid.unsqueeze(-1).to(torch.long),
                    dim=-1
                ).squeeze(-1)
                
                # Negative log likelihood
                loss = -prediction_logprobs.to(logits.dtype).mean()
            else:
                loss = torch.tensor(0.0, device=self.device)
            
            total_loss = total_loss + loss
            num_holdouts += 1
            
            # Track accuracy (only on valid pixels, excluding -100 padding)
            preds = logits.argmax(dim=1)
            # Create combined mask: valid sample AND valid pixel (not -100 padding)
            sample_mask = valid_mask.view(-1, 1, 1).expand_as(holdout_target)
            pixel_mask = (holdout_target != -100)
            combined_mask = sample_mask & pixel_mask
            
            correct = ((preds == holdout_target) & combined_mask).sum().item()
            total_correct += correct
            total_pixels += combined_mask.sum().item()  # Only count valid pixels
        
        if num_holdouts > 0:
            loo_loss = total_loss / num_holdouts
            loo_accuracy = total_correct / max(total_pixels, 1)
        else:
            loo_loss = torch.tensor(0.0, device=self.device)
            loo_accuracy = 0.0
        
        return loo_loss, {
            'loo_loss': loo_loss.item() if torch.is_tensor(loo_loss) else loo_loss,
            'loo_accuracy': loo_accuracy,
            'loo_num_holdouts': num_holdouts,
            'loo_skipped': num_holdouts == 0,
        }
    
    def compute_equivariance_loss(
        self,
        support_features: torch.Tensor,  # (B, N, D, Hs, Ws)
        original_lora_deltas: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute augmentation equivariance loss.
        
        This encourages the model to predict similar LoRA weights
        for augmented versions of the same task.
        
        Args:
            support_features: Original support features
            original_lora_deltas: LoRA deltas from original context
            
        Returns:
            Tuple of (loss, metrics_dict)
        """
        # Apply random augmentation to features
        # For simplicity, we use spatial transforms on features
        B, N, D, Hs, Ws = support_features.shape
        
        total_loss = 0.0
        num_augs = 0
        
        augmentations = ['rotate_90', 'rotate_180', 'rotate_270', 'flip_h']
        selected_augs = augmentations[:self.config.num_augmentations]
        
        for aug_type in selected_augs:
            # Apply augmentation to features
            aug_features = self._augment_features(support_features, aug_type)
            
            # Predict LoRA from augmented features
            aug_lora_deltas = self.model.hyper_lora(aug_features)
            
            # FIXED: Compare actual weight matrices, not just their norms!
            # The old norm comparison allowed w1=[1,0] and w2=[0,1] to appear equal
            # even though they represent completely different adaptations.
            # 
            # We use MSE between weight matrices normalized by their dimension
            # to encourage the SAME adaptation for geometrically equivalent tasks.
            for name in ['gru_reset', 'gru_update', 'gru_candidate', 'output_head']:
                orig_weights = original_lora_deltas[name]  # (B, in_dim, out_dim)
                aug_weights = aug_lora_deltas[name]  # (B, in_dim, out_dim)
                
                # MSE loss on actual weight values (not norms)
                # This ensures equivalent tasks produce equivalent adaptations
                loss = F.mse_loss(orig_weights, aug_weights)
                total_loss = total_loss + loss
            
            num_augs += 1
        
        if num_augs > 0:
            equiv_loss = total_loss / (num_augs * 4)  # 4 layer types
        else:
            equiv_loss = torch.tensor(0.0, device=self.device)
        
        return equiv_loss, {
            'equivariance_loss': equiv_loss.item() if torch.is_tensor(equiv_loss) else equiv_loss,
            'equivariance_num_augs': num_augs,
        }
    
    def _augment_features(
        self,
        features: torch.Tensor,  # (B, N, D, H, W)
        aug_type: str,
    ) -> torch.Tensor:
        """Apply spatial augmentation to features."""
        if aug_type == 'rotate_90':
            return torch.rot90(features, k=1, dims=[-2, -1])
        elif aug_type == 'rotate_180':
            return torch.rot90(features, k=2, dims=[-2, -1])
        elif aug_type == 'rotate_270':
            return torch.rot90(features, k=3, dims=[-2, -1])
        elif aug_type == 'flip_h':
            return torch.flip(features, dims=[-1])
        elif aug_type == 'flip_v':
            return torch.flip(features, dims=[-2])
        else:
            return features
    
    def compute_losses(
        self,
        support_inputs: torch.Tensor,
        support_targets: torch.Tensor,
        support_features: torch.Tensor,
        num_valid_pairs: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Compute all HyperLoRA training losses.
        
        Args:
            support_inputs: Support set inputs
            support_targets: Support set targets  
            support_features: Encoded support features
            num_valid_pairs: Number of valid pairs per sample
            
        Returns:
            Dict with:
                - 'total': Total HyperLoRA loss (weighted sum)
                - 'loo': LOO loss (unweighted)
                - 'equivariance': Equivariance loss (unweighted)
                - 'metrics': Dict of detailed metrics
        """
        if not self.enabled:
            return {
                'total': torch.tensor(0.0, device=self.device),
                'loo': torch.tensor(0.0, device=self.device),
                'equivariance': torch.tensor(0.0, device=self.device),
                'metrics': {'hyperlora_enabled': False},
            }
        
        metrics = {'hyperlora_enabled': True}
        total_loss = torch.tensor(0.0, device=self.device)
        
        # First, predict LoRA from full context (for equivariance comparison)
        lora_deltas = self.model.hyper_lora(support_features)
        
        # Add health metrics
        from sci_arc.models.rlan_modules import compute_hyperlora_health_metrics
        health_metrics = compute_hyperlora_health_metrics(
            self.model.hyper_lora, lora_deltas
        )
        metrics.update(health_metrics)
        
        # LOO loss
        loo_loss = torch.tensor(0.0, device=self.device)
        if self.config.loo_enabled:
            loo_loss, loo_metrics = self.compute_loo_loss(
                support_inputs, support_targets, support_features, num_valid_pairs
            )
            metrics.update(loo_metrics)
            total_loss = total_loss + self.config.loo_loss_weight * loo_loss
        
        # Equivariance loss
        equiv_loss = torch.tensor(0.0, device=self.device)
        if self.config.equivariance_enabled:
            equiv_loss, equiv_metrics = self.compute_equivariance_loss(
                support_features, lora_deltas
            )
            metrics.update(equiv_metrics)
            total_loss = total_loss + self.config.equivariance_loss_weight * equiv_loss
        
        return {
            'total': total_loss,
            'loo': loo_loss,
            'equivariance': equiv_loss,
            'metrics': metrics,
        }
    
    def get_lora_deltas_for_inference(
        self,
        support_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Get LoRA deltas for inference (with sanity check).
        
        Args:
            support_features: Encoded support features
            
        Returns:
            LoRA deltas dict (or empty dict if not enabled)
        """
        if not self.enabled:
            return {}
        
        return self.model.hyper_lora(support_features)
