"""
Test Augmentation Equivariance Loss Integration.

Verifies that:
1. AugmentationEquivarianceLoss is correctly instantiated
2. HyperLoRA.compute_delta_w works with pre-pooled context
3. The loss computation produces valid gradients
4. Augmented contexts produce different LoRA predictions
"""

import pytest
import torch
import torch.nn as nn

from sci_arc.models.rlan_modules.hyper_lora import HyperLoRA, HyperLoRAConfig
from sci_arc.models.rlan_modules.loo_training import (
    AugmentationEquivarianceLoss,
    EquivarianceConfig,
)


class TestEquivarianceLossIntegration:
    """Test the equivariance loss integration."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.hidden_dim = 64
        self.batch_size = 2
        self.num_pairs = 3
        self.h, self.w = 8, 8
        self.device = torch.device('cpu')
        
    def test_hyperlora_compute_delta_w(self):
        """Test that HyperLoRA.compute_delta_w works with pre-pooled context."""
        config = HyperLoRAConfig(
            hidden_dim=self.hidden_dim,
            rank=4,
            scaling=1.0,
        )
        hyper_lora = HyperLoRA(config)
        
        # Create a context vector (what pool_context returns)
        context = torch.randn(self.batch_size, self.hidden_dim)
        
        # Compute deltas
        deltas = hyper_lora.compute_delta_w(context)
        
        # Verify output structure
        assert isinstance(deltas, dict)
        assert 'gru_reset' in deltas
        assert 'gru_update' in deltas
        assert 'gru_candidate' in deltas
        assert 'output_head' in deltas
        
        # Verify shapes
        for key, delta in deltas.items():
            assert delta.shape[0] == self.batch_size
            
    def test_equivariance_loss_forward(self):
        """Test that equivariance loss computes correctly."""
        config = HyperLoRAConfig(
            hidden_dim=self.hidden_dim,
            rank=4,
            scaling=1.0,
        )
        hyper_lora = HyperLoRA(config)
        
        equiv_config = EquivarianceConfig(
            enabled=True,
            loss_weight=0.1,
            num_augmentations=2,
        )
        equiv_loss = AugmentationEquivarianceLoss(equiv_config, self.hidden_dim)
        
        # Create original and augmented contexts
        original_context = torch.randn(self.batch_size, self.hidden_dim, requires_grad=True)
        augmented_contexts = {
            'rotate_90': torch.randn(self.batch_size, self.hidden_dim),
            'flip_h': torch.randn(self.batch_size, self.hidden_dim),
        }
        
        # Compute loss
        loss, metrics = equiv_loss(
            hyper_lora=hyper_lora,
            original_context=original_context,
            augmented_contexts=augmented_contexts,
        )
        
        # Verify output
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar
        assert loss.item() >= 0  # Loss should be non-negative
        assert isinstance(metrics, dict)
        assert 'equivariance_loss' in metrics
        assert 'equivariance_num_augs' in metrics
        assert metrics['equivariance_num_augs'] == 2
        
    def test_equivariance_loss_gradient_flow(self):
        """Test that gradients flow through equivariance loss to HyperLoRA."""
        config = HyperLoRAConfig(
            hidden_dim=self.hidden_dim,
            rank=4,
            scaling=1.0,
        )
        hyper_lora = HyperLoRA(config)
        
        equiv_config = EquivarianceConfig(
            enabled=True,
            loss_weight=0.1,
            num_augmentations=2,
        )
        equiv_loss = AugmentationEquivarianceLoss(equiv_config, self.hidden_dim)
        
        # Create contexts
        original_context = torch.randn(self.batch_size, self.hidden_dim, requires_grad=True)
        augmented_contexts = {
            'rotate_90': torch.randn(self.batch_size, self.hidden_dim),
        }
        
        # Compute loss
        loss, _ = equiv_loss(
            hyper_lora=hyper_lora,
            original_context=original_context,
            augmented_contexts=augmented_contexts,
        )
        
        # Backward
        loss.backward()
        
        # Check gradients exist on HyperLoRA parameters
        grad_count = 0
        for name, param in hyper_lora.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                grad_count += 1
                
        assert grad_count > 0, "No gradients flowed to HyperLoRA parameters"
        
    def test_apply_augmentation(self):
        """Test that apply_augmentation transforms tensors correctly."""
        equiv_config = EquivarianceConfig()
        equiv_loss = AugmentationEquivarianceLoss(equiv_config, self.hidden_dim)
        
        # Create a test tensor
        tensor = torch.arange(16).reshape(1, 1, 4, 4).float()
        
        # Test rotate_90
        rotated = equiv_loss.apply_augmentation(tensor, 'rotate_90')
        assert rotated.shape == tensor.shape
        assert not torch.equal(rotated, tensor)  # Should be different
        
        # Test that rotating 4 times returns to original
        result = tensor
        for _ in range(4):
            result = equiv_loss.apply_augmentation(result, 'rotate_90')
        assert torch.allclose(result, tensor)
        
        # Test flip_h
        flipped = equiv_loss.apply_augmentation(tensor, 'flip_h')
        assert flipped.shape == tensor.shape
        double_flipped = equiv_loss.apply_augmentation(flipped, 'flip_h')
        assert torch.equal(double_flipped, tensor)  # Double flip = identity
        
    def test_equivariance_similar_contexts_lower_loss(self):
        """Test that similar contexts produce lower equivariance loss."""
        config = HyperLoRAConfig(
            hidden_dim=self.hidden_dim,
            rank=4,
            scaling=1.0,
        )
        hyper_lora = HyperLoRA(config)
        
        equiv_config = EquivarianceConfig(
            enabled=True,
            loss_weight=1.0,
        )
        equiv_loss = AugmentationEquivarianceLoss(equiv_config, self.hidden_dim)
        
        original_context = torch.randn(self.batch_size, self.hidden_dim)
        
        # Similar contexts (small noise)
        similar_contexts = {
            'aug1': original_context + 0.01 * torch.randn_like(original_context),
        }
        
        # Very different contexts
        different_contexts = {
            'aug1': torch.randn(self.batch_size, self.hidden_dim) * 10,
        }
        
        loss_similar, _ = equiv_loss(hyper_lora, original_context, similar_contexts)
        loss_different, _ = equiv_loss(hyper_lora, original_context, different_contexts)
        
        # Similar contexts should produce lower loss
        assert loss_similar.item() < loss_different.item(), \
            f"Similar contexts should have lower loss: {loss_similar.item()} vs {loss_different.item()}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
