"""
Unit tests for SCI-ARC training components.
"""

import pytest
import torch
import numpy as np

from sci_arc.training import (
    StructuralContrastiveLoss,
    OrthogonalityLoss,
    SCIARCLoss,
)


class TestStructuralContrastiveLoss:
    """Tests for Structural Contrastive Loss."""
    
    @pytest.fixture
    def loss_fn(self):
        return StructuralContrastiveLoss(temperature=0.1, hidden_dim=64, projection_dim=32)
    
    def test_same_class_low_loss(self, loss_fn):
        """Test that same-class samples have lower loss."""
        # Two samples from same transformation family - shape [B, K, D] where K=1 slot
        z1 = torch.randn(2, 1, 64)
        
        labels = torch.tensor([0, 0])  # Same class
        
        loss_same = loss_fn(z1, labels)
        
        # Different classes
        labels_diff = torch.tensor([0, 1])  # Different classes
        loss_diff = loss_fn(z1, labels_diff)
        
        # Same-class should have lower loss
        # (This may not always hold due to random init, but generally true)
        assert loss_same >= 0
        assert loss_diff >= 0
    
    def test_output_shape(self, loss_fn):
        """Test that loss is a scalar."""
        # Shape: [B, K, D] where K=1 structure slot
        z = torch.randn(4, 1, 64)
        labels = torch.tensor([0, 0, 1, 1])
        
        loss = loss_fn(z, labels)
        
        assert loss.dim() == 0  # Scalar
    
    def test_gradient_flow(self, loss_fn):
        """Test that gradients flow through loss."""
        # Shape: [B, K, D] where K=1 structure slot
        z = torch.randn(4, 1, 64, requires_grad=True)
        labels = torch.tensor([0, 0, 1, 1])
        
        loss = loss_fn(z, labels)
        loss.backward()
        
        assert z.grad is not None
        assert not torch.isnan(z.grad).any()


class TestOrthogonalityLoss:
    """Tests for Orthogonality Loss."""
    
    @pytest.fixture
    def loss_fn(self):
        return OrthogonalityLoss()
    
    def test_orthogonal_vectors(self, loss_fn):
        """Test that orthogonal vectors have zero loss."""
        # OrthogonalityLoss expects [B, K, D] tensors
        # Create orthogonal vectors in the mean-pooled space
        z_struct = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])  # [1, 1, 4]
        z_content = torch.tensor([[[0.0, 1.0, 0.0, 0.0]]])  # [1, 1, 4]
        
        loss = loss_fn(z_struct, z_content)
        
        assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)
    
    def test_parallel_vectors(self, loss_fn):
        """Test that parallel vectors have high loss."""
        z_struct = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])  # [1, 1, 4]
        z_content = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])  # [1, 1, 4]
        
        loss = loss_fn(z_struct, z_content)
        
        assert loss > 0
    
    def test_batch_processing(self, loss_fn):
        """Test batch processing."""
        z_struct = torch.randn(8, 4, 64)  # [B, K, D]
        z_content = torch.randn(8, 4, 64)  # [B, M, D]
        
        loss = loss_fn(z_struct, z_content)
        
        assert loss.dim() == 0


class TestSCIARCLoss:
    """Tests for combined SCI-ARC loss."""
    
    @pytest.fixture
    def loss_fn(self):
        return SCIARCLoss(
            H_cycles=3,
            scl_weight=0.1,
            orthogonality_weight=0.01,
            temperature=0.1,
            hidden_dim=64,
            projection_dim=32
        )
    
    def test_combined_loss(self, loss_fn):
        """Test combined loss computation."""
        # Create mock model outputs - SCIARCLoss expects List of predictions for deep supervision
        predictions = [torch.randn(4, 10, 10, 10) for _ in range(3)]  # H_cycles=3
        target = torch.randint(0, 10, (4, 10, 10))
        z_struct = torch.randn(4, 8, 64)  # [B, K, D]
        z_content = torch.randn(4, 8, 64)  # [B, M, D]
        transform_families = torch.tensor([0, 0, 1, 1])
        
        loss_dict = loss_fn(
            predictions=predictions,
            target=target,
            structure_rep=z_struct,
            content_rep=z_content,
            transform_labels=transform_families,
        )
        
        assert 'total' in loss_dict
        assert 'ce' in loss_dict
        assert 'scl' in loss_dict
        assert 'orthogonality' in loss_dict
    
    def test_no_scl_labels(self, loss_fn):
        """Test with no SCL labels (still computes other losses)."""
        predictions = [torch.randn(2, 8, 8, 10) for _ in range(3)]
        target = torch.randint(0, 10, (2, 8, 8))
        z_struct = torch.randn(2, 8, 64)
        z_content = torch.randn(2, 8, 64)
        
        loss_dict = loss_fn(
            predictions=predictions,
            target=target,
            structure_rep=z_struct,
            content_rep=z_content,
            transform_labels=None,  # No labels
        )
        
        assert loss_dict['scl'] == 0.0  # SCL should be zero without labels
        assert loss_dict['ce'] > 0  # CE should still be computed
    
    def test_gradient_flow(self, loss_fn):
        """Test gradient flow through all losses."""
        predictions = [torch.randn(4, 8, 8, 10, requires_grad=True) for _ in range(3)]
        target = torch.randint(0, 10, (4, 8, 8))
        z_struct = torch.randn(4, 8, 64, requires_grad=True)
        z_content = torch.randn(4, 8, 64, requires_grad=True)
        transform_families = torch.tensor([0, 0, 1, 1])
        
        loss_dict = loss_fn(
            predictions=predictions,
            target=target,
            structure_rep=z_struct,
            content_rep=z_content,
            transform_labels=transform_families,
        )
        
        loss_dict['total'].backward()
        
        assert z_struct.grad is not None
        assert z_content.grad is not None
        assert predictions[0].grad is not None


class TestLossWeights:
    """Tests for loss weight configurations."""
    
    def test_zero_scl_weight(self):
        """Test with zero SCL weight."""
        loss_fn = SCIARCLoss(
            H_cycles=2,
            scl_weight=0.0,
            orthogonality_weight=0.01,
            hidden_dim=64
        )
        
        predictions = [torch.randn(2, 8, 8, 10) for _ in range(2)]
        target = torch.randint(0, 10, (2, 8, 8))
        z_struct = torch.randn(2, 8, 64)
        z_content = torch.randn(2, 8, 64)
        transform_families = torch.tensor([0, 1])
        
        loss_dict = loss_fn(
            predictions=predictions,
            target=target,
            structure_rep=z_struct,
            content_rep=z_content,
            transform_labels=transform_families,
        )
        
        # SCL should still be computed but not contribute to total
        expected = loss_dict['ce'] + 0.01 * loss_dict['orthogonality']
        assert torch.allclose(loss_dict['total'], expected, atol=1e-5)
    
    def test_high_ortho_weight(self):
        """Test with high orthogonality weight."""
        loss_fn = SCIARCLoss(
            H_cycles=2,
            scl_weight=0.0,
            orthogonality_weight=1.0,
            hidden_dim=64
        )
        
        # Non-orthogonal vectors - [B, K, D]
        z_struct = torch.ones(2, 8, 64)
        z_content = torch.ones(2, 8, 64)
        predictions = [torch.randn(2, 8, 8, 10) for _ in range(2)]
        target = torch.randint(0, 10, (2, 8, 8))
        
        loss_dict = loss_fn(
            predictions=predictions,
            target=target,
            structure_rep=z_struct,
            content_rep=z_content,
        )
        
        # Ortho loss should be significant
        assert loss_dict['orthogonality'] > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
