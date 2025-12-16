"""
Tests for CISL (Content-Invariant Structure Learning) logging capabilities.

These tests verify that:
1. CISLLoss tracks per-batch statistics correctly
2. Epoch-level aggregation works properly
3. Stats reset works between epochs
4. All new logging features are functional
"""
import pytest
import torch
from sci_arc.training import CISLLoss, CICLLoss


class TestCISLLogging:
    """Tests for CISL per-batch and epoch-level logging."""
    
    def test_stats_tracking_per_batch(self):
        """Test that stats are tracked for each batch."""
        loss_fn = CISLLoss(
            consist_weight=0.5,
            content_inv_weight=0.5,
            variance_weight=0.1,
            target_std=0.5,
            debug=True
        )
        
        # Process 3 batches
        for batch_idx in range(3):
            z_struct = torch.randn(4, 3, 64)
            z_aug = z_struct + 0.1 * torch.randn_like(z_struct)
            result = loss_fn(z_struct, z_aug)
            
            # Verify stats are in the result
            assert 'stats' in result
            stats = result['stats']
            assert 'z_mean' in stats
            assert 'z_std' in stats
            assert 'z_norm' in stats
            assert 'orig_aug_cos_sim' in stats
        
        # Verify epoch stats were accumulated
        epoch_stats = loss_fn.get_epoch_stats()
        assert epoch_stats['cisl/batches_processed'] == 3
        assert epoch_stats['cisl/consist_avg'] > 0
        assert epoch_stats['cisl/z_std_avg'] > 0
    
    def test_epoch_stats_reset(self):
        """Test that epoch stats reset correctly."""
        loss_fn = CISLLoss()
        
        # Process some batches
        for _ in range(5):
            z = torch.randn(4, 3, 64)
            loss_fn(z)
        
        assert loss_fn.get_epoch_stats()['cisl/batches_processed'] == 5
        
        # Reset
        loss_fn.reset_epoch_stats()
        
        # All stats should be zero/empty
        epoch_stats = loss_fn.get_epoch_stats()
        assert epoch_stats['cisl/batches_processed'] == 0
        assert epoch_stats['cisl/total_avg'] == 0.0
        assert epoch_stats['cisl/consist_avg'] == 0.0
    
    def test_epoch_stats_keys(self):
        """Test that all expected keys are in epoch stats."""
        loss_fn = CISLLoss()
        z = torch.randn(4, 3, 64)
        loss_fn(z)
        
        epoch_stats = loss_fn.get_epoch_stats()
        
        expected_keys = [
            'cisl/consist_avg',
            'cisl/content_inv_avg',
            'cisl/variance_avg',
            'cisl/total_avg',
            'cisl/z_mean_avg',
            'cisl/z_std_avg',
            'cisl/z_norm_avg',
            'cisl/batches_processed',
        ]
        
        for key in expected_keys:
            assert key in epoch_stats, f"Missing key: {key}"
    
    def test_cosine_similarity_with_augmentation(self):
        """Test that cosine similarity is computed when augmentation is provided."""
        loss_fn = CISLLoss(debug=True)
        
        # Similar embeddings should have high cosine similarity
        z_struct = torch.randn(4, 3, 64)
        z_aug = z_struct + 0.01 * torch.randn_like(z_struct)  # Very similar
        
        result = loss_fn(z_struct, z_aug)
        
        cos_sim = result['stats']['orig_aug_cos_sim']
        assert cos_sim > 0.99, f"Expected high cosine sim, got {cos_sim}"
    
    def test_cosine_similarity_without_augmentation(self):
        """Test that cosine similarity is 0 when no augmentation provided."""
        loss_fn = CISLLoss(debug=True)
        
        z_struct = torch.randn(4, 3, 64)
        result = loss_fn(z_struct, z_struct_content_aug=None)
        
        cos_sim = result['stats']['orig_aug_cos_sim']
        assert cos_sim == 0.0
    
    def test_backward_compatibility_alias(self):
        """Test that CICLLoss is still available as alias."""
        assert CISLLoss is CICLLoss
        
        # Can instantiate with either name
        loss1 = CISLLoss()
        loss2 = CICLLoss()
        
        # Both should work identically
        z = torch.randn(4, 3, 64)
        result1 = loss1(z)
        result2 = loss2(z)
        
        # Both should have same keys
        assert result1.keys() == result2.keys()
    
    def test_loss_values_reasonable(self):
        """Test that loss values are in reasonable ranges."""
        loss_fn = CISLLoss(
            consist_weight=0.5,
            content_inv_weight=0.5,
            variance_weight=0.1,
            target_std=0.5
        )
        
        z_struct = torch.randn(4, 3, 64)
        z_aug = z_struct + 0.1 * torch.randn_like(z_struct)
        
        result = loss_fn(z_struct, z_aug)
        
        # All losses should be non-negative
        assert result['consistency'].item() >= 0
        assert result['content_inv'].item() >= 0
        assert result['variance'].item() >= 0
        assert result['total'].item() >= 0
        
        # Losses shouldn't explode for random input
        assert result['total'].item() < 100


class TestCISLNaming:
    """Tests for CISL naming conventions and aliases."""
    
    def test_module_docstring(self):
        """Test that module has correct CISL naming in docstring."""
        from sci_arc.training import cisl_loss
        assert 'Content-Invariant Structure Learning' in cisl_loss.__doc__
    
    def test_class_docstring(self):
        """Test that CISLLoss class has correct naming."""
        assert 'Content-Invariant Structure Learning' in CISLLoss.__doc__


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
