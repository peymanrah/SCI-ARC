"""
Unit tests for CrossAttentionInjector module.

Tests the Phase 2 architectural change: replacing FiLM compression with
Cross-Attention to preserve spatial structure in support set features.
"""

import torch
import pytest

from sci_arc.models.rlan_modules import CrossAttentionInjector


class TestCrossAttentionInjector:
    """Test suite for CrossAttentionInjector."""
    
    def test_initialization(self):
        """Test that CrossAttentionInjector initializes correctly."""
        injector = CrossAttentionInjector(
            hidden_dim=128,
            num_heads=4,
            dropout=0.1,
        )
        
        assert injector.hidden_dim == 128
        assert injector.num_heads == 4
        # Note: head_dim is computed internally but not stored as attribute
    
    def test_forward_pass_shape(self):
        """Test that forward pass produces correct output shape."""
        B, D, H, W = 2, 128, 10, 10
        N = 3  # Number of support pairs
        
        injector = CrossAttentionInjector(
            hidden_dim=D,
            num_heads=4,
            dropout=0.0,  # No dropout for reproducibility
        )
        
        # Create test features
        test_features = torch.randn(B, D, H, W)
        support_features = torch.randn(B, N, D, H, W)
        
        # Run forward pass
        output = injector(test_features, support_features)
        
        # Check output shape
        assert output.shape == (B, D, H, W), f"Expected shape {(B, D, H, W)}, got {output.shape}"
    
    def test_residual_connection(self):
        """Test that residual connection preserves identity when support is zero."""
        B, D, H, W = 2, 128, 10, 10
        N = 3
        
        injector = CrossAttentionInjector(
            hidden_dim=D,
            num_heads=4,
            dropout=0.0,
        )
        
        # Create test features
        test_features = torch.randn(B, D, H, W)
        support_features = torch.zeros(B, N, D, H, W)  # Zero support
        
        # Run forward pass
        output = injector(test_features, support_features)
        
        # Output should be close to input due to residual connection
        # (with layer norm and small FFN activation)
        # Check that output is not drastically different
        diff = (output - test_features).abs().mean()
        assert diff < 1.0, f"Output too different from input: mean abs diff = {diff:.3f}"
    
    def test_multiple_support_pairs(self):
        """Test that injector can handle varying numbers of support pairs."""
        B, D, H, W = 2, 128, 10, 10
        
        injector = CrossAttentionInjector(
            hidden_dim=D,
            num_heads=4,
            dropout=0.0,
        )
        
        test_features = torch.randn(B, D, H, W)
        
        # Test with different numbers of support pairs
        for N in [1, 3, 5]:
            support_features = torch.randn(B, N, D, H, W)
            output = injector(test_features, support_features)
            assert output.shape == (B, D, H, W), f"Failed for N={N}"
    
    def test_gradient_flow(self):
        """Test that gradients flow through the injector."""
        B, D, H, W = 2, 128, 10, 10
        N = 3
        
        injector = CrossAttentionInjector(
            hidden_dim=D,
            num_heads=4,
            dropout=0.0,
        )
        
        test_features = torch.randn(B, D, H, W, requires_grad=True)
        support_features = torch.randn(B, N, D, H, W, requires_grad=True)
        
        # Forward pass
        output = injector(test_features, support_features)
        
        # Backward pass
        loss = output.sum()
        loss.backward()
        
        # Check gradients exist
        assert test_features.grad is not None, "No gradient for test_features"
        assert support_features.grad is not None, "No gradient for support_features"
        
        # Check gradients are non-zero
        assert test_features.grad.abs().sum() > 0, "Zero gradient for test_features"
        assert support_features.grad.abs().sum() > 0, "Zero gradient for support_features"
    
    def test_attention_computation(self):
        """Test that attention scores are computed correctly."""
        B, D, H, W = 1, 64, 5, 5
        N = 2
        
        injector = CrossAttentionInjector(
            hidden_dim=D,
            num_heads=2,
            dropout=0.0,
        )
        
        # Create simple features to verify attention
        test_features = torch.ones(B, D, H, W)
        support_features = torch.ones(B, N, D, H, W)
        
        # Run forward pass
        output = injector(test_features, support_features)
        
        # Output should have same shape
        assert output.shape == test_features.shape
        
        # Output should be different from input (attention modified it)
        # Even with uniform inputs, FFN and layer norm will change values
        assert not torch.allclose(output, test_features, atol=1e-5)
    
    def test_batch_independence(self):
        """Test that batch elements are processed independently."""
        B, D, H, W = 4, 128, 10, 10
        N = 3
        
        injector = CrossAttentionInjector(
            hidden_dim=D,
            num_heads=4,
            dropout=0.0,
        )
        
        # Create features with different values per batch
        test_features = torch.randn(B, D, H, W)
        support_features = torch.randn(B, N, D, H, W)
        
        # Process full batch
        output_batch = injector(test_features, support_features)
        
        # Process each batch element separately
        outputs_individual = []
        for b in range(B):
            output_single = injector(
                test_features[b:b+1],
                support_features[b:b+1]
            )
            outputs_individual.append(output_single)
        
        outputs_individual = torch.cat(outputs_individual, dim=0)
        
        # Results should be identical (no cross-batch interaction)
        assert torch.allclose(output_batch, outputs_individual, atol=1e-5)
    
    def test_different_spatial_sizes(self):
        """Test that injector handles different spatial sizes."""
        B, D, N = 2, 128, 3
        
        injector = CrossAttentionInjector(
            hidden_dim=D,
            num_heads=4,
            dropout=0.0,
        )
        
        # Test different spatial sizes
        for H, W in [(5, 5), (10, 15), (20, 20), (30, 30)]:
            test_features = torch.randn(B, D, H, W)
            support_features = torch.randn(B, N, D, H, W)
            
            output = injector(test_features, support_features)
            assert output.shape == (B, D, H, W), f"Failed for size ({H}, {W})"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
