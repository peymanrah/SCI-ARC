"""
Integration test for Phase 2 Cross-Attention architecture.

Verifies that RLAN can:
1. Initialize with CrossAttentionInjector
2. Run forward pass with support features
3. Preserve gradient flow
4. Handle different batch sizes and support set sizes
"""

import torch
import pytest

from sci_arc.models.rlan import RLAN, RLANConfig


class TestPhase2Integration:
    """Integration tests for Cross-Attention RLAN."""
    
    def test_rlan_with_cross_attention(self):
        """Test RLAN forward pass with cross-attention context injection."""
        B, H, W = 2, 10, 10
        N = 3  # Number of training pairs
        
        # Create model with cross-attention enabled
        config = RLANConfig(
            hidden_dim=64,  # Small for testing
            max_clues=3,
            use_context_encoder=True,
        )
        model = RLAN(config=config)
        
        # Create test data
        test_input = torch.randint(0, 10, (B, H, W))
        train_inputs = torch.randint(0, 10, (B, N, H, W))
        train_outputs = torch.randint(0, 10, (B, N, H, W))
        
        # Forward pass
        logits = model(
            test_input,
            train_inputs=train_inputs,
            train_outputs=train_outputs,
        )
        
        # Check output shape
        assert logits.shape == (B, 10, H, W), f"Expected shape {(B, 10, H, W)}, got {logits.shape}"
    
    def test_gradient_flow_through_cross_attention(self):
        """Test that gradients flow through cross-attention path."""
        B, H, W = 2, 10, 10
        N = 3
        
        config = RLANConfig(
            hidden_dim=64,
            max_clues=3,
            use_context_encoder=True,
        )
        model = RLAN(config=config)
        
        # Create test data with gradients
        test_input = torch.randint(0, 10, (B, H, W))
        train_inputs = torch.randint(0, 10, (B, N, H, W))
        train_outputs = torch.randint(0, 10, (B, N, H, W))
        target = torch.randint(0, 10, (B, H, W))
        
        # Forward pass
        logits = model(
            test_input,
            train_inputs=train_inputs,
            train_outputs=train_outputs,
        )
        
        # Compute loss and backward
        loss = torch.nn.functional.cross_entropy(
            logits, target,
            reduction='mean'
        )
        loss.backward()
        
        # Check that context encoder has gradients
        has_gradients = False
        for param in model.context_encoder.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_gradients = True
                break
        
        assert has_gradients, "No gradients in context encoder"
        
        # Check that context injector has gradients
        has_gradients = False
        for param in model.context_injector.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_gradients = True
                break
        
        assert has_gradients, "No gradients in context injector"
    
    def test_variable_support_set_sizes(self):
        """Test that model handles different numbers of support pairs."""
        B, H, W = 2, 10, 10
        
        config = RLANConfig(
            hidden_dim=64,
            max_clues=3,
            use_context_encoder=True,
        )
        model = RLAN(config=config)
        
        test_input = torch.randint(0, 10, (B, H, W))
        
        # Test with different numbers of support pairs
        for N in [1, 2, 3, 5]:
            train_inputs = torch.randint(0, 10, (B, N, H, W))
            train_outputs = torch.randint(0, 10, (B, N, H, W))
            
            logits = model(
                test_input,
                train_inputs=train_inputs,
                train_outputs=train_outputs,
            )
            
            assert logits.shape == (B, 10, H, W), f"Failed for N={N}"
    
    def test_without_context(self):
        """Test that model can run without support set (legacy mode)."""
        B, H, W = 2, 10, 10
        
        config = RLANConfig(
            hidden_dim=64,
            max_clues=3,
            use_context_encoder=True,
        )
        model = RLAN(config=config)
        
        test_input = torch.randint(0, 10, (B, H, W))
        
        # Forward pass without train_inputs/outputs
        logits = model(test_input)
        
        assert logits.shape == (B, 10, H, W)
    
    def test_intermediates_output(self):
        """Test that model returns intermediates correctly."""
        B, H, W = 2, 10, 10
        N = 3
        
        config = RLANConfig(
            hidden_dim=64,
            max_clues=3,
            use_context_encoder=True,
        )
        model = RLAN(config=config)
        
        test_input = torch.randint(0, 10, (B, H, W))
        train_inputs = torch.randint(0, 10, (B, N, H, W))
        train_outputs = torch.randint(0, 10, (B, N, H, W))
        
        # Forward pass with intermediates
        outputs = model(
            test_input,
            train_inputs=train_inputs,
            train_outputs=train_outputs,
            return_intermediates=True,
        )
        
        # Check that it's a dict
        assert isinstance(outputs, dict)
        
        # Check required keys
        assert 'logits' in outputs
        assert 'features' in outputs
        
        # Check shapes
        assert outputs['logits'].shape == (B, 10, H, W)
        assert outputs['features'].shape == (B, 64, H, W)


class TestPhase25SolverCrossAttention:
    """Tests for Phase 2.5: Solver Cross-Attention to Support Set."""
    
    def test_solver_cross_attention_enabled(self):
        """Test that solver cross-attention is enabled by default."""
        config = RLANConfig(
            hidden_dim=64,
            max_clues=3,
            use_context_encoder=True,
            use_solver_context=True,
        )
        model = RLAN(config=config)
        
        # Check that solver has cross-attention module
        assert model.solver.solver_cross_attn is not None, "Solver cross-attention should be enabled"
    
    def test_solver_cross_attention_disabled(self):
        """Test that solver cross-attention can be disabled."""
        config = RLANConfig(
            hidden_dim=64,
            max_clues=3,
            use_context_encoder=True,
            use_solver_context=False,
        )
        model = RLAN(config=config)
        
        # Check that solver has no cross-attention module
        assert model.solver.solver_cross_attn is None, "Solver cross-attention should be disabled"
    
    def test_gradient_flow_through_solver_cross_attention(self):
        """Test that gradients flow through solver's cross-attention to support features."""
        B, H, W = 2, 10, 10
        N = 3
        
        config = RLANConfig(
            hidden_dim=64,
            max_clues=3,
            use_context_encoder=True,
            use_solver_context=True,
        )
        model = RLAN(config=config)
        
        # Create test data
        test_input = torch.randint(0, 10, (B, H, W))
        train_inputs = torch.randint(0, 10, (B, N, H, W))
        train_outputs = torch.randint(0, 10, (B, N, H, W))
        target = torch.randint(0, 10, (B, H, W))
        
        # Forward pass
        logits = model(
            test_input,
            train_inputs=train_inputs,
            train_outputs=train_outputs,
        )
        
        # Compute loss and backward
        loss = torch.nn.functional.cross_entropy(logits, target, reduction='mean')
        loss.backward()
        
        # Check that solver cross-attention has gradients
        has_gradients = False
        for name, param in model.solver.solver_cross_attn.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_gradients = True
                print(f"  [GRAD] solver_cross_attn.{name}: {param.grad.abs().sum().item():.4f}")
                break
        
        assert has_gradients, "No gradients flowing through solver cross-attention!"
    
    def test_solver_cross_attention_shapes(self):
        """Test solver cross-attention with different grid sizes."""
        from sci_arc.models.rlan_modules.recursive_solver import SolverCrossAttention
        
        cross_attn = SolverCrossAttention(hidden_dim=64, num_heads=4)
        
        # Test with different sizes
        for H, W in [(5, 5), (10, 10), (15, 12)]:
            hidden = torch.randn(2, 64, H, W)
            support = torch.randn(2, 3, 64, H, W)  # 3 support pairs
            
            output = cross_attn(hidden, support)
            
            assert output.shape == hidden.shape, f"Shape mismatch for {H}x{W}"
    
    def test_solver_cross_attention_gating(self):
        """Test that gating mechanism works (output differs from identity)."""
        from sci_arc.models.rlan_modules.recursive_solver import SolverCrossAttention
        
        cross_attn = SolverCrossAttention(hidden_dim=64, num_heads=4)
        
        hidden = torch.randn(2, 64, 10, 10)
        support = torch.randn(2, 3, 64, 10, 10)
        
        output = cross_attn(hidden, support)
        
        # Output should be different from input (not just identity)
        diff = (output - hidden).abs().mean()
        assert diff > 0.01, f"Cross-attention seems to be identity (diff={diff:.4f})"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
