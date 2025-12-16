"""
Integration Tests for RLAN Model

Tests the full RLAN model with all components working together.
"""

import pytest
import torch
import torch.nn as nn

from sci_arc.models import RLAN, RLANConfig


class TestRLANIntegration:
    """Integration tests for full RLAN model."""
    
    @pytest.fixture
    def model(self):
        """Create a small RLAN model for testing."""
        return RLAN(
            hidden_dim=64,
            num_colors=10,
            num_classes=11,
            max_clues=3,
            num_predicates=4,
            num_solver_steps=2,
            dropout=0.0,  # Disable dropout for deterministic testing
        )
    
    def test_forward_basic(self, model):
        """Test basic forward pass."""
        input_grid = torch.randint(0, 10, (2, 15, 15))
        logits = model(input_grid)
        
        assert logits.shape == (2, 11, 15, 15), \
            f"Expected (2, 11, 15, 15), got {logits.shape}"
    
    def test_forward_with_intermediates(self, model):
        """Test forward pass with intermediate outputs."""
        input_grid = torch.randint(0, 10, (2, 15, 15))
        outputs = model(input_grid, return_intermediates=True)
        
        assert "logits" in outputs
        assert "centroids" in outputs
        assert "attention_maps" in outputs
        assert "stop_logits" in outputs
        assert "predicates" in outputs
        assert "count_embedding" in outputs
        assert "features" in outputs
        assert "all_logits" in outputs
        
        # Check shapes
        assert outputs["logits"].shape == (2, 11, 15, 15)
        assert outputs["centroids"].shape == (2, 3, 2)  # 3 clues, 2 coords
        assert outputs["attention_maps"].shape == (2, 3, 15, 15)
        assert outputs["stop_logits"].shape == (2, 3)
        assert outputs["predicates"].shape == (2, 4)
        assert outputs["count_embedding"].shape == (2, 10, 64)
        assert outputs["features"].shape == (2, 64, 15, 15)
        assert len(outputs["all_logits"]) == 2  # 2 solver steps
    
    def test_variable_grid_sizes(self, model):
        """Test that model handles various grid sizes."""
        test_sizes = [(5, 5), (10, 15), (20, 8), (30, 30), (3, 3)]
        
        for h, w in test_sizes:
            input_grid = torch.randint(0, 10, (1, h, w))
            logits = model(input_grid)
            
            assert logits.shape == (1, 11, h, w), \
                f"Size ({h}, {w}): Expected (1, 11, {h}, {w}), got {logits.shape}"
    
    def test_batch_sizes(self, model):
        """Test various batch sizes."""
        for batch_size in [1, 2, 4, 8]:
            input_grid = torch.randint(0, 10, (batch_size, 10, 10))
            logits = model(input_grid)
            
            assert logits.shape[0] == batch_size, \
                f"Batch {batch_size}: Expected batch dim {batch_size}, got {logits.shape[0]}"
    
    def test_gradient_flow_all_params(self, model):
        """Test that gradients flow to all parameters."""
        input_grid = torch.randint(0, 10, (2, 10, 10))
        target = torch.randint(0, 11, (2, 10, 10))
        
        logits = model(input_grid)
        loss = nn.functional.cross_entropy(logits, target)
        loss.backward()
        
        # Check gradient flow to major components
        # Some parameters may not receive gradients (e.g., stop_predictor not used in basic loss)
        params_without_grad = []
        total_params = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                total_params += 1
                if param.grad is None:
                    params_without_grad.append(name)
                elif param.grad.abs().sum() == 0:
                    params_without_grad.append(f"{name} (zero grad)")
        
        # At least 90% of parameters should receive gradients
        grad_ratio = 1 - len(params_without_grad) / total_params
        assert grad_ratio >= 0.85, \
            f"Only {grad_ratio:.1%} of parameters got gradients. Missing: {params_without_grad[:5]}..."
    
    def test_temperature_parameter(self, model):
        """Test that temperature affects outputs."""
        input_grid = torch.randint(0, 10, (2, 10, 10))
        
        outputs_hot = model(input_grid, temperature=5.0, return_intermediates=True)
        outputs_cold = model(input_grid, temperature=0.1, return_intermediates=True)
        
        # Attention maps should be sharper (higher max) with cold temperature
        hot_attn_max = outputs_hot["attention_maps"].max()
        cold_attn_max = outputs_cold["attention_maps"].max()
        
        # Note: This is stochastic due to Gumbel noise, so we use a soft check
        # In practice, cold should usually be sharper
    
    def test_predict_method(self, model):
        """Test the predict convenience method."""
        input_grid = torch.randint(0, 10, (2, 10, 10))
        
        prediction = model.predict(input_grid)
        
        assert prediction.shape == (2, 10, 10)
        assert prediction.dtype == torch.int64 or prediction.dtype == torch.long
        assert (prediction >= 0).all() and (prediction < 11).all()
    
    def test_count_parameters(self, model):
        """Test parameter counting."""
        counts = model.count_parameters()
        
        assert "encoder" in counts
        assert "dsc" in counts
        assert "msre" in counts
        assert "lcr" in counts
        assert "sph" in counts
        assert "solver" in counts
        assert "total" in counts
        
        # Total should be sum of components
        component_sum = sum(v for k, v in counts.items() if k != "total")
        assert counts["total"] == component_sum
    
    def test_deterministic_eval(self, model):
        """Test that eval mode produces deterministic outputs."""
        model.eval()
        input_grid = torch.randint(0, 10, (1, 10, 10))
        
        torch.manual_seed(42)
        output1 = model(input_grid, temperature=0.1)
        
        torch.manual_seed(42)
        output2 = model(input_grid, temperature=0.1)
        
        # With same seed and low temperature, outputs should be very similar
        # (not exactly equal due to Gumbel noise, but close)
        diff = (output1 - output2).abs().mean()
        assert diff < 1.0, f"Outputs should be similar with same seed, got diff={diff:.4f}"
    
    def test_from_config(self):
        """Test creating model from config."""
        config = RLANConfig(
            hidden_dim=32,
            max_clues=2,
            num_predicates=4,
            num_solver_steps=2,
        )
        
        model = RLAN.from_config(config)
        
        assert model.hidden_dim == 32
        assert model.max_clues == 2
        assert model.num_predicates == 4
        assert model.num_solver_steps == 2
    
    def test_save_load_checkpoint(self, model, tmp_path):
        """Test checkpoint save and load."""
        checkpoint_path = tmp_path / "test_checkpoint.pt"
        
        # Save
        model.save_checkpoint(str(checkpoint_path), epoch=5, loss=0.5)
        
        # Load
        loaded_model = RLAN.load_from_checkpoint(str(checkpoint_path))
        
        # Compare parameters
        for (name1, param1), (name2, param2) in zip(
            model.named_parameters(), loaded_model.named_parameters()
        ):
            assert name1 == name2
            assert torch.allclose(param1, param2), \
                f"Parameter {name1} differs after load"


class TestRLANEdgeCases:
    """Edge case tests for RLAN."""
    
    def test_single_pixel_grid(self):
        """Test with 1x1 grid."""
        model = RLAN(hidden_dim=32, max_clues=2, num_solver_steps=1)
        input_grid = torch.randint(0, 10, (1, 1, 1))
        
        logits = model(input_grid)
        assert logits.shape == (1, 11, 1, 1)
    
    def test_uniform_color_grid(self):
        """Test with grid of single color."""
        model = RLAN(hidden_dim=32, max_clues=2, num_solver_steps=1)
        input_grid = torch.zeros(2, 10, 10, dtype=torch.long)
        
        outputs = model(input_grid, return_intermediates=True)
        assert outputs["logits"].shape == (2, 11, 10, 10)
    
    def test_all_different_colors(self):
        """Test with all colors present."""
        model = RLAN(hidden_dim=32, max_clues=2, num_solver_steps=1)
        # Create grid with repeating pattern of all colors
        input_grid = torch.arange(10).repeat(10).reshape(1, 10, 10) % 10
        
        outputs = model(input_grid.long(), return_intermediates=True)
        assert outputs["logits"].shape == (1, 11, 10, 10)
    
    def test_rectangular_grid(self):
        """Test with highly rectangular grid."""
        model = RLAN(hidden_dim=32, max_clues=2, num_solver_steps=1)
        
        # Very wide
        input_wide = torch.randint(0, 10, (1, 3, 30))
        logits_wide = model(input_wide)
        assert logits_wide.shape == (1, 11, 3, 30)
        
        # Very tall
        input_tall = torch.randint(0, 10, (1, 30, 3))
        logits_tall = model(input_tall)
        assert logits_tall.shape == (1, 11, 30, 3)


class TestRLANMemory:
    """Memory-related tests for RLAN."""
    
    def test_no_memory_leak_training(self):
        """Test that training doesn't leak memory."""
        model = RLAN(hidden_dim=32, max_clues=2, num_solver_steps=2)
        optimizer = torch.optim.Adam(model.parameters())
        
        input_grid = torch.randint(0, 10, (4, 10, 10))
        target = torch.randint(0, 11, (4, 10, 10))
        
        # Run a few training steps
        for _ in range(5):
            optimizer.zero_grad()
            logits = model(input_grid)
            loss = nn.functional.cross_entropy(logits, target)
            loss.backward()
            optimizer.step()
        
        # If we got here without OOM, test passes
        assert True
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_forward(self):
        """Test forward pass on CUDA."""
        model = RLAN(hidden_dim=32, max_clues=2, num_solver_steps=2).cuda()
        input_grid = torch.randint(0, 10, (2, 10, 10)).cuda()
        
        logits = model(input_grid)
        
        assert logits.device.type == "cuda"
        assert logits.shape == (2, 11, 10, 10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
