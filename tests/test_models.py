"""
Unit tests for SCI-ARC model components.
"""

import pytest
import torch
import numpy as np

from sci_arc.models import (
    SCIARC,
    SCIARCConfig,
    GridEncoder,
    StructuralEncoder2D,
    ContentEncoder2D,
    CausalBinding2D,
    RecursiveRefinement,
)


class TestGridEncoder:
    """Tests for GridEncoder component."""
    
    def test_output_shape(self):
        """Test that output shape is correct."""
        encoder = GridEncoder(hidden_dim=128, num_colors=10, max_size=30)
        
        # Single grid
        grid = torch.randint(0, 10, (1, 15, 20))  # [B, H, W]
        output = encoder(grid)
        
        assert output.shape == (1, 15, 20, 128)
    
    def test_batch_processing(self):
        """Test batch processing."""
        encoder = GridEncoder(hidden_dim=128)
        
        batch = torch.randint(0, 10, (4, 10, 10))
        output = encoder(batch)
        
        assert output.shape == (4, 10, 10, 128)
    
    def test_different_grid_sizes(self):
        """Test handling of different grid sizes."""
        encoder = GridEncoder(hidden_dim=64, max_size=30)
        
        for h, w in [(5, 5), (10, 15), (25, 30)]:
            grid = torch.randint(0, 10, (1, h, w))
            output = encoder(grid)
            assert output.shape == (1, h, w, 64)
    
    def test_color_embedding(self):
        """Test that different colors produce different embeddings."""
        encoder = GridEncoder(hidden_dim=32)
        
        grid1 = torch.zeros(1, 3, 3, dtype=torch.long)  # All black
        grid2 = torch.ones(1, 3, 3, dtype=torch.long)   # All blue
        
        out1 = encoder(grid1)
        out2 = encoder(grid2)
        
        # Outputs should be different
        assert not torch.allclose(out1, out2)


class TestStructuralEncoder:
    """Tests for StructuralEncoder2D component."""
    
    @pytest.fixture
    def encoder(self):
        return StructuralEncoder2D(
            hidden_dim=64,
            num_structure_slots=4,
            num_layers=2,
            num_heads=4,
        )
    
    def test_output_shape(self, encoder):
        """Test output shapes."""
        input_emb = torch.randn(2, 10, 10, 64)  # [B, H, W, D]
        output_emb = torch.randn(2, 8, 8, 64)   # Can be different size
        
        structure_slots = encoder(input_emb, output_emb)
        
        assert structure_slots.shape == (2, 4, 64)  # 4 structure slots
    
    def test_invariance_properties(self, encoder):
        """Test that similar transformations produce similar structure."""
        # Two pairs with same transformation pattern but different content
        input1 = torch.randn(1, 8, 8, 64)
        output1 = torch.randn(1, 8, 8, 64)
        
        # Slightly perturbed (same transformation with noise)
        input2 = input1 + 0.01 * torch.randn_like(input1)
        output2 = output1 + 0.01 * torch.randn_like(output1)
        
        slots1 = encoder(input1, output1)
        slots2 = encoder(input2, output2)
        
        # Pool slots to single vector for comparison
        z1 = slots1.mean(dim=1)
        z2 = slots2.mean(dim=1)
        
        # Should be similar (but not identical)
        cosine_sim = torch.nn.functional.cosine_similarity(z1, z2, dim=-1)
        assert cosine_sim > 0.8


class TestContentEncoder:
    """Tests for ContentEncoder2D component."""
    
    @pytest.fixture
    def encoder(self):
        return ContentEncoder2D(
            hidden_dim=64,
            max_objects=8,
            num_heads=4,
        )
    
    def test_output_shape(self, encoder):
        """Test output shapes."""
        grid_emb = torch.randn(2, 10, 10, 64)  # [B, H, W, D]
        structure_rep = torch.randn(2, 4, 64)   # [B, K, D]
        
        content_slots = encoder(grid_emb, structure_rep)
        
        assert content_slots.shape == (2, 8, 64)  # 8 object slots
    
    def test_orthogonality_property(self, encoder):
        """Test that content is projected orthogonally to structure."""
        grid_emb = torch.randn(2, 8, 8, 64)
        structure_rep = torch.randn(2, 4, 64)
        
        content_slots = encoder(grid_emb, structure_rep)
        
        # Pool to vectors for comparison
        content_vec = content_slots.mean(dim=1)  # [B, D]
        struct_vec = structure_rep.mean(dim=1)   # [B, D]
        
        # Normalize before checking orthogonality
        content_norm = torch.nn.functional.normalize(content_vec, dim=-1)
        struct_norm = torch.nn.functional.normalize(struct_vec, dim=-1)
        
        # Check orthogonality (dot product should be small)
        dot = (content_norm * struct_norm).sum(dim=-1)
        # Note: Won't be exactly 0 due to imperfect orthogonalization
        assert dot.abs().mean() < 0.5


class TestCausalBinding:
    """Tests for CausalBinding2D component."""
    
    @pytest.fixture
    def binding(self):
        return CausalBinding2D(
            hidden_dim=64, 
            num_structure_slots=4,
            num_content_slots=8,
            num_heads=4
        )
    
    def test_output_shape(self, binding):
        """Test z_task output shape."""
        structure_slots = torch.randn(2, 4, 64)  # [B, K, D]
        content_slots = torch.randn(2, 8, 64)    # [B, M, D]
        
        z_task = binding(structure_slots, content_slots)
        
        assert z_task.shape == (2, 64)


class TestRecursiveRefinement:
    """Tests for RecursiveRefinement component."""
    
    @pytest.fixture
    def refinement(self):
        return RecursiveRefinement(
            hidden_dim=64,
            max_cells=100,  # 10x10
            num_colors=10,
            H_cycles=2,
            L_cycles=2,
            L_layers=2,
            latent_size=16,
        )
    
    def test_output_shape(self, refinement):
        """Test output shapes."""
        x_test_emb = torch.randn(2, 100, 64)  # [B, N, D] flattened test input
        z_task = torch.randn(2, 64)           # [B, D] task embedding
        target_shape = (10, 10)
        
        outputs, final = refinement(x_test_emb, z_task, target_shape)
        
        assert final.shape == (2, 10, 10, 10)  # [B, H, W, num_colors]
        assert len(outputs) == 2  # H_cycles
    
    def test_deep_supervision(self, refinement):
        """Test that intermediate outputs are produced."""
        x_test_emb = torch.randn(1, 64, 64)  # [B, N, D]
        z_task = torch.randn(1, 64)
        target_shape = (8, 8)
        
        outputs, final = refinement(x_test_emb, z_task, target_shape)
        
        # Each intermediate should have same shape as final
        for inter in outputs:
            assert inter.shape == final.shape


class TestSCIARC:
    """Tests for complete SCIARC model."""
    
    @pytest.fixture
    def model(self):
        config = SCIARCConfig(
            hidden_dim=64,
            num_colors=10,
            max_grid_size=30,
            num_structure_slots=4,
            max_objects=8,
            H_cycles=2,
            L_cycles=2,
            L_layers=2,
        )
        return SCIARC(config)
    
    def test_forward_pass(self, model):
        """Test complete forward pass using demo_pairs format."""
        batch_size = 2
        H, W = 10, 10
        
        # Create demo pairs (list of (input, output) tuples)
        demo_pairs = [
            (torch.randint(0, 10, (batch_size, H, W)), 
             torch.randint(0, 10, (batch_size, H, W)))
            for _ in range(3)
        ]
        test_input = torch.randint(0, 10, (batch_size, H, W))
        target_shape = (H, W)
        
        # Use keyword arguments to avoid positional arg confusion
        output = model(demo_pairs=demo_pairs, test_input=test_input, target_shape=target_shape)
        
        # Check output is a dict from _forward_demo_pairs
        assert output['logits'].shape == (batch_size, H, W, 10)
        assert output['z_task'].shape == (batch_size, model.config.hidden_dim)
    
    def test_forward_training(self, model):
        """Test training-compatible forward pass."""
        batch_size = 2
        num_pairs = 3
        H, W = 10, 10
        
        input_grids = torch.randint(0, 10, (batch_size, num_pairs, H, W))
        output_grids = torch.randint(0, 10, (batch_size, num_pairs, H, W))
        test_input = torch.randint(0, 10, (batch_size, H, W))
        test_output = torch.randint(0, 10, (batch_size, H, W))
        grid_mask = torch.ones(batch_size, num_pairs, dtype=torch.bool)
        
        outputs = model.forward_training(
            input_grids=input_grids,
            output_grids=output_grids,
            test_input=test_input,
            test_output=test_output,
            grid_mask=grid_mask
        )
        
        assert 'logits' in outputs
        assert outputs['logits'].shape == (batch_size, H, W, 10)
    
    def test_output_components(self, model):
        """Test that all output components are present."""
        input_grids = torch.randint(0, 10, (1, 2, 8, 8))
        output_grids = torch.randint(0, 10, (1, 2, 8, 8))
        test_input = torch.randint(0, 10, (1, 8, 8))
        test_output = torch.randint(0, 10, (1, 8, 8))
        
        outputs = model.forward_training(
            input_grids=input_grids,
            output_grids=output_grids,
            test_input=test_input,
            test_output=test_output
        )
        
        assert 'logits' in outputs
        assert 'z_task' in outputs
        assert 'z_struct' in outputs
        assert 'z_content' in outputs
        assert 'intermediate_logits' in outputs
    
    def test_parameter_count(self, model):
        """Test parameter count is reasonable."""
        num_params = sum(p.numel() for p in model.parameters())
        
        # Should be less than 20M for this config
        assert num_params < 20_000_000
        # Should be more than 100K
        assert num_params > 100_000
    
    def test_gradient_flow(self, model):
        """Test that gradients flow correctly."""
        input_grids = torch.randint(0, 10, (1, 2, 5, 5))
        output_grids = torch.randint(0, 10, (1, 2, 5, 5))
        test_input = torch.randint(0, 10, (1, 5, 5))
        test_output = torch.randint(0, 10, (1, 5, 5))
        target = torch.randint(0, 10, (1, 5, 5))
        
        outputs = model.forward_training(
            input_grids=input_grids,
            output_grids=output_grids,
            test_input=test_input,
            test_output=test_output
        )
        logits = outputs['logits']
        
        # Compute loss
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, 10), target.view(-1)
        )
        
        loss.backward()
        
        # Check gradients exist for key components (some params may not get gradients
        # due to optional code paths like y_init which is overwritten)
        params_with_grad = sum(1 for p in model.parameters() if p.grad is not None)
        total_params = sum(1 for p in model.parameters() if p.requires_grad)
        
        # At least 90% of trainable params should have gradients
        assert params_with_grad / total_params > 0.9, f"Only {params_with_grad}/{total_params} params got gradients"
    
    def test_deterministic(self, model):
        """Test that model is deterministic in eval mode."""
        model.eval()
        
        # Use demo_pairs format
        demo_pairs = [
            (torch.randint(0, 10, (1, 8, 8)), 
             torch.randint(0, 10, (1, 8, 8)))
            for _ in range(2)
        ]
        test_input = torch.randint(0, 10, (1, 8, 8))
        target_shape = (8, 8)
        
        with torch.no_grad():
            out1 = model(demo_pairs=demo_pairs, test_input=test_input, target_shape=target_shape)
            out2 = model(demo_pairs=demo_pairs, test_input=test_input, target_shape=target_shape)
        
        assert torch.allclose(out1['logits'], out2['logits'])


class TestSCIARCConfig:
    """Tests for model configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = SCIARCConfig()
        
        assert config.hidden_dim == 256
        assert config.num_colors == 10
        assert config.max_grid_size == 30
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = SCIARCConfig(hidden_dim=512, H_cycles=5)
        
        assert config.hidden_dim == 512
        assert config.H_cycles == 5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
