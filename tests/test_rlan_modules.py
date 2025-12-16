"""
Unit Tests for RLAN Modules

Tests each RLAN component individually:
- DynamicSaliencyController
- MultiScaleRelativeEncoding
- LatentCountingRegisters
- SymbolicPredicateHeads
- RecursiveSolver
"""

import pytest
import torch
import torch.nn as nn

from sci_arc.models.rlan_modules import (
    DynamicSaliencyController,
    MultiScaleRelativeEncoding,
    LatentCountingRegisters,
    SymbolicPredicateHeads,
    RecursiveSolver,
)


class TestDynamicSaliencyController:
    """Tests for Dynamic Saliency Controller."""
    
    @pytest.fixture
    def dsc(self):
        return DynamicSaliencyController(hidden_dim=64, max_clues=3)
    
    def test_output_shapes(self, dsc):
        """Test that outputs have correct shapes."""
        x = torch.randn(2, 64, 10, 10)
        centroids, attention_maps, stop_logits = dsc(x)
        
        assert centroids.shape == (2, 3, 2), f"Expected (2, 3, 2), got {centroids.shape}"
        assert attention_maps.shape == (2, 3, 10, 10), f"Expected (2, 3, 10, 10), got {attention_maps.shape}"
        assert stop_logits.shape == (2, 3), f"Expected (2, 3), got {stop_logits.shape}"
    
    def test_attention_sums_to_one(self, dsc):
        """Test that each attention map sums to approximately 1."""
        x = torch.randn(2, 64, 10, 10)
        _, attention_maps, _ = dsc(x)
        
        sums = attention_maps.sum(dim=(-2, -1))  # (B, K)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4), \
            f"Attention maps should sum to 1, got sums: {sums}"
    
    def test_temperature_effect(self, dsc):
        """Test that lower temperature produces sharper attention."""
        torch.manual_seed(42)
        x = torch.randn(2, 64, 10, 10)
        
        _, attn_hot, _ = dsc(x, temperature=5.0)
        _, attn_cold, _ = dsc(x, temperature=0.1)
        
        # Cold should have higher max (sharper)
        hot_max = attn_hot.max(dim=-1).values.max(dim=-1).values.mean()
        cold_max = attn_cold.max(dim=-1).values.max(dim=-1).values.mean()
        
        assert cold_max > hot_max, \
            f"Cold attention should be sharper, got hot={hot_max:.4f}, cold={cold_max:.4f}"
    
    def test_centroid_in_valid_range(self, dsc):
        """Test that centroids are within grid bounds."""
        x = torch.randn(2, 64, 15, 20)  # H=15, W=20
        centroids, _, _ = dsc(x)
        
        # Row should be in [0, H-1]
        assert (centroids[:, :, 0] >= 0).all() and (centroids[:, :, 0] <= 14).all(), \
            "Row centroids out of bounds"
        # Col should be in [0, W-1]
        assert (centroids[:, :, 1] >= 0).all() and (centroids[:, :, 1] <= 19).all(), \
            "Col centroids out of bounds"
    
    def test_gradient_flow(self, dsc):
        """Test that gradients flow through DSC."""
        x = torch.randn(2, 64, 10, 10, requires_grad=True)
        centroids, attention_maps, stop_logits = dsc(x)
        
        loss = centroids.sum() + attention_maps.sum() + stop_logits.sum()
        loss.backward()
        
        assert x.grad is not None, "Gradients should flow to input"
        assert x.grad.abs().sum() > 0, "Gradients should be non-zero"


class TestMultiScaleRelativeEncoding:
    """Tests for Multi-Scale Relative Encoding."""
    
    @pytest.fixture
    def msre(self):
        return MultiScaleRelativeEncoding(hidden_dim=64, encoding_dim=16)
    
    def test_output_shape(self, msre):
        """Test output shape is correct."""
        features = torch.randn(2, 64, 10, 10)
        centroids = torch.rand(2, 3, 2) * 10  # 3 clues
        
        encoded = msre(features, centroids)
        
        # Output should be (B, K, D, H, W)
        assert encoded.shape == (2, 3, 64, 10, 10), \
            f"Expected (2, 3, 64, 10, 10), got {encoded.shape}"
    
    def test_encoding_varies_with_centroid(self, msre):
        """Test that different centroids produce different encodings."""
        features = torch.randn(2, 64, 10, 10)
        
        centroids1 = torch.tensor([[[2.0, 2.0], [5.0, 5.0], [8.0, 8.0]],
                                   [[2.0, 2.0], [5.0, 5.0], [8.0, 8.0]]])
        centroids2 = torch.tensor([[[8.0, 8.0], [5.0, 5.0], [2.0, 2.0]],
                                   [[8.0, 8.0], [5.0, 5.0], [2.0, 2.0]]])
        
        encoded1 = msre(features, centroids1)
        encoded2 = msre(features, centroids2)
        
        # First and last clue encodings should be swapped (approximately)
        # because we swapped their centroids
        diff = (encoded1[:, 0] - encoded2[:, 2]).abs().mean()
        same = (encoded1[:, 0] - encoded1[:, 2]).abs().mean()
        
        # The swapped versions should be more similar than non-swapped
        # (This is a soft check, not strict equality)
        assert diff < same * 2, "Centroid swap should affect encoding"
    
    def test_gradient_flow(self, msre):
        """Test gradient flow through MSRE."""
        features = torch.randn(2, 64, 10, 10, requires_grad=True)
        # Create leaf tensor properly (multiply before enabling grad)
        centroids = (torch.rand(2, 3, 2) * 10).requires_grad_(True)
        
        encoded = msre(features, centroids)
        loss = encoded.sum()
        loss.backward()
        
        assert features.grad is not None, "Gradients should flow to features"
        assert centroids.grad is not None, "Gradients should flow to centroids"


class TestLatentCountingRegisters:
    """Tests for Latent Counting Registers."""
    
    @pytest.fixture
    def lcr(self):
        return LatentCountingRegisters(num_colors=10, hidden_dim=64)
    
    def test_output_shape(self, lcr):
        """Test output shape is correct."""
        grid = torch.randint(0, 10, (2, 10, 10))
        features = torch.randn(2, 64, 10, 10)
        
        count_embedding = lcr(grid, features)
        
        assert count_embedding.shape == (2, 10, 64), \
            f"Expected (2, 10, 64), got {count_embedding.shape}"
    
    def test_count_sensitivity(self, lcr):
        """Test that embeddings differ based on color counts."""
        features = torch.randn(1, 64, 10, 10)
        
        # Grid with mostly color 0
        grid1 = torch.zeros(1, 10, 10, dtype=torch.long)
        grid1[0, 0, :5] = 1  # 5 pixels of color 1
        
        # Grid with mostly color 1
        grid2 = torch.ones(1, 10, 10, dtype=torch.long)
        grid2[0, 0, :5] = 0  # 5 pixels of color 0
        
        embed1 = lcr(grid1, features)
        embed2 = lcr(grid2, features)
        
        # Embeddings for color 0 should differ
        diff_0 = (embed1[:, 0] - embed2[:, 0]).abs().mean()
        # Embeddings for color 1 should also differ
        diff_1 = (embed1[:, 1] - embed2[:, 1]).abs().mean()
        
        assert diff_0 > 0.1, f"Color 0 embeddings should differ, got diff={diff_0:.4f}"
        assert diff_1 > 0.1, f"Color 1 embeddings should differ, got diff={diff_1:.4f}"
    
    def test_get_count_probs(self, lcr):
        """Test count probability computation."""
        grid = torch.zeros(2, 10, 10, dtype=torch.long)
        grid[0, :5, :] = 1  # 50% color 1
        grid[1, :2, :] = 2  # 20% color 2
        
        probs = lcr.get_count_probs(grid)
        
        assert probs.shape == (2, 10), f"Expected (2, 10), got {probs.shape}"
        assert torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-5), \
            "Probabilities should sum to 1"
    
    def test_gradient_flow(self, lcr):
        """Test gradient flow through LCR."""
        grid = torch.randint(0, 10, (2, 10, 10))
        features = torch.randn(2, 64, 10, 10, requires_grad=True)
        
        count_embedding = lcr(grid, features)
        loss = count_embedding.sum()
        loss.backward()
        
        assert features.grad is not None, "Gradients should flow to features"


class TestSymbolicPredicateHeads:
    """Tests for Symbolic Predicate Heads."""
    
    @pytest.fixture
    def sph(self):
        return SymbolicPredicateHeads(hidden_dim=64, num_predicates=8)
    
    def test_output_shape(self, sph):
        """Test output shape is correct."""
        features = torch.randn(2, 64, 10, 10)
        predicates = sph(features)
        
        assert predicates.shape == (2, 8), f"Expected (2, 8), got {predicates.shape}"
    
    def test_output_range(self, sph):
        """Test that predicates are in (0, 1) range."""
        features = torch.randn(4, 64, 10, 10)
        predicates = sph(features, temperature=0.5)
        
        assert (predicates >= 0).all() and (predicates <= 1).all(), \
            "Predicates should be in [0, 1]"
    
    def test_temperature_effect(self, sph):
        """Test that lower temperature produces more extreme values."""
        torch.manual_seed(42)
        features = torch.randn(4, 64, 10, 10)
        
        pred_hot = sph(features, temperature=5.0)
        pred_cold = sph(features, temperature=0.1)
        
        # Cold should be closer to 0 or 1 (more extreme)
        hot_variance = ((pred_hot - 0.5).abs()).mean()
        cold_variance = ((pred_cold - 0.5).abs()).mean()
        
        assert cold_variance > hot_variance, \
            "Cold temperature should produce more extreme predicates"
    
    def test_diversity_loss(self, sph):
        """Test diversity loss computation."""
        features = torch.randn(16, 64, 10, 10)  # Larger batch for meaningful stats
        predicates = sph(features)
        
        diversity_loss = sph.compute_diversity_loss(predicates)
        
        assert diversity_loss.shape == (), "Diversity loss should be scalar"
        assert diversity_loss >= 0, "Diversity loss should be non-negative"
    
    def test_gradient_flow(self, sph):
        """Test gradient flow through SPH."""
        features = torch.randn(2, 64, 10, 10, requires_grad=True)
        predicates = sph(features)
        
        loss = predicates.sum()
        loss.backward()
        
        assert features.grad is not None, "Gradients should flow to features"


class TestRecursiveSolver:
    """Tests for Recursive Solver."""
    
    @pytest.fixture
    def solver(self):
        return RecursiveSolver(
            hidden_dim=64,
            num_classes=11,
            num_steps=3,
            num_predicates=8,
            num_colors=10,
        )
    
    def test_output_shape(self, solver):
        """Test output shape is correct."""
        clue_features = torch.randn(2, 3, 64, 10, 10)  # 3 clues
        count_embedding = torch.randn(2, 10, 64)
        predicates = torch.rand(2, 8)
        input_grid = torch.randint(0, 10, (2, 10, 10))
        
        logits = solver(clue_features, count_embedding, predicates, input_grid)
        
        assert logits.shape == (2, 11, 10, 10), \
            f"Expected (2, 11, 10, 10), got {logits.shape}"
    
    def test_all_steps_output(self, solver):
        """Test that return_all_steps returns correct number of predictions."""
        clue_features = torch.randn(2, 3, 64, 10, 10)
        count_embedding = torch.randn(2, 10, 64)
        predicates = torch.rand(2, 8)
        input_grid = torch.randint(0, 10, (2, 10, 10))
        
        all_logits = solver(
            clue_features, count_embedding, predicates, input_grid,
            return_all_steps=True
        )
        
        assert len(all_logits) == 3, f"Expected 3 steps, got {len(all_logits)}"
        for i, logits in enumerate(all_logits):
            assert logits.shape == (2, 11, 10, 10), \
                f"Step {i}: Expected (2, 11, 10, 10), got {logits.shape}"
    
    def test_with_attention_maps(self, solver):
        """Test that attention maps are used correctly."""
        clue_features = torch.randn(2, 3, 64, 10, 10)
        count_embedding = torch.randn(2, 10, 64)
        predicates = torch.rand(2, 8)
        input_grid = torch.randint(0, 10, (2, 10, 10))
        attention_maps = torch.softmax(torch.randn(2, 3, 10, 10).view(2, 3, -1), dim=-1).view(2, 3, 10, 10)
        
        logits = solver(
            clue_features, count_embedding, predicates, input_grid,
            attention_maps=attention_maps
        )
        
        assert logits.shape == (2, 11, 10, 10)
    
    def test_deep_supervision_loss(self, solver):
        """Test deep supervision loss computation."""
        clue_features = torch.randn(2, 3, 64, 10, 10)
        count_embedding = torch.randn(2, 10, 64)
        predicates = torch.rand(2, 8)
        input_grid = torch.randint(0, 10, (2, 10, 10))
        target = torch.randint(0, 11, (2, 10, 10))
        
        all_logits = solver(
            clue_features, count_embedding, predicates, input_grid,
            return_all_steps=True
        )
        
        loss_fn = nn.CrossEntropyLoss()
        ds_loss = solver.compute_deep_supervision_loss(all_logits, target, loss_fn)
        
        assert ds_loss.shape == (), "Deep supervision loss should be scalar"
        assert ds_loss > 0, "Loss should be positive"
    
    def test_gradient_flow(self, solver):
        """Test gradient flow through solver."""
        clue_features = torch.randn(2, 3, 64, 10, 10, requires_grad=True)
        count_embedding = torch.randn(2, 10, 64, requires_grad=True)
        predicates = torch.rand(2, 8, requires_grad=True)
        input_grid = torch.randint(0, 10, (2, 10, 10))
        
        logits = solver(clue_features, count_embedding, predicates, input_grid)
        loss = logits.sum()
        loss.backward()
        
        assert clue_features.grad is not None, "Gradients should flow to clue_features"
        assert count_embedding.grad is not None, "Gradients should flow to count_embedding"
        assert predicates.grad is not None, "Gradients should flow to predicates"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
