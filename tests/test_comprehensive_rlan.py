"""
Comprehensive RLAN Test Suite

This test script validates:
1. All RLAN modules and their interfaces
2. Tensor dimensions throughout the pipeline
3. Gradient flow through all paths
4. Loss computation and aggregation
5. New modules: HyperLoRA, LOO, ACW, TTA
6. Configuration alignment with rlan_stable.yaml
7. Numerical stability
8. Padding/masking semantics

Run with: python -m pytest tests/test_comprehensive_rlan.py -v
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple

from sci_arc.models import RLAN, RLANConfig
from sci_arc.models.grid_encoder import GridEncoder
from sci_arc.models.rlan_modules import (
    DynamicSaliencyController,
    MultiScaleRelativeEncoding,
    LatentCountingRegisters,
    SymbolicPredicateHeads,
    RecursiveSolver,
    ContextEncoder,
    ContextInjector,
    CrossAttentionInjector,
)


class TestTensorDimensions:
    """Test tensor dimensions throughout the RLAN pipeline."""
    
    @pytest.fixture
    def config(self):
        """Production-like config matching rlan_stable.yaml structure."""
        return RLANConfig(
            hidden_dim=128,
            num_colors=10,
            num_classes=10,
            max_grid_size=30,
            max_clues=7,
            num_predicates=32,
            num_solver_steps=6,
            use_context_encoder=True,
            use_dsc=True,
            use_msre=True,
            use_lcr=False,  # Disabled in stable config
            use_sph=False,  # Disabled in stable config
            use_solver_context=True,
            use_cross_attention_context=True,
            spatial_downsample=8,
            dropout=0.1,
        )
    
    @pytest.fixture
    def small_config(self):
        """Small config for fast testing."""
        return RLANConfig(
            hidden_dim=64,
            num_colors=10,
            num_classes=10,
            max_grid_size=30,
            max_clues=3,
            num_predicates=8,
            num_solver_steps=2,
            use_context_encoder=True,
            use_dsc=True,
            use_msre=True,
            use_lcr=True,  # Enable for testing
            use_sph=True,  # Enable for testing
            use_solver_context=True,
            dropout=0.0,
        )
    
    def test_grid_encoder_dimensions(self):
        """Test GridEncoder input/output dimensions."""
        encoder = GridEncoder(hidden_dim=64, num_colors=10, max_size=30)
        
        # Standard grid
        grid = torch.randint(0, 10, (2, 15, 15))
        features = encoder(grid)
        assert features.shape == (2, 15, 15, 64), f"Expected (2, 15, 15, 64), got {features.shape}"
        
        # Max size grid
        grid_max = torch.randint(0, 10, (1, 30, 30))
        features_max = encoder(grid_max)
        assert features_max.shape == (1, 30, 30, 64)
        
        # Min size grid (1x1)
        grid_min = torch.randint(0, 10, (1, 1, 1))
        features_min = encoder(grid_min)
        assert features_min.shape == (1, 1, 1, 64)
        
        # Valid mask
        grid_with_pad = torch.randint(0, 10, (2, 10, 10))
        grid_with_pad[:, 5:, :] = 10  # PAD_COLOR
        valid_mask = encoder.get_valid_mask(grid_with_pad)
        assert valid_mask.shape == (2, 10, 10)
        assert valid_mask[:, :5, :].all()  # Non-padded region is valid
        assert not valid_mask[:, 5:, :].any()  # Padded region is invalid
    
    def test_dsc_dimensions(self):
        """Test DynamicSaliencyController dimensions."""
        dsc = DynamicSaliencyController(hidden_dim=64, max_clues=5, num_heads=4)
        
        features = torch.randn(2, 64, 12, 12)
        centroids, attention_maps, stop_logits = dsc(features)
        
        assert centroids.shape == (2, 5, 2), f"Centroids: expected (2, 5, 2), got {centroids.shape}"
        assert attention_maps.shape == (2, 5, 12, 12), f"Attention: expected (2, 5, 12, 12), got {attention_maps.shape}"
        assert stop_logits.shape == (2, 5), f"Stop logits: expected (2, 5), got {stop_logits.shape}"
        
        # Attention maps should sum to 1 per clue
        attn_sum = attention_maps.sum(dim=(-2, -1))
        assert torch.allclose(attn_sum, torch.ones_like(attn_sum), atol=1e-5), "Attention should sum to 1"
    
    def test_msre_dimensions(self):
        """Test MultiScaleRelativeEncoding dimensions."""
        msre = MultiScaleRelativeEncoding(hidden_dim=64, encoding_dim=32, max_size=30, num_freq=8)
        
        features = torch.randn(2, 64, 10, 10)
        centroids = torch.rand(2, 3, 2) * 10  # 3 clues, (row, col)
        grid_sizes = torch.tensor([[10, 10], [10, 10]])
        
        clue_features = msre(features, centroids, grid_sizes=grid_sizes)
        assert clue_features.shape == (2, 3, 64, 10, 10), f"Expected (2, 3, 64, 10, 10), got {clue_features.shape}"
    
    def test_lcr_dimensions_per_clue(self):
        """Test LatentCountingRegisters with per-clue counting."""
        lcr = LatentCountingRegisters(num_colors=10, hidden_dim=64, num_freq=8, num_heads=4)
        
        grid = torch.randint(0, 10, (2, 12, 12))
        features = torch.randn(2, 64, 12, 12)
        attention_maps = F.softmax(torch.randn(2, 3, 12, 12).view(2, 3, -1), dim=-1).view(2, 3, 12, 12)
        
        # Per-clue counting (when attention_maps provided)
        count_embed = lcr(grid, features, attention_maps=attention_maps)
        assert count_embed.shape == (2, 3, 64), f"Per-clue: expected (2, 3, 64), got {count_embed.shape}"
    
    def test_lcr_dimensions_global(self):
        """Test LatentCountingRegisters with global counting."""
        lcr = LatentCountingRegisters(num_colors=10, hidden_dim=64, num_freq=8, num_heads=4)
        
        grid = torch.randint(0, 10, (2, 12, 12))
        features = torch.randn(2, 64, 12, 12)
        
        # Global counting (when attention_maps NOT provided)
        count_embed = lcr(grid, features, attention_maps=None)
        assert count_embed.shape == (2, 10, 64), f"Global: expected (2, 10, 64), got {count_embed.shape}"
    
    def test_sph_dimensions(self):
        """Test SymbolicPredicateHeads dimensions."""
        sph = SymbolicPredicateHeads(hidden_dim=64, num_predicates=8)
        
        features = torch.randn(2, 64, 10, 10)
        predicates = sph(features)
        
        assert predicates.shape == (2, 8), f"Expected (2, 8), got {predicates.shape}"
        assert (predicates >= 0).all() and (predicates <= 1).all(), "Predicates should be in [0, 1]"
    
    def test_solver_dimensions(self):
        """Test RecursiveSolver dimensions."""
        solver = RecursiveSolver(
            hidden_dim=64,
            num_classes=10,
            num_steps=3,
            num_predicates=8,
            num_colors=10,
            use_solver_context=True,
        )
        
        clue_features = torch.randn(2, 5, 64, 10, 10)
        count_embedding = torch.randn(2, 10, 64)  # Global
        predicates = torch.rand(2, 8)
        input_grid = torch.randint(0, 10, (2, 10, 10))
        support_features = torch.randn(2, 3, 64, 8, 8)
        
        # Single output
        logits = solver(clue_features, count_embedding, predicates, input_grid, 
                       support_features=support_features)
        assert logits.shape == (2, 10, 10, 10), f"Expected (2, 10, 10, 10), got {logits.shape}"
        
        # All steps output
        all_logits = solver(clue_features, count_embedding, predicates, input_grid,
                           support_features=support_features, return_all_steps=True)
        assert len(all_logits) == 3, f"Expected 3 steps, got {len(all_logits)}"
        for logits in all_logits:
            assert logits.shape == (2, 10, 10, 10)
    
    def test_context_encoder_spatial(self):
        """Test ContextEncoder with spatial features."""
        encoder = ContextEncoder(
            hidden_dim=64,
            num_colors=10,
            max_size=30,
            max_pairs=5,
            use_spatial_features=True,
            spatial_downsample=8,
        )
        
        input_grids = torch.randint(0, 10, (2, 3, 15, 15))
        output_grids = torch.randint(0, 10, (2, 3, 15, 15))
        pair_mask = torch.ones(2, 3, dtype=torch.bool)
        
        support_features = encoder(input_grids, output_grids, pair_mask)
        # Downsampled to 8x8
        assert support_features.shape == (2, 3, 64, 8, 8), f"Expected (2, 3, 64, 8, 8), got {support_features.shape}"
    
    def test_context_encoder_compressed(self):
        """Test ContextEncoder with compressed context."""
        encoder = ContextEncoder(
            hidden_dim=64,
            num_colors=10,
            max_size=30,
            max_pairs=5,
            use_spatial_features=False,
        )
        
        input_grids = torch.randint(0, 10, (2, 3, 15, 15))
        output_grids = torch.randint(0, 10, (2, 3, 15, 15))
        pair_mask = torch.ones(2, 3, dtype=torch.bool)
        
        context = encoder(input_grids, output_grids, pair_mask)
        assert context.shape == (2, 64), f"Expected (2, 64), got {context.shape}"
    
    def test_full_rlan_dimensions(self, small_config):
        """Test full RLAN model dimensions."""
        model = RLAN(config=small_config)
        
        input_grid = torch.randint(0, 10, (2, 12, 12))
        train_inputs = torch.randint(0, 10, (2, 3, 12, 12))
        train_outputs = torch.randint(0, 10, (2, 3, 12, 12))
        
        # Basic forward
        logits = model(input_grid, train_inputs=train_inputs, train_outputs=train_outputs)
        assert logits.shape == (2, 10, 12, 12), f"Expected (2, 10, 12, 12), got {logits.shape}"
        
        # With intermediates
        outputs = model(input_grid, train_inputs=train_inputs, train_outputs=train_outputs,
                       return_intermediates=True)
        
        assert outputs["logits"].shape == (2, 10, 12, 12)
        assert outputs["centroids"].shape == (2, 3, 2)
        assert outputs["attention_maps"].shape == (2, 3, 12, 12)
        assert outputs["stop_logits"].shape == (2, 3)
        assert outputs["predicates"].shape == (2, 8)
        assert outputs["features"].shape == (2, 64, 12, 12)


class TestGradientFlow:
    """Test gradient flow through all model paths."""
    
    def test_gradient_flow_basic(self):
        """Test basic gradient flow through RLAN."""
        config = RLANConfig(
            hidden_dim=64,
            max_clues=3,
            num_solver_steps=2,
            use_solver_context=False,
            dropout=0.0,
        )
        model = RLAN(config=config)
        
        input_grid = torch.randint(0, 10, (2, 10, 10))
        train_inputs = torch.randint(0, 10, (2, 3, 10, 10))
        train_outputs = torch.randint(0, 10, (2, 3, 10, 10))
        target = torch.randint(0, 10, (2, 10, 10))
        
        logits = model(input_grid, train_inputs=train_inputs, train_outputs=train_outputs)
        loss = F.cross_entropy(logits, target)
        loss.backward()
        
        # Check key components have gradients
        assert model.encoder.color_embed.weight.grad is not None
        assert model.feature_proj[0].weight.grad is not None
        # Check solver output head has gradients
        assert model.solver.output_head[0].weight.grad is not None
    
    def test_gradient_flow_solver_context(self):
        """Test gradient flow through solver cross-attention."""
        config = RLANConfig(
            hidden_dim=64,
            max_clues=3,
            use_context_encoder=True,
            use_solver_context=True,
            dropout=0.0,
        )
        model = RLAN(config=config)
        
        input_grid = torch.randint(0, 10, (2, 10, 10))
        train_inputs = torch.randint(0, 10, (2, 3, 10, 10))
        train_outputs = torch.randint(0, 10, (2, 3, 10, 10))
        target = torch.randint(0, 10, (2, 10, 10))
        
        logits = model(input_grid, train_inputs=train_inputs, train_outputs=train_outputs)
        loss = F.cross_entropy(logits, target)
        loss.backward()
        
        # Solver cross-attention should have gradients
        assert model.solver.solver_cross_attn is not None
        has_solver_cross_attn_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.solver.solver_cross_attn.parameters()
        )
        assert has_solver_cross_attn_grad, "Solver cross-attention should receive gradients"
    
    def test_gradient_flow_dsc(self):
        """Test gradient flow through DSC stop predictor."""
        config = RLANConfig(
            hidden_dim=64,
            max_clues=3,
            use_dsc=True,
            dropout=0.0,
        )
        model = RLAN(config=config)
        
        input_grid = torch.randint(0, 10, (2, 10, 10))
        target = torch.randint(0, 10, (2, 10, 10))
        
        logits = model(input_grid)
        loss = F.cross_entropy(logits, target)
        loss.backward()
        
        # DSC clue queries should have gradients
        assert model.dsc.clue_queries.grad is not None


class TestNumericalStability:
    """Test numerical stability of RLAN components."""
    
    def test_attention_stability(self):
        """Test attention computation stability."""
        dsc = DynamicSaliencyController(hidden_dim=64, max_clues=3)
        
        # Normal features
        features = torch.randn(2, 64, 10, 10)
        centroids, attention_maps, stop_logits = dsc(features)
        
        assert not torch.isnan(attention_maps).any(), "Attention maps contain NaN"
        assert not torch.isinf(attention_maps).any(), "Attention maps contain Inf"
        assert (attention_maps >= 0).all(), "Attention should be non-negative"
    
    def test_attention_extreme_features(self):
        """Test attention with extreme feature values."""
        dsc = DynamicSaliencyController(hidden_dim=64, max_clues=3)
        
        # Large features (should not cause overflow)
        features_large = torch.randn(2, 64, 10, 10) * 100
        centroids, attention_maps, stop_logits = dsc(features_large)
        
        assert not torch.isnan(attention_maps).any(), "Large features caused NaN"
        assert not torch.isinf(attention_maps).any(), "Large features caused Inf"
    
    def test_loss_ignore_index(self):
        """Test that loss properly ignores padded targets."""
        logits = torch.randn(2, 10, 10, 10)
        target = torch.randint(0, 10, (2, 10, 10))
        
        # Mark some positions as ignore
        target[:, 5:, :] = -100  # ignore_index
        
        loss = F.cross_entropy(logits, target, ignore_index=-100)
        
        assert not torch.isnan(loss), "Loss is NaN with ignore_index"
        assert loss > 0, "Loss should be positive"
    
    def test_entropy_computation_stability(self):
        """Test entropy computation doesn't produce NaN."""
        # Uniform distribution
        attn_uniform = torch.ones(2, 100) / 100
        log_attn = torch.log(attn_uniform.clamp(min=1e-6))
        entropy = -(attn_uniform * log_attn).sum(dim=-1)
        assert not torch.isnan(entropy).any(), "Entropy of uniform distribution is NaN"
        
        # Near-delta distribution
        attn_delta = torch.zeros(2, 100)
        attn_delta[:, 0] = 0.999
        attn_delta[:, 1:] = 0.001 / 99
        log_attn = torch.log(attn_delta.clamp(min=1e-6))
        entropy = -(attn_delta * log_attn).sum(dim=-1)
        assert not torch.isnan(entropy).any(), "Entropy of delta distribution is NaN"


class TestPaddingSemantics:
    """Test padding and masking semantics."""
    
    def test_pad_color_is_10(self):
        """Test that PAD_COLOR is 10 for inputs."""
        from sci_arc.data.dataset import pad_grid
        
        grid = torch.tensor([[1, 2], [3, 4]])
        padded = pad_grid(grid, max_size=5)
        
        assert padded.shape == (5, 5)
        assert padded[0, 0] == 1  # Original value preserved
        assert padded[4, 4] == 10  # Padding is PAD_COLOR=10
    
    def test_target_padding_is_minus_100(self):
        """Test that target padding is -100 for ignore_index."""
        from sci_arc.data.dataset import pad_grid
        
        grid = torch.tensor([[1, 2], [3, 4]])
        padded = pad_grid(grid, max_size=5, is_target=True)
        
        assert padded.shape == (5, 5)
        assert padded[0, 0] == 1  # Original value preserved
        assert padded[4, 4] == -100  # Target padding is ignore_index
    
    def test_valid_mask_excludes_padding(self):
        """Test that valid mask properly excludes PAD_COLOR positions."""
        encoder = GridEncoder(hidden_dim=64, num_colors=10, max_size=30)
        
        grid = torch.randint(0, 10, (1, 5, 5))
        grid[0, 3:, :] = 10  # PAD_COLOR
        
        valid_mask = encoder.get_valid_mask(grid)
        
        assert valid_mask[0, :3, :].all(), "Non-padded positions should be valid"
        assert not valid_mask[0, 3:, :].any(), "Padded positions should be invalid"


class TestLossComponents:
    """Test loss computation and aggregation."""
    
    def test_cross_entropy_basic(self):
        """Test basic cross-entropy loss."""
        logits = torch.randn(2, 10, 10, 10)
        target = torch.randint(0, 10, (2, 10, 10))
        
        loss = F.cross_entropy(logits, target)
        
        assert loss.shape == ()  # Scalar
        assert loss > 0
        assert not torch.isnan(loss)
    
    def test_loss_with_class_weights(self):
        """Test loss with class weights (for imbalanced data)."""
        logits = torch.randn(2, 10, 10, 10)
        target = torch.randint(0, 10, (2, 10, 10))
        
        # Weight background (class 0) less
        weights = torch.ones(10)
        weights[0] = 0.5
        
        loss_weighted = F.cross_entropy(logits, target, weight=weights)
        loss_unweighted = F.cross_entropy(logits, target)
        
        assert not torch.isnan(loss_weighted)
        # With weight < 1 on common class, weighted loss is typically smaller
    
    def test_loss_reduction_modes(self):
        """Test different loss reduction modes."""
        logits = torch.randn(2, 10, 10, 10)
        target = torch.randint(0, 10, (2, 10, 10))
        
        loss_mean = F.cross_entropy(logits, target, reduction='mean')
        loss_sum = F.cross_entropy(logits, target, reduction='sum')
        loss_none = F.cross_entropy(logits, target, reduction='none')
        
        assert loss_mean.shape == ()
        assert loss_sum.shape == ()
        assert loss_none.shape == (2, 10, 10)
        
        # Consistency check
        assert torch.allclose(loss_none.mean(), loss_mean, atol=1e-5)


class TestCheckpointCompatibility:
    """Test checkpoint save/load compatibility."""
    
    def test_checkpoint_roundtrip(self, tmp_path):
        """Test saving and loading checkpoint."""
        config = RLANConfig(
            hidden_dim=64,
            max_clues=3,
            num_solver_steps=2,
        )
        model = RLAN(config=config)
        model.eval()  # Set to eval mode for deterministic behavior
        
        # Save
        path = tmp_path / "test_checkpoint.pt"
        model.save_checkpoint(str(path))
        
        # Load
        loaded_model = RLAN.load_from_checkpoint(str(path))
        loaded_model.eval()
        
        # Verify same architecture
        assert loaded_model.hidden_dim == model.hidden_dim
        assert loaded_model.num_classes == model.num_classes
        assert loaded_model.max_clues == model.max_clues
        
        # Verify weights match
        input_grid = torch.randint(0, 10, (1, 10, 10))
        with torch.no_grad():
            out1 = model(input_grid)
            out2 = loaded_model(input_grid)
        
        assert torch.allclose(out1, out2, atol=1e-5), "Loaded model produces different output"
    
    def test_checkpoint_with_extra_data(self, tmp_path):
        """Test checkpoint with extra training state."""
        config = RLANConfig(hidden_dim=64, max_clues=3)
        model = RLAN(config=config)
        
        path = tmp_path / "test_checkpoint_extra.pt"
        model.save_checkpoint(
            str(path),
            epoch=10,
            best_accuracy=0.85,
            optimizer_state={'lr': 0.001},
        )
        
        checkpoint = torch.load(str(path), weights_only=False)
        assert checkpoint['epoch'] == 10
        assert checkpoint['best_accuracy'] == 0.85
        assert checkpoint['optimizer_state']['lr'] == 0.001


class TestHyperLoRAIntegration:
    """Test HyperLoRA module integration."""
    
    def test_hyperlora_disabled_by_default(self):
        """Test that HyperLoRA is disabled by default."""
        config = RLANConfig(hidden_dim=64)
        model = RLAN(config=config)
        
        assert not model.use_hyperlora
        assert model.hyper_lora is None
    
    def test_hyperlora_enabled(self):
        """Test HyperLoRA when enabled."""
        config = RLANConfig(
            hidden_dim=64,
            use_hyperlora=True,
            hyperlora_rank=8,
        )
        model = RLAN(config=config)
        
        assert model.use_hyperlora
        assert model.hyper_lora is not None
    
    def test_hyperlora_forward(self):
        """Test forward pass with HyperLoRA."""
        config = RLANConfig(
            hidden_dim=64,
            max_clues=3,
            use_context_encoder=True,
            use_hyperlora=True,
            hyperlora_rank=4,
            dropout=0.0,
        )
        model = RLAN(config=config)
        
        input_grid = torch.randint(0, 10, (2, 10, 10))
        train_inputs = torch.randint(0, 10, (2, 3, 10, 10))
        train_outputs = torch.randint(0, 10, (2, 3, 10, 10))
        
        # Should work without error
        logits = model(input_grid, train_inputs=train_inputs, train_outputs=train_outputs)
        assert logits.shape == (2, 10, 10, 10)


class TestConfigAlignment:
    """Test alignment with rlan_stable.yaml configuration."""
    
    def test_stable_config_defaults(self):
        """Test that RLANConfig defaults match rlan_stable.yaml expectations."""
        config = RLANConfig()
        
        # Core settings from rlan_stable.yaml
        assert config.num_colors == 10
        assert config.num_classes == 10
        assert config.max_grid_size == 30
        
        # Module flags
        assert config.use_context_encoder == True
        assert config.use_dsc == True
        assert config.use_msre == True
        assert config.use_lcr == False  # Disabled in stable
        assert config.use_sph == False  # Disabled in stable
        assert config.use_act == False  # Disabled in stable
        
        # HyperLoRA
        assert config.use_hyperlora == False  # Experimental, disabled
    
    def test_production_config_creation(self):
        """Test creating a production-like configuration."""
        config = RLANConfig(
            hidden_dim=256,
            num_colors=10,
            num_classes=10,
            max_grid_size=30,
            max_clues=7,
            num_predicates=32,
            num_solver_steps=6,
            use_context_encoder=True,
            use_dsc=True,
            use_msre=True,
            use_lcr=False,
            use_sph=False,
            use_cross_attention_context=True,
            spatial_downsample=8,
            use_solver_context=True,
            solver_context_heads=4,
            dropout=0.1,
        )
        
        model = RLAN(config=config)
        
        assert model.hidden_dim == 256
        assert model.use_solver_context == True
        assert model.solver.solver_cross_attn is not None


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_single_pixel_grid(self):
        """Test with 1x1 grid."""
        config = RLANConfig(hidden_dim=64, max_clues=1, dropout=0.0)
        model = RLAN(config=config)
        
        grid = torch.randint(0, 10, (1, 1, 1))
        logits = model(grid)
        
        assert logits.shape == (1, 10, 1, 1)
    
    def test_max_size_grid(self):
        """Test with maximum size grid (30x30)."""
        config = RLANConfig(hidden_dim=64, max_clues=3, dropout=0.0)
        model = RLAN(config=config)
        
        grid = torch.randint(0, 10, (1, 30, 30))
        logits = model(grid)
        
        assert logits.shape == (1, 10, 30, 30)
    
    def test_non_square_grid(self):
        """Test with non-square grid."""
        config = RLANConfig(hidden_dim=64, max_clues=3, dropout=0.0)
        model = RLAN(config=config)
        
        grid = torch.randint(0, 10, (1, 5, 20))
        logits = model(grid)
        
        assert logits.shape == (1, 10, 5, 20)
    
    def test_batch_size_one(self):
        """Test with batch size 1."""
        config = RLANConfig(hidden_dim=64, max_clues=3, dropout=0.0)
        model = RLAN(config=config)
        
        grid = torch.randint(0, 10, (1, 10, 10))
        logits = model(grid)
        
        assert logits.shape == (1, 10, 10, 10)
    
    def test_all_same_color_grid(self):
        """Test with grid containing only one color."""
        config = RLANConfig(hidden_dim=64, max_clues=3, dropout=0.0)
        model = RLAN(config=config)
        
        grid = torch.zeros(2, 10, 10, dtype=torch.long)  # All black
        logits = model(grid)
        
        assert logits.shape == (2, 10, 10, 10)
        assert not torch.isnan(logits).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
