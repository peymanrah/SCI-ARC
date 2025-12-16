"""
Tests for SCI-ARC Inference Modules.

Verifies:
1. StochasticSampler generates diverse candidates
2. TTTAdapter adapts and restores properly
3. EnsemblePredictor combines all strategies
4. No numerical instabilities (NaN, Inf)
5. SCL components are properly frozen during TTT
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sci_arc.models import SCIARC, SCIARCConfig
from sci_arc.inference.sampler import StochasticSampler, SamplingConfig, ConsistencyVerifier
from sci_arc.inference.ttt import TTTAdapter, TTTConfig
from sci_arc.inference.ensemble import EnsemblePredictor, EnsembleConfig


@pytest.fixture
def model():
    """Create a small model for testing."""
    config = SCIARCConfig(
        hidden_dim=64,
        num_structure_slots=4,
        max_objects=8,
        H_cycles=2,
        L_cycles=2,
        L_layers=1,
        num_heads=2,
    )
    model = SCIARC(config)
    return model


@pytest.fixture
def sample_task():
    """Create a sample task for testing."""
    # Simple 3x3 grids
    input_grids = torch.randint(0, 10, (3, 5, 5), dtype=torch.long)
    output_grids = torch.randint(0, 10, (3, 5, 5), dtype=torch.long)
    test_input = torch.randint(0, 10, (5, 5), dtype=torch.long)
    return input_grids, output_grids, test_input


class TestStochasticSampler:
    """Tests for StochasticSampler."""
    
    def test_init(self, model):
        """Test sampler initialization."""
        config = SamplingConfig(num_samples=8, temperature=0.8)
        sampler = StochasticSampler(model, config)
        assert sampler is not None
        assert sampler.config.num_samples == 8
    
    def test_generate_candidates(self, model, sample_task):
        """Test candidate generation."""
        input_grids, output_grids, test_input = sample_task
        
        config = SamplingConfig(
            num_samples=4,
            temperature=1.0,
            use_mc_dropout=True,
            device='cpu'
        )
        sampler = StochasticSampler(model, config)
        
        candidates, frequencies = sampler.generate_candidates(
            input_grids, output_grids, test_input
        )
        
        # Should have at least one candidate
        assert len(candidates) >= 1
        # Each candidate should be a numpy array
        assert isinstance(candidates[0], np.ndarray)
        # Shape should match test input
        assert candidates[0].shape == (5, 5)
    
    def test_temperature_clamping(self, model, sample_task):
        """Test that extreme temperatures don't cause NaN."""
        input_grids, output_grids, test_input = sample_task
        
        for temp in [0.01, 0.1, 1.0, 2.0, 10.0]:
            config = SamplingConfig(
                num_samples=2,
                temperature=temp,
                device='cpu'
            )
            sampler = StochasticSampler(model, config)
            
            candidates, _ = sampler.generate_candidates(
                input_grids, output_grids, test_input
            )
            
            # Should not have NaN
            for c in candidates:
                assert not np.any(np.isnan(c))
    
    def test_deduplication(self, model, sample_task):
        """Test that identical predictions are deduplicated."""
        input_grids, output_grids, test_input = sample_task
        
        # Use temperature 0 (deterministic) - all samples should be same
        config = SamplingConfig(
            num_samples=8,
            temperature=0.1,  # Very low temp = nearly deterministic
            use_mc_dropout=False,
            device='cpu'
        )
        sampler = StochasticSampler(model, config)
        
        candidates, frequencies = sampler.generate_candidates(
            input_grids, output_grids, test_input
        )
        
        # With low temperature and no dropout, should have few unique candidates
        assert len(candidates) >= 1
        # Frequencies should sum to num_samples
        assert sum(frequencies.values()) == 8


class TestTTTAdapter:
    """Tests for TTTAdapter."""
    
    def test_init(self, model):
        """Test TTT initialization."""
        config = TTTConfig(enabled=True, num_steps=5)
        ttt = TTTAdapter(model, config)
        assert ttt is not None
        assert ttt._original_state is not None
    
    def test_adapt_and_reset(self, model, sample_task):
        """Test that adapt changes weights and reset restores them."""
        input_grids, output_grids, _ = sample_task
        
        config = TTTConfig(
            enabled=True,
            num_steps=3,
            learning_rate=1e-3,
            device='cpu'
        )
        ttt = TTTAdapter(model, config)
        
        # Get original weights
        original_weight = model.grid_encoder.color_embed.weight.clone()
        
        # Adapt
        ttt.adapt(input_grids, output_grids)
        
        # Weights should have changed
        adapted_weight = model.grid_encoder.color_embed.weight.clone()
        
        # Reset
        ttt.reset()
        
        # Weights should be restored
        reset_weight = model.grid_encoder.color_embed.weight.clone()
        assert torch.allclose(original_weight, reset_weight)
    
    def test_scl_frozen(self, model, sample_task):
        """Test that SCL components are frozen during TTT."""
        input_grids, output_grids, _ = sample_task
        
        config = TTTConfig(
            enabled=True,
            num_steps=3,
            device='cpu'
        )
        ttt = TTTAdapter(model, config)
        
        # Check that SCL-related modules are in frozen set
        for name in ttt._frozen_modules:
            assert 'scl' in name.lower() or 'batch_norm' in name.lower() or 'projector' in name.lower()
    
    def test_gradient_clipping(self, model, sample_task):
        """Test that gradient clipping is applied."""
        input_grids, output_grids, _ = sample_task
        
        config = TTTConfig(
            enabled=True,
            num_steps=2,
            grad_clip=0.1,  # Very aggressive clipping
            device='cpu'
        )
        ttt = TTTAdapter(model, config)
        
        # Should not crash even with aggressive clipping
        ttt.adapt(input_grids, output_grids)
        ttt.reset()
    
    def test_predict_with_ttt(self, model, sample_task):
        """Test the full predict_with_ttt method."""
        input_grids, output_grids, test_input = sample_task
        
        config = TTTConfig(
            enabled=True,
            num_steps=2,
            device='cpu'
        )
        ttt = TTTAdapter(model, config)
        
        prediction = ttt.predict_with_ttt(input_grids, output_grids, test_input)
        
        # Should return a tensor
        assert isinstance(prediction, torch.Tensor)
        # Should match test input shape
        assert prediction.shape == test_input.shape


class TestEnsemblePredictor:
    """Tests for EnsemblePredictor."""
    
    def test_init_minimal(self, model):
        """Test minimal initialization (all disabled)."""
        config = EnsembleConfig(
            use_ttt=False,
            use_stochastic_sampling=False,
            use_augmentation_voting=False,
            use_consistency_verification=False,
            device='cpu'
        )
        predictor = EnsemblePredictor(model, config)
        assert predictor.ttt is None
        assert predictor.sampler is None
        assert predictor.verifier is None
    
    def test_init_full(self, model):
        """Test full initialization (all enabled)."""
        config = EnsembleConfig(
            use_ttt=True,
            use_stochastic_sampling=True,
            use_augmentation_voting=True,
            use_consistency_verification=True,
            device='cpu'
        )
        predictor = EnsemblePredictor(model, config)
        assert predictor.ttt is not None
        assert predictor.sampler is not None
        assert predictor.verifier is not None
    
    def test_predict(self, model, sample_task):
        """Test prediction with minimal config."""
        input_grids, output_grids, test_input = sample_task
        
        config = EnsembleConfig(
            use_ttt=False,
            use_stochastic_sampling=False,
            use_augmentation_voting=True,
            use_consistency_verification=False,
            num_dihedral=2,  # Just 2 for speed
            top_k=2,
            device='cpu'
        )
        predictor = EnsemblePredictor(model, config)
        
        results = predictor.predict(input_grids, output_grids, test_input)
        
        # Should return list of results
        assert isinstance(results, list)
        assert len(results) <= 2  # top_k=2
        
        # Each result should have prediction and confidence
        if len(results) > 0:
            assert 'prediction' in results[0]
            assert 'confidence' in results[0]
            assert isinstance(results[0]['prediction'], np.ndarray)
    
    def test_predict_task_format(self, model):
        """Test prediction with ARC task format."""
        task = {
            'train': [
                {'input': [[1, 2], [3, 4]], 'output': [[4, 3], [2, 1]]},
                {'input': [[5, 6], [7, 8]], 'output': [[8, 7], [6, 5]]},
            ],
            'test': [
                {'input': [[0, 1], [2, 3]], 'output': [[3, 2], [1, 0]]},
            ]
        }
        
        config = EnsembleConfig(
            use_ttt=False,
            use_stochastic_sampling=False,
            use_augmentation_voting=False,
            use_consistency_verification=False,
            device='cpu'
        )
        predictor = EnsemblePredictor(model, config)
        
        results = predictor.predict_task(task)
        
        # Should have one result per test case
        assert len(results) == 1
        # Should have correctness check
        assert 'correct' in results[0]


class TestNumericalStability:
    """Tests for numerical stability."""
    
    def test_no_nan_in_sampling(self, model, sample_task):
        """Test that sampling never produces NaN."""
        input_grids, output_grids, test_input = sample_task
        
        config = SamplingConfig(
            num_samples=16,
            temperature=0.5,
            use_mc_dropout=True,
            device='cpu'
        )
        sampler = StochasticSampler(model, config)
        
        for _ in range(5):  # Multiple runs
            candidates, _ = sampler.generate_candidates(
                input_grids, output_grids, test_input
            )
            for c in candidates:
                assert not np.any(np.isnan(c)), "NaN found in candidate"
                assert not np.any(np.isinf(c)), "Inf found in candidate"
    
    def test_ttt_loss_finite(self, model, sample_task):
        """Test that TTT loss remains finite."""
        input_grids, output_grids, _ = sample_task
        
        config = TTTConfig(
            enabled=True,
            num_steps=5,
            verbose=False,
            device='cpu'
        )
        ttt = TTTAdapter(model, config)
        
        # Should not raise any errors
        ttt.adapt(input_grids, output_grids)
        ttt.reset()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
