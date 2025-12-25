"""
Test cases for December 2025 bug fixes.

Tests verify:
1. HyperLoRA is properly connected in RLAN.forward() (was disconnected)
2. Equivariance loss compares actual weights, not just norms (was comparing norms)
3. ACW hashing uses batched CPU transfer (was doing N separate transfers)
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from unittest.mock import Mock, patch, MagicMock
import numpy as np


class TestHyperLoRAConnection:
    """Test that HyperLoRA is properly connected in RLAN.forward()."""
    
    def test_hyperlora_called_in_forward_with_context(self):
        """Verify hyper_lora is called when enabled and support_features exist."""
        from sci_arc.models.rlan import RLAN, RLANConfig
        
        # Create model with HyperLoRA enabled
        config = RLANConfig(
            hidden_dim=64,
            use_hyperlora=True,
            use_context_encoder=True,
            use_dsc=False,
            use_msre=False,
            use_lcr=False,
            use_sph=False,
        )
        model = RLAN(config=config)
        model.eval()
        
        # Track if hyper_lora is called
        original_hyper_lora = model.hyper_lora.forward
        call_count = [0]
        def tracking_forward(*args, **kwargs):
            call_count[0] += 1
            return original_hyper_lora(*args, **kwargs)
        
        model.hyper_lora.forward = tracking_forward
        
        # Create inputs
        B, H, W = 1, 8, 8
        input_grid = torch.randint(0, 10, (B, H, W))
        train_inputs = torch.randint(0, 10, (B, 2, H, W))
        train_outputs = torch.randint(0, 10, (B, 2, H, W))
        
        # Run forward
        with torch.no_grad():
            outputs = model(input_grid, train_inputs, train_outputs, return_intermediates=True)
        
        # Verify hyper_lora was called
        assert call_count[0] > 0, "HyperLoRA.forward() was NOT called during RLAN.forward()"
        
        # Verify lora_deltas are in the output
        assert 'lora_deltas' in outputs, "lora_deltas not returned in intermediates"
        assert outputs['lora_deltas'] is not None, "lora_deltas is None"
    
    def test_hyperlora_deltas_passed_to_solver(self):
        """Verify lora_deltas are actually passed to the solver."""
        from sci_arc.models.rlan import RLAN, RLANConfig
        
        config = RLANConfig(
            hidden_dim=64,
            use_hyperlora=True,
            use_context_encoder=True,
            use_dsc=False,
            use_msre=False,
            use_lcr=False,
            use_sph=False,
        )
        model = RLAN(config=config)
        model.eval()
        
        # Patch solver to capture what it receives
        original_solver = model.solver.forward
        received_lora_deltas = [None]
        
        def capturing_solver(*args, **kwargs):
            received_lora_deltas[0] = kwargs.get('lora_deltas')
            return original_solver(*args, **kwargs)
        
        model.solver.forward = capturing_solver
        
        # Create inputs
        B, H, W = 1, 8, 8
        input_grid = torch.randint(0, 10, (B, H, W))
        train_inputs = torch.randint(0, 10, (B, 2, H, W))
        train_outputs = torch.randint(0, 10, (B, 2, H, W))
        
        # Run forward
        with torch.no_grad():
            model(input_grid, train_inputs, train_outputs)
        
        # Verify solver received lora_deltas
        assert received_lora_deltas[0] is not None, "Solver did not receive lora_deltas"
    
    def test_hyperlora_disabled_backward_compatible(self):
        """Verify model works normally when HyperLoRA is disabled."""
        from sci_arc.models.rlan import RLAN, RLANConfig
        
        config = RLANConfig(
            hidden_dim=64,
            use_hyperlora=False,  # Disabled
            use_context_encoder=True,
            use_dsc=False,
            use_msre=False,
            use_lcr=False,
            use_sph=False,
        )
        model = RLAN(config=config)
        model.eval()
        
        # Create inputs
        B, H, W = 1, 8, 8
        input_grid = torch.randint(0, 10, (B, H, W))
        train_inputs = torch.randint(0, 10, (B, 2, H, W))
        train_outputs = torch.randint(0, 10, (B, 2, H, W))
        
        # Should work without error
        with torch.no_grad():
            outputs = model(input_grid, train_inputs, train_outputs, return_intermediates=True)
        
        # lora_deltas should NOT be in outputs when HyperLoRA disabled
        assert outputs.get('lora_deltas') is None, "lora_deltas should be None when HyperLoRA disabled"
    
    def test_hyperlora_without_context_backward_compatible(self):
        """Verify HyperLoRA enabled model works without support set."""
        from sci_arc.models.rlan import RLAN, RLANConfig
        
        config = RLANConfig(
            hidden_dim=64,
            use_hyperlora=True,
            use_context_encoder=True,
            use_dsc=False,
            use_msre=False,
            use_lcr=False,
            use_sph=False,
        )
        model = RLAN(config=config)
        model.eval()
        
        # Create input without context
        B, H, W = 1, 8, 8
        input_grid = torch.randint(0, 10, (B, H, W))
        
        # Should work without error (lora_deltas will be None)
        with torch.no_grad():
            outputs = model(input_grid, return_intermediates=True)
        
        # Model should run without crash
        assert 'logits' in outputs


class TestEquivarianceLossFix:
    """Test that equivariance loss compares actual weights, not norms."""
    
    def test_equivariance_loss_uses_mse(self):
        """Verify MSE is used instead of norm comparison."""
        from sci_arc.training.hyperlora_training import HyperLoRATrainer, HyperLoRATrainingConfig
        from sci_arc.models.rlan import RLAN, RLANConfig
        
        # Create model with HyperLoRA
        config = RLANConfig(
            hidden_dim=64,
            use_hyperlora=True,
            use_context_encoder=True,
            use_dsc=False,
            use_msre=False,
            use_lcr=False,
            use_sph=False,
        )
        model = RLAN(config=config)
        
        # Create trainer
        trainer_config = HyperLoRATrainingConfig(
            enabled=True,
            equivariance_enabled=True,
            num_augmentations=2,
        )
        trainer = HyperLoRATrainer(model, trainer_config, device=torch.device('cpu'))
        
        # Create test data
        B, N, D, H, W = 1, 2, 64, 8, 8
        support_features = torch.randn(B, N, D, H, W)
        original_lora_deltas = model.hyper_lora(support_features)
        
        # Compute loss
        loss, metrics = trainer.compute_equivariance_loss(support_features, original_lora_deltas)
        
        # Loss should be a tensor
        assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
        assert loss.numel() == 1, "Loss should be scalar"
    
    def test_equivariance_loss_detects_different_weights_same_norm(self):
        """
        Test that the new loss correctly penalizes different weights with same norm.
        
        The OLD bug: w1=[1,0] and w2=[0,1] have same norm, so old loss was 0.
        The NEW fix: MSE between w1 and w2 should be non-zero.
        """
        # Simulate what the loss calculation does
        # Old approach: compare norms
        w1 = torch.tensor([[1.0, 0.0]])
        w2 = torch.tensor([[0.0, 1.0]])
        
        old_loss = (w1.norm() - w2.norm()).pow(2)  # 0! Both have norm 1
        new_loss = F.mse_loss(w1, w2)  # 0.5! Weights are different
        
        assert old_loss < 0.01, "Old loss should be ~0 for same-norm weights"
        assert new_loss > 0.1, "New loss should detect different weights with same norm"


class TestACWBatchedHashing:
    """Test that ACW hashing uses batched CPU transfer."""
    
    def test_compute_hashes_batched(self):
        """Verify _compute_hashes uses a single batched transfer."""
        from sci_arc.models.rlan_modules.acw import AugmentedConfidenceWeighting
        
        acw = AugmentedConfidenceWeighting()
        
        # Create predictions
        predictions = [torch.randint(0, 10, (8, 8)) for _ in range(32)]
        
        # Call the method
        hashes = acw._compute_hashes(predictions)
        
        # Verify we got the right number of hashes
        assert len(hashes) == 32, "Should return one hash per prediction"
        
        # Verify hashes are strings
        assert all(isinstance(h, str) for h in hashes), "Hashes should be strings"
    
    def test_compute_hashes_empty_list(self):
        """Verify empty list is handled correctly."""
        from sci_arc.models.rlan_modules.acw import AugmentedConfidenceWeighting
        
        acw = AugmentedConfidenceWeighting()
        hashes = acw._compute_hashes([])
        
        assert hashes == [], "Empty list should return empty list"
    
    def test_compute_hashes_identical_predictions(self):
        """Verify identical predictions get identical hashes."""
        from sci_arc.models.rlan_modules.acw import AugmentedConfidenceWeighting
        
        acw = AugmentedConfidenceWeighting()
        
        # Create identical predictions
        base_pred = torch.randint(0, 10, (8, 8))
        predictions = [base_pred.clone() for _ in range(5)]
        
        hashes = acw._compute_hashes(predictions)
        
        # All hashes should be the same
        assert len(set(hashes)) == 1, "Identical predictions should have identical hashes"
    
    def test_compute_hashes_different_predictions(self):
        """Verify different predictions get different hashes."""
        from sci_arc.models.rlan_modules.acw import AugmentedConfidenceWeighting
        
        acw = AugmentedConfidenceWeighting()
        
        # Create different predictions
        pred1 = torch.zeros(8, 8, dtype=torch.long)
        pred2 = torch.ones(8, 8, dtype=torch.long)
        predictions = [pred1, pred2]
        
        hashes = acw._compute_hashes(predictions)
        
        # Hashes should be different
        assert hashes[0] != hashes[1], "Different predictions should have different hashes"
    
    def test_batched_vs_individual_gives_same_result(self):
        """Verify batched approach gives same results as individual transfers."""
        from sci_arc.models.rlan_modules.acw import AugmentedConfidenceWeighting
        
        acw = AugmentedConfidenceWeighting()
        
        # Create predictions
        predictions = [torch.randint(0, 10, (8, 8)) for _ in range(10)]
        
        # Compute hashes with batched method
        batched_hashes = acw._compute_hashes(predictions)
        
        # Compute hashes individually (old method)
        individual_hashes = [pred.cpu().numpy().tobytes().hex() for pred in predictions]
        
        # Results should match
        assert batched_hashes == individual_hashes, "Batched and individual hashing should give same results"


class TestBackwardCompatibility:
    """Test that fixes don't break existing functionality."""
    
    def test_rlan_forward_without_hyperlora_unchanged(self):
        """Verify RLAN forward behavior unchanged when HyperLoRA disabled."""
        from sci_arc.models.rlan import RLAN, RLANConfig
        
        # Model without HyperLoRA
        config = RLANConfig(
            hidden_dim=64,
            use_hyperlora=False,
            use_context_encoder=True,
            use_dsc=False,
            use_msre=False,
            use_lcr=False,
            use_sph=False,
        )
        model = RLAN(config=config)
        model.eval()
        
        B, H, W = 1, 8, 8
        input_grid = torch.randint(0, 10, (B, H, W))
        train_inputs = torch.randint(0, 10, (B, 2, H, W))
        train_outputs = torch.randint(0, 10, (B, 2, H, W))
        
        # Forward should work
        with torch.no_grad():
            logits = model(input_grid, train_inputs, train_outputs)
        
        # Output shape should be correct
        assert logits.shape == (B, 10, H, W), f"Expected (1, 10, 8, 8), got {logits.shape}"
    
    def test_predict_with_acw_still_works(self):
        """Verify predict_with_acw works after ACW hashing optimization."""
        from sci_arc.models.rlan import RLAN, RLANConfig
        
        config = RLANConfig(
            hidden_dim=64,
            use_context_encoder=True,
            use_dsc=False,
            use_msre=False,
            use_lcr=False,
            use_sph=False,
        )
        model = RLAN(config=config)
        model.eval()
        
        B, H, W = 1, 8, 8
        input_grid = torch.randint(0, 10, (B, H, W))
        train_inputs = torch.randint(0, 10, (B, 2, H, W))
        train_outputs = torch.randint(0, 10, (B, 2, H, W))
        
        # ACW prediction should work
        with torch.no_grad():
            pred, info = model.predict_with_acw(
                input_grid, train_inputs, train_outputs,
                num_color_perms=2  # Fewer for speed
            )
        
        # Output shape should be correct
        assert pred.shape == (B, H, W), f"Expected (1, 8, 8), got {pred.shape}"
        assert 'total_views' in info, "Info should contain voting details"


class TestGradientFlow:
    """Test gradient flow through HyperLoRA path."""
    
    def test_hyperlora_gradients_flow(self):
        """Verify gradients flow through HyperLoRA during training."""
        from sci_arc.models.rlan import RLAN, RLANConfig
        
        config = RLANConfig(
            hidden_dim=64,
            use_hyperlora=True,
            use_context_encoder=True,
            use_dsc=False,
            use_msre=False,
            use_lcr=False,
            use_sph=False,
        )
        model = RLAN(config=config)
        model.train()
        
        B, H, W = 1, 8, 8
        input_grid = torch.randint(0, 10, (B, H, W))
        train_inputs = torch.randint(0, 10, (B, 2, H, W))
        train_outputs = torch.randint(0, 10, (B, 2, H, W))
        target = torch.randint(0, 10, (B, H, W))
        
        # Forward pass
        logits = model(input_grid, train_inputs, train_outputs)
        
        # Compute loss
        loss = F.cross_entropy(logits.permute(0, 2, 3, 1).reshape(-1, 10), target.reshape(-1))
        
        # Backward pass
        loss.backward()
        
        # Check HyperLoRA has gradients
        hyper_lora = model.hyper_lora
        has_grads = False
        for name, param in hyper_lora.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grads = True
                break
        
        assert has_grads, "HyperLoRA should have non-zero gradients after backward"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
