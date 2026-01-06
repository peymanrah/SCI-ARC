"""
Smoke Tests for Program-Guided Training Integration.

These tests verify that the NS-TEPS training integration works end-to-end:
1. PrimitiveHead forward pass works with arbitrary features
2. PrimitiveHeadLoss computes gradients correctly  
3. ProgramGuidedRLAN wraps RLAN without modifying it
4. Joint training with pixel + primitive loss flows gradients correctly
5. PseudoLabelGenerator creates valid training targets

Run with: python -m pytest tests/test_program_guided_training.py -v

Author: AI Research Assistant
Date: January 2026
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional


# Import the modules we're testing
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sci_arc.models.generalization.primitive_head import (
    PrimitiveHead,
    PrimitiveHeadConfig,
    PrimitiveHeadLoss,
    PrimitiveEmbedding,
    ObjectScorer,
    ParameterPredictor,
    PRIMITIVE_NAME_TO_ID,
    PRIMITIVE_TYPE_MAPPING,
)
from sci_arc.models.generalization.program_guided_training import (
    ProgramGuidedRLAN,
    ProgramGuidedConfig,
    ProgramCache,
    PseudoLabelGenerator,
    create_program_guided_rlan,
)


# ============================================================================
# MOCK RLAN MODEL
# ============================================================================

class MockRLAN(nn.Module):
    """
    Minimal mock RLAN for testing.
    
    Simulates RLAN's forward pass with:
    - Encoder producing spatial features
    - Solver producing logits
    - Return intermediates option
    """
    
    def __init__(self, hidden_dim: int = 256, num_classes: int = 10):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # Mock config
        class Config:
            hidden_dim = 256
        self.config = Config()
        
        # Simple encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(10, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
        )
        
        # Simple output head
        self.output_head = nn.Conv2d(hidden_dim, num_classes, 1)
    
    def forward(
        self,
        test_input: torch.Tensor,
        train_inputs: Optional[torch.Tensor] = None,
        train_outputs: Optional[torch.Tensor] = None,
        return_intermediates: bool = False,
        pair_mask: Optional[torch.Tensor] = None,  # Accept like real RLAN
        temperature: float = 1.0,  # Accept like real RLAN
        **kwargs,  # Accept any additional kwargs
    ):
        # Encode
        features = self.encoder(test_input)
        
        # Output
        logits = self.output_head(features)
        
        if return_intermediates:
            return {
                'logits': logits,
                'solver_features': features,
                'encoder_features': features,
            }
        return logits


# ============================================================================
# PRIMITIVE HEAD TESTS
# ============================================================================

class TestPrimitiveHead:
    """Tests for PrimitiveHead module."""
    
    @pytest.fixture
    def config(self):
        return PrimitiveHeadConfig(
            num_primitives=15,
            hidden_dim=256,
            max_objects=20,
            num_params=8,
        )
    
    @pytest.fixture
    def primitive_head(self, config):
        head = PrimitiveHead(config)
        head.primitive_embed.init_primitive_types(PRIMITIVE_TYPE_MAPPING)
        return head
    
    def test_forward_pass_basic(self, primitive_head):
        """Test basic forward pass with spatial features."""
        B, C, H, W = 2, 256, 16, 16
        features = torch.randn(B, C, H, W)
        
        outputs = primitive_head(features, return_trace=True)
        
        # Check all expected outputs exist (using actual key names from implementation)
        assert 'primitive_logits' in outputs
        assert 'object_scores' in outputs
        assert 'param_logits' in outputs  # Not param_discrete
        assert 'param_values' in outputs  # Not param_continuous
        
        # Check shapes
        assert outputs['primitive_logits'].shape == (B, 15)  # num_primitives
        assert outputs['object_scores'].shape[0] == B
        assert outputs['param_logits'].shape == (B, 8, 32)  # num_params, vocab_size
        assert outputs['param_values'].shape == (B, 8)
    
    def test_forward_with_object_mask(self, primitive_head):
        """Test forward pass with object mask."""
        B, C, H, W = 2, 256, 16, 16
        features = torch.randn(B, C, H, W)
        
        # Mock object mask - must be boolean type
        num_objects = 20  # max_objects from config
        object_mask = torch.ones(B, num_objects, dtype=torch.bool)
        
        outputs = primitive_head(features, object_mask=object_mask)
        
        assert outputs['primitive_logits'].shape == (B, 15)
        # Object scores shape depends on max_objects in config (20)
        assert outputs['object_scores'].shape[0] == B
    
    def test_primitive_embedding_hierarchy(self, config):
        """Test primitive embedding with type hierarchy."""
        embed = PrimitiveEmbedding(
            num_primitives=config.num_primitives,
            embed_dim=config.hidden_dim
        )
        embed.init_primitive_types(PRIMITIVE_TYPE_MAPPING)
        
        # Get all embeddings (no input needed - returns all)
        embeddings = embed()  # (num_primitives, hidden_dim)
        
        assert embeddings.shape == (15, 256)
        
        # Embeddings should be different for different types
        # Check variance in embeddings
        assert embeddings.std() > 0.01
    
    def test_gradients_flow(self, primitive_head):
        """Test that gradients flow through all components."""
        B, C, H, W = 2, 256, 16, 16
        features = torch.randn(B, C, H, W, requires_grad=True)
        
        outputs = primitive_head(features, return_trace=True)
        
        # Compute pseudo-loss (using correct key names)
        loss = outputs['primitive_logits'].sum() + outputs['param_values'].sum()
        loss.backward()
        
        assert features.grad is not None
        assert features.grad.abs().sum() > 0


class TestPrimitiveHeadLoss:
    """Tests for PrimitiveHeadLoss."""
    
    @pytest.fixture
    def config(self):
        return PrimitiveHeadConfig(num_primitives=15, num_params=8)
    
    @pytest.fixture
    def loss_fn(self, config):
        return PrimitiveHeadLoss(config)
    
    def test_loss_computation(self, loss_fn):
        """Test loss computation with all components."""
        B = 4
        
        # Predictions use the actual key names from PrimitiveHead
        # Need requires_grad=True for gradients
        outputs = {
            'primitive_logits': torch.randn(B, 15, requires_grad=True),
            'object_scores': torch.randn(B, 10, requires_grad=True),
            'param_logits': torch.randn(B, 8, 32, requires_grad=True),
            'param_values': torch.randn(B, 8, requires_grad=True),
        }
        
        targets = {
            'primitive_ids': torch.randint(0, 15, (B,)),
            'param_discrete': torch.randint(0, 32, (B, 8)),
            'param_continuous': torch.randn(B, 8),
        }
        
        mask = torch.ones(B)
        
        loss_dict = loss_fn(outputs, targets, mask)
        
        assert 'primitive_loss' in loss_dict
        assert 'param_loss' in loss_dict
        assert 'total_loss' in loss_dict
        assert loss_dict['total_loss'].requires_grad
    
    def test_masked_loss(self, loss_fn):
        """Test that masking correctly zeros out losses."""
        B = 4
        
        outputs = {
            'primitive_logits': torch.randn(B, 15),
            'object_scores': torch.randn(B, 10),
            'param_logits': torch.randn(B, 8, 32),
            'param_values': torch.randn(B, 8),
        }
        
        targets = {
            'primitive_ids': torch.randint(0, 15, (B,)),
            'param_discrete': torch.randint(0, 32, (B, 8)),
            'param_continuous': torch.randn(B, 8),
        }
        
        # All zeros mask - should have zero loss
        mask = torch.zeros(B)
        loss_dict = loss_fn(outputs, targets, mask)
        
        # Note: Implementation may add epsilon, check near-zero
        assert loss_dict['total_loss'] < 0.01


# ============================================================================
# PROGRAM GUIDED RLAN TESTS
# ============================================================================

class TestProgramGuidedRLAN:
    """Tests for ProgramGuidedRLAN wrapper."""
    
    @pytest.fixture
    def mock_rlan(self):
        return MockRLAN(hidden_dim=256, num_classes=10)
    
    @pytest.fixture
    def config(self):
        return ProgramGuidedConfig(
            enabled=True,
            primitive_loss_weight=0.3,
            warmup_epochs=0,  # Disable warmup for testing
            curriculum_epochs=1,
            online_mining=False,  # Disable online mining for tests
        )
    
    @pytest.fixture
    def pg_rlan(self, mock_rlan, config):
        return ProgramGuidedRLAN(mock_rlan, config)
    
    def test_wrapper_creation(self, pg_rlan, mock_rlan):
        """Test wrapper is created correctly."""
        assert pg_rlan.base_rlan is mock_rlan
        assert pg_rlan.primitive_head is not None
        assert pg_rlan.config.enabled
    
    def test_forward_pass(self, pg_rlan):
        """Test forward pass through wrapper."""
        B, C, H, W = 2, 10, 16, 16
        test_input = torch.randn(B, C, H, W)
        
        # Basic forward
        logits = pg_rlan(test_input)
        assert logits.shape == (B, 10, H, W)  # num_classes
        
        # Forward with primitives
        outputs = pg_rlan(test_input, return_primitive_outputs=True)
        assert 'logits' in outputs
        assert 'primitive_outputs' in outputs
        assert outputs['primitive_outputs'] is not None
    
    def test_combined_loss(self, pg_rlan):
        """Test combined pixel + primitive loss computation."""
        B, C, H, W = 2, 10, 16, 16
        test_input = torch.randn(B, C, H, W)
        targets = torch.randint(0, 10, (B, H, W))
        
        # Forward with primitives
        outputs = pg_rlan(test_input, return_primitive_outputs=True)
        
        # Create primitive targets
        prim_targets = {
            'primitive_ids': torch.randint(0, 15, (B, 3)),
            'param_discrete': torch.randint(0, 32, (B, 3, 8)),
            'param_continuous': torch.randn(B, 3, 8),
            'has_program': torch.ones(B, dtype=torch.bool),
        }
        
        # Set epoch to enable primitive loss
        pg_rlan.set_epoch(5)
        
        # Compute loss
        losses = pg_rlan.compute_loss(outputs, targets, prim_targets)
        
        assert 'pixel_loss' in losses
        assert 'total_loss' in losses
        assert losses['total_loss'].requires_grad
    
    def test_gradients_flow_to_base_rlan(self, pg_rlan):
        """Test gradients flow from combined loss to base RLAN."""
        B, C, H, W = 2, 10, 16, 16
        test_input = torch.randn(B, C, H, W)
        targets = torch.randint(0, 10, (B, H, W))
        
        # Zero gradients
        pg_rlan.zero_grad()
        
        # Forward
        outputs = pg_rlan(test_input, return_primitive_outputs=True)
        
        # Create primitive targets
        prim_targets = {
            'primitive_ids': torch.randint(0, 15, (B, 3)),
            'param_discrete': torch.randint(0, 32, (B, 3, 8)),
            'param_continuous': torch.randn(B, 3, 8),
            'has_program': torch.ones(B, dtype=torch.bool),
        }
        
        # Set epoch to enable primitive loss
        pg_rlan.set_epoch(5)
        
        # Compute loss
        losses = pg_rlan.compute_loss(outputs, targets, prim_targets)
        
        # Backward
        losses['total_loss'].backward()
        
        # Check base RLAN has gradients
        for param in pg_rlan.base_rlan.parameters():
            if param.grad is not None:
                assert param.grad.abs().sum() > 0
                break
        else:
            pytest.fail("No gradients found in base RLAN")
        
        # Check PrimitiveHead has gradients
        for param in pg_rlan.primitive_head.parameters():
            if param.grad is not None:
                assert param.grad.abs().sum() > 0
                break
        else:
            pytest.fail("No gradients found in PrimitiveHead")
    
    def test_primitive_prior(self, pg_rlan):
        """Test primitive prior extraction for NS-TEPS guidance."""
        B, C, H, W = 2, 10, 16, 16
        test_input = torch.randn(B, C, H, W)
        
        top_ids, top_probs = pg_rlan.get_primitive_prior(test_input)
        
        assert top_ids.shape == (B, 5)  # top-5
        assert top_probs.shape == (B, 5)
        assert (top_probs >= 0).all() and (top_probs <= 1).all()
    
    def test_curriculum_weight(self, pg_rlan):
        """Test curriculum learning weight schedule."""
        # Before warmup
        pg_rlan.set_epoch(0)
        assert pg_rlan._get_primitive_weight() == 0.0
        
        # After curriculum
        pg_rlan.set_epoch(10)
        assert pg_rlan._get_primitive_weight() == pg_rlan.config.primitive_loss_weight
    
    def test_disabled_mode(self, mock_rlan):
        """Test that wrapper works when disabled."""
        config = ProgramGuidedConfig(enabled=False)
        pg_rlan = ProgramGuidedRLAN(mock_rlan, config)
        
        assert pg_rlan.primitive_head is None
        
        # Forward should still work
        B, C, H, W = 2, 10, 16, 16
        test_input = torch.randn(B, C, H, W)
        logits = pg_rlan(test_input)
        assert logits.shape == (B, 10, H, W)


# ============================================================================
# PSEUDO LABEL GENERATOR TESTS
# ============================================================================

class TestPseudoLabelGenerator:
    """Tests for pseudo-label generation."""
    
    @pytest.fixture
    def config(self):
        return ProgramGuidedConfig(
            online_mining=False,  # Disable NS-TEPS for unit tests
            use_cached_programs=False,
        )
    
    def test_program_cache(self):
        """Test program cache operations."""
        cache = ProgramCache()
        
        # Add program
        trace = [('copy_object', {'dx': 0, 'dy': 5})]
        cache.add('task_001', trace, 0.95, 'hash123')
        
        assert cache.has('task_001')
        assert not cache.has('task_002')
        
        retrieved = cache.get('task_001')
        assert retrieved['confidence'] == 0.95
    
    def test_trace_to_targets(self, config):
        """Test conversion of program trace to training targets."""
        generator = PseudoLabelGenerator(config)
        
        trace = [
            ('copy_object', {'dx': 0, 'dy': 5}),
            ('rotate_object', {'angle': 90}),
        ]
        
        targets = generator._trace_to_targets(trace)
        
        assert 'primitive_ids' in targets
        assert 'param_discrete' in targets
        assert 'param_continuous' in targets
        assert targets['has_program']
        
        # Check primitive IDs are valid
        copy_id = PRIMITIVE_NAME_TO_ID.get('copy_object', 0)
        rotate_id = PRIMITIVE_NAME_TO_ID.get('rotate_object', 0)
        assert targets['primitive_ids'][0] == copy_id
        assert targets['primitive_ids'][1] == rotate_id


# ============================================================================
# END-TO-END TRAINING TESTS
# ============================================================================

class TestEndToEndTraining:
    """Test complete training flow."""
    
    def test_training_step(self):
        """Test a complete training step with gradient update."""
        # Setup
        mock_rlan = MockRLAN()
        config = ProgramGuidedConfig(
            enabled=True,
            warmup_epochs=0,
            primitive_loss_weight=0.3,
            online_mining=False,
        )
        pg_rlan = ProgramGuidedRLAN(mock_rlan, config)
        pg_rlan.set_epoch(5)  # Past warmup
        
        optimizer = torch.optim.Adam(pg_rlan.parameters(), lr=1e-4)
        
        # Fake batch
        B, C, H, W = 4, 10, 16, 16
        test_input = torch.randn(B, C, H, W)
        targets = torch.randint(0, 10, (B, H, W))
        
        prim_targets = {
            'primitive_ids': torch.randint(0, 15, (B, 3)),
            'param_discrete': torch.randint(0, 32, (B, 3, 8)),
            'param_continuous': torch.randn(B, 3, 8),
            'has_program': torch.ones(B, dtype=torch.bool),
        }
        
        # Training step
        optimizer.zero_grad()
        outputs = pg_rlan(test_input, return_primitive_outputs=True)
        losses = pg_rlan.compute_loss(outputs, targets, prim_targets)
        losses['total_loss'].backward()
        optimizer.step()
        
        # Success if no errors
        assert True
    
    def test_multiple_training_steps(self):
        """Test multiple training steps reduce loss."""
        # Setup
        mock_rlan = MockRLAN()
        config = ProgramGuidedConfig(
            enabled=True,
            warmup_epochs=0,
            primitive_loss_weight=0.1,
            online_mining=False,
        )
        pg_rlan = ProgramGuidedRLAN(mock_rlan, config)
        pg_rlan.set_epoch(5)
        
        optimizer = torch.optim.Adam(pg_rlan.parameters(), lr=1e-3)
        
        # Fixed batch for overfitting
        B, C, H, W = 4, 10, 16, 16
        test_input = torch.randn(B, C, H, W)
        targets = torch.randint(0, 10, (B, H, W))
        
        prim_targets = {
            'primitive_ids': torch.randint(0, 15, (B, 3)),
            'param_discrete': torch.randint(0, 32, (B, 3, 8)),
            'param_continuous': torch.randn(B, 3, 8),
            'has_program': torch.ones(B, dtype=torch.bool),
        }
        
        # Track losses
        losses_over_time = []
        
        for step in range(10):
            optimizer.zero_grad()
            outputs = pg_rlan(test_input, return_primitive_outputs=True)
            losses = pg_rlan.compute_loss(outputs, targets, prim_targets)
            losses['total_loss'].backward()
            optimizer.step()
            losses_over_time.append(losses['total_loss'].item())
        
        # Loss should decrease (overfitting on fixed batch)
        assert losses_over_time[-1] < losses_over_time[0]
    
    def test_save_load_checkpoint(self, tmp_path):
        """Test saving and loading model checkpoint."""
        # Setup
        mock_rlan = MockRLAN()
        config = ProgramGuidedConfig(enabled=True)
        pg_rlan = ProgramGuidedRLAN(mock_rlan, config)
        
        # Save only state dict (not config) for weights_only=True compatibility
        checkpoint_path = tmp_path / "checkpoint.pt"
        torch.save(pg_rlan.state_dict(), checkpoint_path)
        
        # Create new model
        mock_rlan2 = MockRLAN()
        pg_rlan2 = ProgramGuidedRLAN(mock_rlan2, config)
        
        # Load with weights_only=True (PyTorch 2.6+ default)
        state_dict = torch.load(checkpoint_path, weights_only=True)
        pg_rlan2.load_state_dict(state_dict)
        
        # Compare outputs
        test_input = torch.randn(1, 10, 16, 16)
        pg_rlan.eval()
        pg_rlan2.eval()
        
        with torch.no_grad():
            out1 = pg_rlan(test_input)
            out2 = pg_rlan2(test_input)
        
        assert torch.allclose(out1, out2)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestModularIntegration:
    """Test that integration is truly modular."""
    
    def test_base_rlan_not_modified(self):
        """Test that base RLAN module is not modified."""
        mock_rlan1 = MockRLAN()
        mock_rlan2 = MockRLAN()
        
        # Create wrapper around rlan1
        config = ProgramGuidedConfig(enabled=True)
        pg_rlan = ProgramGuidedRLAN(mock_rlan1, config)
        
        # Check rlan1 and rlan2 have same structure
        assert len(list(mock_rlan1.modules())) == len(list(mock_rlan2.modules()))
        
        # rlan1 should still work standalone
        test_input = torch.randn(1, 10, 16, 16)
        out1 = mock_rlan1(test_input)
        out2 = mock_rlan2(test_input)
        
        # Both should produce same shape output
        assert out1.shape == out2.shape
    
    def test_can_disable_primitive_training(self):
        """Test primitive training can be completely disabled."""
        mock_rlan = MockRLAN()
        config = ProgramGuidedConfig(enabled=False)
        pg_rlan = ProgramGuidedRLAN(mock_rlan, config)
        
        # PrimitiveHead should be None
        assert pg_rlan.primitive_head is None
        
        # Forward should work
        test_input = torch.randn(2, 10, 16, 16)
        output = pg_rlan(test_input, return_primitive_outputs=True)
        
        assert output['logits'] is not None
        assert output['primitive_outputs'] is None
    
    def test_factory_function(self):
        """Test factory function creates correct wrapper."""
        mock_rlan = MockRLAN()
        
        # With default config
        pg_rlan = create_program_guided_rlan(mock_rlan)
        assert isinstance(pg_rlan, ProgramGuidedRLAN)
        assert pg_rlan.config.enabled
        
        # With custom config
        custom_config = ProgramGuidedConfig(
            enabled=True,
            primitive_loss_weight=0.5,
        )
        pg_rlan2 = create_program_guided_rlan(mock_rlan, custom_config)
        assert pg_rlan2.config.primitive_loss_weight == 0.5


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
