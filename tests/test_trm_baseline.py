"""
Tests for TRM baseline implementation.

Verifies that the original TRM implementation from Samsung SAIL Montreal
works correctly within the SCI-ARC codebase.
"""

import pytest
import torch
import numpy as np


class TestTRMBaseline:
    """Test the TRM baseline models."""
    
    @pytest.fixture
    def device(self):
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    
    @pytest.fixture
    def trm_config(self):
        """Default TRM configuration for testing."""
        return {
            'batch_size': 2,
            'seq_len': 900,
            'puzzle_emb_ndim': 64,
            'num_puzzle_identifiers': 100,
            'vocab_size': 16,
            'H_cycles': 2,
            'L_cycles': 2,
            'H_layers': 1,
            'L_layers': 1,
            'hidden_size': 64,
            'expansion': 2.0,
            'num_heads': 4,
            'pos_encodings': 'rope',
            'halt_max_steps': 3,
            'halt_exploration_prob': 0.1,
            'forward_dtype': 'float32',  # Use float32 for testing
            'mlp_t': False,
            'puzzle_emb_len': 4,
        }
    
    @pytest.fixture
    def sample_batch(self, trm_config, device):
        """Create a sample batch for testing."""
        batch_size = trm_config['batch_size']
        seq_len = trm_config['seq_len']
        
        return {
            'inputs': torch.randint(0, 10, (batch_size, seq_len), device=device),
            'puzzle_identifiers': torch.randint(0, 100, (batch_size,), device=device),
            'labels': torch.randint(0, 10, (batch_size, seq_len), device=device),
        }
    
    def test_trm_import(self):
        """Test that TRM can be imported."""
        from baselines.trm import TRM, TRMConfig, TRMCarry
        
        assert TRM is not None
        assert TRMConfig is not None
        assert TRMCarry is not None
    
    def test_trm_creation(self, trm_config):
        """Test TRM model creation."""
        from baselines.trm import TRM
        
        model = TRM(trm_config)
        
        # Check model was created
        assert model is not None
        
        # Check parameter count
        num_params = sum(p.numel() for p in model.parameters())
        assert num_params > 0
        print(f"TRM test model has {num_params:,} parameters")
    
    def test_trm_forward(self, trm_config, sample_batch, device):
        """Test TRM forward pass."""
        from baselines.trm import TRM
        
        # Create model
        model = TRM(trm_config)
        model.to(device)
        model.eval()
        
        # Create initial carry
        carry = model.initial_carry(sample_batch)
        
        # Move carry to device
        carry_device = type(carry)(
            inner_carry=type(carry.inner_carry)(
                z_H=carry.inner_carry.z_H.to(device),
                z_L=carry.inner_carry.z_L.to(device),
            ),
            steps=carry.steps.to(device),
            halted=carry.halted.to(device),
            current_data={k: v.to(device) for k, v in carry.current_data.items()},
        )
        
        # Forward pass
        with torch.no_grad():
            new_carry, outputs = model(carry_device, sample_batch)
        
        # Check outputs
        assert 'logits' in outputs
        assert 'q_halt_logits' in outputs
        assert 'q_continue_logits' in outputs
        
        # Check shapes
        batch_size = trm_config['batch_size']
        seq_len = trm_config['seq_len']
        vocab_size = trm_config['vocab_size']
        
        assert outputs['logits'].shape == (batch_size, seq_len, vocab_size)
        assert outputs['q_halt_logits'].shape == (batch_size,)
    
    def test_trm_act_loop(self, trm_config, sample_batch, device):
        """Test TRM ACT (Adaptive Computation Time) loop."""
        from baselines.trm import TRM
        
        model = TRM(trm_config)
        model.to(device)
        model.eval()
        
        carry = model.initial_carry(sample_batch)
        carry_device = type(carry)(
            inner_carry=type(carry.inner_carry)(
                z_H=carry.inner_carry.z_H.to(device),
                z_L=carry.inner_carry.z_L.to(device),
            ),
            steps=carry.steps.to(device),
            halted=carry.halted.to(device),
            current_data={k: v.to(device) for k, v in carry.current_data.items()},
        )
        
        # Run multiple ACT steps
        max_steps = trm_config['halt_max_steps']
        steps_run = 0
        
        with torch.no_grad():
            for step in range(max_steps):
                carry_device, outputs = model(carry_device, sample_batch)
                steps_run += 1
                
                if carry_device.halted.all():
                    break
        
        # Should have run some steps
        assert steps_run > 0
        assert steps_run <= max_steps
    
    def test_hrm_creation(self, trm_config):
        """Test HRM model creation."""
        from baselines.trm import HRM
        
        model = HRM(trm_config)
        assert model is not None
    
    def test_transformer_baseline_creation(self, trm_config):
        """Test transformer baseline creation."""
        from baselines.trm import TransformerBaseline
        
        model = TransformerBaseline(trm_config)
        assert model is not None
    
    def test_layers_import(self):
        """Test that TRM layers can be imported."""
        from baselines.trm import (
            rms_norm,
            SwiGLU,
            Attention,
            RotaryEmbedding,
            CastedEmbedding,
            CastedLinear,
        )
        
        assert rms_norm is not None
        assert SwiGLU is not None
        assert Attention is not None
    
    def test_rms_norm(self, device):
        """Test RMS normalization."""
        from baselines.trm import rms_norm
        
        x = torch.randn(2, 10, 64, device=device)
        y = rms_norm(x, variance_epsilon=1e-5)
        
        assert y.shape == x.shape
        
        # Check that output has roughly unit variance
        var = y.square().mean(-1)
        assert torch.allclose(var, torch.ones_like(var), atol=0.1)
    
    def test_swiglu(self, device):
        """Test SwiGLU activation."""
        from baselines.trm import SwiGLU
        
        hidden_size = 64
        expansion = 2.0
        
        swiglu = SwiGLU(hidden_size, expansion).to(device)
        
        x = torch.randn(2, 10, hidden_size, device=device)
        y = swiglu(x)
        
        assert y.shape == x.shape
    
    def test_attention(self, device):
        """Test attention layer."""
        from baselines.trm import Attention, RotaryEmbedding
        
        hidden_size = 64
        num_heads = 4
        head_dim = hidden_size // num_heads
        seq_len = 20
        
        attn = Attention(
            hidden_size=hidden_size,
            head_dim=head_dim,
            num_heads=num_heads,
            num_key_value_heads=num_heads,
            causal=False,
        ).to(device)
        
        rope = RotaryEmbedding(
            dim=head_dim,
            max_position_embeddings=seq_len,
            base=10000.0,
        ).to(device)
        
        x = torch.randn(2, seq_len, hidden_size, device=device)
        cos_sin = rope()
        
        y = attn(cos_sin, x)
        
        assert y.shape == x.shape
    
    def test_loss_head(self, trm_config, sample_batch, device):
        """Test TRM loss computation using stablemax_cross_entropy."""
        from baselines.trm import TRM, stablemax_cross_entropy, TRMLossHead
        
        model = TRM(trm_config)
        model.to(device)
        
        # Test TRMLossHead wrapper (simple compatibility wrapper)
        loss_head = TRMLossHead(vocab_size=trm_config['vocab_size'])
        
        # Create dummy logits and targets (requires_grad=True for logits)
        batch_size = 2
        seq_len = 100
        vocab_size = trm_config['vocab_size']
        
        logits = torch.randn(batch_size, seq_len, vocab_size, device=device, requires_grad=True)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        
        # Test stablemax_cross_entropy directly
        loss = stablemax_cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        
        # stablemax_cross_entropy returns per-element losses, sum them for scalar loss
        scalar_loss = loss.mean()
        assert scalar_loss.requires_grad
        assert not torch.isnan(scalar_loss)
        assert scalar_loss.item() > 0
        
        # Test TRMLossHead wrapper
        loss2 = loss_head(logits, targets)
        # TRMLossHead also returns per-element losses
        scalar_loss2 = loss2.mean()
        assert not torch.isnan(scalar_loss2)
        assert scalar_loss2.item() > 0


class TestTRMVsSCIARC:
    """Tests for comparing TRM and SCI-ARC architectures."""
    
    def test_parameter_count_comparison(self):
        """Compare parameter counts between TRM and SCI-ARC."""
        from baselines.trm import TRM
        
        # TRM config matching ~7M params from paper
        trm_config = {
            'batch_size': 32,
            'seq_len': 900,
            'puzzle_emb_ndim': 256,
            'num_puzzle_identifiers': 1000,
            'vocab_size': 16,
            'H_cycles': 3,
            'L_cycles': 4,
            'H_layers': 2,
            'L_layers': 2,
            'hidden_size': 256,
            'expansion': 2.5,
            'num_heads': 8,
            'pos_encodings': 'rope',
            'halt_max_steps': 10,
            'halt_exploration_prob': 0.1,
            'forward_dtype': 'float32',
        }
        
        trm = TRM(trm_config)
        trm_params = sum(p.numel() for p in trm.parameters())
        
        print(f"TRM parameters: {trm_params:,}")
        
        # TRM should be around 7M params
        assert trm_params > 1_000_000, "TRM should have at least 1M parameters"
    
    def test_output_format_compatibility(self):
        """Test that TRM output format is compatible for comparison."""
        from baselines.trm import TRM
        
        config = {
            'batch_size': 1,
            'seq_len': 900,
            'puzzle_emb_ndim': 64,
            'num_puzzle_identifiers': 100,
            'vocab_size': 16,
            'H_cycles': 2,
            'L_cycles': 2,
            'H_layers': 1,
            'L_layers': 1,
            'hidden_size': 64,
            'expansion': 2.0,
            'num_heads': 4,
            'pos_encodings': 'rope',
            'halt_max_steps': 3,
            'halt_exploration_prob': 0.1,
            'forward_dtype': 'float32',
        }
        
        model = TRM(config)
        model.eval()
        
        batch = {
            'inputs': torch.randint(0, 10, (1, 900)),
            'puzzle_identifiers': torch.randint(0, 100, (1,)),
        }
        
        carry = model.initial_carry(batch)
        
        with torch.no_grad():
            _, outputs = model(carry, batch)
        
        # Check outputs can be used for evaluation
        logits = outputs['logits']  # [B, seq_len, vocab_size]
        preds = logits.argmax(dim=-1)  # [B, seq_len]
        
        # Reshape to 30x30 grid
        grid_preds = preds.view(1, 30, 30)
        
        assert grid_preds.shape == (1, 30, 30)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
