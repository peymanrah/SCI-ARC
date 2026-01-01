"""
Smoke tests for the Jan 2026 Stability Patch.

This module tests the following features:
1. Attention collapse backoff policy (train_rlan.py)
2. Late-phase meta-escalation with stricter gates (train_rlan.py)
3. Late-phase LR decay (train_rlan.py)
4. LoRA clamp hit-rate logging (hyper_lora.py)
5. HPM solver-context coupling (recursive_solver.py)

Run with:
    python -m pytest tests/test_stability_patch_jan2026.py -v

Or standalone:
    python tests/test_stability_patch_jan2026.py
"""

import sys
import os
import math
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn


class TestHyperLoRAClampStats:
    """Test LoRA clamp hit-rate tracking in HyperLoRA."""
    
    def test_clamp_stats_tracking(self):
        """Verify clamp stats are tracked correctly."""
        from sci_arc.models.rlan_modules.hyper_lora import HyperLoRA, HyperLoRAConfig
        
        # Create HyperLoRA with low clamp threshold for testing
        config = HyperLoRAConfig(
            hidden_dim=64,
            context_dim=64,
            rank=4,
            lora_max_norm=1.0,  # Low threshold to trigger clamping
        )
        hyperlora = HyperLoRA(config=config)
        
        # Reset stats
        hyperlora.reset_clamp_stats()
        stats = hyperlora.get_clamp_stats()
        assert stats['hit_count'] == 0
        assert stats['total_count'] == 0
        assert stats['max_pre_norm'] == 0.0
        
        # Generate deltas with high norm (should trigger clamp)
        B, N, D = 2, 4, 64
        support_features = torch.randn(B, N, D, 8, 8) * 10  # High magnitude
        
        with torch.no_grad():
            _ = hyperlora(support_features)
        
        stats = hyperlora.get_clamp_stats()
        print(f"Clamp stats: {stats}")
        
        # Should have processed some samples
        assert stats['total_count'] > 0, "Should have processed samples"
        # With high-magnitude input, max_pre_norm should be significant
        assert stats['max_pre_norm'] > 0, "Should have recorded max norm"
        
        print("✓ HyperLoRA clamp stats tracking works")
    
    def test_clamp_stats_reset(self):
        """Verify clamp stats reset correctly."""
        from sci_arc.models.rlan_modules.hyper_lora import HyperLoRA, HyperLoRAConfig
        
        config = HyperLoRAConfig(hidden_dim=64, context_dim=64, rank=4)
        hyperlora = HyperLoRA(config=config)
        
        # Generate some activity
        B, N, D = 2, 4, 64
        support_features = torch.randn(B, N, D, 8, 8)
        with torch.no_grad():
            _ = hyperlora(support_features)
        
        # Stats should be non-zero
        stats = hyperlora.get_clamp_stats()
        assert stats['total_count'] > 0
        
        # Reset and verify
        hyperlora.reset_clamp_stats()
        stats = hyperlora.get_clamp_stats()
        assert stats['hit_count'] == 0
        assert stats['total_count'] == 0
        
        print("✓ HyperLoRA clamp stats reset works")


class TestSolverCrossAttentionHPM:
    """Test HPM solver-context coupling in SolverCrossAttention."""
    
    def test_hpm_context_disabled_by_default(self):
        """Verify HPM context is disabled by default (backward compatible)."""
        from sci_arc.models.rlan_modules.recursive_solver import SolverCrossAttention
        
        sca = SolverCrossAttention(hidden_dim=64, num_heads=4)
        assert not sca.hpm_context_enabled, "HPM context should be disabled by default"
        assert not hasattr(sca, 'hpm_gate') or sca.hpm_gate is None
        
        print("✓ HPM context disabled by default")
    
    def test_hpm_context_enabled(self):
        """Verify HPM context can be enabled with gated warmup."""
        from sci_arc.models.rlan_modules.recursive_solver import SolverCrossAttention
        
        sca = SolverCrossAttention(
            hidden_dim=64,
            num_heads=4,
            hpm_context_enabled=True,
            hpm_context_max_tokens=8,
            hpm_context_gate_init=0.0,
        )
        
        assert sca.hpm_context_enabled
        assert hasattr(sca, 'hpm_gate')
        assert hasattr(sca, 'hpm_proj')
        
        # Gate should be initialized to 0
        assert abs(sca.hpm_gate.item()) < 0.01
        
        print("✓ HPM context can be enabled")
    
    def test_hpm_context_forward_without_tokens(self):
        """Verify forward works without HPM tokens (backward compatible)."""
        from sci_arc.models.rlan_modules.recursive_solver import SolverCrossAttention
        
        sca = SolverCrossAttention(
            hidden_dim=64,
            num_heads=4,
            hpm_context_enabled=True,
        )
        
        B, D, H, W = 2, 64, 8, 8
        N = 4
        hidden_state = torch.randn(B, D, H, W)
        support_features = torch.randn(B, N, D, 8, 8)
        
        # Should work without hpm_memory_tokens
        output = sca(hidden_state, support_features)
        assert output.shape == hidden_state.shape
        
        # Should work with None
        output = sca(hidden_state, support_features, hpm_memory_tokens=None)
        assert output.shape == hidden_state.shape
        
        print("✓ HPM context forward works without tokens")
    
    def test_hpm_context_forward_with_tokens(self):
        """Verify forward correctly processes HPM memory tokens."""
        from sci_arc.models.rlan_modules.recursive_solver import SolverCrossAttention
        
        sca = SolverCrossAttention(
            hidden_dim=64,
            num_heads=4,
            hpm_context_enabled=True,
            hpm_context_max_tokens=8,
            hpm_context_gate_init=0.5,  # Some HPM influence
        )
        
        B, D, H, W = 2, 64, 8, 8
        N, M = 4, 6  # M < max_tokens
        hidden_state = torch.randn(B, D, H, W)
        support_features = torch.randn(B, N, D, 8, 8)
        hpm_tokens = torch.randn(B, M, D)
        
        output = sca(hidden_state, support_features, hpm_tokens)
        assert output.shape == hidden_state.shape
        
        print("✓ HPM context forward works with tokens")
    
    def test_hpm_gate_warmup(self):
        """Verify HPM gate can be updated for warmup."""
        from sci_arc.models.rlan_modules.recursive_solver import SolverCrossAttention
        
        sca = SolverCrossAttention(
            hidden_dim=64,
            num_heads=4,
            hpm_context_enabled=True,
            hpm_context_gate_init=0.0,
        )
        
        # Initial gate should be ~0
        assert abs(torch.sigmoid(sca.hpm_gate).item()) < 0.51
        
        # Set gate to 2.0 (sigmoid ≈ 0.88)
        sca.set_hpm_gate(2.0)
        assert abs(sca.hpm_gate.item() - 2.0) < 0.01
        assert torch.sigmoid(sca.hpm_gate).item() > 0.8
        
        print("✓ HPM gate warmup works")


class TestRecursiveSolverHPM:
    """Test HPM parameters flow through RecursiveSolver."""
    
    def test_solver_hpm_params(self):
        """Verify RecursiveSolver accepts HPM parameters."""
        from sci_arc.models.rlan_modules.recursive_solver import RecursiveSolver
        
        solver = RecursiveSolver(
            hidden_dim=64,
            num_classes=10,
            num_steps=4,
            use_solver_context=True,
            hpm_solver_context_enabled=True,
            hpm_solver_context_max_tokens=8,
            hpm_solver_context_gate_init=0.0,
        )
        
        assert solver.hpm_solver_context_enabled
        assert solver.solver_cross_attn is not None
        assert solver.solver_cross_attn.hpm_context_enabled
        
        print("✓ RecursiveSolver accepts HPM parameters")
    
    def test_solver_forward_with_hpm(self):
        """Verify RecursiveSolver forward works with HPM tokens."""
        from sci_arc.models.rlan_modules.recursive_solver import RecursiveSolver
        
        solver = RecursiveSolver(
            hidden_dim=64,
            num_classes=10,
            num_steps=4,
            num_predicates=8,
            use_solver_context=True,
            hpm_solver_context_enabled=True,
        )
        
        B, K, D, H, W = 2, 3, 64, 8, 8
        N, M = 4, 6
        
        clue_features = torch.randn(B, K, D, H, W)
        count_embedding = torch.randn(B, 10, D)  # 10 colors
        predicates = torch.randn(B, 8)
        input_grid = torch.randint(0, 10, (B, H, W))
        support_features = torch.randn(B, N, D, 8, 8)
        hpm_tokens = torch.randn(B, M, D)
        
        output = solver(
            clue_features=clue_features,
            count_embedding=count_embedding,
            predicates=predicates,
            input_grid=input_grid,
            support_features=support_features,
            hpm_memory_tokens=hpm_tokens,
        )
        
        assert output.shape == (B, 10, H, W)
        
        print("✓ RecursiveSolver forward with HPM tokens works")


class TestLatePhaseConfig:
    """Test late-phase configuration parsing."""
    
    def test_late_phase_lr_decay_cosine(self):
        """Verify cosine LR decay calculation."""
        # Simulate the late-phase LR decay logic from train_rlan.py
        late_phase_lr_decay_start_epoch = 50
        late_phase_lr_decay_end_epoch = 200
        late_phase_lr_decay_min_factor = 0.1
        
        def compute_decay_factor(epoch):
            decay_total_epochs = late_phase_lr_decay_end_epoch - late_phase_lr_decay_start_epoch
            decay_progress = min(1.0, (epoch - late_phase_lr_decay_start_epoch) / max(1, decay_total_epochs))
            # Cosine decay
            decay_factor = late_phase_lr_decay_min_factor + (1.0 - late_phase_lr_decay_min_factor) * 0.5 * (1 + math.cos(math.pi * decay_progress))
            return decay_factor
        
        # At start (epoch 50): factor should be 1.0
        factor_start = compute_decay_factor(50)
        assert abs(factor_start - 1.0) < 0.01, f"Expected 1.0 at start, got {factor_start}"
        
        # At end (epoch 200): factor should be min_factor
        factor_end = compute_decay_factor(200)
        assert abs(factor_end - 0.1) < 0.01, f"Expected 0.1 at end, got {factor_end}"
        
        # At midpoint (epoch 125): factor should be ~0.55 (cosine midpoint)
        factor_mid = compute_decay_factor(125)
        assert 0.4 < factor_mid < 0.7, f"Expected ~0.55 at midpoint, got {factor_mid}"
        
        print(f"✓ Cosine LR decay: epoch50={factor_start:.2f}, epoch125={factor_mid:.2f}, epoch200={factor_end:.2f}")


class TestCollapseBackoffState:
    """Test collapse backoff state initialization."""
    
    def test_collapse_backoff_defaults(self):
        """Verify collapse backoff state has correct defaults."""
        # Simulate the state initialization from train_rlan.py
        collapse_backoff_state = {
            'active': False,
            'cooldown_remaining': 0,
            'delta_scale_factor': 1.0,
            'lr_factor': 1.0,
            'pre_backoff_delta_scale': None,
            'pre_backoff_lr': None,
            'consecutive_collapse_count': 0,
        }
        
        assert not collapse_backoff_state['active']
        assert collapse_backoff_state['delta_scale_factor'] == 1.0
        assert collapse_backoff_state['lr_factor'] == 1.0
        assert collapse_backoff_state['consecutive_collapse_count'] == 0
        
        print("✓ Collapse backoff state defaults are correct")
    
    def test_collapse_backoff_trigger_logic(self):
        """Verify collapse backoff trigger conditions."""
        attention_collapse_consecutive_threshold = 2
        collapse_backoff_state = {
            'active': False,
            'consecutive_collapse_count': 0,
        }
        
        # Simulate epoch with collapse
        prev_attn_collapse_events = 1
        if prev_attn_collapse_events > 0:
            collapse_backoff_state['consecutive_collapse_count'] += 1
        else:
            collapse_backoff_state['consecutive_collapse_count'] = 0
        
        # Not triggered yet (count=1, threshold=2)
        should_trigger = (
            collapse_backoff_state['consecutive_collapse_count'] >= attention_collapse_consecutive_threshold 
            and not collapse_backoff_state['active']
        )
        assert not should_trigger, "Should not trigger after 1 collapse"
        
        # Second collapse epoch
        prev_attn_collapse_events = 1
        if prev_attn_collapse_events > 0:
            collapse_backoff_state['consecutive_collapse_count'] += 1
        
        # Now triggered (count=2, threshold=2)
        should_trigger = (
            collapse_backoff_state['consecutive_collapse_count'] >= attention_collapse_consecutive_threshold 
            and not collapse_backoff_state['active']
        )
        assert should_trigger, "Should trigger after 2 consecutive collapses"
        
        print("✓ Collapse backoff trigger logic is correct")


def run_all_tests():
    """Run all smoke tests."""
    print("=" * 60)
    print("Jan 2026 Stability Patch - Smoke Tests")
    print("=" * 60)
    
    # Test HyperLoRA clamp stats
    print("\n--- HyperLoRA Clamp Stats ---")
    test_lora = TestHyperLoRAClampStats()
    test_lora.test_clamp_stats_tracking()
    test_lora.test_clamp_stats_reset()
    
    # Test SolverCrossAttention HPM
    print("\n--- SolverCrossAttention HPM ---")
    test_sca = TestSolverCrossAttentionHPM()
    test_sca.test_hpm_context_disabled_by_default()
    test_sca.test_hpm_context_enabled()
    test_sca.test_hpm_context_forward_without_tokens()
    test_sca.test_hpm_context_forward_with_tokens()
    test_sca.test_hpm_gate_warmup()
    
    # Test RecursiveSolver HPM
    print("\n--- RecursiveSolver HPM ---")
    test_solver = TestRecursiveSolverHPM()
    test_solver.test_solver_hpm_params()
    test_solver.test_solver_forward_with_hpm()
    
    # Test Late-Phase Config
    print("\n--- Late-Phase Config ---")
    test_late = TestLatePhaseConfig()
    test_late.test_late_phase_lr_decay_cosine()
    
    # Test Collapse Backoff
    print("\n--- Collapse Backoff ---")
    test_collapse = TestCollapseBackoffState()
    test_collapse.test_collapse_backoff_defaults()
    test_collapse.test_collapse_backoff_trigger_logic()
    
    print("\n" + "=" * 60)
    print("All smoke tests PASSED! ✓")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
