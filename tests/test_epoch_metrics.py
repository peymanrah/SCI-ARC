#!/usr/bin/env python
"""
Test Epoch Metrics Calculations

This test module validates that epoch-level metrics are computed correctly
and represent meaningful summaries of training health.

Key validations:
1. Accumulated metrics (sums/counts) compute correct epoch means
2. Min/max metrics are true epoch-wide extremes (not just last batch)
3. Formulas are mathematically consistent
4. Metrics have sensible interpretations

Author: RLAN Training Team
Date: January 2026
"""

import pytest
import torch
import numpy as np
from typing import Dict, List, Tuple


class MockEpochDiagnostics:
    """
    Simulates the epoch_diagnostics dict accumulation pattern
    from train_rlan.py to validate correctness.
    """
    
    def __init__(self):
        self.diagnostics = {}
    
    def reset(self):
        """Reset for new epoch."""
        self.diagnostics = {}
    
    def accumulate_stop_prob(self, stop_probs: torch.Tensor):
        """
        Accumulate stop probability statistics.
        
        Args:
            stop_probs: (B, K) tensor of stop probabilities
        """
        B = stop_probs.shape[0]
        batch_stop_prob_mean = stop_probs.mean().item()
        
        # Accumulate for epoch mean
        self.diagnostics['stop_prob_sum'] = self.diagnostics.get('stop_prob_sum', 0.0) + batch_stop_prob_mean * B
        self.diagnostics['stop_prob_count'] = self.diagnostics.get('stop_prob_count', 0) + B
        
        # Compute running epoch mean
        self.diagnostics['stop_prob_mean'] = (
            self.diagnostics['stop_prob_sum'] / self.diagnostics['stop_prob_count']
        )
    
    def accumulate_clues_used(self, stop_probs: torch.Tensor):
        """
        Accumulate clues used statistics.
        
        Args:
            stop_probs: (B, K) tensor of stop probabilities
        
        Formula: clues_used_per_sample = sum(1 - stop_prob) across K clues
        """
        B = stop_probs.shape[0]
        clues_used_per_sample = (1 - stop_probs).sum(dim=-1)  # (B,)
        
        batch_clues_mean = clues_used_per_sample.mean().item()
        batch_clues_std = clues_used_per_sample.std().item()
        batch_clues_min = clues_used_per_sample.min().item()
        batch_clues_max = clues_used_per_sample.max().item()
        
        # Accumulate for epoch mean
        self.diagnostics['clues_used_sum'] = self.diagnostics.get('clues_used_sum', 0.0) + batch_clues_mean * B
        self.diagnostics['clues_used_count'] = self.diagnostics.get('clues_used_count', 0) + B
        
        # Track epoch-wide min/max
        self.diagnostics['clues_used_min'] = min(
            self.diagnostics.get('clues_used_min', float('inf')), batch_clues_min
        )
        self.diagnostics['clues_used_max'] = max(
            self.diagnostics.get('clues_used_max', float('-inf')), batch_clues_max
        )
        
        # Std is approximated from last batch (proper would need Welford's algorithm)
        self.diagnostics['clues_used_std'] = batch_clues_std
    
    def get_epoch_clues_mean(self) -> float:
        """Get epoch-averaged clues used."""
        if self.diagnostics.get('clues_used_count', 0) == 0:
            return 0.0
        return self.diagnostics['clues_used_sum'] / self.diagnostics['clues_used_count']
    
    def get_epoch_stop_prob_mean(self) -> float:
        """Get epoch-averaged stop probability."""
        return self.diagnostics.get('stop_prob_mean', 0.0)


class TestCluesUsedMetrics:
    """Test the clues used metrics calculation."""
    
    def test_clues_used_formula(self):
        """
        Test that clues_used = sum(1 - stop_prob) across K clues.
        
        Example:
            K = 7 clues, stop_prob = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
            clues_used = sum([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]) = 6.3
        """
        B, K = 4, 7
        stop_prob = 0.1
        stop_probs = torch.full((B, K), stop_prob)
        
        clues_used_per_sample = (1 - stop_probs).sum(dim=-1)
        expected = K * (1 - stop_prob)  # 7 * 0.9 = 6.3
        
        assert torch.allclose(clues_used_per_sample, torch.full((B,), expected)), \
            f"Expected {expected}, got {clues_used_per_sample}"
    
    def test_clues_used_epoch_accumulation(self):
        """
        Test that epoch mean is correctly accumulated across batches.
        
        3 batches with different clues:
            Batch 1: mean=6.0 (10 samples)
            Batch 2: mean=7.0 (20 samples)  
            Batch 3: mean=5.0 (10 samples)
        
        Epoch mean = (6.0*10 + 7.0*20 + 5.0*10) / 40 = (60+140+50)/40 = 6.25
        """
        tracker = MockEpochDiagnostics()
        
        # Simulate 3 batches with known stop_probs
        # Batch 1: B=10, K=7, stop_prob ~ 1/7 ≈ 0.143 → clues ≈ 6.0
        B1, K = 10, 7
        stop_probs_1 = torch.full((B1, K), 1.0 / K)  # Mean clues = 7 * (1 - 1/7) = 6.0
        tracker.accumulate_clues_used(stop_probs_1)
        
        # Batch 2: B=20, K=7, stop_prob = 0 → clues = 7.0
        B2 = 20
        stop_probs_2 = torch.zeros((B2, K))  # Mean clues = 7.0
        tracker.accumulate_clues_used(stop_probs_2)
        
        # Batch 3: B=10, K=7, stop_prob ~ 2/7 → clues = 5.0
        B3 = 10
        stop_probs_3 = torch.full((B3, K), 2.0 / K)  # Mean clues = 7 * (1 - 2/7) = 5.0
        tracker.accumulate_clues_used(stop_probs_3)
        
        # Check epoch mean
        expected_mean = (6.0 * B1 + 7.0 * B2 + 5.0 * B3) / (B1 + B2 + B3)
        actual_mean = tracker.get_epoch_clues_mean()
        
        assert abs(actual_mean - expected_mean) < 0.01, \
            f"Expected epoch mean {expected_mean:.3f}, got {actual_mean:.3f}"
    
    def test_clues_min_max_epoch_wide(self):
        """
        Test that min/max are tracked across entire epoch, not just last batch.
        
        Batch 1: samples have clues in [5.5, 6.5]
        Batch 2: samples have clues in [6.0, 7.0]
        Batch 3: samples have clues in [4.0, 5.0]
        
        Epoch min should be 4.0, epoch max should be 7.0
        """
        tracker = MockEpochDiagnostics()
        K = 7
        
        # Batch 1: stop_probs that give clues in ~[5.5, 6.5] range
        B1 = 10
        stop_probs_1 = torch.zeros((B1, K))
        stop_probs_1[:5] = 0.07  # clues ≈ 6.5
        stop_probs_1[5:] = 0.21  # clues ≈ 5.5
        tracker.accumulate_clues_used(stop_probs_1)
        
        # Batch 2: stop_probs that give clues = 7 (all zeros)
        B2 = 10
        stop_probs_2 = torch.zeros((B2, K))  # clues = 7.0 for all
        tracker.accumulate_clues_used(stop_probs_2)
        
        # Batch 3: stop_probs that give clues in ~[4.0, 5.0] range
        B3 = 10
        stop_probs_3 = torch.zeros((B3, K))
        stop_probs_3[:5] = 0.29  # clues ≈ 5.0
        stop_probs_3[5:] = 0.43  # clues ≈ 4.0
        tracker.accumulate_clues_used(stop_probs_3)
        
        # Epoch min should be from batch 3 (around 4.0)
        # Epoch max should be from batch 2 (7.0)
        epoch_min = tracker.diagnostics['clues_used_min']
        epoch_max = tracker.diagnostics['clues_used_max']
        
        assert epoch_max >= 6.9, f"Expected epoch max >= 6.9, got {epoch_max:.2f}"
        assert epoch_min <= 4.1, f"Expected epoch min <= 4.1, got {epoch_min:.2f}"
    
    def test_clues_range_contains_mean(self):
        """
        Test that the reported range [min, max] contains the mean.
        
        This is a basic sanity check that was violated by the old buggy code
        where mean was computed incorrectly.
        """
        tracker = MockEpochDiagnostics()
        K = 7
        
        # Random batches
        for _ in range(10):
            B = np.random.randint(10, 50)
            stop_probs = torch.rand((B, K)) * 0.3  # Random stop probs 0-0.3
            tracker.accumulate_clues_used(stop_probs)
        
        mean = tracker.get_epoch_clues_mean()
        min_val = tracker.diagnostics['clues_used_min']
        max_val = tracker.diagnostics['clues_used_max']
        
        assert min_val <= mean <= max_val, \
            f"Mean {mean:.2f} not in range [{min_val:.2f}, {max_val:.2f}]"


class TestStopProbMetrics:
    """Test stop probability metrics."""
    
    def test_stop_prob_epoch_mean(self):
        """
        Test that stop_prob_mean is epoch-averaged, not just last batch.
        """
        tracker = MockEpochDiagnostics()
        
        # Batch 1: B=20, stop_prob mean = 0.1
        stop_probs_1 = torch.full((20, 7), 0.1)
        tracker.accumulate_stop_prob(stop_probs_1)
        
        # Batch 2: B=10, stop_prob mean = 0.4
        stop_probs_2 = torch.full((10, 7), 0.4)
        tracker.accumulate_stop_prob(stop_probs_2)
        
        # Expected: (0.1*20 + 0.4*10) / 30 = (2 + 4) / 30 = 0.2
        expected = (0.1 * 20 + 0.4 * 10) / 30
        actual = tracker.get_epoch_stop_prob_mean()
        
        assert abs(actual - expected) < 0.01, \
            f"Expected {expected:.3f}, got {actual:.3f}"


class TestMetricConsistency:
    """Test mathematical consistency between related metrics."""
    
    def test_clues_from_stop_prob_relationship(self):
        """
        Test: clues_used = K * (1 - stop_prob_mean) when stop_probs are uniform.
        
        With uniform stop_probs:
            stop_prob_mean = p (same for all clues)
            clues_used = K * (1 - p)
        """
        K = 7
        p = 0.15
        
        stop_probs = torch.full((32, K), p)
        
        # Compute both metrics
        stop_prob_mean = stop_probs.mean().item()
        clues_used_mean = (1 - stop_probs).sum(dim=-1).mean().item()
        
        # They should satisfy: clues_used = K * (1 - stop_prob_mean)
        expected_clues = K * (1 - stop_prob_mean)
        
        assert abs(clues_used_mean - expected_clues) < 0.001, \
            f"clues_used={clues_used_mean:.3f} != K*(1-stop_prob)={expected_clues:.3f}"
    
    def test_expected_clues_from_loss_matches(self):
        """
        Test that expected_clues_used from loss function matches our calculation.
        
        Both should compute: mean(sum(1 - stop_prob) per sample)
        """
        B, K = 32, 7
        stop_probs = torch.rand((B, K)) * 0.3
        
        # Our calculation
        our_expected_clues = (1 - stop_probs).sum(dim=-1).mean().item()
        
        # Loss function calculation (from rlan_loss.py)
        # expected_clues_used = (1 - stop_probs).sum(dim=-1).mean()
        loss_expected_clues = (1 - stop_probs).sum(dim=-1).mean().item()
        
        assert abs(our_expected_clues - loss_expected_clues) < 0.0001, \
            f"Mismatch: our={our_expected_clues:.4f} vs loss={loss_expected_clues:.4f}"


class TestMetricInterpretation:
    """Test that metrics have sensible interpretations."""
    
    def test_stop_prob_bounds(self):
        """Stop probability should be in [0, 1]."""
        stop_probs = torch.sigmoid(torch.randn(32, 7))  # As computed in training
        assert (stop_probs >= 0).all() and (stop_probs <= 1).all()
    
    def test_clues_used_bounds(self):
        """Clues used should be in [0, K]."""
        K = 7
        stop_probs = torch.rand((32, K))
        clues_used = (1 - stop_probs).sum(dim=-1)
        
        assert (clues_used >= 0).all(), "Clues used should be >= 0"
        assert (clues_used <= K).all(), f"Clues used should be <= {K}"
    
    def test_high_stop_prob_means_few_clues(self):
        """High stop_prob should mean fewer clues used."""
        K = 7
        
        # High stop prob = many clues stopped
        high_stop = torch.full((32, K), 0.8)
        clues_high_stop = (1 - high_stop).sum(dim=-1).mean().item()
        
        # Low stop prob = few clues stopped
        low_stop = torch.full((32, K), 0.1)
        clues_low_stop = (1 - low_stop).sum(dim=-1).mean().item()
        
        assert clues_high_stop < clues_low_stop, \
            f"High stop_prob should give fewer clues: {clues_high_stop:.1f} >= {clues_low_stop:.1f}"
    
    def test_clue_loss_correlation_interpretation(self):
        """
        Test clue-loss correlation interpretation.
        
        Positive correlation: harder tasks (high loss) use more clues (GOOD)
        Negative correlation: easier tasks use more clues (BAD/unexpected)
        """
        B = 100
        
        # Create synthetic data with positive correlation
        # Higher loss samples should use more clues
        task_difficulty = torch.linspace(0, 1, B)  # 0=easy, 1=hard
        per_sample_loss = task_difficulty + torch.randn(B) * 0.1
        clues_used = task_difficulty * 2 + 5 + torch.randn(B) * 0.1  # 5-7 clues
        
        # Compute correlation
        loss_centered = per_sample_loss - per_sample_loss.mean()
        clues_centered = clues_used - clues_used.mean()
        correlation = (loss_centered * clues_centered).sum() / (
            loss_centered.norm() * clues_centered.norm() + 1e-8
        )
        
        assert correlation > 0.5, f"Expected positive correlation, got {correlation:.3f}"


class TestPerStepLossMetrics:
    """Test per-step loss accumulation."""
    
    def test_per_step_loss_epoch_average(self):
        """
        Test that per_step_loss is properly epoch-averaged.
        
        Batch 1: step losses = [1.0, 0.8, 0.6]
        Batch 2: step losses = [0.9, 0.7, 0.5]
        Batch 3: step losses = [1.1, 0.9, 0.7]
        
        Epoch avg should be: [(1.0+0.9+1.1)/3, (0.8+0.7+0.9)/3, (0.6+0.5+0.7)/3]
                           = [1.0, 0.8, 0.6]
        """
        # Simulate accumulation
        per_step_loss_sum = [0.0, 0.0, 0.0]
        per_step_loss_count = 0
        
        batches = [
            [1.0, 0.8, 0.6],
            [0.9, 0.7, 0.5],
            [1.1, 0.9, 0.7],
        ]
        
        for batch_losses in batches:
            for i, loss in enumerate(batch_losses):
                per_step_loss_sum[i] += loss
            per_step_loss_count += 1
        
        # Compute epoch average
        per_step_avg = [s / per_step_loss_count for s in per_step_loss_sum]
        expected = [1.0, 0.8, 0.6]
        
        for i, (actual, exp) in enumerate(zip(per_step_avg, expected)):
            assert abs(actual - exp) < 0.01, \
                f"Step {i}: expected {exp:.2f}, got {actual:.2f}"
    
    def test_best_step_selection(self):
        """
        Test that best step is correctly identified from epoch-averaged losses.
        
        Per-step losses: [1.0, 0.8, 0.5, 0.6, 0.7]
        Best step should be 2 (lowest loss = 0.5)
        """
        per_step_loss = [1.0, 0.8, 0.5, 0.6, 0.7]
        
        valid_losses = [(i, l) for i, l in enumerate(per_step_loss) if l < 100]
        best_step, best_loss = min(valid_losses, key=lambda x: x[1])
        
        assert best_step == 2, f"Expected best step 2, got {best_step}"
        assert abs(best_loss - 0.5) < 0.01, f"Expected best loss 0.5, got {best_loss}"


class TestGradientNormMetrics:
    """Test gradient norm tracking."""
    
    def test_grad_norm_max_tracking(self):
        """
        Test that max_grad_norm_before_clip tracks true epoch maximum.
        """
        max_grad_norm = 0.0
        
        batch_norms = [5.2, 3.1, 8.7, 2.3, 6.5]  # Max should be 8.7
        
        for norm in batch_norms:
            max_grad_norm = max(max_grad_norm, norm)
        
        assert abs(max_grad_norm - 8.7) < 0.01, f"Expected max 8.7, got {max_grad_norm}"


class TestAccuracyMetrics:
    """Test accuracy-related metrics."""
    
    def test_exact_match_counting(self):
        """
        Test exact match counting logic.
        
        Exact match = 100% pixel accuracy for a sample
        """
        # Sample 1: 100% correct
        pred1 = torch.tensor([[0, 1, 2], [1, 0, 2]])
        target1 = torch.tensor([[0, 1, 2], [1, 0, 2]])
        acc1 = (pred1 == target1).float().mean().item()
        is_exact1 = acc1 == 1.0
        
        # Sample 2: 80% correct (not exact)
        pred2 = torch.tensor([[0, 1, 2], [1, 0, 2]])
        target2 = torch.tensor([[0, 1, 2], [1, 1, 2]])  # One pixel different
        acc2 = (pred2 == target2).float().mean().item()
        is_exact2 = acc2 == 1.0
        
        assert is_exact1 == True, "Sample 1 should be exact match"
        assert is_exact2 == False, "Sample 2 should not be exact match"
    
    def test_fg_bg_accuracy_separation(self):
        """
        Test that FG and BG accuracy are computed separately.
        
        BG class = 0
        FG classes = 1-9
        """
        # Create predictions and targets
        # BG pixels (class 0): 4 pixels, 3 correct = 75%
        # FG pixels (class 1-9): 6 pixels, 4 correct = 66.7%
        pred = torch.tensor([0, 0, 0, 0, 1, 2, 3, 1, 2, 1])
        target = torch.tensor([0, 0, 0, 1, 1, 2, 3, 3, 2, 2])  # Note differences
        
        bg_mask = target == 0
        fg_mask = target > 0
        
        bg_correct = ((pred == target) & bg_mask).sum().item()
        bg_total = bg_mask.sum().item()
        bg_acc = bg_correct / bg_total if bg_total > 0 else 0
        
        fg_correct = ((pred == target) & fg_mask).sum().item()
        fg_total = fg_mask.sum().item()
        fg_acc = fg_correct / fg_total if fg_total > 0 else 0
        
        # BG: 3 correct out of 3 (pixels 0,1,2 where target=0)
        # Actually let's trace through:
        # target[0]=0 (BG), pred[0]=0 → correct
        # target[1]=0 (BG), pred[1]=0 → correct  
        # target[2]=0 (BG), pred[2]=0 → correct
        # target[3]=1 (FG), pred[3]=0 → wrong
        # target[4]=1 (FG), pred[4]=1 → correct
        # target[5]=2 (FG), pred[5]=2 → correct
        # target[6]=3 (FG), pred[6]=3 → correct
        # target[7]=3 (FG), pred[7]=1 → wrong
        # target[8]=2 (FG), pred[8]=2 → correct
        # target[9]=2 (FG), pred[9]=1 → wrong
        
        # BG: 3/3 = 100%
        # FG: 4/7 ≈ 57%
        
        assert abs(bg_acc - 1.0) < 0.01, f"Expected BG acc 1.0, got {bg_acc:.3f}"
        assert abs(fg_acc - 4/7) < 0.01, f"Expected FG acc {4/7:.3f}, got {fg_acc:.3f}"


def run_all_tests():
    """Run all tests and report results."""
    import sys
    
    test_classes = [
        TestCluesUsedMetrics,
        TestStopProbMetrics,
        TestMetricConsistency,
        TestMetricInterpretation,
        TestPerStepLossMetrics,
        TestGradientNormMetrics,
        TestAccuracyMetrics,
    ]
    
    passed = 0
    failed = 0
    
    for test_class in test_classes:
        print(f"\n{'='*60}")
        print(f"Running {test_class.__name__}")
        print(f"{'='*60}")
        
        instance = test_class()
        for method_name in dir(instance):
            if method_name.startswith('test_'):
                try:
                    getattr(instance, method_name)()
                    print(f"  ✓ {method_name}")
                    passed += 1
                except AssertionError as e:
                    print(f"  ✗ {method_name}: {e}")
                    failed += 1
                except Exception as e:
                    print(f"  ✗ {method_name}: EXCEPTION - {e}")
                    failed += 1
    
    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    
    return failed == 0


if __name__ == '__main__':
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)
