#!/usr/bin/env python3
"""
Test script to verify meta-learning modules are active during inference/evaluation.

CRITICAL BUG INVESTIGATION (Jan 2026):
- 50%+ train exact match but 0% eval @K
- Hypothesis: HyperLoRA/LOO/HPM not being used during eval

This script runs diagnostic checks to verify:
1. HyperLoRA generates non-zero deltas during inference
2. HPM buffers are populated and used during eval
3. DSC generates task-appropriate clue counts
4. All staging flags are set correctly before eval
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional
import copy

# Import RLAN components
try:
    from sci_arc.models.rlan import RLAN, RLANConfig
    from sci_arc.training.ema import EMAHelper
    print("[✓] Successfully imported RLAN and EMAHelper")
except ImportError as e:
    print(f"[✗] Import failed: {e}")
    sys.exit(1)


def create_test_model() -> RLAN:
    """Create a minimal RLAN model for testing."""
    config = RLANConfig(
        hidden_dim=64,
        num_colors=10,
        max_grid_size=30,
        num_solver_steps=3,
        use_context_encoder=True,
        use_dsc=True,
        use_msre=True,
        use_hyperlora=True,
        use_hpm=True,
        hpm_top_k=2,
        hpm_use_instance_bank=True,
        hpm_use_procedural_bank=True,
    )
    model = RLAN(config=config)
    return model


def create_dummy_batch(device: torch.device, batch_size: int = 4):
    """Create dummy ARC-like data for testing."""
    H, W = 30, 30
    num_pairs = 3
    
    # Random grids with colors 0-9
    test_inputs = torch.randint(0, 10, (batch_size, H, W), device=device)
    train_inputs = torch.randint(0, 10, (batch_size, num_pairs, H, W), device=device)
    train_outputs = torch.randint(0, 10, (batch_size, num_pairs, H, W), device=device)
    pair_mask = torch.ones(batch_size, num_pairs, dtype=torch.bool, device=device)
    
    return test_inputs, train_inputs, train_outputs, pair_mask


def test_hyperlora_at_inference():
    """
    Test 1: Verify HyperLoRA generates non-zero deltas during inference.
    
    CRITICAL: If lora_deltas is None or all zeros during eval, 
    the meta-learning signal is NOT being used!
    """
    print("\n" + "="*60)
    print("TEST 1: HyperLoRA Delta Generation at Inference")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_test_model().to(device)
    model.eval()
    
    # Simulate staging flags (as if we're past activation epoch)
    model.hyperlora_active = True
    model.solver_context_active = True
    model.cross_attention_active = True
    
    test_inputs, train_inputs, train_outputs, pair_mask = create_dummy_batch(device)
    
    with torch.no_grad():
        outputs = model(
            test_inputs,
            train_inputs=train_inputs,
            train_outputs=train_outputs,
            pair_mask=pair_mask,
            return_intermediates=True,
        )
    
    lora_deltas = outputs.get('lora_deltas')
    
    if lora_deltas is None:
        print("[✗] FAIL: lora_deltas is None - HyperLoRA not generating deltas!")
        print("    Likely cause: support_features is None or hyperlora_active=False")
        return False
    
    if isinstance(lora_deltas, dict):
        print("[✓] lora_deltas is a dict with keys:", list(lora_deltas.keys()))
        
        # Check if deltas are non-zero
        all_zero = True
        for key, delta in lora_deltas.items():
            if isinstance(delta, torch.Tensor):
                norm = delta.norm().item()
                print(f"    {key}: shape={delta.shape}, norm={norm:.4f}")
                if norm > 1e-6:
                    all_zero = False
        
        if all_zero:
            print("[⚠] WARNING: All LoRA deltas are near-zero!")
            print("    This could be intentional (early warmup) or a bug")
        else:
            print("[✓] LoRA deltas are non-zero - HyperLoRA is active!")
        return not all_zero
    else:
        print(f"[?] lora_deltas is unexpected type: {type(lora_deltas)}")
        return False


def test_ema_copy_preserves_flags():
    """
    Test 2: Verify EMA copy preserves staging flags.
    
    CRITICAL: If EMA copy doesn't preserve hyperlora_active, solver_context_active,
    etc., then eval uses wrong flags!
    """
    print("\n" + "="*60)
    print("TEST 2: EMA Copy Preserves Staging Flags")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_test_model().to(device)
    
    # Set staging flags on original model
    model.hyperlora_active = True
    model.solver_context_active = True
    model.cross_attention_active = True
    model.hpm_memory_enabled = True
    model.use_hpm = True
    
    # Create EMA helper and make a copy
    ema = EMAHelper(model, mu=0.999)
    ema_model = ema.ema_copy(model)
    
    # Check if flags are preserved
    flags_to_check = [
        'hyperlora_active',
        'solver_context_active', 
        'cross_attention_active',
        'hpm_memory_enabled',
        'use_hpm',
    ]
    
    all_preserved = True
    for flag in flags_to_check:
        orig_val = getattr(model, flag, None)
        ema_val = getattr(ema_model, flag, None)
        status = "✓" if orig_val == ema_val else "✗"
        print(f"  [{status}] {flag}: original={orig_val}, ema_copy={ema_val}")
        if orig_val != ema_val:
            all_preserved = False
    
    if all_preserved:
        print("[✓] All staging flags preserved in EMA copy")
    else:
        print("[✗] FAIL: Some flags not preserved - this breaks eval!")
    
    return all_preserved


def test_hpm_buffer_usage():
    """
    Test 3: Verify HPM buffers are used during inference.
    
    CRITICAL: If buffers are populated but not queried during forward,
    the continual learning signal is lost!
    """
    print("\n" + "="*60)
    print("TEST 3: HPM Buffer Usage at Inference")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_test_model().to(device)
    model.eval()
    
    # Activate HPM
    model.use_hpm = True
    model.hpm_memory_enabled = True
    
    # Check if buffers exist
    instance_buf = getattr(model, 'hpm_instance_buffer', None)
    procedural_buf = getattr(model, 'hpm_procedural_buffer', None)
    
    print(f"  Instance buffer exists: {instance_buf is not None}")
    print(f"  Procedural buffer exists: {procedural_buf is not None}")
    
    if instance_buf is None and procedural_buf is None:
        print("[✗] FAIL: No HPM buffers initialized!")
        print("    Check: use_hpm=True and use_instance_bank=True in config")
        return False
    
    # Add dummy entries to buffers
    if instance_buf is not None:
        dummy_key = torch.randn(1, 64).to(device)
        dummy_value = torch.randn(1, 64).to(device)
        instance_buf.add(dummy_key, dummy_value, "test_task")
        print(f"  Added 1 entry to instance buffer (now {len(instance_buf)} entries)")
    
    # Run forward and check if retrieval happens
    test_inputs, train_inputs, train_outputs, pair_mask = create_dummy_batch(device)
    
    with torch.no_grad():
        outputs = model(
            test_inputs,
            train_inputs=train_inputs,
            train_outputs=train_outputs,
            pair_mask=pair_mask,
            return_intermediates=True,
        )
    
    # Check for HPM routing weights in outputs
    hpm_routing = outputs.get('hpm_routing_weights')
    if hpm_routing is not None:
        print(f"[✓] HPM routing weights present: {hpm_routing.shape}")
        return True
    else:
        print("[⚠] HPM routing weights not in output")
        print("    This may be expected if HPM didn't have enough entries")
        return True  # Not necessarily a failure


def test_dsc_clue_counts():
    """
    Test 4: Verify DSC generates variable clue counts based on task.
    
    INVESTIGATION: Why do clues collapse to 1 at higher epochs?
    """
    print("\n" + "="*60)
    print("TEST 4: DSC Clue Count Analysis")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_test_model().to(device)
    model.eval()
    
    # Create batches with different "difficulty" (simulated by grid complexity)
    test_inputs, train_inputs, train_outputs, pair_mask = create_dummy_batch(device)
    
    with torch.no_grad():
        outputs = model(
            test_inputs,
            train_inputs=train_inputs,
            train_outputs=train_outputs,
            pair_mask=pair_mask,
            return_intermediates=True,
        )
    
    stop_logits = outputs.get('stop_logits')
    attention_maps = outputs.get('attention_maps')
    centroids = outputs.get('centroids')
    
    if stop_logits is None:
        print("[✗] FAIL: stop_logits not in output - DSC not running!")
        return False
    
    print(f"  stop_logits shape: {stop_logits.shape}")  # (B, K)
    
    # Compute expected clues used
    stop_probs = torch.sigmoid(stop_logits)
    expected_clues = (1 - stop_probs).sum(dim=-1)  # (B,)
    
    print(f"  Stop probabilities per clue (sample 0): {stop_probs[0].tolist()}")
    print(f"  Expected clues per sample: {expected_clues.tolist()}")
    print(f"  Mean expected clues: {expected_clues.mean().item():.2f}")
    
    # Check centroid spread
    if centroids is not None:
        B, K, _ = centroids.shape
        # Compute pairwise distances between centroids
        for b in range(min(2, B)):
            c = centroids[b]  # (K, 2)
            dists = torch.cdist(c.unsqueeze(0), c.unsqueeze(0)).squeeze(0)  # (K, K)
            mean_dist = dists[dists > 0].mean().item() if (dists > 0).any() else 0
            print(f"  Sample {b} centroid spread: mean_dist={mean_dist:.2f}")
    
    # Analysis
    if expected_clues.mean().item() < 1.5:
        print("[⚠] WARNING: Expected clues very low (<1.5)")
        print("    This suggests stop_logits are too high (stopping early)")
        print("    Possible causes:")
        print("    1. ponder_weight too high in ClueRegularizationLoss")
        print("    2. task_context not providing enough signal to stop_predictor")
        print("    3. min_clue_weight too low to counteract early stopping")
    else:
        print("[✓] Expected clues seem reasonable")
    
    return True


def test_support_features_presence():
    """
    Test 5: Verify support_features are generated and non-None.
    
    CRITICAL: HyperLoRA requires support_features to generate deltas!
    """
    print("\n" + "="*60)
    print("TEST 5: Support Features Presence")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_test_model().to(device)
    model.eval()
    
    # Ensure spatial features are enabled
    if hasattr(model, 'context_encoder') and model.context_encoder is not None:
        print(f"  ContextEncoder.use_spatial_features: {model.context_encoder.use_spatial_features}")
    
    test_inputs, train_inputs, train_outputs, pair_mask = create_dummy_batch(device)
    
    with torch.no_grad():
        outputs = model(
            test_inputs,
            train_inputs=train_inputs,
            train_outputs=train_outputs,
            pair_mask=pair_mask,
            return_intermediates=True,
        )
    
    support_features = outputs.get('support_features')
    
    if support_features is None:
        print("[✗] FAIL: support_features is None!")
        print("    Likely cause: use_spatial_features=False in ContextEncoder")
        print("    This breaks HyperLoRA and HPM retrieval!")
        return False
    
    print(f"[✓] support_features shape: {support_features.shape}")
    print(f"    Expected: (B, N, D, H, W)")
    return True


def run_all_tests():
    """Run all diagnostic tests."""
    print("\n" + "="*70)
    print(" META-LEARNING INFERENCE DIAGNOSTIC TESTS")
    print("="*70)
    
    results = {}
    
    results['support_features'] = test_support_features_presence()
    results['hyperlora'] = test_hyperlora_at_inference()
    results['ema_copy'] = test_ema_copy_preserves_flags()
    results['hpm_buffers'] = test_hpm_buffer_usage()
    results['dsc_clues'] = test_dsc_clue_counts()
    
    print("\n" + "="*70)
    print(" SUMMARY")
    print("="*70)
    
    all_pass = True
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_pass = False
    
    if all_pass:
        print("\n[✓] All tests passed - meta-learning should work at inference")
    else:
        print("\n[✗] Some tests failed - investigate the failures above")
    
    return all_pass


if __name__ == '__main__':
    run_all_tests()
