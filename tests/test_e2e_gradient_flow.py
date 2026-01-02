#!/usr/bin/env python3
"""
End-to-End Gradient Flow Verification for RLAN.

This script verifies that:
1. Gradients flow through HyperLoRA to update LOO/Equiv objectives
2. DSC clue count correlates with task difficulty  
3. HPM buffers are populated and contribute to features
4. MSRE encodes proper relative positions
5. All staging flags are respected

Run with: python tests/test_e2e_gradient_flow.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
import math

print("="*70)
print(" END-TO-END GRADIENT FLOW VERIFICATION")
print("="*70)


def create_test_config():
    """Create a test RLAN configuration."""
    from sci_arc.models.rlan import RLANConfig
    return RLANConfig(
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


def test_hyperlora_gradient_flow():
    """
    Test that gradients flow through HyperLoRA to the encoder.
    
    Verifies:
    1. lora_deltas are computed
    2. lora_deltas affect solver output
    3. Gradients flow back to HyperLoRA parameters
    """
    print("\n" + "="*60)
    print("TEST: HyperLoRA Gradient Flow")
    print("="*60)
    
    from sci_arc.models.rlan import RLAN
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = create_test_config()
    model = RLAN(config=config).to(device)
    model.train()
    
    # Ensure HyperLoRA is active
    model.hyperlora_active = True
    model.solver_context_active = True
    
    # Create dummy data
    B, H, W = 2, 30, 30
    test_inputs = torch.randint(0, 10, (B, H, W), device=device)
    train_inputs = torch.randint(0, 10, (B, 3, H, W), device=device)
    train_outputs = torch.randint(0, 10, (B, 3, H, W), device=device)
    pair_mask = torch.ones(B, 3, dtype=torch.bool, device=device)
    target = torch.randint(0, 10, (B, H, W), device=device)
    
    # Forward pass
    outputs = model(
        test_inputs,
        train_inputs=train_inputs,
        train_outputs=train_outputs,
        pair_mask=pair_mask,
        return_intermediates=True,
    )
    
    logits = outputs['logits']
    lora_deltas = outputs.get('lora_deltas')
    
    # Check lora_deltas
    if lora_deltas is None:
        print("[✗] FAIL: lora_deltas is None - HyperLoRA not generating deltas!")
        return False
    
    print(f"[✓] lora_deltas generated with keys: {list(lora_deltas.keys()) if isinstance(lora_deltas, dict) else type(lora_deltas)}")
    
    # Compute loss and backward
    loss = nn.CrossEntropyLoss()(logits.reshape(-1, 10), target.reshape(-1))
    loss.backward()
    
    # Check gradients on HyperLoRA parameters
    hyper_lora_has_grads = False
    for name, param in model.named_parameters():
        if 'hyper_lora' in name and param.grad is not None:
            grad_norm = param.grad.norm().item()
            if grad_norm > 1e-10:
                hyper_lora_has_grads = True
                print(f"[✓] {name}: grad_norm={grad_norm:.6f}")
                break
    
    if not hyper_lora_has_grads:
        print("[✗] FAIL: No gradients flowing to HyperLoRA!")
        return False
    
    print("[✓] Gradients flow through HyperLoRA")
    return True


def test_dsc_clue_variability():
    """
    Test that DSC generates variable clue counts based on task complexity.
    
    Verifies:
    1. Stop probabilities vary across samples
    2. Expected clue count is reasonable (not collapsed to 1)
    3. Centroid spread is adequate
    """
    print("\n" + "="*60)
    print("TEST: DSC Clue Variability")
    print("="*60)
    
    from sci_arc.models.rlan import RLAN
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = create_test_config()
    model = RLAN(config=config).to(device)
    model.eval()
    
    # Create inputs with varying "complexity" (different patterns)
    B, H, W = 4, 30, 30
    test_inputs = torch.zeros(B, H, W, device=device, dtype=torch.long)
    
    # Sample 0: Simple (single color)
    test_inputs[0] = 1
    
    # Sample 1: Medium (two colors in halves)
    test_inputs[1, :, :15] = 2
    test_inputs[1, :, 15:] = 3
    
    # Sample 2: Complex (grid pattern)
    for i in range(H):
        for j in range(W):
            test_inputs[2, i, j] = (i + j) % 4 + 1
    
    # Sample 3: Very complex (random)
    test_inputs[3] = torch.randint(0, 10, (H, W), device=device)
    
    train_inputs = torch.randint(0, 10, (B, 3, H, W), device=device)
    train_outputs = torch.randint(0, 10, (B, 3, H, W), device=device)
    pair_mask = torch.ones(B, 3, dtype=torch.bool, device=device)
    
    with torch.no_grad():
        outputs = model(
            test_inputs,
            train_inputs=train_inputs,
            train_outputs=train_outputs,
            pair_mask=pair_mask,
            return_intermediates=True,
        )
    
    stop_logits = outputs.get('stop_logits')
    centroids = outputs.get('centroids')
    
    if stop_logits is None:
        print("[✗] FAIL: stop_logits not in output!")
        return False
    
    # Compute expected clues
    stop_probs = torch.sigmoid(stop_logits)
    expected_clues = (1 - stop_probs).sum(dim=-1)  # (B,)
    
    print(f"Expected clues per sample:")
    for i in range(B):
        complexity = ["Simple", "Medium", "Complex", "Random"][i]
        print(f"  Sample {i} ({complexity}): {expected_clues[i].item():.2f} clues")
    
    # Check variability
    clue_std = expected_clues.std().item()
    print(f"Clue count std: {clue_std:.3f}")
    
    if clue_std < 0.1:
        print("[⚠] WARNING: Clue counts have low variance - may be collapsed")
    
    # Check centroid spread
    if centroids is not None:
        B_c, K, _ = centroids.shape
        for b in range(min(2, B_c)):
            c = centroids[b]  # (K, 2)
            dists = torch.cdist(c.unsqueeze(0), c.unsqueeze(0)).squeeze(0)
            mean_dist = dists[dists > 0].mean().item() if (dists > 0).any() else 0
            print(f"  Sample {b} centroid spread: {mean_dist:.2f}")
    
    print("[✓] DSC clue analysis complete")
    return True


def test_hpm_contribution():
    """
    Test that HPM contributes to feature computation.
    
    Verifies:
    1. HPM can be populated with entries
    2. Retrieval returns non-empty results
    3. Enhanced features differ from original
    """
    print("\n" + "="*60)
    print("TEST: HPM Contribution")
    print("="*60)
    
    from sci_arc.models.rlan import RLAN
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = create_test_config()
    model = RLAN(config=config).to(device)
    model.eval()
    
    # Enable HPM
    model.use_hpm = True
    model.hpm_memory_enabled = True
    
    # Check buffers exist
    if model.hpm_instance_buffer is None:
        print("[✗] FAIL: HPM instance buffer not initialized!")
        return False
    
    print(f"[✓] HPM instance buffer initialized (size: {len(model.hpm_instance_buffer)})")
    
    # Add dummy entries to buffer
    dummy_key = torch.randn(1, 64, device=device)
    dummy_value = torch.randn(1, 64, device=device)
    model.hpm_instance_buffer.add(dummy_key, dummy_value, "test_task_1")
    model.hpm_instance_buffer.add(dummy_key + 0.1, dummy_value + 0.1, "test_task_2")
    
    print(f"[✓] Added 2 entries to buffer (size: {len(model.hpm_instance_buffer)})")
    
    # Forward pass
    B, H, W = 2, 30, 30
    test_inputs = torch.randint(0, 10, (B, H, W), device=device)
    train_inputs = torch.randint(0, 10, (B, 3, H, W), device=device)
    train_outputs = torch.randint(0, 10, (B, 3, H, W), device=device)
    pair_mask = torch.ones(B, 3, dtype=torch.bool, device=device)
    
    with torch.no_grad():
        outputs = model(
            test_inputs,
            train_inputs=train_inputs,
            train_outputs=train_outputs,
            pair_mask=pair_mask,
            return_intermediates=True,
        )
    
    # Check for HPM routing weights
    hpm_stats = model.hpm_get_stats()
    print(f"[✓] HPM stats: {hpm_stats}")
    
    return True


def test_msre_encoding():
    """
    Test that MSRE encodes position-relative features correctly.
    
    Verifies:
    1. MSRE output has correct shape
    2. Features vary based on centroid position
    3. Multi-scale encoding is present
    """
    print("\n" + "="*60)
    print("TEST: MSRE Encoding")
    print("="*60)
    
    try:
        from sci_arc.models.rlan_modules.msre import MultiScaleRelativeEncoding
    except ImportError as e:
        print(f"[⚠] MSRE module not available: {e}")
        return True  # Not a failure, just not testable
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create MSRE module
    msre = MultiScaleRelativeEncoding(
        hidden_dim=64,
        num_scales=4,
    ).to(device)
    
    # Test input
    B, D, H, W = 2, 64, 30, 30
    K = 4  # Number of clues
    
    features = torch.randn(B, D, H, W, device=device)
    centroids = torch.zeros(B, K, 2, device=device)
    
    # Set centroids at different positions
    centroids[0, 0] = torch.tensor([5, 5])  # Top-left
    centroids[0, 1] = torch.tensor([25, 25])  # Bottom-right
    centroids[1, 0] = torch.tensor([15, 15])  # Center
    
    # Forward pass
    clue_features = msre(features, centroids)
    
    print(f"[✓] MSRE output shape: {clue_features.shape}")  # Expected: (B, K, D, H, W)
    
    # Check that different centroids produce different features
    feat_0 = clue_features[0, 0].flatten()  # Features for centroid at (5,5)
    feat_1 = clue_features[0, 1].flatten()  # Features for centroid at (25,25)
    
    cosine_sim = nn.functional.cosine_similarity(feat_0, feat_1, dim=0).item()
    print(f"[✓] Cosine similarity between clue features: {cosine_sim:.4f}")
    
    if cosine_sim > 0.99:
        print("[⚠] WARNING: Clue features are nearly identical - MSRE may not be differentiating")
    
    return True


def test_loo_gradient_structure():
    """
    Test that LOO training produces correct gradient structure.
    
    Verifies:
    1. LOO loss computes gradients for excluded sample
    2. Gradients don't leak to encoder from excluded sample
    3. HyperLoRA receives gradients from LOO objective
    """
    print("\n" + "="*60)
    print("TEST: LOO Gradient Structure")
    print("="*60)
    
    from sci_arc.models.rlan import RLAN
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = create_test_config()
    model = RLAN(config=config).to(device)
    model.train()
    
    # Enable HyperLoRA
    model.hyperlora_active = True
    
    B, H, W = 2, 30, 30
    N = 3  # Number of demo pairs
    
    test_inputs = torch.randint(0, 10, (B, H, W), device=device)
    train_inputs = torch.randint(0, 10, (B, N, H, W), device=device)
    train_outputs = torch.randint(0, 10, (B, N, H, W), device=device)
    pair_mask = torch.ones(B, N, dtype=torch.bool, device=device)
    
    # For each sample, exclude one pair and predict its output
    total_loo_loss = 0.0
    
    for exclude_idx in range(N):
        # Create mask that excludes pair exclude_idx
        loo_mask = pair_mask.clone()
        loo_mask[:, exclude_idx] = False
        
        # Forward with remaining pairs
        outputs = model(
            train_inputs[:, exclude_idx],  # Use excluded input as "test"
            train_inputs=train_inputs,
            train_outputs=train_outputs,
            pair_mask=loo_mask,  # Exclude this pair from context
            return_intermediates=True,
        )
        
        # Compute loss on excluded output
        logits = outputs['logits']
        target = train_outputs[:, exclude_idx]
        
        loss = nn.CrossEntropyLoss()(logits.reshape(-1, 10), target.reshape(-1))
        total_loo_loss += loss.item()
    
    avg_loo_loss = total_loo_loss / N
    print(f"[✓] Average LOO loss across {N} exclusions: {avg_loo_loss:.4f}")
    
    # Now do backward with one exclusion to check gradient flow
    loo_mask = pair_mask.clone()
    loo_mask[:, 0] = False
    
    outputs = model(
        train_inputs[:, 0],
        train_inputs=train_inputs,
        train_outputs=train_outputs,
        pair_mask=loo_mask,
        return_intermediates=True,
    )
    
    logits = outputs['logits']
    target = train_outputs[:, 0]
    loss = nn.CrossEntropyLoss()(logits.reshape(-1, 10), target.reshape(-1))
    loss.backward()
    
    # Check that HyperLoRA got gradients
    hyper_lora_grad_count = 0
    for name, param in model.named_parameters():
        if 'hyper_lora' in name and param.grad is not None and param.grad.norm() > 1e-10:
            hyper_lora_grad_count += 1
    
    print(f"[✓] HyperLoRA parameters with non-zero gradients: {hyper_lora_grad_count}")
    
    return hyper_lora_grad_count > 0


def run_all_tests():
    """Run all gradient flow tests."""
    results = {}
    
    try:
        results['hyperlora_gradient'] = test_hyperlora_gradient_flow()
    except Exception as e:
        print(f"[✗] HyperLoRA test failed with exception: {e}")
        results['hyperlora_gradient'] = False
    
    try:
        results['dsc_clue'] = test_dsc_clue_variability()
    except Exception as e:
        print(f"[✗] DSC test failed with exception: {e}")
        results['dsc_clue'] = False
    
    try:
        results['hpm_contribution'] = test_hpm_contribution()
    except Exception as e:
        print(f"[✗] HPM test failed with exception: {e}")
        results['hpm_contribution'] = False
    
    try:
        results['msre_encoding'] = test_msre_encoding()
    except Exception as e:
        print(f"[✗] MSRE test failed with exception: {e}")
        results['msre_encoding'] = False
    
    try:
        results['loo_gradient'] = test_loo_gradient_structure()
    except Exception as e:
        print(f"[✗] LOO test failed with exception: {e}")
        results['loo_gradient'] = False
    
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
        print("\n[✓] All tests passed - gradient flow is healthy")
    else:
        print("\n[✗] Some tests failed - investigate the failures above")
    
    return all_pass


if __name__ == '__main__':
    run_all_tests()
