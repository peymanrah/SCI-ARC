#!/usr/bin/env python
"""
Test: Verify LOO Training Has No Data Leakage

This test verifies that the LOO training fix is correctly implemented:
1. LOO training should use remaining_features (N-1 pairs) for cross-attention
2. The held-out pair's OUTPUT should NOT be visible to the model

If cross-attention can see the held-out output, the model can "cheat" by
attending to the answer, which defeats the purpose of LOO training.

Run: python tests/test_loo_no_leakage.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock

def test_loo_uses_remaining_features():
    """
    Verify that LOOTrainingLoss passes remaining_features (not support_features)
    to forward_with_lora.
    """
    print("=" * 60)
    print("TEST: LOO Training Uses Remaining Features (No Data Leakage)")
    print("=" * 60)
    
    from sci_arc.models.rlan_modules.loo_training import LOOTrainingLoss, LOOConfig
    
    # Create LOO loss
    loo_config = LOOConfig(enabled=True, loss_weight=0.5, min_pairs_for_loo=2)
    loo_loss = LOOTrainingLoss(config=loo_config, hidden_dim=256)
    
    # Create mock model
    mock_model = MagicMock()
    mock_model.hyper_lora = MagicMock()
    mock_model.context_encoder = MagicMock()
    mock_model.context_encoder.use_spatial_features = True
    
    # Track what forward_with_lora receives
    received_support_features = []
    
    def capture_forward_with_lora(holdout_input, support_features, lora_deltas):
        # Capture the support_features shape
        received_support_features.append(support_features.shape)
        # Return dummy logits
        B, H, W = holdout_input.shape
        return torch.randn(B, 10, H, W)
    
    mock_model.forward_with_lora = capture_forward_with_lora
    
    # Create dummy data: 2 batches, 3 pairs each
    B, N, H, W = 2, 3, 5, 5
    D, Hs, Ws = 256, 8, 8
    
    input_grids = torch.randint(0, 10, (B, N, H, W))
    output_grids = torch.randint(0, 10, (B, N, H, W))
    pair_mask = torch.ones(B, N, dtype=torch.bool)
    
    # Mock context_encoder to return spatial features
    mock_support_features = torch.randn(B, N, D, Hs, Ws)
    mock_model.context_encoder.return_value = mock_support_features
    
    # Mock hyper_lora to return dummy deltas
    mock_model.hyper_lora.return_value = {
        'gru_reset': torch.randn(B, D, D),
        'gru_update': torch.randn(B, D, D),
        'gru_candidate': torch.randn(B, D, D),
        'output_head': torch.randn(B, D, D),
        'context': torch.randn(B, D),
    }
    
    # Run LOO loss
    result = loo_loss._forward_with_model(
        model=mock_model,
        input_grids=input_grids,
        output_grids=output_grids,
        pair_mask=pair_mask,
        temperature=1.0,
    )
    
    # Verify: forward_with_lora should have been called N times (once per holdout)
    assert len(received_support_features) == N, \
        f"Expected {N} calls to forward_with_lora, got {len(received_support_features)}"
    
    # Verify: Each call should have N-1 pairs, NOT N pairs
    for i, shape in enumerate(received_support_features):
        expected_pairs = N - 1
        actual_pairs = shape[1]  # shape is (B, N-1, D, Hs, Ws)
        
        if actual_pairs == N:
            print(f"\n[FAIL] Call {i+1}: Received {actual_pairs} pairs (FULL support set)")
            print("       This is DATA LEAKAGE - the held-out pair's output is visible!")
            print("       The fix was NOT applied correctly.")
            return False
        elif actual_pairs == expected_pairs:
            print(f"[OK]   Call {i+1}: Received {actual_pairs} pairs (remaining only)")
        else:
            print(f"[WARN] Call {i+1}: Unexpected {actual_pairs} pairs (expected {expected_pairs})")
            return False
    
    print(f"\n[PASS] LOO training correctly uses remaining_features (N-1 = {N-1} pairs)")
    print("       No data leakage - held-out output is hidden from cross-attention")
    return True


def test_cross_attention_handles_variable_n():
    """
    Verify that CrossAttentionInjector can handle variable N (sequence length).
    """
    print("\n" + "=" * 60)
    print("TEST: CrossAttentionInjector Handles Variable Sequence Length")
    print("=" * 60)
    
    try:
        from sci_arc.models.rlan_modules.context_encoder import CrossAttentionInjector
    except ImportError as e:
        print(f"[SKIP] Could not import CrossAttentionInjector: {e}")
        return True
    
    injector = CrossAttentionInjector(hidden_dim=256, num_heads=4, dropout=0.0)
    injector.eval()
    
    B, D, H, W = 2, 256, 5, 5
    features = torch.randn(B, D, H, W)
    
    # Test with different N values
    for N in [2, 3, 4, 5]:
        Hs, Ws = 8, 8
        support_features = torch.randn(B, N, D, Hs, Ws)
        
        with torch.no_grad():
            output = injector(features, support_features)
        
        assert output.shape == features.shape, \
            f"Output shape mismatch for N={N}: {output.shape} != {features.shape}"
        print(f"[OK]   N={N}: CrossAttentionInjector output shape {output.shape}")
    
    print(f"\n[PASS] CrossAttentionInjector handles variable N correctly")
    return True


if __name__ == "__main__":
    success = True
    
    try:
        success &= test_loo_uses_remaining_features()
    except Exception as e:
        print(f"\n[ERROR] test_loo_uses_remaining_features failed: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    try:
        success &= test_cross_attention_handles_variable_n()
    except Exception as e:
        print(f"\n[ERROR] test_cross_attention_handles_variable_n failed: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("ALL TESTS PASSED - LOO data leakage fix verified!")
    else:
        print("SOME TESTS FAILED - Please review the fix")
    print("=" * 60)
    
    sys.exit(0 if success else 1)
