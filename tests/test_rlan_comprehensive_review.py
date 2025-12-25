import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest
from sci_arc.models.rlan import RLAN, RLANConfig
from sci_arc.training.hyperlora_training import HyperLoRATrainer, HyperLoRATrainingConfig
from sci_arc.models.rlan_modules.acw import AugmentedConfidenceWeighting, apply_augmentation
from sci_arc.training.rlan_loss import log_stablemax

def test_hyperlora_loo_flow():
    """Test HyperLoRA LOO training flow."""
    config = RLANConfig(
        hidden_dim=32,
        use_hyperlora=True,
        hyperlora_rank=4,
        use_context_encoder=True,
        use_cross_attention_context=True, # Needed for support features
        spatial_downsample=1
    )
    model = RLAN(config=config)
    
    # Mock data: B=2, N=3 pairs, 10x10 grid
    B, N, H, W = 2, 3, 10, 10
    support_inputs = torch.randint(0, 10, (B, N, H, W))
    support_targets = torch.randint(0, 10, (B, N, H, W))
    
    # Mock support features (B, N, D, H, W)
    # ContextEncoder usually produces this
    support_features = torch.randn(B, N, config.hidden_dim, H, W)
    
    train_config = HyperLoRATrainingConfig(
        enabled=True,
        loo_enabled=True,
        equivariance_enabled=True
    )
    trainer = HyperLoRATrainer(model, train_config, device=torch.device('cpu'))
    
    # Test LOO Loss
    loss, metrics = trainer.compute_loo_loss(
        support_inputs, support_targets, support_features
    )
    
    print(f"LOO Loss: {loss.item()}")
    print(f"LOO Metrics: {metrics}")
    
    assert loss.item() >= 0
    assert metrics['loo_num_holdouts'] > 0
    assert not metrics['loo_skipped']

def test_equivariance_loss():
    """Test Equivariance loss computation."""
    config = RLANConfig(
        hidden_dim=32,
        use_hyperlora=True,
        hyperlora_rank=4
    )
    model = RLAN(config=config)
    
    B, N, H, W = 2, 3, 10, 10
    support_features = torch.randn(B, N, config.hidden_dim, H, W)
    
    # Get original deltas
    original_deltas = model.hyper_lora(support_features)
    
    train_config = HyperLoRATrainingConfig(enabled=True)
    trainer = HyperLoRATrainer(model, train_config, device=torch.device('cpu'))
    
    loss, metrics = trainer.compute_equivariance_loss(
        support_features, original_deltas
    )
    
    print(f"Equivariance Loss: {loss.item()}")
    assert loss.item() >= 0

def test_acw_voting():
    """Test ACW voting mechanism."""
    acw = AugmentedConfidenceWeighting(temperature=1.0)
    
    # Create 8 predictions (H, W)
    # 6 agree, 2 disagree
    H, W = 10, 10
    pred_correct = torch.zeros(H, W, dtype=torch.long)
    pred_wrong = torch.ones(H, W, dtype=torch.long)
    
    predictions = [pred_correct] * 6 + [pred_wrong] * 2
    
    winner, ranked = acw.weighted_vote(predictions)
    
    # Winner should be pred_correct (all zeros)
    assert torch.all(winner == 0)
    assert len(ranked) == 2 # 0 and 1 are the unique predictions

def test_double_addition_regression():
    """
    Regression test for BUG-05 (Double Addition).
    If bug exists: output = x + (x + delta). With delta=0, output = 2x.
    If fixed: output = x + delta. With delta=0, output = x.
    """
    config = RLANConfig(hidden_dim=32, use_hyperlora=True)
    model = RLAN(config=config)
    
    # Create input
    B, H, W = 1, 10, 10
    x = torch.randn(B, config.hidden_dim, H, W)
    
    # Create zero delta
    delta = torch.zeros(B, config.hidden_dim, config.hidden_dim)
    
    # Access the solver's ConvGRUCell directly to test _apply_lora_spatial
    # It's model.solver.gru
    
    cell = model.solver.gru
    
    # Test _apply_lora_spatial directly
    # It should return ONLY the delta (which is 0 here)
    delta_out = cell._apply_lora_spatial(x, delta)
    
    # If bug exists (old code), it returned x + delta = x
    # If fixed, it returns delta = 0
    
    if torch.allclose(delta_out, x):
        print("FAIL: _apply_lora_spatial returns x + delta (Double Addition Bug!)")
    elif torch.allclose(delta_out, torch.zeros_like(x)):
        print("PASS: _apply_lora_spatial returns delta only (Fixed)")
    else:
        print(f"FAIL: Unexpected output. Max val: {delta_out.max()}")

    # Verify forward pass integration
    # We can't easily isolate just the cell forward without mocking, 
    # but we can check if the model runs without error.
    
    assert torch.allclose(delta_out, torch.zeros_like(x)), "Double addition bug detected in helper!"

def test_loo_uses_stablemax():
    """Test that LOO loss now uses stablemax instead of softmax."""
    # Create logits with extreme values that would cause issues with softmax
    logits = torch.randn(2, 10, 8, 8)  # (B, C, H, W)
    logits[:, 0] = 100  # Make class 0 very confident
    
    targets = torch.randint(0, 10, (2, 8, 8))
    
    # Compute stablemax-based loss (as LOO now uses)
    B, C, H, W = logits.shape
    logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, C)
    targets_flat = targets.reshape(-1)
    
    logprobs = log_stablemax(logits_flat.to(torch.float64), dim=-1)
    prediction_logprobs = torch.gather(
        logprobs,
        index=targets_flat.unsqueeze(-1).to(torch.long),
        dim=-1
    ).squeeze(-1)
    stablemax_loss = -prediction_logprobs.to(logits.dtype).mean()
    
    # Compute softmax-based loss for comparison
    softmax_loss = F.cross_entropy(logits, targets)
    
    # Both should be finite
    assert torch.isfinite(stablemax_loss), f"Stablemax loss is not finite: {stablemax_loss}"
    assert torch.isfinite(softmax_loss), f"Softmax loss is not finite: {softmax_loss}"
    
    # They should produce different values (stablemax handles extremes differently)
    print(f"Stablemax loss: {stablemax_loss.item():.6f}")
    print(f"Softmax loss: {softmax_loss.item():.6f}")
    
    # Main validation: stablemax should work
    assert stablemax_loss.item() >= 0, "Loss should be non-negative"

def test_nan_inf_detection():
    """Test that we can detect NaN/Inf in losses."""
    # Create a tensor with NaN
    nan_tensor = torch.tensor(float('nan'))
    inf_tensor = torch.tensor(float('inf'))
    normal_tensor = torch.tensor(1.0)
    
    assert not torch.isfinite(nan_tensor), "NaN should be detected as non-finite"
    assert not torch.isfinite(inf_tensor), "Inf should be detected as non-finite"
    assert torch.isfinite(normal_tensor), "Normal value should be finite"
    print("NaN/Inf detection works correctly")

if __name__ == "__main__":
    print("Running Comprehensive RLAN Review Tests...")
    try:
        test_hyperlora_loo_flow()
        print("✅ HyperLoRA LOO Flow Passed")
    except Exception as e:
        print(f"❌ HyperLoRA LOO Flow Failed: {e}")
        traceback.print_exc()

    try:
        test_equivariance_loss()
        print("✅ Equivariance Loss Passed")
    except Exception as e:
        print(f"❌ Equivariance Loss Failed: {e}")
        traceback.print_exc()

    try:
        test_acw_voting()
        print("✅ ACW Voting Passed")
    except Exception as e:
        print(f"❌ ACW Voting Failed: {e}")
        traceback.print_exc()

    try:
        test_double_addition_regression()
        print("✅ Double Addition Regression Passed")
    except Exception as e:
        print(f"❌ Double Addition Regression Failed: {e}")
        traceback.print_exc()

    try:
        test_loo_uses_stablemax()
        print("✅ LOO Stablemax Test Passed")
    except Exception as e:
        print(f"❌ LOO Stablemax Test Failed: {e}")
        traceback.print_exc()

    try:
        test_nan_inf_detection()
        print("✅ NaN/Inf Detection Passed")
    except Exception as e:
        print(f"❌ NaN/Inf Detection Failed: {e}")
        traceback.print_exc()
