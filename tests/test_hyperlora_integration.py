#!/usr/bin/env python
"""
Smoke Test for HyperLoRA Integration.

This script tests that all HyperLoRA components work end-to-end:
1. HyperLoRA module creation
2. LoRA weight prediction from context
3. RecursiveSolver with dynamic weights
4. RLAN forward_with_lora method
5. LOO training loss computation
6. Equivariance loss computation
7. ACW voting
8. Health metrics

Usage:
    python tests/test_hyperlora_integration.py
    
    # Or with pytest:
    pytest tests/test_hyperlora_integration.py -v
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.nn.functional as F


def test_hyperlora_module():
    """Test HyperLoRA module creation and forward pass."""
    print("Testing HyperLoRA module...")
    
    from sci_arc.models.rlan_modules import HyperLoRA, HyperLoRAConfig
    
    config = HyperLoRAConfig(
        hidden_dim=128,
        rank=4,
        scaling=1.0,
        dropout=0.0,
        init_scale=0.01,
    )
    
    hyper_lora = HyperLoRA(config=config)
    
    # Create fake support features: (B, N, D, H, W)
    B, N, D, H, W = 2, 3, 128, 8, 8
    support_features = torch.randn(B, N, D, H, W)
    
    # Forward pass
    deltas = hyper_lora(support_features)
    
    # Check outputs
    assert 'gru_reset' in deltas, "Missing gru_reset delta"
    assert 'gru_update' in deltas, "Missing gru_update delta"
    assert 'gru_candidate' in deltas, "Missing gru_candidate delta"
    assert 'output_head' in deltas, "Missing output_head delta"
    assert 'context' in deltas, "Missing context"
    
    # All LoRA deltas are (B, hidden_dim, hidden_dim)
    assert deltas['gru_reset'].shape == (B, 128, 128), f"Wrong shape: {deltas['gru_reset'].shape}"
    assert deltas['output_head'].shape == (B, 128, 128), f"Wrong shape: {deltas['output_head'].shape}"
    
    # Test health metrics
    magnitudes = hyper_lora.get_lora_magnitude(deltas)
    assert 'lora_gru_reset_magnitude' in magnitudes
    
    diversity = hyper_lora.get_weight_diversity(deltas)
    assert isinstance(diversity, float)
    
    print("  ✓ HyperLoRA module works correctly")
    return True


def test_recursive_solver_with_lora():
    """Test RecursiveSolver accepts LoRA deltas."""
    print("Testing RecursiveSolver with LoRA...")
    
    from sci_arc.models.rlan_modules import RecursiveSolver
    
    solver = RecursiveSolver(
        hidden_dim=128,
        num_classes=10,
        num_steps=3,
        num_predicates=8,
        num_colors=10,
        dropout=0.0,
        use_act=False,
        use_lcr=False,
        use_sph=False,
    )
    
    # Create inputs
    B, K, D, H, W = 2, 5, 128, 8, 8
    clue_features = torch.randn(B, K, D, H, W)
    count_embedding = torch.zeros(B, 10, D)
    predicates = torch.zeros(B, 8)
    input_grid = torch.randint(0, 10, (B, H, W))
    attention_maps = torch.ones(B, K, H, W) / (H * W)
    stop_logits = torch.zeros(B, K)
    
    # Create LoRA deltas - all are (B, D, D)
    lora_deltas = {
        'gru_reset': torch.randn(B, D, D) * 0.01,
        'gru_update': torch.randn(B, D, D) * 0.01,
        'gru_candidate': torch.randn(B, D, D) * 0.01,
        'output_head': torch.randn(B, D, D) * 0.01,  # D x D, applied to hidden state
    }
    
    # Forward without LoRA
    logits_no_lora = solver(
        clue_features=clue_features,
        count_embedding=count_embedding,
        predicates=predicates,
        input_grid=input_grid,
        attention_maps=attention_maps,
        stop_logits=stop_logits,
        lora_deltas=None,
    )
    
    # Forward with LoRA
    logits_with_lora = solver(
        clue_features=clue_features,
        count_embedding=count_embedding,
        predicates=predicates,
        input_grid=input_grid,
        attention_maps=attention_maps,
        stop_logits=stop_logits,
        lora_deltas=lora_deltas,
    )
    
    # Outputs should be different (LoRA modulates weights)
    assert logits_no_lora.shape == (B, 10, H, W)
    assert logits_with_lora.shape == (B, 10, H, W)
    
    # They should be different due to LoRA
    diff = (logits_no_lora - logits_with_lora).abs().mean()
    assert diff > 0, "LoRA had no effect on outputs"
    
    print("  ✓ RecursiveSolver accepts LoRA deltas correctly")
    return True


def test_rlan_with_hyperlora():
    """Test RLAN model with HyperLoRA enabled."""
    print("Testing RLAN with HyperLoRA...")
    
    from sci_arc.models import RLAN, RLANConfig
    
    # Create config with HyperLoRA enabled
    config = RLANConfig(
        hidden_dim=64,  # Small for testing
        num_colors=10,
        num_classes=10,
        max_grid_size=10,
        num_solver_steps=2,
        use_dsc=False,  # Disable for simpler testing
        use_msre=False,
        use_lcr=False,
        use_sph=False,
        use_cross_attention_context=True,
        spatial_downsample=4,
        use_hyperlora=True,
        hyperlora_rank=4,
    )
    
    model = RLAN(config=config)
    
    # Check HyperLoRA was created
    assert hasattr(model, 'hyper_lora'), "Model should have hyper_lora"
    assert model.hyper_lora is not None, "hyper_lora should not be None"
    assert model.use_hyperlora, "use_hyperlora should be True"
    
    # Create dummy inputs
    B, N, H, W = 2, 3, 8, 8
    test_input = torch.randint(0, 10, (B, H, W))
    train_inputs = torch.randint(0, 10, (B, N, H, W))
    train_outputs = torch.randint(0, 10, (B, N, H, W))
    
    # Normal forward pass
    logits = model(test_input, train_inputs, train_outputs)
    assert logits.shape == (B, 10, H, W), f"Wrong logits shape: {logits.shape}"
    
    print("  ✓ RLAN with HyperLoRA works correctly")
    return True


def test_hyperlora_trainer():
    """Test HyperLoRA training helper."""
    print("Testing HyperLoRATrainer...")
    
    from sci_arc.models import RLAN, RLANConfig
    from sci_arc.training import HyperLoRATrainer, HyperLoRATrainingConfig
    
    # Create model with HyperLoRA
    config = RLANConfig(
        hidden_dim=64,
        num_colors=10,
        num_classes=10,
        max_grid_size=10,
        num_solver_steps=2,
        use_dsc=False,
        use_msre=False,
        use_lcr=False,
        use_sph=False,
        use_cross_attention_context=True,
        spatial_downsample=4,
        use_hyperlora=True,
        hyperlora_rank=4,
    )
    model = RLAN(config=config)
    
    # Create trainer
    trainer_config = HyperLoRATrainingConfig(
        enabled=True,
        loo_enabled=True,
        loo_loss_weight=0.5,
        min_pairs_for_loo=2,
        equivariance_enabled=True,
        equivariance_loss_weight=0.1,
    )
    trainer = HyperLoRATrainer(model, trainer_config, device=torch.device('cpu'))
    
    assert trainer.enabled, "Trainer should be enabled"
    
    # Create dummy inputs
    B, N, H, W = 2, 3, 8, 8
    D, Hs, Ws = 64, 4, 4
    support_inputs = torch.randint(0, 10, (B, N, H, W))
    support_targets = torch.randint(0, 10, (B, N, H, W))
    support_features = torch.randn(B, N, D, Hs, Ws)
    
    # Compute losses
    losses = trainer.compute_losses(
        support_inputs, support_targets, support_features
    )
    
    assert 'total' in losses, "Missing total loss"
    assert 'loo' in losses, "Missing LOO loss"
    assert 'equivariance' in losses, "Missing equivariance loss"
    assert 'metrics' in losses, "Missing metrics"
    
    # Check loss values are reasonable
    assert losses['total'].item() >= 0, "Total loss should be non-negative"
    assert 'hyperlora_enabled' in losses['metrics']
    assert losses['metrics']['hyperlora_enabled'] == True
    
    print("  ✓ HyperLoRATrainer works correctly")
    return True


def test_acw_voting():
    """Test Augmented Confidence Weighting voting."""
    print("Testing ACW voting...")
    
    from sci_arc.models.rlan_modules import AugmentedConfidenceWeighting
    
    acw = AugmentedConfidenceWeighting(temperature=1.0)
    
    # Create predictions - some agree, some don't
    pred1 = torch.tensor([[0, 1], [2, 3]])
    pred2 = torch.tensor([[0, 1], [2, 3]])  # Same as pred1
    pred3 = torch.tensor([[0, 1], [2, 3]])  # Same as pred1
    pred4 = torch.tensor([[4, 5], [6, 7]])  # Different
    
    predictions = [pred1, pred2, pred3, pred4]
    
    # Compute consistency
    consistency = acw.compute_consistency_scores(predictions)
    
    # pred1, pred2, pred3 should have higher consistency (agree with 2/3 others)
    # pred4 should have lower consistency (agrees with 0/3 others)
    assert consistency[0] > consistency[3], "Consistent predictions should have higher scores"
    assert consistency[1] > consistency[3], "Consistent predictions should have higher scores"
    assert consistency[2] > consistency[3], "Consistent predictions should have higher scores"
    
    # Weighted vote
    winner, candidates = acw.weighted_vote(predictions)
    
    assert torch.equal(winner, pred1), "Winner should be the majority prediction"
    assert len(candidates) == 2, f"Should have 2 unique predictions, got {len(candidates)}"
    
    print("  ✓ ACW voting works correctly")
    return True


def test_health_metrics():
    """Test health metrics computation."""
    print("Testing health metrics...")
    
    from sci_arc.models.rlan_modules import (
        HyperLoRA, HyperLoRAConfig, compute_hyperlora_health_metrics
    )
    
    config = HyperLoRAConfig(hidden_dim=64, rank=4)
    hyper_lora = HyperLoRA(config=config)
    
    # Create support features and get deltas
    support_features = torch.randn(2, 3, 64, 4, 4)
    deltas = hyper_lora(support_features)
    
    # Compute health metrics
    metrics = compute_hyperlora_health_metrics(hyper_lora, deltas, loo_accuracy=0.85)
    
    # Check required metrics exist
    assert 'lora_total_magnitude' in metrics
    assert 'lora_weight_diversity' in metrics
    assert 'loo_accuracy' in metrics
    assert metrics['loo_accuracy'] == 0.85
    
    # Check health flags
    assert 'lora_health_ok' in metrics
    assert 'lora_health_collapsed' in metrics
    assert 'lora_health_exploding' in metrics
    
    print("  ✓ Health metrics work correctly")
    return True


def test_backward_compatibility():
    """Test that models without HyperLoRA still work."""
    print("Testing backward compatibility...")
    
    from sci_arc.models import RLAN, RLANConfig
    
    # Create config WITHOUT HyperLoRA
    config = RLANConfig(
        hidden_dim=64,
        num_colors=10,
        num_classes=10,
        max_grid_size=10,
        num_solver_steps=2,
        use_hyperlora=False,  # Disabled
    )
    
    model = RLAN(config=config)
    
    # Check HyperLoRA is None
    assert model.hyper_lora is None, "hyper_lora should be None when disabled"
    assert not model.use_hyperlora, "use_hyperlora should be False"
    
    # Normal forward should still work
    B, N, H, W = 2, 3, 8, 8
    test_input = torch.randint(0, 10, (B, H, W))
    train_inputs = torch.randint(0, 10, (B, N, H, W))
    train_outputs = torch.randint(0, 10, (B, N, H, W))
    
    logits = model(test_input, train_inputs, train_outputs)
    assert logits.shape == (B, 10, H, W)
    
    print("  ✓ Backward compatibility maintained")
    return True


def run_all_tests():
    """Run all smoke tests."""
    print("\n" + "=" * 60)
    print("HyperLoRA Integration Smoke Tests")
    print("=" * 60 + "\n")
    
    tests = [
        ("HyperLoRA Module", test_hyperlora_module),
        ("RecursiveSolver with LoRA", test_recursive_solver_with_lora),
        ("RLAN with HyperLoRA", test_rlan_with_hyperlora),
        ("HyperLoRA Trainer", test_hyperlora_trainer),
        ("ACW Voting", test_acw_voting),
        ("Health Metrics", test_health_metrics),
        ("Backward Compatibility", test_backward_compatibility),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        try:
            result = test_fn()
            if result:
                passed += 1
            else:
                failed += 1
                print(f"  ✗ {name} failed (returned False)")
        except Exception as e:
            failed += 1
            print(f"  ✗ {name} failed with exception: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
