"""
Smoke Test for CISL (Content-Invariant Structure Learning)

Tests both new (CISL) and old (CICL) names for backward compatibility:
1. CISLLoss/CICLLoss module initialization
2. Loss computation with various inputs
3. Content/Color permutation function
4. Integration with TrainingConfig
5. Gradient flow
6. Backward compatibility aliases

Run with: python -m pytest tests/test_cicl.py -v
Or:       python tests/test_cicl.py
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_cicl_loss_initialization():
    """Test CISL loss module can be initialized (tests both old and new names)."""
    from sci_arc.training.cisl_loss import CISLLoss
    from sci_arc.training.cicl_loss import CICLLoss
    
    # Test new name
    loss_fn = CISLLoss(
        consist_weight=0.5,
        content_inv_weight=0.5,
        variance_weight=0.1,
        target_std=0.5
    )
    
    assert loss_fn is not None
    assert loss_fn.consist_weight == 0.5
    assert loss_fn.content_inv_weight == 0.5
    assert loss_fn.variance_weight == 0.1
    
    # Test backward compat: CICLLoss should be same as CISLLoss
    assert CICLLoss is CISLLoss
    print("✓ CISLLoss initialization (CICLLoss is alias)")


def test_consistency_loss():
    """Test within-task consistency loss."""
    from sci_arc.training.cicl_loss import WithinTaskConsistencyLoss
    
    loss_fn = WithinTaskConsistencyLoss(normalize=True)
    
    # Single task with K demos
    K, D = 4, 64
    z = torch.randn(K, D)
    loss = loss_fn(z)
    assert loss.shape == ()  # Scalar
    assert loss >= 0
    print(f"✓ Consistency loss (single): {loss.item():.4f}")
    
    # Batch of tasks
    B = 8
    z_batch = torch.randn(B, K, D)
    loss_batch = loss_fn(z_batch)
    assert loss_batch.shape == ()
    assert loss_batch >= 0
    print(f"✓ Consistency loss (batch): {loss_batch.item():.4f}")
    
    # Perfect consistency (all same) should give ~0 loss
    z_same = torch.ones(K, D)
    loss_same = loss_fn(z_same)
    assert loss_same < 1e-5
    print(f"✓ Consistency loss (all same): {loss_same.item():.6f} (should be ~0)")


def test_color_invariance_loss():
    """Test content invariance loss (tests both old and new names)."""
    from sci_arc.training.cisl_loss import ContentInvarianceLoss
    from sci_arc.training.cicl_loss import ColorInvarianceLoss
    
    # Test backward compat
    assert ColorInvarianceLoss is ContentInvarianceLoss
    
    loss_fn = ContentInvarianceLoss(normalize=True)
    
    B, D = 8, 64
    z_orig = torch.randn(B, D)
    z_perm = torch.randn(B, D)
    
    loss = loss_fn(z_orig, z_perm)
    assert loss.shape == ()
    assert loss >= 0
    print(f"✓ Color invariance loss (different): {loss.item():.4f}")
    
    # Same embeddings should give 0 loss
    loss_same = loss_fn(z_orig, z_orig.clone())
    assert loss_same < 1e-5
    print(f"✓ Color invariance loss (identical): {loss_same.item():.6f} (should be ~0)")


def test_variance_loss():
    """Test batch variance loss (anti-collapse)."""
    from sci_arc.training.cicl_loss import BatchVarianceLoss
    
    loss_fn = BatchVarianceLoss(target_std=0.5)
    
    B, D = 16, 64
    
    # Normal random embeddings should have std > 0.5
    z_normal = torch.randn(B, D)
    loss_normal = loss_fn(z_normal)
    print(f"✓ Variance loss (random): {loss_normal.item():.4f}")
    
    # Collapsed embeddings (all zeros) should have high loss
    z_collapsed = torch.zeros(B, D)
    loss_collapsed = loss_fn(z_collapsed)
    assert loss_collapsed > 0.4  # Should be close to target_std
    print(f"✓ Variance loss (collapsed): {loss_collapsed.item():.4f} (should be ~0.5)")
    
    # High variance embeddings should have 0 loss
    z_diverse = torch.randn(B, D) * 2.0  # Higher std
    loss_diverse = loss_fn(z_diverse)
    assert loss_diverse < 0.1
    print(f"✓ Variance loss (diverse): {loss_diverse.item():.4f} (should be ~0)")


def test_cicl_full_loss():
    """Test complete CISL loss computation."""
    from sci_arc.training.cisl_loss import CISLLoss
    
    loss_fn = CISLLoss(
        consist_weight=0.5,
        content_inv_weight=0.5,
        variance_weight=0.1
    )
    
    B, K, D = 8, 4, 64
    z_struct = torch.randn(B, K, D)
    z_struct_color = torch.randn(B, K, D)
    
    result = loss_fn(z_struct, z_struct_color)
    
    assert 'total' in result
    assert 'consistency' in result
    assert 'content_inv' in result  # Was 'color_inv'
    assert 'variance' in result
    
    print(f"\n✓ CISL Full Loss:")
    print(f"  Total: {result['total'].item():.4f}")
    print(f"  Consistency: {result['consistency'].item():.4f}")
    print(f"  Content Inv: {result['content_inv'].item():.4f}")
    print(f"  Variance: {result['variance'].item():.4f}")


def test_color_permutation():
    """Test content permutation function (old name: color permutation)."""
    from sci_arc.training.cisl_loss import apply_content_permutation_batch
    from sci_arc.training.cicl_loss import apply_color_permutation_batch
    
    # Test backward compat
    assert apply_color_permutation_batch is apply_content_permutation_batch
    
    B, K, H, W = 4, 3, 5, 5
    
    # Create grids with known colors
    input_grids = torch.randint(0, 10, (B, K, H, W))
    output_grids = torch.randint(0, 10, (B, K, H, W))
    test_inputs = torch.randint(0, 10, (B, H, W))
    test_outputs = torch.randint(0, 10, (B, H, W))
    
    # Apply permutation using new name
    inp_perm, out_perm, test_in_perm, test_out_perm = apply_content_permutation_batch(
        input_grids, output_grids, test_inputs, test_outputs
    )
    
    # Check shapes are preserved
    assert inp_perm.shape == input_grids.shape
    assert out_perm.shape == output_grids.shape
    assert test_in_perm.shape == test_inputs.shape
    assert test_out_perm.shape == test_outputs.shape
    
    # Check that 0 (background) is preserved
    bg_mask = input_grids == 0
    assert (inp_perm[bg_mask] == 0).all()
    
    # Check that non-zero colors changed (with high probability)
    non_bg_mask = input_grids != 0
    if non_bg_mask.sum() > 0:
        # At least some colors should be different
        num_changed = (inp_perm[non_bg_mask] != input_grids[non_bg_mask]).sum().item()
        print(f"✓ Color permutation: {num_changed}/{non_bg_mask.sum().item()} colors changed")
    else:
        print("✓ Color permutation: No non-background colors to permute")


def test_gradient_flow():
    """Test that gradients flow through CISL loss."""
    from sci_arc.training.cisl_loss import CISLLoss
    
    loss_fn = CISLLoss()
    
    B, K, D = 8, 4, 64
    z = torch.randn(B, K, D, requires_grad=True)
    z_color = torch.randn(B, K, D, requires_grad=True)
    
    result = loss_fn(z, z_color)
    loss = result['total']
    loss.backward()
    
    assert z.grad is not None
    assert z_color.grad is not None
    assert z.grad.norm() > 0
    print(f"✓ Gradient flow: z.grad norm = {z.grad.norm().item():.4f}")


def test_training_config_cicl():
    """Test that TrainingConfig includes CISL parameters (cicl_ prefix for backward compat)."""
    from sci_arc.training.trainer import TrainingConfig
    
    config = TrainingConfig()
    
    # Check CISL parameters exist (use cicl_ prefix for backward compat)
    assert hasattr(config, 'use_cicl')
    assert hasattr(config, 'cicl_consist_weight')
    assert hasattr(config, 'cicl_color_inv_weight')  # Actually content_inv
    assert hasattr(config, 'cicl_variance_weight')
    assert hasattr(config, 'cicl_target_std')
    
    print(f"\n✓ TrainingConfig CISL params (cicl_ prefix for backward compat):")
    print(f"  use_cicl: {config.use_cicl}")
    print(f"  cicl_consist_weight: {config.cicl_consist_weight}")
    print(f"  cicl_color_inv_weight: {config.cicl_color_inv_weight}")
    print(f"  cicl_variance_weight: {config.cicl_variance_weight}")
    print(f"  cicl_target_std: {config.cicl_target_std}")


def test_import_from_init():
    """Test that both CISL and CICL components can be imported from training module."""
    # Test new names (CISL)
    from sci_arc.training import (
        CISLLoss,
        ContentInvarianceLoss,
        apply_content_permutation_batch,
    )
    
    # Test old names (CICL) for backward compat
    from sci_arc.training import (
        CICLLoss,
        WithinTaskConsistencyLoss,
        ColorInvarianceLoss,
        BatchVarianceLoss,
        apply_color_permutation_batch,
    )
    
    # Verify aliases
    assert CISLLoss is CICLLoss
    assert ContentInvarianceLoss is ColorInvarianceLoss
    assert apply_content_permutation_batch is apply_color_permutation_batch
    
    assert WithinTaskConsistencyLoss is not None
    assert BatchVarianceLoss is not None
    print("✓ All CISL/CICL components importable from sci_arc.training")


if __name__ == "__main__":
    print("=" * 60)
    print("CISL Smoke Tests (with CICL backward compatibility)")
    print("=" * 60)
    
    try:
        test_import_from_init()
        test_cicl_loss_initialization()
        test_consistency_loss()
        test_color_invariance_loss()
        test_variance_loss()
        test_cicl_full_loss()
        test_color_permutation()
        test_gradient_flow()
        test_training_config_cicl()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
