#!/usr/bin/env python
"""
Quick verification script to ensure all fixes are working correctly.

Run with: python scripts/verify_fixes.py
"""

import sys
import numpy as np
import torch
sys.path.insert(0, '.')


def test_dsc_no_gumbel():
    """Verify DSC no longer uses Gumbel noise."""
    from sci_arc.models.rlan_modules.dynamic_saliency_controller import (
        DynamicSaliencyController, gumbel_softmax_2d
    )
    
    print("=" * 60)
    print("Test 1: DSC Gumbel Noise Removal")
    print("=" * 60)
    
    # Test that gumbel_softmax_2d is now deterministic even with deterministic=False
    torch.manual_seed(42)
    logits = torch.randn(4, 10, 10)
    
    out1 = gumbel_softmax_2d(logits, temperature=0.5, deterministic=False)
    out2 = gumbel_softmax_2d(logits, temperature=0.5, deterministic=False)
    
    is_deterministic = torch.allclose(out1, out2)
    print(f"  gumbel_softmax_2d is now deterministic: {is_deterministic}")
    
    # Train vs eval should be identical
    out_train = gumbel_softmax_2d(logits, temperature=0.5, deterministic=False)
    out_eval = gumbel_softmax_2d(logits, temperature=0.5, deterministic=True)
    train_eval_same = torch.allclose(out_train, out_eval)
    print(f"  Train and Eval produce same output: {train_eval_same}")
    
    # DSC produces finite outputs
    dsc = DynamicSaliencyController(hidden_dim=128, max_clues=4)
    features = torch.randn(2, 128, 10, 10)
    centroids, attn_maps, stop_logits = dsc(features, temperature=0.5)
    
    all_finite = (
        torch.isfinite(centroids).all() and 
        torch.isfinite(attn_maps).all() and 
        torch.isfinite(stop_logits).all()
    )
    print(f"  DSC outputs all finite: {all_finite}")
    
    passed = is_deterministic and train_eval_same and all_finite
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    return passed


def test_trm_evaluator():
    """Verify TRM-style evaluator with inverse augmentation."""
    from sci_arc.evaluation.trm_style_evaluator import (
        TRMStyleEvaluator, dihedral_transform, inverse_dihedral_transform,
        inverse_color_permutation
    )
    
    print("\n" + "=" * 60)
    print("Test 2: TRM-Style Evaluator")
    print("=" * 60)
    
    # Test all 8 dihedral transforms invert correctly
    arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    all_inverts = True
    for tid in range(8):
        transformed = dihedral_transform(arr, tid)
        recovered = inverse_dihedral_transform(transformed, tid)
        if not np.array_equal(arr, recovered):
            print(f"  FAILED: dihedral transform {tid} does not invert correctly")
            all_inverts = False
            break
    print(f"  All 8 dihedral transforms invert: {all_inverts}")
    
    # Test color permutation inversion
    arr = np.array([[0, 1, 2], [3, 4, 5]])
    color_perm = np.array([0, 3, 1, 2, 4, 5, 6, 7, 8, 9])  # Permute colors 1,2,3
    # After applying perm: 0->0, 1->3, 2->1, 3->2, etc.
    # So arr after perm should be [[0,3,1], [2,4,5]]
    # Inverse should restore original
    arr_with_perm = color_perm[arr]  # Simulate augmented prediction
    recovered = inverse_color_permutation(arr_with_perm, color_perm)
    color_inv_works = np.array_equal(arr, recovered)
    print(f"  Color permutation inversion works: {color_inv_works}")
    
    # Test evaluator with perfect prediction
    evaluator = TRMStyleEvaluator(pass_Ks=[1, 2])
    evaluator.update(
        task_id='test1',
        prediction=np.array([[1, 2], [3, 4]]),
        ground_truth=np.array([[1, 2], [3, 4]]),
        aug_info={'dihedral_id': 0, 'color_perm': None},
        confidence=0.9
    )
    metrics = evaluator.compute_metrics()
    pass_at_1 = metrics.get('pass@1', 0)
    print(f"  Pass@1 for perfect match: {pass_at_1}")
    
    # Test with augmented prediction that needs inverse
    evaluator2 = TRMStyleEvaluator(pass_Ks=[1])
    # Ground truth is [[1,2], [3,4]]
    # Prediction is rotated 90 CCW: [[2,4], [1,3]]
    # After inverse (rotate 90 CW = 270 CCW), should match GT
    gt = np.array([[1, 2], [3, 4]])
    pred_rotated = np.rot90(gt, k=1)  # 90 CCW
    evaluator2.update(
        task_id='test2',
        prediction=pred_rotated,
        ground_truth=gt,
        aug_info={'dihedral_id': 1, 'color_perm': None},  # 1 = 90 CCW
        confidence=0.9
    )
    metrics2 = evaluator2.compute_metrics()
    pass_at_1_aug = metrics2.get('pass@1', 0)
    print(f"  Pass@1 with inverse aug: {pass_at_1_aug}")
    
    passed = all_inverts and color_inv_works and pass_at_1 == 1.0 and pass_at_1_aug == 1.0
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    return passed


def test_configs():
    """Verify configs have correct settings."""
    import yaml
    
    print("\n" + "=" * 60)
    print("Test 3: Configuration Files")
    print("=" * 60)
    
    configs_ok = True
    
    for config_name in ['rlan_stable.yaml', 'rlan_fixed.yaml']:
        try:
            with open(f'configs/{config_name}', 'r') as f:
                config = yaml.safe_load(f)
            
            # Check EMA is disabled
            use_ema = config.get('training', {}).get('use_ema', True)
            if use_ema:
                print(f"  {config_name}: use_ema should be false but is {use_ema}")
                configs_ok = False
            else:
                print(f"  {config_name}: use_ema=false ✓")
            
            # Check eval_every is reasonable (>= 1, was strictly 1 before Dec 2025)
            # Now accepts eval_every >= 1 for flexibility (10 recommended for expensive TTA)
            eval_every = config.get('logging', {}).get('eval_every', 5)
            if eval_every < 1:
                print(f"  {config_name}: eval_every should be >= 1 but is {eval_every}")
                configs_ok = False
            else:
                print(f"  {config_name}: eval_every={eval_every} ✓")
                
        except Exception as e:
            print(f"  {config_name}: ERROR - {e}")
            configs_ok = False
    
    print(f"  Result: {'PASS' if configs_ok else 'FAIL'}")
    return configs_ok


def test_training_imports():
    """Verify training script can import all required modules."""
    print("\n" + "=" * 60)
    print("Test 4: Training Script Imports")
    print("=" * 60)
    
    try:
        from sci_arc.models import RLAN, RLANConfig
        from sci_arc.training import RLANLoss
        from sci_arc.training.ema import EMAHelper
        from sci_arc.data import ARCDataset, collate_sci_arc
        from sci_arc.evaluation.trm_style_evaluator import TRMStyleEvaluator
        from sci_arc.utils.gap_monitor import GapHealthMonitor
        
        print("  All imports successful ✓")
        
        # Quick instantiation test
        evaluator = TRMStyleEvaluator()
        monitor = GapHealthMonitor()
        print("  TRMStyleEvaluator instantiated ✓")
        print("  GapHealthMonitor instantiated ✓")
        
        print("  Result: PASS")
        return True
    except Exception as e:
        print(f"  Import error: {e}")
        print("  Result: FAIL")
        return False


def main():
    print("\n" + "=" * 60)
    print("RLAN FIX VERIFICATION")
    print("=" * 60)
    
    results = []
    
    results.append(("DSC No Gumbel", test_dsc_no_gumbel()))
    results.append(("TRM Evaluator", test_trm_evaluator()))
    results.append(("Configs", test_configs()))
    results.append(("Training Imports", test_training_imports()))
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED! RLAN is ready for production training.")
        print("\nKey fixes verified:")
        print("  1. Gumbel noise removed from DSC (train=eval behavior)")
        print("  2. TRM-style evaluator with inverse augmentation")
        print("  3. EMA disabled for 20-epoch training")
        print("  4. eval_every configured (10 recommended for expensive TTA)")
    else:
        print("SOME TESTS FAILED! Review before production training.")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
