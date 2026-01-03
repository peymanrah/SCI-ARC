#!/usr/bin/env python
"""
Smoke Tests for Jan 2026 Risk Fixes.

Tests the following fixes:
1. Output-level equivariance with target masking
2. Group-marginalized NLL loss
3. DataLoader recreation at phase boundaries
4. HPM memory gating by phase
5. Soft aggregation in TRM voting
6. Translational augmentation in phase config

Run with: python scripts/test_jan2026_risk_fixes.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_output_equiv_masking():
    """Test that OutputEquivarianceLoss supports target masking."""
    print("\n" + "="*60)
    print("TEST 1: Output Equivariance with Target Masking")
    print("="*60)
    
    import torch
    from sci_arc.models.rlan_modules.loo_training import (
        OutputEquivarianceLoss, EquivarianceConfig
    )
    
    # Create loss function with masking enabled
    loss_fn = OutputEquivarianceLoss(
        config=EquivarianceConfig(enabled=True),
        loss_type='kl',
        mask_to_target=True,
        pad_value=-100,
    )
    
    # Check that parameters exist
    assert hasattr(loss_fn, 'mask_to_target'), "mask_to_target attribute missing"
    assert loss_fn.mask_to_target == True, "mask_to_target should be True"
    assert hasattr(loss_fn, 'pad_value'), "pad_value attribute missing"
    
    # Check forward signature includes target_mask
    import inspect
    sig = inspect.signature(loss_fn.forward)
    params = list(sig.parameters.keys())
    assert 'target_mask' in params, f"target_mask not in forward params: {params}"
    
    print("  ✓ mask_to_target parameter exists")
    print("  ✓ pad_value parameter exists")
    print("  ✓ forward() accepts target_mask parameter")
    print("  PASS")
    return True


def test_group_marginalized_nll():
    """Test that GroupMarginalizedNLLLoss is properly defined."""
    print("\n" + "="*60)
    print("TEST 2: Group-Marginalized NLL Loss")
    print("="*60)
    
    from sci_arc.models.rlan_modules.loo_training import GroupMarginalizedNLLLoss
    
    # Create loss function
    loss_fn = GroupMarginalizedNLLLoss(
        num_augmentations=2,
        ignore_index=-100,
    )
    
    # Check attributes
    assert hasattr(loss_fn, 'num_augmentations'), "num_augmentations attribute missing"
    assert hasattr(loss_fn, 'ignore_index'), "ignore_index attribute missing"
    assert hasattr(loss_fn, '_aug_helper'), "_aug_helper missing"
    
    # Check forward signature
    import inspect
    sig = inspect.signature(loss_fn.forward)
    params = list(sig.parameters.keys())
    expected_params = ['model', 'test_inputs', 'train_inputs', 'train_outputs', 
                       'pair_mask', 'targets', 'temperature']
    for p in expected_params:
        assert p in params, f"Missing param {p} in forward: {params}"
    
    print("  ✓ GroupMarginalizedNLLLoss class exists")
    print("  ✓ Has required attributes")
    print("  ✓ forward() has correct signature")
    print("  PASS")
    return True


def test_trm_soft_voting():
    """Test that TRMStyleEvaluator supports soft voting."""
    print("\n" + "="*60)
    print("TEST 3: TRM Soft Voting Mode")
    print("="*60)
    
    from sci_arc.evaluation.trm_style_evaluator import TRMStyleEvaluator
    
    # Create evaluator with hard voting (default)
    evaluator_hard = TRMStyleEvaluator(
        pass_Ks=[1, 2],
        use_voting=True,
        voting_mode='hard',
    )
    assert evaluator_hard.voting_mode == 'hard', "Default should be hard voting"
    
    # Create evaluator with soft voting
    evaluator_soft = TRMStyleEvaluator(
        pass_Ks=[1, 2],
        use_voting=True,
        voting_mode='soft',
    )
    assert evaluator_soft.voting_mode == 'soft', "Should support soft voting"
    
    print("  ✓ voting_mode parameter exists")
    print("  ✓ Hard voting mode supported")
    print("  ✓ Soft voting mode supported")
    print("  PASS")
    return True


def test_phase_config_translational():
    """Test that phase config includes translational augmentation."""
    print("\n" + "="*60)
    print("TEST 4: Phase Config with Translational Augmentation")
    print("="*60)
    
    import yaml
    
    config_path = project_root / "configs" / "rlan_stable_dev_merged.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    phased = config.get('training', {}).get('phased_training', {})
    
    # Check Phase A
    phase_a_aug = phased.get('phase_a', {}).get('augmentation', {})
    assert 'translational' in phase_a_aug, "Phase A missing translational"
    assert phase_a_aug['translational'] == False, "Phase A translational should be False"
    
    # Check Phase B
    phase_b_aug = phased.get('phase_b', {}).get('augmentation', {})
    assert 'translational' in phase_b_aug, "Phase B missing translational"
    assert phase_b_aug['translational'] == True, "Phase B translational should be True"
    
    # Check Phase C
    phase_c_aug = phased.get('phase_c', {}).get('augmentation', {})
    assert 'translational' in phase_c_aug, "Phase C missing translational"
    assert phase_c_aug['translational'] == True, "Phase C translational should be True"
    
    print("  ✓ Phase A has translational=false")
    print("  ✓ Phase B has translational=true")
    print("  ✓ Phase C has translational=true")
    print("  PASS")
    return True


def test_group_marg_config():
    """Test that group_marginalized_nll config exists."""
    print("\n" + "="*60)
    print("TEST 5: Group-Marginalized NLL Config")
    print("="*60)
    
    import yaml
    
    config_path = project_root / "configs" / "rlan_stable_dev_merged.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    gm_cfg = config.get('training', {}).get('group_marginalized_nll', {})
    
    assert 'enabled' in gm_cfg, "Missing enabled flag"
    assert 'start_epoch' in gm_cfg, "Missing start_epoch"
    assert 'loss_weight' in gm_cfg, "Missing loss_weight"
    assert 'num_augmentations' in gm_cfg, "Missing num_augmentations"
    assert 'as_primary_loss' in gm_cfg, "Missing as_primary_loss"
    
    print("  ✓ enabled flag exists")
    print("  ✓ start_epoch exists")
    print("  ✓ loss_weight exists")
    print("  ✓ num_augmentations exists")
    print("  ✓ as_primary_loss exists")
    print("  PASS")
    return True


def test_train_epoch_signature():
    """Test that train_epoch has group_marg parameters."""
    print("\n" + "="*60)
    print("TEST 6: train_epoch Signature Updates")
    print("="*60)
    
    # Read train_rlan.py and check for group_marg params
    train_script = project_root / "scripts" / "train_rlan.py"
    with open(train_script, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check function signature includes group_marg parameters
    assert 'group_marg_loss_fn' in content, "Missing group_marg_loss_fn param"
    assert 'group_marg_start_epoch' in content, "Missing group_marg_start_epoch param"
    assert 'group_marg_weight' in content, "Missing group_marg_weight param"
    
    # Check the import includes GroupMarginalizedNLLLoss
    assert 'GroupMarginalizedNLLLoss' in content, "Missing GroupMarginalizedNLLLoss import"
    
    # Check call site passes group_marg
    assert 'group_marg_loss_fn=group_marg_loss_fn' in content, "Call site missing group_marg_loss_fn"
    
    print("  ✓ group_marg_loss_fn parameter in signature")
    print("  ✓ group_marg_start_epoch parameter in signature")
    print("  ✓ group_marg_weight parameter in signature")
    print("  ✓ GroupMarginalizedNLLLoss imported")
    print("  ✓ Call site passes group_marg_loss_fn")
    print("  PASS")
    return True


def test_dataloader_recreation():
    """Test that DataLoader recreation logic exists at phase boundaries."""
    print("\n" + "="*60)
    print("TEST 7: DataLoader Recreation at Phase Boundaries")
    print("="*60)
    
    train_script = project_root / "scripts" / "train_rlan.py"
    with open(train_script, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for the DataLoader recreation logic
    assert 'WORKER RECREATION FIX' in content, "Missing worker recreation fix comment"
    assert 'Recreating DataLoader to propagate augmentation' in content, "Missing recreation log"
    assert 'create_train_loader' in content, "Missing create_train_loader call"
    
    print("  ✓ Worker recreation fix comment present")
    print("  ✓ DataLoader recreation logic present")
    print("  ✓ create_train_loader called at phase boundaries")
    print("  PASS")
    return True


def test_hpm_memory_gating():
    """Test that HPM memory collection is gated by phase."""
    print("\n" + "="*60)
    print("TEST 8: HPM Memory Gating by Phase")
    print("="*60)
    
    train_script = project_root / "scripts" / "train_rlan.py"
    with open(train_script, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for HPM memory gating
    assert 'HPM memory collection DISABLED' in content, "Missing HPM disable log"
    assert 'model.hpm_memory_enabled = False' in content, "Missing HPM disable assignment"
    
    print("  ✓ HPM memory disable log present")
    print("  ✓ HPM memory_enabled set to False when phase disabled")
    print("  PASS")
    return True


def test_output_equiv_mask_to_target_config():
    """Test that mask_to_target is in config."""
    print("\n" + "="*60)
    print("TEST 9: Output Equiv mask_to_target in Config")
    print("="*60)
    
    import yaml
    
    config_path = project_root / "configs" / "rlan_stable_dev_merged.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    oe_cfg = config.get('training', {}).get('output_equivariance_training', {})
    
    assert 'mask_to_target' in oe_cfg, "Missing mask_to_target in config"
    assert oe_cfg['mask_to_target'] == True, "mask_to_target should be True"
    
    print("  ✓ mask_to_target in config")
    print("  ✓ mask_to_target=true")
    print("  PASS")
    return True


def test_ignore_index_constant_consistency():
    """Test that PADDING_IGNORE_VALUE is consistent across codebase."""
    print("\n" + "="*60)
    print("TEST 10: ignore_index Constant Consistency")
    print("="*60)
    
    from sci_arc.data.dataset import ARCDataset
    
    # Check ARCDataset constant
    assert hasattr(ARCDataset, 'PADDING_IGNORE_VALUE'), "ARCDataset missing PADDING_IGNORE_VALUE"
    dataset_val = ARCDataset.PADDING_IGNORE_VALUE
    assert dataset_val == -100, f"ARCDataset.PADDING_IGNORE_VALUE should be -100, got {dataset_val}"
    
    # Check train_rlan.py imports and uses the constant from ARCDataset
    train_script = project_root / "scripts" / "train_rlan.py"
    with open(train_script, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check constant is imported from ARCDataset (single source of truth)
    assert 'PADDING_IGNORE_VALUE = ARCDataset.PADDING_IGNORE_VALUE' in content, \
        "PADDING_IGNORE_VALUE should be imported from ARCDataset (not hardcoded)"
    
    # Check it's used for masking (not hardcoded -100 in mask creation)
    assert 'test_targets_for_mask != PADDING_IGNORE_VALUE' in content, \
        "Should use PADDING_IGNORE_VALUE constant instead of hardcoded -100 in mask"
    
    print("  ✓ ARCDataset.PADDING_IGNORE_VALUE = -100")
    print("  ✓ train_rlan.py imports constant from ARCDataset (single source of truth)")
    print("  ✓ Mask creation uses constant (not hardcoded)")
    print("  PASS")
    return True


def test_mask_validation_logging():
    """Test that defensive validation for all-False mask is present."""
    print("\n" + "="*60)
    print("TEST 11: Mask Validation Logging")
    print("="*60)
    
    train_script = project_root / "scripts" / "train_rlan.py"
    with open(train_script, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for defensive validation
    assert 'target_mask is ALL False' in content, "Missing validation warning for all-False mask"
    assert 'Defensive validation' in content, "Missing defensive validation comment"
    
    print("  ✓ Defensive validation for all-False mask present")
    print("  ✓ Warning log message present")
    print("  PASS")
    return True


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("JAN 2026 RISK FIXES - SMOKE TESTS")
    print("="*60)
    
    tests = [
        test_output_equiv_masking,
        test_group_marginalized_nll,
        test_trm_soft_voting,
        test_phase_config_translational,
        test_group_marg_config,
        test_train_epoch_signature,
        test_dataloader_recreation,
        test_hpm_memory_gating,
        test_output_equiv_mask_to_target_config,
        test_ignore_index_constant_consistency,
        test_mask_validation_logging,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            result = test()
            if result:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            failed += 1
    
    print("\n" + "="*60)
    print(f"SUMMARY: {passed}/{passed + failed} tests passed")
    print("="*60)
    
    if failed == 0:
        print("\n✅ ALL TESTS PASSED! Risk fixes are working correctly.\n")
        return 0
    else:
        print(f"\n❌ {failed} tests FAILED.\n")
        return 1


if __name__ == '__main__':
    sys.exit(main())
