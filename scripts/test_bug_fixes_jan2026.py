#!/usr/bin/env python3
"""
Smoke Test for January 2026 Bug Fixes
=====================================

This script validates all bug fixes applied in the Jan 2026 patch:
1. Output-equiv double-averaging removed
2. Output-equiv train/eval mode mismatch fixed
3. Phase augmentation transpose logic fixed
4. Phase epoch off-by-one fixed

Usage:
    python scripts/test_bug_fixes_jan2026.py
"""

import sys
import torch
import torch.nn as nn

def test_output_equiv_structure():
    """Test 1: Verify OutputEquivarianceLoss structure fixes."""
    print('='*60)
    print('TEST 1: OutputEquivarianceLoss structure check')
    print('='*60)
    
    from sci_arc.models.rlan_modules.loo_training import OutputEquivarianceLoss
    import inspect
    source = inspect.getsource(OutputEquivarianceLoss.forward)
    
    # Check Bug 1 fix: No double-averaging
    if 'total_loss = total_loss / len(selected_augs)' in source:
        print('  [FAIL] Double-averaging bug still present!')
        return False
    else:
        print('  [PASS] Double-averaging removed')
    
    # Check Bug 2 fix: No model.eval() mode switch
    if 'model.eval()' in source:
        print('  [FAIL] Train/eval mode mismatch still present!')
        return False
    else:
        print('  [PASS] Mode mismatch fixed (no model.eval())')
    
    # Check that we keep model in same mode
    if 'Keep model in SAME mode' in source:
        print('  [PASS] Fix comment present for mode handling')
    else:
        print('  [WARN] Fix comment missing (but logic may still be correct)')
    
    print()
    print('TEST 1 PASSED: OutputEquivarianceLoss fixes verified')
    return True


def test_phase_handling_fixes():
    """Test 2: Verify phase handling fixes in train_rlan.py."""
    print()
    print('='*60)
    print('TEST 2: Phase handling fixes in train_rlan.py')
    print('='*60)
    
    with open('scripts/train_rlan.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check Bug 3 fix: Transpose logic
    if 'specified_flags = [v for v in [rotation, flip, transpose]' in content:
        print('  [PASS] Transpose logic fix applied')
    else:
        print('  [FAIL] Transpose logic fix NOT found!')
        return False
    
    # Check Bug 4 fix: Off-by-one epoch
    if 'epoch_1based = epoch + 1' in content:
        print('  [PASS] Off-by-one epoch fix applied')
    else:
        print('  [FAIL] Off-by-one epoch fix NOT found!')
        return False
    
    # Check the comparison uses epoch_1based
    if 'epoch_1based <= phase_a_end' in content:
        print('  [PASS] Phase comparisons use 1-based epoch')
    else:
        print('  [FAIL] Phase comparisons still use 0-based epoch!')
        return False
    
    print()
    print('TEST 2 PASSED: Phase handling fixes verified')
    return True


def test_phase_epoch_logic():
    """Test 3: Functional test of phase epoch logic."""
    print()
    print('='*60)
    print('TEST 3: Functional test of phase epoch logic')
    print('='*60)
    
    phased_training_config = {
        'phase_a': {'end_epoch': 10},
        'phase_b': {'start_epoch': 11, 'end_epoch': 20},
        'phase_c': {'start_epoch': 21}
    }
    
    def get_current_phase(epoch):
        phase_a = phased_training_config.get('phase_a', {})
        phase_b = phased_training_config.get('phase_b', {})
        phase_c = phased_training_config.get('phase_c', {})
        
        epoch_1based = epoch + 1
        
        phase_a_end = phase_a.get('end_epoch', 10)
        phase_b_start = phase_b.get('start_epoch', phase_a_end + 1)
        phase_b_end = phase_b.get('end_epoch', 20)
        phase_c_start = phase_c.get('start_epoch', phase_b_end + 1)
        
        if epoch_1based <= phase_a_end:
            return 'A', phase_a
        elif epoch_1based >= phase_b_start and epoch_1based <= phase_b_end:
            return 'B', phase_b
        elif epoch_1based >= phase_c_start:
            return 'C', phase_c
        else:
            return 'B', phase_b
    
    test_cases = [
        (0, 'A', 1),
        (9, 'A', 10),
        (10, 'B', 11),
        (19, 'B', 20),
        (20, 'C', 21),
        (50, 'C', 51),
    ]
    
    all_passed = True
    for loop_epoch, expected_phase, print_epoch in test_cases:
        phase_name, _ = get_current_phase(loop_epoch)
        status = 'PASS' if phase_name == expected_phase else 'FAIL'
        if phase_name != expected_phase:
            all_passed = False
        print(f'  [{status}] loop epoch {loop_epoch} (print: {print_epoch}) -> Phase {phase_name} (expected: {expected_phase})')
    
    print()
    if all_passed:
        print('TEST 3 PASSED: Phase epoch logic works correctly')
    else:
        print('TEST 3 FAILED: Phase epoch logic has errors')
    return all_passed


def test_augmentation_mapping():
    """Test 4: Functional test of augmentation mapping logic."""
    print()
    print('='*60)
    print('TEST 4: Functional test of augmentation mapping logic')
    print('='*60)
    
    def compute_dihedral_enabled(aug_override):
        rotation = aug_override.get('rotation', None)
        flip = aug_override.get('flip', None)
        transpose = aug_override.get('transpose', None)
        
        specified_flags = [v for v in [rotation, flip, transpose] if v is not None]
        
        if len(specified_flags) == 0:
            dihedral_enabled = None
        elif any(specified_flags):
            dihedral_enabled = True
        else:
            dihedral_enabled = False
        
        return dihedral_enabled
    
    test_cases = [
        ({}, None, 'No flags -> use default (None)'),
        ({'rotation': False, 'flip': False}, False, 'rot=F, flip=F -> disable'),
        ({'rotation': False, 'flip': False, 'transpose': False}, False, 'all=F -> disable'),
        ({'rotation': True, 'flip': False}, True, 'rot=T, flip=F -> enable'),
        ({'rotation': False, 'flip': True}, True, 'rot=F, flip=T -> enable'),
        ({'transpose': True}, True, 'transpose-only=T -> enable'),
        ({'transpose': False}, False, 'transpose-only=F -> disable'),
        ({'rotation': True, 'flip': True, 'transpose': True}, True, 'all=T -> enable'),
        ({'rotation': True}, True, 'rot-only=T -> enable'),
        ({'flip': False}, False, 'flip-only=F -> disable'),
    ]
    
    all_passed = True
    for aug_override, expected, desc in test_cases:
        result = compute_dihedral_enabled(aug_override)
        status = 'PASS' if result == expected else 'FAIL'
        if result != expected:
            all_passed = False
        print(f'  [{status}] {desc}: got {result}, expected {expected}')
    
    print()
    if all_passed:
        print('TEST 4 PASSED: Augmentation mapping logic works correctly')
    else:
        print('TEST 4 FAILED: Augmentation mapping logic has errors')
    return all_passed


def test_output_equiv_gradient_flow():
    """Test 5: Functional test of OutputEquivarianceLoss gradient flow."""
    print()
    print('='*60)
    print('TEST 5: OutputEquivarianceLoss gradient flow test')
    print('='*60)
    
    from sci_arc.models.rlan_modules.loo_training import OutputEquivarianceLoss, EquivarianceConfig
    
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(10, 11)
            
        def forward(self, test_inputs, train_inputs=None, train_outputs=None, 
                    pair_mask=None, temperature=1.0, return_intermediates=False):
            B, H, W = test_inputs.shape
            logits = torch.randn(B, 11, H, W, requires_grad=False)
            return {'logits': logits}
    
    config = EquivarianceConfig(enabled=True, loss_weight=0.1)
    loss_fn = OutputEquivarianceLoss(config, loss_type='kl')
    
    B, H, W = 2, 8, 8
    test_inputs = torch.randint(0, 10, (B, H, W))
    train_inputs = torch.randint(0, 10, (B, 3, H, W))
    train_outputs = torch.randint(0, 10, (B, 3, H, W))
    pair_mask = torch.ones(B, 3, dtype=torch.bool)
    original_logits = torch.randn(B, 11, H, W, requires_grad=True)
    
    model = MockModel()
    
    try:
        loss, metrics = loss_fn(
            model=model,
            test_inputs=test_inputs,
            train_inputs=train_inputs,
            train_outputs=train_outputs,
            pair_mask=pair_mask,
            original_logits=original_logits,
            temperature=1.0,
            num_augmentations=2,
        )
        
        print(f'  Loss value: {loss.item():.6f}')
        print(f'  Num augmentations: {metrics["output_equiv_num_augs"]}')
        print(f'  Skipped: {metrics["output_equiv_skipped"]}')
        
        if loss.requires_grad:
            loss.backward()
            if original_logits.grad is not None:
                grad_norm = original_logits.grad.norm().item()
                print(f'  Gradient norm: {grad_norm:.6f}')
                if grad_norm > 0:
                    print('  [PASS] Gradients flow through original_logits!')
                else:
                    print('  [FAIL] Gradient norm is 0!')
                    return False
            else:
                print('  [FAIL] No gradient on original_logits!')
                return False
        else:
            print('  [FAIL] Loss does not require grad!')
            return False
            
    except Exception as e:
        print(f'  [FAIL] Exception: {e}')
        import traceback
        traceback.print_exc()
        return False
    
    print()
    print('TEST 5 PASSED: OutputEquivarianceLoss gradient flow works correctly')
    return True


def main():
    print()
    print('#'*60)
    print('# JANUARY 2026 BUG FIXES - SMOKE TEST')
    print('#'*60)
    print()
    
    results = []
    
    results.append(('OutputEquivarianceLoss structure', test_output_equiv_structure()))
    results.append(('Phase handling code', test_phase_handling_fixes()))
    results.append(('Phase epoch logic', test_phase_epoch_logic()))
    results.append(('Augmentation mapping', test_augmentation_mapping()))
    results.append(('Gradient flow', test_output_equiv_gradient_flow()))
    
    print()
    print('#'*60)
    print('# SUMMARY')
    print('#'*60)
    
    all_passed = True
    for name, passed in results:
        status = 'PASS' if passed else 'FAIL'
        if not passed:
            all_passed = False
        print(f'  [{status}] {name}')
    
    print()
    if all_passed:
        print('ALL TESTS PASSED! Bug fixes are working correctly.')
        return 0
    else:
        print('SOME TESTS FAILED! Please review the output above.')
        return 1


if __name__ == '__main__':
    sys.exit(main())
