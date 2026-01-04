#!/usr/bin/env python3
"""
Unit tests for metric-based phase gating.

Tests:
1. Config parsing and default values
2. Gate criteria checking logic
3. Phase transition edge cases
4. Backward compatibility (epoch-only mode)
"""

import sys
import yaml
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_config_structure():
    """Test that YAML config has correct phase_readiness structure."""
    print("\n[Test 1] Config structure:")
    
    cfg = yaml.safe_load(open('configs/rlan_stable_dev_merged.yaml', encoding='utf-8'))
    phased_training_config = cfg.get('training', {}).get('phased_training', {})
    phase_readiness_config = phased_training_config.get('phase_readiness', {})
    
    assert 'gate_a_to_b' in phase_readiness_config, 'Missing gate_a_to_b'
    assert 'gate_b_to_c' in phase_readiness_config, 'Missing gate_b_to_c'
    assert 'use_metric_gating' in phase_readiness_config, 'Missing use_metric_gating'
    
    print('  ✓ gate_a_to_b present')
    print('  ✓ gate_b_to_c present')
    print('  ✓ use_metric_gating present')
    print('  PASSED')


def test_a_to_b_thresholds():
    """Test A->B gate threshold values are reasonable."""
    print("\n[Test 2] A->B gate thresholds:")
    
    cfg = yaml.safe_load(open('configs/rlan_stable_dev_merged.yaml', encoding='utf-8'))
    gate_a = cfg['training']['phased_training']['phase_readiness']['gate_a_to_b']
    
    min_epochs = gate_a.get('min_epochs_in_phase_a', 10)
    shape_max = gate_a.get('shape_mismatch_max', 0.40)
    fg_min = gate_a.get('fg_accuracy_min', 0.45)
    patience = gate_a.get('patience', 2)
    
    assert min_epochs >= 5, f"min_epochs too low: {min_epochs}"
    # shape_mismatch_max=1.0 means disabled (valid for datasets with inherent shape mismatch)
    assert 0.1 <= shape_max <= 1.0, f"shape_mismatch_max out of range: {shape_max}"
    assert 0.2 <= fg_min <= 0.8, f"fg_accuracy_min out of range: {fg_min}"
    assert 1 <= patience <= 5, f"patience out of range: {patience}"
    
    print(f'  min_epochs_in_phase_a: {min_epochs} ✓')
    if shape_max >= 1.0:
        print(f'  shape_mismatch_max: DISABLED (1.0) ✓')
    else:
        print(f'  shape_mismatch_max: {shape_max} ✓')
    print(f'  fg_accuracy_min: {fg_min} ✓')
    print(f'  patience: {patience} ✓')
    print('  PASSED')


def test_b_to_c_thresholds():
    """Test B->C gate threshold values are reasonable."""
    print("\n[Test 3] B->C gate thresholds:")
    
    cfg = yaml.safe_load(open('configs/rlan_stable_dev_merged.yaml', encoding='utf-8'))
    gate_b = cfg['training']['phased_training']['phase_readiness']['gate_b_to_c']
    
    min_epochs = gate_b.get('min_epochs_in_phase_b', 5)
    shape_max = gate_b.get('shape_mismatch_max', 0.25)
    fg_min = gate_b.get('fg_accuracy_min', 0.50)
    tta_min = gate_b.get('tta_exact_match_min', 0.01)
    vote_tie_max = gate_b.get('vote_tie_max', 0.30)
    
    assert min_epochs >= 3, f"min_epochs too low: {min_epochs}"
    # shape_mismatch_max=1.0 means disabled (valid for datasets with inherent shape mismatch)
    assert 0 <= shape_max <= 1.0, f"shape_mismatch_max out of range: {shape_max}"
    assert fg_min > 0.3, f"fg_accuracy_min too lenient: {fg_min}"
    # tta_exact_match_min=0 means disabled (early epochs rarely get exact match)
    assert 0 <= tta_min <= 0.1, f"tta_exact_match_min out of range: {tta_min}"
    assert 0 <= vote_tie_max <= 1.0, f"vote_tie_max out of range: {vote_tie_max}"
    
    print(f'  min_epochs_in_phase_b: {min_epochs} ✓')
    if shape_max >= 1.0:
        print(f'  shape_mismatch_max: DISABLED (1.0) ✓')
    else:
        print(f'  shape_mismatch_max: {shape_max} ✓')
    print(f'  fg_accuracy_min: {fg_min} ✓')
    if tta_min <= 0:
        print(f'  tta_exact_match_min: DISABLED (0) ✓')
    else:
        print(f'  tta_exact_match_min: {tta_min} ✓')
    print(f'  vote_tie_max: {vote_tie_max} ✓')
    print('  PASSED')


def test_backward_compatibility():
    """Test that configs without phase_readiness still work."""
    print("\n[Test 4] Backward compatibility:")
    
    # Simulate old config without phase_readiness
    phased_training_config = {
        'enabled': True,
        'phase_a': {'end_epoch': 10},
        'phase_b': {'start_epoch': 11, 'end_epoch': 20},
        'phase_c': {'start_epoch': 21}
    }
    
    phase_readiness = phased_training_config.get('phase_readiness', {})
    use_metric_gating = phase_readiness.get('use_metric_gating', False)
    
    assert use_metric_gating == False, f"Default should be False, got {use_metric_gating}"
    print('  use_metric_gating defaults to False when config absent ✓')
    
    # Gate defaults should also work
    gate_a = phase_readiness.get('gate_a_to_b', {})
    min_epochs = gate_a.get('min_epochs_in_phase_a', 10)
    assert min_epochs == 10, "Default min_epochs should be 10"
    print('  Gate thresholds have sensible defaults ✓')
    print('  PASSED')


def test_gate_criteria_simulation():
    """Simulate gate criteria checking with mock metrics."""
    print("\n[Test 5] Gate criteria simulation:")
    
    cfg = yaml.safe_load(open('configs/rlan_stable_dev_merged.yaml', encoding='utf-8'))
    gate_a_cfg = cfg['training']['phased_training']['phase_readiness']['gate_a_to_b']
    
    shape_max = gate_a_cfg.get('shape_mismatch_max', 0.40)
    fg_min = gate_a_cfg.get('fg_accuracy_min', 0.35)
    
    # Mock metrics that should FAIL the gate (low FG accuracy)
    mock_eval_metrics_fail = {'fg_accuracy': 0.20}  # Below threshold
    mock_trm_metrics_fail = {
        'shape_mismatch_count': 54,  # 54% - dataset property
        'total_tasks': 100,
        'exact_match': 0.0
    }
    
    fg_ok = mock_eval_metrics_fail['fg_accuracy'] >= fg_min
    assert not fg_ok, f"FG accuracy 20% should fail (threshold={fg_min})"
    print('  Low FG accuracy correctly rejected ✓')
    
    # If shape_mismatch is disabled (1.0), 54% should pass
    if shape_max >= 1.0:
        shape_rate = mock_trm_metrics_fail['shape_mismatch_count'] / mock_trm_metrics_fail['total_tasks']
        shape_ok = shape_rate <= shape_max
        assert shape_ok, "Shape mismatch 54% should pass when disabled (threshold=1.0)"
        print('  Shape mismatch 54% passes when disabled ✓')
    
    # Mock metrics that should PASS the gate
    mock_eval_metrics_pass = {'fg_accuracy': 0.45}  # Above threshold
    
    fg_ok = mock_eval_metrics_pass['fg_accuracy'] >= fg_min
    assert fg_ok, f"FG accuracy 45% should pass (threshold={fg_min})"
    print('  Metrics above threshold correctly accepted ✓')
    print('  PASSED')


def test_code_syntax():
    """Verify the train_rlan.py parses correctly."""
    print("\n[Test 6] Code syntax:")
    
    import ast
    with open('scripts/train_rlan.py', encoding='utf-8') as f:
        code = f.read()
    
    try:
        ast.parse(code)
        print('  train_rlan.py syntax: VALID ✓')
    except SyntaxError as e:
        print(f'  train_rlan.py syntax: FAILED at line {e.lineno}')
        raise
    
    print('  PASSED')


def test_import():
    """Test that train_rlan imports cleanly."""
    print("\n[Test 7] Module import:")
    
    try:
        from train_rlan import main
        print('  train_rlan imports successfully ✓')
    except ImportError as e:
        print(f'  Import failed: {e}')
        raise
    
    print('  PASSED')


def main():
    """Run all tests."""
    print("=" * 60)
    print("METRIC-BASED PHASE GATING UNIT TESTS")
    print("=" * 60)
    
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    try:
        test_config_structure()
        test_a_to_b_thresholds()
        test_b_to_c_thresholds()
        test_backward_compatibility()
        test_gate_criteria_simulation()
        test_code_syntax()
        test_import()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
        return 0
    except AssertionError as e:
        print(f"\nTEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\nTEST ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
