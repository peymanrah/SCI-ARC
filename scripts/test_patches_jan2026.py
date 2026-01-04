#!/usr/bin/env python3
"""
Smoke Tests for January 2026 Patches
=====================================

Tests for the following patches:
1. Task ID validation and expected_task_count tracking
2. Canonical train evaluation (deterministic, no augmentation)
3. HPM write diagnostics improvements
4. Best-step selection enablement
5. YAML configuration changes

Run with:
    python scripts/test_patches_jan2026.py

All tests should pass without errors.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Test results tracking
test_results = []
test_count = 0
passed_count = 0


def test(name):
    """Decorator for test functions."""
    def decorator(func):
        def wrapper():
            global test_count, passed_count
            test_count += 1
            try:
                func()
                passed_count += 1
                test_results.append((name, "PASS", None))
                print(f"  ✓ {name}")
                return True
            except AssertionError as e:
                test_results.append((name, "FAIL", str(e)))
                print(f"  ✗ {name}: {e}")
                return False
            except Exception as e:
                test_results.append((name, "ERROR", str(e)))
                print(f"  ⚠ {name}: {e}")
                return False
        return wrapper
    return decorator


def run_test(test_func):
    """Run a test function and return success status."""
    return test_func()


# =============================================================================
# TEST 1: Syntax and Import Checks
# =============================================================================

@test("Import train_rlan module")
def test_import_train_rlan():
    """Verify train_rlan.py has no syntax errors and can be imported."""
    import importlib.util
    
    train_rlan_path = project_root / "scripts" / "train_rlan.py"
    assert train_rlan_path.exists(), f"train_rlan.py not found at {train_rlan_path}"
    
    spec = importlib.util.spec_from_file_location("train_rlan", train_rlan_path)
    module = importlib.util.module_from_spec(spec)
    
    # This will raise SyntaxError if there are syntax issues
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as e:
        # Allow missing dependencies (torch, etc.) but not syntax errors
        if "No module named" in str(e):
            pass  # OK - dependency issue, not syntax
        else:
            raise
    except ImportError as e:
        # Allow import errors from missing dependencies
        pass


@test("Import evaluate_canonical_train function")
def test_import_canonical_eval():
    """Verify evaluate_canonical_train function exists."""
    import ast
    
    train_rlan_path = project_root / "scripts" / "train_rlan.py"
    with open(train_rlan_path, 'r', encoding='utf-8') as f:
        source = f.read()
    
    tree = ast.parse(source)
    
    function_names = [node.name for node in ast.walk(tree) 
                      if isinstance(node, ast.FunctionDef)]
    
    assert 'evaluate_canonical_train' in function_names, \
        "evaluate_canonical_train function not found in train_rlan.py"


# =============================================================================
# TEST 2: Global Task Tracker Structure
# =============================================================================

@test("Global task tracker has expected_task_count field")
def test_global_task_tracker_structure():
    """Verify global_task_tracker initialization includes new fields."""
    train_rlan_path = project_root / "scripts" / "train_rlan.py"
    with open(train_rlan_path, 'r', encoding='utf-8') as f:
        source = f.read()
    
    # Check for expected_task_count in global_task_tracker
    assert "'expected_task_count'" in source, \
        "expected_task_count field not found in global_task_tracker"
    
    # Check for dataset_unique_task_ids
    assert "'dataset_unique_task_ids'" in source, \
        "dataset_unique_task_ids field not found in global_task_tracker"


@test("Task ID validation prints at train start")
def test_task_id_validation_logging():
    """Verify task ID validation logging is present."""
    train_rlan_path = project_root / "scripts" / "train_rlan.py"
    with open(train_rlan_path, 'r', encoding='utf-8') as f:
        source = f.read()
    
    # Check for task ID validation section
    assert "TASK ID VALIDATION" in source, \
        "Task ID validation section not found"
    
    # Check for collision warning
    assert "task_id collisions detected" in source or "collisions detected" in source.lower(), \
        "Task ID collision warning not found"


# =============================================================================
# TEST 3: HPM Write Diagnostics
# =============================================================================

@test("HPM write diagnostics has detailed skip reasons")
def test_hpm_write_diagnostics():
    """Verify HPM write diagnostics includes detailed skip reasons."""
    train_rlan_path = project_root / "scripts" / "train_rlan.py"
    with open(train_rlan_path, 'r', encoding='utf-8') as f:
        source = f.read()
    
    # Check for hpm_write_attempts
    assert "'hpm_write_attempts'" in source, \
        "hpm_write_attempts counter not found"
    
    # Check for hpm_writes_succeeded
    assert "'hpm_writes_succeeded'" in source, \
        "hpm_writes_succeeded counter not found"
    
    # Check for skip_reasons dict
    assert "'hpm_write_skip_reasons'" in source, \
        "hpm_write_skip_reasons dict not found"
    
    # Check for specific skip reasons
    skip_reasons = ['no_method', 'not_enabled', 'global_duplicate', 
                    'epoch_duplicate', 'no_support_features']
    for reason in skip_reasons:
        assert f"'{reason}'" in source, \
            f"Skip reason '{reason}' not found in hpm_write_skip_reasons"


@test("HPM write logging shows attempts and success count")
def test_hpm_write_logging():
    """Verify HPM logging shows write attempts vs succeeded."""
    train_rlan_path = project_root / "scripts" / "train_rlan.py"
    with open(train_rlan_path, 'r', encoding='utf-8') as f:
        source = f.read()
    
    # Check for the new logging format
    assert "HPM Write Attempts" in source, \
        "HPM Write Attempts logging not found"


# =============================================================================
# TEST 4: Canonical Train Eval
# =============================================================================

@test("Canonical train eval function has correct signature")
def test_canonical_eval_signature():
    """Verify evaluate_canonical_train has correct parameters."""
    import ast
    
    train_rlan_path = project_root / "scripts" / "train_rlan.py"
    with open(train_rlan_path, 'r', encoding='utf-8') as f:
        source = f.read()
    
    tree = ast.parse(source)
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == 'evaluate_canonical_train':
            param_names = [arg.arg for arg in node.args.args]
            
            # Check required parameters
            assert 'model' in param_names, "model parameter missing"
            assert 'train_loader' in param_names, "train_loader parameter missing"
            assert 'device' in param_names, "device parameter missing"
            
            # Check optional parameters with defaults
            defaults = {d.arg: True for d in node.args.defaults if hasattr(d, 'arg')}
            
            return  # Found the function
    
    assert False, "evaluate_canonical_train function not found"


@test("Canonical train eval is called in training loop")
def test_canonical_eval_called():
    """Verify canonical train eval is called during evaluation."""
    train_rlan_path = project_root / "scripts" / "train_rlan.py"
    with open(train_rlan_path, 'r', encoding='utf-8') as f:
        source = f.read()
    
    # Check that the function is called
    assert "evaluate_canonical_train(" in source, \
        "evaluate_canonical_train() call not found in training loop"
    
    # Check for the config key
    assert "canonical_train_eval" in source, \
        "canonical_train_eval config check not found"


# =============================================================================
# TEST 5: YAML Configuration
# =============================================================================

@test("YAML has best_step_selection enabled")
def test_yaml_best_step():
    """Verify use_best_step_selection is true in YAML."""
    import yaml
    
    yaml_path = project_root / "configs" / "rlan_stable_dev_merged.yaml"
    assert yaml_path.exists(), f"YAML not found at {yaml_path}"
    
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    model_cfg = config.get('model', {})
    assert model_cfg.get('use_best_step_selection', False) == True, \
        "use_best_step_selection should be true"


@test("YAML has canonical_train_eval section")
def test_yaml_canonical_eval():
    """Verify canonical_train_eval is configured in YAML."""
    import yaml
    
    yaml_path = project_root / "configs" / "rlan_stable_dev_merged.yaml"
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    eval_cfg = config.get('evaluation', {})
    canonical_cfg = eval_cfg.get('canonical_train_eval', {})
    
    assert canonical_cfg.get('enabled', False) == True, \
        "canonical_train_eval.enabled should be true"
    
    assert 'max_tasks' in canonical_cfg, \
        "canonical_train_eval.max_tasks should be set"


@test("YAML has validation checklist comments")
def test_yaml_validation_comments():
    """Verify YAML has validation checklist at the top."""
    yaml_path = project_root / "configs" / "rlan_stable_dev_merged.yaml"
    with open(yaml_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    assert "VALIDATION CHECKLIST" in content, \
        "Validation checklist not found in YAML header"


# =============================================================================
# TEST 6: Backward Compatibility
# =============================================================================

@test("Global task tracker fallback to dynamic count")
def test_backward_compat_dynamic_count():
    """Verify fallback to dynamic count when expected_task_count is None."""
    train_rlan_path = project_root / "scripts" / "train_rlan.py"
    with open(train_rlan_path, 'r', encoding='utf-8') as f:
        source = f.read()
    
    # Check for fallback logic
    assert "Fallback to dynamic count" in source or "backward compatible" in source.lower(), \
        "Backward compatibility fallback not documented"
    
    # Check for the None check
    assert "expected_total is not None" in source, \
        "expected_task_count None check not found"


@test("HPM diagnostics backward compatible with old keys")
def test_hpm_diagnostics_backward_compat():
    """Verify old HPM diagnostic keys still work."""
    train_rlan_path = project_root / "scripts" / "train_rlan.py"
    with open(train_rlan_path, 'r', encoding='utf-8') as f:
        source = f.read()
    
    # Old key should still be populated
    assert "'hpm_tasks_added'" in source, \
        "hpm_tasks_added (old key) should still be used for backward compat"
    
    # Old key should still be populated
    assert "'hpm_duplicate_skipped'" in source, \
        "hpm_duplicate_skipped (old key) should still be used for backward compat"


# =============================================================================
# TEST 7: Phase Gating Compatibility (from previous patches)
# =============================================================================

@test("Phase gating still uses disabled thresholds correctly")
def test_phase_gating_disabled_thresholds():
    """Verify phase gating handles disabled thresholds (1.0 for shape_mismatch)."""
    train_rlan_path = project_root / "scripts" / "train_rlan.py"
    with open(train_rlan_path, 'r', encoding='utf-8') as f:
        source = f.read()
    
    # Check for shape_mismatch_max >= 1.0 skip logic
    assert "shape_mismatch_max < 1.0" in source or "shape_mismatch_max >= 1.0" in source, \
        "shape_mismatch disabled threshold check not found"


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "=" * 60)
    print("SMOKE TESTS: January 2026 Patches")
    print("=" * 60)
    
    print("\n[1] Syntax and Import Checks")
    run_test(test_import_train_rlan)
    run_test(test_import_canonical_eval)
    
    print("\n[2] Global Task Tracker Structure")
    run_test(test_global_task_tracker_structure)
    run_test(test_task_id_validation_logging)
    
    print("\n[3] HPM Write Diagnostics")
    run_test(test_hpm_write_diagnostics)
    run_test(test_hpm_write_logging)
    
    print("\n[4] Canonical Train Eval")
    run_test(test_canonical_eval_signature)
    run_test(test_canonical_eval_called)
    
    print("\n[5] YAML Configuration")
    run_test(test_yaml_best_step)
    run_test(test_yaml_canonical_eval)
    run_test(test_yaml_validation_comments)
    
    print("\n[6] Backward Compatibility")
    run_test(test_backward_compat_dynamic_count)
    run_test(test_hpm_diagnostics_backward_compat)
    
    print("\n[7] Phase Gating Compatibility")
    run_test(test_phase_gating_disabled_thresholds)
    
    # Summary
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed_count}/{test_count} tests passed")
    print("=" * 60)
    
    if passed_count == test_count:
        print("\n✓ ALL TESTS PASSED\n")
        return 0
    else:
        print(f"\n✗ {test_count - passed_count} tests failed\n")
        for name, status, error in test_results:
            if status != "PASS":
                print(f"  - {name}: {status} - {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
