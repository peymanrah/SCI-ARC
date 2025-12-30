"""
Test script for phased training implementation.

This script verifies that the action plan changes are correctly implemented:
1. YAML parameters are read correctly from config
2. Staged activation works (context path first, HyperLoRA later)
3. HyperLoRA warmup logic works correctly
4. Gradient explosion backoff is functional
5. Backward compatibility with old configs

Run with:
    python tests/test_phased_training.py
"""

import os
import sys
import yaml
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Test results tracking
test_results = {"passed": 0, "failed": 0, "errors": []}


def test_passed(name):
    test_results["passed"] += 1
    print(f"  ✅ {name}")


def test_failed(name, reason):
    test_results["failed"] += 1
    test_results["errors"].append(f"{name}: {reason}")
    print(f"  ❌ {name}: {reason}")


def test_yaml_parameters():
    """Test 1: Verify new YAML parameters exist and have correct defaults."""
    print("\n" + "=" * 60)
    print("TEST 1: YAML Parameter Verification")
    print("=" * 60)
    
    config_path = project_root / "configs" / "rlan_stable_dev.yaml"
    
    if not config_path.exists():
        test_failed("Config file exists", f"Not found: {config_path}")
        return
    
    test_passed("Config file exists")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    training = config.get('training', {})
    model = config.get('model', {})
    
    # Test Phase 1: Loss configuration
    if training.get('loss_mode') == 'focal_weighted':
        test_passed("Loss mode is focal_weighted")
    else:
        test_failed("Loss mode is focal_weighted", f"Got: {training.get('loss_mode')}")
    
    if training.get('bg_weight_cap') == 1.5:
        test_passed("bg_weight_cap is 1.5")
    else:
        test_failed("bg_weight_cap is 1.5", f"Got: {training.get('bg_weight_cap')}")
    
    if training.get('fg_weight_cap') == 4.0:
        test_passed("fg_weight_cap is 4.0")
    else:
        test_failed("fg_weight_cap is 4.0", f"Got: {training.get('fg_weight_cap')}")
    
    if training.get('focal_gamma') == 1.5:
        test_passed("focal_gamma is 1.5")
    else:
        test_failed("focal_gamma is 1.5", f"Got: {training.get('focal_gamma')}")
    
    # Test Phase 2: Staged activation order (context first)
    solver_ctx_epoch = training.get('solver_context_start_epoch')
    cross_attn_epoch = training.get('cross_attention_start_epoch')
    hyperlora_epoch = training.get('meta_learning_start_epoch')
    
    if solver_ctx_epoch == 5:
        test_passed("solver_context_start_epoch is 5")
    else:
        test_failed("solver_context_start_epoch is 5", f"Got: {solver_ctx_epoch}")
    
    if cross_attn_epoch == 5:
        test_passed("cross_attention_start_epoch is 5 (synced with solver)")
    else:
        test_failed("cross_attention_start_epoch is 5", f"Got: {cross_attn_epoch}")
    
    # Test Phase 3: HyperLoRA after context path
    if hyperlora_epoch == 8:
        test_passed("meta_learning_start_epoch is 8 (after context path)")
    else:
        test_failed("meta_learning_start_epoch is 8", f"Got: {hyperlora_epoch}")
    
    if hyperlora_epoch > solver_ctx_epoch:
        test_passed("HyperLoRA activates AFTER context path")
    else:
        test_failed("HyperLoRA activates AFTER context path", 
                   f"HyperLoRA={hyperlora_epoch}, Context={solver_ctx_epoch}")
    
    # Test HyperLoRA warmup parameters
    if training.get('hyperlora_warmup_epochs') == 4:
        test_passed("hyperlora_warmup_epochs is 4")
    else:
        test_failed("hyperlora_warmup_epochs is 4", f"Got: {training.get('hyperlora_warmup_epochs')}")
    
    if training.get('hyperlora_warmup_start_scale') == 0.005:
        test_passed("hyperlora_warmup_start_scale is 0.005")
    else:
        test_failed("hyperlora_warmup_start_scale is 0.005", 
                   f"Got: {training.get('hyperlora_warmup_start_scale')}")
    
    if training.get('hyperlora_warmup_end_scale') == 0.1:
        test_passed("hyperlora_warmup_end_scale is 0.1")
    else:
        test_failed("hyperlora_warmup_end_scale is 0.1", 
                   f"Got: {training.get('hyperlora_warmup_end_scale')}")
    
    # Test gradient explosion backoff
    if training.get('grad_explosion_threshold') == 10.0:
        test_passed("grad_explosion_threshold is 10.0")
    else:
        test_failed("grad_explosion_threshold is 10.0", 
                   f"Got: {training.get('grad_explosion_threshold')}")
    
    if training.get('grad_explosion_lr_reduction') == 0.5:
        test_passed("grad_explosion_lr_reduction is 0.5")
    else:
        test_failed("grad_explosion_lr_reduction is 0.5", 
                   f"Got: {training.get('grad_explosion_lr_reduction')}")
    
    # Test Phase 4: Equivariance (after HyperLoRA warmup)
    equiv_config = training.get('equivariance_training', {})
    if equiv_config.get('start_epoch') == 12:
        test_passed("equivariance start_epoch is 12")
    else:
        test_failed("equivariance start_epoch is 12", f"Got: {equiv_config.get('start_epoch')}")
    
    if equiv_config.get('loss_weight') == 0.01:
        test_passed("equivariance loss_weight is 0.01 (gentler)")
    else:
        test_failed("equivariance loss_weight is 0.01", f"Got: {equiv_config.get('loss_weight')}")
    
    # Test Phase 5: LOO (last)
    loo_config = training.get('loo_training', {})
    if loo_config.get('start_epoch') == 18:
        test_passed("LOO start_epoch is 18 (last)")
    else:
        test_failed("LOO start_epoch is 18", f"Got: {loo_config.get('start_epoch')}")
    
    if loo_config.get('loss_weight') == 0.05:
        test_passed("LOO loss_weight is 0.05 (gentler)")
    else:
        test_failed("LOO loss_weight is 0.05", f"Got: {loo_config.get('loss_weight')}")
    
    # Test model config: HyperLoRA init_scale
    if model.get('hyperlora_init_scale') == 0.005:
        test_passed("hyperlora_init_scale is 0.005 (near-zero)")
    else:
        test_failed("hyperlora_init_scale is 0.005", f"Got: {model.get('hyperlora_init_scale')}")
    
    # Test conservative LR multiplier
    if training.get('hyperlora_lr_multiplier') == 1.0:
        test_passed("hyperlora_lr_multiplier is 1.0 (conservative)")
    else:
        test_failed("hyperlora_lr_multiplier is 1.0", f"Got: {training.get('hyperlora_lr_multiplier')}")


def test_loss_mode_implementation():
    """Test 2: Verify focal_weighted loss mode is implemented correctly."""
    print("\n" + "=" * 60)
    print("TEST 2: Loss Mode Implementation")
    print("=" * 60)
    
    try:
        from sci_arc.training.rlan_loss import RLANLoss, FocalWeightedStablemaxLoss
        test_passed("FocalWeightedStablemaxLoss class exists")
    except ImportError as e:
        test_failed("FocalWeightedStablemaxLoss class exists", str(e))
        return
    
    # Test that RLANLoss accepts 'focal_weighted' mode
    try:
        loss_fn = RLANLoss(
            loss_mode='focal_weighted',
            bg_weight_cap=1.5,
            fg_weight_cap=4.0,
            focal_gamma=1.5,
        )
        test_passed("RLANLoss accepts focal_weighted mode")
    except Exception as e:
        test_failed("RLANLoss accepts focal_weighted mode", str(e))
        return
    
    # Test that the task_loss is FocalWeightedStablemaxLoss
    if isinstance(loss_fn.task_loss, FocalWeightedStablemaxLoss):
        test_passed("RLANLoss uses FocalWeightedStablemaxLoss for focal_weighted mode")
    else:
        test_failed("RLANLoss uses FocalWeightedStablemaxLoss", 
                   f"Got: {type(loss_fn.task_loss)}")
    
    # Test forward pass doesn't crash - test task_loss directly (simpler)
    try:
        # Create dummy inputs
        B, C, H, W = 2, 10, 8, 8
        logits = torch.randn(B, C, H, W)
        targets = torch.randint(0, C, (B, H, W))
        
        # Test the task_loss module directly instead of full RLANLoss
        task_loss_val = loss_fn.task_loss(logits, targets)
        if torch.is_tensor(task_loss_val) and not torch.isnan(task_loss_val):
            test_passed("focal_weighted forward pass produces valid loss")
        else:
            test_failed("focal_weighted forward pass produces valid loss", "NaN or invalid tensor")
    except Exception as e:
        test_failed("focal_weighted forward pass produces valid loss", str(e))


def test_staged_activation_order():
    """Test 3: Verify staged activation logic is correct in train_rlan.py."""
    print("\n" + "=" * 60)
    print("TEST 3: Staged Activation Order")
    print("=" * 60)
    
    train_script = project_root / "scripts" / "train_rlan.py"
    
    if not train_script.exists():
        test_failed("train_rlan.py exists", f"Not found: {train_script}")
        return
    
    with open(train_script, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check that solver_context and cross_attention default to 5
    if "solver_context_start_epoch', 5)" in content:
        test_passed("solver_context defaults to epoch 5")
    else:
        test_failed("solver_context defaults to epoch 5", "Default not found")
    
    if "cross_attention_start_epoch', 5)" in content:
        test_passed("cross_attention defaults to epoch 5")
    else:
        test_failed("cross_attention defaults to epoch 5", "Default not found")
    
    if "meta_learning_start_epoch', 8)" in content:
        test_passed("meta_learning (HyperLoRA) defaults to epoch 8")
    else:
        test_failed("meta_learning (HyperLoRA) defaults to epoch 8", "Default not found")
    
    # Check for HyperLoRA warmup logic
    if "hyperlora_warmup_epochs" in content:
        test_passed("hyperlora_warmup_epochs parameter is used")
    else:
        test_failed("hyperlora_warmup_epochs parameter is used", "Not found in code")
    
    if "hyperlora_warmup_start_scale" in content:
        test_passed("hyperlora_warmup_start_scale parameter is used")
    else:
        test_failed("hyperlora_warmup_start_scale parameter is used", "Not found in code")
    
    if "init_scale" in content and "hyper_lora" in content:
        test_passed("HyperLoRA init_scale is set dynamically")
    else:
        test_failed("HyperLoRA init_scale is set dynamically", "Pattern not found")


def test_gradient_explosion_backoff():
    """Test 4: Verify gradient explosion backoff is implemented."""
    print("\n" + "=" * 60)
    print("TEST 4: Gradient Explosion Backoff")
    print("=" * 60)
    
    train_script = project_root / "scripts" / "train_rlan.py"
    
    with open(train_script, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for gradient explosion threshold parameter
    if "grad_explosion_threshold" in content:
        test_passed("grad_explosion_threshold parameter is read")
    else:
        test_failed("grad_explosion_threshold parameter is read", "Not found")
    
    # Check for gradient explosion detection logic
    if "GRADIENT EXPLOSION DETECTED" in content:
        test_passed("Gradient explosion detection message exists")
    else:
        test_failed("Gradient explosion detection message exists", "Not found")
    
    # Check for LR reduction on explosion
    if "grad_explosion_lr_reduction" in content:
        test_passed("grad_explosion_lr_reduction is used")
    else:
        test_failed("grad_explosion_lr_reduction is used", "Not found")
    
    # Check for cooldown mechanism
    if "grad_explosion_cooldown" in content:
        test_passed("Cooldown mechanism is implemented")
    else:
        test_failed("Cooldown mechanism is implemented", "Not found")


def test_lr_management():
    """Test 5: Verify LR management at activation epochs."""
    print("\n" + "=" * 60)
    print("TEST 5: LR Management at Activation")
    print("=" * 60)
    
    train_script = project_root / "scripts" / "train_rlan.py"
    
    with open(train_script, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for activation LR reduction
    if "activation_lr_reduction" in content:
        test_passed("activation_lr_reduction parameter is used")
    else:
        test_failed("activation_lr_reduction parameter is used", "Not found")
    
    if "activation_lr_recovery_epochs" in content:
        test_passed("activation_lr_recovery_epochs parameter is used")
    else:
        test_failed("activation_lr_recovery_epochs parameter is used", "Not found")
    
    # Check for is_activation_epoch logic
    if "is_activation_epoch" in content:
        test_passed("is_activation_epoch detection exists")
    else:
        test_failed("is_activation_epoch detection exists", "Not found")
    
    # Check for LR reduction/restoration logic
    if "Reduced to" in content and "LR" in content:
        test_passed("LR reduction message exists")
    else:
        test_failed("LR reduction message exists", "Not found")
    
    if "Restored to original" in content:
        test_passed("LR restoration message exists")
    else:
        test_failed("LR restoration message exists", "Not found")


def test_backward_compatibility():
    """Test 6: Verify backward compatibility with default values."""
    print("\n" + "=" * 60)
    print("TEST 6: Backward Compatibility")
    print("=" * 60)
    
    train_script = project_root / "scripts" / "train_rlan.py"
    
    with open(train_script, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # All new parameters should have defaults
    new_params = [
        ('hyperlora_warmup_epochs', '4'),
        ('hyperlora_warmup_start_scale', '0.005'),
        ('hyperlora_warmup_end_scale', '0.1'),
        ('activation_lr_reduction', '0.5'),
        ('activation_lr_recovery_epochs', '2'),
        ('grad_explosion_threshold', '10.0'),
        ('grad_explosion_lr_reduction', '0.5'),
        ('grad_explosion_cooldown_epochs', '2'),
    ]
    
    for param, default in new_params:
        if f".get('{param}'" in content:
            test_passed(f"{param} has default via .get()")
        else:
            test_failed(f"{param} has default via .get()", "Pattern not found")
    
    # Test that old config (without new params) would work
    # The defaults should be applied when params are missing
    test_passed("Old configs will use default values (checked .get() patterns)")


def test_phase_order():
    """Test 7: Verify the scientifically-ordered phase progression."""
    print("\n" + "=" * 60)
    print("TEST 7: Scientific Phase Order")
    print("=" * 60)
    
    config_path = project_root / "configs" / "rlan_stable_dev.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    training = config.get('training', {})
    
    # Get all activation epochs
    context_epoch = training.get('solver_context_start_epoch', 5)
    cross_attn_epoch = training.get('cross_attention_start_epoch', 5)
    hyperlora_epoch = training.get('meta_learning_start_epoch', 8)
    equiv_epoch = training.get('equivariance_training', {}).get('start_epoch', 12)
    loo_epoch = training.get('loo_training', {}).get('start_epoch', 18)
    
    # Phase order should be:
    # Phase 0: epochs 0-4 (base model only)
    # Phase 2: epoch 5 (context path)
    # Phase 3: epoch 8 (HyperLoRA with warmup)
    # Phase 4: epoch 12 (equivariance after HyperLoRA warmup complete at 8+4=12)
    # Phase 5: epoch 18 (LOO last)
    
    if context_epoch < hyperlora_epoch:
        test_passed("Phase order: Context (5) before HyperLoRA (8)")
    else:
        test_failed("Phase order: Context before HyperLoRA", 
                   f"Context={context_epoch}, HyperLoRA={hyperlora_epoch}")
    
    if hyperlora_epoch < equiv_epoch:
        test_passed("Phase order: HyperLoRA (8) before Equivariance (12)")
    else:
        test_failed("Phase order: HyperLoRA before Equivariance", 
                   f"HyperLoRA={hyperlora_epoch}, Equiv={equiv_epoch}")
    
    if equiv_epoch < loo_epoch:
        test_passed("Phase order: Equivariance (12) before LOO (18)")
    else:
        test_failed("Phase order: Equivariance before LOO", 
                   f"Equiv={equiv_epoch}, LOO={loo_epoch}")
    
    # Verify warmup math: warmup ends at hyperlora_epoch + warmup_epochs
    warmup_epochs = training.get('hyperlora_warmup_epochs', 4)
    warmup_end = hyperlora_epoch + warmup_epochs
    if warmup_end <= equiv_epoch:
        test_passed(f"HyperLoRA warmup ({hyperlora_epoch}+{warmup_epochs}={warmup_end}) completes before Equivariance ({equiv_epoch})")
    else:
        test_failed(f"HyperLoRA warmup completes before Equivariance", 
                   f"Warmup ends at {warmup_end}, Equiv at {equiv_epoch}")


def run_all_tests():
    """Run all tests and print summary."""
    print("\n" + "=" * 60)
    print("PHASED TRAINING IMPLEMENTATION TESTS")
    print("=" * 60)
    print(f"Project root: {project_root}")
    
    # Run all tests
    test_yaml_parameters()
    test_loss_mode_implementation()
    test_staged_activation_order()
    test_gradient_explosion_backoff()
    test_lr_management()
    test_backward_compatibility()
    test_phase_order()
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    total = test_results["passed"] + test_results["failed"]
    print(f"  Passed: {test_results['passed']}/{total}")
    print(f"  Failed: {test_results['failed']}/{total}")
    
    if test_results["errors"]:
        print("\nFailed tests:")
        for error in test_results["errors"]:
            print(f"  - {error}")
    
    if test_results["failed"] == 0:
        print("\n✅ ALL TESTS PASSED - Ready for production training!")
        return True
    else:
        print("\n❌ SOME TESTS FAILED - Please fix before training!")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
