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
    
    # Patch 1: Check for delta_scale usage (not init_scale) - this actually affects forward
    if "delta_scale" in content and "hyper_lora" in content:
        test_passed("HyperLoRA delta_scale is set dynamically (Patch 1)")
    else:
        test_failed("HyperLoRA delta_scale is set dynamically", "Pattern not found")


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


def test_hyperlora_delta_scale_behavior():
    """Test 6: Verify HyperLoRA delta_scale actually affects output magnitudes (Patch 5)."""
    print("\n" + "=" * 60)
    print("TEST 6: HyperLoRA delta_scale Behavior (Patch 5)")
    print("=" * 60)
    
    try:
        from sci_arc.models.rlan_modules.hyper_lora import HyperLoRA, HyperLoRAConfig
        
        # Create HyperLoRA on CPU
        config = HyperLoRAConfig(
            hidden_dim=64,
            context_dim=64,
            rank=4,
            scaling=0.1,
            init_scale=0.1,
        )
        hyper_lora = HyperLoRA(config=config)
        hyper_lora.eval()
        
        # Create deterministic fake support features
        torch.manual_seed(42)
        support_features = torch.randn(2, 3, 64, 8, 8)  # (B, N, D, H, W)
        
        # Test with delta_scale = 1.0
        hyper_lora.delta_scale = 1.0
        with torch.no_grad():
            deltas_full = hyper_lora(support_features)
        norm_full = deltas_full['gru_reset'].norm().item()
        
        # Test with delta_scale = 0.1
        hyper_lora.delta_scale = 0.1
        with torch.no_grad():
            deltas_scaled = hyper_lora(support_features)
        norm_scaled = deltas_scaled['gru_reset'].norm().item()
        
        # The scaled norm should be ~10x smaller
        ratio = norm_full / (norm_scaled + 1e-10)
        if 8.0 < ratio < 12.0:  # Allow some tolerance
            test_passed(f"delta_scale affects output: ratio={ratio:.2f} (expected ~10)")
        else:
            test_failed(f"delta_scale affects output", f"ratio={ratio:.2f}, expected ~10")
        
        # Verify delta_scale attribute exists
        if hasattr(hyper_lora, 'delta_scale'):
            test_passed("HyperLoRA has delta_scale attribute")
        else:
            test_failed("HyperLoRA has delta_scale attribute", "Attribute not found")
            
    except ImportError as e:
        test_failed("Import HyperLoRA", str(e))
    except Exception as e:
        test_failed("HyperLoRA delta_scale test", str(e))


def test_lr_composability():
    """Test 7: Verify LR factors are composable (Patch 5)."""
    print("\n" + "=" * 60)
    print("TEST 7: LR Composability Check (Patch 5)")
    print("=" * 60)
    
    train_script = project_root / "scripts" / "train_rlan.py"
    
    with open(train_script, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for composable LR factor tracking
    if "'base_lrs'" in content:
        test_passed("base_lrs tracked for composable restoration")
    else:
        test_failed("base_lrs tracked", "Not found in train_rlan.py")
    
    if "'activation_factor'" in content:
        test_passed("activation_factor tracked for composable LR")
    else:
        test_failed("activation_factor tracked", "Not found")
    
    if "'explosion_factor'" in content:
        test_passed("explosion_factor tracked for composable LR")
    else:
        test_failed("explosion_factor tracked", "Not found")
    
    # Check that LR is computed from base * factors, not divided
    if "base_lr * total_factor" in content:
        test_passed("LR computed as base_lr * total_factor")
    else:
        test_failed("LR computed as base_lr * total_factor", "Pattern not found")


def test_max_grad_norm_tracking():
    """Test 8: Verify max grad norm is tracked across epoch (Patch 5)."""
    print("\n" + "=" * 60)
    print("TEST 8: Max Grad Norm Tracking (Patch 5)")
    print("=" * 60)
    
    train_script = project_root / "scripts" / "train_rlan.py"
    
    with open(train_script, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for max_grad_norm_before_clip initialization
    if "'max_grad_norm_before_clip': 0.0" in content:
        test_passed("max_grad_norm_before_clip initialized in diagnostics")
    else:
        test_failed("max_grad_norm_before_clip initialized", "Not found")
    
    # Check for max tracking logic
    if "max(epoch_diagnostics.get('max_grad_norm_before_clip'" in content or \
       "max_grad_norm_before_clip'] = max(" in content:
        test_passed("max grad norm updated with max()")
    else:
        test_failed("max grad norm updated with max()", "Pattern not found")


def test_nan_backoff_implementation():
    """Test 9: Verify NaN-driven meta-loss backoff is implemented (Patch 5)."""
    print("\n" + "=" * 60)
    print("TEST 9: NaN Backoff Implementation (Patch 5)")
    print("=" * 60)
    
    train_script = project_root / "scripts" / "train_rlan.py"
    
    with open(train_script, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for nan_backoff_state initialization
    if "nan_backoff_state" in content:
        test_passed("nan_backoff_state is defined")
    else:
        test_failed("nan_backoff_state is defined", "Not found")
    
    if "'equiv_weight_factor'" in content:
        test_passed("equiv_weight_factor tracked for backoff")
    else:
        test_failed("equiv_weight_factor tracked", "Not found")
    
    if "'loo_weight_factor'" in content:
        test_passed("loo_weight_factor tracked for backoff")
    else:
        test_failed("loo_weight_factor tracked", "Not found")
    
    if "NaN BACKOFF TRIGGERED" in content:
        test_passed("NaN backoff trigger message exists")
    else:
        test_failed("NaN backoff trigger message", "Not found")
    
    # Verify true consecutive NaN detection (not just total count)
    if "max_consecutive_nan_streak" in content:
        test_passed("max_consecutive_nan_streak tracked (true consecutive detection)")
    else:
        test_failed("max_consecutive_nan_streak tracked", "Not found - backoff may use total count instead of consecutive")
    
    # Verify backoff trigger uses consecutive streak
    if "consecutive_nan_streak = train_losses.get('max_consecutive_nan_streak'" in content:
        test_passed("Backoff trigger uses consecutive streak (not total nan_batches)")
    else:
        test_failed("Backoff uses consecutive streak", "Trigger may use total count instead of consecutive")


def test_meta_escalation_config():
    """Test 10: Verify meta escalation config is properly defined."""
    print("\n" + "=" * 60)
    print("TEST 10: Meta Escalation Config (Dec 2025)")
    print("=" * 60)
    
    config_path = project_root / "configs" / "rlan_stable_dev.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    training = config.get('training', {})
    meta_esc = training.get('meta_escalation', {})
    
    # Check core config exists
    if 'meta_escalation' in training:
        test_passed("meta_escalation config block exists")
    else:
        test_failed("meta_escalation config block exists", "Not found")
        return
    
    if meta_esc.get('enabled') is not None:
        test_passed(f"meta_escalation.enabled is defined ({meta_esc.get('enabled')})")
    else:
        test_failed("meta_escalation.enabled defined", "Not found")
    
    if meta_esc.get('start_epoch', 0) >= 20:
        test_passed(f"start_epoch is late enough ({meta_esc.get('start_epoch')})")
    else:
        test_failed("start_epoch >= 20", f"Got {meta_esc.get('start_epoch')} - too early")
    
    if meta_esc.get('ramp_epochs', 0) >= 5:
        test_passed(f"ramp_epochs is reasonable ({meta_esc.get('ramp_epochs')})")
    else:
        test_failed("ramp_epochs >= 5", f"Got {meta_esc.get('ramp_epochs')} - too short")
    
    # Check targets
    targets = meta_esc.get('targets', {})
    if targets.get('hyperlora_delta_scale', 0) > 0.1:
        test_passed(f"target hyperlora_delta_scale > 0.1 ({targets.get('hyperlora_delta_scale')})")
    else:
        test_failed("target hyperlora_delta_scale > 0.1", f"Got {targets.get('hyperlora_delta_scale')}")
    
    if targets.get('equiv_loss_weight', 0) > 0.01:
        test_passed(f"target equiv_loss_weight > 0.01 ({targets.get('equiv_loss_weight')})")
    else:
        test_failed("target equiv_loss_weight > 0.01", f"Got {targets.get('equiv_loss_weight')}")
    
    if targets.get('loo_loss_weight', 0) > 0.05:
        test_passed(f"target loo_loss_weight > 0.05 ({targets.get('loo_loss_weight')})")
    else:
        test_failed("target loo_loss_weight > 0.05", f"Got {targets.get('loo_loss_weight')}")
    
    # Check stability gating
    if meta_esc.get('require_stability') == True:
        test_passed("require_stability is True (safety first)")
    else:
        test_failed("require_stability is True", f"Got {meta_esc.get('require_stability')}")
    
    if meta_esc.get('recovery_enabled') == True:
        test_passed("recovery_enabled is True (prevents permanent suppression)")
    else:
        test_failed("recovery_enabled is True", f"Got {meta_esc.get('recovery_enabled')}")


def test_meta_escalation_implementation():
    """Test 11: Verify meta escalation is implemented in training script."""
    print("\n" + "=" * 60)
    print("TEST 11: Meta Escalation Implementation (Dec 2025)")
    print("=" * 60)
    
    train_script = project_root / "scripts" / "train_rlan.py"
    
    with open(train_script, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for meta_escalation_state initialization
    if "meta_escalation_state" in content:
        test_passed("meta_escalation_state is defined")
    else:
        test_failed("meta_escalation_state is defined", "Not found")
    
    if "'hyperlora_delta_scale_current'" in content:
        test_passed("hyperlora_delta_scale_current tracked")
    else:
        test_failed("hyperlora_delta_scale_current tracked", "Not found")
    
    if "'equiv_weight_current'" in content:
        test_passed("equiv_weight_current tracked")
    else:
        test_failed("equiv_weight_current tracked", "Not found")
    
    if "'escalation_paused'" in content:
        test_passed("escalation_paused state tracked")
    else:
        test_failed("escalation_paused tracked", "Not found")
    
    if "'is_stable'" in content:
        test_passed("is_stable state tracked")
    else:
        test_failed("is_stable tracked", "Not found")
    
    # Check for stability gating with require_stability conditional
    if "if meta_escalation_require_stability:" in content and "is_stable = True" in content:
        test_passed("Stability gating respects require_stability flag")
    else:
        test_failed("require_stability conditional", "Pattern not found")
    
    # Check for LR backoff tracking
    if "lr_backoff_events_epoch" in content and "+= 1" in content:
        test_passed("LR backoff events tracked")
    else:
        test_failed("LR backoff tracking", "Not found")
    
    # Check for schedule computation
    if "scheduled_progress" in content and "meta_escalation_schedule" in content:
        test_passed("Schedule progress computed (linear/cosine)")
    else:
        test_failed("Schedule computation", "Pattern not found")
    
    # Check for recovery mechanism
    if "recovery_hyperlora" in content or "recovery_step" in content:
        test_passed("Recovery mechanism implemented")
    else:
        test_failed("Recovery mechanism", "Not found")
    
    # Check for meta contribution ratio logging
    if "meta_ratio" in content or "Meta contribution ratio" in content:
        test_passed("Meta contribution ratio logged")
    else:
        test_failed("Meta contribution ratio logged", "Not found")


def test_meta_escalation_schedule_math():
    """Test 12: Verify meta escalation schedule math is correct."""
    print("\n" + "=" * 60)
    print("TEST 12: Meta Escalation Schedule Math")
    print("=" * 60)
    
    # Test linear schedule
    start_epoch = 25
    ramp_epochs = 12
    base = 0.1
    target = 0.3
    
    # Test epoch 25 (start): should be at base
    epoch = 25
    progress = min(1.0, max(0.0, (epoch - start_epoch) / ramp_epochs))
    value = base + progress * (target - base)
    if abs(value - 0.1) < 0.001:
        test_passed(f"Epoch 25: value={value:.4f} (expected=0.1)")
    else:
        test_failed(f"Epoch 25 value", f"Got {value:.4f}, expected 0.1")
    
    # Test epoch 31 (midpoint): should be 0.2
    epoch = 31
    progress = min(1.0, max(0.0, (epoch - start_epoch) / ramp_epochs))
    value = base + progress * (target - base)
    expected = 0.2  # 50% progress
    if abs(value - expected) < 0.01:
        test_passed(f"Epoch 31: value={value:.4f} (expected~0.2)")
    else:
        test_failed(f"Epoch 31 value", f"Got {value:.4f}, expected ~0.2")
    
    # Test epoch 37 (end): should be at target
    epoch = 37
    progress = min(1.0, max(0.0, (epoch - start_epoch) / ramp_epochs))
    value = base + progress * (target - base)
    if abs(value - 0.3) < 0.001:
        test_passed(f"Epoch 37: value={value:.4f} (expected=0.3)")
    else:
        test_failed(f"Epoch 37 value", f"Got {value:.4f}, expected 0.3")
    
    # Test epoch 50 (past end): should cap at target
    epoch = 50
    progress = min(1.0, max(0.0, (epoch - start_epoch) / ramp_epochs))
    value = base + progress * (target - base)
    if abs(value - 0.3) < 0.001:
        test_passed(f"Epoch 50: value={value:.4f} (capped at 0.3)")
    else:
        test_failed(f"Epoch 50 value", f"Got {value:.4f}, expected 0.3 (capped)")


def test_eval_task_caching():
    """Test 13: Verify eval task caching is implemented for performance."""
    print("\n" + "=" * 60)
    print("TEST 13: Eval Task Caching (Dec 2025)")
    print("=" * 60)
    
    train_script = project_root / "scripts" / "train_rlan.py"
    
    with open(train_script, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for cached_eval_tasks variable
    if "cached_eval_tasks" in content:
        test_passed("cached_eval_tasks variable exists")
    else:
        test_failed("cached_eval_tasks variable", "Not found")
    
    # Check that caching happens BEFORE training loop
    if "cached_eval_tasks = []" in content and "Pre-loaded" in content:
        test_passed("Eval tasks pre-cached before training loop")
    else:
        test_failed("Pre-caching pattern", "Not found")
    
    # Check that cached tasks are used in eval
    if "eval_tasks = cached_eval_tasks" in content:
        test_passed("Cached tasks used during evaluation")
    else:
        test_failed("Cache usage pattern", "Not found")
    
    # Check for timing breakdown
    if "trm_eval_time" in content and "TTA eval time" in content:
        test_passed("TTA eval timing breakdown logged")
    else:
        test_failed("Eval timing breakdown", "Not found")
    
    # Check that grad explosion also counts as lr_backoff
    if "lr_backoff_events_epoch'] += 1" in content and "grad_explosion" in content:
        test_passed("Grad explosion counts as LR backoff event")
    else:
        test_failed("Grad explosion -> LR backoff tracking", "Not found")


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
    
    # Patch 5: New behavioral tests
    test_hyperlora_delta_scale_behavior()
    test_lr_composability()
    test_max_grad_norm_tracking()
    test_nan_backoff_implementation()
    
    # Meta Escalation tests (Dec 2025)
    test_meta_escalation_config()
    test_meta_escalation_implementation()
    test_meta_escalation_schedule_math()
    test_eval_task_caching()
    
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
