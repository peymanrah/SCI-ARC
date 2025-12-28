# RLAN_STABLE.YAML Config vs Codebase Audit Report

## Executive Summary

This audit compares the `configs/rlan_stable.yaml` configuration with actual implementations in:
1. **train_rlan.py** (standalone script) - ✅ Properly reads most configs from YAML
2. **SCIARCTrainer** (trainer.py) - ⚠️ Uses hardcoded defaults via TrainingConfig dataclass

---

## MODEL SECTION AUDIT

### 1. `use_hyperlora: true`
| Aspect | Status | Details |
|--------|--------|---------|
| Config Read | ✅ train_rlan.py | Line 457: `use_hyperlora=model_config.get('use_hyperlora', False)` |
| Implementation | ✅ rlan.py | Lines 348-365: Creates HyperLoRA when `config.use_hyperlora=True` |
| SCIARCTrainer | ⚠️ Indirect | SCIARCTrainer checks `hasattr(model, 'hyper_lora')` - works if model was built correctly |

**Verdict**: ✅ WORKING - Config is properly read and HyperLoRA is created.

---

### 2. `use_solver_context: true`
| Aspect | Status | Details |
|--------|--------|---------|
| Config Read | ✅ train_rlan.py | Line 462: `use_solver_context=model_config.get('use_solver_context', True)` |
| Implementation | ✅ rlan.py | Line 330: `self.use_solver_context = config.use_solver_context` |
| RecursiveSolver | ✅ | Line 500: `use_solver_context: bool = True` - creates SolverContextAttention when True |

**Verdict**: ✅ WORKING - Solver cross-attention to support set is enabled.

---

### 3. `use_cross_attention_context: true` with `spatial_downsample: 8`
| Aspect | Status | Details |
|--------|--------|---------|
| Config Read | ✅ rlan.py | Line 244: `use_cross_attn_context = config.use_cross_attention_context` |
| Config Read | ✅ rlan.py | Line 245: `spatial_downsample = config.spatial_downsample` |
| Implementation | ✅ rlan.py | Lines 267-273: Creates CrossAttentionInjector when enabled |
| ContextEncoder | ✅ | Line 265: `spatial_downsample=spatial_downsample` passed to ContextEncoder |

**Verdict**: ✅ WORKING - CrossAttention with 8x8 spatial downsample is enabled.

---

### 4. `use_lcr: false`, `use_sph: false`
| Aspect | Status | Details |
|--------|--------|---------|
| Config Read | ✅ train_rlan.py | Implicit via RLANConfig defaults |
| Implementation | ✅ rlan.py | Lines 303-319: LCR and SPH only created if flags are True |
| RLANConfig | ✅ | Line 73-74: `use_lcr: bool = False`, `use_sph: bool = False` as defaults |

**Verdict**: ✅ WORKING - LCR and SPH are disabled as expected.

---

### 5. `use_dsc: true`, `use_msre: true`
| Aspect | Status | Details |
|--------|--------|---------|
| Config Read | ✅ train_rlan.py | Line 454-455: Read from model_config |
| Implementation | ✅ rlan.py | Lines 285-300: DSC and MSRE created when True |

**Verdict**: ✅ WORKING - DSC and MSRE are enabled as expected.

---

## TRAINING SECTION AUDIT

### 6. `loo_training.enabled: true` with `loss_weight: 0.5`
| Aspect | Status | Details |
|--------|--------|---------|
| **train_rlan.py** | ✅ | Lines 2868-2882: Reads `loo_training.enabled`, `loss_weight`, `min_pairs_for_loo` |
| Implementation | ✅ | Line 2873-2881: Creates LOOTrainingLoss with correct config |
| **SCIARCTrainer** | ⚠️ HARDCODED | Line 110-111: `use_loo: bool = False`, `loo_weight: float = 0.5` in TrainingConfig |
| Gap | ❌ | **SCIARCTrainer defaults to LOO DISABLED** unless TrainingConfig is manually set |

```python
# train_rlan.py (CORRECT):
loo_config = config.get('training', {}).get('loo_training', {})
use_loo = loo_config.get('enabled', False) and config.get('model', {}).get('use_hyperlora', False)
loo_loss_fn = LOOTrainingLoss(...)

# SCIARCTrainer (HARDCODED DEFAULT):
class TrainingConfig:
    use_loo: bool = False  # <-- HARDCODED!
```

**Verdict**: 
- ✅ **train_rlan.py**: WORKING - Reads from YAML correctly
- ⚠️ **SCIARCTrainer**: USES HARDCODED DEFAULT (False) - Needs manual TrainingConfig override

---

### 7. `equivariance_training.enabled: true` with `loss_weight: 0.1`
| Aspect | Status | Details |
|--------|--------|---------|
| **train_rlan.py** | ✅ | Lines 2884-2899: Reads `equivariance_training.enabled`, `loss_weight`, `num_augmentations` |
| Implementation | ✅ | Line 2891-2897: Creates AugmentationEquivarianceLoss with correct config |
| **SCIARCTrainer** | ⚠️ HARDCODED | Line 114-116: `use_equivariance: bool = False` in TrainingConfig |
| Gap | ❌ | **SCIARCTrainer defaults to Equivariance DISABLED** |

```python
# train_rlan.py (CORRECT):
equiv_config = config.get('training', {}).get('equivariance_training', {})
use_equivariance = equiv_config.get('enabled', False) and config.get('model', {}).get('use_hyperlora', False)

# SCIARCTrainer (HARDCODED DEFAULT):
class TrainingConfig:
    use_equivariance: bool = False  # <-- HARDCODED!
```

**Verdict**:
- ✅ **train_rlan.py**: WORKING - Reads from YAML correctly
- ⚠️ **SCIARCTrainer**: USES HARDCODED DEFAULT (False)

---

### 8. `hyperlora_lr_multiplier: 10.0`
| Aspect | Status | Details |
|--------|--------|---------|
| **train_rlan.py** | ✅ | Line 542: `hyperlora_lr_mult = train_config.get('hyperlora_lr_multiplier', 10.0)` |
| Implementation | ✅ | Lines 623-630: HyperLoRA params get separate param group with 10x LR |
| **SCIARCTrainer** | ✅ FIXED | Now reads `self.config.hyperlora_lr_multiplier` from TrainingConfig |

```python
# train_rlan.py (READS CONFIG):
hyperlora_lr_mult = train_config.get('hyperlora_lr_multiplier', 10.0)

# SCIARCTrainer (NOW READS CONFIG):
hyperlora_lr_multiplier = self.config.hyperlora_lr_multiplier  # Reads from TrainingConfig
```

**Verdict**:
- ✅ **train_rlan.py**: WORKING - Reads from YAML
- ✅ **SCIARCTrainer**: FIXED - Now reads from TrainingConfig (add `hyperlora_lr_multiplier` when constructing)

---

### 9. `use_ema: false`
| Aspect | Status | Details |
|--------|--------|---------|
| **train_rlan.py** | ✅ FIXED | Line 2861: `use_ema = config.get('training', {}).get('use_ema', False)` |
| **SCIARCTrainer** | ✅ | Line 118: `use_ema: bool = False` in TrainingConfig |

```python
# train_rlan.py (FIXED):
use_ema = config.get('training', {}).get('use_ema', False)  # Now defaults to False

# Config file says:
use_ema: false  # Matches default
```

**Verdict**:
- ✅ **train_rlan.py**: FIXED - Default now matches config file
- ✅ **SCIARCTrainer**: Correctly defaults to False

---

### 10. `loss_mode: 'stablemax'`
| Aspect | Status | Details |
|--------|--------|---------|
| **train_rlan.py** | ✅ | Line 495: `loss_mode=train_config.get('loss_mode', 'focal_stablemax')` |
| Implementation | ✅ | RLANLoss Lines 1116-1146: Correctly switches loss function based on loss_mode |
| **SCIARCTrainer** | ❌ | **No loss_mode in TrainingConfig** - SCIARCTrainer doesn't configure loss_mode |

```python
# train_rlan.py (CORRECT):
loss_fn = RLANLoss(
    loss_mode=train_config.get('loss_mode', 'focal_stablemax'),
    ...
)

# SCIARCTrainer - NO loss_mode support in TrainingConfig!
# If using SCIARCTrainer, must pass RLANLoss separately with loss_mode set
```

**Verdict**:
- ✅ **train_rlan.py**: WORKING - Reads and applies loss_mode
- ❌ **SCIARCTrainer**: Does NOT support loss_mode in TrainingConfig

---

## SUMMARY TABLE

| Config Setting | train_rlan.py | SCIARCTrainer | Notes |
|----------------|---------------|---------------|-------|
| `use_hyperlora: true` | ✅ Reads YAML | ⚠️ Checks model attr | Works via model creation |
| `use_solver_context: true` | ✅ Reads YAML | ⚠️ Via model | Works via model creation |
| `use_cross_attention_context: true` | ✅ Reads YAML | ⚠️ Via model | Works via model creation |
| `spatial_downsample: 8` | ✅ Reads YAML | ⚠️ Via model | Works via model creation |
| `use_lcr: false` | ✅ Via RLANConfig | ✅ Via model | Correctly disabled |
| `use_sph: false` | ✅ Via RLANConfig | ✅ Via model | Correctly disabled |
| `use_dsc: true` | ✅ Reads YAML | ⚠️ Via model | Works via model creation |
| `use_msre: true` | ✅ Reads YAML | ⚠️ Via model | Works via model creation |
| `loo_training.enabled: true` | ✅ Reads YAML | ❌ **Defaults FALSE** | **GAP** |
| `loo_training.loss_weight: 0.5` | ✅ Reads YAML | ⚠️ Hardcoded | Works if enabled |
| `equivariance_training.enabled: true` | ✅ Reads YAML | ❌ **Defaults FALSE** | **GAP** |
| `equivariance_training.loss_weight: 0.1` | ✅ Reads YAML | ⚠️ Hardcoded | Works if enabled |
| `hyperlora_lr_multiplier: 10.0` | ✅ Reads YAML | ✅ **FIXED** | Now reads from config |
| `use_ema: false` | ✅ **FIXED** | ✅ Default FALSE | Fixed in train_rlan.py |
| `loss_mode: 'stablemax'` | ✅ Reads YAML | ❌ **Not supported** | **GAP** |

---

## FIXES APPLIED (This Audit)

### Fix 1: train_rlan.py use_ema default
**File:** `scripts/train_rlan.py` Line 2861
```python
# BEFORE:
use_ema = config.get('training', {}).get('use_ema', True)

# AFTER:
use_ema = config.get('training', {}).get('use_ema', False)
```

### Fix 2: SCIARCTrainer hyperlora_lr_multiplier
**File:** `sci_arc/training/trainer.py`

Added `hyperlora_lr_multiplier: float = 10.0` to TrainingConfig dataclass.

Updated `_create_optimizer()` to read from config:
```python
# BEFORE:
hyperlora_lr_multiplier = 10.0  # Hardcoded

# AFTER:
hyperlora_lr_multiplier = self.config.hyperlora_lr_multiplier  # From config
```

---

## REMAINING CRITICAL GAPS

### 1. **SCIARCTrainer doesn't read training config from YAML**
The `TrainingConfig` dataclass uses Python defaults, not YAML values. To use SCIARCTrainer with YAML config:

```python
# CURRENT (broken):
trainer = SCIARCTrainer(model, loader, loss_fn, TrainingConfig())
# Uses hardcoded defaults, ignores YAML

# SHOULD BE:
training_cfg = config['training']
trainer_config = TrainingConfig(
    use_loo=training_cfg.get('loo_training', {}).get('enabled', False),
    loo_weight=training_cfg.get('loo_training', {}).get('loss_weight', 0.5),
    use_equivariance=training_cfg.get('equivariance_training', {}).get('enabled', False),
    equivariance_weight=training_cfg.get('equivariance_training', {}).get('loss_weight', 0.1),
    # ... etc
)
```

### 2. ~~train_rlan.py has wrong default for use_ema~~ FIXED
See "FIXES APPLIED" section above.

### 3. **SCIARCTrainer doesn't support loss_mode**
The `TrainingConfig` dataclass has no `loss_mode` field. The loss function must be configured separately.

---

## RECOMMENDATIONS

1. **Use train_rlan.py for training** - It properly reads YAML configs
2. **Fix use_ema default** in train_rlan.py line 2861 (change True → False)
3. **If using SCIARCTrainer**, manually construct TrainingConfig with YAML values
4. **Add YAML config loading to TrainingConfig** with a `from_yaml()` class method

---

## VERIFICATION COMMANDS

```bash
# Verify HyperLoRA is created:
# Look for: "RLAN Module Config: Enabled=[..., HyperLoRA, ...]"

# Verify LOO training is enabled:
# Look for: "LOO training enabled: weight=0.5, min_pairs=2"

# Verify Equivariance is enabled:
# Look for: "Equivariance training enabled: weight=0.1, num_augs=4"

# Verify loss mode:
# Look for: "Loss Mode: STABLEMAX (pure cross-entropy)"
```
