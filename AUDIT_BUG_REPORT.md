# SCI-ARC Code Audit Bug Report

**Audit Date**: 2025-01-XX  
**Auditor**: Code Review Agent  
**Scope**: Exhaustive review of SCI-ARC codebase for bugs, theory/implementation mismatches, and engineering issues  
**Test Suite Status**: 300+ passed, 1 skipped, 4 xfailed

---

## Bug Summary

| Bug # | Title | Severity | Category | Status |
|-------|-------|----------|----------|--------|
| 1 | LOO Training Module Not Integrated | HIGH | Theory/Implementation Mismatch | ✅ FIXED |
| 2 | `z_struct`/`z_content` Not Returned by RLAN | HIGH | Dead Code / Theory Mismatch | ✅ FIXED |
| 3 | `encode_structure_only` Method Missing | HIGH | Runtime Error (Latent) | ✅ FIXED |
| 4 | EMA Not Integrated into Training | MEDIUM | Missing Feature | ✅ FIXED |
| 5 | HyperLoRATrainer Exists But Not Used | MEDIUM | Dead Code | ⚠️ Deferred |
| 6 | AugmentationEquivarianceLoss Defined But Not Used | MEDIUM | Meta-Learning Gap | ✅ FIXED |
| 7 | CombinedMetaLoss Defined But Not Used | MEDIUM | Dead Code | ⚠️ Deferred |
| 8 | SCIARCDataset Ignores color_permutation_prob Config | HIGH | Config Bug | ✅ FIXED |
| 9 | SCIARCDataset Ignores translational_augment Config | HIGH | Config Bug | ✅ FIXED |

---

## Fixed Bugs Summary

### ✅ Bug #1: LOO Training - FIXED
**Changes Made**:
- Added `use_loo`, `loo_weight`, `loo_min_pairs` fields to `TrainingConfig`
- Imported `LOOTrainingLoss` and `LOOConfig` in trainer.py
- Added LOO loss initialization in `SCIARCTrainer.__init__`
- Integrated LOO loss computation in `_compute_losses()` when enabled

### ✅ Bug #2: z_struct/z_content - FIXED
**Changes Made**:
- Added `structure_projector` and `content_projector` nn.Sequential modules to RLAN
- Added z_struct, z_struct_demos, z_content computation in `forward()` return_intermediates block
- Structure latent captures transformation rules, content latent captures grid appearance

### ✅ Bug #3: encode_structure_only - FIXED
**Changes Made**:
- Added `encode_structure_only()` method to RLAN model
- Method encodes structure without running full solver for CISL content invariance

### ✅ Bug #4: EMA Integration - FIXED
**Changes Made**:
- Added `use_ema`, `ema_decay` fields to `TrainingConfig`
- Imported `EMAHelper` in trainer.py
- Added EMA initialization in `SCIARCTrainer.__init__`
- Added EMA update after optimizer step
- Added EMA swap in/out in `validate()` method

---

## Remaining Issues (Deferred)  
**Location**: [sci_arc/training/trainer.py#L669-L672](sci_arc/training/trainer.py#L669-L672) calls `self.model.encode_structure_only()` but this method doesn't exist on RLAN

**Symptom**: If CISL is enabled (`use_cicl=True`) AND `cicl_color_inv_weight > 0` (default is 0.5), the training loop will crash with `AttributeError: 'RLAN' object has no attribute 'encode_structure_only'`.

**Evidence**:
```python
# Trainer code (lines 669-672):
with torch.no_grad():
    z_struct_content_aug = self.model.encode_structure_only(  # <-- CRASH
        input_grids=input_grids_perm,
        output_grids=output_grids_perm,
    ).detach()

# RLAN methods:
.\.venv\Scripts\python.exe -c "from sci_arc.models import RLAN; print('encode_structure_only' in dir(RLAN))"
# Output: False
```

**Root cause hypothesis**: The CISL (Content-Invariant Structure Learning) feature was partially implemented in the trainer but the corresponding method was never added to RLAN.

**Minimal reproduction**:
```powershell
# This will crash at first training step when use_cicl=True:
.\.venv\Scripts\python.exe -c "from sci_arc.models import RLAN, RLANConfig; m = RLAN(config=RLANConfig(hidden_dim=32)); m.encode_structure_only(None, None)"
# AttributeError: 'RLAN' object has no attribute 'encode_structure_only'
```

**Suggested fix**:
1. Implement `encode_structure_only()` method on RLAN that returns structure embedding without running the full solver
2. OR set `cicl_color_inv_weight=0.0` as default until the method is implemented
3. OR gate the code path more defensively with `hasattr(self.model, 'encode_structure_only')`

---

### Bug #4: EMA Not Integrated into Training

**Severity**: MEDIUM  
**Category**: Missing Feature  
**Location**: [sci_arc/training/ema.py](sci_arc/training/ema.py) exists but is never imported/used by [sci_arc/training/trainer.py](sci_arc/training/trainer.py)

**Symptom**: Exponential Moving Average (EMA) of model weights is a standard technique for stable evaluation (used by TRM), but it's not used in SCI-ARC training despite having a complete implementation.

**Evidence**:
```python
# grep for EMA/ema in trainer.py:
# Only matches are for "remaining" (epochs remaining, ETA)
# EMAHelper is never imported or instantiated
```

**Root cause hypothesis**: EMA helper was ported from TRM but integration into the training loop was never completed.

**Minimal reproduction**:
```powershell
.\.venv\Scripts\python.exe -c "import re; text = open('sci_arc/training/trainer.py', encoding='utf-8').read(); print('EMAHelper in trainer:', 'EMAHelper' in text)"
# Output: EMAHelper in trainer: False
```

**Suggested fix**:
1. Add `use_ema` flag to `TrainingConfig` (default False for backward compatibility)
2. Initialize `EMAHelper` in `SCIARCTrainer.__init__` when enabled
3. Call `ema.update()` after each optimizer step
4. Use EMA weights for validation when enabled
5. Save/load EMA state in checkpoints

---

### Bug #5: HyperLoRATrainer Exists But Not Used (DEFERRED)

**Severity**: MEDIUM  
**Category**: Dead Code  
**Status**: ⚠️ Deferred - Not blocking, LOO training now integrated via LOOTrainingLoss  
**Location**: [sci_arc/training/hyperlora_training.py](sci_arc/training/hyperlora_training.py) defines `HyperLoRATrainer` class but it's never imported/used

**Symptom**: `HyperLoRATrainer` provides specialized LOO-based training for HyperLoRA but is never called. The main `SCIARCTrainer` has HyperLoRA parameter groups with 10x learning rate but doesn't use the specialized trainer.

**Note**: With Bug #1 fixed, LOO training is now integrated directly via `LOOTrainingLoss`. This specialized trainer may be redundant or could serve as an alternative training approach.

**Suggested resolution**:
1. Remove if redundant with new LOO integration
2. OR keep as alternative specialized trainer for HyperLoRA-only fine-tuning

---

### ✅ Bug #6: AugmentationEquivarianceLoss - FIXED

**Severity**: MEDIUM  
**Category**: Meta-Learning Gap  
**Status**: ✅ FIXED  
**Location**: [scripts/train_rlan.py](scripts/train_rlan.py)

**Changes Made**:
- Added `AugmentationEquivarianceLoss` and `EquivarianceConfig` imports to `train_rlan.py`
- Initialized `equiv_loss_fn` when `equivariance_training.enabled=True` in config
- Added equivariance loss computation in training loop after LOO loss
- Added `HyperLoRA.compute_delta_w(context)` method for computing deltas from pre-pooled context
- Added `equiv_loss` tracking in epoch diagnostics
- Created `tests/test_equivariance_loss_integration.py` with 5 passing tests

**Why This Matters for Meta-Learning**:
The equivariance loss ensures that HyperLoRA predicts similar LoRA weights for rotated/flipped versions of the same task. This is critical because ARC tasks are equivariant under the D4 dihedral group - a "fill the corner" task is the same transformation regardless of orientation. Without this loss, HyperLoRA may waste capacity learning separate weights for each orientation.

---

### ✅ Bug #8: SCIARCDataset color_permutation_prob Ignored - FIXED

**Severity**: HIGH  
**Category**: Config Bug  
**Status**: ✅ FIXED  
**Location**: [sci_arc/data/dataset.py#L1151](sci_arc/data/dataset.py#L1151)

**Changes Made**:
- Changed `do_color_perm = random.random() < 0.5` to `do_color_perm = self.color_permutation and random.random() < self.color_permutation_prob`
- Users can now control color permutation probability via config

---

### ✅ Bug #9: SCIARCDataset translational_augment Ignored - FIXED

**Severity**: HIGH  
**Category**: Config Bug  
**Status**: ✅ FIXED  
**Location**: [sci_arc/data/dataset.py#L1163](sci_arc/data/dataset.py#L1163)

**Changes Made**:
- Changed `do_translate = random.random() < 0.3` to `do_translate = self.translational_augment and random.random() < 0.3`
- Users can now disable translational augmentation via config

---

### Bug #7: CombinedMetaLoss Defined But Not Used (DEFERRED)

**Severity**: MEDIUM  
**Category**: Dead Code  
**Status**: ⚠️ Deferred - Wrapper class, components now used directly  
**Location**: [sci_arc/models/rlan_modules/loo_training.py](sci_arc/models/rlan_modules/loo_training.py) defines `CombinedMetaLoss` but it's never imported/used

**Symptom**: `CombinedMetaLoss` orchestrates LOO loss + equivariance loss but is never instantiated in the training pipeline.

**Note**: With Bug #1 fixed, `LOOTrainingLoss` is now used directly. `CombinedMetaLoss` can be integrated later if augmentation equivariance is desired.

**Root cause hypothesis**: Same as Bug #1 - meta-learning losses were developed but not integrated.

**Suggested fix**: Integrate with Bug #1 fix - create `use_meta_learning` config flag.

---

## Non-Bug Observations

### Observation 1: Checkpoint Compatibility
The checkpoint format is straightforward and includes all necessary state (model, optimizer, scheduler, scaler, epoch, global_step, best metrics). This is correct.

### Observation 2: Train/Eval Mode Handling
`train_epoch()` always calls `self.model.train()` at start. `validate()` calls `self.model.eval()`. This is correct.

### Observation 3: Gradient Clipping
Gradient clipping is applied correctly after unscaling (for AMP) and before optimizer step.

### Observation 4: Loss Ignore Index
The `-100` ignore index is properly used for padding in cross-entropy losses. This was fixed in a previous session.

### Observation 5: Forward Training Exists
The `forward_training()` wrapper on RLAN exists and works correctly. This was fixed in a previous session.

---

## Recommendations

1. **Priority 1**: Fix Bug #3 (encode_structure_only) - this is a latent crash that will occur if CISL is enabled
2. **Priority 2**: Fix Bug #2 (z_struct) - either implement structure embeddings or remove dead loss code
3. **Priority 3**: Fix Bug #1 (LOO training) - this is core to the claimed theory

The other bugs (4-7) are lower priority dead code issues that don't cause crashes but represent incomplete features.

---

## Test Verification

All existing tests pass (225 passed, 2 xfailed for known issues). The bugs identified here are:
- Latent crashes (Bug #3 only triggers with specific config)
- Dead code paths that never execute
- Missing features that are documented but not implemented

To create tests for these bugs, see the suggested reproduction commands above.
