# SCI-ARC Bug Fixes Summary

This document summarizes all fixes applied during this debugging session.

## Issues Found and Fixed

### 1. GradScaler Double-Unscale Bug
**File**: `scripts/train_rlan.py`
**Symptom**: RuntimeError in epoch 2: "unscale_() has already been called once"
**Cause**: When NaN gradients detected, code called `continue` without calling `scaler.update()`
**Fix**: Added `scaler.update()` before `continue` when skipping optimizer step

```python
# BEFORE:
if has_nan_grad:
    continue  # Bug: scaler not updated

# AFTER:
if has_nan_grad:
    scaler.update()  # Reset scaler state
    continue
```

### 2. BG Accuracy 1% (WeightedStablemaxLoss Bug)
**File**: `sci_arc/training/rlan_loss.py` (lines ~265-285)
**Symptom**: BG accuracy 1%, model under-predicting background
**Cause**: Post-cap normalization crushed BG weight (0.41 vs expected 2.0)

**Old Algorithm**:
1. raw_weights = 1/freq
2. Apply caps (BG=2.0, FG=5.0)
3. Normalize by sum → BG gets crushed!

**New Algorithm (BALANCED GRADIENT APPROACH)**:
1. BG weight = bg_weight_cap (fixed at 2.0)
2. FG weights = inverse-freq rescaled to [1.0, fg_weight_cap]
3. No post-normalization

**Result**: BG gradient share now ~52% (matches pixel proportion)

### 3. Solver Degradation (Step 0 Best, Later Steps Worse)
**File**: `sci_arc/models/rlan_modules/recursive_solver.py` (lines ~594-596)
**Symptom**: Step 0 had lowest loss, later steps degraded by 0.5%
**Cause**: Residual connection too weak (0.1) couldn't prevent hidden state drift

**Fix**: Increased residual strength from 0.1 to 0.3

```python
# BEFORE:
h_new = 0.9 * h_new + 0.1 * h_initial

# AFTER:
h_new = 0.7 * h_new + 0.3 * h_initial
```

**Result**: Loss now decreases across steps (verified with test script)

### 4. FG/BG Accuracy Not Logged
**File**: `scripts/train_rlan.py`
**Symptom**: No visibility into BG vs FG accuracy per batch
**Fix**: Added:
- `batch_fg_acc`, `batch_bg_acc` variables
- Running windows (last 50 batches) for FG/BG accuracy
- FG/BG accuracy display in batch logs

### 5. Augmentation 0% (Cache Missing aug_info)
**File**: `configs/rlan_stable.yaml`
**Symptom**: "Color Permutation: 0%, Translational: 0%" in epoch logs
**Cause**: Cache built with `track_augmentation: false`, samples lack `aug_info` field
**Fix**: 
1. Changed config to `track_augmentation: true`
2. Cache must be deleted and regenerated

## Files Modified

### sci_arc/training/rlan_loss.py
- WeightedStablemaxLoss: New balanced gradient weight calculation
- **NEW**: FocalWeightedStablemaxLoss class (combines our weights + focal modulation)
- Added `focal_weighted` loss mode to RLANLoss

### sci_arc/models/rlan_modules/recursive_solver.py
- RecursiveSolver: Stronger residual connection (0.3 instead of 0.1)

### scripts/train_rlan.py
- GradScaler: Added `scaler.update()` before continue on NaN
- Added FG/BG accuracy tracking and logging

### configs/rlan_stable.yaml
- Changed `track_augmentation: false` to `track_augmentation: true`

### configs/rlan_fair.yaml
- Updated loss_mode to `focal_weighted` (RECOMMENDED)
- Added `bg_weight_cap: 2.0` and `fg_weight_cap: 5.0` config options

## Loss Function Exploration

After investigating alternatives to WeightedStablemaxLoss, we found:

| Approach | Result |
|----------|--------|
| Pure Learnable Weights | ❌ Collapse to near-zero (model "cheats") |
| Constrained Learnable | ❌ Converge to lower bound |
| Class-Balanced (Cui et al. 2019) | ⚠️ Downweights BG (opposite of what we want) |
| Temperature Blending | ✓ Learned t=0.989 → confirms freq-based is optimal |
| **FocalWeightedStablemaxLoss** | ✅ **RECOMMENDED** - keeps BG/FG balance + dynamic focus |

### FocalWeightedStablemaxLoss (NEW - RECOMMENDED)

Combines:
1. **Our weight philosophy**: BG=cap, FG scaled 1.0-cap (maintains BG/FG gradient balance)
2. **Focal modulation**: `(1-p_t)^gamma` per pixel (dynamic focus on hard pixels)
3. **Stablemax**: Numerically stable alternative to softmax

Benefits:
- Same BG/FG gradient balance as WeightedStablemaxLoss
- PLUS: Easy pixels automatically down-weighted as training progresses
- Early training: All pixels hard → focal≈1.0 → all get attention
- Later training: Easy pixels → focal→0 → focus on hard pixels
- No collapse risk (weights computed from data, not learnable)

Usage in config:
```yaml
training:
  loss_mode: 'focal_weighted'  # RECOMMENDED
  focal_gamma: 2.0  # Focusing parameter
  bg_weight_cap: 2.0  # BG weight
  fg_weight_cap: 5.0  # Max FG weight
```

## New Test Scripts Created

1. `scripts/test_bg_accuracy_issue.py` - BG weight investigation
2. `scripts/test_solver_degradation.py` - Solver step analysis
3. `scripts/test_augmentation.py` - Augmentation verification

## Deployment Instructions

### For Production Training Server:

1. **Copy modified files**:
   ```
   sci_arc/training/rlan_loss.py
   sci_arc/models/rlan_modules/recursive_solver.py
   scripts/train_rlan.py
   configs/rlan_stable.yaml
   ```

2. **Delete old cache** (CRITICAL):
   ```bash
   rm ./cache/rlan_stable_400k.pkl
   ```

3. **Restart training**:
   - Cache will automatically rebuild with aug_info tracking
   - Monitor first epoch to verify:
     - BG accuracy should be > 30% (not 1%)
     - Color Permutation should be ~30%
     - Translational should be > 90%
     - Per-step loss should decrease (not increase)

4. **Expected improvements**:
   - BG accuracy: 1% → 30-50%
   - Solver: Step 6 loss < Step 0 loss
   - Augmentation: 0% → 30% color perm, 100% translational
   - No more GradScaler crashes

## Verification Commands

```bash
# Test WeightedStablemaxLoss fix
python -c "
import torch
from sci_arc.training.rlan_loss import WeightedStablemaxLoss
loss_fn = WeightedStablemaxLoss(bg_weight_cap=2.0, fg_weight_cap=5.0)
# Should see BG weight = 2.0, not crushed
"

# Test solver fix
python scripts/test_solver_degradation.py
# Should see 'HEALTHY: Loss decreases from step 0 to step 6'

# Test augmentation
python scripts/test_augmentation.py
# Should see all augmentations working correctly
```
