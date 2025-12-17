# SCI-ARC Training Issue Analysis - Epoch 235 Diagnosis

## ✅ FIXES IMPLEMENTED

The following fixes have been applied to address the Epoch 235 training collapse:

### Fix 1: Reduced Color Permutation Rate (IMPLEMENTED)
- **File**: `sci_arc/data/dataset.py`
- **Change**: Added `color_permutation_prob` parameter (default 0.3 = 30%)
- **Config**: `configs/rlan_core_ablation.yaml` now has `color_permutation_prob: 0.3`

### Fix 2: Rebalanced BG/FG Weights (IMPLEMENTED)
- **File**: `sci_arc/training/rlan_loss.py`
- **Change**: Added configurable `bg_weight_cap` and `fg_weight_cap` parameters
- **New defaults**: `bg_weight_cap=2.0`, `fg_weight_cap=5.0` (was 1.0/10.0)
- **Config**: Updated in `configs/rlan_core_ablation.yaml`

### Fix 3: Neutral BG Bias (IMPLEMENTED)
- **File**: `sci_arc/models/rlan_modules/recursive_solver.py`
- **Change**: `bg_bias` changed from `-0.5` to `0.0`
- **Rationale**: Neutral initialization lets weighted loss guide learning

### Fix 4: Stop Logits Diagnostics (IMPLEMENTED)
- **File**: `scripts/train_rlan.py`
- **Change**: Added stop_logits mean/std/min/max tracking
- **Warning**: Prints CRITICAL warning if |mean| > 5.0 (sigmoid saturation)

---

## Critical Issues Identified (Original Analysis)

### 1. Color Mode Collapse to Color 6 (83.8%)

**Observation**: 
```
Pred %: [0.4, 1.3, 4.1, 1.0, 5.8, 0.6, 83.8, 1.4, 1.1, 0.5]
Target %: [93.0, 0.9, 1.4, 0.5, 1.1, 0.5, 0.5, 0.8, 0.5, 0.7]
```

**Root Cause**: 
- The model found a local minimum where predicting color 6 everywhere minimizes loss
- With 10x FG weighting and bg_bias=-0.5, the model learned to avoid BG entirely
- Color 6 may have coincidentally aligned with common FG colors in augmented data
- **100% color permutation** means every sample has randomly shuffled colors, making color identity meaningless!

**Why 100% Color Permutation Breaks Learning**:
- Color permutation is applied 100% of the time (see config: `color_permutation: true`)
- This means the model NEVER sees consistent color patterns
- If color 5 represents "the object" in task A, it might be color 2 in the augmented version
- The model cannot learn "color 5 means X" because color identities are random
- Only spatial/shape patterns remain stable

**Solution**: Reduce color permutation rate to 30-50%

### 2. Background Collapse (0.4% vs 93% target)

**Observation**:
- Model predicts almost no background
- FG Coverage is 1428% (14x over-prediction)

**Root Cause**:
- `bg_bias = -0.5` combined with `fg_weight_cap = 10.0` creates 10x+ incentive for FG
- Initial P(bg) ≈ 6% was already biased against BG
- The weighted loss amplifies any FG error by 10x, but BG error only by 1x
- Model learned: "better to over-predict FG than under-predict it"

**Mathematical Analysis**:
```
Loss for FG pixel predicted as BG: ~10x base loss
Loss for BG pixel predicted as FG: ~1x base loss

Over 93% BG data:
- If model predicts all FG: BG loss = 0.93 * 1x = 0.93
- If model predicts all BG: FG loss = 0.07 * 10x = 0.70

Model optimizes toward predicting more FG!
```

**Solutions**:
1. Increase `bg_weight_cap` from 1.0 to 2.0-3.0
2. Reduce `fg_weight_cap` from 10.0 to 5.0
3. Change `bg_bias` from -0.5 to 0.0 or +0.5

### 3. Stop Predictor Gradient = 0.0000

**Observation**: StopPred Grad = 0.0000

**Root Cause Analysis**:

The gradient should flow:
```
task_loss → logits → aggregated → clue_usage → stop_probs → stop_logits → stop_predictor
```

The issue is in `_aggregate_clues()`:
```python
expected_clues = clue_usage.sum(dim=-1, keepdim=True).detach()  # DETACHED!
aggregated = aggregated / (expected_clues.squeeze(1) + 1e-6)
```

The divisor is detached, but gradient should still flow through the numerator:
```python
weighted = clue_features * clue_usage  # clue_usage has gradient
aggregated = weighted.sum(dim=1)  # sum preserves gradient
```

**Why Gradient Might Be Zero**:

1. **stop_probs ≈ 0 or ≈ 1**: If sigmoid(stop_logits) is saturated:
   - `d(sigmoid)/d(x) ≈ 0` when sigmoid is near 0 or 1
   - Gradient vanishes through sigmoid

2. **clue_features dominate**: If clue_features have much larger magnitude than clue_usage:
   - Gradient flows more to clue_features
   - stop_predictor gets negligible gradient

3. **Task loss itself has low gradient**: If the model reached a local minimum:
   - Overall gradient is small
   - stop_predictor gradient is proportionally small

**Verification Needed**: Check stop_logits values
- If all ≈ -10 (sigmoid ≈ 0): clue_usage ≈ 1, but gradient vanishes
- If all ≈ +10 (sigmoid ≈ 1): clue_usage ≈ 0, gradient vanishes

**Solutions**:
1. Add stop_logits to loss diagnostics
2. Initialize stop_logits closer to 0 (sigmoid = 0.5)
3. Add gradient scaling for stop_predictor
4. Consider detaching clue_features in _aggregate_clues for gradient flow control

### 4. Low Clue Count Variance (std = 0.08)

**Observation**: All samples use nearly identical number of clues

**Root Cause**:
- Stop predictor is not learning (grad = 0)
- Without learning signal, stop_logits stay at initialization
- All samples produce same stop_probs

**Connection to Issue #3**: This is a symptom, not a root cause

### 5. Solver Degradation (Step 0 Best)

**Observation**: Later solver steps produce worse predictions than step 0

**Root Cause**:
- The 0.9/0.1 residual connection helps but doesn't fully fix this
- The GRU hidden state may be carrying corrupted information
- Deep supervision at 0.5 weight may not be enough

**Note**: This is secondary to the color collapse issue

---

## Recommended Fixes (Priority Order) - ALL IMPLEMENTED ✅

### Fix 1: Reduce Color Permutation Rate ✅ DONE

**File**: `sci_arc/data/dataset.py`, `configs/rlan_core_ablation.yaml`

```yaml
augmentation:
  enabled: true
  rotation: true
  flip: true
  transpose: true
  color_permutation: true
  color_permutation_prob: 0.3  # ← NEW: Only 30% to preserve color identity!
  translational: true
```

### Fix 2: Rebalance BG/FG Weights ✅ DONE

**File**: `sci_arc/training/rlan_loss.py`, `configs/rlan_core_ablation.yaml`

```yaml
# In loss configuration:
bg_weight_cap: 2.0    # Increased from 1.0
fg_weight_cap: 5.0    # Reduced from 10.0
```

This makes the ratio 5:2 instead of 10:1, reducing incentive to over-predict FG.

### Fix 3: Fix BG Bias ✅ DONE

**File**: `sci_arc/models/rlan_modules/recursive_solver.py`

Changed from:
```python
final_layer.bias[0] = -0.5  # OLD: Mild negative for background
```
To:
```python
final_layer.bias[0] = 0.0  # NEW: Neutral - let weighted loss guide learning
```

### Fix 4: Stop Logits Diagnostics ✅ DONE

**File**: `scripts/train_rlan.py`

Now logs:
```
Stop Logits: mean=X.XX, std=X.XX, range=[X.X, X.X]
[CRITICAL] Stop logits saturated! |mean|=X.X > 5.0
```

If stop_logits are saturated (|mean| > 5), the gradient will be near zero.

### Fix 5: Gradient Flow Enhancement (PENDING - if needed)

Consider adding a small auxiliary loss on stop_logits to ensure gradient flows:
```python
# Encourage stop_logits to stay in learnable range
stop_logit_reg = 0.001 * (stop_logits ** 2).mean()  # L2 regularization
```

This prevents sigmoid saturation. **Only implement if stop_logits saturation persists after other fixes.**

---

## Implementation Status

| Fix | Status | Files Modified |
|-----|--------|----------------|
| 1. Color perm prob 30% | ✅ DONE | dataset.py, train_rlan.py, config |
| 2. BG/FG weight caps | ✅ DONE | rlan_loss.py, train_rlan.py, config |
| 3. Neutral bg_bias | ✅ DONE | recursive_solver.py |
| 4. Stop logits diagnostics | ✅ DONE | train_rlan.py |
| 5. Stop logits L2 reg | ⏳ PENDING | (only if saturation persists) |

---

## Verification Metrics

After applying fixes, watch for:

1. **Pred % distribution** should be closer to Target %
2. **BG prediction** should increase from 0.4% toward 93%
3. **FG Coverage** should drop from 1428% toward 100%
4. **Stop Predictor Grad** should be > 0.001
5. **Clue Count std** should increase from 0.08
6. **Stop Logits mean** should stay between -3 and +3

---

## Mathematical Note: Loss Weight Balance

For ARC data with ~87% BG:
```
Optimal weight ratio = FG_freq / BG_freq = 0.13 / 0.87 ≈ 0.15
Inverse ratio for equalization = 6.7x more weight on FG
```

With caps:
- `fg_weight_cap = 5.0`, `bg_weight_cap = 2.0`
- Effective ratio: 5:2 = 2.5x
- This is milder than the mathematical optimum, which should help prevent collapse

The key insight: Even if FG is rare, we shouldn't weight it SO heavily that the model abandons BG entirely.
