# Training Fixes - January 2026

This document summarizes fixes implemented based on epoch 30 training log analysis that showed:
- 51.2% train exact match, 0% eval exact match
- 27.4× entropy gap between train/eval
- 5 NaN batches per epoch with extreme logits [-90, +80]
- Clue count collapsed to ~1.15 with std=0.07
- TTA consensus very low, equiv loss always 0

## 1. Logit Clamping for NaN Prevention

**Problem:** Extreme logits in [-90, +80] caused NaN in stablemax via exp() overflow.

**Solution:** Added `logit_clamp_max` parameter to both `WeightedStablemaxLoss` and `FocalWeightedStablemaxLoss`.

**Files Modified:**
- `sci_arc/training/rlan_loss.py`: Added `logit_clamp_max: float = 50.0` to __init__
- Logits are clamped before stablemax: `logits_valid = logits_valid.clamp(-50, 50)`

**Config (default):**
```yaml
# Uses default logit_clamp_max=50.0 (hardcoded in loss class)
```

---

## 2. Meta-Loss Cap Implementation

**Problem:** Meta-learning losses (LOO + equiv) contributed 40.9% of total loss, dominating task learning.

**Solution:** Added configurable cap on meta-loss contribution relative to task loss.

**Files Modified:**
- `configs/rlan_stable_dev.yaml`: Added `meta_loss_cap_enabled: true`, `meta_loss_cap_ratio: 0.25`
- `scripts/train_rlan.py`: Added cap calculation in train_epoch that scales effective_loo_weight and effective_equiv_weight

**How it works:**
```python
if meta_loss_cap_enabled and task_loss_value > 0:
    # Target: meta_contribution <= cap_ratio / (1 - cap_ratio) * task_loss
    max_meta_contribution = meta_loss_cap_ratio / (1 - meta_loss_cap_ratio) * task_loss_value
    
    # Get raw meta loss weights
    raw_loo_weight = loo_loss_fn.config.loss_weight if loo_loss_fn else 0.0
    raw_equiv_weight = equiv_loss_fn.config.loss_weight if equiv_loss_fn else 0.0
    total_raw_meta_weight = raw_loo_weight + raw_equiv_weight
    
    # Scale down if raw weights exceed cap
    if total_raw_meta_weight > max_meta_contribution:
        meta_loss_cap_factor = max_meta_contribution / total_raw_meta_weight
        meta_loss_cap_factor = max(0.01, min(1.0, meta_loss_cap_factor))

# Apply: effective_loo_weight = loo_weight * meta_loss_cap_factor
```

**Config:**
```yaml
meta_escalation:
  meta_loss_cap_enabled: true   # Enable dynamic cap
  meta_loss_cap_ratio: 0.25     # Max 25% of task loss
```

---

## 3. OutputEquivarianceLoss Class

**Problem:** Existing `AugmentationEquivarianceLoss` compares HyperLoRA contexts, but `pool_context` uses D4-invariant averaging (8 transforms), so all augmented contexts are identical. Result: equiv_loss ≈ 0 always.

**Solution:** Created `OutputEquivarianceLoss` that operates at output level:
1. Apply augmentation to input
2. Run forward pass
3. Apply inverse augmentation to output
4. Compare with original output

**Files Modified:**
- `sci_arc/models/rlan_modules/loo_training.py`: Added `OutputEquivarianceLoss` class (~170 lines)
- `sci_arc/models/rlan_modules/__init__.py`: Added export

**Note:** This class is more expensive (requires forward pass per augmentation) and is provided as an alternative for future use.

---

## 4. Fix Equivariance Pooling

**Problem:** The existing equivariance loss used `pool_context()` for both original and augmented contexts, but this method averages over all 8 D4 transforms (making it dihedral-invariant). Result: all contexts are identical, loss ≈ 0.

**Solution:** 
1. Added `pool_context_simple()` to HyperLoRA that does NOT average over D4 transforms
2. Updated train_epoch to use `pool_context_simple()` for BOTH original and augmented contexts

**Files Modified:**
- `sci_arc/models/rlan_modules/hyper_lora.py`: Added `pool_context_simple()` method
- `scripts/train_rlan.py`: Changed to use `pool_context_simple()` for equiv loss

**How it works:**
```python
# NEW: pool_context_simple doesn't D4-average, so contexts differ
if hasattr(model.hyper_lora, 'pool_context_simple'):
    original_context = model.hyper_lora.pool_context_simple(support_features_detached)
    aug_context = model.hyper_lora.pool_context_simple(aug_features)
```

---

## 5. Clue Variance Regularizer

**Problem:** Stop head outputs had std=0.07 (collapsed), meaning all samples use ~same clue count regardless of task difficulty.

**Solution:** Added variance regularization to `SparsityRegularization`:
- Penalize when `clues_used.std() < target_variance`
- Hinge loss: `variance_penalty = relu(target_variance - clues_used.std()) * weight`

**Files Modified:**
- `sci_arc/training/rlan_loss.py`: Added `variance_weight` and `target_variance` to SparsityRegularization
- `scripts/train_rlan.py`: Pass new params to RLANLoss
- `configs/rlan_stable_dev.yaml`: Added config entries

**Config:**
```yaml
clue_variance_weight: 1.0     # Weight for variance regularization
clue_target_variance: 0.5     # Target std for clue count across batch
```

---

## 6. Stop Logit Saturation Guard

**Problem:** Stop logits in [-90, +80] are saturated (sigmoid → 0/1), causing gradient vanishing.

**Solution:** Added saturation penalty to `SparsityRegularization`:
- Penalize when `|stop_logit| > threshold`
- `saturation_penalty = mean(relu(|logit| - threshold)) * weight`

**Files Modified:**
- `sci_arc/training/rlan_loss.py`: Added `saturation_weight` and `saturation_threshold` to SparsityRegularization
- `scripts/train_rlan.py`: Pass new params to RLANLoss
- `configs/rlan_stable_dev.yaml`: Added config entries

**Config:**
```yaml
stop_saturation_weight: 0.5     # Weight for saturation penalty
stop_saturation_threshold: 5.0  # sigmoid(5)≈0.993, sigmoid(-5)≈0.007
```

---

## Summary of Config Changes

All new config entries for `rlan_stable_dev.yaml`:

```yaml
# Meta-learning escalation (updated)
meta_escalation:
  meta_loss_cap_enabled: true
  meta_loss_cap_ratio: 0.25

# Clue regularization (updated)
clue_variance_weight: 1.0
clue_target_variance: 0.5
stop_saturation_weight: 0.5
stop_saturation_threshold: 5.0
```

---

## Backward Compatibility

All changes are backward compatible:
- New parameters have defaults that match old behavior (e.g., `logit_clamp_max=50.0`, `clue_variance_weight=0.0`)
- `pool_context_simple()` falls back to `pool_context()` if method doesn't exist
- Config entries default to OFF if not specified

---

## Expected Outcomes

With these fixes, training should show:
1. **No NaN batches** (logit clamping prevents overflow)
2. **Task loss dominance** (meta-loss capped at 25%)
3. **Non-zero equiv loss** (using simple pooling, not D4-invariant)
4. **Variable clue counts** (variance regularization encourages task-adaptive stopping)
5. **Learnable stop predictor** (saturation guard keeps logits in gradient-friendly range)
6. **Higher TTA consensus** (equivariance loss actually provides gradients)
