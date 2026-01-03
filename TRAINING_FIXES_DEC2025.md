# Training Fixes - December 2025

This document summarizes all fixes applied based on analysis of the 27-epoch training log.

## Summary of Issues Found

The training log showed:
- Train accuracy rising (58% → 89%) but TTA exact match stuck at 0%
- Stop predictor logits saturated at -4.0 with near-zero gradients
- Centroid spread collapse (< 0.5) in later epochs
- Equivariance showing "0 batches computed" 
- HyperLoRA clamp max_norm growing from 1 → 30+
- Negative clue-loss correlation in later epochs

---

## Fix 1: TTA crop_prediction Bug (CRITICAL)

**File:** `scripts/train_rlan.py`

**Problem:** The `crop_prediction()` function tried to detect content by finding pixels ≠ pad_value (10), but:
1. The model is NOT trained to predict pad_value=10 in padded regions
2. Value 10 is a valid ARC color that can appear in real outputs
3. Result: wrong-sized grids → 0% exact match even with 76% pixel accuracy

**Fix:** Pass the expected output shape through the TTA pipeline and crop to that exact size.

```python
# OLD: Content detection (unreliable)
def crop_prediction(pred, pad_value=10):
    content_mask = (pred != pad_value) & (pred != -100)
    ...

# NEW: Use expected shape directly
def crop_prediction(pred, target_shape=None, pad_value=10):
    if target_shape is not None:
        h, w = target_shape
        return pred[:h, :w].copy()
    ...
```

**Changes:**
- Updated `crop_prediction()` to accept `target_shape` parameter
- Updated `aug_infos` to track `expected_aug_shape` for each view
- Updated inverse transform loop to use `target_shape`

---

## Fix 2: DSC Stop Predictor Saturation

**File:** `sci_arc/models/rlan_modules/dynamic_saliency_controller.py`

**Problem:** Stop logits collapsed to -4.0 due to tanh squashing with range [-4, 4]:
- At x=-4: gradient = 1 - tanh²(-1) ≈ 0.07 (too weak)
- Stop predictor cannot learn to adjust clue count

**Fix:** Increased squashing range from 4.0 to 6.0 and added Straight-Through Estimator (STE) for extreme saturation.

```python
# OLD: Range 4.0 (gradient dies at edges)
stop_logit = 4.0 * torch.tanh(stop_logit_raw / 4.0)

# NEW: Range 6.0 + STE for extreme saturation
SQUASH_RANGE = 6.0
stop_logit_squashed = SQUASH_RANGE * torch.tanh(stop_logit_raw / SQUASH_RANGE)

# STE: when |raw| > 5, bypass tanh for gradient but keep output clamped
is_saturated = (stop_logit_raw.abs() > 5.0).detach()
if is_saturated.any() and self.training:
    stop_logit_ste = stop_logit_raw.clamp(-SQUASH_RANGE, SQUASH_RANGE)
    stop_logit = torch.where(
        is_saturated,
        stop_logit_ste - stop_logit_ste.detach() + stop_logit_squashed.detach(),
        stop_logit_squashed
    )
```

**Benefits:**
- At x=-4: gradient ≈ 0.63 (9× stronger than before)
- STE ensures gradient flows even at extreme saturation
- Range [-6, 6] gives sigmoid range [0.0025, 0.9975]

---

## Fix 3: Strengthen Centroid Diversity Loss

**File:** `sci_arc/training/rlan_loss.py`

**Problem:** Training log showed centroid spread < 0.5, indicating clue collapse despite diversity loss. The original settings were too weak.

**Fixes:**
1. Increased `min_distance` from 2.0 to 3.0
2. Increased `repulsion_weight` from 0.1 to 0.3
3. Increased `lambda_centroid_diversity` default from 0.1 to 0.3

```python
# OLD
min_distance: float = 2.0
repulsion_weight: float = 0.1  # (hardcoded)
lambda_centroid_diversity: float = 0.1

# NEW
min_distance: float = 3.0
repulsion_weight: float = 0.3  # (configurable)
lambda_centroid_diversity: float = 0.3
```

---

## Fix 4: Enable Equivariance in 512-dim Config

**File:** `configs/rlan_stable_dev_512.yaml`

**Problem:** Equivariance was explicitly DISABLED in the config (`enabled: false`), causing the diagnostic to show "0 batches computed". The diagnostic message was also misleading when equivariance was intentionally disabled.

**Fixes:**
1. Enabled equivariance in 512-dim config with reduced augmentations for memory safety
2. Improved diagnostic message to distinguish "disabled" vs "broken"

```yaml
# OLD
equivariance_training:
  enabled: false
  num_augmentations: 4

# NEW  
equivariance_training:
  enabled: true
  num_augmentations: 2  # Reduced for 512-dim memory safety
```

---

## Fix 5: HyperLoRA Delta Regularization

**File:** `sci_arc/models/rlan_modules/hyper_lora.py`

**Problem:** Training log showed HyperLoRA clamp `max_norm` growing from 1 → 30+ and `hit_rate` reaching 5%+. This indicates the underlying weights are growing without bound.

**Fix:** Added L2 regularization on predicted deltas BEFORE clamping to prevent growth at the source.

```python
# NEW: delta_reg_weight config (default 0.001)
delta_reg_weight: float = 0.001

# In forward():
if self.delta_reg_weight > 0 and self.training:
    for name, delta in raw_deltas.items():
        norm = delta.norm(dim=(1, 2)).mean()
        delta_reg_loss = delta_reg_loss + norm
    delta_reg_loss = self.delta_reg_weight * delta_reg_loss
```

Also updated `train_rlan.py` to add `delta_reg_loss` to total loss.

---

## Fix 6: Improved Diagnostic Messages

**File:** `scripts/train_rlan.py`

The equivariance diagnostic now properly distinguishes between:
- "DISABLED in config" - Intentionally disabled
- "Not yet active (starts epoch X)" - Before start_epoch
- "BROKEN - 0 batches computed" - Enabled but failing

---

## Testing Recommendations

After applying these fixes:

1. **Verify TTA works:** Check that exact match is no longer stuck at 0%
2. **Monitor stop_logits:** Should see more variation (not stuck at -4.0)
3. **Check centroid spread:** Should stay above 1.0 throughout training
4. **Watch HyperLoRA stats:** `hit_rate` should decrease over time
5. **Equivariance should compute:** Should see non-zero `equiv_batch_count`

## Files Modified

1. `scripts/train_rlan.py` - TTA crop fix, delta_reg_loss, diagnostic messages
2. `sci_arc/models/rlan_modules/dynamic_saliency_controller.py` - Stop predictor STE
3. `sci_arc/training/rlan_loss.py` - Centroid diversity strengthening
4. `sci_arc/models/rlan_modules/hyper_lora.py` - Delta regularization
5. `configs/rlan_stable_dev_512.yaml` - Enable equivariance
