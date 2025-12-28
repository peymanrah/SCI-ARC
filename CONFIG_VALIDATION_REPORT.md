# rlan_stable.yaml Configuration Validation Report

**Date:** Session 2 Continuation  
**Config File:** `configs/rlan_stable.yaml`  
**Purpose:** Validate production training configuration for meta-learning stability

---

## 🔧 FIXES APPLIED THIS SESSION

### Issue #1: hyperlora_init_scale Mismatch (FIXED ✅)

**Problem:** Config had `hyperlora_init_scale: 0.01` but Bug Fix #6 increased HyperLoRA's default to `0.1` for stronger meta-learning signal.

**Analysis:**
- At `init_scale=0.01`, the adaptation effect was only 0.0001 loss delta (negligible)
- Higher init_scale = stronger initial signal for meta-learning to latch onto

**Fix Applied:**
- Updated `configs/rlan_stable.yaml`: `hyperlora_init_scale: 0.1`
- Updated `sci_arc/models/rlan.py` RLANConfig default: `hyperlora_init_scale: float = 0.1`

---

## ✅ VALIDATED SETTINGS (CORRECT)

### 1. `use_best_step_selection: false` in Training

**Status:** ✅ CORRECT

**Rationale:**
- During training, you want consistent gradient flow from the **last step**
- Entropy-based selection would add noise to gradients
- The model learns to refine over all 6 steps, not just the "best" one

**Recommendation:** Keep `false` for training. Use `true` only at inference if desired.

---

### 2. `spatial_downsample: 8` with HyperLoRA

**Status:** ✅ COMPATIBLE

**Analysis:**
- ContextEncoder downsamples 30×30 → 8×8 spatial features
- HyperLoRA applies `AdaptiveAvgPool2d(1)` to pool 8×8 → 1×1 context vector
- The 8×8 intermediate preserves spatial structure for cross-attention
- Final pooling for HyperLoRA context generation works regardless of input size

**Why This Works:**
```python
# ContextEncoder output: (B, D, 8, 8) from spatial_downsample=8
# HyperLoRA pooling: AdaptiveAvgPool2d(1) → (B, D, 1, 1) → (B, D)
# This gives a good summary of the entire context
```

**Recommendation:** No change needed. 8×8 is a good balance.

---

### 3. HyperLoRA Parameters

| Parameter | Value | Assessment |
|-----------|-------|------------|
| `hyperlora_rank` | 8 | ✅ Good balance (8-16 recommended) |
| `hyperlora_scaling` | 1.0 | ⚠️ See note below |
| `hyperlora_dropout` | 0.0 | ✅ OK for stable training |
| `hyperlora_init_scale` | 0.1 | ✅ FIXED (was 0.01) |
| `hyperlora_lr_multiplier` | 10.0 | ✅ Correct (HyperLoRA needs faster learning) |

**Note on `hyperlora_scaling: 1.0`:**
- The code's HyperLoRAConfig uses `scaling: 0.1` as default
- Config uses `1.0` which is 10× stronger
- This may be intentional with the 10× LR multiplier to compensate
- **Monitor:** Watch for LoRA magnitude explosions in training logs

---

### 4. LOO Training Configuration

**Status:** ✅ WELL CONFIGURED

| Parameter | Value | Assessment |
|-----------|-------|------------|
| `loo_training.enabled` | true | ✅ Essential for meta-learning |
| `loo_training.loss_weight` | 0.5 | ✅ Good balance with task loss |
| `loo_training.min_pairs_for_loo` | 2 | ✅ Minimum needed |

**LOO Purpose:** Explicitly trains HyperLoRA to generalize from N-1 examples to the Nth.

---

### 5. Equivariance Training

**Status:** ✅ WELL CONFIGURED

| Parameter | Value | Assessment |
|-----------|-------|------------|
| `equivariance_training.enabled` | true | ✅ Good regularizer |
| `equivariance_training.loss_weight` | 0.1 | ✅ Light weight (doesn't dominate) |
| `equivariance_training.num_augmentations` | 4 | ✅ Reasonable |

**Purpose:** Ensures LoRA weights are consistent across rotated/flipped versions.

---

### 6. Data Augmentation (TRM-Style)

**Status:** ✅ PROPERLY CONFIGURED

| Parameter | Value | Assessment |
|-----------|-------|------------|
| `color_permutation_prob` | 1.0 | ✅ TRM-style (always permute) |
| `track_augmentation` | true | ✅ Required for inverse aug |
| `use_inverse_aug` | true | ✅ Critical for proper evaluation |

**Why 100% Color Permutation Works:**
- TRM uses 1000× pre-generated augmentations with color permutation
- This config uses 400K cached samples (500 per task)
- Each sample has random augmentation baked in
- Naturally balances class distribution over training

**Important:** The equivariance loss is compatible because:
- Equivariance measures consistency of LoRA **magnitudes** (not values)
- Color permutation changes what colors mean, but the **rule** stays the same
- The LoRA should have similar structure regardless of color mapping

---

### 7. Evaluation Settings

**Status:** ✅ PROPERLY CONFIGURED

| Parameter | Value | Assessment |
|-----------|-------|------------|
| `use_trm_style_eval` | true | ✅ Standard evaluation |
| `num_augmented_views` | 8 | ✅ Full D4 group |
| `num_color_perms` | 4 | ✅ 32 total views |
| `use_voting` | true | ✅ Aggregate predictions |
| `use_inverse_aug` | true | ✅ CRITICAL - undo before comparison |

---

### 8. Inference Best Step Selection

**Status:** ✅ CORRECTLY SET

```yaml
inference:
  use_best_step_selection: false  # Use last step (matches training)
```

**Recommendation:** This is a good default. If you want to experiment with entropy-based selection:
```bash
python evaluate_rlan.py --checkpoint best.pt --use-best-step
```

---

## ⚠️ ITEMS TO MONITOR

### 1. LoRA Magnitude During Training

With the new meta-learning metrics added, monitor these in wandb/logs:
- `lora_magnitude_gru_reset` 
- `lora_magnitude_gru_update`
- `lora_magnitude_gru_candidate`
- `lora_magnitude_output_head`
- `lora_diversity` (should be > 0, indicates task-specific adaptation)

**Warning Signs:**
- If all magnitudes → 0: HyperLoRA is collapsing (context not informative)
- If magnitudes → ∞: Instability (need lower scaling or gradient clipping)
- If `lora_diversity` → 0: All tasks getting same LoRA (no meta-learning)

### 2. LOO Accuracy

Monitor `loo_accuracy` and `loo_skipped_ratio`:
- `loo_accuracy` should be > 0.5 (better than random)
- `loo_skipped_ratio` should be low (< 0.3 ideally)

If LOO accuracy is low, the HyperLoRA isn't learning to generalize.

### 3. Equivariance Loss

Monitor `equiv_delta_norm`:
- Should be small (< 1.0) indicating consistent LoRA across augmentations
- If large, the model is overfitting to specific orientations

---

## 📊 LOSS WEIGHTING SUMMARY

```
L_total = L_task + 0.5 × L_loo + 0.1 × L_equiv
```

| Loss | Weight | Purpose |
|------|--------|---------|
| Task Loss | 1.0 | Direct performance (predict correct output) |
| LOO Loss | 0.5 | Few-shot generalization (predict Nth from N-1) |
| Equivariance Loss | 0.1 | Regularization (transform invariance) |

This is a well-balanced setup:
- Task loss dominates (drives accuracy)
- LOO provides explicit meta-learning signal
- Equivariance provides light regularization

---

## 🔑 KEY INSIGHTS

### Why `use_best_step_selection: false` is Correct for Training

1. **Gradient Flow:** Training needs consistent gradients from all steps
2. **Step Refinement:** Model learns to refine predictions over 6 steps
3. **Entropy-Based Selection:** Only makes sense at inference (no gradients)

### Why `spatial_downsample: 8` Works with HyperLoRA

1. **ContextEncoder:** Produces 8×8 feature maps (good spatial resolution)
2. **Cross-Attention:** Benefits from spatial structure (8×8 = 64 positions)
3. **HyperLoRA Pooling:** AdaptiveAvgPool2d(1) collapses any size to context

### Why `color_permutation_prob: 1.0` is Safe

1. **TRM Proven:** TRM uses this successfully
2. **Rule Invariance:** The underlying rule is the same regardless of colors
3. **Equivariance Loss:** Measures LoRA structure, not specific color mappings
4. **Inverse Aug at Eval:** Colors are reversed for proper comparison

---

## ✅ FINAL VALIDATION

| Category | Status |
|----------|--------|
| HyperLoRA Parameters | ✅ Fixed (`init_scale: 0.1`) |
| Best Step Selection | ✅ Correct (`false` for training) |
| Spatial Downsample | ✅ Compatible with HyperLoRA |
| LOO Training | ✅ Properly configured |
| Equivariance Training | ✅ Properly configured |
| Data Augmentation | ✅ TRM-style, correctly implemented |
| Evaluation Settings | ✅ Inverse aug enabled |
| Meta-Learning Metrics | ✅ Added in this session |

**Recommendation:** Config is production-ready after the `hyperlora_init_scale` fix.

---

## 📝 POST-FIX: Cache Regeneration Required

Since `hyperlora_init_scale` changed from 0.01 to 0.1, you should:

1. **Delete old checkpoints** (if training from scratch):
   ```bash
   rm -rf checkpoints/rlan_stable/*
   ```

2. **No cache regeneration needed** (cache only stores augmented data, not model params)

3. **Verify the fix** by checking early training logs for:
   - Non-zero `lora_magnitude_*` values
   - Positive `lora_diversity`
   - `loo_accuracy` > 0.5 after a few epochs
