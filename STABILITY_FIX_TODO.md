# Training Stability Fix TODO

**Date:** January 2026  
**Based on:** Log analysis from "fix DSC task-conditioned stop pred" git version  
**Symptom:** FG/BG accuracy drops from 93% → 79% after epoch 55

---

## Root Causes Identified

| # | Issue | Evidence | Severity |
|---|-------|----------|----------|
| 1 | LoRA Delta Norm Explosion | 1.27 → 1.80 (+42%) | 🔴 CRITICAL |
| 2 | Stop Predictor Saturation | Clues stuck at 1.0, stop_logit=3.1 | 🟠 HIGH |
| 3 | Attention Collapse | max 0.98 → 0.005 | 🔴 CRITICAL |
| 4 | Solver Logit Explosion | ±67 exceeded soft_clamp threshold | 🟠 HIGH |
| 5 | Meta-Escalation Didn't Help | Only checks NaN, not accuracy | 🟡 MEDIUM |

---

## P0: Critical Fixes (Must Have)

### P0.1: Add LoRA Delta Norm Hard Clamping

**File:** `sci_arc/models/rlan_modules/hyper_lora.py`

**Current:** No clamping in `compute_delta_w()`
```python
deltas = {
    'gru_reset': scale * self.gru_reset_lora.compute_delta_w(context),
    ...
}
```

**Fix:** Add per-sample norm clamping
```python
def _clamp_delta_norm(self, delta: torch.Tensor, max_norm: float = 3.0) -> torch.Tensor:
    """Clamp per-sample LoRA delta to max L2 norm."""
    # delta: (B, D, D)
    norms = delta.norm(dim=(1, 2), keepdim=True)  # (B, 1, 1)
    scale = torch.clamp(max_norm / (norms + 1e-8), max=1.0)
    return delta * scale

deltas = {
    'gru_reset': self._clamp_delta_norm(scale * self.gru_reset_lora.compute_delta_w(context)),
    ...
}
```

**Config:** Add `lora_max_norm: 3.0` to YAML

---

### P0.2: Add Attention Collapse Detection to Meta-Escalation

**File:** `scripts/train_rlan.py`

**Current:** Only checks NaN and gradient explosion
```python
is_stable = (prev_nan_streak < nan_streak_threshold and 
             prev_grad_events < grad_explosion_threshold and ...)
```

**Fix:** Add attention entropy check
```python
# After epoch ends, compute attention health
attention_entropy = epoch_diagnostics.get('dsc_entropy_mean', 0.0)
attention_max = epoch_diagnostics.get('attention_max', 1.0)

# Attention collapse: max < 0.1 or entropy > 5.0
attention_collapsed = attention_max < 0.1 or attention_entropy > 5.0

is_stable = (prev_nan_streak < nan_streak_threshold and 
             prev_grad_events < grad_explosion_threshold and
             not attention_collapsed and ...)
```

---

### P0.3: Lower Solver Logit Clamping Threshold

**File:** `sci_arc/models/rlan_modules/recursive_solver.py`

**Current:** `soft_clamp_logits(x, threshold=1000.0)`

**Fix:** 
```python
def soft_clamp_logits(x: torch.Tensor, threshold: float = 50.0, max_val: float = 100.0) -> torch.Tensor:
```

This matches DSC's ±50 clamp and prevents the ±67 values seen in the log.

---

## P1: High Priority Fixes

### P1.1: Reduce Stop Predictor Initial Bias

**File:** `sci_arc/models/rlan_modules/dynamic_saliency_controller.py`

**Current:** `_init_stop_predictor_for_entropy_coupling(init_bias=-1.0)`
- This gives initial stop prob ~0.27 for clue 1, which quickly saturates to 0.98

**Fix:** Start with even lower bias or add entropy regularization
```python
def _init_stop_predictor_for_entropy_coupling(self, init_bias: float = -2.0):
    # Lower bias = more clues active initially
```

---

### P1.2: Add Accuracy-Based Meta-Escalation Pause

**File:** `scripts/train_rlan.py`

**Current:** No accuracy tracking for pause decisions

**Fix:**
```python
# Track accuracy EMA
accuracy_ema = 0.9 * accuracy_ema + 0.1 * epoch_accuracy
accuracy_drop = prev_best_accuracy - accuracy_ema

# Pause if accuracy drops >5% from best
accuracy_collapsed = accuracy_drop > 0.05

is_stable = (... and not accuracy_collapsed)
```

---

### P1.3: Add LoRA Norm Regularization Loss

**File:** `sci_arc/training/losses.py` or trainer

**Current:** No regularization on LoRA delta magnitude

**Fix:**
```python
def lora_norm_regularization(deltas: Dict[str, torch.Tensor], target_norm: float = 1.0) -> torch.Tensor:
    """Encourage LoRA deltas to stay near target_norm."""
    norms = []
    for name, delta in deltas.items():
        if name != 'context':
            norm = delta.norm(dim=(1, 2)).mean()
            norms.append((norm - target_norm).abs())
    return sum(norms) / len(norms)
```

Add to loss with small weight (e.g., 0.001).

---

## P2: Nice to Have

### P2.1: Dynamic Clue Diversity Loss

Encourage different clues to attend to different regions:
```python
def clue_diversity_loss(attention_maps: torch.Tensor) -> torch.Tensor:
    # attention_maps: (B, K, H, W)
    # Encourage low overlap between clue attention maps
    overlap = (attention_maps[:, :, None] * attention_maps[:, None, :]).sum(dim=(-1, -2))  # (B, K, K)
    # Mask diagonal
    mask = 1 - torch.eye(K, device=overlap.device)
    return (overlap * mask).mean()
```

### P2.2: Log LoRA Norm Per-Sample Distribution

Track not just mean but also max LoRA norm per batch to catch outliers.

---

## Verification Checklist

After implementing fixes:

- [ ] LoRA delta norm stays < 3.0 for 100+ epochs
- [ ] Stop logits have variance > 0.5 (clue count varies per task)
- [ ] Attention max stays > 0.3 (no collapse)
- [ ] Solver logits stay within ±50
- [ ] Accuracy doesn't drop >5% from peak during training
- [ ] Meta-escalation pauses on attention collapse

---

## Config Changes

Add to `configs/rlan_stable_dev.yaml`:

```yaml
# P0.1: LoRA norm clamping
hyper_lora:
  lora_max_norm: 3.0  # Hard clamp per-sample norm

# P0.3: Lower solver logit threshold
recursive_solver:
  logit_clamp_threshold: 50.0  # Was 1000.0

# P1.2: Accuracy-based pause
meta_escalation:
  accuracy_drop_threshold: 0.05  # Pause if acc drops 5%
  attention_collapse_threshold: 0.1  # Pause if attn max < 0.1

# P1.3: LoRA regularization
loss_weights:
  lora_norm_reg: 0.001
```
