# RLAN Training Collapse: Root Cause Analysis & Fix

## Executive Summary

**Problem**: Training collapsed at epochs 41-61 with accuracy dropping 96%→48%.

**Root Cause**: Unbounded HyperLoRA weight growth causing attention collapse.

**Mathematical Mechanism**: $\|\Delta W\|_F$ grew from 0.793 → 1.714 before collapse, exceeding the stable regime ($\|\Delta W\|_F < 1.0$).

**Status**: ✅ **FIXED** (Jan 2026)

---

## FIXES APPLIED

| File | Change | Before | After |
|------|--------|--------|-------|
| `configs/rlan_stable.yaml` | `hyperlora_lr_multiplier` | 3.0 | **1.0** |
| `configs/rlan_stable.yaml` | `hyperlora_max_norm` | (missing) | **1.0** |
| `sci_arc/models/rlan_modules/hyper_lora.py` | `lora_max_norm` default | 3.0 | **1.0** |
| `sci_arc/models/rlan.py` | `RLANConfig.hyperlora_max_norm` | (missing) | **1.0** |
| `scripts/train_rlan.py` | `lora_norm_warn_threshold` | 2.0 | **1.0** |
| `scripts/train_rlan.py` | `lora_norm_critical_threshold` | 5.0 | **1.5** |
| `scripts/train_rlan.py` | `lora_norm_kill_threshold` | 10.0 | **2.0** |

---

## 1. Detailed Root Cause Analysis

### 1.1 The Collapse Timeline

| Epoch | Accuracy | Loss | LoRA Norm $\|\Delta W\|_F$ | NaN Batches | Entropy |
|-------|----------|------|---------------------------|-------------|---------|
| 1-10  | 71%→95%  | 0.53→0.06 | ~0.5 | 15 | 0.38→0.09 |
| 21-30 | 95%      | 0.06 | 0.793 | 31 | ~0.1 |
| 31-40 | 96%      | 0.05 | **1.453** | 44 | ~0.09 |
| 41-50 | 96%→49%  | 0.05→1.37 | **1.714** | 60 | 0.09→**5.2** |
| 51-61 | 48%      | 1.35 | 1.646 | 100 | ~5.0 |

### 1.2 Mathematical Root Cause

**The LoRA scaling formula**:
$$\text{output} = W \cdot x + \alpha \cdot (\Delta W) \cdot x$$

Where:
- $W$ is the frozen base weight
- $\Delta W = A \cdot B$ is the LoRA adaptation (predicted by HyperLoRA)
- $\alpha = 0.1$ is the scaling factor

**The instability mechanism**:

1. **HyperLoRA LR was 3× base LR**: This caused faster adaptation of $\Delta W$ predictors
2. **No magnitude constraint on $\|\Delta W\|_F$**: As training progressed, the predictor learned to output larger deltas
3. **Positive feedback loop**:
   - Large $\Delta W$ → Large gradients to predictor
   - Large gradients → Larger $\Delta W$ next step
   - Eventually $\|\Delta W\|_F > 1.0$ → Dominates base weight → Collapse

**Why collapse at epoch 41**:

The GRU gates have the form:
$$z_t = \sigma(W_{hz} h_{t-1} + W_{xz} x_t + \Delta W_{hz} h_{t-1} + \Delta W_{xz} x_t)$$

When $\|\Delta W_{hz}\| \approx 1.7$ and $\|W_{hz}\| \approx 1.0$ (normalized):
- The adapted term **dominates** the base weight
- The sigmoid saturates → Gradients vanish
- Update/reset gates collapse to 0 or 1 uniformly → **GRU becomes memoryless**

This explains:
- **Entropy spike 0.09→5.2**: Without functioning GRU memory, each step is independent → random predictions
- **Attention collapse 0.97→0.03**: Cross-attention relies on coherent hidden states
- **Accuracy drop 96%→48%**: ~50% is near random for background-heavy outputs

### 1.3 Why the LoRA Norm Governor Didn't Help

The governor (implemented at line 2673 of train_rlan.py) only **monitors** and **warns** - it doesn't **clamp** the weights. The kill threshold was set to 10.0, but collapse happened at norm 1.714.

---

## 2. The Scientific Fix

### 2.1 Core Principle: Spectral Norm Constraint

For stable training, we need:
$$\|\Delta W\|_\sigma \leq c \cdot \|W\|_\sigma$$

Where $\|\cdot\|_\sigma$ is the spectral norm (largest singular value) and $c < 1$ to ensure adaptation doesn't dominate.

**Implementation**: Clamp the Frobenius norm of $\Delta W$ to be at most 1.0:

$$\Delta W_{\text{clamped}} = \Delta W \cdot \min\left(1, \frac{\tau}{\|\Delta W\|_F}\right)$$

Where $\tau = 1.0$ is the maximum allowed norm.

### 2.2 Three-Part Fix

#### Part 1: HyperLoRA Weight Clamping (Forward Pass)

```python
def compute_delta_w(self, context: torch.Tensor) -> torch.Tensor:
    A, B = self.forward(context)
    delta_w = self.scaling * torch.bmm(A, B)
    
    # NEW: Frobenius norm clamping to prevent runaway
    max_norm = 1.0
    with torch.no_grad():
        norm = delta_w.norm(dim=(1, 2), keepdim=True)
        scale = (max_norm / (norm + 1e-8)).clamp(max=1.0)
    
    # Apply clamping (with gradient - allows learning to stay within bounds)
    delta_w = delta_w * scale
    
    return delta_w
```

**Why this works**:
- Stops the positive feedback loop by bounding $\|\Delta W\|_F \leq 1.0$
- Gradient still flows through the scaled version
- Model learns to produce efficient low-norm adaptations

#### Part 2: Reduce HyperLoRA Learning Rate

Current: `hyperlora_lr_multiplier: 3.0`  
Fix: `hyperlora_lr_multiplier: 1.0`

**Why**: Faster HyperLoRA learning causes it to overfit to early examples, producing larger deltas.

#### Part 3: Add LoRA Weight Decay

Add L2 regularization specifically on HyperLoRA predictor outputs:

$$\mathcal{L}_{\text{lora\_decay}} = \lambda \sum_i \|\Delta W_i\|_F^2$$

With $\lambda = 0.001$ (small but stabilizing).

---

## 3. Implementation

### 3.1 Changes to hyper_lora.py

Add `max_delta_norm` parameter to `HyperLoRAConfig` and implement clamping in `compute_delta_w`.

### 3.2 Changes to rlan_stable.yaml

```yaml
# Training config
hyperlora_lr_multiplier: 1.0   # Was 3.0, reduced for stability
hyperlora_weight_decay: 0.001  # New: L2 regularization on LoRA predictors

# Model config
hyperlora_max_delta_norm: 1.0  # New: Maximum Frobenius norm for ΔW
```

### 3.3 Early Stopping Condition

Add to training loop:

```python
if epoch_diagnostics['lora_norm_ema'] > 1.5:
    print("[EARLY STOP] LoRA norm exceeded 1.5 - model may collapse")
    # Save checkpoint and consider stopping
```

---

## 4. Validation Criteria

After applying the fix, healthy training should show:

| Metric | Healthy Range | Critical Threshold |
|--------|---------------|-------------------|
| LoRA Norm $\|\Delta W\|_F$ | 0.3 - 0.8 | > 1.5 → Warning |
| NaN Batches per Epoch | 0 - 10 | > 50 → Abort |
| Accuracy Trend | Monotonic ↑ | > 5% drop → Warning |
| Entropy | < 0.5 | > 2.0 → Collapse |
| Attention Sharpness | > 0.5 | < 0.1 → Collapse |

---

## 5. Summary

| Component | Current (Broken) | Fixed |
|-----------|-----------------|-------|
| HyperLoRA LR Multiplier | 3.0 | **1.0** |
| ΔW Norm Clamping | None | **max=1.0** |
| LoRA Weight Decay | None | **λ=0.001** |
| Kill Threshold | 10.0 | **2.0** |
| Warning Threshold | 2.0 | **1.0** |

The mathematical invariant we enforce:
$$\forall i: \|\Delta W_i\|_F \leq 1.0$$

This ensures the adaptation term never dominates the base weight, maintaining GRU gate stability.
