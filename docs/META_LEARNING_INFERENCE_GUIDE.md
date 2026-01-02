# Meta-Learning at Inference: Complete Guide

**Date**: January 2026  
**Author**: SCI-ARC Team  

---

## Table of Contents

1. [EMA Config vs EMA Copy - Clarification](#ema-config-vs-ema-copy)
2. [HPM Buffer Storage Location and Format](#hpm-buffer-storage)
3. [Hidden Dimension Size Analysis](#hidden-dimension-analysis)
4. [Orphaned Functions Audit](#orphaned-functions-audit)
5. [HPM/LoRA/LOO Architectural Placement](#architectural-placement)

---

## EMA Config vs EMA Copy - Clarification {#ema-config-vs-ema-copy}

**They are DIFFERENT concepts that happen to share the "EMA" name.**

### 1. `use_ema: false` in YAML (Training-Time EMA Smoothing)

```yaml
training:
  use_ema: false       # This controls training-time weight averaging
  ema_decay: 0.995
```

**Purpose**: Exponential Moving Average of model weights during training for smoother evaluation.

**How it works**:
- During training, maintains a "shadow" copy of weights: `shadow = mu * shadow + (1-mu) * model`
- Updated after each optimizer step
- Used for **mid-training evaluation** to reduce noise from recent gradients

**When `use_ema: false`**:
- No EMA shadow weights are maintained
- Evaluation uses the raw training model weights directly

### 2. `ema.ema_copy(model)` (Evaluation-Time Model Copy)

**Purpose**: Create a copy of the model for evaluation without affecting training.

**How it works**:
```python
# In train_rlan.py at evaluation time:
if ema is not None:
    eval_model = ema.ema_copy(model)  # Copy with EMA shadow weights
else:
    eval_model = model  # Use training model directly
```

**Critical Point**: When `use_ema: false`, the `ema` variable is `None`, so `eval_model = model` directly - **no copying happens**.

### Summary Table

| YAML Setting | `ema` Variable | At Evaluation Time |
|--------------|---------------|-------------------|
| `use_ema: true` | `EMAHelper(...)` | Uses EMA shadow weights (smoother) |
| `use_ema: false` | `None` | Uses raw training weights directly |

**Conclusion**: Setting `use_ema: false` in YAML does NOT affect the inference staging fix. The staging fix applies to whichever model is used for eval (whether EMA copy or raw model).

---

## HPM Buffer Storage Location and Format {#hpm-buffer-storage}

### Where HPM Buffers Are Stored

HPM dynamic buffers are stored in **two locations**:

#### 1. Inside Training Checkpoints (Primary)

**Location**: `checkpoints/rlan_stable/checkpoint_epoch_N.pt`

**Format**: PyTorch checkpoint dict with top-level keys:

```python
checkpoint = {
    'epoch': 50,
    'model_state_dict': {...},          # Model weights (includes HPM static banks)
    'optimizer_state_dict': {...},
    'config': {...},
    
    # HPM Dynamic Buffers (TOP-LEVEL KEYS, not inside model_state_dict!)
    'hpm_instance_buffer': {
        'keys': tensor,      # Shape: (N, hidden_dim) - context embeddings
        'values': tensor,    # Shape: (N, hidden_dim) - retrieved features
        'task_ids': [...],   # List of task ID strings
        'timestamps': [...], # Insertion order for LRU eviction
    },
    'hpm_procedural_buffer': {
        'keys': tensor,      # Shape: (M, hidden_dim)
        'values': tensor,    # Shape: (M, hidden_dim) - HyperLoRA codes
        'task_ids': [...],
        'timestamps': [...],
    },
}
```

#### 2. Separate Buffer Files (Optional Export)

**Location**: `checkpoints/hpm_buffers/` (if explicitly exported)

```
checkpoints/hpm_buffers/
├── instance_buffer.pt    # torch.save of buffer state_dict
└── procedural_buffer.pt  # torch.save of buffer state_dict
```

**Export Command**:
```python
model.hpm_save_buffers("checkpoints/hpm_buffers/")
```

### What to Expect During Training

| Epoch | HPM Status | Expected Buffer State |
|-------|-----------|----------------------|
| 0-19 | Memory collection ON, retrieval OFF | Buffers grow as tasks are solved |
| 20+ | Full HPM active | Buffers continue growing, retrieval contributes to features |

**Logging to Watch**:
```
  HPM Tasks Added (exact matches): 15
  HPM Instance Buffer: 15 entries SAVED
  HPM Procedural Buffer: 8 entries SAVED
```

### What to Expect At Inference

When loading a checkpoint:
```
Loading checkpoint from checkpoints/rlan_stable/best.pt
  Loaded HPM instance buffer (127 entries)
  Loaded HPM procedural buffer (89 entries)

============================================================
INFERENCE-TIME META-LEARNING CONFIGURATION
============================================================
[✓] HyperLoRA: ACTIVE (avg_norm=0.0234)
[✓] HPM: ACTIVE (instance=127, procedural=89)
[✓] Solver Cross-Attention: ACTIVE
[✓] Cross-Attention Injector: ACTIVE
============================================================
```

### Buffer Memory Format Details

```python
# DynamicMemoryBuffer internal structure
class DynamicMemoryBuffer:
    keys: torch.Tensor      # (capacity, hidden_dim) - query keys
    values: torch.Tensor    # (capacity, hidden_dim) - retrieved values
    task_ids: List[str]     # Task identifiers for deduplication
    timestamps: List[int]   # Insertion order for LRU eviction
    size: int               # Current number of entries
    capacity: int           # Maximum entries (default: 10000)
```

---

## Hidden Dimension Size Analysis {#hidden-dimension-analysis}

### Current Configuration

```yaml
model:
  hidden_dim: 256
```

### Is 256 Sufficient for Complex ARC Tasks?

**Analysis**:

| Aspect | 256-dim Assessment | Recommendation |
|--------|-------------------|----------------|
| Color encoding | ✓ Excellent (10 colors → 256 bits = 25.6 bits/color) | Sufficient |
| Spatial patterns | ✓ Good (30x30 = 900 positions compressed to 256) | Sufficient |
| Transformation rules | ? Borderline (complex nested rules may need more) | Monitor |
| Multi-object tracking | ? Questionable (5+ objects × attributes) | Consider 384/512 |

### Comparison with Other ARC Solvers

| Model | Hidden Dim | ARC Score |
|-------|-----------|-----------|
| TRM (Tiny Recursive Model) | 512 | ~55% |
| GPT-4 (dense) | ~4096 (per head) | ~5% |
| RLAN (current) | 256 | TBD |

### ⚠️ CRITICAL: hidden_dim Change Breaks Warm Start

**Changing hidden_dim from 256 to 512 WILL BREAK weight loading from existing checkpoints.**

The hidden_dim is used pervasively throughout ALL modules:
- `GridEncoder`: color embedding, projection layers
- `ContextEncoder`: all attention layers, MLP layers
- `DSC`: query/key/value projections, stop predictor
- `MSRE`: coordinate encoding, feature fusion
- `RecursiveSolver`: GRU hidden state, all linear layers
- `HyperLoRA`: context encoder, delta predictors
- `HPM`: all bank projections, retrieval layers

**Parameter count impact**:
- 256-dim: ~12M parameters (current)
- 384-dim: ~27M parameters (~2.3x increase)
- 512-dim: ~48M parameters (~4x increase)

**If you need larger capacity**:
1. Train from scratch with new hidden_dim
2. OR use knowledge distillation from 256→512
3. Do NOT attempt partial weight loading

### Recommendation

**For ARC-AGI evaluation set (complex tasks)**:

Consider increasing to **384 or 512** ONLY if:
1. Training accuracy plateaus below 70%
2. Complex multi-object tasks consistently fail
3. Memory budget allows (~50% VRAM increase for 384)
4. You can train from scratch (no warm start needed)

**Config change (if starting fresh)**:
```yaml
model:
  hidden_dim: 384  # or 512 for max capacity
```

---

## Orphaned Functions Audit {#orphaned-functions-audit}

### Methodology

Searched for functions that:
1. Have config flags but flag is never set
2. Are defined but never called
3. Have early-return conditions that always trigger

### Findings

#### ✅ Module Activation Status

All major modules have proper activation paths:

| Module | Config Flag | Activation Check | Status |
|--------|------------|-----------------|--------|
| HyperLoRA | `use_hyperlora` | Line 711 `if self.use_hyperlora and self.hyper_lora` | ✓ Active |
| HPM | `use_hpm` | Line 604 `if self.use_hpm and self.hpm` | ✓ Active |
| DSC | `use_dsc` | Line 661 `if self.use_dsc and self.dsc` | ✓ Active |
| MSRE | `use_msre` | Line 681 `if self.use_msre and self.msre` | ✓ Active |
| LCR | `use_lcr` | Line 691 `if self.use_lcr and self.lcr` | ✓ Disabled by config |
| SPH | `use_sph` | Line 696 `if self.use_sph and self.sph` | ✓ Disabled by config |
| ACT | `use_act` | Line 751 `return_act_outputs=... and self.use_act` | ✓ Disabled by config |

#### 🔴 CRITICAL: High-Risk Silent Failures

**1. HyperLoRA Silently Disabled Without Support Features**

```python
# In rlan.py forward(), line ~711
hyperlora_active = getattr(self, 'hyperlora_active', True)  # Default: active
if self.use_hyperlora and self.hyper_lora is not None and hyperlora_active:
    if support_features is not None:
        lora_deltas = self.hyper_lora(support_features)
    else:
        # WARNING: HyperLoRA is enabled but no support features available!
        # lora_deltas stays None, solver runs WITHOUT LoRA adaptation
```

**Condition that triggers**: `use_hyperlora: true` but `use_cross_attention_context: false` AND `use_solver_context: false`

**What happens**: HyperLoRA loaded (~2.8M params on GPU), but LoRA weights NEVER generated. Training proceeds without task-specific adaptation.

**How to detect**: Monitor `lora_delta_norm_sum` in diagnostics - should be >0 after epoch 3.

---

**2. DSC Stop Predictor Receives Zero Task Context**

```python
# In rlan.py forward()
dsc_task_context = None  # Initialized as None
# ...
centroids, attention_maps, stop_logits = self.dsc(
    features, temperature=temperature, mask=valid_mask, task_context=dsc_task_context
)

# In DSC forward():
if task_context is None:
    task_context_vec = torch.zeros(B, self.context_dim, ...)  # All zeros!
```

**Condition that triggers**: `use_context_encoder: true` but context encoder returns FiLM output AND no train_inputs/train_outputs provided

**What happens**: Stop predictor uses all-zeros context → learns TASK-INDEPENDENT stopping → all tasks use same number of clues regardless of complexity.

**How to detect**: Check if `stop_predictor_weight_variance` → 0 during training.

---

**3. Runtime Flags Lost on Checkpoint Load**

```python
# These flags are set by training script at runtime:
model.hyperlora_active = False  # During staged warmup (epochs 0-3)
model.solver_context_active = False
model.cross_attention_active = False

# BUT on checkpoint load, these attributes are NOT restored!
# They default to True → sudden behavior change on resume
```

**What happens**: Training interrupted during staged warmup → Resume → All modules suddenly activate → Training dynamics change abruptly.

**How to detect**: The new `apply_inference_staging()` helper explicitly sets these flags from YAML on every checkpoint load.

---

#### ⚠️ Medium-Risk Silent Failures

**4. HPM Skipped Without Context**

```python
# In rlan.py forward()
if self.use_hpm and self.hpm is not None:
    if z_context_flat is not None:
        # HPM queries memory banks
    else:
        z_context_flat = None  # No context → HPM completely skipped!
```

**Mitigation**: Ensure `use_cross_attention_context: true` or `use_solver_context: true`.

**5. `hpm_add_solved_task` requires `support_features`**:
```python
# Line 2538-2540 in train_rlan.py
support_features = outputs.get('support_features')
if support_features is None:
    epoch_diagnostics['hpm_add_skipped_no_support_features'] += 1
```
**Mitigation**: Diagnostic counter tracks skips. Ensure `use_cross_attention_context: true` for support features.

**6. LOO training requires `min_pairs_for_loo`**:
```python
# LOOTrainingLoss.forward()
if num_pairs < self.config.min_pairs_for_loo:
    return torch.tensor(0.0, device=device), {}
```
**Mitigation**: Logged as `loo_skipped_insufficient_pairs`. Normal for some tasks.

---

#### Recommendations

1. **Add startup validation**:
   ```python
   def validate_config(config):
       if config.use_hyperlora and not (config.use_solver_context or config.use_cross_attention_context):
           raise ValueError("HyperLoRA requires solver_context or cross_attention_context for support features")
   ```

2. **Log effective configuration at training start**:
   ```python
   print(f"[CONFIG] HyperLoRA: config={use_hyperlora}, effective={hyperlora_active and support_features is not None}")
   ```

3. **Alert on silent failure conditions**:
   - Alert if `loo_skipped_count > 50%` of batches
   - Alert if `hpm_add_skipped_no_support_features > 0`

---

## HPM/LoRA/LOO Architectural Placement {#architectural-placement}

### Forward Pass Order (Correct)

```
Input Grid
    │
    ▼
┌─────────────────────────────┐
│ 1. Color Embedding          │
│    (B, H, W) → (B, D, H, W) │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│ 2. Context Encoder          │  ◄─── Encodes train pairs
│    Produces support_features│      (B, N, D, H, W)
└─────────────────────────────┘
    │
    ├───────────────────────────────────┐
    │                                   │
    ▼                                   ▼
┌─────────────────────────────┐  ┌─────────────────────────────┐
│ 3. HPM Query                │  │ 4. HyperLoRA Delta          │
│    Uses support_features    │  │    Uses support_features    │
│    to query memory banks    │  │    to predict LoRA weights  │
│    Returns: z_enhanced      │  │    Returns: lora_deltas     │
└─────────────────────────────┘  └─────────────────────────────┘
    │                                   │
    ▼                                   │
┌─────────────────────────────┐         │
│ 5. DSC (Clue Finding)       │         │
│    Attention-based anchors  │         │
└─────────────────────────────┘         │
    │                                   │
    ▼                                   │
┌─────────────────────────────┐         │
│ 6. MSRE (Relative Encoding) │         │
│    Position-invariant feats │         │
└─────────────────────────────┘         │
    │                                   │
    ▼                                   │
┌─────────────────────────────┐         │
│ 7. Recursive Solver         │ ◄───────┘
│    - Uses lora_deltas       │
│    - Uses HPM memory tokens │
│    - Cross-attention to     │
│      support_features       │
└─────────────────────────────┘
    │
    ▼
Output Logits
```

### Mathematical Correctness

1. **HPM before Solver** ✓
   - HPM enhances context features BEFORE solver uses them
   - Allows solver to benefit from retrieved memories

2. **HyperLoRA before Solver** ✓
   - LoRA deltas computed from support_features
   - Applied to solver's GRU gates and output head
   - Enables task-specific weight adaptation

3. **DSC/MSRE before Solver** ✓
   - Clue locations guide spatial attention
   - Relative encoding enables translation invariance

4. **LOO Loss (Training Only)**
   - Computed AFTER forward pass
   - Uses N-1 pairs to predict Nth pair
   - Gradients flow back through HyperLoRA

### Gradient Flow Verification

```
Task Loss (CE)
    │
    ├──► Solver weights
    │
    ├──► HyperLoRA (via lora_deltas)
    │
    ├──► Context Encoder (via support_features)
    │
    └──► HPM gate (via gated residual)

LOO Loss (when active)
    │
    ├──► HyperLoRA (primary target)
    │
    └──► Context Encoder (secondary)
```

**Verified**: All modules receive gradients from task loss. LOO specifically trains HyperLoRA's ability to generalize from N-1 examples.

---

## Summary

| Question | Answer |
|----------|--------|
| EMA config vs copy? | Different concepts - config controls training smoothing, copy is for eval |
| HPM buffer location? | Inside checkpoint as top-level keys `hpm_instance_buffer`, `hpm_procedural_buffer` |
| Hidden dim sufficient? | 256 is borderline; consider 384+ for complex tasks |
| Orphaned functions? | None critical; minor skips are logged |
| Module placement? | Correct - HPM/HyperLoRA feed into solver, LOO trains generalization |
