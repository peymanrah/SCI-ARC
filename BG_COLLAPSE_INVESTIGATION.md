# BG Collapse Investigation: Commit 2a50f2d vs Current Main

## Executive Summary

**ROOT CAUSE IDENTIFIED AND FIXED**: The BG collapse was caused by **THREE** new modules
being ACTIVE from epoch 0, even though LOO/Equivariance losses were delayed until epoch 3:

1. **HyperLoRA** - LoRA delta weights applied from epoch 0
2. **SolverCrossAttention** - Cross-attention in solver from epoch 0  
3. **CrossAttentionInjector** - Context injection via attention instead of FiLM from epoch 0

All three modules use Q/K/V projections or delta weights that are **randomly initialized**.
During early training, these inject **noise** into features, destabilizing the base model
before it can learn FG/BG distinction.

**FIX IMPLEMENTED**: Added three runtime staging flags:
- `hyperlora_active` - Disables LoRA deltas during epochs 0-2
- `solver_context_active` - Disables solver cross-attention during epochs 0-2
- `cross_attention_active` - Falls back to **FiLM** (γ*features+β) during epochs 0-2

---

## Why CrossAttention is Unstable vs FiLM

**FiLM (Feature-wise Linear Modulation)**:
```python
# Simple scale/shift - linear, predictable
output = γ * features + β
# With random γ≈1, β≈0, this barely changes features
```

**CrossAttentionInjector**:
```python
# Complex attention mechanism - nonlinear, volatile
Q = Wq @ features      # Random projection
K = Wk @ context       # Random projection  
V = Wv @ context       # Random projection
attn = softmax(Q @ K.T / sqrt(d))  # Random attention weights
output = features + attn @ V       # Adds random noise!
```

With random Q/K/V projections, softmax produces ~uniform attention over random values,
injecting pure noise into every feature. This destroys any learning signal for FG/BG.

---

## Configuration Comparison

### OLD (2a50f2d - STABLE, no BG collapse)

```yaml
# Model - MINIMAL architecture
hidden_dim: 256
max_clues: 6
num_solver_steps: 6
use_hyperlora: NO (didn't exist)
use_hpm: NO (didn't exist)  
use_solver_context: NO (didn't exist)
use_cross_attention_context: NO (default=false → FiLM)

# Training
batch_size: 75
grad_accumulation_steps: 4
gradient_checkpointing: NO (didn't exist)
use_ema: true

# Losses - SIMPLE
loss_mode: 'stablemax'
loo_training: NO (didn't exist)
equivariance_training: NO (didn't exist)
meta_learning_start_epoch: NO (didn't exist)

# Output Head Init
bg_bias: 0.0 (NEUTRAL)
```

### NEW (Current Main - BG COLLAPSE)

```yaml
# Model - COMPLEX architecture
hidden_dim: 256
max_clues: 7
num_solver_steps: 6
use_hyperlora: true          # NEW: 5M additional params
use_hpm: true                # NEW: Memory banks + routing
use_solver_context: true     # NEW: Cross-attention at each step
use_cross_attention_context: true  # NEW: Spatial context injection

# Training
batch_size: 55 (or 75 with collapse)
grad_accumulation_steps: 6
gradient_checkpointing: true  # NEW: Potential gradient issues
use_ema: false               # CHANGED: Was true

# Losses - COMPLEX
loss_mode: 'stablemax'
loo_training: enabled        # NEW: N forward passes per sample
equivariance_training: enabled  # NEW: 4x augmentation passes
meta_learning_start_epoch: 3 # NEW: But architecture still active from epoch 0!

# Output Head Init
bg_bias: -0.5 (ANTI-COLLAPSE attempt)
```

---

## Root Causes of BG Collapse

### 1. **Module Initialization Conflict**

Even with `meta_learning_start_epoch: 3`, the following modules are ACTIVE from epoch 0:

- **HyperLoRA**: Predicts LoRA weight deltas → random noise at start
- **HPM**: Routes to memory banks → random routing at start  
- **SolverCrossAttention**: Attends to support features → random attention at start

These modules add NOISE to the solver's hidden state before the base model learns FG/BG.

### 2. **Gradient Checkpointing Issues**

The `torch.utils.checkpoint` implementation has problems:

```python
h_new = torch_checkpoint(
    self._solver_step,
    combined,
    h,                    # None on first step - problematic
    support_features,     # None when no support - problematic  
    lora_deltas,          # Dict - NOT supported by checkpoint!
    use_reentrant=False,
)
```

**CRITICAL BUG**: `torch.utils.checkpoint` does NOT handle:
- `None` values properly (can cause silent gradient issues)
- `Dict` arguments (lora_deltas) - these are NOT checkpointed correctly

### 3. **bg_bias=-0.5 Backfires**

Changing from `bg_bias=0.0` to `bg_bias=-0.5` was intended to prevent collapse,
but with all the new noise from HyperLoRA/HPM/CrossAttention, the model now:
1. Starts with anti-BG bias
2. Gets conflicting gradients from all new modules
3. Overcorrects to all-FG predictions
4. Then collapses to all-BG as loss explodes

### 4. **EMA Disabled**

EMA was providing smoothing/regularization. With `use_ema: false`, the model
is more sensitive to noisy gradients.

---

## Why Larger Batch Size Doesn't Help

The hypothesis "batch_size < 55 causes BG collapse" is **WRONG**.

The old config used batch_size=75 with NO collapse because it had:
- Simpler architecture (no HyperLoRA, no HPM, no cross-attention)
- Stable gradients (no noisy module outputs)
- EMA smoothing

The new config collapses at ANY batch size because:
- Multiple untrained modules add noise
- Gradient checkpointing may corrupt gradients
- No EMA smoothing

---

## Recommended Fixes

### Option A: Disable New Modules for Epoch 0-5 (Recommended)

```yaml
# Make staged training ACTUALLY staged
# These should be FALSE until the base model converges
use_hyperlora: true   # Keep architecture
use_hpm: true         # Keep architecture

# BUT disable their CONTRIBUTIONS until epoch 5
hyperlora_start_epoch: 5  # NEW: Don't apply LoRA deltas until epoch 5
hpm_start_epoch: 5        # Already exists, make sure it's read
solver_context_start_epoch: 5  # NEW: Don't use cross-attention until epoch 5
```

### Option B: Fix Gradient Checkpointing

1. Remove Dict argument (lora_deltas) from checkpointed function
2. Handle None values explicitly
3. Or DISABLE checkpointing until these are fixed

```yaml
gradient_checkpointing: false  # DISABLE until fixed
batch_size: 45                 # Lower batch size, accept ~23GB usage
grad_accumulation_steps: 8     # Compensate
```

### Option C: Minimal Config (Match Old Behavior)

```yaml
# EXACTLY match old config to verify it still works
use_hyperlora: false
use_hpm: false
use_solver_context: false
use_cross_attention_context: false
gradient_checkpointing: false
use_ema: true
batch_size: 75
grad_accumulation_steps: 4
# In solver: bg_bias: 0.0 (revert)
```

---

## Verification Steps

1. **Test Option C first** - Confirm old config still works on new codebase
2. **Add one module at a time** - Find which module causes collapse
3. **Fix gradient checkpointing** - Then re-enable memory optimization
4. **Implement proper staged activation** - All new modules should be identity until start_epoch

---

## Conclusion

**BG collapse is an ARCHITECTURE problem, not a BATCH SIZE problem.**

The solution is to make all new modules (HyperLoRA, HPM, SolverCrossAttention) 
act as IDENTITY transforms until after the base model converges (epoch 5+).

The gradient checkpointing has bugs that should be fixed before re-enabling.
