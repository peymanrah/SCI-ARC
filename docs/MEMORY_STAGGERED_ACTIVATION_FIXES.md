# Memory Management & Staggered Activation Fixes
## December 2025 Update

**Author**: Implementation Agent
**Problem**: 12GB+ memory spill at epoch 4 causing training crash
**Solution**: Staggered module activation schedule

---

## 🚨 CRITICAL ISSUE: Memory Overflow at Epoch 4

### Root Cause Analysis

Training crashed at epoch 4, batch 367 due to memory overflow:
- **GPU**: RTX 3090 (24GB VRAM)
- **Peak Memory**: 32,746MB (142% of available - spilled to shared memory)
- **Root Cause**: HyperLoRA + SolverCrossAttention + CrossAttentionInjector ALL activated at epoch 3 simultaneously

Memory progression observed:
| Batch | Peak Memory | Status |
|-------|-------------|--------|
| Batch 0 | 18,960MB | Safe (HyperLoRA activating) |
| Batch 1 | 21,221MB | Near limit |
| Batch 2-3 | 32,746MB | **OVERFLOW** (all modules active) |

### Additional Issues Discovered
- TTA consensus dropped to 3% (model not dihedral-invariant)
- Eval entropy 50x higher than train (generalization failure)
- Color permutation at 100% flagged as "CRITICAL" but still enabled

---

## ✅ FIXES APPLIED

### 1. Staggered Module Activation Schedule

**OLD**: All modules activate at epoch 3 (meta_learning_start_epoch)
**NEW**: Each module activates at a different epoch with 2-epoch gaps

| Epoch | Module | Purpose |
|-------|--------|---------|
| 3 | HyperLoRA | Task-aware LoRA deltas |
| 5 | SolverCrossAttention | Cross-attention in solver loop |
| 7 | CrossAttentionInjector | Cross-attention for context injection |
| 8 | Equivariance Loss | Dihedral invariance training |
| 12 | LOO Loss | Leave-one-out generalization |
| 14 | HPM | Hierarchical Pattern Memory |

**Files Modified**:
- `configs/rlan_stable_dev.yaml`: Added `solver_context_start_epoch: 5`, `cross_attention_start_epoch: 7`, moved `hpm_start_epoch` from 3 to 14
- `scripts/train_rlan.py`: Updated activation logic to read separate epoch settings

### 2. Smart Memory Management Module

**Created**: `sci_arc/utils/memory_manager.py`

```python
class MemoryManager:
    """
    Smart GPU memory management for RLAN training.
    
    Key methods:
    - estimate_forward_memory_mb(): Estimate memory for forward pass
    - get_safe_batch_size(): Get max safe batch size for config
    - can_activate_modules(): Check if modules can be activated safely
    - get_staggered_activation_schedule(): Get recommended activation epochs
    """
```

### 3. Memory Test Script

**Created**: `scripts/test_memory_all_epochs.py`

Tests memory usage for all epoch configurations to verify training stays within 24GB VRAM.

```bash
# Run memory test
python scripts/test_memory_all_epochs.py --config configs/rlan_stable_dev.yaml
```

### 4. Dataset Configuration Change

**OLD**: 50 tasks × 1000 samples = 50,000 total samples
**NEW**: 400 tasks × 50 samples = 20,000 total samples

**Rationale**: More task diversity is better than more augmentations of fewer tasks.

**Config Changes**:
```yaml
data:
  max_tasks: null      # Use all 400 tasks (was 50)
  samples_per_task: 50  # Samples per task (was 1000)
  num_cached_samples: 20000  # Total cached (was 50000)
  cache_path: "./cache/rlan_stable_20k_400tasks.pkl"
```

### 5. Training Speed Fix

**Issue**: Training time regressed from 20min to 60min per epoch (3x slower)

**Root Cause**: Cache clearing every 10 batches instead of every 50 batches

**Fix**: Changed cache clearing frequency from every 10 batches to every 100 batches

```python
# OLD (line 1880)
if batch_idx % 10 == 0:
    torch.cuda.empty_cache()

# NEW
if batch_idx > 0 and batch_idx % 100 == 0:
    torch.cuda.empty_cache()
```

---

## 📋 Config Summary (rlan_stable_dev.yaml)

### Training Schedule
```yaml
training:
  meta_learning_start_epoch: 3   # HyperLoRA
  solver_context_start_epoch: 5  # SolverCrossAttention
  cross_attention_start_epoch: 7 # CrossAttentionInjector
  
  equivariance_training:
    start_epoch: 8
    
  loo_training:
    start_epoch: 12

model:
  hpm_start_epoch: 14
```

### Memory Safety
- Batch size: 80 (safe with gradient checkpointing)
- Gradient accumulation: 4 steps (effective batch = 320)
- Gradient checkpointing: Enabled (trades ~33% compute for 40% memory reduction)

---

## 🧪 Tests Added

### test_staggered_activation.py

New test file with comprehensive tests:
1. `test_epoch_0_2_no_modules_active()` - Verify staging works
2. `test_epoch_3_hyperlora_only()` - Only HyperLoRA at epoch 3
3. `test_epoch_5_solver_context_added()` - SolverCrossAttention at epoch 5
4. `test_epoch_7_cross_attention_added()` - All three at epoch 7
5. `test_epoch_14_hpm_added()` - HPM at epoch 14
6. `test_epoch_gap_between_activations()` - Verify minimum gaps
7. `test_forward_pass_at_each_stage()` - Forward pass works
8. `test_config_file_staggered_settings()` - Config validation
9. `test_backward_compatibility()` - Model works without explicit flags

**Run tests**:
```bash
python -m pytest tests/test_staggered_activation.py -v -s
```

---

## 📊 TRM Color Permutation Investigation

**Question**: Does TRM use 100% color permutation or something smaller?

**Answer**: TRM also uses 100% color permutation probability.

From `others/TRM/arc_loader.py` line 101:
```python
color_perm_prob = 1.0  # Default is 100%
```

**Conclusion**: Our 100% color permutation is consistent with TRM. The "CRITICAL" flag was overly cautious.

---

## 🔍 Files Changed Summary

| File | Change |
|------|--------|
| `configs/rlan_stable_dev.yaml` | Added staggered epoch settings, changed 400×50 samples |
| `scripts/train_rlan.py` | Read new epoch settings, staggered activation logic, fixed cache clearing |
| `sci_arc/utils/memory_manager.py` | **NEW** - Smart memory management |
| `scripts/test_memory_all_epochs.py` | **NEW** - Memory test script |
| `tests/test_staggered_activation.py` | **NEW** - Smoke tests |

---

## ⚠️ Backward Compatibility

All changes are backward compatible:
- If `solver_context_start_epoch` is not in config, defaults to `meta_learning_start_epoch + 2`
- If `cross_attention_start_epoch` is not in config, defaults to `solver_context_start_epoch + 2`
- Model forward pass works without explicit flag setting
