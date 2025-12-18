# RLAN Training Stability: Complete Analysis & Fixes

## Executive Summary

After extensive ablation testing, we've identified and fixed multiple issues causing NaN/instability in RLAN training. The model now achieves **100% accuracy** on grid expansion tasks.

---

## Key Findings

### 1. Grid Expansion Works ✅

The model CAN learn tasks where output is larger than input:

| Task | Grid Expansion | Epochs to 100% | Status |
|------|----------------|----------------|--------|
| f5b8619d | 6→12 (4x) | 94 | ✓ SUCCESS |
| b91ae062 | 3→12 (14x) | 98 | ✓ SUCCESS |
| 007bbfb7 | 3→9 (8x tiling) | 132 | ✓ SUCCESS |
| d515da2d (simple) | Same size | 2 | ✓ SUCCESS |

### 2. Root Causes of NaN (Now Fixed)

| Issue | Problem | Fix |
|-------|---------|-----|
| Scheduler dynamics | OneCycle/Cosine can cause learning rate spikes | Added `scheduler: none` option |
| Per-module LR multipliers | 10x boost on DSC/MSRE caused instability | Set all to 1.0 |
| Padding value corruption | pad_grid() could overwrite -100 with 0 | Fixed to preserve -100 |
| Gradient accumulation | Large gradients with FG emphasis | gradient_clip: 1.0 |

### 3. Cached vs On-the-Fly Samples

**Finding**: Cached samples work well for per-task training.

- Test environment: 8 cached samples, repeated each epoch
- All tests achieved 100% with enough epochs
- Simple tasks: 2-5 epochs
- Expansion tasks: 94-132 epochs

**Implication**: For multi-task training, caching a subset of samples per task is viable.

---

## Recommended Configuration

### configs/rlan_stable.yaml (Maximum Stability)

```yaml
# Training
training:
  batch_size: 8
  learning_rate: 5e-4
  weight_decay: 0.01
  max_epochs: 250
  gradient_clip: 1.0
  scheduler: none           # Constant LR - most stable
  warmup_epochs: 0
  
  # Per-module LR multipliers - ALL 1.0 for stability
  dsc_lr_multiplier: 1.0
  msre_lr_multiplier: 1.0
  lcr_lr_multiplier: 1.0
  sph_lr_multiplier: 1.0

# Loss - minimal regularization
loss:
  mode: weighted_stablemax
  bg_weight_cap: 2.0
  fg_weight_cap: 5.0
  lambda_entropy: 0.0
  lambda_sparsity: 0.0
  lambda_predicate: 0.0
  lambda_curriculum: 0.0
  lambda_deep_supervision: 0.0
```

### configs/rlan_minimal.yaml (With Scheduler)

```yaml
training:
  scheduler: cosine         # Use cosine if you want LR decay
  warmup_epochs: 10         # 10 epoch warmup prevents early NaN
  min_lr_ratio: 0.01        # Don't decay too far
```

---

## Code Fixes Applied

### 1. sci_arc/data/dataset.py - pad_grid()

```python
def pad_grid(grid: Union[ARCGrid, np.ndarray, torch.Tensor], 
             max_size: int,
             pad_value: int = 0) -> torch.Tensor:
    # FIX: Detect if grid contains ignore_index (-100) and use that for padding
    if isinstance(grid, (np.ndarray, list)):
        grid_arr = np.array(grid) if isinstance(grid, list) else grid
        if np.any(grid_arr == -100):
            pad_value = -100  # Preserve ignore_index in targets
```

### 2. scripts/train_rlan.py - Scheduler

```python
# Added "none" scheduler option
elif scheduler_type == 'none':
    scheduler = None
    print(f"Scheduler: None (constant LR={config.training.learning_rate})")
```

### 3. scripts/train_rlan.py - Checkpoint Handling

```python
# Fixed checkpoint save/load for None scheduler
if scheduler is not None:
    checkpoint['scheduler_state_dict'] = scheduler.state_dict()

# Load
if 'scheduler_state_dict' in checkpoint and scheduler is not None:
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
```

---

## Testing Commands

### Quick Validation
```bash
# Test simple task (should reach 100% in 2-5 epochs)
python scripts/test_prod_ablation.py --config ablation_stable --task d515da2d

# Test grid expansion task (should reach 100% in ~100 epochs)
python scripts/test_expansion_task.py 0  # f5b8619d: 6→12
python scripts/test_expansion_task.py 2  # 007bbfb7: 3→9 tiling
```

### Production Training
```bash
# Stable config (recommended for debugging)
python scripts/train_rlan.py --config configs/rlan_stable.yaml

# Minimal config (recommended for final training)
python scripts/train_rlan.py --config configs/rlan_minimal.yaml
```

---

## Next Steps

1. **Multi-task training**: Test if the stable config works across multiple tasks
2. **Caching strategy**: Implement sample caching in production dataloader
3. **Epoch scaling**: Harder tasks need more epochs - consider per-task adaptive stopping
4. **Context encoder**: Provide train_inputs/train_outputs to leverage ContextEncoder

---

## Appendix: Ablation Results

From `scripts/test_prod_ablation.py`:

| Config | Epochs to 100% | Notes |
|--------|----------------|-------|
| ablation_stable | 2 | Constant LR, minimal regularization |
| ablation_minimal | 3 | Cosine scheduler, minimal regularization |
| ablation_onecycle | ~5 | OneCycle, works but higher variance |

From `scripts/test_expansion_task.py`:

| Task | Description | Epochs | Notes |
|------|-------------|--------|-------|
| f5b8619d | 4x expansion | 94 | Variable output sizes |
| b91ae062 | 14x expansion | 98 | Mixed expansion ratios |
| 007bbfb7 | 8x tiling | 132 | Consistent 3x3→9x9 |

---

*Document generated from ablation study conducted on RLAN architecture.*
