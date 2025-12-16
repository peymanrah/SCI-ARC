# RLAN Training Guide for ARC-AGI

## Quick Start

```bash
# Base training (256 hidden, 1000 epochs)
python scripts/train_rlan.py --config configs/rlan_base.yaml

# Fair TRM comparison (512 hidden)
python scripts/train_rlan.py --config configs/rlan_fair.yaml

# Resume from checkpoint
python scripts/train_rlan.py --config configs/rlan_base.yaml --resume auto
```

---

## Single GPU Training: Disadvantages & Mitigations

### The Disadvantage

TRM trains with **768 global batch size across 8 GPUs** for 100K epochs.
RLAN on single GPU: **96 batch × 2 accumulation = 192 effective batch**.

| Aspect | Multi-GPU (TRM) | Single GPU (RLAN) |
|--------|-----------------|-------------------|
| Batch size | 768 | 192 |
| Gradient noise | Low (averaged) | Higher (noisier) |
| Training speed | Parallel | Sequential |
| Memory | 8×80GB | 24GB |

### How RLAN Compensates

1. **Infinite Augmentation** (2.9M variations per task)
   - TRM: Pre-computed 1000 augmentations
   - RLAN: On-the-fly generation, never repeats exact sample
   - **Effect**: Each epoch sees completely new data

2. **More Frequent Updates**
   - TRM: 768-sample gradient (very stable, slow adaptation)
   - RLAN: 192-sample gradient (noisier, faster adaptation)
   - **Effect**: Gradient noise helps escape local minima

3. **Structured Inductive Biases**
   - RLAN's DSC/MSRE/LCR/SPH provide strong priors
   - Requires fewer samples to learn patterns
   - **Effect**: 100× fewer epochs needed (1K vs 100K)

4. **In-Context Learning (ContextEncoder)**
   - Learns from training pairs, not puzzle memorization
   - Generalizes better from fewer examples
   - **Effect**: Better sample efficiency

### Optimal Single-GPU Settings

```yaml
training:
  batch_size: 96          # Maximize GPU memory usage
  grad_accumulation: 2    # Effective batch = 192
  max_epochs: 1000        # Sufficient with infinite augmentation
  learning_rate: 3e-4     # Higher LR for smaller batch
  
data:
  cache_samples: false    # CRITICAL: Enable infinite augmentation
  augmentation:
    color_permutation: true  # 362,880× diversity boost
```

---

## Debugging Background Collapse

### Symptoms
- `non_bg_accuracy` = 0%
- `bg_ratio_pred` >> `bg_ratio_target` (e.g., 99% vs 85%)
- `colors_used` = 1 (only predicting background)
- Model just copies input or outputs all zeros

### Causes & Fixes

| Cause | Detection | Fix |
|-------|-----------|-----|
| **ContextEncoder not working** | Check if train_inputs/outputs passed | Ensure collate_fn provides context |
| **Focal alpha too low** | BG dominates | Increase `focal_alpha` to 0.5-0.75 |
| **Learning rate too high** | Loss explodes then flatlines | Reduce LR by 2-5× |
| **DSC not learning** | `dsc_entropy` very high (>5) | Increase `lambda_entropy` |
| **No augmentation** | Overfits to training set | Enable `color_permutation: true` |

### Warning Signs in Logs

```
⚠️  [WARNING] BACKGROUND COLLAPSE DETECTED! (1/5)
    Reasons: BG excess: 15.2%, Non-BG acc: 0.5%
```

The training script will **automatically stop** after 5 consecutive warnings to prevent wasted compute.

---

## Metrics to Monitor

### Core Metrics (Must Improve)
| Metric | Good | Bad | Critical |
|--------|------|-----|----------|
| `task_accuracy` | > 0.1 | < 0.01 | 0 = complete failure |
| `non_bg_accuracy` | > 0.3 | < 0.1 | 0 = background collapse |
| `pixel_accuracy` | > 0.9 | < 0.8 | Misleading if BG dominates |

### Debugging Metrics (For Diagnosis)
| Metric | Meaning | Expected Range |
|--------|---------|----------------|
| `dsc_entropy` | Attention sharpness | 2-4 (lower = sharper) |
| `dsc_clues_used` | Spatial anchors active | 2-4 |
| `predicate_activation` | Symbolic reasoning | 0.1-0.5 |
| `colors_used` | Prediction diversity | Should match target |

### Loss Components
| Loss | Purpose | Watch For |
|------|---------|-----------|
| `focal_loss` | Main task loss | Should decrease |
| `entropy_loss` | Sharpen attention | Should decrease |
| `sparsity_loss` | Use fewer clues | Can increase early |
| `predicate_loss` | Decorrelate predicates | Stable ~0.1 |

---

## Config Validation Checklist

Before training, verify these settings:

### Model
- [ ] `hidden_dim`: 256 (base) or 512 (fair comparison)
- [ ] `num_solver_steps`: 6 (enough for complex tasks)
- [ ] `max_clues`: 5 (sufficient for most patterns)

### Training
- [ ] `max_epochs`: 1000 (with infinite augmentation)
- [ ] `batch_size`: 96 (for 24GB GPU)
- [ ] `grad_accumulation_steps`: 2 (effective = 192)
- [ ] `focal_alpha`: 0.25-0.5 (higher if collapsing)
- [ ] `focal_gamma`: 2.0 (standard)
- [ ] `scheduler`: "onecycle" (faster convergence)

### Data
- [ ] `cache_samples`: false (CRITICAL for diversity)
- [ ] `color_permutation`: true (362,880× boost)
- [ ] `num_workers`: 8 (use CPU cores)
- [ ] `prefetch_factor`: 4 (keep GPU fed)

### Device
- [ ] `mixed_precision`: true (AMP for speed)
- [ ] `dtype`: "bfloat16" (more stable)
- [ ] `compile`: true (20-40% speedup)

---

## Epoch Budget Recommendations

| Training Goal | Epochs | Time (RTX 3090) |
|--------------|--------|-----------------|
| Quick test | 10 | ~10 min |
| Sanity check | 100 | ~2 hours |
| Development | 500 | ~10 hours |
| Full training | 1000 | ~24 hours |
| Competition | 2000+ | 2-3 days |

### Early Stopping Heuristics

- If `task_accuracy` = 0 after 50 epochs → Something is wrong
- If `non_bg_accuracy` < 0.01 after 100 epochs → Background collapse
- If loss stops decreasing for 100 epochs → Try lower LR

---

## Advanced: Multi-GPU Training (Future)

If you have multiple GPUs, you can increase effective batch:

```python
# DDP wrapper (not yet implemented)
model = torch.nn.parallel.DistributedDataParallel(model)
```

Expected benefits:
- 2 GPUs: 2× batch → ~15% faster convergence
- 4 GPUs: 4× batch → ~25% faster convergence
- 8 GPUs: 8× batch → ~30% faster convergence (diminishing returns)

Note: RLAN's infinite augmentation already provides most of the diversity benefits, so multi-GPU mainly helps with speed, not accuracy.
