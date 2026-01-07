# RLAN Deep Diagnostic Report

**Date:** January 6, 2026  
**Checkpoint:** warmup3.pt (Epoch 10)  
**Tasks Tested:** 6 ARC tasks (small/medium/large)

---

## 🔴 CRITICAL BUG FOUND: DSC Slot Collapse

### Evidence

| Metric | Expected | Actual | Severity |
|--------|----------|--------|----------|
| Centroid max distance | > 0.5 | **0.063** | 🔴 CRITICAL |
| Attention map correlation | < 0.5 | **0.9934** | 🔴 CRITICAL |
| MSRE clue correlation | < 0.3 | **~1.0** | 🔴 CRITICAL |

### What This Means

1. **DSC is discovering 7 nearly identical spatial anchors** - all centroids are clustered around (0.38, 0.39) normalized coordinates

2. **Attention maps are 99.3% correlated** - all 7 "slots" are attending to the same region

3. **MSRE produces identical clue features** - since the relative coordinates are nearly identical for all clues, the clue features are indistinguishable

4. **The solver has NO diverse context** - it receives 7 copies of essentially the same feature, which is why accuracy doesn't improve across iterations

### Root Cause Analysis

The DSC (Dynamic Saliency Controller) is supposed to discover diverse spatial anchors using a differentiable soft k-means algorithm. However:

```
Expected behavior:
  Clue 0: (0.1, 0.1) - top-left corner
  Clue 1: (0.5, 0.5) - center
  Clue 2: (0.9, 0.9) - bottom-right
  ...

Actual behavior:
  Clue 0: (0.40, 0.42) - center-ish
  Clue 1: (0.37, 0.37) - center-ish  
  Clue 2: (0.38, 0.38) - center-ish
  ...
```

**Why this happens:**

## 🔧 ROOT CAUSE IDENTIFIED (Jan 6, 2026)

The `CentroidDiversityLoss` class EXISTS in `sci_arc/training/rlan_loss.py:978` and is correctly wired, BUT:

1. **`rlan_stable_prod.yaml` was MISSING `lambda_centroid_diversity`** 
2. **`train_rlan.py` defaults to 0.1** when not in config (`train_config.get('lambda_centroid_diversity', 0.1)`)
3. **0.1 is too weak** - documentation recommends 0.5 for collapse prevention

**Training log would have shown:**
```
Centroid Diversity: lambda=0.1 (prevents DSC clue collapse)
```

**But 0.1 is insufficient** - the loss function exists but wasn't strong enough to prevent mode collapse.

## ✅ FIX APPLIED

1. **Added to `rlan_stable_prod.yaml`:**
   ```yaml
   lambda_centroid_diversity: 0.5  # CRITICAL: Prevents DSC clue collapse
   ```

2. **Fixed `train_rlan.py` default from 0.1 → 0.3:**
   ```python
   lambda_centroid_diversity=train_config.get('lambda_centroid_diversity', 0.3),
   ```

3. **Warmup3.pt checkpoint is BROKEN** - trained with lambda=0.1, DSC already collapsed

**NEXT STEP:** Retrain from scratch with `lambda_centroid_diversity: 0.5`

---

## Previous Analysis (for reference)

**Why this was originally thought to happen:**
1. Initial centroid embeddings may not be diverse enough
2. No repulsion loss to prevent centroid clustering
3. Gradient signal may be too weak to push centroids apart
4. The attention softmax may have too high temperature (flat attention)

---

## Solver Step-by-Step Analysis

### Task-by-Task Results

| Task | Step 1 | Step 6 | Change | Pattern |
|------|--------|--------|--------|---------|
| 007bbfb7 (3x3→9x9) | 58.0% | 55.6% | **-2.4%** | 🔴 Degradation |
| 00d62c1b (20x20) | 77.5% | 95.0% | **+17.5%** | ✅ Improvement |
| 025d127b (10x10) | 66.0% | 76.0% | **+10%** | ⚠️ Peak at step 4 (84%), then drops |
| 0520fde7 (3x7→3x3) | N/A | N/A | N/A | Size mismatch issue |
| 045e512c (21x21) | ~85% | ~83% | **-2%** | 🔴 Unstable oscillation |
| 0962bcdd (12x12) | ~80% | ~75% | **-5%** | 🔴 First-step degradation |

### Why Some Tasks Improve Despite DSC Collapse

Task `00d62c1b` (flood fill) improves from 77.5% to 95% because:
- The task is relatively simple (fill holes with dominant color)
- The solver's GRU state evolution alone can capture this pattern
- Even with collapsed clues, the spatial context helps

Tasks that degrade are likely more complex transformations that require diverse spatial reasoning.

---

## Feature Quality Analysis

### Encoder Features ✅ HEALTHY
- **Dead channels:** 0/256
- **Channel correlation:** 0.574 (moderate, acceptable)
- All channels are active and produce diverse outputs

### MSRE Clue Features 🔴 COLLAPSED
- **Clue correlation:** ~1.0 (all clues identical)
- This is a direct consequence of DSC collapse
- The solver is effectively using 1 clue instead of 7

### DSC Attention Maps 🔴 COLLAPSED
- **Pairwise correlation:** 0.9934
- All slots attend to the same ~center region
- No spatial differentiation

---

## Visualizations Generated

All visualizations are in `scripts/outputs/deep_diagnosis/`:

| File | Description |
|------|-------------|
| `*_solver_diagnosis.png` | Step-by-step predictions with accuracy curves |
| `*_input_signals.png` | Encoder feature channels and clue feature heatmaps |
| `*_dsc_attention.png` | DSC slot attention maps (all identical) |
| `*_step_changes.png` | What pixels changed between solver steps |

### What the Visualizations Show

1. **Solver Diagnosis:** The accuracy curve is nearly flat or oscillating, not monotonically improving

2. **DSC Attention:** All 7 heatmaps look identical - centered soft attention with no differentiation

3. **Step Changes:** Very few pixels change between steps, suggesting the solver is stuck

---

## Recommended Fixes

### Fix 1: Centroid Diversity Loss (Priority: HIGH)

Add a repulsion loss to push centroids apart:

```python
def centroid_diversity_loss(centroids):
    """Encourage centroids to spread out."""
    B, K, _ = centroids.shape
    # Pairwise distances
    dists = torch.cdist(centroids, centroids)  # (B, K, K)
    # Mask diagonal
    mask = 1 - torch.eye(K, device=centroids.device)
    # Encourage minimum distance > threshold
    min_dist = 0.2  # At least 20% of grid apart
    loss = F.relu(min_dist - dists * mask).mean()
    return loss
```

### Fix 2: Orthogonal Centroid Initialization (Priority: MEDIUM)

Initialize centroid queries to be maximally spread:

```python
# Initialize centroids in a grid pattern
angles = torch.linspace(0, 2*np.pi, K+1)[:-1]
init_centroids = torch.stack([
    torch.cos(angles) * 0.4 + 0.5,  # Spread around center
    torch.sin(angles) * 0.4 + 0.5,
], dim=-1)
```

### Fix 3: Lower Attention Temperature (Priority: MEDIUM)

The attention may be too soft. Consider:
- Lower temperature in softmax (sharper attention)
- Use hard attention during evaluation

### Fix 4: Per-Clue Learnable Embeddings (Priority: LOW)

Add learnable per-clue embeddings to MSRE so clues are differentiated even if coordinates are similar:

```python
self.clue_embeddings = nn.Parameter(torch.randn(max_clues, hidden_dim) * 0.1)
# In forward:
clue_features = clue_features + self.clue_embeddings.view(1, K, D, 1, 1)
```

---

## Conclusion

The **primary bottleneck** in RLAN's reasoning is **DSC slot collapse**. The model has learned to place all 7 spatial anchors in the same location, which makes the MSRE clue features identical and prevents the solver from using diverse spatial context.

**Immediate action needed:**
1. Add centroid diversity loss during training
2. Re-train with diversity constraint
3. Monitor centroid spread during training

---

## Files Created

| Script | Purpose |
|--------|---------|
| `scripts/diagnose_rlan_deep.py` | Deep diagnostic with visualizations |
| `scripts/diagnose_clue_diversity.py` | DSC/MSRE collapse analysis |
| `scripts/outputs/deep_diagnosis/` | All visualization outputs |

---

*Report generated by automated diagnostic pipeline*
