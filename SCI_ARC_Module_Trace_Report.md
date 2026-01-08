# SCI-ARC RLAN Production Module Trace Report

**Generated:** 2026-01-06  
**Checkpoint:** warmup3.pt (Epoch 10)  
**Device:** CPU (Testing)  
**Total Tasks Analyzed:** 6

---

## Executive Summary

### ✅ **NO BUGS DETECTED**

All 6 tasks processed successfully with **zero NaN values, zero Inf values, and healthy statistical distributions** across all modules.

### Key Findings

| Metric | Value | Assessment |
|--------|-------|------------|
| Tasks with Bugs | 0/6 | ✅ Clean |
| Total NaN Count | 0 | ✅ Numerically stable |
| Total Inf Count | 0 | ✅ No overflow |
| Mean First Step Accuracy | 74.54% | Baseline |
| Mean Last Step Accuracy | 75.94% | +1.40% improvement |
| Mean Solver Improvement | +1.40% | ⚠️ Modest |

---

## Module Order Verification (Production-Accurate)

The trace follows the **exact production forward pass order** from `sci_arc/models/rlan.py`:

```
1. GridEncoder.encode() → features (B, 256, H, W)
2. ContextEncoder → support_features (B, N, 256, 30, 30) + dsc_task_context (B, 256)
3. Features + Context Injection → enhanced features (B, 256, H, W)
4. DSC (Differentiable Slot Clustering):
   - 4a. centroids (B, 7, 2)
   - 4b. attention_maps (B, 7, H, W)
   - 4c. stop_logits (B, 7)
5. MSRE (Multi-Scale Reasoning Encoder) → clue_features (B, 7, 256, H, W)
6. [LCR - Disabled in warmup3.pt]
7. [SPH - Disabled in warmup3.pt]
8. [HyperLoRA - Disabled in warmup3.pt]
9. RecursiveSolver → step_i_logits (B, 10, H, W) for i = 1..6
```

---

## Per-Task Analysis

### Task 1: `007bbfb7` (Small - 3x3 → 9x9)

| Solver Step | Accuracy | Entropy | Confidence | Improvement |
|-------------|----------|---------|------------|-------------|
| Step 1 | 88.89% | 0.131 | 96.21% | +88.89% |
| Step 2 | 88.89% | 0.025 | 99.38% | 0.00% |
| Step 3 | 77.78% | 0.011 | 99.80% | **-11.11%** ⚠️ |
| Step 4 | 77.78% | 0.005 | 99.93% | 0.00% |
| Step 5 | 77.78% | 0.006 | 99.91% | 0.00% |
| Step 6 | 77.78% | 0.006 | 99.91% | 0.00% |

**Observation:** Accuracy **degrades** at step 3 and never recovers. The model becomes over-confident (99.93%) but incorrect.

### Task 2: `00d62c1b` (Small - 20x20)

| Solver Step | Accuracy | Entropy | Confidence | Improvement |
|-------------|----------|---------|------------|-------------|
| Step 1 | 81.00% | 0.152 | 93.75% | +81.00% |
| Step 2 | 96.00% | 0.059 | 97.76% | **+15.00%** ✅ |
| Step 3 | 96.25% | 0.033 | 98.90% | +0.25% |
| Step 4 | 97.25% | 0.023 | 99.18% | **+1.00%** ✅ |
| Step 5 | 96.75% | 0.023 | 99.14% | -0.50% |
| Step 6 | 96.75% | 0.022 | 99.24% | 0.00% |

**Observation:** Excellent progressive improvement! Peak at step 4 (97.25%).

### Task 3: `025d127b` (Medium - 10x10)

| Solver Step | Accuracy | Entropy | Confidence | Improvement |
|-------------|----------|---------|------------|-------------|
| Step 1 | 90.00% | 0.080 | 96.54% | +90.00% |
| Step 2 | 86.00% | 0.149 | 93.12% | **-4.00%** ⚠️ |
| Step 3 | 90.00% | 0.073 | 96.74% | +4.00% |
| Step 4 | 91.00% | 0.074 | 97.57% | +1.00% |
| Step 5 | 90.00% | 0.083 | 96.95% | -1.00% |
| Step 6 | 90.00% | 0.083 | 96.76% | 0.00% |

**Observation:** Step 2 shows temporary regression, but recovers.

### Task 4: `0520fde7` (Medium - 3x7 → 3x3)

| Solver Step | Accuracy | Entropy | Confidence | Improvement |
|-------------|----------|---------|------------|-------------|
| Step 1 | 22.22% | 0.073 | 97.65% | +22.22% |
| Step 2 | 22.22% | 0.056 | 98.52% | 0.00% |
| Step 3 | 22.22% | 0.075 | 97.50% | 0.00% |
| Step 4 | 33.33% | 0.098 | 95.94% | **+11.11%** ✅ |
| Step 5 | 33.33% | 0.079 | 97.63% | 0.00% |
| Step 6 | 33.33% | 0.070 | 98.16% | 0.00% |

**Observation:** Very low accuracy - model struggles with size-changing tasks. Only 1/3 correct.

### Task 5: `045e512c` (Large - 21x21)

| Solver Step | Accuracy | Entropy | Confidence | Improvement |
|-------------|----------|---------|------------|-------------|
| Step 1 | 85.26% | 0.115 | 96.72% | +85.26% |
| Step 2 | 82.99% | 0.060 | 97.86% | -2.27% |
| Step 3 | 83.90% | 0.057 | 97.87% | +0.91% |
| Step 4 | 78.68% | 0.099 | 96.26% | **-5.22%** ⚠️ |
| Step 5 | 84.35% | 0.061 | 97.75% | +5.67% |
| Step 6 | 82.77% | 0.070 | 97.27% | -1.59% |

**Observation:** Highly unstable - step 4 shows major regression, step 5 recovers. Oscillating behavior.

### Task 6: `0962bcdd` (Large - 12x12)

| Solver Step | Accuracy | Entropy | Confidence | Improvement |
|-------------|----------|---------|------------|-------------|
| Step 1 | 79.86% | 0.030 | 98.84% | +79.86% |
| Step 2 | 72.92% | 0.086 | 96.86% | **-6.94%** ⚠️ |
| Step 3 | 70.83% | 0.054 | 97.82% | -2.08% |
| Step 4 | 74.31% | 0.055 | 98.11% | +3.47% |
| Step 5 | 77.08% | 0.033 | 98.88% | +2.78% |
| Step 6 | 75.00% | 0.052 | 97.86% | -2.08% |

**Observation:** Step 1 is best! Progressive degradation until step 3, then partial recovery.

---

## Module Statistics Summary

### GridEncoder Features

| Task | Shape | Min | Max | Mean | Std |
|------|-------|-----|-----|------|-----|
| 007bbfb7 | (1, 256, 3, 3) | -0.711 | 4.357 | -0.002 | 1.011 |
| 00d62c1b | (1, 256, 20, 20) | -0.712 | 4.609 | -0.001 | 1.012 |
| 025d127b | (1, 256, 10, 10) | -0.711 | 4.560 | -0.001 | 1.011 |
| 0520fde7 | (1, 256, 3, 7) | -0.712 | 4.357 | -0.001 | 1.011 |
| 045e512c | (1, 256, 21, 21) | -0.711 | 4.652 | -0.001 | 1.012 |
| 0962bcdd | (1, 256, 12, 12) | -0.712 | 4.560 | -0.001 | 1.011 |

**Assessment:** ✅ All GridEncoder outputs show consistent statistics with mean ≈ 0 and std ≈ 1 (well-normalized).

### DSC Attention Maps

| Task | Shape | Min | Max | Mean | Std |
|------|-------|-----|-----|------|-----|
| 007bbfb7 | (1, 7, 3, 3) | 0.008 | 0.185 | 0.111 | 0.067 |
| 00d62c1b | (1, 7, 20, 20) | 0.0002 | 0.018 | 0.0025 | 0.004 |
| 025d127b | (1, 7, 10, 10) | 0.001 | 0.061 | 0.010 | 0.016 |
| 0520fde7 | (1, 7, 3, 7) | 0.003 | 0.087 | 0.048 | 0.034 |
| 045e512c | (1, 7, 21, 21) | 0.0005 | 0.043 | 0.002 | 0.005 |
| 0962bcdd | (1, 7, 12, 12) | 0.002 | 0.072 | 0.007 | 0.012 |

**Assessment:** ✅ Attention maps properly normalized (mean ≈ 1/HW). Larger grids have smaller mean attention (as expected).

### DSC Stop Logits

| Task | Min | Max | Mean | Std |
|------|-----|-----|------|-----|
| 007bbfb7 | 0.817 | 0.838 | 0.827 | 0.007 |
| 00d62c1b | 1.052 | 1.069 | 1.060 | 0.006 |
| 025d127b | 0.802 | 0.841 | 0.821 | 0.013 |
| 0520fde7 | 1.100 | 1.109 | 1.105 | 0.003 |
| 045e512c | 1.044 | 1.061 | 1.053 | 0.006 |
| 0962bcdd | 0.932 | 0.953 | 0.943 | 0.008 |

**Assessment:** ⚠️ All stop logits are positive (~0.8-1.1), meaning sigmoid(stop_logit) ≈ 0.69-0.75. This suggests the model predicts **all 7 clues should be used** for most tasks. No early stopping.

### MSRE Clue Features

| Task | Shape | Min | Max | Mean | Std |
|------|-------|-----|-----|------|-----|
| 007bbfb7 | (1, 7, 256, 3, 3) | -1.32 | 8.96 | -0.048 | 1.51 |
| 00d62c1b | (1, 7, 256, 20, 20) | -1.43 | 10.37 | -0.054 | 1.53 |
| 025d127b | (1, 7, 256, 10, 10) | -1.34 | 10.18 | -0.071 | 1.50 |
| 0520fde7 | (1, 7, 256, 3, 7) | -1.32 | 10.29 | -0.056 | 1.52 |
| 045e512c | (1, 7, 256, 21, 21) | -1.44 | 10.02 | -0.048 | 1.54 |
| 0962bcdd | (1, 7, 256, 12, 12) | -1.41 | 10.80 | -0.067 | 1.51 |

**Assessment:** ✅ Consistent feature distributions. Slightly higher std (1.5 vs 1.0) indicates feature expansion through MSRE.

### Solver Logits (Step 1 vs Step 6)

| Task | Step 1 Mean | Step 6 Mean | Δ Mean | Step 1 Std | Step 6 Std |
|------|-------------|-------------|--------|------------|------------|
| 007bbfb7 | -13.15 | -13.63 | -0.48 | 9.20 | 9.78 |
| 00d62c1b | -13.76 | -14.18 | -0.42 | 9.72 | 9.72 |
| 025d127b | -14.57 | -14.45 | +0.12 | 9.68 | 9.70 |
| 0520fde7 | -12.46 | -12.67 | -0.21 | 9.85 | 9.70 |
| 045e512c | -14.64 | -12.55 | +2.09 | 9.80 | 9.69 |
| 0962bcdd | -14.83 | -13.97 | +0.86 | 9.36 | 9.05 |

**Assessment:** ✅ Logit means become slightly less negative over solver iterations. Std stays stable (~9.5).

---

## Critical Analysis: Solver Improvement Patterns

### Pattern 1: "First Step Best" (2/6 tasks)

Tasks `007bbfb7` and `0962bcdd` show **maximum accuracy at step 1** with subsequent degradation.

```
007bbfb7: 88.89% → 77.78% (-11.11%)
0962bcdd: 79.86% → 75.00% (-4.86%)
```

**Root Cause Hypothesis:** 
- The recursive solver **over-refines** simple patterns
- Initial predictions capture the core transformation
- Subsequent iterations introduce noise

### Pattern 2: "Progressive Improvement" (1/6 tasks)

Only task `00d62c1b` shows consistent improvement:

```
00d62c1b: 81.00% → 96.00% → 97.25% (+16.25%)
```

**This is the expected behavior** - solver iterations should refine the solution.

### Pattern 3: "Unstable Oscillation" (3/6 tasks)

Tasks `025d127b`, `045e512c`, `0520fde7` show oscillating accuracy:

```
025d127b: 90% → 86% → 90% → 91% → 90%
045e512c: 85% → 83% → 84% → 79% → 84% → 83%
0520fde7: 22% → 22% → 22% → 33% (stuck at low accuracy)
```

---

## Recommendations

### 1. **Solver Step Selection Strategy**
Current approach uses **last step** output. Consider:
- Best-of-K selection using entropy or confidence
- Early stopping when accuracy plateaus
- Weighted ensemble of solver steps

### 2. **Size-Changing Task Handling**
Task `0520fde7` (3x7 → 3x3) shows poor accuracy (33%).
- The model may not properly handle input/output size mismatches
- Consider specialized handling for size-changing transformations

### 3. **Confidence Calibration**
Model shows **high confidence even when wrong**:
- Task `0520fde7`: 98% confidence but only 33% accurate
- Consider temperature scaling or focal loss for calibration

### 4. **DSC Stop Logits**
All stop logits are positive (~0.8-1.1), suggesting:
- Model always uses all 7 clues
- Early stopping mechanism may not be effective
- Consider training with explicit clue count supervision

---

## Visualizations Generated

The following PNG visualizations were generated for 4 tasks:

| Task | Files |
|------|-------|
| 007bbfb7 | module_stats.png, features.png, dsc_attention.png, solver_steps.png |
| 00d62c1b | module_stats.png, features.png, dsc_attention.png, solver_steps.png |
| 025d127b | module_stats.png, features.png, dsc_attention.png, solver_steps.png |
| 0520fde7 | module_stats.png, dsc_attention.png (partial) |

Location: `scripts/outputs/module_trace/<task_id>/`

---

## Conclusion

### Numerical Stability: ✅ PASS
- Zero NaN values across all modules
- Zero Inf values (no overflow)
- Consistent statistical distributions

### Module Order: ✅ VERIFIED
- Forward pass matches production codebase exactly
- GridEncoder → ContextEncoder → DSC → MSRE → Solver

### Solver Performance: ⚠️ NEEDS ATTENTION
- Only 1/6 tasks show consistent improvement
- 2/6 tasks show first-step degradation
- Mean improvement of only +1.4% across 6 solver steps

### Recommendations Priority:
1. **HIGH:** Implement best-step selection instead of last-step
2. **MEDIUM:** Investigate size-changing task handling
3. **LOW:** DSC stop logit calibration

---

*Report generated by production-accurate RLAN module tracing script*
