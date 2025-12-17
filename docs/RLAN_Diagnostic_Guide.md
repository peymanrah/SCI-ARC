# RLAN Comprehensive Diagnostic Logging System

## Overview

This document defines the complete diagnostic signals needed to debug RLAN training issues. The goal is to make a single training log sufficient to identify the root cause of any learning problem.

## RLAN Data Flow & Critical Checkpoints

```
Input Grid (B, H, W)
    │
    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  1. ENCODER (Conv + Transformer)                                              │
│     ► CHECK: features finite? max/min/std reasonable?                         │
│     ► CHECK: gradient norm reaching encoder?                                  │
│     Output: features (B, D, H, W)                                             │
└──────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  2. CONTEXT ENCODER (Cross-attention on train pairs)                          │
│     ► CHECK: context vector magnitude (too small = no signal)                 │
│     ► CHECK: context variance across batch (zero = collapsed)                 │
│     ► CHECK: cross-attn entropy (uniform = not learning from examples)        │
│     Output: context (B, D)                                                    │
└──────────────────────────────────────────────────────────────────────────────┘
    │
    ├── Context injected via FiLM ──►
    │
    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  3. DSC (Dynamic Saliency Controller)                                         │
│     ► CHECK: attention_maps entropy (per clue, should decrease over epochs)   │
│     ► CHECK: attention_max (should increase, meaning sharper focus)           │
│     ► CHECK: centroid_spread (should be > 3, clues at different locations)    │
│     ► CHECK: stop_prob (should evolve, not stuck at init value)               │
│     ► CHECK: stop_logits gradient norm (is learning signal reaching?)         │
│     ► CHECK: entropy input to stop_predictor (coupling working?)              │
│     Output: centroids (B, K, 2), attention_maps (B, K, H, W), stop_logits (B, K)
└──────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  4. MSRE (Multi-Scale Relative Encoding)                                      │
│     ► CHECK: relative coordinates (are they centered on centroids?)           │
│     ► CHECK: encoding magnitude (not collapsed to zeros)                      │
│     ► CHECK: gradient norm reaching MSRE?                                     │
│     Output: clue_features (B, K, D, H, W)                                     │
└──────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  5. RECURSIVE SOLVER                                                          │
│     ► CHECK: per-step loss (should DECREASE across steps)                     │
│     ► CHECK: step-wise improvement (later steps better than earlier)          │
│     ► CHECK: residual magnitude (are updates meaningful?)                     │
│     ► CHECK: ACT halting (if enabled, is it learning when to stop?)           │
│     Output: logits (B, C, H, W), all_logits [list of (B, C, H, W)]            │
└──────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  6. LOSS COMPUTATION                                                          │
│     ► CHECK: task_loss (should decrease)                                      │
│     ► CHECK: class weights (are they balanced for bg/fg?)                     │
│     ► CHECK: entropy_loss (should decrease as attention sharpens)             │
│     ► CHECK: sparsity_loss components (min_clue, ponder, entropy_ponder)      │
│     ► CHECK: deep_supervision_loss (non-zero if intermediate steps help)      │
│     ► CHECK: loss_mode used (stablemax vs focal vs weighted)                  │
└──────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  7. BACKWARD PASS (Gradient Flow)                                             │
│     ► CHECK: total_grad_norm (should be reasonable, not exploding/vanishing)  │
│     ► CHECK: per-module grad norms (encoder, DSC, solver, context)            │
│     ► CHECK: any NaN/Inf in gradients?                                        │
│     ► CHECK: clipped gradients? how much?                                     │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Red Flags & Root Cause Mapping

| Symptom | Likely Root Cause | What to Check |
|---------|-------------------|---------------|
| Task accuracy = 0% after 50 epochs | Background collapse OR context not flowing | BG ratio, non-BG accuracy, context vector magnitude |
| Pixel acc high but task acc = 0 | Mostly predicting background correctly | Non-BG accuracy, colors used |
| DSC entropy not decreasing | Temperature too high OR no gradient to DSC | Temperature schedule, DSC grad norm |
| stop_prob stuck at init | Weak pondering gradient OR init_bias too strong | ponder_weight, init_bias, stop_logits grad |
| Per-step loss not improving | Solver not refining OR clues not informative | Clue features magnitude, residual magnitude |
| All clues at same location | Centroid diversity loss missing OR temperature too high | centroid_spread, temperature |
| Context vector near zero | Context encoder not learning | Cross-attention entropy, context grad norm |
| Sparsity loss = 0 | DSC using all clues (expected > min_clues) | DSC Clues Used, stop_prob |

## Diagnostic Signals to Log Every Epoch

### Tier 1: CRITICAL (Must Monitor Every Epoch)

```
=== EPOCH {N} DIAGNOSTIC SUMMARY ===

[LEARNING SIGNAL]
  Task Loss: {value} (should ↓)
  Pixel Accuracy: {value} (should ↑)
  Task Accuracy: {value} (should eventually > 0)
  Non-BG Accuracy: {value} (should ↑, not just BG)

[DSC HEALTH]
  Attention Entropy: {per_clue} (should ↓ toward ~2-3)
  Attention Max: {value} (should ↑ toward 0.1+)
  Stop Prob Mean: {value} (should evolve from init)
  Clues Used: {value} (should eventually < max_clues)
  Centroid Spread: {value} (should > 3)

[GRADIENT FLOW]
  DSC Grad Norm: {value} (must be > 0.001)
  Solver Grad Norm: {value} (must be > 0.001)
  Context Grad Norm: {value} (must be > 0.001 if enabled)
  
[CLASS BALANCE]
  BG Ratio (pred/target): {value}% / {value}%
  Colors Predicted: {list}
  FG Weight Used: {value} (for weighted_stablemax)
```

### Tier 2: IMPORTANT (Every 5-10 Epochs)

```
[SOLVER DYNAMICS]
  Per-Step Loss: [step1, step2, ..., stepN]
  Step Improvement: {first - last / first}%
  Residual Magnitude: {per_step}

[LOSS BREAKDOWN]
  Task Loss: {value} × 1.0 = {contribution}
  Entropy Loss: {value} × {weight} = {contribution}
  Sparsity Loss: {value} × {weight} = {contribution}
    - min_clue_penalty: {value}
    - base_pondering: {value}
    - entropy_pondering: {value}
  Deep Supervision: {value} × {weight} = {contribution}

[CONTEXT ENCODER] (if enabled)
  Context Vector Magnitude: {mean, std}
  Context Variance (across batch): {value}
  Cross-Attn Entropy: {value}

[ATTENTION QUALITY]
  Per-Clue Entropy: [clue1, clue2, ..., clueK]
  Per-Clue Max: [clue1, clue2, ..., clueK]
  Entropy Input to Stop Predictor: {mean}
```

### Tier 3: DEEP DEBUG (On-Demand or When Issues Detected)

```
[NUMERICAL STABILITY]
  Logits Range: [{min}, {max}]
  Any NaN in forward: {true/false}
  Any Inf in forward: {true/false}
  Gradient Clipping Applied: {count} times

[WEIGHT STATISTICS]
  Encoder Weight Norm: {value}
  DSC Weight Norm: {value}
  Solver Weight Norm: {value}
  Stop Predictor Weights: mean={value}, std={value}

[ACTIVATION DISTRIBUTIONS]
  Features: mean={value}, std={value}
  Context: mean={value}, std={value}
  Clue Features: mean={value}, std={value}
  
[ATTENTION INTERNALS]
  Query Norm: {value}
  Key Norm: {value}
  Attention Scores Pre-Softmax: mean={value}, std={value}
  Gumbel Noise Scale: {value}
```

## Warning Triggers (Auto-Alert in Logs)

```python
# These should trigger [WARNING] in logs automatically:

if dsc_grad_norm < 0.001 and epoch > 5:
    log("[WARNING] DSC gradients near zero - not learning!")

if entropy_mean > 5.0 and epoch > 25:
    log("[WARNING] Attention still diffuse after 25 epochs - check temperature!")

if stop_prob_mean unchanged from init (±0.02) and epoch > 20:
    log("[WARNING] Stop probability not evolving - check ponder_weight and init_bias!")

if centroid_spread < 2.0 and epoch > 20:
    log("[WARNING] Clues clustered together - not finding diverse anchors!")

if per_step_loss[-1] > per_step_loss[0] and epoch > 10:
    log("[WARNING] Solver making predictions WORSE - check residual connections!")

if task_accuracy == 0 and pixel_accuracy > 0.8 and epoch > 30:
    log("[WARNING] Background collapse detected - check class weighting!")

if non_bg_accuracy < 0.05 and epoch > 30:
    log("[WARNING] Not learning foreground at all - check weighted_stablemax!")

if context_magnitude < 0.1:
    log("[WARNING] Context vector near zero - ContextEncoder not contributing!")

if any(isnan(grad) or isinf(grad)):
    log("[CRITICAL] NaN/Inf in gradients - numerical instability!")
```
