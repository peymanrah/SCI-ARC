# RLAN Visual Demo Training Log

**Date:** 2025-12-17 22:07:53
**Device:** cpu
**Max Epochs:** 200
**Learning Rate:** 0.0005

## Examples

1. **Object Movement (4x4):** Move gray(5) to red(2) marker position
2. **Pattern Tiling (2x2→6x6):** Tile pattern with alternating rotation
3. **Conditional Logic (3x3):** Fill transformation

## Model Configuration

- hidden_dim: 256
- max_clues: 4
- num_solver_steps: 4
- use_dsc: True
- use_msre: True
- Parameters: 12,113,772

## Loss Configuration

- loss_mode: weighted_stablemax
- lambda_sparsity: 0.1
- min_clues: 1.0
- min_clue_weight: 5.0
- lambda_deep_supervision: 0.5

---

# Training Progress

## Epoch 1

**Temperature:** 0.9965

### Losses
| Loss | Value |
|------|-------|
| total_loss | 1.229038 |
| task_loss | 0.818003 |
| focal_loss | 0.818003 |
| entropy_loss | 2.648053 |
| sparsity_loss | 0.055593 |
| predicate_loss | 0.000000 |
| curriculum_loss | 1.061338 |
| deep_supervision_loss | 0.810952 |
| act_loss | 0.000000 |
| loss_mode | weighted_stablemax |
| sparsity_min_clue_penalty | 0.000000 |
| sparsity_base_pondering | 0.024260 |
| sparsity_entropy_pondering | 0.031333 |
| expected_clues_used | 2.425976 |
| stop_prob_from_loss | 0.393506 |
| clues_used_std | 0.163526 |
| per_sample_clue_penalty_mean | 0.000000 |

### Metrics
- **Total Accuracy:** 11.98%
- **BG Accuracy:** 13.70%
- **FG Accuracy:** 6.52%

### Per-Example Metrics
| Example | Accuracy | BG Acc | FG Acc | Exact Match | Target Classes | Pred Classes |
|---------|----------|--------|--------|-------------|----------------|---------------|
| 1 | 17.2% | 17.5% | 0.0% | ❌ | [0, 5] | [0, 2, 3, 4, 5, 6, 7, 8] |
| 2 | 9.4% | 10.7% | 8.3% | ❌ | [0, 2, 3, 7, 8] | [0, 2, 3, 4, 5, 7, 8, 9] |
| 3 | 9.4% | 10.9% | 0.0% | ❌ | [0, 1, 2] | [0, 1, 2, 3, 4, 5, 6, 7, 8] |

### DSC Analysis
| Example | Clues Used | Stop Probs | Stop Logits |
|---------|------------|------------|-------------|
| 1 | 2.34 | [0.334, 0.716, 0.429, 0.183] | [-0.69, 0.92, -0.29, -1.50] |
| 2 | 2.32 | [0.479, 0.485, 0.362, 0.349] | [-0.08, -0.06, -0.57, -0.62] |
| 3 | 2.61 | [0.315, 0.420, 0.440, 0.210] | [-0.78, -0.32, -0.24, -1.32] |

### Attention Entropy (per clue)
| Example | Clue 0 | Clue 1 | Clue 2 | Clue 3 | Clue 4 | Clue 5 |
|---------|--------|--------|--------|--------|--------|--------|
| 1 | 0.759 | 0.689 | 0.503 | 0.636 |
| 2 | 0.392 | 0.532 | 0.708 | 0.833 |
| 3 | 0.885 | 0.196 | 0.831 | 0.676 |

### Gradient Norms (selected modules)
- **encoder:** 0.252822
- **feature_proj:** 0.417249
- **context_encoder:** 0.950121
- **context_injector:** 0.189357
- **dsc:** 0.357423
- **msre:** 0.376285
- **solver:** 4.853251

### Predictions vs Targets

**Example 1:**

Target:
```
0 0 0 0
0 0 0 0
0 0 0 0
0 0 0 5
```

Prediction:
```
0 0 0 5
3 3 3 3
4 5 3 4
4 3 7 3
```

**Example 2:**

Target:
```
3 2 3 2 3 2
7 8 7 8 7 8
2 3 2 3 2 3
8 7 8 7 8 7
3 2 3 2 3 2
7 8 7 8 7 8
```

Prediction:
```
9 2 0 0 0 0
2 5 0 3 5 3
7 0 3 4 3 4
0 3 3 3 3 3
8 3 3 7 2 2
3 5 2 3 3 2
```

**Example 3:**

Target:
```
2 2 2
1 1 1
1 1 1
```

Prediction:
```
4 6 5
2 3 0
7 0 0
```


## Epoch 2

**Temperature:** 0.9931

### Losses
| Loss | Value |
|------|-------|
| total_loss | 0.943309 |
| task_loss | 0.622430 |
| focal_loss | 0.622430 |
| entropy_loss | 2.630549 |
| sparsity_loss | 0.075294 |
| predicate_loss | 0.000000 |
| curriculum_loss | 2.062506 |
| deep_supervision_loss | 0.626698 |
| act_loss | 0.000000 |
| loss_mode | weighted_stablemax |
| sparsity_min_clue_penalty | 0.000000 |
| sparsity_base_pondering | 0.033152 |
| sparsity_entropy_pondering | 0.042141 |
| expected_clues_used | 3.315236 |
| stop_prob_from_loss | 0.171191 |
| clues_used_std | 0.124663 |
| per_sample_clue_penalty_mean | 0.000000 |

### Metrics
- **Total Accuracy:** 70.83%
- **BG Accuracy:** 88.36%
- **FG Accuracy:** 15.22%

### Per-Example Metrics
| Example | Accuracy | BG Acc | FG Acc | Exact Match | Target Classes | Pred Classes |
|---------|----------|--------|--------|-------------|----------------|---------------|
| 1 | 84.4% | 85.7% | 0.0% | ❌ | [0, 5] | [0, 2, 8] |
| 2 | 48.4% | 96.4% | 11.1% | ❌ | [0, 2, 3, 7, 8] | [0, 2, 8] |
| 3 | 79.7% | 87.3% | 33.3% | ❌ | [0, 1, 2] | [0, 2, 8] |

### DSC Analysis
| Example | Clues Used | Stop Probs | Stop Logits |
|---------|------------|------------|-------------|
| 1 | 3.30 | [0.154, 0.153, 0.124, 0.268] | [-1.71, -1.71, -1.96, -1.01] |
| 2 | 3.45 | [0.156, 0.122, 0.161, 0.115] | [-1.69, -1.97, -1.65, -2.04] |
| 3 | 3.20 | [0.219, 0.216, 0.203, 0.164] | [-1.27, -1.29, -1.37, -1.63] |

### Attention Entropy (per clue)
| Example | Clue 0 | Clue 1 | Clue 2 | Clue 3 | Clue 4 | Clue 5 |
|---------|--------|--------|--------|--------|--------|--------|
| 1 | 0.775 | 0.707 | 0.787 | 0.647 |
| 2 | 0.764 | 0.677 | 0.298 | 0.675 |
| 3 | 0.395 | 0.588 | 0.531 | 0.747 |

### Gradient Norms (selected modules)
- **encoder:** 0.059588
- **feature_proj:** 0.102555
- **context_encoder:** 0.120261
- **context_injector:** 0.052242
- **dsc:** 0.097725
- **msre:** 0.085706
- **solver:** 1.335707


## Epoch 3

**Temperature:** 0.9897

### Losses
| Loss | Value |
|------|-------|
| total_loss | 0.928230 |
| task_loss | 0.611157 |
| focal_loss | 0.611157 |
| entropy_loss | 2.583061 |
| sparsity_loss | 0.061626 |
| predicate_loss | 0.000000 |
| curriculum_loss | 1.610945 |
| deep_supervision_loss | 0.621821 |
| act_loss | 0.000000 |
| loss_mode | weighted_stablemax |
| sparsity_min_clue_penalty | 0.000000 |
| sparsity_base_pondering | 0.027611 |
| sparsity_entropy_pondering | 0.034015 |
| expected_clues_used | 2.761109 |
| stop_prob_from_loss | 0.309723 |
| clues_used_std | 0.129944 |
| per_sample_clue_penalty_mean | 0.000000 |

### Metrics
- **Total Accuracy:** 42.71%
- **BG Accuracy:** 44.52%
- **FG Accuracy:** 36.96%

### Per-Example Metrics
| Example | Accuracy | BG Acc | FG Acc | Exact Match | Target Classes | Pred Classes |
|---------|----------|--------|--------|-------------|----------------|---------------|
| 1 | 34.4% | 34.9% | 0.0% | ❌ | [0, 5] | [0, 1, 2, 7] |
| 2 | 46.9% | 78.6% | 22.2% | ❌ | [0, 2, 3, 7, 8] | [0, 1, 2, 7] |
| 3 | 46.9% | 38.2% | 100.0% | ❌ | [0, 1, 2] | [0, 1, 2, 7] |

### DSC Analysis
| Example | Clues Used | Stop Probs | Stop Logits |
|---------|------------|------------|-------------|
| 1 | 2.91 | [0.187, 0.344, 0.211, 0.349] | [-1.47, -0.65, -1.32, -0.62] |
| 2 | 2.66 | [0.279, 0.203, 0.364, 0.491] | [-0.95, -1.37, -0.56, -0.04] |
| 3 | 2.71 | [0.211, 0.401, 0.410, 0.266] | [-1.32, -0.40, -0.36, -1.01] |

### Attention Entropy (per clue)
| Example | Clue 0 | Clue 1 | Clue 2 | Clue 3 | Clue 4 | Clue 5 |
|---------|--------|--------|--------|--------|--------|--------|
| 1 | 0.207 | 0.490 | 0.815 | 0.355 |
| 2 | 0.318 | 0.747 | 0.793 | 0.870 |
| 3 | 0.863 | 0.798 | 0.490 | 0.707 |

### Gradient Norms (selected modules)
- **encoder:** 0.036979
- **feature_proj:** 0.063134
- **context_encoder:** 0.082130
- **context_injector:** 0.040048
- **dsc:** 0.045633
- **msre:** 0.070734
- **solver:** 1.480526


## Epoch 4

**Temperature:** 0.9862

### Losses
| Loss | Value |
|------|-------|
| total_loss | 0.853131 |
| task_loss | 0.565381 |
| focal_loss | 0.565381 |
| entropy_loss | 2.375450 |
| sparsity_loss | 0.034387 |
| predicate_loss | 0.000000 |
| curriculum_loss | 0.423857 |
| deep_supervision_loss | 0.568622 |
| act_loss | 0.000000 |
| loss_mode | weighted_stablemax |
| sparsity_min_clue_penalty | 0.000000 |
| sparsity_base_pondering | 0.016165 |
| sparsity_entropy_pondering | 0.018223 |
| expected_clues_used | 1.616482 |
| stop_prob_from_loss | 0.595879 |
| clues_used_std | 0.292192 |
| per_sample_clue_penalty_mean | 0.000000 |

### Metrics
- **Total Accuracy:** 71.35%
- **BG Accuracy:** 84.93%
- **FG Accuracy:** 28.26%

### Per-Example Metrics
| Example | Accuracy | BG Acc | FG Acc | Exact Match | Target Classes | Pred Classes |
|---------|----------|--------|--------|-------------|----------------|---------------|
| 1 | 79.7% | 81.0% | 0.0% | ❌ | [0, 5] | [0, 2, 7] |
| 2 | 51.6% | 96.4% | 16.7% | ❌ | [0, 2, 3, 7, 8] | [0, 1, 2, 7, 8] |
| 3 | 82.8% | 83.6% | 77.8% | ❌ | [0, 1, 2] | [0, 1, 2, 7, 8] |

### DSC Analysis
| Example | Clues Used | Stop Probs | Stop Logits |
|---------|------------|------------|-------------|
| 1 | 1.88 | [0.678, 0.471, 0.505, 0.463] | [0.75, -0.12, 0.02, -0.15] |
| 2 | 1.66 | [0.776, 0.634, 0.447, 0.480] | [1.24, 0.55, -0.21, -0.08] |
| 3 | 1.30 | [0.629, 0.690, 0.662, 0.714] | [0.53, 0.80, 0.67, 0.92] |

### Attention Entropy (per clue)
| Example | Clue 0 | Clue 1 | Clue 2 | Clue 3 | Clue 4 | Clue 5 |
|---------|--------|--------|--------|--------|--------|--------|
| 1 | 0.835 | 0.698 | 0.173 | 0.376 |
| 2 | 0.499 | 0.794 | 0.640 | 0.740 |
| 3 | 0.221 | 0.652 | 0.631 | 0.595 |

### Gradient Norms (selected modules)
- **encoder:** 0.046317
- **feature_proj:** 0.080105
- **context_encoder:** 0.094567
- **context_injector:** 0.056885
- **dsc:** 0.125180
- **msre:** 0.061251
- **solver:** 0.908592


## Epoch 5

**Temperature:** 0.9828

### Losses
| Loss | Value |
|------|-------|
| total_loss | 1.084928 |
| task_loss | 0.810559 |
| focal_loss | 0.810559 |
| entropy_loss | 2.696430 |
| sparsity_loss | 0.010563 |
| predicate_loss | 0.000000 |
| curriculum_loss | 0.027003 |
| deep_supervision_loss | 0.546626 |
| act_loss | 0.000000 |
| loss_mode | weighted_stablemax |
| sparsity_min_clue_penalty | 0.542966 |
| sparsity_base_pondering | 0.004570 |
| sparsity_entropy_pondering | 0.005993 |
| expected_clues_used | 0.457034 |
| stop_prob_from_loss | 0.885741 |
| clues_used_std | 0.128313 |
| per_sample_clue_penalty_mean | 0.271483 |

### Metrics
- **Total Accuracy:** 68.75%
- **BG Accuracy:** 80.82%
- **FG Accuracy:** 30.43%

### Per-Example Metrics
| Example | Accuracy | BG Acc | FG Acc | Exact Match | Target Classes | Pred Classes |
|---------|----------|--------|--------|-------------|----------------|---------------|
| 1 | 75.0% | 76.2% | 0.0% | ❌ | [0, 5] | [0, 2, 7, 8] |
| 2 | 53.1% | 92.9% | 22.2% | ❌ | [0, 2, 3, 7, 8] | [0, 1, 2, 3, 7, 8] |
| 3 | 78.1% | 80.0% | 66.7% | ❌ | [0, 1, 2] | [0, 1, 2, 7, 8] |

### DSC Analysis
| Example | Clues Used | Stop Probs | Stop Logits |
|---------|------------|------------|-------------|
| 1 | 0.40 | [0.914, 0.885, 0.895, 0.907] | [2.37, 2.04, 2.14, 2.28] |
| 2 | 0.60 | [0.906, 0.814, 0.844, 0.832] | [2.27, 1.48, 1.68, 1.60] |
| 3 | 0.37 | [0.911, 0.927, 0.911, 0.882] | [2.33, 2.55, 2.32, 2.01] |

### Attention Entropy (per clue)
| Example | Clue 0 | Clue 1 | Clue 2 | Clue 3 | Clue 4 | Clue 5 |
|---------|--------|--------|--------|--------|--------|--------|
| 1 | 0.673 | 0.823 | 0.279 | 0.767 |
| 2 | 0.789 | 0.842 | 0.522 | 0.710 |
| 3 | 0.414 | 0.750 | 0.673 | 0.539 |

### Gradient Norms (selected modules)
- **encoder:** 0.959119
- **feature_proj:** 1.676346
- **context_encoder:** 1.931357
- **context_injector:** 1.345950
- **dsc:** 3.670722
- **msre:** 0.043251
- **solver:** 1.038471


## Epoch 6

**Temperature:** 0.9794

### Losses
| Loss | Value |
|------|-------|
| total_loss | 0.761756 |
| task_loss | 0.498702 |
| focal_loss | 0.498702 |
| entropy_loss | 2.046156 |
| sparsity_loss | 0.053276 |
| predicate_loss | 0.000000 |
| curriculum_loss | 1.297224 |
| deep_supervision_loss | 0.515454 |
| act_loss | 0.000000 |
| loss_mode | weighted_stablemax |
| sparsity_min_clue_penalty | 0.000000 |
| sparsity_base_pondering | 0.026640 |
| sparsity_entropy_pondering | 0.026636 |
| expected_clues_used | 2.664023 |
| stop_prob_from_loss | 0.333994 |
| clues_used_std | 0.230344 |
| per_sample_clue_penalty_mean | 0.000000 |

### Metrics
- **Total Accuracy:** 66.15%
- **BG Accuracy:** 72.60%
- **FG Accuracy:** 45.65%

### Per-Example Metrics
| Example | Accuracy | BG Acc | FG Acc | Exact Match | Target Classes | Pred Classes |
|---------|----------|--------|--------|-------------|----------------|---------------|
| 1 | 62.5% | 63.5% | 0.0% | ❌ | [0, 5] | [0, 1, 2, 7, 8] |
| 2 | 59.4% | 92.9% | 33.3% | ❌ | [0, 2, 3, 7, 8] | [0, 1, 2, 3, 7, 8] |
| 3 | 76.6% | 72.7% | 100.0% | ❌ | [0, 1, 2] | [0, 1, 2, 8] |

### DSC Analysis
| Example | Clues Used | Stop Probs | Stop Logits |
|---------|------------|------------|-------------|
| 1 | 2.44 | [0.414, 0.410, 0.389, 0.347] | [-0.35, -0.36, -0.45, -0.63] |
| 2 | 2.65 | [0.374, 0.298, 0.339, 0.338] | [-0.52, -0.86, -0.67, -0.67] |
| 3 | 2.90 | [0.241, 0.270, 0.373, 0.216] | [-1.15, -0.99, -0.52, -1.29] |

### Attention Entropy (per clue)
| Example | Clue 0 | Clue 1 | Clue 2 | Clue 3 | Clue 4 | Clue 5 |
|---------|--------|--------|--------|--------|--------|--------|
| 1 | 0.770 | 0.288 | 0.427 | 0.765 |
| 2 | 0.353 | 0.795 | 0.262 | 0.305 |
| 3 | 0.737 | 0.739 | 0.019 | 0.445 |


## Epoch 7

**Temperature:** 0.9760

### Losses
| Loss | Value |
|------|-------|
| total_loss | 0.733146 |
| task_loss | 0.478524 |
| focal_loss | 0.478524 |
| entropy_loss | 2.643575 |
| sparsity_loss | 0.082839 |
| predicate_loss | 0.000000 |
| curriculum_loss | 2.278578 |
| deep_supervision_loss | 0.492675 |
| act_loss | 0.000000 |
| loss_mode | weighted_stablemax |
| sparsity_min_clue_penalty | 0.000000 |
| sparsity_base_pondering | 0.036449 |
| sparsity_entropy_pondering | 0.046389 |
| expected_clues_used | 3.644933 |
| stop_prob_from_loss | 0.088767 |
| clues_used_std | 0.043868 |
| per_sample_clue_penalty_mean | 0.000000 |

### Metrics
- **Total Accuracy:** 51.04%
- **BG Accuracy:** 53.42%
- **FG Accuracy:** 43.48%

### Per-Example Metrics
| Example | Accuracy | BG Acc | FG Acc | Exact Match | Target Classes | Pred Classes |
|---------|----------|--------|--------|-------------|----------------|---------------|
| 1 | 40.6% | 41.3% | 0.0% | ❌ | [0, 5] | [0, 1, 2, 7] |
| 2 | 57.8% | 92.9% | 30.6% | ❌ | [0, 2, 3, 7, 8] | [0, 1, 2, 7] |
| 3 | 54.7% | 47.3% | 100.0% | ❌ | [0, 1, 2] | [0, 1, 2, 7] |

### DSC Analysis
| Example | Clues Used | Stop Probs | Stop Logits |
|---------|------------|------------|-------------|
| 1 | 3.60 | [0.131, 0.104, 0.075, 0.089] | [-1.89, -2.16, -2.51, -2.32] |
| 2 | 3.69 | [0.086, 0.097, 0.077, 0.051] | [-2.37, -2.23, -2.48, -2.92] |
| 3 | 3.65 | [0.195, 0.060, 0.048, 0.051] | [-1.42, -2.76, -2.98, -2.93] |

### Attention Entropy (per clue)
| Example | Clue 0 | Clue 1 | Clue 2 | Clue 3 | Clue 4 | Clue 5 |
|---------|--------|--------|--------|--------|--------|--------|
| 1 | 0.607 | 0.360 | 0.447 | 0.582 |
| 2 | 0.752 | 0.359 | 0.752 | 0.801 |
| 3 | 0.778 | 0.785 | 0.775 | 0.630 |


## Epoch 8

**Temperature:** 0.9727

### Losses
| Loss | Value |
|------|-------|
| total_loss | 0.685223 |
| task_loss | 0.446516 |
| focal_loss | 0.446516 |
| entropy_loss | 2.000571 |
| sparsity_loss | 0.075367 |
| predicate_loss | 0.000000 |
| curriculum_loss | 2.641215 |
| deep_supervision_loss | 0.462341 |
| act_loss | 0.000000 |
| loss_mode | weighted_stablemax |
| sparsity_min_clue_penalty | 0.000000 |
| sparsity_base_pondering | 0.038333 |
| sparsity_entropy_pondering | 0.037034 |
| expected_clues_used | 3.833288 |
| stop_prob_from_loss | 0.041678 |
| clues_used_std | 0.020226 |
| per_sample_clue_penalty_mean | 0.000000 |

### Metrics
- **Total Accuracy:** 67.71%
- **BG Accuracy:** 73.29%
- **FG Accuracy:** 50.00%

### Per-Example Metrics
| Example | Accuracy | BG Acc | FG Acc | Exact Match | Target Classes | Pred Classes |
|---------|----------|--------|--------|-------------|----------------|---------------|
| 1 | 67.2% | 68.3% | 0.0% | ❌ | [0, 5] | [0, 2, 7] |
| 2 | 62.5% | 92.9% | 38.9% | ❌ | [0, 2, 3, 7, 8] | [0, 1, 2, 3, 7] |
| 3 | 73.4% | 69.1% | 100.0% | ❌ | [0, 1, 2] | [0, 1, 2, 7] |

### DSC Analysis
| Example | Clues Used | Stop Probs | Stop Logits |
|---------|------------|------------|-------------|
| 1 | 3.81 | [0.036, 0.050, 0.049, 0.055] | [-3.30, -2.95, -2.96, -2.85] |
| 2 | 3.84 | [0.037, 0.036, 0.050, 0.035] | [-3.27, -3.28, -2.95, -3.33] |
| 3 | 3.85 | [0.037, 0.053, 0.032, 0.032] | [-3.25, -2.89, -3.42, -3.42] |

### Attention Entropy (per clue)
| Example | Clue 0 | Clue 1 | Clue 2 | Clue 3 | Clue 4 | Clue 5 |
|---------|--------|--------|--------|--------|--------|--------|
| 1 | 0.580 | 0.283 | 0.122 | 0.189 |
| 2 | 0.629 | 0.757 | 0.160 | 0.651 |
| 3 | 0.635 | 0.268 | 0.687 | 0.812 |


## Epoch 9

**Temperature:** 0.9693

### Losses
| Loss | Value |
|------|-------|
| total_loss | 0.612416 |
| task_loss | 0.396019 |
| focal_loss | 0.396019 |
| entropy_loss | 2.418802 |
| sparsity_loss | 0.083700 |
| predicate_loss | 0.000000 |
| curriculum_loss | 2.655472 |
| deep_supervision_loss | 0.416052 |
| act_loss | 0.000000 |
| loss_mode | weighted_stablemax |
| sparsity_min_clue_penalty | 0.000000 |
| sparsity_base_pondering | 0.038669 |
| sparsity_entropy_pondering | 0.045031 |
| expected_clues_used | 3.866880 |
| stop_prob_from_loss | 0.033280 |
| clues_used_std | 0.022660 |
| per_sample_clue_penalty_mean | 0.000000 |

### Metrics
- **Total Accuracy:** 64.06%
- **BG Accuracy:** 60.96%
- **FG Accuracy:** 73.91%

### Per-Example Metrics
| Example | Accuracy | BG Acc | FG Acc | Exact Match | Target Classes | Pred Classes |
|---------|----------|--------|--------|-------------|----------------|---------------|
| 1 | 51.6% | 52.4% | 0.0% | ❌ | [0, 5] | [0, 2, 3, 7, 8] |
| 2 | 81.2% | 96.4% | 69.4% | ❌ | [0, 2, 3, 7, 8] | [0, 1, 2, 3, 7, 8] |
| 3 | 59.4% | 52.7% | 100.0% | ❌ | [0, 1, 2] | [0, 1, 2, 3, 7, 8] |

### DSC Analysis
| Example | Clues Used | Stop Probs | Stop Logits |
|---------|------------|------------|-------------|
| 1 | 3.87 | [0.048, 0.027, 0.027, 0.033] | [-2.98, -3.60, -3.57, -3.39] |
| 2 | 3.85 | [0.051, 0.027, 0.037, 0.040] | [-2.92, -3.58, -3.26, -3.19] |
| 3 | 3.89 | [0.028, 0.029, 0.029, 0.023] | [-3.55, -3.50, -3.50, -3.75] |

### Attention Entropy (per clue)
| Example | Clue 0 | Clue 1 | Clue 2 | Clue 3 | Clue 4 | Clue 5 |
|---------|--------|--------|--------|--------|--------|--------|
| 1 | 0.186 | 0.598 | 0.591 | 0.461 |
| 2 | 0.565 | 0.459 | 0.714 | 0.656 |
| 3 | 0.748 | 0.594 | 0.630 | 0.777 |


## Epoch 10

**Temperature:** 0.9659

### Losses
| Loss | Value |
|------|-------|
| total_loss | 0.601148 |
| task_loss | 0.390951 |
| focal_loss | 0.390951 |
| entropy_loss | 2.306275 |
| sparsity_loss | 0.082168 |
| predicate_loss | 0.000000 |
| curriculum_loss | 2.695194 |
| deep_supervision_loss | 0.403959 |
| act_loss | 0.000000 |
| loss_mode | weighted_stablemax |
| sparsity_min_clue_penalty | 0.000000 |
| sparsity_base_pondering | 0.038953 |
| sparsity_entropy_pondering | 0.043215 |
| expected_clues_used | 3.895328 |
| stop_prob_from_loss | 0.026168 |
| clues_used_std | 0.005720 |
| per_sample_clue_penalty_mean | 0.000000 |

### Metrics
- **Total Accuracy:** 75.00%
- **BG Accuracy:** 79.45%
- **FG Accuracy:** 60.87%

### Per-Example Metrics
| Example | Accuracy | BG Acc | FG Acc | Exact Match | Target Classes | Pred Classes |
|---------|----------|--------|--------|-------------|----------------|---------------|
| 1 | 76.6% | 77.8% | 0.0% | ❌ | [0, 5] | [0, 2, 7, 8] |
| 2 | 73.4% | 100.0% | 52.8% | ❌ | [0, 2, 3, 7, 8] | [0, 2, 3, 7, 8] |
| 3 | 75.0% | 70.9% | 100.0% | ❌ | [0, 1, 2] | [0, 1, 2, 7, 8] |

### DSC Analysis
| Example | Clues Used | Stop Probs | Stop Logits |
|---------|------------|------------|-------------|
| 1 | 3.89 | [0.026, 0.033, 0.026, 0.027] | [-3.64, -3.39, -3.64, -3.60] |
| 2 | 3.90 | [0.023, 0.027, 0.026, 0.024] | [-3.75, -3.59, -3.64, -3.72] |
| 3 | 3.90 | [0.024, 0.028, 0.027, 0.024] | [-3.70, -3.53, -3.57, -3.70] |

### Attention Entropy (per clue)
| Example | Clue 0 | Clue 1 | Clue 2 | Clue 3 | Clue 4 | Clue 5 |
|---------|--------|--------|--------|--------|--------|--------|
| 1 | 0.635 | 0.515 | 0.429 | 0.376 |
| 2 | 0.684 | 0.263 | 0.548 | 0.695 |
| 3 | 0.757 | 0.563 | 0.565 | 0.625 |

### Gradient Norms (selected modules)
- **encoder:** 0.033875
- **feature_proj:** 0.056994
- **context_encoder:** 0.043300
- **context_injector:** 0.038635
- **dsc:** 0.023713
- **msre:** 0.088617
- **solver:** 1.969083

### Predictions vs Targets

**Example 1:**

Target:
```
0 0 0 0
0 0 0 0
0 0 0 0
0 0 0 5
```

Prediction:
```
2 2 2 2
7 8 8 8
0 0 0 0
0 0 0 0
```

**Example 2:**

Target:
```
3 2 3 2 3 2
7 8 7 8 7 8
2 3 2 3 2 3
8 7 8 7 8 7
3 2 3 2 3 2
7 8 7 8 7 8
```

Prediction:
```
3 2 3 2 2 2
7 8 7 8 8 8
2 3 2 3 0 0
8 7 8 0 0 0
0 0 0 0 0 0
7 8 8 0 8 0
```

**Example 3:**

Target:
```
2 2 2
1 1 1
1 1 1
```

Prediction:
```
2 2 2
1 1 1
1 1 1
```


## Epoch 11

**Temperature:** 0.9626

### Losses
| Loss | Value |
|------|-------|
| total_loss | 0.569542 |
| task_loss | 0.365955 |
| focal_loss | 0.365955 |
| entropy_loss | 2.571519 |
| sparsity_loss | 0.087398 |
| predicate_loss | 0.000000 |
| curriculum_loss | 2.697231 |
| deep_supervision_loss | 0.389694 |
| act_loss | 0.000000 |
| loss_mode | weighted_stablemax |
| sparsity_min_clue_penalty | 0.000000 |
| sparsity_base_pondering | 0.039058 |
| sparsity_entropy_pondering | 0.048339 |
| expected_clues_used | 3.905847 |
| stop_prob_from_loss | 0.023538 |
| clues_used_std | 0.002995 |
| per_sample_clue_penalty_mean | 0.000000 |

### Metrics
- **Total Accuracy:** 58.33%
- **BG Accuracy:** 50.68%
- **FG Accuracy:** 82.61%

### Per-Example Metrics
| Example | Accuracy | BG Acc | FG Acc | Exact Match | Target Classes | Pred Classes |
|---------|----------|--------|--------|-------------|----------------|---------------|
| 1 | 42.2% | 42.9% | 0.0% | ❌ | [0, 5] | [0, 2, 7, 8] |
| 2 | 82.8% | 85.7% | 80.6% | ❌ | [0, 2, 3, 7, 8] | [0, 2, 3, 7, 8] |
| 3 | 50.0% | 41.8% | 100.0% | ❌ | [0, 1, 2] | [0, 1, 2, 3, 7, 8] |

### DSC Analysis
| Example | Clues Used | Stop Probs | Stop Logits |
|---------|------------|------------|-------------|
| 1 | 3.90 | [0.024, 0.022, 0.029, 0.022] | [-3.70, -3.80, -3.49, -3.80] |
| 2 | 3.91 | [0.020, 0.022, 0.023, 0.028] | [-3.91, -3.82, -3.77, -3.56] |
| 3 | 3.91 | [0.025, 0.026, 0.021, 0.021] | [-3.65, -3.62, -3.83, -3.85] |

### Attention Entropy (per clue)
| Example | Clue 0 | Clue 1 | Clue 2 | Clue 3 | Clue 4 | Clue 5 |
|---------|--------|--------|--------|--------|--------|--------|
| 1 | 0.639 | 0.628 | 0.014 | 0.636 |
| 2 | 0.802 | 0.684 | 0.633 | 0.461 |
| 3 | 0.815 | 0.553 | 0.766 | 0.788 |


## Epoch 12

**Temperature:** 0.9593

### Losses
| Loss | Value |
|------|-------|
| total_loss | 0.651112 |
| task_loss | 0.431537 |
| focal_loss | 0.431537 |
| entropy_loss | 2.482732 |
| sparsity_loss | 0.085753 |
| predicate_loss | 0.000000 |
| curriculum_loss | 2.682525 |
| deep_supervision_loss | 0.421999 |
| act_loss | 0.000000 |
| loss_mode | weighted_stablemax |
| sparsity_min_clue_penalty | 0.000000 |
| sparsity_base_pondering | 0.039074 |
| sparsity_entropy_pondering | 0.046679 |
| expected_clues_used | 3.907360 |
| stop_prob_from_loss | 0.023160 |
| clues_used_std | 0.007328 |
| per_sample_clue_penalty_mean | 0.000000 |

### Metrics
- **Total Accuracy:** 80.73%
- **BG Accuracy:** 91.10%
- **FG Accuracy:** 47.83%

### Per-Example Metrics
| Example | Accuracy | BG Acc | FG Acc | Exact Match | Target Classes | Pred Classes |
|---------|----------|--------|--------|-------------|----------------|---------------|
| 1 | 85.9% | 87.3% | 0.0% | ❌ | [0, 5] | [0, 1, 2, 7] |
| 2 | 64.1% | 100.0% | 36.1% | ❌ | [0, 2, 3, 7, 8] | [0, 2, 3, 7, 8] |
| 3 | 92.2% | 90.9% | 100.0% | ❌ | [0, 1, 2] | [0, 1, 2] |

### DSC Analysis
| Example | Clues Used | Stop Probs | Stop Logits |
|---------|------------|------------|-------------|
| 1 | 3.91 | [0.023, 0.020, 0.023, 0.021] | [-3.75, -3.88, -3.74, -3.83] |
| 2 | 3.91 | [0.021, 0.023, 0.025, 0.020] | [-3.83, -3.75, -3.68, -3.87] |
| 3 | 3.90 | [0.025, 0.025, 0.028, 0.024] | [-3.67, -3.67, -3.55, -3.72] |

### Attention Entropy (per clue)
| Example | Clue 0 | Clue 1 | Clue 2 | Clue 3 | Clue 4 | Clue 5 |
|---------|--------|--------|--------|--------|--------|--------|
| 1 | 0.747 | 0.747 | 0.687 | 0.596 |
| 2 | 0.790 | 0.511 | 0.698 | 0.740 |
| 3 | 0.263 | 0.736 | 0.002 | 0.647 |


## Epoch 13

**Temperature:** 0.9559

### Losses
| Loss | Value |
|------|-------|
| total_loss | 0.534665 |
| task_loss | 0.347182 |
| focal_loss | 0.347182 |
| entropy_loss | 2.458309 |
| sparsity_loss | 0.085370 |
| predicate_loss | 0.000000 |
| curriculum_loss | 2.674833 |
| deep_supervision_loss | 0.357894 |
| act_loss | 0.000000 |
| loss_mode | weighted_stablemax |
| sparsity_min_clue_penalty | 0.000000 |
| sparsity_base_pondering | 0.039103 |
| sparsity_entropy_pondering | 0.046266 |
| expected_clues_used | 3.910330 |
| stop_prob_from_loss | 0.022417 |
| clues_used_std | 0.002796 |
| per_sample_clue_penalty_mean | 0.000000 |

### Metrics
- **Total Accuracy:** 72.40%
- **BG Accuracy:** 69.86%
- **FG Accuracy:** 80.43%

### Per-Example Metrics
| Example | Accuracy | BG Acc | FG Acc | Exact Match | Target Classes | Pred Classes |
|---------|----------|--------|--------|-------------|----------------|---------------|
| 1 | 64.1% | 65.1% | 0.0% | ❌ | [0, 5] | [0, 2, 3, 7] |
| 2 | 87.5% | 100.0% | 77.8% | ❌ | [0, 2, 3, 7, 8] | [0, 2, 3, 7, 8] |
| 3 | 65.6% | 60.0% | 100.0% | ❌ | [0, 1, 2] | [0, 1, 2, 3, 7, 8] |

### DSC Analysis
| Example | Clues Used | Stop Probs | Stop Logits |
|---------|------------|------------|-------------|
| 1 | 3.91 | [0.021, 0.020, 0.020, 0.032] | [-3.84, -3.90, -3.87, -3.42] |
| 2 | 3.91 | [0.024, 0.021, 0.022, 0.021] | [-3.69, -3.85, -3.79, -3.85] |
| 3 | 3.91 | [0.023, 0.022, 0.021, 0.021] | [-3.73, -3.79, -3.82, -3.85] |

### Attention Entropy (per clue)
| Example | Clue 0 | Clue 1 | Clue 2 | Clue 3 | Clue 4 | Clue 5 |
|---------|--------|--------|--------|--------|--------|--------|
| 1 | 0.493 | 0.685 | 0.693 | 0.022 |
| 2 | 0.575 | 0.575 | 0.702 | 0.596 |
| 3 | 0.777 | 0.568 | 0.728 | 0.679 |


## Epoch 14

**Temperature:** 0.9526

### Losses
| Loss | Value |
|------|-------|
| total_loss | 0.533902 |
| task_loss | 0.345010 |
| focal_loss | 0.345010 |
| entropy_loss | 2.596506 |
| sparsity_loss | 0.088059 |
| predicate_loss | 0.000000 |
| curriculum_loss | 2.665490 |
| deep_supervision_loss | 0.360171 |
| act_loss | 0.000000 |
| loss_mode | weighted_stablemax |
| sparsity_min_clue_penalty | 0.000000 |
| sparsity_base_pondering | 0.039156 |
| sparsity_entropy_pondering | 0.048904 |
| expected_clues_used | 3.915576 |
| stop_prob_from_loss | 0.021106 |
| clues_used_std | 0.002537 |
| per_sample_clue_penalty_mean | 0.000000 |

### Metrics
- **Total Accuracy:** 63.54%
- **BG Accuracy:** 56.16%
- **FG Accuracy:** 86.96%

### Per-Example Metrics
| Example | Accuracy | BG Acc | FG Acc | Exact Match | Target Classes | Pred Classes |
|---------|----------|--------|--------|-------------|----------------|---------------|
| 1 | 45.3% | 46.0% | 0.0% | ❌ | [0, 5] | [0, 2, 3, 7, 8] |
| 2 | 89.1% | 92.9% | 86.1% | ❌ | [0, 2, 3, 7, 8] | [0, 2, 3, 7, 8] |
| 3 | 56.2% | 49.1% | 100.0% | ❌ | [0, 1, 2] | [0, 1, 2, 3, 7, 8] |

### DSC Analysis
| Example | Clues Used | Stop Probs | Stop Logits |
|---------|------------|------------|-------------|
| 1 | 3.91 | [0.020, 0.024, 0.020, 0.023] | [-3.90, -3.69, -3.89, -3.75] |
| 2 | 3.92 | [0.022, 0.020, 0.019, 0.021] | [-3.80, -3.89, -3.92, -3.84] |
| 3 | 3.92 | [0.022, 0.020, 0.020, 0.022] | [-3.81, -3.88, -3.90, -3.80] |

### Attention Entropy (per clue)
| Example | Clue 0 | Clue 1 | Clue 2 | Clue 3 | Clue 4 | Clue 5 |
|---------|--------|--------|--------|--------|--------|--------|
| 1 | 0.726 | 0.242 | 0.645 | 0.571 |
| 2 | 0.685 | 0.728 | 0.624 | 0.726 |
| 3 | 0.595 | 0.562 | 0.759 | 0.630 |


## Epoch 15

**Temperature:** 0.9493

### Losses
| Loss | Value |
|------|-------|
| total_loss | 0.488069 |
| task_loss | 0.313522 |
| focal_loss | 0.313522 |
| entropy_loss | 2.291525 |
| sparsity_loss | 0.082266 |
| predicate_loss | 0.000000 |
| curriculum_loss | 2.649938 |
| deep_supervision_loss | 0.332642 |
| act_loss | 0.000000 |
| loss_mode | weighted_stablemax |
| sparsity_min_clue_penalty | 0.000000 |
| sparsity_base_pondering | 0.039135 |
| sparsity_entropy_pondering | 0.043131 |
| expected_clues_used | 3.913474 |
| stop_prob_from_loss | 0.021632 |
| clues_used_std | 0.004698 |
| per_sample_clue_penalty_mean | 0.000000 |

### Metrics
- **Total Accuracy:** 64.06%
- **BG Accuracy:** 54.11%
- **FG Accuracy:** 95.65%

### Per-Example Metrics
| Example | Accuracy | BG Acc | FG Acc | Exact Match | Target Classes | Pred Classes |
|---------|----------|--------|--------|-------------|----------------|---------------|
| 1 | 40.6% | 41.3% | 0.0% | ❌ | [0, 5] | [0, 2, 3, 7, 8] |
| 2 | 96.9% | 96.4% | 97.2% | ❌ | [0, 2, 3, 7, 8] | [0, 2, 3, 7, 8] |
| 3 | 54.7% | 47.3% | 100.0% | ❌ | [0, 1, 2] | [0, 1, 2, 3, 7, 8] |

### DSC Analysis
| Example | Clues Used | Stop Probs | Stop Logits |
|---------|------------|------------|-------------|
| 1 | 3.91 | [0.023, 0.020, 0.023, 0.024] | [-3.76, -3.89, -3.73, -3.69] |
| 2 | 3.91 | [0.020, 0.022, 0.022, 0.024] | [-3.91, -3.80, -3.79, -3.70] |
| 3 | 3.92 | [0.021, 0.020, 0.020, 0.020] | [-3.83, -3.87, -3.91, -3.89] |

### Attention Entropy (per clue)
| Example | Clue 0 | Clue 1 | Clue 2 | Clue 3 | Clue 4 | Clue 5 |
|---------|--------|--------|--------|--------|--------|--------|
| 1 | 0.565 | 0.720 | 0.491 | 0.547 |
| 2 | 0.791 | 0.243 | 0.226 | 0.687 |
| 3 | 0.654 | 0.647 | 0.533 | 0.508 |


## Epoch 16

**Temperature:** 0.9461

### Losses
| Loss | Value |
|------|-------|
| total_loss | 0.470107 |
| task_loss | 0.300547 |
| focal_loss | 0.300547 |
| entropy_loss | 2.272623 |
| sparsity_loss | 0.081964 |
| predicate_loss | 0.000000 |
| curriculum_loss | 2.635681 |
| deep_supervision_loss | 0.322729 |
| act_loss | 0.000000 |
| loss_mode | weighted_stablemax |
| sparsity_min_clue_penalty | 0.000000 |
| sparsity_base_pondering | 0.039160 |
| sparsity_entropy_pondering | 0.042804 |
| expected_clues_used | 3.915979 |
| stop_prob_from_loss | 0.021005 |
| clues_used_std | 0.002046 |
| per_sample_clue_penalty_mean | 0.000000 |

### Metrics
- **Total Accuracy:** 63.54%
- **BG Accuracy:** 55.48%
- **FG Accuracy:** 89.13%

### Per-Example Metrics
| Example | Accuracy | BG Acc | FG Acc | Exact Match | Target Classes | Pred Classes |
|---------|----------|--------|--------|-------------|----------------|---------------|
| 1 | 40.6% | 41.3% | 0.0% | ❌ | [0, 5] | [0, 2, 3, 7, 8] |
| 2 | 92.2% | 96.4% | 88.9% | ❌ | [0, 2, 3, 7, 8] | [0, 2, 3, 7, 8] |
| 3 | 57.8% | 50.9% | 100.0% | ❌ | [0, 1, 2] | [0, 1, 2, 3, 7, 8] |

### DSC Analysis
| Example | Clues Used | Stop Probs | Stop Logits |
|---------|------------|------------|-------------|
| 1 | 3.92 | [0.022, 0.020, 0.022, 0.020] | [-3.78, -3.89, -3.80, -3.91] |
| 2 | 3.91 | [0.022, 0.020, 0.019, 0.025] | [-3.81, -3.87, -3.94, -3.67] |
| 3 | 3.92 | [0.022, 0.020, 0.020, 0.021] | [-3.81, -3.91, -3.90, -3.84] |

### Attention Entropy (per clue)
| Example | Clue 0 | Clue 1 | Clue 2 | Clue 3 | Clue 4 | Clue 5 |
|---------|--------|--------|--------|--------|--------|--------|
| 1 | 0.457 | 0.435 | 0.570 | 0.830 |
| 2 | 0.177 | 0.453 | 0.661 | 0.593 |
| 3 | 0.482 | 0.619 | 0.677 | 0.603 |


## Epoch 17

**Temperature:** 0.9428

### Losses
| Loss | Value |
|------|-------|
| total_loss | 0.449661 |
| task_loss | 0.286524 |
| focal_loss | 0.286524 |
| entropy_loss | 2.563559 |
| sparsity_loss | 0.087495 |
| predicate_loss | 0.000000 |
| curriculum_loss | 2.622351 |
| deep_supervision_loss | 0.308776 |
| act_loss | 0.000000 |
| loss_mode | weighted_stablemax |
| sparsity_min_clue_penalty | 0.000000 |
| sparsity_base_pondering | 0.039183 |
| sparsity_entropy_pondering | 0.048312 |
| expected_clues_used | 3.918290 |
| stop_prob_from_loss | 0.020427 |
| clues_used_std | 0.001523 |
| per_sample_clue_penalty_mean | 0.000000 |

### Metrics
- **Total Accuracy:** 68.23%
- **BG Accuracy:** 60.96%
- **FG Accuracy:** 91.30%

### Per-Example Metrics
| Example | Accuracy | BG Acc | FG Acc | Exact Match | Target Classes | Pred Classes |
|---------|----------|--------|--------|-------------|----------------|---------------|
| 1 | 50.0% | 50.8% | 0.0% | ❌ | [0, 5] | [0, 2, 3, 7, 8] |
| 2 | 95.3% | 100.0% | 91.7% | ❌ | [0, 2, 3, 7, 8] | [0, 2, 3, 7, 8] |
| 3 | 59.4% | 52.7% | 100.0% | ❌ | [0, 1, 2] | [0, 1, 2, 3, 7, 8] |

### DSC Analysis
| Example | Clues Used | Stop Probs | Stop Logits |
|---------|------------|------------|-------------|
| 1 | 3.92 | [0.022, 0.020, 0.020, 0.020] | [-3.79, -3.91, -3.89, -3.90] |
| 2 | 3.92 | [0.020, 0.020, 0.024, 0.020] | [-3.90, -3.89, -3.72, -3.92] |
| 3 | 3.92 | [0.022, 0.019, 0.019, 0.020] | [-3.80, -3.92, -3.94, -3.90] |

### Attention Entropy (per clue)
| Example | Clue 0 | Clue 1 | Clue 2 | Clue 3 | Clue 4 | Clue 5 |
|---------|--------|--------|--------|--------|--------|--------|
| 1 | 0.830 | 0.576 | 0.591 | 0.526 |
| 2 | 0.724 | 0.459 | 0.121 | 0.781 |
| 3 | 0.877 | 0.615 | 0.711 | 0.585 |


## Epoch 18

**Temperature:** 0.9395

### Losses
| Loss | Value |
|------|-------|
| total_loss | 0.422616 |
| task_loss | 0.268028 |
| focal_loss | 0.268028 |
| entropy_loss | 2.287799 |
| sparsity_loss | 0.082305 |
| predicate_loss | 0.000000 |
| curriculum_loss | 2.608074 |
| deep_supervision_loss | 0.292715 |
| act_loss | 0.000000 |
| loss_mode | weighted_stablemax |
| sparsity_min_clue_penalty | 0.000000 |
| sparsity_base_pondering | 0.039184 |
| sparsity_entropy_pondering | 0.043121 |
| expected_clues_used | 3.918435 |
| stop_prob_from_loss | 0.020391 |
| clues_used_std | 0.004395 |
| per_sample_clue_penalty_mean | 0.000000 |

### Metrics
- **Total Accuracy:** 81.25%
- **BG Accuracy:** 80.14%
- **FG Accuracy:** 84.78%

### Per-Example Metrics
| Example | Accuracy | BG Acc | FG Acc | Exact Match | Target Classes | Pred Classes |
|---------|----------|--------|--------|-------------|----------------|---------------|
| 1 | 71.9% | 71.4% | 100.0% | ❌ | [0, 5] | [0, 2, 3, 5, 7, 8] |
| 2 | 89.1% | 100.0% | 80.6% | ❌ | [0, 2, 3, 7, 8] | [0, 2, 3, 7, 8] |
| 3 | 82.8% | 80.0% | 100.0% | ❌ | [0, 1, 2] | [0, 1, 2, 3, 7, 8] |

### DSC Analysis
| Example | Clues Used | Stop Probs | Stop Logits |
|---------|------------|------------|-------------|
| 1 | 3.92 | [0.019, 0.020, 0.020, 0.020] | [-3.94, -3.89, -3.87, -3.91] |
| 2 | 3.92 | [0.020, 0.019, 0.020, 0.019] | [-3.89, -3.92, -3.87, -3.94] |
| 3 | 3.91 | [0.022, 0.022, 0.022, 0.020] | [-3.77, -3.79, -3.77, -3.90] |

### Attention Entropy (per clue)
| Example | Clue 0 | Clue 1 | Clue 2 | Clue 3 | Clue 4 | Clue 5 |
|---------|--------|--------|--------|--------|--------|--------|
| 1 | 0.769 | 0.429 | 0.459 | 0.535 |
| 2 | 0.723 | 0.650 | 0.550 | 0.715 |
| 3 | 0.498 | 0.325 | 0.344 | 0.605 |


## Epoch 19

**Temperature:** 0.9363

### Losses
| Loss | Value |
|------|-------|
| total_loss | 0.379340 |
| task_loss | 0.238462 |
| focal_loss | 0.238462 |
| entropy_loss | 2.386660 |
| sparsity_loss | 0.084202 |
| predicate_loss | 0.000000 |
| curriculum_loss | 2.596169 |
| deep_supervision_loss | 0.264916 |
| act_loss | 0.000000 |
| loss_mode | weighted_stablemax |
| sparsity_min_clue_penalty | 0.000000 |
| sparsity_base_pondering | 0.039201 |
| sparsity_entropy_pondering | 0.045001 |
| expected_clues_used | 3.920137 |
| stop_prob_from_loss | 0.019966 |
| clues_used_std | 0.002890 |
| per_sample_clue_penalty_mean | 0.000000 |

### Metrics
- **Total Accuracy:** 76.56%
- **BG Accuracy:** 69.18%
- **FG Accuracy:** 100.00%

### Per-Example Metrics
| Example | Accuracy | BG Acc | FG Acc | Exact Match | Target Classes | Pred Classes |
|---------|----------|--------|--------|-------------|----------------|---------------|
| 1 | 57.8% | 57.1% | 100.0% | ❌ | [0, 5] | [0, 2, 3, 5, 7, 8] |
| 2 | 100.0% | 100.0% | 100.0% | ✅ | [0, 2, 3, 7, 8] | [0, 2, 3, 7, 8] |
| 3 | 71.9% | 67.3% | 100.0% | ❌ | [0, 1, 2] | [0, 1, 2, 3, 7, 8] |

### DSC Analysis
| Example | Clues Used | Stop Probs | Stop Logits |
|---------|------------|------------|-------------|
| 1 | 3.92 | [0.019, 0.019, 0.019, 0.020] | [-3.94, -3.92, -3.94, -3.88] |
| 2 | 3.92 | [0.019, 0.020, 0.019, 0.020] | [-3.92, -3.89, -3.93, -3.89] |
| 3 | 3.92 | [0.023, 0.020, 0.021, 0.020] | [-3.76, -3.89, -3.87, -3.90] |

### Attention Entropy (per clue)
| Example | Clue 0 | Clue 1 | Clue 2 | Clue 3 | Clue 4 | Clue 5 |
|---------|--------|--------|--------|--------|--------|--------|
| 1 | 0.748 | 0.648 | 0.476 | 0.525 |
| 2 | 0.702 | 0.639 | 0.675 | 0.559 |
| 3 | 0.281 | 0.705 | 0.336 | 0.592 |


## Epoch 20

**Temperature:** 0.9330

### Losses
| Loss | Value |
|------|-------|
| total_loss | 0.355389 |
| task_loss | 0.221374 |
| focal_loss | 0.221374 |
| entropy_loss | 2.212728 |
| sparsity_loss | 0.080918 |
| predicate_loss | 0.000000 |
| curriculum_loss | 2.581222 |
| deep_supervision_loss | 0.251846 |
| act_loss | 0.000000 |
| loss_mode | weighted_stablemax |
| sparsity_min_clue_penalty | 0.000000 |
| sparsity_base_pondering | 0.039198 |
| sparsity_entropy_pondering | 0.041719 |
| expected_clues_used | 3.919820 |
| stop_prob_from_loss | 0.020045 |
| clues_used_std | 0.000418 |
| per_sample_clue_penalty_mean | 0.000000 |

### Metrics
- **Total Accuracy:** 76.04%
- **BG Accuracy:** 68.49%
- **FG Accuracy:** 100.00%

### Per-Example Metrics
| Example | Accuracy | BG Acc | FG Acc | Exact Match | Target Classes | Pred Classes |
|---------|----------|--------|--------|-------------|----------------|---------------|
| 1 | 64.1% | 63.5% | 100.0% | ❌ | [0, 5] | [0, 2, 3, 5, 7, 8] |
| 2 | 100.0% | 100.0% | 100.0% | ✅ | [0, 2, 3, 7, 8] | [0, 2, 3, 7, 8] |
| 3 | 64.1% | 58.2% | 100.0% | ❌ | [0, 1, 2] | [0, 1, 2, 3, 7, 8] |

### DSC Analysis
| Example | Clues Used | Stop Probs | Stop Logits |
|---------|------------|------------|-------------|
| 1 | 3.92 | [0.019, 0.021, 0.020, 0.020] | [-3.93, -3.82, -3.91, -3.91] |
| 2 | 3.92 | [0.020, 0.019, 0.021, 0.020] | [-3.90, -3.92, -3.84, -3.88] |
| 3 | 3.92 | [0.021, 0.019, 0.021, 0.019] | [-3.84, -3.95, -3.86, -3.93] |

### Attention Entropy (per clue)
| Example | Clue 0 | Clue 1 | Clue 2 | Clue 3 | Clue 4 | Clue 5 |
|---------|--------|--------|--------|--------|--------|--------|
| 1 | 0.666 | 0.169 | 0.632 | 0.664 |
| 2 | 0.728 | 0.564 | 0.572 | 0.469 |
| 3 | 0.213 | 0.621 | 0.387 | 0.700 |

### Gradient Norms (selected modules)
- **encoder:** 0.024228
- **feature_proj:** 0.038000
- **context_encoder:** 0.027387
- **context_injector:** 0.021048
- **dsc:** 0.014884
- **msre:** 0.046039
- **solver:** 0.669939

### Predictions vs Targets

**Example 1:**

Target:
```
0 0 0 0
0 0 0 0
0 0 0 0
0 0 0 5
```

Prediction:
```
2 0 0 0
0 0 0 0
3 2 3 2
0 0 7 5
```

**Example 2:**

Target:
```
3 2 3 2 3 2
7 8 7 8 7 8
2 3 2 3 2 3
8 7 8 7 8 7
3 2 3 2 3 2
7 8 7 8 7 8
```

Prediction:
```
3 2 3 2 3 2
7 8 7 8 7 8
2 3 2 3 2 3
8 7 8 7 8 7
3 2 3 2 3 2
7 8 7 8 7 8
```

**Example 3:**

Target:
```
2 2 2
1 1 1
1 1 1
```

Prediction:
```
2 2 2
1 1 1
1 1 1
```


## Epoch 21

**Temperature:** 0.9298

### Losses
| Loss | Value |
|------|-------|
| total_loss | 0.346466 |
| task_loss | 0.216263 |
| focal_loss | 0.216263 |
| entropy_loss | 1.835813 |
| sparsity_loss | 0.073766 |
| predicate_loss | 0.000000 |
| curriculum_loss | 2.562202 |
| deep_supervision_loss | 0.245652 |
| act_loss | 0.000000 |
| loss_mode | weighted_stablemax |
| sparsity_min_clue_penalty | 0.000000 |
| sparsity_base_pondering | 0.039173 |
| sparsity_entropy_pondering | 0.034594 |
| expected_clues_used | 3.917256 |
| stop_prob_from_loss | 0.020686 |
| clues_used_std | 0.002459 |
| per_sample_clue_penalty_mean | 0.000000 |

### Metrics
- **Total Accuracy:** 86.46%
- **BG Accuracy:** 82.88%
- **FG Accuracy:** 97.83%

### Per-Example Metrics
| Example | Accuracy | BG Acc | FG Acc | Exact Match | Target Classes | Pred Classes |
|---------|----------|--------|--------|-------------|----------------|---------------|
| 1 | 79.7% | 79.4% | 100.0% | ❌ | [0, 5] | [0, 2, 3, 5, 7, 8] |
| 2 | 98.4% | 100.0% | 97.2% | ❌ | [0, 2, 3, 7, 8] | [0, 2, 3, 7, 8] |
| 3 | 81.2% | 78.2% | 100.0% | ❌ | [0, 1, 2] | [0, 1, 2, 3, 7, 8] |

### DSC Analysis
| Example | Clues Used | Stop Probs | Stop Logits |
|---------|------------|------------|-------------|
| 1 | 3.92 | [0.019, 0.021, 0.023, 0.022] | [-3.94, -3.82, -3.77, -3.81] |
| 2 | 3.92 | [0.023, 0.021, 0.019, 0.020] | [-3.75, -3.83, -3.95, -3.87] |
| 3 | 3.92 | [0.020, 0.020, 0.020, 0.020] | [-3.87, -3.90, -3.88, -3.91] |

### Attention Entropy (per clue)
| Example | Clue 0 | Clue 1 | Clue 2 | Clue 3 | Clue 4 | Clue 5 |
|---------|--------|--------|--------|--------|--------|--------|
| 1 | 0.576 | 0.444 | 0.209 | 0.331 |
| 2 | 0.252 | 0.546 | 0.653 | 0.287 |
| 3 | 0.658 | 0.587 | 0.393 | 0.361 |


## Epoch 22

**Temperature:** 0.9266

### Losses
| Loss | Value |
|------|-------|
| total_loss | 0.314143 |
| task_loss | 0.193927 |
| focal_loss | 0.193927 |
| entropy_loss | 1.921887 |
| sparsity_loss | 0.075418 |
| predicate_loss | 0.000000 |
| curriculum_loss | 2.549727 |
| deep_supervision_loss | 0.225348 |
| act_loss | 0.000000 |
| loss_mode | weighted_stablemax |
| sparsity_min_clue_penalty | 0.000000 |
| sparsity_base_pondering | 0.039191 |
| sparsity_entropy_pondering | 0.036227 |
| expected_clues_used | 3.919132 |
| stop_prob_from_loss | 0.020217 |
| clues_used_std | 0.002132 |
| per_sample_clue_penalty_mean | 0.000000 |

### Metrics
- **Total Accuracy:** 83.85%
- **BG Accuracy:** 78.77%
- **FG Accuracy:** 100.00%

### Per-Example Metrics
| Example | Accuracy | BG Acc | FG Acc | Exact Match | Target Classes | Pred Classes |
|---------|----------|--------|--------|-------------|----------------|---------------|
| 1 | 76.6% | 76.2% | 100.0% | ❌ | [0, 5] | [0, 2, 3, 5, 7, 8] |
| 2 | 100.0% | 100.0% | 100.0% | ✅ | [0, 2, 3, 7, 8] | [0, 2, 3, 7, 8] |
| 3 | 75.0% | 70.9% | 100.0% | ❌ | [0, 1, 2] | [0, 1, 2, 3, 7, 8] |

### DSC Analysis
| Example | Clues Used | Stop Probs | Stop Logits |
|---------|------------|------------|-------------|
| 1 | 3.92 | [0.021, 0.022, 0.020, 0.020] | [-3.86, -3.80, -3.92, -3.89] |
| 2 | 3.92 | [0.021, 0.021, 0.020, 0.021] | [-3.85, -3.86, -3.88, -3.86] |
| 3 | 3.92 | [0.019, 0.020, 0.021, 0.019] | [-3.94, -3.91, -3.87, -3.93] |

### Attention Entropy (per clue)
| Example | Clue 0 | Clue 1 | Clue 2 | Clue 3 | Clue 4 | Clue 5 |
|---------|--------|--------|--------|--------|--------|--------|
| 1 | 0.367 | 0.338 | 0.416 | 0.519 |
| 2 | 0.414 | 0.342 | 0.439 | 0.510 |
| 3 | 0.747 | 0.580 | 0.369 | 0.504 |


## Epoch 23

**Temperature:** 0.9234

### Losses
| Loss | Value |
|------|-------|
| total_loss | 0.300852 |
| task_loss | 0.183696 |
| focal_loss | 0.183696 |
| entropy_loss | 2.101032 |
| sparsity_loss | 0.078785 |
| predicate_loss | 0.000000 |
| curriculum_loss | 2.536478 |
| deep_supervision_loss | 0.218556 |
| act_loss | 0.000000 |
| loss_mode | weighted_stablemax |
| sparsity_min_clue_penalty | 0.000000 |
| sparsity_base_pondering | 0.039185 |
| sparsity_entropy_pondering | 0.039600 |
| expected_clues_used | 3.918539 |
| stop_prob_from_loss | 0.020365 |
| clues_used_std | 0.001664 |
| per_sample_clue_penalty_mean | 0.000000 |

### Metrics
- **Total Accuracy:** 87.50%
- **BG Accuracy:** 83.56%
- **FG Accuracy:** 100.00%

### Per-Example Metrics
| Example | Accuracy | BG Acc | FG Acc | Exact Match | Target Classes | Pred Classes |
|---------|----------|--------|--------|-------------|----------------|---------------|
| 1 | 79.7% | 79.4% | 100.0% | ❌ | [0, 5] | [0, 2, 3, 5, 7, 8] |
| 2 | 100.0% | 100.0% | 100.0% | ✅ | [0, 2, 3, 7, 8] | [0, 2, 3, 7, 8] |
| 3 | 82.8% | 80.0% | 100.0% | ❌ | [0, 1, 2] | [0, 1, 2, 3, 7, 8] |

### DSC Analysis
| Example | Clues Used | Stop Probs | Stop Logits |
|---------|------------|------------|-------------|
| 1 | 3.92 | [0.019, 0.023, 0.020, 0.021] | [-3.96, -3.74, -3.92, -3.85] |
| 2 | 3.92 | [0.019, 0.020, 0.020, 0.023] | [-3.94, -3.89, -3.87, -3.74] |
| 3 | 3.92 | [0.020, 0.020, 0.020, 0.019] | [-3.87, -3.91, -3.89, -3.93] |

### Attention Entropy (per clue)
| Example | Clue 0 | Clue 1 | Clue 2 | Clue 3 | Clue 4 | Clue 5 |
|---------|--------|--------|--------|--------|--------|--------|
| 1 | 0.796 | 0.648 | 0.431 | 0.267 |
| 2 | 0.766 | 0.471 | 0.455 | 0.311 |
| 3 | 0.307 | 0.660 | 0.406 | 0.544 |


## Epoch 24

**Temperature:** 0.9202

### Losses
| Loss | Value |
|------|-------|
| total_loss | 0.298698 |
| task_loss | 0.185699 |
| focal_loss | 0.185699 |
| entropy_loss | 1.955772 |
| sparsity_loss | 0.075991 |
| predicate_loss | 0.000000 |
| curriculum_loss | 2.515955 |
| deep_supervision_loss | 0.210798 |
| act_loss | 0.000000 |
| loss_mode | weighted_stablemax |
| sparsity_min_clue_penalty | 0.000000 |
| sparsity_base_pondering | 0.039154 |
| sparsity_entropy_pondering | 0.036837 |
| expected_clues_used | 3.915443 |
| stop_prob_from_loss | 0.021139 |
| clues_used_std | 0.003480 |
| per_sample_clue_penalty_mean | 0.000000 |

### Metrics
- **Total Accuracy:** 88.02%
- **BG Accuracy:** 84.93%
- **FG Accuracy:** 97.83%

### Per-Example Metrics
| Example | Accuracy | BG Acc | FG Acc | Exact Match | Target Classes | Pred Classes |
|---------|----------|--------|--------|-------------|----------------|---------------|
| 1 | 81.2% | 82.5% | 0.0% | ❌ | [0, 5] | [0, 2, 3, 7, 8] |
| 2 | 98.4% | 96.4% | 100.0% | ❌ | [0, 2, 3, 7, 8] | [0, 2, 3, 7, 8] |
| 3 | 84.4% | 81.8% | 100.0% | ❌ | [0, 1, 2] | [0, 1, 2, 3, 7, 8] |

### DSC Analysis
| Example | Clues Used | Stop Probs | Stop Logits |
|---------|------------|------------|-------------|
| 1 | 3.92 | [0.024, 0.020, 0.021, 0.019] | [-3.72, -3.89, -3.86, -3.92] |
| 2 | 3.91 | [0.020, 0.020, 0.025, 0.023] | [-3.87, -3.90, -3.65, -3.76] |
| 3 | 3.92 | [0.019, 0.020, 0.022, 0.020] | [-3.93, -3.89, -3.80, -3.87] |

### Attention Entropy (per clue)
| Example | Clue 0 | Clue 1 | Clue 2 | Clue 3 | Clue 4 | Clue 5 |
|---------|--------|--------|--------|--------|--------|--------|
| 1 | 0.149 | 0.536 | 0.569 | 0.516 |
| 2 | 0.614 | 0.469 | 0.587 | 0.029 |
| 3 | 0.756 | 0.373 | 0.557 | 0.489 |


## Epoch 25

**Temperature:** 0.9170

### Losses
| Loss | Value |
|------|-------|
| total_loss | 0.271971 |
| task_loss | 0.166998 |
| focal_loss | 0.166998 |
| entropy_loss | 1.881687 |
| sparsity_loss | 0.074574 |
| predicate_loss | 0.000000 |
| curriculum_loss | 2.498095 |
| deep_supervision_loss | 0.195032 |
| act_loss | 0.000000 |
| loss_mode | weighted_stablemax |
| sparsity_min_clue_penalty | 0.000000 |
| sparsity_base_pondering | 0.039135 |
| sparsity_entropy_pondering | 0.035438 |
| expected_clues_used | 3.913537 |
| stop_prob_from_loss | 0.021616 |
| clues_used_std | 0.002884 |
| per_sample_clue_penalty_mean | 0.000000 |

### Metrics
- **Total Accuracy:** 99.48%
- **BG Accuracy:** 99.32%
- **FG Accuracy:** 100.00%

### Per-Example Metrics
| Example | Accuracy | BG Acc | FG Acc | Exact Match | Target Classes | Pred Classes |
|---------|----------|--------|--------|-------------|----------------|---------------|
| 1 | 100.0% | 100.0% | 100.0% | ✅ | [0, 5] | [0, 5] |
| 2 | 100.0% | 100.0% | 100.0% | ✅ | [0, 2, 3, 7, 8] | [0, 2, 3, 7, 8] |
| 3 | 98.4% | 98.2% | 100.0% | ❌ | [0, 1, 2] | [0, 1, 2] |

### DSC Analysis
| Example | Clues Used | Stop Probs | Stop Logits |
|---------|------------|------------|-------------|
| 1 | 3.92 | [0.020, 0.022, 0.022, 0.020] | [-3.88, -3.79, -3.81, -3.90] |
| 2 | 3.91 | [0.020, 0.020, 0.026, 0.020] | [-3.89, -3.90, -3.62, -3.87] |
| 3 | 3.91 | [0.022, 0.024, 0.021, 0.023] | [-3.80, -3.70, -3.85, -3.76] |

### Attention Entropy (per clue)
| Example | Clue 0 | Clue 1 | Clue 2 | Clue 3 | Clue 4 | Clue 5 |
|---------|--------|--------|--------|--------|--------|--------|
| 1 | 0.749 | 0.247 | 0.472 | 0.672 |
| 2 | 0.784 | 0.588 | 0.083 | 0.543 |
| 3 | 0.328 | 0.322 | 0.249 | 0.395 |


## Epoch 26

**Temperature:** 0.9138

### Losses
| Loss | Value |
|------|-------|
| total_loss | 0.257116 |
| task_loss | 0.155083 |
| focal_loss | 0.155083 |
| entropy_loss | 1.534450 |
| sparsity_loss | 0.068042 |
| predicate_loss | 0.000000 |
| curriculum_loss | 2.481668 |
| deep_supervision_loss | 0.190458 |
| act_loss | 0.000000 |
| loss_mode | weighted_stablemax |
| sparsity_min_clue_penalty | 0.000000 |
| sparsity_base_pondering | 0.039147 |
| sparsity_entropy_pondering | 0.028895 |
| expected_clues_used | 3.914691 |
| stop_prob_from_loss | 0.021327 |
| clues_used_std | 0.002159 |
| per_sample_clue_penalty_mean | 0.000000 |

### Metrics
- **Total Accuracy:** 98.96%
- **BG Accuracy:** 98.63%
- **FG Accuracy:** 100.00%

### Per-Example Metrics
| Example | Accuracy | BG Acc | FG Acc | Exact Match | Target Classes | Pred Classes |
|---------|----------|--------|--------|-------------|----------------|---------------|
| 1 | 98.4% | 98.4% | 100.0% | ❌ | [0, 5] | [0, 5, 8] |
| 2 | 100.0% | 100.0% | 100.0% | ✅ | [0, 2, 3, 7, 8] | [0, 2, 3, 7, 8] |
| 3 | 98.4% | 98.2% | 100.0% | ❌ | [0, 1, 2] | [0, 1, 2] |

### DSC Analysis
| Example | Clues Used | Stop Probs | Stop Logits |
|---------|------------|------------|-------------|
| 1 | 3.91 | [0.022, 0.020, 0.023, 0.020] | [-3.79, -3.88, -3.74, -3.89] |
| 2 | 3.91 | [0.022, 0.023, 0.021, 0.021] | [-3.78, -3.76, -3.86, -3.82] |
| 3 | 3.92 | [0.022, 0.020, 0.020, 0.021] | [-3.78, -3.88, -3.90, -3.86] |

### Attention Entropy (per clue)
| Example | Clue 0 | Clue 1 | Clue 2 | Clue 3 | Clue 4 | Clue 5 |
|---------|--------|--------|--------|--------|--------|--------|
| 1 | 0.413 | 0.402 | 0.210 | 0.590 |
| 2 | 0.231 | 0.378 | 0.485 | 0.242 |
| 3 | 0.191 | 0.382 | 0.424 | 0.479 |


## Epoch 27

**Temperature:** 0.9107

### Losses
| Loss | Value |
|------|-------|
| total_loss | 0.249551 |
| task_loss | 0.150939 |
| focal_loss | 0.150939 |
| entropy_loss | 1.849804 |
| sparsity_loss | 0.073832 |
| predicate_loss | 0.000000 |
| curriculum_loss | 2.452241 |
| deep_supervision_loss | 0.182456 |
| act_loss | 0.000000 |
| loss_mode | weighted_stablemax |
| sparsity_min_clue_penalty | 0.000000 |
| sparsity_base_pondering | 0.039067 |
| sparsity_entropy_pondering | 0.034765 |
| expected_clues_used | 3.906691 |
| stop_prob_from_loss | 0.023327 |
| clues_used_std | 0.001117 |
| per_sample_clue_penalty_mean | 0.000000 |

### Metrics
- **Total Accuracy:** 90.62%
- **BG Accuracy:** 87.67%
- **FG Accuracy:** 100.00%

### Per-Example Metrics
| Example | Accuracy | BG Acc | FG Acc | Exact Match | Target Classes | Pred Classes |
|---------|----------|--------|--------|-------------|----------------|---------------|
| 1 | 90.6% | 90.5% | 100.0% | ❌ | [0, 5] | [0, 5, 7, 8] |
| 2 | 95.3% | 89.3% | 100.0% | ❌ | [0, 2, 3, 7, 8] | [0, 2, 3, 7, 8] |
| 3 | 85.9% | 83.6% | 100.0% | ❌ | [0, 1, 2] | [0, 1, 2, 3, 7, 8] |

### DSC Analysis
| Example | Clues Used | Stop Probs | Stop Logits |
|---------|------------|------------|-------------|
| 1 | 3.91 | [0.022, 0.025, 0.027, 0.021] | [-3.81, -3.68, -3.58, -3.83] |
| 2 | 3.91 | [0.027, 0.022, 0.022, 0.021] | [-3.57, -3.77, -3.81, -3.83] |
| 3 | 3.91 | [0.026, 0.023, 0.023, 0.022] | [-3.64, -3.74, -3.77, -3.82] |

### Attention Entropy (per clue)
| Example | Clue 0 | Clue 1 | Clue 2 | Clue 3 | Clue 4 | Clue 5 |
|---------|--------|--------|--------|--------|--------|--------|
| 1 | 0.624 | 0.309 | 0.466 | 0.533 |
| 2 | 0.280 | 0.469 | 0.357 | 0.581 |
| 3 | 0.305 | 0.403 | 0.479 | 0.533 |


## Epoch 28

**Temperature:** 0.9075

### Losses
| Loss | Value |
|------|-------|
| total_loss | 0.238137 |
| task_loss | 0.143812 |
| focal_loss | 0.143812 |
| entropy_loss | 1.849967 |
| sparsity_loss | 0.073781 |
| predicate_loss | 0.000000 |
| curriculum_loss | 2.436879 |
| deep_supervision_loss | 0.173894 |
| act_loss | 0.000000 |
| loss_mode | weighted_stablemax |
| sparsity_min_clue_penalty | 0.000000 |
| sparsity_base_pondering | 0.039044 |
| sparsity_entropy_pondering | 0.034737 |
| expected_clues_used | 3.904355 |
| stop_prob_from_loss | 0.023911 |
| clues_used_std | 0.004664 |
| per_sample_clue_penalty_mean | 0.000000 |

### Metrics
- **Total Accuracy:** 99.48%
- **BG Accuracy:** 100.00%
- **FG Accuracy:** 97.83%

### Per-Example Metrics
| Example | Accuracy | BG Acc | FG Acc | Exact Match | Target Classes | Pred Classes |
|---------|----------|--------|--------|-------------|----------------|---------------|
| 1 | 98.4% | 100.0% | 0.0% | ❌ | [0, 5] | [0] |
| 2 | 100.0% | 100.0% | 100.0% | ✅ | [0, 2, 3, 7, 8] | [0, 2, 3, 7, 8] |
| 3 | 100.0% | 100.0% | 100.0% | ✅ | [0, 1, 2] | [0, 1, 2] |

### DSC Analysis
| Example | Clues Used | Stop Probs | Stop Logits |
|---------|------------|------------|-------------|
| 1 | 3.90 | [0.026, 0.023, 0.024, 0.024] | [-3.63, -3.76, -3.70, -3.71] |
| 2 | 3.91 | [0.023, 0.023, 0.021, 0.023] | [-3.73, -3.74, -3.83, -3.76] |
| 3 | 3.90 | [0.026, 0.026, 0.021, 0.026] | [-3.63, -3.61, -3.82, -3.61] |

### Attention Entropy (per clue)
| Example | Clue 0 | Clue 1 | Clue 2 | Clue 3 | Clue 4 | Clue 5 |
|---------|--------|--------|--------|--------|--------|--------|
| 1 | 0.408 | 0.396 | 0.352 | 0.521 |
| 2 | 0.564 | 0.255 | 0.431 | 0.510 |
| 3 | 0.735 | 0.370 | 0.550 | 0.246 |


## Epoch 29

**Temperature:** 0.9044

### Losses
| Loss | Value |
|------|-------|
| total_loss | 0.224295 |
| task_loss | 0.134344 |
| focal_loss | 0.134344 |
| entropy_loss | 1.770055 |
| sparsity_loss | 0.072341 |
| predicate_loss | 0.000000 |
| curriculum_loss | 2.428038 |
| deep_supervision_loss | 0.165433 |
| act_loss | 0.000000 |
| loss_mode | weighted_stablemax |
| sparsity_min_clue_penalty | 0.000000 |
| sparsity_base_pondering | 0.039066 |
| sparsity_entropy_pondering | 0.033275 |
| expected_clues_used | 3.906588 |
| stop_prob_from_loss | 0.023353 |
| clues_used_std | 0.010889 |
| per_sample_clue_penalty_mean | 0.000000 |

### Metrics
- **Total Accuracy:** 99.48%
- **BG Accuracy:** 100.00%
- **FG Accuracy:** 97.83%

### Per-Example Metrics
| Example | Accuracy | BG Acc | FG Acc | Exact Match | Target Classes | Pred Classes |
|---------|----------|--------|--------|-------------|----------------|---------------|
| 1 | 98.4% | 100.0% | 0.0% | ❌ | [0, 5] | [0] |
| 2 | 100.0% | 100.0% | 100.0% | ✅ | [0, 2, 3, 7, 8] | [0, 2, 3, 7, 8] |
| 3 | 100.0% | 100.0% | 100.0% | ✅ | [0, 1, 2] | [0, 1, 2] |

### DSC Analysis
| Example | Clues Used | Stop Probs | Stop Logits |
|---------|------------|------------|-------------|
| 1 | 3.91 | [0.022, 0.020, 0.022, 0.022] | [-3.78, -3.90, -3.80, -3.80] |
| 2 | 3.89 | [0.024, 0.026, 0.033, 0.023] | [-3.71, -3.62, -3.37, -3.76] |
| 3 | 3.91 | [0.022, 0.021, 0.023, 0.022] | [-3.78, -3.84, -3.73, -3.80] |

### Attention Entropy (per clue)
| Example | Clue 0 | Clue 1 | Clue 2 | Clue 3 | Clue 4 | Clue 5 |
|---------|--------|--------|--------|--------|--------|--------|
| 1 | 0.581 | 0.600 | 0.338 | 0.533 |
| 2 | 0.678 | 0.287 | 0.187 | 0.426 |
| 3 | 0.423 | 0.393 | 0.269 | 0.393 |


## Epoch 30

**Temperature:** 0.9013

### Losses
| Loss | Value |
|------|-------|
| total_loss | 0.220524 |
| task_loss | 0.131555 |
| focal_loss | 0.131555 |
| entropy_loss | 1.583947 |
| sparsity_loss | 0.068783 |
| predicate_loss | 0.000000 |
| curriculum_loss | 2.409664 |
| deep_supervision_loss | 0.164181 |
| act_loss | 0.000000 |
| loss_mode | weighted_stablemax |
| sparsity_min_clue_penalty | 0.000000 |
| sparsity_base_pondering | 0.039021 |
| sparsity_entropy_pondering | 0.029762 |
| expected_clues_used | 3.902135 |
| stop_prob_from_loss | 0.024466 |
| clues_used_std | 0.008811 |
| per_sample_clue_penalty_mean | 0.000000 |

### Metrics
- **Total Accuracy:** 96.88%
- **BG Accuracy:** 95.89%
- **FG Accuracy:** 100.00%

### Per-Example Metrics
| Example | Accuracy | BG Acc | FG Acc | Exact Match | Target Classes | Pred Classes |
|---------|----------|--------|--------|-------------|----------------|---------------|
| 1 | 100.0% | 100.0% | 100.0% | ✅ | [0, 5] | [0, 5] |
| 2 | 90.6% | 78.6% | 100.0% | ❌ | [0, 2, 3, 7, 8] | [0, 2, 3, 7, 8] |
| 3 | 100.0% | 100.0% | 100.0% | ✅ | [0, 1, 2] | [0, 1, 2] |

### DSC Analysis
| Example | Clues Used | Stop Probs | Stop Logits |
|---------|------------|------------|-------------|
| 1 | 3.90 | [0.023, 0.021, 0.030, 0.027] | [-3.75, -3.83, -3.46, -3.59] |
| 2 | 3.90 | [0.028, 0.023, 0.025, 0.028] | [-3.54, -3.73, -3.66, -3.56] |
| 3 | 3.91 | [0.020, 0.022, 0.025, 0.021] | [-3.88, -3.79, -3.67, -3.86] |

### Attention Entropy (per clue)
| Example | Clue 0 | Clue 1 | Clue 2 | Clue 3 | Clue 4 | Clue 5 |
|---------|--------|--------|--------|--------|--------|--------|
| 1 | 0.687 | 0.392 | 0.021 | 0.142 |
| 2 | 0.471 | 0.398 | 0.509 | 0.096 |
| 3 | 0.703 | 0.544 | 0.186 | 0.422 |

### Gradient Norms (selected modules)
- **encoder:** 0.016923
- **feature_proj:** 0.026538
- **context_encoder:** 0.041109
- **context_injector:** 0.017603
- **dsc:** 0.017839
- **msre:** 0.026241
- **solver:** 0.710535

### Predictions vs Targets

**Example 1:**

Target:
```
0 0 0 0
0 0 0 0
0 0 0 0
0 0 0 5
```

Prediction:
```
0 0 0 0
0 0 0 0
0 0 0 0
0 0 0 5
```

**Example 2:**

Target:
```
3 2 3 2 3 2
7 8 7 8 7 8
2 3 2 3 2 3
8 7 8 7 8 7
3 2 3 2 3 2
7 8 7 8 7 8
```

Prediction:
```
3 2 3 2 3 2
7 8 7 8 7 8
2 3 2 3 2 3
8 7 8 7 8 7
3 2 3 2 3 2
7 8 7 8 7 8
```

**Example 3:**

Target:
```
2 2 2
1 1 1
1 1 1
```

Prediction:
```
2 2 2
1 1 1
1 1 1
```


## Epoch 31

**Temperature:** 0.8981

### Losses
| Loss | Value |
|------|-------|
| total_loss | 0.216866 |
| task_loss | 0.130478 |
| focal_loss | 0.130478 |
| entropy_loss | 1.710302 |
| sparsity_loss | 0.071208 |
| predicate_loss | 0.000000 |
| curriculum_loss | 2.393344 |
| deep_supervision_loss | 0.158534 |
| act_loss | 0.000000 |
| loss_mode | weighted_stablemax |
| sparsity_min_clue_penalty | 0.000000 |
| sparsity_base_pondering | 0.039068 |
| sparsity_entropy_pondering | 0.032140 |
| expected_clues_used | 3.906849 |
| stop_prob_from_loss | 0.023288 |
| clues_used_std | 0.007897 |
| per_sample_clue_penalty_mean | 0.000000 |

### Metrics
- **Total Accuracy:** 93.75%
- **BG Accuracy:** 91.78%
- **FG Accuracy:** 100.00%

### Per-Example Metrics
| Example | Accuracy | BG Acc | FG Acc | Exact Match | Target Classes | Pred Classes |
|---------|----------|--------|--------|-------------|----------------|---------------|
| 1 | 87.5% | 87.3% | 100.0% | ❌ | [0, 5] | [0, 2, 3, 5, 7, 8] |
| 2 | 93.8% | 85.7% | 100.0% | ❌ | [0, 2, 3, 7, 8] | [0, 2, 3, 7, 8] |
| 3 | 100.0% | 100.0% | 100.0% | ✅ | [0, 1, 2] | [0, 1, 2] |

### DSC Analysis
| Example | Clues Used | Stop Probs | Stop Logits |
|---------|------------|------------|-------------|
| 1 | 3.91 | [0.025, 0.020, 0.022, 0.022] | [-3.67, -3.88, -3.77, -3.82] |
| 2 | 3.90 | [0.028, 0.025, 0.027, 0.022] | [-3.54, -3.67, -3.59, -3.78] |
| 3 | 3.91 | [0.021, 0.024, 0.023, 0.021] | [-3.84, -3.72, -3.76, -3.86] |

### Attention Entropy (per clue)
| Example | Clue 0 | Clue 1 | Clue 2 | Clue 3 | Clue 4 | Clue 5 |
|---------|--------|--------|--------|--------|--------|--------|
| 1 | 0.181 | 0.511 | 0.423 | 0.434 |
| 2 | 0.664 | 0.236 | 0.256 | 0.506 |
| 3 | 0.669 | 0.322 | 0.313 | 0.421 |


## Epoch 32

**Temperature:** 0.8950

### Losses
| Loss | Value |
|------|-------|
| total_loss | 0.210462 |
| task_loss | 0.122068 |
| focal_loss | 0.122068 |
| entropy_loss | 1.776805 |
| sparsity_loss | 0.072581 |
| predicate_loss | 0.000000 |
| curriculum_loss | 2.391077 |
| deep_supervision_loss | 0.162272 |
| act_loss | 0.000000 |
| loss_mode | weighted_stablemax |
| sparsity_min_clue_penalty | 0.000000 |
| sparsity_base_pondering | 0.039133 |
| sparsity_entropy_pondering | 0.033447 |
| expected_clues_used | 3.913337 |
| stop_prob_from_loss | 0.021666 |
| clues_used_std | 0.007263 |
| per_sample_clue_penalty_mean | 0.000000 |

### Metrics
- **Total Accuracy:** 100.00%
- **BG Accuracy:** 100.00%
- **FG Accuracy:** 100.00%

### Per-Example Metrics
| Example | Accuracy | BG Acc | FG Acc | Exact Match | Target Classes | Pred Classes |
|---------|----------|--------|--------|-------------|----------------|---------------|
| 1 | 100.0% | 100.0% | 100.0% | ✅ | [0, 5] | [0, 5] |
| 2 | 100.0% | 100.0% | 100.0% | ✅ | [0, 2, 3, 7, 8] | [0, 2, 3, 7, 8] |
| 3 | 100.0% | 100.0% | 100.0% | ✅ | [0, 1, 2] | [0, 1, 2] |

### DSC Analysis
| Example | Clues Used | Stop Probs | Stop Logits |
|---------|------------|------------|-------------|
| 1 | 3.92 | [0.020, 0.019, 0.021, 0.021] | [-3.90, -3.94, -3.85, -3.86] |
| 2 | 3.91 | [0.024, 0.022, 0.028, 0.020] | [-3.70, -3.81, -3.53, -3.87] |
| 3 | 3.91 | [0.022, 0.022, 0.021, 0.020] | [-3.81, -3.79, -3.83, -3.88] |

### Attention Entropy (per clue)
| Example | Clue 0 | Clue 1 | Clue 2 | Clue 3 | Clue 4 | Clue 5 |
|---------|--------|--------|--------|--------|--------|--------|
| 1 | 0.671 | 0.478 | 0.267 | 0.487 |
| 2 | 0.674 | 0.405 | 0.229 | 0.478 |
| 3 | 0.439 | 0.286 | 0.241 | 0.472 |


## 🎉 SUCCESS: All examples solved at epoch 32!

---

# Final Summary

- **Best Accuracy:** 100.00% (epoch 32)
- **All Exact Match:** ✅ Yes

