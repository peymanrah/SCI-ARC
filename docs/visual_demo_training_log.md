# RLAN Visual Demo Training Log

**Date:** 2025-12-17 20:35:18
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
- min_clues: 2.0
- min_clue_weight: 5.0
- lambda_deep_supervision: 0.5

---

# Training Progress

## Epoch 1

**Temperature:** 0.9965

### Losses
| Loss | Value |
|------|-------|
| total_loss | 1.272954 |
| task_loss | 0.848060 |
| focal_loss | 0.848060 |
| entropy_loss | 2.656512 |
| sparsity_loss | 0.071821 |
| predicate_loss | 0.000000 |
| curriculum_loss | 1.896614 |
| deep_supervision_loss | 0.835424 |
| act_loss | 0.000000 |
| loss_mode | weighted_stablemax |
| sparsity_min_clue_penalty | 0.000000 |
| sparsity_base_pondering | 0.031380 |
| sparsity_entropy_pondering | 0.040442 |
| expected_clues_used | 3.137985 |
| stop_prob_from_loss | 0.215504 |
| clues_used_std | 0.156469 |
| per_sample_clue_penalty_mean | 0.000000 |

### Metrics
- **Total Accuracy:** 4.17%
- **BG Accuracy:** 2.05%
- **FG Accuracy:** 10.87%

### Per-Example Metrics
| Example | Accuracy | BG Acc | FG Acc | Exact Match | Target Classes | Pred Classes |
|---------|----------|--------|--------|-------------|----------------|---------------|
| 1 | 3.1% | 3.2% | 0.0% | ❌ | [0, 5] | [0, 1, 3, 4, 6, 7, 8] |
| 2 | 7.8% | 0.0% | 13.9% | ❌ | [0, 2, 3, 7, 8] | [1, 3, 4, 6, 7, 8, 9] |
| 3 | 1.6% | 1.8% | 0.0% | ❌ | [0, 1, 2] | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] |

### DSC Analysis
| Example | Clues Used | Stop Probs | Stop Logits |
|---------|------------|------------|-------------|
| 1 | 3.06 | [0.236, 0.218, 0.241, 0.242] | [-1.18, -1.28, -1.15, -1.14] |
| 2 | 3.03 | [0.347, 0.132, 0.140, 0.348] | [-0.63, -1.88, -1.82, -0.63] |
| 3 | 3.32 | [0.129, 0.125, 0.241, 0.187] | [-1.91, -1.94, -1.15, -1.47] |

### Attention Entropy (per clue)
| Example | Clue 0 | Clue 1 | Clue 2 | Clue 3 | Clue 4 | Clue 5 |
|---------|--------|--------|--------|--------|--------|--------|
| 1 | 0.777 | 0.751 | 0.386 | 0.709 |
| 2 | 0.530 | 0.811 | 0.802 | 0.689 |
| 3 | 0.762 | 0.648 | 0.284 | 0.515 |

### Gradient Norms (selected modules)
- **encoder:** 0.139968
- **feature_proj:** 0.253349
- **context_encoder:** 0.765009
- **context_injector:** 0.130557
- **dsc:** 0.203252
- **msre:** 0.237928
- **solver:** 4.920971

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
8 6 8 6
4 3 3 4
3 6 6 3
3 4 6 3
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
8 3 8 8 7 6
3 3 3 9 3 3
4 3 4 4 3 3
3 3 9 7 7 1
3 3 7 4 4 1
3 3 3 7 7 3
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
8 0 3
4 4 8
3 3 7
```


## Epoch 2

**Temperature:** 0.9931

### Losses
| Loss | Value |
|------|-------|
| total_loss | 0.939997 |
| task_loss | 0.620140 |
| focal_loss | 0.620140 |
| entropy_loss | 2.455837 |
| sparsity_loss | 0.078965 |
| predicate_loss | 0.000000 |
| curriculum_loss | 2.417709 |
| deep_supervision_loss | 0.623921 |
| act_loss | 0.000000 |
| loss_mode | weighted_stablemax |
| sparsity_min_clue_penalty | 0.000000 |
| sparsity_base_pondering | 0.036134 |
| sparsity_entropy_pondering | 0.042831 |
| expected_clues_used | 3.613421 |
| stop_prob_from_loss | 0.096645 |
| clues_used_std | 0.058512 |
| per_sample_clue_penalty_mean | 0.000000 |

### Metrics
- **Total Accuracy:** 75.52%
- **BG Accuracy:** 91.78%
- **FG Accuracy:** 23.91%

### Per-Example Metrics
| Example | Accuracy | BG Acc | FG Acc | Exact Match | Target Classes | Pred Classes |
|---------|----------|--------|--------|-------------|----------------|---------------|
| 1 | 85.9% | 87.3% | 0.0% | ❌ | [0, 5] | [0, 2, 8] |
| 2 | 54.7% | 100.0% | 19.4% | ❌ | [0, 2, 3, 7, 8] | [0, 2, 3, 7, 8] |
| 3 | 85.9% | 92.7% | 44.4% | ❌ | [0, 1, 2] | [0, 1, 2, 3, 8] |

### DSC Analysis
| Example | Clues Used | Stop Probs | Stop Logits |
|---------|------------|------------|-------------|
| 1 | 3.68 | [0.067, 0.125, 0.048, 0.078] | [-2.63, -1.94, -2.98, -2.47] |
| 2 | 3.58 | [0.098, 0.093, 0.132, 0.096] | [-2.22, -2.27, -1.88, -2.24] |
| 3 | 3.58 | [0.103, 0.142, 0.060, 0.116] | [-2.16, -1.80, -2.75, -2.04] |

### Attention Entropy (per clue)
| Example | Clue 0 | Clue 1 | Clue 2 | Clue 3 | Clue 4 | Clue 5 |
|---------|--------|--------|--------|--------|--------|--------|
| 1 | 0.681 | 0.624 | 0.844 | 0.790 |
| 2 | 0.824 | 0.386 | 0.875 | 0.406 |
| 3 | 0.329 | 0.551 | 0.696 | 0.079 |

### Gradient Norms (selected modules)
- **encoder:** 0.075018
- **feature_proj:** 0.132185
- **context_encoder:** 0.178830
- **context_injector:** 0.067429
- **dsc:** 0.062798
- **msre:** 0.123535
- **solver:** 1.466049


## Epoch 3

**Temperature:** 0.9897

### Losses
| Loss | Value |
|------|-------|
| total_loss | 0.899619 |
| task_loss | 0.589897 |
| focal_loss | 0.589897 |
| entropy_loss | 2.716869 |
| sparsity_loss | 0.080496 |
| predicate_loss | 0.000000 |
| curriculum_loss | 2.292672 |
| deep_supervision_loss | 0.603344 |
| act_loss | 0.000000 |
| loss_mode | weighted_stablemax |
| sparsity_min_clue_penalty | 0.000000 |
| sparsity_base_pondering | 0.034789 |
| sparsity_entropy_pondering | 0.045707 |
| expected_clues_used | 3.478912 |
| stop_prob_from_loss | 0.130272 |
| clues_used_std | 0.062807 |
| per_sample_clue_penalty_mean | 0.000000 |

### Metrics
- **Total Accuracy:** 56.77%
- **BG Accuracy:** 62.33%
- **FG Accuracy:** 39.13%

### Per-Example Metrics
| Example | Accuracy | BG Acc | FG Acc | Exact Match | Target Classes | Pred Classes |
|---------|----------|--------|--------|-------------|----------------|---------------|
| 1 | 51.6% | 52.4% | 0.0% | ❌ | [0, 5] | [0, 1, 2, 7] |
| 2 | 53.1% | 89.3% | 25.0% | ❌ | [0, 2, 3, 7, 8] | [0, 1, 2, 7] |
| 3 | 65.6% | 60.0% | 100.0% | ❌ | [0, 1, 2] | [0, 1, 2, 7] |

### DSC Analysis
| Example | Clues Used | Stop Probs | Stop Logits |
|---------|------------|------------|-------------|
| 1 | 3.41 | [0.150, 0.134, 0.102, 0.207] | [-1.73, -1.87, -2.17, -1.34] |
| 2 | 3.51 | [0.094, 0.091, 0.155, 0.150] | [-2.26, -2.31, -1.69, -1.73] |
| 3 | 3.52 | [0.123, 0.099, 0.149, 0.110] | [-1.97, -2.21, -1.74, -2.09] |

### Attention Entropy (per clue)
| Example | Clue 0 | Clue 1 | Clue 2 | Clue 3 | Clue 4 | Clue 5 |
|---------|--------|--------|--------|--------|--------|--------|
| 1 | 0.676 | 0.644 | 0.792 | 0.529 |
| 2 | 0.867 | 0.771 | 0.588 | 0.256 |
| 3 | 0.494 | 0.714 | 0.697 | 0.813 |

### Gradient Norms (selected modules)
- **encoder:** 0.067350
- **feature_proj:** 0.119863
- **context_encoder:** 0.134096
- **context_injector:** 0.053008
- **dsc:** 0.089557
- **msre:** 0.085257
- **solver:** 2.289415


## Epoch 4

**Temperature:** 0.9862

### Losses
| Loss | Value |
|------|-------|
| total_loss | 0.863184 |
| task_loss | 0.569675 |
| focal_loss | 0.569675 |
| entropy_loss | 2.765840 |
| sparsity_loss | 0.058031 |
| predicate_loss | 0.000000 |
| curriculum_loss | 1.233965 |
| deep_supervision_loss | 0.575411 |
| act_loss | 0.000000 |
| loss_mode | weighted_stablemax |
| sparsity_min_clue_penalty | 0.000000 |
| sparsity_base_pondering | 0.024554 |
| sparsity_entropy_pondering | 0.033477 |
| expected_clues_used | 2.455351 |
| stop_prob_from_loss | 0.386162 |
| clues_used_std | 0.140143 |
| per_sample_clue_penalty_mean | 0.000000 |

### Metrics
- **Total Accuracy:** 52.08%
- **BG Accuracy:** 58.22%
- **FG Accuracy:** 32.61%

### Per-Example Metrics
| Example | Accuracy | BG Acc | FG Acc | Exact Match | Target Classes | Pred Classes |
|---------|----------|--------|--------|-------------|----------------|---------------|
| 1 | 50.0% | 50.8% | 0.0% | ❌ | [0, 5] | [0, 2, 8] |
| 2 | 54.7% | 85.7% | 30.6% | ❌ | [0, 2, 3, 7, 8] | [0, 2, 8] |
| 3 | 51.6% | 52.7% | 44.4% | ❌ | [0, 1, 2] | [0, 1, 2, 8] |

### DSC Analysis
| Example | Clues Used | Stop Probs | Stop Logits |
|---------|------------|------------|-------------|
| 1 | 2.60 | [0.321, 0.274, 0.588, 0.222] | [-0.75, -0.98, 0.35, -1.25] |
| 2 | 2.32 | [0.426, 0.403, 0.446, 0.410] | [-0.30, -0.39, -0.22, -0.36] |
| 3 | 2.46 | [0.292, 0.330, 0.540, 0.382] | [-0.88, -0.71, 0.16, -0.48] |

### Attention Entropy (per clue)
| Example | Clue 0 | Clue 1 | Clue 2 | Clue 3 | Clue 4 | Clue 5 |
|---------|--------|--------|--------|--------|--------|--------|
| 1 | 0.624 | 0.810 | 0.599 | 0.742 |
| 2 | 0.530 | 0.607 | 0.724 | 0.773 |
| 3 | 0.706 | 0.872 | 0.229 | 0.764 |

### Gradient Norms (selected modules)
- **encoder:** 0.027486
- **feature_proj:** 0.049699
- **context_encoder:** 0.089759
- **context_injector:** 0.031861
- **dsc:** 0.045640
- **msre:** 0.045032
- **solver:** 1.454014


## Epoch 5

**Temperature:** 0.9828

### Losses
| Loss | Value |
|------|-------|
| total_loss | 0.994893 |
| task_loss | 0.723063 |
| focal_loss | 0.723063 |
| entropy_loss | 2.417895 |
| sparsity_loss | 0.037614 |
| predicate_loss | 0.000000 |
| curriculum_loss | 0.699293 |
| deep_supervision_loss | 0.536136 |
| act_loss | 0.000000 |
| loss_mode | weighted_stablemax |
| sparsity_min_clue_penalty | 0.391718 |
| sparsity_base_pondering | 0.016963 |
| sparsity_entropy_pondering | 0.020652 |
| expected_clues_used | 1.696251 |
| stop_prob_from_loss | 0.575937 |
| clues_used_std | 0.755051 |
| per_sample_clue_penalty_mean | 0.195859 |

### Metrics
- **Total Accuracy:** 66.15%
- **BG Accuracy:** 76.71%
- **FG Accuracy:** 32.61%

### Per-Example Metrics
| Example | Accuracy | BG Acc | FG Acc | Exact Match | Target Classes | Pred Classes |
|---------|----------|--------|--------|-------------|----------------|---------------|
| 1 | 68.8% | 69.8% | 0.0% | ❌ | [0, 5] | [0, 2, 8] |
| 2 | 54.7% | 92.9% | 25.0% | ❌ | [0, 2, 3, 7, 8] | [0, 2, 7, 8] |
| 3 | 75.0% | 76.4% | 66.7% | ❌ | [0, 1, 2] | [0, 1, 2, 8] |

### DSC Analysis
| Example | Clues Used | Stop Probs | Stop Logits |
|---------|------------|------------|-------------|
| 1 | 2.11 | [0.523, 0.424, 0.326, 0.620] | [0.09, -0.31, -0.73, 0.49] |
| 2 | 0.82 | [0.723, 0.837, 0.837, 0.777] | [0.96, 1.64, 1.64, 1.25] |
| 3 | 2.16 | [0.360, 0.577, 0.457, 0.449] | [-0.57, 0.31, -0.17, -0.20] |

### Attention Entropy (per clue)
| Example | Clue 0 | Clue 1 | Clue 2 | Clue 3 | Clue 4 | Clue 5 |
|---------|--------|--------|--------|--------|--------|--------|
| 1 | 0.893 | 0.863 | 0.645 | 0.746 |
| 2 | 0.293 | 0.495 | 0.300 | 0.768 |
| 3 | 0.255 | 0.226 | 0.767 | 0.725 |

### Gradient Norms (selected modules)
- **encoder:** 0.818837
- **feature_proj:** 1.504406
- **context_encoder:** 2.012478
- **context_injector:** 1.122653
- **dsc:** 2.429895
- **msre:** 0.128634
- **solver:** 1.437052


## Epoch 6

**Temperature:** 0.9794

### Losses
| Loss | Value |
|------|-------|
| total_loss | 0.735467 |
| task_loss | 0.482081 |
| focal_loss | 0.482081 |
| entropy_loss | 2.629043 |
| sparsity_loss | 0.083883 |
| predicate_loss | 0.000000 |
| curriculum_loss | 2.460750 |
| deep_supervision_loss | 0.489995 |
| act_loss | 0.000000 |
| loss_mode | weighted_stablemax |
| sparsity_min_clue_penalty | 0.000000 |
| sparsity_base_pondering | 0.036848 |
| sparsity_entropy_pondering | 0.047035 |
| expected_clues_used | 3.684789 |
| stop_prob_from_loss | 0.078803 |
| clues_used_std | 0.071059 |
| per_sample_clue_penalty_mean | 0.000000 |

### Metrics
- **Total Accuracy:** 55.21%
- **BG Accuracy:** 53.42%
- **FG Accuracy:** 60.87%

### Per-Example Metrics
| Example | Accuracy | BG Acc | FG Acc | Exact Match | Target Classes | Pred Classes |
|---------|----------|--------|--------|-------------|----------------|---------------|
| 1 | 45.3% | 46.0% | 0.0% | ❌ | [0, 5] | [0, 1, 2, 3, 7, 8] |
| 2 | 67.2% | 85.7% | 52.8% | ❌ | [0, 2, 3, 7, 8] | [0, 1, 2, 3, 7, 8] |
| 3 | 53.1% | 45.5% | 100.0% | ❌ | [0, 1, 2] | [0, 1, 2, 3, 7, 8] |

### DSC Analysis
| Example | Clues Used | Stop Probs | Stop Logits |
|---------|------------|------------|-------------|
| 1 | 3.75 | [0.046, 0.070, 0.066, 0.063] | [-3.03, -2.58, -2.65, -2.70] |
| 2 | 3.61 | [0.085, 0.081, 0.068, 0.154] | [-2.38, -2.42, -2.62, -1.70] |
| 3 | 3.69 | [0.123, 0.082, 0.052, 0.054] | [-1.96, -2.41, -2.90, -2.86] |

### Attention Entropy (per clue)
| Example | Clue 0 | Clue 1 | Clue 2 | Clue 3 | Clue 4 | Clue 5 |
|---------|--------|--------|--------|--------|--------|--------|
| 1 | 0.789 | 0.815 | 0.511 | 0.709 |
| 2 | 0.781 | 0.665 | 0.752 | 0.203 |
| 3 | 0.178 | 0.609 | 0.780 | 0.793 |


## Epoch 7

**Temperature:** 0.9760

### Losses
| Loss | Value |
|------|-------|
| total_loss | 0.779672 |
| task_loss | 0.510056 |
| focal_loss | 0.510056 |
| entropy_loss | 2.285871 |
| sparsity_loss | 0.081055 |
| predicate_loss | 0.000000 |
| curriculum_loss | 2.656004 |
| deep_supervision_loss | 0.523022 |
| act_loss | 0.000000 |
| loss_mode | weighted_stablemax |
| sparsity_min_clue_penalty | 0.000000 |
| sparsity_base_pondering | 0.038445 |
| sparsity_entropy_pondering | 0.042610 |
| expected_clues_used | 3.844494 |
| stop_prob_from_loss | 0.038877 |
| clues_used_std | 0.026056 |
| per_sample_clue_penalty_mean | 0.000000 |

### Metrics
- **Total Accuracy:** 72.40%
- **BG Accuracy:** 85.62%
- **FG Accuracy:** 30.43%

### Per-Example Metrics
| Example | Accuracy | BG Acc | FG Acc | Exact Match | Target Classes | Pred Classes |
|---------|----------|--------|--------|-------------|----------------|---------------|
| 1 | 81.2% | 82.5% | 0.0% | ❌ | [0, 5] | [0, 2, 7, 8] |
| 2 | 50.0% | 96.4% | 13.9% | ❌ | [0, 2, 3, 7, 8] | [0, 1, 2, 7] |
| 3 | 85.9% | 83.6% | 100.0% | ❌ | [0, 1, 2] | [0, 1, 2, 7, 8] |

### DSC Analysis
| Example | Clues Used | Stop Probs | Stop Logits |
|---------|------------|------------|-------------|
| 1 | 3.83 | [0.029, 0.080, 0.037, 0.026] | [-3.51, -2.44, -3.26, -3.61] |
| 2 | 3.87 | [0.032, 0.040, 0.027, 0.026] | [-3.40, -3.17, -3.59, -3.62] |
| 3 | 3.83 | [0.027, 0.056, 0.057, 0.029] | [-3.58, -2.83, -2.81, -3.50] |

### Attention Entropy (per clue)
| Example | Clue 0 | Clue 1 | Clue 2 | Clue 3 | Clue 4 | Clue 5 |
|---------|--------|--------|--------|--------|--------|--------|
| 1 | 0.720 | 0.006 | 0.543 | 0.881 |
| 2 | 0.554 | 0.392 | 0.714 | 0.891 |
| 3 | 0.782 | 0.202 | 0.157 | 0.753 |


## Epoch 8

**Temperature:** 0.9727

### Losses
| Loss | Value |
|------|-------|
| total_loss | 0.719687 |
| task_loss | 0.470444 |
| focal_loss | 0.470444 |
| entropy_loss | 2.574888 |
| sparsity_loss | 0.087131 |
| predicate_loss | 0.000000 |
| curriculum_loss | 2.723901 |
| deep_supervision_loss | 0.481059 |
| act_loss | 0.000000 |
| loss_mode | weighted_stablemax |
| sparsity_min_clue_penalty | 0.000000 |
| sparsity_base_pondering | 0.038907 |
| sparsity_entropy_pondering | 0.048224 |
| expected_clues_used | 3.890724 |
| stop_prob_from_loss | 0.027319 |
| clues_used_std | 0.002301 |
| per_sample_clue_penalty_mean | 0.000000 |

### Metrics
- **Total Accuracy:** 56.77%
- **BG Accuracy:** 54.79%
- **FG Accuracy:** 63.04%

### Per-Example Metrics
| Example | Accuracy | BG Acc | FG Acc | Exact Match | Target Classes | Pred Classes |
|---------|----------|--------|--------|-------------|----------------|---------------|
| 1 | 39.1% | 39.7% | 0.0% | ❌ | [0, 5] | [0, 2, 3, 7] |
| 2 | 71.9% | 89.3% | 58.3% | ❌ | [0, 2, 3, 7, 8] | [0, 2, 3, 7, 8] |
| 3 | 59.4% | 54.5% | 88.9% | ❌ | [0, 1, 2] | [0, 1, 2, 3, 7] |

### DSC Analysis
| Example | Clues Used | Stop Probs | Stop Logits |
|---------|------------|------------|-------------|
| 1 | 3.89 | [0.025, 0.030, 0.027, 0.028] | [-3.65, -3.47, -3.57, -3.56] |
| 2 | 3.89 | [0.031, 0.023, 0.027, 0.025] | [-3.44, -3.74, -3.59, -3.64] |
| 3 | 3.89 | [0.023, 0.027, 0.023, 0.037] | [-3.75, -3.58, -3.73, -3.26] |

### Attention Entropy (per clue)
| Example | Clue 0 | Clue 1 | Clue 2 | Clue 3 | Clue 4 | Clue 5 |
|---------|--------|--------|--------|--------|--------|--------|
| 1 | 0.710 | 0.388 | 0.594 | 0.741 |
| 2 | 0.322 | 0.851 | 0.674 | 0.653 |
| 3 | 0.814 | 0.702 | 0.669 | 0.312 |


## Epoch 9

**Temperature:** 0.9693

### Losses
| Loss | Value |
|------|-------|
| total_loss | 0.697296 |
| task_loss | 0.453811 |
| focal_loss | 0.453811 |
| entropy_loss | 2.806804 |
| sparsity_loss | 0.091823 |
| predicate_loss | 0.000000 |
| curriculum_loss | 2.728562 |
| deep_supervision_loss | 0.468606 |
| act_loss | 0.000000 |
| loss_mode | weighted_stablemax |
| sparsity_min_clue_penalty | 0.000000 |
| sparsity_base_pondering | 0.039072 |
| sparsity_entropy_pondering | 0.052751 |
| expected_clues_used | 3.907152 |
| stop_prob_from_loss | 0.023212 |
| clues_used_std | 0.006037 |
| per_sample_clue_penalty_mean | 0.000000 |

### Metrics
- **Total Accuracy:** 55.73%
- **BG Accuracy:** 48.63%
- **FG Accuracy:** 78.26%

### Per-Example Metrics
| Example | Accuracy | BG Acc | FG Acc | Exact Match | Target Classes | Pred Classes |
|---------|----------|--------|--------|-------------|----------------|---------------|
| 1 | 35.9% | 36.5% | 0.0% | ❌ | [0, 5] | [0, 2, 3, 7, 8] |
| 2 | 81.2% | 85.7% | 77.8% | ❌ | [0, 2, 3, 7, 8] | [0, 2, 3, 7, 8] |
| 3 | 50.0% | 43.6% | 88.9% | ❌ | [0, 1, 2] | [0, 1, 2, 3, 7, 8] |

### DSC Analysis
| Example | Clues Used | Stop Probs | Stop Logits |
|---------|------------|------------|-------------|
| 1 | 3.90 | [0.026, 0.022, 0.028, 0.024] | [-3.64, -3.78, -3.55, -3.72] |
| 2 | 3.91 | [0.021, 0.021, 0.022, 0.025] | [-3.85, -3.86, -3.79, -3.68] |
| 3 | 3.91 | [0.025, 0.022, 0.021, 0.023] | [-3.68, -3.77, -3.85, -3.77] |

### Attention Entropy (per clue)
| Example | Clue 0 | Clue 1 | Clue 2 | Clue 3 | Clue 4 | Clue 5 |
|---------|--------|--------|--------|--------|--------|--------|
| 1 | 0.659 | 0.636 | 0.428 | 0.576 |
| 2 | 0.709 | 0.756 | 0.688 | 0.651 |
| 3 | 0.740 | 0.595 | 0.899 | 0.762 |


## Epoch 10

**Temperature:** 0.9659

### Losses
| Loss | Value |
|------|-------|
| total_loss | 0.676292 |
| task_loss | 0.440153 |
| focal_loss | 0.440153 |
| entropy_loss | 2.685732 |
| sparsity_loss | 0.089600 |
| predicate_loss | 0.000000 |
| curriculum_loss | 2.715732 |
| deep_supervision_loss | 0.454358 |
| act_loss | 0.000000 |
| loss_mode | weighted_stablemax |
| sparsity_min_clue_penalty | 0.000000 |
| sparsity_base_pondering | 0.039093 |
| sparsity_entropy_pondering | 0.050507 |
| expected_clues_used | 3.909319 |
| stop_prob_from_loss | 0.022670 |
| clues_used_std | 0.001343 |
| per_sample_clue_penalty_mean | 0.000000 |

### Metrics
- **Total Accuracy:** 57.29%
- **BG Accuracy:** 56.85%
- **FG Accuracy:** 58.70%

### Per-Example Metrics
| Example | Accuracy | BG Acc | FG Acc | Exact Match | Target Classes | Pred Classes |
|---------|----------|--------|--------|-------------|----------------|---------------|
| 1 | 51.6% | 52.4% | 0.0% | ❌ | [0, 5] | [0, 2, 3, 7, 8] |
| 2 | 65.6% | 85.7% | 50.0% | ❌ | [0, 2, 3, 7, 8] | [0, 2, 3, 7, 8] |
| 3 | 54.7% | 47.3% | 100.0% | ❌ | [0, 1, 2] | [0, 1, 2, 3, 7, 8] |

### DSC Analysis
| Example | Clues Used | Stop Probs | Stop Logits |
|---------|------------|------------|-------------|
| 1 | 3.91 | [0.022, 0.021, 0.022, 0.025] | [-3.81, -3.87, -3.79, -3.68] |
| 2 | 3.91 | [0.025, 0.021, 0.023, 0.022] | [-3.64, -3.83, -3.75, -3.81] |
| 3 | 3.91 | [0.024, 0.021, 0.024, 0.022] | [-3.70, -3.85, -3.70, -3.78] |

### Attention Entropy (per clue)
| Example | Clue 0 | Clue 1 | Clue 2 | Clue 3 | Clue 4 | Clue 5 |
|---------|--------|--------|--------|--------|--------|--------|
| 1 | 0.884 | 0.764 | 0.746 | 0.350 |
| 2 | 0.298 | 0.606 | 0.559 | 0.784 |
| 3 | 0.820 | 0.745 | 0.493 | 0.699 |

### Gradient Norms (selected modules)
- **encoder:** 0.069718
- **feature_proj:** 0.122150
- **context_encoder:** 0.094359
- **context_injector:** 0.071744
- **dsc:** 0.104653
- **msre:** 0.117971
- **solver:** 0.888651

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
3 2 3 0
7 8 0 0
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
2 2 2 2 2 2
7 8 7 8 8 8
2 3 2 3 3 3
8 8 8 0 0 0
3 8 0 8 0 0
7 8 8 0 0 0
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
| total_loss | 0.639439 |
| task_loss | 0.413961 |
| focal_loss | 0.413961 |
| entropy_loss | 2.811016 |
| sparsity_loss | 0.092071 |
| predicate_loss | 0.000000 |
| curriculum_loss | 2.708134 |
| deep_supervision_loss | 0.432542 |
| act_loss | 0.000000 |
| loss_mode | weighted_stablemax |
| sparsity_min_clue_penalty | 0.000000 |
| sparsity_base_pondering | 0.039144 |
| sparsity_entropy_pondering | 0.052927 |
| expected_clues_used | 3.914386 |
| stop_prob_from_loss | 0.021404 |
| clues_used_std | 0.006072 |
| per_sample_clue_penalty_mean | 0.000000 |

### Metrics
- **Total Accuracy:** 63.02%
- **BG Accuracy:** 67.12%
- **FG Accuracy:** 50.00%

### Per-Example Metrics
| Example | Accuracy | BG Acc | FG Acc | Exact Match | Target Classes | Pred Classes |
|---------|----------|--------|--------|-------------|----------------|---------------|
| 1 | 62.5% | 63.5% | 0.0% | ❌ | [0, 5] | [0, 2, 3, 7, 8] |
| 2 | 62.5% | 92.9% | 38.9% | ❌ | [0, 2, 3, 7, 8] | [0, 2, 3, 7, 8] |
| 3 | 64.1% | 58.2% | 100.0% | ❌ | [0, 1, 2] | [0, 1, 2, 3, 7, 8] |

### DSC Analysis
| Example | Clues Used | Stop Probs | Stop Logits |
|---------|------------|------------|-------------|
| 1 | 3.91 | [0.023, 0.021, 0.023, 0.024] | [-3.76, -3.83, -3.77, -3.70] |
| 2 | 3.91 | [0.023, 0.023, 0.021, 0.021] | [-3.77, -3.74, -3.84, -3.86] |
| 3 | 3.92 | [0.019, 0.020, 0.020, 0.020] | [-3.93, -3.88, -3.91, -3.91] |

### Attention Entropy (per clue)
| Example | Clue 0 | Clue 1 | Clue 2 | Clue 3 | Clue 4 | Clue 5 |
|---------|--------|--------|--------|--------|--------|--------|
| 1 | 0.564 | 0.756 | 0.683 | 0.570 |
| 2 | 0.558 | 0.461 | 0.638 | 0.774 |
| 3 | 0.799 | 0.737 | 0.808 | 0.762 |


## Epoch 12

**Temperature:** 0.9593

### Losses
| Loss | Value |
|------|-------|
| total_loss | 0.581207 |
| task_loss | 0.375859 |
| focal_loss | 0.375859 |
| entropy_loss | 2.708932 |
| sparsity_loss | 0.090130 |
| predicate_loss | 0.000000 |
| curriculum_loss | 2.693145 |
| deep_supervision_loss | 0.392669 |
| act_loss | 0.000000 |
| loss_mode | weighted_stablemax |
| sparsity_min_clue_penalty | 0.000000 |
| sparsity_base_pondering | 0.039137 |
| sparsity_entropy_pondering | 0.050993 |
| expected_clues_used | 3.913724 |
| stop_prob_from_loss | 0.021569 |
| clues_used_std | 0.002400 |
| per_sample_clue_penalty_mean | 0.000000 |

### Metrics
- **Total Accuracy:** 65.62%
- **BG Accuracy:** 67.12%
- **FG Accuracy:** 60.87%

### Per-Example Metrics
| Example | Accuracy | BG Acc | FG Acc | Exact Match | Target Classes | Pred Classes |
|---------|----------|--------|--------|-------------|----------------|---------------|
| 1 | 59.4% | 60.3% | 0.0% | ❌ | [0, 5] | [0, 2, 3, 7, 8] |
| 2 | 70.3% | 92.9% | 52.8% | ❌ | [0, 2, 3, 7, 8] | [0, 2, 3, 7, 8] |
| 3 | 67.2% | 61.8% | 100.0% | ❌ | [0, 1, 2] | [0, 1, 2, 3, 7, 8] |

### DSC Analysis
| Example | Clues Used | Stop Probs | Stop Logits |
|---------|------------|------------|-------------|
| 1 | 3.91 | [0.021, 0.022, 0.024, 0.023] | [-3.85, -3.81, -3.73, -3.76] |
| 2 | 3.91 | [0.023, 0.021, 0.020, 0.023] | [-3.77, -3.83, -3.89, -3.76] |
| 3 | 3.92 | [0.020, 0.023, 0.020, 0.020] | [-3.88, -3.75, -3.88, -3.87] |

### Attention Entropy (per clue)
| Example | Clue 0 | Clue 1 | Clue 2 | Clue 3 | Clue 4 | Clue 5 |
|---------|--------|--------|--------|--------|--------|--------|
| 1 | 0.715 | 0.633 | 0.721 | 0.453 |
| 2 | 0.575 | 0.571 | 0.772 | 0.415 |
| 3 | 0.844 | 0.601 | 0.811 | 0.706 |


## Epoch 13

**Temperature:** 0.9559

### Losses
| Loss | Value |
|------|-------|
| total_loss | 0.545001 |
| task_loss | 0.351177 |
| focal_loss | 0.351177 |
| entropy_loss | 2.465811 |
| sparsity_loss | 0.085463 |
| predicate_loss | 0.000000 |
| curriculum_loss | 2.675647 |
| deep_supervision_loss | 0.370555 |
| act_loss | 0.000000 |
| loss_mode | weighted_stablemax |
| sparsity_min_clue_penalty | 0.000000 |
| sparsity_base_pondering | 0.039089 |
| sparsity_entropy_pondering | 0.046373 |
| expected_clues_used | 3.908930 |
| stop_prob_from_loss | 0.022768 |
| clues_used_std | 0.005621 |
| per_sample_clue_penalty_mean | 0.000000 |

### Metrics
- **Total Accuracy:** 63.02%
- **BG Accuracy:** 58.22%
- **FG Accuracy:** 78.26%

### Per-Example Metrics
| Example | Accuracy | BG Acc | FG Acc | Exact Match | Target Classes | Pred Classes |
|---------|----------|--------|--------|-------------|----------------|---------------|
| 1 | 46.9% | 47.6% | 0.0% | ❌ | [0, 5] | [0, 2, 3, 7, 8] |
| 2 | 84.4% | 96.4% | 75.0% | ❌ | [0, 2, 3, 7, 8] | [0, 2, 3, 7, 8] |
| 3 | 57.8% | 50.9% | 100.0% | ❌ | [0, 1, 2] | [0, 1, 2, 3, 7, 8] |

### DSC Analysis
| Example | Clues Used | Stop Probs | Stop Logits |
|---------|------------|------------|-------------|
| 1 | 3.91 | [0.022, 0.020, 0.022, 0.026] | [-3.80, -3.88, -3.82, -3.63] |
| 2 | 3.91 | [0.021, 0.021, 0.022, 0.023] | [-3.85, -3.84, -3.80, -3.76] |
| 3 | 3.90 | [0.022, 0.024, 0.025, 0.026] | [-3.79, -3.71, -3.65, -3.63] |

### Attention Entropy (per clue)
| Example | Clue 0 | Clue 1 | Clue 2 | Clue 3 | Clue 4 | Clue 5 |
|---------|--------|--------|--------|--------|--------|--------|
| 1 | 0.720 | 0.658 | 0.841 | 0.380 |
| 2 | 0.849 | 0.636 | 0.618 | 0.484 |
| 3 | 0.713 | 0.355 | 0.588 | 0.273 |


## Epoch 14

**Temperature:** 0.9526

### Losses
| Loss | Value |
|------|-------|
| total_loss | 0.511017 |
| task_loss | 0.328007 |
| focal_loss | 0.328007 |
| entropy_loss | 2.401278 |
| sparsity_loss | 0.084284 |
| predicate_loss | 0.000000 |
| curriculum_loss | 2.659647 |
| deep_supervision_loss | 0.349164 |
| act_loss | 0.000000 |
| loss_mode | weighted_stablemax |
| sparsity_min_clue_penalty | 0.000000 |
| sparsity_base_pondering | 0.039078 |
| sparsity_entropy_pondering | 0.045207 |
| expected_clues_used | 3.907776 |
| stop_prob_from_loss | 0.023056 |
| clues_used_std | 0.015667 |
| per_sample_clue_penalty_mean | 0.000000 |

### Metrics
- **Total Accuracy:** 65.10%
- **BG Accuracy:** 59.59%
- **FG Accuracy:** 82.61%

### Per-Example Metrics
| Example | Accuracy | BG Acc | FG Acc | Exact Match | Target Classes | Pred Classes |
|---------|----------|--------|--------|-------------|----------------|---------------|
| 1 | 45.3% | 46.0% | 0.0% | ❌ | [0, 5] | [0, 2, 3, 7, 8] |
| 2 | 89.1% | 100.0% | 80.6% | ❌ | [0, 2, 3, 7, 8] | [0, 2, 3, 7, 8] |
| 3 | 60.9% | 54.5% | 100.0% | ❌ | [0, 1, 2] | [0, 1, 2, 3, 7, 8] |

### DSC Analysis
| Example | Clues Used | Stop Probs | Stop Logits |
|---------|------------|------------|-------------|
| 1 | 3.91 | [0.021, 0.022, 0.022, 0.022] | [-3.85, -3.80, -3.81, -3.80] |
| 2 | 3.92 | [0.022, 0.020, 0.020, 0.019] | [-3.81, -3.91, -3.89, -3.95] |
| 3 | 3.89 | [0.027, 0.022, 0.021, 0.041] | [-3.60, -3.81, -3.83, -3.16] |

### Attention Entropy (per clue)
| Example | Clue 0 | Clue 1 | Clue 2 | Clue 3 | Clue 4 | Clue 5 |
|---------|--------|--------|--------|--------|--------|--------|
| 1 | 0.638 | 0.609 | 0.769 | 0.619 |
| 2 | 0.527 | 0.722 | 0.779 | 0.814 |
| 3 | 0.311 | 0.513 | 0.554 | 0.073 |


## Epoch 15

**Temperature:** 0.9493

### Losses
| Loss | Value |
|------|-------|
| total_loss | 0.489236 |
| task_loss | 0.312753 |
| focal_loss | 0.312753 |
| entropy_loss | 2.597099 |
| sparsity_loss | 0.088046 |
| predicate_loss | 0.000000 |
| curriculum_loss | 2.644786 |
| deep_supervision_loss | 0.335356 |
| act_loss | 0.000000 |
| loss_mode | weighted_stablemax |
| sparsity_min_clue_penalty | 0.000000 |
| sparsity_base_pondering | 0.039138 |
| sparsity_entropy_pondering | 0.048907 |
| expected_clues_used | 3.913840 |
| stop_prob_from_loss | 0.021540 |
| clues_used_std | 0.004448 |
| per_sample_clue_penalty_mean | 0.000000 |

### Metrics
- **Total Accuracy:** 66.67%
- **BG Accuracy:** 60.27%
- **FG Accuracy:** 86.96%

### Per-Example Metrics
| Example | Accuracy | BG Acc | FG Acc | Exact Match | Target Classes | Pred Classes |
|---------|----------|--------|--------|-------------|----------------|---------------|
| 1 | 50.0% | 50.8% | 0.0% | ❌ | [0, 5] | [0, 2, 3, 7, 8] |
| 2 | 90.6% | 96.4% | 86.1% | ❌ | [0, 2, 3, 7, 8] | [0, 2, 3, 7, 8] |
| 3 | 59.4% | 52.7% | 100.0% | ❌ | [0, 1, 2] | [0, 1, 2, 3, 7, 8] |

### DSC Analysis
| Example | Clues Used | Stop Probs | Stop Logits |
|---------|------------|------------|-------------|
| 1 | 3.91 | [0.028, 0.023, 0.020, 0.020] | [-3.56, -3.74, -3.91, -3.89] |
| 2 | 3.92 | [0.019, 0.021, 0.020, 0.021] | [-3.94, -3.83, -3.88, -3.83] |
| 3 | 3.91 | [0.020, 0.022, 0.025, 0.020] | [-3.90, -3.82, -3.67, -3.90] |

### Attention Entropy (per clue)
| Example | Clue 0 | Clue 1 | Clue 2 | Clue 3 | Clue 4 | Clue 5 |
|---------|--------|--------|--------|--------|--------|--------|
| 1 | 0.326 | 0.465 | 0.845 | 0.687 |
| 2 | 0.732 | 0.591 | 0.607 | 0.600 |
| 3 | 0.741 | 0.701 | 0.444 | 0.754 |


## Epoch 16

**Temperature:** 0.9461

### Losses
| Loss | Value |
|------|-------|
| total_loss | 0.460695 |
| task_loss | 0.294871 |
| focal_loss | 0.294871 |
| entropy_loss | 2.869983 |
| sparsity_loss | 0.093259 |
| predicate_loss | 0.000000 |
| curriculum_loss | 2.639644 |
| deep_supervision_loss | 0.312996 |
| act_loss | 0.000000 |
| loss_mode | weighted_stablemax |
| sparsity_min_clue_penalty | 0.000000 |
| sparsity_base_pondering | 0.039177 |
| sparsity_entropy_pondering | 0.054082 |
| expected_clues_used | 3.917749 |
| stop_prob_from_loss | 0.020563 |
| clues_used_std | 0.004570 |
| per_sample_clue_penalty_mean | 0.000000 |

### Metrics
- **Total Accuracy:** 73.96%
- **BG Accuracy:** 68.49%
- **FG Accuracy:** 91.30%

### Per-Example Metrics
| Example | Accuracy | BG Acc | FG Acc | Exact Match | Target Classes | Pred Classes |
|---------|----------|--------|--------|-------------|----------------|---------------|
| 1 | 64.1% | 63.5% | 100.0% | ❌ | [0, 5] | [0, 2, 3, 5, 7, 8] |
| 2 | 93.8% | 100.0% | 88.9% | ❌ | [0, 2, 3, 7, 8] | [0, 2, 3, 7, 8] |
| 3 | 64.1% | 58.2% | 100.0% | ❌ | [0, 1, 2] | [0, 1, 2, 3, 7, 8] |

### DSC Analysis
| Example | Clues Used | Stop Probs | Stop Logits |
|---------|------------|------------|-------------|
| 1 | 3.91 | [0.022, 0.019, 0.025, 0.021] | [-3.79, -3.92, -3.67, -3.83] |
| 2 | 3.92 | [0.020, 0.019, 0.020, 0.021] | [-3.91, -3.93, -3.88, -3.83] |
| 3 | 3.92 | [0.020, 0.019, 0.020, 0.020] | [-3.89, -3.94, -3.88, -3.90] |

### Attention Entropy (per clue)
| Example | Clue 0 | Clue 1 | Clue 2 | Clue 3 | Clue 4 | Clue 5 |
|---------|--------|--------|--------|--------|--------|--------|
| 1 | 0.574 | 0.733 | 0.501 | 0.683 |
| 2 | 0.687 | 0.755 | 0.785 | 0.545 |
| 3 | 0.596 | 0.868 | 0.833 | 0.722 |


## Epoch 17

**Temperature:** 0.9428

### Losses
| Loss | Value |
|------|-------|
| total_loss | 0.469486 |
| task_loss | 0.302500 |
| focal_loss | 0.302500 |
| entropy_loss | 2.345382 |
| sparsity_loss | 0.083253 |
| predicate_loss | 0.000000 |
| curriculum_loss | 2.607110 |
| deep_supervision_loss | 0.317321 |
| act_loss | 0.000000 |
| loss_mode | weighted_stablemax |
| sparsity_min_clue_penalty | 0.000000 |
| sparsity_base_pondering | 0.039096 |
| sparsity_entropy_pondering | 0.044157 |
| expected_clues_used | 3.909588 |
| stop_prob_from_loss | 0.022603 |
| clues_used_std | 0.004411 |
| per_sample_clue_penalty_mean | 0.000000 |

### Metrics
- **Total Accuracy:** 88.54%
- **BG Accuracy:** 96.58%
- **FG Accuracy:** 63.04%

### Per-Example Metrics
| Example | Accuracy | BG Acc | FG Acc | Exact Match | Target Classes | Pred Classes |
|---------|----------|--------|--------|-------------|----------------|---------------|
| 1 | 96.9% | 98.4% | 0.0% | ❌ | [0, 5] | [0, 7] |
| 2 | 75.0% | 100.0% | 55.6% | ❌ | [0, 2, 3, 7, 8] | [0, 2, 3, 7, 8] |
| 3 | 93.8% | 92.7% | 100.0% | ❌ | [0, 1, 2] | [0, 1, 2] |

### DSC Analysis
| Example | Clues Used | Stop Probs | Stop Logits |
|---------|------------|------------|-------------|
| 1 | 3.91 | [0.022, 0.024, 0.022, 0.020] | [-3.80, -3.72, -3.79, -3.89] |
| 2 | 3.91 | [0.021, 0.020, 0.022, 0.025] | [-3.85, -3.91, -3.79, -3.65] |
| 3 | 3.90 | [0.033, 0.019, 0.024, 0.020] | [-3.38, -3.93, -3.72, -3.91] |

### Attention Entropy (per clue)
| Example | Clue 0 | Clue 1 | Clue 2 | Clue 3 | Clue 4 | Clue 5 |
|---------|--------|--------|--------|--------|--------|--------|
| 1 | 0.514 | 0.433 | 0.424 | 0.734 |
| 2 | 0.810 | 0.772 | 0.576 | 0.296 |
| 3 | 0.081 | 0.834 | 0.434 | 0.860 |


## Epoch 18

**Temperature:** 0.9395

### Losses
| Loss | Value |
|------|-------|
| total_loss | 0.514578 |
| task_loss | 0.328060 |
| focal_loss | 0.328060 |
| entropy_loss | 2.351384 |
| sparsity_loss | 0.083454 |
| predicate_loss | 0.000000 |
| curriculum_loss | 2.607414 |
| deep_supervision_loss | 0.356347 |
| act_loss | 0.000000 |
| loss_mode | weighted_stablemax |
| sparsity_min_clue_penalty | 0.000000 |
| sparsity_base_pondering | 0.039143 |
| sparsity_entropy_pondering | 0.044310 |
| expected_clues_used | 3.914335 |
| stop_prob_from_loss | 0.021416 |
| clues_used_std | 0.008775 |
| per_sample_clue_penalty_mean | 0.000000 |

### Metrics
- **Total Accuracy:** 59.38%
- **BG Accuracy:** 49.32%
- **FG Accuracy:** 91.30%

### Per-Example Metrics
| Example | Accuracy | BG Acc | FG Acc | Exact Match | Target Classes | Pred Classes |
|---------|----------|--------|--------|-------------|----------------|---------------|
| 1 | 37.5% | 36.5% | 100.0% | ❌ | [0, 5] | [0, 2, 3, 5, 7, 8] |
| 2 | 85.9% | 82.1% | 88.9% | ❌ | [0, 2, 3, 7, 8] | [0, 2, 3, 7, 8] |
| 3 | 54.7% | 47.3% | 100.0% | ❌ | [0, 1, 2] | [0, 1, 2, 3, 7, 8] |

### DSC Analysis
| Example | Clues Used | Stop Probs | Stop Logits |
|---------|------------|------------|-------------|
| 1 | 3.92 | [0.019, 0.020, 0.020, 0.020] | [-3.94, -3.89, -3.89, -3.90] |
| 2 | 3.90 | [0.019, 0.025, 0.022, 0.030] | [-3.93, -3.67, -3.80, -3.48] |
| 3 | 3.92 | [0.021, 0.022, 0.020, 0.019] | [-3.86, -3.77, -3.91, -3.92] |

### Attention Entropy (per clue)
| Example | Clue 0 | Clue 1 | Clue 2 | Clue 3 | Clue 4 | Clue 5 |
|---------|--------|--------|--------|--------|--------|--------|
| 1 | 0.727 | 0.505 | 0.570 | 0.599 |
| 2 | 0.764 | 0.273 | 0.665 | 0.061 |
| 3 | 0.649 | 0.316 | 0.829 | 0.827 |


## Epoch 19

**Temperature:** 0.9363

### Losses
| Loss | Value |
|------|-------|
| total_loss | 0.413595 |
| task_loss | 0.265309 |
| focal_loss | 0.265309 |
| entropy_loss | 2.447896 |
| sparsity_loss | 0.085099 |
| predicate_loss | 0.000000 |
| curriculum_loss | 2.586864 |
| deep_supervision_loss | 0.279552 |
| act_loss | 0.000000 |
| loss_mode | weighted_stablemax |
| sparsity_min_clue_penalty | 0.000000 |
| sparsity_base_pondering | 0.039049 |
| sparsity_entropy_pondering | 0.046050 |
| expected_clues_used | 3.904860 |
| stop_prob_from_loss | 0.023785 |
| clues_used_std | 0.015127 |
| per_sample_clue_penalty_mean | 0.000000 |

### Metrics
- **Total Accuracy:** 87.50%
- **BG Accuracy:** 85.62%
- **FG Accuracy:** 93.48%

### Per-Example Metrics
| Example | Accuracy | BG Acc | FG Acc | Exact Match | Target Classes | Pred Classes |
|---------|----------|--------|--------|-------------|----------------|---------------|
| 1 | 79.7% | 81.0% | 0.0% | ❌ | [0, 5] | [0, 2, 3, 7] |
| 2 | 96.9% | 100.0% | 94.4% | ❌ | [0, 2, 3, 7, 8] | [0, 2, 3, 7, 8] |
| 3 | 85.9% | 83.6% | 100.0% | ❌ | [0, 1, 2] | [0, 1, 2, 7, 8] |

### DSC Analysis
| Example | Clues Used | Stop Probs | Stop Logits |
|---------|------------|------------|-------------|
| 1 | 3.91 | [0.022, 0.025, 0.021, 0.026] | [-3.78, -3.66, -3.86, -3.61] |
| 2 | 3.92 | [0.021, 0.020, 0.020, 0.020] | [-3.84, -3.91, -3.91, -3.89] |
| 3 | 3.89 | [0.021, 0.022, 0.021, 0.046] | [-3.82, -3.78, -3.83, -3.04] |

### Attention Entropy (per clue)
| Example | Clue 0 | Clue 1 | Clue 2 | Clue 3 | Clue 4 | Clue 5 |
|---------|--------|--------|--------|--------|--------|--------|
| 1 | 0.415 | 0.642 | 0.722 | 0.322 |
| 2 | 0.437 | 0.744 | 0.720 | 0.698 |
| 3 | 0.727 | 0.745 | 0.737 | 0.154 |


## Epoch 20

**Temperature:** 0.9330

### Losses
| Loss | Value |
|------|-------|
| total_loss | 0.421068 |
| task_loss | 0.268576 |
| focal_loss | 0.268576 |
| entropy_loss | 2.364446 |
| sparsity_loss | 0.083561 |
| predicate_loss | 0.000000 |
| curriculum_loss | 2.561452 |
| deep_supervision_loss | 0.288271 |
| act_loss | 0.000000 |
| loss_mode | weighted_stablemax |
| sparsity_min_clue_penalty | 0.000000 |
| sparsity_base_pondering | 0.039069 |
| sparsity_entropy_pondering | 0.044492 |
| expected_clues_used | 3.906934 |
| stop_prob_from_loss | 0.023267 |
| clues_used_std | 0.004645 |
| per_sample_clue_penalty_mean | 0.000000 |

### Metrics
- **Total Accuracy:** 91.15%
- **BG Accuracy:** 95.89%
- **FG Accuracy:** 76.09%

### Per-Example Metrics
| Example | Accuracy | BG Acc | FG Acc | Exact Match | Target Classes | Pred Classes |
|---------|----------|--------|--------|-------------|----------------|---------------|
| 1 | 93.8% | 95.2% | 0.0% | ❌ | [0, 5] | [0, 2, 7] |
| 2 | 84.4% | 100.0% | 72.2% | ❌ | [0, 2, 3, 7, 8] | [0, 2, 3, 7, 8] |
| 3 | 95.3% | 94.5% | 100.0% | ❌ | [0, 1, 2] | [0, 1, 2, 7] |

### DSC Analysis
| Example | Clues Used | Stop Probs | Stop Logits |
|---------|------------|------------|-------------|
| 1 | 3.90 | [0.024, 0.033, 0.021, 0.019] | [-3.73, -3.38, -3.86, -3.92] |
| 2 | 3.91 | [0.023, 0.019, 0.033, 0.020] | [-3.75, -3.94, -3.39, -3.88] |
| 3 | 3.91 | [0.020, 0.022, 0.021, 0.025] | [-3.88, -3.81, -3.84, -3.68] |

### Attention Entropy (per clue)
| Example | Clue 0 | Clue 1 | Clue 2 | Clue 3 | Clue 4 | Clue 5 |
|---------|--------|--------|--------|--------|--------|--------|
| 1 | 0.467 | 0.384 | 0.810 | 0.772 |
| 2 | 0.304 | 0.822 | 0.018 | 0.484 |
| 3 | 0.707 | 0.755 | 0.753 | 0.545 |

### Gradient Norms (selected modules)
- **encoder:** 0.067023
- **feature_proj:** 0.116496
- **context_encoder:** 0.172867
- **context_injector:** 0.082767
- **dsc:** 0.063336
- **msre:** 0.135732
- **solver:** 1.214691

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
3 2 3 2 0 2
7 8 7 8 7 0
2 3 2 3 2 0
8 7 8 7 0 0
3 2 3 2 0 0
7 0 7 0 7 0
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
| total_loss | 0.387985 |
| task_loss | 0.243659 |
| focal_loss | 0.243659 |
| entropy_loss | 2.225088 |
| sparsity_loss | 0.080733 |
| predicate_loss | 0.000000 |
| curriculum_loss | 2.521465 |
| deep_supervision_loss | 0.272504 |
| act_loss | 0.000000 |
| loss_mode | weighted_stablemax |
| sparsity_min_clue_penalty | 0.000000 |
| sparsity_base_pondering | 0.038933 |
| sparsity_entropy_pondering | 0.041800 |
| expected_clues_used | 3.893319 |
| stop_prob_from_loss | 0.026670 |
| clues_used_std | 0.026808 |
| per_sample_clue_penalty_mean | 0.000000 |

### Metrics
- **Total Accuracy:** 93.23%
- **BG Accuracy:** 93.84%
- **FG Accuracy:** 91.30%

### Per-Example Metrics
| Example | Accuracy | BG Acc | FG Acc | Exact Match | Target Classes | Pred Classes |
|---------|----------|--------|--------|-------------|----------------|---------------|
| 1 | 90.6% | 92.1% | 0.0% | ❌ | [0, 5] | [0, 2, 3, 7] |
| 2 | 95.3% | 100.0% | 91.7% | ❌ | [0, 2, 3, 7, 8] | [0, 2, 3, 7, 8] |
| 3 | 93.8% | 92.7% | 100.0% | ❌ | [0, 1, 2] | [0, 1, 2, 7, 8] |

### DSC Analysis
| Example | Clues Used | Stop Probs | Stop Logits |
|---------|------------|------------|-------------|
| 1 | 3.86 | [0.041, 0.022, 0.053, 0.022] | [-3.16, -3.79, -2.89, -3.79] |
| 2 | 3.91 | [0.028, 0.019, 0.019, 0.020] | [-3.54, -3.93, -3.93, -3.89] |
| 3 | 3.90 | [0.020, 0.026, 0.026, 0.024] | [-3.87, -3.63, -3.61, -3.72] |

### Attention Entropy (per clue)
| Example | Clue 0 | Clue 1 | Clue 2 | Clue 3 | Clue 4 | Clue 5 |
|---------|--------|--------|--------|--------|--------|--------|
| 1 | 0.263 | 0.833 | 0.123 | 0.691 |
| 2 | 0.090 | 0.663 | 0.688 | 0.547 |
| 3 | 0.768 | 0.725 | 0.508 | 0.521 |


## Epoch 22

**Temperature:** 0.9266

### Losses
| Loss | Value |
|------|-------|
| total_loss | 0.356470 |
| task_loss | 0.223463 |
| focal_loss | 0.223463 |
| entropy_loss | 2.183800 |
| sparsity_loss | 0.079811 |
| predicate_loss | 0.000000 |
| curriculum_loss | 2.517867 |
| deep_supervision_loss | 0.250051 |
| act_loss | 0.000000 |
| loss_mode | weighted_stablemax |
| sparsity_min_clue_penalty | 0.000000 |
| sparsity_base_pondering | 0.038910 |
| sparsity_entropy_pondering | 0.040902 |
| expected_clues_used | 3.890963 |
| stop_prob_from_loss | 0.027259 |
| clues_used_std | 0.015789 |
| per_sample_clue_penalty_mean | 0.000000 |

### Metrics
- **Total Accuracy:** 83.85%
- **BG Accuracy:** 78.77%
- **FG Accuracy:** 100.00%

### Per-Example Metrics
| Example | Accuracy | BG Acc | FG Acc | Exact Match | Target Classes | Pred Classes |
|---------|----------|--------|--------|-------------|----------------|---------------|
| 1 | 70.3% | 69.8% | 100.0% | ❌ | [0, 5] | [0, 2, 3, 5, 7, 8] |
| 2 | 100.0% | 100.0% | 100.0% | ✅ | [0, 2, 3, 7, 8] | [0, 2, 3, 7, 8] |
| 3 | 81.2% | 78.2% | 100.0% | ❌ | [0, 1, 2] | [0, 1, 2, 7, 8] |

### DSC Analysis
| Example | Clues Used | Stop Probs | Stop Logits |
|---------|------------|------------|-------------|
| 1 | 3.89 | [0.027, 0.027, 0.026, 0.025] | [-3.59, -3.58, -3.63, -3.66] |
| 2 | 3.90 | [0.020, 0.024, 0.021, 0.030] | [-3.91, -3.70, -3.82, -3.46] |
| 3 | 3.87 | [0.032, 0.026, 0.026, 0.042] | [-3.41, -3.60, -3.62, -3.13] |

### Attention Entropy (per clue)
| Example | Clue 0 | Clue 1 | Clue 2 | Clue 3 | Clue 4 | Clue 5 |
|---------|--------|--------|--------|--------|--------|--------|
| 1 | 0.634 | 0.629 | 0.734 | 0.787 |
| 2 | 0.603 | 0.440 | 0.412 | 0.043 |
| 3 | 0.371 | 0.592 | 0.723 | 0.334 |


## Epoch 23

**Temperature:** 0.9234

### Losses
| Loss | Value |
|------|-------|
| total_loss | 0.341399 |
| task_loss | 0.209977 |
| focal_loss | 0.209977 |
| entropy_loss | 2.623935 |
| sparsity_loss | 0.087144 |
| predicate_loss | 0.000000 |
| curriculum_loss | 2.418719 |
| deep_supervision_loss | 0.245415 |
| act_loss | 0.000000 |
| loss_mode | weighted_stablemax |
| sparsity_min_clue_penalty | 0.000000 |
| sparsity_base_pondering | 0.038478 |
| sparsity_entropy_pondering | 0.048666 |
| expected_clues_used | 3.847797 |
| stop_prob_from_loss | 0.038051 |
| clues_used_std | 0.087011 |
| per_sample_clue_penalty_mean | 0.000000 |

### Metrics
- **Total Accuracy:** 77.08%
- **BG Accuracy:** 69.86%
- **FG Accuracy:** 100.00%

### Per-Example Metrics
| Example | Accuracy | BG Acc | FG Acc | Exact Match | Target Classes | Pred Classes |
|---------|----------|--------|--------|-------------|----------------|---------------|
| 1 | 60.9% | 60.3% | 100.0% | ❌ | [0, 5] | [0, 2, 3, 5, 7, 8] |
| 2 | 100.0% | 100.0% | 100.0% | ✅ | [0, 2, 3, 7, 8] | [0, 2, 3, 7, 8] |
| 3 | 70.3% | 65.5% | 100.0% | ❌ | [0, 1, 2] | [0, 1, 2, 3, 7, 8] |

### DSC Analysis
| Example | Clues Used | Stop Probs | Stop Logits |
|---------|------------|------------|-------------|
| 1 | 3.75 | [0.046, 0.100, 0.057, 0.046] | [-3.03, -2.20, -2.81, -3.02] |
| 2 | 3.92 | [0.019, 0.020, 0.020, 0.022] | [-3.95, -3.90, -3.89, -3.80] |
| 3 | 3.87 | [0.044, 0.032, 0.025, 0.026] | [-3.07, -3.42, -3.65, -3.63] |

### Attention Entropy (per clue)
| Example | Clue 0 | Clue 1 | Clue 2 | Clue 3 | Clue 4 | Clue 5 |
|---------|--------|--------|--------|--------|--------|--------|
| 1 | 0.697 | 0.458 | 0.502 | 0.591 |
| 2 | 0.745 | 0.652 | 0.737 | 0.494 |
| 3 | 0.647 | 0.647 | 0.649 | 0.753 |


## Epoch 24

**Temperature:** 0.9202

### Losses
| Loss | Value |
|------|-------|
| total_loss | 0.294645 |
| task_loss | 0.179861 |
| focal_loss | 0.179861 |
| entropy_loss | 2.137622 |
| sparsity_loss | 0.079217 |
| predicate_loss | 0.000000 |
| curriculum_loss | 2.498143 |
| deep_supervision_loss | 0.213723 |
| act_loss | 0.000000 |
| loss_mode | weighted_stablemax |
| sparsity_min_clue_penalty | 0.000000 |
| sparsity_base_pondering | 0.039062 |
| sparsity_entropy_pondering | 0.040155 |
| expected_clues_used | 3.906165 |
| stop_prob_from_loss | 0.023459 |
| clues_used_std | 0.008807 |
| per_sample_clue_penalty_mean | 0.000000 |

### Metrics
- **Total Accuracy:** 98.44%
- **BG Accuracy:** 97.95%
- **FG Accuracy:** 100.00%

### Per-Example Metrics
| Example | Accuracy | BG Acc | FG Acc | Exact Match | Target Classes | Pred Classes |
|---------|----------|--------|--------|-------------|----------------|---------------|
| 1 | 100.0% | 100.0% | 100.0% | ✅ | [0, 5] | [0, 5] |
| 2 | 100.0% | 100.0% | 100.0% | ✅ | [0, 2, 3, 7, 8] | [0, 2, 3, 7, 8] |
| 3 | 95.3% | 94.5% | 100.0% | ❌ | [0, 1, 2] | [0, 1, 2, 7, 8] |

### DSC Analysis
| Example | Clues Used | Stop Probs | Stop Logits |
|---------|------------|------------|-------------|
| 1 | 3.90 | [0.027, 0.023, 0.028, 0.022] | [-3.59, -3.73, -3.54, -3.78] |
| 2 | 3.92 | [0.022, 0.022, 0.019, 0.020] | [-3.78, -3.78, -3.94, -3.88] |
| 3 | 3.90 | [0.024, 0.024, 0.027, 0.022] | [-3.72, -3.70, -3.58, -3.81] |

### Attention Entropy (per clue)
| Example | Clue 0 | Clue 1 | Clue 2 | Clue 3 | Clue 4 | Clue 5 |
|---------|--------|--------|--------|--------|--------|--------|
| 1 | 0.667 | 0.731 | 0.470 | 0.686 |
| 2 | 0.315 | 0.215 | 0.564 | 0.412 |
| 3 | 0.569 | 0.549 | 0.343 | 0.646 |


## Epoch 25

**Temperature:** 0.9170

### Losses
| Loss | Value |
|------|-------|
| total_loss | 0.299102 |
| task_loss | 0.181450 |
| focal_loss | 0.181450 |
| entropy_loss | 2.523299 |
| sparsity_loss | 0.086670 |
| predicate_loss | 0.000000 |
| curriculum_loss | 2.498651 |
| deep_supervision_loss | 0.217971 |
| act_loss | 0.000000 |
| loss_mode | weighted_stablemax |
| sparsity_min_clue_penalty | 0.000000 |
| sparsity_base_pondering | 0.039152 |
| sparsity_entropy_pondering | 0.047518 |
| expected_clues_used | 3.915237 |
| stop_prob_from_loss | 0.021191 |
| clues_used_std | 0.003942 |
| per_sample_clue_penalty_mean | 0.000000 |

### Metrics
- **Total Accuracy:** 97.92%
- **BG Accuracy:** 98.63%
- **FG Accuracy:** 95.65%

### Per-Example Metrics
| Example | Accuracy | BG Acc | FG Acc | Exact Match | Target Classes | Pred Classes |
|---------|----------|--------|--------|-------------|----------------|---------------|
| 1 | 98.4% | 100.0% | 0.0% | ❌ | [0, 5] | [0] |
| 2 | 98.4% | 100.0% | 97.2% | ❌ | [0, 2, 3, 7, 8] | [0, 2, 3, 7, 8] |
| 3 | 96.9% | 96.4% | 100.0% | ❌ | [0, 1, 2] | [0, 1, 2] |

### DSC Analysis
| Example | Clues Used | Stop Probs | Stop Logits |
|---------|------------|------------|-------------|
| 1 | 3.91 | [0.022, 0.022, 0.024, 0.019] | [-3.81, -3.80, -3.69, -3.93] |
| 2 | 3.92 | [0.019, 0.021, 0.021, 0.019] | [-3.92, -3.83, -3.86, -3.95] |
| 3 | 3.91 | [0.022, 0.022, 0.020, 0.023] | [-3.79, -3.81, -3.87, -3.76] |

### Attention Entropy (per clue)
| Example | Clue 0 | Clue 1 | Clue 2 | Clue 3 | Clue 4 | Clue 5 |
|---------|--------|--------|--------|--------|--------|--------|
| 1 | 0.548 | 0.625 | 0.517 | 0.794 |
| 2 | 0.613 | 0.371 | 0.578 | 0.721 |
| 3 | 0.725 | 0.546 | 0.795 | 0.448 |


## Epoch 26

**Temperature:** 0.9138

### Losses
| Loss | Value |
|------|-------|
| total_loss | 0.257790 |
| task_loss | 0.153656 |
| focal_loss | 0.153656 |
| entropy_loss | 2.608932 |
| sparsity_loss | 0.088351 |
| predicate_loss | 0.000000 |
| curriculum_loss | 2.488097 |
| deep_supervision_loss | 0.190597 |
| act_loss | 0.000000 |
| loss_mode | weighted_stablemax |
| sparsity_min_clue_penalty | 0.000000 |
| sparsity_base_pondering | 0.039185 |
| sparsity_entropy_pondering | 0.049166 |
| expected_clues_used | 3.918514 |
| stop_prob_from_loss | 0.020372 |
| clues_used_std | 0.002927 |
| per_sample_clue_penalty_mean | 0.000000 |

### Metrics
- **Total Accuracy:** 98.44%
- **BG Accuracy:** 98.63%
- **FG Accuracy:** 97.83%

### Per-Example Metrics
| Example | Accuracy | BG Acc | FG Acc | Exact Match | Target Classes | Pred Classes |
|---------|----------|--------|--------|-------------|----------------|---------------|
| 1 | 98.4% | 100.0% | 0.0% | ❌ | [0, 5] | [0] |
| 2 | 96.9% | 92.9% | 100.0% | ❌ | [0, 2, 3, 7, 8] | [0, 2, 3, 7, 8] |
| 3 | 100.0% | 100.0% | 100.0% | ✅ | [0, 1, 2] | [0, 1, 2] |

### DSC Analysis
| Example | Clues Used | Stop Probs | Stop Logits |
|---------|------------|------------|-------------|
| 1 | 3.92 | [0.020, 0.023, 0.019, 0.019] | [-3.88, -3.76, -3.93, -3.93] |
| 2 | 3.92 | [0.020, 0.019, 0.020, 0.020] | [-3.90, -3.93, -3.89, -3.92] |
| 3 | 3.92 | [0.022, 0.020, 0.021, 0.021] | [-3.80, -3.87, -3.85, -3.83] |

### Attention Entropy (per clue)
| Example | Clue 0 | Clue 1 | Clue 2 | Clue 3 | Clue 4 | Clue 5 |
|---------|--------|--------|--------|--------|--------|--------|
| 1 | 0.619 | 0.542 | 0.766 | 0.742 |
| 2 | 0.600 | 0.637 | 0.485 | 0.537 |
| 3 | 0.483 | 0.680 | 0.693 | 0.744 |


## Epoch 27

**Temperature:** 0.9107

### Losses
| Loss | Value |
|------|-------|
| total_loss | 0.244419 |
| task_loss | 0.149042 |
| focal_loss | 0.149042 |
| entropy_loss | 2.345199 |
| sparsity_loss | 0.083225 |
| predicate_loss | 0.000000 |
| curriculum_loss | 2.457200 |
| deep_supervision_loss | 0.174108 |
| act_loss | 0.000000 |
| loss_mode | weighted_stablemax |
| sparsity_min_clue_penalty | 0.000000 |
| sparsity_base_pondering | 0.039089 |
| sparsity_entropy_pondering | 0.044136 |
| expected_clues_used | 3.908913 |
| stop_prob_from_loss | 0.022772 |
| clues_used_std | 0.016800 |
| per_sample_clue_penalty_mean | 0.000000 |

### Metrics
- **Total Accuracy:** 94.79%
- **BG Accuracy:** 93.84%
- **FG Accuracy:** 97.83%

### Per-Example Metrics
| Example | Accuracy | BG Acc | FG Acc | Exact Match | Target Classes | Pred Classes |
|---------|----------|--------|--------|-------------|----------------|---------------|
| 1 | 98.4% | 100.0% | 0.0% | ❌ | [0, 5] | [0] |
| 2 | 85.9% | 67.9% | 100.0% | ❌ | [0, 2, 3, 7, 8] | [0, 2, 3, 7, 8] |
| 3 | 100.0% | 100.0% | 100.0% | ✅ | [0, 1, 2] | [0, 1, 2] |

### DSC Analysis
| Example | Clues Used | Stop Probs | Stop Logits |
|---------|------------|------------|-------------|
| 1 | 3.89 | [0.024, 0.036, 0.029, 0.022] | [-3.71, -3.30, -3.52, -3.79] |
| 2 | 3.92 | [0.023, 0.019, 0.020, 0.021] | [-3.74, -3.94, -3.88, -3.87] |
| 3 | 3.92 | [0.020, 0.020, 0.020, 0.020] | [-3.87, -3.88, -3.91, -3.90] |

### Attention Entropy (per clue)
| Example | Clue 0 | Clue 1 | Clue 2 | Clue 3 | Clue 4 | Clue 5 |
|---------|--------|--------|--------|--------|--------|--------|
| 1 | 0.436 | 0.253 | 0.445 | 0.528 |
| 2 | 0.251 | 0.688 | 0.684 | 0.505 |
| 3 | 0.596 | 0.798 | 0.818 | 0.765 |


## Epoch 28

**Temperature:** 0.9075

### Losses
| Loss | Value |
|------|-------|
| total_loss | 0.236276 |
| task_loss | 0.142714 |
| focal_loss | 0.142714 |
| entropy_loss | 2.512331 |
| sparsity_loss | 0.086386 |
| predicate_loss | 0.000000 |
| curriculum_loss | 2.450088 |
| deep_supervision_loss | 0.169846 |
| act_loss | 0.000000 |
| loss_mode | weighted_stablemax |
| sparsity_min_clue_penalty | 0.000000 |
| sparsity_base_pondering | 0.039090 |
| sparsity_entropy_pondering | 0.047295 |
| expected_clues_used | 3.909026 |
| stop_prob_from_loss | 0.022743 |
| clues_used_std | 0.013734 |
| per_sample_clue_penalty_mean | 0.000000 |

### Metrics
- **Total Accuracy:** 93.23%
- **BG Accuracy:** 91.78%
- **FG Accuracy:** 97.83%

### Per-Example Metrics
| Example | Accuracy | BG Acc | FG Acc | Exact Match | Target Classes | Pred Classes |
|---------|----------|--------|--------|-------------|----------------|---------------|
| 1 | 98.4% | 100.0% | 0.0% | ❌ | [0, 5] | [0] |
| 2 | 81.2% | 57.1% | 100.0% | ❌ | [0, 2, 3, 7, 8] | [0, 2, 3, 7, 8] |
| 3 | 100.0% | 100.0% | 100.0% | ✅ | [0, 1, 2] | [0, 1, 2] |

### DSC Analysis
| Example | Clues Used | Stop Probs | Stop Logits |
|---------|------------|------------|-------------|
| 1 | 3.91 | [0.022, 0.023, 0.020, 0.025] | [-3.80, -3.77, -3.88, -3.65] |
| 2 | 3.92 | [0.019, 0.020, 0.019, 0.019] | [-3.92, -3.89, -3.93, -3.94] |
| 3 | 3.89 | [0.024, 0.021, 0.035, 0.026] | [-3.72, -3.86, -3.32, -3.63] |

### Attention Entropy (per clue)
| Example | Clue 0 | Clue 1 | Clue 2 | Clue 3 | Clue 4 | Clue 5 |
|---------|--------|--------|--------|--------|--------|--------|
| 1 | 0.743 | 0.740 | 0.804 | 0.499 |
| 2 | 0.679 | 0.577 | 0.858 | 0.704 |
| 3 | 0.437 | 0.762 | 0.068 | 0.377 |


## Epoch 29

**Temperature:** 0.9044

### Losses
| Loss | Value |
|------|-------|
| total_loss | 0.242253 |
| task_loss | 0.144529 |
| focal_loss | 0.144529 |
| entropy_loss | 2.833924 |
| sparsity_loss | 0.092547 |
| predicate_loss | 0.000000 |
| curriculum_loss | 2.442615 |
| deep_supervision_loss | 0.176940 |
| act_loss | 0.000000 |
| loss_mode | weighted_stablemax |
| sparsity_min_clue_penalty | 0.000000 |
| sparsity_base_pondering | 0.039169 |
| sparsity_entropy_pondering | 0.053378 |
| expected_clues_used | 3.916909 |
| stop_prob_from_loss | 0.020773 |
| clues_used_std | 0.004713 |
| per_sample_clue_penalty_mean | 0.000000 |

### Metrics
- **Total Accuracy:** 93.23%
- **BG Accuracy:** 91.78%
- **FG Accuracy:** 97.83%

### Per-Example Metrics
| Example | Accuracy | BG Acc | FG Acc | Exact Match | Target Classes | Pred Classes |
|---------|----------|--------|--------|-------------|----------------|---------------|
| 1 | 98.4% | 100.0% | 0.0% | ❌ | [0, 5] | [0] |
| 2 | 81.2% | 57.1% | 100.0% | ❌ | [0, 2, 3, 7, 8] | [0, 2, 3, 7, 8] |
| 3 | 100.0% | 100.0% | 100.0% | ✅ | [0, 1, 2] | [0, 1, 2] |

### DSC Analysis
| Example | Clues Used | Stop Probs | Stop Logits |
|---------|------------|------------|-------------|
| 1 | 3.92 | [0.021, 0.021, 0.020, 0.021] | [-3.85, -3.86, -3.88, -3.82] |
| 2 | 3.92 | [0.020, 0.020, 0.020, 0.020] | [-3.91, -3.91, -3.91, -3.92] |
| 3 | 3.91 | [0.021, 0.022, 0.023, 0.022] | [-3.82, -3.81, -3.76, -3.81] |

### Attention Entropy (per clue)
| Example | Clue 0 | Clue 1 | Clue 2 | Clue 3 | Clue 4 | Clue 5 |
|---------|--------|--------|--------|--------|--------|--------|
| 1 | 0.705 | 0.756 | 0.654 | 0.775 |
| 2 | 0.635 | 0.571 | 0.530 | 0.741 |
| 3 | 0.678 | 0.676 | 0.733 | 0.725 |


## Epoch 30

**Temperature:** 0.9013

### Losses
| Loss | Value |
|------|-------|
| total_loss | 0.239158 |
| task_loss | 0.141941 |
| focal_loss | 0.141941 |
| entropy_loss | 2.678160 |
| sparsity_loss | 0.089408 |
| predicate_loss | 0.000000 |
| curriculum_loss | 2.406436 |
| deep_supervision_loss | 0.176554 |
| act_loss | 0.000000 |
| loss_mode | weighted_stablemax |
| sparsity_min_clue_penalty | 0.000000 |
| sparsity_base_pondering | 0.039062 |
| sparsity_entropy_pondering | 0.050346 |
| expected_clues_used | 3.906248 |
| stop_prob_from_loss | 0.023438 |
| clues_used_std | 0.011235 |
| per_sample_clue_penalty_mean | 0.000000 |

### Metrics
- **Total Accuracy:** 97.40%
- **BG Accuracy:** 97.26%
- **FG Accuracy:** 97.83%

### Per-Example Metrics
| Example | Accuracy | BG Acc | FG Acc | Exact Match | Target Classes | Pred Classes |
|---------|----------|--------|--------|-------------|----------------|---------------|
| 1 | 98.4% | 100.0% | 0.0% | ❌ | [0, 5] | [0] |
| 2 | 93.8% | 85.7% | 100.0% | ❌ | [0, 2, 3, 7, 8] | [0, 2, 3, 7, 8] |
| 3 | 100.0% | 100.0% | 100.0% | ✅ | [0, 1, 2] | [0, 1, 2] |

### DSC Analysis
| Example | Clues Used | Stop Probs | Stop Logits |
|---------|------------|------------|-------------|
| 1 | 3.89 | [0.033, 0.024, 0.022, 0.026] | [-3.38, -3.71, -3.79, -3.60] |
| 2 | 3.91 | [0.020, 0.032, 0.020, 0.021] | [-3.87, -3.41, -3.89, -3.87] |
| 3 | 3.92 | [0.021, 0.020, 0.021, 0.021] | [-3.86, -3.88, -3.85, -3.84] |

### Attention Entropy (per clue)
| Example | Clue 0 | Clue 1 | Clue 2 | Clue 3 | Clue 4 | Clue 5 |
|---------|--------|--------|--------|--------|--------|--------|
| 1 | 0.569 | 0.768 | 0.746 | 0.724 |
| 2 | 0.484 | 0.162 | 0.771 | 0.531 |
| 3 | 0.776 | 0.705 | 0.817 | 0.675 |

### Gradient Norms (selected modules)
- **encoder:** 0.026661
- **feature_proj:** 0.046729
- **context_encoder:** 0.026564
- **context_injector:** 0.029293
- **dsc:** 0.040160
- **msre:** 0.066735
- **solver:** 0.817953

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
| total_loss | 0.220504 |
| task_loss | 0.134422 |
| focal_loss | 0.134422 |
| entropy_loss | 2.551668 |
| sparsity_loss | 0.086916 |
| predicate_loss | 0.000000 |
| curriculum_loss | 2.388570 |
| deep_supervision_loss | 0.154780 |
| act_loss | 0.000000 |
| loss_mode | weighted_stablemax |
| sparsity_min_clue_penalty | 0.000000 |
| sparsity_base_pondering | 0.039025 |
| sparsity_entropy_pondering | 0.047892 |
| expected_clues_used | 3.902458 |
| stop_prob_from_loss | 0.024385 |
| clues_used_std | 0.014064 |
| per_sample_clue_penalty_mean | 0.000000 |

### Metrics
- **Total Accuracy:** 98.96%
- **BG Accuracy:** 99.32%
- **FG Accuracy:** 97.83%

### Per-Example Metrics
| Example | Accuracy | BG Acc | FG Acc | Exact Match | Target Classes | Pred Classes |
|---------|----------|--------|--------|-------------|----------------|---------------|
| 1 | 98.4% | 100.0% | 0.0% | ❌ | [0, 5] | [0] |
| 2 | 98.4% | 96.4% | 100.0% | ❌ | [0, 2, 3, 7, 8] | [0, 2, 3, 7, 8] |
| 3 | 100.0% | 100.0% | 100.0% | ✅ | [0, 1, 2] | [0, 1, 2] |

### DSC Analysis
| Example | Clues Used | Stop Probs | Stop Logits |
|---------|------------|------------|-------------|
| 1 | 3.89 | [0.026, 0.034, 0.022, 0.030] | [-3.61, -3.36, -3.81, -3.49] |
| 2 | 3.92 | [0.021, 0.020, 0.021, 0.021] | [-3.84, -3.91, -3.84, -3.82] |
| 3 | 3.90 | [0.027, 0.024, 0.024, 0.022] | [-3.57, -3.69, -3.69, -3.79] |

### Attention Entropy (per clue)
| Example | Clue 0 | Clue 1 | Clue 2 | Clue 3 | Clue 4 | Clue 5 |
|---------|--------|--------|--------|--------|--------|--------|
| 1 | 0.578 | 0.595 | 0.877 | 0.466 |
| 2 | 0.460 | 0.640 | 0.561 | 0.546 |
| 3 | 0.718 | 0.652 | 0.642 | 0.630 |


## Epoch 32

**Temperature:** 0.8950

### Losses
| Loss | Value |
|------|-------|
| total_loss | 0.212689 |
| task_loss | 0.125319 |
| focal_loss | 0.125319 |
| entropy_loss | 2.145726 |
| sparsity_loss | 0.079055 |
| predicate_loss | 0.000000 |
| curriculum_loss | 2.348592 |
| deep_supervision_loss | 0.158928 |
| act_loss | 0.000000 |
| loss_mode | weighted_stablemax |
| sparsity_min_clue_penalty | 0.000000 |
| sparsity_base_pondering | 0.038868 |
| sparsity_entropy_pondering | 0.040186 |
| expected_clues_used | 3.886849 |
| stop_prob_from_loss | 0.028288 |
| clues_used_std | 0.007335 |
| per_sample_clue_penalty_mean | 0.000000 |

### Metrics
- **Total Accuracy:** 96.35%
- **BG Accuracy:** 95.89%
- **FG Accuracy:** 97.83%

### Per-Example Metrics
| Example | Accuracy | BG Acc | FG Acc | Exact Match | Target Classes | Pred Classes |
|---------|----------|--------|--------|-------------|----------------|---------------|
| 1 | 98.4% | 100.0% | 0.0% | ❌ | [0, 5] | [0] |
| 2 | 92.2% | 82.1% | 100.0% | ❌ | [0, 2, 3, 7, 8] | [0, 2, 3, 7, 8] |
| 3 | 98.4% | 98.2% | 100.0% | ❌ | [0, 1, 2] | [0, 1, 2] |

### DSC Analysis
| Example | Clues Used | Stop Probs | Stop Logits |
|---------|------------|------------|-------------|
| 1 | 3.88 | [0.046, 0.026, 0.027, 0.023] | [-3.03, -3.64, -3.60, -3.76] |
| 2 | 3.89 | [0.026, 0.023, 0.021, 0.037] | [-3.63, -3.73, -3.85, -3.25] |
| 3 | 3.89 | [0.027, 0.027, 0.027, 0.030] | [-3.58, -3.59, -3.59, -3.48] |

### Attention Entropy (per clue)
| Example | Clue 0 | Clue 1 | Clue 2 | Clue 3 | Clue 4 | Clue 5 |
|---------|--------|--------|--------|--------|--------|--------|
| 1 | 0.188 | 0.555 | 0.639 | 0.723 |
| 2 | 0.221 | 0.616 | 0.720 | 0.226 |
| 3 | 0.497 | 0.663 | 0.693 | 0.451 |


## Epoch 33

**Temperature:** 0.8919

### Losses
| Loss | Value |
|------|-------|
| total_loss | 0.212705 |
| task_loss | 0.128191 |
| focal_loss | 0.128191 |
| entropy_loss | 2.356764 |
| sparsity_loss | 0.083263 |
| predicate_loss | 0.000000 |
| curriculum_loss | 2.350595 |
| deep_supervision_loss | 0.152377 |
| act_loss | 0.000000 |
| loss_mode | weighted_stablemax |
| sparsity_min_clue_penalty | 0.000000 |
| sparsity_base_pondering | 0.038990 |
| sparsity_entropy_pondering | 0.044273 |
| expected_clues_used | 3.898968 |
| stop_prob_from_loss | 0.025258 |
| clues_used_std | 0.004596 |
| per_sample_clue_penalty_mean | 0.000000 |

### Metrics
- **Total Accuracy:** 91.15%
- **BG Accuracy:** 88.36%
- **FG Accuracy:** 100.00%

### Per-Example Metrics
| Example | Accuracy | BG Acc | FG Acc | Exact Match | Target Classes | Pred Classes |
|---------|----------|--------|--------|-------------|----------------|---------------|
| 1 | 100.0% | 100.0% | 100.0% | ✅ | [0, 5] | [0, 5] |
| 2 | 76.6% | 46.4% | 100.0% | ❌ | [0, 2, 3, 7, 8] | [0, 2, 3, 7, 8] |
| 3 | 96.9% | 96.4% | 100.0% | ❌ | [0, 1, 2] | [0, 1, 2] |

### DSC Analysis
| Example | Clues Used | Stop Probs | Stop Logits |
|---------|------------|------------|-------------|
| 1 | 3.89 | [0.033, 0.026, 0.023, 0.024] | [-3.38, -3.64, -3.73, -3.70] |
| 2 | 3.90 | [0.026, 0.021, 0.020, 0.034] | [-3.64, -3.86, -3.91, -3.36] |
| 3 | 3.90 | [0.031, 0.021, 0.023, 0.022] | [-3.43, -3.86, -3.73, -3.80] |

### Attention Entropy (per clue)
| Example | Clue 0 | Clue 1 | Clue 2 | Clue 3 | Clue 4 | Clue 5 |
|---------|--------|--------|--------|--------|--------|--------|
| 1 | 0.310 | 0.721 | 0.665 | 0.669 |
| 2 | 0.334 | 0.681 | 0.720 | 0.011 |
| 3 | 0.293 | 0.902 | 0.719 | 0.776 |


## Epoch 34

**Temperature:** 0.8888

### Losses
| Loss | Value |
|------|-------|
| total_loss | 0.217312 |
| task_loss | 0.132515 |
| focal_loss | 0.132515 |
| entropy_loss | 2.285685 |
| sparsity_loss | 0.081677 |
| predicate_loss | 0.000000 |
| curriculum_loss | 2.324239 |
| deep_supervision_loss | 0.153259 |
| act_loss | 0.000000 |
| loss_mode | weighted_stablemax |
| sparsity_min_clue_penalty | 0.000000 |
| sparsity_base_pondering | 0.038885 |
| sparsity_entropy_pondering | 0.042792 |
| expected_clues_used | 3.888488 |
| stop_prob_from_loss | 0.027878 |
| clues_used_std | 0.023253 |
| per_sample_clue_penalty_mean | 0.000000 |

### Metrics
- **Total Accuracy:** 90.10%
- **BG Accuracy:** 87.67%
- **FG Accuracy:** 97.83%

### Per-Example Metrics
| Example | Accuracy | BG Acc | FG Acc | Exact Match | Target Classes | Pred Classes |
|---------|----------|--------|--------|-------------|----------------|---------------|
| 1 | 98.4% | 100.0% | 0.0% | ❌ | [0, 5] | [0] |
| 2 | 76.6% | 46.4% | 100.0% | ❌ | [0, 2, 3, 7, 8] | [0, 2, 3, 7, 8] |
| 3 | 95.3% | 94.5% | 100.0% | ❌ | [0, 1, 2] | [0, 1, 2] |

### DSC Analysis
| Example | Clues Used | Stop Probs | Stop Logits |
|---------|------------|------------|-------------|
| 1 | 3.87 | [0.039, 0.029, 0.040, 0.025] | [-3.20, -3.52, -3.18, -3.68] |
| 2 | 3.91 | [0.020, 0.020, 0.020, 0.026] | [-3.88, -3.89, -3.90, -3.61] |
| 3 | 3.88 | [0.024, 0.037, 0.028, 0.028] | [-3.71, -3.27, -3.56, -3.55] |

### Attention Entropy (per clue)
| Example | Clue 0 | Clue 1 | Clue 2 | Clue 3 | Clue 4 | Clue 5 |
|---------|--------|--------|--------|--------|--------|--------|
| 1 | 0.544 | 0.471 | 0.378 | 0.582 |
| 2 | 0.643 | 0.567 | 0.816 | 0.249 |
| 3 | 0.822 | 0.360 | 0.434 | 0.729 |


## Epoch 35

**Temperature:** 0.8858

### Losses
| Loss | Value |
|------|-------|
| total_loss | 0.197186 |
| task_loss | 0.119145 |
| focal_loss | 0.119145 |
| entropy_loss | 2.377560 |
| sparsity_loss | 0.083603 |
| predicate_loss | 0.000000 |
| curriculum_loss | 2.322134 |
| deep_supervision_loss | 0.139362 |
| act_loss | 0.000000 |
| loss_mode | weighted_stablemax |
| sparsity_min_clue_penalty | 0.000000 |
| sparsity_base_pondering | 0.038965 |
| sparsity_entropy_pondering | 0.044638 |
| expected_clues_used | 3.896474 |
| stop_prob_from_loss | 0.025881 |
| clues_used_std | 0.031132 |
| per_sample_clue_penalty_mean | 0.000000 |

### Metrics
- **Total Accuracy:** 95.83%
- **BG Accuracy:** 94.52%
- **FG Accuracy:** 100.00%

### Per-Example Metrics
| Example | Accuracy | BG Acc | FG Acc | Exact Match | Target Classes | Pred Classes |
|---------|----------|--------|--------|-------------|----------------|---------------|
| 1 | 100.0% | 100.0% | 100.0% | ✅ | [0, 5] | [0, 5] |
| 2 | 87.5% | 71.4% | 100.0% | ❌ | [0, 2, 3, 7, 8] | [0, 2, 3, 7, 8] |
| 3 | 100.0% | 100.0% | 100.0% | ✅ | [0, 1, 2] | [0, 1, 2] |

### DSC Analysis
| Example | Clues Used | Stop Probs | Stop Logits |
|---------|------------|------------|-------------|
| 1 | 3.92 | [0.022, 0.020, 0.022, 0.020] | [-3.79, -3.88, -3.79, -3.87] |
| 2 | 3.91 | [0.020, 0.025, 0.022, 0.020] | [-3.90, -3.66, -3.79, -3.91] |
| 3 | 3.86 | [0.026, 0.049, 0.033, 0.032] | [-3.63, -2.97, -3.39, -3.39] |

### Attention Entropy (per clue)
| Example | Clue 0 | Clue 1 | Clue 2 | Clue 3 | Clue 4 | Clue 5 |
|---------|--------|--------|--------|--------|--------|--------|
| 1 | 0.601 | 0.570 | 0.616 | 0.736 |
| 2 | 0.734 | 0.366 | 0.586 | 0.694 |
| 3 | 0.705 | 0.131 | 0.774 | 0.346 |


## Epoch 36

**Temperature:** 0.8827

### Losses
| Loss | Value |
|------|-------|
| total_loss | 0.190220 |
| task_loss | 0.114272 |
| focal_loss | 0.114272 |
| entropy_loss | 2.492689 |
| sparsity_loss | 0.086050 |
| predicate_loss | 0.000000 |
| curriculum_loss | 2.329815 |
| deep_supervision_loss | 0.134688 |
| act_loss | 0.000000 |
| loss_mode | weighted_stablemax |
| sparsity_min_clue_penalty | 0.000000 |
| sparsity_base_pondering | 0.039124 |
| sparsity_entropy_pondering | 0.046925 |
| expected_clues_used | 3.912441 |
| stop_prob_from_loss | 0.021890 |
| clues_used_std | 0.006467 |
| per_sample_clue_penalty_mean | 0.000000 |

### Metrics
- **Total Accuracy:** 98.44%
- **BG Accuracy:** 97.95%
- **FG Accuracy:** 100.00%

### Per-Example Metrics
| Example | Accuracy | BG Acc | FG Acc | Exact Match | Target Classes | Pred Classes |
|---------|----------|--------|--------|-------------|----------------|---------------|
| 1 | 100.0% | 100.0% | 100.0% | ✅ | [0, 5] | [0, 5] |
| 2 | 95.3% | 89.3% | 100.0% | ❌ | [0, 2, 3, 7, 8] | [0, 2, 3, 7, 8] |
| 3 | 100.0% | 100.0% | 100.0% | ✅ | [0, 1, 2] | [0, 1, 2] |

### DSC Analysis
| Example | Clues Used | Stop Probs | Stop Logits |
|---------|------------|------------|-------------|
| 1 | 3.91 | [0.023, 0.025, 0.023, 0.021] | [-3.77, -3.65, -3.74, -3.86] |
| 2 | 3.92 | [0.020, 0.020, 0.021, 0.019] | [-3.89, -3.88, -3.86, -3.94] |
| 3 | 3.91 | [0.022, 0.020, 0.027, 0.021] | [-3.80, -3.87, -3.58, -3.83] |

### Attention Entropy (per clue)
| Example | Clue 0 | Clue 1 | Clue 2 | Clue 3 | Clue 4 | Clue 5 |
|---------|--------|--------|--------|--------|--------|--------|
| 1 | 0.681 | 0.374 | 0.534 | 0.734 |
| 2 | 0.619 | 0.739 | 0.512 | 0.784 |
| 3 | 0.684 | 0.751 | 0.217 | 0.564 |


## Epoch 37

**Temperature:** 0.8796

### Losses
| Loss | Value |
|------|-------|
| total_loss | 0.190726 |
| task_loss | 0.112158 |
| focal_loss | 0.112158 |
| entropy_loss | 2.743025 |
| sparsity_loss | 0.090698 |
| predicate_loss | 0.000000 |
| curriculum_loss | 2.311955 |
| deep_supervision_loss | 0.138995 |
| act_loss | 0.000000 |
| loss_mode | weighted_stablemax |
| sparsity_min_clue_penalty | 0.000000 |
| sparsity_base_pondering | 0.039102 |
| sparsity_entropy_pondering | 0.051596 |
| expected_clues_used | 3.910157 |
| stop_prob_from_loss | 0.022461 |
| clues_used_std | 0.006974 |
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
| 1 | 3.91 | [0.025, 0.024, 0.023, 0.024] | [-3.68, -3.71, -3.76, -3.71] |
| 2 | 3.92 | [0.019, 0.022, 0.020, 0.021] | [-3.92, -3.79, -3.91, -3.86] |
| 3 | 3.91 | [0.022, 0.027, 0.021, 0.024] | [-3.81, -3.59, -3.86, -3.72] |

### Attention Entropy (per clue)
| Example | Clue 0 | Clue 1 | Clue 2 | Clue 3 | Clue 4 | Clue 5 |
|---------|--------|--------|--------|--------|--------|--------|
| 1 | 0.630 | 0.518 | 0.715 | 0.547 |
| 2 | 0.786 | 0.522 | 0.815 | 0.642 |
| 3 | 0.739 | 0.363 | 0.801 | 0.836 |


## 🎉 SUCCESS: All examples solved at epoch 37!

---

# Final Summary

- **Best Accuracy:** 100.00% (epoch 37)
- **All Exact Match:** ✅ Yes

