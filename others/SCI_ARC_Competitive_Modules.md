# SCI-ARC Competitive Modules Implementation Guide

**Status: IMPLEMENTED** (December 2024)

This document details the competitive inference modules implemented in `sci_arc/inference/` to maximize ARC-AGI-1/2 performance. These modules focus on **Inference**, **Adaptation**, and **Verification**.

## Implementation Summary

| Module | File | Status | Config Key |
|--------|------|--------|------------|
| Stochastic Sampling | `sci_arc/inference/sampler.py` | ✅ Complete | `inference.use_stochastic_sampling` |
| Test-Time Training | `sci_arc/inference/ttt.py` (`TTTAdapter`) | ✅ Complete | `inference.use_ttt` |
| Ensemble Predictor | `sci_arc/inference/ensemble.py` | ✅ Complete | Multiple keys |
| Consistency Verification | `sci_arc/inference/sampler.py` | ✅ Complete | `inference.use_consistency_verification` |

## 1. Stochastic Sampling (Monte Carlo Dropout + Temperature)

**File:** `sci_arc/inference/sampler.py`

### Concept
Instead of a single greedy forward pass, we generate N candidate predictions with stochasticity enabled, then aggregate via voting.

### Implementation Details

**Key Features:**
- **MC Dropout**: Enables dropout layers during inference to create diverse predictions
- **Temperature Scaling**: Adjusts softmax sharpness for exploration/exploitation tradeoff
- **Top-K/Nucleus Sampling**: Constrains sampling to high-probability outputs
- **Automatic Deduplication**: Counts frequency of identical predictions

**Mathematical Stability Safeguards:**
```python
# Temperature clamped to prevent numerical issues
temperature = max(0.1, min(2.0, temperature))

# Guard against NaN/Inf in logits
scaled_logits = torch.nan_to_num(scaled_logits, nan=0.0, posinf=10.0, neginf=-10.0)

# Probability normalization guard
probs_flat = probs_flat.clamp(min=1e-8)
probs_flat = probs_flat / probs_flat.sum(dim=-1, keepdim=True)
```

### Configuration
```yaml
inference:
  use_stochastic_sampling: true
  num_samples: 32
  sampling_temperature: 0.8
  use_mc_dropout: true
```

---

## 2. Test-Time Training (TTT)

**File:** `sci_arc/inference/ttt.py`

### Concept
Fine-tune the model on the task's demonstration pairs immediately before inference. This adapts the model to the specific transformation logic of the current task.

### Critical Design Decisions

**Why These Modules Are Fine-Tuned:**
- `grid_encoder`: Adapts to specific color patterns and object shapes
- `structural_encoder`: Learns the specific transformation rule

**Why SCL Components Are FROZEN:**
- The SCL projection head (`batch_norm`, `projector`) learns a **global** contrastive space across ALL transformation types
- Fine-tuning on a single task would bias this space toward one transformation
- This would destroy the carefully learned clustering and hurt generalization

### Mathematical Stability Safeguards
```python
# Gradient clipping (critical for small batch sizes)
torch.nn.utils.clip_grad_norm_(self._trainable_params, self.config.grad_clip)

# Label smoothing for regularization
loss = F.cross_entropy(logits, targets, label_smoothing=0.1)

# State restoration after each task
def reset(self):
    self._restore_state()  # CRITICAL: Prevents cross-task contamination
```

### Leave-One-Out Training
The TTT implementation uses leave-one-out cross-validation on the demos:
1. For each demo pair (input_i, output_i)
2. Train on all OTHER demos
3. Validate by predicting output_i from input_i
4. This maximizes training signal from limited data

### Configuration
```yaml
inference:
  use_ttt: true
  ttt_steps: 20
  ttt_learning_rate: 1.0e-4
  ttt_grad_clip: 1.0
  ttt_modules:
    - grid_encoder
    - structural_encoder
```

---

## 3. Consistency Verification (Alternative to Bi-Directional)

**File:** `sci_arc/inference/sampler.py` (`ConsistencyVerifier` class)

### Why NOT Bi-Directional Verification
The original proposal suggested training an inverse model. However:
- Doubles model parameters and training time
- Requires architectural changes
- May not generalize as well as the forward model

### Alternative: Augmentation Consistency
A correct prediction should be **consistent** under geometric augmentations:
1. Apply dihedral transform to inputs
2. Get prediction
3. Apply inverse transform to prediction
4. Check if it matches the original prediction

High consistency = higher confidence in the prediction.

### Implementation
```python
def compute_consistency_score(self, candidate, input_grids, output_grids, test_input):
    matches = 0
    for tid in range(num_augments):
        # Augment → Predict → Inverse
        aug_pred = inverse_transform(model(transform(inputs, tid)), tid)
        if np.array_equal(aug_pred, candidate):
            matches += 1
    return matches / num_augments
```

### Configuration
```yaml
inference:
  use_consistency_verification: true
```

---

## 4. Ensemble Predictor

**File:** `sci_arc/inference/ensemble.py`

### Pipeline
```
Task → [TTT Adapt] → [Augmentation Vote] → [Stochastic Sample] → [Aggregate] → [Verify] → Ranked Predictions
```

Each component can be toggled for ablation studies.

### Confidence Scoring
```python
confidence = 0.7 * frequency + 0.3 * consistency
```
- Frequency: How often this prediction appeared in sampling
- Consistency: How stable it is under augmentations

---

## 5. Ablation Study Support

The evaluation script `scripts/evaluate_competitive.py` supports systematic ablation:

```bash
# Baseline (no inference modules)
python scripts/evaluate_competitive.py --ablation baseline

# Full pipeline
python scripts/evaluate_competitive.py --ablation full

# Complete ablation study (all modes)
python scripts/evaluate_competitive.py --ablation all
```

### Ablation Modes
| Mode | TTT | Sampling | Voting | Consistency |
|------|-----|----------|--------|-------------|
| `baseline` | ❌ | ❌ | ❌ | ❌ |
| `voting_only` | ❌ | ❌ | ✅ | ❌ |
| `no_ttt` | ❌ | ✅ | ✅ | ✅ |
| `no_sampling` | ✅ | ❌ | ✅ | ✅ |
| `no_consistency` | ✅ | ✅ | ✅ | ❌ |
| `full` | ✅ | ✅ | ✅ | ✅ |

---

## 6. Usage Examples

### Quick Evaluation
```python
from sci_arc.models import SCIARC
from sci_arc.inference import EnsemblePredictor, EnsembleConfig

# Load model
model = SCIARC(config)
model.load_state_dict(checkpoint['model_state_dict'])

# Configure inference
config = EnsembleConfig(
    use_ttt=True,
    use_stochastic_sampling=True,
    use_augmentation_voting=True,
    num_samples=32,
    ttt_steps=20,
)

# Create predictor
predictor = EnsemblePredictor(model, config)

# Predict
results = predictor.predict_task(task)
top_prediction = results[0]['predictions'][0]['prediction']
```

### Command Line
```bash
# Full competitive evaluation
python scripts/evaluate_competitive.py \
    --config configs/competitive.yaml \
    --checkpoint checkpoints/best.pt \
    --split evaluation

# Ablation study
python scripts/evaluate_competitive.py \
    --config configs/competitive.yaml \
    --ablation all \
    --output-dir outputs/ablation
```

---

## 7. References

### Self-Consistency
- Paper: [Self-Consistency Improves Chain of Thought Reasoning](https://arxiv.org/abs/2203.11171)
- Key insight: Multiple samples + majority voting improves reasoning accuracy

### Test-Time Training
- Paper: [Test-Time Training with Self-Supervision](https://arxiv.org/abs/1909.13231)
- ARC-specific: [MARC Repository](https://github.com/ekinakyurek/marc)
- ARC-TTT Example: [BY571/ARC-TTT](https://github.com/BY571/ARC-TTT)

### Monte Carlo Dropout
- Paper: [Dropout as a Bayesian Approximation](https://arxiv.org/abs/1506.02142)
- Key insight: Dropout at test time provides uncertainty estimates

---

## 8. Expected Performance Impact

Based on literature and architecture analysis:

| Configuration | Expected Accuracy |
|--------------|------------------|
| SCI-ARC baseline (no modules) | ~20-25% |
| + Augmentation Voting | +5-10% |
| + Stochastic Sampling | +3-5% |
| + Test-Time Training | +15-25% |
| Full pipeline | ~45-55% |

Note: These are estimates. Actual performance depends on training quality and task distribution.

