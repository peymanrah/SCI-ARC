# TRM vs RLAN: Critical Comparison for Generalization Gap Fix

## Executive Summary

RLAN shows **0.2% eval exact match** vs **20.7% train exact match** (100x gap).
This document analyzes why TRM succeeds at evaluation while RLAN fails.

---

## 1. Key Architectural Differences

### 1.1 TRM Architecture (What Works)

```
TRM uses a DETERMINISTIC architecture at both train and eval:

Input → TokenEmbedding + PuzzleEmbedding → RoPE/Learned Positional Encoding
     ↓
     ├─ H-cycles (outer loop, H_cycles-1 without grad)
     │   └─ L-cycles (inner loop)
     │       └─ L_level (Attention/MLP + RMS Norm)
     ↓
LM Head → Output Predictions
Q Head → Halt/Continue Logits (for adaptive computation)
```

**Critical TRM Properties:**
1. **NO Gumbel softmax** - pure deterministic attention
2. **NO temperature decay** - standard softmax scaling
3. **NO stochastic components** - same behavior train/eval
4. **Inverse augmentation** at eval - undoes transforms on predictions
5. **Aggregated voting** - multiple predictions → vote for consensus
6. **Q-head confidence** - ranks predictions by quality

### 1.2 RLAN Architecture (What Fails)

```
RLAN uses STOCHASTIC attention during training, DETERMINISTIC during eval:

Input → ContextEncoder (256-dim bottleneck) → Task Context
     ↓
DSC (Dynamic Saliency Controller):
  - ClueQueryBank (static learned vectors, NOT task-conditioned)
  - Gumbel-Softmax attention (STOCHASTIC when training!)
  - Temperature τ (decays from 1.0 → 0.5)
     ↓
MSRE (Multi-Scale Relative Encoding):
  - Sine/Cosine positional features
     ↓
RecursiveSolver → Predictions
```

**Critical RLAN Problems:**
1. **Gumbel noise dependency** - Model learns to exploit noise patterns
2. **Temperature changes** - τ decays during training but fixed at eval
3. **Static clue queries** - Not conditioned on task, can't adapt
4. **Context bottleneck** - 256-dim vector loses task-specific info
5. **Entropy loss timing** - Computed POST-noise (measures noise variance)

---

## 2. The Gumbel Noise Problem (Root Cause #1)

### 2.1 What RLAN Does

```python
# In DSC forward (dynamic_saliency_controller.py:355-358)
attention = gumbel_softmax_2d(
    attn_scores, 
    temperature=temperature,
    deterministic=not self.training  # TRUE at eval = no noise!
)
```

**Training Mode (Gumbel ON):**
```
logits = [3.0, 1.0, 0.5]
gumbel_noise = [-0.1, 0.8, -0.2]  # Random samples
noisy_logits = [2.9, 1.8, 0.3]    # Different ranking!
attention = softmax(noisy_logits / τ)
```

**Eval Mode (Gumbel OFF):**
```
logits = [3.0, 1.0, 0.5]          # Same logits
attention = softmax(logits / τ)   # VERY different result!
```

### 2.2 Mathematical Impact

Let $L = (l_1, l_2, ..., l_k)$ be attention logits and $G = (g_1, ..., g_k)$ be Gumbel noise.

**Training:** 
$$a_i^{train} = \frac{\exp((l_i + g_i)/\tau)}{\sum_j \exp((l_j + g_j)/\tau)}$$

**Eval:**
$$a_i^{eval} = \frac{\exp(l_i/\tau)}{\sum_j \exp(l_j/\tau)}$$

The expected difference is NOT zero because Gumbel noise has **non-linear effects through softmax**.

### 2.3 Why Entropy is 0.02 (Train) vs 3.82 (Eval)

The model learns attention patterns that **REQUIRE** Gumbel noise for stability:
- **Training entropy 0.02**: Noise + learned logits → sharp effective attention
- **Eval entropy 3.82**: Same logits WITHOUT noise → diffuse, uncertain attention

The 180x entropy increase proves the model depends on noise properties.

---

## 3. TRM's Solution: Zero Stochastic Components

TRM uses **standard softmax attention** - identical train and eval:

```python
# TRM attention (models/layers.py)
class Attention(nn.Module):
    def forward(self, cos_sin, hidden_states):
        # Standard scaled dot-product attention
        # NO Gumbel noise, NO temperature decay
        attn_weights = torch.matmul(Q, K.T) / sqrt(d_k)
        attn_weights = F.softmax(attn_weights, dim=-1)
        ...
```

**Result:** TRM attention behaves identically during training and evaluation.

---

## 4. Inverse Augmentation (TRM's Key Eval Trick)

### 4.1 TRM Approach

```python
# evaluators/arc.py
def update_batch(self, batch, preds):
    for identifier, input, pred, q in zip(...):
        name = self.identifier_map[identifier]
        orig_name, _inverse_fn = inverse_aug(name)  # Get inverse transform!
        
        pred = _inverse_fn(_crop(pred))  # UNDO augmentation on prediction
```

TRM encodes augmentation info in the identifier:
```
"task_123|||t5|||0139267845"
           ^    ^
           |    color permutation
           dihedral transform ID
```

At eval, it **reverses** the transform to get canonical predictions.

### 4.2 RLAN Problem

RLAN applies random augmentation during eval but does NOT reverse it:
```python
# RLAN eval (train_rlan.py)
for batch in eval_loader:
    # batch has random augmentation applied
    outputs = model(batch)
    preds = outputs['pred']  # Still in augmented space!
    # Compare to ground truth in original space → MISMATCH
```

---

## 5. Aggregated Voting (TRM's Robustness)

TRM generates **many predictions per task** and votes:

```python
# evaluators/arc.py
for h, stats in p_map.items():
    stats[1] /= stats[0]  # Average q-value

p_map = sorted(p_map.items(), key=lambda kv: kv[1], reverse=True)  # Sort by confidence

for i, k in enumerate(self.pass_Ks):  # Test at K=1,2,5,10,100,1000
    ok = any(h == label_hash for h, stats in p_map[:k])
```

**Benefits:**
- Multiple augmented views → same canonical answer
- Confidence scores filter bad predictions
- Pass@K metrics show improvement with more tries

---

## 6. EMA Lag Issue

### 6.1 Observed Symptoms

```
train_stop_value: 0.574
eval_stop_value: 0.342
```

EMA model (used for eval) is lagging behind training model.

### 6.2 Root Cause

With `ema_decay=0.999`:
- Each update: $\theta_{ema} = 0.999 \cdot \theta_{ema} + 0.001 \cdot \theta$
- Needs ~1000 steps to catch up to new weights
- After 5 epochs × ~200 steps = 1000 updates
- EMA still reflects early-training weights!

### 6.3 TRM Comparison

TRM also uses EMA but:
1. Runs for 100K+ epochs (EMA fully converged)
2. Uses `ema_rate=0.999` but many more updates
3. Evaluates at longer intervals

---

## 7. Recommended Fixes

### Fix 1: Remove Gumbel Noise Dependency (CRITICAL)

Option A - Use same noise distribution at eval:
```python
# In gumbel_softmax_2d()
if deterministic:
    # Use a fixed seed based on attention scores to get reproducible noise
    torch.manual_seed(int(attn_scores.sum().item() * 1e6) % (2**32))
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(attn_scores) + 1e-10) + 1e-10)
    attn_scores = attn_scores + gumbel_noise
```

Option B - Train without Gumbel noise (recommended):
```python
# Remove Gumbel entirely, use straight-through estimator
attention = F.softmax(attn_scores / temperature, dim=-1)
# For gradient: use straight-through if needed for hard attention
```

Option C - KL regularization to minimize noise impact:
```python
# During training, add loss term
noisy_attn = gumbel_softmax_2d(logits, temp, deterministic=False)
clean_attn = gumbel_softmax_2d(logits, temp, deterministic=True)
kl_loss = F.kl_div(clean_attn.log(), noisy_attn, reduction='batchmean')
total_loss += 0.1 * kl_loss  # Encourage consistency
```

### Fix 2: Fix Entropy Loss Timing

Current (broken):
```python
# Entropy computed on POST-noise attention
entropy_loss = entropy_of_attention(noisy_attention)  # Measures noise variance
```

Fixed:
```python
# Entropy on raw logits BEFORE any noise
raw_probs = F.softmax(attn_scores / temperature, dim=-1)
entropy_loss = entropy_of_attention(raw_probs)  # Measures true uncertainty
```

### Fix 3: Add Inverse Augmentation at Eval

```python
# In evaluation loop
def inverse_transform(pred, aug_info):
    """Reverse augmentation to canonical space."""
    if aug_info.rotation:
        pred = torch.rot90(pred, -aug_info.rotation // 90, dims=(-2, -1))
    if aug_info.flip_h:
        pred = torch.flip(pred, dims=[-1])
    if aug_info.flip_v:
        pred = torch.flip(pred, dims=[-2])
    if aug_info.color_perm:
        inv_perm = torch.argsort(aug_info.color_perm)
        pred = inv_perm[pred]
    return pred
```

### Fix 4: Change eval_every to 1

```yaml
# configs/rlan_stable.yaml
logging:
  eval_every: 1  # Was 5, now eval every epoch
```

### Fix 5: Add Train/Eval Gap Monitoring

```python
class GapHealthMonitor:
    def __init__(self, warning_threshold=0.2, critical_threshold=0.5):
        self.train_metrics = []
        self.eval_metrics = []
        
    def check_gap(self, train_metric, eval_metric, metric_name):
        gap = abs(train_metric - eval_metric) / max(train_metric, 0.01)
        if gap > self.critical_threshold:
            logger.critical(f"CRITICAL GAP in {metric_name}: train={train_metric:.4f}, eval={eval_metric:.4f}, gap={gap:.1%}")
        elif gap > self.warning_threshold:
            logger.warning(f"Warning: gap in {metric_name}: train={train_metric:.4f}, eval={eval_metric:.4f}, gap={gap:.1%}")
```

### Fix 6: Make Clue Queries Task-Conditioned

Current (static):
```python
self.clue_query_bank = nn.Parameter(torch.randn(num_clues, hidden_dim))
```

Fixed (conditioned on task context):
```python
self.clue_query_projector = nn.Linear(context_dim, num_clues * hidden_dim)

def get_clue_queries(self, context):
    return self.clue_query_projector(context).view(-1, self.num_clues, self.hidden_dim)
```

---

## 8. Recommended New Training Config

Create `configs/rlan_fixed.yaml`:

```yaml
model:
  # ... existing settings ...
  
  # NEW: Disable Gumbel noise (train like eval)
  use_gumbel_softmax: false  # Use straight-through softmax instead
  
  # NEW: Task-conditioned clue queries
  task_conditioned_clues: true

training:
  # ... existing settings ...
  
  # NEW: Fixed temperature (no decay = same train/eval)
  temperature_start: 0.5
  temperature_end: 0.5  # Same as start = no decay
  
  # NEW: Entropy on raw logits
  entropy_on_raw_logits: true
  
  # NEW: KL consistency loss
  lambda_kl_consistency: 0.1

evaluation:
  # NEW: Inverse augmentation
  use_inverse_augmentation: true
  
  # NEW: Aggregated voting like TRM
  num_augmented_views: 8  # Generate 8 predictions per task
  use_voting: true

logging:
  eval_every: 1  # Eval every epoch for gap monitoring
```

---

## 9. Summary: Why TRM Works and RLAN Fails

| Aspect | TRM | RLAN | Fix |
|--------|-----|------|-----|
| Attention | Deterministic softmax | Gumbel softmax (stochastic) | Remove Gumbel |
| Train/Eval | Identical behavior | Different noise | Match train=eval |
| Temperature | Fixed | Decays | Fix at 0.5 |
| Augmentation | Inverse at eval | Not reversed | Add inverse |
| Voting | Aggregated | Single prediction | Add voting |
| Clue queries | N/A | Static | Make dynamic |
| Entropy loss | N/A | Post-noise | Pre-noise |
| EMA | Converged (100K epochs) | Lagging (5 epochs) | Faster decay |

**The fundamental issue:** RLAN was designed with training tricks (Gumbel noise, temperature annealing) that improve training metrics but break generalization because eval doesn't use them.

**The fix:** Make RLAN train the same way it evaluates - no stochastic components, fixed temperature, inverse augmentation.
