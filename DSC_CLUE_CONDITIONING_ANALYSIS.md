# DSC Clue Conditioning & Solver Depth Analysis
## Deep Investigation Report (January 2026)

---

## Executive Summary

This analysis investigates three critical architectural questions:

1. **Are clue counts task-conditioned?** ✅ YES - Mathematically verified
2. **Should solver steps increase from 4 to 6-7?** ✅ YES - Config already uses 7
3. **Is ARPS search depth sufficient?** ⚠️ POSSIBLY INSUFFICIENT for complex tasks

---

## 1. DSC Clue Count Task-Conditioning Analysis

### 1.1 Mathematical Proof: Clue Counts ARE Task-Conditioned

**Data Flow Trace:**

```
Task (i) in Batch B
    ↓
ContextEncoder.encode()
    ↓ processes support pairs for task i only
pool_context_from_support()  
    ↓ mean(dim=(1,3,4)) → pools over (pairs, H, W) BUT NOT over batch!
task_context: (B, D)  ← Each batch item has DIFFERENT context vector
    ↓
DSC.forward(task_context)
    ↓
stop_predictor([attended_features, confidence, task_context_vec])
    ↓
stop_logits: (B, K) ← Per-sample stop decisions
    ↓
expected_clues_used = (1 - stop_probs).sum(dim=-1)
    ↓
Result: (B,) ← EACH BATCH ITEM HAS INDEPENDENT CLUE COUNT
```

**Key Code Locations Verified:**

| Component | File | Lines | Operation | Preserves Task-Level? |
|-----------|------|-------|-----------|----------------------|
| Context Pooling | `rlan.py` | 966-979 | `mean(dim=(1,3,4))` | ✅ YES - pools over spatial/pairs, NOT batch |
| Stop Predictor Input | `dynamic_saliency_controller.py` | 476-479 | `cat([attended, conf, task_context])` | ✅ YES - task_context is (B, C) |
| Stop Logits | `dynamic_saliency_controller.py` | 482 | `stop_predictor(stop_input)` | ✅ YES - MLP per sample |
| Expected Clues | `rlan_loss.py` | 884 | `(1-stop_probs).sum(dim=-1)` | ✅ YES - sum over clues K, output (B,) |
| Min Clue Penalty | `rlan_loss.py` | 888 | `relu(min_clues - expected)` | ✅ YES - per-sample (B,) |
| Variance Reg | `rlan_loss.py` | 918-923 | `expected_clues_used.std()` | ✅ YES - std ACROSS batch = per-task variation |

### 1.2 The Only Batch-Level Operations Are INTENTIONAL

1. **`base_pondering = expected_clues_used.mean() * ponder_weight`**
   - This is correct: regularization to minimize total clue usage
   - Gradient flows back to each sample proportionally

2. **`variance_penalty = relu(target_variance - clues_used.std()) * weight`**
   - This ENCOURAGES different clue counts per task
   - `std()` across batch measures task-level variation
   - Penalty when variance is LOW (all tasks use same clues)

### 1.3 Variance Regularization Ensures Task-Dependent Clue Counts

```python
# From rlan_loss.py lines 918-923
clues_used_variance = expected_clues_used.std()  # Std across batch
if self.variance_weight > 0:
    variance_deficit = F.relu(self.target_variance - clues_used_variance)
    variance_penalty = variance_deficit * self.variance_weight
```

**Config Values (rlan_stable_dev.yaml):**
- `clue_variance_weight: 1.0` - Strong weight on variance regularization
- `clue_target_variance: 0.5` - Target std of 0.5 clues across batch

**Interpretation:** If all tasks in a batch use the same number of clues (e.g., all use 3.5), variance = 0, penalty is high. This FORCES the model to learn task-dependent clue counts.

### 1.4 CONCLUSION: Task-Conditioning is CORRECT

The implementation correctly:
1. Computes task_context per-sample (B, D)
2. Passes task_context to stop_predictor for each sample
3. Computes clue counts per-sample (B,)
4. Regularizes for VARIANCE across samples (encourages task-dependence)

**No batch-level averaging "washes away" task conditioning.**

---

## 2. Centroid Diversity Loss Investigation

### 2.1 Why Centroid Spread Was 0.00

The CentroidDiversityLoss has been updated (see lines 978-1100 in rlan_loss.py):

**Previous Issues (Fixed Dec 2025):**
- `min_distance: 2.0` was too small - clues could overlap at 2 pixels
- `repulsion_weight: 0.1` was too weak - insufficient continuous gradient

**Current Settings (Fixed):**
```python
class CentroidDiversityLoss(nn.Module):
    def __init__(
        self,
        min_distance: float = 3.0,      # INCREASED from 2.0
        weight_by_usage: bool = True,
        repulsion_weight: float = 0.3,  # INCREASED from 0.1 (3x stronger)
    ):
```

### 2.2 Soft Masking in DSC

The DSC also uses cumulative masking to force spatial diversity:

```python
# From dynamic_saliency_controller.py lines 533-541
# Increase from 0.9 to 0.98 to force stronger spatial diversity
mask_update = 1.0 - 0.98 * attention.detach()
cumulative_mask = cumulative_mask * mask_update
cumulative_mask = cumulative_mask.clamp(min=1e-6)
```

**Effect:** When attention=1.0, mask becomes 0.02 (was 0.1). This strongly discourages subsequent clues from attending to the same region.

### 2.3 CONCLUSION: Centroid Collapse Has Been Addressed

The combination of:
1. Increased `min_distance: 3.0`
2. Stronger `repulsion_weight: 0.3`
3. Harder cumulative masking (0.98 vs 0.9)

Should prevent centroid collapse. If still observing spread < 0.5, consider:
- Increasing `lambda_centroid_diversity` from 0.3 to 0.5
- Increasing `min_distance` to 4.0 for 30x30 grids

---

## 3. Solver Steps Analysis: 4 vs 6-7

### 3.1 Current Configuration

```yaml
# rlan_stable_dev.yaml
model:
  num_solver_steps: 7   # Increased from 5 for more iteration capacity
```

The config already uses **7 solver steps**, not 4.

### 3.2 Evidence for Step Selection

From documentation and investigation logs:

| Source | Recommendation | Reasoning |
|--------|----------------|-----------|
| `INVESTIGATION_SUMMARY.md` | Reduce from 6 to 4 | "logs showed best step was 4 anyway" |
| `rlan_stable.yaml` | 5 steps | "Reduced from 6 to save ~150MB" |
| `rlan_stable_dev.yaml` | 7 steps | "User request Jan 2026" |
| `rlan_stable_dev_512.yaml` | 7 steps | "Same - iteration count (this is RLAN's 'depth')" |
| `README.md` | 6 steps | "Recursive solver iterations" |

### 3.3 Step Count Tradeoffs

| Steps | Memory | Complex Task Accuracy | Simple Task Risk |
|-------|--------|----------------------|------------------|
| 4 | Lowest | May underperform | Minimal over-iteration |
| 5 | Moderate | Good balance | Low risk |
| 6 | Moderate | Standard | Some over-iteration |
| 7 | Highest | Best for complex | Higher over-iteration risk |

### 3.4 Best-Step Selection Mitigates Over-Iteration

```yaml
# rlan_stable_dev.yaml
model:
  use_best_step_selection: false  # Training: use all steps
  # At inference: set true to use entropy-based step selection
```

**Recommendation:** Keep `num_solver_steps: 7` but enable `use_best_step_selection: true` at inference. This gives:
- Maximum capacity for complex tasks
- Automatic early stopping for simple tasks

### 3.5 CONCLUSION: 7 Steps is Already Configured

The production config uses 7 steps. No change needed. For efficiency:
- Consider `use_best_step_selection: true` at inference
- Could reduce to 5-6 for faster training if memory is tight

---

## 4. ARPS Search Depth Analysis

### 4.1 Current ARPS Configuration

```yaml
# rlan_stable_dev_ablation.yaml
arps:
  enabled: true
  max_program_length: 8        # Max tokens in program sequence
  beam_size: 32                # Beam search width
  top_k_proposals: 4           # Top-K programs to verify
```

### 4.2 Search Space Analysis

**Token Types:**
- Operations: ~10 (translate, rotate, scale, copy, etc.)
- Color arguments: 10 (0-9)
- Offset arguments: ~60 (-30 to +30 for each axis)

**Effective Search Space:**
- At depth 8: O(80^8) = 10^15 possible programs
- Beam size 32: explores only 32*8 = 256 paths
- Top-K 4: verifies only 4 final programs

### 4.3 Generalization Concerns

| Metric | Current Value | Concern |
|--------|---------------|---------|
| `max_program_length` | 8 | May be insufficient for multi-step transformations |
| `beam_size` | 32 | Reasonable, but could miss good programs |
| `top_k_proposals` | 4 | Very narrow - may miss best program |

**ARC Task Complexity:**
- Simple tasks (fill, copy): 2-3 tokens sufficient
- Medium tasks (multi-step): 4-6 tokens
- Complex tasks (nested patterns): 8+ tokens potentially needed

### 4.4 Recommendations for ARPS

For better generalization:

```yaml
arps:
  max_program_length: 12       # INCREASE: Allow longer programs for complex tasks
  beam_size: 64                # INCREASE: Explore more paths
  top_k_proposals: 8           # INCREASE: Verify more candidate programs
```

**Memory/Time Tradeoff:**
- 2x beam_size = ~2x time
- +4 program length = ~1.5x time
- 2x top_k = ~2x verification time

### 4.5 CONCLUSION: ARPS Search May Be Insufficient

Current settings are conservative. For production generalization, consider:
1. Increase `max_program_length` to 10-12
2. Increase `beam_size` to 64
3. Increase `top_k_proposals` to 8

---

## 5. Summary of Findings

### ✅ VERIFIED: DSC Clue Count is Task-Conditioned

- Each batch item has independent `task_context` vector
- `expected_clues_used` is computed per-sample (B,)
- Variance regularization ENCOURAGES task-dependent clue counts
- No batch-level averaging washes away task conditioning

### ✅ VERIFIED: Centroid Diversity is Strengthened

- `min_distance: 3.0` (was 2.0)
- `repulsion_weight: 0.3` (was 0.1)
- Cumulative mask: 0.98 (was 0.9)
- If still collapsing, increase `lambda_centroid_diversity`

### ✅ VERIFIED: Solver Steps Already at 7

- Production config uses `num_solver_steps: 7`
- Enable `use_best_step_selection: true` at inference
- No change needed for step count

### ⚠️ RECOMMENDATION: Increase ARPS Search Depth

| Parameter | Current | Recommended | Reason |
|-----------|---------|-------------|--------|
| `max_program_length` | 8 | 12 | Complex tasks need longer programs |
| `beam_size` | 32 | 64 | More exploration for diverse solutions |
| `top_k_proposals` | 4 | 8 | More candidates for verification |

---

## 6. Appendix: Key Code Excerpts

### 6.1 Task Context Computation (rlan.py)

```python
def pool_context_from_support(self, support_features: torch.Tensor) -> torch.Tensor:
    """
    Pool support features to get task context.
    
    Args:
        support_features: (B, N_pairs, D, H, W) support set features
        
    Returns:
        task_context: (B, D) pooled context per task
    """
    # Pool over pairs, height, width BUT NOT BATCH
    # Each batch item = different task
    task_context = support_features.mean(dim=(1, 3, 4))  # (B, D)
    return task_context
```

### 6.2 Stop Predictor Input (dynamic_saliency_controller.py)

```python
# Concatenate attended features with confidence and global task context
stop_input = torch.cat(
    [attended_features, attn_confidence, task_context_vec], dim=-1
)  # (B, D+1+C)

# Predict stop probability with entropy-aware input
stop_logit_raw = self.stop_predictor(stop_input).squeeze(-1)  # (B,)
```

### 6.3 Clue Count Computation (rlan_loss.py)

```python
# Expected clues used = sum of (1 - stop_prob) across clues
expected_clues_used = (1 - stop_probs).sum(dim=-1)  # (B,)

# Variance regularization: penalize low variance in clue usage
clues_used_variance = expected_clues_used.std()  # Std across batch
variance_deficit = F.relu(self.target_variance - clues_used_variance)
variance_penalty = variance_deficit * self.variance_weight
```

---

## 7. Action Items

1. **No change needed for DSC** - Task-conditioning is correct
2. **No change needed for solver steps** - Already at 7
3. **Consider updating ARPS config** if generalization is poor:
   ```yaml
   arps:
     max_program_length: 12
     beam_size: 64
     top_k_proposals: 8
   ```
4. **Monitor centroid spread** - If < 1.0 after many epochs, increase `lambda_centroid_diversity` to 0.5

---

*Document generated: January 2026*
*Author: GitHub Copilot Analysis*
