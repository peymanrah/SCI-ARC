# Solver Plateau Analysis & Fixes (December 2025)

## Problem Statement
The RecursiveSolver accuracy plateaus after step 4, with only 1.4% mean improvement across 6 steps. Later steps sometimes **degrade** accuracy instead of improving it.

### Evidence from `trace_report.json`:
| Task | Step 1 | Step 2 | Step 3 | Step 4 | Step 5 | Step 6 | Trend |
|------|--------|--------|--------|--------|--------|--------|-------|
| 007bbfb7 | 88.9% | 88.9% | 77.8% | 77.8% | 77.8% | 77.8% | **DEGRADES** |
| 00d62c1b | 81.0% | 96.0% | 96.2% | 97.3% | 96.7% | 96.7% | Plateaus |
| 025d127b | 90.0% | 86.0% | 90.0% | 91.0% | 90.0% | 90.0% | Oscillates |

---

## Root Cause Analysis

### Root Cause 1: DSC Centroid Collapse
**Finding**: All 7 DSC centroids collapse to the center (~0.45, 0.50) with std < 0.02

**Evidence**:
```
Task 007bbfb7: max_centroid_dist = 0.063 (collapsed)
Task 00d62c1b: max_centroid_dist = 0.012 (severely collapsed)
```

**Impact**: When centroids collapse, MSRE produces **identical relative coordinates** for all clues. The solver cannot distinguish which clue to attend to.

**Fix**: `lambda_centroid_diversity=0.5` loss term (already committed in `20823fe`)

### Root Cause 2: Fixed Solver Input Each Step ⚠️ CRITICAL
**Finding**: The combined input `torch.cat([aggregated, input_embed], dim=1)` is **IDENTICAL at every step**.

**Evidence** from code analysis:
```python
# Line 1084 (BEFORE loop):
aggregated = self._aggregate_clues(clue_features, ...)  # Fixed!
input_embed = self.input_embed(input_grid)  # Fixed!

# Inside loop:
for t in range(num_steps):
    combined = torch.cat([aggregated, input_embed], dim=1)  # Same every step!
```

**Impact**: Only the GRU hidden state `h` can change, but with identical input, the GRU saturates quickly. Later steps cannot "see" their mistakes.

**Fix**: ✅ **Entropy-Guided Refinement** (implemented this session)

### Root Cause 3: Residual Lock (30% locked to step 0)
**Finding**: `h_new = 0.7 * h_new + 0.3 * h_initial` forces 30% of hidden state to step 0's output.

**Impact**: Even if the solver COULD improve, 30% of its capacity is locked to the initial (potentially incorrect) prediction.

**Fix**: ✅ **Progressive Residual Decay** (implemented this session)

### Root Cause 4: GRU Saturation
**Finding**: Logit changes decrease over steps: 5.7 → 4.3 → 5.3 → 4.0

**Impact**: GRU memory saturates when receiving identical input repeatedly.

**Root**: This is a **symptom** of Root Causes 1-3, not an independent issue.

---

## Implemented Fixes

### Fix 1: Entropy-Guided Refinement (`use_entropy_refinement=True`)

**Location**: `sci_arc/models/rlan_modules/recursive_solver.py`

**What it does**:
- After each step, compute prediction entropy: H(p) = -Σ p·log(p)
- Inject entropy map + soft predictions into next step's input
- Uses gated addition: `aggregated_refined = aggregated + gate * entropy_features`

**How it breaks the plateau**:
- Step 1: Solver sees base aggregated features
- Step 2+: Solver sees base features **+ where it was uncertain**
- High-entropy regions get more attention in later steps
- Allows iterative error correction

**Code added** (constructor):
```python
if use_entropy_refinement:
    self.entropy_proj = nn.Sequential(
        nn.Conv2d(1 + num_classes, hidden_dim, 3, padding=1),
        nn.GELU(),
        nn.GroupNorm(8, hidden_dim),
    )
    self.entropy_gate = nn.Sequential(
        nn.Conv2d(hidden_dim * 2, hidden_dim, 1),
        nn.Sigmoid(),
    )
```

**Code added** (forward loop):
```python
if self.use_entropy_refinement and prev_logits is not None:
    prev_probs = F.softmax(prev_logits, dim=1)
    entropy = -torch.sum(prev_probs * torch.log(prev_probs + 1e-8), dim=1, keepdim=True)
    entropy = entropy / max_entropy  # Normalize to [0, 1]
    
    entropy_input = torch.cat([entropy, prev_probs], dim=1)
    entropy_features = self.entropy_proj(entropy_input)
    gate = self.entropy_gate(torch.cat([aggregated, entropy_features], dim=1))
    aggregated_refined = aggregated + gate * entropy_features
```

### Fix 2: Progressive Residual Decay

**Location**: `sci_arc/models/rlan_modules/recursive_solver.py`, lines 1207-1220

**What it does**:
- Steps 1-2: Keep strong residual (0.3) for stability
- Steps 3+: Reduce residual to 0.1-0.15 for more flexibility

**Before**:
```python
h_new = 0.7 * h_new + 0.3 * h_initial  # Fixed 30% lock
```

**After**:
```python
if t <= 2:
    residual_weight = 0.3  # Early: strong anchor
else:
    residual_weight = max(0.1, 0.3 - 0.05 * (t - 2))  # Late: allow change
h_new = (1.0 - residual_weight) * h_new + residual_weight * h_initial
```

---

## Expected Impact

### Before Fixes:
- Mean improvement: **1.4%** over 6 steps
- Pattern: Improve steps 1-2, plateau steps 3-6, sometimes degrade

### After Fixes (Expected):
- Mean improvement: **5-10%** over 6 steps
- Pattern: Steady improvement each step as solver refines uncertain regions
- No degradation (progressive decay protects but doesn't lock)

### Step-by-Step Improvement Mechanism:
| Step | Entropy Input | Residual Weight | Expected Effect |
|------|---------------|-----------------|-----------------|
| 1 | None (first step) | N/A | Base prediction |
| 2 | High entropy everywhere | 0.3 | Global refinement |
| 3 | Medium entropy on edges | 0.25 | Edge refinement |
| 4 | Low entropy on easy, high on hard | 0.20 | Focus on hard pixels |
| 5 | Residual uncertainty | 0.15 | Final corrections |
| 6 | Minimal uncertainty | 0.10 | Polish |

---

## Optimal Step Count Analysis

### Current State (before fixes):
- **6 steps is already too many** - plateaus at step 4
- More steps = no benefit (same input → same output)

### After Fixes:
- **8-10 steps recommended** - entropy feedback enables true iterative refinement
- Each step now receives DIFFERENT input
- Diminishing returns after 10 steps (entropy converges to zero)

### Testing Recommendation:
```python
# In config.yaml:
solver_num_steps: 8  # or 10

# Or override at inference:
logits = solver(x, num_steps_override=10)
```

---

## Verification Steps

1. **Re-run diagnostic**:
   ```bash
   python scripts/diagnose_solver_plateau.py
   ```
   Expected: Logit changes should NOT decrease monotonically

2. **Check step accuracy during training**:
   Look for steady improvement instead of plateau

3. **Compare warmup3 vs warmup4**:
   warmup4 should show better per-step improvement

---

## Related Files Modified
- `sci_arc/models/rlan_modules/recursive_solver.py`
  - Added `use_entropy_refinement=True` parameter
  - Added `entropy_proj` and `entropy_gate` modules
  - Modified forward loop to inject entropy each step
  - Changed residual from fixed 0.3 to progressive decay

## Commits
- `20823fe`: DSC centroid diversity loss fix
- (This session): Entropy-guided refinement + progressive residual decay
