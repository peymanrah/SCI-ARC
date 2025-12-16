# Deep Analysis: 1D Rasterization vs 2D Grid-Based for SCI-ARC

## Executive Summary

Based on empirical testing and theoretical analysis:

| Criterion | 1D Rasterization | 2D Grid-Based | Winner |
|-----------|-----------------|---------------|--------|
| **Theoretical SCI Alignment** | ✅ Direct transfer from SCAN | ⚠️ Requires adaptation | 1D |
| **Spatial Structure** | ⚠️ Must be learned | ✅ Native | 2D |
| **Representation Collapse Risk** | ⚠️ Higher (saw 0.9999 similarity) | ✅ Lower | 2D |
| **Transform Clustering** | ⚠️ 0.0000 separation | ✅ 0.1421 separation | 2D |
| **Implementation Simplicity** | ✅ Standard 1D Transformer | ⚠️ 2D attention | 1D |
| **ARC Benchmark Fit** | ⚠️ Loses 2D structure | ✅ Native 2D | 2D |

**RECOMMENDATION: 2D Grid-Based is better for ARC**

---

## Test Results

### Quick Test (Simple Data)
```
1D Rasterization:
  Same-class similarity: 1.0000
  Diff-class similarity: 0.1218
  SEPARATION: 0.8782

2D Grid-Based:
  Same-class similarity: 1.0000
  Diff-class similarity: 0.5726
  SEPARATION: 0.4274

Winner: 1D (but on trivial data)
```

### Comprehensive Test (Diverse Patterns + 7 Transforms)
```
1D Rasterization:
  Same-class similarity: 0.9999
  Diff-class similarity: 0.9999
  SEPARATION: 0.0000  ← COLLAPSED!

2D Grid-Based:
  Same-class similarity: 0.8491
  Diff-class similarity: 0.7070
  SEPARATION: 0.1421

Winner: 2D (1D collapsed)
```

---

## Why 1D Collapsed on Complex Data

The 1D model produced identical embeddings for ALL inputs (`similarity = 0.9999`).
This is **representation collapse** - the model found a trivial solution.

### Root Cause Analysis

1. **Sequence length explosion**: 5×5 grid × 2 (input+output) = 50 tokens
   - Transformer attention is O(n²) = 2500 operations
   - Easy to learn "mean" of all tokens (collapses to constant)

2. **Lost spatial locality**: In 1D, adjacent 2D cells are distant
   ```
   2D: [1,2]    →  1D: [1,2,3,4]
       [3,4]        Cell 1 and 3 are adjacent in 2D but 2 positions apart in 1D
   ```

3. **Positional encoding limitations**: Row/col embeddings add 2D info but
   the model can still "shortcut" by averaging

### Why 2D Didn't Collapse

1. **Explicit difference computation**: `diff = proj(concat(input, output))`
   - Forces model to compute transformation, not just memorize
   
2. **2D position encoding**: Native spatial relationships preserved
   - Rotation = permutation of 2D positions (learnable pattern)

---

## The Real Problem: Positive Pair Definition

**CRITICAL INSIGHT**: Neither 1D nor 2D is the core issue.

The fundamental problem with current SCI-ARC is:

### Current Approach (Broken)
```python
# Group by dihedral augmentation (0-7)
transform_family = augment_idx  # 0=original, 1=rot90, 2=rot180, ...

# Positive pairs: Same input rotated differently
# This teaches: "Detect which rotation was applied to THIS grid"
```

### What SCI Actually Requires
```python
# Positive pairs: DIFFERENT inputs with SAME transformation rule
# Example: 
#   Task A: Input1 → rotate_90 → Output1
#   Task B: Input2 → rotate_90 → Output2
# 
# S(Task A) ≈ S(Task B) because both use "rotate_90" rule
# This teaches: "Detect the RULE, not the input"
```

### The ARC Challenge

ARC doesn't provide ground-truth transformation labels. We must INFER them.

Options:
1. **Task-based grouping**: Assume all demos in a task share the same rule
   - `transform_family = hash(task_id)`
   - Problem: Need multiple tasks with same rule for positives

2. **Rule inference**: Analyze (input, output) to detect rule type
   - Compute: rotation angle, flip axis, color mapping
   - Group tasks by inferred rule
   - Challenge: Some rules are complex/composite

3. **Contrastive without labels**: Use instance discrimination
   - Each task is its own class
   - Learn general transformation features
   - Rely on downstream task to cluster

---

## Recommendation for ARC-AGI

### Architecture
**Use 2D Grid-Based** because:
- ARC is inherently 2D spatial reasoning
- Native rotation/flip understanding
- Lower representation collapse risk
- Already implemented in current SCI-ARC

### SCL Improvement
**Fix positive pair definition**:

```python
# Current (wrong):
transform_family = augment_idx  # Groups by rotation of SAME input

# Better (task-based):
transform_family = hash(task_id)  # Groups all demos of SAME task

# Best (rule-based):
transform_family = infer_rule(input, output)  # Groups by detected rule
```

### Implementation Priority

1. **Immediate**: Change `scl_family_mode = "task"` (already implemented)
2. **Next**: Add rule inference module
3. **Later**: Consider 1D for specific compositional tasks

---

## When to Use 1D Rasterization

1D might be better for:
- **Compositional language-like tasks**: If ARC had text-like structure
- **Very small grids**: Where 2D overhead isn't worth it
- **Hybrid approach**: CNN for spatial features → 1D for composition

For standard ARC-AGI benchmark, **stick with 2D**.

---

## Files Created

1. `quick_1d_vs_2d_test.py` - Simple comparison (100 samples)
2. `comprehensive_1d_vs_2d_test.py` - Full comparison (200 samples, 7 transforms)
3. `test_1d_vs_2d_structure.py` - Detailed analysis with plotting

Run with:
```powershell
.\.venv\Scripts\python.exe comprehensive_1d_vs_2d_test.py
```
