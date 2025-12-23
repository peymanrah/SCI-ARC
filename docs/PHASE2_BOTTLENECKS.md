# Phase 2 Analysis: Bottlenecks and Solutions

## Critical Issue: Color Permutation Failure

### Problem
Training log shows:
```
Color Permutation: 0.0% (0/400)
```

**Config says:** `color_permutation_prob: 1.0` (100%)  
**Reality:** 0% samples permuted

### Consequences
1. **Color Mode Collapse:** 51% of FG predictions are single color
2. **Uneven Per-Color Accuracy:**
   - Color 0 (BG): 96%
   - Colors 1-5: 52-79%
   - Colors 6-8: 42-50% (WEAK)
3. **Overfitting to Color Identity:** Model learns "color 1 → rule X" instead of "transformation regardless of color"

### Root Cause
The config `color_permutation: true` is correctly set, but:
- Training script correctly reads it from `dataset.augmentation.color_permutation`
- ARCDataset receives `color_permutation=True, color_permutation_prob=1.0`
- **BUT** 0% samples are being permuted in practice

This suggests a bug in the ARCDataset color permutation logic OR the random seed is preventing it.

### TRM Approach
**TRM uses 100% color permutation:**
- Forces color-agnostic learning
- Model must learn transformation rules independent of color
- Prevents memorizing "color 1 means X"

**From TRM code:**
```python
# Color permutation on EVERY sample
color_perm = np.random.permutation(range(1, 10))  # Shuffle 1-9, keep 0 fixed
grid = apply_color_perm(grid, color_perm)
```

### Solution
1. **Verify color permutation is actually running:**
   - Add debug prints in ARCDataset.__getitem__
   - Check if `random.random() < self.color_permutation_prob` is triggering
   
2. **Force 100% permutation:**
   - Change config to `color_permutation_prob: 1.0` (already done)
   - Verify ARCDataset receives correct value
   
3. **Alternative: Pre-generate permuted cache:**
   - Like TRM, generate 1000x augmented samples
   - Each sample has random color permutation baked in

---

## Question 1: Does Solver Need Direct Context Access?

### Current Architecture
```
Support Set → ContextEncoder → (B, N, D, H, W) spatial features
                                        ↓
Test Features ← CrossAttentionInjector ← support_features
        ↓
    Features (context-modulated)
        ↓
    DSC → MSRE → clue_features (B, K, D, H, W)
        ↓
    RecursiveSolver → predictions
```

### What Solver Currently Receives
1. **clue_features:** `(B, K, D, H, W)` - These are **already context-modulated**
   - Test features were modulated by CrossAttentionInjector
   - DSC sees context-aware features
   - MSRE computes relative coords on context-aware features
   
2. **count_embedding:** From LCR (not context-dependent)
3. **predicates:** From SPH (not context-dependent)
4. **attention_maps:** From DSC (on context-aware features)

### TRM Architecture
```
Support Set → Embed → (B, N, L, D)
                           ↓
Test Features ← Self-Attention (includes support) ← support_features
        ↓
    Transformer Decoder with cross-attention to support
        ↓
    predictions
```

**Key difference:** TRM's decoder has **explicit cross-attention** to support set at every layer.

### Analysis: Is This a Bottleneck?

**YES - Potential bottleneck:**

The CrossAttentionInjector modulates features **once** at the beginning, but:
1. **DSC computes clues on modulated features** - This might contaminate clue discovery
2. **Solver never sees raw support features** - Can't query "which support pixels match this output region?"
3. **Information flow is one-way** - Context → Features, but Solver can't look back at context

**TRM's advantage:**
- Decoder can **re-attend** to support set at each layer
- Can ask different questions of support set as it refines predictions
- Direct gradient flow from output → support features

### Solution: Phase 2.5 - Solver-Level Context Injection

Add **support_features** parameter to RecursiveSolver:

```python
class RecursiveSolver(nn.Module):
    def __init__(self, ..., use_support_attention=False):
        self.use_support_attention = use_support_attention
        if use_support_attention:
            self.support_attn = CrossAttentionInjector(hidden_dim, num_heads=4)
    
    def forward(
        self,
        clue_features,
        support_features=None,  # NEW: (B, N, D, H, W)
        ...
    ):
        # In refinement loop:
        for t in range(num_steps):
            h = self.gru(features, h)
            
            # NEW: Re-attend to support features at each step
            if self.use_support_attention and support_features is not None:
                h = self.support_attn(h, support_features)
            
            logits = self.output_head(h)
```

**Benefits:**
1. Solver can query support set during refinement
2. Different questions at each step (early: "what regions?", late: "what colors?")
3. Matches TRM's cross-attention decoder

**Risks:**
1. More parameters (another CrossAttentionInjector)
2. Slower training
3. May not help if CrossAttention at feature level is sufficient

---

## Question 2: Color Permutation Probability

### Your Observation
> "per color accuracy isnt uniform in the early epochs, is this because we use .5 for color permutation, should we increase that to 1"

**Answer: YES!**

### Current Config
```yaml
color_permutation_prob: 1.0  # Should be 100%
```

But training log shows:
```
Color Permutation: 0.0% (0/400)
```

**This is the bug!** Config says 100%, reality is 0%.

### What TRM Does
**TRM: 100% color permutation**
```python
# From TRM data pipeline
for sample in dataset:
    # ALWAYS permute colors
    color_perm = np.random.permutation(range(1, 10))
    sample = apply_perm(sample, color_perm)
```

### Why This Matters

**With 0% color permutation:**
- Model sees "red square → blue circle" 1000 times
- Learns: "Red means square, blue means circle"
- **FAILS** on eval: "blue square → red circle" (never seen blue squares!)

**With 100% color permutation:**
- Model sees "red square → blue circle"
- Next sample: "green square → yellow circle" (colors shuffled)
- Learns: "Shape X → Shape Y regardless of color"
- **SUCCEEDS** on eval: generalizes to any color combination

### Per-Color Accuracy Analysis

From your log:
```
Color:       0     1     2     3     4     5     6     7     8     9
Acc%:      96%   55%   52%   55%   74%   79%   50%   42%   49%   70%
Target:  53.4%  7.6%  6.2%  8.3%  8.2%  4.5%  2.6%  1.6%  5.9%  1.7%
```

**Why uneven?**
1. **Color 0 (BG): 96%** - Easy, it's 53% of pixels
2. **Colors 6,7,8: 42-50%** - RARE in training (1-3% of pixels)
   - Without color permutation, model never learns these
   - With permutation, all colors become equally common

**Solution:** Fix color permutation bug, all colors will become ~10% each (permuted distribution is uniform).

---

## Recommended Actions

### Immediate (Fix Color Permutation)
1. **Debug ARCDataset color permutation:**
   ```python
   # Add to dataset.py line 287
   if self.color_permutation:
       print(f"[DEBUG] Applying color perm with prob {self.color_permutation_prob}")
       if random.random() < self.color_permutation_prob:
           print(f"[DEBUG] Color perm APPLIED!")
   ```

2. **Verify config parsing:**
   - Check that `dataset.augmentation.color_permutation: true` is being read
   - Print value in train_rlan.py after loading config

3. **Quick fix: Force permutation:**
   ```python
   # In train_rlan.py, override config
   color_permutation = True
   color_permutation_prob = 1.0  # FORCE 100%
   ```

### Short-term (Phase 2.5)
1. **Run 20 epochs with color permutation working**
   - Should see uniform per-color accuracy
   - Should see better generalization to eval set

2. **Add solver-level context injection:**
   - Pass support_features to RecursiveSolver
   - Add CrossAttentionInjector in refinement loop
   - Test if eval EM improves

### Long-term (Phase 3)
1. **Decouple DSC from context:**
   - Run DSC on ORIGINAL features (before context injection)
   - Clue discovery should be universal, not task-specific
   - Inject context AFTER DSC, only for solver

2. **Scale up:**
   - Increase batch size
   - More epochs
   - Full 400 training tasks (not just 50)

---

## Expected Improvements

### With Color Permutation Fixed
- **Per-color accuracy:** [96, 55, 52...] → [75, 75, 75...] (uniform)
- **Color mode collapse:** 51% → <20% (no single color dominance)
- **Eval EM:** 0% → 5-10% (actually learns abstract rules)

### With Solver Context Injection
- **Eval EM:** 5-10% → 15-25% (can query support during refinement)
- **Train-eval gap:** Smaller (better generalization)

### With Both Fixes
- **Eval EM:** 15-25% → **30-45%** (approaching TRM's 45%)
- **Generalization:** Strong (learns color-agnostic transformations)

---

## Summary

**Primary Bottleneck:** Color permutation not working (0% instead of 100%)  
**Secondary Bottleneck:** Solver lacks direct support access (one-way context flow)  
**Solution:** Fix color perm first, then add solver context in Phase 2.5
