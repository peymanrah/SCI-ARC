# SCI-ARC Implementation Fixes Summary

## Date: Session Continuation

This document summarizes the critical fixes implemented to improve RLAN training stability and ARC learning capability.

---

## Fix 1: Deterministic Eval Mode (P0 - Critical)

### Problem
`gumbel_softmax_2d()` and `gumbel_sigmoid()` always added stochastic Gumbel noise, even during evaluation. This caused:
- Random predictions at inference time
- Non-reproducible evaluation metrics
- Validation accuracy fluctuating randomly

### Solution

**File: `sci_arc/models/rlan_modules/dynamic_saliency_controller.py`**
- Added `deterministic: bool = False` parameter to `gumbel_softmax_2d()`
- When `deterministic=True`, skip Gumbel noise and use regular temperature-scaled softmax
- DSC `forward()` now passes `deterministic=not self.training`

**File: `sci_arc/models/rlan_modules/symbolic_predicate_heads.py`**
- Added `deterministic: bool = False` parameter to `gumbel_sigmoid()`
- When `deterministic=True`, use regular sigmoid without noise
- SPH `forward()` and `forward_with_logits()` now pass `deterministic=not self.training`

### Result
- Training: Gumbel noise enables exploration
- Evaluation: Deterministic predictions for reproducible metrics

---

## Fix 2: Explicit Padding Token (P0 - Critical)

### Problem
Input grids were padded with value 0 (black), making padding indistinguishable from actual black pixels. This caused:
- DSC anchoring on padding regions
- Confusion about grid boundaries
- Loss of spatial structure understanding

### Solution

**File: `sci_arc/data/dataset.py`**
- Added `PAD_COLOR = 10` constant for input padding
- Updated `_pad_grid()` to use `PAD_COLOR` for input grids (not targets)
- Targets still use `-100` (PADDING_IGNORE_VALUE) for loss masking

**File: `sci_arc/models/grid_encoder.py`**
- Added `PAD_COLOR = 10` class constant
- Changed from `num_colors` to `num_embeddings = num_colors + 1` for embedding table
- Embedding table now has 11 entries: colors 0-9 plus padding token 10
- Config still passes `num_colors: 10` (backward compatible)
- Added `get_valid_mask(grid)` helper method: returns boolean mask where padding=False
- Added `get_grid_sizes(grid)` helper method: computes actual (H, W) for each sample

### Result
- Model can distinguish padding from black pixels
- Valid mask available for downstream modules
- No config changes required (backward compatible)

---

## Fix 3: Grid Sizes Passed to MSRE (P1 - Important)

### Problem
MSRE never received actual grid sizes - always used full padded dimensions. This broke:
- Scale-invariant relative position encoding
- The paper's claim of size-independent spatial reasoning
- Small grids being treated same as large grids

### Solution

**File: `sci_arc/models/rlan.py`**
- In `forward()`, compute `valid_mask` and `grid_sizes` from input grid
- Pass `valid_mask` to DSC: `self.dsc(features, temperature, mask=valid_mask)`
- Pass `grid_sizes` to MSRE: `self.msre(features, centroids, grid_sizes=grid_sizes)`

### Result
- MSRE normalizes coordinates by actual grid size (not padded size)
- DSC attention is masked to valid regions only
- True scale-invariant processing as claimed in paper

---

## Fix 4: DSC Recurrence (P2 - Paper Alignment)

### Problem
**Paper claims**: "Spatial UNet conditioned on previous hidden state H_{t-1}. Iterative selection depends on hidden state."
**Code did**: Per-clue learned query vectors with no recurrence between clues.

This meant clue k's discovery was independent of clues 0..k-1, missing the iterative refinement the paper described.

### Solution

**File: `sci_arc/models/rlan_modules/dynamic_saliency_controller.py`**
- Added `self.query_gru = nn.GRUCell(hidden_dim, hidden_dim)` for recurrence
- Initialize `query_state = zeros(B, D)` before clue loop
- Each clue's query is modulated by recurrent state: `query = query + query_state`
- After computing `attended_features`, update state: `query_state = self.query_gru(attended_features, query_state)`

### Result
- Clue k's discovery now depends on what clues 0..k-1 attended to
- Implements the paper's "iterative selection depends on H_{t-1}"
- Enables true sequential clue discovery for complex multi-anchor tasks

---

## Fix 5: MSRE Log-Polar Encoding (P2 - Paper Alignment)

### Problem
**Paper claims**: `r = log(sqrt(dx² + dy²) + 1)` and normalize by `max(H,W)`
**Code did**: Linear radius normalized by diagonal, separate H/W normalization

Log-radius is specifically helpful for multi-scale generalization and "rings/dilation/expansion" behaviors.

### Solution

**File: `sci_arc/models/rlan_modules/multi_scale_relative_encoding.py`**
- Changed radius computation from linear to log-polar: `log_radius = torch.log(euclidean_dist + 1)`
- Changed normalization from separate H/W to `max(H,W)` for consistent aspect ratio handling
- Fixed angle computation: `atan2(abs_col, abs_row)` (col=x, row=y in image coords)
- Normalized log_radius by `log(max_possible_dist + 1)`

### Result
- Log-polar encoding provides better multi-scale generalization
- Consistent normalization across different aspect ratios
- Better "ring" and "expansion" pattern recognition

---

## Fix 6: LCR Per-Clue Counting (P2 - Paper Alignment)

### Problem
**Paper claims**: `c_t = sum_{i,j} M_t(i,j) * OneHot(X_{i,j})` (per-clue, attention-weighted)
**Code did**: Global counting ignoring attention_maps entirely

For tasks like "fill region with majority color inside boundary", global counts are wrong - need region-conditioned counts.

### Solution

**File: `sci_arc/models/rlan_modules/latent_counting_registers.py`**
- Added `forward_per_clue(grid, attention_maps)` method implementing paper formula
- For each clue k, compute: `weighted_counts = (color_onehot * attn_k).sum(dim=(-2,-1))`
- Returns `(B, K, hidden_dim)` per-clue embeddings instead of global
- Modified `forward()` to dispatch to per-clue counting when `attention_maps` provided

**File: `sci_arc/models/rlan.py`**
- Pass `attention_maps` to LCR when enabled: `self.lcr(input_grid, features, mask=valid_mask, attention_maps=attention_maps)`

**File: `sci_arc/models/rlan_modules/recursive_solver.py`**
- Updated `_inject_counts()` to handle both global (B, num_colors, D) and per-clue (B, K, D) count embeddings
- Modified `forward()` to inject per-clue counts into `clue_features` BEFORE aggregation
- This ensures each clue k's aggregated features include clue k's local color statistics

### Result
- Each clue gets its own color statistics from its attended region
- Enables region-conditioned counting for complex tasks
- Backward compatible (global counting when attention_maps=None)
- Per-clue counts are injected BEFORE clue aggregation (paper design)

---

## Summary of Changed Files

| File | Changes |
|------|---------|
| `dynamic_saliency_controller.py` | Deterministic mode + GRU recurrence |
| `symbolic_predicate_heads.py` | Deterministic mode for eval |
| `multi_scale_relative_encoding.py` | Log-polar encoding + max(H,W) normalization |
| `latent_counting_registers.py` | Per-clue attention-weighted counting |
| `dataset.py` | PAD_COLOR=10 for input padding |
| `grid_encoder.py` | 11 embeddings + helper methods |
| `rlan.py` | Compute and pass valid_mask, grid_sizes, attention_maps |

---

## Testing the Fixes

Run training with minimal config:
```bash
python train_rlan.py --config configs/rlan_minimal.yaml
```

Expected improvements:
1. **Stable validation metrics** - No more random fluctuations due to deterministic eval
2. **Better spatial learning** - DSC won't waste clues on padding
3. **Scale invariance** - MSRE properly normalizes by actual grid size
4. **Sequential clue discovery** - DSC now builds clues iteratively
5. **Multi-scale patterns** - Log-polar encoding helps with rings/expansions
6. **Region-aware counting** - LCR counts within attended regions (when enabled)

---

## Paper ↔ Code Alignment Summary

| Module | Paper Claims | Code Before | Code After |
|--------|-------------|-------------|------------|
| DSC | UNet conditioned on H_{t-1} | Independent queries | GRU recurrence |
| DSC | Gumbel-softmax | Stochastic always | Deterministic at eval |
| MSRE | log(dist + 1) | Linear radius | Log-polar |
| MSRE | max(H,W) normalization | Separate H/W | max(H,W) |
| LCR | Per-clue attention-weighted | Global counting | Per-clue weighted |
| All | Padding handling | 0 (black) confusion | PAD_COLOR=10 |
