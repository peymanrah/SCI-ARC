# RLAN vs TRM: Attention/Convolution/Loss Comparison

## Executive Summary

| Aspect | TRM | RLAN | Same/Different |
|--------|-----|------|----------------|
| **Grid Padding** | Pads to 30×30, uses PAD=0, EOS=1, colors=2-11 | Pads to 30×30, uses colors=0-9, PAD=10 | Similar (both 30×30) |
| **Attention Masking** | NO masking for padding | NO masking for padding | ✅ Same |
| **Compute Full Grid** | YES - all 900 tokens | YES - all 900 tokens | ✅ Same |
| **Loss Masking** | IGNORE_LABEL_ID=-100 | -100 for padding positions | ✅ Same |
| **Convolution** | None (pure transformer) | 3×3 Conv2d with padding=1 | Different architecture |

## Detailed Analysis

### 1. Grid Representation

**TRM:**
```python
# From dataset/build_arc_dataset.py lines 49-72
# PAD: 0, <eos>: 1, digits: 2 ... 11
# All grids padded to ARCMaxGridSize = 30

grid = np.pad(grid + 2, (...), constant_values=0)  # Shift colors by 2

# Add <eos> marker at grid boundaries
eos_row, eos_col = pad_r + nrow, pad_c + ncol
if eos_row < ARCMaxGridSize:
    grid[eos_row, pad_c:eos_col] = 1  # EOS token
```

**RLAN:**
```python
# From sci_arc/data/dataset.py
PAD_COLOR = 10  # Padding token
# Colors 0-9 are actual ARC colors, 10 is padding

# Padding happens in collate_fn
grid = F.pad(grid, padding, value=PAD_COLOR)
```

**Key Insight:** Both approaches pad to 30×30, but with different token schemes. TRM shifts all colors by +2 to reserve 0 for padding and 1 for EOS. RLAN keeps colors 0-9 unchanged and uses 10 for padding.

### 2. Attention Computation

**TRM (from models/layers.py lines 99-137):**
```python
class Attention(nn.Module):
    def forward(self, cos_sin, hidden_states):
        batch_size, seq_len, _ = hidden_states.shape  # seq_len = 900

        qkv = self.qkv_proj(hidden_states)
        
        # NO ATTENTION MASK!
        # Computes attention over ALL 900 tokens including padding
        attn_output = scaled_dot_product_attention(
            query=query, key=key, value=value,
            is_causal=self.causal  # False for ARC (bidirectional)
        )
        return self.o_proj(attn_output)
```

**RLAN DSC (from sci_arc/models/rlan_modules/dynamic_saliency_controller.py lines 307-352):**
```python
def forward(self, features, temperature, mask):
    B, D, H, W = features.shape  # H=W=30
    
    # Flatten: (B, D, H, W) -> (B, H*W, D) = (B, 900, D)
    features_flat = features.permute(0, 2, 3, 1).reshape(B, H * W, D)
    
    # Compute attention over ALL 900 tokens
    attn_scores = torch.einsum('bd,bnd->bn', q, k_proj) / self.scale
    attn_scores = attn_scores.view(B, H, W)
    
    # mask is ONLY for progressive clue discovery (already-attended regions)
    # NOT for padding - padding is included in attention computation!
    if mask is not None:
        attn_scores = attn_scores + torch.log(mask)
    
    attention = gumbel_softmax_2d(attn_scores, temperature)
```

**Conclusion:** Both TRM and RLAN compute attention over the FULL 30×30 grid (900 tokens). Neither masks out padding from attention computation. The mask in RLAN's DSC is for progressive masking (preventing re-attending to same location), not for padding.

### 3. Convolution Handling

**TRM:** Pure transformer, no convolutions.

**RLAN (from sci_arc/models/rlan_modules/recursive_solver.py):**
```python
# ConvGRU uses 3x3 convolutions with padding=1
self.reset_gate = nn.Conv2d(
    hidden_dim * 2, hidden_dim,
    kernel_size=kernel_size, padding=padding  # padding=1
)

# Convolutions operate on FULL 30×30 grid
# No masking for padding regions
```

**Key Insight:** RLAN's convolutions operate on the full 30×30 padded grid. The padding=1 in Conv2d is for spatial padding (to maintain size), not for ignoring PAD_COLOR pixels.

### 4. Loss Computation

**TRM (from models/losses.py):**
```python
IGNORE_LABEL_ID = -100  # Ignored in cross-entropy

# During dataset building, padding positions get IGNORE_LABEL_ID
labels[padding_positions] = IGNORE_LABEL_ID
```

**RLAN (from sci_arc/data/dataset.py and sci_arc/training/rlan_loss.py):**
```python
# In dataset.py collate_fn:
targets_padded[padding] = -100  # Ignore in loss

# In rlan_loss.py:
valid_mask = targets_flat != -100  # Only compute loss on valid positions
loss = stablemax_cross_entropy(logits[valid_mask], targets[valid_mask])
```

**Conclusion:** Both use exactly the same approach: `-100` for padding positions, which gets ignored during loss computation.

### 5. Why Computing on Full 30×30 is CORRECT

**Theoretical Justification:**

1. **Spatial Context:** ARC puzzles have spatial relationships that may extend to grid boundaries. Even "empty" padding provides context about where the actual content ends.

2. **Positional Encoding:** Both TRM (RoPE) and RLAN (coordinate embeddings) encode absolute positions. If we mask padding in attention, the model loses information about grid boundaries.

3. **EOS Marker (TRM):** TRM explicitly marks grid boundaries with EOS tokens, making padding semantically meaningful - it tells the model "this is outside the grid."

4. **Padding Embedding (RLAN):** RLAN has a dedicated embedding for PAD_COLOR=10, meaning the model learns what padding looks like and can use it for boundary detection.

5. **Loss Masking is Sufficient:** We don't need to mask attention because:
   - The model learns that padding positions don't affect the answer
   - The loss already ignores padding positions
   - The gradient only flows through valid predictions

### 6. Potential Optimization: Attention Masking for Padding

While both TRM and RLAN work without masking, there's a potential optimization:

**Pros of NOT masking:**
- Simpler implementation
- Model can learn grid boundary patterns
- Padding embedding provides semantic information

**Pros of masking padding:**
- Reduced attention entropy (sharper attention to content)
- Potentially faster convergence
- More focused gradient flow

**Recommendation:** Given TRM's proven success without attention masking, RLAN should continue the current approach. The padding embedding (color 10) already provides semantic differentiation.

### 7. Architecture Comparison Summary

| Component | TRM | RLAN |
|-----------|-----|------|
| **Base Architecture** | Pure transformer | CNN + attention hybrid |
| **Position Encoding** | RoPE | Coordinate embeddings |
| **Attention Type** | Multi-head self-attention | Query-key attention (DSC) |
| **Recursion** | L_cycles × H_cycles | num_iterations |
| **State Update** | Residual + RMSNorm | ConvGRU |
| **Loss Function** | Stablemax CE | Stablemax CE ✓ |
| **Grid Size Handling** | Fixed 900 tokens | Fixed 30×30 spatial |

### 8. Key Takeaways

1. **RLAN's approach is aligned with TRM:** Both compute on full 30×30, mask only at loss level.

2. **No changes needed for padding handling:** The current implementation is correct.

3. **The difference is architecture, not masking:** TRM is pure transformer, RLAN is CNN-attention hybrid.

4. **Convolutions on full grid are fine:** The padding embedding learns to represent "outside grid" which convolutions can use for edge detection.

5. **Focus on other improvements:** Since masking strategy is correct, optimization efforts should focus elsewhere (e.g., clue discovery, recursive iteration count).
