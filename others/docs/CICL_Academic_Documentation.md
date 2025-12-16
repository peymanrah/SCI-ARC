# CICL: Color-Invariant Consistency Learning

> **⚠️ DEPRECATED: This document has been renamed to CISL (Content-Invariant Structure Learning).**
>
> Please see [CISL_Academic_Documentation.md](./CISL_Academic_Documentation.md) for the updated documentation.
>
> The CICL name is preserved for backward compatibility, but the preferred terminology is now CISL
> to reflect the general-purpose nature of content-invariant structure learning.

---

## Quick Redirect

The content of this document has been moved. For the latest documentation, see:

**[CISL_Academic_Documentation.md](./CISL_Academic_Documentation.md)**

### Name Change Summary

| Old Term | New Term |
|----------|----------|
| CICL (Color-Invariant Consistency Learning) | CISL (Content-Invariant Structure Learning) |
| `CICLLoss` class | `CISLLoss` class (CICLLoss is an alias) |
| `color_inv_weight` parameter | `content_inv_weight` parameter |
| `z_struct_color_aug` argument | `z_struct_content_aug` argument |
| `L_color_inv` loss component | `L_content_inv` loss component |

### Why the Rename?

1. **Generalization:** "Content-Invariant" is more general than "Color-Invariant"
2. **Applicability:** The technique works for any content permutation, not just colors
3. **Clarity:** CISL better describes the goal of learning structure independent of content

### Backward Compatibility

All old names continue to work:

```python
# Both work identically
from sci_arc.training import CISLLoss  # New preferred name
from sci_arc.training import CICLLoss  # Old name (alias)
```

---

*For full documentation, see [CISL_Academic_Documentation.md](./CISL_Academic_Documentation.md)*
