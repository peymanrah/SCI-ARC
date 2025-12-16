"""
CICL (Color-Invariant Consistency Learning) - Backward Compatibility Module.

This module has been renamed to CISL (Content-Invariant Structure Learning).
All classes and functions are re-exported from cisl_loss.py for backward compatibility.

The new name better reflects the general-purpose nature of the approach:
- CISL works for any content-structure separation, not just colors
- Applicable to: ARC (colors), NLP (entities), Vision (objects), Graphs (nodes)

Import from here for backward compatibility, or use the new names directly:
    
    # Backward compatible (old names):
    from sci_arc.training.cicl_loss import CICLLoss, ColorInvarianceLoss
    
    # New preferred names:
    from sci_arc.training.cisl_loss import CISLLoss, ContentInvarianceLoss
"""

# Re-export everything from cisl_loss for backward compatibility
from .cisl_loss import (
    # Main loss class (CICLLoss is alias for CISLLoss)
    CISLLoss,
    CISLLoss as CICLLoss,
    
    # Component losses
    WithinTaskConsistencyLoss,
    ContentInvarianceLoss,
    ContentInvarianceLoss as ColorInvarianceLoss,  # Old name
    BatchVarianceLoss,
    
    # Utility functions
    apply_content_permutation,
    apply_content_permutation as apply_color_permutation,  # Old name
    apply_content_permutation_batch,
    apply_content_permutation_batch as apply_color_permutation_batch,  # Old name
)

__all__ = [
    # New names (preferred)
    'CISLLoss',
    'ContentInvarianceLoss',
    'apply_content_permutation',
    'apply_content_permutation_batch',
    
    # Old names (backward compat)
    'CICLLoss',
    'ColorInvarianceLoss',
    'apply_color_permutation',
    'apply_color_permutation_batch',
    
    # Unchanged names
    'WithinTaskConsistencyLoss',
    'BatchVarianceLoss',
]
