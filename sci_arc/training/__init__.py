"""
SCI-ARC Training Components.

Provides:
- Loss functions (SCL, orthogonality, stablemax)
- CISL: Content-Invariant Structure Learning (generalized from CICL)
- CICL: Color-Invariant Consistency Learning (backward compatibility alias)
- Trainer with curriculum learning
- EMA (Exponential Moving Average) helper
"""

from .losses import (
    StructuralContrastiveLoss,
    OrthogonalityLoss,
    SCIARCLoss,
    stablemax_cross_entropy,
)
from .cisl_loss import (
    CISLLoss,
    CICLLoss,  # Backward compatibility alias
    WithinTaskConsistencyLoss,
    ContentInvarianceLoss,
    ColorInvarianceLoss,  # Backward compatibility alias
    BatchVarianceLoss,
    apply_content_permutation_batch,
    apply_color_permutation_batch,  # Backward compatibility alias
)
from .trainer import (
    SCIARCTrainer,
    TrainingConfig,
    train_sci_arc,
)
from .ema import (
    EMAHelper,
    EMAWrapper,
)

__all__ = [
    # Losses
    'StructuralContrastiveLoss',
    'OrthogonalityLoss',
    'SCIARCLoss',
    'stablemax_cross_entropy',
    # CISL (Content-Invariant Structure Learning)
    'CISLLoss',
    'CICLLoss',  # Backward compatibility
    'WithinTaskConsistencyLoss',
    'ContentInvarianceLoss',
    'ColorInvarianceLoss',  # Backward compatibility
    'BatchVarianceLoss',
    'apply_content_permutation_batch',
    'apply_color_permutation_batch',  # Backward compatibility
    # Trainer
    'SCIARCTrainer',
    'TrainingConfig',
    'train_sci_arc',
    # EMA
    'EMAHelper',
    'EMAWrapper',
]
