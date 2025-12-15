"""
SCI-ARC Training Components.

Provides:
- Loss functions (SCL, orthogonality, stablemax)
- Trainer with curriculum learning
- EMA (Exponential Moving Average) helper
"""

from .losses import (
    StructuralContrastiveLoss,
    OrthogonalityLoss,
    SCIARCLoss,
    stablemax_cross_entropy,
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
    # Trainer
    'SCIARCTrainer',
    'TrainingConfig',
    'train_sci_arc',
    # EMA
    'EMAHelper',
    'EMAWrapper',
]
