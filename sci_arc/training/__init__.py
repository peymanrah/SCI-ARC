"""
RLAN Training Components.

Provides:
- RLAN loss functions (Focal, Entropy, Sparsity, Predicate Diversity, Curriculum)
- Trainer with curriculum learning
- EMA (Exponential Moving Average) helper

For legacy SCI-ARC/CISL loss functions, see others/training/
"""

from .rlan_loss import (
    RLANLoss,
    FocalLoss,
    EntropyRegularization,
    SparsityRegularization,
    PredicateDiversityLoss,
    CurriculumPenalty,
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
from .loss_logger import (
    LossLogger,
    LossStats,
)
from .hyperlora_training import (
    HyperLoRATrainer,
    HyperLoRATrainingConfig,
)

__all__ = [
    # RLAN Loss
    'RLANLoss',
    'FocalLoss',
    'EntropyRegularization',
    'SparsityRegularization',
    'PredicateDiversityLoss',
    'CurriculumPenalty',
    # Trainer
    'SCIARCTrainer',
    'TrainingConfig',
    'train_sci_arc',
    # EMA
    'EMAHelper',
    'EMAWrapper',
    # Loss Logger
    'LossLogger',
    'LossStats',
    # HyperLoRA Training
    'HyperLoRATrainer',
    'HyperLoRATrainingConfig',
]
