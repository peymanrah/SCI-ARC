"""
RLAN Training Components.

Provides:
- RLAN loss functions (Focal, Entropy, Sparsity, Predicate Diversity, Curriculum)
- Trainer with curriculum learning (LEGACY - see note below)
- EMA (Exponential Moving Average) helper

⚠️  IMPORTANT: PRODUCTION TRAINING PATH
========================================
For production training, DO NOT use SCIARCTrainer directly.
Instead, use the standalone training script:

    python scripts/train_rlan.py --config configs/rlan_stable.yaml

Why? SCIARCTrainer uses hardcoded TrainingConfig defaults and does NOT
read YAML config for LOO training, equivariance, or loss_mode.
The train_rlan.py script properly reads ALL config from YAML.

SCIARCTrainer is exported here for backward compatibility with unit tests.
========================================

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
