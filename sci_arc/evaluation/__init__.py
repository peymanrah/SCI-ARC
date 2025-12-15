"""
SCI-ARC Evaluation Components.

Provides:
- ARCEvaluator: Full evaluation pipeline
- Metrics: pixel/task accuracy, IoU, etc.
- Visualization utilities
- Augmentation voting (from TRM)
"""

from .evaluator import (
    ARCEvaluator,
    EvaluationConfig,
    visualize_prediction,
    generate_submission,
)
from .metrics import (
    ARCMetrics,
    compute_arc_metrics,
    pixel_accuracy,
    task_accuracy,
    size_accuracy,
    color_accuracy,
    partial_match_score,
    mean_iou,
    iou_per_color,
    normalized_edit_distance,
)
from .voting import (
    AugmentationVoter,
    dihedral_transform_torch,
    inverse_dihedral_transform_torch,
    vote_predictions,
    pass_at_k,
)

__all__ = [
    # Evaluator
    'ARCEvaluator',
    'EvaluationConfig',
    'visualize_prediction',
    'generate_submission',
    # Metrics
    'ARCMetrics',
    'compute_arc_metrics',
    'pixel_accuracy',
    'task_accuracy',
    'size_accuracy',
    'color_accuracy',
    'partial_match_score',
    'mean_iou',
    'iou_per_color',
    'normalized_edit_distance',
    # Voting (TRM-style)
    'AugmentationVoter',
    'dihedral_transform_torch',
    'inverse_dihedral_transform_torch',
    'vote_predictions',
    'pass_at_k',
]
