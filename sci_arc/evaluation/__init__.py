"""
SCI-ARC Evaluation Module.

Provides comprehensive evaluation tools for ARC models including:
- Pixel, task, size, color accuracy metrics
- Non-background and IoU metrics
- ARCMetrics accumulator class
- Visualization utilities
"""

from .metrics import (
    pixel_accuracy,
    task_accuracy,
    size_accuracy,
    color_accuracy,
    non_background_accuracy,
    iou_per_color,
    mean_iou,
    partial_match_score,
    levenshtein_distance,
    normalized_edit_distance,
    ARCMetrics,
)

from .visualization import (
    visualize_grid,
    visualize_prediction,
    grid_to_image,
    save_grid_comparison,
    ARC_COLORS,
    ARC_COLORS_HEX,
)

__all__ = [
    # Metrics
    'pixel_accuracy',
    'task_accuracy',
    'size_accuracy',
    'color_accuracy',
    'non_background_accuracy',
    'iou_per_color',
    'mean_iou',
    'partial_match_score',
    'levenshtein_distance',
    'normalized_edit_distance',
    'ARCMetrics',
    # Visualization
    'visualize_grid',
    'visualize_prediction',
    'grid_to_image',
    'save_grid_comparison',
    'ARC_COLORS',
    'ARC_COLORS_HEX',
]
