"""
Comprehensive Evaluation Metrics for ARC.

Provides all the metrics used in CISL/SCI-ARC evaluation:
- Pixel accuracy
- Task accuracy (exact match)
- Size accuracy (output size correct)
- Color accuracy (colors used match)
- Non-background accuracy (excluding color 0)
- IoU per color
- Mean IoU
- Partial match score
- Levenshtein distance
"""

from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import numpy as np


def pixel_accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Compute pixel-wise accuracy between prediction and target.
    
    Args:
        pred: Predicted grid, shape (H, W)
        target: Target grid, shape (H, W)
        
    Returns:
        Accuracy in [0, 1]
    """
    if pred.shape != target.shape:
        # Size mismatch - compare minimum overlap
        min_h = min(pred.shape[0], target.shape[0])
        min_w = min(pred.shape[1], target.shape[1])
        pred = pred[:min_h, :min_w]
        target = target[:min_h, :min_w]
        
    if pred.size == 0:
        return 1.0 if target.size == 0 else 0.0
        
    return (pred == target).mean()


def task_accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Compute task-level accuracy (exact match).
    
    Args:
        pred: Predicted grid
        target: Target grid
        
    Returns:
        1.0 if exact match, 0.0 otherwise
    """
    if pred.shape != target.shape:
        return 0.0
    return 1.0 if np.array_equal(pred, target) else 0.0


def size_accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Check if prediction has correct output size.
    
    Args:
        pred: Predicted grid
        target: Target grid
        
    Returns:
        1.0 if sizes match, 0.0 otherwise
    """
    return 1.0 if pred.shape == target.shape else 0.0


def color_accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Compute how well the prediction uses the same colors as target.
    
    Returns:
        Jaccard similarity of color sets
    """
    pred_colors = set(pred.flatten())
    target_colors = set(target.flatten())
    
    if len(pred_colors) == 0 and len(target_colors) == 0:
        return 1.0
    
    intersection = pred_colors & target_colors
    union = pred_colors | target_colors
    
    if len(union) == 0:
        return 1.0
        
    return len(intersection) / len(union)


def non_background_accuracy(
    pred: np.ndarray,
    target: np.ndarray,
    background_color: int = 0
) -> float:
    """
    Compute accuracy only for non-background pixels.
    
    This is critical for ARC since background dominates many grids.
    
    Args:
        pred: Predicted grid
        target: Target grid
        background_color: Color to exclude (default: 0)
        
    Returns:
        Accuracy for non-background pixels, or 1.0 if no non-background pixels
    """
    if pred.shape != target.shape:
        min_h = min(pred.shape[0], target.shape[0])
        min_w = min(pred.shape[1], target.shape[1])
        pred = pred[:min_h, :min_w]
        target = target[:min_h, :min_w]
    
    mask = target != background_color
    
    if mask.sum() == 0:
        # No non-background pixels in target
        # Check if prediction also has no non-background pixels
        pred_mask = pred != background_color
        return 1.0 if pred_mask.sum() == 0 else 0.0
        
    return (pred[mask] == target[mask]).mean()


def iou_per_color(
    pred: np.ndarray,
    target: np.ndarray,
    num_colors: int = 10
) -> Dict[int, float]:
    """
    Compute Intersection over Union for each color.
    
    Args:
        pred: Predicted grid
        target: Target grid
        num_colors: Maximum number of colors
        
    Returns:
        Dict mapping color -> IoU score
    """
    if pred.shape != target.shape:
        min_h = min(pred.shape[0], target.shape[0])
        min_w = min(pred.shape[1], target.shape[1])
        pred = pred[:min_h, :min_w]
        target = target[:min_h, :min_w]
    
    iou_scores = {}
    
    # Get all colors present in either grid
    all_colors = set(pred.flatten()) | set(target.flatten())
    
    for color in range(num_colors):
        if color not in all_colors:
            continue
            
        pred_mask = (pred == color)
        target_mask = (target == color)
        
        intersection = (pred_mask & target_mask).sum()
        union = (pred_mask | target_mask).sum()
        
        if union == 0:
            continue  # Color not present in either
            
        iou_scores[color] = intersection / union
    
    return iou_scores


def mean_iou(
    pred: np.ndarray,
    target: np.ndarray,
    exclude_background: bool = True,
    num_colors: int = 10
) -> float:
    """
    Compute mean Intersection over Union across all colors.
    
    Args:
        pred: Predicted grid
        target: Target grid
        exclude_background: Whether to exclude color 0
        num_colors: Maximum number of colors
        
    Returns:
        Mean IoU score
    """
    iou_scores = iou_per_color(pred, target, num_colors)
    
    if exclude_background and 0 in iou_scores:
        del iou_scores[0]
    
    if not iou_scores:
        return 1.0  # Empty grids match
    
    return sum(iou_scores.values()) / len(iou_scores)


def levenshtein_distance(pred: np.ndarray, target: np.ndarray) -> int:
    """
    Compute Levenshtein distance between flattened grids.
    
    Useful for measuring how close a prediction is.
    """
    pred_flat = pred.flatten().tolist()
    target_flat = target.flatten().tolist()
    
    m, n = len(pred_flat), len(target_flat)
    
    # Use dynamic programming
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_flat[i-1] == target_flat[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    return dp[m][n]


def normalized_edit_distance(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Compute normalized edit distance.
    
    Returns:
        Score in [0, 1] where 1 is perfect match
    """
    dist = levenshtein_distance(pred, target)
    max_len = max(pred.size, target.size)
    
    if max_len == 0:
        return 1.0
    
    return 1.0 - (dist / max_len)


def partial_match_score(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    """
    Compute a comprehensive partial match score.
    
    Returns multiple metrics to understand partial correctness:
    - pixel_accuracy: Overall pixel accuracy
    - non_background_accuracy: Accuracy on non-background pixels
    - size_match: Whether sizes match
    - color_jaccard: Jaccard similarity of color sets
    - mean_iou: Mean IoU across colors
    - normalized_edit: Normalized edit distance
    
    Returns:
        Dict with all partial match metrics
    """
    return {
        'pixel_accuracy': pixel_accuracy(pred, target),
        'non_background_accuracy': non_background_accuracy(pred, target),
        'size_match': size_accuracy(pred, target),
        'color_jaccard': color_accuracy(pred, target),
        'mean_iou': mean_iou(pred, target),
        'normalized_edit': normalized_edit_distance(pred, target),
    }


class ARCMetrics:
    """
    Accumulator for ARC evaluation metrics.
    
    Tracks:
    - Overall accuracy
    - Per-task metrics
    - Difficulty stratification
    - All partial match scores
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.total_tasks = 0
        self.correct_tasks = 0
        self.total_pixels = 0
        self.correct_pixels = 0
        
        self.size_matches = 0
        self.partial_scores = []
        
        # Aggregate metrics
        self.sum_pixel_accuracy = 0.0
        self.sum_non_background_accuracy = 0.0
        self.sum_color_jaccard = 0.0
        self.sum_mean_iou = 0.0
        
        self.per_task_results = {}
        self.by_difficulty = defaultdict(lambda: {'total': 0, 'correct': 0})
    
    def update(
        self,
        task_id: str,
        pred: np.ndarray,
        target: np.ndarray,
        difficulty: Optional[str] = None,
    ):
        """
        Update metrics with a single prediction.
        
        Args:
            task_id: Task identifier
            pred: Predicted grid
            target: Ground truth grid
            difficulty: Optional difficulty label
        """
        self.total_tasks += 1
        
        # Compute all metrics
        is_correct = np.array_equal(pred, target) if pred.shape == target.shape else False
        
        if is_correct:
            self.correct_tasks += 1
        
        # Pixel accuracy
        pix_acc = pixel_accuracy(pred, target)
        self.sum_pixel_accuracy += pix_acc
        
        # Pixel counts
        if pred.shape == target.shape:
            correct_pix = (pred == target).sum()
            self.correct_pixels += correct_pix
            self.total_pixels += target.size
        else:
            self.total_pixels += target.size
        
        # Size match
        if pred.shape == target.shape:
            self.size_matches += 1
        
        # Partial match metrics
        partial = partial_match_score(pred, target)
        self.partial_scores.append(partial)
        
        self.sum_non_background_accuracy += partial['non_background_accuracy']
        self.sum_color_jaccard += partial['color_jaccard']
        self.sum_mean_iou += partial['mean_iou']
        
        # Store per-task results
        self.per_task_results[task_id] = {
            'is_correct': is_correct,
            'pixel_accuracy': pix_acc,
            'size_match': pred.shape == target.shape,
            'pred_shape': pred.shape,
            'target_shape': target.shape,
            **partial,
        }
        
        # Track by difficulty
        if difficulty:
            self.by_difficulty[difficulty]['total'] += 1
            if is_correct:
                self.by_difficulty[difficulty]['correct'] += 1
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics."""
        n = max(self.total_tasks, 1)
        
        return {
            'total_tasks': self.total_tasks,
            'correct_tasks': self.correct_tasks,
            'task_accuracy': self.correct_tasks / n,
            'pixel_accuracy': self.sum_pixel_accuracy / n,
            'size_accuracy': self.size_matches / n,
            'non_background_accuracy': self.sum_non_background_accuracy / n,
            'color_accuracy': self.sum_color_jaccard / n,
            'mean_iou': self.sum_mean_iou / n,
            'total_pixels': self.total_pixels,
            'correct_pixels': self.correct_pixels,
            'overall_pixel_accuracy': self.correct_pixels / max(self.total_pixels, 1),
        }
    
    def get_detailed_results(self) -> Dict[str, Any]:
        """Get all detailed per-task results."""
        summary = self.get_summary()
        summary['per_task'] = self.per_task_results
        summary['by_difficulty'] = dict(self.by_difficulty)
        return summary
    
    def print_summary(self):
        """Print formatted summary."""
        summary = self.get_summary()
        
        print("=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        print(f"Total Tasks: {summary['total_tasks']}")
        print(f"Correct Tasks: {summary['correct_tasks']}")
        print("-" * 40)
        print(f"Task Accuracy:           {summary['task_accuracy']:.4f} ({summary['task_accuracy']*100:.2f}%)")
        print(f"Pixel Accuracy:          {summary['pixel_accuracy']:.4f} ({summary['pixel_accuracy']*100:.2f}%)")
        print(f"Size Accuracy:           {summary['size_accuracy']:.4f} ({summary['size_accuracy']*100:.2f}%)")
        print(f"Non-Background Accuracy: {summary['non_background_accuracy']:.4f} ({summary['non_background_accuracy']*100:.2f}%)")
        print(f"Color Accuracy:          {summary['color_accuracy']:.4f} ({summary['color_accuracy']*100:.2f}%)")
        print(f"Mean IoU:                {summary['mean_iou']:.4f} ({summary['mean_iou']*100:.2f}%)")
        print("=" * 60)
        
        # Print by difficulty if available
        if self.by_difficulty:
            print("\nBy Difficulty:")
            for diff, stats in sorted(self.by_difficulty.items()):
                acc = stats['correct'] / max(stats['total'], 1)
                print(f"  {diff}: {stats['correct']}/{stats['total']} ({acc*100:.1f}%)")
