"""
Evaluation metrics for ARC tasks.

Implements:
1. Pixel-level accuracy
2. Task-level accuracy (exact match)
3. Partial match metrics
4. IoU for object-level evaluation
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


def pixel_accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Compute pixel-wise accuracy.
    
    Args:
        pred: Predicted grid [H, W]
        target: Ground truth grid [H, W]
    
    Returns:
        Accuracy in [0, 1]
    """
    if pred.shape != target.shape:
        # Handle size mismatch
        min_h = min(pred.shape[0], target.shape[0])
        min_w = min(pred.shape[1], target.shape[1])
        pred = pred[:min_h, :min_w]
        target = target[:min_h, :min_w]
    
    return (pred == target).mean()


def task_accuracy(pred: np.ndarray, target: np.ndarray) -> bool:
    """
    Check if prediction exactly matches target.
    
    Args:
        pred: Predicted grid [H, W]
        target: Ground truth grid [H, W]
    
    Returns:
        True if exact match
    """
    if pred.shape != target.shape:
        return False
    return np.array_equal(pred, target)


def size_accuracy(pred: np.ndarray, target: np.ndarray) -> bool:
    """Check if predicted size matches target."""
    return pred.shape == target.shape


def color_accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Compute color set accuracy.
    
    Checks if the same colors appear in both grids.
    """
    pred_colors = set(pred.flatten())
    target_colors = set(target.flatten())
    
    if len(target_colors) == 0:
        return 1.0 if len(pred_colors) == 0 else 0.0
    
    intersection = pred_colors & target_colors
    union = pred_colors | target_colors
    
    return len(intersection) / len(union)


def partial_match_score(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    """
    Compute multiple partial match metrics.
    
    Returns:
        Dict with:
        - pixel_accuracy: fraction of correct pixels
        - size_match: 1 if same size, 0 otherwise
        - color_accuracy: IoU of color sets
        - non_background_accuracy: accuracy excluding background (0)
    """
    metrics = {}
    
    # Size match
    metrics['size_match'] = 1.0 if pred.shape == target.shape else 0.0
    
    # Pixel accuracy (with resizing if needed)
    metrics['pixel_accuracy'] = pixel_accuracy(pred, target)
    
    # Color accuracy
    metrics['color_accuracy'] = color_accuracy(pred, target)
    
    # Non-background accuracy
    if pred.shape == target.shape:
        non_bg_mask = target != 0
        if non_bg_mask.any():
            metrics['non_background_accuracy'] = (pred[non_bg_mask] == target[non_bg_mask]).mean()
        else:
            metrics['non_background_accuracy'] = 1.0 if not (pred != 0).any() else 0.0
    else:
        metrics['non_background_accuracy'] = 0.0
    
    return metrics


def iou_per_color(pred: np.ndarray, target: np.ndarray) -> Dict[int, float]:
    """
    Compute IoU for each color.
    
    Returns:
        Dict mapping color -> IoU score
    """
    if pred.shape != target.shape:
        return {}
    
    all_colors = set(pred.flatten()) | set(target.flatten())
    iou_scores = {}
    
    for color in all_colors:
        pred_mask = (pred == color)
        target_mask = (target == color)
        
        intersection = (pred_mask & target_mask).sum()
        union = (pred_mask | target_mask).sum()
        
        if union > 0:
            iou_scores[color] = intersection / union
        else:
            iou_scores[color] = 1.0
    
    return iou_scores


def mean_iou(pred: np.ndarray, target: np.ndarray, exclude_background: bool = True) -> float:
    """
    Compute mean IoU across all colors.
    
    Args:
        pred: Predicted grid
        target: Target grid
        exclude_background: If True, exclude color 0 from mean
    
    Returns:
        Mean IoU score
    """
    iou_scores = iou_per_color(pred, target)
    
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


class ARCMetrics:
    """
    Accumulator for ARC evaluation metrics.
    
    Tracks:
    - Overall accuracy
    - Per-task metrics
    - Difficulty stratification
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
        
        # Task accuracy
        is_correct = task_accuracy(pred, target)
        if is_correct:
            self.correct_tasks += 1
        
        # Size match
        if size_accuracy(pred, target):
            self.size_matches += 1
            
            # Pixel accuracy (only if size matches)
            self.total_pixels += target.size
            self.correct_pixels += (pred == target).sum()
        
        # Partial match metrics
        partial = partial_match_score(pred, target)
        self.partial_scores.append(partial)
        
        # Per-task results
        self.per_task_results[task_id] = {
            'correct': is_correct,
            'pixel_accuracy': partial['pixel_accuracy'],
            'size_match': partial['size_match'],
            'pred_shape': pred.shape,
            'target_shape': target.shape,
        }
        
        # By difficulty
        if difficulty:
            self.by_difficulty[difficulty]['total'] += 1
            if is_correct:
                self.by_difficulty[difficulty]['correct'] += 1
    
    def compute(self) -> Dict[str, float]:
        """
        Compute final metrics.
        
        Returns:
            Dict with all metrics
        """
        metrics = {}
        
        # Task accuracy
        metrics['task_accuracy'] = (
            self.correct_tasks / self.total_tasks if self.total_tasks > 0 else 0.0
        )
        
        # Pixel accuracy
        metrics['pixel_accuracy'] = (
            self.correct_pixels / self.total_pixels if self.total_pixels > 0 else 0.0
        )
        
        # Size accuracy
        metrics['size_accuracy'] = (
            self.size_matches / self.total_tasks if self.total_tasks > 0 else 0.0
        )
        
        # Average partial scores
        if self.partial_scores:
            for key in self.partial_scores[0].keys():
                values = [p[key] for p in self.partial_scores]
                metrics[f'avg_{key}'] = sum(values) / len(values)
        
        # By difficulty
        for diff, stats in self.by_difficulty.items():
            if stats['total'] > 0:
                metrics[f'accuracy_{diff}'] = stats['correct'] / stats['total']
        
        metrics['total_tasks'] = self.total_tasks
        metrics['correct_tasks'] = self.correct_tasks
        
        return metrics
    
    def summary(self) -> str:
        """Generate summary string."""
        metrics = self.compute()
        
        lines = [
            "=" * 50,
            "ARC Evaluation Results",
            "=" * 50,
            f"Task Accuracy: {metrics['task_accuracy']:.2%} ({self.correct_tasks}/{self.total_tasks})",
            f"Pixel Accuracy: {metrics['pixel_accuracy']:.2%}",
            f"Size Accuracy: {metrics['size_accuracy']:.2%}",
        ]
        
        if 'avg_non_background_accuracy' in metrics:
            lines.append(f"Non-BG Accuracy: {metrics['avg_non_background_accuracy']:.2%}")
        
        # Difficulty breakdown
        if self.by_difficulty:
            lines.append("\nBy Difficulty:")
            for diff in sorted(self.by_difficulty.keys()):
                stats = self.by_difficulty[diff]
                acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
                lines.append(f"  {diff}: {acc:.2%} ({stats['correct']}/{stats['total']})")
        
        lines.append("=" * 50)
        
        return "\n".join(lines)


def compute_arc_metrics(
    predictions: Dict[str, np.ndarray],
    targets: Dict[str, np.ndarray],
    difficulty_map: Optional[Dict[str, str]] = None,
) -> Dict[str, float]:
    """
    Compute metrics for a batch of predictions.
    
    Args:
        predictions: Dict mapping task_id -> predicted grid
        targets: Dict mapping task_id -> target grid
        difficulty_map: Optional dict mapping task_id -> difficulty
    
    Returns:
        Dict of metrics
    """
    metrics = ARCMetrics()
    
    for task_id, pred in predictions.items():
        if task_id not in targets:
            continue
        
        target = targets[task_id]
        difficulty = difficulty_map.get(task_id) if difficulty_map else None
        
        metrics.update(task_id, pred, target, difficulty)
    
    return metrics.compute()
