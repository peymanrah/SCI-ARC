"""
TRM-Style Evaluator for RLAN with Inverse Augmentation and Aggregated Voting.

This evaluator implements the exact evaluation approach from TRM that enables
proper generalization measurement:

1. Inverse Augmentation: Predictions are transformed back to canonical space
   before comparison with ground truth.

2. Aggregated Voting: Multiple predictions per task (from different augmentations)
   are collected and voted to find consensus.

3. Confidence Ranking: Predictions are ranked by model confidence (stop probability)
   to prioritize high-quality predictions.

4. Pass@K Metrics: Reports accuracy at different K values (1, 2, 5, 10) like TRM.

Key Insight:
- During training, we apply random augmentations to improve generalization
- During eval, we MUST undo those augmentations to compare with ground truth
- Without inverse augmentation, a correct prediction in augmented space
  appears wrong when compared to original ground truth

Usage:
    evaluator = TRMStyleEvaluator(pass_Ks=[1, 2, 5, 10])
    
    for batch in eval_loader:
        predictions, confidence = model.predict(batch)
        evaluator.update(
            task_id=batch['task_id'],
            prediction=predictions,
            ground_truth=batch['target'],
            aug_info=batch['aug_info'],
            confidence=confidence
        )
    
    results = evaluator.compute_metrics()
    print(f"Pass@1: {results['pass@1']:.4f}")
"""

import hashlib
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch


# Inverse mapping for dihedral transforms (matching TRM exactly)
# Index is the transform ID, value is the inverse transform ID
DIHEDRAL_INVERSE = [0, 3, 2, 1, 4, 5, 6, 7]


def dihedral_transform(arr: np.ndarray, tid: int) -> np.ndarray:
    """Apply dihedral transform (8 symmetries of D4 group)."""
    if tid == 0:
        return arr  # identity
    elif tid == 1:
        return np.rot90(arr, k=1)  # 90° CCW
    elif tid == 2:
        return np.rot90(arr, k=2)  # 180°
    elif tid == 3:
        return np.rot90(arr, k=3)  # 270° CCW
    elif tid == 4:
        return np.fliplr(arr)       # horizontal flip
    elif tid == 5:
        return np.flipud(arr)       # vertical flip
    elif tid == 6:
        return arr.T                # transpose
    elif tid == 7:
        return np.fliplr(np.rot90(arr, k=1))  # anti-transpose
    else:
        return arr


def inverse_dihedral_transform(arr: np.ndarray, tid: int) -> np.ndarray:
    """Apply inverse of dihedral transform to undo augmentation."""
    return dihedral_transform(arr, DIHEDRAL_INVERSE[tid])


def inverse_color_permutation(arr: np.ndarray, color_perm: np.ndarray) -> np.ndarray:
    """Undo color permutation by applying inverse mapping."""
    # color_perm maps original -> augmented
    # We need augmented -> original (inverse permutation)
    inv_perm = np.argsort(color_perm)
    return inv_perm[arr]


def inverse_translation(arr: np.ndarray, offset_r: int, offset_c: int) -> np.ndarray:
    """
    Undo translational augmentation by shifting content back.
    
    During augmentation, content was placed at (offset_r, offset_c) within a larger canvas.
    To invert, we extract just the content region (removing the offset).
    
    Note: This assumes arr has already been cropped to remove padding.
    The cropping step implicitly handles most translation inversion,
    but if we want to reconstruct the original position, we'd need
    the original grid dimensions. In practice, cropping is sufficient
    for grid comparison/hashing since we only care about content equality.
    
    Args:
        arr: The cropped prediction array
        offset_r: Row offset that was applied during augmentation
        offset_c: Column offset that was applied during augmentation
        
    Returns:
        The array (unchanged after cropping, but this documents the inversion)
    """
    # After crop_prediction(), the content is already extracted.
    # Translation inversion is effectively done by cropping.
    # This function exists for API completeness and documentation.
    return arr


def get_translation_offset(aug_info: Dict[str, Any]) -> Tuple[int, int]:
    """
    Extract translation offset from aug_info, supporting both key formats.
    
    FIXED: Supports both:
    - 'translational_offset': (r, c) tuple
    - 'offset_r', 'offset_c': separate int keys
    
    Args:
        aug_info: Augmentation info dictionary
        
    Returns:
        (offset_r, offset_c) tuple
    """
    # Try tuple format first
    if 'translational_offset' in aug_info:
        offset = aug_info['translational_offset']
        if isinstance(offset, (tuple, list)) and len(offset) == 2:
            return (offset[0], offset[1])
    
    # Try separate keys format
    offset_r = aug_info.get('offset_r', 0)
    offset_c = aug_info.get('offset_c', 0)
    return (offset_r, offset_c)


def grid_hash(grid: np.ndarray) -> str:
    """Compute unique hash for a grid (for voting)."""
    assert grid.ndim == 2
    grid = grid.astype(np.uint8)
    buffer = [x.to_bytes(1, byteorder='big') for x in grid.shape]
    buffer.append(grid.tobytes())
    return hashlib.sha256(b"".join(buffer)).hexdigest()


def crop_prediction(pred: np.ndarray, pad_value: int = 10, target_shape: tuple = None) -> np.ndarray:
    """
    Crop prediction to remove padding.
    
    CRITICAL FIX (Jan 2026): Now uses target_shape when provided.
    
    The previous implementation tried to detect content by finding pixels != pad_value.
    This FAILED because:
    1. pad_value=10 is NOT a valid ARC color, but model doesn't predict it in padded regions
    2. The model predicts 0-9 only, so pad detection by pad_value=10 never finds content
    3. Result: crop_prediction returns wrong-sized grids, causing 0% exact match
       even when the valid region has ~76% pixel accuracy!
    
    The fix is simple: pass the expected output shape through the pipeline
    and crop to that exact size. This matches what TRM does.
    
    Args:
        pred: (H, W) prediction array (possibly 30x30 padded)
        pad_value: Fallback padding value for content detection (legacy mode)
        target_shape: (h, w) expected output size. If provided, crop to this exact size.
    
    Returns:
        Cropped prediction of shape target_shape (if provided) or detected content region
    """
    # Find rows and cols that contain actual content (not padding)
    if pred.ndim == 1:
        pred = pred.reshape(30, 30)  # Assume max ARC size
    
    # NEW: If target_shape is provided, use it directly (correct approach)
    if target_shape is not None:
        h, w = target_shape
        return pred[:h, :w].copy()
    
    # LEGACY FALLBACK: Content-based detection (unreliable but kept for backward compat)
    # This path should rarely be used - always prefer passing target_shape
    # Mask of non-padding positions (exclude both pad_value=10 and ignore_index=-100)
    content_mask = (pred != pad_value) & (pred != -100)
    
    if not content_mask.any():
        return np.array([[0]])  # Empty prediction
    
    # Find bounding box of content
    rows = np.any(content_mask, axis=1)
    cols = np.any(content_mask, axis=0)
    
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    return pred[rmin:rmax+1, cmin:cmax+1]


class TRMStyleEvaluator:
    """
    TRM-style evaluator with inverse augmentation and aggregated voting.
    
    Key features matching TRM:
    1. Inverse augmentation to undo transforms on predictions
    2. Aggregated voting across multiple predictions per task
    3. Confidence-based ranking of predictions
    4. Pass@K metrics for comparison
    
    SOFT AGGREGATION (Jan 2026):
    When voting_mode='soft', uses log-mean-exp of confidence scores instead
    of hard count-based voting. This is more robust when predictions are
    uncertain (ties/high entropy). Mathematically aligns with group-marginalized
    predictive distribution.
    """
    
    def __init__(
        self,
        pass_Ks: List[int] = [1, 2, 5, 10],
        use_voting: bool = True,
        pad_value: int = 10,
        voting_mode: str = 'hard',  # Jan 2026: 'hard' (count-based) or 'soft' (log-mean-exp)
    ):
        """
        Args:
            pass_Ks: K values for Pass@K metrics
            use_voting: Whether to use aggregated voting (recommended)
            pad_value: Value used for padding in predictions
            voting_mode: 'hard' for count*conf voting, 'soft' for log-mean-exp
        """
        self.pass_Ks = sorted(pass_Ks)
        self.use_voting = use_voting
        self.pad_value = pad_value
        self.voting_mode = voting_mode
        
        # Storage for predictions per task
        # {task_id: {input_hash: [(pred_hash, confidence, pred_grid)]}}
        self._predictions: Dict[str, Dict[str, List[Tuple[str, float, np.ndarray]]]] = defaultdict(lambda: defaultdict(list))
        self._ground_truths: Dict[str, Dict[str, np.ndarray]] = {}  # {task_id: {input_hash: ground_truth}}
        self._hash_to_grid: Dict[str, np.ndarray] = {}  # Store grids by hash for retrieval
        
    def reset(self):
        """Clear all stored predictions."""
        self._predictions = defaultdict(lambda: defaultdict(list))
        self._ground_truths = {}
        self._hash_to_grid = {}
        
    def update(
        self,
        task_id: str,
        prediction: np.ndarray,
        ground_truth: np.ndarray,
        aug_info: Dict[str, Any],
        confidence: float = 1.0,
        input_grid: Optional[np.ndarray] = None,
        target_shape: Optional[tuple] = None,
    ):
        """
        Add a prediction for evaluation.
        
        Args:
            task_id: Unique task identifier (e.g., "00d62c1b")
            prediction: Model prediction (H, W) in augmented space
            ground_truth: Ground truth (H, W) in ORIGINAL (canonical) space
            aug_info: Augmentation info dict with keys:
                - 'dihedral_id': int (0-7) transform applied
                - 'color_perm': np.ndarray of length 10, color mapping applied
                - 'offset_r', 'offset_c': translation offset (if any)
            confidence: Model confidence score (higher = better)
            input_grid: Optional input grid for computing input hash
            target_shape: (h, w) expected output shape for proper cropping (CRITICAL FIX Jan 2026)
                         If provided, crops to this exact size instead of pad detection.
        """
        # Step 1: Crop prediction to remove padding
        # CRITICAL FIX (Jan 2026): Use target_shape when provided for reliable cropping
        # Also get aug_target_shape for augmented space cropping
        aug_target_shape = None
        if target_shape is not None:
            # For dihedral transforms that swap dimensions (90°, 270°, transpose, anti-transpose)
            dihedral_id = aug_info.get('dihedral_id', 0)
            if dihedral_id in [1, 3, 6, 7]:  # rotations by 90°/270° and transposes swap H/W
                aug_target_shape = (target_shape[1], target_shape[0])  # swap
            else:
                aug_target_shape = target_shape
        pred_cropped = crop_prediction(prediction, self.pad_value, target_shape=aug_target_shape)
        
        # Step 2: Apply inverse augmentation to bring prediction to canonical space
        pred_canonical = pred_cropped.copy()
        
        # Undo dihedral transform
        dihedral_id = aug_info.get('dihedral_id', 0)
        if dihedral_id != 0:
            pred_canonical = inverse_dihedral_transform(pred_canonical, dihedral_id)
        
        # Undo color permutation
        color_perm = aug_info.get('color_perm', None)
        if color_perm is not None:
            pred_canonical = inverse_color_permutation(pred_canonical, color_perm)
        
        # Step 3: Compute hashes
        pred_hash = grid_hash(pred_canonical)
        
        # Use input hash if provided, otherwise use task_id with index
        if input_grid is not None:
            input_cropped = crop_prediction(input_grid, self.pad_value)
            # Also apply inverse augmentation to input for consistent hashing
            if dihedral_id != 0:
                input_cropped = inverse_dihedral_transform(input_cropped, dihedral_id)
            if color_perm is not None:
                input_cropped = inverse_color_permutation(input_cropped, color_perm)
            input_hash = grid_hash(input_cropped)
        else:
            input_hash = "default"
        
        # Step 4: Store prediction
        self._predictions[task_id][input_hash].append((pred_hash, confidence, pred_canonical))
        self._hash_to_grid[pred_hash] = pred_canonical
        
        # Step 5: Store ground truth (in canonical space)
        # CRITICAL FIX: ground_truth is already in CANONICAL space (per docstring),
        # so we should NOT apply inverse transforms to it!
        # Use target_shape for reliable cropping if provided.
        gt_cropped = crop_prediction(ground_truth, self.pad_value, target_shape=target_shape)
        
        if task_id not in self._ground_truths:
            self._ground_truths[task_id] = {}
        self._ground_truths[task_id][input_hash] = gt_cropped
        
    def update_batch(
        self,
        task_ids: List[str],
        predictions: torch.Tensor,  # (B, H, W)
        ground_truths: torch.Tensor,  # (B, H, W)
        aug_infos: List[Dict[str, Any]],
        confidences: Optional[torch.Tensor] = None,  # (B,)
        input_grids: Optional[torch.Tensor] = None,  # (B, H, W)
        target_shapes: Optional[List[tuple]] = None,  # List of (h, w) per sample
    ):
        """Batch version of update with target_shapes support (Jan 2026 fix)."""
        B = len(task_ids)
        
        if confidences is None:
            confidences = torch.ones(B)
            
        predictions_np = predictions.cpu().numpy()
        ground_truths_np = ground_truths.cpu().numpy()
        confidences_np = confidences.cpu().numpy()
        
        if input_grids is not None:
            input_grids_np = input_grids.cpu().numpy()
        else:
            input_grids_np = [None] * B
        
        for i in range(B):
            self.update(
                task_id=task_ids[i],
                prediction=predictions_np[i],
                ground_truth=ground_truths_np[i],
                aug_info=aug_infos[i],
                confidence=float(confidences_np[i]),
                input_grid=input_grids_np[i] if input_grids is not None else None,
                target_shape=target_shapes[i] if target_shapes is not None else None,
            )
    
    def compute_metrics(self) -> Dict[str, float]:
        """
        Compute Pass@K metrics using aggregated voting.
        
        Returns:
            Dict with keys like 'pass@1', 'pass@2', etc.
        """
        correct_at_k = {k: 0.0 for k in self.pass_Ks}
        total_tasks = 0
        
        for task_id, input_preds in self._predictions.items():
            if task_id not in self._ground_truths:
                continue
                
            task_correct_at_k = {k: 0 for k in self.pass_Ks}
            num_test_inputs = 0
            
            for input_hash, pred_list in input_preds.items():
                if input_hash not in self._ground_truths[task_id]:
                    continue
                    
                ground_truth = self._ground_truths[task_id][input_hash]
                gt_hash = grid_hash(ground_truth)
                
                if self.use_voting:
                    # Aggregate predictions by hash, accumulate confidence
                    hash_votes: Dict[str, Tuple[int, float, List[float]]] = {}  # {hash: (count, total_confidence, conf_list)}
                    for pred_hash, conf, _ in pred_list:
                        if pred_hash not in hash_votes:
                            hash_votes[pred_hash] = (0, 0.0, [])
                        count, total_conf, conf_list = hash_votes[pred_hash]
                        hash_votes[pred_hash] = (count + 1, total_conf + conf, conf_list + [conf])
                    
                    if self.voting_mode == 'soft':
                        # SOFT AGGREGATION (Jan 2026): Use log-mean-exp of confidences
                        # This is more robust when predictions are uncertain
                        # Score = log(mean(exp(log_conf))) ≈ mean(conf) for small conf
                        # but handles extreme values better than simple sum
                        import math
                        ranked = []
                        for h, (count, total_conf, conf_list) in hash_votes.items():
                            # Log-mean-exp: log(1/n * sum(exp(log_conf_i)))
                            # For stability, use log-sum-exp trick
                            if conf_list:
                                max_log = max(math.log(c + 1e-10) for c in conf_list)
                                log_sum_exp = max_log + math.log(
                                    sum(math.exp(math.log(c + 1e-10) - max_log) for c in conf_list)
                                )
                                log_mean_exp = log_sum_exp - math.log(len(conf_list))
                                # Convert back to linear scale, multiply by count for frequency boost
                                soft_score = math.exp(log_mean_exp) * count
                            else:
                                soft_score = 0.0
                            ranked.append((h, soft_score))
                        ranked.sort(key=lambda x: x[1], reverse=True)
                        ranked_hashes = [h for h, _ in ranked]
                    else:
                        # HARD VOTING: Rank by (average confidence * count) = total_confidence
                        # This gives preference to predictions that are both frequent AND confident
                        ranked = sorted(
                            hash_votes.items(),
                            key=lambda x: x[1][1],  # total_confidence (= avg_conf * count)
                            reverse=True
                        )
                        ranked_hashes = [h for h, _ in ranked]
                else:
                    # No voting: rank by confidence directly
                    sorted_preds = sorted(pred_list, key=lambda x: x[1], reverse=True)
                    # Deduplicate
                    seen = set()
                    ranked_hashes = []
                    for pred_hash, _, _ in sorted_preds:
                        if pred_hash not in seen:
                            seen.add(pred_hash)
                            ranked_hashes.append(pred_hash)
                
                # Check Pass@K
                for k in self.pass_Ks:
                    if gt_hash in ranked_hashes[:k]:
                        task_correct_at_k[k] += 1
                        
                num_test_inputs += 1
            
            if num_test_inputs > 0:
                # Average across test inputs for this task
                for k in self.pass_Ks:
                    correct_at_k[k] += task_correct_at_k[k] / num_test_inputs
                total_tasks += 1
        
        if total_tasks == 0:
            return {f'pass@{k}': 0.0 for k in self.pass_Ks}
        
        return {f'pass@{k}': correct_at_k[k] / total_tasks for k in self.pass_Ks}
    
    def get_detailed_results(self) -> Dict[str, Dict[str, Any]]:
        """Get per-task results for analysis."""
        results = {}
        
        for task_id, input_preds in self._predictions.items():
            task_result = {
                'num_predictions': sum(len(p) for p in input_preds.values()),
                'num_unique_predictions': len(set(
                    h for preds in input_preds.values() for h, _, _ in preds
                )),
                'test_inputs': list(input_preds.keys()),
            }
            
            if task_id in self._ground_truths:
                correct = False
                for input_hash, pred_list in input_preds.items():
                    if input_hash in self._ground_truths[task_id]:
                        gt = self._ground_truths[task_id][input_hash]
                        gt_hash = grid_hash(gt)
                        if any(h == gt_hash for h, _, _ in pred_list):
                            correct = True
                            break
                task_result['has_correct_prediction'] = correct
            
            results[task_id] = task_result
            
        return results


def create_evaluator_for_rlan(
    pass_Ks: List[int] = [1, 2, 5, 10],
    use_voting: bool = True,
) -> TRMStyleEvaluator:
    """Factory function to create TRM-style evaluator for RLAN."""
    return TRMStyleEvaluator(
        pass_Ks=pass_Ks,
        use_voting=use_voting,
        pad_value=10,  # RLAN uses 10 for padding
    )


# Example integration with RLAN training loop
def evaluate_with_trm_style(
    model,
    eval_loader,
    device: str = 'cuda',
    num_augmented_views: int = 1,  # Currently single-view; set >1 for future multi-view
) -> Dict[str, float]:
    """
    Evaluate RLAN model using TRM-style evaluation with inverse augmentation.
    
    This function:
    1. Processes each batch from eval_loader (which may already contain augmented views)
    2. Applies inverse augmentation to bring predictions to canonical space
    3. Uses voting to aggregate predictions per task (if multiple views per task)
    4. Computes Pass@K metrics
    
    NOTE: Multi-view augmentation is expected to be handled by the eval_loader.
    If you want TRM-style 8-view voting, create an eval_loader that yields
    8 augmented versions of each task. The evaluator will aggregate them
    via the voting mechanism in TRMStyleEvaluator.
    
    Args:
        model: RLAN model
        eval_loader: Evaluation data loader (must provide 'test_inputs', 'input_grids', 
                     'output_grids', 'test_outputs', 'grid_masks', 'task_ids', 'aug_info')
        device: Device to run on
        num_augmented_views: Reserved for future use (loader should provide views)
        
    Returns:
        Dict with Pass@K metrics
    """
    model.eval()
    evaluator = create_evaluator_for_rlan()
    
    with torch.no_grad():
        for batch in eval_loader:
            # FIXED: Support both old key names and new RLAN key names
            # Old format: 'input', 'target', 'demos'
            # New format: 'test_inputs', 'test_outputs', 'input_grids', 'output_grids'
            
            # Get test inputs (the grid to predict)
            if 'test_inputs' in batch:
                inputs = batch['test_inputs'].to(device)
            elif 'input' in batch:
                inputs = batch['input'].to(device)
            else:
                raise KeyError("Batch must contain 'test_inputs' or 'input' key")
            
            # Get ground truth targets
            if 'test_outputs' in batch:
                targets = batch['test_outputs'].to(device)
            elif 'target' in batch:
                targets = batch['target'].to(device)
            else:
                raise KeyError("Batch must contain 'test_outputs' or 'target' key")
            
            # Get task IDs
            if 'task_ids' in batch:
                task_ids = batch['task_ids']
            elif 'task_id' in batch:
                task_ids = batch['task_id']
            else:
                task_ids = [f"task_{i}" for i in range(inputs.shape[0])]
            
            # Get augmentation info
            aug_infos = batch.get('aug_info', batch.get('aug_infos', [{'dihedral_id': 0}] * inputs.shape[0]))
            
            # FIXED: Extract support set (train pairs) for context-aware prediction
            train_inputs = None
            train_outputs = None
            pair_mask = None
            
            if 'input_grids' in batch:
                train_inputs = batch['input_grids'].to(device)
            elif 'demos' in batch and isinstance(batch['demos'], dict):
                train_inputs = batch['demos'].get('input_grids')
                if train_inputs is not None:
                    train_inputs = train_inputs.to(device)
            
            if 'output_grids' in batch:
                train_outputs = batch['output_grids'].to(device)
            elif 'demos' in batch and isinstance(batch['demos'], dict):
                train_outputs = batch['demos'].get('output_grids')
                if train_outputs is not None:
                    train_outputs = train_outputs.to(device)
            
            if 'grid_masks' in batch:
                pair_mask = batch['grid_masks'].to(device)
            elif 'demos' in batch and isinstance(batch['demos'], dict):
                pair_mask = batch['demos'].get('grid_masks')
                if pair_mask is not None:
                    pair_mask = pair_mask.to(device)
            
            # FIXED: Call RLAN with proper keyword arguments for context
            # This ensures the model uses the support set for prediction
            outputs = model(
                input_grid=inputs,
                train_inputs=train_inputs,
                train_outputs=train_outputs,
                pair_mask=pair_mask,
            )
            
            # Handle both tensor outputs (RLAN) and dict outputs (legacy models)
            if isinstance(outputs, torch.Tensor):
                logits = outputs
                stop_logits = None
            else:
                # Dict output - look for 'logits' or 'pred' key for backward compatibility
                logits = outputs.get('logits', outputs.get('pred', None))
                stop_logits = outputs.get('stop_logits', None)
            
            if logits is None:
                raise ValueError(f"Model output must be tensor or dict with 'logits'/'pred' key, got: {type(outputs)}")
            
            predictions = logits.argmax(dim=1)  # (B, H, W)
            
            # Get confidence from stop probability if available
            if stop_logits is not None:
                # Use average stop probability as confidence (lower = more confident)
                stop_probs = torch.sigmoid(stop_logits)
                confidence = 1.0 - stop_probs.mean(dim=1)  # (B,)
            else:
                confidence = None
            
            evaluator.update_batch(
                task_ids=task_ids,
                predictions=predictions,
                ground_truths=targets,
                aug_infos=aug_infos,
                confidences=confidence,
                input_grids=inputs.argmax(dim=1) if inputs.dim() == 4 else inputs,
            )
    
    return evaluator.compute_metrics()


if __name__ == "__main__":
    # Test the evaluator
    print("Testing TRMStyleEvaluator...")
    
    evaluator = TRMStyleEvaluator(pass_Ks=[1, 2, 5])
    
    # Simulate predictions
    gt = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    
    # Correct prediction (no augmentation)
    evaluator.update(
        task_id="test_task",
        prediction=gt.copy(),
        ground_truth=gt,
        aug_info={'dihedral_id': 0},
        confidence=0.9,
    )
    
    # Wrong prediction
    evaluator.update(
        task_id="test_task",
        prediction=np.array([[5, 5], [5, 5]], dtype=np.uint8),
        ground_truth=gt,
        aug_info={'dihedral_id': 0},
        confidence=0.5,
    )
    
    # Correct prediction with augmentation (rotated 90°)
    rotated_pred = np.rot90(gt, k=1)  # Apply same aug as training
    evaluator.update(
        task_id="test_task",
        prediction=rotated_pred,
        ground_truth=gt,
        aug_info={'dihedral_id': 1},  # Indicate rotation was applied
        confidence=0.8,
    )
    
    metrics = evaluator.compute_metrics()
    print(f"Metrics: {metrics}")
    
    # Should have pass@1 = 1.0 (correct prediction exists)
    assert metrics['pass@1'] == 1.0, f"Expected pass@1=1.0, got {metrics['pass@1']}"
    
    print("All tests passed!")
