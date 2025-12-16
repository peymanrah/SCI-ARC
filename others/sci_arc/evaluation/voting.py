"""
Augmentation Voting for SCI-ARC Evaluation.

From TRM's evaluators/arc.py - uses test-time augmentation with voting
for more robust predictions.

Strategy:
1. Apply each of 8 dihedral transforms to test input
2. Run model on each transformed input
3. Apply inverse transform to each prediction
4. Vote (mode) across all predictions for final answer

This significantly improves accuracy on ARC by leveraging
symmetry properties of most transformations.
"""

from typing import List, Tuple, Optional, Dict, Any
from collections import Counter

import numpy as np
import torch
import torch.nn as nn


# Inverse mapping for each dihedral transform
DIHEDRAL_INVERSE = [0, 3, 2, 1, 4, 5, 6, 7]


def dihedral_transform_torch(tensor: torch.Tensor, tid: int) -> torch.Tensor:
    """
    Apply dihedral transform to a 2D tensor.
    
    Args:
        tensor: [H, W] or [B, H, W] tensor
        tid: Transform ID (0-7)
    
    Returns:
        Transformed tensor
    """
    if tid == 0:
        return tensor.clone()
    elif tid == 1:
        return torch.rot90(tensor, k=1, dims=(-2, -1))
    elif tid == 2:
        return torch.rot90(tensor, k=2, dims=(-2, -1))
    elif tid == 3:
        return torch.rot90(tensor, k=3, dims=(-2, -1))
    elif tid == 4:
        return torch.flip(tensor, dims=[-1])  # horizontal flip
    elif tid == 5:
        return torch.flip(tensor, dims=[-2])  # vertical flip
    elif tid == 6:
        # Transpose
        if tensor.dim() == 2:
            return tensor.T.contiguous()
        else:
            return tensor.transpose(-2, -1).contiguous()
    elif tid == 7:
        # Anti-transpose
        return torch.flip(torch.rot90(tensor, k=1, dims=(-2, -1)), dims=[-1])
    else:
        return tensor.clone()


def inverse_dihedral_transform_torch(tensor: torch.Tensor, tid: int) -> torch.Tensor:
    """Apply inverse of dihedral transform."""
    return dihedral_transform_torch(tensor, DIHEDRAL_INVERSE[tid])


def dihedral_transform_np(arr: np.ndarray, tid: int) -> np.ndarray:
    """
    Apply dihedral transform to a 2D numpy array.
    
    Args:
        arr: [H, W] numpy array
        tid: Transform ID (0-7)
    
    Returns:
        Transformed array
    """
    if tid == 0:
        return arr.copy()
    elif tid == 1:
        return np.rot90(arr, k=1)
    elif tid == 2:
        return np.rot90(arr, k=2)
    elif tid == 3:
        return np.rot90(arr, k=3)
    elif tid == 4:
        return np.fliplr(arr)
    elif tid == 5:
        return np.flipud(arr)
    elif tid == 6:
        return arr.T.copy()
    elif tid == 7:
        return np.fliplr(np.rot90(arr, k=1))
    else:
        return arr.copy()


def inverse_dihedral_transform_np(arr: np.ndarray, tid: int) -> np.ndarray:
    """Apply inverse of dihedral transform."""
    return dihedral_transform_np(arr, DIHEDRAL_INVERSE[tid])


def vote_predictions(predictions: List[np.ndarray]) -> np.ndarray:
    """
    Vote across multiple predictions to get final answer.
    
    Uses mode (most common value) at each pixel position.
    
    Args:
        predictions: List of [H, W] arrays with same shape
    
    Returns:
        Voted prediction [H, W]
    """
    if len(predictions) == 1:
        return predictions[0]
    
    # Stack predictions
    stacked = np.stack(predictions, axis=0)  # [N, H, W]
    
    # Mode at each position
    result = np.zeros_like(predictions[0])
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            values = stacked[:, i, j]
            counter = Counter(values)
            result[i, j] = counter.most_common(1)[0][0]
    
    return result


def vote_predictions_torch(predictions: torch.Tensor) -> torch.Tensor:
    """
    Vote across multiple predictions (PyTorch version).
    
    Args:
        predictions: [N, H, W] tensor
    
    Returns:
        Voted prediction [H, W]
    """
    # Mode along first dimension
    result, _ = torch.mode(predictions, dim=0)
    return result


class AugmentationVoter:
    """
    Evaluator that uses test-time augmentation with voting.
    
    This matches TRM's evaluation strategy for improved accuracy.
    """
    
    def __init__(
        self,
        model: nn.Module,
        num_dihedral: int = 8,
        use_color_perms: bool = False,
        num_color_perms: int = 3,
        device: str = 'cuda'
    ):
        """
        Args:
            model: SCI-ARC model
            num_dihedral: Number of dihedral transforms to use (1-8)
            use_color_perms: Whether to also use color permutations
            num_color_perms: Number of color permutations per dihedral
            device: Device to run on
        """
        self.model = model
        self.num_dihedral = min(8, max(1, num_dihedral))
        self.use_color_perms = use_color_perms
        self.num_color_perms = num_color_perms
        self.device = device
    
    def predict_with_voting(
        self,
        demo_inputs: torch.Tensor,      # [N, H, W]
        demo_outputs: torch.Tensor,     # [N, H, W]
        test_input: torch.Tensor,       # [H, W]
        demo_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Make prediction using augmentation voting.
        
        Args:
            demo_inputs: Demonstration input grids [N, H, W]
            demo_outputs: Demonstration output grids [N, H, W]
            test_input: Test input grid [H, W]
            demo_mask: Optional mask for valid demos [N]
        
        Returns:
            (voted_prediction, all_predictions)
        """
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for tid in range(self.num_dihedral):
                # Transform inputs
                aug_demos_in = dihedral_transform_torch(demo_inputs, tid)
                aug_demos_out = dihedral_transform_torch(demo_outputs, tid)
                aug_test = dihedral_transform_torch(test_input, tid)
                
                # Add batch dimension
                aug_demos_in = aug_demos_in.unsqueeze(0).to(self.device)  # [1, N, H, W]
                aug_demos_out = aug_demos_out.unsqueeze(0).to(self.device)
                aug_test = aug_test.unsqueeze(0).to(self.device)  # [1, H, W]
                # Create dummy test_output for shape inference
                test_output_dummy = torch.zeros_like(aug_test)
                
                # Run model - use forward_training which accepts batched format
                output = self.model.forward_training(
                    input_grids=aug_demos_in,
                    output_grids=aug_demos_out,
                    test_input=aug_test,
                    test_output=test_output_dummy,
                    grid_mask=demo_mask.unsqueeze(0) if demo_mask is not None else None
                )
                
                # Get prediction (argmax over colors)
                if hasattr(output, 'final_prediction'):
                    pred = output.final_prediction[0]  # [H, W]
                elif isinstance(output, dict) and 'logits' in output:
                    pred = output['logits'][0].argmax(dim=-1)  # [H, W]
                else:
                    pred = output[0].argmax(dim=-1)
                
                # Apply inverse transform
                pred_np = pred.cpu().numpy()
                pred_inv = inverse_dihedral_transform_np(pred_np, tid)
                predictions.append(pred_inv)
        
        # Vote
        voted = vote_predictions(predictions)
        
        return voted, predictions
    
    def evaluate_task(
        self,
        task: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Evaluate a single ARC task with voting.
        
        Args:
            task: Task dictionary with 'train' and 'test' pairs
        
        Returns:
            Dictionary with predictions and metrics
        """
        # Parse task
        train_inputs = [torch.tensor(p['input'], dtype=torch.long) for p in task['train']]
        train_outputs = [torch.tensor(p['output'], dtype=torch.long) for p in task['train']]
        
        # Stack demos
        demo_inputs = torch.stack(train_inputs)  # [N, H, W]
        demo_outputs = torch.stack(train_outputs)
        
        results = []
        for test_pair in task['test']:
            test_input = torch.tensor(test_pair['input'], dtype=torch.long)
            
            # Predict with voting
            prediction, all_preds = self.predict_with_voting(
                demo_inputs, demo_outputs, test_input
            )
            
            result = {
                'prediction': prediction,
                'all_predictions': all_preds,
            }
            
            # Check if we have ground truth
            if 'output' in test_pair:
                target = np.array(test_pair['output'])
                correct = np.array_equal(prediction, target)
                result['target'] = target
                result['correct'] = correct
            
            results.append(result)
        
        return {'results': results}


def pass_at_k(
    model: nn.Module,
    tasks: List[Dict],
    k: int = 2,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Compute pass@K metric.
    
    Allow K attempts per puzzle. Score = % puzzles where any attempt is correct.
    
    Args:
        model: Model to evaluate
        tasks: List of task dictionaries
        k: Number of attempts allowed per puzzle
        device: Device to run on
    
    Returns:
        Dictionary with pass@k metrics
    """
    voter = AugmentationVoter(model, num_dihedral=8, device=device)
    
    correct_any = 0
    correct_first = 0
    total = 0
    
    for task in tasks:
        eval_result = voter.evaluate_task(task)
        
        for result in eval_result['results']:
            if 'correct' not in result:
                continue
            
            total += 1
            
            # First attempt correct
            if result['correct']:
                correct_first += 1
            
            # Any of top K predictions correct
            target = result['target']
            any_correct = False
            for i, pred in enumerate(result['all_predictions'][:k]):
                if np.array_equal(pred, target):
                    any_correct = True
                    break
            
            if any_correct:
                correct_any += 1
    
    return {
        'pass@1': correct_first / max(1, total),
        f'pass@{k}': correct_any / max(1, total),
        'total_puzzles': total,
    }
