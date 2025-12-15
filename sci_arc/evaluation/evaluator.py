"""
ARC Evaluator for SCI-ARC.

Provides:
1. ARCEvaluator: Full evaluation pipeline
2. Multi-attempt evaluation (2 attempts like official ARC)
3. Visualization of predictions
4. Comparison with baselines
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .metrics import ARCMetrics, task_accuracy, pixel_accuracy, partial_match_score


@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""
    num_attempts: int = 2  # Official ARC allows 2 attempts
    batch_size: int = 1
    use_voting: bool = True  # Use ensemble voting for multiple attempts
    temperature: float = 1.0  # Softmax temperature for sampling
    device: str = 'cuda'
    save_predictions: bool = True
    output_dir: str = './evaluation_results'


class ARCEvaluator:
    """
    Evaluator for ARC tasks.
    
    Features:
    - Multiple attempt generation
    - Ensemble voting
    - Detailed per-task analysis
    - Visualization
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[EvaluationConfig] = None,
    ):
        self.model = model
        self.config = config or EvaluationConfig()
        
        self.device = torch.device(
            self.config.device if torch.cuda.is_available() else 'cpu'
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Output directory
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    @torch.no_grad()
    def predict_single(
        self,
        input_grids: List[torch.Tensor],
        output_grids: List[torch.Tensor],
        test_input: torch.Tensor,
        num_attempts: int = 1,
    ) -> List[np.ndarray]:
        """
        Generate predictions for a single task.
        
        Args:
            input_grids: List of input grids from training examples
            output_grids: List of output grids from training examples
            test_input: Test input grid to predict output for
            num_attempts: Number of prediction attempts
        
        Returns:
            List of predicted grids (one per attempt)
        """
        # Prepare batch (single task)
        # Stack training examples
        max_pairs = len(input_grids)
        max_h = max(max(g.shape[0] for g in input_grids), test_input.shape[0])
        max_w = max(max(g.shape[1] for g in input_grids), test_input.shape[1])
        max_size = max(max_h, max_w, 30)  # At least 30
        
        # Pad grids
        def pad_grid(g, size):
            padded = torch.zeros(size, size, dtype=torch.long)
            h, w = g.shape
            padded[:h, :w] = g
            return padded
        
        input_batch = torch.stack([pad_grid(g, max_size) for g in input_grids])
        output_batch = torch.stack([pad_grid(g, max_size) for g in output_grids])
        test_batch = pad_grid(test_input, max_size)
        
        # Add batch dimension
        input_batch = input_batch.unsqueeze(0).to(self.device)  # [1, N, H, W]
        output_batch = output_batch.unsqueeze(0).to(self.device)
        test_batch = test_batch.unsqueeze(0).to(self.device)  # [1, H, W]
        grid_mask = torch.ones(1, max_pairs, dtype=torch.bool, device=self.device)
        
        predictions = []
        
        for attempt in range(num_attempts):
            # Forward pass
            outputs = self.model(
                input_grids=input_batch,
                output_grids=output_batch,
                test_input=test_batch,
                grid_mask=grid_mask,
            )
            
            logits = outputs['logits']  # [1, H, W, C]
            
            if attempt == 0 or not self.config.use_voting:
                # Greedy decoding for first attempt
                pred = logits.argmax(dim=-1)  # [1, H, W]
            else:
                # Sample with temperature for diversity
                probs = torch.softmax(logits / self.config.temperature, dim=-1)
                pred = torch.multinomial(
                    probs.view(-1, probs.shape[-1]), 
                    num_samples=1
                ).view(1, logits.shape[1], logits.shape[2])
            
            pred_np = pred[0].cpu().numpy()
            
            # Crop to estimated output size
            # For now, use test input size (common in ARC)
            h, w = test_input.shape
            pred_np = pred_np[:h, :w]
            
            predictions.append(pred_np)
        
        return predictions
    
    def evaluate(
        self,
        dataloader: DataLoader,
        targets: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate on a dataset.
        
        Args:
            dataloader: DataLoader with test tasks
            targets: Optional dict of ground truth (if not in dataloader)
        
        Returns:
            Dict with metrics and per-task results
        """
        self.model.eval()
        metrics = ARCMetrics()
        
        all_predictions = {}
        per_task_details = {}
        
        for batch in dataloader:
            task_ids = batch['task_ids']
            
            for i, task_id in enumerate(task_ids):
                # Extract single task data
                num_pairs = batch['num_pairs'][i].item()
                
                input_grids = [batch['input_grids'][i, j] for j in range(num_pairs)]
                output_grids = [batch['output_grids'][i, j] for j in range(num_pairs)]
                test_input = batch['test_inputs'][i]
                
                # Get target
                if targets and task_id in targets:
                    target = targets[task_id]
                else:
                    target = batch['test_outputs'][i].numpy()
                
                # Generate predictions
                predictions = self.predict_single(
                    input_grids=input_grids,
                    output_grids=output_grids,
                    test_input=test_input,
                    num_attempts=self.config.num_attempts,
                )
                
                # Check each attempt
                best_pred = None
                is_correct = False
                best_score = 0.0
                
                for attempt_idx, pred in enumerate(predictions):
                    if task_accuracy(pred, target):
                        is_correct = True
                        best_pred = pred
                        break
                    
                    # Track best partial match
                    score = pixel_accuracy(pred, target)
                    if score > best_score:
                        best_score = score
                        best_pred = pred
                
                # Update metrics
                if best_pred is not None:
                    metrics.update(task_id, best_pred, target)
                    all_predictions[task_id] = best_pred
                
                per_task_details[task_id] = {
                    'correct': is_correct,
                    'num_attempts': len(predictions),
                    'predictions': predictions,
                    'target': target,
                    'best_pixel_accuracy': best_score if not is_correct else 1.0,
                }
        
        # Compile results
        results = {
            'metrics': metrics.compute(),
            'summary': metrics.summary(),
            'predictions': all_predictions,
            'per_task': per_task_details,
        }
        
        # Save predictions if configured
        if self.config.save_predictions:
            self._save_predictions(all_predictions)
        
        return results
    
    def _save_predictions(self, predictions: Dict[str, np.ndarray]):
        """Save predictions to JSON."""
        # Convert to serializable format
        output = {}
        for task_id, pred in predictions.items():
            output[task_id] = pred.tolist()
        
        path = self.output_dir / 'predictions.json'
        with open(path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"Saved predictions to {path}")
    
    def evaluate_with_voting(
        self,
        dataloader: DataLoader,
        num_ensemble: int = 5,
    ) -> Dict[str, Any]:
        """
        Evaluate with ensemble voting.
        
        Generates multiple predictions and uses majority voting.
        """
        self.model.eval()
        
        all_predictions = defaultdict(list)
        
        # Generate multiple predictions per task
        for _ in range(num_ensemble):
            for batch in dataloader:
                task_ids = batch['task_ids']
                
                for i, task_id in enumerate(task_ids):
                    num_pairs = batch['num_pairs'][i].item()
                    
                    input_grids = [batch['input_grids'][i, j] for j in range(num_pairs)]
                    output_grids = [batch['output_grids'][i, j] for j in range(num_pairs)]
                    test_input = batch['test_inputs'][i]
                    
                    preds = self.predict_single(
                        input_grids, output_grids, test_input, num_attempts=1
                    )
                    all_predictions[task_id].append(preds[0])
        
        # Majority voting
        voted_predictions = {}
        for task_id, preds in all_predictions.items():
            # Stack predictions
            stacked = np.stack(preds, axis=0)  # [num_ensemble, H, W]
            
            # Mode voting per pixel
            from scipy import stats
            voted = stats.mode(stacked, axis=0)[0].squeeze(0)
            voted_predictions[task_id] = voted
        
        # Evaluate voted predictions
        return voted_predictions
    
    def compare_with_trm(
        self,
        trm_predictions: Dict[str, np.ndarray],
        targets: Dict[str, np.ndarray],
    ) -> Dict[str, Any]:
        """
        Compare SCI-ARC with TRM predictions.
        
        Args:
            trm_predictions: TRM model predictions
            targets: Ground truth
        
        Returns:
            Comparison metrics
        """
        sci_arc_metrics = ARCMetrics()
        trm_metrics = ARCMetrics()
        
        comparison = {
            'both_correct': 0,
            'sci_arc_only': 0,
            'trm_only': 0,
            'neither': 0,
            'details': {},
        }
        
        for task_id, target in targets.items():
            # We need SCI-ARC predictions - this should be called after evaluate()
            # For now, assume self.last_predictions exists
            sci_arc_pred = getattr(self, 'last_predictions', {}).get(task_id)
            trm_pred = trm_predictions.get(task_id)
            
            if sci_arc_pred is None or trm_pred is None:
                continue
            
            sci_correct = task_accuracy(sci_arc_pred, target)
            trm_correct = task_accuracy(trm_pred, target)
            
            if sci_correct and trm_correct:
                comparison['both_correct'] += 1
            elif sci_correct:
                comparison['sci_arc_only'] += 1
            elif trm_correct:
                comparison['trm_only'] += 1
            else:
                comparison['neither'] += 1
            
            sci_arc_metrics.update(task_id, sci_arc_pred, target)
            trm_metrics.update(task_id, trm_pred, target)
            
            comparison['details'][task_id] = {
                'sci_arc_correct': sci_correct,
                'trm_correct': trm_correct,
                'sci_arc_pixel_acc': pixel_accuracy(sci_arc_pred, target),
                'trm_pixel_acc': pixel_accuracy(trm_pred, target),
            }
        
        comparison['sci_arc_metrics'] = sci_arc_metrics.compute()
        comparison['trm_metrics'] = trm_metrics.compute()
        
        return comparison


def visualize_prediction(
    input_grid: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    save_path: Optional[str] = None,
) -> Optional[np.ndarray]:
    """
    Visualize a prediction alongside input and target.
    
    Returns:
        RGB image array if matplotlib available
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        print("matplotlib required for visualization")
        return None
    
    # ARC color palette
    ARC_COLORS = [
        '#000000',  # 0: black (background)
        '#0074D9',  # 1: blue
        '#FF4136',  # 2: red
        '#2ECC40',  # 3: green
        '#FFDC00',  # 4: yellow
        '#AAAAAA',  # 5: gray
        '#F012BE',  # 6: magenta
        '#FF851B',  # 7: orange
        '#7FDBFF',  # 8: cyan
        '#870C25',  # 9: brown
    ]
    
    cmap = mcolors.ListedColormap(ARC_COLORS)
    norm = mcolors.BoundaryNorm(range(11), cmap.N)
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    axes[0].imshow(input_grid, cmap=cmap, norm=norm)
    axes[0].set_title('Input')
    axes[0].axis('off')
    
    axes[1].imshow(target, cmap=cmap, norm=norm)
    axes[1].set_title('Target')
    axes[1].axis('off')
    
    axes[2].imshow(prediction, cmap=cmap, norm=norm)
    is_correct = np.array_equal(prediction, target)
    axes[2].set_title(f'Prediction ({"✓" if is_correct else "✗"})')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    # Return as numpy array
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    
    return img


def generate_submission(
    predictions: Dict[str, np.ndarray],
    output_path: str = 'submission.json',
):
    """
    Generate Kaggle submission file.
    
    Format: {"task_id": [[row1], [row2], ...], ...}
    """
    submission = {}
    
    for task_id, pred in predictions.items():
        submission[task_id] = pred.tolist()
    
    with open(output_path, 'w') as f:
        json.dump(submission, f)
    
    print(f"Generated submission with {len(submission)} tasks at {output_path}")
