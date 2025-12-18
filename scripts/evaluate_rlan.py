#!/usr/bin/env python
"""
Comprehensive RLAN Evaluation Script - CISL Production Parity

This script provides complete evaluation of RLAN models with all metrics and outputs
matching CISL's production evaluation:

Features:
- All metrics: pixel, task, size, color, non-background accuracy, IoU
- Detailed JSON output per task
- Test-Time Augmentation (TTA) with dihedral transforms
- Attention pattern analysis for interpretability
- File logging (TeeLogger)
- HTML report generation compatible output

Usage:
    python scripts/evaluate_rlan.py --checkpoint checkpoints/rlan/best.pt
    python scripts/evaluate_rlan.py --checkpoint checkpoints/rlan/best.pt --use-tta
    python scripts/evaluate_rlan.py --checkpoint checkpoints/rlan/best.pt --detailed-output
    python scripts/evaluate_rlan.py --checkpoint checkpoints/rlan/best.pt --visualize
"""

import argparse
import json
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sci_arc.models import RLAN
from sci_arc.data import ARCDataset
from sci_arc.evaluation import (
    pixel_accuracy,
    task_accuracy,
    size_accuracy,
    color_accuracy,
    non_background_accuracy,
    mean_iou,
    iou_per_color,
    partial_match_score,
    ARCMetrics,
    visualize_prediction,
    ARC_COLORS_HEX,
)


class TeeLogger:
    """Logger that writes to both stdout and a file (encoding-safe for Windows)."""
    def __init__(self, log_path: Path):
        self.terminal = sys.stdout
        self.log_path = log_path
        # Use UTF-8 with error handling for Windows compatibility
        self.log_file = open(log_path, 'w', encoding='utf-8', errors='replace', buffering=1)
        
    def write(self, message):
        # Handle potential encoding issues on Windows terminal
        try:
            self.terminal.write(message)
        except UnicodeEncodeError:
            # Fallback to ASCII-safe version for Windows cmd
            self.terminal.write(message.encode('ascii', errors='replace').decode('ascii'))
        self.log_file.write(message)
        self.log_file.flush()
        
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()
        
    def close(self):
        self.log_file.close()
        sys.stdout = self.terminal


def apply_dihedral_transform(grid: torch.Tensor, tid: int) -> torch.Tensor:
    """Apply one of 8 dihedral transforms."""
    if tid == 0:
        return grid
    elif tid == 1:
        return torch.rot90(grid, k=1, dims=[-2, -1])
    elif tid == 2:
        return torch.rot90(grid, k=2, dims=[-2, -1])
    elif tid == 3:
        return torch.rot90(grid, k=3, dims=[-2, -1])
    elif tid == 4:
        return torch.flip(grid, dims=[-1])  # horizontal flip
    elif tid == 5:
        return torch.flip(grid, dims=[-2])  # vertical flip
    elif tid == 6:
        return grid.transpose(-2, -1)  # transpose
    elif tid == 7:
        return torch.flip(torch.rot90(grid, k=1, dims=[-2, -1]), dims=[-1])
    return grid


def inverse_dihedral_transform(grid: torch.Tensor, tid: int) -> torch.Tensor:
    """Apply inverse of dihedral transform."""
    # Inverse mapping
    inverse_map = [0, 3, 2, 1, 4, 5, 6, 7]
    inv_tid = inverse_map[tid]
    return apply_dihedral_transform(grid, inv_tid)


def predict_with_tta(
    model: RLAN,
    input_grid: torch.Tensor,
    num_transforms: int = 8,
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    Predict with Test-Time Augmentation using dihedral transforms.
    
    Applies all 8 dihedral transforms, then aggregates via majority voting.
    """
    device = input_grid.device
    all_predictions = []
    
    for tid in range(num_transforms):
        # Transform input
        transformed = apply_dihedral_transform(input_grid, tid)
        
        # Get prediction
        with torch.no_grad():
            logits = model(transformed, temperature=temperature)
            pred = logits.argmax(dim=1)
        
        # Inverse transform prediction
        pred_inv = inverse_dihedral_transform(pred, tid)
        all_predictions.append(pred_inv)
    
    # Stack predictions: (num_transforms, B, H, W)
    stacked = torch.stack(all_predictions, dim=0)
    
    # Majority voting
    num_classes = 10
    votes = F.one_hot(stacked.long(), num_classes=num_classes).sum(dim=0)  # (B, H, W, num_classes)
    final_pred = votes.argmax(dim=-1)  # (B, H, W)
    
    return final_pred


def save_detailed_predictions(
    output_dir: Path,
    task_id: str,
    input_grid: np.ndarray,
    target_grid: np.ndarray,
    prediction_grid: np.ndarray,
    is_correct: bool,
    attempt_num: int = 0,
) -> Dict[str, Any]:
    """
    Save detailed prediction information for a single task.
    
    Matches CISL's output format for compatibility with analyze_evaluation.py.
    """
    # Compute all metrics
    metrics = partial_match_score(prediction_grid, target_grid)
    
    # Find diff positions (where prediction differs from target)
    if prediction_grid.shape == target_grid.shape:
        diff_mask = prediction_grid != target_grid
        diff_positions = list(zip(*np.where(diff_mask)))
        diff_positions = [(int(r), int(c)) for r, c in diff_positions]
    else:
        diff_positions = []
    
    # Compute color statistics
    pred_colors = set(int(c) for c in prediction_grid.flatten())
    target_colors = set(int(c) for c in target_grid.flatten())
    
    detail = {
        'task_id': task_id,
        'attempt': attempt_num,
        'is_correct': is_correct,
        
        # Grid shapes
        'input_shape': list(input_grid.shape),
        'target_shape': list(target_grid.shape),
        'prediction_shape': list(prediction_grid.shape),
        'size_match': prediction_grid.shape == target_grid.shape,
        
        # All metrics
        'pixel_accuracy': float(metrics['pixel_accuracy']),
        'non_background_accuracy': float(metrics['non_background_accuracy']),
        'color_jaccard': float(metrics['color_jaccard']),
        'mean_iou': float(metrics['mean_iou']),
        'normalized_edit': float(metrics['normalized_edit']),
        
        # Color analysis
        'pred_colors': sorted(list(pred_colors)),
        'target_colors': sorted(list(target_colors)),
        'color_match': pred_colors == target_colors,
        
        # Diff analysis
        'num_diff_pixels': len(diff_positions),
        'diff_positions': diff_positions[:100],  # Limit for JSON size
        
        # Grids for visualization (as lists for JSON)
        'input_grid': input_grid.tolist(),
        'target_grid': target_grid.tolist(),
        'prediction_grid': prediction_grid.tolist(),
    }
    
    return detail


def trim_grid(grid: np.ndarray) -> np.ndarray:
    """
    Trim padding from grid by finding actual content bounds.
    
    Removes trailing zeros (padding) from both dimensions.
    """
    if grid.size == 0:
        return grid
    
    # Find rows with any non-zero
    row_mask = np.any(grid != 0, axis=1)
    if not row_mask.any():
        return grid[:1, :1]  # All zeros - return minimal grid
    
    # Find columns with any non-zero
    col_mask = np.any(grid != 0, axis=0)
    if not col_mask.any():
        return grid[:1, :1]
    
    # Find bounds
    rows = np.where(row_mask)[0]
    cols = np.where(col_mask)[0]
    
    if len(rows) == 0 or len(cols) == 0:
        return grid[:1, :1]
    
    # Include from 0 to max+1 to preserve full grid content
    max_row = rows.max() + 1
    max_col = cols.max() + 1
    
    return grid[:max_row, :max_col]


def analyze_attention_patterns(
    model: RLAN,
    dataloader: DataLoader,
    device: torch.device,
    num_samples: int = 10,
    temperature: float = 0.1,
) -> Dict[str, Any]:
    """Analyze attention patterns for interpretability."""
    model.eval()
    
    analysis = {
        'avg_active_clues': 0.0,
        'avg_attention_entropy': 0.0,
        'predicate_activations': [],
        'samples': [],
    }
    
    num_analyzed = 0
    
    for batch in dataloader:
        if num_analyzed >= num_samples:
            break
        
        input_grids = batch['test_input'].to(device)
        
        with torch.no_grad():
            try:
                outputs = model(input_grids, temperature=temperature, return_intermediates=True)
                
                if 'attention_maps' not in outputs:
                    break
                
                batch_size = input_grids.shape[0]
                for i in range(batch_size):
                    if num_analyzed >= num_samples:
                        break
                    
                    attention_maps = outputs['attention_maps'][i]
                    stop_logits = outputs['stop_logits'][i]
                    predicates = outputs.get('predicates', [None])[i]
                    
                    # Count active clues
                    stop_probs = torch.sigmoid(stop_logits)
                    active = (stop_probs < 0.5).sum().item()
                    analysis['avg_active_clues'] += active
                    
                    # Compute attention entropy
                    attention_flat = attention_maps.view(attention_maps.shape[0], -1)
                    entropy = -(attention_flat * torch.log(attention_flat + 1e-10)).sum(dim=-1).mean()
                    analysis['avg_attention_entropy'] += entropy.item()
                    
                    # Record predicate activations
                    if predicates is not None:
                        analysis['predicate_activations'].append(predicates.cpu().tolist())
                    
                    analysis['samples'].append({
                        'active_clues': active,
                        'attention_entropy': entropy.item(),
                    })
                    
                    num_analyzed += 1
            except Exception as e:
                print(f"Warning: Could not analyze attention patterns: {e}")
                break
    
    # Average
    if num_analyzed > 0:
        analysis['avg_active_clues'] /= num_analyzed
        analysis['avg_attention_entropy'] /= num_analyzed
    
    return analysis


def evaluate_model(
    model: RLAN,
    dataloader: DataLoader,
    device: torch.device,
    use_tta: bool = False,
    output_dir: Optional[Path] = None,
    detailed_output: bool = False,
    visualize: bool = False,
    temperature: float = 0.1,
) -> Dict[str, Any]:
    """
    Evaluate model with comprehensive metrics.
    
    Returns results matching CISL's evaluate.py output format.
    
    Args:
        temperature: Softmax temperature (should match training end temperature)
    """
    model.eval()
    
    metrics = ARCMetrics()
    all_details = []
    correct_count = 0
    incorrect_count = 0
    
    print("\nRunning evaluation...")
    
    for batch_idx, batch in enumerate(dataloader):
        # Get data
        test_inputs = batch['test_input'].to(device)
        test_outputs = batch['test_output'].to(device)
        task_ids = batch.get('task_id', [f'task_{batch_idx}_{i}' for i in range(test_inputs.shape[0])])
        
        if isinstance(task_ids, torch.Tensor):
            task_ids = [str(t.item()) if t.dim() == 0 else str(t.tolist()) for t in task_ids]
        elif not isinstance(task_ids, list):
            task_ids = [str(task_ids)]
        
        with torch.no_grad():
            if use_tta:
                predictions = predict_with_tta(model, test_inputs, temperature=temperature)
            else:
                logits = model(test_inputs, temperature=temperature)
                predictions = logits.argmax(dim=1)
        
        # Process each sample
        batch_size = test_inputs.shape[0]
        for i in range(batch_size):
            task_id = task_ids[i] if i < len(task_ids) else f'task_{batch_idx}_{i}'
            
            # Get numpy arrays
            input_np = test_inputs[i].cpu().numpy()
            target_np = test_outputs[i].cpu().numpy()
            pred_np = predictions[i].cpu().numpy()
            
            # Trim padding
            target_trimmed = trim_grid(target_np)
            pred_trimmed = pred_np[:target_trimmed.shape[0], :target_trimmed.shape[1]]
            input_trimmed = trim_grid(input_np)
            
            # Check correctness
            is_correct = np.array_equal(pred_trimmed, target_trimmed)
            
            if is_correct:
                correct_count += 1
            else:
                incorrect_count += 1
            
            # Update metrics accumulator
            metrics.update(task_id, pred_trimmed, target_trimmed)
            
            # Save detailed prediction
            if detailed_output and output_dir:
                detail = save_detailed_predictions(
                    output_dir=output_dir,
                    task_id=task_id,
                    input_grid=input_trimmed,
                    target_grid=target_trimmed,
                    prediction_grid=pred_trimmed,
                    is_correct=is_correct,
                    attempt_num=0,
                )
                all_details.append(detail)
            
            # Print progress
            status = "[OK]" if is_correct else "[X]"
            pix_acc = pixel_accuracy(pred_trimmed, target_trimmed)
            print(f"  [{batch_idx+1}/{len(dataloader)}] Task {task_id}: {status} "
                  f"(pixel_acc={pix_acc:.2%})")
        
        # Visualize if requested
        if visualize and output_dir:
            viz_dir = output_dir / 'visualizations'
            viz_dir.mkdir(exist_ok=True)
            
            for i in range(batch_size):
                task_id = task_ids[i] if i < len(task_ids) else f'task_{batch_idx}_{i}'
                input_np = test_inputs[i].cpu().numpy()
                target_np = test_outputs[i].cpu().numpy()
                pred_np = predictions[i].cpu().numpy()
                
                input_trimmed = trim_grid(input_np)
                target_trimmed = trim_grid(target_np)
                pred_trimmed = pred_np[:target_trimmed.shape[0], :target_trimmed.shape[1]]
                
                try:
                    visualize_prediction(
                        input_grid=input_trimmed,
                        target=target_trimmed,
                        prediction=pred_trimmed,
                        save_path=str(viz_dir / f'{task_id}.png'),
                        title=f'Task: {task_id}',
                    )
                except Exception as e:
                    print(f"Warning: Could not save visualization for {task_id}: {e}")
    
    # Get final summary
    summary = metrics.get_summary()
    
    # Build results
    results = {
        'metrics': summary,
        'summary': _format_summary(summary, correct_count, incorrect_count, use_tta),
        'per_task': metrics.per_task_results,
        'all_details': all_details,
        'correct_count': correct_count,
        'incorrect_count': incorrect_count,
    }
    
    return results


def _format_summary(summary: Dict, correct: int, incorrect: int, use_tta: bool) -> str:
    """Format summary as string for printing."""
    lines = [
        "=" * 60,
        "EVALUATION RESULTS",
        "=" * 60,
        f"Total Tasks: {summary['total_tasks']}",
        f"Correct Tasks: {correct}",
        f"Incorrect Tasks: {incorrect}",
        "-" * 40,
        f"Task Accuracy:           {summary['task_accuracy']:.4f} ({summary['task_accuracy']*100:.2f}%)",
        f"Pixel Accuracy:          {summary['pixel_accuracy']:.4f} ({summary['pixel_accuracy']*100:.2f}%)",
        f"Size Accuracy:           {summary['size_accuracy']:.4f} ({summary['size_accuracy']*100:.2f}%)",
        f"Non-Background Accuracy: {summary['non_background_accuracy']:.4f} ({summary['non_background_accuracy']*100:.2f}%)",
        f"Color Accuracy:          {summary['color_accuracy']:.4f} ({summary['color_accuracy']*100:.2f}%)",
        f"Mean IoU:                {summary['mean_iou']:.4f} ({summary['mean_iou']*100:.2f}%)",
        "=" * 60,
    ]
    
    if use_tta:
        lines.insert(-1, "(with Test-Time Augmentation)")
    
    return '\n'.join(lines)


def load_model(checkpoint_path: str, device: torch.device) -> RLAN:
    """Load RLAN model from checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Get model config
    if 'config' in checkpoint:
        model_config = checkpoint['config']
        if isinstance(model_config, dict) and 'model' in model_config:
            model_config = model_config['model']
    else:
        model_config = {}
    
    # Create model
    model = RLAN(
        hidden_dim=model_config.get('hidden_dim', 128),
        num_colors=model_config.get('num_colors', 10),
        num_classes=model_config.get('num_classes', 10),
        max_clues=model_config.get('max_clues', 5),
        num_predicates=model_config.get('num_predicates', 8),
        num_solver_steps=model_config.get('num_solver_steps', 6),
        use_act=model_config.get('use_act', False),
    )
    
    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive RLAN Evaluation with CISL Parity"
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file (optional)')
    parser.add_argument('--data-path', type=str, default=None,
                        help='Path to evaluation data')
    parser.add_argument('--output', type=str, default='./evaluation_results',
                        help='Output directory for results')
    parser.add_argument('--use-tta', action='store_true',
                        help='Use test-time augmentation (8 dihedral transforms)')
    parser.add_argument('--detailed-output', action='store_true',
                        help='Save detailed prediction vs reference for each task')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualization images')
    parser.add_argument('--analyze-attention', action='store_true',
                        help='Analyze attention patterns')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for evaluation')
    parser.add_argument('--temperature', type=float, default=0.1,
                        help='Temperature for softmax (lower = sharper, use same as training end)')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup file logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = output_dir / f'evaluation_log_{timestamp}.txt'
    tee_logger = TeeLogger(log_path)
    sys.stdout = tee_logger
    
    print("=" * 60)
    print("RLAN Evaluation")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Log file: {log_path}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output directory: {args.output}")
    print(f"Use TTA: {args.use_tta}")
    print(f"Detailed output: {args.detailed_output}")
    print("=" * 60)
    
    # Setup device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Load config
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    
    # Load model
    model = load_model(args.checkpoint, device)
    
    # Log model configuration
    print("\n" + "=" * 60)
    print("Model Configuration:")
    print("=" * 60)
    print(f"  Hidden dim: {model.hidden_dim}")
    print(f"  Num colors: {model.num_colors}")
    print(f"  Num classes: {model.num_classes}")
    print("=" * 60)
    
    # Create dataset
    data_path = args.data_path
    if data_path is None:
        data_path = config.get('data', {}).get('eval_path')
        if data_path is None:
            data_path = './data/arc-agi/data/evaluation'
    
    print(f"\nLoading data from: {data_path}")
    dataset = ARCDataset(data_path, augment=False)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    print(f"Evaluating on {len(dataset)} tasks")
    
    # Create details directory if needed
    if args.detailed_output:
        details_dir = output_dir / 'detailed_predictions'
        details_dir.mkdir(exist_ok=True)
    else:
        details_dir = None
    
    # Run evaluation
    print(f"\nUsing temperature: {args.temperature}")
    results = evaluate_model(
        model=model,
        dataloader=dataloader,
        device=device,
        use_tta=args.use_tta,
        output_dir=output_dir,
        detailed_output=args.detailed_output,
        visualize=args.visualize,
        temperature=args.temperature,
    )
    
    # Print results
    print("\n" + results['summary'])
    
    # Save metrics
    metrics_path = output_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(results['metrics'], f, indent=2)
    print(f"\nSaved metrics to {metrics_path}")
    
    # Save detailed predictions
    if args.detailed_output and results['all_details']:
        details_dir = output_dir / 'detailed_predictions'
        details_dir.mkdir(exist_ok=True)
        
        all_details_path = details_dir / 'all_predictions.json'
        with open(all_details_path, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'checkpoint': args.checkpoint,
                'use_tta': args.use_tta,
                'total_tasks': len(results['all_details']),
                'correct': results['correct_count'],
                'incorrect': results['incorrect_count'],
                'accuracy': results['correct_count'] / len(results['all_details']) if results['all_details'] else 0.0,
                'predictions': results['all_details'],
            }, f, indent=2)
        
        print(f"Saved detailed predictions to {all_details_path}")
        
        # Save individual task files
        for detail in results['all_details']:
            task_path = details_dir / f"{detail['task_id']}.json"
            with open(task_path, 'w') as f:
                json.dump(detail, f, indent=2)
    
    # Attention analysis
    if args.analyze_attention:
        print("\n" + "=" * 60)
        print("Attention Pattern Analysis")
        print("=" * 60)
        
        analysis = analyze_attention_patterns(model, dataloader, device, temperature=args.temperature)
        
        print(f"Average Active Clues: {analysis['avg_active_clues']:.2f}")
        print(f"Average Attention Entropy: {analysis['avg_attention_entropy']:.4f}")
        
        if analysis['predicate_activations']:
            import torch
            avg_preds = torch.tensor(analysis['predicate_activations']).mean(dim=0)
            print("Average Predicate Activations:")
            for i, val in enumerate(avg_preds.tolist()):
                print(f"  Predicate {i}: {val:.3f}")
        
        # Save analysis
        analysis_path = output_dir / 'attention_analysis.json'
        with open(analysis_path, 'w') as f:
            json.dump({
                'avg_active_clues': analysis['avg_active_clues'],
                'avg_attention_entropy': analysis['avg_attention_entropy'],
            }, f, indent=2)
        print(f"\nSaved attention analysis to {analysis_path}")
    
    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print("=" * 60)
    
    # Close logger
    tee_logger.close()
    
    return results


if __name__ == '__main__':
    main()
