#!/usr/bin/env python
"""
SCI-ARC Evaluation Script.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best_model.pt --data ./data/arc-agi
    python scripts/evaluate.py --checkpoint checkpoints/best_model.pt --split evaluation
    python scripts/evaluate.py --checkpoint checkpoints/best_model.pt --save-predictions --detailed-output
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np

from sci_arc.models import SCIARC, SCIARCConfig
from sci_arc.data import SCIARCDataset, collate_sci_arc, create_dataloader
from sci_arc.evaluation import ARCEvaluator, EvaluationConfig, ARCMetrics


class TeeLogger:
    """Logger that writes to both stdout and a file."""
    def __init__(self, log_path: Path):
        self.terminal = sys.stdout
        self.log_path = log_path
        self.log_file = open(log_path, 'w', encoding='utf-8', buffering=1)
        
    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()
        
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()
        
    def close(self):
        self.log_file.close()
        sys.stdout = self.terminal


def load_model(checkpoint_path: str, device: str = 'cuda') -> SCIARC:
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get config from checkpoint
    config_dict = checkpoint.get('config', {})
    
    # Build model config (with defaults for backward compatibility)
    model_config = SCIARCConfig(
        hidden_dim=config_dict.get('hidden_dim', 256),
        num_colors=config_dict.get('num_colors', 10),
        max_grid_size=config_dict.get('max_grid_size', 30),
        num_structure_slots=config_dict.get('num_structure_slots', 8),
        num_abstraction_layers=config_dict.get('num_abstraction_layers', 3),
        max_objects=config_dict.get('max_objects', 16),
        H_cycles=config_dict.get('H_cycles', 3),
        L_cycles=config_dict.get('L_cycles', 4),
        L_layers=config_dict.get('L_layers', 2),
        dropout=config_dict.get('dropout', 0.1),
    )
    
    model = SCIARC(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Loaded model from {checkpoint_path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"  Best val accuracy: {checkpoint.get('best_val_accuracy', 'unknown')}")
    
    return model


def grid_to_list(grid: np.ndarray) -> list:
    """Convert numpy grid to nested list for JSON serialization."""
    return grid.tolist() if isinstance(grid, np.ndarray) else grid


def save_detailed_predictions(
    output_dir: Path,
    task_id: str,
    input_grid: np.ndarray,
    target_grid: np.ndarray,
    prediction_grid: np.ndarray,
    is_correct: bool,
    attempt_num: int = 0,
):
    """
    Save detailed prediction vs reference for a single task.
    
    Creates a JSON file with:
    - task_id
    - input grid
    - target/reference grid
    - model prediction grid  
    - correctness flag
    - grid dimensions
    - pixel-level diff statistics
    """
    # Compute diff statistics
    if prediction_grid is not None and target_grid is not None:
        pred = np.array(prediction_grid)
        target = np.array(target_grid)
        
        # Handle size mismatch
        if pred.shape != target.shape:
            size_match = False
            pixels_correct = 0
            pixels_total = target.size
            diff_positions = []
        else:
            size_match = True
            diff_mask = (pred != target)
            pixels_correct = int(np.sum(~diff_mask))
            pixels_total = int(target.size)
            diff_positions = np.argwhere(diff_mask).tolist()
    else:
        size_match = False
        pixels_correct = 0
        pixels_total = 0 if target_grid is None else np.array(target_grid).size
        diff_positions = []
    
    detail = {
        'task_id': task_id,
        'attempt': attempt_num,
        'is_correct': is_correct,
        'input_grid': grid_to_list(input_grid),
        'target_grid': grid_to_list(target_grid) if target_grid is not None else None,
        'prediction_grid': grid_to_list(prediction_grid) if prediction_grid is not None else None,
        'input_shape': list(input_grid.shape) if isinstance(input_grid, np.ndarray) else None,
        'target_shape': list(target_grid.shape) if isinstance(target_grid, np.ndarray) else None,
        'prediction_shape': list(prediction_grid.shape) if isinstance(prediction_grid, np.ndarray) else None,
        'size_match': size_match,
        'pixels_correct': pixels_correct,
        'pixels_total': pixels_total,
        'pixel_accuracy': pixels_correct / pixels_total if pixels_total > 0 else 0.0,
        'diff_positions': diff_positions[:100],  # Limit to first 100 diffs
        'num_diff_positions': len(diff_positions),
    }
    
    return detail


def main():
    parser = argparse.ArgumentParser(description='Evaluate SCI-ARC model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data', type=str, default='./data/arc-agi',
                        help='Path to ARC data directory')
    parser.add_argument('--split', type=str, default='evaluation',
                        choices=['training', 'evaluation'],
                        help='Data split to evaluate on')
    parser.add_argument('--output', type=str, default='./evaluation_results',
                        help='Output directory for results')
    parser.add_argument('--num-attempts', type=int, default=2,
                        help='Number of prediction attempts')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--save-predictions', action='store_true',
                        help='Save predictions to file')
    parser.add_argument('--detailed-output', action='store_true',
                        help='Save detailed prediction vs reference for each task')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualization images')
    
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
    print("SCI-ARC Evaluation")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Log file: {log_path}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data: {args.data}/{args.split}")
    print(f"Output directory: {args.output}")
    print(f"Detailed output: {args.detailed_output}")
    print("=" * 60)
    
    # Check device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Load model
    model = load_model(args.checkpoint, device)
    
    # Load data
    print(f"\nLoading data from: {args.data}/{args.split}")
    
    dataloader = create_dataloader(
        data_dir=args.data,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=0,  # Single-threaded for evaluation
        shuffle=False,
        augment=False,
    )
    
    print(f"Evaluating on {len(dataloader.dataset)} tasks")
    
    # Create evaluator
    eval_config = EvaluationConfig(
        num_attempts=args.num_attempts,
        batch_size=args.batch_size,
        device=device,
        save_predictions=args.save_predictions,
        output_dir=args.output,
    )
    
    evaluator = ARCEvaluator(model, eval_config)
    
    # Run evaluation
    print("\nRunning evaluation...")
    results = evaluator.evaluate(dataloader)
    
    # Print results
    print("\n" + results['summary'])
    
    # Save metrics
    metrics_path = output_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(results['metrics'], f, indent=2)
    print(f"\nSaved metrics to {metrics_path}")
    
    # Detailed output: save prediction vs reference for each task
    if args.detailed_output:
        print("\nSaving detailed predictions...")
        details_dir = output_dir / 'detailed_predictions'
        details_dir.mkdir(exist_ok=True)
        
        all_details = []
        correct_count = 0
        incorrect_count = 0
        
        # Iterate through dataset to get inputs and run inference
        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                task_ids = batch['task_ids']
                test_inputs = batch['test_inputs']
                test_outputs = batch['test_outputs']
                
                # Move to device
                input_grids = batch['input_grids'].to(device)
                output_grids = batch['output_grids'].to(device)
                test_input = test_inputs.to(device)
                test_output = test_outputs.to(device)
                grid_masks = batch.get('grid_masks')
                if grid_masks is not None:
                    grid_masks = grid_masks.to(device)
                
                # Forward pass
                outputs = model(
                    input_grids=input_grids,
                    output_grids=output_grids,
                    test_input=test_input,
                    test_output=test_output,
                    grid_mask=grid_masks,
                )
                
                # Get predictions
                logits = outputs['logits']  # [B, H, W, num_colors]
                predictions = logits.argmax(dim=-1).cpu().numpy()  # [B, H, W]
                
                # Process each item in batch
                for i in range(len(task_ids)):
                    task_id = task_ids[i]
                    input_grid = test_inputs[i].numpy()
                    target_grid = test_outputs[i].numpy()
                    pred_grid = predictions[i]
                    
                    # Trim padding (zeros) from grids if needed
                    # Find actual grid size from input
                    input_nonzero = np.where(input_grid != 0)
                    if len(input_nonzero[0]) > 0:
                        max_h = input_nonzero[0].max() + 1
                        max_w = input_nonzero[1].max() + 1
                    else:
                        max_h, max_w = input_grid.shape
                    
                    target_nonzero = np.where(target_grid != 0)
                    if len(target_nonzero[0]) > 0:
                        target_h = max(target_nonzero[0].max() + 1, 1)
                        target_w = max(target_nonzero[1].max() + 1, 1)
                    else:
                        target_h, target_w = target_grid.shape
                    
                    # Use target size for comparison
                    target_trimmed = target_grid[:target_h, :target_w]
                    pred_trimmed = pred_grid[:target_h, :target_w]
                    input_trimmed = input_grid[:max_h, :max_w]
                    
                    is_correct = np.array_equal(pred_trimmed, target_trimmed)
                    
                    if is_correct:
                        correct_count += 1
                    else:
                        incorrect_count += 1
                    
                    # Save detail
                    detail = save_detailed_predictions(
                        output_dir=details_dir,
                        task_id=task_id,
                        input_grid=input_trimmed,
                        target_grid=target_trimmed,
                        prediction_grid=pred_trimmed,
                        is_correct=is_correct,
                        attempt_num=0,
                    )
                    all_details.append(detail)
                    
                    # Print progress for each task
                    status = "✓" if is_correct else "✗"
                    print(f"  [{batch_idx+1}/{len(dataloader)}] Task {task_id}: {status} "
                          f"(pixel_acc={detail['pixel_accuracy']:.2%})")
        
        # Save all details to a single JSON file
        all_details_path = details_dir / 'all_predictions.json'
        with open(all_details_path, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'checkpoint': args.checkpoint,
                'split': args.split,
                'total_tasks': len(all_details),
                'correct': correct_count,
                'incorrect': incorrect_count,
                'accuracy': correct_count / len(all_details) if all_details else 0.0,
                'predictions': all_details,
            }, f, indent=2)
        
        print(f"\nSaved detailed predictions to {all_details_path}")
        print(f"Correct: {correct_count}/{len(all_details)} ({100*correct_count/len(all_details):.1f}%)")
        
        # Also save individual task files for easy browsing
        for detail in all_details:
            task_path = details_dir / f"{detail['task_id']}.json"
            with open(task_path, 'w') as f:
                json.dump(detail, f, indent=2)
    
    # Visualizations
    if args.visualize:
        from sci_arc.evaluation import visualize_prediction
        
        viz_dir = output_dir / 'visualizations'
        viz_dir.mkdir(exist_ok=True)
        
        print(f"\nGenerating visualizations in {viz_dir}")
        
        for task_id, details in results['per_task'].items():
            if len(details['predictions']) > 0:
                viz_path = viz_dir / f'{task_id}.png'
                
                # Get test input from dataset
                for batch in dataloader:
                    if batch['task_ids'][0] == task_id:
                        test_input = batch['test_inputs'][0].numpy()
                        break
                else:
                    continue
                
                visualize_prediction(
                    input_grid=test_input,
                    target=details['target'],
                    prediction=details['predictions'][0],
                    save_path=str(viz_path),
                )
        
        print(f"Saved {len(results['per_task'])} visualizations")
    
    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print("=" * 60)
    
    # Close logger
    tee_logger.close()
    
    return results


if __name__ == '__main__':
    main()
