#!/usr/bin/env python
"""
SCI-ARC Evaluation Script.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best_model.pt --data ./data/arc-agi
    python scripts/evaluate.py --checkpoint checkpoints/best_model.pt --split evaluation
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np

from sci_arc.models import SCIARC, SCIARCConfig
from sci_arc.data import SCIARCDataset, collate_sci_arc, create_dataloader
from sci_arc.evaluation import ARCEvaluator, EvaluationConfig, ARCMetrics


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
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualization images')
    
    args = parser.parse_args()
    
    # Check device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
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
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_path = output_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(results['metrics'], f, indent=2)
    print(f"\nSaved metrics to {metrics_path}")
    
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
    
    return results


if __name__ == '__main__':
    main()
