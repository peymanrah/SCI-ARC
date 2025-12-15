#!/usr/bin/env python
"""
SCI-ARC Competitive Evaluation Script.

Evaluates SCI-ARC on ARC-AGI-1/2 with full inference pipeline:
1. Test-Time Training (TTT)
2. Stochastic Sampling
3. Augmentation Voting
4. Consistency Verification

Supports ablation studies to measure individual component impact.

Usage:
    # Full evaluation
    python scripts/evaluate_competitive.py --config configs/competitive.yaml --checkpoint best.pt
    
    # Ablation study (all modes)
    python scripts/evaluate_competitive.py --config configs/competitive.yaml --ablation all
    
    # Single ablation mode
    python scripts/evaluate_competitive.py --config configs/competitive.yaml --ablation no_ttt
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
import torch
import numpy as np
from tqdm import tqdm

from sci_arc.models import SCIARC, SCIARCConfig
from sci_arc.inference import EnsemblePredictor, EnsembleConfig
from sci_arc.inference.ensemble import evaluate_with_ablation


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_tasks(arc_dir: str, split: str = 'evaluation') -> List[Dict[str, Any]]:
    """
    Load ARC tasks from directory.
    
    Args:
        arc_dir: Path to ARC data directory
        split: 'training' or 'evaluation'
    
    Returns:
        List of task dictionaries
    """
    task_dir = Path(arc_dir) / split
    tasks = []
    
    for task_file in sorted(task_dir.glob('*.json')):
        with open(task_file, 'r') as f:
            task = json.load(f)
            task['task_id'] = task_file.stem
            tasks.append(task)
    
    return tasks


def load_model(config: dict, checkpoint_path: Optional[str] = None) -> SCIARC:
    """Load SCI-ARC model from config and optional checkpoint."""
    # Build model config
    model_config = SCIARCConfig(
        hidden_dim=config['model'].get('hidden_dim', 256),
        num_colors=config['model'].get('num_colors', 10),
        max_grid_size=config['model'].get('max_grid_size', 30),
        num_structure_slots=config['model'].get('num_structure_slots', 8),
        max_objects=config['model'].get('max_objects', 16),
        num_heads=config['model'].get('structure_heads', 8),
        dropout=config['model'].get('dropout', 0.1),
        H_cycles=config['model'].get('H_cycles', 3),
        L_cycles=config['model'].get('L_cycles', 4),
        L_layers=config['model'].get('L_layers', 2),
    )
    
    model = SCIARC(model_config)
    
    # Load checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    
    return model


def build_ensemble_config(config: dict) -> EnsembleConfig:
    """Build EnsembleConfig from YAML config."""
    inference_config = config.get('inference', {})
    
    return EnsembleConfig(
        use_ttt=inference_config.get('use_ttt', True),
        use_stochastic_sampling=inference_config.get('use_stochastic_sampling', True),
        use_augmentation_voting=inference_config.get('use_augmentation_voting', True),
        use_consistency_verification=inference_config.get('use_consistency_verification', True),
        num_samples=inference_config.get('num_samples', 32),
        temperature=inference_config.get('sampling_temperature', 0.8),
        num_dihedral=inference_config.get('voting_num_dihedral', 8),
        ttt_learning_rate=inference_config.get('ttt_learning_rate', 1e-4),
        ttt_steps=inference_config.get('ttt_steps', 20),
        ttt_modules=inference_config.get('ttt_modules', ['grid_encoder', 'structural_encoder']),
        top_k=inference_config.get('top_k', 3),
        device=config.get('hardware', {}).get('device', 'cuda'),
        verbose=False,
    )


def evaluate_single_mode(
    model: SCIARC,
    tasks: List[Dict[str, Any]],
    config: EnsembleConfig,
    mode: str = 'full',
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate model in a single ablation mode.
    
    Args:
        model: SCI-ARC model
        tasks: List of ARC tasks
        config: Ensemble configuration
        mode: Ablation mode
        verbose: Print progress
    
    Returns:
        Evaluation results
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Evaluating mode: {mode}")
        print(f"{'='*60}")
    
    start_time = time.time()
    results = evaluate_with_ablation(model, tasks, config, ablation_mode=mode)
    elapsed = time.time() - start_time
    
    results['time_seconds'] = elapsed
    results['time_per_task'] = elapsed / max(1, len(tasks))
    
    if verbose:
        print(f"Accuracy: {results['accuracy']*100:.2f}% ({results['correct']}/{results['total']})")
        print(f"Time: {elapsed:.1f}s ({results['time_per_task']:.2f}s/task)")
    
    return results


def run_full_evaluation(
    model: SCIARC,
    tasks: List[Dict[str, Any]],
    config: EnsembleConfig,
    output_dir: Path,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run full evaluation with detailed per-task results.
    
    Args:
        model: SCI-ARC model
        tasks: List of ARC tasks
        config: Ensemble configuration
        output_dir: Directory to save results
        verbose: Print progress
    
    Returns:
        Detailed evaluation results
    """
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    predictor = EnsemblePredictor(model, config)
    
    all_results = []
    correct = 0
    total = 0
    
    iterator = tqdm(tasks, desc="Evaluating") if verbose else tasks
    
    for task in iterator:
        task_id = task.get('task_id', 'unknown')
        
        try:
            task_results = predictor.predict_task(task)
            
            task_correct = 0
            task_total = len(task_results)
            
            for result in task_results:
                if 'correct' in result:
                    total += 1
                    if result['correct']:
                        correct += 1
                        task_correct += 1
            
            all_results.append({
                'task_id': task_id,
                'correct': task_correct,
                'total': task_total,
                'predictions': [
                    {
                        'rank': p['rank'],
                        'confidence': p['confidence'],
                        'frequency': p.get('frequency', 0),
                        'consistency': p.get('consistency', 0),
                    }
                    for r in task_results
                    for p in r['predictions'][:3]  # Top 3 only
                ]
            })
            
        except Exception as e:
            print(f"Error on task {task_id}: {e}")
            all_results.append({
                'task_id': task_id,
                'error': str(e),
            })
    
    # Summary
    accuracy = correct / max(1, total)
    
    summary = {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'config': {
            'use_ttt': config.use_ttt,
            'use_stochastic_sampling': config.use_stochastic_sampling,
            'use_augmentation_voting': config.use_augmentation_voting,
            'use_consistency_verification': config.use_consistency_verification,
            'ttt_steps': config.ttt_steps,
            'num_samples': config.num_samples,
        },
        'per_task': all_results,
    }
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'evaluation_results.json'
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    if verbose:
        print(f"\nResults saved to: {output_file}")
    
    return summary


def run_ablation_study(
    model: SCIARC,
    tasks: List[Dict[str, Any]],
    config: EnsembleConfig,
    output_dir: Path,
) -> Dict[str, Any]:
    """
    Run complete ablation study.
    
    Evaluates all ablation modes and compares results.
    """
    print("\n" + "="*60)
    print("ABLATION STUDY")
    print("="*60)
    
    modes = [
        'baseline',      # No inference modules
        'voting_only',   # + Augmentation voting
        'no_ttt',        # Full - TTT
        'no_sampling',   # Full - Sampling
        'no_consistency',# Full - Consistency
        'full',          # All enabled
    ]
    
    # Custom modes for specific ablations
    ablation_configs = {
        'baseline': EnsembleConfig(
            use_ttt=False,
            use_stochastic_sampling=False,
            use_augmentation_voting=False,
            use_consistency_verification=False,
            device=config.device,
        ),
        'voting_only': EnsembleConfig(
            use_ttt=False,
            use_stochastic_sampling=False,
            use_augmentation_voting=True,
            use_consistency_verification=False,
            num_dihedral=config.num_dihedral,
            device=config.device,
        ),
        'no_ttt': EnsembleConfig(
            use_ttt=False,
            use_stochastic_sampling=config.use_stochastic_sampling,
            use_augmentation_voting=config.use_augmentation_voting,
            use_consistency_verification=config.use_consistency_verification,
            num_samples=config.num_samples,
            temperature=config.temperature,
            num_dihedral=config.num_dihedral,
            device=config.device,
        ),
        'no_sampling': EnsembleConfig(
            use_ttt=config.use_ttt,
            use_stochastic_sampling=False,
            use_augmentation_voting=config.use_augmentation_voting,
            use_consistency_verification=config.use_consistency_verification,
            ttt_steps=config.ttt_steps,
            ttt_learning_rate=config.ttt_learning_rate,
            num_dihedral=config.num_dihedral,
            device=config.device,
        ),
        'no_consistency': EnsembleConfig(
            use_ttt=config.use_ttt,
            use_stochastic_sampling=config.use_stochastic_sampling,
            use_augmentation_voting=config.use_augmentation_voting,
            use_consistency_verification=False,
            ttt_steps=config.ttt_steps,
            ttt_learning_rate=config.ttt_learning_rate,
            num_samples=config.num_samples,
            temperature=config.temperature,
            num_dihedral=config.num_dihedral,
            device=config.device,
        ),
        'full': config,
    }
    
    results = {}
    
    for mode in modes:
        mode_config = ablation_configs[mode]
        result = evaluate_single_mode(model, tasks, mode_config, mode)
        results[mode] = result
    
    # Print comparison table
    print("\n" + "="*60)
    print("ABLATION RESULTS SUMMARY")
    print("="*60)
    print(f"{'Mode':<20} {'Accuracy':>10} {'Correct':>10} {'Time (s)':>10}")
    print("-"*60)
    
    for mode in modes:
        r = results[mode]
        print(f"{mode:<20} {r['accuracy']*100:>9.2f}% {r['correct']:>10} {r['time_seconds']:>10.1f}")
    
    # Calculate deltas from baseline
    baseline_acc = results['baseline']['accuracy']
    print("\n" + "-"*60)
    print("Delta from baseline:")
    for mode in modes[1:]:
        delta = results[mode]['accuracy'] - baseline_acc
        print(f"  {mode}: {delta*100:+.2f}%")
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'ablation_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='SCI-ARC Competitive Evaluation')
    parser.add_argument('--config', type=str, default='configs/competitive.yaml',
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint')
    parser.add_argument('--arc-dir', type=str, default=None,
                        help='Path to ARC data directory (overrides config)')
    parser.add_argument('--split', type=str, default='evaluation',
                        choices=['training', 'evaluation'],
                        help='Dataset split to evaluate')
    parser.add_argument('--ablation', type=str, default=None,
                        choices=['all', 'baseline', 'no_ttt', 'no_sampling', 
                                'no_voting', 'no_consistency', 'full'],
                        help='Run ablation study')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for results')
    parser.add_argument('--max-tasks', type=int, default=None,
                        help='Maximum number of tasks to evaluate (for testing)')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Determine paths
    arc_dir = args.arc_dir or config.get('data', {}).get('arc_dir', './data/arc-agi/data')
    output_dir = Path(args.output_dir or config.get('logging', {}).get('output_dir', './outputs'))
    
    # Find checkpoint
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        # Try to find latest checkpoint
        checkpoint_dir = Path(config.get('logging', {}).get('checkpoint_dir', './checkpoints'))
        candidates = list(checkpoint_dir.glob('checkpoint_*.pt'))
        if candidates:
            checkpoint_path = str(max(candidates, key=lambda p: p.stat().st_mtime))
            print(f"Using latest checkpoint: {checkpoint_path}")
    
    # Load tasks
    print(f"Loading tasks from: {arc_dir}/{args.split}")
    tasks = load_tasks(arc_dir, args.split)
    print(f"Loaded {len(tasks)} tasks")
    
    if args.max_tasks:
        tasks = tasks[:args.max_tasks]
        print(f"Limited to {len(tasks)} tasks")
    
    # Load model
    print("Loading model...")
    model = load_model(config, checkpoint_path)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Build ensemble config
    ensemble_config = build_ensemble_config(config)
    
    # Run evaluation
    if args.ablation == 'all':
        results = run_ablation_study(model, tasks, ensemble_config, output_dir)
    elif args.ablation:
        results = evaluate_single_mode(model, tasks, ensemble_config, args.ablation, args.verbose)
    else:
        results = run_full_evaluation(model, tasks, ensemble_config, output_dir, args.verbose)
    
    # Print final summary
    if isinstance(results, dict) and 'accuracy' in results:
        print(f"\n{'='*60}")
        print(f"FINAL RESULT: {results['accuracy']*100:.2f}% accuracy")
        print(f"{'='*60}")
    
    return results


if __name__ == '__main__':
    main()
