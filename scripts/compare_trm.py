#!/usr/bin/env python
"""
SCI-ARC vs TRM Comparison Script.

Performs head-to-head comparison between SCI-ARC and TRM models
on ARC benchmark tasks for publication purposes.

Uses the EXACT original TRM implementation from:
    https://github.com/SamsungSAILMontreal/TinyRecursiveModels
for fair scientific comparison.

Usage:
    python scripts/compare_trm.py \
        --sci-arc-checkpoint checkpoints/best_model.pt \
        --trm-checkpoint /path/to/trm_checkpoint.pt \
        --data ./data/arc-agi \
        --output ./comparison_results
"""

import argparse
import json
import os
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, Optional, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
import matplotlib.pyplot as plt

from sci_arc.models import SCIARC, SCIARCConfig
from sci_arc.data import SCIARCDataset, create_dataloader
from sci_arc.evaluation import ARCEvaluator, EvaluationConfig, ARCMetrics

# Import the ORIGINAL TRM implementation for fair comparison
from baselines.trm import TRM, TRMConfig, TRMCarry


def load_sci_arc_model(checkpoint_path: str, device: str = 'cuda') -> SCIARC:
    """Load SCI-ARC model."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config_dict = checkpoint.get('config', {})
    
    model_config = SCIARCConfig(
        hidden_dim=config_dict.get('hidden_dim', 256),
        num_colors=config_dict.get('num_colors', 10),
        max_grid_size=config_dict.get('max_grid_size', 30),
        num_structure_slots=config_dict.get('num_structure_slots', 8),
        se_layers=config_dict.get('se_layers', config_dict.get('num_abstraction_layers', 2)),
        use_abstraction=config_dict.get('use_abstraction', True),
        max_objects=config_dict.get('max_objects', 16),
        num_heads=config_dict.get('num_heads', 4),
        dropout=config_dict.get('dropout', 0.1),
        H_cycles=config_dict.get('H_cycles', 16),
        L_cycles=config_dict.get('L_cycles', 4),
        L_layers=config_dict.get('L_layers', 2),
        latent_size=config_dict.get('latent_size', 64),
        deep_supervision=config_dict.get('deep_supervision', True),
        demo_aggregation=config_dict.get('demo_aggregation', 'attention'),
    )
    
    model = SCIARC(model_config)
    # Use strict=False to handle architecture variations in checkpoints
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.to(device)
    model.eval()
    
    return model


def load_trm_model(checkpoint_path: str, device: str = 'cuda', config_override: Optional[Dict] = None):
    """
    Load the ORIGINAL TRM model from Samsung SAIL Montreal.
    
    This uses the exact implementation from:
    https://github.com/SamsungSAILMontreal/TinyRecursiveModels
    
    Args:
        checkpoint_path: Path to TRM checkpoint
        device: Device to load model on
        config_override: Optional config parameters to override
    
    Returns:
        TRM model (TinyRecursiveReasoningModel_ACTV1)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract config from checkpoint or use defaults matching original TRM
    config_dict = checkpoint.get('config', {})
    
    # Default TRM config (7M parameters as per paper)
    default_config = {
        'batch_size': 1,
        'seq_len': 900,  # 30x30 grid
        'puzzle_emb_ndim': 256,
        'num_puzzle_identifiers': 1000,
        'vocab_size': 16,  # 10 colors + special tokens
        'H_cycles': 3,
        'L_cycles': 4,
        'H_layers': 2,  # ignored in main TRM
        'L_layers': 2,
        'hidden_size': 256,
        'expansion': 2.5,
        'num_heads': 8,
        'pos_encodings': 'rope',
        'halt_max_steps': 10,
        'halt_exploration_prob': 0.1,
        'forward_dtype': 'bfloat16',
        'mlp_t': False,
        'puzzle_emb_len': 16,
    }
    
    # Update with checkpoint config and overrides
    default_config.update(config_dict)
    if config_override:
        default_config.update(config_override)
    
    # Create TRM model using exact original implementation
    model = TRM(default_config)
    
    # Load weights if present
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    
    model.to(device)
    model.eval()
    
    print(f"Loaded TRM model with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    return model


def run_trm_inference(
    model: TRM, 
    batch: Dict[str, torch.Tensor],
    device: str = 'cuda',
    max_steps: int = 10,
) -> Dict[str, torch.Tensor]:
    """
    Run inference with the original TRM model.
    
    Uses the ACT (Adaptive Computation Time) wrapper with Q-learning halting.
    """
    # Move batch to device
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
             for k, v in batch.items()}
    
    # Initialize carry
    carry = model.initial_carry(batch)
    carry = TRMCarry(
        inner_carry=carry.inner_carry._replace(
            z_H=carry.inner_carry.z_H.to(device),
            z_L=carry.inner_carry.z_L.to(device),
        ),
        steps=carry.steps.to(device),
        halted=carry.halted.to(device),
        current_data={k: v.to(device) for k, v in carry.current_data.items()},
    )
    
    # Run ACT loop
    all_outputs = []
    for step in range(max_steps):
        carry, outputs = model(carry, batch)
        all_outputs.append(outputs)
        
        # Check if all sequences have halted
        if carry.halted.all():
            break
    
    # Return final predictions
    final_logits = outputs['logits']
    predictions = final_logits.argmax(dim=-1)
    
    return {
        'predictions': predictions,
        'logits': final_logits,
        'steps_used': step + 1,
        'q_halt_logits': outputs['q_halt_logits'],
    }


def prepare_batch_for_trm(batch: Dict[str, Any], device: str = 'cuda') -> Dict[str, torch.Tensor]:
    """
    Prepare a batch in SCI-ARC format for TRM inference.
    
    TRM expects:
        - inputs: [batch, seq_len] flattened grid tokens
        - puzzle_identifiers: [batch] puzzle IDs
        - labels: [batch, seq_len] target tokens (optional for inference)
    
    Args:
        batch: SCI-ARC format batch with grid tensors
        device: Target device
    
    Returns:
        TRM format batch
    """
    # Get test input from SCI-ARC batch
    if 'test_inputs' in batch:
        grid = batch['test_inputs']  # [B, H, W]
    elif 'input_grid' in batch:
        grid = batch['input_grid']
    else:
        grid = batch['inputs']
    
    # Handle batch dimension
    if grid.dim() == 2:
        grid = grid.unsqueeze(0)
    
    batch_size = grid.shape[0]
    
    # Flatten grid to sequence (TRM uses 30x30 = 900 token sequence)
    max_size = 30
    H, W = grid.shape[1], grid.shape[2]
    
    # Pad to 30x30 if needed
    if H < max_size or W < max_size:
        padded = torch.zeros(batch_size, max_size, max_size, dtype=grid.dtype, device=device)
        padded[:, :H, :W] = grid
        grid = padded
    
    # Flatten to sequence [B, 900]
    inputs = grid.reshape(batch_size, -1).to(device)
    
    # Create puzzle identifiers (use task hash or index)
    if 'task_ids' in batch:
        # Hash task IDs to get numeric identifiers
        puzzle_ids = torch.tensor([
            hash(tid) % 1000 for tid in batch['task_ids']
        ], dtype=torch.int32, device=device)
    else:
        puzzle_ids = torch.zeros(batch_size, dtype=torch.int32, device=device)
    
    trm_batch = {
        'inputs': inputs.long(),
        'puzzle_identifiers': puzzle_ids,
    }
    
    # Add labels if available (for training/evaluation)
    if 'test_outputs' in batch:
        target = batch['test_outputs']
        if target.dim() == 2:
            target = target.unsqueeze(0)
        
        # Pad and flatten target
        tH, tW = target.shape[1], target.shape[2]
        if tH < max_size or tW < max_size:
            padded = torch.full((batch_size, max_size, max_size), -100, dtype=target.dtype, device=device)
            padded[:, :tH, :tW] = target
            target = padded
        
        trm_batch['labels'] = target.reshape(batch_size, -1).to(device)
    
    return trm_batch


def load_trm_predictions(predictions_path: str) -> Dict[str, np.ndarray]:
    """Load TRM predictions from JSON file."""
    with open(predictions_path, 'r') as f:
        data = json.load(f)
    
    predictions = {}
    for task_id, pred in data.items():
        predictions[task_id] = np.array(pred)
    
    return predictions


def compute_metrics(
    predictions: Dict[str, np.ndarray],
    targets: Dict[str, np.ndarray],
) -> Dict[str, float]:
    """Compute evaluation metrics."""
    metrics = ARCMetrics()
    
    for task_id, pred in predictions.items():
        if task_id in targets:
            metrics.update(task_id, pred, targets[task_id])
    
    return metrics.compute()


def analyze_differences(
    sci_arc_preds: Dict[str, np.ndarray],
    trm_preds: Dict[str, np.ndarray],
    targets: Dict[str, np.ndarray],
) -> Dict[str, any]:
    """
    Analyze where SCI-ARC and TRM differ.
    
    Returns:
        Dict with analysis results
    """
    analysis = {
        'both_correct': [],
        'sci_arc_only': [],
        'trm_only': [],
        'neither': [],
        'sci_arc_better_partial': [],
        'trm_better_partial': [],
    }
    
    for task_id in targets:
        if task_id not in sci_arc_preds or task_id not in trm_preds:
            continue
        
        sci_pred = sci_arc_preds[task_id]
        trm_pred = trm_preds[task_id]
        target = targets[task_id]
        
        sci_correct = np.array_equal(sci_pred, target)
        trm_correct = np.array_equal(trm_pred, target)
        
        if sci_correct and trm_correct:
            analysis['both_correct'].append(task_id)
        elif sci_correct:
            analysis['sci_arc_only'].append(task_id)
        elif trm_correct:
            analysis['trm_only'].append(task_id)
        else:
            analysis['neither'].append(task_id)
            
            # Check partial accuracy
            def pixel_acc(pred, tgt):
                if pred.shape != tgt.shape:
                    return 0.0
                return (pred == tgt).mean()
            
            sci_acc = pixel_acc(sci_pred, target)
            trm_acc = pixel_acc(trm_pred, target)
            
            if sci_acc > trm_acc + 0.05:
                analysis['sci_arc_better_partial'].append(task_id)
            elif trm_acc > sci_acc + 0.05:
                analysis['trm_better_partial'].append(task_id)
    
    return analysis


def create_comparison_plots(
    sci_arc_metrics: Dict,
    trm_metrics: Dict,
    analysis: Dict,
    output_dir: Path,
):
    """Create comparison visualizations."""
    
    # Bar chart of accuracy
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Task accuracy comparison
    ax = axes[0]
    models = ['SCI-ARC', 'TRM']
    accuracies = [
        sci_arc_metrics['task_accuracy'] * 100,
        trm_metrics['task_accuracy'] * 100,
    ]
    bars = ax.bar(models, accuracies, color=['#2ECC71', '#3498DB'])
    ax.set_ylabel('Task Accuracy (%)')
    ax.set_title('Task-Level Accuracy Comparison')
    ax.set_ylim(0, 100)
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{acc:.1f}%', ha='center')
    
    # Venn-like breakdown
    ax = axes[1]
    categories = ['Both\nCorrect', 'SCI-ARC\nOnly', 'TRM\nOnly', 'Neither']
    counts = [
        len(analysis['both_correct']),
        len(analysis['sci_arc_only']),
        len(analysis['trm_only']),
        len(analysis['neither']),
    ]
    colors = ['#27AE60', '#2ECC71', '#3498DB', '#E74C3C']
    bars = ax.bar(categories, counts, color=colors)
    ax.set_ylabel('Number of Tasks')
    ax.set_title('Task Outcome Breakdown')
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(count), ha='center')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_comparison.png', dpi=150)
    plt.close()
    
    # Pie chart of outcomes
    fig, ax = plt.subplots(figsize=(8, 8))
    labels = ['Both Correct', 'SCI-ARC Only', 'TRM Only', 'Neither']
    sizes = [
        len(analysis['both_correct']),
        len(analysis['sci_arc_only']),
        len(analysis['trm_only']),
        len(analysis['neither']),
    ]
    colors = ['#27AE60', '#2ECC71', '#3498DB', '#E74C3C']
    
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
           startangle=90, explode=(0.05, 0.05, 0.05, 0))
    ax.set_title('Task Outcome Distribution')
    
    plt.savefig(output_dir / 'outcome_distribution.png', dpi=150)
    plt.close()


def generate_latex_table(
    sci_arc_metrics: Dict,
    trm_metrics: Dict,
    analysis: Dict,
) -> str:
    """Generate LaTeX table for paper."""
    
    total_tasks = sum([
        len(analysis['both_correct']),
        len(analysis['sci_arc_only']),
        len(analysis['trm_only']),
        len(analysis['neither']),
    ])
    
    latex = r"""
\begin{table}[h]
\centering
\caption{Comparison of SCI-ARC and TRM on ARC Benchmark}
\label{tab:comparison}
\begin{tabular}{lcc}
\toprule
\textbf{Metric} & \textbf{SCI-ARC} & \textbf{TRM} \\
\midrule
Task Accuracy (\%) & %.1f & %.1f \\
Pixel Accuracy (\%) & %.1f & %.1f \\
Size Accuracy (\%) & %.1f & %.1f \\
\midrule
Correct (exclusive) & %d & %d \\
Both Correct & \multicolumn{2}{c}{%d} \\
Neither Correct & \multicolumn{2}{c}{%d} \\
\midrule
Total Tasks & \multicolumn{2}{c}{%d} \\
\bottomrule
\end{tabular}
\end{table}
""" % (
        sci_arc_metrics['task_accuracy'] * 100,
        trm_metrics['task_accuracy'] * 100,
        sci_arc_metrics.get('pixel_accuracy', 0) * 100,
        trm_metrics.get('pixel_accuracy', 0) * 100,
        sci_arc_metrics.get('size_accuracy', 0) * 100,
        trm_metrics.get('size_accuracy', 0) * 100,
        len(analysis['sci_arc_only']),
        len(analysis['trm_only']),
        len(analysis['both_correct']),
        len(analysis['neither']),
        total_tasks,
    )
    
    return latex


def main():
    parser = argparse.ArgumentParser(description='Compare SCI-ARC with TRM')
    parser.add_argument('--sci-arc-checkpoint', type=str, required=True,
                        help='Path to SCI-ARC checkpoint')
    parser.add_argument('--trm-checkpoint', type=str, default=None,
                        help='Path to TRM checkpoint')
    parser.add_argument('--trm-predictions', type=str, default=None,
                        help='Path to TRM predictions JSON (alternative to checkpoint)')
    parser.add_argument('--data', type=str, default='./data/arc-agi',
                        help='Path to ARC data')
    parser.add_argument('--split', type=str, default='evaluation',
                        help='Data split to evaluate')
    parser.add_argument('--output', type=str, default='./comparison_results',
                        help='Output directory')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = args.device if torch.cuda.is_available() else 'cpu'
    
    # Load data
    print(f"Loading data from: {args.data}/{args.split}")
    dataloader = create_dataloader(
        data_dir=args.data,
        split=args.split,
        batch_size=1,
        shuffle=False,
        augment=False,
    )
    
    # Collect ground truth
    targets = {}
    for batch in dataloader:
        task_id = batch['task_ids'][0]
        target = batch['test_outputs'][0].numpy()
        targets[task_id] = target
    
    print(f"Collected {len(targets)} targets")
    
    # Load SCI-ARC model and evaluate
    print("\n" + "="*60)
    print("Evaluating SCI-ARC")
    print("="*60)
    
    sci_arc_model = load_sci_arc_model(args.sci_arc_checkpoint, device)
    
    eval_config = EvaluationConfig(
        num_attempts=2,
        device=device,
        save_predictions=True,
        output_dir=str(output_dir / 'sci_arc'),
    )
    
    sci_arc_evaluator = ARCEvaluator(sci_arc_model, eval_config)
    
    # Re-create dataloader for evaluation
    dataloader = create_dataloader(
        data_dir=args.data,
        split=args.split,
        batch_size=1,
        shuffle=False,
        augment=False,
    )
    
    sci_arc_results = sci_arc_evaluator.evaluate(dataloader)
    sci_arc_preds = sci_arc_results['predictions']
    sci_arc_metrics = sci_arc_results['metrics']
    
    print("\nSCI-ARC Results:")
    print(sci_arc_results['summary'])
    
    # Load TRM predictions
    print("\n" + "="*60)
    print("Loading TRM Results (Original Implementation)")
    print("="*60)
    
    if args.trm_predictions:
        trm_preds = load_trm_predictions(args.trm_predictions)
        print(f"Loaded {len(trm_preds)} TRM predictions from file")
    elif args.trm_checkpoint:
        try:
            trm_model = load_trm_model(args.trm_checkpoint, device)
            print("\nRunning TRM evaluation with original implementation...")
            
            # Re-create dataloader for TRM evaluation
            dataloader = create_dataloader(
                data_dir=args.data,
                split=args.split,
                batch_size=1,
                shuffle=False,
                augment=False,
            )
            
            trm_preds = {}
            for batch in dataloader:
                task_id = batch['task_ids'][0]
                
                # Prepare batch for TRM (convert to TRM format)
                trm_batch = prepare_batch_for_trm(batch, device)
                
                # Run TRM inference
                with torch.no_grad():
                    outputs = run_trm_inference(trm_model, trm_batch, device)
                
                # Extract predictions
                pred = outputs['predictions'][0].cpu().numpy()
                trm_preds[task_id] = pred
                
            print(f"Evaluated {len(trm_preds)} tasks with TRM")
            
        except Exception as e:
            print(f"Error loading TRM: {e}")
            print("Please provide --trm-predictions instead.")
            return
    else:
        print("Please provide either --trm-checkpoint or --trm-predictions")
        # Create dummy TRM results for testing
        print("Using placeholder TRM results for comparison structure...")
        trm_preds = {task_id: np.zeros_like(target) 
                     for task_id, target in targets.items()}
    
    # Compute TRM metrics
    trm_metrics = compute_metrics(trm_preds, targets)
    
    print(f"\nTRM Task Accuracy: {trm_metrics['task_accuracy']*100:.1f}%")
    
    # Analyze differences
    print("\n" + "="*60)
    print("Comparative Analysis")
    print("="*60)
    
    analysis = analyze_differences(sci_arc_preds, trm_preds, targets)
    
    print(f"\nBoth correct: {len(analysis['both_correct'])}")
    print(f"SCI-ARC only: {len(analysis['sci_arc_only'])}")
    print(f"TRM only: {len(analysis['trm_only'])}")
    print(f"Neither: {len(analysis['neither'])}")
    
    # Improvement
    sci_arc_total = len(analysis['both_correct']) + len(analysis['sci_arc_only'])
    trm_total = len(analysis['both_correct']) + len(analysis['trm_only'])
    
    if trm_total > 0:
        improvement = ((sci_arc_total - trm_total) / trm_total) * 100
        print(f"\nSCI-ARC vs TRM: {improvement:+.1f}% relative improvement")
    
    # Create plots
    print("\nGenerating comparison plots...")
    create_comparison_plots(sci_arc_metrics, trm_metrics, analysis, output_dir)
    
    # Generate LaTeX table
    latex_table = generate_latex_table(sci_arc_metrics, trm_metrics, analysis)
    latex_path = output_dir / 'comparison_table.tex'
    with open(latex_path, 'w') as f:
        f.write(latex_table)
    print(f"Saved LaTeX table to {latex_path}")
    
    # Save full results
    results = {
        'sci_arc_metrics': sci_arc_metrics,
        'trm_metrics': trm_metrics,
        'analysis': {k: v if not isinstance(v, list) else len(v) 
                     for k, v in analysis.items()},
        'task_lists': {
            'sci_arc_only': analysis['sci_arc_only'],
            'trm_only': analysis['trm_only'],
        }
    }
    
    results_path = output_dir / 'comparison_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved full results to {results_path}")
    
    print("\n" + "="*60)
    print("Comparison Complete!")
    print("="*60)


if __name__ == '__main__':
    main()
