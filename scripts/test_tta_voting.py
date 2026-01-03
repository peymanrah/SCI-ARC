#!/usr/bin/env python3
"""
TTA Voting Consistency Test
============================

This script tests that TTA (Test-Time Augmentation) voting works correctly
on training tasks that the model has solved with exact match. The goal is to
verify that:

1. TTA produces consistent votes across dihedral transforms
2. Inverse transforms are applied correctly
3. The majority vote agrees with the correct answer on known-solved tasks
4. Consensus ratio is high (indicating equivariance is working)

USAGE:
    # From repo root with .venv activated
    python scripts/test_tta_voting.py
    
    # With more tasks
    python scripts/test_tta_voting.py --num-tasks 20
    
    # With specific task IDs
    python scripts/test_tta_voting.py --task-ids "00d62c1b,017c7c7b"

EXPECTED OUTCOMES:
- Tasks that were solved during training should have high consensus (>75%)
- TTA should produce the same final prediction as the majority vote
- Inverse transforms should correctly map back to canonical space

Author: SCI-ARC Team
Date: January 2026
"""

import argparse
import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# =============================================================================
# AUGMENTATION UTILITIES (copied from train_rlan.py for standalone usage)
# =============================================================================

def apply_dihedral(grid: np.ndarray, transform_id: int) -> np.ndarray:
    """
    Apply one of 8 dihedral transforms (D4 group).
    
    MATCHES train_rlan.py definition exactly for consistency:
    Transform IDs:
    0: identity
    1: rotate 90° CCW
    2: rotate 180°
    3: rotate 270° CCW
    4: flip horizontal (fliplr)
    5: flip vertical (flipud)
    6: transpose
    7: anti-transpose (fliplr + rot90)
    """
    if transform_id == 0:
        return grid.copy()  # identity
    elif transform_id == 1:
        return np.rot90(grid, k=1)  # 90° CCW
    elif transform_id == 2:
        return np.rot90(grid, k=2)  # 180°
    elif transform_id == 3:
        return np.rot90(grid, k=3)  # 270° CCW
    elif transform_id == 4:
        return np.fliplr(grid)  # horizontal flip
    elif transform_id == 5:
        return np.flipud(grid)  # vertical flip
    elif transform_id == 6:
        return grid.T  # transpose
    elif transform_id == 7:
        return np.fliplr(np.rot90(grid, k=1))  # anti-transpose
    else:
        return grid.copy()


# Inverse mapping: DIHEDRAL_INVERSE[t] gives the transform that undoes t
# Same as train_rlan.py for consistency
DIHEDRAL_INVERSE = [0, 3, 2, 1, 4, 5, 6, 7]


def inverse_dihedral(grid: np.ndarray, transform_id: int) -> np.ndarray:
    """
    Apply inverse of dihedral transform.
    Uses the same inverse mapping as train_rlan.py.
    """
    return apply_dihedral(grid, DIHEDRAL_INVERSE[transform_id])


def apply_color_permutation(grid: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """Apply color permutation to grid."""
    return perm[grid]


def inverse_color_perm(grid: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """Apply inverse of color permutation."""
    inv_perm = np.zeros(10, dtype=np.int64)
    for i, p in enumerate(perm):
        inv_perm[p] = i
    return inv_perm[grid]


def pad_grid(grid: np.ndarray, max_size: int = 30, pad_value: int = 10, is_target: bool = False) -> np.ndarray:
    """Pad grid to max_size."""
    h, w = grid.shape
    padded = np.full((max_size, max_size), -100 if is_target else pad_value, dtype=np.int64)
    padded[:h, :w] = grid
    return padded


def crop_prediction(pred: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """Crop prediction to target shape."""
    h, w = target_shape
    return pred[:h, :w].copy()


def grid_hash(grid: np.ndarray) -> str:
    """Create a hash of a grid for voting."""
    return grid.tobytes().hex()


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model(checkpoint_path: str, device: torch.device):
    """Load RLAN model from checkpoint."""
    from sci_arc.models import RLAN, RLANConfig
    
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract model config from checkpoint
    if 'model_config' in checkpoint:
        model_config = checkpoint['model_config']
    elif 'config' in checkpoint and 'model' in checkpoint['config']:
        model_config = checkpoint['config']['model']
    else:
        raise ValueError("Could not find model config in checkpoint")
    
    # Create RLANConfig with all fields from checkpoint to match saved model
    rlan_config = RLANConfig(
        hidden_dim=model_config.get('hidden_dim', 256),
        num_classes=model_config.get('num_classes', 11),
        max_clues=model_config.get('max_clues', 5),
        num_solver_steps=model_config.get('num_solver_steps', 7),
        use_act=model_config.get('use_act', False),
        use_hyperlora=model_config.get('use_hyper_lora', model_config.get('use_hyperlora', True)),
        use_context_encoder=model_config.get('use_context_encoder', True),
        use_cross_attention_context=model_config.get('use_cross_attention_context', True),
        spatial_downsample=model_config.get('spatial_downsample', 8),
        use_hpm=model_config.get('use_hpm', False),
        use_dsc=model_config.get('use_dsc', True),
        use_msre=model_config.get('use_msre', True),
        hyperlora_rank=model_config.get('hyperlora_rank', 8),
        hyperlora_scaling=model_config.get('hyperlora_scaling', 1.0),
        # HPM-specific config from checkpoint
        hpm_primitives_per_bank=model_config.get('hpm_primitives_per_bank', 10),
        hpm_levels_per_bank=model_config.get('hpm_levels_per_bank', 3),
        hpm_top_k=model_config.get('hpm_top_k', 2),
        hpm_use_compositional_bank=model_config.get('hpm_use_compositional_bank', True),
        hpm_use_pattern_bank=model_config.get('hpm_use_pattern_bank', True),
        hpm_use_relational_bank=model_config.get('hpm_use_relational_bank', True),
        hpm_use_concept_bank=model_config.get('hpm_use_concept_bank', False),
        hpm_use_procedural_bank=model_config.get('hpm_use_procedural_bank', False),
        hpm_use_instance_bank=model_config.get('hpm_use_instance_bank', False),
    )
    
    model = RLAN(config=rlan_config)
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        raise ValueError("Could not find state dict in checkpoint")
    
    # Remove "module." prefix if present (from DataParallel)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    model.eval()
    
    print(f"  Model loaded: hidden_dim={rlan_config.hidden_dim}, max_clues={rlan_config.max_clues}")
    
    return model


def load_training_tasks(train_path: str, max_tasks: Optional[int] = None) -> List[Dict]:
    """Load training tasks from directory."""
    path = Path(train_path)
    tasks = []
    
    if path.is_dir():
        json_files = sorted(path.glob("*.json"))
        for json_file in json_files[:max_tasks] if max_tasks else json_files:
            with open(json_file) as f:
                task = json.load(f)
            task['task_id'] = json_file.stem
            tasks.append(task)
    else:
        raise ValueError(f"Train path must be a directory: {train_path}")
    
    return tasks


# =============================================================================
# TTA VOTING TEST
# =============================================================================

def run_tta_for_task(
    model,
    task: Dict,
    device: torch.device,
    temperature: float = 0.5,
    num_dihedral: int = 8,
    num_color_perms: int = 1,
    max_size: int = 30,
    verbose: bool = False,
) -> Dict:
    """
    Run TTA voting on a single task and return detailed results.
    
    Returns:
        Dict with:
        - predictions: List of (canonical_pred, dihedral_id, color_perm_idx)
        - vote_counts: Dict mapping grid_hash -> count
        - winner_pred: The prediction with most votes
        - winner_votes: Number of votes for winner
        - total_views: Total number of augmented views
        - consensus_ratio: winner_votes / total_views
        - correct: Whether winner matches ground truth
        - unique_preds: Number of unique predictions
    """
    model.eval()
    
    # Parse task
    train_inputs_np = [np.array(p['input'], dtype=np.int64) for p in task['train']]
    train_outputs_np = [np.array(p['output'], dtype=np.int64) for p in task['train']]
    test_pair = task['test'][0]
    test_input = np.array(test_pair['input'], dtype=np.int64)
    test_output = np.array(test_pair.get('output', []), dtype=np.int64)
    
    has_ground_truth = test_output.size > 0
    num_pairs = len(train_inputs_np)
    max_pairs = 10
    
    total_views = num_dihedral * num_color_perms
    
    # =========================================================================
    # Prepare all augmented views
    # =========================================================================
    batch_test_inputs = []
    batch_train_inputs = []
    batch_train_outputs = []
    batch_pair_masks = []
    aug_infos = []  # (dihedral_id, color_perm, expected_shape)
    
    for dihedral_id in range(num_dihedral):
        for color_idx in range(num_color_perms):
            # Color permutation
            color_perm = None
            color_aug_train_in = train_inputs_np
            color_aug_train_out = train_outputs_np
            color_aug_test_in = test_input
            
            if color_idx > 0:
                color_perm = np.arange(10, dtype=np.int64)
                color_perm[1:] = np.random.permutation(9) + 1
                color_aug_train_in = [apply_color_permutation(g, color_perm) for g in train_inputs_np]
                color_aug_train_out = [apply_color_permutation(g, color_perm) for g in train_outputs_np]
                color_aug_test_in = apply_color_permutation(test_input, color_perm)
            
            # Dihedral transform
            aug_train_in = [apply_dihedral(g, dihedral_id) for g in color_aug_train_in]
            aug_train_out = [apply_dihedral(g, dihedral_id) for g in color_aug_train_out]
            aug_test_in = apply_dihedral(color_aug_test_in, dihedral_id)
            
            # Expected output shape in augmented space - use TEST OUTPUT shape after dihedral
            # (not train output shape, since test output may differ)
            aug_test_out_shape = apply_dihedral(test_output, dihedral_id).shape if has_ground_truth else aug_train_out[0].shape
            expected_shape = aug_test_out_shape
            
            # Pad grids
            # BUG FIX: train_outputs are CONTEXT (fed to ContextEncoder), NOT supervised targets
            # They must be padded with PAD_COLOR=10, not -100 (which is for loss masking only)
            train_in_padded = [pad_grid(g, max_size, is_target=False) for g in aug_train_in]
            train_out_padded = [pad_grid(g, max_size, is_target=False) for g in aug_train_out]  # FIX: is_target=False
            test_in_padded = pad_grid(aug_test_in, max_size, is_target=False)
            
            # Build tensors
            input_grids = torch.stack([torch.from_numpy(g) for g in train_in_padded])
            output_grids = torch.stack([torch.from_numpy(g) for g in train_out_padded])
            test_input_t = torch.from_numpy(test_in_padded)
            
            # Pair mask
            pair_mask = torch.zeros(max_pairs, dtype=torch.bool)
            pair_mask[:num_pairs] = True
            
            # Pad to max_pairs
            if num_pairs < max_pairs:
                pad_in = input_grids[0:1].expand(max_pairs - num_pairs, -1, -1)
                pad_out = output_grids[0:1].expand(max_pairs - num_pairs, -1, -1)
                input_grids = torch.cat([input_grids, pad_in], dim=0)
                output_grids = torch.cat([output_grids, pad_out], dim=0)
            
            batch_test_inputs.append(test_input_t)
            batch_train_inputs.append(input_grids)
            batch_train_outputs.append(output_grids)
            batch_pair_masks.append(pair_mask)
            aug_infos.append((dihedral_id, color_perm, expected_shape, color_idx))
    
    # =========================================================================
    # Batched forward pass
    # =========================================================================
    batch_test_inputs = torch.stack(batch_test_inputs).to(device)
    batch_train_inputs = torch.stack(batch_train_inputs).to(device)
    batch_train_outputs = torch.stack(batch_train_outputs).to(device)
    batch_pair_masks = torch.stack(batch_pair_masks).to(device)
    
    with torch.no_grad():
        outputs = model(
            batch_test_inputs,
            train_inputs=batch_train_inputs,
            train_outputs=batch_train_outputs,
            pair_mask=batch_pair_masks,
            temperature=temperature,
            return_intermediates=True,
        )
    
    logits = outputs['logits']  # (V, C, H, W)
    preds = logits.argmax(dim=1).cpu().numpy()  # (V, H, W)
    
    # =========================================================================
    # Inverse transforms and voting
    # =========================================================================
    predictions = []
    vote_counts = {}
    
    for view_idx, (dihedral_id, color_perm, expected_shape, color_idx) in enumerate(aug_infos):
        pred = preds[view_idx]
        
        # Crop to expected shape
        pred_cropped = crop_prediction(pred, expected_shape)
        
        # Inverse dihedral
        pred_canonical = inverse_dihedral(pred_cropped, dihedral_id)
        
        # Inverse color permutation
        if color_perm is not None:
            pred_canonical = inverse_color_perm(pred_canonical, color_perm)
        
        predictions.append({
            'pred': pred_canonical,
            'dihedral_id': dihedral_id,
            'color_idx': color_idx,
        })
        
        # Vote
        h = grid_hash(pred_canonical)
        if h not in vote_counts:
            vote_counts[h] = {'count': 0, 'grid': pred_canonical}
        vote_counts[h]['count'] += 1
    
    # Find winner
    ranked = sorted(vote_counts.values(), key=lambda x: x['count'], reverse=True)
    winner = ranked[0]
    
    # Check correctness
    correct = False
    if has_ground_truth:
        if winner['grid'].shape == test_output.shape:
            correct = np.array_equal(winner['grid'], test_output)
    
    # Consensus analysis
    consensus_ratio = winner['count'] / total_views
    
    # Per-dihedral vote distribution (for debugging)
    dihedral_votes = defaultdict(int)
    for pred_info in predictions:
        h = grid_hash(pred_info['pred'])
        winner_hash = grid_hash(winner['grid'])
        if h == winner_hash:
            dihedral_votes[pred_info['dihedral_id']] += 1
    
    result = {
        'predictions': predictions,
        'vote_counts': {h: v['count'] for h, v in vote_counts.items()},
        'winner_pred': winner['grid'],
        'winner_votes': winner['count'],
        'total_views': total_views,
        'consensus_ratio': consensus_ratio,
        'correct': correct,
        'has_ground_truth': has_ground_truth,
        'ground_truth': test_output if has_ground_truth else None,
        'unique_preds': len(vote_counts),
        'dihedral_votes': dict(dihedral_votes),
    }
    
    if verbose:
        print(f"  Task {task.get('task_id', 'unknown')}:")
        print(f"    Unique predictions: {result['unique_preds']}")
        print(f"    Winner votes: {result['winner_votes']}/{result['total_views']} ({result['consensus_ratio']:.1%})")
        print(f"    Correct: {result['correct']}")
        print(f"    Dihedral vote distribution: {dict(dihedral_votes)}")
    
    return result


def run_tta_voting_test(
    model,
    tasks: List[Dict],
    device: torch.device,
    temperature: float = 0.5,
    num_dihedral: int = 8,
    verbose: bool = False,
) -> Dict:
    """
    Run TTA voting test on multiple tasks.
    """
    print(f"\n{'='*60}")
    print("TTA VOTING CONSISTENCY TEST")
    print(f"{'='*60}")
    print(f"  Tasks to test: {len(tasks)}")
    print(f"  Temperature: {temperature}")
    print(f"  Dihedral transforms: {num_dihedral}")
    print()
    
    results = []
    
    for i, task in enumerate(tasks):
        task_id = task.get('task_id', f'task_{i}')
        
        try:
            result = run_tta_for_task(
                model, task, device, temperature=temperature,
                num_dihedral=num_dihedral, num_color_perms=1, verbose=verbose
            )
            result['task_id'] = task_id
            results.append(result)
            
            # Progress
            if (i + 1) % 10 == 0 or i == len(tasks) - 1:
                print(f"  Processed {i + 1}/{len(tasks)} tasks...")
                
        except Exception as e:
            print(f"  ERROR on task {task_id}: {e}")
            import traceback
            traceback.print_exc()
    
    # =========================================================================
    # Aggregate statistics
    # =========================================================================
    correct_count = sum(1 for r in results if r['correct'])
    tasks_with_gt = sum(1 for r in results if r['has_ground_truth'])
    
    consensus_ratios = [r['consensus_ratio'] for r in results]
    mean_consensus = np.mean(consensus_ratios) if consensus_ratios else 0.0
    min_consensus = np.min(consensus_ratios) if consensus_ratios else 0.0
    max_consensus = np.max(consensus_ratios) if consensus_ratios else 0.0
    
    unique_pred_counts = [r['unique_preds'] for r in results]
    mean_unique = np.mean(unique_pred_counts) if unique_pred_counts else 0.0
    
    # High consensus tasks (>75%)
    high_consensus_count = sum(1 for r in results if r['consensus_ratio'] >= 0.75)
    
    # Perfect consensus tasks (100%)
    perfect_consensus_count = sum(1 for r in results if r['consensus_ratio'] == 1.0)
    
    # Low consensus tasks (<50%)
    low_consensus_tasks = [(r['task_id'], r['consensus_ratio'], r['unique_preds']) 
                          for r in results if r['consensus_ratio'] < 0.5]
    
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    
    if tasks_with_gt > 0:
        print(f"\n  Accuracy: {correct_count}/{tasks_with_gt} ({100*correct_count/tasks_with_gt:.1f}%)")
    else:
        print(f"\n  Accuracy: N/A (no ground truth available)")
    
    print(f"\n  Consensus Statistics:")
    print(f"    Mean consensus:    {mean_consensus:.1%}")
    print(f"    Min consensus:     {min_consensus:.1%}")
    print(f"    Max consensus:     {max_consensus:.1%}")
    print(f"    High (≥75%):       {high_consensus_count}/{len(results)} ({100*high_consensus_count/len(results) if results else 0:.1f}%)")
    print(f"    Perfect (100%):    {perfect_consensus_count}/{len(results)} ({100*perfect_consensus_count/len(results) if results else 0:.1f}%)")
    
    print(f"\n  Prediction Diversity:")
    print(f"    Mean unique preds: {mean_unique:.2f}")
    
    if low_consensus_tasks:
        print(f"\n  Low Consensus Tasks (<50%):")
        for task_id, cons, unique in low_consensus_tasks[:10]:
            print(f"    - {task_id}: {cons:.1%} consensus, {unique} unique preds")
        if len(low_consensus_tasks) > 10:
            print(f"    ... and {len(low_consensus_tasks) - 10} more")
    
    # =========================================================================
    # Dihedral consistency check
    # =========================================================================
    print(f"\n  Dihedral Transform Consistency:")
    
    # Get actual num_dihedral from first result (handles variable num_dihedral)
    actual_num_dihedral = results[0]['total_views'] if results else 8
    
    # Count tasks where all transforms vote for same prediction
    all_same = sum(1 for r in results if r['unique_preds'] == 1)
    print(f"    All {actual_num_dihedral} transforms agree: {all_same}/{len(results)} ({100*all_same/len(results) if results else 0:.1f}%)")
    
    # Check which transforms most often disagree
    transform_disagree_count = defaultdict(int)
    for r in results:
        if r['unique_preds'] > 1:
            # Check which transforms didn't vote for winner
            winner_hash = grid_hash(r['winner_pred'])
            for pred_info in r['predictions']:
                if grid_hash(pred_info['pred']) != winner_hash:
                    transform_disagree_count[pred_info['dihedral_id']] += 1
    
    if transform_disagree_count:
        print(f"    Disagreement by transform ID:")
        for d_id in range(8):
            count = transform_disagree_count.get(d_id, 0)
            print(f"      D{d_id}: {count} disagreements")
    
    print(f"\n{'='*60}")
    
    # Return summary for programmatic use
    return {
        'num_tasks': len(results),
        'accuracy': correct_count / tasks_with_gt if tasks_with_gt > 0 else None,
        'mean_consensus': mean_consensus,
        'min_consensus': min_consensus,
        'max_consensus': max_consensus,
        'high_consensus_ratio': high_consensus_count / len(results) if results else 0,
        'perfect_consensus_ratio': perfect_consensus_count / len(results) if results else 0,
        'mean_unique_preds': mean_unique,
        'all_tasks_results': results,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Test TTA voting consistency on training tasks")
    
    parser.add_argument(
        '--checkpoint',
        default='./checkpoints/rlan_stable_dev/latest.pt',
        help='Path to model checkpoint',
    )
    parser.add_argument(
        '--train-path',
        default='./data/arc-agi/data/training',
        help='Path to training tasks directory',
    )
    parser.add_argument(
        '--num-tasks',
        type=int,
        default=10,
        help='Number of tasks to test (default: 10)',
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run on ALL training tasks (400 tasks) instead of sampling',
    )
    parser.add_argument(
        '--task-ids',
        type=str,
        default=None,
        help='Comma-separated list of specific task IDs to test',
    )
    parser.add_argument(
        '--num-dihedral',
        type=int,
        default=8,
        help='Number of dihedral transforms (1-8, default: 8)',
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.5,
        help='DSC attention temperature (default: 0.5)',
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed per-task results',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility',
    )
    
    args = parser.parse_args()
    
    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Device
    device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Load model
    if not Path(args.checkpoint).exists():
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        return 1
    
    model = load_model(args.checkpoint, device)
    
    # Load tasks
    if not Path(args.train_path).exists():
        print(f"ERROR: Training path not found: {args.train_path}")
        return 1
    
    all_tasks = load_training_tasks(args.train_path)
    print(f"Loaded {len(all_tasks)} training tasks")
    
    # Filter to specific tasks if requested
    if args.task_ids:
        task_ids = set(args.task_ids.split(','))
        tasks = [t for t in all_tasks if t.get('task_id') in task_ids]
        if not tasks:
            print(f"ERROR: No tasks found matching IDs: {args.task_ids}")
            return 1
        print(f"Selected {len(tasks)} specific tasks: {[t['task_id'] for t in tasks]}")
    elif args.all:
        # Run ALL training tasks
        tasks = all_tasks
        print(f"Running on ALL {len(tasks)} training tasks")
    else:
        # Random sample
        import random
        random.seed(args.seed)
        tasks = random.sample(all_tasks, min(args.num_tasks, len(all_tasks)))
        print(f"Sampled {len(tasks)} random tasks")
    
    # Run test
    summary = run_tta_voting_test(
        model, tasks, device,
        temperature=args.temperature,
        num_dihedral=args.num_dihedral,
        verbose=args.verbose,
    )
    
    # Final verdict
    print(f"\n{'='*60}")
    print("TEST VERDICT")
    print(f"{'='*60}")
    
    issues = []
    
    if summary['mean_consensus'] < 0.5:
        issues.append(f"Mean consensus too low: {summary['mean_consensus']:.1%} (expected ≥50%)")
    
    if summary['high_consensus_ratio'] < 0.3:
        issues.append(f"Too few high-consensus tasks: {summary['high_consensus_ratio']:.1%} (expected ≥30%)")
    
    if summary['accuracy'] is not None and summary['accuracy'] < 0.1:
        issues.append(f"Accuracy very low: {summary['accuracy']:.1%} (expected ≥10% on training set)")
    
    if issues:
        print("  ⚠️  ISSUES DETECTED:")
        for issue in issues:
            print(f"    - {issue}")
        print("\n  This may indicate problems with:")
        print("    - Inverse transform implementation")
        print("    - Equivariance loss (always ~0 bug)")
        print("    - Model overfitting without learning dihedral equivariance")
    else:
        print("  ✅ TTA VOTING APPEARS FUNCTIONAL")
        print(f"    - Mean consensus: {summary['mean_consensus']:.1%}")
        if summary['accuracy'] is not None:
            print(f"    - Accuracy on test set: {summary['accuracy']:.1%}")
    
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
