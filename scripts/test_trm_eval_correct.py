#!/usr/bin/env python
"""
PROPER TRM-Style Evaluation Test for RLAN Checkpoints.

This script implements TRM evaluation CORRECTLY:

1. Apply AUGMENTATION during eval (just like training)
2. Get predictions in AUGMENTED space
3. Apply INVERSE AUGMENTATION to predictions
4. Compare inverse-transformed predictions to ORIGINAL ground truth
5. VOTE across multiple augmented versions of same task

TRM Key Insight:
- TRM pre-generates 1000 augmented versions of each puzzle
- During eval, it runs inference on ALL augmented versions
- It inverse-transforms each prediction back to original space
- It votes to find consensus prediction

RLAN Approach:
- We run eval N times with random augmentation
- Each time, we track dihedral_id
- We inverse-transform predictions
- We vote across all predictions for same task

Usage:
    python scripts/test_trm_eval_correct.py --checkpoint checkpoint/rlan-stable/best.pt
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any
import hashlib

import torch
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sci_arc.data.dataset import ARCDataset, collate_sci_arc, inverse_dihedral_transform, DIHEDRAL_INVERSE
from sci_arc.models.rlan import RLAN, RLANConfig


def grid_hash(grid: np.ndarray) -> str:
    """Compute unique hash for a grid (for voting)."""
    if grid.ndim != 2:
        return ""
    grid = grid.astype(np.uint8)
    buffer = [x.to_bytes(1, byteorder='big') for x in grid.shape]
    buffer.append(grid.tobytes())
    return hashlib.sha256(b"".join(buffer)).hexdigest()


def crop_to_content(grid: np.ndarray, pad_value: int = -100) -> np.ndarray:
    """Crop grid to remove padding, keeping only content."""
    if grid.ndim != 2:
        return grid
    
    # Find content (non-padding) region
    # Use both -100 (target padding) and 10 (input padding)
    content_mask = (grid != pad_value) & (grid != 10) & (grid >= 0) & (grid <= 9)
    
    if not content_mask.any():
        return np.array([[0]], dtype=np.int64)
    
    rows = np.any(content_mask, axis=1)
    cols = np.any(content_mask, axis=0)
    
    if not rows.any() or not cols.any():
        return np.array([[0]], dtype=np.int64)
    
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    cropped = grid[rmin:rmax+1, cmin:cmax+1].copy()
    # Clamp to valid colors
    cropped = np.clip(cropped, 0, 9)
    return cropped


def load_checkpoint(checkpoint_path: str, device: str = 'cpu'):
    """Load model from checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get config
    config_dict = checkpoint.get('config', {})
    
    # Build model config - RLANConfig uses different param names
    model_config = RLANConfig(
        hidden_dim=config_dict.get('hidden_dim', 256),
        num_colors=config_dict.get('num_colors', 11),
        max_grid_size=config_dict.get('max_grid_size', 30),
        num_solver_steps=config_dict.get('H_cycles', 16),
        dropout=config_dict.get('dropout', 0.1),
        max_clues=config_dict.get('dsc_num_clues', 32),
        
        # RLAN module flags - map from checkpoint to RLANConfig
        use_context_encoder=config_dict.get('enable_context_encoder', True),
        use_dsc=config_dict.get('enable_dsc', True),
        use_lcr=config_dict.get('enable_lcr', False),
        use_sph=config_dict.get('enable_sph', False),
        use_act=config_dict.get('enable_act', False),
        use_msre=config_dict.get('enable_msre', True),
    )
    
    model = RLAN(model_config)
    
    # Load weights
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    
    epoch = checkpoint.get('epoch', 'unknown')
    best_acc = checkpoint.get('best_accuracy', 'unknown')
    print(f"  Epoch: {epoch}, Best accuracy: {best_acc}")
    
    return model, checkpoint


def run_eval_pass(
    model,
    dataset,
    device: str,
    add_gumbel: bool = False,
) -> Dict[str, List[Tuple[np.ndarray, float, int, np.ndarray]]]:
    """
    Run one evaluation pass with augmentation.
    
    Returns:
        Dict mapping task_id -> list of (inverse_pred, confidence, dihedral_id, original_target)
    """
    from torch.utils.data import DataLoader
    
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_sci_arc,
        num_workers=0,
    )
    
    results = defaultdict(list)
    
    with torch.no_grad():
        for batch in loader:
            task_id = batch['task_ids'][0]
            
            # Get augmentation info
            aug_info = batch.get('aug_info', [{}])[0]
            dihedral_id = aug_info.get('dihedral_id', 0)
            
            # Move to device
            input_grids = batch['input_grids'].to(device)
            output_grids = batch['output_grids'].to(device)
            test_input = batch['test_inputs'].to(device)
            test_target = batch['test_outputs'].to(device)
            
            # Forward pass
            outputs = model(
                input_grids=input_grids,
                output_grids=output_grids,
                test_input=test_input,
                test_output=test_target,
            )
            
            # Get prediction
            logits = outputs['logits']  # [B, H, W, C]
            pred = logits.argmax(dim=-1)[0].cpu().numpy()  # [H, W]
            
            # Confidence from stop probability or entropy
            confidence = 1.0
            if 'stop_prob' in outputs:
                confidence = outputs['stop_prob'].mean().item()
            
            # Get target in ORIGINAL space (before augmentation)
            # The target is already in augmented space, we need to inverse it too
            target = test_target[0].cpu().numpy()
            
            # Crop both to content
            pred_cropped = crop_to_content(pred, pad_value=-100)
            target_cropped = crop_to_content(target, pad_value=-100)
            
            # Apply inverse dihedral to BOTH pred and target to get to original space
            if dihedral_id != 0:
                pred_canonical = inverse_dihedral_transform(pred_cropped, dihedral_id)
                target_canonical = inverse_dihedral_transform(target_cropped, dihedral_id)
            else:
                pred_canonical = pred_cropped
                target_canonical = target_cropped
            
            results[task_id].append((pred_canonical, confidence, dihedral_id, target_canonical))
    
    return results


def vote_predictions(
    all_results: List[Dict[str, List[Tuple[np.ndarray, float, int, np.ndarray]]]]
) -> Dict[str, Tuple[np.ndarray, np.ndarray, bool]]:
    """
    Vote across multiple evaluation passes to get consensus prediction.
    
    Returns:
        Dict mapping task_id -> (voted_pred, target, is_correct)
    """
    # Aggregate all predictions per task
    task_predictions = defaultdict(list)
    task_targets = {}
    
    for pass_results in all_results:
        for task_id, predictions in pass_results.items():
            for pred, conf, did, target in predictions:
                task_predictions[task_id].append((pred, conf))
                task_targets[task_id] = target  # All targets should be same after inverse
    
    # Vote for each task
    final_results = {}
    for task_id, predictions in task_predictions.items():
        target = task_targets[task_id]
        
        # Count votes by hash
        vote_counts = defaultdict(lambda: {'count': 0, 'conf_sum': 0.0, 'grid': None})
        for pred, conf in predictions:
            h = grid_hash(pred)
            vote_counts[h]['count'] += 1
            vote_counts[h]['conf_sum'] += conf
            vote_counts[h]['grid'] = pred
        
        # Pick winner by vote count, then confidence
        winner = max(vote_counts.values(), 
                     key=lambda x: (x['count'], x['conf_sum']))
        voted_pred = winner['grid']
        
        # Check correctness
        is_correct = np.array_equal(voted_pred, target)
        
        final_results[task_id] = (voted_pred, target, is_correct)
    
    return final_results


def main():
    parser = argparse.ArgumentParser(description='Proper TRM-Style Evaluation')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data-dir', type=str, default='data/arc-agi/data/evaluation')
    parser.add_argument('--num-passes', type=int, default=8,
                        help='Number of evaluation passes with different augmentations')
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--add-gumbel', action='store_true')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    
    device = args.device
    print(f"Using device: {device}")
    
    # Load model
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    
    # Optionally add Gumbel noise
    if args.add_gumbel:
        print("\n  Adding Gumbel noise during eval...")
        if hasattr(model, 'dsc') and model.dsc is not None:
            model.dsc._use_gumbel_backward_compat = True
            print(f"  ✓ Patched DSC with Gumbel noise")
    
    print(f"\n{'='*60}")
    print("TRM-STYLE EVALUATION (CORRECT)")
    print(f"{'='*60}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Num passes: {args.num_passes}")
    print(f"  Add Gumbel: {args.add_gumbel}")
    print(f"{'='*60}\n")
    
    # Run multiple passes with augmentation
    all_results = []
    
    for pass_idx in range(args.num_passes):
        print(f"Pass {pass_idx + 1}/{args.num_passes}...")
        
        # Load dataset WITH augmentation (different random augmentation each pass)
        dataset = ARCDataset(
            args.data_dir,
            augment=True,  # KEY: Apply augmentation!
            color_permutation=False,  # Skip color perm for now
            translational_augment=False,  # Skip translation for simplicity
            track_augmentation=True,  # Track dihedral_id
        )
        
        if args.max_samples:
            # Subset
            dataset.tasks = dataset.tasks[:args.max_samples]
        
        # Run eval
        pass_results = run_eval_pass(model, dataset, device, args.add_gumbel)
        all_results.append(pass_results)
        
        # Quick stats for this pass
        correct = sum(1 for preds in pass_results.values() 
                      for pred, _, _, tgt in preds 
                      if np.array_equal(pred, tgt))
        total = sum(len(preds) for preds in pass_results.values())
        print(f"  Pass {pass_idx + 1}: {correct}/{total} correct ({100*correct/total:.2f}%)")
    
    print(f"\n{'='*60}")
    print("VOTING RESULTS")
    print(f"{'='*60}")
    
    # Vote across passes
    final_results = vote_predictions(all_results)
    
    # Compute metrics
    correct = sum(1 for _, _, is_correct in final_results.values() if is_correct)
    total = len(final_results)
    
    print(f"\n  Total Tasks: {total}")
    print(f"  Exact Matches (after voting): {correct}")
    print(f"  Exact Match %: {100*correct/total:.2f}%")
    
    # Also show pass@1 (no voting, just first prediction per task)
    first_pass = all_results[0]
    pass1_correct = sum(1 for preds in first_pass.values() 
                        for pred, _, _, tgt in preds 
                        if np.array_equal(pred, tgt))
    pass1_total = sum(len(preds) for preds in first_pass.values())
    print(f"\n  Pass@1 (no voting): {pass1_correct}/{pass1_total} ({100*pass1_correct/pass1_total:.2f}%)")
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  ★ EXACT MATCH (with voting): {100*correct/total:.2f}%")
    print(f"  ★ PASS@1 (no voting): {100*pass1_correct/pass1_total:.2f}%")
    print(f"{'='*60}")
    
    # Save results
    checkpoint_dir = Path(args.checkpoint).parent
    results_path = checkpoint_dir / 'trm_eval_correct_results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'checkpoint': args.checkpoint,
            'num_passes': args.num_passes,
            'add_gumbel': args.add_gumbel,
            'total_tasks': total,
            'exact_matches_voted': correct,
            'exact_match_pct_voted': 100*correct/total,
            'pass1_correct': pass1_correct,
            'pass1_total': pass1_total,
            'pass1_pct': 100*pass1_correct/pass1_total,
        }, f, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == '__main__':
    main()
