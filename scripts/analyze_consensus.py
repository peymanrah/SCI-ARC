#!/usr/bin/env python3
"""
Analyze Perfect-Consensus Tasks
Check if the 22 perfect-consensus tasks from the TTA test are correct or consistently wrong.
"""

import json
import sys
import os
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sci_arc.data.arc_dataset import load_arc_data

def load_model(checkpoint_path, device):
    """Load the RLAN model from checkpoint."""
    from sci_arc.models.rlan import RLAN
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})
    
    model_config = config.get('model', {})
    hidden_dim = model_config.get('hidden_dim', 256)
    max_clues = model_config.get('max_clues', 7)
    
    model = RLAN(
        hidden_dim=hidden_dim,
        max_clues=max_clues,
        num_colors=10,
    )
    
    state_dict = checkpoint['model_state_dict']
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    
    return model, config

def apply_dihedral(grid: torch.Tensor, transform_id: int) -> torch.Tensor:
    """Apply dihedral transform to grid."""
    if transform_id == 0:
        return grid
    elif transform_id == 1:
        return torch.rot90(grid, 1, dims=(-2, -1))
    elif transform_id == 2:
        return torch.rot90(grid, 2, dims=(-2, -1))
    elif transform_id == 3:
        return torch.rot90(grid, 3, dims=(-2, -1))
    elif transform_id == 4:
        return torch.flip(grid, dims=[-1])
    elif transform_id == 5:
        return torch.flip(grid, dims=[-2])
    elif transform_id == 6:
        return torch.rot90(torch.flip(grid, dims=[-1]), 1, dims=(-2, -1))
    elif transform_id == 7:
        return torch.rot90(torch.flip(grid, dims=[-1]), 3, dims=(-2, -1))
    return grid

DIHEDRAL_INVERSE = [0, 3, 2, 1, 4, 5, 6, 7]

def pad_to_size(grid, target_h, target_w, value=0, is_target=False):
    """Pad grid to target size."""
    h, w = grid.shape[-2:]
    pad_h = max(0, target_h - h)
    pad_w = max(0, target_w - w)
    if pad_h > 0 or pad_w > 0:
        if is_target:
            # Target: use -1 for padding (ignore in loss)
            padded = torch.full((*grid.shape[:-2], target_h, target_w), -1, dtype=grid.dtype, device=grid.device)
        else:
            padded = torch.full((*grid.shape[:-2], target_h, target_w), value, dtype=grid.dtype, device=grid.device)
        padded[..., :h, :w] = grid
        return padded
    return grid

def run_inference(model, task, device, temperature=0.1, num_dihedral=8):
    """Run inference with TTA and return predictions + ground truth."""
    train_pairs = task.get('train', [])
    test_pairs = task.get('test', [])
    
    if not test_pairs:
        return None, None, None
    
    # Build support set
    max_h = max(max(p['input'].shape[0], p['output'].shape[0]) for p in train_pairs + test_pairs)
    max_w = max(max(p['input'].shape[1], p['output'].shape[1]) for p in train_pairs + test_pairs)
    max_size = max(max_h, max_w, 30)
    
    train_inputs = []
    train_outputs = []
    for pair in train_pairs:
        inp = torch.tensor(pair['input'], dtype=torch.long)
        out = torch.tensor(pair['output'], dtype=torch.long)
        train_inputs.append(pad_to_size(inp, max_size, max_size, 0, is_target=False))
        train_outputs.append(pad_to_size(out, max_size, max_size, -1, is_target=True))
    
    train_inputs = torch.stack(train_inputs).unsqueeze(0).to(device)
    train_outputs = torch.stack(train_outputs).unsqueeze(0).to(device)
    
    test_pair = test_pairs[0]
    test_input = torch.tensor(test_pair['input'], dtype=torch.long)
    test_output = torch.tensor(test_pair['output'], dtype=torch.long) if 'output' in test_pair else None
    
    predictions = []
    
    for d_id in range(num_dihedral):
        # Transform test input
        transformed_input = apply_dihedral(test_input.float(), d_id).long()
        padded_input = pad_to_size(transformed_input, max_size, max_size, 0, is_target=False)
        batch_input = padded_input.unsqueeze(0).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(
                batch_input,
                train_inputs, train_outputs,
                temperature=temperature,
                return_intermediates=True,
            )
        
        logits = outputs['final_logits']
        pred = logits.argmax(dim=-1).squeeze()
        
        # Crop to expected output size
        if test_output is not None:
            exp_h, exp_w = test_output.shape
            # Transform expected shape for non-identity transforms
            if d_id in [1, 3, 6, 7]:  # Rotations that swap h/w
                exp_h, exp_w = exp_w, exp_h
            pred_cropped = pred[:exp_h, :exp_w]
        else:
            pred_cropped = pred
        
        # Apply inverse transform
        inv_id = DIHEDRAL_INVERSE[d_id]
        pred_canonical = apply_dihedral(pred_cropped.float(), inv_id).long()
        predictions.append(pred_canonical.cpu().numpy())
    
    return predictions, test_output.numpy() if test_output is not None else None

def main():
    print("=" * 60)
    print("PERFECT-CONSENSUS TASK ANALYSIS")
    print("=" * 60)
    
    device = torch.device('cpu')
    checkpoint_path = 'checkpoints/rlan_stable_dev/latest.pt'
    
    print(f"\nLoading model from {checkpoint_path}...")
    model, config = load_model(checkpoint_path, device)
    print("  Model loaded")
    
    # Load training tasks
    train_tasks, eval_tasks = load_arc_data('data/arc-agi_training_challenges.json', split_ratio=1.0)
    tasks_by_id = {t['id']: t for t in train_tasks}
    print(f"  Loaded {len(train_tasks)} training tasks")
    
    # Run on all tasks and identify perfect consensus ones
    print("\nRunning inference to identify perfect-consensus tasks...")
    
    perfect_consensus = []
    high_consensus = []
    
    for i, task in enumerate(train_tasks):
        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(train_tasks)} tasks...")
        
        try:
            predictions, ground_truth = run_inference(model, task, device, temperature=0.1, num_dihedral=8)
            
            if predictions is None or ground_truth is None:
                continue
            
            # Check consensus
            unique_preds = set()
            for pred in predictions:
                unique_preds.add(tuple(pred.flatten().tolist()))
            
            num_unique = len(unique_preds)
            consensus = 8 - num_unique + 1  # How many agree with most common
            
            # Get voted prediction (most common)
            from collections import Counter
            pred_hashes = [tuple(p.flatten().tolist()) for p in predictions]
            hash_counts = Counter(pred_hashes)
            most_common_hash = hash_counts.most_common(1)[0][0]
            voted_pred = np.array(most_common_hash).reshape(ground_truth.shape)
            
            is_correct = np.array_equal(voted_pred, ground_truth)
            
            if num_unique == 1:  # Perfect consensus
                perfect_consensus.append({
                    'task_id': task['id'],
                    'is_correct': is_correct,
                    'prediction_shape': voted_pred.shape,
                    'ground_truth_shape': ground_truth.shape,
                })
            elif num_unique <= 3:  # High consensus
                high_consensus.append({
                    'task_id': task['id'],
                    'is_correct': is_correct,
                    'num_unique': num_unique,
                })
        except Exception as e:
            print(f"  Error on task {task['id']}: {e}")
            continue
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    # Perfect consensus analysis
    perfect_correct = sum(1 for t in perfect_consensus if t['is_correct'])
    perfect_wrong = len(perfect_consensus) - perfect_correct
    
    print(f"\nPERFECT CONSENSUS TASKS (all 8 transforms agree): {len(perfect_consensus)}")
    print(f"  Correct: {perfect_correct}/{len(perfect_consensus)} ({100*perfect_correct/len(perfect_consensus) if perfect_consensus else 0:.1f}%)")
    print(f"  Wrong:   {perfect_wrong}/{len(perfect_consensus)} ({100*perfect_wrong/len(perfect_consensus) if perfect_consensus else 0:.1f}%)")
    
    if perfect_consensus:
        print("\n  Correct tasks:")
        for t in perfect_consensus:
            if t['is_correct']:
                print(f"    ✓ {t['task_id']}")
        
        print("\n  Wrong tasks (model is confidently wrong):")
        for t in perfect_consensus:
            if not t['is_correct']:
                print(f"    ✗ {t['task_id']}")
    
    # High consensus analysis
    high_correct = sum(1 for t in high_consensus if t['is_correct'])
    high_wrong = len(high_consensus) - high_correct
    
    print(f"\nHIGH CONSENSUS TASKS (2-3 unique predictions): {len(high_consensus)}")
    print(f"  Correct: {high_correct}/{len(high_consensus)} ({100*high_correct/len(high_consensus) if high_consensus else 0:.1f}%)")
    print(f"  Wrong:   {high_wrong}/{len(high_consensus)} ({100*high_wrong/len(high_consensus) if high_consensus else 0:.1f}%)")
    
    print("\n" + "=" * 60)
    print("DIAGNOSIS")
    print("=" * 60)
    
    if perfect_consensus:
        if perfect_correct == 0:
            print("\n⚠️  ALL perfect-consensus tasks are WRONG")
            print("   → Model produces consistent but incorrect outputs")
            print("   → Training did not teach correct transformations")
        elif perfect_wrong == 0:
            print("\n✓ ALL perfect-consensus tasks are CORRECT")
            print("   → Model works for some subset, but lacks coverage")
        else:
            print(f"\n  Mixed: {perfect_correct} correct, {perfect_wrong} wrong")
            print("   → Model partially learned, but has systematic biases")

if __name__ == "__main__":
    main()
