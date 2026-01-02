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

from sci_arc.models import RLAN, RLANConfig
from sci_arc.data import ARCDataset
from sci_arc.data.dataset import collate_sci_arc  # CRITICAL: Use same collate as training!
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


def apply_color_permutation(grid: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    """Apply color permutation to grid. perm[old_color] = new_color.
    
    CRITICAL: Handles PAD_COLOR=10 and ignore_index=-100 safely by leaving them unchanged.
    """
    # perm is shape (10,) mapping old color to new color (colors 0-9 only)
    result = grid.clone()
    # Only permute valid color values (0-9)
    valid_mask = (grid >= 0) & (grid <= 9)
    result[valid_mask] = perm[grid[valid_mask].long()]
    return result


def inverse_color_permutation(grid: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    """Apply inverse of color permutation.
    
    CRITICAL: Handles PAD_COLOR=10 and ignore_index=-100 safely by leaving them unchanged.
    """
    # Create inverse permutation: inv_perm[new_color] = old_color
    inv_perm = torch.argsort(perm)
    result = grid.clone()
    # Only permute valid color values (0-9)
    valid_mask = (grid >= 0) & (grid <= 9)
    result[valid_mask] = inv_perm[grid[valid_mask].long()]
    return result


def generate_color_permutation(device: torch.device) -> torch.Tensor:
    """Generate random color permutation (color 0=black stays fixed, 1-9 shuffled)."""
    perm = torch.arange(10, device=device)
    # Shuffle colors 1-9 only (keep 0=black fixed)
    shuffled = torch.randperm(9, device=device) + 1
    perm[1:] = shuffled
    return perm


def crop_prediction_torch(pred: torch.Tensor, pad_value: int = 10) -> torch.Tensor:
    """
    Crop prediction to remove padding - matching train_rlan.py's crop_prediction().
    CRITICAL: Must crop BEFORE inverse transform to handle size changes correctly!
    """
    # Handle batch dimension
    squeeze = False
    if pred.dim() == 2:
        pred = pred.unsqueeze(0)
        squeeze = True
    
    batch_size = pred.shape[0]
    cropped = []
    
    for b in range(batch_size):
        p = pred[b]  # (H, W)
        content_mask = (p != pad_value) & (p != -100)
        
        if not content_mask.any():
            cropped.append(torch.zeros(1, 1, dtype=pred.dtype, device=pred.device))
            continue
        
        rows = content_mask.any(dim=1)
        cols = content_mask.any(dim=0)
        
        row_indices = torch.where(rows)[0]
        col_indices = torch.where(cols)[0]
        
        if len(row_indices) == 0 or len(col_indices) == 0:
            cropped.append(torch.zeros(1, 1, dtype=pred.dtype, device=pred.device))
            continue
        
        rmin, rmax = row_indices[0], row_indices[-1]
        cmin, cmax = col_indices[0], col_indices[-1]
        
        cropped.append(p[rmin:rmax+1, cmin:cmax+1])
    
    # Return first if we squeezed
    if squeeze:
        return cropped[0]
    return cropped


def grid_hash_torch(grid: torch.Tensor) -> str:
    """Create a hash of a grid for voting (matching train_rlan.py)."""
    return grid.cpu().numpy().tobytes().hex()


def predict_with_tta(
    model: RLAN,
    input_grid: torch.Tensor,
    train_inputs: Optional[torch.Tensor] = None,
    train_outputs: Optional[torch.Tensor] = None,
    pair_mask: Optional[torch.Tensor] = None,
    num_dihedral: int = 8,
    num_color_perms: int = 4,
    temperature: float = 0.1,
    num_steps_override: Optional[int] = None,
) -> Tuple[torch.Tensor, List[List[Dict[str, Any]]]]:
    """
    Predict with Test-Time Augmentation using dihedral transforms + color permutations.
    
    MATCHES train_rlan.py's evaluate_trm_style() EXACTLY!
    
    OPTIMIZATION (Dec 2025): Batch all augmented views together per sample.
    Instead of V sequential forward passes, we do 1 forward pass with B=V.
    This gives ~V× speedup (typically 32× for 8 dihedral × 4 color).
    
    TRM-style TTA order (Dec 2025):
    1. Apply color permutation FIRST to ALL grids (test + train context)
    2. Apply dihedral transform SECOND
    3. Get prediction
    4. CROP prediction to remove padding  <-- CRITICAL!
    5. Apply inverse dihedral FIRST
    6. Apply inverse color permutation SECOND
    7. Vote using hash-based aggregation (not pixel-wise)
    
    Args:
        num_dihedral: Number of dihedral transforms (1-8, default 8 = full D4 group)
        num_color_perms: Number of color permutations per dihedral (default 4)
        
    Returns:
        (voted_predictions, ranked_candidates_list) where:
        - voted_predictions: (B, H_max, W_max) tensor of voted predictions
        - ranked_candidates_list: List of length B, each containing ranked candidates for Pass@K
    """
    device = input_grid.device
    total_views = num_dihedral * num_color_perms
    batch_size = input_grid.shape[0]
    
    # Process each sample in the batch with batched TTA
    all_predictions_per_sample = [[] for _ in range(batch_size)]
    
    for b in range(batch_size):
        # Extract single sample
        sample_test = input_grid[b:b+1]  # (1, H, W)
        sample_train_in = train_inputs[b:b+1] if train_inputs is not None else None  # (1, K, H, W)
        sample_train_out = train_outputs[b:b+1] if train_outputs is not None else None  # (1, K, H, W)
        sample_pair_mask = pair_mask[b:b+1] if pair_mask is not None else None  # (1, K)
        
        # ================================================================
        # BATCHED VIEW PREPARATION: Build all augmented views at once
        # ================================================================
        batch_test_views = []      # Will be (V, H, W)
        batch_train_in_views = []  # Will be (V, K, H, W)
        batch_train_out_views = [] # Will be (V, K, H, W)
        batch_pair_mask_views = [] # Will be (V, K)
        aug_infos = []             # (color_perm, tid) for inverse transforms
        
        for color_idx in range(num_color_perms):
            # Generate color permutation (or identity if color_idx == 0)
            if color_idx == 0:
                color_perm = torch.arange(10, device=device)  # Identity
            else:
                color_perm = generate_color_permutation(device)
            
            # Apply color permutation to test input and training context
            color_test = apply_color_permutation(sample_test, color_perm)
            color_train_in = apply_color_permutation(sample_train_in, color_perm) if sample_train_in is not None else None
            color_train_out = apply_color_permutation(sample_train_out, color_perm) if sample_train_out is not None else None
            
            for tid in range(num_dihedral):
                # Apply dihedral transform to color-permuted grids
                transformed_test = apply_dihedral_transform(color_test, tid)
                transformed_train_in = apply_dihedral_transform(color_train_in, tid) if color_train_in is not None else None
                transformed_train_out = apply_dihedral_transform(color_train_out, tid) if color_train_out is not None else None
                
                # Collect for batching (squeeze the batch dim since we'll re-stack)
                batch_test_views.append(transformed_test.squeeze(0))  # (H, W)
                if transformed_train_in is not None:
                    batch_train_in_views.append(transformed_train_in.squeeze(0))  # (K, H, W)
                if transformed_train_out is not None:
                    batch_train_out_views.append(transformed_train_out.squeeze(0))  # (K, H, W)
                if sample_pair_mask is not None:
                    batch_pair_mask_views.append(sample_pair_mask.squeeze(0))  # (K,)
                
                aug_infos.append((color_perm, tid))
        
        # ================================================================
        # BATCHED FORWARD PASS: All views in one call
        # ================================================================
        batch_test_views = torch.stack(batch_test_views, dim=0)  # (V, H, W)
        batch_train_in_views = torch.stack(batch_train_in_views, dim=0) if batch_train_in_views else None  # (V, K, H, W)
        batch_train_out_views = torch.stack(batch_train_out_views, dim=0) if batch_train_out_views else None  # (V, K, H, W)
        batch_pair_mask_views = torch.stack(batch_pair_mask_views, dim=0) if batch_pair_mask_views else None  # (V, K)
        
        with torch.no_grad():
            if getattr(model, 'use_best_step_selection', False):
                # Use best-step selection (entropy-based)
                outputs = model(
                    batch_test_views,
                    train_inputs=batch_train_in_views,
                    train_outputs=batch_train_out_views,
                    pair_mask=batch_pair_mask_views,
                    temperature=temperature,
                    return_intermediates=True,
                    num_steps_override=num_steps_override,
                )
                all_logits = outputs.get('all_logits')
                if all_logits and len(all_logits) > 1:
                    best_logits, _, _ = model.solver.select_best_step_by_entropy(all_logits)
                    logits = best_logits
                else:
                    logits = outputs['logits']
            else:
                logits = model(
                    batch_test_views,
                    train_inputs=batch_train_in_views,
                    train_outputs=batch_train_out_views,
                    pair_mask=batch_pair_mask_views,
                    temperature=temperature,
                    num_steps_override=num_steps_override,
                )
            preds = logits.argmax(dim=1)  # (V, H, W)
        
        # ================================================================
        # INVERSE TRANSFORMS AND COLLECT (sequential per view)
        # ================================================================
        for v, (color_perm, tid) in enumerate(aug_infos):
            pred = preds[v]  # (H, W)
            
            # 1. Crop prediction to remove padding (BEFORE inverse transform!)
            pred_cropped = crop_prediction_torch(pred, pad_value=10)
            
            # 2. Inverse dihedral FIRST
            pred_inv_dihedral = inverse_dihedral_transform(pred_cropped, tid)
            
            # 3. Inverse color permutation SECOND
            pred_canonical = inverse_color_permutation(pred_inv_dihedral, color_perm)
            
            # Store as numpy for hash-based voting
            all_predictions_per_sample[b].append(pred_canonical.cpu().numpy())
    
    # Vote SEPARATELY for each sample (matching train_rlan.py)
    winner_tensors = []
    ranked_candidates_list = []
    max_h, max_w = 1, 1  # Track max dims for final tensor
    
    for b in range(batch_size):
        sample_preds = all_predictions_per_sample[b]
        
        # Hash-based voting (matching train_rlan.py EXACTLY)
        vote_counts = {}  # {hash: {'count': int, 'grid': np.array}}
        
        for pred in sample_preds:
            h = pred.tobytes().hex()
            if h not in vote_counts:
                vote_counts[h] = {'count': 0, 'grid': pred}
            vote_counts[h]['count'] += 1
        
        # Rank predictions by vote count (descending)
        ranked_preds = sorted(vote_counts.values(), key=lambda x: x['count'], reverse=True)
        
        # Build ranked candidates list for Pass@K
        ranked_candidates = []
        for rank, p in enumerate(ranked_preds):
            ranked_candidates.append({
                'rank': rank + 1,
                'grid': p['grid'],
                'count': p['count'],
                'frequency': p['count'] / total_views,
            })
        ranked_candidates_list.append(ranked_candidates)
        
        # Winner is top-ranked prediction
        winner_grid = ranked_preds[0]['grid'] if ranked_preds else np.array([[0]])
        winner_tensors.append(torch.from_numpy(winner_grid).to(device))
        
        # Track max dimensions
        max_h = max(max_h, winner_grid.shape[0])
        max_w = max(max_w, winner_grid.shape[1])
    
    # Pad all winners to same size and stack (using pad_value=10 for consistency)
    padded_winners = []
    for wt in winner_tensors:
        h, w = wt.shape
        if h < max_h or w < max_w:
            padded = torch.full((max_h, max_w), 10, dtype=wt.dtype, device=device)
            padded[:h, :w] = wt
            padded_winners.append(padded)
        else:
            padded_winners.append(wt)
    
    # Stack into batch tensor
    predictions = torch.stack(padded_winners, dim=0)  # (B, max_H, max_W)
    
    return predictions, ranked_candidates_list


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


# Padding constants (must match sci_arc/data/dataset.py)
PAD_COLOR = 10  # Input grid padding
PADDING_IGNORE_VALUE = -100  # Target grid padding (ignore_index for loss)


def trim_grid(grid: np.ndarray, is_target: bool = True) -> np.ndarray:
    """
    Trim padding from grid by finding actual content bounds.
    
    Handles two padding schemes:
    - Target grids: padded with -100 (PADDING_IGNORE_VALUE)
    - Input grids: padded with 10 (PAD_COLOR)
    
    Args:
        grid: The grid to trim
        is_target: If True, treats -100 as padding; if False, treats 10 as padding
    
    Returns:
        Trimmed grid with padding removed
    """
    if grid.size == 0:
        return grid
    
    # Determine padding value based on grid type
    if is_target:
        # Target grids: -100 is padding, 0-9 are valid colors
        content_mask = (grid != PADDING_IGNORE_VALUE)
    else:
        # Input grids: 10 is padding, 0-9 are valid colors
        content_mask = (grid != PAD_COLOR)
    
    # Find rows with any content
    row_mask = np.any(content_mask, axis=1)
    if not row_mask.any():
        # No content found - return minimal grid
        # For all-zeros grid, return the single cell (valid content)
        if is_target and np.any(grid != PADDING_IGNORE_VALUE):
            return grid[:1, :1]
        elif not is_target and np.any(grid != PAD_COLOR):
            return grid[:1, :1]
        return grid[:1, :1]
    
    # Find columns with any content
    col_mask = np.any(content_mask, axis=0)
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
    num_dihedral: int = 8,
    num_color_perms: int = 4,
    output_dir: Optional[Path] = None,
    detailed_output: bool = False,
    visualize: bool = False,
    temperature: float = 0.1,
    num_steps_override: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Evaluate model with comprehensive metrics.
    
    CRITICAL: This function MUST use the same logic as train_rlan.py's evaluate()!
    - Passes train_inputs, train_outputs, pair_mask to model (for context encoder)
    - Uses collate_fn output format: test_inputs (plural), input_grids, output_grids
    
    Returns results matching CISL's evaluate.py output format.
    
    Args:
        temperature: Softmax temperature (should match training end temperature)
        num_steps_override: If provided, run this many solver steps instead of model default
    """
    model.eval()
    
    metrics = ARCMetrics()
    all_details = []
    correct_count = 0
    incorrect_count = 0
    
    # Pass@K tracking (matching train_rlan.py's evaluate_trm_style)
    pass_at_k_correct = {1: 0, 2: 0, 3: 0}
    total_evaluated = 0
    
    print("\nRunning evaluation...")
    
    for batch_idx, batch in enumerate(dataloader):
        # =============================================================
        # CRITICAL: Use same keys as train_rlan.py's evaluate()!
        # =============================================================
        # The collate_fn returns: test_inputs, test_outputs, input_grids, output_grids, grid_masks
        # We MUST pass training context to the model for context encoder to work!
        
        # Handle both collate_fn format (plural) and raw dataset format (singular)
        if 'test_inputs' in batch:
            # Collate format (from collate_fn) - PREFERRED
            test_inputs = batch['test_inputs'].to(device)
            test_outputs = batch['test_outputs'].to(device)
            train_inputs = batch['input_grids'].to(device)
            train_outputs = batch['output_grids'].to(device)
            pair_mask = batch.get('grid_masks')
            if pair_mask is not None:
                pair_mask = pair_mask.to(device)
        else:
            # Raw dataset format (singular) - FALLBACK (less accurate!)
            test_inputs = batch['test_input'].to(device)
            test_outputs = batch['test_output'].to(device)
            # Extract train inputs/outputs from batch if available
            train_inputs = batch.get('input_grids')
            train_outputs = batch.get('output_grids')
            if train_inputs is not None:
                train_inputs = torch.stack([t.to(device) if isinstance(t, torch.Tensor) else torch.tensor(t, device=device) for t in train_inputs])
                train_outputs = torch.stack([t.to(device) if isinstance(t, torch.Tensor) else torch.tensor(t, device=device) for t in train_outputs])
            pair_mask = None
            print(f"[WARN] Using raw dataset format without collate_fn - context may be incomplete!")
        
        task_ids = batch.get('task_ids', batch.get('task_id', [f'task_{batch_idx}_{i}' for i in range(test_inputs.shape[0])]))
        
        if isinstance(task_ids, torch.Tensor):
            task_ids = [str(t.item()) if t.dim() == 0 else str(t.tolist()) for t in task_ids]
        elif not isinstance(task_ids, list):
            task_ids = [str(task_ids)]
        
        with torch.no_grad():
            ranked_candidates = None  # For Pass@K computation
            if use_tta:
                predictions, ranked_candidates = predict_with_tta(
                    model, test_inputs,
                    train_inputs=train_inputs,
                    train_outputs=train_outputs,
                    pair_mask=pair_mask,
                    num_dihedral=num_dihedral,
                    num_color_perms=num_color_perms,
                    temperature=temperature,
                    num_steps_override=num_steps_override
                )
            else:
                # Check if best-step selection is enabled
                if getattr(model, 'use_best_step_selection', False):
                    outputs = model(
                        test_inputs,
                        train_inputs=train_inputs,
                        train_outputs=train_outputs,
                        pair_mask=pair_mask,
                        temperature=temperature,
                        return_intermediates=True,
                        num_steps_override=num_steps_override,
                    )
                    all_logits = outputs['all_logits']
                    if all_logits and len(all_logits) > 1:
                        best_logits, best_step, _ = model.solver.select_best_step_by_entropy(all_logits)
                        logits = best_logits
                    else:
                        logits = outputs['logits']
                else:
                    logits = model(
                        test_inputs,
                        train_inputs=train_inputs,
                        train_outputs=train_outputs,
                        pair_mask=pair_mask,
                        temperature=temperature,
                        num_steps_override=num_steps_override,
                    )
                predictions = logits.argmax(dim=1)
        
        # Process each sample
        batch_size = test_inputs.shape[0]
        for i in range(batch_size):
            task_id = task_ids[i] if i < len(task_ids) else f'task_{batch_idx}_{i}'
            total_evaluated += 1
            
            # Get numpy arrays
            input_np = test_inputs[i].cpu().numpy()
            target_np = test_outputs[i].cpu().numpy()
            pred_np = predictions[i].cpu().numpy()
            
            # Trim padding from target
            target_trimmed = trim_grid(target_np)
            
            # For TTA with ranked candidates, compute Pass@K
            if use_tta and ranked_candidates is not None:
                # Get this sample's ranked candidates
                sample_candidates = ranked_candidates[i] if i < len(ranked_candidates) else []
                
                # Compute Pass@K using ranked candidates (matching train_rlan.py EXACTLY)
                for k in [1, 2, 3]:
                    top_k_preds = sample_candidates[:k]
                    is_in_top_k = any(
                        p['grid'].shape == target_trimmed.shape and 
                        np.array_equal(p['grid'], target_trimmed) 
                        for p in top_k_preds
                    )
                    if is_in_top_k:
                        pass_at_k_correct[k] += 1
            else:
                # Without TTA, Pass@K is same as Pass@1
                pred_trimmed = pred_np[:target_trimmed.shape[0], :target_trimmed.shape[1]]
                is_correct_now = np.array_equal(pred_trimmed, target_trimmed)
                for k in [1, 2, 3]:
                    if is_correct_now:
                        pass_at_k_correct[k] += 1
            
            # Trim prediction for regular metrics
            pred_trimmed = pred_np[:target_trimmed.shape[0], :target_trimmed.shape[1]]
            input_trimmed = trim_grid(input_np, is_target=False)
            
            # Check correctness (Pass@1)
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
                
                input_trimmed = trim_grid(input_np, is_target=False)
                target_trimmed = trim_grid(target_np, is_target=True)
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
    
    # Compute Pass@K rates
    pass_at_k_rates = {}
    if total_evaluated > 0:
        for k in [1, 2, 3]:
            pass_at_k_rates[f'pass@{k}'] = pass_at_k_correct[k] / total_evaluated
    else:
        for k in [1, 2, 3]:
            pass_at_k_rates[f'pass@{k}'] = 0.0
    
    # Build results
    results = {
        'metrics': summary,
        'summary': _format_summary(summary, correct_count, incorrect_count, use_tta, pass_at_k_rates),
        'per_task': metrics.per_task_results,
        'all_details': all_details,
        'correct_count': correct_count,
        'incorrect_count': incorrect_count,
        'pass_at_k': pass_at_k_rates,
        'total_evaluated': total_evaluated,
    }
    
    return results


def _format_summary(summary: Dict, correct: int, incorrect: int, use_tta: bool, pass_at_k: Dict[str, float] = None) -> str:
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
    ]
    
    # Add Pass@K metrics if available
    if pass_at_k:
        lines.append("-" * 40)
        lines.append("Pass@K Metrics (Ranked Candidates):")
        for k in [1, 2, 3]:
            key = f'pass@{k}'
            if key in pass_at_k:
                lines.append(f"  Pass@{k}:               {pass_at_k[key]:.4f} ({pass_at_k[key]*100:.2f}%)")
    
    lines.append("=" * 60)
    
    if use_tta:
        lines.insert(-1, "(with Test-Time Augmentation)")
    
    return '\n'.join(lines)


def load_model(checkpoint_path: str, device: torch.device, config_override: Optional[Dict] = None) -> Tuple[RLAN, Dict]:
    """
    Load RLAN model from checkpoint with full config support.
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model to
        config_override: Optional config dict to override checkpoint config
        
    Returns:
        Tuple of (model, full_config_dict)
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Get full config (not just model section)
    full_config = {}
    if 'config' in checkpoint:
        full_config = checkpoint['config'] if isinstance(checkpoint['config'], dict) else {}
    
    # Override with provided config if any
    if config_override:
        full_config.update(config_override)
    
    # Extract model config section
    model_config = full_config.get('model', {})
    
    # Create model using RLANConfig for proper constructor compatibility
    # FIX (Jan 2026): RLAN.__init__ expects config=RLANConfig, not individual kwargs
    rlan_config = RLANConfig(
        hidden_dim=model_config.get('hidden_dim', 128),
        num_colors=model_config.get('num_colors', 10),
        num_classes=model_config.get('num_classes', 10),
        max_clues=model_config.get('max_clues', 5),
        num_predicates=model_config.get('num_predicates', 8),
        num_solver_steps=model_config.get('num_solver_steps', 6),
        use_act=model_config.get('use_act', False),
        # HyperLoRA config
        use_hyperlora=model_config.get('use_hyperlora', False),
        hyperlora_rank=model_config.get('hyperlora_rank', 8),
        hyperlora_scaling=model_config.get('hyperlora_scaling', 1.0),
        # HPM config
        use_hpm=model_config.get('use_hpm', False),
        hpm_top_k=model_config.get('hpm_top_k', 2),
        hpm_use_instance_bank=model_config.get('hpm_use_instance_bank', False),
        hpm_use_procedural_bank=model_config.get('hpm_use_procedural_bank', False),
        # Solver context
        use_solver_context=model_config.get('use_solver_context', False),
        solver_context_heads=model_config.get('solver_context_heads', 4),
        # Cross-attention
        use_cross_attention_context=model_config.get('use_cross_attention_context', False),
        spatial_downsample=model_config.get('spatial_downsample', 8),
        # DSC/MSRE
        use_dsc=model_config.get('use_dsc', True),
        use_msre=model_config.get('use_msre', True),
    )
    model = RLAN(config=rlan_config)
    
    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    # Load HPM dynamic buffers if present
    if 'hpm_instance_buffer' in checkpoint:
        if hasattr(model, 'hpm_instance_buffer') and model.hpm_instance_buffer is not None:
            try:
                model.hpm_instance_buffer.load_state_dict(checkpoint['hpm_instance_buffer'])
                print(f"  Loaded HPM instance buffer ({len(model.hpm_instance_buffer)} entries)")
            except Exception as e:
                print(f"  Warning: Could not load HPM instance buffer: {e}")
    
    if 'hpm_procedural_buffer' in checkpoint:
        if hasattr(model, 'hpm_procedural_buffer') and model.hpm_procedural_buffer is not None:
            try:
                model.hpm_procedural_buffer.load_state_dict(checkpoint['hpm_procedural_buffer'])
                print(f"  Loaded HPM procedural buffer ({len(model.hpm_procedural_buffer)} entries)")
            except Exception as e:
                print(f"  Warning: Could not load HPM procedural buffer: {e}")
    
    model = model.to(device)
    model.eval()
    
    # Apply inference staging from config with HPM buffer loading and staleness checks
    # This honors hpm_buffer_auto_load, hpm_buffer_path, and hpm_buffer_stale_days from YAML
    try:
        from sci_arc.utils.inference_staging import apply_inference_staging_with_hpm_loading
        staging_results = apply_inference_staging_with_hpm_loading(
            model, full_config, checkpoint=checkpoint, verbose=True
        )
        active_modules = staging_results.get('staging', {})
    except ImportError:
        print("Warning: inference_staging helper not available, using defaults")
        # Fallback: enable all meta-learning modules
        if hasattr(model, 'hyperlora_active'):
            model.hyperlora_active = True
        if hasattr(model, 'use_hpm'):
            model.use_hpm = True
        if hasattr(model, 'hpm_memory_enabled'):
            model.hpm_memory_enabled = True
        if hasattr(model, 'solver_context_active'):
            model.solver_context_active = True
        if hasattr(model, 'cross_attention_active'):
            model.cross_attention_active = True
    
    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    
    return model, full_config


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
    parser.add_argument('--use-tta', action='store_true', default=None,
                        help='Use test-time augmentation (default: from YAML inference.use_tta, or False)')
    parser.add_argument('--no-tta', action='store_true',
                        help='Force disable TTA (overrides YAML)')
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
    parser.add_argument('--temperature', type=float, default=None,
                        help='Temperature for softmax (default: from YAML inference.temperature, or 0.5)')
    
    # =============================================================
    # BEST-STEP SELECTION & SOLVER OVERRIDE (Dec 2025)
    # =============================================================
    parser.add_argument('--use-best-step', action='store_true',
                        help='Enable best-step selection using lowest entropy (most confident step)')
    parser.add_argument('--no-best-step', action='store_true',
                        help='Force disable best-step selection (use last step)')
    parser.add_argument('--num-steps', type=int, default=None,
                        help='Override solver steps (e.g., train with 6, infer with 10)')
    
    # =============================================================
    # TTA CONFIGURATION (Dec 2025)
    # =============================================================
    # Match training evaluation settings for consistent results
    parser.add_argument('--num-dihedral', type=int, default=None,
                        help='Number of dihedral transforms for TTA (default: from YAML, or 8)')
    parser.add_argument('--num-color-perms', type=int, default=None,
                        help='Number of color permutations per dihedral (default: from YAML, or 4)')
    parser.add_argument('--no-color-perms', action='store_true',
                        help='Disable color permutations in TTA (only use dihedral transforms)')
    args = parser.parse_args()
    
    # =========================================================================
    # YAML FALLBACK FOR INFERENCE SETTINGS (Jan 2026 Fix)
    # =========================================================================
    # Load YAML config first to get inference defaults
    # NOTE: If no --config provided, we'll also try checkpoint's embedded config later
    yaml_config = {}
    if args.config:
        with open(args.config, 'r', encoding='utf-8') as f:
            yaml_config = yaml.safe_load(f) or {}
    inference_cfg = yaml_config.get('inference', {})
    
    # Apply YAML defaults when CLI args are None
    if args.temperature is None:
        args.temperature = float(inference_cfg.get('temperature', 0.5))
    if args.num_dihedral is None:
        args.num_dihedral = int(inference_cfg.get('num_dihedral', 8))
    if args.num_color_perms is None:
        args.num_color_perms = int(inference_cfg.get('num_color_perms', 4))
    if args.num_steps is None:
        args.num_steps = inference_cfg.get('num_steps_override', None)
    
    # TTA: CLI takes priority, then YAML, then default False
    # --no-tta forces disable; --use-tta forces enable; otherwise use YAML
    if args.no_tta:
        args.use_tta = False
    elif args.use_tta is None:
        args.use_tta = bool(inference_cfg.get('use_tta', False))
    # else: args.use_tta was explicitly set via --use-tta (True)
    
    # Determine TTA settings (after YAML fallback applied)
    num_color_perms = 1 if args.no_color_perms else args.num_color_perms
    num_dihedral = args.num_dihedral
    total_views = num_dihedral * num_color_perms if args.use_tta else 1
    
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
    if args.use_tta:
        print(f"  Dihedral transforms: {num_dihedral}")
        print(f"  Color permutations: {num_color_perms}")
        print(f"  Total views: {total_views}")
    print(f"Detailed output: {args.detailed_output}")
    print(f"Best-step selection: {'enabled' if args.use_best_step else 'disabled' if args.no_best_step else 'from config'}")
    print(f"Solver steps override: {args.num_steps if args.num_steps else 'None (use model default)'}")
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
    
    # Use the yaml_config we already loaded for inference fallback
    config = yaml_config
    
    # Load model with full config and HPM buffer restoration
    model, full_config = load_model(args.checkpoint, device, config_override=config if config else None)
    
    # =========================================================================
    # POST-LOAD FALLBACK: Use checkpoint's config when no --config provided
    # =========================================================================
    # FIX (Jan 2026): If user didn't pass --config, we should still honor the
    # inference settings that were used during training (stored in checkpoint).
    # This ensures evaluate_rlan.py without --config uses training's temperature.
    # =========================================================================
    if not args.config and full_config:
        ckpt_inference_cfg = full_config.get('inference', {})
        # Only update if we used hardcoded defaults (0.5, 8, 4)
        # These magic numbers match the argparse defaults in YAML fallback section
        if args.temperature == 0.5 and 'temperature' in ckpt_inference_cfg:
            args.temperature = float(ckpt_inference_cfg.get('temperature', 0.5))
            print(f"[CHECKPOINT] Using temperature from checkpoint: {args.temperature}")
        if args.num_dihedral == 8 and 'num_dihedral' in ckpt_inference_cfg:
            args.num_dihedral = int(ckpt_inference_cfg.get('num_dihedral', 8))
        if args.num_color_perms == 4 and 'num_color_perms' in ckpt_inference_cfg:
            args.num_color_perms = int(ckpt_inference_cfg.get('num_color_perms', 4))
        # Recalculate TTA views if updated
        num_color_perms = 1 if args.no_color_perms else args.num_color_perms
        num_dihedral = args.num_dihedral
        total_views = num_dihedral * num_color_perms if args.use_tta else 1
    
    # =============================================================
    # APPLY INFERENCE OVERRIDES (Dec 2025)
    # =============================================================
    # Best-step selection: use entropy to pick most confident step
    if args.use_best_step:
        model.use_best_step_selection = True
        print(f"\n[OVERRIDE] Best-step selection: ENABLED (entropy-based)")
    elif args.no_best_step:
        model.use_best_step_selection = False
        print(f"\n[OVERRIDE] Best-step selection: DISABLED (use last step)")
    else:
        # Use model's saved setting (from checkpoint or config)
        best_step_status = getattr(model, 'use_best_step_selection', False)
        print(f"\n[CONFIG] Best-step selection: {best_step_status}")
    
    # Solver steps override: run more/fewer iterations
    num_steps_override = args.num_steps
    if num_steps_override:
        original_steps = model.solver.num_steps
        print(f"[OVERRIDE] Solver steps: {original_steps} -> {num_steps_override}")
    
    # Log model configuration
    print("\n" + "=" * 60)
    print("Model Configuration:")
    print("=" * 60)
    print(f"  Hidden dim: {model.hidden_dim}")
    print(f"  Num colors: {model.num_colors}")
    print(f"  Num classes: {model.num_classes}")
    print(f"  Solver steps: {model.solver.num_steps} (override: {num_steps_override})")
    print(f"  Best-step selection: {getattr(model, 'use_best_step_selection', False)}")
    print("=" * 60)
    
    # Create dataset
    data_path = args.data_path
    if data_path is None:
        data_path = config.get('data', {}).get('eval_path')
        if data_path is None:
            data_path = './data/arc-agi/data/evaluation'
    
    print(f"\nLoading data from: {data_path}")
    dataset = ARCDataset(data_path, augment=False)
    # CRITICAL: Use same collate_fn as training for proper batch format!
    # This ensures train_inputs/outputs are properly batched for context encoder
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=0,
        collate_fn=collate_sci_arc,  # MUST use for context encoder to work!
    )
    
    print(f"Evaluating on {len(dataset)} tasks")
    
    # Create details directory if needed
    if args.detailed_output:
        details_dir = output_dir / 'detailed_predictions'
        details_dir.mkdir(exist_ok=True)
    else:
        details_dir = None
    
    # Run evaluation
    print(f"\nUsing temperature: {args.temperature}")
    if args.use_tta:
        print(f"TTA: {num_dihedral} dihedral x {num_color_perms} color perms = {total_views} views")
    results = evaluate_model(
        model=model,
        dataloader=dataloader,
        device=device,
        use_tta=args.use_tta,
        num_dihedral=num_dihedral,
        num_color_perms=num_color_perms,
        output_dir=output_dir,
        detailed_output=args.detailed_output,
        visualize=args.visualize,
        temperature=args.temperature,
        num_steps_override=num_steps_override,
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
