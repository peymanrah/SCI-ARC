#!/usr/bin/env python3
"""
Test-Time Augmentation (TTA) Evaluator for RLAN.

This script implements proper TTA evaluation that mirrors what the model
saw during training. Since training used augmented samples, we apply
the same augmentations at test time and vote across predictions.

CRITICAL INSIGHT:
- Model was trained with ContextEncoder receiving AUGMENTED demonstrations
- The model learned task patterns from AUGMENTED (input, output) pairs
- At eval, we MUST apply the SAME augmentation to BOTH:
  1. The demonstration pairs (train_inputs, train_outputs) 
  2. The test input
- Then inverse transform the prediction and vote

This gives the model a fair chance by testing it in the same "language"
it was trained on - with the ContextEncoder receiving augmented demonstrations
just like during training.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import hashlib

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sci_arc.models.rlan import RLAN
from sci_arc.data.dataset import ARCDataset, collate_sci_arc
from functools import partial


# =============================================================================
# DIHEDRAL TRANSFORMS (D4 group - 8 symmetries)
# =============================================================================

def apply_dihedral(arr: np.ndarray, tid: int) -> np.ndarray:
    """Apply dihedral transform (8 symmetries of D4 group)."""
    if tid == 0:
        return arr.copy()  # identity
    elif tid == 1:
        return np.rot90(arr, k=1)  # 90° CCW
    elif tid == 2:
        return np.rot90(arr, k=2)  # 180°
    elif tid == 3:
        return np.rot90(arr, k=3)  # 270° CCW
    elif tid == 4:
        return np.fliplr(arr)  # horizontal flip
    elif tid == 5:
        return np.flipud(arr)  # vertical flip
    elif tid == 6:
        return arr.T  # transpose
    elif tid == 7:
        return np.fliplr(np.rot90(arr, k=1))  # anti-transpose
    else:
        return arr.copy()


# Inverse mapping: DIHEDRAL_INVERSE[t] gives the transform that undoes t
DIHEDRAL_INVERSE = [0, 3, 2, 1, 4, 5, 6, 7]


def inverse_dihedral(arr: np.ndarray, tid: int) -> np.ndarray:
    """Apply inverse of dihedral transform."""
    return apply_dihedral(arr, DIHEDRAL_INVERSE[tid])


# =============================================================================
# COLOR PERMUTATION
# =============================================================================

def generate_color_perm() -> np.ndarray:
    """Generate random color permutation (keeping 0=black fixed)."""
    perm = np.arange(10, dtype=np.int64)
    perm[1:] = np.random.permutation(9) + 1  # Shuffle colors 1-9
    return perm


def apply_color_perm(arr: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """Apply color permutation to grid."""
    return perm[arr]


def inverse_color_perm(arr: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """Apply inverse color permutation."""
    inv_perm = np.argsort(perm)
    return inv_perm[arr]


# =============================================================================
# GRID UTILITIES
# =============================================================================

def pad_grid(grid: np.ndarray, max_size: int = 30, 
             offset: Tuple[int, int] = (0, 0),
             pad_value: int = 10,
             is_target: bool = False) -> np.ndarray:
    """Pad grid to max_size with optional offset."""
    h, w = grid.shape
    padded = np.full((max_size, max_size), -100 if is_target else pad_value, dtype=np.int64)
    r, c = offset
    padded[r:r+h, c:c+w] = grid
    return padded


def crop_prediction(pred: np.ndarray, target_shape: tuple = None, pad_value: int = 10) -> np.ndarray:
    """
    Crop prediction to remove padding and match expected output size.
    
    Model predictions are in range 0-9 (valid colors), so we can't rely on
    detecting pad tokens. Instead:
    1. If target_shape is provided, crop to that size
    2. Otherwise, try to find content bounds by excluding common padding indicators
    
    Args:
        pred: Prediction grid (may be 30x30 with padding)
        target_shape: Expected (H, W) shape from ground truth
        pad_value: Padding value to exclude (default 10, though model outputs 0-9)
    
    Returns:
        Cropped prediction grid
    """
    if pred.ndim == 1:
        pred = pred.reshape(30, 30)
    
    # If target shape is provided, simply crop to that size
    if target_shape is not None:
        h, w = target_shape
        return pred[:h, :w]
    
    # Otherwise, try to find content bounds
    # Model predictions are 0-9, so we look for the actual content region
    # by finding rows/cols that have non-zero values (background is usually 0)
    # But this is imperfect since 0 can be valid content
    
    # First try: look for pad_value (10) or -100 which shouldn't appear in valid predictions
    content_mask = (pred != pad_value) & (pred != -100) & (pred >= 0) & (pred <= 9)
    
    if not content_mask.any():
        # All padding or empty - return minimal grid
        return np.array([[0]])
    
    # Find bounding box of content
    rows = np.any(content_mask, axis=1)
    cols = np.any(content_mask, axis=0)
    
    if not rows.any() or not cols.any():
        return np.array([[0]])
    
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    return pred[rmin:rmax+1, cmin:cmax+1]


def grid_hash(grid: np.ndarray) -> str:
    """Compute hash for voting."""
    grid = np.ascontiguousarray(grid.astype(np.uint8))
    return hashlib.sha256(grid.tobytes()).hexdigest()[:16]


# =============================================================================
# TEST-TIME AUGMENTATION EVALUATOR
# =============================================================================

class TTAEvaluator:
    """
    Test-Time Augmentation evaluator with voting.
    
    For each eval sample:
    1. Apply N augmentations (dihedral × color perm)
    2. Run model inference on each augmented version
    3. Inverse transform predictions back to canonical space
    4. Vote across predictions (weighted by confidence if available)
    5. Compare voted prediction with ground truth
    """
    
    def __init__(
        self,
        model: RLAN,
        device: torch.device,
        max_size: int = 30,
        num_dihedral: int = 8,  # Use all 8 dihedral transforms
        num_color_perms: int = 4,  # Random color perms per dihedral
        use_gumbel: bool = False,  # P0.1: DISABLED - Gumbel noise at eval creates train/eval mismatch (16x entropy ratio)
        temperature: float = 0.5,
    ):
        self.model = model
        self.device = device
        self.max_size = max_size
        self.num_dihedral = num_dihedral
        self.num_color_perms = num_color_perms
        self.use_gumbel = use_gumbel
        self.temperature = temperature
        
        self.total_views = num_dihedral * num_color_perms
        print(f"TTA Evaluator: {num_dihedral} dihedral × {num_color_perms} color = {self.total_views} views per sample")
    
    def augment_task(
        self,
        train_inputs: List[np.ndarray],
        train_outputs: List[np.ndarray],
        test_input: np.ndarray,
        dihedral_id: int,
        color_perm: Optional[np.ndarray] = None,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
        """Apply augmentation to entire task."""
        # Apply dihedral
        aug_train_inputs = [apply_dihedral(g, dihedral_id) for g in train_inputs]
        aug_train_outputs = [apply_dihedral(g, dihedral_id) for g in train_outputs]
        aug_test_input = apply_dihedral(test_input, dihedral_id)
        
        # Apply color permutation
        if color_perm is not None:
            aug_train_inputs = [apply_color_perm(g, color_perm) for g in aug_train_inputs]
            aug_train_outputs = [apply_color_perm(g, color_perm) for g in aug_train_outputs]
            aug_test_input = apply_color_perm(aug_test_input, color_perm)
        
        return aug_train_inputs, aug_train_outputs, aug_test_input
    
    def prepare_batch(
        self,
        train_inputs: List[np.ndarray],
        train_outputs: List[np.ndarray],
        test_input: np.ndarray,
    ) -> Dict[str, torch.Tensor]:
        """Prepare tensors for model inference."""
        # Pad grids
        train_inputs_padded = [pad_grid(g, self.max_size, is_target=False) for g in train_inputs]
        train_outputs_padded = [pad_grid(g, self.max_size, is_target=True) for g in train_outputs]
        test_input_padded = pad_grid(test_input, self.max_size, is_target=False)
        
        # Stack and convert to tensors
        input_grids = torch.stack([torch.from_numpy(g) for g in train_inputs_padded])  # (K, H, W)
        output_grids = torch.stack([torch.from_numpy(g) for g in train_outputs_padded])  # (K, H, W)
        test_input_t = torch.from_numpy(test_input_padded)  # (H, W)
        
        # Create pair mask
        num_pairs = len(train_inputs)
        max_pairs = 10  # Max train pairs
        pair_mask = torch.zeros(max_pairs, dtype=torch.bool)
        pair_mask[:num_pairs] = True
        
        # Pad to max_pairs
        if num_pairs < max_pairs:
            pad_input = input_grids[0:1].expand(max_pairs - num_pairs, -1, -1)
            pad_output = output_grids[0:1].expand(max_pairs - num_pairs, -1, -1)
            input_grids = torch.cat([input_grids, pad_input], dim=0)
            output_grids = torch.cat([output_grids, pad_output], dim=0)
        
        return {
            'input_grids': input_grids.unsqueeze(0).to(self.device),  # (1, K, H, W)
            'output_grids': output_grids.unsqueeze(0).to(self.device),  # (1, K, H, W)
            'test_inputs': test_input_t.unsqueeze(0).to(self.device),  # (1, H, W)
            'grid_masks': pair_mask.unsqueeze(0).to(self.device),  # (1, K)
        }
    
    def get_prediction(
        self,
        train_inputs: List[np.ndarray],
        train_outputs: List[np.ndarray],
        test_input: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """Run model and get prediction with confidence."""
        batch = self.prepare_batch(train_inputs, train_outputs, test_input)
        
        with torch.no_grad():
            outputs = self.model(
                batch['test_inputs'],
                train_inputs=batch['input_grids'],
                train_outputs=batch['output_grids'],
                pair_mask=batch['grid_masks'],
                temperature=self.temperature,
                return_intermediates=True,
            )
            
            logits = outputs['logits']  # (1, C, H, W)
            
            # Apply Gumbel noise if enabled (helps with diversity)
            if self.use_gumbel:
                gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
                logits = logits + gumbel_noise * 0.5
            
            pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()  # (H, W)
            
            # Confidence from logits entropy
            probs = F.softmax(logits, dim=1)
            entropy = -(probs * (probs + 1e-10).log()).sum(dim=1).mean()
            confidence = 1.0 / (1.0 + entropy.item())
        
        return pred, confidence
    
    def evaluate_task(
        self,
        task: Dict,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Evaluate a single task using TTA + voting.
        
        Returns:
            (is_correct, details_dict)
        """
        # Parse task
        train_inputs = [np.array(p['input'], dtype=np.int64) for p in task['train']]
        train_outputs = [np.array(p['output'], dtype=np.int64) for p in task['train']]
        test_pair = task['test'][0]
        test_input = np.array(test_pair['input'], dtype=np.int64)
        test_output = np.array(test_pair['output'], dtype=np.int64)
        
        # Collect predictions from all augmented views
        predictions = []  # List of (canonical_pred, confidence)
        
        for dihedral_id in range(self.num_dihedral):
            for color_idx in range(self.num_color_perms):
                # Generate color permutation (or None for first)
                color_perm = generate_color_perm() if color_idx > 0 else None
                
                # Augment task
                aug_train_in, aug_train_out, aug_test_in = self.augment_task(
                    train_inputs, train_outputs, test_input,
                    dihedral_id, color_perm
                )
                
                # Get prediction in augmented space
                pred_aug, confidence = self.get_prediction(
                    aug_train_in, aug_train_out, aug_test_in
                )
                
                # Inverse transform to canonical space
                # Note: crop using expected output shape (which may differ after augmentation)
                # After dihedral transform, the expected shape might be transposed
                aug_out_shape = aug_train_out[0].shape if aug_train_out else test_output.shape
                pred_canonical = crop_prediction(pred_aug, target_shape=aug_out_shape)
                pred_canonical = inverse_dihedral(pred_canonical, dihedral_id)
                if color_perm is not None:
                    pred_canonical = inverse_color_perm(pred_canonical, color_perm)
                
                predictions.append((pred_canonical, confidence))
        
        # Vote across predictions
        vote_counts = defaultdict(lambda: {'count': 0, 'confidence': 0.0, 'grid': None})
        
        for pred, conf in predictions:
            h = grid_hash(pred)
            vote_counts[h]['count'] += 1
            vote_counts[h]['confidence'] += conf
            vote_counts[h]['grid'] = pred
        
        # Find winner (by count, then by confidence)
        winner_hash = max(
            vote_counts.keys(),
            key=lambda h: (vote_counts[h]['count'], vote_counts[h]['confidence'])
        )
        winner_grid = vote_counts[winner_hash]['grid']
        winner_votes = vote_counts[winner_hash]['count']
        
        # Compare with ground truth
        gt_cropped = test_output  # Already in canonical form
        
        # Check if shapes match and values match
        is_correct = (
            winner_grid.shape == gt_cropped.shape and
            np.array_equal(winner_grid, gt_cropped)
        )
        
        details = {
            'num_views': self.total_views,
            'unique_predictions': len(vote_counts),
            'winner_votes': winner_votes,
            'winner_shape': winner_grid.shape,
            'gt_shape': gt_cropped.shape,
            'shapes_match': winner_grid.shape == gt_cropped.shape,
        }
        
        return is_correct, details
    
    def evaluate_dataset(
        self,
        tasks: List[Dict],
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Evaluate all tasks in dataset."""
        correct = 0
        total = len(tasks)
        
        all_details = []
        
        iterator = tqdm(tasks, desc="TTA Evaluation") if verbose else tasks
        
        for task in iterator:
            is_correct, details = self.evaluate_task(task)
            if is_correct:
                correct += 1
            details['task_id'] = task.get('task_id', 'unknown')
            details['correct'] = is_correct
            all_details.append(details)
            
            if verbose:
                iterator.set_postfix({'accuracy': f'{correct}/{len(all_details)} ({100*correct/len(all_details):.1f}%)'})
        
        return {
            'exact_match': correct / total,
            'correct': correct,
            'total': total,
            'details': all_details,
        }


# =============================================================================
# SIMPLE EVALUATOR (for comparison)
# =============================================================================

class SimpleEvaluator:
    """Simple evaluation without TTA (baseline)."""
    
    def __init__(
        self,
        model: RLAN,
        device: torch.device,
        max_size: int = 30,
        use_gumbel: bool = True,
        temperature: float = 0.5,
    ):
        self.model = model
        self.device = device
        self.max_size = max_size
        self.use_gumbel = use_gumbel
        self.temperature = temperature
    
    def evaluate_task(self, task: Dict) -> Tuple[bool, Dict[str, Any]]:
        """Evaluate single task without augmentation."""
        # Parse task
        train_inputs = [np.array(p['input'], dtype=np.int64) for p in task['train']]
        train_outputs = [np.array(p['output'], dtype=np.int64) for p in task['train']]
        test_pair = task['test'][0]
        test_input = np.array(test_pair['input'], dtype=np.int64)
        test_output = np.array(test_pair['output'], dtype=np.int64)
        
        # Prepare batch (same as TTA evaluator)
        train_inputs_padded = [pad_grid(g, self.max_size, is_target=False) for g in train_inputs]
        train_outputs_padded = [pad_grid(g, self.max_size, is_target=True) for g in train_outputs]
        test_input_padded = pad_grid(test_input, self.max_size, is_target=False)
        
        input_grids = torch.stack([torch.from_numpy(g) for g in train_inputs_padded])
        output_grids = torch.stack([torch.from_numpy(g) for g in train_outputs_padded])
        test_input_t = torch.from_numpy(test_input_padded)
        
        num_pairs = len(train_inputs)
        max_pairs = 10
        pair_mask = torch.zeros(max_pairs, dtype=torch.bool)
        pair_mask[:num_pairs] = True
        
        if num_pairs < max_pairs:
            pad_input = input_grids[0:1].expand(max_pairs - num_pairs, -1, -1)
            pad_output = output_grids[0:1].expand(max_pairs - num_pairs, -1, -1)
            input_grids = torch.cat([input_grids, pad_input], dim=0)
            output_grids = torch.cat([output_grids, pad_output], dim=0)
        
        batch = {
            'input_grids': input_grids.unsqueeze(0).to(self.device),
            'output_grids': output_grids.unsqueeze(0).to(self.device),
            'test_inputs': test_input_t.unsqueeze(0).to(self.device),
            'grid_masks': pair_mask.unsqueeze(0).to(self.device),
        }
        
        with torch.no_grad():
            outputs = self.model(
                batch['test_inputs'],
                train_inputs=batch['input_grids'],
                train_outputs=batch['output_grids'],
                pair_mask=batch['grid_masks'],
                temperature=self.temperature,
                return_intermediates=True,
            )
            
            logits = outputs['logits']
            
            if self.use_gumbel:
                gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
                logits = logits + gumbel_noise * 0.5
            
            pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()
        
        # Crop prediction to match expected output size
        pred_cropped = crop_prediction(pred, target_shape=test_output.shape)
        
        is_correct = (
            pred_cropped.shape == test_output.shape and
            np.array_equal(pred_cropped, test_output)
        )
        
        return is_correct, {
            'pred_shape': pred_cropped.shape,
            'gt_shape': test_output.shape,
            'shapes_match': pred_cropped.shape == test_output.shape,
        }
    
    def evaluate_dataset(self, tasks: List[Dict], verbose: bool = True) -> Dict[str, Any]:
        """Evaluate all tasks."""
        correct = 0
        total = len(tasks)
        
        iterator = tqdm(tasks, desc="Simple Evaluation") if verbose else tasks
        
        for task in iterator:
            is_correct, _ = self.evaluate_task(task)
            if is_correct:
                correct += 1
            if verbose:
                iterator.set_postfix({'accuracy': f'{correct}/{total} ({100*correct/total:.1f}%)'})
        
        return {
            'exact_match': correct / total,
            'correct': correct,
            'total': total,
        }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="TTA Evaluation for RLAN")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--eval-path', type=str, default='data/arc-agi_evaluation_challenges.json', help='Eval data path')
    parser.add_argument('--num-dihedral', type=int, default=8, help='Number of dihedral transforms (1-8)')
    parser.add_argument('--num-color-perms', type=int, default=4, help='Color permutations per dihedral')
    parser.add_argument('--use-gumbel', action='store_true', help='Enable Gumbel noise (disabled by default for eval consistency)')
    parser.add_argument('--no-gumbel', action='store_true', help='[Deprecated] Disable Gumbel noise (now default behavior)')
    parser.add_argument('--temperature', type=float, default=0.5, help='DSC temperature')
    parser.add_argument('--compare-simple', action='store_true', help='Also run simple eval for comparison')
    parser.add_argument('--max-tasks', type=int, default=None, help='Limit number of tasks')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint['config']
    
    # Create model with correct parameters matching RLAN constructor
    model_cfg = config['model']
    
    # Build RLANConfig to properly configure all module flags
    from sci_arc.models.rlan import RLANConfig
    rlan_config = RLANConfig(
        hidden_dim=model_cfg.get('hidden_dim', 256),
        num_colors=model_cfg.get('num_colors', 10),
        num_classes=model_cfg.get('num_classes', 10),
        max_grid_size=model_cfg.get('max_grid_size', 30),
        max_clues=model_cfg.get('max_clues', 6),
        num_predicates=model_cfg.get('num_predicates', 32),
        num_solver_steps=model_cfg.get('num_solver_steps', 6),
        use_act=model_cfg.get('use_act', False),
        dropout=model_cfg.get('dropout', 0.1),
        # CRITICAL: Match the module flags from checkpoint
        use_context_encoder=model_cfg.get('use_context_encoder', True),
        use_dsc=model_cfg.get('use_dsc', True),
        use_msre=model_cfg.get('use_msre', True),
        use_lcr=model_cfg.get('use_lcr', False),  # Was False in checkpoint
        use_sph=model_cfg.get('use_sph', False),  # Was False in checkpoint
    )
    model = RLAN(config=rlan_config).to(device)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    print(f"Best accuracy during training: {checkpoint.get('best_accuracy', 'N/A')}")
    
    # Load eval data
    print(f"\nLoading eval data: {args.eval_path}")
    
    # Check if path is a directory (individual JSON files) or single JSON file
    eval_path = Path(args.eval_path)
    tasks = []
    
    if eval_path.is_dir():
        # Load individual JSON files from directory
        json_files = sorted(eval_path.glob('*.json'))
        for json_file in json_files:
            with open(json_file, 'r') as f:
                task = json.load(f)
            task['task_id'] = json_file.stem
            tasks.append(task)
    else:
        # Load single JSON file with all tasks
        with open(args.eval_path, 'r') as f:
            eval_data = json.load(f)
        for task_id, task in eval_data.items():
            task['task_id'] = task_id
            tasks.append(task)
    
    if args.max_tasks:
        tasks = tasks[:args.max_tasks]
    
    print(f"Evaluating {len(tasks)} tasks")
    
    # Gumbel noise: disabled by default for eval consistency (P0.1 fix)
    # --use-gumbel explicitly enables it, --no-gumbel is deprecated (already default)
    use_gumbel = args.use_gumbel and not args.no_gumbel
    if args.no_gumbel and not args.use_gumbel:
        print("Note: --no-gumbel is now default behavior (deprecated flag)")
    if use_gumbel:
        print("Note: Gumbel noise ENABLED (may cause train/eval mismatch)")
    
    # Run simple evaluation first (for comparison)
    if args.compare_simple:
        print("\n" + "="*60)
        print("SIMPLE EVALUATION (no augmentation)")
        print("="*60)
        
        simple_eval = SimpleEvaluator(
            model=model,
            device=device,
            use_gumbel=use_gumbel,
            temperature=args.temperature,
        )
        simple_results = simple_eval.evaluate_dataset(tasks)
        print(f"\nSimple Exact Match: {simple_results['correct']}/{simple_results['total']} ({100*simple_results['exact_match']:.2f}%)")
    
    # Run TTA evaluation
    print("\n" + "="*60)
    print(f"TTA EVALUATION ({args.num_dihedral} dihedral × {args.num_color_perms} color = {args.num_dihedral * args.num_color_perms} views)")
    print("="*60)
    
    tta_eval = TTAEvaluator(
        model=model,
        device=device,
        num_dihedral=args.num_dihedral,
        num_color_perms=args.num_color_perms,
        use_gumbel=use_gumbel,
        temperature=args.temperature,
    )
    tta_results = tta_eval.evaluate_dataset(tasks)
    
    print(f"\nTTA Exact Match: {tta_results['correct']}/{tta_results['total']} ({100*tta_results['exact_match']:.2f}%)")
    
    # Analysis of voting effectiveness
    unique_preds = [d['unique_predictions'] for d in tta_results['details']]
    winner_votes = [d['winner_votes'] for d in tta_results['details']]
    
    print(f"\nVoting Analysis:")
    print(f"  Avg unique predictions per task: {np.mean(unique_preds):.1f}")
    print(f"  Avg winner votes: {np.mean(winner_votes):.1f} / {args.num_dihedral * args.num_color_perms}")
    print(f"  Max agreement: {max(winner_votes)} / {args.num_dihedral * args.num_color_perms}")
    
    # Comparison summary
    if args.compare_simple:
        print("\n" + "="*60)
        print("COMPARISON SUMMARY")
        print("="*60)
        print(f"  Simple Eval:  {simple_results['correct']}/{simple_results['total']} ({100*simple_results['exact_match']:.2f}%)")
        print(f"  TTA Eval:     {tta_results['correct']}/{tta_results['total']} ({100*tta_results['exact_match']:.2f}%)")
        improvement = tta_results['correct'] - simple_results['correct']
        print(f"  Improvement:  {'+' if improvement >= 0 else ''}{improvement} tasks")


if __name__ == '__main__':
    main()
