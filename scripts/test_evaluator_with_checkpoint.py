#!/usr/bin/env python3
"""
Test TRM-Style Evaluator with Previous Checkpoint

This script tests:
1. Load best.pt checkpoint (extracting non-EMA weights)
2. Optionally add Gumbel noise during eval (for backward compat with old training)
3. Use TRM-style evaluator with inverse augmentation and voting
4. Report exact match on eval dataset

Usage:
    python scripts/test_evaluator_with_checkpoint.py --checkpoint checkpoint/rlan-stable/best.pt
    python scripts/test_evaluator_with_checkpoint.py --checkpoint checkpoint/rlan-stable/best.pt --add-gumbel
"""

import argparse
import sys
import os
import random
from pathlib import Path
from functools import partial
import json

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sci_arc.models.rlan import RLAN, RLANConfig
from sci_arc.data.dataset import ARCDataset, collate_sci_arc
from sci_arc.evaluation.trm_style_evaluator import TRMStyleEvaluator, grid_hash
import yaml


def set_seed(seed=42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_checkpoint(checkpoint_path: str, device: torch.device, use_ema: bool = True):
    """
    Load checkpoint and extract weights.
    
    Args:
        checkpoint_path: Path to checkpoint
        device: Target device
        use_ema: If True, prefer EMA weights. If False, use raw model weights.
        
    Returns:
        state_dict, checkpoint_info dict
    """
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    info = {
        'epoch': checkpoint.get('epoch', 'unknown'),
        'best_exact_match': checkpoint.get('best_exact_match', 'unknown'),
        'train_loss': checkpoint.get('train_loss', 'unknown'),
    }
    
    # Check what's in checkpoint
    keys = list(checkpoint.keys())
    print(f"  Checkpoint keys: {keys[:10]}...")
    
    # Determine state dict to use
    if use_ema and 'ema_state_dict' in checkpoint:
        print("  Using EMA weights from checkpoint")
        state_dict = checkpoint['ema_state_dict']
        info['weight_type'] = 'ema'
    elif 'model_state_dict' in checkpoint:
        print("  Using raw model weights from checkpoint")
        state_dict = checkpoint['model_state_dict']
        info['weight_type'] = 'raw'
    else:
        # Assume it's the state dict directly
        print("  Checkpoint appears to be state dict directly")
        state_dict = checkpoint
        info['weight_type'] = 'direct'
    
    print(f"  Epoch: {info['epoch']}")
    print(f"  Reported best exact match: {info['best_exact_match']}")
    
    return state_dict, info


def create_model(config_path: str = None):
    """Create RLAN model from config."""
    if config_path is None:
        config_path = "configs/rlan_stable.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_cfg = config['model']
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
        dsc_num_heads=model_cfg.get('dsc_num_heads', 4),
        msre_encoding_dim=model_cfg.get('msre_encoding_dim', 32),
        msre_num_freq=model_cfg.get('msre_num_freq', 8),
        lcr_num_freq=model_cfg.get('lcr_num_freq', 8),
        lcr_num_heads=model_cfg.get('lcr_num_heads', 4),
        use_context_encoder=model_cfg.get('use_context_encoder', True),
        use_dsc=model_cfg.get('use_dsc', True),
        use_msre=model_cfg.get('use_msre', True),
        use_lcr=model_cfg.get('use_lcr', False),
        use_sph=model_cfg.get('use_sph', False),
        use_learned_pos=model_cfg.get('use_learned_pos', False),
    )
    
    return RLAN(config=rlan_config), config


def monkey_patch_gumbel_noise(model, add_gumbel: bool = False, temperature: float = 0.5):
    """
    Optionally add Gumbel noise to DSC during eval.
    
    For backward compatibility with old training that used Gumbel noise.
    """
    if not add_gumbel:
        return
    
    # Find DSC module
    dsc = None
    for name, module in model.named_modules():
        if 'dynamic_saliency' in name.lower() or 'dsc' in name.lower():
            dsc = module
            break
    
    if dsc is None:
        print("WARNING: Could not find DSC module for Gumbel patching")
        return
    
    # Store original forward
    original_forward = dsc.forward
    
    def gumbel_forward(*args, **kwargs):
        # Call original forward
        result = original_forward(*args, **kwargs)
        
        # Add Gumbel noise to attention if present
        if isinstance(result, tuple) and len(result) >= 2:
            clue_features, attention = result[0], result[1]
            # Add Gumbel noise to sharpen attention
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(attention) + 1e-20) + 1e-20)
            noisy_attention = (attention + gumbel_noise * temperature)
            noisy_attention = F.softmax(noisy_attention.view(attention.size(0), -1), dim=-1)
            noisy_attention = noisy_attention.view_as(attention)
            return (clue_features, noisy_attention) + result[2:]
        
        return result
    
    dsc.forward = gumbel_forward
    print(f"  ✓ Patched DSC with Gumbel noise (temperature={temperature})")


def load_eval_data(data_path: str = None, max_samples: int = None, use_augmentation: bool = False):
    """Load evaluation dataset."""
    if data_path is None:
        # Try to find evaluation data
        possible_paths = [
            "data/arc-agi/data/evaluation",
            "data/evaluation",
            "data/arc-agi_evaluation_combined.json",
        ]
        for p in possible_paths:
            if os.path.exists(p):
                data_path = p
                break
    
    if data_path is None:
        print("ERROR: Could not find evaluation data")
        return None
    
    print(f"Loading evaluation data from {data_path}...")
    
    # Try to load dataset
    try:
        dataset = ARCDataset(
            data_path=data_path,
            augment=use_augmentation,
            track_augmentation=True,  # Need this for inverse aug
        )
        
        # If max_samples is set, create a subset
        if max_samples is not None and max_samples < len(dataset):
            indices = list(range(min(max_samples, len(dataset))))
            dataset = Subset(dataset, indices)
            print(f"  Using subset of {len(dataset)} samples")
        else:
            print(f"  Loaded {len(dataset)} samples")
        return dataset
    except Exception as e:
        print(f"  Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return None


def evaluate_model(
    model,
    eval_loader,
    device,
    use_voting: bool = True,
):
    """Run TRM-style evaluation."""
    model.eval()
    evaluator = TRMStyleEvaluator(
        pass_Ks=[1, 2, 5],
        use_voting=use_voting,
        pad_value=10,
    )
    
    total_samples = 0
    exact_matches = 0
    pixel_correct = 0
    total_pixels = 0
    dihedral_id_counts = [0] * 8  # Track dihedral usage
    
    print("\nRunning evaluation...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_loader):
            # The ARCDataset batch has different keys:
            # 'test_inputs' (B, H, W) or 'input_grids' 
            # 'test_outputs' (B, H, W) or 'output_grids'
            # 'task_ids', 'aug_stats', 'aug_info'
            
            # Get inputs - try different key names
            if 'test_inputs' in batch:
                inputs = batch['test_inputs'].to(device)
            elif 'input' in batch:
                inputs = batch['input'].to(device)
            else:
                print(f"  WARNING: Could not find input in batch keys: {batch.keys()}")
                continue
            
            # Get targets
            if 'test_outputs' in batch:
                targets = batch['test_outputs'].to(device)
            elif 'target' in batch:
                targets = batch['target'].to(device)
            else:
                print(f"  WARNING: Could not find target in batch keys: {batch.keys()}")
                continue
            
            B = inputs.size(0)
            
            # Get demo pairs for context encoding
            train_inputs = None
            train_outputs = None
            if 'input_grids' in batch and 'output_grids' in batch:
                # Stack input/output pairs for context
                input_grids = batch['input_grids']  # (B, num_pairs, H, W)
                output_grids = batch['output_grids']  # (B, num_pairs, H, W)
                if input_grids.dim() == 4:
                    # Replace -100 padding with 10 (pad color) for one-hot encoding
                    input_grids = torch.where(input_grids < 0, torch.tensor(10), input_grids)
                    output_grids = torch.where(output_grids < 0, torch.tensor(10), output_grids)
                    train_inputs = input_grids.to(device)
                    train_outputs = output_grids.to(device)
            
            # Forward pass - get input in correct format (B, H, W) not one-hot
            try:
                # Model expects (B, H, W) input grid with color indices 0-9
                if inputs.dim() == 3:
                    input_indices = inputs  # Already (B, H, W)
                else:
                    input_indices = inputs.argmax(dim=1)  # (B, C, H, W) -> (B, H, W)
                
                outputs = model(
                    input_grid=input_indices.long(),
                    train_inputs=train_inputs.long() if train_inputs is not None else None,
                    train_outputs=train_outputs.long() if train_outputs is not None else None,
                    return_intermediates=True,
                )
                # Model returns 'logits' not 'pred'
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                predictions = logits.argmax(dim=1)  # (B, num_classes, H, W) -> (B, H, W)
            except Exception as e:
                print(f"  WARNING: Model forward failed: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            # Get confidence
            if 'stop_logits' in outputs:
                stop_probs = torch.sigmoid(outputs['stop_logits'])
                confidence = 1.0 - stop_probs.mean(dim=1)
            else:
                confidence = torch.ones(B, device=device)
            
            # Compute basic metrics
            for i in range(B):
                pred = predictions[i].cpu().numpy()
                target = targets[i].cpu().numpy()
                
                # Extract task ID
                if 'task_ids' in batch:
                    task_ids = batch['task_ids']
                    task_id = task_ids[i] if isinstance(task_ids, (list, tuple)) else str(batch_idx * B + i)
                elif 'task_id' in batch:
                    task_id = batch['task_id'][i] if isinstance(batch['task_id'], (list, tuple)) else str(batch_idx * B + i)
                else:
                    task_id = f"task_{batch_idx * B + i}"
                
                # Get augmentation info - check per-sample aug_info first
                if 'aug_info' in batch and len(batch['aug_info']) > i:
                    aug_info = batch['aug_info'][i]
                elif 'aug_stats' in batch:
                    # Fallback: no per-sample info, use default
                    aug_info = {'dihedral_id': 0, 'color_perm': None}
                else:
                    aug_info = {'dihedral_id': 0, 'color_perm': None}
                
                # Track dihedral usage
                did = aug_info.get('dihedral_id', 0)
                dihedral_id_counts[did] += 1
                
                # Update evaluator
                evaluator.update(
                    task_id=task_id,
                    prediction=pred,
                    ground_truth=target,
                    aug_info=aug_info,
                    confidence=float(confidence[i].cpu()),
                )
                
                # Simple metrics (without inverse aug for comparison)
                mask = (target != -100)  # Non-padding (target uses -100 for padding)
                if mask.any():
                    correct = (pred[mask] == target[mask]).sum()
                    total = mask.sum()
                    pixel_correct += correct
                    total_pixels += total
                    
                    # Exact match (all pixels correct)
                    if correct == total:
                        exact_matches += 1
                
                total_samples += 1
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {total_samples} samples...")
    
    # Log dihedral distribution
    print(f"\n  Dihedral ID distribution: {dihedral_id_counts}")
    print(f"  Non-identity augmentations: {sum(dihedral_id_counts[1:])}/{total_samples}")
    
    # Compute TRM-style metrics
    trm_metrics = evaluator.compute_metrics()
    
    # Compute simple metrics
    simple_metrics = {
        'total_samples': total_samples,
        'exact_matches': exact_matches,
        'exact_match_pct': exact_matches / max(total_samples, 1) * 100,
        'pixel_accuracy': pixel_correct / max(total_pixels, 1) * 100 if total_pixels > 0 else 0,
    }
    
    return trm_metrics, simple_metrics


def main():
    parser = argparse.ArgumentParser(description='Test TRM-Style Evaluator with Checkpoint')
    parser.add_argument('--checkpoint', type=str, default='checkpoint/rlan-stable/best.pt',
                        help='Path to checkpoint file')
    parser.add_argument('--config', type=str, default='configs/rlan_stable.yaml',
                        help='Path to config file')
    parser.add_argument('--data-path', type=str, default=None,
                        help='Path to evaluation data')
    parser.add_argument('--add-gumbel', action='store_true',
                        help='Add Gumbel noise during eval (for old training compatibility)')
    parser.add_argument('--no-ema', action='store_true',
                        help='Use raw model weights instead of EMA')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum samples to evaluate')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for evaluation')
    parser.add_argument('--use-augmentation', action='store_true',
                        help='Apply augmentation during eval')
    parser.add_argument('--no-voting', action='store_true',
                        help='Disable voting in TRM evaluator')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load checkpoint
    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        return
    
    state_dict, checkpoint_info = load_checkpoint(
        args.checkpoint,
        device,
        use_ema=not args.no_ema
    )
    
    # Create model
    model, config = create_model(args.config)
    
    # Load weights
    try:
        model.load_state_dict(state_dict, strict=False)
        print("  ✓ Model weights loaded successfully")
    except Exception as e:
        print(f"  WARNING: Some weights could not be loaded: {e}")
    
    model = model.to(device)
    
    # Optionally add Gumbel noise
    if args.add_gumbel:
        print("\n  Adding Gumbel noise during eval (backward compat mode)...")
        monkey_patch_gumbel_noise(model, add_gumbel=True, temperature=0.5)
    
    # Load eval data
    eval_dataset = load_eval_data(
        args.data_path,
        max_samples=args.max_samples,
        use_augmentation=args.use_augmentation
    )
    
    if eval_dataset is None:
        print("ERROR: Could not load evaluation data")
        return
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_sci_arc,
    )
    
    # Run evaluation
    print("\n" + "="*60)
    print("EVALUATION CONFIGURATION:")
    print("="*60)
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Weight type: {checkpoint_info['weight_type']}")
    print(f"  Add Gumbel: {args.add_gumbel}")
    print(f"  Use voting: {not args.no_voting}")
    print(f"  Use augmentation: {args.use_augmentation}")
    print(f"  Samples: {len(eval_dataset)}")
    print("="*60)
    
    trm_metrics, simple_metrics = evaluate_model(
        model,
        eval_loader,
        device,
        use_voting=not args.no_voting,
    )
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS:")
    print("="*60)
    
    print("\n--- Simple Metrics (No Inverse Augmentation) ---")
    print(f"  Total Samples: {simple_metrics['total_samples']}")
    print(f"  Exact Matches: {simple_metrics['exact_matches']}")
    print(f"  Exact Match %: {simple_metrics['exact_match_pct']:.2f}%")
    print(f"  Pixel Accuracy: {simple_metrics['pixel_accuracy']:.2f}%")
    
    print("\n--- TRM-Style Metrics (With Inverse Aug + Voting) ---")
    for k, v in sorted(trm_metrics.items()):
        print(f"  {k}: {v*100:.2f}%")
    
    print("\n" + "="*60)
    print("SUMMARY:")
    print("="*60)
    print(f"  ★ EXACT MATCH (Simple): {simple_metrics['exact_match_pct']:.2f}%")
    print(f"  ★ PASS@1 (TRM-Style): {trm_metrics.get('pass@1', 0)*100:.2f}%")
    print("="*60)
    
    # Save results
    results = {
        'checkpoint': args.checkpoint,
        'config': {
            'add_gumbel': args.add_gumbel,
            'use_ema': not args.no_ema,
            'use_voting': not args.no_voting,
            'use_augmentation': args.use_augmentation,
        },
        'checkpoint_info': checkpoint_info,
        'simple_metrics': simple_metrics,
        'trm_metrics': {k: float(v) for k, v in trm_metrics.items()},
    }
    
    output_path = Path(args.checkpoint).parent / 'eval_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")
    
    return results


if __name__ == '__main__':
    main()
