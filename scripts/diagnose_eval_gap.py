#!/usr/bin/env python3
"""
Diagnostic Script: Analyze Train/Eval Generalization Gap

This script tests hypotheses for why eval performance is so much worse than training:
1. Gumbel noise: Does adding noise during eval improve attention sharpness?
2. Augmentation: Does augmenting eval data like training help?
3. Combined: Do both together give even better results?

Hypotheses:
- H1: Model relies on Gumbel noise for sharp attention (trained with noise, fails without)
- H2: Model learned augmented patterns, can't recognize canonical (non-augmented) data
- H3: Context encoding is too weak (FiLM bottleneck)

Usage:
    python scripts/diagnose_eval_gap.py --checkpoint checkpoints/rlan_stable/best.pt
"""

import argparse
import sys
import os
import random
from pathlib import Path
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sci_arc.models.rlan import RLAN, RLANConfig
from sci_arc.data.dataset import ARCDataset, collate_sci_arc
import yaml


def set_seed(seed=42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model_and_config(checkpoint_path: str, device: torch.device):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Load config
    config_path = Path(checkpoint_path).parent.parent.parent / "configs" / "rlan_stable.yaml"
    if not config_path.exists():
        config_path = Path("configs/rlan_stable.yaml")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create model config
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
    
    # Create model
    model = RLAN(config=rlan_config)
    
    # Load weights (handle both EMA and regular checkpoints)
    state_dict = checkpoint.get('ema_state_dict', checkpoint.get('model_state_dict', checkpoint))
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    
    print(f"Loaded model from {checkpoint_path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"  Best Exact Match: {checkpoint.get('best_exact_match', 'unknown')}")
    
    return model, config


def compute_attention_entropy(attention_maps: torch.Tensor) -> float:
    """Compute mean entropy of attention maps."""
    B, K, H, W = attention_maps.shape
    attn_flat = attention_maps.view(B, K, -1).clamp(min=1e-6)
    entropy = -(attn_flat * attn_flat.log()).sum(dim=-1)  # (B, K)
    return entropy.mean().item()


def evaluate_with_mode(
    model: RLAN,
    dataloader: DataLoader,
    device: torch.device,
    temperature: float = 1.0,
    force_gumbel_noise: bool = False,
    description: str = "Eval",
) -> dict:
    """
    Evaluate model with specific settings.
    
    Args:
        model: RLAN model
        dataloader: Evaluation dataloader
        device: Device to use
        temperature: Softmax temperature
        force_gumbel_noise: If True, add Gumbel noise even during eval
        description: Description for logging
    """
    model.eval()
    
    total_correct = 0
    total_valid_pixels = 0
    correct_tasks = 0
    total_tasks = 0
    entropy_sum = 0.0
    num_batches = 0
    
    # Monkey-patch DSC to optionally use Gumbel noise during eval
    original_training = model.training
    if force_gumbel_noise and model.dsc is not None:
        # Temporarily set to training mode for Gumbel noise
        model.dsc.train()
    
    with torch.no_grad():
        for batch in dataloader:
            test_inputs = batch['test_inputs'].to(device)
            test_outputs = batch['test_outputs'].to(device)
            train_inputs = batch['input_grids'].to(device)
            train_outputs = batch['output_grids'].to(device)
            pair_mask = batch['grid_masks'].to(device)
            
            # Forward pass
            outputs = model(
                test_inputs,
                train_inputs=train_inputs,
                train_outputs=train_outputs,
                pair_mask=pair_mask,
                temperature=temperature,
                return_intermediates=True,
            )
            
            logits = outputs['logits']
            predictions = logits.argmax(dim=1)
            
            # Compute metrics on valid pixels only
            valid_mask = test_outputs >= 0
            batch_valid_pixels = valid_mask.sum().item()
            
            if batch_valid_pixels > 0:
                correct_mask = (predictions == test_outputs) & valid_mask
                total_correct += correct_mask.sum().item()
                total_valid_pixels += batch_valid_pixels
            
            # Exact match (all valid pixels correct)
            batch_size = test_inputs.shape[0]
            for i in range(batch_size):
                task_valid_mask = test_outputs[i] >= 0
                if task_valid_mask.any():
                    task_correct = ((predictions[i] == test_outputs[i]) | ~task_valid_mask).all()
                    if task_correct:
                        correct_tasks += 1
                total_tasks += 1
            
            # Attention entropy
            if 'attention_maps' in outputs:
                entropy_sum += compute_attention_entropy(outputs['attention_maps'])
                num_batches += 1
    
    # Restore model state
    if force_gumbel_noise and model.dsc is not None:
        model.dsc.eval()
    
    pixel_acc = total_correct / max(total_valid_pixels, 1)
    exact_match = correct_tasks / max(total_tasks, 1)
    mean_entropy = entropy_sum / max(num_batches, 1)
    
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"  Exact Match: {correct_tasks}/{total_tasks} ({exact_match:.1%})")
    print(f"  Pixel Accuracy: {pixel_acc:.1%}")
    print(f"  Attention Entropy: {mean_entropy:.4f}")
    print(f"  Temperature: {temperature}")
    print(f"  Gumbel Noise: {'ON' if force_gumbel_noise else 'OFF'}")
    
    return {
        'exact_match': exact_match,
        'correct_tasks': correct_tasks,
        'total_tasks': total_tasks,
        'pixel_accuracy': pixel_acc,
        'attention_entropy': mean_entropy,
    }


def create_eval_dataloader(
    data_path: str,
    max_size: int = 30,
    batch_size: int = 32,
    augment: bool = False,
    color_permutation: bool = False,
    translational_augment: bool = False,
) -> DataLoader:
    """Create evaluation dataloader with optional augmentation."""
    dataset = ARCDataset(
        data_path,
        max_size=max_size,
        augment=augment,
        color_permutation=color_permutation,
        color_permutation_prob=1.0 if color_permutation else 0.0,
        translational_augment=translational_augment,
        ignore_padding_in_loss=True,
    )
    
    collate_fn = partial(collate_sci_arc, max_grid_size=max_size)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )


def run_experiments(checkpoint_path: str, eval_data_path: str, device: torch.device):
    """Run all diagnostic experiments."""
    
    # Load model
    model, config = load_model_and_config(checkpoint_path, device)
    
    # Get temperature from config (match training)
    temp_start = config.get('training', {}).get('temperature_start', 1.0)
    temp_end = config.get('training', {}).get('temperature_end', 0.5)
    # Use temperature at epoch 5 (linear interpolation)
    epoch = 5
    max_epochs = config.get('training', {}).get('max_epochs', 200)
    temperature = temp_start + (temp_end - temp_start) * min(epoch / max_epochs, 1.0)
    print(f"\nUsing temperature: {temperature:.3f} (epoch {epoch} setting)")
    
    results = {}
    
    print("\n" + "="*70)
    print("EXPERIMENT 1: Baseline (No Noise, No Augmentation)")
    print("="*70)
    eval_loader = create_eval_dataloader(
        eval_data_path, 
        max_size=30, 
        batch_size=32,
        augment=False,
        color_permutation=False,
        translational_augment=False,
    )
    results['baseline'] = evaluate_with_mode(
        model, eval_loader, device, 
        temperature=temperature,
        force_gumbel_noise=False,
        description="Baseline: Standard Eval (no noise, no augment)"
    )
    
    print("\n" + "="*70)
    print("EXPERIMENT 2: With Gumbel Noise (Like Training)")
    print("="*70)
    # Same dataloader, but with Gumbel noise
    results['with_gumbel'] = evaluate_with_mode(
        model, eval_loader, device,
        temperature=temperature,
        force_gumbel_noise=True,
        description="With Gumbel Noise: Eval with training-style noise"
    )
    
    print("\n" + "="*70)
    print("EXPERIMENT 3: With Augmentation (Like Training)")
    print("="*70)
    set_seed(42)  # Reset seed for reproducible augmentation
    aug_loader = create_eval_dataloader(
        eval_data_path,
        max_size=30,
        batch_size=32,
        augment=True,
        color_permutation=True,
        translational_augment=True,
    )
    results['with_augment'] = evaluate_with_mode(
        model, aug_loader, device,
        temperature=temperature,
        force_gumbel_noise=False,
        description="With Augmentation: Eval data augmented like training"
    )
    
    print("\n" + "="*70)
    print("EXPERIMENT 4: With BOTH Gumbel Noise AND Augmentation")
    print("="*70)
    set_seed(42)  # Reset seed for same augmentation
    aug_loader2 = create_eval_dataloader(
        eval_data_path,
        max_size=30,
        batch_size=32,
        augment=True,
        color_permutation=True,
        translational_augment=True,
    )
    results['with_both'] = evaluate_with_mode(
        model, aug_loader2, device,
        temperature=temperature,
        force_gumbel_noise=True,
        description="Both Gumbel + Augmentation: Full training-like eval"
    )
    
    print("\n" + "="*70)
    print("EXPERIMENT 5: Higher Temperature (Softer Attention)")
    print("="*70)
    results['high_temp'] = evaluate_with_mode(
        model, eval_loader, device,
        temperature=1.5,  # Higher than training
        force_gumbel_noise=False,
        description="High Temperature: Softer attention (temp=1.5)"
    )
    
    print("\n" + "="*70)
    print("EXPERIMENT 6: Lower Temperature (Sharper Attention)")
    print("="*70)
    results['low_temp'] = evaluate_with_mode(
        model, eval_loader, device,
        temperature=0.1,  # Very low
        force_gumbel_noise=False,
        description="Low Temperature: Force sharper attention (temp=0.1)"
    )
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY: Comparing All Experiments")
    print("="*70)
    print(f"\n{'Experiment':<40} {'Exact Match':<15} {'Entropy':<12} {'Pixel Acc':<12}")
    print("-"*79)
    
    for name, res in results.items():
        em = f"{res['correct_tasks']}/{res['total_tasks']} ({res['exact_match']:.1%})"
        ent = f"{res['attention_entropy']:.4f}"
        pa = f"{res['pixel_accuracy']:.1%}"
        print(f"{name:<40} {em:<15} {ent:<12} {pa:<12}")
    
    # Analysis
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)
    
    baseline_entropy = results['baseline']['attention_entropy']
    baseline_em = results['baseline']['exact_match']
    
    # Check H1: Gumbel noise hypothesis
    gumbel_entropy = results['with_gumbel']['attention_entropy']
    gumbel_em = results['with_gumbel']['exact_match']
    entropy_drop = baseline_entropy - gumbel_entropy
    em_gain = gumbel_em - baseline_em
    
    print(f"\nH1 (Gumbel Noise Dependency):")
    print(f"   Entropy: {baseline_entropy:.4f} → {gumbel_entropy:.4f} (Δ = {entropy_drop:+.4f})")
    print(f"   Exact Match: {baseline_em:.1%} → {gumbel_em:.1%} (Δ = {em_gain:+.1%})")
    if entropy_drop > 1.0:
        print(f"   ✅ CONFIRMED: Gumbel noise significantly sharpens attention!")
        print(f"      Root cause: Model relies on noise, not discriminative logits")
    elif entropy_drop > 0.1:
        print(f"   🟡 PARTIAL: Gumbel noise helps somewhat")
    else:
        print(f"   ❌ NOT CONFIRMED: Gumbel noise doesn't explain the gap")
    
    # Check H2: Augmentation hypothesis
    aug_entropy = results['with_augment']['attention_entropy']
    aug_em = results['with_augment']['exact_match']
    entropy_drop_aug = baseline_entropy - aug_entropy
    em_gain_aug = aug_em - baseline_em
    
    print(f"\nH2 (Augmentation Mismatch):")
    print(f"   Entropy: {baseline_entropy:.4f} → {aug_entropy:.4f} (Δ = {entropy_drop_aug:+.4f})")
    print(f"   Exact Match: {baseline_em:.1%} → {aug_em:.1%} (Δ = {em_gain_aug:+.1%})")
    if em_gain_aug > 0.05:
        print(f"   ✅ CONFIRMED: Augmentation helps generalization!")
    elif em_gain_aug > 0.01:
        print(f"   🟡 PARTIAL: Augmentation helps slightly")
    else:
        print(f"   ❌ NOT CONFIRMED: Augmentation doesn't explain the gap")
    
    # Check combined effect
    both_entropy = results['with_both']['attention_entropy']
    both_em = results['with_both']['exact_match']
    
    print(f"\nCombined Effect (Gumbel + Augmentation):")
    print(f"   Entropy: {baseline_entropy:.4f} → {both_entropy:.4f}")
    print(f"   Exact Match: {baseline_em:.1%} → {both_em:.1%}")
    
    # Overall conclusion
    print(f"\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    
    if entropy_drop > 1.0:
        print("""
The PRIMARY issue is Gumbel noise dependency.
The model has learned to rely on random noise to select attention locations.
Without noise, attention logits are not discriminative enough.

RECOMMENDED FIX:
1. Add entropy loss on RAW attention logits (before Gumbel noise)
2. Use lower temperature annealing schedule
3. Consider straight-through estimator for hard attention
""")
    elif em_gain_aug > 0.05:
        print("""
The PRIMARY issue is augmentation mismatch.
The model learned patterns specific to augmented data.

RECOMMENDED FIX:
1. Augment eval data during inference
2. Train with more diverse augmentation
3. Consider test-time augmentation (TTA) with ensemble
""")
    else:
        print("""
Neither Gumbel noise nor augmentation explains the gap.
The issue is likely deeper:

RECOMMENDED FIX:
1. Context encoder bottleneck - context is not expressive enough
2. DSC queries are not task-conditioned
3. Need richer context injection (cross-attention, not just FiLM)
""")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Diagnose train/eval generalization gap")
    parser.add_argument(
        '--checkpoint', 
        type=str, 
        default='checkpoints/rlan_stable/best.pt',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--eval-data',
        type=str,
        default='./data/arc-agi/data/evaluation',
        help='Path to evaluation data'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Verify paths exist
    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found at {args.checkpoint}")
        sys.exit(1)
    
    if not os.path.exists(args.eval_data):
        print(f"ERROR: Eval data not found at {args.eval_data}")
        sys.exit(1)
    
    # Run experiments
    run_experiments(args.checkpoint, args.eval_data, device)


if __name__ == '__main__':
    main()
