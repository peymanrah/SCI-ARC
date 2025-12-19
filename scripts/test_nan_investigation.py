#!/usr/bin/env python3
"""
NaN Investigation Test
======================

This script investigates the root cause of NaN occurring around batch 7205
in production training. We test:

1. CPU training (no AMP) - to rule out AMP as the cause
2. GPU with AMP disabled - to isolate AMP
3. GPU with AMP enabled - to replicate production

The key question: Is this an AMP issue, or a numerical instability in the model?

Usage:
    # Activate venv first
    python scripts/test_nan_investigation.py --mode cpu --batches 500
    python scripts/test_nan_investigation.py --mode gpu_no_amp --batches 500
    python scripts/test_nan_investigation.py --mode gpu_amp --batches 500
"""

import sys
import os
import math
import random
import argparse
from pathlib import Path
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import yaml

# Import the actual production code
from sci_arc.models.rlan import RLAN, RLANConfig
from sci_arc.training.rlan_loss import RLANLoss


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_synthetic_batch(batch_size, h, w, num_pairs=3, device='cpu'):
    """Create a synthetic batch that mimics ARC data."""
    # Test inputs/outputs
    test_inputs = torch.randint(0, 10, (batch_size, h, w), device=device)
    test_outputs = torch.randint(0, 10, (batch_size, h, w), device=device)
    
    # Training pairs
    train_inputs = torch.randint(0, 10, (batch_size, num_pairs, h, w), device=device)
    train_outputs = torch.randint(0, 10, (batch_size, num_pairs, h, w), device=device)
    
    # Pair mask (all valid)
    pair_mask = torch.ones(batch_size, num_pairs, dtype=torch.bool, device=device)
    
    return {
        'test_inputs': test_inputs,
        'test_outputs': test_outputs,
        'train_inputs': train_inputs,
        'train_outputs': train_outputs,
        'pair_mask': pair_mask,
    }


def check_model_health(model, prefix=""):
    """Check if model has any NaN/Inf parameters."""
    issues = []
    for name, param in model.named_parameters():
        if not torch.isfinite(param).all():
            issues.append(f"{prefix}{name}: has NaN/Inf")
        if param.grad is not None and not torch.isfinite(param.grad).all():
            issues.append(f"{prefix}{name}.grad: has NaN/Inf")
    return issues


def run_training_simulation(
    mode: str,
    num_batches: int,
    batch_size: int = 8,
    grid_size: int = 15,
    lr: float = 5e-4,
    gradient_clip: float = 1.0,
):
    """
    Run training simulation in specified mode.
    
    Args:
        mode: 'cpu', 'gpu_no_amp', or 'gpu_amp'
        num_batches: Number of batches to simulate
        batch_size: Batch size
        grid_size: Grid size (H=W)
        lr: Learning rate
        gradient_clip: Gradient clipping value
    """
    print("=" * 70)
    print(f"NaN INVESTIGATION - Mode: {mode}")
    print("=" * 70)
    
    # Determine device and AMP settings
    if mode == 'cpu':
        device = torch.device('cpu')
        use_amp = False
    elif mode == 'gpu_no_amp':
        if not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            device = torch.device('cpu')
        else:
            device = torch.device('cuda')
        use_amp = False
    elif mode == 'gpu_amp':
        if not torch.cuda.is_available():
            print("CUDA not available, cannot test AMP")
            return None
        device = torch.device('cuda')
        use_amp = True
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    print(f"Device: {device}")
    print(f"AMP: {use_amp}")
    print(f"Batches: {num_batches}")
    print(f"Batch size: {batch_size}")
    print(f"Grid size: {grid_size}x{grid_size}")
    print()
    
    set_seed(42)
    
    # Load production config
    config_path = project_root / 'configs' / 'rlan_stable.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_cfg = config['model']
    train_cfg = config['training']
    
    # Create model with EXACT production settings
    rlan_config = RLANConfig(
        hidden_dim=model_cfg['hidden_dim'],
        num_colors=model_cfg['num_colors'],
        num_classes=model_cfg['num_classes'],
        max_grid_size=grid_size,  # Smaller for faster testing
        max_clues=model_cfg['max_clues'],
        num_predicates=model_cfg['num_predicates'],
        num_solver_steps=model_cfg['num_solver_steps'],
        dropout=model_cfg['dropout'],
        use_act=model_cfg.get('use_act', False),
        use_context_encoder=model_cfg.get('use_context_encoder', True),
        use_dsc=model_cfg.get('use_dsc', True),
        use_msre=model_cfg.get('use_msre', True),
        use_lcr=model_cfg.get('use_lcr', False),
        use_sph=model_cfg.get('use_sph', False),
        use_learned_pos=model_cfg.get('use_learned_pos', False),
    )
    
    model = RLAN(config=rlan_config).to(device)
    model.train()
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model: {param_count:,} parameters")
    
    # Create loss function with EXACT production settings
    loss_fn = RLANLoss(
        focal_gamma=train_cfg.get('focal_gamma', 2.0),
        focal_alpha=train_cfg.get('focal_alpha', 0.75),
        lambda_entropy=train_cfg.get('lambda_entropy', 0.01),
        lambda_sparsity=train_cfg.get('lambda_sparsity', 0.5),
        lambda_predicate=train_cfg.get('lambda_predicate', 0.01),
        lambda_curriculum=train_cfg.get('lambda_curriculum', 0.0),
        lambda_deep_supervision=train_cfg.get('lambda_deep_supervision', 0.0),
        lambda_act=train_cfg.get('lambda_act', 0.0),
        min_clues=train_cfg.get('min_clues', 2.5),
        min_clue_weight=train_cfg.get('min_clue_weight', 5.0),
        ponder_weight=train_cfg.get('ponder_weight', 0.02),
        entropy_ponder_weight=train_cfg.get('entropy_ponder_weight', 0.02),
        max_clues=model_cfg['max_clues'],
        use_stablemax=train_cfg.get('use_stablemax', True),
        loss_mode=train_cfg.get('loss_mode', 'weighted_stablemax'),
        bg_weight_cap=train_cfg.get('bg_weight_cap', 2.0),
        fg_weight_cap=train_cfg.get('fg_weight_cap', 5.0),
    )
    
    # Create optimizer with EXACT production settings
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=train_cfg.get('weight_decay', 0.01),
        betas=(0.9, 0.95),  # TRM-style
    )
    
    # GradScaler for AMP
    scaler = GradScaler() if use_amp else None
    
    temperature = train_cfg.get('temperature_start', 1.0)
    
    # Training stats
    nan_count = 0
    first_nan_batch = None
    consecutive_nan = 0
    max_consecutive_nan = 10
    loss_history = []
    grad_norm_history = []
    
    print("\nStarting training simulation...")
    print("-" * 70)
    
    for batch_idx in range(num_batches):
        # Create synthetic batch
        batch = create_synthetic_batch(batch_size, grid_size, grid_size, device=device)
        
        optimizer.zero_grad()
        
        # Forward pass
        if use_amp:
            with autocast('cuda'):
                outputs = model(
                    batch['test_inputs'],
                    train_inputs=batch['train_inputs'],
                    train_outputs=batch['train_outputs'],
                    pair_mask=batch['pair_mask'],
                    temperature=temperature,
                    return_intermediates=True,
                )
                
                losses = loss_fn(
                    logits=outputs['logits'],
                    targets=batch['test_outputs'],
                    attention_maps=outputs['attention_maps'],
                    stop_logits=outputs['stop_logits'],
                    predicates=outputs['predicates'],
                    epoch=0,
                    max_epochs=100,
                )
                
                loss = losses['total_loss']
        else:
            outputs = model(
                batch['test_inputs'],
                train_inputs=batch['train_inputs'],
                train_outputs=batch['train_outputs'],
                pair_mask=batch['pair_mask'],
                temperature=temperature,
                return_intermediates=True,
            )
            
            losses = loss_fn(
                logits=outputs['logits'],
                targets=batch['test_outputs'],
                attention_maps=outputs['attention_maps'],
                stop_logits=outputs['stop_logits'],
                predicates=outputs['predicates'],
                epoch=0,
                max_epochs=100,
            )
            
            loss = losses['total_loss']
        
        # Check for NaN in loss
        if not torch.isfinite(loss):
            nan_count += 1
            consecutive_nan += 1
            
            if first_nan_batch is None:
                first_nan_batch = batch_idx
                print(f"\n[FIRST NaN] Batch {batch_idx}")
                print(f"  Loss components:")
                for k, v in losses.items():
                    if torch.is_tensor(v):
                        print(f"    {k}: {v.item():.6f}")
                
                # Check model health
                issues = check_model_health(model)
                if issues:
                    print(f"  Model issues: {issues[:5]}")
                else:
                    print(f"  Model parameters are all finite")
                
                # Check outputs
                if not torch.isfinite(outputs['logits']).all():
                    print(f"  Logits have NaN/Inf")
                if not torch.isfinite(outputs['attention_maps']).all():
                    print(f"  Attention maps have NaN/Inf")
            
            if consecutive_nan >= max_consecutive_nan:
                print(f"\n[ABORT] {max_consecutive_nan} consecutive NaN batches!")
                break
            
            optimizer.zero_grad()
            continue
        
        # Reset consecutive counter
        consecutive_nan = 0
        loss_history.append(loss.item())
        
        # Backward pass
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()
        
        # Check gradient norms
        total_grad_norm = 0.0
        has_nan_grad = False
        for p in model.parameters():
            if p.grad is not None:
                if not torch.isfinite(p.grad).all():
                    has_nan_grad = True
                    break
                total_grad_norm += p.grad.norm().item() ** 2
        total_grad_norm = total_grad_norm ** 0.5
        
        if has_nan_grad:
            nan_count += 1
            consecutive_nan += 1
            if first_nan_batch is None:
                first_nan_batch = batch_idx
                print(f"\n[FIRST NaN GRAD] Batch {batch_idx}")
                print(f"  Loss was: {loss.item():.6f}")
                issues = check_model_health(model, prefix="  ")
                if issues:
                    for issue in issues[:10]:
                        print(f"    {issue}")
            
            if consecutive_nan >= max_consecutive_nan:
                print(f"\n[ABORT] {max_consecutive_nan} consecutive NaN gradient batches!")
                break
            
            optimizer.zero_grad()
            continue
        
        grad_norm_history.append(total_grad_norm)
        
        # Clip gradients
        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        
        # Optimizer step
        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        
        # Log progress
        if (batch_idx + 1) % 50 == 0 or batch_idx == 0:
            avg_loss = sum(loss_history[-50:]) / len(loss_history[-50:])
            avg_grad = sum(grad_norm_history[-50:]) / len(grad_norm_history[-50:]) if grad_norm_history else 0
            print(f"Batch {batch_idx+1:5d}/{num_batches}: loss={avg_loss:.4f}, grad_norm={avg_grad:.4f}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Mode: {mode}")
    print(f"Batches completed: {batch_idx + 1}")
    print(f"NaN occurrences: {nan_count}")
    print(f"First NaN at batch: {first_nan_batch}")
    
    if loss_history:
        print(f"Final loss: {loss_history[-1]:.6f}")
        print(f"Min loss: {min(loss_history):.6f}")
        print(f"Max loss: {max(loss_history):.6f}")
    
    if grad_norm_history:
        print(f"Avg grad norm: {sum(grad_norm_history)/len(grad_norm_history):.6f}")
        print(f"Max grad norm: {max(grad_norm_history):.6f}")
    
    result = {
        'mode': mode,
        'nan_count': nan_count,
        'first_nan_batch': first_nan_batch,
        'batches_completed': batch_idx + 1,
        'final_loss': loss_history[-1] if loss_history else None,
    }
    
    if nan_count == 0:
        print("\n✅ NO NaN DETECTED - Mode is stable!")
    else:
        print(f"\n❌ NaN DETECTED at batch {first_nan_batch}")
    
    return result


def main():
    parser = argparse.ArgumentParser(description='NaN Investigation')
    parser.add_argument('--mode', type=str, default='cpu',
                        choices=['cpu', 'gpu_no_amp', 'gpu_amp', 'all'],
                        help='Test mode')
    parser.add_argument('--batches', type=int, default=500,
                        help='Number of batches to simulate')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--grid-size', type=int, default=15,
                        help='Grid size (smaller = faster)')
    args = parser.parse_args()
    
    if args.mode == 'all':
        modes = ['cpu', 'gpu_no_amp', 'gpu_amp']
    else:
        modes = [args.mode]
    
    results = {}
    for mode in modes:
        print("\n" + "=" * 70)
        result = run_training_simulation(
            mode=mode,
            num_batches=args.batches,
            batch_size=args.batch_size,
            grid_size=args.grid_size,
        )
        results[mode] = result
        print()
    
    # Final comparison
    if len(results) > 1:
        print("\n" + "=" * 70)
        print("COMPARISON SUMMARY")
        print("=" * 70)
        print(f"{'Mode':<15} {'NaN Count':<12} {'First NaN':<12} {'Verdict'}")
        print("-" * 55)
        for mode, result in results.items():
            if result:
                verdict = "✅ STABLE" if result['nan_count'] == 0 else "❌ UNSTABLE"
                first_nan = result['first_nan_batch'] if result['first_nan_batch'] else "N/A"
                print(f"{mode:<15} {result['nan_count']:<12} {first_nan:<12} {verdict}")
        
        # Diagnosis
        print("\nDIAGNOSIS:")
        cpu_nan = results.get('cpu', {}).get('nan_count', 0)
        gpu_no_amp_nan = results.get('gpu_no_amp', {}).get('nan_count', 0)
        gpu_amp_nan = results.get('gpu_amp', {}).get('nan_count', 0)
        
        if cpu_nan == 0 and gpu_no_amp_nan == 0 and gpu_amp_nan > 0:
            print("  → AMP (Mixed Precision) is the root cause")
            print("  → Recommendation: Disable mixed_precision or use bfloat16 instead of float16")
        elif cpu_nan == 0 and gpu_no_amp_nan > 0:
            print("  → GPU computation (CUDA kernels) is the root cause")
            print("  → Recommendation: Check for CUDA-specific numerical issues")
        elif cpu_nan > 0:
            print("  → Model/Loss numerical instability is the root cause")
            print("  → Recommendation: Review loss computation and attention mechanisms")
        else:
            print("  → No NaN detected in any mode - issue may be data-dependent")
            print("  → Recommendation: Test with actual ARC data, not synthetic")


if __name__ == '__main__':
    main()
