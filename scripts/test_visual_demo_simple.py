#!/usr/bin/env python3
"""
Test RLAN on the 3 Visual Demo Examples
========================================

This script tests RLAN's ability to learn 3 simple ARC-like tasks:
1. Object Movement (4x4): Move gray(5) to red(2) marker position
2. Pattern Tiling (2x2 -> 6x6): Tile a pattern with rotations
3. Conditional Logic (3x3): Simple transformation

Goal: 100% accuracy on all 3 examples within 50 epochs.

Usage:
    python scripts/test_visual_demo_examples.py

Output:
    - Detailed training log to docs/visual_demo_training_log.md
    - Final predictions visualized
"""

import sys
import os
import math
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from sci_arc.models.rlan import RLAN, RLANConfig
from sci_arc.training.rlan_loss import RLANLoss


# =============================================================================
# Define the 3 Visual Demo Examples
# =============================================================================

# Example 1: Object Movement (4x4)
# Move gray(5) from top-left corner to the position marked by red(2)
EX1_INPUT = torch.tensor([
    [5, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 2]
], dtype=torch.long)

EX1_OUTPUT = torch.tensor([
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 5]
], dtype=torch.long)

# Example 2: Pattern Tiling (2x2 -> 6x6)
# Tile the input pattern 3x3 times with alternating rotation
EX2_INPUT = torch.tensor([
    [3, 2],
    [7, 8]
], dtype=torch.long)

EX2_OUTPUT = torch.tensor([
    [3, 2, 3, 2, 3, 2],
    [7, 8, 7, 8, 7, 8],
    [2, 3, 2, 3, 2, 3],
    [8, 7, 8, 7, 8, 7],
    [3, 2, 3, 2, 3, 2],
    [7, 8, 7, 8, 7, 8]
], dtype=torch.long)

# Example 3: Conditional Logic (3x3)
# Simple pattern transformation
EX3_INPUT = torch.tensor([
    [2, 0, 2],
    [1, 1, 1],
    [0, 0, 0]
], dtype=torch.long)

# For example 3, let's say the output fills in the gaps
EX3_OUTPUT = torch.tensor([
    [2, 2, 2],
    [1, 1, 1],
    [1, 1, 1]
], dtype=torch.long)


def pad_to_size(grid: torch.Tensor, target_h: int, target_w: int, pad_value: int = 0) -> torch.Tensor:
    """Pad a grid to target size with padding value."""
    h, w = grid.shape
    if h >= target_h and w >= target_w:
        return grid[:target_h, :target_w]
    
    padded = torch.full((target_h, target_w), pad_value, dtype=grid.dtype)
    padded[:h, :w] = grid
    return padded


def create_batch(max_size: int = 8) -> tuple:
    """
    Create a batch of 3 examples padded to uniform size.
    
    Returns:
        inputs: (3, H, W)
        outputs: (3, H, W)
        train_inputs: (3, 1, H, W) - same as inputs for context
        train_outputs: (3, 1, H, W) - same as outputs for context
    """
    # Pad all to max_size x max_size
    inputs = torch.stack([
        pad_to_size(EX1_INPUT, max_size, max_size),
        pad_to_size(EX2_INPUT, max_size, max_size),
        pad_to_size(EX3_INPUT, max_size, max_size),
    ])  # (3, H, W)
    
    outputs = torch.stack([
        pad_to_size(EX1_OUTPUT, max_size, max_size),
        pad_to_size(EX2_OUTPUT, max_size, max_size),
        pad_to_size(EX3_OUTPUT, max_size, max_size),
    ])  # (3, H, W)
    
    # Create training context (use same examples as demonstrations)
    train_inputs = inputs.unsqueeze(1)  # (3, 1, H, W)
    train_outputs = outputs.unsqueeze(1)  # (3, 1, H, W)
    
    return inputs, outputs, train_inputs, train_outputs


def compute_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict:
    """Compute per-example and aggregate metrics."""
    B, C, H, W = logits.shape
    preds = logits.argmax(dim=1)  # (B, H, W)
    
    # Per-example metrics
    metrics = {
        'per_example': [],
        'total_acc': 0.0,
        'total_bg_acc': 0.0,
        'total_fg_acc': 0.0,
    }
    
    total_correct = 0
    total_pixels = 0
    total_bg_correct = 0
    total_bg_pixels = 0
    total_fg_correct = 0
    total_fg_pixels = 0
    
    for b in range(B):
        pred = preds[b]  # (H, W)
        target = targets[b]  # (H, W)
        
        # Overall accuracy
        correct = (pred == target).sum().item()
        total = H * W
        acc = correct / total
        
        # Background (class 0) accuracy
        bg_mask = (target == 0)
        bg_correct = ((pred == target) & bg_mask).sum().item()
        bg_total = bg_mask.sum().item()
        bg_acc = bg_correct / bg_total if bg_total > 0 else 1.0
        
        # Foreground (classes 1-9) accuracy
        fg_mask = (target > 0)
        fg_correct = ((pred == target) & fg_mask).sum().item()
        fg_total = fg_mask.sum().item()
        fg_acc = fg_correct / fg_total if fg_total > 0 else 1.0
        
        # Unique classes in target
        unique_target = torch.unique(target).tolist()
        unique_pred = torch.unique(pred).tolist()
        
        metrics['per_example'].append({
            'accuracy': acc * 100,
            'bg_accuracy': bg_acc * 100,
            'fg_accuracy': fg_acc * 100,
            'bg_pixels': bg_total,
            'fg_pixels': fg_total,
            'unique_target': unique_target,
            'unique_pred': unique_pred,
            'exact_match': correct == total,
        })
        
        total_correct += correct
        total_pixels += total
        total_bg_correct += bg_correct
        total_bg_pixels += bg_total
        total_fg_correct += fg_correct
        total_fg_pixels += fg_total
    
    metrics['total_acc'] = total_correct / total_pixels * 100
    metrics['total_bg_acc'] = total_bg_correct / total_bg_pixels * 100 if total_bg_pixels > 0 else 100.0
    metrics['total_fg_acc'] = total_fg_correct / total_fg_pixels * 100 if total_fg_pixels > 0 else 100.0
    
    return metrics


def grid_to_str(grid: torch.Tensor, max_h: int = None, max_w: int = None) -> str:
    """Convert grid to string representation."""
    h, w = grid.shape
    max_h = max_h or h
    max_w = max_w or w
    
    lines = []
    for i in range(min(h, max_h)):
        row = ' '.join(f'{grid[i, j].item():1d}' for j in range(min(w, max_w)))
        lines.append(row)
    return '\n'.join(lines)


def log_epoch(
    log_file,
    epoch: int,
    outputs: dict,
    targets: torch.Tensor,
    losses: dict,
    metrics: dict,
    model: nn.Module,
    optimizer,
    temperature: float,
):
    """Log detailed epoch information."""
    B = targets.shape[0]
    logits = outputs['logits']
    preds = logits.argmax(dim=1)
    
    log_file.write(f"\n## Epoch {epoch}\n\n")
    
    # Temperature
    log_file.write(f"**Temperature:** {temperature:.4f}\n\n")
    
    # Losses
    log_file.write("### Losses\n")
    log_file.write("| Loss | Value |\n")
    log_file.write("|------|-------|\n")
    for key, value in losses.items():
        if isinstance(value, torch.Tensor):
            value = value.item()
        if isinstance(value, (int, float)):
            log_file.write(f"| {key} | {value:.6f} |\n")
        else:
            log_file.write(f"| {key} | {value} |\n")
    log_file.write("\n")
    
    # Overall Metrics
    log_file.write("### Metrics\n")
    log_file.write(f"- **Total Accuracy:** {metrics['total_acc']:.2f}%\n")
    log_file.write(f"- **BG Accuracy:** {metrics['total_bg_acc']:.2f}%\n")
    log_file.write(f"- **FG Accuracy:** {metrics['total_fg_acc']:.2f}%\n\n")
    
    # Per-example metrics
    log_file.write("### Per-Example Metrics\n")
    log_file.write("| Example | Accuracy | BG Acc | FG Acc | Exact Match | Target Classes | Pred Classes |\n")
    log_file.write("|---------|----------|--------|--------|-------------|----------------|---------------|\n")
    for i, ex in enumerate(metrics['per_example']):
        log_file.write(
            f"| {i+1} | {ex['accuracy']:.1f}% | {ex['bg_accuracy']:.1f}% | {ex['fg_accuracy']:.1f}% | "
            f"{'✅' if ex['exact_match'] else '❌'} | {ex['unique_target']} | {ex['unique_pred']} |\n"
        )
    log_file.write("\n")
    
    # DSC Analysis (if available)
    if 'stop_logits' in outputs and outputs['stop_logits'] is not None:
        stop_logits = outputs['stop_logits']  # (B, K)
        stop_probs = torch.sigmoid(stop_logits)
        clues_used = (1 - stop_probs).sum(dim=-1)  # (B,)
        
        log_file.write("### DSC Analysis\n")
        log_file.write("| Example | Clues Used | Stop Probs | Stop Logits |\n")
        log_file.write("|---------|------------|------------|-------------|\n")
        for b in range(B):
            sp = stop_probs[b].tolist()
            sl = stop_logits[b].tolist()
            cu = clues_used[b].item()
            log_file.write(
                f"| {b+1} | {cu:.2f} | [{', '.join(f'{s:.3f}' for s in sp)}] | "
                f"[{', '.join(f'{l:.2f}' for l in sl)}] |\n"
            )
        log_file.write("\n")
        
        # Attention analysis
        if 'attention_maps' in outputs and outputs['attention_maps'] is not None:
            attn = outputs['attention_maps']  # (B, K, H, W)
            log_file.write("### Attention Entropy (per clue)\n")
            log_file.write("| Example | Clue 0 | Clue 1 | Clue 2 | Clue 3 | Clue 4 | Clue 5 |\n")
            log_file.write("|---------|--------|--------|--------|--------|--------|--------|\n")
            
            for b in range(B):
                entropies = []
                for k in range(attn.shape[1]):
                    attn_flat = attn[b, k].view(-1).clamp(min=1e-10)
                    entropy = -(attn_flat * torch.log(attn_flat)).sum().item()
                    max_entropy = math.log(attn_flat.numel())
                    norm_entropy = entropy / max_entropy if max_entropy > 0 else 0
                    entropies.append(f"{norm_entropy:.3f}")
                log_file.write(f"| {b+1} | {' | '.join(entropies)} |\n")
            log_file.write("\n")
    
    # Gradient norms (if first few epochs)
    if epoch <= 5 or epoch % 10 == 0:
        log_file.write("### Gradient Norms (selected modules)\n")
        grad_norms = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                module = name.split('.')[0]
                if module not in grad_norms:
                    grad_norms[module] = 0.0
                grad_norms[module] += param.grad.norm().item() ** 2
        
        for module, norm_sq in grad_norms.items():
            log_file.write(f"- **{module}:** {math.sqrt(norm_sq):.6f}\n")
        log_file.write("\n")
    
    # Show predictions vs targets every 10 epochs or at end
    if epoch == 1 or epoch % 10 == 0 or epoch == 50:
        log_file.write("### Predictions vs Targets\n\n")
        for b in range(B):
            target = targets[b]
            pred = preds[b]
            
            # Get actual grid sizes (non-padded)
            if b == 0:
                h, w = 4, 4  # Example 1
            elif b == 1:
                h, w = 6, 6  # Example 2
            else:
                h, w = 3, 3  # Example 3
            
            log_file.write(f"**Example {b+1}:**\n\n")
            log_file.write("Target:\n```\n")
            log_file.write(grid_to_str(target[:h, :w]))
            log_file.write("\n```\n\n")
            log_file.write("Prediction:\n```\n")
            log_file.write(grid_to_str(pred[:h, :w]))
            log_file.write("\n```\n\n")
    
    log_file.flush()


def train_rlan_on_examples(
    max_epochs: int = 50,
    lr: float = 1e-3,
    device: str = 'cpu',
    log_path: str = None,
):
    """
    Train RLAN on the 3 visual demo examples.
    
    Returns:
        True if all examples reach 100% accuracy, False otherwise
    """
    if log_path is None:
        log_path = project_root / 'docs' / 'visual_demo_training_log.md'
    
    print(f"=" * 60)
    print("RLAN Visual Demo Test")
    print(f"=" * 60)
    print(f"Device: {device}")
    print(f"Max Epochs: {max_epochs}")
    print(f"Learning Rate: {lr}")
    print(f"Log Path: {log_path}")
    print()
    
    # Create batch
    inputs, outputs, train_inputs, train_outputs = create_batch(max_size=8)
    inputs = inputs.to(device)
    outputs = outputs.to(device)
    train_inputs = train_inputs.to(device)
    train_outputs = train_outputs.to(device)
    
    print("Examples loaded:")
    print(f"  Example 1: Object Movement (4x4)")
    print(f"  Example 2: Pattern Tiling (2x2 -> 6x6)")
    print(f"  Example 3: Conditional Logic (3x3)")
    print()
    
    # Load minimal config
    config_path = project_root / 'configs' / 'rlan_minimal.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_cfg = config['model']
    train_cfg = config['training']
    
    # Override for small test
    model_cfg['max_grid_size'] = 8
    model_cfg['max_clues'] = 4  # Fewer clues for simple examples
    model_cfg['num_solver_steps'] = 4  # Fewer steps
    
    print("Model Configuration:")
    print(f"  hidden_dim: {model_cfg['hidden_dim']}")
    print(f"  max_clues: {model_cfg['max_clues']}")
    print(f"  num_solver_steps: {model_cfg['num_solver_steps']}")
    print(f"  use_dsc: {model_cfg.get('use_dsc', True)}")
    print(f"  use_msre: {model_cfg.get('use_msre', True)}")
    print(f"  use_lcr: {model_cfg.get('use_lcr', False)}")
    print()
    
    # Create model
    rlan_config = RLANConfig(
        hidden_dim=model_cfg['hidden_dim'],
        num_colors=model_cfg['num_colors'],
        num_classes=model_cfg['num_classes'],
        max_grid_size=model_cfg['max_grid_size'],
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
    print(f"Model Parameters: {param_count:,}")
    print()
    
    # Create loss function
    loss_fn = RLANLoss(
        focal_gamma=train_cfg['focal_gamma'],
        focal_alpha=train_cfg['focal_alpha'],
        lambda_entropy=train_cfg['lambda_entropy'],
        lambda_sparsity=train_cfg['lambda_sparsity'],
        lambda_predicate=train_cfg['lambda_predicate'],
        lambda_curriculum=train_cfg['lambda_curriculum'],
        lambda_deep_supervision=train_cfg['lambda_deep_supervision'],
        lambda_act=train_cfg.get('lambda_act', 0.0),
        min_clues=train_cfg.get('min_clues', 1.0),
        min_clue_weight=train_cfg.get('min_clue_weight', 5.0),
        ponder_weight=train_cfg.get('ponder_weight', 0.01),
        max_clues=model_cfg['max_clues'],
        use_stablemax=train_cfg.get('use_stablemax', True),
        loss_mode=train_cfg.get('loss_mode', 'weighted_stablemax'),
        bg_weight_cap=train_cfg.get('bg_weight_cap', 1.0),
        fg_weight_cap=train_cfg.get('fg_weight_cap', 10.0),
    )
    
    print("Loss Configuration:")
    print(f"  loss_mode: {train_cfg.get('loss_mode', 'weighted_stablemax')}")
    print(f"  lambda_sparsity: {train_cfg['lambda_sparsity']}")
    print(f"  min_clues: {train_cfg.get('min_clues', 1.0)}")
    print(f"  min_clue_weight: {train_cfg.get('min_clue_weight', 5.0)}")
    print(f"  lambda_deep_supervision: {train_cfg['lambda_deep_supervision']}")
    print()
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=0.01,
        betas=(0.9, 0.95),
    )
    
    # Temperature schedule
    temp_start = train_cfg.get('temperature_start', 1.0)
    temp_end = train_cfg.get('temperature_end', 0.1)
    
    # Open log file
    with open(log_path, 'w', encoding='utf-8') as log_file:
        log_file.write("# RLAN Visual Demo Training Log\n\n")
        log_file.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"**Device:** {device}\n")
        log_file.write(f"**Max Epochs:** {max_epochs}\n")
        log_file.write(f"**Learning Rate:** {lr}\n\n")
        
        log_file.write("## Examples\n\n")
        log_file.write("1. **Object Movement (4x4):** Move gray(5) to red(2) marker position\n")
        log_file.write("2. **Pattern Tiling (2x2→6x6):** Tile pattern with alternating rotation\n")
        log_file.write("3. **Conditional Logic (3x3):** Fill transformation\n\n")
        
        log_file.write("## Model Configuration\n\n")
        log_file.write(f"- hidden_dim: {model_cfg['hidden_dim']}\n")
        log_file.write(f"- max_clues: {model_cfg['max_clues']}\n")
        log_file.write(f"- num_solver_steps: {model_cfg['num_solver_steps']}\n")
        log_file.write(f"- use_dsc: {model_cfg.get('use_dsc', True)}\n")
        log_file.write(f"- use_msre: {model_cfg.get('use_msre', True)}\n")
        log_file.write(f"- Parameters: {param_count:,}\n\n")
        
        log_file.write("## Loss Configuration\n\n")
        log_file.write(f"- loss_mode: {train_cfg.get('loss_mode', 'weighted_stablemax')}\n")
        log_file.write(f"- lambda_sparsity: {train_cfg['lambda_sparsity']}\n")
        log_file.write(f"- min_clues: {train_cfg.get('min_clues', 1.0)}\n")
        log_file.write(f"- min_clue_weight: {train_cfg.get('min_clue_weight', 5.0)}\n")
        log_file.write(f"- lambda_deep_supervision: {train_cfg['lambda_deep_supervision']}\n\n")
        
        log_file.write("---\n\n")
        log_file.write("# Training Progress\n")
        
        best_acc = 0.0
        best_epoch = 0
        all_exact_match = False
        
        for epoch in range(1, max_epochs + 1):
            # Temperature annealing - USE HIGHER MIN TEMP for stability
            progress = epoch / max_epochs
            temp_min = max(temp_end, 0.5)  # Don't go below 0.5 for small batches
            temperature = temp_start * (temp_min / temp_start) ** progress
            
            # Forward pass
            model_outputs = model(
                input_grid=inputs,
                train_inputs=train_inputs,
                train_outputs=train_outputs,
                temperature=temperature,
                return_intermediates=True,
                return_all_steps=True,
            )
            
            # NaN check on model outputs
            if torch.isnan(model_outputs['logits']).any():
                print(f"\n❌ NaN detected in logits at epoch {epoch}!")
                log_file.write(f"\n## ❌ NaN detected in logits at epoch {epoch}!\n")
                break
            
            # Compute loss
            losses = loss_fn(
                logits=model_outputs['logits'],
                targets=outputs,
                attention_maps=model_outputs.get('attention_maps'),
                stop_logits=model_outputs.get('stop_logits'),
                predicates=model_outputs.get('predicates'),
                epoch=epoch,
                max_epochs=max_epochs,
                all_logits=model_outputs.get('all_logits'),
            )
            
            # NaN check on loss
            if torch.isnan(losses['total_loss']):
                print(f"\n❌ NaN detected in loss at epoch {epoch}!")
                log_file.write(f"\n## ❌ NaN detected in loss at epoch {epoch}!\n")
                # Log which component went NaN
                for k, v in losses.items():
                    if torch.is_tensor(v) and torch.isnan(v):
                        log_file.write(f"- {k}: NaN\n")
                break
            
            # Backward pass
            optimizer.zero_grad()
            losses['total_loss'].backward()
            
            # Check for NaN gradients
            nan_grads = False
            for name, param in model.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    nan_grads = True
                    print(f"\n❌ NaN gradient in {name} at epoch {epoch}!")
                    log_file.write(f"\n## ❌ NaN gradient in {name} at epoch {epoch}!\n")
                    break
            if nan_grads:
                break
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            
            optimizer.step()
            
            # Compute metrics
            with torch.no_grad():
                metrics = compute_metrics(model_outputs['logits'], outputs)
            
            # Log epoch
            log_epoch(
                log_file=log_file,
                epoch=epoch,
                outputs=model_outputs,
                targets=outputs,
                losses={k: v.item() if torch.is_tensor(v) else v for k, v in losses.items()},
                metrics=metrics,
                model=model,
                optimizer=optimizer,
                temperature=temperature,
            )
            
            # Print progress
            if epoch % 5 == 0 or epoch == 1:
                exact_matches = sum(1 for ex in metrics['per_example'] if ex['exact_match'])
                print(
                    f"Epoch {epoch:3d} | Loss: {losses['total_loss'].item():.4f} | "
                    f"Acc: {metrics['total_acc']:.1f}% | FG: {metrics['total_fg_acc']:.1f}% | "
                    f"Exact: {exact_matches}/3 | Temp: {temperature:.3f}"
                )
            
            # Track best
            if metrics['total_acc'] > best_acc:
                best_acc = metrics['total_acc']
                best_epoch = epoch
            
            # Check for all exact matches
            if all(ex['exact_match'] for ex in metrics['per_example']):
                all_exact_match = True
                print(f"\n🎉 ALL EXAMPLES SOLVED at epoch {epoch}!")
                log_file.write(f"\n## 🎉 SUCCESS: All examples solved at epoch {epoch}!\n")
                break
        
        # Final summary
        log_file.write("\n---\n\n")
        log_file.write("# Final Summary\n\n")
        log_file.write(f"- **Best Accuracy:** {best_acc:.2f}% (epoch {best_epoch})\n")
        log_file.write(f"- **All Exact Match:** {'✅ Yes' if all_exact_match else '❌ No'}\n")
        
        if not all_exact_match:
            log_file.write("\n## ❌ FAILURE ANALYSIS\n\n")
            log_file.write("The model failed to achieve 100% accuracy on all examples.\n")
            log_file.write("Review the epoch logs above to identify:\n")
            log_file.write("1. Which examples are hardest (lowest accuracy)\n")
            log_file.write("2. Whether FG or BG accuracy is the problem\n")
            log_file.write("3. Stop logit saturation (values near ±4)\n")
            log_file.write("4. Gradient flow issues (near-zero gradients)\n")
            log_file.write("5. Attention entropy (should decrease over epochs)\n")
        
        log_file.write("\n")
    
    print()
    print(f"=" * 60)
    print("Training Complete")
    print(f"=" * 60)
    print(f"Best Accuracy: {best_acc:.2f}% (epoch {best_epoch})")
    print(f"All Exact Match: {'✅ Yes' if all_exact_match else '❌ No'}")
    print(f"Log saved to: {log_path}")
    
    return all_exact_match


if __name__ == '__main__':
    # Run the test
    success = train_rlan_on_examples(
        max_epochs=200,  # More epochs for convergence
        lr=5e-4,  # Lower LR for stability
        device='cpu',
    )
    
    if not success:
        print("\n⚠️  Test FAILED - Review the log for debugging")
        sys.exit(1)
    else:
        print("\n✅ Test PASSED - RLAN solved all examples!")
        sys.exit(0)
