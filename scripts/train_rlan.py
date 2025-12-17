#!/usr/bin/env python
"""
Train RLAN Model on ARC Dataset - Production-Ready Training Script

Features matching CISL production training:
- Mixed precision (AMP) for RTX 3090 efficiency
- Reproducible training with seed control
- Auto-resume from checkpoints
- Gradient accumulation for larger effective batch size
- File logging for reproducibility
- WandB integration (optional)
- Cache samples mode for testing vs infinite diversity for competitive training

Usage:
    python scripts/train_rlan.py --config configs/rlan_base.yaml
    python scripts/train_rlan.py --config configs/rlan_small.yaml
    python scripts/train_rlan.py --config configs/rlan_base.yaml --resume auto
    python scripts/train_rlan.py --config configs/rlan_base.yaml --resume checkpoints/rlan_base/latest.pt
"""

import argparse
import os
import sys
import time
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any, List
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sci_arc.models import RLAN, RLANConfig
from sci_arc.training import RLANLoss
from sci_arc.training.ema import EMAHelper
from sci_arc.data import ARCDataset, collate_sci_arc

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


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


def set_seed(seed: int, deterministic: bool = False):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def get_curriculum_stage(epoch: int, curriculum_stages: List[int]) -> int:
    """
    Determine curriculum stage based on current epoch.
    
    Args:
        epoch: Current training epoch (0-indexed)
        curriculum_stages: List of epoch thresholds [stage1_end, stage2_end, stage3_end]
            e.g., [100, 300, 600] means:
            - epochs 0-99: stage 1 (easy only)
            - epochs 100-299: stage 2 (easy + medium)
            - epochs 300-599: stage 3 (easy + medium + hard)
            - epochs 600+: stage 0 (all data, no filtering)
    
    Returns:
        Curriculum stage (1=easy, 2=medium, 3=hard, 0=all)
    """
    if not curriculum_stages:
        return 0  # No curriculum, use all data
    
    for stage_idx, threshold in enumerate(curriculum_stages):
        if epoch < threshold:
            return stage_idx + 1  # stages are 1-indexed
    
    return 0  # Beyond all thresholds, use all data


def create_train_loader(
    config: dict,
    curriculum_stage: int = 0,
    max_grid_size: int = 30,
) -> DataLoader:
    """
    Create training dataloader with optional curriculum filtering.
    
    Args:
        config: Full configuration dict
        curriculum_stage: Curriculum stage (0=all, 1=easy, 2=medium, 3=hard)
        max_grid_size: Maximum grid size for padding
    
    Returns:
        DataLoader for training
    """
    data_cfg = config['data']
    train_cfg = config['training']
    
    augment_cfg = data_cfg.get('augmentation', {})
    augment_enabled = augment_cfg.get('enabled', True)
    color_permutation = augment_cfg.get('color_permutation', False)
    translational_augment = augment_cfg.get('translational', True)  # TRM-style offset
    
    # Enable augmentation tracking for debugging (verifies diversity)
    track_augmentation = config.get('logging', {}).get('track_augmentation', True)
    
    train_dataset = ARCDataset(
        data_cfg['train_path'],
        max_size=max_grid_size,
        augment=augment_enabled,
        color_permutation=color_permutation,
        translational_augment=translational_augment,
        curriculum_stage=curriculum_stage,  # Apply curriculum filtering!
        track_augmentation=track_augmentation,  # Enable for diversity logging
    )
    
    batch_size = train_cfg['batch_size']
    num_workers = data_cfg.get('num_workers', 0)
    pin_memory = data_cfg.get('pin_memory', True)
    prefetch_factor = data_cfg.get('prefetch_factor', 2) if num_workers > 0 else None
    persistent_workers = data_cfg.get('persistent_workers', False) and num_workers > 0
    
    collate_fn = partial(collate_sci_arc, max_grid_size=max_grid_size)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        collate_fn=collate_fn,
        drop_last=True,
    )
    
    return train_loader


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Find the most recent checkpoint in the checkpoint directory."""
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        return None
    
    # Check for the dedicated "latest" checkpoint
    latest_path = checkpoint_path / 'latest.pt'
    if latest_path.exists():
        return str(latest_path)
    
    # Otherwise, find the highest epoch number
    checkpoints = list(checkpoint_path.glob('epoch_*.pt'))
    if not checkpoints:
        return None
    
    def get_epoch(p):
        try:
            return int(p.stem.split('_')[-1])
        except:
            return -1
    
    checkpoints.sort(key=get_epoch, reverse=True)
    return str(checkpoints[0])


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def override_config(config: dict, overrides: list) -> dict:
    """Apply command-line overrides to config.
    
    Format: key.subkey=value
    Examples:
        training.batch_size=32
        training.max_epochs=100
        logging.use_wandb=true
        model.hidden_dim=512
    """
    import ast
    for override in overrides:
        if '=' not in override:
            continue
        key_path, value = override.split('=', 1)
        keys = key_path.split('.')
        
        # Navigate to the right level
        current = config
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Set value with type inference
        final_key = keys[-1]
        try:
            # Try to parse as Python literal (bool, int, float, list)
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            # Keep as string
            pass
        
        current[final_key] = value
        print(f"  Override: {key_path} = {value}")
    
    return config


def get_temperature(epoch: int, config: dict) -> float:
    """Get temperature for Gumbel-softmax based on epoch."""
    tau_start = config['training']['temperature_start']
    tau_end = config['training']['temperature_end']
    max_epochs = config['training']['max_epochs']
    
    # Exponential decay
    progress = epoch / max_epochs
    temperature = tau_start * (tau_end / tau_start) ** progress
    
    return temperature


def create_model(config: dict) -> RLAN:
    """Create RLAN model from config."""
    from sci_arc.models.rlan import RLANConfig
    
    model_config = config['model']
    
    # Create RLANConfig with all parameters including ablation flags
    rlan_config = RLANConfig(
        hidden_dim=model_config['hidden_dim'],
        num_colors=model_config['num_colors'],
        num_classes=model_config['num_classes'],
        max_grid_size=model_config['max_grid_size'],
        max_clues=model_config['max_clues'],
        num_predicates=model_config['num_predicates'],
        num_solver_steps=model_config['num_solver_steps'],
        use_act=model_config.get('use_act', False),
        dropout=model_config['dropout'],
        dsc_num_heads=model_config.get('dsc_num_heads', 4),
        lcr_num_heads=model_config.get('lcr_num_heads', 4),
        msre_encoding_dim=model_config.get('msre_encoding_dim', 32),
        msre_num_freq=model_config.get('msre_num_freq', 8),
        lcr_num_freq=model_config.get('lcr_num_freq', 8),
        # Module ablation flags
        use_context_encoder=model_config.get('use_context_encoder', True),
        use_dsc=model_config.get('use_dsc', True),
        use_msre=model_config.get('use_msre', True),
        use_lcr=model_config.get('use_lcr', True),
        use_sph=model_config.get('use_sph', True),
        use_learned_pos=model_config.get('use_learned_pos', False),
    )
    
    model = RLAN(config=rlan_config)
    
    return model


def create_loss(config: dict) -> RLANLoss:
    """Create RLAN loss function from config."""
    train_config = config['training']
    model_config = config['model']
    
    loss_fn = RLANLoss(
        focal_gamma=train_config['focal_gamma'],
        focal_alpha=train_config['focal_alpha'],
        lambda_entropy=train_config['lambda_entropy'],
        lambda_sparsity=train_config['lambda_sparsity'],
        lambda_predicate=train_config['lambda_predicate'],
        lambda_curriculum=train_config['lambda_curriculum'],
        lambda_deep_supervision=train_config['lambda_deep_supervision'],
        lambda_act=train_config.get('lambda_act', 0.1),  # ACT halting loss weight
        min_clues=train_config.get('min_clues', 1.0),  # Minimum clues to use
        ponder_weight=train_config.get('ponder_weight', 0.1),  # Base cost per clue
        entropy_ponder_weight=train_config.get('entropy_ponder_weight', 0.05),  # Extra cost for diffuse attention
        max_clues=model_config['max_clues'],
        use_stablemax=train_config.get('use_stablemax', True),
        loss_mode=train_config.get('loss_mode', 'focal_stablemax'),  # TRM uses 'stablemax'
    )
    
    return loss_fn


def create_optimizer(model: nn.Module, config: dict, steps_per_epoch: int = None):
    """Create optimizer and scheduler.
    
    Args:
        model: The model to optimize
        config: Configuration dict
        steps_per_epoch: Number of training batches per epoch (for OneCycle)
    """
    train_config = config['training']
    
    # Separate parameters with and without weight decay
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'bias' in name or 'norm' in name or 'embedding' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    param_groups = [
        {'params': decay_params, 'weight_decay': train_config['weight_decay']},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ]
    
    # Get optimizer betas (TRM uses beta2=0.95 instead of default 0.999)
    beta1 = train_config.get('beta1', 0.9)
    beta2 = train_config.get('beta2', 0.95)  # TRM default, more stable for recursive models
    
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=train_config['learning_rate'],
        betas=(beta1, beta2),
    )
    
    # Create scheduler based on config
    scheduler_type = train_config.get('scheduler', 'cosine')
    max_epochs = train_config['max_epochs']
    warmup_epochs = train_config.get('warmup_epochs', 10)
    min_lr = train_config.get('min_lr', 1e-6)
    
    if scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max_epochs - warmup_epochs,
            eta_min=min_lr,
        )
    elif scheduler_type == 'onecycle':
        # OneCycleLR needs total_steps = epochs × steps_per_epoch
        total_steps = max_epochs * (steps_per_epoch or 1)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=train_config['learning_rate'],
            total_steps=total_steps,
            pct_start=warmup_epochs / max_epochs,  # Warmup fraction
            anneal_strategy='cos',
            div_factor=25.0,  # initial_lr = max_lr / 25
            final_div_factor=1000.0,  # final_lr = max_lr / 1000
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=50,
            T_mult=2,
            eta_min=min_lr,
        )
    
    return optimizer, scheduler


def warmup_lr(optimizer, step: int, warmup_steps: int, base_lr: float):
    """Apply linear warmup to learning rate."""
    if step < warmup_steps:
        lr_scale = float(step + 1) / float(max(1, warmup_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = base_lr * lr_scale


def compute_module_grad_norms(model: RLAN) -> Dict[str, float]:
    """Compute gradient norms for key modules to verify learning signal."""
    grad_norms = {}
    
    # DSC gradients
    if hasattr(model, 'dsc') and model.dsc is not None:
        dsc_grad = 0.0
        dsc_count = 0
        for name, param in model.dsc.named_parameters():
            if param.grad is not None:
                dsc_grad += param.grad.norm().item() ** 2
                dsc_count += 1
        grad_norms['dsc'] = (dsc_grad ** 0.5) if dsc_count > 0 else 0.0
    
    # Encoder gradients
    if hasattr(model, 'encoder') and model.encoder is not None:
        enc_grad = 0.0
        enc_count = 0
        for name, param in model.encoder.named_parameters():
            if param.grad is not None:
                enc_grad += param.grad.norm().item() ** 2
                enc_count += 1
        grad_norms['encoder'] = (enc_grad ** 0.5) if enc_count > 0 else 0.0
    
    # Solver gradients
    if hasattr(model, 'solver') and model.solver is not None:
        solver_grad = 0.0
        solver_count = 0
        for name, param in model.solver.named_parameters():
            if param.grad is not None:
                solver_grad += param.grad.norm().item() ** 2
                solver_count += 1
        grad_norms['solver'] = (solver_grad ** 0.5) if solver_count > 0 else 0.0
    
    return grad_norms


def train_epoch(
    model: RLAN,
    dataloader: DataLoader,
    loss_fn: RLANLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    config: dict,
    scaler: Optional[GradScaler] = None,
    global_step: int = 0,
    ema: Optional[EMAHelper] = None,
) -> Dict[str, float]:
    """
    Train for one epoch with augmentation diversity tracking.
    
    Returns losses dict AND augmentation statistics for debugging.
    """
    model.train()
    
    total_losses = {
        'total_loss': 0.0,
        'focal_loss': 0.0,
        'entropy_loss': 0.0,
        'sparsity_loss': 0.0,
        'predicate_loss': 0.0,
        'curriculum_loss': 0.0,
        'deep_supervision_loss': 0.0,  # FIX: Was missing, caused zero reporting
        'act_loss': 0.0,  # ACT halting loss
    }
    num_batches = 0
    total_samples = 0  # Track total samples processed
    
    # Augmentation statistics for the epoch (CRITICAL for verifying diversity)
    epoch_aug_stats = {
        'dihedral_counts': [0] * 8,  # Count per dihedral transform ID (0-7)
        'color_perm_count': 0,       # Samples with color permutation
        'translational_count': 0,    # Samples with translational offset
        'unique_offsets': 0,         # Running unique offset count (approximate)
    }
    
    # Diagnostic statistics for debugging training dynamics
    # ======================================================
    # These metrics help identify WHY training succeeds or fails
    # ======================================================
    epoch_diagnostics = {
        # Gradient flow (are gradients reaching each module?)
        'dsc_grad_norm_sum': 0.0,     # Gradient norm flowing to DSC
        'encoder_grad_norm_sum': 0.0,  # Gradient norm to encoder
        'solver_grad_norm_sum': 0.0,   # Gradient norm to solver
        
        # DSC attention quality (is DSC learning meaningful patterns?)
        'per_clue_entropy': [],        # Per-clue entropy breakdown (K values)
        'centroid_spread': 0.0,        # How spread out are clue centroids
        'all_logits_count': 0,         # Verify deep supervision is receiving steps
        
        # Attention statistics (is attention focusing or diffuse?)
        'attn_max_mean': 0.0,          # Mean of max attention values (higher = sharper)
        'attn_min_mean': 0.0,          # Mean of min attention values (lower = sharper)
        'stop_prob_mean': 0.0,         # Mean stop probability (how many clues used)
        
        # Per-step loss (is deep supervision working?)
        'per_step_loss': [],           # Loss at each solver step
    }
    
    temperature = get_temperature(epoch, config)
    max_epochs = config['training']['max_epochs']
    gradient_clip = config['training']['gradient_clip']
    grad_accumulation_steps = config['training'].get('grad_accumulation_steps', 1)
    use_amp = config['device'].get('mixed_precision', False) and device.type == 'cuda'
    log_every = config['logging'].get('log_every', 10)
    warmup_epochs = config['training'].get('warmup_epochs', 10)
    warmup_steps = warmup_epochs * len(dataloader)
    base_lr = config['training']['learning_rate']
    
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(dataloader):
        # Apply warmup
        warmup_lr(optimizer, global_step, warmup_steps, base_lr)
        
        # Collect augmentation statistics from batch (CRITICAL for diversity verification)
        if 'aug_stats' in batch:
            batch_aug = batch['aug_stats']
            for i in range(8):
                epoch_aug_stats['dihedral_counts'][i] += batch_aug['dihedral_counts'][i]
            epoch_aug_stats['color_perm_count'] += batch_aug['color_perm_count']
            epoch_aug_stats['translational_count'] += batch_aug['translational_count']
            epoch_aug_stats['unique_offsets'] += batch_aug['unique_offsets']
        
        # Track sample count
        batch_size = batch['test_inputs'].shape[0]
        total_samples += batch_size
        
        # Move batch to device - handle collated batch format
        test_inputs = batch['test_inputs'].to(device)
        test_outputs = batch['test_outputs'].to(device)
        
        # Get training context (CRITICAL for ARC learning!)
        train_inputs = batch['input_grids'].to(device)  # (B, N, H, W)
        train_outputs = batch['output_grids'].to(device)  # (B, N, H, W)
        pair_mask = batch['grid_masks'].to(device)  # (B, N)
        
        # Forward pass with optional mixed precision
        if use_amp and scaler is not None:
            with autocast('cuda'):
                outputs = model(
                    test_inputs,
                    train_inputs=train_inputs,
                    train_outputs=train_outputs,
                    pair_mask=pair_mask,
                    temperature=temperature,
                    return_intermediates=True,
                )
                
                losses = loss_fn(
                    logits=outputs['logits'],
                    targets=test_outputs,
                    attention_maps=outputs['attention_maps'],
                    stop_logits=outputs['stop_logits'],
                    predicates=outputs['predicates'],
                    epoch=epoch,
                    max_epochs=max_epochs,
                    all_logits=outputs.get('all_logits'),
                    act_outputs=outputs.get('act_outputs'),
                )
                
                # Scale loss for gradient accumulation
                loss = losses['total_loss'] / grad_accumulation_steps
            
            scaler.scale(loss).backward()
            
            # Step optimizer after accumulation
            if (batch_idx + 1) % grad_accumulation_steps == 0:
                if gradient_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                
                # Capture gradient norms before they're cleared (first accumulation step only)
                if batch_idx < grad_accumulation_steps:
                    grad_norms = compute_module_grad_norms(model)
                    epoch_diagnostics['dsc_grad_norm_sum'] += grad_norms.get('dsc', 0.0)
                    epoch_diagnostics['encoder_grad_norm_sum'] += grad_norms.get('encoder', 0.0)
                    epoch_diagnostics['solver_grad_norm_sum'] += grad_norms.get('solver', 0.0)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                # Update EMA after optimizer step
                if ema is not None:
                    ema.update(model)
        else:
            outputs = model(
                test_inputs,
                train_inputs=train_inputs,
                train_outputs=train_outputs,
                pair_mask=pair_mask,
                temperature=temperature,
                return_intermediates=True,
            )
            
            losses = loss_fn(
                logits=outputs['logits'],
                targets=test_outputs,
                attention_maps=outputs['attention_maps'],
                stop_logits=outputs['stop_logits'],
                predicates=outputs['predicates'],
                epoch=epoch,
                max_epochs=max_epochs,
                all_logits=outputs.get('all_logits'),
                act_outputs=outputs.get('act_outputs'),
            )
            
            loss = losses['total_loss'] / grad_accumulation_steps
            loss.backward()
            
            if (batch_idx + 1) % grad_accumulation_steps == 0:
                if gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                
                # Capture gradient norms before they're cleared (first accumulation step only)
                if batch_idx < grad_accumulation_steps:
                    grad_norms = compute_module_grad_norms(model)
                    epoch_diagnostics['dsc_grad_norm_sum'] += grad_norms.get('dsc', 0.0)
                    epoch_diagnostics['encoder_grad_norm_sum'] += grad_norms.get('encoder', 0.0)
                    epoch_diagnostics['solver_grad_norm_sum'] += grad_norms.get('solver', 0.0)
                
                optimizer.step()
                optimizer.zero_grad()
                
                # Update EMA after optimizer step
                if ema is not None:
                    ema.update(model)
        
        # Accumulate losses
        for key in total_losses:
            if key in losses:
                total_losses[key] += losses[key].item()
        num_batches += 1
        global_step += 1
        
        # Collect diagnostics (first batch of each epoch only to avoid overhead)
        if batch_idx == 0:
            # Track all_logits count for deep supervision verification
            all_logits = outputs.get('all_logits')
            if all_logits is not None:
                epoch_diagnostics['all_logits_count'] = len(all_logits)
                
                # Per-step loss breakdown (verify deep supervision is working)
                with torch.no_grad():
                    step_losses = []
                    for step_logits in all_logits:
                        # Check for inf/nan in logits (numerical instability)
                        if torch.isfinite(step_logits).all():
                            step_loss = torch.nn.functional.cross_entropy(
                                step_logits, test_outputs, reduction='mean'
                            )
                            # Clamp to prevent inf in display
                            step_losses.append(min(step_loss.item(), 100.0))
                        else:
                            # Log warning and use placeholder
                            step_losses.append(float('nan'))
                    epoch_diagnostics['per_step_loss'] = step_losses
            
            # Per-clue entropy breakdown
            if 'attention_maps' in outputs:
                attn = outputs['attention_maps']  # (B, K, H, W)
                B, K, H, W = attn.shape
                attn_flat = attn.view(B, K, -1)
                attn_clamp = attn_flat.clamp(min=1e-10)
                per_clue_entropy = -(attn_clamp * attn_clamp.log()).sum(dim=-1).mean(dim=0)  # (K,)
                epoch_diagnostics['per_clue_entropy'] = per_clue_entropy.tolist()
                
                # Attention sharpness statistics
                attn_max = attn_flat.max(dim=-1)[0].mean()  # Mean of max attention
                attn_min = attn_flat.min(dim=-1)[0].mean()  # Mean of min attention
                epoch_diagnostics['attn_max_mean'] = attn_max.item()
                epoch_diagnostics['attn_min_mean'] = attn_min.item()
            
            # Stop probability (how many clues are used)
            if 'stop_logits' in outputs:
                stop_probs = torch.sigmoid(outputs['stop_logits'])  # (B, K)
                epoch_diagnostics['stop_prob_mean'] = stop_probs.mean().item()
            
            # Centroid spread (how diverse are clue locations)
            if 'centroids' in outputs:
                centroids = outputs['centroids']  # (B, K, 2)
                # Compute mean distance between clue centroids
                if centroids.shape[1] > 1:
                    centroid_mean = centroids.mean(dim=1, keepdim=True)  # (B, 1, 2)
                    spread = ((centroids - centroid_mean) ** 2).sum(dim=-1).sqrt().mean()
                    epoch_diagnostics['centroid_spread'] = spread.item()
        
        # Log progress
        if batch_idx % log_every == 0:
            current_lr = optimizer.param_groups[0]['lr']
            loss_mode = losses.get('loss_mode', 'task')
            task_loss_val = losses.get('task_loss', losses.get('focal_loss', torch.tensor(0.0)))
            task_loss_val = task_loss_val.item() if hasattr(task_loss_val, 'item') else task_loss_val
            print(f"  Batch {batch_idx}/{len(dataloader)}: "
                  f"loss={losses['total_loss'].item():.4f}, "
                  f"{loss_mode}={task_loss_val:.4f}, "
                  f"temp={temperature:.3f}, lr={current_lr:.2e}")
    
    # Average losses
    for key in total_losses:
        total_losses[key] /= max(num_batches, 1)
    
    # Add epoch-level statistics
    total_losses['total_samples'] = total_samples
    total_losses['num_batches'] = num_batches
    
    # Add augmentation diversity stats (CRITICAL for debugging)
    total_losses['aug_stats'] = epoch_aug_stats
    
    # Add training diagnostics (CRITICAL for debugging training dynamics)
    total_losses['diagnostics'] = epoch_diagnostics
    
    return total_losses, global_step


def evaluate(
    model: RLAN,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model on validation set with detailed metrics for debugging."""
    model.eval()
    
    total_correct = 0
    total_pixels = 0
    total_tasks = 0
    correct_tasks = 0
    
    # Additional metrics for debugging
    total_non_bg_correct = 0
    total_non_bg_pixels = 0
    color_predictions = [0] * 10  # Count predictions per color (0-9)
    color_targets = [0] * 10
    
    # Module-specific debugging metrics
    dsc_entropy_sum = 0.0
    dsc_usage_sum = 0.0  # How many clues are used (non-stopped)
    predicate_activation_sum = 0.0
    num_eval_samples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            test_inputs = batch['test_inputs'].to(device)
            test_outputs = batch['test_outputs'].to(device)
            
            # Get training context for proper evaluation
            train_inputs = batch['input_grids'].to(device)
            train_outputs = batch['output_grids'].to(device)
            pair_mask = batch['grid_masks'].to(device)
            
            # Predict with low temperature (sharp) and context, return intermediates
            outputs = model(
                test_inputs,
                train_inputs=train_inputs,
                train_outputs=train_outputs,
                pair_mask=pair_mask,
                temperature=0.1,
                return_intermediates=True,
            )
            
            logits = outputs['logits']
            predictions = logits.argmax(dim=1)
            
            # Pixel accuracy
            correct = (predictions == test_outputs).float()
            total_correct += correct.sum().item()
            total_pixels += test_outputs.numel()
            
            # Non-background accuracy (critical for detecting background collapse)
            non_bg_mask = test_outputs > 0  # Non-background pixels
            if non_bg_mask.any():
                non_bg_correct = ((predictions == test_outputs) & non_bg_mask).float()
                total_non_bg_correct += non_bg_correct.sum().item()
                total_non_bg_pixels += non_bg_mask.sum().item()
            
            # Color distribution tracking (ARC colors 0-9)
            for c in range(10):
                color_predictions[c] += (predictions == c).sum().item()
                color_targets[c] += (test_outputs == c).sum().item()
            
            # Task accuracy (all pixels correct)
            batch_size = test_inputs.shape[0]
            for i in range(batch_size):
                if (predictions[i] == test_outputs[i]).all():
                    correct_tasks += 1
                total_tasks += 1
            
            # DSC metrics (attention maps)
            if 'attention_maps' in outputs:
                attn = outputs['attention_maps']  # (B, K, H, W)
                attn_flat = attn.view(attn.shape[0], attn.shape[1], -1)
                # Entropy of attention (lower = sharper = better)
                attn_clamp = attn_flat.clamp(min=1e-10)
                entropy = -(attn_clamp * attn_clamp.log()).sum(dim=-1).mean()
                dsc_entropy_sum += entropy.item() * batch_size
            
            # DSC usage (from stop_logits)
            if 'stop_logits' in outputs:
                stop_probs = torch.sigmoid(outputs['stop_logits'])  # (B, K)
                usage = (stop_probs < 0.5).float().sum(dim=-1).mean()  # Avg clues used
                dsc_usage_sum += usage.item() * batch_size
            
            # Predicate activations
            if 'predicates' in outputs:
                pred_act = outputs['predicates'].abs().mean()
                predicate_activation_sum += pred_act.item() * batch_size
            
            num_eval_samples += batch_size
    
    pixel_accuracy = total_correct / max(total_pixels, 1)
    task_accuracy = correct_tasks / max(total_tasks, 1)
    non_bg_accuracy = total_non_bg_correct / max(total_non_bg_pixels, 1)
    
    # Calculate background ratio in predictions
    bg_ratio_pred = color_predictions[0] / max(sum(color_predictions), 1)
    bg_ratio_target = color_targets[0] / max(sum(color_targets), 1)
    
    # Color diversity (how many colors are actually predicted)
    colors_used = sum(1 for c in color_predictions if c > 0)
    colors_target = sum(1 for c in color_targets if c > 0)
    
    return {
        'pixel_accuracy': pixel_accuracy,
        'task_accuracy': task_accuracy,
        'non_bg_accuracy': non_bg_accuracy,
        'bg_ratio_pred': bg_ratio_pred,
        'bg_ratio_target': bg_ratio_target,
        'colors_used': colors_used,
        'colors_target': colors_target,
        'dsc_entropy': dsc_entropy_sum / max(num_eval_samples, 1),
        'dsc_clues_used': dsc_usage_sum / max(num_eval_samples, 1),
        'predicate_activation': predicate_activation_sum / max(num_eval_samples, 1),
    }


def save_checkpoint(
    model: RLAN,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    global_step: int,
    losses: Dict[str, float],
    best_accuracy: float,
    config: dict,
    path: str,
):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'losses': losses,
        'best_accuracy': best_accuracy,
        'config': config,
    }
    torch.save(checkpoint, path)
    print(f"  Saved checkpoint to {path}")


def load_checkpoint(
    model: RLAN,
    optimizer: torch.optim.Optimizer,
    scheduler,
    path: str,
) -> tuple:
    """Load training checkpoint. Returns (start_epoch, global_step, best_accuracy)."""
    checkpoint = torch.load(path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint['epoch']
    global_step = checkpoint.get('global_step', 0)
    best_accuracy = checkpoint.get('best_accuracy', 0.0)
    
    print(f"Loaded checkpoint from {path} (epoch {epoch})")
    
    return epoch + 1, global_step, best_accuracy


def cleanup_old_checkpoints(checkpoint_dir: Path, keep_last_n: int = 5):
    """Remove old checkpoints, keeping only the last N."""
    checkpoints = list(checkpoint_dir.glob('epoch_*.pt'))
    
    def get_epoch(p):
        try:
            return int(p.stem.split('_')[-1])
        except:
            return -1
    
    checkpoints.sort(key=get_epoch, reverse=True)
    
    for checkpoint in checkpoints[keep_last_n:]:
        checkpoint.unlink()


def main():
    parser = argparse.ArgumentParser(
        description="Train RLAN on ARC",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Config Overrides:
  You can override any config value using: key.subkey=value
  
  Examples:
    python scripts/train_rlan.py --config configs/rlan_base.yaml training.batch_size=32
    python scripts/train_rlan.py --config configs/rlan_base.yaml training.max_epochs=100 logging.use_wandb=true
    python scripts/train_rlan.py --config configs/rlan_base.yaml model.hidden_dim=512 training.learning_rate=3e-4
        """
    )
    parser.add_argument('--config', type=str, default='configs/rlan_base.yaml',
                        help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from (use "auto" for latest)')
    parser.add_argument('--no-resume', action='store_true',
                        help='Start fresh even if checkpoints exist')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu)')
    parser.add_argument('overrides', nargs='*',
                        help='Config overrides in format key.subkey=value')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Apply command-line overrides
    if args.overrides:
        print("Applying config overrides:")
        config = override_config(config, args.overrides)
    
    print(f"=" * 60)
    print(f"RLAN Training")
    print(f"=" * 60)
    print(f"Config: {args.config}")
    
    # Set seed for reproducibility
    hw_cfg = config.get('hardware', {})
    seed = hw_cfg.get('seed', 42)
    deterministic = hw_cfg.get('deterministic', False)
    set_seed(seed, deterministic)
    print(f"Seed: {seed}, Deterministic: {deterministic}")
    
    # Setup device
    if args.device:
        device = torch.device(args.device)
    elif config['device']['use_cuda'] and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Setup checkpoint directory and logging
    log_cfg = config.get('logging', {})
    checkpoint_dir = Path(log_cfg.get('checkpoint_dir', 'checkpoints/rlan'))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # File logging
    tee_logger = None
    if log_cfg.get('log_to_file', True):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_path = checkpoint_dir / f'training_log_{timestamp}.txt'
        tee_logger = TeeLogger(log_path)
        sys.stdout = tee_logger
        print(f"Logging to: {log_path}")
        print(f"Timestamp: {datetime.now().isoformat()}")
        print(f"Python: {sys.version}")
        print(f"PyTorch: {torch.__version__}")
    
    # Handle resume logic
    resume_path = None
    if not args.no_resume:
        if args.resume == 'auto':
            resume_path = find_latest_checkpoint(str(checkpoint_dir))
            if resume_path:
                print(f"Auto-resume: found checkpoint {resume_path}")
        elif args.resume:
            resume_path = args.resume
    
    if resume_path:
        print(f"Resuming from: {resume_path}")
    else:
        print("Starting fresh training")
    
    # Create model
    model = create_model(config)
    model = model.to(device)
    
    # Print parameter count
    param_counts = model.count_parameters()
    print(f"\nModel parameters:")
    for name, count in param_counts.items():
        print(f"  {name}: {count:,}")
    
    # ================================================================
    # MODULE ABLATION STATUS
    # ================================================================
    model_cfg = config['model']
    print(f"\n{'='*60}")
    print("MODULE ABLATION STATUS")
    print(f"{'='*60}")
    
    # Core modules
    ctx_enc = model_cfg.get('use_context_encoder', True)
    use_dsc = model_cfg.get('use_dsc', True)
    use_msre = model_cfg.get('use_msre', True)
    use_lcr = model_cfg.get('use_lcr', True)
    use_sph = model_cfg.get('use_sph', True)
    use_act = model_cfg.get('use_act', False)
    use_learned = model_cfg.get('use_learned_pos', False)
    
    print(f"  ContextEncoder: {'ENABLED' if ctx_enc else 'DISABLED'} (task signal from demos)")
    print(f"  DSC:            {'ENABLED' if use_dsc else 'DISABLED'} (dynamic spatial clues - CORE)")
    print(f"  MSRE:           {'ENABLED' if use_msre else 'DISABLED'} (multi-scale relative encoding - CORE)")
    print(f"  LCR:            {'ENABLED' if use_lcr else 'DISABLED'} (latent counting registers)")
    print(f"  SPH:            {'ENABLED' if use_sph else 'DISABLED'} (symbolic predicate heads)")
    print(f"  ACT:            {'ENABLED' if use_act else 'DISABLED'} (adaptive computation time)")
    print(f"  Pos Encoding:   {'LEARNED' if use_learned else 'SINUSOIDAL'}")
    
    # Identify ablation mode
    if use_dsc and use_msre and not use_lcr and not use_sph:
        print(f"\n  >>> CORE ABLATION MODE: Testing DSC + MSRE novelty <<<")
    elif all([ctx_enc, use_dsc, use_msre, use_lcr, use_sph]):
        print(f"\n  >>> FULL MODEL MODE: All modules enabled <<<")
    else:
        disabled = []
        if not ctx_enc: disabled.append('ContextEncoder')
        if not use_dsc: disabled.append('DSC')
        if not use_msre: disabled.append('MSRE')
        if not use_lcr: disabled.append('LCR')
        if not use_sph: disabled.append('SPH')
        print(f"\n  >>> CUSTOM ABLATION: Disabled=[{', '.join(disabled)}] <<<")
    
    print(f"{'='*60}")

    # ================================================================
    # RLAN TRAINING REGIME CONFIGURATION
    # ================================================================
    train_cfg = config['training']
    print(f"\n{'='*60}")
    print("RLAN TRAINING REGIME")
    print(f"{'='*60}")
    print(f"  Batch Size: {train_cfg['batch_size']}")
    print(f"  Grad Accumulation: {train_cfg.get('grad_accumulation_steps', 1)}")
    print(f"  Effective Batch: {train_cfg['batch_size'] * train_cfg.get('grad_accumulation_steps', 1)}")
    print(f"  Learning Rate: {train_cfg['learning_rate']:.1e}")
    print(f"  Weight Decay: {train_cfg['weight_decay']}")
    print(f"  Optimizer: {train_cfg.get('optimizer', 'adamw')} (beta1={train_cfg.get('beta1', 0.9)}, beta2={train_cfg.get('beta2', 0.95)})")
    print(f"  Scheduler: {train_cfg.get('scheduler', 'cosine')}")
    print(f"  Warmup Epochs: {train_cfg.get('warmup_epochs', 10)}")
    print(f"  Max Epochs: {train_cfg['max_epochs']}")
    
    # Loss configuration
    loss_mode = train_cfg.get('loss_mode', 'focal_stablemax')
    print(f"\nLoss Configuration:")
    print(f"  Loss Mode: {loss_mode.upper()}")
    if 'focal' in loss_mode:
        print(f"    gamma={train_cfg['focal_gamma']}, alpha={train_cfg['focal_alpha']}")
    
    # Only show active auxiliary losses
    print(f"\nAuxiliary Loss Weights (only non-zero shown):")
    active_aux = []
    if train_cfg.get('lambda_entropy', 0) > 0:
        print(f"  lambda_entropy={train_cfg['lambda_entropy']} (DSC attention sharpness)")
        active_aux.append('entropy')
    if train_cfg.get('lambda_sparsity', 0) > 0:
        print(f"  lambda_sparsity={train_cfg['lambda_sparsity']} (clue efficiency)")
        active_aux.append('sparsity')
    if train_cfg.get('lambda_predicate', 0) > 0:
        print(f"  lambda_predicate={train_cfg['lambda_predicate']} (predicate diversity)")
        active_aux.append('predicate')
    if train_cfg.get('lambda_curriculum', 0) > 0:
        print(f"  lambda_curriculum={train_cfg['lambda_curriculum']} (complexity penalty)")
        active_aux.append('curriculum')
    if train_cfg.get('lambda_deep_supervision', 0) > 0:
        print(f"  lambda_deep_supervision={train_cfg['lambda_deep_supervision']} (intermediate step losses)")
        active_aux.append('deep_supervision')
    if not active_aux:
        print(f"  (none - pure task loss only)")
    
    print(f"\nTemperature Schedule (Gumbel-Softmax):")
    print(f"  Start: {train_cfg['temperature_start']}, End: {train_cfg['temperature_end']}")
    print(f"{'='*60}")
    
    # Create loss function
    loss_fn = create_loss(config)
    
    # Create datasets
    data_cfg = config['data']
    max_grid_size = data_cfg.get('max_grid_size', 30)
    augment_cfg = data_cfg.get('augmentation', {})
    augment_enabled = augment_cfg.get('enabled', True)
    color_permutation = augment_cfg.get('color_permutation', False)
    translational_augment = augment_cfg.get('translational', True)
    
    print(f"\nLoading data from: {data_cfg['train_path']}")
    print(f"Cache samples: {data_cfg.get('cache_samples', False)}")
    
    # ================================================================
    # AUGMENTATION CONFIGURATION (matching TRM's 3 augmentation types)
    # ================================================================
    print(f"\n{'='*60}")
    print("AUGMENTATION CONFIGURATION")
    print(f"{'='*60}")
    print(f"  1. Dihedral (D4 group): {'ENABLED' if augment_enabled else 'DISABLED'}")
    if augment_enabled:
        print(f"     - 8 transforms: identity, rot90, rot180, rot270, flipLR, flipUD, transpose, anti-transpose")
    print(f"  2. Color Permutation:   {'ENABLED' if color_permutation else 'DISABLED'}")
    if color_permutation:
        print(f"     - 9! = 362,880 permutations (colors 1-9 shuffled, 0 fixed)")
    print(f"  3. Translational:       {'ENABLED' if translational_augment else 'DISABLED'}")
    if translational_augment:
        print(f"     - Random offset within 30x30 canvas (~100 positions)")
    
    # Calculate augmentation diversity
    dihedral_transforms = 8 if augment_enabled else 1
    color_perms = 362880 if color_permutation else 1  # 9!
    translational_positions = 100 if translational_augment else 1  # approximate
    total_augmentations = dihedral_transforms * color_perms * translational_positions
    print(f"\n  Total Diversity: {dihedral_transforms} x {color_perms:,} x ~{translational_positions} = ~{total_augmentations:,} unique per task")
    print(f"  Mode: On-the-fly (EACH sample is NEW random augmentation)")
    print(f"  Advantage: Infinite diversity vs TRM's fixed 1000 samples")
    print(f"{'='*60}")
    
    # ================================================================
    # CURRICULUM LEARNING SETUP
    # ================================================================
    train_cfg = config['training']
    use_curriculum = train_cfg.get('use_curriculum', False)
    curriculum_stages = train_cfg.get('curriculum_stages', [100, 300, 600])
    
    if use_curriculum:
        print(f"\nCurriculum learning ENABLED (percentile-based):")
        print(f"  Stage 1: epochs 0-{curriculum_stages[0]-1} (70% easiest tasks)")
        print(f"  Stage 2: epochs {curriculum_stages[0]}-{curriculum_stages[1]-1} (90% of tasks)")
        print(f"  Stage 3: epochs {curriculum_stages[1]}-{curriculum_stages[2]-1} (100% all tasks)")
        print(f"  Stage 4: epochs {curriculum_stages[2]}+ (all tasks, no filtering)")
        # Start at stage 1 (70% of tasks)
        current_curriculum_stage = 1
    else:
        print("\nCurriculum learning DISABLED (using all data from epoch 1, like TRM)")
        current_curriculum_stage = 0  # 0 means all data
    
    # Create initial train loader (with curriculum stage if enabled)
    train_loader = create_train_loader(
        config,
        curriculum_stage=current_curriculum_stage,
        max_grid_size=max_grid_size,
    )
    
    # Eval dataset (no curriculum filtering - always full eval)
    eval_dataset = ARCDataset(
        data_cfg['eval_path'],
        max_size=max_grid_size,
        augment=False,
        color_permutation=False,
    )
    
    # Create eval loader
    batch_size = train_cfg['batch_size']
    eval_batch_size = train_cfg.get('eval_batch_size', batch_size)
    num_workers = data_cfg.get('num_workers', 0)
    pin_memory = data_cfg.get('pin_memory', True)
    
    collate_fn = partial(collate_sci_arc, max_grid_size=max_grid_size)
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=min(num_workers, 4),  # Fewer workers for eval
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    
    print(f"Train samples: {len(train_loader.dataset)}, batches: {len(train_loader)}")
    print(f"Eval samples: {len(eval_dataset)}, batches: {len(eval_loader)}")
    
    # Create optimizer and scheduler (needs loader length for OneCycle)
    optimizer, scheduler = create_optimizer(model, config, steps_per_epoch=len(train_loader))
    
    # Setup mixed precision
    scaler = None
    if config['device'].get('mixed_precision', False) and device.type == 'cuda':
        scaler = GradScaler('cuda')
        print("Using mixed precision training (AMP)")
    
    # Initialize wandb if enabled (disabled by default - not installed in production)
    wandb_enabled = False
    if log_cfg.get('use_wandb', False) and WANDB_AVAILABLE:
        try:
            wandb.login()
            wandb.init(
                project=log_cfg.get('wandb_project', 'rlan-arc'),
                name=log_cfg.get('wandb_run_name') or f"rlan-{time.strftime('%Y%m%d-%H%M%S')}",
                config=config,
            )
            wandb_enabled = True
            print("WandB logging enabled")
        except Exception as e:
            print(f"Warning: WandB init failed: {e}")
    else:
        print("WandB logging disabled (use_wandb=false or wandb not installed)")
    
    # Resume from checkpoint if specified
    start_epoch = 0
    global_step = 0
    best_task_accuracy = 0.0
    
    if resume_path:
        start_epoch, global_step, best_task_accuracy = load_checkpoint(
            model, optimizer, scheduler, resume_path
        )
    
    # Initialize EMA for stable evaluation
    use_ema = config.get('training', {}).get('use_ema', True)
    ema_decay = config.get('training', {}).get('ema_decay', 0.999)
    ema = None
    if use_ema:
        ema = EMAHelper(model, mu=ema_decay, device=device)
        print(f"EMA enabled with decay={ema_decay}")
    
    # Training loop
    max_epochs = config['training']['max_epochs']
    save_every = log_cfg.get('save_every', 10)
    eval_every = log_cfg.get('eval_every', 1)
    keep_last_n = log_cfg.get('keep_last_n', 5)
    
    print(f"\nStarting training from epoch {start_epoch} to {max_epochs}")
    print("=" * 60)
    
    # Collapse detection state
    collapse_warnings = 0
    max_collapse_warnings = 5  # Stop after this many consecutive warnings
    
    for epoch in range(start_epoch, max_epochs):
        epoch_start = time.time()
        
        print(f"\nEpoch {epoch + 1}/{max_epochs}")
        print("-" * 40)
        
        # ================================================================
        # CURRICULUM STAGE CHECK - Recreate dataloader if stage changes
        # ================================================================
        if use_curriculum:
            new_stage = get_curriculum_stage(epoch, curriculum_stages)
            if new_stage != current_curriculum_stage:
                # Percentile-based stage names
                stage_names = {0: "ALL 100%", 1: "70% EASY", 2: "90% TASKS", 3: "100% ALL"}
                print(f"\n" + "=" * 60)
                print(f"  CURRICULUM STAGE TRANSITION: {stage_names.get(current_curriculum_stage, '?')} -> {stage_names.get(new_stage, '?')}")
                print(f"=" * 60)
                
                current_curriculum_stage = new_stage
                train_loader = create_train_loader(
                    config,
                    curriculum_stage=current_curriculum_stage,
                    max_grid_size=max_grid_size,
                )
                print(f"  New train samples: {len(train_loader.dataset)}, batches: {len(train_loader)}")
        
        # Train
        train_losses, global_step = train_epoch(
            model, train_loader, loss_fn, optimizer, device,
            epoch, config, scaler, global_step, ema
        )
        
        # Update scheduler
        scheduler.step()
        
        epoch_time = time.time() - epoch_start
        
        # Log curriculum stage if enabled
        stage_str = ""
        if use_curriculum:
            stage_names = {0: "100%", 1: "70%", 2: "90%", 3: "100%"}
            stage_str = f", Curriculum: {stage_names.get(current_curriculum_stage, '?')} tasks"
        
        # Get loss mode for accurate logging
        loss_mode = train_losses.get('loss_mode', 'focal')
        task_loss_val = train_losses.get('task_loss', train_losses.get('focal_loss', 0))
        deep_sup_loss = train_losses.get('deep_supervision_loss', 0)
        
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Total Loss: {train_losses['total_loss']:.4f}")
        print(f"  Task Loss ({loss_mode}): {task_loss_val:.4f}")
        
        # Only show auxiliary losses if they have non-zero weight in config
        if train_cfg.get('lambda_entropy', 0) > 0:
            print(f"  Entropy Loss: {train_losses.get('entropy_loss', 0):.4f} (weight={train_cfg['lambda_entropy']})")
        if train_cfg.get('lambda_sparsity', 0) > 0:
            print(f"  Sparsity Loss: {train_losses.get('sparsity_loss', 0):.4f} (weight={train_cfg['lambda_sparsity']})")
        if train_cfg.get('lambda_predicate', 0) > 0:
            print(f"  Predicate Loss: {train_losses.get('predicate_loss', 0):.4f} (weight={train_cfg['lambda_predicate']})")
        if train_cfg.get('lambda_curriculum', 0) > 0:
            print(f"  Curriculum Loss: {train_losses.get('curriculum_loss', 0):.4f} (weight={train_cfg['lambda_curriculum']})")
        if train_cfg.get('lambda_deep_supervision', 0) > 0:
            print(f"  Deep Supervision: {deep_sup_loss:.4f} (weight={train_cfg['lambda_deep_supervision']})")
        if train_cfg.get('lambda_act', 0) > 0 and config['model'].get('use_act', False):
            act_loss_val = train_losses.get('act_loss', 0)
            print(f"  ACT Loss: {act_loss_val:.4f} (weight={train_cfg['lambda_act']})")
        
        print(f"  Time: {epoch_time:.1f}s, LR: {optimizer.param_groups[0]['lr']:.2e}{stage_str}")
        
        # ================================================================
        # AUGMENTATION DIVERSITY LOGGING (CRITICAL for quality assurance)
        # ================================================================
        # This verifies that on-the-fly augmentation is truly random and
        # covers all transformation types uniformly each epoch.
        # ================================================================
        aug_stats = train_losses.get('aug_stats', {})
        total_samples = train_losses.get('total_samples', 0)
        num_batches = train_losses.get('num_batches', 0)
        
        # Initialize augmentation metrics (for logging even if wandb not installed)
        color_pct = 0.0
        trans_pct = 0.0
        
        print(f"  Samples Processed: {total_samples} ({num_batches} batches)")
        
        if aug_stats and total_samples > 0:
            dihedral_counts = aug_stats.get('dihedral_counts', [0]*8)
            color_perm_count = aug_stats.get('color_perm_count', 0)
            translational_count = aug_stats.get('translational_count', 0)
            
            # Dihedral distribution (should be ~uniform across 8 transforms)
            dihedral_total = sum(dihedral_counts)
            if dihedral_total > 0:
                dihedral_pcts = [f"{c/dihedral_total*100:.1f}%" for c in dihedral_counts]
                print(f"  Dihedral Distribution: [{', '.join(dihedral_pcts)}]")
                
                # Check for uniformity (should be ~12.5% each)
                expected_pct = 100.0 / 8  # 12.5%
                max_deviation = max(abs(c/dihedral_total*100 - expected_pct) for c in dihedral_counts)
                if max_deviation > 5.0:  # More than 5% deviation from uniform
                    print(f"    [!] Non-uniform dihedral distribution (max dev: {max_deviation:.1f}%)")
            
            # Color permutation rate
            color_pct = color_perm_count / total_samples * 100 if total_samples > 0 else 0
            print(f"  Color Permutation: {color_pct:.1f}% ({color_perm_count}/{total_samples})")
            
            # Translational augmentation rate
            trans_pct = translational_count / total_samples * 100 if total_samples > 0 else 0
            unique_offsets = aug_stats.get('unique_offsets', 0)
            print(f"  Translational Aug: {trans_pct:.1f}% ({translational_count}/{total_samples}), unique offsets: {unique_offsets}")
            
            # Overall augmentation quality indicator
            aug_enabled_count = dihedral_total  # At least dihedral is applied
            aug_quality = "GOOD" if (color_pct > 90 and trans_pct > 80) else "OK" if aug_enabled_count > 0 else "NONE"
            print(f"  Aug Quality: {aug_quality}")
        
        # ================================================================
        # TRAINING DIAGNOSTICS (CRITICAL for debugging learning dynamics)
        # ================================================================
        diagnostics = train_losses.get('diagnostics', {})
        if diagnostics:
            print("  --- Training Diagnostics ---")
            
            # Deep supervision verification
            all_logits_count = diagnostics.get('all_logits_count', 0)
            if all_logits_count > 0:
                print(f"  Solver Steps: {all_logits_count} (deep supervision active)")
                
                # Per-step loss breakdown
                per_step_loss = diagnostics.get('per_step_loss', [])
                if per_step_loss:
                    loss_str = ', '.join(f"{l:.3f}" for l in per_step_loss)
                    print(f"  Per-Step Loss: [{loss_str}]")
                    # Check if loss decreases across steps (good = later steps better)
                    if len(per_step_loss) > 1:
                        if per_step_loss[-1] < per_step_loss[0]:
                            improvement = (per_step_loss[0] - per_step_loss[-1]) / per_step_loss[0] * 100
                            print(f"    Step improvement: {improvement:.1f}% (later steps better)")
                        else:
                            print(f"    [!] Later steps NOT improving - solver not refining!")
            else:
                print(f"  Solver Steps: [!] NO INTERMEDIATE LOGITS (deep supervision disabled!)")
            
            # Gradient flow diagnostics
            dsc_grad = diagnostics.get('dsc_grad_norm_sum', 0)
            enc_grad = diagnostics.get('encoder_grad_norm_sum', 0)
            solver_grad = diagnostics.get('solver_grad_norm_sum', 0)
            print(f"  Grad Norms: DSC={dsc_grad:.4f}, Encoder={enc_grad:.4f}, Solver={solver_grad:.4f}")
            if dsc_grad < 0.001 and enc_grad > 0:
                print(f"    [!] DSC gradients near zero - not learning!")
            if solver_grad < 0.001 and enc_grad > 0:
                print(f"    [!] Solver gradients near zero - check architecture!")
            
            # Attention sharpness (is DSC focusing?)
            attn_max = diagnostics.get('attn_max_mean', 0)
            attn_min = diagnostics.get('attn_min_mean', 0)
            if attn_max > 0:
                print(f"  Attention: max={attn_max:.4f}, min={attn_min:.6f}")
                # For a 30x30 grid, uniform attention = 1/900 = 0.0011
                # Sharp attention should have max >> 0.0011
                if attn_max < 0.01:
                    print(f"    [!] Attention too diffuse (max < 0.01) - DSC not focusing!")
                elif attn_max > 0.1:
                    print(f"    Attention is sharp (good!)")
            
            # Stop probability (how many clues used)
            stop_prob = diagnostics.get('stop_prob_mean', 0)
            if stop_prob > 0:
                clues_used = (1 - stop_prob) * all_logits_count if all_logits_count > 0 else 0
                print(f"  Stop Prob: {stop_prob:.3f} (approx {clues_used:.1f} clues active)")
            
            # Per-clue entropy breakdown
            per_clue_entropy = diagnostics.get('per_clue_entropy', [])
            if per_clue_entropy:
                entropy_str = ', '.join(f"{e:.2f}" for e in per_clue_entropy)
                mean_entropy = sum(per_clue_entropy) / len(per_clue_entropy)
                max_entropy = 6.80  # ln(900) for 30x30 grid
                print(f"  Per-Clue Entropy: [{entropy_str}] (mean={mean_entropy:.2f}, max={max_entropy:.2f})")
                
                # Check if all clues have similar entropy (bad - not differentiating)
                if len(per_clue_entropy) > 1:
                    entropy_std = (sum((e - mean_entropy)**2 for e in per_clue_entropy) / len(per_clue_entropy)) ** 0.5
                    if entropy_std < 0.1:
                        print(f"    [!] Clues have uniform entropy (std={entropy_std:.3f}) - not differentiating!")
                
                # Check if entropy is too high (attention too diffuse)
                if mean_entropy > 5.0:
                    print(f"    [!] High entropy ({mean_entropy:.2f}) - attention still diffuse!")
                elif mean_entropy < 3.0:
                    print(f"    Good entropy ({mean_entropy:.2f}) - attention is focused!")
            
            # Centroid spread
            centroid_spread = diagnostics.get('centroid_spread', 0)
            if centroid_spread > 0:
                print(f"  Centroid Spread: {centroid_spread:.2f} (higher=more diverse)")
                if centroid_spread < 2.0:
                    print(f"    [!] Clues clustered (spread < 2) - should spread out")
                elif centroid_spread > 8.0:
                    print(f"    Good spread - clues are distributed across grid")
        
        # Evaluate
        if (epoch + 1) % eval_every == 0:
            # Use EMA model for evaluation if available
            if ema is not None:
                eval_model = ema.ema_copy(model)
                eval_model = eval_model.to(device)
            else:
                eval_model = model
            
            eval_metrics = evaluate(eval_model, eval_loader, device)
            
            # Core metrics
            print(f"  Pixel Accuracy: {eval_metrics['pixel_accuracy']:.4f}")
            print(f"  Task Accuracy: {eval_metrics['task_accuracy']:.4f}")
            print(f"  Non-BG Accuracy: {eval_metrics['non_bg_accuracy']:.4f}")
            print(f"  BG Ratio (pred/target): {eval_metrics['bg_ratio_pred']:.2%} / {eval_metrics['bg_ratio_target']:.2%}")
            print(f"  Colors Used (pred/target): {eval_metrics['colors_used']} / {eval_metrics['colors_target']}")
            
            # Module-specific metrics for debugging
            print(f"  DSC Entropy: {eval_metrics['dsc_entropy']:.4f} (lower=sharper)")
            print(f"  DSC Clues Used: {eval_metrics['dsc_clues_used']:.2f}")
            print(f"  Predicate Activation: {eval_metrics['predicate_activation']:.4f}")
            
            if ema is not None:
                print("  (Using EMA weights for evaluation)")
            
            # ============================================================
            # BACKGROUND COLLAPSE DETECTION - CRITICAL FOR DEBUGGING
            # ============================================================
            is_collapsing = False
            collapse_reasons = []
            
            # Check 1: Excessive background prediction
            bg_excess = eval_metrics['bg_ratio_pred'] - eval_metrics['bg_ratio_target']
            if bg_excess > 0.15:  # 15% more BG than target
                is_collapsing = True
                collapse_reasons.append(f"BG excess: {bg_excess:.1%}")
            
            # Check 2: Zero non-background accuracy
            if eval_metrics['non_bg_accuracy'] < 0.01 and epoch > 10:
                is_collapsing = True
                collapse_reasons.append(f"Non-BG acc: {eval_metrics['non_bg_accuracy']:.1%}")
            
            # Check 3: Too few colors predicted
            if eval_metrics['colors_used'] < eval_metrics['colors_target'] - 3:
                is_collapsing = True
                collapse_reasons.append(f"Colors: {eval_metrics['colors_used']}/{eval_metrics['colors_target']}")
            
            # Check 4: DSC not differentiating (very high entropy)
            if eval_metrics['dsc_entropy'] > 5.0 and epoch > 20:
                is_collapsing = True
                collapse_reasons.append(f"DSC entropy: {eval_metrics['dsc_entropy']:.2f}")
            
            if is_collapsing:
                collapse_warnings += 1
                print(f"\n  ⚠️  [WARNING] BACKGROUND COLLAPSE DETECTED! ({collapse_warnings}/{max_collapse_warnings})")
                print(f"      Reasons: {', '.join(collapse_reasons)}")
                print(f"      Consider: Lower learning rate, increase focal_alpha, check ContextEncoder")
                
                if collapse_warnings >= max_collapse_warnings:
                    print(f"\n  🛑 [CRITICAL] {max_collapse_warnings} consecutive collapse warnings!")
                    print(f"      Training appears to have failed. Please review:")
                    print(f"      1. ContextEncoder - is it receiving training pairs?")
                    print(f"      2. focal_alpha - try increasing to 0.5-0.75")
                    print(f"      3. learning_rate - try reducing by 2-5x")
                    print(f"      4. lambda_entropy - try increasing to focus attention")
                    print(f"\n      Stopping training to prevent wasted compute.")
                    break
            else:
                collapse_warnings = 0  # Reset on good epoch
            
            # Log to wandb - all loss components for complete monitoring
            if wandb_enabled:
                # Prepare augmentation stats for wandb
                aug_log = {}
                if aug_stats:
                    dihedral_counts = aug_stats.get('dihedral_counts', [0]*8)
                    dihedral_total = sum(dihedral_counts)
                    if dihedral_total > 0:
                        for i, count in enumerate(dihedral_counts):
                            aug_log[f'aug/dihedral_{i}'] = count / dihedral_total
                    aug_log['aug/color_perm_pct'] = color_pct
                    aug_log['aug/translational_pct'] = trans_pct
                    aug_log['aug/total_samples'] = total_samples
                
                wandb.log({
                    'epoch': epoch + 1,
                    # All loss components
                    'train_loss': train_losses['total_loss'],
                    'focal_loss': train_losses['focal_loss'],
                    'entropy_loss': train_losses.get('entropy_loss', 0.0),
                    'sparsity_loss': train_losses.get('sparsity_loss', 0.0),
                    'predicate_loss': train_losses.get('predicate_loss', 0.0),
                    'curriculum_loss': train_losses.get('curriculum_loss', 0.0),
                    # Evaluation metrics
                    'pixel_accuracy': eval_metrics['pixel_accuracy'],
                    'task_accuracy': eval_metrics['task_accuracy'],
                    'non_bg_accuracy': eval_metrics['non_bg_accuracy'],
                    'bg_ratio_pred': eval_metrics['bg_ratio_pred'],
                    'bg_ratio_target': eval_metrics['bg_ratio_target'],
                    'colors_used': eval_metrics['colors_used'],
                    'colors_target': eval_metrics['colors_target'],
                    # Module debugging
                    'dsc_entropy': eval_metrics['dsc_entropy'],
                    'dsc_clues_used': eval_metrics['dsc_clues_used'],
                    'predicate_activation': eval_metrics['predicate_activation'],
                    # Training state
                    'lr': optimizer.param_groups[0]['lr'],
                    'temperature': get_temperature(epoch, config),
                    'collapse_warnings': collapse_warnings,
                    # Augmentation diversity (CRITICAL for quality assurance)
                    **aug_log,
                })
            
            # Save best model
            if eval_metrics['task_accuracy'] > best_task_accuracy:
                best_task_accuracy = eval_metrics['task_accuracy']
                best_path = checkpoint_dir / "best.pt"
                save_checkpoint(
                    model, optimizer, scheduler, epoch, global_step,
                    train_losses, best_task_accuracy, config, str(best_path)
                )
                print(f"  New best task accuracy: {best_task_accuracy:.4f}")
        
        # Save periodic checkpoint
        if (epoch + 1) % save_every == 0:
            checkpoint_path = checkpoint_dir / f"epoch_{epoch + 1}.pt"
            save_checkpoint(
                model, optimizer, scheduler, epoch, global_step,
                train_losses, best_task_accuracy, config, str(checkpoint_path)
            )
            cleanup_old_checkpoints(checkpoint_dir, keep_last_n)
        
        # Save latest checkpoint
        latest_path = checkpoint_dir / "latest.pt"
        save_checkpoint(
            model, optimizer, scheduler, epoch, global_step,
            train_losses, best_task_accuracy, config, str(latest_path)
        )
    
    print("\n" + "=" * 60)
    print(f"Training complete! Best task accuracy: {best_task_accuracy:.4f}")
    
    # Cleanup
    if tee_logger:
        tee_logger.close()


if __name__ == "__main__":
    main()
