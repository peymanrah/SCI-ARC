#!/usr/bin/env python
"""
SCI-ARC Training Script.

Usage:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/small.yaml --data.arc_dir /path/to/arc
    python scripts/train.py --resume checkpoints/checkpoint_epoch_50.pt
    python scripts/train.py --resume auto  # Auto-resume from latest checkpoint
"""

import argparse
import os
import sys
from pathlib import Path
from glob import glob

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
import torch
import numpy as np
import random

from sci_arc.models import SCIARC, SCIARCConfig
from sci_arc.data import SCIARCDataset, collate_sci_arc, create_dataloader
from sci_arc.training import SCIARCTrainer, TrainingConfig, SCIARCLoss


def find_latest_checkpoint(checkpoint_dir: str) -> str:
    """Find the most recent checkpoint in the checkpoint directory."""
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        return None
    
    # First, check for the dedicated "latest" checkpoint
    latest_path = checkpoint_path / 'checkpoint_latest.pt'
    if latest_path.exists():
        return str(latest_path)
    
    # Otherwise, find the highest epoch number
    checkpoints = list(checkpoint_path.glob('checkpoint_epoch_*.pt'))
    if not checkpoints:
        return None
    
    # Sort by epoch number
    def get_epoch(p):
        try:
            return int(p.stem.split('_')[-1])
        except:
            return -1
    
    checkpoints.sort(key=get_epoch, reverse=True)
    return str(checkpoints[0])


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def override_config(config: dict, overrides: list) -> dict:
    """Apply command-line overrides to config.
    
    Format: key.subkey=value
    """
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
            # Try to parse as Python literal
            import ast
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            # Keep as string
            pass
        
        current[final_key] = value
    
    return config


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


def build_model(config: dict) -> SCIARC:
    """Build SCI-ARC model from config."""
    model_cfg = config['model']
    
    model_config = SCIARCConfig(
        hidden_dim=model_cfg['hidden_dim'],
        num_colors=model_cfg['num_colors'],
        max_grid_size=model_cfg['max_grid_size'],
        num_structure_slots=model_cfg['num_structure_slots'],
        se_layers=model_cfg.get('num_abstraction_layers', model_cfg.get('se_layers', 2)),
        num_heads=model_cfg.get('structure_heads', model_cfg.get('num_heads', 8)),
        max_objects=model_cfg['max_objects'],
        H_cycles=model_cfg['H_cycles'],
        L_cycles=model_cfg['L_cycles'],
        L_layers=model_cfg['L_layers'],
        dropout=model_cfg.get('dropout', 0.1),
    )
    
    model = SCIARC(model_config)
    
    # Print model info
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: SCI-ARC")
    print(f"  Total parameters: {num_params:,}")
    print(f"  Trainable parameters: {num_trainable:,}")
    print(f"  Hidden dim: {model_config.hidden_dim}")
    print(f"  H_cycles: {model_config.H_cycles}, L_cycles: {model_config.L_cycles}")
    
    return model


def build_training_config(config: dict) -> TrainingConfig:
    """Build training config from dict."""
    train_cfg = config['training']
    log_cfg = config.get('logging', {})
    hw_cfg = config.get('hardware', {})
    
    return TrainingConfig(
        learning_rate=train_cfg['learning_rate'],
        weight_decay=train_cfg['weight_decay'],
        max_epochs=train_cfg['max_epochs'],
        warmup_epochs=train_cfg['warmup_epochs'],
        grad_clip=train_cfg['grad_clip'],
        grad_accumulation_steps=train_cfg.get('grad_accumulation_steps', 1),
        batch_size=train_cfg['batch_size'],
        eval_batch_size=train_cfg['eval_batch_size'],
        scl_weight=train_cfg['scl_weight'],
        ortho_weight=train_cfg['ortho_weight'],
        deep_supervision_weight=train_cfg['deep_supervision_weight'],
        scheduler_type=train_cfg.get('scheduler_type', 'cosine'),
        min_lr=train_cfg.get('min_lr', 1e-6),
        use_amp=train_cfg.get('use_amp', True),
        checkpoint_dir=log_cfg.get('checkpoint_dir', './checkpoints'),
        save_every=log_cfg.get('save_every', 5),
        keep_last_n=log_cfg.get('keep_last_n', 3),
        log_every=log_cfg.get('log_every', 10),
        eval_every=log_cfg.get('eval_every', 1),
        use_wandb=log_cfg.get('use_wandb', True),
        wandb_project=log_cfg.get('wandb_project', 'sci-arc'),
        wandb_run_name=log_cfg.get('wandb_run_name'),
        log_to_file=log_cfg.get('log_to_file', True),  # Enable file logging by default
        use_curriculum=train_cfg.get('use_curriculum', True),
        curriculum_stages=train_cfg.get('curriculum_stages', [10, 30, 60]),
        device=hw_cfg.get('device', 'cuda'),
        seed=hw_cfg.get('seed', 42),
    )


def main():
    parser = argparse.ArgumentParser(description='Train SCI-ARC model')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from (use "auto" for latest)')
    parser.add_argument('--no-resume', action='store_true',
                        help='Start fresh even if checkpoints exist')
    parser.add_argument('overrides', nargs='*',
                        help='Config overrides in format key.subkey=value')
    
    args = parser.parse_args()
    
    # Load and override config
    config = load_config(args.config)
    config = override_config(config, args.overrides)
    
    # Set seed
    hw_cfg = config.get('hardware', {})
    set_seed(hw_cfg.get('seed', 42), hw_cfg.get('deterministic', False))
    
    # Determine checkpoint directory
    log_cfg = config.get('logging', {})
    checkpoint_dir = log_cfg.get('checkpoint_dir', './checkpoints')
    
    # Handle resume logic
    resume_path = None
    if not args.no_resume:
        if args.resume == 'auto':
            # Find latest checkpoint
            resume_path = find_latest_checkpoint(checkpoint_dir)
            if resume_path:
                print(f"Auto-resume: found checkpoint {resume_path}")
        elif args.resume:
            resume_path = args.resume
    
    # Print config summary
    print("=" * 60)
    print("SCI-ARC Training")
    print("=" * 60)
    print(f"Config: {args.config}")
    if resume_path:
        print(f"Resuming from: {resume_path}")
    else:
        print("Starting fresh training")
    
    # Build model
    model = build_model(config)
    
    # Build data loaders
    data_cfg = config['data']
    train_cfg = config['training']
    hw_cfg = config.get('hardware', {})
    seed = hw_cfg.get('seed', 42)
    
    print(f"\nLoading data from: {data_cfg['arc_dir']}")
    
    # Windows with multiprocessing: use reasonable number of workers
    # With 24 CPU cores, 8 workers is efficient for keeping GPU fed
    # Set to 0 only if you encounter multiprocessing errors
    num_workers = data_cfg.get('num_workers', 8)
    cache_samples = data_cfg.get('cache_samples', False)
    cache_augmentations = data_cfg.get('cache_augmentations', 8)
    
    if cache_samples:
        print(f"Caching enabled: pre-generating {cache_augmentations} augmentations per task")
    else:
        print(f"Using {num_workers} data loading workers")
    
    train_loader = create_dataloader(
        data_dir=data_cfg['arc_dir'],
        split='training',
        batch_size=train_cfg['batch_size'],
        num_workers=num_workers,
        shuffle=True,
        augment=data_cfg.get('augment', True),
        max_grid_size=config['model'].get('max_grid_size', 30),
        seed=seed if hw_cfg.get('deterministic', False) else None,
        cache_samples=cache_samples,
        cache_augmentations=cache_augmentations,
        use_augment_family=data_cfg.get('use_augment_family', True),  # Critical for SCL learning
    )
    
    val_loader = create_dataloader(
        data_dir=data_cfg['arc_dir'],
        split='evaluation',
        batch_size=train_cfg['eval_batch_size'],
        num_workers=num_workers,
        shuffle=False,
        augment=False,
        max_grid_size=config['model'].get('max_grid_size', 30),
        seed=seed if hw_cfg.get('deterministic', False) else None,
        cache_samples=cache_samples,  # Cache validation too for consistency
        cache_augmentations=1,  # No augmentation variants needed for validation
    )
    
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Build loss function with flattened slots for SCL (prevents representation collapse)
    hidden_dim = config['model']['hidden_dim']
    num_structure_slots = config['model'].get('num_structure_slots', 8)
    
    # === CLASS WEIGHTS FOR BACKGROUND COLLAPSE PREVENTION ===
    # ARC grids are ~85% background (color 0). Standard CE optimizes for accuracy,
    # so the model learns "predict 0 everywhere" = 85% accuracy (local minimum).
    # Class weights penalize background errors less, forcing focus on content.
    device = hw_cfg.get('device', 'cuda')
    class_weights = None
    if train_cfg.get('use_class_weights', False):
        # Background (0) gets low weight, non-background gets high weight
        class_weights = torch.ones(config['model']['num_colors'], device=device)
        class_weights[0] = 0.1  # Background weight = 0.1 (10x less important)
        print(f"Using class weights: background=0.1, others=1.0")
    
    # Focal loss gamma: 0 = disabled, 2.0 = recommended for class imbalance
    focal_gamma = train_cfg.get('focal_gamma', 0.0)
    if focal_gamma > 0:
        print(f"Using Focal Loss with gamma={focal_gamma}")
    
    # Label smoothing: prevents overconfident predictions
    label_smoothing = train_cfg.get('label_smoothing', 0.0)
    if label_smoothing > 0:
        print(f"Using label smoothing: {label_smoothing}")
    
    loss_fn = SCIARCLoss(
        H_cycles=config['model']['H_cycles'],
        scl_weight=train_cfg['scl_weight'],
        orthogonality_weight=train_cfg['ortho_weight'],
        temperature=0.1,
        hidden_dim=hidden_dim,
        projection_dim=hidden_dim // 2,
        num_structure_slots=num_structure_slots,  # For SCL slot flattening
        label_smoothing=label_smoothing,
        focal_gamma=focal_gamma,
        class_weights=class_weights,
    )
    
    # Build training config
    training_config = build_training_config(config)
    
    # Disable wandb if not logged in (for local training)
    if training_config.use_wandb:
        try:
            import wandb
            wandb.login()
        except:
            print("Warning: wandb not configured, disabling logging")
            training_config.use_wandb = False
    
    # Create trainer
    trainer = SCIARCTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        config=training_config,
    )
    
    # Resume if specified
    if resume_path:
        trainer.load_checkpoint(resume_path)
    
    # Train
    trainer.train()
    
    print("\nTraining complete!")


if __name__ == '__main__':
    main()
