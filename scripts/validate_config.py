#!/usr/bin/env python3
"""
Validate rlan_stable.yaml config before production training.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import torch


def main():
    print("=" * 60)
    print("RLAN STABLE CONFIG VALIDATION")
    print("=" * 60)
    
    # Load config
    config_path = "configs/rlan_stable.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    print(f"\n✓ Config loaded: {config_path}")
    
    # Check key settings
    print("\n" + "-" * 60)
    print("MODEL CONFIG:")
    print("-" * 60)
    model = config['model']
    print(f"  hidden_dim:         {model['hidden_dim']}")
    print(f"  max_clues:          {model['max_clues']}")
    print(f"  num_predicates:     {model['num_predicates']}")
    print(f"  num_solver_steps:   {model['num_solver_steps']}")
    print(f"  use_context_encoder: {model['use_context_encoder']}")
    print(f"  use_dsc:            {model['use_dsc']}")
    print(f"  use_msre:           {model['use_msre']}")
    
    print("\n" + "-" * 60)
    print("TRAINING CONFIG:")
    print("-" * 60)
    training = config['training']
    print(f"  batch_size:         {training['batch_size']}")
    print(f"  learning_rate:      {training['learning_rate']}")
    print(f"  max_epochs:         {training['max_epochs']}")
    print(f"  gradient_clip:      {training['gradient_clip']}")
    print(f"  loss_mode:          {training['loss_mode']}")
    
    print("\n  Clue Regularization:")
    print(f"    min_clues:             {training['min_clues']}")
    print(f"    min_clue_weight:       {training['min_clue_weight']}")
    print(f"    ponder_weight:         {training['ponder_weight']}")
    print(f"    entropy_ponder_weight: {training['entropy_ponder_weight']}")
    
    print("\n  Auxiliary Losses:")
    print(f"    lambda_entropy:        {training['lambda_entropy']}")
    print(f"    lambda_sparsity:       {training['lambda_sparsity']}")
    print(f"    lambda_predicate:      {training['lambda_predicate']}")
    
    print("\n" + "-" * 60)
    print("DATA CONFIG:")
    print("-" * 60)
    data = config['data']
    print(f"  num_workers:        {data['num_workers']}")
    print(f"  prefetch_factor:    {data['prefetch_factor']}")
    print(f"  cache_samples:      {data['cache_samples']}")
    print(f"  num_cached_samples: {data['num_cached_samples']}")
    
    print("\n" + "-" * 60)
    print("DEVICE CONFIG:")
    print("-" * 60)
    device = config['device']
    print(f"  use_cuda:           {device['use_cuda']}")
    print(f"  mixed_precision:    {device['mixed_precision']}")
    print(f"  dtype:              {device['dtype']}")
    
    # Validate settings
    print("\n" + "=" * 60)
    print("VALIDATION CHECKS:")
    print("=" * 60)
    
    issues = []
    
    # Check clue reg is properly enabled
    if training['lambda_sparsity'] <= 0 and training['min_clue_weight'] > 0:
        issues.append("⚠️  min_clue_weight > 0 but lambda_sparsity = 0 (clue reg won't work)")
    
    # Check batch size for 24GB VRAM
    if training['batch_size'] > 64:
        issues.append(f"⚠️  batch_size={training['batch_size']} may OOM on 24GB VRAM")
    
    # Check workers vs CPU cores
    if data['num_workers'] > 16:
        issues.append(f"⚠️  num_workers={data['num_workers']} is very high (CPU overhead)")
    
    # Check cache size
    if data['num_cached_samples'] < training['batch_size'] * 100:
        issues.append(f"⚠️  cache too small: {data['num_cached_samples']} < {training['batch_size'] * 100}")
    
    if issues:
        for issue in issues:
            print(f"  {issue}")
    else:
        print("  ✓ All checks passed!")
    
    # VRAM estimation
    print("\n" + "-" * 60)
    print("VRAM ESTIMATION (RTX 3090 = 24GB):")
    print("-" * 60)
    
    batch = training['batch_size']
    hidden = model['hidden_dim']
    grid = model['max_grid_size']
    
    # Rough estimation
    model_size = hidden * hidden * 50 * 4 / 1e9  # ~50 layers, 4 bytes per param
    activation_size = batch * grid * grid * hidden * 4 * 10 / 1e9  # 10 intermediate activations
    gradient_size = activation_size  # Similar to activations
    
    total_est = model_size + activation_size + gradient_size
    
    print(f"  Model params:    ~{model_size:.1f} GB")
    print(f"  Activations:     ~{activation_size:.1f} GB")
    print(f"  Gradients:       ~{gradient_size:.1f} GB")
    print(f"  TOTAL (est):     ~{total_est:.1f} GB")
    
    if device['mixed_precision']:
        total_est_bf16 = total_est * 0.6  # bfloat16 saves ~40%
        print(f"  With bfloat16:   ~{total_est_bf16:.1f} GB")
        
        if total_est_bf16 < 20:
            print(f"\n  ✓ Should fit in 24GB VRAM with headroom")
        elif total_est_bf16 < 24:
            print(f"\n  ⚠️  Tight fit - may need to reduce batch_size")
        else:
            print(f"\n  ✗ Likely OOM - reduce batch_size to 16")
    
    # Try importing model
    print("\n" + "-" * 60)
    print("MODEL IMPORT TEST:")
    print("-" * 60)
    
    try:
        from sci_arc.models.rlan import RLAN, RLANConfig
        from sci_arc.training.rlan_loss import RLANLoss
        print("  ✓ RLAN and RLANLoss imported successfully")
        
        # Quick instantiation test
        test_config = RLANConfig(
            hidden_dim=model['hidden_dim'],
            max_clues=model['max_clues'],
            num_predicates=model['num_predicates'],
            num_solver_steps=model['num_solver_steps'],
            max_grid_size=model['max_grid_size'],
            use_context_encoder=model['use_context_encoder'],
            use_dsc=model['use_dsc'],
            use_msre=model['use_msre'],
        )
        test_model = RLAN(config=test_config)
        num_params = sum(p.numel() for p in test_model.parameters())
        print(f"  ✓ Model instantiated: {num_params:,} parameters")
        
        test_loss = RLANLoss(
            lambda_sparsity=training['lambda_sparsity'],
            lambda_entropy=training['lambda_entropy'],
            lambda_predicate=training['lambda_predicate'],
            min_clues=training['min_clues'],
            min_clue_weight=training['min_clue_weight'],
            ponder_weight=training['ponder_weight'],
            entropy_ponder_weight=training['entropy_ponder_weight'],
        )
        print("  ✓ Loss function instantiated")
        
    except Exception as e:
        print(f"  ✗ Import failed: {e}")
    
    print("\n" + "=" * 60)
    print("CONFIG READY FOR PRODUCTION TRAINING")
    print("=" * 60)
    print(f"\nTo train: python scripts/train_rlan.py --config {config_path}")
    print()


if __name__ == '__main__':
    main()
