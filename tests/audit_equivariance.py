
import torch
import torch.nn as nn
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sci_arc.models.rlan_modules.loo_training import AugmentationEquivarianceLoss, EquivarianceConfig
from sci_arc.models.rlan_modules.hyper_lora import HyperLoRA, HyperLoRAConfig

def test_equivariance_loss():
    print("\n=== Testing Augmentation Equivariance Loss ===")
    
    # 1. Setup Components
    hidden_dim = 32
    config = EquivarianceConfig(enabled=True, num_augmentations=2)
    loss_fn = AugmentationEquivarianceLoss(config, hidden_dim)
    
    hyper_config = HyperLoRAConfig(hidden_dim=hidden_dim, context_dim=hidden_dim)
    hyper_lora = HyperLoRA(hyper_config)
    
    # 2. Create Dummy Data
    B, N, D, H, W = 2, 3, hidden_dim, 10, 10
    support_features = torch.randn(B, N, D, H, W)
    
    # 3. Simulate Training Loop Logic
    # Get original context
    original_context = hyper_lora.pool_context(support_features)
    
    # Generate augmented contexts
    augmented_contexts = {}
    aug_types = ['rotate_90', 'flip_h']
    
    for aug_type in aug_types:
        # Apply augmentation (logic from train_rlan.py)
        # support_features: (B, N, D, H, W)
        # apply_augmentation expects (B, N, H, W, D) if we permute
        
        # In train_rlan.py:
        # aug_features = equiv_loss_fn.apply_augmentation(
        #     support_features.permute(0, 1, 3, 4, 2),  # (B, N, H, W, D)
        #     aug_type
        # ).permute(0, 1, 4, 2, 3)  # Back to (B, N, D, H, W)
        
        permuted = support_features.permute(0, 1, 3, 4, 2) # (B, N, H, W, D)
        augmented = loss_fn.apply_augmentation(permuted, aug_type)
        aug_features = augmented.permute(0, 1, 4, 2, 3) # (B, N, D, H, W)
        
        aug_context = hyper_lora.pool_context(aug_features)
        augmented_contexts[aug_type] = aug_context
        
    # 4. Run Loss
    print("  Running Equivariance Loss...")
    try:
        loss, metrics = loss_fn(
            hyper_lora=hyper_lora,
            original_context=original_context,
            augmented_contexts=augmented_contexts
        )
        print(f"  Equivariance Loss computed: {loss.item()}")
        print(f"  Metrics: {metrics}")
        print("  [PASS] Equivariance Loss runs successfully.")
    except Exception as e:
        print(f"  [FAIL] Error during Equivariance Loss: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_equivariance_loss()
