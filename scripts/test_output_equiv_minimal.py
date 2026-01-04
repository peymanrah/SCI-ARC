#!/usr/bin/env python
"""
Minimal test to reproduce OUTPUT_EQUIV "too many indices" error.
"""
import sys
import traceback
sys.path.insert(0, '.')

import torch
import torch.nn as nn

# Minimal mock to identify the exact failure
print("=== OUTPUT_EQUIV Error Reproduction ===\n")

# Check what dimensions are expected
print("Checking apply_augmentation behavior:")

class MockOutputEquiv:
    def apply_augmentation(self, tensor, aug_type, inverse=False):
        """Apply or inverse-apply an augmentation to (B, C, H, W) or (B, H, W) tensor."""
        is_3d = tensor.dim() == 3
        if is_3d:
            tensor = tensor.unsqueeze(1)
        
        if aug_type == 'rotate_90':
            if inverse:
                result = torch.rot90(tensor, k=-1, dims=(2, 3))
            else:
                result = torch.rot90(tensor, k=1, dims=(2, 3))
        elif aug_type == 'rotate_180':
            result = torch.rot90(tensor, k=2, dims=(2, 3))
        else:
            result = tensor
            
        if is_3d:
            result = result.squeeze(1)
            
        return result

mock = MockOutputEquiv()

# Test with different tensor shapes
test_cases = [
    ("test_inputs 3D", torch.randn(2, 30, 30)),
    ("test_inputs 4D", torch.randn(2, 1, 30, 30)),
    ("train_inputs 4D", torch.randn(2, 3, 30, 30)),  # B, N, H, W
    ("logits 4D", torch.randn(2, 11, 30, 30)),  # B, C, H, W
]

for name, tensor in test_cases:
    try:
        result = mock.apply_augmentation(tensor, 'rotate_90')
        print(f"✓ {name}: {tensor.shape} -> {result.shape}")
    except Exception as e:
        print(f"✗ {name}: {tensor.shape} -> {e}")

print("\n=== Now testing with actual model ===\n")

# Try to load and test with actual model
try:
    from sci_arc.models.rlan import RLAN, RLANConfig
    from sci_arc.models.rlan_modules.loo_training import OutputEquivarianceLoss, EquivarianceConfig
    
    # Create minimal model with config
    config = RLANConfig(
        hidden_dim=64,
        num_solver_steps=3,
        use_context_encoder=True,
        use_hyperlora=False,  # Disable to simplify
    )
    model = RLAN(config=config)
    model.eval()
    
    # Create OUTPUT_EQUIV loss
    equiv_config = EquivarianceConfig(enabled=True)
    output_equiv = OutputEquivarianceLoss(equiv_config)
    
    # Create test tensors
    B, N, H, W = 2, 3, 15, 15
    test_inputs = torch.randint(0, 10, (B, H, W))
    train_inputs = torch.randint(0, 10, (B, N, H, W))
    train_outputs = torch.randint(0, 10, (B, N, H, W))
    pair_mask = torch.ones(B, N, dtype=torch.bool)
    
    print(f"Input shapes:")
    print(f"  test_inputs: {test_inputs.shape}")
    print(f"  train_inputs: {train_inputs.shape}")
    print(f"  train_outputs: {train_outputs.shape}")
    
    # Run forward to get original logits
    with torch.no_grad():
        outputs = model(
            test_inputs,
            train_inputs=train_inputs,
            train_outputs=train_outputs,
            pair_mask=pair_mask,
            return_intermediates=True,  # Need dict for ['logits']
        )
    original_logits = outputs['logits'].clone().requires_grad_(True)
    print(f"  original_logits: {original_logits.shape}")
    
    print("\nAttempting OUTPUT_EQUIV forward...")
    
    # Now try OUTPUT_EQUIV
    try:
        loss, metrics = output_equiv(
            model=model,
            test_inputs=test_inputs,
            train_inputs=train_inputs,
            train_outputs=train_outputs,
            pair_mask=pair_mask,
            original_logits=original_logits,
        )
        print(f"✓ OUTPUT_EQUIV succeeded: loss = {loss.item():.4f}")
    except Exception as e:
        print(f"✗ OUTPUT_EQUIV failed: {e}")
        traceback.print_exc()
        
except Exception as e:
    print(f"Setup failed: {e}")
    traceback.print_exc()

print("\n=== Done ===")
