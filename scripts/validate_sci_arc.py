#!/usr/bin/env python
"""
SCI-ARC Validation Script

Comprehensive validation of SCI-ARC implementation:
1. Module imports and initialization
2. Forward pass validation
3. Training interface check
4. Parameter counting
5. Comparison with TRM baseline

Run this script to ensure the implementation is bug-free.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn


def print_header(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")


def print_result(check: str, passed: bool, details: str = ""):
    """Print a check result."""
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  [{status}] {check}")
    if details:
        print(f"          {details}")


def test_imports():
    """Test that all modules can be imported."""
    print_header("1. MODULE IMPORTS")
    
    all_passed = True
    
    try:
        from sci_arc.config import SCIARCConfig
        print_result("SCIARCConfig", True)
    except Exception as e:
        print_result("SCIARCConfig", False, str(e))
        all_passed = False
    
    try:
        from sci_arc.models.grid_encoder import GridEncoder
        print_result("GridEncoder", True)
    except Exception as e:
        print_result("GridEncoder", False, str(e))
        all_passed = False
    
    try:
        from sci_arc.models.structural_encoder import StructuralEncoder2D
        print_result("StructuralEncoder2D", True)
    except Exception as e:
        print_result("StructuralEncoder2D", False, str(e))
        all_passed = False
    
    try:
        from sci_arc.models.content_encoder import ContentEncoder2D
        print_result("ContentEncoder2D", True)
    except Exception as e:
        print_result("ContentEncoder2D", False, str(e))
        all_passed = False
    
    try:
        from sci_arc.models.causal_binding import CausalBinding2D
        print_result("CausalBinding2D", True)
    except Exception as e:
        print_result("CausalBinding2D", False, str(e))
        all_passed = False
    
    try:
        from sci_arc.models.recursive_refinement import RecursiveRefinement
        print_result("RecursiveRefinement", True)
    except Exception as e:
        print_result("RecursiveRefinement", False, str(e))
        all_passed = False
    
    try:
        from sci_arc.models.sci_arc import SCIARC
        print_result("SCIARC (main model)", True)
    except Exception as e:
        print_result("SCIARC (main model)", False, str(e))
        all_passed = False
    
    try:
        from sci_arc.training.losses import SCIARCLoss
        print_result("SCIARCLoss", True)
    except Exception as e:
        print_result("SCIARCLoss", False, str(e))
        all_passed = False
    
    try:
        from sci_arc.training.trainer import SCIARCTrainer
        print_result("SCIARCTrainer", True)
    except Exception as e:
        print_result("SCIARCTrainer", False, str(e))
        all_passed = False
    
    return all_passed


def test_model_instantiation():
    """Test that models can be instantiated."""
    print_header("2. MODEL INSTANTIATION")
    
    all_passed = True
    
    try:
        from sci_arc.config import SCIARCConfig
        config = SCIARCConfig()
        print_result("SCIARCConfig default", True, f"hidden_dim={config.hidden_dim}")
    except Exception as e:
        print_result("SCIARCConfig default", False, str(e))
        all_passed = False
        return all_passed
    
    try:
        from sci_arc.models.grid_encoder import GridEncoder
        encoder = GridEncoder(hidden_dim=config.hidden_dim)
        print_result("GridEncoder instantiation", True)
    except Exception as e:
        print_result("GridEncoder instantiation", False, str(e))
        all_passed = False
    
    try:
        from sci_arc.models.structural_encoder import StructuralEncoder2D
        se = StructuralEncoder2D(hidden_dim=config.hidden_dim)
        print_result("StructuralEncoder2D instantiation", True)
    except Exception as e:
        print_result("StructuralEncoder2D instantiation", False, str(e))
        all_passed = False
    
    try:
        from sci_arc.models.content_encoder import ContentEncoder2D
        ce = ContentEncoder2D(hidden_dim=config.hidden_dim)
        print_result("ContentEncoder2D instantiation", True)
    except Exception as e:
        print_result("ContentEncoder2D instantiation", False, str(e))
        all_passed = False
    
    try:
        from sci_arc.models.causal_binding import CausalBinding2D
        cb = CausalBinding2D(hidden_dim=config.hidden_dim)
        print_result("CausalBinding2D instantiation", True)
    except Exception as e:
        print_result("CausalBinding2D instantiation", False, str(e))
        all_passed = False
    
    try:
        from sci_arc.models.recursive_refinement import RecursiveRefinement
        rr = RecursiveRefinement(hidden_dim=config.hidden_dim)
        print_result("RecursiveRefinement instantiation", True)
    except Exception as e:
        print_result("RecursiveRefinement instantiation", False, str(e))
        all_passed = False
    
    try:
        from sci_arc.models.sci_arc import SCIARC
        model = SCIARC(config)
        print_result("SCIARC full model instantiation", True)
    except Exception as e:
        print_result("SCIARC full model instantiation", False, str(e))
        all_passed = False
    
    return all_passed


def test_forward_pass():
    """Test forward pass through the model."""
    print_header("3. FORWARD PASS VALIDATION")
    
    all_passed = True
    
    try:
        from sci_arc.config import SCIARCConfig
        from sci_arc.models.sci_arc import SCIARC
        
        config = SCIARCConfig(hidden_dim=64, H_cycles=2, L_cycles=2)  # Small for testing
        model = SCIARC(config)
        model.eval()
        
        # Create dummy data
        batch_size = 2
        demo_in = torch.randint(0, 10, (batch_size, 3, 5, 5))  # 3 demos, 5x5 grids
        demo_out = torch.randint(0, 10, (batch_size, 3, 5, 5))
        test_in = torch.randint(0, 10, (batch_size, 5, 5))
        target_shape = (5, 5)
        
        demo_pairs = list(zip(
            [demo_in[:, i] for i in range(3)],
            [demo_out[:, i] for i in range(3)]
        ))
        
        with torch.no_grad():
            output = model(demo_pairs, test_in, target_shape)
        
        print_result("Forward pass execution", True)
        print_result("Output type check", hasattr(output, 'final_prediction'), 
                    f"Type: {type(output).__name__}")
        print_result("Output shape", output.final_prediction.shape == (batch_size, 5, 5, 10),
                    f"Shape: {output.final_prediction.shape}")
        print_result("structure_rep exists", output.structure_rep is not None,
                    f"Shape: {output.structure_rep.shape if output.structure_rep is not None else 'N/A'}")
        print_result("content_rep exists", output.content_rep is not None,
                    f"Shape: {output.content_rep.shape if output.content_rep is not None else 'N/A'}")
        print_result("z_task exists", output.z_task is not None,
                    f"Shape: {output.z_task.shape if output.z_task is not None else 'N/A'}")
        
    except Exception as e:
        print_result("Forward pass", False, str(e))
        import traceback
        traceback.print_exc()
        all_passed = False
    
    return all_passed


def test_training_interface():
    """Test the training interface (forward_training)."""
    print_header("4. TRAINING INTERFACE")
    
    all_passed = True
    
    try:
        from sci_arc.config import SCIARCConfig
        from sci_arc.models.sci_arc import SCIARC
        
        config = SCIARCConfig(hidden_dim=64, H_cycles=2, L_cycles=2)
        model = SCIARC(config)
        model.train()
        
        # Create dummy data in batched format for forward_training
        batch_size = 2
        num_pairs = 3
        demo_in = torch.randint(0, 10, (batch_size, num_pairs, 5, 5))
        demo_out = torch.randint(0, 10, (batch_size, num_pairs, 5, 5))
        test_in = torch.randint(0, 10, (batch_size, 5, 5))
        target = torch.randint(0, 10, (batch_size, 5, 5))
        
        # Test forward_training with batched format
        outputs = model.forward_training(demo_in, demo_out, test_in, target)
        
        print_result("forward_training() exists", True)
        print_result("Returns dict", isinstance(outputs, dict),
                    f"Type: {type(outputs).__name__}")
        print_result("Has 'logits' key", 'logits' in outputs)
        print_result("Has 'z_struct' key", 'z_struct' in outputs)
        print_result("Has 'z_content' key", 'z_content' in outputs)
        print_result("Has 'z_task' key", 'z_task' in outputs)
        
        # Test backward pass
        from sci_arc.training.losses import SCIARCLoss
        loss_fn = SCIARCLoss(H_cycles=config.H_cycles)
        
        logits = outputs['logits']
        loss = nn.functional.cross_entropy(
            logits.view(-1, 10),
            target.view(-1)
        )
        loss.backward()
        
        print_result("Backward pass", True, f"Loss: {loss.item():.4f}")
        
        # Check gradients exist
        has_grads = any(p.grad is not None for p in model.parameters())
        print_result("Gradients computed", has_grads)
        
    except Exception as e:
        print_result("Training interface", False, str(e))
        import traceback
        traceback.print_exc()
        all_passed = False
    
    return all_passed


def test_loss_function():
    """Test the loss function."""
    print_header("5. LOSS FUNCTION")
    
    all_passed = True
    
    try:
        from sci_arc.config import SCIARCConfig
        from sci_arc.training.losses import SCIARCLoss
        
        config = SCIARCConfig(hidden_dim=64)
        loss_fn = SCIARCLoss(H_cycles=config.H_cycles)
        
        batch_size = 2
        h, w = 5, 5
        num_colors = 10
        hidden_dim = 64
        
        # Create dummy inputs
        logits = torch.randn(batch_size, h, w, num_colors)
        target = torch.randint(0, num_colors, (batch_size, h, w))
        z_struct = torch.randn(batch_size, 16, hidden_dim)
        z_content = torch.randn(batch_size, 16, hidden_dim)
        intermediate = [torch.randn(batch_size, h, w, num_colors) for _ in range(config.H_cycles)]
        transform_labels = torch.randint(0, 3, (batch_size,))  # 3 transform families
        
        print_result("SCIARCLoss instantiation", True)
        
        # Test SCL loss (via internal module)
        scl_loss = loss_fn.scl(z_struct, transform_labels)
        print_result("SCL loss", True, f"Value: {scl_loss.item():.4f}")
        
        # Test orthogonality loss (via internal module)
        orth_loss = loss_fn.orthogonality(z_struct, z_content)
        print_result("Orthogonality loss", True, f"Value: {orth_loss.item():.4f}")
        
        # Test deep supervision
        deep_loss = loss_fn.deep_supervision(intermediate, target)
        print_result("Deep supervision loss", True, f"Value: {deep_loss.item():.4f}")
        
        # Test combined loss
        losses = loss_fn(intermediate, target, z_struct, z_content, transform_labels)
        print_result("Combined loss", True, f"Total: {losses['total'].item():.4f}")
        
    except Exception as e:
        print_result("Loss function", False, str(e))
        import traceback
        traceback.print_exc()
        all_passed = False
    
    return all_passed


def test_parameter_count():
    """Count parameters and compare with TRM."""
    print_header("6. PARAMETER COUNT & TRM COMPARISON")
    
    all_passed = True
    
    try:
        from sci_arc.config import SCIARCConfig
        from sci_arc.models.sci_arc import SCIARC
        from sci_arc.utils.model_analysis import (
            count_parameters, count_parameters_by_component, compare_with_trm
        )
        
        # Create standard config
        config = SCIARCConfig()
        model = SCIARC(config)
        
        total_params = count_parameters(model)
        component_counts = count_parameters_by_component(model)
        comparison = compare_with_trm(model)
        
        print(f"\n  SCI-ARC Parameters: {comparison['sci_arc_params_formatted']}")
        print(f"  TRM Parameters:     7.00M (reference)")
        print(f"  Ratio:              {comparison['param_ratio']:.2f}x")
        print(f"\n  Parameters by Component:")
        print(f"  {'-'*40}")
        
        for name, count in component_counts.items():
            if name != '_total':
                pct = count / total_params * 100 if total_params > 0 else 0
                print(f"    {name}: {count:,} ({pct:.1f}%)")
        
        print_result("Parameter count computed", True)
        print_result("Competitive with TRM", comparison['is_competitive'],
                    f"Within 1.5x: {comparison['param_ratio']:.2f}x")
        
    except Exception as e:
        print_result("Parameter count", False, str(e))
        import traceback
        traceback.print_exc()
        all_passed = False
    
    return all_passed


def test_memory_efficiency():
    """Test memory-efficient training pattern."""
    print_header("7. MEMORY EFFICIENCY")
    
    all_passed = True
    
    try:
        from sci_arc.config import SCIARCConfig
        from sci_arc.models.recursive_refinement import RecursiveRefinement
        
        config = SCIARCConfig(hidden_dim=64, H_cycles=4, L_cycles=2)
        rr = RecursiveRefinement(
            hidden_dim=config.hidden_dim,
            H_cycles=config.H_cycles,
            L_cycles=config.L_cycles
        )
        rr.train()
        
        # Create inputs
        batch_size = 2
        x_test = torch.randn(batch_size, 25, 64)  # 5x5 flattened
        z_task = torch.randn(batch_size, 64)
        
        # Test with memory_efficient=True
        outputs_efficient, final_efficient = rr(x_test, z_task, (5, 5), memory_efficient=True)
        print_result("Memory-efficient forward", True, 
                    f"H_cycles-1 without grad = {config.H_cycles - 1}")
        
        # Test with memory_efficient=False
        outputs_full, final_full = rr(x_test, z_task, (5, 5), memory_efficient=False)
        print_result("Full-gradient forward", True,
                    f"All {config.H_cycles} with grad")
        
        # Both should have same number of outputs
        print_result("Output count match", 
                    len(outputs_efficient) == len(outputs_full) == config.H_cycles)
        
        # Final outputs should have same shape
        print_result("Output shape match",
                    final_efficient.shape == final_full.shape,
                    f"Shape: {final_efficient.shape}")
        
    except Exception as e:
        print_result("Memory efficiency", False, str(e))
        import traceback
        traceback.print_exc()
        all_passed = False
    
    return all_passed


def test_trm_baseline():
    """Test TRM baseline exists and runs."""
    print_header("8. TRM BASELINE VALIDATION")
    
    all_passed = True
    
    try:
        from baselines.trm import TRM
        
        print_result("TRM imports", True)
        
        # Use complete config dict required by TRM
        config_dict = {
            'batch_size': 2,
            'seq_len': 256,
            'num_puzzle_identifiers': 100,
            'vocab_size': 12,  # 10 colors + PAD + EOS
            'hidden_size': 64,
            'H_cycles': 2,
            'L_cycles': 2,
            'H_layers': 1,
            'L_layers': 1,
            'expansion': 2.5,
            'num_heads': 4,
            'pos_encodings': 'rope',
            'halt_max_steps': 16,
            'halt_exploration_prob': 0.1,
        }
        model = TRM(config_dict)
        
        print_result("TRM instantiation", True)
        
        # Count TRM parameters
        trm_params = sum(p.numel() for p in model.parameters())
        print_result("TRM parameters", True, f"{trm_params:,} ({trm_params/1e6:.2f}M)")
        
    except ImportError as e:
        print_result("TRM baseline", False, f"Import error: {e}")
        print("          (TRM baseline may not be implemented yet)")
        all_passed = False
    except Exception as e:
        print_result("TRM baseline", False, str(e))
        import traceback
        traceback.print_exc()
        all_passed = False
    
    return all_passed


def run_all_tests():
    """Run all validation tests."""
    print("\n" + "="*60)
    print(" SCI-ARC IMPLEMENTATION VALIDATION")
    print(" Comprehensive Bug-Free Check")
    print("="*60)
    
    results = {
        "Imports": test_imports(),
        "Instantiation": test_model_instantiation(),
        "Forward Pass": test_forward_pass(),
        "Training Interface": test_training_interface(),
        "Loss Function": test_loss_function(),
        "Parameter Count": test_parameter_count(),
        "Memory Efficiency": test_memory_efficiency(),
        "TRM Baseline": test_trm_baseline(),
    }
    
    # Summary
    print_header("VALIDATION SUMMARY")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓" if result else "✗"
        print(f"  [{status}] {test_name}")
    
    print(f"\n  Total: {passed}/{total} test suites passed")
    
    if passed == total:
        print("\n  ✓ ALL VALIDATIONS PASSED - SCI-ARC IS READY!")
    else:
        print(f"\n  ✗ {total - passed} test suite(s) failed - review above errors")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
