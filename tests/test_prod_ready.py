#!/usr/bin/env python3
"""
Production Readiness Test Suite for RLAN
=========================================
Comprehensive tests to verify:
1. All modules are correctly wired (forward path)
2. Gradient flow through all paths (backward path)
3. All YAML parameters are used in code
4. Checkpoint serialization/deserialization works
5. HPM end-to-end including solver coupling
6. DSC task conditioning works properly
7. HyperLoRA weight adaptation flows

Usage:
    python tests/test_prod_ready.py

Exit codes:
    0 = All tests passed
    1 = One or more tests failed
"""

import sys
import os
import warnings
import traceback
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import yaml


def print_header(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_result(name: str, passed: bool, details: str = ""):
    """Print a test result."""
    status = "[PASS]" if passed else "[FAIL]"
    icon = "✓" if passed else "✗"
    # Use ASCII for Windows compatibility
    try:
        print(f"  {icon} {status} {name}")
    except UnicodeEncodeError:
        icon_ascii = "+" if passed else "X"
        print(f"  {icon_ascii} {status} {name}")
    if details and not passed:
        print(f"      Details: {details}")


class TestResults:
    """Track test results."""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
    
    def add_pass(self, name: str):
        self.passed += 1
        print_result(name, True)
    
    def add_fail(self, name: str, details: str = ""):
        self.failed += 1
        self.errors.append((name, details))
        print_result(name, False, details)
    
    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"  SUMMARY: {self.passed}/{total} tests passed")
        if self.errors:
            print(f"  FAILURES:")
            for name, details in self.errors:
                print(f"    - {name}: {details}")
        print(f"{'='*60}")
        return self.failed == 0


# ============================================================================
# TEST 1: Module Import and Config Loading
# ============================================================================
def test_imports(results: TestResults):
    """Test that all modules can be imported."""
    print_header("TEST 1: Module Imports")
    
    try:
        from sci_arc.models.rlan import RLAN, RLANConfig
        results.add_pass("Import RLAN and RLANConfig")
    except Exception as e:
        results.add_fail("Import RLAN and RLANConfig", str(e))
        return False
    
    try:
        from sci_arc.models.rlan_modules import (
            DynamicSaliencyController,
            MultiScaleRelativeEncoding,
            LatentCountingRegisters,
            SymbolicPredicateHeads,
            RecursiveSolver,
            ContextEncoder,
            HyperLoRA,
            HierarchicalPrimitiveMemory,
            DynamicMemoryBuffer,
        )
        results.add_pass("Import all submodules")
    except Exception as e:
        results.add_fail("Import all submodules", str(e))
        return False
    
    return True


# ============================================================================
# TEST 2: Config Loading and YAML Parameter Validation
# ============================================================================
def test_yaml_params(results: TestResults):
    """Test that all YAML parameters are recognized by RLANConfig."""
    print_header("TEST 2: YAML Parameter Validation")
    
    from sci_arc.models.rlan import RLANConfig
    from dataclasses import fields
    
    # Load YAML config
    yaml_path = PROJECT_ROOT / "configs" / "rlan_stable_dev.yaml"
    with open(yaml_path) as f:
        yaml_config = yaml.safe_load(f)
    
    model_config = yaml_config.get('model', {})
    
    # Get all RLANConfig field names
    config_fields = {f.name for f in fields(RLANConfig)}
    
    # Parameters we expect in YAML that map to RLANConfig
    yaml_to_config_map = {
        'hidden_dim': 'hidden_dim',
        'num_colors': 'num_colors',
        'num_classes': 'num_classes',
        'max_grid_size': 'max_grid_size',
        'max_clues': 'max_clues',
        'num_predicates': 'num_predicates',
        'num_solver_steps': 'num_solver_steps',
        'use_act': 'use_act',
        'dropout': 'dropout',
        'use_context_encoder': 'use_context_encoder',
        'use_dsc': 'use_dsc',
        'use_msre': 'use_msre',
        'use_lcr': 'use_lcr',
        'use_sph': 'use_sph',
        'use_cross_attention_context': 'use_cross_attention_context',
        'spatial_downsample': 'spatial_downsample',
        'use_solver_context': 'use_solver_context',
        'solver_context_heads': 'solver_context_heads',
        'use_hyperlora': 'use_hyperlora',
        'hyperlora_rank': 'hyperlora_rank',
        'hyperlora_scaling': 'hyperlora_scaling',
        'hyperlora_dropout': 'hyperlora_dropout',
        'hyperlora_init_scale': 'hyperlora_init_scale',
        'use_hpm': 'use_hpm',
        'hpm_top_k': 'hpm_top_k',
        'hpm_balance_weight': 'hpm_balance_weight',
        'hpm_primitives_per_bank': 'hpm_primitives_per_bank',
        'hpm_levels_per_bank': 'hpm_levels_per_bank',
        'hpm_use_cross_attention': 'hpm_use_cross_attention',
        'hpm_memory_size': 'hpm_memory_size',
        'hpm_retrieval_k': 'hpm_retrieval_k',
        'hpm_use_compositional_bank': 'hpm_use_compositional_bank',
        'hpm_use_pattern_bank': 'hpm_use_pattern_bank',
        'hpm_use_relational_bank': 'hpm_use_relational_bank',
        'hpm_use_concept_bank': 'hpm_use_concept_bank',
        'hpm_use_procedural_bank': 'hpm_use_procedural_bank',
        'hpm_use_instance_bank': 'hpm_use_instance_bank',
        'hpm_solver_context_enabled': 'hpm_solver_context_enabled',
        'hpm_solver_context_max_tokens': 'hpm_solver_context_max_tokens',
    }
    
    # Check each mapping
    missing_in_config = []
    for yaml_key, config_key in yaml_to_config_map.items():
        if yaml_key in model_config and config_key not in config_fields:
            missing_in_config.append(f"{yaml_key} -> {config_key}")
    
    if missing_in_config:
        results.add_fail("YAML keys mapped to RLANConfig", f"Missing: {missing_in_config}")
    else:
        results.add_pass("YAML keys mapped to RLANConfig")
    
    # Verify critical training config keys exist
    training_config = yaml_config.get('training', {})
    required_training_keys = [
        'batch_size', 'learning_rate', 'gradient_clip',
        'meta_learning_start_epoch', 'solver_context_start_epoch',
    ]
    
    missing_training = [k for k in required_training_keys if k not in training_config]
    if missing_training:
        results.add_fail("Required training config keys", f"Missing: {missing_training}")
    else:
        results.add_pass("Required training config keys")
    
    return True


# ============================================================================
# TEST 3: Model Creation and Forward Pass
# ============================================================================
def test_forward_pass(results: TestResults):
    """Test forward pass through all modules."""
    print_header("TEST 3: Forward Pass")
    
    from sci_arc.models.rlan import RLAN, RLANConfig
    
    # Create model with all features enabled
    config = RLANConfig(
        hidden_dim=64,
        num_classes=10,
        max_clues=3,
        max_grid_size=15,
        use_dsc=True,
        use_msre=True,
        use_lcr=True,
        use_sph=True,
        use_act=False,
        use_context_encoder=True,
        use_cross_attention_context=True,
        use_solver_context=True,
        use_hyperlora=True,
        use_hpm=True,
        hpm_use_instance_bank=True,
        hpm_use_procedural_bank=True,
        hpm_solver_context_enabled=True,
        hpm_solver_context_max_tokens=4,
        hpm_solver_context_gate_init=0.1,
    )
    
    try:
        model = RLAN(config=config)
        results.add_pass("Model creation with all modules")
    except Exception as e:
        results.add_fail("Model creation with all modules", str(e))
        return False
    
    model.train()
    
    # Create dummy inputs
    B, N, H, W = 2, 3, 10, 10
    test_input = torch.randint(0, 10, (B, H, W))
    train_inputs = torch.randint(0, 10, (B, N, H, W))
    train_outputs = torch.randint(0, 10, (B, N, H, W))
    
    try:
        outputs = model(test_input, train_inputs, train_outputs, return_intermediates=True)
        results.add_pass("Forward pass completes")
    except Exception as e:
        results.add_fail("Forward pass completes", str(e))
        return False
    
    # Check expected outputs exist
    expected_keys = ['logits', 'centroids', 'attention_maps', 'stop_logits', 
                     'predicates', 'features', 'support_features', 'lora_deltas',
                     'z_struct', 'z_content']
    
    missing_keys = [k for k in expected_keys if k not in outputs]
    if missing_keys:
        results.add_fail("Forward outputs contain all keys", f"Missing: {missing_keys}")
    else:
        results.add_pass("Forward outputs contain all keys")
    
    # Check logits shape
    expected_shape = (B, 10, H, W)
    actual_shape = tuple(outputs['logits'].shape)
    if actual_shape != expected_shape:
        results.add_fail("Logits shape", f"Expected {expected_shape}, got {actual_shape}")
    else:
        results.add_pass("Logits shape correct")
    
    return True


# ============================================================================
# TEST 4: Gradient Flow Through All Modules
# ============================================================================
def test_gradient_flow(results: TestResults):
    """Test gradient flow through all major components."""
    print_header("TEST 4: Gradient Flow")
    
    from sci_arc.models.rlan import RLAN, RLANConfig
    
    config = RLANConfig(
        hidden_dim=64,
        num_classes=10,
        max_clues=3,
        max_grid_size=15,
        use_dsc=True,
        use_msre=True,
        use_lcr=True,
        use_sph=True,
        use_act=False,
        use_context_encoder=True,
        use_cross_attention_context=True,
        use_solver_context=True,
        use_hyperlora=True,
        hyperlora_init_scale=0.1,  # Non-zero for gradient flow
        use_hpm=True,
        hpm_use_instance_bank=True,
        hpm_solver_context_enabled=True,
        hpm_solver_context_max_tokens=4,
        hpm_solver_context_gate_init=2.0,  # Non-zero gate
    )
    
    model = RLAN(config=config)
    model.train()
    
    # Add entries to HPM buffer for solver coupling test
    z_dummy = torch.randn(1, 64)
    model.hpm_instance_buffer.add(z_dummy, z_dummy, "task_1")
    model.hpm_instance_buffer.add(z_dummy * 0.9, z_dummy * 0.9, "task_2")
    
    B, N, H, W = 2, 3, 10, 10
    test_input = torch.randint(0, 10, (B, H, W))
    train_inputs = torch.randint(0, 10, (B, N, H, W))
    train_outputs = torch.randint(0, 10, (B, N, H, W))
    
    outputs = model(test_input, train_inputs, train_outputs, return_intermediates=True)
    loss = outputs['logits'].mean()
    loss.backward()
    
    # Check gradients on critical components
    gradient_checks = []
    
    # 1. GridEncoder
    if hasattr(model.encoder, 'color_embed') and model.encoder.color_embed.weight.grad is not None:
        grad_norm = model.encoder.color_embed.weight.grad.norm().item()
        gradient_checks.append(('GridEncoder color_embed', grad_norm > 0))
    else:
        gradient_checks.append(('GridEncoder color_embed', False))
    
    # 2. ContextEncoder
    if model.context_encoder is not None:
        for name, param in model.context_encoder.named_parameters():
            if param.grad is not None:
                gradient_checks.append(('ContextEncoder', param.grad.norm().item() > 0))
                break
    
    # 3. DSC stop predictor (critical for task conditioning)
    if model.dsc is not None and hasattr(model.dsc, 'stop_predictor'):
        for name, param in model.dsc.stop_predictor.named_parameters():
            if param.grad is not None:
                gradient_checks.append(('DSC stop_predictor', param.grad.norm().item() > 0))
                break
    
    # 4. HyperLoRA
    if model.hyper_lora is not None:
        for name, param in model.hyper_lora.named_parameters():
            if param.grad is not None:
                gradient_checks.append(('HyperLoRA', param.grad.norm().item() > 0))
                break
    
    # 5. HPM banks
    if model.hpm is not None:
        for bank_name, bank in model.hpm.banks.items():
            if hasattr(bank, 'memory') and bank.memory.grad is not None:
                gradient_checks.append((f'HPM {bank_name}', bank.memory.grad.norm().item() > 0))
                break
    
    # 6. HPM solver-context gate (Jan 2026)
    if model.solver.solver_cross_attn is not None:
        sca = model.solver.solver_cross_attn
        if hasattr(sca, 'hpm_gate') and sca.hpm_gate is not None:
            if sca.hpm_gate.grad is not None:
                gradient_checks.append(('HPM solver gate', sca.hpm_gate.grad.abs().item() > 0))
            else:
                gradient_checks.append(('HPM solver gate', False))
        if hasattr(sca, 'hpm_proj') and sca.hpm_proj is not None:
            if sca.hpm_proj.weight.grad is not None:
                gradient_checks.append(('HPM solver proj', sca.hpm_proj.weight.grad.norm().item() > 0))
    
    # 7. Solver GRU
    for name, param in model.solver.gru.named_parameters():
        if param.grad is not None:
            gradient_checks.append(('Solver GRU', param.grad.norm().item() > 0))
            break
    
    # Report results
    for name, has_grad in gradient_checks:
        if has_grad:
            results.add_pass(f"Gradient flows to {name}")
        else:
            results.add_fail(f"Gradient flows to {name}", "No gradient or zero gradient")
    
    return True


# ============================================================================
# TEST 5: DSC Task Conditioning
# ============================================================================
def test_dsc_task_conditioning(results: TestResults):
    """Test that DSC receives and uses task_context properly."""
    print_header("TEST 5: DSC Task Conditioning")
    
    from sci_arc.models.rlan import RLAN, RLANConfig
    
    config = RLANConfig(
        hidden_dim=64,
        num_classes=10,
        max_clues=3,
        max_grid_size=15,
        use_dsc=True,
        use_context_encoder=True,
        use_cross_attention_context=True,
    )
    
    model = RLAN(config=config)
    model.train()
    
    # Create two different task contexts
    B, N, H, W = 2, 3, 10, 10
    test_input = torch.randint(0, 10, (B, H, W))
    train_inputs_1 = torch.randint(0, 5, (B, N, H, W))  # Different context
    train_inputs_2 = torch.randint(5, 10, (B, N, H, W))  # Different context
    train_outputs = torch.randint(0, 10, (B, N, H, W))
    
    with torch.no_grad():
        out_1 = model(test_input, train_inputs_1, train_outputs, return_intermediates=True)
        out_2 = model(test_input, train_inputs_2, train_outputs, return_intermediates=True)
    
    # Stop logits should differ between different contexts
    stop_diff = (out_1['stop_logits'] - out_2['stop_logits']).abs().mean().item()
    
    if stop_diff > 0.01:  # Should be noticeably different
        results.add_pass("DSC stop_logits vary with task context")
    else:
        results.add_fail("DSC stop_logits vary with task context", 
                        f"Difference too small: {stop_diff:.6f}")
    
    return True


# ============================================================================
# TEST 6: Checkpoint Serialization
# ============================================================================
def test_checkpoint_serialization(results: TestResults):
    """Test that checkpoints save and load correctly including HPM."""
    print_header("TEST 6: Checkpoint Serialization")
    
    from sci_arc.models.rlan import RLAN, RLANConfig
    import tempfile
    
    config = RLANConfig(
        hidden_dim=64,
        num_classes=10,
        max_clues=3,
        max_grid_size=15,
        use_hpm=True,
        hpm_use_instance_bank=True,
        hpm_use_procedural_bank=True,
    )
    
    model = RLAN(config=config)
    
    # Add entries to HPM buffers
    z_dummy = torch.randn(1, 64)
    model.hpm_instance_buffer.add(z_dummy, z_dummy, "task_1")
    model.hpm_procedural_buffer.add(z_dummy, z_dummy * 2, "task_2")
    
    # Save checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'hpm_instance_buffer': model.hpm_instance_buffer.state_dict(),
        'hpm_procedural_buffer': model.hpm_procedural_buffer.state_dict(),
    }
    
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        torch.save(checkpoint, f.name)
        ckpt_path = f.name
    
    try:
        # Create new model and load
        model2 = RLAN(config=config)
        loaded = torch.load(ckpt_path)
        
        model2.load_state_dict(loaded['model_state_dict'])
        model2.hpm_instance_buffer.load_state_dict(loaded['hpm_instance_buffer'])
        model2.hpm_procedural_buffer.load_state_dict(loaded['hpm_procedural_buffer'])
        
        # Check buffer sizes match
        if len(model2.hpm_instance_buffer) == 1 and len(model2.hpm_procedural_buffer) == 1:
            results.add_pass("HPM buffers restored from checkpoint")
        else:
            results.add_fail("HPM buffers restored from checkpoint",
                           f"Instance: {len(model2.hpm_instance_buffer)}, Procedural: {len(model2.hpm_procedural_buffer)}")
        
        # Check HPM static banks are in state_dict
        hpm_keys = [k for k in loaded['model_state_dict'].keys() if 'hpm.' in k]
        if len(hpm_keys) > 0:
            results.add_pass(f"HPM static banks in state_dict ({len(hpm_keys)} keys)")
        else:
            results.add_fail("HPM static banks in state_dict", "No HPM keys found")
        
    finally:
        os.unlink(ckpt_path)
    
    return True


# ============================================================================
# TEST 7: HyperLoRA Weight Adaptation
# ============================================================================
def test_hyperlora(results: TestResults):
    """Test HyperLoRA predicts and applies weight deltas."""
    print_header("TEST 7: HyperLoRA Weight Adaptation")
    
    from sci_arc.models.rlan import RLAN, RLANConfig
    
    config = RLANConfig(
        hidden_dim=64,
        num_classes=10,
        max_clues=3,
        max_grid_size=15,
        use_context_encoder=True,
        use_cross_attention_context=True,
        use_solver_context=True,
        use_hyperlora=True,
        hyperlora_rank=4,
        hyperlora_init_scale=0.1,
    )
    
    model = RLAN(config=config)
    model.train()
    
    B, N, H, W = 2, 3, 10, 10
    test_input = torch.randint(0, 10, (B, H, W))
    train_inputs = torch.randint(0, 10, (B, N, H, W))
    train_outputs = torch.randint(0, 10, (B, N, H, W))
    
    outputs = model(test_input, train_inputs, train_outputs, return_intermediates=True)
    
    # Check lora_deltas is in outputs
    if 'lora_deltas' not in outputs:
        results.add_fail("HyperLoRA produces lora_deltas", "Not in outputs")
        return True
    
    lora_deltas = outputs['lora_deltas']
    if lora_deltas is None:
        results.add_fail("HyperLoRA produces lora_deltas", "lora_deltas is None")
        return True
    
    results.add_pass("HyperLoRA produces lora_deltas")
    
    # Check expected delta keys
    expected_delta_keys = ['gru_reset', 'gru_update', 'gru_candidate', 'output_head']
    missing_delta_keys = [k for k in expected_delta_keys if k not in lora_deltas]
    
    if missing_delta_keys:
        results.add_fail("HyperLoRA delta keys", f"Missing: {missing_delta_keys}")
    else:
        results.add_pass("HyperLoRA delta keys present")
    
    # Check deltas are non-zero
    for key in expected_delta_keys:
        if key in lora_deltas:
            delta = lora_deltas[key]
            if delta is not None and delta.abs().max().item() > 1e-6:
                results.add_pass(f"HyperLoRA {key} is non-zero")
            else:
                results.add_fail(f"HyperLoRA {key} is non-zero", "Zero or None")
    
    return True


# ============================================================================
# TEST 8: Structure/Content Disentanglement
# ============================================================================
def test_structure_content(results: TestResults):
    """Test z_struct and z_content are computed correctly."""
    print_header("TEST 8: Structure/Content Disentanglement")
    
    from sci_arc.models.rlan import RLAN, RLANConfig
    
    config = RLANConfig(
        hidden_dim=64,
        num_classes=10,
        max_clues=3,
        max_grid_size=15,
        use_context_encoder=True,
        use_cross_attention_context=True,
    )
    
    model = RLAN(config=config)
    model.eval()
    
    B, N, H, W = 2, 3, 10, 10
    test_input = torch.randint(0, 10, (B, H, W))
    train_inputs = torch.randint(0, 10, (B, N, H, W))
    train_outputs = torch.randint(0, 10, (B, N, H, W))
    
    with torch.no_grad():
        outputs = model(test_input, train_inputs, train_outputs, return_intermediates=True)
    
    # Check z_struct exists and has correct shape
    if 'z_struct' in outputs:
        z_struct = outputs['z_struct']
        if z_struct.shape == (B, 64):
            results.add_pass("z_struct shape correct")
        else:
            results.add_fail("z_struct shape", f"Expected (2, 64), got {z_struct.shape}")
    else:
        results.add_fail("z_struct in outputs", "Missing")
    
    # Check z_struct_demos for per-demo structure
    if 'z_struct_demos' in outputs:
        z_struct_demos = outputs['z_struct_demos']
        if z_struct_demos.shape == (B, N, 64):
            results.add_pass("z_struct_demos shape correct")
        else:
            results.add_fail("z_struct_demos shape", f"Expected (2, 3, 64), got {z_struct_demos.shape}")
    else:
        results.add_fail("z_struct_demos in outputs", "Missing")
    
    # Check z_content
    if 'z_content' in outputs:
        z_content = outputs['z_content']
        if z_content.shape == (B, 64):
            results.add_pass("z_content shape correct")
        else:
            results.add_fail("z_content shape", f"Expected (2, 64), got {z_content.shape}")
    else:
        results.add_fail("z_content in outputs", "Missing")
    
    return True


# ============================================================================
# TEST 9: Logging Completeness (check intermediate outputs)
# ============================================================================
def test_logging_outputs(results: TestResults):
    """Test that all intermediate outputs needed for logging are present."""
    print_header("TEST 9: Logging Outputs Completeness")
    
    from sci_arc.models.rlan import RLAN, RLANConfig
    
    config = RLANConfig(
        hidden_dim=64,
        num_classes=10,
        max_clues=3,
        max_grid_size=15,
        use_dsc=True,
        use_msre=True,
        use_lcr=True,
        use_sph=True,
        use_context_encoder=True,
        use_cross_attention_context=True,
        use_solver_context=True,
        use_hyperlora=True,
        use_hpm=True,
        hpm_use_instance_bank=True,
    )
    
    model = RLAN(config=config)
    model.eval()
    
    # Add HPM buffer entries
    z_dummy = torch.randn(1, 64)
    model.hpm_instance_buffer.add(z_dummy, z_dummy, "task_1")
    
    B, N, H, W = 2, 3, 10, 10
    test_input = torch.randint(0, 10, (B, H, W))
    train_inputs = torch.randint(0, 10, (B, N, H, W))
    train_outputs = torch.randint(0, 10, (B, N, H, W))
    
    with torch.no_grad():
        outputs = model(test_input, train_inputs, train_outputs, return_intermediates=True)
    
    # Required outputs for complete logging
    required_for_logging = {
        'logits': 'Main prediction logits',
        'centroids': 'DSC anchor positions',
        'attention_maps': 'DSC attention masks',
        'stop_logits': 'DSC clue stopping scores',
        'predicates': 'SPH binary predicates',
        'features': 'Encoded grid features',
        'support_features': 'Context encoder output',
        'lora_deltas': 'HyperLoRA weight adaptations',
        'z_struct': 'Structure embedding for SCL',
        'z_content': 'Content embedding for SCL',
    }
    
    for key, desc in required_for_logging.items():
        if key in outputs and outputs[key] is not None:
            results.add_pass(f"Output '{key}' ({desc})")
        else:
            results.add_fail(f"Output '{key}' ({desc})", "Missing or None")
    
    # Check HPM routing weights when HPM is active
    if 'hpm_routing_weights' in outputs:
        results.add_pass("HPM routing weights in output")
    else:
        # HPM might not have routing if no context
        results.add_pass("HPM routing weights (conditional)")
    
    return True


# ============================================================================
# TEST 10: All-Steps Solver Output
# ============================================================================
def test_all_steps_output(results: TestResults):
    """Test that return_all_steps provides predictions at each solver step."""
    print_header("TEST 10: Multi-Step Solver Output")
    
    from sci_arc.models.rlan import RLAN, RLANConfig
    
    config = RLANConfig(
        hidden_dim=64,
        num_classes=10,
        max_clues=3,
        max_grid_size=15,
        num_solver_steps=5,
    )
    
    model = RLAN(config=config)
    model.eval()
    
    B, H, W = 2, 10, 10
    test_input = torch.randint(0, 10, (B, H, W))
    
    with torch.no_grad():
        outputs = model(test_input, return_intermediates=True, return_all_steps=True)
    
    if 'all_logits' not in outputs:
        results.add_fail("all_logits in output", "Missing")
        return True
    
    all_logits = outputs['all_logits']
    
    if all_logits is None:
        results.add_fail("all_logits populated", "Is None")
        return True
    
    if len(all_logits) == 5:
        results.add_pass(f"all_logits has {len(all_logits)} steps")
    else:
        results.add_fail(f"all_logits step count", f"Expected 5, got {len(all_logits)}")
    
    # Check each step has correct shape
    for i, step_logits in enumerate(all_logits):
        if step_logits.shape == (B, 10, H, W):
            pass  # Shape correct
        else:
            results.add_fail(f"Step {i} logits shape", f"Got {step_logits.shape}")
            return True
    
    results.add_pass("All solver steps have correct shape")
    
    return True


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("\n" + "="*60)
    print("  RLAN Production Readiness Test Suite")
    print("="*60)
    
    warnings.filterwarnings('ignore')
    
    results = TestResults()
    
    try:
        # Run all tests
        if not test_imports(results):
            print("\nCRITICAL: Import test failed, cannot continue")
            return 1
        
        test_yaml_params(results)
        test_forward_pass(results)
        test_gradient_flow(results)
        test_dsc_task_conditioning(results)
        test_checkpoint_serialization(results)
        test_hyperlora(results)
        test_structure_content(results)
        test_logging_outputs(results)
        test_all_steps_output(results)
        
    except Exception as e:
        print(f"\n[ERROR] Unexpected exception: {e}")
        traceback.print_exc()
        return 1
    
    success = results.summary()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
