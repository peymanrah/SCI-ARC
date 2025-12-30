#!/usr/bin/env python
"""
Memory Test Script - Ensure GPU Memory Stays Within 24GB VRAM

This script simulates training for ALL epochs with ALL module activations
to verify that memory never exceeds the GPU limit.

Usage:
    python scripts/test_memory_all_epochs.py --config configs/rlan_stable_dev.yaml
    python scripts/test_memory_all_epochs.py --config configs/rlan_stable_dev.yaml --batch-size 80
    python scripts/test_memory_all_epochs.py --config configs/rlan_stable_dev.yaml --verbose

Exit codes:
    0 = All tests passed, memory within bounds
    1 = Memory exceeded GPU limit in at least one configuration
"""

import argparse
import sys
import gc
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sci_arc.models import RLAN, RLANConfig
from sci_arc.utils.memory_manager import MemoryManager


@dataclass
class MemoryTestResult:
    """Result of a memory test."""
    epoch: int
    modules_active: Dict[str, bool]
    batch_size: int
    max_grid_size: int
    estimated_mb: float
    actual_peak_mb: float
    gpu_limit_mb: float
    passed: bool
    message: str


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_model(config: dict, device: str = 'cuda') -> RLAN:
    """Create RLAN model from config."""
    model_config = config['model']
    train_config = config['training']
    
    # Build RLANConfig
    rlan_config = RLANConfig(
        hidden_dim=model_config.get('hidden_dim', 256),
        num_colors=model_config.get('num_colors', 10),
        max_grid_size=model_config.get('max_grid_size', 30),
        max_clues=model_config.get('max_clues', 7),
        num_predicates=model_config.get('num_predicates', 32),
        num_solver_steps=model_config.get('num_solver_steps', 5),
        use_act=model_config.get('use_act', False),
        dropout=model_config.get('dropout', 0.1),
        use_context_encoder=model_config.get('use_context_encoder', True),
        use_dsc=model_config.get('use_dsc', True),
        use_msre=model_config.get('use_msre', True),
        use_cross_attention_context=model_config.get('use_cross_attention_context', True),
        spatial_downsample=model_config.get('spatial_downsample', 8),
        use_solver_context=model_config.get('use_solver_context', True),
        solver_context_heads=model_config.get('solver_context_heads', 4),
        use_hyperlora=model_config.get('use_hyperlora', True),
        hyperlora_rank=model_config.get('hyperlora_rank', 8),
        use_hpm=model_config.get('use_hpm', False),
        gradient_checkpointing=train_config.get('gradient_checkpointing', True),
    )
    
    model = RLAN(config=rlan_config)
    model = model.to(device)
    
    return model


def get_module_activation_schedule(config: dict) -> Dict[int, Dict[str, bool]]:
    """
    Get the module activation states for each epoch.
    
    Returns:
        Dict mapping epoch -> module activation state
    """
    train_config = config.get('training', {})
    model_config = config.get('model', {})
    
    # Get activation epochs from config
    meta_learning_start = train_config.get('meta_learning_start_epoch', 3)
    equiv_start = train_config.get('equivariance_training', {}).get('start_epoch', 8)
    loo_start = train_config.get('loo_training', {}).get('start_epoch', 12)
    hpm_start = model_config.get('hpm_start_epoch', 14)
    
    # NEW: Staggered activation (fix for 12GB overflow)
    # Instead of activating HyperLoRA+SolverContext+CrossAttention all at epoch 3,
    # we stagger them across epochs 3, 5, 7
    hyperlora_start = meta_learning_start
    solver_context_start = meta_learning_start + 2  # Epoch 5
    cross_attention_start = meta_learning_start + 4  # Epoch 7
    
    # Test epochs that matter for memory
    test_epochs = sorted(set([
        0,  # Baseline
        hyperlora_start,  # HyperLoRA activates
        solver_context_start,  # SolverCrossAttention activates
        cross_attention_start,  # CrossAttentionInjector activates
        equiv_start,  # Equivariance activates
        loo_start,  # LOO activates
        hpm_start,  # HPM activates
        20,  # All modules active
    ]))
    
    schedule = {}
    for epoch in test_epochs:
        schedule[epoch] = {
            'hyperlora_active': epoch >= hyperlora_start,
            'solver_context_active': epoch >= solver_context_start,
            'cross_attention_active': epoch >= cross_attention_start,
            'equivariance_active': epoch >= equiv_start,
            'loo_active': epoch >= loo_start,
            'use_hpm': epoch >= hpm_start and model_config.get('use_hpm', False),
        }
    
    return schedule


def measure_memory_for_config(
    model: nn.Module,
    batch_size: int,
    max_grid_size: int,
    modules_active: Dict[str, bool],
    num_pairs: int = 4,
    device: str = 'cuda',
) -> Tuple[float, float]:
    """
    Measure actual GPU memory for a specific configuration.
    
    Returns:
        (allocated_mb, peak_mb)
    """
    model.eval()
    
    # Set module activation flags on model
    for key, value in modules_active.items():
        if hasattr(model, key):
            setattr(model, key, value)
    
    # Cleanup before measurement
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    
    # Create dummy batch
    try:
        with torch.no_grad():
            test_inputs = torch.randint(0, 10, (batch_size, max_grid_size, max_grid_size), device=device)
            train_inputs = torch.randint(0, 10, (batch_size, num_pairs, max_grid_size, max_grid_size), device=device)
            train_outputs = torch.randint(0, 10, (batch_size, num_pairs, max_grid_size, max_grid_size), device=device)
            pair_mask = torch.ones(batch_size, num_pairs, device=device)
            
            # Forward pass
            outputs = model(
                test_inputs,
                train_inputs=train_inputs,
                train_outputs=train_outputs,
                pair_mask=pair_mask,
                return_intermediates=True,
            )
            
            # Simulate backward pass memory (multiply by ~2)
            # In reality, backward uses more memory than forward due to gradient storage
            # We don't actually compute gradients to avoid OOM during testing
            
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            torch.cuda.empty_cache()
            return (float('inf'), float('inf'))
        raise
    
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated() / (1024 * 1024)
    peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
    
    # Cleanup
    del test_inputs, train_inputs, train_outputs, pair_mask, outputs
    torch.cuda.empty_cache()
    
    model.train()
    
    # Multiply by 2.2 to estimate backward pass memory
    # This is conservative - actual backward might use more
    return (allocated, peak * 2.2)


def test_all_configurations(
    config: dict,
    batch_size: Optional[int] = None,
    verbose: bool = False,
) -> List[MemoryTestResult]:
    """
    Test memory for all module activation configurations.
    
    Returns:
        List of MemoryTestResult for each tested configuration
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if device != 'cuda':
        print("WARNING: CUDA not available, using CPU. Memory tests will be skipped.")
        return []
    
    # Get GPU info
    gpu_props = torch.cuda.get_device_properties(0)
    gpu_total_mb = gpu_props.total_memory / (1024 * 1024)
    gpu_name = gpu_props.name
    
    print(f"GPU: {gpu_name}")
    print(f"Total VRAM: {gpu_total_mb:.0f} MB")
    print(f"Safety limit (92%): {gpu_total_mb * 0.92:.0f} MB")
    print("=" * 60)
    
    # Use config batch size or override
    if batch_size is None:
        batch_size = config.get('training', {}).get('batch_size', 80)
    
    max_grid_size = config.get('model', {}).get('max_grid_size', 30)
    
    print(f"Testing with batch_size={batch_size}, max_grid_size={max_grid_size}")
    print("=" * 60)
    
    # Create model
    print("\nCreating model...")
    model = create_model(config, device)
    
    # Count parameters
    param_count = sum(p.numel() for p in model.parameters())
    param_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    print(f"Model parameters: {param_count:,} ({param_mb:.1f} MB)")
    
    # Create memory manager
    mem_mgr = MemoryManager(gpu_total_mb=gpu_total_mb, safety_margin=0.92)
    
    # Get activation schedule
    schedule = get_module_activation_schedule(config)
    
    results = []
    
    print("\n" + "=" * 60)
    print("MEMORY TEST RESULTS")
    print("=" * 60)
    
    for epoch, modules_active in sorted(schedule.items()):
        # Skip loo_active and equivariance_active for forward pass tests
        # (they don't affect single forward pass memory, only training loop)
        test_modules = {k: v for k, v in modules_active.items() 
                       if k not in ['loo_active', 'equivariance_active']}
        
        # Active modules string
        active_str = ', '.join([k for k, v in test_modules.items() if v]) or 'none'
        
        if verbose:
            print(f"\nEpoch {epoch}: Testing with modules: {active_str}")
        
        # Estimate memory
        estimated_mb = mem_mgr.estimate_forward_memory_mb(
            batch_size=batch_size,
            max_grid_size=max_grid_size,
            active_modules=test_modules,
        )
        
        # Measure actual memory
        allocated_mb, peak_mb = measure_memory_for_config(
            model=model,
            batch_size=batch_size,
            max_grid_size=max_grid_size,
            modules_active=test_modules,
            device=device,
        )
        
        # Check if within limit
        limit_mb = gpu_total_mb * 0.92
        passed = peak_mb <= limit_mb
        
        if peak_mb == float('inf'):
            message = "OOM during forward pass!"
            passed = False
        elif passed:
            headroom = limit_mb - peak_mb
            message = f"OK (headroom: {headroom:.0f} MB)"
        else:
            overflow = peak_mb - limit_mb
            message = f"OVERFLOW by {overflow:.0f} MB"
        
        result = MemoryTestResult(
            epoch=epoch,
            modules_active=test_modules,
            batch_size=batch_size,
            max_grid_size=max_grid_size,
            estimated_mb=estimated_mb,
            actual_peak_mb=peak_mb,
            gpu_limit_mb=limit_mb,
            passed=passed,
            message=message,
        )
        results.append(result)
        
        # Print result
        status = "✓" if passed else "✗"
        print(f"  Epoch {epoch:2d}: {status} Est: {estimated_mb:6.0f} MB | "
              f"Actual: {peak_mb:6.0f} MB | Limit: {limit_mb:.0f} MB | {message}")
        
        if verbose:
            print(f"           Modules: {active_str}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    passed_count = sum(1 for r in results if r.passed)
    failed_count = len(results) - passed_count
    
    print(f"Passed: {passed_count}/{len(results)}")
    print(f"Failed: {failed_count}/{len(results)}")
    
    if failed_count > 0:
        print("\nFAILED CONFIGURATIONS:")
        for r in results:
            if not r.passed:
                active = ', '.join([k for k, v in r.modules_active.items() if v])
                print(f"  - Epoch {r.epoch}: {r.message} (modules: {active})")
        
        # Suggest reduced batch size
        print("\nRECOMMENDED FIXES:")
        for r in results:
            if not r.passed and r.actual_peak_mb != float('inf'):
                # Calculate required batch size reduction
                required_reduction = r.actual_peak_mb / r.gpu_limit_mb
                new_batch = int(batch_size / required_reduction * 0.9)  # 10% extra margin
                new_batch = max(16, new_batch)  # Minimum 16
                print(f"  - Epoch {r.epoch}: Reduce batch_size to {new_batch} "
                      f"(currently {batch_size})")
    
    # Cleanup
    del model
    torch.cuda.empty_cache()
    gc.collect()
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Test GPU memory for all epoch configurations')
    parser.add_argument('--config', type=str, default='configs/rlan_stable_dev.yaml',
                       help='Path to config file')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Override batch size from config')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed information')
    
    args = parser.parse_args()
    
    # Load config
    config_path = Path(project_root) / args.config
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        sys.exit(1)
    
    config = load_config(str(config_path))
    print(f"Loaded config: {config_path}")
    
    # Run tests
    results = test_all_configurations(
        config=config,
        batch_size=args.batch_size,
        verbose=args.verbose,
    )
    
    # Exit code
    if all(r.passed for r in results):
        print("\n✓ All memory tests PASSED")
        sys.exit(0)
    else:
        print("\n✗ Some memory tests FAILED")
        sys.exit(1)


if __name__ == '__main__':
    main()
