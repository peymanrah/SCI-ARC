"""
Inference-Time Staging Helper for RLAN Meta-Learning Modules.

This module reads from YAML config and sets appropriate staging flags
for evaluation/inference, with graceful handling for:
- Empty HPM buffers (no solved tasks yet)
- Untrained HyperLoRA (weights at init values)
- Missing module states in checkpoint

DESIGN PRINCIPLES:
1. YAML is the enforcing force - all defaults read from config
2. Graceful degradation - if a module can't be activated, fall back
3. Clear logging - always log what's active and why
4. Consistent paths - same logic for train_rlan.py eval and evaluate_rlan.py

Author: SCI-ARC Team
Date: January 2026
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import warnings
import time


def get_inference_meta_learning_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract inference-time meta-learning settings from config.
    
    Args:
        config: Full YAML config dict
        
    Returns:
        Dict with meta_learning settings, with defaults for missing values
    """
    inference_config = config.get('inference', {})
    meta_config = inference_config.get('meta_learning', {})
    
    # Build config with defaults
    result = {
        'hyperlora': {
            'enable': meta_config.get('hyperlora', {}).get('enable', True),
            'require_trained': meta_config.get('hyperlora', {}).get('require_trained', True),
            'fallback_on_failure': meta_config.get('hyperlora', {}).get('fallback_on_failure', True),
        },
        'hpm': {
            'enable': meta_config.get('hpm', {}).get('enable', True),
            'require_nonempty_buffers': meta_config.get('hpm', {}).get('require_nonempty_buffers', True),
            'min_buffer_entries': meta_config.get('hpm', {}).get('min_buffer_entries', 1),
            'use_static_banks': meta_config.get('hpm', {}).get('use_static_banks', True),
            'use_dynamic_banks': meta_config.get('hpm', {}).get('use_dynamic_banks', True),
        },
        'solver_context': {
            'enable': meta_config.get('solver_context', {}).get('enable', True),
        },
        'cross_attention': {
            'enable': meta_config.get('cross_attention', {}).get('enable', True),
        },
    }
    
    return result


def check_hyperlora_validity(model: nn.Module) -> Tuple[bool, str]:
    """
    Check if HyperLoRA has been trained (weights are non-trivial).
    
    Args:
        model: RLAN model
        
    Returns:
        (is_valid, reason_string)
    """
    if not hasattr(model, 'hyper_lora') or model.hyper_lora is None:
        return False, "HyperLoRA module not present"
    
    # Check if any HyperLoRA parameter has non-trivial values
    total_norm = 0.0
    param_count = 0
    for name, param in model.hyper_lora.named_parameters():
        total_norm += param.norm().item()
        param_count += 1
    
    if param_count == 0:
        return False, "HyperLoRA has no parameters"
    
    avg_norm = total_norm / param_count
    
    # HyperLoRA with near-zero init would have very small norms
    # Trained HyperLoRA should have learned meaningful weights
    if avg_norm < 0.001:
        return False, f"HyperLoRA weights near-zero (avg_norm={avg_norm:.6f})"
    
    return True, f"HyperLoRA valid (avg_norm={avg_norm:.4f})"


def check_hpm_buffer_validity(model: nn.Module, min_entries: int = 1) -> Tuple[bool, str, Dict[str, int]]:
    """
    Check if HPM dynamic buffers have valid entries.
    
    Args:
        model: RLAN model
        min_entries: Minimum entries required in any buffer
        
    Returns:
        (is_valid, reason_string, buffer_sizes_dict)
    """
    buffer_sizes = {
        'instance': 0,
        'procedural': 0,
    }
    
    if not hasattr(model, 'use_hpm') or not model.use_hpm:
        return False, "HPM not enabled in model", buffer_sizes
    
    if hasattr(model, 'hpm_instance_buffer') and model.hpm_instance_buffer is not None:
        buffer_sizes['instance'] = len(model.hpm_instance_buffer)
    
    if hasattr(model, 'hpm_procedural_buffer') and model.hpm_procedural_buffer is not None:
        buffer_sizes['procedural'] = len(model.hpm_procedural_buffer)
    
    total_entries = buffer_sizes['instance'] + buffer_sizes['procedural']
    
    if total_entries < min_entries:
        return False, f"HPM buffers have {total_entries} entries (need {min_entries})", buffer_sizes
    
    return True, f"HPM buffers valid (instance={buffer_sizes['instance']}, procedural={buffer_sizes['procedural']})", buffer_sizes


def apply_inference_staging(
    model: nn.Module, 
    config: Dict[str, Any],
    verbose: bool = True,
) -> Dict[str, bool]:
    """
    Apply inference-time staging flags to model based on YAML config.
    
    This is the main entry point for setting up a model for evaluation/inference.
    It reads from YAML config and handles graceful fallbacks.
    
    Args:
        model: RLAN model to configure
        config: Full YAML config dict
        verbose: Whether to print status messages
        
    Returns:
        Dict mapping module names to their active status
    """
    meta_config = get_inference_meta_learning_config(config)
    active_modules = {}
    
    if verbose:
        print("\n" + "="*60)
        print("INFERENCE-TIME META-LEARNING CONFIGURATION")
        print("="*60)
    
    # ===== HyperLoRA =====
    hyperlora_cfg = meta_config['hyperlora']
    if hyperlora_cfg['enable']:
        if hyperlora_cfg['require_trained']:
            is_valid, reason = check_hyperlora_validity(model)
            if is_valid:
                model.hyperlora_active = True
                active_modules['hyperlora'] = True
                if verbose:
                    print(f"[✓] HyperLoRA: ACTIVE ({reason})")
            else:
                model.hyperlora_active = hyperlora_cfg['fallback_on_failure']
                active_modules['hyperlora'] = hyperlora_cfg['fallback_on_failure']
                if verbose:
                    status = "FALLBACK" if hyperlora_cfg['fallback_on_failure'] else "DISABLED"
                    print(f"[!] HyperLoRA: {status} ({reason})")
        else:
            model.hyperlora_active = True
            active_modules['hyperlora'] = True
            if verbose:
                print(f"[✓] HyperLoRA: ACTIVE (require_trained=False)")
    else:
        model.hyperlora_active = False
        active_modules['hyperlora'] = False
        if verbose:
            print(f"[✗] HyperLoRA: DISABLED (config: enable=False)")
    
    # ===== HPM =====
    hpm_cfg = meta_config['hpm']
    if hpm_cfg['enable']:
        if hpm_cfg['require_nonempty_buffers'] and hpm_cfg['use_dynamic_banks']:
            is_valid, reason, sizes = check_hpm_buffer_validity(model, hpm_cfg['min_buffer_entries'])
            if is_valid:
                model.use_hpm = True
                model.hpm_memory_enabled = True
                active_modules['hpm'] = True
                if verbose:
                    print(f"[✓] HPM: ACTIVE ({reason})")
            else:
                # HPM can still work with static banks only
                if hpm_cfg['use_static_banks'] and hasattr(model, 'hpm') and model.hpm is not None:
                    model.use_hpm = True
                    model.hpm_memory_enabled = False  # No dynamic retrieval
                    active_modules['hpm'] = True
                    if verbose:
                        print(f"[!] HPM: STATIC-ONLY ({reason}, using learned banks)")
                else:
                    model.use_hpm = False
                    model.hpm_memory_enabled = False
                    active_modules['hpm'] = False
                    if verbose:
                        print(f"[✗] HPM: DISABLED ({reason})")
        else:
            model.use_hpm = True
            model.hpm_memory_enabled = True
            active_modules['hpm'] = True
            if verbose:
                print(f"[✓] HPM: ACTIVE (require_nonempty_buffers=False)")
    else:
        model.use_hpm = False
        model.hpm_memory_enabled = False
        active_modules['hpm'] = False
        if verbose:
            print(f"[✗] HPM: DISABLED (config: enable=False)")
    
    # ===== Solver Cross-Attention =====
    solver_ctx_cfg = meta_config['solver_context']
    if solver_ctx_cfg['enable'] and hasattr(model, 'solver') and hasattr(model.solver, 'use_context'):
        model.solver_context_active = True
        active_modules['solver_context'] = True
        if verbose:
            print(f"[✓] Solver Cross-Attention: ACTIVE")
    else:
        model.solver_context_active = False
        active_modules['solver_context'] = False
        if verbose:
            reason = "config: enable=False" if not solver_ctx_cfg['enable'] else "module not present"
            print(f"[✗] Solver Cross-Attention: DISABLED ({reason})")
    
    # ===== Cross-Attention Injector =====
    cross_attn_cfg = meta_config['cross_attention']
    if cross_attn_cfg['enable'] and hasattr(model, 'cross_attention_injector'):
        model.cross_attention_active = True
        active_modules['cross_attention'] = True
        if verbose:
            print(f"[✓] Cross-Attention Injector: ACTIVE")
    else:
        model.cross_attention_active = False
        active_modules['cross_attention'] = False
        if verbose:
            reason = "config: enable=False" if not cross_attn_cfg['enable'] else "module not present"
            print(f"[✗] Cross-Attention Injector: DISABLED ({reason})")
    
    if verbose:
        print("="*60 + "\n")
    
    return active_modules


def apply_inference_staging_with_defaults(
    model: nn.Module,
    verbose: bool = True,
) -> Dict[str, bool]:
    """
    Apply inference staging with sensible defaults (no config needed).
    
    Use this when you don't have access to the YAML config.
    All meta-learning modules are enabled with graceful fallback.
    
    Args:
        model: RLAN model to configure
        verbose: Whether to print status messages
        
    Returns:
        Dict mapping module names to their active status
    """
    # Create default config that enables everything with graceful fallback
    default_config = {
        'inference': {
            'meta_learning': {
                'hyperlora': {
                    'enable': True,
                    'require_trained': True,
                    'fallback_on_failure': True,
                },
                'hpm': {
                    'enable': True,
                    'require_nonempty_buffers': True,
                    'min_buffer_entries': 1,
                    'use_static_banks': True,
                    'use_dynamic_banks': True,
                },
                'solver_context': {'enable': True},
                'cross_attention': {'enable': True},
            }
        }
    }
    
    return apply_inference_staging(model, default_config, verbose=verbose)


def get_hpm_buffer_info(model: nn.Module) -> Dict[str, Any]:
    """
    Get detailed information about HPM buffer state.
    
    Useful for debugging and logging.
    
    Args:
        model: RLAN model
        
    Returns:
        Dict with buffer information
    """
    info = {
        'hpm_enabled': getattr(model, 'use_hpm', False),
        'memory_enabled': getattr(model, 'hpm_memory_enabled', False),
        'instance_buffer': {
            'exists': hasattr(model, 'hpm_instance_buffer') and model.hpm_instance_buffer is not None,
            'size': 0,
            'task_ids': [],
        },
        'procedural_buffer': {
            'exists': hasattr(model, 'hpm_procedural_buffer') and model.hpm_procedural_buffer is not None,
            'size': 0,
            'task_ids': [],
        },
        'static_banks': [],
    }
    
    if info['instance_buffer']['exists']:
        buf = model.hpm_instance_buffer
        info['instance_buffer']['size'] = len(buf)
        if hasattr(buf, 'task_ids'):
            info['instance_buffer']['task_ids'] = list(buf.task_ids)[:10]  # First 10
    
    if info['procedural_buffer']['exists']:
        buf = model.hpm_procedural_buffer
        info['procedural_buffer']['size'] = len(buf)
        if hasattr(buf, 'task_ids'):
            info['procedural_buffer']['task_ids'] = list(buf.task_ids)[:10]
    
    # Check static banks
    if hasattr(model, 'hpm') and model.hpm is not None:
        for name in ['COMPOSITIONAL', 'PATTERN', 'RELATIONAL', 'CONCEPT']:
            if hasattr(model.hpm, f'bank_{name.lower()}'):
                info['static_banks'].append(name)
    
    return info

def load_hpm_buffers_from_path(
    model: nn.Module,
    hpm_buffer_path: str,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Load HPM buffers from separate files at the configured path.
    
    This is an alternative to loading from checkpoint for:
    1. Clear separation of buffer versioning
    2. Using different buffers than checkpoint contains
    3. Inference with specific buffer snapshots
    
    Args:
        model: RLAN model
        hpm_buffer_path: Path to HPM buffer directory
        verbose: Whether to print status
        
    Returns:
        Dict with load results and health info
    """
    results = {
        'instance_loaded': False,
        'procedural_loaded': False,
        'instance_size': 0,
        'procedural_size': 0,
        'warnings': [],
    }
    
    hpm_dir = Path(hpm_buffer_path)
    
    if not hpm_dir.exists():
        results['warnings'].append(f"HPM buffer path does not exist: {hpm_dir}")
        if verbose:
            print(f"[!] HPM buffer path not found: {hpm_dir}")
        return results
    
    # Load instance buffer
    instance_path = hpm_dir / "instance_buffer.pt"
    if instance_path.exists():
        if hasattr(model, 'hpm_instance_buffer') and model.hpm_instance_buffer is not None:
            try:
                buffer_state = torch.load(instance_path, map_location='cpu')
                model.hpm_instance_buffer.load_state_dict(buffer_state)
                results['instance_loaded'] = True
                results['instance_size'] = len(model.hpm_instance_buffer)
                
                # Check timestamp for staleness
                if 'save_timestamp' in buffer_state:
                    age_days = (time.time() - buffer_state['save_timestamp']) / 86400
                    save_epoch = buffer_state.get('save_epoch', 'unknown')
                    if verbose:
                        print(f"[✓] Loaded HPM instance buffer: {results['instance_size']} entries (epoch {save_epoch}, {age_days:.1f} days old)")
                else:
                    if verbose:
                        print(f"[✓] Loaded HPM instance buffer: {results['instance_size']} entries")
            except Exception as e:
                results['warnings'].append(f"Failed to load instance buffer: {e}")
                if verbose:
                    print(f"[!] Failed to load instance buffer: {e}")
        else:
            results['warnings'].append("Model has no hpm_instance_buffer to load into")
    else:
        if verbose:
            print(f"[!] Instance buffer file not found: {instance_path}")
    
    # Load procedural buffer
    procedural_path = hpm_dir / "procedural_buffer.pt"
    if procedural_path.exists():
        if hasattr(model, 'hpm_procedural_buffer') and model.hpm_procedural_buffer is not None:
            try:
                buffer_state = torch.load(procedural_path, map_location='cpu')
                model.hpm_procedural_buffer.load_state_dict(buffer_state)
                results['procedural_loaded'] = True
                results['procedural_size'] = len(model.hpm_procedural_buffer)
                
                if 'save_timestamp' in buffer_state:
                    age_days = (time.time() - buffer_state['save_timestamp']) / 86400
                    save_epoch = buffer_state.get('save_epoch', 'unknown')
                    if verbose:
                        print(f"[✓] Loaded HPM procedural buffer: {results['procedural_size']} entries (epoch {save_epoch}, {age_days:.1f} days old)")
                else:
                    if verbose:
                        print(f"[✓] Loaded HPM procedural buffer: {results['procedural_size']} entries")
            except Exception as e:
                results['warnings'].append(f"Failed to load procedural buffer: {e}")
                if verbose:
                    print(f"[!] Failed to load procedural buffer: {e}")
        else:
            results['warnings'].append("Model has no hpm_procedural_buffer to load into")
    else:
        if verbose:
            print(f"[!] Procedural buffer file not found: {procedural_path}")
    
    return results


def check_hpm_buffer_staleness(
    config: Dict[str, Any],
    hpm_buffer_path: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Health check for HPM buffer staleness.
    
    Warns if buffers are older than configured threshold.
    
    Args:
        config: YAML config with model.hpm_buffer_stale_days
        hpm_buffer_path: Override path (else uses config)
        verbose: Whether to print warnings
        
    Returns:
        Dict with health check results
    """
    results = {
        'healthy': True,
        'warnings': [],
        'instance_age_days': None,
        'procedural_age_days': None,
    }
    
    # Get path and threshold from config
    if hpm_buffer_path is None:
        hpm_buffer_path = config.get('model', {}).get('hpm_buffer_path', None)
    
    if hpm_buffer_path is None:
        results['warnings'].append("No hpm_buffer_path configured")
        return results
    
    stale_threshold_days = config.get('model', {}).get('hpm_buffer_stale_days', 7)
    hpm_dir = Path(hpm_buffer_path)
    
    current_time = time.time()
    
    # Check instance buffer
    instance_path = hpm_dir / "instance_buffer.pt"
    if instance_path.exists():
        try:
            buffer_state = torch.load(instance_path, map_location='cpu')
            if 'save_timestamp' in buffer_state:
                age_days = (current_time - buffer_state['save_timestamp']) / 86400
                results['instance_age_days'] = age_days
                
                if age_days > stale_threshold_days:
                    results['healthy'] = False
                    warning = f"Instance buffer is {age_days:.1f} days old (threshold: {stale_threshold_days} days)"
                    results['warnings'].append(warning)
                    if verbose:
                        print(f"⚠️  WARNING: {warning}")
        except Exception as e:
            results['warnings'].append(f"Could not check instance buffer: {e}")
    
    # Check procedural buffer
    procedural_path = hpm_dir / "procedural_buffer.pt"
    if procedural_path.exists():
        try:
            buffer_state = torch.load(procedural_path, map_location='cpu')
            if 'save_timestamp' in buffer_state:
                age_days = (current_time - buffer_state['save_timestamp']) / 86400
                results['procedural_age_days'] = age_days
                
                if age_days > stale_threshold_days:
                    results['healthy'] = False
                    warning = f"Procedural buffer is {age_days:.1f} days old (threshold: {stale_threshold_days} days)"
                    results['warnings'].append(warning)
                    if verbose:
                        print(f"⚠️  WARNING: {warning}")
        except Exception as e:
            results['warnings'].append(f"Could not check procedural buffer: {e}")
    
    if results['healthy'] and verbose:
        print(f"[✓] HPM buffer health check passed")
    
    return results


def apply_inference_staging_with_hpm_loading(
    model: nn.Module,
    config: Dict[str, Any],
    checkpoint: Optional[Dict[str, Any]] = None,
    hpm_buffer_path_override: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Complete inference setup: load HPM buffers, apply staging, health checks.
    
    Priority for HPM buffer loading:
    1. hpm_buffer_path_override (command line argument)
    2. config['model']['hpm_buffer_path'] (YAML)
    3. checkpoint['hpm_instance_buffer'] / checkpoint['hpm_procedural_buffer'] (fallback)
    
    Args:
        model: RLAN model
        config: YAML config
        checkpoint: Loaded checkpoint dict (optional fallback)
        hpm_buffer_path_override: Command-line override for HPM path
        verbose: Print status messages
        
    Returns:
        Dict with staging results and health info
    """
    results = {
        'staging': {},
        'hpm_load': {},
        'health_check': {},
    }
    
    # Determine HPM buffer path
    hpm_path = hpm_buffer_path_override or config.get('model', {}).get('hpm_buffer_path')
    
    # First try loading from separate buffer files
    if hpm_path:
        auto_load = config.get('model', {}).get('hpm_buffer_auto_load', True)
        if auto_load:
            if verbose:
                print(f"\n--- HPM Buffer Loading (from {hpm_path}) ---")
            results['hpm_load'] = load_hpm_buffers_from_path(model, hpm_path, verbose)
            results['health_check'] = check_hpm_buffer_staleness(config, hpm_path, verbose)
    
    # Fallback to checkpoint if buffers not loaded from separate files
    if checkpoint:
        if not results['hpm_load'].get('instance_loaded', False):
            if 'hpm_instance_buffer' in checkpoint:
                if hasattr(model, 'hpm_instance_buffer') and model.hpm_instance_buffer is not None:
                    try:
                        model.hpm_instance_buffer.load_state_dict(checkpoint['hpm_instance_buffer'])
                        if verbose:
                            print(f"[✓] Loaded HPM instance buffer from checkpoint ({len(model.hpm_instance_buffer)} entries)")
                    except Exception as e:
                        if verbose:
                            print(f"[!] Failed to load instance buffer from checkpoint: {e}")
        
        if not results['hpm_load'].get('procedural_loaded', False):
            if 'hpm_procedural_buffer' in checkpoint:
                if hasattr(model, 'hpm_procedural_buffer') and model.hpm_procedural_buffer is not None:
                    try:
                        model.hpm_procedural_buffer.load_state_dict(checkpoint['hpm_procedural_buffer'])
                        if verbose:
                            print(f"[✓] Loaded HPM procedural buffer from checkpoint ({len(model.hpm_procedural_buffer)} entries)")
                    except Exception as e:
                        if verbose:
                            print(f"[!] Failed to load procedural buffer from checkpoint: {e}")
    
    # Apply inference staging
    results['staging'] = apply_inference_staging(model, config, verbose)
    
    return results