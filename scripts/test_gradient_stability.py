#!/usr/bin/env python3
"""
Comprehensive Gradient Flow and Stability Analysis
===================================================

This script performs deep analysis of:
1. Gradient flow through all modules (no dead ends)
2. Numerical stability (no NaN, no explosion)
3. Signal quality (meaningful gradients, not vanishing)
4. Attention quality (sharp, not diffuse)
5. Loss landscape (smooth, no cliffs)

Key areas to investigate for NaN:
- Gumbel softmax in DSC (log of small values)
- Stablemax in loss (division, log operations)
- Entropy computation (log of attention)
- GRU updates (potential accumulation)
- Mixed precision (float16 overflow)

Run: python scripts/test_gradient_stability.py
"""

import sys
import math
from pathlib import Path
from collections import defaultdict
import warnings

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from sci_arc.models.rlan import RLAN, RLANConfig
from sci_arc.training.rlan_loss import (
    RLANLoss, 
    stablemax, 
    log_stablemax,
    WeightedStablemaxLoss,
    FocalStablemaxLoss,
)


def check_tensor_health(tensor, name, verbose=True):
    """Check tensor for NaN, Inf, and extreme values."""
    issues = []
    
    if tensor is None:
        return {"name": name, "status": "None", "issues": []}
    
    if not torch.is_tensor(tensor):
        return {"name": name, "status": "Not tensor", "issues": []}
    
    if not tensor.requires_grad and tensor.grad_fn is None:
        status = "no_grad"
    else:
        status = "has_grad"
    
    nan_count = torch.isnan(tensor).sum().item()
    inf_count = torch.isinf(tensor).sum().item()
    
    if nan_count > 0:
        issues.append(f"NaN: {nan_count}")
    if inf_count > 0:
        issues.append(f"Inf: {inf_count}")
    
    if tensor.numel() > 0 and torch.isfinite(tensor).any():
        finite = tensor[torch.isfinite(tensor)]
        if finite.numel() > 0:
            max_val = finite.abs().max().item()
            min_val = finite.abs().min().item()
            mean_val = finite.mean().item()
            std_val = finite.std().item() if finite.numel() > 1 else 0
            
            if max_val > 1e6:
                issues.append(f"Large values: max={max_val:.2e}")
            if max_val < 1e-10 and tensor.requires_grad:
                issues.append(f"Vanishing: max={max_val:.2e}")
        else:
            max_val = min_val = mean_val = std_val = float('nan')
    else:
        max_val = min_val = mean_val = std_val = float('nan')
    
    result = {
        "name": name,
        "status": status,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "max": max_val,
        "min": min_val,
        "mean": mean_val,
        "std": std_val,
        "nan": nan_count,
        "inf": inf_count,
        "issues": issues,
    }
    
    if verbose and issues:
        print(f"  ⚠️  {name}: {', '.join(issues)}")
    
    return result


def test_stablemax_stability():
    """Test stablemax function for numerical stability."""
    print("=" * 70)
    print("TEST 1: Stablemax Numerical Stability")
    print("=" * 70)
    
    test_cases = [
        ("Normal values", torch.randn(100)),
        ("Large positive", torch.tensor([100.0, 500.0, 1000.0])),
        ("Large negative", torch.tensor([-100.0, -500.0, -1000.0])),
        ("Near zero", torch.tensor([1e-10, 1e-20, 1e-30])),
        ("Mixed extreme", torch.tensor([-1000.0, 0.0, 1000.0])),
        ("Very large", torch.tensor([1e10, 1e20, 1e30])),
    ]
    
    all_passed = True
    for name, x in test_cases:
        x.requires_grad = True
        
        # Forward
        s = stablemax(x)
        log_s = log_stablemax(x.unsqueeze(0), dim=-1).squeeze(0)
        
        # Check outputs
        has_nan = torch.isnan(s).any() or torch.isnan(log_s).any()
        has_inf = torch.isinf(s).any() or torch.isinf(log_s).any()
        
        # Backward
        loss = s.sum() + log_s.sum()
        try:
            loss.backward()
            grad_ok = torch.isfinite(x.grad).all()
        except Exception as e:
            grad_ok = False
        
        status = "✅ PASS" if not has_nan and not has_inf and grad_ok else "❌ FAIL"
        if status == "❌ FAIL":
            all_passed = False
        
        print(f"  {status} {name}: s_max={s.max():.2e}, s_min={s.min():.2e}, "
              f"log_max={log_s.max():.2e}, grad_ok={grad_ok}")
    
    return all_passed


def test_gumbel_softmax_stability():
    """Test Gumbel softmax for numerical stability."""
    print("\n" + "=" * 70)
    print("TEST 2: Gumbel Softmax 2D Stability")
    print("=" * 70)
    
    from sci_arc.models.rlan_modules.dynamic_saliency_controller import gumbel_softmax_2d
    
    test_cases = [
        ("Normal logits", torch.randn(2, 10, 10)),
        ("Large positive", torch.randn(2, 10, 10) + 50),
        ("Large negative", torch.randn(2, 10, 10) - 50),
        ("Sharp peak", torch.zeros(2, 10, 10)),  # Will add spike
        ("Uniform", torch.zeros(2, 10, 10)),
    ]
    
    # Add sharp peak to one case
    test_cases[3][1][0, 5, 5] = 100.0
    test_cases[3][1][1, 3, 7] = 100.0
    
    temperatures = [5.0, 1.0, 0.1, 0.01]
    
    all_passed = True
    for name, logits in test_cases:
        logits = logits.clone().requires_grad_(True)
        
        for temp in temperatures:
            # Forward
            attention = gumbel_softmax_2d(logits, temperature=temp, deterministic=True)
            
            # Check properties
            has_nan = torch.isnan(attention).any()
            has_inf = torch.isinf(attention).any()
            sums_to_one = (attention.sum(dim=(-2, -1)) - 1.0).abs().max() < 0.01
            min_val = attention.min().item()
            max_val = attention.max().item()
            
            # Entropy (should be lower for sharp distributions)
            attn_flat = attention.view(attention.shape[0], -1)
            entropy = -(attn_flat * attn_flat.clamp(min=1e-10).log()).sum(dim=-1).mean().item()
            
            # Backward
            loss = attention.sum()
            if logits.grad is not None:
                logits.grad.zero_()
            try:
                loss.backward(retain_graph=True)
                grad_ok = torch.isfinite(logits.grad).all()
                grad_max = logits.grad.abs().max().item()
            except Exception as e:
                grad_ok = False
                grad_max = float('nan')
            
            status = "✅" if not has_nan and not has_inf and grad_ok else "❌"
            if status == "❌":
                all_passed = False
                print(f"  {status} {name} T={temp}: NaN={has_nan}, Inf={has_inf}, "
                      f"grad_ok={grad_ok}, min={min_val:.2e}")
            elif temp == 0.1:  # Only print for one temperature
                print(f"  {status} {name} T={temp}: entropy={entropy:.2f}, "
                      f"max={max_val:.4f}, grad_max={grad_max:.2e}")
    
    return all_passed


def test_entropy_computation_stability():
    """Test entropy computation for numerical stability."""
    print("\n" + "=" * 70)
    print("TEST 3: Entropy Computation Stability")
    print("=" * 70)
    
    # Simulate different attention distributions
    B, H, W = 2, 30, 30
    
    test_cases = []
    
    # Uniform attention
    uniform = torch.ones(B, H, W) / (H * W)
    test_cases.append(("Uniform", uniform))
    
    # Sharp peak (one pixel has all attention)
    sharp = torch.zeros(B, H, W)
    sharp[:, 15, 15] = 1.0
    test_cases.append(("Sharp peak", sharp))
    
    # Near-zero everywhere except spike
    sparse = torch.full((B, H, W), 1e-10)
    sparse[:, 10, 10] = 1.0 - 1e-10 * (H*W - 1)
    test_cases.append(("Sparse (1e-10)", sparse))
    
    # Very sparse
    very_sparse = torch.full((B, H, W), 1e-30)
    very_sparse[:, 10, 10] = 1.0
    test_cases.append(("Very sparse (1e-30)", very_sparse))
    
    # Softmax of large logits (common scenario)
    large_logits = torch.randn(B, H, W) * 50
    softmax_attn = F.softmax(large_logits.view(B, -1), dim=-1).view(B, H, W)
    test_cases.append(("Softmax large logits", softmax_attn))
    
    all_passed = True
    for name, attention in test_cases:
        attention = attention.clone().requires_grad_(True)
        
        # Method 1: Direct entropy (naive)
        try:
            attn_flat = attention.view(B, -1)
            log_attn_naive = torch.log(attn_flat)
            entropy_naive = -(attn_flat * log_attn_naive).sum(dim=-1).mean()
            naive_ok = torch.isfinite(entropy_naive)
        except:
            entropy_naive = torch.tensor(float('nan'))
            naive_ok = False
        
        # Method 2: Clamped entropy (safe)
        attn_clamped = attention.view(B, -1).clamp(min=1e-10)
        log_attn_safe = torch.log(attn_clamped)
        entropy_safe = -(attn_clamped * log_attn_safe).sum(dim=-1).mean()
        safe_ok = torch.isfinite(entropy_safe)
        
        # Method 3: Strong clamp (1e-6)
        attn_strong = attention.view(B, -1).clamp(min=1e-6)
        log_attn_strong = torch.log(attn_strong)
        entropy_strong = -(attn_strong * log_attn_strong).sum(dim=-1).mean()
        strong_ok = torch.isfinite(entropy_strong)
        
        # Backward test
        if safe_ok:
            entropy_safe.backward(retain_graph=True)
            grad_safe_ok = attention.grad is not None and torch.isfinite(attention.grad).all()
            grad_max = attention.grad.abs().max().item() if attention.grad is not None else float('nan')
        else:
            grad_safe_ok = False
            grad_max = float('nan')
        
        status = "✅" if safe_ok and grad_safe_ok else "❌"
        if status == "❌":
            all_passed = False
        
        print(f"  {status} {name}:")
        print(f"      Naive entropy: {entropy_naive.item():.4f} (ok={naive_ok})")
        print(f"      Safe entropy:  {entropy_safe.item():.4f} (ok={safe_ok}, grad_max={grad_max:.2e})")
    
    return all_passed


def test_full_model_gradient_flow():
    """Test gradient flow through complete RLAN model."""
    print("\n" + "=" * 70)
    print("TEST 4: Full Model Gradient Flow")
    print("=" * 70)
    
    # Load config
    config_path = project_root / 'configs' / 'rlan_stable.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_cfg = config['model']
    train_cfg = config['training']
    
    device = torch.device('cpu')  # Use CPU for stability testing
    print(f"  Device: {device}")
    
    # Create model
    rlan_config = RLANConfig(
        hidden_dim=model_cfg['hidden_dim'],
        num_colors=model_cfg['num_colors'],
        num_classes=model_cfg['num_classes'],
        max_grid_size=15,
        max_clues=model_cfg['max_clues'],
        num_predicates=model_cfg['num_predicates'],
        num_solver_steps=model_cfg['num_solver_steps'],
        dropout=0.0,  # Disable dropout for deterministic testing
        use_act=False,
        use_context_encoder=model_cfg.get('use_context_encoder', True),
        use_dsc=model_cfg.get('use_dsc', True),
        use_msre=model_cfg.get('use_msre', True),
        use_lcr=model_cfg.get('use_lcr', False),
        use_sph=model_cfg.get('use_sph', False),
    )
    
    model = RLAN(config=rlan_config).to(device)
    model.train()
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Model: {param_count:,} parameters")
    
    # Create loss function
    loss_fn = RLANLoss(
        focal_gamma=train_cfg.get('focal_gamma', 2.0),
        focal_alpha=train_cfg.get('focal_alpha', 0.25),
        lambda_entropy=train_cfg.get('lambda_entropy', 0.1),
        lambda_sparsity=train_cfg.get('lambda_sparsity', 0.05),
        max_clues=model_cfg['max_clues'],
        loss_mode=train_cfg.get('loss_mode', 'weighted_stablemax'),
        bg_weight_cap=train_cfg.get('bg_weight_cap', 2.0),
        fg_weight_cap=train_cfg.get('fg_weight_cap', 5.0),
    )
    
    print(f"  Loss Mode: {train_cfg.get('loss_mode', 'weighted_stablemax')}")
    
    # Create test data
    B, H, W = 4, 10, 10
    test_input = torch.randint(0, 10, (B, H, W), device=device)
    test_output = torch.randint(0, 10, (B, H, W), device=device)
    train_inputs = torch.randint(0, 10, (B, 3, H, W), device=device)
    train_outputs = torch.randint(0, 10, (B, 3, H, W), device=device)
    pair_mask = torch.ones(B, 3, dtype=torch.bool, device=device)
    
    # Forward pass
    model.zero_grad()
    outputs = model(
        test_input,
        train_inputs=train_inputs,
        train_outputs=train_outputs,
        pair_mask=pair_mask,
        temperature=1.0,
        return_intermediates=True,
    )
    
    # Check all outputs
    print("\n  Forward pass outputs:")
    for key, tensor in outputs.items():
        if isinstance(tensor, torch.Tensor):
            result = check_tensor_health(tensor, key, verbose=True)
            if not result['issues']:
                print(f"    ✅ {key}: shape={result['shape']}, range=[{result['min']:.4f}, {result['max']:.4f}]")
        elif isinstance(tensor, list) and len(tensor) > 0:
            print(f"    📋 {key}: list of {len(tensor)} tensors")
    
    # Compute loss
    losses = loss_fn(
        logits=outputs['logits'],
        targets=test_output,
        attention_maps=outputs['attention_maps'],
        stop_logits=outputs['stop_logits'],
        predicates=outputs['predicates'],
        epoch=0,
        max_epochs=100,
        all_logits=outputs['all_logits'],
    )
    
    print("\n  Loss components:")
    total_loss = losses['total_loss']
    for key, value in losses.items():
        if isinstance(value, torch.Tensor) and value.dim() == 0:
            print(f"    {key}: {value.item():.6f}")
    
    # Backward pass
    total_loss.backward()
    
    # Check gradients per module
    print("\n  Gradient health per module:")
    module_grads = defaultdict(list)
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            module = name.split('.')[0]
            grad_norm = param.grad.norm().item()
            has_nan = torch.isnan(param.grad).any().item()
            has_inf = torch.isinf(param.grad).any().item()
            module_grads[module].append({
                'name': name,
                'grad_norm': grad_norm,
                'has_nan': has_nan,
                'has_inf': has_inf,
            })
    
    all_passed = True
    for module, grads in module_grads.items():
        total_norm = sum(g['grad_norm'] for g in grads if math.isfinite(g['grad_norm']))
        any_nan = any(g['has_nan'] for g in grads)
        any_inf = any(g['has_inf'] for g in grads)
        
        status = "✅" if not any_nan and not any_inf and total_norm > 0 else "❌"
        if status == "❌":
            all_passed = False
        
        print(f"    {status} {module}: total_grad_norm={total_norm:.4f}, "
              f"n_params={len(grads)}, nan={any_nan}, inf={any_inf}")
    
    # Specific checks for critical modules
    print("\n  Critical path analysis:")
    
    # DSC entropy coupling check
    if 'dsc_entropy_inputs' in outputs:
        entropy_inputs = outputs['dsc_entropy_inputs']
        print(f"    DSC entropy inputs: mean={entropy_inputs.mean():.4f}, "
              f"std={entropy_inputs.std():.4f}")
    
    # Stop probability check
    stop_probs = torch.sigmoid(outputs['stop_logits'])
    print(f"    Stop probabilities: mean={stop_probs.mean():.4f}, "
          f"std={stop_probs.std():.4f}")
    expected_clues = (1 - stop_probs).sum(dim=-1).mean()
    print(f"    Expected clues used: {expected_clues:.2f}")
    
    # Attention sharpness check
    attn = outputs['attention_maps']
    attn_flat = attn.view(B, -1)
    attn_entropy = -(attn_flat.clamp(min=1e-10) * attn_flat.clamp(min=1e-10).log()).sum(dim=-1).mean()
    max_entropy = math.log(H * W)
    print(f"    Attention entropy: {attn_entropy:.4f} / {max_entropy:.2f} (lower=sharper)")
    
    return all_passed


def test_mixed_precision_stability():
    """Test model stability under mixed precision (float16 and bfloat16)."""
    print("\n" + "=" * 70)
    print("TEST 5: Mixed Precision Stability")
    print("=" * 70)
    
    # Skip if CUDA not available
    if not torch.cuda.is_available():
        print("  ⚠️  CUDA not available, testing CPU only")
        dtypes = [torch.float32]
    else:
        dtypes = [torch.float32, torch.float16, torch.bfloat16]
    
    # Create simple test case
    B, C, H, W = 4, 10, 10, 10
    
    for dtype in dtypes:
        dtype_name = str(dtype).split('.')[-1]
        device = torch.device('cuda' if torch.cuda.is_available() and dtype != torch.float32 else 'cpu')
        
        # Test stablemax
        logits = torch.randn(B, C, H, W, device=device, dtype=dtype)
        targets = torch.randint(0, C, (B, H, W), device=device)
        
        try:
            loss_fn = WeightedStablemaxLoss(reduction='mean')
            # Stablemax uses float64 internally, but inputs can be any dtype
            loss = loss_fn(logits, targets)
            loss_ok = torch.isfinite(loss)
            
            status = "✅" if loss_ok else "❌"
            print(f"  {status} {dtype_name}: loss={loss.item():.4f}")
        except Exception as e:
            print(f"  ❌ {dtype_name}: {str(e)[:50]}")


def test_gradient_accumulation_stability():
    """Test stability over many gradient accumulation steps."""
    print("\n" + "=" * 70)
    print("TEST 6: Gradient Accumulation Stability (Simulated Long Training)")
    print("=" * 70)
    
    # Smaller model for speed
    config = RLANConfig(
        hidden_dim=64,
        num_colors=10,
        num_classes=10,
        max_grid_size=10,
        max_clues=3,
        num_predicates=4,
        num_solver_steps=3,
        use_act=False,
        use_context_encoder=False,  # Faster
        use_dsc=True,
        use_msre=True,
        use_lcr=False,
        use_sph=False,
    )
    
    device = torch.device('cpu')
    model = RLAN(config=config).to(device)
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    loss_fn = WeightedStablemaxLoss()
    
    B, H, W = 2, 8, 8
    num_steps = 50
    
    losses = []
    grad_norms = []
    
    print(f"  Running {num_steps} forward-backward steps...")
    
    for step in range(num_steps):
        # Generate random data (simulates different batches)
        test_input = torch.randint(0, 10, (B, H, W), device=device)
        test_output = torch.randint(0, 10, (B, H, W), device=device)
        
        optimizer.zero_grad()
        
        # Forward
        logits = model(test_input, temperature=1.0)
        loss = loss_fn(logits, test_output)
        
        # Backward
        loss.backward()
        
        # Compute grad norm
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.norm().item() ** 2
        total_norm = total_norm ** 0.5
        
        # Clip and step
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        losses.append(loss.item())
        grad_norms.append(total_norm)
        
        # Check for instability
        if not math.isfinite(loss.item()):
            print(f"  ❌ NaN/Inf loss at step {step}")
            return False
        
        if step % 10 == 0:
            print(f"    Step {step}: loss={loss.item():.4f}, grad_norm={total_norm:.4f}")
    
    # Analyze stability
    import statistics
    loss_mean = statistics.mean(losses)
    loss_std = statistics.stdev(losses) if len(losses) > 1 else 0
    grad_mean = statistics.mean(grad_norms)
    grad_std = statistics.stdev(grad_norms) if len(grad_norms) > 1 else 0
    
    print(f"\n  Summary over {num_steps} steps:")
    print(f"    Loss: mean={loss_mean:.4f}, std={loss_std:.4f}")
    print(f"    Grad norm: mean={grad_mean:.4f}, std={grad_std:.4f}")
    
    all_finite = all(math.isfinite(l) for l in losses)
    no_explosion = max(grad_norms) < 1000
    
    status = "✅ PASS" if all_finite and no_explosion else "❌ FAIL"
    print(f"\n  {status}: all_finite={all_finite}, no_explosion={no_explosion}")
    
    return all_finite and no_explosion


def test_loss_component_magnitudes():
    """Analyze loss component magnitudes to ensure balanced training."""
    print("\n" + "=" * 70)
    print("TEST 7: Loss Component Magnitudes")
    print("=" * 70)
    
    # Create test data
    B, C, H, W = 4, 10, 10, 10
    K = 5
    P = 8
    
    logits = torch.randn(B, C, H, W)
    targets = torch.randint(0, C, (B, H, W))
    attention_maps = F.softmax(torch.randn(B, K, H * W), dim=-1).view(B, K, H, W)
    stop_logits = torch.randn(B, K)
    predicates = torch.sigmoid(torch.randn(B, P))
    
    # Create loss function with typical config
    loss_fn = RLANLoss(
        focal_gamma=2.0,
        focal_alpha=0.25,
        lambda_entropy=0.1,
        lambda_sparsity=0.05,
        lambda_predicate=0.01,
        lambda_curriculum=0.0,
        lambda_deep_supervision=0.0,
        max_clues=K,
        loss_mode='weighted_stablemax',
    )
    
    # Compute losses
    losses = loss_fn(
        logits=logits,
        targets=targets,
        attention_maps=attention_maps,
        stop_logits=stop_logits,
        predicates=predicates,
        epoch=0,
        max_epochs=100,
    )
    
    print("  Loss component analysis:")
    print("  " + "-" * 50)
    
    total = losses['total_loss'].item()
    components = []
    
    for key, value in losses.items():
        if isinstance(value, torch.Tensor) and value.dim() == 0:
            val = value.item()
            if key != 'total_loss' and key != 'loss_mode':
                components.append((key, val))
    
    # Sort by magnitude
    components.sort(key=lambda x: abs(x[1]), reverse=True)
    
    for name, val in components:
        pct = (val / total * 100) if total != 0 else 0
        bar = "█" * int(pct / 5) if pct > 0 else ""
        print(f"    {name:25s}: {val:8.4f} ({pct:5.1f}%) {bar}")
    
    print("  " + "-" * 50)
    print(f"    {'TOTAL':25s}: {total:8.4f}")
    
    # Check for dominance issues
    if components:
        top_component = components[0][1]
        if top_component > total * 0.95 and len(components) > 1:
            print(f"\n  ⚠️  Warning: {components[0][0]} dominates the loss ({components[0][1]/total*100:.1f}%)")
            print(f"      Other components may have vanishing gradients.")
    
    return True


def main():
    """Run all stability tests."""
    print("=" * 70)
    print("COMPREHENSIVE GRADIENT FLOW AND STABILITY ANALYSIS")
    print("=" * 70)
    print()
    
    results = {}
    
    # Run tests
    results['stablemax'] = test_stablemax_stability()
    results['gumbel_softmax'] = test_gumbel_softmax_stability()
    results['entropy'] = test_entropy_computation_stability()
    results['full_model'] = test_full_model_gradient_flow()
    results['mixed_precision'] = test_mixed_precision_stability()
    results['accumulation'] = test_gradient_accumulation_stability()
    results['loss_components'] = test_loss_component_magnitudes()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    all_passed = True
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        if not passed:
            all_passed = False
        print(f"  {status} {name}")
    
    print()
    if all_passed:
        print("🎉 All stability tests passed!")
    else:
        print("⚠️  Some tests failed - investigate before production training")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
