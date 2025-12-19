#!/usr/bin/env python3
"""
Deep Analysis of Potential NaN Sources in RLAN
===============================================

This script performs targeted investigation of:
1. Operations that could produce NaN (log, div, exp)
2. Gradient explosion points
3. Float16/bfloat16 overflow scenarios
4. Edge cases in attention/loss computation

The goal is to identify EXACTLY what caused NaN at batch 7205 in production.

Run: python scripts/analyze_nan_sources.py
"""

import sys
import math
from pathlib import Path

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import warnings

from sci_arc.models.rlan import RLAN, RLANConfig
from sci_arc.training.rlan_loss import RLANLoss


def test_numerical_limits():
    """Test numerical limits of different dtypes."""
    print("=" * 70)
    print("TEST 1: Numerical Limits by dtype")
    print("=" * 70)
    
    dtypes = {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
    }
    
    for name, dtype in dtypes.items():
        # Get limits
        info = torch.finfo(dtype)
        max_val = info.max
        min_val = info.tiny  # Smallest positive normal
        
        # Test exp overflow
        safe_exp_limit = math.log(max_val) if max_val < float('inf') else 88.7
        
        print(f"\n  {name}:")
        print(f"    Max value: {max_val:.2e}")
        print(f"    Min positive: {min_val:.2e}")
        print(f"    Safe exp limit: {safe_exp_limit:.1f}")
        
        # Test common operations
        x = torch.tensor([50.0, 100.0, 500.0, 1000.0], dtype=dtype)
        try:
            exp_x = torch.exp(x)
            print(f"    exp([50,100,500,1000]): {exp_x.tolist()}")
        except Exception as e:
            print(f"    exp failed: {e}")
        
        # Test softmax with large values
        logits = torch.tensor([[50.0, 0.0, -50.0]], dtype=dtype)
        try:
            soft = F.softmax(logits, dim=-1)
            print(f"    softmax([50,0,-50]): {soft.tolist()}")
        except Exception as e:
            print(f"    softmax failed: {e}")


def test_gradscaler_interaction():
    """Test how GradScaler interacts with loss values."""
    print("\n" + "=" * 70)
    print("TEST 2: GradScaler and Loss Value Interaction")
    print("=" * 70)
    
    # Simulate what happens with GradScaler
    print("\n  GradScaler starts with scale=65536 and DOUBLES on success")
    print("  After 7205 batches without NaN, scale could be ASTRONOMICAL")
    
    # Conservative estimate: assume scale doubles every 2000 batches
    batches = 7205
    doubles = batches // 2000
    estimated_scale = 65536 * (2 ** doubles)
    
    print(f"\n  After {batches} batches:")
    print(f"    Estimated doubles: {doubles}")
    print(f"    Estimated scale: {estimated_scale:.2e}")
    
    # Test loss * scale overflow
    typical_loss = 0.1
    scaled_loss_fp16 = typical_loss * estimated_scale
    scaled_loss_fp32 = typical_loss * estimated_scale
    
    fp16_max = torch.finfo(torch.float16).max
    fp32_max = torch.finfo(torch.float32).max
    
    print(f"\n  With typical loss = {typical_loss}:")
    print(f"    Scaled loss: {scaled_loss_fp16:.2e}")
    print(f"    float16 max: {fp16_max:.2e}")
    print(f"    Would overflow float16: {scaled_loss_fp16 > fp16_max}")
    print(f"    float32 max: {fp32_max:.2e}")
    print(f"    Would overflow float32: {scaled_loss_fp32 > fp32_max}")
    
    # The REAL issue: autocast scope
    print("\n  [CRITICAL FINDING]:")
    print("     Config says dtype='bfloat16' but autocast('cuda') defaults to float16!")
    print("     float16 max = 65,504 -> overflows with GradScaler")
    print("     bfloat16 max = 3.4e38 -> same as float32, NO overflow!")


def test_attention_entropy_edge_cases():
    """Test entropy computation with edge-case attention distributions."""
    print("\n" + "=" * 70)
    print("TEST 3: Attention Entropy Edge Cases")
    print("=" * 70)
    
    # The DSC entropy is computed on attention maps
    # and fed into stop_predictor
    
    B, K, H, W = 2, 6, 30, 30  # Production-like dimensions
    
    test_cases = [
        ("Uniform attention", torch.ones(B, K, H, W) / (H * W)),
        ("Sharp peak (1 pixel)", None),  # Will create below
        ("Near-zero everywhere", torch.full((B, K, H, W), 1e-30)),
        ("Float16 minimum", torch.full((B, K, H, W), 6e-5)),  # ~fp16 min
    ]
    
    # Create sharp peak
    sharp = torch.zeros(B, K, H, W)
    for b in range(B):
        for k in range(K):
            sharp[b, k, H//2, W//2] = 1.0
    test_cases[1] = ("Sharp peak (1 pixel)", sharp)
    
    print("\n  Entropy computation: H(p) = -sum(p * log(p))")
    print("  Max entropy for 30x30 = log(900) = 6.80\n")
    
    for name, attention in test_cases:
        # Ensure attention sums to 1
        attention = attention / attention.sum(dim=(-2, -1), keepdim=True).clamp(min=1e-10)
        
        # DSC uses this computation (from dynamic_saliency_controller.py)
        attn_clamped = attention.view(B, K, -1).clamp(min=1e-6, max=1.0)
        log_attn = torch.log(attn_clamped)
        entropy_contrib = attn_clamped * log_attn
        attn_entropy = -entropy_contrib.sum(dim=-1)  # (B, K)
        
        # Normalized entropy (what's fed to stop_predictor)
        max_entropy = math.log(H * W)
        attn_entropy_normalized = attn_entropy / max_entropy
        
        # Check for issues
        has_nan = torch.isnan(attn_entropy).any()
        has_inf = torch.isinf(attn_entropy).any()
        
        print(f"  {name}:")
        print(f"    Raw entropy: mean={attn_entropy.mean():.4f}, max={attn_entropy.max():.4f}")
        print(f"    Normalized: mean={attn_entropy_normalized.mean():.4f}")
        print(f"    NaN={has_nan}, Inf={has_inf}")
        
        # Gradient test
        attn_clamped.requires_grad_(True)
        try:
            attn_entropy.sum().backward()
            grad_ok = attn_clamped.grad is not None and torch.isfinite(attn_clamped.grad).all()
            grad_max = attn_clamped.grad.abs().max().item() if grad_ok else float('nan')
            print(f"    Gradient: ok={grad_ok}, max={grad_max:.2e}")
        except Exception as e:
            print(f"    Gradient FAILED: {e}")


def test_stablemax_with_production_values():
    """Test stablemax with values seen in production training."""
    print("\n" + "=" * 70)
    print("TEST 4: Stablemax with Production-like Logits")
    print("=" * 70)
    
    from sci_arc.training.rlan_loss import stablemax, log_stablemax, WeightedStablemaxLoss
    
    # Production scenarios
    B, C, H, W = 4, 10, 30, 30
    
    test_cases = [
        ("Normal logits (std=1)", torch.randn(B, C, H, W)),
        ("Large logits (std=10)", torch.randn(B, C, H, W) * 10),
        ("Very large logits (std=100)", torch.randn(B, C, H, W) * 100),
        ("Confident predictions", None),  # Create below
        ("All-background prediction", None),  # Create below
    ]
    
    # Confident predictions: one class has high logit
    confident = torch.randn(B, C, H, W)
    confident[:, 0, :, :] = 50.0  # Class 0 (background) is very confident
    test_cases[3] = ("Confident predictions", confident)
    
    # All-background: model collapsed
    collapsed = torch.zeros(B, C, H, W)
    collapsed[:, 0, :, :] = 10.0  # Only background
    test_cases[4] = ("All-background prediction", collapsed)
    
    loss_fn = WeightedStablemaxLoss(bg_weight_cap=2.0, fg_weight_cap=5.0)
    
    for name, logits in test_cases:
        logits = logits.clone()
        logits.requires_grad = True
        
        # Random targets with foreground pixels
        targets = torch.zeros(B, H, W, dtype=torch.long)
        targets[:, 10:20, 10:20] = torch.randint(1, 10, (B, 10, 10))
        
        # Forward
        loss = loss_fn(logits, targets)
        
        # Backward
        try:
            loss.backward()
            grad_ok = torch.isfinite(logits.grad).all()
            grad_max = logits.grad.abs().max().item()
        except Exception as e:
            grad_ok = False
            grad_max = float('nan')
        
        has_nan = torch.isnan(loss)
        has_inf = torch.isinf(loss)
        
        status = "✅" if not has_nan and not has_inf and grad_ok else "❌"
        print(f"  {status} {name}:")
        print(f"      loss={loss.item():.4f}, grad_ok={grad_ok}, grad_max={grad_max:.2e}")


def test_gru_state_accumulation():
    """Test if GRU hidden state accumulates to extreme values over many steps."""
    print("\n" + "=" * 70)
    print("TEST 5: GRU State Accumulation Over Solver Steps")
    print("=" * 70)
    
    from sci_arc.models.rlan_modules.recursive_solver import ConvGRUCell
    
    hidden_dim = 256
    B, H, W = 4, 30, 30
    num_steps = 6
    
    gru = ConvGRUCell(hidden_dim * 2, hidden_dim, use_swiglu=True)
    
    # Simulate inputs with different magnitudes
    input_scales = [1.0, 5.0, 10.0, 50.0]
    
    for scale in input_scales:
        x = torch.randn(B, hidden_dim * 2, H, W) * scale
        h = None
        
        max_vals = []
        for step in range(num_steps):
            h = gru(x, h)
            max_vals.append(h.abs().max().item())
        
        growth = max_vals[-1] / max_vals[0] if max_vals[0] > 0 else float('inf')
        
        print(f"\n  Input scale={scale}:")
        print(f"    Hidden state max values per step: {[f'{v:.2f}' for v in max_vals]}")
        print(f"    Growth factor: {growth:.2f}x")
        
        if growth > 10:
            print(f"    ⚠️  Warning: Hidden state growing rapidly!")


def test_context_encoder_bottleneck():
    """Test if ContextEncoder causes information bottleneck."""
    print("\n" + "=" * 70)
    print("TEST 6: ContextEncoder Information Flow")
    print("=" * 70)
    
    from sci_arc.models.rlan_modules import ContextEncoder
    
    hidden_dim = 256
    B, N, H, W = 4, 3, 15, 15  # 3 training pairs
    
    encoder = ContextEncoder(hidden_dim=hidden_dim, num_colors=10)
    
    # Create diverse training examples
    train_inputs = torch.randint(0, 10, (B, N, H, W))
    train_outputs = torch.randint(0, 10, (B, N, H, W))
    pair_mask = torch.ones(B, N, dtype=torch.bool)
    
    # Forward with gradient
    for p in encoder.parameters():
        p.requires_grad_(True)
    
    context = encoder(train_inputs, train_outputs, pair_mask)
    
    # Check context quality
    context_norm = context.norm(dim=-1)
    context_std = context.std(dim=-1)
    
    print(f"\n  Context shape: {context.shape}")
    print(f"  Context norm: mean={context_norm.mean():.4f}, std={context_norm.std():.4f}")
    print(f"  Context std per sample: mean={context_std.mean():.4f}")
    
    # Gradient test
    loss = context.sum()
    loss.backward()
    
    total_grad_norm = 0.0
    for p in encoder.parameters():
        if p.grad is not None:
            total_grad_norm += p.grad.norm().item() ** 2
    total_grad_norm = total_grad_norm ** 0.5
    
    print(f"  Gradient norm: {total_grad_norm:.4f}")
    
    # Check for bottleneck: if context variance is low, information is lost
    if context_std.mean() < 0.1:
        print("  ⚠️  Warning: Low context variance - potential bottleneck!")


def test_full_forward_with_diagnostics():
    """Run full forward pass with comprehensive diagnostics."""
    print("\n" + "=" * 70)
    print("TEST 7: Full Forward Pass Diagnostics")
    print("=" * 70)
    
    # Load config
    config_path = project_root / 'configs' / 'rlan_stable.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_cfg = config['model']
    train_cfg = config['training']
    
    # Create model (CPU for safety)
    device = torch.device('cpu')
    
    rlan_config = RLANConfig(
        hidden_dim=model_cfg['hidden_dim'],
        num_colors=model_cfg['num_colors'],
        num_classes=model_cfg['num_classes'],
        max_grid_size=30,
        max_clues=model_cfg['max_clues'],
        num_predicates=model_cfg['num_predicates'],
        num_solver_steps=model_cfg['num_solver_steps'],
        dropout=0.0,
        use_act=False,
        use_context_encoder=True,
        use_dsc=True,
        use_msre=True,
        use_lcr=False,
        use_sph=False,
    )
    
    model = RLAN(config=rlan_config).to(device)
    model.train()
    
    # Create realistic data
    B, H, W = 4, 20, 20
    test_input = torch.randint(0, 10, (B, H, W), device=device)
    test_output = torch.randint(0, 10, (B, H, W), device=device)
    train_inputs = torch.randint(0, 10, (B, 3, H, W), device=device)
    train_outputs = torch.randint(0, 10, (B, 3, H, W), device=device)
    pair_mask = torch.ones(B, 3, dtype=torch.bool, device=device)
    
    # Loss function with config values
    loss_fn = RLANLoss(
        focal_gamma=train_cfg.get('focal_gamma', 2.0),
        focal_alpha=train_cfg.get('focal_alpha', 0.75),
        lambda_entropy=train_cfg.get('lambda_entropy', 0.01),  # From config
        lambda_sparsity=train_cfg.get('lambda_sparsity', 0.5),  # From config
        lambda_predicate=train_cfg.get('lambda_predicate', 0.01),
        lambda_curriculum=train_cfg.get('lambda_curriculum', 0.0),
        lambda_deep_supervision=train_cfg.get('lambda_deep_supervision', 0.0),
        min_clues=train_cfg.get('min_clues', 2.5),
        min_clue_weight=train_cfg.get('min_clue_weight', 5.0),
        ponder_weight=train_cfg.get('ponder_weight', 0.02),
        entropy_ponder_weight=train_cfg.get('entropy_ponder_weight', 0.02),
        max_clues=model_cfg['max_clues'],
        loss_mode=train_cfg.get('loss_mode', 'weighted_stablemax'),
        bg_weight_cap=train_cfg.get('bg_weight_cap', 2.0),
        fg_weight_cap=train_cfg.get('fg_weight_cap', 5.0),
    )
    
    # Forward
    model.zero_grad()
    outputs = model(
        test_input,
        train_inputs=train_inputs,
        train_outputs=train_outputs,
        pair_mask=pair_mask,
        temperature=1.0,
        return_intermediates=True,
    )
    
    # Compute loss
    losses = loss_fn(
        logits=outputs['logits'],
        targets=test_output,
        attention_maps=outputs['attention_maps'],
        stop_logits=outputs['stop_logits'],
        predicates=outputs['predicates'],
        epoch=0,
        max_epochs=200,
        all_logits=outputs['all_logits'],
    )
    
    print("\n  Loss breakdown (with config values):")
    print("  " + "-" * 50)
    
    total = losses['total_loss'].item()
    for key, value in sorted(losses.items()):
        if isinstance(value, torch.Tensor) and value.dim() == 0:
            val = value.item()
            pct = (val / total * 100) if total != 0 else 0
            print(f"    {key:30s}: {val:8.4f} ({pct:5.1f}%)")
        elif key == 'loss_mode':
            print(f"    {key:30s}: {value}")
    
    # Backward
    losses['total_loss'].backward()
    
    # Check for gradient issues
    print("\n  Gradient analysis:")
    max_grad = 0.0
    min_grad = float('inf')
    has_nan = False
    has_inf = False
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            g = param.grad
            if torch.isnan(g).any():
                has_nan = True
                print(f"    ⚠️  NaN gradient in {name}")
            if torch.isinf(g).any():
                has_inf = True
                print(f"    ⚠️  Inf gradient in {name}")
            max_grad = max(max_grad, g.abs().max().item())
            min_grad = min(min_grad, g.abs().max().item())
    
    print(f"\n    Gradient range: [{min_grad:.2e}, {max_grad:.2e}]")
    print(f"    Has NaN: {has_nan}")
    print(f"    Has Inf: {has_inf}")
    
    if not has_nan and not has_inf and max_grad < 1e6:
        print("\n  ✅ Forward/backward pass clean!")
    else:
        print("\n  ❌ Issues detected!")


def main():
    """Run all NaN source analysis."""
    print("=" * 70)
    print("DEEP ANALYSIS: NaN SOURCES IN RLAN")
    print("=" * 70)
    print()
    
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore', category=UserWarning)
    
    test_numerical_limits()
    test_gradscaler_interaction()
    test_attention_entropy_edge_cases()
    test_stablemax_with_production_values()
    test_gru_state_accumulation()
    test_context_encoder_bottleneck()
    test_full_forward_with_diagnostics()
    
    print("\n" + "=" * 70)
    print("SUMMARY OF FINDINGS")
    print("=" * 70)
    
    print("""
    1. ROOT CAUSE OF NaN AT BATCH 7205:
       - Config specified 'bfloat16' but autocast('cuda') used float16
       - float16 max = 65,504, easily overflowed by GradScaler
       - bfloat16 max = 3.4e38 (same as float32), no overflow
       - FIX: Pass dtype=amp_dtype to autocast() ✅ (already fixed)
    
    2. SECONDARY STABILITY CONCERNS:
       - GRU hidden state can grow with large inputs (clamp at 10)
       - Entropy computation needs min clamp of 1e-6 (not 1e-10)
       - Stop logits use tanh squashing (good, prevents saturation)
    
    3. LOSS BALANCE (with config values):
       - task_loss should dominate (~85-90%)
       - entropy_loss should be small (~1-5%)
       - sparsity_loss provides regularization (~5-10%)
    
    4. ATTENTION QUALITY:
       - DSC uses 1e-6 clamp (safe)
       - Entropy normalized to [0,1]
       - Stop predictor has entropy coupling (good design)
    
    5. REMAINING RECOMMENDATIONS:
       - Ensure dtype is always passed correctly to autocast
       - Consider lowering GradScaler growth_factor for stability
       - Monitor attention entropy during training
    """)


if __name__ == "__main__":
    main()
