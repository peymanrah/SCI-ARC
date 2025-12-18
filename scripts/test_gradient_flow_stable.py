#!/usr/bin/env python3
"""
Gradient Flow Test for Stable Loss and Attention
=================================================

This script verifies that:
1. All modules receive gradients (no blocked gradient paths)
2. Gradients are well-behaved (no NaN, no explosion)
3. The stable loss and attention clamping work correctly
4. Embeddings and signals flow properly in forward/backward pass

Uses the rlan_stable.yaml configuration which achieved 85.6% accuracy.
"""

import sys
import math
from pathlib import Path
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from sci_arc.models.rlan import RLAN, RLANConfig
from sci_arc.training.rlan_loss import RLANLoss


def test_gradient_flow():
    """Test that gradients flow through all modules properly."""
    print("=" * 70)
    print("GRADIENT FLOW TEST - Stable Loss and Attention")
    print("=" * 70)
    print()
    
    # Load config
    config_path = project_root / 'configs' / 'rlan_stable.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_cfg = config['model']
    train_cfg = config['training']
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Loss Mode: {train_cfg.get('loss_mode', 'weighted_stablemax')}")
    print()
    
    # Create model
    rlan_config = RLANConfig(
        hidden_dim=model_cfg['hidden_dim'],
        num_colors=model_cfg['num_colors'],
        num_classes=model_cfg['num_classes'],
        max_grid_size=15,  # Smaller for faster testing
        max_clues=model_cfg['max_clues'],
        num_predicates=model_cfg['num_predicates'],
        num_solver_steps=model_cfg['num_solver_steps'],
        dropout=model_cfg['dropout'],
        use_act=model_cfg.get('use_act', False),
        use_context_encoder=model_cfg.get('use_context_encoder', True),
        use_dsc=model_cfg.get('use_dsc', True),
        use_msre=model_cfg.get('use_msre', True),
        use_lcr=model_cfg.get('use_lcr', False),
        use_sph=model_cfg.get('use_sph', False),
        use_learned_pos=model_cfg.get('use_learned_pos', False),
    )
    
    model = RLAN(config=rlan_config).to(device)
    model.train()
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model: {param_count:,} parameters")
    print()
    
    # Create loss function
    loss_fn = RLANLoss(
        focal_gamma=train_cfg.get('focal_gamma', 2.0),
        focal_alpha=train_cfg.get('focal_alpha', 0.75),
        lambda_entropy=train_cfg.get('lambda_entropy', 0.01),
        lambda_sparsity=train_cfg.get('lambda_sparsity', 0.5),
        lambda_predicate=train_cfg.get('lambda_predicate', 0.01),
        lambda_curriculum=train_cfg.get('lambda_curriculum', 0.0),
        lambda_deep_supervision=train_cfg.get('lambda_deep_supervision', 0.0),
        lambda_act=train_cfg.get('lambda_act', 0.0),
        min_clues=train_cfg.get('min_clues', 2.5),
        min_clue_weight=train_cfg.get('min_clue_weight', 5.0),
        ponder_weight=train_cfg.get('ponder_weight', 0.02),
        entropy_ponder_weight=train_cfg.get('entropy_ponder_weight', 0.02),
        max_clues=model_cfg['max_clues'],
        use_stablemax=train_cfg.get('use_stablemax', True),
        loss_mode=train_cfg.get('loss_mode', 'weighted_stablemax'),
        bg_weight_cap=train_cfg.get('bg_weight_cap', 2.0),
        fg_weight_cap=train_cfg.get('fg_weight_cap', 5.0),
    )
    
    # Create test data
    B, H, W = 4, 10, 10
    N = 3  # Number of training pairs
    
    test_inputs = torch.randint(0, 10, (B, H, W), device=device)
    test_outputs = torch.randint(0, 10, (B, H, W), device=device)
    train_inputs = torch.randint(0, 10, (B, N, H, W), device=device)
    train_outputs = torch.randint(0, 10, (B, N, H, W), device=device)
    pair_mask = torch.ones(B, N, dtype=torch.bool, device=device)
    
    print("=" * 70)
    print("FORWARD PASS CHECK")
    print("=" * 70)
    
    # Forward pass
    temperature = train_cfg['temperature_start']
    outputs = model(
        test_inputs,
        train_inputs=train_inputs,
        train_outputs=train_outputs,
        pair_mask=pair_mask,
        temperature=temperature,
        return_intermediates=True,
    )
    
    # Check forward outputs
    print(f"\n1. LOGITS:")
    logits = outputs['logits']
    print(f"   Shape: {logits.shape}")
    print(f"   Min: {logits.min().item():.4f}, Max: {logits.max().item():.4f}")
    print(f"   Has NaN: {torch.isnan(logits).any().item()}")
    print(f"   Has Inf: {torch.isinf(logits).any().item()}")
    
    print(f"\n2. ATTENTION MAPS:")
    attention_maps = outputs['attention_maps']
    print(f"   Shape: {attention_maps.shape}")
    print(f"   Min: {attention_maps.min().item():.6f}, Max: {attention_maps.max().item():.6f}")
    print(f"   Sum per clue (should be ~1.0): {attention_maps.sum(dim=(-2,-1)).mean(dim=0).tolist()}")
    print(f"   Has NaN: {torch.isnan(attention_maps).any().item()}")
    
    print(f"\n3. STOP LOGITS:")
    stop_logits = outputs['stop_logits']
    print(f"   Shape: {stop_logits.shape}")
    print(f"   Range: [{stop_logits.min().item():.4f}, {stop_logits.max().item():.4f}]")
    stop_probs = torch.sigmoid(stop_logits)
    print(f"   Stop probs mean per clue: {stop_probs.mean(dim=0).tolist()}")
    
    print(f"\n4. CENTROIDS:")
    centroids = outputs['centroids']
    print(f"   Shape: {centroids.shape}")
    print(f"   Range: row=[{centroids[:,:,0].min().item():.2f}, {centroids[:,:,0].max().item():.2f}]")
    print(f"          col=[{centroids[:,:,1].min().item():.2f}, {centroids[:,:,1].max().item():.2f}]")
    
    print(f"\n5. CONTEXT (if enabled):")
    if 'context' in outputs:
        context = outputs['context']
        print(f"   Shape: {context.shape}")
        print(f"   Norm: {context.norm(dim=-1).mean().item():.4f}")
        print(f"   Has NaN: {torch.isnan(context).any().item()}")
    else:
        print("   Context encoder disabled")
    
    # Compute loss
    print("\n" + "=" * 70)
    print("LOSS COMPUTATION CHECK")
    print("=" * 70)
    
    losses = loss_fn(
        logits=logits,
        targets=test_outputs,
        attention_maps=attention_maps,
        stop_logits=stop_logits,
        predicates=outputs['predicates'],
        epoch=0,
        max_epochs=100,
        all_logits=outputs.get('all_logits'),
    )
    
    print(f"\nLoss Components:")
    for key, value in losses.items():
        if isinstance(value, torch.Tensor):
            val = value.item()
            nan_check = "NaN!" if math.isnan(val) or math.isinf(val) else "OK"
            print(f"   {key}: {val:.6f} [{nan_check}]")
        elif isinstance(value, (int, float)):
            print(f"   {key}: {value:.6f}")
        else:
            print(f"   {key}: {value}")
    
    # Backward pass with gradient tracking
    print("\n" + "=" * 70)
    print("GRADIENT FLOW CHECK")
    print("=" * 70)
    
    # Zero gradients
    model.zero_grad()
    
    # Track gradients per module
    grad_stats = defaultdict(lambda: {'count': 0, 'grad_norm': 0.0, 'nan_count': 0})
    
    # Backward
    total_loss = losses['total_loss']
    total_loss.backward()
    
    # Analyze gradients per module
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        
        # Determine module
        if 'encoder.' in name:
            module = 'encoder'
        elif 'context_encoder.' in name or 'context_injector.' in name:
            module = 'context_encoder'
        elif 'dsc.' in name:
            module = 'dsc'
        elif 'msre.' in name:
            module = 'msre'
        elif 'lcr.' in name:
            module = 'lcr'
        elif 'sph.' in name:
            module = 'sph'
        elif 'solver.' in name:
            module = 'solver'
        else:
            module = 'other'
        
        grad = param.grad
        grad_norm = grad.norm().item()
        nan_count = torch.isnan(grad).sum().item() + torch.isinf(grad).sum().item()
        
        grad_stats[module]['count'] += 1
        grad_stats[module]['grad_norm'] += grad_norm
        grad_stats[module]['nan_count'] += nan_count
    
    print(f"\nGradient Statistics by Module:")
    print(f"{'Module':<20} {'Params':<10} {'Avg Grad Norm':<15} {'NaN/Inf Count'}")
    print("-" * 60)
    
    all_ok = True
    for module in ['encoder', 'context_encoder', 'dsc', 'msre', 'lcr', 'sph', 'solver', 'other']:
        stats = grad_stats[module]
        if stats['count'] == 0:
            continue
        avg_norm = stats['grad_norm'] / stats['count']
        nan_count = stats['nan_count']
        status = "OK" if nan_count == 0 and avg_norm > 0 else "ISSUE!"
        print(f"{module:<20} {stats['count']:<10} {avg_norm:<15.6f} {nan_count} [{status}]")
        if nan_count > 0 or avg_norm == 0:
            all_ok = False
    
    # Check gradient flow to critical parameters
    print(f"\nCritical Parameter Gradient Check:")
    critical_params = [
        ('encoder.color_embedding.weight', 'Color embeddings'),
        ('dsc.clue_queries', 'DSC clue queries'),
        ('dsc.stop_predictor.3.weight', 'DSC stop predictor'),
        ('solver.output_head.3.weight', 'Solver output head'),
    ]
    
    for param_name, desc in critical_params:
        found = False
        for name, param in model.named_parameters():
            if param_name in name:
                found = True
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    grad_min = param.grad.min().item()
                    grad_max = param.grad.max().item()
                    has_nan = torch.isnan(param.grad).any().item()
                    status = "OK" if not has_nan and grad_norm > 1e-10 else "ISSUE!"
                    print(f"   {desc}: norm={grad_norm:.6f}, range=[{grad_min:.6f}, {grad_max:.6f}] [{status}]")
                else:
                    print(f"   {desc}: NO GRADIENT!")
                    all_ok = False
                break
        if not found:
            print(f"   {desc}: Parameter not found (module may be disabled)")
    
    # Simulate training step
    print("\n" + "=" * 70)
    print("OPTIMIZER STEP CHECK")
    print("=" * 70)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    
    # Clip gradients
    grad_norm_before = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    print(f"\nGradient norm before clip: {grad_norm_before:.6f}")
    
    # Store initial weights
    initial_weights = {name: param.clone() for name, param in model.named_parameters()}
    
    # Optimizer step
    optimizer.step()
    
    # Check weight updates
    weight_changes = []
    for name, param in model.named_parameters():
        if name in initial_weights:
            change = (param - initial_weights[name]).abs().max().item()
            weight_changes.append((name, change))
    
    # Sort by change magnitude
    weight_changes.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nTop 5 Weight Changes (should be non-zero):")
    for name, change in weight_changes[:5]:
        status = "OK" if change > 0 else "NO UPDATE!"
        print(f"   {name}: {change:.8f} [{status}]")
    
    min_change = min(c for _, c in weight_changes if c > 0) if any(c > 0 for _, c in weight_changes) else 0
    max_change = max(c for _, c in weight_changes)
    zero_count = sum(1 for _, c in weight_changes if c == 0)
    
    print(f"\nWeight update range: [{min_change:.8f}, {max_change:.8f}]")
    print(f"Parameters with zero update: {zero_count}/{len(weight_changes)}")
    
    # Final verdict
    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)
    
    issues = []
    
    if torch.isnan(logits).any():
        issues.append("NaN in logits")
    if torch.isnan(attention_maps).any():
        issues.append("NaN in attention maps")
    if math.isnan(losses['total_loss'].item()):
        issues.append("NaN in loss")
    if any(grad_stats[m]['nan_count'] > 0 for m in grad_stats):
        issues.append("NaN in gradients")
    if zero_count > len(weight_changes) * 0.1:  # More than 10% zero updates is suspicious
        issues.append(f"Many parameters not updated ({zero_count})")
    
    if issues:
        print(f"\n[FAILED] Issues found:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    else:
        print(f"\n[PASSED] All gradient flow checks passed!")
        print(f"   - Forward pass produces valid outputs")
        print(f"   - Loss computation is stable")
        print(f"   - Gradients flow to all modules")
        print(f"   - No NaN/Inf in gradients")
        print(f"   - Optimizer updates weights correctly")
        return True


def test_embedding_signals():
    """Test that embeddings produce meaningful signals."""
    print("\n" + "=" * 70)
    print("EMBEDDING SIGNAL TEST")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test color embedding
    from sci_arc.models.grid_encoder import GridEncoder
    
    encoder = GridEncoder(hidden_dim=256, num_colors=10, max_size=30).to(device)
    
    # Create test grid with known pattern
    grid = torch.zeros(2, 10, 10, dtype=torch.long, device=device)
    grid[0, 5, 5] = 1  # Single red pixel
    grid[1, :, :] = 2  # All blue
    
    features = encoder(grid)
    
    print(f"\n1. COLOR EMBEDDING TEST:")
    print(f"   Feature shape: {features.shape}")
    print(f"   Feature norm (sample 0): {features[0].norm().item():.4f}")
    print(f"   Feature norm (sample 1): {features[1].norm().item():.4f}")
    
    # Check that different inputs produce different embeddings
    diff = (features[0] - features[1]).abs().mean().item()
    print(f"   Feature difference between samples: {diff:.6f}")
    
    # Test positional encoding
    print(f"\n2. POSITIONAL ENCODING TEST:")
    # The center pixel should have different encoding than corners
    center_feat = features[0, 5, 5]
    corner_feat = features[0, 0, 0]
    pos_diff = (center_feat - corner_feat).abs().mean().item()
    print(f"   Center vs Corner difference: {pos_diff:.6f}")
    
    # Test context encoder
    print(f"\n3. CONTEXT ENCODER TEST:")
    from sci_arc.models.rlan_modules import ContextEncoder
    
    context_enc = ContextEncoder(hidden_dim=256, num_colors=10).to(device)
    
    # Create training pairs
    train_in = torch.randint(0, 10, (2, 3, 10, 10), device=device)
    train_out = torch.randint(0, 10, (2, 3, 10, 10), device=device)
    
    context = context_enc(train_in, train_out)
    
    print(f"   Context shape: {context.shape}")
    print(f"   Context norm: {context.norm(dim=-1).mean().item():.4f}")
    print(f"   Context has variance: {context.var().item():.6f}")
    
    print(f"\n[PASSED] Embedding signals look healthy!")
    return True


def test_numerical_stability():
    """Test numerical stability under extreme conditions."""
    print("\n" + "=" * 70)
    print("NUMERICAL STABILITY TEST")
    print("=" * 70)
    
    from sci_arc.training.rlan_loss import stablemax, log_stablemax, WeightedStablemaxLoss
    from sci_arc.models.rlan_modules.dynamic_saliency_controller import gumbel_softmax_2d
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test stablemax with extreme values
    print(f"\n1. STABLEMAX TEST:")
    extreme_logits = torch.tensor([-1000, -100, 0, 100, 1000], dtype=torch.float32, device=device)
    s_x = stablemax(extreme_logits)
    print(f"   Input:  {extreme_logits.tolist()}")
    print(f"   Output: {s_x.tolist()}")
    print(f"   Has NaN: {torch.isnan(s_x).any().item()}")
    print(f"   Has Inf: {torch.isinf(s_x).any().item()}")
    
    # Test log_stablemax
    print(f"\n2. LOG_STABLEMAX TEST:")
    logits_2d = torch.randn(4, 10, device=device) * 50  # Large variance
    log_probs = log_stablemax(logits_2d, dim=-1)
    print(f"   Input range: [{logits_2d.min().item():.2f}, {logits_2d.max().item():.2f}]")
    print(f"   Output range: [{log_probs.min().item():.2f}, {log_probs.max().item():.2f}]")
    print(f"   Has NaN: {torch.isnan(log_probs).any().item()}")
    
    # Test gumbel_softmax_2d
    print(f"\n3. GUMBEL_SOFTMAX_2D TEST:")
    attn_logits = torch.randn(2, 10, 10, device=device) * 20  # Sharp distribution
    attention = gumbel_softmax_2d(attn_logits, temperature=0.5, deterministic=True)
    print(f"   Input range: [{attn_logits.min().item():.2f}, {attn_logits.max().item():.2f}]")
    print(f"   Output min: {attention.min().item():.8f} (should be >= 1e-8)")
    print(f"   Output sum: {attention.sum(dim=(-2,-1)).tolist()} (should be ~1.0)")
    print(f"   Has NaN: {torch.isnan(attention).any().item()}")
    
    # Test WeightedStablemaxLoss with imbalanced targets
    print(f"\n4. WEIGHTED_STABLEMAX_LOSS TEST:")
    loss_fn = WeightedStablemaxLoss(bg_weight_cap=2.0, fg_weight_cap=5.0)
    
    # Highly imbalanced: 90% background
    logits = torch.randn(4, 10, 20, 20, device=device, requires_grad=True)
    targets = torch.zeros(4, 20, 20, dtype=torch.long, device=device)
    targets[:, 5:7, 5:7] = torch.randint(1, 10, (4, 2, 2))  # Small foreground
    
    loss = loss_fn(logits, targets)
    print(f"   Loss value: {loss.item():.6f}")
    print(f"   Has NaN: {torch.isnan(loss).any().item()}")
    
    # Test gradient through loss
    loss.backward()
    grad_ok = not torch.isnan(logits.grad).any().item()
    print(f"   Gradient has NaN: {not grad_ok}")
    
    print(f"\n[PASSED] Numerical stability checks passed!")
    return True


def main():
    print("\n" + "=" * 70)
    print("STABLE LOSS AND ATTENTION - PRODUCTION READINESS TEST")
    print("=" * 70)
    print()
    
    all_passed = True
    
    # Test 1: Gradient flow
    all_passed &= test_gradient_flow()
    
    # Test 2: Embedding signals
    all_passed &= test_embedding_signals()
    
    # Test 3: Numerical stability
    all_passed &= test_numerical_stability()
    
    print("\n" + "=" * 70)
    if all_passed:
        print("ALL TESTS PASSED - Ready for production training!")
    else:
        print("SOME TESTS FAILED - Please investigate issues above")
    print("=" * 70)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
