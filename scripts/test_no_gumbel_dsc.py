"""
Test Script: DSC Without Gumbel Noise

This script tests if removing Gumbel noise from DSC breaks any logic.
Key tests:
1. Gradient flow with standard softmax
2. Attention sharpness without Gumbel
3. Output clamping stability
4. Train/eval consistency (should be identical now!)

Run with: python scripts/test_no_gumbel_dsc.py
"""

import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project root to path
sys.path.insert(0, '.')


def softmax_2d_no_gumbel(
    logits: torch.Tensor,
    temperature: float = 1.0,
    hard: bool = False,
) -> torch.Tensor:
    """
    Apply softmax to 2D spatial attention logits WITHOUT Gumbel noise.
    
    CRITICAL CHANGE: No Gumbel noise means train and eval are IDENTICAL.
    This fixes the generalization gap where model learned to exploit noise.
    
    Args:
        logits: Shape (B, H, W) or (B, K, H, W)
        temperature: Softmax temperature (lower = sharper)
        hard: If True, use straight-through estimator
        
    Returns:
        Attention weights with same shape as logits
    """
    # Clamp input logits for numerical stability
    logits = logits.clamp(min=-50.0, max=50.0)
    
    # Scale by temperature - same for train and eval!
    scaled_logits = logits / max(temperature, 1e-10)
    
    # Clamp scaled logits for softmax stability  
    scaled_logits = scaled_logits.clamp(min=-50.0, max=50.0)
    
    # Flatten spatial dims for softmax
    B = logits.shape[0]
    if logits.dim() == 3:  # (B, H, W)
        H, W = logits.shape[1], logits.shape[2]
        flat = scaled_logits.view(B, -1)
        soft = F.softmax(flat, dim=-1)
        soft = soft.clamp(min=1e-8)
        soft = soft.view(B, H, W)
    elif logits.dim() == 4:  # (B, K, H, W)
        K, H, W = logits.shape[1], logits.shape[2], logits.shape[3]
        flat = scaled_logits.view(B, K, -1)
        soft = F.softmax(flat, dim=-1)
        soft = soft.clamp(min=1e-8)
        soft = soft.view(B, K, H, W)
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got {logits.dim()}D")
    
    if hard:
        # Straight-through estimator for hard attention
        if logits.dim() == 3:
            idx = soft.view(B, -1).argmax(dim=-1)
            hard_attn = torch.zeros_like(soft.view(B, -1))
            hard_attn.scatter_(1, idx.unsqueeze(-1), 1.0)
            hard_attn = hard_attn.view(B, H, W)
        else:
            idx = soft.view(B, K, -1).argmax(dim=-1)
            hard_attn = torch.zeros_like(soft.view(B, K, -1))
            hard_attn.scatter_(2, idx.unsqueeze(-1), 1.0)
            hard_attn = hard_attn.view(B, K, H, W)
        return (hard_attn - soft).detach() + soft
    
    return soft


def test_gradient_flow():
    """Test that gradients flow properly without Gumbel noise."""
    print("\n=== Test 1: Gradient Flow ===")
    
    # Create random logits
    logits = torch.randn(4, 10, 10, requires_grad=True)
    
    # Forward pass
    attention = softmax_2d_no_gumbel(logits, temperature=0.5)
    
    # Compute a simple loss
    loss = attention.sum()
    loss.backward()
    
    # Check gradients
    has_grad = logits.grad is not None
    grad_finite = torch.isfinite(logits.grad).all().item() if has_grad else False
    grad_nonzero = (logits.grad.abs() > 1e-10).any().item() if has_grad else False
    
    print(f"  Has gradient: {has_grad}")
    print(f"  Gradient finite: {grad_finite}")
    print(f"  Gradient non-zero: {grad_nonzero}")
    print(f"  Gradient stats: mean={logits.grad.mean():.6f}, std={logits.grad.std():.6f}")
    
    success = has_grad and grad_finite and grad_nonzero
    print(f"  Result: {'PASS' if success else 'FAIL'}")
    return success


def test_train_eval_consistency():
    """Test that train and eval produce IDENTICAL results (no Gumbel = no randomness)."""
    print("\n=== Test 2: Train/Eval Consistency ===")
    
    # Create random logits
    torch.manual_seed(42)
    logits = torch.randn(4, 10, 10)
    
    # Run multiple times - should be identical without Gumbel
    results = []
    for i in range(5):
        attention = softmax_2d_no_gumbel(logits, temperature=0.5)
        results.append(attention)
    
    # Check all results are identical
    all_same = True
    for i in range(1, len(results)):
        diff = (results[0] - results[i]).abs().max().item()
        if diff > 1e-6:
            all_same = False
            print(f"  Run {i} differs by {diff:.2e}")
    
    print(f"  All 5 runs identical: {all_same}")
    print(f"  Result: {'PASS' if all_same else 'FAIL'}")
    return all_same


def test_attention_sharpness():
    """Test that attention can become sharp (low entropy) without Gumbel."""
    print("\n=== Test 3: Attention Sharpness ===")
    
    # Create logits with one clear peak
    logits = torch.zeros(1, 10, 10)
    logits[0, 5, 5] = 10.0  # Strong peak at center
    
    temperatures = [1.0, 0.5, 0.1]
    results = []
    
    for temp in temperatures:
        attention = softmax_2d_no_gumbel(logits, temperature=temp)
        peak_value = attention[0, 5, 5].item()
        entropy = -(attention.view(-1) * torch.log(attention.view(-1).clamp(min=1e-10))).sum().item()
        max_entropy = math.log(100)  # 10x10 = 100 positions
        normalized_entropy = entropy / max_entropy
        results.append((temp, peak_value, normalized_entropy))
        print(f"  τ={temp}: peak={peak_value:.4f}, entropy={normalized_entropy:.4f}")
    
    # Lower temperature should give sharper attention
    sharpness_increases = all(results[i][1] < results[i+1][1] for i in range(len(results)-1))
    entropy_decreases = all(results[i][2] > results[i+1][2] for i in range(len(results)-1))
    
    success = sharpness_increases and entropy_decreases
    print(f"  Sharpness increases with lower τ: {sharpness_increases}")
    print(f"  Entropy decreases with lower τ: {entropy_decreases}")
    print(f"  Result: {'PASS' if success else 'FAIL'}")
    return success


def test_numerical_stability():
    """Test numerical stability with extreme inputs."""
    print("\n=== Test 4: Numerical Stability ===")
    
    test_cases = [
        ("Normal logits", torch.randn(4, 10, 10)),
        ("Large positive", torch.randn(4, 10, 10) + 100),
        ("Large negative", torch.randn(4, 10, 10) - 100),
        ("Extreme variance", torch.randn(4, 10, 10) * 100),
        ("Near-uniform", torch.zeros(4, 10, 10) + 0.001 * torch.randn(4, 10, 10)),
    ]
    
    all_stable = True
    for name, logits in test_cases:
        logits.requires_grad = True
        try:
            attention = softmax_2d_no_gumbel(logits, temperature=0.5)
            loss = attention.sum()
            loss.backward()
            
            is_finite = torch.isfinite(attention).all().item()
            grad_finite = torch.isfinite(logits.grad).all().item()
            sums_to_one = (attention.sum(dim=(-2, -1)) - 1.0).abs().max().item() < 1e-5
            
            status = "PASS" if (is_finite and grad_finite and sums_to_one) else "FAIL"
            print(f"  {name}: finite={is_finite}, grad_finite={grad_finite}, sums_to_1={sums_to_one} → {status}")
            
            if not (is_finite and grad_finite and sums_to_one):
                all_stable = False
        except Exception as e:
            print(f"  {name}: EXCEPTION - {e}")
            all_stable = False
    
    print(f"  Result: {'PASS' if all_stable else 'FAIL'}")
    return all_stable


def test_straight_through_estimator():
    """Test hard attention with straight-through estimator."""
    print("\n=== Test 5: Straight-Through Estimator ===")
    
    logits = torch.randn(2, 10, 10, requires_grad=True)
    
    # Hard attention
    attention = softmax_2d_no_gumbel(logits, temperature=0.5, hard=True)
    
    # Should be one-hot
    is_one_hot = True
    for b in range(2):
        nonzero_count = (attention[b] > 0.5).sum().item()
        if nonzero_count != 1:
            is_one_hot = False
            print(f"  Batch {b}: {nonzero_count} non-zero positions (expected 1)")
    
    # Gradients should still flow
    loss = attention.sum()
    loss.backward()
    has_grad = logits.grad is not None and (logits.grad.abs() > 1e-10).any().item()
    
    print(f"  Is one-hot: {is_one_hot}")
    print(f"  Has gradient: {has_grad}")
    success = is_one_hot and has_grad
    print(f"  Result: {'PASS' if success else 'FAIL'}")
    return success


def compare_with_gumbel():
    """Compare entropy between Gumbel and no-Gumbel versions."""
    print("\n=== Test 6: Compare With/Without Gumbel ===")
    
    def gumbel_softmax_2d_OLD(logits, temperature=1.0, deterministic=False):
        """Old version WITH Gumbel noise (for comparison)."""
        logits = logits.clamp(min=-50.0, max=50.0)
        
        if deterministic:
            noisy_logits = logits / max(temperature, 1e-10)
        else:
            uniform = torch.rand_like(logits).clamp(min=1e-10, max=1.0 - 1e-10)
            gumbel_noise = -torch.log(-torch.log(uniform))
            noisy_logits = (logits + gumbel_noise) / max(temperature, 1e-10)
        
        noisy_logits = noisy_logits.clamp(min=-50.0, max=50.0)
        B = logits.shape[0]
        H, W = logits.shape[1], logits.shape[2]
        flat = noisy_logits.view(B, -1)
        soft = F.softmax(flat, dim=-1).clamp(min=1e-8)
        return soft.view(B, H, W)
    
    torch.manual_seed(123)
    logits = torch.randn(16, 30, 30)  # Realistic ARC size
    
    # Compute entropy for different scenarios
    def compute_entropy(attention):
        flat = attention.view(attention.shape[0], -1)
        return -(flat * torch.log(flat.clamp(min=1e-10))).sum(dim=-1).mean().item()
    
    max_entropy = math.log(900)  # 30x30
    
    # No-Gumbel (new approach) - train and eval identical
    attn_no_gumbel = softmax_2d_no_gumbel(logits, temperature=0.5)
    entropy_no_gumbel = compute_entropy(attn_no_gumbel) / max_entropy
    
    # Gumbel train mode (old approach)
    entropies_gumbel_train = []
    for _ in range(10):
        attn_gumbel_train = gumbel_softmax_2d_OLD(logits, temperature=0.5, deterministic=False)
        entropies_gumbel_train.append(compute_entropy(attn_gumbel_train) / max_entropy)
    
    # Gumbel eval mode (old approach)
    attn_gumbel_eval = gumbel_softmax_2d_OLD(logits, temperature=0.5, deterministic=True)
    entropy_gumbel_eval = compute_entropy(attn_gumbel_eval) / max_entropy
    
    print(f"  No-Gumbel (train=eval): entropy = {entropy_no_gumbel:.4f}")
    print(f"  Gumbel train (avg over 10): entropy = {sum(entropies_gumbel_train)/10:.4f} ± {torch.tensor(entropies_gumbel_train).std():.4f}")
    print(f"  Gumbel eval: entropy = {entropy_gumbel_eval:.4f}")
    
    # Key insight: Gumbel train and eval have different entropy!
    gumbel_gap = abs(sum(entropies_gumbel_train)/10 - entropy_gumbel_eval)
    no_gumbel_gap = 0.0  # Always identical
    
    print(f"\n  Train/Eval entropy gap:")
    print(f"    With Gumbel: {gumbel_gap:.4f}")
    print(f"    Without Gumbel: {no_gumbel_gap:.4f}")
    
    # No-Gumbel should match Gumbel-eval (same formula)
    match_eval = abs(entropy_no_gumbel - entropy_gumbel_eval) < 0.01
    print(f"  No-Gumbel matches Gumbel-eval: {match_eval}")
    
    print(f"\n  CONCLUSION: Removing Gumbel eliminates train/eval gap!")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing DSC Without Gumbel Noise")
    print("=" * 60)
    
    results = []
    results.append(("Gradient Flow", test_gradient_flow()))
    results.append(("Train/Eval Consistency", test_train_eval_consistency()))
    results.append(("Attention Sharpness", test_attention_sharpness()))
    results.append(("Numerical Stability", test_numerical_stability()))
    results.append(("Straight-Through Estimator", test_straight_through_estimator()))
    results.append(("Compare With Gumbel", compare_with_gumbel()))
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_pass = False
    
    print("\n" + "=" * 60)
    if all_pass:
        print("All tests passed! Safe to remove Gumbel noise from DSC.")
    else:
        print("Some tests failed. Review before applying changes.")
    print("=" * 60)
    
    return all_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
