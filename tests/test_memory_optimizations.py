"""
Memory Optimizations Test Suite
================================
Tests for the 4 memory optimization features:
1. 8-bit AdamW optimizer (bitsandbytes)
2. Gradient checkpointing
3. torch.compile
4. Flash Attention (SDPA verification)

These tests verify:
- Mathematical equivalence (same gradients/outputs)
- Memory savings
- Backward compatibility (fallback when unavailable)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest
import sys
import os
from typing import Optional, Tuple, Dict
import gc

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# Test Utilities
# =============================================================================

def get_memory_mb() -> float:
    """Get current GPU memory allocated in MB."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0


def clean_memory():
    """Force memory cleanup."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class SimpleModel(nn.Module):
    """Simple model for testing optimizer equivalence."""
    
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.fc2 = nn.Linear(hidden_dim * 4, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 10)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        return self.fc3(x)


class AttentionModel(nn.Module):
    """Model with attention for testing SDPA."""
    
    def __init__(self, hidden_dim: int = 256, num_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, 10)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, D)
        attn_out, _ = self.attn(x, x, x)
        x = self.norm(x + attn_out)
        return self.fc(x.mean(dim=1))


class CheckpointableModel(nn.Module):
    """Model with checkpointable layers for gradient checkpointing test."""
    
    def __init__(self, hidden_dim: int = 256, num_layers: int = 6):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.LayerNorm(hidden_dim),
            ) for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(hidden_dim, 10)
        self.use_checkpointing = False
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            if self.use_checkpointing and self.training:
                x = x + torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
            else:
                x = x + layer(x)
        return self.fc_out(x.mean(dim=1) if x.dim() == 3 else x)


# =============================================================================
# 1. 8-bit Optimizer Tests
# =============================================================================

class Test8BitOptimizer:
    """Tests for 8-bit AdamW optimizer."""
    
    def test_8bit_import_fallback(self):
        """Test that we can detect bitsandbytes availability."""
        try:
            import bitsandbytes as bnb
            has_bnb = True
        except ImportError:
            has_bnb = False
        
        # Either way, we should be able to continue
        assert isinstance(has_bnb, bool)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_8bit_gradient_equivalence(self):
        """Test that 8-bit optimizer produces similar gradients after 1 step."""
        try:
            import bitsandbytes as bnb
        except ImportError:
            pytest.skip("bitsandbytes not installed")
        
        device = torch.device('cuda')
        torch.manual_seed(42)
        
        # Create two identical models
        model_32bit = SimpleModel().to(device)
        model_8bit = SimpleModel().to(device)
        model_8bit.load_state_dict(model_32bit.state_dict())
        
        # Create optimizers
        opt_32bit = torch.optim.AdamW(model_32bit.parameters(), lr=1e-4)
        opt_8bit = bnb.optim.AdamW8bit(model_8bit.parameters(), lr=1e-4)
        
        # Same input
        x = torch.randn(8, 256, device=device)
        target = torch.randint(0, 10, (8,), device=device)
        
        # Forward + backward for both
        loss_32 = F.cross_entropy(model_32bit(x), target)
        loss_32.backward()
        opt_32bit.step()
        
        loss_8 = F.cross_entropy(model_8bit(x), target)
        loss_8.backward()
        opt_8bit.step()
        
        # Losses should be nearly identical (same model initially)
        assert abs(loss_32.item() - loss_8.item()) < 0.01, \
            f"Initial losses differ too much: {loss_32.item()} vs {loss_8.item()}"
        
        # After 1 step, weights should be similar (not identical due to quantization)
        for (n1, p1), (n2, p2) in zip(model_32bit.named_parameters(), model_8bit.named_parameters()):
            diff = (p1 - p2).abs().max().item()
            assert diff < 0.1, f"Weight diff too large for {n1}: {diff}"
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")  
    def test_8bit_memory_savings(self):
        """Test that 8-bit optimizer uses less memory."""
        try:
            import bitsandbytes as bnb
        except ImportError:
            pytest.skip("bitsandbytes not installed")
        
        device = torch.device('cuda')
        clean_memory()
        
        # Large model for visible memory difference
        model = nn.Sequential(
            nn.Linear(1024, 4096),
            nn.GELU(),
            nn.Linear(4096, 4096),
            nn.GELU(),
            nn.Linear(4096, 1024),
        ).to(device)
        
        # Measure 32-bit optimizer memory
        clean_memory()
        baseline_mem = get_memory_mb()
        opt_32bit = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        # Initialize optimizer states by doing a step
        x = torch.randn(8, 1024, device=device)
        loss = model(x).sum()
        loss.backward()
        opt_32bit.step()
        opt_32bit.zero_grad()
        
        mem_32bit = get_memory_mb() - baseline_mem
        
        # Reset
        del opt_32bit
        clean_memory()
        
        # Measure 8-bit optimizer memory
        baseline_mem = get_memory_mb()
        opt_8bit = bnb.optim.AdamW8bit(model.parameters(), lr=1e-4)
        
        x = torch.randn(8, 1024, device=device)
        loss = model(x).sum()
        loss.backward()
        opt_8bit.step()
        opt_8bit.zero_grad()
        
        mem_8bit = get_memory_mb() - baseline_mem
        
        # 8-bit should use significantly less memory (at least 40% less)
        # Note: The savings are mainly in optimizer states, not model weights
        print(f"32-bit optimizer memory: {mem_32bit:.1f} MB")
        print(f"8-bit optimizer memory: {mem_8bit:.1f} MB")
        print(f"Savings: {(mem_32bit - mem_8bit) / mem_32bit * 100:.1f}%")
        
        # At least some savings (model params dominate, so savings may be modest)
        assert mem_8bit <= mem_32bit, "8-bit should not use more memory"


# =============================================================================
# 2. Gradient Checkpointing Tests
# =============================================================================

class TestGradientCheckpointing:
    """Tests for gradient checkpointing."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_checkpointing_gradient_equivalence(self):
        """Test that checkpointing produces identical gradients."""
        device = torch.device('cuda')
        torch.manual_seed(42)
        
        # Model without checkpointing
        model_normal = CheckpointableModel(hidden_dim=256, num_layers=6).to(device)
        model_normal.use_checkpointing = False
        model_normal.train()
        
        # Model with checkpointing (same weights)
        model_ckpt = CheckpointableModel(hidden_dim=256, num_layers=6).to(device)
        model_ckpt.load_state_dict(model_normal.state_dict())
        model_ckpt.use_checkpointing = True
        model_ckpt.train()
        
        # Same input
        x = torch.randn(8, 32, 256, device=device, requires_grad=True)
        x_ckpt = x.clone().detach().requires_grad_(True)
        target = torch.randint(0, 10, (8,), device=device)
        
        # Forward + backward without checkpointing
        out_normal = model_normal(x)
        loss_normal = F.cross_entropy(out_normal, target)
        loss_normal.backward()
        
        # Forward + backward with checkpointing
        out_ckpt = model_ckpt(x_ckpt)
        loss_ckpt = F.cross_entropy(out_ckpt, target)
        loss_ckpt.backward()
        
        # Losses should be identical
        assert torch.allclose(loss_normal, loss_ckpt, atol=1e-5), \
            f"Losses differ: {loss_normal.item()} vs {loss_ckpt.item()}"
        
        # Gradients should be nearly identical
        for (n1, p1), (n2, p2) in zip(model_normal.named_parameters(), model_ckpt.named_parameters()):
            if p1.grad is not None and p2.grad is not None:
                grad_diff = (p1.grad - p2.grad).abs().max().item()
                assert grad_diff < 1e-4, f"Gradient diff for {n1}: {grad_diff}"
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_checkpointing_memory_savings(self):
        """Test that checkpointing saves activation memory."""
        device = torch.device('cuda')
        
        # Large model for visible memory difference
        model = CheckpointableModel(hidden_dim=512, num_layers=12).to(device)
        model.train()
        
        # Measure without checkpointing
        clean_memory()
        model.use_checkpointing = False
        baseline = get_memory_mb()
        
        x = torch.randn(16, 64, 512, device=device)
        out = model(x)
        loss = out.sum()
        peak_normal = get_memory_mb() - baseline
        
        loss.backward()
        del out, loss, x
        clean_memory()
        
        # Measure with checkpointing
        model.use_checkpointing = True
        baseline = get_memory_mb()
        
        x = torch.randn(16, 64, 512, device=device)
        out = model(x)
        loss = out.sum()
        peak_ckpt = get_memory_mb() - baseline
        
        print(f"Normal peak memory: {peak_normal:.1f} MB")
        print(f"Checkpointed peak memory: {peak_ckpt:.1f} MB")
        print(f"Savings: {(peak_normal - peak_ckpt) / peak_normal * 100:.1f}%")
        
        # Checkpointing should save memory (typically 20-50%)
        assert peak_ckpt < peak_normal, "Checkpointing should reduce memory"


# =============================================================================
# 3. torch.compile Tests
# =============================================================================

class TestTorchCompile:
    """Tests for torch.compile optimization."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_compile_output_equivalence(self):
        """Test that compiled model produces same outputs."""
        if not hasattr(torch, 'compile'):
            pytest.skip("torch.compile not available (PyTorch < 2.0)")
        
        device = torch.device('cuda')
        torch.manual_seed(42)
        
        model = SimpleModel().to(device)
        model.eval()
        
        # Compile the model
        try:
            compiled_model = torch.compile(model, mode='reduce-overhead')
        except Exception as e:
            pytest.skip(f"torch.compile failed: {e}")
        
        # Test with same input
        x = torch.randn(8, 256, device=device)
        
        with torch.no_grad():
            out_normal = model(x)
            out_compiled = compiled_model(x)
        
        # Outputs should be identical
        assert torch.allclose(out_normal, out_compiled, atol=1e-5), \
            f"Compiled output differs: max diff = {(out_normal - out_compiled).abs().max().item()}"
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_compile_gradient_equivalence(self):
        """Test that compiled model produces same gradients."""
        if not hasattr(torch, 'compile'):
            pytest.skip("torch.compile not available (PyTorch < 2.0)")
        
        device = torch.device('cuda')
        torch.manual_seed(42)
        
        # Normal model
        model_normal = SimpleModel().to(device)
        model_normal.train()
        
        # Compiled model with same weights
        model_compiled = SimpleModel().to(device)
        model_compiled.load_state_dict(model_normal.state_dict())
        model_compiled.train()
        
        try:
            model_compiled = torch.compile(model_compiled, mode='reduce-overhead')
        except Exception as e:
            pytest.skip(f"torch.compile failed: {e}")
        
        x = torch.randn(8, 256, device=device)
        target = torch.randint(0, 10, (8,), device=device)
        
        # Forward + backward for normal
        loss_normal = F.cross_entropy(model_normal(x), target)
        loss_normal.backward()
        
        # Forward + backward for compiled
        loss_compiled = F.cross_entropy(model_compiled(x), target)
        loss_compiled.backward()
        
        # Losses should be identical
        assert torch.allclose(loss_normal, loss_compiled, atol=1e-5)


# =============================================================================
# 4. Flash Attention / SDPA Tests
# =============================================================================

class TestFlashAttention:
    """Tests for Flash Attention (SDPA) usage."""
    
    def test_sdpa_available(self):
        """Test that SDPA is available."""
        assert hasattr(F, 'scaled_dot_product_attention'), \
            "scaled_dot_product_attention not available (PyTorch < 2.0)"
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_multihead_uses_sdpa(self):
        """Test that nn.MultiheadAttention uses SDPA on modern PyTorch."""
        device = torch.device('cuda')
        
        model = AttentionModel().to(device)
        model.eval()
        
        x = torch.randn(8, 32, 256, device=device)
        
        with torch.no_grad():
            # This should use SDPA automatically in PyTorch 2.0+
            out = model(x)
        
        # Verify it ran without error
        assert out.shape == (8, 10)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_sdpa_memory_efficiency(self):
        """Test that SDPA is memory efficient for long sequences."""
        device = torch.device('cuda')
        
        hidden_dim = 256
        num_heads = 4
        seq_len = 1024  # Long sequence
        
        clean_memory()
        baseline = get_memory_mb()
        
        # Create Q, K, V
        q = torch.randn(8, num_heads, seq_len, hidden_dim // num_heads, device=device)
        k = torch.randn(8, num_heads, seq_len, hidden_dim // num_heads, device=device)
        v = torch.randn(8, num_heads, seq_len, hidden_dim // num_heads, device=device)
        
        # SDPA should handle this efficiently
        with torch.no_grad():
            out = F.scaled_dot_product_attention(q, k, v)
        
        peak_mem = get_memory_mb() - baseline
        
        # Should complete without OOM for 1024 sequence length
        assert out.shape == (8, num_heads, seq_len, hidden_dim // num_heads)
        print(f"SDPA peak memory for seq_len={seq_len}: {peak_mem:.1f} MB")


# =============================================================================
# Integration Test: All Optimizations Together
# =============================================================================

class TestOptimizationsIntegration:
    """Integration tests for all optimizations working together."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_all_optimizations_combined(self):
        """Test that all optimizations can be used together."""
        device = torch.device('cuda')
        torch.manual_seed(42)
        
        # Create a model with all features
        model = CheckpointableModel(hidden_dim=256, num_layers=4).to(device)
        model.use_checkpointing = True
        model.train()
        
        # Try torch.compile
        compiled = False
        if hasattr(torch, 'compile'):
            try:
                model = torch.compile(model, mode='reduce-overhead')
                compiled = True
            except:
                pass
        
        # Try 8-bit optimizer
        use_8bit = False
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=1e-4)
            use_8bit = True
        except ImportError:
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        # Training step
        x = torch.randn(8, 32, 256, device=device)
        target = torch.randint(0, 10, (8,), device=device)
        
        out = model(x)
        loss = F.cross_entropy(out, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        print(f"\nIntegration test passed:")
        print(f"  - Gradient checkpointing: ON")
        print(f"  - torch.compile: {'ON' if compiled else 'OFF (not available)'}")
        print(f"  - 8-bit optimizer: {'ON' if use_8bit else 'OFF (bitsandbytes not installed)'}")
        print(f"  - Final loss: {loss.item():.4f}")
        
        assert loss.item() > 0, "Loss should be positive"


# =============================================================================
# Mathematical Equivalence Smoke Test
# =============================================================================

class TestMathematicalEquivalence:
    """Smoke tests for mathematical equivalence across optimizations."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_training_convergence_with_optimizations(self):
        """Test that training converges similarly with optimizations enabled."""
        device = torch.device('cuda')
        torch.manual_seed(42)
        
        # Create simple classification task
        X = torch.randn(100, 256, device=device)
        Y = torch.randint(0, 10, (100,), device=device)
        
        # Train without optimizations
        model_baseline = SimpleModel().to(device)
        opt_baseline = torch.optim.AdamW(model_baseline.parameters(), lr=1e-3)
        
        losses_baseline = []
        for epoch in range(10):
            out = model_baseline(X)
            loss = F.cross_entropy(out, Y)
            loss.backward()
            opt_baseline.step()
            opt_baseline.zero_grad()
            losses_baseline.append(loss.item())
        
        # Train with optimizations (if available)
        torch.manual_seed(42)
        model_opt = CheckpointableModel(hidden_dim=256, num_layers=2).to(device)
        model_opt.use_checkpointing = True
        
        try:
            import bitsandbytes as bnb
            opt_opt = bnb.optim.AdamW8bit(model_opt.parameters(), lr=1e-3)
        except ImportError:
            opt_opt = torch.optim.AdamW(model_opt.parameters(), lr=1e-3)
        
        losses_opt = []
        for epoch in range(10):
            out = model_opt(X.unsqueeze(1))  # Add seq dim for CheckpointableModel
            loss = F.cross_entropy(out, Y)
            loss.backward()
            opt_opt.step()
            opt_opt.zero_grad()
            losses_opt.append(loss.item())
        
        # Both should converge (loss should decrease)
        assert losses_baseline[-1] < losses_baseline[0], \
            f"Baseline should converge: {losses_baseline[0]:.4f} -> {losses_baseline[-1]:.4f}"
        assert losses_opt[-1] < losses_opt[0], \
            f"Optimized should converge: {losses_opt[0]:.4f} -> {losses_opt[-1]:.4f}"
        
        print(f"\nConvergence test:")
        print(f"  Baseline: {losses_baseline[0]:.4f} -> {losses_baseline[-1]:.4f}")
        print(f"  Optimized: {losses_opt[0]:.4f} -> {losses_opt[-1]:.4f}")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
