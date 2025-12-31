"""
Test Script: Task-Conditioned Clue Queries

This script tests if making clue queries task-conditioned breaks any logic.

Current DSC:
- clue_queries = nn.Parameter(max_clues, hidden_dim)  # STATIC
- Each task uses same K query vectors
- Cannot adapt to task-specific patterns

Proposed change:
- clue_queries = context_proj(task_context)  # DYNAMIC
- Queries are projected from task context
- Can learn task-specific attention patterns

Key tests:
1. Gradient flow through projected queries
2. Output stability (no exploding activations)
3. Compatibility with stop predictor
4. Clamping effectiveness

Run with: python scripts/test_task_conditioned_clues.py
"""

import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, '.')


class TaskConditionedDSC(nn.Module):
    """
    Dynamic Saliency Controller with task-conditioned clue queries.
    
    Instead of static learned queries, this version projects queries
    from the task context, allowing adaptation to each specific task.
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        context_dim: int = 256,  # Dimension of context encoder output
        max_clues: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.max_clues = max_clues
        
        # NEW: Project task context to clue queries
        # Input: context vector (context_dim)
        # Output: K query vectors (max_clues * hidden_dim)
        self.context_to_queries = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, max_clues * hidden_dim),
        )
        
        # Initialize to approximate uniform queries initially
        # This ensures stable starting point
        self._init_query_projection()
        
        # Keep existing query processing
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Query normalization for stability
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.feature_norm = nn.LayerNorm(hidden_dim)
        
        # Stop predictor
        self.stop_predictor = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        # GRU for recurrence
        self.query_gru = nn.GRUCell(hidden_dim, hidden_dim)
        
        self.scale = math.sqrt(hidden_dim)
        
    def _init_query_projection(self):
        """Initialize query projection for stable but expressive outputs."""
        # Use moderate weights to balance stability with expressiveness
        # Too small: queries don't vary with context
        # Too large: queries explode
        layers = [m for m in self.context_to_queries.modules() if isinstance(m, nn.Linear)]
        for i, layer in enumerate(layers):
            if i == len(layers) - 1:
                # Last layer: slightly larger to ensure variation
                nn.init.xavier_uniform_(layer.weight, gain=0.5)
            else:
                nn.init.xavier_uniform_(layer.weight, gain=1.0)
            nn.init.zeros_(layer.bias)
    
    def get_clue_queries(self, context: torch.Tensor) -> torch.Tensor:
        """
        Generate task-conditioned clue queries from context.
        
        Args:
            context: Task context vector (B, context_dim)
            
        Returns:
            queries: Clue queries (B, max_clues, hidden_dim)
        """
        B = context.shape[0]
        
        # Project context to queries
        queries_flat = self.context_to_queries(context)  # (B, max_clues * hidden_dim)
        queries = queries_flat.view(B, self.max_clues, self.hidden_dim)  # (B, K, D)
        
        # CRITICAL: Clamp queries for stability
        # Without clamping, queries can explode → NaN in attention
        queries = queries.clamp(min=-10.0, max=10.0)
        
        return queries
    
    def forward(
        self,
        features: torch.Tensor,  # (B, D, H, W) encoded grid features
        task_context: torch.Tensor = None,   # (B, context_dim) task context
        temperature: float = 0.5,
    ):
        """
        Extract clue anchors using task-conditioned queries.
        """
        B, D, H, W = features.shape
        K = self.max_clues
        
        # Backward compat: accept task_context kwarg
        context = task_context if task_context is not None else torch.zeros(B, self.hidden_dim, device=features.device)
        
        # Get task-conditioned queries
        clue_queries = self.get_clue_queries(context)  # (B, K, D)
        
        # Reshape features
        features_flat = features.permute(0, 2, 3, 1).reshape(B, H * W, D)
        features_flat = self.feature_norm(features_flat)
        
        # Initialize outputs
        all_centroids = []
        all_attention_maps = []
        all_stop_logits = []
        
        cumulative_mask = torch.ones(B, H, W, device=features.device)
        query_state = torch.zeros(B, self.hidden_dim, device=features.device)
        
        for k in range(K):
            # Get task-conditioned query for this clue
            query = clue_queries[:, k, :]  # (B, D)
            query = query + query_state  # Add recurrent state
            query = self.query_norm(query)
            
            # Compute attention
            q = self.query_proj(query)
            k_proj = self.key_proj(features_flat)
            v = self.value_proj(features_flat)
            
            attn_scores = torch.einsum('bd,bnd->bn', q, k_proj) / self.scale
            attn_scores = attn_scores.view(B, H, W)
            
            # Apply mask
            safe_mask = cumulative_mask.clamp(min=1e-6)
            attn_scores = attn_scores + torch.log(safe_mask)
            attn_scores = attn_scores.clamp(min=-50.0, max=50.0)
            
            # Softmax (no Gumbel!)
            scaled_logits = attn_scores / max(temperature, 1e-10)
            scaled_logits = scaled_logits.clamp(min=-50.0, max=50.0)
            attention = F.softmax(scaled_logits.view(B, -1), dim=-1).view(B, H, W)
            attention = attention.clamp(min=1e-8)
            
            # Compute centroid
            row_grid = torch.arange(H, device=features.device).view(1, H, 1).expand(B, H, W)
            col_grid = torch.arange(W, device=features.device).view(1, 1, W).expand(B, H, W)
            row_centroid = (attention * row_grid.float()).sum(dim=(-2, -1))
            col_centroid = (attention * col_grid.float()).sum(dim=(-2, -1))
            centroid = torch.stack([row_centroid, col_centroid], dim=-1)
            
            # Attended features
            attention_flat = attention.view(B, H * W, 1)
            attended_features = (v * attention_flat).sum(dim=1)
            
            # Update recurrent state
            query_state = self.query_gru(attended_features, query_state)
            
            # Compute entropy for stop predictor
            attn_clamped = attention.view(B, -1).clamp(min=1e-6)
            entropy = -(attn_clamped * torch.log(attn_clamped)).sum(dim=-1, keepdim=True)
            max_entropy = math.log(H * W + 1e-6)
            entropy_normalized = entropy / max_entropy
            
            # Stop prediction
            stop_input = torch.cat([attended_features, entropy_normalized], dim=-1)
            stop_logit_raw = self.stop_predictor(stop_input).squeeze(-1)
            stop_logit = 4.0 * torch.tanh(stop_logit_raw / 4.0)  # Soft clamp
            
            # Update mask
            mask_update = 1.0 - 0.9 * attention.detach()
            cumulative_mask = (cumulative_mask * mask_update).clamp(min=1e-6)
            
            all_centroids.append(centroid)
            all_attention_maps.append(attention)
            all_stop_logits.append(stop_logit)
        
        centroids = torch.stack(all_centroids, dim=1)
        attention_maps = torch.stack(all_attention_maps, dim=1)
        stop_logits = torch.stack(all_stop_logits, dim=1)
        
        return centroids, attention_maps, stop_logits


def test_gradient_flow():
    """Test gradients flow through task-conditioned queries."""
    print("\n=== Test 1: Gradient Flow ===")
    
    dsc = TaskConditionedDSC(hidden_dim=64, context_dim=64, max_clues=4)
    
    # Random inputs
    features = torch.randn(4, 64, 10, 10, requires_grad=True)
    context = torch.randn(4, 64, requires_grad=True)
    
    # Forward
    centroids, attention_maps, stop_logits = dsc(features, task_context=context)
    
    # Compute loss
    loss = attention_maps.sum() + stop_logits.sum()
    loss.backward()
    
    # Check gradients
    has_feature_grad = features.grad is not None and (features.grad.abs() > 1e-10).any()
    has_context_grad = context.grad is not None and (context.grad.abs() > 1e-10).any()
    grad_finite = torch.isfinite(features.grad).all() and torch.isfinite(context.grad).all()
    
    print(f"  Features have gradient: {has_feature_grad}")
    print(f"  Context has gradient: {has_context_grad}")
    print(f"  Gradients finite: {grad_finite}")
    
    success = has_feature_grad and has_context_grad and grad_finite
    print(f"  Result: {'PASS' if success else 'FAIL'}")
    return success


def test_output_stability():
    """Test outputs are stable and well-bounded."""
    print("\n=== Test 2: Output Stability ===")
    
    dsc = TaskConditionedDSC(hidden_dim=64, context_dim=64, max_clues=4)
    
    test_cases = [
        ("Normal", torch.randn(4, 64, 10, 10), torch.randn(4, 64)),
        ("Large context", torch.randn(4, 64, 10, 10), torch.randn(4, 64) * 10),
        ("Large features", torch.randn(4, 64, 10, 10) * 10, torch.randn(4, 64)),
        ("Near-zero", torch.randn(4, 64, 10, 10) * 0.01, torch.randn(4, 64) * 0.01),
    ]
    
    all_stable = True
    for name, features, context in test_cases:
        try:
            centroids, attention_maps, stop_logits = dsc(features, task_context=context)
            
            is_finite = (
                torch.isfinite(centroids).all() and
                torch.isfinite(attention_maps).all() and
                torch.isfinite(stop_logits).all()
            )
            
            attention_sums_to_one = (attention_maps.sum(dim=(-2, -1)) - 1.0).abs().max() < 1e-4
            stop_in_range = (stop_logits.abs() <= 4.0).all()  # Should be in [-4, 4] after tanh clamp
            
            status = "PASS" if (is_finite and attention_sums_to_one and stop_in_range) else "FAIL"
            print(f"  {name}: finite={is_finite}, attn_sum={attention_sums_to_one}, stop_range={stop_in_range} → {status}")
            
            if not (is_finite and attention_sums_to_one and stop_in_range):
                all_stable = False
        except Exception as e:
            print(f"  {name}: EXCEPTION - {e}")
            all_stable = False
    
    print(f"  Result: {'PASS' if all_stable else 'FAIL'}")
    return all_stable


def test_context_conditioning():
    """Test that different contexts produce different queries."""
    print("\n=== Test 3: Context Conditioning ===")
    
    dsc = TaskConditionedDSC(hidden_dim=64, context_dim=64, max_clues=4)
    
    # Same features, different contexts
    features = torch.randn(2, 64, 10, 10)
    context1 = torch.randn(1, 64)
    context2 = torch.randn(1, 64)
    
    # Get queries for each context
    queries1 = dsc.get_clue_queries(context1)
    queries2 = dsc.get_clue_queries(context2)
    
    # Should be different
    diff = (queries1 - queries2).abs().mean().item()
    are_different = diff > 0.01
    
    print(f"  Query difference: {diff:.4f}")
    print(f"  Queries are different: {are_different}")
    
    # Same context should give same queries
    queries1_again = dsc.get_clue_queries(context1)
    same = (queries1 - queries1_again).abs().max().item() < 1e-6
    print(f"  Same context → same queries: {same}")
    
    success = are_different and same
    print(f"  Result: {'PASS' if success else 'FAIL'}")
    return success


def test_stop_predictor_integration():
    """Test stop predictor works with task-conditioned queries."""
    print("\n=== Test 4: Stop Predictor Integration ===")
    
    dsc = TaskConditionedDSC(hidden_dim=64, context_dim=64, max_clues=4)
    
    features = torch.randn(8, 64, 10, 10)
    context = torch.randn(8, 64)
    
    centroids, attention_maps, stop_logits = dsc(features, task_context=context)
    
    # Stop logits should be in reasonable range
    stop_in_range = (stop_logits.abs() <= 4.0).all().item()
    
    # Stop probabilities
    stop_probs = torch.sigmoid(stop_logits)
    probs_valid = (stop_probs >= 0).all() and (stop_probs <= 1).all()
    
    # Compute expected clues (like in loss)
    not_stopped = 1.0 - stop_probs
    reach_prob = torch.cumprod(
        torch.cat([torch.ones_like(not_stopped[:, :1]), not_stopped[:, :-1]], dim=1),
        dim=1
    )
    expected_clues = reach_prob.sum(dim=1).mean().item()
    
    print(f"  Stop logits in [-4, 4]: {stop_in_range}")
    print(f"  Stop probs valid: {probs_valid}")
    print(f"  Expected clues: {expected_clues:.2f}")
    
    success = stop_in_range and probs_valid and 1.0 <= expected_clues <= 4.0
    print(f"  Result: {'PASS' if success else 'FAIL'}")
    return success


def test_backward_compatibility():
    """Test that we can still load static clue queries for compatibility."""
    print("\n=== Test 5: Backward Compatibility ===")
    
    # Check if we can fall back to static queries
    # This is important for loading old checkpoints
    
    class HybridDSC(TaskConditionedDSC):
        """DSC that supports both static and task-conditioned queries."""
        
        def __init__(self, *args, use_task_conditioning: bool = True, **kwargs):
            super().__init__(*args, **kwargs)
            self.use_task_conditioning = use_task_conditioning
            
            # Keep static queries for backward compatibility
            self.static_queries = nn.Parameter(torch.randn(self.max_clues, self.hidden_dim))
            nn.init.xavier_uniform_(self.static_queries.unsqueeze(0))
        
        def get_clue_queries(self, context: torch.Tensor) -> torch.Tensor:
            B = context.shape[0]
            
            if self.use_task_conditioning:
                return super().get_clue_queries(context)
            else:
                # Use static queries (old behavior)
                return self.static_queries.unsqueeze(0).expand(B, -1, -1)
    
    # Test static mode
    dsc_static = HybridDSC(hidden_dim=64, context_dim=64, max_clues=4, use_task_conditioning=False)
    features = torch.randn(4, 64, 10, 10)
    context = torch.randn(4, 64)
    
    centroids, attention_maps, stop_logits = dsc_static(features, context)
    static_works = torch.isfinite(attention_maps).all().item()
    
    # Test dynamic mode
    dsc_dynamic = HybridDSC(hidden_dim=64, context_dim=64, max_clues=4, use_task_conditioning=True)
    centroids, attention_maps, stop_logits = dsc_dynamic(features, context)
    dynamic_works = torch.isfinite(attention_maps).all().item()
    
    print(f"  Static mode works: {static_works}")
    print(f"  Dynamic mode works: {dynamic_works}")
    
    success = static_works and dynamic_works
    print(f"  Result: {'PASS' if success else 'FAIL'}")
    return success


def main():
    print("=" * 60)
    print("Testing Task-Conditioned Clue Queries")
    print("=" * 60)
    
    results = []
    results.append(("Gradient Flow", test_gradient_flow()))
    results.append(("Output Stability", test_output_stability()))
    results.append(("Context Conditioning", test_context_conditioning()))
    results.append(("Stop Predictor Integration", test_stop_predictor_integration()))
    results.append(("Backward Compatibility", test_backward_compatibility()))
    
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
        print("All tests passed! Task-conditioned clue queries are safe to implement.")
        print("\nRecommendation:")
        print("1. Add use_task_conditioning flag to DSC (default=True for new training)")
        print("2. Keep static_queries for loading old checkpoints")
        print("3. Use HybridDSC pattern for backward compatibility")
    else:
        print("Some tests failed. Review before implementing.")
    print("=" * 60)
    
    return all_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
