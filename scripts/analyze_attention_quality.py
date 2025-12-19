#!/usr/bin/env python3
"""
Attention Quality Analysis
==========================

This script analyzes if DSC attention is learning meaningful spatial patterns:
1. Attention entropy (should decrease during training)
2. Clue diversity (different clues should attend to different regions)
3. Centroid spread (should cover the grid, not cluster)
4. Stop probability coupling (should correlate with attention quality)

Run: python scripts/analyze_attention_quality.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn.functional as F
import yaml
import math

from sci_arc.models.rlan import RLAN, RLANConfig


def analyze_attention_quality():
    """Analyze DSC attention quality."""
    print("=" * 70)
    print("ATTENTION QUALITY ANALYSIS")
    print("=" * 70)
    
    # Load config
    config_path = project_root / 'configs' / 'rlan_stable.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_cfg = config['model']
    
    # Create model
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
    
    model = RLAN(config=rlan_config)
    model.eval()  # Use deterministic attention
    
    print(f"\nModel: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Max clues: {model_cfg['max_clues']}")
    
    # Create test data with different patterns
    B, H, W = 4, 15, 15
    
    test_patterns = create_test_patterns(B, H, W)
    
    for name, (test_input, train_inputs, train_outputs) in test_patterns.items():
        print(f"\n{'=' * 70}")
        print(f"Pattern: {name}")
        print("=" * 70)
        
        pair_mask = torch.ones(B, train_inputs.shape[1], dtype=torch.bool)
        
        with torch.no_grad():
            outputs = model(
                test_input,
                train_inputs=train_inputs,
                train_outputs=train_outputs,
                pair_mask=pair_mask,
                temperature=1.0,  # Standard temperature
                return_intermediates=True,
            )
        
        attention_maps = outputs['attention_maps']  # (B, K, H, W)
        stop_logits = outputs['stop_logits']  # (B, K)
        centroids = outputs['centroids']  # (B, K, 2)
        
        analyze_single_pattern(attention_maps, stop_logits, centroids, H, W)


def create_test_patterns(B, H, W):
    """Create test patterns with known structure."""
    patterns = {}
    
    # 1. Single object (one clear anchor point)
    single_obj = torch.zeros(B, H, W, dtype=torch.long)
    single_obj[:, 5:10, 5:10] = 1  # Red square
    patterns['Single Object'] = (
        single_obj.clone(),
        single_obj.unsqueeze(1).repeat(1, 2, 1, 1),
        single_obj.unsqueeze(1).repeat(1, 2, 1, 1),
    )
    
    # 2. Two objects (two anchor points)
    two_obj = torch.zeros(B, H, W, dtype=torch.long)
    two_obj[:, 2:5, 2:5] = 2    # Blue square top-left
    two_obj[:, 10:13, 10:13] = 3  # Green square bottom-right
    patterns['Two Objects'] = (
        two_obj.clone(),
        two_obj.unsqueeze(1).repeat(1, 2, 1, 1),
        two_obj.unsqueeze(1).repeat(1, 2, 1, 1),
    )
    
    # 3. Scattered pixels (many potential anchors)
    scattered = torch.zeros(B, H, W, dtype=torch.long)
    for i in range(H):
        for j in range(W):
            if (i + j) % 3 == 0:
                scattered[:, i, j] = (i + j) % 9 + 1
    patterns['Scattered'] = (
        scattered.clone(),
        scattered.unsqueeze(1).repeat(1, 2, 1, 1),
        scattered.unsqueeze(1).repeat(1, 2, 1, 1),
    )
    
    # 4. Empty grid (no clear anchors)
    empty = torch.zeros(B, H, W, dtype=torch.long)
    patterns['Empty'] = (
        empty.clone(),
        empty.unsqueeze(1).repeat(1, 2, 1, 1),
        empty.unsqueeze(1).repeat(1, 2, 1, 1),
    )
    
    return patterns


def analyze_single_pattern(attention_maps, stop_logits, centroids, H, W):
    """Analyze attention for a single pattern."""
    B, K, _, _ = attention_maps.shape
    
    # 1. Attention Entropy per clue
    attn_flat = attention_maps.view(B, K, -1)
    attn_clamped = attn_flat.clamp(min=1e-10)
    entropy_per_clue = -(attn_clamped * attn_clamped.log()).sum(dim=-1)  # (B, K)
    max_entropy = math.log(H * W)
    normalized_entropy = entropy_per_clue / max_entropy
    
    print(f"\n1. Attention Entropy (0=sharp, 1=uniform):")
    print(f"   Per-clue: {[f'{e:.2f}' for e in normalized_entropy.mean(dim=0).tolist()]}")
    print(f"   Mean: {normalized_entropy.mean():.3f}")
    
    # 2. Stop Probabilities
    stop_probs = torch.sigmoid(stop_logits)  # (B, K)
    expected_clues = (1 - stop_probs).sum(dim=-1).mean()
    
    print(f"\n2. Stop Probabilities:")
    print(f"   Per-clue: {[f'{p:.2f}' for p in stop_probs.mean(dim=0).tolist()]}")
    print(f"   Expected clues used: {expected_clues:.2f}")
    
    # 3. Centroid Spread
    centroid_mean = centroids.mean(dim=1)  # (B, 2)
    centroid_std = centroids.std(dim=1)  # (B, 2)
    
    # Distance between clue centroids
    centroid_dists = []
    for b in range(B):
        for i in range(K):
            for j in range(i+1, K):
                dist = ((centroids[b, i] - centroids[b, j]) ** 2).sum().sqrt()
                centroid_dists.append(dist.item())
    
    mean_dist = sum(centroid_dists) / len(centroid_dists) if centroid_dists else 0
    
    print(f"\n3. Centroid Analysis:")
    print(f"   Mean position: row={centroid_mean[:, 0].mean():.1f}, col={centroid_mean[:, 1].mean():.1f}")
    print(f"   Spread (std): row={centroid_std[:, 0].mean():.2f}, col={centroid_std[:, 1].mean():.2f}")
    print(f"   Mean pairwise distance: {mean_dist:.2f} (grid size={H}x{W})")
    
    # 4. Entropy-Stop Coupling
    # Good design: low entropy (sharp attention) should correlate with lower stop probability
    # (model found a good anchor, should use it)
    entropy_flat = normalized_entropy.view(-1)
    stop_flat = stop_probs.view(-1)
    
    # Simple correlation check
    entropy_mean = entropy_flat.mean()
    stop_mean = stop_flat.mean()
    correlation = ((entropy_flat - entropy_mean) * (stop_flat - stop_mean)).mean()
    entropy_std = entropy_flat.std()
    stop_std = stop_flat.std()
    if entropy_std > 0 and stop_std > 0:
        correlation = correlation / (entropy_std * stop_std)
    else:
        correlation = 0.0
    
    print(f"\n4. Entropy-Stop Coupling:")
    print(f"   Correlation: {correlation:.3f}")
    if correlation > 0.3:
        print(f"   Interpretation: Sharp attention -> higher stop prob (good, found anchor)")
    elif correlation < -0.3:
        print(f"   Interpretation: Sharp attention -> lower stop prob (model keeps looking)")
    else:
        print(f"   Interpretation: Weak coupling (model not using entropy signal)")
    
    # 5. Attention Focus Quality
    # Max attention value per clue (higher = sharper focus)
    max_attn = attn_flat.max(dim=-1)[0]  # (B, K)
    
    print(f"\n5. Attention Focus:")
    print(f"   Max attention per clue: {[f'{m:.3f}' for m in max_attn.mean(dim=0).tolist()]}")
    print(f"   Ideal for single pixel: {1.0:.3f}")
    
    # Summary
    print(f"\n{'=' * 40}")
    print("Quality Assessment:")
    
    quality_score = 0
    
    # Check 1: Entropy should be low for simple patterns
    if normalized_entropy.mean() < 0.5:
        print("  [+] Attention is focused (low entropy)")
        quality_score += 1
    else:
        print("  [-] Attention is diffuse (high entropy)")
    
    # Check 2: Centroids should spread for multi-object patterns
    if mean_dist > H / 4:
        print("  [+] Centroids are well-spread")
        quality_score += 1
    else:
        print("  [-] Centroids are clustered")
    
    # Check 3: Stop probabilities should vary
    if stop_probs.std() > 0.1:
        print("  [+] Stop probabilities show variation")
        quality_score += 1
    else:
        print("  [-] Stop probabilities are uniform")
    
    print(f"\nOverall: {quality_score}/3 quality checks passed")


def main():
    print("=" * 70)
    print("DSC ATTENTION QUALITY ANALYSIS")
    print("=" * 70)
    print("\nThis analyzes UNTRAINED model attention patterns.")
    print("After training, attention should become sharper and more meaningful.\n")
    
    analyze_attention_quality()
    
    print("\n" + "=" * 70)
    print("INTERPRETATION GUIDE")
    print("=" * 70)
    print("""
For a well-trained model:
1. Entropy should DECREASE during training (sharper attention)
2. Centroids should cover task-relevant regions
3. Stop probabilities should adapt to task complexity
4. Entropy-stop coupling should be positive (sharp -> stop)

For an UNTRAINED model (like this test):
- Attention is likely diffuse (high entropy)
- Centroids may cluster near center
- Stop probabilities are near initialization (~0.3)

The key insight: These patterns should CHANGE during training.
If they don't improve, there's a learning signal issue.
""")


if __name__ == "__main__":
    main()
