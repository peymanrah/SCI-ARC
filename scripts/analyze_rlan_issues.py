"""
Deep analysis of RLAN issues from visualization and reports.
Identifies bugs and theoretical problems in the implementation.
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sci_arc.models.rlan import RLAN, RLANConfig


def analyze_prediction_shape_bug():
    """Analyze the prediction shape bug."""
    print("\n" + "="*70)
    print("ISSUE 1: Prediction Shape Bug Analysis")
    print("="*70)
    
    # The bug: predictions = output.argmax(dim=-1) gives wrong shape
    # output shape: (B, num_classes, H, W) = (1, 10, 10, 25)
    # argmax(dim=-1) takes argmax over W dimension, giving (1, 10, 10)
    # Should be argmax(dim=1) to get argmax over classes, giving (1, 10, 25)
    
    print("\n[BUG IDENTIFIED] In test_rlan_with_checkpoint.py:")
    print("  Line: predictions = output.argmax(dim=-1)")
    print("  ")
    print("  output shape:     (B, num_classes, H, W) = (1, 10, H, W)")
    print("  argmax(dim=-1):   takes max over W (last dim) -> (1, 10, H)")
    print("  argmax(dim=1):    takes max over classes -> (1, H, W) [CORRECT]")
    print("  ")
    print("  FIX: Change to 'predictions = output.argmax(dim=1)'")
    
    # Demonstrate
    fake_output = torch.randn(1, 10, 14, 9)  # (B, C, H, W)
    wrong_pred = fake_output.argmax(dim=-1)
    correct_pred = fake_output.argmax(dim=1)
    
    print(f"\n  Example with output shape {list(fake_output.shape)}:")
    print(f"    argmax(dim=-1) shape: {list(wrong_pred.shape)} [WRONG]")
    print(f"    argmax(dim=1) shape:  {list(correct_pred.shape)} [CORRECT]")
    
    return True


def analyze_stop_prob_collapse():
    """Analyze why stop probabilities are identical."""
    print("\n" + "="*70)
    print("ISSUE 2: Stop Probability Collapse")
    print("="*70)
    
    # All tasks have identical stop probs: 0.00247
    # This means sigmoid(logit) = 0.00247
    # So logit = log(0.00247 / (1 - 0.00247)) = -5.99
    
    prob = 0.00247
    logit = np.log(prob / (1 - prob))
    
    print(f"\n[OBSERVATION] All stop probs = {prob:.5f} across ALL tasks")
    print(f"  Implied logit = ln(p/(1-p)) = {logit:.2f}")
    print("\n[DIAGNOSIS] The stop predictor has collapsed to a constant output.")
    print("  Possible causes:")
    print("    1. Stop predictor weights are near-zero (vanishing gradients)")
    print("    2. No supervision signal for stop probs (loss not connected)")
    print("    3. Initialization bias is dominating")
    
    return True


def analyze_centroid_distribution():
    """Analyze why centroids cluster in the middle."""
    print("\n" + "="*70)
    print("ISSUE 3: Centroid Distribution Analysis")
    print("="*70)
    
    print("\n[THEORY] DSC should place clue centroids at foreground object centers")
    print("  If image has 4 separate FG blocks, 4 clues should be at their centers")
    
    print("\n[OBSERVATION] Centroids tend to cluster around grid center")
    print("  This happens because:")
    print("    1. Attention is computed via cosine similarity + position affinity")
    print("    2. Position encoding creates bias toward center (learnable pos_encoding)")
    print("    3. Softmax over HxW gives weighted average -> tends toward center")
    
    print("\n[CRITICAL FLAW] Current DSC finds clues via LEARNED attention, not segmentation")
    print("  The attention mechanism:")
    print("    - Uses query vectors Q = Linear(features)")  
    print("    - Keys K come from same features + position encoding")
    print("    - No explicit foreground/background segmentation")
    print("    - Centroids are weighted average of attention, not object centers")
    
    print("\n[PROPOSED FIX] Segment-then-Center approach:")
    print("  1. Segment input into connected components (FG blobs)")
    print("  2. Compute centroid of each connected component")
    print("  3. Use these as clue anchor points")
    print("  4. Much simpler, deterministic, and aligned with theory")
    
    return True


def analyze_attention_mechanism():
    """Deep dive into DSC attention issues."""
    print("\n" + "="*70)
    print("ISSUE 4: DSC Attention Mechanism Analysis")
    print("="*70)
    
    # Load DSC code to understand the attention formula
    print("\n[CURRENT DSC ATTENTION FORMULA]")
    print("  attn_scores = base_attn * 2.0 + q_pos_affinity * 16.0")
    print("  where:")
    print("    base_attn = cosine_similarity(query, key_features)")
    print("    q_pos_affinity = cosine_similarity(query, pos_encoding)")
    
    print("\n[PROBLEM] Position affinity has 8x higher weight than content!")
    print("  This means attention is primarily position-based, not content-based")
    print("  Centroids will be placed based on position, not foreground objects")
    
    print("\n[EVIDENCE FROM REPORTS]")
    print("  - Small grids (3x3): 79% FG attention - position matters less")
    print("  - Large grids (21x21): 2.3% FG attention - position dominates")
    print("  As grid size increases, position encoding dominates more")
    
    return True


def analyze_encoder_features():
    """Analyze encoder feature maps."""
    print("\n" + "="*70)
    print("ISSUE 5: Encoder Feature Analysis")
    print("="*70)
    
    print("\n[OBSERVATION] Encoder features are shown as 'mean across channels'")
    print("  This visualization loses information about:")
    print("    - Per-channel activations that detect specific colors")
    print("    - Spatial patterns that encode structure")
    print("    - Color-specific features that should differentiate FG from BG")
    
    print("\n[EXPECTED BEHAVIOR]")
    print("  - Encoder should learn distinct features for each color")
    print("  - Features at FG pixels should be different from BG pixels")
    print("  - Color 0 (black/BG) should have distinct feature signature")
    
    print("\n[POTENTIAL ISSUE]")
    print("  - If encoder uses LayerNorm, all features have mean≈0, std≈1")
    print("  - This makes it harder to distinguish FG from BG based on magnitude")
    print("  - Need to rely on feature DIRECTION, not magnitude")
    
    return True


def propose_fixes():
    """Propose concrete fixes for identified issues."""
    print("\n" + "="*70)
    print("PROPOSED FIXES")
    print("="*70)
    
    print("\n[FIX 1] Prediction shape bug in test script:")
    print("  Change: predictions = output.argmax(dim=-1)")
    print("  To:     predictions = output.argmax(dim=1)")
    
    print("\n[FIX 2] Connected-component based clue discovery:")
    print("""
    def find_clue_centroids_from_segmentation(input_grid):
        '''Find clue centroids by segmenting foreground objects.'''
        # Get foreground mask
        fg_mask = (input_grid > 0)  # Non-black pixels
        
        # Find connected components
        from scipy import ndimage
        labeled, num_features = ndimage.label(fg_mask.cpu().numpy())
        
        # Get centroid of each component
        centroids = []
        for i in range(1, num_features + 1):
            coords = np.where(labeled == i)
            cy = coords[0].mean() / (H - 1)  # Normalize to [0,1]
            cx = coords[1].mean() / (W - 1)
            centroids.append([cy, cx])
        
        return torch.tensor(centroids)
    """)
    
    print("\n[FIX 3] Reduce position affinity weight in DSC:")
    print("  Current: attn_scores = base_attn * 2.0 + q_pos_affinity * 16.0")
    print("  Proposed: attn_scores = base_attn * 4.0 + q_pos_affinity * 1.0")
    print("  This makes content-based attention dominate over position")
    
    print("\n[FIX 4] Add foreground-focused attention bias:")
    print("""
    # In DSC attention computation:
    fg_mask = (input_grid > 0).float()  # (B, H, W)
    fg_bias = fg_mask * 5.0  # Add strong bias toward foreground
    attn_scores = attn_scores + fg_bias.unsqueeze(1)
    """)
    
    print("\n[FIX 5] Stop predictor initialization:")
    print("  Initialize stop predictor bias to 0.0 (not negative)")
    print("  Add explicit supervision for stop probs in loss function")
    
    return True


def main():
    print("="*70)
    print("RLAN IMPLEMENTATION ANALYSIS")
    print("="*70)
    
    analyze_prediction_shape_bug()
    analyze_stop_prob_collapse()
    analyze_centroid_distribution()
    analyze_attention_mechanism()
    analyze_encoder_features()
    propose_fixes()
    
    print("\n" + "="*70)
    print("SUMMARY OF CRITICAL BUGS")
    print("="*70)
    print("""
    1. [BUG] Prediction argmax uses wrong dimension (dim=-1 should be dim=1)
    
    2. [BUG] Stop probabilities collapsed to constant 0.00247 for all tasks
       - Stop predictor not learning, outputs constant regardless of input
    
    3. [DESIGN FLAW] DSC attention uses position affinity with 8x higher weight
       than content-based attention, causing centroids to cluster at center
    
    4. [DESIGN FLAW] Centroids found via learned attention, not FG segmentation
       - Theory says clues should be at FG object centers
       - Implementation uses soft attention weighted average
       
    5. [OBSERVATION] Foreground attention ratio drops dramatically on large grids
       - 3x3: 79%, 6x6: 14%, 21x21: 2.3%
       - Position encoding dominates as grid size increases
    """)


if __name__ == "__main__":
    main()
