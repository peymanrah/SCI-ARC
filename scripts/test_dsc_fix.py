#!/usr/bin/env python3
"""Quick test to verify DSC position encoding fix."""

import sys
import os
import json
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sci_arc.models.rlan import RLAN, RLANConfig

def main():
    print("="*70)
    print("DSC POSITION ENCODING FIX VERIFICATION")
    print("="*70)
    
    # Load checkpoint
    checkpoint_path = "checkpoints/warmup3.pt"
    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    config_dict = checkpoint['config']['model']
    valid_fields = {f.name for f in RLANConfig.__dataclass_fields__.values()}
    filtered_config = {k: v for k, v in config_dict.items() if k in valid_fields}
    rlan_config = RLANConfig(**filtered_config)
    
    model = RLAN(config=rlan_config)
    
    # Load weights (non-strict because we added pos_encoding buffer)
    result = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    print(f"Loaded: {len(result.missing_keys)} missing, {len(result.unexpected_keys)} unexpected")
    if result.missing_keys:
        print(f"  Missing: {result.missing_keys}")
    
    model.eval()
    
    # Create test input
    print("\n" + "="*70)
    print("TESTING DSC WITH POSITION ENCODING FIX")
    print("="*70)
    
    # Load a real task
    task_path = "data/arc-agi/data/training/007bbfb7.json"
    with open(task_path) as f:
        task = json.load(f)
    
    # Prepare input
    def pad_grid(grid, size=30):
        h, w = len(grid), len(grid[0])
        padded = torch.zeros(size, size, dtype=torch.long)
        for i in range(min(h, size)):
            for j in range(min(w, size)):
                padded[i, j] = grid[i][j]
        return padded
    
    test_input = pad_grid(task['test'][0]['input']).unsqueeze(0)  # (1, 30, 30)
    print(f"\nTest input shape: {test_input.shape}")
    print(f"Non-zero pixels: {(test_input > 0).sum().item()}")
    
    # Get features
    with torch.no_grad():
        features = model.encoder(test_input)  # (1, 30, 30, 256)
        features_perm = features.permute(0, 3, 1, 2)  # (1, 256, 30, 30)
        
        print(f"\nFeatures shape: {features_perm.shape}")
        print(f"  min={features_perm.min().item():.4f}, max={features_perm.max().item():.4f}")
        
        # Run DSC forward
        centroids, attn_maps, stop_logits = model.dsc(features_perm, temperature=1.0)
        
        print(f"\n--- DSC Output ---")
        print(f"Centroids shape: {centroids.shape}")
        print(f"Attention maps shape: {attn_maps.shape}")
        
        # Check centroid spread
        centroid_spread = centroids.std(dim=1).mean().item()
        print(f"\nCentroid spread (std across 7 clues): {centroid_spread:.4f}")
        
        if centroid_spread < 0.5:
            print("  ⚠️ STILL COLLAPSED! Centroids not spreading apart")
        else:
            print("  ✅ Centroids are SPREAD (fix working!)")
        
        # Check attention peakedness
        for k in range(min(3, centroids.shape[1])):
            attn_k = attn_maps[0, k]  # (30, 30)
            attn_max = attn_k.max().item()
            centroid_k = centroids[0, k].tolist()
            print(f"\n  Clue {k}: centroid=({centroid_k[0]:.1f}, {centroid_k[1]:.1f}), attn_max={attn_max:.4f}")
            
            if attn_max < 0.01:
                print(f"    ⚠️ Attention nearly uniform (max < 0.01)")
            elif attn_max > 0.05:
                print(f"    ✅ Attention is PEAKED (max > 0.05)")
    
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    print("""
If centroid_spread > 0.5 and attn_max > 0.05:
  -> Position encoding fix is WORKING
  -> DSC can now distinguish spatial positions

If still collapsed:
  -> Position encoding scale may need adjustment
  -> Or the existing trained weights are fighting the fix
  -> May need to retrain from scratch or fine-tune
""")

if __name__ == "__main__":
    main()
