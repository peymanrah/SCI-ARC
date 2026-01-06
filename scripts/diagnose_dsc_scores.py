#!/usr/bin/env python3
"""
FOCUSED DSC Collapse Diagnostic - traces exact attention score computation.

The previous diagnostic revealed:
- ALL centroids at (10.50, 10.50) = exact center of 22x22 grid
- ALL attention maps have IDENTICAL values (max=mean=0.002066 = 1/484)
- This means attention is PERFECTLY UNIFORM (no spatial preference)

This script traces WHY attention scores are uniform:
1. What are the actual attention score values BEFORE softmax?
2. Are the features spatially uniform?
3. Are the query projections working?
4. Is the dot product producing any discrimination?
"""

import sys
import os
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_checkpoint_and_model(checkpoint_path: str):
    """Load checkpoint and create model."""
    from sci_arc.models.rlan import RLAN, RLANConfig
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Extract config
    if 'config' in checkpoint:
        full_config = checkpoint['config']
        if 'model' in full_config and isinstance(full_config['model'], dict):
            config_dict = full_config['model']
        else:
            config_dict = full_config
    else:
        raise ValueError("No config found in checkpoint")
    
    print(f"  Config keys: {list(config_dict.keys())}")
    
    # Build RLANConfig from dict, only using valid fields
    valid_fields = {f.name for f in RLANConfig.__dataclass_fields__.values()}
    filtered_config = {k: v for k, v in config_dict.items() if k in valid_fields}
    print(f"  Using {len(filtered_config)} valid config fields")
    
    rlan_config = RLANConfig(**filtered_config)
    
    # Create model
    print("Creating model with config...")
    model = RLAN(config=rlan_config)
    
    # Load weights
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict'))
    if state_dict is not None:
        try:
            result = model.load_state_dict(state_dict, strict=True)
            print("  Loaded weights (strict)")
        except Exception as e:
            result = model.load_state_dict(state_dict, strict=False)
            print(f"  Loaded weights (non-strict): {len(result.missing_keys)} missing, {len(result.unexpected_keys)} unexpected")
    
    model.eval()
    return model, rlan_config

def load_arc_task(task_path: str, max_grid_size: int = 22):
    """Load a single ARC task."""
    with open(task_path) as f:
        task_data = json.load(f)
    
    train_pairs = task_data.get('train', [])
    test_pairs = task_data.get('test', [])
    
    if not train_pairs or not test_pairs:
        return None
    
    def pad_grid(grid, size):
        h, w = len(grid), len(grid[0]) if grid else 0
        padded = torch.zeros(size, size, dtype=torch.long)
        for i in range(min(h, size)):
            for j in range(min(w, size)):
                padded[i, j] = grid[i][j]
        return padded
    
    train_inputs = []
    train_outputs = []
    for pair in train_pairs[:5]:
        train_inputs.append(pad_grid(pair['input'], max_grid_size))
        train_outputs.append(pad_grid(pair['output'], max_grid_size))
    
    # Pad to max_pairs=5
    while len(train_inputs) < 5:
        train_inputs.append(torch.zeros(max_grid_size, max_grid_size, dtype=torch.long))
        train_outputs.append(torch.zeros(max_grid_size, max_grid_size, dtype=torch.long))
    
    test_input = pad_grid(test_pairs[0]['input'], max_grid_size)
    
    train_inputs = torch.stack(train_inputs)
    train_outputs = torch.stack(train_outputs)
    
    return {
        'train_inputs': train_inputs,
        'train_outputs': train_outputs,
        'test_input': test_input,
        'num_pairs': min(len(task_data['train']), 5)
    }

def analyze_dsc_step_by_step(model, features, task_context=None):
    """
    Step-by-step analysis of DSC forward pass.
    
    Args:
        model: RLAN model with DSC
        features: (B, D, H, W) encoded features
        task_context: Optional task context tensor
    """
    dsc = model.dsc
    B, D, H, W = features.shape
    K = dsc.max_clues
    
    print("\n" + "="*70)
    print("STEP-BY-STEP DSC ANALYSIS")
    print("="*70)
    
    print(f"\nInput features shape: {features.shape}")
    print(f"  min={features.min().item():.4f}, max={features.max().item():.4f}, mean={features.mean().item():.4f}")
    
    # Check spatial variance in features
    spatial_var = features.var(dim=(-2, -1)).mean().item()
    print(f"  Spatial variance (avg over B,D): {spatial_var:.6f}")
    
    # Reshape features
    features_flat = features.permute(0, 2, 3, 1).reshape(B, H * W, D)
    print(f"\nFeatures flattened: {features_flat.shape}")
    print(f"  min={features_flat.min().item():.4f}, max={features_flat.max().item():.4f}")
    
    # Apply feature norm
    features_normed = dsc.feature_norm(features_flat)
    print(f"\nAfter feature_norm: {features_normed.shape}")
    print(f"  min={features_normed.min().item():.4f}, max={features_normed.max().item():.4f}")
    
    # Check if all spatial positions have similar features
    # Compute pairwise cosine similarity between random positions
    pos1 = features_normed[:, 0, :]  # First position
    pos2 = features_normed[:, 240, :]  # Middle position
    pos3 = features_normed[:, H*W-1, :]  # Last position
    
    cos_12 = F.cosine_similarity(pos1, pos2, dim=-1).mean().item()
    cos_13 = F.cosine_similarity(pos1, pos3, dim=-1).mean().item()
    cos_23 = F.cosine_similarity(pos2, pos3, dim=-1).mean().item()
    print(f"\nFeature similarity across positions:")
    print(f"  Pos 0 vs Pos 240: cos_sim={cos_12:.4f}")
    print(f"  Pos 0 vs Pos {H*W-1}: cos_sim={cos_13:.4f}")
    print(f"  Pos 240 vs Pos {H*W-1}: cos_sim={cos_23:.4f}")
    
    if cos_12 > 0.95 and cos_13 > 0.95:
        print("  ⚠️ WARNING: Features are nearly IDENTICAL across all positions!")
        print("     This will cause uniform attention (collapse)!")
    
    # Analyze clue queries
    print(f"\nClue queries shape: {dsc.clue_queries.shape}")
    print(f"  Norms: {[f'{dsc.clue_queries[k].norm().item():.3f}' for k in range(K)]}")
    
    # Query-key projections
    q0 = dsc.query_proj(dsc.clue_queries[0:1])  # (1, D)
    print(f"\nQuery 0 projected: norm={q0.norm().item():.4f}")
    
    k_proj = dsc.key_proj(features_normed)  # (B, H*W, D)
    print(f"Key projection shape: {k_proj.shape}")
    print(f"  min={k_proj.min().item():.4f}, max={k_proj.max().item():.4f}, std={k_proj.std().item():.4f}")
    
    # ALSO TEST WITHOUT feature_norm
    print("\n--- Test WITHOUT feature_norm ---")
    k_proj_raw = dsc.key_proj(features_flat)  # Skip LayerNorm!
    print(f"Key projection (no LayerNorm): std={k_proj_raw.std().item():.4f}")
    
    q0_expanded = q0.expand(B, -1)
    attn_scores_raw = torch.einsum('bd,bnd->bn', q0_expanded, k_proj_raw) / math.sqrt(D)
    print(f"Raw attention scores (no LayerNorm):")
    print(f"  min={attn_scores_raw.min().item():.4f}, max={attn_scores_raw.max().item():.4f}")
    print(f"  std={attn_scores_raw.std().item():.4f}, range={attn_scores_raw.max().item() - attn_scores_raw.min().item():.4f}")
    
    # Apply softmax to raw scores (no LayerNorm)
    attn_soft_raw = F.softmax(attn_scores_raw, dim=-1)
    print(f"After softmax (no LayerNorm):")
    print(f"  max={attn_soft_raw.max().item():.6f}")
    print(f"  uniform value={1/(H*W):.6f}")
    
    if attn_soft_raw.max().item() > 0.01:
        print("  ✅ WITHOUT LayerNorm: Attention is PEAKED (working correctly)")
        print("     This confirms LayerNorm is causing the collapse!")
    else:
        print("  ⚠️ Still uniform even without LayerNorm")
    
    print("\n--- End no-LayerNorm test ---")
    
    # Compute attention scores (WITH LayerNorm - original path)
    scale = math.sqrt(D)
    attn_scores = torch.einsum('bd,bnd->bn', q0_expanded, k_proj) / scale
    print(f"\nRaw attention scores (before mask/softmax):")
    print(f"  Shape: {attn_scores.shape}")
    print(f"  min={attn_scores.min().item():.4f}, max={attn_scores.max().item():.4f}")
    print(f"  mean={attn_scores.mean().item():.4f}, std={attn_scores.std().item():.4f}")
    print(f"  range (max-min)={attn_scores.max().item() - attn_scores.min().item():.4f}")
    
    # Check score variance
    if attn_scores.std().item() < 0.1:
        print(f"  ⚠️ CRITICAL: Attention score std={attn_scores.std().item():.4f} < 0.1")
        print(f"     After softmax, this will be nearly UNIFORM!")
        print(f"     Cause: Features too similar OR Query doesn't discriminate")
    
    # Show score histogram
    scores_flat = attn_scores.view(-1)
    print(f"\nScore distribution (first batch):")
    for pctl in [0, 25, 50, 75, 100]:
        val = torch.quantile(scores_flat, pctl / 100).item()
        print(f"  {pctl}th percentile: {val:.4f}")
    
    # Apply softmax
    attn_2d = attn_scores.view(B, H, W)
    attn_soft = F.softmax(attn_scores, dim=-1).view(B, H, W)
    print(f"\nAfter softmax:")
    print(f"  max={attn_soft.max().item():.6f}")
    print(f"  min={attn_soft.min().item():.6f}")
    print(f"  Expected uniform value (1/{H*W}): {1/(H*W):.6f}")
    
    if attn_soft.max().item() < 0.01:
        print(f"  ⚠️ CRITICAL: max attention = {attn_soft.max().item():.6f} (uniform!)")
        print(f"     CENTROIDS WILL COLLAPSE TO CENTER")
    
    # Compute centroid
    row_grid = torch.arange(H).float().view(1, H, 1).expand(B, H, W)
    col_grid = torch.arange(W).float().view(1, 1, W).expand(B, H, W)
    
    row_centroid = (attn_soft * row_grid).sum(dim=(-2, -1))
    col_centroid = (attn_soft * col_grid).sum(dim=(-2, -1))
    
    print(f"\nCentroid from uniform attention:")
    print(f"  Expected (center): ({(H-1)/2:.2f}, {(W-1)/2:.2f})")
    print(f"  Computed: ({row_centroid.mean().item():.2f}, {col_centroid.mean().item():.2f})")
    
    return {
        'spatial_var': spatial_var,
        'feature_cos_sim': (cos_12 + cos_13 + cos_23) / 3,
        'attn_score_std': attn_scores.std().item(),
        'attn_score_range': (attn_scores.max() - attn_scores.min()).item(),
        'attn_max': attn_soft.max().item()
    }

def diagnose_encoder(model, train_inputs, train_outputs, test_input, pair_mask):
    """
    Diagnose whether the encoder is producing spatially uniform features.
    """
    print("\n" + "="*70)
    print("ENCODER DIAGNOSIS")
    print("="*70)
    
    B = train_inputs.shape[0]
    device = next(model.parameters()).device
    
    # Move to device
    train_inputs = train_inputs.to(device)
    train_outputs = train_outputs.to(device)
    test_input = test_input.to(device)
    pair_mask = pair_mask.to(device)
    
    # Get embeddings through GridEncoder
    with torch.no_grad():
        # Run through encoder (GridEncoder expects (B, H, W) LongTensor)
        test_features = model.encoder(test_input)  # (B, D, H, W)
        print(f"\nTest features from GridEncoder: {test_features.shape}")
        print(f"  min={test_features.min().item():.4f}, max={test_features.max().item():.4f}")
        
        # Spatial variance in features
        feat_spatial_var = test_features.var(dim=(2, 3)).mean().item()
        print(f"  Spatial variance: {feat_spatial_var:.6f}")
        
        # Count non-zero pixels
        nonzero_mask = (test_input > 0).float()
        nonzero_count = nonzero_mask.sum(dim=(-2, -1))
        print(f"  Non-zero pixels per sample: {nonzero_count.tolist()}")
        
        # Check if empty pixels dominate
        H, W = test_input.shape[-2:]
        total_pixels = H * W
        empty_ratio = 1 - nonzero_count / total_pixels
        print(f"  Empty pixel ratio: {empty_ratio.mean().item():.2%}")
        
        if empty_ratio.mean().item() > 0.9:
            print("  ⚠️ WARNING: >90% of pixels are empty (background)")
            print("     The encoder sees mostly identical background pixels!")
        
        # Check feature variance across spatial dimensions
        enc_spatial_var = test_features.var(dim=(2, 3)).mean().item()
        print(f"\nEncoder output spatial variance: {enc_spatial_var:.6f}")
        
        if enc_spatial_var < 0.01:
            print("  ⚠️ CRITICAL: Encoder output has VERY LOW spatial variance!")
            print("     This is the ROOT CAUSE of centroid collapse!")
            print("     Possible causes:")
            print("     1. Too much pooling/normalization squashing spatial info")
            print("     2. Encoder not learning spatial features")
            print("     3. Empty/padded regions dominating")

def main():
    print("="*70)
    print("FOCUSED DSC COLLAPSE DIAGNOSTIC")
    print("="*70)
    
    # Find checkpoint
    checkpoint_path = "checkpoints/warmup3.pt"
    if not os.path.exists(checkpoint_path):
        checkpoint_path = "checkpoints/rlan_stable_ablation/warmup3.pt"
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found")
        return
    
    # Load model
    model, config = load_checkpoint_and_model(checkpoint_path)
    device = 'cpu'
    model = model.to(device)
    
    if not model.use_dsc:
        print("❌ Model doesn't use DSC")
        return
    
    # Find tasks
    task_dir = "data/arc-agi/data/training"
    if not os.path.exists(task_dir):
        task_dir = "../data/arc-agi/data/training"
    
    task_files = sorted([f for f in os.listdir(task_dir) if f.endswith('.json')])[:3]
    
    # Load tasks
    max_grid_size = getattr(config, 'max_grid_size', 22)
    tasks = []
    for tf in task_files:
        task = load_arc_task(os.path.join(task_dir, tf), max_grid_size)
        if task:
            tasks.append(task)
    
    print(f"\nLoaded {len(tasks)} tasks")
    
    if not tasks:
        print("❌ No tasks loaded")
        return
    
    # Prepare batch
    train_inputs = torch.stack([t['train_inputs'] for t in tasks])
    train_outputs = torch.stack([t['train_outputs'] for t in tasks])
    test_inputs = torch.stack([t['test_input'] for t in tasks])
    
    pair_mask = torch.zeros(len(tasks), 5)
    for i, t in enumerate(tasks):
        pair_mask[i, :t['num_pairs']] = 1.0
    
    print(f"\nBatch shapes:")
    print(f"  train_inputs: {train_inputs.shape}")
    print(f"  test_inputs: {test_inputs.shape}")
    
    # Diagnose encoder
    diagnose_encoder(model, train_inputs, train_outputs, test_inputs, pair_mask)
    
    # Get encoded features and run DSC directly
    print("\n" + "="*70)
    print("DSC ANALYSIS (DIRECT)")
    print("="*70)
    
    with torch.no_grad():
        train_inputs = train_inputs.to(device)
        train_outputs = train_outputs.to(device)
        test_inputs = test_inputs.to(device)
        pair_mask = pair_mask.to(device)
        
        # GridEncoder directly produces features
        B, H, W = test_inputs.shape
        test_features = model.encoder(test_inputs)  # (B, H, W, D) - GridEncoder output
        
        # Need to permute to (B, D, H, W) for DSC
        test_features_perm = test_features.permute(0, 3, 1, 2)  # (B, D, H, W)
        
        print(f"\nTest features shape: {test_features.shape}")
        print(f"  Permuted for DSC: {test_features_perm.shape}")
        print(f"  min={test_features.min().item():.4f}, max={test_features.max().item():.4f}")
        
        # Skip context encoder - run DSC analysis with task_context=None
        # This isolates the DSC behavior without task conditioning
        analyze_dsc_step_by_step(model, test_features_perm, task_context=None)
        
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    print("""
If feature cosine similarity is high (>0.9) across positions:
  -> Features are spatially uniform -> DSC has nothing to distinguish
  -> ROOT CAUSE: Encoder not preserving spatial structure
  
If attention score std < 0.1:
  -> Scores too similar -> Softmax produces uniform attention
  -> ROOT CAUSE: Query doesn't discriminate between positions
  
POTENTIAL FIXES:
1. Ensure encoder preserves spatial variance (check LayerNorm, pooling)
2. Use position encoding so even identical pixels have different features
3. Lower softmax temperature to sharpen attention
4. Use focal loss to encourage peaked attention
5. Train with attention entropy regularization
""")

if __name__ == "__main__":
    main()
