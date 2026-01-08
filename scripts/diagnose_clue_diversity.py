"""
Quick diagnostic to check DSC centroid diversity and MSRE clue differentiation.
This tests the root cause of MSRE channel correlation = 1.0.
"""

import sys
import json
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sci_arc.models.rlan import RLAN, RLANConfig


def main():
    print("=" * 70)
    print("DSC CENTROID & MSRE CLUE DIVERSITY ANALYSIS")
    print("=" * 70)
    
    # Load model
    ckpt_path = project_root / "checkpoints" / "warmup3.pt"
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    ckpt_config = checkpoint.get('config', {})
    
    config = RLANConfig(
        hidden_dim=ckpt_config.get('hidden_dim', 256),
        max_clues=ckpt_config.get('max_clues', 7),
        num_solver_steps=ckpt_config.get('num_solver_steps', 6),
        use_dsc=ckpt_config.get('use_dsc', True),
        use_msre=ckpt_config.get('use_msre', True),
        use_context_encoder=ckpt_config.get('use_context_encoder', True),
        use_hyperlora=ckpt_config.get('use_hyperlora', False),
        use_hpm=ckpt_config.get('use_hpm', False),
        use_lcr=ckpt_config.get('use_lcr', False),
        use_sph=ckpt_config.get('use_sph', False),
        dsc_use_complexity_signals=ckpt_config.get('dsc_use_complexity_signals', False),
    )
    
    model = RLAN(config=config)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    
    # Load a test task
    arc_dir = project_root / "data" / "arc-agi" / "data" / "training"
    with open(arc_dir / "007bbfb7.json") as f:
        task = json.load(f)
    
    # Prepare input
    def pad_grid(grid, size=30):
        grid = np.array(grid)
        h, w = grid.shape
        padded = np.zeros((size, size), dtype=np.int64)
        padded[:h, :w] = grid
        return padded
    
    train = task["train"]
    test = task["test"][0]
    
    support_in = torch.tensor(np.stack([pad_grid(ex["input"]) for ex in train] + 
                                       [np.zeros((30, 30))] * (5 - len(train)))).unsqueeze(0)
    support_out = torch.tensor(np.stack([pad_grid(ex["output"]) for ex in train] + 
                                        [np.zeros((30, 30))] * (5 - len(train)))).unsqueeze(0)
    query = torch.tensor(pad_grid(test["input"])).unsqueeze(0)
    
    print(f"\nTest task: 007bbfb7")
    print(f"Input shape: {np.array(test['input']).shape}")
    print(f"Output shape: {np.array(test['output']).shape}")
    
    # Run forward pass with intermediates
    with torch.no_grad():
        outputs = model(query, train_inputs=support_in, train_outputs=support_out, 
                       return_intermediates=True)
    
    # Analyze centroids
    centroids = outputs['centroids'][0].numpy()  # (K, 2)
    print(f"\n--- DSC CENTROIDS ---")
    print(f"Shape: {centroids.shape}")
    print(f"Centroids (row, col in [0,1]):")
    for i, c in enumerate(centroids):
        print(f"  Clue {i}: ({c[0]:.4f}, {c[1]:.4f})")
    
    # Compute centroid diversity
    centroid_dists = []
    for i in range(len(centroids)):
        for j in range(i+1, len(centroids)):
            dist = np.sqrt(np.sum((centroids[i] - centroids[j])**2))
            centroid_dists.append(dist)
    
    print(f"\nCentroid pairwise distances:")
    print(f"  Min: {min(centroid_dists):.4f}")
    print(f"  Max: {max(centroid_dists):.4f}")
    print(f"  Mean: {np.mean(centroid_dists):.4f}")
    
    if max(centroid_dists) < 0.1:
        print(f"  ⚠️ WARNING: All centroids are very close together!")
        print(f"     This means DSC is not discovering diverse spatial anchors.")
    
    # Analyze attention maps
    attn = outputs['attention_maps'][0].numpy()  # (K, H, W)
    print(f"\n--- DSC ATTENTION MAPS ---")
    print(f"Shape: {attn.shape}")
    
    attn_correlations = []
    for i in range(attn.shape[0]):
        for j in range(i+1, attn.shape[0]):
            corr = np.corrcoef(attn[i].flatten(), attn[j].flatten())[0, 1]
            attn_correlations.append(corr)
    
    print(f"Attention map pairwise correlations:")
    print(f"  Min: {min(attn_correlations):.4f}")
    print(f"  Max: {max(attn_correlations):.4f}")
    print(f"  Mean: {np.mean(attn_correlations):.4f}")
    
    if np.mean(attn_correlations) > 0.9:
        print(f"  ⚠️ WARNING: Attention maps are nearly identical!")
        print(f"     DSC slots are not differentiating spatial regions.")
    
    # Analyze MSRE clue features directly
    # We need to manually run MSRE to get the clue features
    features = model.encode(query)
    valid_mask = model.encoder.get_valid_mask(query)
    
    dsc_out = model.dsc(features, mask=valid_mask)
    
    # DSC returns tuple: (centroids, attention_maps, stop_logits) 
    # or dict depending on version
    if isinstance(dsc_out, dict):
        dsc_centroids = dsc_out['centroids']
    else:
        dsc_centroids = dsc_out[0]  # First element is centroids
    
    clue_features = model.msre(features, dsc_centroids)
    
    clue_feat = clue_features[0].detach().numpy()  # (K, D, H, W)
    print(f"\n--- MSRE CLUE FEATURES ---")
    print(f"Shape: {clue_feat.shape}")
    
    # Compute per-clue statistics
    print(f"\nPer-clue statistics (pooled over spatial dims):")
    clue_vectors = []
    for i in range(clue_feat.shape[0]):
        feat_i = clue_feat[i]  # (D, H, W)
        pooled = feat_i.mean(axis=(1, 2))  # (D,) global average pool
        clue_vectors.append(pooled)
        print(f"  Clue {i}: mean={pooled.mean():.4f}, std={pooled.std():.4f}")
    
    # Compute clue feature correlations
    clue_correlations = []
    for i in range(len(clue_vectors)):
        for j in range(i+1, len(clue_vectors)):
            corr = np.corrcoef(clue_vectors[i], clue_vectors[j])[0, 1]
            clue_correlations.append(corr)
    
    print(f"\nClue feature pairwise correlations (after MSRE):")
    print(f"  Min: {min(clue_correlations):.4f}")
    print(f"  Max: {max(clue_correlations):.4f}")
    print(f"  Mean: {np.mean(clue_correlations):.4f}")
    
    if np.mean(clue_correlations) > 0.95:
        print(f"\n  🔴 CRITICAL BUG CONFIRMED: Clue features are nearly identical!")
        print(f"     Root cause analysis:")
        print(f"     1. MSRE takes input features and adds relative coordinate encoding")
        print(f"     2. If centroids are too similar, relative coordinates are too similar")
        print(f"     3. Resulting clue features become identical")
        print(f"\n  Proposed fixes:")
        print(f"     A. Enforce centroid diversity during DSC training (repulsion loss)")
        print(f"     B. Use learned per-clue embeddings in MSRE (not just coordinates)")
        print(f"     C. Apply dropout or noise to break symmetry")
    elif np.mean(clue_correlations) > 0.8:
        print(f"\n  ⚠️ WARNING: Clue features have high correlation")
        print(f"     This may limit the model's reasoning capacity")
    else:
        print(f"\n  ✅ Clue features appear diverse")
    
    # Compute effective rank of clue features
    clue_matrix = np.stack(clue_vectors)  # (K, D)
    U, S, Vh = np.linalg.svd(clue_matrix, full_matrices=False)
    effective_rank = (S > S[0] * 0.01).sum()  # Count singular values > 1% of max
    
    print(f"\n--- EFFECTIVE RANK ANALYSIS ---")
    print(f"Singular values: {S[:5]}")
    print(f"Effective rank (threshold=1% of max): {effective_rank}/{len(S)}")
    
    if effective_rank <= 2:
        print(f"  🔴 VERY LOW RANK: Only {effective_rank} independent dimensions")
        print(f"     All clue features span a 1-2D subspace, not diverse at all!")
    
    print("\n" + "=" * 70)
    print("DIAGNOSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
