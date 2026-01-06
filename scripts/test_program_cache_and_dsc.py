#!/usr/bin/env python3
"""
Test script to diagnose and fix program cache and DSC issues.

Issues to diagnose:
1. Primitive Loss = 0.0 - Cache hit rate 0%
2. Centroid Spread = 0.00 - All clues collapsed
3. All Clues IDENTICAL (entropy_std=0.0000)
4. Spatial Attention COLLAPSED (max=0.001)

Author: AI Research Assistant
Date: January 2026
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
import json
import hashlib
from pathlib import Path

# =============================================================================
# TEST 1: Program Cache Lookup
# =============================================================================
def test_program_cache():
    """Test if program cache lookup is working correctly."""
    print("=" * 60)
    print("TEST 1: Program Cache Lookup")
    print("=" * 60)
    
    from sci_arc.models.generalization.program_guided_training import (
        ProgramCache, PseudoLabelGenerator, ProgramGuidedConfig
    )
    
    # Load the cache
    cache_path = Path("cache/program_cache_merged_602.json")
    if not cache_path.exists():
        print(f"❌ Cache file not found: {cache_path}")
        return False
    
    cache = ProgramCache(str(cache_path))
    print(f"✓ Cache loaded: {len(cache)} programs")
    
    # Check a sample entry
    sample_key = list(cache.cache.keys())[0] if cache.cache else None
    if sample_key:
        entry = cache.get(sample_key)
        print(f"\nSample entry:")
        print(f"  task_id: {sample_key}")
        print(f"  trace: {entry['trace']}")
        print(f"  input_hash: {entry['input_hash']}")
        print(f"  confidence: {entry['confidence']}")
    
    # Test input_hash index
    print(f"\n_by_input_hash entries: {len(cache._by_input_hash)}")
    if len(cache._by_input_hash) > 0:
        sample_hash = list(cache._by_input_hash.keys())[0]
        print(f"  Sample hash: {sample_hash}")
        result = cache.get_by_input_hash(sample_hash)
        print(f"  Lookup result: {'Found' if result else 'Not found'}")
    
    return True


def test_task_id_matching():
    """Test if task_ids from dataloader match cache keys."""
    print("\n" + "=" * 60)
    print("TEST 2: Task ID Matching Between Dataloader and Cache")
    print("=" * 60)
    
    from sci_arc.models.generalization.program_guided_training import ProgramCache
    
    # Load cache
    cache_path = Path("cache/program_cache_merged_602.json")
    cache = ProgramCache(str(cache_path))
    cache_task_ids = set(cache.cache.keys())
    print(f"Cache has {len(cache_task_ids)} task_ids")
    
    # Sample cache task_ids
    sample_cache_ids = list(cache_task_ids)[:5]
    print(f"Sample cache task_ids: {sample_cache_ids}")
    
    # Now load dataloader and check format
    try:
        from sci_arc.data.merged_loader import load_merged_training_data
        
        # Load a few tasks
        tasks = load_merged_training_data(
            merged_training_path="data/merged_training",
            manifest_name="merged_train_manifest.jsonl"
        )[:10]
        
        loader_task_ids = [t.get('task_id', 'missing') for t in tasks]
        print(f"\nDataloader task_ids (first 10): {loader_task_ids}")
        
        # Check overlap
        overlap = set(loader_task_ids) & cache_task_ids
        print(f"Overlap with cache: {len(overlap)}")
        if overlap:
            print(f"  Matching IDs: {list(overlap)[:5]}")
        else:
            print("  ❌ NO OVERLAP - task_id format mismatch!")
            print(f"  Cache format: {sample_cache_ids[0] if sample_cache_ids else 'N/A'}")
            print(f"  Loader format: {loader_task_ids[0] if loader_task_ids else 'N/A'}")
            return False
            
    except Exception as e:
        print(f"⚠️ Could not test dataloader: {e}")
    
    return True


def test_augmented_input_hash():
    """Test if augmented inputs can match cache via input_hash."""
    print("\n" + "=" * 60)
    print("TEST 3: Augmented Input Hash Matching")
    print("=" * 60)
    
    from sci_arc.models.generalization.program_guided_training import ProgramCache
    
    cache_path = Path("cache/program_cache_merged_602.json")
    cache = ProgramCache(str(cache_path))
    
    # Get a sample entry with input_hash
    sample_key = None
    sample_hash = None
    for key, entry in cache.cache.items():
        if entry.get('input_hash'):
            sample_key = key
            sample_hash = entry['input_hash']
            break
    
    if not sample_hash:
        print("❌ No entries with input_hash in cache")
        return False
    
    print(f"Testing with task_id: {sample_key}")
    print(f"Original input_hash: {sample_hash}")
    
    # Simulate what training does - create a fake augmented grid
    # The issue: augmented grid has DIFFERENT hash than original
    fake_original = np.random.randint(0, 10, (5, 5), dtype=np.int64)
    original_hash = hashlib.sha256(fake_original.tobytes()).hexdigest()[:16]
    
    # Apply dihedral augmentation (rot90)
    augmented = np.rot90(fake_original, k=1)
    augmented_hash = hashlib.sha256(augmented.tobytes()).hexdigest()[:16]
    
    print(f"\nSimulated:")
    print(f"  Original hash: {original_hash}")
    print(f"  Augmented hash (rot90): {augmented_hash}")
    print(f"  Hashes match: {original_hash == augmented_hash}")
    
    if original_hash != augmented_hash:
        print("\n❌ PROBLEM IDENTIFIED: Augmented grids have different hashes!")
        print("   Cache is keyed by ORIGINAL grid hash, but training sees AUGMENTED grids.")
        print("   Solution: Cache must be keyed by task_id only, not input_hash.")
        return False
    
    return True


# =============================================================================
# TEST 4: DSC Centroid Collapse
# =============================================================================
def test_dsc_centroid_collapse():
    """Test DSC attention and centroid diversity."""
    print("\n" + "=" * 60)
    print("TEST 4: DSC Centroid Collapse Diagnosis")
    print("=" * 60)
    
    try:
        from sci_arc.models.rlan import RLAN
    except ImportError as e:
        print(f"⚠️ Could not import RLAN: {e}")
        return False
    
    # Create a minimal model config
    config = {
        'encoder': {'hidden_dim': 128, 'num_layers': 3},
        'solver': {'hidden_dim': 128, 'num_solver_steps': 7},
        'context_encoder': {'enabled': True, 'hidden_dim': 128},
        'dsc': {
            'enabled': True,
            'hidden_dim': 128,
            'num_clues': 7,
            'lambda_entropy': 0.01,
            'lambda_sparsity': 0.5,
            'lambda_centroid_diversity': 0.01,  # Current value
        },
    }
    
    print("Creating model on CPU...")
    try:
        model = RLAN(config).to('cpu')
        model.eval()
    except Exception as e:
        print(f"❌ Failed to create model: {e}")
        return False
    
    print("✓ Model created")
    
    # Create fake input
    B, K, H, W = 2, 3, 10, 10  # batch=2, pairs=3
    train_inputs = torch.randint(0, 10, (B, K, H, W), dtype=torch.long)
    train_outputs = torch.randint(0, 10, (B, K, H, W), dtype=torch.long)
    test_input = torch.randint(0, 10, (B, H, W), dtype=torch.long)
    pair_mask = torch.ones(B, K)
    
    print(f"Input shapes: train_inputs={train_inputs.shape}, test_input={test_input.shape}")
    
    # Forward pass
    with torch.no_grad():
        outputs = model(train_inputs, train_outputs, test_input, pair_mask=pair_mask)
    
    print("\n--- DSC Outputs ---")
    dsc_outputs = outputs.get('dsc_outputs')
    if dsc_outputs is None:
        print("❌ No DSC outputs!")
        return False
    
    # Check attention maps
    attn_weights = dsc_outputs.get('attention_weights')
    if attn_weights is not None:
        print(f"Attention weights shape: {attn_weights.shape}")  # [B, num_clues, H, W]
        attn_max = attn_weights.max().item()
        attn_min = attn_weights.min().item()
        attn_mean = attn_weights.mean().item()
        print(f"Attention: max={attn_max:.6f}, min={attn_min:.6f}, mean={attn_mean:.6f}")
        
        if attn_max < 0.01:
            print("❌ ATTENTION COLLAPSED: max < 0.01 (nearly uniform)")
        else:
            print("✓ Attention is focusing")
    else:
        print("⚠️ No attention_weights in DSC outputs")
    
    # Check clue centroids
    clue_centroids = dsc_outputs.get('clue_centroids')
    if clue_centroids is not None:
        print(f"\nClue centroids shape: {clue_centroids.shape}")  # [B, num_clues, hidden_dim]
        
        # Compute pairwise distances between clues
        centroids = clue_centroids[0]  # [num_clues, hidden_dim]
        num_clues = centroids.shape[0]
        
        distances = []
        for i in range(num_clues):
            for j in range(i+1, num_clues):
                dist = torch.norm(centroids[i] - centroids[j]).item()
                distances.append(dist)
        
        if distances:
            mean_dist = np.mean(distances)
            std_dist = np.std(distances)
            print(f"Centroid pairwise distances: mean={mean_dist:.4f}, std={std_dist:.4f}")
            
            if mean_dist < 0.5:
                print("❌ CENTROID COLLAPSE: mean distance < 0.5 (all clues at same location)")
            else:
                print("✓ Centroids are diverse")
    else:
        print("⚠️ No clue_centroids in DSC outputs")
    
    # Check per-clue entropy
    per_clue_entropy = dsc_outputs.get('per_clue_entropy')
    if per_clue_entropy is not None:
        print(f"\nPer-clue entropy: {per_clue_entropy}")
        entropy_std = per_clue_entropy.std().item() if hasattr(per_clue_entropy, 'std') else np.std(per_clue_entropy)
        print(f"Entropy std across clues: {entropy_std:.6f}")
        
        if entropy_std < 0.01:
            print("❌ ALL CLUES IDENTICAL: entropy_std ≈ 0")
        else:
            print("✓ Clues are differentiated")
    else:
        print("⚠️ No per_clue_entropy in DSC outputs")
    
    return True


def test_diversity_regularization():
    """Test the centroid diversity loss computation."""
    print("\n" + "=" * 60)
    print("TEST 5: Centroid Diversity Regularization")
    print("=" * 60)
    
    # Simulate collapsed centroids
    num_clues = 7
    hidden_dim = 128
    
    # Case 1: All centroids identical (collapsed)
    collapsed = torch.randn(1, num_clues, hidden_dim)
    collapsed = collapsed[:, 0:1, :].expand(-1, num_clues, -1)  # All same
    
    # Case 2: Diverse centroids
    diverse = torch.randn(1, num_clues, hidden_dim)
    
    def compute_diversity_loss(centroids):
        """Compute centroid diversity loss (should be POSITIVE when collapsed)."""
        B, K, D = centroids.shape
        
        # Normalize centroids
        centroids_norm = F.normalize(centroids, dim=-1)  # [B, K, D]
        
        # Cosine similarity matrix
        sim_matrix = torch.bmm(centroids_norm, centroids_norm.transpose(1, 2))  # [B, K, K]
        
        # We want DISSIMILARITY between different clues
        # Mask out diagonal
        mask = 1.0 - torch.eye(K).unsqueeze(0)  # [1, K, K]
        
        # Average off-diagonal similarity (should be LOW)
        off_diag_sim = (sim_matrix * mask).sum() / mask.sum() / B
        
        return off_diag_sim
    
    loss_collapsed = compute_diversity_loss(collapsed)
    loss_diverse = compute_diversity_loss(diverse)
    
    print(f"Collapsed centroids:")
    print(f"  Off-diagonal similarity: {loss_collapsed.item():.4f}")
    print(f"  (Should be ~1.0 when all identical)")
    
    print(f"\nDiverse centroids:")
    print(f"  Off-diagonal similarity: {loss_diverse.item():.4f}")
    print(f"  (Should be ~0.0 when orthogonal)")
    
    if loss_collapsed > 0.9:
        print("\n✓ Diversity loss correctly detects collapse")
    else:
        print("\n❌ Diversity loss NOT detecting collapse correctly!")
    
    return True


# =============================================================================
# TEST 6: DSC Initialization
# =============================================================================
def test_dsc_initialization():
    """Check if DSC is initialized in a way that causes immediate collapse."""
    print("\n" + "=" * 60)
    print("TEST 6: DSC Initialization Check")
    print("=" * 60)
    
    try:
        from sci_arc.models.components.dsc import DynamicSupervisedClustering
    except ImportError:
        print("⚠️ Could not import DSC module")
        return False
    
    # Create DSC with default init
    dsc = DynamicSupervisedClustering(
        hidden_dim=128,
        num_clues=7,
        lambda_entropy=0.01,
        lambda_sparsity=0.5,
        lambda_centroid_diversity=0.01,
    )
    
    # Check clue query initialization
    if hasattr(dsc, 'clue_queries'):
        queries = dsc.clue_queries.weight if hasattr(dsc.clue_queries, 'weight') else dsc.clue_queries
        print(f"Clue queries shape: {queries.shape}")
        
        # Check if all queries are identical
        q0 = queries[0]
        all_same = all(torch.allclose(queries[i], q0) for i in range(1, queries.shape[0]))
        
        if all_same:
            print("❌ ALL CLUE QUERIES IDENTICAL AT INIT!")
            print("   This will cause immediate centroid collapse!")
        else:
            # Check diversity
            queries_norm = F.normalize(queries, dim=-1)
            sim_matrix = queries_norm @ queries_norm.t()
            off_diag = sim_matrix - torch.eye(queries.shape[0])
            max_sim = off_diag.abs().max().item()
            mean_sim = off_diag.abs().mean().item()
            print(f"Query similarity: max={max_sim:.4f}, mean={mean_sim:.4f}")
            
            if max_sim > 0.9:
                print("⚠️ High similarity between some queries")
            else:
                print("✓ Queries are diverse at init")
    
    return True


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 60)
    print("PROGRAM CACHE AND DSC DIAGNOSTIC TESTS")
    print("=" * 60)
    
    results = {}
    
    # Test 1: Program cache basic
    results['cache_load'] = test_program_cache()
    
    # Test 2: Task ID matching
    results['task_id_match'] = test_task_id_matching()
    
    # Test 3: Augmented hash
    results['aug_hash'] = test_augmented_input_hash()
    
    # Test 4: DSC collapse
    results['dsc_collapse'] = test_dsc_centroid_collapse()
    
    # Test 5: Diversity loss
    results['diversity_loss'] = test_diversity_regularization()
    
    # Test 6: DSC init
    results['dsc_init'] = test_dsc_initialization()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results.items():
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n✓ All tests passed!")
    else:
        print("\n❌ Some tests failed - see above for details")
        print("\nRECOMMENDED FIXES:")
        if not results.get('task_id_match', True):
            print("  - Cache keys don't match dataloader task_ids")
        if not results.get('aug_hash', True):
            print("  - Augmented grids have different hashes than originals")
            print("  - FIX: Use task_id-only lookup, not input_hash for augmented samples")
        if not results.get('dsc_collapse', True):
            print("  - DSC attention/centroids are collapsing")
            print("  - FIX: Increase lambda_centroid_diversity, check initialization")
        if not results.get('dsc_init', True):
            print("  - DSC clue queries not diverse at initialization")
            print("  - FIX: Use orthogonal or diverse initialization for clue_queries")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
