#!/usr/bin/env python3
"""
Test script to verify Rolling Cache vs Static Cache behavior.

This script tests the ACTUAL behavior of the data loading code, not just
what the logs claim. It verifies:

1. Rolling cache produces pool_size samples (not samples_per_task × num_tasks)
2. Static cache produces samples_per_task × num_tasks samples
3. Rolling cache refresh actually replaces 30% of samples each epoch
4. Augmentations are actually applied (not just claimed in logs)

Run:
    python tests/test_rolling_vs_static.py
"""

import sys
import os
import yaml
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sci_arc.data.dataset import ARCDataset
from sci_arc.data.rolling_cache import (
    RollingRefreshCache,
    RollingCacheDataset,
    create_rolling_cache_from_config,
)


def test_static_cache_sample_count():
    """Test that static cache produces exactly samples_per_task × num_tasks samples."""
    print("\n" + "="*60)
    print("TEST 1: Static Cache Sample Count")
    print("="*60)
    
    # Create minimal config
    data_path = str(project_root / "data" / "training")
    if not os.path.exists(data_path):
        print(f"  [SKIP] Training data not found at {data_path}")
        return True
    
    # Create dataset with static cache
    samples_per_task = 10
    max_tasks = 20  # Use subset for speed
    
    dataset = ARCDataset(
        data_path,
        max_size=30,
        augment=True,
        color_permutation=True,
        cache_samples=True,
        num_cached_samples=samples_per_task * max_tasks,
        max_tasks=max_tasks,
    )
    
    expected_samples = samples_per_task * max_tasks
    actual_samples = len(dataset)
    
    print(f"  samples_per_task: {samples_per_task}")
    print(f"  max_tasks: {max_tasks}")
    print(f"  Expected samples: {expected_samples}")
    print(f"  Actual samples: {actual_samples}")
    
    if actual_samples == expected_samples:
        print("  ✅ PASS: Static cache has correct sample count")
        return True
    else:
        print(f"  ❌ FAIL: Expected {expected_samples}, got {actual_samples}")
        return False


def test_rolling_cache_sample_count():
    """Test that rolling cache produces exactly pool_size samples."""
    print("\n" + "="*60)
    print("TEST 2: Rolling Cache Sample Count")
    print("="*60)
    
    data_path = str(project_root / "data" / "training")
    if not os.path.exists(data_path):
        print(f"  [SKIP] Training data not found at {data_path}")
        return True
    
    # Create base dataset first
    max_tasks = 20  # Use subset for speed
    base_dataset = ARCDataset(
        data_path,
        max_size=30,
        augment=True,
        color_permutation=True,
        max_tasks=max_tasks,
    )
    
    # Create rolling cache with specific pool_size
    pool_size = 500  # Use small pool for testing
    
    rolling_cache = RollingRefreshCache(
        tasks=base_dataset.tasks,
        generate_sample_fn=base_dataset._generate_sample,
        pool_size=pool_size,
        refresh_fraction=0.30,
        verbose=False,
    )
    
    actual_samples = len(rolling_cache)
    
    print(f"  pool_size (config): {pool_size}")
    print(f"  num_tasks: {len(base_dataset.tasks)}")
    print(f"  Actual samples in pool: {actual_samples}")
    
    if actual_samples == pool_size:
        print("  ✅ PASS: Rolling cache has correct pool_size samples")
        return True
    else:
        print(f"  ❌ FAIL: Expected {pool_size}, got {actual_samples}")
        return False


def test_rolling_cache_refresh():
    """Test that rolling cache actually refreshes 30% of samples."""
    print("\n" + "="*60)
    print("TEST 3: Rolling Cache Refresh (30%)")
    print("="*60)
    
    data_path = str(project_root / "data" / "training")
    if not os.path.exists(data_path):
        print(f"  [SKIP] Training data not found at {data_path}")
        return True
    
    # Create base dataset
    max_tasks = 10
    base_dataset = ARCDataset(
        data_path,
        max_size=30,
        augment=True,
        color_permutation=True,
        max_tasks=max_tasks,
    )
    
    pool_size = 100
    refresh_fraction = 0.30
    
    rolling_cache = RollingRefreshCache(
        tasks=base_dataset.tasks,
        generate_sample_fn=base_dataset._generate_sample,
        pool_size=pool_size,
        refresh_fraction=refresh_fraction,
        verbose=False,
    )
    
    # Get epoch 0 samples (sample unique identifiers - use test_input hash)
    def get_sample_fingerprints(cache):
        """Get fingerprints of all samples in cache."""
        fingerprints = set()
        for i in range(len(cache)):
            sample = cache[i]
            # Use hash of test_input tensor as fingerprint
            test_input = sample['test_input']
            if isinstance(test_input, torch.Tensor):
                fp = hash(test_input.numpy().tobytes())
            else:
                fp = hash(test_input.tobytes())
            fingerprints.add(fp)
        return fingerprints
    
    epoch0_fingerprints = get_sample_fingerprints(rolling_cache)
    
    # Trigger epoch swap (simulates end of epoch 0)
    rolling_cache.swap_epoch(1)
    
    epoch1_fingerprints = get_sample_fingerprints(rolling_cache)
    
    # Count how many samples changed
    unchanged = epoch0_fingerprints.intersection(epoch1_fingerprints)
    changed = len(epoch0_fingerprints) - len(unchanged)
    pct_changed = changed / len(epoch0_fingerprints) * 100
    
    print(f"  pool_size: {pool_size}")
    print(f"  refresh_fraction: {refresh_fraction * 100:.0f}%")
    print(f"  Epoch 0 unique samples: {len(epoch0_fingerprints)}")
    print(f"  Epoch 1 unique samples: {len(epoch1_fingerprints)}")
    print(f"  Samples unchanged: {len(unchanged)}")
    print(f"  Samples changed: {changed} ({pct_changed:.1f}%)")
    
    expected_changed_min = pool_size * refresh_fraction * 0.8  # 80% tolerance
    expected_changed_max = pool_size * refresh_fraction * 1.2  # 120% tolerance
    
    if expected_changed_min <= changed <= expected_changed_max:
        print(f"  ✅ PASS: ~{refresh_fraction*100:.0f}% samples refreshed")
        return True
    else:
        print(f"  ❌ FAIL: Expected {refresh_fraction*100:.0f}% refresh, got {pct_changed:.1f}%")
        return False


def test_augmentation_actually_applied():
    """Test that augmentations are actually applied to samples."""
    print("\n" + "="*60)
    print("TEST 4: Augmentations Actually Applied")
    print("="*60)
    
    data_path = str(project_root / "data" / "training")
    if not os.path.exists(data_path):
        print(f"  [SKIP] Training data not found at {data_path}")
        return True
    
    # Create dataset with augmentation enabled
    dataset = ARCDataset(
        data_path,
        max_size=30,
        augment=True,
        color_permutation=True,
        color_permutation_prob=1.0,  # 100% color permutation
        translational_augment=True,
        max_tasks=5,
    )
    
    # Generate multiple samples and check for diversity
    num_samples = 50
    dihedral_ids = []
    color_perms_applied = 0
    trans_offsets = []
    
    for i in range(num_samples):
        sample = dataset._generate_sample(0)  # Always from same task
        aug_info = sample.get('aug_info', {})
        
        dihedral_ids.append(aug_info.get('dihedral_id', 0))
        if aug_info.get('color_perm') is not None:
            color_perms_applied += 1
        offset = aug_info.get('translational_offset', (0, 0))
        trans_offsets.append(offset)
    
    unique_dihedrals = len(set(dihedral_ids))
    unique_offsets = len(set(trans_offsets))
    
    print(f"  Samples generated: {num_samples}")
    print(f"  Unique dihedral IDs: {unique_dihedrals}/8")
    print(f"  Color permutations applied: {color_perms_applied}/{num_samples}")
    print(f"  Unique translational offsets: {unique_offsets}")
    
    # Check diversity
    pass_dihedral = unique_dihedrals >= 6  # Should have most of 8 dihedrals
    pass_color = color_perms_applied >= num_samples * 0.8  # 80% should have color perm
    pass_trans = unique_offsets >= num_samples * 0.5  # 50% unique offsets
    
    all_pass = pass_dihedral and pass_color and pass_trans
    
    if pass_dihedral:
        print(f"  ✅ Dihedral: {unique_dihedrals}/8 unique transforms")
    else:
        print(f"  ❌ Dihedral: Only {unique_dihedrals}/8 - augmentation may be broken!")
    
    if pass_color:
        print(f"  ✅ Color perm: {color_perms_applied}/{num_samples} samples")
    else:
        print(f"  ❌ Color perm: Only {color_perms_applied}/{num_samples} - augmentation broken!")
    
    if pass_trans:
        print(f"  ✅ Translational: {unique_offsets} unique offsets")
    else:
        print(f"  ❌ Translational: Only {unique_offsets} - may be broken")
    
    return all_pass


def test_dataloader_sample_count():
    """Test that DataLoader exposes correct number of samples."""
    print("\n" + "="*60)
    print("TEST 5: DataLoader Sample Count Verification")
    print("="*60)
    
    data_path = str(project_root / "data" / "training")
    if not os.path.exists(data_path):
        print(f"  [SKIP] Training data not found at {data_path}")
        return True
    
    from torch.utils.data import DataLoader
    
    # Test static
    max_tasks = 10
    samples_per_task = 20
    
    static_dataset = ARCDataset(
        data_path,
        max_size=30,
        augment=True,
        cache_samples=True,
        num_cached_samples=samples_per_task * max_tasks,
        max_tasks=max_tasks,
    )
    
    batch_size = 10
    static_loader = DataLoader(static_dataset, batch_size=batch_size, shuffle=True)
    static_batches = len(static_loader)
    static_expected_batches = (samples_per_task * max_tasks + batch_size - 1) // batch_size
    
    print(f"  STATIC CACHE:")
    print(f"    Dataset size: {len(static_dataset)}")
    print(f"    Batch size: {batch_size}")
    print(f"    Expected batches: {static_expected_batches}")
    print(f"    Actual batches: {static_batches}")
    
    # Test rolling
    pool_size = 300
    rolling_cache = RollingRefreshCache(
        tasks=static_dataset.tasks,
        generate_sample_fn=static_dataset._generate_sample,
        pool_size=pool_size,
        verbose=False,
    )
    rolling_dataset = RollingCacheDataset(rolling_cache)
    rolling_loader = DataLoader(rolling_dataset, batch_size=batch_size, shuffle=True)
    rolling_batches = len(rolling_loader)
    rolling_expected_batches = (pool_size + batch_size - 1) // batch_size
    
    print(f"  ROLLING CACHE:")
    print(f"    Pool size: {pool_size}")
    print(f"    Dataset size: {len(rolling_dataset)}")
    print(f"    Batch size: {batch_size}")
    print(f"    Expected batches: {rolling_expected_batches}")
    print(f"    Actual batches: {rolling_batches}")
    
    static_pass = static_batches == static_expected_batches
    rolling_pass = rolling_batches == rolling_expected_batches
    
    if static_pass:
        print(f"  ✅ Static DataLoader batch count correct")
    else:
        print(f"  ❌ Static DataLoader batch count WRONG!")
    
    if rolling_pass:
        print(f"  ✅ Rolling DataLoader batch count correct")
    else:
        print(f"  ❌ Rolling DataLoader batch count WRONG!")
    
    return static_pass and rolling_pass


def test_yaml_config_respected():
    """Test that YAML config values are actually used."""
    print("\n" + "="*60)
    print("TEST 6: YAML Config Values Respected")
    print("="*60)
    
    data_path = str(project_root / "data" / "training")
    if not os.path.exists(data_path):
        print(f"  [SKIP] Training data not found at {data_path}")
        return True
    
    # Create config dict simulating YAML
    config = {
        'data': {
            'rolling_cache': {
                'pool_size': 123,  # Weird number to ensure it's used
                'refresh_fraction': 0.42,  # Weird number
                'anti_repeat_window': 7,
            }
        }
    }
    
    base_dataset = ARCDataset(
        data_path,
        max_size=30,
        max_tasks=5,
    )
    
    rolling_cache = create_rolling_cache_from_config(
        tasks=base_dataset.tasks,
        generate_sample_fn=base_dataset._generate_sample,
        config=config,
        verbose=False,
    )
    
    print(f"  Config pool_size: {config['data']['rolling_cache']['pool_size']}")
    print(f"  Actual pool_size: {rolling_cache.pool_size}")
    print(f"  Actual samples: {len(rolling_cache)}")
    
    config_respected = (
        rolling_cache.pool_size == 123 and
        rolling_cache.refresh_fraction == 0.42 and
        rolling_cache.anti_repeat_window == 7 and
        len(rolling_cache) == 123
    )
    
    if config_respected:
        print(f"  ✅ PASS: YAML config values are respected")
        return True
    else:
        print(f"  ❌ FAIL: YAML config values NOT respected!")
        print(f"    pool_size: {rolling_cache.pool_size} (expected 123)")
        print(f"    refresh_fraction: {rolling_cache.refresh_fraction} (expected 0.42)")
        print(f"    anti_repeat_window: {rolling_cache.anti_repeat_window} (expected 7)")
        return False


def main():
    """Run all tests."""
    print("\n" + "#"*60)
    print("# ROLLING VS STATIC CACHE VERIFICATION")
    print("#"*60)
    
    results = []
    
    results.append(("Static Cache Sample Count", test_static_cache_sample_count()))
    results.append(("Rolling Cache Sample Count", test_rolling_cache_sample_count()))
    results.append(("Rolling Cache Refresh", test_rolling_cache_refresh()))
    results.append(("Augmentations Applied", test_augmentation_actually_applied()))
    results.append(("DataLoader Sample Count", test_dataloader_sample_count()))
    results.append(("YAML Config Respected", test_yaml_config_respected()))
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    passed = 0
    failed = 0
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {name}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\n  Total: {passed}/{len(results)} passed")
    
    if failed > 0:
        print("\n  ⚠️  SOME TESTS FAILED - CHECK DATA LOADING CODE!")
        return 1
    else:
        print("\n  ✅ ALL TESTS PASSED")
        return 0


if __name__ == "__main__":
    sys.exit(main())
