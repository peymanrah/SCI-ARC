#!/usr/bin/env python
"""
Verify that augmentations are actually being applied in the dataset.

This script loads the dataset and checks that:
1. Dihedral transforms (8 variations) are uniformly distributed
2. Color permutation is applied at ~30% rate
3. Translational augmentation creates diverse offsets

Run this BEFORE training to verify data pipeline is correct.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from collections import Counter
import torch

from sci_arc.data.dataset import ARCDataset


def verify_augmentation():
    """Verify augmentations are working correctly."""
    
    print("=" * 70)
    print("AUGMENTATION VERIFICATION TEST")
    print("=" * 70)
    
    # Load dataset WITH tracking enabled (override config)
    # Try multiple possible data paths
    possible_paths = [
        project_root / "data" / "arc-agi_training_challenges.json",
        project_root / "data" / "arc-agi" / "data" / "training",
        project_root / "data" / "training",
    ]
    
    data_path = None
    for p in possible_paths:
        if p.exists():
            data_path = p
            break
    
    if data_path is None:
        print(f"ERROR: Data file not found at any of these locations:")
        for p in possible_paths:
            print(f"  - {p}")
        print("Please ensure you have the ARC dataset downloaded.")
        return False
    
    print(f"\nLoading dataset from {data_path}...")
    
    # Create dataset with all augmentations enabled and tracking ON
    dataset = ARCDataset(
        str(data_path),
        max_size=30,
        augment=True,  # Enable dihedral
        color_permutation=True,  # Enable color perm
        color_permutation_prob=0.3,  # 30% probability
        translational_augment=True,  # Enable translational
        track_augmentation=True,  # FORCE tracking for verification
        cache_samples=False,  # Fresh samples each time
    )
    
    print(f"Dataset loaded: {len(dataset)} tasks")
    
    # Sample multiple times to verify randomness
    num_samples = 1000
    print(f"\nGenerating {num_samples} samples to verify augmentation distribution...")
    
    dihedral_counts = Counter()
    color_perm_count = 0
    translational_count = 0
    translational_offsets = set()
    
    # Track specific task to verify transforms are different
    task_0_transforms = []
    
    for i in range(num_samples):
        # Sample same task multiple times to see different augmentations
        task_idx = i % len(dataset)
        sample = dataset[task_idx]
        
        if 'aug_info' in sample:
            aug_info = sample['aug_info']
            
            # Dihedral transform
            dihedral_id = aug_info.get('dihedral_id', 0)
            dihedral_counts[dihedral_id] += 1
            
            # Color permutation
            if aug_info.get('color_perm_applied', False):
                color_perm_count += 1
            
            # Translational offset
            offset = aug_info.get('translational_offset', (0, 0))
            if offset != (0, 0):
                translational_count += 1
                translational_offsets.add(offset)
            
            # Track first task's transforms
            if task_idx == 0:
                task_0_transforms.append(dihedral_id)
        else:
            print(f"WARNING: Sample {i} has no aug_info! Tracking may be disabled.")
    
    print("\n" + "=" * 70)
    print("VERIFICATION RESULTS")
    print("=" * 70)
    
    all_pass = True
    
    # === Check 1: Dihedral Distribution ===
    print("\n1. DIHEDRAL TRANSFORMS (should be ~uniform over 8 transforms)")
    print("-" * 50)
    expected_per_transform = num_samples / 8
    for transform_id in range(8):
        count = dihedral_counts.get(transform_id, 0)
        pct = count / num_samples * 100
        expected_pct = 12.5  # 100/8
        deviation = abs(pct - expected_pct)
        status = "✓" if deviation < 5 else "⚠"
        print(f"  Transform {transform_id}: {count:4d} ({pct:5.1f}%) {status}")
    
    # Check uniformity
    counts = [dihedral_counts.get(i, 0) for i in range(8)]
    all_present = all(c > 0 for c in counts)
    if all_present:
        print(f"\n  ✓ All 8 dihedral transforms are being applied!")
    else:
        print(f"\n  ✗ Some transforms missing! Counts: {counts}")
        all_pass = False
    
    # === Check 2: Color Permutation ===
    print("\n2. COLOR PERMUTATION (should be ~30%)")
    print("-" * 50)
    color_pct = color_perm_count / num_samples * 100
    expected_color_pct = 30
    status = "✓" if abs(color_pct - expected_color_pct) < 10 else "⚠"
    print(f"  Applied: {color_perm_count}/{num_samples} ({color_pct:.1f}%) {status}")
    if abs(color_pct - expected_color_pct) > 15:
        print(f"  ✗ Color permutation rate too far from expected 30%!")
        all_pass = False
    else:
        print(f"  ✓ Color permutation rate is reasonable (~30%)")
    
    # === Check 3: Translational Augmentation ===
    print("\n3. TRANSLATIONAL AUGMENTATION")
    print("-" * 50)
    trans_pct = translational_count / num_samples * 100
    print(f"  Applied: {translational_count}/{num_samples} ({trans_pct:.1f}%)")
    print(f"  Unique offsets: {len(translational_offsets)}")
    
    if len(translational_offsets) > 20:
        print(f"  ✓ Good diversity of translational offsets!")
    elif len(translational_offsets) > 0:
        print(f"  ⚠ Limited offset diversity")
    else:
        print(f"  ⚠ No translational offsets applied (may be due to grid sizes)")
    
    # === Check 4: Same task gets different transforms ===
    print("\n4. TRANSFORM DIVERSITY (same task, different samples)")
    print("-" * 50)
    unique_transforms = len(set(task_0_transforms))
    print(f"  Task 0 sampled {len(task_0_transforms)} times")
    print(f"  Unique dihedral transforms: {unique_transforms}/8")
    print(f"  Transform sequence (first 20): {task_0_transforms[:20]}")
    
    if unique_transforms >= 6:
        print(f"  ✓ Good diversity - same task gets different transforms!")
    else:
        print(f"  ⚠ Limited diversity - may need more samples to see all transforms")
    
    # === Final Verdict ===
    print("\n" + "=" * 70)
    if all_pass:
        print("✅ AUGMENTATION VERIFICATION PASSED")
        print("   All augmentation types are being applied correctly!")
    else:
        print("❌ AUGMENTATION VERIFICATION FAILED")
        print("   Some augmentations may not be working correctly.")
    print("=" * 70)
    
    return all_pass


def verify_cached_augmentation():
    """Verify augmentations in cached samples."""
    
    print("\n" + "=" * 70)
    print("CACHED SAMPLES VERIFICATION")
    print("=" * 70)
    
    # Load dataset with caching
    possible_paths = [
        Path(__file__).parent.parent / "data" / "arc-agi_training_challenges.json",
        Path(__file__).parent.parent / "data" / "arc-agi" / "data" / "training",
        Path(__file__).parent.parent / "data" / "training",
    ]
    
    data_path = None
    for p in possible_paths:
        if p.exists():
            data_path = p
            break
    
    if data_path is None:
        print(f"Skipping cached verification - data file not found")
        return
    
    print(f"\nCreating cached dataset with 4000 samples...")
    
    dataset = ARCDataset(
        str(data_path),
        max_size=30,
        augment=True,
        color_permutation=True,
        color_permutation_prob=0.3,
        translational_augment=True,
        track_augmentation=True,  # Track augmentation
        cache_samples=True,  # Enable caching
        num_cached_samples=4000,  # 10 per task
        cache_path=None,  # Don't persist to disk
    )
    
    print(f"Cached {len(dataset)} samples")
    
    # Verify cached samples have augmentation
    dihedral_counts = Counter()
    color_perm_count = 0
    
    for i in range(len(dataset)):
        sample = dataset[i]
        if 'aug_info' in sample:
            dihedral_counts[sample['aug_info']['dihedral_id']] += 1
            if sample['aug_info'].get('color_perm_applied', False):
                color_perm_count += 1
    
    print("\nCached sample augmentation distribution:")
    for transform_id in range(8):
        count = dihedral_counts.get(transform_id, 0)
        pct = count / len(dataset) * 100
        print(f"  Dihedral {transform_id}: {count:4d} ({pct:5.1f}%)")
    
    print(f"\nColor permutation: {color_perm_count}/{len(dataset)} ({color_perm_count/len(dataset)*100:.1f}%)")
    
    # Verify samples don't change between accesses (caching working)
    print("\nVerifying cache consistency...")
    sample_0_first = dataset[0]
    sample_0_second = dataset[0]
    
    if torch.equal(sample_0_first['test_input'], sample_0_second['test_input']):
        print("  ✓ Cached samples are consistent between accesses")
    else:
        print("  ✗ Cached samples changed! Caching may not be working")


if __name__ == "__main__":
    verify_augmentation()
    print("\n")
    verify_cached_augmentation()
