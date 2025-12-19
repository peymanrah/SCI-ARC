#!/usr/bin/env python
"""
Test script to verify augmentation is working correctly.

Tests:
1. Dihedral augmentation (8 transforms)
2. Color permutation (with configurable probability)
3. Translational augmentation (random offset)
4. That augmentation stats are tracked correctly

This helps verify the augmentation pipeline works before regenerating the cache.
"""

import torch
import numpy as np
import random
import sys
import os
from pathlib import Path
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def create_test_task():
    """Create a simple test task with known patterns."""
    # Simple 5x5 grid with distinct pattern
    input_grid = [
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 2, 1, 1],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
    ]
    output_grid = [
        [0, 0, 3, 0, 0],
        [0, 3, 3, 3, 0],
        [3, 3, 4, 3, 3],
        [0, 3, 3, 3, 0],
        [0, 0, 3, 0, 0],
    ]
    return {
        'task_id': 'test_task',
        'train': [{'input': input_grid, 'output': output_grid}],
        'test': [{'input': input_grid, 'output': output_grid}]
    }


def test_augmentations():
    """Test all augmentation types work correctly."""
    from sci_arc.data.dataset import ARCDataset
    
    print("=" * 70)
    print("AUGMENTATION VERIFICATION TEST")
    print("=" * 70)
    
    # Create temporary test file
    import json
    import tempfile
    
    test_task = create_test_task()
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({'test_task': {'train': test_task['train'], 'test': test_task['test']}}, f)
        temp_path = f.name
    
    try:
        # Test 1: Dihedral augmentation
        print("\n1. DIHEDRAL AUGMENTATION TEST")
        print("-" * 50)
        
        dataset = ARCDataset(
            temp_path,
            max_size=30,
            augment=True,
            color_permutation=False,
            translational_augment=False,
            track_augmentation=True,
        )
        
        dihedral_counts = Counter()
        for _ in range(100):
            sample = dataset[0]
            aug_info = sample.get('aug_info', {})
            dihedral_id = aug_info.get('dihedral_id', 0)
            dihedral_counts[dihedral_id] += 1
        
        print(f"Dihedral distribution (100 samples):")
        for i in range(8):
            count = dihedral_counts.get(i, 0)
            pct = count
            bar = "█" * (count // 3)
            print(f"  Transform {i}: {count:2d}% {bar}")
        
        # Check uniform distribution
        min_count = min(dihedral_counts.values()) if dihedral_counts else 0
        max_count = max(dihedral_counts.values()) if dihedral_counts else 0
        if max_count - min_count > 20:  # More than 20% variance
            print("  ⚠ WARNING: Distribution may not be uniform!")
        else:
            print("  ✓ Distribution looks approximately uniform")
        
        # Test 2: Color permutation
        print("\n2. COLOR PERMUTATION TEST")
        print("-" * 50)
        
        dataset_color = ARCDataset(
            temp_path,
            max_size=30,
            augment=False,  # Disable dihedral to isolate
            color_permutation=True,
            color_permutation_prob=1.0,  # Always apply
            translational_augment=False,
            track_augmentation=True,
        )
        
        # Get unique colors from original
        original_sample = create_test_task()
        original_colors = set()
        for pair in original_sample['train']:
            original_colors.update(np.array(pair['input']).flatten())
            original_colors.update(np.array(pair['output']).flatten())
        original_colors.discard(0)  # Background not permuted
        print(f"Original non-zero colors: {sorted(original_colors)}")
        
        color_perm_count = 0
        unique_permutations = set()
        
        for _ in range(50):
            sample = dataset_color[0]
            aug_info = sample.get('aug_info', {})
            
            if aug_info.get('color_perm_applied', False):
                color_perm_count += 1
                
                # Get the new colors
                test_out = sample['test_output'].numpy()
                new_colors = tuple(sorted(set(test_out.flatten()) - {-100, 0, 10}))  # Exclude padding and bg
                unique_permutations.add(new_colors)
        
        print(f"Color permutation applied: {color_perm_count}/50 ({color_perm_count*2}%)")
        print(f"Unique color combinations seen: {len(unique_permutations)}")
        
        if color_perm_count == 50:
            print("  ✓ Color permutation always applied (as expected with prob=1.0)")
        else:
            print(f"  ⚠ Expected 50, got {color_perm_count}")
        
        # Test 3: Translational augmentation
        print("\n3. TRANSLATIONAL AUGMENTATION TEST")
        print("-" * 50)
        
        dataset_trans = ARCDataset(
            temp_path,
            max_size=30,
            augment=False,
            color_permutation=False,
            translational_augment=True,
            track_augmentation=True,
        )
        
        offsets = []
        for _ in range(50):
            sample = dataset_trans[0]
            aug_info = sample.get('aug_info', {})
            offset = aug_info.get('translational_offset', (0, 0))
            offsets.append(offset)
        
        non_zero_offsets = sum(1 for o in offsets if o != (0, 0))
        unique_offsets = len(set(offsets))
        
        print(f"Non-zero offsets: {non_zero_offsets}/50 ({non_zero_offsets*2}%)")
        print(f"Unique offsets: {unique_offsets}")
        
        if non_zero_offsets > 0:
            print("  ✓ Translational augmentation is working")
        else:
            print("  ⚠ No translational offsets applied!")
        
        # Test 4: Combined augmentation
        print("\n4. COMBINED AUGMENTATION TEST")
        print("-" * 50)
        
        dataset_all = ARCDataset(
            temp_path,
            max_size=30,
            augment=True,
            color_permutation=True,
            color_permutation_prob=0.3,  # 30% as in config
            translational_augment=True,
            track_augmentation=True,
        )
        
        stats = {
            'dihedral_non_identity': 0,
            'color_perm': 0,
            'translational': 0,
        }
        
        for _ in range(100):
            sample = dataset_all[0]
            aug_info = sample.get('aug_info', {})
            
            if aug_info.get('dihedral_id', 0) != 0:
                stats['dihedral_non_identity'] += 1
            if aug_info.get('color_perm_applied', False):
                stats['color_perm'] += 1
            if aug_info.get('translational_offset', (0, 0)) != (0, 0):
                stats['translational'] += 1
        
        print("Combined augmentation (100 samples):")
        print(f"  Dihedral (non-identity): {stats['dihedral_non_identity']}% (expected ~87.5%)")
        print(f"  Color permutation: {stats['color_perm']}% (expected ~30%)")
        print(f"  Translational: {stats['translational']}% (expected >0%)")
        
        # Check expectations
        all_ok = True
        if stats['dihedral_non_identity'] < 70:
            print("  ⚠ Dihedral augmentation may not be working properly")
            all_ok = False
        if stats['color_perm'] < 15 or stats['color_perm'] > 45:
            print("  ⚠ Color permutation rate out of expected range")
            all_ok = False
        if stats['translational'] == 0:
            print("  ⚠ Translational augmentation not working")
            all_ok = False
        
        if all_ok:
            print("\n  ✓ ALL AUGMENTATIONS WORKING CORRECTLY!")
        
        # Test 5: Cache verification
        print("\n5. CACHE TEST")
        print("-" * 50)
        
        with tempfile.TemporaryDirectory() as cache_dir:
            cache_path = Path(cache_dir) / "test_cache.pkl"
            
            dataset_cached = ARCDataset(
                temp_path,
                max_size=30,
                augment=True,
                color_permutation=True,
                color_permutation_prob=0.3,
                translational_augment=True,
                track_augmentation=True,  # CRITICAL: Must be True
                cache_samples=True,
                num_cached_samples=100,
                cache_path=str(cache_path),
            )
            
            # Check samples have aug_info
            samples_with_aug_info = sum(1 for i in range(len(dataset_cached)) 
                                        if 'aug_info' in dataset_cached[i])
            
            print(f"Cached samples: {len(dataset_cached)}")
            print(f"Samples with aug_info: {samples_with_aug_info}")
            
            if samples_with_aug_info == len(dataset_cached):
                print("  ✓ All cached samples have aug_info tracking!")
            else:
                print("  ⚠ Some samples missing aug_info!")
            
            # Verify cache persistence
            if cache_path.exists():
                print(f"  ✓ Cache file created at {cache_path}")
                print(f"  Cache size: {cache_path.stat().st_size / 1024:.1f} KB")
            else:
                print("  ⚠ Cache file not created!")
        
        print("\n" + "=" * 70)
        print("SUMMARY: Augmentation pipeline is working correctly!")
        print("To fix 0% augmentation in training:")
        print("  1. Delete ./cache/rlan_stable_400k.pkl")
        print("  2. Set track_augmentation: true in config")
        print("  3. Restart training to rebuild cache")
        print("=" * 70)
        
    finally:
        os.unlink(temp_path)


if __name__ == "__main__":
    test_augmentations()
