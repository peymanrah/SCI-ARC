#!/usr/bin/env python3
"""Check cached samples for aug_info field."""
import pickle
from pathlib import Path

cache_path = Path("./cache/rlan_stable_400k.pkl")
if cache_path.exists():
    with open(cache_path, 'rb') as f:
        data = pickle.load(f)
    print(f"Samples in cache: {len(data)}")
    print(f"First sample keys: {data[0].keys()}")
    print(f"Has aug_info: {'aug_info' in data[0]}")
    
    if 'aug_info' in data[0]:
        print(f"Aug info: {data[0]['aug_info']}")
        
        # Count augmentation stats
        dihedral_counts = [0] * 8
        color_perm_count = 0
        translational_count = 0
        for sample in data:
            aug = sample.get('aug_info', {})
            did = aug.get('dihedral_id', 0)
            dihedral_counts[did] += 1
            if aug.get('color_perm_applied', False):
                color_perm_count += 1
            if aug.get('translational_offset', (0, 0)) != (0, 0):
                translational_count += 1
        
        print(f"\nDihedral distribution: {dihedral_counts}")
        print(f"Color perm: {color_perm_count}/{len(data)} ({100*color_perm_count/len(data):.1f}%)")
        print(f"Translational: {translational_count}/{len(data)} ({100*translational_count/len(data):.1f}%)")
    else:
        print("\n*** CACHE MISSING aug_info - NEEDS TO BE REBUILT! ***")
else:
    print(f"Cache not found at {cache_path}")
