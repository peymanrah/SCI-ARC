#!/usr/bin/env python3
"""Check max grid size and sample count in ARC data."""

import json
import os
from pathlib import Path

def check_arc_stats():
    paths = [
        './data/arc-agi/data/training',
        './data/arc-agi/data/evaluation',
    ]
    
    total_tasks = 0
    total_samples = 0
    max_h, max_w = 0, 0
    
    for path in paths:
        if not os.path.exists(path):
            print(f"Path not found: {path}")
            continue
            
        tasks = [f for f in os.listdir(path) if f.endswith('.json')]
        print(f"\n{path}: {len(tasks)} tasks")
        total_tasks += len(tasks)
        
        for task_file in tasks:
            with open(os.path.join(path, task_file)) as f:
                task = json.load(f)
            
            for split in ['train', 'test']:
                for pair in task.get(split, []):
                    total_samples += 1
                    for grid_type in ['input', 'output']:
                        if grid_type in pair:
                            h = len(pair[grid_type])
                            w = len(pair[grid_type][0]) if h > 0 else 0
                            max_h = max(max_h, h)
                            max_w = max(max_w, w)
    
    print(f"\n{'='*50}")
    print(f"SUMMARY:")
    print(f"{'='*50}")
    print(f"Total tasks:   {total_tasks}")
    print(f"Total samples: {total_samples}")
    print(f"Max grid size: {max_h} x {max_w}")
    print()
    
    # Check official ARC spec
    print("OFFICIAL ARC-AGI SPEC:")
    print("  - Max grid dimension: 30x30")
    print("  - Colors: 0-9 (10 colors)")
    print()
    
    if max_h <= 30 and max_w <= 30:
        print("✓ max_grid_size=30 is SUFFICIENT for all ARC tasks")
    else:
        print(f"⚠️  max_grid_size=30 is NOT enough! Found {max_h}x{max_w}")
    
    # Augmentation calculation
    print()
    print("AUGMENTATION ANALYSIS:")
    print("  - TRM uses: 1000x augmentation per task")
    print(f"  - TRM total: {total_tasks} × 1000 = {total_tasks * 1000:,} samples")
    print()
    print("  - 8 dihedral transforms (rotation + flip)")
    print("  - 362,880 color permutations (9!)")
    print("  - ~100 translational positions")
    print("  - INFINITE unique combinations!")
    print()
    
    # Cache recommendations
    print("CACHE SIZE RECOMMENDATIONS:")
    samples_per_task = [32, 64, 160, 400, 1000]
    for n in samples_per_task:
        total = total_tasks * n
        print(f"  {n:4d} per task = {total:>8,} total samples")
    
    print()
    print("NOTES:")
    print("  - TRM uses 1000x pre-generated augmentation (arc-aug-1000 folder)")
    print("  - RLAN uses on-the-fly augmentation (infinite unique samples)")
    print("  - With cache_samples=false, model sees NEW data every epoch")
    print("  - 64,000 cached = 160 per task (good for initial memorization)")
    print("  - For generalization: use cache_samples=false")


if __name__ == '__main__':
    check_arc_stats()
