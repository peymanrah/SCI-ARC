#!/usr/bin/env python3
"""
Dihedral Round-Trip Sanity Check
Verifies: inverse(transform(x)) == x for all 8 D4 transforms
"""

import torch
import numpy as np

# D4 Dihedral transforms (must match train_rlan.py and test_tta_voting.py)
def apply_dihedral(grid: torch.Tensor, transform_id: int) -> torch.Tensor:
    """Apply dihedral transform to grid. Grid shape: (..., H, W)"""
    if transform_id == 0:  # Identity
        return grid
    elif transform_id == 1:  # Rot90
        return torch.rot90(grid, 1, dims=(-2, -1))
    elif transform_id == 2:  # Rot180
        return torch.rot90(grid, 2, dims=(-2, -1))
    elif transform_id == 3:  # Rot270
        return torch.rot90(grid, 3, dims=(-2, -1))
    elif transform_id == 4:  # Flip horizontal
        return torch.flip(grid, dims=[-1])
    elif transform_id == 5:  # Flip vertical
        return torch.flip(grid, dims=[-2])
    elif transform_id == 6:  # Flip + Rot90
        return torch.rot90(torch.flip(grid, dims=[-1]), 1, dims=(-2, -1))
    elif transform_id == 7:  # Flip + Rot270
        return torch.rot90(torch.flip(grid, dims=[-1]), 3, dims=(-2, -1))
    else:
        raise ValueError(f"Invalid transform_id: {transform_id}")

# DIHEDRAL_INVERSE mapping (must match train_rlan.py)
DIHEDRAL_INVERSE = [0, 3, 2, 1, 4, 5, 6, 7]

TRANSFORM_NAMES = [
    "D0: Identity",
    "D1: Rot90",
    "D2: Rot180", 
    "D3: Rot270",
    "D4: FlipH",
    "D5: FlipV",
    "D6: FlipH+Rot90",
    "D7: FlipH+Rot270"
]

def test_roundtrip():
    print("=" * 60)
    print("DIHEDRAL ROUND-TRIP SANITY CHECK")
    print("=" * 60)
    print(f"\nDIHEDRAL_INVERSE mapping: {DIHEDRAL_INVERSE}")
    print()
    
    # Test on several grid sizes and shapes
    test_cases = [
        (3, 3),   # Square small
        (5, 5),   # Square medium
        (3, 7),   # Tall
        (7, 3),   # Wide
        (10, 10), # Larger
    ]
    
    all_passed = True
    
    for h, w in test_cases:
        print(f"\nTesting grid size: {h}x{w}")
        print("-" * 40)
        
        # Create a non-symmetric grid with unique values at each position
        grid = torch.arange(h * w).reshape(h, w).float()
        
        for t_id in range(8):
            inv_id = DIHEDRAL_INVERSE[t_id]
            
            # Forward transform
            transformed = apply_dihedral(grid, t_id)
            
            # Inverse transform
            recovered = apply_dihedral(transformed, inv_id)
            
            # Check if round-trip is identity
            is_equal = torch.equal(grid, recovered)
            status = "✓ PASS" if is_equal else "✗ FAIL"
            
            if not is_equal:
                all_passed = False
                print(f"  {TRANSFORM_NAMES[t_id]:<20} → inverse={inv_id:<2} {status}")
                print(f"    Original:\n{grid}")
                print(f"    Transformed:\n{transformed}")
                print(f"    Recovered:\n{recovered}")
                print(f"    Diff:\n{grid - recovered}")
            else:
                print(f"  {TRANSFORM_NAMES[t_id]:<20} → inverse={inv_id:<2} {status}")
    
    print()
    print("=" * 60)
    if all_passed:
        print("✓ ALL ROUND-TRIP TESTS PASSED")
        print("  Inverse transform implementation is CORRECT")
    else:
        print("✗ SOME ROUND-TRIP TESTS FAILED")
        print("  There may be a bug in the inverse mapping!")
    print("=" * 60)
    
    # Additional: Test that transform -> inverse -> original works for random grids
    print("\n\nAdditional: Testing 100 random grids...")
    random_failures = 0
    for _ in range(100):
        h, w = np.random.randint(2, 15, size=2)
        grid = torch.randint(0, 10, (h, w)).float()
        
        for t_id in range(8):
            inv_id = DIHEDRAL_INVERSE[t_id]
            transformed = apply_dihedral(grid, t_id)
            recovered = apply_dihedral(transformed, inv_id)
            
            if not torch.equal(grid, recovered):
                random_failures += 1
                print(f"  Random failure: {h}x{w} grid, transform {t_id}")
    
    if random_failures == 0:
        print("✓ All 800 random round-trips passed (100 grids × 8 transforms)")
    else:
        print(f"✗ {random_failures} random round-trip failures detected")
    
    return all_passed and random_failures == 0

if __name__ == "__main__":
    test_roundtrip()
