#!/usr/bin/env python
"""Quick check of task_id format in caches."""
import pickle
import json
from pathlib import Path

print("=" * 50)
print("Checking task_id formats")
print("=" * 50)

# Check program cache
cache_path = Path("cache/program_cache_merged_602.json")
if cache_path.exists():
    with open(cache_path) as f:
        program_cache = json.load(f)
    print(f"\nProgram cache: {len(program_cache)} entries")
    print(f"Sample task_ids: {list(program_cache.keys())[:5]}")
else:
    print("Program cache not found")

# Check rolling cache chunks
chunk_path = Path("cache/chunk_0000.pkl")
if chunk_path.exists():
    with open(chunk_path, 'rb') as f:
        chunk_data = pickle.load(f)
    print(f"\nRolling cache chunk: {len(chunk_data)} samples")
    sample = chunk_data[0]
    print(f"Sample task_id: {sample.get('task_id', 'MISSING')}")
    print(f"Sample keys: {list(sample.keys())}")
else:
    print("Rolling cache chunk not found")

# Check main dataset cache
main_cache = Path("cache/rlan_stable_merged_602tasks.pkl")
if main_cache.exists():
    with open(main_cache, 'rb') as f:
        main_data = pickle.load(f)
    print(f"\nMain dataset cache: {len(main_data)} samples")
    sample = main_data[0]
    print(f"Sample task_id: {sample.get('task_id', 'MISSING')}")
else:
    print("Main dataset cache not found")


# TEST THE FIX
print("\n" + "=" * 50)
print("Testing ProgramCache with raw task_id lookup")
print("=" * 50)

from sci_arc.models.generalization.program_guided_training import ProgramCache

cache = ProgramCache("cache/program_cache_merged_602.json")
print(f"\nLoaded {len(cache)} programs")
print(f"raw_task_id index size: {len(cache._by_raw_task_id)}")

# Test lookup with raw task_id
if chunk_path.exists():
    with open(chunk_path, 'rb') as f:
        chunk_data = pickle.load(f)
    
    test_task_id = chunk_data[0].get('task_id', 'unknown')
    print(f"\nTesting lookup with dataset task_id: '{test_task_id}'")
    
    has_program = cache.has(test_task_id)
    program = cache.get(test_task_id)
    
    print(f"  has('{test_task_id}'): {has_program}")
    print(f"  get('{test_task_id}'): {'Found' if program else 'Not found'}")
    
    if program:
        print(f"  trace: {program['trace']}")
        print("\n✓ FIX WORKS! Raw task_id lookup successful!")
    else:
        print(f"\n❌ FIX NOT WORKING - task may not have a cached program")
        # Check if any raw task_ids from dataset are in cache
        print("\nChecking overlap...")
        dataset_raw_ids = set(s.get('task_id', 'x') for s in chunk_data[:100])
        cache_raw_ids = set(cache._by_raw_task_id.keys())
        overlap = dataset_raw_ids & cache_raw_ids
        print(f"  Dataset raw IDs (first 100): {len(dataset_raw_ids)}")
        print(f"  Cache raw IDs: {len(cache_raw_ids)}")
        print(f"  Overlap: {len(overlap)}")
        if overlap:
            print(f"  Sample overlapping: {list(overlap)[:5]}")
            # Test one that should work
            test_id = list(overlap)[0]
            print(f"\nRe-testing with overlapping task_id: '{test_id}'")
            print(f"  has('{test_id}'): {cache.has(test_id)}")
            prog = cache.get(test_id)
            print(f"  get('{test_id}'): {'Found' if prog else 'Not found'}")
            if prog:
                print(f"  trace: {prog['trace']}")
                print("\n✓ FIX WORKS! Lookup successful for overlapping task_id!")

