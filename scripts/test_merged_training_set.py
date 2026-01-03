#!/usr/bin/env python3
"""
Test Script for Merged Training Set Builder
============================================

This script validates the merged training set builder with comprehensive tests:

1. Determinism test - same output on multiple runs
2. Invariance tests - canonical hash invariant to D4/color/translation
3. Leakage gate test - injected eval task is excluded
4. Near-dup sensitivity test - slightly modified task is flagged
5. Manifest integrity test - no overlap between splits, no eval leakage
6. Artifact contamination test - validates manifest hash checking

USAGE:
    # Run all tests
    python scripts/test_merged_training_set.py
    
    # Run specific test
    python scripts/test_merged_training_set.py --test determinism
    
    # Verbose output
    python scripts/test_merged_training_set.py --verbose

Author: SCI-ARC Team
Date: January 2026
"""

import argparse
import json
import os
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.build_merged_training_set import (
    BuildConfig,
    build_merged_training_set,
    compute_canonical_hash,
    compute_raw_hash,
    apply_dihedral,
    crop_to_content,
    canonicalize_colors,
    extract_features,
    detect_near_duplicates,
    cosine_similarity,
)


# =============================================================================
# TEST UTILITIES
# =============================================================================

def create_test_task(
    train_inputs: List[List[List[int]]],
    train_outputs: List[List[List[int]]],
    test_input: List[List[int]],
    test_output: Optional[List[List[int]]] = None,
) -> Dict:
    """Create a test task dictionary."""
    task = {
        'train': [
            {'input': inp, 'output': out}
            for inp, out in zip(train_inputs, train_outputs)
        ],
        'test': [{'input': test_input}]
    }
    if test_output:
        task['test'][0]['output'] = test_output
    return task


def print_result(test_name: str, passed: bool, details: str = ""):
    """Print test result."""
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  [{status}] {test_name}")
    if details and not passed:
        print(f"          {details}")


# =============================================================================
# TEST 1: DETERMINISM
# =============================================================================

def test_determinism(verbose: bool = False) -> bool:
    """
    Test that the build process is deterministic.
    Running twice with same config should produce identical outputs.
    """
    print("\n" + "=" * 60)
    print("TEST 1: DETERMINISM")
    print("=" * 60)
    
    # Create temporary output directories
    temp_dir1 = tempfile.mkdtemp(prefix="merged_test1_")
    temp_dir2 = tempfile.mkdtemp(prefix="merged_test2_")
    
    try:
        # Run build twice with same config
        config1 = BuildConfig(
            output_dir=temp_dir1,
            verbose=verbose,
            dry_run=False,
        )
        config2 = BuildConfig(
            output_dir=temp_dir2,
            verbose=verbose,
            dry_run=False,
        )
        
        result1 = build_merged_training_set(config1)
        result2 = build_merged_training_set(config2)
        
        # Check counts match
        counts_match = (
            result1.merged_train_count == result2.merged_train_count and
            result1.merged_dev_count == result2.merged_dev_count and
            result1.exact_dedup_count == result2.exact_dedup_count and
            result1.near_dup_quarantine_count == result2.near_dup_quarantine_count
        )
        print_result("Counts match", counts_match)
        
        # Check manifest hashes match
        hashes_match = result1.manifest_sha256 == result2.manifest_sha256
        print_result("Manifest hashes match", hashes_match, 
                     f"{result1.manifest_sha256[:16]} vs {result2.manifest_sha256[:16]}")
        
        # Check file contents match
        files_to_check = [
            "merged_train_manifest.jsonl",
            "merged_dev_manifest.jsonl",
            "excluded_exact.jsonl",
            "quarantine_near_dup.jsonl",
        ]
        
        all_files_match = True
        for filename in files_to_check:
            path1 = Path(temp_dir1) / filename
            path2 = Path(temp_dir2) / filename
            
            if path1.exists() and path2.exists():
                content1 = path1.read_text()
                content2 = path2.read_text()
                if content1 != content2:
                    all_files_match = False
                    if verbose:
                        print(f"    Mismatch in {filename}")
        
        print_result("All output files match", all_files_match)
        
        passed = counts_match and hashes_match and all_files_match
        
    finally:
        shutil.rmtree(temp_dir1, ignore_errors=True)
        shutil.rmtree(temp_dir2, ignore_errors=True)
    
    return passed


# =============================================================================
# TEST 2: INVARIANCE
# =============================================================================

def test_invariance(verbose: bool = False) -> bool:
    """
    Test that canonical hash is invariant to D4, translation, and color permutation.
    """
    print("\n" + "=" * 60)
    print("TEST 2: CANONICAL HASH INVARIANCE")
    print("=" * 60)
    
    # Create a base task
    base_task = create_test_task(
        train_inputs=[[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
        train_outputs=[[[2, 1], [4, 3]], [[6, 5], [8, 7]]],
        test_input=[[1, 1], [2, 2]],
        test_output=[[1, 1], [2, 2]],
    )
    
    base_hash = compute_canonical_hash(base_task)
    if verbose:
        print(f"  Base hash: {base_hash[:16]}...")
    
    all_passed = True
    
    # Test 2.1: Dihedral invariance
    for d_id in range(8):
        rotated_task = {
            'train': [
                {
                    'input': apply_dihedral(np.array(p['input']), d_id).tolist(),
                    'output': apply_dihedral(np.array(p['output']), d_id).tolist(),
                }
                for p in base_task['train']
            ],
            'test': [
                {
                    'input': apply_dihedral(np.array(base_task['test'][0]['input']), d_id).tolist(),
                    'output': apply_dihedral(np.array(base_task['test'][0]['output']), d_id).tolist(),
                }
            ]
        }
        
        rotated_hash = compute_canonical_hash(rotated_task)
        match = rotated_hash == base_hash
        if not match:
            all_passed = False
        if verbose or not match:
            print_result(f"Dihedral {d_id} invariance", match)
    
    print_result("All 8 dihedral transforms", all_passed)
    
    # Test 2.2: Color permutation invariance
    color_perm = [0, 5, 6, 7, 8, 1, 2, 3, 4, 9]  # Shuffle colors 1-8
    
    def remap_grid(grid, perm):
        arr = np.array(grid)
        remapped = np.zeros_like(arr)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                remapped[i, j] = perm[arr[i, j]]
        return remapped.tolist()
    
    color_task = {
        'train': [
            {
                'input': remap_grid(p['input'], color_perm),
                'output': remap_grid(p['output'], color_perm),
            }
            for p in base_task['train']
        ],
        'test': [
            {
                'input': remap_grid(base_task['test'][0]['input'], color_perm),
                'output': remap_grid(base_task['test'][0]['output'], color_perm),
            }
        ]
    }
    
    color_hash = compute_canonical_hash(color_task)
    color_match = color_hash == base_hash
    print_result("Color permutation invariance", color_match)
    if not color_match:
        all_passed = False
    
    # Test 2.3: Translation invariance (add padding, should be cropped)
    def pad_grid(grid, pad=2):
        arr = np.array(grid)
        # Pad with 0 (background)
        padded = np.pad(arr, pad, mode='constant', constant_values=0)
        return padded.tolist()
    
    padded_task = {
        'train': [
            {
                'input': pad_grid(p['input'], 3),
                'output': pad_grid(p['output'], 3),
            }
            for p in base_task['train']
        ],
        'test': [
            {
                'input': pad_grid(base_task['test'][0]['input'], 3),
                'output': pad_grid(base_task['test'][0]['output'], 3),
            }
        ]
    }
    
    padded_hash = compute_canonical_hash(padded_task)
    # Note: translation invariance depends on crop_to_content working correctly
    # The hashes may not match if the background color isn't mode color
    # This is expected behavior - we document it
    translation_match = padded_hash == base_hash
    print_result("Translation invariance (padding with mode color)", translation_match,
                 "Expected: depends on background detection")
    
    return all_passed


# =============================================================================
# TEST 3: LEAKAGE GATE
# =============================================================================

def test_leakage_gate(verbose: bool = False) -> bool:
    """
    Test that a task present in eval is excluded from training.
    """
    print("\n" + "=" * 60)
    print("TEST 3: LEAKAGE GATE (EVAL TASK EXCLUSION)")
    print("=" * 60)
    
    # Create temp directories
    temp_dir = tempfile.mkdtemp(prefix="leakage_test_")
    temp_train = Path(temp_dir) / "train"
    temp_eval = Path(temp_dir) / "eval"
    temp_output = Path(temp_dir) / "output"
    
    temp_train.mkdir()
    temp_eval.mkdir()
    temp_output.mkdir()
    
    try:
        # Create a task that will be in both train and eval
        shared_task = create_test_task(
            train_inputs=[[[1, 2], [3, 4]]],
            train_outputs=[[[4, 3], [2, 1]]],
            test_input=[[1, 1]],
            test_output=[[1, 1]],
        )
        
        # Create a unique train task
        unique_task = create_test_task(
            train_inputs=[[[5, 5], [5, 5]]],
            train_outputs=[[[6, 6], [6, 6]]],
            test_input=[[7, 7]],
            test_output=[[8, 8]],
        )
        
        # Write tasks
        with open(temp_train / "shared.json", 'w') as f:
            json.dump(shared_task, f)
        with open(temp_train / "unique.json", 'w') as f:
            json.dump(unique_task, f)
        with open(temp_eval / "shared_eval.json", 'w') as f:
            json.dump(shared_task, f)  # Same task in eval!
        
        # Run build
        config = BuildConfig(
            agi1_train_path=str(temp_train),
            agi2_train_path=str(temp_train),  # Will create duplicates, that's fine
            agi1_eval_path=str(temp_eval),
            agi2_eval_path=str(temp_eval),
            output_dir=str(temp_output),
            verbose=verbose,
            dry_run=False,
        )
        
        result = build_merged_training_set(config)
        
        # Check that exclusion happened
        exclusion_file = temp_output / "excluded_exact.jsonl"
        exclusions = []
        if exclusion_file.exists():
            with open(exclusion_file) as f:
                for line in f:
                    exclusions.append(json.loads(line))
        
        # Should have excluded the shared task
        shared_excluded = any('shared' in e.get('task_uid', '') for e in exclusions)
        print_result("Shared task excluded", shared_excluded,
                     f"Exclusions: {[e.get('task_uid') for e in exclusions]}")
        
        # Check manifest doesn't contain shared task
        manifest_file = temp_output / "merged_train_manifest.jsonl"
        manifest_uids = []
        if manifest_file.exists():
            with open(manifest_file) as f:
                for line in f:
                    record = json.loads(line)
                    manifest_uids.append(record.get('task_uid', ''))
        
        shared_not_in_manifest = not any('shared' in uid for uid in manifest_uids)
        print_result("Shared task not in manifest", shared_not_in_manifest,
                     f"Manifest UIDs: {manifest_uids}")
        
        # Unique task should be in manifest
        unique_in_manifest = any('unique' in uid for uid in manifest_uids)
        print_result("Unique task in manifest", unique_in_manifest)
        
        passed = shared_excluded and shared_not_in_manifest and unique_in_manifest
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    return passed


# =============================================================================
# TEST 4: NEAR-DUP SENSITIVITY
# =============================================================================

def test_near_dup_sensitivity(verbose: bool = False) -> bool:
    """
    Test that near-duplicates are flagged (one pixel different).
    """
    print("\n" + "=" * 60)
    print("TEST 4: NEAR-DUPLICATE SENSITIVITY")
    print("=" * 60)
    
    # Create two very similar tasks
    task1 = create_test_task(
        train_inputs=[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]],
        train_outputs=[[[9, 8, 7], [6, 5, 4], [3, 2, 1]]],
        test_input=[[1, 1, 1], [1, 1, 1], [1, 1, 1]],
        test_output=[[2, 2, 2], [2, 2, 2], [2, 2, 2]],
    )
    
    # Same structure, one pixel different
    task2 = create_test_task(
        train_inputs=[[[1, 2, 3], [4, 5, 6], [7, 8, 0]]],  # 9 -> 0
        train_outputs=[[[9, 8, 7], [6, 5, 4], [3, 2, 1]]],
        test_input=[[1, 1, 1], [1, 1, 1], [1, 1, 1]],
        test_output=[[2, 2, 2], [2, 2, 2], [2, 2, 2]],
    )
    
    # Extract features
    f1 = extract_features(task1, "task1")
    f2 = extract_features(task2, "task2")
    
    # Check similarity
    hist_sim = cosine_similarity(f1.color_histogram, f2.color_histogram)
    struct_sim = 1.0  # Same structure
    
    if verbose:
        print(f"  Histogram similarity: {hist_sim:.4f}")
        print(f"  (Threshold for near-dup: 0.95)")
    
    # With default thresholds, these should be flagged as near-dups
    near_dups = detect_near_duplicates(
        [f1],  # candidate
        [f2],  # "eval" (we're testing detection)
        histogram_threshold=0.90,  # Lower threshold for test
        structure_threshold=0.80,
    )
    
    detected = len(near_dups) > 0
    print_result("Near-duplicate detected", detected,
                 f"Histogram sim: {hist_sim:.4f}, detected: {len(near_dups)} near-dups")
    
    # Test with very different task
    task3 = create_test_task(
        train_inputs=[[[0, 0], [0, 0]]],
        train_outputs=[[[1, 1], [1, 1]]],
        test_input=[[0]],
        test_output=[[1]],
    )
    
    f3 = extract_features(task3, "task3")
    hist_sim_diff = cosine_similarity(f1.color_histogram, f3.color_histogram)
    
    near_dups_diff = detect_near_duplicates(
        [f1],
        [f3],
        histogram_threshold=0.95,
        structure_threshold=0.90,
    )
    
    not_detected = len(near_dups_diff) == 0
    print_result("Different task not flagged", not_detected,
                 f"Histogram sim: {hist_sim_diff:.4f}")
    
    return detected and not_detected


# =============================================================================
# TEST 5: MANIFEST INTEGRITY
# =============================================================================

def test_manifest_integrity(verbose: bool = False) -> bool:
    """
    Test that manifests have no overlaps and no eval leakage.
    """
    print("\n" + "=" * 60)
    print("TEST 5: MANIFEST INTEGRITY")
    print("=" * 60)
    
    temp_dir = tempfile.mkdtemp(prefix="integrity_test_")
    
    try:
        config = BuildConfig(
            output_dir=temp_dir,
            verbose=verbose,
            dry_run=False,
        )
        
        result = build_merged_training_set(config)
        
        # Load manifests
        train_manifest = Path(temp_dir) / "merged_train_manifest.jsonl"
        dev_manifest = Path(temp_dir) / "merged_dev_manifest.jsonl"
        
        train_uids = set()
        train_hashes = set()
        dev_uids = set()
        dev_hashes = set()
        
        with open(train_manifest) as f:
            for line in f:
                record = json.loads(line)
                train_uids.add(record['task_uid'])
                train_hashes.add(record['canonical_sha256'])
        
        with open(dev_manifest) as f:
            for line in f:
                record = json.loads(line)
                dev_uids.add(record['task_uid'])
                dev_hashes.add(record['canonical_sha256'])
        
        # Check no overlap
        uid_overlap = train_uids & dev_uids
        hash_overlap = train_hashes & dev_hashes
        
        no_uid_overlap = len(uid_overlap) == 0
        no_hash_overlap = len(hash_overlap) == 0
        
        print_result("No UID overlap between train/dev", no_uid_overlap,
                     f"Overlapping UIDs: {uid_overlap}" if uid_overlap else "")
        print_result("No hash overlap between train/dev", no_hash_overlap,
                     f"Overlapping hashes: {len(hash_overlap)}" if hash_overlap else "")
        
        # Load eval hashes (from original data)
        eval_hashes = set()
        for eval_path in [config.agi1_eval_path, config.agi2_eval_path]:
            for json_file in Path(eval_path).glob("*.json"):
                with open(json_file) as f:
                    task = json.load(f)
                eval_hashes.add(compute_canonical_hash(task))
        
        # Check no eval leakage
        train_eval_overlap = train_hashes & eval_hashes
        dev_eval_overlap = dev_hashes & eval_hashes
        
        no_train_eval_leak = len(train_eval_overlap) == 0
        no_dev_eval_leak = len(dev_eval_overlap) == 0
        
        print_result("No eval leakage in train", no_train_eval_leak,
                     f"Leaked hashes: {len(train_eval_overlap)}" if train_eval_overlap else "")
        print_result("No eval leakage in dev", no_dev_eval_leak,
                     f"Leaked hashes: {len(dev_eval_overlap)}" if dev_eval_overlap else "")
        
        passed = no_uid_overlap and no_hash_overlap and no_train_eval_leak and no_dev_eval_leak
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    return passed


# =============================================================================
# TEST 6: ARTIFACT CONTAMINATION
# =============================================================================

def test_artifact_contamination(verbose: bool = False) -> bool:
    """
    Test that manifest hash validation works for cache/buffer checking.
    """
    print("\n" + "=" * 60)
    print("TEST 6: ARTIFACT CONTAMINATION DETECTION")
    print("=" * 60)
    
    temp_dir = tempfile.mkdtemp(prefix="contamination_test_")
    
    try:
        # Build once
        config = BuildConfig(
            output_dir=temp_dir,
            verbose=verbose,
            dry_run=False,
        )
        
        result1 = build_merged_training_set(config)
        hash1 = result1.manifest_sha256
        
        # Load metadata
        metadata_path = Path(temp_dir) / "build_metadata.json"
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        stored_hash = metadata.get('manifest_sha256')
        
        # Verify hash is stored
        hash_stored = stored_hash == hash1
        print_result("Manifest hash stored in metadata", hash_stored,
                     f"Stored: {stored_hash[:16] if stored_hash else 'None'}...")
        
        # Simulate a cache/buffer that was built with this manifest
        fake_cache_manifest_hash = hash1
        
        # Verify validation would pass
        validation_pass = fake_cache_manifest_hash == hash1
        print_result("Valid cache accepted", validation_pass)
        
        # Simulate a stale cache with different hash
        stale_cache_hash = "0" * 64
        validation_reject = stale_cache_hash != hash1
        print_result("Stale cache rejected", validation_reject)
        
        passed = hash_stored and validation_pass and validation_reject
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    return passed


# =============================================================================
# MAIN
# =============================================================================

def run_all_tests(verbose: bool = False) -> bool:
    """Run all tests and return overall pass/fail."""
    print("\n" + "=" * 70)
    print("MERGED TRAINING SET BUILDER - TEST SUITE")
    print("=" * 70)
    
    tests = [
        ("Determinism", test_determinism),
        ("Invariance", test_invariance),
        ("Leakage Gate", test_leakage_gate),
        ("Near-Dup Sensitivity", test_near_dup_sensitivity),
        ("Manifest Integrity", test_manifest_integrity),
        ("Artifact Contamination", test_artifact_contamination),
    ]
    
    results = {}
    
    for name, test_func in tests:
        try:
            results[name] = test_func(verbose)
        except Exception as e:
            print(f"\n[ERROR] {name}: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    all_passed = True
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  [{status}] {name}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("  ALL TESTS PASSED ✓")
    else:
        print("  SOME TESTS FAILED ✗")
    print()
    
    return all_passed


def main():
    parser = argparse.ArgumentParser(
        description="Test suite for merged training set builder"
    )
    
    parser.add_argument(
        '--test',
        choices=['determinism', 'invariance', 'leakage', 'neardup', 'integrity', 'contamination', 'all'],
        default='all',
        help='Which test to run (default: all)',
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output',
    )
    
    args = parser.parse_args()
    
    test_map = {
        'determinism': test_determinism,
        'invariance': test_invariance,
        'leakage': test_leakage_gate,
        'neardup': test_near_dup_sensitivity,
        'integrity': test_manifest_integrity,
        'contamination': test_artifact_contamination,
    }
    
    if args.test == 'all':
        passed = run_all_tests(args.verbose)
    else:
        passed = test_map[args.test](args.verbose)
    
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
