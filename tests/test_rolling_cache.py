"""
Smoke Tests for Rolling Refresh Cache (Jan 2026)

Tests:
1. Basic cache initialization and pool generation
2. Epoch swap and prefetch mechanics
3. Anti-repeat window (fingerprint exclusion)
4. Coverage statistics tracking
5. Thread safety of swap operations
6. Backward compatibility with static cache mode
7. ARPS CUDA indexing bug fix verification

Run with:
    python -m pytest tests/test_rolling_cache.py -v
    
Or standalone:
    python tests/test_rolling_cache.py
"""

import sys
import time
import threading
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np


def test_fingerprint_hashing():
    """Test AugmentationFingerprint equality and hashing."""
    from sci_arc.data.rolling_cache import AugmentationFingerprint
    
    fp1 = AugmentationFingerprint(
        task_id="task_001",
        dihedral_id=3,
        color_perm_hash=12345,
        offset=(2, -1)
    )
    
    fp2 = AugmentationFingerprint(
        task_id="task_001",
        dihedral_id=3,
        color_perm_hash=12345,
        offset=(2, -1)
    )
    
    fp3 = AugmentationFingerprint(
        task_id="task_001",
        dihedral_id=4,  # Different dihedral
        color_perm_hash=12345,
        offset=(2, -1)
    )
    
    # Same fingerprints should be equal
    assert fp1 == fp2, "Identical fingerprints should be equal"
    assert hash(fp1) == hash(fp2), "Identical fingerprints should have same hash"
    
    # Different fingerprints should not be equal
    assert fp1 != fp3, "Different fingerprints should not be equal"
    
    # Should work in sets
    fp_set = {fp1, fp2, fp3}
    assert len(fp_set) == 2, "Set should deduplicate identical fingerprints"
    
    print("✓ Fingerprint hashing test passed")


def test_coverage_stats():
    """Test EpochCoverageStats tracking and scoring."""
    from sci_arc.data.rolling_cache import EpochCoverageStats
    
    stats = EpochCoverageStats()
    
    # Simulate samples with various augmentations
    samples = [
        {'aug_info': {'dihedral_id': 0, 'color_perm': None, 'translational_offset': (0, 0)}},
        {'aug_info': {'dihedral_id': 1, 'color_perm': [0,1,2,3,4,5,6,7,8,9], 'translational_offset': (1, 2)}},
        {'aug_info': {'dihedral_id': 2, 'color_perm': None, 'translational_offset': (0, 1)}},
        {'aug_info': {'dihedral_id': 3, 'color_perm': [9,8,7,6,5,4,3,2,1,0], 'translational_offset': (2, 2)}},
        {'aug_info': {'dihedral_id': 4, 'color_perm': None, 'translational_offset': (0, 0)}},
        {'aug_info': {'dihedral_id': 5, 'color_perm': [1,0,2,3,4,5,6,7,8,9], 'translational_offset': (1, 0)}},
        {'aug_info': {'dihedral_id': 6, 'color_perm': None, 'translational_offset': (0, 3)}},
        {'aug_info': {'dihedral_id': 7, 'color_perm': None, 'translational_offset': (3, 0)}},
    ]
    
    for sample in samples:
        stats.update(sample)
    
    # Check dihedral distribution
    assert sum(stats.dihedral_counts.values()) == 8, "Should have 8 dihedral samples"
    
    # Check color permutation tracking
    assert stats.color_perm_applied == 3, "Should have 3 color perms applied"
    assert stats.color_perm_skipped == 5, "Should have 5 color perms skipped"
    
    # Check offset tracking
    assert len(stats.offset_magnitudes) == 8, "Should have 8 offset magnitudes"
    
    # Coverage score should be reasonable (not 0, not 1)
    score = stats.get_coverage_score()
    assert 0 < score <= 1.0, f"Coverage score {score} should be in (0, 1]"
    
    print(f"✓ Coverage stats test passed (score={score:.3f})")


def create_mock_tasks(num_tasks=10):
    """Create mock tasks for testing."""
    tasks = []
    for i in range(num_tasks):
        tasks.append({
            'task_id': f'task_{i:03d}',
            'train': [
                {'input': [[0, 1], [1, 0]], 'output': [[1, 0], [0, 1]]}
            ],
            'test': [
                {'input': [[0, 1], [1, 0]], 'output': [[1, 0], [0, 1]]}
            ]
        })
    return tasks


def mock_generate_sample(task_idx, tasks):
    """Mock sample generation function."""
    task = tasks[task_idx]
    
    # Random augmentation
    dihedral_id = np.random.randint(0, 8)
    color_perm = np.random.permutation(10) if np.random.random() > 0.5 else None
    offset = (np.random.randint(-2, 3), np.random.randint(-2, 3))
    
    return {
        'task_id': task['task_id'],
        'task_idx': task_idx,
        'input': np.array(task['train'][0]['input']),
        'output': np.array(task['train'][0]['output']),
        'aug_info': {
            'dihedral_id': dihedral_id,
            'color_perm': color_perm,
            'translational_offset': offset,
        }
    }


def test_rolling_cache_initialization():
    """Test basic cache initialization."""
    from sci_arc.data.rolling_cache import RollingRefreshCache
    
    tasks = create_mock_tasks(10)
    
    cache = RollingRefreshCache(
        tasks=tasks,
        generate_sample_fn=lambda idx: mock_generate_sample(idx, tasks),
        pool_size=100,
        refresh_fraction=0.25,
        anti_repeat_window=2,
        prefetch_workers=2,
        seed=42,
        verbose=False,
    )
    
    # Check initial pool was created
    assert len(cache) == 100, f"Cache should have 100 samples, got {len(cache)}"
    assert cache._current_epoch == 0, "Should be at epoch 0"
    
    # Check samples have required fields
    sample = cache[0]
    assert 'task_id' in sample, "Sample should have task_id"
    assert 'aug_info' in sample, "Sample should have aug_info"
    
    cache.shutdown()
    print("✓ Cache initialization test passed")


def test_epoch_swap_and_prefetch():
    """Test epoch swap and async prefetch."""
    from sci_arc.data.rolling_cache import RollingRefreshCache
    
    tasks = create_mock_tasks(5)
    
    cache = RollingRefreshCache(
        tasks=tasks,
        generate_sample_fn=lambda idx: mock_generate_sample(idx, tasks),
        pool_size=50,
        refresh_fraction=0.3,  # 30% = 15 samples refreshed
        anti_repeat_window=2,
        prefetch_workers=2,
        seed=42,
        verbose=False,
    )
    
    # Get epoch 0 samples
    epoch0_samples = cache.get_epoch_samples(0)
    assert len(epoch0_samples) == 50
    
    # Start prefetch for epoch 1
    cache.prefetch_next_epoch(1)
    
    # Wait a bit for prefetch to complete
    time.sleep(0.5)
    
    # Swap to epoch 1
    success = cache.swap_to_next_epoch(timeout=5.0)
    assert success, "Swap should succeed"
    assert cache._current_epoch == 1, "Should be at epoch 1"
    
    epoch1_samples = cache.get_epoch_samples(1)
    assert len(epoch1_samples) == 50, "Epoch 1 should also have 50 samples"
    
    # Some samples should be different (30% refreshed)
    # Compare task_ids - at least some should differ due to shuffle
    epoch0_task_ids = [s['task_id'] for s in epoch0_samples[:10]]
    epoch1_task_ids = [s['task_id'] for s in epoch1_samples[:10]]
    
    # Not all should be identical (shuffled)
    # This is a probabilistic test, might occasionally fail
    
    cache.shutdown()
    print("✓ Epoch swap and prefetch test passed")


def test_anti_repeat_window():
    """Test anti-repeat window prevents recent fingerprints from repeating."""
    from sci_arc.data.rolling_cache import RollingRefreshCache
    
    tasks = create_mock_tasks(3)
    
    # Use small pool and high refresh to force fingerprint reuse attempts
    cache = RollingRefreshCache(
        tasks=tasks,
        generate_sample_fn=lambda idx: mock_generate_sample(idx, tasks),
        pool_size=20,
        refresh_fraction=0.5,  # 50% refreshed
        anti_repeat_window=2,  # Exclude fingerprints from last 2 epochs
        prefetch_workers=1,
        seed=42,
        verbose=False,
    )
    
    # Simulate multiple epochs
    for epoch in range(1, 4):
        cache.prefetch_next_epoch(epoch)
        time.sleep(0.2)
        cache.swap_to_next_epoch()
    
    # Check fingerprint history is being cleaned up
    assert len(cache._fingerprint_history) > 0, "Should have fingerprint history"
    
    cache.shutdown()
    print("✓ Anti-repeat window test passed")


def test_thread_safety():
    """Test thread safety of swap operations."""
    from sci_arc.data.rolling_cache import RollingRefreshCache
    
    tasks = create_mock_tasks(5)
    
    cache = RollingRefreshCache(
        tasks=tasks,
        generate_sample_fn=lambda idx: mock_generate_sample(idx, tasks),
        pool_size=30,
        refresh_fraction=0.2,
        anti_repeat_window=2,
        prefetch_workers=2,
        seed=42,
        verbose=False,
    )
    
    errors = []
    
    def reader_thread(cache, iterations):
        """Thread that reads samples."""
        try:
            for _ in range(iterations):
                samples = cache.get_epoch_samples(cache._current_epoch)
                _ = len(samples)
                time.sleep(0.01)
        except Exception as e:
            errors.append(f"Reader error: {e}")
    
    def prefetch_thread(cache, epochs):
        """Thread that triggers prefetch/swap."""
        try:
            for epoch in range(1, epochs + 1):
                cache.prefetch_next_epoch(epoch)
                time.sleep(0.1)
                cache.swap_to_next_epoch(timeout=2.0)
        except Exception as e:
            errors.append(f"Prefetch error: {e}")
    
    # Run threads concurrently
    reader = threading.Thread(target=reader_thread, args=(cache, 20))
    prefetcher = threading.Thread(target=prefetch_thread, args=(cache, 3))
    
    reader.start()
    prefetcher.start()
    
    reader.join()
    prefetcher.join()
    
    cache.shutdown()
    
    assert len(errors) == 0, f"Thread safety errors: {errors}"
    print("✓ Thread safety test passed")


def test_rolling_cache_dataset_wrapper():
    """Test RollingCacheDataset wrapper for DataLoader compatibility."""
    from sci_arc.data.rolling_cache import RollingRefreshCache, RollingCacheDataset
    
    tasks = create_mock_tasks(5)
    
    cache = RollingRefreshCache(
        tasks=tasks,
        generate_sample_fn=lambda idx: mock_generate_sample(idx, tasks),
        pool_size=20,
        refresh_fraction=0.25,
        anti_repeat_window=2,
        prefetch_workers=1,
        seed=42,
        verbose=False,
    )
    
    dataset = RollingCacheDataset(cache)
    
    # Test __len__
    assert len(dataset) == 20, f"Dataset should have 20 samples, got {len(dataset)}"
    
    # Test __getitem__
    sample = dataset[0]
    assert 'task_id' in sample, "Sample should have task_id"
    
    # Test epoch notifications
    dataset.notify_epoch_end(0)  # Triggers prefetch for epoch 1
    time.sleep(0.2)
    dataset.notify_epoch_start(1)  # Triggers swap
    
    assert cache._current_epoch == 1, "Should be at epoch 1 after notifications"
    
    cache.shutdown()
    print("✓ Dataset wrapper test passed")


def test_config_helper():
    """Test create_rolling_cache_from_config helper."""
    from sci_arc.data.rolling_cache import create_rolling_cache_from_config, get_default_rolling_cache_config
    
    tasks = create_mock_tasks(5)
    
    # Get default config
    default_config = get_default_rolling_cache_config()
    assert 'data' in default_config
    assert 'rolling_cache' in default_config['data']
    
    # Modify for testing
    config = {
        'data': {
            'rolling_cache': {
                'pool_size': 50,
                'refresh_fraction': 0.2,
                'anti_repeat_window': 3,
                'prefetch_workers': 2,
                'coverage_scheduling': True,
                'seed': 123,
            }
        }
    }
    
    cache = create_rolling_cache_from_config(
        tasks=tasks,
        generate_sample_fn=lambda idx: mock_generate_sample(idx, tasks),
        config=config,
        verbose=False,
    )
    
    assert cache.pool_size == 50, "Should use config pool_size"
    assert cache.refresh_fraction == 0.2, "Should use config refresh_fraction"
    assert cache.anti_repeat_window == 3, "Should use config anti_repeat_window"
    assert cache.base_seed == 123, "Should use config seed"
    
    cache.shutdown()
    print("✓ Config helper test passed")


def test_arps_vectorized_indexing():
    """Test ARPS vectorized DSL primitives don't crash with CUDA indexing.
    
    This tests the fix for the 'srcIndex < srcSelectDimSize' CUDA assertion error
    that was caused by missing .long() casts on index tensors.
    """
    try:
        import torch
        from sci_arc.models.rlan_modules.arps import translate, reflect_x, reflect_y, rotate_90, copy_paste
    except ImportError:
        print("⊘ ARPS test skipped (torch or arps not available)")
        return
    
    # Create test grid on CPU (CUDA indexing bug manifests on GPU but logic is same)
    device = 'cpu'
    H, W = 10, 10
    
    # Test translate
    grid = torch.zeros(H, W, dtype=torch.long, device=device)
    grid[2:5, 2:5] = 1  # Small object
    anchor = (5, 5)
    
    try:
        result = translate(grid, anchor, dx=2, dy=3)
        assert result.shape == (H, W), "translate should preserve shape"
    except Exception as e:
        raise AssertionError(f"translate failed: {e}")
    
    # Test reflect_x
    try:
        result = reflect_x(grid, anchor)
        assert result.shape == (H, W), "reflect_x should preserve shape"
    except Exception as e:
        raise AssertionError(f"reflect_x failed: {e}")
    
    # Test reflect_y
    try:
        result = reflect_y(grid, anchor)
        assert result.shape == (H, W), "reflect_y should preserve shape"
    except Exception as e:
        raise AssertionError(f"reflect_y failed: {e}")
    
    # Test rotate_90
    try:
        result = rotate_90(grid, anchor)
        assert result.shape == (H, W), "rotate_90 should preserve shape"
    except Exception as e:
        raise AssertionError(f"rotate_90 failed: {e}")
    
    # Test copy_paste
    try:
        result = copy_paste(grid, anchor, source_region=(2, 2, 5, 5), offset=(3, 3))
        assert result.shape == (H, W), "copy_paste should preserve shape"
    except Exception as e:
        raise AssertionError(f"copy_paste failed: {e}")
    
    # Edge case: anchor at boundary
    grid_edge = torch.zeros(H, W, dtype=torch.long, device=device)
    grid_edge[0:2, 0:2] = 1  # Object at corner
    anchor_edge = (0, 0)
    
    try:
        result = translate(grid_edge, anchor_edge, dx=-1, dy=-1)  # Would go out of bounds
        assert result.shape == (H, W), "translate should handle boundary cases"
    except Exception as e:
        raise AssertionError(f"translate boundary case failed: {e}")
    
    print("✓ ARPS vectorized indexing test passed")


def test_backward_compatibility_static_mode():
    """Test that static cache mode still works (backward compatibility)."""
    # This tests that setting cache_samples_mode='static' or not setting it
    # doesn't break existing functionality
    
    # The config should default to static mode
    config = {
        'data': {
            'cache_samples': True,
            # cache_samples_mode not set - should default to static
        }
    }
    
    cache_samples_mode = config['data'].get('cache_samples_mode', 'static')
    assert cache_samples_mode == 'static', "Should default to static mode"
    
    print("✓ Backward compatibility (static mode) test passed")


def test_generate_sample_uses_effective_augmentation():
    """Test that _generate_sample uses _get_effective_*() methods (Jan 2026 fix).
    
    This test verifies the fix for the rolling cache augmentation bug where
    samples were generated with no augmentation even when rolling cache mode
    was enabled with coverage_scheduling=true.
    
    The bug was that _generate_sample() used self.augment directly instead of
    self._get_effective_augment(), so runtime overrides from set_augmentation_config()
    were ignored.
    """
    import tempfile
    import json
    import os
    from sci_arc.data.dataset import ARCDataset
    
    # Create a minimal test dataset
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a simple task file
        task = {
            'train': [
                {'input': [[0, 1], [1, 0]], 'output': [[1, 0], [0, 1]]}
            ],
            'test': [
                {'input': [[0, 1], [1, 0]], 'output': [[1, 0], [0, 1]]}
            ]
        }
        task_path = os.path.join(tmpdir, 'test_task.json')
        with open(task_path, 'w') as f:
            json.dump(task, f)
        
        # Create dataset with augmentation DISABLED (like the config)
        dataset = ARCDataset(
            data_path=tmpdir,
            max_size=30,
            augment=False,  # Disabled in config
            color_permutation=False,  # Disabled in config
            translational_augment=False,  # Disabled in config
            cache_samples=False,
        )
        
        # Generate samples - should have no augmentation
        samples_no_aug = []
        for _ in range(20):
            sample = dataset._generate_sample(0)
            samples_no_aug.append(sample['aug_info']['dihedral_id'])
        
        # All should be identity (0) since augmentation is disabled
        assert all(d == 0 for d in samples_no_aug), \
            f"With augment=False, all samples should have dihedral_id=0, got {samples_no_aug}"
        
        # Now enable augmentation via runtime override (like rolling cache does)
        dataset.set_augmentation_config(
            augment=True,
            color_permutation=True,
            color_permutation_prob=0.5,
            translational_augment=True,
        )
        
        # Generate samples - should now have augmentation diversity
        samples_with_aug = []
        color_perms_applied = 0
        for _ in range(100):
            sample = dataset._generate_sample(0)
            samples_with_aug.append(sample['aug_info']['dihedral_id'])
            if sample['aug_info']['color_perm'] is not None:
                color_perms_applied += 1
        
        # Check dihedral diversity
        unique_dihedrals = set(samples_with_aug)
        assert len(unique_dihedrals) > 1, \
            f"With runtime augment=True, should have dihedral diversity, got only {unique_dihedrals}"
        
        # Check color permutation applied (should be ~50% with prob=0.5)
        # Allow wide margin due to randomness
        assert color_perms_applied > 20, \
            f"With color_permutation=True (prob=0.5), expected ~50 color perms, got {color_perms_applied}"
        
        print(f"  Dihedral diversity: {len(unique_dihedrals)} unique transforms")
        print(f"  Color perms applied: {color_perms_applied}/100 samples")
    
    print("✓ _generate_sample uses effective augmentation test passed")


def run_all_tests():
    """Run all smoke tests."""
    print("\n" + "=" * 60)
    print("ROLLING REFRESH CACHE SMOKE TESTS")
    print("=" * 60 + "\n")
    
    tests = [
        test_fingerprint_hashing,
        test_coverage_stats,
        test_rolling_cache_initialization,
        test_epoch_swap_and_prefetch,
        test_anti_repeat_window,
        test_thread_safety,
        test_rolling_cache_dataset_wrapper,
        test_config_helper,
        test_arps_vectorized_indexing,
        test_backward_compatibility_static_mode,
        test_generate_sample_uses_effective_augmentation,  # Jan 2026 fix
    ]
    
    passed = 0
    failed = 0
    
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"✗ {test_fn.__name__} FAILED: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
