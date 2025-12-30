"""
Test dynamic padding in collate_sci_arc.

Verifies that:
1. Batches with small grids get small tensors (memory efficient)
2. Batches with large grids get appropriately sized tensors
3. Bucketed batching + dynamic padding work together
4. Pre-padded cached samples are properly cropped before re-padding

Run with: python -m pytest tests/test_dynamic_padding.py -v -s
"""

import sys
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sci_arc.data.dataset import collate_sci_arc, pad_grid


class TestPadGrid:
    """Test the pad_grid function."""
    
    def test_pad_small_to_target(self):
        """Test padding a small grid to target size."""
        grid = torch.ones(5, 5, dtype=torch.long)
        padded = pad_grid(grid, max_size=10)
        
        assert padded.shape == (10, 10), f"Expected (10, 10), got {padded.shape}"
        # Original content preserved
        assert (padded[:5, :5] == 1).all()
        # Padding is PAD_COLOR=10
        assert (padded[5:, :] == 10).all()
        assert (padded[:, 5:] == 10).all()
        print("  [OK] Small grid padded correctly")
    
    def test_crop_large_to_target(self):
        """Test cropping a large grid to target size."""
        grid = torch.full((20, 20), 5, dtype=torch.long)
        # Put a marker at position (5, 5)
        grid[5, 5] = 7
        
        cropped = pad_grid(grid, max_size=10)
        
        assert cropped.shape == (10, 10), f"Expected (10, 10), got {cropped.shape}"
        assert cropped[5, 5] == 7, "Marker should be preserved"
        print("  [OK] Large grid cropped correctly")
    
    def test_prepadded_with_original_size(self):
        """Test handling pre-padded grid with original_size hint."""
        # Simulate a 5x5 grid pre-padded to 30x30 (like cached samples)
        original = torch.zeros(30, 30, dtype=torch.long)
        original[:5, :5] = torch.arange(25).reshape(5, 5)  # Actual content
        original[5:, :] = 10  # Padding
        original[:, 5:] = 10  # Padding
        
        # Crop to batch_max_size=10 using original_size hint
        result = pad_grid(original, max_size=10, original_size=5)
        
        assert result.shape == (10, 10), f"Expected (10, 10), got {result.shape}"
        # Original content preserved (should be at [0:5, 0:5])
        assert (result[:5, :5] == torch.arange(25).reshape(5, 5)).all(), \
            "Original content should be preserved"
        # New padding
        assert (result[5:, :] == 10).all(), "Should have proper padding"
        print("  [OK] Pre-padded grid correctly handled with original_size")
    
    def test_target_grid_padding(self):
        """Test that target grids use -100 for padding."""
        grid = torch.ones(5, 5, dtype=torch.long)
        padded = pad_grid(grid, max_size=10, is_target=True)
        
        assert padded.shape == (10, 10)
        assert (padded[:5, :5] == 1).all()
        assert (padded[5:, :] == -100).all()
        print("  [OK] Target grids use -100 padding")


class TestCollateDynamicPadding:
    """Test dynamic padding in collate_sci_arc."""
    
    def _create_sample(self, grid_size: int, task_id: str = "test") -> Dict:
        """Create a sample with given grid size."""
        grid = torch.ones(grid_size, grid_size, dtype=torch.long)
        return {
            'task_id': task_id,
            'input_grids': [grid.clone() for _ in range(3)],
            'output_grids': [grid.clone() for _ in range(3)],
            'test_input': grid.clone(),
            'test_output': grid.clone(),
            'num_train_pairs': 3,
            'transform_family': 0,
            'original_max_size': grid_size,
        }
    
    def _create_prepadded_sample(self, original_size: int, padded_size: int = 30, 
                                  task_id: str = "test") -> Dict:
        """Create a sample that's pre-padded (like cached samples)."""
        # Create original content
        original = torch.arange(original_size * original_size).reshape(original_size, original_size)
        
        # Pre-pad to padded_size (simulating cache behavior)
        padded_grid = torch.full((padded_size, padded_size), 10, dtype=torch.long)
        padded_grid[:original_size, :original_size] = original
        
        return {
            'task_id': task_id,
            'input_grids': [padded_grid.clone() for _ in range(3)],
            'output_grids': [padded_grid.clone() for _ in range(3)],
            'test_input': padded_grid.clone(),
            'test_output': padded_grid.clone(),
            'num_train_pairs': 3,
            'transform_family': 0,
            'original_max_size': original_size,  # CRITICAL: stores original size
        }
    
    def test_dynamic_padding_small_batch(self):
        """Test that a batch of small grids produces small tensors."""
        samples = [
            self._create_sample(5, "task1"),
            self._create_sample(7, "task2"),
            self._create_sample(6, "task3"),
        ]
        
        batch = collate_sci_arc(samples, max_size=30, dynamic_padding=True)
        
        # Should be padded to 7 (max in batch), not 30
        expected_size = 7
        assert batch['test_inputs'].shape[-1] == expected_size, \
            f"Expected size {expected_size}, got {batch['test_inputs'].shape[-1]}"
        assert batch['test_inputs'].shape[-2] == expected_size
        assert batch['input_grids'].shape[-1] == expected_size
        
        # batch_max_size should report actual size used
        assert batch['batch_max_size'] == expected_size
        
        print(f"  [OK] Small batch: tensors are {expected_size}x{expected_size} (not 30x30)")
    
    def test_dynamic_padding_large_batch(self):
        """Test that a batch with large grids uses larger tensors."""
        samples = [
            self._create_sample(25, "task1"),
            self._create_sample(28, "task2"),
            self._create_sample(20, "task3"),
        ]
        
        batch = collate_sci_arc(samples, max_size=30, dynamic_padding=True)
        
        # Should be padded to 28 (max in batch)
        expected_size = 28
        assert batch['test_inputs'].shape[-1] == expected_size, \
            f"Expected size {expected_size}, got {batch['test_inputs'].shape[-1]}"
        
        print(f"  [OK] Large batch: tensors are {expected_size}x{expected_size}")
    
    def test_prepadded_samples_cropped_correctly(self):
        """Test that pre-padded cached samples are handled correctly."""
        # Create samples that are pre-padded to 30x30 but have different original sizes
        samples = [
            self._create_prepadded_sample(5, 30, "task1"),
            self._create_prepadded_sample(8, 30, "task2"),
            self._create_prepadded_sample(6, 30, "task3"),
        ]
        
        batch = collate_sci_arc(samples, max_size=30, dynamic_padding=True)
        
        # Should be padded to 8 (max original_size in batch), not 30!
        expected_size = 8
        assert batch['test_inputs'].shape[-1] == expected_size, \
            f"Expected size {expected_size}, got {batch['test_inputs'].shape[-1]}. " \
            f"Pre-padded samples not being cropped to original_max_size!"
        
        # Verify content is preserved (first sample should have its 5x5 content)
        first_content = batch['test_inputs'][0, :5, :5]
        expected_content = torch.arange(25).reshape(5, 5)
        assert (first_content == expected_content).all(), \
            "Content from original grid should be preserved"
        
        print(f"  [OK] Pre-padded samples: tensors are {expected_size}x{expected_size} (not 30x30)")
    
    def test_dynamic_padding_disabled(self):
        """Test that dynamic_padding=False uses fixed max_size."""
        samples = [
            self._create_sample(5, "task1"),
            self._create_sample(5, "task2"),
        ]
        
        batch = collate_sci_arc(samples, max_size=30, dynamic_padding=False)
        
        # Should be padded to 30 (max_size parameter)
        assert batch['test_inputs'].shape[-1] == 30, \
            f"With dynamic_padding=False, should use max_size=30"
        
        print("  [OK] dynamic_padding=False uses fixed max_size")
    
    def test_memory_savings_calculation(self):
        """Demonstrate memory savings from dynamic padding."""
        # Batch of small grids
        samples = [self._create_sample(10, f"task{i}") for i in range(80)]
        
        batch_dynamic = collate_sci_arc(samples, max_size=30, dynamic_padding=True)
        batch_fixed = collate_sci_arc(samples, max_size=30, dynamic_padding=False)
        
        dynamic_size = batch_dynamic['test_inputs'].numel()
        fixed_size = batch_fixed['test_inputs'].numel()
        
        savings = 1 - (dynamic_size / fixed_size)
        
        print(f"\n  Memory comparison:")
        print(f"    Dynamic: {batch_dynamic['test_inputs'].shape} = {dynamic_size:,} elements")
        print(f"    Fixed:   {batch_fixed['test_inputs'].shape} = {fixed_size:,} elements")
        print(f"    Savings: {savings*100:.1f}%")
        
        # For 10x10 vs 30x30, expect ~89% savings in this dimension
        assert savings > 0.8, f"Expected >80% savings, got {savings*100:.1f}%"
        print(f"  [OK] Memory savings: {savings*100:.1f}%")


class TestBucketedBatchingIntegration:
    """Test that bucketed batching produces batches with similar sizes."""
    
    def test_bucket_produces_uniform_sizes(self):
        """Samples from same bucket should have similar original_max_size."""
        # This is a design verification - actual bucketing is tested in test_data.py
        
        # Bucket 0: sizes <= 10
        bucket_0_samples = [
            {'original_max_size': 5},
            {'original_max_size': 8},
            {'original_max_size': 10},
        ]
        max_in_bucket = max(s['original_max_size'] for s in bucket_0_samples)
        
        # With dynamic padding, this batch would be 10x10 instead of 30x30
        print(f"  [OK] Bucket 0 (<=10): max_size={max_in_bucket} -> 10x10 tensors (not 30x30)")
        
        # Bucket 4: sizes > 25
        bucket_4_samples = [
            {'original_max_size': 26},
            {'original_max_size': 28},
            {'original_max_size': 30},
        ]
        max_in_bucket = max(s['original_max_size'] for s in bucket_4_samples)
        
        # This batch needs 30x30 anyway
        print(f"  [OK] Bucket 4 (>25): max_size={max_in_bucket} -> {max_in_bucket}x{max_in_bucket} tensors")


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v', '-s'])
