import pytest
import torch
import numpy as np
import os
import json
from sci_arc.data.dataset import ARCDataset

class TestDatasetConfigBugs:
    def setup_method(self):
        # Create a dummy task file
        self.dummy_task_path = "tests/data/dummy_task.json"
        os.makedirs("tests/data", exist_ok=True)
        with open(self.dummy_task_path, "w") as f:
            json.dump({
                "dummy_task": {
                    "train": [{"input": [[1, 2], [3, 4]], "output": [[1, 1], [1, 1]]}],
                    "test": [{"input": [[1, 2], [3, 4]], "output": [[1, 1], [1, 1]]}]
                }
            }, f)

    def teardown_method(self):
        if os.path.exists(self.dummy_task_path):
            os.remove(self.dummy_task_path)

    def test_color_permutation_prob_zero_no_permutation(self):
        """
        Verify that color_permutation_prob=0.0 prevents color permutation.
        """
        dataset = ARCDataset(
            data_path=self.dummy_task_path,
            max_size=4,  # Small size for easier verification
            augment=False,  # Disable dihedral so we can check colors directly
            color_permutation=True,
            color_permutation_prob=0.0,  # Should NEVER permute
            translational_augment=False,
            cache_samples=False
        )
        
        permuted_count = 0
        num_trials = 30
        
        for _ in range(num_trials):
            sample = dataset[0]
            # Check first 2x2 region (the actual data, not padding)
            input_grid = sample['input_grids'][0][:2, :2]
            original_vals = set([1, 2, 3, 4])
            actual_vals = set(input_grid.flatten().tolist())
            
            if actual_vals != original_vals:
                permuted_count += 1
                
        print(f"Permuted count with prob=0.0: {permuted_count}/{num_trials}")
        assert permuted_count == 0, f"Color permutation happened {permuted_count} times despite prob=0.0"

    def test_color_permutation_prob_one_always_permutes(self):
        """
        Verify that color_permutation_prob=1.0 always applies color permutation.
        """
        dataset = ARCDataset(
            data_path=self.dummy_task_path,
            max_size=4,
            augment=False,
            color_permutation=True,
            color_permutation_prob=1.0,  # Should ALWAYS permute
            translational_augment=False,
            cache_samples=False
        )
        
        permuted_count = 0
        num_trials = 30
        
        for _ in range(num_trials):
            sample = dataset[0]
            input_grid = sample['input_grids'][0][:2, :2]
            original_vals = set([1, 2, 3, 4])
            actual_vals = set(input_grid.flatten().tolist())
            
            # Color permutation changes values (unless identity perm, which is 1/362880)
            if actual_vals != original_vals:
                permuted_count += 1
                
        print(f"Permuted count with prob=1.0: {permuted_count}/{num_trials}")
        # Should be at least 90% permuted (identity permutation is rare)
        assert permuted_count >= num_trials * 0.9, f"Only permuted {permuted_count}/{num_trials} times with prob=1.0"

    def test_translational_augment_disabled_preserves_position(self):
        """
        Verify that translational_augment=False keeps data at consistent position.
        When disabled, the data should always be at position (0, 0) after padding.
        """
        dataset = ARCDataset(
            data_path=self.dummy_task_path,
            max_size=10,
            augment=False,
            color_permutation=False,
            translational_augment=False,  # Should NOT translate
            cache_samples=False
        )
        
        # All samples should have data at the same position
        positions_vary = False
        reference_grid = None
        
        for _ in range(20):
            sample = dataset[0]
            grid = sample['input_grids'][0]
            
            if reference_grid is None:
                reference_grid = grid
            elif not torch.equal(grid, reference_grid):
                positions_vary = True
                break
                
        assert not positions_vary, "Position varied despite translational_augment=False"

    def test_translational_augment_enabled_varies_position(self):
        """
        Verify that translational_augment=True varies the data position.
        """
        dataset = ARCDataset(
            data_path=self.dummy_task_path,
            max_size=10,
            augment=False,
            color_permutation=False,
            translational_augment=True,  # Should translate
            cache_samples=False
        )
        
        # With translational augmentation, we should see varied positions
        grids = []
        for _ in range(30):
            sample = dataset[0]
            grids.append(sample['input_grids'][0].clone())
        
        # Check if positions vary (not all grids identical)
        all_same = all(torch.equal(g, grids[0]) for g in grids)
        print(f"All grids same with translational_augment=True: {all_same}")
        # With prob=0.3 for translation and 30 samples, we expect variation
        assert not all_same, "All positions identical despite translational_augment=True"
