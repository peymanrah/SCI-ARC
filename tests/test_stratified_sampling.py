import unittest
import random
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

# Mock ARCTask for testing
@dataclass
class ARCTask:
    task_id: str
    train_pairs: List[Tuple[np.ndarray, np.ndarray]]
    test_pairs: List[Tuple[np.ndarray, np.ndarray]]
    transform_family: int = -1
    metadata: dict = None

# Import the function to test
from sci_arc.data.dataset import stratified_sample_arc_tasks

class TestStratifiedSampling(unittest.TestCase):
    
    def create_mock_task(self, task_id, grid_size, num_pairs, size_change=False):
        """Create a mock task with specific properties."""
        # Create grids
        inp = np.zeros((grid_size, grid_size))
        out = np.zeros((grid_size, grid_size))
        
        if size_change:
            out = np.zeros((grid_size + 2, grid_size + 2))
            
        train_pairs = [(inp, out) for _ in range(num_pairs)]
        test_pairs = [(inp, out)]
        
        return ARCTask(
            task_id=task_id,
            train_pairs=train_pairs,
            test_pairs=test_pairs
        )
    
    def test_stratification_coverage(self):
        """Test that sampling covers all strata."""
        tasks = []
        
        # Create tasks for each stratum combination
        # 3 grid sizes * 3 pair counts * 2 change types = 18 strata
        grid_sizes = [5, 15, 25]  # small, medium, large
        pair_counts = [2, 4, 6]   # few, medium, many
        change_types = [False, True] # same, change
        
        # Create 10 tasks for each stratum (180 total)
        for gs in grid_sizes:
            for pc in pair_counts:
                for ct in change_types:
                    for i in range(10):
                        tid = f"task_{gs}_{pc}_{ct}_{i}"
                        tasks.append(self.create_mock_task(tid, gs, pc, ct))
        
        # Sample 25 tasks (approx 1-2 per stratum)
        sampled = stratified_sample_arc_tasks(tasks, n_samples=25, seed=42, verbose=True)
        
        self.assertEqual(len(sampled), 25)
        
        # Verify coverage
        # We expect high coverage of the 18 strata
        covered_strata = set()
        for t in sampled:
            # Re-derive stratum key
            gs = t.train_pairs[0][0].shape[0]
            pc = len(t.train_pairs)
            ct = t.train_pairs[0][0].shape != t.train_pairs[0][1].shape
            
            gs_bucket = "small" if gs <= 10 else "medium" if gs <= 20 else "large"
            pc_bucket = "few_shot" if pc <= 2 else "medium_shot" if pc <= 4 else "many_shot"
            ct_bucket = "size_change" if ct else "same_size"
            
            key = f"{gs_bucket}_{pc_bucket}_{ct_bucket}"
            covered_strata.add(key)
            
        print(f"\nCovered {len(covered_strata)}/18 strata with 25 samples")
        self.assertGreater(len(covered_strata), 15, "Should cover most strata")
        
    def test_determinism(self):
        """Test that sampling is deterministic with fixed seed."""
        tasks = [self.create_mock_task(f"t{i}", 10, 3) for i in range(100)]
        
        s1 = stratified_sample_arc_tasks(tasks, 10, seed=123, verbose=False)
        s2 = stratified_sample_arc_tasks(tasks, 10, seed=123, verbose=False)
        s3 = stratified_sample_arc_tasks(tasks, 10, seed=456, verbose=False)
        
        # Same seed -> same tasks
        self.assertEqual([t.task_id for t in s1], [t.task_id for t in s2])
        
        # Different seed -> different tasks (likely)
        self.assertNotEqual([t.task_id for t in s1], [t.task_id for t in s3])

if __name__ == '__main__':
    unittest.main()
