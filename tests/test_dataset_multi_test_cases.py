import json
from pathlib import Path

import pytest

from sci_arc.data.dataset import SCIARCDataset


def _find_task_with_multiple_tests(split_dir: Path) -> Path:
    for fp in sorted(split_dir.glob("*.json")):
        with open(fp, "r", encoding="utf-8") as f:
            task = json.load(f)
        if isinstance(task.get("test"), list) and len(task["test"]) > 1:
            return fp
    raise FileNotFoundError("No ARC task with multiple test cases found")


def test_evaluation_split_auto_expands_test_pairs():
    """Verify evaluation split automatically enables expand_test_pairs for deterministic coverage."""
    data_root = Path("data/arc-agi/data")
    
    # Find a task with multiple test cases
    eval_dir = data_root / "evaluation"
    task_fp = _find_task_with_multiple_tests(eval_dir)
    task_id = task_fp.stem
    
    with open(task_fp, "r", encoding="utf-8") as f:
        task = json.load(f)
    
    num_tests = len(task["test"])
    assert num_tests > 1, "Test requires a task with multiple test cases"
    
    # Create evaluation dataset - should auto-enable expand_test_pairs
    ds = SCIARCDataset(str(data_root), split="evaluation", augment=False)
    
    # Check that expand_test_pairs was auto-enabled
    assert ds.expand_test_pairs, "expand_test_pairs should be auto-enabled for evaluation split"
    
    # Check that the expanded index covers all test pairs for this task
    task_test_indices = [(task_idx, test_idx) for task_idx, test_idx in ds._expanded_index 
                         if ds.tasks[task_idx].task_id == task_id]
    
    assert len(task_test_indices) == num_tests, (
        f"Expected {num_tests} entries for task {task_id}, got {len(task_test_indices)}"
    )


def test_training_split_does_not_auto_expand():
    """Verify training split does NOT auto-enable expand_test_pairs (preserves random selection)."""
    data_root = Path("data/arc-agi/data")
    
    ds = SCIARCDataset(str(data_root), split="training", augment=False)
    
    # Training should NOT auto-expand (random test selection is fine for training)
    assert not ds.expand_test_pairs, "expand_test_pairs should NOT be auto-enabled for training split"
