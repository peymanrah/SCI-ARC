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


@pytest.mark.xfail(
    reason=(
        "SCIARCDataset indexes by task, not by (task,test_idx). For tasks with multiple ARC test items, "
        "the dataset provides only a single sample per task and selects a random test pair, so an evaluation "
        "pass cannot deterministically cover all official test items."
    ),
    strict=True,
)
def test_dataset_should_represent_all_test_items_not_just_first():
    data_root = Path("data/arc-agi/data")
    training_dir = data_root / "training"

    task_fp = _find_task_with_multiple_tests(training_dir)
    task_id = task_fp.stem

    with open(task_fp, "r", encoding="utf-8") as f:
        task = json.load(f)

    # Ground truth: there are multiple tests
    assert len(task["test"]) > 1

    ds = SCIARCDataset(str(data_root), split="training", augment=False)

    # Dataset has exactly one entry per task_id, regardless of number of test items.
    occurrences = sum(1 for t in ds.tasks if t.task_id == task_id)

    # Scientific requirement (for deterministic evaluation): one dataset item per test input.
    # This intentionally fails today (occurrences is 1).
    assert occurrences == len(task["test"])
