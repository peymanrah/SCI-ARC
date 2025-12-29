import os
from pathlib import Path

import pytest

from sci_arc.data.dataset import SCIARCDataset


def _arc_agi_data_root() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / "data" / "arc-agi" / "data"


def _list_task_ids(task_dir: Path) -> set[str]:
    return {p.stem for p in task_dir.glob("*.json")}


def test_sciarcdataset_split_loads_only_expected_directory():
    data_root = _arc_agi_data_root()
    if not data_root.exists():
        pytest.skip(f"Real ARC-AGI data not found at: {data_root}")

    train_dir = data_root / "training"
    eval_dir = data_root / "evaluation"
    assert train_dir.exists(), f"Missing: {train_dir}"
    assert eval_dir.exists(), f"Missing: {eval_dir}"

    train_ids_on_disk = _list_task_ids(train_dir)
    eval_ids_on_disk = _list_task_ids(eval_dir)
    assert train_ids_on_disk, "No training JSON tasks found"
    assert eval_ids_on_disk, "No evaluation JSON tasks found"

    train_ds = SCIARCDataset(str(data_root), split="training", augment=False)
    eval_ds = SCIARCDataset(str(data_root), split="evaluation", augment=False)

    train_ids_loaded = {t.task_id for t in train_ds.tasks}
    eval_ids_loaded = {t.task_id for t in eval_ds.tasks}

    # Strongest signal: loaded IDs should be subsets of the correct on-disk split.
    assert train_ids_loaded.issubset(train_ids_on_disk)
    assert eval_ids_loaded.issubset(eval_ids_on_disk)

    # And should not accidentally include IDs from the other split.
    assert (train_ids_loaded & eval_ids_on_disk) == set()
    assert (eval_ids_loaded & train_ids_on_disk) == set()


def test_sciarcdataset_raises_on_missing_split_directory():
    """Verify SCIARCDataset raises FileNotFoundError when split subdir is missing.
    
    This prevents accidental data leakage from misconfigured paths.
    """
    data_root = _arc_agi_data_root()
    if not data_root.exists():
        pytest.skip(f"Real ARC-AGI data not found at: {data_root}")

    training_dir = data_root / "training"
    assert training_dir.exists()

    # This is an easy user error: passing the split directory as data_dir.
    # Should raise FileNotFoundError, not silently load wrong data.
    with pytest.raises(FileNotFoundError, match="Split directory.*not found"):
        SCIARCDataset(str(training_dir), split="evaluation", augment=False)
