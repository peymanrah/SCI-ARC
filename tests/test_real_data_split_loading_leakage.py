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


@pytest.mark.xfail(
    strict=True,
    reason=(
        "SCIARCDataset._load_tasks silently falls back to data_dir when split subdir is missing; "
        "passing data_dir=<.../training> with split='evaluation' leaks training tasks into evaluation. "
        "This should raise or produce empty dataset to prevent accidental leakage."
    ),
)
def test_sciarcdataset_should_not_silently_fallback_when_split_missing():
    data_root = _arc_agi_data_root()
    if not data_root.exists():
        pytest.skip(f"Real ARC-AGI data not found at: {data_root}")

    training_dir = data_root / "training"
    evaluation_dir = data_root / "evaluation"
    assert training_dir.exists() and evaluation_dir.exists()

    # This is an easy user error: passing the split directory as data_dir.
    ds = SCIARCDataset(str(training_dir), split="evaluation", augment=False)

    eval_ids_on_disk = _list_task_ids(evaluation_dir)
    loaded_ids = {t.task_id for t in ds.tasks}

    # Expected behavior (safer): should NOT load training tasks here.
    # Current behavior: will fall back to data_dir and load training tasks.
    assert loaded_ids.issubset(eval_ids_on_disk)
