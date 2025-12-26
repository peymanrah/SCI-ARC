import json
from pathlib import Path

import pytest


DATA_ROOT = Path("data/arc-agi/data")


def _load_task(fp: Path) -> dict:
    with open(fp, "r", encoding="utf-8") as f:
        return json.load(f)


def _is_rectangular(grid) -> bool:
    if not isinstance(grid, list) or not grid:
        return False
    if not all(isinstance(row, list) for row in grid):
        return False
    w = len(grid[0])
    return all(len(row) == w for row in grid)


def _iter_grids(task: dict):
    for section in ("train", "test"):
        pairs = task.get(section)
        if not isinstance(pairs, list):
            continue
        for pair in pairs:
            yield pair.get("input"), (section, "input")
            yield pair.get("output"), (section, "output")


def test_arc_agi_train_eval_task_id_sets_do_not_overlap():
    train_dir = DATA_ROOT / "training"
    eval_dir = DATA_ROOT / "evaluation"

    train_ids = {p.stem for p in train_dir.glob("*.json")}
    eval_ids = {p.stem for p in eval_dir.glob("*.json")}

    assert train_ids, "No training tasks found"
    assert eval_ids, "No evaluation tasks found"

    overlap = sorted(train_ids & eval_ids)
    assert overlap == [], f"Train/Eval task-id overlap found: {overlap[:10]}"


@pytest.mark.parametrize("split", ["training", "evaluation"])
def test_arc_agi_schema_and_value_ranges_on_small_sample(split: str):
    split_dir = DATA_ROOT / split
    files = sorted(split_dir.glob("*.json"))
    assert files, f"No tasks found in {split_dir}"

    # Keep test fast: sample first 25 tasks deterministically.
    for fp in files[:25]:
        task = _load_task(fp)
        assert "train" in task and "test" in task, f"Missing train/test in {fp.name}"
        assert isinstance(task["train"], list) and len(task["train"]) >= 1
        assert isinstance(task["test"], list) and len(task["test"]) >= 1

        for grid, (section, kind) in _iter_grids(task):
            assert _is_rectangular(grid), f"Non-rectangular {section}.{kind} grid in {fp.name}"
            # Validate value range (ARC-AGI is 0-9)
            for row in grid:
                for v in row:
                    assert isinstance(v, int), f"Non-int color in {fp.name}"
                    assert 0 <= v <= 9, f"Out-of-range color {v} in {fp.name}"
