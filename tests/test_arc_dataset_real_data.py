from __future__ import annotations

from pathlib import Path

import pytest
import torch


@pytest.mark.cpu
def test_arcdataset_loads_real_training_dir_and_has_expected_keys_and_types():
    from sci_arc.data.dataset import ARCDataset

    train_dir = Path("data/arc-agi/data/training")
    assert train_dir.is_dir(), "Expected ARC-AGI training dir under data/"

    ds = ARCDataset(
        data_path=str(train_dir),
        max_size=30,
        augment=False,
        translational_augment=False,
        color_permutation=False,
        ignore_padding_in_loss=True,
        max_tasks=5,
    )

    assert len(ds) > 0
    sample = ds[0]
    assert isinstance(sample, dict)

    # Minimal schema expectations
    for k in ["task_id", "input_grids", "output_grids", "test_input", "test_output", "num_train_pairs"]:
        assert k in sample, f"Missing key {k}"

    assert isinstance(sample["task_id"], str)
    assert isinstance(sample["num_train_pairs"], int) and sample["num_train_pairs"] >= 1

    assert isinstance(sample["test_input"], torch.Tensor)
    assert isinstance(sample["test_output"], torch.Tensor)
    # Indices may be int32 or int64; model embedding can accept either.
    assert sample["test_input"].dtype in (torch.int32, torch.int64)
    assert sample["test_output"].dtype in (torch.int32, torch.int64)


@pytest.mark.cpu
def test_arcdataset_padding_uses_ignore_index_minus_100_for_targets():
    from sci_arc.data.dataset import ARCDataset

    train_dir = Path("data/arc-agi/data/training")
    ds = ARCDataset(
        data_path=str(train_dir),
        max_size=30,
        augment=False,
        translational_augment=False,
        color_permutation=False,
        ignore_padding_in_loss=True,
        max_tasks=25,
    )

    # ARCDataset returns max_size x max_size tensors; confirm padding sentinel appears
    # in at least one sample.
    found = False
    for i in range(min(50, len(ds))):
        s = ds[i]
        if (s["test_output"] == -100).any().item():
            found = True
            break
    assert found, "Expected at least one sample to contain -100 ignore_index padding"


@pytest.mark.cpu
def test_arcdataset_input_dtype_consistent_with_targets():
    from sci_arc.data.dataset import ARCDataset

    train_dir = Path("data/arc-agi/data/training")
    ds = ARCDataset(
        data_path=str(train_dir),
        max_size=30,
        augment=False,
        translational_augment=False,
        color_permutation=False,
        ignore_padding_in_loss=True,
        max_tasks=5,
    )

    s = ds[0]
    assert s["test_input"].dtype == s["test_output"].dtype
