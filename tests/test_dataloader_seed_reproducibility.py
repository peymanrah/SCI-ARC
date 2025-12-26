from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.cpu
def test_create_dataloader_seed_reproducible_for_shuffle_when_augment_disabled():
    """With augment=False, seed should deterministically control shuffle order for the first batch."""
    from sci_arc.data.dataset import create_dataloader

    data_root = Path("data/arc-agi/data")
    assert data_root.is_dir(), "Expected data/arc-agi/data to exist"

    # Two loaders, same seed, num_workers=0 to stay CPU-only and simple.
    loader1 = create_dataloader(
        data_dir=str(data_root),
        split="training",
        batch_size=2,
        shuffle=True,
        augment=False,
        max_grid_size=30,
        num_workers=0,
        seed=123,
        cache_samples=False,
    )

    loader2 = create_dataloader(
        data_dir=str(data_root),
        split="training",
        batch_size=2,
        shuffle=True,
        augment=False,
        max_grid_size=30,
        num_workers=0,
        seed=123,
        cache_samples=False,
    )

    b1 = next(iter(loader1))
    b2 = next(iter(loader2))


    assert b1["task_ids"] == b2["task_ids"], "First batch task order differs despite identical seed"


@pytest.mark.cpu
@pytest.mark.xfail(strict=True, reason="Deterministic augmentations require seeding at iteration time, not just at dataloader creation; RNG state consumed during dataset init breaks reproducibility")
def test_create_dataloader_seed_insufficient_for_deterministic_augmentations_when_num_workers_zero():
    """Strict repro for augmentation non-determinism with num_workers=0."""
    from sci_arc.data.dataset import create_dataloader

    data_root = Path("data/arc-agi/data")
    assert data_root.is_dir(), "Expected data/arc-agi/data to exist"

    loader1 = create_dataloader(
        data_dir=str(data_root),
        split="training",
        batch_size=2,
        shuffle=True,
        augment=True,
        max_grid_size=30,
        num_workers=0,
        seed=123,
        cache_samples=False,
    )

    loader2 = create_dataloader(
        data_dir=str(data_root),
        split="training",
        batch_size=2,
        shuffle=True,
        augment=True,
        max_grid_size=30,
        num_workers=0,
        seed=123,
        cache_samples=False,
    )

    b1 = next(iter(loader1))
    b2 = next(iter(loader2))

    assert (b1["test_inputs"] == b2["test_inputs"]).all(), "Augmented batch differs despite identical seed"
