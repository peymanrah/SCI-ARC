from __future__ import annotations

from pathlib import Path

import pytest
import torch


@pytest.mark.cpu
def test_cpu_forward_smoke_on_real_arc_batch_with_context():
    """End-to-end smoke: real data -> collate -> model forward (CPU-only)."""
    from sci_arc.data.dataset import SCIARCDataset, collate_sci_arc
    from sci_arc.models import RLAN, RLANConfig

    data_root = Path("data/arc-agi/data")
    assert (data_root / "training").is_dir(), "Expected ARC-AGI training folder under data/"

    ds = SCIARCDataset(str(data_root), split="training", augment=False, max_grid_size=30, expand_test_pairs=False)
    assert len(ds) > 0

    # Build a tiny batch (2 samples) from real data.
    samples = [ds[0], ds[min(1, len(ds) - 1)]]
    batch = collate_sci_arc(samples, max_grid_size=30)

    # Minimal model config for CPU test.
    config = RLANConfig(
        hidden_dim=64,
        num_colors=10,
        num_classes=10,
        max_grid_size=30,
        num_solver_steps=2,
        use_dsc=False,
        use_msre=False,
        use_lcr=False,
        use_sph=False,
        use_context_encoder=True,
        use_cross_attention_context=False,
        dropout=0.0,
        use_hyperlora=False,
    )
    model = RLAN(config=config)
    model.eval()

    with torch.no_grad():
        logits = model(
            input_grid=batch["test_inputs"],
            train_inputs=batch["input_grids"],
            train_outputs=batch["output_grids"],
            pair_mask=batch["grid_masks"],
        )

    assert logits.dim() == 4
    assert logits.shape[0] == batch["test_inputs"].shape[0]
    assert logits.shape[-2:] == batch["test_inputs"].shape[-2:]
    assert torch.isfinite(logits).all().item(), "Non-finite logits"


@pytest.mark.cpu
def test_collate_sci_arc_pads_target_grids_with_ignore_index():
    """Targets should be padded with -100 so loss/metrics can ignore padding."""
    from sci_arc.data.dataset import SCIARCDataset, collate_sci_arc

    data_root = Path("data/arc-agi/data")
    ds = SCIARCDataset(str(data_root), split="training", augment=False, max_grid_size=30)

    # Find a task whose test_output is not already 30x30 so padding must occur.
    sample = None
    for i in range(min(100, len(ds))):
        s = ds[i]
        h, w = s["test_output"].shape
        if h < 30 or w < 30:
            sample = s
            break
    assert sample is not None, "Could not find a small grid needing padding in first 100 tasks"

    # Force fixed 30x30 padding to ensure padding regions exist.
    # With dynamic padding enabled, effective_max can shrink to the sample size,
    # and then there will be no padded region (so no -100 to assert on).
    batch = collate_sci_arc([sample], max_grid_size=30, dynamic_padding=False)
    targets = batch["test_outputs"]
    assert (targets == -100).any().item(), "Expected -100 in padded target regions"


@pytest.mark.cpu
def test_evaluate_with_trm_style_contract_mismatch_with_rlan_model():
    """Strict repro: TRM-style eval helper currently cannot run with the shipped RLAN forward API."""
    from sci_arc.data.dataset import SCIARCDataset, collate_sci_arc
    from sci_arc.models import RLAN, RLANConfig
    from sci_arc.evaluation.trm_style_evaluator import evaluate_with_trm_style

    data_root = Path("data/arc-agi/data")
    ds = SCIARCDataset(str(data_root), split="training", augment=False, max_grid_size=30)
    batch = collate_sci_arc([ds[0]], max_grid_size=30)

    config = RLANConfig(
        hidden_dim=64,
        num_colors=10,
        num_classes=10,
        max_grid_size=30,
        num_solver_steps=1,
        use_dsc=False,
        use_msre=False,
        use_lcr=False,
        use_sph=False,
        use_context_encoder=True,
        use_cross_attention_context=False,
        dropout=0.0,
        use_hyperlora=False,
    )
    model = RLAN(config=config)

    class _OneBatch:
        def __iter__(self):
            yield {
                "input": batch["test_inputs"],
                "target": batch["test_outputs"],
                "task_id": batch["task_ids"],
                "aug_info": batch.get("aug_info", [{"dihedral_id": 0}]),
                "demos": {
                    "input_grids": batch["input_grids"],
                    "output_grids": batch["output_grids"],
                    "grid_masks": batch.get("grid_masks"),
                },
            }

    evaluate_with_trm_style(model, _OneBatch(), device="cpu", num_augmented_views=1)
