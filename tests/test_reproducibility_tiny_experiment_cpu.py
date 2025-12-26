from __future__ import annotations

import hashlib
import random
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
import numpy as np


def _hash_tensor(t: torch.Tensor) -> str:
    # Stable hash for exact equality checks
    b = t.detach().cpu().numpy().tobytes()
    return hashlib.sha256(b).hexdigest()


@pytest.mark.cpu
def test_tiny_cpu_experiment_reproducible_when_seeds_fixed_and_augment_disabled():
    """Reality check: same seed + augment=False => identical first-step loss + logits hash."""
    from sci_arc.data.dataset import ARCDataset
    from sci_arc.models import RLAN, RLANConfig

    train_dir = Path("data/arc-agi/data/training")
    assert train_dir.is_dir()

    def run_once(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)

        ds = ARCDataset(
            data_path=str(train_dir),
            max_size=30,
            augment=False,
            translational_augment=False,
            color_permutation=False,
            ignore_padding_in_loss=True,
            max_tasks=3,
        )

        s = ds[0]
        # Build minimal batch tensors
        test_input = s["test_input"].unsqueeze(0)
        train_inputs = torch.stack(s["input_grids"], dim=0).unsqueeze(0)
        train_outputs = torch.stack(s["output_grids"], dim=0).unsqueeze(0)

        config = RLANConfig(
            hidden_dim=64,
            num_solver_steps=1,
            dropout=0.0,
            use_hyperlora=False,
            use_dsc=False,
            use_msre=False,
            use_lcr=False,
            use_sph=False,
            use_context_encoder=True,
            use_cross_attention_context=False,
        )
        model = RLAN(config=config)
        model.eval()

        with torch.no_grad():
            logits = model(
                input_grid=test_input,
                train_inputs=train_inputs,
                train_outputs=train_outputs,
                pair_mask=torch.ones(1, train_inputs.shape[1], dtype=torch.bool),
            )

        # Loss against the provided test_output (ignore padding)
        target = s["test_output"].unsqueeze(0)
        loss = F.cross_entropy(logits, target, ignore_index=-100)
        return float(loss.item()), _hash_tensor(logits)

    loss1, h1 = run_once(123)
    loss2, h2 = run_once(123)

    assert loss1 == loss2
    assert h1 == h2
