from __future__ import annotations

import pytest
import torch


@pytest.mark.cpu
def test_predict_with_acw_color_perm_handles_ignore_index_targets():
    from sci_arc.models import RLAN, RLANConfig

    config = RLANConfig(
        hidden_dim=32,
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

    # Minimal batch with one support pair.
    input_grid = torch.zeros(1, 2, 2, dtype=torch.long)
    train_inputs = torch.zeros(1, 1, 2, 2, dtype=torch.long)

    # Include ignore_index in support outputs to simulate padded targets.
    train_outputs = torch.tensor([[[[-100, 1], [2, 3]]]], dtype=torch.long)

    # num_color_perms>1 forces the color permutation branch.
    model.predict_with_acw(
        input_grid,
        train_inputs=train_inputs,
        train_outputs=train_outputs,
        num_color_perms=2,
        temperature=0.5,
    )
