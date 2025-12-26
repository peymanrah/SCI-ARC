from __future__ import annotations

import inspect

import pytest


@pytest.mark.cpu
def test_scitrc_trainer_requires_forward_training_but_rlan_lacks_it():
    """Strict repro: the packaged trainer loop is incompatible with the shipped RLAN model API."""
    from sci_arc.models import RLAN, RLANConfig
    from sci_arc.training.trainer import SCIARCTrainer

    # Verify trainer hard-depends on forward_training.
    src = inspect.getsource(SCIARCTrainer.train_epoch)
    assert "forward_training" in src

    model = RLAN(config=RLANConfig(hidden_dim=32, num_solver_steps=1, dropout=0.0, use_hyperlora=False))

    # This assertion is expected to fail until RLAN implements forward_training
    # or trainer is updated to call the actual RLAN.forward(...) API.
    assert hasattr(model, "forward_training"), "RLAN is missing forward_training required by SCIARCTrainer"
