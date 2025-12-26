import subprocess
import sys
from pathlib import Path

import pytest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _script_path(rel: str) -> Path:
    p = _repo_root() / rel
    assert p.exists(), f"Missing script: {p}"
    return p


@pytest.mark.slow
def test_train_rlan_script_help_runs_on_cpu():
    script = _script_path("scripts/train_rlan.py")
    proc = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]


@pytest.mark.slow
def test_evaluate_rlan_script_help_runs_on_cpu():
    script = _script_path("scripts/evaluate_rlan.py")
    proc = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]


@pytest.mark.slow
def test_train_rlan_script_checkpoint_helpers_roundtrip_cpu(tmp_path: Path):
    """Validates the *script* checkpoint format (not just model methods).

    Runs in a subprocess to avoid polluting pytest process with train_rlan.py signal handlers.
    """

    script = _script_path("scripts/train_rlan.py")
    ckpt = tmp_path / "ckpt.pt"

    code = f"""
import importlib.util
from pathlib import Path

import torch

spec = importlib.util.spec_from_file_location('train_rlan_script', r'{script.as_posix()}')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

from sci_arc.models import RLAN, RLANConfig

cfg = RLANConfig(
    hidden_dim=16,
    num_solver_steps=1,
    use_context_encoder=False,
    use_dsc=False,
    use_msre=False,
    use_lcr=False,
    use_sph=False,
    use_act=False,
    dropout=0.0,
)
model = RLAN(config=cfg)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

# Save
mod.save_checkpoint(
    model=model,
    optimizer=opt,
    scheduler=None,
    epoch=3,
    global_step=123,
    losses={{'total': 1.23}},
    best_accuracy=0.42,
    config={{'dummy': True}},
    path=r'{ckpt.as_posix()}',
)
assert Path(r'{ckpt.as_posix()}').exists()

# Load
start_epoch, global_step, best_acc = mod.load_checkpoint(
    model=model,
    optimizer=opt,
    scheduler=None,
    path=r'{ckpt.as_posix()}',
    reset_optimizer=False,
    reset_scheduler=False,
)
assert start_epoch == 4
assert global_step == 123
assert abs(best_acc - 0.42) < 1e-9
"""

    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
