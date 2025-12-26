import numpy as np
import pytest
import torch

from sci_arc.evaluation.trm_style_evaluator import (
    dihedral_transform,
    inverse_dihedral_transform,
    inverse_color_permutation,
)
from sci_arc.models.rlan_modules.acw import apply_augmentation


@pytest.mark.parametrize("h,w", [(1, 1), (2, 3), (3, 2), (5, 5), (7, 4)])
@pytest.mark.parametrize("tid", list(range(8)))
def test_trm_dihedral_inverse_is_identity(h: int, w: int, tid: int):
    rng = np.random.default_rng(seed=tid * 1000 + h * 100 + w)
    grid = rng.integers(0, 10, size=(h, w), dtype=np.int64)

    aug = dihedral_transform(grid, tid)
    restored = inverse_dihedral_transform(aug, tid)

    np.testing.assert_array_equal(restored, grid)


@pytest.mark.parametrize(
    "aug_type,tid",
    [
        ("identity", 0),
        ("rotate_90", 1),
        ("rotate_180", 2),
        ("rotate_270", 3),
        ("flip_h", 4),
        ("flip_v", 5),
        ("transpose", 6),
        ("transpose_neg", 7),
    ],
)
@pytest.mark.parametrize("shape", [(9, 11), (10, 10), (3, 7)])
def test_acw_and_trm_dihedral_match_on_numpy_equivalent(aug_type: str, tid: int, shape):
    # Ensure the augmentation definitions are consistent across modules.
    rng = np.random.default_rng(seed=hash((aug_type, tid, shape)) % (2**32))
    grid = rng.integers(0, 10, size=shape, dtype=np.int64)

    # ACW operates on torch tensors.
    torch_grid = torch.from_numpy(grid)
    acw_aug = apply_augmentation(torch_grid, aug_type=aug_type, inverse=False).cpu().numpy()

    # TRM evaluator operates on numpy.
    trm_aug = dihedral_transform(grid, tid)

    np.testing.assert_array_equal(acw_aug, trm_aug)


@pytest.mark.parametrize(
    "aug_type",
    [
        "identity",
        "rotate_90",
        "rotate_180",
        "rotate_270",
        "flip_h",
        "flip_v",
        "transpose",
        "transpose_neg",
    ],
)
@pytest.mark.parametrize("ndim", [2, 3, 4])
def test_acw_apply_then_inverse_restores_tensor(aug_type: str, ndim: int):
    # Test all supported input ranks:
    # - (H, W)
    # - (B, H, W)
    # - (B, C, H, W)
    torch.manual_seed(0)

    if ndim == 2:
        x = torch.randint(0, 10, (7, 4), dtype=torch.long)
    elif ndim == 3:
        x = torch.randint(0, 10, (2, 7, 4), dtype=torch.long)
    else:
        x = torch.randint(0, 10, (2, 1, 7, 4), dtype=torch.long)

    y = apply_augmentation(x, aug_type=aug_type, inverse=False)
    z = apply_augmentation(y, aug_type=aug_type, inverse=True)

    assert z.shape == x.shape
    assert torch.equal(z, x)


def test_color_permutation_inverse_is_identity_for_random_perm():
    rng = np.random.default_rng(seed=123)

    # Create a random permutation over colors 0..9.
    perm = np.arange(10, dtype=np.int64)
    rng.shuffle(perm)

    # Create a random ARC grid.
    grid = rng.integers(0, 10, size=(9, 11), dtype=np.int64)

    # Apply forward perm: original -> augmented.
    aug = perm[grid]

    # Inverse should restore.
    restored = inverse_color_permutation(aug, perm)
    np.testing.assert_array_equal(restored, grid)
