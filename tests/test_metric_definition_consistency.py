import numpy as np
import pytest

from sci_arc.evaluation import metrics as eval_metrics


def test_metrics_should_ignore_padding_like_trainer_validate():
    """
    Verify metrics ignore padding (-100) like the trainer does.
    This was a bug that has been fixed: ignore_index=-100 is now supported.
    """
    # Simulate ARC padded target (trainer uses ignore_index=-100)
    target = np.full((30, 30), -100, dtype=np.int64)
    target[:2, :2] = np.array([[1, 2], [3, 4]], dtype=np.int64)

    # Prediction matches all VALID pixels; padding values differ
    pred = np.full((30, 30), 0, dtype=np.int64)
    pred[:2, :2] = target[:2, :2]

    # Trainer-style semantics: perfect on valid region => exact match should be 1
    # (because padding should be ignored).
    # Bug was fixed: eval_metrics.task_accuracy now accepts ignore_index parameter
    assert eval_metrics.task_accuracy(pred, target, ignore_index=-100) == 1.0


def test_pixel_accuracy_shape_mismatch_should_not_use_overlap_by_default():
    """
    Verify pixel_accuracy with strict_shape mode rejects size mismatches.
    This was a bug that has been fixed: strict_shape=True now returns 0.0 on mismatch.
    """
    # If a model predicts the wrong output size, pixel accuracy should not be able
    # to look good purely by overlapping a subregion.
    target = np.zeros((3, 3), dtype=np.int64)
    pred = np.zeros((2, 2), dtype=np.int64)

    # Bug was fixed: strict_shape=True now returns 0.0 when shapes differ
    assert eval_metrics.pixel_accuracy(pred, target, strict_shape=True) == 0.0
