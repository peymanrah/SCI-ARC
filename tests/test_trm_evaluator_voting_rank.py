import numpy as np
import pytest

from sci_arc.evaluation.trm_style_evaluator import TRMStyleEvaluator


def test_trm_style_voting_should_account_for_vote_count_not_only_avg_confidence():
    """
    Verify TRMStyleEvaluator voting ranks by total_confidence (count * avg).
    This was a bug that has been fixed: voting now ranks by total confidence, not just avg.
    """
    evaluator = TRMStyleEvaluator(pass_Ks=[1], use_voting=True, pad_value=10)

    # Ground truth should match candidate A.
    gt = np.array([[1, 1], [1, 1]], dtype=np.int64)

    # Candidate A: appears 3x with slightly lower avg confidence
    cand_a = gt.copy()
    for conf in (0.80, 0.80, 0.80):
        evaluator.update(
            task_id="t",
            prediction=cand_a,
            ground_truth=gt,
            aug_info={"dihedral_id": 0, "color_perm": None},
            confidence=conf,
        )

    # Candidate B: appears 1x with higher avg confidence
    cand_b = np.array([[2, 2], [2, 2]], dtype=np.int64)
    evaluator.update(
        task_id="t",
        prediction=cand_b,
        ground_truth=gt,
        aug_info={"dihedral_id": 0, "color_perm": None},
        confidence=0.90,
    )

    # If voting accounts for count, Pass@1 should be 1.0.
    # Current implementation ranks by avg confidence only -> candidate B wins -> Pass@1=0.0.
    metrics = evaluator.compute_metrics()
    assert metrics["pass@1"] == 1.0
