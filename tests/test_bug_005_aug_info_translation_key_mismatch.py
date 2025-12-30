import pytest
from pathlib import Path


@pytest.mark.cpu
def test_aug_info_translation_key_is_consistent_between_dataset_and_evaluator():
    """
    Both translation key formats are now supported for backward compatibility.
    
    This was fixed by:
    1. Adding offset_r/offset_c keys alongside translational_offset in dataset.py
    2. Adding get_translation_offset() helper in evaluator that handles both formats
    """
    repo_root = Path(__file__).resolve().parents[1]
    dataset_py = repo_root / "sci_arc" / "data" / "dataset.py"
    evaluator_py = repo_root / "sci_arc" / "evaluation" / "trm_style_evaluator.py"

    ds_text = dataset_py.read_text(encoding="utf-8")
    ev_text = evaluator_py.read_text(encoding="utf-8")

    # Dataset should provide both formats
    assert "translational_offset" in ds_text, "Dataset should use 'translational_offset'"
    assert "offset_r" in ds_text and "offset_c" in ds_text, \
        "Dataset should also provide 'offset_r' and 'offset_c' for compatibility"
    
    # Evaluator should have helper to handle both formats
    assert "get_translation_offset" in ev_text, \
        "Evaluator should have get_translation_offset helper to handle both key formats"
