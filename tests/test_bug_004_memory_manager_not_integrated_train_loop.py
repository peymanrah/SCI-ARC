import pytest
from pathlib import Path


@pytest.mark.cpu
def test_train_script_should_use_memory_manager_for_safe_batch_sizing():
    """
    MemoryManager is now integrated into train_rlan.py for safe batch sizing.
    
    This was fixed by importing MemoryManager and using get_safe_batch_size() 
    to validate batch sizes before training starts.
    """
    repo_root = Path(__file__).resolve().parents[1]
    train_script = repo_root / "scripts" / "train_rlan.py"
    text = train_script.read_text(encoding="utf-8")

    assert "get_memory_manager(" in text or "MemoryManager(" in text, (
        "Expected training script to use MemoryManager/get_memory_manager for safe batch sizing, "
        "but no references were found."
    )
