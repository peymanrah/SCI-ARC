"""
Pytest configuration and shared fixtures.
"""

import pytest
import torch
import numpy as np
import json
import tempfile
from pathlib import Path


@pytest.fixture(scope="session")
def device():
    """Get available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def sample_arc_task():
    """Create a sample ARC task dict."""
    return {
        'train': [
            {
                'input': [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                'output': [[9, 8, 7], [6, 5, 4], [3, 2, 1]],
            },
            {
                'input': [[0, 1], [2, 3]],
                'output': [[3, 2], [1, 0]],
            },
        ],
        'test': [
            {
                'input': [[1, 0], [0, 1]],
                'output': [[1, 0], [0, 1]],
            },
        ],
    }


@pytest.fixture
def temp_arc_dataset(sample_arc_task):
    """Create a temporary ARC dataset directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create training split
        training_dir = Path(tmpdir) / 'training'
        training_dir.mkdir()
        
        # Create evaluation split
        eval_dir = Path(tmpdir) / 'evaluation'
        eval_dir.mkdir()
        
        # Create sample tasks
        for i in range(3):
            with open(training_dir / f'task_{i}.json', 'w') as f:
                json.dump(sample_arc_task, f)
        
        # One evaluation task
        with open(eval_dir / 'eval_task.json', 'w') as f:
            json.dump(sample_arc_task, f)
        
        yield tmpdir


@pytest.fixture
def sample_batch():
    """Create a sample batch for model testing."""
    batch_size = 2
    num_pairs = 3
    H, W = 10, 10
    num_colors = 10
    
    return {
        'task_ids': ['task_0', 'task_1'],
        'input_grids': torch.randint(0, num_colors, (batch_size, num_pairs, H, W)),
        'output_grids': torch.randint(0, num_colors, (batch_size, num_pairs, H, W)),
        'test_inputs': torch.randint(0, num_colors, (batch_size, H, W)),
        'test_outputs': torch.randint(0, num_colors, (batch_size, H, W)),
        'transform_families': torch.tensor([0, 1]),
        'num_pairs': torch.tensor([num_pairs, num_pairs]),
        'grid_masks': torch.ones(batch_size, num_pairs, dtype=torch.bool),
    }


@pytest.fixture
def small_model_config():
    """Create a small model config for fast testing."""
    from sci_arc.models import SCIARCConfig
    
    return SCIARCConfig(
        hidden_dim=32,
        num_colors=10,
        max_grid_size=15,
        num_structure_slots=2,
        se_layers=1,
        use_abstraction=True,
        max_objects=4,
        H_cycles=1,
        L_cycles=1,
        L_layers=1,
        dropout=0.0,
    )


@pytest.fixture
def small_model(small_model_config):
    """Create a small model for testing."""
    from sci_arc.models import SCIARC
    
    return SCIARC(small_model_config)


@pytest.fixture
def data_dir():
    """Get the path to the ARC-AGI data directory."""
    data_path = Path(__file__).parent.parent / "data" / "arc-agi" / "data"
    if not data_path.exists():
        pytest.skip(f"Data directory not found: {data_path}")
    return data_path


def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests as requiring GPU"
    )


def pytest_collection_modifyitems(config, items):
    """Skip GPU tests if CUDA not available."""
    if not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(reason="CUDA not available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)
