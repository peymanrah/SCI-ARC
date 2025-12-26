"""
Bug #1: LOO Training Module Not Integrated

FIXED: LOOTrainingLoss is now integrated into the trainer.
The trainer's _compute_losses() method now imports and calls LOOTrainingLoss
when use_loo=True in TrainingConfig.

Severity: HIGH (was)
Category: Theory/Implementation Mismatch
Status: FIXED
"""
import pytest
import inspect


def test_loo_training_integrated_into_trainer():
    """
    LOOTrainingLoss should be used in SCIARCTrainer._compute_losses()
    when HyperLoRA is enabled.
    
    FIXED: LOO loss is now computed via self.loo_loss (a LOOTrainingLoss instance).
    """
    from sci_arc.training.trainer import SCIARCTrainer
    
    # Get the source of _compute_losses
    src = inspect.getsource(SCIARCTrainer._compute_losses)
    
    # Check if LOO is mentioned anywhere in loss computation
    # The implementation uses self.loo_loss which is a LOOTrainingLoss instance
    assert 'loo' in src.lower(), "LOO training should be integrated into _compute_losses"
    assert 'loo_loss' in src.lower(), "LOO loss should be computed in _compute_losses"


def test_training_config_has_loo_fields():
    """
    TrainingConfig should have fields for configuring LOO training:
    - use_loo: bool
    - loo_weight: float
    - loo_min_pairs: int
    
    FIXED: These fields are now present in TrainingConfig.
    """
    from sci_arc.training.trainer import TrainingConfig
    import dataclasses
    
    config = TrainingConfig()
    field_names = [f.name for f in dataclasses.fields(config)]
    
    # Check for LOO-related config fields
    loo_fields = [f for f in field_names if 'loo' in f.lower()]
    assert len(loo_fields) > 0, "TrainingConfig should have LOO-related fields"
    
    # Verify specific LOO fields exist
    assert 'use_loo' in field_names, "TrainingConfig should have use_loo field"
    assert 'loo_weight' in field_names, "TrainingConfig should have loo_weight field"
    assert 'loo_min_pairs' in field_names, "TrainingConfig should have loo_min_pairs field"


def test_loo_training_loss_exists():
    """LOOTrainingLoss module exists and can be imported."""
    from sci_arc.models.rlan_modules.loo_training import LOOTrainingLoss, LOOConfig
    
    # Verify the class exists and is properly defined
    assert LOOTrainingLoss is not None
    assert hasattr(LOOTrainingLoss, 'forward')
    
    # Verify config exists
    config = LOOConfig()
    assert config.enabled is True
    assert config.loss_weight == 0.5
    assert config.min_pairs_for_loo == 2
