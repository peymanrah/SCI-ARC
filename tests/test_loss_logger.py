"""
Tests for the LossLogger component.
"""

import pytest
import tempfile
import json
from pathlib import Path
import math

from sci_arc.training.loss_logger import LossLogger, LossStats


class TestLossStats:
    """Tests for LossStats dataclass."""
    
    def test_initial_state(self):
        """Test initial state of LossStats."""
        stats = LossStats(name="test")
        assert stats.current == 0.0
        assert stats.running_mean == 0.0
        assert stats.num_samples == 0
        assert stats.min_value == float('inf')
        assert stats.max_value == float('-inf')
    
    def test_single_update(self):
        """Test single value update."""
        stats = LossStats(name="test")
        stats.update(1.5)
        
        assert stats.num_samples == 1
        assert stats.current == 1.5
        assert stats.running_mean == 1.5
        assert stats.min_value == 1.5
        assert stats.max_value == 1.5
    
    def test_multiple_updates(self):
        """Test multiple value updates."""
        stats = LossStats(name="test")
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        for v in values:
            stats.update(v)
        
        assert stats.num_samples == 5
        assert stats.current == 5.0
        assert stats.min_value == 1.0
        assert stats.max_value == 5.0
        assert abs(stats.running_mean - 3.0) < 0.01
    
    def test_zero_detection(self):
        """Test zero value detection for collapse warning."""
        stats = LossStats(name="test")
        
        # Add 15 zeros and 5 non-zeros (75% zeros, >50% threshold)
        for _ in range(15):
            stats.update(0.0)
        for _ in range(5):
            stats.update(1.0)
        
        assert stats.is_zero_count == 15
        # >50% zeros should trigger warning
        warning = stats.get_collapse_warning()
        assert warning is not None
        assert "zero" in warning.lower()
    
    def test_nan_detection(self):
        """Test NaN value detection."""
        stats = LossStats(name="test")
        
        for _ in range(10):
            stats.update(1.0)
        stats.update(float('nan'))
        stats.update(float('nan'))
        
        assert stats.is_nan_count == 2
        # More than 1% NaN should trigger warning
        warning = stats.get_collapse_warning()
        assert warning is not None
        assert "NaN" in warning
    
    def test_ema_update(self):
        """Test exponential moving average update."""
        stats = LossStats(name="test", ema_alpha=0.5)
        
        stats.update(1.0)
        assert stats.ema_value == 1.0  # First value
        
        stats.update(3.0)
        # EMA = 0.5 * 3.0 + 0.5 * 1.0 = 2.0
        assert abs(stats.ema_value - 2.0) < 0.01
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        stats = LossStats(name="test")
        stats.update(1.5)
        
        d = stats.to_dict()
        assert "test/current" in d
        assert "test/ema" in d
        assert "test/mean" in d
        assert d["test/current"] == 1.5


class TestLossLogger:
    """Tests for LossLogger class."""
    
    def test_initialization(self):
        """Test LossLogger initialization."""
        logger = LossLogger(log_dir=None, enable_file_logging=False)
        
        assert logger.global_step == 0
        assert len(logger.stats) > 0
        assert 'total' in logger.stats
        assert 'task' in logger.stats
    
    def test_log_step(self):
        """Test step logging."""
        logger = LossLogger(log_dir=None, enable_file_logging=False, enable_console=False)
        
        losses = {
            'total': 2.5,
            'task': 2.0,
            'deep': 0.3,
            'scl': 0.1,
            'ortho': 0.1,
        }
        
        logger.log_step(losses, epoch=0, step=0, lr=1e-4)
        
        assert logger.global_step == 1
        assert logger.stats['total'].current == 2.5
        assert logger.stats['task'].current == 2.0
    
    def test_log_step_with_tensors(self):
        """Test step logging with tensor values."""
        import torch
        
        logger = LossLogger(log_dir=None, enable_file_logging=False, enable_console=False)
        
        losses = {
            'total': torch.tensor(2.5),
            'task': torch.tensor(2.0),
            'deep': torch.tensor(0.3),
        }
        
        logger.log_step(losses, epoch=0, step=0, lr=1e-4)
        
        assert logger.stats['total'].current == 2.5
    
    def test_epoch_summary(self):
        """Test epoch summary computation."""
        logger = LossLogger(log_dir=None, enable_file_logging=False, enable_console=False)
        
        # Log several steps
        for i in range(10):
            losses = {
                'total': 2.0 - 0.1 * i,  # Decreasing loss
                'task': 1.5 - 0.08 * i,
            }
            logger.log_step(losses, epoch=0, step=i, lr=1e-4)
        
        summary = logger.log_epoch_summary(epoch=0)
        
        assert 'epoch' in summary
        assert 'total_mean' in summary
        assert summary['epoch'] == 0
    
    def test_file_logging(self):
        """Test file logging writes to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = LossLogger(
                log_dir=tmpdir,
                enable_file_logging=True,
                enable_console=False
            )
            
            losses = {'total': 1.5, 'task': 1.0}
            logger.log_step(losses, epoch=0, step=0, lr=1e-4)
            
            # Check that log file was created
            log_files = list(Path(tmpdir).glob("loss_log_*.jsonl"))
            assert len(log_files) == 1
            
            # Check content
            with open(log_files[0], 'r') as f:
                entry = json.loads(f.readline())
                assert entry['loss_total'] == 1.5
                assert entry['loss_task'] == 1.0
    
    def test_health_check(self):
        """Test health check functionality."""
        logger = LossLogger(log_dir=None, enable_file_logging=False, enable_console=False)
        
        # Log many zero values (should trigger collapse warning)
        for i in range(20):
            losses = {
                'total': 0.0,
                'task': 0.0,
            }
            logger.log_step(losses, epoch=0, step=i, lr=1e-4)
        
        warnings = logger.check_health()
        assert len(warnings) > 0
    
    def test_trend_detection(self):
        """Test trend detection in loss values."""
        logger = LossLogger(log_dir=None, enable_file_logging=False, enable_console=False)
        
        # Log decreasing loss
        for i in range(50):
            losses = {'total': 5.0 - 0.08 * i}
            logger.log_step(losses, epoch=0, step=i, lr=1e-4)
        
        # Internal trend should be detected
        trend = logger._compute_trend()
        assert "↓" in trend  # Decreasing trend
    
    def test_get_stats_dict(self):
        """Test getting all stats as dictionary."""
        logger = LossLogger(log_dir=None, enable_file_logging=False, enable_console=False)
        
        losses = {'total': 2.5, 'task': 2.0}
        logger.log_step(losses, epoch=0, step=0, lr=1e-4)
        
        stats = logger.get_stats_dict()
        
        assert 'total' in stats
        assert 'total/current' in stats['total']
        assert stats['total']['total/current'] == 2.5


class TestLossLoggerIntegration:
    """Integration tests for LossLogger with actual training scenarios."""
    
    def test_full_training_simulation(self):
        """Simulate a full training run with multiple epochs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = LossLogger(
                log_dir=tmpdir,
                log_interval=5,
                enable_file_logging=True,
                enable_console=False
            )
            
            # Simulate 3 epochs with 20 steps each
            for epoch in range(3):
                for step in range(20):
                    # Simulate decreasing loss over training
                    base_loss = 5.0 - epoch * 1.0 - step * 0.02
                    losses = {
                        'total': base_loss,
                        'task': base_loss * 0.6,
                        'deep': base_loss * 0.2,
                        'entropy': base_loss * 0.1,
                        'sparsity': base_loss * 0.1,
                    }
                    logger.log_step(losses, epoch=epoch, step=step, lr=1e-4 * (0.9 ** epoch))
                
                logger.log_epoch_summary(epoch)
            
            # Verify logging worked
            log_files = list(Path(tmpdir).glob("loss_log_*.jsonl"))
            assert len(log_files) == 1
            
            with open(log_files[0], 'r') as f:
                lines = f.readlines()
                assert len(lines) == 60  # 3 epochs * 20 steps
            
            # Verify stats are reasonable
            assert logger.stats['total'].running_mean > 0
            assert logger.stats['total'].min_value < logger.stats['total'].max_value
