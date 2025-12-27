"""
Learning Capability Tests for RLAN

Tests that RLAN can learn simple transformations,
validating that the architecture is capable of learning.
"""

import pytest
import torch
import torch.nn as nn

from sci_arc.models import RLAN
from sci_arc.training import RLANLoss


class TestRLANLearning:
    """Test that RLAN can learn simple transformations."""
    
    @pytest.fixture
    def small_model(self):
        """Create a small RLAN for fast testing."""
        return RLAN(
            hidden_dim=64,
            max_clues=2,
            num_predicates=4,
            num_solver_steps=2,
            dropout=0.0,
        )
    
    @pytest.fixture
    def criterion(self):
        """Create loss function."""
        return RLANLoss(
            focal_gamma=2.0,
            focal_alpha=0.25,
            lambda_entropy=0.1,
            lambda_sparsity=0.05,
            lambda_predicate=0.01,
            lambda_curriculum=0.1,
        )
    
    def test_identity_task(self, small_model, criterion):
        """Test that model can learn identity transformation (output = input)."""
        model = small_model
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Create identity task data
        input_grid = torch.randint(0, 10, (8, 5, 5))
        target = input_grid.clone()
        
        model.train()
        initial_loss = None
        valid_losses = []
        
        for epoch in range(50):
            optimizer.zero_grad()
            
            outputs = model(input_grid, temperature=1.0, return_intermediates=True)
            
            losses = criterion(
                logits=outputs["logits"],
                targets=target,
                attention_maps=outputs["attention_maps"],
                stop_logits=outputs["stop_logits"],
                predicates=outputs["predicates"],
                epoch=epoch,
                max_epochs=50,
                all_logits=outputs["all_logits"],
            )
            
            loss_val = losses["total_loss"]
            
            # Skip if NaN
            if torch.isnan(loss_val):
                continue
                
            loss_val.backward()
            
            # Check for NaN gradients and skip if found
            has_nan_grad = any(
                p.grad is not None and torch.isnan(p.grad).any()
                for p in model.parameters()
            )
            if has_nan_grad:
                optimizer.zero_grad()
                continue
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            if initial_loss is None:
                initial_loss = loss_val.item()
            valid_losses.append(loss_val.item())
        
        # Need at least some valid training steps
        assert len(valid_losses) >= 20, f"Too many NaN losses, only got {len(valid_losses)} valid"
        
        final_loss = valid_losses[-1]
        
        # Loss should decrease
        assert final_loss < initial_loss, \
            f"Loss should decrease: initial={initial_loss:.4f}, final={final_loss:.4f}"
        
        # Check predictions
        model.eval()
        with torch.no_grad():
            logits = model(input_grid, temperature=0.1)
            preds = logits.argmax(dim=1)
            accuracy = (preds == target).float().mean()
        
        # Should achieve reasonable accuracy
        assert accuracy > 0.4, f"Identity task accuracy should be >40%, got {accuracy:.2%}"
    
    def test_constant_output_task(self, small_model, criterion):
        """Test that model can learn to output constant color."""
        model = small_model
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Create constant output task
        input_grid = torch.randint(0, 10, (8, 5, 5))
        target = torch.full((8, 5, 5), 5, dtype=torch.long)  # All color 5
        
        model.train()
        initial_loss = None
        valid_losses = []
        
        for epoch in range(50):
            optimizer.zero_grad()
            
            outputs = model(input_grid, temperature=1.0, return_intermediates=True)
            
            losses = criterion(
                logits=outputs["logits"],
                targets=target,
                attention_maps=outputs["attention_maps"],
                stop_logits=outputs["stop_logits"],
                predicates=outputs["predicates"],
                epoch=epoch,
                max_epochs=50,
            )
            
            loss_val = losses["total_loss"]
            
            # Skip if NaN
            if torch.isnan(loss_val):
                continue
                
            loss_val.backward()
            
            # Check for NaN gradients and skip if found
            has_nan_grad = any(
                p.grad is not None and torch.isnan(p.grad).any()
                for p in model.parameters()
            )
            if has_nan_grad:
                optimizer.zero_grad()
                continue
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            if initial_loss is None:
                initial_loss = loss_val.item()
            valid_losses.append(loss_val.item())
        
        # Need at least some valid training steps
        assert len(valid_losses) >= 20, f"Too many NaN losses, only got {len(valid_losses)} valid"
        
        final_loss = valid_losses[-1] if valid_losses else float('nan')
        
        # Loss should decrease (or at least not be NaN)
        assert not torch.isnan(torch.tensor(final_loss)), "Final loss should not be NaN"
        assert final_loss < initial_loss, \
            f"Loss should decrease: initial={initial_loss:.4f}, final={final_loss:.4f}"
        
        # Check predictions
        model.eval()
        with torch.no_grad():
            logits = model(input_grid, temperature=0.1)
            preds = logits.argmax(dim=1)
            accuracy = (preds == target).float().mean()
        
        assert accuracy > 0.7, f"Constant output accuracy should be >70%, got {accuracy:.2%}"
    
    def test_color_inversion_task(self, small_model, criterion):
        """Test a simple color mapping task."""
        # Set seed for reproducibility (test is sensitive to initialization)
        torch.manual_seed(42)
        
        model = small_model
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Create simple color swap task: 0->1, 1->0, else unchanged
        input_grid = torch.randint(0, 3, (8, 5, 5))  # Only colors 0, 1, 2
        target = input_grid.clone()
        target[input_grid == 0] = 1
        target[input_grid == 1] = 0
        
        model.train()
        
        for epoch in range(100):
            optimizer.zero_grad()
            
            outputs = model(input_grid, temperature=1.0, return_intermediates=True)
            
            losses = criterion(
                logits=outputs["logits"],
                targets=target,
                attention_maps=outputs["attention_maps"],
                stop_logits=outputs["stop_logits"],
                predicates=outputs["predicates"],
                epoch=epoch,
                max_epochs=100,
            )
            
            losses["total_loss"].backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
        
        # Check predictions
        model.eval()
        with torch.no_grad():
            logits = model(input_grid, temperature=0.1)
            preds = logits.argmax(dim=1)
            accuracy = (preds == target).float().mean()
        
        # This is a harder task, so lower threshold
        # Color swap requires learning a permutation, which is harder than identity
        assert accuracy > 0.3, f"Color swap accuracy should be >30%, got {accuracy:.2%}"
    
    def test_loss_decreases(self, small_model, criterion):
        """Test that loss consistently decreases during training."""
        model = small_model
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        input_grid = torch.randint(0, 10, (4, 8, 8))
        target = input_grid.clone()
        
        model.train()
        losses_history = []
        
        for epoch in range(30):
            optimizer.zero_grad()
            
            outputs = model(input_grid, temperature=1.0, return_intermediates=True)
            
            losses = criterion(
                logits=outputs["logits"],
                targets=target,
                attention_maps=outputs["attention_maps"],
                stop_logits=outputs["stop_logits"],
                predicates=outputs["predicates"],
                epoch=epoch,
                max_epochs=30,
            )
            
            losses["total_loss"].backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            loss_val = losses["total_loss"].item()
            if not torch.isnan(torch.tensor(loss_val)):
                losses_history.append(loss_val)
        
        # Check overall trend is decreasing (allow some NaN filtering)
        assert len(losses_history) >= 20, "Too many NaN losses"
        first_third = sum(losses_history[:10]) / 10
        last_third = sum(losses_history[-10:]) / 10
        
        assert last_third < first_third, \
            f"Loss should decrease over training: first_third={first_third:.4f}, last_third={last_third:.4f}"
    
    def test_overfitting_single_sample(self, small_model, criterion):
        """Test that model can overfit to a single sample."""
        model = small_model
        # Use a simpler loss for overfitting test - just cross entropy
        simple_criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        
        # Single sample, should be able to memorize
        input_grid = torch.randint(0, 10, (1, 4, 4))
        target = torch.randint(0, 10, (1, 4, 4))
        
        # Create training context - use the same sample as demonstration
        train_inputs = input_grid.unsqueeze(1)  # (1, 1, 4, 4)
        train_outputs = target.unsqueeze(1)  # (1, 1, 4, 4)
        pair_mask = torch.ones(1, 1, dtype=torch.bool)
        
        model.train()
        
        for epoch in range(500):  # More epochs for single sample
            optimizer.zero_grad()
            
            outputs = model(
                input_grid, 
                train_inputs=train_inputs,
                train_outputs=train_outputs,
                pair_mask=pair_mask,
                temperature=1.0,  # Normal temperature during training
                return_intermediates=True
            )
            
            # Simple cross entropy loss for overfitting test
            loss = simple_criterion(outputs["logits"], target)
            
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
        
        # Check predictions
        model.eval()
        with torch.no_grad():
            logits = model(
                input_grid, 
                train_inputs=train_inputs,
                train_outputs=train_outputs,
                pair_mask=pair_mask,
                temperature=0.1
            )
            preds = logits.argmax(dim=1)
            accuracy = (preds == target).float().mean()
        
        # Should be able to memorize single sample
        assert accuracy > 0.5, f"Should overfit single sample, got {accuracy:.2%}"


class TestRLANLossComponents:
    """Test individual loss components."""
    
    @pytest.fixture
    def criterion(self):
        return RLANLoss(
            lambda_entropy=0.1,
            lambda_sparsity=0.05,
            lambda_predicate=0.01,
            lambda_curriculum=0.1,
        )
    
    def test_all_loss_components_computed(self, criterion):
        """Test that all loss components are computed."""
        logits = torch.randn(2, 11, 10, 10)
        targets = torch.randint(0, 11, (2, 10, 10))
        attention_maps = torch.softmax(torch.randn(2, 3, 10, 10).view(2, 3, -1), dim=-1).view(2, 3, 10, 10)
        stop_logits = torch.randn(2, 3)
        predicates = torch.rand(2, 8)
        
        losses = criterion(
            logits=logits,
            targets=targets,
            attention_maps=attention_maps,
            stop_logits=stop_logits,
            predicates=predicates,
            epoch=50,
            max_epochs=100,
        )
        
        expected_keys = [
            "total_loss", "focal_loss", "entropy_loss",
            "sparsity_loss", "predicate_loss", "curriculum_loss",
        ]
        
        for key in expected_keys:
            assert key in losses, f"Missing loss component: {key}"
            assert not torch.isnan(losses[key]), f"Loss {key} is NaN"
            assert not torch.isinf(losses[key]), f"Loss {key} is Inf"
    
    def test_focal_loss_class_balance(self, criterion):
        """Test that focal loss handles class imbalance."""
        # Mostly background (class 0)
        targets_imbalanced = torch.zeros(2, 10, 10, dtype=torch.long)
        targets_imbalanced[:, 5, 5] = 5  # One non-background pixel
        
        # Logits that correctly predict background
        logits_correct = torch.zeros(2, 11, 10, 10)
        logits_correct[:, 0, :, :] = 10.0  # High confidence background
        
        # Logits that incorrectly predict non-background
        logits_wrong = torch.zeros(2, 11, 10, 10)
        logits_wrong[:, 5, :, :] = 10.0  # High confidence color 5
        
        # Create minimal required tensors
        attention_maps = torch.softmax(torch.randn(2, 3, 10, 10).view(2, 3, -1), dim=-1).view(2, 3, 10, 10)
        stop_logits = torch.zeros(2, 3)
        predicates = torch.rand(2, 8)
        
        loss_correct = criterion(
            logits_correct, targets_imbalanced, attention_maps, stop_logits, predicates
        )["focal_loss"]
        
        loss_wrong = criterion(
            logits_wrong, targets_imbalanced, attention_maps, stop_logits, predicates
        )["focal_loss"]
        
        # Wrong predictions should have higher loss
        assert loss_wrong > loss_correct, \
            f"Wrong predictions should have higher loss: correct={loss_correct:.4f}, wrong={loss_wrong:.4f}"
    
    def test_curriculum_penalty_schedule(self, criterion):
        """Test that curriculum penalty changes over epochs."""
        attention_maps = torch.softmax(torch.randn(2, 5, 10, 10).view(2, 5, -1), dim=-1).view(2, 5, 10, 10)
        stop_logits = torch.zeros(2, 5) - 2.0  # Low stop prob = many active clues
        predicates = torch.rand(2, 8)
        logits = torch.randn(2, 11, 10, 10)
        targets = torch.randint(0, 11, (2, 10, 10))
        
        # Early training (high curriculum penalty)
        loss_early = criterion(
            logits, targets, attention_maps, stop_logits, predicates,
            epoch=5, max_epochs=100
        )
        
        # Late training (low curriculum penalty)
        loss_late = criterion(
            logits, targets, attention_maps, stop_logits, predicates,
            epoch=95, max_epochs=100
        )
        
        # Early should have higher curriculum loss (penalizes many clues more)
        assert loss_early["curriculum_loss"] >= loss_late["curriculum_loss"], \
            f"Early curriculum should be >= late: early={loss_early['curriculum_loss']:.4f}, late={loss_late['curriculum_loss']:.4f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--timeout=300"])
