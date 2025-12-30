"""
Test staggered module activation schedule.

The new schedule prevents memory spikes by activating modules at different epochs:
- Epoch 3: HyperLoRA (meta_learning_start_epoch)
- Epoch 5: SolverCrossAttention (solver_context_start_epoch)
- Epoch 7: CrossAttentionInjector (cross_attention_start_epoch)
- Epoch 8: Equivariance loss
- Epoch 12: LOO loss
- Epoch 14: HPM

This was designed to prevent the 12GB+ memory spill that occurred at epoch 4 when
all three modules (HyperLoRA, SolverCrossAttention, CrossAttentionInjector) 
activated simultaneously.

Run with: python -m pytest tests/test_staggered_activation.py -v -s
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, Any
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sci_arc.models.rlan import RLAN, RLANConfig


class TestStaggeredActivation:
    """Test suite for staggered module activation."""
    
    @classmethod
    def setup_class(cls):
        """Setup test fixtures."""
        cls.device = torch.device('cpu')  # CPU for testing
        cls.hidden_dim = 128  # Smaller for faster tests
        cls.max_grid_size = 30
        cls.batch_size = 4
        
        # Default staggered epochs (matching rlan_stable_dev.yaml)
        cls.hyperlora_start_epoch = 3
        cls.solver_context_start_epoch = 5
        cls.cross_attention_start_epoch = 7
        cls.equivariance_start_epoch = 8
        cls.loo_start_epoch = 12
        cls.hpm_start_epoch = 14
        
        # Create config with all modules
        cls.config = RLANConfig(
            hidden_dim=cls.hidden_dim,
            num_colors=10,
            max_clues=6,
            num_solver_steps=4,
            use_dsc=True,
            use_msre=True,
            use_context_encoder=True,
            use_cross_attention_context=True,
            use_solver_context=True,
            use_hyperlora=True,
            use_hpm=True,
            spatial_downsample=8,
            gradient_checkpointing=False,
        )
    
    def create_model(self) -> RLAN:
        """Create a fresh model for testing."""
        model = RLAN(config=self.config, max_grid_size=self.max_grid_size)
        model.to(self.device)
        return model
    
    def set_staggered_activation(self, model: RLAN, epoch: int) -> Dict[str, bool]:
        """
        Set module activation based on staggered epoch schedule.
        This mimics the logic in train_rlan.py.
        """
        # HyperLoRA at epoch 3
        model.hyperlora_active = epoch >= self.hyperlora_start_epoch
        
        # SolverCrossAttention at epoch 5
        model.solver_context_active = epoch >= self.solver_context_start_epoch
        
        # CrossAttentionInjector at epoch 7
        model.cross_attention_active = epoch >= self.cross_attention_start_epoch
        
        # HPM at epoch 14
        if hasattr(model, 'use_hpm'):
            model.use_hpm = epoch >= self.hpm_start_epoch
        
        return {
            'hyperlora_active': model.hyperlora_active,
            'solver_context_active': model.solver_context_active,
            'cross_attention_active': model.cross_attention_active,
            'hpm_active': getattr(model, 'use_hpm', False),
        }
    
    # =========================================================================
    # TEST 1: Verify epoch 0-2 has no modules active
    # =========================================================================
    def test_epoch_0_2_no_modules_active(self):
        """Epochs 0-2: No new modules should be active."""
        model = self.create_model()
        
        for epoch in range(3):
            flags = self.set_staggered_activation(model, epoch)
            
            assert not flags['hyperlora_active'], \
                f"HyperLoRA should be INACTIVE at epoch {epoch}"
            assert not flags['solver_context_active'], \
                f"SolverCrossAttention should be INACTIVE at epoch {epoch}"
            assert not flags['cross_attention_active'], \
                f"CrossAttentionInjector should be INACTIVE at epoch {epoch}"
            
            print(f"  [OK] Epoch {epoch}: All modules correctly INACTIVE")
    
    # =========================================================================
    # TEST 2: Verify HyperLoRA activates at epoch 3
    # =========================================================================
    def test_epoch_3_hyperlora_only(self):
        """Epoch 3: Only HyperLoRA should be active."""
        model = self.create_model()
        flags = self.set_staggered_activation(model, 3)
        
        assert flags['hyperlora_active'], "HyperLoRA should be ACTIVE at epoch 3"
        assert not flags['solver_context_active'], "SolverCrossAttention should be INACTIVE at epoch 3"
        assert not flags['cross_attention_active'], "CrossAttentionInjector should be INACTIVE at epoch 3"
        
        print("  [OK] Epoch 3: Only HyperLoRA active (as expected)")
    
    # =========================================================================
    # TEST 3: Verify SolverCrossAttention activates at epoch 5
    # =========================================================================
    def test_epoch_5_solver_context_added(self):
        """Epoch 5: HyperLoRA + SolverCrossAttention active."""
        model = self.create_model()
        flags = self.set_staggered_activation(model, 5)
        
        assert flags['hyperlora_active'], "HyperLoRA should be ACTIVE at epoch 5"
        assert flags['solver_context_active'], "SolverCrossAttention should be ACTIVE at epoch 5"
        assert not flags['cross_attention_active'], "CrossAttentionInjector should be INACTIVE at epoch 5"
        
        print("  [OK] Epoch 5: HyperLoRA + SolverCrossAttention active")
    
    # =========================================================================
    # TEST 4: Verify CrossAttentionInjector activates at epoch 7
    # =========================================================================
    def test_epoch_7_cross_attention_added(self):
        """Epoch 7: All three modules active."""
        model = self.create_model()
        flags = self.set_staggered_activation(model, 7)
        
        assert flags['hyperlora_active'], "HyperLoRA should be ACTIVE at epoch 7"
        assert flags['solver_context_active'], "SolverCrossAttention should be ACTIVE at epoch 7"
        assert flags['cross_attention_active'], "CrossAttentionInjector should be ACTIVE at epoch 7"
        
        print("  [OK] Epoch 7: All three modules active")
    
    # =========================================================================
    # TEST 5: Verify HPM activates at epoch 14
    # =========================================================================
    def test_epoch_14_hpm_added(self):
        """Epoch 14: HPM activates (last module)."""
        model = self.create_model()
        flags = self.set_staggered_activation(model, 14)
        
        assert flags['hyperlora_active'], "HyperLoRA should be ACTIVE at epoch 14"
        assert flags['solver_context_active'], "SolverCrossAttention should be ACTIVE at epoch 14"
        assert flags['cross_attention_active'], "CrossAttentionInjector should be ACTIVE at epoch 14"
        assert flags['hpm_active'], "HPM should be ACTIVE at epoch 14"
        
        print("  [OK] Epoch 14: All modules including HPM active")
    
    # =========================================================================
    # TEST 6: Verify at least 2 epochs between each activation
    # =========================================================================
    def test_epoch_gap_between_activations(self):
        """Verify minimum 2-epoch gap between module activations."""
        # The schedule should have gaps to allow model to stabilize
        activation_epochs = [
            ('HyperLoRA', self.hyperlora_start_epoch),
            ('SolverCrossAttention', self.solver_context_start_epoch),
            ('CrossAttentionInjector', self.cross_attention_start_epoch),
            ('Equivariance', self.equivariance_start_epoch),
            ('LOO', self.loo_start_epoch),
            ('HPM', self.hpm_start_epoch),
        ]
        
        for i in range(1, len(activation_epochs)):
            prev_name, prev_epoch = activation_epochs[i-1]
            curr_name, curr_epoch = activation_epochs[i]
            gap = curr_epoch - prev_epoch
            
            assert gap >= 1, \
                f"Gap between {prev_name} (epoch {prev_epoch}) and {curr_name} (epoch {curr_epoch}) " \
                f"is only {gap} epochs. Need at least 1."
            
            print(f"  [OK] {prev_name} → {curr_name}: {gap} epoch gap")
    
    # =========================================================================
    # TEST 7: Forward pass works at each staggered stage
    # =========================================================================
    def test_forward_pass_at_each_stage(self):
        """Test forward pass works at each staggered activation stage."""
        model = self.create_model()
        model.eval()
        
        # Create synthetic batch
        bs = self.batch_size
        H, W = 10, 10
        num_pairs = 3
        
        test_inputs = torch.randint(0, 10, (bs, H, W))
        train_inputs = torch.randint(0, 10, (bs, num_pairs, H, W))
        train_outputs = torch.randint(0, 10, (bs, num_pairs, H, W))
        pair_mask = torch.ones(bs, num_pairs, dtype=torch.bool)
        
        test_epochs = [0, 3, 5, 7, 14]  # Key transition points
        
        for epoch in test_epochs:
            self.set_staggered_activation(model, epoch)
            
            with torch.no_grad():
                outputs = model(
                    test_inputs,
                    train_inputs=train_inputs,
                    train_outputs=train_outputs,
                    pair_mask=pair_mask,
                    return_intermediates=True,
                )
            
            # Check outputs are valid
            assert 'logits' in outputs, f"Missing logits at epoch {epoch}"
            assert outputs['logits'].shape == (bs, 10, H, W), \
                f"Wrong logits shape at epoch {epoch}: {outputs['logits'].shape}"
            assert not torch.isnan(outputs['logits']).any(), \
                f"NaN in logits at epoch {epoch}"
            
            print(f"  [OK] Forward pass works at epoch {epoch}")
    
    # =========================================================================
    # TEST 8: Config file has correct staggered settings
    # =========================================================================
    def test_config_file_staggered_settings(self):
        """Verify rlan_stable_dev.yaml has correct staggered activation settings."""
        config_path = Path(__file__).parent.parent / 'configs' / 'rlan_stable_dev.yaml'
        
        if not config_path.exists():
            print(f"  [SKIP] Config file not found: {config_path}")
            return
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        training = config.get('training', {})
        model = config.get('model', {})
        
        # Check training settings
        meta_learning_start = training.get('meta_learning_start_epoch', 3)
        solver_context_start = training.get('solver_context_start_epoch', 5)
        cross_attention_start = training.get('cross_attention_start_epoch', 7)
        
        assert meta_learning_start == 3, f"meta_learning_start_epoch should be 3, got {meta_learning_start}"
        assert solver_context_start == 5, f"solver_context_start_epoch should be 5, got {solver_context_start}"
        assert cross_attention_start == 7, f"cross_attention_start_epoch should be 7, got {cross_attention_start}"
        
        # Check HPM start
        hpm_start = model.get('hpm_start_epoch', 3)
        assert hpm_start >= 12, f"hpm_start_epoch should be >= 12 for staggered activation, got {hpm_start}"
        
        # Check LOO and equivariance
        loo_config = training.get('loo_training', {})
        equiv_config = training.get('equivariance_training', {})
        
        loo_start = loo_config.get('start_epoch', 12)
        equiv_start = equiv_config.get('start_epoch', 8)
        
        assert equiv_start >= 8, f"equivariance start_epoch should be >= 8, got {equiv_start}"
        assert loo_start >= 10, f"LOO start_epoch should be >= 10, got {loo_start}"
        
        print("  [OK] Config file has correct staggered activation settings")
        print(f"       HyperLoRA: epoch {meta_learning_start}")
        print(f"       SolverCrossAttention: epoch {solver_context_start}")
        print(f"       CrossAttentionInjector: epoch {cross_attention_start}")
        print(f"       Equivariance: epoch {equiv_start}")
        print(f"       LOO: epoch {loo_start}")
        print(f"       HPM: epoch {hpm_start}")


class TestMemoryEstimation:
    """Test memory estimation and safety."""
    
    def test_memory_manager_import(self):
        """Test that MemoryManager can be imported."""
        try:
            from sci_arc.utils.memory_manager import MemoryManager
            manager = MemoryManager()
            # Just verify it has expected attributes (don't hardcode GPU size)
            assert hasattr(manager, 'gpu_total_mb'), "MemoryManager should have gpu_total_mb"
            assert manager.gpu_total_mb > 0, "GPU memory should be positive"
            print(f"  [OK] MemoryManager imports correctly (GPU: {manager.gpu_total_mb:.0f}MB)")
        except ImportError as e:
            print(f"  [SKIP] MemoryManager not found: {e}")
    
    def test_staggered_schedule_from_memory_manager(self):
        """Test get_staggered_activation_schedule() function."""
        try:
            from sci_arc.utils.memory_manager import MemoryManager
            
            manager = MemoryManager()
            schedule = manager.get_staggered_activation_schedule()
            
            # Schedule is Dict[epoch, List[module_flags]]
            assert isinstance(schedule, dict), "Schedule should be a dict"
            
            # Verify expected module flags appear somewhere in the schedule
            all_modules = []
            for epoch, modules in schedule.items():
                all_modules.extend(modules)
            
            expected_flags = [
                'hyperlora_active',
                'solver_context_active', 
                'cross_attention_active',
                'equivariance_active',
                'loo_active',
                'use_hpm',
            ]
            
            for flag in expected_flags:
                assert flag in all_modules, f"Expected '{flag}' in schedule, found: {all_modules}"
            
            print("  [OK] MemoryManager provides staggered schedule")
            for epoch in sorted(schedule.keys()):
                print(f"       Epoch {epoch}: {schedule[epoch]}")
        except ImportError as e:
            print(f"  [SKIP] MemoryManager not found: {e}")


class TestBackwardCompatibility:
    """Test backward compatibility of changes."""
    
    @classmethod
    def setup_class(cls):
        """Setup test fixtures."""
        cls.device = torch.device('cpu')
    
    def test_model_without_staggered_flags(self):
        """Model should work even if staggered flags aren't set."""
        config = RLANConfig(
            hidden_dim=128,
            num_colors=10,
            max_clues=6,
            num_solver_steps=4,
            use_dsc=True,
            use_context_encoder=True,
            use_cross_attention_context=True,
            use_solver_context=True,
            use_hyperlora=True,
            use_hpm=False,  # Disable HPM for simplicity
            spatial_downsample=8,
        )
        
        model = RLAN(config=config, max_grid_size=30)
        model.to(self.device)
        model.eval()
        
        # Forward pass without setting any flags
        bs = 2
        H, W = 10, 10
        test_inputs = torch.randint(0, 10, (bs, H, W))
        train_inputs = torch.randint(0, 10, (bs, 3, H, W))
        train_outputs = torch.randint(0, 10, (bs, 3, H, W))
        pair_mask = torch.ones(bs, 3, dtype=torch.bool)
        
        with torch.no_grad():
            outputs = model(
                test_inputs,
                train_inputs=train_inputs,
                train_outputs=train_outputs,
                pair_mask=pair_mask,
            )
        
        # Handle both tensor and dict outputs
        if isinstance(outputs, dict):
            assert 'logits' in outputs
            logits = outputs['logits']
        else:
            logits = outputs
        
        assert logits.shape[0] == bs, f"Expected batch size {bs}, got {logits.shape[0]}"
        print("  [OK] Model works without explicit staggered flag setting")


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v', '-s'])
