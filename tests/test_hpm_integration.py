"""
Integration Tests for HPM with RLAN Model

Tests HPM integration with full RLAN model:
1. Forward pass with use_hpm=True
2. Forward pass with use_hpm=False (unchanged behavior)
3. Training loop integration
4. Memory efficiency (no VRAM growth)
5. Checkpoint save/load with HPM
6. Dynamic buffer population from task completions

Note: These tests require the full RLAN model and may take longer to run.
"""

import pytest
import torch
import torch.nn as nn
import gc
import yaml
from pathlib import Path

# Import RLAN components
try:
    from sci_arc.models.rlan import RLAN, RLANConfig
    from sci_arc.models.rlan_modules.hpm import HPMConfig, HierarchicalPrimitiveMemory
    from sci_arc.models.rlan_modules.dynamic_buffer import DynamicMemoryBuffer
    RLAN_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import RLAN: {e}")
    RLAN_AVAILABLE = False


# Skip all tests if RLAN is not available
pytestmark = pytest.mark.skipif(not RLAN_AVAILABLE, reason="RLAN not available")


class TestRLANHPMIntegration:
    """Tests for HPM integration with RLAN model."""
    
    def test_rlan_without_hpm(self):
        """Smoke test #9: RLAN with use_hpm=False unchanged."""
        config = RLANConfig(
            hidden_dim=64,
            num_solver_steps=2,
            dropout=0.0,
            use_hpm=False,
        )
        model = RLAN(config=config)
        model.eval()
        
        # Check HPM is not created
        assert model.hpm is None
        
        # Forward pass should work
        x = torch.randint(0, 10, (1, 16, 16))
        with torch.no_grad():
            logits = model(x)
        
        assert logits.shape == (1, 10, 16, 16)  # (batch, num_classes, H, W)
    
    def test_rlan_with_hpm_enabled(self):
        """Test RLAN with HPM enabled."""
        config = RLANConfig(
            hidden_dim=64,
            num_solver_steps=2,
            dropout=0.0,
            use_hpm=True,
            hpm_top_k=2,
            hpm_use_compositional_bank=True,
            hpm_use_pattern_bank=True,
        )
        model = RLAN(config=config)
        model.eval()
        
        # Check HPM is created
        assert model.hpm is not None
        assert isinstance(model.hpm, HierarchicalPrimitiveMemory)
        
        # Forward pass should work
        x = torch.randint(0, 10, (1, 16, 16))
        with torch.no_grad():
            logits = model(x)
        
        assert logits.shape == (1, 10, 16, 16)
    
    def test_hpm_gate_starts_at_zero(self):
        """Smoke test #5: Gate starts at 0 (no HPM contribution initially)."""
        config = RLANConfig(
            hidden_dim=64,
            use_hpm=True,
            hpm_use_compositional_bank=True,
            hpm_use_pattern_bank=True,
        )
        model = RLAN(config=config)
        
        gate_value = model.hpm.get_gate_value()
        assert abs(gate_value) < 1e-6
    
    def test_hpm_balance_loss_in_training(self):
        """Test HPM balance loss is computed during training."""
        config = RLANConfig(
            hidden_dim=64,
            num_solver_steps=2,
            use_hpm=True,
            hpm_use_compositional_bank=True,
            hpm_use_pattern_bank=True,
        )
        model = RLAN(config=config)
        model.train()
        
        # Reset epoch stats
        model.hpm_on_epoch_start()
        
        # Forward pass to accumulate routing stats
        x = torch.randint(0, 10, (4, 16, 16))
        _ = model(x)
        
        # Get balance loss
        balance_loss = model.hpm_get_load_balance_loss()
        
        assert balance_loss.item() >= 0
    
    def test_hpm_backward_callback(self):
        """Test HPM backward callback (gradient routing)."""
        config = RLANConfig(
            hidden_dim=64,
            num_solver_steps=2,
            use_hpm=True,
            hpm_use_compositional_bank=True,
            hpm_use_pattern_bank=True,
        )
        model = RLAN(config=config)
        model.train()
        
        x = torch.randint(0, 10, (1, 16, 16))
        logits = model(x)
        
        # Simulate loss and backward
        loss = logits.sum()
        loss.backward()
        
        # Callback should work without error
        model.hpm_on_backward()
    
    def test_hpm_stats(self):
        """Test HPM stats retrieval."""
        config = RLANConfig(
            hidden_dim=64,
            num_solver_steps=2,
            use_hpm=True,
            hpm_use_compositional_bank=True,
            hpm_use_pattern_bank=True,
        )
        model = RLAN(config=config)
        model.train()
        
        # Forward pass
        x = torch.randint(0, 10, (4, 16, 16))
        model(x)
        
        # Get stats
        stats = model.hpm_get_stats()
        
        assert 'gate_value' in stats
        # Check bank-specific stats exist
        assert any('bank_' in k for k in stats.keys())
    
    def test_hpm_with_dynamic_banks(self):
        """Test HPM with dynamic banks enabled."""
        config = RLANConfig(
            hidden_dim=64,
            num_solver_steps=2,
            use_hpm=True,
            hpm_use_compositional_bank=True,
            hpm_use_procedural_bank=True,
            hpm_use_instance_bank=True,
        )
        model = RLAN(config=config)
        model.eval()
        
        # Check dynamic buffers are created (with hpm_ prefix)
        assert hasattr(model, 'hpm_procedural_buffer')
        assert hasattr(model, 'hpm_instance_buffer')
        
        # Forward pass should work (empty buffers)
        x = torch.randint(0, 10, (1, 16, 16))
        with torch.no_grad():
            logits = model(x)
        
        assert logits.shape == (1, 10, 16, 16)
    
    def test_hpm_task_completion(self):
        """Smoke test #7: Dynamic buffer grows on task completion."""
        config = RLANConfig(
            hidden_dim=64,
            use_hpm=True,
            hpm_use_compositional_bank=True,
            hpm_use_procedural_bank=True,
        )
        model = RLAN(config=config)
        
        # Initial buffer size
        initial_size = len(model.hpm_procedural_buffer)
        
        # Simulate task completion (note: z_context, z_task, task_id)
        z_context = torch.randn(64)
        z_task = torch.randn(64)
        model.hpm_on_task_complete(
            z_context=z_context,
            z_task=z_task,
            task_id='test_task_001'
        )
        
        # Buffer should grow
        assert len(model.hpm_procedural_buffer) == initial_size + 1
    
    def test_hpm_buffer_save_load(self, tmp_path):
        """Test dynamic buffer save and load."""
        config = RLANConfig(
            hidden_dim=64,
            use_hpm=True,
            hpm_use_compositional_bank=True,
            hpm_use_procedural_bank=True,
        )
        model = RLAN(config=config)
        
        # Add to buffer
        for i in range(5):
            model.hpm_on_task_complete(
                z_context=torch.randn(64),
                z_task=torch.randn(64),
                task_id=f'task_{i}'
            )
        
        # Save
        save_dir = tmp_path / 'hpm_buffers'
        model.hpm_save_buffers(str(save_dir))
        
        # Clear buffer
        model.hpm_procedural_buffer.clear()
        assert len(model.hpm_procedural_buffer) == 0
        
        # Load
        model.hpm_load_buffers(str(save_dir))
        assert len(model.hpm_procedural_buffer) == 5


class TestHPMMemoryEfficiency:
    """Tests for HPM memory efficiency."""
    
    def test_no_memory_leak_forward(self):
        """Test no memory leak during forward passes."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        device = torch.device('cuda')
        config = RLANConfig(
            hidden_dim=64,
            num_solver_steps=2,
            use_hpm=True,
            hpm_use_compositional_bank=True,
            hpm_use_pattern_bank=True,
        )
        model = RLAN(config=config).to(device)
        model.eval()
        
        # Clear cache
        torch.cuda.empty_cache()
        gc.collect()
        
        initial_memory = torch.cuda.memory_allocated(device)
        
        # Many forward passes
        for _ in range(100):
            x = torch.randint(0, 10, (4, 16, 16), device=device)
            with torch.no_grad():
                _ = model(x)
        
        torch.cuda.empty_cache()
        gc.collect()
        
        final_memory = torch.cuda.memory_allocated(device)
        
        # Memory should not grow significantly
        memory_growth = final_memory - initial_memory
        assert memory_growth < 1e7  # Less than 10MB growth
    
    def test_no_memory_leak_training(self):
        """Test no memory leak during training."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        device = torch.device('cuda')
        config = RLANConfig(
            hidden_dim=64,
            num_solver_steps=2,
            use_hpm=True,
            hpm_use_compositional_bank=True,
            hpm_use_pattern_bank=True,
        )
        model = RLAN(config=config).to(device)
        model.train()
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        # Clear cache
        torch.cuda.empty_cache()
        gc.collect()
        
        initial_memory = torch.cuda.memory_allocated(device)
        
        # Many training steps
        for _ in range(20):
            model.hpm_on_epoch_start()
            
            x = torch.randint(0, 10, (4, 16, 16), device=device)
            logits = model(x)
            
            loss = logits.sum() + model.hpm_get_load_balance_loss()
            loss.backward()
            model.hpm_on_backward()
            
            optimizer.step()
            optimizer.zero_grad()
        
        torch.cuda.empty_cache()
        gc.collect()
        
        final_memory = torch.cuda.memory_allocated(device)
        
        # Memory should not grow significantly
        memory_growth = final_memory - initial_memory
        assert memory_growth < 5e7  # Less than 50MB growth


class TestHPMYAMLConfig:
    """Test HPM configuration from YAML files."""
    
    def test_load_yaml_config(self):
        """Test loading HPM config from YAML."""
        yaml_content = """
model:
  hidden_dim: 128
  patch_size: 16
  use_hpm: true
  hpm_top_k: 3
  hpm_balance_weight: 0.02
  hpm_primitives_per_bank: 32
  hpm_levels_per_bank: 3
  hpm_use_cross_attention: true
  hpm_memory_size: 5000
  hpm_retrieval_k: 10
  hpm_use_compositional_bank: true
  hpm_use_pattern_bank: true
  hpm_use_relational_bank: true
  hpm_use_concept_bank: false
  hpm_use_procedural_bank: true
  hpm_use_instance_bank: false
"""
        import io
        config_data = yaml.safe_load(io.StringIO(yaml_content))
        model_config = config_data['model']
        
        hpm_config = HPMConfig.from_dict(model_config)
        
        assert hpm_config.d_model == 128
        assert hpm_config.top_k == 3
        assert hpm_config.balance_loss_weight == 0.02
        assert hpm_config.primitives_per_bank == 32
        assert hpm_config.n_levels_per_bank == 3  # Fixed attribute name
        assert hpm_config.use_cross_attention == True
        assert hpm_config.use_procedural_bank == True
        assert hpm_config.use_instance_bank == False
    
    def test_hpm_disabled_by_default(self):
        """Test HPM is disabled by default."""
        # Create RLANConfig with defaults - use_hpm should default to False
        config = RLANConfig(hidden_dim=64)
        assert config.use_hpm == False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
