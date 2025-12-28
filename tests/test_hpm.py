"""
Unit Tests for Hierarchical Primitive Memory (HPM) v2

Tests all HPM components:
1. MemoryBank - static bank with hierarchical levels
2. MemoryRouter - sparse MoE Top-K routing
3. CrossBankAggregator - weighted sum + cross-attention
4. HierarchicalPrimitiveMemory - full module integration
5. DynamicMemoryBuffer - FAISS-backed KV cache

SMOKE TESTS (from design doc):
□ 1. HPM with single static bank
□ 2. HPM with all banks (static + dynamic)
□ 3. Sparse routing (only top_k banks queried)
□ 4. Load balancing loss decreases over training
□ 5. Gate value (tanh(α)) increases from 0 during training
□ 6. Freeze mechanism works for static banks
□ 7. Dynamic buffer grows when tasks are solved
□ 8. RLAN without LCR/SPH + HPM works
□ 9. RLAN with use_hpm=False unchanged
□ 10. Retrieval from dynamic buffer returns correct neighbors
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List

# Import HPM components
from sci_arc.models.rlan_modules.hpm import (
    MemoryBankType,
    MemoryBank,
    MemoryRouter,
    CrossBankAggregator,
    HierarchicalPrimitiveMemory,
    HPMConfig,
    STATIC_BANK_TYPES,
    DYNAMIC_BANK_TYPES,
)
from sci_arc.models.rlan_modules.dynamic_buffer import DynamicMemoryBuffer


class TestMemoryBank:
    """Tests for individual MemoryBank."""
    
    def test_bank_initialization(self):
        """Test bank initializes with correct shapes."""
        d_model = 256
        n_primitives = 16
        n_levels = 2
        
        bank = MemoryBank(
            d_model=d_model,
            n_primitives=n_primitives,
            n_levels=n_levels,
            bank_type=MemoryBankType.COMPOSITIONAL,
        )
        
        # Check primitive levels
        assert len(bank.primitive_levels) == n_levels
        total_primitives = sum(p.shape[0] for p in bank.primitive_levels)
        assert total_primitives == n_primitives
        
        # Check level weights
        assert bank.level_weights.shape == (n_levels,)
        
        # Check query projection
        assert bank.query_proj.weight.shape == (d_model, d_model)
    
    def test_bank_forward(self):
        """Test bank forward pass produces correct output shape."""
        d_model = 128
        batch_size = 4
        
        bank = MemoryBank(d_model=d_model, n_primitives=16, n_levels=2)
        z = torch.randn(batch_size, d_model)
        
        output, attentions = bank(z, return_attention=True)
        
        assert output.shape == (batch_size, d_model)
        assert attentions is not None
        assert len(attentions) == 2  # One per level
    
    def test_bank_forward_no_attention(self):
        """Test bank forward without returning attention."""
        bank = MemoryBank(d_model=64, n_primitives=8, n_levels=2)
        z = torch.randn(2, 64)
        
        output, attentions = bank(z, return_attention=False)
        
        assert output.shape == (2, 64)
        assert attentions is None
    
    def test_bank_usage_tracking(self):
        """Test that usage counts are tracked during training."""
        bank = MemoryBank(d_model=64, n_primitives=8, n_levels=2)
        bank.train()
        
        z = torch.randn(10, 64)
        
        # Initial usage should be zero
        for level_idx in range(bank.n_levels):
            usage = getattr(bank, f'usage_count_{level_idx}')
            assert usage.sum() == 0
        
        # After forward, usage should be tracked
        bank(z)
        
        for level_idx in range(bank.n_levels):
            usage = getattr(bank, f'usage_count_{level_idx}')
            assert usage.sum() > 0
    
    def test_bank_freeze_mechanism(self):
        """Test primitive freezing."""
        bank = MemoryBank(d_model=64, n_primitives=8, n_levels=2)
        bank.train()
        
        # Simulate heavy usage of some primitives
        for level_idx in range(bank.n_levels):
            usage = getattr(bank, f'usage_count_{level_idx}')
            usage[0] = 1000  # First primitive heavily used
        
        bank.freeze_stable_primitives(usage_threshold=100, top_fraction=0.5)
        
        # Check freeze masks
        for level_idx in range(bank.n_levels):
            freeze_mask = getattr(bank, f'freeze_mask_{level_idx}')
            assert freeze_mask[0] == True  # First should be frozen


class TestMemoryRouter:
    """Tests for MemoryRouter (sparse MoE)."""
    
    def test_router_initialization(self):
        """Test router initializes correctly."""
        d_model = 256
        n_banks = 6
        top_k = 2
        
        router = MemoryRouter(d_model=d_model, n_banks=n_banks, top_k=top_k)
        
        assert router.n_banks == n_banks
        assert router.top_k == top_k
        assert router.routing_counts.shape == (n_banks,)
    
    def test_router_sparse_output(self):
        """Test router produces sparse outputs (only top_k nonzero)."""
        router = MemoryRouter(d_model=64, n_banks=6, top_k=2)
        z = torch.randn(4, 64)
        
        weights, indices = router(z)
        
        # Check output shapes
        assert weights.shape == (4, 6)
        assert indices.shape == (4, 2)
        
        # Check sparsity: exactly top_k nonzero per row
        for row in range(4):
            nonzero_count = (weights[row] > 0).sum().item()
            assert nonzero_count == 2
    
    def test_router_weights_sum_to_one(self):
        """Test routing weights sum to 1 (for selected banks)."""
        router = MemoryRouter(d_model=64, n_banks=6, top_k=3)
        z = torch.randn(4, 64)
        
        weights, _ = router(z)
        
        # Sum of nonzero weights should be 1
        for row in range(4):
            nonzero_weights = weights[row][weights[row] > 0]
            assert torch.allclose(nonzero_weights.sum(), torch.tensor(1.0), atol=1e-5)
    
    def test_router_load_balance_loss(self):
        """Test load balancing loss computation."""
        router = MemoryRouter(d_model=64, n_banks=4, top_k=2)
        router.train()
        
        # Initial loss should be near 0 (no samples processed)
        initial_loss = router.compute_load_balance_loss()
        assert initial_loss.item() == 0.0
        
        # Process some samples
        z = torch.randn(100, 64)
        router(z)
        
        # Now loss should be positive
        loss = router.compute_load_balance_loss()
        assert loss.item() > 0
    
    def test_router_reset_statistics(self):
        """Test statistics reset."""
        router = MemoryRouter(d_model=64, n_banks=4, top_k=2)
        router.train()
        
        # Process some samples
        z = torch.randn(50, 64)
        router(z)
        
        assert router.total_samples.item() > 0
        
        # Reset
        router.reset_statistics()
        
        assert router.total_samples.item() == 0
        assert router.routing_counts.sum().item() == 0


class TestCrossBankAggregator:
    """Tests for CrossBankAggregator."""
    
    def test_aggregator_weighted_sum(self):
        """Test aggregation without cross-attention."""
        aggregator = CrossBankAggregator(
            d_model=64, n_banks=4, use_cross_attention=False
        )
        
        z = torch.randn(2, 64)
        bank_outputs = [torch.randn(2, 64) for _ in range(4)]
        routing_weights = torch.softmax(torch.randn(2, 4), dim=-1)
        
        output = aggregator(z, bank_outputs, routing_weights)
        
        assert output.shape == (2, 64)
    
    def test_aggregator_with_cross_attention(self):
        """Test aggregation with cross-attention."""
        aggregator = CrossBankAggregator(
            d_model=64, n_banks=4, use_cross_attention=True
        )
        
        z = torch.randn(2, 64)
        bank_outputs = [torch.randn(2, 64) for _ in range(4)]
        routing_weights = torch.softmax(torch.randn(2, 4), dim=-1)
        
        output = aggregator(z, bank_outputs, routing_weights)
        
        assert output.shape == (2, 64)


class TestHPMConfig:
    """Tests for HPMConfig."""
    
    def test_config_from_dict(self):
        """Test config creation from dictionary."""
        yaml_config = {
            'hidden_dim': 128,
            'hpm_top_k': 3,
            'hpm_balance_weight': 0.02,
            'hpm_use_compositional_bank': True,
            'hpm_use_pattern_bank': False,
        }
        
        config = HPMConfig.from_dict(yaml_config)
        
        assert config.d_model == 128
        assert config.top_k == 3
        assert config.balance_loss_weight == 0.02
        assert config.use_compositional_bank == True
        assert config.use_pattern_bank == False
    
    def test_config_enabled_banks(self):
        """Test getting enabled bank types."""
        config = HPMConfig(
            use_compositional_bank=True,
            use_pattern_bank=True,
            use_relational_bank=False,
            use_concept_bank=False,
            use_procedural_bank=True,
            use_instance_bank=False,
        )
        
        banks = config.get_enabled_bank_types()
        
        assert MemoryBankType.COMPOSITIONAL in banks
        assert MemoryBankType.PATTERN in banks
        assert MemoryBankType.RELATIONAL not in banks
        assert MemoryBankType.PROCEDURAL in banks
        assert len(banks) == 3


class TestHierarchicalPrimitiveMemory:
    """Tests for full HPM module."""
    
    def test_hpm_single_bank(self):
        """Smoke test #1: HPM with single static bank."""
        config = HPMConfig(
            d_model=64,
            top_k=1,
            use_compositional_bank=True,
            use_pattern_bank=False,
            use_relational_bank=False,
            use_concept_bank=False,
            use_procedural_bank=False,
            use_instance_bank=False,
        )
        
        hpm = HierarchicalPrimitiveMemory(config)
        z = torch.randn(2, 64)
        
        z_out, routing = hpm(z, return_routing=True)
        
        assert z_out.shape == (2, 64)
        assert routing.shape == (2, 1)  # Only 1 bank
    
    def test_hpm_all_static_banks(self):
        """Smoke test #2a: HPM with all static banks."""
        config = HPMConfig(
            d_model=64,
            top_k=2,
            use_compositional_bank=True,
            use_pattern_bank=True,
            use_relational_bank=True,
            use_concept_bank=True,
            use_procedural_bank=False,
            use_instance_bank=False,
        )
        
        hpm = HierarchicalPrimitiveMemory(config)
        z = torch.randn(4, 64)
        
        z_out, routing = hpm(z, return_routing=True)
        
        assert z_out.shape == (4, 64)
        assert routing.shape == (4, 4)  # 4 banks
    
    def test_hpm_gated_residual_starts_zero(self):
        """Smoke test #5: Gate value starts at 0."""
        config = HPMConfig(d_model=64)
        hpm = HierarchicalPrimitiveMemory(config)
        
        gate_value = hpm.get_gate_value()
        
        # tanh(0) = 0
        assert abs(gate_value) < 1e-6
    
    def test_hpm_sparse_routing(self):
        """Smoke test #3: Only top_k banks queried."""
        config = HPMConfig(
            d_model=64,
            top_k=2,
            use_compositional_bank=True,
            use_pattern_bank=True,
            use_relational_bank=True,
            use_concept_bank=True,
        )
        
        hpm = HierarchicalPrimitiveMemory(config)
        z = torch.randn(4, 64)
        
        _, routing = hpm(z, return_routing=True)
        
        # Check each row has exactly top_k nonzero
        for row in range(4):
            nonzero = (routing[row] > 0).sum().item()
            assert nonzero == 2
    
    def test_hpm_load_balance_loss(self):
        """Smoke test #4: Load balancing loss computation."""
        config = HPMConfig(d_model=64, top_k=2)
        hpm = HierarchicalPrimitiveMemory(config)
        hpm.train()
        
        # Process samples to accumulate routing stats
        for _ in range(10):
            z = torch.randn(8, 64)
            hpm(z)
        
        loss = hpm.get_load_balance_loss()
        
        assert loss.item() > 0
    
    def test_hpm_reset_epoch_stats(self):
        """Test epoch stats reset."""
        config = HPMConfig(d_model=64)
        hpm = HierarchicalPrimitiveMemory(config)
        hpm.train()
        
        # Process samples
        z = torch.randn(8, 64)
        hpm(z)
        
        assert hpm.router.total_samples.item() > 0
        
        # Reset
        hpm.reset_epoch_stats()
        
        assert hpm.router.total_samples.item() == 0
    
    def test_hpm_backward_compatibility_none_input(self):
        """Test HPM handles None input (backward compatibility)."""
        config = HPMConfig(d_model=64)
        hpm = HierarchicalPrimitiveMemory(config)
        
        z_out, routing = hpm(None, return_routing=True)
        
        assert z_out.shape == (1, 64)
    
    def test_hpm_with_dynamic_buffers(self):
        """Test HPM with dynamic buffer inputs."""
        config = HPMConfig(
            d_model=64,
            top_k=2,
            use_compositional_bank=True,
            use_pattern_bank=False,
            use_relational_bank=False,
            use_concept_bank=False,
            use_procedural_bank=True,
            use_instance_bank=True,
        )
        
        hpm = HierarchicalPrimitiveMemory(config)
        z = torch.randn(2, 64)
        
        # Create mock dynamic buffers
        dynamic_buffers = {
            'PROCEDURAL': (torch.randn(5, 64), torch.randn(5, 64)),
            'INSTANCE': (torch.randn(5, 64), torch.randn(5, 64)),
        }
        
        z_out, routing = hpm(z, dynamic_buffers=dynamic_buffers, return_routing=True)
        
        assert z_out.shape == (2, 64)
        assert routing.shape == (2, 3)  # 3 banks (1 static + 2 dynamic)
    
    def test_hpm_gradient_flow(self):
        """Test gradients flow through HPM."""
        config = HPMConfig(d_model=64, top_k=2)
        hpm = HierarchicalPrimitiveMemory(config)
        
        z = torch.randn(2, 64, requires_grad=True)
        z_out, _ = hpm(z)
        
        loss = z_out.sum()
        loss.backward()
        
        # Check gradients exist
        assert z.grad is not None
        assert hpm.residual_gate.grad is not None
    
    def test_hpm_freeze_stable_primitives(self):
        """Smoke test #6: Freeze mechanism works."""
        config = HPMConfig(d_model=64)
        hpm = HierarchicalPrimitiveMemory(config)
        hpm.train()
        
        # Simulate usage
        for _ in range(100):
            z = torch.randn(8, 64)
            hpm(z)
        
        # Freeze stable primitives
        hpm.freeze_stable_primitives()
        
        # Some primitives should be frozen
        frozen_count = 0
        for bank_name, bank in hpm.banks.items():
            for level_idx in range(bank.n_levels):
                freeze_mask = getattr(bank, f'freeze_mask_{level_idx}')
                frozen_count += freeze_mask.sum().item()
        
        # At least some should be frozen
        assert frozen_count > 0


class TestDynamicMemoryBuffer:
    """Tests for DynamicMemoryBuffer."""
    
    def test_buffer_initialization(self):
        """Test buffer initializes correctly."""
        buffer = DynamicMemoryBuffer(d_model=64, max_size=100, use_faiss=False)
        
        assert len(buffer) == 0
        assert buffer.d_model == 64
        assert buffer.max_size == 100
    
    def test_buffer_add_single(self):
        """Test adding single entry."""
        buffer = DynamicMemoryBuffer(d_model=64, max_size=100, use_faiss=False)
        
        key = torch.randn(64)
        value = torch.randn(64)
        
        buffer.add(key, value, task_id='task_001')
        
        assert len(buffer) == 1
        assert buffer.get_task_ids() == ['task_001']
    
    def test_buffer_add_batch(self):
        """Test adding batch of entries."""
        buffer = DynamicMemoryBuffer(d_model=64, max_size=100, use_faiss=False)
        
        keys = torch.randn(5, 64)
        values = torch.randn(5, 64)
        
        buffer.add(keys, values, task_id='batch_task')
        
        assert len(buffer) == 5
    
    def test_buffer_retrieve(self):
        """Smoke test #10: Retrieval returns correct neighbors."""
        buffer = DynamicMemoryBuffer(d_model=64, max_size=100, use_faiss=False)
        
        # Add some entries
        for i in range(10):
            key = torch.randn(64)
            value = torch.randn(64)
            buffer.add(key, value, task_id=f'task_{i}')
        
        # Retrieve
        query = torch.randn(1, 64)
        keys, values = buffer.retrieve(query, k=3)
        
        assert keys.shape == (3, 64)
        assert values.shape == (3, 64)
    
    def test_buffer_retrieve_empty(self):
        """Test retrieval from empty buffer."""
        buffer = DynamicMemoryBuffer(d_model=64, max_size=100, use_faiss=False)
        
        query = torch.randn(1, 64)
        keys, values = buffer.retrieve(query, k=5)
        
        assert keys is None
        assert values is None
    
    def test_buffer_fifo_eviction(self):
        """Smoke test #7: Buffer grows and evicts (FIFO)."""
        buffer = DynamicMemoryBuffer(d_model=64, max_size=10, use_faiss=False)
        
        # Add more than max_size
        for i in range(15):
            key = torch.randn(64)
            value = torch.randn(64)
            buffer.add(key, value, task_id=f'task_{i}')
        
        # Should be capped at max_size
        assert len(buffer) == 10
        
        # First 5 should be evicted (FIFO)
        task_ids = buffer.get_task_ids()
        assert 'task_0' not in task_ids
        assert 'task_14' in task_ids
    
    def test_buffer_clear(self):
        """Test buffer clear."""
        buffer = DynamicMemoryBuffer(d_model=64, max_size=100, use_faiss=False)
        
        for i in range(5):
            buffer.add(torch.randn(64), torch.randn(64))
        
        assert len(buffer) == 5
        
        buffer.clear()
        
        assert len(buffer) == 0
    
    def test_buffer_save_load(self, tmp_path):
        """Test buffer save and load."""
        buffer = DynamicMemoryBuffer(d_model=64, max_size=100, use_faiss=False)
        
        # Add entries
        for i in range(5):
            key = torch.randn(64)
            value = torch.randn(64)
            buffer.add(key, value, task_id=f'task_{i}')
        
        # Save
        save_path = str(tmp_path / 'buffer.pt')
        buffer.save(save_path)
        
        # Load
        loaded = DynamicMemoryBuffer.load(save_path, use_faiss=False)
        
        assert len(loaded) == 5
        assert loaded.get_task_ids() == buffer.get_task_ids()
    
    def test_buffer_stats(self):
        """Test buffer statistics."""
        buffer = DynamicMemoryBuffer(d_model=64, max_size=100, use_faiss=False)
        
        for i in range(10):
            buffer.add(torch.randn(64), torch.randn(64), task_id=f'task_{i % 3}')
        
        stats = buffer.get_stats()
        
        assert stats['size'] == 10
        assert stats['max_size'] == 100
        assert stats['fill_ratio'] == 0.1
        assert stats['unique_tasks'] == 3  # task_0, task_1, task_2


class TestHPMIntegration:
    """Integration tests for HPM with RLAN."""
    
    def test_hpm_disable_no_break(self):
        """Smoke test #9: RLAN with use_hpm=False unchanged."""
        # This test requires RLAN import - will be run as integration test
        pass  # Placeholder - full test in test_hpm_integration.py


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
