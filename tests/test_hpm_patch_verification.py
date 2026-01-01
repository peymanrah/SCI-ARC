"""
Verification tests for HPM patches:
1. Eviction tracking (FAISS rebuild fix)
2. state_dict/load_state_dict canonical API
3. Backward compatibility with old format
4. retrieve_batch for per-sample retrieval
5. RLAN model HPM state management
"""
import pytest
import torch
import tempfile
import os

from sci_arc.models.rlan_modules.dynamic_buffer import DynamicMemoryBuffer


class TestDynamicMemoryBufferPatches:
    """Tests for DynamicMemoryBuffer patches."""
    
    def test_eviction_tracking(self):
        """P0.1: Test eviction tracking for FAISS rebuild logic."""
        # Use correct API: d_model instead of key_dim/value_dim
        buf = DynamicMemoryBuffer(d_model=64, max_size=5, use_faiss=False)
        
        # Add 5 items (fill buffer)
        for i in range(5):
            buf.add(torch.randn(64), torch.randn(64), f'task_{i}')
        
        assert len(buf) == 5, f'Expected 5, got {len(buf)}'
        assert buf._last_eviction_count == 0, 'Should have 0 evictions before overflow'
        
        # Add one more to trigger eviction
        buf.add(torch.randn(64), torch.randn(64), 'task_5')
        assert len(buf) == 5, f'Expected 5 after eviction, got {len(buf)}'
        assert buf._last_eviction_count == 1, f'Should have 1 eviction, got {buf._last_eviction_count}'
        
        # Stats should include eviction count
        stats = buf.get_stats()
        assert 'eviction_count' in stats, 'get_stats should include eviction_count'
        assert stats['eviction_count'] == 1, f'Stats eviction_count should be 1'
        
    def test_state_dict_canonical_format(self):
        """P1.3: Test canonical state_dict format with version field."""
        buf = DynamicMemoryBuffer(d_model=64, max_size=10, use_faiss=False)
        
        # Add some items
        for i in range(5):
            buf.add(torch.randn(64), torch.randn(64), f'task_{i}')
        
        state = buf.state_dict()
        
        # Check required fields
        assert 'version' in state, 'state_dict should have version field'
        assert state['version'] == '2.0', 'Version should be 2.0'
        assert 'keys' in state, 'state_dict should have keys'
        assert 'values' in state, 'state_dict should have values'
        assert 'task_ids' in state, 'state_dict should have task_ids'
        
        # Check data integrity
        assert len(state['keys']) == 5, f'Keys count mismatch: {len(state["keys"])}'
        assert len(state['values']) == 5, f'Values count mismatch'
        assert len(state['task_ids']) == 5, f'Task IDs count mismatch'
        
    def test_load_state_dict_new_format(self):
        """P1.3: Test loading from new canonical format."""
        buf = DynamicMemoryBuffer(d_model=64, max_size=10, use_faiss=False)
        for i in range(5):
            buf.add(torch.randn(64), torch.randn(64), f'task_{i}')
        
        state = buf.state_dict()
        
        # Load into new buffer
        buf2 = DynamicMemoryBuffer(d_model=64, max_size=10, use_faiss=False)
        buf2.load_state_dict(state)
        
        assert len(buf2) == len(buf), f'Loaded buffer size mismatch'
        
    def test_load_state_dict_backward_compatible(self):
        """P1.3: Test backward compatibility with old format (no version field)."""
        buf = DynamicMemoryBuffer(d_model=64, max_size=10, use_faiss=False)
        for i in range(5):
            buf.add(torch.randn(64), torch.randn(64), f'task_{i}')
        
        # Simulate old format (no version field) - get current keys/values
        state = buf.state_dict()
        old_format = {
            'keys': state['keys'],
            'values': state['values'],
            'task_ids': state['task_ids'],
        }
        
        # Load from old format
        buf2 = DynamicMemoryBuffer(d_model=64, max_size=10, use_faiss=False)
        buf2.load_state_dict(old_format)
        
        assert len(buf2) == 5, f'Loaded buffer should have 5 items from old format'
        
    def test_retrieve_batch_basic(self):
        """P0.2: Test retrieve_batch for per-sample retrieval."""
        buf = DynamicMemoryBuffer(d_model=64, max_size=100, use_faiss=False)
        
        # Add items with different task IDs
        for i in range(20):
            buf.add(torch.randn(64), torch.randn(64), f'task_{i % 5}')
        
        # Query with batch of samples
        queries = torch.randn(3, 64)
        
        keys, values, stats = buf.retrieve_batch(queries, k=2)
        
        assert keys is not None, 'retrieve_batch should return keys'
        assert values is not None, 'retrieve_batch should return values'
        assert stats is not None, 'retrieve_batch should return stats'
        assert 'avg_similarity' in stats, 'Stats should have avg_similarity'
        assert 'retrieved' in stats, 'Stats should have retrieved count'
        
        # Check shapes
        assert keys.shape[0] == 3, 'Should have 3 samples'
        assert keys.shape[1] == 2, 'Should have k=2 per sample'
        assert values.shape[2] == 64, 'Value dim should be 64 (d_model)'
        
    def test_retrieve_batch_empty_buffer(self):
        """Test retrieve_batch on empty buffer."""
        buf = DynamicMemoryBuffer(d_model=64, max_size=100, use_faiss=False)
        
        queries = torch.randn(3, 64)
        
        keys, values, stats = buf.retrieve_batch(queries, k=2)
        
        assert keys is None, 'retrieve_batch on empty buffer should return None keys'
        assert values is None, 'retrieve_batch on empty buffer should return None values'
        
    def test_retrieve_batch_similarity_stats(self):
        """Test that retrieve_batch properly computes similarity statistics."""
        buf = DynamicMemoryBuffer(d_model=64, max_size=100, use_faiss=False)
        
        # Add known items
        key = torch.randn(64)
        key = key / key.norm()  # Normalize
        buf.add(key.clone(), torch.randn(64), 'task_0')
        
        # Query with exact match
        queries = key.unsqueeze(0)  # Shape: [1, 64]
        
        keys, values, stats = buf.retrieve_batch(queries, k=1)
        
        assert keys is not None
        # Exact match should have very high similarity
        assert stats['avg_similarity'] > 0.99, f'Exact match should have ~1.0 similarity, got {stats["avg_similarity"]}'


class TestRLANHPMState:
    """Tests for RLAN model HPM state management."""
    
    def test_get_hpm_state_structure(self):
        """P1.3/P1.4: Test get_hpm_state returns proper structure."""
        from sci_arc.models.rlan import RLAN, RLANConfig
        
        # Create minimal RLAN model with HPM and dynamic banks enabled
        config = RLANConfig(
            hidden_dim=64,
            use_hpm=True,
            hpm_use_instance_bank=True,
            hpm_use_procedural_bank=True,
        )
        model = RLAN(config=config)
        
        # Ensure HPM is enabled
        assert model.use_hpm, "HPM should be enabled"
        
        state = model.get_hpm_state()
        
        assert 'version' in state, 'Should have version'
        assert 'use_hpm' in state, 'Should have use_hpm flag'
        assert 'instance' in state, 'Should have instance key'
        assert 'procedural' in state, 'Should have procedural key'
        
    def test_load_hpm_state_roundtrip(self):
        """P1.3: Test save/load roundtrip for HPM state."""
        from sci_arc.models.rlan import RLAN, RLANConfig
        
        config = RLANConfig(
            hidden_dim=64,
            use_hpm=True,
            hpm_use_instance_bank=True,
            hpm_use_procedural_bank=True,
        )
        model = RLAN(config=config)
        
        # Ensure HPM buffers exist
        assert model.hpm_instance_buffer is not None, "Instance buffer should exist"
        assert model.hpm_procedural_buffer is not None, "Procedural buffer should exist"
        
        # Populate buffers
        for i in range(5):
            model.hpm_instance_buffer.add(torch.randn(64), torch.randn(64), f'task_{i}')
            model.hpm_procedural_buffer.add(torch.randn(64), torch.randn(64), f'task_{i}')
        
        # Get state
        state = model.get_hpm_state()
        
        # Create new model and load
        model2 = RLAN(config=config)
        model2.load_hpm_state(state)
        
        assert len(model2.hpm_instance_buffer) == 5, 'Instance buffer should have 5 items'
        assert len(model2.hpm_procedural_buffer) == 5, 'Procedural buffer should have 5 items'
        
    def test_export_import_hpm_memory(self):
        """P1.4: Test standalone HPM memory export/import."""
        from sci_arc.models.rlan import RLAN, RLANConfig
        
        config = RLANConfig(
            hidden_dim=64,
            use_hpm=True,
            hpm_use_instance_bank=True,
            hpm_use_procedural_bank=True,
        )
        model = RLAN(config=config)
        
        # Populate buffers
        for i in range(3):
            model.hpm_instance_buffer.add(torch.randn(64), torch.randn(64), f'task_{i}')
            model.hpm_procedural_buffer.add(torch.randn(64), torch.randn(64), f'task_{i}')
        
        # Export to temp file
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'hpm_memory.pt')
            model.export_hpm_memory(path, split_tag='test_split')
            
            assert os.path.exists(path), 'Export file should exist'
            
            # Import into new model
            model2 = RLAN(config=config)
            metadata = model2.import_hpm_memory(path)
            
            assert len(model2.hpm_instance_buffer) == 3
            assert len(model2.hpm_procedural_buffer) == 3
            assert metadata['split_tag'] == 'test_split'


class TestHPMBatchRetrieval:
    """TODO 7: Regression tests for per-sample HPM batch retrieval (no batch mixing)."""
    
    def test_retrieve_batch_no_cross_sample_mixing(self):
        """Critical: Verify each sample gets its own neighbors, not shared across batch.
        
        This test creates two distinct clusters in the buffer, then queries with
        B=2 where each query is closer to a different cluster. Each sample should
        retrieve from its own cluster only.
        """
        buf = DynamicMemoryBuffer(d_model=64, max_size=100, use_faiss=False)
        
        # Create two distinct clusters by using orthogonal base vectors
        cluster_a_base = torch.randn(64)
        cluster_a_base = cluster_a_base / cluster_a_base.norm()
        
        cluster_b_base = torch.randn(64)
        # Make B orthogonal to A (Gram-Schmidt)
        cluster_b_base = cluster_b_base - (cluster_b_base @ cluster_a_base) * cluster_a_base
        cluster_b_base = cluster_b_base / cluster_b_base.norm()
        
        # Add 10 items to each cluster with small perturbations
        for i in range(10):
            noise = torch.randn(64) * 0.1
            key_a = cluster_a_base + noise
            buf.add(key_a, torch.ones(64) * 1.0, f'cluster_a_{i}')  # Value=1 for cluster A
            
            noise = torch.randn(64) * 0.1
            key_b = cluster_b_base + noise
            buf.add(key_b, torch.ones(64) * 2.0, f'cluster_b_{i}')  # Value=2 for cluster B
        
        # Query: sample 0 is close to cluster A, sample 1 is close to cluster B
        queries = torch.stack([
            cluster_a_base + torch.randn(64) * 0.05,
            cluster_b_base + torch.randn(64) * 0.05,
        ], dim=0)  # [B=2, D=64]
        
        keys, values, stats = buf.retrieve_batch(queries, k=5)
        
        # Check shapes
        assert keys.shape == (2, 5, 64), f'Keys shape: {keys.shape}'
        assert values.shape == (2, 5, 64), f'Values shape: {values.shape}'
        
        # Check that sample 0 retrieved cluster A (values ~= 1.0)
        sample_0_mean_value = values[0].mean().item()
        assert abs(sample_0_mean_value - 1.0) < 0.5, f'Sample 0 should retrieve cluster A (value ~1.0), got {sample_0_mean_value}'
        
        # Check that sample 1 retrieved cluster B (values ~= 2.0)
        sample_1_mean_value = values[1].mean().item()
        assert abs(sample_1_mean_value - 2.0) < 0.5, f'Sample 1 should retrieve cluster B (value ~2.0), got {sample_1_mean_value}'
        
    def test_retrieve_batch_different_neighbors_per_sample(self):
        """Verify that different queries can retrieve different neighbors."""
        buf = DynamicMemoryBuffer(d_model=32, max_size=100, use_faiss=False)
        
        # Add indexed entries so we can verify which were retrieved
        for i in range(20):
            key = torch.zeros(32)
            key[i % 32] = 1.0  # One-hot-like keys
            value = torch.zeros(32)
            value[0] = float(i)  # Store index in value[0]
            buf.add(key, value, f'task_{i}')
        
        # Create queries that should match different indices
        q0 = torch.zeros(32)
        q0[0] = 1.0  # Should match index 0
        
        q1 = torch.zeros(32)
        q1[5] = 1.0  # Should match index 5
        
        queries = torch.stack([q0, q1], dim=0)
        
        keys, values, stats = buf.retrieve_batch(queries, k=3)
        
        # Check that each sample retrieved different entries
        sample_0_indices = values[0, :, 0].tolist()
        sample_1_indices = values[1, :, 0].tolist()
        
        # Sample 0 should have retrieved entries 0, 0 (it matches index 0 pattern)
        # Sample 1 should have retrieved entries 5, 5 (it matches index 5 pattern)
        assert 0.0 in sample_0_indices, f'Sample 0 should have retrieved index 0, got {sample_0_indices}'
        assert 5.0 in sample_1_indices, f'Sample 1 should have retrieved index 5, got {sample_1_indices}'
        
    def test_hpm_dynamic_bank_handles_batch_keys(self):
        """Test that HPM's dynamic bank attention correctly handles [B, k, D] keys."""
        from sci_arc.models.rlan_modules.hpm import HierarchicalPrimitiveMemory, HPMConfig
        
        config = HPMConfig(
            d_model=64,
            top_k=2,
            use_instance_bank=True,
            use_procedural_bank=False,
            use_compositional_bank=True,
            use_pattern_bank=True,
            use_relational_bank=False,
            use_concept_bank=False,
        )
        
        hpm = HierarchicalPrimitiveMemory(config)
        
        B = 4
        k = 5
        D = 64
        
        # Create batch input
        z = torch.randn(B, D)
        
        # Create per-sample dynamic buffers [B, k, D]
        dynamic_buffers = {
            'INSTANCE': (torch.randn(B, k, D), torch.randn(B, k, D))
        }
        
        # Forward pass should work without errors
        z_enhanced, routing = hpm(z, dynamic_buffers=dynamic_buffers, return_routing=True)
        
        assert z_enhanced.shape == (B, D), f'Output shape mismatch: {z_enhanced.shape}'
        assert routing.shape[0] == B, f'Routing batch dimension mismatch'
        
    def test_global_dedup_contains_task(self):
        """TODO 3: Test global deduplication via contains_task."""
        buf = DynamicMemoryBuffer(d_model=64, max_size=100, use_faiss=False)
        
        # Add some tasks
        buf.add(torch.randn(64), torch.randn(64), 'task_unique_001')
        buf.add(torch.randn(64), torch.randn(64), 'task_unique_002')
        buf.add(torch.randn(64), torch.randn(64), 'task_unique_003')
        
        # Check contains_task
        assert buf.contains_task('task_unique_001'), 'Should contain task_unique_001'
        assert buf.contains_task('task_unique_002'), 'Should contain task_unique_002'
        assert not buf.contains_task('task_nonexistent'), 'Should not contain task_nonexistent'
        
        # Check unique task IDs
        unique_ids = buf.get_unique_task_ids()
        assert len(unique_ids) == 3, f'Should have 3 unique tasks, got {len(unique_ids)}'
        
    def test_load_hpm_state_with_force_load(self):
        """TODO 5: Test loading HPM state even when use_hpm is False."""
        from sci_arc.models.rlan import RLAN, RLANConfig
        
        # Create model with HPM enabled and populate buffers
        config = RLANConfig(
            hidden_dim=64,
            use_hpm=True,
            hpm_use_instance_bank=True,
            hpm_use_procedural_bank=True,
        )
        model1 = RLAN(config=config)
        
        for i in range(5):
            model1.hpm_instance_buffer.add(torch.randn(64), torch.randn(64), f'task_{i}')
        
        state = model1.get_hpm_state()
        
        # Create model with HPM disabled but buffers still created
        # (simulating inference-time loading where use_hpm=False during staging)
        model2 = RLAN(config=config)
        model2.use_hpm = False  # Simulate staged-off state
        
        # Without force_load, should skip loading (old behavior)
        model2.hpm_instance_buffer.clear()
        model2.load_hpm_state(state, force_load=False)
        # Should still load because buffers exist (relaxed gating)
        assert len(model2.hpm_instance_buffer) == 5, 'With relaxed gating, should load even if use_hpm=False'
        
        # With force_load=True, should definitely load
        model2.hpm_instance_buffer.clear()
        model2.load_hpm_state(state, force_load=True)
        assert len(model2.hpm_instance_buffer) == 5, 'With force_load=True, should load'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
