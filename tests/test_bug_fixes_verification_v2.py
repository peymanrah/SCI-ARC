"""
Smoke tests to verify bug fixes from the consolidated bug report.

These tests verify that:
1. BucketedBatchSampler uses metadata-only sizing (no __getitem__ calls)
2. MemoryManager is imported and used in train_rlan.py
3. aug_info includes both translation key formats for backward compatibility
4. TRMStyleEvaluator has translation helpers and proper RLAN call signature
5. All fixes don't break existing functionality

Run with: pytest tests/test_bug_fixes_verification_v2.py -v
"""
import pytest
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.mark.cpu
class TestBucketedBatchSamplerMetadataOnly:
    """Test that BucketedBatchSampler uses metadata without calling __getitem__."""
    
    def test_sampler_uses_metadata_method(self):
        """Verify _get_max_grid_size_from_task_metadata method exists."""
        from sci_arc.data.dataset import BucketedBatchSampler
        
        assert hasattr(BucketedBatchSampler, '_get_max_grid_size_from_task_metadata'), \
            "BucketedBatchSampler should have _get_max_grid_size_from_task_metadata method"
    
    def test_sampler_with_tasks_metadata_avoids_getitem(self):
        """BucketedBatchSampler should use tasks metadata without __getitem__."""
        from sci_arc.data.dataset import BucketedBatchSampler
        
        class MetadataOnlyDataset:
            """Dataset with tasks metadata but exploding __getitem__."""
            
            def __init__(self, n: int = 20):
                self._n = n
                self._getitem_calls = 0
                # Provide tasks metadata for sizing
                self.tasks = [
                    {
                        "task_id": str(i),
                        "train": [
                            {"input": [[0] * (5 + i % 5)] * (5 + i % 5), 
                             "output": [[0] * (5 + i % 5)] * (5 + i % 5)},
                        ],
                        "test": [
                            {"input": [[0] * (5 + i % 5)] * (5 + i % 5), 
                             "output": [[0] * (5 + i % 5)] * (5 + i % 5)},
                        ],
                    }
                    for i in range(n)
                ]
            
            def __len__(self):
                return self._n
            
            def __getitem__(self, idx: int):
                self._getitem_calls += 1
                # Return minimal sample to not crash if called
                return {
                    'task_id': str(idx),
                    'original_max_size': 10,
                    'test_input': [[0]],
                }
        
        ds = MetadataOnlyDataset(n=20)
        
        # Build sampler - should NOT call __getitem__ if metadata works
        sampler = BucketedBatchSampler(
            dataset=ds, 
            batch_size=4, 
            bucket_boundaries=[10, 20],
            drop_last=False
        )
        
        # With proper metadata-based sizing, __getitem__ should NOT be called
        assert ds._getitem_calls == 0, \
            f"Expected 0 __getitem__ calls with metadata available, got {ds._getitem_calls}"
        
        # Verify sampler works
        assert len(sampler) > 0, "Sampler should have batches"


@pytest.mark.cpu
class TestMemoryManagerIntegration:
    """Test that MemoryManager is properly integrated into training script."""
    
    def test_memory_manager_imported_in_train_script(self):
        """train_rlan.py should import MemoryManager."""
        train_script = project_root / "scripts" / "train_rlan.py"
        text = train_script.read_text(encoding="utf-8")
        
        assert "from sci_arc.utils.memory_manager import" in text, \
            "train_rlan.py should import from memory_manager"
        assert "get_memory_manager" in text or "MemoryManager" in text, \
            "train_rlan.py should reference MemoryManager or get_memory_manager"
    
    def test_memory_manager_used_for_batch_validation(self):
        """train_rlan.py should use MemoryManager for batch size validation."""
        train_script = project_root / "scripts" / "train_rlan.py"
        text = train_script.read_text(encoding="utf-8")
        
        assert "get_safe_batch_size" in text, \
            "train_rlan.py should call get_safe_batch_size for memory-aware batch sizing"
    
    def test_memory_manager_factory_works(self):
        """get_memory_manager should create a MemoryManager instance."""
        from sci_arc.utils.memory_manager import get_memory_manager
        
        config = {'training': {'memory_safety_margin': 0.9}}
        mgr = get_memory_manager(config)
        
        assert mgr is not None
        assert hasattr(mgr, 'get_safe_batch_size')
        assert hasattr(mgr, 'estimate_forward_memory_mb')


@pytest.mark.cpu
class TestAugInfoTranslationKeyFormats:
    """Test that aug_info includes both translation key formats."""
    
    def test_dataset_includes_both_key_formats(self):
        """ARCDataset aug_info should have translational_offset AND offset_r/offset_c."""
        from sci_arc.data.dataset import ARCDataset
        
        data_root = Path("data/arc-agi/data")
        if not (data_root / "training").is_dir():
            pytest.skip("ARC data not available")
        
        ds = ARCDataset(
            str(data_root / "training"),
            max_size=30,
            augment=False,
            translational_augment=True,
            track_augmentation=True,
        )
        
        if len(ds) == 0:
            pytest.skip("No samples in dataset")
        
        sample = ds[0]
        assert 'aug_info' in sample, "Sample should have aug_info when track_augmentation=True"
        
        aug_info = sample['aug_info']
        # Check both key formats are present
        assert 'translational_offset' in aug_info, \
            "aug_info should have 'translational_offset' key"
        assert 'offset_r' in aug_info, \
            "aug_info should have 'offset_r' key for compatibility"
        assert 'offset_c' in aug_info, \
            "aug_info should have 'offset_c' key for compatibility"
    
    def test_key_formats_are_consistent(self):
        """Both key formats should have consistent values."""
        from sci_arc.data.dataset import ARCDataset
        
        data_root = Path("data/arc-agi/data")
        if not (data_root / "training").is_dir():
            pytest.skip("ARC data not available")
        
        ds = ARCDataset(
            str(data_root / "training"),
            max_size=30,
            augment=False,
            translational_augment=True,
            track_augmentation=True,
        )
        
        if len(ds) == 0:
            pytest.skip("No samples in dataset")
        
        sample = ds[0]
        aug_info = sample['aug_info']
        
        offset_tuple = aug_info['translational_offset']
        offset_r = aug_info['offset_r']
        offset_c = aug_info['offset_c']
        
        assert offset_tuple[0] == offset_r, \
            f"offset_r mismatch: tuple[0]={offset_tuple[0]}, offset_r={offset_r}"
        assert offset_tuple[1] == offset_c, \
            f"offset_c mismatch: tuple[1]={offset_tuple[1]}, offset_c={offset_c}"


@pytest.mark.cpu
class TestTRMStyleEvaluatorFixes:
    """Test TRMStyleEvaluator translation and context handling fixes."""
    
    def test_translation_helpers_exist(self):
        """TRMStyleEvaluator should have translation helper functions."""
        from sci_arc.evaluation import trm_style_evaluator as module
        
        assert hasattr(module, 'inverse_translation'), \
            "Module should have inverse_translation function"
        assert hasattr(module, 'get_translation_offset'), \
            "Module should have get_translation_offset function"
    
    def test_get_translation_offset_handles_both_formats(self):
        """get_translation_offset should handle both key formats."""
        from sci_arc.evaluation.trm_style_evaluator import get_translation_offset
        
        # Test tuple format
        aug_info_tuple = {'translational_offset': (5, 10)}
        r, c = get_translation_offset(aug_info_tuple)
        assert r == 5 and c == 10, f"Expected (5, 10), got ({r}, {c})"
        
        # Test separate keys format
        aug_info_separate = {'offset_r': 3, 'offset_c': 7}
        r, c = get_translation_offset(aug_info_separate)
        assert r == 3 and c == 7, f"Expected (3, 7), got ({r}, {c})"
        
        # Test both formats (tuple should take precedence)
        aug_info_both = {
            'translational_offset': (1, 2),
            'offset_r': 99,
            'offset_c': 99,
        }
        r, c = get_translation_offset(aug_info_both)
        assert r == 1 and c == 2, f"Tuple format should take precedence: got ({r}, {c})"
        
        # Test missing keys (should default to 0, 0)
        aug_info_empty = {}
        r, c = get_translation_offset(aug_info_empty)
        assert r == 0 and c == 0, f"Missing keys should default to (0, 0), got ({r}, {c})"
    
    def test_evaluate_with_trm_style_has_improved_context_handling(self):
        """evaluate_with_trm_style should properly extract context from batch."""
        from sci_arc.evaluation.trm_style_evaluator import evaluate_with_trm_style
        import inspect
        
        source = inspect.getsource(evaluate_with_trm_style)
        
        # Should handle multiple key name formats
        assert 'test_inputs' in source or 'input_grids' in source, \
            "Function should support RLAN batch key names"
        
        # Should pass train_inputs/train_outputs to model
        assert 'train_inputs=' in source, \
            "Function should pass train_inputs keyword arg to model"
        assert 'train_outputs=' in source, \
            "Function should pass train_outputs keyword arg to model"


@pytest.mark.cpu
class TestNoRegressions:
    """Test that fixes don't break existing functionality."""
    
    def test_collate_sci_arc_still_works(self):
        """collate_sci_arc should still produce valid batches."""
        import torch
        from sci_arc.data.dataset import collate_sci_arc
        
        # Create minimal samples
        samples = [
            {
                'task_id': 'test_0',
                'input_grids': [torch.randint(0, 10, (5, 5))],
                'output_grids': [torch.randint(0, 10, (5, 5))],
                'test_input': torch.randint(0, 10, (5, 5)),
                'test_output': torch.randint(0, 10, (5, 5)),
                'num_train_pairs': 1,
                'transform_family': 0,
                'original_max_size': 5,
            },
            {
                'task_id': 'test_1',
                'input_grids': [torch.randint(0, 10, (8, 8))],
                'output_grids': [torch.randint(0, 10, (8, 8))],
                'test_input': torch.randint(0, 10, (8, 8)),
                'test_output': torch.randint(0, 10, (8, 8)),
                'num_train_pairs': 1,
                'transform_family': 0,
                'original_max_size': 8,
            },
        ]
        
        batch = collate_sci_arc(samples, max_size=30)
        
        assert 'test_inputs' in batch
        assert 'test_outputs' in batch
        assert 'input_grids' in batch
        assert 'output_grids' in batch
        assert batch['test_inputs'].shape[0] == 2
    
    def test_rlan_forward_still_works(self):
        """RLAN forward pass should still work with context."""
        import torch
        from sci_arc.models import RLAN, RLANConfig
        
        config = RLANConfig(
            hidden_dim=32,
            num_solver_steps=1,
            use_context_encoder=True,
            use_dsc=False,
            use_msre=False,
            use_lcr=False,
            use_sph=False,
            dropout=0.0,
        )
        model = RLAN(config=config)
        model.eval()
        
        B, N, H, W = 2, 2, 8, 8
        test_inputs = torch.randint(0, 10, (B, H, W))
        train_inputs = torch.randint(0, 10, (B, N, H, W))
        train_outputs = torch.randint(0, 10, (B, N, H, W))
        pair_mask = torch.ones(B, N, dtype=torch.bool)
        
        with torch.no_grad():
            logits = model(
                input_grid=test_inputs,
                train_inputs=train_inputs,
                train_outputs=train_outputs,
                pair_mask=pair_mask,
            )
        
        assert logits.shape == (B, 10, H, W), f"Unexpected logits shape: {logits.shape}"
        assert torch.isfinite(logits).all(), "Logits should be finite"
    
    def test_trm_evaluator_update_still_works(self):
        """TRMStyleEvaluator.update should still work."""
        import numpy as np
        from sci_arc.evaluation.trm_style_evaluator import TRMStyleEvaluator
        
        evaluator = TRMStyleEvaluator(pass_Ks=[1, 2])
        
        gt = np.array([[1, 2], [3, 4]], dtype=np.uint8)
        pred = gt.copy()
        
        # Should not raise
        evaluator.update(
            task_id="test",
            prediction=pred,
            ground_truth=gt,
            aug_info={'dihedral_id': 0, 'translational_offset': (0, 0), 'offset_r': 0, 'offset_c': 0},
            confidence=1.0,
        )
        
        metrics = evaluator.compute_metrics()
        assert metrics['pass@1'] == 1.0, f"Expected pass@1=1.0, got {metrics['pass@1']}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
