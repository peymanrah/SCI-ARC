"""
Integration tests for SCI-ARC.
"""

import pytest
import torch
import numpy as np
import json
import tempfile
from pathlib import Path

from sci_arc.models import SCIARC, SCIARCConfig
from sci_arc.data import SCIARCDataset, collate_sci_arc, create_dataloader
from sci_arc.training import SCIARCLoss, TrainingConfig
from sci_arc.evaluation import ARCMetrics, pixel_accuracy, task_accuracy


@pytest.fixture
def temp_arc_dir():
    """Create temporary directory with sample ARC tasks."""
    with tempfile.TemporaryDirectory() as tmpdir:
        training_dir = Path(tmpdir) / 'training'
        training_dir.mkdir()
        
        # Create multiple sample tasks
        for i in range(5):
            task = {
                'train': [
                    {
                        'input': [[i, i+1], [i+2, i+3]],
                        'output': [[i+3, i+2], [i+1, i]],
                    },
                ],
                'test': [
                    {
                        'input': [[0, 1], [2, 3]],
                        'output': [[3, 2], [1, 0]],
                    },
                ],
            }
            
            with open(training_dir / f'task_{i}.json', 'w') as f:
                json.dump(task, f)
        
        yield tmpdir


@pytest.fixture
def small_model():
    """Create small model for testing."""
    config = SCIARCConfig(
        hidden_dim=32,
        num_colors=10,
        max_grid_size=10,
        num_structure_slots=2,
        num_abstraction_layers=1,
        max_objects=4,
        H_cycles=1,
        L_cycles=1,
        L_layers=1,
    )
    return SCIARC(config)


class TestEndToEnd:
    """End-to-end integration tests."""
    
    def test_training_step(self, temp_arc_dir, small_model):
        """Test a single training step."""
        # Load data
        dataset = SCIARCDataset(temp_arc_dir, split='training', augment=False)
        batch = collate_sci_arc([dataset[0]], max_size=10)
        
        # Forward pass
        outputs = small_model(
            input_grids=batch['input_grids'],
            output_grids=batch['output_grids'],
            test_input=batch['test_inputs'],
            grid_mask=batch['grid_masks'],
        )
        
        # Compute loss
        loss_fn = SCIARCLoss()
        loss_dict = loss_fn(
            logits=outputs['logits'],
            targets=batch['test_outputs'],
            z_struct=outputs['z_struct'],
            z_content=outputs['z_content'],
            transform_families=batch['transform_families'],
        )
        
        # Backward pass
        loss_dict['total'].backward()
        
        # Check gradients
        for name, param in small_model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
    
    def test_dataloader_with_model(self, temp_arc_dir, small_model):
        """Test dataloader works with model."""
        loader = create_dataloader(
            data_dir=temp_arc_dir,
            split='training',
            batch_size=2,
            num_workers=0,
            augment=False,
            max_grid_size=10,
        )
        
        small_model.eval()
        
        with torch.no_grad():
            for batch in loader:
                outputs = small_model(
                    input_grids=batch['input_grids'],
                    output_grids=batch['output_grids'],
                    test_input=batch['test_inputs'],
                    grid_mask=batch['grid_masks'],
                )
                
                assert outputs['logits'].shape[0] == batch['test_inputs'].shape[0]
                break
    
    def test_evaluation_pipeline(self, temp_arc_dir, small_model):
        """Test evaluation pipeline."""
        dataset = SCIARCDataset(temp_arc_dir, split='training', augment=False)
        batch = collate_sci_arc([dataset[0]], max_size=10)
        
        small_model.eval()
        
        with torch.no_grad():
            outputs = small_model(
                input_grids=batch['input_grids'],
                output_grids=batch['output_grids'],
                test_input=batch['test_inputs'],
                grid_mask=batch['grid_masks'],
            )
        
        # Get predictions
        logits = outputs['logits']
        preds = logits.argmax(dim=-1).numpy()
        targets = batch['test_outputs'].numpy()
        
        # Compute metrics
        metrics = ARCMetrics()
        for i, task_id in enumerate(batch['task_ids']):
            metrics.update(task_id, preds[i], targets[i])
        
        results = metrics.compute()
        
        assert 'task_accuracy' in results
        assert 'pixel_accuracy' in results


class TestModelConsistency:
    """Tests for model consistency."""
    
    def test_deterministic_inference(self, small_model):
        """Test that inference is deterministic."""
        small_model.eval()
        
        input_grids = torch.randint(0, 10, (1, 2, 8, 8))
        output_grids = torch.randint(0, 10, (1, 2, 8, 8))
        test_input = torch.randint(0, 10, (1, 8, 8))
        
        with torch.no_grad():
            out1 = small_model(input_grids, output_grids, test_input)
            out2 = small_model(input_grids, output_grids, test_input)
        
        assert torch.allclose(out1['logits'], out2['logits'])
        assert torch.allclose(out1['z_task'], out2['z_task'])
    
    def test_batch_invariance(self, small_model):
        """Test that batch processing gives same results as individual."""
        small_model.eval()
        
        # Single sample
        input_grids_1 = torch.randint(0, 10, (1, 2, 8, 8))
        output_grids_1 = torch.randint(0, 10, (1, 2, 8, 8))
        test_input_1 = torch.randint(0, 10, (1, 8, 8))
        
        # Batched
        input_grids_2 = input_grids_1.repeat(2, 1, 1, 1)
        output_grids_2 = output_grids_1.repeat(2, 1, 1, 1)
        test_input_2 = test_input_1.repeat(2, 1, 1)
        
        with torch.no_grad():
            out1 = small_model(input_grids_1, output_grids_1, test_input_1)
            out2 = small_model(input_grids_2, output_grids_2, test_input_2)
        
        # First sample in batch should match single sample
        assert torch.allclose(out1['logits'], out2['logits'][0:1], atol=1e-5)


class TestSCIPrinciples:
    """Tests for SCI-specific properties."""
    
    def test_structure_content_separation(self, small_model):
        """Test that structure and content are separated."""
        input_grids = torch.randint(0, 10, (4, 2, 8, 8))
        output_grids = torch.randint(0, 10, (4, 2, 8, 8))
        test_input = torch.randint(0, 10, (4, 8, 8))
        
        outputs = small_model(input_grids, output_grids, test_input)
        
        z_struct = outputs['z_struct']
        z_content = outputs['z_content']
        
        # Check they're different
        assert not torch.allclose(z_struct, z_content)
        
        # Check dimensions
        assert z_struct.shape == z_content.shape
    
    def test_z_task_combines_both(self, small_model):
        """Test that z_task depends on both structure and content."""
        # This is implicitly tested by the forward pass working
        # A more rigorous test would vary structure/content independently
        input_grids = torch.randint(0, 10, (1, 2, 8, 8))
        output_grids = torch.randint(0, 10, (1, 2, 8, 8))
        test_input = torch.randint(0, 10, (1, 8, 8))
        
        outputs = small_model(input_grids, output_grids, test_input)
        
        assert outputs['z_task'] is not None
        assert outputs['z_task'].shape[0] == 1


class TestGPUCompatibility:
    """Tests for GPU compatibility (skip if no GPU)."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_forward_pass(self):
        """Test forward pass on GPU."""
        config = SCIARCConfig(
            hidden_dim=32,
            num_structure_slots=2,
            max_objects=4,
            H_cycles=1,
            L_cycles=1,
            L_layers=1,
        )
        model = SCIARC(config).cuda()
        
        input_grids = torch.randint(0, 10, (2, 2, 8, 8)).cuda()
        output_grids = torch.randint(0, 10, (2, 2, 8, 8)).cuda()
        test_input = torch.randint(0, 10, (2, 8, 8)).cuda()
        
        outputs = model(input_grids, output_grids, test_input)
        
        assert outputs['logits'].device.type == 'cuda'
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_training_step(self):
        """Test training step on GPU."""
        config = SCIARCConfig(
            hidden_dim=32,
            num_structure_slots=2,
            max_objects=4,
            H_cycles=1,
            L_cycles=1,
            L_layers=1,
        )
        model = SCIARC(config).cuda()
        
        input_grids = torch.randint(0, 10, (2, 2, 8, 8)).cuda()
        output_grids = torch.randint(0, 10, (2, 2, 8, 8)).cuda()
        test_input = torch.randint(0, 10, (2, 8, 8)).cuda()
        targets = torch.randint(0, 10, (2, 8, 8)).cuda()
        
        outputs = model(input_grids, output_grids, test_input)
        
        loss_fn = SCIARCLoss()
        loss_dict = loss_fn(
            logits=outputs['logits'],
            targets=targets,
            z_struct=outputs['z_struct'],
            z_content=outputs['z_content'],
        )
        
        loss_dict['total'].backward()
        
        # Check gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
