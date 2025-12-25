
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytest
from pathlib import Path
import json

from sci_arc.models.rlan import RLAN, RLANConfig
from sci_arc.data.dataset import SCIARCDataset, collate_sci_arc as collate_fn
from sci_arc.evaluation.trm_style_evaluator import TRMStyleEvaluator, inverse_dihedral_transform, inverse_color_permutation

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_real_task(task_id="007bbfb7"):
    """Load a real ARC task for testing."""
    data_path = Path("c:/Users/perahmat/Downloads/SCI-ARC/data/arc-agi/data/training") / f"{task_id}.json"
    if not data_path.exists():
        pytest.skip(f"Data file {data_path} not found")
    
    with open(data_path, 'r') as f:
        task = json.load(f)
    return task

def create_model():
    """Create RLAN model with HyperLoRA enabled."""
    config = RLANConfig(
        hidden_dim=64,
        num_colors=11,
        num_classes=11,
        use_hyperlora=True,
        use_solver_context=True,
        num_solver_steps=2,
        dropout=0.1
    )
    model = RLAN(config=config).to(device)
    return model

class TestInferenceConsistency:
    
    def test_hyperlora_active_during_inference(self):
        """
        Hypothesis: HyperLoRA must be active during inference.
        This requires passing train_inputs/train_outputs (support set) to the forward pass.
        """
        model = create_model()
        model.eval()
        
        dataset = SCIARCDataset(
            data_dir="c:/Users/perahmat/Downloads/SCI-ARC/data/arc-agi/data",
            split='training',
            augment=False
        )
        
        # Get a sample
        sample = dataset[0]
        batch = collate_fn([sample])
        
        input_grids = batch['input_grids'].to(device)   # Support inputs
        output_grids = batch['output_grids'].to(device) # Support outputs
        test_input = batch['test_inputs'].to(device)    # Query input
        
        # 1. Run WITHOUT support set (Standard Inference Mistake)
        # If we just pass test_input, HyperLoRA should warn or be silent
        with torch.no_grad():
            # Note: RLAN.forward signature: (input_grid, train_inputs=None, train_outputs=None, ...)
            out_no_support = model(test_input, return_intermediates=True)
            
        # Check if lora_deltas is None
        assert out_no_support.get('lora_deltas') is None, "HyperLoRA should be inactive without support set"
        
        # 2. Run WITH support set (Correct Inference)
        with torch.no_grad():
            out_with_support = model(
                test_input,
                train_inputs=input_grids,
                train_outputs=output_grids,
                return_intermediates=True
            )
            
        # Check if lora_deltas is present
        deltas = out_with_support.get('lora_deltas')
        assert deltas is not None, "HyperLoRA should be active when support set is provided"
        
        # Check if deltas are non-zero (we fixed init to 0.1, so they should be significant)
        # deltas is a dict of tensors
        for name, delta in deltas.items():
            norm = delta.norm().item()
            print(f"LoRA Delta {name} norm: {norm}")
            assert norm > 0, f"LoRA delta {name} is zero! Init fix might have failed."

    def test_tta_consistency(self):
        """
        Hypothesis: TTA (Test Time Augmentation) followed by Inverse Augmentation
        should yield consistent results.
        """
        # We don't need a trained model, just need to verify the transform logic
        # using the evaluator's helper functions.
        
        # Create a dummy grid
        grid = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ], dtype=np.int32)
        
        # Test Dihedral 1 (Rotate 90 CCW)
        # Original:
        # 1 2 3
        # 4 5 6
        # 7 8 9
        #
        # Rot90:
        # 3 6 9
        # 2 5 8
        # 1 4 7
        
        from sci_arc.evaluation.trm_style_evaluator import dihedral_transform, inverse_dihedral_transform
        
        aug_grid = dihedral_transform(grid, 1)
        restored_grid = inverse_dihedral_transform(aug_grid, 1)
        
        np.testing.assert_array_equal(grid, restored_grid, "Dihedral inverse failed")
        
        # Test Color Permutation
        # Map: 1->2, 2->3, 3->1 (others identity)
        perm = np.arange(10) # 0-9
        perm[1] = 2
        perm[2] = 3
        perm[3] = 1
        
        # Apply permutation: grid value '1' becomes perm[1] = 2
        aug_grid_color = perm[grid]
        
        # Check value at (0,0) which was 1
        assert aug_grid_color[0,0] == 2
        
        # Inverse
        from sci_arc.evaluation.trm_style_evaluator import inverse_color_permutation
        restored_grid_color = inverse_color_permutation(aug_grid_color, perm)
        
        np.testing.assert_array_equal(grid, restored_grid_color, "Color permutation inverse failed")

    def test_gradient_flow_bottlenecks(self):
        """
        Hypothesis: Check for vanishing gradients in deep layers, specifically
        ContextEncoder -> HyperLoRA -> Solver.
        """
        model = create_model()
        model.train()
        
        dataset = SCIARCDataset(
            data_dir="c:/Users/perahmat/Downloads/SCI-ARC/data/arc-agi/data",
            split='training',
            augment=False
        )
        sample = dataset[0]
        batch = collate_fn([sample])
        
        input_grids = batch['input_grids'].to(device)
        output_grids = batch['output_grids'].to(device)
        test_input = batch['test_inputs'].to(device)
        test_output = batch['test_outputs'].to(device)
        
        # Forward
        outputs = model(
            test_input,
            train_inputs=input_grids,
            train_outputs=output_grids,
            return_intermediates=True
        )
        
        loss = F.cross_entropy(outputs['logits'].reshape(-1, 11), test_output.reshape(-1), ignore_index=-100)
        loss.backward()
        
        print("\nGradient Norms:")
        
        # Check Context Encoder grads
        ce_grads = []
        for name, param in model.context_encoder.named_parameters():
            if param.grad is not None:
                ce_grads.append(param.grad.norm().item())
        avg_ce = sum(ce_grads)/len(ce_grads) if ce_grads else 0
        print(f"ContextEncoder avg grad: {avg_ce:.6f}")
        
        # Check HyperLoRA grads
        hl_grads = []
        for name, param in model.hyper_lora.named_parameters():
            if param.grad is not None:
                hl_grads.append(param.grad.norm().item())
        avg_hl = sum(hl_grads)/len(hl_grads) if hl_grads else 0
        print(f"HyperLoRA avg grad:      {avg_hl:.6f}")
        
        # Check Solver grads
        solver_grads = []
        for name, param in model.solver.named_parameters():
            if param.grad is not None:
                solver_grads.append(param.grad.norm().item())
        avg_solver = sum(solver_grads)/len(solver_grads) if solver_grads else 0
        print(f"Solver avg grad:         {avg_solver:.6f}")
        
        # Assertions
        assert avg_ce > 0, "ContextEncoder gradients are zero! (Bottleneck)"
        assert avg_hl > 0, "HyperLoRA gradients are zero! (Bottleneck)"
        
        # Check relative scale
        # If CE is vanishingly small compared to Solver, we have a problem
        ratio = avg_ce / (avg_solver + 1e-9)
        print(f"Ratio CE/Solver:         {ratio:.4f}")
        
        # We want this ratio to be reasonable (e.g., > 0.01)
        # If it's 1e-5, the context encoder isn't learning from the solver's needs
        assert ratio > 0.01, f"ContextEncoder gradient is too small relative to Solver ({ratio:.6f}). Potential bottleneck."

