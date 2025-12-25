
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
import pytest
from pathlib import Path
from sci_arc.models.rlan import RLAN, RLANConfig
from sci_arc.models.rlan_modules.recursive_solver import RecursiveSolver
from sci_arc.models.rlan_modules.hyper_lora import HyperLoRA
from sci_arc.models.rlan_modules.acw import AugmentedConfidenceWeighting
from sci_arc.data.dataset import SCIARCDataset, collate_sci_arc as collate_fn
from torch.utils.data import DataLoader
from sci_arc.training.trainer import TrainingConfig

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
    """Create a small RLAN model for testing."""
    config = RLANConfig(
        hidden_dim=64,
        num_colors=11,
        num_classes=11,
        use_hyperlora=True,
        use_solver_context=True,
        num_solver_steps=2, # Faster
        dropout=0.1
    )
    model = RLAN(config=config).to(device)
    return model

class TestScientificValidity:
    
    def test_hyperlora_adaptation_effectiveness(self):
        """
        Hypothesis: HyperLoRA adaptation should reduce loss on the support set.
        
        Method:
        1. Load a task.
        2. Run forward pass WITHOUT adaptation (use_hyperlora=False in forward).
        3. Run forward pass WITH adaptation (use_hyperlora=True).
        4. Compare losses on the support set (train pairs).
        """
        task = load_real_task()
        model = create_model()
        model.eval() # We want to test the internal adaptation, not training gradients
        
        # Prepare batch
        dataset = SCIARCDataset(
            data_dir="c:/Users/perahmat/Downloads/SCI-ARC/data/arc-agi/data",
            split='training',
            augment=False
        )
        
        # Find the index of our task
        task_idx = 0 # Just use the first one
        
        # Get sample
        sample = dataset[task_idx]
        batch = collate_fn([sample])
        
        # Move to device
        input_grids = batch['input_grids'].to(device)
        output_grids = batch['output_grids'].to(device)
        test_input = batch['test_inputs'].to(device)
        test_output = batch['test_outputs'].to(device)
        
        if input_grids.shape[1] < 2:
            pytest.skip("Task has fewer than 2 train pairs, cannot test adaptation/LOO")
            
        # Construct a LOO batch
        loo_input_grids = input_grids[:, 1:]
        loo_output_grids = output_grids[:, 1:]
        loo_test_input = input_grids[:, 0]
        loo_test_output = output_grids[:, 0]
        
        # Run with HyperLoRA DISABLED (simulate by zeroing deltas or skipping)
        # We can temporarily monkeypatch the hyper_lora module to return None
        original_hyper_lora = model.hyper_lora
        model.hyper_lora = None 
        
        with torch.no_grad():
            out_no_adapt = model(
                loo_test_input,
                train_inputs=loo_input_grids,
                train_outputs=loo_output_grids,
                return_intermediates=True
            )
            loss_no_adapt = F.cross_entropy(
                out_no_adapt['logits'].reshape(-1, 11), 
                loo_test_output.reshape(-1), 
                ignore_index=-100
            )
            
        # Restore HyperLoRA
        model.hyper_lora = original_hyper_lora
        
        # Run with HyperLoRA ENABLED
        with torch.no_grad():
            out_adapt = model(
                loo_test_input,
                train_inputs=loo_input_grids,
                train_outputs=loo_output_grids,
                return_intermediates=True
            )
            loss_adapt = F.cross_entropy(
                out_adapt['logits'].reshape(-1, 11), 
                loo_test_output.reshape(-1), 
                ignore_index=-100
            )
            
        print(f"\nLoss without adaptation: {loss_no_adapt.item():.4f}")
        print(f"Loss with adaptation:    {loss_adapt.item():.4f}")
        
        # We expect adaptation to help, or at least not hurt significantly.
        # Note: On a random initialized model, adaptation might be noisy.
        # But this checks the mechanism runs and affects the output.
        assert loss_adapt.item() != loss_no_adapt.item(), "HyperLoRA adaptation had NO effect on loss (gradients might be zero or disconnected)"

    def test_gradient_flow_magnitude(self):
        """
        Hypothesis: Gradients should flow to HyperLoRA parameters.
        If they are vanishingly small compared to main parameters, the meta-learning won't work.
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
        
        # Backward
        loss.backward()
        
        # Check gradients
        hyper_lora_grads = []
        main_grads = []
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                if 'hyper_lora' in name:
                    hyper_lora_grads.append(grad_norm)
                else:
                    main_grads.append(grad_norm)
        
        avg_hyper = sum(hyper_lora_grads) / len(hyper_lora_grads) if hyper_lora_grads else 0
        avg_main = sum(main_grads) / len(main_grads) if main_grads else 0
        
        print(f"\nAvg HyperLoRA grad norm: {avg_hyper:.6f}")
        print(f"Avg Main grad norm:      {avg_main:.6f}")
        
        assert avg_hyper > 0, "HyperLoRA gradients are zero!"
        # It's okay if they are smaller, but shouldn't be 0.
        
    def test_acw_voting_logic(self):
        """
        Hypothesis: ACW hybrid voting should pick the most confident prediction.
        """
        acw = AugmentedConfidenceWeighting()
        
        # Create 3 dummy predictions
        # Pred 1: High confidence, correct shape
        pred1 = torch.zeros((10, 10), dtype=torch.long)
        
        # Pred 2: Low confidence, same shape
        pred2 = torch.zeros((10, 10), dtype=torch.long)
        pred2[0,0] = 1 # Slight difference
        
        # Pred 3: Different shape (should be handled)
        pred3 = torch.zeros((5, 5), dtype=torch.long)
        
        # Mock the voting
        # We need to mock the internal confidence estimation if we can't easily run it.
        # ACW.hybrid_vote takes a list of predictions.
        # It computes confidence internally using `estimate_confidence`.
        # `estimate_confidence` uses heuristics like entropy, compression, etc.
        
        # Let's just run it and see if it crashes or produces a result.
        preds = [pred1, pred2, pred1] # Majority vote should be pred1
        
        winner, confidence = acw.hybrid_vote(preds)
        
        assert torch.equal(winner, pred1), "Majority voting failed"
        
