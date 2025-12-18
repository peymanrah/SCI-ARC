#!/usr/bin/env python3
"""
Quick Clue Regularization Test - verify settings don't break training.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from sci_arc.models.rlan import RLAN, RLANConfig
from sci_arc.training.rlan_loss import RLANLoss


def create_simple_task():
    """Simple 2x expansion task."""
    inp = np.random.randint(1, 5, (3, 3))
    out = np.repeat(np.repeat(inp, 2, axis=0), 2, axis=1)
    return inp, out


def pad_grid(grid, size=10, is_target=False):
    h, w = grid.shape
    padded = np.full((size, size), -100 if is_target else 10, dtype=np.int64)
    padded[:h, :w] = grid
    return padded


def test_config(name, loss_kwargs, max_epochs=30):
    """Test a specific config and return epochs to converge."""
    print(f"\n{name}:")
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = 'cpu'
    
    config = RLANConfig(
        hidden_dim=64,
        max_clues=4,
        num_predicates=8,
        num_solver_steps=2,
        max_grid_size=10,
        use_context_encoder=True,
    )
    model = RLAN(config=config).to(device)
    loss_fn = RLANLoss(**loss_kwargs).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    
    # Prepare single sample batch
    demos_in = []
    demos_out = []
    for _ in range(2):
        inp, out = create_simple_task()
        demos_in.append(torch.from_numpy(pad_grid(inp, 10, False)))
        demos_out.append(torch.from_numpy(pad_grid(out, 10, True)))
    
    test_in, test_out = create_simple_task()
    
    demo_inputs = torch.stack(demos_in, dim=0).unsqueeze(0).to(device)
    demo_outputs = torch.stack(demos_out, dim=0).unsqueeze(0).to(device)
    test_input = torch.from_numpy(pad_grid(test_in, 10, False)).unsqueeze(0).to(device)
    test_target = torch.from_numpy(pad_grid(test_out, 10, True)).unsqueeze(0).to(device)
    
    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        outputs = model(test_input, train_inputs=demo_inputs, train_outputs=demo_outputs, return_intermediates=True)
        
        loss_dict = loss_fn(
            logits=outputs['logits'],
            targets=test_target,
            stop_logits=outputs.get('stop_logits'),
            attention_maps=outputs.get('attention_maps'),
            predicates=outputs.get('predicates'),
            epoch=epoch,
            max_epochs=max_epochs,
        )
        
        loss = loss_dict['total_loss']
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Check accuracy
        with torch.no_grad():
            preds = outputs['logits'].argmax(dim=1)
            valid = test_target != -100
            if valid.sum() > 0:
                acc = ((preds == test_target) & valid).sum().float() / valid.sum()
                if acc.item() == 1.0:
                    clues = loss_dict.get('expected_clues_used', 0)
                    print(f"  ✓ 100% at epoch {epoch}, clues={clues:.1f}")
                    return epoch
    
    print(f"  ✗ Did not converge in {max_epochs} epochs")
    return None


def main():
    print("=" * 60)
    print("QUICK CLUE REGULARIZATION VERIFICATION")
    print("=" * 60)
    
    # Test WITHOUT clue reg
    epochs_without = test_config(
        "WITHOUT Clue Reg (weights=0)",
        {
            'lambda_sparsity': 0.5,
            'min_clues': 2.0,
            'min_clue_weight': 0.0,
            'ponder_weight': 0.0,
            'entropy_ponder_weight': 0.0,
        }
    )
    
    # Test WITH clue reg
    epochs_with = test_config(
        "WITH Clue Reg (weights active)",
        {
            'lambda_sparsity': 0.5,
            'min_clues': 2.5,
            'min_clue_weight': 5.0,
            'ponder_weight': 0.02,
            'entropy_ponder_weight': 0.02,
        }
    )
    
    print("\n" + "=" * 60)
    print("RESULT:")
    print("=" * 60)
    
    if epochs_without and epochs_with:
        diff = epochs_with - epochs_without
        print(f"  WITHOUT clue reg: {epochs_without} epochs")
        print(f"  WITH clue reg:    {epochs_with} epochs (diff: {diff:+d})")
        
        if abs(diff) <= 5:
            print("\n✅ Clue regularization is STABLE (similar convergence)")
        elif diff > 0:
            print(f"\n⚠️  Clue regularization adds {diff} epochs")
        else:
            print(f"\n✅ Clue regularization is FASTER by {-diff} epochs")
    elif epochs_without and not epochs_with:
        print("⚠️  Clue regularization BREAKS training")
    elif not epochs_without and epochs_with:
        print("✅ Clue regularization HELPS training converge")
    else:
        print("⚠️  Neither config converged - reduce task complexity")


if __name__ == '__main__':
    main()
