#!/usr/bin/env python
"""
Quick RLAN Learning Test
========================
Minimal test to verify RLAN can learn on a small dataset.
Runs 50 epochs on 3 tasks with simplified output.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sci_arc.data import ARCDataset
from sci_arc.models import RLAN
from sci_arc.models.rlan import RLANConfig

def main():
    print("=" * 60)
    print("RLAN Quick Learning Test")
    print("=" * 60)
    
    # Config
    device = torch.device('cpu')
    max_tasks = 3  # Very small for fast test
    max_epochs = 50
    batch_size = 3  # Match task count
    lr = 3e-4  # Slightly higher for faster learning
    
    print(f"Device: {device}")
    print(f"Tasks: {max_tasks}, Epochs: {max_epochs}, Batch: {batch_size}")
    print()
    
    # Load data (NO augmentation for overfitting test)
    print("Loading data (NO augmentation - overfitting test)...")
    train_dataset = ARCDataset(
        './data/arc-agi/data/training',
        max_size=30,
        augment=False,  # NO augmentation
        max_tasks=max_tasks
    )
    print(f"Loaded {len(train_dataset.tasks)} tasks")
    
    # Create model
    print("\nCreating model...")
    config = RLANConfig(
        hidden_dim=128,  # Smaller for speed
        num_colors=10,
        num_classes=10,
        max_grid_size=30,
        max_clues=4,  # Fewer clues
        num_predicates=16,
        num_solver_steps=4,
        use_act=False,
        dropout=0.0,  # No dropout for overfitting
        use_context_encoder=True,
        use_dsc=True,
        use_msre=True,
        use_lcr=False,
        use_sph=False,
    )
    model = RLAN(config=config).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_params:,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    
    # Create batch from all samples
    print("\nPreparing fixed batch...")
    samples = [train_dataset[i] for i in range(len(train_dataset.tasks))]
    
    # Stack into batch - dataset uses test_input/test_output
    inputs = torch.stack([s['test_input'] for s in samples]).to(device)
    targets = torch.stack([s['test_output'] for s in samples]).to(device)
    
    # Create masks (valid pixels are not padding=-100)
    masks = targets != -100
    
    # Create demos from input/output pairs
    demos = []
    for s in samples:
        demo_pairs = list(zip(s['input_grids'], s['output_grids']))
        demos.append(demo_pairs)
    
    print(f"Batch shape: {inputs.shape}")
    print(f"Target unique values: {torch.unique(targets).tolist()}")
    
    # Training loop
    print("\n" + "=" * 60)
    print("Training (overfitting on fixed batch)")
    print("=" * 60)
    print(f"{'Epoch':>5} | {'Loss':>8} | {'Acc':>6} | {'Exact':>6} | {'Status':<20}")
    print("-" * 60)
    
    best_acc = 0
    best_exact = 0
    
    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        # Forward
        output = model(inputs, demos)
        # output can be dict (training) or tensor (inference)
        if isinstance(output, dict):
            logits = output['logits']  # (B, 10, H, W)
        else:
            logits = output  # Direct tensor
        
        # Loss (simple CE, ignore padding)
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')
        targets_loss = targets.clone()
        targets_loss[~masks] = -1
        loss = loss_fn(logits, targets_loss)
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Metrics
        preds = logits.argmax(dim=1)
        valid_mask = masks
        correct = (preds == targets) & valid_mask
        total_valid = valid_mask.sum().item()
        acc = correct.sum().item() / total_valid if total_valid > 0 else 0
        
        # Exact match
        exact_matches = 0
        for i in range(len(samples)):
            sample_mask = masks[i]
            sample_correct = correct[i][sample_mask].all().item()
            if sample_correct:
                exact_matches += 1
        exact_pct = exact_matches / len(samples) * 100
        
        # Status
        if acc > best_acc:
            best_acc = acc
            status = "↑ New best acc"
        elif exact_matches > best_exact:
            best_exact = exact_matches
            status = "↑ New best exact"
        else:
            status = ""
        
        # Print every 5 epochs or if improved
        if epoch % 5 == 0 or epoch == 1 or status:
            print(f"{epoch:>5} | {loss.item():>8.4f} | {acc*100:>5.1f}% | {exact_pct:>5.1f}% | {status:<20}")
    
    print("=" * 60)
    print(f"\nFinal Results:")
    print(f"  Best Accuracy: {best_acc*100:.1f}%")
    print(f"  Best Exact Match: {best_exact}/{len(samples)} ({best_exact/len(samples)*100:.0f}%)")
    
    if best_exact > 0:
        print("\n✅ SUCCESS: RLAN can overfit on training data!")
        print("   → Architecture is capable of learning")
    else:
        print("\n⚠️ WARNING: RLAN did not achieve any exact matches")
        print("   → May need more epochs or architecture investigation")
    
    return best_exact > 0

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
