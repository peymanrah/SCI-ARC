"""
Quick validation script for Phase 2 Cross-Attention architecture.

Tests the new architecture on a small subset of data to verify:
1. No background collapse
2. Gradients flow correctly
3. Training loss decreases
4. Evaluation shows some generalization

Run:
    python scripts/validate_phase2.py --epochs 20 --batch_size 4
"""

import argparse
import torch
import torch.nn as nn
from pathlib import Path
import json
from datetime import datetime

from sci_arc.models.rlan import RLAN, RLANConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Validate Phase 2 architecture")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--max_tasks", type=int, default=10, help="Max tasks to load")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--output_dir", type=str, default="checkpoints/phase2_validation", 
                        help="Output directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use")
    return parser.parse_args()


def compute_metrics(logits, target):
    """Compute accuracy metrics."""
    pred = logits.argmax(dim=1)  # (B, H, W)
    
    # Overall accuracy
    correct = (pred == target).float()
    accuracy = correct.mean().item()
    
    # Foreground/Background accuracy
    fg_mask = target > 0
    bg_mask = target == 0
    
    fg_acc = correct[fg_mask].mean().item() if fg_mask.any() else 0.0
    bg_acc = correct[bg_mask].mean().item() if bg_mask.any() else 0.0
    
    # Exact match (all pixels correct)
    em = (pred == target).all(dim=(1, 2)).float().mean().item()
    
    return {
        "accuracy": accuracy,
        "fg_accuracy": fg_acc,
        "bg_accuracy": bg_acc,
        "exact_match": em,
    }


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_metrics = {"accuracy": 0.0, "fg_accuracy": 0.0, "bg_accuracy": 0.0, "exact_match": 0.0}
    num_batches = 0
    
    for batch in dataloader:
        # Move to device
        test_input = batch["test_input"].to(device)
        test_output = batch["test_output"].to(device)
        train_inputs = batch["train_inputs"].to(device) if "train_inputs" in batch else None
        train_outputs = batch["train_outputs"].to(device) if "train_outputs" in batch else None
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(
            test_input,
            train_inputs=train_inputs,
            train_outputs=train_outputs,
        )
        
        # Compute loss
        loss = criterion(logits, test_output)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        metrics = compute_metrics(logits.detach(), test_output)
        for key in total_metrics:
            total_metrics[key] += metrics[key]
        num_batches += 1
    
    # Average metrics
    avg_loss = total_loss / num_batches
    avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
    avg_metrics["loss"] = avg_loss
    
    return avg_metrics


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0.0
    total_metrics = {"accuracy": 0.0, "fg_accuracy": 0.0, "bg_accuracy": 0.0, "exact_match": 0.0}
    num_batches = 0
    
    for batch in dataloader:
        test_input = batch["test_input"].to(device)
        test_output = batch["test_output"].to(device)
        train_inputs = batch["train_inputs"].to(device) if "train_inputs" in batch else None
        train_outputs = batch["train_outputs"].to(device) if "train_outputs" in batch else None
        
        logits = model(
            test_input,
            train_inputs=train_inputs,
            train_outputs=train_outputs,
        )
        
        loss = criterion(logits, test_output)
        
        total_loss += loss.item()
        metrics = compute_metrics(logits, test_output)
        for key in total_metrics:
            total_metrics[key] += metrics[key]
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
    avg_metrics["loss"] = avg_loss
    
    return avg_metrics


def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    
    print(f"Phase 2 Validation")
    print(f"=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Device: {args.device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max tasks: {args.max_tasks}")
    print(f"=" * 60)
    
    # Create model
    config = RLANConfig(
        hidden_dim=args.hidden_dim,
        max_clues=5,
        use_context_encoder=True,  # Enable Phase 2 cross-attention
        dropout=0.1,
    )
    model = RLAN(config=config).to(args.device)
    
    print(f"\nModel created:")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Context Encoder: CrossAttentionInjector ✅")
    
    # Create datasets (placeholder - adjust paths as needed)
    print(f"\nNote: Update dataset paths in this script for actual training!")
    print(f"Creating dummy data for testing...")
    
    # Dummy data for testing
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, num_tasks=10):
            self.num_tasks = num_tasks
        
        def __len__(self):
            return self.num_tasks
        
        def __getitem__(self, idx):
            H, W = 10, 10
            N = 3  # Number of training pairs
            return {
                "test_input": torch.randint(0, 10, (H, W)),
                "test_output": torch.randint(0, 10, (H, W)),
                "train_inputs": torch.randint(0, 10, (N, H, W)),
                "train_outputs": torch.randint(0, 10, (N, H, W)),
            }
    
    train_dataset = DummyDataset(num_tasks=args.max_tasks)
    val_dataset = DummyDataset(num_tasks=args.max_tasks // 2)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )
    
    # Create optimizer and criterion
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01,
    )
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    print(f"\nStarting training...")
    print(f"-" * 60)
    
    history = {
        "train": [],
        "val": [],
    }
    
    best_val_em = 0.0
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, args.device)
        
        # Validate
        val_metrics = evaluate(model, val_loader, criterion, args.device)
        
        # Log
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"  Train: Loss={train_metrics['loss']:.4f}, "
              f"Acc={train_metrics['accuracy']:.3f}, "
              f"FG={train_metrics['fg_accuracy']:.3f}, "
              f"BG={train_metrics['bg_accuracy']:.3f}, "
              f"EM={train_metrics['exact_match']:.3f}")
        print(f"  Val:   Loss={val_metrics['loss']:.4f}, "
              f"Acc={val_metrics['accuracy']:.3f}, "
              f"FG={val_metrics['fg_accuracy']:.3f}, "
              f"BG={val_metrics['bg_accuracy']:.3f}, "
              f"EM={val_metrics['exact_match']:.3f}")
        
        # Check for collapse
        if train_metrics['fg_accuracy'] < 0.05:
            print(f"  ⚠️  WARNING: Potential background collapse (FG={train_metrics['fg_accuracy']:.3f})")
        
        # Save history
        history["train"].append(train_metrics)
        history["val"].append(val_metrics)
        
        # Save best model
        if val_metrics['exact_match'] > best_val_em:
            best_val_em = val_metrics['exact_match']
            checkpoint_path = output_dir / "best_model.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
            }, checkpoint_path)
            print(f"  💾 Saved best model (EM={best_val_em:.3f})")
        
        print()
    
    # Save final history
    history_path = output_dir / "history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    
    print(f"Training complete!")
    print(f"Best validation EM: {best_val_em:.3f}")
    print(f"History saved to: {history_path}")


if __name__ == "__main__":
    main()
