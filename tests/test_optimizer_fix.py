"""
Test to verify HyperLoRA gets 10x learning rate in optimizer.
"""
import torch
from torch.utils.data import DataLoader
from sci_arc.training.trainer import SCIARCTrainer, TrainingConfig
from sci_arc.models.rlan import RLAN, RLANConfig


def test_hyperlora_lr_multiplier():
    """Verify HyperLoRA parameters get 10x learning rate."""
    # Create model with HyperLoRA enabled
    config = RLANConfig(
        hidden_dim=64,
        use_hyperlora=True,
        use_solver_context=True,
        num_solver_steps=2
    )
    model = RLAN(config=config)
    
    # Create training config
    t_config = TrainingConfig(
        device='cpu',
        learning_rate=3e-4,
        use_wandb=False,
        log_to_file=False
    )
    
    # Mock minimal requirements
    class MockLoss:
        def to(self, device):
            return self
    
    fake_loader = DataLoader([torch.zeros(1)], batch_size=1)
    
    # Create trainer (this creates the optimizer)
    trainer = SCIARCTrainer(
        model=model,
        train_loader=fake_loader,
        val_loader=None,
        loss_fn=MockLoss(),
        config=t_config
    )
    
    # Check param groups
    print(f"\nNumber of param groups: {len(trainer.optimizer.param_groups)}")
    
    base_lr = t_config.learning_rate
    hyperlora_found = False
    
    for i, group in enumerate(trainer.optimizer.param_groups):
        lr = group.get('lr', base_lr)
        num_params = len(group['params'])
        print(f"Group {i}: LR={lr:.6f}, params={num_params}")
        
        # Check if this is the HyperLoRA group (10x LR)
        if lr > base_lr * 5:  # Looking for 10x, use 5x as threshold
            hyperlora_found = True
            expected_lr = base_lr * 10
            assert abs(lr - expected_lr) < 1e-9, f"Expected {expected_lr}, got {lr}"
            print(f"  -> HyperLoRA group detected! LR = {lr/base_lr:.1f}x base")
    
    assert hyperlora_found, "No HyperLoRA param group with 10x LR found!"
    print("\n✓ HyperLoRA LR fix verified: meta-learner gets 10x learning rate")


if __name__ == '__main__':
    test_hyperlora_lr_multiplier()
