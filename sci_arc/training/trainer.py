"""
SCI-ARC Training Loop.

Implements:
1. Full training loop with mixed precision
2. Curriculum learning schedule
3. Deep supervision (from TRM)
4. Wandb logging
5. Checkpoint saving/loading
6. Gradient accumulation
7. File-based logging for reproducibility
"""

import os
import sys
import time
import math
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, List
from dataclasses import dataclass, field
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class TeeLogger:
    """
    Logger that writes to both stdout and a file.
    Captures all print() output for debugging and reproducibility.
    """
    def __init__(self, log_path: Path):
        self.terminal = sys.stdout
        self.log_path = log_path
        self.log_file = open(log_path, 'w', encoding='utf-8', buffering=1)  # Line buffered
        
    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()  # Ensure immediate write
        
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()
        
    def close(self):
        self.log_file.close()
        sys.stdout = self.terminal


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Optimization
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    max_epochs: int = 100
    warmup_epochs: int = 5
    grad_clip: float = 1.0
    grad_accumulation_steps: int = 1
    
    # Batch size
    batch_size: int = 32
    eval_batch_size: int = 64
    
    # Loss weights
    scl_weight: float = 0.1
    ortho_weight: float = 0.01
    deep_supervision_weight: float = 0.5
    
    # Scheduler
    scheduler_type: str = 'cosine'  # 'cosine', 'onecycle', 'constant'
    min_lr: float = 1e-6
    
    # Mixed precision
    use_amp: bool = True
    
    # Checkpointing
    checkpoint_dir: str = './checkpoints'
    save_every: int = 5
    keep_last_n: int = 3
    
    # Logging
    log_every: int = 10
    eval_every: int = 1  # Every N epochs
    use_wandb: bool = True
    wandb_project: str = 'sci-arc'
    wandb_run_name: Optional[str] = None
    log_to_file: bool = True  # Enable file logging
    
    # Curriculum learning
    use_curriculum: bool = True
    curriculum_stages: List[int] = field(default_factory=lambda: [10, 30, 60])  # Epoch thresholds
    
    # Device
    device: str = 'cuda'
    
    # Reproducibility
    seed: int = 42


class SCIARCTrainer:
    """
    Trainer for SCI-ARC model.
    
    Implements full training loop with:
    - Curriculum learning
    - Deep supervision
    - Mixed precision
    - Logging and checkpointing
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        loss_fn: nn.Module,
        config: TrainingConfig,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.config = config
        
        # Move model to device
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Mixed precision
        self.scaler = GradScaler('cuda') if config.use_amp and self.device.type == 'cuda' else None
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.best_val_accuracy = 0.0
        
        # Checkpoint directory
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # File logging
        self.tee_logger = None
        if config.log_to_file:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_path = self.checkpoint_dir / f'training_log_{timestamp}.txt'
            self.tee_logger = TeeLogger(log_path)
            sys.stdout = self.tee_logger
            print(f"Logging to: {log_path}")
            print(f"Timestamp: {datetime.now().isoformat()}")
            print(f"Python: {sys.version}")
            print(f"PyTorch: {torch.__version__}")
            print(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        
        # Initialize wandb
        self.wandb_run = None
        if config.use_wandb and WANDB_AVAILABLE:
            self._init_wandb()
    
    def _create_optimizer(self) -> AdamW:
        """Create AdamW optimizer with weight decay filtering."""
        # Separate parameters that should and shouldn't have weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if 'bias' in name or 'norm' in name or 'embedding' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        param_groups = [
            {'params': decay_params, 'weight_decay': self.config.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ]
        
        return AdamW(param_groups, lr=self.config.learning_rate)
    
    def _create_scheduler(self):
        """Create learning rate scheduler."""
        total_steps = len(self.train_loader) * self.config.max_epochs
        warmup_steps = len(self.train_loader) * self.config.warmup_epochs
        
        if self.config.scheduler_type == 'cosine':
            return CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps - warmup_steps,
                eta_min=self.config.min_lr,
            )
        elif self.config.scheduler_type == 'onecycle':
            return OneCycleLR(
                self.optimizer,
                max_lr=self.config.learning_rate,
                total_steps=total_steps,
                pct_start=warmup_steps / total_steps,
                anneal_strategy='cos',
            )
        else:
            return None
    
    def _init_wandb(self):
        """Initialize Weights & Biases logging."""
        run_name = self.config.wandb_run_name or f"sci-arc-{time.strftime('%Y%m%d-%H%M%S')}"
        
        self.wandb_run = wandb.init(
            project=self.config.wandb_project,
            name=run_name,
            config={
                'model': type(self.model).__name__,
                'training': self.config.__dict__,
                'model_params': sum(p.numel() for p in self.model.parameters()),
            }
        )
    
    def _warmup_lr(self, step: int, warmup_steps: int):
        """Linear warmup."""
        if step < warmup_steps:
            lr_scale = float(step) / float(max(1, warmup_steps))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.config.learning_rate * lr_scale
    
    def _get_curriculum_stage(self, epoch: int) -> int:
        """Get curriculum stage based on epoch."""
        if not self.config.use_curriculum:
            return 0  # All data
        
        for i, threshold in enumerate(self.config.curriculum_stages):
            if epoch < threshold:
                return i + 1
        return 0  # After all stages, use all data
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        epoch_losses = {
            'total': 0.0,
            'task': 0.0,
            'scl': 0.0,
            'ortho': 0.0,
            'deep': 0.0,
        }
        num_batches = 0
        
        warmup_steps = len(self.train_loader) * self.config.warmup_epochs
        epoch_start_time = time.time()
        batch_start_time = time.time()
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            batch = self._to_device(batch)
            
            # Warmup
            self._warmup_lr(self.global_step, warmup_steps)
            
            # Forward pass with mixed precision
            if self.scaler:
                with autocast('cuda'):
                    outputs = self.model.forward_training(
                        input_grids=batch['input_grids'],
                        output_grids=batch['output_grids'],
                        test_input=batch['test_inputs'],
                        test_output=batch['test_outputs'],
                        grid_mask=batch.get('grid_masks'),
                    )
                    losses = self._compute_losses(outputs, batch)
            else:
                outputs = self.model.forward_training(
                    input_grids=batch['input_grids'],
                    output_grids=batch['output_grids'],
                    test_input=batch['test_inputs'],
                    test_output=batch['test_outputs'],
                    grid_mask=batch.get('grid_masks'),
                )
                losses = self._compute_losses(outputs, batch)
            
            total_loss = losses['total'] / self.config.grad_accumulation_steps
            
            # Backward pass
            if self.scaler:
                self.scaler.scale(total_loss).backward()
            else:
                total_loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.config.grad_accumulation_steps == 0:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                
                if self.scheduler and self.global_step >= warmup_steps:
                    self.scheduler.step()
            
            # Accumulate losses
            for key in epoch_losses:
                if key in losses:
                    epoch_losses[key] += losses[key].item()
            num_batches += 1
            self.global_step += 1
            
            # Logging with timing
            if batch_idx % self.config.log_every == 0:
                batch_time = time.time() - batch_start_time
                self._log_step(batch_idx, losses, batch_time)
                batch_start_time = time.time()
        
        # Log epoch summary
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {self.current_epoch + 1} completed in {epoch_time:.1f}s "
              f"({epoch_time/num_batches:.2f}s/batch)")
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= max(num_batches, 1)
        
        return epoch_losses
    
    def _compute_losses(self, outputs: Dict, batch: Dict) -> Dict[str, torch.Tensor]:
        """Compute all losses."""
        losses = {}
        
        # Main task loss (grid prediction)
        logits = outputs['logits']  # [B, H, W, num_colors]
        targets = batch['test_outputs']  # [B, H, W]
        
        # Reshape for cross entropy
        B, H, W, C = logits.shape
        logits_flat = logits.view(B * H * W, C)
        targets_flat = targets.view(B * H * W)
        
        task_loss = F.cross_entropy(logits_flat, targets_flat)
        losses['task'] = task_loss
        
        # Deep supervision loss (intermediate predictions)
        deep_loss = torch.tensor(0.0, device=self.device)
        if 'intermediate_logits' in outputs and outputs['intermediate_logits']:
            for inter_logits in outputs['intermediate_logits']:
                inter_flat = inter_logits.view(B * H * W, C)
                deep_loss += F.cross_entropy(inter_flat, targets_flat)
            deep_loss /= len(outputs['intermediate_logits'])
        losses['deep'] = deep_loss
        
        # SCL loss - use the loss_fn's scl component if available
        scl_loss = torch.tensor(0.0, device=self.device)
        if 'z_struct' in outputs and 'transform_families' in batch:
            z_struct = outputs['z_struct']  # [B, K, D]
            transform_families = batch['transform_families'].to(self.device)  # [B]
            
            # Use the SCL component from the loss function
            if hasattr(self.loss_fn, 'scl'):
                scl_loss = self.loss_fn.scl(z_struct, transform_families)
        losses['scl'] = scl_loss
        
        # Orthogonality loss - use the loss_fn's orthogonality component
        ortho_loss = torch.tensor(0.0, device=self.device)
        if 'z_struct' in outputs and 'z_content' in outputs:
            z_struct = outputs['z_struct']  # [B, K, D]
            z_content = outputs['z_content']  # [B, M, D]
            
            if hasattr(self.loss_fn, 'orthogonality'):
                ortho_loss = self.loss_fn.orthogonality(z_struct, z_content)
        losses['ortho'] = ortho_loss
        
        # Total loss
        total = task_loss
        total = total + self.config.scl_weight * scl_loss
        total = total + self.config.ortho_weight * ortho_loss
        total = total + self.config.deep_supervision_weight * deep_loss
        losses['total'] = total
        
        return losses
    
    def _to_device(self, batch: Dict) -> Dict:
        """Move batch tensors to device."""
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device)
            else:
                device_batch[key] = value
        return device_batch
    
    def _log_step(self, batch_idx: int, losses: Dict, batch_time: float = 0.0):
        """Log training step with timing."""
        lr = self.optimizer.param_groups[0]['lr']
        
        # Display epoch as 1-indexed to match header (Epoch 1/100)
        log_str = f"Epoch {self.current_epoch + 1} [{batch_idx + 1}/{len(self.train_loader)}] "
        log_str += f"Loss: {losses['total'].item():.4f} "
        log_str += f"(task={losses['task'].item():.4f}, "
        log_str += f"scl={losses['scl'].item():.4f}, "
        log_str += f"ortho={losses['ortho'].item():.4f}) "
        log_str += f"LR: {lr:.2e} "
        log_str += f"[{batch_time:.2f}s]"
        
        print(log_str)
        
        if self.wandb_run:
            wandb.log({
                'train/loss_total': losses['total'].item(),
                'train/loss_task': losses['task'].item(),
                'train/loss_scl': losses['scl'].item(),
                'train/loss_ortho': losses['ortho'].item(),
                'train/loss_deep': losses['deep'].item(),
                'train/lr': lr,
                'train/epoch': self.current_epoch,
                'train/step': self.global_step,
            })
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation."""
        if not self.val_loader:
            return {}
        
        self.model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_pixels = 0
        total_tasks_correct = 0
        total_tasks = 0
        
        for batch in self.val_loader:
            batch = self._to_device(batch)
            
            outputs = self.model(
                input_grids=batch['input_grids'],
                output_grids=batch['output_grids'],
                test_input=batch['test_inputs'],
                grid_mask=batch['grid_masks'],
            )
            
            losses = self._compute_losses(outputs, batch)
            total_loss += losses['total'].item()
            
            # Compute accuracy
            logits = outputs['logits']  # [B, H, W, C]
            preds = logits.argmax(dim=-1)  # [B, H, W]
            targets = batch['test_outputs']
            
            # Pixel accuracy
            correct = (preds == targets).sum()
            total_correct += correct.item()
            total_pixels += targets.numel()
            
            # Task accuracy (entire grid must match)
            for i in range(preds.shape[0]):
                if torch.all(preds[i] == targets[i]):
                    total_tasks_correct += 1
                total_tasks += 1
        
        metrics = {
            'val_loss': total_loss / len(self.val_loader),
            'pixel_accuracy': total_correct / total_pixels,
            'task_accuracy': total_tasks_correct / total_tasks,
        }
        
        print(f"\nValidation: Loss={metrics['val_loss']:.4f}, "
              f"Pixel Acc={metrics['pixel_accuracy']:.4f}, "
              f"Task Acc={metrics['task_accuracy']:.4f}")
        
        if self.wandb_run:
            wandb.log({
                'val/loss': metrics['val_loss'],
                'val/pixel_accuracy': metrics['pixel_accuracy'],
                'val/task_accuracy': metrics['task_accuracy'],
                'val/epoch': self.current_epoch,
            })
        
        return metrics
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'best_val_loss': self.best_val_loss,
            'best_val_accuracy': self.best_val_accuracy,
            'config': self.config.__dict__,
        }
        
        # Save latest
        path = self.checkpoint_dir / f'checkpoint_epoch_{self.current_epoch}.pt'
        torch.save(checkpoint, path)
        
        # Save best
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"Saved best model to {best_path}")
        
        # Clean old checkpoints
        self._cleanup_checkpoints()
    
    def _cleanup_checkpoints(self):
        """Keep only the last N checkpoints."""
        checkpoints = sorted(
            self.checkpoint_dir.glob('checkpoint_epoch_*.pt'),
            key=lambda p: int(p.stem.split('_')[-1])
        )
        
        while len(checkpoints) > self.config.keep_last_n:
            old_ckpt = checkpoints.pop(0)
            old_ckpt.unlink()
    
    def load_checkpoint(self, path: str):
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if checkpoint['scaler_state_dict'] and self.scaler:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_val_accuracy = checkpoint['best_val_accuracy']
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def train(self):
        """Full training loop."""
        print(f"\nStarting training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"\n{'='*60}")
        print("Training Configuration:")
        print(f"{'='*60}")
        for key, value in self.config.__dict__.items():
            print(f"  {key}: {value}")
        print(f"{'='*60}\n")
        
        for epoch in range(self.current_epoch, self.config.max_epochs):
            self.current_epoch = epoch
            
            # Curriculum stage
            stage = self._get_curriculum_stage(epoch)
            if stage > 0 and hasattr(self.train_loader.dataset, 'curriculum_stage'):
                old_stage = self.train_loader.dataset.curriculum_stage
                if old_stage != stage:
                    print(f"\nCurriculum: Moving to stage {stage}")
                    self.train_loader.dataset.curriculum_stage = stage
            
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{self.config.max_epochs}")
            print(f"{'='*60}")
            
            # Train
            train_losses = self.train_epoch()
            
            # Validate
            if (epoch + 1) % self.config.eval_every == 0:
                val_metrics = self.validate()
                
                # Check for improvement
                is_best = False
                if val_metrics.get('task_accuracy', 0) > self.best_val_accuracy:
                    self.best_val_accuracy = val_metrics['task_accuracy']
                    is_best = True
                if val_metrics.get('val_loss', float('inf')) < self.best_val_loss:
                    self.best_val_loss = val_metrics['val_loss']
                    is_best = True
                
                # Save checkpoint
                if (epoch + 1) % self.config.save_every == 0 or is_best:
                    self.save_checkpoint(is_best=is_best)
        
        print("\nTraining complete!")
        print(f"Best validation accuracy: {self.best_val_accuracy:.4f}")
        print(f"Finished at: {datetime.now().isoformat()}")
        
        if self.wandb_run:
            wandb.finish()
        
        # Close file logger
        if self.tee_logger:
            print(f"\nLog saved to: {self.tee_logger.log_path}")
            self.tee_logger.close()


def train_sci_arc(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    loss_fn: Optional[nn.Module] = None,
    config: Optional[TrainingConfig] = None,
    resume_from: Optional[str] = None,
):
    """
    Main training function.
    
    Args:
        model: SCI-ARC model
        train_loader: Training DataLoader
        val_loader: Optional validation DataLoader
        loss_fn: Loss function module
        config: Training configuration
        resume_from: Path to checkpoint to resume from
    """
    if config is None:
        config = TrainingConfig()
    
    if loss_fn is None:
        from .losses import SCIARCLoss
        loss_fn = SCIARCLoss()
    
    trainer = SCIARCTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        config=config,
    )
    
    if resume_from:
        trainer.load_checkpoint(resume_from)
    
    trainer.train()
    
    return trainer
