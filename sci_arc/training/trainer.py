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
    
    # Batch size (reduced from 32/64 to prevent VRAM overflow)
    batch_size: int = 16
    eval_batch_size: int = 16
    
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
        
        # Move loss function to device (important for projection head in SCL)
        if hasattr(self.loss_fn, 'to'):
            self.loss_fn = self.loss_fn.to(self.device)
        
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
        
        # Accumulate losses as Python floats to avoid GPU memory buildup
        # Note: .item() does cause a sync, but it's necessary to prevent
        # GPU tensor accumulation which causes memory fragmentation and
        # eventually spills to slow shared GPU memory (system RAM over PCIe)
        epoch_losses = {
            'total': 0.0,
            'task': 0.0,
            'scl': 0.0,
            'ortho': 0.0,
            'deep': 0.0,
        }
        num_batches = 0
        
        # === DIAGNOSTICS: Track batch-level diversity for infinite data debugging ===
        batch_transform_families = []  # Track transform families to verify diversity
        
        warmup_steps = len(self.train_loader) * self.config.warmup_epochs
        epoch_start_time = time.time()
        batch_start_time = time.time()
        data_time = 0.0  # Track time waiting for data
        transfer_time = 0.0  # Track time for CPU->GPU transfer
        forward_time = 0.0  # Track forward pass time
        backward_time = 0.0  # Track backward pass time
        
        for batch_idx, batch in enumerate(self.train_loader):
            data_time = time.time() - batch_start_time  # Time spent waiting for this batch
            
            # Move batch to device
            transfer_start = time.time()
            batch = self._to_device(batch)
            transfer_time = time.time() - transfer_start
            
            # === DIAGNOSTIC: Track transform_family diversity (for infinite data debugging) ===
            if 'transform_families' in batch and batch_idx < 3:  # Only first 3 batches
                batch_transform_families.append(batch['transform_families'].tolist())
            
            # Warmup
            self._warmup_lr(self.global_step, warmup_steps)
            
            # Forward pass with mixed precision
            forward_start = time.time()
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
            forward_time = time.time() - forward_start
            
            total_loss = losses['total'] / self.config.grad_accumulation_steps
            
            # Backward pass
            backward_start = time.time()
            if self.scaler:
                self.scaler.scale(total_loss).backward()
            else:
                total_loss.backward()
            backward_time = time.time() - backward_start
            
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
            
            # Accumulate losses as Python floats (prevents GPU memory fragmentation)
            for key in epoch_losses:
                if key in losses:
                    epoch_losses[key] += losses[key].item()
            num_batches += 1
            self.global_step += 1
            
            # Logging with timing - sync CUDA only when we log (intentional)
            if batch_idx % self.config.log_every == 0:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()  # Ensure GPU work is complete for accurate timing
                batch_time = time.time() - batch_start_time
                self._log_step(batch_idx, losses, batch_time, data_time, transfer_time, 
                              forward_time, backward_time)
            batch_start_time = time.time()
        
        # Log epoch summary
        epoch_time = time.time() - epoch_start_time
        epochs_remaining = self.config.max_epochs - self.current_epoch - 1
        eta_seconds = epoch_time * epochs_remaining
        eta_str = f"{int(eta_seconds // 3600)}h {int((eta_seconds % 3600) // 60)}m" if eta_seconds > 3600 else f"{int(eta_seconds // 60)}m {int(eta_seconds % 60)}s"
        print(f"Epoch {self.current_epoch + 1} completed in {epoch_time:.1f}s "
              f"({epoch_time/num_batches:.2f}s/batch) | ETA: {eta_str}")
        
        # === DIAGNOSTIC: Log batch diversity (first epoch only for infinite data debugging) ===
        if self.current_epoch == 0 and batch_transform_families:
            print(f"\n[INFINITE DATA CHECK] First 3 batches transform_families:")
            for i, families in enumerate(batch_transform_families):
                unique_count = len(set(families))
                print(f"  Batch {i+1}: {unique_count} unique families out of {len(families)} samples")
                # We expect ~8 unique families (the 8 dihedral transforms)
                # Low diversity would be 1-2 families dominating
                if unique_count < 4:
                    print(f"    [!] Low diversity ({unique_count} families) - check augmentation settings")
                else:
                    print(f"    [+] Good diversity for SCL ({unique_count} transform families)")
        
        # Average losses (already Python floats)
        for key in epoch_losses:
            epoch_losses[key] /= max(num_batches, 1)
        
        return epoch_losses
    
    def _compute_losses(self, outputs: Dict, batch: Dict) -> Dict[str, torch.Tensor]:
        """Compute all losses.
        
        IMPORTANT: Uses self.loss_fn.deep_supervision for task/deep losses which includes:
        - Focal Loss (gamma > 0): Down-weights easy examples (background pixels)
        - Class Weights: Penalizes non-background errors more heavily  
        - Label Smoothing: Prevents overconfident predictions
        
        These are CRITICAL for preventing background collapse on ARC's ~85% background grids.
        """
        losses = {}
        
        # Main task loss (grid prediction)
        logits = outputs['logits']  # [B, H, W, num_colors]
        targets = batch['test_outputs']  # [B, H, W]
        B, H, W, C = logits.shape
        
        # === USE LOSS_FN.DEEP_SUPERVISION FOR FOCAL LOSS SUPPORT ===
        # This is the CRITICAL fix: use the deep_supervision component that has
        # Focal Loss, Class Weights, and Label Smoothing built-in
        if hasattr(self.loss_fn, 'deep_supervision'):
            # Compute task loss (final prediction only) using Focal Loss
            task_loss = self.loss_fn.deep_supervision([logits], targets)
            losses['task'] = task_loss
            
            # Compute deep supervision loss from intermediate predictions
            deep_loss = torch.tensor(0.0, device=self.device)
            if 'intermediate_logits' in outputs and outputs['intermediate_logits']:
                # Intermediate predictions loss (also with Focal Loss)
                deep_loss = self.loss_fn.deep_supervision(outputs['intermediate_logits'], targets)
            losses['deep'] = deep_loss
            
            # === DIAGNOSTIC: Log Focal Loss effect on first few batches ===
            if not hasattr(self, '_focal_debug_count'):
                self._focal_debug_count = 0
            if self._focal_debug_count < 3:
                ds = self.loss_fn.deep_supervision
                focal_gamma = getattr(ds, 'focal_gamma', 0.0)
                class_weights = getattr(ds, 'class_weights', None)
                
                # Compute per-class prediction distribution
                with torch.no_grad():
                    preds = logits.argmax(dim=-1)  # [B, H, W]
                    total_pixels = preds.numel()
                    
                    # Count background vs non-background predictions
                    bg_preds = (preds == 0).sum().item()
                    content_preds = total_pixels - bg_preds
                    bg_targets = (targets == 0).sum().item()
                    content_targets = total_pixels - bg_targets
                    
                    print(f"\n[FOCAL LOSS DEBUG - Batch {self._focal_debug_count + 1}]")
                    print(f"  Focal gamma: {focal_gamma:.1f}, Class weights: {'Active' if class_weights is not None else 'None'}")
                    print(f"  Target distribution: {bg_targets/total_pixels*100:.1f}% background, {content_targets/total_pixels*100:.1f}% content")
                    print(f"  Pred distribution:   {bg_preds/total_pixels*100:.1f}% background, {content_preds/total_pixels*100:.1f}% content")
                    print(f"  Task loss: {task_loss.item():.4f}")
                    
                    # Compare with what standard CE would give
                    logits_flat = logits.view(B * H * W, C)
                    targets_flat = targets.view(B * H * W)
                    standard_ce = F.cross_entropy(logits_flat, targets_flat)
                    print(f"  Standard CE (for comparison): {standard_ce.item():.4f}")
                    
                    if focal_gamma > 0 and task_loss.item() < standard_ce.item():
                        print(f"  [+] Focal Loss is DOWN-WEIGHTING easy samples (loss reduced by {standard_ce.item() - task_loss.item():.4f})")
                    elif focal_gamma > 0:
                        print(f"  [+] Focal Loss active (may increase loss on hard samples)")
                
                self._focal_debug_count += 1
        else:
            # Fallback to standard CE (without Focal Loss) if loss_fn doesn't have deep_supervision
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
            
            # Debug: Log transform_family distribution (first few batches only)
            if not hasattr(self, '_scl_debug_count'):
                self._scl_debug_count = 0
            if self._scl_debug_count < 3:
                unique, counts = torch.unique(transform_families, return_counts=True)
                print(f"\n[SCL DEBUG] Transform families in batch:")
                print(f"  Unique values: {unique.tolist()}")
                print(f"  Counts: {counts.tolist()}")
                print(f"  z_struct shape: {z_struct.shape}")
                print(f"  z_struct mean: {z_struct.mean().item():.6f}, std: {z_struct.std().item():.6f}")
                
                # Check if z_struct varies across samples (using FLATTENING, not pooling)
                B = z_struct.size(0)
                z_flat = z_struct.reshape(B, -1)  # [B, K*D] - FLATTEN, not pool!
                z_norm = torch.nn.functional.normalize(z_flat, dim=-1)
                
                # Check similarity of first few samples (BEFORE projection, BEFORE batchnorm)
                sim_01 = (z_norm[0] * z_norm[1]).sum().item()
                sim_02 = (z_norm[0] * z_norm[2]).sum().item()
                sim_12 = (z_norm[1] * z_norm[2]).sum().item()
                print(f"  Pre-BatchNorm similarities: (0,1)={sim_01:.4f}, (0,2)={sim_02:.4f}, (1,2)={sim_12:.4f}")
                
                # Check AFTER BatchNorm (if available)
                if hasattr(self.loss_fn, 'scl') and hasattr(self.loss_fn.scl, 'batch_norm'):
                    with torch.no_grad():
                        z_bn = self.loss_fn.scl.batch_norm(z_flat)  # Apply batch norm
                        z_bn_norm = torch.nn.functional.normalize(z_bn, dim=-1)
                        sim_01_bn = (z_bn_norm[0] * z_bn_norm[1]).sum().item()
                        sim_02_bn = (z_bn_norm[0] * z_bn_norm[2]).sum().item()
                        sim_12_bn = (z_bn_norm[1] * z_bn_norm[2]).sum().item()
                        print(f"  Post-BatchNorm similarities: (0,1)={sim_01_bn:.4f}, (0,2)={sim_02_bn:.4f}, (1,2)={sim_12_bn:.4f}")
                
                # Check AFTER projection (if available)
                if hasattr(self.loss_fn, 'scl') and hasattr(self.loss_fn.scl, 'projector'):
                    with torch.no_grad():
                        z_bn = self.loss_fn.scl.batch_norm(z_flat)  # Apply batch norm first
                        z_proj = self.loss_fn.scl.projector(z_bn)  # Then project
                        z_proj_norm = torch.nn.functional.normalize(z_proj, dim=-1)
                        sim_01_p = (z_proj_norm[0] * z_proj_norm[1]).sum().item()
                        sim_02_p = (z_proj_norm[0] * z_proj_norm[2]).sum().item()
                        sim_12_p = (z_proj_norm[1] * z_proj_norm[2]).sum().item()
                        print(f"  Post-projection similarities: (0,1)={sim_01_p:.4f}, (0,2)={sim_02_p:.4f}, (1,2)={sim_12_p:.4f}")
                
                self._scl_debug_count += 1
            
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
    
    def _log_step(self, batch_idx: int, losses: Dict, batch_time: float = 0.0, 
                  data_time: float = 0.0, transfer_time: float = 0.0,
                  forward_time: float = 0.0, backward_time: float = 0.0):
        """Log training step with timing breakdown."""
        lr = self.optimizer.param_groups[0]['lr']
        other_time = batch_time - data_time - transfer_time - forward_time - backward_time
        
        # Display epoch as 1-indexed to match header (Epoch 1/100)
        log_str = f"Epoch {self.current_epoch + 1} [{batch_idx + 1}/{len(self.train_loader)}] "
        log_str += f"Loss: {losses['total'].item():.4f} "
        log_str += f"(task={losses['task'].item():.4f}, "
        log_str += f"scl={losses['scl'].item():.4f}, "
        log_str += f"ortho={losses['ortho'].item():.4f}) "
        log_str += f"LR: {lr:.2e} "
        
        # Show timing breakdown when batch is slow (>5s)
        if batch_time > 5.0:
            log_str += f"[{batch_time:.1f}s: fwd={forward_time:.1f}s bwd={backward_time:.1f}s data={data_time:.1f}s xfer={transfer_time:.1f}s other={other_time:.1f}s]"
            # Add GPU info for slow batches - use memory_reserved for accurate total
            # memory_allocated = tensors only, memory_reserved = all CUDA allocations
            if torch.cuda.is_available():
                mem_allocated = torch.cuda.memory_allocated() / 1024**3
                mem_reserved = torch.cuda.memory_reserved() / 1024**3
                mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                log_str += f" GPU:{mem_reserved:.1f}/{mem_total:.1f}GB (alloc:{mem_allocated:.1f}GB)"
        else:
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
        num_val_batches = len(self.val_loader)
        val_start_time = time.time()
        
        for batch_idx, batch in enumerate(self.val_loader):
            batch_start = time.time()
            batch = self._to_device(batch)
            
            # Use forward_training which accepts the batched format
            outputs = self.model.forward_training(
                input_grids=batch['input_grids'],
                output_grids=batch['output_grids'],
                test_input=batch['test_inputs'],
                test_output=batch['test_outputs'],
                grid_mask=batch.get('grid_masks'),
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
            
            # Task accuracy (entire grid must match) - VECTORIZED
            # Flatten spatial dims and check if all match per sample
            B = preds.shape[0]
            preds_flat = preds.view(B, -1)  # [B, H*W]
            targets_flat = targets.view(B, -1)  # [B, H*W]
            task_correct = (preds_flat == targets_flat).all(dim=1)  # [B] bool
            total_tasks_correct += task_correct.sum().item()
            total_tasks += B
            
            batch_time = time.time() - batch_start
            print(f"  Validation batch {batch_idx + 1}/{num_val_batches} [{batch_time:.1f}s]", end='\r')
        
        val_time = time.time() - val_start_time
        
        metrics = {
            'val_loss': total_loss / len(self.val_loader),
            'pixel_accuracy': total_correct / total_pixels,
            'task_accuracy': total_tasks_correct / total_tasks,
        }
        
        print(f"\nValidation complete in {val_time:.1f}s: Loss={metrics['val_loss']:.4f}, "
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
        
        # Save epoch checkpoint
        path = self.checkpoint_dir / f'checkpoint_epoch_{self.current_epoch}.pt'
        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path.name}")
        
        # Always save a "latest" checkpoint for easy resume
        latest_path = self.checkpoint_dir / 'checkpoint_latest.pt'
        torch.save(checkpoint, latest_path)
        
        # Save best
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"Saved best model to {best_path}")
        
        # Clean old checkpoints (but keep latest and best)
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
    
    def _log_loss_config(self):
        """Log loss function configuration to verify Focal Loss, Class Weights, etc are active.
        
        This is CRITICAL for debugging background collapse issues.
        If Focal Loss/Class Weights are not active, the model will collapse to predicting background.
        """
        print(f"\n{'='*60}")
        print("Loss Function Configuration:")
        print(f"{'='*60}")
        
        if not hasattr(self.loss_fn, 'deep_supervision'):
            print("  [!] WARNING: loss_fn has no deep_supervision component!")
            print("  [!] Focal Loss, Class Weights, Label Smoothing will NOT be used!")
            print(f"  Loss function type: {type(self.loss_fn).__name__}")
            return
        
        ds = self.loss_fn.deep_supervision
        
        # Focal Loss
        focal_gamma = getattr(ds, 'focal_gamma', 0.0)
        if focal_gamma > 0:
            print(f"  [+] FOCAL LOSS: gamma={focal_gamma:.1f} (DOWN-WEIGHTS EASY/BACKGROUND PIXELS)")
            print(f"    Formula: FL(pt) = -(1-pt)^{focal_gamma:.1f} * log(pt)")
            print(f"    Effect: Easy samples (background) get ~{(1-0.85)**focal_gamma:.3f}x weight")
            print(f"            Hard samples (content) get ~{(1-0.15)**focal_gamma:.3f}x weight")
        else:
            print(f"  [-] Focal Loss: DISABLED (gamma=0)")
            print(f"    [!] Model may collapse to predicting mostly background!")
        
        # Class Weights
        class_weights = getattr(ds, 'class_weights', None)
        if class_weights is not None:
            print(f"\n  [+] CLASS WEIGHTS: ACTIVE")
            bg_weight = class_weights[0].item() if class_weights.numel() > 0 else 1.0
            other_weight = class_weights[1].item() if class_weights.numel() > 1 else 1.0
            print(f"    Background (0): {bg_weight:.2f}")
            print(f"    Other colors:   {other_weight:.2f}")
            print(f"    Effect: Non-background errors penalized {other_weight/bg_weight:.0f}x more")
        else:
            print(f"\n  [-] Class Weights: DISABLED")
            print(f"    [!] All colors weighted equally (background dominates)")
        
        # Label Smoothing
        label_smoothing = getattr(ds, 'label_smoothing', 0.0)
        if label_smoothing > 0:
            print(f"\n  [+] LABEL SMOOTHING: {label_smoothing:.2f}")
            print(f"    Effect: Prevents overconfident predictions")
        else:
            print(f"\n  [-] Label Smoothing: DISABLED")
        
        # Deep Supervision
        num_steps = getattr(ds, 'num_steps', 1)
        weight_schedule = getattr(ds, 'weight_schedule', 'unknown')
        print(f"\n  Deep Supervision:")
        print(f"    Refinement steps: {num_steps}")
        print(f"    Weight schedule: {weight_schedule}")
        
        # Summary
        print(f"\n  === ANTI-BACKGROUND-COLLAPSE STATUS ===")
        active_defenses = []
        if focal_gamma > 0:
            active_defenses.append(f"Focal(gamma={focal_gamma})")
        if class_weights is not None:
            active_defenses.append("ClassWeights")
        if label_smoothing > 0:
            active_defenses.append(f"LabelSmooth({label_smoothing})")
        
        if active_defenses:
            print(f"  [+] Active: {', '.join(active_defenses)}")
            print(f"  Model should focus on content pixels, not just background.")
        else:
            print(f"  [!] NO DEFENSES ACTIVE!")
            print(f"  [!] Model will likely collapse to predicting all background!")
            print(f"  [!] Enable focal_gamma, use_class_weights in config!")
        
        print(f"{'='*60}")
    
    def _log_dataset_info(self):
        """Log dataset configuration for debugging (especially for infinite data mode)."""
        dataset = self.train_loader.dataset
        
        print(f"\n{'='*60}")
        print("Dataset Configuration:")
        print(f"{'='*60}")
        
        # Basic info
        print(f"  Dataset size: {len(dataset)} samples")
        print(f"  Batch size: {self.train_loader.batch_size}")
        print(f"  Batches per epoch: {len(self.train_loader)}")
        
        # Cache mode (CRITICAL for understanding training behavior)
        if hasattr(dataset, 'cache_samples'):
            cache_mode = dataset.cache_samples
            print(f"\n  === CACHING MODE ===")
            if cache_mode:
                print(f"  Mode: CACHED (fixed {len(dataset)} samples)")
                print(f"  Behavior: Same samples every epoch (potential overfitting)")
                if hasattr(dataset, 'cache_augmentations'):
                    print(f"  Cached augmentations: {dataset.cache_augmentations} per task")
            else:
                print(f"  Mode: INFINITE (on-the-fly generation)")
                print(f"  Behavior: Every batch is UNIQUE (maximum generalization)")
                print(f"  Samples per epoch: {len(self.train_loader) * self.train_loader.batch_size}")
                total_unique = len(self.train_loader) * self.train_loader.batch_size * self.config.max_epochs
                print(f"  Total unique samples ({self.config.max_epochs} epochs): ~{total_unique:,}")
        
        # Augmentation info
        if hasattr(dataset, 'augment'):
            print(f"\n  === AUGMENTATION ===")
            print(f"  Enabled: {dataset.augment}")
            if hasattr(dataset, 'use_augment_family'):
                print(f"  Use augment as transform_family (for SCL): {dataset.use_augment_family}")
        
        # SCL info
        if hasattr(self.loss_fn, 'scl'):
            print(f"\n  === SCL CONFIGURATION ===")
            print(f"  SCL weight: {self.config.scl_weight}")
            if hasattr(self.loss_fn.scl, 'temperature'):
                temp = self.loss_fn.scl.temperature
                if hasattr(temp, 'item'):
                    print(f"  Temperature: {temp.item():.4f} (learnable)")
                else:
                    print(f"  Temperature: {temp}")
            if hasattr(self.loss_fn.scl, 'batch_norm'):
                print(f"  BatchNorm: Enabled (removes background signal)")
        
        print(f"{'='*60}")
    
    def _log_epoch_summary(self, epoch: int, train_losses: Dict[str, float], 
                           val_metrics: Optional[Dict[str, float]] = None):
        """Log detailed epoch summary with SCL health checks."""
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch + 1} SUMMARY")
        print(f"{'='*60}")
        
        # Training losses
        print(f"Training Losses:")
        print(f"  Total: {train_losses['total']:.6f}")
        print(f"  Task:  {train_losses['task']:.6f}")
        print(f"  SCL:   {train_losses['scl']:.6f}")
        print(f"  Ortho: {train_losses['ortho']:.6f}")
        print(f"  Deep:  {train_losses['deep']:.6f}")
        
        # SCL health check
        scl_loss = train_losses['scl']
        log_batch_size = math.log(self.config.batch_size)
        print(f"\n  === SCL HEALTH CHECK ===")
        if scl_loss < 0.1:
            print(f"  [+] SCL loss very low ({scl_loss:.4f}) - Excellent clustering!")
        elif scl_loss < 1.0:
            print(f"  [+] SCL loss moderate ({scl_loss:.4f}) - Good progress")
        elif scl_loss < 3.0:
            print(f"  [!] SCL loss high ({scl_loss:.4f}) - Still learning")
        elif scl_loss > log_batch_size - 0.5:
            # Near random chance - expected in early epochs
            epoch = self.current_epoch + 1
            if epoch <= 5:
                print(f"  [~] SCL loss ~ random ({scl_loss:.2f} vs log(B)={log_batch_size:.2f})")
                print(f"    Expected at epoch {epoch} - contrastive learning takes time")
            elif epoch <= 20:
                print(f"  [!] SCL loss still high ({scl_loss:.2f}) at epoch {epoch}")
                print(f"    Monitor: Should decrease to <4.0 by epoch 20")
            else:
                print(f"  [-] SCL loss ~ log(batch_size)={log_batch_size:.2f}")
                print(f"    WARNING: Possible representation collapse at epoch {epoch}!")
                print(f"    Check: Are z_struct embeddings diverse?")
        else:
            print(f"  [?] SCL loss: {scl_loss:.4f}")
        
        # Temperature tracking (if learnable)
        if hasattr(self.loss_fn, 'scl') and hasattr(self.loss_fn.scl, 'temperature'):
            temp = self.loss_fn.scl.temperature
            if hasattr(temp, 'item'):
                print(f"  Current temperature: {temp.item():.4f}")
        
        # Validation metrics
        if val_metrics:
            print(f"\nValidation Metrics:")
            print(f"  Loss:     {val_metrics.get('val_loss', 0):.6f}")
            print(f"  Pixel Acc: {val_metrics.get('pixel_accuracy', 0)*100:.2f}%")
            print(f"  Task Acc:  {val_metrics.get('task_accuracy', 0)*100:.2f}%")
            
            # Best tracker
            print(f"\nBest So Far:")
            print(f"  Best Task Acc: {self.best_val_accuracy*100:.2f}%")
            print(f"  Best Val Loss: {self.best_val_loss:.6f}")
        
        # GPU memory
        if torch.cuda.is_available():
            mem_alloc = torch.cuda.memory_allocated() / 1024**3
            mem_reserved = torch.cuda.memory_reserved() / 1024**3
            mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"\nGPU Memory: {mem_reserved:.1f}GB reserved / {mem_total:.1f}GB total")
        
        print(f"{'='*60}")
    
    def load_checkpoint(self, path: str):
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if checkpoint['scaler_state_dict'] and self.scaler:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.current_epoch = checkpoint['epoch'] + 1  # Resume from NEXT epoch
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_val_accuracy = checkpoint['best_val_accuracy']
        
        print(f"Loaded checkpoint from epoch {checkpoint['epoch'] + 1}, will resume from epoch {self.current_epoch + 1}")
    
    def train(self):
        """Full training loop."""
        print(f"\nStarting training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"\n{'='*60}")
        print("Training Configuration:")
        print(f"{'='*60}")
        for key, value in self.config.__dict__.items():
            print(f"  {key}: {value}")
        print(f"{'='*60}")
        
        # === LOG LOSS FUNCTION CONFIGURATION (CRITICAL FOR DEBUGGING) ===
        self._log_loss_config()
        
        # === DIAGNOSTIC: Dataset info for debugging infinite data ===
        self._log_dataset_info()
        print()
        
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
            sys.stdout.flush()  # Ensure header is visible
            
            # Train
            train_losses = self.train_epoch()
            sys.stdout.flush()  # Flush after training
            
            # Clear CUDA cache to defragment memory before validation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Validate
            val_metrics = None
            if (epoch + 1) % self.config.eval_every == 0:
                print("Starting validation...")
                sys.stdout.flush()
                val_metrics = self.validate()
                sys.stdout.flush()  # Flush after validation
                
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
            
            # Log detailed epoch summary
            self._log_epoch_summary(epoch, train_losses, val_metrics)
        
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
