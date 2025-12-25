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

from .loss_logger import LossLogger
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
    Logger that writes to both stdout and a file (encoding-safe for Windows).
    Captures all print() output for debugging and reproducibility.
    """
    def __init__(self, log_path: Path):
        self.terminal = sys.stdout
        self.log_path = log_path
        # Use UTF-8 with error handling for Windows compatibility
        self.log_file = open(log_path, 'w', encoding='utf-8', errors='replace', buffering=1)
        
    def write(self, message):
        # Handle potential encoding issues on Windows terminal
        try:
            self.terminal.write(message)
        except UnicodeEncodeError:
            # Fallback to ASCII-safe version for Windows cmd
            self.terminal.write(message.encode('ascii', errors='replace').decode('ascii'))
        self.log_file.write(message)
        self.log_file.flush()
        
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
    
    # Loss weights (legacy SCL)
    scl_weight: float = 0.1
    ortho_weight: float = 0.01
    deep_supervision_weight: float = 0.5
    
    # CISL: Content-Invariant Structure Learning (replaces SCL for few-shot)
    # Note: Config params use 'cicl_' prefix for backward compatibility
    use_cicl: bool = False                   # Enable CISL instead of SCL (set True in config)
    cicl_consist_weight: float = 0.5         # Within-task consistency weight (λ₁)
    cicl_color_inv_weight: float = 0.5       # Content invariance weight (λ₂)
    cicl_variance_weight: float = 0.1        # Batch variance weight (λ₃, anti-collapse)
    cicl_target_std: float = 0.5             # Target std (γ) for variance regularization
    
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
    
    # Validation settings
    # TTA (Test-Time Augmentation) for validation - matches inference behavior
    val_use_tta: bool = True              # Use TTA during validation (recommended)
    val_num_dihedral: int = 8             # Dihedral transforms (1-8)
    val_num_color_perms: int = 1          # Color perms per dihedral (1=fast, 4=thorough)
    
    # Curriculum learning
    use_curriculum: bool = True
    curriculum_stages: List[int] = field(default_factory=lambda: [10, 30, 60])  # Epoch thresholds
    
    # Device
    device: str = 'cuda'
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = False  # Enable CUDA deterministic mode for reproducibility (slower)
    
    # Numerical stability
    check_nan_inf: bool = True  # Check for NaN/Inf in losses during training


class SCIARCTrainer:
    """
    Trainer for SCI-ARC model.
    
    Implements full training loop with:
    - Curriculum learning
    - Deep supervision
    - Mixed precision
    - Logging and checkpointing
    - CISL: Content-Invariant Structure Learning
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
        
        # Set deterministic mode for reproducibility if requested
        if config.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True, warn_only=True)
            print("[Trainer] Deterministic mode enabled (slower but reproducible)")
        
        # Move loss function to device (important for projection head in SCL)
        if hasattr(self.loss_fn, 'to'):
            self.loss_fn = self.loss_fn.to(self.device)
        
        # Initialize CISL loss if enabled (Content-Invariant Structure Learning)
        # Note: CICLLoss is an alias for CISLLoss for backward compatibility
        self.cisl_loss = None
        if config.use_cicl:
            from .cisl_loss import CISLLoss
            self.cisl_loss = CISLLoss(
                consist_weight=config.cicl_consist_weight,
                content_inv_weight=config.cicl_color_inv_weight,  # Renamed from color_inv
                variance_weight=config.cicl_variance_weight,
                target_std=config.cicl_target_std,
                debug=True  # Enable per-batch statistics
            ).to(self.device)
            print(f"[CISL] Enabled Content-Invariant Structure Learning:")
            print(f"  - Consistency weight (L1): {config.cicl_consist_weight}")
            print(f"  - Content invariance weight (L2): {config.cicl_color_inv_weight}")
            print(f"  - Variance weight (L3): {config.cicl_variance_weight}")
            print(f"  - Target std (gamma): {config.cicl_target_std}")
        
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
        
        # Initialize structured loss logger for per-component tracking
        self.loss_logger = LossLogger(
            log_dir=str(self.checkpoint_dir / 'loss_logs'),
            log_interval=config.log_every,
            enable_file_logging=config.log_to_file,
            enable_console=False,  # We have custom console logging
            wandb_run=None,  # Will be set after wandb init
            verbose=False,
        )
        
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
            # Connect loss logger to wandb
            self.loss_logger.wandb_run = self.wandb_run
    
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
        
        # Reset CISL debug counter and epoch stats at start of each epoch
        self._cisl_debug_count = 0
        if self.cisl_loss is not None:
            self.cisl_loss.reset_epoch_stats()
        
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
            
            # Check for NaN/Inf in losses (numerical stability guard)
            if self.config.check_nan_inf:
                if not torch.isfinite(total_loss):
                    print(f"\n[!] WARNING: Non-finite loss detected at step {self.global_step}!")
                    print(f"    total_loss = {total_loss.item()}")
                    for k, v in losses.items():
                        if torch.is_tensor(v) and not torch.isfinite(v):
                            print(f"    {k} = {v.item()} (NON-FINITE)")
                    print("    Skipping this batch to prevent training corruption.")
                    self.optimizer.zero_grad()
                    continue
            
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
                
                # === GRADIENT HEALTH METRICS (before clipping) ===
                # Compute gradient statistics for debugging
                grad_metrics = self._compute_gradient_metrics()
                losses.update(grad_metrics)
                
                if self.scaler:
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
        
        # === CISL Epoch Summary ===
        if self.config.use_cicl and self.cisl_loss is not None:
            cisl_epoch_stats = self.cisl_loss.get_epoch_stats()
            print(f"\n[CISL EPOCH SUMMARY] Epoch {self.current_epoch + 1}:")
            print(f"  L_consist (avg): {cisl_epoch_stats['cisl/consist_avg']:.6f}")
            print(f"  L_content_inv (avg): {cisl_epoch_stats['cisl/content_inv_avg']:.6f}")
            print(f"  L_variance (avg): {cisl_epoch_stats['cisl/variance_avg']:.6f}")
            print(f"  CISL total (avg): {cisl_epoch_stats['cisl/total_avg']:.6f}")
            print(f"  Embedding mean (avg): {cisl_epoch_stats['cisl/z_mean_avg']:.6f}")
            print(f"  Embedding std (avg): {cisl_epoch_stats['cisl/z_std_avg']:.6f}")
            print(f"  Embedding norm (avg): {cisl_epoch_stats['cisl/z_norm_avg']:.4f}")
            print(f"  Batches processed: {cisl_epoch_stats['cisl/batches_processed']}")
            
            # Log to wandb
            if self.wandb_run:
                wandb.log({
                    'epoch/cisl_consist_avg': cisl_epoch_stats['cisl/consist_avg'],
                    'epoch/cisl_content_inv_avg': cisl_epoch_stats['cisl/content_inv_avg'],
                    'epoch/cisl_variance_avg': cisl_epoch_stats['cisl/variance_avg'],
                    'epoch/cisl_total_avg': cisl_epoch_stats['cisl/total_avg'],
                    'epoch/cisl_z_mean_avg': cisl_epoch_stats['cisl/z_mean_avg'],
                    'epoch/cisl_z_std_avg': cisl_epoch_stats['cisl/z_std_avg'],
                    'epoch/cisl_z_norm_avg': cisl_epoch_stats['cisl/z_norm_avg'],
                    'epoch': self.current_epoch + 1,
                })
        
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
        
        # === STRUCTURED EPOCH SUMMARY ===
        # Log epoch-level loss statistics and health checks
        epoch_summary = self.loss_logger.log_epoch_summary(self.current_epoch)
        
        # Check for health issues and warn
        health_warnings = self.loss_logger.check_health()
        if health_warnings:
            print("\n[!] LOSS HEALTH WARNINGS:")
            for warning in health_warnings:
                print(f"    {warning}")
        
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
                    # Create valid mask (exclude padding: -100)
                    valid_mask = (targets != -100)
                    total_pixels = valid_mask.sum().item()
                    
                    if total_pixels > 0:
                        # Count background vs non-background predictions (only valid pixels)
                        bg_preds = ((preds == 0) & valid_mask).sum().item()
                        content_preds = total_pixels - bg_preds
                        bg_targets = ((targets == 0) & valid_mask).sum().item()
                        content_targets = total_pixels - bg_targets
                        
                        print(f"\n[FOCAL LOSS DEBUG - Batch {self._focal_debug_count + 1}]")
                        print(f"  Focal gamma: {focal_gamma:.1f}, Class weights: {'Active' if class_weights is not None else 'None'}")
                        print(f"  Valid pixels: {total_pixels} (excluding -100 padding)")
                        print(f"  Target distribution: {bg_targets/total_pixels*100:.1f}% background, {content_targets/total_pixels*100:.1f}% content")
                        print(f"  Pred distribution:   {bg_preds/total_pixels*100:.1f}% background, {content_preds/total_pixels*100:.1f}% content")
                        print(f"  Task loss: {task_loss.item():.4f}")
                        
                        # Compare with what standard CE would give (with ignore_index)
                        logits_flat = logits.view(B * H * W, C)
                        targets_flat = targets.view(B * H * W)
                        standard_ce = F.cross_entropy(logits_flat, targets_flat, ignore_index=-100)
                        print(f"  Standard CE (for comparison): {standard_ce.item():.4f}")
                        
                        if focal_gamma > 0 and task_loss.item() < standard_ce.item():
                            print(f"  [+] Focal Loss is DOWN-WEIGHTING easy samples (loss reduced by {standard_ce.item() - task_loss.item():.4f})")
                        elif focal_gamma > 0:
                            print(f"  [+] Focal Loss active (may increase loss on hard samples)")
                
                self._focal_debug_count += 1
        else:
            # Fallback to standard CE (without Focal Loss) if loss_fn doesn't have deep_supervision
            # CRITICAL: Use ignore_index=-100 to handle padded targets
            logits_flat = logits.view(B * H * W, C)
            targets_flat = targets.view(B * H * W)
            
            task_loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=-100)
            losses['task'] = task_loss
            
            # Deep supervision loss (intermediate predictions)
            deep_loss = torch.tensor(0.0, device=self.device)
            if 'intermediate_logits' in outputs and outputs['intermediate_logits']:
                for inter_logits in outputs['intermediate_logits']:
                    inter_flat = inter_logits.view(B * H * W, C)
                    deep_loss += F.cross_entropy(inter_flat, targets_flat, ignore_index=-100)
                deep_loss /= len(outputs['intermediate_logits'])
            losses['deep'] = deep_loss
        
        # SCL loss - use the loss_fn's scl component if available
        # OR use CISL (Content-Invariant Structure Learning) if enabled
        scl_loss = torch.tensor(0.0, device=self.device)
        cisl_losses = {'consistency': torch.tensor(0.0, device=self.device),
                       'content_inv': torch.tensor(0.0, device=self.device),
                       'variance': torch.tensor(0.0, device=self.device)}
        
        if 'z_struct' in outputs:
            z_struct = outputs['z_struct']  # [B, K, D] - aggregated
            z_struct_demos = outputs.get('z_struct_demos')  # [B, P, K, D] - per-demo (for consistency)
            
            if self.config.use_cicl and self.cisl_loss is not None:
                # === CISL: Content-Invariant Structure Learning ===
                # This replaces SCL with a more stable, few-shot-appropriate approach
                
                if not hasattr(self, '_cisl_debug_count'):
                    self._cisl_debug_count = 0
                
                # === CONTENT INVARIANCE: Compute z_struct for color-permuted inputs ===
                # This is the key CISL idea: S(task) = S(permute_colors(task))
                # We apply random color permutation and verify structure is unchanged
                z_struct_content_aug = None
                if self.config.cicl_color_inv_weight > 0:
                    from .cisl_loss import apply_content_permutation_batch
                    
                    # Apply color permutation to all grids
                    input_grids_perm, output_grids_perm, test_input_perm, test_output_perm = \
                        apply_content_permutation_batch(
                            batch['input_grids'],
                            batch['output_grids'],
                            batch['test_inputs'],
                            batch['test_outputs']
                        )
                    
                    # Use efficient structure-only encoding (skips refinement)
                    # This saves ~50% compute vs full forward_training
                    with torch.no_grad():
                        z_struct_content_aug = self.model.encode_structure_only(
                            input_grids=input_grids_perm,
                            output_grids=output_grids_perm,
                        ).detach()
                
                # Compute CISL losses (includes stats tracking)
                # Pass z_struct_demos [B, P, K, D] for proper consistency across demos
                cisl_result = self.cisl_loss(
                    z_struct=z_struct,                      # [B, K, D] - for variance/content_inv
                    z_struct_demos=z_struct_demos,          # [B, P, K, D] - for consistency
                    z_struct_content_aug=z_struct_content_aug,
                )
                
                cisl_losses = {
                    'consistency': cisl_result['consistency'],
                    'content_inv': cisl_result['content_inv'],
                    'variance': cisl_result['variance']
                }
                
                # Use CISL total as the "scl_loss" for backward compatibility
                scl_loss = cisl_result['total']
                
                # Get embedding statistics
                cisl_stats = cisl_result.get('stats', {})
                
                # Per-batch debug logging (first 5 batches per epoch)
                if self._cisl_debug_count < 5:
                    print(f"\n[CISL DEBUG] Content-Invariant Structure Learning - Batch {self._cisl_debug_count + 1}:")
                    print(f"  z_struct shape: {z_struct.shape}")
                    print(f"  Embedding stats: mean={cisl_stats.get('z_mean', 0):.6f}, std={cisl_stats.get('z_std', 0):.6f}, norm={cisl_stats.get('z_norm', 0):.4f}")
                    print(f"  L_consist: {cisl_result['consistency'].item():.6f} (weight: {self.config.cicl_consist_weight})")
                    print(f"  L_content_inv: {cisl_result['content_inv'].item():.6f} (weight: {self.config.cicl_color_inv_weight})")
                    print(f"  L_variance: {cisl_result['variance'].item():.6f} (weight: {self.config.cicl_variance_weight})")
                    print(f"  CISL total (weighted): {cisl_result['total'].item():.6f}")
                    if z_struct_content_aug is not None:
                        print(f"  Content-augmented z_struct: {z_struct_content_aug.shape}")
                        print(f"  Orig-Aug cosine similarity: {cisl_stats.get('orig_aug_cos_sim', 0):.4f}")
                    else:
                        print(f"  [!] No content-augmented z_struct - L_content_inv = 0")
                    self._cisl_debug_count += 1
                    
            elif 'transform_families' in batch:
                # === Legacy SCL ===
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
        # Add CISL sub-losses for logging (use cisl_ prefix for consistency)
        losses['cisl_consist'] = cisl_losses['consistency']
        losses['cisl_content_inv'] = cisl_losses['content_inv']
        losses['cisl_variance'] = cisl_losses['variance']
        
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
        
        # ===============================================
        # COMPREHENSIVE HEALTH METRICS FOR DEBUGGING
        # ===============================================
        # These metrics enable remote debugging by logging:
        # - Logit health (mean, std, min, max for numerical stability)
        # - Prediction quality (accuracy, per-class accuracy)
        # - Confidence metrics (entropy, prediction confidence)
        
        with torch.no_grad():
            # === LOGIT HEALTH METRICS ===
            # Monitor for numerical instability (overflow/underflow)
            logits = outputs['logits']  # [B, H, W, C]
            losses['logit_mean'] = logits.mean().item()
            losses['logit_std'] = logits.std().item()
            losses['logit_max'] = logits.max().item()
            losses['logit_min'] = logits.min().item()
            
            # === PREDICTION QUALITY METRICS ===
            targets = batch['test_outputs']  # [B, H, W]
            valid_mask = (targets != -100)  # Exclude padding
            
            if valid_mask.any():
                preds = logits.argmax(dim=-1)  # [B, H, W]
                
                # Overall accuracy (on valid pixels only)
                correct = ((preds == targets) & valid_mask).sum().item()
                total_pixels = valid_mask.sum().item()
                losses['accuracy'] = correct / max(total_pixels, 1)
                
                # Background accuracy (class 0)
                bg_mask = (targets == 0) & valid_mask
                if bg_mask.any():
                    bg_correct = ((preds == 0) & bg_mask).sum().item()
                    losses['bg_accuracy'] = bg_correct / max(bg_mask.sum().item(), 1)
                else:
                    losses['bg_accuracy'] = 0.0
                
                # Foreground accuracy (classes 1-10)
                fg_mask = (targets > 0) & (targets <= 10) & valid_mask
                if fg_mask.any():
                    fg_correct = ((preds == targets) & fg_mask).sum().item()
                    losses['fg_accuracy'] = fg_correct / max(fg_mask.sum().item(), 1)
                else:
                    losses['fg_accuracy'] = 0.0
            else:
                losses['accuracy'] = 0.0
                losses['bg_accuracy'] = 0.0
                losses['fg_accuracy'] = 0.0
            
            # === CONFIDENCE METRICS ===
            # Low confidence may indicate uncertain predictions
            probs = torch.softmax(logits, dim=-1)  # [B, H, W, C]
            max_probs, _ = probs.max(dim=-1)  # [B, H, W]
            if valid_mask.any():
                losses['confidence_mean'] = max_probs[valid_mask].mean().item()
            else:
                losses['confidence_mean'] = 0.0
            
            # Prediction entropy (high = uncertain, low = confident)
            # Only on valid pixels
            if valid_mask.any():
                probs_valid = probs[valid_mask]  # [N_valid, C]
                # Avoid log(0) with small epsilon
                entropy = -(probs_valid * (probs_valid + 1e-10).log()).sum(dim=-1).mean()
                losses['pred_entropy'] = entropy.item()
            else:
                losses['pred_entropy'] = 0.0
        
        return losses
    
    def _compute_gradient_metrics(self) -> Dict[str, float]:
        """
        Compute gradient statistics for debugging.
        
        Should be called BEFORE gradient clipping to get true gradient magnitudes.
        Returns metrics useful for detecting:
        - Exploding gradients (large grad_norm, grad_max)
        - Vanishing gradients (very small grad_norm)
        - NaN/Inf gradients (grad_has_nan = True)
        """
        total_norm = 0.0
        grad_max = 0.0
        grad_min = float('inf')
        grad_count = 0
        has_nan = False
        has_inf = False
        
        for param in self.model.parameters():
            if param.grad is not None:
                grad_data = param.grad.data
                
                # Check for NaN/Inf
                if torch.isnan(grad_data).any():
                    has_nan = True
                if torch.isinf(grad_data).any():
                    has_inf = True
                
                # Compute norms (on valid gradients)
                param_norm = grad_data.norm(2).item()
                total_norm += param_norm ** 2
                
                # Track min/max
                if grad_data.numel() > 0:
                    grad_max = max(grad_max, grad_data.abs().max().item())
                    grad_min = min(grad_min, grad_data.abs().min().item())
                    grad_count += 1
        
        total_norm = total_norm ** 0.5
        
        # Handle case where no gradients exist
        if grad_count == 0:
            grad_min = 0.0
        
        return {
            'grad_norm': total_norm,
            'grad_max': grad_max,
            'grad_min': grad_min,
            'grad_has_nan': 1.0 if has_nan else 0.0,
            'grad_has_inf': 1.0 if has_inf else 0.0,
        }
    
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
        
        # === STRUCTURED LOSS LOGGING ===
        # Log all loss components to the structured logger for tracking and analysis
        structured_losses = {
            'total': losses['total'].item() if isinstance(losses['total'], torch.Tensor) else losses['total'],
            'task': losses['task'].item() if isinstance(losses['task'], torch.Tensor) else losses['task'],
            'scl': losses['scl'].item() if isinstance(losses['scl'], torch.Tensor) else losses['scl'],
            'ortho': losses['ortho'].item() if isinstance(losses['ortho'], torch.Tensor) else losses['ortho'],
            'deep': losses['deep'].item() if isinstance(losses['deep'], torch.Tensor) else losses['deep'],
        }
        
        # Add CISL losses if enabled
        if self.config.use_cicl:
            structured_losses['cisl_consist'] = losses.get('cisl_consist', torch.tensor(0.0))
            if isinstance(structured_losses['cisl_consist'], torch.Tensor):
                structured_losses['cisl_consist'] = structured_losses['cisl_consist'].item()
            structured_losses['cisl_content_inv'] = losses.get('cisl_content_inv', torch.tensor(0.0))
            if isinstance(structured_losses['cisl_content_inv'], torch.Tensor):
                structured_losses['cisl_content_inv'] = structured_losses['cisl_content_inv'].item()
            structured_losses['cisl_variance'] = losses.get('cisl_variance', torch.tensor(0.0))
            if isinstance(structured_losses['cisl_variance'], torch.Tensor):
                structured_losses['cisl_variance'] = structured_losses['cisl_variance'].item()
        
        # Add comprehensive health metrics (logit health, accuracy breakdown, confidence)
        for metric_key in ['logit_mean', 'logit_std', 'logit_max', 'logit_min',
                           'accuracy', 'bg_accuracy', 'fg_accuracy', 
                           'confidence_mean', 'pred_entropy',
                           'grad_norm', 'grad_max', 'grad_min', 'grad_has_nan', 'grad_has_inf']:
            if metric_key in losses:
                val = losses[metric_key]
                structured_losses[metric_key] = val.item() if isinstance(val, torch.Tensor) else val
        
        # Log to structured logger (file + statistics tracking)
        self.loss_logger.log_step(
            losses=structured_losses,
            epoch=self.current_epoch,
            step=batch_idx,
            lr=lr,
            extra_metrics={
                'batch_time': batch_time,
                'forward_time': forward_time,
                'backward_time': backward_time,
            }
        )
        
        if self.wandb_run:
            log_dict = {
                'train/loss_total': structured_losses['total'],
                'train/loss_task': structured_losses['task'],
                'train/loss_scl': structured_losses['scl'],
                'train/loss_ortho': structured_losses['ortho'],
                'train/loss_deep': structured_losses['deep'],
                'train/lr': lr,
                'train/epoch': self.current_epoch,
                'train/step': self.global_step,
            }
            # Add CISL losses if enabled (per-batch logging)
            if self.config.use_cicl:
                log_dict['train/cisl_consist'] = structured_losses.get('cisl_consist', 0.0)
                log_dict['train/cisl_content_inv'] = structured_losses.get('cisl_content_inv', 0.0)
                log_dict['train/cisl_variance'] = structured_losses.get('cisl_variance', 0.0)
                log_dict['train/cisl_total'] = structured_losses['scl']  # CISL total goes into scl slot
            
            # Add comprehensive health metrics
            log_dict['train/logit_mean'] = structured_losses.get('logit_mean', 0.0)
            log_dict['train/logit_std'] = structured_losses.get('logit_std', 0.0)
            log_dict['train/logit_max'] = structured_losses.get('logit_max', 0.0)
            log_dict['train/logit_min'] = structured_losses.get('logit_min', 0.0)
            log_dict['train/accuracy'] = structured_losses.get('accuracy', 0.0)
            log_dict['train/bg_accuracy'] = structured_losses.get('bg_accuracy', 0.0)
            log_dict['train/fg_accuracy'] = structured_losses.get('fg_accuracy', 0.0)
            log_dict['train/confidence_mean'] = structured_losses.get('confidence_mean', 0.0)
            log_dict['train/pred_entropy'] = structured_losses.get('pred_entropy', 0.0)
            
            # Add gradient health metrics
            log_dict['train/grad_norm'] = structured_losses.get('grad_norm', 0.0)
            log_dict['train/grad_max'] = structured_losses.get('grad_max', 0.0)
            log_dict['train/grad_min'] = structured_losses.get('grad_min', 0.0)
            log_dict['train/grad_has_nan'] = structured_losses.get('grad_has_nan', 0.0)
            log_dict['train/grad_has_inf'] = structured_losses.get('grad_has_inf', 0.0)
            
            wandb.log(log_dict)
    
    @torch.no_grad()
    def validate(self, use_tta: bool = False, num_dihedral: int = 8, num_color_perms: int = 1) -> Dict[str, float]:
        """
        Run validation with proper accuracy computation.
        
        This method correctly handles:
        - Padding exclusion: -100 in targets is ignored
        - Exact match: compares ONLY valid (non-padding) pixels
        - Optional TTA: when enabled, matches inference behavior
        
        Args:
            use_tta: If True, use Test-Time Augmentation (matches inference).
                     If False, single forward pass (faster but less accurate).
            num_dihedral: Number of dihedral transforms for TTA (1-8, default 8)
            num_color_perms: Color permutations per dihedral for TTA (default 1)
        
        Returns:
            Dict with val_loss, pixel_accuracy, task_accuracy
        
        Note: For TTA evaluation, this method applies augmentations to BOTH
        the demonstration pairs AND the test input (matching TRM-style exactly).
        """
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
        
        # ===============================================
        # COMPREHENSIVE VALIDATION METRICS
        # ===============================================
        # Accumulate detailed metrics for debugging
        val_metrics = {
            'bg_correct': 0,      # Background class accuracy
            'bg_total': 0,
            'fg_correct': 0,      # Foreground class accuracy
            'fg_total': 0,
            'logit_sum': 0.0,     # For computing mean
            'logit_sq_sum': 0.0,  # For computing std
            'logit_max': float('-inf'),
            'logit_min': float('inf'),
            'logit_count': 0,
            'confidence_sum': 0.0,  # Mean prediction confidence
            'entropy_sum': 0.0,     # Mean prediction entropy
            'confidence_count': 0,
        }
        
        # Import augmentation utilities if TTA enabled
        if use_tta:
            from sci_arc.models.rlan_modules.acw import apply_augmentation, AugmentedConfidenceWeighting
        
        for batch_idx, batch in enumerate(self.val_loader):
            batch_start = time.time()
            batch = self._to_device(batch)
            
            B = batch['test_inputs'].shape[0]
            targets = batch['test_outputs']  # [B, H, W] with -100 for padding
            
            if use_tta:
                # ============================================================
                # TTA MODE: Matches inference behavior exactly
                # ============================================================
                # Augmentation order (matching TRM-style):
                # Forward: color perm FIRST → dihedral SECOND
                # Inverse: inverse dihedral FIRST → inverse color SECOND
                
                augmentations = [
                    'identity', 'rotate_90', 'rotate_180', 'rotate_270',
                    'flip_h', 'flip_v', 'transpose', 'transpose_neg'
                ][:num_dihedral]
                
                # Collect predictions per sample
                all_preds_per_sample = [[] for _ in range(B)]
                
                for color_idx in range(num_color_perms):
                    # Step 1: Apply color permutation FIRST
                    if color_idx == 0:
                        color_perm = None
                        color_test = batch['test_inputs']
                        color_train_in = batch['input_grids']
                        color_train_out = batch['output_grids']
                    else:
                        # CRITICAL: Use 11 entries to handle PAD_COLOR=10 in padded grids
                        color_perm = torch.arange(11, device=self.device)
                        shuffled = torch.randperm(9, device=self.device) + 1
                        color_perm[1:10] = shuffled  # Only permute colors 1-9, keep 0 and 10 fixed
                        color_test = color_perm[batch['test_inputs'].clamp(0, 10).long()]
                        color_train_in = color_perm[batch['input_grids'].clamp(0, 10).long()]
                        color_train_out = color_perm[batch['output_grids'].clamp(0, 10).long()]
                    
                    for aug in augmentations:
                        # Step 2: Apply dihedral transform SECOND
                        aug_test = apply_augmentation(color_test, aug)
                        # Handle train grids: (B, N, H, W)
                        N = color_train_in.shape[1]
                        aug_train_in = torch.stack([
                            apply_augmentation(color_train_in[:, i], aug) for i in range(N)
                        ], dim=1)
                        aug_train_out = torch.stack([
                            apply_augmentation(color_train_out[:, i], aug) for i in range(N)
                        ], dim=1)
                        
                        # Forward pass
                        outputs = self.model.forward_training(
                            input_grids=aug_train_in,
                            output_grids=aug_train_out,
                            test_input=aug_test,
                            test_output=batch['test_outputs'],  # Not augmented for loss
                            grid_mask=batch.get('grid_masks'),
                        )
                        
                        preds = outputs['logits'].argmax(dim=-1)  # [B, H, W]
                        
                        # Step 3: Inverse dihedral FIRST
                        preds = apply_augmentation(preds, aug, inverse=True)
                        
                        # Step 4: Inverse color permutation SECOND
                        if color_perm is not None:
                            inv_color_perm = torch.argsort(color_perm)
                            preds = inv_color_perm[preds.long()]
                        
                        # Collect per sample
                        for b in range(B):
                            all_preds_per_sample[b].append(preds[b])
                
                # Vote per sample using hybrid voting
                acw = AugmentedConfidenceWeighting()
                final_preds = []
                for b in range(B):
                    winner, _ = acw.hybrid_vote(all_preds_per_sample[b])
                    final_preds.append(winner)
                
                # Compute accuracy using voted predictions
                for b in range(B):
                    target = targets[b]  # [H, W]
                    pred = final_preds[b]  # May be different shape after voting
                    
                    # Create valid mask for target (exclude -100 padding)
                    valid_mask = (target != -100)
                    
                    if valid_mask.any():
                        # Get target bounds
                        valid_rows = valid_mask.any(dim=1)
                        valid_cols = valid_mask.any(dim=0)
                        r_min, r_max = torch.where(valid_rows)[0][[0, -1]]
                        c_min, c_max = torch.where(valid_cols)[0][[0, -1]]
                        
                        # Crop target to valid region
                        target_cropped = target[r_min:r_max+1, c_min:c_max+1]
                        
                        # Crop prediction to same region (CRITICAL FIX!)
                        # The prediction may still have padding from the model output
                        pred_cropped = pred[r_min:r_max+1, c_min:c_max+1] if (
                            pred.shape[0] >= r_max+1 and pred.shape[1] >= c_max+1
                        ) else pred
                        
                        # Check if prediction matches (shape and values)
                        if pred_cropped.shape == target_cropped.shape:
                            task_correct = torch.equal(pred_cropped, target_cropped)
                            pixel_correct = (pred_cropped == target_cropped).sum().item()
                            pixel_total = target_cropped.numel()
                        else:
                            task_correct = False
                            pixel_correct = 0
                            pixel_total = target_cropped.numel()
                        
                        total_tasks_correct += int(task_correct)
                        total_correct += pixel_correct
                        total_pixels += pixel_total
                    
                    total_tasks += 1
                
                # Loss computation (use single forward pass, just for monitoring)
                outputs = self.model.forward_training(
                    input_grids=batch['input_grids'],
                    output_grids=batch['output_grids'],
                    test_input=batch['test_inputs'],
                    test_output=batch['test_outputs'],
                    grid_mask=batch.get('grid_masks'),
                )
                losses = self._compute_losses(outputs, batch)
                total_loss += losses['total'].item()
                
            else:
                # ============================================================
                # FAST MODE: Single forward pass (no TTA)
                # ============================================================
                # This is faster but measures a different metric than inference.
                # Still accurate for what it measures (single-pass accuracy).
                
                outputs = self.model.forward_training(
                    input_grids=batch['input_grids'],
                    output_grids=batch['output_grids'],
                    test_input=batch['test_inputs'],
                    test_output=batch['test_outputs'],
                    grid_mask=batch.get('grid_masks'),
                )
                
                losses = self._compute_losses(outputs, batch)
                total_loss += losses['total'].item()
                
                # Compute accuracy on valid pixels only
                logits = outputs['logits']  # [B, H, W, C]
                preds = logits.argmax(dim=-1)  # [B, H, W]
                
                # Valid pixel mask: exclude -100 padding
                valid_mask = (targets != -100)  # [B, H, W]
                
                # Pixel accuracy (only on valid pixels)
                correct = ((preds == targets) & valid_mask).sum()
                total_correct += correct.item()
                total_pixels += valid_mask.sum().item()
                
                # === COMPREHENSIVE VALIDATION METRICS ===
                # Background accuracy (class 0)
                bg_mask = (targets == 0) & valid_mask
                if bg_mask.any():
                    bg_correct = ((preds == 0) & bg_mask).sum().item()
                    val_metrics['bg_correct'] += bg_correct
                    val_metrics['bg_total'] += bg_mask.sum().item()
                
                # Foreground accuracy (classes 1-10)
                fg_mask = (targets > 0) & (targets <= 10) & valid_mask
                if fg_mask.any():
                    fg_correct = ((preds == targets) & fg_mask).sum().item()
                    val_metrics['fg_correct'] += fg_correct
                    val_metrics['fg_total'] += fg_mask.sum().item()
                
                # Logit statistics (for numerical stability monitoring)
                val_metrics['logit_sum'] += logits.sum().item()
                val_metrics['logit_sq_sum'] += (logits ** 2).sum().item()
                val_metrics['logit_max'] = max(val_metrics['logit_max'], logits.max().item())
                val_metrics['logit_min'] = min(val_metrics['logit_min'], logits.min().item())
                val_metrics['logit_count'] += logits.numel()
                
                # Confidence and entropy (prediction quality)
                if valid_mask.any():
                    probs = torch.softmax(logits, dim=-1)  # [B, H, W, C]
                    max_probs, _ = probs.max(dim=-1)  # [B, H, W]
                    val_metrics['confidence_sum'] += max_probs[valid_mask].sum().item()
                    
                    # Entropy on valid pixels
                    probs_valid = probs[valid_mask]  # [N_valid, C]
                    entropy_batch = -(probs_valid * (probs_valid + 1e-10).log()).sum(dim=-1).sum()
                    val_metrics['entropy_sum'] += entropy_batch.item()
                    val_metrics['confidence_count'] += valid_mask.sum().item()
                
                # Task accuracy: ALL valid pixels must match
                for b in range(B):
                    sample_mask = valid_mask[b]  # [H, W]
                    if sample_mask.any():
                        sample_preds = preds[b][sample_mask]
                        sample_targets = targets[b][sample_mask]
                        task_correct = torch.equal(sample_preds, sample_targets)
                        total_tasks_correct += int(task_correct)
                    total_tasks += 1
            
            batch_time = time.time() - batch_start
            tta_str = f" [TTA {num_dihedral}×{num_color_perms}]" if use_tta else ""
            print(f"  Val batch {batch_idx + 1}/{num_val_batches}{tta_str} [{batch_time:.1f}s]", end='\r')
        
        val_time = time.time() - val_start_time
        
        # ===============================================
        # COMPUTE COMPREHENSIVE VALIDATION METRICS
        # ===============================================
        metrics = {
            'val_loss': total_loss / max(len(self.val_loader), 1),
            'pixel_accuracy': total_correct / max(total_pixels, 1),
            'task_accuracy': total_tasks_correct / max(total_tasks, 1),
            'use_tta': use_tta,
            'num_views': num_dihedral * num_color_perms if use_tta else 1,
        }
        
        # Add per-class accuracy
        metrics['bg_accuracy'] = val_metrics['bg_correct'] / max(val_metrics['bg_total'], 1)
        metrics['fg_accuracy'] = val_metrics['fg_correct'] / max(val_metrics['fg_total'], 1)
        
        # Add logit health metrics
        if val_metrics['logit_count'] > 0:
            logit_mean = val_metrics['logit_sum'] / val_metrics['logit_count']
            logit_var = (val_metrics['logit_sq_sum'] / val_metrics['logit_count']) - (logit_mean ** 2)
            metrics['logit_mean'] = logit_mean
            metrics['logit_std'] = logit_var ** 0.5 if logit_var > 0 else 0.0
            metrics['logit_max'] = val_metrics['logit_max']
            metrics['logit_min'] = val_metrics['logit_min']
        else:
            metrics['logit_mean'] = 0.0
            metrics['logit_std'] = 0.0
            metrics['logit_max'] = 0.0
            metrics['logit_min'] = 0.0
        
        # Add confidence and entropy metrics
        if val_metrics['confidence_count'] > 0:
            metrics['confidence_mean'] = val_metrics['confidence_sum'] / val_metrics['confidence_count']
            metrics['entropy_mean'] = val_metrics['entropy_sum'] / val_metrics['confidence_count']
        else:
            metrics['confidence_mean'] = 0.0
            metrics['entropy_mean'] = 0.0
        
        # ===============================================
        # COMPREHENSIVE VALIDATION LOGGING
        # ===============================================
        tta_str = f" (TTA {metrics['num_views']} views)" if use_tta else " (fast mode)"
        print(f"\n{'='*60}")
        print(f"VALIDATION SUMMARY - Epoch {self.current_epoch + 1}{tta_str}")
        print(f"{'='*60}")
        print(f"  Time: {val_time:.1f}s")
        print(f"  Loss: {metrics['val_loss']:.4f}")
        print(f"\n  ACCURACY:")
        print(f"    Pixel Accuracy:      {metrics['pixel_accuracy']:.4f}")
        print(f"    Task Accuracy:       {metrics['task_accuracy']:.4f}")
        print(f"    Background Accuracy: {metrics['bg_accuracy']:.4f}")
        print(f"    Foreground Accuracy: {metrics['fg_accuracy']:.4f}")
        print(f"\n  PREDICTION HEALTH:")
        print(f"    Confidence (mean):   {metrics['confidence_mean']:.4f}")
        print(f"    Entropy (mean):      {metrics['entropy_mean']:.4f}")
        print(f"\n  LOGIT HEALTH:")
        print(f"    Mean: {metrics['logit_mean']:.4f}, Std: {metrics['logit_std']:.4f}")
        print(f"    Range: [{metrics['logit_min']:.4f}, {metrics['logit_max']:.4f}]")
        print(f"{'='*60}")
        
        if self.wandb_run:
            wandb.log({
                'val/loss': metrics['val_loss'],
                'val/pixel_accuracy': metrics['pixel_accuracy'],
                'val/task_accuracy': metrics['task_accuracy'],
                'val/bg_accuracy': metrics['bg_accuracy'],
                'val/fg_accuracy': metrics['fg_accuracy'],
                'val/confidence_mean': metrics['confidence_mean'],
                'val/entropy_mean': metrics['entropy_mean'],
                'val/logit_mean': metrics['logit_mean'],
                'val/logit_std': metrics['logit_std'],
                'val/logit_max': metrics['logit_max'],
                'val/logit_min': metrics['logit_min'],
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
                
                # Use TTA if configured (recommended for accurate generalization metrics)
                val_metrics = self.validate(
                    use_tta=self.config.val_use_tta,
                    num_dihedral=self.config.val_num_dihedral,
                    num_color_perms=self.config.val_num_color_perms,
                )
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
        from .rlan_loss import RLANLoss
        loss_fn = RLANLoss()
    
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
