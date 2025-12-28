"""
Loss Logger for RLAN Training.

Provides structured, per-component loss logging with:
1. Per-step logging of all loss components
2. Moving averages for trend detection
3. Ratio-to-total metrics to detect component dominance
4. Silent collapse/regression detection
5. Console and optional file output

Usage:
    logger = LossLogger(log_dir="./logs", log_interval=10)
    
    # In training loop:
    logger.log_step(losses, epoch, step, lr)
    
    # At epoch end:
    logger.log_epoch_summary(epoch)
"""

import os
import json
from pathlib import Path
from typing import Dict, Optional, List, Any
from datetime import datetime
from collections import deque
from dataclasses import dataclass, field, asdict
import math

import torch


@dataclass
class LossStats:
    """Statistics for a single loss component."""
    name: str
    current: float = 0.0
    running_mean: float = 0.0
    running_std: float = 0.0
    min_value: float = float('inf')
    max_value: float = float('-inf')
    num_samples: int = 0
    is_zero_count: int = 0  # Count of zero values (collapse detection)
    is_nan_count: int = 0   # Count of NaN values (stability detection)
    
    # For moving average computation
    ema_alpha: float = 0.1  # Exponential moving average decay
    ema_value: float = 0.0
    
    def update(self, value: float):
        """Update statistics with a new value."""
        self.num_samples += 1
        self.current = value
        
        # Handle special cases
        if math.isnan(value):
            self.is_nan_count += 1
            return
        if abs(value) < 1e-10:
            self.is_zero_count += 1
        
        # Update min/max
        self.min_value = min(self.min_value, value)
        self.max_value = max(self.max_value, value)
        
        # Update running mean and std (Welford's algorithm)
        if self.num_samples == 1:
            self.running_mean = value
            self.running_std = 0.0
            self.ema_value = value
        else:
            delta = value - self.running_mean
            self.running_mean += delta / self.num_samples
            delta2 = value - self.running_mean
            # Online variance update
            if self.num_samples > 1:
                self.running_std = math.sqrt(
                    ((self.num_samples - 2) * self.running_std ** 2 + delta * delta2) / (self.num_samples - 1)
                ) if self.num_samples > 1 else 0.0
            
            # EMA update
            self.ema_value = self.ema_alpha * value + (1 - self.ema_alpha) * self.ema_value
    
    def get_collapse_warning(self) -> Optional[str]:
        """Check for signs of collapse or instability."""
        if self.num_samples < 10:
            return None
        
        zero_ratio = self.is_zero_count / self.num_samples
        nan_ratio = self.is_nan_count / self.num_samples
        
        if nan_ratio > 0.01:  # More than 1% NaN
            return f"[!] {self.name}: {nan_ratio:.1%} NaN values detected (stability issue)"
        if zero_ratio > 0.5:  # More than 50% zeros
            return f"[!] {self.name}: {zero_ratio:.1%} zero values (possible collapse)"
        
        return None
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for logging."""
        return {
            f"{self.name}/current": self.current,
            f"{self.name}/ema": self.ema_value,
            f"{self.name}/mean": self.running_mean,
            f"{self.name}/std": self.running_std,
            f"{self.name}/min": self.min_value if self.min_value != float('inf') else 0.0,
            f"{self.name}/max": self.max_value if self.max_value != float('-inf') else 0.0,
        }


class LossLogger:
    """
    Structured loss logger with component tracking and health monitoring.
    
    Features:
    - Per-component loss tracking with moving averages
    - Ratio-to-total computation (detect dominance)
    - Collapse/NaN detection with warnings
    - Console + optional file + optional wandb logging
    - Epoch summaries with trend analysis
    """
    
    # Standard loss components in RLAN (expanded for HyperLoRA and comprehensive tracking)
    LOSS_COMPONENTS = [
        'total', 'task', 'deep', 'entropy', 'sparsity', 
        'predicate', 'curriculum', 'act', 'scl', 'ortho',
        'cisl_consist', 'cisl_content_inv', 'cisl_variance',
        # HyperLoRA-specific losses
        'loo', 'equivariance', 'hyperlora_total',
        # META-LEARNING HEALTH METRICS (Dec 2025)
        'loo_accuracy',              # LOO prediction accuracy (should be >0.5)
        'loo_skipped_ratio',         # Ratio of batches with LOO skipped (want low)
        'lora_magnitude_gru_reset',  # LoRA weight magnitude (want non-zero)
        'lora_magnitude_gru_update', # LoRA weight magnitude
        'lora_magnitude_gru_candidate',  # LoRA weight magnitude
        'lora_magnitude_output_head',    # LoRA weight magnitude
        'lora_diversity',            # Weight diversity across batch (want high)
        'equiv_delta_norm',          # Equivariance delta norm (want low)
        # LCR (Latent Counting Registers) health metrics
        'lcr_magnitude',             # Count embedding magnitude (want non-zero)
        'lcr_std',                   # Count embedding std (want variation)
        'lcr_color_diversity',       # Diversity across colors (want differentiation)
        # SPH (Symbolic Predicate Heads) health metrics
        'sph_mean',                  # Predicate mean (want ~0.5 for balanced)
        'sph_std',                   # Predicate std (want variation)
        'sph_activation_ratio',      # Ratio of active predicates (want 0.2-0.8)
        # Gradient health metrics
        'grad_norm', 'grad_max', 'grad_min', 'grad_has_nan', 'grad_has_inf',
        # Logit health metrics  
        'logit_mean', 'logit_std', 'logit_max', 'logit_min',
        # Prediction quality metrics
        'accuracy', 'bg_accuracy', 'fg_accuracy',
        # Confidence/entropy metrics
        'confidence_mean', 'pred_entropy',
        # Attention/clue metrics
        'attention_entropy', 'num_active_clues', 'stop_prob_mean',
    ]
    
    def __init__(
        self,
        log_dir: Optional[str] = None,
        log_interval: int = 10,
        enable_file_logging: bool = True,
        enable_console: bool = True,
        wandb_run: Optional[Any] = None,
        verbose: bool = False,
    ):
        """
        Initialize the loss logger.
        
        Args:
            log_dir: Directory for log files (None = no file logging)
            log_interval: Log to console every N steps
            enable_file_logging: Write detailed logs to file
            enable_console: Print summaries to console
            wandb_run: Optional wandb run for cloud logging
            verbose: Print every step (for debugging)
        """
        self.log_dir = Path(log_dir) if log_dir else None
        self.log_interval = log_interval
        self.enable_file_logging = enable_file_logging and log_dir is not None
        self.enable_console = enable_console
        self.wandb_run = wandb_run
        self.verbose = verbose
        
        # Initialize stats for each component
        self.stats: Dict[str, LossStats] = {}
        for name in self.LOSS_COMPONENTS:
            self.stats[name] = LossStats(name=name)
        
        # Per-epoch tracking
        self.epoch_losses: Dict[str, List[float]] = {name: [] for name in self.LOSS_COMPONENTS}
        self.current_epoch = 0
        self.global_step = 0
        
        # Recent history for trend detection (last 100 steps)
        self.recent_total: deque = deque(maxlen=100)
        
        # Setup file logging
        if self.enable_file_logging and self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.loss_log_path = self.log_dir / f"loss_log_{timestamp}.jsonl"
            self.summary_log_path = self.log_dir / f"loss_summary_{timestamp}.txt"
        else:
            self.loss_log_path = None
            self.summary_log_path = None
    
    def _extract_loss_value(self, losses: Dict, key: str) -> float:
        """Extract float value from loss dict (handles tensors)."""
        value = losses.get(key, 0.0)
        if isinstance(value, torch.Tensor):
            value = value.item()
        return float(value) if not math.isnan(value) else 0.0
    
    def log_step(
        self,
        losses: Dict[str, Any],
        epoch: int,
        step: int,
        lr: float,
        extra_metrics: Optional[Dict[str, float]] = None,
    ):
        """
        Log a single training step.
        
        Args:
            losses: Dictionary of loss values (can be tensors or floats)
            epoch: Current epoch number
            step: Current step within epoch
            lr: Current learning rate
            extra_metrics: Additional metrics to log
        """
        self.current_epoch = epoch
        self.global_step += 1
        
        # Extract and update stats for each component
        log_entry = {
            'epoch': epoch,
            'step': step,
            'global_step': self.global_step,
            'lr': lr,
            'timestamp': datetime.now().isoformat(),
        }
        
        total_loss = self._extract_loss_value(losses, 'total')
        
        for name in self.LOSS_COMPONENTS:
            value = self._extract_loss_value(losses, name)
            self.stats[name].update(value)
            self.epoch_losses[name].append(value)
            log_entry[f'loss_{name}'] = value
            
            # Compute ratio to total (detect dominance)
            if total_loss > 1e-10 and name != 'total':
                ratio = value / total_loss if not math.isnan(value) else 0.0
                log_entry[f'ratio_{name}'] = ratio
        
        # Track recent history
        self.recent_total.append(total_loss)
        
        # Add extra metrics
        if extra_metrics:
            log_entry.update(extra_metrics)
        
        # File logging (every step)
        if self.enable_file_logging and self.loss_log_path:
            with open(self.loss_log_path, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        
        # Console logging (periodic)
        should_log = (step + 1) % self.log_interval == 0 or self.verbose
        if should_log and self.enable_console:
            self._print_step_summary(epoch, step, losses, lr)
        
        # Wandb logging
        if self.wandb_run:
            self._log_to_wandb(log_entry)
    
    def _print_step_summary(self, epoch: int, step: int, losses: Dict, lr: float):
        """Print a concise step summary to console."""
        total = self._extract_loss_value(losses, 'total')
        task = self._extract_loss_value(losses, 'task')
        deep = self._extract_loss_value(losses, 'deep')
        entropy = self._extract_loss_value(losses, 'entropy')
        sparsity = self._extract_loss_value(losses, 'sparsity')
        
        # Compute trend indicator
        trend = self._compute_trend()
        
        print(
            f"[E{epoch+1}|S{step+1}] "
            f"Loss: {total:.4f}{trend} "
            f"(task={task:.4f}, deep={deep:.4f}, ent={entropy:.4f}, spar={sparsity:.4f}) "
            f"LR: {lr:.2e}"
        )
    
    def _compute_trend(self) -> str:
        """Compute trend indicator based on recent history."""
        if len(self.recent_total) < 20:
            return ""
        
        recent_mean = sum(list(self.recent_total)[-20:]) / 20
        older_mean = sum(list(self.recent_total)[:20]) / 20
        
        if older_mean < 1e-10:
            return ""
        
        change = (recent_mean - older_mean) / older_mean
        
        if change < -0.1:
            return " ↓"  # Decreasing (good)
        elif change > 0.1:
            return " ↑"  # Increasing (bad)
        else:
            return " →"  # Stable
    
    def _log_to_wandb(self, log_entry: Dict):
        """Log to wandb with proper prefixing."""
        try:
            import wandb
            wandb_dict = {
                'train/step': log_entry['global_step'],
                'train/epoch': log_entry['epoch'],
                'train/lr': log_entry['lr'],
            }
            
            for name in self.LOSS_COMPONENTS:
                key = f'loss_{name}'
                if key in log_entry:
                    wandb_dict[f'train/{key}'] = log_entry[key]
                
                ratio_key = f'ratio_{name}'
                if ratio_key in log_entry:
                    wandb_dict[f'train/{ratio_key}'] = log_entry[ratio_key]
            
            # Add EMA values
            for name, stats in self.stats.items():
                wandb_dict[f'train/ema_{name}'] = stats.ema_value
            
            wandb.log(wandb_dict)
        except Exception as e:
            pass  # Silently fail if wandb has issues
    
    def log_epoch_summary(self, epoch: int) -> Dict[str, float]:
        """
        Log end-of-epoch summary with statistics and health checks.
        
        Args:
            epoch: Completed epoch number
            
        Returns:
            Dictionary with epoch summary metrics
        """
        summary = {
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
        }
        
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1} Loss Summary")
        print(f"{'='*60}")
        
        # Compute per-component stats for this epoch
        for name in self.LOSS_COMPONENTS:
            values = self.epoch_losses[name]
            if not values:
                continue
            
            mean_val = sum(values) / len(values)
            min_val = min(values)
            max_val = max(values)
            
            summary[f'{name}_mean'] = mean_val
            summary[f'{name}_min'] = min_val
            summary[f'{name}_max'] = max_val
            
            stats = self.stats[name]
            print(f"  {name:20s}: mean={mean_val:.6f}, min={min_val:.6f}, max={max_val:.6f}, EMA={stats.ema_value:.6f}")
        
        # Health checks
        print(f"\n--- Health Checks ---")
        warnings = []
        for name, stats in self.stats.items():
            warning = stats.get_collapse_warning()
            if warning:
                warnings.append(warning)
                print(warning)
        
        if not warnings:
            print("[OK] No collapse or stability issues detected")
        
        # === META-LEARNING HEALTH CHECK ===
        print(f"\n--- Meta-Learning Health ---")
        meta_warnings = []
        
        # LOO Training Health
        loo_acc_vals = self.epoch_losses.get('loo_accuracy', [])
        loo_skip_vals = self.epoch_losses.get('loo_skipped_ratio', [])
        if loo_acc_vals:
            mean_loo_acc = sum(loo_acc_vals) / len(loo_acc_vals)
            summary['loo_accuracy_mean'] = mean_loo_acc
            if mean_loo_acc < 0.3:
                meta_warnings.append(f"  [!] LOO accuracy very low ({mean_loo_acc:.1%}) - HyperLoRA not learning")
                print(f"  [!] LOO accuracy: {mean_loo_acc:.1%} (CRITICAL: < 30%)")
            elif mean_loo_acc < 0.5:
                meta_warnings.append(f"  [!] LOO accuracy below random ({mean_loo_acc:.1%})")
                print(f"  [!] LOO accuracy: {mean_loo_acc:.1%} (WARNING: < 50%)")
            else:
                print(f"  [OK] LOO accuracy: {mean_loo_acc:.1%}")
        
        if loo_skip_vals:
            mean_skip = sum(loo_skip_vals) / len(loo_skip_vals)
            summary['loo_skipped_ratio_mean'] = mean_skip
            if mean_skip > 0.5:
                meta_warnings.append(f"  [!] LOO skipped too often ({mean_skip:.1%})")
                print(f"  [!] LOO skipped ratio: {mean_skip:.1%} (WARNING: > 50%)")
            else:
                print(f"  [OK] LOO skipped ratio: {mean_skip:.1%}")
        
        # LoRA Magnitude Health (want non-zero, not too large)
        lora_keys = ['lora_magnitude_gru_reset', 'lora_magnitude_gru_update', 
                     'lora_magnitude_gru_candidate', 'lora_magnitude_output_head']
        for lora_key in lora_keys:
            lora_vals = self.epoch_losses.get(lora_key, [])
            if lora_vals:
                mean_mag = sum(lora_vals) / len(lora_vals)
                summary[f'{lora_key}_mean'] = mean_mag
                if mean_mag < 1e-6:
                    meta_warnings.append(f"  [!] {lora_key} collapsed to ~0")
                    print(f"  [!] {lora_key}: {mean_mag:.6f} (COLLAPSED)")
                elif mean_mag > 100:
                    meta_warnings.append(f"  [!] {lora_key} exploding ({mean_mag:.1f})")
                    print(f"  [!] {lora_key}: {mean_mag:.2f} (EXPLODING)")
                else:
                    print(f"  [OK] {lora_key}: {mean_mag:.4f}")
        
        # LoRA Diversity (want > 0 for meta-learning)
        diversity_vals = self.epoch_losses.get('lora_diversity', [])
        if diversity_vals:
            mean_div = sum(diversity_vals) / len(diversity_vals)
            summary['lora_diversity_mean'] = mean_div
            if mean_div < 1e-6:
                meta_warnings.append("  [!] LoRA diversity collapsed - all tasks getting same weights")
                print(f"  [!] LoRA diversity: {mean_div:.6f} (NO META-LEARNING)")
            else:
                print(f"  [OK] LoRA diversity: {mean_div:.4f}")
        
        # Equivariance Health (want low delta norm)
        equiv_vals = self.epoch_losses.get('equiv_delta_norm', [])
        if equiv_vals:
            mean_equiv = sum(equiv_vals) / len(equiv_vals)
            summary['equiv_delta_norm_mean'] = mean_equiv
            if mean_equiv > 10.0:
                meta_warnings.append(f"  [!] Equivariance delta high ({mean_equiv:.2f}) - LoRA not transform-invariant")
                print(f"  [!] Equiv delta norm: {mean_equiv:.4f} (HIGH)")
            else:
                print(f"  [OK] Equiv delta norm: {mean_equiv:.4f}")
        
        if not meta_warnings and (loo_acc_vals or diversity_vals):
            print("[OK] Meta-learning appears healthy")
        elif not loo_acc_vals and not diversity_vals:
            print("[--] Meta-learning not enabled or no data")
        
        summary['meta_warnings'] = meta_warnings
        
        # === LCR/SPH HEALTH CHECK ===
        lcr_vals = self.epoch_losses.get('lcr_magnitude', [])
        sph_vals = self.epoch_losses.get('sph_activation_ratio', [])
        if lcr_vals or sph_vals:
            print(f"\n--- LCR/SPH Health ---")
            
            # LCR Health
            if lcr_vals:
                mean_lcr_mag = sum(lcr_vals) / len(lcr_vals)
                summary['lcr_magnitude_mean'] = mean_lcr_mag
                lcr_div_vals = self.epoch_losses.get('lcr_color_diversity', [])
                mean_lcr_div = sum(lcr_div_vals) / len(lcr_div_vals) if lcr_div_vals else 0.0
                
                if mean_lcr_mag < 1e-6:
                    print(f"  [!] LCR magnitude collapsed: {mean_lcr_mag:.6f}")
                    meta_warnings.append("LCR magnitude collapsed")
                elif mean_lcr_div < 1e-4:
                    print(f"  [!] LCR no color differentiation: {mean_lcr_div:.6f}")
                    meta_warnings.append("LCR not differentiating colors")
                else:
                    print(f"  [OK] LCR magnitude: {mean_lcr_mag:.4f}, color diversity: {mean_lcr_div:.4f}")
            
            # SPH Health
            if sph_vals:
                mean_sph_act = sum(sph_vals) / len(sph_vals)
                summary['sph_activation_ratio_mean'] = mean_sph_act
                sph_std_vals = self.epoch_losses.get('sph_std', [])
                mean_sph_std = sum(sph_std_vals) / len(sph_std_vals) if sph_std_vals else 0.0
                
                if mean_sph_act < 0.05 or mean_sph_act > 0.95:
                    print(f"  [!] SPH activation ratio extreme: {mean_sph_act:.1%}")
                    meta_warnings.append(f"SPH activation ratio extreme ({mean_sph_act:.1%})")
                elif mean_sph_std < 1e-4:
                    print(f"  [!] SPH no predicate variation: std={mean_sph_std:.6f}")
                    meta_warnings.append("SPH predicates not varying")
                else:
                    print(f"  [OK] SPH activation: {mean_sph_act:.1%}, variation: {mean_sph_std:.4f}")
        
        # Compute total loss trend for this epoch
        total_values = self.epoch_losses['total']
        if len(total_values) >= 10:
            first_half = sum(total_values[:len(total_values)//2]) / (len(total_values)//2)
            second_half = sum(total_values[len(total_values)//2:]) / (len(total_values)//2)
            
            if first_half > 1e-10:
                epoch_trend = (second_half - first_half) / first_half
                trend_str = "↓ decreasing" if epoch_trend < -0.05 else "↑ increasing" if epoch_trend > 0.05 else "→ stable"
                print(f"\n[Trend] Total loss within epoch: {trend_str} ({epoch_trend:+.1%})")
                summary['epoch_trend'] = epoch_trend
        
        print(f"{'='*60}\n")
        
        # Write summary to file
        if self.enable_file_logging and self.summary_log_path:
            with open(self.summary_log_path, 'a') as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"Epoch {epoch + 1} Summary\n")
                f.write(json.dumps(summary, indent=2) + '\n')
        
        # Reset epoch losses for next epoch
        self.epoch_losses = {name: [] for name in self.LOSS_COMPONENTS}
        
        return summary
    
    def get_stats_dict(self) -> Dict[str, Dict[str, float]]:
        """Get all statistics as a nested dictionary."""
        return {name: stats.to_dict() for name, stats in self.stats.items()}
    
    def check_health(self) -> List[str]:
        """
        Run health checks and return list of warnings.
        
        Returns:
            List of warning messages (empty if healthy)
        """
        warnings = []
        
        for name, stats in self.stats.items():
            warning = stats.get_collapse_warning()
            if warning:
                warnings.append(warning)
        
        # Check for loss explosion
        if self.stats['total'].ema_value > 100:
            warnings.append(f"[!] Total loss EMA is very high: {self.stats['total'].ema_value:.2f}")
        
        # Check for component dominance (one component > 80% of total)
        total_ema = self.stats['total'].ema_value
        if total_ema > 1e-10:
            for name, stats in self.stats.items():
                if name != 'total' and stats.ema_value / total_ema > 0.8:
                    warnings.append(f"[!] {name} dominates total loss ({stats.ema_value/total_ema:.1%})")
        
        return warnings
