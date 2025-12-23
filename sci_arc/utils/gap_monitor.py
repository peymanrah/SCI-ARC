"""
Train/Eval Gap Health Monitor for RLAN.

This module provides early warning detection for generalization gaps during training.
The key insight is that small gaps early in training often become catastrophic later.

Monitored Metrics:
1. Exact Match: train_acc - eval_acc
2. Entropy: eval_entropy - train_entropy (higher eval = bad)
3. Stop Value: train_stop - eval_stop (lag indicates EMA issues)
4. Loss: eval_loss - train_loss

Usage:
    monitor = GapHealthMonitor()
    
    for epoch in range(max_epochs):
        train_metrics = train_one_epoch(model, train_loader)
        eval_metrics = evaluate(model, eval_loader)
        
        # Check for gaps
        alerts = monitor.check_health(train_metrics, eval_metrics, epoch)
        
        for alert in alerts:
            if alert.severity == 'critical':
                logger.critical(alert.message)
            elif alert.severity == 'warning':
                logger.warning(alert.message)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum
import math


class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class HealthAlert:
    """Alert raised when gap threshold is exceeded."""
    metric: str
    severity: AlertSeverity
    message: str
    train_value: float
    eval_value: float
    gap: float
    threshold: float


class GapHealthMonitor:
    """
    Monitors train/eval gaps and raises alerts when thresholds are exceeded.
    
    Thresholds are calibrated based on observed behavior:
    - Healthy training: <10% relative gap on most metrics
    - Warning: 10-30% relative gap 
    - Critical: >30% relative gap OR >5x entropy ratio
    """
    
    def __init__(
        self,
        exact_match_warning: float = 0.10,  # 10% absolute difference
        exact_match_critical: float = 0.20,  # 20% absolute difference
        entropy_ratio_warning: float = 2.0,  # eval/train ratio
        entropy_ratio_critical: float = 5.0,  # 5x worse
        stop_value_warning: float = 0.15,    # 15% absolute difference
        stop_value_critical: float = 0.25,   # 25% difference
        loss_ratio_warning: float = 1.5,     # 50% worse
        loss_ratio_critical: float = 2.0,    # 2x worse
    ):
        self.thresholds = {
            'exact_match': {
                'warning': exact_match_warning,
                'critical': exact_match_critical,
            },
            'entropy': {
                'warning': entropy_ratio_warning,
                'critical': entropy_ratio_critical,
            },
            'stop_value': {
                'warning': stop_value_warning,
                'critical': stop_value_critical,
            },
            'loss': {
                'warning': loss_ratio_warning,
                'critical': loss_ratio_critical,
            }
        }
        
        # History for trend detection
        self.history: List[Dict[str, float]] = []
        
    def check_health(
        self,
        train_metrics: Dict[str, float],
        eval_metrics: Dict[str, float],
        epoch: int = 0,
    ) -> List[HealthAlert]:
        """
        Check train/eval gaps and return any alerts.
        
        Args:
            train_metrics: Dict with keys like 'exact_match', 'dsc_entropy', 'stop_value', 'loss'
            eval_metrics: Same structure as train_metrics
            epoch: Current epoch number
            
        Returns:
            List of HealthAlert objects for any exceeded thresholds
        """
        alerts = []
        gap_record = {'epoch': epoch}
        
        # Check exact match gap (train - eval, higher train is bad)
        if 'exact_match' in train_metrics and 'exact_match' in eval_metrics:
            train_acc = train_metrics['exact_match']
            eval_acc = eval_metrics['exact_match']
            gap = train_acc - eval_acc
            gap_record['exact_match_gap'] = gap
            
            if gap > self.thresholds['exact_match']['critical']:
                alerts.append(HealthAlert(
                    metric='exact_match',
                    severity=AlertSeverity.CRITICAL,
                    message=f"CRITICAL: Exact match gap {gap:.1%} (train={train_acc:.1%}, eval={eval_acc:.1%}). Model is severely overfitting!",
                    train_value=train_acc,
                    eval_value=eval_acc,
                    gap=gap,
                    threshold=self.thresholds['exact_match']['critical'],
                ))
            elif gap > self.thresholds['exact_match']['warning']:
                alerts.append(HealthAlert(
                    metric='exact_match',
                    severity=AlertSeverity.WARNING,
                    message=f"Warning: Exact match gap {gap:.1%} (train={train_acc:.1%}, eval={eval_acc:.1%}). Consider more regularization.",
                    train_value=train_acc,
                    eval_value=eval_acc,
                    gap=gap,
                    threshold=self.thresholds['exact_match']['warning'],
                ))
        
        # Check entropy ratio (eval/train, higher eval is bad)
        train_entropy_key = next((k for k in train_metrics if 'entropy' in k.lower()), None)
        eval_entropy_key = next((k for k in eval_metrics if 'entropy' in k.lower()), None)
        
        if train_entropy_key and eval_entropy_key:
            train_entropy = train_metrics[train_entropy_key]
            eval_entropy = eval_metrics[eval_entropy_key]
            
            # Avoid division by zero
            if train_entropy > 0.001:
                ratio = eval_entropy / train_entropy
                gap_record['entropy_ratio'] = ratio
                
                if ratio > self.thresholds['entropy']['critical']:
                    alerts.append(HealthAlert(
                        metric='entropy',
                        severity=AlertSeverity.CRITICAL,
                        message=f"CRITICAL: Entropy ratio {ratio:.1f}x (train={train_entropy:.4f}, eval={eval_entropy:.4f}). Attention not transferring!",
                        train_value=train_entropy,
                        eval_value=eval_entropy,
                        gap=ratio,
                        threshold=self.thresholds['entropy']['critical'],
                    ))
                elif ratio > self.thresholds['entropy']['warning']:
                    alerts.append(HealthAlert(
                        metric='entropy',
                        severity=AlertSeverity.WARNING,
                        message=f"Warning: Entropy ratio {ratio:.1f}x (train={train_entropy:.4f}, eval={eval_entropy:.4f}). Check attention stability.",
                        train_value=train_entropy,
                        eval_value=eval_entropy,
                        gap=ratio,
                        threshold=self.thresholds['entropy']['warning'],
                    ))
        
        # Check stop value gap (train - eval, indicates EMA lag)
        train_stop_key = next((k for k in train_metrics if 'stop' in k.lower()), None)
        eval_stop_key = next((k for k in eval_metrics if 'stop' in k.lower()), None)
        
        if train_stop_key and eval_stop_key:
            train_stop = train_metrics[train_stop_key]
            eval_stop = eval_metrics[eval_stop_key]
            gap = abs(train_stop - eval_stop)
            gap_record['stop_gap'] = gap
            
            if gap > self.thresholds['stop_value']['critical']:
                alerts.append(HealthAlert(
                    metric='stop_value',
                    severity=AlertSeverity.CRITICAL,
                    message=f"CRITICAL: Stop value gap {gap:.3f} (train={train_stop:.3f}, eval={eval_stop:.3f}). EMA severely lagging!",
                    train_value=train_stop,
                    eval_value=eval_stop,
                    gap=gap,
                    threshold=self.thresholds['stop_value']['critical'],
                ))
            elif gap > self.thresholds['stop_value']['warning']:
                alerts.append(HealthAlert(
                    metric='stop_value',
                    severity=AlertSeverity.WARNING,
                    message=f"Warning: Stop value gap {gap:.3f} (train={train_stop:.3f}, eval={eval_stop:.3f}). Consider faster EMA decay.",
                    train_value=train_stop,
                    eval_value=eval_stop,
                    gap=gap,
                    threshold=self.thresholds['stop_value']['warning'],
                ))
        
        # Check loss ratio (eval/train, higher eval is bad)
        train_loss_key = next((k for k in train_metrics if 'loss' in k.lower() and 'total' not in k.lower()), None)
        eval_loss_key = next((k for k in eval_metrics if 'loss' in k.lower() and 'total' not in k.lower()), None)
        
        if train_loss_key and eval_loss_key:
            train_loss = train_metrics[train_loss_key]
            eval_loss = eval_metrics[eval_loss_key]
            
            if train_loss > 0.001:
                ratio = eval_loss / train_loss
                gap_record['loss_ratio'] = ratio
                
                if ratio > self.thresholds['loss']['critical']:
                    alerts.append(HealthAlert(
                        metric='loss',
                        severity=AlertSeverity.CRITICAL,
                        message=f"CRITICAL: Loss ratio {ratio:.1f}x (train={train_loss:.4f}, eval={eval_loss:.4f}). Severe generalization issue!",
                        train_value=train_loss,
                        eval_value=eval_loss,
                        gap=ratio,
                        threshold=self.thresholds['loss']['critical'],
                    ))
                elif ratio > self.thresholds['loss']['warning']:
                    alerts.append(HealthAlert(
                        metric='loss',
                        severity=AlertSeverity.WARNING,
                        message=f"Warning: Loss ratio {ratio:.1f}x (train={train_loss:.4f}, eval={eval_loss:.4f}). Monitor closely.",
                        train_value=train_loss,
                        eval_value=eval_loss,
                        gap=ratio,
                        threshold=self.thresholds['loss']['warning'],
                    ))
        
        # Store history for trend analysis
        self.history.append(gap_record)
        
        # Check for worsening trends
        if len(self.history) >= 3:
            trend_alerts = self._check_trends()
            alerts.extend(trend_alerts)
        
        return alerts
    
    def _check_trends(self) -> List[HealthAlert]:
        """Check for worsening trends over recent epochs."""
        alerts = []
        
        if len(self.history) < 3:
            return alerts
        
        recent = self.history[-3:]
        
        # Check if exact_match_gap is consistently increasing
        if all('exact_match_gap' in h for h in recent):
            gaps = [h['exact_match_gap'] for h in recent]
            if gaps[0] < gaps[1] < gaps[2] and gaps[2] > 0.05:
                alerts.append(HealthAlert(
                    metric='exact_match_trend',
                    severity=AlertSeverity.WARNING,
                    message=f"Warning: Exact match gap increasing over last 3 epochs ({gaps[0]:.1%} → {gaps[1]:.1%} → {gaps[2]:.1%}). Overfitting accelerating!",
                    train_value=gaps[0],
                    eval_value=gaps[2],
                    gap=gaps[2] - gaps[0],
                    threshold=0.0,
                ))
        
        # Check if entropy_ratio is consistently increasing
        if all('entropy_ratio' in h for h in recent):
            ratios = [h['entropy_ratio'] for h in recent]
            if ratios[0] < ratios[1] < ratios[2] and ratios[2] > 1.5:
                alerts.append(HealthAlert(
                    metric='entropy_trend',
                    severity=AlertSeverity.WARNING,
                    message=f"Warning: Entropy ratio increasing over last 3 epochs ({ratios[0]:.1f}x → {ratios[1]:.1f}x → {ratios[2]:.1f}x). Attention diverging!",
                    train_value=ratios[0],
                    eval_value=ratios[2],
                    gap=ratios[2] - ratios[0],
                    threshold=0.0,
                ))
        
        return alerts
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics over training."""
        if not self.history:
            return {}
        
        summary = {
            'num_epochs': len(self.history),
        }
        
        # Compute average gaps
        for key in ['exact_match_gap', 'entropy_ratio', 'stop_gap', 'loss_ratio']:
            values = [h[key] for h in self.history if key in h]
            if values:
                summary[f'avg_{key}'] = sum(values) / len(values)
                summary[f'max_{key}'] = max(values)
                summary[f'min_{key}'] = min(values)
        
        return summary
    
    def reset(self):
        """Clear history."""
        self.history = []


def integrate_with_training_loop(trainer, monitor: GapHealthMonitor):
    """
    Example integration with RLAN training loop.
    
    Add this to your training script:
    
    ```python
    from sci_arc.utils.gap_monitor import GapHealthMonitor
    
    monitor = GapHealthMonitor()
    
    for epoch in range(max_epochs):
        train_metrics = train_one_epoch(...)
        eval_metrics = evaluate(...)
        
        # Check gaps
        alerts = monitor.check_health(train_metrics, eval_metrics, epoch)
        
        for alert in alerts:
            if alert.severity == AlertSeverity.CRITICAL:
                logger.critical(alert.message)
                # Optionally: early stop, reduce LR, increase regularization
            elif alert.severity == AlertSeverity.WARNING:
                logger.warning(alert.message)
    
    # End of training: print summary
    summary = monitor.get_summary()
    logger.info(f"Gap Summary: {summary}")
    ```
    """
    pass


if __name__ == "__main__":
    # Test the monitor
    print("Testing GapHealthMonitor...")
    
    monitor = GapHealthMonitor()
    
    # Simulate training progression with increasing gap
    test_cases = [
        # (epoch, train_metrics, eval_metrics)
        (1, {'exact_match': 0.05, 'dsc_entropy': 0.02, 'stop_value': 0.5, 'task_loss': 1.0},
            {'exact_match': 0.04, 'dsc_entropy': 0.03, 'stop_value': 0.48, 'task_loss': 1.1}),
        (2, {'exact_match': 0.10, 'dsc_entropy': 0.02, 'stop_value': 0.55, 'task_loss': 0.8},
            {'exact_match': 0.05, 'dsc_entropy': 0.08, 'stop_value': 0.45, 'task_loss': 1.2}),
        (3, {'exact_match': 0.15, 'dsc_entropy': 0.02, 'stop_value': 0.60, 'task_loss': 0.6},
            {'exact_match': 0.02, 'dsc_entropy': 0.50, 'stop_value': 0.35, 'task_loss': 1.5}),
        (4, {'exact_match': 0.20, 'dsc_entropy': 0.02, 'stop_value': 0.65, 'task_loss': 0.4},
            {'exact_match': 0.01, 'dsc_entropy': 1.00, 'stop_value': 0.30, 'task_loss': 2.0}),
    ]
    
    print("\nSimulating training with increasing generalization gap:\n")
    
    for epoch, train_metrics, eval_metrics in test_cases:
        print(f"=== Epoch {epoch} ===")
        alerts = monitor.check_health(train_metrics, eval_metrics, epoch)
        
        if not alerts:
            print("  ✓ All healthy")
        else:
            for alert in alerts:
                icon = "⚠️" if alert.severity == AlertSeverity.WARNING else "🚨"
                print(f"  {icon} {alert.message}")
        print()
    
    print("=== Summary ===")
    summary = monitor.get_summary()
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    
    print("\n✓ Monitor test completed!")
