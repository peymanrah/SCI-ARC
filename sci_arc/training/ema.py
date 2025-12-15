"""
Exponential Moving Average (EMA) Helper for SCI-ARC.

From TRM's models/ema.py - provides model weight averaging for stable evaluation.

EMA smooths model weights during training:
    shadow_weights = mu * shadow_weights + (1 - mu) * model_weights

This typically improves evaluation performance by reducing noise from
recent gradient updates.
"""

import copy
from typing import Dict, Optional

import torch
import torch.nn as nn


class EMAHelper:
    """
    Exponential Moving Average helper for model weights.
    
    Maintains shadow weights that are an exponential moving average
    of the model's weights. These shadow weights typically give
    better evaluation performance than the raw model weights.
    
    Usage:
        ema = EMAHelper(model, mu=0.999)
        
        for batch in dataloader:
            # Training step
            loss.backward()
            optimizer.step()
            
            # Update EMA
            ema.update(model)
        
        # Evaluation with EMA weights
        ema_model = ema.ema_copy(model)
        ema_model.eval()
        output = ema_model(input)
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        mu: float = 0.999,
        device: Optional[torch.device] = None
    ):
        """
        Initialize EMA with a copy of the model's weights.
        
        Args:
            model: The model to track
            mu: EMA decay rate (higher = slower update, more smoothing)
                Typical values: 0.999, 0.9999
            device: Device to store shadow weights on (defaults to model's device)
        """
        self.mu = mu
        self.device = device
        
        # Store a copy of the model's state dict
        # Only track parameters (not buffers like batch norm running stats)
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
                if device is not None:
                    self.shadow[name] = self.shadow[name].to(device)
    
    def update(self, model: nn.Module) -> None:
        """
        Update shadow weights with current model weights.
        
        shadow = mu * shadow + (1 - mu) * model
        
        Using torch.lerp for efficiency:
            shadow.lerp_(model, 1 - mu)
        
        Args:
            model: Current model with updated weights
        """
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.shadow and param.requires_grad:
                    # lerp_: shadow = shadow + (1 - mu) * (model - shadow)
                    #       = mu * shadow + (1 - mu) * model
                    self.shadow[name].lerp_(param.data, 1 - self.mu)
    
    def ema_copy(self, model: nn.Module) -> nn.Module:
        """
        Create a copy of the model with EMA weights.
        
        This is useful for evaluation - you want to evaluate with
        EMA weights while training continues with regular weights.
        
        Args:
            model: Model architecture to copy
        
        Returns:
            New model instance with EMA weights loaded
        """
        # Create a deep copy of the model
        ema_model = copy.deepcopy(model)
        
        # Load shadow weights
        ema_state = ema_model.state_dict()
        for name in self.shadow:
            if name in ema_state:
                ema_state[name] = self.shadow[name]
        
        ema_model.load_state_dict(ema_state)
        return ema_model
    
    def apply_shadow(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """
        Apply shadow weights to model and return original weights.
        
        Use this for in-place EMA evaluation:
            original = ema.apply_shadow(model)
            output = model(input)  # Uses EMA weights
            ema.restore(model, original)
        
        Args:
            model: Model to apply shadow weights to
        
        Returns:
            Dictionary of original weights (for restoration)
        """
        original = {}
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.shadow:
                    original[name] = param.data.clone()
                    param.data.copy_(self.shadow[name])
        return original
    
    def restore(self, model: nn.Module, original: Dict[str, torch.Tensor]) -> None:
        """
        Restore original weights after EMA evaluation.
        
        Args:
            model: Model to restore
            original: Original weights from apply_shadow()
        """
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in original:
                    param.data.copy_(original[name])
    
    def state_dict(self) -> Dict:
        """
        Get EMA state for checkpointing.
        
        Returns:
            Dictionary containing mu and shadow weights
        """
        return {
            'mu': self.mu,
            'shadow': self.shadow,
        }
    
    def load_state_dict(self, state_dict: Dict) -> None:
        """
        Load EMA state from checkpoint.
        
        Args:
            state_dict: State dict from state_dict()
        """
        self.mu = state_dict['mu']
        self.shadow = state_dict['shadow']


class EMAWrapper(nn.Module):
    """
    Wrapper that maintains both regular and EMA versions of a model.
    
    Convenient for automatic EMA updates during training.
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        mu: float = 0.999,
        update_after_step: int = 0
    ):
        """
        Args:
            model: Model to wrap
            mu: EMA decay rate
            update_after_step: Start updating EMA after this many steps
        """
        super().__init__()
        self.model = model
        self.ema = EMAHelper(model, mu=mu)
        self.update_after_step = update_after_step
        self.step = 0
    
    def forward(self, *args, **kwargs):
        """Forward through the regular model."""
        return self.model(*args, **kwargs)
    
    def update_ema(self) -> None:
        """Call after each training step to update EMA."""
        self.step += 1
        if self.step > self.update_after_step:
            self.ema.update(self.model)
    
    def ema_model(self) -> nn.Module:
        """Get model with EMA weights for evaluation."""
        return self.ema.ema_copy(self.model)
