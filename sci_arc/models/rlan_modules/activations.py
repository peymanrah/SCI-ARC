"""
Activation Functions for RLAN

Includes SwiGLU (Swish-Gated Linear Unit) as used in modern transformers
like LLaMA, PaLM, and TRM. SwiGLU consistently outperforms GELU/ReLU.

SwiGLU: output = (Swish(xW) ⊙ xV) W_out
where Swish(x) = x * sigmoid(x) = SiLU(x)

Reference:
- GLU Variants Improve Transformer (Shazeer, 2020)
- LLaMA: Open and Efficient Foundation Language Models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _find_multiple(a: int, b: int) -> int:
    """Round up a to the nearest multiple of b."""
    return (-(a // -b)) * b


class SwiGLU(nn.Module):
    """
    Swish-Gated Linear Unit.
    
    A gated activation function that computes:
        output = SiLU(x @ W_gate) * (x @ W_up) @ W_down
    
    Where SiLU(x) = x * sigmoid(x)
    
    This is more expressive than standard FFN with GELU/ReLU,
    at the cost of ~50% more parameters in the projection.
    
    Args:
        hidden_size: Input/output dimension
        expansion: Expansion factor for intermediate dimension
                   Default 4 gives intermediate = 4 * hidden_size * 2/3
        bias: Whether to use bias in projections
        
    Note:
        The intermediate dimension is rounded to nearest multiple of 256
        for efficiency, following LLaMA's implementation.
    """
    
    def __init__(
        self,
        hidden_size: int,
        expansion: float = 4.0,
        bias: bool = False,
    ):
        super().__init__()
        
        # Compute intermediate size (2/3 of expansion to maintain param count)
        # Then round to multiple of 256 for efficiency
        intermediate = _find_multiple(
            round(expansion * hidden_size * 2 / 3), 
            256
        )
        
        # Combined gate and up projection (more efficient than separate)
        self.gate_up_proj = nn.Linear(hidden_size, intermediate * 2, bias=bias)
        self.down_proj = nn.Linear(intermediate, hidden_size, bias=bias)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize with truncated normal, following LLaMA."""
        nn.init.trunc_normal_(self.gate_up_proj.weight, std=0.02)
        nn.init.trunc_normal_(self.down_proj.weight, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (..., hidden_size)
            
        Returns:
            Output tensor of shape (..., hidden_size)
        """
        # Split into gate and up projections
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        
        # SiLU activation on gate, then element-wise multiply
        return self.down_proj(F.silu(gate) * up)


class SwiGLUConv2d(nn.Module):
    """
    2D Convolutional SwiGLU for spatial features.
    
    Same as SwiGLU but operates on 2D feature maps.
    Useful in RecursiveSolver and other spatial modules.
    
    Args:
        channels: Number of input/output channels
        expansion: Expansion factor for intermediate channels
        kernel_size: Convolution kernel size
    """
    
    def __init__(
        self,
        channels: int,
        expansion: float = 4.0,
        kernel_size: int = 1,
    ):
        super().__init__()
        
        intermediate = _find_multiple(
            round(expansion * channels * 2 / 3),
            64  # Smaller multiple for conv (less params)
        )
        
        padding = kernel_size // 2
        
        self.gate_up_proj = nn.Conv2d(
            channels, intermediate * 2,
            kernel_size=kernel_size, padding=padding, bias=False
        )
        self.down_proj = nn.Conv2d(
            intermediate, channels,
            kernel_size=kernel_size, padding=padding, bias=False
        )
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.trunc_normal_(self.gate_up_proj.weight, std=0.02)
        nn.init.trunc_normal_(self.down_proj.weight, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Output tensor of shape (B, C, H, W)
        """
        gate, up = self.gate_up_proj(x).chunk(2, dim=1)  # Split on channel dim
        return self.down_proj(F.silu(gate) * up)


# For backwards compatibility, also provide standard FFN
class FFN(nn.Module):
    """
    Standard Feed-Forward Network with GELU.
    
    Provided for comparison/fallback. Use SwiGLU for best results.
    """
    
    def __init__(
        self,
        hidden_size: int,
        expansion: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        intermediate = round(hidden_size * expansion)
        
        self.net = nn.Sequential(
            nn.Linear(hidden_size, intermediate),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate, hidden_size),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
