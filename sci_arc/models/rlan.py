"""
RLAN: Recursive Latent Attractor Networks for ARC

The main RLAN model that orchestrates all submodules:
- GridEncoder: Embeds input grids
- DynamicSaliencyController: Discovers spatial anchors
- MultiScaleRelativeEncoding: Computes relative coordinates
- LatentCountingRegisters: Soft color counting
- SymbolicPredicateHeads: Binary predicates
- RecursiveSolver: Iterative output generation

RLAN treats reasoning as coordinate transformation relative to
dynamically discovered spatial features, rather than absolute
position-based pattern matching.

Example Usage:
    model = RLAN(hidden_dim=128, max_clues=5)
    logits = model(input_grid)  # (B, 11, H, W)
    
    # With intermediate outputs for loss computation
    outputs = model(input_grid, return_intermediates=True)
    # outputs contains: logits, centroids, attention_maps, stop_logits, predicates
"""

from dataclasses import dataclass
from typing import Dict, Optional, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from sci_arc.models.grid_encoder import GridEncoder
from sci_arc.models.rlan_modules import (
    DynamicSaliencyController,
    MultiScaleRelativeEncoding,
    LatentCountingRegisters,
    SymbolicPredicateHeads,
    RecursiveSolver,
)


@dataclass
class RLANConfig:
    """Configuration for RLAN model."""
    
    # Core dimensions
    hidden_dim: int = 128
    num_colors: int = 10
    num_classes: int = 11  # 10 colors + background
    max_grid_size: int = 30
    
    # DSC parameters
    max_clues: int = 5
    dsc_num_heads: int = 4
    
    # MSRE parameters
    msre_encoding_dim: int = 32
    msre_num_freq: int = 8
    
    # LCR parameters
    lcr_num_freq: int = 8
    lcr_num_heads: int = 4
    
    # SPH parameters
    num_predicates: int = 8
    
    # Solver parameters
    num_solver_steps: int = 6
    
    # Training parameters
    dropout: float = 0.1


class RLAN(nn.Module):
    """
    Recursive Latent Attractor Network for ARC.
    
    A neural architecture that reasons in relative coordinate spaces
    anchored to dynamically discovered spatial features.
    
    Architecture Flow:
        Input Grid → Encoder → Features
                          ↓
                    ┌─────┴─────┐
                    ↓           ↓
                   DSC         LCR
                    ↓           ↓
                  MSRE         SPH
                    ↓           ↓
                    └─────┬─────┘
                          ↓
                    RecursiveSolver
                          ↓
                    Output Logits
    """
    
    def __init__(
        self,
        hidden_dim: int = 128,
        num_colors: int = 10,
        num_classes: int = 11,
        max_grid_size: int = 30,
        max_clues: int = 5,
        num_predicates: int = 8,
        num_solver_steps: int = 6,
        dropout: float = 0.1,
        config: Optional[RLANConfig] = None,
    ):
        """
        Initialize RLAN model.
        
        Args:
            hidden_dim: Feature dimension throughout the model
            num_colors: Number of ARC colors (0-9)
            num_classes: Output classes (typically 11: 10 colors + background)
            max_grid_size: Maximum grid dimension (ARC max is 30)
            max_clues: Maximum spatial anchors to discover
            num_predicates: Number of binary predicates
            num_solver_steps: Refinement iterations
            dropout: Dropout probability
            config: Optional RLANConfig (overrides individual params)
        """
        super().__init__()
        
        # Use config if provided
        if config is not None:
            hidden_dim = config.hidden_dim
            num_colors = config.num_colors
            num_classes = config.num_classes
            max_grid_size = config.max_grid_size
            max_clues = config.max_clues
            num_predicates = config.num_predicates
            num_solver_steps = config.num_solver_steps
            dropout = config.dropout
        
        self.hidden_dim = hidden_dim
        self.num_colors = num_colors
        self.num_classes = num_classes
        self.max_clues = max_clues
        self.num_predicates = num_predicates
        self.num_solver_steps = num_solver_steps
        
        # Grid Encoder (reuse existing implementation)
        self.encoder = GridEncoder(
            hidden_dim=hidden_dim,
            num_colors=num_colors,
            max_size=max_grid_size,
            dropout=dropout,
        )
        
        # Feature projection to channel-first format
        self.feature_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        
        # Dynamic Saliency Controller
        self.dsc = DynamicSaliencyController(
            hidden_dim=hidden_dim,
            max_clues=max_clues,
            num_heads=config.dsc_num_heads if config else 4,
            dropout=dropout,
        )
        
        # Multi-Scale Relative Encoding
        self.msre = MultiScaleRelativeEncoding(
            hidden_dim=hidden_dim,
            encoding_dim=config.msre_encoding_dim if config else 32,
            max_size=max_grid_size,
            num_freq=config.msre_num_freq if config else 8,
        )
        
        # Latent Counting Registers
        self.lcr = LatentCountingRegisters(
            num_colors=num_colors,
            hidden_dim=hidden_dim,
            num_freq=config.lcr_num_freq if config else 8,
            num_heads=config.lcr_num_heads if config else 4,
            dropout=dropout,
        )
        
        # Symbolic Predicate Heads
        self.sph = SymbolicPredicateHeads(
            hidden_dim=hidden_dim,
            num_predicates=num_predicates,
            dropout=dropout,
        )
        
        # Recursive Solver
        self.solver = RecursiveSolver(
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_steps=num_solver_steps,
            num_predicates=num_predicates,
            num_colors=num_colors,
            dropout=dropout,
        )
    
    def encode(self, grid: torch.Tensor) -> torch.Tensor:
        """
        Encode input grid to features.
        
        Args:
            grid: Shape (B, H, W) input grid with color indices
            
        Returns:
            features: Shape (B, D, H, W) encoded features
        """
        # Encode grid: (B, H, W) -> (B, H, W, D)
        features = self.encoder(grid)
        
        # Project features
        features = self.feature_proj(features)  # (B, H, W, D)
        
        # Convert to channel-first: (B, H, W, D) -> (B, D, H, W)
        features = features.permute(0, 3, 1, 2)
        
        return features
    
    def forward(
        self,
        input_grid: torch.Tensor,
        temperature: float = 1.0,
        return_intermediates: bool = False,
        return_all_steps: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through RLAN.
        
        Args:
            input_grid: Shape (B, H, W) input grid with color indices 0-9
            temperature: Gumbel-softmax temperature for DSC and SPH
            return_intermediates: If True, return all intermediate outputs
            return_all_steps: If True, return predictions at all solver steps
            
        Returns:
            If return_intermediates=False:
                logits: Shape (B, num_classes, H, W)
            If return_intermediates=True:
                Dict with keys:
                    - logits: (B, num_classes, H, W)
                    - all_logits: List of (B, num_classes, H, W) if return_all_steps
                    - centroids: (B, K, 2)
                    - attention_maps: (B, K, H, W)
                    - stop_logits: (B, K)
                    - predicates: (B, P)
                    - count_embedding: (B, num_colors, D)
                    - features: (B, D, H, W)
        """
        B, H, W = input_grid.shape
        
        # 1. Encode grid
        features = self.encode(input_grid)  # (B, D, H, W)
        
        # 2. Dynamic Saliency Controller - find clue anchors
        centroids, attention_maps, stop_logits = self.dsc(
            features, temperature=temperature
        )  # (B, K, 2), (B, K, H, W), (B, K)
        
        # 3. Multi-Scale Relative Encoding - compute relative coordinates
        clue_features = self.msre(
            features, centroids, grid_sizes=None
        )  # (B, K, D, H, W)
        
        # 4. Latent Counting Registers - soft counting
        count_embedding = self.lcr(input_grid, features)  # (B, num_colors, D)
        
        # 5. Symbolic Predicate Heads - binary predicates
        predicates = self.sph(features, temperature=temperature)  # (B, P)
        
        # 6. Recursive Solver - generate output
        if return_all_steps or return_intermediates:
            all_logits = self.solver(
                clue_features=clue_features,
                count_embedding=count_embedding,
                predicates=predicates,
                input_grid=input_grid,
                attention_maps=attention_maps,
                return_all_steps=True,
            )
            logits = all_logits[-1]
        else:
            logits = self.solver(
                clue_features=clue_features,
                count_embedding=count_embedding,
                predicates=predicates,
                input_grid=input_grid,
                attention_maps=attention_maps,
                return_all_steps=False,
            )
            all_logits = None
        
        if return_intermediates:
            return {
                "logits": logits,
                "all_logits": all_logits,
                "centroids": centroids,
                "attention_maps": attention_maps,
                "stop_logits": stop_logits,
                "predicates": predicates,
                "count_embedding": count_embedding,
                "features": features,
            }
        else:
            return logits
    
    def predict(self, input_grid: torch.Tensor) -> torch.Tensor:
        """
        Make a prediction (argmax of logits).
        
        Args:
            input_grid: Shape (B, H, W) input grid
            
        Returns:
            prediction: Shape (B, H, W) predicted grid
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(input_grid, temperature=0.1)
            prediction = logits.argmax(dim=1)
        return prediction
    
    def count_parameters(self) -> Dict[str, int]:
        """Count parameters per module."""
        counts = {
            "encoder": sum(p.numel() for p in self.encoder.parameters()),
            "feature_proj": sum(p.numel() for p in self.feature_proj.parameters()),
            "dsc": sum(p.numel() for p in self.dsc.parameters()),
            "msre": sum(p.numel() for p in self.msre.parameters()),
            "lcr": sum(p.numel() for p in self.lcr.parameters()),
            "sph": sum(p.numel() for p in self.sph.parameters()),
            "solver": sum(p.numel() for p in self.solver.parameters()),
        }
        counts["total"] = sum(counts.values())
        return counts
    
    @classmethod
    def from_config(cls, config: RLANConfig) -> "RLAN":
        """Create RLAN from configuration."""
        return cls(config=config)
    
    def save_checkpoint(self, path: str, **extra_data):
        """Save model checkpoint."""
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "config": {
                "hidden_dim": self.hidden_dim,
                "num_colors": self.num_colors,
                "num_classes": self.num_classes,
                "max_clues": self.max_clues,
                "num_predicates": self.num_predicates,
                "num_solver_steps": self.num_solver_steps,
            },
            **extra_data,
        }
        torch.save(checkpoint, path)
    
    @classmethod
    def load_from_checkpoint(cls, path: str, device: str = "cpu") -> "RLAN":
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        
        model = cls(**checkpoint["config"])
        model.load_state_dict(checkpoint["model_state_dict"])
        
        return model
