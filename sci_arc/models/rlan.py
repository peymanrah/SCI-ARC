"""
RLAN: Recursive Latent Attractor Networks for ARC

The main RLAN model that orchestrates all submodules:
- GridEncoder: Embeds input grids
- ContextEncoder: Encodes training examples to understand task (CRITICAL!)
- DynamicSaliencyController: Discovers spatial anchors
- MultiScaleRelativeEncoding: Computes relative coordinates
- LatentCountingRegisters: Soft color counting
- SymbolicPredicateHeads: Binary predicates
- RecursiveSolver: Iterative output generation

RLAN treats reasoning as coordinate transformation relative to
dynamically discovered spatial features, conditioned on the task
context learned from training examples.

Example Usage:
    model = RLAN(hidden_dim=128, max_clues=5)
    
    # With training context (recommended for ARC)
    logits = model(
        test_input,
        train_inputs=train_input_grids,
        train_outputs=train_output_grids,
    )
    
    # Or for legacy single-grid mode
    logits = model(input_grid)
    
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
    ContextEncoder,
    ContextInjector,
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
    use_act: bool = False  # Adaptive Computation Time
    use_solver_feedback: bool = False  # Use prediction feedback in solver (disabled - argmax breaks gradients)
    
    # Training parameters
    dropout: float = 0.1
    
    # =============================================================
    # MODULE ABLATION FLAGS
    # =============================================================
    # Enable/disable individual modules for ablation studies
    # When disabled, modules output zeros or identity transformations
    #
    # RLAN Novelty Analysis:
    #   - DSC + MSRE: Core novelty (spatial anchoring + relative coords)
    #   - ContextEncoder: Critical but has bottleneck issue
    #   - LCR + SPH: Auxiliary (nice-to-have, not core)
    # =============================================================
    use_context_encoder: bool = True   # Encode training pairs (4.2M params)
    use_dsc: bool = True               # Dynamic Saliency Controller (266K)
    use_msre: bool = True              # Multi-Scale Relative Encoding (109K)
    use_lcr: bool = True               # Latent Counting Registers (403K)
    use_sph: bool = True               # Symbolic Predicate Heads (232K)
    
    # Positional encoding option
    use_learned_pos: bool = False       # Use learned pos embed vs sinusoidal (default)


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
        use_act: bool = False,
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
            use_act: Whether to use Adaptive Computation Time
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
            use_act = config.use_act
            dropout = config.dropout
        
        self.hidden_dim = hidden_dim
        self.num_colors = num_colors
        self.num_classes = num_classes
        self.max_clues = max_clues
        self.num_predicates = num_predicates
        self.num_solver_steps = num_solver_steps
        self.use_act = use_act
        self.max_grid_size = max_grid_size
        
        # Module ablation flags (default to True if no config)
        self.use_context_encoder = config.use_context_encoder if config else True
        self.use_dsc = config.use_dsc if config else True
        self.use_msre = config.use_msre if config else True
        self.use_lcr = config.use_lcr if config else True
        self.use_sph = config.use_sph if config else True
        self.use_learned_pos = config.use_learned_pos if config else False
        
        # Grid Encoder (reuse existing implementation) - ALWAYS REQUIRED
        self.encoder = GridEncoder(
            hidden_dim=hidden_dim,
            num_colors=num_colors,
            max_size=max_grid_size,
            dropout=dropout,
            use_learned_pos=self.use_learned_pos,  # Learned vs sinusoidal positional encoding
        )
        
        # Feature projection to channel-first format - ALWAYS REQUIRED
        self.feature_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        
        # Context Encoder - learns from training examples (OPTIONAL)
        if self.use_context_encoder:
            self.context_encoder = ContextEncoder(
                hidden_dim=hidden_dim,
                num_colors=num_colors,
                max_size=max_grid_size,
                max_pairs=5,  # ARC has 2-5 training pairs
                num_heads=config.dsc_num_heads if config else 4,
                dropout=dropout,
            )
            self.context_injector = ContextInjector(hidden_dim=hidden_dim)
        else:
            self.context_encoder = None
            self.context_injector = None
        
        # Dynamic Saliency Controller (OPTIONAL - but core novelty)
        if self.use_dsc:
            self.dsc = DynamicSaliencyController(
                hidden_dim=hidden_dim,
                max_clues=max_clues,
                num_heads=config.dsc_num_heads if config else 4,
                dropout=dropout,
            )
        else:
            self.dsc = None
        
        # Multi-Scale Relative Encoding (OPTIONAL - depends on DSC)
        if self.use_msre and self.use_dsc:
            self.msre = MultiScaleRelativeEncoding(
                hidden_dim=hidden_dim,
                encoding_dim=config.msre_encoding_dim if config else 32,
                max_size=max_grid_size,
                num_freq=config.msre_num_freq if config else 8,
            )
        else:
            self.msre = None
        
        # Latent Counting Registers (OPTIONAL)
        if self.use_lcr:
            self.lcr = LatentCountingRegisters(
                num_colors=num_colors,
                hidden_dim=hidden_dim,
                num_freq=config.lcr_num_freq if config else 8,
                num_heads=config.lcr_num_heads if config else 4,
                dropout=dropout,
            )
        else:
            self.lcr = None
        
        # Symbolic Predicate Heads (OPTIONAL)
        if self.use_sph:
            self.sph = SymbolicPredicateHeads(
                hidden_dim=hidden_dim,
                num_predicates=num_predicates,
                dropout=dropout,
            )
        else:
            self.sph = None
        
        # Recursive Solver - ALWAYS REQUIRED (core output generation)
        # Pass ablation flags so solver can skip unused components
        self.solver = RecursiveSolver(
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_steps=num_solver_steps,
            num_predicates=num_predicates,
            num_colors=num_colors,
            dropout=dropout,
            use_act=self.use_act,  # Enable Adaptive Computation Time
            use_lcr=self.use_lcr,  # Skip count injection if LCR disabled
            use_sph=self.use_sph,  # Skip predicate gating if SPH disabled
            use_feedback=config.use_solver_feedback,  # Disabled by default (argmax breaks gradients)
        )
        
        # Print module configuration
        enabled = []
        disabled = []
        for name, flag in [
            ('ContextEncoder', self.use_context_encoder),
            ('DSC', self.use_dsc),
            ('MSRE', self.use_msre),
            ('LCR', self.use_lcr),
            ('SPH', self.use_sph),
            ('ACT', self.use_act),
        ]:
            (enabled if flag else disabled).append(name)
        
        if disabled:
            print(f"RLAN Module Config: Enabled=[{', '.join(enabled)}], Disabled=[{', '.join(disabled)}]")
    
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
        train_inputs: Optional[torch.Tensor] = None,
        train_outputs: Optional[torch.Tensor] = None,
        pair_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        return_intermediates: bool = False,
        return_all_steps: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through RLAN.
        
        Args:
            input_grid: Shape (B, H, W) input grid with color indices 0-9
            train_inputs: Shape (B, N, H, W) training input grids (RECOMMENDED!)
            train_outputs: Shape (B, N, H, W) training output grids
            pair_mask: Shape (B, N) boolean mask for valid pairs
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
                    - context: (B, D) if train_inputs provided
        """
        B, H, W = input_grid.shape
        
        # 1. Encode grid - ALWAYS REQUIRED
        features = self.encode(input_grid)  # (B, D, H, W)
        
        # 2. Encode training context if provided and enabled
        context = None
        if self.use_context_encoder and self.context_encoder is not None:
            if train_inputs is not None and train_outputs is not None:
                context = self.context_encoder(
                    train_inputs, train_outputs, pair_mask
                )  # (B, D)
                # Inject context into features via FiLM
                features = self.context_injector(features, context)
        
        # 3. Dynamic Saliency Controller - find clue anchors (if enabled)
        if self.use_dsc and self.dsc is not None:
            centroids, attention_maps, stop_logits = self.dsc(
                features, temperature=temperature
            )  # (B, K, 2), (B, K, H, W), (B, K)
        else:
            # Default: single centered anchor, uniform attention
            K = self.max_clues
            centroids = torch.zeros(B, K, 2, device=features.device)
            centroids[:, :, 0] = H / 2  # row center
            centroids[:, :, 1] = W / 2  # col center
            attention_maps = torch.ones(B, K, H, W, device=features.device) / (H * W)
            stop_logits = torch.zeros(B, K, device=features.device)
        
        # 4. Multi-Scale Relative Encoding - compute relative coordinates (if enabled)
        if self.use_msre and self.msre is not None:
            clue_features = self.msre(
                features, centroids, grid_sizes=None
            )  # (B, K, D, H, W)
        else:
            # Default: just broadcast features across K clues
            clue_features = features.unsqueeze(1).expand(-1, self.max_clues, -1, -1, -1)
        
        # 5. Latent Counting Registers - soft counting (if enabled)
        if self.use_lcr and self.lcr is not None:
            count_embedding = self.lcr(input_grid, features)  # (B, num_colors, D)
        else:
            # Default: zeros (RecursiveSolver will skip count injection when use_lcr=False)
            # Using zeros instead of empty() to avoid inf/nan garbage values
            count_embedding = torch.zeros(
                B, self.num_colors, self.hidden_dim, device=features.device
            )
        
        # 6. Symbolic Predicate Heads - binary predicates (if enabled)
        if self.use_sph and self.sph is not None:
            predicates = self.sph(features, temperature=temperature)  # (B, P)
        else:
            # Default: zeros (RecursiveSolver will skip predicate gating when use_sph=False)
            # Using zeros instead of empty() to avoid inf/nan garbage values
            predicates = torch.zeros(B, self.num_predicates, device=features.device)
        
        # 7. Recursive Solver - generate output
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
            result = {
                "logits": logits,
                "all_logits": all_logits,
                "centroids": centroids,
                "attention_maps": attention_maps,
                "stop_logits": stop_logits,
                "predicates": predicates,
                "count_embedding": count_embedding,
                "features": features,
            }
            if context is not None:
                result["context"] = context
            return result
        else:
            return logits
    
    def predict(
        self,
        input_grid: torch.Tensor,
        train_inputs: Optional[torch.Tensor] = None,
        train_outputs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Make a prediction (argmax of logits).
        
        Args:
            input_grid: Shape (B, H, W) input grid
            train_inputs: Optional (B, N, H, W) training inputs for context
            train_outputs: Optional (B, N, H, W) training outputs for context
            
        Returns:
            prediction: Shape (B, H, W) predicted grid
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(
                input_grid,
                train_inputs=train_inputs,
                train_outputs=train_outputs,
                temperature=0.1
            )
            prediction = logits.argmax(dim=1)
        return prediction
    
    def count_parameters(self) -> Dict[str, int]:
        """Count parameters per module (handles disabled modules)."""
        counts = {
            "encoder": sum(p.numel() for p in self.encoder.parameters()),
            "feature_proj": sum(p.numel() for p in self.feature_proj.parameters()),
            "context_encoder": sum(p.numel() for p in self.context_encoder.parameters()) if self.context_encoder else 0,
            "context_injector": sum(p.numel() for p in self.context_injector.parameters()) if self.context_injector else 0,
            "dsc": sum(p.numel() for p in self.dsc.parameters()) if self.dsc else 0,
            "msre": sum(p.numel() for p in self.msre.parameters()) if self.msre else 0,
            "lcr": sum(p.numel() for p in self.lcr.parameters()) if self.lcr else 0,
            "sph": sum(p.numel() for p in self.sph.parameters()) if self.sph else 0,
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
                "use_act": self.use_act,
                "max_grid_size": self.max_grid_size,
                # Save ablation flags
                "use_context_encoder": self.use_context_encoder,
                "use_dsc": self.use_dsc,
                "use_msre": self.use_msre,
                "use_lcr": self.use_lcr,
                "use_sph": self.use_sph,
            },
            **extra_data,
        }
        torch.save(checkpoint, path)
    
    @classmethod
    def load_from_checkpoint(cls, path: str, device: str = "cpu") -> "RLAN":
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        
        model = cls(**checkpoint["config"])
        model.load_state_dict(checkpoint["model_state_dict"])
        
        return model
