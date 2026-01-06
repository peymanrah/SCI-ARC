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
from typing import Dict, Optional, Union, List, Tuple, Any

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
    CrossAttentionInjector,
    HyperLoRA,
    HyperLoRAConfig,
    HierarchicalPrimitiveMemory,
    HPMConfig,
    DynamicMemoryBuffer,
)
from sci_arc.models.rlan_modules.acw import AugmentedConfidenceWeighting, apply_augmentation


@dataclass
class RLANConfig:
    """Configuration for RLAN model.
    
    IMPORTANT: Each field should only be defined ONCE to avoid dataclass override issues.
    """
    
    # Core dimensions
    hidden_dim: int = 128
    num_colors: int = 10
    num_classes: int = 10  # 10 colors (0-9), no boundary markers needed
    max_grid_size: int = 30
    
    # Module flags
    use_context_encoder: bool = True   # Encode training pairs (4.2M params)
    use_dsc: bool = True               # Dynamic Saliency Controller (266K)
    use_msre: bool = True              # Multi-Scale Relative Encoding (109K)
    use_lcr: bool = False              # Latent Counting Registers (403K)
    use_sph: bool = False              # Symbolic Predicate Heads (232K)
    use_act: bool = False              # Adaptive Computation Time
    
    # Context settings
    use_cross_attention_context: bool = False
    spatial_downsample: int = 1
    
    # DSC parameters
    max_clues: int = 5
    dsc_num_heads: int = 4
    dsc_use_complexity_signals: bool = True  # Jan 2026: Task-aware stop prediction (ENABLED by default)
    
    # MSRE parameters
    msre_encoding_dim: int = 32
    msre_num_freq: int = 8
    
    # LCR parameters
    lcr_num_freq: int = 8
    lcr_num_heads: int = 4
    
    # SPH parameters
    num_predicates: int = 16
    
    # Solver parameters
    num_solver_steps: int = 6
    use_solver_feedback: bool = False  # Use prediction feedback in solver (disabled - argmax breaks gradients)
    use_solver_context: bool = True  # Phase 2.5: Solver cross-attention to support set
    solver_context_heads: int = 4  # Number of attention heads for solver cross-attention
    use_best_step_selection: bool = False  # Phase 3: Select best step by loss (train) or entropy (eval)
    
    # HyperLoRA settings (meta-learning weight adaptation)
    use_hyperlora: bool = False
    hyperlora_rank: int = 8
    hyperlora_scaling: float = 1.0
    hyperlora_dropout: float = 0.0
    hyperlora_init_scale: float = 0.1  # FIXED: Was 0.01, increased for stronger meta-learning signal
    hyperlora_max_norm: float = 1.0    # STABILITY FIX: Clamp LoRA delta L2 norm (was 3.0, caused collapse)
    
    # HPM (Hierarchical Primitive Memory v2) settings
    use_hpm: bool = False                    # Enable HPM for continual learning
    hpm_top_k: int = 2                       # Number of banks to route to per sample
    hpm_balance_weight: float = 0.01         # Load balancing loss weight
    hpm_primitives_per_bank: int = 16        # Number of primitives per static bank
    hpm_levels_per_bank: int = 2             # Hierarchical levels per bank
    hpm_use_cross_attention: bool = True     # Use cross-attention aggregation
    hpm_memory_size: int = 10000             # Max entries in dynamic banks
    hpm_retrieval_k: int = 5                 # Number of neighbors to retrieve
    hpm_use_compositional_bank: bool = True  # Static: Compositional transforms
    hpm_use_pattern_bank: bool = True        # Static: Holistic patterns
    hpm_use_relational_bank: bool = True     # Static: Spatial relationships
    hpm_use_concept_bank: bool = False       # Static: Domain knowledge
    hpm_use_procedural_bank: bool = False    # Dynamic: HyperLoRA codes
    hpm_use_instance_bank: bool = False      # Dynamic: ContextEncoder cache
    
    # HPM Solver-Context Coupling (Jan 2026)
    hpm_solver_context_enabled: bool = True      # Enable HPM→solver cross-attention coupling
    hpm_solver_context_max_tokens: int = 8       # Max HPM memory tokens to inject per step
    hpm_solver_context_gate_init: float = 0.0    # Initial gate value (0.0 = no influence until warmup)
    
    # Training parameters
    dropout: float = 0.1
    gradient_checkpointing: bool = False  # Enable to reduce memory ~40%, trade compute for memory
    
    # Positional encoding option
    use_learned_pos: bool = False  # Use learned pos embed vs sinusoidal (default)


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
        num_classes: int = 10,
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
            num_classes: Output classes (10 colors, no boundary markers)
            max_grid_size: Maximum grid dimension (ARC max is 30)
            max_clues: Maximum spatial anchors to discover
            num_predicates: Number of binary predicates
            num_solver_steps: Refinement iterations
            use_act: Whether to use Adaptive Computation Time
            dropout: Dropout probability
            config: Optional RLANConfig (overrides individual params)
        """
        super().__init__()
        
        # BUG FIX #1: Detect accidental positional arg usage
        # RLAN(config) would silently fail because config becomes hidden_dim
        if isinstance(hidden_dim, RLANConfig):
            raise TypeError(
                "RLANConfig passed as positional argument. Use keyword: RLAN(config=config)"
            )
        
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
        self.dropout = dropout  # BUG FIX #7: Store for checkpoint saving
        
        # Memory optimization flags (set externally by training script)
        self.use_gradient_checkpointing = False  # Enable for activation memory savings
        
        # Module ablation flags (default to True if no config)
        self.use_context_encoder = config.use_context_encoder if config else True
        self.use_dsc = config.use_dsc if config else True
        self.use_msre = config.use_msre if config else True
        self.use_lcr = config.use_lcr if config else True
        self.use_sph = config.use_sph if config else True
        self.use_learned_pos = config.use_learned_pos if config else False
        
        # Structure/Content Disentanglement for SCL and orthogonality losses
        # z_struct: captures transformation rule (should be same across demo pairs)
        # z_content: captures grid content/appearance (varies across pairs)
        self.structure_projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.content_projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
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
        # Use cross-attention context injection (experimental - high memory, can destabilize training)
        # Default: FiLM conditioning (stable, proven to work)
        use_cross_attn_context = config.use_cross_attention_context if config and hasattr(config, 'use_cross_attention_context') else False
        spatial_downsample = config.spatial_downsample if config and hasattr(config, 'spatial_downsample') else 1
        
        # CRITICAL FIX: Need spatial features if ANY of:
        # 1. CrossAttentionInjector is used (use_cross_attention_context=True)
        # 2. Solver cross-attention to support set is used (use_solver_context=True)
        # 3. HyperLoRA is used (use_hyperlora=True) - requires (B, N, D, H, W) for LOO training
        # All modes need (B, N, D, H, W) spatial features from ContextEncoder
        use_solver_context_flag = config.use_solver_context if config else True
        use_hyperlora_flag = config.use_hyperlora if config else False
        needs_spatial_features = use_cross_attn_context or use_solver_context_flag or use_hyperlora_flag
        
        if self.use_context_encoder:
            self.context_encoder = ContextEncoder(
                hidden_dim=hidden_dim,
                num_colors=num_colors,
                max_size=max_grid_size,
                max_pairs=5,  # ARC has 2-5 training pairs
                num_heads=config.dsc_num_heads if config else 4,
                dropout=dropout,
                use_spatial_features=needs_spatial_features,  # Spatial features for cross-attn OR solver context
                spatial_downsample=spatial_downsample,        # Downsample support features
            )
            if use_cross_attn_context:
                # Cross-attention: attends to ALL support pixels (high memory)
                self.context_injector = CrossAttentionInjector(
                    hidden_dim=hidden_dim,
                    num_heads=config.dsc_num_heads if config else 4,
                    dropout=dropout,
                )
                # FiLM fallback for staged training (used in early epochs)
                # Cross-attention uses Q/K/V projections that are randomly initialized
                # and inject noise. FiLM (scale/shift) is more stable for early training.
                self.film_fallback_injector = ContextInjector(hidden_dim=hidden_dim)
            else:
                # FiLM: stable, compresses context to vector (default)
                self.context_injector = ContextInjector(hidden_dim=hidden_dim)
                self.film_fallback_injector = None  # Not needed, already using FiLM
        else:
            self.context_encoder = None
            self.context_injector = None
        
        # Dynamic Saliency Controller (OPTIONAL - but core novelty)
        if self.use_dsc:
            # Jan 2026: use_complexity_signals=False for backward compat with existing checkpoints
            # Set to True when training from scratch to enable task-aware stop prediction
            self.dsc = DynamicSaliencyController(
                hidden_dim=hidden_dim,
                max_clues=max_clues,
                num_heads=config.dsc_num_heads if config else 4,
                dropout=dropout,
                context_dim=hidden_dim,  # DSC uses context from task encoder
                use_complexity_signals=getattr(config, 'dsc_use_complexity_signals', False) if config else False,
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
                use_per_clue_mode=self.use_dsc,  # BUG FIX #4: Only create cross-attn if DSC disabled
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
        # Phase 2.5: Pass support set cross-attention flag
        # Phase 3: Best-step selection for handling over-iteration
        self.use_solver_context = config.use_solver_context if config else True
        self.use_best_step_selection = config.use_best_step_selection if config else False
        self.gradient_checkpointing = config.gradient_checkpointing if config else False
        # HPM solver-context coupling: only enable if HPM is also enabled
        hpm_solver_context_enabled = (
            (config.hpm_solver_context_enabled if config else True) and
            (config.use_hpm if config else False)
        )
        self.hpm_solver_context_enabled = hpm_solver_context_enabled
        
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
            use_feedback=config.use_solver_feedback if config else False,  # Disabled by default (argmax breaks gradients)
            use_solver_context=self.use_solver_context,  # Phase 2.5: Solver cross-attention to support set
            num_context_heads=config.solver_context_heads if config else 4,
            use_dsc=self.use_dsc,  # BUG FIX #5: DSC provides per-clue counts, bypassing count_proj
            gradient_checkpointing=self.gradient_checkpointing,  # Memory optimization
            # HPM Solver-Context Coupling (Jan 2026)
            hpm_solver_context_enabled=hpm_solver_context_enabled,
            hpm_solver_context_max_tokens=config.hpm_solver_context_max_tokens if config else 8,
            hpm_solver_context_gate_init=config.hpm_solver_context_gate_init if config else 0.0,
        )
        
        # HyperLoRA for meta-learning weight adaptation (optional)
        self.use_hyperlora = config.use_hyperlora if config else False
        if self.use_hyperlora:
            # BUG FIX #7: Store HyperLoRA config values for checkpoint saving
            self._hyperlora_rank = config.hyperlora_rank if config else 8
            self._hyperlora_scaling = config.hyperlora_scaling if config else 1.0
            self._hyperlora_dropout = config.hyperlora_dropout if config else 0.0
            self._hyperlora_init_scale = config.hyperlora_init_scale if config else 0.01
            # STABILITY FIX (Jan 2026): Pass lora_max_norm from YAML to prevent training collapse
            self._hyperlora_max_norm = config.hyperlora_max_norm if config and hasattr(config, 'hyperlora_max_norm') else 1.0
            hyperlora_config = HyperLoRAConfig(
                hidden_dim=hidden_dim,
                context_dim=hidden_dim,
                rank=self._hyperlora_rank,
                scaling=self._hyperlora_scaling,
                dropout=self._hyperlora_dropout,
                init_scale=self._hyperlora_init_scale,
                lora_max_norm=self._hyperlora_max_norm,  # Stability fix: clamp LoRA delta norm
            )
            self.hyper_lora = HyperLoRA(config=hyperlora_config)
        else:
            self.hyper_lora = None
        
        # HPM (Hierarchical Primitive Memory v2) for continual learning (optional)
        self.use_hpm = config.use_hpm if config else False
        if self.use_hpm:
            # Create HPM config from model config
            hpm_config = HPMConfig(
                d_model=hidden_dim,
                top_k=config.hpm_top_k if config else 2,
                balance_loss_weight=config.hpm_balance_weight if config else 0.01,
                primitives_per_bank=config.hpm_primitives_per_bank if config else 16,
                n_levels_per_bank=config.hpm_levels_per_bank if config else 2,
                use_cross_attention=config.hpm_use_cross_attention if config else True,
                max_dynamic_buffer_size=config.hpm_memory_size if config else 10000,
                dynamic_retrieval_k=config.hpm_retrieval_k if config else 5,
                use_compositional_bank=config.hpm_use_compositional_bank if config else True,
                use_pattern_bank=config.hpm_use_pattern_bank if config else True,
                use_relational_bank=config.hpm_use_relational_bank if config else True,
                use_concept_bank=config.hpm_use_concept_bank if config else False,
                use_procedural_bank=config.hpm_use_procedural_bank if config else False,
                use_instance_bank=config.hpm_use_instance_bank if config else False,
            )
            self.hpm = HierarchicalPrimitiveMemory(hpm_config)
            self._hpm_config = hpm_config  # Store for checkpoint saving
            
            # Dynamic buffers for Instance and Procedural banks
            if hpm_config.use_instance_bank:
                self.hpm_instance_buffer = DynamicMemoryBuffer(
                    d_model=hidden_dim,
                    max_size=hpm_config.max_dynamic_buffer_size,
                    use_faiss=True,
                )
            else:
                self.hpm_instance_buffer = None
            
            if hpm_config.use_procedural_bank:
                self.hpm_procedural_buffer = DynamicMemoryBuffer(
                    d_model=hidden_dim,
                    max_size=hpm_config.max_dynamic_buffer_size,
                    use_faiss=True,
                )
            else:
                self.hpm_procedural_buffer = None
        else:
            self.hpm = None
            self.hpm_instance_buffer = None
            self.hpm_procedural_buffer = None
        
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
            ('SolverContext', self.use_solver_context),  # Phase 2.5
            ('HyperLoRA', self.use_hyperlora),  # Meta-learning
            ('HPM', self.use_hpm),  # Hierarchical Primitive Memory
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
        num_steps_override: Optional[int] = None,  # Override solver steps at inference
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
            num_steps_override: If provided, run this many solver steps instead of default.
                               Useful for inference-time experimentation (e.g., train with 6, infer with 10).
            
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
        
        # Compute valid mask and grid sizes from input for downstream modules
        # This allows MSRE to do scale-invariant encoding and DSC to avoid anchoring on padding
        valid_mask = self.encoder.get_valid_mask(input_grid)  # (B, H, W)
        grid_sizes = self.encoder.get_grid_sizes(input_grid)  # (B, 2)
        
        # 2. Encode training context if provided and enabled
        context = None  # (B, D) for FiLM mode
        dsc_task_context = None  # (B, D) for DSC stop predictor (ALWAYS computed when context available)
        support_features = None  # (B, N, D, H, W) for cross-attention mode (for solver)
        if self.use_context_encoder and self.context_encoder is not None:
            if train_inputs is not None and train_outputs is not None:
                context_output = self.context_encoder(
                    train_inputs, train_outputs, pair_mask
                )
                # ContextEncoder returns (B, D) for FiLM or (B, N, D, H, W) for cross-attention
                # depending on use_spatial_features flag
                if self.context_encoder.use_spatial_features:
                    # Spatial features mode: (B, N, D, H, W)
                    support_features = context_output  # Keep for solver cross-attention
                    
                    # CRITICAL FIX: Always compute a pooled task context for DSC
                    # Even when cross-attention injection is used for features, DSC's stop
                    # predictor needs a task embedding to make task-dependent stop decisions.
                    # Without this, DSC gets task_context=None -> zero vector -> frozen stop probs.
                    dsc_task_context = self.pool_context_from_support(context_output)  # (B, D)
                    
                    # STAGED CROSS-ATTENTION: Check if cross-attention is active
                    # During early epochs, cross-attention injects random noise from untrained
                    # Q/K/V projections. FiLM pooling is more stable.
                    cross_attention_active = getattr(self, 'cross_attention_active', True)
                    
                    # Check what kind of injector we have
                    if isinstance(self.context_injector, CrossAttentionInjector) and cross_attention_active:
                        # CrossAttentionInjector: pass spatial features directly
                        features = self.context_injector(features, context_output)
                    else:
                        # FiLM fallback: pool spatial features to context vector
                        # Used when: 1) use_solver_context=True but use_cross_attention_context=False
                        #            2) Early epochs when cross_attention_active=False
                        context = dsc_task_context  # Reuse already-pooled context
                        # Use dedicated fallback injector if available, else main injector
                        fallback_injector = getattr(self, 'film_fallback_injector', None) or self.context_injector
                        if fallback_injector is not None:
                            features = fallback_injector(features, context)
                else:
                    # FiLM mode: compressed context vector
                    context = context_output  # (B, D)
                    dsc_task_context = context  # Same as FiLM context
                    features = self.context_injector(features, context)
        
        # 2.5. Hierarchical Primitive Memory (HPM v2) - enhance context with memory (if enabled)
        # MEMORY EFFICIENT: Gated residual starts at 0, Top-K routing only queries k banks
        hpm_routing_weights = None
        hpm_retrieval_stats = {}  # Track retrieval stats for debugging
        hpm_memory_tokens = None  # Jan 2026: For HPM solver-context coupling
        if self.use_hpm and self.hpm is not None:
            # Get context vector for HPM query (pool if spatial features)
            if support_features is not None:
                # Pool spatial features: (B, N, D, H, W) -> (B, D)
                z_context_flat = support_features.mean(dim=(1, 3, 4))
            elif context is not None:
                z_context_flat = context
            else:
                # No context available - skip HPM
                z_context_flat = None
            
            if z_context_flat is not None:
                # Prepare dynamic buffers if available
                # FIX: Check hpm_memory_enabled to support STATIC-ONLY mode from inference staging
                # When hpm_memory_enabled=False, we skip dynamic buffer retrieval but still use static banks
                hpm_memory_enabled = getattr(self, 'hpm_memory_enabled', True)
                dynamic_buffers = {}
                
                if hpm_memory_enabled and self.hpm_instance_buffer is not None and len(self.hpm_instance_buffer) > 0:
                    keys, values, stats = self.hpm_instance_buffer.retrieve_batch(
                        z_context_flat, k=self._hpm_config.dynamic_retrieval_k
                    )
                    if keys is not None:
                        dynamic_buffers['INSTANCE'] = (keys, values)  # [B, k, D]
                        # Collect values for solver-context coupling (Jan 2026)
                        hpm_memory_tokens = values  # (B, k, D)
                    if stats:
                        hpm_retrieval_stats['instance'] = stats
                
                if hpm_memory_enabled and self.hpm_procedural_buffer is not None and len(self.hpm_procedural_buffer) > 0:
                    keys, values, stats = self.hpm_procedural_buffer.retrieve_batch(
                        z_context_flat, k=self._hpm_config.dynamic_retrieval_k
                    )
                    if keys is not None:
                        dynamic_buffers['PROCEDURAL'] = (keys, values)  # [B, k, D]
                        # Concatenate procedural values if instance already exists
                        if hpm_memory_tokens is not None:
                            hpm_memory_tokens = torch.cat([hpm_memory_tokens, values], dim=1)
                        else:
                            hpm_memory_tokens = values
                    if stats:
                        hpm_retrieval_stats['procedural'] = stats
                
                # Enhance context with HPM
                z_context_enhanced, hpm_routing_weights = self.hpm(
                    z_context_flat,
                    dynamic_buffers=dynamic_buffers if dynamic_buffers else None,
                    return_routing=True,
                )
                
                # Inject enhanced context back into features
                # Use additive injection to preserve original features
                # z_context_enhanced is (B, D), features is (B, D, H, W)
                hpm_contribution = z_context_enhanced.unsqueeze(-1).unsqueeze(-1)  # (B, D, 1, 1)
                features = features + hpm_contribution  # Broadcast addition
        
        # 3. Dynamic Saliency Controller - find clue anchors (if enabled)
        if self.use_dsc and self.dsc is not None:
            # CRITICAL: Use dsc_task_context (always pooled) instead of context (may be None in cross-attn mode)
            # ENHANCED (Jan 2026): Pass input_grid for FG bias + task complexity signals
            centroids, attention_maps, stop_logits = self.dsc(
                features, temperature=temperature, mask=valid_mask, task_context=dsc_task_context,
                input_grid=input_grid
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
                features, centroids, grid_sizes=grid_sizes
            )  # (B, K, D, H, W)
        else:
            # Default: just broadcast features across K clues
            clue_features = features.unsqueeze(1).expand(-1, self.max_clues, -1, -1, -1)
        
        # 5. Latent Counting Registers - soft counting (if enabled)
        # Paper: c_t = sum_{i,j} M_t(i,j) * OneHot(X_{i,j}) (per-clue, attention-weighted)
        if self.use_lcr and self.lcr is not None:
            # Pass attention_maps for per-clue counting (paper formulation)
            count_embedding = self.lcr(
                input_grid, features, 
                mask=valid_mask,
                attention_maps=attention_maps  # Per-clue counting from DSC attention
            )  # (B, K, D) per-clue, or (B, num_colors, D) if no attention_maps
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
        
        # 6.5. HyperLoRA - compute task-specific weight adaptations (if enabled)
        # CRITICAL FIX: This must be called BEFORE solver so LoRA weights are used!
        # STAGING: hyperlora_active flag allows disabling LoRA during early epochs
        # while keeping the module trainable (it still gets gradients via LOO/Equiv later)
        lora_deltas = None
        hyperlora_active = getattr(self, 'hyperlora_active', True)  # Default: active
        if self.use_hyperlora and self.hyper_lora is not None and hyperlora_active:
            if support_features is not None:
                # HyperLoRA expects (B, N, D, H, W) and returns weight deltas
                lora_deltas = self.hyper_lora(support_features)

                # Optional inference-time LOO sanity check (Jan 2026)
                # If predicted LoRA deltas cannot solve support pairs, mask them out.
                # This avoids silent degradation when HyperLoRA is loaded but misbehaving.
                if (not self.training) and getattr(self, 'loo_sanity_check_enabled', False):
                    threshold = float(getattr(self, 'loo_sanity_threshold', 0.9))
                    try:
                        pass_mask, sanity_metrics = self.verify_lora_on_support(
                            support_inputs=train_inputs,
                            support_targets=train_outputs,
                            support_features=support_features,
                            lora_deltas=lora_deltas,
                            threshold=threshold,
                        )
                        self._last_lora_sanity_metrics = sanity_metrics

                        # Mask out LoRA deltas for failing samples (keep base model for those)
                        if (pass_mask is not None) and (not bool(pass_mask.all().item())):
                            # Only warn once per process to avoid log spam.
                            if not hasattr(self, '_loo_sanity_warned'):
                                import warnings
                                warnings.warn(
                                    f"[HyperLoRA] LOO sanity check failed for some samples: "
                                    f"acc={sanity_metrics.get('sanity_check_accuracy', 0.0):.3f}, "
                                    f"pass_rate={sanity_metrics.get('sanity_check_pass_rate', 0.0):.3f}, "
                                    f"threshold={sanity_metrics.get('sanity_check_threshold', threshold):.2f}. "
                                    "Masking LoRA deltas for failing samples.",
                                    UserWarning,
                                )
                                self._loo_sanity_warned = True

                            mask = pass_mask.to(support_features.device).float()
                            for k, v in list(lora_deltas.items()):
                                if isinstance(v, torch.Tensor) and v.dim() >= 1 and v.shape[0] == mask.shape[0]:
                                    view_shape = [mask.shape[0]] + [1] * (v.dim() - 1)
                                    lora_deltas[k] = v * mask.view(*view_shape)
                    except Exception:
                        # Safety: never crash inference if sanity check fails.
                        pass
            else:
                # HyperLoRA is enabled but no spatial support features available
                # This can happen if:
                # 1. use_cross_attention_context=False and use_solver_context=False
                # 2. No train_inputs/train_outputs provided
                # Log warning on first occurrence to help debugging
                if not hasattr(self, '_hyperlora_warned'):
                    import warnings
                    warnings.warn(
                        "[HyperLoRA] Enabled but no spatial support_features available. "
                        "LoRA weights will NOT be applied. Ensure use_solver_context=True or "
                        "use_cross_attention_context=True, and provide train_inputs/train_outputs.",
                        UserWarning
                    )
                    self._hyperlora_warned = True
            # Note: If no support_features, lora_deltas stays None and solver
            # runs without adaptation (backward compatible behavior)
        
        # 7. Recursive Solver - generate output
        # CRITICAL: Pass stop_logits to solver for clue aggregation weighting
        # This creates gradient flow: task_loss -> stop_probs -> stop_predictor
        # Making clue count a TRUE latent variable learned from target grids
        # Phase 2.5: Pass support_features for solver cross-attention
        # STAGING: solver_context_active flag allows disabling cross-attention during early epochs
        solver_context_active = getattr(self, 'solver_context_active', True)  # Default: active
        effective_support_features = support_features if solver_context_active else None
        
        act_outputs = None
        if return_all_steps or return_intermediates:
            solver_output = self.solver(
                clue_features=clue_features,
                count_embedding=count_embedding,
                predicates=predicates,
                input_grid=input_grid,
                attention_maps=attention_maps,
                stop_logits=stop_logits,  # For latent clue count learning
                support_features=effective_support_features,  # STAGED: None during early epochs
                return_all_steps=True,
                return_act_outputs=return_intermediates and self.use_act,
                num_steps_override=num_steps_override,
                lora_deltas=lora_deltas,  # HyperLoRA weight adaptations (None during early epochs)
                hpm_memory_tokens=hpm_memory_tokens,  # Jan 2026: HPM solver-context coupling
            )
            # Handle ACT outputs if returned
            if isinstance(solver_output, tuple):
                all_logits, act_outputs = solver_output
            else:
                all_logits = solver_output
            logits = all_logits[-1]
        else:
            logits = self.solver(
                clue_features=clue_features,
                count_embedding=count_embedding,
                predicates=predicates,
                input_grid=input_grid,
                attention_maps=attention_maps,
                stop_logits=stop_logits,  # For latent clue count learning
                support_features=effective_support_features,  # STAGED: None during early epochs
                return_all_steps=False,
                num_steps_override=num_steps_override,
                lora_deltas=lora_deltas,  # HyperLoRA weight adaptations (None during early epochs)
                hpm_memory_tokens=hpm_memory_tokens,  # Jan 2026: HPM solver-context coupling
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
            if support_features is not None:
                result["support_features"] = support_features
                
                # ===== BUG FIX #2: Compute z_struct and z_content for SCL/ortho losses =====
                # z_struct: Structure embedding from aggregated context (transformation rule)
                # This captures WHAT transformation is being applied (same across all demos)
                # Pool support features spatially and across pairs: (B, N, D, H, W) -> (B, D)
                z_struct_raw = support_features.mean(dim=(1, 3, 4))  # (B, D)
                z_struct = self.structure_projector(z_struct_raw)  # (B, D)
                result["z_struct"] = z_struct
                
                # z_struct_demos: Per-demo structure embeddings for consistency loss
                # Shape: (B, N, D) - one embedding per demo pair
                B_sf, N_sf, D_sf, H_sf, W_sf = support_features.shape
                z_struct_demos_raw = support_features.mean(dim=(3, 4))  # (B, N, D)
                z_struct_demos = self.structure_projector(z_struct_demos_raw)  # (B, N, D)
                result["z_struct_demos"] = z_struct_demos
                
                # z_content: Content embedding from test input features (appearance)
                # This captures the CONTENT (colors, patterns) which should vary
                z_content_raw = features.mean(dim=(2, 3))  # (B, D) - pool spatial dims
                z_content = self.content_projector(z_content_raw)  # (B, D)
                result["z_content"] = z_content
                
            if lora_deltas is not None:
                result["lora_deltas"] = lora_deltas
            if act_outputs is not None:
                result["act_outputs"] = act_outputs
            # Add HPM routing weights and retrieval stats if available
            if hpm_routing_weights is not None:
                result["hpm_routing_weights"] = hpm_routing_weights
            if hpm_retrieval_stats:
                result["hpm_retrieval_stats"] = hpm_retrieval_stats
            # Add DSC diagnostics if available
            if self.use_dsc and self.dsc is not None:
                entropy_inputs = self.dsc.get_last_entropy_inputs()
                if entropy_inputs is not None:
                    result["dsc_entropy_inputs"] = entropy_inputs
            return result
        else:
            return logits
    
    def forward_training(
        self,
        input_grids: torch.Tensor,
        output_grids: torch.Tensor,
        test_input: torch.Tensor,
        test_output: torch.Tensor = None,
        grid_mask: torch.Tensor = None,
        temperature: float = 1.0,
        return_intermediates: bool = True,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with SCIARCTrainer-compatible argument names.
        
        This is a wrapper around forward() that maps the trainer's argument names
        to the forward() method's expected argument names for backward compatibility.
        
        Args:
            input_grids: Shape (B, N, H, W) training input grids
            output_grids: Shape (B, N, H, W) training output grids  
            test_input: Shape (B, H, W) test input grid to predict
            test_output: Shape (B, H, W) test output (not used in forward, for loss computation)
            grid_mask: Shape (B, N) boolean mask for valid training pairs
            temperature: Gumbel-softmax temperature
            return_intermediates: If True, return dict with intermediate outputs
            **kwargs: Additional arguments passed to forward()
            
        Returns:
            Dict with 'logits' and intermediate outputs if return_intermediates=True
        """
        # Map trainer argument names to forward() argument names:
        # - test_input -> input_grid (the grid we're predicting)
        # - input_grids -> train_inputs (context/support examples)
        # - output_grids -> train_outputs (context/support labels)
        # - grid_mask -> pair_mask (which context pairs are valid)
        return self.forward(
            input_grid=test_input,
            train_inputs=input_grids,
            train_outputs=output_grids,
            pair_mask=grid_mask,
            temperature=temperature,
            return_intermediates=return_intermediates,
            **kwargs,
        )
    
    def encode_structure_only(
        self,
        input_grids: torch.Tensor,
        output_grids: torch.Tensor,
        pair_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode only the structure (transformation rule) from input-output pairs.
        
        This is an efficient method that skips the full solver and only computes
        the z_struct embedding. Used for CISL content invariance loss where we
        need to compare structure embeddings of original vs color-permuted tasks.
        
        BUG FIX #3: This method was missing and caused AttributeError when
        CISL was enabled with cicl_color_inv_weight > 0.
        
        Args:
            input_grids: Shape (B, N, H, W) training input grids
            output_grids: Shape (B, N, H, W) training output grids
            pair_mask: Shape (B, N) boolean mask for valid training pairs
            
        Returns:
            z_struct: Shape (B, D) structure embedding capturing transformation rule
        """
        if not self.use_context_encoder or self.context_encoder is None:
            # No context encoder - return zeros
            B = input_grids.shape[0]
            return torch.zeros(B, self.hidden_dim, device=input_grids.device)
        
        # Encode training pairs to get support features
        context_output = self.context_encoder(input_grids, output_grids, pair_mask)
        
        if self.context_encoder.use_spatial_features:
            # Spatial features mode: (B, N, D, H, W)
            support_features = context_output
            # Pool to get z_struct: (B, N, D, H, W) -> (B, D)
            z_struct_raw = support_features.mean(dim=(1, 3, 4))
        else:
            # FiLM mode: already (B, D)
            z_struct_raw = context_output
        
        # Project through structure projector
        z_struct = self.structure_projector(z_struct_raw)
        
        return z_struct
    
    def pool_context_from_support(
        self,
        support_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Pool context vector from support features.
        
        Args:
            support_features: Shape (B, N, D, Hs, Ws) encoded support features
            
        Returns:
            context: Shape (B, D) pooled context vector
        """
        # Global average pool across pairs and spatial dims
        return support_features.mean(dim=(1, 3, 4))  # (B, D)
    
    def forward_with_lora(
        self,
        input_grid: torch.Tensor,
        support_features: torch.Tensor,
        lora_deltas: Dict[str, torch.Tensor],
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Forward pass with explicit LoRA weight deltas.
        
        Used by LOO training where we want to test generalization
        using LoRA weights predicted from a subset of examples.
        
        Args:
            input_grid: Shape (B, H, W) input grid
            support_features: Shape (B, N, D, Hs, Ws) encoded support features
            lora_deltas: Dict of LoRA weight deltas from HyperLoRA
            temperature: Softmax temperature
            
        Returns:
            logits: Shape (B, num_classes, H, W) output logits
        """
        B, H, W = input_grid.shape
        
        # 1. Encode grid
        features = self.encode(input_grid)  # (B, D, H, W)
        
        # Compute valid mask and grid sizes
        valid_mask = self.encoder.get_valid_mask(input_grid)
        grid_sizes = self.encoder.get_grid_sizes(input_grid)
        
        # 2. Apply cross-attention context injection if we have spatial features
        if self.context_injector is not None and isinstance(self.context_injector, CrossAttentionInjector):
            features = self.context_injector(features, support_features)
        
        # 3. DSC - find clue anchors
        if self.use_dsc and self.dsc is not None:
            # ENHANCED (Jan 2026): Pass input_grid for FG bias + task complexity signals
            centroids, attention_maps, stop_logits = self.dsc(
                features, temperature=temperature, mask=valid_mask, input_grid=input_grid
            )
        else:
            K = self.max_clues
            centroids = torch.zeros(B, K, 2, device=features.device)
            centroids[:, :, 0] = H / 2
            centroids[:, :, 1] = W / 2
            attention_maps = torch.ones(B, K, H, W, device=features.device) / (H * W)
            stop_logits = torch.zeros(B, K, device=features.device)
        
        # 4. MSRE - relative encodings
        if self.use_msre and self.msre is not None:
            clue_features = self.msre(features, centroids, grid_sizes=grid_sizes)
        else:
            clue_features = features.unsqueeze(1).expand(-1, self.max_clues, -1, -1, -1)
        
        # 5. LCR - counting
        if self.use_lcr and self.lcr is not None:
            count_embedding = self.lcr(
                input_grid, features, mask=valid_mask, attention_maps=attention_maps
            )
        else:
            count_embedding = torch.zeros(
                B, self.num_colors, self.hidden_dim, device=features.device
            )
        
        # 6. SPH - predicates
        if self.use_sph and self.sph is not None:
            predicates = self.sph(features, temperature=temperature)
        else:
            predicates = torch.zeros(B, self.num_predicates, device=features.device)
        
        # 7. Solver with LoRA modulation
        logits = self.solver(
            clue_features=clue_features,
            count_embedding=count_embedding,
            predicates=predicates,
            input_grid=input_grid,
            attention_maps=attention_maps,
            stop_logits=stop_logits,
            support_features=support_features,
            return_all_steps=False,
            lora_deltas=lora_deltas,  # Pass LoRA weights to solver
        )
        
        return logits
    
    def verify_lora_on_support(
        self,
        support_inputs: torch.Tensor,  # (B, N, H, W) or (B, N, C, H, W)
        support_targets: torch.Tensor,  # (B, N, H, W)
        support_features: torch.Tensor,  # (B, N, D, Hs, Ws)
        lora_deltas: Dict[str, torch.Tensor],
        threshold: float = 0.9,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Verify that LoRA weights work on the support set (sanity check).
        
        This is the LOO Sanity Check - if predicted LoRA weights can't
        even get >90% accuracy on the training examples, we shouldn't
        use them for the test example.
        
        Args:
            support_inputs: Support set input grids
            support_targets: Support set target grids
            support_features: Encoded support features
            lora_deltas: Predicted LoRA weight deltas
            threshold: Minimum accuracy to accept LoRA weights
            
        Returns:
            Tuple of (pass_mask, metrics)
            - pass_mask: (B,) boolean tensor - True if sample passes
            - metrics: Dict with accuracy, per-sample scores
        """
        B, N = support_targets.shape[:2]
        device = support_targets.device
        
        total_correct = 0
        total_pixels = 0
        per_sample_accuracy = []
        
        with torch.no_grad():
            for pair_idx in range(N):
                # Get this pair
                if support_inputs.dim() == 5:
                    pair_input = support_inputs[:, pair_idx, 0]  # (B, H, W)
                else:
                    pair_input = support_inputs[:, pair_idx]  # (B, H, W)
                pair_target = support_targets[:, pair_idx]  # (B, H, W)
                
                # Predict with LoRA
                logits = self.forward_with_lora(
                    pair_input, support_features, lora_deltas
                )
                preds = logits.argmax(dim=1)  # (B, H, W)
                
                # Compute accuracy - CRITICAL FIX: mask out ignore_index (-100)
                valid_mask = (pair_target != -100)
                correct = ((preds == pair_target) & valid_mask).float()
                total_correct += correct.sum(dim=(1, 2))  # (B,)
                total_pixels += valid_mask.sum(dim=(1, 2)).float()  # Only count valid pixels
        
        # Per-sample accuracy across all pairs
        per_sample_accuracy = total_correct / total_pixels  # (B,)
        pass_mask = per_sample_accuracy >= threshold
        
        metrics = {
            'sanity_check_accuracy': per_sample_accuracy.mean().item(),
            'sanity_check_pass_rate': pass_mask.float().mean().item(),
            'sanity_check_threshold': threshold,
        }
        
        return pass_mask, metrics

    def inference(
        self,
        input_grid: torch.Tensor,
        train_inputs: Optional[torch.Tensor] = None,
        train_outputs: Optional[torch.Tensor] = None,
        voting_method: str = 'hybrid',
        num_color_perms: int = 4,
        temperature: float = 0.5,  # CRITICAL: Match training temperature, not 0.1!
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Primary inference method - uses Hybrid voting by default.
        
        This is the recommended method for production inference.
        Uses Test-Time Augmentation (8 dihedral × num_color_perms views)
        and votes using the specified method.
        
        Args:
            input_grid: (B, H, W) input grid
            train_inputs: (B, N, H, W) support inputs
            train_outputs: (B, N, H, W) support outputs
            voting_method: 'hybrid' (default), 'acw', or 'trm'
            num_color_perms: Color permutations per dihedral (default: 4)
            temperature: Softmax temperature
            
        Returns:
            prediction: (B, H, W) final prediction
            info: Dict with voting details
        """
        if voting_method == 'hybrid':
            return self.predict_with_hybrid(
                input_grid, train_inputs, train_outputs,
                num_color_perms=num_color_perms, temperature=temperature
            )
        elif voting_method == 'acw':
            return self.predict_with_acw(
                input_grid, train_inputs, train_outputs,
                num_color_perms=num_color_perms, temperature=temperature
            )
        elif voting_method == 'trm':
            # TRM-style: use ACW but return based on raw counts
            pred, info = self.predict_with_acw(
                input_grid, train_inputs, train_outputs,
                num_color_perms=num_color_perms, temperature=temperature
            )
            info['voting_method'] = 'trm'
            return pred, info
        else:
            raise ValueError(f"Unknown voting_method: {voting_method}")

    def predict(
        self,
        input_grid: torch.Tensor,
        train_inputs: Optional[torch.Tensor] = None,
        train_outputs: Optional[torch.Tensor] = None,
        temperature: float = 0.5,  # CRITICAL: Match training temperature, not 0.1!
        num_steps_override: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Make a prediction (argmax of logits).
        
        If use_best_step_selection is enabled, uses the step with lowest
        entropy (most confident prediction) instead of always using the last step.
        
        Args:
            input_grid: Shape (B, H, W) input grid
            train_inputs: Optional (B, N, H, W) training inputs for context
            train_outputs: Optional (B, N, H, W) training outputs for context
            temperature: Softmax temperature (should match training end temp)
            num_steps_override: If provided, run this many solver steps instead of default.
                               Example: Train with 6 steps, infer with 10 to see if more helps.
            
        Returns:
            prediction: Shape (B, H, W) predicted grid
        """
        self.eval()
        with torch.no_grad():
            if self.use_best_step_selection:
                # Get all steps and select best by entropy
                outputs = self.forward(
                    input_grid,
                    train_inputs=train_inputs,
                    train_outputs=train_outputs,
                    temperature=temperature,
                    return_intermediates=True,
                    return_all_steps=True,
                    num_steps_override=num_steps_override,
                )
                all_logits = outputs['all_logits']
                if all_logits and len(all_logits) > 1:
                    # Use solver's entropy-based selection
                    best_logits, best_step, info = self.solver.select_best_step_by_entropy(all_logits)
                    logits = best_logits
                else:
                    logits = outputs['logits']
            else:
                # Use last step (default behavior)
                logits = self.forward(
                    input_grid,
                    train_inputs=train_inputs,
                    train_outputs=train_outputs,
                    temperature=temperature,
                    num_steps_override=num_steps_override,
                )
            prediction = logits.argmax(dim=1)
        return prediction

    def predict_with_acw(
        self,
        input_grid: torch.Tensor,
        train_inputs: Optional[torch.Tensor] = None,
        train_outputs: Optional[torch.Tensor] = None,
        augmentations: Optional[List[str]] = None,
        num_color_perms: int = 4,
        temperature: float = 0.5,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Predict using Augmented Confidence Weighting (ACW).
        
        Generates predictions for multiple augmented views of the input,
        then aggregates them using consistency-weighted voting.
        
        MATCHES training evaluation (evaluate_trm_style):
        - Forward: color permutation FIRST, then dihedral transform
        - Inverse: inverse dihedral FIRST, then inverse color
        
        Args:
            input_grid: (B, H, W) input grid
            train_inputs: (B, N, H, W) support inputs
            train_outputs: (B, N, H, W) support outputs
            augmentations: List of dihedral augmentations (default: all 8 symmetries)
            num_color_perms: Number of color permutations per dihedral (default: 4)
            temperature: Softmax temperature for prediction
            
        Returns:
            winner: (B, H, W) consensus prediction
            info: Dict with voting details
        """
        if augmentations is None:
            augmentations = [
                'identity', 'rotate_90', 'rotate_180', 'rotate_270',
                'flip_h', 'flip_v', 'transpose', 'transpose_neg'
            ]
            
        B, H, W = input_grid.shape
        device = input_grid.device
        all_predictions = []
        
        self.eval()
        with torch.no_grad():
            # Run prediction for each color permutation × dihedral augmentation
            for color_idx in range(num_color_perms):
                # Step 1: Apply color permutation FIRST (matching TRM-style order)
                if color_idx == 0:
                    color_perm = None  # Identity permutation
                    color_input = input_grid
                    color_train_inputs = train_inputs
                    color_train_outputs = train_outputs
                else:
                    # Generate random color permutation (keep 0=black fixed, keep 10=PAD fixed)
                    # CRITICAL: Use 11 entries to handle PAD_COLOR=10 in padded grids
                    color_perm = torch.arange(11, device=device)
                    shuffled = torch.randperm(9, device=device) + 1
                    color_perm[1:10] = shuffled  # Only permute colors 1-9, keep 0 and 10 fixed
                    
                    # Apply to all grids (clamp to valid range for safety)
                    color_input = color_perm[input_grid.clamp(0, 10).long()]
                    if train_inputs is not None:
                        color_train_inputs = color_perm[train_inputs.clamp(0, 10).long()]
                        # For train_outputs, handle ignore_index=-100 (from target padding)
                        # Mask out -100, apply permutation, then restore -100
                        ignore_mask = train_outputs < 0
                        safe_outputs = train_outputs.clamp(0, 10).long()
                        color_train_outputs = color_perm[safe_outputs]
                        color_train_outputs[ignore_mask] = train_outputs[ignore_mask]
                    else:
                        color_train_inputs = None
                        color_train_outputs = None
                
                for aug in augmentations:
                    # Step 2: Apply dihedral transform SECOND
                    aug_input = apply_augmentation(color_input, aug)
                    
                    # Augment support set if provided
                    aug_train_inputs = None
                    aug_train_outputs = None
                    if color_train_inputs is not None and color_train_outputs is not None:
                        # train_inputs is (B, N, H, W)
                        N = color_train_inputs.shape[1]
                        aug_train_inputs = torch.stack([
                            apply_augmentation(color_train_inputs[:, i], aug) for i in range(N)
                        ], dim=1)
                        aug_train_outputs = torch.stack([
                            apply_augmentation(color_train_outputs[:, i], aug) for i in range(N)
                        ], dim=1)
                    
                    # Step 3: Predict
                    pred = self.predict(
                        aug_input, 
                        train_inputs=aug_train_inputs, 
                        train_outputs=aug_train_outputs,
                        temperature=temperature
                    )
                    
                    # Step 4: Inverse dihedral FIRST (CRITICAL: before inverse color!)
                    inv_dihedral = apply_augmentation(pred, aug, inverse=True)
                    
                    # Step 5: Inverse color permutation SECOND
                    if color_perm is not None:
                        inv_color_perm = torch.argsort(color_perm)
                        inv_pred = inv_color_perm[inv_dihedral.long()]
                    else:
                        inv_pred = inv_dihedral
                    
                    all_predictions.append(inv_pred)
            
        # Vote per sample
        winners = []
        batch_info = []
        
        acw = AugmentedConfidenceWeighting()
        
        for b in range(B):
            # Collect predictions for this sample
            sample_preds = [p[b] for p in all_predictions]
            
            # Perform weighted vote
            winner, candidates = acw.weighted_vote(sample_preds)
            winners.append(winner)
            batch_info.append(candidates)
        
        total_views = len(augmentations) * num_color_perms
        return torch.stack(winners), {
            'candidates': batch_info,
            'num_dihedral': len(augmentations),
            'num_color_perms': num_color_perms,
            'total_views': total_views,
            'voting_method': 'acw',
        }

    def predict_with_hybrid(
        self,
        input_grid: torch.Tensor,
        train_inputs: Optional[torch.Tensor] = None,
        train_outputs: Optional[torch.Tensor] = None,
        augmentations: Optional[List[str]] = None,
        num_color_perms: int = 4,
        temperature: float = 0.5,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Predict using Hybrid voting (TRM + ACW ensemble).
        
        Uses both TRM-style count voting and ACW consistency-weighted voting.
        When they agree → high confidence
        When they disagree → use ACW (more robust to noise)
        
        This provides the best of both worlds:
        - TRM: Simple, deterministic, reproducible
        - ACW: Better at filtering inconsistent predictions
        
        Args:
            input_grid: (B, H, W) input grid
            train_inputs: (B, N, H, W) support inputs
            train_outputs: (B, N, H, W) support outputs
            augmentations: List of dihedral augmentations (default: all 8 symmetries)
            num_color_perms: Number of color permutations per dihedral (default: 4)
            temperature: Softmax temperature for prediction
            
        Returns:
            winner: (B, H, W) consensus prediction
            info: Dict with voting details including agreement status
        """
        if augmentations is None:
            augmentations = [
                'identity', 'rotate_90', 'rotate_180', 'rotate_270',
                'flip_h', 'flip_v', 'transpose', 'transpose_neg'
            ]
            
        B, H, W = input_grid.shape
        device = input_grid.device
        all_predictions = []
        
        self.eval()
        with torch.no_grad():
            # Generate all augmented predictions (same as predict_with_acw)
            for color_idx in range(num_color_perms):
                if color_idx == 0:
                    color_perm = None
                    color_input = input_grid
                    color_train_inputs = train_inputs
                    color_train_outputs = train_outputs
                else:
                    # CRITICAL: Use 11 entries to handle PAD_COLOR=10 in padded grids
                    color_perm = torch.arange(11, device=device)
                    shuffled = torch.randperm(9, device=device) + 1
                    color_perm[1:10] = shuffled  # Only permute colors 1-9, keep 0 and 10 fixed
                    
                    color_input = color_perm[input_grid.clamp(0, 10).long()]
                    if train_inputs is not None:
                        color_train_inputs = color_perm[train_inputs.clamp(0, 10).long()]
                        # BUG FIX #2: Handle ignore_index=-100 in train_outputs
                        # (same fix as predict_with_acw lines 1061-1066)
                        ignore_mask = train_outputs < 0
                        safe_outputs = train_outputs.clamp(0, 10).long()
                        color_train_outputs = color_perm[safe_outputs]
                        color_train_outputs[ignore_mask] = train_outputs[ignore_mask]
                    else:
                        color_train_inputs = None
                        color_train_outputs = None
                
                for aug in augmentations:
                    aug_input = apply_augmentation(color_input, aug)
                    
                    aug_train_inputs = None
                    aug_train_outputs = None
                    if color_train_inputs is not None and color_train_outputs is not None:
                        N = color_train_inputs.shape[1]
                        aug_train_inputs = torch.stack([
                            apply_augmentation(color_train_inputs[:, i], aug) for i in range(N)
                        ], dim=1)
                        aug_train_outputs = torch.stack([
                            apply_augmentation(color_train_outputs[:, i], aug) for i in range(N)
                        ], dim=1)
                    
                    pred = self.predict(
                        aug_input, 
                        train_inputs=aug_train_inputs, 
                        train_outputs=aug_train_outputs,
                        temperature=temperature
                    )
                    
                    inv_dihedral = apply_augmentation(pred, aug, inverse=True)
                    
                    if color_perm is not None:
                        inv_color_perm = torch.argsort(color_perm)
                        inv_pred = inv_color_perm[inv_dihedral.long()]
                    else:
                        inv_pred = inv_dihedral
                    
                    all_predictions.append(inv_pred)
            
        # Hybrid vote per sample
        winners = []
        batch_info = []
        
        acw = AugmentedConfidenceWeighting()
        
        for b in range(B):
            sample_preds = [p[b] for p in all_predictions]
            winner, info = acw.hybrid_vote(sample_preds)
            winners.append(winner)
            batch_info.append(info)
        
        # Aggregate agreement stats
        num_agree = sum(1 for info in batch_info if info.get('agree', False))
        
        total_views = len(augmentations) * num_color_perms
        return torch.stack(winners), {
            'batch_info': batch_info,
            'num_dihedral': len(augmentations),
            'num_color_perms': num_color_perms,
            'total_views': total_views,
            'voting_method': 'hybrid',
            'agreement_rate': num_agree / B if B > 0 else 0,
        }

    
    def get_best_step_logits(
        self,
        all_logits: List[torch.Tensor],
        targets: Optional[torch.Tensor] = None,
        loss_fn: Optional[torch.nn.Module] = None,
    ) -> Tuple[torch.Tensor, int, dict]:
        """
        Select the best step's logits using the solver's selection method.
        
        Wrapper for solver.select_best_step_combined().
        
        Args:
            all_logits: List of predictions from each step
            targets: Ground truth (optional, for training)
            loss_fn: Loss function (optional, for training)
            
        Returns:
            best_logits: Best prediction
            best_step: Index of best step
            info: Dict with method, values
        """
        return self.solver.select_best_step_combined(all_logits, targets, loss_fn)
    
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
            "hyper_lora": sum(p.numel() for p in self.hyper_lora.parameters()) if self.hyper_lora else 0,
            "hpm": sum(p.numel() for p in self.hpm.parameters()) if self.hpm else 0,
        }
        counts["total"] = sum(counts.values())
        counts["trainable"] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return counts
    
    # =========================================================================
    # HPM (Hierarchical Primitive Memory) Helper Methods
    # =========================================================================
    
    def hpm_on_epoch_start(self):
        """Call at epoch start to reset HPM statistics.
        
        This resets routing statistics for load balancing loss computation.
        Should be called in training loop at the start of each epoch.
        """
        if self.use_hpm and self.hpm is not None:
            self.hpm.reset_epoch_stats()
    
    def hpm_get_load_balance_loss(self) -> torch.Tensor:
        """Get HPM load balancing loss for training.
        
        This loss encourages uniform bank usage, preventing mode collapse.
        Should be added to the total loss with appropriate weighting.
        
        Returns:
            Load balancing loss (scalar tensor, 0.0 if HPM disabled)
        """
        if self.use_hpm and self.hpm is not None:
            return self.hpm.get_load_balance_loss()
        return torch.tensor(0.0)
    
    def hpm_on_backward(self):
        """Call after loss.backward() to apply gradient routing.
        
        This zeros gradients for frozen primitives in static banks,
        implementing selective protection for continual learning.
        """
        if self.use_hpm and self.hpm is not None:
            self.hpm.apply_gradient_routing()
    
    def hpm_on_task_complete(
        self,
        z_context: torch.Tensor,
        z_task: Optional[torch.Tensor] = None,
        task_id: Optional[str] = None,
        force_write: bool = False,
    ):
        """Call after successfully completing a task for continual learning.
        
        This:
        1. Freezes stable primitives in static banks (if HPM active)
        2. Stores embeddings in dynamic buffers for future retrieval
        
        TODO 2 FIX: Allow buffer writes even when use_hpm is staged off.
        The `force_write` parameter or `hpm_memory_enabled` attribute allows
        populating buffers before HPM activation (hpm_start_epoch).
        
        Args:
            z_context: Context embedding from ContextEncoder [B, D] or [D]
            z_task: Optional task embedding from HyperLoRA [B, D] or [D]
            task_id: Optional task identifier for debugging
            force_write: If True, write to buffers even if use_hpm=False
        """
        # Check if memory writing is allowed
        hpm_memory_enabled = getattr(self, 'hpm_memory_enabled', self.use_hpm)
        if not (self.use_hpm or force_write or hpm_memory_enabled):
            return
        
        # Freeze stable primitives only if HPM is active (not just memory writing)
        if self.use_hpm and self.hpm is not None:
            self.hpm.freeze_stable_primitives()
        
        # Store in Instance buffer (always if buffer exists and memory enabled)
        if self.hpm_instance_buffer is not None:
            self.hpm_instance_buffer.add(z_context, z_context, task_id)
        
        # Store in Procedural buffer (if HyperLoRA available)
        if self.hpm_procedural_buffer is not None and z_task is not None:
            self.hpm_procedural_buffer.add(z_context, z_task, task_id)
    
    # Alias for backward compatibility with training script
    def hpm_add_solved_task(
        self,
        z_context: torch.Tensor,
        z_task: Optional[torch.Tensor] = None,
        task_id: Optional[str] = None,
        force_write: bool = False,
    ):
        """Alias for hpm_on_task_complete - called by training script when a task is solved.
        
        TODO 2: Added force_write parameter for staged HPM memory population.
        """
        return self.hpm_on_task_complete(z_context, z_task, task_id, force_write=force_write)
    
    def hpm_buffer_contains_task(self, task_id: str) -> bool:
        """Check if a task_id already exists in any HPM buffer (for global dedup).
        
        TODO 3: Global dedup across epochs - check before adding to prevent duplicates.
        
        Args:
            task_id: Task identifier to check
            
        Returns:
            True if task_id is already in either buffer
        """
        if self.hpm_instance_buffer is not None and self.hpm_instance_buffer.contains_task(task_id):
            return True
        if self.hpm_procedural_buffer is not None and self.hpm_procedural_buffer.contains_task(task_id):
            return True
        return False
    
    def hpm_get_stats(self) -> dict:
        """Get HPM statistics for logging.
        
        Returns:
            Dict with gate_value, routing_distribution, load_balance_loss, buffer sizes
        """
        if not self.use_hpm or self.hpm is None:
            return {}
        
        stats = self.hpm.get_stats()
        
        # Add buffer sizes
        if self.hpm_instance_buffer is not None:
            stats['instance_buffer_size'] = len(self.hpm_instance_buffer)
        if self.hpm_procedural_buffer is not None:
            stats['procedural_buffer_size'] = len(self.hpm_procedural_buffer)
        
        return stats
    
    def get_hpm_state(self) -> dict:
        """Get canonical HPM state for serialization.
        
        This is the single source of truth for HPM memory persistence.
        Use for both training checkpoints and inference export.
        
        Returns:
            Dict with instance and procedural buffer states + metadata
        """
        state = {
            'version': '2.0',
            'use_hpm': self.use_hpm,
            'instance': None,
            'procedural': None,
        }
        
        if self.hpm_instance_buffer is not None and len(self.hpm_instance_buffer) > 0:
            state['instance'] = self.hpm_instance_buffer.state_dict()
        
        if self.hpm_procedural_buffer is not None and len(self.hpm_procedural_buffer) > 0:
            state['procedural'] = self.hpm_procedural_buffer.state_dict()
        
        return state
    
    def load_hpm_state(self, state: dict, force_load: bool = False) -> None:
        """Load HPM state from canonical format.
        
        TODO 5 FIX: Allow loading memory even when use_hpm is temporarily disabled.
        This supports staged HPM activation and inference-time toggles.
        
        Args:
            state: Dict from get_hpm_state() or checkpoint
            force_load: If True, load memory even if use_hpm=False (for inference/staging)
        """
        # TODO 5: Relaxed gating - allow loading if force_load or buffers exist
        if not (self.use_hpm or force_load):
            # Only skip if both HPM is disabled AND we're not forcing
            # Still try to load if buffers exist (created from config)
            if self.hpm_instance_buffer is None and self.hpm_procedural_buffer is None:
                return
        
        if state.get('instance') is not None and self.hpm_instance_buffer is not None:
            self.hpm_instance_buffer.load_state_dict(state['instance'])
        
        if state.get('procedural') is not None and self.hpm_procedural_buffer is not None:
            self.hpm_procedural_buffer.load_state_dict(state['procedural'])
    
    def export_hpm_memory(
        self, 
        path: str, 
        split_tag: str = 'train',
        epoch_range: Optional[tuple] = None,
        config_hash: Optional[str] = None,
        dataset_hash: Optional[str] = None,
        solved_criterion: str = 'exact_match',
    ) -> None:
        """Export HPM memory as a standalone artifact for inference.
        
        This creates a versioned, auditable memory artifact that can be
        loaded independently of model weights for inference-time adaptation.
        
        TODO 8: Enhanced provenance metadata for auditability.
        
        Args:
            path: Output file path (e.g., 'hpm_memory.pt')
            split_tag: Dataset split this memory was trained on (for provenance)
            epoch_range: (start_epoch, end_epoch) tuple for training range
            config_hash: Hash of training config for reproducibility
            dataset_hash: Hash of dataset used for training
            solved_criterion: Criterion used for adding tasks ('exact_match', 'high_accuracy', etc.)
        """
        import time
        
        state = self.get_hpm_state()
        state['metadata'] = {
            # Core provenance
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'split_tag': split_tag,
            'instance_size': len(self.hpm_instance_buffer) if self.hpm_instance_buffer else 0,
            'procedural_size': len(self.hpm_procedural_buffer) if self.hpm_procedural_buffer else 0,
            # TODO 8: Extended provenance for auditability
            'epoch_range': epoch_range,
            'config_hash': config_hash,
            'dataset_hash': dataset_hash,
            'solved_criterion': solved_criterion,
            # Unique task counts for dedup verification
            'unique_instance_tasks': len(self.hpm_instance_buffer.get_unique_task_ids()) if self.hpm_instance_buffer else 0,
            'unique_procedural_tasks': len(self.hpm_procedural_buffer.get_unique_task_ids()) if self.hpm_procedural_buffer else 0,
        }
        
        torch.save(state, path)
    
    def import_hpm_memory(self, path: str, force_load: bool = True) -> dict:
        """Import HPM memory from standalone artifact.
        
        TODO 4: Explicit inference-time loading path.
        TODO 5: Uses force_load=True by default to support inference toggles.
        
        Args:
            path: Path to hpm_memory.pt file
            force_load: If True (default), load even if use_hpm=False
            
        Returns:
            Metadata dict with provenance info
        """
        state = torch.load(path, map_location='cpu')
        self.load_hpm_state(state, force_load=force_load)
        return state.get('metadata', {})
    
    def hpm_save_buffers(self, directory: str):
        """Save HPM dynamic buffers to disk.
        
        Args:
            directory: Directory to save buffers
        """
        import os
        if not self.use_hpm:
            return
        
        os.makedirs(directory, exist_ok=True)
        
        if self.hpm_instance_buffer is not None and len(self.hpm_instance_buffer) > 0:
            self.hpm_instance_buffer.save(os.path.join(directory, 'instance_buffer.pt'))
        
        if self.hpm_procedural_buffer is not None and len(self.hpm_procedural_buffer) > 0:
            self.hpm_procedural_buffer.save(os.path.join(directory, 'procedural_buffer.pt'))
    
    def hpm_load_buffers(self, directory: str):
        """Load HPM dynamic buffers from disk.
        
        Args:
            directory: Directory containing saved buffers
        """
        import os
        if not self.use_hpm:
            return
        
        instance_path = os.path.join(directory, 'instance_buffer.pt')
        if os.path.exists(instance_path) and self.hpm_instance_buffer is not None:
            self.hpm_instance_buffer = DynamicMemoryBuffer.load(instance_path)
        
        procedural_path = os.path.join(directory, 'procedural_buffer.pt')
        if os.path.exists(procedural_path) and self.hpm_procedural_buffer is not None:
            self.hpm_procedural_buffer = DynamicMemoryBuffer.load(procedural_path)
    
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
                "use_solver_context": self.use_solver_context,  # Phase 2.5
                "use_best_step_selection": self.use_best_step_selection,  # Phase 3
                # HyperLoRA config - BUG FIX #7: Save full HyperLoRA config
                "use_hyperlora": self.use_hyperlora,
                "hyperlora_rank": getattr(self, '_hyperlora_rank', 8),
                "hyperlora_scaling": getattr(self, '_hyperlora_scaling', 1.0),
                "hyperlora_dropout": getattr(self, '_hyperlora_dropout', 0.0),
                "hyperlora_init_scale": getattr(self, '_hyperlora_init_scale', 0.01),
                # HPM config
                "use_hpm": self.use_hpm,
                # Additional flags for conditional module recreation
                "use_learned_pos": self.use_learned_pos if hasattr(self, 'use_learned_pos') else False,
                "dropout": self.dropout if hasattr(self, 'dropout') else 0.1,
            },
            **extra_data,
        }
        torch.save(checkpoint, path)
    
    @classmethod
    def load_from_checkpoint(cls, path: str, device: str = "cpu") -> "RLAN":
        """Load model from checkpoint including HPM dynamic buffers."""
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        
        # Extract saved config and filter to only RLANConfig fields
        saved_config = checkpoint["config"]
        
        # Get valid RLANConfig field names
        import dataclasses
        valid_fields = {f.name for f in dataclasses.fields(RLANConfig)}
        
        # Filter to only valid config fields (ignore ablation flags stored separately)
        config_kwargs = {k: v for k, v in saved_config.items() if k in valid_fields}
        
        # Create config and model
        config = RLANConfig(**config_kwargs)
        model = cls(config=config)
        model.load_state_dict(checkpoint["model_state_dict"])
        
        # Restore HPM dynamic buffers (critical for inference with continual learning)
        if 'hpm_instance_buffer' in checkpoint:
            if hasattr(model, 'hpm_instance_buffer') and model.hpm_instance_buffer is not None:
                buf_data = checkpoint['hpm_instance_buffer']
                model.hpm_instance_buffer.clear()
                for key, value, task_id in zip(buf_data['keys'], buf_data['values'], buf_data['task_ids']):
                    model.hpm_instance_buffer._keys.append(key)
                    model.hpm_instance_buffer._values.append(value)
                    model.hpm_instance_buffer._task_ids.append(task_id)
                if model.hpm_instance_buffer.use_faiss and model.hpm_instance_buffer._faiss_index is not None:
                    model.hpm_instance_buffer._rebuild_faiss_index()
        
        if 'hpm_procedural_buffer' in checkpoint:
            if hasattr(model, 'hpm_procedural_buffer') and model.hpm_procedural_buffer is not None:
                buf_data = checkpoint['hpm_procedural_buffer']
                model.hpm_procedural_buffer.clear()
                for key, value, task_id in zip(buf_data['keys'], buf_data['values'], buf_data['task_ids']):
                    model.hpm_procedural_buffer._keys.append(key)
                    model.hpm_procedural_buffer._values.append(value)
                    model.hpm_procedural_buffer._task_ids.append(task_id)
                if model.hpm_procedural_buffer.use_faiss and model.hpm_procedural_buffer._faiss_index is not None:
                    model.hpm_procedural_buffer._rebuild_faiss_index()
        
        return model
