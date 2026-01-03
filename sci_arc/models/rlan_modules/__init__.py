"""
RLAN Submodules Package

Recursive Latent Attractor Networks for ARC reasoning.

Core Modules:
- DynamicSaliencyController: Discovers spatial anchors (clue pixels)
- MultiScaleRelativeEncoding: Relative coordinate representations
- LatentCountingRegisters: Soft counting for numerical reasoning
- SymbolicPredicateHeads: Binary predicates for conditional logic
- RecursiveSolver: Iterative refinement decoder
- ContextEncoder: Encodes training examples to understand task

Enhanced Modules (TRM-inspired improvements):
- SwiGLU: Swish-Gated Linear Unit (superior to GELU)
- ACTController: Adaptive Computation Time for variable reasoning depth
- RotaryPositionEmbedding2D: RoPE for relative position encoding
- HybridPositionEncoding: Combined learned + RoPE positions
"""

from .dynamic_saliency_controller import DynamicSaliencyController
from .multi_scale_relative_encoding import MultiScaleRelativeEncoding
from .latent_counting_registers import LatentCountingRegisters
from .symbolic_predicate_heads import SymbolicPredicateHeads
from .recursive_solver import RecursiveSolver
from .context_encoder import ContextEncoder, ContextInjector, PairEncoder, CrossAttentionInjector, SpatialPairEncoder
from .activations import SwiGLU, SwiGLUConv2d, FFN
from .adaptive_computation import ACTController, AdaptiveHaltHead, ACTState, PonderingCost
from .positional_encoding import (
    RotaryPositionEmbedding,
    RotaryPositionEmbedding2D,
    LearnedPositionEmbedding2D,
    HybridPositionEncoding,
)
from .hyper_lora import (
    HyperLoRA,
    HyperLoRAConfig,
    LoRAPredictor,
    LoRAApplicator,
    compute_hyperlora_health_metrics,
)
from .loo_training import (
    LOOConfig,
    EquivarianceConfig,
    LOOTrainingLoss,
    AugmentationEquivarianceLoss,
    OutputEquivarianceLoss,  # Jan 2026: Output-level equiv for TTA consensus
    CombinedMetaLoss,
    create_augmented_contexts,
    AugmentedConfidenceWeighting,
)
from .hpm import (
    MemoryBankType,
    MemoryBank,
    MemoryRouter,
    CrossBankAggregator,
    HierarchicalPrimitiveMemory,
    HPMConfig,
    STATIC_BANK_TYPES,
    DYNAMIC_BANK_TYPES,
)
from .dynamic_buffer import DynamicMemoryBuffer

__all__ = [
    # Core modules
    "DynamicSaliencyController",
    "MultiScaleRelativeEncoding",
    "LatentCountingRegisters",
    "SymbolicPredicateHeads",
    "RecursiveSolver",
    "ContextEncoder",
    "ContextInjector",
    "PairEncoder",
    "CrossAttentionInjector",
    "SpatialPairEncoder",
    # Activations
    "SwiGLU",
    "SwiGLUConv2d",
    "FFN",
    # Adaptive computation
    "ACTController",
    "AdaptiveHaltHead",
    "ACTState",
    "PonderingCost",
    # Positional encoding
    "RotaryPositionEmbedding",
    "RotaryPositionEmbedding2D",
    "LearnedPositionEmbedding2D",
    "HybridPositionEncoding",
    # HyperLoRA (Meta-learning weight adaptation)
    "HyperLoRA",
    "HyperLoRAConfig",
    "LoRAPredictor",
    "LoRAApplicator",
    "compute_hyperlora_health_metrics",
    # LOO Training (Leave-One-Out meta-learning)
    "LOOConfig",
    "EquivarianceConfig",
    "LOOTrainingLoss",
    "AugmentationEquivarianceLoss",
    "CombinedMetaLoss",
    "create_augmented_contexts",
    "AugmentedConfidenceWeighting",
    # HPM (Hierarchical Primitive Memory v2)
    "MemoryBankType",
    "MemoryBank",
    "MemoryRouter",
    "CrossBankAggregator",
    "HierarchicalPrimitiveMemory",
    "HPMConfig",
    "STATIC_BANK_TYPES",
    "DYNAMIC_BANK_TYPES",
    "DynamicMemoryBuffer",
]

