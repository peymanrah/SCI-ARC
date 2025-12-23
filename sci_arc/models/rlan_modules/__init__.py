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
]

