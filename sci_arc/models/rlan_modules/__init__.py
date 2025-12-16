"""
RLAN Submodules Package

Recursive Latent Attractor Networks for ARC reasoning.

Modules:
- DynamicSaliencyController: Discovers spatial anchors (clue pixels)
- MultiScaleRelativeEncoding: Relative coordinate representations
- LatentCountingRegisters: Soft counting for numerical reasoning
- SymbolicPredicateHeads: Binary predicates for conditional logic
- RecursiveSolver: Iterative refinement decoder
"""

from .dynamic_saliency_controller import DynamicSaliencyController
from .multi_scale_relative_encoding import MultiScaleRelativeEncoding
from .latent_counting_registers import LatentCountingRegisters
from .symbolic_predicate_heads import SymbolicPredicateHeads
from .recursive_solver import RecursiveSolver

__all__ = [
    "DynamicSaliencyController",
    "MultiScaleRelativeEncoding",
    "LatentCountingRegisters",
    "SymbolicPredicateHeads",
    "RecursiveSolver",
]
