"""
RLAN: Relational Latent Attractor Networks for Abstract Reasoning

A neural architecture for solving ARC (Abstraction and Reasoning Corpus) tasks
through relational reasoning and latent attractor dynamics.

Key Components:
- GridEncoder: Encodes ARC grids into embeddings
- DynamicSaliencyController: Discovers important regions via Gumbel-softmax attention
- MultiScaleRelativeEncoding: Creates relative spatial encodings
- LatentCountingRegisters: Soft counting for numerical reasoning
- SymbolicPredicateHeads: Binary predicate computation
- RecursiveSolver: ConvGRU-based iterative refinement

For legacy SCI-ARC/CISL components, see others/
"""

__version__ = "0.2.0"
__author__ = "RLAN Team"

from sci_arc.models.grid_encoder import GridEncoder, SinusoidalPositionalEncoding2D
from sci_arc.models.rlan import RLAN, RLANConfig
from sci_arc.models.rlan_modules import (
    DynamicSaliencyController,
    MultiScaleRelativeEncoding,
    LatentCountingRegisters,
    SymbolicPredicateHeads,
    RecursiveSolver,
)
from sci_arc.training.rlan_loss import (
    RLANLoss,
    FocalLoss,
    EntropyRegularization,
    SparsityRegularization,
    PredicateDiversityLoss,
    CurriculumPenalty,
)

__all__ = [
    # Core Model
    "RLAN",
    "RLANConfig",
    # Encoder
    "GridEncoder",
    "SinusoidalPositionalEncoding2D",
    # RLAN Modules
    "DynamicSaliencyController",
    "MultiScaleRelativeEncoding",
    "LatentCountingRegisters",
    "SymbolicPredicateHeads",
    "RecursiveSolver",
    # Loss Functions
    "RLANLoss",
    "FocalLoss",
    "EntropyRegularization",
    "SparsityRegularization",
    "PredicateDiversityLoss",
    "CurriculumPenalty",
]
