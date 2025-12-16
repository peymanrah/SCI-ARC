"""RLAN Model Components

This module provides the RLAN (Relational Latent Abstract Network) architecture
for ARC (Abstraction and Reasoning Corpus) tasks.

For legacy SCI-ARC/CISL components, see others/models/
"""

from sci_arc.models.grid_encoder import GridEncoder, SinusoidalPositionalEncoding2D

# RLAN components
from sci_arc.models.rlan import RLAN, RLANConfig
from sci_arc.models.rlan_modules import (
    DynamicSaliencyController,
    MultiScaleRelativeEncoding,
    LatentCountingRegisters,
    SymbolicPredicateHeads,
    RecursiveSolver,
)

__all__ = [
    # Shared components
    "GridEncoder",
    "SinusoidalPositionalEncoding2D",
    # RLAN
    "RLAN",
    "RLANConfig",
    "DynamicSaliencyController",
    "MultiScaleRelativeEncoding",
    "LatentCountingRegisters",
    "SymbolicPredicateHeads",
    "RecursiveSolver",
]
