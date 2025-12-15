"""
SCI-ARC: Structural Causal Invariance for Abstract Reasoning Corpus

A novel combination of SCI's structure-content separation with TRM's recursive refinement
for state-of-the-art performance on ARC-AGI benchmarks.

Key Components:
- StructuralEncoder2D: Extracts transformation patterns (what operation?)
- ContentEncoder2D: Extracts objects/content (which objects?)
- CausalBinding2D: Binds structure to content (z_task)
- RecursiveRefinement: TRM-style iterative answer improvement

Novel Contributions:
1. First application of structural invariance to visual reasoning
2. Explicit structure-content separation for 2D grids
3. Structural Contrastive Loss (SCL) for transformation clustering
4. Integration with recursive refinement for improved accuracy
"""

__version__ = "0.1.0"
__author__ = "SCI-ARC Team"

from sci_arc.models.sci_arc import SCIARC, SCIARCConfig
from sci_arc.models.grid_encoder import GridEncoder, SinusoidalPositionalEncoding2D
from sci_arc.models.structural_encoder import StructuralEncoder2D, AbstractionLayer2D
from sci_arc.models.content_encoder import ContentEncoder2D, OrthogonalProjector
from sci_arc.models.causal_binding import CausalBinding2D
from sci_arc.models.recursive_refinement import RecursiveRefinement
from sci_arc.training.losses import StructuralContrastiveLoss, SCIARCLoss, OrthogonalityLoss

__all__ = [
    "SCIARC",
    "SCIARCConfig",
    "GridEncoder",
    "SinusoidalPositionalEncoding2D",
    "StructuralEncoder2D",
    "AbstractionLayer2D",
    "ContentEncoder2D",
    "OrthogonalProjector",
    "CausalBinding2D",
    "RecursiveRefinement",
    "StructuralContrastiveLoss",
    "SCIARCLoss",
    "OrthogonalityLoss",
]
