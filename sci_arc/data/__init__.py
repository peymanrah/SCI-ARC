"""
SCI-ARC Data Pipeline.

Provides:
- SCIARCDataset: PyTorch Dataset for ARC tasks
- Collation functions for batching
- Transformation family labels for SCL
- Dihedral transforms (matching TRM exactly)
"""

from .dataset import (
    SCIARCDataset,
    ARCTask,
    SCIARCSample,
    collate_sci_arc,
    create_dataloader,
    pad_grid,
    TRMCompatibleDataset,
    # Dihedral transforms
    dihedral_transform,
    inverse_dihedral_transform,
    DIHEDRAL_INVERSE,
)

from .transform_families import (
    TRANSFORM_FAMILIES,
    FAMILY_TO_NAME,
    NUM_TRANSFORM_FAMILIES,
    get_transform_family,
    infer_transform_from_grids,
    get_rearc_transform_family,
    CONCEPT_ARC_MAP,
)

__all__ = [
    # Dataset
    'SCIARCDataset',
    'ARCTask',
    'SCIARCSample',
    'collate_sci_arc',
    'create_dataloader',
    'pad_grid',
    'TRMCompatibleDataset',
    # Dihedral transforms
    'dihedral_transform',
    'inverse_dihedral_transform',
    'DIHEDRAL_INVERSE',
    # Transform families
    'TRANSFORM_FAMILIES',
    'FAMILY_TO_NAME',
    'NUM_TRANSFORM_FAMILIES',
    'get_transform_family',
    'infer_transform_from_grids',
    'get_rearc_transform_family',
    'CONCEPT_ARC_MAP',
]
