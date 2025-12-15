# Baselines Package
# Contains thin wrappers that import from ORIGINAL UNMODIFIED baseline implementations
# The actual TRM code is in TinyRecursiveModels-main/ (unmodified)

from baselines.trm import (
    TRM,
    TRMConfig,
    TRMCarry,
    HRM,
    HRMConfig,
    TransformerBaseline,
    TransformerBaselineConfig,
    TRMLossHead,
    TRM_AVAILABLE,
)

__all__ = [
    'TRM',
    'TRMConfig',
    'TRMCarry',
    'HRM',
    'HRMConfig',
    'TransformerBaseline',
    'TransformerBaselineConfig',
    'TRMLossHead',
    'TRM_AVAILABLE',
]

