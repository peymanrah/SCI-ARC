# TRM Baseline - Thin Wrapper
# ===========================================================================
# This module provides access to the ORIGINAL UNMODIFIED TRM implementation
# located in TinyRecursiveModels-main/
#
# For fair scientific comparison, we do NOT modify the original TRM code.
# This wrapper only provides convenient imports for SCI-ARC integration.
#
# Original Source:
#   Repository: https://github.com/SamsungSAILMontreal/TinyRecursiveModels
#   Paper: "Less is More: Recursive Reasoning with Tiny Networks"
#   Authors: Samsung SAIL Montreal
#   License: Apache 2.0
# ===========================================================================

import sys
from pathlib import Path

# Add the original TRM directory to Python path
_trm_path = Path(__file__).parent.parent.parent / "TinyRecursiveModels-main"
if _trm_path.exists() and str(_trm_path) not in sys.path:
    sys.path.insert(0, str(_trm_path))

# Import from original TRM (UNMODIFIED)
try:
    from models.recursive_reasoning.trm import (
        TinyRecursiveReasoningModel_ACTV1,
        TinyRecursiveReasoningModel_ACTV1Config,
        TinyRecursiveReasoningModel_ACTV1Carry,
        TinyRecursiveReasoningModel_ACTV1InnerCarry,
    )
    from models.layers import (
        Attention,
        SwiGLU,
        LinearSwish,
        RotaryEmbedding,
        CastedLinear,
        CastedEmbedding,
        rms_norm,
    )
    from models.sparse_embedding import CastedSparseEmbedding
    from models.common import trunc_normal_init_
    from models.ema import EMAHelper
    from models.losses import stablemax_cross_entropy
    
    # Alias for convenience
    TRM = TinyRecursiveReasoningModel_ACTV1
    TRMConfig = TinyRecursiveReasoningModel_ACTV1Config
    
    TRM_AVAILABLE = True
    
except ImportError as e:
    TRM_AVAILABLE = False
    _import_error = str(e)
    
    def _raise_import_error(*args, **kwargs):
        raise ImportError(
            f"Original TRM not found. Please ensure TinyRecursiveModels-main/ "
            f"exists in the project root.\n"
            f"Download from: https://github.com/SamsungSAILMontreal/TinyRecursiveModels\n"
            f"Original error: {_import_error}"
        )
    
    TRM = _raise_import_error
    TRMConfig = _raise_import_error


__all__ = [
    'TRM',
    'TRMConfig',
    'TinyRecursiveReasoningModel_ACTV1',
    'TinyRecursiveReasoningModel_ACTV1Config',
    'TinyRecursiveReasoningModel_ACTV1Carry',
    'TinyRecursiveReasoningModel_ACTV1InnerCarry',
    'Attention',
    'SwiGLU',
    'LinearSwish',
    'RotaryEmbedding',
    'CastedLinear',
    'CastedEmbedding',
    'CastedSparseEmbedding',
    'rms_norm',
    'trunc_normal_init_',
    'EMAHelper',
    'stablemax_cross_entropy',
    'TRM_AVAILABLE',
]
