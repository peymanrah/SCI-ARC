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
    # Main TRM model
    from models.recursive_reasoning.trm import (
        TinyRecursiveReasoningModel_ACTV1,
        TinyRecursiveReasoningModel_ACTV1Config,
        TinyRecursiveReasoningModel_ACTV1Carry,
        TinyRecursiveReasoningModel_ACTV1InnerCarry,
    )
    
    # HRM model  
    from models.recursive_reasoning.hrm import (
        HierarchicalReasoningModel_ACTV1 as HRM_Original,
        HierarchicalReasoningModel_ACTV1Config as HRMConfig_Original,
    )
    
    # Transformer baseline (original TRM uses Model_ACTV2 naming)
    from models.recursive_reasoning.transformers_baseline import (
        Model_ACTV2 as TransformersBaseline_Original,
        Model_ACTV2Config as TransformersBaselineConfig_Original,
    )
    
    # Layers
    from models.layers import (
        Attention,
        SwiGLU,
        RotaryEmbedding,
        CastedLinear,
        CastedEmbedding,
        rms_norm,
    )
    from models.sparse_embedding import CastedSparseEmbedding
    from models.common import trunc_normal_init_
    from models.ema import EMAHelper
    from models.losses import stablemax_cross_entropy
    
    # Convenient aliases
    TRM = TinyRecursiveReasoningModel_ACTV1
    TRMConfig = TinyRecursiveReasoningModel_ACTV1Config
    TRMCarry = TinyRecursiveReasoningModel_ACTV1Carry
    TRMInnerCarry = TinyRecursiveReasoningModel_ACTV1InnerCarry
    
    HRM = HRM_Original
    HRMConfig = HRMConfig_Original
    
    TransformerBaseline = TransformersBaseline_Original
    TransformerBaselineConfig = TransformersBaselineConfig_Original
    
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
    TRMCarry = None
    TRMInnerCarry = None
    HRM = _raise_import_error
    HRMConfig = None
    TransformerBaseline = _raise_import_error
    TransformerBaselineConfig = None


# Note: TRMLossHead was a custom addition in the old baselines/trm/
# The original TRM uses stablemax_cross_entropy directly
# For compatibility, we provide a simple wrapper
class TRMLossHead:
    """
    Compatibility wrapper for old TRMLossHead references.
    The original TRM uses stablemax_cross_entropy directly.
    """
    def __init__(self, vocab_size=12):
        self.vocab_size = vocab_size
    
    def __call__(self, logits, targets):
        if TRM_AVAILABLE:
            return stablemax_cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
        else:
            import torch.nn.functional as F
            return F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))


__all__ = [
    # Main exports
    'TRM',
    'TRMConfig',
    'TRMCarry',
    'TRMInnerCarry',
    'HRM',
    'HRMConfig',
    'TransformerBaseline',
    'TransformerBaselineConfig',
    # Original class names
    'TinyRecursiveReasoningModel_ACTV1',
    'TinyRecursiveReasoningModel_ACTV1Config',
    'TinyRecursiveReasoningModel_ACTV1Carry',
    'TinyRecursiveReasoningModel_ACTV1InnerCarry',
    # Layers
    'Attention',
    'SwiGLU',
    'RotaryEmbedding',
    'CastedLinear',
    'CastedEmbedding',
    'CastedSparseEmbedding',
    'rms_norm',
    'trunc_normal_init_',
    # Training utilities
    'EMAHelper',
    'stablemax_cross_entropy',
    'TRMLossHead',
    # Availability flag
    'TRM_AVAILABLE',
]
