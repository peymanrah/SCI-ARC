# =============================================================================
# TRM Baseline Package
# Source: https://github.com/SamsungSAILMontreal/TinyRecursiveModels
# Authors: Samsung SAIL Montreal
# 
# This package contains the original TRM implementation for fair comparison
# with SCI-ARC in the NeurIPS publication.
# =============================================================================

from baselines.trm.models.recursive_reasoning.trm import (
    TinyRecursiveReasoningModel_ACTV1,
    TinyRecursiveReasoningModel_ACTV1Config,
    TinyRecursiveReasoningModel_ACTV1Carry,
    TRM,
    TRMConfig,
    TRMCarry,
)

from baselines.trm.models.recursive_reasoning.hrm import (
    HierarchicalReasoningModel_ACTV1,
    HierarchicalReasoningModel_ACTV1Config,
    HRM,
    HRMConfig,
)

from baselines.trm.models.recursive_reasoning.transformers_baseline import (
    Model_ACTV2,
    Model_ACTV2Config,
    TransformerBaseline,
    TransformerBaselineConfig,
)

from baselines.trm.loss_head import (
    TRMLossHead,
    TRMWithLoss,
)

__all__ = [
    # Main TRM models
    'TinyRecursiveReasoningModel_ACTV1',
    'TinyRecursiveReasoningModel_ACTV1Config',
    'TinyRecursiveReasoningModel_ACTV1Carry',
    'TRM',
    'TRMConfig',
    'TRMCarry',
    
    # HRM models
    'HierarchicalReasoningModel_ACTV1',
    'HierarchicalReasoningModel_ACTV1Config',
    'HRM',
    'HRMConfig',
    
    # Transformer baseline
    'Model_ACTV2',
    'Model_ACTV2Config',
    'TransformerBaseline',
    'TransformerBaselineConfig',
    
    # Training utilities
    'TRMLossHead',
    'TRMWithLoss',
]
