"""
Generalization Modules for RLAN

These modules are designed to enable generalization to never-seen-before rules
in ARC-AGI 1 and ARC-AGI 2. They implement System 2 (deliberate reasoning) 
on top of the base RLAN System 1 (pattern matching).

KEY DESIGN PRINCIPLES:
1. MODULAR: Each component WRAPS RLAN, does not modify it
2. REMOVABLE: Can be deleted without affecting base RLAN codebase  
3. OPTIONAL: Enabled/disabled via config flags
4. FALLBACK: If enhancement fails, falls back to base RLAN

Modules:
- TEPS: Test-Time Exhaustive Program Search
  - Synthesizes programs from a DSL that explain all training pairs
  - Executes found program on test input
  - Falls back to RLAN if no program found

- NS-TEPS: Neuro-Symbolic TEPS (v2)
  - Object-level primitives with DSC-style object extraction
  - Program trace generation instead of pixel prediction
  - Compositional object transformations

- ConsistencyVerifier: Self-consistency verification
  - Checks if predicted transformation matches training patterns
  - Provides confidence scores for predictions

- HASR: Hindsight-Aware Solver Refinement
  - Test-time LoRA adaptation on pseudo-labels
  - Evolutionary test-time compute (ETTC) loop
  
- LOOVerifier: Leave-One-Out verification
  - Generalization confidence via LOO consistency
  - Ranks candidates by held-out prediction accuracy

Author: AI Research Assistant  
Date: January 2026
"""

from .teps import TEPS, TEPSConfig, PrimitiveLibrary, Program
from .consistency_verifier import ConsistencyVerifier, ConsistencyConfig
from .enhanced_inference import (
    EnhancedInference, 
    EnhancedInferenceConfig,
    run_enhanced_inference,
)
from .ns_teps import NSTEPS, NSTEPSConfig, ObjectExtractor, ObjectPrimitiveLibrary, ProgramTrace
from .hasr import HASR, HASRConfig, LoRALayer, AdaptiveRefinementModule
from .loo_verifier import LOOVerifier, LOOVerifierConfig, VerifierRanker
from .synoptic_rlan import SynopticRLAN, SRLANConfig, run_srlan_inference
from .primitive_head import (
    PrimitiveHead, 
    PrimitiveHeadConfig, 
    PrimitiveHeadLoss,
    PrimitiveEmbedding,
    ObjectScorer,
    ParameterPredictor,
    PRIMITIVE_NAME_TO_ID,
    PRIMITIVE_TYPE_MAPPING,
)
from .program_guided_training import (
    ProgramGuidedRLAN,
    ProgramGuidedConfig,
    ProgramCache,
    PseudoLabelGenerator,
    create_program_guided_rlan,
)

__all__ = [
    # TEPS - Program Synthesis
    "TEPS",
    "TEPSConfig", 
    "PrimitiveLibrary",
    "Program",
    # NS-TEPS - Neuro-Symbolic Program Synthesis
    "NSTEPS",
    "NSTEPSConfig",
    "ObjectExtractor",
    "ObjectPrimitiveLibrary",
    "ProgramTrace",
    # Consistency Verification
    "ConsistencyVerifier",
    "ConsistencyConfig",
    # HASR - Hindsight-Aware Solver Refinement
    "HASR",
    "HASRConfig",
    "LoRALayer",
    "AdaptiveRefinementModule",
    # LOO Verifier
    "LOOVerifier",
    "LOOVerifierConfig",
    "VerifierRanker",
    # Enhanced Inference Pipeline
    "EnhancedInference",
    "EnhancedInferenceConfig",
    "run_enhanced_inference",
    # Synoptic-RLAN v3.0
    "SynopticRLAN",
    "SRLANConfig",
    "run_srlan_inference",
    # PrimitiveHead - Program-Guided Training
    "PrimitiveHead",
    "PrimitiveHeadConfig",
    "PrimitiveHeadLoss",
    "PrimitiveEmbedding",
    "ObjectScorer",
    "ParameterPredictor",
    "PRIMITIVE_NAME_TO_ID",
    "PRIMITIVE_TYPE_MAPPING",
    # Program-Guided RLAN
    "ProgramGuidedRLAN",
    "ProgramGuidedConfig",
    "ProgramCache",
    "PseudoLabelGenerator",
    "create_program_guided_rlan",
]
