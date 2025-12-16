"""
SCI-ARC Inference Module

Provides competitive inference capabilities:
1. Stochastic Sampling (Monte Carlo Dropout, Temperature)
2. Test-Time Training (TTT)
3. Ensemble Voting with Consistency Verification

These modules are designed for ablation studies and can be
enabled/disabled via configuration.
"""

from .sampler import StochasticSampler, SamplingConfig
from .ttt import TTTAdapter, TTTConfig

# Backwards compatibility alias
TestTimeTrainer = TTTAdapter
from .ensemble import EnsemblePredictor, EnsembleConfig

__all__ = [
    'StochasticSampler',
    'SamplingConfig',
    'TestTimeTrainer', 
    'TTTConfig',
    'EnsemblePredictor',
    'EnsembleConfig',
]
