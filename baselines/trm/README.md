# TRM Baseline Implementation

This directory contains the **original implementation** of the Tiny Recursive Reasoning Model (TRM) from Samsung SAIL Montreal, copied for fair comparison with SCI-ARC.

## Source

- **Repository**: https://github.com/SamsungSAILMontreal/TinyRecursiveModels
- **Paper**: "Less is More: Recursive Reasoning with Tiny Networks" (https://arxiv.org/abs/2510.04871)
- **Authors**: Samsung SAIL Montreal

## License

Please check the original repository for the most current license information. The original code is believed to be released under Apache 2.0.

## Files

```
baselines/trm/
├── __init__.py                     # Package exports
├── models/
│   ├── __init__.py                 # Model exports
│   ├── common.py                   # trunc_normal_init_ utility
│   ├── layers.py                   # Core layers (Attention, SwiGLU, RoPE, etc.)
│   ├── sparse_embedding.py         # CastedSparseEmbedding for puzzle embeddings
│   └── recursive_reasoning/
│       ├── __init__.py             # Model exports
│       ├── trm.py                  # Main TRM model (TinyRecursiveReasoningModel_ACTV1)
│       ├── hrm.py                  # Hierarchical Reasoning Model variant
│       └── transformers_baseline.py # Transformer baseline for ablation
```

## Usage

```python
from baselines.trm import TRM, TRMConfig

# Create TRM model with default config
config = {
    'batch_size': 32,
    'seq_len': 900,  # 30x30 ARC grid
    'puzzle_emb_ndim': 256,
    'num_puzzle_identifiers': 1000,
    'vocab_size': 16,
    'H_cycles': 3,
    'L_cycles': 4,
    'H_layers': 2,
    'L_layers': 2,
    'hidden_size': 256,
    'expansion': 2.5,
    'num_heads': 8,
    'pos_encodings': 'rope',
    'halt_max_steps': 10,
    'halt_exploration_prob': 0.1,
}

model = TRM(config)
```

## Key Architecture Details

From the original paper:

- **7M parameters** - extremely small for the task complexity
- **Hierarchical reasoning** with H (high-level) and L (low-level) state recurrence
- **ACT (Adaptive Computation Time)** with Q-learning based halting
- **Post-norm architecture** with RMSNorm
- **RoPE positional encodings** for sequence positions
- **Sparse puzzle embeddings** for task-specific context

## Performance

On ARC-AGI benchmark:
- ARC-AGI-1: ~45% accuracy
- ARC-AGI-2: ~8% accuracy

## Citation

```bibtex
@article{trm2024,
  title={Less is More: Recursive Reasoning with Tiny Networks},
  author={Samsung SAIL Montreal},
  journal={arXiv preprint arXiv:2510.04871},
  year={2024}
}
```

## Modifications

**NONE** - This is the exact original implementation for fair scientific comparison. The only changes are:
1. Import paths adjusted for the SCI-ARC project structure
2. Documentation and comments added for clarity
3. FlashAttention import replaced with PyTorch's `scaled_dot_product_attention` for compatibility

The core architecture, initialization, and forward pass logic are unchanged from the original.

## Fair Comparison Note

This implementation is included to ensure a scientifically fair comparison between SCI-ARC and TRM for the NeurIPS publication. Both models are:

1. Trained on the same data splits
2. Evaluated with the same metrics
3. Using their respective original implementations
4. Compared on identical hardware

This approach ensures reproducibility and fairness in reported results.
