# TRM Baseline - Original Implementation

## CRITICAL: Fair Comparison Guarantee

This directory provides a **thin wrapper** that imports from the **ORIGINAL UNMODIFIED** TRM implementation located in `TinyRecursiveModels-main/`.

**We do NOT modify the original TRM code in any way.**

This ensures a fair scientific comparison between SCI-ARC and TRM as published by the original authors.

## Original Source

- **Repository**: https://github.com/SamsungSAILMontreal/TinyRecursiveModels
- **Paper**: "Less is More: Recursive Reasoning with Tiny Networks" (https://arxiv.org/abs/2510.04871)
- **Authors**: Samsung SAIL Montreal
- **License**: Apache 2.0 (see original repository)

## Directory Structure

```
SCI-ARC/
├── TinyRecursiveModels-main/      # ORIGINAL UNMODIFIED TRM code
│   ├── models/
│   │   ├── common.py              # Original utilities
│   │   ├── ema.py                 # Original EMA helper
│   │   ├── layers.py              # Original layers (Attention, SwiGLU, etc.)
│   │   ├── losses.py              # Original losses (stablemax_cross_entropy)
│   │   ├── sparse_embedding.py    # Original sparse embeddings
│   │   └── recursive_reasoning/
│   │       └── trm.py             # Original TRM model
│   ├── dataset/                   # Original data processing
│   ├── pretrain.py                # Original training script
│   └── ...
│
└── baselines/trm/
    └── __init__.py                # Thin wrapper (imports from original)
```

## Usage

```python
from baselines.trm import TRM, TRMConfig, TRM_AVAILABLE

if TRM_AVAILABLE:
    # Create TRM model using original implementation
    config = TRMConfig(
        batch_size=32,
        seq_len=900,
        puzzle_emb_ndim=256,
        num_puzzle_identifiers=1000,
        vocab_size=16,
        H_cycles=3,
        L_cycles=4,
        H_layers=2,
        L_layers=2,
        hidden_size=256,
        expansion=2.5,
        num_heads=8,
        pos_encodings='rope',
        halt_max_steps=10,
        halt_exploration_prob=0.1,
    )
    model = TRM(config)
else:
    print("TRM not available - download original repo first")
```

## Verification

To verify the original TRM is unmodified, you can:

1. Download fresh from GitHub:
   ```bash
   git clone https://github.com/SamsungSAILMontreal/TinyRecursiveModels.git TinyRecursiveModels-fresh
   ```

2. Compare with our copy:
   ```bash
   diff -r TinyRecursiveModels-main/ TinyRecursiveModels-fresh/
   # Should show NO differences (except .git folder)
   ```

## Key Architecture (from original paper)

- **7M parameters** - extremely small for the task complexity
- **Hierarchical reasoning** with H (high-level) and L (low-level) state recurrence
- **ACT (Adaptive Computation Time)** with Q-learning based halting
- **Post-norm architecture** with RMSNorm
- **RoPE positional encodings** for sequence positions
- **Sparse puzzle embeddings** for task-specific context

## Published Performance

On ARC-AGI benchmark (as reported in original paper):
- ARC-AGI-1: ~45% task accuracy
- ARC-AGI-2: ~8% task accuracy

## Citation

If you use TRM in your research, please cite the original authors:

```bibtex
@article{trm2024,
  title={Less is More: Recursive Reasoning with Tiny Networks},
  author={Samsung SAIL Montreal},
  year={2024},
  url={https://github.com/SamsungSAILMontreal/TinyRecursiveModels}
}
```
