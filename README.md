# SCI-ARC: Structural Causal Invariance for Abstract Reasoning Corpus

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/pytorch-2.1+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-35%2F35%20passing-brightgreen.svg)]()

A novel approach combining **Structural Causal Invariance (SCI)** principles with **Tiny Recursive Models (TRM)** architecture for the Abstraction and Reasoning Corpus (ARC) benchmark.

## 🚀 Quick Start (Production)

```bash
# Clone from remote repository
git clone https://github.com/peymanrah/SCI-ARC.git
cd SCI-ARC

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .

# Download ARC data
mkdir -p data/arc-agi
# Download from https://github.com/fchollet/ARC-AGI

# Validate installation
python scripts/validate_sci_arc.py
python scripts/validate_trm_practices.py

# Train
python scripts/train.py --config configs/default.yaml

# Evaluate
python scripts/evaluate.py --checkpoint checkpoints/best_model.pt
```

## ✅ Validation Status

| Suite | Tests | Status |
|-------|-------|--------|
| Core SCI-ARC | 8/8 | ✅ PASS |
| TRM Best Practices | 27/27 | ✅ PASS |
| **Total** | **35/35** | ✅ **ALL PASS** |

## Overview

SCI-ARC addresses a key limitation in current neural approaches to abstract reasoning: the conflation of *what* transforms (content) with *how* it transforms (structure). By explicitly separating these two aspects and learning structure-invariant representations, SCI-ARC achieves better generalization to novel tasks.

### Key Features

- **Structure-Content Separation**: Explicit encoders for structural patterns vs. content information
- **Structural Contrastive Learning (SCL)**: Learn invariant structural representations across tasks with same transformation type. Uses a SimCLR-style projection head to prevent representation collapse.
- **Orthogonality Constraint**: Ensure structure and content representations are independent
- **Causal Binding**: Combine structure and content while preserving causal relationships
- **TRM-Style Recursive Refinement**: Iterative answer refinement with deep supervision

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           SCI-ARC Architecture                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐    ┌───────────────────────────────────────────────┐  │
│  │ Train Pairs  │───▶│            Grid Encoder                       │  │
│  │ {(In,Out)}   │    │  Color Embedding + 2D Sinusoidal Pos Enc      │  │
│  └──────────────┘    └───────────────────┬───────────────────────────┘  │
│                                          │                               │
│                    ┌─────────────────────┴─────────────────────┐        │
│                    │                                           │        │
│              ┌─────▼─────┐                             ┌──────▼──────┐  │
│              │ Structural│                             │   Content   │  │
│              │  Encoder  │                             │   Encoder   │  │
│              │           │                             │             │  │
│              │  Cross-   │                             │   Object    │  │
│              │ Attention │                             │   Queries   │  │
│              │ Slots     │                             │             │  │
│              └─────┬─────┘                             └──────┬──────┘  │
│                    │                                          │         │
│              ┌─────▼─────┐       ⊥ Orthogonality       ┌─────▼─────┐   │
│              │  z_struct │◄──────────────────────────▶│ z_content │   │
│              └─────┬─────┘                             └─────┬─────┘   │
│                    │                                          │         │
│                    └───────────────┬──────────────────────────┘         │
│                                    │                                    │
│                            ┌───────▼───────┐                           │
│                            │Causal Binding │                           │
│                            │               │                           │
│                            │    z_task     │                           │
│                            └───────┬───────┘                           │
│                                    │                                    │
│  ┌──────────────┐          ┌───────▼───────┐                           │
│  │  Test Input  │─────────▶│   Recursive   │                           │
│  └──────────────┘          │  Refinement   │                           │
│                            │  (TRM-style)  │                           │
│                            │               │                           │
│                            │  H_cycles x   │                           │
│                            │  L_cycles     │                           │
│                            └───────┬───────┘                           │
│                                    │                                    │
│                            ┌───────▼───────┐                           │
│                            │ Predicted Out │                           │
│                            └───────────────┘                           │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Installation

```bash
# Clone repository
git clone https://github.com/sci-arc/sci-arc.git
cd sci-arc

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Quick Start

### Training

```bash
# Train with default configuration
python scripts/train.py --config configs/default.yaml

# Train small model for quick experiments
python scripts/train.py --config configs/small.yaml

# Train with custom data path
python scripts/train.py --config configs/default.yaml data.arc_dir=/path/to/arc

# Resume training from checkpoint
python scripts/train.py --config configs/default.yaml --resume checkpoints/checkpoint_epoch_50.pt
```

### Evaluation

```bash
# Evaluate trained model
python scripts/evaluate.py --checkpoint checkpoints/best_model.pt --data ./data/arc-agi

# Generate visualizations
python scripts/evaluate.py --checkpoint checkpoints/best_model.pt --visualize

# Generate submission file
python scripts/evaluate.py --checkpoint checkpoints/best_model.pt --save-predictions
```

### Comparison with TRM

```bash
# Compare SCI-ARC with TRM
python scripts/compare_trm.py \
    --sci-arc-checkpoint checkpoints/best_model.pt \
    --trm-predictions /path/to/trm_predictions.json \
    --output ./comparison_results
```

## Configuration

Configurations are YAML files in `configs/`. Key parameters:

```yaml
model:
  hidden_dim: 256        # Hidden dimension
  num_structure_slots: 8 # Number of structure prototypes
  max_objects: 16        # Maximum content objects
  H_cycles: 3            # Outer refinement cycles
  L_cycles: 4            # Inner refinement cycles

training:
  learning_rate: 3e-4
  batch_size: 32
  scl_weight: 0.1        # Structural contrastive loss weight
  ortho_weight: 0.01     # Orthogonality loss weight
  projection_dim: 128    # SCL projection head dimension (prevents representation collapse)
```

## Data

SCI-ARC supports multiple ARC datasets:

1. **ARC-AGI-1**: Original ARC training/evaluation set
2. **ARC-AGI-2**: Extended evaluation set
3. **RE-ARC**: Synthetic tasks from DSL (recommended for training)
4. **ConceptARC**: Categorized by concept type

Place data in `./data/arc-agi/training/` and `./data/arc-agi/evaluation/`.

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_models.py -v

# Run with coverage
pytest tests/ --cov=sci_arc --cov-report=html
```

## Project Structure

```
sci-arc/
├── sci_arc/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── grid_encoder.py        # Grid → embeddings
│   │   ├── structural_encoder.py  # Extract structure
│   │   ├── content_encoder.py     # Extract content
│   │   ├── causal_binding.py      # Bind structure + content
│   │   ├── recursive_refinement.py # TRM-style decoder
│   │   └── sci_arc.py             # Complete model
│   ├── training/
│   │   ├── __init__.py
│   │   ├── losses.py              # SCL, orthogonality
│   │   └── trainer.py             # Training loop
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py             # ARC dataset
│   │   └── transform_families.py  # Transformation labels
│   └── evaluation/
│       ├── __init__.py
│       ├── evaluator.py           # Evaluation pipeline
│       └── metrics.py             # ARC metrics
├── configs/
│   ├── default.yaml
│   ├── small.yaml
│   └── large.yaml
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── compare_trm.py
├── tests/
│   ├── test_models.py
│   ├── test_training.py
│   ├── test_data.py
│   └── test_integration.py
├── requirements.txt
├── setup.py
└── README.md
```

## Citation

If you use SCI-ARC in your research, please cite:

```bibtex
@article{sciarc2024,
  title={Structural Causal Invariance for Abstract Reasoning},
  author={SCI-ARC Team},
  journal={arXiv preprint},
  year={2024}
}
```

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- [ARC Prize](https://arcprize.org/) for the benchmark
- [TRM](https://github.com/da03/TRM) for the recursive model architecture
- [SCI Paper](https://arxiv.org/abs/xxxx) for structural causal invariance theory

## TRM Best Practices Implemented

SCI-ARC incorporates all best practices from the TinyRecursiveModels codebase:

| Feature | Description |
|---------|-------------|
| 8 Dihedral Transforms | Full D4 symmetry group for augmentation |
| Color Permutation | 9! permutations (colors 1-9, 0 fixed) |
| Translational Augmentation | Random grid translation |
| stablemax_cross_entropy | Numerically stable loss function |
| EMAHelper | Exponential moving average for weights |
| Augmentation Voting | Test-time augmentation with inverse transforms |
| TRM Token Format | PAD=0, EOS=1, colors=2-11, vocab_size=12 |
| Memory-Efficient Training | H_cycles-1 without gradients |

## Model Parameters

- **SCI-ARC**: 7.11M parameters (1.02x TRM)
- **TRM Reference**: 7.00M parameters
