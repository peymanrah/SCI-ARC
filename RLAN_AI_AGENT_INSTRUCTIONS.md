# RLAN AI Agent Instruction Document

## Production Training & Evaluation Guide

### Hardware Target
- **GPU**: NVIDIA RTX 3090 (24GB VRAM)
- **CPU**: 24 cores (48 virtual)
- **RAM**: 128GB
- **OS**: Windows

---

## ARCHITECTURE OVERVIEW

### RLAN vs CISL Comparison

| Feature | RLAN | CISL |
|---------|------|------|
| **Architecture** | Recursive Latent Attention Network | Structural Causal Invariance Learning |
| **Latent Space** | Discrete (VQ-style) | Continuous + Structure/Content separation |
| **Attention** | Multi-scale recursive | Causal binding |
| **Key Innovation** | Discrete reasoning via Gumbel-Softmax | Structure-Content orthogonality |
| **Parameters** | ~7.8M (base) | ~7.1M (default) |
| **Recursion** | Latent refinement cycles | H_cycles, L_cycles |

### Parameter Comparison

| Config | Params | Hidden Dim | Num Slots | Purpose |
|--------|--------|------------|-----------|---------|
| `rlan_small.yaml` | ~2M | 128 | 8 | Fast testing, CI/CD |
| `rlan_base.yaml` | ~7.8M | 256 | 16 | Production training |

---

## STEP 1: ENVIRONMENT SETUP

### 1.1 Clone Repository
```powershell
# PRODUCTION ENVIRONMENT: C:\Users\perahmat\Downloads\SCI-ARC on Windows
cd C:\Users\perahmat\Downloads\SCI-ARC
```

### 1.2 Create Virtual Environment
```powershell
# Create venv with Python 3.10+
python -m venv .venv

# Activate (Windows - PRODUCTION)
.\.venv\Scripts\activate
```

### 1.3 Install PyTorch with CUDA
```powershell
# CRITICAL: Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

### 1.4 Install Other Dependencies
```powershell
pip install -r requirements.txt
pip install -e .
```

### 1.5 Verify Installation
```powershell
# Verify GPU setup
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"

# Verify RLAN tests pass
python -m pytest tests/ -v

# Verify end-to-end flow
python scripts/verify_rlan_flow.py
```

**Expected Output:**
```
CUDA: True, GPU: NVIDIA GeForce RTX 3090
ALL TESTS PASSED - RLAN IS READY!
```

---

## STEP 2: DATA SETUP

### 2.1 Download ARC-AGI Dataset
```powershell
mkdir data
cd data

# Download ARC-AGI-1 (training + evaluation)
git clone https://github.com/fchollet/ARC-AGI.git arc-agi

cd ..
```

### 2.2 Verify Data Structure
```powershell
# Data structure should be:
# C:\...\SCI-ARC\data\arc-agi\data\training\
# C:\...\SCI-ARC\data\arc-agi\data\evaluation\
Get-ChildItem ".\data\arc-agi\data\" -Directory
```

### 2.3 Verify Data Count
```powershell
# Should show 400 tasks each
(Get-ChildItem ".\data\arc-agi\data\training\*.json").Count
(Get-ChildItem ".\data\arc-agi\data\evaluation\*.json").Count
```

---

## STEP 3: TRAINING

### 3.1 Full Production Training
```powershell
# PRODUCTION TRAINING COMMAND (Windows)
# Terminal logs are saved automatically to checkpoints/rlan_base/training_log_*.txt
.\.venv\Scripts\python.exe -u scripts/train_rlan.py `
    --config configs/rlan_base.yaml `
    logging.checkpoint_dir=./checkpoints/rlan_base `
    hardware.seed=42
```

### 3.2 Training with Custom Parameters
```powershell
# Override specific parameters on command line
.\.venv\Scripts\python.exe -u scripts/train_rlan.py `
    --config configs/rlan_base.yaml `
    training.batch_size=32 `
    training.max_epochs=100 `
    training.learning_rate=3e-4 `
    logging.checkpoint_dir=./checkpoints/rlan_custom
```

### 3.3 Resume Training from Checkpoint
```powershell
# Auto-resume from latest checkpoint in checkpoint_dir
.\.venv\Scripts\python.exe -u scripts/train_rlan.py `
    --config configs/rlan_base.yaml `
    --resume auto

# Resume from specific checkpoint file
.\.venv\Scripts\python.exe -u scripts/train_rlan.py `
    --config configs/rlan_base.yaml `
    --resume checkpoints/rlan_base/epoch_50.pt
```

### 3.4 Training with WandB Logging (Optional)
```powershell
# WandB is DISABLED by default (not installed in production)
# To enable WandB logging:
# 1. Install: pip install wandb
# 2. Login: wandb login
# 3. Run with override:
.\.venv\Scripts\python.exe -u scripts/train_rlan.py `
    --config configs/rlan_base.yaml `
    logging.use_wandb=true `
    logging.wandb_project=rlan-arc
```

### 3.5 Small Model Training (Fast Testing)
```powershell
# ~2M params, for quick tests or debugging
.\.venv\Scripts\python.exe -u scripts/train_rlan.py `
    --config configs/rlan_small.yaml `
    training.max_epochs=10 `
    logging.checkpoint_dir=./checkpoints/rlan_small
```

### 3.6 Force Fresh Start (Ignore Existing Checkpoints)
```powershell
.\.venv\Scripts\python.exe -u scripts/train_rlan.py `
    --config configs/rlan_base.yaml `
    --no-resume
```

### 3.7 CPU Training (Debug Only)
```powershell
.\.venv\Scripts\python.exe -u scripts/train_rlan.py `
    --config configs/rlan_small.yaml `
    --device cpu `
    training.max_epochs=2
```

---

## STEP 4: EVALUATION

### 4.1 Standard Evaluation
```powershell
# PRODUCTION EVALUATION COMMAND (Windows)
# Evaluation logs saved to output directory
.\.venv\Scripts\python.exe -u scripts/evaluate_rlan.py `
    --checkpoint checkpoints/rlan_base/best.pt `
    --data-path ./data/arc-agi/data/evaluation `
    --output ./evaluation_results/rlan_base_eval
```

### 4.2 Evaluation with Test-Time Augmentation (TTA)
```powershell
# TTA applies 8 dihedral transforms for ensemble predictions
.\.venv\Scripts\python.exe -u scripts/evaluate_rlan.py `
    --checkpoint checkpoints/rlan_base/best.pt `
    --data-path ./data/arc-agi/data/evaluation `
    --output ./evaluation_results/rlan_base_tta `
    --use-tta
```

### 4.3 Detailed Output with Per-Task Predictions
```powershell
# --detailed-output saves prediction vs reference for each task
.\.venv\Scripts\python.exe -u scripts/evaluate_rlan.py `
    --checkpoint checkpoints/rlan_base/best.pt `
    --data-path ./data/arc-agi/data/evaluation `
    --output ./evaluation_results/rlan_detailed `
    --detailed-output
```

### 4.4 Full Evaluation with All Options
```powershell
# Complete evaluation with TTA, visualization, and attention analysis
.\.venv\Scripts\python.exe -u scripts/evaluate_rlan.py `
    --checkpoint checkpoints/rlan_base/best.pt `
    --data-path ./data/arc-agi/data/evaluation `
    --output ./evaluation_results/rlan_full `
    --use-tta `
    --detailed-output `
    --visualize `
    --analyze-attention `
    --batch-size 16
```

### 4.5 Evaluate on Training Set (Debug)
```powershell
# For debugging - check if model is learning
.\.venv\Scripts\python.exe -u scripts/evaluate_rlan.py `
    --checkpoint checkpoints/rlan_base/best.pt `
    --data-path ./data/arc-agi/data/training `
    --output ./evaluation_results/rlan_train_debug
```

---

## STEP 5: ANALYSIS & REPORTING

### 5.1 Generate HTML Analysis Report
```powershell
.\.venv\Scripts\python.exe -u scripts/analyze_rlan_evaluation.py `
    --results ./evaluation_results/rlan_base_eval `
    --generate-html `
    --output ./evaluation_results/rlan_base_eval/analysis_report.html `
    --max-viz 50
```

### 5.2 Quick Results Summary
```powershell
# View metrics JSON
Get-Content ./evaluation_results/rlan_base_eval/evaluation_summary.json | ConvertFrom-Json
```

### 5.3 Record Evaluation Results
```powershell
# Expected output structure:
# evaluation_results/
# ├── evaluation_log_YYYYMMDD_HHMMSS.txt   # Full terminal output
# ├── evaluation_summary.json               # Aggregate metrics
# ├── detailed_predictions.json             # Per-task predictions (if --detailed-output)
# └── visualizations/                       # Images (if --visualize)
```

---

## STEP 6: REPRODUCIBILITY VERIFICATION

### 6.1 Run Deterministic Test
```powershell
# Train twice with same seed - results should be identical
.\.venv\Scripts\python.exe -u scripts/train_rlan.py `
    --config configs/rlan_small.yaml `
    hardware.seed=42 `
    hardware.deterministic=true `
    training.max_epochs=5 `
    logging.checkpoint_dir=./checkpoints/repro_run1

.\.venv\Scripts\python.exe -u scripts/train_rlan.py `
    --config configs/rlan_small.yaml `
    hardware.seed=42 `
    hardware.deterministic=true `
    training.max_epochs=5 `
    logging.checkpoint_dir=./checkpoints/repro_run2

# Compare checkpoints
.\.venv\Scripts\python.exe -c "import torch; m1 = torch.load('checkpoints/repro_run1/epoch_5.pt', weights_only=False); m2 = torch.load('checkpoints/repro_run2/epoch_5.pt', weights_only=False); print('MATCH!' if all(torch.equal(m1['model_state_dict'][k], m2['model_state_dict'][k]) for k in m1['model_state_dict']) else 'MISMATCH')"
```

### 6.2 Log Software Versions
```powershell
.\.venv\Scripts\python.exe -c "import torch; import numpy as np; import sys; print(f'Python: {sys.version}'); print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}'); print(f'NumPy: {np.__version__}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
```

---

## CONFIGURATION REFERENCE

### Configuration Files Summary

| Config | Params | Purpose | Use For |
|--------|--------|---------|---------|
| `rlan_base.yaml` | ~7.8M | Production Training | Main experiments |
| `rlan_small.yaml` | ~2M | Fast Testing | Debugging, CI/CD |

### Key Config Parameters (`configs/rlan_base.yaml`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model.hidden_dim` | 256 | Feature dimension (affects memory) |
| `model.num_slots` | 16 | Number of latent slots |
| `model.num_heads` | 8 | Attention heads |
| `training.batch_size` | 64 | Batch size (max ~128 on RTX 3090) |
| `training.max_epochs` | 250 | Total training epochs |
| `training.learning_rate` | 1e-4 | Initial learning rate |
| `training.use_ema` | true | Use EMA for stable evaluation |
| `training.ema_decay` | 0.999 | EMA decay rate |
| `training.grad_accumulation_steps` | 1 | Gradient accumulation |
| `device.mixed_precision` | true | Enable AMP (required for efficiency) |
| `logging.save_every` | 10 | Checkpoint interval (epochs) |
| `logging.log_every` | 1 | Log interval (epochs) |
| `logging.log_to_file` | true | Save terminal output to file |
| `logging.use_wandb` | false | WandB logging (disabled by default) |
| `hardware.seed` | 42 | Random seed |
| `hardware.deterministic` | false | Full determinism (slower) |

### Command-Line Override Examples

```powershell
# Training batch size
training.batch_size=32

# Learning rate
training.learning_rate=3e-4

# Max epochs
training.max_epochs=100

# Checkpoint directory
logging.checkpoint_dir=./checkpoints/my_experiment

# Enable WandB
logging.use_wandb=true

# Model hidden dimension
model.hidden_dim=512

# EMA decay
training.ema_decay=0.9999

# Seed
hardware.seed=123
```

### Critical Paths

| Path | Purpose |
|------|---------|
| `C:\Users\perahmat\Downloads\SCI-ARC\` | Project root (PRODUCTION) |
| `./data/arc-agi/data/training/` | Training data (400 tasks) |
| `./data/arc-agi/data/evaluation/` | Evaluation data (400 tasks) |
| `./checkpoints/rlan_base/` | RLAN base model checkpoints |
| `./checkpoints/rlan_small/` | RLAN small model checkpoints |
| `./evaluation_results/` | Evaluation outputs |
| `./configs/rlan_base.yaml` | Base config |
| `./configs/rlan_small.yaml` | Small config |

---

## TROUBLESHOOTING

### Out of Memory (OOM)
```powershell
# Reduce batch size with gradient accumulation to maintain effective batch size
.\.venv\Scripts\python.exe -u scripts/train_rlan.py `
    --config configs/rlan_base.yaml `
    training.batch_size=16 `
    training.grad_accumulation_steps=4
```

### CUDA Not Available
```powershell
# Verify CUDA installation
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with CUDA
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

### Reproducibility Issues
```powershell
# Ensure deterministic mode
.\.venv\Scripts\python.exe -u scripts/train_rlan.py `
    --config configs/rlan_base.yaml `
    hardware.deterministic=true `
    hardware.seed=42
```

### Training Not Improving
```powershell
# Try different learning rate
.\.venv\Scripts\python.exe -u scripts/train_rlan.py `
    --config configs/rlan_base.yaml `
    training.learning_rate=3e-4

# Check focal gamma (2.0 recommended for class imbalance)
# Edit configs/rlan_base.yaml:
# loss.focal_gamma: 2.0
```

### Multiprocessing Errors (Windows)
Set `num_workers: 0` in config (already default for Windows)

### Encoding Errors (Windows Terminal)
Already fixed - TeeLogger uses `errors='replace'` for Unicode safety

---

## DEBUGGING: TRAINING LOGS AND OUTPUTS

All terminal output during training is automatically saved:

```
checkpoints/rlan_base/
├── training_log_20241215_143022.txt   # Full terminal output
├── epoch_10.pt                         # Checkpoint at epoch 10
├── epoch_20.pt                         # Checkpoint at epoch 20
├── latest.pt                           # Most recent checkpoint
└── best.pt                             # Best validation checkpoint
```

**Key log contents:**
- Config dump
- Epoch progress with all losses (focal, entropy, sparsity, predicate, curriculum, deep_supervision)
- Learning rate schedule
- Temperature annealing
- Validation metrics

**View recent logs:**
```powershell
Get-Content -Tail 50 checkpoints/rlan_base/training_log_*.txt
```

---

## DEBUGGING: DETAILED EVALUATION OUTPUT

Use `--detailed-output` for per-task analysis:

```powershell
.\.venv\Scripts\python.exe -u scripts/evaluate_rlan.py `
    --checkpoint checkpoints/rlan_base/best.pt `
    --output ./evaluation_results/debug `
    --detailed-output
```

This creates:

```
evaluation_results/debug/
├── evaluation_log_20241215_150000.txt  # Full terminal output
├── evaluation_summary.json              # Aggregate metrics
└── detailed_predictions.json            # Per-task predictions
```

**Per-task details include:**
- Input/target/prediction grids
- Pixel accuracy per task
- Size match status
- Color distribution
- Diff positions (where model got wrong)

---

## EXPECTED RESULTS

### Training Metrics (per epoch)
| Metric | Start | End (Converged) |
|--------|-------|-----------------|
| Focal Loss | ~2.0 | ~0.5 |
| Total Loss | ~3.0 | ~1.0 |
| Temperature | 5.0 | 0.1 |

### Evaluation Metrics (target goals)
| Metric | Random Baseline | Target | Notes |
|--------|-----------------|--------|-------|
| Task Accuracy | 0% | 15-25% | Full task correct |
| Pixel Accuracy | ~10% | 80%+ | Per-pixel correct |
| Size Accuracy | ~10% | 90%+ | Output size correct |

### Training Time Estimates
| Config | GPU | Time per Epoch | Total (250 epochs) |
|--------|-----|----------------|-------------------|
| rlan_small | RTX 3090 | ~2-3 min | ~8-12 hours |
| rlan_base | RTX 3090 | ~4-5 min | ~17-21 hours |

---

## QUICK REFERENCE COMMANDS

```powershell
# === TRAINING ===
# Full production training
python scripts/train_rlan.py --config configs/rlan_base.yaml

# Resume from latest
python scripts/train_rlan.py --config configs/rlan_base.yaml --resume auto

# Custom parameters
python scripts/train_rlan.py --config configs/rlan_base.yaml training.batch_size=32 training.max_epochs=100

# Small model (fast test)
python scripts/train_rlan.py --config configs/rlan_small.yaml

# === EVALUATION ===
# Standard evaluation
python scripts/evaluate_rlan.py --checkpoint checkpoints/rlan_base/best.pt

# With TTA and detailed output
python scripts/evaluate_rlan.py --checkpoint checkpoints/rlan_base/best.pt --use-tta --detailed-output --output ./evaluation_results/full_eval

# === ANALYSIS ===
# Generate HTML report
python scripts/analyze_rlan_evaluation.py --results ./evaluation_results/full_eval --generate-html

# === TESTING ===
# All tests
python -m pytest tests/ -v

# End-to-end verify
python scripts/verify_rlan_flow.py

# Comprehensive RLAN tests
python scripts/test_rlan_comprehensive.py
```

---

## FILE LOCATIONS REFERENCE

| Category | Path |
|----------|------|
| **Configs** | |
| Base config | `configs/rlan_base.yaml` |
| Small config | `configs/rlan_small.yaml` |
| **Scripts** | |
| Training | `scripts/train_rlan.py` |
| Evaluation | `scripts/evaluate_rlan.py` |
| Analysis | `scripts/analyze_rlan_evaluation.py` |
| E2E Verify | `scripts/verify_rlan_flow.py` |
| Comprehensive Test | `scripts/test_rlan_comprehensive.py` |
| **Model** | |
| RLAN model | `sci_arc/models/rlan.py` |
| RLAN modules | `sci_arc/models/rlan_modules/` |
| Loss function | `sci_arc/training/rlan_loss.py` |
| **Evaluation** | |
| Metrics | `sci_arc/evaluation/metrics.py` |
| Visualizations | `sci_arc/evaluation/visualization.py` |
| **Outputs** | |
| Checkpoints | `checkpoints/rlan_base/` |
| Eval results | `evaluation_results/` |
| **Tests** | |
| Unit tests | `tests/` |

---

## PRODUCTION CHECKLIST

Before starting production training:

- [ ] GPU memory verified (`nvidia-smi`)
- [ ] Data paths verified (400 tasks each in training/evaluation)
- [ ] Virtual environment activated (`.\.venv\Scripts\activate`)
- [ ] Tests passing (`pytest tests/ -v`)
- [ ] Config reviewed (`configs/rlan_base.yaml`)
- [ ] Checkpoint directory writable
- [ ] Log file location verified
- [ ] End-to-end flow verified (`python scripts/verify_rlan_flow.py`)

---

## CHECKLIST FOR FULL EXPERIMENT

### Training Phase
- [ ] Base config reviewed
- [ ] Seed set for reproducibility (42)
- [ ] Training started with full logging
- [ ] Checkpoints being saved every N epochs
- [ ] Losses decreasing as expected
- [ ] EMA enabled (use_ema: true)

### Evaluation Phase
- [ ] Best checkpoint evaluated
- [ ] TTA evaluation done
- [ ] Detailed predictions saved
- [ ] Metrics match expectations

### Logging & Debugging
- [ ] Training logs saved (`training_log_*.txt`)
- [ ] Evaluation logs saved (`evaluation_log_*.txt`)
- [ ] Detailed predictions exported (`--detailed-output`)
- [ ] HTML report generated

---

**Last Updated**: 2024
**RLAN Version**: Production Ready  
**Hardware Verified**: RTX 3090 24GB VRAM
