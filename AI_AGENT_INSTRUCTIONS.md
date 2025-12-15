# AI Agent Instructions: SCI-ARC Production Setup & TRM Comparison
# For Reproducible Results (Nature/NeurIPS Publication Standard)

## OVERVIEW

You are setting up SCI-ARC (Structural Causal Invariance for Abstract Reasoning Corpus)
in a production environment to:
1. Train SCI-ARC from scratch
2. Evaluate on ARC benchmark
3. Compare results with TRM (Tiny Recursive Models) baseline
4. Generate reproducible results for publication

**Target Hardware:** NVIDIA RTX 3090 (24GB VRAM), CUDA 12.6
**Expected Training Time:** ~24-48 hours for full training
**Reproducibility:** Bit-exact reproducibility enabled

---

## STEP 1: ENVIRONMENT SETUP

### 1.1 Clone Repository
```bash
cd /home/user/projects  # or your preferred directory
git clone https://github.com/peymanrah/SCI-ARC.git
cd SCI-ARC
```

### 1.2 Verify Repository
```bash
git log --oneline -5
# Should show commits including "Add GPU/CUDA 12.6 support for RTX 3090"
```

### 1.3 Create Virtual Environment
```bash
# Create venv with Python 3.10 (recommended for reproducibility)
python3.10 -m venv .venv

# Activate (Linux/Mac)
source .venv/bin/activate

# Activate (Windows)
# .venv\Scripts\activate
```

### 1.4 Install PyTorch with CUDA 12.6
```bash
# CRITICAL: Install PyTorch with CUDA 12.6 support
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu126
```

### 1.5 Install Other Dependencies
```bash
pip install -r requirements.txt
pip install -e .
```

### 1.6 Verify Installation
```bash
# Verify GPU setup
python scripts/verify_gpu_setup.py

# Verify SCI-ARC (8/8 tests should pass)
python scripts/validate_sci_arc.py

# Verify TRM best practices (27/27 tests should pass)
python scripts/validate_trm_practices.py
```

**Expected Output:**
```
Total: 8/8 test suites passed
ALL VALIDATIONS PASSED - SCI-ARC IS READY!

Passed: 27
Failed: 0
ALL TRM BEST PRACTICES CORRECTLY IMPLEMENTED!
```

---

## STEP 2: DATA SETUP

### 2.1 Download ARC-AGI Dataset
```bash
mkdir -p data
cd data

# Download ARC-AGI-1 (training + evaluation)
git clone https://github.com/fchollet/ARC-AGI.git arc-agi

cd ..
```

### 2.2 Verify Data Structure
```bash
ls -la data/arc-agi/
# Should contain: data/training/, data/evaluation/
```

### 2.3 (Optional) Download RE-ARC for Additional Training Data
```bash
# RE-ARC provides synthetic augmented data
cd data
git clone https://github.com/michaelhodel/re-arc.git rearc
cd ..
```

---

## STEP 3: TRAINING SCI-ARC

### 3.1 Training Command (Reproducible)
```bash
python scripts/train.py \
    --config configs/reproducible.yaml \
    data.arc_dir=./data/arc-agi/data
```

### 3.2 Training with Weights & Biases Logging
```bash
# First, login to W&B
wandb login

# Then train with logging
python scripts/train.py \
    --config configs/reproducible.yaml \
    data.arc_dir=./data/arc-agi/data \
    logging.use_wandb=true \
    logging.wandb_project=sci-arc-publication \
    logging.wandb_run_name=sci-arc-seed42-v1
```

### 3.3 Training with Custom Seed (for variance analysis)
```bash
# Run 1: seed=42
python scripts/train.py --config configs/reproducible.yaml hardware.seed=42

# Run 2: seed=123
python scripts/train.py --config configs/reproducible.yaml hardware.seed=123

# Run 3: seed=456
python scripts/train.py --config configs/reproducible.yaml hardware.seed=456
```

### 3.4 Resume Training from Checkpoint
```bash
python scripts/train.py \
    --config configs/reproducible.yaml \
    --resume checkpoints/checkpoint_epoch_50.pt
```

### 3.5 Monitor Training
```bash
# Check GPU usage
nvidia-smi -l 1

# Check training logs
tail -f outputs/train.log
```

---

## STEP 4: EVALUATION

### 4.1 Evaluate SCI-ARC on ARC-AGI Evaluation Set
```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/best_model.pt \
    --data ./data/arc-agi/data/evaluation \
    --output ./results/sci_arc_eval.json \
    --use-voting \
    --num-attempts 2
```

### 4.2 Evaluate with All Metrics
```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/best_model.pt \
    --data ./data/arc-agi/data/evaluation \
    --output ./results/sci_arc_full_eval.json \
    --use-voting \
    --num-attempts 2 \
    --compute-all-metrics
```

### 4.3 Generate Submission File (Kaggle Format)
```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/best_model.pt \
    --data ./data/arc-agi/data/evaluation \
    --output ./results/submission.json \
    --format kaggle
```

---

## STEP 5: TRM BASELINE COMPARISON

### 5.1 Train TRM Baseline
```bash
# SCI-ARC includes TRM baseline in baselines/trm/
python scripts/train_trm_baseline.py \
    --config configs/trm_baseline.yaml \
    --data ./data/arc-agi/data
```

### 5.2 Evaluate TRM Baseline
```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/trm_baseline/best_model.pt \
    --model-type trm \
    --data ./data/arc-agi/data/evaluation \
    --output ./results/trm_eval.json
```

### 5.3 Compare SCI-ARC vs TRM
```bash
python scripts/compare_trm.py \
    --sci-arc-checkpoint checkpoints/best_model.pt \
    --trm-checkpoint checkpoints/trm_baseline/best_model.pt \
    --data ./data/arc-agi/data/evaluation \
    --output ./results/comparison.json
```

---

## STEP 6: REPRODUCIBILITY VERIFICATION

### 6.1 Run Deterministic Test
```bash
# Train twice with same seed - results should be identical
python scripts/train.py --config configs/reproducible.yaml hardware.seed=42 \
    logging.checkpoint_dir=./checkpoints/run1

python scripts/train.py --config configs/reproducible.yaml hardware.seed=42 \
    logging.checkpoint_dir=./checkpoints/run2

# Compare checkpoints (should be identical)
python -c "
import torch
m1 = torch.load('checkpoints/run1/checkpoint_epoch_10.pt')
m2 = torch.load('checkpoints/run2/checkpoint_epoch_10.pt')
for k in m1['model_state_dict']:
    if not torch.equal(m1['model_state_dict'][k], m2['model_state_dict'][k]):
        print(f'MISMATCH: {k}')
        break
else:
    print('REPRODUCIBILITY VERIFIED: All weights match!')
"
```

### 6.2 Log Software Versions
```bash
python -c "
import torch
import numpy as np
import sys

print('=== Software Versions for Publication ===')
print(f'Python: {sys.version}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
print(f'cuDNN: {torch.backends.cudnn.version()}')
print(f'NumPy: {np.__version__}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
"
```

---

## STEP 7: PUBLICATION RESULTS

### 7.1 Generate Publication Metrics
```bash
python scripts/generate_publication_results.py \
    --sci-arc-checkpoint checkpoints/best_model.pt \
    --trm-checkpoint checkpoints/trm_baseline/best_model.pt \
    --data ./data/arc-agi/data \
    --output ./results/publication/
```

### 7.2 Expected Metrics to Report

| Metric | SCI-ARC | TRM | Description |
|--------|---------|-----|-------------|
| Task Accuracy | X% | Y% | % of tasks fully correct |
| Pixel Accuracy | X% | Y% | % of pixels correct |
| Pass@2 | X% | Y% | % correct with 2 attempts |
| Parameters | 7.11M | 7.00M | Model size |
| Training Time | Xh | Yh | Time to convergence |

### 7.3 Generate Figures
```bash
# Training curves
python scripts/plot_training.py --log ./outputs/train.log --output ./figures/

# Comparison plots
python scripts/plot_comparison.py \
    --sci-arc-results ./results/sci_arc_eval.json \
    --trm-results ./results/trm_eval.json \
    --output ./figures/comparison/
```

---

## CONFIGURATION REFERENCE

### Key Configuration Files

| File | Purpose |
|------|---------|
| `configs/reproducible.yaml` | Reproducible training (use this) |
| `configs/default.yaml` | Default settings |
| `configs/small.yaml` | Quick experiments |
| `configs/large.yaml` | Maximum performance |

### Key Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `hidden_dim` | 256 | Model width |
| `H_cycles` | 3 | Recursive depth |
| `L_cycles` | 4 | Refinement iterations |
| `batch_size` | 32 | Fits in 24GB VRAM |
| `learning_rate` | 3e-4 | With warmup |
| `max_epochs` | 100 | Full training |
| `seed` | 42 | Reproducibility |
| `deterministic` | true | Bit-exact results |

### Critical Paths

| Path | Purpose |
|------|---------|
| `./data/arc-agi/data/training/` | Training data |
| `./data/arc-agi/data/evaluation/` | Evaluation data |
| `./checkpoints/` | Model checkpoints |
| `./results/` | Evaluation results |
| `./outputs/` | Training logs |

---

## TROUBLESHOOTING

### Out of Memory (OOM)
```bash
# Reduce batch size
python scripts/train.py --config configs/reproducible.yaml training.batch_size=16

# Or increase gradient accumulation
python scripts/train.py --config configs/reproducible.yaml \
    training.batch_size=16 \
    training.grad_accumulation_steps=2
```

### CUDA Not Available
```bash
# Verify CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with CUDA
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

### Reproducibility Issues
```bash
# Ensure deterministic mode
python scripts/train.py --config configs/reproducible.yaml \
    hardware.deterministic=true \
    hardware.seed=42

# Check cuDNN
python -c "
import torch
print(f'cuDNN deterministic: {torch.backends.cudnn.deterministic}')
print(f'cuDNN benchmark: {torch.backends.cudnn.benchmark}')
"
```

---

## CHECKLIST FOR PUBLICATION

- [ ] Environment created with exact versions
- [ ] ARC-AGI data downloaded
- [ ] SCI-ARC trained with seed=42
- [ ] TRM baseline trained with seed=42
- [ ] Reproducibility verified (two runs match)
- [ ] Evaluation completed on both models
- [ ] Comparison metrics generated
- [ ] Software versions logged
- [ ] Results exported to JSON
- [ ] Figures generated

---

## CONTACT

Repository: https://github.com/peymanrah/SCI-ARC
