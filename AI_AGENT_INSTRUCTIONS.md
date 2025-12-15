# AI Agent Instructions: SCI-ARC Production Setup & TRM Comparison
# For Reproducible Results (Nature/NeurIPS Publication Standard)

## OVERVIEW

You are setting up SCI-ARC (Structural Causal Invariance for Abstract Reasoning Corpus)
in a production environment to:

1. **PHASE A**: Train TRM baseline (7M params) for comparison
2. **PHASE B**: Train SCI-ARC with **default config** (~7M params) - FAIR comparison
3. **PHASE C**: Train SCI-ARC with **competitive config** (~10-12M params) - Maximum performance
4. Generate reproducible results for Nature/NeurIPS publication

**Target Hardware:** NVIDIA RTX 3090 (24GB VRAM), CUDA 12.6
**Reproducibility:** Bit-exact reproducibility enabled

---

## EXPERIMENTAL DESIGN RATIONALE

### Why Two SCI-ARC Experiments?

To properly evaluate SCI-ARC's contribution, we need two comparisons:

| Experiment | Config | Params | Purpose |
|------------|--------|--------|---------|
| **Experiment 1** | `default.yaml` | ~7M | **Fair comparison** - Isolate impact of SCI architecture vs TRM |
| **Experiment 2** | `competitive.yaml` | ~10-12M | **Maximum performance** - Show scaling behavior |

### Architecture Comparison Table

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ARCHITECTURE COMPARISON: TRM vs SCI-ARC                  │
├─────────────────────┬─────────────┬─────────────────┬───────────────────────┤
│ Component           │ TRM (7M)    │ SCI-ARC Default │ SCI-ARC Competitive   │
│                     │             │ (7.1M)          │ (10-12M)              │
├─────────────────────┼─────────────┼─────────────────┼───────────────────────┤
│ hidden_dim          │ 256         │ 256             │ 384                   │
│ H_cycles            │ 3           │ 3               │ 4                     │
│ L_cycles            │ 4-6         │ 4               │ 6                     │
│ L_layers            │ 2           │ 2               │ 3                     │
│ Attention heads     │ 8           │ 8               │ 12                    │
├─────────────────────┼─────────────┼─────────────────┼───────────────────────┤
│ Structure Encoder   │ ✗ (none)    │ ✓ (8 slots)     │ ✓ (12 slots)          │
│ Content Encoder     │ ✗ (none)    │ ✓ (16 objects)  │ ✓ (20 objects)        │
│ Causal Binding      │ ✗ (none)    │ ✓ (8 heads)     │ ✓ (12 heads)          │
│ SCL Loss            │ ✗ (none)    │ ✓ (0.1 weight)  │ ✓ (0.15 weight)       │
├─────────────────────┼─────────────┼─────────────────┼───────────────────────┤
│ RE-ARC Data         │ ✗           │ ✗               │ ✓                     │
│ max_epochs          │ 100         │ 100             │ 150                   │
│ Training Time       │ ~12-24h     │ ~12-24h         │ ~36-48h               │
└─────────────────────┴─────────────┴─────────────────┴───────────────────────┘
```

### Expected Results Visualization

```
Task Accuracy (%) on ARC-AGI Evaluation Set
                                                    
    60% ─┬──────────────────────────────────────────  ← Target: 55%+
        │                                    ████████
    55% ─┤                                   ████████
        │                           ████████ ████████
    50% ─┤                  ████████ ████████ ████████  ← Target: 50%+
        │         ████████ ████████ ████████ ████████
    45% ─┤████████ ████████ ████████ ████████ ████████  ← TRM baseline
        │████████ ████████ ████████ ████████ ████████
    40% ─┤████████ ████████ ████████ ████████ ████████
        │████████ ████████ ████████ ████████ ████████
    35% ─┴────────┴────────┴────────┴────────┴────────
            TRM      SCI-ARC   SCI-ARC    Improvement
          Baseline   Default  Competitive  Analysis
           (7M)       (7M)     (10-12M)
                        
         ├────────────┤         ├──────────┤
          FAIR COMPARISON        SCALING ANALYSIS
          (same params)          (more capacity)
```

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
# Should show recent commits
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
ls -la data/arc-agi/data/
# Should contain: training/, evaluation/
```

### 2.3 Download RE-ARC (Required for Competitive Config)
```bash
# RE-ARC provides synthetic augmented data
cd data
git clone https://github.com/michaelhodel/re-arc.git rearc
cd ..
```

---

## STEP 3: PHASE A - TRAIN TRM BASELINE

This establishes the baseline to beat.

### 3.1 Train TRM Baseline
```bash
python scripts/train_trm_baseline.py \
    --config configs/trm_baseline.yaml \
    --data ./data/arc-agi/data \
    --output-dir ./checkpoints/trm_baseline \
    --seed 42
```

### 3.2 Evaluate TRM Baseline
```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/trm_baseline/best_model.pt \
    --model-type trm \
    --data ./data/arc-agi/data/evaluation \
    --output ./results/trm_baseline_eval.json \
    --use-voting \
    --num-attempts 2
```

### 3.3 Record TRM Results
```bash
# Expected: ~45% task accuracy (as reported in TRM paper)
cat ./results/trm_baseline_eval.json
```

---

## STEP 4: PHASE B - TRAIN SCI-ARC DEFAULT (Fair Comparison)

**CRITICAL: This uses ~7M parameters to match TRM exactly.**

This isolates the impact of SCI-ARC's architecture (Structure-Content separation, 
Causal Binding, SCL loss) without confounding from additional parameters.

### 4.1 Train SCI-ARC Default
```bash
python scripts/train.py \
    --config configs/default.yaml \
    data.arc_dir=./data/arc-agi/data \
    logging.checkpoint_dir=./checkpoints/sci_arc_default \
    logging.wandb_run_name=sci-arc-default-7M-seed42 \
    hardware.seed=42
```

### 4.2 Evaluate SCI-ARC Default
```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/sci_arc_default/best_model.pt \
    --data ./data/arc-agi/data/evaluation \
    --output ./results/sci_arc_default_eval.json \
    --use-voting \
    --num-attempts 2
```

### 4.3 Compare Default SCI-ARC vs TRM (Fair Comparison)
```bash
python scripts/compare_trm.py \
    --sci-arc-checkpoint checkpoints/sci_arc_default/best_model.pt \
    --trm-checkpoint checkpoints/trm_baseline/best_model.pt \
    --data ./data/arc-agi/data/evaluation \
    --output ./results/fair_comparison.json
```

**Expected Result:**
- SCI-ARC Default: ~48-50% task accuracy
- TRM Baseline: ~45% task accuracy
- **Improvement from SCI architecture: +3-5%**

---

## STEP 5: PHASE C - TRAIN SCI-ARC COMPETITIVE (Maximum Performance)

**This uses ~10-12M parameters for maximum ARC performance.**

### 5.1 Train SCI-ARC Competitive
```bash
python scripts/train.py \
    --config configs/competitive.yaml \
    data.arc_dir=./data/arc-agi/data \
    data.rearc_dir=./data/rearc \
    logging.checkpoint_dir=./checkpoints/sci_arc_competitive \
    logging.wandb_run_name=sci-arc-competitive-12M-seed42 \
    hardware.seed=42
```

### 5.2 Evaluate SCI-ARC Competitive
```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/sci_arc_competitive/best_model.pt \
    --data ./data/arc-agi/data/evaluation \
    --output ./results/sci_arc_competitive_eval.json \
    --use-voting \
    --num-attempts 2
```

### 5.3 Full Three-Way Comparison
```bash
python scripts/compare_trm.py \
    --sci-arc-checkpoint checkpoints/sci_arc_competitive/best_model.pt \
    --sci-arc-default-checkpoint checkpoints/sci_arc_default/best_model.pt \
    --trm-checkpoint checkpoints/trm_baseline/best_model.pt \
    --data ./data/arc-agi/data/evaluation \
    --output ./results/full_comparison.json
```

**Expected Result:**
- SCI-ARC Competitive: ~53-55% task accuracy
- SCI-ARC Default: ~48-50% task accuracy
- TRM Baseline: ~45% task accuracy
- **Improvement from scaling: +5-7% additional**

---

## STEP 6: REPRODUCIBILITY VERIFICATION

### 6.1 Run Deterministic Test
```bash
# Train twice with same seed - results should be identical
python scripts/train.py --config configs/default.yaml hardware.seed=42 \
    logging.checkpoint_dir=./checkpoints/repro_run1

python scripts/train.py --config configs/default.yaml hardware.seed=42 \
    logging.checkpoint_dir=./checkpoints/repro_run2

# Compare checkpoints (should be identical)
python -c "
import torch
m1 = torch.load('checkpoints/repro_run1/checkpoint_epoch_10.pt')
m2 = torch.load('checkpoints/repro_run2/checkpoint_epoch_10.pt')
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

## STEP 7: GENERATE PUBLICATION RESULTS

### 7.1 Generate All Metrics
```bash
python scripts/generate_publication_results.py \
    --sci-arc-checkpoint checkpoints/sci_arc_competitive/best_model.pt \
    --sci-arc-default-checkpoint checkpoints/sci_arc_default/best_model.pt \
    --trm-checkpoint checkpoints/trm_baseline/best_model.pt \
    --data ./data/arc-agi/data \
    --output ./results/publication/
```

### 7.2 Expected Results Table for Paper

| Model | Params | Task Acc | Pixel Acc | Pass@2 | Notes |
|-------|--------|----------|-----------|--------|-------|
| TRM Baseline | 7.0M | 45.0% | 78.0% | 47.0% | Published baseline |
| **SCI-ARC Default** | **7.1M** | **48-50%** | **82%** | **50-52%** | **Fair comparison** |
| **SCI-ARC Competitive** | **10-12M** | **53-55%** | **86%** | **55-58%** | **Maximum performance** |

### 7.3 Key Insights to Report

1. **Architectural Impact** (Default vs TRM):
   - Same parameter count (~7M)
   - +3-5% improvement from SCI architecture alone
   - Structure-Content separation is beneficial

2. **Scaling Behavior** (Competitive vs Default):
   - 1.5x parameters
   - +5-7% additional improvement
   - More structure slots and heads help

3. **Total Improvement** (Competitive vs TRM):
   - +8-10% task accuracy improvement
   - Same training paradigm, better architecture

---

## CONFIGURATION REFERENCE

### Configuration Files Summary

| Config | Params | Purpose | Use For |
|--------|--------|---------|---------|
| `default.yaml` | ~7.1M | Development & Fair Comparison | Experiment 1: Isolate SCI impact |
| `competitive.yaml` | ~10-12M | Maximum Performance | Experiment 2: Best results |
| `reproducible.yaml` | ~7.1M | Bit-exact Reproduction | Verification runs |
| `small.yaml` | ~3M | Fast Experiments | Debugging, CI/CD |
| `large.yaml` | ~15M | Research/Ablation | Capacity studies |

### Default vs Competitive Config Comparison

| Parameter | Default (7M) | Competitive (10-12M) | Impact |
|-----------|--------------|----------------------|--------|
| `hidden_dim` | 256 | 384 | +50% capacity |
| `H_cycles` | 3 | 4 | Deeper recursion |
| `L_cycles` | 4 | 6 | More refinement |
| `L_layers` | 2 | 3 | Richer layers |
| `structure_slots` | 8 | 12 | More patterns |
| `num_heads` | 8 | 12 | Finer attention |
| `max_epochs` | 100 | 150 | Full convergence |
| `scl_weight` | 0.1 | 0.15 | Stronger SCL |
| `rearc_dir` | null | ./data/rearc | More data |

### Critical Paths

| Path | Purpose |
|------|---------|
| `./data/arc-agi/data/training/` | Training data |
| `./data/arc-agi/data/evaluation/` | Evaluation data |
| `./data/rearc/` | RE-ARC synthetic data |
| `./checkpoints/trm_baseline/` | TRM baseline models |
| `./checkpoints/sci_arc_default/` | SCI-ARC default models |
| `./checkpoints/sci_arc_competitive/` | SCI-ARC competitive models |
| `./results/` | Evaluation results |

---

## TROUBLESHOOTING

### Out of Memory (OOM)
```bash
# For competitive config, reduce batch size
python scripts/train.py --config configs/competitive.yaml training.batch_size=16

# Or increase gradient accumulation
python scripts/train.py --config configs/competitive.yaml \
    training.batch_size=16 \
    training.grad_accumulation_steps=4
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
python scripts/train.py --config configs/default.yaml \
    hardware.deterministic=true \
    hardware.seed=42

# Check cuDNN settings
python -c "
import torch
print(f'cuDNN deterministic: {torch.backends.cudnn.deterministic}')
print(f'cuDNN benchmark: {torch.backends.cudnn.benchmark}')
"
```

---

## CHECKLIST FOR PUBLICATION

### Phase A: TRM Baseline
- [ ] TRM baseline trained with seed=42
- [ ] TRM evaluated on ARC-AGI evaluation set
- [ ] TRM results recorded (~45% expected)

### Phase B: Fair Comparison (Default Config)
- [ ] SCI-ARC default trained with seed=42
- [ ] SCI-ARC default evaluated
- [ ] Comparison vs TRM (same param count)
- [ ] Improvement from architecture quantified

### Phase C: Maximum Performance (Competitive Config)
- [ ] RE-ARC data downloaded
- [ ] SCI-ARC competitive trained with seed=42
- [ ] SCI-ARC competitive evaluated
- [ ] Scaling improvement quantified

### Reproducibility & Publication
- [ ] Environment with exact versions documented
- [ ] Reproducibility verified (two runs match)
- [ ] Three-way comparison generated
- [ ] Software versions logged
- [ ] Results exported to JSON
- [ ] Figures generated

---

## CONTACT

Repository: https://github.com/peymanrah/SCI-ARC
