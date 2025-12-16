# RLAN AI Agent Instruction Document

## Production Training & Evaluation Guide

### Hardware Target
- **GPU**: NVIDIA RTX 3090 (24GB VRAM)
- **CPU**: 24 cores (48 virtual)
- **RAM**: 128GB
- **OS**: Windows

---

## 1. ENVIRONMENT SETUP

### 1.1 Prerequisites
```powershell
# Navigate to project root
cd C:\Users\perahmat\Downloads\SCI-ARC

# Activate virtual environment
.\.venv\Scripts\activate

# Verify Python version
python --version  # Should be Python 3.10+
```

### 1.2 Verify Installation
```powershell
# Run tests to verify all modules work
python -m pytest tests/ -v

# Check CUDA availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
```

---

## 2. DATA SETUP

### 2.1 Expected Directory Structure
```
data/
└── arc-agi/
    └── data/
        ├── training/          # 400 training tasks (JSON files)
        │   ├── task1.json
        │   ├── task2.json
        │   └── ...
        └── evaluation/        # 400 evaluation tasks (JSON files)
            ├── task1.json
            └── ...
```

### 2.2 Verify Data Paths
```powershell
# Check training data exists
Get-ChildItem "./data/arc-agi/data/training/*.json" | Measure-Object

# Check evaluation data exists
Get-ChildItem "./data/arc-agi/data/evaluation/*.json" | Measure-Object
```

---

## 3. TRAINING COMMANDS

### 3.1 Standard Training (Base Model - 7.8M params)
```powershell
# Full production training
python scripts/train_rlan.py --config configs/rlan_base.yaml
```

**Expected Output**:
- Training logs to console
- Log file saved to `checkpoints/rlan_base/training_log_YYYYMMDD_HHMMSS.txt`
- Checkpoints at `checkpoints/rlan_base/epoch_N.pt`
- Best model at `checkpoints/rlan_base/best.pt`

### 3.2 Small Model Training (2M params - for testing)
```powershell
python scripts/train_rlan.py --config configs/rlan_small.yaml
```

### 3.3 Resume Training from Checkpoint
```powershell
# Auto-resume from latest checkpoint
python scripts/train_rlan.py --config configs/rlan_base.yaml --resume auto

# Resume from specific checkpoint
python scripts/train_rlan.py --config configs/rlan_base.yaml --resume checkpoints/rlan_base/epoch_50.pt
```

### 3.4 Training with WandB Logging (Optional - Not installed by default)
```powershell
# WandB is DISABLED by default in production
# To enable (only if wandb is installed):
# 1. Install wandb: pip install wandb
# 2. Login: wandb login  
# 3. Edit configs/rlan_base.yaml: set use_wandb: true
# 4. Then train:
python scripts/train_rlan.py --config configs/rlan_base.yaml
```

### 3.5 Training on CPU (debugging only)
```powershell
python scripts/train_rlan.py --config configs/rlan_small.yaml --device cpu
```

---

## 4. EVALUATION COMMANDS

### 4.1 Standard Evaluation
```powershell
# Evaluate best checkpoint
python scripts/evaluate_rlan.py --checkpoint checkpoints/rlan_base/best.pt
```

### 4.2 Evaluation with Test-Time Augmentation (TTA)
```powershell
python scripts/evaluate_rlan.py --checkpoint checkpoints/rlan_base/best.pt --use-tta
```

### 4.3 Detailed Output with Visualizations
```powershell
python scripts/evaluate_rlan.py --checkpoint checkpoints/rlan_base/best.pt --detailed-output --visualize
```

### 4.4 Evaluate on Training Set (for debugging)
```powershell
python scripts/evaluate_rlan.py --checkpoint checkpoints/rlan_base/best.pt --data-path ./data/arc-agi/data/training
```

---

## 5. ANALYSIS & REPORTING

### 5.1 Generate HTML Analysis Report
```powershell
python scripts/analyze_rlan_evaluation.py --results evaluation_results/evaluation_summary.json --output evaluation_results/analysis_report.html
```

### 5.2 Verify End-to-End Flow
```powershell
python scripts/verify_rlan_flow.py
```

---

## 6. CONFIGURATION REFERENCE

### 6.1 Key Config Parameters (`configs/rlan_base.yaml`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model.hidden_dim` | 256 | Feature dimension (affects memory) |
| `training.batch_size` | 64 | Batch size (max ~128 on RTX 3090) |
| `training.max_epochs` | 250 | Total training epochs |
| `training.learning_rate` | 1e-4 | Initial learning rate |
| `training.use_ema` | true | Use EMA for stable evaluation |
| `training.ema_decay` | 0.999 | EMA decay rate |
| `device.mixed_precision` | true | Enable AMP (required for efficiency) |
| `data.cache_samples` | false | Set true for testing, false for competitive |
| `logging.save_every` | 10 | Checkpoint interval (epochs) |
| `logging.use_wandb` | false | WandB logging (disabled by default) |

### 6.2 Memory Optimization
For VRAM issues, reduce batch size:
```yaml
training:
  batch_size: 32  # Reduced from 64
  grad_accumulation_steps: 2  # Maintains effective batch size
```

---

## 7. MONITORING & DEBUGGING

### 7.1 Check GPU Memory
```powershell
nvidia-smi
```

### 7.2 View Training Logs
```powershell
Get-Content -Tail 50 checkpoints/rlan_base/training_log_*.txt
```

### 7.3 Check Checkpoint Contents
```powershell
python -c "import torch; cp = torch.load('checkpoints/rlan_base/latest.pt', map_location='cpu'); print(f'Epoch: {cp[\"epoch\"]}, Best Acc: {cp[\"best_accuracy\"]:.4f}')"
```

---

## 8. EXPECTED RESULTS

### 8.1 Training Metrics (per epoch)
- **Focal Loss**: Should decrease from ~2.0 to ~0.5
- **Total Loss**: Should decrease from ~3.0 to ~1.0
- **Temperature**: Anneals from 5.0 → 0.1

### 8.2 Evaluation Metrics (target goals)
| Metric | Random Baseline | Target | SOTA |
|--------|-----------------|--------|------|
| Task Accuracy | 0% | 15-25% | ~30% |
| Pixel Accuracy | ~10% | 80%+ | 90%+ |

### 8.3 Training Time Estimates
| Config | GPU | Time per Epoch | Total (250 epochs) |
|--------|-----|----------------|-------------------|
| rlan_small | RTX 3090 | ~2-3 min | ~8-12 hours |
| rlan_base | RTX 3090 | ~4-5 min | ~17-21 hours |

---

## 9. TROUBLESHOOTING

### 9.1 CUDA Out of Memory
```powershell
# Reduce batch size
# Edit configs/rlan_base.yaml:
# training.batch_size: 32

# Or use smaller model
python scripts/train_rlan.py --config configs/rlan_small.yaml
```

### 9.2 Training Not Improving
1. Check learning rate (try 3e-4 or 5e-5)
2. Verify data augmentation is enabled
3. Check focal_gamma (2.0 recommended)
4. Ensure temperature is annealing properly

### 9.3 Import Errors
```powershell
# Ensure you're in the correct directory
cd C:\Users\perahmat\Downloads\SCI-ARC

# Reinstall dependencies
pip install -r requirements.txt
```

### 9.4 Multiprocessing Errors (Windows)
Set `num_workers: 0` in config (already default for Windows)

---

## 10. QUICK REFERENCE COMMANDS

```powershell
# === TRAINING ===
python scripts/train_rlan.py --config configs/rlan_base.yaml                    # Full training
python scripts/train_rlan.py --config configs/rlan_base.yaml --resume auto      # Resume training
python scripts/train_rlan.py --config configs/rlan_small.yaml                   # Small model

# === EVALUATION ===
python scripts/evaluate_rlan.py --checkpoint checkpoints/rlan_base/best.pt      # Standard eval
python scripts/evaluate_rlan.py --checkpoint checkpoints/rlan_base/best.pt --use-tta --detailed-output  # Full eval

# === ANALYSIS ===
python scripts/analyze_rlan_evaluation.py --results evaluation_results/evaluation_summary.json

# === TESTING ===
python -m pytest tests/ -v                                                       # All tests
python scripts/verify_rlan_flow.py                                               # End-to-end verify
```

---

## 11. FILE LOCATIONS REFERENCE

| Category | Path |
|----------|------|
| Configs | `configs/rlan_base.yaml`, `configs/rlan_small.yaml` |
| Training script | `scripts/train_rlan.py` |
| Evaluation script | `scripts/evaluate_rlan.py` |
| Analysis script | `scripts/analyze_rlan_evaluation.py` |
| RLAN model | `sci_arc/models/rlan.py` |
| RLAN modules | `sci_arc/models/rlan_modules/` |
| Loss function | `sci_arc/training/rlan_loss.py` |
| Evaluation metrics | `sci_arc/evaluation/metrics.py` |
| Training checkpoints | `checkpoints/rlan_base/` |
| Evaluation results | `evaluation_results/` |
| Tests | `tests/` |

---

## 12. PRODUCTION CHECKLIST

Before starting production training:

- [ ] GPU memory verified (`nvidia-smi`)
- [ ] Data paths verified (400 tasks each in training/evaluation)
- [ ] Virtual environment activated
- [ ] Tests passing (`pytest tests/ -v`)
- [ ] Config reviewed (`configs/rlan_base.yaml`)
- [ ] Checkpoint directory writable
- [ ] Log file location verified
- [ ] WandB logged in (if using)

---

**Last Updated**: 2024
**RLAN Version**: Production Ready
**Hardware Verified**: RTX 3090 24GB VRAM
