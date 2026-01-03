Logging to: checkpoints\rlan_stable_merged\training_log_20260103_124454.txt
Timestamp: 2026-01-03T12:44:54.265265
Python: 3.13.5 (tags/v3.13.5:6cb20a2, Jun 11 2025, 16:15:46) [MSC v.1943 64 bit (AMD64)]
PyTorch: 2.9.1+cu126
Starting fresh training

num_classes validation: OK (10 classes for colors 0-9)
[CONFIG] Meta-learning validation: PASSED
[RecursiveSolver] Output head initialized: bg_bias=0.0 (NEUTRAL), fg_bias=0.0
[SolverCrossAttention] HPM context coupling ENABLED (max_tokens=4, gate_init=0.00)
[RecursiveSolver] Phase 2.5: Solver cross-attention ENABLED (4 heads)
[RecursiveSolver] Jan 2026: HPM solver-context coupling ENABLED (max_tokens=4)
[HPM] Initialized with 5 banks:
      Static: ['COMPOSITIONAL', 'PATTERN', 'RELATIONAL']
      Dynamic: ['PROCEDURAL', 'INSTANCE']
      Top-K routing: k=2
      Gated residual: alpha=0 (starts at 0 contribution)
RLAN Module Config: Enabled=[ContextEncoder, DSC, MSRE, SolverContext, HyperLoRA, HPM], Disabled=[LCR, SPH, ACT]

[MEMORY BASELINE] Model on GPU:
  Allocated: 75MB
  Reserved:  80MB
  GPU Total: 24576MB
  Headroom:  24496MB

[MEMORY MANAGER] Initialized:
  GPU Total: 24576MB
  Usable (with safety margin): 22609MB
  Min batch size: 16
  Max batch size: 128

Model parameters:
  encoder: 67,712
  feature_proj: 66,304
  context_encoder: 3,940,224
  context_injector: 789,760
  dsc: 660,865
  msre: 109,152
  lcr: 0
  sph: 0
  solver: 7,449,611
  hyper_lora: 5,000,448
  hpm: 748,817
  total: 18,832,893

============================================================
MODULE ABLATION STATUS
============================================================
  ContextEncoder: ENABLED (task signal from demos)
  DSC:            ENABLED (dynamic spatial clues - CORE)
  MSRE:           ENABLED (multi-scale relative encoding - CORE)
  LCR:            DISABLED (latent counting registers)
  SPH:            DISABLED (symbolic predicate heads)
  ACT:            DISABLED (adaptive computation time)
  Pos Encoding:   SINUSOIDAL

  >>> CORE ABLATION MODE: Testing DSC + MSRE novelty <<<
============================================================

============================================================
RLAN TRAINING REGIME
============================================================
  Batch Size: 50
  Grad Accumulation: 7
  Effective Batch: 350
  Learning Rate: 5.0e-04
  Weight Decay: 0.01
  Optimizer: adamw (beta1=0.9, beta2=0.95)
  Scheduler: none
  Warmup Epochs: 0
  Max Epochs: 200

Loss Configuration:
  Loss Mode: FOCAL_WEIGHTED
    gamma=1.2, alpha=0.75

Auxiliary Loss Weights (only non-zero shown):
  lambda_entropy=0.01 (DSC attention sharpness)
  lambda_sparsity=0.5 (clue efficiency)
    min_clues=1.0 (min clues before penalty)
    min_clue_weight=5.0 (penalty strength)
    ponder_weight=0.02 (cost per clue)
  lambda_predicate=0.01 (predicate diversity)

Temperature Schedule (DSC Attention):
  Start: 1.0, End: 0.5
  Note: Pure softmax (no Gumbel noise) - identical train/eval behavior
============================================================
  Loss Mode: FOCAL_WEIGHTED (bg_cap=1.0, fg_cap=6.0, gamma=1.2) [RECOMMENDED]
  Clue Regularization: min_clues=1.0, min_clue_weight=5.0, ponder_weight=0.02, entropy_weight=0.02
  Clue Variance Regularization: weight=1.0, target_std=0.5
  Stop Saturation Guard: weight=0.5, threshold=5.0
  Centroid Diversity: lambda=0.3 (prevents DSC clue collapse)

Loading data from: ./data/arc-agi/data/training
Cache samples: False

============================================================
AUGMENTATION CONFIGURATION
============================================================
  1. Dihedral (D4 group): ENABLED
     - 8 transforms: identity, rot90, rot180, rot270, flipLR, flipUD, transpose, anti-transpose
  2. Color Permutation:   ENABLED
     - 9! = 362,880 permutations (colors 1-9 shuffled, 0 fixed)
     - Probability: 50% (CRITICAL: 100% breaks color identity learning!)
  3. Translational:       ENABLED
     - Random offset within 30x30 canvas (~100 positions)

  Total Diversity: 8 x 181,440 x ~100 = ~145,152,000 unique per task
  Mode: On-the-fly (EACH sample is NEW random augmentation)
  Advantage: Infinite diversity vs TRM's fixed 1000 samples
============================================================

Curriculum learning DISABLED (using all data from epoch 1, like TRM)
WARNING: use_merged_training=true but manifest not found: data\merged_training\merged_train_manifest.jsonl
         Run: python scripts/build_merged_training_set.py
         Falling back to standard train_path
  [WARNING] use_merged_training=True but manifest not found, falling back to train_path
  [Auto Compute] num_cached_samples = 400 tasks × 50 = 20000
Loaded 400 tasks from ./data/arc-agi/data/training
  Using BUCKETED BATCHING (groups samples by grid size)
    Bucket boundaries: [10, 15, 20, 25] → 5 buckets
[BucketedBatchSampler] Building 5 buckets for 400 samples (metadata-only)...
  [OK] All 400 samples sized via metadata (no __getitem__ calls)
  Bucket 0 (grid <=10): 178 samples
  Bucket 1 (grid <=15): 88 samples
  Bucket 2 (grid <=20): 76 samples
  Bucket 3 (grid <=25): 28 samples
  Bucket 4 (grid >25): 30 samples
Loaded 400 tasks from ./data/arc-agi/data/evaluation
Train samples: 400, batches: 5
Eval samples: 400, batches: 8

[MEMORY CHECK] batch_size=50 is within safe limits
  Optimizer param groups:
    DSC: 19 params @ 1.0x LR (5.00e-04)
    MSRE: 10 params @ 1.0x LR (5.00e-04)
    HyperLoRA: 38 params @ 1.0x LR (5.00e-04) [META-LEARNING]
    Other: 140 params @ 1x LR (5.00e-04)
  [WARNING] bitsandbytes not installed, falling back to standard AdamW
           Install with: pip install bitsandbytes
Using mixed precision training (AMP) with dtype=bfloat16
  → bfloat16 has same exponent range as fp32 - less likely to overflow/underflow

============================================================
DTYPE VALIDATION CHECK
============================================================
  Config dtype: bfloat16
  PyTorch dtype: torch.bfloat16
  GPU bfloat16 support: True
  Autocast test dtype: torch.float32
  bfloat16 range: [-3e+38, 3e+38]
  bfloat16 tiny: 1e-38
  ✓ Dtype validation passed
============================================================
WandB logging disabled (use_wandb=false or wandb not installed)
LOO training configured: weight=0.05, min_pairs=2, max_pairs=4, start_epoch=25
OUTPUT-level equivariance training configured: weight=0.02, num_augs=2, start_epoch=22, type=kl, mask_to_target=True
[META LOSS CAP] Enabled: meta-loss capped at 25% of total loss

============================================================
ATTENTION COLLAPSE BACKOFF ENABLED (Jan 2026)
============================================================
  Trigger: 2+ consecutive epochs with attn_max < 0.02
  On collapse: delta_scale *= 0.5, LR *= 0.5
  Cooldown: 3 epochs before restore begins
  Restore rate: +20% per stable epoch
============================================================


============================================================
HPM SOLVER-CONTEXT COUPLING ENABLED (Jan 2026)
============================================================
  Start epoch: 81
  Gate warmup: 20 epochs
  Gate max: 0.3
  Logit clamp: 5.0
  Auto-disable on instability: True
============================================================


============================================================
LATE-PHASE META-ESCALATION ENABLED (Jan 2026)
============================================================
  Start epoch: 51
  Stricter stability gates:
    Max grad events/epoch: 0
    Max LR backoff events/epoch: 0
    Max NaN streak/epoch: 0
    Max attention collapse events/epoch: 0
  LR Decay:
    Schedule: cosine
    Range: epochs 51 - 200
    Factor: 1.0 → 0.1
============================================================


============================================================
META ESCALATION ENABLED (Late-Phase Strength Increase)
============================================================
  Schedule: linear
  Start epoch: 26
  Ramp epochs: 12
  Targets:
    HyperLoRA delta_scale: 0.05 → 0.2
    Equiv weight: 0.0 → 0.03
    LOO weight: 0.05 → 0.08
    HPM balance weight: 0.01 → 0.01
  Stability gating:
    Mode: GATED (pauses on instability, resumes when stable)
    Max allowed per epoch: nan_streak=3, grad_events=2, lr_events=2
  Recovery: +5% per stable window
============================================================


============================================================
STAGED META-LEARNING ENABLED (Scientifically Ordered)
============================================================
  Phase 2 - Context Path (epoch 12+):
    SolverCrossAttention + CrossAttentionInjector
  Phase 3 - HyperLoRA (epoch 22+):
    With warmup: scale 0.005 → 0.05 over 8 epochs
  Phase 5 - LOO Loss: Epoch 26+ (weight=0.05)

  Safety:
    LR reduction at activation: 0.5x for 2 epochs
    Grad explosion threshold: 10.0x clip → 0.5x LR
============================================================

[HPM Memory] Buffer population ENABLED from epoch 0 (hpm_memory_start_epoch=0)

============================================================
STAGED HPM ENABLED
============================================================
  Phase 1 (epochs 1-20): HPM inactive, memory collection active
    - HPM module exists but use_hpm=False during forward
    - No HPM load balancing loss added
    - TODO 2: Dynamic buffers ARE populated for solved tasks
  Phase 2 (epochs 21+): HPM activated
    - HPM contributes to features (gated residual)
    - Load balancing loss ensures all banks utilized
============================================================

  [HPM] Temporarily disabled until epoch 21
  [HPM] Memory collection ENABLED (buffers grow before activation)

============================================================
STAGGERED MODULE CONTRIBUTIONS (Prevents Memory Spike)
============================================================
  [HyperLoRA] LoRA deltas disabled until epoch 22
    - Module trains via LOO/Equiv losses after activation
  [SolverCrossAttention] Disabled until epoch 12
    - Solver runs without cross-attention initially
  [CrossAttentionInjector] Disabled until epoch 16
    - Using FiLM fallback (pool+scale/shift) for stable early training
============================================================


============================================================
PHASED TRAINING ENABLED (Jan 2026)
============================================================
  Phase A (Base Solver): epochs 1-10
    - Augmentation: rotation=False, flip=False
    - Meta: HyperLoRA=False, LOO=False
  Phase B (Geometric Aug): epochs 11-20
    - Augmentation: rotation=True, flip=True
    - Meta: HyperLoRA=False, LOO=False
  Phase C (Full Meta): epochs 21+
    - Augmentation: rotation=True, color_perm=True
    - Meta: All enabled (respects individual start_epochs)
============================================================

  [Eval Cache] Pre-loaded 100 TRM eval tasks

Starting training from epoch 0 to 200
============================================================
  [ARCDataset] Augmentation config updated:
    dihedral=False, color_perm=False (prob=0.50), translational=False

============================================================
  PHASE A ACTIVE (epoch 1)
============================================================
    HyperLoRA: DISABLED by phase
    LOO Loss: DISABLED by phase
    Equivariance: DISABLED by phase
    HPM: DISABLED by phase
    Augmentation: rot=False, flip=False, trans=False, color=False
============================================================

  [Phase Change] Recreating DataLoader to propagate augmentation config to workers...
WARNING: use_merged_training=true but manifest not found: data\merged_training\merged_train_manifest.jsonl
         Run: python scripts/build_merged_training_set.py
         Falling back to standard train_path
  [WARNING] use_merged_training=True but manifest not found, falling back to train_path
  [Auto Compute] num_cached_samples = 400 tasks × 50 = 20000
Loaded 400 tasks from ./data/arc-agi/data/training
  Using BUCKETED BATCHING (groups samples by grid size)
    Bucket boundaries: [10, 15, 20, 25] → 5 buckets
[BucketedBatchSampler] Building 5 buckets for 400 samples (metadata-only)...
  [OK] All 400 samples sized via metadata (no __getitem__ calls)
  Bucket 0 (grid <=10): 178 samples
  Bucket 1 (grid <=15): 88 samples
  Bucket 2 (grid <=20): 76 samples
  Bucket 3 (grid <=25): 28 samples
  Bucket 4 (grid >25): 30 samples
  [ARCDataset] Augmentation config updated:
    dihedral=False, color_perm=False (prob=0.50), translational=False
  [Phase Change] DataLoader recreated with 5 batches
  [Phase Override] HPM memory collection DISABLED for phase A

Epoch 1/200
----------------------------------------
  [Batch 0] Grid size: 30x30 (batch_max_size=30)

  [MEMORY] Epoch 0 Batch 0:
    Baseline: alloc=75MB, reserved=80MB
    01_batch_on_gpu: alloc=84MB (+9MB), reserved=102MB
    02_after_forward: alloc=13237MB (+13152MB), reserved=14012MB
    03_before_backward: alloc=13284MB (+47MB), reserved=14050MB
    04_after_backward: alloc=219MB (-13065MB), reserved=4938MB
    PEAK: alloc=13779MB, reserved=14050MB / 24576MB (57.2%)

    [BREAKDOWN] Largest increase at '02_after_forward' (+13152MB):
      Model params: 73.4MB
      Model grads:  0.0MB
      Optimizer:    146.7MB (estimate (Adam-like: AdamW))
      Batch data:   4.1MB
      Activations:  68.3MB
      Output tensors (top 5):
        features: 43.9MB
        support_features: 15.6MB
        all_logits: 6.0MB
        attention_maps: 1.2MB
        logits: 0.9MB
      Module params (top 5):
        solver: 28.4MB
        hyper_lora: 19.1MB
        context_encoder: 15.0MB
        context_injector: 3.0MB
        hpm: 2.9MB
      Inactive modules: hyperlora, solver_context, cross_attention, hpm, loo, equivariance
  Batch 0/5: loss=0.4116, focal_weighted=0.2658, batch_acc=21.8%, exact=0/50 (0.0%), running_acc=21.8%, lr=5.00e-04
    [TaskTrack] Global_Solved: 0/50, Epoch_Solved: 0, New_Puzzles: 0
    FG: batch=6.2% run50=6.2% | BG: batch=34.0% run50=34.0%
    Per-Color: [0:39% 1:2% 2:3% 3:17% 4:6% 5:10% 6:4% 7:7% 8:6% 9:6%]
    Running50: [0:39% 1:2% 2:3% 3:17% 4:6% 5:10% 6:4% 7:7% 8:6% 9:6%]
    Solver: [2.172, 2.172, 2.172, 2.172, 2.172, 2.172, 2.172] ⚠ best=0
  [Batch 1] Grid size: 30x30 (batch_max_size=30)

  [MEMORY] Epoch 0 Batch 1:
    Baseline: alloc=75MB, reserved=80MB
    01_batch_on_gpu: alloc=211MB (+136MB), reserved=4940MB
    02_after_forward: alloc=14240MB (+14029MB), reserved=15026MB
    03_before_backward: alloc=14269MB (+29MB), reserved=15040MB
    04_after_backward: alloc=280MB (-13988MB), reserved=6440MB
    PEAK: alloc=14782MB, reserved=15348MB / 24576MB (62.5%)

    [BREAKDOWN] Largest increase at '02_after_forward' (+14029MB):
      Model params: 73.4MB
      Model grads:  45.9MB
      Optimizer:    146.7MB (estimate (Adam-like: AdamW))
      Batch data:   4.8MB
      Activations:  71.5MB
      Output tensors (top 5):
        features: 43.9MB
        support_features: 18.8MB
        all_logits: 6.0MB
        attention_maps: 1.2MB
        logits: 0.9MB
      Module params (top 5):
        solver: 28.4MB
        hyper_lora: 19.1MB
        context_encoder: 15.0MB
        context_injector: 3.0MB
        hpm: 2.9MB
      Inactive modules: hyperlora, solver_context, cross_attention, hpm, loo, equivariance
  Batch 1/5: loss=0.2831, focal_weighted=0.1495, batch_acc=18.0%, exact=0/100 (0.0%), running_acc=19.9%, lr=5.00e-04
    [TaskTrack] Global_Solved: 0/100, Epoch_Solved: 0, New_Puzzles: 0
    FG: batch=7.7% run50=7.0% | BG: batch=26.9% run50=30.5%
    Per-Color: [0:30% 1:1% 2:3% 3:19% 4:11% 5:6% 6:6% 7:25% 8:1% 9:0%]
    Running50: [0:35% 1:1% 2:3% 3:18% 4:8% 5:8% 6:5% 7:16% 8:4% 9:3%]
  [Batch 2] Grid size: 30x30 (batch_max_size=30)

  [MEMORY] Epoch 0 Batch 2:
    Baseline: alloc=75MB, reserved=80MB
    01_batch_on_gpu: alloc=214MB (+139MB), reserved=6440MB
    02_after_forward: alloc=16028MB (+15814MB), reserved=16866MB
    03_before_backward: alloc=16055MB (+28MB), reserved=16878MB
    04_after_backward: alloc=289MB (-15766MB), reserved=8328MB
    PEAK: alloc=16570MB, reserved=17186MB / 24576MB (69.9%)

    [BREAKDOWN] Largest increase at '02_after_forward' (+15814MB):
      Model params: 73.4MB
      Model grads:  45.9MB
      Optimizer:    146.7MB (estimate (Adam-like: AdamW))
      Batch data:   6.2MB
      Activations:  77.8MB
      Output tensors (top 5):
        features: 43.9MB
        support_features: 25.0MB
        all_logits: 6.0MB
        attention_maps: 1.2MB
        logits: 0.9MB
      Module params (top 5):
        solver: 28.4MB
        hyper_lora: 19.1MB
        context_encoder: 15.0MB
        context_injector: 3.0MB
        hpm: 2.9MB
      Inactive modules: hyperlora, solver_context, cross_attention, hpm, loo, equivariance
  Batch 2/5: loss=0.2615, focal_weighted=0.1286, batch_acc=18.5%, exact=0/150 (0.0%), running_acc=19.4%, lr=5.00e-04
    [TaskTrack] Global_Solved: 0/150, Epoch_Solved: 0, New_Puzzles: 0
    FG: batch=8.2% run50=7.4% | BG: batch=26.4% run50=29.1%
    Per-Color: [0:30% 1:0% 2:4% 3:28% 4:14% 5:7% 6:3% 7:12% 8:3% 9:0%]
    Running50: [0:33% 1:1% 2:3% 3:21% 4:10% 5:8% 6:4% 7:15% 8:3% 9:2%]
  [Batch 3] Grid size: 30x30 (batch_max_size=30)

  [MEMORY] Epoch 0 Batch 3:
    Baseline: alloc=75MB, reserved=80MB
    01_batch_on_gpu: alloc=217MB (+142MB), reserved=8328MB
    02_after_forward: alloc=13350MB (+13132MB), reserved=14078MB
    03_before_backward: alloc=13436MB (+86MB), reserved=14156MB
    04_after_backward: alloc=281MB (-13155MB), reserved=14464MB
    PEAK: alloc=13892MB, reserved=14464MB / 24576MB (58.9%)

    [BREAKDOWN] Largest increase at '02_after_forward' (+13132MB):
      Model params: 73.4MB
      Model grads:  45.9MB
      Optimizer:    146.7MB (estimate (Adam-like: AdamW))
      Batch data:   4.1MB
      Activations:  68.3MB
      Output tensors (top 5):
        features: 43.9MB
        support_features: 15.6MB
        all_logits: 6.0MB
        attention_maps: 1.2MB
        logits: 0.9MB
      Module params (top 5):
        solver: 28.4MB
        hyper_lora: 19.1MB
        context_encoder: 15.0MB
        context_injector: 3.0MB
        hpm: 2.9MB
      Inactive modules: hyperlora, solver_context, cross_attention, hpm, loo, equivariance
  Batch 3/5: loss=1.0455, focal_weighted=0.8819, batch_acc=22.1%, exact=0/200 (0.0%), running_acc=20.1%, lr=5.00e-04
    [TaskTrack] Global_Solved: 0/200, Epoch_Solved: 0, New_Puzzles: 0
    FG: batch=5.4% run50=6.9% | BG: batch=39.6% run50=31.7%
    Per-Color: [0:43% 1:1% 2:3% 3:8% 4:5% 5:14% 6:6% 7:9% 8:6% 9:7%]
    Running50: [0:36% 1:1% 2:3% 3:18% 4:9% 5:9% 6:5% 7:13% 8:4% 9:3%]
  [Batch 4] Grid size: 30x30 (batch_max_size=30)

  [MEMORY] Epoch 0 Batch 4:
    Baseline: alloc=75MB, reserved=80MB
    01_batch_on_gpu: alloc=210MB (+135MB), reserved=14464MB
    02_after_forward: alloc=17809MB (+17599MB), reserved=18682MB
    03_before_backward: alloc=17834MB (+25MB), reserved=18694MB
    04_after_backward: alloc=293MB (-17541MB), reserved=10206MB
    PEAK: alloc=18351MB, reserved=19002MB / 24576MB (77.3%)

    [BREAKDOWN] Largest increase at '02_after_forward' (+17599MB):
      Model params: 73.4MB
      Model grads:  45.9MB
      Optimizer:    146.7MB (estimate (Adam-like: AdamW))
      Batch data:   7.6MB
      Activations:  84.1MB
      Output tensors (top 5):
        features: 43.9MB
        support_features: 31.2MB
        all_logits: 6.0MB
        attention_maps: 1.2MB
        logits: 0.9MB
      Module params (top 5):
        solver: 28.4MB
        hyper_lora: 19.1MB
        context_encoder: 15.0MB
        context_injector: 3.0MB
        hpm: 2.9MB
      Inactive modules: hyperlora, solver_context, cross_attention, hpm, loo, equivariance
  Batch 4/5: loss=0.2642, focal_weighted=0.1293, batch_acc=19.1%, exact=1/250 (0.4%), running_acc=19.9%, lr=5.00e-04
    [TaskTrack] Global_Solved: 1/250, Epoch_Solved: 1, New_Puzzles: 1
    FG: batch=11.4% run50=7.8% | BG: batch=27.4% run50=30.9%
    Per-Color: [0:31% 1:2% 2:3% 3:16% 4:6% 5:27% 6:0% 7:13% 8:1% 9:22%]
    Running50: [0:35% 1:1% 2:3% 3:18% 4:8% 5:13% 6:4% 7:13% 8:3% 9:7%]

  [EPOCH MEMORY] Peak: 322MB alloc, 10206MB reserved

Epoch 1 Summary:
  Total Loss: 0.4532
  Task Loss (focal): 0.3110
  [Global Task Progress] Unique Solved: 1/250 (0.4%)
    NEW puzzles solved this epoch: 1
  Entropy Loss: 4.3049 (weight=0.01)
  Sparsity Loss: 0.1709 (weight=0.5)
  Predicate Loss: 0.0000 (weight=0.01)
  Time: 67.9s, LR: 5.00e-04
    Per-module LRs: DSC:5.00e-04, MSRE:5.00e-04, Other:5.00e-04
  Temperature: 1.0000 (lower=sharper attention)
  Samples Processed: 250 (5 batches)
  Dihedral Distribution: [100.0%, 0.0%, 0.0%, 0.0%, 0.0%, 0.0%, 0.0%, 0.0%]
    [!] Non-uniform dihedral distribution (max dev: 87.5%)
  Color Permutation: 0.0% (0/250)
  Translational Aug: 0.0% (0/250), unique offsets: 0
  Aug Quality: OK
  --- Training Diagnostics ---
  Solver Steps: 7 (deep supervision active)
  Per-Step Loss (epoch avg, 1 batches): [2.1719, 2.1719, 2.1719, 2.1719, 2.1719, 2.1719, 2.1719]
    [!] SOLVER DEGRADATION: Step 0 is best! Later steps 0.0% worse!
    [!] Best: step 0 (2.1719), Worst: step 0 (2.1719)
  Best-Step Histogram: [s0:1]
  Solver Health: Last step best: 0.0%, Earlier step best: 100.0%
  Avg Step Improvement: 0.0% (step0→stepN)
    ⚠️ SOLVER OVER-ITERATION WARNING: 100% of batches had earlier step as best!
    ⚠️ Consider: (1) enabling best-step selection, (2) reducing num_solver_steps, (3) enabling ACT
  Grad Norms: DSC=0.0000, StopPred=0.0000, Encoder=0.0000, Solver=0.0000
  StopPred Weight Var: 2.80e-02
  Attention: max=0.0173, min=0.000000
  Stop Prob: 0.283 (approx 5.0 clues active)
  Stop Probs Std: 0.093 (global std across batch×clues)
  Clues Used: mean=5.02, std=0.32, range=[4.4, 5.6]
  Clue-Loss Correlation: -0.142 (unexpected negative - check gradient flow)
  Stop Logits: mean=-0.98, std=0.47, range=[-2.2, 0.4]
    Clue count varies by task (per-sample coupling active)
  Per-Clue Entropy: [4.71, 4.71, 4.71, 4.70, 4.70, 4.71, 4.71] (mean=4.71, max=6.80)
    [!] Clues have uniform entropy (std=0.003) - not differentiating!
  Centroid Spread: 0.11 (higher=more diverse)
    🚨 CRITICAL COLLAPSE: Spread=0.11 < 0.5 - all clues at same location!
    [!] Stop predictor cannot differentiate - needs diversity regularizer
  --- Stop Predictor Coupling ---
  Entropy Input to Stop: 0.6920 (normalized, lower=sharper)
  Per-Clue Entropy Input: [0.693, 0.692, 0.692, 0.692, 0.691, 0.692, 0.692]
  Per-Clue Stop Prob: [0.289, 0.281, 0.277, 0.275, 0.270, 0.311, 0.281]
  --- Sparsity Loss Breakdown (Per-Sample Coupled) ---
  Min Clue Penalty: 0.0000 (per-sample avg)
  Per-Sample Clue Penalty (scaled): 0.0000
  Base Pondering: 0.1003 (clues=5.02)
  Entropy Pondering: 0.0696
  --- Per-Class Distribution (Valid Pixels Only) ---
  Padding: 87.7% of grid (ignored in loss)
  Pred %: [33.8, 0.7, 2.4, 12.9, 5.6, 20.3, 1.6, 16.5, 1.7, 4.5]
  Target %: [58.9, 4.7, 4.2, 4.8, 4.7, 4.9, 7.1, 1.3, 8.9, 0.6]
  Per-Class Acc %: [38, 1, 3, 12, 6, 11, 5, 10, 5, 8]
  ==================================================
  PER-SAMPLE TRAINING ACCURACY (Epoch 1)
  ==================================================
  ★ Mean Accuracy: 19.9%
  ★ Exact Match: 1/250 (0.4%)
  ★ High Acc (≥90%): 1/250 (0.4%)
  FG Accuracy: 7.8%
  BG Accuracy: 30.9%
  Batch Trend: 21.8% → 20.6% (↓ 1.1pp)
  Accuracy Distribution: 0-25%:50%, 25-50%:50%, 50-75%:0%, 75-90%:0%, 90-100%:0%
  Running Window (last 5 batches): 19.9% ± 1.7%

  --- PER-COLOR ACCURACY (10 classes) ---
  Color:       0     1     2     3     4     5     6     7     8     9
  Acc%:      38%    1%    3%   12%    6%   11%    5%   10%    5%    8%
  Target:  56.1%  6.9%  5.1%  7.4%  4.7%  5.2%  4.0%  2.1%  6.8%  1.6%
  Pred:    34.0%  1.0%  2.4% 12.9%  5.2% 21.6%  1.9% 14.6%  1.8%  4.8%
  [!] Weak colors (<50% acc): 0(Black), 1(Blue), 2(Red), 3(Green), 4(Yellow), 5(Gray), 6(Pink), 7(Orange), 8(Cyan), 9(Brown)
  [!] Under-predicting color 1 (Blue): 1.0% vs target 6.9%
  [!] Under-predicting color 2 (Red): 2.4% vs target 5.1%
  [!] Under-predicting color 6 (Pink): 1.9% vs target 4.0%
  [!] Under-predicting color 8 (Cyan): 1.8% vs target 6.8%
  ==================================================

  ==================================================
  LEARNING TRAJECTORY (Epoch 1)
  ==================================================
  ==================================================

  ############################################################
  🚦 TRAINING HEALTH CHECK - Epoch 1
  ############################################################
  ✓ Attention sharpening (0.69 < 0.7)
  ✓ No NaN/Inf issues
  ✓ No color mode collapse
  ⚠ Stop probs nearly uniform (global=0.093, per_clue=0.012)
  ⚠ Negative coupling (r=-0.19) - early epoch OK
  ✗ Centroid COLLAPSE (0.1 < 0.5)
  --------------------------------------------------------
  RESULT: 3/6 checks passed
  STATUS: 🟠 WARNING
  → Multiple issues detected. Consider intervention soon.

  RECOMMENDED ACTIONS:
  ############################################################
  HPM Instance Buffer: 0 entries (empty, not saved)
  HPM Procedural Buffer: 0 entries (empty, not saved)
  Saved checkpoint to checkpoints\rlan_stable_merged\latest.pt
  [ARCDataset] Augmentation config updated:
    dihedral=False, color_perm=False (prob=0.50), translational=False

Epoch 2/200
----------------------------------------
  [Batch 0] Grid size: 30x30 (batch_max_size=30)

  [MEMORY] Epoch 1 Batch 0:
    Baseline: alloc=92MB, reserved=10206MB
    01_batch_on_gpu: alloc=104MB (+12MB), reserved=10228MB
    02_after_forward: alloc=17707MB (+17603MB), reserved=18602MB
    03_before_backward: alloc=17735MB (+28MB), reserved=18618MB
    04_after_backward: alloc=237MB (-17497MB), reserved=19366MB
    PEAK: alloc=18249MB, reserved=19366MB / 24576MB (78.8%)

    [BREAKDOWN] Largest increase at '02_after_forward' (+17603MB):
      Model params: 73.4MB
      Model grads:  0.0MB
      Optimizer:    146.7MB (estimate (Adam-like: AdamW))
      Batch data:   7.6MB
      Activations:  84.1MB
      Output tensors (top 5):
        features: 43.9MB
        support_features: 31.2MB
        all_logits: 6.0MB
        attention_maps: 1.2MB
        logits: 0.9MB
      Module params (top 5):
        solver: 28.4MB
        hyper_lora: 19.1MB
        context_encoder: 15.0MB
        context_injector: 3.0MB
        hpm: 2.9MB
      Inactive modules: hyperlora, solver_context, cross_attention, hpm, loo, equivariance
  Batch 0/5: loss=0.2683, focal_weighted=0.1345, batch_acc=19.2%, exact=1/50 (2.0%), running_acc=19.2%, lr=5.00e-04
    [TaskTrack] Global_Solved: 1/256, Epoch_Solved: 1, New_Puzzles: 0
    FG: batch=7.3% run50=7.3% | BG: batch=23.0% run50=23.0%
    Per-Color: [0:32% 1:4% 2:3% 3:16% 4:3% 5:8% 6:1% 7:13% 8:1% 9:33%]
    Running50: [0:32% 1:4% 2:3% 3:16% 4:3% 5:8% 6:1% 7:13% 8:1% 9:33%]
    Solver: [2.125, 2.141, 2.141, 2.141, 2.141, 2.141, 2.141] ⚠ best=0
  [Batch 1] Grid size: 30x30 (batch_max_size=30)

  [MEMORY] Epoch 1 Batch 1:
    Baseline: alloc=92MB, reserved=10206MB
    01_batch_on_gpu: alloc=212MB (+119MB), reserved=19368MB
    02_after_forward: alloc=13341MB (+13129MB), reserved=19368MB
    03_before_backward: alloc=13427MB (+86MB), reserved=19438MB
    04_after_backward: alloc=274MB (-13153MB), reserved=19438MB
    PEAK: alloc=13883MB, reserved=19438MB / 24576MB (79.1%)

    [BREAKDOWN] Largest increase at '02_after_forward' (+13129MB):
      Model params: 73.4MB
      Model grads:  45.9MB
      Optimizer:    146.7MB (estimate (Adam-like: AdamW))
      Batch data:   4.1MB
      Activations:  68.3MB
      Output tensors (top 5):
        features: 43.9MB
        support_features: 15.6MB
        all_logits: 6.0MB
        attention_maps: 1.2MB
        logits: 0.9MB
      Module params (top 5):
        solver: 28.4MB
        hyper_lora: 19.1MB
        context_encoder: 15.0MB
        context_injector: 3.0MB
        hpm: 2.9MB
      Inactive modules: hyperlora, solver_context, cross_attention, hpm, loo, equivariance
  Batch 1/5: loss=0.8636, focal_weighted=0.7017, batch_acc=21.8%, exact=1/100 (1.0%), running_acc=20.5%, lr=5.00e-04
    [TaskTrack] Global_Solved: 1/274, Epoch_Solved: 1, New_Puzzles: 0
    FG: batch=6.1% run50=6.7% | BG: batch=38.1% run50=30.6%
    Per-Color: [0:40% 1:1% 2:3% 3:9% 4:8% 5:12% 6:5% 7:13% 8:5% 9:9%]
    Running50: [0:36% 1:2% 2:3% 3:12% 4:5% 5:10% 6:3% 7:13% 8:3% 9:21%]
  [Batch 2] Grid size: 30x30 (batch_max_size=30)

  [MEMORY] Epoch 1 Batch 2:
    Baseline: alloc=92MB, reserved=10206MB
    01_batch_on_gpu: alloc=213MB (+120MB), reserved=19438MB
    02_after_forward: alloc=15130MB (+14917MB), reserved=15896MB
    03_before_backward: alloc=15174MB (+44MB), reserved=15936MB
    04_after_backward: alloc=285MB (-14890MB), reserved=7396MB
    PEAK: alloc=15672MB, reserved=19438MB / 24576MB (79.1%)

    [BREAKDOWN] Largest increase at '02_after_forward' (+14917MB):
      Model params: 73.4MB
      Model grads:  45.9MB
      Optimizer:    146.7MB (estimate (Adam-like: AdamW))
      Batch data:   5.5MB
      Activations:  74.6MB
      Output tensors (top 5):
        features: 43.9MB
        support_features: 21.9MB
        all_logits: 6.0MB
        attention_maps: 1.2MB
        logits: 0.9MB
      Module params (top 5):
        solver: 28.4MB
        hyper_lora: 19.1MB
        context_encoder: 15.0MB
        context_injector: 3.0MB
        hpm: 2.9MB
      Inactive modules: hyperlora, solver_context, cross_attention, hpm, loo, equivariance
  Batch 2/5: loss=0.4439, focal_weighted=0.2970, batch_acc=22.6%, exact=1/150 (0.7%), running_acc=21.2%, lr=5.00e-04
    [TaskTrack] Global_Solved: 1/301, Epoch_Solved: 1, New_Puzzles: 0
    FG: batch=4.6% run50=6.0% | BG: batch=34.5% run50=31.9%
    Per-Color: [0:41% 1:1% 2:2% 3:12% 4:4% 5:5% 6:4% 7:23% 8:7% 9:7%]
    Running50: [0:37% 1:2% 2:3% 3:12% 4:5% 5:8% 6:4% 7:16% 8:4% 9:16%]
  [Batch 3] Grid size: 30x30 (batch_max_size=30)

  [MEMORY] Epoch 1 Batch 3:
    Baseline: alloc=92MB, reserved=10206MB
    01_batch_on_gpu: alloc=214MB (+121MB), reserved=7396MB
    02_after_forward: alloc=16023MB (+15809MB), reserved=16848MB
    03_before_backward: alloc=16047MB (+24MB), reserved=16860MB
    04_after_backward: alloc=289MB (-15759MB), reserved=17520MB
    PEAK: alloc=16565MB, reserved=17520MB / 24576MB (71.3%)

    [BREAKDOWN] Largest increase at '02_after_forward' (+15809MB):
      Model params: 73.4MB
      Model grads:  45.9MB
      Optimizer:    146.7MB (estimate (Adam-like: AdamW))
      Batch data:   6.2MB
      Activations:  77.8MB
      Output tensors (top 5):
        features: 43.9MB
        support_features: 25.0MB
        all_logits: 6.0MB
        attention_maps: 1.2MB
        logits: 0.9MB
      Module params (top 5):
        solver: 28.4MB
        hyper_lora: 19.1MB
        context_encoder: 15.0MB
        context_injector: 3.0MB
        hpm: 2.9MB
      Inactive modules: hyperlora, solver_context, cross_attention, hpm, loo, equivariance
  Batch 3/5: loss=0.2860, focal_weighted=0.1551, batch_acc=15.6%, exact=1/200 (0.5%), running_acc=19.8%, lr=5.00e-04
    [TaskTrack] Global_Solved: 1/311, Epoch_Solved: 1, New_Puzzles: 0
    FG: batch=6.4% run50=6.1% | BG: batch=23.6% run50=29.8%
    Per-Color: [0:29% 1:3% 2:3% 3:19% 4:10% 5:2% 6:0% 7:8% 8:4% 9:12%]
    Running50: [0:35% 1:2% 2:3% 3:14% 4:6% 5:6% 6:3% 7:14% 8:4% 9:15%]
  [Batch 4] Grid size: 30x30 (batch_max_size=30)

  [MEMORY] Epoch 1 Batch 4:
    Baseline: alloc=92MB, reserved=10206MB
    01_batch_on_gpu: alloc=209MB (+117MB), reserved=17520MB
    02_after_forward: alloc=15123MB (+14914MB), reserved=17520MB
    03_before_backward: alloc=15149MB (+26MB), reserved=17520MB
    04_after_backward: alloc=279MB (-14870MB), reserved=17520MB
    PEAK: alloc=15665MB, reserved=17520MB / 24576MB (71.3%)

    [BREAKDOWN] Largest increase at '02_after_forward' (+14914MB):
      Model params: 73.4MB
      Model grads:  45.9MB
      Optimizer:    146.7MB (estimate (Adam-like: AdamW))
      Batch data:   5.5MB
      Activations:  74.6MB
      Output tensors (top 5):
        features: 43.9MB
        support_features: 21.9MB
        all_logits: 6.0MB
        attention_maps: 1.2MB
        logits: 0.9MB
      Module params (top 5):
        solver: 28.4MB
        hyper_lora: 19.1MB
        context_encoder: 15.0MB
        context_injector: 3.0MB
        hpm: 2.9MB
      Inactive modules: hyperlora, solver_context, cross_attention, hpm, loo, equivariance
  Batch 4/5: loss=0.2589, focal_weighted=0.1240, batch_acc=16.0%, exact=1/250 (0.4%), running_acc=19.0%, lr=5.00e-04
    [TaskTrack] Global_Solved: 1/323, Epoch_Solved: 1, New_Puzzles: 0
    FG: batch=6.5% run50=6.2% | BG: batch=29.5% run50=29.7%
    Per-Color: [0:31% 1:2% 2:5% 3:12% 4:3% 5:13% 6:3% 7:11% 8:2% 9:0%]
    Running50: [0:34% 1:2% 2:3% 3:14% 4:5% 5:8% 6:3% 7:14% 8:4% 9:12%]

  [EPOCH MEMORY] Peak: 298MB alloc, 17520MB reserved (Δ-24MB from prev epoch)

Epoch 2 Summary:
  Total Loss: 0.4241
  Task Loss (focal): 0.2825
  [Global Task Progress] Unique Solved: 1/323 (0.3%)
    NEW puzzles solved this epoch: 0
  Entropy Loss: 4.2459 (weight=0.01)
  Sparsity Loss: 0.1710 (weight=0.5)
  Predicate Loss: 0.0000 (weight=0.01)
  Time: 7.0s, LR: 5.00e-04
    Per-module LRs: DSC:5.00e-04, MSRE:5.00e-04, Other:5.00e-04
  Temperature: 0.9965 (lower=sharper attention)
  Samples Processed: 250 (5 batches)
  Dihedral Distribution: [100.0%, 0.0%, 0.0%, 0.0%, 0.0%, 0.0%, 0.0%, 0.0%]
    [!] Non-uniform dihedral distribution (max dev: 87.5%)
  Color Permutation: 0.0% (0/250)
  Translational Aug: 0.0% (0/250), unique offsets: 0
  Aug Quality: OK
  --- Training Diagnostics ---
  Solver Steps: 7 (deep supervision active)
  Per-Step Loss (epoch avg, 1 batches): [2.1250, 2.1406, 2.1406, 2.1406, 2.1406, 2.1406, 2.1406]
    [!] SOLVER DEGRADATION: Step 0 is best! Later steps 0.7% worse!
    [!] Best: step 0 (2.1250), Worst: step 1 (2.1406)
  Best-Step Histogram: [s0:1]
  Solver Health: Last step best: 0.0%, Earlier step best: 100.0%
  Avg Step Improvement: -0.7% (step0→stepN)
    ⚠️ SOLVER OVER-ITERATION WARNING: 100% of batches had earlier step as best!
    ⚠️ Consider: (1) enabling best-step selection, (2) reducing num_solver_steps, (3) enabling ACT
    ⚠️ SOLVER DEGRADATION: Avg improvement is negative! More steps = worse predictions!
  Grad Norms: DSC=0.0000, StopPred=0.0000, Encoder=0.0000, Solver=0.0000
  StopPred Weight Var: 2.80e-02
  Attention: max=0.0490, min=0.000000
  Stop Prob: 0.229 (approx 5.4 clues active)
  Stop Probs Std: 0.080 (global std across batch×clues)
  Clues Used: mean=5.40, std=0.26, range=[4.7, 5.9]
  Clue-Loss Correlation: +0.168 (learning - per-sample coupling active)
  Stop Logits: mean=-1.28, std=0.48, range=[-2.9, 0.1]
  Per-Clue Entropy: [3.68, 3.67, 3.67, 3.67, 3.67, 3.68, 3.68] (mean=3.67, max=6.80)
    [!] Clues have uniform entropy (std=0.002) - not differentiating!
  Centroid Spread: 0.06 (higher=more diverse)
    🚨 CRITICAL COLLAPSE: Spread=0.06 < 0.5 - all clues at same location!
    [!] Stop predictor cannot differentiate - needs diversity regularizer
  --- Stop Predictor Coupling ---
  Entropy Input to Stop: 0.5402 (normalized, lower=sharper)
  Per-Clue Entropy Input: [0.541, 0.540, 0.540, 0.540, 0.540, 0.540, 0.541]
  Per-Clue Stop Prob: [0.241, 0.238, 0.226, 0.223, 0.224, 0.212, 0.235]
  --- Sparsity Loss Breakdown (Per-Sample Coupled) ---
  Min Clue Penalty: 0.0000 (per-sample avg)
  Per-Sample Clue Penalty (scaled): 0.0000
  Base Pondering: 0.1080 (clues=5.40)
  Entropy Pondering: 0.0583
  --- Per-Class Distribution (Valid Pixels Only) ---
  Padding: 93.7% of grid (ignored in loss)
  Pred %: [30.1, 1.4, 2.1, 15.1, 5.8, 21.0, 1.5, 16.5, 2.1, 4.5]
  Target %: [65.0, 6.4, 7.0, 2.4, 4.4, 3.3, 2.4, 2.7, 6.0, 0.5]
  Per-Class Acc %: [36, 2, 3, 10, 6, 9, 4, 14, 4, 11]
  ==================================================
  PER-SAMPLE TRAINING ACCURACY (Epoch 2)
  ==================================================
  ★ Mean Accuracy: 19.0%
  ★ Exact Match: 1/250 (0.4%)
  ★ High Acc (≥90%): 1/250 (0.4%)
  FG Accuracy: 6.2%
  BG Accuracy: 29.7%
  Batch Trend: 19.2% → 15.8% (↓ 3.4pp)
  Accuracy Distribution: 0-25%:75%, 25-50%:25%, 50-75%:0%, 75-90%:0%, 90-100%:0%
    [!] Many samples stuck at low accuracy - check data/model!
  Running Window (last 5 batches): 19.0% ± 2.9%

  --- PER-COLOR ACCURACY (10 classes) ---
  Color:       0     1     2     3     4     5     6     7     8     9
  Acc%:      36%    2%    3%   10%    6%    9%    4%   14%    4%   11%
  Target:  59.5%  7.3%  5.7%  7.8%  4.4%  4.6%  2.3%  1.8%  5.4%  1.1%
  Pred:    34.2%  1.2%  2.1% 13.1%  5.8% 21.0%  1.8% 14.7%  1.6%  4.4%
  [!] Weak colors (<50% acc): 0(Black), 1(Blue), 2(Red), 3(Green), 4(Yellow), 5(Gray), 6(Pink), 7(Orange), 8(Cyan), 9(Brown)
  [!] Under-predicting color 1 (Blue): 1.2% vs target 7.3%
  [!] Under-predicting color 2 (Red): 2.1% vs target 5.7%
  [!] Under-predicting color 8 (Cyan): 1.6% vs target 5.4%
  ==================================================

  ==================================================
  LEARNING TRAJECTORY (Epoch 2)
  ==================================================
  Stop Prob:   0.229 ↓ (init=0.27, task-dependent)
  Exp. Clues:  5.40 (latent variable, task-dependent)
  Attn Entropy: 3.67 ↓ (max=6.8, sharper=better)
  Task Loss:   0.2825 ↓
  Train Acc:   19.0% →
  Exact Match: 0.4% →
  Best Step:   0 (later=better refinement)
  FG Coverage: 199.6% of target ↑
  ==================================================

  ############################################################
  🚦 TRAINING HEALTH CHECK - Epoch 2
  ############################################################
  ✓ Attention sharpening (0.54 < 0.7)
  ✓ No NaN/Inf issues
  ✓ No color mode collapse
  ⚠ Stop probs nearly uniform (global=0.080, per_clue=0.010)
  ⚠ Negative coupling (r=-0.33) - early epoch OK
  ✗ Centroid COLLAPSE (0.1 < 0.5)
  --------------------------------------------------------
  RESULT: 3/6 checks passed
  STATUS: 🟠 WARNING
  → Multiple issues detected. Consider intervention soon.

  RECOMMENDED ACTIONS:
  ############################################################
  HPM Instance Buffer: 0 entries (empty, not saved)
  HPM Procedural Buffer: 0 entries (empty, not saved)
  Saved checkpoint to checkpoints\rlan_stable_merged\latest.pt
  [ARCDataset] Augmentation config updated:
    dihedral=False, color_perm=False (prob=0.50), translational=False

Epoch 3/200
----------------------------------------
  [Batch 0] Grid size: 30x30 (batch_max_size=30)

  [MEMORY] Epoch 2 Batch 0:
    Baseline: alloc=92MB, reserved=17520MB
    01_batch_on_gpu: alloc=106MB (+14MB), reserved=17542MB
    02_after_forward: alloc=15917MB (+15811MB), reserved=17542MB
    03_before_backward: alloc=15942MB (+25MB), reserved=17542MB
    04_after_backward: alloc=232MB (-15710MB), reserved=17542MB
    PEAK: alloc=16459MB, reserved=17542MB / 24576MB (71.4%)

    [BREAKDOWN] Largest increase at '02_after_forward' (+15811MB):
      Model params: 73.4MB
      Model grads:  0.0MB
      Optimizer:    146.7MB (estimate (Adam-like: AdamW))
      Batch data:   6.2MB
      Activations:  77.8MB
      Output tensors (top 5):
        features: 43.9MB
        support_features: 25.0MB
        all_logits: 6.0MB
        attention_maps: 1.2MB
        logits: 0.9MB
      Module params (top 5):
        solver: 28.4MB
        hyper_lora: 19.1MB
        context_encoder: 15.0MB
        context_injector: 3.0MB
        hpm: 2.9MB
      Inactive modules: hyperlora, solver_context, cross_attention, hpm, loo, equivariance
  Batch 0/5: loss=0.2567, focal_weighted=0.1224, batch_acc=17.7%, exact=0/50 (0.0%), running_acc=17.7%, lr=5.00e-04
    [TaskTrack] Global_Solved: 1/323, Epoch_Solved: 0, New_Puzzles: 0
    FG: batch=7.9% run50=7.9% | BG: batch=25.7% run50=25.7%
    Per-Color: [0:32% 1:1% 2:2% 3:22% 4:8% 5:14% 6:0% 7:11% 8:1% 9:6%]
    Running50: [0:32% 1:1% 2:2% 3:22% 4:8% 5:14% 6:0% 7:11% 8:1% 9:6%]
    Solver: [2.141, 2.141, 2.141, 2.141, 2.141, 2.141, 2.141] ⚠ best=0
  [Batch 1] Grid size: 30x30 (batch_max_size=30)

  [MEMORY] Epoch 2 Batch 1:
    Baseline: alloc=92MB, reserved=17520MB
    01_batch_on_gpu: alloc=212MB (+120MB), reserved=17544MB
    02_after_forward: alloc=17804MB (+17592MB), reserved=18750MB
    03_before_backward: alloc=17832MB (+28MB), reserved=18756MB
    04_after_backward: alloc=292MB (-17540MB), reserved=19196MB
    PEAK: alloc=18346MB, reserved=19196MB / 24576MB (78.1%)

    [BREAKDOWN] Largest increase at '02_after_forward' (+17592MB):
      Model params: 73.4MB
      Model grads:  45.9MB
      Optimizer:    146.7MB (estimate (Adam-like: AdamW))
      Batch data:   7.6MB
      Activations:  84.1MB
      Output tensors (top 5):
        features: 43.9MB
        support_features: 31.2MB
        all_logits: 6.0MB
        attention_maps: 1.2MB
        logits: 0.9MB
      Module params (top 5):
        solver: 28.4MB
        hyper_lora: 19.1MB
        context_encoder: 15.0MB
        context_injector: 3.0MB
        hpm: 2.9MB
      Inactive modules: hyperlora, solver_context, cross_attention, hpm, loo, equivariance
  Batch 1/5: loss=0.3336, focal_weighted=0.1975, batch_acc=16.8%, exact=0/100 (0.0%), running_acc=17.2%, lr=5.00e-04
    [TaskTrack] Global_Solved: 1/323, Epoch_Solved: 0, New_Puzzles: 0
    FG: batch=5.9% run50=6.9% | BG: batch=26.5% run50=26.1%
    Per-Color: [0:29% 1:3% 2:6% 3:17% 4:12% 5:10% 6:3% 7:11% 8:1% 9:2%]
    Running50: [0:31% 1:2% 2:4% 3:20% 4:10% 5:12% 6:1% 7:11% 8:1% 9:4%]
  [Batch 2] Grid size: 30x30 (batch_max_size=30)

  [MEMORY] Epoch 2 Batch 2:
    Baseline: alloc=92MB, reserved=17520MB
    01_batch_on_gpu: alloc=213MB (+121MB), reserved=19196MB
    02_after_forward: alloc=13343MB (+13129MB), reserved=19196MB
    03_before_backward: alloc=13418MB (+75MB), reserved=19252MB
    04_after_backward: alloc=275MB (-13143MB), reserved=19252MB
    PEAK: alloc=13885MB, reserved=19252MB / 24576MB (78.3%)

    [BREAKDOWN] Largest increase at '02_after_forward' (+13129MB):
      Model params: 73.4MB
      Model grads:  45.9MB
      Optimizer:    146.7MB (estimate (Adam-like: AdamW))
      Batch data:   4.1MB
      Activations:  68.3MB
      Output tensors (top 5):
        features: 43.9MB
        support_features: 15.6MB
        all_logits: 6.0MB
        attention_maps: 1.2MB
        logits: 0.9MB
      Module params (top 5):
        solver: 28.4MB
        hyper_lora: 19.1MB
        context_encoder: 15.0MB
        context_injector: 3.0MB
        hpm: 2.9MB
      Inactive modules: hyperlora, solver_context, cross_attention, hpm, loo, equivariance
  Batch 2/5: loss=0.9354, focal_weighted=0.7744, batch_acc=18.9%, exact=0/150 (0.0%), running_acc=17.8%, lr=5.00e-04
    [TaskTrack] Global_Solved: 1/328, Epoch_Solved: 0, New_Puzzles: 0
    FG: batch=5.7% run50=6.5% | BG: batch=37.1% run50=29.8%
    Per-Color: [0:38% 1:1% 2:2% 3:10% 4:6% 5:21% 6:6% 7:10% 8:6% 9:2%]
    Running50: [0:33% 1:2% 2:3% 3:16% 4:9% 5:15% 6:3% 7:11% 8:3% 9:3%]
  [Batch 3] Grid size: 30x30 (batch_max_size=30)

  [MEMORY] Epoch 2 Batch 3:
    Baseline: alloc=92MB, reserved=17520MB
    01_batch_on_gpu: alloc=212MB (+119MB), reserved=19252MB
    02_after_forward: alloc=14238MB (+14026MB), reserved=19252MB
    03_before_backward: alloc=14264MB (+26MB), reserved=19252MB
    04_after_backward: alloc=281MB (-13983MB), reserved=19252MB
    PEAK: alloc=14780MB, reserved=19252MB / 24576MB (78.3%)

    [BREAKDOWN] Largest increase at '02_after_forward' (+14026MB):
      Model params: 73.4MB
      Model grads:  45.9MB
      Optimizer:    146.7MB (estimate (Adam-like: AdamW))
      Batch data:   4.8MB
      Activations:  71.5MB
      Output tensors (top 5):
        features: 43.9MB
        support_features: 18.8MB
        all_logits: 6.0MB
        attention_maps: 1.2MB
        logits: 0.9MB
      Module params (top 5):
        solver: 28.4MB
        hyper_lora: 19.1MB
        context_encoder: 15.0MB
        context_injector: 3.0MB
        hpm: 2.9MB
      Inactive modules: hyperlora, solver_context, cross_attention, hpm, loo, equivariance
  Batch 3/5: loss=0.2798, focal_weighted=0.1510, batch_acc=16.5%, exact=0/200 (0.0%), running_acc=17.5%, lr=5.00e-04
    [TaskTrack] Global_Solved: 1/328, Epoch_Solved: 0, New_Puzzles: 0
    FG: batch=6.6% run50=6.5% | BG: batch=25.7% run50=28.8%
    Per-Color: [0:32% 1:2% 2:1% 3:17% 4:5% 5:5% 6:4% 7:12% 8:6% 9:8%]
    Running50: [0:33% 1:2% 2:3% 3:17% 4:8% 5:13% 6:3% 7:11% 8:3% 9:4%]
  [Batch 4] Grid size: 30x30 (batch_max_size=30)

  [MEMORY] Epoch 2 Batch 4:
    Baseline: alloc=92MB, reserved=17520MB
    01_batch_on_gpu: alloc=208MB (+116MB), reserved=19252MB
    02_after_forward: alloc=16017MB (+15808MB), reserved=19252MB
    03_before_backward: alloc=16062MB (+46MB), reserved=19252MB
    04_after_backward: alloc=283MB (-15780MB), reserved=19252MB
    PEAK: alloc=16559MB, reserved=19252MB / 24576MB (78.3%)

    [BREAKDOWN] Largest increase at '02_after_forward' (+15808MB):
      Model params: 73.4MB
      Model grads:  45.9MB
      Optimizer:    146.7MB (estimate (Adam-like: AdamW))
      Batch data:   6.2MB
      Activations:  77.8MB
      Output tensors (top 5):
        features: 43.9MB
        support_features: 25.0MB
        all_logits: 6.0MB
        attention_maps: 1.2MB
        logits: 0.9MB
      Module params (top 5):
        solver: 28.4MB
        hyper_lora: 19.1MB
        context_encoder: 15.0MB
        context_injector: 3.0MB
        hpm: 2.9MB
      Inactive modules: hyperlora, solver_context, cross_attention, hpm, loo, equivariance
  Batch 4/5: loss=0.4504, focal_weighted=0.3024, batch_acc=22.1%, exact=0/250 (0.0%), running_acc=18.4%, lr=5.00e-04
    [TaskTrack] Global_Solved: 1/334, Epoch_Solved: 0, New_Puzzles: 0
    FG: batch=6.3% run50=6.5% | BG: batch=33.3% run50=29.7%
    Per-Color: [0:41% 1:3% 2:2% 3:12% 4:2% 5:11% 6:3% 7:11% 8:6% 9:7%]
    Running50: [0:34% 1:2% 2:3% 3:16% 4:7% 5:12% 6:3% 7:11% 8:4% 9:5%]

  [EPOCH MEMORY] Peak: 305MB alloc, 19252MB reserved (Δ+7MB from prev epoch)

Epoch 3 Summary:
  Total Loss: 0.4512
  Task Loss (focal): 0.3095
  [Global Task Progress] Unique Solved: 1/334 (0.3%)
    NEW puzzles solved this epoch: 0
  Entropy Loss: 4.2593 (weight=0.01)
  Sparsity Loss: 0.1707 (weight=0.5)
  Predicate Loss: 0.0000 (weight=0.01)
  Time: 4.3s, LR: 5.00e-04
    Per-module LRs: DSC:5.00e-04, MSRE:5.00e-04, Other:5.00e-04
  Temperature: 0.9931 (lower=sharper attention)
  Samples Processed: 250 (5 batches)
  Dihedral Distribution: [100.0%, 0.0%, 0.0%, 0.0%, 0.0%, 0.0%, 0.0%, 0.0%]
    [!] Non-uniform dihedral distribution (max dev: 87.5%)
  Color Permutation: 0.0% (0/250)
  Translational Aug: 0.0% (0/250), unique offsets: 0
  Aug Quality: OK
  --- Training Diagnostics ---
  Solver Steps: 7 (deep supervision active)
  Per-Step Loss (epoch avg, 1 batches): [2.1406, 2.1406, 2.1406, 2.1406, 2.1406, 2.1406, 2.1406]
    [!] SOLVER DEGRADATION: Step 0 is best! Later steps 0.0% worse!
    [!] Best: step 0 (2.1406), Worst: step 0 (2.1406)
  Best-Step Histogram: [s0:1]
  Solver Health: Last step best: 0.0%, Earlier step best: 100.0%
  Avg Step Improvement: 0.0% (step0→stepN)
    ⚠️ SOLVER OVER-ITERATION WARNING: 100% of batches had earlier step as best!
    ⚠️ Consider: (1) enabling best-step selection, (2) reducing num_solver_steps, (3) enabling ACT
  Grad Norms: DSC=0.0000, StopPred=0.0000, Encoder=0.0000, Solver=0.0000
  StopPred Weight Var: 2.80e-02
  Attention: max=0.0415, min=0.000000
  Stop Prob: 0.248 (approx 5.3 clues active)
  Stop Probs Std: 0.087 (global std across batch×clues)
  Clues Used: mean=5.26, std=0.35, range=[4.2, 5.9]
  Clue-Loss Correlation: -0.026 (weak - per-sample coupling may need tuning)
  Stop Logits: mean=-1.16, std=0.47, range=[-2.8, 0.3]
    Clue count varies by task (per-sample coupling active)
  Per-Clue Entropy: [3.83, 3.82, 3.82, 3.82, 3.82, 3.82, 3.82] (mean=3.82, max=6.80)
    [!] Clues have uniform entropy (std=0.003) - not differentiating!
  Centroid Spread: 0.07 (higher=more diverse)
    🚨 CRITICAL COLLAPSE: Spread=0.07 < 0.5 - all clues at same location!
    [!] Stop predictor cannot differentiate - needs diversity regularizer
  --- Stop Predictor Coupling ---
  Entropy Input to Stop: 0.5618 (normalized, lower=sharper)
  Per-Clue Entropy Input: [0.562, 0.562, 0.561, 0.561, 0.562, 0.562, 0.562]
  Per-Clue Stop Prob: [0.252, 0.252, 0.248, 0.236, 0.243, 0.256, 0.254]
  --- Sparsity Loss Breakdown (Per-Sample Coupled) ---
  Min Clue Penalty: 0.0000 (per-sample avg)
  Per-Sample Clue Penalty (scaled): 0.0000
  Base Pondering: 0.1053 (clues=5.26)
  Entropy Pondering: 0.0591
  --- Per-Class Distribution (Valid Pixels Only) ---
  Padding: 94.3% of grid (ignored in loss)
  Pred %: [29.9, 1.4, 2.6, 14.6, 5.4, 22.7, 1.2, 15.4, 1.7, 5.2]
  Target %: [61.3, 10.1, 7.6, 6.8, 2.8, 2.5, 0.4, 3.4, 4.4, 0.6]
  Per-Class Acc %: [36, 2, 2, 13, 6, 14, 5, 11, 5, 3]
  ==================================================
  PER-SAMPLE TRAINING ACCURACY (Epoch 3)
  ==================================================
  ★ Mean Accuracy: 18.4%
  ★ Exact Match: 0/250 (0.0%)
  ★ High Acc (≥90%): 0/250 (0.0%)
  FG Accuracy: 6.5%
  BG Accuracy: 29.7%
  Batch Trend: 17.7% → 19.3% (↑ 1.6pp)
  Accuracy Distribution: 0-25%:75%, 25-50%:25%, 50-75%:0%, 75-90%:0%, 90-100%:0%
    [!] Many samples stuck at low accuracy - check data/model!
  Running Window (last 5 batches): 18.4% ± 2.0%

  --- PER-COLOR ACCURACY (10 classes) ---
  Color:       0     1     2     3     4     5     6     7     8     9
  Acc%:      36%    2%    2%   13%    6%   14%    5%   11%    5%    3%
  Target:  56.0%  8.4%  6.3%  7.1%  4.3%  4.5%  2.4%  2.6%  6.7%  1.7%
  Pred:    32.6%  1.3%  2.5% 13.9%  6.1% 20.5%  1.9% 14.7%  1.9%  4.7%
  [!] Weak colors (<50% acc): 0(Black), 1(Blue), 2(Red), 3(Green), 4(Yellow), 5(Gray), 6(Pink), 7(Orange), 8(Cyan), 9(Brown)
  [!] Under-predicting color 1 (Blue): 1.3% vs target 8.4%
  [!] Under-predicting color 2 (Red): 2.5% vs target 6.3%
  [!] Under-predicting color 8 (Cyan): 1.9% vs target 6.7%
  ==================================================

  ==================================================
  LEARNING TRAJECTORY (Epoch 3)
  ==================================================
  Stop Prob:   0.248 ↑ (init=0.27, task-dependent)
  Exp. Clues:  5.26 (latent variable, task-dependent)
  Attn Entropy: 3.82 ↑ (max=6.8, sharper=better)
  Task Loss:   0.3095 ↑
  Train Acc:   18.4% →
  Exact Match: 0.0% ↓
  Best Step:   0 (later=better refinement)
  FG Coverage: 181.2% of target ↓
  ==================================================

  ############################################################
  🚦 TRAINING HEALTH CHECK - Epoch 3
  ############################################################
  ✓ Attention sharpening (0.56 < 0.7)
  ✓ No NaN/Inf issues
  ✓ No color mode collapse
  ⚠ Stop probs nearly uniform (global=0.087, per_clue=0.006)
  ⚠ Negative coupling (r=-0.63) - early epoch OK
  ⚠ Loss slowly decreasing (0.311 → 0.310)
  ⚠ Accuracy flat (19.9% → 18.4%) - early epoch
  ✗ Centroid COLLAPSE (0.1 < 0.5)
  --------------------------------------------------------
  RESULT: 3/8 checks passed
  STATUS: 🟠 WARNING
  → Multiple issues detected. Consider intervention soon.

  RECOMMENDED ACTIONS:
  ############################################################
  HPM Instance Buffer: 0 entries (empty, not saved)
  HPM Procedural Buffer: 0 entries (empty, not saved)
  Saved checkpoint to checkpoints\rlan_stable_merged\latest.pt
  [ARCDataset] Augmentation config updated:
    dihedral=False, color_perm=False (prob=0.50), translational=False

Epoch 4/200
----------------------------------------
  [Batch 0] Grid size: 30x30 (batch_max_size=30)

  [MEMORY] Epoch 3 Batch 0:
    Baseline: alloc=92MB, reserved=19252MB
    01_batch_on_gpu: alloc=101MB (+9MB), reserved=19274MB
    02_after_forward: alloc=14130MB (+14029MB), reserved=19274MB
    03_before_backward: alloc=14156MB (+26MB), reserved=19274MB
    04_after_backward: alloc=222MB (-13934MB), reserved=19274MB
    PEAK: alloc=14672MB, reserved=19274MB / 24576MB (78.4%)

    [BREAKDOWN] Largest increase at '02_after_forward' (+14029MB):
      Model params: 73.4MB
      Model grads:  0.0MB
      Optimizer:    146.7MB (estimate (Adam-like: AdamW))
      Batch data:   4.8MB
      Activations:  71.5MB
      Output tensors (top 5):
        features: 43.9MB
        support_features: 18.8MB
        all_logits: 6.0MB
        attention_maps: 1.2MB
        logits: 0.9MB
      Module params (top 5):
        solver: 28.4MB
        hyper_lora: 19.1MB
        context_encoder: 15.0MB
        context_injector: 3.0MB
        hpm: 2.9MB
      Inactive modules: hyperlora, solver_context, cross_attention, hpm, loo, equivariance
  Batch 0/5: loss=0.2947, focal_weighted=0.1695, batch_acc=15.8%, exact=0/50 (0.0%), running_acc=15.8%, lr=5.00e-04
    [TaskTrack] Global_Solved: 1/334, Epoch_Solved: 0, New_Puzzles: 0
    FG: batch=8.9% run50=8.9% | BG: batch=21.5% run50=21.5%
    Per-Color: [0:25% 1:4% 2:3% 3:30% 4:7% 5:14% 6:6% 7:14% 8:0% 9:6%]
    Running50: [0:25% 1:4% 2:3% 3:30% 4:7% 5:14% 6:6% 7:14% 8:0% 9:6%]
    Solver: [3.000, 2.984, 2.984, 2.984, 2.984, 2.984, 2.984] ⚠ best=1
  [Batch 1] Grid size: 30x30 (batch_max_size=30)

  [MEMORY] Epoch 3 Batch 1:
    Baseline: alloc=92MB, reserved=19252MB
    01_batch_on_gpu: alloc=210MB (+118MB), reserved=19276MB
    02_after_forward: alloc=13341MB (+13131MB), reserved=19276MB
    03_before_backward: alloc=13426MB (+84MB), reserved=19276MB
    04_after_backward: alloc=275MB (-13150MB), reserved=19276MB
    PEAK: alloc=13884MB, reserved=19276MB / 24576MB (78.4%)

    [BREAKDOWN] Largest increase at '02_after_forward' (+13131MB):
      Model params: 73.4MB
      Model grads:  45.9MB
      Optimizer:    146.7MB (estimate (Adam-like: AdamW))
      Batch data:   4.1MB
      Activations:  68.3MB
      Output tensors (top 5):
        features: 43.9MB
        support_features: 15.6MB
        all_logits: 6.0MB
        attention_maps: 1.2MB
        logits: 0.9MB
      Module params (top 5):
        solver: 28.4MB
        hyper_lora: 19.1MB
        context_encoder: 15.0MB
        context_injector: 3.0MB
        hpm: 2.9MB
      Inactive modules: hyperlora, solver_context, cross_attention, hpm, loo, equivariance
  Batch 1/5: loss=0.6846, focal_weighted=0.5227, batch_acc=21.7%, exact=0/100 (0.0%), running_acc=18.7%, lr=5.00e-04
    [TaskTrack] Global_Solved: 1/336, Epoch_Solved: 0, New_Puzzles: 0
    FG: batch=5.7% run50=7.3% | BG: batch=38.7% run50=30.1%
    Per-Color: [0:40% 1:1% 2:4% 3:8% 4:8% 5:10% 6:5% 7:5% 8:6% 9:5%]
    Running50: [0:33% 1:2% 2:4% 3:19% 4:8% 5:12% 6:5% 7:9% 8:3% 9:6%]
  [Batch 2] Grid size: 30x30 (batch_max_size=30)

  [MEMORY] Epoch 3 Batch 2:
    Baseline: alloc=92MB, reserved=19252MB
    01_batch_on_gpu: alloc=213MB (+120MB), reserved=19276MB
    02_after_forward: alloc=15128MB (+14916MB), reserved=19276MB
    03_before_backward: alloc=15153MB (+24MB), reserved=19276MB
    04_after_backward: alloc=285MB (-14868MB), reserved=19276MB
    PEAK: alloc=15670MB, reserved=19276MB / 24576MB (78.4%)

    [BREAKDOWN] Largest increase at '02_after_forward' (+14916MB):
      Model params: 73.4MB
      Model grads:  45.9MB
      Optimizer:    146.7MB (estimate (Adam-like: AdamW))
      Batch data:   5.5MB
      Activations:  74.6MB
      Output tensors (top 5):
        features: 43.9MB
        support_features: 21.9MB
        all_logits: 6.0MB
        attention_maps: 1.2MB
        logits: 0.9MB
      Module params (top 5):
        solver: 28.4MB
        hyper_lora: 19.1MB
        context_encoder: 15.0MB
        context_injector: 3.0MB
        hpm: 2.9MB
      Inactive modules: hyperlora, solver_context, cross_attention, hpm, loo, equivariance
  Batch 2/5: loss=0.2628, focal_weighted=0.1324, batch_acc=17.6%, exact=1/150 (0.7%), running_acc=18.4%, lr=5.00e-04
    [TaskTrack] Global_Solved: 1/336, Epoch_Solved: 1, New_Puzzles: 0
    FG: batch=8.9% run50=7.8% | BG: batch=24.5% run50=28.3%
    Per-Color: [0:30% 1:0% 2:2% 3:30% 4:2% 5:7% 6:2% 7:18% 8:3% 9:9%]
    Running50: [0:32% 1:2% 2:3% 3:23% 4:6% 5:11% 6:4% 7:12% 8:3% 9:7%]
  [Batch 3] Grid size: 30x30 (batch_max_size=30)

  [MEMORY] Epoch 3 Batch 3:
    Baseline: alloc=92MB, reserved=19252MB
    01_batch_on_gpu: alloc=212MB (+120MB), reserved=19276MB
    02_after_forward: alloc=13343MB (+13131MB), reserved=19276MB
    03_before_backward: alloc=13389MB (+46MB), reserved=19276MB
    04_after_backward: alloc=276MB (-13113MB), reserved=19276MB
    PEAK: alloc=13886MB, reserved=19276MB / 24576MB (78.4%)

    [BREAKDOWN] Largest increase at '02_after_forward' (+13131MB):
      Model params: 73.4MB
      Model grads:  45.9MB
      Optimizer:    146.7MB (estimate (Adam-like: AdamW))
      Batch data:   4.1MB
      Activations:  68.3MB
      Output tensors (top 5):
        features: 43.9MB
        support_features: 15.6MB
        all_logits: 6.0MB
        attention_maps: 1.2MB
        logits: 0.9MB
      Module params (top 5):
        solver: 28.4MB
        hyper_lora: 19.1MB
        context_encoder: 15.0MB
        context_injector: 3.0MB
        hpm: 2.9MB
      Inactive modules: hyperlora, solver_context, cross_attention, hpm, loo, equivariance
  Batch 3/5: loss=0.4230, focal_weighted=0.2785, batch_acc=25.5%, exact=1/200 (0.5%), running_acc=20.2%, lr=5.00e-04
    [TaskTrack] Global_Solved: 1/339, Epoch_Solved: 1, New_Puzzles: 0
    FG: batch=6.9% run50=7.6% | BG: batch=36.9% run50=30.4%
    Per-Color: [0:42% 1:1% 2:3% 3:11% 4:4% 5:17% 6:8% 7:21% 8:2% 9:3%]
    Running50: [0:34% 1:1% 2:3% 3:20% 4:5% 5:12% 6:5% 7:14% 8:3% 9:6%]
  [Batch 4] Grid size: 30x30 (batch_max_size=30)

  [MEMORY] Epoch 3 Batch 4:
    Baseline: alloc=92MB, reserved=19252MB
    01_batch_on_gpu: alloc=207MB (+114MB), reserved=19276MB
    02_after_forward: alloc=13339MB (+13132MB), reserved=19276MB
    03_before_backward: alloc=13366MB (+27MB), reserved=19276MB
    04_after_backward: alloc=271MB (-13095MB), reserved=19276MB
    PEAK: alloc=13881MB, reserved=19276MB / 24576MB (78.4%)

    [BREAKDOWN] Largest increase at '02_after_forward' (+13132MB):
      Model params: 73.4MB
      Model grads:  45.9MB
      Optimizer:    146.7MB (estimate (Adam-like: AdamW))
      Batch data:   4.1MB
      Activations:  68.3MB
      Output tensors (top 5):
        features: 43.9MB
        support_features: 15.6MB
        all_logits: 6.0MB
        attention_maps: 1.2MB
        logits: 0.9MB
      Module params (top 5):
        solver: 28.4MB
        hyper_lora: 19.1MB
        context_encoder: 15.0MB
        context_injector: 3.0MB
        hpm: 2.9MB
      Inactive modules: hyperlora, solver_context, cross_attention, hpm, loo, equivariance
  Batch 4/5: loss=0.2726, focal_weighted=0.1411, batch_acc=18.2%, exact=1/250 (0.4%), running_acc=19.8%, lr=5.00e-04
    [TaskTrack] Global_Solved: 1/339, Epoch_Solved: 1, New_Puzzles: 0
    FG: batch=5.5% run50=7.2% | BG: batch=28.0% run50=30.0%
    Per-Color: [0:35% 1:2% 2:4% 3:23% 4:8% 5:5% 6:3% 7:11% 8:1% 9:15%]
    Running50: [0:34% 1:1% 2:3% 3:20% 4:6% 5:11% 6:5% 7:14% 8:2% 9:8%]

  [EPOCH MEMORY] Peak: 284MB alloc, 19276MB reserved (Δ-20MB from prev epoch)

Epoch 4 Summary:
  Total Loss: 0.3876
  Task Loss (focal): 0.2488
  [Global Task Progress] Unique Solved: 1/339 (0.3%)
    NEW puzzles solved this epoch: 0
  Entropy Loss: 4.2208 (weight=0.01)
  Sparsity Loss: 0.1656 (weight=0.5)
  Predicate Loss: 0.0000 (weight=0.01)
  Time: 4.0s, LR: 5.00e-04
    Per-module LRs: DSC:5.00e-04, MSRE:5.00e-04, Other:5.00e-04
  Temperature: 0.9897 (lower=sharper attention)
  Samples Processed: 250 (5 batches)
  Dihedral Distribution: [100.0%, 0.0%, 0.0%, 0.0%, 0.0%, 0.0%, 0.0%, 0.0%]
    [!] Non-uniform dihedral distribution (max dev: 87.5%)
  Color Permutation: 0.0% (0/250)
  Translational Aug: 0.0% (0/250), unique offsets: 0
  Aug Quality: OK
  --- Training Diagnostics ---
  Solver Steps: 7 (deep supervision active)
  Per-Step Loss (epoch avg, 1 batches): [3.0000, 2.9844, 2.9844, 2.9844, 2.9844, 2.9844, 2.9844]
    [!] Best step is 1 (middle), not last - solver may be over-iterating!
    [!] Best: step 1 (2.9844), Final: step 6 (2.9844)
    [!] Potential gain from best-step selection: 0.0% lower loss
  Best-Step Histogram: []
  Solver Health: Last step best: 0.0%, Earlier step best: 100.0%
  Avg Step Improvement: 0.5% (step0→stepN)
    ⚠️ SOLVER OVER-ITERATION WARNING: 100% of batches had earlier step as best!
    ⚠️ Consider: (1) enabling best-step selection, (2) reducing num_solver_steps, (3) enabling ACT
  Grad Norms: DSC=0.0000, StopPred=0.0000, Encoder=0.0000, Solver=0.0000
  StopPred Weight Var: 2.80e-02
  Attention: max=0.0607, min=0.000000
  Stop Prob: 0.277 (approx 5.1 clues active)
  Stop Probs Std: 0.095 (global std across batch×clues)
  Clues Used: mean=5.06, std=0.32, range=[4.4, 5.7]
  Clue-Loss Correlation: +0.142 (learning - per-sample coupling active)
  Stop Logits: mean=-1.02, std=0.50, range=[-2.8, 0.4]
    Clue count varies by task (per-sample coupling active)
  Per-Clue Entropy: [3.48, 3.48, 3.47, 3.47, 3.47, 3.48, 3.48] (mean=3.48, max=6.80)
    [!] Clues have uniform entropy (std=0.002) - not differentiating!
  Centroid Spread: 0.06 (higher=more diverse)
    🚨 CRITICAL COLLAPSE: Spread=0.06 < 0.5 - all clues at same location!
    [!] Stop predictor cannot differentiate - needs diversity regularizer
  --- Stop Predictor Coupling ---
  Entropy Input to Stop: 0.5110 (normalized, lower=sharper)
  Per-Clue Entropy Input: [0.511, 0.511, 0.511, 0.511, 0.511, 0.511, 0.511]
  Per-Clue Stop Prob: [0.297, 0.289, 0.266, 0.271, 0.277, 0.268, 0.273]
  --- Sparsity Loss Breakdown (Per-Sample Coupled) ---
  Min Clue Penalty: 0.0000 (per-sample avg)
  Per-Sample Clue Penalty (scaled): 0.0000
  Base Pondering: 0.1012 (clues=5.06)
  Entropy Pondering: 0.0518
  --- Per-Class Distribution (Valid Pixels Only) ---
  Padding: 94.3% of grid (ignored in loss)
  Pred %: [24.9, 1.5, 2.6, 21.0, 7.7, 16.8, 2.0, 16.5, 1.9, 5.1]
  Target %: [54.1, 8.3, 8.0, 4.3, 4.3, 10.6, 3.2, 1.7, 3.8, 1.9]
  Per-Class Acc %: [36, 2, 3, 13, 7, 12, 6, 11, 4, 6]
  ==================================================
  PER-SAMPLE TRAINING ACCURACY (Epoch 4)
  ==================================================
  ★ Mean Accuracy: 19.8%
  ★ Exact Match: 1/250 (0.4%)
  ★ High Acc (≥90%): 1/250 (0.4%)
  FG Accuracy: 7.2%
  BG Accuracy: 30.0%
  Batch Trend: 15.8% → 21.9% (↑ 6.0pp)
    ✓ Accuracy improving within epoch - learning is active!
  Accuracy Distribution: 0-25%:50%, 25-50%:50%, 50-75%:0%, 75-90%:0%, 90-100%:0%
  Running Window (last 5 batches): 19.8% ± 3.4%

  --- PER-COLOR ACCURACY (10 classes) ---
  Color:       0     1     2     3     4     5     6     7     8     9
  Acc%:      36%    2%    3%   13%    7%   12%    6%   11%    4%    6%
  Target:  57.6%  7.3%  6.3%  7.3%  4.7%  6.1%  2.2%  2.1%  5.4%  0.9%
  Pred:    34.1%  1.2%  2.1% 12.9%  5.5% 21.6%  1.9% 14.8%  1.6%  4.2%
  [!] Weak colors (<50% acc): 0(Black), 1(Blue), 2(Red), 3(Green), 4(Yellow), 5(Gray), 6(Pink), 7(Orange), 8(Cyan), 9(Brown)
  [!] Under-predicting color 1 (Blue): 1.2% vs target 7.3%
  [!] Under-predicting color 2 (Red): 2.1% vs target 6.3%
  [!] Under-predicting color 8 (Cyan): 1.6% vs target 5.4%
  ==================================================

  ==================================================
  LEARNING TRAJECTORY (Epoch 4)
  ==================================================
  Stop Prob:   0.277 ↑ (init=0.27, task-dependent)
  Exp. Clues:  5.06 (latent variable, task-dependent)
  Attn Entropy: 3.48 ↓ (max=6.8, sharper=better)
  Task Loss:   0.2488 ↓
  Train Acc:   19.8% ↑
  Exact Match: 0.4% ↑
  Best Step:   1 (later=better refinement)
  FG Coverage: 163.5% of target ↓
  ==================================================

  ############################################################
  🚦 TRAINING HEALTH CHECK - Epoch 4
  ############################################################
  ✓ Attention sharpening (0.51 < 0.7)
  ✓ No NaN/Inf issues
  ✓ No color mode collapse
  ⚠ Stop probs nearly uniform (global=0.095, per_clue=0.011)
  ⚠ Negative coupling (r=-0.23) - early epoch OK
  ⚠ Loss slowly decreasing (0.311 → 0.249)
  ⚠ Accuracy flat (19.9% → 19.8%) - early epoch
  ✗ Centroid COLLAPSE (0.1 < 0.5)
  --------------------------------------------------------
  RESULT: 3/8 checks passed
  STATUS: 🟠 WARNING
  → Multiple issues detected. Consider intervention soon.

  RECOMMENDED ACTIONS:
  ############################################################
  HPM Instance Buffer: 0 entries (empty, not saved)
  HPM Procedural Buffer: 0 entries (empty, not saved)
  Saved checkpoint to checkpoints\rlan_stable_merged\latest.pt
  [ARCDataset] Augmentation config updated:
    dihedral=False, color_perm=False (prob=0.50), translational=False

Epoch 5/200
----------------------------------------
  [Batch 0] Grid size: 30x30 (batch_max_size=30)

  [MEMORY] Epoch 4 Batch 0:
    Baseline: alloc=92MB, reserved=19276MB
    01_batch_on_gpu: alloc=103MB (+11MB), reserved=19298MB
    02_after_forward: alloc=14132MB (+14029MB), reserved=19298MB
    03_before_backward: alloc=14160MB (+28MB), reserved=19298MB
    04_after_backward: alloc=224MB (-13936MB), reserved=19298MB
    PEAK: alloc=14675MB, reserved=19298MB / 24576MB (78.5%)

    [BREAKDOWN] Largest increase at '02_after_forward' (+14029MB):
      Model params: 73.4MB
      Model grads:  0.0MB
      Optimizer:    146.7MB (estimate (Adam-like: AdamW))
      Batch data:   4.8MB
      Activations:  71.5MB
      Output tensors (top 5):
        features: 43.9MB
        support_features: 18.8MB
        all_logits: 6.0MB
        attention_maps: 1.2MB
        logits: 0.9MB
      Module params (top 5):
        solver: 28.4MB
        hyper_lora: 19.1MB
        context_encoder: 15.0MB
        context_injector: 3.0MB
        hpm: 2.9MB
      Inactive modules: hyperlora, solver_context, cross_attention, hpm, loo, equivariance
  Batch 0/5: loss=0.2902, focal_weighted=0.1594, batch_acc=17.1%, exact=0/50 (0.0%), running_acc=17.1%, lr=5.00e-04
    [TaskTrack] Global_Solved: 1/339, Epoch_Solved: 0, New_Puzzles: 0
    FG: batch=4.5% run50=4.5% | BG: batch=28.6% run50=28.6%
    Per-Color: [0:33% 1:1% 2:1% 3:14% 4:7% 5:7% 6:0% 7:9% 8:0% 9:7%]
    Running50: [0:33% 1:1% 2:1% 3:14% 4:7% 5:7% 6:0% 7:9% 8:0% 9:7%]
    Solver: [2.469, 2.469, 2.469, 2.469, 2.469, 2.469, 2.469] ⚠ best=0
  [Batch 1] Grid size: 30x30 (batch_max_size=30)

  [MEMORY] Epoch 4 Batch 1:
    Baseline: alloc=92MB, reserved=19276MB
    01_batch_on_gpu: alloc=211MB (+118MB), reserved=19300MB
    02_after_forward: alloc=16017MB (+15807MB), reserved=19300MB
    03_before_backward: alloc=16042MB (+25MB), reserved=19300MB
    04_after_backward: alloc=285MB (-15757MB), reserved=19300MB
    PEAK: alloc=16559MB, reserved=19300MB / 24576MB (78.5%)

    [BREAKDOWN] Largest increase at '02_after_forward' (+15807MB):
      Model params: 73.4MB
      Model grads:  45.9MB
      Optimizer:    146.7MB (estimate (Adam-like: AdamW))
      Batch data:   6.2MB
      Activations:  77.8MB
      Output tensors (top 5):
        features: 43.9MB
        support_features: 25.0MB
        all_logits: 6.0MB
        attention_maps: 1.2MB
        logits: 0.9MB
      Module params (top 5):
        solver: 28.4MB
        hyper_lora: 19.1MB
        context_encoder: 15.0MB
        context_injector: 3.0MB
        hpm: 2.9MB
      Inactive modules: hyperlora, solver_context, cross_attention, hpm, loo, equivariance
  Batch 1/5: loss=0.2779, focal_weighted=0.1498, batch_acc=15.2%, exact=0/100 (0.0%), running_acc=16.1%, lr=5.00e-04
    [TaskTrack] Global_Solved: 1/339, Epoch_Solved: 0, New_Puzzles: 0
    FG: batch=7.2% run50=5.8% | BG: batch=21.5% run50=25.0%
    Per-Color: [0:29% 1:6% 2:3% 3:22% 4:10% 5:14% 6:3% 7:9% 8:7% 9:11%]
    Running50: [0:31% 1:4% 2:2% 3:18% 4:9% 5:10% 6:2% 7:9% 8:3% 9:9%]
  [Batch 2] Grid size: 30x30 (batch_max_size=30)

  [MEMORY] Epoch 4 Batch 2:
    Baseline: alloc=92MB, reserved=19276MB
    01_batch_on_gpu: alloc=214MB (+122MB), reserved=19300MB
    02_after_forward: alloc=13345MB (+13131MB), reserved=19300MB
    03_before_backward: alloc=13425MB (+81MB), reserved=19300MB
    04_after_backward: alloc=277MB (-13148MB), reserved=19300MB
    PEAK: alloc=13887MB, reserved=19300MB / 24576MB (78.5%)

    [BREAKDOWN] Largest increase at '02_after_forward' (+13131MB):
      Model params: 73.4MB
      Model grads:  45.9MB
      Optimizer:    146.7MB (estimate (Adam-like: AdamW))
      Batch data:   4.1MB
      Activations:  68.3MB
      Output tensors (top 5):
        features: 43.9MB
        support_features: 15.6MB
        all_logits: 6.0MB
        attention_maps: 1.2MB
        logits: 0.9MB
      Module params (top 5):
        solver: 28.4MB
        hyper_lora: 19.1MB
        context_encoder: 15.0MB
        context_injector: 3.0MB
        hpm: 2.9MB
      Inactive modules: hyperlora, solver_context, cross_attention, hpm, loo, equivariance
  Batch 2/5: loss=0.8563, focal_weighted=0.6965, batch_acc=22.2%, exact=0/150 (0.0%), running_acc=18.1%, lr=5.00e-04
    [TaskTrack] Global_Solved: 1/340, Epoch_Solved: 0, New_Puzzles: 0
    FG: batch=6.0% run50=5.9% | BG: batch=39.0% run50=29.7%
    Per-Color: [0:41% 1:0% 2:4% 3:12% 4:8% 5:14% 6:5% 7:12% 8:6% 9:8%]
    Running50: [0:34% 1:3% 2:2% 3:16% 4:8% 5:11% 6:3% 7:10% 8:4% 9:9%]
  [Batch 3] Grid size: 30x30 (batch_max_size=30)

  [MEMORY] Epoch 4 Batch 3:
    Baseline: alloc=92MB, reserved=19276MB
    01_batch_on_gpu: alloc=216MB (+124MB), reserved=19300MB
    02_after_forward: alloc=16025MB (+15809MB), reserved=19300MB
    03_before_backward: alloc=16073MB (+48MB), reserved=19300MB
    04_after_backward: alloc=291MB (-15782MB), reserved=19300MB
    PEAK: alloc=16567MB, reserved=19300MB / 24576MB (78.5%)

    [BREAKDOWN] Largest increase at '02_after_forward' (+15809MB):
      Model params: 73.4MB
      Model grads:  45.9MB
      Optimizer:    146.7MB (estimate (Adam-like: AdamW))
      Batch data:   6.2MB
      Activations:  77.8MB
      Output tensors (top 5):
        features: 43.9MB
        support_features: 25.0MB
        all_logits: 6.0MB
        attention_maps: 1.2MB
        logits: 0.9MB
      Module params (top 5):
        solver: 28.4MB
        hyper_lora: 19.1MB
        context_encoder: 15.0MB
        context_injector: 3.0MB
        hpm: 2.9MB
      Inactive modules: hyperlora, solver_context, cross_attention, hpm, loo, equivariance
  Batch 3/5: loss=0.4485, focal_weighted=0.2977, batch_acc=25.2%, exact=0/200 (0.0%), running_acc=19.9%, lr=5.00e-04
    [TaskTrack] Global_Solved: 1/340, Epoch_Solved: 0, New_Puzzles: 0
    FG: batch=6.5% run50=6.0% | BG: batch=36.3% run50=31.3%
    Per-Color: [0:41% 1:0% 2:1% 3:18% 4:6% 5:6% 6:4% 7:17% 8:5% 9:6%]
    Running50: [0:36% 1:2% 2:2% 3:16% 4:8% 5:10% 6:3% 7:12% 8:5% 9:8%]
  [Batch 4] Grid size: 30x30 (batch_max_size=30)

  [MEMORY] Epoch 4 Batch 4:
    Baseline: alloc=92MB, reserved=19276MB
    01_batch_on_gpu: alloc=212MB (+120MB), reserved=19300MB
    02_after_forward: alloc=17803MB (+17591MB), reserved=19300MB
    03_before_backward: alloc=17828MB (+25MB), reserved=19300MB
    04_after_backward: alloc=292MB (-17537MB), reserved=19300MB
    PEAK: alloc=18345MB, reserved=19300MB / 24576MB (78.5%)

    [BREAKDOWN] Largest increase at '02_after_forward' (+17591MB):
      Model params: 73.4MB
      Model grads:  45.9MB
      Optimizer:    146.7MB (estimate (Adam-like: AdamW))
      Batch data:   7.6MB
      Activations:  84.1MB
      Output tensors (top 5):
        features: 43.9MB
        support_features: 31.2MB
        all_logits: 6.0MB
        attention_maps: 1.2MB
        logits: 0.9MB
      Module params (top 5):
        solver: 28.4MB
        hyper_lora: 19.1MB
        context_encoder: 15.0MB
        context_injector: 3.0MB
        hpm: 2.9MB
      Inactive modules: hyperlora, solver_context, cross_attention, hpm, loo, equivariance
  Batch 4/5: loss=0.2661, focal_weighted=0.1292, batch_acc=19.0%, exact=0/250 (0.0%), running_acc=19.7%, lr=5.00e-04
    [TaskTrack] Global_Solved: 1/340, Epoch_Solved: 0, New_Puzzles: 0
    FG: batch=9.2% run50=6.7% | BG: batch=27.3% run50=30.5%
    Per-Color: [0:31% 1:2% 2:1% 3:27% 4:11% 5:12% 6:0% 7:11% 8:1% 9:7%]
    Running50: [0:35% 1:2% 2:2% 3:18% 4:8% 5:10% 6:2% 7:12% 8:4% 9:8%]

  [EPOCH MEMORY] Peak: 321MB alloc, 19300MB reserved (Δ+36MB from prev epoch)

Epoch 5 Summary:
  Total Loss: 0.4278
  Task Loss (focal): 0.2865
  [Global Task Progress] Unique Solved: 1/340 (0.3%)
    NEW puzzles solved this epoch: 0
  Entropy Loss: 4.2325 (weight=0.01)
  Sparsity Loss: 0.1704 (weight=0.5)
  Predicate Loss: 0.0000 (weight=0.01)
  Time: 4.3s, LR: 5.00e-04
    Per-module LRs: DSC:5.00e-04, MSRE:5.00e-04, Other:5.00e-04
  Temperature: 0.9862 (lower=sharper attention)
  Samples Processed: 250 (5 batches)
  Dihedral Distribution: [100.0%, 0.0%, 0.0%, 0.0%, 0.0%, 0.0%, 0.0%, 0.0%]
    [!] Non-uniform dihedral distribution (max dev: 87.5%)
  Color Permutation: 0.0% (0/250)
  Translational Aug: 0.0% (0/250), unique offsets: 0
  Aug Quality: OK
  --- Training Diagnostics ---
  Solver Steps: 7 (deep supervision active)
  Per-Step Loss (epoch avg, 1 batches): [2.4688, 2.4688, 2.4688, 2.4688, 2.4688, 2.4688, 2.4688]
    [!] SOLVER DEGRADATION: Step 0 is best! Later steps 0.0% worse!
    [!] Best: step 0 (2.4688), Worst: step 0 (2.4688)
  Best-Step Histogram: [s0:1]
  Solver Health: Last step best: 0.0%, Earlier step best: 100.0%
  Avg Step Improvement: 0.0% (step0→stepN)
    ⚠️ SOLVER OVER-ITERATION WARNING: 100% of batches had earlier step as best!
    ⚠️ Consider: (1) enabling best-step selection, (2) reducing num_solver_steps, (3) enabling ACT
  Grad Norms: DSC=0.0000, StopPred=0.0000, Encoder=0.0000, Solver=0.0000
  StopPred Weight Var: 2.80e-02
  Attention: max=0.0479, min=0.000000
  Stop Prob: 0.271 (approx 5.1 clues active)
  Stop Probs Std: 0.091 (global std across batch×clues)
  Clues Used: mean=5.10, std=0.33, range=[4.1, 5.7]
  Clue-Loss Correlation: -0.147 (unexpected negative - check gradient flow)
  Stop Logits: mean=-1.05, std=0.51, range=[-2.6, 0.3]
    Clue count varies by task (per-sample coupling active)
  Per-Clue Entropy: [3.76, 3.76, 3.76, 3.76, 3.76, 3.76, 3.76] (mean=3.76, max=6.80)
    [!] Clues have uniform entropy (std=0.002) - not differentiating!
  Centroid Spread: 0.06 (higher=more diverse)
    🚨 CRITICAL COLLAPSE: Spread=0.06 < 0.5 - all clues at same location!
    [!] Stop predictor cannot differentiate - needs diversity regularizer
  --- Stop Predictor Coupling ---
  Entropy Input to Stop: 0.5527 (normalized, lower=sharper)
  Per-Clue Entropy Input: [0.553, 0.553, 0.552, 0.552, 0.553, 0.553, 0.553]
  Per-Clue Stop Prob: [0.277, 0.281, 0.258, 0.254, 0.262, 0.289, 0.281]
  --- Sparsity Loss Breakdown (Per-Sample Coupled) ---
  Min Clue Penalty: 0.0000 (per-sample avg)
  Per-Sample Clue Penalty (scaled): 0.0000
  Base Pondering: 0.1020 (clues=5.10)
  Entropy Pondering: 0.0565
  --- Per-Class Distribution (Valid Pixels Only) ---
  Padding: 93.6% of grid (ignored in loss)
  Pred %: [32.4, 1.6, 2.3, 14.7, 5.5, 20.2, 1.3, 15.9, 1.5, 4.8]
  Target %: [60.1, 5.3, 10.1, 6.1, 5.2, 3.2, 0.9, 3.5, 4.7, 0.9]
  Per-Class Acc %: [37, 1, 2, 16, 8, 11, 4, 12, 5, 8]
  [WARN] FG color preference: 31% are color 0
  ==================================================
  PER-SAMPLE TRAINING ACCURACY (Epoch 5)
  ==================================================
  ★ Mean Accuracy: 19.7%
  ★ Exact Match: 0/250 (0.0%)
  ★ High Acc (≥90%): 0/250 (0.0%)
  FG Accuracy: 6.7%
  BG Accuracy: 30.5%
  Batch Trend: 17.1% → 22.1% (↑ 5.0pp)
    ✓ Accuracy improving within epoch - learning is active!
  Accuracy Distribution: 0-25%:75%, 25-50%:25%, 50-75%:0%, 75-90%:0%, 90-100%:0%
    [!] Many samples stuck at low accuracy - check data/model!
  Running Window (last 5 batches): 19.7% ± 3.6%

  --- PER-COLOR ACCURACY (10 classes) ---
  Color:       0     1     2     3     4     5     6     7     8     9
  Acc%:      37%    1%    2%   16%    8%   11%    4%   12%    5%    8%
  Target:  58.4%  6.3%  6.3%  5.5%  4.4%  5.0%  2.6%  2.4%  6.8%  2.4%
  Pred:    33.8%  1.1%  2.4% 13.9%  5.7% 20.1%  2.0% 14.4%  1.9%  4.8%
  [!] Weak colors (<50% acc): 0(Black), 1(Blue), 2(Red), 3(Green), 4(Yellow), 5(Gray), 6(Pink), 7(Orange), 8(Cyan), 9(Brown)
  [!] Under-predicting color 1 (Blue): 1.1% vs target 6.3%
  [!] Under-predicting color 2 (Red): 2.4% vs target 6.3%
  [!] Under-predicting color 8 (Cyan): 1.9% vs target 6.8%
  ==================================================

  ==================================================
  LEARNING TRAJECTORY (Epoch 5)
  ==================================================
  Stop Prob:   0.271 → (init=0.27, task-dependent)
  Exp. Clues:  5.10 (latent variable, task-dependent)
  Attn Entropy: 3.76 ↑ (max=6.8, sharper=better)
  Task Loss:   0.2865 ↑
  Train Acc:   19.7% →
  Exact Match: 0.0% ↓
  Best Step:   0 (later=better refinement)
  FG Coverage: 169.3% of target ↑
  ==================================================

  ############################################################
  🚦 TRAINING HEALTH CHECK - Epoch 5
  ############################################################
  ✓ Attention sharpening (0.55 < 0.7)
  ✓ No NaN/Inf issues
  ⚠ Stop probs nearly uniform (global=0.091, per_clue=0.013)
  ⚠ Negative coupling (r=-0.74) - early epoch OK
  ⚠ Loss slowly decreasing (0.311 → 0.287)
  ⚠ Accuracy flat (19.9% → 19.7%) - early epoch
  ⚠ Color preference (31% one color)
  ✗ Centroid COLLAPSE (0.1 < 0.5)
  --------------------------------------------------------
  RESULT: 2/8 checks passed
  STATUS: 🟠 WARNING
  → Multiple issues detected. Consider intervention soon.

  RECOMMENDED ACTIONS:
  ############################################################
  HPM Instance Buffer: 0 entries (empty, not saved)
  HPM Procedural Buffer: 0 entries (empty, not saved)
  Saved checkpoint to checkpoints\rlan_stable_merged\latest.pt
  [ARCDataset] Augmentation config updated:
    dihedral=False, color_perm=False (prob=0.50), translational=False

Epoch 6/200
----------------------------------------
  [Batch 0] Grid size: 30x30 (batch_max_size=30)

  [MEMORY] Epoch 5 Batch 0:
    Baseline: alloc=92MB, reserved=19300MB
    01_batch_on_gpu: alloc=103MB (+10MB), reserved=19322MB
    02_after_forward: alloc=15914MB (+15811MB), reserved=19322MB
    03_before_backward: alloc=15958MB (+44MB), reserved=19322MB
    04_after_backward: alloc=230MB (-15728MB), reserved=19322MB
    PEAK: alloc=16456MB, reserved=19322MB / 24576MB (78.6%)

    [BREAKDOWN] Largest increase at '02_after_forward' (+15811MB):
      Model params: 73.4MB
      Model grads:  0.0MB
      Optimizer:    146.7MB (estimate (Adam-like: AdamW))
      Batch data:   6.2MB
      Activations:  77.8MB
      Output tensors (top 5):
        features: 43.9MB
        support_features: 25.0MB
        all_logits: 6.0MB
        attention_maps: 1.2MB
        logits: 0.9MB
      Module params (top 5):
        solver: 28.4MB
        hyper_lora: 19.1MB
        context_encoder: 15.0MB
        context_injector: 3.0MB
        hpm: 2.9MB
      Inactive modules: hyperlora, solver_context, cross_attention, hpm, loo, equivariance
  Batch 0/5: loss=0.4578, focal_weighted=0.3094, batch_acc=22.7%, exact=0/50 (0.0%), running_acc=22.7%, lr=5.00e-04
    [TaskTrack] Global_Solved: 1/342, Epoch_Solved: 0, New_Puzzles: 0
    FG: batch=6.5% run50=6.5% | BG: batch=35.7% run50=35.7%
    Per-Color: [0:42% 1:0% 2:2% 3:13% 4:5% 5:16% 6:4% 7:18% 8:5% 9:3%]
    Running50: [0:42% 1:0% 2:2% 3:13% 4:5% 5:16% 6:4% 7:18% 8:5% 9:3%]
    Solver: [2.641, 2.641, 2.641, 2.641, 2.641, 2.641, 2.641] ⚠ best=0
  [Batch 1] Grid size: 30x30 (batch_max_size=30)

  [MEMORY] Epoch 5 Batch 1:
    Baseline: alloc=92MB, reserved=19300MB
    01_batch_on_gpu: alloc=211MB (+119MB), reserved=19324MB
    02_after_forward: alloc=13343MB (+13131MB), reserved=19324MB
    03_before_backward: alloc=13425MB (+82MB), reserved=19324MB
    04_after_backward: alloc=274MB (-13150MB), reserved=19324MB
    PEAK: alloc=13885MB, reserved=19324MB / 24576MB (78.6%)

    [BREAKDOWN] Largest increase at '02_after_forward' (+13131MB):
      Model params: 73.4MB
      Model grads:  45.9MB
      Optimizer:    146.7MB (estimate (Adam-like: AdamW))
      Batch data:   4.1MB
      Activations:  68.3MB
      Output tensors (top 5):
        features: 43.9MB
        support_features: 15.6MB
        all_logits: 6.0MB
        attention_maps: 1.2MB
        logits: 0.9MB
      Module params (top 5):
        solver: 28.4MB
        hyper_lora: 19.1MB
        context_encoder: 15.0MB
        context_injector: 3.0MB
        hpm: 2.9MB
      Inactive modules: hyperlora, solver_context, cross_attention, hpm, loo, equivariance
  Batch 1/5: loss=0.7231, focal_weighted=0.5592, batch_acc=21.2%, exact=0/100 (0.0%), running_acc=22.0%, lr=5.00e-04
    [TaskTrack] Global_Solved: 1/342, Epoch_Solved: 0, New_Puzzles: 0
    FG: batch=5.3% run50=5.9% | BG: batch=39.3% run50=37.5%
    Per-Color: [0:40% 1:1% 2:3% 3:8% 4:7% 5:5% 6:3% 7:8% 8:5% 9:9%]
    Running50: [0:41% 1:0% 2:3% 3:10% 4:6% 5:10% 6:3% 7:13% 8:5% 9:6%]
  [Batch 2] Grid size: 30x30 (batch_max_size=30)

  [MEMORY] Epoch 5 Batch 2:
    Baseline: alloc=92MB, reserved=19300MB
    01_batch_on_gpu: alloc=215MB (+122MB), reserved=19324MB
    02_after_forward: alloc=15131MB (+14916MB), reserved=19324MB
    03_before_backward: alloc=15157MB (+27MB), reserved=19324MB
    04_after_backward: alloc=287MB (-14870MB), reserved=19324MB
    PEAK: alloc=15673MB, reserved=19324MB / 24576MB (78.6%)

    [BREAKDOWN] Largest increase at '02_after_forward' (+14916MB):
      Model params: 73.4MB
      Model grads:  45.9MB
      Optimizer:    146.7MB (estimate (Adam-like: AdamW))
      Batch data:   5.5MB
      Activations:  74.6MB
      Output tensors (top 5):
        features: 43.9MB
        support_features: 21.9MB
        all_logits: 6.0MB
        attention_maps: 1.2MB
        logits: 0.9MB
      Module params (top 5):
        solver: 28.4MB
        hyper_lora: 19.1MB
        context_encoder: 15.0MB
        context_injector: 3.0MB
        hpm: 2.9MB
      Inactive modules: hyperlora, solver_context, cross_attention, hpm, loo, equivariance
  Batch 2/5: loss=0.2987, focal_weighted=0.1640, batch_acc=18.5%, exact=0/150 (0.0%), running_acc=20.8%, lr=5.00e-04
    [TaskTrack] Global_Solved: 1/342, Epoch_Solved: 0, New_Puzzles: 0
    FG: batch=7.2% run50=6.4% | BG: batch=27.6% run50=34.2%
    Per-Color: [0:33% 1:0% 2:3% 3:25% 4:6% 5:11% 6:2% 7:15% 8:3% 9:0%]
    Running50: [0:38% 1:0% 2:3% 3:15% 4:6% 5:10% 6:3% 7:14% 8:4% 9:4%]
  [Batch 3] Grid size: 30x30 (batch_max_size=30)

  [MEMORY] Epoch 5 Batch 3:
    Baseline: alloc=92MB, reserved=19300MB
    01_batch_on_gpu: alloc=218MB (+125MB), reserved=19324MB
    02_after_forward: alloc=17807MB (+17590MB), reserved=19324MB
    03_before_backward: alloc=17837MB (+30MB), reserved=19324MB
    04_after_backward: alloc=299MB (-17538MB), reserved=19324MB
    PEAK: alloc=18349MB, reserved=19324MB / 24576MB (78.6%)

    [BREAKDOWN] Largest increase at '02_after_forward' (+17590MB):
      Model params: 73.4MB
      Model grads:  45.9MB
      Optimizer:    146.7MB (estimate (Adam-like: AdamW))
      Batch data:   7.6MB
      Activations:  84.1MB
      Output tensors (top 5):
        features: 43.9MB
        support_features: 31.2MB
        all_logits: 6.0MB
        attention_maps: 1.2MB
        logits: 0.9MB
      Module params (top 5):
        solver: 28.4MB
        hyper_lora: 19.1MB
        context_encoder: 15.0MB
        context_injector: 3.0MB
        hpm: 2.9MB
      Inactive modules: hyperlora, solver_context, cross_attention, hpm, loo, equivariance
  Batch 3/5: loss=0.3110, focal_weighted=0.1720, batch_acc=20.5%, exact=0/200 (0.0%), running_acc=20.7%, lr=5.00e-04
    [TaskTrack] Global_Solved: 1/342, Epoch_Solved: 0, New_Puzzles: 0
    FG: batch=8.0% run50=6.8% | BG: batch=30.0% run50=33.2%
    Per-Color: [0:34% 1:4% 2:3% 3:18% 4:8% 5:13% 6:0% 7:13% 8:1% 9:12%]
    Running50: [0:37% 1:1% 2:3% 3:16% 4:6% 5:11% 6:2% 7:14% 8:4% 9:6%]
  [Batch 4] Grid size: 30x30 (batch_max_size=30)

  [MEMORY] Epoch 5 Batch 4:
    Baseline: alloc=92MB, reserved=19300MB
    01_batch_on_gpu: alloc=212MB (+120MB), reserved=19324MB
    02_after_forward: alloc=16016MB (+15803MB), reserved=19324MB
    03_before_backward: alloc=16039MB (+24MB), reserved=19324MB
    04_after_backward: alloc=284MB (-15755MB), reserved=19324MB
    PEAK: alloc=16558MB, reserved=19324MB / 24576MB (78.6%)

    [BREAKDOWN] Largest increase at '02_after_forward' (+15803MB):
      Model params: 73.4MB
      Model grads:  45.9MB
      Optimizer:    146.7MB (estimate (Adam-like: AdamW))
      Batch data:   6.2MB
      Activations:  77.8MB
      Output tensors (top 5):
        features: 43.9MB
        support_features: 25.0MB
        all_logits: 6.0MB
        attention_maps: 1.2MB
        logits: 0.9MB
      Module params (top 5):
        solver: 28.4MB
        hyper_lora: 19.1MB
        context_encoder: 15.0MB
        context_injector: 3.0MB
        hpm: 2.9MB
      Inactive modules: hyperlora, solver_context, cross_attention, hpm, loo, equivariance
  Batch 4/5: loss=0.2433, focal_weighted=0.1133, batch_acc=15.3%, exact=1/250 (0.4%), running_acc=19.7%, lr=5.00e-04
    [TaskTrack] Global_Solved: 1/342, Epoch_Solved: 1, New_Puzzles: 0
    FG: batch=7.5% run50=6.9% | BG: batch=20.9% run50=30.7%
    Per-Color: [0:27% 1:2% 2:4% 3:9% 4:10% 5:6% 6:0% 7:10% 8:2% 9:9%]
    Running50: [0:35% 1:1% 2:3% 3:14% 4:7% 5:10% 6:2% 7:13% 8:3% 9:6%]

Epoch 6 Summary:
  Total Loss: 0.4068
  Task Loss (focal): 0.2636
  [Global Task Progress] Unique Solved: 1/342 (0.3%)
    NEW puzzles solved this epoch: 0
    ⚠️ No new puzzles solved in last 5 epochs - model may be plateauing
  Entropy Loss: 4.2925 (weight=0.01)
  Sparsity Loss: 0.1732 (weight=0.5)
  Predicate Loss: 0.0000 (weight=0.01)
  Time: 4.3s, LR: 5.00e-04
    Per-module LRs: DSC:5.00e-04, MSRE:5.00e-04, Other:5.00e-04
  Temperature: 0.9828 (lower=sharper attention)
  Samples Processed: 250 (5 batches)
  Dihedral Distribution: [100.0%, 0.0%, 0.0%, 0.0%, 0.0%, 0.0%, 0.0%, 0.0%]
    [!] Non-uniform dihedral distribution (max dev: 87.5%)
  Color Permutation: 0.0% (0/250)
  Translational Aug: 0.0% (0/250), unique offsets: 0
  Aug Quality: OK
  --- Training Diagnostics ---
  Solver Steps: 7 (deep supervision active)
  Per-Step Loss (epoch avg, 1 batches): [2.6406, 2.6406, 2.6406, 2.6406, 2.6406, 2.6406, 2.6406]
    [!] SOLVER DEGRADATION: Step 0 is best! Later steps 0.0% worse!
    [!] Best: step 0 (2.6406), Worst: step 0 (2.6406)
  Best-Step Histogram: [s0:1]
  Solver Health: Last step best: 0.0%, Earlier step best: 100.0%
  Avg Step Improvement: 0.0% (step0→stepN)
    ⚠️ SOLVER OVER-ITERATION WARNING: 100% of batches had earlier step as best!
    ⚠️ Consider: (1) enabling best-step selection, (2) reducing num_solver_steps, (3) enabling ACT
  Grad Norms: DSC=0.0000, StopPred=0.0000, Encoder=0.0000, Solver=0.0000
  StopPred Weight Var: 2.80e-02
  Attention: max=0.0203, min=0.000000
  Stop Prob: 0.244 (approx 5.3 clues active)
  Stop Probs Std: 0.081 (global std across batch×clues)
  Clues Used: mean=5.29, std=0.36, range=[4.1, 6.0]
  Clue-Loss Correlation: +0.073 (weak - per-sample coupling may need tuning)
  Stop Logits: mean=-1.18, std=0.44, range=[-2.7, 0.3]
    Clue count varies by task (per-sample coupling active)
  Per-Clue Entropy: [4.60, 4.60, 4.59, 4.59, 4.59, 4.59, 4.60] (mean=4.59, max=6.80)
    [!] Clues have uniform entropy (std=0.003) - not differentiating!
  Centroid Spread: 0.10 (higher=more diverse)
    🚨 CRITICAL COLLAPSE: Spread=0.10 < 0.5 - all clues at same location!
    [!] Stop predictor cannot differentiate - needs diversity regularizer
  --- Stop Predictor Coupling ---
  Entropy Input to Stop: 0.6755 (normalized, lower=sharper)
  Per-Clue Entropy Input: [0.676, 0.676, 0.675, 0.675, 0.675, 0.675, 0.676]
  Per-Clue Stop Prob: [0.254, 0.241, 0.243, 0.229, 0.238, 0.244, 0.256]
  --- Sparsity Loss Breakdown (Per-Sample Coupled) ---
  Min Clue Penalty: 0.0000 (per-sample avg)
  Per-Sample Clue Penalty (scaled): 0.0000
  Base Pondering: 0.1058 (clues=5.29)
  Entropy Pondering: 0.0715
  --- Per-Class Distribution (Valid Pixels Only) ---
  Padding: 88.3% of grid (ignored in loss)
  Pred %: [36.6, 0.9, 2.1, 12.2, 5.5, 18.8, 1.1, 17.4, 1.4, 3.9]
  Target %: [61.9, 5.2, 5.3, 4.2, 3.3, 4.8, 2.9, 1.3, 9.5, 1.5]
  Per-Class Acc %: [38, 1, 3, 11, 7, 11, 3, 14, 5, 7]
  ==================================================
  PER-SAMPLE TRAINING ACCURACY (Epoch 6)
  ==================================================
  ★ Mean Accuracy: 19.7%
  ★ Exact Match: 1/250 (0.4%)
  ★ High Acc (≥90%): 1/250 (0.4%)
  FG Accuracy: 6.9%
  BG Accuracy: 30.7%
  Batch Trend: 22.7% → 17.9% (↓ 4.8pp)
  Accuracy Distribution: 0-25%:75%, 25-50%:25%, 50-75%:0%, 75-90%:0%, 90-100%:0%
    [!] Many samples stuck at low accuracy - check data/model!
  Running Window (last 5 batches): 19.7% ± 2.6%

  --- PER-COLOR ACCURACY (10 classes) ---
  Color:       0     1     2     3     4     5     6     7     8     9
  Acc%:      38%    1%    3%   11%    7%   11%    3%   14%    5%    7%
  Target:  59.3%  6.8%  5.5%  6.7%  4.2%  5.0%  2.2%  1.4%  6.6%  2.4%
  Pred:    34.5%  1.1%  2.4% 13.2%  5.5% 20.3%  1.9% 15.3%  1.7%  4.1%
  [!] Weak colors (<50% acc): 0(Black), 1(Blue), 2(Red), 3(Green), 4(Yellow), 5(Gray), 6(Pink), 7(Orange), 8(Cyan), 9(Brown)
  [!] Under-predicting color 1 (Blue): 1.1% vs target 6.8%
  [!] Under-predicting color 2 (Red): 2.4% vs target 5.5%
  [!] Under-predicting color 8 (Cyan): 1.7% vs target 6.6%
  ==================================================

  ==================================================
  LEARNING TRAJECTORY (Epoch 6)
  ==================================================
  Stop Prob:   0.244 ↓ (init=0.27, task-dependent)
  Exp. Clues:  5.29 (latent variable, task-dependent)
  Attn Entropy: 4.59 ↑ (max=6.8, sharper=better)
  Task Loss:   0.2636 ↓
  Train Acc:   19.7% →
  Exact Match: 0.4% ↑
  Best Step:   0 (later=better refinement)
  FG Coverage: 166.6% of target ↓
  [!] Potential issues: stop_prob not increasing, attention not sharpening, train_accuracy not improving
  ==================================================

  ############################################################
  🚦 TRAINING HEALTH CHECK - Epoch 6
  ############################################################
  ✓ Attention sharpening (0.68 < 0.7)
  ✓ No NaN/Inf issues
  ✓ No color mode collapse
  ⚠ Stop probs nearly uniform (global=0.081, per_clue=0.008)
  ⚠ Negative coupling (r=-0.75) - early epoch OK
  ⚠ Loss slowly decreasing (0.311 → 0.264)
  ⚠ Accuracy flat (19.9% → 19.7%) - early epoch
  ✗ Centroid COLLAPSE (0.1 < 0.5)
  --------------------------------------------------------
  RESULT: 3/8 checks passed
  STATUS: 🟠 WARNING
  → Multiple issues detected. Consider intervention soon.

  RECOMMENDED ACTIONS:
  ############################################################
  HPM Instance Buffer: 0 entries (empty, not saved)
  HPM Procedural Buffer: 0 entries (empty, not saved)
  Saved checkpoint to checkpoints\rlan_stable_merged\latest.pt
  [ARCDataset] Augmentation config updated:
    dihedral=False, color_perm=False (prob=0.50), translational=False

Epoch 7/200
----------------------------------------
  [Batch 0] Grid size: 30x30 (batch_max_size=30)

  [MEMORY] Epoch 6 Batch 0:
    Baseline: alloc=92MB, reserved=19324MB
    01_batch_on_gpu: alloc=103MB (+10MB), reserved=19346MB
    02_after_forward: alloc=15023MB (+14920MB), reserved=19346MB
    03_before_backward: alloc=15068MB (+46MB), reserved=19346MB
    04_after_backward: alloc=227MB (-14842MB), reserved=19346MB
    PEAK: alloc=15565MB, reserved=19346MB / 24576MB (78.7%)

    [BREAKDOWN] Largest increase at '02_after_forward' (+14920MB):
      Model params: 73.4MB
      Model grads:  0.0MB
      Optimizer:    146.7MB (estimate (Adam-like: AdamW))
      Batch data:   5.5MB
      Activations:  74.6MB
      Output tensors (top 5):
        features: 43.9MB
        support_features: 21.9MB
        all_logits: 6.0MB
        attention_maps: 1.2MB
        logits: 0.9MB
      Module params (top 5):
        solver: 28.4MB
        hyper_lora: 19.1MB
        context_encoder: 15.0MB
        context_injector: 3.0MB
        hpm: 2.9MB
      Inactive modules: hyperlora, solver_context, cross_attention, hpm, loo, equivariance
  Batch 0/5: loss=0.4709, focal_weighted=0.3225, batch_acc=19.6%, exact=0/50 (0.0%), running_acc=19.6%, lr=5.00e-04
    [TaskTrack] Global_Solved: 1/342, Epoch_Solved: 0, New_Puzzles: 0
    FG: batch=4.1% run50=4.1% | BG: batch=32.2% run50=32.2%
    Per-Color: [0:39% 1:1% 2:2% 3:11% 4:4% 5:6% 6:5% 7:4% 8:8% 9:4%]
    Running50: [0:39% 1:1% 2:2% 3:11% 4:4% 5:6% 6:5% 7:4% 8:8% 9:4%]
    Solver: [2.672, 2.688, 2.688, 2.688, 2.688, 2.688, 2.688] ⚠ best=0
  [Batch 1] Grid size: 30x30 (batch_max_size=30)

  [MEMORY] Epoch 6 Batch 1:
    Baseline: alloc=92MB, reserved=19324MB
    01_batch_on_gpu: alloc=213MB (+121MB), reserved=19348MB
    02_after_forward: alloc=14237MB (+14023MB), reserved=19348MB
    03_before_backward: alloc=14264MB (+27MB), reserved=19348MB
    04_after_backward: alloc=281MB (-13983MB), reserved=19348MB
    PEAK: alloc=14779MB, reserved=19348MB / 24576MB (78.7%)

    [BREAKDOWN] Largest increase at '02_after_forward' (+14023MB):
      Model params: 73.4MB
      Model grads:  45.9MB
      Optimizer:    146.7MB (estimate (Adam-like: AdamW))
      Batch data:   4.8MB
      Activations:  71.5MB
      Output tensors (top 5):
        features: 43.9MB
        support_features: 18.8MB
        all_logits: 6.0MB
        attention_maps: 1.2MB
        logits: 0.9MB
      Module params (top 5):
        solver: 28.4MB
        hyper_lora: 19.1MB
        context_encoder: 15.0MB
        context_injector: 3.0MB
        hpm: 2.9MB
      Inactive modules: hyperlora, solver_context, cross_attention, hpm, loo, equivariance
  Batch 1/5: loss=0.2734, focal_weighted=0.1432, batch_acc=18.8%, exact=0/100 (0.0%), running_acc=19.2%, lr=5.00e-04
    [TaskTrack] Global_Solved: 1/342, Epoch_Solved: 0, New_Puzzles: 0
    FG: batch=8.1% run50=6.1% | BG: batch=26.8% run50=29.5%
    Per-Color: [0:33% 1:1% 2:2% 3:22% 4:5% 5:6% 6:2% 7:15% 8:1% 9:5%]
    Running50: [0:36% 1:1% 2:2% 3:16% 4:5% 5:6% 6:4% 7:9% 8:5% 9:4%]
  [Batch 2] Grid size: 30x30 (batch_max_size=30)

  [MEMORY] Epoch 6 Batch 2:
    Baseline: alloc=92MB, reserved=19324MB
    01_batch_on_gpu: alloc=216MB (+124MB), reserved=19348MB
    02_after_forward: alloc=17807MB (+17591MB), reserved=19348MB
    03_before_backward: alloc=17835MB (+28MB), reserved=19348MB
    04_after_backward: alloc=297MB (-17538MB), reserved=19348MB
    PEAK: alloc=18348MB, reserved=19348MB / 24576MB (78.7%)

    [BREAKDOWN] Largest increase at '02_after_forward' (+17591MB):
      Model params: 73.4MB
      Model grads:  45.9MB
      Optimizer:    146.7MB (estimate (Adam-like: AdamW))
      Batch data:   7.6MB
      Activations:  84.1MB
      Output tensors (top 5):
        features: 43.9MB
        support_features: 31.2MB
        all_logits: 6.0MB
        attention_maps: 1.2MB
        logits: 0.9MB
      Module params (top 5):
        solver: 28.4MB
        hyper_lora: 19.1MB
        context_encoder: 15.0MB
        context_injector: 3.0MB
        hpm: 2.9MB
      Inactive modules: hyperlora, solver_context, cross_attention, hpm, loo, equivariance
  Batch 2/5: loss=0.2745, focal_weighted=0.1363, batch_acc=19.2%, exact=1/150 (0.7%), running_acc=19.2%, lr=5.00e-04
    [TaskTrack] Global_Solved: 1/342, Epoch_Solved: 1, New_Puzzles: 0
    FG: batch=8.0% run50=6.7% | BG: batch=28.0% run50=29.0%
    Per-Color: [0:32% 1:2% 2:1% 3:16% 4:10% 5:11% 6:7% 7:7% 8:3% 9:0%]
    Running50: [0:35% 1:1% 2:2% 3:16% 4:6% 5:8% 6:5% 7:9% 8:4% 9:3%]
  [Batch 3] Grid size: 30x30 (batch_max_size=30)

  [MEMORY] Epoch 6 Batch 3:
    Baseline: alloc=92MB, reserved=19324MB
    01_batch_on_gpu: alloc=215MB (+123MB), reserved=19348MB
    02_after_forward: alloc=16020MB (+15804MB), reserved=19348MB
    03_before_backward: alloc=16044MB (+24MB), reserved=19348MB
    04_after_backward: alloc=287MB (-15757MB), reserved=19348MB
    PEAK: alloc=16561MB, reserved=19348MB / 24576MB (78.7%)

    [BREAKDOWN] Largest increase at '02_after_forward' (+15804MB):
      Model params: 73.4MB
      Model grads:  45.9MB
      Optimizer:    146.7MB (estimate (Adam-like: AdamW))
      Batch data:   6.2MB
      Activations:  77.8MB
      Output tensors (top 5):
        features: 43.9MB
        support_features: 25.0MB
        all_logits: 6.0MB
        attention_maps: 1.2MB
        logits: 0.9MB
      Module params (top 5):
        solver: 28.4MB
        hyper_lora: 19.1MB
        context_encoder: 15.0MB
        context_injector: 3.0MB
        hpm: 2.9MB
      Inactive modules: hyperlora, solver_context, cross_attention, hpm, loo, equivariance
  Batch 3/5: loss=0.2974, focal_weighted=0.1653, batch_acc=14.8%, exact=1/200 (0.5%), running_acc=18.1%, lr=5.00e-04
    [TaskTrack] Global_Solved: 1/342, Epoch_Solved: 1, New_Puzzles: 0
    FG: batch=4.7% run50=6.2% | BG: batch=25.5% run50=28.1%
    Per-Color: [0:29% 1:0% 2:2% 3:16% 4:14% 5:11% 6:0% 7:18% 8:1% 9:4%]
    Running50: [0:33% 1:1% 2:2% 3:16% 4:8% 5:8% 6:4% 7:11% 8:3% 9:3%]
  [Batch 4] Grid size: 30x30 (batch_max_size=30)

  [MEMORY] Epoch 6 Batch 4:
    Baseline: alloc=92MB, reserved=19324MB
    01_batch_on_gpu: alloc=208MB (+115MB), reserved=19348MB
    02_after_forward: alloc=13338MB (+13130MB), reserved=19348MB
    03_before_backward: alloc=13416MB (+78MB), reserved=19348MB
    04_after_backward: alloc=271MB (-13145MB), reserved=19348MB
    PEAK: alloc=13880MB, reserved=19348MB / 24576MB (78.7%)

    [BREAKDOWN] Largest increase at '02_after_forward' (+13130MB):
      Model params: 73.4MB
      Model grads:  45.9MB
      Optimizer:    146.7MB (estimate (Adam-like: AdamW))
      Batch data:   4.1MB
      Activations:  68.3MB
      Output tensors (top 5):
        features: 43.9MB
        support_features: 15.6MB
        all_logits: 6.0MB
        attention_maps: 1.2MB
        logits: 0.9MB
      Module params (top 5):
        solver: 28.4MB
        hyper_lora: 19.1MB
        context_encoder: 15.0MB
        context_injector: 3.0MB
        hpm: 2.9MB
      Inactive modules: hyperlora, solver_context, cross_attention, hpm, loo, equivariance
  Batch 4/5: loss=0.8308, focal_weighted=0.6716, batch_acc=20.4%, exact=1/250 (0.4%), running_acc=18.6%, lr=5.00e-04
    [TaskTrack] Global_Solved: 1/342, Epoch_Solved: 1, New_Puzzles: 0
    FG: batch=6.2% run50=6.2% | BG: batch=36.2% run50=29.7%
    Per-Color: [0:39% 1:1% 2:4% 3:11% 4:9% 5:9% 6:6% 7:14% 8:6% 9:8%]
    Running50: [0:34% 1:1% 2:2% 3:15% 4:8% 5:8% 6:4% 7:12% 8:4% 9:4%]

Epoch 7 Summary:
  Total Loss: 0.4294
  Task Loss (focal): 0.2878
  [Global Task Progress] Unique Solved: 1/342 (0.3%)
    NEW puzzles solved this epoch: 0
    ⚠️ No new puzzles solved in last 5 epochs - model may be plateauing
  Entropy Loss: 4.2749 (weight=0.01)
  Sparsity Loss: 0.1704 (weight=0.5)
  Predicate Loss: 0.0000 (weight=0.01)
  Time: 4.3s, LR: 5.00e-04
    Per-module LRs: DSC:5.00e-04, MSRE:5.00e-04, Other:5.00e-04
  Temperature: 0.9794 (lower=sharper attention)
  Samples Processed: 250 (5 batches)
  Dihedral Distribution: [100.0%, 0.0%, 0.0%, 0.0%, 0.0%, 0.0%, 0.0%, 0.0%]
    [!] Non-uniform dihedral distribution (max dev: 87.5%)
  Color Permutation: 0.0% (0/250)
  Translational Aug: 0.0% (0/250), unique offsets: 0
  Aug Quality: OK
  --- Training Diagnostics ---
  Solver Steps: 7 (deep supervision active)
  Per-Step Loss (epoch avg, 1 batches): [2.6719, 2.6875, 2.6875, 2.6875, 2.6875, 2.6875, 2.6875]
    [!] SOLVER DEGRADATION: Step 0 is best! Later steps 0.6% worse!
    [!] Best: step 0 (2.6719), Worst: step 1 (2.6875)
  Best-Step Histogram: [s0:1]
  Solver Health: Last step best: 0.0%, Earlier step best: 100.0%
  Avg Step Improvement: -0.6% (step0→stepN)
    ⚠️ SOLVER OVER-ITERATION WARNING: 100% of batches had earlier step as best!
    ⚠️ Consider: (1) enabling best-step selection, (2) reducing num_solver_steps, (3) enabling ACT
    ⚠️ SOLVER DEGRADATION: Avg improvement is negative! More steps = worse predictions!
  Grad Norms: DSC=0.0000, StopPred=0.0000, Encoder=0.0000, Solver=0.0000
  StopPred Weight Var: 2.80e-02
  Attention: max=0.0187, min=0.000000
  Stop Prob: 0.256 (approx 5.2 clues active)
  Stop Probs Std: 0.085 (global std across batch×clues)
  Clues Used: mean=5.21, std=0.37, range=[3.8, 6.0]
  Clue-Loss Correlation: -0.180 (unexpected negative - check gradient flow)
  Stop Logits: mean=-1.12, std=0.45, range=[-2.5, 0.5]
    Clue count varies by task (per-sample coupling active)
  Per-Clue Entropy: [4.67, 4.67, 4.67, 4.67, 4.66, 4.67, 4.67] (mean=4.67, max=6.80)
    [!] Clues have uniform entropy (std=0.003) - not differentiating!
  Centroid Spread: 0.10 (higher=more diverse)
    🚨 CRITICAL COLLAPSE: Spread=0.10 < 0.5 - all clues at same location!
    [!] Stop predictor cannot differentiate - needs diversity regularizer
  --- Stop Predictor Coupling ---
  Entropy Input to Stop: 0.6862 (normalized, lower=sharper)
  Per-Clue Entropy Input: [0.687, 0.686, 0.686, 0.686, 0.686, 0.686, 0.687]
  Per-Clue Stop Prob: [0.260, 0.254, 0.238, 0.250, 0.256, 0.271, 0.258]
  --- Sparsity Loss Breakdown (Per-Sample Coupled) ---
  Min Clue Penalty: 0.0000 (per-sample avg)
  Per-Sample Clue Penalty (scaled): 0.0000
  Base Pondering: 0.1042 (clues=5.21)
  Entropy Pondering: 0.0717
  --- Per-Class Distribution (Valid Pixels Only) ---
  Padding: 88.0% of grid (ignored in loss)
  Pred %: [34.1, 0.9, 2.4, 13.0, 5.7, 19.6, 1.8, 15.7, 2.0, 4.8]
  Target %: [57.9, 5.4, 4.3, 5.6, 3.8, 4.3, 6.1, 1.4, 9.9, 1.4]
  Per-Class Acc %: [36, 1, 3, 12, 7, 8, 5, 11, 6, 7]
  ==================================================
  PER-SAMPLE TRAINING ACCURACY (Epoch 7)
  ==================================================
  ★ Mean Accuracy: 18.6%
  ★ Exact Match: 1/250 (0.4%)
  ★ High Acc (≥90%): 1/250 (0.4%)
  FG Accuracy: 6.2%
  BG Accuracy: 29.7%
  Batch Trend: 19.6% → 17.6% (↓ 2.0pp)
  Accuracy Distribution: 0-25%:75%, 25-50%:25%, 50-75%:0%, 75-90%:0%, 90-100%:0%
    [!] Many samples stuck at low accuracy - check data/model!
  Running Window (last 5 batches): 18.6% ± 2.0%

  --- PER-COLOR ACCURACY (10 classes) ---
  Color:       0     1     2     3     4     5     6     7     8     9
  Acc%:      36%    1%    3%   12%    7%    8%    5%   11%    6%    7%
  Target:  55.5%  7.1%  5.9%  7.1%  4.3%  5.3%  3.8%  1.8%  7.1%  2.1%
  Pred:    32.4%  1.2%  2.3% 14.2%  6.0% 20.4%  2.1% 14.6%  1.9%  4.7%
  [!] Weak colors (<50% acc): 0(Black), 1(Blue), 2(Red), 3(Green), 4(Yellow), 5(Gray), 6(Pink), 7(Orange), 8(Cyan), 9(Brown)
  [!] Under-predicting color 1 (Blue): 1.2% vs target 7.1%
  [!] Under-predicting color 2 (Red): 2.3% vs target 5.9%
  [!] Under-predicting color 8 (Cyan): 1.9% vs target 7.1%
  ==================================================

  ==================================================
  LEARNING TRAJECTORY (Epoch 7)
  ==================================================
  Stop Prob:   0.256 ↑ (init=0.27, task-dependent)
  Exp. Clues:  5.21 (latent variable, task-dependent)
  Attn Entropy: 4.67 ↑ (max=6.8, sharper=better)
  Task Loss:   0.2878 ↑
  Train Acc:   18.6% ↓
  Exact Match: 0.4% →
  Best Step:   0 (later=better refinement)
  FG Coverage: 156.5% of target ↓
  [!] Potential issues: stop_prob not increasing, attention not sharpening, task_loss not decreasing, train_accuracy not improving
  ==================================================

  ############################################################
  🚦 TRAINING HEALTH CHECK - Epoch 7
  ############################################################
  ✓ Attention sharpening (0.69 < 0.7)
  ✓ No NaN/Inf issues
  ✓ No color mode collapse
  ⚠ Stop probs nearly uniform (global=0.085, per_clue=0.009)
  ⚠ Negative coupling (r=-0.29) - early epoch OK
  ⚠ Loss slowly decreasing (0.311 → 0.288)
  ⚠ Accuracy flat (19.9% → 18.6%) - early epoch
  ✗ Centroid COLLAPSE (0.1 < 0.5)
  --------------------------------------------------------
  RESULT: 3/8 checks passed
  STATUS: 🟠 WARNING
  → Multiple issues detected. Consider intervention soon.

  RECOMMENDED ACTIONS:
  ############################################################
  HPM Instance Buffer: 0 entries (empty, not saved)
  HPM Procedural Buffer: 0 entries (empty, not saved)
  Saved checkpoint to checkpoints\rlan_stable_merged\latest.pt
  [ARCDataset] Augmentation config updated:
    dihedral=False, color_perm=False (prob=0.50), translational=False

Epoch 8/200
----------------------------------------
  [Batch 0] Grid size: 30x30 (batch_max_size=30)

  [MEMORY] Epoch 7 Batch 0:
    Baseline: alloc=92MB, reserved=19348MB
    01_batch_on_gpu: alloc=103MB (+10MB), reserved=19370MB
    02_after_forward: alloc=15914MB (+15811MB), reserved=19370MB
    03_before_backward: alloc=15961MB (+47MB), reserved=19370MB
    04_after_backward: alloc=230MB (-15731MB), reserved=19370MB
    PEAK: alloc=16456MB, reserved=19370MB / 24576MB (78.8%)

    [BREAKDOWN] Largest increase at '02_after_forward' (+15811MB):
      Model params: 73.4MB
      Model grads:  0.0MB
      Optimizer:    146.7MB (estimate (Adam-like: AdamW))
      Batch data:   6.2MB
      Activations:  77.8MB
      Output tensors (top 5):
        features: 43.9MB
        support_features: 25.0MB
        all_logits: 6.0MB
        attention_maps: 1.2MB
        logits: 0.9MB
      Module params (top 5):
        solver: 28.4MB
        hyper_lora: 19.1MB
        context_encoder: 15.0MB
        context_injector: 3.0MB
        hpm: 2.9MB
      Inactive modules: hyperlora, solver_context, cross_attention, hpm, loo, equivariance
  Batch 0/5: loss=0.3985, focal_weighted=0.2501, batch_acc=25.5%, exact=0/50 (0.0%), running_acc=25.5%, lr=5.00e-04
    [TaskTrack] Global_Solved: 1/342, Epoch_Solved: 0, New_Puzzles: 0
    FG: batch=5.5% run50=5.5% | BG: batch=36.7% run50=36.7%
    Per-Color: [0:41% 1:2% 2:1% 3:13% 4:2% 5:9% 6:2% 7:19% 8:5% 9:12%]
    Running50: [0:41% 1:2% 2:1% 3:13% 4:2% 5:9% 6:2% 7:19% 8:5% 9:12%]
    Solver: [1.992, 1.992, 1.992, 1.992, 1.992, 1.984, 1.984] ⚠ best=5
  [Batch 1] Grid size: 30x30 (batch_max_size=30)

  [MEMORY] Epoch 7 Batch 1:
    Baseline: alloc=92MB, reserved=19348MB
    01_batch_on_gpu: alloc=211MB (+118MB), reserved=19372MB
    02_after_forward: alloc=13342MB (+13131MB), reserved=19372MB
    03_before_backward: alloc=13423MB (+81MB), reserved=19372MB
    04_after_backward: alloc=274MB (-13149MB), reserved=19372MB
    PEAK: alloc=13885MB, reserved=19372MB / 24576MB (78.8%)

    [BREAKDOWN] Largest increase at '02_after_forward' (+13131MB):
      Model params: 73.4MB
      Model grads:  45.9MB
      Optimizer:    146.7MB (estimate (Adam-like: AdamW))
      Batch data:   4.1MB
      Activations:  68.3MB
      Output tensors (top 5):
        features: 43.9MB
        support_features: 15.6MB
        all_logits: 6.0MB
        attention_maps: 1.2MB
        logits: 0.9MB
      Module params (top 5):
        solver: 28.4MB
        hyper_lora: 19.1MB
        context_encoder: 15.0MB
        context_injector: 3.0MB
        hpm: 2.9MB
      Inactive modules: hyperlora, solver_context, cross_attention, hpm, loo, equivariance
  Batch 1/5: loss=0.8832, focal_weighted=0.7218, batch_acc=21.4%, exact=0/100 (0.0%), running_acc=23.4%, lr=5.00e-04
    [TaskTrack] Global_Solved: 1/342, Epoch_Solved: 0, New_Puzzles: 0
    FG: batch=6.2% run50=5.9% | BG: batch=39.2% run50=37.9%
    Per-Color: [0:41% 1:1% 2:2% 3:12% 4:5% 5:19% 6:4% 7:10% 8:6% 9:7%]
    Running50: [0:41% 1:1% 2:2% 3:13% 4:3% 5:14% 6:3% 7:15% 8:6% 9:9%]
  [Batch 2] Grid size: 30x30 (batch_max_size=30)

  [MEMORY] Epoch 7 Batch 2:
    Baseline: alloc=92MB, reserved=19348MB
    01_batch_on_gpu: alloc=211MB (+119MB), reserved=19372MB
    02_after_forward: alloc=14236MB (+14025MB), reserved=19372MB
    03_before_backward: alloc=14264MB (+28MB), reserved=19372MB
    04_after_backward: alloc=280MB (-13984MB), reserved=19372MB
    PEAK: alloc=14779MB, reserved=19372MB / 24576MB (78.8%)

    [BREAKDOWN] Largest increase at '02_after_forward' (+14025MB):
      Model params: 73.4MB
      Model grads:  45.9MB
      Optimizer:    146.7MB (estimate (Adam-like: AdamW))
      Batch data:   4.8MB
      Activations:  71.5MB
      Output tensors (top 5):
        features: 43.9MB
        support_features: 18.8MB
        all_logits: 6.0MB
        attention_maps: 1.2MB
        logits: 0.9MB
      Module params (top 5):
        solver: 28.4MB
        hyper_lora: 19.1MB
        context_encoder: 15.0MB
        context_injector: 3.0MB
        hpm: 2.9MB
      Inactive modules: hyperlora, solver_context, cross_attention, hpm, loo, equivariance
  Batch 2/5: loss=0.2702, focal_weighted=0.1425, batch_acc=17.1%, exact=0/150 (0.0%), running_acc=21.3%, lr=5.00e-04
    [TaskTrack] Global_Solved: 1/342, Epoch_Solved: 0, New_Puzzles: 0
    FG: batch=9.4% run50=7.1% | BG: batch=23.5% run50=33.1%
    Per-Color: [0:28% 1:1% 2:4% 3:24% 4:8% 5:15% 6:3% 7:21% 8:2% 9:0%]
    Running50: [0:37% 1:1% 2:2% 3:16% 4:5% 5:14% 6:3% 7:17% 8:4% 9:6%]
  [Batch 3] Grid size: 30x30 (batch_max_size=30)

  [MEMORY] Epoch 7 Batch 3:
    Baseline: alloc=92MB, reserved=19348MB
    01_batch_on_gpu: alloc=215MB (+122MB), reserved=19372MB
    02_after_forward: alloc=14239MB (+14024MB), reserved=19372MB
    03_before_backward: alloc=14266MB (+27MB), reserved=19372MB
    04_after_backward: alloc=283MB (-13983MB), reserved=19372MB
    PEAK: alloc=14782MB, reserved=19372MB / 24576MB (78.8%)

    [BREAKDOWN] Largest increase at '02_after_forward' (+14024MB):
      Model params: 73.4MB
      Model grads:  45.9MB
      Optimizer:    146.7MB (estimate (Adam-like: AdamW))
      Batch data:   4.8MB
      Activations:  71.5MB
      Output tensors (top 5):
        features: 43.9MB
        support_features: 18.8MB
        all_logits: 6.0MB
        attention_maps: 1.2MB
        logits: 0.9MB
      Module params (top 5):
        solver: 28.4MB
        hyper_lora: 19.1MB
        context_encoder: 15.0MB
        context_injector: 3.0MB
        hpm: 2.9MB
      Inactive modules: hyperlora, solver_context, cross_attention, hpm, loo, equivariance
  Batch 3/5: loss=0.2868, focal_weighted=0.1546, batch_acc=16.2%, exact=0/200 (0.0%), running_acc=20.0%, lr=5.00e-04
    [TaskTrack] Global_Solved: 1/342, Epoch_Solved: 0, New_Puzzles: 0
    FG: batch=5.3% run50=6.6% | BG: batch=25.7% run50=31.3%
    Per-Color: [0:31% 1:2% 2:1% 3:19% 4:9% 5:3% 6:0% 7:15% 8:1% 9:12%]
    Running50: [0:35% 1:1% 2:2% 3:17% 4:6% 5:11% 6:2% 7:16% 8:4% 9:8%]
  [Batch 4] Grid size: 30x30 (batch_max_size=30)

  [MEMORY] Epoch 7 Batch 4:
    Baseline: alloc=92MB, reserved=19348MB
    01_batch_on_gpu: alloc=210MB (+118MB), reserved=19372MB
    02_after_forward: alloc=17803MB (+17593MB), reserved=19372MB
    03_before_backward: alloc=17827MB (+24MB), reserved=19372MB
    04_after_backward: alloc=291MB (-17536MB), reserved=19372MB
    PEAK: alloc=18345MB, reserved=19372MB / 24576MB (78.8%)

    [BREAKDOWN] Largest increase at '02_after_forward' (+17593MB):
      Model params: 73.4MB
      Model grads:  45.9MB
      Optimizer:    146.7MB (estimate (Adam-like: AdamW))
      Batch data:   7.6MB
      Activations:  84.1MB
      Output tensors (top 5):
        features: 43.9MB
        support_features: 31.2MB
        all_logits: 6.0MB
        attention_maps: 1.2MB
        logits: 0.9MB
      Module params (top 5):
        solver: 28.4MB
        hyper_lora: 19.1MB
        context_encoder: 15.0MB
        context_injector: 3.0MB
        hpm: 2.9MB
      Inactive modules: hyperlora, solver_context, cross_attention, hpm, loo, equivariance
  Batch 4/5: loss=0.2459, focal_weighted=0.1114, batch_acc=17.5%, exact=0/250 (0.0%), running_acc=19.5%, lr=5.00e-04
    [TaskTrack] Global_Solved: 1/342, Epoch_Solved: 0, New_Puzzles: 0
    FG: batch=6.9% run50=6.7% | BG: batch=28.8% run50=30.8%
    Per-Color: [0:34% 1:1% 2:2% 3:15% 4:10% 5:16% 6:0% 7:12% 8:5% 9:0%]
    Running50: [0:35% 1:1% 2:2% 3:17% 4:7% 5:12% 6:2% 7:15% 8:4% 9:6%]

Epoch 8 Summary:
  Total Loss: 0.4169
  Task Loss (focal): 0.2761
  [Global Task Progress] Unique Solved: 1/342 (0.3%)
    NEW puzzles solved this epoch: 0
    ⚠️ No new puzzles solved in last 5 epochs - model may be plateauing
  Entropy Loss: 4.2421 (weight=0.01)
  Sparsity Loss: 0.1695 (weight=0.5)
  Predicate Loss: 0.0000 (weight=0.01)
  Time: 4.2s, LR: 5.00e-04
    Per-module LRs: DSC:5.00e-04, MSRE:5.00e-04, Other:5.00e-04
  Temperature: 0.9760 (lower=sharper attention)
  Samples Processed: 250 (5 batches)
  Dihedral Distribution: [100.0%, 0.0%, 0.0%, 0.0%, 0.0%, 0.0%, 0.0%, 0.0%]
    [!] Non-uniform dihedral distribution (max dev: 87.5%)
  Color Permutation: 0.0% (0/250)
  Translational Aug: 0.0% (0/250), unique offsets: 0
  Aug Quality: OK
  --- Training Diagnostics ---
  Solver Steps: 7 (deep supervision active)
  Per-Step Loss (epoch avg, 1 batches): [1.9922, 1.9922, 1.9922, 1.9922, 1.9922, 1.9844, 1.9844]
    [!] Best step is 5 (middle), not last - solver may be over-iterating!
    [!] Best: step 5 (1.9844), Final: step 6 (1.9844)
    [!] Potential gain from best-step selection: 0.0% lower loss
  Best-Step Histogram: []
  Solver Health: Last step best: 0.0%, Earlier step best: 100.0%
  Avg Step Improvement: 0.4% (step0→stepN)
    ⚠️ SOLVER OVER-ITERATION WARNING: 100% of batches had earlier step as best!
    ⚠️ Consider: (1) enabling best-step selection, (2) reducing num_solver_steps, (3) enabling ACT
  Grad Norms: DSC=0.0000, StopPred=0.0000, Encoder=0.0000, Solver=0.0000
  StopPred Weight Var: 2.80e-02
  Attention: max=0.0192, min=0.000000
  Stop Prob: 0.250 (approx 5.2 clues active)
  Stop Probs Std: 0.080 (global std across batch×clues)
  Clues Used: mean=5.25, std=0.33, range=[4.3, 5.8]
  Clue-Loss Correlation: -0.043 (weak - per-sample coupling may need tuning)
  Stop Logits: mean=-1.14, std=0.43, range=[-2.2, 0.2]
    Clue count varies by task (per-sample coupling active)
  Per-Clue Entropy: [4.64, 4.63, 4.63, 4.63, 4.63, 4.63, 4.63] (mean=4.63, max=6.80)
    [!] Clues have uniform entropy (std=0.003) - not differentiating!
  Centroid Spread: 0.10 (higher=more diverse)
    🚨 CRITICAL COLLAPSE: Spread=0.10 < 0.5 - all clues at same location!
    [!] Stop predictor cannot differentiate - needs diversity regularizer
  --- Stop Predictor Coupling ---
  Entropy Input to Stop: 0.6808 (normalized, lower=sharper)
  Per-Clue Entropy Input: [0.682, 0.681, 0.680, 0.680, 0.680, 0.681, 0.681]
  Per-Clue Stop Prob: [0.242, 0.246, 0.262, 0.258, 0.232, 0.256, 0.252]
  --- Sparsity Loss Breakdown (Per-Sample Coupled) ---
  Min Clue Penalty: 0.0000 (per-sample avg)
  Per-Sample Clue Penalty (scaled): 0.0000
  Base Pondering: 0.1050 (clues=5.25)
  Entropy Pondering: 0.0716
  --- Per-Class Distribution (Valid Pixels Only) ---
  Padding: 87.4% of grid (ignored in loss)
  Pred %: [38.9, 0.9, 1.7, 12.0, 3.5, 22.1, 1.1, 15.0, 1.0, 3.7]
  Target %: [69.2, 3.9, 4.1, 5.1, 3.8, 4.6, 2.1, 1.0, 5.7, 0.4]
  Per-Class Acc %: [39, 1, 2, 14, 5, 13, 3, 13, 5, 8]
  [WARN] FG color preference: 34% are color 0
  ==================================================
  PER-SAMPLE TRAINING ACCURACY (Epoch 8)
  ==================================================
  ★ Mean Accuracy: 19.5%
  ★ Exact Match: 0/250 (0.0%)
  ★ High Acc (≥90%): 0/250 (0.0%)
  FG Accuracy: 6.7%
  BG Accuracy: 30.8%
  Batch Trend: 25.5% → 16.8% (↓ 8.7pp)
    [!] ACCURACY DECLINING within epoch - potential overfitting or instability!
  Accuracy Distribution: 0-25%:0%, 25-50%:100%, 50-75%:0%, 75-90%:0%, 90-100%:0%
  Running Window (last 5 batches): 19.5% ± 3.5%

  --- PER-COLOR ACCURACY (10 classes) ---
  Color:       0     1     2     3     4     5     6     7     8     9
  Acc%:      39%    1%    2%   14%    5%   13%    3%   13%    5%    8%
  Target:  61.2%  6.2%  5.5%  6.2%  4.5%  4.8%  2.2%  1.9%  5.7%  1.6%
  Pred:    35.0%  1.2%  2.1% 13.3%  5.2% 20.9%  1.9% 14.6%  1.6%  4.2%
  [!] Weak colors (<50% acc): 0(Black), 1(Blue), 2(Red), 3(Green), 4(Yellow), 5(Gray), 6(Pink), 7(Orange), 8(Cyan), 9(Brown)
  [!] Under-predicting color 1 (Blue): 1.2% vs target 6.2%
  [!] Under-predicting color 2 (Red): 2.1% vs target 5.5%
  [!] Under-predicting color 8 (Cyan): 1.6% vs target 5.7%
  ==================================================

  ==================================================
  LEARNING TRAJECTORY (Epoch 8)
  ==================================================
  Stop Prob:   0.250 → (init=0.27, task-dependent)
  Exp. Clues:  5.25 (latent variable, task-dependent)
  Attn Entropy: 4.63 ↓ (max=6.8, sharper=better)
  Task Loss:   0.2761 ↓
  Train Acc:   19.5% →
  Exact Match: 0.0% ↓
  Best Step:   5 (later=better refinement)
  FG Coverage: 198.7% of target ↑
  [!] Potential issues: stop_prob not increasing, attention not sharpening, train_accuracy not improving
  ==================================================

  ############################################################
  🚦 TRAINING HEALTH CHECK - Epoch 8
  ############################################################
  ✓ Attention sharpening (0.68 < 0.7)
  ✓ No NaN/Inf issues
  ⚠ Stop probs nearly uniform (global=0.080, per_clue=0.009)
  ⚠ Weak confidence-stop coupling (r=0.20)
  ⚠ Loss slowly decreasing (0.311 → 0.276)
  ⚠ Accuracy flat (19.9% → 19.5%) - early epoch
  ⚠ Color preference (34% one color)
  ✗ Centroid COLLAPSE (0.1 < 0.5)
  --------------------------------------------------------
  RESULT: 2/8 checks passed
  STATUS: 🟠 WARNING
  → Multiple issues detected. Consider intervention soon.

  RECOMMENDED ACTIONS:
  ############################################################
  HPM Instance Buffer: 0 entries (empty, not saved)
  HPM Procedural Buffer: 0 entries (empty, not saved)
  Saved checkpoint to checkpoints\rlan_stable_merged\latest.pt
  [ARCDataset] Augmentation config updated:
    dihedral=False, color_perm=False (prob=0.50), translational=False

Epoch 9/200
----------------------------------------
  [Batch 0] Grid size: 30x30 (batch_max_size=30)

  [MEMORY] Epoch 8 Batch 0:
    Baseline: alloc=92MB, reserved=19372MB
    01_batch_on_gpu: alloc=104MB (+12MB), reserved=19394MB
    02_after_forward: alloc=13240MB (+13136MB), reserved=19394MB
    03_before_backward: alloc=13320MB (+80MB), reserved=19394MB
    04_after_backward: alloc=220MB (-13100MB), reserved=19394MB
    PEAK: alloc=13783MB, reserved=19394MB / 24576MB (78.9%)

    [BREAKDOWN] Largest increase at '02_after_forward' (+13136MB):
      Model params: 73.4MB
      Model grads:  0.0MB
      Optimizer:    146.7MB (estimate (Adam-like: AdamW))
      Batch data:   4.1MB
      Activations:  68.3MB
      Output tensors (top 5):
        features: 43.9MB
        support_features: 15.6MB
        all_logits: 6.0MB
        attention_maps: 1.2MB
        logits: 0.9MB
      Module params (top 5):
        solver: 28.4MB
        hyper_lora: 19.1MB
        context_encoder: 15.0MB
        context_injector: 3.0MB
        hpm: 2.9MB
      Inactive modules: hyperlora, solver_context, cross_attention, hpm, loo, equivariance
  Batch 0/5: loss=0.7515, focal_weighted=0.5913, batch_acc=21.3%, exact=0/50 (0.0%), running_acc=21.3%, lr=5.00e-04
    [TaskTrack] Global_Solved: 1/342, Epoch_Solved: 0, New_Puzzles: 0
    FG: batch=5.1% run50=5.1% | BG: batch=39.0% run50=39.0%
    Per-Color: [0:39% 1:1% 2:3% 3:8% 4:7% 5:9% 6:5% 7:7% 8:7% 9:7%]
    Running50: [0:39% 1:1% 2:3% 3:8% 4:7% 5:9% 6:5% 7:7% 8:7% 9:7%]
    Solver: [2.609, 2.609, 2.609, 2.594, 2.594, 2.594, 2.594] ⚠ best=3
  [Batch 1] Grid size: 30x30 (batch_max_size=30)

  [MEMORY] Epoch 8 Batch 1:
    Baseline: alloc=92MB, reserved=19372MB
    01_batch_on_gpu: alloc=211MB (+118MB), reserved=19396MB
    02_after_forward: alloc=17803MB (+17592MB), reserved=19396MB
    03_before_backward: alloc=17829MB (+27MB), reserved=19396MB
    04_after_backward: alloc=292MB (-17537MB), reserved=19396MB
    PEAK: alloc=18345MB, reserved=19396MB / 24576MB (78.9%)

    [BREAKDOWN] Largest increase at '02_after_forward' (+17592MB):
      Model params: 73.4MB
      Model grads:  45.9MB
      Optimizer:    146.7MB (estimate (Adam-like: AdamW))
      Batch data:   7.6MB
      Activations:  84.1MB
      Output tensors (top 5):
        features: 43.9MB
        support_features: 31.2MB
        all_logits: 6.0MB
        attention_maps: 1.2MB
        logits: 0.9MB
      Module params (top 5):
        solver: 28.4MB
        hyper_lora: 19.1MB
        context_encoder: 15.0MB
        context_injector: 3.0MB
        hpm: 2.9MB
      Inactive modules: hyperlora, solver_context, cross_attention, hpm, loo, equivariance
  Batch 1/5: loss=0.2532, focal_weighted=0.1188, batch_acc=16.5%, exact=0/100 (0.0%), running_acc=18.9%, lr=5.00e-04
    [TaskTrack] Global_Solved: 1/342, Epoch_Solved: 0, New_Puzzles: 0
    FG: batch=6.2% run50=5.7% | BG: batch=24.9% run50=31.9%
    Per-Color: [0:31% 1:4% 2:3% 3:19% 4:5% 5:15% 6:6% 7:9% 8:2% 9:0%]
    Running50: [0:35% 1:3% 2:3% 3:14% 4:6% 5:12% 6:5% 7:8% 8:4% 9:3%]
  [Batch 2] Grid size: 30x30 (batch_max_size=30)

  [MEMORY] Epoch 8 Batch 2:
    Baseline: alloc=92MB, reserved=19372MB
    01_batch_on_gpu: alloc=215MB (+123MB), reserved=19396MB
    02_after_forward: alloc=14237MB (+14022MB), reserved=19396MB
    03_before_backward: alloc=14263MB (+26MB), reserved=19396MB
    04_after_backward: alloc=281MB (-13982MB), reserved=19396MB
    PEAK: alloc=14780MB, reserved=19396MB / 24576MB (78.9%)

    [BREAKDOWN] Largest increase at '02_after_forward' (+14022MB):
      Model params: 73.4MB
      Model grads:  45.9MB
      Optimizer:    146.7MB (estimate (Adam-like: AdamW))
      Batch data:   4.8MB
      Activations:  71.5MB
      Output tensors (top 5):
        features: 43.9MB
        support_features: 18.8MB
        all_logits: 6.0MB
        attention_maps: 1.2MB
        logits: 0.9MB
      Module params (top 5):
        solver: 28.4MB
        hyper_lora: 19.1MB
        context_encoder: 15.0MB
        context_injector: 3.0MB
        hpm: 2.9MB
      Inactive modules: hyperlora, solver_context, cross_attention, hpm, loo, equivariance
  Batch 2/5: loss=0.2525, focal_weighted=0.1215, batch_acc=16.0%, exact=0/150 (0.0%), running_acc=17.9%, lr=5.00e-04
    [TaskTrack] Global_Solved: 1/342, Epoch_Solved: 0, New_Puzzles: 0
    FG: batch=7.8% run50=6.4% | BG: batch=24.1% run50=29.3%
    Per-Color: [0:30% 1:1% 2:2% 3:29% 4:7% 5:10% 6:2% 7:9% 8:3% 9:13%]
    Running50: [0:34% 1:2% 2:3% 3:19% 4:7% 5:11% 6:4% 7:9% 8:4% 9:7%]
  [Batch 3] Grid size: 30x30 (batch_max_size=30)

  [MEMORY] Epoch 8 Batch 3:
    Baseline: alloc=92MB, reserved=19372MB
    01_batch_on_gpu: alloc=213MB (+120MB), reserved=19396MB
    02_after_forward: alloc=16020MB (+15808MB), reserved=19396MB
    03_before_backward: alloc=16047MB (+27MB), reserved=19396MB
    04_after_backward: alloc=287MB (-15760MB), reserved=19396MB
    PEAK: alloc=16563MB, reserved=19396MB / 24576MB (78.9%)

    [BREAKDOWN] Largest increase at '02_after_forward' (+15808MB):
      Model params: 73.4MB
      Model grads:  45.9MB
      Optimizer:    146.7MB (estimate (Adam-like: AdamW))
      Batch data:   6.2MB
      Activations:  77.8MB
      Output tensors (top 5):
        features: 43.9MB
        support_features: 25.0MB
        all_logits: 6.0MB
        attention_maps: 1.2MB
        logits: 0.9MB
      Module params (top 5):
        solver: 28.4MB
        hyper_lora: 19.1MB
        context_encoder: 15.0MB
        context_injector: 3.0MB
        hpm: 2.9MB
      Inactive modules: hyperlora, solver_context, cross_attention, hpm, loo, equivariance
  Batch 3/5: loss=0.3220, focal_weighted=0.1901, batch_acc=17.7%, exact=0/200 (0.0%), running_acc=17.9%, lr=5.00e-04
    [TaskTrack] Global_Solved: 1/342, Epoch_Solved: 0, New_Puzzles: 0
    FG: batch=7.6% run50=6.7% | BG: batch=24.6% run50=28.1%
    Per-Color: [0:31% 1:2% 2:5% 3:16% 4:8% 5:8% 6:0% 7:14% 8:0% 9:8%]
    Running50: [0:33% 1:2% 2:3% 3:18% 4:7% 5:10% 6:3% 7:10% 8:3% 9:7%]
  [Batch 4] Grid size: 30x30 (batch_max_size=30)

  [MEMORY] Epoch 8 Batch 4:
    Baseline: alloc=92MB, reserved=19372MB
    01_batch_on_gpu: alloc=207MB (+115MB), reserved=19396MB
    02_after_forward: alloc=13337MB (+13130MB), reserved=19396MB
    03_before_backward: alloc=13382MB (+46MB), reserved=19396MB
    04_after_backward: alloc=270MB (-13113MB), reserved=19396MB
    PEAK: alloc=13879MB, reserved=19396MB / 24576MB (78.9%)

    [BREAKDOWN] Largest increase at '02_after_forward' (+13130MB):
      Model params: 73.4MB
      Model grads:  45.9MB
      Optimizer:    146.7MB (estimate (Adam-like: AdamW))
      Batch data:   4.1MB
      Activations:  68.3MB
      Output tensors (top 5):
        features: 43.9MB
        support_features: 15.6MB
        all_logits: 6.0MB
        attention_maps: 1.2MB
        logits: 0.9MB
      Module params (top 5):
        solver: 28.4MB
        hyper_lora: 19.1MB
        context_encoder: 15.0MB
        context_injector: 3.0MB
        hpm: 2.9MB
      Inactive modules: hyperlora, solver_context, cross_attention, hpm, loo, equivariance
  Batch 4/5: loss=0.4781, focal_weighted=0.3343, batch_acc=19.8%, exact=0/250 (0.0%), running_acc=18.3%, lr=5.00e-04
    [TaskTrack] Global_Solved: 1/342, Epoch_Solved: 0, New_Puzzles: 0
    FG: batch=7.1% run50=6.7% | BG: batch=30.5% run50=28.6%
    Per-Color: [0:37% 1:0% 2:2% 3:15% 4:3% 5:10% 6:8% 7:9% 8:6% 9:2%]
    Running50: [0:34% 1:2% 2:3% 3:18% 4:6% 5:10% 6:4% 7:10% 8:3% 9:6%]

Epoch 9 Summary:
  Total Loss: 0.4115
  Task Loss (focal): 0.2712
  [Global Task Progress] Unique Solved: 1/342 (0.3%)
    NEW puzzles solved this epoch: 0
    ⚠️ No new puzzles solved in last 5 epochs - model may be plateauing
  Entropy Loss: 4.2165 (weight=0.01)
  Sparsity Loss: 0.1688 (weight=0.5)
  Predicate Loss: 0.0000 (weight=0.01)
  Time: 4.2s, LR: 5.00e-04
    Per-module LRs: DSC:5.00e-04, MSRE:5.00e-04, Other:5.00e-04
  Temperature: 0.9727 (lower=sharper attention)
  Samples Processed: 250 (5 batches)
  Dihedral Distribution: [100.0%, 0.0%, 0.0%, 0.0%, 0.0%, 0.0%, 0.0%, 0.0%]
    [!] Non-uniform dihedral distribution (max dev: 87.5%)
  Color Permutation: 0.0% (0/250)
  Translational Aug: 0.0% (0/250), unique offsets: 0
  Aug Quality: OK
  --- Training Diagnostics ---
  Solver Steps: 7 (deep supervision active)
  Per-Step Loss (epoch avg, 1 batches): [2.6094, 2.6094, 2.6094, 2.5938, 2.5938, 2.5938, 2.5938]
    [!] Best step is 3 (middle), not last - solver may be over-iterating!
    [!] Best: step 3 (2.5938), Final: step 6 (2.5938)
    [!] Potential gain from best-step selection: 0.0% lower loss
  Best-Step Histogram: []
  Solver Health: Last step best: 0.0%, Earlier step best: 100.0%
  Avg Step Improvement: 0.6% (step0→stepN)
    ⚠️ SOLVER OVER-ITERATION WARNING: 100% of batches had earlier step as best!
    ⚠️ Consider: (1) enabling best-step selection, (2) reducing num_solver_steps, (3) enabling ACT
  Grad Norms: DSC=0.0000, StopPred=0.0000, Encoder=0.0000, Solver=0.0000
  StopPred Weight Var: 2.80e-02
  Attention: max=0.0125, min=0.000000
  Stop Prob: 0.254 (approx 5.2 clues active)
  Stop Probs Std: 0.089 (global std across batch×clues)
  Clues Used: mean=5.22, std=0.37, range=[3.9, 6.2]
  Clue-Loss Correlation: -0.333 (unexpected negative - check gradient flow)
  Stop Logits: mean=-1.13, std=0.48, range=[-2.8, 0.4]
    Clue count varies by task (per-sample coupling active)
  Per-Clue Entropy: [5.36, 5.35, 5.34, 5.34, 5.34, 5.34, 5.34] (mean=5.35, max=6.80)
    [!] Clues have uniform entropy (std=0.006) - not differentiating!
    [!] High entropy (5.35) - attention still diffuse!
  Centroid Spread: 0.24 (higher=more diverse)
    🚨 CRITICAL COLLAPSE: Spread=0.24 < 0.5 - all clues at same location!
    [!] Stop predictor cannot differentiate - needs diversity regularizer
  --- Stop Predictor Coupling ---
  Entropy Input to Stop: 0.7858 (normalized, lower=sharper)
  Per-Clue Entropy Input: [0.787, 0.787, 0.786, 0.785, 0.785, 0.785, 0.786]
  Per-Clue Stop Prob: [0.266, 0.247, 0.256, 0.254, 0.264, 0.236, 0.252]
  --- Sparsity Loss Breakdown (Per-Sample Coupled) ---
  Min Clue Penalty: 0.0000 (per-sample avg)
  Per-Sample Clue Penalty (scaled): 0.0000
  Base Pondering: 0.1045 (clues=5.23)
  Entropy Pondering: 0.0825
  --- Per-Class Distribution (Valid Pixels Only) ---
  Padding: 77.2% of grid (ignored in loss)
  Pred %: [36.9, 1.2, 2.2, 11.1, 6.0, 20.4, 2.3, 13.5, 2.1, 4.3]
  Target %: [57.4, 10.1, 3.8, 9.6, 2.6, 3.6, 1.7, 1.2, 6.5, 3.5]
  Per-Class Acc %: [37, 1, 3, 11, 6, 10, 5, 9, 5, 7]
  [WARN] FG color preference: 34% are color 0
  ==================================================
  PER-SAMPLE TRAINING ACCURACY (Epoch 9)
  ==================================================
  ★ Mean Accuracy: 18.3%
  ★ Exact Match: 0/250 (0.0%)
  ★ High Acc (≥90%): 0/250 (0.0%)
  FG Accuracy: 6.7%
  BG Accuracy: 28.6%
  Batch Trend: 21.3% → 18.8% (↓ 2.5pp)
  Accuracy Distribution: 0-25%:75%, 25-50%:25%, 50-75%:0%, 75-90%:0%, 90-100%:0%
    [!] Many samples stuck at low accuracy - check data/model!
  Running Window (last 5 batches): 18.3% ± 2.0%

  --- PER-COLOR ACCURACY (10 classes) ---
  Color:       0     1     2     3     4     5     6     7     8     9
  Acc%:      37%    1%    3%   11%    6%   10%    5%    9%    5%    7%
  Target:  58.0%  8.4%  4.6%  8.1%  3.4%  4.4%  2.3%  1.6%  6.7%  2.6%
  Pred:    33.6%  1.3%  2.3% 13.7%  5.9% 20.3%  1.9% 14.7%  1.8%  4.7%
  [!] Weak colors (<50% acc): 0(Black), 1(Blue), 2(Red), 3(Green), 4(Yellow), 5(Gray), 6(Pink), 7(Orange), 8(Cyan), 9(Brown)
  [!] Under-predicting color 1 (Blue): 1.3% vs target 8.4%
  [!] Under-predicting color 2 (Red): 2.3% vs target 4.6%
  [!] Under-predicting color 8 (Cyan): 1.8% vs target 6.7%
  ==================================================

  ==================================================
  LEARNING TRAJECTORY (Epoch 9)
  ==================================================
  Stop Prob:   0.254 → (init=0.27, task-dependent)
  Exp. Clues:  5.23 (latent variable, task-dependent)
  Attn Entropy: 5.35 ↑ (max=6.8, sharper=better)
  Task Loss:   0.2712 →
  Train Acc:   18.3% ↓
  Exact Match: 0.0% →
  Best Step:   3 (later=better refinement)
  FG Coverage: 148.2% of target ↓
  [!] Potential issues: stop_prob not increasing, attention not sharpening, train_accuracy not improving
  ==================================================

  ############################################################
  🚦 TRAINING HEALTH CHECK - Epoch 9
  ############################################################
  ✓ No NaN/Inf issues
  ⚠ Attention still diffuse (0.79)
  ⚠ Stop probs nearly uniform (global=0.089, per_clue=0.009)
  ⚠ Negative coupling (r=-0.30) - early epoch OK
  ⚠ Loss slowly decreasing (0.311 → 0.271)
  ⚠ Accuracy flat (19.9% → 18.3%) - early epoch
  ⚠ Color preference (34% one color)
  ✗ Centroid COLLAPSE (0.2 < 0.5)
  --------------------------------------------------------
  RESULT: 1/8 checks passed
  STATUS: 🟠 WARNING
  → Multiple issues detected. Consider intervention soon.

  RECOMMENDED ACTIONS:
  ############################################################
  HPM Instance Buffer: 0 entries (empty, not saved)
  HPM Procedural Buffer: 0 entries (empty, not saved)
  Saved checkpoint to checkpoints\rlan_stable_merged\latest.pt
  [ARCDataset] Augmentation config updated:
    dihedral=False, color_perm=False (prob=0.50), translational=False

Epoch 10/200
----------------------------------------
  [Batch 0] Grid size: 30x30 (batch_max_size=30)

  [MEMORY] Epoch 9 Batch 0:
    Baseline: alloc=92MB, reserved=19396MB
    01_batch_on_gpu: alloc=104MB (+12MB), reserved=19418MB
    02_after_forward: alloc=15915MB (+15811MB), reserved=19418MB
    03_before_backward: alloc=15964MB (+49MB), reserved=19418MB
    04_after_backward: alloc=231MB (-15733MB), reserved=19418MB
    PEAK: alloc=16457MB, reserved=19418MB / 24576MB (79.0%)

    [BREAKDOWN] Largest increase at '02_after_forward' (+15811MB):
      Model params: 73.4MB
      Model grads:  0.0MB
      Optimizer:    146.7MB (estimate (Adam-like: AdamW))
      Batch data:   6.2MB
      Activations:  77.8MB
      Output tensors (top 5):
        features: 43.9MB
        support_features: 25.0MB
        all_logits: 6.0MB
        attention_maps: 1.2MB
        logits: 0.9MB
      Module params (top 5):
        solver: 28.4MB
        hyper_lora: 19.1MB
        context_encoder: 15.0MB
        context_injector: 3.0MB
        hpm: 2.9MB
      Inactive modules: hyperlora, solver_context, cross_attention, hpm, loo, equivariance
  Batch 0/5: loss=0.4331, focal_weighted=0.2816, batch_acc=24.3%, exact=0/50 (0.0%), running_acc=24.3%, lr=5.00e-04
    [TaskTrack] Global_Solved: 1/342, Epoch_Solved: 0, New_Puzzles: 0
    FG: batch=6.0% run50=6.0% | BG: batch=38.1% run50=38.1%
    Per-Color: [0:43% 1:1% 2:3% 3:14% 4:4% 5:10% 6:4% 7:22% 8:4% 9:3%]
    Running50: [0:43% 1:1% 2:3% 3:14% 4:4% 5:10% 6:4% 7:22% 8:4% 9:3%]
    Solver: [2.156, 2.156, 2.156, 2.156, 2.141, 2.141, 2.141] ⚠ best=4
  [Batch 1] Grid size: 30x30 (batch_max_size=30)

  [MEMORY] Epoch 9 Batch 1:
    Baseline: alloc=92MB, reserved=19396MB
    01_batch_on_gpu: alloc=215MB (+122MB), reserved=19420MB
    02_after_forward: alloc=15128MB (+14914MB), reserved=19420MB
    03_before_backward: alloc=15153MB (+24MB), reserved=19420MB
    04_after_backward: alloc=285MB (-14868MB), reserved=19420MB
    PEAK: alloc=15671MB, reserved=19420MB / 24576MB (79.0%)

    [BREAKDOWN] Largest increase at '02_after_forward' (+14914MB):
      Model params: 73.4MB
      Model grads:  45.9MB
      Optimizer:    146.7MB (estimate (Adam-like: AdamW))
      Batch data:   5.5MB
      Activations:  74.6MB
      Output tensors (top 5):
        features: 43.9MB
        support_features: 21.9MB
        all_logits: 6.0MB
        attention_maps: 1.2MB
        logits: 0.9MB
      Module params (top 5):
        solver: 28.4MB
        hyper_lora: 19.1MB
        context_encoder: 15.0MB
        context_injector: 3.0MB
        hpm: 2.9MB
      Inactive modules: hyperlora, solver_context, cross_attention, hpm, loo, equivariance
  Batch 1/5: loss=0.2999, focal_weighted=0.1704, batch_acc=16.9%, exact=0/100 (0.0%), running_acc=20.6%, lr=5.00e-04
    [TaskTrack] Global_Solved: 1/342, Epoch_Solved: 0, New_Puzzles: 0
    FG: batch=9.1% run50=7.5% | BG: batch=24.7% run50=31.4%
    Per-Color: [0:30% 1:5% 2:4% 3:21% 4:11% 5:19% 6:3% 7:10% 8:0% 9:7%]
    Running50: [0:36% 1:3% 2:4% 3:17% 4:8% 5:15% 6:4% 7:16% 8:2% 9:5%]
  [Batch 2] Grid size: 30x30 (batch_max_size=30)

  [MEMORY] Epoch 9 Batch 2:
    Baseline: alloc=92MB, reserved=19396MB
    01_batch_on_gpu: alloc=215MB (+122MB), reserved=19420MB
    02_after_forward: alloc=17805MB (+17590MB), reserved=19420MB
    03_before_backward: alloc=17833MB (+28MB), reserved=19420MB
    04_after_backward: alloc=296MB (-17538MB), reserved=19420MB
    PEAK: alloc=18347MB, reserved=19420MB / 24576MB (79.0%)

    [BREAKDOWN] Largest increase at '02_after_forward' (+17590MB):
      Model params: 73.4MB
      Model grads:  45.9MB
      Optimizer:    146.7MB (estimate (Adam-like: AdamW))
      Batch data:   7.6MB
      Activations:  84.1MB
      Output tensors (top 5):
        features: 43.9MB
        support_features: 31.2MB
        all_logits: 6.0MB
        attention_maps: 1.2MB
        logits: 0.9MB
      Module params (top 5):
        solver: 28.4MB
        hyper_lora: 19.1MB
        context_encoder: 15.0MB
        context_injector: 3.0MB
        hpm: 2.9MB
      Inactive modules: hyperlora, solver_context, cross_attention, hpm, loo, equivariance
  Batch 2/5: loss=0.2776, focal_weighted=0.1382, batch_acc=19.0%, exact=0/150 (0.0%), running_acc=20.0%, lr=5.00e-04
    [TaskTrack] Global_Solved: 1/342, Epoch_Solved: 0, New_Puzzles: 0
    FG: batch=7.6% run50=7.5% | BG: batch=32.5% run50=31.7%
    Per-Color: [0:33% 1:1% 2:2% 3:10% 4:10% 5:13% 6:0% 7:16% 8:1% 9:0%]
    Running50: [0:35% 1:2% 2:3% 3:15% 4:9% 5:14% 6:3% 7:16% 8:2% 9:3%]
  [Batch 3] Grid size: 30x30 (batch_max_size=30)

  [MEMORY] Epoch 9 Batch 3:
    Baseline: alloc=92MB, reserved=19396MB
    01_batch_on_gpu: alloc=215MB (+122MB), reserved=19420MB
    02_after_forward: alloc=13344MB (+13129MB), reserved=19420MB
    03_before_backward: alloc=13421MB (+78MB), reserved=19420MB
    04_after_backward: alloc=277MB (-13144MB), reserved=19420MB
    PEAK: alloc=13886MB, reserved=19420MB / 24576MB (79.0%)

    [BREAKDOWN] Largest increase at '02_after_forward' (+13129MB):
      Model params: 73.4MB
      Model grads:  45.9MB
      Optimizer:    146.7MB (estimate (Adam-like: AdamW))
      Batch data:   4.1MB
      Activations:  68.3MB
      Output tensors (top 5):
        features: 43.9MB
        support_features: 15.6MB
        all_logits: 6.0MB
        attention_maps: 1.2MB
        logits: 0.9MB
      Module params (top 5):
        solver: 28.4MB
        hyper_lora: 19.1MB
        context_encoder: 15.0MB
        context_injector: 3.0MB
        hpm: 2.9MB
      Inactive modules: hyperlora, solver_context, cross_attention, hpm, loo, equivariance
  Batch 3/5: loss=0.8677, focal_weighted=0.7055, batch_acc=21.0%, exact=0/200 (0.0%), running_acc=20.3%, lr=5.00e-04
    [TaskTrack] Global_Solved: 1/342, Epoch_Solved: 0, New_Puzzles: 0
    FG: batch=7.7% run50=7.6% | BG: batch=36.3% run50=32.9%
    Per-Color: [0:40% 1:2% 2:3% 3:10% 4:8% 5:20% 6:4% 7:8% 8:7% 9:7%]
    Running50: [0:36% 1:2% 2:3% 3:14% 4:8% 5:16% 6:3% 7:14% 8:3% 9:4%]
  [Batch 4] Grid size: 30x30 (batch_max_size=30)

  [MEMORY] Epoch 9 Batch 4:
    Baseline: alloc=92MB, reserved=19396MB
    01_batch_on_gpu: alloc=207MB (+115MB), reserved=19420MB
    02_after_forward: alloc=14232MB (+14025MB), reserved=19420MB
    03_before_backward: alloc=14259MB (+27MB), reserved=19420MB
    04_after_backward: alloc=276MB (-13983MB), reserved=19420MB
    PEAK: alloc=14775MB, reserved=19420MB / 24576MB (79.0%)

    [BREAKDOWN] Largest increase at '02_after_forward' (+14025MB):
      Model params: 73.4MB
      Model grads:  45.9MB
      Optimizer:    146.7MB (estimate (Adam-like: AdamW))
      Batch data:   4.8MB
      Activations:  71.5MB
      Output tensors (top 5):
        features: 43.9MB
        support_features: 18.8MB
        all_logits: 6.0MB
        attention_maps: 1.2MB
        logits: 0.9MB
      Module params (top 5):
        solver: 28.4MB
        hyper_lora: 19.1MB
        context_encoder: 15.0MB
        context_injector: 3.0MB
        hpm: 2.9MB
      Inactive modules: hyperlora, solver_context, cross_attention, hpm, loo, equivariance
  Batch 4/5: loss=0.2735, focal_weighted=0.1432, batch_acc=17.7%, exact=1/250 (0.4%), running_acc=19.8%, lr=5.00e-04
    [TaskTrack] Global_Solved: 1/342, Epoch_Solved: 1, New_Puzzles: 0
    FG: batch=7.3% run50=7.5% | BG: batch=23.1% run50=30.9%
    Per-Color: [0:31% 1:0% 2:4% 3:20% 4:5% 5:5% 6:2% 7:10% 8:6% 9:0%]
    Running50: [0:35% 1:2% 2:3% 3:15% 4:8% 5:13% 6:3% 7:13% 8:4% 9:4%]

Epoch 10 Summary:
  Total Loss: 0.4304
  Task Loss (focal): 0.2878
  [Global Task Progress] Unique Solved: 1/342 (0.3%)
    NEW puzzles solved this epoch: 0
    ⚠️ No new puzzles solved in last 5 epochs - model may be plateauing
  Entropy Loss: 4.2878 (weight=0.01)
  Sparsity Loss: 0.1721 (weight=0.5)
  Predicate Loss: 0.0000 (weight=0.01)
  Time: 4.3s, LR: 5.00e-04
    Per-module LRs: DSC:5.00e-04, MSRE:5.00e-04, Other:5.00e-04
  Temperature: 0.9693 (lower=sharper attention)
  Samples Processed: 250 (5 batches)
  Dihedral Distribution: [100.0%, 0.0%, 0.0%, 0.0%, 0.0%, 0.0%, 0.0%, 0.0%]
    [!] Non-uniform dihedral distribution (max dev: 87.5%)
  Color Permutation: 0.0% (0/250)
  Translational Aug: 0.0% (0/250), unique offsets: 0
  Aug Quality: OK
  --- Training Diagnostics ---
  Solver Steps: 7 (deep supervision active)
  Per-Step Loss (epoch avg, 1 batches): [2.1562, 2.1562, 2.1562, 2.1562, 2.1406, 2.1406, 2.1406]
    [!] Best step is 4 (middle), not last - solver may be over-iterating!
    [!] Best: step 4 (2.1406), Final: step 6 (2.1406)
    [!] Potential gain from best-step selection: 0.0% lower loss
  Best-Step Histogram: []
  Solver Health: Last step best: 0.0%, Earlier step best: 100.0%
  Avg Step Improvement: 0.7% (step0→stepN)
    ⚠️ SOLVER OVER-ITERATION WARNING: 100% of batches had earlier step as best!
    ⚠️ Consider: (1) enabling best-step selection, (2) reducing num_solver_steps, (3) enabling ACT
  Grad Norms: DSC=0.0000, StopPred=0.0000, Encoder=0.0000, Solver=0.0000
  StopPred Weight Var: 2.80e-02
  Attention: max=0.0133, min=0.000000
  Stop Prob: 0.249 (approx 5.3 clues active)
  Stop Probs Std: 0.070 (global std across batch×clues)
  Clues Used: mean=5.26, std=0.27, range=[4.5, 5.8]
  Clue-Loss Correlation: +0.107 (learning - per-sample coupling active)
  Stop Logits: mean=-1.14, std=0.38, range=[-2.4, -0.0]
  Per-Clue Entropy: [4.82, 4.81, 4.81, 4.81, 4.81, 4.81, 4.81] (mean=4.81, max=6.80)
    [!] Clues have uniform entropy (std=0.003) - not differentiating!
  Centroid Spread: 0.11 (higher=more diverse)
    🚨 CRITICAL COLLAPSE: Spread=0.11 < 0.5 - all clues at same location!
    [!] Stop predictor cannot differentiate - needs diversity regularizer
  --- Stop Predictor Coupling ---
  Entropy Input to Stop: 0.7071 (normalized, lower=sharper)
  Per-Clue Entropy Input: [0.708, 0.707, 0.707, 0.707, 0.707, 0.707, 0.707]
  Per-Clue Stop Prob: [0.241, 0.249, 0.254, 0.252, 0.258, 0.241, 0.250]
  --- Sparsity Loss Breakdown (Per-Sample Coupled) ---
  Min Clue Penalty: 0.0000 (per-sample avg)
  Per-Sample Clue Penalty (scaled): 0.0000
  Base Pondering: 0.1051 (clues=5.25)
  Entropy Pondering: 0.0744
  --- Per-Class Distribution (Valid Pixels Only) ---
  Padding: 86.9% of grid (ignored in loss)
  Pred %: [38.5, 0.4, 2.3, 11.2, 4.0, 20.4, 0.9, 18.0, 0.9, 3.5]
  Target %: [62.4, 3.0, 4.0, 5.1, 4.2, 5.3, 4.8, 1.0, 9.6, 0.6]
    [!] Missing foreground colors: [1]
  Per-Class Acc %: [39, 2, 3, 12, 7, 13, 4, 12, 4, 6]
  [WARN] FG color preference: 32% are color 0
  ==================================================
  PER-SAMPLE TRAINING ACCURACY (Epoch 10)
  ==================================================
  ★ Mean Accuracy: 19.8%
  ★ Exact Match: 1/250 (0.4%)
  ★ High Acc (≥90%): 1/250 (0.4%)
  FG Accuracy: 7.5%
  BG Accuracy: 30.9%
  Batch Trend: 24.3% → 19.3% (↓ 4.9pp)
  Accuracy Distribution: 0-25%:75%, 25-50%:25%, 50-75%:0%, 75-90%:0%, 90-100%:0%
    [!] Many samples stuck at low accuracy - check data/model!
  Running Window (last 5 batches): 19.8% ± 2.6%

  --- PER-COLOR ACCURACY (10 classes) ---
  Color:       0     1     2     3     4     5     6     7     8     9
  Acc%:      39%    2%    3%   12%    7%   13%    4%   12%    4%    6%
  Target:  56.7%  6.6%  4.9%  8.0%  5.0%  4.5%  3.0%  1.9%  7.7%  1.7%
  Pred:    33.5%  1.1%  2.5% 12.8%  5.8% 20.8%  2.2% 14.9%  1.9%  4.4%
  [!] Weak colors (<50% acc): 0(Black), 1(Blue), 2(Red), 3(Green), 4(Yellow), 5(Gray), 6(Pink), 7(Orange), 8(Cyan), 9(Brown)
  [!] Under-predicting color 1 (Blue): 1.1% vs target 6.6%
  [!] Under-predicting color 8 (Cyan): 1.9% vs target 7.7%
  ==================================================

  ==================================================
  LEARNING TRAJECTORY (Epoch 10)
  ==================================================
  Stop Prob:   0.249 → (init=0.27, task-dependent)
  Exp. Clues:  5.25 (latent variable, task-dependent)
  Attn Entropy: 4.81 ↓ (max=6.8, sharper=better)
  Task Loss:   0.2878 ↑
  Train Acc:   19.8% ↑
  Exact Match: 0.4% ↑
  Best Step:   4 (later=better refinement)
  FG Coverage: 163.7% of target ↑
  [!] Potential issues: stop_prob not increasing, attention not sharpening, task_loss not decreasing, train_accuracy not improving
  ==================================================

  ############################################################
  🚦 TRAINING HEALTH CHECK - Epoch 10
  ############################################################
  ✓ Good confidence-stop coupling (r=0.68)
  ✓ No NaN/Inf issues
  ⚠ Attention still diffuse (0.71)
  ⚠ Stop probs nearly uniform (global=0.070, per_clue=0.006)
  ⚠ Loss slowly decreasing (0.311 → 0.288)
  ⚠ Accuracy flat (19.9% → 19.8%) - early epoch
  ⚠ Color preference (32% one color)
  ✗ Centroid COLLAPSE (0.1 < 0.5)
  --------------------------------------------------------
  RESULT: 2/8 checks passed
  STATUS: 🟠 WARNING
  → Multiple issues detected. Consider intervention soon.

  RECOMMENDED ACTIONS:
  ############################################################

  [Eval] Running evaluation on 8 batches... 8/8
  [TRM-Eval] Running TTA on 100 tasks × 32 views
  [TRM-Eval] BATCHED: 100 forward passes (B=32 each) instead of 3200
  [TRM-Eval] Task 10/100 (10%)
  [TRM-Eval] Task 20/100 (20%)
  [TRM-Eval] Task 30/100 (30%)
  [TRM-Eval] Task 40/100 (40%)
  [TRM-Eval] Task 50/100 (50%)
  [TRM-Eval] Task 60/100 (60%)
  [TRM-Eval] Task 70/100 (70%)
  [TRM-Eval] Task 80/100 (80%)
  [TRM-Eval] Task 90/100 (90%)
  [TRM-Eval] Task 100/100 (100%)
  [TRM-Eval] Complete. Exact match: 0/100 (0.0%)
  [TRM-Eval] ⚠️ Shape mismatch info: 54/100 tasks had train output shape != test output shape
  [TRM-Eval] ℹ️ Vote ties: 97/100 tasks had multiple predictions with same vote count

  --- TRM-Style TTA Evaluation (8 dihedral x 4 color = 32 views) ---
  ★ TTA Exact Match (Pass@1): 0/100 (0.0%)
  ⏱️ TTA eval time: 225.9s (2.26s/task)
  Pass@K: Pass@1: 0.0% | Pass@2: 0.0% | Pass@3: 0.0%
  Avg Unique Predictions: 31.8 / 32
  Avg Winner Votes: 1.1 / 32

  --- Generalization Health (SINGLE-SHOT) ---
  Train Tasks (first-sample): 1/250 (0.4%)
  Eval Tasks (TTA): 0/100 (0.0%)
  Train→Eval Gap: 0.4% [true single-shot comparison]
  (Any-sample train: 0.4% | Sample-level: 0.4%)
  ✅ Healthy gap: 0.4% - Good generalization!
  🚨 CRITICAL CONSENSUS: 3% < 15% - Equivariance broken!
  --- Evaluation Metrics (Valid Pixels Only) ---
  ★ EXACT MATCH: 0/400 tasks (0.0%)
  Pixel Accuracy: 0.2255
  FG Accuracy (colors 1-9): 0.0606
  BG Accuracy (black): 0.3725
  Class Ratios (pred/target):
    BG (black): 29.0% / 52.9%
    FG (colors): 71.0% / 47.1%
  Colors Used (pred/target): 10 / 10
  DSC Entropy: 5.1266 (lower=sharper)
  DSC Clues Used: 5.20
  Eval Stop Prob: 0.258
  Predicate Activation: 0.0000
  Eval Temperature: 0.969 (matched to training)

  --- Train vs Eval Entropy Delta ---
  Train DSC Entropy: 4.8099
  Eval DSC Entropy:  5.1266
  Delta (Eval - Train): +0.3167
  Ratio (Eval / Train): 1.07x
  ✅ Healthy: Train-eval entropy aligned (1.07x)
  HPM Instance Buffer: 0 entries (empty, not saved)
  HPM Procedural Buffer: 0 entries (empty, not saved)
  Saved checkpoint to checkpoints\rlan_stable_merged\epoch_10.pt
  HPM Instance Buffer: 0 entries (empty, not saved)
  HPM Procedural Buffer: 0 entries (empty, not saved)
  Saved checkpoint to checkpoints\rlan_stable_merged\latest.pt
  [ARCDataset] Augmentation config updated:
    dihedral=True, color_perm=False (prob=0.50), translational=True

============================================================
  PHASE B ACTIVE (epoch 11)
============================================================
    HyperLoRA: DISABLED by phase
    LOO Loss: DISABLED by phase
    Equivariance: DISABLED by phase
    HPM: DISABLED by phase
    Augmentation: rot=True, flip=True, trans=True, color=False
============================================================

  [Phase Change] Recreating DataLoader to propagate augmentation config to workers...
WARNING: use_merged_training=true but manifest not found: data\merged_training\merged_train_manifest.jsonl
         Run: python scripts/build_merged_training_set.py
         Falling back to standard train_path
  [WARNING] use_merged_training=True but manifest not found, falling back to train_path
  [Auto Compute] num_cached_samples = 400 tasks × 50 = 20000
Loaded 400 tasks from ./data/arc-agi/data/training
  Using BUCKETED BATCHING (groups samples by grid size)
    Bucket boundaries: [10, 15, 20, 25] → 5 buckets
[BucketedBatchSampler] Building 5 buckets for 400 samples (metadata-only)...
  [OK] All 400 samples sized via metadata (no __getitem__ calls)
  Bucket 0 (grid <=10): 178 samples
  Bucket 1 (grid <=15): 88 samples
  Bucket 2 (grid <=20): 76 samples
  Bucket 3 (grid <=25): 28 samples
  Bucket 4 (grid >25): 30 samples
  [ARCDataset] Augmentation config updated:
    dihedral=True, color_perm=False (prob=0.50), translational=True
  [Phase Change] DataLoader recreated with 5 batches
  [Phase Override] HyperLoRA deactivated for phase B

Epoch 11/200
----------------------------------------
  [Batch 0] Grid size: 30x30 (batch_max_size=30)

  [MEMORY] Epoch 10 Batch 0:
    Baseline: alloc=92MB, reserved=290MB
    01_batch_on_gpu: alloc=106MB (+14MB), reserved=312MB
    02_after_forward: alloc=23332MB (+23226MB), reserved=24396MB
    03_before_backward: alloc=23378MB (+46MB), reserved=24436MB
    04_after_backward: alloc=239MB (-23139MB), reserved=25140MB
    PEAK: alloc=23976MB, reserved=25140MB / 24576MB (102.3%)

    [BREAKDOWN] Largest increase at '02_after_forward' (+23226MB):
      Model params: 73.4MB
      Model grads:  0.0MB
      Optimizer:    146.7MB (estimate (Adam-like: AdamW))
      Batch data:   6.2MB
      Activations:  77.8MB
      Output tensors (top 5):
        features: 43.9MB
        support_features: 25.0MB
        all_logits: 6.0MB
        attention_maps: 1.2MB
        logits: 0.9MB
      Module params (top 5):
        solver: 28.4MB
        hyper_lora: 19.1MB
        context_encoder: 15.0MB
        context_injector: 3.0MB
        hpm: 2.9MB
      Active modules: solver_context, cross_attention, hpm
      Inactive modules: hyperlora, loo, equivariance

  [MEMORY] First forward pass with staged modules active:
  [MEMORY] Batch 0: alloc=239MB, reserved=25140MB, headroom=-564MB
  [WARNING] >95% GPU memory used! Training may slow due to shared memory.
  Batch 0/5: loss=0.4395, focal_weighted=0.2892, batch_acc=13.3%, exact=0/50 (0.0%), running_acc=13.3%, lr=5.00e-04, hpm=0.0050
    [TaskTrack] Global_Solved: 1/342, Epoch_Solved: 0, New_Puzzles: 0
    FG: batch=6.3% run50=6.3% | BG: batch=16.0% run50=16.0%
    Per-Color: [0:23% 1:5% 2:1% 3:6% 4:5% 5:11% 6:8% 7:47% 8:3% 9:9%]
    Running50: [0:23% 1:5% 2:1% 3:6% 4:5% 5:11% 6:8% 7:47% 8:3% 9:9%]
    Solver: [2.391, 2.406, 2.406, 2.391, 2.391, 2.391, 2.375] ✓
  [Batch 1] Grid size: 30x30 (batch_max_size=30)

  [MEMORY] Epoch 10 Batch 1:
    Baseline: alloc=92MB, reserved=290MB
    01_batch_on_gpu: alloc=220MB (+127MB), reserved=25142MB
    02_after_forward: alloc=26575MB (+26356MB), reserved=29542MB
    03_before_backward: alloc=26602MB (+27MB), reserved=29544MB
    04_after_backward: alloc=300MB (-26302MB), reserved=30424MB
    PEAK: alloc=27420MB, reserved=30424MB / 24576MB (123.8%)

    [BREAKDOWN] Largest increase at '02_after_forward' (+26356MB):
      Model params: 73.4MB
      Model grads:  52.0MB
      Optimizer:    146.7MB (estimate (Adam-like: AdamW))
      Batch data:   7.6MB
      Activations:  84.1MB
      Output tensors (top 5):
        features: 43.9MB
        support_features: 31.2MB
        all_logits: 6.0MB
        attention_maps: 1.2MB
        logits: 0.9MB
      Module params (top 5):
        solver: 28.4MB
        hyper_lora: 19.1MB
        context_encoder: 15.0MB
        context_injector: 3.0MB
        hpm: 2.9MB
      Active modules: solver_context, cross_attention, hpm
      Inactive modules: hyperlora, loo, equivariance
  [MEMORY] Batch 1: alloc=300MB, reserved=30424MB, headroom=-5848MB
  [WARNING] >95% GPU memory used! Training may slow due to shared memory.
  Batch 1/5: loss=0.2742, focal_weighted=0.1371, batch_acc=12.5%, exact=1/100 (1.0%), running_acc=12.9%, lr=5.00e-04, hpm=0.0051
    [TaskTrack] Global_Solved: 2/342, Epoch_Solved: 1, New_Puzzles: 1
    FG: batch=9.4% run50=7.9% | BG: batch=12.7% run50=14.3%
    Per-Color: [0:18% 1:7% 2:0% 3:9% 4:11% 5:11% 6:12% 7:26% 8:1% 9:44%]
    Running50: [0:20% 1:6% 2:0% 3:8% 4:8% 5:11% 6:10% 7:36% 8:2% 9:27%]
  [Batch 2] Grid size: 30x30 (batch_max_size=30)

  [MEMORY] Epoch 10 Batch 2:
    Baseline: alloc=92MB, reserved=290MB
    01_batch_on_gpu: alloc=253MB (+160MB), reserved=30424MB
    02_after_forward: alloc=18758MB (+18505MB), reserved=30424MB
    03_before_backward: alloc=18833MB (+75MB), reserved=30478MB
    04_after_backward: alloc=315MB (-18517MB), reserved=30478MB
    PEAK: alloc=19132MB, reserved=30478MB / 24576MB (124.0%)

    [BREAKDOWN] Largest increase at '02_after_forward' (+18505MB):
      Model params: 73.4MB
      Model grads:  52.0MB
      Optimizer:    146.7MB (estimate (Adam-like: AdamW))
      Batch data:   4.1MB
      Activations:  68.3MB
      Output tensors (top 5):
        features: 43.9MB
        support_features: 15.6MB
        all_logits: 6.0MB
        attention_maps: 1.2MB
        logits: 0.9MB
      Module params (top 5):
        solver: 28.4MB
        hyper_lora: 19.1MB
        context_encoder: 15.0MB
        context_injector: 3.0MB
        hpm: 2.9MB
      Active modules: solver_context, cross_attention, hpm
      Inactive modules: hyperlora, loo, equivariance
  [MEMORY] Batch 2: alloc=315MB, reserved=30478MB, headroom=-5902MB
  [WARNING] >95% GPU memory used! Training may slow due to shared memory.
  Batch 2/5: loss=0.8415, focal_weighted=0.6764, batch_acc=15.4%, exact=2/150 (1.3%), running_acc=13.7%, lr=5.00e-04, hpm=0.0050
    [TaskTrack] Global_Solved: 3/342, Epoch_Solved: 2, New_Puzzles: 2
    FG: batch=8.8% run50=8.2% | BG: batch=20.2% run50=16.3%
    Per-Color: [0:24% 1:3% 2:0% 3:5% 4:10% 5:15% 6:11% 7:11% 8:9% 9:21%]
    Running50: [0:21% 1:5% 2:0% 3:7% 4:9% 5:12% 6:10% 7:28% 8:4% 9:25%]
  [Batch 3] Grid size: 30x30 (batch_max_size=30)

  [MEMORY] Epoch 10 Batch 3:
    Baseline: alloc=92MB, reserved=290MB
    01_batch_on_gpu: alloc=235MB (+143MB), reserved=30478MB
    02_after_forward: alloc=21881MB (+21645MB), reserved=30478MB
    03_before_backward: alloc=21909MB (+28MB), reserved=30478MB
    04_after_backward: alloc=307MB (-21601MB), reserved=30478MB
    PEAK: alloc=22422MB, reserved=30478MB / 24576MB (124.0%)

    [BREAKDOWN] Largest increase at '02_after_forward' (+21645MB):
      Model params: 73.4MB
      Model grads:  52.0MB
      Optimizer:    146.7MB (estimate (Adam-like: AdamW))
      Batch data:   5.5MB
      Activations:  74.6MB
      Output tensors (top 5):
        features: 43.9MB
        support_features: 21.9MB
        all_logits: 6.0MB
        attention_maps: 1.2MB
        logits: 0.9MB
      Module params (top 5):
        solver: 28.4MB
        hyper_lora: 19.1MB
        context_encoder: 15.0MB
        context_injector: 3.0MB
        hpm: 2.9MB
      Active modules: solver_context, cross_attention, hpm
      Inactive modules: hyperlora, loo, equivariance
  Batch 3/5: loss=0.3017, focal_weighted=0.1704, batch_acc=11.3%, exact=2/200 (1.0%), running_acc=13.1%, lr=5.00e-04, hpm=0.0050
    [TaskTrack] Global_Solved: 3/342, Epoch_Solved: 2, New_Puzzles: 2
    FG: batch=7.0% run50=7.9% | BG: batch=14.6% run50=15.9%
    Per-Color: [0:18% 1:4% 2:0% 3:22% 4:9% 5:14% 6:1% 7:28% 8:0% 9:18%]
    Running50: [0:21% 1:5% 2:0% 3:10% 4:9% 5:13% 6:8% 7:28% 8:3% 9:23%]
  [Batch 4] Grid size: 30x30 (batch_max_size=30)

  [MEMORY] Epoch 10 Batch 4:
    Baseline: alloc=92MB, reserved=290MB
    01_batch_on_gpu: alloc=230MB (+138MB), reserved=30478MB
    02_after_forward: alloc=21875MB (+21644MB), reserved=30478MB
    03_before_backward: alloc=21902MB (+27MB), reserved=30478MB
    04_after_backward: alloc=301MB (-21601MB), reserved=30478MB
    PEAK: alloc=22417MB, reserved=30478MB / 24576MB (124.0%)

    [BREAKDOWN] Largest increase at '02_after_forward' (+21644MB):
      Model params: 73.4MB
      Model grads:  52.0MB
      Optimizer:    146.7MB (estimate (Adam-like: AdamW))
      Batch data:   5.5MB
      Activations:  74.6MB
      Output tensors (top 5):
        features: 43.9MB
        support_features: 21.9MB
        all_logits: 6.0MB
        attention_maps: 1.2MB
        logits: 0.9MB
      Module params (top 5):
        solver: 28.4MB
        hyper_lora: 19.1MB
        context_encoder: 15.0MB
        context_injector: 3.0MB
        hpm: 2.9MB
      Active modules: solver_context, cross_attention, hpm
      Inactive modules: hyperlora, loo, equivariance
  Batch 4/5: loss=0.2930, focal_weighted=0.1589, batch_acc=11.1%, exact=2/250 (0.8%), running_acc=12.7%, lr=5.00e-04, hpm=0.0050
    [TaskTrack] Global_Solved: 3/342, Epoch_Solved: 2, New_Puzzles: 2
    FG: batch=7.8% run50=7.9% | BG: batch=10.8% run50=14.8%
    Per-Color: [0:17% 1:10% 2:1% 3:20% 4:5% 5:10% 6:5% 7:16% 8:2% 9:20%]
    Running50: [0:20% 1:6% 2:0% 3:12% 4:8% 5:12% 6:7% 7:25% 8:3% 9:22%]

Epoch 11 Summary:
  Total Loss: 0.4300
  Task Loss (focal): 0.2864
  [Global Task Progress] Unique Solved: 3/342 (0.9%)
    NEW puzzles solved this epoch: 2
  Entropy Loss: 4.2790 (weight=0.01)
  Sparsity Loss: 0.1639 (weight=0.5)
  Predicate Loss: 0.0000 (weight=0.01)
  HPM Balance Loss: 0.0050 (weight=0.01)
  HPM Gate Value: 0.0000 (0=no contribution, 1=full)
  HPM Instance Buffer: 2 entries
  HPM Procedural Buffer: 0 entries
  HPM Tasks Added (exact matches): 2
  Time: 78.5s, LR: 5.00e-04
  HyperLoRA Clamp: hit_rate=0.0% (0/14400), max_norm=0.64 (threshold=1.0)
    Per-module LRs: DSC:5.00e-04, MSRE:5.00e-04, Other:5.00e-04
  Temperature: 0.9659 (lower=sharper attention)
  Samples Processed: 250 (5 batches)
  Dihedral Distribution: [8.8%, 17.2%, 15.6%, 13.6%, 11.6%, 10.4%, 8.0%, 14.8%]
  Color Permutation: 0.0% (0/250)
  Translational Aug: 99.6% (249/250), unique offsets: 238
  Aug Quality: OK
  --- Training Diagnostics ---
  Solver Steps: 7 (deep supervision active)
  Per-Step Loss (epoch avg, 1 batches): [2.3906, 2.4062, 2.4062, 2.3906, 2.3906, 2.3906, 2.3750]
    ✓ Step improvement: 0.7% (later steps better - GOOD!)
  Best-Step Histogram: []
  Solver Health: Last step best: 100.0%, Earlier step best: 0.0%
  Avg Step Improvement: 0.7% (step0→stepN)
  Grad Norms: DSC=0.0000, StopPred=0.0000, Encoder=0.0000, Solver=0.0000
  StopPred Weight Var: 2.80e-02
  Attention: max=0.0196, min=0.000000
  Stop Prob: 0.279 (approx 5.0 clues active)
  Stop Probs Std: 0.088 (global std across batch×clues)
  Clues Used: mean=5.04, std=0.36, range=[4.1, 5.9]
  Clue-Loss Correlation: -0.224 (unexpected negative - check gradient flow)
  Stop Logits: mean=-1.00, std=0.45, range=[-2.6, 0.3]
    Clue count varies by task (per-sample coupling active)
  Per-Clue Entropy: [4.63, 4.63, 4.63, 4.63, 4.63, 4.63, 4.63] (mean=4.63, max=6.80)
    [!] Clues have uniform entropy (std=0.002) - not differentiating!
  Centroid Spread: 0.11 (higher=more diverse)
    🚨 CRITICAL COLLAPSE: Spread=0.11 < 0.5 - all clues at same location!
    [!] Stop predictor cannot differentiate - needs diversity regularizer
  --- Stop Predictor Coupling ---
  Entropy Input to Stop: 0.6806 (normalized, lower=sharper)
  Per-Clue Entropy Input: [0.681, 0.681, 0.681, 0.680, 0.680, 0.681, 0.681]
  Per-Clue Stop Prob: [0.264, 0.295, 0.258, 0.283, 0.305, 0.281, 0.262]
  --- Sparsity Loss Breakdown (Per-Sample Coupled) ---
  Min Clue Penalty: 0.0000 (per-sample avg)
  Per-Sample Clue Penalty (scaled): 0.0000
  Base Pondering: 0.1010 (clues=5.05)
  Entropy Pondering: 0.0693
  --- Per-Class Distribution (Valid Pixels Only) ---
  Padding: 87.9% of grid (ignored in loss)
  Pred %: [20.0, 2.2, 0.9, 8.1, 5.3, 22.3, 1.7, 26.3, 2.4, 10.9]
  Target %: [64.8, 5.2, 5.7, 4.4, 3.6, 4.2, 3.6, 1.1, 6.2, 1.2]
  Per-Class Acc %: [22, 5, 0, 7, 8, 12, 8, 23, 5, 17]
  ==================================================
  PER-SAMPLE TRAINING ACCURACY (Epoch 11)
  ==================================================
  ★ Mean Accuracy: 12.7%
  ★ Exact Match: 2/250 (0.8%)
  ★ High Acc (≥90%): 2/250 (0.8%)
  FG Accuracy: 7.9%
  BG Accuracy: 14.8%
  Batch Trend: 13.3% → 11.2% (↓ 2.1pp)
  Accuracy Distribution: 0-25%:100%, 25-50%:0%, 50-75%:0%, 75-90%:0%, 90-100%:0%
    [!] Many samples stuck at low accuracy - check data/model!
  Running Window (last 5 batches): 12.7% ± 1.6%

  --- PER-COLOR ACCURACY (10 classes) ---
  Color:       0     1     2     3     4     5     6     7     8     9
  Acc%:      22%    5%    0%    7%    8%   12%    8%   23%    5%   17%
  Target:  59.3%  7.4%  5.4%  7.7%  4.0%  4.7%  2.7%  1.8%  5.6%  1.3%
  Pred:    18.8%  2.3%  0.6%  7.6%  6.8% 22.1%  2.9% 25.9%  2.8% 10.2%
  [!] Weak colors (<50% acc): 0(Black), 1(Blue), 2(Red), 3(Green), 4(Yellow), 5(Gray), 6(Pink), 7(Orange), 8(Cyan), 9(Brown)
  [!] Under-predicting color 1 (Blue): 2.3% vs target 7.4%
  [!] Under-predicting color 2 (Red): 0.6% vs target 5.4%
  ==================================================

  ############################################################
  🚦 TRAINING HEALTH CHECK - Epoch 11
  ############################################################
  ✓ Attention sharpening (0.68 < 0.7)
  ✓ Good confidence-stop coupling (r=0.81)
  ✓ No NaN/Inf issues
  ✓ No color mode collapse
  ⚠ Stop probs nearly uniform (global=0.088, per_clue=0.017)
  ⚠ Loss slowly decreasing (0.311 → 0.286)
  ⚠ Accuracy flat (19.9% → 12.7%) - early epoch
  ✗ Centroid COLLAPSE (0.1 < 0.5)
  --------------------------------------------------------
  RESULT: 4/8 checks passed
  STATUS: 🟠 WARNING
  → Multiple issues detected. Consider intervention soon.

  RECOMMENDED ACTIONS:
  ############################################################
  HPM Instance Buffer: 2 entries SAVED
    → Saved to: checkpoints\rlan_stable_merged\hpm\instance_buffer.pt
  HPM Procedural Buffer: 0 entries (empty, not saved)
  Saved checkpoint to checkpoints\rlan_stable_merged\latest.pt
  [ARCDataset] Augmentation config updated:
    dihedral=True, color_perm=False (prob=0.50), translational=True

Epoch 12/200
----------------------------------------
  [LR] Reduced to 2.50e-04 (×0.5) for stability

============================================================
PHASE 2: CONTEXT PATH ACTIVATED (epoch 12)
============================================================
  SolverCrossAttention: NOW ACTIVE in solver loop
  GPU MEMORY: 146MB / 24576MB (0.6%)
============================================================

  [Batch 0] Grid size: 30x30 (batch_max_size=30)

  [MEMORY] Epoch 11 Batch 0:
    Baseline: alloc=92MB, reserved=30478MB
    01_batch_on_gpu: alloc=106MB (+14MB), reserved=30500MB
    02_after_forward: alloc=26464MB (+26358MB), reserved=30500MB
    03_before_backward: alloc=26489MB (+24MB), reserved=30500MB
    04_after_backward: alloc=246MB (-26243MB), reserved=30500MB
    PEAK: alloc=27309MB, reserved=30500MB / 24576MB (124.1%)

    [BREAKDOWN] Largest increase at '02_after_forward' (+26358MB):
      Model params: 73.4MB
      Model grads:  0.0MB
      Optimizer:    146.7MB (estimate (Adam-like: AdamW))
      Batch data:   7.6MB
      Activations:  84.1MB
      Output tensors (top 5):
        features: 43.9MB
        support_features: 31.2MB
        all_logits: 6.0MB
        attention_maps: 1.2MB
        logits: 0.9MB
      Module params (top 5):
        solver: 28.4MB
        hyper_lora: 19.1MB
        context_encoder: 15.0MB
        context_injector: 3.0MB
        hpm: 2.9MB
      Active modules: solver_context, cross_attention, hpm
      Inactive modules: hyperlora, loo, equivariance

  [MEMORY] First forward pass with staged modules active:
  [MEMORY] Batch 0: alloc=246MB, reserved=30500MB, headroom=-5924MB
  [WARNING] >95% GPU memory used! Training may slow due to shared memory.
  Batch 0/5: loss=0.2601, focal_weighted=0.1261, batch_acc=11.2%, exact=0/50 (0.0%), running_acc=11.2%, lr=2.50e-04, hpm=0.0051
    [TaskTrack] Global_Solved: 3/342, Epoch_Solved: 0, New_Puzzles: 0
    FG: batch=7.6% run50=7.6% | BG: batch=12.6% run50=12.6%
    Per-Color: [0:18% 1:5% 2:0% 3:18% 4:8% 5:20% 6:12% 7:14% 8:6% 9:23%]
    Running50: [0:18% 1:5% 2:0% 3:18% 4:8% 5:20% 6:12% 7:14% 8:6% 9:23%]
    Solver: [2.422, 2.422, 2.422, 2.422, 2.406, 2.406, 2.406] ⚠ best=4
  [Batch 1] Grid size: 30x30 (batch_max_size=30)

  [MEMORY] Epoch 11 Batch 1:
    Baseline: alloc=92MB, reserved=30478MB
    01_batch_on_gpu: alloc=221MB (+128MB), reserved=30502MB
    02_after_forward: alloc=23432MB (+23212MB), reserved=30502MB
    03_before_backward: alloc=23460MB (+28MB), reserved=30502MB
    04_after_backward: alloc=293MB (-23167MB), reserved=30502MB
    PEAK: alloc=24075MB, reserved=30502MB / 24576MB (124.1%)

    [BREAKDOWN] Largest increase at '02_after_forward' (+23212MB):
      Model params: 73.4MB
      Model grads:  52.0MB
      Optimizer:    146.7MB (estimate (Adam-like: AdamW))
      Batch data:   6.2MB
      Activations:  77.8MB
      Output tensors (top 5):
        features: 43.9MB
        support_features: 25.0MB
        all_logits: 6.0MB
        attention_maps: 1.2MB
        logits: 0.9MB
      Module params (top 5):
        solver: 28.4MB
        hyper_lora: 19.1MB
        context_encoder: 15.0MB
        context_injector: 3.0MB
        hpm: 2.9MB
      Active modules: solver_context, cross_attention, hpm
      Inactive modules: hyperlora, loo, equivariance
  [MEMORY] Batch 1: alloc=293MB, reserved=30502MB, headroom=-5926MB
  [WARNING] >95% GPU memory used! Training may slow due to shared memory.
  Batch 1/5: loss=0.2849, focal_weighted=0.1503, batch_acc=9.4%, exact=0/100 (0.0%), running_acc=10.3%, lr=2.50e-04, hpm=0.0050
    [TaskTrack] Global_Solved: 3/342, Epoch_Solved: 0, New_Puzzles: 0
    FG: batch=5.1% run50=6.4% | BG: batch=12.7% run50=12.7%
    Per-Color: [0:18% 1:5% 2:0% 3:8% 4:7% 5:7% 6:3% 7:31% 8:3% 9:0%]
    Running50: [0:18% 1:5% 2:0% 3:13% 4:7% 5:14% 6:7% 7:22% 8:5% 9:11%]
  [Batch 2] Grid size: 30x30 (batch_max_size=30)

  [MEMORY] Epoch 11 Batch 2:
    Baseline: alloc=92MB, reserved=30478MB
    01_batch_on_gpu: alloc=221MB (+128MB), reserved=30502MB
    02_after_forward: alloc=23432MB (+23212MB), reserved=30502MB
    03_before_backward: alloc=23478MB (+46MB), reserved=30502MB
    04_after_backward: alloc=294MB (-23184MB), reserved=30502MB
    PEAK: alloc=24075MB, reserved=30502MB / 24576MB (124.1%)

    [BREAKDOWN] Largest increase at '02_after_forward' (+23212MB):
      Model params: 73.4MB
      Model grads:  52.0MB
      Optimizer:    146.7MB (estimate (Adam-like: AdamW))
      Batch data:   6.2MB
      Activations:  77.8MB
      Output tensors (top 5):
        features: 43.9MB
        support_features: 25.0MB
        all_logits: 6.0MB
        attention_maps: 1.2MB
        logits: 0.9MB
      Module params (top 5):
        solver: 28.4MB
        hyper_lora: 19.1MB
        context_encoder: 15.0MB
        context_injector: 3.0MB
        hpm: 2.9MB
      Active modules: solver_context, cross_attention, hpm
      Inactive modules: hyperlora, loo, equivariance
  [MEMORY] Batch 2: alloc=294MB, reserved=30502MB, headroom=-5926MB
  [WARNING] >95% GPU memory used! Training may slow due to shared memory.
  Batch 2/5: loss=0.4484, focal_weighted=0.2932, batch_acc=13.1%, exact=0/150 (0.0%), running_acc=11.2%, lr=2.50e-04, hpm=0.0050
    [TaskTrack] Global_Solved: 3/342, Epoch_Solved: 0, New_Puzzles: 0
    FG: batch=5.5% run50=6.1% | BG: batch=17.4% run50=14.2%
    Per-Color: [0:23% 1:1% 2:1% 3:6% 4:8% 5:10% 6:6% 7:15% 8:6% 9:8%]
    Running50: [0:20% 1:4% 2:0% 3:11% 4:8% 5:12% 6:7% 7:20% 8:5% 9:10%]
  [Batch 3] Grid size: 30x30 (batch_max_size=30)

  [MEMORY] Epoch 11 Batch 3:
    Baseline: alloc=92MB, reserved=30478MB
    01_batch_on_gpu: alloc=219MB (+126MB), reserved=30502MB
    02_after_forward: alloc=20297MB (+20078MB), reserved=30502MB
    03_before_backward: alloc=20324MB (+27MB), reserved=30502MB
    04_after_backward: alloc=285MB (-20038MB), reserved=30502MB
    PEAK: alloc=20737MB, reserved=30502MB / 24576MB (124.1%)

    [BREAKDOWN] Largest increase at '02_after_forward' (+20078MB):
      Model params: 73.4MB
      Model grads:  52.0MB
      Optimizer:    146.7MB (estimate (Adam-like: AdamW))
      Batch data:   4.8MB
      Activations:  71.5MB
      Output tensors (top 5):
        features: 43.9MB
        support_features: 18.8MB
        all_logits: 6.0MB
        attention_maps: 1.2MB
        logits: 0.9MB
      Module params (top 5):
        solver: 28.4MB
        hyper_lora: 19.1MB
        context_encoder: 15.0MB
        context_injector: 3.0MB
        hpm: 2.9MB
      Active modules: solver_context, cross_attention, hpm
      Inactive modules: hyperlora, loo, equivariance
  Batch 3/5: loss=0.2803, focal_weighted=0.1490, batch_acc=9.7%, exact=0/200 (0.0%), running_acc=10.9%, lr=2.50e-04, hpm=0.0050
    [TaskTrack] Global_Solved: 3/342, Epoch_Solved: 0, New_Puzzles: 0
    FG: batch=6.4% run50=6.1% | BG: batch=10.4% run50=13.3%
    Per-Color: [0:15% 1:3% 2:0% 3:7% 4:9% 5:10% 6:5% 7:25% 8:0% 9:11%]
    Running50: [0:19% 1:4% 2:0% 3:10% 4:8% 5:12% 6:7% 7:21% 8:4% 9:10%]
  [Batch 4] Grid size: 30x30 (batch_max_size=30)

  [MEMORY] Epoch 11 Batch 4:
    Baseline: alloc=92MB, reserved=30478MB
    01_batch_on_gpu: alloc=212MB (+120MB), reserved=30502MB
    02_after_forward: alloc=18721MB (+18509MB), reserved=30502MB
    03_before_backward: alloc=18806MB (+84MB), reserved=30502MB
    04_after_backward: alloc=277MB (-18528MB), reserved=30502MB
    PEAK: alloc=19096MB, reserved=30502MB / 24576MB (124.1%)

    [BREAKDOWN] Largest increase at '02_after_forward' (+18509MB):
      Model params: 73.4MB
      Model grads:  52.0MB
      Optimizer:    146.7MB (estimate (Adam-like: AdamW))
      Batch data:   4.1MB
      Activations:  68.3MB
      Output tensors (top 5):
        features: 43.9MB
        support_features: 15.6MB
        all_logits: 6.0MB
        attention_maps: 1.2MB
        logits: 0.9MB
      Module params (top 5):
        solver: 28.4MB
        hyper_lora: 19.1MB
        context_encoder: 15.0MB
        context_injector: 3.0MB
        hpm: 2.9MB
      Active modules: solver_context, cross_attention, hpm
      Inactive modules: hyperlora, loo, equivariance
  Batch 4/5: loss=0.9184, focal_weighted=0.7546, batch_acc=14.1%, exact=0/250 (0.0%), running_acc=11.5%, lr=2.50e-04, hpm=0.0050
    [TaskTrack] Global_Solved: 3/342, Epoch_Solved: 0, New_Puzzles: 0
    FG: batch=5.5% run50=6.0% | BG: batch=22.1% run50=15.1%
    Per-Color: [0:23% 1:1% 2:2% 3:4% 4:14% 5:14% 6:11% 7:12% 8:7% 9:15%]
    Running50: [0:19% 1:3% 2:1% 3:8% 4:9% 5:12% 6:7% 7:19% 8:4% 9:11%]

Epoch 12 Summary:
  Total Loss: 0.4385
  Task Loss (focal): 0.2946
  [Global Task Progress] Unique Solved: 3/342 (0.9%)
    NEW puzzles solved this epoch: 0
  Entropy Loss: 4.2966 (weight=0.01)
  Sparsity Loss: 0.1640 (weight=0.5)
  Predicate Loss: 0.0000 (weight=0.01)
  HPM Balance Loss: 0.0050 (weight=0.01)
  HPM Gate Value: 0.0000 (0=no contribution, 1=full)
  HPM Instance Buffer: 2 entries
  HPM Procedural Buffer: 0 entries
  HPM Tasks Added (exact matches): 0
    (No exact matches this epoch: 0)
  Time: 38.0s, LR: 2.50e-04
    Per-module LRs: DSC:2.50e-04, MSRE:2.50e-04, Other:2.50e-04
  Temperature: 0.9626 (lower=sharper attention)
  Samples Processed: 250 (5 batches)
  Dihedral Distribution: [13.6%, 10.8%, 13.2%, 13.2%, 12.4%, 13.6%, 8.8%, 14.4%]
  Color Permutation: 0.0% (0/250)
  Translational Aug: 99.6% (249/250), unique offsets: 228
  Aug Quality: OK
  --- Training Diagnostics ---
  Solver Steps: 7 (deep supervision active)
  Per-Step Loss (epoch avg, 1 batches): [2.4219, 2.4219, 2.4219, 2.4219, 2.4062, 2.4062, 2.4062]
    [!] Best step is 4 (middle), not last - solver may be over-iterating!
    [!] Best: step 4 (2.4062), Final: step 6 (2.4062)
    [!] Potential gain from best-step selection: 0.0% lower loss
  Best-Step Histogram: []
  Solver Health: Last step best: 0.0%, Earlier step best: 100.0%
  Avg Step Improvement: 0.6% (step0→stepN)
    ⚠️ SOLVER OVER-ITERATION WARNING: 100% of batches had earlier step as best!
    ⚠️ Consider: (1) enabling best-step selection, (2) reducing num_solver_steps, (3) enabling ACT
  Grad Norms: DSC=0.0000, StopPred=0.0000, Encoder=0.0000, Solver=0.0000
  StopPred Weight Var: 2.80e-02
  Attention: max=0.0526, min=0.000000
  Stop Prob: 0.271 (approx 5.1 clues active)
  Stop Probs Std: 0.102 (global std across batch×clues)
  Clues Used: mean=5.10, std=0.44, range=[3.9, 6.0]
  Clue-Loss Correlation: -0.310 (unexpected negative - check gradient flow)
  Stop Logits: mean=-1.05, std=0.54, range=[-2.8, 0.7]
    Clue count varies by task (per-sample coupling active)
  Per-Clue Entropy: [3.65, 3.65, 3.65, 3.65, 3.65, 3.65, 3.65] (mean=3.65, max=6.80)
    [!] Clues have uniform entropy (std=0.001) - not differentiating!
  Centroid Spread: 0.06 (higher=more diverse)
    🚨 CRITICAL COLLAPSE: Spread=0.06 < 0.5 - all clues at same location!
    [!] Stop predictor cannot differentiate - needs diversity regularizer
  --- Stop Predictor Coupling ---
  Entropy Input to Stop: 0.5366 (normalized, lower=sharper)
  Per-Clue Entropy Input: [0.537, 0.536, 0.536, 0.536, 0.536, 0.537, 0.537]
  Per-Clue Stop Prob: [0.262, 0.256, 0.279, 0.281, 0.277, 0.273, 0.273]
  --- Sparsity Loss Breakdown (Per-Sample Coupled) ---
  Min Clue Penalty: 0.0000 (per-sample avg)
  Per-Sample Clue Penalty (scaled): 0.0000
  Base Pondering: 0.1020 (clues=5.10)
  Entropy Pondering: 0.0552
  --- Per-Class Distribution (Valid Pixels Only) ---
  Padding: 94.8% of grid (ignored in loss)
  Pred %: [16.2, 3.4, 0.7, 8.9, 7.9, 20.1, 1.6, 22.9, 3.7, 14.7]
  Target %: [53.0, 6.8, 10.2, 4.9, 9.1, 5.8, 0.7, 3.0, 4.6, 1.9]
  Per-Class Acc %: [21, 3, 1, 6, 10, 13, 8, 16, 6, 15]
  ==================================================
  PER-SAMPLE TRAINING ACCURACY (Epoch 12)
  ==================================================
  ★ Mean Accuracy: 11.5%
  ★ Exact Match: 0/250 (0.0%)
  ★ High Acc (≥90%): 0/250 (0.0%)
  FG Accuracy: 6.0%
  BG Accuracy: 15.1%
  Batch Trend: 11.2% → 11.9% (→ 0.6pp)
  Accuracy Distribution: 0-25%:100%, 25-50%:0%, 50-75%:0%, 75-90%:0%, 90-100%:0%
    [!] Many samples stuck at low accuracy - check data/model!
  Running Window (last 5 batches): 11.5% ± 1.8%

  --- PER-COLOR ACCURACY (10 classes) ---
  Color:       0     1     2     3     4     5     6     7     8     9
  Acc%:      21%    3%    1%    6%   10%   13%    8%   16%    6%   15%
  Target:  55.8%  7.2%  6.1%  8.0%  4.4%  5.1%  2.9%  2.5%  6.7%  1.3%
  Pred:    17.8%  2.2%  0.7%  7.5%  7.7% 21.8%  2.9% 26.1%  3.3%  9.9%
  [!] Weak colors (<50% acc): 0(Black), 1(Blue), 2(Red), 3(Green), 4(Yellow), 5(Gray), 6(Pink), 7(Orange), 8(Cyan), 9(Brown)
  [!] Under-predicting color 1 (Blue): 2.2% vs target 7.2%
  [!] Under-predicting color 2 (Red): 0.7% vs target 6.1%
  [!] Under-predicting color 8 (Cyan): 3.3% vs target 6.7%
  ==================================================

  ############################################################
  🚦 TRAINING HEALTH CHECK - Epoch 12
  ############################################################
  ✓ Attention sharpening (0.54 < 0.7)
  ✓ Stop probs adapting (global=0.102, per_clue=0.009)
  ✓ No NaN/Inf issues
  ✓ No color mode collapse
  ⚠ Negative coupling (r=-0.03) - early epoch OK
  ⚠ Loss slowly decreasing (0.311 → 0.295)
  ✗ Centroid COLLAPSE (0.1 < 0.5)
  ✗ Accuracy not improving (19.9% → 11.5%)
  --------------------------------------------------------
  RESULT: 4/8 checks passed
  STATUS: 🟠 WARNING
  → Multiple issues detected. Consider intervention soon.

  RECOMMENDED ACTIONS:
  ############################################################
  HPM Instance Buffer: 2 entries SAVED
    → Saved to: checkpoints\rlan_stable_merged\hpm\instance_buffer.pt
  HPM Procedural Buffer: 0 entries (empty, not saved)
  Saved checkpoint to checkpoints\rlan_stable_merged\latest.pt
  [ARCDataset] Augmentation config updated:
    dihedral=True, color_perm=False (prob=0.50), translational=True

Epoch 13/200
----------------------------------------
  [Batch 0] Grid size: 30x30 (batch_max_size=30)

  [MEMORY] Epoch 12 Batch 0:
    Baseline: alloc=92MB, reserved=30502MB
    01_batch_on_gpu: alloc=102MB (+10MB), reserved=30524MB
    02_after_forward: alloc=21751MB (+21648MB), reserved=30524MB
    03_before_backward: alloc=21779MB (+28MB), reserved=30524MB
    04_after_backward: alloc=231MB (-21547MB), reserved=30524MB
    PEAK: alloc=22293MB, reserved=30524MB / 24576MB (124.2%)

    [BREAKDOWN] Largest increase at '02_after_forward' (+21648MB):
      Model params: 73.4MB
      Model grads:  0.0MB
      Optimizer:    146.7MB (estimate (Adam-like: AdamW))
      Batch data:   5.5MB
      Activations:  74.6MB
      Output tensors (top 5):
        features: 43.9MB
        support_features: 21.9MB
        all_logits: 6.0MB
        attention_maps: 1.2MB
        logits: 0.9MB
      Module params (top 5):
        solver: 28.4MB
        hyper_lora: 19.1MB
        context_encoder: 15.0MB
        context_injector: 3.0MB
        hpm: 2.9MB
      Active modules: solver_context, cross_attention, hpm
      Inactive modules: hyperlora, loo, equivariance

  [MEMORY] First forward pass with staged modules active:
  [MEMORY] Batch 0: alloc=231MB, reserved=30524MB, headroom=-5948MB
  [WARNING] >95% GPU memory used! Training may slow due to shared memory.
  Batch 0/5: loss=0.2874, focal_weighted=0.1553, batch_acc=10.0%, exact=0/50 (0.0%), running_acc=10.0%, lr=2.50e-04, hpm=0.0051
    [TaskTrack] Global_Solved: 3/342, Epoch_Solved: 0, New_Puzzles: 0
    FG: batch=8.0% run50=8.0% | BG: batch=11.4% run50=11.4%
    Per-Color: [0:15% 1:4% 2:0% 3:14% 4:6% 5:6% 6:4% 7:27% 8:6% 9:27%]
    Running50: [0:15% 1:4% 2:0% 3:14% 4:6% 5:6% 6:4% 7:27% 8:6% 9:27%]
    Solver: [2.516, 2.516, 2.500, 2.500, 2.500, 2.484, 2.484] ⚠ best=5
  [Batch 1] Grid size: 30x30 (batch_max_size=30)

  [MEMORY] Epoch 12 Batch 1:
    Baseline: alloc=92MB, reserved=30502MB
    01_batch_on_gpu: alloc=215MB (+123MB), reserved=30526MB
    02_after_forward: alloc=18722MB (+18507MB), reserved=30526MB
    03_before_backward: alloc=18803MB (+81MB), reserved=30526MB
    04_after_backward: alloc=280MB (-18524MB), reserved=30526MB
    PEAK: alloc=19098MB, reserved=30526MB / 24576MB (124.2%)

    [BREAKDOWN] Largest increase at '02_after_forward' (+18507MB):
      Model params: 73.4MB
      Model grads:  52.0MB
      Optimizer:    146.7MB (estimate (Adam-like: AdamW))
      Batch data:   4.1MB
      Activations:  68.3MB
      Output tensors (top 5):
        features: 43.9MB
        support_features: 15.6MB
        all_logits: 6.0MB
        attention_maps: 1.2MB
        logits: 0.9MB
      Module params (top 5):
        solver: 28.4MB
        hyper_lora: 19.1MB
        context_encoder: 15.0MB
        context_injector: 3.0MB
        hpm: 2.9MB
      Active modules: solver_context, cross_attention, hpm
      Inactive modules: hyperlora, loo, equivariance
  [MEMORY] Batch 1: alloc=280MB, reserved=30526MB, headroom=-5950MB
  [WARNING] >95% GPU memory used! Training may slow due to shared memory.
  Batch 1/5: loss=1.0142, focal_weighted=0.8489, batch_acc=12.2%, exact=0/100 (0.0%), running_acc=11.1%, lr=2.50e-04, hpm=0.0051
    [TaskTrack] Global_Solved: 3/342, Epoch_Solved: 0, New_Puzzles: 0
    FG: batch=6.7% run50=7.3% | BG: batch=16.9% run50=14.2%
    Per-Color: [0:21% 1:2% 2:1% 3:6% 4:8% 5:12% 6:10% 7:8% 8:9% 9:13%]
    Running50: [0:18% 1:3% 2:1% 3:10% 4:7% 5:9% 6:7% 7:17% 8:7% 9:20%]
  [Batch 2] Grid size: 30x30 (batch_max_size=30)

  [MEMORY] Epoch 12 Batch 2:
    Baseline: alloc=92MB, reserved=30502MB
    01_batch_on_gpu: alloc=219MB (+126MB), reserved=30526MB
    02_after_forward: alloc=20298MB (+20079MB), reserved=30526MB
    03_before_backward: alloc=20324MB (+26MB), reserved=30526MB
    04_after_backward: alloc=287MB (-20037MB), reserved=30526MB
    PEAK: alloc=20739MB, reserved=30526MB / 24576MB (124.2%)

    [BREAKDOWN] Largest increase at '02_after_forward' (+20079MB):
      Model params: 73.4MB
      Model grads:  52.0MB
      Optimizer:    146.7MB (estimate (Adam-like: AdamW))
      Batch data:   4.8MB
      Activations:  71.5MB
      Output tensors (top 5):
        features: 43.9MB
        support_features: 18.8MB
        all_logits: 6.0MB
        attention_maps: 1.2MB
        logits: 0.9MB
      Module params (top 5):
        solver: 28.4MB
        hyper_lora: 19.1MB
        context_encoder: 15.0MB
        context_injector: 3.0MB
        hpm: 2.9MB
      Active modules: solver_context, cross_attention, hpm
      Inactive modules: hyperlora, loo, equivariance
  [MEMORY] Batch 2: alloc=287MB, reserved=30526MB, headroom=-5950MB
  [WARNING] >95% GPU memory used! Training may slow due to shared memory.
  Batch 2/5: loss=0.2946, focal_weighted=0.1677, batch_acc=10.5%, exact=0/150 (0.0%), running_acc=10.9%, lr=2.50e-04, hpm=0.0051
    [TaskTrack] Global_Solved: 3/342, Epoch_Solved: 0, New_Puzzles: 0
    FG: batch=7.6% run50=7.4% | BG: batch=10.5% run50=12.9%
    Per-Color: [0:15% 1:7% 2:2% 3:8% 4:8% 5:12% 6:7% 7:15% 8:8% 9:19%]
    Running50: [0:17% 1:4% 2:1% 3:9% 4:8% 5:10% 6:7% 7:17% 8:7% 9:20%]
  [Batch 3] Grid size: 30x30 (batch_max_size=30)

  [MEMORY] Epoch 12 Batch 3:
    Baseline: alloc=92MB, reserved=30502MB
    01_batch_on_gpu: alloc=221MB (+129MB), reserved=30526MB
    02_after_forward: alloc=23438MB (+23217MB), reserved=30526MB
    03_before_backward: alloc=23482MB (+44MB), reserved=30526MB
    04_after_backward: alloc=296MB (-23186MB), reserved=30526MB
    PEAK: alloc=24081MB, reserved=30526MB / 24576MB (124.2%)

    [BREAKDOWN] Largest increase at '02_after_forward' (+23217MB):
      Model params: 73.4MB
      Model grads:  52.0MB
      Optimizer:    146.7MB (estimate (Adam-like: AdamW))
      Batch data:   6.2MB
      Activations:  77.8MB
      Output tensors (top 5):
        features: 43.9MB
        support_features: 25.0MB
        all_logits: 6.0MB
        attention_maps: 1.2MB
        logits: 0.9MB
      Module params (top 5):
        solver: 28.4MB
        hyper_lora: 19.1MB
        context_encoder: 15.0MB
        context_injector: 3.0MB
        hpm: 2.9MB
      Active modules: solver_context, cross_attention, hpm
      Inactive modules: hyperlora, loo, equivariance
  Batch 3/5: loss=0.4570, focal_weighted=0.3065, batch_acc=14.4%, exact=0/200 (0.0%), running_acc=11.8%, lr=2.50e-04, hpm=0.0051
    [TaskTrack] Global_Solved: 3/342, Epoch_Solved: 0, New_Puzzles: 0
    FG: batch=6.6% run50=7.2% | BG: batch=17.6% run50=14.1%
    Per-Color: [0:22% 1:7% 2:1% 3:5% 4:2% 5:7% 6:8% 7:25% 8:4% 9:9%]
    Running50: [0:18% 1:5% 2:1% 3:8% 4:6% 5:9% 6:7% 7:19% 8:6% 9:17%]
  [Batch 4] Grid size: 30x30 (batch_max_size=30)

  [MEMORY] Epoch 12 Batch 4:
    Baseline: alloc=92MB, reserved=30502MB
    01_batch_on_gpu: alloc=217MB (+124MB), reserved=30526MB
    02_after_forward: alloc=26572MB (+26355MB), reserved=30526MB
    03_before_backward: alloc=26598MB (+26MB), reserved=30526MB
    04_after_backward: alloc=297MB (-26301MB), reserved=30526MB
    PEAK: alloc=27417MB, reserved=30526MB / 24576MB (124.2%)

    [BREAKDOWN] Largest increase at '02_after_forward' (+26355MB):
      Model params: 73.4MB
      Model grads:  52.0MB
      Optimizer:    146.7MB (estimate (Adam-like: AdamW))
      Batch data:   7.6MB
      Activations:  84.1MB
      Output tensors (top 5):
        features: 43.9MB
        support_features: 31.2MB
        all_logits: 6.0MB
        attention_maps: 1.2MB
        logits: 0.9MB
      Module params (top 5):
        solver: 28.4MB
        hyper_lora: 19.1MB
        context_encoder: 15.0MB
        context_injector: 3.0MB
        hpm: 2.9MB
      Active modules: solver_context, cross_attention, hpm
      Inactive modules: hyperlora, loo, equivariance
  Batch 4/5: loss=0.2732, focal_weighted=0.1333, batch_acc=7.8%, exact=0/250 (0.0%), running_acc=11.0%, lr=2.50e-04, hpm=0.0050
    [TaskTrack] Global_Solved: 3/342, Epoch_Solved: 0, New_Puzzles: 0
    FG: batch=5.1% run50=6.8% | BG: batch=9.8% run50=13.2%
    Per-Color: [0:16% 1:5% 2:0% 3:16% 4:10% 5:10% 6:3% 7:4% 8:2% 9:30%]
    Running50: [0:18% 1:5% 2:1% 3:10% 4:7% 5:9% 6:6% 7:16% 8:5% 9:20%]

Epoch 13 Summary:
  Total Loss: 0.4653
  Task Loss (focal): 0.3223
  [Global Task Progress] Unique Solved: 3/342 (0.9%)
    NEW puzzles solved this epoch: 0
  Entropy Loss: 4.2794 (weight=0.01)
  Sparsity Loss: 0.1626 (weight=0.5)
  Predicate Loss: 0.0000 (weight=0.01)
  HPM Balance Loss: 0.0051 (weight=0.01)
  HPM Gate Value: 0.0000 (0=no contribution, 1=full)
  HPM Instance Buffer: 2 entries
  HPM Procedural Buffer: 0 entries
  HPM Tasks Added (exact matches): 0
    (No exact matches this epoch: 0)
  Time: 37.3s, LR: 2.50e-04
    Per-module LRs: DSC:2.50e-04, MSRE:2.50e-04, Other:2.50e-04
  Temperature: 0.9593 (lower=sharper attention)
  Samples Processed: 250 (5 batches)
  Dihedral Distribution: [12.0%, 13.2%, 15.6%, 14.4%, 12.4%, 12.8%, 8.8%, 10.8%]
  Color Permutation: 0.0% (0/250)
  Translational Aug: 100.0% (250/250), unique offsets: 236
  Aug Quality: OK
  --- Training Diagnostics ---
  Solver Steps: 7 (deep supervision active)
  Per-Step Loss (epoch avg, 1 batches): [2.5156, 2.5156, 2.5000, 2.5000, 2.5000, 2.4844, 2.4844]
    [!] Best step is 5 (middle), not last - solver may be over-iterating!
    [!] Best: step 5 (2.4844), Final: step 6 (2.4844)
    [!] Potential gain from best-step selection: 0.0% lower loss
  Best-Step Histogram: []
  Solver Health: Last step best: 0.0%, Earlier step best: 100.0%
  Avg Step Improvement: 1.2% (step0→stepN)
    ⚠️ SOLVER OVER-ITERATION WARNING: 100% of batches had earlier step as best!
    ⚠️ Consider: (1) enabling best-step selection, (2) reducing num_solver_steps, (3) enabling ACT
  Grad Norms: DSC=0.0000, StopPred=0.0000, Encoder=0.0000, Solver=0.0000
  StopPred Weight Var: 2.80e-02
  Attention: max=0.0450, min=0.000000
  Stop Prob: 0.314 (approx 4.8 clues active)
  Stop Probs Std: 0.121 (global std across batch×clues)
  Clues Used: mean=4.80, std=0.59, range=[2.9, 6.1]
  Clue-Loss Correlation: -0.114 (unexpected negative - check gradient flow)
  Stop Logits: mean=-0.84, std=0.59, range=[-2.8, 0.9]
    [+] High variance - strong per-task clue adaptation!
  Per-Clue Entropy: [3.79, 3.79, 3.79, 3.79, 3.79, 3.79, 3.79] (mean=3.79, max=6.80)
    [!] Clues have uniform entropy (std=0.001) - not differentiating!
  Centroid Spread: 0.06 (higher=more diverse)
    🚨 CRITICAL COLLAPSE: Spread=0.06 < 0.5 - all clues at same location!
    [!] Stop predictor cannot differentiate - needs diversity regularizer
  --- Stop Predictor Coupling ---
  Entropy Input to Stop: 0.5574 (normalized, lower=sharper)
  Per-Clue Entropy Input: [0.558, 0.557, 0.557, 0.557, 0.557, 0.557, 0.558]
  Per-Clue Stop Prob: [0.322, 0.311, 0.332, 0.312, 0.316, 0.283, 0.320]
  --- Sparsity Loss Breakdown (Per-Sample Coupled) ---
  Min Clue Penalty: 0.0000 (per-sample avg)
  Per-Sample Clue Penalty (scaled): 0.0000
  Base Pondering: 0.0961 (clues=4.80)
  Entropy Pondering: 0.0543
  --- Per-Class Distribution (Valid Pixels Only) ---
  Padding: 93.8% of grid (ignored in loss)
  Pred %: [14.0, 3.3, 0.5, 10.4, 7.4, 20.9, 1.1, 26.5, 3.2, 12.7]
  Target %: [59.8, 9.6, 8.3, 4.3, 2.8, 6.2, 1.8, 2.7, 3.8, 0.8]
    [!] Missing foreground colors: [2]
  Per-Class Acc %: [19, 4, 1, 8, 7, 10, 8, 16, 6, 14]
  ==================================================
  PER-SAMPLE TRAINING ACCURACY (Epoch 13)
  ==================================================
  ★ Mean Accuracy: 11.0%
  ★ Exact Match: 0/250 (0.0%)
  ★ High Acc (≥90%): 0/250 (0.0%)
  FG Accuracy: 6.8%
  BG Accuracy: 13.2%
  Batch Trend: 10.0% → 11.1% (↑ 1.1pp)
  Accuracy Distribution: 0-25%:100%, 25-50%:0%, 50-75%:0%, 75-90%:0%, 90-100%:0%
    [!] Many samples stuck at low accuracy - check data/model!
  Running Window (last 5 batches): 11.0% ± 2.2%

  --- PER-COLOR ACCURACY (10 classes) ---
  Color:       0     1     2     3     4     5     6     7     8     9
  Acc%:      19%    4%    1%    8%    7%   10%    8%   16%    6%   14%
  Target:  56.1%  8.1%  6.3%  6.2%  3.7%  6.1%  2.9%  2.6%  5.5%  2.5%
  Pred:    17.0%  2.5%  0.7%  8.5%  7.2% 21.4%  3.0% 26.2%  3.2% 10.4%
  [!] Weak colors (<50% acc): 0(Black), 1(Blue), 2(Red), 3(Green), 4(Yellow), 5(Gray), 6(Pink), 7(Orange), 8(Cyan), 9(Brown)
  [!] Under-predicting color 1 (Blue): 2.5% vs target 8.1%
  [!] Under-predicting color 2 (Red): 0.7% vs target 6.3%
  ==================================================

  ############################################################
  🚦 TRAINING HEALTH CHECK - Epoch 13
  ############################################################
  ✓ Attention sharpening (0.56 < 0.7)
  ✓ Stop probs adapting (global=0.121, per_clue=0.014)
  ✓ No NaN/Inf issues
  ✓ No color mode collapse
  ⚠ Negative coupling (r=-0.03) - early epoch OK
  ✗ Centroid COLLAPSE (0.1 < 0.5)
  ✗ Loss not decreasing (0.311 → 0.322)
  ✗ Accuracy not improving (19.9% → 11.0%)
  --------------------------------------------------------
  RESULT: 4/8 checks passed
  STATUS: 🔴 UNHEALTHY
  → Training likely failing. STOP and investigate!

  RECOMMENDED ACTIONS:
    - Reduce learning rate or check data pipeline
  ############################################################
  HPM Instance Buffer: 2 entries SAVED
    → Saved to: checkpoints\rlan_stable_merged\hpm\instance_buffer.pt
  HPM Procedural Buffer: 0 entries (empty, not saved)
  Saved checkpoint to checkpoints\rlan_stable_merged\latest.pt
  [ARCDataset] Augmentation config updated:
    dihedral=True, color_perm=False (prob=0.50), translational=True

Epoch 14/200
----------------------------------------
  [LR] Restored to original after 2 epochs recovery
  [Batch 0] Grid size: 30x30 (batch_max_size=30)

  [MEMORY] Epoch 13 Batch 0:
    Baseline: alloc=92MB, reserved=30526MB
    01_batch_on_gpu: alloc=106MB (+13MB), reserved=30548MB
    02_after_forward: alloc=26464MB (+26358MB), reserved=30548MB
    03_before_backward: alloc=26489MB (+25MB), reserved=30548MB
    04_after_backward: alloc=245MB (-26243MB), reserved=30548MB
    PEAK: alloc=27308MB, reserved=30548MB / 24576MB (124.3%)

    [BREAKDOWN] Largest increase at '02_after_forward' (+26358MB):
      Model params: 73.4MB
      Model grads:  0.0MB
      Optimizer:    146.7MB (estimate (Adam-like: AdamW))
      Batch data:   7.6MB
      Activations:  84.1MB
      Output tensors (top 5):
        features: 43.9MB
        support_features: 31.2MB
        all_logits: 6.0MB
        attention_maps: 1.2MB
        logits: 0.9MB
      Module params (top 5):
        solver: 28.4MB
        hyper_lora: 19.1MB
        context_encoder: 15.0MB
        context_injector: 3.0MB
        hpm: 2.9MB
      Active modules: solver_context, cross_attention, hpm
      Inactive modules: hyperlora, loo, equivariance

  [MEMORY] First forward pass with staged modules active:
  [MEMORY] Batch 0: alloc=245MB, reserved=30548MB, headroom=-5972MB
  [WARNING] >95% GPU memory used! Training may slow due to shared memory.
  Batch 0/5: loss=0.2877, focal_weighted=0.1504, batch_acc=8.0%, exact=0/50 (0.0%), running_acc=8.0%, lr=5.00e-04, hpm=0.0051
    [TaskTrack] Global_Solved: 3/342, Epoch_Solved: 0, New_Puzzles: 0
    FG: batch=5.8% run50=5.8% | BG: batch=9.1% run50=9.1%
    Per-Color: [0:16% 1:4% 2:0% 3:11% 4:12% 5:11% 6:0% 7:8% 8:2% 9:6%]
    Running50: [0:16% 1:4% 2:0% 3:11% 4:12% 5:11% 6:0% 7:8% 8:2% 9:6%]
    Solver: [2.812, 2.812, 2.812, 2.797, 2.797, 2.797, 2.797] ⚠ best=3
  [Batch 1] Grid size: 30x30 (batch_max_size=30)

  [MEMORY] Epoch 13 Batch 1:
    Baseline: alloc=92MB, reserved=30526MB
    01_batch_on_gpu: alloc=218MB (+126MB), reserved=30550MB
    02_after_forward: alloc=21861MB (+21643MB), reserved=30550MB
    03_before_backward: alloc=21909MB (+48MB), reserved=30550MB
    04_after_backward: alloc=288MB (-21622MB), reserved=30550MB
    PEAK: alloc=22403MB, reserved=30550MB / 24576MB (124.3%)

    [BREAKDOWN] Largest increase at '02_after_forward' (+21643MB):
      Model params: 73.4MB
      Model grads:  52.0MB
      Optimizer:    146.7MB (estimate (Adam-like: AdamW))
      Batch data:   5.5MB
      Activations:  74.6MB
      Output tensors (top 5):
        features: 43.9MB
        support_features: 21.9MB
        all_logits: 6.0MB
        attention_maps: 1.2MB
        logits: 0.9MB
      Module params (top 5):
        solver: 28.4MB
        hyper_lora: 19.1MB
        context_encoder: 15.0MB
        context_injector: 3.0MB
        hpm: 2.9MB
      Active modules: solver_context, cross_attention, hpm
      Inactive modules: hyperlora, loo, equivariance
  [MEMORY] Batch 1: alloc=288MB, reserved=30550MB, headroom=-5974MB
  [WARNING] >95% GPU memory used! Training may slow due to shared memory.
  Batch 1/5: loss=0.4375, focal_weighted=0.2868, batch_acc=14.4%, exact=0/100 (0.0%), running_acc=11.2%, lr=5.00e-04, hpm=0.0050
    [TaskTrack] Global_Solved: 3/342, Epoch_Solved: 0, New_Puzzles: 0
    FG: batch=7.8% run50=6.8% | BG: batch=16.7% run50=12.9%
    Per-Color: [0:22% 1:0% 2:0% 3:8% 4:8% 5:15% 6:4% 7:32% 8:8% 9:10%]
    Running50: [0:19% 1:2% 2:0% 3:9% 4:10% 5:13% 6:2% 7:20% 8:5% 9:8%]
  [Batch 2] Grid size: 30x30 (batch_max_size=30)

  [MEMORY] Epoch 13 Batch 2:
    Baseline: alloc=92MB, reserved=30526MB
    01_batch_on_gpu: alloc=219MB (+127MB), reserved=30550MB
    02_after_forward: alloc=18727MB (+18508MB), reserved=30550MB
    03_before_backward: alloc=18814MB (+87MB), reserved=30552MB
    04_after_backward: alloc=283MB (-18531MB), reserved=30552MB
    PEAK: alloc=19102MB, reserved=30552MB / 24576MB (124.3%)

    [BREAKDOWN] Largest increase at '02_after_forward' (+18508MB):
      Model params: 73.4MB
      Model grads:  52.0MB
      Optimizer:    146.7MB (estimate (Adam-like: AdamW))
      Batch data:   4.1MB
      Activations:  68.3MB
      Output tensors (top 5):
        features: 43.9MB
        support_features: 15.6MB
        all_logits: 6.0MB
        attention_maps: 1.2MB
        logits: 0.9MB
      Module params (top 5):
        solver: 28.4MB
        hyper_lora: 19.1MB
        context_encoder: 15.0MB
        context_injector: 3.0MB
        hpm: 2.9MB
      Active modules: solver_context, cross_attention, hpm
      Inactive modules: hyperlora, loo, equivariance
  [MEMORY] Batch 2: alloc=283MB, reserved=30552MB, headroom=-5976MB
  [WARNING] >95% GPU memory used! Training may slow due to shared memory.
  Batch 2/5: loss=0.9064, focal_weighted=0.7412, batch_acc=12.9%, exact=0/150 (0.0%), running_acc=11.8%, lr=5.00e-04, hpm=0.0050
    [TaskTrack] Global_Solved: 3/342, Epoch_Solved: 0, New_Puzzles: 0
    FG: batch=6.4% run50=6.7% | BG: batch=19.0% run50=14.9%
    Per-Color: [0:22% 1:3% 2:1% 3:5% 4:11% 5:8% 6:15% 7:16% 8:7% 9:10%]
    Running50: [0:20% 1:3% 2:0% 3:8% 4:10% 5:11% 6:6% 7:18% 8:6% 9:9%]
  [Batch 3] Grid size: 30x30 (batch_max_size=30)

  [MEMORY] Epoch 13 Batch 3:
    Baseline: alloc=92MB, reserved=30526MB
    01_batch_on_gpu: alloc=219MB (+127MB), reserved=30552MB
    02_after_forward: alloc=20299MB (+20080MB), reserved=30552MB
    03_before_backward: alloc=20326MB (+27MB), reserved=30552MB
    04_after_backward: alloc=288MB (-20038MB), reserved=30552MB
    PEAK: alloc=20740MB, reserved=30552MB / 24576MB (124.3%)

    [BREAKDOWN] Largest increase at '02_after_forward' (+20080MB):
      Model params: 73.4MB
      Model grads:  52.0MB
      Optimizer:    146.7MB (estimate (Adam-like: AdamW))
      Batch data:   4.8MB
      Activations:  71.5MB
      Output tensors (top 5):
        features: 43.9MB
        support_features: 18.8MB
        all_logits: 6.0MB
        attention_maps: 1.2MB
        logits: 0.9MB
      Module params (top 5):
        solver: 28.4MB
        hyper_lora: 19.1MB
        context_encoder: 15.0MB
        context_injector: 3.0MB
        hpm: 2.9MB
      Active modules: solver_context, cross_attention, hpm
      Inactive modules: hyperlora, loo, equivariance
  Batch 3/5: loss=0.2961, focal_weighted=0.1650, batch_acc=12.1%, exact=1/200 (0.5%), running_acc=11.8%, lr=5.00e-04, hpm=0.0051
    [TaskTrack] Global_Solved: 3/342, Epoch_Solved: 1, New_Puzzles: 0
    FG: batch=9.1% run50=7.3% | BG: batch=12.3% run50=14.3%
    Per-Color: [0:15% 1:9% 2:0% 3:8% 4:7% 5:9% 6:7% 7:16% 8:2% 9:27%]
    Running50: [0:19% 1:4% 2:0% 3:8% 4:9% 5:11% 6:6% 7:18% 8:5% 9:13%]
  [Batch 4] Grid size: 30x30 (batch_max_size=30)

  [MEMORY] Epoch 13 Batch 4:
    Baseline: alloc=92MB, reserved=30526MB
    01_batch_on_gpu: alloc=215MB (+123MB), reserved=30552MB
    02_after_forward: alloc=23428MB (+23213MB), reserved=30552MB
    03_before_backward: alloc=23456MB (+27MB), reserved=30552MB
    04_after_backward: alloc=289MB (-23167MB), reserved=30552MB
    PEAK: alloc=24071MB, reserved=30552MB / 24576MB (124.3%)

    [BREAKDOWN] Largest increase at '02_after_forward' (+23213MB):
      Model params: 73.4MB
      Model grads:  52.0MB
      Optimizer:    146.7MB (estimate (Adam-like: AdamW))
      Batch data:   6.2MB
      Activations:  77.8MB
      Output tensors (top 5):
        features: 43.9MB
        support_features: 25.0MB
        all_logits: 6.0MB
        attention_maps: 1.2MB
        logits: 0.9MB
      Module params (top 5):
        solver: 28.4MB
        hyper_lora: 19.1MB
        context_encoder: 15.0MB
        context_injector: 3.0MB
        hpm: 2.9MB
      Active modules: solver_context, cross_attention, hpm
      Inactive modules: hyperlora, loo, equivariance
  Batch 4/5: loss=0.2851, focal_weighted=0.1510, batch_acc=11.0%, exact=1/250 (0.4%), running_acc=11.7%, lr=5.00e-04, hpm=0.0050
    [TaskTrack] Global_Solved: 3/342, Epoch_Solved: 1, New_Puzzles: 0
    FG: batch=7.9% run50=7.4% | BG: batch=12.4% run50=13.9%
    Per-Color: [0:17% 1:1% 2:0% 3:10% 4:7% 5:9% 6:5% 7:28% 8:1% 9:12%]
    Running50: [0:18% 1:4% 2:0% 3:8% 4:9% 5:10% 6:6% 7:20% 8:4% 9:13%]

Epoch 14 Summary:
  Total Loss: 0.4426
  Task Loss (focal): 0.2989
  [Global Task Progress] Unique Solved: 3/342 (0.9%)
    NEW puzzles solved this epoch: 0
  Entropy Loss: 4.2909 (weight=0.01)
  Sparsity Loss: 0.1639 (weight=0.5)
  Predicate Loss: 0.0000 (weight=0.01)
  HPM Balance Loss: 0.0051 (weight=0.01)
  HPM Gate Value: 0.0000 (0=no contribution, 1=full)
  HPM Instance Buffer: 2 entries
  HPM Procedural Buffer: 0 entries
  HPM Tasks Added (exact matches): 0
    (No exact matches this epoch: 1)
  Time: 38.0s, LR: 5.00e-04
    Per-module LRs: DSC:5.00e-04, MSRE:5.00e-04, Other:5.00e-04
  Temperature: 0.9559 (lower=sharper attention)
  Samples Processed: 250 (5 batches)
  Dihedral Distribution: [16.0%, 10.8%, 12.0%, 14.8%, 9.6%, 11.2%, 13.6%, 12.0%]
  Color Permutation: 0.0% (0/250)
  Translational Aug: 100.0% (250/250), unique offsets: 229
  Aug Quality: OK
  --- Training Diagnostics ---
  Solver Steps: 7 (deep supervision active)
  Per-Step Loss (epoch avg, 1 batches): [2.8125, 2.8125, 2.8125, 2.7969, 2.7969, 2.7969, 2.7969]
    [!] Best step is 3 (middle), not last - solver may be over-iterating!
    [!] Best: step 3 (2.7969), Final: step 6 (2.7969)
    [!] Potential gain from best-step selection: 0.0% lower loss
  Best-Step Histogram: []
  Solver Health: Last step best: 0.0%, Earlier step best: 100.0%
  Avg Step Improvement: 0.6% (step0→stepN)
    ⚠️ SOLVER OVER-ITERATION WARNING: 100% of batches had earlier step as best!
    ⚠️ Consider: (1) enabling best-step selection, (2) reducing num_solver_steps, (3) enabling ACT
  Grad Norms: DSC=0.0000, StopPred=0.0000, Encoder=0.0000, Solver=0.0000
  StopPred Weight Var: 2.80e-02
  Attention: max=0.0446, min=0.000000
  Stop Prob: 0.268 (approx 5.1 clues active)
  Stop Probs Std: 0.090 (global std across batch×clues)
  Clues Used: mean=5.13, std=0.40, range=[3.4, 5.8]
  Clue-Loss Correlation: -0.139 (unexpected negative - check gradient flow)
  Stop Logits: mean=-1.05, std=0.46, range=[-2.3, 0.6]
    Clue count varies by task (per-sample coupling active)
  Per-Clue Entropy: [3.82, 3.82, 3.82, 3.82, 3.82, 3.82, 3.82] (mean=3.82, max=6.80)
    [!] Clues have uniform entropy (std=0.001) - not differentiating!
  Centroid Spread: 0.06 (higher=more diverse)
    🚨 CRITICAL COLLAPSE: Spread=0.06 < 0.5 - all clues at same location!
    [!] Stop predictor cannot differentiate - needs diversity regularizer
  --- Stop Predictor Coupling ---
  Entropy Input to Stop: 0.5611 (normalized, lower=sharper)
  Per-Clue Entropy Input: [0.561, 0.561, 0.561, 0.561, 0.561, 0.561, 0.561]
  Per-Clue Stop Prob: [0.249, 0.287, 0.273, 0.264, 0.266, 0.262, 0.277]
  --- Sparsity Loss Breakdown (Per-Sample Coupled) ---
  Min Clue Penalty: 0.0000 (per-sample avg)
  Per-Sample Clue Penalty (scaled): 0.0000
  Base Pondering: 0.1024 (clues=5.12)
  Entropy Pondering: 0.0579
  --- Per-Class Distribution (Valid Pixels Only) ---
  Padding: 94.6% of grid (ignored in loss)
  Pred %: [14.4, 3.2, 0.9, 14.5, 6.6, 18.1, 1.3, 26.1, 3.0, 11.8]
  Target %: [56.8, 8.3, 5.6, 5.8, 3.5, 12.4, 1.6, 1.1, 3.5, 1.4]
  Per-Class Acc %: [20, 3, 0, 7, 10, 10, 8, 20, 6, 11]
  ==================================================
  PER-SAMPLE TRAINING ACCURACY (Epoch 14)
  ==================================================
  ★ Mean Accuracy: 11.7%
  ★ Exact Match: 1/250 (0.4%)
  ★ High Acc (≥90%): 1/250 (0.4%)
  FG Accuracy: 7.4%
  BG Accuracy: 13.9%
  Batch Trend: 8.0% → 11.5% (↑ 3.5pp)
    ✓ Accuracy improving within epoch - learning is active!
  Accuracy Distribution: 0-25%:100%, 25-50%:0%, 50-75%:0%, 75-90%:0%, 90-100%:0%
    [!] Many samples stuck at low accuracy - check data/model!
  Running Window (last 5 batches): 11.7% ± 2.2%

  --- PER-COLOR ACCURACY (10 classes) ---
  Color:       0     1     2     3     4     5     6     7     8     9
  Acc%:      20%    3%    0%    7%   10%   10%    8%   20%    6%   11%
  Target:  55.2%  7.7%  5.7%  8.3%  4.7%  5.3%  2.8%  1.8%  6.7%  1.9%
  Pred:    17.7%  2.1%  0.8%  8.5%  7.0% 20.9%  3.0% 26.7%  3.4%  9.9%
  [!] Weak colors (<50% acc): 0(Black), 1(Blue), 2(Red), 3(Green), 4(Yellow), 5(Gray), 6(Pink), 7(Orange), 8(Cyan), 9(Brown)
  [!] Under-predicting color 1 (Blue): 2.1% vs target 7.7%
  [!] Under-predicting color 2 (Red): 0.8% vs target 5.7%
  ==================================================

  ############################################################
  🚦 TRAINING HEALTH CHECK - Epoch 14
  ############################################################
  ✓ Attention sharpening (0.56 < 0.7)
  ✓ No NaN/Inf issues
  ✓ No color mode collapse
  ⚠ Stop probs nearly uniform (global=0.090, per_clue=0.011)
  ⚠ Weak confidence-stop coupling (r=0.06)
  ⚠ Loss slowly decreasing (0.311 → 0.299)
  ✗ Centroid COLLAPSE (0.1 < 0.5)
  ✗ Accuracy not improving (19.9% → 11.7%)
  --------------------------------------------------------
  RESULT: 3/8 checks passed
  STATUS: 🟠 WARNING
  → Multiple issues detected. Consider intervention soon.

  RECOMMENDED ACTIONS:
  ############################################################
  HPM Instance Buffer: 2 entries SAVED
    → Saved to: checkpoints\rlan_stable_merged\hpm\instance_buffer.pt
  HPM Procedural Buffer: 0 entries (empty, not saved)
  Saved checkpoint to checkpoints\rlan_stable_merged\latest.pt
  [ARCDataset] Augmentation config updated:
    dihedral=True, color_perm=False (prob=0.50), translational=True

Epoch 15/200
----------------------------------------
  [Batch 0] Grid size: 30x30 (batch_max_size=30)

  [MEMORY] Epoch 14 Batch 0:
    Baseline: alloc=92MB, reserved=30552MB
    01_batch_on_gpu: alloc=105MB (+12MB), reserved=30574MB
    02_after_forward: alloc=23323MB (+23218MB), reserved=30574MB
    03_before_backward: alloc=23368MB (+45MB), reserved=30574MB
    04_after_backward: alloc=238MB (-23130MB), reserved=30574MB
    PEAK: alloc=23966MB, reserved=30574MB / 24576MB (124.4%)

    [BREAKDOWN] Largest increase at '02_after_forward' (+23218MB):
      Model params: 73.4MB
      Model grads:  0.0MB
      Optimizer:    146.7MB (estimate (Adam-like: AdamW))
      Batch data:   6.2MB
      Activations:  77.8MB
      Output tensors (top 5):
        features: 43.9MB
        support_features: 25.0MB
        all_logits: 6.0MB
        attention_maps: 1.2MB
        logits: 0.9MB
      Module params (top 5):
        solver: 28.4MB
        hyper_lora: 19.1MB
        context_encoder: 15.0MB
        context_injector: 3.0MB
        hpm: 2.9MB
      Active modules: solver_context, cross_attention, hpm
      Inactive modules: hyperlora, loo, equivariance

  [MEMORY] First forward pass with staged modules active:
  [MEMORY] Batch 0: alloc=238MB, reserved=30574MB, headroom=-5998MB
  [WARNING] >95% GPU memory used! Training may slow due to shared memory.
  Batch 0/5: loss=0.4504, focal_weighted=0.2986, batch_acc=14.3%, exact=0/50 (0.0%), running_acc=14.3%, lr=5.00e-04, hpm=0.0050
    [TaskTrack] Global_Solved: 3/342, Epoch_Solved: 0, New_Puzzles: 0
    FG: batch=7.0% run50=7.0% | BG: batch=17.0% run50=17.0%
    Per-Color: [0:22% 1:4% 2:0% 3:9% 4:5% 5:12% 6:3% 7:20% 8:3% 9:17%]
    Running50: [0:22% 1:4% 2:0% 3:9% 4:5% 5:12% 6:3% 7:20% 8:3% 9:17%]
    Solver: [2.500, 2.500, 2.500, 2.500, 2.500, 2.484, 2.484] ⚠ best=5
  [Batch 1] Grid size: 30x30 (batch_max_size=30)

  [MEMORY] Epoch 14 Batch 1:
    Baseline: alloc=92MB, reserved=30552MB
    01_batch_on_gpu: alloc=218MB (+126MB), reserved=30576MB
    02_after_forward: alloc=23431MB (+23213MB), reserved=30576MB
    03_before_backward: alloc=23457MB (+26MB), reserved=30576MB
    04_after_backward: alloc=292MB (-23165MB), reserved=30576MB
    PEAK: alloc=24075MB, reserved=30576MB / 24576MB (124.4%)

    [BREAKDOWN] Largest increase at '02_after_forward' (+23213MB):
      Model params: 73.4MB
      Model grads:  52.0MB
      Optimizer:    146.7MB (estimate (Adam-like: AdamW))
      Batch data:   6.2MB
      Activations:  77.8MB
      Output tensors (top 5):
        features: 43.9MB
        support_features: 25.0MB
        all_logits: 6.0MB
        attention_maps: 1.2MB
        logits: 0.9MB
      Module params (top 5):
        solver: 28.4MB
        hyper_lora: 19.1MB
        context_encoder: 15.0MB
        context_injector: 3.0MB
        hpm: 2.9MB
      Active modules: solver_context, cross_attention, hpm
      Inactive modules: hyperlora, loo, equivariance
  [MEMORY] Batch 1: alloc=292MB, reserved=30576MB, headroom=-6000MB
  [WARNING] >95% GPU memory used! Training may slow due to shared memory.
  Batch 1/5: loss=0.3282, focal_weighted=0.1979, batch_acc=9.9%, exact=0/100 (0.0%), running_acc=12.1%, lr=5.00e-04, hpm=0.0050
    [TaskTrack] Global_Solved: 3/342, Epoch_Solved: 0, New_Puzzles: 0
    FG: batch=7.4% run50=7.2% | BG: batch=10.7% run50=13.9%
    Per-Color: [0:16% 1:4% 2:0% 3:15% 4:3% 5:4% 6:2% 7:28% 8:9% 9:17%]
    Running50: [0:19% 1:4% 2:0% 3:12% 4:4% 5:8% 6:3% 7:24% 8:6% 9:17%]
  [Batch 2] Grid size: 30x30 (batch_max_size=30)

  [MEMORY] Epoch 14 Batch 2:
    Baseline: alloc=92MB, reserved=30552MB
    01_batch_on_gpu: alloc=222MB (+130MB), reserved=30576MB
    02_after_forward: alloc=18727MB (+18505MB), reserved=30576MB
    03_before_backward: alloc=18753MB (+25MB), reserved=30576MB
    04_after_backward: alloc=286MB (-18467MB), reserved=30576MB
    PEAK: alloc=19103MB, reserved=30576MB / 24576MB (124.4%)

    [BREAKDOWN] Largest increase at '02_after_forward' (+18505MB):
      Model params: 73.4MB
      Model grads:  52.0MB
      Optimizer:    146.7MB (estimate (Adam-like: AdamW))
      Batch data:   4.1MB
      Activations:  68.3MB
      Output tensors (top 5):
        features: 43.9MB
        support_features: 15.6MB
        all_logits: 6.0MB
        attention_maps: 1.2MB
        logits: 0.9MB
      Module params (top 5):
        solver: 28.4MB
        hyper_lora: 19.1MB
        context_encoder: 15.0MB
        context_injector: 3.0MB
        hpm: 2.9MB
      Active modules: solver_context, cross_attention, hpm
      Inactive modules: hyperlora, loo, equivariance
  [MEMORY] Batch 2: alloc=286MB, reserved=30576MB, headroom=-6000MB
  [WARNING] >95% GPU memory used! Training may slow due to shared memory.
  Batch 2/5: loss=0.2597, focal_weighted=0.1348, batch_acc=8.6%, exact=0/150 (0.0%), running_acc=10.9%, lr=5.00e-04, hpm=0.0051
    [TaskTrack] Global_Solved: 3/342, Epoch_Solved: 0, New_Puzzles: 0
    FG: batch=5.2% run50=6.6% | BG: batch=10.4% run50=12.7%
    Per-Color: [0:16% 1:3% 2:0% 3:6% 4:11% 5:18% 6:4% 7:16% 8:3% 9:5%]
    Running50: [0:18% 1:4% 2:0% 3:10% 4:7% 5:11% 6:3% 7:21% 8:5% 9:13%]
  [Batch 3] Grid size: 30x30 (batch_max_size=30)

  [MEMORY] Epoch 14 Batch 3:
    Baseline: alloc=92MB, reserved=30552MB
    01_batch_on_gpu: alloc=221MB (+128MB), reserved=30576MB
    02_after_forward: alloc=26576MB (+26356MB), reserved=30576MB
    03_before_backward: alloc=26603MB (+27MB), reserved=30576MB
    04_after_backward: alloc=303MB (-26300MB), reserved=30576MB
    PEAK: alloc=27421MB, reserved=30576MB / 24576MB (124.4%)

    [BREAKDOWN] Largest increase at '02_after_forward' (+26356MB):
      Model params: 73.4MB
      Model grads:  52.0MB
      Optimizer:    146.7MB (estimate (Adam-like: AdamW))
      Batch data:   7.6MB
      Activations:  84.1MB
      Output tensors (top 5):
        features: 43.9MB
        support_features: 31.2MB
        all_logits: 6.0MB
        attention_maps: 1.2MB
        logits: 0.9MB
      Module params (top 5):
        solver: 28.4MB
        hyper_lora: 19.1MB
        context_encoder: 15.0MB
        context_injector: 3.0MB
        hpm: 2.9MB
      Active modules: solver_context, cross_attention, hpm
      Inactive modules: hyperlora, loo, equivariance
  Batch 3/5: loss=0.2797, focal_weighted=0.1422, batch_acc=11.7%, exact=1/200 (0.5%), running_acc=11.1%, lr=5.00e-04, hpm=0.0050
    [TaskTrack] Global_Solved: 3/342, Epoch_Solved: 1, New_Puzzles: 0
    FG: batch=8.0% run50=6.9% | BG: batch=13.6% run50=12.9%
    Per-Color: [0:17% 1:2% 2:0% 3:9% 4:7% 5:14% 6:0% 7:20% 8:0% 9:31%]
    Running50: [0:18% 1:3% 2:0% 3:10% 4:7% 5:12% 6:2% 7:21% 8:4% 9:18%]
  [Batch 4] Grid size: 30x30 (batch_max_size=30)

  [MEMORY] Epoch 14 Batch 4:
    Baseline: alloc=92MB, reserved=30552MB
    01_batch_on_gpu: alloc=217MB (+124MB), reserved=30576MB
    02_after_forward: alloc=18719MB (+18502MB), reserved=30576MB
    03_before_backward: alloc=18799MB (+80MB), reserved=30576MB
    04_after_backward: alloc=279MB (-18520MB), reserved=30576MB
    PEAK: alloc=19094MB, reserved=30576MB / 24576MB (124.4%)

    [BREAKDOWN] Largest increase at '02_after_forward' (+18502MB):
      Model params: 73.4MB
      Model grads:  52.0MB
      Optimizer:    146.7MB (estimate (Adam-like: AdamW))
      Batch data:   4.1MB
      Activations:  68.3MB
      Output tensors (top 5):
        features: 43.9MB
        support_features: 15.6MB
        all_logits: 6.0MB
        attention_maps: 1.2MB
        logits: 0.9MB
      Module params (top 5):
        solver: 28.4MB
        hyper_lora: 19.1MB
        context_encoder: 15.0MB
        context_injector: 3.0MB
        hpm: 2.9MB
      Active modules: solver_context, cross_attention, hpm
      Inactive modules: hyperlora, loo, equivariance
  Batch 4/5: loss=0.8824, focal_weighted=0.7188, batch_acc=13.8%, exact=1/250 (0.4%), running_acc=11.6%, lr=5.00e-04, hpm=0.0050
    [TaskTrack] Global_Solved: 3/342, Epoch_Solved: 1, New_Puzzles: 0
    FG: batch=6.1% run50=6.8% | BG: batch=19.4% run50=14.2%
    Per-Color: [0:23% 1:1% 2:2% 3:4% 4:10% 5:12% 6:9% 7:15% 8:7% 9:14%]
    Running50: [0:19% 1:3% 2:0% 3:8% 4:7% 5:12% 6:4% 7:20% 8:5% 9:17%]

Epoch 15 Summary:
  Total Loss: 0.4401
  Task Loss (focal): 0.2985
  [Global Task Progress] Unique Solved: 3/342 (0.9%)
    NEW puzzles solved this epoch: 0
  Entropy Loss: 4.2361 (weight=0.01)
  Sparsity Loss: 0.1608 (weight=0.5)
  Predicate Loss: 0.0000 (weight=0.01)
  HPM Balance Loss: 0.0050 (weight=0.01)
  HPM Gate Value: 0.0000 (0=no contribution, 1=full)
  HPM Instance Buffer: 2 entries
  HPM Procedural Buffer: 0 entries
  HPM Tasks Added (exact matches): 0
    (No exact matches this epoch: 1)
  Time: 36.5s, LR: 5.00e-04
    Per-module LRs: DSC:5.00e-04, MSRE:5.00e-04, Other:5.00e-04
  Temperature: 0.9526 (lower=sharper attention)
  Samples Processed: 250 (5 batches)
  Dihedral Distribution: [15.6%, 16.0%, 8.4%, 12.4%, 12.8%, 14.0%, 7.6%, 13.2%]
  Color Permutation: 0.0% (0/250)
  Translational Aug: 100.0% (250/250), unique offsets: 234
  Aug Quality: OK
  --- Training Diagnostics ---
  Solver Steps: 7 (deep supervision active)
  Per-Step Loss (epoch avg, 1 batches): [2.5000, 2.5000, 2.5000, 2.5000, 2.5000, 2.4844, 2.4844]
    [!] Best step is 5 (middle), not last - solver may be over-iterating!
    [!] Best: step 5 (2.4844), Final: step 6 (2.4844)
    [!] Potential gain from best-step selection: 0.0% lower loss
  Best-Step Histogram: []
  Solver Health: Last step best: 0.0%, Earlier step best: 100.0%
  Avg Step Improvement: 0.6% (step0→stepN)
    ⚠️ SOLVER OVER-ITERATION WARNING: 100% of batches had earlier step as best!
    ⚠️ Consider: (1) enabling best-step selection, (2) reducing num_solver_steps, (3) enabling ACT
  Grad Norms: DSC=0.0000, StopPred=0.0000, Encoder=0.0000, Solver=0.0000
  StopPred Weight Var: 2.80e-02
  Attention: max=0.0196, min=0.000000
  Stop Prob: 0.270 (approx 5.1 clues active)
  Stop Probs Std: 0.091 (global std across batch×clues)
  Clues Used: mean=5.11, std=0.40, range=[3.4, 5.7]
  Clue-Loss Correlation: -0.014 (weak - per-sample coupling may need tuning)
  Stop Logits: mean=-1.04, std=0.47, range=[-2.3, 1.1]
    Clue count varies by task (per-sample coupling active)
  Per-Clue Entropy: [4.67, 4.67, 4.67, 4.66, 4.66, 4.67, 4.67] (mean=4.67, max=6.80)
    [!] Clues have uniform entropy (std=0.002) - not differentiating!
  Centroid Spread: 0.11 (higher=more diverse)
    🚨 CRITICAL COLLAPSE: Spread=0.11 < 0.5 - all clues at same location!
    [!] Stop predictor cannot differentiate - needs diversity regularizer
  --- Stop Predictor Coupling ---
  Entropy Input to Stop: 0.6859 (normalized, lower=sharper)
  Per-Clue Entropy Input: [0.686, 0.686, 0.686, 0.686, 0.686, 0.686, 0.686]
  Per-Clue Stop Prob: [0.271, 0.254, 0.254, 0.262, 0.275, 0.293, 0.279]
  --- Sparsity Loss Breakdown (Per-Sample Coupled) ---
  Min Clue Penalty: 0.0000 (per-sample avg)
  Per-Sample Clue Penalty (scaled): 0.0000
  Base Pondering: 0.1022 (clues=5.11)
  Entropy Pondering: 0.0704
  --- Per-Class Distribution (Valid Pixels Only) ---
  Padding: 88.0% of grid (ignored in loss)
  Pred %: [20.4, 1.4, 0.4, 6.5, 7.0, 22.8, 1.1, 28.8, 1.5, 10.0]
  Target %: [66.3, 3.7, 4.1, 4.7, 4.4, 3.8, 2.9, 1.0, 7.3, 1.6]
    [!] Missing foreground colors: [2]
  Per-Class Acc %: [21, 3, 1, 6, 8, 12, 5, 19, 5, 16]
  ==================================================
  PER-SAMPLE TRAINING ACCURACY (Epoch 15)
  ==================================================
  ★ Mean Accuracy: 11.6%
  ★ Exact Match: 1/250 (0.4%)
  ★ High Acc (≥90%): 1/250 (0.4%)
  FG Accuracy: 6.8%
  BG Accuracy: 14.2%
  Batch Trend: 14.3% → 12.7% (↓ 1.5pp)
  Accuracy Distribution: 0-25%:100%, 25-50%:0%, 50-75%:0%, 75-90%:0%, 90-100%:0%
    [!] Many samples stuck at low accuracy - check data/model!
  Running Window (last 5 batches): 11.6% ± 2.2%

  --- PER-COLOR ACCURACY (10 classes) ---
  Color:       0     1     2     3     4     5     6     7     8     9
  Acc%:      21%    3%    1%    6%    8%   12%    5%   19%    5%   16%
  Target:  59.6%  6.1%  5.0%  7.1%  4.9%  5.0%  2.8%  1.8%  6.1%  1.6%
  Pred:    17.6%  2.5%  0.7%  7.6%  8.1% 21.5%  2.8% 27.4%  2.7%  9.1%
  [!] Weak colors (<50% acc): 0(Black), 1(Blue), 2(Red), 3(Green), 4(Yellow), 5(Gray), 6(Pink), 7(Orange), 8(Cyan), 9(Brown)
  [!] Under-predicting color 1 (Blue): 2.5% vs target 6.1%
  [!] Under-predicting color 2 (Red): 0.7% vs target 5.0%
  [!] Under-predicting color 8 (Cyan): 2.7% vs target 6.1%
  ==================================================

  ==================================================
  LEARNING TRAJECTORY (Epoch 15)
  ==================================================
  Stop Prob:   0.270 → (init=0.27, task-dependent)
  Exp. Clues:  5.11 (latent variable, task-dependent)
  Attn Entropy: 4.67 ↑ (max=6.8, sharper=better)
  Task Loss:   0.2985 →
  Train Acc:   11.6% →
  Exact Match: 0.4% →
  Best Step:   5 (later=better refinement)
  FG Coverage: 236.0% of target ↑
  [!] Potential issues: stop_prob not increasing, attention not sharpening, task_loss not decreasing, train_accuracy not improving
  ==================================================

  ############################################################
  🚦 TRAINING HEALTH CHECK - Epoch 15
  ############################################################
  ✓ Attention sharpening (0.69 < 0.7)
  ✓ No NaN/Inf issues
  ✓ No color mode collapse
  ⚠ Stop probs nearly uniform (global=0.091, per_clue=0.013)
  ⚠ Negative coupling (r=-0.05) - early epoch OK
  ⚠ Loss slowly decreasing (0.311 → 0.298)
  ✗ Centroid COLLAPSE (0.1 < 0.5)
  ✗ Accuracy not improving (19.9% → 11.6%)
  --------------------------------------------------------
  RESULT: 3/8 checks passed
  STATUS: 🟠 WARNING
  → Multiple issues detected. Consider intervention soon.

  RECOMMENDED ACTIONS:
  ############################################################
  HPM Instance Buffer: 2 entries SAVED
    → Saved to: checkpoints\rlan_stable_merged\hpm\instance_buffer.pt
  HPM Procedural Buffer: 0 entries (empty, not saved)
  Saved checkpoint to checkpoints\rlan_stable_merged\latest.pt
  [ARCDataset] Augmentation config updated:
    dihedral=True, color_perm=False (prob=0.50), translational=True

Epoch 16/200
----------------------------------------
  [LR] Reduced to 2.50e-04 (×0.5) for stability

============================================================
PHASE 2: CROSS-ATTENTION INJECTOR ACTIVATED (epoch 16)
============================================================
  CrossAttentionInjector: NOW ACTIVE (was using FiLM fallback)
  GPU MEMORY: 147MB / 24576MB (0.6%)
============================================================

  [Batch 0] Grid size: 30x30 (batch_max_size=30)

  [MEMORY] Epoch 15 Batch 0:
    Baseline: alloc=92MB, reserved=30576MB
    01_batch_on_gpu: alloc=103MB (+10MB), reserved=30598MB
    02_after_forward: alloc=23321MB (+23218MB), reserved=30598MB
    03_before_backward: alloc=23350MB (+29MB), reserved=30598MB
    04_after_backward: alloc=236MB (-23113MB), reserved=30598MB
    PEAK: alloc=23964MB, reserved=30598MB / 24576MB (124.5%)

    [BREAKDOWN] Largest increase at '02_after_forward' (+23218MB):
      Model params: 73.4MB
      Model grads:  0.0MB
      Optimizer:    146.7MB (estimate (Adam-like: AdamW))
      Batch data:   6.2MB
      Activations:  77.8MB
      Output tensors (top 5):
        features: 43.9MB
        support_features: 25.0MB
        all_logits: 6.0MB
        attention_maps: 1.2MB
        logits: 0.9MB
      Module params (top 5):
        solver: 28.4MB
        hyper_lora: 19.1MB
        context_encoder: 15.0MB
        context_injector: 3.0MB
        hpm: 2.9MB
      Active modules: solver_context, cross_attention, hpm
      Inactive modules: hyperlora, loo, equivariance

  [MEMORY] First forward pass with staged modules active:
  [MEMORY] Batch 0: alloc=236MB, reserved=30598MB, headroom=-6022MB
  [WARNING] >95% GPU memory used! Training may slow due to shared memory.
  Batch 0/5: loss=0.2924, focal_weighted=0.1558, batch_acc=10.6%, exact=0/50 (0.0%), running_acc=10.6%, lr=2.50e-04, hpm=0.0050
    [TaskTrack] Global_Solved: 3/342, Epoch_Solved: 0, New_Puzzles: 0
    FG: batch=6.7% run50=6.7% | BG: batch=12.7% run50=12.7%
    Per-Color: [0:17% 1:8% 2:1% 3:10% 4:6% 5:14% 6:0% 7:14% 8:2% 9:31%]
    Running50: [0:17% 1:8% 2:1% 3:10% 4:6% 5:14% 6:0% 7:14% 8:2% 9:31%]
    Solver: [2.328, 2.328, 2.312, 2.312, 2.297, 2.297, 2.297] ⚠ best=4
  [Batch 1] Grid size: 30x30 (batch_max_size=30)

  [MEMORY] Epoch 15 Batch 1:
    Baseline: alloc=92MB, reserved=30576MB
    01_batch_on_gpu: alloc=218MB (+126MB), reserved=30600MB
    02_after_forward: alloc=18723MB (+18505MB), reserved=30600MB
    03_before_backward: alloc=18803MB (+80MB), reserved=30600MB
    04_after_backward: alloc=282MB (-18521MB), reserved=30600MB
    PEAK: alloc=19098MB, reserved=30600MB / 24576MB (124.5%)

    [BREAKDOWN] Largest increase at '02_after_forward' (+18505MB):
      Model params: 73.4MB
      Model grads:  52.0MB
      Optimizer:    146.7MB (estimate (Adam-like: AdamW))
      Batch data:   4.1MB
      Activations:  68.3MB
      Output tensors (top 5):
        features: 43.9MB
        support_features: 15.6MB
        all_logits: 6.0MB
        attention_maps: 1.2MB
        logits: 0.9MB
      Module params (top 5):
        solver: 28.4MB
        hyper_lora: 19.1MB
        context_encoder: 15.0MB
        context_injector: 3.0MB
        hpm: 2.9MB
      Active modules: solver_context, cross_attention, hpm
      Inactive modules: hyperlora, loo, equivariance
  [MEMORY] Batch 1: alloc=282MB, reserved=30600MB, headroom=-6024MB
  [WARNING] >95% GPU memory used! Training may slow due to shared memory.
  Batch 1/5: loss=0.7731, focal_weighted=0.6096, batch_acc=13.2%, exact=0/100 (0.0%), running_acc=11.9%, lr=2.50e-04, hpm=0.0051
    [TaskTrack] Global_Solved: 3/342, Epoch_Solved: 0, New_Puzzles: 0
    FG: batch=5.7% run50=6.2% | BG: batch=18.7% run50=15.7%
    Per-Color: [0:21% 1:2% 2:0% 3:4% 4:10% 5:10% 6:10% 7:20% 8:6% 9:15%]
    Running50: [0:19% 1:5% 2:1% 3:7% 4:8% 5:12% 6:5% 7:17% 8:4% 9:23%]
  [Batch 2] Grid size: 30x30 (batch_max_size=30)

  [MEMORY] Epoch 15 Batch 2:
    Baseline: alloc=92MB, reserved=30576MB
    01_batch_on_gpu: alloc=222MB (+130MB), reserved=30600MB
    02_after_forward: alloc=23437MB (+23215MB), reserved=30600MB
    03_before_backward: alloc=23480MB (+43MB), reserved=30600MB
    04_after_backward: alloc=297MB (-23182MB), reserved=30600MB
    PEAK: alloc=24081MB, reserved=30600MB / 24576MB (124.5%)

    [BREAKDOWN] Largest increase at '02_after_forward' (+23215MB):
      Model params: 73.4MB
      Model grads:  52.0MB
      Optimizer:    146.7MB (estimate (Adam-like: AdamW))
      Batch data:   6.2MB
      Activations:  77.8MB
      Output tensors (top 5):
        features: 43.9MB
        support_features: 25.0MB
        all_logits: 6.0MB
        attention_maps: 1.2MB
        logits: 0.9MB
      Module params (top 5):
        solver: 28.4MB
        hyper_lora: 19.1MB
        context_encoder: 15.0MB
        context_injector: 3.0MB
        hpm: 2.9MB
      Active modules: solver_context, cross_attention, hpm
      Inactive modules: hyperlora, loo, equivariance
  [MEMORY] Batch 2: alloc=297MB, reserved=30600MB, headroom=-6024MB
  [WARNING] >95% GPU memory used! Training may slow due to shared memory.
  Batch 2/5: loss=0.4603, focal_weighted=0.3094, batch_acc=12.1%, exact=0/150 (0.0%), running_acc=12.0%, lr=2.50e-04, hpm=0.0050
    [TaskTrack] Global_Solved: 3/342, Epoch_Solved: 0, New_Puzzles: 0
    FG: batch=4.8% run50=5.7% | BG: batch=15.8% run50=15.7%
    Per-Color: [0:21% 1:4% 2:0% 3:8% 4:7% 5:8% 6:4% 7:20% 8:6% 9:10%]
    Running50: [0:20% 1:5% 2:1% 3:7% 4:8% 5:11% 6:5% 7:18% 8:5% 9:19%]
  [Batch 3] Grid size: 30x30 (batch_max_size=30)

  [MEMORY] Epoch 15 Batch 3:
    Baseline: alloc=92MB, reserved=30576MB
    01_batch_on_gpu: alloc=224MB (+132MB), reserved=30600MB
    02_after_forward: alloc=26577MB (+26353MB), reserved=30600MB
    03_before_backward: alloc=26600MB (+24MB), reserved=30600MB
    04_after_backward: alloc=304MB (-26296MB), reserved=30600MB
    PEAK: alloc=27422MB, reserved=30600MB / 24576MB (124.5%)

    [BREAKDOWN] Largest increase at '02_after_forward' (+26353MB):
      Model params: 73.4MB
      Model grads:  52.0MB
      Optimizer:    146.7MB (estimate (Adam-like: AdamW))
      Batch data:   7.6MB
      Activations:  84.1MB
      Output tensors (top 5):
        features: 43.9MB
        support_features: 31.2MB
        all_logits: 6.0MB
        attention_maps: 1.2MB
        logits: 0.9MB
      Module params (top 5):
        solver: 28.4MB
        hyper_lora: 19.1MB
        context_encoder: 15.0MB
        context_injector: 3.0MB
        hpm: 2.9MB
      Active modules: solver_context, cross_attention, hpm
      Inactive modules: hyperlora, loo, equivariance
  Batch 3/5: loss=0.2896, focal_weighted=0.1570, batch_acc=8.4%, exact=0/200 (0.0%), running_acc=11.1%, lr=2.50e-04, hpm=0.0050
    [TaskTrack] Global_Solved: 3/342, Epoch_Solved: 0, New_Puzzles: 0
    FG: batch=7.5% run50=6.2% | BG: batch=8.7% run50=13.9%
    Per-Color: [0:15% 1:9% 2:1% 3:8% 4:9% 5:3% 6:0% 7:20% 8:2% 9:9%]
    Running50: [0:19% 1:6% 2:1% 3:7% 4:8% 5:9% 6:3% 7:19% 8:4% 9:16%]
  [Batch 4] Grid size: 30x30 (batch_max_size=30)

  [MEMORY] Epoch 15 Batch 4:
    Baseline: alloc=92MB, reserved=30576MB
    01_batch_on_gpu: alloc=218MB (+126MB), reserved=30600MB
    02_after_forward: alloc=21857MB (+21639MB), reserved=30600MB
    03_before_backward: alloc=21884MB (+27MB), reserved=30600MB
    04_after_backward: alloc=287MB (-21598MB), reserved=30600MB
    PEAK: alloc=22399MB, reserved=30600MB / 24576MB (124.5%)

    [BREAKDOWN] Largest increase at '02_after_forward' (+21639MB):
      Model params: 73.4MB
      Model grads:  52.0MB
      Optimizer:    146.7MB (estimate (Adam-like: AdamW))
      Batch data:   5.5MB
      Activations:  74.6MB
      Output tensors (top 5):
        features: 43.9MB
        support_features: 21.9MB
        all_logits: 6.0MB
        attention_maps: 1.2MB
        logits: 0.9MB
      Module params (top 5):
        solver: 28.4MB
        hyper_lora: 19.1MB
        context_encoder: 15.0MB
        context_injector: 3.0MB
        hpm: 2.9MB
      Active modules: solver_context, cross_attention, hpm
      Inactive modules: hyperlora, loo, equivariance
  Batch 4/5: loss=0.3039, focal_weighted=0.1709, batch_acc=10.3%, exact=0/250 (0.0%), running_acc=10.9%, lr=2.50e-04, hpm=0.0050
    [TaskTrack] Global_Solved: 3/342, Epoch_Solved: 0, New_Puzzles: 0
    FG: batch=5.9% run50=6.1% | BG: batch=13.2% run50=13.8%
    Per-Color: [0:18% 1:5% 2:0% 3:5% 4:5% 5:6% 6:3% 7:23% 8:4% 9:22%]
    Running50: [0:19% 1:6% 2:1% 3:7% 4:7% 5:8% 6:3% 7:20% 8:4% 9:17%]

Epoch 16 Summary:
  Total Loss: 0.4238
  Task Loss (focal): 0.2806
  [Global Task Progress] Unique Solved: 3/342 (0.9%)
    NEW puzzles solved this epoch: 0
    ⚠️ No new puzzles solved in last 5 epochs - model may be plateauing
  Entropy Loss: 4.2487 (weight=0.01)
  Sparsity Loss: 0.1639 (weight=0.5)
  Predicate Loss: 0.0000 (weight=0.01)
  HPM Balance Loss: 0.0050 (weight=0.01)
  HPM Gate Value: 0.0000 (0=no contribution, 1=full)
  HPM Instance Buffer: 2 entries
  HPM Procedural Buffer: 0 entries
  HPM Tasks Added (exact matches): 0
    (No exact matches this epoch: 0)
  Time: 35.5s, LR: 2.50e-04
    Per-module LRs: DSC:2.50e-04, MSRE:2.50e-04, Other:2.50e-04
  Temperature: 0.9493 (lower=sharper attention)
  Samples Processed: 250 (5 batches)
  Dihedral Distribution: [13.2%, 12.4%, 13.6%, 14.0%, 12.4%, 11.6%, 10.8%, 12.0%]
  Color Permutation: 0.0% (0/250)
  Translational Aug: 99.6% (249/250), unique offsets: 228
  Aug Quality: OK
  --- Training Diagnostics ---
  Solver Steps: 7 (deep supervision active)
  Per-Step Loss (epoch avg, 1 batches): [2.3281, 2.3281, 2.3125, 2.3125, 2.2969, 2.2969, 2.2969]
    [!] Best step is 4 (middle), not last - solver may be over-iterating!
    [!] Best: step 4 (2.2969), Final: step 6 (2.2969)
    [!] Potential gain from best-step selection: 0.0% lower loss
  Best-Step Histogram: []
  Solver Health: Last step best: 0.0%, Earlier step best: 100.0%
  Avg Step Improvement: 1.3% (step0→stepN)
    ⚠️ SOLVER OVER-ITERATION WARNING: 100% of batches had earlier step as best!
    ⚠️ Consider: (1) enabling best-step selection, (2) reducing num_solver_steps, (3) enabling ACT
  Grad Norms: DSC=0.0000, StopPred=0.0000, Encoder=0.0000, Solver=0.0000
  StopPred Weight Var: 2.80e-02
  Attention: max=0.0438, min=0.000000
  Stop Prob: 0.287 (approx 5.0 clues active)
  Stop Probs Std: 0.098 (global std across batch×clues)
  Clues Used: mean=4.99, std=0.39, range=[4.1, 6.0]
  Clue-Loss Correlation: -0.259 (unexpected negative - check gradient flow)
  Stop Logits: mean=-0.96, std=0.50, range=[-2.8, 0.9]
    Clue count varies by task (per-sample coupling active)
  Per-Clue Entropy: [3.88, 3.88, 3.88, 3.88, 3.88, 3.88, 3.89] (mean=3.88, max=6.80)
    [!] Clues have uniform entropy (std=0.001) - not differentiating!
  Centroid Spread: 0.07 (higher=more diverse)
    🚨 CRITICAL COLLAPSE: Spread=0.07 < 0.5 - all clues at same location!
    [!] Stop predictor cannot differentiate - needs diversity regularizer
  --- Stop Predictor Coupling ---
  Entropy Input to Stop: 0.5709 (normalized, lower=sharper)
  Per-Clue Entropy Input: [0.571, 0.571, 0.571, 0.571, 0.571, 0.571, 0.571]
  Per-Clue Stop Prob: [0.266, 0.285, 0.273, 0.309, 0.307, 0.289, 0.273]
  --- Sparsity Loss Breakdown (Per-Sample Coupled) ---
  Min Clue Penalty: 0.0000 (per-sample avg)
  Per-Sample Clue Penalty (scaled): 0.0000
  Base Pondering: 0.0999 (clues=5.00)
  Entropy Pondering: 0.0575
  --- Per-Class Distribution (Valid Pixels Only) ---
  Padding: 93.2% of grid (ignored in loss)
  Pred %: [16.1, 3.4, 0.8, 9.9, 7.9, 20.4, 1.5, 26.1, 2.7, 11.4]
  Target %: [61.1, 4.7, 9.4, 4.4, 4.7, 7.5, 1.0, 3.2, 3.4, 0.5]
  Per-Class Acc %: [19, 5, 1, 6, 8, 10, 6, 18, 5, 16]
  ==================================================
  PER-SAMPLE TRAINING ACCURACY (Epoch 16)
  ==================================================
  ★ Mean Accuracy: 10.9%
  ★ Exact Match: 0/250 (0.0%)
  ★ High Acc (≥90%): 0/250 (0.0%)
  FG Accuracy: 6.1%
  BG Accuracy: 13.8%
  Batch Trend: 10.6% → 9.3% (↓ 1.3pp)
  Accuracy Distribution: 0-25%:100%, 25-50%:0%, 50-75%:0%, 75-90%:0%, 90-100%:0%
    [!] Many samples stuck at low accuracy - check data/model!
  Running Window (last 5 batches): 10.9% ± 1.6%

  --- PER-COLOR ACCURACY (10 classes) ---
  Color:       0     1     2     3     4     5     6     7     8     9
  Acc%:      19%    5%    1%    6%    8%   10%    6%   18%    5%   16%
  Target:  58.4%  7.6%  6.0%  6.4%  3.8%  5.3%  2.7%  2.1%  6.0%  1.7%
  Pred:    17.2%  2.8%  0.7%  8.2%  7.8% 21.0%  3.0% 26.8%  3.0%  9.4%
  [!] Weak colors (<50% acc): 0(Black), 1(Blue), 2(Red), 3(Green), 4(Yellow), 5(Gray), 6(Pink), 7(Orange), 8(Cyan), 9(Brown)
  [!] Under-predicting color 1 (Blue): 2.8% vs target 7.6%
  [!] Under-predicting color 2 (Red): 0.7% vs target 6.0%
  [!] Under-predicting color 8 (Cyan): 3.0% vs target 6.0%
  ==================================================

  ############################################################
  🚦 TRAINING HEALTH CHECK - Epoch 16
  ############################################################
  ✓ Attention sharpening (0.57 < 0.7)
  ✓ Good confidence-stop coupling (r=0.54)
  ✓ No NaN/Inf issues
  ✓ No color mode collapse
  ⚠ Stop probs nearly uniform (global=0.098, per_clue=0.016)
  ⚠ Loss slowly decreasing (0.311 → 0.281)
  ✗ Centroid COLLAPSE (0.1 < 0.5)
  ✗ Accuracy not improving (19.9% → 10.9%)
  --------------------------------------------------------
  RESULT: 4/8 checks passed
  STATUS: 🟠 WARNING
  → Multiple issues detected. Consider intervention soon.

  RECOMMENDED ACTIONS:
  ############################################################
  HPM Instance Buffer: 2 entries SAVED
    → Saved to: checkpoints\rlan_stable_merged\hpm\instance_buffer.pt
  HPM Procedural Buffer: 0 entries (empty, not saved)
  Saved checkpoint to checkpoints\rlan_stable_merged\latest.pt
  [ARCDataset] Augmentation config updated:
    dihedral=True, color_perm=False (prob=0.50), translational=True

Epoch 17/200
----------------------------------------
  [Batch 0] Grid size: 30x30 (batch_max_size=30)

  [MEMORY] Epoch 16 Batch 0:
    Baseline: alloc=92MB, reserved=30600MB
    01_batch_on_gpu: alloc=106MB (+14MB), reserved=30622MB
    02_after_forward: alloc=23324MB (+23218MB), reserved=30622MB
    03_before_backward: alloc=23371MB (+47MB), reserved=30622MB
    04_after_backward: alloc=240MB (-23131MB), reserved=30622MB
    PEAK: alloc=23967MB, reserved=30622MB / 24576MB (124.6%)

    [BREAKDOWN] Largest increase at '02_after_forward' (+23218MB):
      Model params: 73.4MB
      Model grads:  0.0MB
      Optimizer:    146.7MB (estimate (Adam-like: AdamW))
      Batch data:   6.2MB
      Activations:  77.8MB
      Output tensors (top 5):
        features: 43.9MB
        support_features: 25.0MB
        all_logits: 6.0MB
        attention_maps: 1.2MB
        logits: 0.9MB
      Module params (top 5):
        solver: 28.4MB
        hyper_lora: 19.1MB
        context_encoder: 15.0MB
        context_injector: 3.0MB
        hpm: 2.9MB
      Active modules: solver_context, cross_attention, hpm
      Inactive modules: hyperlora, loo, equivariance

  [MEMORY] First forward pass with staged modules active:
  [MEMORY] Batch 0: alloc=240MB, reserved=30622MB, headroom=-6046MB
  [WARNING] >95% GPU memory used! Training may slow due to shared memory.
  Batch 0/5: loss=0.4846, focal_weighted=0.3351, batch_acc=12.5%, exact=0/50 (0.0%), running_acc=12.5%, lr=2.50e-04, hpm=0.0050
    [TaskTrack] Global_Solved: 3/342, Epoch_Solved: 0, New_Puzzles: 0
    FG: batch=5.7% run50=5.7% | BG: batch=16.7% run50=16.7%
    Per-Color: [0:22% 1:5% 2:0% 3:8% 4:5% 5:10% 6:7% 7:28% 8:8% 9:1%]
    Running50: [0:22% 1:5% 2:0% 3:8% 4:5% 5:10% 6:7% 7:28% 8:8% 9:1%]
    Solver: [2.703, 2.703, 2.703, 2.703, 2.703, 2.688, 2.688] ⚠ best=5
  [Batch 1] Grid size: 30x30 (batch_max_size=30)

  [MEMORY] Epoch 16 Batch 1:
    Baseline: alloc=92MB, reserved=30600MB
    01_batch_on_gpu: alloc=221MB (+129MB), reserved=30624MB
    02_after_forward: alloc=26574MB (+26353MB), reserved=30624MB
    03_before_backward: alloc=26600MB (+26MB), reserved=30624MB
    04_after_backward: alloc=301MB (-26299MB), reserved=30624MB
    PEAK: alloc=27419MB, reserved=30624MB / 24576MB (124.6%)

    [BREAKDOWN] Largest increase at '02_after_forward' (+26353MB):
      Model params: 73.4MB
      Model grads:  52.0MB
      Optimizer:    146.7MB (estimate (Adam-like: AdamW))
      Batch data:   7.6MB
      Activations:  84.1MB
      Output tensors (top 5):
        features: 43.9MB
        support_features: 31.2MB
        all_logits: 6.0MB
        attention_maps: 1.2MB
        logits: 0.9MB
      Module params (top 5):
        solver: 28.4MB
        hyper_lora: 19.1MB
        context_encoder: 15.0MB
        context_injector: 3.0MB
        hpm: 2.9MB
      Active modules: solver_context, cross_attention, hpm
      Inactive modules: hyperlora, loo, equivariance
  [MEMORY] Batch 1: alloc=301MB, reserved=30624MB, headroom=-6048MB
  [WARNING] >95% GPU memory used! Training may slow due to shared memory.
  Batch 1/5: loss=0.2753, focal_weighted=0.1390, batch_acc=11.1%, exact=1/100 (1.0%), running_acc=11.8%, lr=2.50e-04, hpm=0.0051
    [TaskTrack] Global_Solved: 3/342, Epoch_Solved: 1, New_Puzzles: 0
    FG: batch=7.6% run50=6.7% | BG: batch=11.5% run50=14.1%
    Per-Color: [0:16% 1:2% 2:0% 3:10% 4:14% 5:10% 6:16% 7:22% 8:2% 9:15%]
    Running50: [0:19% 1:4% 2:0% 3:9% 4:10% 5:10% 6:11% 7:25% 8:5% 9:8%]
  [Batch 2] Grid size: 30x30 (batch_max_size=30)

  [MEMORY] Epoch 16 Batch 2:
    Baseline: alloc=92MB, reserved=30600MB
    01_batch_on_gpu: alloc=223MB (+131MB), reserved=30624MB
    02_after_forward: alloc=21863MB (+21640MB), reserved=30624MB
    03_before_backward: alloc=21889MB (+26MB), reserved=30624MB
    04_after_backward: alloc=292MB (-21597MB), reserved=30624MB
    PEAK: alloc=22405MB, reserved=30624MB / 24576MB (124.6%)

    [BREAKDOWN] Largest increase at '02_after_forward' (+21640MB):
      Model params: 73.4MB
      Model grads:  52.0MB
      Optimizer:    146.7MB (estimate (Adam-like: AdamW))
      Batch data:   5.5MB
      Activations:  74.6MB
      Output tensors (top 5):
        features: 43.9MB
        support_features: 21.9MB
        all_logits: 6.0MB
        attention_maps: 1.2MB
        logits: 0.9MB
      Module params (top 5):
        solver: 28.4MB
        hyper_lora: 19.1MB
        context_encoder: 15.0MB
        context_injector: 3.0MB
        hpm: 2.9MB
      Active modules: solver_context, cross_attention, hpm
      Inactive modules: hyperlora, loo, equivariance
  [MEMORY] Batch 2: alloc=292MB, reserved=30624MB, headroom=-6048MB
  [WARNING] >95% GPU memory used! Training may slow due to shared memory.
  Batch 2/5: loss=0.2857, focal_weighted=0.1544, batch_acc=10.9%, exact=1/150 (0.7%), running_acc=11.5%, lr=2.50e-04, hpm=0.0050
    [TaskTrack] Global_Solved: 3/342, Epoch_Solved: 1, New_Puzzles: 0
    FG: batch=6.4% run50=6.6% | BG: batch=13.3% run50=13.8%
    Per-Color: [0:17% 1:7% 2:1% 3:7% 4:7% 5:12% 6:0% 7:23% 8:2% 9:25%]
    Running50: [0:18% 1:5% 2:0% 3:8% 4:9% 5:10% 6:7% 7:24% 8:4% 9:14%]
  [Batch 3] Grid size: 30x30 (batch_max_size=30)

  [MEMORY] Epoch 16 Batch 3:
    Baseline: alloc=92MB, reserved=30600MB
    01_batch_on_gpu: alloc=220MB (+128MB), reserved=30624MB
    02_after_forward: alloc=20296MB (+20076MB), reserved=30624MB
    03_before_backward: alloc=20324MB (+28MB), reserved=30624MB
    04_after_backward: alloc=288MB (-20036MB), reserved=30624MB
    PEAK: alloc=20738MB, reserved=30624MB / 24576MB (124.6%)

    [BREAKDOWN] Largest increase at '02_after_forward' (+20076MB):
      Model params: 73.4MB
      Model grads:  52.0MB
      Optimizer:    146.7MB (estimate (Adam-like: AdamW))
      Batch data:   4.8MB
      Activations:  71.5MB
      Output tensors (top 5):
        features: 43.9MB
        support_features: 18.8MB
        all_logits: 6.0MB
        attention_maps: 1.2MB
        logits: 0.9MB
      Module params (top 5):
        solver: 28.4MB
        hyper_lora: 19.1MB
        context_encoder: 15.0MB
        context_injector: 3.0MB
        hpm: 2.9MB
      Active modules: solver_context, cross_attention, hpm
      Inactive modules: hyperlora, loo, equivariance
  Batch 3/5: loss=0.2944, focal_weighted=0.1669, batch_acc=9.3%, exact=1/200 (0.5%), running_acc=10.9%, lr=2.50e-04, hpm=0.0050
    [TaskTrack] Global_Solved: 3/342, Epoch_Solved: 1, New_Puzzles: 0
    FG: batch=5.5% run50=6.3% | BG: batch=10.7% run50=13.1%
    Per-Color: [0:17% 1:5% 2:0% 3:12% 4:6% 5:10% 6:2% 7:18% 8:4% 9:13%]
    Running50: [0:18% 1:5% 2:0% 3:9% 4:8% 5:10% 6:6% 7:23% 8:4% 9:14%]
  [Batch 4] Grid size: 30x30 (batch_max_size=30)

  [MEMORY] Epoch 16 Batch 4:
    Baseline: alloc=92MB, reserved=30600MB
    01_batch_on_gpu: alloc=214MB (+122MB), reserved=30624MB
    02_after_forward: alloc=18720MB (+18506MB), reserved=30624MB
    03_before_backward: alloc=18797MB (+77MB), reserved=30624MB
    04_after_backward: alloc=278MB (-18518MB), reserved=30624MB
    PEAK: alloc=19095MB, reserved=30624MB / 24576MB (124.6%)

    [BREAKDOWN] Largest increase at '02_after_forward' (+18506MB):
      Model params: 73.4MB
      Model grads:  52.0MB
      Optimizer:    146.7MB (estimate (Adam-like: AdamW))
      Batch data:   4.1MB
      Activations:  68.3MB
      Output tensors (top 5):
        features: 43.9MB
        support_features: 15.6MB
        all_logits: 6.0MB
        attention_maps: 1.2MB
        logits: 0.9MB
      Module params (top 5):
        solver: 28.4MB
        hyper_lora: 19.1MB
        context_encoder: 15.0MB
        context_injector: 3.0MB
        hpm: 2.9MB
      Active modules: solver_context, cross_attention, hpm
      Inactive modules: hyperlora, loo, equivariance
  Batch 4/5: loss=0.7113, focal_weighted=0.5454, batch_acc=14.4%, exact=1/250 (0.4%), running_acc=11.6%, lr=2.50e-04, hpm=0.0050
    [TaskTrack] Global_Solved: 3/342, Epoch_Solved: 1, New_Puzzles: 0
    FG: batch=7.3% run50=6.5% | BG: batch=20.0% run50=14.4%
    Per-Color: [0:24% 1:1% 2:2% 3:4% 4:11% 5:8% 6:11% 7:14% 8:7% 9:14%]
    Running50: [0:19% 1:4% 2:0% 3:8% 4:9% 5:10% 6:7% 7:21% 8:4% 9:14%]

Epoch 17 Summary:
  Total Loss: 0.4102
  Task Loss (focal): 0.2682
  [Global Task Progress] Unique Solved: 3/342 (0.9%)
    NEW puzzles solved this epoch: 0
    ⚠️ No new puzzles solved in last 5 epochs - model may be plateauing
  Entropy Loss: 4.2491 (weight=0.01)
  Sparsity Loss: 0.1615 (weight=0.5)
  Predicate Loss: 0.0000 (weight=0.01)
  HPM Balance Loss: 0.0050 (weight=0.01)
  HPM Gate Value: 0.0000 (0=no contribution, 1=full)
  HPM Instance Buffer: 2 entries
  HPM Procedural Buffer: 0 entries
  HPM Tasks Added (exact matches): 0
    (No exact matches this epoch: 1)
  Time: 38.7s, LR: 2.50e-04
    Per-module LRs: DSC:2.50e-04, MSRE:2.50e-04, Other:2.50e-04
  Temperature: 0.9461 (lower=sharper attention)
  Samples Processed: 250 (5 batches)
  Dihedral Distribution: [15.6%, 10.8%, 11.6%, 13.2%, 9.6%, 14.8%, 14.8%, 9.6%]
  Color Permutation: 0.0% (0/250)
  Translational Aug: 99.2% (248/250), unique offsets: 230
  Aug Quality: OK
  --- Training Diagnostics ---
  Solver Steps: 7 (deep supervision active)
  Per-Step Loss (epoch avg, 1 batches): [2.7031, 2.7031, 2.7031, 2.7031, 2.7031, 2.6875, 2.6875]
    [!] Best step is 5 (middle), not last - solver may be over-iterating!
    [!] Best: step 5 (2.6875), Final: step 6 (2.6875)
    [!] Potential gain from best-step selection: 0.0% lower loss
  Best-Step Histogram: []
  Solver Health: Last step best: 0.0%, Earlier step best: 100.0%
  Avg Step Improvement: 0.6% (step0→stepN)
    ⚠️ SOLVER OVER-ITERATION WARNING: 100% of batches had earlier step as best!
    ⚠️ Consider: (1) enabling best-step selection, (2) reducing num_solver_steps, (3) enabling ACT
  Grad Norms: DSC=0.0000, StopPred=0.0000, Encoder=0.0000, Solver=0.0000
  StopPred Weight Var: 2.80e-02
  Attention: max=0.0216, min=0.000000
  Stop Prob: 0.279 (approx 5.0 clues active)
  Stop Probs Std: 0.097 (global std across batch×clues)
  Clues Used: mean=5.04, std=0.43, range=[3.4, 5.7]
  Clue-Loss Correlation: -0.286 (unexpected negative - check gradient flow)
  Stop Logits: mean=-0.99, std=0.48, range=[-2.4, 1.0]
    Clue count varies by task (per-sample coupling active)
  Per-Clue Entropy: [4.59, 4.59, 4.59, 4.59, 4.59, 4.59, 4.59] (mean=4.59, max=6.80)
    [!] Clues have uniform entropy (std=0.001) - not differentiating!
  Centroid Spread: 0.10 (higher=more diverse)
    🚨 CRITICAL COLLAPSE: Spread=0.10 < 0.5 - all clues at same location!
    [!] Stop predictor cannot differentiate - needs diversity regularizer
  --- Stop Predictor Coupling ---
  Entropy Input to Stop: 0.6749 (normalized, lower=sharper)
  Per-Clue Entropy Input: [0.675, 0.675, 0.675, 0.675, 0.675, 0.675, 0.675]
  Per-Clue Stop Prob: [0.287, 0.301, 0.271, 0.268, 0.268, 0.293, 0.273]
  --- Sparsity Loss Breakdown (Per-Sample Coupled) ---
  Min Clue Penalty: 0.0000 (per-sample avg)
  Per-Sample Clue Penalty (scaled): 0.0000
  Base Pondering: 0.1008 (clues=5.04)
  Entropy Pondering: 0.0686
  --- Per-Class Distribution (Valid Pixels Only) ---
  Padding: 87.6% of grid (ignored in loss)
  Pred %: [18.8, 2.1, 0.7, 7.5, 7.9, 21.5, 1.5, 27.6, 2.9, 9.6]
  Target %: [62.8, 4.5, 4.1, 5.9, 4.1, 2.6, 4.5, 1.2, 9.2, 1.2]
  Per-Class Acc %: [21, 3, 1, 7, 9, 9, 8, 21, 7, 12]
  ==================================================
  PER-SAMPLE TRAINING ACCURACY (Epoch 17)
  ==================================================
  ★ Mean Accuracy: 11.6%
  ★ Exact Match: 1/250 (0.4%)
  ★ High Acc (≥90%): 1/250 (0.4%)
  FG Accuracy: 6.5%
  BG Accuracy: 14.4%
  Batch Trend: 12.5% → 11.9% (→ 0.7pp)
  Accuracy Distribution: 0-25%:100%, 25-50%:0%, 50-75%:0%, 75-90%:0%, 90-100%:0%
    [!] Many samples stuck at low accuracy - check data/model!
  Running Window (last 5 batches): 11.6% ± 1.7%

  --- PER-COLOR ACCURACY (10 classes) ---
  Color:       0     1     2     3     4     5     6     7     8     9
  Acc%:      21%    3%    1%    7%    9%    9%    8%   21%    7%   12%
  Target:  59.9%  7.3%  5.0%  5.9%  4.6%  4.1%  3.1%  1.4%  6.4%  2.3%
  Pred:    18.6%  2.2%  0.7%  7.9%  7.4% 21.4%  2.4% 26.7%  3.1%  9.5%
  [!] Weak colors (<50% acc): 0(Black), 1(Blue), 2(Red), 3(Green), 4(Yellow), 5(Gray), 6(Pink), 7(Orange), 8(Cyan), 9(Brown)
  [!] Under-predicting color 1 (Blue): 2.2% vs target 7.3%
  [!] Under-predicting color 2 (Red): 0.7% vs target 5.0%
  [!] Under-predicting color 8 (Cyan): 3.1% vs target 6.4%
  ==================================================

  ############################################################
  🚦 TRAINING HEALTH CHECK - Epoch 17
  ############################################################
  ✓ Attention sharpening (0.68 < 0.7)
  ✓ No NaN/Inf issues
  ✓ No color mode collapse
  ⚠ Stop probs nearly uniform (global=0.097, per_clue=0.012)
  ⚠ Loss slowly decreasing (0.311 → 0.268)
  ✗ Centroid COLLAPSE (0.1 < 0.5)
  ✗ No confidence-stop coupling (r=-0.05)
  ✗ Accuracy not improving (19.9% → 11.6%)
  --------------------------------------------------------
  RESULT: 3/8 checks passed
  STATUS: 🔴 UNHEALTHY
  → Training likely failing. STOP and investigate!

  RECOMMENDED ACTIONS:
  ############################################################
  HPM Instance Buffer: 2 entries SAVED
    → Saved to: checkpoints\rlan_stable_merged\hpm\instance_buffer.pt
  HPM Procedural Buffer: 0 entries (empty, not saved)
  Saved checkpoint to checkpoints\rlan_stable_merged\latest.pt
  [ARCDataset] Augmentation config updated:
    dihedral=True, color_perm=False (prob=0.50), translational=True

Epoch 18/200
----------------------------------------
  [LR] Restored to original after 2 epochs recovery
  [Batch 0] Grid size: 30x30 (batch_max_size=30)

  [MEMORY] Epoch 17 Batch 0:
    Baseline: alloc=92MB, reserved=30624MB
    01_batch_on_gpu: alloc=101MB (+8MB), reserved=30646MB
    02_after_forward: alloc=18613MB (+18512MB), reserved=30646MB
    03_before_backward: alloc=18663MB (+50MB), reserved=30646MB
    04_after_backward: alloc=224MB (-18439MB), reserved=30646MB
    PEAK: alloc=18988MB, reserved=30646MB / 24576MB (124.7%)

    [BREAKDOWN] Largest increase at '02_after_forward' (+18512MB):
      Model params: 73.4MB
      Model grads:  0.0MB
      Optimizer:    146.7MB (estimate (Adam-like: AdamW))
      Batch data:   4.1MB
      Activations:  68.3MB
      Output tensors (top 5):
        features: 43.9MB
        support_features: 15.6MB
        all_logits: 6.0MB
        attention_maps: 1.2MB
        logits: 0.9MB
      Module params (top 5):
        solver: 28.4MB
        hyper_lora: 19.1MB
        context_encoder: 15.0MB
        context_injector: 3.0MB
        hpm: 2.9MB
      Active modules: solver_context, cross_attention, hpm
      Inactive modules: hyperlora, loo, equivariance

  [MEMORY] First forward pass with staged modules active:
  [MEMORY] Batch 0: alloc=224MB, reserved=30646MB, headroom=-6070MB
  [WARNING] >95% GPU memory used! Training may slow due to shared memory.
  Batch 0/5: loss=0.5093, focal_weighted=0.3618, batch_acc=13.5%, exact=0/50 (0.0%), running_acc=13.5%, lr=5.00e-04, hpm=0.0051
    [TaskTrack] Global_Solved: 3/342, Epoch_Solved: 0, New_Puzzles: 0
    FG: batch=7.7% run50=7.7% | BG: batch=15.8% run50=15.8%
    Per-Color: [0:21% 1:5% 2:0% 3:7% 4:7% 5:18% 6:4% 7:25% 8:8% 9:8%]
    Running50: [0:21% 1:5% 2:0% 3:7% 4:7% 5:18% 6:4% 7:25% 8:8% 9:8%]
    Solver: [2.797, 2.797, 2.797, 2.781, 2.781, 2.781, 2.781] ⚠ best=3
  [Batch 1] Grid size: 30x30 (batch_max_size=30)

  [MEMORY] Epoch 17 Batch 1:
    Baseline: alloc=92MB, reserved=30624MB
    01_batch_on_gpu: alloc=216MB (+123MB), reserved=30648MB
    02_after_forward: alloc=18722MB (+18506MB), reserved=30648MB
    03_before_backward: alloc=18805MB (+83MB), reserved=30648MB
    04_after_backward: alloc=281MB (-18524MB), reserved=30648MB
    PEAK: alloc=19097MB, reserved=30648MB / 24576MB (124.7%)

    [BREAKDOWN] Largest increase at '02_after_forward' (+18506MB):
      Model params: 73.4MB
      Model grads:  52.0MB
      Optimizer:    146.7MB (estimate (Adam-like: AdamW))
      Batch data:   4.1MB
      Activations:  68.3MB
      Output tensors (top 5):
        features: 43.9MB
        support_features: 15.6MB
        all_logits: 6.0MB
        attention_maps: 1.2MB
        logits: 0.9MB
      Module params (top 5):
        solver: 28.4MB
        hyper_lora: 19.1MB
        context_encoder: 15.0MB
        context_injector: 3.0MB
        hpm: 2.9MB
      Active modules: solver_context, cross_attention, hpm
      Inactive modules: hyperlora, loo, equivariance
  [MEMORY] Batch 1: alloc=281MB, reserved=30648MB, headroom=-6072MB
  [WARNING] >95% GPU memory used! Training may slow due to shared memory.
  Batch 1/5: loss=0.8976, focal_weighted=0.7359, batch_acc=13.0%, exact=0/100 (0.0%), running_acc=13.2%, lr=5.00e-04, hpm=0.0051
    [TaskTrack] Global_Solved: 3/342, Epoch_Solved: 0, New_Puzzles: 0
    FG: batch=6.3% run50=7.0% | BG: batch=18.7% run50=17.2%
    Per-Color: [0:22% 1:2% 2:2% 3:6% 4:8% 5:8% 6:9% 7:9% 8:7% 9:7%]
    Running50: [0:21% 1:4% 2:1% 3:6% 4:7% 5:13% 6:6% 7:17% 8:7% 9:7%]
  [Batch 2] Grid size: 30x30 (batch_max_size=30)

  [MEMORY] Epoch 17 Batch 2:
    Baseline: alloc=92MB, reserved=30624MB
    01_batch_on_gpu: alloc=218MB (+126MB), reserved=30648MB
    02_after_forward: alloc=23432MB (+23213MB), reserved=30648MB
    03_before_backward: alloc=23458MB (+27MB), reserved=30648MB
    04_after_backward: alloc=294MB (-23165MB), reserved=30648MB
    PEAK: alloc=24075MB, reserved=30648MB / 24576MB (124.7%)

    [BREAKDOWN] Largest increase at '02_after_forward' (+23213MB):
      Model params: 73.4MB
      Model grads:  52.0MB
      Optimizer:    146.7MB (estimate (Adam-like: AdamW))
      Batch data:   6.2MB
      Activations:  77.8MB
      Output tensors (top 5):
        features: 43.9MB
        support_features: 25.0MB
        all_logits: 6.0MB
        attention_maps: 1.2MB
        logits: 0.9MB
      Module params (top 5):
        solver: 28.4MB
        hyper_lora: 19.1MB
        context_encoder: 15.0MB
        context_injector: 3.0MB
        hpm: 2.9MB
      Active modules: solver_context, cross_attention, hpm
      Inactive modules: hyperlora, loo, equivariance
  [MEMORY] Batch 2: alloc=294MB, reserved=30648MB, headroom=-6072MB
  [WARNING] >95% GPU memory used! Training may slow due to shared memory.
  Batch 2/5: loss=0.2677, focal_weighted=0.1365, batch_acc=9.6%, exact=0/150 (0.0%), running_acc=12.0%, lr=5.00e-04, hpm=0.0051
    [TaskTrack] Global_Solved: 3/342, Epoch_Solved: 0, New_Puzzles: 0
    FG: batch=4.3% run50=6.1% | BG: batch=13.3% run50=15.9%
    Per-Color: [0:18% 1:3% 2:0% 3:6% 4:6% 5:5% 6:3% 7:20% 8:3% 9:0%]
    Running50: [0:20% 1:4% 2:1% 3:6% 4:7% 5:10% 6:5% 7:18% 8:6% 9:5%]
  [Batch 3] Grid size: 30x30 (batch_max_size=30)

  [MEMORY] Epoch 17 Batch 3:
    Baseline: alloc=92MB, reserved=30624MB
    01_batch_on_gpu: alloc=222MB (+130MB), reserved=30648MB
    02_after_forward: alloc=20297MB (+20075MB), reserved=30648MB
    03_before_backward: alloc=20325MB (+28MB), reserved=30648MB
    04_after_backward: alloc=289MB (-20036MB), reserved=30648MB
    PEAK: alloc=20738MB, reserved=30648MB / 24576MB (124.7%)

    [BREAKDOWN] Largest increase at '02_after_forward' (+20075MB):
      Model params: 73.4MB
      Model grads:  52.0MB
      Optimizer:    146.7MB (estimate (Adam-like: AdamW))
      Batch data:   4.8MB
      Activations:  71.5MB
      Output tensors (top 5):
        features: 43.9MB
        support_features: 18.8MB
        all_logits: 6.0MB
        attention_maps: 1.2MB
        logits: 0.9MB
      Module params (top 5):
        solver: 28.4MB
        hyper_lora: 19.1MB
        context_encoder: 15.0MB
        context_injector: 3.0MB
        hpm: 2.9MB
      Active modules: solver_context, cross_attention, hpm
      Inactive modules: hyperlora, loo, equivariance
  Batch 3/5: loss=0.3141, focal_weighted=0.1793, batch_acc=11.3%, exact=0/200 (0.0%), running_acc=11.8%, lr=5.00e-04, hpm=0.0051
    [TaskTrack] Global_Solved: 3/342, Epoch_Solved: 0, New_Puzzles: 0
    FG: batch=8.7% run50=6.8% | BG: batch=10.5% run50=14.6%
    Per-Color: [0:16% 1:8% 2:0% 3:13% 4:11% 5:6% 6:0% 7:29% 8:2% 9:20%]
    Running50: [0:19% 1:5% 2:0% 3:8% 4:8% 5:9% 6:4% 7:21% 8:5% 9:9%]
  [Batch 4] Grid size: 30x30 (batch_max_size=30)

  [MEMORY] Epoch 17 Batch 4:
    Baseline: alloc=92MB, reserved=30624MB
    01_batch_on_gpu: alloc=216MB (+123MB), reserved=30648MB
    02_after_forward: alloc=26571MB (+26355MB), reserved=30648MB
    03_before_backward: alloc=26595MB (+25MB), reserved=30648MB
    04_after_backward: alloc=297MB (-26298MB), reserved=30648MB
    PEAK: alloc=27416MB, reserved=30648MB / 24576MB (124.7%)

    [BREAKDOWN] Largest increase at '02_after_forward' (+26355MB):
      Model params: 73.4MB
      Model grads:  52.0MB
      Optimizer:    146.7MB (estimate (Adam-like: AdamW))
      Batch data:   7.6MB
      Activations:  84.1MB
      Output tensors (top 5):
        features: 43.9MB
        support_features: 31.2MB
        all_logits: 6.0MB
        attention_maps: 1.2MB
        logits: 0.9MB
      Module params (top 5):
        solver: 28.4MB
        hyper_lora: 19.1MB
        context_encoder: 15.0MB
        context_injector: 3.0MB
        hpm: 2.9MB
      Active modules: solver_context, cross_attention, hpm
      Inactive modules: hyperlora, loo, equivariance
  Batch 4/5: loss=0.2800, focal_weighted=0.1489, batch_acc=12.2%, exact=1/250 (0.4%), running_acc=11.9%, lr=5.00e-04, hpm=0.0051
    [TaskTrack] Global_Solved: 3/342, Epoch_Solved: 1, New_Puzzles: 0
    FG: batch=11.0% run50=7.6% | BG: batch=8.5% run50=13.4%
    Per-Color: [0:13% 1:3% 2:1% 3:11% 4:13% 5:4% 6:3% 7:26% 8:8% 9:19%]
    Running50: [0:18% 1:4% 2:1% 3:9% 4:9% 5:8% 6:4% 7:22% 8:6% 9:11%]

Epoch 18 Summary:
  Total Loss: 0.4538
  Task Loss (focal): 0.3125
  [Global Task Progress] Unique Solved: 3/342 (0.9%)
    NEW puzzles solved this epoch: 0
    ⚠️ No new puzzles solved in last 5 epochs - model may be plateauing
  Entropy Loss: 4.2300 (weight=0.01)
  Sparsity Loss: 0.1601 (weight=0.5)
  Predicate Loss: 0.0000 (weight=0.01)
  HPM Balance Loss: 0.0051 (weight=0.01)
  HPM Gate Value: 0.0000 (0=no contribution, 1=full)
  HPM Instance Buffer: 2 entries
  HPM Procedural Buffer: 0 entries
  HPM Tasks Added (exact matches): 0
    (No exact matches this epoch: 1)
  Time: 35.1s, LR: 5.00e-04
    Per-module LRs: DSC:5.00e-04, MSRE:5.00e-04, Other:5.00e-04
  Temperature: 0.9428 (lower=sharper attention)
  Samples Processed: 250 (5 batches)
  Dihedral Distribution: [9.6%, 14.0%, 14.8%, 12.4%, 13.6%, 12.0%, 14.0%, 9.6%]
  Color Permutation: 0.0% (0/250)
  Translational Aug: 100.0% (250/250), unique offsets: 236
  Aug Quality: OK
  --- Training Diagnostics ---
  Solver Steps: 7 (deep supervision active)
  Per-Step Loss (epoch avg, 1 batches): [2.7969, 2.7969, 2.7969, 2.7812, 2.7812, 2.7812, 2.7812]
    [!] Best step is 3 (middle), not last - solver may be over-iterating!
    [!] Best: step 3 (2.7812), Final: step 6 (2.7812)
    [!] Potential gain from best-step selection: 0.0% lower loss
  Best-Step Histogram: []
  Solver Health: Last step best: 0.0%, Earlier step best: 100.0%
  Avg Step Improvement: 0.6% (step0→stepN)
    ⚠️ SOLVER OVER-ITERATION WARNING: 100% of batches had earlier step as best!
    ⚠️ Consider: (1) enabling best-step selection, (2) reducing num_solver_steps, (3) enabling ACT
  Grad Norms: DSC=0.0000, StopPred=0.0000, Encoder=0.0000, Solver=0.0000
  StopPred Weight Var: 2.80e-02
  Attention: max=0.0191, min=0.000000
  Stop Prob: 0.307 (approx 4.9 clues active)
  Stop Probs Std: 0.101 (global std across batch×clues)
  Clues Used: mean=4.85, std=0.36, range=[3.9, 5.7]
  Clue-Loss Correlation: -0.242 (unexpected negative - check gradient flow)
  Stop Logits: mean=-0.86, std=0.49, range=[-2.1, 0.7]
    Clue count varies by task (per-sample coupling active)
  Per-Clue Entropy: [4.67, 4.67, 4.67, 4.67, 4.67, 4.67, 4.67] (mean=4.67, max=6.80)
    [!] Clues have uniform entropy (std=0.002) - not differentiating!
  Centroid Spread: 0.12 (higher=more diverse)
    🚨 CRITICAL COLLAPSE: Spread=0.12 < 0.5 - all clues at same location!
    [!] Stop predictor cannot differentiate - needs diversity regularizer
  --- Stop Predictor Coupling ---
  Entropy Input to Stop: 0.6869 (normalized, lower=sharper)
  Per-Clue Entropy Input: [0.687, 0.687, 0.687, 0.687, 0.686, 0.687, 0.687]
  Per-Clue Stop Prob: [0.312, 0.336, 0.275, 0.314, 0.307, 0.307, 0.301]
  --- Sparsity Loss Breakdown (Per-Sample Coupled) ---
  Min Clue Penalty: 0.0000 (per-sample avg)
  Per-Sample Clue Penalty (scaled): 0.0000
  Base Pondering: 0.0969 (clues=4.85)
  Entropy Pondering: 0.0668
  --- Per-Class Distribution (Valid Pixels Only) ---
  Padding: 87.0% of grid (ignored in loss)
  Pred %: [17.0, 2.3, 0.5, 7.5, 7.6, 22.0, 1.8, 28.6, 3.7, 9.0]
  Target %: [59.1, 4.1, 5.5, 6.0, 3.1, 5.0, 3.6, 1.9, 10.5, 1.3]
    [!] Missing foreground colors: [2]
  Per-Class Acc %: [20, 4, 1, 7, 8, 11, 5, 19, 7, 9]
  ==================================================
  PER-SAMPLE TRAINING ACCURACY (Epoch 18)
  ==================================================
  ★ Mean Accuracy: 11.9%
  ★ Exact Match: 1/250 (0.4%)
  ★ High Acc (≥90%): 1/250 (0.4%)
  FG Accuracy: 7.6%
  BG Accuracy: 13.4%
  Batch Trend: 13.5% → 11.7% (↓ 1.7pp)
  Accuracy Distribution: 0-25%:100%, 25-50%:0%, 50-75%:0%, 75-90%:0%, 90-100%:0%
    [!] Many samples stuck at low accuracy - check data/model!
  Running Window (last 5 batches): 11.9% ± 1.4%

  --- PER-COLOR ACCURACY (10 classes) ---
  Color:       0     1     2     3     4     5     6     7     8     9
  Acc%:      20%    4%    1%    7%    8%   11%    5%   19%    7%    9%
  Target:  58.6%  6.8%  5.4%  5.9%  3.6%  5.6%  3.0%  2.2%  7.4%  1.5%
  Pred:    17.0%  2.8%  0.6%  8.4%  8.0% 20.8%  2.4% 26.9%  3.4%  9.7%
  [!] Weak colors (<50% acc): 0(Black), 1(Blue), 2(Red), 3(Green), 4(Yellow), 5(Gray), 6(Pink), 7(Orange), 8(Cyan), 9(Brown)
  [!] Under-predicting color 1 (Blue): 2.8% vs target 6.8%
  [!] Under-predicting color 2 (Red): 0.6% vs target 5.4%
  [!] Under-predicting color 8 (Cyan): 3.4% vs target 7.4%
  ==================================================

  ############################################################
  🚦 TRAINING HEALTH CHECK - Epoch 18
  ############################################################
  ✓ Attention sharpening (0.69 < 0.7)
  ✓ Stop probs adapting (global=0.101, per_clue=0.017)
  ✓ No NaN/Inf issues
  ✓ No color mode collapse
  ⚠ Weak confidence-stop coupling (r=0.12)
  ✗ Centroid COLLAPSE (0.1 < 0.5)
  ✗ Loss not decreasing (0.311 → 0.312)
  ✗ Accuracy not improving (19.9% → 11.9%)
  --------------------------------------------------------
  RESULT: 4/8 checks passed
  STATUS: 🔴 UNHEALTHY
  → Training likely failing. STOP and investigate!

  RECOMMENDED ACTIONS:
    - Reduce learning rate or check data pipeline
  ############################################################
  HPM Instance Buffer: 2 entries SAVED
    → Saved to: checkpoints\rlan_stable_merged\hpm\instance_buffer.pt
  HPM Procedural Buffer: 0 entries (empty, not saved)
  Saved checkpoint to checkpoints\rlan_stable_merged\latest.pt
  [ARCDataset] Augmentation config updated:
    dihedral=True, color_perm=False (prob=0.50), translational=True

Epoch 19/200
----------------------------------------
  [Batch 0] Grid size: 30x30 (batch_max_size=30)

  [MEMORY] Epoch 18 Batch 0:
    Baseline: alloc=92MB, reserved=30648MB
    01_batch_on_gpu: alloc=105MB (+12MB), reserved=30670MB
    02_after_forward: alloc=26463MB (+26358MB), reserved=30670MB
    03_before_backward: alloc=26491MB (+28MB), reserved=30670MB
    04_after_backward: alloc=245MB (-26247MB), reserved=30670MB
    PEAK: alloc=27307MB, reserved=30670MB / 24576MB (124.8%)

    [BREAKDOWN] Largest increase at '02_after_forward' (+26358MB):
      Model params: 73.4MB
      Model grads:  0.0MB
      Optimizer:    146.7MB (estimate (Adam-like: AdamW))
      Batch data:   7.6MB
      Activations:  84.1MB
      Output tensors (top 5):
        features: 43.9MB
        support_features: 31.2MB
        all_logits: 6.0MB
        attention_maps: 1.2MB
        logits: 0.9MB
      Module params (top 5):
        solver: 28.4MB
        hyper_lora: 19.1MB
        context_encoder: 15.0MB
        context_injector: 3.0MB
        hpm: 2.9MB
      Active modules: solver_context, cross_attention, hpm
      Inactive modules: hyperlora, loo, equivariance

  [MEMORY] First forward pass with staged modules active:
  [MEMORY] Batch 0: alloc=245MB, reserved=30670MB, headroom=-6094MB
  [WARNING] >95% GPU memory used! Training may slow due to shared memory.
  Batch 0/5: loss=0.2906, focal_weighted=0.1557, batch_acc=11.7%, exact=0/50 (0.0%), running_acc=11.7%, lr=5.00e-04, hpm=0.0051
    [TaskTrack] Global_Solved: 3/342, Epoch_Solved: 0, New_Puzzles: 0
    FG: batch=8.0% run50=8.0% | BG: batch=13.2% run50=13.2%
    Per-Color: [0:18% 1:2% 2:0% 3:11% 4:7% 5:8% 6:3% 7:24% 8:5% 9:23%]
    Running50: [0:18% 1:2% 2:0% 3:11% 4:7% 5:8% 6:3% 7:24% 8:5% 9:23%]
    Solver: [2.422, 2.422, 2.422, 2.422, 2.422, 2.422, 2.406] ✓
  [Batch 1] Grid size: 30x30 (batch_max_size=30)

  [MEMORY] Epoch 18 Batch 1:
    Baseline: alloc=92MB, reserved=30648MB
    01_batch_on_gpu: alloc=220MB (+127MB), reserved=30672MB
    02_after_forward: alloc=20296MB (+20077MB), reserved=30672MB
    03_before_backward: alloc=20321MB (+25MB), reserved=30672MB
    04_after_backward: alloc=285MB (-20036MB), reserved=30672MB
    PEAK: alloc=20737MB, reserved=30672MB / 24576MB (124.8%)

    [BREAKDOWN] Largest increase at '02_after_forward' (+20077MB):
      Model params: 73.4MB
      Model grads:  52.0MB
      Optimizer:    146.7MB (estimate (Adam-like: AdamW))
      Batch data:   4.8MB
      Activations:  71.5MB
      Output tensors (top 5):
        features: 43.9MB
        support_features: 18.8MB
        all_logits: 6.0MB
        attention_maps: 1.2MB
        logits: 0.9MB
      Module params (top 5):
        solver: 28.4MB
        hyper_lora: 19.1MB
        context_encoder: 15.0MB
        context_injector: 3.0MB
        hpm: 2.9MB
      Active modules: solver_context, cross_attention, hpm
      Inactive modules: hyperlora, loo, equivariance
  [MEMORY] Batch 1: alloc=285MB, reserved=30672MB, headroom=-6096MB
  [WARNING] >95% GPU memory used! Training may slow due to shared memory.
  Batch 1/5: loss=0.2632, focal_weighted=0.1355, batch_acc=9.5%, exact=0/100 (0.0%), running_acc=10.6%, lr=5.00e-04, hpm=0.0050
    [TaskTrack] Global_Solved: 3/342, Epoch_Solved: 0, New_Puzzles: 0
    FG: batch=8.5% run50=8.2% | BG: batch=9.6% run50=11.4%
    Per-Color: [0:15% 1:7% 2:0% 3:9% 4:15% 5:14% 6:7% 7:21% 8:4% 9:17%]
    Running50: [0:16% 1:4% 2:0% 3:10% 4:11% 5:11% 6:5% 7:22% 8:5% 9:20%]
  [Batch 2] Grid size: 30x30 (batch_max_size=30)

  [MEMORY] Epoch 18 Batch 2:
    Baseline: alloc=92MB, reserved=30648MB
    01_batch_on_gpu: alloc=219MB (+126MB), reserved=30672MB
    02_after_forward: alloc=23433MB (+23214MB), reserved=30672MB
    03_before_backward: alloc=23480MB (+47MB), reserved=30672MB
    04_after_backward: alloc=293MB (-23186MB), reserved=30672MB
    PEAK: alloc=24075MB, reserved=30672MB / 24576MB (124.8%)

    [BREAKDOWN] Largest increase at '02_after_forward' (+23214MB):
      Model params: 73.4MB
      Model grads:  52.0MB
      Optimizer:    146.7MB (estimate (Adam-like: AdamW))
      Batch data:   6.2MB
      Activations:  77.8MB
      Output tensors (top 5):
        features: 43.9MB
        support_features: 25.0MB
        all_logits: 6.0MB
        attention_maps: 1.2MB
        logits: 0.9MB
      Module params (top 5):
        solver: 28.4MB
        hyper_lora: 19.1MB
        context_encoder: 15.0MB
        context_injector: 3.0MB
        hpm: 2.9MB
      Active modules: solver_context, cross_attention, hpm
      Inactive modules: hyperlora, loo, equivariance
  [MEMORY] Batch 2: alloc=293MB, reserved=30672MB, headroom=-6096MB
  [WARNING] >95% GPU memory used! Training may slow due to shared memory.
  Batch 2/5: loss=0.4778, focal_weighted=0.3269, batch_acc=14.1%, exact=0/150 (0.0%), running_acc=11.8%, lr=5.00e-04, hpm=0.0050
    [TaskTrack] Global_Solved: 3/342, Epoch_Solved: 0, New_Puzzles: 0
    FG: batch=6.6% run50=7.7% | BG: batch=17.3% run50=13.4%
    Per-Color: [0:21% 1:3% 2:1% 3:6% 4:5% 5:11% 6:4% 7:39% 8:6% 9:14%]
    Running50: [0:18% 1:4% 2:0% 3:9% 4:9% 5:11% 6:5% 7:28% 8:5% 9:18%]
  [Batch 3] Grid size: 30x30 (batch_max_size=30)

  [MEMORY] Epoch 18 Batch 3:
    Baseline: alloc=92MB, reserved=30648MB
    01_batch_on_gpu: alloc=218MB (+125MB), reserved=30672MB
    02_after_forward: alloc=18725MB (+18507MB), reserved=30672MB
    03_before_backward: alloc=18807MB (+82MB), reserved=30672MB
    04_after_backward: alloc=281MB (-18525MB), reserved=30672MB
    PEAK: alloc=19100MB, reserved=30672MB / 24576MB (124.8%)

    [BREAKDOWN] Largest increase at '02_after_forward' (+18507MB):
      Model params: 73.4MB
      Model grads:  52.0MB
      Optimizer:    146.7MB (estimate (Adam-like: AdamW))
      Batch data:   4.1MB
      Activations:  68.3MB
      Output tensors (top 5):
        features: 43.9MB
        support_features: 15.6MB
        all_logits: 6.0MB
        attention_maps: 1.2MB
        logits: 0.9MB
      Module params (top 5):
        solver: 28.4MB
        hyper_lora: 19.1MB
        context_encoder: 15.0MB
        context_injector: 3.0MB
        hpm: 2.9MB
      Active modules: solver_context, cross_attention, hpm
      Inactive modules: hyperlora, loo, equivariance
  Batch 3/5: loss=0.9460, focal_weighted=0.7811, batch_acc=13.8%, exact=0/200 (0.0%), running_acc=12.3%, lr=5.00e-04, hpm=0.0050
    [TaskTrack] Global_Solved: 3/342, Epoch_Solved: 0, New_Puzzles: 0
    FG: batch=7.3% run50=7.6% | BG: batch=19.1% run50=14.8%
    Per-Color: [0:22% 1:4% 2:1% 3:5% 4:8% 5:9% 6:7% 7:18% 8:9% 9:7%]
    Running50: [0:19% 1:4% 2:0% 3:8% 4:9% 5:11% 6:5% 7:25% 8:6% 9:15%]
  [Batch 4] Grid size: 30x30 (batch_max_size=30)

  [MEMORY] Epoch 18 Batch 4:
    Baseline: alloc=92MB, reserved=30648MB
    01_batch_on_gpu: alloc=212MB (+119MB), reserved=30672MB
    02_after_forward: alloc=18721MB (+18509MB), reserved=30672MB
    03_before_backward: alloc=18751MB (+30MB), reserved=30672MB
    04_after_backward: alloc=277MB (-18474MB), reserved=30672MB
    PEAK: alloc=19096MB, reserved=30672MB / 24576MB (124.8%)

    [BREAKDOWN] Largest increase at '02_after_forward' (+18509MB):
      Model params: 73.4MB
      Model grads:  52.0MB
      Optimizer:    146.7MB (estimate (Adam-like: AdamW))
      Batch data:   4.1MB
      Activations:  68.3MB
      Output tensors (top 5):
        features: 43.9MB
        support_features: 15.6MB
        all_logits: 6.0MB
        attention_maps: 1.2MB
        logits: 0.9MB
      Module params (top 5):
        solver: 28.4MB
        hyper_lora: 19.1MB
        context_encoder: 15.0MB
        context_injector: 3.0MB
        hpm: 2.9MB
      Active modules: solver_context, cross_attention, hpm
      Inactive modules: hyperlora, loo, equivariance
  Batch 4/5: loss=0.2873, focal_weighted=0.1540, batch_acc=9.8%, exact=0/250 (0.0%), running_acc=11.8%, lr=5.00e-04, hpm=0.0051
    [TaskTrack] Global_Solved: 3/342, Epoch_Solved: 0, New_Puzzles: 0
    FG: batch=5.7% run50=7.2% | BG: batch=12.6% run50=14.4%
    Per-Color: [0:16% 1:4% 2:0% 3:10% 4:6% 5:12% 6:0% 7:14% 8:1% 9:24%]
    Running50: [0:18% 1:4% 2:0% 3:8% 4:8% 5:11% 6:4% 7:23% 8:5% 9:17%]

Epoch 19 Summary:
  Total Loss: 0.4530
  Task Loss (focal): 0.3106
  [Global Task Progress] Unique Solved: 3/342 (0.9%)
    NEW puzzles solved this epoch: 0
    ⚠️ No new puzzles solved in last 5 epochs - model may be plateauing
  Entropy Loss: 4.2637 (weight=0.01)
  Sparsity Loss: 0.1617 (weight=0.5)
  Predicate Loss: 0.0000 (weight=0.01)
  HPM Balance Loss: 0.0051 (weight=0.01)
  HPM Gate Value: 0.0000 (0=no contribution, 1=full)
  HPM Instance Buffer: 2 entries
  HPM Procedural Buffer: 0 entries
  HPM Tasks Added (exact matches): 0
    (No exact matches this epoch: 0)
  Time: 39.7s, LR: 5.00e-04
    Per-module LRs: DSC:5.00e-04, MSRE:5.00e-04, Other:5.00e-04
  Temperature: 0.9395 (lower=sharper attention)
  Samples Processed: 250 (5 batches)
  Dihedral Distribution: [15.6%, 15.6%, 10.8%, 12.0%, 11.2%, 12.4%, 9.6%, 12.8%]
  Color Permutation: 0.0% (0/250)
  Translational Aug: 100.0% (250/250), unique offsets: 238
  Aug Quality: OK
  --- Training Diagnostics ---
  Solver Steps: 7 (deep supervision active)
  Per-Step Loss (epoch avg, 1 batches): [2.4219, 2.4219, 2.4219, 2.4219, 2.4219, 2.4219, 2.4062]
    ✓ Step improvement: 0.6% (later steps better - GOOD!)
  Best-Step Histogram: []
  Solver Health: Last step best: 100.0%, Earlier step best: 0.0%
  Avg Step Improvement: 0.6% (step0→stepN)
  Grad Norms: DSC=0.0000, StopPred=0.0000, Encoder=0.0000, Solver=0.0000
  StopPred Weight Var: 2.80e-02
  Attention: max=0.0498, min=0.000000
  Stop Prob: 0.277 (approx 5.1 clues active)
  Stop Probs Std: 0.107 (global std across batch×clues)
  Clues Used: mean=5.06, std=0.49, range=[3.1, 6.0]
  Clue-Loss Correlation: -0.118 (unexpected negative - check gradient flow)
  Stop Logits: mean=-1.02, std=0.54, range=[-2.6, 1.1]
    Clue count varies by task (per-sample coupling active)
  Per-Clue Entropy: [3.74, 3.74, 3.74, 3.74, 3.74, 3.74, 3.74] (mean=3.74, max=6.80)
    [!] Clues have uniform entropy (std=0.002) - not differentiating!
  Centroid Spread: 0.06 (higher=more diverse)
    🚨 CRITICAL COLLAPSE: Spread=0.06 < 0.5 - all clues at same location!
    [!] Stop predictor cannot differentiate - needs diversity regularizer
  --- Stop Predictor Coupling ---
  Entropy Input to Stop: 0.5498 (normalized, lower=sharper)
  Per-Clue Entropy Input: [0.550, 0.550, 0.550, 0.550, 0.550, 0.550, 0.550]
  Per-Clue Stop Prob: [0.260, 0.305, 0.285, 0.262, 0.277, 0.283, 0.273]
  --- Sparsity Loss Breakdown (Per-Sample Coupled) ---
  Min Clue Penalty: 0.0000 (per-sample avg)
  Per-Sample Clue Penalty (scaled): 0.0000
  Base Pondering: 0.1011 (clues=5.05)
  Entropy Pondering: 0.0560
  --- Per-Class Distribution (Valid Pixels Only) ---
  Padding: 93.6% of grid (ignored in loss)
  Pred %: [17.2, 2.9, 0.7, 10.6, 7.9, 16.9, 0.9, 26.7, 2.8, 13.4]
  Target %: [58.7, 6.8, 8.5, 7.3, 3.1, 5.1, 2.4, 4.4, 2.9, 0.8]
  Per-Class Acc %: [20, 4, 0, 7, 7, 10, 5, 22, 7, 11]
  ==================================================
  PER-SAMPLE TRAINING ACCURACY (Epoch 19)
  ==================================================
  ★ Mean Accuracy: 11.8%
  ★ Exact Match: 0/250 (0.0%)
  ★ High Acc (≥90%): 0/250 (0.0%)
  FG Accuracy: 7.2%
  BG Accuracy: 14.4%
  Batch Trend: 11.7% → 11.8% (→ 0.1pp)
  Accuracy Distribution: 0-25%:100%, 25-50%:0%, 50-75%:0%, 75-90%:0%, 90-100%:0%
    [!] Many samples stuck at low accuracy - check data/model!
  Running Window (last 5 batches): 11.8% ± 1.9%

  --- PER-COLOR ACCURACY (10 classes) ---
  Color:       0     1     2     3     4     5     6     7     8     9
  Acc%:      20%    4%    0%    7%    7%   10%    5%   22%    7%   11%
  Target:  56.4%  7.2%  5.9%  6.9%  4.6%  5.9%  2.6%  2.4%  6.1%  1.9%
  Pred:    17.5%  2.6%  0.6%  8.1%  7.4% 20.9%  3.0% 26.5%  3.5%  9.8%
  [!] Weak colors (<50% acc): 0(Black), 1(Blue), 2(Red), 3(Green), 4(Yellow), 5(Gray), 6(Pink), 7(Orange), 8(Cyan), 9(Brown)
  [!] Under-predicting color 1 (Blue): 2.6% vs target 7.2%
  [!] Under-predicting color 2 (Red): 0.6% vs target 5.9%
  ==================================================

  ############################################################
  🚦 TRAINING HEALTH CHECK - Epoch 19
  ############################################################
  ✓ Attention sharpening (0.55 < 0.7)
  ✓ Stop probs adapting (global=0.107, per_clue=0.014)
  ✓ No NaN/Inf issues
  ✓ No color mode collapse
  ⚠ Weak confidence-stop coupling (r=0.22)
  ⚠ Loss slowly decreasing (0.311 → 0.311)
  ✗ Centroid COLLAPSE (0.1 < 0.5)
  ✗ Accuracy not improving (19.9% → 11.8%)
  --------------------------------------------------------
  RESULT: 4/8 checks passed
  STATUS: 🟠 WARNING
  → Multiple issues detected. Consider intervention soon.

  RECOMMENDED ACTIONS:
  ############################################################
  HPM Instance Buffer: 2 entries SAVED
    → Saved to: checkpoints\rlan_stable_merged\hpm\instance_buffer.pt
  HPM Procedural Buffer: 0 entries (empty, not saved)
  Saved checkpoint to checkpoints\rlan_stable_merged\latest.pt
  [ARCDataset] Augmentation config updated:
    dihedral=True, color_perm=False (prob=0.50), translational=True

Epoch 20/200
----------------------------------------
  [Batch 0] Grid size: 30x30 (batch_max_size=30)

  [MEMORY] Epoch 19 Batch 0:
    Baseline: alloc=92MB, reserved=30672MB
    01_batch_on_gpu: alloc=102MB (+10MB), reserved=30694MB
    02_after_forward: alloc=18614MB (+18512MB), reserved=30694MB
    03_before_backward: alloc=18698MB (+84MB), reserved=30694MB
    04_after_backward: alloc=226MB (-18472MB), reserved=30694MB
    PEAK: alloc=18989MB, reserved=30694MB / 24576MB (124.9%)

    [BREAKDOWN] Largest increase at '02_after_forward' (+18512MB):
      Model params: 73.4MB
      Model grads:  0.0MB
      Optimizer:    146.7MB (estimate (Adam-like: AdamW))
      Batch data:   4.1MB
      Activations:  68.3MB
      Output tensors (top 5):
        features: 43.9MB
        support_features: 15.6MB
        all_logits: 6.0MB
        attention_maps: 1.2MB
        logits: 0.9MB
      Module params (top 5):
        solver: 28.4MB
        hyper_lora: 19.1MB
        context_encoder: 15.0MB
        context_injector: 3.0MB
        hpm: 2.9MB
      Active modules: solver_context, cross_attention, hpm
      Inactive modules: hyperlora, loo, equivariance

  [MEMORY] First forward pass with staged modules active:
  [MEMORY] Batch 0: alloc=226MB, reserved=30694MB, headroom=-6118MB
  [WARNING] >95% GPU memory used! Training may slow due to shared memory.
  Batch 0/5: loss=0.9690, focal_weighted=0.8057, batch_acc=15.6%, exact=0/50 (0.0%), running_acc=15.6%, lr=5.00e-04, hpm=0.0051
    [TaskTrack] Global_Solved: 3/342, Epoch_Solved: 0, New_Puzzles: 0
    FG: batch=7.5% run50=7.5% | BG: batch=21.1% run50=21.1%
    Per-Color: [0:25% 1:2% 2:1% 3:6% 4:14% 5:12% 6:11% 7:16% 8:7% 9:5%]
    Running50: [0:25% 1:2% 2:1% 3:6% 4:14% 5:12% 6:11% 7:16% 8:7% 9:5%]
    Solver: [3.422, 3.422, 3.406, 3.406, 3.391, 3.391, 3.391] ⚠ best=4
  [Batch 1] Grid size: 30x30 (batch_max_size=30)

  [MEMORY] Epoch 19 Batch 1:
    Baseline: alloc=92MB, reserved=30672MB
    01_batch_on_gpu: alloc=219MB (+126MB), reserved=30696MB
    02_after_forward: alloc=21862MB (+21643MB), reserved=30696MB
    03_before_backward: alloc=21888MB (+26MB), reserved=30696MB
    04_after_backward: alloc=291MB (-21597MB), reserved=30696MB
    PEAK: alloc=22404MB, reserved=30696MB / 24576MB (124.9%)

    [BREAKDOWN] Largest increase at '02_after_forward' (+21643MB):
      Model params: 73.4MB
      Model grads:  52.0MB
      Optimizer:    146.7MB (estimate (Adam-like: AdamW))
      Batch data:   5.5MB
      Activations:  74.6MB
      Output tensors (top 5):
        features: 43.9MB
        support_features: 21.9MB
        all_logits: 6.0MB
        attention_maps: 1.2MB
        logits: 0.9MB
      Module params (top 5):
        solver: 28.4MB
        hyper_lora: 19.1MB
        context_encoder: 15.0MB
        context_injector: 3.0MB
        hpm: 2.9MB
      Active modules: solver_context, cross_attention, hpm
      Inactive modules: hyperlora, loo, equivariance
  [MEMORY] Batch 1: alloc=291MB, reserved=30696MB, headroom=-6120MB
  [WARNING] >95% GPU memory used! Training may slow due to shared memory.
  Batch 1/5: loss=0.2995, focal_weighted=0.1678, batch_acc=8.9%, exact=0/100 (0.0%), running_acc=12.2%, lr=5.00e-04, hpm=0.0051
    [TaskTrack] Global_Solved: 3/342, Epoch_Solved: 0, New_Puzzles: 0
    FG: batch=5.8% run50=6.6% | BG: batch=11.4% run50=16.2%
    Per-Color: [0:16% 1:4% 2:0% 3:10% 4:10% 5:6% 6:3% 7:16% 8:3% 9:5%]
    Running50: [0:21% 1:3% 2:0% 3:8% 4:12% 5:9% 6:7% 7:16% 8:5% 9:5%]
  [Batch 2] Grid size: 30x30 (batch_max_size=30)


