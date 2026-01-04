  Batch 30/32: loss=0.3500, focal_weighted=0.3182, batch_acc=72.2%, exact=8/582 (1.4%), running_acc=58.8%, lr=3.92e-04, loo=0.7102, hpm=0.0056
    [TaskTrack] Global_Solved: 12/411, Epoch_Solved: 6, New_Puzzles: 0
    FG: batch=69.6% run50=45.1% | BG: batch=69.5% run50=73.5%
    Per-Color: [0:94% 1:66% 2:90% 3:65% 4:64% 5:85% 6:36% 7:95% 8:88% 9:52%]
    Running50: [0:87% 1:51% 2:52% 3:53% 4:48% 5:63% 6:43% 7:67% 8:60% 9:49%]
  Batch 31/32: loss=0.1211, focal_weighted=0.0891, batch_acc=47.9%, exact=8/602 (1.3%), running_acc=58.4%, lr=3.92e-04, loo=1.4844, hpm=0.0056
    [TaskTrack] Global_Solved: 12/411, Epoch_Solved: 6, New_Puzzles: 0
    FG: batch=30.2% run50=44.6% | BG: batch=71.2% run50=73.4%
    Per-Color: [0:84% 1:20% 2:23% 3:27% 4:6% 5:43% 6:51% 7:63% 8:92% 9:13%]
    Running50: [0:87% 1:50% 2:51% 3:52% 4:47% 5:62% 6:43% 7:67% 8:61% 9:47%]

Epoch 100 Summary:
  Total Loss: 0.2896
  Task Loss (focal): 0.2572
  [Global Task Progress] Unique Solved: 12/411 (2.9%)
    NEW puzzles solved this epoch: 0
    ⚠️ No new puzzles solved in last 5 epochs - model may be plateauing
  Entropy Loss: 0.1632 (weight=0.01)
  Sparsity Loss: 0.0221 (weight=0.5)
  Predicate Loss: 0.0000 (weight=0.01)
  HPM Balance Loss: 0.0056 (weight=0.01)
  HPM Gate Value: 0.0011 (0=no contribution, 1=full)
  HPM Instance Buffer: 11 entries
  HPM Procedural Buffer: 6 entries
  HPM Tasks Added (exact matches): 0
    (No exact matches this epoch: 8)
  HPM Retrieval Stats (32 batches):
    Instance: avg_sim=0.915, total_retrieved=3010
    Procedural: avg_sim=0.915, total_retrieved=3010
  Time: 53.6s, LR: 3.92e-04
  HyperLoRA Clamp: hit_rate=0.0% (0/14448), max_norm=0.47 (threshold=1.0)
  HPM Solver Coupling: gate=0.285, batches=0, tokens=0
  Meta Escalation: progress=100.0%, stable=True
    HyperLoRA: 0.1955/0.2000
    Equiv: 0.0291/0.0300
    LOO: 0.0791/0.0800
    HPM Balance: 0.0100/0.0100
    ⚠️ PAUSED (nan=0, grad=0)
    Meta contribution ratio: 26.2% of total loss
    Per-module LRs: DSC:3.92e-04, MSRE:3.92e-04, Other:3.92e-04
  Temperature: 0.7096 (lower=sharper attention)
  Samples Processed: 602 (32 batches)
  Dihedral Distribution: [11.1%, 13.0%, 14.0%, 13.1%, 14.5%, 13.8%, 9.8%, 10.8%]
  Color Permutation: 49.7% (299/602)
  Translational Aug: 94.0% (566/602), unique offsets: 550
  Aug Quality: OK
  --- Training Diagnostics ---
  Solver Steps: 7 (deep supervision active)
  Per-Step Loss (epoch avg, 1 batches): [0.9766, 0.9141, 0.9062, 0.9023, 0.8984, 0.8984, 0.8984]
    [!] Best step is 4 (middle), not last - solver may be over-iterating!
    [!] Best: step 4 (0.8984), Final: step 6 (0.8984)
    [!] Potential gain from best-step selection: 0.0% lower loss
  Best-Step Histogram: []
  Solver Health: Last step best: 0.0%, Earlier step best: 100.0%
  Avg Step Improvement: 8.0% (step0→stepN)
    ⚠️ SOLVER OVER-ITERATION WARNING: 100% of batches had earlier step as best!
    ⚠️ Consider: (1) enabling best-step selection, (2) reducing num_solver_steps, (3) enabling ACT
  Grad Norms: DSC=0.0047, StopPred=0.0029, Encoder=0.0312, Solver=0.1819
              ContextEnc=0.0196, MSRE=0.0025
  StopPred Weight Var: 2.80e-02
  Attention: max=0.9224, min=0.000000
    Attention is sharp (good!)
  Stop Prob: 0.852 (approx 1.0 clues active)
  Stop Probs Std: 0.342 (global std across batch×clues)
  Clues Used: mean=1.04, std=0.03, range=[1.0, 1.2]
  Clue-Loss Correlation: +0.263 (learning - per-sample coupling active)
  Stop Logits: mean=3.36, std=3.16, range=[-4.7, 5.4]
    [!] Stop logits approaching saturation |mean|=3.4
    [!] Low variance - clue count not adapting per-task!
  Per-Clue Entropy: [0.25, 0.20, 0.19, 0.23, 0.21, 0.20, 0.21] (mean=0.21, max=6.80)
    [!] Clues have uniform entropy (std=0.020) - not differentiating!
    Good entropy (0.21) - attention is focused!
  Centroid Spread: 4.88 (higher=more diverse)
  --- Stop Predictor Coupling ---
  Entropy Input to Stop: 0.0313 (normalized, lower=sharper)
  Per-Clue Entropy Input: [0.037, 0.029, 0.028, 0.033, 0.031, 0.030, 0.031]
  Per-Clue Stop Prob: [0.015, 0.988, 0.988, 0.988, 0.984, 0.988, 0.992]
  --- Sparsity Loss Breakdown (Per-Sample Coupled) ---
  Min Clue Penalty: 0.0002 (per-sample avg)
  Per-Sample Clue Penalty (scaled): 0.0005
  Base Pondering: 0.0210 (clues=1.05)
  Entropy Pondering: 0.0008
    [+] Per-sample penalty correctly scaled by λ_sparsity
  --- META-LEARNING (HyperLoRA + LOO) ---
  LOO Loss (avg): 1.1517
  LOO Accuracy (N-1→Nth): 69.7%
  LOO Holdouts/batch: 4.0
  LOO Batches: 32 computed, 0 skipped
    Learning: HyperLoRA starting to generalize
  Equivariance: DISABLED in config
  --- Meta-Learning Health Summary ---
  Overall: 🔄 [LOO ✓ | Equiv skipped | No HyperLoRA grads] (score=0.5/1.0)
  --- Detailed Attribution ---
  LoRA Delta Norm (avg): 0.0784 ✓ (healthy range)
  Context Magnitude (avg): 0.2256 ✓
  HPM Routing Entropy (avg): 0.686 (moderate specialization)
  --- Gradient Clipping ---
  Grad Norm (before clip): 0.1907
    Gradients within bounds
  --- Per-Class Distribution (Valid Pixels Only) ---
  Padding: 75.5% of grid (ignored in loss)
  Pred %: [32.5, 3.2, 2.6, 18.4, 7.3, 4.3, 10.8, 5.4, 15.0, 0.6]
  Target %: [27.0, 3.5, 5.2, 17.0, 8.0, 4.4, 10.9, 8.1, 15.0, 1.0]
  Per-Class Acc %: [90, 64, 65, 75, 70, 65, 59, 80, 76, 55]
  ==================================================
  PER-SAMPLE TRAINING ACCURACY (Epoch 100)
  ==================================================
  ★ Mean Accuracy: 58.4%
  ★ Exact Match: 8/602 (1.3%)
  ★ High Acc (≥90%): 117/602 (19.4%)
  FG Accuracy: 44.6%
  BG Accuracy: 73.4%
  Batch Trend: 57.1% → 61.0% (↑ 3.8pp)
    ✓ Accuracy improving within epoch - learning is active!
  Accuracy Distribution: 0-25%:6%, 25-50%:12%, 50-75%:31%, 75-90%:25%, 90-100%:25%
  Running Window (last 32 batches): 58.4% ± 8.5%

  --- PER-COLOR ACCURACY (10 classes) ---
  Color:       0     1     2     3     4     5     6     7     8     9
  Acc%:      90%   64%   65%   75%   70%   65%   59%   80%   76%   55%
  Target:  28.6%  9.5%  7.5%  9.0% 11.3%  5.8%  6.3%  9.0%  9.3%  3.8%
  Pred:    32.6%  8.0%  6.5% 10.4%  9.5%  5.1%  5.0%  9.5%  9.1%  4.4%
  ==================================================

  ==================================================
  LEARNING TRAJECTORY (Epoch 100)
  ==================================================
  Stop Prob:   0.852 → (init=0.27, task-dependent)
  Exp. Clues:  1.05 (latent variable, task-dependent)
  Attn Entropy: 0.21 ↑ (max=6.8, sharper=better)
  Task Loss:   0.2572 ↑
  Train Acc:   58.4% ↓
  Exact Match: 1.3% ↓
  Best Step:   4 (later=better refinement)
  FG Coverage: 92.5% of target ↓
  ✓ Learning trajectory looks healthy!
  ==================================================

  ############################################################
  🚦 TRAINING HEALTH CHECK - Epoch 100
  ############################################################
  ✓ Attention sharpening (0.03 < 0.7)
  ✓ Stop probs adapting (global=0.342, per_clue=0.340)
  ✓ Good confidence-stop coupling (r=0.84)
  ✓ Loss decreasing (1.597 → 0.257)
  ✓ Accuracy improving (25.9% → 58.4%)
  ✓ No NaN/Inf issues
  ✓ No color mode collapse
  ⚠ Centroids moderately spread (4.9)
  --------------------------------------------------------
  RESULT: 7/8 checks passed
  STATUS: 🟢 HEALTHY
  → Training is progressing well. Continue!
  ############################################################

  [Eval] Running evaluation on 10 batches... 10/10
  [TRM-Eval] Running TTA on 100 tasks × 32 views
  [TRM-Eval] MULTI-TASK BATCHED: ~13 forward passes (B=256 max)
  [TRM-Eval] Tasks 1-8/100 (8%)
  [TRM-Eval] Tasks 9-16/100 (16%)
  [TRM-Eval] Tasks 17-24/100 (24%)
  [TRM-Eval] Tasks 25-32/100 (32%)
  [TRM-Eval] Tasks 33-40/100 (40%)
  [TRM-Eval] Tasks 41-48/100 (48%)
  [TRM-Eval] Tasks 49-56/100 (56%)
  [TRM-Eval] Tasks 57-64/100 (64%)
  [TRM-Eval] Tasks 65-72/100 (72%)
  [TRM-Eval] Tasks 73-80/100 (80%)
  [TRM-Eval] Tasks 81-88/100 (88%)
  [TRM-Eval] Tasks 89-96/100 (96%)
  [TRM-Eval] Tasks 97-100/100 (100%)
  [TRM-Eval] Complete. Exact match: 0/100 (0.0%)
  [TRM-Eval] ⚠️ Shape mismatch info: 54/100 tasks had train output shape != test output shape
  [TRM-Eval] ℹ️ Vote ties: 44/100 tasks had multiple predictions with same vote count

  --- TRM-Style TTA Evaluation (8 dihedral x 4 color = 32 views) ---
  ★ TTA Exact Match (Pass@1): 0/100 (0.0%)
  ⏱️ TTA eval time: 222.3s (2.22s/task)
  Pass@K: Pass@1: 0.0% | Pass@2: 0.0% | Pass@3: 0.0%
  Avg Unique Predictions: 10.0 / 32
  Avg Winner Votes: 13.2 / 32

  --- Generalization Health (SINGLE-SHOT) ---
  Train Tasks (first-sample): 3/411 (0.7%)
  Eval Tasks (TTA): 0/100 (0.0%)
  Train→Eval Gap: 0.7% [true single-shot comparison]
  (Any-sample train: 1.5% | Sample-level: 1.3%)
  ✅ Healthy gap: 0.7% - Good generalization!
  ⚠️ Moderate consensus: 41%
  --- Evaluation Metrics (Valid Pixels Only) ---
  ★ EXACT MATCH: 1/400 tasks (0.2%)
  Pixel Accuracy: 0.7717
  FG Accuracy (colors 1-9): 0.5971
  BG Accuracy (black): 0.9273
  Class Ratios (pred/target):
    BG (black): 59.3% / 52.9%
    FG (colors): 40.7% / 47.1%
  Colors Used (pred/target): 10 / 10
  DSC Entropy: 4.0940 (lower=sharper)
  DSC Clues Used: 2.34
  Eval Stop Prob: 0.666
  Predicate Activation: 0.0000
  Eval Temperature: 0.710 (matched to training)

  --- Train vs Eval Entropy Delta ---
  Train DSC Entropy: 0.2128
  Eval DSC Entropy:  4.0940
  Delta (Eval - Train): +3.8812
  Ratio (Eval / Train): 19.24x
  🚨 CRITICAL: Eval entropy 19.2x higher than train!
      Model not generalizing - attention collapses on unseen data
  HPM Instance Buffer: 11 entries SAVED
    → Saved to: checkpoints\rlan_stable_merged\hpm\instance_buffer.pt
  HPM Procedural Buffer: 6 entries SAVED
    → Saved to: checkpoints\rlan_stable_merged\hpm\procedural_buffer.pt
  Saved checkpoint to checkpoints\rlan_stable_merged\epoch_100.pt
  HPM Instance Buffer: 11 entries SAVED
    → Saved to: checkpoints\rlan_stable_merged\hpm\instance_buffer.pt
  HPM Procedural Buffer: 6 entries SAVED
    → Saved to: checkpoints\rlan_stable_merged\hpm\procedural_buffer.pt
  Saved checkpoint to checkpoints\rlan_stable_merged\latest.pt
  [ARCDataset] Augmentation config updated:
    dihedral=True, color_perm=True (prob=0.30), translational=True

Epoch 101/200

