# ARC Diagnostic Test Log

Generated: 2025-12-17T22:44:27.038979

## Training Log

```
[22:43:09] [INFO] ======================================================================
[22:43:09] [INFO] ARC DIAGNOSTIC TEST
[22:43:09] [INFO] ======================================================================
[22:43:09] [INFO] Device: cpu
[22:43:09] [INFO] Num Tasks: 1
[22:43:09] [INFO] Max Epochs: 100
[22:43:09] [INFO] Use Train Context: True
[22:43:09] [INFO] 
[22:43:09] [INFO] Task 007bbfb7: train=5, max_in=3x3, max_out=9x9
[22:43:09] [INFO] Loaded 1 samples from 1 tasks
[22:43:09] [INFO] Total with augmentation: 8
[22:43:09] [INFO] Dynamic grid size from batch: 9x9
[22:43:09] [INFO] Model Parameters: 12,118,636
[22:43:11] [INFO] Optimizer: AdamW, LR=0.0005
[22:43:11] [INFO] 
[22:43:11] [INFO] ======================================================================
[22:43:11] [INFO] STARTING TRAINING
[22:43:11] [INFO] ======================================================================
[22:43:12] [INFO] Epoch   1/100 | Loss: 1.9298 | Acc: 3.4% | Exact: 0/8 (0.0%) | Temp: 1.000
[22:43:13] [INFO] Epoch   2/100 | Loss: 1.2746 | Acc: 46.1% | Exact: 0/8 (0.0%) | Temp: 0.993
[22:43:13] [INFO] Epoch   3/100 | Loss: 1.0590 | Acc: 50.3% | Exact: 0/8 (0.0%) | Temp: 0.986
[22:43:14] [INFO] Epoch   4/100 | Loss: 1.3419 | Acc: 55.6% | Exact: 0/8 (0.0%) | Temp: 0.979
[22:43:15] [INFO] Epoch   5/100 | Loss: 0.7298 | Acc: 55.6% | Exact: 0/8 (0.0%) | Temp: 0.973
[22:43:18] [INFO] Epoch  10/100 | Loss: 0.7097 | Acc: 60.3% | Exact: 0/8 (0.0%) | Temp: 0.940
[22:43:22] [INFO] Epoch  15/100 | Loss: 1.0275 | Acc: 60.5% | Exact: 0/8 (0.0%) | Temp: 0.908
[22:43:26] [INFO] 
--- Epoch 20 Detailed Diagnostics ---
[22:43:26] [INFO]   logits: shape=[8, 10, 9, 9], range=[-4.3064, 8.1016], mean=-1.2268
[22:43:26] [INFO]   attention_maps: shape=[8, 6, 9, 9], range=[0.0000, 0.9299], mean=0.0123
[22:43:26] [INFO]   stop_logits: shape=[8, 6], range=[-0.9372, 1.0106], mean=0.1043
[22:43:26] [INFO] Per-sample accuracy:
[22:43:26] [INFO]   [2] 007bbfb7: 60.5% (✗)
[22:43:26] [INFO]   [1] 007bbfb7: 60.5% (✗)
[22:43:26] [INFO]   [4] 007bbfb7: 60.5% (✗)
[22:43:26] [INFO]   [0] 007bbfb7: 60.5% (✗)
[22:43:26] [INFO] Sample 0 prediction:
[22:43:26] [INFO]   Sample 0 (9x9): 49/81 pixels correct (60.5%)
[22:43:26] [INFO]     Target: [[0, 0, 0, 0, 7, 7, 0, 7, 7], [0, 0, 0, 7, 0, 7, 7, 0, 7], [0, 0, 0, 7, 0, 7, 7, 0, 7], [0, 7, 7, 0, 0, 0, 0, 7, 7], [7, 0, 7, 0, 0, 0, 7, 0, 7], [7, 0, 7, 0, 0, 0, 7, 0, 7], [0, 7, 7, 0, 0, 0, 0, 7, 7], [7, 0, 7, 0, 0, 0, 7, 0, 7], [7, 0, 7, 0, 0, 0, 7, 0, 7]]
[22:43:26] [INFO]     Pred:   [[7, 7, 7, 7, 7, 7, 7, 7, 7], [7, 0, 0, 0, 0, 0, 0, 0, 7], [7, 0, 0, 0, 0, 0, 0, 0, 7], [7, 0, 0, 0, 0, 0, 0, 0, 7], [7, 0, 0, 0, 0, 0, 0, 0, 7], [7, 0, 0, 0, 0, 0, 0, 0, 7], [7, 0, 0, 0, 0, 0, 0, 0, 7], [7, 0, 0, 0, 0, 0, 0, 0, 7], [7, 7, 7, 7, 7, 7, 7, 7, 7]]
[22:43:26] [INFO]   Gradient stats:
[22:43:26] [INFO]     encoder: avg_norm=3.1734e-02, max_norm=1.1978e-01
[22:43:26] [INFO]     feature_proj: avg_norm=6.0161e-02, max_norm=2.1684e-01
[22:43:26] [INFO]     context_encoder: avg_norm=9.4873e-03, max_norm=9.8466e-02
[22:43:26] [INFO]     context_injector: avg_norm=7.5076e-02, max_norm=2.1109e-01
[22:43:26] [INFO]     dsc: avg_norm=9.4273e-02, max_norm=6.3530e-01
[22:43:26] [INFO]     msre: avg_norm=1.4110e-04, max_norm=1.0732e-03
[22:43:26] [INFO]     solver: avg_norm=1.2776e-02, max_norm=9.1908e-02
[22:43:26] [INFO] Epoch  20/100 | Loss: 0.5553 | Acc: 60.5% | Exact: 0/8 (0.0%) | Temp: 0.877
[22:43:30] [INFO] Epoch  25/100 | Loss: 0.5382 | Acc: 60.6% | Exact: 0/8 (0.0%) | Temp: 0.847
[22:43:33] [INFO] Epoch  30/100 | Loss: 0.5174 | Acc: 61.7% | Exact: 0/8 (0.0%) | Temp: 0.818
[22:43:37] [INFO] Epoch  35/100 | Loss: 0.5001 | Acc: 63.4% | Exact: 0/8 (0.0%) | Temp: 0.790
[22:43:41] [INFO] 
--- Epoch 40 Detailed Diagnostics ---
[22:43:41] [INFO]   logits: shape=[8, 10, 9, 9], range=[-6.1551, 11.7881], mean=-2.3479
[22:43:41] [INFO]   attention_maps: shape=[8, 6, 9, 9], range=[0.0000, 0.9903], mean=0.0123
[22:43:41] [INFO]   stop_logits: shape=[8, 6], range=[-1.7448, 0.2620], mean=-0.9393
[22:43:41] [INFO] Per-sample accuracy:
[22:43:41] [INFO]   [5] 007bbfb7: 61.7% (✗)
[22:43:41] [INFO]   [2] 007bbfb7: 61.7% (✗)
[22:43:41] [INFO]   [4] 007bbfb7: 63.0% (✗)
[22:43:41] [INFO]   [6] 007bbfb7: 60.5% (✗)
[22:43:41] [INFO] Sample 0 prediction:
[22:43:41] [INFO]   Sample 0 (9x9): 50/81 pixels correct (61.7%)
[22:43:41] [INFO]     Target: [[7, 7, 0, 7, 7, 0, 0, 0, 0], [7, 0, 7, 7, 0, 7, 0, 0, 0], [7, 0, 7, 7, 0, 7, 0, 0, 0], [7, 7, 0, 0, 0, 0, 7, 7, 0], [7, 0, 7, 0, 0, 0, 7, 0, 7], [7, 0, 7, 0, 0, 0, 7, 0, 7], [7, 7, 0, 0, 0, 0, 7, 7, 0], [7, 0, 7, 0, 0, 0, 7, 0, 7], [7, 0, 7, 0, 0, 0, 7, 0, 7]]
[22:43:41] [INFO]     Pred:   [[7, 7, 7, 7, 7, 7, 7, 7, 7], [7, 0, 7, 0, 0, 0, 7, 0, 7], [7, 0, 7, 0, 0, 0, 7, 0, 7], [7, 0, 0, 0, 0, 0, 0, 0, 7], [7, 0, 0, 0, 0, 0, 0, 0, 7], [7, 0, 0, 0, 0, 0, 0, 0, 7], [7, 7, 7, 0, 0, 0, 7, 0, 7], [7, 0, 0, 0, 0, 0, 0, 0, 7], [7, 7, 7, 7, 7, 7, 7, 7, 7]]
[22:43:41] [INFO]   Gradient stats:
[22:43:41] [INFO]     encoder: avg_norm=2.1589e-03, max_norm=7.8990e-03
[22:43:41] [INFO]     feature_proj: avg_norm=3.2994e-03, max_norm=1.2538e-02
[22:43:41] [INFO]     context_encoder: avg_norm=1.5175e-04, max_norm=1.5419e-03
[22:43:41] [INFO]     context_injector: avg_norm=1.5360e-03, max_norm=4.2239e-03
[22:43:41] [INFO]     dsc: avg_norm=2.1612e-03, max_norm=1.2520e-02
[22:43:41] [INFO]     msre: avg_norm=1.0096e-03, max_norm=7.3106e-03
[22:43:41] [INFO]     solver: avg_norm=9.7411e-02, max_norm=8.4282e-01
[22:43:41] [INFO] Epoch  40/100 | Loss: 0.4913 | Acc: 63.0% | Exact: 0/8 (0.0%) | Temp: 0.763
[22:43:45] [INFO] Epoch  45/100 | Loss: 0.4671 | Acc: 65.7% | Exact: 0/8 (0.0%) | Temp: 0.737
[22:43:48] [INFO] Epoch  50/100 | Loss: 0.4548 | Acc: 67.7% | Exact: 0/8 (0.0%) | Temp: 0.712
[22:43:52] [INFO] Epoch  55/100 | Loss: 0.4358 | Acc: 71.0% | Exact: 0/8 (0.0%) | Temp: 0.688
[22:43:57] [INFO] 
--- Epoch 60 Detailed Diagnostics ---
[22:43:57] [INFO]   logits: shape=[8, 10, 9, 9], range=[-7.5288, 14.2148], mean=-3.1484
[22:43:57] [INFO]   attention_maps: shape=[8, 6, 9, 9], range=[0.0000, 0.9988], mean=0.0123
[22:43:57] [INFO]   stop_logits: shape=[8, 6], range=[-1.6713, -0.0681], mean=-0.9689
[22:43:57] [INFO] Per-sample accuracy:
[22:43:57] [INFO]   [6] 007bbfb7: 71.6% (✗)
[22:43:57] [INFO]   [5] 007bbfb7: 74.1% (✗)
[22:43:57] [INFO]   [0] 007bbfb7: 67.9% (✗)
[22:43:57] [INFO]   [3] 007bbfb7: 74.1% (✗)
[22:43:57] [INFO] Sample 0 prediction:
[22:43:57] [INFO]   Sample 0 (9x9): 58/81 pixels correct (71.6%)
[22:43:57] [INFO]     Target: [[0, 0, 0, 0, 7, 7, 0, 7, 7], [0, 0, 0, 7, 0, 0, 7, 0, 0], [0, 0, 0, 7, 7, 7, 7, 7, 7], [0, 7, 7, 0, 0, 0, 0, 0, 0], [7, 0, 0, 0, 0, 0, 0, 0, 0], [7, 7, 7, 0, 0, 0, 0, 0, 0], [0, 7, 7, 0, 7, 7, 0, 7, 7], [7, 0, 0, 7, 0, 0, 7, 0, 0], [7, 7, 7, 7, 7, 7, 7, 7, 7]]
[22:43:57] [INFO]     Pred:   [[0, 7, 7, 0, 7, 7, 7, 7, 7], [7, 0, 0, 7, 0, 0, 7, 0, 7], [7, 7, 7, 0, 7, 0, 7, 7, 7], [7, 7, 7, 0, 0, 0, 7, 0, 7], [7, 0, 0, 0, 0, 0, 7, 0, 7], [7, 7, 7, 0, 0, 0, 7, 0, 7], [7, 7, 7, 7, 7, 7, 7, 7, 7], [7, 0, 7, 0, 0, 0, 7, 0, 7], [7, 7, 7, 7, 7, 7, 7, 7, 7]]
[22:43:57] [INFO]   Gradient stats:
[22:43:57] [INFO]     encoder: avg_norm=6.3045e-03, max_norm=2.0336e-02
[22:43:57] [INFO]     feature_proj: avg_norm=8.5018e-03, max_norm=3.2078e-02
[22:43:57] [INFO]     context_encoder: avg_norm=4.2834e-04, max_norm=4.3145e-03
[22:43:57] [INFO]     context_injector: avg_norm=4.2956e-03, max_norm=1.2192e-02
[22:43:57] [INFO]     dsc: avg_norm=6.3347e-03, max_norm=3.9313e-02
[22:43:57] [INFO]     msre: avg_norm=2.0771e-03, max_norm=1.5888e-02
[22:43:57] [INFO]     solver: avg_norm=5.7858e-02, max_norm=4.5447e-01
[22:43:57] [INFO] Epoch  60/100 | Loss: 0.4103 | Acc: 70.7% | Exact: 0/8 (0.0%) | Temp: 0.664
[22:44:00] [INFO] Epoch  65/100 | Loss: 0.3803 | Acc: 73.9% | Exact: 0/8 (0.0%) | Temp: 0.642
[22:44:04] [INFO] Epoch  70/100 | Loss: 0.3521 | Acc: 76.2% | Exact: 0/8 (0.0%) | Temp: 0.620
[22:44:08] [INFO] Epoch  75/100 | Loss: 0.3381 | Acc: 76.9% | Exact: 0/8 (0.0%) | Temp: 0.599
[22:44:12] [INFO] 
--- Epoch 80 Detailed Diagnostics ---
[22:44:12] [INFO]   logits: shape=[8, 10, 9, 9], range=[-8.7019, 17.9292], mean=-4.0047
[22:44:12] [INFO]   attention_maps: shape=[8, 6, 9, 9], range=[0.0000, 0.9859], mean=0.0123
[22:44:12] [INFO]   stop_logits: shape=[8, 6], range=[-1.3171, 0.2490], mean=-0.8234
[22:44:12] [INFO] Per-sample accuracy:
[22:44:12] [INFO]   [1] 007bbfb7: 75.3% (✗)
[22:44:12] [INFO]   [6] 007bbfb7: 81.5% (✗)
[22:44:12] [INFO]   [4] 007bbfb7: 74.1% (✗)
[22:44:12] [INFO]   [7] 007bbfb7: 74.1% (✗)
[22:44:12] [INFO] Sample 0 prediction:
[22:44:12] [INFO]   Sample 0 (9x9): 61/81 pixels correct (75.3%)
[22:44:12] [INFO]     Target: [[7, 7, 0, 7, 7, 0, 0, 0, 0], [0, 0, 7, 0, 0, 7, 0, 0, 0], [7, 7, 7, 7, 7, 7, 0, 0, 0], [0, 0, 0, 0, 0, 0, 7, 7, 0], [0, 0, 0, 0, 0, 0, 0, 0, 7], [0, 0, 0, 0, 0, 0, 7, 7, 7], [7, 7, 0, 7, 7, 0, 7, 7, 0], [0, 0, 7, 0, 0, 7, 0, 0, 7], [7, 7, 7, 7, 7, 7, 7, 7, 7]]
[22:44:12] [INFO]     Pred:   [[7, 7, 0, 7, 7, 0, 7, 7, 7], [0, 0, 7, 0, 0, 7, 7, 0, 7], [7, 7, 7, 7, 7, 7, 7, 7, 7], [0, 0, 0, 0, 0, 0, 7, 7, 7], [0, 0, 0, 0, 0, 0, 7, 0, 7], [7, 7, 7, 0, 0, 0, 7, 0, 7], [7, 7, 7, 7, 7, 7, 7, 7, 7], [7, 0, 7, 0, 0, 0, 7, 0, 7], [7, 7, 7, 7, 7, 7, 7, 7, 7]]
[22:44:12] [INFO]   Gradient stats:
[22:44:12] [INFO]     encoder: avg_norm=6.0755e-03, max_norm=2.2726e-02
[22:44:12] [INFO]     feature_proj: avg_norm=9.0334e-03, max_norm=3.4702e-02
[22:44:12] [INFO]     context_encoder: avg_norm=2.7562e-04, max_norm=2.7924e-03
[22:44:12] [INFO]     context_injector: avg_norm=3.0550e-03, max_norm=7.1859e-03
[22:44:12] [INFO]     dsc: avg_norm=4.7772e-03, max_norm=2.9131e-02
[22:44:12] [INFO]     msre: avg_norm=5.6232e-04, max_norm=4.2643e-03
[22:44:12] [INFO]     solver: avg_norm=4.4206e-02, max_norm=3.9380e-01
[22:44:12] [INFO] Epoch  80/100 | Loss: 0.3259 | Acc: 76.7% | Exact: 0/8 (0.0%) | Temp: 0.578
[22:44:15] [INFO] Epoch  85/100 | Loss: 0.3134 | Acc: 77.6% | Exact: 0/8 (0.0%) | Temp: 0.559
[22:44:19] [INFO] Epoch  90/100 | Loss: 0.3464 | Acc: 78.2% | Exact: 0/8 (0.0%) | Temp: 0.540
[22:44:23] [INFO] Epoch  95/100 | Loss: 0.3039 | Acc: 77.6% | Exact: 0/8 (0.0%) | Temp: 0.521
[22:44:27] [INFO] 
--- Epoch 100 Detailed Diagnostics ---
[22:44:27] [INFO]   logits: shape=[8, 10, 9, 9], range=[-10.5043, 18.4150], mean=-5.1603
[22:44:27] [INFO]   attention_maps: shape=[8, 6, 9, 9], range=[0.0000, 0.9985], mean=0.0123
[22:44:27] [INFO]   stop_logits: shape=[8, 6], range=[-1.3674, 0.2479], mean=-0.7149
[22:44:27] [INFO] Per-sample accuracy:
[22:44:27] [INFO]   [5] 007bbfb7: 76.5% (✗)
[22:44:27] [INFO]   [7] 007bbfb7: 76.5% (✗)
[22:44:27] [INFO]   [0] 007bbfb7: 72.8% (✗)
[22:44:27] [INFO]   [1] 007bbfb7: 75.3% (✗)
[22:44:27] [INFO] Sample 0 prediction:
[22:44:27] [INFO]   Sample 0 (9x9): 62/81 pixels correct (76.5%)
[22:44:27] [INFO]     Target: [[7, 7, 0, 7, 7, 0, 0, 0, 0], [7, 0, 7, 7, 0, 7, 0, 0, 0], [7, 0, 7, 7, 0, 7, 0, 0, 0], [7, 7, 0, 0, 0, 0, 7, 7, 0], [7, 0, 7, 0, 0, 0, 7, 0, 7], [7, 0, 7, 0, 0, 0, 7, 0, 7], [7, 7, 0, 0, 0, 0, 7, 7, 0], [7, 0, 7, 0, 0, 0, 7, 0, 7], [7, 0, 7, 0, 0, 0, 7, 0, 7]]
[22:44:27] [INFO]     Pred:   [[7, 7, 0, 7, 7, 0, 0, 7, 7], [7, 0, 7, 7, 0, 7, 7, 0, 7], [7, 0, 7, 7, 0, 7, 7, 7, 7], [7, 7, 0, 0, 0, 0, 7, 0, 7], [7, 0, 7, 0, 0, 0, 7, 0, 7], [7, 0, 7, 0, 0, 0, 7, 0, 7], [7, 7, 7, 7, 7, 7, 7, 7, 7], [7, 0, 7, 0, 0, 0, 7, 0, 7], [7, 7, 7, 7, 7, 7, 7, 7, 7]]
[22:44:27] [INFO]   Gradient stats:
[22:44:27] [INFO]     encoder: avg_norm=3.9423e-03, max_norm=1.4482e-02
[22:44:27] [INFO]     feature_proj: avg_norm=5.7219e-03, max_norm=2.1558e-02
[22:44:27] [INFO]     context_encoder: avg_norm=2.7248e-04, max_norm=2.6391e-03
[22:44:27] [INFO]     context_injector: avg_norm=2.6665e-03, max_norm=7.0113e-03
[22:44:27] [INFO]     dsc: avg_norm=4.4984e-03, max_norm=2.7555e-02
[22:44:27] [INFO]     msre: avg_norm=6.2787e-04, max_norm=4.7568e-03
[22:44:27] [INFO]     solver: avg_norm=2.3653e-02, max_norm=2.2361e-01
[22:44:27] [INFO] Epoch 100/100 | Loss: 0.2994 | Acc: 77.5% | Exact: 0/8 (0.0%) | Temp: 0.503
[22:44:27] [INFO] 
[22:44:27] [INFO] ======================================================================
[22:44:27] [INFO] TRAINING COMPLETE
[22:44:27] [INFO] ======================================================================
[22:44:27] [INFO] Best Accuracy: 79.0% at epoch 87
[22:44:27] [INFO] Best Exact Match: 0/8
```
