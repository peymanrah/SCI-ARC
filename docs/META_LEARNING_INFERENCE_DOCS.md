# SCI-ARC Meta-Learning & Inference Documentation

## Bug Fixes Applied (December 2025)

### 1. LOO Tensor Mismatch (FIXED)
- **Issue**: LOOTrainingLoss passed 4D float tensor to GridEncoder expecting 3D int
- **Fix**: Now uses `model.context_encoder(input_grids, output_grids)` directly
- **Location**: `sci_arc/models/rlan_modules/loo_training.py` lines 167-185

### 2. Meta-Learning Disconnect (FIXED)
- **Issue**: LOO used `model.encoder` (input-only) but inference uses `context_encoder` (input+output pairs)
- **Impact**: HyperLoRA was trained on different distribution than inference
- **Fix**: LOO now calls `model.context_encoder(input_grids, output_grids, pair_mask)`
- **Location**: `sci_arc/models/rlan_modules/loo_training.py`

### 3. HyperLoRA Spatial Feature Dependency (FIXED)
- **Issue**: RLAN didn't enforce `needs_spatial_features=True` when HyperLoRA enabled
- **Fix**: Updated RLAN.__init__ to include `use_hyperlora` in dependency check
- **Location**: `sci_arc/models/rlan.py` lines 240-245

### 4. AvgPool vs MaxPool for HyperLoRA (TESTED)
- **Result**: AvgPool is OPTIMAL
- **Test**: Ran on 20 ARC tasks
  - AvgPool discriminability: 0.072
  - MaxPool discriminability: 0.036
- **Recommendation**: Keep current AdaptiveAvgPool2d implementation

---

## Context Injection: FiLM vs CrossAttention

### Current Codebase Status
Both modes exist and are configurable:

```python
# RLANConfig defaults
use_cross_attention_context: bool = False  # FiLM is default
use_solver_context: bool = True            # Solver cross-attention enabled by default
```

### FiLM Conditioning (Default)
```python
# When use_cross_attention_context=False
class ContextInjector:
    """FiLM: y = γ(context) * x + β(context)"""
    def forward(features, context):
        # context: (B, D) - pooled vector
        scale = 2 * sigmoid(scale_net(context))  # [0, 2] range
        shift = shift_net(context)
        return scale * features + shift
```

### CrossAttention Conditioning
```python
# When use_cross_attention_context=True
class CrossAttentionInjector:
    """Full cross-attention to spatial support features"""
    def forward(features, support_features):
        # support_features: (B, N, D, H, W)
        # Attend to all spatial positions across all pairs
        ...
```

**Note**: Even with FiLM, the Solver can still cross-attend to support features via `use_solver_context=True`.

---

## HyperLoRA Weight Adjustment in Solver

### How LoRA Deltas Modify GRU Weights

```python
# RecursiveSolver.ConvGRU.forward

def forward(x, h, lora_deltas=None):
    # 1. Compute base gate logits
    r_logits = reset_gate(combined)
    z_logits = update_gate(combined)
    
    # 2. Apply LoRA modulation (BEFORE activation)
    if lora_deltas is not None:
        if 'gru_reset' in lora_deltas:
            r_delta = _apply_lora_spatial(r_logits, lora_deltas['gru_reset'])
            r_logits = r_logits + r_delta  # Modulated!
            
        if 'gru_update' in lora_deltas:
            z_delta = _apply_lora_spatial(z_logits, lora_deltas['gru_update'])
            z_logits = z_logits + z_delta  # Modulated!
    
    # 3. Apply activations AFTER LoRA
    r = sigmoid(r_logits)
    z = sigmoid(z_logits)
    
    # 4. Candidate computation with LoRA
    if lora_deltas is not None and 'gru_candidate' in lora_deltas:
        cand_delta = _apply_lora_spatial(cand_logits, lora_deltas['gru_candidate'])
        cand_logits = cand_logits + cand_delta
    
    h_candidate = tanh(cand_logits)
    h_new = (1 - z) * h + z * h_candidate
    return h_new
```

### _apply_lora_spatial Implementation
```python
def _apply_lora_spatial(features, delta_w):
    """
    features: (B, D, H, W)
    delta_w: (B, D, D) predicted weight delta
    
    Computes: features @ delta_w (in spatial format)
    """
    B, D, H, W = features.shape
    features_flat = features.permute(0, 2, 3, 1).reshape(B, H*W, D)
    delta_out = torch.bmm(features_flat, delta_w)  # (B, H*W, D)
    return delta_out.reshape(B, H, W, D).permute(0, 3, 1, 2)
```

---

## HyperLoRA + LOO Training Integration

### Complete Flow in train_rlan.py

```python
# Training loop
for batch in dataloader:
    # === Step 1: Standard Forward ===
    outputs = model(
        test_input,
        train_inputs=train_inputs,
        train_outputs=train_outputs,
        return_intermediates=True,
    )
    losses = loss_fn(outputs['logits'], test_output, ...)
    
    # === Step 2: LOO Meta-Learning ===
    if loo_loss_fn is not None and model.hyper_lora is not None:
        if num_pairs >= min_pairs_for_loo:
            loo_result = loo_loss_fn(
                model=model,
                input_grids=train_inputs,
                output_grids=train_outputs,
                pair_mask=pair_mask,
            )
            loo_loss = loo_result['loo_loss'] * loss_weight
            losses['total_loss'] += loo_loss
    
    # === Step 3: Equivariance Loss ===
    if equiv_loss_fn is not None and model.hyper_lora is not None:
        support_features = outputs.get('support_features')
        lora_deltas = outputs.get('lora_deltas')
        if support_features is not None:
            original_context = lora_deltas.get('context')
            augmented_contexts = {
                aug: model.hyper_lora.pool_context(
                    apply_augmentation(support_features, aug)
                )
                for aug in ['rotate_90', 'flip_h', ...]
            }
            equiv_loss, _ = equiv_loss_fn(
                model.hyper_lora, original_context, augmented_contexts
            )
            losses['total_loss'] += equiv_loss
    
    # === Step 4: Backprop ===
    loss.backward()
    optimizer.step()
```

### LOO Loss Internal Logic

```python
class LOOTrainingLoss:
    def _forward_with_model(self, model, input_grids, output_grids, pair_mask):
        # CRITICAL: Encode PAIRS (input+output), not just inputs
        support_features = model.context_encoder(
            input_grids, output_grids, pair_mask
        )  # (B, N, D, H, W)
        
        total_loss = 0
        for holdout_idx in range(N):
            # Leave out pair i
            remaining = support_features[:, [j for j in range(N) if j != holdout_idx]]
            
            # Predict LoRA from N-1 pairs
            lora_deltas = model.hyper_lora(remaining)
            
            # Test on held-out pair
            logits = model.forward_with_lora(
                input_grids[:, holdout_idx],
                support_features,  # Full support for cross-attention
                lora_deltas,       # LoRA from N-1 only
            )
            total_loss += cross_entropy(logits, output_grids[:, holdout_idx])
        
        return {'loo_loss': total_loss / N, ...}
```

---

## ACW + TTA Hybrid Inference

### TTA (Test-Time Augmentation)

```python
def predict_with_tta(model, input_grid, train_inputs, train_outputs):
    augmentations = [
        'identity', 'rotate_90', 'rotate_180', 'rotate_270',
        'flip_h', 'flip_v', 'transpose', 'transpose_neg'
    ]
    
    all_predictions = []
    for aug in augmentations:
        # 1. Augment input AND support set
        aug_input = apply_augmentation(input_grid, aug)
        aug_train_in = apply_augmentation(train_inputs, aug)
        aug_train_out = apply_augmentation(train_outputs, aug)
        
        # 2. Predict
        pred = model.predict(aug_input, aug_train_in, aug_train_out)
        
        # 3. Inverse transform
        inv_pred = apply_augmentation(pred, aug, inverse=True)
        all_predictions.append(inv_pred)
    
    return all_predictions  # 8 predictions
```

### ACW (Augmented Confidence Weighting)

```python
class AugmentedConfidenceWeighting:
    def weighted_vote(self, predictions):
        # 1. Compute pairwise consistency (IoU)
        consistency_matrix = compute_pairwise_iou(predictions)
        consistency_scores = consistency_matrix.mean(dim=1)
        
        # 2. Group identical predictions
        unique_preds = group_by_hash(predictions)
        
        # 3. Weighted score = count × avg_consistency
        for pred_group in unique_preds:
            pred_group['weighted_score'] = (
                pred_group['count'] * pred_group['avg_consistency']
            )
        
        # 4. Winner = highest weighted score
        winner = max(unique_preds, key=lambda x: x['weighted_score'])
        return winner['grid'], sorted_candidates
```

### TRM-Style Majority Voting

```python
def majority_vote(predictions):
    vote_counts = {}
    for pred in predictions:
        key = pred.tobytes().hex()
        vote_counts[key] = vote_counts.get(key, 0) + 1
    winner = max(vote_counts, key=vote_counts.get)
    return grids[winner]
```

### Hybrid Voting

```python
def hybrid_vote(predictions):
    trm_winner = majority_vote(predictions)
    acw_winner, _ = acw.weighted_vote(predictions)
    
    if torch.equal(trm_winner, acw_winner):
        return trm_winner, {"method": "consensus"}
    else:
        return acw_winner, {"method": "acw_override"}
```

### Voting Example

```
8 TTA predictions:

Prediction A: [[1,2,0],[0,0,0],[0,0,0]] × 5
Prediction B: [[1,2,0],[0,1,0],[0,0,0]] × 2
Prediction C: [[0,0,0],[0,0,0],[0,0,0]] × 1

TRM Vote:
  A: 5 → Winner

ACW Vote:
  A: consistency=0.85, score=5×0.85=4.25 → Winner
  B: consistency=0.60, score=2×0.60=1.20
  C: consistency=0.20, score=1×0.20=0.20

Hybrid: Both agree → A with high confidence
```

### When One Method is Disabled

```python
def predict_ensemble(use_tta=True, use_acw=True):
    if not use_tta and not use_acw:
        return model.predict(...)  # Single prediction
    
    if use_tta:
        predictions = predict_with_tta(...)
    else:
        predictions = [model.predict(...)]  # Just identity
    
    if use_acw:
        winner, _ = acw.weighted_vote(predictions)
    else:
        winner = majority_vote(predictions)
    
    return winner
```

---

## Summary Table

| Component | Location | Status |
|-----------|----------|--------|
| FiLM Conditioning | `context_encoder.py:ContextInjector` | ✅ Default |
| CrossAttention | `context_encoder.py:CrossAttentionInjector` | ✅ Optional |
| LoRA in Solver | `recursive_solver.py:ConvGRU` | ✅ Implemented |
| LOO Training | `loo_training.py:LOOTrainingLoss` | ✅ Fixed |
| Equivariance | `loo_training.py:AugmentationEquivarianceLoss` | ✅ Implemented |
| TTA | `rlan.py:predict_with_acw` | ✅ 8 dihedral |
| ACW | `acw.py:weighted_vote` | ✅ Consistency voting |
| Hybrid | `acw.py:hybrid_vote` | ✅ TRM+ACW ensemble |
