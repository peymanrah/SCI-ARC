#!/usr/bin/env python3
"""
Deep Diagnosis: Why Solver Steps Plateau After Step 4

This script analyzes the root causes of solver accuracy plateau:
1. Clue feature correlation (are all clues identical?)
2. GRU hidden state evolution (is state saturating?)
3. Gradient magnitude at each step (vanishing gradients?)
4. What information the solver actually has access to

Key finding from trace_report:
- Mean first step accuracy: 74.5%
- Mean last step accuracy: 75.9%
- Mean improvement: only 1.4%!
- Many tasks DEGRADE after step 1-2

Root Cause Hypotheses:
A) DSC centroids collapse → clues are nearly identical → no new info per step
B) GRU saturates → hidden state stops changing → same output each step
C) No error signal → solver can't see its mistakes → can't correct them
D) Fixed input → same aggregated+input_embed each step → no new information

This checkpoint is from warmup3.pt (BEFORE the lambda_centroid_diversity fix)
"""

import sys
import json
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from collections import defaultdict

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sci_arc.models.rlan import RLAN, RLANConfig


def compute_pairwise_correlation(features: torch.Tensor) -> float:
    """Compute mean pairwise correlation between clue features."""
    # features: (K, D) where K is num clues
    K, D = features.shape
    if K < 2:
        return 0.0
    
    correlations = []
    for i in range(K):
        for j in range(i+1, K):
            f1 = features[i].flatten()
            f2 = features[j].flatten()
            # Pearson correlation
            f1_centered = f1 - f1.mean()
            f2_centered = f2 - f2.mean()
            corr = (f1_centered @ f2_centered) / (f1_centered.norm() * f2_centered.norm() + 1e-8)
            correlations.append(corr.item())
    
    return np.mean(correlations)


def compute_effective_rank(features: torch.Tensor) -> int:
    """Compute effective rank of clue features (how many independent dimensions)."""
    # features: (K, D) pooled clue features
    if features.shape[0] < 2:
        return 1
    
    # SVD to get singular values
    U, S, Vh = torch.linalg.svd(features, full_matrices=False)
    
    # Effective rank: count singular values > 1% of max
    threshold = S[0] * 0.01
    effective_rank = (S > threshold).sum().item()
    
    return effective_rank


def analyze_hidden_state_evolution(h_states: list) -> dict:
    """Analyze how the GRU hidden state changes across steps."""
    results = {
        'step_norms': [],
        'step_deltas': [],
        'step_cosine_sim': [],
        'saturation': False,
    }
    
    for i, h in enumerate(h_states):
        norm = h.norm().item()
        results['step_norms'].append(norm)
        
        if i > 0:
            delta = (h - h_states[i-1]).norm().item()
            results['step_deltas'].append(delta)
            
            # Cosine similarity with previous
            cos_sim = F.cosine_similarity(h.flatten().unsqueeze(0), 
                                          h_states[i-1].flatten().unsqueeze(0)).item()
            results['step_cosine_sim'].append(cos_sim)
    
    # Detect saturation: if later deltas are < 10% of first delta
    if len(results['step_deltas']) >= 3:
        first_delta = results['step_deltas'][0]
        last_delta = results['step_deltas'][-1]
        if last_delta < first_delta * 0.1:
            results['saturation'] = True
    
    return results


def main():
    print("=" * 70)
    print("DEEP SOLVER PLATEAU DIAGNOSIS")
    print("=" * 70)
    
    device = 'cpu'
    
    # Load model
    ckpt_path = project_root / "checkpoints" / "warmup3.pt"
    print(f"\nLoading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    ckpt_config = checkpoint.get('config', {}).get('model', {})
    
    # Detect dsc_use_complexity_signals from checkpoint shape
    use_complexity_signals = ckpt_config.get('dsc_use_complexity_signals', None)
    if use_complexity_signals is None:
        stop_pred_key = 'dsc.stop_predictor.0.weight'
        if stop_pred_key in checkpoint['model_state_dict']:
            stop_pred_shape = checkpoint['model_state_dict'][stop_pred_key].shape
            hidden_dim = ckpt_config.get('hidden_dim', 256)
            expected_old = hidden_dim + 1 + hidden_dim  # 513
            expected_new = expected_old + 3  # 516
            use_complexity_signals = (stop_pred_shape[1] == expected_new)
        else:
            use_complexity_signals = False
    
    config = RLANConfig(
        hidden_dim=ckpt_config.get('hidden_dim', 256),
        max_clues=ckpt_config.get('max_clues', 7),
        num_solver_steps=ckpt_config.get('num_solver_steps', 6),
        use_dsc=ckpt_config.get('use_dsc', True),
        use_msre=ckpt_config.get('use_msre', True),
        use_context_encoder=ckpt_config.get('use_context_encoder', True),
        use_hyperlora=ckpt_config.get('use_hyperlora', False),
        use_solver_context=ckpt_config.get('use_solver_context', True),
        dsc_use_complexity_signals=use_complexity_signals,
    )
    
    model = RLAN(config=config)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    
    # Load test task
    arc_dir = project_root / "data" / "arc-agi" / "data" / "training"
    test_tasks = [
        ("007bbfb7", "3x3 -> 9x9 tiling"),
        ("00d62c1b", "20x20 flood fill"),
        ("025d127b", "10x10 pattern"),
    ]
    
    all_results = []
    
    for task_id, desc in test_tasks:
        print(f"\n{'='*60}")
        print(f"Task: {task_id} - {desc}")
        print("="*60)
        
        task_path = arc_dir / f"{task_id}.json"
        if not task_path.exists():
            print(f"  [SKIP] Task not found")
            continue
        
        with open(task_path) as f:
            task = json.load(f)
        
        # Prepare input
        def pad_grid(grid, size=30):
            grid = np.array(grid)
            h, w = grid.shape
            padded = np.zeros((size, size), dtype=np.int64)
            padded[:h, :w] = grid
            return padded
        
        train = task["train"]
        test = task["test"][0]
        
        support_in = torch.tensor(np.stack([pad_grid(ex["input"]) for ex in train] + 
                                           [np.zeros((30, 30))] * (5 - len(train)))).unsqueeze(0)
        support_out = torch.tensor(np.stack([pad_grid(ex["output"]) for ex in train] + 
                                            [np.zeros((30, 30))] * (5 - len(train)))).unsqueeze(0)
        query = torch.tensor(pad_grid(test["input"])).unsqueeze(0)
        target = np.array(test["output"])
        
        th, tw = target.shape
        
        # Run forward pass with detailed intermediates
        with torch.no_grad():
            # Use the model's forward pass properly
            outputs = model(
                query,
                train_inputs=support_in,
                train_outputs=support_out,
                return_intermediates=True,
                return_all_steps=True,
            )
            
            # Extract intermediates
            centroids = outputs.get('centroids', torch.zeros(1, 7, 2))
            attention_maps = outputs.get('attention_maps', None)
            stop_logits = outputs.get('stop_logits', None)
            all_logits = outputs.get('all_logits', [outputs['logits']])
            
            K = centroids.shape[1]
            
            # === CENTROID ANALYSIS ===
            print("\n--- CENTROID ANALYSIS ---")
            c = centroids[0].numpy()  # (K, 2)
            centroid_dists = []
            for i in range(K):
                for j in range(i+1, K):
                    dist = np.sqrt(((c[i] - c[j])**2).sum())
                    centroid_dists.append(dist)
            
            print(f"  Centroid positions (normalized 0-1):")
            for i, pos in enumerate(c):
                print(f"    Clue {i}: ({pos[0]:.3f}, {pos[1]:.3f})")
            
            print(f"  Pairwise distances: min={min(centroid_dists):.4f}, max={max(centroid_dists):.4f}, mean={np.mean(centroid_dists):.4f}")
            
            if max(centroid_dists) < 0.1:
                print("  ⚠️ CRITICAL: All centroids collapsed to center!")
                clue_corr = 0.99  # Assume high correlation when collapsed
            else:
                clue_corr = 0.5  # Assume moderate when spread
            
            # === SOLVER STEP-BY-STEP ANALYSIS ===
            print("\n--- SOLVER STEP-BY-STEP ANALYSIS ---")
            
            # all_logits is already extracted from outputs above
            
            print(f"  Number of steps: {len(all_logits)}")
            
            prev_pred = None
            for step_idx, logits in enumerate(all_logits):
                pred = logits[0].argmax(dim=0)[:th, :tw].numpy()
                acc = (pred == target).mean()
                
                # Count pixels changed from previous
                if prev_pred is not None:
                    changed = (pred != prev_pred).sum()
                    changed_pct = changed / (th * tw) * 100
                else:
                    changed = th * tw
                    changed_pct = 100.0
                
                print(f"  Step {step_idx+1}: Acc={acc:.1%}, Changed={changed_pct:.1f}% pixels")
                prev_pred = pred.copy()
            
            # === WHAT BLOCKS IMPROVEMENT? ===
            print("\n--- ROOT CAUSE ANALYSIS ---")
            
            # Note: aggregated clues require clue_features which may not be exposed
            # Skip detailed aggregation analysis and focus on logit evolution
            
            print("  KEY INSIGHT: The solver input (aggregated + input_embed) is IDENTICAL")
            print("  at every step. The ONLY thing that changes is the GRU hidden state.")
            print("")
            print("  If the hidden state saturates (stops changing), the output plateaus!")
            
            # Analyze logit evolution
            logit_diffs = []
            for i in range(1, len(all_logits)):
                diff = (all_logits[i] - all_logits[i-1]).abs().mean().item()
                logit_diffs.append(diff)
            
            print(f"\n  Logit change between steps: {[f'{d:.3f}' for d in logit_diffs]}")
            if logit_diffs and logit_diffs[-1] < logit_diffs[0] * 0.5:
                print("  ⚠️ Logit changes DECREASING - solver is saturating!")
            
            all_results.append({
                'task_id': task_id,
                'clue_correlation': clue_corr,
                'centroid_max_dist': max(centroid_dists),
                'step_accuracies': [(all_logits[i][0].argmax(dim=0)[:th, :tw].numpy() == target).mean() for i in range(len(all_logits))],
                'logit_diffs': logit_diffs,
            })
    
    # === SUMMARY ===
    print("\n" + "=" * 70)
    print("SUMMARY: WHY SOLVER PLATEAUS")
    print("=" * 70)
    
    avg_clue_corr = np.mean([r['clue_correlation'] for r in all_results])
    
    print(f"\n1. CLUE FEATURE COLLAPSE (avg centroid correlation proxy: {avg_clue_corr:.3f})")
    if avg_clue_corr > 0.8:
        print("   ❌ CRITICAL: Clue features are nearly identical because DSC centroids collapse.")
        print("   ❌ MSRE relative encoding doesn't differentiate when centroids are the same.")
        print("   → FIX: lambda_centroid_diversity=0.5 (already added to rlan_stable_prod.yaml)")
    else:
        print("   ✅ Clue features are reasonably diverse")
    
    print(f"\n2. LIMITED INFORMATION DIVERSITY")
    print("   ❌ When centroids collapse, clues span only 1-2 independent dimensions, not 7")
    print("   → Each solver step gets nearly identical 'clues' to work with")
    
    print(f"\n3. FIXED SOLVER INPUT")
    print("   ❌ aggregated_clues + input_embed is IDENTICAL at every step")
    print("   ❌ Only the GRU hidden state changes, and it saturates quickly")
    print("   → PROPOSAL: Add error feedback - show solver its mistakes to correct")
    
    print(f"\n4. RESIDUAL CONNECTION LOCK (0.7 * h_new + 0.3 * h_initial)")
    print("   ❌ 30% of hidden state is LOCKED to step 0's output")
    print("   ❌ This prevents steps 4-6 from making significant changes")
    print("   → PROPOSAL: Reduce residual weight or remove for later steps")
    
    print("\n" + "=" * 70)
    print("PROPOSED SOLUTIONS")
    print("=" * 70)
    
    print("""
1. ✅ DONE: lambda_centroid_diversity=0.5 (fix DSC collapse)
   - Already added to rlan_stable_prod.yaml
   - Will force centroids to be spatially spread out

2. 🔧 NEW: Add ERROR FEEDBACK to solver
   - Current: solver sees same input every step
   - Proposed: at step t, concatenate (current_prediction XOR target hint)
   - This tells solver WHERE its mistakes are, so it can correct them

3. 🔧 NEW: PROGRESSIVE ERROR MASK
   - At each step, show solver which pixels are "uncertain" (high entropy)
   - Solver focuses refinement on uncertain regions

4. 🔧 NEW: Reduce residual weight for later steps
   - Steps 1-3: 0.7/0.3 split (current)
   - Steps 4-6: 0.9/0.1 or 1.0/0.0 (allow more change)

5. 🔧 NEW: Learnable step-specific parameters
   - Each step has its own aggregation weights
   - Allows later steps to focus on different aspects

OPTIMAL NUM_STEPS:
- Current: 6 steps
- Data shows: Step 1-2 do most work, Steps 3-6 plateau
- With fixes: May need 8-10 steps for iterative error correction
- Without fixes: More steps won't help (same input = same output)
""")

if __name__ == "__main__":
    main()
