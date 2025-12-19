#!/usr/bin/env python
"""
Test script to investigate solver degradation issue.

The problem: Step 0 has lowest loss, later steps have higher loss (~0.5% worse).
This is counterintuitive - iterative refinement should improve, not degrade.

This script investigates:
1. GRU update gate values (is z too high, ignoring previous state?)
2. Hidden state drift across steps
3. Output logits variance per step
4. Residual connection effectiveness
"""

import torch
import torch.nn.functional as F
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def analyze_solver_behavior():
    """Analyze what happens during solver iterations."""
    
    from sci_arc.models.rlan import RLAN, RLANConfig
    from sci_arc.training.rlan_loss import WeightedStablemaxLoss
    
    print("=" * 70)
    print("SOLVER DEGRADATION INVESTIGATION")
    print("=" * 70)
    
    # Load model config
    config = RLANConfig(
        hidden_dim=384,
        max_clues=6,
        num_solver_steps=6,
        max_grid_size=30,
    )
    
    model = RLAN(config=config)
    model.eval()
    
    # Create sample input
    B, H, W = 2, 15, 15
    input_grid = torch.randint(0, 10, (B, H, W))
    target_grid = torch.randint(0, 10, (B, H, W))
    
    # Forward pass with hooks to capture internal states
    solver_states = {
        'h_states': [],
        'z_gates': [],
        'r_gates': [],
        'step_predictions': [],
    }
    
    def hook_gru_gates(module, input, output):
        """Hook to capture GRU gate values."""
        x, h = input[0], input[1] if len(input) > 1 else None
        # We need to re-compute the gates to capture them
        # This is a bit wasteful but works for debugging
        if h is not None:
            combined = torch.cat([x, h], dim=1)
            with torch.no_grad():
                r = torch.sigmoid(module.reset_gate(combined))
                z = torch.sigmoid(module.update_gate(combined))
                solver_states['z_gates'].append(z.mean().item())
                solver_states['r_gates'].append(r.mean().item())
                solver_states['h_states'].append(output.detach().clone())
    
    # Register hook
    hook = model.solver.gru.register_forward_hook(hook_gru_gates)
    
    try:
        with torch.no_grad():
            outputs = model(input_grid, return_all_steps=True, return_intermediates=True)
            all_logits = outputs['all_logits']
            
            print(f"\n1. SOLVER STEP ANALYSIS")
            print("-" * 50)
            print(f"   Number of solver steps: {len(all_logits)}")
            
            # Analyze each step's predictions
            loss_fn = WeightedStablemaxLoss(bg_weight_cap=2.0, fg_weight_cap=5.0)
            
            print(f"\n2. PER-STEP METRICS")
            print("-" * 50)
            
            for step_idx, step_logits in enumerate(all_logits):
                step_loss = loss_fn(step_logits, target_grid).item()
                step_pred = step_logits.argmax(dim=1)
                step_acc = (step_pred == target_grid).float().mean().item() * 100
                
                # Logit statistics
                logit_std = step_logits.std().item()
                logit_max = step_logits.max().item()
                logit_min = step_logits.min().item()
                
                # Softmax entropy (uncertainty)
                probs = F.softmax(step_logits, dim=1)
                entropy = -(probs * probs.log().clamp(-100, 0)).sum(dim=1).mean().item()
                
                z_val = solver_states['z_gates'][step_idx] if step_idx < len(solver_states['z_gates']) else 0.0
                r_val = solver_states['r_gates'][step_idx] if step_idx < len(solver_states['r_gates']) else 0.0
                
                print(f"   Step {step_idx}: Loss={step_loss:.4f}, Acc={step_acc:.1f}%, "
                      f"Entropy={entropy:.3f}, σ(logits)={logit_std:.2f}")
                print(f"            z_gate={z_val:.3f}, r_gate={r_val:.3f}")
            
            # Analyze hidden state drift
            print(f"\n3. HIDDEN STATE DRIFT")
            print("-" * 50)
            
            if len(solver_states['h_states']) >= 2:
                h0 = solver_states['h_states'][0]
                for step_idx, h_t in enumerate(solver_states['h_states'][1:], 1):
                    diff = (h_t - h0).abs().mean().item()
                    cos_sim = F.cosine_similarity(h0.flatten(), h_t.flatten(), dim=0).item()
                    print(f"   Step 0 → Step {step_idx}: "
                          f"L1 diff = {diff:.4f}, Cosine sim = {cos_sim:.4f}")
            
            # Analyze prediction consistency
            print(f"\n4. PREDICTION CONSISTENCY")
            print("-" * 50)
            
            pred_0 = all_logits[0].argmax(dim=1)
            for step_idx, step_logits in enumerate(all_logits[1:], 1):
                pred_t = step_logits.argmax(dim=1)
                changed = (pred_t != pred_0).float().mean().item() * 100
                improved = ((pred_t == target_grid) & (pred_0 != target_grid)).float().sum().item()
                regressed = ((pred_t != target_grid) & (pred_0 == target_grid)).float().sum().item()
                print(f"   Step 0 → Step {step_idx}: "
                      f"Changed={changed:.1f}%, Improved={int(improved)}, Regressed={int(regressed)}")
            
            # DIAGNOSIS
            print(f"\n5. DIAGNOSIS")
            print("=" * 50)
            
            z_gates = solver_states['z_gates']
            if z_gates:
                avg_z = sum(z_gates) / len(z_gates)
                print(f"   Average z_gate: {avg_z:.3f}")
                if avg_z > 0.7:
                    print("   → HIGH z_gate: GRU updates heavily, ignoring previous state")
                    print("   → Solution: Add z_gate regularization or reduce GRU capacity")
                elif avg_z < 0.3:
                    print("   → LOW z_gate: GRU barely updates, step 0 dominates")
                else:
                    print("   → MODERATE z_gate: Balanced update behavior")
            
            # Check if later steps regress
            losses = [loss_fn(step_logits, target_grid).item() for step_logits in all_logits]
            if losses[-1] > losses[0]:
                pct_increase = (losses[-1] - losses[0]) / losses[0] * 100
                print(f"\n   ⚠ DEGRADATION DETECTED: Step 6 loss is {pct_increase:.1f}% higher than Step 0")
                print("   Possible causes:")
                print("   1. Residual connection too weak (currently 0.1)")
                print("   2. GRU overwriting good step 0 features")
                print("   3. Output head not optimized for multi-step")
                print("   4. Deep supervision weight too low for early steps")
            else:
                print(f"\n   ✓ HEALTHY: Loss decreases from step 0 to step 6")
    
    finally:
        hook.remove()
    
    # Suggestion box
    print(f"\n6. POTENTIAL FIXES")
    print("=" * 50)
    print("""
    Option A: Increase residual connection strength
    - Change from 0.9 * h_new + 0.1 * h_initial
    - To:       0.7 * h_new + 0.3 * h_initial
    - This preserves more of step 0's good features

    Option B: Add skip connection from step 0 output
    - Instead of just h_initial, also add step 0's logits
    - step_t_logits = step_t_logits + 0.1 * step_0_logits

    Option C: Reduce number of solver steps
    - If degradation starts at step 3, use 3 steps instead of 6
    - Check per-step loss to find optimal step count

    Option D: Add output refinement instead of hidden refinement
    - Output probability estimates at step 0
    - Later steps refine the probability, not the hidden state
    - This is more stable as probabilities are bounded

    Option E: Learnable residual weights
    - Replace fixed 0.9/0.1 with learnable per-step weights
    - Model can learn when to preserve vs update
    """)


if __name__ == "__main__":
    analyze_solver_behavior()
