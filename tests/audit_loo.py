
import torch
import torch.nn as nn
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sci_arc.models.rlan import RLAN, RLANConfig
from sci_arc.models.rlan_modules.loo_training import LOOTrainingLoss, LOOConfig

def test_loo_context_encoder_usage():
    print("\n=== Testing LOO Training Loss Context Encoder Usage ===")
    
    # 1. Setup Model
    config = RLANConfig(
        hidden_dim=32,
        use_hyperlora=True,
        use_context_encoder=True,
        use_solver_context=True
    )
    model = RLAN(config=config)
    
    # 2. Setup LOO Loss
    loo_config = LOOConfig(enabled=True, min_pairs_for_loo=2)
    loo_loss_fn = LOOTrainingLoss(loo_config, hidden_dim=32)
    
    # 3. Register Hook on ContextEncoder
    context_encoder_called = False
    def hook(module, input, output):
        nonlocal context_encoder_called
        context_encoder_called = True
        print("  [Hook] ContextEncoder called!")
        
    handle = model.context_encoder.register_forward_hook(hook)
    
    # 4. Create Dummy Data
    B, N, H, W = 2, 3, 10, 10
    input_grids = torch.randint(0, 10, (B, N, H, W))
    output_grids = torch.randint(0, 10, (B, N, H, W))
    pair_mask = torch.ones((B, N), dtype=torch.bool)
    
    # 5. Run LOO Loss
    print("  Running LOO Loss...")
    try:
        result = loo_loss_fn(
            model=model,
            input_grids=input_grids,
            output_grids=output_grids,
            pair_mask=pair_mask
        )
        print(f"  LOO Loss computed: {result['loo_loss'].item()}")
    except Exception as e:
        print(f"  Error during LOO Loss: {e}")
        import traceback
        traceback.print_exc()
    
    # 6. Verify Hook
    if context_encoder_called:
        print("  [PASS] ContextEncoder WAS called (Unexpected if bug exists).")
    else:
        print("  [FAIL] ContextEncoder was NOT called. (CONFIRMS BUG)")
        print("  Explanation: LOOTrainingLoss is using model.encoder (inputs only) instead of model.context_encoder (input+output pairs).")
        print("  This means HyperLoRA is trained on inputs only, but used on pairs at inference.")

    handle.remove()

if __name__ == "__main__":
    test_loo_context_encoder_usage()
