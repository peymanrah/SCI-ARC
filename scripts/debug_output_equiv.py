#!/usr/bin/env python
"""
Diagnostic script to reproduce OUTPUT_EQUIV error with full traceback.
"""
import sys
import traceback
import torch
import torch.nn.functional as F
import yaml
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    # Load config
    config_path = project_root / "configs" / "rlan_stable_dev_ablation.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Import modules
    from sci_arc.models import RLAN, RLANConfig
    from sci_arc.models.rlan_modules.loo_training import OutputEquivarianceLoss, EquivarianceConfig

    print("Setting up model and loss...")
    model_cfg = config["model"]
    rlan_config = RLANConfig(
        hidden_dim=model_cfg.get("hidden_dim", 256),
        num_colors=model_cfg.get("num_colors", 10),
        num_classes=model_cfg.get("num_classes", 10),
        max_grid_size=model_cfg.get("max_grid_size", 30),
        max_clues=model_cfg.get("max_clues", 7),
        num_predicates=model_cfg.get("num_predicates", 32),
        num_solver_steps=model_cfg.get("num_solver_steps", 4),
        use_context_encoder=model_cfg.get("use_context_encoder", True),
        use_dsc=model_cfg.get("use_dsc", True),
        use_msre=model_cfg.get("use_msre", True),
        use_hyperlora=model_cfg.get("use_hyperlora", False),
        use_hpm=model_cfg.get("use_hpm", False),
        use_solver_context=model_cfg.get("use_solver_context", True),
        use_cross_attention_context=model_cfg.get("use_cross_attention_context", True),
    )

    model = RLAN(config=rlan_config)
    model.eval()

    # Create output equiv loss
    output_equiv_loss_fn = OutputEquivarianceLoss(
        config=EquivarianceConfig(enabled=True, loss_weight=0.01, num_augmentations=2),
        loss_type="kl",
        mask_to_target=True,
    )

    print("Creating test batch (mimicking real training batch)...")
    # Mimic real batch shape - use 30x30 like the log showed
    B, H, W = 2, 30, 30
    N = 3  # Training pairs

    test_inputs = torch.randint(0, 10, (B, H, W))
    train_inputs = torch.randint(0, 10, (B, N, H, W))
    train_outputs = torch.randint(0, 10, (B, N, H, W))
    pair_mask = torch.ones(B, N, dtype=torch.bool)

    print(f"test_inputs shape: {test_inputs.shape}")
    print(f"train_inputs shape: {train_inputs.shape}")
    print(f"train_outputs shape: {train_outputs.shape}")
    print(f"pair_mask shape: {pair_mask.shape}")

    # First do a normal forward to get original_logits
    print("Running initial forward pass...")
    with torch.no_grad():
        outputs = model(
            test_inputs,
            train_inputs=train_inputs,
            train_outputs=train_outputs,
            pair_mask=pair_mask,
            temperature=1.0,
            return_intermediates=False,
        )
    original_logits = outputs["logits"]
    print(f"original_logits shape: {original_logits.shape}")

    # Now try output equiv loss - this should trigger the error
    print("\nTrying output equivariance loss (this may fail)...")
    try:
        # Create target mask like training does
        test_targets = torch.randint(0, 10, (B, H, W))
        target_mask = test_targets != -100

        result, metrics = output_equiv_loss_fn(
            model=model,
            test_inputs=test_inputs,
            train_inputs=train_inputs,
            train_outputs=train_outputs,
            pair_mask=pair_mask,
            original_logits=original_logits.detach().requires_grad_(True),
            temperature=1.0,
            num_augmentations=2,
            target_mask=target_mask,
        )
        print(f"Success! Loss: {result.item():.4f}")
    except Exception as e:
        print(f"\n*** ERROR CAUGHT ***")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {e}")
        print("\n*** FULL TRACEBACK ***")
        traceback.print_exc()


if __name__ == "__main__":
    main()
