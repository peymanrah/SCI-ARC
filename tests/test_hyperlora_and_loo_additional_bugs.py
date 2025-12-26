import pytest
import torch


def test_hyperlora_init_scale_is_respected_by_predictors():
    """
    Verify HyperLoRA.init_scale is properly threaded to LoRAPredictor instances.
    This was a bug that has been fixed: config.init_scale is now passed to all predictors.
    """
    from sci_arc.models.rlan_modules.hyper_lora import HyperLoRA, HyperLoRAConfig

    torch.manual_seed(0)
    h1 = HyperLoRA(
        config=HyperLoRAConfig(
            hidden_dim=32,
            context_dim=32,
            rank=2,
            scaling=0.1,
            dropout=0.0,
            init_scale=0.01,
        )
    )
    w1 = h1.gru_reset_lora.predict_A[-1].weight.detach().clone()

    torch.manual_seed(0)
    h2 = HyperLoRA(
        config=HyperLoRAConfig(
            hidden_dim=32,
            context_dim=32,
            rank=2,
            scaling=0.1,
            dropout=0.0,
            init_scale=0.5,
        )
    )
    w2 = h2.gru_reset_lora.predict_A[-1].weight.detach().clone()

    # If init_scale were plumbed through, the same RNG stream should yield weights scaled by init_scale,
    # so the tensors should not be identical.
    assert not torch.equal(w1, w2)


def test_loo_training_loss_hyperlora_interface_is_callable_end_to_end():
    """
    Verify LOOTrainingLoss is callable end-to-end with HyperLoRA.
    This was a bug that has been fixed: LOOTrainingLoss now uses the correct HyperLoRA interface.
    """
    from sci_arc.models import RLAN, RLANConfig
    from sci_arc.models.rlan_modules.loo_training import LOOTrainingLoss, LOOConfig

    config = RLANConfig(
        hidden_dim=32,
        num_colors=10,
        num_classes=10,
        max_grid_size=10,
        num_solver_steps=1,
        use_dsc=False,
        use_msre=False,
        use_lcr=False,
        use_sph=False,
        use_context_encoder=True,
        use_cross_attention_context=True,
        spatial_downsample=1,
        dropout=0.0,
        use_hyperlora=True,
        hyperlora_rank=2,
    )
    model = RLAN(config=config)
    hyper_lora = model.hyper_lora

    loo = LOOTrainingLoss(LOOConfig(enabled=True, min_pairs_for_loo=2), hidden_dim=config.hidden_dim)

    B, N, H, W = 2, 3, 8, 8
    # LOOTrainingLoss signature expects (B, N, C, H, W), but the rest of the codebase typically uses (B, H, W)
    support_inputs = torch.randint(0, 10, (B, N, 1, H, W), dtype=torch.long)
    support_targets = torch.randint(0, 10, (B, N, H, W), dtype=torch.long)
    support_features = torch.randn(B, N, config.hidden_dim, H, W)

    # LOOTrainingLoss returns a dict, not a tuple
    result = loo(
        hyper_lora=hyper_lora,
        rlan=model,
        support_inputs=support_inputs,
        support_targets=support_targets,
        support_features=support_features,
    )

    assert "loo_loss" in result
    assert torch.is_tensor(result["loo_loss"])
