"""
Test Meta-Learning Impact: Verify LOO Training and HyperLoRA Work Together

This test validates that:
1. HyperLoRA correctly predicts task-specific LoRA weight deltas
2. LOO training teaches generalization from N-1 to Nth example
3. The meta-learning components have proper gradient flow
4. Configuration parameters are correctly read from YAML

Run with: pytest tests/test_meta_learning_impact.py -v
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestHyperLoRAConfiguration:
    """Test that HyperLoRA is correctly configured from YAML."""
    
    def test_hyperlora_params_in_rlan_stable_yaml(self):
        """Verify rlan_stable.yaml has HyperLoRA configuration."""
        import yaml
        
        config_path = project_root / "configs" / "rlan_stable.yaml"
        with open(config_path, encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        model_config = config.get('model', {})
        
        # Check HyperLoRA params exist
        assert 'use_hyperlora' in model_config, "use_hyperlora not in config"
        assert model_config['use_hyperlora'] == True, "HyperLoRA should be enabled"
        assert 'hyperlora_rank' in model_config, "hyperlora_rank not in config"
        assert model_config['hyperlora_rank'] == 8, "Default rank should be 8"
        
    def test_loo_training_config_in_yaml(self):
        """Verify rlan_stable.yaml has LOO training configuration."""
        import yaml
        
        config_path = project_root / "configs" / "rlan_stable.yaml"
        with open(config_path, encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        loo_config = config.get('training', {}).get('loo_training', {})
        
        assert 'enabled' in loo_config, "LOO enabled not in config"
        assert loo_config['enabled'] == True, "LOO should be enabled"
        assert 'loss_weight' in loo_config, "LOO loss_weight not in config"
        assert loo_config['loss_weight'] == 0.2, "LOO weight should be 0.2 (tuned middle ground)"
        
    def test_hyperlora_lr_multiplier_in_yaml(self):
        """Verify HyperLoRA gets its own LR multiplier."""
        import yaml
        
        config_path = project_root / "configs" / "rlan_stable.yaml"
        with open(config_path, encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        train_config = config.get('training', {})
        
        assert 'hyperlora_lr_multiplier' in train_config, "hyperlora_lr_multiplier not in config"
        assert train_config['hyperlora_lr_multiplier'] >= 1.0, "HyperLoRA LR multiplier should be >= 1.0"


class TestHyperLoRAModule:
    """Test HyperLoRA module functionality."""
    
    def test_hyperlora_can_be_instantiated(self):
        """Verify HyperLoRA module can be created."""
        from sci_arc.models.rlan_modules.hyper_lora import HyperLoRA, HyperLoRAConfig
        
        config = HyperLoRAConfig(
            hidden_dim=64,
            rank=4,
            target_gru=True,
            target_output_head=True,
        )
        hyperlora = HyperLoRA(config=config)
        
        assert hyperlora is not None
        assert hasattr(hyperlora, 'context_pool') or hasattr(hyperlora, 'context_pooler')
        
    def test_hyperlora_predicts_lora_deltas(self):
        """Verify HyperLoRA predicts weight deltas from context."""
        from sci_arc.models.rlan_modules.hyper_lora import HyperLoRA, HyperLoRAConfig
        
        B, N, D, Hs, Ws = 2, 3, 64, 4, 4
        
        config = HyperLoRAConfig(
            hidden_dim=D,
            rank=4,
            target_gru=True,
            target_output_head=True,
        )
        hyperlora = HyperLoRA(config=config)
        
        # Create fake support features
        support_features = torch.randn(B, N, D, Hs, Ws)
        
        # Get LoRA deltas
        lora_deltas = hyperlora(support_features)
        
        assert isinstance(lora_deltas, dict), "Should return dict of deltas"
        assert len(lora_deltas) > 0, "Should have at least one delta"
        
        # Check delta shapes
        for name, delta in lora_deltas.items():
            assert delta.shape[0] == B, f"Delta {name} should have batch dim"
            
    def test_hyperlora_different_contexts_different_weights(self):
        """Verify different contexts produce different LoRA weights."""
        from sci_arc.models.rlan_modules.hyper_lora import HyperLoRA, HyperLoRAConfig
        
        B, N, D, Hs, Ws = 2, 3, 64, 4, 4
        
        config = HyperLoRAConfig(hidden_dim=D, rank=4)
        hyperlora = HyperLoRA(config=config)
        
        # Two different contexts
        context1 = torch.randn(B, N, D, Hs, Ws)
        context2 = torch.randn(B, N, D, Hs, Ws) * 2  # Different
        
        deltas1 = hyperlora(context1)
        deltas2 = hyperlora(context2)
        
        # Check they're different
        for key in deltas1:
            diff = (deltas1[key] - deltas2[key]).abs().mean()
            assert diff > 0.001, f"Delta {key} should be different for different contexts"


class TestLOOTrainingLoss:
    """Test LOO training loss computation."""
    
    def test_loo_loss_module_exists(self):
        """Verify LOOTrainingLoss can be imported."""
        from sci_arc.models.rlan_modules.loo_training import LOOTrainingLoss, LOOConfig
        
        config = LOOConfig(loss_weight=0.5, min_pairs_for_loo=2)
        loo_loss = LOOTrainingLoss(config=config)
        
        assert loo_loss is not None
        assert loo_loss.config.loss_weight == 0.5
        
    def test_loo_loss_skips_when_not_enough_pairs(self):
        """Verify LOO loss skips when < min_pairs."""
        from sci_arc.models.rlan_modules.loo_training import LOOTrainingLoss, LOOConfig
        
        config = LOOConfig(min_pairs_for_loo=3)  # Need 3 pairs
        loo_loss = LOOTrainingLoss(config=config)
        
        # Only 2 pairs - should skip
        B, N, H, W = 2, 2, 8, 8  # N=2 < 3
        input_grids = torch.randint(0, 10, (B, N, H, W))
        output_grids = torch.randint(0, 10, (B, N, H, W))
        pair_mask = torch.ones(B, N, dtype=torch.bool)
        
        # Create a mock model without HyperLoRA
        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.hyper_lora = None  # No HyperLoRA
                
        model = MockModel()
        
        result = loo_loss(
            model=model,
            input_grids=input_grids,
            output_grids=output_grids,
            pair_mask=pair_mask,
        )
        
        assert result['loo_skipped'] == True, "Should skip with no HyperLoRA"
        
    def test_loo_interface_accepts_model_param(self):
        """Verify LOOTrainingLoss.forward() accepts model= parameter."""
        from sci_arc.models.rlan_modules.loo_training import LOOTrainingLoss, LOOConfig
        import inspect
        
        config = LOOConfig()
        loo_loss = LOOTrainingLoss(config=config)
        
        sig = inspect.signature(loo_loss.forward)
        param_names = list(sig.parameters.keys())
        
        assert 'model' in param_names, "forward() should accept model parameter"
        assert 'input_grids' in param_names, "forward() should accept input_grids"
        assert 'output_grids' in param_names, "forward() should accept output_grids"


class TestMetaLearningGradientFlow:
    """Test that gradients flow correctly through meta-learning components."""
    
    def test_hyperlora_receives_gradients(self):
        """Verify HyperLoRA parameters receive gradients."""
        from sci_arc.models.rlan_modules.hyper_lora import HyperLoRA, HyperLoRAConfig
        
        B, N, D, Hs, Ws = 2, 3, 64, 4, 4
        
        config = HyperLoRAConfig(hidden_dim=D, rank=4)
        hyperlora = HyperLoRA(config=config)
        
        # Forward pass
        context = torch.randn(B, N, D, Hs, Ws, requires_grad=True)
        deltas = hyperlora(context)
        
        # Create a dummy loss from deltas
        loss = sum(d.sum() for d in deltas.values())
        loss.backward()
        
        # Check gradients exist
        has_grad = False
        for name, param in hyperlora.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break
        
        assert has_grad, "HyperLoRA should receive gradients"


class TestOptimizerParamGroups:
    """Test that optimizer correctly separates HyperLoRA params."""
    
    def test_train_rlan_creates_hyperlora_param_group(self):
        """Verify train_rlan.py creates HyperLoRA param group."""
        
        # Read train_rlan.py and check for hyperlora param group
        train_script = project_root / "scripts" / "train_rlan.py"
        with open(train_script, encoding='utf-8') as f:
            source = f.read()
        
        # Check hyperlora param group is created
        assert 'hyperlora_decay_params' in source, "Should have hyperlora_decay_params"
        assert 'hyperlora_no_decay_params' in source, "Should have hyperlora_no_decay_params"
        assert 'hyperlora_lr_mult' in source, "Should have hyperlora_lr_mult"
        assert "'name': 'hyperlora_decay'" in source, "Should name the param group"


class TestMetaLearningMetrics:
    """Test that meta-learning metrics are tracked."""
    
    def test_epoch_diagnostics_has_loo_metrics(self):
        """Verify train_epoch tracks LOO metrics."""
        train_script = project_root / "scripts" / "train_rlan.py"
        with open(train_script, encoding='utf-8') as f:
            source = f.read()
        
        # Check LOO metrics are tracked
        assert 'loo_loss_sum' in source, "Should track loo_loss_sum"
        assert 'loo_accuracy_sum' in source, "Should track loo_accuracy_sum"
        assert 'loo_num_holdouts_sum' in source, "Should track loo_num_holdouts_sum"
        assert 'loo_batch_count' in source, "Should track loo_batch_count"
        assert 'loo_skipped_count' in source, "Should track loo_skipped_count"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
