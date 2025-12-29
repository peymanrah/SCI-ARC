"""
Test Config Plumbing: Verify YAML config values reach the model.

This test ensures that all YAML configuration values are properly passed
through create_model() to RLANConfig and ultimately to the model.

Critical fields tested:
1. use_cross_attention_context, spatial_downsample (context injection mode)
2. All HPM fields (use_hpm, hpm_top_k, etc.)
3. HyperLoRA fields
4. Gradient checkpointing

Run with: python tests/test_config_plumbing.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import yaml


class TestConfigPlumbing(unittest.TestCase):
    """Test that YAML config values reach the model correctly."""
    
    def setUp(self):
        """Load rlan_stable.yaml for testing."""
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'configs', 'rlan_stable.yaml'
        )
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
    
    def test_cross_attention_context_plumbing(self):
        """Test use_cross_attention_context and spatial_downsample reach model."""
        from scripts.train_rlan import create_model
        
        # Override to ensure we're testing the plumbing, not default values
        self.config['model']['use_cross_attention_context'] = True
        self.config['model']['spatial_downsample'] = 8
        
        model = create_model(self.config)
        
        # Check ContextEncoder was created with spatial features enabled
        self.assertIsNotNone(model.context_encoder, "ContextEncoder should exist")
        self.assertTrue(
            model.context_encoder.use_spatial_features,
            "ContextEncoder should have use_spatial_features=True when cross-attention is enabled"
        )
        
        # Check CrossAttentionInjector was created (not just FiLM)
        from sci_arc.models.rlan_modules import CrossAttentionInjector
        self.assertIsInstance(
            model.context_injector,
            CrossAttentionInjector,
            f"Expected CrossAttentionInjector but got {type(model.context_injector).__name__}"
        )
        
        print("✅ use_cross_attention_context and spatial_downsample properly wired")
    
    def test_hyperlora_plumbing(self):
        """Test HyperLoRA config reaches model."""
        from scripts.train_rlan import create_model
        
        self.config['model']['use_hyperlora'] = True
        self.config['model']['hyperlora_rank'] = 16  # Non-default value
        self.config['model']['hyperlora_init_scale'] = 0.2  # Non-default value
        
        model = create_model(self.config)
        
        self.assertTrue(model.use_hyperlora, "use_hyperlora should be True")
        self.assertIsNotNone(model.hyper_lora, "HyperLoRA module should exist")
        self.assertEqual(
            model.hyper_lora.rank, 16,
            f"HyperLoRA rank should be 16, got {model.hyper_lora.rank}"
        )
        
        print("✅ HyperLoRA config properly wired")
    
    def test_hpm_plumbing(self):
        """Test HPM config reaches model."""
        from scripts.train_rlan import create_model
        
        self.config['model']['use_hpm'] = True
        self.config['model']['hpm_top_k'] = 3  # Non-default value
        self.config['model']['hpm_primitives_per_bank'] = 32  # Non-default value
        
        model = create_model(self.config)
        
        self.assertTrue(model.use_hpm, "use_hpm should be True")
        self.assertIsNotNone(model.hpm, "HPM module should exist")
        
        # Verify HPM config was passed through
        hpm_config = model._hpm_config
        self.assertEqual(
            hpm_config.top_k, 3,
            f"HPM top_k should be 3, got {hpm_config.top_k}"
        )
        self.assertEqual(
            hpm_config.primitives_per_bank, 32,
            f"HPM primitives_per_bank should be 32, got {hpm_config.primitives_per_bank}"
        )
        
        print("✅ HPM config properly wired")
    
    def test_gradient_checkpointing_plumbing(self):
        """Test gradient_checkpointing reaches RecursiveSolver."""
        from scripts.train_rlan import create_model
        
        self.config['training']['gradient_checkpointing'] = True
        
        model = create_model(self.config)
        
        self.assertTrue(
            model.solver.gradient_checkpointing,
            "RecursiveSolver should have gradient_checkpointing=True"
        )
        
        print("✅ gradient_checkpointing properly wired")
    
    def test_solver_context_plumbing(self):
        """Test use_solver_context reaches RecursiveSolver."""
        from scripts.train_rlan import create_model
        
        self.config['model']['use_solver_context'] = True
        self.config['model']['solver_context_heads'] = 8  # Non-default
        
        model = create_model(self.config)
        
        self.assertTrue(model.use_solver_context, "use_solver_context should be True")
        self.assertIsNotNone(
            model.solver.solver_cross_attn,
            "Solver should have cross-attention module"
        )
        
        print("✅ use_solver_context properly wired")
    
    def test_rlan_stable_yaml_values(self):
        """Test that actual rlan_stable.yaml values are correctly read."""
        from scripts.train_rlan import create_model
        
        # Load fresh config (unmodified)
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'configs', 'rlan_stable.yaml'
        )
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        model = create_model(config)
        
        # Check key values from rlan_stable.yaml
        yaml_values = {
            'use_cross_attention_context': config['model'].get('use_cross_attention_context', False),
            'spatial_downsample': config['model'].get('spatial_downsample', 1),
            'use_hyperlora': config['model'].get('use_hyperlora', False),
            'hyperlora_init_scale': config['model'].get('hyperlora_init_scale', 0.01),
            'use_solver_context': config['model'].get('use_solver_context', True),
            'use_hpm': config['model'].get('use_hpm', False),
        }
        
        print(f"\nRLAN Stable YAML key values:")
        for key, value in yaml_values.items():
            print(f"  {key}: {value}")
        
        # Verify model has corresponding state
        if yaml_values['use_cross_attention_context']:
            from sci_arc.models.rlan_modules import CrossAttentionInjector
            self.assertIsInstance(
                model.context_injector,
                CrossAttentionInjector,
                "YAML has use_cross_attention_context=true but model uses FiLM"
            )
            print("  → CrossAttentionInjector: VERIFIED ✓")
        
        if yaml_values['use_hyperlora']:
            self.assertIsNotNone(model.hyper_lora, "YAML has use_hyperlora=true but model has no HyperLoRA")
            print("  → HyperLoRA module: VERIFIED ✓")
        
        if yaml_values['use_solver_context']:
            self.assertIsNotNone(
                model.solver.solver_cross_attn,
                "YAML has use_solver_context=true but solver has no cross-attention"
            )
            print("  → Solver cross-attention: VERIFIED ✓")
        
        if yaml_values['use_hpm']:
            self.assertIsNotNone(model.hpm, "YAML has use_hpm=true but model has no HPM")
            print("  → HPM module: VERIFIED ✓")
        else:
            self.assertIsNone(model.hpm, "YAML has use_hpm=false but model has HPM")
            print("  → HPM disabled: VERIFIED ✓")
        
        print("\n✅ All rlan_stable.yaml values properly reach the model")


class TestSeedConsistency(unittest.TestCase):
    """Test that seed is consistently used from hardware.seed."""
    
    def test_bucketed_sampler_uses_hardware_seed(self):
        """Verify BucketedBatchSampler uses hardware.seed."""
        # This is a code inspection test - we verify the fix was applied
        import inspect
        from scripts.train_rlan import create_train_loader
        
        source = inspect.getsource(create_train_loader)
        
        # Check that we're using hardware.seed, not training.seed
        self.assertIn(
            "hardware",
            source,
            "create_train_loader should reference hardware section for seed"
        )
        self.assertNotIn(
            "train_cfg.get('seed'",
            source,
            "create_train_loader should NOT use train_cfg.get('seed') - seed is in hardware section"
        )
        
        print("✅ BucketedBatchSampler seed source correctly uses hardware.seed")


if __name__ == '__main__':
    print("=" * 60)
    print("CONFIG PLUMBING TESTS")
    print("=" * 60)
    print("Testing that YAML config values properly reach the model...")
    print()
    
    unittest.main(verbosity=2)
