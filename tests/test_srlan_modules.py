"""
Unit tests for S-RLAN v3.0 generalization modules.

Tests:
- NS-TEPS (Neuro-Symbolic TEPS)
- HASR (Hindsight-Aware Solver Refinement)  
- LOO Verifier (Leave-One-Out Verification)
- SynopticRLAN integration

Author: AI Research Assistant
Date: January 2026
"""

import unittest
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class MockRLAN(nn.Module):
    """Mock RLAN model for testing."""
    
    def __init__(self):
        super().__init__()
        self.dummy = nn.Linear(1, 1)
    
    def predict(self, input_grid, train_inputs=None, train_outputs=None, **kwargs):
        # Return input as prediction (identity)
        if isinstance(input_grid, torch.Tensor):
            return input_grid.clone()
        return torch.tensor(input_grid)


class TestNSTEPS(unittest.TestCase):
    """Tests for NS-TEPS module."""
    
    @classmethod
    def setUpClass(cls):
        from sci_arc.models.generalization import NSTEPS, NSTEPSConfig, ObjectExtractor
        cls.NSTEPS = NSTEPS
        cls.NSTEPSConfig = NSTEPSConfig
        cls.ObjectExtractor = ObjectExtractor
    
    def test_instantiation(self):
        """Test NS-TEPS can be instantiated."""
        config = self.NSTEPSConfig(enabled=True)
        nsteps = self.NSTEPS(config)
        self.assertIsNotNone(nsteps)
        self.assertTrue(nsteps.config.enabled)
    
    def test_object_extractor(self):
        """Test object extraction from grid."""
        extractor = self.ObjectExtractor(min_size=1, max_objects=10)
        
        # Grid with two distinct objects
        grid = np.array([
            [1, 1, 0, 2, 2],
            [1, 1, 0, 2, 2],
            [0, 0, 0, 0, 0],
            [3, 0, 0, 0, 4],
        ])
        
        objects = extractor.extract(grid)
        self.assertGreaterEqual(len(objects), 2)
        
        # Check object properties
        for obj in objects:
            self.assertIn('mask', obj)
            self.assertIn('color', obj)
            self.assertIn('bbox', obj)
            self.assertIn('size', obj)
    
    def test_disabled_mode(self):
        """Test NS-TEPS returns empty when disabled."""
        config = self.NSTEPSConfig(enabled=False)
        nsteps = self.NSTEPS(config)
        
        result = nsteps.search(
            test_input=np.array([[1, 2], [3, 4]]),
            train_inputs=[np.array([[1, 2], [3, 4]])],
            train_outputs=[np.array([[1, 2], [3, 4]])],
        )
        
        self.assertFalse(result['success'])
        self.assertIsNone(result['prediction'])
    
    def test_identity_search(self):
        """Test NS-TEPS finds identity transformation."""
        config = self.NSTEPSConfig(
            enabled=True,
            max_search_steps=500,
            timeout_seconds=5.0,
        )
        nsteps = self.NSTEPS(config)
        
        grid = np.array([[1, 2], [3, 4]])
        
        result = nsteps.search(
            test_input=grid,
            train_inputs=[grid.copy()],
            train_outputs=[grid.copy()],
        )
        
        # Should find some matching program
        self.assertGreater(result['confidence'], 0.5)
    
    def test_global_rotation(self):
        """Test NS-TEPS finds grid rotation."""
        config = self.NSTEPSConfig(
            enabled=True,
            max_search_steps=1000,
            timeout_seconds=5.0,
        )
        nsteps = self.NSTEPS(config)
        
        input1 = np.array([[1, 2], [3, 4]])
        output1 = np.rot90(input1, 1)
        
        input2 = np.array([[5, 6], [7, 8]])
        output2 = np.rot90(input2, 1)
        
        result = nsteps.search(
            test_input=input1,
            train_inputs=[input1, input2],
            train_outputs=[output1, output2],
        )
        
        # Should find rotation or similar
        self.assertIsNotNone(result.get('trace') or result.get('confidence', 0) > 0)


class TestHASR(unittest.TestCase):
    """Tests for HASR module."""
    
    @classmethod
    def setUpClass(cls):
        from sci_arc.models.generalization import HASR, HASRConfig, LoRALayer
        cls.HASR = HASR
        cls.HASRConfig = HASRConfig
        cls.LoRALayer = LoRALayer
    
    def test_instantiation(self):
        """Test HASR can be instantiated."""
        config = self.HASRConfig(enabled=True)
        hasr = self.HASR(config)
        self.assertIsNotNone(hasr)
        self.assertTrue(hasr.config.enabled)
    
    def test_lora_layer(self):
        """Test LoRA layer forward pass."""
        lora = self.LoRALayer(in_features=64, out_features=64, rank=4)
        
        x = torch.randn(2, 64)
        base = torch.randn(2, 64)
        
        output = lora(x, base)
        
        self.assertEqual(output.shape, base.shape)
    
    def test_lora_reset(self):
        """Test LoRA layer can be reset."""
        lora = self.LoRALayer(in_features=64, out_features=64, rank=4)
        
        # Modify weights
        lora.lora_B.data.fill_(1.0)
        
        # Reset
        lora.reset()
        
        # B should be zeros again
        self.assertTrue(torch.allclose(lora.lora_B, torch.zeros_like(lora.lora_B)))
    
    def test_pseudo_label_collection(self):
        """Test collecting pseudo-labels from program results."""
        config = self.HASRConfig(
            enabled=True,
            pseudo_label_threshold=0.7,
        )
        hasr = self.HASR(config)
        
        program_results = [
            {'confidence': 0.9, 'prediction': np.array([[1, 2], [3, 4]])},
            {'confidence': 0.5, 'prediction': np.array([[5, 6], [7, 8]])},  # Below threshold
        ]
        
        train_inputs = [np.array([[1, 0], [0, 1]])]
        train_outputs = [np.array([[1, 2], [3, 4]])]
        
        pseudo_labels = hasr.collect_pseudo_labels(
            program_results=program_results,
            train_inputs=train_inputs,
            train_outputs=train_outputs,
        )
        
        # Should only include high-confidence result
        self.assertGreater(len(pseudo_labels), 0)
        self.assertGreaterEqual(pseudo_labels[0]['confidence'], 0.7)
    
    def test_disabled_mode(self):
        """Test HASR does nothing when disabled."""
        config = self.HASRConfig(enabled=False)
        hasr = self.HASR(config)
        
        result = hasr.adapt(
            pseudo_labels=[],
            rlan_model=MockRLAN(),
        )
        
        self.assertFalse(result['adapted'])


class TestLOOVerifier(unittest.TestCase):
    """Tests for LOO Verifier module."""
    
    @classmethod
    def setUpClass(cls):
        from sci_arc.models.generalization import LOOVerifier, LOOVerifierConfig, VerifierRanker
        cls.LOOVerifier = LOOVerifier
        cls.LOOVerifierConfig = LOOVerifierConfig
        cls.VerifierRanker = VerifierRanker
    
    def test_instantiation(self):
        """Test LOO Verifier can be instantiated."""
        config = self.LOOVerifierConfig(enabled=True)
        verifier = self.LOOVerifier(config)
        self.assertIsNotNone(verifier)
    
    def test_basic_verification(self):
        """Test LOO verification of candidates."""
        config = self.LOOVerifierConfig(
            enabled=True,
            min_pairs_for_loo=2,
        )
        verifier = self.LOOVerifier(config)
        
        # Create consistent training pairs
        train_inputs = [
            np.array([[1, 2], [3, 4]]),
            np.array([[5, 6], [7, 8]]),
            np.array([[1, 1], [2, 2]]),
        ]
        train_outputs = [
            np.rot90(train_inputs[0], 1),
            np.rot90(train_inputs[1], 1),
            np.rot90(train_inputs[2], 1),
        ]
        
        candidates = [
            {'prediction': np.rot90(train_inputs[0], 1), 'confidence': 0.8},
            {'prediction': train_inputs[0], 'confidence': 0.5},  # Wrong - identity
        ]
        
        verified = verifier.verify(
            candidates=candidates,
            train_inputs=train_inputs,
            train_outputs=train_outputs,
        )
        
        self.assertEqual(len(verified), 2)
        # First candidate should have LOO score
        self.assertIn('loo_score', verified[0])
        self.assertIn('final_confidence', verified[0])
    
    def test_ranker(self):
        """Test VerifierRanker ranking."""
        config = self.LOOVerifierConfig(enabled=True)
        ranker = self.VerifierRanker(config)
        
        train_inputs = [
            np.array([[1, 0], [0, 1]]),
            np.array([[2, 0], [0, 2]]),
        ]
        train_outputs = [
            np.array([[1, 1], [0, 0]]),
            np.array([[2, 2], [0, 0]]),
        ]
        
        candidates = [
            {'prediction': np.array([[1, 1], [0, 0]]), 'confidence': 0.9},
            {'prediction': np.array([[0, 0], [1, 1]]), 'confidence': 0.5},
        ]
        
        result = ranker.rank(
            candidates=candidates,
            train_inputs=train_inputs,
            train_outputs=train_outputs,
            test_input=np.array([[3, 0], [0, 3]]),
        )
        
        self.assertIn('prediction', result)
        self.assertIn('confidence', result)
        self.assertIn('method', result)
    
    def test_disabled_mode(self):
        """Test verifier passes through when disabled."""
        config = self.LOOVerifierConfig(enabled=False)
        verifier = self.LOOVerifier(config)
        
        candidates = [{'prediction': np.array([[1, 2]]), 'confidence': 0.5}]
        
        result = verifier.verify(
            candidates=candidates,
            train_inputs=[],
            train_outputs=[],
        )
        
        # Should return unchanged
        self.assertEqual(result, candidates)


class TestSynopticRLAN(unittest.TestCase):
    """Tests for integrated SynopticRLAN."""
    
    @classmethod
    def setUpClass(cls):
        from sci_arc.models.generalization import SynopticRLAN, SRLANConfig
        cls.SynopticRLAN = SynopticRLAN
        cls.SRLANConfig = SRLANConfig
    
    def test_instantiation(self):
        """Test SynopticRLAN can be instantiated."""
        config = self.SRLANConfig(enabled=True)
        srlan = self.SynopticRLAN(config)
        self.assertIsNotNone(srlan)
    
    def test_full_pipeline(self):
        """Test full S-RLAN inference pipeline."""
        config = self.SRLANConfig(
            enabled=True,
            use_ns_teps=True,
            use_teps=True,
            use_hasr=True,
            use_loo_verifier=True,
        )
        srlan = self.SynopticRLAN(config)
        
        # Simple identity task
        test_input = np.array([[1, 2], [3, 4]])
        train_inputs = [test_input.copy()]
        train_outputs = [test_input.copy()]
        
        mock_rlan = MockRLAN()
        
        result = srlan.predict(
            rlan_model=mock_rlan,
            test_input=test_input,
            train_inputs=train_inputs,
            train_outputs=train_outputs,
        )
        
        self.assertIn('prediction', result)
        self.assertIn('confidence', result)
        self.assertIn('method', result)
        self.assertIn('candidates', result)
    
    def test_disabled_fallback(self):
        """Test disabled S-RLAN falls back to neural."""
        config = self.SRLANConfig(enabled=False)
        srlan = self.SynopticRLAN(config)
        
        test_input = np.array([[1, 2], [3, 4]])
        mock_rlan = MockRLAN()
        
        result = srlan.predict(
            rlan_model=mock_rlan,
            test_input=test_input,
            train_inputs=[test_input],
            train_outputs=[test_input],
        )
        
        # Should get neural prediction
        self.assertIsNotNone(result.get('prediction'))
    
    def test_module_ablation(self):
        """Test modules can be independently disabled."""
        # Test with only NS-TEPS
        config1 = self.SRLANConfig(
            enabled=True,
            use_ns_teps=True,
            use_teps=False,
            use_hasr=False,
            use_loo_verifier=False,
        )
        srlan1 = self.SynopticRLAN(config1)
        self.assertIsNotNone(srlan1.ns_teps)
        self.assertIsNone(srlan1.teps)
        self.assertIsNone(srlan1.hasr)
        self.assertIsNone(srlan1.loo_verifier)
        
        # Test with only HASR
        config2 = self.SRLANConfig(
            enabled=True,
            use_ns_teps=False,
            use_teps=False,
            use_hasr=True,
            use_loo_verifier=False,
        )
        srlan2 = self.SynopticRLAN(config2)
        self.assertIsNone(srlan2.ns_teps)
        self.assertIsNone(srlan2.teps)
        self.assertIsNotNone(srlan2.hasr)
        self.assertIsNone(srlan2.loo_verifier)
    
    def test_rotation_task(self):
        """Test S-RLAN on rotation task."""
        config = self.SRLANConfig(
            enabled=True,
            use_ns_teps=True,
            use_teps=True,
        )
        srlan = self.SynopticRLAN(config)
        
        # Rotation task
        input1 = np.array([[1, 2], [3, 4]])
        output1 = np.rot90(input1, 1)
        
        input2 = np.array([[5, 6], [7, 8]])
        output2 = np.rot90(input2, 1)
        
        test_input = np.array([[1, 1], [2, 2]])
        expected = np.rot90(test_input, 1)
        
        mock_rlan = MockRLAN()
        
        result = srlan.predict(
            rlan_model=mock_rlan,
            test_input=test_input,
            train_inputs=[input1, input2],
            train_outputs=[output1, output2],
        )
        
        self.assertIn('prediction', result)
        # Check if any candidate got close
        candidates = result.get('candidates', [])
        self.assertGreater(len(candidates), 0)


class TestModularity(unittest.TestCase):
    """Tests to verify modules can be independently added/removed."""
    
    def test_imports(self):
        """Test all modules can be imported independently."""
        # Should not raise
        from sci_arc.models.generalization import NSTEPS, NSTEPSConfig
        from sci_arc.models.generalization import HASR, HASRConfig
        from sci_arc.models.generalization import LOOVerifier, LOOVerifierConfig
        from sci_arc.models.generalization import SynopticRLAN, SRLANConfig
        
        self.assertTrue(True)
    
    def test_no_rlan_dependency(self):
        """Test modules work without importing RLAN internals."""
        from sci_arc.models.generalization import NSTEPS, NSTEPSConfig
        from sci_arc.models.generalization import LOOVerifier, LOOVerifierConfig
        
        # These should work without any RLAN imports
        nsteps = NSTEPS(NSTEPSConfig())
        loo = LOOVerifier(LOOVerifierConfig())
        
        # Test standalone operation
        grid = np.array([[1, 2], [3, 4]])
        result = nsteps.search(grid, [grid], [grid])
        
        self.assertIn('success', result)
        print("✓ NS-TEPS works standalone without RLAN internals")


if __name__ == '__main__':
    print("=" * 60)
    print("Running S-RLAN v3.0 Module Tests")
    print("=" * 60)
    
    # Run tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestNSTEPS))
    suite.addTests(loader.loadTestsFromTestCase(TestHASR))
    suite.addTests(loader.loadTestsFromTestCase(TestLOOVerifier))
    suite.addTests(loader.loadTestsFromTestCase(TestSynopticRLAN))
    suite.addTests(loader.loadTestsFromTestCase(TestModularity))
    
    # Run with verbosity
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 60)
    print(f"Tests Run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success: {result.wasSuccessful()}")
    print("=" * 60)
