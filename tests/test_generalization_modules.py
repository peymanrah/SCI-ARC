"""
Tests for Generalization Modules: TEPS, ConsistencyVerifier, EnhancedInference

These tests validate that the generalization modules:
1. Instantiate correctly
2. Produce correct outputs for simple cases
3. Are truly modular (don't require RLAN internals)
4. Can be enabled/disabled via config

Run with: python -m pytest tests/test_generalization_modules.py -v
Or: python tests/test_generalization_modules.py

Author: AI Research Assistant
Date: January 2026
"""

import sys
import os
import unittest
import numpy as np
import torch
import torch.nn as nn

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# Mock RLAN for testing
# =============================================================================

class MockRLAN(nn.Module):
    """Mock RLAN model for testing enhanced inference."""
    
    def __init__(self):
        super().__init__()
        self.dummy_param = nn.Parameter(torch.zeros(1))
    
    def predict(
        self, 
        input_grid: torch.Tensor,
        train_inputs: torch.Tensor = None,
        train_outputs: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """Return a dummy prediction (copy of input)."""
        if input_grid.dim() == 3:
            return input_grid.squeeze(0)
        return input_grid


# =============================================================================
# Test TEPS
# =============================================================================

class TestTEPS(unittest.TestCase):
    """Tests for Test-Time Exhaustive Program Search."""
    
    @classmethod
    def setUpClass(cls):
        from sci_arc.models.generalization.teps import TEPS, TEPSConfig, PrimitiveLibrary
        cls.TEPS = TEPS
        cls.TEPSConfig = TEPSConfig
        cls.PrimitiveLibrary = PrimitiveLibrary
    
    def test_instantiation(self):
        """Test TEPS can be instantiated."""
        config = self.TEPSConfig(enabled=True)
        teps = self.TEPS(config)
        self.assertIsNotNone(teps)
        self.assertTrue(len(teps.primitives) > 0)
    
    def test_primitive_library(self):
        """Test primitive library contains expected operations."""
        library = self.PrimitiveLibrary()
        primitives = library.get_primitives()
        
        # Check for key primitives
        prim_names = [p.name for p in primitives]
        self.assertIn('identity', prim_names)
        self.assertIn('rotate_90', prim_names)
        self.assertIn('flip_horizontal', prim_names)
        self.assertIn('crop_to_content', prim_names)
        
    def test_identity_search(self):
        """Test TEPS finds identity when input == output."""
        config = self.TEPSConfig(
            enabled=True,
            max_search_steps=100,
            timeout_seconds=2.0,
        )
        teps = self.TEPS(config)
        
        # Create trivial task: input == output
        test_input = np.array([
            [1, 2, 0],
            [0, 1, 0],
            [0, 0, 1],
        ])
        
        train_inputs = [test_input.copy(), test_input.copy()]
        train_outputs = [test_input.copy(), test_input.copy()]
        
        result = teps.search(
            test_input=test_input,
            train_inputs=train_inputs,
            train_outputs=train_outputs,
        )
        
        self.assertTrue(result['success'])
        # The prediction should equal the input (any identity-preserving program works)
        np.testing.assert_array_equal(result['prediction'], test_input)
        # Program should be one that preserves identity (could be 'identity' or 'crop_to_content', etc.)
        self.assertIn(result['program'].primitive.name, ['identity', 'crop_to_content'])
    
    def test_rotate_90_search(self):
        """Test TEPS finds rotate_90 transformation."""
        config = self.TEPSConfig(
            enabled=True,
            max_search_steps=500,
            timeout_seconds=3.0,
        )
        teps = self.TEPS(config)
        
        # Create task where output = rotate_90(input)
        input1 = np.array([
            [1, 2],
            [3, 4],
        ])
        output1 = np.rot90(input1, 1)
        
        input2 = np.array([
            [5, 6],
            [7, 8],
        ])
        output2 = np.rot90(input2, 1)
        
        test_input = np.array([
            [1, 1],
            [2, 2],
        ])
        expected_output = np.rot90(test_input, 1)
        
        result = teps.search(
            test_input=test_input,
            train_inputs=[input1, input2],
            train_outputs=[output1, output2],
        )
        
        self.assertTrue(result['success'])
        np.testing.assert_array_equal(result['prediction'], expected_output)
    
    def test_flip_horizontal_search(self):
        """Test TEPS finds flip_horizontal transformation."""
        config = self.TEPSConfig(
            enabled=True,
            max_search_steps=500,
            timeout_seconds=3.0,
        )
        teps = self.TEPS(config)
        
        # Create task where output = flip_horizontal(input)
        input1 = np.array([
            [1, 2, 3],
            [4, 5, 6],
        ])
        output1 = np.flip(input1, axis=1)
        
        input2 = np.array([
            [7, 8, 9],
            [1, 2, 3],
        ])
        output2 = np.flip(input2, axis=1)
        
        test_input = np.array([
            [1, 0, 0],
            [0, 0, 1],
        ])
        expected_output = np.flip(test_input, axis=1)
        
        result = teps.search(
            test_input=test_input,
            train_inputs=[input1, input2],
            train_outputs=[output1, output2],
        )
        
        self.assertTrue(result['success'])
        np.testing.assert_array_equal(result['prediction'], expected_output)
    
    def test_disabled_mode(self):
        """Test TEPS returns empty result when disabled."""
        config = self.TEPSConfig(enabled=False)
        teps = self.TEPS(config)
        
        result = teps.search(
            test_input=np.zeros((3, 3)),
            train_inputs=[np.zeros((3, 3))],
            train_outputs=[np.zeros((3, 3))],
        )
        
        self.assertFalse(result['success'])
        self.assertTrue(result['stats'].get('disabled', False))
    
    def test_no_match_returns_partial(self):
        """Test TEPS returns partial match when no perfect match found."""
        config = self.TEPSConfig(
            enabled=True,
            max_search_steps=50,  # Low to ensure no match
            timeout_seconds=1.0,
        )
        teps = self.TEPS(config)
        
        # Create complex task that won't be found
        test_input = np.array([[1, 2], [3, 4]])
        train_inputs = [test_input]
        train_outputs = [np.array([[9, 8], [7, 6]])]  # Random transformation
        
        result = teps.search(
            test_input=test_input,
            train_inputs=train_inputs,
            train_outputs=train_outputs,
        )
        
        self.assertFalse(result['success'])
        # Should still return stats
        self.assertIn('steps', result['stats'])


# =============================================================================
# Test ConsistencyVerifier
# =============================================================================

class TestConsistencyVerifier(unittest.TestCase):
    """Tests for ConsistencyVerifier."""
    
    @classmethod
    def setUpClass(cls):
        from sci_arc.models.generalization.consistency_verifier import (
            ConsistencyVerifier, ConsistencyConfig
        )
        cls.ConsistencyVerifier = ConsistencyVerifier
        cls.ConsistencyConfig = ConsistencyConfig
    
    def test_instantiation(self):
        """Test ConsistencyVerifier can be instantiated."""
        config = self.ConsistencyConfig(enabled=True, hidden_dim=64)
        verifier = self.ConsistencyVerifier(config)
        self.assertIsNotNone(verifier)
    
    def test_consistent_prediction(self):
        """Test that consistent predictions get high scores."""
        config = self.ConsistencyConfig(
            enabled=True,
            hidden_dim=64,
            use_structural_features=False,  # Use only diff features for speed
            use_diff_features=True,
        )
        verifier = self.ConsistencyVerifier(config)
        
        # Same transformation: identity
        test_input = torch.tensor([[1, 2], [3, 4]])
        prediction = torch.tensor([[1, 2], [3, 4]])  # Identity
        
        train_inputs = [
            torch.tensor([[5, 6], [7, 8]]),
            torch.tensor([[1, 1], [1, 1]]),
        ]
        train_outputs = [
            torch.tensor([[5, 6], [7, 8]]),  # Identity
            torch.tensor([[1, 1], [1, 1]]),  # Identity
        ]
        
        result = verifier.verify(
            test_input=test_input,
            prediction=prediction,
            train_inputs=train_inputs,
            train_outputs=train_outputs,
        )
        
        # Identity transformation should be consistent
        self.assertGreater(result['score'], 0.5)
    
    def test_disabled_mode(self):
        """Test verifier returns high score when disabled."""
        config = self.ConsistencyConfig(enabled=False)
        verifier = self.ConsistencyVerifier(config)
        
        result = verifier.verify(
            test_input=torch.zeros(3, 3),
            prediction=torch.ones(3, 3),
            train_inputs=[torch.zeros(3, 3)],
            train_outputs=[torch.zeros(3, 3)],
        )
        
        self.assertEqual(result['score'], 1.0)
        self.assertTrue(result['is_consistent'])
    
    def test_diff_features(self):
        """Test diff feature computation."""
        config = self.ConsistencyConfig(
            enabled=True,
            use_structural_features=False,
            use_diff_features=True,
        )
        verifier = self.ConsistencyVerifier(config)
        
        # Same size, some changes
        input_grid = np.array([[0, 1], [2, 0]])
        output_grid = np.array([[0, 3], [4, 0]])
        
        features = verifier._compute_diff_features(input_grid, output_grid)
        
        # Should have: 2 size ratios + 10 color changes + 1 pixel change ratio = 13
        self.assertEqual(len(features), 13)


# =============================================================================
# Test EnhancedInference
# =============================================================================

class TestEnhancedInference(unittest.TestCase):
    """Tests for EnhancedInference pipeline."""
    
    @classmethod
    def setUpClass(cls):
        from sci_arc.models.generalization.enhanced_inference import (
            EnhancedInference, EnhancedInferenceConfig
        )
        from sci_arc.models.generalization.teps import TEPSConfig
        from sci_arc.models.generalization.consistency_verifier import ConsistencyConfig
        
        cls.EnhancedInference = EnhancedInference
        cls.EnhancedInferenceConfig = EnhancedInferenceConfig
        cls.TEPSConfig = TEPSConfig
        cls.ConsistencyConfig = ConsistencyConfig
    
    def test_instantiation(self):
        """Test EnhancedInference can be instantiated."""
        config = self.EnhancedInferenceConfig(enabled=True)
        enhanced = self.EnhancedInference(config)
        self.assertIsNotNone(enhanced)
        self.assertIsNotNone(enhanced.teps)
        self.assertIsNotNone(enhanced.verifier)
    
    def test_teps_priority(self):
        """Test that TEPS match is preferred when found."""
        config = self.EnhancedInferenceConfig(
            enabled=True,
            use_teps=True,
            prefer_teps_over_neural=True,
            teps_config=self.TEPSConfig(
                enabled=True,
                max_search_steps=200,
                timeout_seconds=2.0,
            ),
            use_verification=False,
        )
        enhanced = self.EnhancedInference(config)
        
        mock_rlan = MockRLAN()
        
        # Identity task - TEPS should find it
        test_input = np.array([[1, 2], [3, 4]])
        train_inputs = [np.array([[5, 6], [7, 8]])]
        train_outputs = [np.array([[5, 6], [7, 8]])]  # Identity
        
        result = enhanced.predict(
            rlan_model=mock_rlan,
            test_input=test_input,
            train_inputs=train_inputs,
            train_outputs=train_outputs,
        )
        
        self.assertEqual(result['method'], 'teps')
        np.testing.assert_array_equal(result['prediction'], test_input)
    
    def test_fallback_to_neural(self):
        """Test fallback to neural when TEPS fails."""
        config = self.EnhancedInferenceConfig(
            enabled=True,
            use_teps=True,
            teps_config=self.TEPSConfig(
                enabled=True,
                max_search_steps=10,  # Very low - will fail
                timeout_seconds=0.5,
            ),
            use_verification=False,
        )
        enhanced = self.EnhancedInference(config)
        
        mock_rlan = MockRLAN()
        
        # Complex task TEPS won't find
        test_input = np.array([[1, 2], [3, 4]])
        train_inputs = [np.array([[1, 2], [3, 4]])]
        train_outputs = [np.array([[9, 8], [7, 6]])]  # Random
        
        result = enhanced.predict(
            rlan_model=mock_rlan,
            test_input=test_input,
            train_inputs=train_inputs,
            train_outputs=train_outputs,
        )
        
        self.assertIn(result['method'], ['neural', 'teps_fallback'])
    
    def test_disabled_mode(self):
        """Test disabled mode uses base RLAN."""
        config = self.EnhancedInferenceConfig(enabled=False)
        enhanced = self.EnhancedInference(config)
        
        mock_rlan = MockRLAN()
        
        test_input = np.array([[1, 2], [3, 4]])
        
        result = enhanced.predict(
            rlan_model=mock_rlan,
            test_input=test_input,
            train_inputs=[test_input],
            train_outputs=[test_input],
        )
        
        # Should just use mock RLAN (returns input copy)
        self.assertIsNotNone(result['prediction'])


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration(unittest.TestCase):
    """Integration tests for all generalization modules."""
    
    def test_modules_import(self):
        """Test all modules can be imported."""
        from sci_arc.models.generalization import (
            TEPS, TEPSConfig, PrimitiveLibrary, Program,
            ConsistencyVerifier, ConsistencyConfig,
            EnhancedInference, EnhancedInferenceConfig,
            run_enhanced_inference,
        )
        
        self.assertIsNotNone(TEPS)
        self.assertIsNotNone(ConsistencyVerifier)
        self.assertIsNotNone(EnhancedInference)
        self.assertIsNotNone(run_enhanced_inference)
    
    def test_full_pipeline(self):
        """Test full enhanced inference pipeline."""
        from sci_arc.models.generalization import (
            EnhancedInference, EnhancedInferenceConfig,
            TEPSConfig, ConsistencyConfig,
        )
        
        config = EnhancedInferenceConfig(
            enabled=True,
            use_teps=True,
            use_verification=True,
            teps_config=TEPSConfig(
                enabled=True,
                max_search_steps=300,
                timeout_seconds=2.0,
            ),
            verification_config=ConsistencyConfig(
                enabled=True,
                hidden_dim=64,
                use_structural_features=False,  # Faster
            ),
        )
        
        enhanced = EnhancedInference(config)
        mock_rlan = MockRLAN()
        
        # Test on rotate_90 task
        input1 = np.array([[1, 2], [3, 4]])
        output1 = np.rot90(input1, 1)
        input2 = np.array([[5, 6], [7, 8]])
        output2 = np.rot90(input2, 1)
        
        test_input = np.array([[1, 0], [0, 2]])
        expected = np.rot90(test_input, 1)
        
        result = enhanced.predict(
            rlan_model=mock_rlan,
            test_input=test_input,
            train_inputs=[input1, input2],
            train_outputs=[output1, output2],
        )
        
        self.assertIsNotNone(result['prediction'])
        
        # If TEPS succeeded, prediction should match
        if result['method'] == 'teps':
            np.testing.assert_array_equal(result['prediction'], expected)
    
    def test_modularity(self):
        """Test modules work independently without RLAN internals."""
        from sci_arc.models.generalization.teps import TEPS, TEPSConfig
        
        # TEPS should work completely standalone
        teps = TEPS(TEPSConfig())
        
        # No torch, no RLAN - just numpy
        test_input = np.array([[1, 2], [3, 4]])
        
        result = teps.search(
            test_input=test_input,
            train_inputs=[test_input],
            train_outputs=[test_input],  # Identity
        )
        
        self.assertTrue(result['success'])
        print("✓ TEPS works standalone without any RLAN code")


# =============================================================================
# Main Entry Point
# =============================================================================

def run_all_tests():
    """Run all tests."""
    print("=" * 70)
    print("Running Generalization Module Tests")
    print("=" * 70)
    print()
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestTEPS))
    suite.addTests(loader.loadTestsFromTestCase(TestConsistencyVerifier))
    suite.addTests(loader.loadTestsFromTestCase(TestEnhancedInference))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print()
    print("=" * 70)
    print(f"Tests Run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success: {result.wasSuccessful()}")
    print("=" * 70)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
