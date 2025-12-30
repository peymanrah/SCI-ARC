"""
Test script for dynamic padding with translational augmentation.

CRITICAL FIX VALIDATION (Dec 2025):
- Validates that content-aware cropping works correctly for pre-padded grids
- Ensures translational augmentation + dynamic padding don't corrupt data
- Tests both input grids (PAD=10) and target grids (PAD=-100)

Run with: python -m pytest tests/test_dynamic_padding_translational.py -v -s
Or standalone: python tests/test_dynamic_padding_translational.py
"""

import sys
import os
import unittest
import torch
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sci_arc.data.dataset import pad_grid, _extract_content_region, collate_sci_arc


class TestContentAwareCropping(unittest.TestCase):
    """Test the _extract_content_region helper function."""
    
    def test_content_at_origin(self):
        """Content at top-left (no offset) should work."""
        # 5x5 content at origin, padded to 30x30
        content = torch.arange(25).reshape(5, 5)
        padded = torch.full((30, 30), 10, dtype=torch.long)  # PAD=10
        padded[:5, :5] = content
        
        extracted = _extract_content_region(padded, target_size=5, is_target=False)
        
        self.assertEqual(extracted.shape, (5, 5))
        self.assertTrue(torch.equal(extracted, content))
        print("  [OK] Content at origin extracted correctly")
    
    def test_content_with_offset(self):
        """Content at nonzero offset should be extracted correctly."""
        # 5x5 content at offset (10, 15), padded to 30x30
        content = torch.arange(25).reshape(5, 5)
        padded = torch.full((30, 30), 10, dtype=torch.long)
        padded[10:15, 15:20] = content  # offset (10, 15)
        
        extracted = _extract_content_region(padded, target_size=5, is_target=False)
        
        self.assertEqual(extracted.shape, (5, 5))
        self.assertTrue(torch.equal(extracted, content))
        print("  [OK] Content with offset (10, 15) extracted correctly")
    
    def test_content_at_bottom_right(self):
        """Content at bottom-right corner should work."""
        # 5x5 content at offset (25, 25), padded to 30x30
        content = torch.arange(25).reshape(5, 5) + 1  # Values 1-25 (not 0 which could be black)
        padded = torch.full((30, 30), 10, dtype=torch.long)
        padded[25:30, 25:30] = content
        
        extracted = _extract_content_region(padded, target_size=5, is_target=False)
        
        self.assertEqual(extracted.shape, (5, 5))
        self.assertTrue(torch.equal(extracted, content))
        print("  [OK] Content at bottom-right extracted correctly")
    
    def test_target_grid_with_ignore_value(self):
        """Target grids use -100 as padding, not 10."""
        # 4x4 content at offset (5, 5), target grid
        content = torch.arange(16).reshape(4, 4)
        padded = torch.full((30, 30), -100, dtype=torch.long)  # Target uses -100
        padded[5:9, 5:9] = content
        
        extracted = _extract_content_region(padded, target_size=4, is_target=True)
        
        self.assertEqual(extracted.shape, (4, 4))
        self.assertTrue(torch.equal(extracted, content))
        print("  [OK] Target grid with -100 padding extracted correctly")
    
    def test_content_with_black_pixels(self):
        """Content containing black pixels (0) should not confuse extraction."""
        # Content with some zeros (black pixels)
        content = torch.tensor([
            [0, 1, 2],
            [3, 0, 5],
            [6, 7, 0]
        ], dtype=torch.long)
        padded = torch.full((30, 30), 10, dtype=torch.long)
        padded[8:11, 12:15] = content  # offset (8, 12)
        
        extracted = _extract_content_region(padded, target_size=3, is_target=False)
        
        self.assertEqual(extracted.shape, (3, 3))
        self.assertTrue(torch.equal(extracted, content))
        print("  [OK] Content with black pixels (0) preserved correctly")
    
    def test_non_square_content(self):
        """Non-square content should extract correct bounding box."""
        # 3x6 content
        content = torch.arange(18).reshape(3, 6) + 1
        padded = torch.full((30, 30), 10, dtype=torch.long)
        padded[4:7, 10:16] = content
        
        extracted = _extract_content_region(padded, target_size=6, is_target=False)
        
        self.assertEqual(extracted.shape, (3, 6))
        self.assertTrue(torch.equal(extracted, content))
        print("  [OK] Non-square content (3x6) extracted correctly")


class TestPadGridWithOriginalSize(unittest.TestCase):
    """Test pad_grid with original_size parameter (pre-padded cache fix)."""
    
    def test_prepadded_no_offset(self):
        """Pre-padded grid with content at origin."""
        # 5x5 content at origin, padded to 30x30, dynamic pad to 10x10
        content = torch.arange(25).reshape(5, 5)
        prepadded = torch.full((30, 30), 10, dtype=torch.long)
        prepadded[:5, :5] = content
        
        result = pad_grid(prepadded, max_size=10, original_size=5, is_target=False)
        
        self.assertEqual(result.shape, (10, 10))
        # Content should be in top-left 5x5, rest should be PAD=10
        self.assertTrue(torch.equal(result[:5, :5], content))
        self.assertTrue((result[5:, :] == 10).all())
        self.assertTrue((result[:, 5:] == 10).all())
        print("  [OK] Pre-padded grid (no offset) handled correctly")
    
    def test_prepadded_with_offset(self):
        """Pre-padded grid with translational offset - CRITICAL TEST."""
        # 5x5 content at offset (10, 15), padded to 30x30, dynamic pad to 8x8
        content = torch.arange(25).reshape(5, 5) + 1  # Values 1-25
        prepadded = torch.full((30, 30), 10, dtype=torch.long)
        prepadded[10:15, 15:20] = content  # OFFSET!
        
        result = pad_grid(prepadded, max_size=8, original_size=5, is_target=False)
        
        self.assertEqual(result.shape, (8, 8))
        # Content should now be in top-left 5x5 after extraction + re-padding
        self.assertTrue(torch.equal(result[:5, :5], content))
        # Rest should be PAD=10
        self.assertTrue((result[5:, :] == 10).all())
        self.assertTrue((result[:5, 5:] == 10).all())
        print("  [OK] Pre-padded grid WITH OFFSET handled correctly (critical fix validated)")
    
    def test_prepadded_target_with_offset(self):
        """Pre-padded target grid (uses -100) with offset."""
        content = torch.arange(9).reshape(3, 3)
        prepadded = torch.full((30, 30), -100, dtype=torch.long)  # Target padding
        prepadded[20:23, 5:8] = content  # Offset (20, 5)
        
        result = pad_grid(prepadded, max_size=6, original_size=3, is_target=True)
        
        self.assertEqual(result.shape, (6, 6))
        self.assertTrue(torch.equal(result[:3, :3], content))
        self.assertTrue((result[3:, :] == -100).all())  # Target uses -100
        self.assertTrue((result[:3, 3:] == -100).all())
        print("  [OK] Pre-padded TARGET grid with offset handled correctly")
    
    def test_no_original_size_passthrough(self):
        """Without original_size, grid should pass through normally."""
        grid = torch.arange(16).reshape(4, 4)
        result = pad_grid(grid, max_size=8, original_size=None, is_target=False)
        
        self.assertEqual(result.shape, (8, 8))
        self.assertTrue(torch.equal(result[:4, :4], grid))
        print("  [OK] No original_size passthrough works")


class TestCollateWithTranslation(unittest.TestCase):
    """Test collate_sci_arc with pre-padded + translated samples."""
    
    def _create_sample_with_offset(self, grid_size: int, offset: tuple, pad_to: int = 30):
        """Create a sample that simulates cached + translated augmentation."""
        r_off, c_off = offset
        
        # Create content grids
        input_content = torch.arange(grid_size * grid_size).reshape(grid_size, grid_size)
        output_content = torch.arange(grid_size * grid_size).reshape(grid_size, grid_size) + 100
        test_in_content = torch.arange(grid_size * grid_size).reshape(grid_size, grid_size) + 200
        test_out_content = torch.arange(grid_size * grid_size).reshape(grid_size, grid_size) + 300
        
        # Pre-pad with translational offset (simulating cached sample)
        def prepad(content, is_target=False):
            pad_val = -100 if is_target else 10
            padded = torch.full((pad_to, pad_to), pad_val, dtype=torch.long)
            h, w = content.shape
            padded[r_off:r_off+h, c_off:c_off+w] = content
            return padded
        
        return {
            'task_id': f'test_offset_{r_off}_{c_off}',
            'input_grids': [prepad(input_content, False)],
            'output_grids': [prepad(output_content, False)],
            'test_input': prepad(test_in_content, False),
            'test_output': prepad(test_out_content, True),  # Target!
            'num_train_pairs': 1,
            'transform_family': 0,
            'original_max_size': grid_size,  # CRITICAL: stores original size
            'aug_info': {
                'dihedral_id': 0,
                'color_perm': None,
                'translational_offset': offset,
            }
        }
    
    def test_collate_with_various_offsets(self):
        """Collate samples with different translational offsets."""
        samples = [
            self._create_sample_with_offset(5, (0, 0)),    # No offset
            self._create_sample_with_offset(5, (10, 10)),  # Middle offset
            self._create_sample_with_offset(5, (20, 20)),  # Large offset
        ]
        
        batch = collate_sci_arc(samples, max_size=30, dynamic_padding=True)
        
        # Should be padded to 5 (max original_max_size in batch), not 30!
        expected_size = 5
        self.assertEqual(batch['test_inputs'].shape, (3, expected_size, expected_size))
        self.assertEqual(batch['input_grids'].shape, (3, 1, expected_size, expected_size))
        
        # Verify content is preserved (not corrupted by cropping)
        for i in range(3):
            test_in = batch['test_inputs'][i]
            # Content should be in top-left corner after extraction + re-padding
            # Original content was values 200-224 for test_input
            content_region = test_in[:5, :5]
            expected_content = torch.arange(25).reshape(5, 5) + 200
            self.assertTrue(torch.equal(content_region, expected_content),
                           f"Sample {i}: Content corrupted! Got {content_region}, expected {expected_content}")
        
        print("  [OK] Collate with various translational offsets works correctly")
    
    def test_collate_mixed_sizes_with_offset(self):
        """Collate samples of different sizes, all with offsets."""
        samples = [
            self._create_sample_with_offset(3, (5, 5)),   # 3x3 with offset
            self._create_sample_with_offset(5, (15, 10)), # 5x5 with offset
            self._create_sample_with_offset(8, (2, 20)),  # 8x8 with offset
        ]
        
        batch = collate_sci_arc(samples, max_size=30, dynamic_padding=True)
        
        # Should pad to max(3, 5, 8) = 8
        expected_size = 8
        self.assertEqual(batch['test_inputs'].shape, (3, expected_size, expected_size))
        self.assertEqual(batch['batch_max_size'], expected_size)
        
        print("  [OK] Collate mixed sizes with offsets works correctly")


class TestUnicodeLogging(unittest.TestCase):
    """Test that HPM can be imported without Unicode errors on Windows."""
    
    def test_hpm_import_no_unicode_error(self):
        """HPM module should import without triggering Unicode encoding errors."""
        try:
            # This import triggers HPM class definition which has print statements
            # On Windows with cp1252, Unicode characters would cause errors
            from sci_arc.models.rlan_modules.hpm import HierarchicalPrimitiveMemory
            print("  [OK] HPM module imported without Unicode errors")
        except UnicodeEncodeError as e:
            self.fail(f"HPM import caused UnicodeEncodeError: {e}")


def run_tests():
    """Run all tests and provide summary."""
    print("=" * 70)
    print("DYNAMIC PADDING + TRANSLATIONAL AUGMENTATION FIX VALIDATION")
    print("=" * 70)
    print()
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestContentAwareCropping))
    suite.addTests(loader.loadTestsFromTestCase(TestPadGridWithOriginalSize))
    suite.addTests(loader.loadTestsFromTestCase(TestCollateWithTranslation))
    suite.addTests(loader.loadTestsFromTestCase(TestUnicodeLogging))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print()
    print("=" * 70)
    if result.wasSuccessful():
        print("ALL TESTS PASSED - Fixes validated successfully!")
        print()
        print("Summary of validated fixes:")
        print("  1. Content-aware cropping for pre-padded grids with translational offset")
        print("  2. Unicode character (alpha) replaced with ASCII for Windows compatibility")
        print("  3. Dynamic padding + bucketed batching work with translated samples")
    else:
        print("SOME TESTS FAILED - Please review the output above")
    print("=" * 70)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
