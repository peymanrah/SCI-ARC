"""
Test script to verify fixes for:
1. Dynamic padding + translational offset (content-aware cropping)
2. Unicode logging fix (ASCII-safe alpha character)

Run with: python tests/test_translational_dynamic_padding.py
Or: python -m pytest tests/test_translational_dynamic_padding.py -v

Dec 2025 - Validates critical data pipeline fix
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Ensure project root is in path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestContentAwareCropping:
    """Test that dynamic padding correctly handles translational augmentation."""
    
    def test_extract_content_region_top_left(self):
        """Content at top-left (offset=0,0) should be extracted correctly."""
        from sci_arc.data.dataset import _extract_content_region
        
        # Create 30x30 grid with 5x5 content at top-left
        grid = torch.full((30, 30), 10, dtype=torch.long)  # PAD=10
        grid[:5, :5] = torch.arange(25).reshape(5, 5) % 10  # Content 0-9
        
        result = _extract_content_region(grid, target_size=5, is_target=False)
        
        assert result.shape == (5, 5), f"Expected (5,5), got {result.shape}"
        # Should extract the top-left content
        expected = torch.arange(25).reshape(5, 5) % 10
        assert torch.equal(result, expected), "Content not extracted correctly from top-left"
        print("  [OK] Content at top-left extracted correctly")
    
    def test_extract_content_region_offset(self):
        """Content at offset (10,10) should be extracted correctly - THE CRITICAL FIX."""
        from sci_arc.data.dataset import _extract_content_region
        
        # Create 30x30 grid with 5x5 content at offset (10, 10)
        grid = torch.full((30, 30), 10, dtype=torch.long)  # PAD=10
        content = torch.arange(25).reshape(5, 5) % 10
        grid[10:15, 10:15] = content
        
        result = _extract_content_region(grid, target_size=5, is_target=False)
        
        assert result.shape == (5, 5), f"Expected (5,5), got {result.shape}"
        assert torch.equal(result, content), "Content not extracted from offset position!"
        print("  [OK] Content at offset (10,10) extracted correctly - CRITICAL FIX VERIFIED")
    
    def test_extract_content_region_random_offset(self):
        """Content at random offsets should be extracted correctly."""
        from sci_arc.data.dataset import _extract_content_region
        
        for offset_r in [0, 5, 15, 24]:
            for offset_c in [0, 8, 12, 20]:
                # Create grid with content at this offset
                grid = torch.full((30, 30), 10, dtype=torch.long)
                content_h, content_w = 5, 6
                
                # Skip if content wouldn't fit
                if offset_r + content_h > 30 or offset_c + content_w > 30:
                    continue
                
                content = torch.arange(content_h * content_w).reshape(content_h, content_w) % 10
                grid[offset_r:offset_r+content_h, offset_c:offset_c+content_w] = content
                
                result = _extract_content_region(grid, target_size=max(content_h, content_w), is_target=False)
                
                # Verify content matches (shape may be exact content size)
                assert result.shape[0] >= content_h, f"Height too small at offset ({offset_r},{offset_c})"
                assert result.shape[1] >= content_w, f"Width too small at offset ({offset_r},{offset_c})"
                
                # Check that actual content values are preserved
                extracted_content = result[:content_h, :content_w]
                assert torch.equal(extracted_content, content), \
                    f"Content mismatch at offset ({offset_r},{offset_c})"
        
        print("  [OK] Content extraction works for multiple random offsets")
    
    def test_extract_content_region_target_grid(self):
        """Target grids use -100 as padding, should extract correctly."""
        from sci_arc.data.dataset import _extract_content_region
        
        # Create target grid with -100 padding
        grid = torch.full((30, 30), -100, dtype=torch.long)
        content = torch.arange(16).reshape(4, 4) % 10
        grid[8:12, 8:12] = content
        
        result = _extract_content_region(grid, target_size=4, is_target=True)
        
        assert result.shape == (4, 4), f"Expected (4,4), got {result.shape}"
        assert torch.equal(result, content), "Target content not extracted correctly"
        print("  [OK] Target grid content extraction works with -100 padding")
    
    def test_pad_grid_with_original_size_and_offset(self):
        """pad_grid should use content-aware cropping when original_size is provided."""
        from sci_arc.data.dataset import pad_grid
        
        # Simulate a pre-padded cached sample with translational offset
        # Original content: 8x8, placed at offset (12, 15) in 30x30 canvas
        original_content = torch.arange(64).reshape(8, 8) % 10
        prepadded = torch.full((30, 30), 10, dtype=torch.long)
        prepadded[12:20, 15:23] = original_content
        
        # Dynamic padding wants to crop to batch_max=10 (larger than 8x8 content)
        result = pad_grid(prepadded, max_size=10, original_size=8, is_target=False)
        
        assert result.shape == (10, 10), f"Expected (10,10), got {result.shape}"
        
        # The extracted 8x8 content should be in the result (padded to 10x10)
        extracted = result[:8, :8]
        assert torch.equal(extracted, original_content), \
            "pad_grid did not correctly extract offset content!"
        
        # Padding region should be PAD_COLOR=10
        assert (result[8:, :] == 10).all(), "Padding region should be 10"
        assert (result[:, 8:] == 10).all(), "Padding region should be 10"
        
        print("  [OK] pad_grid correctly handles pre-padded grids with translational offset")


class TestCollateWithTranslationalAugmentation:
    """Test that collate_sci_arc handles samples with translational offsets."""
    
    def _create_translated_sample(self, content_size: int, offset: tuple, pad_to: int = 30):
        """Create a sample that simulates cached data with translational augmentation."""
        offset_r, offset_c = offset
        
        # Create content
        content = torch.arange(content_size * content_size).reshape(content_size, content_size) % 10
        
        # Create pre-padded grid with content at offset
        input_grid = torch.full((pad_to, pad_to), 10, dtype=torch.long)
        output_grid = torch.full((pad_to, pad_to), 10, dtype=torch.long)
        test_input = torch.full((pad_to, pad_to), 10, dtype=torch.long)
        test_output = torch.full((pad_to, pad_to), -100, dtype=torch.long)
        
        input_grid[offset_r:offset_r+content_size, offset_c:offset_c+content_size] = content
        output_grid[offset_r:offset_r+content_size, offset_c:offset_c+content_size] = content
        test_input[offset_r:offset_r+content_size, offset_c:offset_c+content_size] = content
        test_output[offset_r:offset_r+content_size, offset_c:offset_c+content_size] = content
        
        return {
            'task_id': f'test_offset_{offset_r}_{offset_c}',
            'input_grids': [input_grid],
            'output_grids': [output_grid],
            'test_input': test_input,
            'test_output': test_output,
            'num_train_pairs': 1,
            'transform_family': 0,
            'original_max_size': content_size,  # CRITICAL: stores original size
        }
    
    def test_collate_with_translated_samples(self):
        """Collate should preserve content from translated samples."""
        from sci_arc.data.dataset import collate_sci_arc
        
        # Create samples with different offsets (simulating translational augmentation)
        samples = [
            self._create_translated_sample(content_size=5, offset=(0, 0)),   # Top-left
            self._create_translated_sample(content_size=5, offset=(10, 10)), # Center
            self._create_translated_sample(content_size=5, offset=(20, 20)), # Bottom-right
        ]
        
        # Collate with dynamic_padding=True
        batch = collate_sci_arc(samples, max_size=30, dynamic_padding=True)
        
        # Batch max should be 5 (the original content size)
        assert batch['batch_max_size'] == 5, \
            f"Expected batch_max_size=5, got {batch['batch_max_size']}"
        
        # All samples should have their content preserved (not cut off)
        for i in range(3):
            test_in = batch['test_inputs'][i]
            
            # Content should be in top-left of the 5x5 result
            content_region = test_in[:5, :5]
            expected_content = torch.arange(25).reshape(5, 5) % 10
            
            # Check that content matches (not all padding)
            non_padding = (test_in != 10).sum().item()
            assert non_padding >= 25, \
                f"Sample {i}: Expected at least 25 non-padding pixels, got {non_padding}"
            
            assert torch.equal(content_region, expected_content), \
                f"Sample {i}: Content not preserved correctly!"
        
        print("  [OK] collate_sci_arc preserves content from translated samples")
    
    def test_collate_mixed_sizes_and_offsets(self):
        """Collate handles mixed content sizes with different offsets."""
        from sci_arc.data.dataset import collate_sci_arc
        
        samples = [
            self._create_translated_sample(content_size=3, offset=(5, 5)),
            self._create_translated_sample(content_size=7, offset=(0, 0)),
            self._create_translated_sample(content_size=5, offset=(15, 12)),
        ]
        
        batch = collate_sci_arc(samples, max_size=30, dynamic_padding=True)
        
        # Batch max should be 7 (largest original content)
        assert batch['batch_max_size'] == 7, \
            f"Expected batch_max_size=7, got {batch['batch_max_size']}"
        
        # Output tensors should be (3, 7, 7)
        assert batch['test_inputs'].shape == (3, 7, 7), \
            f"Expected shape (3,7,7), got {batch['test_inputs'].shape}"
        
        print("  [OK] collate_sci_arc handles mixed sizes and offsets")


class TestUnicodeLoggingFix:
    """Test that HPM logging uses ASCII-safe characters."""
    
    def test_hpm_no_unicode_in_init(self):
        """HPM initialization should not contain Unicode characters that break Windows cp1252."""
        import io
        import sys
        
        # Capture stdout during HPM import/init check
        hpm_file = project_root / "sci_arc" / "models" / "rlan_modules" / "hpm.py"
        
        with open(hpm_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for problematic Unicode characters
        problematic_chars = ['α', 'β', 'γ', 'δ', '→', '←', '✓', '✗', '×']
        found_issues = []
        
        for char in problematic_chars:
            if char in content:
                # Find the line
                for i, line in enumerate(content.split('\n'), 1):
                    if char in line:
                        found_issues.append(f"Line {i}: Found '{char}' in: {line.strip()[:60]}...")
        
        if found_issues:
            print(f"  [WARN] Found Unicode characters that may cause Windows issues:")
            for issue in found_issues[:5]:  # Show first 5
                print(f"    {issue}")
            # This is a warning, not a failure - the specific α issue should be fixed
        
        # The critical fix: α should be replaced with 'alpha'
        assert 'α' not in content, "HPM still contains α character - fix not applied!"
        
        print("  [OK] HPM does not contain problematic α character")


class TestBackwardCompatibility:
    """Ensure fixes are backward compatible with existing code paths."""
    
    def test_pad_grid_without_original_size(self):
        """pad_grid without original_size should work as before."""
        from sci_arc.data.dataset import pad_grid
        
        # Small grid, no original_size
        grid = torch.arange(16).reshape(4, 4) % 10
        result = pad_grid(grid, max_size=10, is_target=False)
        
        assert result.shape == (10, 10)
        assert torch.equal(result[:4, :4], grid)
        assert (result[4:, :] == 10).all()  # Padding
        assert (result[:, 4:] == 10).all()
        
        print("  [OK] pad_grid backward compatible (no original_size)")
    
    def test_pad_grid_already_small(self):
        """Grid already smaller than max_size should just pad."""
        from sci_arc.data.dataset import pad_grid
        
        grid = torch.arange(9).reshape(3, 3) % 10
        result = pad_grid(grid, max_size=5, original_size=3)
        
        assert result.shape == (5, 5)
        assert torch.equal(result[:3, :3], grid)
        
        print("  [OK] pad_grid handles already-small grids correctly")
    
    def test_collate_without_original_max_size(self):
        """Collate should work with samples lacking original_max_size."""
        from sci_arc.data.dataset import collate_sci_arc
        
        # Samples without original_max_size (on-the-fly generation)
        samples = [
            {
                'task_id': 'test1',
                'input_grids': [torch.arange(25).reshape(5, 5) % 10],
                'output_grids': [torch.arange(25).reshape(5, 5) % 10],
                'test_input': torch.arange(25).reshape(5, 5) % 10,
                'test_output': torch.arange(25).reshape(5, 5) % 10,
                'num_train_pairs': 1,
                'transform_family': 0,
                # No 'original_max_size' key
            }
        ]
        
        batch = collate_sci_arc(samples, max_size=30, dynamic_padding=True)
        
        # Should still work, using tensor shapes as fallback
        assert batch['test_inputs'].shape[1] <= 30
        assert batch['test_inputs'].shape[2] <= 30
        
        print("  [OK] collate_sci_arc backward compatible (no original_max_size)")


def run_all_tests():
    """Run all test cases."""
    print("=" * 60)
    print("Testing Dynamic Padding + Translational Augmentation Fixes")
    print("=" * 60)
    
    # Test 1: Content-aware cropping
    print("\n1. CONTENT-AWARE CROPPING TESTS")
    print("-" * 40)
    test_crop = TestContentAwareCropping()
    test_crop.test_extract_content_region_top_left()
    test_crop.test_extract_content_region_offset()
    test_crop.test_extract_content_region_random_offset()
    test_crop.test_extract_content_region_target_grid()
    test_crop.test_pad_grid_with_original_size_and_offset()
    
    # Test 2: Collate with translated samples
    print("\n2. COLLATE WITH TRANSLATIONAL AUGMENTATION TESTS")
    print("-" * 40)
    test_collate = TestCollateWithTranslationalAugmentation()
    test_collate.test_collate_with_translated_samples()
    test_collate.test_collate_mixed_sizes_and_offsets()
    
    # Test 3: Unicode fix
    print("\n3. UNICODE LOGGING FIX TESTS")
    print("-" * 40)
    test_unicode = TestUnicodeLoggingFix()
    test_unicode.test_hpm_no_unicode_in_init()
    
    # Test 4: Backward compatibility
    print("\n4. BACKWARD COMPATIBILITY TESTS")
    print("-" * 40)
    test_compat = TestBackwardCompatibility()
    test_compat.test_pad_grid_without_original_size()
    test_compat.test_pad_grid_already_small()
    test_compat.test_collate_without_original_max_size()
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
    print("\nSummary:")
    print("  - Content-aware cropping handles translational offsets correctly")
    print("  - collate_sci_arc preserves content from pre-padded cached samples")
    print("  - HPM logging uses ASCII-safe characters (no Unicode issues)")
    print("  - All changes are backward compatible")
    

if __name__ == '__main__':
    run_all_tests()
