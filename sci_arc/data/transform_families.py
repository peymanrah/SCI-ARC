"""
Transformation Family Labels for Structural Contrastive Loss.

KEY FOR SCL: Tasks with same transformation should cluster together.

This file provides:
1. TRANSFORM_FAMILIES: Mapping of transformation names to integer labels
2. get_transform_family(): Function to get family label for a task

For SCL to work, we need to know which tasks share the same transformation type.
During training, tasks with the same transformation family are positive pairs,
and tasks with different families are negative pairs.

Sources of transformation labels:
1. RE-ARC: Transformation type is encoded in the task generation
2. BARC: LLM-generated tasks may have metadata
3. Original ARC: Manual annotation or rule-based inference
"""

from typing import Dict, Optional
import hashlib


# Transformation family definitions
# Grouped by category for clarity

TRANSFORM_FAMILIES = {
    # === GEOMETRIC TRANSFORMATIONS ===
    'rotate_90': 0,
    'rotate_180': 1,
    'rotate_270': 2,
    'flip_horizontal': 3,
    'flip_vertical': 4,
    'transpose': 5,
    'reflect_diagonal': 6,
    
    # === SCALING TRANSFORMATIONS ===
    'upscale_2x': 7,
    'upscale_3x': 8,
    'downscale_2x': 9,
    'downscale_3x': 10,
    'tile_2x2': 11,
    'tile_3x3': 12,
    
    # === COLOR TRANSFORMATIONS ===
    'color_swap': 13,
    'color_invert': 14,
    'recolor_by_rule': 15,
    'color_gradient': 16,
    'color_replace': 17,
    
    # === OBJECT OPERATIONS ===
    'copy_object': 18,
    'move_object': 19,
    'delete_object': 20,
    'duplicate_object': 21,
    'mirror_object': 22,
    'resize_object': 23,
    
    # === PATTERN OPERATIONS ===
    'extend_pattern': 24,
    'complete_grid': 25,
    'fill_enclosed': 26,
    'repeat_pattern': 27,
    'symmetrize': 28,
    
    # === LOGICAL OPERATIONS ===
    'boolean_and': 29,
    'boolean_or': 30,
    'boolean_xor': 31,
    'mask_apply': 32,
    
    # === STRUCTURAL OPERATIONS ===
    'extract_subgrid': 33,
    'crop_to_object': 34,
    'insert_into_grid': 35,
    'sort_objects': 36,
    'align_objects': 37,
    
    # === COMPOSITE OPERATIONS ===
    'rotate_and_color': 38,
    'scale_and_flip': 39,
    'move_and_copy': 40,
    
    # === SPECIAL ===
    'identity': 41,  # No transformation
    'unknown': 42,   # Unknown/unclassified
}

# Reverse mapping for lookup
FAMILY_TO_NAME = {v: k for k, v in TRANSFORM_FAMILIES.items()}

# Number of transformation families
NUM_TRANSFORM_FAMILIES = len(TRANSFORM_FAMILIES)


# Keyword patterns for automatic detection
TRANSFORM_KEYWORDS = {
    'rotate': ['rotate', 'rot90', 'rot180', 'rot270', 'rotation'],
    'flip': ['flip', 'mirror', 'reflect', 'horizontal', 'vertical'],
    'scale': ['scale', 'resize', 'upscale', 'downscale', 'zoom'],
    'tile': ['tile', 'repeat', 'tessellate'],
    'color': ['color', 'recolor', 'paint', 'fill'],
    'copy': ['copy', 'duplicate', 'clone'],
    'move': ['move', 'translate', 'shift'],
    'delete': ['delete', 'remove', 'erase'],
    'pattern': ['pattern', 'extend', 'complete', 'continue'],
    'boolean': ['and', 'or', 'xor', 'mask'],
    'crop': ['crop', 'extract', 'subgrid'],
}


def get_transform_family(
    task_id: str, 
    task_metadata: Optional[Dict] = None,
    use_hash_fallback: bool = True
) -> int:
    """
    Get transformation family for a task.
    
    Priority:
    1. Explicit metadata (from RE-ARC, BARC)
    2. Keyword matching in task ID
    3. Hash-based fallback (consistent but not semantically meaningful)
    
    Args:
        task_id: Task identifier string
        task_metadata: Optional dict with 'transform_type' key
        use_hash_fallback: If True, use hash for unknown tasks
    
    Returns:
        Integer transformation family label
    """
    # 1. Check explicit metadata
    if task_metadata and 'transform_type' in task_metadata:
        transform_type = task_metadata['transform_type'].lower().replace(' ', '_')
        if transform_type in TRANSFORM_FAMILIES:
            return TRANSFORM_FAMILIES[transform_type]
    
    # 2. Try to infer from task_id using keywords
    task_lower = task_id.lower()
    
    # Check for specific transformation patterns
    for family_name, family_idx in TRANSFORM_FAMILIES.items():
        # Direct match
        if family_name.replace('_', '') in task_lower.replace('_', '').replace('-', ''):
            return family_idx
    
    # Check keyword patterns
    for category, keywords in TRANSFORM_KEYWORDS.items():
        for keyword in keywords:
            if keyword in task_lower:
                # Map category to a representative transformation
                category_map = {
                    'rotate': TRANSFORM_FAMILIES['rotate_90'],
                    'flip': TRANSFORM_FAMILIES['flip_horizontal'],
                    'scale': TRANSFORM_FAMILIES['upscale_2x'],
                    'tile': TRANSFORM_FAMILIES['tile_2x2'],
                    'color': TRANSFORM_FAMILIES['color_swap'],
                    'copy': TRANSFORM_FAMILIES['copy_object'],
                    'move': TRANSFORM_FAMILIES['move_object'],
                    'delete': TRANSFORM_FAMILIES['delete_object'],
                    'pattern': TRANSFORM_FAMILIES['extend_pattern'],
                    'boolean': TRANSFORM_FAMILIES['boolean_and'],
                    'crop': TRANSFORM_FAMILIES['extract_subgrid'],
                }
                if category in category_map:
                    return category_map[category]
    
    # 3. Hash-based fallback
    if use_hash_fallback:
        # Use hash for consistent but arbitrary assignment
        # This at least ensures the same task always gets the same label
        hash_value = int(hashlib.md5(task_id.encode()).hexdigest(), 16)
        return hash_value % NUM_TRANSFORM_FAMILIES
    
    # Default to unknown
    return TRANSFORM_FAMILIES['unknown']


def infer_transform_from_grids(
    input_grid,
    output_grid
) -> int:
    """
    Attempt to infer transformation type from input/output grids.
    
    This is a heuristic approach that looks for common patterns.
    
    Args:
        input_grid: np.ndarray of input
        output_grid: np.ndarray of output
    
    Returns:
        Inferred transformation family
    """
    import numpy as np
    
    in_h, in_w = input_grid.shape
    out_h, out_w = output_grid.shape
    
    # Check size relationships
    if out_h == in_w and out_w == in_h:
        # Possible rotation or transpose
        if np.array_equal(output_grid, np.rot90(input_grid)):
            return TRANSFORM_FAMILIES['rotate_90']
        elif np.array_equal(output_grid, np.rot90(input_grid, 2)):
            return TRANSFORM_FAMILIES['rotate_180']
        elif np.array_equal(output_grid, np.rot90(input_grid, 3)):
            return TRANSFORM_FAMILIES['rotate_270']
        elif np.array_equal(output_grid, input_grid.T):
            return TRANSFORM_FAMILIES['transpose']
    
    if out_h == in_h and out_w == in_w:
        # Same size - check for flips, color changes, etc.
        if np.array_equal(output_grid, np.fliplr(input_grid)):
            return TRANSFORM_FAMILIES['flip_horizontal']
        elif np.array_equal(output_grid, np.flipud(input_grid)):
            return TRANSFORM_FAMILIES['flip_vertical']
        elif np.array_equal(output_grid, input_grid):
            return TRANSFORM_FAMILIES['identity']
    
    # Check for scaling
    if out_h == in_h * 2 and out_w == in_w * 2:
        return TRANSFORM_FAMILIES['upscale_2x']
    elif out_h == in_h // 2 and out_w == in_w // 2:
        return TRANSFORM_FAMILIES['downscale_2x']
    
    # Default to unknown
    return TRANSFORM_FAMILIES['unknown']


# RE-ARC specific transformation mapping
# RE-ARC generates synthetic tasks with known transformations
REARC_TRANSFORM_MAP = {
    'rotate': TRANSFORM_FAMILIES['rotate_90'],
    'flip': TRANSFORM_FAMILIES['flip_horizontal'],
    'transpose': TRANSFORM_FAMILIES['transpose'],
    'scale': TRANSFORM_FAMILIES['upscale_2x'],
    'color_swap': TRANSFORM_FAMILIES['color_swap'],
    'move': TRANSFORM_FAMILIES['move_object'],
    'copy': TRANSFORM_FAMILIES['copy_object'],
    'fill': TRANSFORM_FAMILIES['fill_enclosed'],
    'extend': TRANSFORM_FAMILIES['extend_pattern'],
    'mask': TRANSFORM_FAMILIES['mask_apply'],
}


def get_rearc_transform_family(task_id: str, metadata: Dict) -> int:
    """
    Get transformation family for RE-ARC generated task.
    
    RE-ARC tasks have explicit transformation type in metadata.
    """
    if 'transform' in metadata:
        transform = metadata['transform'].lower()
        for key, family in REARC_TRANSFORM_MAP.items():
            if key in transform:
                return family
    
    return get_transform_family(task_id, metadata)


# ConceptARC categories
CONCEPT_ARC_MAP = {
    # Concept categories from ConceptARC
    'above_below': TRANSFORM_FAMILIES['move_object'],
    'center': TRANSFORM_FAMILIES['align_objects'],
    'clean_up': TRANSFORM_FAMILIES['delete_object'],
    'complete_shape': TRANSFORM_FAMILIES['complete_grid'],
    'copy': TRANSFORM_FAMILIES['copy_object'],
    'count': TRANSFORM_FAMILIES['extract_subgrid'],
    'extend_to_boundary': TRANSFORM_FAMILIES['extend_pattern'],
    'extract_objects': TRANSFORM_FAMILIES['crop_to_object'],
    'fill': TRANSFORM_FAMILIES['fill_enclosed'],
    'horizontal_vertical': TRANSFORM_FAMILIES['flip_horizontal'],
    'inside_outside': TRANSFORM_FAMILIES['mask_apply'],
    'move_towards': TRANSFORM_FAMILIES['move_object'],
    'order': TRANSFORM_FAMILIES['sort_objects'],
    'reflect_and_rotate': TRANSFORM_FAMILIES['rotate_90'],
    'resizing': TRANSFORM_FAMILIES['upscale_2x'],
    'same_and_different': TRANSFORM_FAMILIES['boolean_xor'],
    'top_bottom_left_right': TRANSFORM_FAMILIES['align_objects'],
}
