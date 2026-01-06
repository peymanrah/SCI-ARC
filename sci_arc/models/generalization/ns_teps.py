"""
NS-TEPS: Neuro-Symbolic Test-Time Exhaustive Program Search

Extends TEPS with neuro-symbolic capabilities:
1. Object mask extraction (simulated DSC)
2. Latent primitive library with object-level operations
3. Program trace generator instead of pixel prediction

WHY THIS GENERALIZES:
- Works at the OBJECT level, not pixel level
- Discovers compositional rules: "for each object, apply transform X"
- Program traces are interpretable and verifiable

Integration with RLAN:
- Uses DSC-style object extraction
- Generates program traces, not pixels
- Falls back to base TEPS if no object-level solution found

Author: AI Research Assistant
Date: January 2026
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Set, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
import time
from collections import defaultdict
from scipy import ndimage


class ObjectPrimitiveType(Enum):
    """Categories of object-level primitives."""
    OBJECT_TRANSFORM = auto()    # Transform individual objects
    OBJECT_RELATION = auto()     # Relations between objects
    GLOBAL_ARRANGEMENT = auto()  # Arrange all objects
    OBJECT_FILTER = auto()       # Select subset of objects
    OBJECT_COMBINE = auto()      # Combine/merge objects


@dataclass
class ObjectPrimitive:
    """An object-level primitive operation."""
    name: str
    ptype: ObjectPrimitiveType
    executor: Callable  # (objects: List[np.ndarray], grid: np.ndarray, params: dict) -> np.ndarray
    param_space: Optional[List[Any]] = None  # Possible parameter values
    
    def __hash__(self):
        return hash(self.name)
    
    def __eq__(self, other):
        return isinstance(other, ObjectPrimitive) and self.name == other.name


@dataclass
class ProgramTrace:
    """A trace of primitive applications."""
    steps: List[Tuple[ObjectPrimitive, Dict[str, Any]]]  # (primitive, params)
    
    def __repr__(self):
        return " -> ".join(f"{s[0].name}({s[1]})" for s in self.steps)
    
    def __hash__(self):
        return hash(str(self))
    
    def __eq__(self, other):
        return str(self) == str(other)


@dataclass
class NSTEPSConfig:
    """Configuration for NS-TEPS."""
    enabled: bool = True
    max_search_steps: int = 2000
    timeout_seconds: float = 15.0
    max_trace_length: int = 3
    min_object_size: int = 1
    max_objects: int = 20
    sample_count: int = 1000  # Number of program traces to sample
    match_threshold: float = 0.95  # Required match for success


class ObjectExtractor:
    """
    Extracts objects from ARC grids using connected component analysis.
    Simulates the DSC (Dynamic Spatial Cluing) object extraction.
    """
    
    def __init__(self, min_size: int = 1, max_objects: int = 20):
        self.min_size = min_size
        self.max_objects = max_objects
    
    def extract(self, grid: np.ndarray) -> List[Dict[str, Any]]:
        """
        Extract objects from grid.
        
        Returns list of objects with:
        - 'mask': boolean mask of object pixels
        - 'color': dominant color of object
        - 'bbox': (y1, x1, y2, x2) bounding box
        - 'pixels': list of (y, x) coordinates
        - 'size': number of pixels
        """
        objects = []
        
        # Extract objects for each non-background color
        for color in range(1, 10):  # Colors 1-9 (0 is background)
            color_mask = (grid == color)
            if not color_mask.any():
                continue
            
            # Find connected components
            labeled, num_features = ndimage.label(color_mask)
            
            for obj_id in range(1, num_features + 1):
                obj_mask = (labeled == obj_id)
                size = obj_mask.sum()
                
                if size < self.min_size:
                    continue
                
                # Get bounding box
                ys, xs = np.where(obj_mask)
                bbox = (ys.min(), xs.min(), ys.max() + 1, xs.max() + 1)
                
                objects.append({
                    'mask': obj_mask,
                    'color': color,
                    'bbox': bbox,
                    'pixels': list(zip(ys.tolist(), xs.tolist())),
                    'size': size,
                })
                
                if len(objects) >= self.max_objects:
                    break
            
            if len(objects) >= self.max_objects:
                break
        
        # Sort by size (largest first) for consistent ordering
        objects.sort(key=lambda x: -x['size'])
        
        return objects


class ObjectPrimitiveLibrary:
    """
    Library of object-level primitives for NS-TEPS.
    
    Each primitive operates on objects extracted from the grid.
    """
    
    def __init__(self):
        self.primitives: Dict[str, ObjectPrimitive] = {}
        self._build_library()
    
    def _build_library(self):
        """Build the library of object primitives."""
        
        # ============================================
        # OBJECT TRANSFORM PRIMITIVES
        # ============================================
        
        def rotate_objects_90(objects, grid, params):
            """Rotate each object 90 degrees around its center."""
            result = np.zeros_like(grid)
            for obj in objects:
                y1, x1, y2, x2 = obj['bbox']
                obj_grid = grid[y1:y2, x1:x2].copy()
                obj_grid = np.where(obj['mask'][y1:y2, x1:x2], obj_grid, 0)
                rotated = np.rot90(obj_grid, 1)
                # Place back (may go out of bounds, handle gracefully)
                rh, rw = rotated.shape
                try:
                    result[y1:y1+rh, x1:x1+rw] = np.where(rotated > 0, rotated, result[y1:y1+rh, x1:x1+rw])
                except:
                    pass
            return result
        
        self.primitives['rotate_objects_90'] = ObjectPrimitive(
            name='rotate_objects_90',
            ptype=ObjectPrimitiveType.OBJECT_TRANSFORM,
            executor=rotate_objects_90,
        )
        
        def flip_objects_horizontal(objects, grid, params):
            """Flip each object horizontally."""
            result = np.zeros_like(grid)
            for obj in objects:
                y1, x1, y2, x2 = obj['bbox']
                obj_grid = grid[y1:y2, x1:x2].copy()
                obj_grid = np.where(obj['mask'][y1:y2, x1:x2], obj_grid, 0)
                flipped = np.fliplr(obj_grid)
                result[y1:y2, x1:x2] = np.where(flipped > 0, flipped, result[y1:y2, x1:x2])
            return result
        
        self.primitives['flip_objects_horizontal'] = ObjectPrimitive(
            name='flip_objects_horizontal',
            ptype=ObjectPrimitiveType.OBJECT_TRANSFORM,
            executor=flip_objects_horizontal,
        )
        
        def flip_objects_vertical(objects, grid, params):
            """Flip each object vertically."""
            result = np.zeros_like(grid)
            for obj in objects:
                y1, x1, y2, x2 = obj['bbox']
                obj_grid = grid[y1:y2, x1:x2].copy()
                obj_grid = np.where(obj['mask'][y1:y2, x1:x2], obj_grid, 0)
                flipped = np.flipud(obj_grid)
                result[y1:y2, x1:x2] = np.where(flipped > 0, flipped, result[y1:y2, x1:x2])
            return result
        
        self.primitives['flip_objects_vertical'] = ObjectPrimitive(
            name='flip_objects_vertical',
            ptype=ObjectPrimitiveType.OBJECT_TRANSFORM,
            executor=flip_objects_vertical,
        )
        
        # ============================================
        # COLOR PRIMITIVES
        # ============================================
        
        def recolor_objects(objects, grid, params):
            """Recolor all objects to a specific color."""
            target_color = params.get('color', 1)
            result = grid.copy()
            for obj in objects:
                result[obj['mask']] = target_color
            return result
        
        self.primitives['recolor_objects'] = ObjectPrimitive(
            name='recolor_objects',
            ptype=ObjectPrimitiveType.OBJECT_TRANSFORM,
            executor=recolor_objects,
            param_space=[{'color': c} for c in range(1, 10)],
        )
        
        def swap_colors(objects, grid, params):
            """Swap two colors in the grid."""
            c1, c2 = params.get('c1', 1), params.get('c2', 2)
            result = grid.copy()
            mask1 = (grid == c1)
            mask2 = (grid == c2)
            result[mask1] = c2
            result[mask2] = c1
            return result
        
        self.primitives['swap_colors'] = ObjectPrimitive(
            name='swap_colors',
            ptype=ObjectPrimitiveType.OBJECT_TRANSFORM,
            executor=swap_colors,
            param_space=[{'c1': c1, 'c2': c2} for c1 in range(1, 10) for c2 in range(c1+1, 10)],
        )
        
        # ============================================
        # GLOBAL ARRANGEMENT PRIMITIVES
        # ============================================
        
        def gravity_down(objects, grid, params):
            """Apply gravity: move all objects down."""
            result = np.zeros_like(grid)
            h, w = grid.shape
            for obj in objects:
                y1, x1, y2, x2 = obj['bbox']
                obj_h = y2 - y1
                # Move to bottom
                new_y1 = h - obj_h
                obj_grid = grid[y1:y2, x1:x2].copy()
                obj_grid = np.where(obj['mask'][y1:y2, x1:x2], obj_grid, 0)
                result[new_y1:h, x1:x2] = np.where(obj_grid > 0, obj_grid, result[new_y1:h, x1:x2])
            return result
        
        self.primitives['gravity_down'] = ObjectPrimitive(
            name='gravity_down',
            ptype=ObjectPrimitiveType.GLOBAL_ARRANGEMENT,
            executor=gravity_down,
        )
        
        def gravity_left(objects, grid, params):
            """Apply gravity: move all objects left."""
            result = np.zeros_like(grid)
            for obj in objects:
                y1, x1, y2, x2 = obj['bbox']
                obj_w = x2 - x1
                # Move to left edge
                obj_grid = grid[y1:y2, x1:x2].copy()
                obj_grid = np.where(obj['mask'][y1:y2, x1:x2], obj_grid, 0)
                result[y1:y2, 0:obj_w] = np.where(obj_grid > 0, obj_grid, result[y1:y2, 0:obj_w])
            return result
        
        self.primitives['gravity_left'] = ObjectPrimitive(
            name='gravity_left',
            ptype=ObjectPrimitiveType.GLOBAL_ARRANGEMENT,
            executor=gravity_left,
        )
        
        def mirror_horizontal(objects, grid, params):
            """Mirror the entire grid horizontally."""
            return np.fliplr(grid)
        
        self.primitives['mirror_horizontal'] = ObjectPrimitive(
            name='mirror_horizontal',
            ptype=ObjectPrimitiveType.GLOBAL_ARRANGEMENT,
            executor=mirror_horizontal,
        )
        
        def mirror_vertical(objects, grid, params):
            """Mirror the entire grid vertically."""
            return np.flipud(grid)
        
        self.primitives['mirror_vertical'] = ObjectPrimitive(
            name='mirror_vertical',
            ptype=ObjectPrimitiveType.GLOBAL_ARRANGEMENT,
            executor=mirror_vertical,
        )
        
        def rotate_grid_90(objects, grid, params):
            """Rotate entire grid 90 degrees."""
            return np.rot90(grid, 1)
        
        self.primitives['rotate_grid_90'] = ObjectPrimitive(
            name='rotate_grid_90',
            ptype=ObjectPrimitiveType.GLOBAL_ARRANGEMENT,
            executor=rotate_grid_90,
        )
        
        def rotate_grid_180(objects, grid, params):
            """Rotate entire grid 180 degrees."""
            return np.rot90(grid, 2)
        
        self.primitives['rotate_grid_180'] = ObjectPrimitive(
            name='rotate_grid_180',
            ptype=ObjectPrimitiveType.GLOBAL_ARRANGEMENT,
            executor=rotate_grid_180,
        )
        
        def rotate_grid_270(objects, grid, params):
            """Rotate entire grid 270 degrees."""
            return np.rot90(grid, 3)
        
        self.primitives['rotate_grid_270'] = ObjectPrimitive(
            name='rotate_grid_270',
            ptype=ObjectPrimitiveType.GLOBAL_ARRANGEMENT,
            executor=rotate_grid_270,
        )
        
        def tile_2x2(objects, grid, params):
            """Tile the grid 2x2."""
            return np.tile(grid, (2, 2))
        
        self.primitives['tile_2x2'] = ObjectPrimitive(
            name='tile_2x2',
            ptype=ObjectPrimitiveType.GLOBAL_ARRANGEMENT,
            executor=tile_2x2,
        )
        
        # ============================================
        # OBJECT FILTER PRIMITIVES
        # ============================================
        
        def keep_largest(objects, grid, params):
            """Keep only the largest object."""
            if not objects:
                return np.zeros_like(grid)
            largest = max(objects, key=lambda o: o['size'])
            result = np.zeros_like(grid)
            result[largest['mask']] = grid[largest['mask']]
            return result
        
        self.primitives['keep_largest'] = ObjectPrimitive(
            name='keep_largest',
            ptype=ObjectPrimitiveType.OBJECT_FILTER,
            executor=keep_largest,
        )
        
        def keep_smallest(objects, grid, params):
            """Keep only the smallest object."""
            if not objects:
                return np.zeros_like(grid)
            smallest = min(objects, key=lambda o: o['size'])
            result = np.zeros_like(grid)
            result[smallest['mask']] = grid[smallest['mask']]
            return result
        
        self.primitives['keep_smallest'] = ObjectPrimitive(
            name='keep_smallest',
            ptype=ObjectPrimitiveType.OBJECT_FILTER,
            executor=keep_smallest,
        )
        
        def remove_background(objects, grid, params):
            """Remove background, keep only objects."""
            result = np.zeros_like(grid)
            for obj in objects:
                result[obj['mask']] = grid[obj['mask']]
            return result
        
        self.primitives['remove_background'] = ObjectPrimitive(
            name='remove_background',
            ptype=ObjectPrimitiveType.OBJECT_FILTER,
            executor=remove_background,
        )
        
        # ============================================
        # IDENTITY (for composition base)
        # ============================================
        
        def identity(objects, grid, params):
            """Return grid unchanged."""
            return grid.copy()
        
        self.primitives['identity'] = ObjectPrimitive(
            name='identity',
            ptype=ObjectPrimitiveType.OBJECT_TRANSFORM,
            executor=identity,
        )
    
    def get_primitive(self, name: str) -> Optional[ObjectPrimitive]:
        return self.primitives.get(name)
    
    def all_primitives(self) -> List[ObjectPrimitive]:
        return list(self.primitives.values())
    
    def primitives_by_type(self, ptype: ObjectPrimitiveType) -> List[ObjectPrimitive]:
        return [p for p in self.primitives.values() if p.ptype == ptype]


class NSTEPS(nn.Module):
    """
    Neuro-Symbolic Test-Time Exhaustive Program Search.
    
    Extends TEPS with object-level reasoning and program trace generation.
    """
    
    def __init__(self, config: NSTEPSConfig = None):
        super().__init__()
        self.config = config or NSTEPSConfig()
        self.object_extractor = ObjectExtractor(
            min_size=self.config.min_object_size,
            max_objects=self.config.max_objects,
        )
        self.primitive_library = ObjectPrimitiveLibrary()
    
    @torch.no_grad()
    def search(
        self,
        test_input: np.ndarray,
        train_inputs: List[np.ndarray],
        train_outputs: List[np.ndarray],
    ) -> Dict[str, Any]:
        """
        Search for a program trace that explains all training pairs.
        
        Returns:
            Dict with:
            - success: Whether a matching program was found
            - prediction: The predicted output (or None)
            - trace: The program trace (or None)
            - confidence: Confidence score
            - steps_searched: Number of programs tried
        """
        if not self.config.enabled:
            return {
                'success': False,
                'prediction': None,
                'trace': None,
                'confidence': 0.0,
                'steps_searched': 0,
            }
        
        start_time = time.time()
        steps_searched = 0
        best_trace = None
        best_match = 0.0
        best_prediction = None
        
        # Extract objects from all grids
        train_objects = [self.object_extractor.extract(inp) for inp in train_inputs]
        test_objects = self.object_extractor.extract(test_input)
        
        # Generate program traces to try
        traces_to_try = self._generate_traces()
        
        for trace in traces_to_try:
            if time.time() - start_time > self.config.timeout_seconds:
                break
            if steps_searched >= self.config.max_search_steps:
                break
            
            steps_searched += 1
            
            # Try this trace on all training pairs
            match_score = self._evaluate_trace(
                trace, train_inputs, train_outputs, train_objects
            )
            
            if match_score > best_match:
                best_match = match_score
                best_trace = trace
                
                # Generate prediction for test input
                try:
                    best_prediction = self._execute_trace(trace, test_input, test_objects)
                except Exception:
                    best_prediction = None
            
            # Perfect match found
            if match_score >= self.config.match_threshold:
                return {
                    'success': True,
                    'prediction': best_prediction,
                    'trace': best_trace,
                    'confidence': match_score,
                    'steps_searched': steps_searched,
                }
        
        # Return best partial match
        return {
            'success': best_match >= self.config.match_threshold,
            'prediction': best_prediction,
            'trace': best_trace,
            'confidence': best_match,
            'steps_searched': steps_searched,
        }
    
    def _generate_traces(self) -> List[ProgramTrace]:
        """Generate program traces to search."""
        traces = []
        primitives = self.primitive_library.all_primitives()
        
        # Single-step traces
        for prim in primitives:
            if prim.param_space:
                for params in prim.param_space[:5]:  # Limit param combinations
                    traces.append(ProgramTrace(steps=[(prim, params)]))
            else:
                traces.append(ProgramTrace(steps=[(prim, {})]))
        
        # Two-step traces (compositions)
        if self.config.max_trace_length >= 2:
            single_traces = traces.copy()
            for t1 in single_traces[:30]:  # Limit to prevent explosion
                for t2 in single_traces[:30]:
                    traces.append(ProgramTrace(
                        steps=t1.steps + t2.steps
                    ))
        
        # Limit total traces
        return traces[:self.config.sample_count]
    
    def _evaluate_trace(
        self,
        trace: ProgramTrace,
        train_inputs: List[np.ndarray],
        train_outputs: List[np.ndarray],
        train_objects: List[List[Dict]],
    ) -> float:
        """Evaluate how well a trace explains training pairs."""
        total_match = 0.0
        
        for inp, out, objects in zip(train_inputs, train_outputs, train_objects):
            try:
                predicted = self._execute_trace(trace, inp, objects)
                if predicted is not None and predicted.shape == out.shape:
                    match = np.mean(predicted == out)
                    total_match += match
                else:
                    total_match += 0.0
            except Exception:
                total_match += 0.0
        
        return total_match / len(train_inputs) if train_inputs else 0.0
    
    def _execute_trace(
        self,
        trace: ProgramTrace,
        grid: np.ndarray,
        objects: List[Dict],
    ) -> np.ndarray:
        """Execute a program trace on a grid."""
        result = grid.copy()
        current_objects = objects
        
        for primitive, params in trace.steps:
            result = primitive.executor(current_objects, result, params)
            # Re-extract objects after each step
            current_objects = self.object_extractor.extract(result)
        
        return result
