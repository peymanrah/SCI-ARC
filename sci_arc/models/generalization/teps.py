"""
TEPS: Test-Time Exhaustive Program Search

At inference time, enumerates programs that could explain all training pairs,
then executes the found program on the test input.

WHY THIS GENERALIZES:
- A program that explains ALL training examples captures the underlying rule
- This is RULE DISCOVERY, not pattern matching
- Works on novel compositions never seen during training
- The neural model guides search but correctness is verified symbolically

Integration with RLAN:
- Uses RLAN's context encoder to guide program search (optional)
- Falls back to RLAN's neural prediction if search fails
- Does NOT modify any RLAN code - pure wrapper

Author: AI Research Assistant
Date: January 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Callable, Dict, Any, Set
from dataclasses import dataclass, field
from enum import Enum, auto
import heapq
import time
import numpy as np


class PrimitiveType(Enum):
    """Categories of primitives for type-guided search."""
    GRID_TO_GRID = auto()      # Full grid transformation
    OBJECT_TO_OBJECT = auto()  # Per-object transformation
    SELECTOR = auto()          # Selects subset of grid/objects
    COMBINER = auto()          # Combines multiple inputs


@dataclass
class Primitive:
    """A primitive operation in the DSL."""
    name: str
    ptype: PrimitiveType
    arity: int  # Number of child programs (0 = atomic)
    executor: Callable
    
    def __hash__(self):
        return hash(self.name)
    
    def __eq__(self, other):
        return isinstance(other, Primitive) and self.name == other.name


@dataclass
class Program:
    """A program is a tree of primitive applications."""
    primitive: Primitive
    arguments: List['Program'] = field(default_factory=list)
    constant: Optional[Any] = None
    
    def __repr__(self):
        if self.arguments:
            args = ", ".join(repr(a) for a in self.arguments)
            return f"{self.primitive.name}({args})"
        elif self.constant is not None:
            return f"{self.primitive.name}[{self.constant}]"
        return self.primitive.name
    
    def __hash__(self):
        return hash(str(self))
    
    def __eq__(self, other):
        return str(self) == str(other)
    
    def __lt__(self, other):
        """For heap comparison."""
        return str(self) < str(other)


class PrimitiveLibrary:
    """
    Library of grid transformation primitives.
    
    Design principles:
    1. Completeness: Can express common ARC transformations
    2. Composability: Primitives combine naturally  
    3. Efficiency: Fast execution for search
    """
    
    def __init__(self):
        self.primitives: List[Primitive] = []
        self._build_library()
    
    def _build_library(self):
        """Build the complete primitive library."""
        
        # === IDENTITY ===
        self.primitives.append(Primitive(
            name='identity',
            ptype=PrimitiveType.GRID_TO_GRID,
            arity=0,
            executor=lambda g, **kw: g.clone() if isinstance(g, torch.Tensor) else np.copy(g),
        ))
        
        # === ROTATIONS ===
        self.primitives.append(Primitive(
            name='rotate_90',
            ptype=PrimitiveType.GRID_TO_GRID,
            arity=0,
            executor=self._rotate_90,
        ))
        
        self.primitives.append(Primitive(
            name='rotate_180',
            ptype=PrimitiveType.GRID_TO_GRID,
            arity=0,
            executor=self._rotate_180,
        ))
        
        self.primitives.append(Primitive(
            name='rotate_270',
            ptype=PrimitiveType.GRID_TO_GRID,
            arity=0,
            executor=self._rotate_270,
        ))
        
        # === FLIPS ===
        self.primitives.append(Primitive(
            name='flip_horizontal',
            ptype=PrimitiveType.GRID_TO_GRID,
            arity=0,
            executor=self._flip_horizontal,
        ))
        
        self.primitives.append(Primitive(
            name='flip_vertical',
            ptype=PrimitiveType.GRID_TO_GRID,
            arity=0,
            executor=self._flip_vertical,
        ))
        
        self.primitives.append(Primitive(
            name='transpose',
            ptype=PrimitiveType.GRID_TO_GRID,
            arity=0,
            executor=self._transpose,
        ))
        
        # === COLOR SWAPS (common pairs) ===
        for c1 in range(10):
            for c2 in range(c1 + 1, 10):
                self.primitives.append(Primitive(
                    name=f'swap_{c1}_{c2}',
                    ptype=PrimitiveType.GRID_TO_GRID,
                    arity=0,
                    executor=self._make_color_swap(c1, c2),
                ))
        
        # === RECOLOR (change all non-background to color) ===
        for c in range(1, 10):
            self.primitives.append(Primitive(
                name=f'recolor_{c}',
                ptype=PrimitiveType.GRID_TO_GRID,
                arity=0,
                executor=self._make_recolor(c),
            ))
        
        # === OBJECT OPERATIONS ===
        self.primitives.append(Primitive(
            name='crop_to_content',
            ptype=PrimitiveType.GRID_TO_GRID,
            arity=0,
            executor=self._crop_to_content,
        ))
        
        self.primitives.append(Primitive(
            name='extract_largest_object',
            ptype=PrimitiveType.GRID_TO_GRID,
            arity=0,
            executor=self._extract_largest_object,
        ))
        
        self.primitives.append(Primitive(
            name='fill_holes',
            ptype=PrimitiveType.GRID_TO_GRID,
            arity=0,
            executor=self._fill_holes,
        ))
        
        # === TILING ===
        self.primitives.append(Primitive(
            name='tile_2x2',
            ptype=PrimitiveType.GRID_TO_GRID,
            arity=0,
            executor=self._tile_2x2,
        ))
        
        self.primitives.append(Primitive(
            name='tile_3x3',
            ptype=PrimitiveType.GRID_TO_GRID,
            arity=0,
            executor=self._tile_3x3,
        ))
        
        # === SCALING ===
        self.primitives.append(Primitive(
            name='scale_2x',
            ptype=PrimitiveType.GRID_TO_GRID,
            arity=0,
            executor=self._scale_2x,
        ))
        
        self.primitives.append(Primitive(
            name='scale_down_2x',
            ptype=PrimitiveType.GRID_TO_GRID,
            arity=0,
            executor=self._scale_down_2x,
        ))
        
        # === COMPOSITION (arity > 0) ===
        self.primitives.append(Primitive(
            name='compose',
            ptype=PrimitiveType.COMBINER,
            arity=2,
            executor=self._compose,
        ))
    
    # === Primitive Implementations ===
    
    def _to_numpy(self, g):
        """Convert grid to numpy if tensor."""
        if isinstance(g, torch.Tensor):
            return g.cpu().numpy()
        return np.array(g)
    
    def _to_tensor(self, g, device='cpu'):
        """Convert grid to tensor."""
        if isinstance(g, torch.Tensor):
            return g
        return torch.tensor(g, dtype=torch.long, device=device)
    
    def _rotate_90(self, g, **kw):
        g = self._to_numpy(g)
        return np.rot90(g, 1)
    
    def _rotate_180(self, g, **kw):
        g = self._to_numpy(g)
        return np.rot90(g, 2)
    
    def _rotate_270(self, g, **kw):
        g = self._to_numpy(g)
        return np.rot90(g, 3)
    
    def _flip_horizontal(self, g, **kw):
        g = self._to_numpy(g)
        return np.flip(g, axis=1)
    
    def _flip_vertical(self, g, **kw):
        g = self._to_numpy(g)
        return np.flip(g, axis=0)
    
    def _transpose(self, g, **kw):
        g = self._to_numpy(g)
        return g.T
    
    def _make_color_swap(self, c1: int, c2: int) -> Callable:
        def swap(g, **kw):
            g = self._to_numpy(g)
            result = g.copy()
            mask1 = (g == c1)
            mask2 = (g == c2)
            result[mask1] = c2
            result[mask2] = c1
            return result
        return swap
    
    def _make_recolor(self, c: int) -> Callable:
        def recolor(g, **kw):
            g = self._to_numpy(g)
            result = g.copy()
            result[g != 0] = c
            return result
        return recolor
    
    def _crop_to_content(self, g, **kw):
        """Crop grid to bounding box of non-zero content."""
        g = self._to_numpy(g)
        non_bg = (g != 0)
        if not non_bg.any():
            return g
        
        rows = np.where(non_bg.any(axis=1))[0]
        cols = np.where(non_bg.any(axis=0))[0]
        
        if len(rows) == 0 or len(cols) == 0:
            return g
        
        return g[rows.min():rows.max()+1, cols.min():cols.max()+1]
    
    def _extract_largest_object(self, g, **kw):
        """Extract the largest connected component."""
        try:
            from scipy import ndimage
        except ImportError:
            return self._to_numpy(g)
        
        g = self._to_numpy(g)
        result = np.zeros_like(g)
        
        max_size = 0
        max_component = None
        max_color = 0
        
        for color in range(1, 10):
            mask = (g == color)
            if not mask.any():
                continue
            labeled, num = ndimage.label(mask)
            for i in range(1, num + 1):
                component = (labeled == i)
                size = component.sum()
                if size > max_size:
                    max_size = size
                    max_component = component
                    max_color = color
        
        if max_component is not None:
            result[max_component] = max_color
        
        return result
    
    def _fill_holes(self, g, **kw):
        """Fill enclosed background regions."""
        try:
            from scipy import ndimage
        except ImportError:
            return self._to_numpy(g)
        
        g = self._to_numpy(g)
        result = g.copy()
        
        # Find background connected to edges
        bg = (g == 0)
        labeled, _ = ndimage.label(bg)
        
        # Mark regions connected to boundary
        H, W = g.shape
        edge_labels = set()
        edge_labels.update(labeled[0, :].tolist())
        edge_labels.update(labeled[-1, :].tolist())
        edge_labels.update(labeled[:, 0].tolist())
        edge_labels.update(labeled[:, -1].tolist())
        
        # Fill non-edge background with most common non-bg color
        colors, counts = np.unique(g[g != 0], return_counts=True)
        if len(colors) > 0:
            fill_color = colors[counts.argmax()]
            for i in range(1, labeled.max() + 1):
                if i not in edge_labels:
                    result[labeled == i] = fill_color
        
        return result
    
    def _tile_2x2(self, g, **kw):
        g = self._to_numpy(g)
        return np.tile(g, (2, 2))
    
    def _tile_3x3(self, g, **kw):
        g = self._to_numpy(g)
        return np.tile(g, (3, 3))
    
    def _scale_2x(self, g, **kw):
        g = self._to_numpy(g)
        return np.repeat(np.repeat(g, 2, axis=0), 2, axis=1)
    
    def _scale_down_2x(self, g, **kw):
        g = self._to_numpy(g)
        H, W = g.shape
        if H < 2 or W < 2:
            return g
        # Take every other pixel
        return g[::2, ::2]
    
    def _compose(self, g, result1, result2, **kw):
        """Compose two transformations: result2 is applied to result1."""
        # result2 is the final result (second transformation applied)
        return result2
    
    def get_primitives(self) -> List[Primitive]:
        return self.primitives
    
    def get_atomic_primitives(self) -> List[Primitive]:
        """Get primitives with arity 0 (no arguments)."""
        return [p for p in self.primitives if p.arity == 0]


@dataclass
class TEPSConfig:
    """Configuration for TEPS."""
    enabled: bool = True
    max_program_depth: int = 2  # Max depth of program tree
    max_search_steps: int = 1000  # Max programs to evaluate
    beam_width: int = 50  # Beam width for search
    timeout_seconds: float = 5.0  # Timeout per task
    require_all_pairs: bool = True  # Must match ALL training pairs
    use_neural_prior: bool = False  # Use neural model to guide search


class TEPS(nn.Module):
    """
    Test-Time Exhaustive Program Search
    
    At test time, enumerates programs ordered by simplicity (and optionally
    neural prior probability) until finding one that explains all training pairs.
    
    This module WRAPS the base RLAN - it doesn't modify it.
    
    Usage:
        teps = TEPS(TEPSConfig())
        
        # At inference time:
        result = teps.search(
            test_input=test_grid,
            train_inputs=[in1, in2, ...],
            train_outputs=[out1, out2, ...],
        )
        
        if result['success']:
            prediction = result['prediction']
            program = result['program']
        else:
            # Fall back to neural model
            prediction = rlan.predict(...)
    """
    
    def __init__(self, config: TEPSConfig = None):
        super().__init__()
        
        self.config = config or TEPSConfig()
        self.library = PrimitiveLibrary()
        self.primitives = self.library.get_primitives()
        
        # Statistics
        self.last_search_stats: Dict[str, Any] = {}
    
    def search(
        self,
        test_input: np.ndarray,
        train_inputs: List[np.ndarray],
        train_outputs: List[np.ndarray],
    ) -> Dict[str, Any]:
        """
        Search for a program explaining all training pairs.
        
        Args:
            test_input: Test input grid (H, W) numpy array
            train_inputs: List of training input grids
            train_outputs: List of training output grids
            
        Returns:
            result: Dict with:
                - success: bool - whether a program was found
                - prediction: Optional[np.ndarray] - predicted output
                - program: Optional[Program] - the found program
                - stats: Dict - search statistics
        """
        if not self.config.enabled:
            return {
                'success': False,
                'prediction': None,
                'program': None,
                'stats': {'disabled': True},
            }
        
        start_time = time.time()
        
        # Convert inputs to numpy if needed
        def to_np(g):
            if isinstance(g, torch.Tensor):
                return g.cpu().numpy()
            return np.array(g)
        
        test_input = to_np(test_input)
        train_inputs = [to_np(g) for g in train_inputs]
        train_outputs = [to_np(g) for g in train_outputs]
        
        # Priority queue: (priority, program)
        # Lower priority = try first (simpler programs)
        pq: List[Tuple[int, Program]] = []
        seen_programs: Set[str] = set()
        
        # Initialize with atomic programs (depth 1)
        atomic_prims = self.library.get_atomic_primitives()
        for prim in atomic_prims:
            program = Program(primitive=prim, arguments=[])
            pq.append((1, program))  # Priority 1 = depth 1
            seen_programs.add(str(program))
        
        heapq.heapify(pq)
        
        best_program = None
        best_score = -1
        best_prediction = None
        steps = 0
        
        while pq and steps < self.config.max_search_steps:
            if time.time() - start_time > self.config.timeout_seconds:
                break
            
            priority, program = heapq.heappop(pq)
            steps += 1
            
            # Verify program on all training pairs
            matches = 0
            total_similarity = 0.0
            valid = True
            
            for train_in, train_out in zip(train_inputs, train_outputs):
                try:
                    predicted = self._execute_program(program, train_in)
                    
                    if predicted is None:
                        valid = False
                        break
                    
                    predicted = np.array(predicted)
                    
                    if predicted.shape == train_out.shape:
                        if np.array_equal(predicted, train_out):
                            matches += 1
                            total_similarity += 1.0
                        else:
                            sim = (predicted == train_out).mean()
                            total_similarity += sim
                    else:
                        valid = False
                        break
                        
                except Exception:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Check if this program explains ALL training pairs
            if matches == len(train_inputs):
                try:
                    prediction = self._execute_program(program, test_input)
                    if prediction is not None:
                        self.last_search_stats = {
                            'steps': steps,
                            'time': time.time() - start_time,
                            'program': str(program),
                            'success': True,
                        }
                        return {
                            'success': True,
                            'prediction': np.array(prediction),
                            'program': program,
                            'stats': self.last_search_stats,
                        }
                except Exception:
                    pass
            
            # Track best partial match
            avg_similarity = total_similarity / len(train_inputs) if train_inputs else 0
            if avg_similarity > best_score:
                best_score = avg_similarity
                best_program = program
                try:
                    best_prediction = self._execute_program(program, test_input)
                except:
                    best_prediction = None
            
            # Expand: create composed programs
            if priority < self.config.max_program_depth:
                new_programs = self._expand_program(program)
                for new_program in new_programs:
                    prog_str = str(new_program)
                    if prog_str not in seen_programs:
                        seen_programs.add(prog_str)
                        heapq.heappush(pq, (priority + 1, new_program))
            
            # Prune beam
            if len(pq) > self.config.beam_width * 3:
                pq = heapq.nsmallest(self.config.beam_width, pq)
                heapq.heapify(pq)
        
        # Search exhausted
        self.last_search_stats = {
            'steps': steps,
            'time': time.time() - start_time,
            'best_score': best_score,
            'best_program': str(best_program) if best_program else None,
            'success': False,
        }
        
        return {
            'success': False,
            'prediction': np.array(best_prediction) if best_prediction is not None else None,
            'program': best_program,
            'stats': self.last_search_stats,
        }
    
    def _execute_program(
        self, 
        program: Program, 
        grid: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Execute a program on a grid."""
        prim = program.primitive
        
        if prim.arity == 0:
            return prim.executor(grid)
        elif prim.arity == 2:
            # Compose: apply both arguments in sequence
            arg1_result = self._execute_program(program.arguments[0], grid)
            if arg1_result is None:
                return None
            arg2_result = self._execute_program(program.arguments[1], arg1_result)
            return arg2_result
        
        return None
    
    def _expand_program(self, subprogram: Program) -> List[Program]:
        """Create new programs using subprogram as argument."""
        results = []
        
        # Compose with each atomic primitive
        atomic_prims = self.library.get_atomic_primitives()
        for prim in atomic_prims:
            if prim.name != 'identity':
                # Apply prim after subprogram
                compose_prim = self.primitives[-1]  # 'compose'
                new_program = Program(
                    primitive=compose_prim,
                    arguments=[subprogram, Program(primitive=prim, arguments=[])],
                )
                results.append(new_program)
        
        return results
    
    def forward(self, *args, **kwargs):
        """Forward is not used - use search() instead."""
        raise NotImplementedError("TEPS uses search() method, not forward()")
