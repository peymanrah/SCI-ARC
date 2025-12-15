"""
ARC Dataset Implementation for SCI-ARC.

This module provides:
1. SCIARCDataset: PyTorch Dataset for ARC tasks
2. Augmentation pipeline for grid transformations (matching TRM exactly)
3. Collate function for batching variable-size grids
4. Support for ARC-AGI-1, ARC-AGI-2, RE-ARC, and ConceptARC

Following TRM's data preparation approach with SCI-specific additions.

CRITICAL: Uses same augmentation as TRM:
- 8 dihedral transforms (D4 group)
- Color permutation (9! for colors 1-9, keeping 0 fixed)
- Translational augmentation
"""

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from functools import partial

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from .transform_families import (
    get_transform_family,
    infer_transform_from_grids,
    NUM_TRANSFORM_FAMILIES,
)


# ============================================================================
# DIHEDRAL TRANSFORMS (Matching TRM exactly from dataset/common.py)
# ============================================================================

# Inverse mapping for each dihedral transform
DIHEDRAL_INVERSE = [0, 3, 2, 1, 4, 5, 6, 7]


def dihedral_transform(arr: np.ndarray, tid: int) -> np.ndarray:
    """
    Apply one of 8 dihedral symmetries (D4 group).
    
    Matches TRM's dataset/common.py exactly.
    
    tid:
        0: identity
        1: rotate 90° CCW
        2: rotate 180°
        3: rotate 270° CCW (= 90° CW)
        4: horizontal flip (left-right)
        5: vertical flip (up-down)
        6: transpose (reflect along main diagonal)
        7: anti-transpose (reflect along anti-diagonal)
    """
    if tid == 0:
        return arr.copy()  # identity
    elif tid == 1:
        return np.rot90(arr, k=1)
    elif tid == 2:
        return np.rot90(arr, k=2)
    elif tid == 3:
        return np.rot90(arr, k=3)
    elif tid == 4:
        return np.fliplr(arr)       # horizontal flip
    elif tid == 5:
        return np.flipud(arr)       # vertical flip
    elif tid == 6:
        return arr.T.copy()         # transpose (reflection along main diagonal)
    elif tid == 7:
        return np.fliplr(np.rot90(arr, k=1))  # anti-diagonal reflection
    else:
        return arr.copy()


def inverse_dihedral_transform(arr: np.ndarray, tid: int) -> np.ndarray:
    """Apply inverse of dihedral transform."""
    return dihedral_transform(arr, DIHEDRAL_INVERSE[tid])


@dataclass
class ARCTask:
    """Represents a single ARC task."""
    task_id: str
    train_pairs: List[Tuple[np.ndarray, np.ndarray]]  # (input, output) pairs
    test_pairs: List[Tuple[np.ndarray, np.ndarray]]   # (input, output) pairs
    transform_family: int = -1  # Transformation family for SCL
    metadata: Optional[Dict] = None


@dataclass
class SCIARCSample:
    """A single training sample for SCI-ARC."""
    task_id: str
    input_grids: List[np.ndarray]   # All input grids (train + test input)
    output_grids: List[np.ndarray]  # All output grids (train + test output)
    test_input: np.ndarray          # Test input to predict
    test_output: np.ndarray         # Ground truth test output
    transform_family: int           # For SCL
    num_train_pairs: int            # Number of training examples


class SCIARCDataset(Dataset):
    """
    PyTorch Dataset for ARC tasks.
    
    Features:
    - Loads tasks from JSON format
    - Applies data augmentation (rotation, flip, color permutation)
    - Provides transformation family labels for SCL
    - Supports curriculum learning (easy -> hard)
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'training',
        augment: bool = True,
        max_grid_size: int = 30,
        num_colors: int = 10,
        include_rearc: bool = False,
        rearc_dir: Optional[str] = None,
        transform_fn: Optional[Callable] = None,
        curriculum_stage: int = 0,  # 0=all, 1=easy, 2=medium, 3=hard
    ):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Path to ARC data directory
            split: 'training' or 'evaluation'
            augment: Whether to apply data augmentation
            max_grid_size: Maximum grid dimension (for padding)
            num_colors: Number of possible colors (10 for ARC)
            include_rearc: Whether to include RE-ARC synthetic data
            rearc_dir: Path to RE-ARC data if include_rearc
            transform_fn: Optional custom transform function
            curriculum_stage: Curriculum learning stage
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.augment = augment
        self.max_grid_size = max_grid_size
        self.num_colors = num_colors
        self.transform_fn = transform_fn
        self.curriculum_stage = curriculum_stage
        
        # Load tasks
        self.tasks = self._load_tasks()
        
        # Optionally add RE-ARC data
        if include_rearc and rearc_dir:
            rearc_tasks = self._load_rearc(rearc_dir)
            self.tasks.extend(rearc_tasks)
        
        # Apply curriculum filtering
        if curriculum_stage > 0:
            self.tasks = self._filter_by_difficulty(curriculum_stage)
        
        print(f"Loaded {len(self.tasks)} tasks from {split}")
    
    def _load_tasks(self) -> List[ARCTask]:
        """Load tasks from JSON files."""
        tasks = []
        
        # Standard ARC directory structure
        split_dir = self.data_dir / self.split
        if not split_dir.exists():
            # Try alternate structure (ARC-AGI)
            split_dir = self.data_dir / f"{self.split}_challenges"
        
        if not split_dir.exists():
            # Single directory with all tasks
            split_dir = self.data_dir
        
        # Load JSON files
        for json_file in split_dir.glob('*.json'):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                task_id = json_file.stem
                
                # Parse train pairs
                train_pairs = []
                if 'train' in data:
                    for pair in data['train']:
                        inp = np.array(pair['input'], dtype=np.int64)
                        out = np.array(pair['output'], dtype=np.int64)
                        train_pairs.append((inp, out))
                
                # Parse test pairs
                test_pairs = []
                if 'test' in data:
                    for pair in data['test']:
                        inp = np.array(pair['input'], dtype=np.int64)
                        # Output may not exist in evaluation set
                        if 'output' in pair:
                            out = np.array(pair['output'], dtype=np.int64)
                        else:
                            out = None
                        test_pairs.append((inp, out))
                
                # Infer transformation family
                if train_pairs:
                    transform_family = infer_transform_from_grids(
                        train_pairs[0][0], train_pairs[0][1]
                    )
                else:
                    transform_family = get_transform_family(task_id)
                
                task = ARCTask(
                    task_id=task_id,
                    train_pairs=train_pairs,
                    test_pairs=test_pairs,
                    transform_family=transform_family,
                    metadata={'source': 'arc'}
                )
                tasks.append(task)
                
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
                continue
        
        return tasks
    
    def _load_rearc(self, rearc_dir: str) -> List[ARCTask]:
        """Load RE-ARC synthetic tasks."""
        tasks = []
        rearc_path = Path(rearc_dir)
        
        for json_file in rearc_path.glob('*.json'):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                task_id = f"rearc_{json_file.stem}"
                
                train_pairs = []
                for pair in data.get('train', []):
                    inp = np.array(pair['input'], dtype=np.int64)
                    out = np.array(pair['output'], dtype=np.int64)
                    train_pairs.append((inp, out))
                
                test_pairs = []
                for pair in data.get('test', []):
                    inp = np.array(pair['input'], dtype=np.int64)
                    out = np.array(pair.get('output', pair['input']), dtype=np.int64)
                    test_pairs.append((inp, out))
                
                # RE-ARC has explicit transform info
                metadata = data.get('metadata', {})
                metadata['source'] = 'rearc'
                transform_family = get_transform_family(task_id, metadata)
                
                task = ARCTask(
                    task_id=task_id,
                    train_pairs=train_pairs,
                    test_pairs=test_pairs,
                    transform_family=transform_family,
                    metadata=metadata
                )
                tasks.append(task)
                
            except Exception as e:
                continue
        
        return tasks
    
    def _filter_by_difficulty(self, stage: int) -> List[ARCTask]:
        """Filter tasks by difficulty for curriculum learning."""
        def task_difficulty(task: ARCTask) -> int:
            """Estimate task difficulty based on grid size and pairs."""
            if not task.train_pairs:
                return 3
            
            max_size = 0
            for inp, out in task.train_pairs:
                max_size = max(max_size, inp.shape[0], inp.shape[1])
                max_size = max(max_size, out.shape[0], out.shape[1])
            
            num_pairs = len(task.train_pairs)
            
            if max_size <= 10 and num_pairs >= 3:
                return 1  # Easy
            elif max_size <= 20 and num_pairs >= 2:
                return 2  # Medium
            else:
                return 3  # Hard
        
        return [t for t in self.tasks if task_difficulty(t) <= stage]
    
    def __len__(self) -> int:
        return len(self.tasks)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        task = self.tasks[idx]
        
        # Randomly select a test pair
        if task.test_pairs:
            test_idx = random.randint(0, len(task.test_pairs) - 1)
            test_input, test_output = task.test_pairs[test_idx]
        else:
            # If no test pairs, use last train pair
            test_input, test_output = task.train_pairs[-1]
        
        # Collect all grids
        input_grids = [pair[0] for pair in task.train_pairs]
        output_grids = [pair[1] for pair in task.train_pairs]
        
        # Apply augmentation
        if self.augment:
            input_grids, output_grids, test_input, test_output = self._augment(
                input_grids, output_grids, test_input, test_output
            )
        
        # Custom transform
        if self.transform_fn:
            return self.transform_fn({
                'task_id': task.task_id,
                'input_grids': input_grids,
                'output_grids': output_grids,
                'test_input': test_input,
                'test_output': test_output,
                'transform_family': task.transform_family,
            })
        
        # Convert to tensors
        input_tensors = [torch.tensor(g, dtype=torch.long) for g in input_grids]
        output_tensors = [torch.tensor(g, dtype=torch.long) for g in output_grids]
        
        return {
            'task_id': task.task_id,
            'input_grids': input_tensors,
            'output_grids': output_tensors,
            'test_input': torch.tensor(test_input, dtype=torch.long),
            'test_output': torch.tensor(test_output, dtype=torch.long),
            'transform_family': task.transform_family,
            'num_train_pairs': len(task.train_pairs),
        }
    
    def _augment(
        self,
        input_grids: List[np.ndarray],
        output_grids: List[np.ndarray],
        test_input: np.ndarray,
        test_output: np.ndarray
    ) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray, np.ndarray]:
        """
        Apply data augmentation matching TRM exactly.
        
        Augmentations (applied consistently to all grids in a task):
        1. Dihedral transforms (8 total - full D4 group)
        2. Color permutation (9! for colors 1-9, keeping 0 fixed)
        3. Translational augmentation (optional)
        
        CRITICAL: All augmentations match TRM's dataset/build_arc_dataset.py
        """
        # Dihedral transform (0-7)
        dihedral_id = random.randint(0, 7)
        
        # Color permutation (keep 0 fixed, permute 1-9)
        do_color_perm = random.random() < 0.5
        if do_color_perm:
            # TRM style: permute colors 1-9, keep 0 (black/background) fixed
            color_perm = list(range(1, self.num_colors))  # [1, 2, ..., 9]
            random.shuffle(color_perm)
            # color_map: old_color -> new_color
            color_map = {0: 0}  # 0 stays 0
            for i, new_c in enumerate(color_perm):
                color_map[i + 1] = new_c
        else:
            color_map = None
        
        # Translational augmentation
        do_translate = random.random() < 0.3
        if do_translate:
            # Find max grid dimensions to determine safe translation range
            all_grids = input_grids + output_grids + [test_input, test_output]
            max_h = max(g.shape[0] for g in all_grids)
            max_w = max(g.shape[1] for g in all_grids)
            # Translation range: up to 4 cells, but stay within bounds
            max_translate_r = min(4, self.max_grid_size - max_h)
            max_translate_c = min(4, self.max_grid_size - max_w)
            translate_r = random.randint(0, max(0, max_translate_r))
            translate_c = random.randint(0, max(0, max_translate_c))
        else:
            translate_r = translate_c = 0
        
        def transform_grid(grid: np.ndarray) -> np.ndarray:
            g = grid.copy()
            
            # 1. Dihedral transform
            g = dihedral_transform(g, dihedral_id)
            
            # 2. Color permutation
            if color_map:
                g_new = np.zeros_like(g)
                for old_c, new_c in color_map.items():
                    g_new[g == old_c] = new_c
                g = g_new
            
            # 3. Translational augmentation (pad grid)
            if translate_r > 0 or translate_c > 0:
                h, w = g.shape
                new_g = np.zeros((h + translate_r, w + translate_c), dtype=g.dtype)
                new_g[translate_r:translate_r + h, translate_c:translate_c + w] = g
                g = new_g
            
            return g.copy()  # Ensure contiguous
        
        # Apply to all grids
        aug_inputs = [transform_grid(g) for g in input_grids]
        aug_outputs = [transform_grid(g) for g in output_grids]
        aug_test_in = transform_grid(test_input)
        aug_test_out = transform_grid(test_output)
        
        return aug_inputs, aug_outputs, aug_test_in, aug_test_out


def pad_grid(grid: torch.Tensor, max_size: int, pad_value: int = 0) -> torch.Tensor:
    """Pad grid to max_size x max_size."""
    h, w = grid.shape
    if h >= max_size and w >= max_size:
        return grid[:max_size, :max_size]
    
    padded = torch.full((max_size, max_size), pad_value, dtype=grid.dtype)
    padded[:min(h, max_size), :min(w, max_size)] = grid[:min(h, max_size), :min(w, max_size)]
    return padded


def collate_sci_arc(batch: List[Dict], max_size: int = 30, max_grid_size: int = None) -> Dict[str, Any]:
    """
    Collate function for batching variable-size ARC grids.
    
    Strategy:
    - Pad all grids to max_size x max_size
    - Stack into batch tensors
    - Handle variable number of train pairs with padding
    
    Args:
        batch: List of samples from SCIARCDataset
        max_size: Maximum grid size for padding
        max_grid_size: Alias for max_size (for compatibility)
    
    Returns:
        Batched dictionary with:
        - task_ids: List of task IDs
        - input_grids: [B, max_pairs, H, W]
        - output_grids: [B, max_pairs, H, W]
        - test_inputs: [B, H, W]
        - test_outputs: [B, H, W]
        - transform_families: [B]
        - num_pairs: [B] actual number of train pairs per sample
        - grid_masks: [B, max_pairs] mask for valid train pairs
    """
    # Handle both parameter names
    if max_grid_size is not None:
        max_size = max_grid_size
    
    batch_size = len(batch)
    
    # Find max number of train pairs
    max_pairs = max(sample['num_train_pairs'] for sample in batch)
    
    # Initialize tensors
    input_grids = torch.zeros(batch_size, max_pairs, max_size, max_size, dtype=torch.long)
    output_grids = torch.zeros(batch_size, max_pairs, max_size, max_size, dtype=torch.long)
    test_inputs = torch.zeros(batch_size, max_size, max_size, dtype=torch.long)
    test_outputs = torch.zeros(batch_size, max_size, max_size, dtype=torch.long)
    transform_families = torch.zeros(batch_size, dtype=torch.long)
    num_pairs = torch.zeros(batch_size, dtype=torch.long)
    grid_masks = torch.zeros(batch_size, max_pairs, dtype=torch.bool)
    
    task_ids = []
    
    for i, sample in enumerate(batch):
        task_ids.append(sample['task_id'])
        n_pairs = sample['num_train_pairs']
        num_pairs[i] = n_pairs
        
        # Pad and store input/output grids
        for j in range(n_pairs):
            input_grids[i, j] = pad_grid(sample['input_grids'][j], max_size)
            output_grids[i, j] = pad_grid(sample['output_grids'][j], max_size)
            grid_masks[i, j] = True
        
        # Pad test grids
        test_inputs[i] = pad_grid(sample['test_input'], max_size)
        test_outputs[i] = pad_grid(sample['test_output'], max_size)
        
        # Transform family
        transform_families[i] = sample['transform_family']
    
    return {
        'task_ids': task_ids,
        'input_grids': input_grids,
        'output_grids': output_grids,
        'test_inputs': test_inputs,
        'test_outputs': test_outputs,
        'transform_families': transform_families,
        'num_pairs': num_pairs,
        'grid_masks': grid_masks,
    }


def seed_worker(worker_id):
    """
    Seed worker for reproducible data loading.
    
    PyTorch DataLoader workers need proper seeding for reproducibility.
    This follows PyTorch's recommended approach for deterministic data loading.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_dataloader(
    data_dir: str,
    split: str = 'training',
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = True,
    augment: bool = True,
    max_grid_size: int = 30,
    seed: int = None,
    **kwargs
) -> DataLoader:
    """
    Create a DataLoader for ARC training.
    
    Args:
        data_dir: Path to ARC data
        split: 'training' or 'evaluation'
        batch_size: Batch size
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle data
        augment: Whether to apply augmentation
        max_grid_size: Maximum grid size
        seed: Random seed for reproducibility
        **kwargs: Additional args for SCIARCDataset
    
    Returns:
        PyTorch DataLoader
    """
    dataset = SCIARCDataset(
        data_dir=data_dir,
        split=split,
        augment=augment,
        max_grid_size=max_grid_size,
        **kwargs
    )
    
    # Setup generator for reproducibility
    g = None
    worker_init = None
    if seed is not None:
        g = torch.Generator()
        g.manual_seed(seed)
        worker_init = seed_worker
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=partial(collate_sci_arc, max_grid_size=max_grid_size),
        pin_memory=True,
        drop_last=True if shuffle else False,
        worker_init_fn=worker_init,
        generator=g,
    )
    
    return loader


# For TRM compatibility
class TRMCompatibleDataset(SCIARCDataset):
    """
    Dataset formatted for TRM-style training.
    
    TRM expects:
    - Flattened 30x30 grid = 900 tokens per grid
    - Token format: PAD=0, EOS=1, colors=2-11 (vocab_size=12)
    - Translational augmentation with EOS markers
    
    This matches TRM's dataset/build_arc_dataset.py exactly.
    """
    
    # TRM token format (from build_arc_dataset.py)
    PAD_TOKEN = 0
    EOS_TOKEN = 1
    COLOR_OFFSET = 2  # Colors 0-9 map to tokens 2-11
    VOCAB_SIZE = 12
    GRID_SIZE = 30  # Fixed 30x30 grid
    SEQ_LEN = 900   # 30 * 30
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return TRM-compatible format."""
        sample = super().__getitem__(idx)
        
        # Convert to TRM format
        sample['trm_format'] = self._to_trm_format(sample)
        
        return sample
    
    def _grid_to_trm_sequence(
        self, 
        inp: np.ndarray, 
        out: np.ndarray, 
        do_translation: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert input/output grid pair to TRM's flattened sequence format.
        
        Matches TRM's np_grid_to_seq_translational_augment() exactly.
        
        Returns:
            (input_seq, output_seq) each of shape [900]
        """
        # Compute random translation offset
        if do_translation:
            max_r = self.GRID_SIZE - max(inp.shape[0], out.shape[0])
            max_c = self.GRID_SIZE - max(inp.shape[1], out.shape[1])
            pad_r = np.random.randint(0, max(1, max_r + 1))
            pad_c = np.random.randint(0, max(1, max_c + 1))
        else:
            pad_r = pad_c = 0
        
        result = []
        for grid in [inp, out]:
            nrow, ncol = grid.shape
            
            # Pad grid with color offset (colors 0-9 -> tokens 2-11)
            padded = np.pad(
                grid + self.COLOR_OFFSET, 
                ((pad_r, self.GRID_SIZE - pad_r - nrow), 
                 (pad_c, self.GRID_SIZE - pad_c - ncol)), 
                constant_values=self.PAD_TOKEN
            )
            
            # Add EOS markers at grid boundaries
            eos_row, eos_col = pad_r + nrow, pad_c + ncol
            if eos_row < self.GRID_SIZE:
                padded[eos_row, pad_c:eos_col] = self.EOS_TOKEN
            if eos_col < self.GRID_SIZE:
                padded[pad_r:eos_row, eos_col] = self.EOS_TOKEN
            
            result.append(padded.flatten())
        
        return result[0], result[1]
    
    def _to_trm_format(self, sample: Dict) -> Dict:
        """Convert sample to TRM's sequence format."""
        # For each train pair, create flattened sequences
        train_input_seqs = []
        train_output_seqs = []
        
        n_pairs = sample['num_train_pairs']
        for i in range(n_pairs):
            inp_grid = sample['input_grids'][i].numpy() if torch.is_tensor(sample['input_grids'][i]) else sample['input_grids'][i]
            out_grid = sample['output_grids'][i].numpy() if torch.is_tensor(sample['output_grids'][i]) else sample['output_grids'][i]
            
            inp_seq, out_seq = self._grid_to_trm_sequence(inp_grid, out_grid, do_translation=False)
            train_input_seqs.append(inp_seq)
            train_output_seqs.append(out_seq)
        
        # Test input/output
        test_inp = sample['test_input'].numpy() if torch.is_tensor(sample['test_input']) else sample['test_input']
        test_out = sample['test_output'].numpy() if torch.is_tensor(sample['test_output']) else sample['test_output']
        
        test_inp_seq, test_out_seq = self._grid_to_trm_sequence(test_inp, test_out, do_translation=False)
        
        return {
            'train_input_seqs': torch.tensor(np.stack(train_input_seqs), dtype=torch.long),  # [N, 900]
            'train_output_seqs': torch.tensor(np.stack(train_output_seqs), dtype=torch.long),  # [N, 900]
            'test_input_seq': torch.tensor(test_inp_seq, dtype=torch.long),  # [900]
            'test_output_seq': torch.tensor(test_out_seq, dtype=torch.long),  # [900]
            'vocab_size': self.VOCAB_SIZE,
            'seq_len': self.SEQ_LEN,
        }
