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
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.nn.utils.rnn import pad_sequence

from .transform_families import (
    get_transform_family,
    infer_transform_from_grids,
    NUM_TRANSFORM_FAMILIES,
)


# ============================================================================
# Stratified Sampling for Representative Testing
# ============================================================================

def stratified_sample_tasks(
    tasks: List[Dict],
    n_samples: int,
    seed: int = 42,
    verbose: bool = True
) -> List[Dict]:
    """
    Stratified sampling to ensure representative coverage of ARC tasks.
    
    Stratifies by multiple dimensions to ensure the sample is representative:
    1. Grid size bucket (small/medium/large) - preserves bucketed batching benefits
    2. Number of train pairs (1-5+) - covers few-shot to many-shot tasks
    3. Transform complexity (inferred from grid analysis)
    
    This guarantees that even a 5% sample will cover the full diversity of
    the 400 ARC training tasks, enabling valid meta-learning signal during
    smoke tests and epoch-end validation.
    
    Args:
        tasks: List of task dictionaries with 'task_id', 'train', 'test' keys
        n_samples: Number of tasks to sample
        seed: Random seed for deterministic, reproducible sampling
        verbose: Print stratification statistics
        
    Returns:
        Stratified subset of tasks
    """
    if n_samples >= len(tasks):
        return tasks
    
    # Use fixed seed for deterministic coverage
    rng = random.Random(seed)
    
    # Compute stratification features for each task
    def get_max_grid_size(task: Dict) -> int:
        """Get maximum grid dimension across all grids in task."""
        max_dim = 0
        for pair in task.get('train', []):
            inp = pair.get('input', [])
            out = pair.get('output', [])
            if inp:
                max_dim = max(max_dim, len(inp), len(inp[0]) if inp else 0)
            if out:
                max_dim = max(max_dim, len(out), len(out[0]) if out else 0)
        for pair in task.get('test', []):
            inp = pair.get('input', [])
            if inp:
                max_dim = max(max_dim, len(inp), len(inp[0]) if inp else 0)
        return max_dim
    
    def get_grid_size_bucket(size: int) -> str:
        """Bucket grid sizes for stratification (matches BucketedBatchSampler)."""
        if size <= 10:
            return "small"
        elif size <= 20:
            return "medium"
        else:
            return "large"
    
    def get_num_pairs_bucket(task: Dict) -> str:
        """Bucket number of training pairs."""
        n = len(task.get('train', []))
        if n <= 2:
            return "few_shot"  # 1-2 pairs
        elif n <= 4:
            return "medium_shot"  # 3-4 pairs
        else:
            return "many_shot"  # 5+ pairs
    
    def get_size_change_type(task: Dict) -> str:
        """Detect if task involves grid size changes."""
        for pair in task.get('train', []):
            inp = pair.get('input', [])
            out = pair.get('output', [])
            if inp and out:
                in_h, in_w = len(inp), len(inp[0]) if inp else 0
                out_h, out_w = len(out), len(out[0]) if out else 0
                if in_h != out_h or in_w != out_w:
                    return "size_change"
        return "same_size"
    
    # Create stratification key for each task
    strata: Dict[str, List[Dict]] = {}
    for task in tasks:
        grid_size = get_max_grid_size(task)
        key = (
            get_grid_size_bucket(grid_size),
            get_num_pairs_bucket(task),
            get_size_change_type(task)
        )
        strata_key = f"{key[0]}_{key[1]}_{key[2]}"
        if strata_key not in strata:
            strata[strata_key] = []
        strata[strata_key].append(task)
    
    # Sample proportionally from each stratum
    sampled = []
    remaining = n_samples
    strata_list = list(strata.items())
    rng.shuffle(strata_list)  # Randomize order for fairness
    
    # First pass: proportional allocation
    for strata_key, strata_tasks in strata_list:
        # Proportional allocation based on stratum size
        stratum_proportion = len(strata_tasks) / len(tasks)
        stratum_allocation = max(1, int(n_samples * stratum_proportion))
        stratum_allocation = min(stratum_allocation, len(strata_tasks), remaining)
        
        rng.shuffle(strata_tasks)
        sampled.extend(strata_tasks[:stratum_allocation])
        remaining -= stratum_allocation
        
        if remaining <= 0:
            break
    
    # Second pass: fill remaining slots with under-represented strata
    if remaining > 0:
        # Collect unused tasks from all strata
        used_ids = {t['task_id'] for t in sampled}
        unused = [t for t in tasks if t['task_id'] not in used_ids]
        rng.shuffle(unused)
        sampled.extend(unused[:remaining])
    
    # Final shuffle to avoid stratum ordering bias during training
    rng.shuffle(sampled)
    
    if verbose:
        # Report coverage statistics
        sampled_strata = {}
        for task in sampled:
            grid_size = get_max_grid_size(task)
            key = f"{get_grid_size_bucket(grid_size)}_{get_num_pairs_bucket(task)}_{get_size_change_type(task)}"
            sampled_strata[key] = sampled_strata.get(key, 0) + 1
        
        print(f"  Stratified sampling: {len(sampled)}/{len(tasks)} tasks ({100*len(sampled)/len(tasks):.1f}%)")
        print(f"  Strata covered: {len(sampled_strata)}/{len(strata)} ({100*len(sampled_strata)/len(strata):.1f}%)")
        print(f"  Distribution: " + ", ".join(f"{k}={v}" for k, v in sorted(sampled_strata.items())))
    
    return sampled


# ============================================================================
# BucketedBatchSampler: Groups samples by grid size for memory efficiency
# ============================================================================

class BucketedBatchSampler(Sampler):
    """
    Batch sampler that groups samples by grid size for memory efficiency.
    
    Problem: When batching variable-size grids, ALL samples are padded to the
    MAX size in the batch. One 30x30 grid in a batch of 69 15x15 grids causes
    all 70 samples to use 30x30 memory = 4x waste!
    
    Solution: Group samples into buckets by grid size, create batches within
    buckets, then shuffle the batch ORDER (not the samples within batches).
    
    This gives:
    - Memory efficiency: Similar sizes per batch = minimal padding
    - Unbiased learning: Batch order is random each epoch
    - Same sample frequency: Each sample seen exactly once per epoch
    
    Usage:
        sampler = BucketedBatchSampler(
            dataset,
            batch_size=70,
            bucket_boundaries=[10, 15, 20, 25],  # Grid size thresholds
            drop_last=False,
        )
        loader = DataLoader(dataset, batch_sampler=sampler)
    """
    
    def __init__(
        self,
        dataset,
        batch_size: int,
        bucket_boundaries: List[int] = None,
        drop_last: bool = False,
        shuffle: bool = True,
        seed: int = 42,
    ):
        """
        Args:
            dataset: Dataset with cached samples
            batch_size: Number of samples per batch
            bucket_boundaries: Grid size thresholds [10, 15, 20, 25] creates buckets:
                               [0-10], [11-15], [16-20], [21-25], [26+]
                               Default: [10, 15, 20, 25] covers ARC grid sizes well
            drop_last: Whether to drop incomplete batches
            shuffle: Whether to shuffle batch order each epoch
            seed: Random seed for reproducibility
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.bucket_boundaries = bucket_boundaries or [10, 15, 20, 25]
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        
        # Build buckets: map sample indices to their grid size bucket
        self._build_buckets()
    
    def _get_sample_max_grid_size(self, idx: int) -> int:
        """Get the maximum grid dimension for a sample."""
        sample = self.dataset[idx]
        
        max_size = 0
        
        # Check test input/output
        if 'test_input' in sample:
            t = sample['test_input']
            if isinstance(t, torch.Tensor):
                # Get non-padded size by finding last non-pad row/col
                # For now, just use tensor shape (it's padded to max_size anyway)
                # The key is relative sizes between samples
                max_size = max(max_size, t.shape[-1], t.shape[-2])
            elif isinstance(t, np.ndarray):
                max_size = max(max_size, t.shape[-1], t.shape[-2])
        
        # Check train input/output grids (list of tensors)
        for key in ['input_grids', 'output_grids']:
            if key in sample:
                for g in sample[key]:
                    if isinstance(g, torch.Tensor):
                        max_size = max(max_size, g.shape[-1], g.shape[-2])
                    elif isinstance(g, np.ndarray):
                        max_size = max(max_size, g.shape[-1], g.shape[-2])
        
        return max_size
    
    def _get_bucket_id(self, grid_size: int) -> int:
        """Map grid size to bucket ID."""
        for i, boundary in enumerate(self.bucket_boundaries):
            if grid_size <= boundary:
                return i
        return len(self.bucket_boundaries)  # Largest bucket
    
    def _build_buckets(self):
        """Build sample index buckets based on grid sizes."""
        num_buckets = len(self.bucket_boundaries) + 1
        self.buckets = [[] for _ in range(num_buckets)]
        
        # Sample a subset for large datasets to speed up bucket building
        dataset_len = len(self.dataset)
        
        print(f"[BucketedBatchSampler] Building {num_buckets} buckets for {dataset_len:,} samples...")
        
        # For cached samples, we can access grid sizes efficiently
        for idx in range(dataset_len):
            # Use a fast path if dataset stores grid size metadata
            if hasattr(self.dataset, '_cached_samples') and self.dataset._cached_samples:
                sample = self.dataset._cached_samples[idx]
                # Use stored original_max_size if available (before padding)
                # This is CRITICAL: padded grids are all max_size (e.g., 30x30)
                # but we want to bucket by ORIGINAL size for memory efficiency
                if 'original_max_size' in sample:
                    max_size = sample['original_max_size']
                else:
                    # Fallback: compute from tensor shapes (all padded, so less useful)
                    max_size = 0
                    if 'test_input' in sample:
                        t = sample['test_input']
                        if isinstance(t, torch.Tensor):
                            max_size = max(max_size, t.shape[-1], t.shape[-2])
                    for key in ['input_grids', 'output_grids']:
                        if key in sample:
                            for g in sample[key]:
                                if isinstance(g, torch.Tensor):
                                    max_size = max(max_size, g.shape[-1], g.shape[-2])
            else:
                max_size = self._get_sample_max_grid_size(idx)
            
            bucket_id = self._get_bucket_id(max_size)
            self.buckets[bucket_id].append(idx)
        
        # Print bucket statistics
        for i, bucket in enumerate(self.buckets):
            if i < len(self.bucket_boundaries):
                size_range = f"<={self.bucket_boundaries[i]}"
            else:
                size_range = f">{self.bucket_boundaries[-1]}"
            print(f"  Bucket {i} (grid {size_range}): {len(bucket):,} samples")
        
        # Pre-compute batches
        self._create_batches()
    
    def _create_batches(self):
        """Create batches from buckets."""
        self.batches = []
        
        for bucket in self.buckets:
            # Shuffle within bucket for variety
            bucket_copy = bucket.copy()
            random.Random(self.seed + self.epoch).shuffle(bucket_copy)
            
            # Create batches
            for i in range(0, len(bucket_copy), self.batch_size):
                batch = bucket_copy[i:i + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    self.batches.append(batch)
        
        # Shuffle batch order (KEY: unbiased learning)
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(self.batches)
    
    def __iter__(self):
        """Iterate over batches."""
        # Recreate batches with current epoch's shuffle
        self._create_batches()
        
        for batch in self.batches:
            yield batch
    
    def __len__(self):
        """Number of batches."""
        return len(self.batches)
    
    def set_epoch(self, epoch: int):
        """Set epoch for shuffling (call before each epoch)."""
        self.epoch = epoch


# ============================================================================
# ARCDataset: Simple JSON-based loader for RLAN training
# ============================================================================

class ARCDataset(Dataset):
    """
    Simple ARC Dataset that loads from JSON files or directories.
    
    This is a lightweight dataset class for RLAN training that supports:
    - Single combined JSON file (e.g., arc-agi_training_combined.json)
    - Directory with individual JSON task files (e.g., ./data/arc-agi/data/training/)
    
    Augmentation (matching TRM exactly):
    - 8 dihedral transforms (D4 group): rotations + flips + transposes
    - Color permutation (9! = 362,880 possibilities): permute colors 1-9, keep 0 fixed
    - Translational augmentation: random offset within 30×30 canvas
    - Total: 8 × 362,880 × ~100 positions = ~290M unique augmentations per task!
    
    With cache_samples=false (default), EVERY __getitem__ call generates a NEW
    random augmentation, providing infinite diversity during training.
    
    This is SUPERIOR to TRM's pre-generated 1000× augmentation because:
    - TRM sees same 1000 augmentations every epoch → overfitting risk
    - RLAN sees completely NEW random augmentations each epoch → better generalization
    
    AUGMENTATION TRACKING:
    - Tracks which augmentations are applied per sample
    - Returns augmentation metadata for logging and debugging
    - Enables verification of truly random augmentation distribution
    """
    
    # Padding value for target grids (used to ignore padding in loss)
    # -100 is the standard PyTorch ignore_index value for cross-entropy loss
    # Padding pixels with this value are ignored in loss computation
    PADDING_IGNORE_VALUE = -100
    
    # Padding value for INPUT grids (distinguishable from black=0)
    # Using 10 as padding token for inputs allows the model to distinguish
    # actual black pixels (0) from padding regions. GridEncoder must handle 11 colors.
    PAD_COLOR = 10
    
    # RLAN uses 2D spatial structure (B, H, W, D) - no boundary markers needed
    # Unlike TRM which flattens grids to 1D sequences and uses EOS tokens,
    # RLAN maintains spatial structure so boundary markers are unnecessary.
    # Colors 0-9 are used directly, plus 10 for padding (11 classes total for encoder)
    
    def __init__(
        self,
        data_path: str,
        max_size: int = 30,
        augment: bool = True,
        color_permutation: bool = False,
        color_permutation_prob: float = 1.0,  # NEW: probability of applying color permutation
        translational_augment: bool = True,  # NEW: TRM-style random offset
        curriculum_stage: int = 0,  # 0=all, 1=easy, 2=medium, 3=hard
        track_augmentation: bool = True,  # Track augmentation stats for debugging
        ignore_padding_in_loss: bool = True,  # NEW: Use -100 for padding in targets
        cache_samples: bool = False,  # NEW: Cache pre-generated samples for memorization
        num_cached_samples: int = 32000,  # NEW: Number of cached samples to generate
        cache_path: str = None,  # NEW: Path to load/save cached samples
        cache_load_percent: float = 100.0,  # NEW: Percentage of cache to load (1-100)
        max_tasks: int = None,  # NEW: Limit number of tasks loaded (for testing)
        stratified_seed: int = 42,  # NEW: Seed for deterministic stratified sampling
    ):
        """
        Initialize ARCDataset.
        
        Args:
            data_path: Path to JSON file or directory containing task JSONs
            max_size: Maximum grid size (grids are padded to this size)
            augment: Whether to apply dihedral augmentation (rotation, flip, transpose)
            color_permutation: Whether to apply random color permutation (9! possibilities)
            color_permutation_prob: Probability of applying color permutation (0.0-1.0)
                                    100% can break color identity learning!
                                    Recommended: 0.3-0.5 for balanced augmentation
            translational_augment: Whether to apply random positional offset (TRM-style)
            curriculum_stage: Curriculum learning stage (0=all, 1=easy, 2=+medium, 3=+hard)
            track_augmentation: Whether to return augmentation metadata for logging
            ignore_padding_in_loss: If True, use -100 as padding value for target grids
                                    so loss ignores padding pixels
            cache_samples: If True, pre-generate cached samples for consistent training
                          This allows the model to see the SAME samples multiple epochs
                          which is REQUIRED for learning hard tasks (>100 epochs needed)
            num_cached_samples: Number of samples to pre-generate (default: 32000)
                               With 400 tasks, this is 80 augmented versions per task
            cache_path: Path to save/load cached samples (for persistence across runs)
            cache_load_percent: Percentage of cache to load (1-100, default: 100)
                               Use 10-20% for quick testing without loading full cache
                               E.g., 10% of 50GB = 5GB, loads in ~20s instead of 3min
            stratified_seed: Seed for deterministic stratified sampling (default: 42)
        """
        self.data_path = Path(data_path)
        self.max_size = max_size
        self.augment = augment
        self.color_permutation = color_permutation
        self.color_permutation_prob = color_permutation_prob
        self.translational_augment = translational_augment
        self.curriculum_stage = curriculum_stage
        self.track_augmentation = track_augmentation
        self.ignore_padding_in_loss = ignore_padding_in_loss
        self.cache_samples = cache_samples
        self.num_cached_samples = num_cached_samples
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache_load_percent = min(100.0, max(1.0, cache_load_percent))  # Clamp to 1-100%
        self.max_tasks = max_tasks
        self.stratified_seed = stratified_seed
        
        # Cached sample storage
        self._cached_samples: List[Dict[str, Any]] = []
        
        # Store task_ids BEFORE any filtering for cache validation
        self._all_task_ids: set = set()
        
        # Load tasks
        self.tasks = self._load_tasks()
        
        # Limit tasks if max_tasks specified (for testing)
        # Uses STRATIFIED sampling to ensure representative coverage
        if max_tasks is not None and max_tasks > 0:
            # ARCDataset uses Dicts, so we use stratified_sample_tasks
            self.tasks = stratified_sample_tasks(
                self.tasks,
                n_samples=max_tasks,
                seed=stratified_seed,  # Configurable seed for reproducibility
                verbose=True
            )
        
        # Apply curriculum filtering if enabled
        if curriculum_stage > 0:
            self.tasks = self._filter_by_difficulty(curriculum_stage)
        
        # Store active task_ids for cache filtering
        self._active_task_ids = {t.get('task_id', str(i)) for i, t in enumerate(self.tasks)}
        
        print(f"Loaded {len(self.tasks)} tasks from {data_path}")
        
        # Build cache if enabled
        if cache_samples:
            self._build_cache()
    
    def _load_tasks(self) -> List[Dict]:
        """Load tasks from JSON file or directory."""
        tasks = []
        
        if self.data_path.is_file() and self.data_path.suffix == '.json':
            # Single JSON file (combined format)
            with open(self.data_path, 'r') as f:
                data = json.load(f)
            
            # Handle different JSON formats
            if isinstance(data, dict):
                # Format: {task_id: {train: [...], test: [...]}}
                for task_id, task_data in data.items():
                    tasks.append({
                        'task_id': task_id,
                        'train': task_data.get('train', []),
                        'test': task_data.get('test', [])
                    })
            elif isinstance(data, list):
                # Format: [{task_id: ..., train: [...], test: [...]}]
                for task_data in data:
                    tasks.append(task_data)
        else:
            # Directory with individual JSON files
            json_files = list(self.data_path.glob('*.json'))
            for json_file in json_files:
                try:
                    with open(json_file, 'r') as f:
                        task_data = json.load(f)
                    tasks.append({
                        'task_id': json_file.stem,
                        'train': task_data.get('train', []),
                        'test': task_data.get('test', [])
                    })
                except Exception as e:
                    print(f"Warning: Failed to load {json_file}: {e}")
        
        return tasks
    
    def _build_cache(self):
        """
        Build cache of pre-generated samples for consistent training.
        
        When cache_samples=True, we generate `num_cached_samples` augmented
        versions upfront. This ensures the model sees the SAME samples every
        epoch, which is CRITICAL for learning hard tasks that need >100 epochs.
        
        The cache can optionally be saved to disk for persistence.
        
        AUTOMATIC CHUNKING & PARTIAL LOADING:
        1. If chunked cache exists (same name as pkl + .chunks/), load from chunks
        2. If only monolithic pkl exists, auto-convert to chunks, then load
        3. cache_load_percent controls how many chunks to load
        4. Chunk filenames derive from source pkl name for tracking
        5. VALIDATION: Checks if cache settings match current settings
        """
        import time
        
        if not self.cache_path:
            # No cache path specified, generate in-memory
            self._generate_cache_in_memory()
            return
        
        # Derive chunked cache directory name from pkl filename
        # e.g., rlan_stable_400k_v3.pkl -> rlan_stable_400k_v3.pkl.chunks/
        chunked_cache_dir = Path(str(self.cache_path) + '.chunks')
        load_percent = self.cache_load_percent
        
        # CASE 1: Chunked cache exists - VALIDATE then load
        if chunked_cache_dir.exists() and (chunked_cache_dir / 'meta.pkl').exists():
            # Validate cache before loading
            is_valid, reason = self._validate_cache(chunked_cache_dir)
            if not is_valid:
                print(f"\n{'!'*60}")
                print(f"CACHE INVALIDATED: {reason}")
                print(f"{'!'*60}")
                print(f"  Cache path: {chunked_cache_dir}")
                print(f"  Action: Regenerating cache with current settings...")
                print(f"{'!'*60}\n")
                
                # Delete old cache
                import shutil
                shutil.rmtree(chunked_cache_dir)
                if self.cache_path.exists():
                    self.cache_path.unlink()
                
                # Generate new cache
                self._generate_cache_in_memory()
                if self.cache_path:
                    self._save_cache_to_disk()
                return
            
            return self._load_chunked_cache(chunked_cache_dir, load_percent)
        
        # CASE 2: Monolithic pkl exists but no chunks
        # Old pkl files don't have validation metadata, so we MUST regenerate
        # to ensure cache matches current augmentation settings
        if self.cache_path.exists():
            print(f"\n{'!'*60}")
            print(f"CACHE INVALIDATED: Old monolithic cache without validation metadata")
            print(f"{'!'*60}")
            print(f"  Cache path: {self.cache_path}")
            print(f"  Action: Deleting old cache and regenerating with current settings...")
            print(f"{'!'*60}\n")
            
            # Delete old monolithic cache (no chunks to delete in this case)
            import shutil
            self.cache_path.unlink()
            
            # Generate new cache with validation metadata
            self._generate_cache_in_memory()
            if self.cache_path:
                self._save_cache_to_disk()
            return
        
        # CASE 3: No cache exists - generate from scratch
        self._generate_cache_in_memory()
        
        # Save both monolithic and chunked versions for future runs
        if self.cache_path:
            self._save_cache_to_disk()
    
    def _validate_cache(self, cache_dir: Path) -> tuple:
        """
        Validate that cached data matches current augmentation settings.
        
        Returns:
            (is_valid: bool, reason: str)
            
        CRITICAL (Dec 2025): This prevents using stale caches that were
        generated with different augmentation settings.
        """
        import pickle
        
        CURRENT_CACHE_VERSION = 4  # Must match version in _save_chunked_cache
        
        meta_path = cache_dir / 'meta.pkl'
        if not meta_path.exists():
            return False, "meta.pkl not found"
        
        try:
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
        except Exception as e:
            return False, f"Failed to read meta.pkl: {e}"
        
        # Check cache version
        cache_version = meta.get('cache_version', 0)
        if cache_version < CURRENT_CACHE_VERSION:
            return False, f"Cache version {cache_version} < current {CURRENT_CACHE_VERSION}"
        
        # Check augmentation settings (only if present in meta)
        cached_aug = meta.get('augmentation_settings', {})
        if cached_aug:
            # Check critical settings that affect sample generation
            mismatches = []
            
            if cached_aug.get('augment') != self.augment:
                mismatches.append(f"augment: cached={cached_aug.get('augment')}, current={self.augment}")
            
            if cached_aug.get('color_permutation') != self.color_permutation:
                mismatches.append(f"color_permutation: cached={cached_aug.get('color_permutation')}, current={self.color_permutation}")
            
            if cached_aug.get('color_permutation_prob') != self.color_permutation_prob:
                mismatches.append(f"color_permutation_prob: cached={cached_aug.get('color_permutation_prob')}, current={self.color_permutation_prob}")
            
            if cached_aug.get('translational_augment') != self.translational_augment:
                mismatches.append(f"translational_augment: cached={cached_aug.get('translational_augment')}, current={self.translational_augment}")
            
            if cached_aug.get('max_size') != self.max_size:
                mismatches.append(f"max_size: cached={cached_aug.get('max_size')}, current={self.max_size}")
            
            if mismatches:
                return False, f"Augmentation settings mismatch: {'; '.join(mismatches)}"
        
        # Cache is valid
        print(f"  Cache validated: version={cache_version}, samples={meta.get('total_samples', 'unknown')}")
        return True, "Cache is valid"
    
    def _generate_cache_in_memory(self):
        """Generate cache samples in memory."""
        import time
        
        print(f"Building cache with {self.num_cached_samples} samples...")
        start_time = time.time()
        
        # Calculate samples per task
        num_tasks = len(self.tasks)
        samples_per_task = max(1, self.num_cached_samples // num_tasks)
        
        self._cached_samples = []
        for task_idx in range(num_tasks):
            for aug_idx in range(samples_per_task):
                sample = self._generate_sample(task_idx)
                self._cached_samples.append(sample)
        
        elapsed = time.time() - start_time
        print(f"Cached {len(self._cached_samples)} samples in {elapsed:.1f}s "
              f"({samples_per_task} per task)")
    
    def _save_cache_to_disk(self):
        """Save current cache to disk (both monolithic pkl and chunked format)."""
        import pickle
        
        if not self.cache_path or not self._cached_samples:
            return
            
        print(f"Saving cache to {self.cache_path}...")
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, 'wb') as f:
                pickle.dump(self._cached_samples, f)
            print(f"Monolithic cache saved successfully")
            
            # Also save as chunked cache for fast partial loading
            chunked_dir = Path(str(self.cache_path) + '.chunks')
            print(f"Creating chunked cache at {chunked_dir}...")
            self._save_chunked_cache(self._cached_samples, chunked_dir)
        except Exception as e:
            print(f"Warning: Failed to save cache: {e}")
    
    def _load_chunked_cache(self, cache_dir: Path, load_percent: float) -> None:
        """Load samples from chunked cache format (supports true partial loading).
        
        CRITICAL (Dec 2025): If max_tasks is set, we filter cached samples
        to only include samples from active tasks. This ensures max_tasks
        actually limits what the model sees, even when using a pre-built cache.
        """
        import pickle
        import time
        
        start_time = time.time()
        
        # Read metadata
        meta_path = cache_dir / 'meta.pkl'
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        
        total_samples = meta['total_samples']
        chunk_size = meta['chunk_size']
        num_chunks = meta['num_chunks']
        
        # Check if we need to filter by task_id (max_tasks is active)
        filter_by_task = hasattr(self, '_active_task_ids') and self.max_tasks is not None
        
        # DEBUG: Always print filter status
        print(f"  DEBUG: max_tasks={self.max_tasks}, has _active_task_ids={hasattr(self, '_active_task_ids')}")
        print(f"  DEBUG: filter_by_task={filter_by_task}")
        if hasattr(self, '_active_task_ids'):
            print(f"  DEBUG: len(_active_task_ids)={len(self._active_task_ids)}")
            # Show a few sample IDs
            sample_ids = list(self._active_task_ids)[:5]
            print(f"  DEBUG: sample _active_task_ids={sample_ids}")
        
        # Calculate how many samples to load
        num_to_load = max(1, int(total_samples * load_percent / 100))
        chunks_to_load = (num_to_load + chunk_size - 1) // chunk_size  # Ceiling division
        
        if load_percent < 100:
            print(f"Loading {load_percent:.0f}% of chunked cache ({chunks_to_load}/{num_chunks} chunks)...")
        else:
            print(f"Loading chunked cache ({num_chunks} chunks)...")
        
        if filter_by_task:
            print(f"  Filtering to {len(self._active_task_ids)} active tasks (max_tasks={self.max_tasks})")
        
        self._cached_samples = []
        samples_before_filter = 0
        first_chunk_debug = True  # Show debug for first chunk only
        
        for chunk_idx in range(min(chunks_to_load, num_chunks)):
            chunk_path = cache_dir / f'chunk_{chunk_idx:04d}.pkl'
            with open(chunk_path, 'rb') as f:
                chunk_samples = pickle.load(f)
            
            # DEBUG: Show task_ids from first chunk
            if first_chunk_debug and chunk_samples:
                chunk_task_ids = {s.get('task_id') for s in chunk_samples[:20]}
                print(f"  DEBUG: First chunk sample task_ids: {list(chunk_task_ids)[:5]}")
                first_chunk_debug = False
            
            if filter_by_task:
                # Filter samples to only include active tasks
                samples_before_filter += len(chunk_samples)
                chunk_samples = [s for s in chunk_samples if s.get('task_id') in self._active_task_ids]
            
            self._cached_samples.extend(chunk_samples)
        
        # Trim to exact count (after filtering)
        if len(self._cached_samples) > num_to_load:
            self._cached_samples = self._cached_samples[:num_to_load]
        
        elapsed = time.time() - start_time
        
        if filter_by_task:
            print(f"Loaded {len(self._cached_samples):,} samples from chunked cache in {elapsed:.1f}s")
            print(f"  (filtered from {samples_before_filter:,} → {len(self._cached_samples):,} samples)")
        else:
            print(f"Loaded {len(self._cached_samples):,} samples from chunked cache in {elapsed:.1f}s")
    
    def _save_chunked_cache(self, samples: list, cache_dir: Path, chunk_size: int = 10000) -> None:
        """
        Save samples as chunked cache for fast partial loading.
        
        CRITICAL: Samples are SHUFFLED before chunking to ensure each chunk
        contains a representative mix of all tasks/difficulties. This allows
        partial loading (e.g., 4%) to still cover all task types.
        """
        import pickle
        import random
        
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        num_samples = len(samples)
        num_chunks = (num_samples + chunk_size - 1) // chunk_size
        
        # CRITICAL: Shuffle samples so each chunk has representative task mix
        # Without this, chunk_0 would only have tasks 0-9 (sequential order)
        # With shuffling, chunk_0 has random samples from ALL 400 tasks
        print(f"Shuffling {num_samples:,} samples for representative chunk distribution...")
        indices = list(range(num_samples))
        random.seed(42)  # Fixed seed for reproducibility across runs
        random.shuffle(indices)
        
        # Save chunks with shuffled order
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, num_samples)
            chunk_indices = indices[start_idx:end_idx]
            chunk_samples = [samples[i] for i in chunk_indices]
            
            chunk_path = cache_dir / f'chunk_{chunk_idx:04d}.pkl'
            with open(chunk_path, 'wb') as f:
                pickle.dump(chunk_samples, f)
        
        # Save metadata (includes shuffle info and augmentation settings for validation)
        meta = {
            'total_samples': num_samples,
            'chunk_size': chunk_size,
            'num_chunks': num_chunks,
            'shuffled': True,  # Flag indicating representative distribution
            'shuffle_seed': 42,
            # =====================================================
            # CACHE VALIDATION FIELDS (Dec 2025)
            # =====================================================
            # These fields allow automatic detection of stale caches.
            # If current settings don't match cached settings, the cache
            # is invalid and should be regenerated.
            # =====================================================
            'cache_version': 4,  # Increment when cache format changes
            'augmentation_settings': {
                'augment': getattr(self, 'augment', True),
                'color_permutation': getattr(self, 'color_permutation', False),
                'color_permutation_prob': getattr(self, 'color_permutation_prob', 1.0),
                'translational_augment': getattr(self, 'translational_augment', True),
                'max_size': getattr(self, 'max_size', 30),
            },
            'num_tasks': len(getattr(self, 'tasks', [])),
            'data_path': str(getattr(self, 'data_path', '')),
        }
        meta_path = cache_dir / 'meta.pkl'
        with open(meta_path, 'wb') as f:
            pickle.dump(meta, f)
        
        print(f"Chunked cache saved: {num_chunks} chunks of {chunk_size} samples each (SHUFFLED for representative sampling)")
        print(f"  Cache version: {meta['cache_version']}, Tasks: {meta['num_tasks']}")

    def _generate_sample(self, task_idx: int) -> Dict[str, Any]:
        """Generate a single sample with current augmentation settings."""
        task = self.tasks[task_idx]
        
        # Parse train pairs
        train_inputs = []
        train_outputs = []
        for pair in task['train']:
            inp = np.array(pair['input'], dtype=np.int64)
            out = np.array(pair['output'], dtype=np.int64)
            train_inputs.append(inp)
            train_outputs.append(out)
        
        # Parse test pairs (use first test case)
        test_pair = task['test'][0] if task['test'] else task['train'][0]
        test_input = np.array(test_pair['input'], dtype=np.int64)
        test_output = np.array(test_pair.get('output', test_pair['input']), dtype=np.int64)
        
        # Initialize augmentation tracking
        # CRITICAL: Store actual permutation arrays, not just booleans!
        # This enables inverse transforms during TRM-style evaluation.
        aug_info = {
            'dihedral_id': 0,
            'color_perm': None,  # Store actual permutation array (not just boolean!)
            'translational_offset': (0, 0),
        }
        
        # CRITICAL ORDER: TRM applies color permutation FIRST, then dihedral
        # Forward: color_perm → dihedral
        # Inverse: inverse_dihedral → inverse_color
        
        # Step 1: Apply color permutation FIRST (like TRM)
        if self.color_permutation and random.random() < self.color_permutation_prob:
            train_inputs, train_outputs, test_input, test_output, color_perm = self._augment_color(
                train_inputs, train_outputs, test_input, test_output
            )
            aug_info['color_perm'] = color_perm  # Store actual array for inverse!
        
        # Step 2: Apply dihedral augmentation SECOND (like TRM)
        if self.augment:
            train_inputs, train_outputs, test_input, test_output, dihedral_id = self._augment_dihedral_tracked(
                train_inputs, train_outputs, test_input, test_output
            )
            aug_info['dihedral_id'] = dihedral_id
        
        # Compute translational offset
        offset = None
        if self.translational_augment:
            all_grids = train_inputs + train_outputs + [test_input, test_output]
            max_h = max(g.shape[0] for g in all_grids)
            max_w = max(g.shape[1] for g in all_grids)
            max_offset_r = self.max_size - max_h
            max_offset_c = self.max_size - max_w
            if max_offset_r > 0 and max_offset_c > 0:
                offset = (random.randint(0, max_offset_r), random.randint(0, max_offset_c))
                aug_info['translational_offset'] = offset
        
        # Pad grids
        train_inputs_padded = [self._pad_grid(g, offset, is_target=False) for g in train_inputs]
        # CRITICAL FIX: Use is_target=False for train outputs too!
        # Train outputs go to ContextEncoder (not loss), so they need PAD_COLOR=10
        # to distinguish from black (0). Only test_output should use -100 for loss masking.
        train_outputs_padded = [self._pad_grid(g, offset, is_target=False) for g in train_outputs]
        test_input_padded = self._pad_grid(test_input, offset, is_target=False)
        test_output_padded = self._pad_grid(test_output, offset, is_target=True)
        
        # Compute ORIGINAL max grid size BEFORE padding for bucketed batching
        # This allows BucketedBatchSampler to group similar-sized tasks together
        all_original_grids = train_inputs + train_outputs + [test_input, test_output]
        original_max_size = max(max(g.shape[0], g.shape[1]) for g in all_original_grids)
        
        result = {
            'task_id': task.get('task_id', str(task_idx)),
            'input_grids': [torch.from_numpy(g) for g in train_inputs_padded],
            'output_grids': [torch.from_numpy(g) for g in train_outputs_padded],
            'test_input': torch.from_numpy(test_input_padded),
            'test_output': torch.from_numpy(test_output_padded),
            'num_train_pairs': len(train_inputs),
            'transform_family': 0,
            'original_max_size': original_max_size,  # For bucketed batching
        }
        
        if self.track_augmentation:
            result['aug_info'] = aug_info
        
        return result
    
    def __len__(self) -> int:
        if self.cache_samples and self._cached_samples:
            return len(self._cached_samples)
        return len(self.tasks)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample.
        
        If cache_samples=True, returns a pre-generated cached sample.
        Otherwise, generates fresh random augmentation on-the-fly.
        """
        # Return cached sample if available
        if self.cache_samples and self._cached_samples:
            return self._cached_samples[idx]
        
        # Generate on-the-fly (original behavior)
        task = self.tasks[idx]
        
        # Parse train pairs
        train_inputs = []
        train_outputs = []
        for pair in task['train']:
            inp = np.array(pair['input'], dtype=np.int64)
            out = np.array(pair['output'], dtype=np.int64)
            train_inputs.append(inp)
            train_outputs.append(out)
        
        # Parse test pairs (use first test case)
        test_pair = task['test'][0] if task['test'] else task['train'][0]
        test_input = np.array(test_pair['input'], dtype=np.int64)
        test_output = np.array(test_pair.get('output', test_pair['input']), dtype=np.int64)
        
        # Initialize augmentation tracking
        # CRITICAL: Store actual permutation arrays, not just booleans!
        # This enables inverse transforms during TRM-style evaluation.
        aug_info = {
            'dihedral_id': 0,  # Identity
            'color_perm': None,  # Store actual permutation array (not just boolean!)
            'translational_offset': (0, 0),
        }
        
        # CRITICAL ORDER: TRM applies color permutation FIRST, then dihedral
        # Forward: color_perm → dihedral
        # Inverse: inverse_dihedral → inverse_color
        
        # Step 1: Apply color permutation FIRST (like TRM)
        if self.color_permutation and random.random() < self.color_permutation_prob:
            train_inputs, train_outputs, test_input, test_output, color_perm = self._augment_color(
                train_inputs, train_outputs, test_input, test_output
            )
            aug_info['color_perm'] = color_perm  # Store actual array for inverse!
        
        # Step 2: Apply dihedral augmentation SECOND (like TRM)
        if self.augment:
            train_inputs, train_outputs, test_input, test_output, dihedral_id = self._augment_dihedral_tracked(
                train_inputs, train_outputs, test_input, test_output
            )
            aug_info['dihedral_id'] = dihedral_id
        
        # Compute random translational offset (shared across all grids in this sample)
        # TRM-style: random position within 30×30 canvas
        offset = None
        if self.translational_augment:
            # Find max grid dimensions in this task to determine valid offset range
            all_grids = train_inputs + train_outputs + [test_input, test_output]
            max_h = max(g.shape[0] for g in all_grids)
            max_w = max(g.shape[1] for g in all_grids)
            
            # Random offset that keeps all grids within bounds
            max_offset_r = self.max_size - max_h
            max_offset_c = self.max_size - max_w
            if max_offset_r > 0 and max_offset_c > 0:
                offset = (random.randint(0, max_offset_r), random.randint(0, max_offset_c))
                aug_info['translational_offset'] = offset
        
        # Pad grids (with optional translational offset)
        # CRITICAL FIX: Use is_target=False for train outputs too!
        # Train outputs go to ContextEncoder (not loss), so they need PAD_COLOR=10
        # to distinguish from black (0). Only test_output should use -100 for loss masking.
        train_inputs_padded = [self._pad_grid(g, offset, is_target=False) for g in train_inputs]
        train_outputs_padded = [self._pad_grid(g, offset, is_target=False) for g in train_outputs]
        test_input_padded = self._pad_grid(test_input, offset, is_target=False)
        test_output_padded = self._pad_grid(test_output, offset, is_target=True)
        
        result = {
            'task_id': task.get('task_id', str(idx)),
            'input_grids': [torch.from_numpy(g) for g in train_inputs_padded],
            'output_grids': [torch.from_numpy(g) for g in train_outputs_padded],
            'test_input': torch.from_numpy(test_input_padded),
            'test_output': torch.from_numpy(test_output_padded),
            'num_train_pairs': len(train_inputs),
            'transform_family': 0,  # Not used in RLAN
        }
        
        # Add augmentation tracking info if enabled
        if self.track_augmentation:
            result['aug_info'] = aug_info
        
        return result
    
    def _pad_grid(self, grid: np.ndarray, offset: Tuple[int, int] = None, is_target: bool = False) -> np.ndarray:
        """
        Pad grid to max_size x max_size with optional translational offset.
        
        RLAN uses 2D spatial structure (B, H, W, D) unlike TRM which flattens
        grids to 1D sequences. Therefore, boundary markers are not needed.
        Colors 0-9 are used directly (10 classes total).
        
        Args:
            grid: Input grid of shape (H, W) with values 0-9
            offset: Optional (row_offset, col_offset) for translational augmentation
            is_target: If True and ignore_padding_in_loss is enabled, use -100 as padding
                       so that loss function ignores padding pixels
        
        Returns:
            Padded grid of shape (max_size, max_size)
            - Content: values 0-9 (colors)
            - Padding: -100 (for targets) or 0 (for inputs)
        """
        h, w = grid.shape
        if h >= self.max_size and w >= self.max_size:
            result = grid[:self.max_size, :self.max_size].copy()
            return result
        
        # Determine padding value:
        # - Targets: use -100 (PADDING_IGNORE_VALUE) so loss ignores padding
        # - Inputs: use 10 (PAD_COLOR) so model distinguishes padding from black (0)
        if is_target and self.ignore_padding_in_loss:
            pad_value = self.PADDING_IGNORE_VALUE
        else:
            pad_value = self.PAD_COLOR  # Use 10 for inputs to distinguish from black
        
        # Use int64 consistently for all grids (needed for -100 values and PyTorch compatibility)
        padded = np.full((self.max_size, self.max_size), pad_value, dtype=np.int64)
        
        # Calculate offset
        r_off, c_off = offset if offset is not None else (0, 0)
        
        # Copy grid content to padded canvas at offset position
        h_copy = min(h, self.max_size - r_off)
        w_copy = min(w, self.max_size - c_off)
        padded[r_off:r_off+h_copy, c_off:c_off+w_copy] = grid[:h_copy, :w_copy]
        
        return padded
    
    def _augment_dihedral_tracked(
        self,
        train_inputs: List[np.ndarray],
        train_outputs: List[np.ndarray],
        test_input: np.ndarray,
        test_output: np.ndarray
    ) -> Tuple:
        """
        Apply random dihedral augmentation (8 transforms from D4 group) with tracking.
        
        Returns the augmented grids PLUS the dihedral_id for logging.
        
        Dihedral Transform IDs (matching TRM exactly):
            0: identity
            1: rotate 90° CCW
            2: rotate 180°
            3: rotate 270° CCW (90° CW)
            4: horizontal flip (left-right)
            5: vertical flip (up-down)
            6: transpose (main diagonal)
            7: anti-transpose (anti-diagonal)
        """
        # Random dihedral transform (0-7) - UNIFORM distribution
        dihedral_id = random.randint(0, 7)
        
        def transform(g):
            if dihedral_id == 0:
                return g.copy()
            elif dihedral_id == 1:
                return np.rot90(g, k=1).copy()
            elif dihedral_id == 2:
                return np.rot90(g, k=2).copy()
            elif dihedral_id == 3:
                return np.rot90(g, k=3).copy()
            elif dihedral_id == 4:
                return np.fliplr(g).copy()
            elif dihedral_id == 5:
                return np.flipud(g).copy()
            elif dihedral_id == 6:
                return g.T.copy()  # transpose
            else:  # 7
                return np.fliplr(np.rot90(g, k=1)).copy()  # anti-transpose
        
        aug_inputs = [transform(g) for g in train_inputs]
        aug_outputs = [transform(g) for g in train_outputs]
        aug_test_in = transform(test_input)
        aug_test_out = transform(test_output)
        
        return aug_inputs, aug_outputs, aug_test_in, aug_test_out, dihedral_id
    
    def _augment_dihedral(
        self,
        train_inputs: List[np.ndarray],
        train_outputs: List[np.ndarray],
        test_input: np.ndarray,
        test_output: np.ndarray
    ) -> Tuple:
        """Apply random dihedral augmentation (8 transforms from D4 group)."""
        # Random dihedral transform (0-7)
        dihedral_id = random.randint(0, 7)
        
        def transform(g):
            if dihedral_id == 0:
                return g.copy()
            elif dihedral_id == 1:
                return np.rot90(g, k=1).copy()
            elif dihedral_id == 2:
                return np.rot90(g, k=2).copy()
            elif dihedral_id == 3:
                return np.rot90(g, k=3).copy()
            elif dihedral_id == 4:
                return np.fliplr(g).copy()
            elif dihedral_id == 5:
                return np.flipud(g).copy()
            elif dihedral_id == 6:
                return g.T.copy()  # transpose
            else:  # 7
                return np.fliplr(np.rot90(g, k=1)).copy()  # anti-transpose
        
        aug_inputs = [transform(g) for g in train_inputs]
        aug_outputs = [transform(g) for g in train_outputs]
        aug_test_in = transform(test_input)
        aug_test_out = transform(test_output)
        
        return aug_inputs, aug_outputs, aug_test_in, aug_test_out
    
    def _augment_color(
        self,
        train_inputs: List[np.ndarray],
        train_outputs: List[np.ndarray],
        test_input: np.ndarray,
        test_output: np.ndarray
    ) -> Tuple:
        """
        Apply random color permutation (9! = 362,880 possibilities).
        
        Permutes colors 1-9 while keeping 0 (black/background) fixed.
        This is the SAME augmentation used by TRM.
        
        Returns:
            Tuple of (aug_inputs, aug_outputs, aug_test_in, aug_test_out, perm)
            where perm is the color permutation array for inverse transform.
        """
        # Generate random permutation of colors 1-9
        perm = np.arange(10, dtype=np.int64)  # [0, 1, 2, ..., 9]
        perm[1:] = np.random.permutation(9) + 1  # Shuffle 1-9, keep 0 fixed
        
        def apply_perm(g):
            return perm[g].astype(g.dtype)
        
        aug_inputs = [apply_perm(g) for g in train_inputs]
        aug_outputs = [apply_perm(g) for g in train_outputs]
        aug_test_in = apply_perm(test_input)
        aug_test_out = apply_perm(test_output)
        
        # Return permutation array for inverse transform during eval
        return aug_inputs, aug_outputs, aug_test_in, aug_test_out, perm
    
    # Keep old method name for backward compatibility
    def _augment(
        self,
        train_inputs: List[np.ndarray],
        train_outputs: List[np.ndarray],
        test_input: np.ndarray,
        test_output: np.ndarray
    ) -> Tuple:
        """Backward compatibility: alias for _augment_dihedral."""
        return self._augment_dihedral(train_inputs, train_outputs, test_input, test_output)

    def _filter_by_difficulty(self, stage: int) -> List[Dict]:
        """
        Filter tasks by difficulty for curriculum learning.
        
        Uses PERCENTILE-based filtering to ensure enough tasks per stage:
        - Stage 1 (easy): 70% of tasks (smallest grids by max dimension)
        - Stage 2 (+medium): 90% of tasks
        - Stage 3 (+hard): All tasks (100%)
        
        This approach ensures:
        - At least 280/400 tasks in Stage 1 (7+ batches with batch_size=75)
        - At least 360/400 tasks in Stage 2
        - All 400 tasks in Stage 3
        
        Args:
            stage: 1=easy (70%), 2=easy+medium (90%), 3=all (100%)
        
        Returns:
            Filtered list of tasks
        """
        if stage >= 3 or stage == 0:
            return self.tasks  # All tasks
        
        def get_task_complexity(task: Dict) -> float:
            """
            Compute complexity score for a task.
            
            Combines max grid dimension and inverse of example count.
            Higher score = harder task.
            """
            train_pairs = task.get('train', [])
            if not train_pairs:
                return 100.0  # Unknown = very hard
            
            # Find max grid dimension across all train pairs
            max_size = 0
            for pair in train_pairs:
                inp = np.array(pair['input'])
                out = np.array(pair['output'])
                max_size = max(max_size, inp.shape[0], inp.shape[1])
                max_size = max(max_size, out.shape[0], out.shape[1])
            
            num_pairs = len(train_pairs)
            
            # Complexity = grid_size + penalty for few examples
            # More examples = easier (inverse relationship)
            example_penalty = max(0, 5 - num_pairs)  # 0-4 penalty
            return max_size + example_penalty
        
        # Compute complexity for all tasks
        task_complexities = [(t, get_task_complexity(t)) for t in self.tasks]
        
        # Sort by complexity (easiest first)
        task_complexities.sort(key=lambda x: x[1])
        
        # Percentile thresholds
        # Stage 1: 70% of tasks, Stage 2: 90% of tasks
        percentiles = {1: 0.70, 2: 0.90}
        threshold_pct = percentiles.get(stage, 1.0)
        
        # Select top N% easiest tasks
        num_to_select = int(len(self.tasks) * threshold_pct)
        selected = [t for t, _ in task_complexities[:num_to_select]]
        
        return selected


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


def stratified_sample_arc_tasks(
    tasks: List['ARCTask'],
    n_samples: int,
    seed: int = 42,
    verbose: bool = True
) -> List['ARCTask']:
    """
    Stratified sampling for ARCTask objects (used by SCIARCDataset).
    
    Same stratification logic as stratified_sample_tasks but for ARCTask dataclass.
    Ensures representative coverage for meta-learning validation.
    
    Stratifies by:
    1. Grid size bucket (small/medium/large) - preserves bucketed batching benefits
    2. Number of train pairs (1-2, 3-4, 5+) - covers few-shot to many-shot
    3. Size change type (same_size vs size_change) - structural diversity
    
    Args:
        tasks: List of ARCTask objects
        n_samples: Number of tasks to sample
        seed: Random seed for deterministic sampling
        verbose: Print statistics
        
    Returns:
        Stratified subset of ARCTask objects
    """
    if n_samples >= len(tasks):
        return tasks
    
    rng = random.Random(seed)
    
    def get_max_grid_size(task: 'ARCTask') -> int:
        max_dim = 0
        for inp, out in task.train_pairs:
            max_dim = max(max_dim, inp.shape[0], inp.shape[1], out.shape[0], out.shape[1])
        for inp, out in task.test_pairs:
            max_dim = max(max_dim, inp.shape[0], inp.shape[1])
        return max_dim
    
    def get_grid_size_bucket(size: int) -> str:
        if size <= 10:
            return "small"
        elif size <= 20:
            return "medium"
        else:
            return "large"
    
    def get_num_pairs_bucket(task: 'ARCTask') -> str:
        n = len(task.train_pairs)
        if n <= 2:
            return "few_shot"
        elif n <= 4:
            return "medium_shot"
        else:
            return "many_shot"
    
    def get_size_change_type(task: 'ARCTask') -> str:
        for inp, out in task.train_pairs:
            if inp.shape != out.shape:
                return "size_change"
        return "same_size"
    
    # Create strata
    strata: Dict[str, List['ARCTask']] = {}
    for task in tasks:
        grid_size = get_max_grid_size(task)
        key = f"{get_grid_size_bucket(grid_size)}_{get_num_pairs_bucket(task)}_{get_size_change_type(task)}"
        if key not in strata:
            strata[key] = []
        strata[key].append(task)
    
    # Sample proportionally from each stratum
    sampled = []
    remaining = n_samples
    strata_list = list(strata.items())
    rng.shuffle(strata_list)
    
    for strata_key, strata_tasks in strata_list:
        stratum_proportion = len(strata_tasks) / len(tasks)
        stratum_allocation = max(1, int(n_samples * stratum_proportion))
        stratum_allocation = min(stratum_allocation, len(strata_tasks), remaining)
        
        rng.shuffle(strata_tasks)
        sampled.extend(strata_tasks[:stratum_allocation])
        remaining -= stratum_allocation
        
        if remaining <= 0:
            break
    
    # Fill remaining slots
    if remaining > 0:
        used_ids = {t.task_id for t in sampled}
        unused = [t for t in tasks if t.task_id not in used_ids]
        rng.shuffle(unused)
        sampled.extend(unused[:remaining])
    
    rng.shuffle(sampled)
    
    if verbose:
        sampled_strata = {}
        for task in sampled:
            grid_size = get_max_grid_size(task)
            key = f"{get_grid_size_bucket(grid_size)}_{get_num_pairs_bucket(task)}_{get_size_change_type(task)}"
            sampled_strata[key] = sampled_strata.get(key, 0) + 1
        
        print(f"  Stratified sampling: {len(sampled)}/{len(tasks)} tasks ({100*len(sampled)/len(tasks):.1f}%)")
        print(f"  Strata covered: {len(sampled_strata)}/{len(strata)} ({100*len(sampled_strata)/len(strata):.1f}%)")
    
    return sampled


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
    - Optional in-memory caching for faster training
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
        cache_samples: bool = False,  # Enable in-memory caching
        cache_augmentations: int = 8,  # Number of augmented versions to pre-generate per task
        use_augment_family: bool = True,  # DEPRECATED: use scl_family_mode instead
        scl_family_mode: str = "task",  # "task" (augmentation invariance), "augment" (dihedral), "inferred"
        expand_test_pairs: bool = False,  # NEW: If True, expand dataset to cover all test pairs per task
        color_permutation: bool = True,  # Enable color permutation augmentation
        color_permutation_prob: float = 0.5,  # Probability of applying color permutation
        translational_augment: bool = False,  # Enable translational augmentation
        max_tasks: int = None,  # Limit number of tasks (uses stratified sampling for representativeness)
        stratified_seed: int = 42,  # Seed for deterministic stratified sampling
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
            cache_samples: If True, cache processed samples in memory
            cache_augmentations: Number of pre-generated augmentations per task (only used when cache_samples=True and augment=True)
            use_augment_family: DEPRECATED - use scl_family_mode instead
            scl_family_mode: How to assign transform_family for SCL:
                - "task": Use task_id hash (augmentation invariance - RECOMMENDED)
                  All augmented versions of the same task become positive pairs.
                  This teaches the model that rotated/flipped versions are the same task.
                - "augment": Use dihedral augmentation type (0-7)
                  Samples with same augmentation become positive pairs.
                  This teaches "rotation detection" not "task understanding".
                - "inferred": Use infer_transform_from_grids result
                  Only works if tasks have detectable simple transforms.
            expand_test_pairs: If True, expand dataset indexing to (task, test_idx) so that
                evaluation can deterministically cover all official test inputs. Default False
                for backward compatibility (random test pair selection for training).
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.augment = augment
        self.max_grid_size = max_grid_size
        self.num_colors = num_colors
        self.transform_fn = transform_fn
        self.curriculum_stage = curriculum_stage
        self.cache_samples = cache_samples
        self.cache_augmentations = cache_augmentations
        self.color_permutation = color_permutation
        self.color_permutation_prob = color_permutation_prob
        self.translational_augment = translational_augment
        self.max_tasks = max_tasks
        self.stratified_seed = stratified_seed
        
        # Auto-enable expand_test_pairs for evaluation to ensure deterministic coverage
        # of all official test inputs (some ARC tasks have multiple test cases)
        if split == "evaluation" and not expand_test_pairs:
            expand_test_pairs = True
        self.expand_test_pairs = expand_test_pairs
        
        # Handle scl_family_mode with backward compatibility
        # DEPRECATED: use_augment_family is ignored if scl_family_mode is explicitly set
        self.scl_family_mode = scl_family_mode
        # Legacy support: if using old parameter explicitly
        if not use_augment_family and scl_family_mode == "task":
            # Old code that set use_augment_family=False wanted inferred behavior
            self.scl_family_mode = "inferred"
        
        # Cache storage
        self._cache: Dict[int, Any] = {}
        self._augmented_cache: List[Dict[str, Any]] = []  # Pre-generated augmentations
        
        # Load tasks
        self.tasks = self._load_tasks()
        
        # Optionally add RE-ARC data
        if include_rearc and rearc_dir:
            rearc_tasks = self._load_rearc(rearc_dir)
            self.tasks.extend(rearc_tasks)
        
        # Apply stratified sampling if max_tasks specified
        # This ensures representative coverage for meta-learning validation
        if max_tasks is not None and max_tasks > 0 and len(self.tasks) > max_tasks:
            # SCIARCDataset uses ARCTask objects, so we use stratified_sample_arc_tasks
            self.tasks = stratified_sample_arc_tasks(
                self.tasks,
                n_samples=max_tasks,
                seed=stratified_seed,
                verbose=True
            )
        
        # Apply curriculum filtering
        if curriculum_stage > 0:
            self.tasks = self._filter_by_difficulty(curriculum_stage)
        
        # Build expanded index for deterministic evaluation if requested
        self._expanded_index: List[Tuple[int, int]] = []  # (task_idx, test_idx)
        if expand_test_pairs:
            for task_idx, task in enumerate(self.tasks):
                num_tests = len(task.test_pairs) if task.test_pairs else 1
                for test_idx in range(num_tests):
                    self._expanded_index.append((task_idx, test_idx))
            print(f"Expanded to {len(self._expanded_index)} (task, test_idx) pairs from {len(self.tasks)} tasks")
        else:
            print(f"Loaded {len(self.tasks)} tasks from {split}")
        
        # Pre-populate cache if enabled
        if cache_samples:
            self._build_cache()
    
    def _load_tasks(self) -> List[ARCTask]:
        """Load tasks from JSON files."""
        tasks = []
        
        # Standard ARC directory structure
        split_dir = self.data_dir / self.split
        if not split_dir.exists():
            # Try alternate structure (ARC-AGI)
            split_dir = self.data_dir / f"{self.split}_challenges"
        
        if not split_dir.exists():
            # FAIL CLOSED: Do not silently fall back to data_dir
            # This prevents accidental train/eval data leakage
            raise FileNotFoundError(
                f"Split directory '{self.split}' not found in {self.data_dir}. "
                f"Expected one of: {self.data_dir / self.split}, {self.data_dir / f'{self.split}_challenges'}. "
                f"Please ensure your data_dir points to the parent containing 'training/' and 'evaluation/' subdirectories, "
                f"not directly to a split directory."
            )
        
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
    
    def _build_cache(self):
        """Pre-build cache of processed samples for faster training."""
        import time
        start_time = time.time()
        
        if self.augment:
            # Pre-generate multiple augmented versions of each task
            print(f"Building cache with {self.cache_augmentations} augmentations per task...")
            for task_idx in range(len(self.tasks)):
                for aug_idx in range(self.cache_augmentations):
                    sample = self._process_task(task_idx, apply_augment=True)
                    self._augmented_cache.append(sample)
            print(f"Cached {len(self._augmented_cache)} augmented samples in {time.time() - start_time:.1f}s")
        else:
            # Cache processed samples without augmentation (for validation)
            print(f"Building cache for {len(self.tasks)} tasks...")
            for idx in range(len(self.tasks)):
                self._cache[idx] = self._process_task(idx, apply_augment=False)
            print(f"Cached {len(self._cache)} samples in {time.time() - start_time:.1f}s")
    
    def __len__(self) -> int:
        if self.cache_samples and self.augment and self._augmented_cache:
            return len(self._augmented_cache)
        if self.expand_test_pairs and self._expanded_index:
            return len(self._expanded_index)
        return len(self.tasks)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Return from cache if available
        if self.cache_samples:
            if self.augment and self._augmented_cache:
                # Return pre-generated augmented sample
                return self._augmented_cache[idx]
            elif idx in self._cache:
                # Return cached non-augmented sample
                return self._cache[idx]
        
        # Handle expanded test pair indexing for deterministic evaluation
        if self.expand_test_pairs and self._expanded_index:
            task_idx, test_idx = self._expanded_index[idx]
            return self._process_task(task_idx, apply_augment=self.augment, fixed_test_idx=test_idx)
        
        # Process on-the-fly (original behavior)
        return self._process_task(idx, apply_augment=self.augment)
    
    def _process_task(
        self,
        idx: int,
        apply_augment: bool = False,
        fixed_test_idx: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Process a single task into a training sample.
        
        Args:
            idx: Task index (always indexes into self.tasks)
            apply_augment: Whether to apply augmentation
            fixed_test_idx: If provided, use this specific test pair index instead of random.
                            Used when expand_test_pairs=True for deterministic evaluation.
            
        Returns:
            Dictionary with processed sample tensors
        """
        task = self.tasks[idx]
        
        # Create per-sample RNG for deterministic test pair selection
        sample_rng = random.Random(idx)
        
        # Select a test pair (deterministic if fixed_test_idx provided, else per-sample random)
        if task.test_pairs:
            if fixed_test_idx is not None:
                test_idx = fixed_test_idx
            else:
                test_idx = sample_rng.randint(0, len(task.test_pairs) - 1)
            test_input, test_output = task.test_pairs[test_idx]
        else:
            # If no test pairs, use last train pair
            test_idx = 0
            test_input, test_output = task.train_pairs[-1]
        
        # Collect all grids
        input_grids = [pair[0] for pair in task.train_pairs]
        output_grids = [pair[1] for pair in task.train_pairs]
        
        # Default to task's inferred transform family
        transform_family = task.transform_family
        
        # Apply augmentation if requested
        if apply_augment:
            input_grids, output_grids, test_input, test_output, augment_info = self._augment(
                input_grids, output_grids, test_input, test_output, sample_idx=idx
            )
            
            # Determine transform_family based on scl_family_mode
            if self.scl_family_mode == "augment":
                # Use dihedral augmentation type (0-7)
                # Positive pairs: different tasks with same augmentation
                transform_family = augment_info['dihedral_id']
            elif self.scl_family_mode == "task":
                # Use task_id hash for augmentation invariance (RECOMMENDED)
                # Positive pairs: same task with different augmentations
                # This teaches the model that rotated/flipped versions are the same task
                # Hash to a reasonable number of families to ensure batch diversity
                transform_family = hash(task.task_id) % 400  # 400 tasks = 400 families
            elif self.scl_family_mode == "inferred":
                # Use the task's inferred transform family (from infer_transform_from_grids)
                # Only useful if tasks have detectable simple transforms
                transform_family = task.transform_family
            else:
                # Default to task mode
                transform_family = hash(task.task_id) % 400
        
        # Custom transform
        if self.transform_fn:
            return self.transform_fn({
                'task_id': task.task_id,
                'input_grids': input_grids,
                'output_grids': output_grids,
                'test_input': test_input,
                'test_output': test_output,
                'transform_family': transform_family,
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
            'transform_family': transform_family,
            'num_train_pairs': len(task.train_pairs),
            'test_idx': test_idx,  # Track which test pair was used (for deterministic eval)
        }
    
    def _augment(
        self,
        input_grids: List[np.ndarray],
        output_grids: List[np.ndarray],
        test_input: np.ndarray,
        test_output: np.ndarray,
        sample_idx: int = 0,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Apply data augmentation matching TRM exactly.
        
        Augmentations (applied consistently to all grids in a task):
        1. Dihedral transforms (8 total - full D4 group)
        2. Color permutation (9! for colors 1-9, keeping 0 fixed)
        3. Translational augmentation (optional)
        
        CRITICAL: All augmentations match TRM's dataset/build_arc_dataset.py
        
        Args:
            sample_idx: Sample index used to create per-sample deterministic RNG.
                        Combined with global seed for reproducibility.
        
        Returns:
            Tuple of (aug_inputs, aug_outputs, aug_test_in, aug_test_out, augment_info)
            where augment_info contains {'dihedral_id': int, 'color_permuted': bool}
            for use in SCL transform_family assignment.
        """
        # Create per-sample RNG for deterministic augmentation
        # This ensures same augmentation for same (seed, sample_idx) pair
        sample_rng = random.Random(sample_idx)
        
        # Dihedral transform (0-7)
        dihedral_id = sample_rng.randint(0, 7)
        
        # Color permutation (keep 0 fixed, permute 1-9)
        # Only apply if color_permutation is enabled AND probability check passes
        do_color_perm = self.color_permutation and sample_rng.random() < self.color_permutation_prob
        if do_color_perm:
            # TRM style: permute colors 1-9, keep 0 (black/background) fixed
            color_perm = list(range(1, self.num_colors))  # [1, 2, ..., 9]
            sample_rng.shuffle(color_perm)
            # color_map: old_color -> new_color
            color_map = {0: 0}  # 0 stays 0
            for i, new_c in enumerate(color_perm):
                color_map[i + 1] = new_c
        else:
            color_map = None
        
        # Translational augmentation (only if enabled via config)
        do_translate = self.translational_augment and sample_rng.random() < 0.3
        if do_translate:
            # Find max grid dimensions to determine safe translation range
            all_grids = input_grids + output_grids + [test_input, test_output]
            max_h = max(g.shape[0] for g in all_grids)
            max_w = max(g.shape[1] for g in all_grids)
            # Translation range: up to 4 cells, but stay within bounds
            max_translate_r = min(4, self.max_grid_size - max_h)
            max_translate_c = min(4, self.max_grid_size - max_w)
            translate_r = sample_rng.randint(0, max(0, max_translate_r))
            translate_c = sample_rng.randint(0, max(0, max_translate_c))
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
        
        # Return augmentation info for SCL transform_family assignment
        augment_info = {
            'dihedral_id': dihedral_id,  # 0-7, used as transform_family for SCL
            'color_permuted': do_color_perm,
            'translated': do_translate,
        }
        
        return aug_inputs, aug_outputs, aug_test_in, aug_test_out, augment_info


def pad_grid(grid: torch.Tensor, max_size: int, pad_value: int = None, is_target: bool = False) -> torch.Tensor:
    """
    Pad grid to max_size x max_size.
    
    Args:
        grid: Input grid tensor
        max_size: Target size
        pad_value: Value for padding. If None, use explicit defaults:
                   - Use -100 for targets (is_target=True) so loss ignores padding
                   - Use 10 (PAD_COLOR) for inputs so model distinguishes from black (0)
        is_target: If True and pad_value is None, use -100 for padding
    """
    h, w = grid.shape
    if h >= max_size and w >= max_size:
        return grid[:max_size, :max_size]
    
    # CRITICAL FIX: Use explicit padding values, not content-based inference
    # This prevents footgun where 0 is used for inputs (0 is a real ARC color!)
    if pad_value is None:
        if is_target:
            pad_value = -100  # Target grid: ignore_index for loss
        else:
            pad_value = 10    # Input grid: PAD_COLOR to distinguish from black (0)
    
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
    - Collect augmentation statistics for diversity logging
    
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
        - aug_stats: Optional dict with augmentation statistics for this batch
    """
    # Handle both parameter names
    if max_grid_size is not None:
        max_size = max_grid_size
    
    batch_size = len(batch)
    
    # Find max number of train pairs
    max_pairs = max(sample['num_train_pairs'] for sample in batch)
    
    # Padding constants
    PAD_COLOR = 10  # For input grids (distinguishes from black=0)
    PADDING_IGNORE_VALUE = -100  # For target grids (loss ignores)
    
    # Initialize tensors with proper padding values
    # - Input/output grids (used by encoder/context encoder): PAD_COLOR=10
    # - Test inputs: PAD_COLOR=10
    # - Test outputs (used by loss): PADDING_IGNORE_VALUE=-100
    input_grids = torch.full((batch_size, max_pairs, max_size, max_size), PAD_COLOR, dtype=torch.long)
    output_grids = torch.full((batch_size, max_pairs, max_size, max_size), PAD_COLOR, dtype=torch.long)
    test_inputs = torch.full((batch_size, max_size, max_size), PAD_COLOR, dtype=torch.long)
    test_outputs = torch.full((batch_size, max_size, max_size), PADDING_IGNORE_VALUE, dtype=torch.long)
    transform_families = torch.zeros(batch_size, dtype=torch.long)
    num_pairs = torch.zeros(batch_size, dtype=torch.long)
    grid_masks = torch.zeros(batch_size, max_pairs, dtype=torch.bool)
    
    task_ids = []
    
    # Augmentation statistics tracking
    dihedral_counts = [0] * 8  # Count of each dihedral transform (0-7)
    color_perm_count = 0
    translational_count = 0
    translational_offsets = []  # For computing offset diversity
    
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
        test_outputs[i] = pad_grid(sample['test_output'], max_size, is_target=True)
        
        # Transform family
        transform_families[i] = sample['transform_family']
        
        # Collect augmentation info if present
        if 'aug_info' in sample:
            aug_info = sample['aug_info']
            dihedral_id = aug_info.get('dihedral_id', 0)
            dihedral_counts[dihedral_id] += 1
            
            # Check if color_perm array is present (not just boolean check)
            if aug_info.get('color_perm') is not None:
                color_perm_count += 1
            
            offset = aug_info.get('translational_offset', (0, 0))
            if offset != (0, 0):
                translational_count += 1
                translational_offsets.append(offset)
    
    result = {
        'task_ids': task_ids,
        'input_grids': input_grids,
        'output_grids': output_grids,
        'test_inputs': test_inputs,
        'test_outputs': test_outputs,
        'transform_families': transform_families,
        'num_pairs': num_pairs,
        'grid_masks': grid_masks,
    }
    
    # Add augmentation stats if any samples had tracking info
    if any('aug_info' in sample for sample in batch):
        # Compute offset diversity (number of unique offsets)
        unique_offsets = len(set(translational_offsets)) if translational_offsets else 0
        
        result['aug_stats'] = {
            'dihedral_counts': dihedral_counts,  # [count_id0, count_id1, ..., count_id7]
            'color_perm_count': color_perm_count,
            'translational_count': translational_count,
            'unique_offsets': unique_offsets,
            'batch_size': batch_size,
        }
        
        # Also include per-sample aug_info for TRM-style inverse augmentation
        result['aug_info'] = [sample.get('aug_info', {'dihedral_id': 0}) for sample in batch]
    
    return result


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
    batch_size: int = 16,  # Reduced from 32 to prevent VRAM overflow
    num_workers: int = 4,
    shuffle: bool = True,
    augment: bool = True,
    max_grid_size: int = 30,
    seed: int = None,
    cache_samples: bool = False,
    cache_augmentations: int = 8,
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
        cache_samples: If True, pre-cache all samples in memory (eliminates data loading stalls)
        cache_augmentations: Number of pre-generated augmentations per task when caching
        **kwargs: Additional args for SCIARCDataset
    
    Returns:
        PyTorch DataLoader
    """
    # CRITICAL: Seed RNG BEFORE creating dataset so that:
    # 1. Cache building uses deterministic augmentations
    # 2. Any randomness during dataset init is reproducible
    if seed is not None:
        import random
        import numpy as np
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
    
    dataset = SCIARCDataset(
        data_dir=data_dir,
        split=split,
        augment=augment,
        max_grid_size=max_grid_size,
        cache_samples=cache_samples,
        cache_augmentations=cache_augmentations,
        **kwargs
    )
    
    # When using cached samples, we can use fewer workers since data is in memory
    effective_workers = 0 if cache_samples else num_workers
    
    # Setup generator for reproducibility
    g = None
    worker_init = None
    if seed is not None:
        g = torch.Generator()
        g.manual_seed(seed)
        worker_init = seed_worker
        
        # CRITICAL FIX: When num_workers=0, worker_init_fn is never called.
        # We need to seed the main process RNG directly for reproducibility.
        if effective_workers == 0:
            import random
            import numpy as np
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=effective_workers,
        collate_fn=partial(collate_sci_arc, max_grid_size=max_grid_size),
        pin_memory=True if not cache_samples else False,  # No need to pin if already in memory
        drop_last=True if shuffle else False,
        worker_init_fn=worker_init,
        generator=g,
        prefetch_factor=4 if effective_workers > 0 else None,  # Prefetch more batches
        persistent_workers=True if effective_workers > 0 else False,  # Keep workers alive
    )
    
    if cache_samples:
        print(f"DataLoader created with caching enabled (num_workers=0, data in memory)")
    
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


# =============================================================================
# UTILITY: Convert monolithic pickle cache to chunked format for fast partial loading
# =============================================================================

def convert_to_chunked_cache(pickle_path: str, chunk_size: int = 10000) -> None:
    """
    Convert a monolithic pickle cache to chunked format for fast partial loading.
    
    CRITICAL: Samples are SHUFFLED before chunking to ensure each chunk
    contains a representative mix of all tasks/difficulties. This allows
    partial loading (e.g., 4%) to still cover all task types.
    
    Usage:
        python -c "from sci_arc.data.dataset import convert_to_chunked_cache; convert_to_chunked_cache('./cache/rlan_stable_400k_v3.pkl')"
    
    This creates a .chunks directory next to the pickle file with:
        - meta.pkl: metadata (total_samples, chunk_size, num_chunks, shuffled=True)
        - chunk_0000.pkl, chunk_0001.pkl, ...: individual chunks (shuffled samples)
    
    After conversion, set cache_load_percent in config to load only needed chunks.
    E.g., cache_load_percent=4 loads ~4% of samples with REPRESENTATIVE task mix.
    """
    import pickle
    import time
    import random
    from pathlib import Path
    
    pickle_path = Path(pickle_path)
    if not pickle_path.exists():
        print(f"Error: {pickle_path} does not exist")
        return
    
    chunks_dir = Path(str(pickle_path) + '.chunks')
    if chunks_dir.exists():
        print(f"Chunked cache already exists at {chunks_dir}")
        print(f"Delete it first if you want to regenerate: rm -r {chunks_dir}")
        return
    
    print(f"Loading {pickle_path}...")
    start_time = time.time()
    
    with open(pickle_path, 'rb') as f:
        samples = pickle.load(f)
    
    load_time = time.time() - start_time
    print(f"Loaded {len(samples):,} samples in {load_time:.1f}s")
    
    # Create chunks directory
    chunks_dir.mkdir(parents=True, exist_ok=True)
    
    num_samples = len(samples)
    num_chunks = (num_samples + chunk_size - 1) // chunk_size
    
    # CRITICAL: Shuffle samples so each chunk has representative task mix
    print(f"Shuffling samples for representative distribution across chunks...")
    indices = list(range(num_samples))
    random.seed(42)  # Fixed seed for reproducibility
    random.shuffle(indices)
    
    print(f"Saving {num_chunks} chunks of {chunk_size} samples each (SHUFFLED)...")
    
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, num_samples)
        chunk_indices = indices[start_idx:end_idx]
        chunk_samples = [samples[i] for i in chunk_indices]
        
        chunk_path = chunks_dir / f'chunk_{chunk_idx:04d}.pkl'
        with open(chunk_path, 'wb') as f:
            pickle.dump(chunk_samples, f)
        
        if (chunk_idx + 1) % 10 == 0:
            print(f"  Saved chunk {chunk_idx + 1}/{num_chunks}")
    
    # Save metadata
    meta = {
        'total_samples': num_samples,
        'chunk_size': chunk_size,
        'num_chunks': num_chunks,
        'shuffled': True,
        'shuffle_seed': 42,
    }
    meta_path = chunks_dir / 'meta.pkl'
    with open(meta_path, 'wb') as f:
        pickle.dump(meta, f)
    
    total_time = time.time() - start_time
    print(f"\nDone! Chunked cache saved to {chunks_dir}")
    print(f"Total time: {total_time:.1f}s")
    print(f"\nSamples are SHUFFLED - each chunk contains representative task mix!")
    print(f"Now you can use cache_load_percent to load only what you need:")
    print(f"  cache_load_percent: 4    # Load ~{num_samples * 4 // 100:,} samples (all task types)")
    print(f"  cache_load_percent: 10   # Load ~{num_samples * 10 // 100:,} samples (all task types)")
    print(f"  cache_load_percent: 100  # Load all {num_samples:,} samples")
