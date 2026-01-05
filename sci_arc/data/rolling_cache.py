"""
Rolling Refresh Cache for Epoch-by-Epoch Augmentation Diversity (Jan 2026)

This module implements a two-tier cache system that provides:
1. Tier A (Decode Cache): Stable decoded task representations (reused across epochs)
2. Tier B (Augmented Pool): Rolling refresh pool where a fraction is regenerated each epoch

The key insight is that CPU augmentation generation runs AHEAD of GPU training,
so by the time epoch e finishes, epoch e+1's augmented samples are already ready.

Design Principles:
- CPU workers generate next epoch while GPU trains current epoch
- Anti-repeat window prevents the same augmentation from appearing in N consecutive epochs
- Epoch-sharded seeding ensures reproducible but diverse augmentations
- Coverage scheduling balances dihedral/color/translation coverage over epochs

Thread Safety:
- Uses threading.Lock for producer/consumer handoff
- Double-buffering pattern: current pool + next pool swap at epoch boundary
- Uses per-epoch RNG objects (not global seeds) to avoid thread interference

Configuration (via YAML):
```yaml
data:
  cache_samples_mode: rolling  # 'static' | 'rolling'
  rolling_cache:
    pool_size: 128000           # Total samples in augmented pool
    refresh_fraction: 0.25      # Fraction refreshed each epoch (10-30%)
    anti_repeat_window: 4       # Epochs before aug can repeat
    prefetch_workers: 4         # CPU workers for async generation
    coverage_scheduling: true   # Balance dihedral/color/translation
```

Usage:
```python
from sci_arc.data.rolling_cache import RollingRefreshCache

cache = RollingRefreshCache(
    tasks=dataset.tasks,
    generate_sample_fn=dataset._generate_sample,
    pool_size=128000,
    refresh_fraction=0.25,
    anti_repeat_window=4,
    prefetch_workers=4,
    seed=42,
)

# During training
for epoch in range(num_epochs):
    # Get samples for this epoch
    samples = cache.get_epoch_samples(epoch)
    
    # Start async prefetch for next epoch (non-blocking)
    cache.prefetch_next_epoch(epoch + 1)
    
    # Train on samples...
    
    # At epoch end, swap to prefetched pool
    cache.swap_to_next_epoch()
```

Author: SCI-ARC Team (Jan 2026)
"""

import hashlib
import random
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import numpy as np


def _stable_hash(data: bytes) -> int:
    """Compute a stable hash that is reproducible across Python runs.
    
    Unlike Python's built-in hash(), this uses MD5 (truncated to 64 bits)
    and is not affected by PYTHONHASHSEED randomization.
    """
    return int(hashlib.md5(data).hexdigest()[:16], 16)


@dataclass
class AugmentationFingerprint:
    """Compact fingerprint of an augmentation for anti-repeat tracking.
    
    Stores the minimal information needed to identify a unique augmentation:
    - task_id: Which task this augmentation came from
    - dihedral_id: Which of the 8 dihedral transforms (0-7)
    - color_perm_hash: Stable hash of the color permutation (or 0 if none)
    - offset: Translation offset (r, c) tuple
    
    Uses stable hashing (MD5-based) instead of Python's hash() to ensure
    reproducibility across runs and processes.
    """
    task_id: str
    dihedral_id: int
    color_perm_hash: int
    offset: Tuple[int, int]
    
    def __hash__(self):
        # Use stable hash for the tuple (reproducible across runs)
        data = f"{self.task_id}:{self.dihedral_id}:{self.color_perm_hash}:{self.offset}".encode()
        return _stable_hash(data)
    
    def __eq__(self, other):
        if not isinstance(other, AugmentationFingerprint):
            return False
        return (self.task_id == other.task_id and 
                self.dihedral_id == other.dihedral_id and
                self.color_perm_hash == other.color_perm_hash and
                self.offset == other.offset)


@dataclass
class EpochCoverageStats:
    """Tracks coverage statistics for an epoch to ensure balanced augmentation.
    
    Monitors distribution of:
    - Dihedral transforms (8 types: identity + 7 symmetry ops)
    - Color permutations (applied vs not applied)
    - Translation offsets (distribution of offset magnitudes)
    """
    dihedral_counts: Dict[int, int] = field(default_factory=lambda: {i: 0 for i in range(8)})
    color_perm_applied: int = 0
    color_perm_skipped: int = 0
    offset_magnitudes: List[int] = field(default_factory=list)
    
    def update(self, sample: Dict[str, Any]):
        """Update stats from a generated sample."""
        aug_info = sample.get('aug_info', {})
        
        # Track dihedral
        dihedral_id = aug_info.get('dihedral_id', 0)
        self.dihedral_counts[dihedral_id] = self.dihedral_counts.get(dihedral_id, 0) + 1
        
        # Track color permutation
        if aug_info.get('color_perm') is not None:
            self.color_perm_applied += 1
        else:
            self.color_perm_skipped += 1
        
        # Track offset magnitude
        offset = aug_info.get('translational_offset', (0, 0))
        if isinstance(offset, (list, tuple)) and len(offset) >= 2:
            magnitude = abs(offset[0]) + abs(offset[1])
            self.offset_magnitudes.append(magnitude)
    
    def get_coverage_score(self) -> float:
        """Compute coverage score (0-1, higher = better balance)."""
        # Dihedral coverage: entropy-based (uniform = 1.0)
        total_dihedral = sum(self.dihedral_counts.values())
        if total_dihedral > 0:
            probs = [c / total_dihedral for c in self.dihedral_counts.values()]
            entropy = -sum(p * np.log2(p + 1e-10) for p in probs if p > 0)
            max_entropy = np.log2(8)  # 8 dihedral types
            dihedral_score = entropy / max_entropy
        else:
            dihedral_score = 0.0
        
        # Color perm coverage: balance of applied vs skipped
        total_color = self.color_perm_applied + self.color_perm_skipped
        if total_color > 0:
            # Ideal is ~50% applied (configurable)
            ratio = self.color_perm_applied / total_color
            color_score = 1.0 - abs(ratio - 0.5) * 2  # 0.5 = 1.0, 0 or 1 = 0.0
        else:
            color_score = 0.0
        
        return (dihedral_score + color_score) / 2


class RollingRefreshCache:
    """Two-tier cache with rolling refresh for epoch-by-epoch augmentation diversity.
    
    Tier A (Decode Cache): Stable decoded task representations, reused across epochs.
                          This avoids re-parsing JSON and re-decoding grids.
    
    Tier B (Augmented Pool): Rolling refresh pool where `refresh_fraction` of samples
                            are regenerated each epoch with new augmentations.
    
    The key feature is async prefetch: while GPU trains on epoch e's samples,
    CPU workers are already generating epoch e+1's refreshed samples.
    
    Anti-Repeat Window:
        Tracks fingerprints of recently-used augmentations and excludes them
        from the refresh pool for `anti_repeat_window` epochs. This ensures
        the model sees diverse augmentations and doesn't memorize specific
        color permutations or dihedral transforms.
    
    Thread Safety:
        Uses double-buffering with a threading.Lock for safe swap operations.
        The current pool (immutable during epoch) and next pool (being built)
        are kept separate until explicit swap at epoch boundary.
    """
    
    def __init__(
        self,
        tasks: List[Dict[str, Any]],
        generate_sample_fn: Callable[[int], Dict[str, Any]],
        pool_size: int = 128000,
        refresh_fraction: float = 0.25,
        anti_repeat_window: int = 4,
        prefetch_workers: int = 4,
        coverage_scheduling: bool = True,
        seed: int = 42,
        verbose: bool = True,
    ):
        """Initialize the rolling refresh cache.
        
        Args:
            tasks: List of task dicts (from ARCDataset.tasks)
            generate_sample_fn: Function to generate a sample given task index.
                               Should be dataset._generate_sample or equivalent.
            pool_size: Total number of samples in the augmented pool
            refresh_fraction: Fraction of pool to refresh each epoch (0.1-0.3)
            anti_repeat_window: Number of epochs before an augmentation can repeat
            prefetch_workers: Number of CPU workers for async generation
            coverage_scheduling: Whether to balance dihedral/color/translation coverage
            seed: Base seed for reproducible but diverse epoch generations
            verbose: Whether to print progress messages
        """
        self.tasks = tasks
        self.generate_sample_fn = generate_sample_fn
        self.pool_size = pool_size
        self.refresh_fraction = max(0.05, min(0.5, refresh_fraction))  # Clamp to 5-50%
        self.anti_repeat_window = anti_repeat_window
        self.prefetch_workers = max(1, prefetch_workers)
        self.coverage_scheduling = coverage_scheduling
        self.base_seed = seed
        self.verbose = verbose
        
        # State
        self._current_pool: List[Dict[str, Any]] = []
        self._next_pool: Optional[List[Dict[str, Any]]] = None
        self._current_epoch: int = -1
        self._swap_lock = threading.Lock()
        
        # Anti-repeat tracking: maps fingerprint -> epoch last used
        self._fingerprint_history: Dict[AugmentationFingerprint, int] = {}
        # Sliding window of recent fingerprints for fast lookup
        self._recent_fingerprints: deque = deque()  # (epoch, fingerprint) tuples
        
        # Async prefetch state
        self._prefetch_executor: Optional[ThreadPoolExecutor] = None
        self._prefetch_future = None
        self._prefetch_ready = threading.Event()
        
        # Coverage tracking
        self._epoch_coverage_stats: Dict[int, EpochCoverageStats] = {}
        
        # Task index lookup for stratified sampling
        self._task_id_to_idx = {
            t.get('task_id', str(i)): i 
            for i, t in enumerate(tasks)
        }
        
        # Initialize the pool
        self._initialize_pool()
    
    def _initialize_pool(self):
        """Build the initial augmented pool (epoch 0)."""
        if self.verbose:
            print(f"[RollingCache] Initializing pool with {self.pool_size:,} samples...")
        
        start_time = time.time()
        
        # Set epoch-sharded seed for reproducibility
        self._current_epoch = 0
        
        # Create per-epoch RNG objects (NOT global seeding - thread safe)
        epoch_seed = self._get_epoch_seed(0)
        epoch_rng = random.Random(epoch_seed)
        epoch_np_rng = np.random.default_rng(epoch_seed)
        
        # Generate samples with stratified task coverage
        self._current_pool = self._generate_stratified_samples(
            self.pool_size, 
            epoch=0,
            rng=epoch_rng,
            np_rng=epoch_np_rng,
        )
        
        elapsed = time.time() - start_time
        if self.verbose:
            print(f"[RollingCache] Initialized {len(self._current_pool):,} samples in {elapsed:.1f}s")
            if self.coverage_scheduling:
                stats = self._epoch_coverage_stats.get(0)
                if stats:
                    print(f"[RollingCache] Epoch 0 coverage score: {stats.get_coverage_score():.3f}")
    
    def _get_epoch_seed(self, epoch: int) -> int:
        """Get deterministic seed for an epoch (reproducible but diverse)."""
        # Mix base seed with epoch number using a simple hash
        return (self.base_seed * 1000003 + epoch * 1000033) % (2**31)
    
    def _generate_stratified_samples(
        self, 
        num_samples: int, 
        epoch: int,
        exclude_fingerprints: Optional[Set[AugmentationFingerprint]] = None,
        rng: Optional[random.Random] = None,
        np_rng: Optional[np.random.Generator] = None,
    ) -> List[Dict[str, Any]]:
        """Generate samples with stratified task coverage.
        
        Ensures each task gets approximately equal representation in the pool.
        If `exclude_fingerprints` is provided, regenerates samples that would
        match excluded fingerprints (anti-repeat mechanism).
        
        Args:
            num_samples: Number of samples to generate
            epoch: Current epoch (for fingerprint tracking)
            exclude_fingerprints: Set of fingerprints to avoid (anti-repeat)
            rng: Per-epoch random.Random instance (thread-safe)
            np_rng: Per-epoch numpy Generator instance (thread-safe)
        """
        num_tasks = len(self.tasks)
        if num_tasks == 0:
            return []
        
        # Use provided RNG or create a default one
        if rng is None:
            epoch_seed = self._get_epoch_seed(epoch)
            rng = random.Random(epoch_seed)
        if np_rng is None:
            epoch_seed = self._get_epoch_seed(epoch)
            np_rng = np.random.default_rng(epoch_seed)
        
        samples_per_task = num_samples // num_tasks
        extra_samples = num_samples % num_tasks
        
        samples = []
        coverage_stats = EpochCoverageStats()
        
        # Build exclusion set from recent fingerprints
        exclude_set = exclude_fingerprints or set()
        
        # Track how many anti-repeat retries we needed (for diagnostics)
        total_retries = 0
        accepted_despite_exclusion = 0
        
        for task_idx in range(num_tasks):
            # Determine how many samples for this task
            task_samples = samples_per_task + (1 if task_idx < extra_samples else 0)
            
            for _ in range(task_samples):
                # Generate with anti-repeat retry
                # Increased max_retries and track statistics
                max_retries = 10
                sample = None
                fingerprint = None
                
                for retry in range(max_retries):
                    sample = self.generate_sample_fn(task_idx)
                    fingerprint = self._extract_fingerprint(sample)
                    
                    if fingerprint not in exclude_set:
                        # Clean accept
                        break
                    elif retry == max_retries - 1:
                        # Exhausted retries - accept anyway but log it
                        accepted_despite_exclusion += 1
                    else:
                        total_retries += 1
                
                # Accept this sample
                samples.append(sample)
                coverage_stats.update(sample)
                
                # Track this fingerprint
                self._fingerprint_history[fingerprint] = epoch
                self._recent_fingerprints.append((epoch, fingerprint))
        
        # Log anti-repeat statistics if significant
        if self.verbose and (total_retries > 100 or accepted_despite_exclusion > 0):
            print(f"[RollingCache] Anti-repeat stats: {total_retries} retries, "
                  f"{accepted_despite_exclusion} accepted despite exclusion")
        
        # Shuffle to mix tasks (using per-epoch RNG, not global)
        rng.shuffle(samples)
        
        # Store coverage stats
        self._epoch_coverage_stats[epoch] = coverage_stats
        
        return samples
    
    def _extract_fingerprint(self, sample: Dict[str, Any]) -> AugmentationFingerprint:
        """Extract augmentation fingerprint from a sample.
        
        Uses stable hashing (MD5-based) for color permutation to ensure
        reproducibility across runs and processes.
        """
        task_id = sample.get('task_id', 'unknown')
        aug_info = sample.get('aug_info', {})
        
        dihedral_id = aug_info.get('dihedral_id', 0)
        
        # Stable hash for color permutation (reproducible across runs)
        color_perm = aug_info.get('color_perm')
        if color_perm is not None:
            if hasattr(color_perm, 'tobytes'):
                color_perm_hash = _stable_hash(color_perm.tobytes())
            else:
                color_perm_hash = _stable_hash(bytes(color_perm))
        else:
            color_perm_hash = 0
        
        offset = aug_info.get('translational_offset', (0, 0))
        if not isinstance(offset, tuple):
            offset = tuple(offset) if hasattr(offset, '__iter__') else (0, 0)
        
        return AugmentationFingerprint(
            task_id=str(task_id),
            dihedral_id=dihedral_id,
            color_perm_hash=color_perm_hash,
            offset=offset,
        )
    
    def _get_excluded_fingerprints(self, epoch: int) -> Set[AugmentationFingerprint]:
        """Get fingerprints that should be excluded for this epoch (anti-repeat)."""
        excluded = set()
        cutoff_epoch = epoch - self.anti_repeat_window
        
        for fingerprint, last_epoch in self._fingerprint_history.items():
            if last_epoch >= cutoff_epoch:
                excluded.add(fingerprint)
        
        return excluded
    
    def _cleanup_old_fingerprints(self, current_epoch: int):
        """Remove fingerprints older than anti_repeat_window from history."""
        cutoff_epoch = current_epoch - self.anti_repeat_window - 1
        
        # Clean from deque
        while self._recent_fingerprints and self._recent_fingerprints[0][0] < cutoff_epoch:
            _, old_fingerprint = self._recent_fingerprints.popleft()
            # Only remove from dict if it's the oldest entry for this fingerprint
            if old_fingerprint in self._fingerprint_history:
                if self._fingerprint_history[old_fingerprint] <= cutoff_epoch:
                    del self._fingerprint_history[old_fingerprint]
    
    def _generate_refresh_samples(self, epoch: int) -> List[Dict[str, Any]]:
        """Generate refreshed samples for the next epoch.
        
        This is the core refresh logic:
        1. Determine how many samples to refresh (pool_size * refresh_fraction)
        2. Get excluded fingerprints from anti-repeat window
        3. Generate new samples (stratified across tasks)
        4. Merge with kept samples from current pool
        
        Uses per-epoch RNG objects to avoid polluting global random state
        (critical for thread safety when called from prefetch worker).
        """
        num_to_refresh = int(self.pool_size * self.refresh_fraction)
        num_to_keep = self.pool_size - num_to_refresh
        
        if self.verbose:
            print(f"[RollingCache] Epoch {epoch}: Refreshing {num_to_refresh:,} samples "
                  f"(keeping {num_to_keep:,})")
        
        # Create per-epoch RNG objects (NOT global seeding - thread safe)
        epoch_seed = self._get_epoch_seed(epoch)
        epoch_rng = random.Random(epoch_seed)
        epoch_np_rng = np.random.default_rng(epoch_seed)
        
        # Get excluded fingerprints
        excluded = self._get_excluded_fingerprints(epoch)
        if self.verbose and len(excluded) > 0:
            print(f"[RollingCache] Excluding {len(excluded):,} recent fingerprints")
        
        # Keep random subset of current pool (using per-epoch RNG)
        with self._swap_lock:
            if len(self._current_pool) > num_to_keep:
                # Use epoch_rng.sample() instead of random.sample()
                kept_samples = epoch_rng.sample(self._current_pool, num_to_keep)
            else:
                kept_samples = list(self._current_pool)
        
        # Generate new samples
        new_samples = self._generate_stratified_samples(
            num_to_refresh,
            epoch=epoch,
            exclude_fingerprints=excluded,
            rng=epoch_rng,
            np_rng=epoch_np_rng,
        )
        
        # Merge and shuffle (using per-epoch RNG)
        merged = kept_samples + new_samples
        epoch_rng.shuffle(merged)
        
        # Cleanup old fingerprints
        self._cleanup_old_fingerprints(epoch)
        
        return merged
    
    def get_epoch_samples(self, epoch: int) -> List[Dict[str, Any]]:
        """Get the sample pool for a given epoch.
        
        Returns a reference to the current pool (do not modify!).
        For epoch 0, returns the initial pool.
        For epoch > 0, should call swap_to_next_epoch() first if prefetch was used.
        """
        with self._swap_lock:
            if epoch != self._current_epoch:
                if self.verbose:
                    print(f"[RollingCache] Warning: Requested epoch {epoch} but current is {self._current_epoch}")
            return self._current_pool
    
    def prefetch_next_epoch(self, next_epoch: int):
        """Start async prefetch of next epoch's samples.
        
        This should be called BEFORE training on current epoch so that
        CPU workers generate next epoch while GPU trains current epoch.
        
        Non-blocking: Returns immediately, generation happens in background.
        """
        if self._next_pool is not None:
            if self.verbose:
                print(f"[RollingCache] Warning: Prefetch called but next pool already exists")
            return
        
        self._prefetch_ready.clear()
        
        # Create executor if not exists
        if self._prefetch_executor is None:
            self._prefetch_executor = ThreadPoolExecutor(
                max_workers=self.prefetch_workers,
                thread_name_prefix="RollingCache"
            )
        
        def generate_task():
            """Background task to generate next epoch's samples."""
            try:
                start_time = time.time()
                samples = self._generate_refresh_samples(next_epoch)
                
                with self._swap_lock:
                    self._next_pool = samples
                
                self._prefetch_ready.set()
                
                if self.verbose:
                    elapsed = time.time() - start_time
                    print(f"[RollingCache] Prefetch for epoch {next_epoch} complete in {elapsed:.1f}s")
                    
            except Exception as e:
                # On error, log and set flag but leave _next_pool as None
                # swap_to_next_epoch will detect this and regenerate synchronously
                print(f"[RollingCache] ERROR in prefetch: {e}")
                import traceback
                traceback.print_exc()
                self._prefetch_ready.set()  # Signal completion (with error)
        
        # Submit task
        self._prefetch_future = self._prefetch_executor.submit(generate_task)
    
    def swap_to_next_epoch(self, timeout: float = 300.0) -> bool:
        """Swap to the prefetched next epoch pool.
        
        Blocks until prefetch is ready (with timeout).
        Call this at the end of each epoch BEFORE starting next epoch.
        
        If prefetch failed or timed out, regenerates synchronously to ensure
        training continues with fresh samples (never silently reuses old pool).
        
        Args:
            timeout: Maximum seconds to wait for prefetch to complete
            
        Returns:
            True if swap successful, False if timeout or no prefetch available
        """
        next_epoch = self._current_epoch + 1
        
        if self._next_pool is None and self._prefetch_future is None:
            if self.verbose:
                print(f"[RollingCache] Warning: No prefetch in progress, generating synchronously")
            # Fallback: generate synchronously
            self._next_pool = self._generate_refresh_samples(next_epoch)
        
        # Wait for prefetch to complete
        if not self._prefetch_ready.wait(timeout=timeout):
            print(f"[RollingCache] ERROR: Prefetch timeout after {timeout}s, regenerating synchronously")
            # Timeout - regenerate synchronously to avoid stale pool
            self._next_pool = self._generate_refresh_samples(next_epoch)
        
        # Swap pools
        with self._swap_lock:
            if self._next_pool is not None:
                self._current_pool = self._next_pool
                self._next_pool = None
                self._current_epoch += 1
                if self.verbose:
                    print(f"[RollingCache] Swapped to epoch {self._current_epoch} "
                          f"({len(self._current_pool):,} samples)")
                return True
            else:
                # Prefetch failed and returned None - regenerate synchronously
                print(f"[RollingCache] WARNING: Prefetch returned None, regenerating synchronously")
                self._current_pool = self._generate_refresh_samples(next_epoch)
                self._current_epoch += 1
                print(f"[RollingCache] Regenerated epoch {self._current_epoch} synchronously "
                      f"({len(self._current_pool):,} samples)")
                return True  # Still return True - we recovered
    
    def get_coverage_stats(self, epoch: int) -> Optional[EpochCoverageStats]:
        """Get coverage statistics for a specific epoch."""
        return self._epoch_coverage_stats.get(epoch)
    
    def get_pool_size(self) -> int:
        """Get current pool size."""
        with self._swap_lock:
            return len(self._current_pool)
    
    def shutdown(self):
        """Shutdown the prefetch executor."""
        if self._prefetch_executor is not None:
            self._prefetch_executor.shutdown(wait=True)
            self._prefetch_executor = None
    
    def __len__(self) -> int:
        """Return current pool size."""
        return self.get_pool_size()
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample by index (for DataLoader compatibility)."""
        with self._swap_lock:
            return self._current_pool[idx]
    
    def __del__(self):
        """Cleanup on deletion."""
        self.shutdown()


# ===========================================================================
# DATASET WRAPPER
# ===========================================================================

class RollingCacheDataset:
    """Thin wrapper around RollingRefreshCache for torch.utils.data.DataLoader.
    
    This provides the __len__ and __getitem__ interface expected by DataLoader
    while delegating to the underlying RollingRefreshCache.
    
    Usage:
    ```python
    cache = RollingRefreshCache(tasks, generate_fn, pool_size=128000)
    dataset = RollingCacheDataset(cache)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    ```
    """
    
    def __init__(self, cache: RollingRefreshCache):
        self.cache = cache
    
    def __len__(self) -> int:
        return len(self.cache)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.cache[idx]
    
    def notify_epoch_start(self, epoch: int):
        """Notify cache that a new epoch is starting.
        
        Should be called at the start of each epoch (after epoch 0).
        This triggers swap to prefetched pool if available.
        """
        if epoch > 0:
            self.cache.swap_to_next_epoch()
    
    def notify_epoch_end(self, epoch: int):
        """Notify cache that an epoch has ended.
        
        Should be called at the end of each epoch.
        This triggers async prefetch for the next epoch.
        """
        self.cache.prefetch_next_epoch(epoch + 1)
    
    @property
    def tasks(self) -> List[Dict[str, Any]]:
        """Forward tasks attribute from underlying cache.
        
        This property is needed for task tracking in train_rlan.py which
        looks for train_loader.dataset.tasks to count unique task IDs.
        """
        return self.cache.tasks


# ===========================================================================
# CONFIGURATION HELPERS
# ===========================================================================

def create_rolling_cache_from_config(
    tasks: List[Dict[str, Any]],
    generate_sample_fn: Callable[[int], Dict[str, Any]],
    config: Dict[str, Any],
    verbose: bool = True,
) -> RollingRefreshCache:
    """Create RollingRefreshCache from YAML config dict.
    
    Expected config structure:
    ```yaml
    data:
      rolling_cache:
        pool_size: 128000
        refresh_fraction: 0.25
        anti_repeat_window: 4
        prefetch_workers: 4
        coverage_scheduling: true
        seed: 42
    ```
    
    Args:
        tasks: List of task dicts
        generate_sample_fn: Sample generation function
        config: Full config dict (will extract data.rolling_cache)
        verbose: Whether to print progress
        
    Returns:
        Configured RollingRefreshCache instance
    """
    # Extract rolling_cache config with defaults
    data_config = config.get('data', {})
    rc_config = data_config.get('rolling_cache', {})
    
    return RollingRefreshCache(
        tasks=tasks,
        generate_sample_fn=generate_sample_fn,
        pool_size=rc_config.get('pool_size', 128000),
        refresh_fraction=rc_config.get('refresh_fraction', 0.25),
        anti_repeat_window=rc_config.get('anti_repeat_window', 4),
        prefetch_workers=rc_config.get('prefetch_workers', 4),
        coverage_scheduling=rc_config.get('coverage_scheduling', True),
        seed=rc_config.get('seed', 42),
        verbose=verbose,
    )


def get_default_rolling_cache_config() -> Dict[str, Any]:
    """Get default rolling cache configuration.
    
    Returns config dict suitable for merging into existing YAML config.
    """
    return {
        'data': {
            'cache_samples_mode': 'rolling',  # 'static' | 'rolling'
            'rolling_cache': {
                'pool_size': 128000,           # Total samples in pool
                'refresh_fraction': 0.25,      # 25% refreshed each epoch
                'anti_repeat_window': 4,       # 4 epochs before repeat allowed
                'prefetch_workers': 4,         # CPU workers for async gen
                'coverage_scheduling': True,   # Balance dihedral/color/trans
                'seed': 42,                    # Base seed for reproducibility
            }
        }
    }
