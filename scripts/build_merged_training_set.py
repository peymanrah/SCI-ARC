#!/usr/bin/env python3
"""
Merged Training Set Builder for SCI-ARC RLAN
=============================================

This is a STANDALONE script that creates a merged training set from:
- ARC-AGI-1 training tasks (400 tasks)
- ARC-AGI-2 training tasks (1000 tasks, includes some ARC-AGI-1 tasks + new tasks)

WITHOUT leaking into evaluation sets:
- ARC-AGI-1 evaluation (400 tasks)
- ARC-AGI-2 evaluation (120 tasks)

The script implements:
1. Canonical fingerprinting (D4 + translation + color permutation invariant)
2. Exact deduplication against eval sets
3. Near-duplicate detection (conservative quarantine)
4. Audit ledgers for transparency
5. Merged manifest generation

USAGE:
    python scripts/build_merged_training_set.py --output-dir ./data/merged_training

This script does NOT modify the existing codebase or data loading.
It creates a new manifest that can be used by setting:
    data.use_merged_training: true
in rlan_stable_dev.yaml

Author: SCI-ARC Team
Date: January 2026
"""

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
import random

import numpy as np


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class BuildConfig:
    """Configuration for building the merged training set."""
    
    # Data paths (relative to repo root)
    agi1_train_path: str = "./data/arc-agi/data/training"
    agi1_eval_path: str = "./data/arc-agi/data/evaluation"
    agi2_train_path: str = "./data/arc-agi-2/data/training"
    agi2_eval_path: str = "./data/arc-agi-2/data/evaluation"
    
    # Output paths
    output_dir: str = "./data/merged_training"
    
    # Near-duplicate detection thresholds
    near_dup_histogram_threshold: float = 0.95  # Cosine similarity of color histograms
    near_dup_structure_threshold: float = 0.90  # Structural similarity score
    
    # Dev split configuration
    dev_split_ratio: float = 0.05  # 5% of merged training for dev set
    dev_split_seed: int = 42  # Deterministic split
    
    # Processing options
    verbose: bool = True
    dry_run: bool = False  # If True, don't write files


# =============================================================================
# CANONICAL FINGERPRINTING (D4 + Translation + Color Invariant)
# =============================================================================

def crop_to_content(grid: np.ndarray) -> np.ndarray:
    """
    Crop grid to its content bounding box.
    Uses mode color as background proxy.
    """
    if grid.size == 0:
        return grid
    
    # Find mode (most common) color as background
    unique, counts = np.unique(grid, return_counts=True)
    mode_color = unique[np.argmax(counts)]
    
    # Find content mask (non-background)
    content_mask = grid != mode_color
    
    if not content_mask.any():
        # All same color - return 1x1 with that color
        return grid[0:1, 0:1]
    
    # Find bounding box
    rows = np.any(content_mask, axis=1)
    cols = np.any(content_mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    return grid[rmin:rmax+1, cmin:cmax+1]


def apply_dihedral(grid: np.ndarray, transform_id: int) -> np.ndarray:
    """
    Apply one of 8 dihedral transforms (D4 group).
    
    Transform IDs:
    0: identity
    1: rotate 90° CCW
    2: rotate 180°
    3: rotate 270° CCW (90° CW)
    4: flip horizontal
    5: flip horizontal + rotate 90°
    6: flip horizontal + rotate 180°
    7: flip horizontal + rotate 270°
    """
    # Apply rotation
    rotations = transform_id % 4
    result = np.rot90(grid, k=rotations)
    
    # Apply flip if needed
    if transform_id >= 4:
        result = np.fliplr(result)
    
    return result


def canonicalize_colors(grids: List[np.ndarray]) -> Tuple[List[np.ndarray], Dict[int, int]]:
    """
    Remap colors by first-appearance order across all grids.
    Returns remapped grids and the color mapping used.
    """
    # Build first-appearance ordering
    color_order = []
    seen = set()
    
    for grid in grids:
        for row in grid:
            for val in row:
                val = int(val)
                if val not in seen:
                    color_order.append(val)
                    seen.add(val)
    
    # Create mapping: original -> canonical (0, 1, 2, ...)
    color_map = {orig: new for new, orig in enumerate(color_order)}
    
    # Apply mapping
    remapped = []
    for grid in grids:
        new_grid = np.zeros_like(grid)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                new_grid[i, j] = color_map.get(int(grid[i, j]), grid[i, j])
        remapped.append(new_grid)
    
    return remapped, color_map


def serialize_task(task_grids: List[np.ndarray]) -> bytes:
    """Serialize a list of grids to bytes deterministically."""
    data = []
    for grid in task_grids:
        data.append(grid.tolist())
    return json.dumps(data, sort_keys=True, separators=(',', ':')).encode('utf-8')


def compute_canonical_hash(task: Dict) -> str:
    """
    Compute canonical hash for a task that is invariant to:
    - Dihedral transforms (D4 group)
    - Translation (content cropping)
    - Color relabeling (first-appearance order)
    
    Returns SHA-256 hex digest.
    """
    # Extract all grids from the task
    grids = []
    
    for pair in task.get('train', []):
        grids.append(np.array(pair['input'], dtype=np.int64))
        grids.append(np.array(pair['output'], dtype=np.int64))
    
    for pair in task.get('test', []):
        grids.append(np.array(pair['input'], dtype=np.int64))
        if 'output' in pair:
            grids.append(np.array(pair['output'], dtype=np.int64))
    
    if not grids:
        return hashlib.sha256(b"empty_task").hexdigest()
    
    # Try all 8 dihedral transforms and pick lexicographically smallest
    min_serialization = None
    
    for d_id in range(8):
        # Apply dihedral to all grids
        transformed = [apply_dihedral(g, d_id) for g in grids]
        
        # Crop to content
        cropped = [crop_to_content(g) for g in transformed]
        
        # Canonicalize colors
        remapped, _ = canonicalize_colors(cropped)
        
        # Serialize
        serialized = serialize_task(remapped)
        
        # Keep minimum
        if min_serialization is None or serialized < min_serialization:
            min_serialization = serialized
    
    return hashlib.sha256(min_serialization).hexdigest()


def compute_raw_hash(task: Dict) -> str:
    """Compute raw hash of task JSON for identity tracking."""
    serialized = json.dumps(task, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(serialized).hexdigest()


# =============================================================================
# NEAR-DUPLICATE DETECTION
# =============================================================================

@dataclass
class TaskFeatures:
    """Feature vector for near-duplicate detection."""
    task_uid: str
    num_train_pairs: int
    max_grid_height: int
    max_grid_width: int
    has_size_change: bool
    color_histogram: np.ndarray  # Normalized histogram over all grids
    total_cells: int
    unique_colors: int
    
    def to_dict(self) -> Dict:
        return {
            'task_uid': self.task_uid,
            'num_train_pairs': self.num_train_pairs,
            'max_grid_height': self.max_grid_height,
            'max_grid_width': self.max_grid_width,
            'has_size_change': self.has_size_change,
            'color_histogram': self.color_histogram.tolist(),
            'total_cells': self.total_cells,
            'unique_colors': self.unique_colors,
        }


def extract_features(task: Dict, task_uid: str) -> TaskFeatures:
    """Extract features for near-duplicate detection."""
    grids = []
    
    for pair in task.get('train', []):
        grids.append(np.array(pair['input'], dtype=np.int64))
        grids.append(np.array(pair['output'], dtype=np.int64))
    
    for pair in task.get('test', []):
        grids.append(np.array(pair['input'], dtype=np.int64))
        if 'output' in pair:
            grids.append(np.array(pair['output'], dtype=np.int64))
    
    # Compute features
    num_train = len(task.get('train', []))
    max_h = max(g.shape[0] for g in grids) if grids else 0
    max_w = max(g.shape[1] for g in grids) if grids else 0
    
    # Check for size changes
    has_size_change = False
    for pair in task.get('train', []):
        inp = np.array(pair['input'])
        out = np.array(pair['output'])
        if inp.shape != out.shape:
            has_size_change = True
            break
    
    # Color histogram (normalized)
    histogram = np.zeros(10, dtype=np.float64)
    total_cells = 0
    for g in grids:
        for val in g.flat:
            if 0 <= val <= 9:
                histogram[val] += 1
                total_cells += 1
    
    if total_cells > 0:
        histogram /= total_cells
    
    # Unique colors
    unique_colors = len(set(int(v) for g in grids for v in g.flat))
    
    return TaskFeatures(
        task_uid=task_uid,
        num_train_pairs=num_train,
        max_grid_height=max_h,
        max_grid_width=max_w,
        has_size_change=has_size_change,
        color_histogram=histogram,
        total_cells=total_cells,
        unique_colors=unique_colors,
    )


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def compute_structural_similarity(f1: TaskFeatures, f2: TaskFeatures) -> float:
    """
    Compute structural similarity between two tasks.
    Returns a score in [0, 1] where 1 means very similar.
    """
    score = 0.0
    weights = 0.0
    
    # Same number of train pairs (weight 2)
    if f1.num_train_pairs == f2.num_train_pairs:
        score += 2.0
    weights += 2.0
    
    # Same size change pattern (weight 3)
    if f1.has_size_change == f2.has_size_change:
        score += 3.0
    weights += 3.0
    
    # Similar max dimensions (weight 2)
    h_diff = abs(f1.max_grid_height - f2.max_grid_height)
    w_diff = abs(f1.max_grid_width - f2.max_grid_width)
    if h_diff <= 2 and w_diff <= 2:
        score += 2.0
    weights += 2.0
    
    # Similar unique color count (weight 1)
    if abs(f1.unique_colors - f2.unique_colors) <= 1:
        score += 1.0
    weights += 1.0
    
    return score / weights if weights > 0 else 0.0


def detect_near_duplicates(
    candidate_features: List[TaskFeatures],
    eval_features: List[TaskFeatures],
    histogram_threshold: float = 0.95,
    structure_threshold: float = 0.90,
) -> Dict[str, List[Dict]]:
    """
    Detect near-duplicates between candidates and eval tasks.
    
    Returns: {candidate_uid: [{'eval_uid': ..., 'histogram_sim': ..., 'struct_sim': ...}]}
    """
    near_dups = {}
    
    for cand in candidate_features:
        suspects = []
        
        for eval_f in eval_features:
            # Quick structural check first
            struct_sim = compute_structural_similarity(cand, eval_f)
            if struct_sim < structure_threshold:
                continue
            
            # Histogram similarity (more expensive)
            hist_sim = cosine_similarity(cand.color_histogram, eval_f.color_histogram)
            
            if hist_sim >= histogram_threshold:
                suspects.append({
                    'eval_uid': eval_f.task_uid,
                    'histogram_similarity': round(hist_sim, 4),
                    'structural_similarity': round(struct_sim, 4),
                })
        
        if suspects:
            near_dups[cand.task_uid] = suspects
    
    return near_dups


# =============================================================================
# TASK LOADING
# =============================================================================

@dataclass
class TaskRecord:
    """Record for a single task in the registry."""
    task_uid: str
    source: str  # agi1_train, agi2_train, agi1_eval, agi2_eval
    path: str
    raw_sha256: str
    canonical_sha256: str
    num_train_pairs: int
    max_grid_size: int
    has_size_change: bool
    
    def to_dict(self) -> Dict:
        return asdict(self)


def load_tasks_from_directory(
    dir_path: str,
    source_label: str,
    verbose: bool = False,
) -> List[Tuple[Dict, TaskRecord]]:
    """
    Load all tasks from a directory.
    Returns list of (task_dict, TaskRecord) tuples.
    """
    path = Path(dir_path)
    if not path.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")
    
    json_files = sorted(path.glob("*.json"))
    
    tasks = []
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                task = json.load(f)
            
            task_uid = f"{source_label}:{json_file.stem}"
            
            # Compute hashes
            raw_hash = compute_raw_hash(task)
            canonical_hash = compute_canonical_hash(task)
            
            # Extract quick metadata
            num_pairs = len(task.get('train', []))
            
            max_size = 0
            has_size_change = False
            for pair in task.get('train', []):
                inp = pair.get('input', [[]])
                out = pair.get('output', [[]])
                max_size = max(max_size, len(inp), len(inp[0]) if inp else 0)
                max_size = max(max_size, len(out), len(out[0]) if out else 0)
                if len(inp) != len(out) or (inp and out and len(inp[0]) != len(out[0])):
                    has_size_change = True
            
            for pair in task.get('test', []):
                inp = pair.get('input', [[]])
                max_size = max(max_size, len(inp), len(inp[0]) if inp else 0)
            
            record = TaskRecord(
                task_uid=task_uid,
                source=source_label,
                path=str(json_file),
                raw_sha256=raw_hash,
                canonical_sha256=canonical_hash,
                num_train_pairs=num_pairs,
                max_grid_size=max_size,
                has_size_change=has_size_change,
            )
            
            tasks.append((task, record))
            
        except Exception as e:
            if verbose:
                print(f"Warning: Failed to load {json_file}: {e}")
    
    if verbose:
        print(f"  Loaded {len(tasks)} tasks from {source_label}")
    
    return tasks


# =============================================================================
# STRATIFIED DEV SPLIT
# =============================================================================

def stratified_dev_split(
    tasks: List[TaskRecord],
    dev_ratio: float = 0.05,
    seed: int = 42,
) -> Tuple[List[TaskRecord], List[TaskRecord]]:
    """
    Split tasks into train and dev sets with stratification.
    
    Stratifies by:
    - Grid size bucket (small/medium/large)
    - Number of train pairs bucket (few/medium/many)
    - Size change flag
    """
    rng = random.Random(seed)
    
    # Define buckets
    def get_bucket(record: TaskRecord) -> str:
        # Size bucket
        if record.max_grid_size <= 10:
            size_bucket = "small"
        elif record.max_grid_size <= 20:
            size_bucket = "medium"
        else:
            size_bucket = "large"
        
        # Pairs bucket
        if record.num_train_pairs <= 2:
            pairs_bucket = "few"
        elif record.num_train_pairs <= 4:
            pairs_bucket = "medium"
        else:
            pairs_bucket = "many"
        
        # Size change
        change = "change" if record.has_size_change else "same"
        
        return f"{size_bucket}_{pairs_bucket}_{change}"
    
    # Group by bucket
    buckets: Dict[str, List[TaskRecord]] = defaultdict(list)
    for record in tasks:
        bucket = get_bucket(record)
        buckets[bucket].append(record)
    
    # Sample from each bucket
    train_records = []
    dev_records = []
    
    for bucket_name, bucket_tasks in buckets.items():
        rng.shuffle(bucket_tasks)
        n_dev = max(1, int(len(bucket_tasks) * dev_ratio))
        dev_records.extend(bucket_tasks[:n_dev])
        train_records.extend(bucket_tasks[n_dev:])
    
    return train_records, dev_records


# =============================================================================
# MAIN BUILD PIPELINE
# =============================================================================

@dataclass
class BuildResult:
    """Result of the build process."""
    merged_train_count: int
    merged_dev_count: int
    exact_dedup_count: int
    near_dup_quarantine_count: int
    manifest_sha256: str
    build_timestamp: str
    
    def to_dict(self) -> Dict:
        return asdict(self)


def build_merged_training_set(config: BuildConfig) -> BuildResult:
    """
    Main entry point for building the merged training set.
    """
    if config.verbose:
        print("=" * 70)
        print("MERGED TRAINING SET BUILDER")
        print("=" * 70)
        print(f"  ARC-AGI-1 train: {config.agi1_train_path}")
        print(f"  ARC-AGI-2 train: {config.agi2_train_path}")
        print(f"  ARC-AGI-1 eval:  {config.agi1_eval_path}")
        print(f"  ARC-AGI-2 eval:  {config.agi2_eval_path}")
        print(f"  Output dir:      {config.output_dir}")
        print()
    
    # ==========================================================================
    # PHASE 1: Load all task sets
    # ==========================================================================
    if config.verbose:
        print("PHASE 1: Loading task sets...")
    
    agi1_train_tasks = load_tasks_from_directory(
        config.agi1_train_path, "agi1_train", config.verbose
    )
    agi2_train_tasks = load_tasks_from_directory(
        config.agi2_train_path, "agi2_train", config.verbose
    )
    agi1_eval_tasks = load_tasks_from_directory(
        config.agi1_eval_path, "agi1_eval", config.verbose
    )
    agi2_eval_tasks = load_tasks_from_directory(
        config.agi2_eval_path, "agi2_eval", config.verbose
    )
    
    # Build eval canonical hash set
    eval_canonical_hashes: Set[str] = set()
    eval_features: List[TaskFeatures] = []
    
    for task, record in agi1_eval_tasks + agi2_eval_tasks:
        eval_canonical_hashes.add(record.canonical_sha256)
        eval_features.append(extract_features(task, record.task_uid))
    
    if config.verbose:
        print(f"  Total eval tasks: {len(eval_canonical_hashes)} unique canonical hashes")
        print()
    
    # ==========================================================================
    # PHASE 2: Exact deduplication
    # ==========================================================================
    if config.verbose:
        print("PHASE 2: Exact deduplication against eval sets...")
    
    # Candidate training tasks
    candidates = agi1_train_tasks + agi2_train_tasks
    
    # Track exact duplicates
    exact_dups = []
    surviving_candidates = []
    
    for task, record in candidates:
        if record.canonical_sha256 in eval_canonical_hashes:
            exact_dups.append({
                'task_uid': record.task_uid,
                'source': record.source,
                'canonical_sha256': record.canonical_sha256,
                'reason': 'exact_canonical_match_with_eval',
            })
        else:
            surviving_candidates.append((task, record))
    
    if config.verbose:
        print(f"  Candidates before dedup: {len(candidates)}")
        print(f"  Exact duplicates removed: {len(exact_dups)}")
        print(f"  Candidates after dedup:  {len(surviving_candidates)}")
        print()
    
    # ==========================================================================
    # PHASE 3: Near-duplicate detection
    # ==========================================================================
    if config.verbose:
        print("PHASE 3: Near-duplicate detection...")
    
    # Extract features for surviving candidates
    candidate_features = [
        extract_features(task, record.task_uid) 
        for task, record in surviving_candidates
    ]
    
    # Detect near-duplicates
    near_dups = detect_near_duplicates(
        candidate_features,
        eval_features,
        histogram_threshold=config.near_dup_histogram_threshold,
        structure_threshold=config.near_dup_structure_threshold,
    )
    
    # Quarantine near-duplicates
    quarantined_uids = set(near_dups.keys())
    final_candidates = [
        (task, record) for task, record in surviving_candidates
        if record.task_uid not in quarantined_uids
    ]
    
    if config.verbose:
        print(f"  Near-duplicates quarantined: {len(near_dups)}")
        print(f"  Final candidates: {len(final_candidates)}")
        print()
    
    # ==========================================================================
    # PHASE 4: Deduplication within training sets
    # ==========================================================================
    if config.verbose:
        print("PHASE 4: Deduplication within training sets...")
    
    # Remove duplicates between AGI-1 and AGI-2 training (prefer AGI-1 for consistency)
    seen_canonical: Set[str] = set()
    deduplicated = []
    internal_dups = []
    
    # Sort to prefer AGI-1 (comes first alphabetically)
    sorted_candidates = sorted(final_candidates, key=lambda x: x[1].source)
    
    for task, record in sorted_candidates:
        if record.canonical_sha256 in seen_canonical:
            internal_dups.append({
                'task_uid': record.task_uid,
                'source': record.source,
                'canonical_sha256': record.canonical_sha256,
                'reason': 'duplicate_within_training_sets',
            })
        else:
            seen_canonical.add(record.canonical_sha256)
            deduplicated.append((task, record))
    
    if config.verbose:
        print(f"  Internal duplicates removed: {len(internal_dups)}")
        print(f"  Final unique tasks: {len(deduplicated)}")
        print()
    
    # ==========================================================================
    # PHASE 5: Stratified dev split
    # ==========================================================================
    if config.verbose:
        print("PHASE 5: Stratified dev/train split...")
    
    records_only = [record for _, record in deduplicated]
    train_records, dev_records = stratified_dev_split(
        records_only,
        dev_ratio=config.dev_split_ratio,
        seed=config.dev_split_seed,
    )
    
    if config.verbose:
        print(f"  Train set: {len(train_records)} tasks")
        print(f"  Dev set:   {len(dev_records)} tasks")
        print()
    
    # ==========================================================================
    # PHASE 6: Write outputs
    # ==========================================================================
    if config.verbose:
        print("PHASE 6: Writing outputs...")
    
    if not config.dry_run:
        output_path = Path(config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Write merged train manifest
        train_manifest_path = output_path / "merged_train_manifest.jsonl"
        with open(train_manifest_path, 'w') as f:
            for record in train_records:
                f.write(json.dumps(record.to_dict()) + '\n')
        
        # Write merged dev manifest
        dev_manifest_path = output_path / "merged_dev_manifest.jsonl"
        with open(dev_manifest_path, 'w') as f:
            for record in dev_records:
                f.write(json.dumps(record.to_dict()) + '\n')
        
        # Write exclusion ledger (exact dups)
        exclusion_path = output_path / "excluded_exact.jsonl"
        with open(exclusion_path, 'w') as f:
            for entry in exact_dups:
                f.write(json.dumps(entry) + '\n')
            for entry in internal_dups:
                f.write(json.dumps(entry) + '\n')
        
        # Write quarantine ledger (near dups)
        quarantine_path = output_path / "quarantine_near_dup.jsonl"
        with open(quarantine_path, 'w') as f:
            for task_uid, suspects in near_dups.items():
                entry = {
                    'task_uid': task_uid,
                    'reason': 'near_duplicate_to_eval',
                    'suspects': suspects,
                }
                f.write(json.dumps(entry) + '\n')
        
        # Write build metadata
        build_timestamp = datetime.now().isoformat()
        
        # Compute manifest hash
        manifest_content = open(train_manifest_path, 'rb').read()
        manifest_sha256 = hashlib.sha256(manifest_content).hexdigest()
        
        metadata = {
            'build_timestamp': build_timestamp,
            'manifest_sha256': manifest_sha256,
            'config': asdict(config),
            'stats': {
                'agi1_train_loaded': len(agi1_train_tasks),
                'agi2_train_loaded': len(agi2_train_tasks),
                'agi1_eval_loaded': len(agi1_eval_tasks),
                'agi2_eval_loaded': len(agi2_eval_tasks),
                'exact_dups_removed': len(exact_dups),
                'internal_dups_removed': len(internal_dups),
                'near_dups_quarantined': len(near_dups),
                'final_train_count': len(train_records),
                'final_dev_count': len(dev_records),
            },
        }
        
        metadata_path = output_path / "build_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        if config.verbose:
            print(f"  Wrote: {train_manifest_path}")
            print(f"  Wrote: {dev_manifest_path}")
            print(f"  Wrote: {exclusion_path}")
            print(f"  Wrote: {quarantine_path}")
            print(f"  Wrote: {metadata_path}")
            print()
            print(f"  Manifest SHA256: {manifest_sha256}")
    else:
        build_timestamp = datetime.now().isoformat()
        manifest_sha256 = "DRY_RUN_NO_FILES_WRITTEN"
        if config.verbose:
            print("  [DRY RUN] No files written")
    
    # ==========================================================================
    # DONE
    # ==========================================================================
    if config.verbose:
        print()
        print("=" * 70)
        print("BUILD COMPLETE")
        print("=" * 70)
        print(f"  Merged training set: {len(train_records)} tasks")
        print(f"  Dev set: {len(dev_records)} tasks")
        print(f"  Excluded (exact dup): {len(exact_dups) + len(internal_dups)}")
        print(f"  Quarantined (near dup): {len(near_dups)}")
        print()
    
    return BuildResult(
        merged_train_count=len(train_records),
        merged_dev_count=len(dev_records),
        exact_dedup_count=len(exact_dups) + len(internal_dups),
        near_dup_quarantine_count=len(near_dups),
        manifest_sha256=manifest_sha256,
        build_timestamp=build_timestamp,
    )


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Build merged training set from ARC-AGI-1 + ARC-AGI-2 without leakage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Build with defaults
    python scripts/build_merged_training_set.py
    
    # Custom output directory
    python scripts/build_merged_training_set.py --output-dir ./data/my_merged
    
    # Dry run (don't write files)
    python scripts/build_merged_training_set.py --dry-run
    
    # Quiet mode
    python scripts/build_merged_training_set.py --quiet
        """
    )
    
    parser.add_argument(
        '--output-dir',
        default='./data/merged_training',
        help='Output directory for manifests and ledgers (default: ./data/merged_training)',
    )
    parser.add_argument(
        '--agi1-train',
        default='./data/arc-agi/data/training',
        help='Path to ARC-AGI-1 training tasks',
    )
    parser.add_argument(
        '--agi2-train',
        default='./data/arc-agi-2/data/training',
        help='Path to ARC-AGI-2 training tasks',
    )
    parser.add_argument(
        '--agi1-eval',
        default='./data/arc-agi/data/evaluation',
        help='Path to ARC-AGI-1 evaluation tasks',
    )
    parser.add_argument(
        '--agi2-eval',
        default='./data/arc-agi-2/data/evaluation',
        help='Path to ARC-AGI-2 evaluation tasks',
    )
    parser.add_argument(
        '--dev-ratio',
        type=float,
        default=0.05,
        help='Fraction of merged training to use as dev set (default: 0.05)',
    )
    parser.add_argument(
        '--dev-seed',
        type=int,
        default=42,
        help='Random seed for dev split (default: 42)',
    )
    parser.add_argument(
        '--near-dup-hist-threshold',
        type=float,
        default=0.95,
        help='Histogram similarity threshold for near-dup detection (default: 0.95)',
    )
    parser.add_argument(
        '--near-dup-struct-threshold',
        type=float,
        default=0.90,
        help='Structural similarity threshold for near-dup detection (default: 0.90)',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help="Don't write any files, just report what would be done",
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output',
    )
    
    args = parser.parse_args()
    
    config = BuildConfig(
        agi1_train_path=args.agi1_train,
        agi2_train_path=args.agi2_train,
        agi1_eval_path=args.agi1_eval,
        agi2_eval_path=args.agi2_eval,
        output_dir=args.output_dir,
        dev_split_ratio=args.dev_ratio,
        dev_split_seed=args.dev_seed,
        near_dup_histogram_threshold=args.near_dup_hist_threshold,
        near_dup_structure_threshold=args.near_dup_struct_threshold,
        verbose=not args.quiet,
        dry_run=args.dry_run,
    )
    
    try:
        result = build_merged_training_set(config)
        
        # Print summary for scripts to capture
        print(f"\n[RESULT] train={result.merged_train_count} dev={result.merged_dev_count} " 
              f"excluded={result.exact_dedup_count} quarantined={result.near_dup_quarantine_count} "
              f"manifest_sha256={result.manifest_sha256[:16]}...")
        
        return 0
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
