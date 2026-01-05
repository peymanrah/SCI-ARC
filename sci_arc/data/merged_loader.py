"""
Merged Training Set Loader for SCI-ARC RLAN
============================================

This module provides a loader for the merged training set created by
build_merged_training_set.py. It is designed to be used alongside the
existing data loading code WITHOUT modifying it.

USAGE:
    from sci_arc.data.merged_loader import get_merged_task_paths, validate_manifest_hash
    
    if config.data.use_merged_training:
        task_paths = get_merged_task_paths(config.data.merged_training_path)
        # Use task_paths instead of loading from train_path
    else:
        # Use existing train_path loading (unchanged behavior)

This module is STANDALONE and does NOT modify the existing dataset.py.
The training script should check the config flag and choose which loader to use.

Author: SCI-ARC Team
Date: January 2026
"""

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class ManifestRecord:
    """Record from the merged training manifest."""
    task_uid: str
    source: str
    path: str
    raw_sha256: str
    canonical_sha256: str
    num_train_pairs: int
    max_grid_size: int
    has_size_change: bool


def load_manifest(manifest_path: str) -> List[ManifestRecord]:
    """
    Load records from a JSONL manifest file.
    
    Args:
        manifest_path: Path to the manifest JSONL file
        
    Returns:
        List of ManifestRecord objects
    """
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    
    records = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            records.append(ManifestRecord(
                task_uid=data['task_uid'],
                source=data['source'],
                path=data['path'],
                raw_sha256=data['raw_sha256'],
                canonical_sha256=data['canonical_sha256'],
                num_train_pairs=data['num_train_pairs'],
                max_grid_size=data['max_grid_size'],
                has_size_change=data['has_size_change'],
            ))
    
    return records


def get_merged_task_paths(
    merged_training_path: str,
    manifest_name: str = "merged_train_manifest.jsonl",
) -> List[str]:
    """
    Get list of task file paths from the merged training manifest.
    
    Args:
        merged_training_path: Path to the merged training directory
        manifest_name: Name of the manifest file (default: merged_train_manifest.jsonl)
        
    Returns:
        List of absolute paths to task JSON files
    """
    manifest_path = Path(merged_training_path) / manifest_name
    records = load_manifest(str(manifest_path))
    
    return [record.path for record in records]


def get_merged_task_info(
    merged_training_path: str,
    manifest_name: str = "merged_train_manifest.jsonl",
) -> Tuple[List[str], Dict[str, str]]:
    """
    Get list of task file paths and a mapping from path to unique task_uid.
    
    This function solves the task ID collision problem where AGI-1 and AGI-2
    can have tasks with the same filename (e.g., 'f0f8a26d.json' exists in both).
    The task_uid includes the source prefix (e.g., 'agi1_train:f0f8a26d') making
    it globally unique.
    
    Args:
        merged_training_path: Path to the merged training directory
        manifest_name: Name of the manifest file (default: merged_train_manifest.jsonl)
        
    Returns:
        Tuple of:
            - List of absolute paths to task JSON files
            - Dict mapping file path -> task_uid for unique identification
    """
    manifest_path = Path(merged_training_path) / manifest_name
    records = load_manifest(str(manifest_path))
    
    paths = [record.path for record in records]
    path_to_uid = {record.path: record.task_uid for record in records}
    
    return paths, path_to_uid


def get_merged_dev_paths(merged_training_path: str) -> List[str]:
    """
    Get list of task file paths from the merged dev manifest.
    
    Args:
        merged_training_path: Path to the merged training directory
        
    Returns:
        List of absolute paths to dev task JSON files
    """
    return get_merged_task_paths(
        merged_training_path,
        manifest_name="merged_dev_manifest.jsonl"
    )


def compute_manifest_hash(manifest_path: str) -> str:
    """
    Compute SHA-256 hash of a manifest file.
    
    Args:
        manifest_path: Path to the manifest file
        
    Returns:
        SHA-256 hex digest
    """
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    
    with open(path, 'rb') as f:
        content = f.read()
    
    return hashlib.sha256(content).hexdigest()


def validate_manifest_hash(
    merged_training_path: str,
    expected_hash: Optional[str] = None,
) -> Tuple[bool, str, Optional[str]]:
    """
    Validate the manifest hash against an expected value or return current hash.
    
    This is used to ensure caches/HPM buffers were built with the same
    training set version.
    
    Args:
        merged_training_path: Path to the merged training directory
        expected_hash: If provided, validate against this hash
        
    Returns:
        Tuple of (is_valid, current_hash, expected_hash)
        If expected_hash is None, is_valid is always True
    """
    manifest_path = Path(merged_training_path) / "merged_train_manifest.jsonl"
    
    try:
        current_hash = compute_manifest_hash(str(manifest_path))
    except FileNotFoundError:
        return (False, "", expected_hash)
    
    if expected_hash is None:
        return (True, current_hash, None)
    
    is_valid = current_hash == expected_hash
    return (is_valid, current_hash, expected_hash)


def load_build_metadata(merged_training_path: str) -> Dict:
    """
    Load build metadata from the merged training directory.
    
    Args:
        merged_training_path: Path to the merged training directory
        
    Returns:
        Dict containing build metadata
    """
    metadata_path = Path(merged_training_path) / "build_metadata.json"
    
    if not metadata_path.exists():
        raise FileNotFoundError(f"Build metadata not found: {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        return json.load(f)


def get_training_stats(merged_training_path: str) -> Dict:
    """
    Get statistics about the merged training set.
    
    Args:
        merged_training_path: Path to the merged training directory
        
    Returns:
        Dict with training set statistics
    """
    metadata = load_build_metadata(merged_training_path)
    stats = metadata.get('stats', {})
    
    return {
        'total_train_tasks': stats.get('final_train_count', 0),
        'total_dev_tasks': stats.get('final_dev_count', 0),
        'agi1_train_original': stats.get('agi1_train_loaded', 0),
        'agi2_train_original': stats.get('agi2_train_loaded', 0),
        'exact_dups_removed': stats.get('exact_dups_removed', 0),
        'internal_dups_removed': stats.get('internal_dups_removed', 0),
        'near_dups_quarantined': stats.get('near_dups_quarantined', 0),
        'manifest_sha256': metadata.get('manifest_sha256', ''),
        'build_timestamp': metadata.get('build_timestamp', ''),
    }


def load_tasks_from_manifest(
    merged_training_path: str,
    manifest_name: str = "merged_train_manifest.jsonl",
    max_tasks: Optional[int] = None,
    stratified_seed: int = 42,
) -> List[Dict]:
    """
    Load task dictionaries from a manifest.
    
    This function loads the actual task JSON files referenced in the manifest.
    It can optionally limit to max_tasks with stratified sampling.
    
    Args:
        merged_training_path: Path to the merged training directory
        manifest_name: Name of the manifest file
        max_tasks: If set, limit to this many tasks (stratified sampling)
        stratified_seed: Seed for stratified sampling
        
    Returns:
        List of task dictionaries with 'task_id', 'train', 'test' keys
    """
    manifest_path = Path(merged_training_path) / manifest_name
    records = load_manifest(str(manifest_path))
    
    # Apply stratified sampling if max_tasks is set
    if max_tasks is not None and max_tasks < len(records):
        import random
        from collections import defaultdict
        
        rng = random.Random(stratified_seed)
        
        # Stratify by size bucket and pair count
        def get_bucket(record: ManifestRecord) -> str:
            size = "small" if record.max_grid_size <= 10 else (
                "medium" if record.max_grid_size <= 20 else "large"
            )
            pairs = "few" if record.num_train_pairs <= 2 else (
                "medium" if record.num_train_pairs <= 4 else "many"
            )
            change = "change" if record.has_size_change else "same"
            return f"{size}_{pairs}_{change}"
        
        buckets: Dict[str, List[ManifestRecord]] = defaultdict(list)
        for record in records:
            buckets[get_bucket(record)].append(record)
        
        # Sample proportionally from each bucket
        sampled = []
        for bucket_name, bucket_records in buckets.items():
            rng.shuffle(bucket_records)
            n_sample = max(1, int(len(bucket_records) * max_tasks / len(records)))
            sampled.extend(bucket_records[:n_sample])
        
        # Trim to exact count if needed
        rng.shuffle(sampled)
        records = sampled[:max_tasks]
    
    # Load actual task files
    tasks = []
    for record in records:
        try:
            with open(record.path, 'r') as f:
                task = json.load(f)
            
            # Ensure task_id is set
            if 'task_id' not in task:
                task['task_id'] = record.task_uid
            
            tasks.append(task)
        except Exception as e:
            print(f"Warning: Failed to load {record.path}: {e}")
    
    return tasks


# =============================================================================
# Integration helpers for training script
# =============================================================================

def should_use_merged_training(config_data: Dict) -> bool:
    """
    Check if merged training should be used based on config.
    
    Args:
        config_data: The 'data' section of the config
        
    Returns:
        True if use_merged_training is enabled and path exists
    """
    use_merged = config_data.get('use_merged_training', False)
    
    if not use_merged:
        return False
    
    merged_path = config_data.get('merged_training_path', './data/merged_training')
    manifest_path = Path(merged_path) / "merged_train_manifest.jsonl"
    
    if not manifest_path.exists():
        print(f"WARNING: use_merged_training=true but manifest not found: {manifest_path}")
        print("         Run: python scripts/build_merged_training_set.py")
        print("         Falling back to standard train_path")
        return False
    
    return True


def print_merged_training_info(merged_training_path: str):
    """
    Print info about the merged training set for logging.
    """
    try:
        stats = get_training_stats(merged_training_path)
        
        print("=" * 60)
        print("MERGED TRAINING SET (ARC-AGI-1 + ARC-AGI-2)")
        print("=" * 60)
        print(f"  Train tasks:        {stats['total_train_tasks']}")
        print(f"  Dev tasks:          {stats['total_dev_tasks']}")
        print(f"  Original AGI-1:     {stats['agi1_train_original']}")
        print(f"  Original AGI-2:     {stats['agi2_train_original']}")
        print(f"  Exact dups removed: {stats['exact_dups_removed']}")
        print(f"  Near dups quarant:  {stats['near_dups_quarantined']}")
        print(f"  Manifest hash:      {stats['manifest_sha256'][:16]}...")
        print(f"  Build timestamp:    {stats['build_timestamp']}")
        print("=" * 60)
        
    except Exception as e:
        print(f"WARNING: Could not load merged training stats: {e}")
