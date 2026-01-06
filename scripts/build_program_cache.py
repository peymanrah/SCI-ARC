#!/usr/bin/env python3
"""
Build Program Cache for Program-Guided Training
================================================
This script pre-mines NS-TEPS programs for all training tasks and saves them
to a program cache file. This is REQUIRED before training with program_guided.enabled=true
when online_mining=false.

Usage:
    # For 602 merged tasks:
    python scripts/build_program_cache.py --output cache/program_cache_merged_602.json --use-merged

    # For 400 original tasks:
    python scripts/build_program_cache.py --output cache/program_cache_400.json

    # With custom config:
    python scripts/build_program_cache.py --config configs/rlan_stable_dev_merged.yaml --output cache/program_cache_merged_602.json

The script will:
1. Load all training tasks (merged or original)
2. Run NS-TEPS on each task to find a valid program
3. Cache successful programs with task_id, trace, confidence, and input_hash
4. Save the cache to JSON for use during training

PERFORMANCE:
- ~2-5 seconds per task with timeout_seconds=2.0
- 602 tasks ≈ 20-50 minutes
- Can be parallelized (--num-workers flag)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import yaml
except ImportError:
    print("Error: PyYAML is required. Install with: pip install pyyaml")
    sys.exit(1)


def load_merged_tasks(merged_path: str) -> List[Dict[str, Any]]:
    """Load tasks from merged training manifest."""
    manifest_path = Path(merged_path) / "merged_train_manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Merged manifest not found at {manifest_path}. "
            "Run: python scripts/build_merged_training_set.py first."
        )
    
    tasks = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                tasks.append(json.loads(line))
    
    print(f"Loaded {len(tasks)} tasks from merged manifest")
    return tasks


def load_original_tasks(train_path: str) -> List[Dict[str, Any]]:
    """Load tasks from original ARC training directory."""
    train_dir = Path(train_path)
    if not train_dir.exists():
        raise FileNotFoundError(f"Training directory not found: {train_path}")
    
    tasks = []
    for task_file in sorted(train_dir.glob("*.json")):
        with open(task_file, 'r', encoding='utf-8') as f:
            task_data = json.load(f)
            task_id = task_file.stem
            tasks.append({
                'task_id': task_id,
                'task_path': str(task_file),
                'train': task_data.get('train', [])
            })
    
    print(f"Loaded {len(tasks)} tasks from {train_path}")
    return tasks


def load_task_data(task_info: Dict) -> Dict:
    """Load full task data from file."""
    # Handle both 'task_path' (original) and 'path' (merged manifest) keys
    task_path = task_info.get('task_path') or task_info.get('path')
    if task_path:
        with open(task_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    elif 'train' in task_info:
        return task_info
    else:
        raise ValueError(f"Invalid task info format: missing 'path', 'task_path', or 'train' key")


def mine_single_task(task_info: Dict, nsteps_config: Dict) -> Optional[Dict]:
    """
    Mine a program for a single task.
    
    Returns dict with task_id, trace, confidence, input_hash if successful.
    """
    try:
        # Import NS-TEPS here to avoid multiprocessing import issues
        from sci_arc.models.generalization.ns_teps import NSTEPS, NSTEPSConfig

        # Create NS-TEPS searcher
        nsteps = NSTEPS(
            NSTEPSConfig(
                enabled=True,
                max_search_steps=nsteps_config.get('max_search_steps', 500),
                timeout_seconds=nsteps_config.get('timeout_seconds', 2.0),
                max_trace_length=nsteps_config.get('max_trace_length', 3),
                min_object_size=nsteps_config.get('min_object_size', 1),
                max_objects=nsteps_config.get('max_objects', 20),
                sample_count=nsteps_config.get('sample_count', 200),
                match_threshold=nsteps_config.get('match_threshold', 0.95),
            )
        )
        
        # Load task data
        task_data = load_task_data(task_info)
        # Handle both 'task_id' (original) and 'task_uid' (merged manifest)
        task_id = task_info.get('task_id') or task_info.get('task_uid', 'unknown')
        
        train_pairs = task_data.get('train', [])
        if not train_pairs:
            return None
        
        # Convert to numpy arrays
        train_inputs = [np.array(p['input']) for p in train_pairs]
        train_outputs = [np.array(p['output']) for p in train_pairs]
        
        # Compute stable input hash
        input_hash = hashlib.sha256(train_inputs[0].tobytes()).hexdigest()[:16]
        
        # Run NS-TEPS search
        result = nsteps.search(
            test_input=train_inputs[0],
            train_inputs=train_inputs,
            train_outputs=train_outputs,
        )
        
        if result.get('success', False) and result.get('trace') is not None:
            trace = result['trace']
            confidence = result.get('confidence', 1.0)
            
            if confidence >= nsteps_config.get('match_threshold', 0.95):
                # Convert ProgramTrace to serializable format
                trace_list = []
                for step in trace.steps:
                    prim = step[0]
                    params = step[1]
                    prim_name = prim.name if hasattr(prim, 'name') else str(prim)
                    trace_list.append((prim_name, params))
                
                return {
                    'task_id': task_id,
                    'trace': trace_list,
                    'confidence': confidence,
                    'input_hash': input_hash,
                }
        
        return None
        
    except Exception as e:
        print(f"[WARNING] Mining failed for {task_info.get('task_id', 'unknown')}: {e}")
        return None


def build_program_cache(
    tasks: List[Dict],
    nsteps_config: Dict,
    output_path: str,
    num_workers: int = 1,
    verbose: bool = True,
) -> Dict:
    """Build program cache for all tasks.

    Output format matches ProgramCache.load():
      { task_id: {trace, confidence, input_hash, timestamp}, ... }
    """

    cache: Dict[str, Dict[str, Any]] = {}
    
    successful = 0
    failed = 0
    start_time = time.time()
    
    if num_workers == 1:
        # Single-threaded execution
        for i, task_info in enumerate(tasks):
            # Handle both 'task_id' (original) and 'task_uid' (merged manifest)
            task_id = task_info.get('task_id') or task_info.get('task_uid', f'task_{i}')
            result = mine_single_task(task_info, nsteps_config)
            
            if result is not None:
                cache[task_id] = {
                    'trace': result['trace'],
                    'confidence': float(result.get('confidence', 1.0)),
                    'input_hash': result.get('input_hash', ''),
                    'timestamp': time.time(),
                }
                successful += 1
                if verbose:
                    print(f"[{i+1}/{len(tasks)}] [OK] {task_id} (conf={result['confidence']:.2f})")
            else:
                failed += 1
                if verbose:
                    print(f"[{i+1}/{len(tasks)}] [FAIL] {task_id} - no program found")
    else:
        # Multi-process execution
        print(f"Using {num_workers} workers for parallel mining...")
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(mine_single_task, task_info, nsteps_config): task_info
                for task_info in tasks
            }
            
            for i, future in enumerate(as_completed(futures)):
                task_info = futures[future]
                # Handle both 'task_id' (original) and 'task_uid' (merged manifest)
                task_id = task_info.get('task_id') or task_info.get('task_uid', 'unknown')
                
                try:
                    result = future.result()
                    if result is not None:
                        cache[task_id] = {
                            'trace': result['trace'],
                            'confidence': float(result.get('confidence', 1.0)),
                            'input_hash': result.get('input_hash', ''),
                            'timestamp': time.time(),
                        }
                        successful += 1
                        if verbose:
                            print(f"[{i+1}/{len(tasks)}] [OK] {task_id}")
                    else:
                        failed += 1
                        if verbose:
                            print(f"[{i+1}/{len(tasks)}] [FAIL] {task_id}")
                except Exception as e:
                    failed += 1
                    print(f"[{i+1}/{len(tasks)}] [ERR] {task_id}: {e}")
    
    elapsed = time.time() - start_time
    
    # Save cache
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(cache, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Program Cache Built Successfully")
    print(f"{'='*60}")
    print(f"  Total tasks:    {len(tasks)}")
    print(f"  Successful:     {successful} ({100*successful/len(tasks):.1f}%)")
    print(f"  Failed:         {failed} ({100*failed/len(tasks):.1f}%)")
    print(f"  Time elapsed:   {elapsed:.1f}s ({elapsed/len(tasks):.2f}s per task)")
    print(f"  Output:         {output_path}")
    print(f"{'='*60}\n")
    
    return cache


def main():
    parser = argparse.ArgumentParser(
        description="Build program cache for program-guided training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='./cache/program_cache_merged_602.json',
        help='Output path for program cache JSON file'
    )
    
    parser.add_argument(
        '--use-merged',
        action='store_true',
        help='Use merged training set (602 tasks). Requires build_merged_training_set.py first.'
    )
    
    parser.add_argument(
        '--train-path',
        type=str,
        default='./data/arc-agi/data/training',
        help='Path to original ARC training directory (used when --use-merged is False)'
    )
    
    parser.add_argument(
        '--merged-path',
        type=str,
        default='./data/merged_training',
        help='Path to merged training manifest directory'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Optional: Load NS-TEPS config from YAML file'
    )
    
    parser.add_argument(
        '--max-steps',
        type=int,
        default=500,
        help='Max search steps for NS-TEPS'
    )
    
    parser.add_argument(
        '--timeout',
        type=float,
        default=2.0,
        help='Timeout per task in seconds'
    )
    
    parser.add_argument(
        '--max-trace-length',
        type=int,
        default=3,
        help='Max primitive sequence length'
    )
    
    parser.add_argument(
        '--min-confidence',
        type=float,
        default=0.95,
        help='Minimum confidence to accept program'
    )
    
    parser.add_argument(
        '--num-workers',
        type=int,
        default=1,
        help='Number of parallel workers (1=single-threaded)'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress per-task output'
    )
    
    args = parser.parse_args()
    
    # Build NS-TEPS config
    nsteps_config = {
        'max_search_steps': args.max_steps,
        'timeout_seconds': args.timeout,
        'max_trace_length': args.max_trace_length,
        'match_threshold': args.min_confidence,
    }
    
    # Override from YAML config if provided
    if args.config:
        with open(args.config, 'r', encoding='utf-8') as f:
            yaml_config = yaml.safe_load(f)
        
        pg_config = yaml_config.get('program_guided', {})
        nsteps_yaml = pg_config.get('nsteps', {})
        
        if nsteps_yaml:
            nsteps_config.update({
                'max_search_steps': nsteps_yaml.get('max_search_steps', nsteps_config['max_search_steps']),
                'timeout_seconds': nsteps_yaml.get('timeout_seconds', nsteps_config['timeout_seconds']),
                'max_trace_length': nsteps_yaml.get('max_trace_length', nsteps_config['max_trace_length']),
                'match_threshold': nsteps_yaml.get('match_threshold', nsteps_config['match_threshold']),
            })
            print(f"Loaded NS-TEPS config from {args.config}")
    
    print(f"NS-TEPS Configuration:")
    for k, v in nsteps_config.items():
        print(f"  {k}: {v}")
    print()
    
    # Load tasks
    if args.use_merged:
        tasks = load_merged_tasks(args.merged_path)
    else:
        tasks = load_original_tasks(args.train_path)
    
    # Build cache
    build_program_cache(
        tasks=tasks,
        nsteps_config=nsteps_config,
        output_path=args.output,
        num_workers=args.num_workers,
        verbose=not args.quiet,
    )


if __name__ == '__main__':
    main()
