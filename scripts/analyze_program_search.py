"""
Program Search Analysis for SCI-ARC RLAN

This script analyzes how TEPS and NS-TEPS program search works
to help solve ARC puzzles, showing step-by-step program execution.
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Any, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sci_arc.models.generalization.teps import TEPS, TEPSConfig
from sci_arc.models.generalization.ns_teps import NSTEPS, NSTEPSConfig


@dataclass
class ProgramSearchResult:
    """Statistics for a program search attempt."""
    task_id: str
    search_type: str  # "TEPS" or "NS-TEPS"
    programs_searched: int
    programs_successful: int
    best_accuracy: float
    best_program_name: str
    execution_time: float
    all_programs_tried: List[Dict[str, Any]]


def load_arc_task(task_id: str, arc_dir: Path) -> Optional[Dict]:
    """Load an ARC task by ID."""
    task_file = arc_dir / f"{task_id}.json"
    if not task_file.exists():
        return None
    with open(task_file) as f:
        return json.load(f)


def compute_accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute pixel-wise accuracy."""
    if pred.shape != target.shape:
        return 0.0
    return float(np.mean(pred == target))


def analyze_teps_search(task: Dict, task_id: str) -> ProgramSearchResult:
    """Analyze TEPS program search on a task."""
    import time
    
    train = task["train"]
    test = task["test"][0]
    
    train_inputs = [np.array(ex["input"]) for ex in train]
    train_outputs = [np.array(ex["output"]) for ex in train]
    test_input = np.array(test["input"])
    test_target = np.array(test["output"])
    
    config = TEPSConfig()
    config.enabled = True
    config.timeout_seconds = 10.0
    teps = TEPS(config)
    programs_tried = []
    best_accuracy = 0.0
    best_program = "None"
    
    start_time = time.time()
    
    try:
        result = teps.search(test_input, train_inputs, train_outputs)
        
        if result['success'] and result['prediction'] is not None:
            pred = result['prediction']
            acc = compute_accuracy(pred, test_target)
            program_name = str(result.get('program', 'Unknown'))
            programs_tried.append({
                "name": program_name,
                "accuracy": acc,
                "output_shape": tuple(pred.shape)
            })
            if acc > best_accuracy:
                best_accuracy = acc
                best_program = program_name
        else:
            programs_tried.append({
                "name": "no_match",
                "stats": result.get('stats', {})
            })
    except Exception as e:
        programs_tried.append({
            "name": "error",
            "error": str(e)
        })
    
    execution_time = time.time() - start_time
    
    return ProgramSearchResult(
        task_id=task_id,
        search_type="TEPS",
        programs_searched=len(programs_tried),
        programs_successful=len([p for p in programs_tried if "accuracy" in p]),
        best_accuracy=best_accuracy,
        best_program_name=best_program,
        execution_time=execution_time,
        all_programs_tried=programs_tried
    )


def analyze_ns_teps_search(task: Dict, task_id: str) -> ProgramSearchResult:
    """Analyze NS-TEPS program search on a task."""
    import time
    
    train = task["train"]
    test = task["test"][0]
    
    train_inputs = [np.array(ex["input"]) for ex in train]
    train_outputs = [np.array(ex["output"]) for ex in train]
    test_input = np.array(test["input"])
    test_target = np.array(test["output"])
    
    config = NSTEPSConfig()
    config.enabled = True
    config.timeout_seconds = 10.0
    ns_teps = NSTEPS(config)
    programs_tried = []
    best_accuracy = 0.0
    best_program = "None"
    
    start_time = time.time()
    
    try:
        result = ns_teps.search(test_input, train_inputs, train_outputs)
        
        if result['success'] and result['prediction'] is not None:
            pred = result['prediction']
            acc = compute_accuracy(pred, test_target)
            program_name = str(result.get('program', 'Unknown'))
            programs_tried.append({
                "name": program_name,
                "accuracy": acc,
                "output_shape": tuple(pred.shape)
            })
            if acc > best_accuracy:
                best_accuracy = acc
                best_program = program_name
        else:
            programs_tried.append({
                "name": "no_match",
                "stats": result.get('stats', {})
            })
    except Exception as e:
        programs_tried.append({
            "name": "error",
            "error": str(e)
        })
    
    execution_time = time.time() - start_time
    
    return ProgramSearchResult(
        task_id=task_id,
        search_type="NS-TEPS",
        programs_searched=len(programs_tried),
        programs_successful=len([p for p in programs_tried if "accuracy" in p]),
        best_accuracy=best_accuracy,
        best_program_name=best_program,
        execution_time=execution_time,
        all_programs_tried=programs_tried
    )


def main():
    print("=" * 70)
    print("PROGRAM SEARCH ANALYSIS - TEPS & NS-TEPS")
    print("=" * 70)
    
    # Define test tasks
    test_tasks = [
        ("007bbfb7", "small", "Tiling/repetition pattern"),
        ("00d62c1b", "small", "Color flood fill"),
        ("025d127b", "medium", "Pattern completion"),
        ("0520fde7", "medium", "Column selection"),
        ("045e512c", "large", "Shape transformation"),
        ("0962bcdd", "large", "Symmetry detection"),
    ]
    
    # Find ARC data directory
    arc_dirs = [
        project_root / "data" / "arc-agi" / "data" / "training",
        project_root / "data" / "training",
        project_root / "arc-agi" / "data" / "training",
    ]
    
    arc_dir = None
    for d in arc_dirs:
        if d.exists():
            arc_dir = d
            break
    
    if arc_dir is None:
        print("ERROR: Could not find ARC data directory")
        return
    
    print(f"\nARC Data: {arc_dir}")
    
    all_results = []
    
    print("\n" + "=" * 70)
    print("TEPS SEARCH ANALYSIS")
    print("=" * 70)
    
    for task_id, size, description in test_tasks:
        task = load_arc_task(task_id, arc_dir)
        if task is None:
            print(f"\n[SKIP] Task {task_id} not found")
            continue
        
        print(f"\n--- Task: {task_id} ({size}) ---")
        print(f"Description: {description}")
        
        # Get task dimensions
        test_input = np.array(task["test"][0]["input"])
        test_output = np.array(task["test"][0]["output"])
        print(f"Input shape: {test_input.shape}")
        print(f"Output shape: {test_output.shape}")
        
        # Run TEPS search
        result = analyze_teps_search(task, task_id)
        all_results.append(asdict(result))
        
        print(f"TEPS Result:")
        print(f"  - Programs searched: {result.programs_searched}")
        print(f"  - Best program: {result.best_program_name}")
        print(f"  - Best accuracy: {result.best_accuracy:.2%}")
        print(f"  - Execution time: {result.execution_time:.3f}s")
        
        if result.all_programs_tried:
            for prog in result.all_programs_tried:
                if "error" in prog:
                    print(f"  - Error: {prog['error'][:50]}...")
                elif "accuracy" in prog:
                    print(f"  - {prog['name']}: {prog['accuracy']:.2%}")
                else:
                    print(f"  - {prog['name']}: no match")
    
    print("\n" + "=" * 70)
    print("NS-TEPS SEARCH ANALYSIS")
    print("=" * 70)
    
    for task_id, size, description in test_tasks:
        task = load_arc_task(task_id, arc_dir)
        if task is None:
            continue
        
        print(f"\n--- Task: {task_id} ({size}) ---")
        
        # Run NS-TEPS search
        result = analyze_ns_teps_search(task, task_id)
        all_results.append(asdict(result))
        
        print(f"NS-TEPS Result:")
        print(f"  - Programs searched: {result.programs_searched}")
        print(f"  - Best program: {result.best_program_name}")
        print(f"  - Best accuracy: {result.best_accuracy:.2%}")
        print(f"  - Execution time: {result.execution_time:.3f}s")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    teps_results = [r for r in all_results if r["search_type"] == "TEPS"]
    ns_teps_results = [r for r in all_results if r["search_type"] == "NS-TEPS"]
    
    teps_solved = sum(1 for r in teps_results if r["best_accuracy"] > 0.99)
    ns_teps_solved = sum(1 for r in ns_teps_results if r["best_accuracy"] > 0.99)
    
    print(f"\nTEPS:")
    print(f"  - Tasks tested: {len(teps_results)}")
    print(f"  - Tasks solved (>99% accuracy): {teps_solved}")
    print(f"  - Mean best accuracy: {np.mean([r['best_accuracy'] for r in teps_results]):.2%}")
    print(f"  - Mean execution time: {np.mean([r['execution_time'] for r in teps_results]):.3f}s")
    
    print(f"\nNS-TEPS:")
    print(f"  - Tasks tested: {len(ns_teps_results)}")
    print(f"  - Tasks solved (>99% accuracy): {ns_teps_solved}")
    print(f"  - Mean best accuracy: {np.mean([r['best_accuracy'] for r in ns_teps_results]):.2%}")
    print(f"  - Mean execution time: {np.mean([r['execution_time'] for r in ns_teps_results]):.3f}s")
    
    # Save results
    output_dir = project_root / "scripts" / "outputs" / "program_search"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "program_search_results.json", "w") as f:
        json.dump({
            "total_tasks": len(test_tasks),
            "teps_solved": teps_solved,
            "ns_teps_solved": ns_teps_solved,
            "results": all_results
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_dir / 'program_search_results.json'}")


if __name__ == "__main__":
    main()
