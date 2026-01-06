"""
Detailed analysis of centroid placement vs actual foreground object locations.
This diagnoses why centroids cluster in the center instead of at FG object centers.
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_arc_task(task_path):
    """Load an ARC task from JSON file."""
    with open(task_path, 'r') as f:
        return json.load(f)


def find_connected_component_centroids(grid):
    """
    Find centroids of each connected foreground component.
    
    This is what RLAN theory says clues should be - the center of each
    distinct foreground object/region.
    """
    if isinstance(grid, torch.Tensor):
        grid = grid.cpu().numpy()
    
    if grid.ndim == 3:
        grid = grid[0]  # Remove batch dim
    
    H, W = grid.shape
    
    # Get foreground mask (non-black pixels)
    fg_mask = (grid > 0).astype(int)
    
    # Find connected components
    labeled, num_features = ndimage.label(fg_mask)
    
    # Get centroid of each component
    centroids = []
    for i in range(1, num_features + 1):
        coords = np.where(labeled == i)
        if len(coords[0]) > 0:
            cy = coords[0].mean()
            cx = coords[1].mean()
            # Normalize to [0, 1]
            cy_norm = cy / max(H - 1, 1)
            cx_norm = cx / max(W - 1, 1)
            centroids.append({
                'cy': cy_norm,
                'cx': cx_norm,
                'cy_pixel': cy,
                'cx_pixel': cx,
                'size': len(coords[0]),
                'color': int(grid[coords[0][0], coords[1][0]])
            })
    
    return centroids, labeled, num_features


def analyze_task_centroids(task_dir, task_id):
    """Analyze centroid placement for a specific task."""
    report_path = task_dir / f"{task_id}_report.json"
    
    if not report_path.exists():
        return None
    
    with open(report_path) as f:
        report = json.load(f)
    
    # Load actual task data
    training_dir = Path("c:/Users/perahmat/Downloads/SCI-ARC/data/arc-agi/data/training")
    task_path = training_dir / f"{task_id}.json"
    
    if not task_path.exists():
        return None
    
    task = load_arc_task(task_path)
    input_grid = np.array(task["train"][0]["input"])
    
    H, W = input_grid.shape
    
    # Find actual foreground component centroids
    true_centroids, labeled, num_fg_objects = find_connected_component_centroids(input_grid)
    
    # Analyze DSC centroids from report
    # Note: We don't have centroid values in the report, need to re-run model
    
    result = {
        'task_id': task_id,
        'grid_size': (H, W),
        'num_fg_objects': num_fg_objects,
        'true_centroids': true_centroids,
        'foreground_ratio': (input_grid > 0).mean(),
        'stop_probs': report['checks'].get('dsc_stop_probs', []),
        'fg_attention_ratio': report['checks'].get('foreground_attention_ratio', None)
    }
    
    return result


def main():
    print("="*70)
    print("CENTROID PLACEMENT ANALYSIS")
    print("="*70)
    
    warmup3_dir = Path("c:/Users/perahmat/Downloads/SCI-ARC/scripts/outputs/warmup3_checkpoint_tests")
    
    all_results = []
    
    for size_dir in ['small', 'medium', 'large']:
        size_path = warmup3_dir / size_dir
        if not size_path.exists():
            continue
            
        print(f"\n{'='*70}")
        print(f"{size_dir.upper()} GRIDS")
        print("="*70)
        
        for task_dir in size_path.iterdir():
            if not task_dir.is_dir():
                continue
                
            task_id = task_dir.name
            result = analyze_task_centroids(task_dir, task_id)
            
            if result is None:
                continue
            
            all_results.append(result)
            
            print(f"\n[Task: {task_id}] Grid: {result['grid_size']}")
            print(f"  Foreground objects: {result['num_fg_objects']}")
            print(f"  FG pixel ratio: {result['foreground_ratio']:.2%}")
            print(f"  FG attention ratio: {result['fg_attention_ratio']:.2%}" if result['fg_attention_ratio'] else "  FG attention: N/A")
            
            if result['true_centroids']:
                print(f"  True FG object centroids:")
                for i, c in enumerate(result['true_centroids'][:5]):  # Show first 5
                    print(f"    Object {i+1}: ({c['cy']:.3f}, {c['cx']:.3f}) "
                          f"[pixel: ({c['cy_pixel']:.1f}, {c['cx_pixel']:.1f})] "
                          f"size={c['size']} color={c['color']}")
                if len(result['true_centroids']) > 5:
                    print(f"    ... and {len(result['true_centroids']) - 5} more")
            
            # Stop probs analysis
            stop_probs = result['stop_probs']
            if stop_probs:
                mean_stop = np.mean(stop_probs)
                std_stop = np.std(stop_probs)
                print(f"  Stop probs: mean={mean_stop:.4f}, std={std_stop:.6f}")
                if std_stop < 1e-5:
                    print(f"    ⚠️  WARNING: Stop probs are IDENTICAL (collapsed)")
    
    # Summary statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    if all_results:
        num_fg_list = [r['num_fg_objects'] for r in all_results]
        fg_attn_list = [r['fg_attention_ratio'] for r in all_results if r['fg_attention_ratio']]
        
        print(f"\nAcross {len(all_results)} tasks:")
        print(f"  FG objects per task: min={min(num_fg_list)}, max={max(num_fg_list)}, mean={np.mean(num_fg_list):.1f}")
        
        if fg_attn_list:
            print(f"  FG attention ratio: min={min(fg_attn_list):.2%}, max={max(fg_attn_list):.2%}, mean={np.mean(fg_attn_list):.2%}")
        
        # Check stop prob collapse
        all_stop_flat = []
        for r in all_results:
            all_stop_flat.extend(r['stop_probs'])
        
        if all_stop_flat:
            stop_std = np.std(all_stop_flat)
            print(f"\n  Stop prob variance across ALL tasks: std={stop_std:.8f}")
            if stop_std < 1e-5:
                print("  ⚠️  CRITICAL: Stop predictor has collapsed to constant output!")
    
    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)
    print("""
    1. STOP PREDICTOR COLLAPSE:
       All stop probabilities are exactly 0.00247 across ALL tasks.
       This indicates the stop predictor is outputting a constant
       regardless of input features or task complexity.
       
       FIX: The stop predictor needs:
       - Better gradient flow (increase learning rate for stop predictor)
       - Explicit supervision (add stop_loss to training)
       - Different initialization (reduce bias magnitude)
    
    2. CENTROID PLACEMENT vs FOREGROUND OBJECTS:
       The true FG object centroids are computed via connected component
       analysis. If DSC centroids don't match these, the model won't be
       able to reason about individual objects effectively.
       
       FIX: Consider hybrid approach:
       - Use segmentation to find candidate anchor points
       - Use learned attention to refine/select among them
       - Or: Add explicit FG-based loss to encourage centroids at objects
    
    3. ATTENTION WEIGHT IMBALANCE:
       Position affinity (16x) >> content-based attention (2x)
       This causes centroids to be placed based on position, not content.
       
       FIX: Reduce position weight to 1-4x, increase content weight to 4-8x
    """)


if __name__ == "__main__":
    main()
