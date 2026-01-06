"""
Test RLAN modules with trained checkpoint on real ARC training data.
Generates visualizations for each module's input/output organized by task ID and grid size.
"""

import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sci_arc.models.rlan import RLAN, RLANConfig

# ARC color palette
ARC_COLORS = [
    '#000000',  # 0: black
    '#0074D9',  # 1: blue
    '#FF4136',  # 2: red
    '#2ECC40',  # 3: green
    '#FFDC00',  # 4: yellow
    '#AAAAAA',  # 5: gray
    '#F012BE',  # 6: magenta
    '#FF851B',  # 7: orange
    '#7FDBFF',  # 8: cyan
    '#870C25',  # 9: maroon
]


def get_arc_cmap():
    """Get a colormap for ARC grids."""
    from matplotlib.colors import ListedColormap
    return ListedColormap(ARC_COLORS)


def load_arc_task(task_path):
    """Load an ARC task from JSON file."""
    with open(task_path, 'r') as f:
        task = json.load(f)
    return task


def get_grid_size_category(h, w):
    """Categorize grid by size."""
    max_dim = max(h, w)
    if max_dim <= 10:
        return "small"
    elif max_dim <= 20:
        return "medium"
    else:
        return "large"


def find_tasks_by_size(data_dir, target_sizes=("small", "medium", "large"), max_per_size=3):
    """Find ARC tasks categorized by grid size."""
    tasks_by_size = {size: [] for size in target_sizes}
    
    json_files = list(Path(data_dir).glob("*.json"))
    
    for task_path in json_files:
        try:
            task = load_arc_task(task_path)
            # Get first training example's input grid size
            if task.get("train") and len(task["train"]) > 0:
                grid = task["train"][0]["input"]
                h, w = len(grid), len(grid[0])
                category = get_grid_size_category(h, w)
                
                if category in tasks_by_size and len(tasks_by_size[category]) < max_per_size:
                    tasks_by_size[category].append({
                        "path": task_path,
                        "task_id": task_path.stem,
                        "grid_size": (h, w),
                        "task": task
                    })
        except Exception as e:
            continue
    
    return tasks_by_size


def load_model_with_checkpoint(checkpoint_path, device='cpu'):
    """Load RLAN model with trained checkpoint."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get model config from checkpoint (nested under 'model' key)
    full_config = checkpoint.get('config', {})
    model_config = full_config.get('model', {})
    
    # Create RLANConfig from checkpoint config
    config = RLANConfig(
        hidden_dim=model_config.get('hidden_dim', 256),
        num_colors=model_config.get('num_colors', 10),
        num_classes=model_config.get('num_classes', 10),
        max_grid_size=model_config.get('max_grid_size', 30),
        max_clues=model_config.get('max_clues', 7),
        num_predicates=model_config.get('num_predicates', 32),
        num_solver_steps=model_config.get('num_solver_steps', 6),
        use_act=model_config.get('use_act', False),
        dropout=model_config.get('dropout', 0.1),
        use_context_encoder=model_config.get('use_context_encoder', True),
        use_dsc=model_config.get('use_dsc', True),
        use_msre=model_config.get('use_msre', True),
        use_lcr=model_config.get('use_lcr', False),
        use_sph=model_config.get('use_sph', False),
        use_solver_context=model_config.get('use_solver_context', True),
        use_cross_attention_context=model_config.get('use_cross_attention_context', True),
        dsc_num_heads=model_config.get('dsc_num_heads', 4),
        msre_encoding_dim=model_config.get('msre_encoding_dim', 32),
        msre_num_freq=model_config.get('msre_num_freq', 8),
    )
    
    # Create model
    model = RLAN(config=config)
    
    # Load weights with strict=False to handle architecture differences
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    if missing_keys:
        print(f"  Warning: Missing keys: {len(missing_keys)} (these use random init)")
        for k in missing_keys[:5]:
            print(f"    - {k}")
        if len(missing_keys) > 5:
            print(f"    ... and {len(missing_keys) - 5} more")
    
    if unexpected_keys:
        print(f"  Warning: Unexpected keys: {len(unexpected_keys)} (ignored from checkpoint)")
        for k in unexpected_keys[:5]:
            print(f"    - {k}")
        if len(unexpected_keys) > 5:
            print(f"    ... and {len(unexpected_keys) - 5} more")
    
    model.to(device)
    model.eval()
    
    print(f"  Model: hidden_dim={config.hidden_dim}, max_clues={config.max_clues}")
    print(f"  Loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    
    return model, model_config


def visualize_grid(ax, grid, title, cmap=None):
    """Visualize an ARC grid."""
    if cmap is None:
        cmap = get_arc_cmap()
    
    if isinstance(grid, torch.Tensor):
        grid = grid.cpu().numpy()
    
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=9)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add grid lines
    h, w = grid.shape
    for i in range(h + 1):
        ax.axhline(i - 0.5, color='white', linewidth=0.5)
    for j in range(w + 1):
        ax.axvline(j - 0.5, color='white', linewidth=0.5)


def visualize_attention_map(ax, attn_map, centroids, title, grid_shape):
    """Visualize attention map with centroid markers."""
    if isinstance(attn_map, torch.Tensor):
        attn_map = attn_map.cpu().numpy()
    if isinstance(centroids, torch.Tensor):
        centroids = centroids.cpu().numpy()
    
    im = ax.imshow(attn_map, cmap='hot', vmin=0)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Plot centroid
    H, W = grid_shape
    cy, cx = centroids[0] * (H - 1), centroids[1] * (W - 1)
    ax.plot(cx, cy, 'c+', markersize=12, markeredgewidth=2)
    ax.plot(cx, cy, 'co', markersize=8, markerfacecolor='none', markeredgewidth=1)


def visualize_features(ax, features, title):
    """Visualize feature map (mean across channels)."""
    if isinstance(features, torch.Tensor):
        features = features.cpu().numpy()
    
    # Mean across channel dimension
    if len(features.shape) == 3:
        feat_vis = features.mean(axis=0)
    else:
        feat_vis = features
    
    im = ax.imshow(feat_vis, cmap='viridis')
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def test_single_task(model, task_info, output_dir, device='cpu'):
    """Test RLAN modules on a single ARC task and generate visualizations.
    
    Uses the FULL production forward pass with training pairs (context encoder).
    This mimics exactly how RLAN is called during training.
    """
    task_id = task_info["task_id"]
    task = task_info["task"]
    h, w = task_info["grid_size"]
    
    print(f"\n  Testing task {task_id} ({h}x{w})")
    
    # Get ALL training examples (not just the first one)
    train_pairs = task["train"]
    num_pairs = len(train_pairs)
    
    # Find max dimensions across all pairs
    max_h = max(len(pair["input"]) for pair in train_pairs)
    max_w = max(len(pair["input"][0]) for pair in train_pairs)
    max_h_out = max(len(pair["output"]) for pair in train_pairs)
    max_w_out = max(len(pair["output"][0]) for pair in train_pairs)
    
    # Use the larger of input/output sizes
    pad_h = max(max_h, max_h_out, h)
    pad_w = max(max_w, max_w_out, w)
    
    # Pad grids to uniform size
    def pad_grid(grid, target_h, target_w, pad_value=0):
        """Pad a grid to target dimensions."""
        arr = np.array(grid)
        curr_h, curr_w = arr.shape
        padded = np.full((target_h, target_w), pad_value, dtype=np.int64)
        padded[:curr_h, :curr_w] = arr
        return padded
    
    # Prepare training inputs and outputs
    train_inputs_list = []
    train_outputs_list = []
    for pair in train_pairs:
        train_inputs_list.append(pad_grid(pair["input"], pad_h, pad_w))
        train_outputs_list.append(pad_grid(pair["output"], pad_h, pad_w))
    
    train_inputs = torch.tensor(np.stack(train_inputs_list), dtype=torch.long).unsqueeze(0).to(device)  # (1, N, H, W)
    train_outputs = torch.tensor(np.stack(train_outputs_list), dtype=torch.long).unsqueeze(0).to(device)  # (1, N, H, W)
    pair_mask = torch.ones(1, num_pairs, device=device)  # (1, N) all pairs valid
    
    # Prepare test input (use first training input as "test" for visualization)
    input_grid = pad_grid(task["train"][0]["input"], pad_h, pad_w)
    target_grid = pad_grid(task["train"][0]["output"], pad_h, pad_w)
    x = torch.tensor(input_grid, dtype=torch.long).unsqueeze(0).to(device)  # (1, H, W)
    
    results = {
        "task_id": task_id,
        "grid_size": (h, w),
        "num_train_pairs": num_pairs,
        "checks": {}
    }
    
    with torch.no_grad():
        # PRODUCTION FORWARD PASS - exactly as in training!
        # This uses context encoder with training pairs
        outputs = model(
            input_grid=x,
            train_inputs=train_inputs,
            train_outputs=train_outputs,
            pair_mask=pair_mask,
            temperature=1.0,
            return_intermediates=True
        )
        
        # Extract outputs
        logits = outputs.get('logits', outputs) if isinstance(outputs, dict) else outputs
        features = outputs.get('features') if isinstance(outputs, dict) else None
        centroids = outputs.get('centroids') if isinstance(outputs, dict) else None
        attn_maps = outputs.get('attention_maps') if isinstance(outputs, dict) else None
        stop_logits = outputs.get('stop_logits') if isinstance(outputs, dict) else None
        context = outputs.get('context') if isinstance(outputs, dict) else None
        
        # If outputs is just logits (return_intermediates not working), fall back
        if features is None:
            features = model.encode(x)
        
        predictions = logits.argmax(dim=1)  # (B, H, W) - argmax over class dimension
        
        # Record checks
        results["checks"]["encoder_output_shape"] = list(features.shape)
        results["checks"]["encoder_output_valid"] = not (torch.isnan(features).any() or torch.isinf(features).any())
        results["checks"]["output_shape"] = list(logits.shape)
        results["checks"]["prediction_shape"] = list(predictions.shape)
        results["checks"]["unique_colors_predicted"] = predictions.unique().tolist()
        
        # Context encoder check
        results["checks"]["context_encoder_used"] = context is not None
        if context is not None:
            results["checks"]["context_shape"] = list(context.shape) if hasattr(context, 'shape') else str(type(context))
        
        # DSC checks
        if centroids is not None:
            results["checks"]["dsc_centroids_shape"] = list(centroids.shape)
            results["checks"]["dsc_centroids_in_range"] = bool((centroids >= 0).all() and (centroids <= 1).all())
        else:
            # Fall back to direct DSC call for visualization if not returned
            if model.dsc is not None:
                centroids, attn_maps, stop_logits = model.dsc(features)
                results["checks"]["dsc_centroids_shape"] = list(centroids.shape)
                results["checks"]["dsc_centroids_in_range"] = bool((centroids >= 0).all() and (centroids <= 1).all())
            else:
                results["checks"]["dsc_enabled"] = False
        
        if attn_maps is not None:
            results["checks"]["dsc_attn_shape"] = list(attn_maps.shape)
            results["checks"]["dsc_attn_sums_to_one"] = bool(torch.allclose(attn_maps.sum(dim=(-2, -1)), torch.ones(attn_maps.shape[:2], device=device), atol=1e-4))
        
        if stop_logits is not None:
            stop_probs = torch.sigmoid(stop_logits)
            results["checks"]["dsc_stop_probs"] = stop_probs[0].tolist()
        
        # MSRE check (via direct module call for visualization purposes)
        msre_out = None
        if model.msre is not None and centroids is not None:
            msre_out = model.msre(features, centroids)
            results["checks"]["msre_output_shape"] = list(msre_out.shape)
            results["checks"]["msre_output_valid"] = not (torch.isnan(msre_out).any() or torch.isinf(msre_out).any())
        else:
            results["checks"]["msre_enabled"] = model.msre is not None
        
        # LCR check
        lcr_out = None
        if model.lcr is not None and msre_out is not None and attn_maps is not None:
            lcr_out = model.lcr(msre_out, attn_maps)
            results["checks"]["lcr_output_shape"] = list(lcr_out.shape)
            results["checks"]["lcr_output_valid"] = not (torch.isnan(lcr_out).any() or torch.isinf(lcr_out).any())
        else:
            results["checks"]["lcr_enabled"] = False
        
        # SPH check
        predicates = None
        if model.sph is not None and lcr_out is not None:
            predicates = model.sph(lcr_out)
            results["checks"]["sph_output_shape"] = list(predicates.shape)
            results["checks"]["sph_output_valid"] = not (torch.isnan(predicates).any() or torch.isinf(predicates).any())
        else:
            results["checks"]["sph_enabled"] = False
        
        # Check foreground attention (if DSC enabled)
        fg_ratio = None
        if attn_maps is not None:
            fg_mask = (x[0] > 0)
            fg_attn_total = 0
            bg_attn_total = 0
            for k in range(attn_maps.shape[1]):
                fg_attn_total += attn_maps[0, k][fg_mask].sum().item()
                bg_attn_total += attn_maps[0, k][~fg_mask].sum().item()
            
            fg_ratio = fg_attn_total / (fg_attn_total + bg_attn_total + 1e-6)
            results["checks"]["foreground_attention_ratio"] = fg_ratio
        else:
            results["checks"]["foreground_attention_ratio"] = None
            fg_ratio = None
    
    # Create visualization
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(f"Task: {task_id} | Grid: {h}x{w} | Train Pairs: {num_pairs}", fontsize=14, fontweight='bold')
    
    # Row 1: Input, Target, Prediction
    ax1 = fig.add_subplot(4, 5, 1)
    visualize_grid(ax1, input_grid, "Input Grid")
    
    ax2 = fig.add_subplot(4, 5, 2)
    visualize_grid(ax2, target_grid, "Target Grid")
    
    ax3 = fig.add_subplot(4, 5, 3)
    visualize_grid(ax3, predictions[0], "Prediction")
    
    # Row 1: Encoder features
    ax4 = fig.add_subplot(4, 5, 4)
    visualize_features(ax4, features[0], "Encoder Features (mean)")
    
    ax5 = fig.add_subplot(4, 5, 5)
    # Show feature statistics as text
    ax5.axis('off')
    
    centroid_str = "N/A" if centroids is None else str(centroids[0].cpu().numpy().round(3))
    stop_str = "N/A" if stop_logits is None else str(torch.sigmoid(stop_logits)[0].cpu().numpy().round(3))
    
    feat_stats = f"""Feature Statistics:
Shape: {list(features.shape)}
Mean: {features.mean().item():.4f}
Std: {features.std().item():.4f}
Min: {features.min().item():.4f}
Max: {features.max().item():.4f}

DSC Centroids (normalized):
{centroid_str}

Stop Probs:
{stop_str}
"""
    ax5.text(0.1, 0.9, feat_stats, transform=ax5.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace')
    
    # Row 2 & 3: Attention maps for clues 1-8
    if attn_maps is not None and centroids is not None:
        for k in range(min(8, attn_maps.shape[1])):
            ax = fig.add_subplot(4, 5, 6 + k)
            visualize_attention_map(ax, attn_maps[0, k], centroids[0, k], 
                                   f"Clue {k+1} Attn", (h, w))
    else:
        for k in range(8):
            ax = fig.add_subplot(4, 5, 6 + k)
            ax.text(0.5, 0.5, "DSC Disabled", ha='center', va='center')
            ax.axis('off')
    
    # Row 3 continued: MSRE visualization for first 2 clues
    ax_msre1 = fig.add_subplot(4, 5, 14)
    if msre_out is not None:
        visualize_features(ax_msre1, msre_out[0, 0], "MSRE Clue 1 (mean)")
    else:
        ax_msre1.text(0.5, 0.5, "MSRE Disabled", ha='center', va='center')
        ax_msre1.axis('off')
    
    ax_msre2 = fig.add_subplot(4, 5, 15)
    if msre_out is not None and msre_out.shape[1] > 1:
        visualize_features(ax_msre2, msre_out[0, 1], "MSRE Clue 2 (mean)")
    else:
        ax_msre2.text(0.5, 0.5, "MSRE Disabled", ha='center', va='center')
        ax_msre2.axis('off')
    
    # Row 4: LCR and SPH outputs
    ax_lcr1 = fig.add_subplot(4, 5, 16)
    if lcr_out is not None:
        visualize_features(ax_lcr1, lcr_out[0, 0], "LCR Clue 1 (mean)")
    else:
        ax_lcr1.text(0.5, 0.5, "LCR Disabled", ha='center', va='center')
        ax_lcr1.axis('off')
    
    ax_lcr2 = fig.add_subplot(4, 5, 17)
    if lcr_out is not None and lcr_out.shape[1] > 1:
        visualize_features(ax_lcr2, lcr_out[0, 1], "LCR Clue 2 (mean)")
    else:
        ax_lcr2.text(0.5, 0.5, "LCR Disabled", ha='center', va='center')
        ax_lcr2.axis('off')
    
    ax_sph = fig.add_subplot(4, 5, 18)
    if predicates is not None:
        # SPH shape: (B, K, num_predicates) or (B, P)
        sph_vis = predicates[0].cpu().numpy()
        im = ax_sph.imshow(sph_vis.reshape(-1, 1) if len(sph_vis.shape) == 1 else sph_vis, 
                          aspect='auto', cmap='RdBu')
        ax_sph.set_title("SPH Predicates", fontsize=9)
        ax_sph.set_xlabel("Predicate")
        ax_sph.set_ylabel("Clue")
    else:
        ax_sph.text(0.5, 0.5, "SPH Disabled", ha='center', va='center')
        ax_sph.axis('off')
    
    # Test summary
    ax_summary = fig.add_subplot(4, 5, 19)
    ax_summary.axis('off')
    
    # Count passed checks
    passed = sum(1 for k, v in results["checks"].items() 
                 if isinstance(v, bool) and v)
    total_bool = sum(1 for k, v in results["checks"].items() 
                     if isinstance(v, bool))
    
    fg_str = f"{fg_ratio:.2%}" if fg_ratio is not None else "N/A"
    centroid_valid = results["checks"].get("dsc_centroids_in_range", "N/A")
    attn_norm = results["checks"].get("dsc_attn_sums_to_one", "N/A")
    
    summary_text = f"""Test Summary:
Checks Passed: {passed}/{total_bool}

Foreground Attention: {fg_str}
(higher = better focus on objects)

Output valid: {results["checks"]["output_shape"]}
Centroids in [0,1]: {centroid_valid}
Attn normalized: {attn_norm}
"""
    ax_summary.text(0.1, 0.9, summary_text, transform=ax_summary.transAxes, fontsize=10,
                   verticalalignment='top', fontfamily='monospace')
    
    # Color legend
    ax_legend = fig.add_subplot(4, 5, 20)
    ax_legend.axis('off')
    patches = [mpatches.Patch(color=ARC_COLORS[i], label=f'{i}') for i in range(10)]
    ax_legend.legend(handles=patches, loc='center', ncol=5, title='ARC Colors')
    
    plt.tight_layout()
    
    # Save visualization
    task_output_dir = output_dir / get_grid_size_category(h, w) / task_id
    task_output_dir.mkdir(parents=True, exist_ok=True)
    
    viz_path = task_output_dir / f"{task_id}_visualization.png"
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save JSON report
    report_path = task_output_dir / f"{task_id}_report.json"
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"    Saved: {viz_path.name}")
    
    return results


def generate_summary_report(all_results, output_dir):
    """Generate a summary report across all tasks."""
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_tasks": len(all_results),
        "by_size": {},
        "all_checks_summary": {}
    }
    
    # Group by size
    for result in all_results:
        h, w = result["grid_size"]
        size_cat = get_grid_size_category(h, w)
        if size_cat not in summary["by_size"]:
            summary["by_size"][size_cat] = []
        summary["by_size"][size_cat].append(result["task_id"])
    
    # Aggregate check results
    check_counts = {}
    for result in all_results:
        for check_name, check_value in result["checks"].items():
            if isinstance(check_value, bool):
                if check_name not in check_counts:
                    check_counts[check_name] = {"passed": 0, "failed": 0}
                if check_value:
                    check_counts[check_name]["passed"] += 1
                else:
                    check_counts[check_name]["failed"] += 1
    
    summary["all_checks_summary"] = check_counts
    
    # Save summary
    summary_path = output_dir / "test_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Total tasks tested: {len(all_results)}")
    
    for size, tasks in summary["by_size"].items():
        print(f"  {size.upper()}: {len(tasks)} tasks")
    
    print("\nCheck Results:")
    for check_name, counts in check_counts.items():
        total = counts["passed"] + counts["failed"]
        status = "✓" if counts["failed"] == 0 else "✗"
        print(f"  {status} {check_name}: {counts['passed']}/{total}")
    
    return summary


def main():
    # Configuration
    checkpoint_path = Path("c:/Users/perahmat/Downloads/SCI-ARC/checkpoints/warmup3.pt")
    data_dir = Path("c:/Users/perahmat/Downloads/SCI-ARC/data/arc-agi/data/training")
    output_dir = Path("c:/Users/perahmat/Downloads/SCI-ARC/scripts/outputs/warmup3_checkpoint_tests")
    
    device = 'cpu'
    
    print("="*60)
    print("RLAN Checkpoint Testing on Real ARC Data")
    print("="*60)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model with checkpoint
    model, model_config = load_model_with_checkpoint(checkpoint_path, device)
    
    # Find tasks by size
    print("\nFinding ARC tasks by grid size...")
    tasks_by_size = find_tasks_by_size(data_dir, max_per_size=3)
    
    for size, tasks in tasks_by_size.items():
        print(f"  {size.upper()}: {len(tasks)} tasks found")
        for t in tasks:
            print(f"    - {t['task_id']} ({t['grid_size'][0]}x{t['grid_size'][1]})")
    
    # Test all tasks
    all_results = []
    
    for size in ["small", "medium", "large"]:
        print(f"\n{'='*60}")
        print(f"Testing {size.upper()} grid tasks")
        print("="*60)
        
        for task_info in tasks_by_size[size]:
            try:
                result = test_single_task(model, task_info, output_dir, device)
                all_results.append(result)
            except Exception as e:
                print(f"  ERROR testing {task_info['task_id']}: {e}")
                import traceback
                traceback.print_exc()
    
    # Generate summary report
    generate_summary_report(all_results, output_dir)
    
    print(f"\n\nAll outputs saved to: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
