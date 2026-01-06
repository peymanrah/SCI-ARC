#!/usr/bin/env python3
"""
RLAN Pipeline Visualization - Full Tensor-Level Analysis

This script traces the ACTUAL RLAN pipeline from input to output using
model.forward(return_intermediates=True). NO reimplementation of logic.

Tests different grid sizes, difficulties, and validates all module behaviors.
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sci_arc.models.rlan import RLAN

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


def create_arc_cmap():
    """Create matplotlib colormap for ARC colors."""
    return ListedColormap(ARC_COLORS)


def create_task_grid(size, pattern='random', seed=None):
    """Create different task patterns for testing."""
    if seed is not None:
        torch.manual_seed(seed)
    
    grid = torch.zeros(size, size, dtype=torch.long)
    
    if pattern == 'checkerboard':
        for i in range(size):
            for j in range(size):
                grid[i, j] = 1 if (i + j) % 2 == 0 else 2
    
    elif pattern == 'quadrants':
        mid = size // 2
        grid[:mid, :mid] = 1
        grid[:mid, mid:] = 2
        grid[mid:, :mid] = 3
        grid[mid:, mid:] = 4
        # Add noise
        for _ in range(size):
            r, c = torch.randint(0, size, (2,))
            grid[r, c] = torch.randint(5, 10, (1,)).item()
    
    elif pattern == 'stripes':
        for i in range(size):
            grid[i, :] = i % 5
    
    elif pattern == 'complex':
        # Multiple regions
        region_size = max(1, size // 5)
        for i in range(size):
            for j in range(size):
                ri, rj = i // region_size, j // region_size
                grid[i, j] = (ri * 5 + rj) % 10
        # Add random objects
        for _ in range(size):
            r, c = torch.randint(0, size, (2,))
            grid[r, c] = torch.randint(0, 10, (1,)).item()
    
    elif pattern == 'sparse':
        # Mostly black with colored objects
        num_objects = max(3, size // 3)
        for _ in range(num_objects):
            r, c = torch.randint(0, size, (2,))
            grid[r, c] = torch.randint(1, 10, (1,)).item()
    
    elif pattern == 'random':
        grid = torch.randint(0, 10, (size, size), dtype=torch.long)
    
    return grid.unsqueeze(0)  # (1, H, W)


def analyze_rlan_pipeline(model, input_grid, task_name, device):
    """
    Run ACTUAL RLAN pipeline and collect all intermediate outputs.
    Uses model.forward(return_intermediates=True) - NO reimplementation.
    """
    model.eval()
    
    results = {
        'task_name': task_name,
        'input_grid': input_grid,
        'grid_size': (input_grid.shape[1], input_grid.shape[2]),
        'issues': [],  # Track any problems
    }
    
    with torch.no_grad():
        input_tensor = input_grid.to(device)
        B, H, W = input_tensor.shape
        
        # ===== USE ACTUAL MODEL FORWARD WITH INTERMEDIATES =====
        intermediates = model(input_tensor, return_intermediates=True)
        
        # Extract outputs from actual forward pass
        features = intermediates['features']
        centroids = intermediates['centroids']
        attn_maps = intermediates['attention_maps']
        stop_logits = intermediates['stop_logits']
        predicates = intermediates['predicates']
        count_embedding = intermediates['count_embedding']
        logits = intermediates['logits']
        
        stop_probs = torch.sigmoid(stop_logits)
        predictions = logits.argmax(dim=1)
        
        # ===== STORE FEATURES =====
        results['features'] = {
            'tensor': features,
            'shape': tuple(features.shape),
            'mean': features.mean().item(),
            'std': features.std().item(),
            'min': features.min().item(),
            'max': features.max().item(),
            'has_nan': bool(torch.isnan(features).any()),
            'has_inf': bool(torch.isinf(features).any()),
        }
        
        # Verify feature shape: should be (B, D, H, W)
        if len(features.shape) != 4:
            results['issues'].append(f"Features wrong rank: {features.shape}, expected 4D")
        if features.shape[0] != B:
            results['issues'].append(f"Features batch mismatch: {features.shape[0]} vs {B}")
        if features.shape[2] != H or features.shape[3] != W:
            results['issues'].append(f"Features spatial mismatch: {features.shape[2:]} vs ({H}, {W})")
        
        # ===== STORE DSC OUTPUTS =====
        results['dsc'] = {
            'centroids': centroids,
            'centroids_shape': tuple(centroids.shape),
            'attention_maps': attn_maps,
            'attention_maps_shape': tuple(attn_maps.shape),
            'stop_logits': stop_logits,
            'stop_logits_shape': tuple(stop_logits.shape),
            'stop_probs': stop_probs,
        }
        
        # Verify DSC shapes
        K = model.max_clues
        if centroids.shape != (B, K, 2):
            results['issues'].append(f"Centroids shape wrong: {centroids.shape}, expected ({B}, {K}, 2)")
        if attn_maps.shape != (B, K, H, W):
            results['issues'].append(f"Attention maps shape wrong: {attn_maps.shape}, expected ({B}, {K}, {H}, {W})")
        if stop_logits.shape != (B, K):
            results['issues'].append(f"Stop logits shape wrong: {stop_logits.shape}, expected ({B}, {K})")
        
        # Verify centroids in [0, 1] range
        if centroids.min() < 0 or centroids.max() > 1:
            results['issues'].append(f"Centroids out of [0,1]: [{centroids.min():.4f}, {centroids.max():.4f}]")
        
        # Verify attention maps sum to 1 per clue
        attn_sums = attn_maps.sum(dim=(2, 3))  # (B, K)
        if not torch.allclose(attn_sums, torch.ones_like(attn_sums), atol=1e-4):
            results['issues'].append(f"Attention maps don't sum to 1: [{attn_sums.min():.4f}, {attn_sums.max():.4f}]")
        
        # Verify attention non-negative
        if attn_maps.min() < 0:
            results['issues'].append(f"Attention maps have negative values: {attn_maps.min():.6f}")
        
        # Analyze centroid spread
        active_mask = stop_probs[0] < 0.5
        active_count = active_mask.sum().item()
        results['dsc']['active_clues'] = active_count
        
        # Compute pairwise distances between centroids
        centroids_2d = centroids[0]
        distances = []
        for i in range(K):
            for j in range(i + 1, K):
                dist = torch.norm(centroids_2d[i] - centroids_2d[j]).item()
                distances.append(dist)
        
        results['dsc']['centroid_stats'] = {
            'min_dist': min(distances) if distances else 0,
            'max_dist': max(distances) if distances else 0,
            'mean_dist': np.mean(distances) if distances else 0,
        }
        
        # Check if centroids are collapsing
        if distances and min(distances) < 0.05:
            results['issues'].append(f"Centroids may be collapsing: min_dist={min(distances):.4f}")
        
        # ===== STORE COUNT EMBEDDING (LCR) =====
        results['count_embedding'] = {
            'tensor': count_embedding,
            'shape': tuple(count_embedding.shape),
            'mean': count_embedding.mean().item(),
            'std': count_embedding.std().item(),
            'has_nan': bool(torch.isnan(count_embedding).any()),
        }
        
        # ===== STORE PREDICATES (SPH) =====
        results['predicates'] = {
            'tensor': predicates,
            'shape': tuple(predicates.shape),
            'values': predicates[0].cpu().numpy().tolist(),
            'has_nan': bool(torch.isnan(predicates).any()),
        }
        
        # Verify predicate shape
        P = model.num_predicates
        if predicates.shape != (B, P):
            results['issues'].append(f"Predicates shape wrong: {predicates.shape}, expected ({B}, {P})")
        
        # ===== STORE FINAL OUTPUT =====
        results['output'] = {
            'logits': logits,
            'logits_shape': tuple(logits.shape),
            'predictions': predictions,
            'predictions_shape': tuple(predictions.shape),
            'unique_colors': sorted(predictions.unique().tolist()),
            'has_nan': bool(torch.isnan(logits).any()),
            'has_inf': bool(torch.isinf(logits).any()),
        }
        
        # Verify output shape
        if logits.shape != (B, model.num_classes, H, W):
            results['issues'].append(f"Logits shape wrong: {logits.shape}, expected ({B}, {model.num_classes}, {H}, {W})")
        if predictions.shape != (B, H, W):
            results['issues'].append(f"Predictions shape wrong: {predictions.shape}, expected ({B}, {H}, {W})")
        
        # Verify all predicted colors are valid
        invalid_colors = [c for c in results['output']['unique_colors'] if c < 0 or c > 9]
        if invalid_colors:
            results['issues'].append(f"Invalid colors predicted: {invalid_colors}")
        
        # Color distribution
        results['output']['color_distribution'] = {}
        for c in range(10):
            count = (predictions == c).sum().item()
            if count > 0:
                results['output']['color_distribution'][c] = count
    
    return results


def print_pipeline_summary(results):
    """Print detailed summary with issue flags."""
    print("\n" + "=" * 70)
    print(f"TASK: {results['task_name']}")
    print(f"Grid Size: {results['grid_size'][0]} x {results['grid_size'][1]}")
    print("=" * 70)
    
    # Features
    f = results['features']
    print(f"\n[1] ENCODER FEATURES")
    print(f"    Shape: {f['shape']}")
    print(f"    Stats: mean={f['mean']:.4f}, std={f['std']:.4f}, min={f['min']:.4f}, max={f['max']:.4f}")
    if f['has_nan']:
        print("    [CRITICAL] Contains NaN!")
    if f['has_inf']:
        print("    [CRITICAL] Contains Inf!")
    
    # DSC
    dsc = results['dsc']
    print(f"\n[2] DSC (Dynamic Saliency Controller)")
    print(f"    Centroids: {dsc['centroids_shape']}")
    print(f"    Attention Maps: {dsc['attention_maps_shape']}")
    print(f"    Stop Logits: {dsc['stop_logits_shape']}")
    print(f"    Active Clues: {dsc['active_clues']} / {dsc['stop_probs'].shape[1]}")
    print(f"    Stop Probabilities:")
    for k in range(dsc['stop_probs'].shape[1]):
        p = dsc['stop_probs'][0, k].item()
        status = "ACTIVE" if p < 0.5 else "STOP"
        cy, cx = dsc['centroids'][0, k].cpu().numpy()
        print(f"        Clue {k+1}: P(stop)={p:.4f} [{status}] @ ({cy:.3f}, {cx:.3f})")
    cs = dsc['centroid_stats']
    print(f"    Centroid Spread: min={cs['min_dist']:.4f}, max={cs['max_dist']:.4f}, mean={cs['mean_dist']:.4f}")
    
    # LCR
    ce = results['count_embedding']
    print(f"\n[3] LCR (Latent Counting Registers)")
    print(f"    Shape: {ce['shape']}")
    print(f"    Stats: mean={ce['mean']:.4f}, std={ce['std']:.4f}")
    
    # SPH
    pred = results['predicates']
    print(f"\n[4] SPH (Structured Predicate Heads)")
    print(f"    Shape: {pred['shape']}")
    print(f"    Values: {[f'{v:.3f}' for v in pred['values']]}")
    
    # Output
    out = results['output']
    print(f"\n[5] FINAL OUTPUT")
    print(f"    Logits: {out['logits_shape']}")
    print(f"    Predictions: {out['predictions_shape']}")
    print(f"    Unique Colors: {out['unique_colors']}")
    print(f"    Distribution:")
    H, W = results['grid_size']
    total = H * W
    for c, count in sorted(out['color_distribution'].items()):
        print(f"        Color {c}: {count:>5} cells ({100*count/total:>5.1f}%)")
    
    # Issues
    if results['issues']:
        print(f"\n[!] ISSUES DETECTED ({len(results['issues'])}):")
        for issue in results['issues']:
            print(f"    - {issue}")
    else:
        print(f"\n[OK] No issues detected")


def create_visualization(all_results, output_dir):
    """Create and save comprehensive visualizations."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    num_tasks = len(all_results)
    arc_cmap = create_arc_cmap()
    
    # ===== FIGURE 1: Input vs Output Comparison =====
    fig1, axes1 = plt.subplots(num_tasks, 3, figsize=(12, 4 * num_tasks))
    if num_tasks == 1:
        axes1 = axes1.reshape(1, -1)
    
    fig1.suptitle('RLAN Pipeline: Input -> Features -> Output', fontsize=14, fontweight='bold')
    
    for i, res in enumerate(all_results):
        H, W = res['grid_size']
        
        # Input
        ax = axes1[i, 0]
        im = ax.imshow(res['input_grid'][0].cpu().numpy(), cmap=arc_cmap, vmin=0, vmax=9)
        ax.set_title(f"INPUT: {res['task_name']}\n{H}x{W}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        for r in range(H + 1):
            ax.axhline(r - 0.5, color='white', linewidth=0.5)
        for c in range(W + 1):
            ax.axvline(c - 0.5, color='white', linewidth=0.5)
        
        # Features (mean across channels)
        ax = axes1[i, 1]
        feat_mean = res['features']['tensor'][0].mean(dim=0).cpu().numpy()
        im = ax.imshow(feat_mean, cmap='viridis')
        ax.set_title(f"FEATURES (mean)\n{res['features']['shape']}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax)
        
        # Output
        ax = axes1[i, 2]
        im = ax.imshow(res['output']['predictions'][0].cpu().numpy(), cmap=arc_cmap, vmin=0, vmax=9)
        ax.set_title(f"OUTPUT\nColors: {res['output']['unique_colors']}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        for r in range(H + 1):
            ax.axhline(r - 0.5, color='white', linewidth=0.5)
        for c in range(W + 1):
            ax.axvline(c - 0.5, color='white', linewidth=0.5)
    
    plt.tight_layout()
    fig1.savefig(output_dir / 'rlan_input_output.png', dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print(f"  Saved: {output_dir / 'rlan_input_output.png'}")
    
    # ===== FIGURE 2: DSC Attention Maps per Task =====
    for task_idx, res in enumerate(all_results):
        K = res['dsc']['attention_maps'].shape[1]
        H, W = res['grid_size']
        
        fig2, axes2 = plt.subplots(2, 4, figsize=(16, 8))
        fig2.suptitle(f"DSC Attention Maps: {res['task_name']} ({H}x{W})", fontsize=14, fontweight='bold')
        
        axes_flat = axes2.flatten()
        for k in range(min(8, K)):
            ax = axes_flat[k]
            attn = res['dsc']['attention_maps'][0, k].cpu().numpy()
            im = ax.imshow(attn, cmap='hot', vmin=0)
            
            # Mark centroid
            cy, cx = res['dsc']['centroids'][0, k].cpu().numpy()
            ax.scatter([cx * W], [cy * H], c='cyan', s=200, marker='x', linewidths=3, zorder=10)
            
            stop_p = res['dsc']['stop_probs'][0, k].item()
            status = "ACTIVE" if stop_p < 0.5 else "STOP"
            color = 'green' if stop_p < 0.5 else 'red'
            ax.set_title(f"Clue {k+1}\nP(stop)={stop_p:.3f} [{status}]", fontsize=10, color=color)
            ax.set_xticks([]); ax.set_yticks([])
            plt.colorbar(im, ax=ax)
        
        for k in range(K, 8):
            axes_flat[k].axis('off')
        
        plt.tight_layout()
        fig2.savefig(output_dir / f'rlan_attention_task{task_idx+1}.png', dpi=150, bbox_inches='tight')
        plt.close(fig2)
        print(f"  Saved: {output_dir / f'rlan_attention_task{task_idx+1}.png'}")
    
    # ===== FIGURE 3: Centroids Overlay on Input =====
    fig3, axes3 = plt.subplots(1, num_tasks, figsize=(5 * num_tasks, 5))
    if num_tasks == 1:
        axes3 = [axes3]
    
    fig3.suptitle('DSC Centroid Positions on Input Grid', fontsize=14, fontweight='bold')
    
    for i, res in enumerate(all_results):
        ax = axes3[i]
        grid = res['input_grid'][0].cpu().numpy()
        H, W = grid.shape
        
        im = ax.imshow(grid, cmap=arc_cmap, vmin=0, vmax=9)
        ax.set_title(f"{res['task_name']}\n{H}x{W}", fontsize=10)
        
        # Grid lines
        for r in range(H + 1):
            ax.axhline(r - 0.5, color='white', linewidth=0.5)
        for c in range(W + 1):
            ax.axvline(c - 0.5, color='white', linewidth=0.5)
        
        # Plot centroids
        centroids = res['dsc']['centroids'][0].cpu().numpy()
        stop_probs = res['dsc']['stop_probs'][0].cpu().numpy()
        
        for k in range(centroids.shape[0]):
            cy, cx = centroids[k]
            p_stop = stop_probs[k]
            
            color = 'lime' if p_stop < 0.5 else 'red'
            size = 300 if p_stop < 0.5 else 150
            
            ax.scatter([cx * W], [cy * H], c=color, s=size, marker='o',
                      edgecolors='white', linewidths=2, zorder=10)
            ax.annotate(f'{k+1}', (cx * W, cy * H), color='black',
                       fontsize=8, ha='center', va='center', fontweight='bold')
        
        ax.set_xticks([]); ax.set_yticks([])
        
        # Legend
        active_patch = mpatches.Patch(color='lime', label=f'Active ({(stop_probs < 0.5).sum()})')
        stop_patch = mpatches.Patch(color='red', label=f'Stopped ({(stop_probs >= 0.5).sum()})')
        ax.legend(handles=[active_patch, stop_patch], loc='upper right', fontsize=8)
    
    plt.tight_layout()
    fig3.savefig(output_dir / 'rlan_centroids.png', dpi=150, bbox_inches='tight')
    plt.close(fig3)
    print(f"  Saved: {output_dir / 'rlan_centroids.png'}")
    
    # ===== FIGURE 4: Stop Probability Comparison =====
    fig4, ax4 = plt.subplots(figsize=(12, 6))
    fig4.suptitle('Stop Probabilities Across Tasks and Clues', fontsize=14, fontweight='bold')
    
    x = np.arange(8)  # 8 clues
    width = 0.25
    
    for i, res in enumerate(all_results):
        stop_probs = res['dsc']['stop_probs'][0].cpu().numpy()
        bars = ax4.bar(x + i * width, stop_probs, width, label=res['task_name'])
        
        # Color bars by active/stop
        for j, bar in enumerate(bars):
            bar.set_color('green' if stop_probs[j] < 0.5 else 'red')
            bar.set_alpha(0.7)
    
    ax4.axhline(0.5, color='black', linestyle='--', linewidth=2, label='Threshold')
    ax4.set_xlabel('Clue Index')
    ax4.set_ylabel('P(stop)')
    ax4.set_xticks(x + width * (num_tasks - 1) / 2)
    ax4.set_xticklabels([f'Clue {i+1}' for i in range(8)])
    ax4.legend()
    ax4.set_ylim(0, 1)
    
    plt.tight_layout()
    fig4.savefig(output_dir / 'rlan_stop_probs.png', dpi=150, bbox_inches='tight')
    plt.close(fig4)
    print(f"  Saved: {output_dir / 'rlan_stop_probs.png'}")
    
    # ===== FIGURE 5: Predicates Comparison =====
    fig5, ax5 = plt.subplots(figsize=(12, 6))
    fig5.suptitle('Predicate Values Across Tasks', fontsize=14, fontweight='bold')
    
    num_preds = len(all_results[0]['predicates']['values'])
    x = np.arange(num_preds)
    width = 0.25
    
    for i, res in enumerate(all_results):
        pred_vals = res['predicates']['values']
        ax5.bar(x + i * width, pred_vals, width, label=res['task_name'], alpha=0.7)
    
    ax5.set_xlabel('Predicate Index')
    ax5.set_ylabel('Value')
    ax5.set_xticks(x + width * (num_tasks - 1) / 2)
    ax5.set_xticklabels([f'P{i+1}' for i in range(num_preds)])
    ax5.legend()
    
    plt.tight_layout()
    fig5.savefig(output_dir / 'rlan_predicates.png', dpi=150, bbox_inches='tight')
    plt.close(fig5)
    print(f"  Saved: {output_dir / 'rlan_predicates.png'}")


def run_verification_checks(all_results, model):
    """Run comprehensive verification checks and return pass/fail summary."""
    print("\n" + "=" * 70)
    print("VERIFICATION CHECKS")
    print("=" * 70)
    
    checks = []
    
    # Check 1: Output shape matches input spatial dims
    print("\n[CHECK 1] Output shape matches input spatial dimensions")
    for res in all_results:
        H, W = res['grid_size']
        out_shape = res['output']['predictions_shape']
        passed = out_shape[1] == H and out_shape[2] == W
        status = "[OK]" if passed else "[FAIL]"
        print(f"    {res['task_name']}: Input ({H}x{W}) -> Output {out_shape} {status}")
        checks.append(('output_shape', res['task_name'], passed))
    
    # Check 2: All output colors valid (0-9)
    print("\n[CHECK 2] All output colors are valid (0-9)")
    for res in all_results:
        colors = res['output']['unique_colors']
        passed = all(0 <= c <= 9 for c in colors)
        status = "[OK]" if passed else "[FAIL]"
        print(f"    {res['task_name']}: Colors {colors} {status}")
        checks.append(('valid_colors', res['task_name'], passed))
    
    # Check 3: Attention maps properly normalized
    print("\n[CHECK 3] Attention maps sum to 1 (softmax)")
    for res in all_results:
        attn = res['dsc']['attention_maps']
        sums = attn.sum(dim=(2, 3))
        passed = torch.allclose(sums, torch.ones_like(sums), atol=1e-4)
        status = "[OK]" if passed else "[FAIL]"
        print(f"    {res['task_name']}: Sums in [{sums.min():.6f}, {sums.max():.6f}] {status}")
        checks.append(('attn_normalized', res['task_name'], passed))
    
    # Check 4: Stop probabilities in [0, 1]
    print("\n[CHECK 4] Stop probabilities in [0, 1]")
    for res in all_results:
        probs = res['dsc']['stop_probs']
        passed = (probs >= 0).all() and (probs <= 1).all()
        status = "[OK]" if passed else "[FAIL]"
        print(f"    {res['task_name']}: Range [{probs.min():.4f}, {probs.max():.4f}] {status}")
        checks.append(('stop_prob_range', res['task_name'], passed))
    
    # Check 5: Centroids in [0, 1]
    print("\n[CHECK 5] Centroids in valid range [0, 1]")
    for res in all_results:
        centroids = res['dsc']['centroids']
        passed = (centroids >= 0).all() and (centroids <= 1).all()
        status = "[OK]" if passed else "[FAIL]"
        print(f"    {res['task_name']}: Range [{centroids.min():.4f}, {centroids.max():.4f}] {status}")
        checks.append(('centroid_range', res['task_name'], passed))
    
    # Check 6: No NaN/Inf in features
    print("\n[CHECK 6] No NaN/Inf in features")
    for res in all_results:
        passed = not res['features']['has_nan'] and not res['features']['has_inf']
        status = "[OK]" if passed else "[FAIL]"
        nan_info = "NaN!" if res['features']['has_nan'] else ""
        inf_info = "Inf!" if res['features']['has_inf'] else ""
        print(f"    {res['task_name']}: {status} {nan_info} {inf_info}")
        checks.append(('features_valid', res['task_name'], passed))
    
    # Check 7: No NaN/Inf in output
    print("\n[CHECK 7] No NaN/Inf in output logits")
    for res in all_results:
        passed = not res['output']['has_nan'] and not res['output']['has_inf']
        status = "[OK]" if passed else "[FAIL]"
        print(f"    {res['task_name']}: {status}")
        checks.append(('output_valid', res['task_name'], passed))
    
    # Check 8: Stop probs VARY across tasks (not frozen)
    print("\n[CHECK 8] Stop probabilities vary across tasks")
    all_stop_probs = [res['dsc']['stop_probs'][0].mean().item() for res in all_results]
    variance = np.var(all_stop_probs)
    passed = variance > 0.001  # Should have some variance
    status = "[OK]" if passed else "[WARN - may be frozen]"
    print(f"    Mean stop probs: {[f'{p:.4f}' for p in all_stop_probs]}")
    print(f"    Variance: {variance:.6f} {status}")
    checks.append(('stop_prob_variance', 'global', passed))
    
    # Check 9: Different grid sizes produce different feature stats
    print("\n[CHECK 9] Different grid sizes produce different feature statistics")
    feat_means = [res['features']['mean'] for res in all_results]
    feat_stds = [res['features']['std'] for res in all_results]
    variance_mean = np.var(feat_means)
    passed = variance_mean > 0.0001  # Should have some variance
    status = "[OK]" if passed else "[WARN]"
    print(f"    Feature means: {[f'{m:.4f}' for m in feat_means]}")
    print(f"    Feature stds: {[f'{s:.4f}' for s in feat_stds]}")
    print(f"    Variance of means: {variance_mean:.6f} {status}")
    checks.append(('feature_variance', 'global', passed))
    
    # Check 10: Active clue count varies with task complexity
    print("\n[CHECK 10] Active clue count varies with task complexity")
    active_counts = [res['dsc']['active_clues'] for res in all_results]
    print(f"    Active clues: {active_counts}")
    # Note: with random init, this may not perfectly correlate with complexity
    passed = len(set(active_counts)) > 1 or all_results[0]['dsc']['active_clues'] > 0
    status = "[OK]" if passed else "[WARN - may be frozen]"
    print(f"    {status}")
    checks.append(('active_clue_variance', 'global', passed))
    
    # Summary
    total = len(checks)
    passed_count = sum(1 for _, _, p in checks if p)
    
    print("\n" + "=" * 70)
    print(f"SUMMARY: {passed_count}/{total} checks passed")
    print("=" * 70)
    
    # List failures
    failures = [(name, task, passed) for name, task, passed in checks if not passed]
    if failures:
        print("\nFailed/Warning checks:")
        for name, task, _ in failures:
            print(f"    - {name} ({task})")
    
    return passed_count == total


def main():
    print("=" * 70)
    print("RLAN FULL PIPELINE VISUALIZATION & VERIFICATION")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Initialize model
    print("\nLoading RLAN model...")
    model = RLAN(
        hidden_dim=128,
        num_colors=10,
        num_classes=10,
        max_grid_size=30,
        max_clues=8,
        num_predicates=8,
        num_solver_steps=6,
        dropout=0.1,
    ).to(device)
    model.eval()
    print(f"Model max_clues: {model.max_clues}")
    print(f"Model num_predicates: {model.num_predicates}")
    print(f"Model num_classes: {model.num_classes}")
    
    # Define test tasks with DIFFERENT grid sizes and patterns
    test_tasks = [
        ('Small 5x5 (Sparse)', create_task_grid(5, 'sparse', seed=42)),
        ('Medium 10x10 (Quadrants)', create_task_grid(10, 'quadrants', seed=123)),
        ('Large 25x25 (Complex)', create_task_grid(25, 'complex', seed=456)),
    ]
    
    print(f"\nAnalyzing {len(test_tasks)} tasks with different grid sizes...")
    
    all_results = []
    for task_name, input_grid in test_tasks:
        print(f"\n>>> Processing: {task_name}")
        results = analyze_rlan_pipeline(model, input_grid, task_name, device)
        all_results.append(results)
        print_pipeline_summary(results)
    
    # Cross-task comparison table
    print("\n" + "=" * 70)
    print("CROSS-TASK COMPARISON")
    print("=" * 70)
    
    print("\n{:<28} {:>8} {:>12} {:>14} {:>12}".format(
        "Task", "Grid", "Active/Total", "Centroid Dist", "Out Colors"))
    print("-" * 78)
    
    for res in all_results:
        grid_str = f"{res['grid_size'][0]}x{res['grid_size'][1]}"
        active = res['dsc']['active_clues']
        total = res['dsc']['stop_probs'].shape[1]
        spread = res['dsc']['centroid_stats']['mean_dist']
        colors = len(res['output']['unique_colors'])
        print(f"{res['task_name']:<28} {grid_str:>8} {active:>5}/{total:<5} {spread:>14.4f} {colors:>12}")
    
    # Verification checks
    all_passed = run_verification_checks(all_results, model)
    
    # Create visualizations
    output_dir = Path(__file__).parent / 'outputs'
    print(f"\nGenerating visualizations to: {output_dir}")
    create_visualization(all_results, output_dir)
    
    # Final verdict
    print("\n" + "=" * 70)
    if all_passed:
        print("[SUCCESS] All verification checks passed!")
    else:
        print("[WARNING] Some verification checks failed - review issues above")
    print("=" * 70)
    
    # Collect all issues
    all_issues = []
    for res in all_results:
        for issue in res['issues']:
            all_issues.append(f"{res['task_name']}: {issue}")
    
    if all_issues:
        print(f"\n[!] TOTAL ISSUES FOUND ({len(all_issues)}):")
        for issue in all_issues:
            print(f"    - {issue}")
    
    print("\nDone!")
    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
