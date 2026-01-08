"""
Deep Diagnostic Visualization for RLAN

This script creates detailed visualizations to understand WHY accuracy doesn't improve:
1. Input signal quality at each module
2. Solver predictions step-by-step with diff highlighting
3. Attention map quality analysis
4. Clue feature distinctiveness
5. Root cause identification (bad input vs bad algo)
"""

import sys
import json
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ARC color palette (0-9)
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
ARC_CMAP = ListedColormap(ARC_COLORS)


def plot_arc_grid(ax, grid, title="", show_values=False):
    """Plot an ARC grid with proper colors."""
    grid = np.array(grid)
    ax.imshow(grid, cmap=ARC_CMAP, vmin=0, vmax=9)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    
    if show_values and grid.size <= 100:
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                ax.text(j, i, str(grid[i, j]), ha='center', va='center',
                       fontsize=6, color='white' if grid[i, j] in [0, 9] else 'black')


def plot_diff_grid(ax, pred, target, title=""):
    """Plot prediction with error highlighting (red border on wrong pixels)."""
    pred = np.array(pred)
    target = np.array(target)
    
    ax.imshow(pred, cmap=ARC_CMAP, vmin=0, vmax=9)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Highlight errors
    if pred.shape == target.shape:
        errors = pred != target
        for i in range(pred.shape[0]):
            for j in range(pred.shape[1]):
                if errors[i, j]:
                    rect = plt.Rectangle((j-0.5, i-0.5), 1, 1, 
                                         fill=False, edgecolor='red', linewidth=2)
                    ax.add_patch(rect)


def load_checkpoint_and_model():
    """Load checkpoint and create model."""
    from sci_arc.models.rlan import RLAN, RLANConfig
    
    ckpt_path = project_root / "checkpoints" / "warmup3.pt"
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    
    ckpt_config = checkpoint.get('config', {})
    
    # Create RLANConfig from checkpoint
    config = RLANConfig(
        hidden_dim=ckpt_config.get('hidden_dim', 256),
        num_colors=10,
        num_classes=10,
        max_clues=ckpt_config.get('max_clues', 7),
        num_solver_steps=ckpt_config.get('num_solver_steps', 6),
        use_dsc=ckpt_config.get('use_dsc', True),
        use_msre=ckpt_config.get('use_msre', True),
        use_context_encoder=ckpt_config.get('use_context_encoder', True),
        use_hyperlora=ckpt_config.get('use_hyperlora', False),
        use_hpm=ckpt_config.get('use_hpm', False),
        use_lcr=ckpt_config.get('use_lcr', False),
        use_sph=ckpt_config.get('use_sph', False),
        dsc_use_complexity_signals=ckpt_config.get('dsc_use_complexity_signals', False),
    )
    
    model = RLAN(config=config)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    
    return model, ckpt_config


def load_arc_task(task_id: str) -> Optional[Dict]:
    """Load ARC task by ID."""
    arc_dir = project_root / "data" / "arc-agi" / "data" / "training"
    task_file = arc_dir / f"{task_id}.json"
    if not task_file.exists():
        return None
    with open(task_file) as f:
        return json.load(f)


def prepare_batch(task: Dict, device='cpu') -> Tuple[torch.Tensor, ...]:
    """Prepare task for model input."""
    train = task["train"]
    test = task["test"][0]
    
    # Create padded grids (30x30)
    def pad_grid(grid, size=30):
        grid = np.array(grid)
        h, w = grid.shape
        padded = np.zeros((size, size), dtype=np.int64)
        padded[:h, :w] = grid
        return padded
    
    # Prepare support pairs
    support_inputs = []
    support_outputs = []
    for ex in train:
        support_inputs.append(pad_grid(ex["input"]))
        support_outputs.append(pad_grid(ex["output"]))
    
    # Pad to 5 pairs
    while len(support_inputs) < 5:
        support_inputs.append(np.zeros((30, 30), dtype=np.int64))
        support_outputs.append(np.zeros((30, 30), dtype=np.int64))
    
    support_inputs = torch.tensor(np.stack(support_inputs[:5])).unsqueeze(0).to(device)
    support_outputs = torch.tensor(np.stack(support_outputs[:5])).unsqueeze(0).to(device)
    
    # Query input
    query_input = pad_grid(test["input"])
    query_tensor = torch.tensor(query_input).unsqueeze(0).to(device)
    
    # Target
    target = np.array(test["output"])
    
    # Mask
    h, w = np.array(test["input"]).shape
    mask = torch.zeros(1, 30, 30, dtype=torch.bool, device=device)
    mask[0, :h, :w] = True
    
    return query_tensor, support_inputs, support_outputs, target, mask, (h, w)


@torch.no_grad()
def trace_forward_detailed(model, query, support_in, support_out, mask, input_size):
    """Trace forward pass with detailed intermediate outputs using RLAN's built-in return_intermediates."""
    results = {}
    
    # Use RLAN's forward with return_intermediates and return_all_steps
    outputs = model(
        query,
        train_inputs=support_in,
        train_outputs=support_out,
        return_intermediates=True,
        return_all_steps=True
    )
    
    # Extract features
    results['encoder_features'] = outputs.get('features', torch.zeros(1))
    results['features_after_context'] = outputs.get('features', torch.zeros(1))
    
    # DSC outputs
    results['dsc_centroids'] = outputs.get('centroids', torch.zeros(1, 7, 2))
    results['dsc_attention'] = outputs.get('attention_maps', torch.zeros(1, 7, 30, 30))
    results['dsc_stop_logits'] = outputs.get('stop_logits', torch.zeros(1, 7))
    
    # MSRE clue features (may not be directly exposed, try to get from model)
    if hasattr(model, 'msre') and model.msre is not None:
        # Try to access cached clue features if available
        if hasattr(model.msre, '_last_output'):
            results['msre_clue_features'] = model.msre._last_output
        else:
            results['msre_clue_features'] = outputs.get('features', torch.zeros(1)).unsqueeze(1).repeat(1, 7, 1, 1, 1)
    
    # Solver step-by-step
    all_logits = outputs.get('all_logits', [outputs.get('logits', torch.zeros(1, 10, 30, 30))])
    
    solver_steps = []
    for step_idx, logits in enumerate(all_logits):
        probs = F.softmax(logits, dim=1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
        confidence = probs.max(dim=1)[0].mean()
        pred = logits.argmax(dim=1)
        
        solver_steps.append({
            'step': step_idx + 1,
            'logits': logits.clone(),
            'prediction': pred.clone(),
            'h_state': torch.zeros(1),  # Not directly accessible
            'attn_weights': None,
            'entropy': entropy.item(),
            'confidence': confidence.item(),
        })
    
    results['solver_steps'] = solver_steps
    
    return results


def analyze_feature_quality(features: torch.Tensor, name: str) -> Dict:
    """Analyze feature quality metrics."""
    f = features.detach().cpu().numpy()
    
    # Channel-wise variance (should be non-zero for active features)
    channel_var = np.var(f, axis=(0, 2, 3))
    dead_channels = np.sum(channel_var < 1e-6)
    
    # Spatial variance (should be non-zero if features are spatially distinct)
    spatial_var = np.var(f, axis=(0, 1))
    
    # Feature correlation (should be low for diverse features)
    f_flat = f.reshape(f.shape[1], -1)
    if f_flat.shape[1] > 1:
        corr = np.corrcoef(f_flat)
        mean_corr = np.mean(np.abs(corr[np.triu_indices(len(corr), k=1)]))
    else:
        mean_corr = 0.0
    
    return {
        'name': name,
        'shape': f.shape,
        'mean': float(np.mean(f)),
        'std': float(np.std(f)),
        'min': float(np.min(f)),
        'max': float(np.max(f)),
        'dead_channels': int(dead_channels),
        'total_channels': f.shape[1] if len(f.shape) > 1 else 1,
        'spatial_var_mean': float(np.mean(spatial_var)),
        'mean_channel_correlation': float(mean_corr),
    }


def create_diagnostic_visualization(task_id: str, task: Dict, results: Dict, 
                                   target: np.ndarray, input_size: Tuple[int, int],
                                   output_dir: Path):
    """Create comprehensive diagnostic visualizations."""
    h, w = input_size
    th, tw = target.shape
    
    # === FIGURE 1: Solver Step-by-Step with Diffs ===
    fig1, axes1 = plt.subplots(3, 7, figsize=(21, 9))
    fig1.suptitle(f'Task {task_id}: Solver Step-by-Step Analysis', fontsize=14, fontweight='bold')
    
    # Row 1: Input and predictions at each step
    query_np = results['solver_steps'][0]['prediction'][0, :h, :w].cpu().numpy()
    test_input = np.array(task["test"][0]["input"])
    
    plot_arc_grid(axes1[0, 0], test_input, "Test Input")
    
    accuracies = []
    for i, step_data in enumerate(results['solver_steps']):
        pred = step_data['prediction'][0, :th, :tw].cpu().numpy()
        acc = np.mean(pred == target) if pred.shape == target.shape else 0.0
        accuracies.append(acc)
        plot_arc_grid(axes1[0, i+1], pred, f"Step {i+1}: {acc:.1%}")
    
    # Row 2: Diff from target (red = wrong)
    axes1[1, 0].axis('off')
    axes1[1, 0].text(0.5, 0.5, "Target\n↓", ha='center', va='center', fontsize=12)
    
    for i, step_data in enumerate(results['solver_steps']):
        pred = step_data['prediction'][0, :th, :tw].cpu().numpy()
        if pred.shape == target.shape:
            plot_diff_grid(axes1[1, i+1], pred, target, f"Errors Step {i+1}")
        else:
            axes1[1, i+1].text(0.5, 0.5, "Shape\nmismatch", ha='center', va='center')
            axes1[1, i+1].axis('off')
    
    # Row 3: Entropy and confidence
    steps = list(range(1, len(results['solver_steps']) + 1))
    entropies = [s['entropy'] for s in results['solver_steps']]
    confidences = [s['confidence'] for s in results['solver_steps']]
    
    axes1[2, 0].plot(steps, accuracies, 'go-', linewidth=2, markersize=8, label='Accuracy')
    axes1[2, 0].set_ylim(0, 1.1)
    axes1[2, 0].set_xlabel('Step')
    axes1[2, 0].set_ylabel('Accuracy')
    axes1[2, 0].set_title('Accuracy Over Steps')
    axes1[2, 0].legend()
    axes1[2, 0].grid(True, alpha=0.3)
    
    axes1[2, 1].plot(steps, entropies, 'ro-', linewidth=2, markersize=8, label='Entropy')
    axes1[2, 1].set_xlabel('Step')
    axes1[2, 1].set_ylabel('Entropy')
    axes1[2, 1].set_title('Prediction Entropy')
    axes1[2, 1].legend()
    axes1[2, 1].grid(True, alpha=0.3)
    
    axes1[2, 2].plot(steps, confidences, 'bo-', linewidth=2, markersize=8, label='Confidence')
    axes1[2, 2].set_ylim(0, 1.1)
    axes1[2, 2].set_xlabel('Step')
    axes1[2, 2].set_ylabel('Confidence')
    axes1[2, 2].set_title('Max Confidence')
    axes1[2, 2].legend()
    axes1[2, 2].grid(True, alpha=0.3)
    
    # Show improvement/degradation
    acc_change = accuracies[-1] - accuracies[0]
    color = 'green' if acc_change >= 0 else 'red'
    axes1[2, 3].text(0.5, 0.5, f"Δ Accuracy:\n{acc_change:+.1%}", 
                     ha='center', va='center', fontsize=16, color=color, fontweight='bold')
    axes1[2, 3].axis('off')
    
    # Target
    plot_arc_grid(axes1[2, 4], target, "Ground Truth Target", show_values=True)
    
    # Final prediction with values
    final_pred = results['solver_steps'][-1]['prediction'][0, :th, :tw].cpu().numpy()
    plot_arc_grid(axes1[2, 5], final_pred, "Final Prediction", show_values=True)
    
    axes1[2, 6].axis('off')
    
    plt.tight_layout()
    fig1.savefig(output_dir / f"{task_id}_solver_diagnosis.png", dpi=150, bbox_inches='tight')
    plt.close(fig1)
    
    # === FIGURE 2: Input Signal Quality ===
    fig2, axes2 = plt.subplots(2, 4, figsize=(16, 8))
    fig2.suptitle(f'Task {task_id}: Input Signal Quality Analysis', fontsize=14, fontweight='bold')
    
    # Encoder features (sample channels)
    enc_feat = results['encoder_features'][0].cpu().numpy()
    for i in range(4):
        ch_idx = i * (enc_feat.shape[0] // 4)
        im = axes2[0, i].imshow(enc_feat[ch_idx, :h, :w], cmap='viridis')
        axes2[0, i].set_title(f'Encoder Ch {ch_idx}')
        plt.colorbar(im, ax=axes2[0, i], fraction=0.046)
    
    # MSRE clue features (per clue)
    if 'msre_clue_features' in results:
        clue_feat = results['msre_clue_features'][0].cpu().numpy()  # (K, C, H, W)
        for i in range(min(4, clue_feat.shape[0])):
            # Average across channels for visualization
            clue_avg = np.mean(clue_feat[i], axis=0)[:h, :w]
            im = axes2[1, i].imshow(clue_avg, cmap='plasma')
            axes2[1, i].set_title(f'Clue {i} (avg)')
            plt.colorbar(im, ax=axes2[1, i], fraction=0.046)
    
    plt.tight_layout()
    fig2.savefig(output_dir / f"{task_id}_input_signals.png", dpi=150, bbox_inches='tight')
    plt.close(fig2)
    
    # === FIGURE 3: DSC Attention Analysis ===
    if 'dsc_attention' in results:
        fig3, axes3 = plt.subplots(2, 4, figsize=(16, 8))
        fig3.suptitle(f'Task {task_id}: DSC Attention Maps (Slot Focus)', fontsize=14, fontweight='bold')
        
        attn = results['dsc_attention'][0].cpu().numpy()  # (K, H, W)
        
        for i in range(min(7, attn.shape[0])):
            ax = axes3[i // 4, i % 4]
            attn_map = attn[i, :h, :w]
            im = ax.imshow(attn_map, cmap='hot')
            ax.set_title(f'Slot {i}: max={attn_map.max():.3f}')
            plt.colorbar(im, ax=ax, fraction=0.046)
        
        # Show combined attention
        combined = np.sum(attn[:, :h, :w], axis=0)
        ax = axes3[1, 3]
        im = ax.imshow(combined, cmap='hot')
        ax.set_title('Combined Attention')
        plt.colorbar(im, ax=ax, fraction=0.046)
        
        plt.tight_layout()
        fig3.savefig(output_dir / f"{task_id}_dsc_attention.png", dpi=150, bbox_inches='tight')
        plt.close(fig3)
    
    # === FIGURE 4: Step-by-Step Diff Analysis ===
    fig4, axes4 = plt.subplots(2, 6, figsize=(18, 6))
    fig4.suptitle(f'Task {task_id}: What Changes Between Steps?', fontsize=14, fontweight='bold')
    
    for i in range(6):
        if i == 0:
            # First step vs input
            prev = test_input
            curr = results['solver_steps'][0]['prediction'][0, :th, :tw].cpu().numpy()
            title = "Input → Step 1"
        else:
            prev = results['solver_steps'][i-1]['prediction'][0, :th, :tw].cpu().numpy()
            curr = results['solver_steps'][i]['prediction'][0, :th, :tw].cpu().numpy()
            title = f"Step {i} → {i+1}"
        
        # Compute what changed
        if prev.shape == curr.shape:
            changed = (prev != curr).astype(float)
            num_changed = np.sum(changed)
            axes4[0, i].imshow(changed, cmap='Reds', vmin=0, vmax=1)
            axes4[0, i].set_title(f'{title}\n{int(num_changed)} pixels changed')
        else:
            axes4[0, i].text(0.5, 0.5, "Shape\nmismatch", ha='center', va='center')
        axes4[0, i].set_xticks([])
        axes4[0, i].set_yticks([])
        
        # Show the prediction at this step
        plot_arc_grid(axes4[1, i], curr, f"After Step {i+1}")
    
    plt.tight_layout()
    fig4.savefig(output_dir / f"{task_id}_step_changes.png", dpi=150, bbox_inches='tight')
    plt.close(fig4)
    
    return accuracies, entropies, confidences


def diagnose_problem(accuracies: List[float], feature_stats: Dict) -> str:
    """Diagnose the likely root cause of the problem."""
    diagnosis = []
    
    # Check accuracy pattern
    if accuracies[-1] < accuracies[0]:
        diagnosis.append("❌ SOLVER DEGRADATION: Final accuracy lower than first step")
        diagnosis.append("   → Solver may be over-refining or learning wrong patterns")
    elif max(accuracies) - min(accuracies) < 0.01:
        diagnosis.append("⚠️ SOLVER STUCK: No improvement across steps")
        diagnosis.append("   → Solver may not be learning from clue features")
    elif accuracies[-1] > 0.95:
        diagnosis.append("✅ SOLVER WORKING: High final accuracy")
    
    # Check feature quality
    enc_stats = feature_stats.get('encoder_features', {})
    if enc_stats.get('dead_channels', 0) > enc_stats.get('total_channels', 256) * 0.5:
        diagnosis.append("❌ ENCODER ISSUE: Many dead channels in encoder features")
        diagnosis.append("   → Encoder may not be extracting useful features")
    
    if enc_stats.get('mean_channel_correlation', 0) > 0.8:
        diagnosis.append("⚠️ ENCODER ISSUE: High feature correlation")
        diagnosis.append("   → Features may be redundant, not diverse enough")
    
    clue_stats = feature_stats.get('msre_clue_features', {})
    if clue_stats.get('std', 1) < 0.1:
        diagnosis.append("❌ MSRE ISSUE: Low variance in clue features")
        diagnosis.append("   → Clue features may all look similar")
    
    if not diagnosis:
        diagnosis.append("? UNCLEAR: No obvious issues detected")
        diagnosis.append("   → May need to check training data or hyperparameters")
    
    return "\n".join(diagnosis)


def main():
    print("=" * 70)
    print("DEEP DIAGNOSTIC VISUALIZATION FOR RLAN")
    print("=" * 70)
    
    # Load model
    print("\nLoading model...")
    model, config = load_checkpoint_and_model()
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test tasks
    test_tasks = [
        ("007bbfb7", "Tiling 3x3 → 9x9"),
        ("00d62c1b", "Flood fill 20x20"),
        ("025d127b", "Pattern completion 10x10"),
        ("0520fde7", "Column selection 3x7 → 3x3"),
        ("045e512c", "Shape transform 21x21"),
        ("0962bcdd", "Symmetry 12x12"),
    ]
    
    output_dir = project_root / "scripts" / "outputs" / "deep_diagnosis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_diagnoses = []
    
    for task_id, description in test_tasks:
        print(f"\n{'='*70}")
        print(f"Task: {task_id} - {description}")
        print("=" * 70)
        
        task = load_arc_task(task_id)
        if task is None:
            print(f"  [SKIP] Task not found")
            continue
        
        # Prepare batch
        query, support_in, support_out, target, mask, input_size = prepare_batch(task)
        
        print(f"  Input size: {input_size}")
        print(f"  Target size: {target.shape}")
        
        # Trace forward pass
        results = trace_forward_detailed(model, query, support_in, support_out, mask, input_size)
        
        # Analyze feature quality
        feature_stats = {}
        for key in ['encoder_features', 'features_after_context', 'msre_clue_features']:
            if key in results:
                stats = analyze_feature_quality(results[key], key)
                feature_stats[key] = stats
                print(f"  {key}:")
                print(f"    - Shape: {stats['shape']}")
                print(f"    - Dead channels: {stats['dead_channels']}/{stats['total_channels']}")
                print(f"    - Channel correlation: {stats['mean_channel_correlation']:.3f}")
        
        # Create visualizations
        print(f"\n  Creating visualizations...")
        accuracies, entropies, confidences = create_diagnostic_visualization(
            task_id, task, results, target, input_size, output_dir
        )
        
        # Print solver step analysis
        print(f"\n  Solver Step Analysis:")
        for i, acc in enumerate(accuracies):
            change = "" if i == 0 else f" ({accuracies[i] - accuracies[i-1]:+.1%})"
            print(f"    Step {i+1}: Acc={acc:.1%}{change}, Entropy={entropies[i]:.3f}, Conf={confidences[i]:.1%}")
        
        # Diagnose problem
        diagnosis = diagnose_problem(accuracies, feature_stats)
        print(f"\n  Diagnosis:")
        for line in diagnosis.split("\n"):
            print(f"    {line}")
        
        all_diagnoses.append({
            'task_id': task_id,
            'description': description,
            'accuracies': accuracies,
            'diagnosis': diagnosis,
        })
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY OF ALL DIAGNOSES")
    print("=" * 70)
    
    improving = 0
    degrading = 0
    stuck = 0
    
    for d in all_diagnoses:
        acc = d['accuracies']
        if acc[-1] > acc[0] + 0.01:
            improving += 1
        elif acc[-1] < acc[0] - 0.01:
            degrading += 1
        else:
            stuck += 1
    
    print(f"\n  Tasks showing improvement: {improving}/{len(all_diagnoses)}")
    print(f"  Tasks showing degradation: {degrading}/{len(all_diagnoses)}")
    print(f"  Tasks stuck (no change): {stuck}/{len(all_diagnoses)}")
    
    print(f"\n  Visualizations saved to: {output_dir}")
    
    # Save JSON report
    report = {
        'summary': {
            'improving': improving,
            'degrading': degrading,
            'stuck': stuck,
            'total': len(all_diagnoses)
        },
        'tasks': all_diagnoses
    }
    
    with open(output_dir / "diagnosis_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"  Report saved to: {output_dir / 'diagnosis_report.json'}")


if __name__ == "__main__":
    main()
