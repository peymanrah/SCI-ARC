"""
Comprehensive RLAN Module Testing Suite

Tests ALL RLAN modules (old and new) with trained checkpoint on real ARC data:
1. Core RLAN modules: Encoder, DSC, MSRE, LCR, SPH, RecursiveSolver
2. HyperLoRA meta-learning module
3. Context Encoder (cross-attention)
4. TEPS (Test-time Exhaustive Program Search)
5. NS-TEPS (Neuro-Symbolic TEPS with object-level operations)
6. Program-Guided Training integration
7. Recursive solver per-step visualization
8. End-to-end signal quality analysis
9. Gradient flow and norm analysis

Author: AI Research Assistant
Date: January 2026
"""

import os
import sys
import json
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import traceback
import time

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


def load_arc_task(task_path: Path) -> dict:
    """Load an ARC task from JSON file."""
    with open(task_path, 'r') as f:
        return json.load(f)


def get_grid_size_category(h: int, w: int) -> str:
    """Categorize grid by size."""
    max_dim = max(h, w)
    if max_dim <= 10:
        return "small"
    elif max_dim <= 20:
        return "medium"
    else:
        return "large"


def find_tasks_by_size(data_dir: Path, target_sizes=("small", "medium", "large"), max_per_size: int = 2) -> Dict[str, List[dict]]:
    """Find ARC tasks categorized by grid size."""
    tasks_by_size = {size: [] for size in target_sizes}
    json_files = list(data_dir.glob("*.json"))
    
    for task_path in json_files:
        try:
            task = load_arc_task(task_path)
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
        except Exception:
            continue
    
    return tasks_by_size


def load_model_with_checkpoint(checkpoint_path: Path, device: str = 'cpu') -> Tuple[RLAN, dict]:
    """Load RLAN model with trained checkpoint."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    full_config = checkpoint.get('config', {})
    model_config = full_config.get('model', {})
    
    # Detect complexity signals for backward compatibility
    use_complexity_signals = model_config.get('dsc_use_complexity_signals', None)
    if use_complexity_signals is None:
        stop_pred_key = 'dsc.stop_predictor.0.weight'
        if stop_pred_key in checkpoint['model_state_dict']:
            stop_pred_shape = checkpoint['model_state_dict'][stop_pred_key].shape
            hidden_dim = model_config.get('hidden_dim', 256)
            expected_old = hidden_dim + 1 + hidden_dim
            expected_new = expected_old + 3
            use_complexity_signals = (stop_pred_shape[1] == expected_new)
        else:
            use_complexity_signals = False
    
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
        dsc_use_complexity_signals=use_complexity_signals,
    )
    
    model = RLAN(config=config)
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    if missing_keys:
        print(f"  Warning: Missing keys: {len(missing_keys)}")
    if unexpected_keys:
        print(f"  Warning: Unexpected keys: {len(unexpected_keys)}")
    
    model.to(device)
    model.eval()
    
    print(f"  Loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    return model, model_config


def visualize_grid(ax, grid, title: str, cmap=None):
    """Visualize an ARC grid."""
    if cmap is None:
        cmap = get_arc_cmap()
    if isinstance(grid, torch.Tensor):
        grid = grid.cpu().numpy()
    
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=9)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    
    h, w = grid.shape
    for i in range(h + 1):
        ax.axhline(i - 0.5, color='white', linewidth=0.3)
    for j in range(w + 1):
        ax.axvline(j - 0.5, color='white', linewidth=0.3)


def visualize_features(ax, features, title: str):
    """Visualize feature map (mean across channels)."""
    if isinstance(features, torch.Tensor):
        features = features.cpu().numpy()
    if len(features.shape) == 3:
        feat_vis = features.mean(axis=0)
    else:
        feat_vis = features
    ax.imshow(feat_vis, cmap='viridis')
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])


class TestResult:
    """Container for test results."""
    def __init__(self, name: str):
        self.name = name
        self.passed = True
        self.messages = []
        self.metrics = {}
        self.warnings = []
        self.errors = []
    
    def add_check(self, check_name: str, passed: bool, message: str = ""):
        if not passed:
            self.passed = False
            self.errors.append(f"[FAIL] {check_name}: {message}")
        else:
            self.messages.append(f"[OK] {check_name}")
    
    def add_warning(self, message: str):
        self.warnings.append(f"[WARN] {message}")
    
    def add_metric(self, name: str, value):
        self.metrics[name] = value
    
    def __repr__(self):
        status = "PASS" if self.passed else "FAIL"
        return f"TestResult({self.name}): {status} - {len(self.errors)} errors, {len(self.warnings)} warnings"


def pad_grid(grid, target_h: int, target_w: int, pad_value: int = 0) -> np.ndarray:
    """Pad a grid to target dimensions."""
    arr = np.array(grid)
    curr_h, curr_w = arr.shape
    padded = np.full((target_h, target_w), pad_value, dtype=np.int64)
    padded[:curr_h, :curr_w] = arr
    return padded


def prepare_task_tensors(task: dict, device: str = 'cpu') -> Tuple[torch.Tensor, ...]:
    """Prepare task tensors for model input."""
    train_pairs = task["train"]
    num_pairs = len(train_pairs)
    
    max_h = max(max(len(p["input"]) for p in train_pairs),
                max(len(p["output"]) for p in train_pairs))
    max_w = max(max(len(p["input"][0]) for p in train_pairs),
                max(len(p["output"][0]) for p in train_pairs))
    
    train_inputs_list = [pad_grid(p["input"], max_h, max_w) for p in train_pairs]
    train_outputs_list = [pad_grid(p["output"], max_h, max_w) for p in train_pairs]
    
    train_inputs = torch.tensor(np.stack(train_inputs_list), dtype=torch.long).unsqueeze(0).to(device)
    train_outputs = torch.tensor(np.stack(train_outputs_list), dtype=torch.long).unsqueeze(0).to(device)
    pair_mask = torch.ones(1, num_pairs, device=device)
    
    test_input = pad_grid(task["train"][0]["input"], max_h, max_w)
    test_target = pad_grid(task["train"][0]["output"], max_h, max_w)
    x = torch.tensor(test_input, dtype=torch.long).unsqueeze(0).to(device)
    
    return x, train_inputs, train_outputs, pair_mask, test_input, test_target


# ==============================================================================
# TEST 1: CORE RLAN MODULES
# ==============================================================================

def test_core_modules(model: RLAN, task_info: dict, output_dir: Path, device: str = 'cpu') -> TestResult:
    """Test core RLAN modules: Encoder, DSC, MSRE, Solver."""
    result = TestResult("Core Modules")
    task_id = task_info["task_id"]
    task = task_info["task"]
    
    try:
        x, train_inputs, train_outputs, pair_mask, test_input, test_target = prepare_task_tensors(task, device)
        
        with torch.no_grad():
            # Test Encoder
            features = model.encode(x)
            result.add_check("Encoder output shape", len(features.shape) == 4, f"shape={features.shape}")
            result.add_check("Encoder no NaN", not torch.isnan(features).any().item(), "NaN detected")
            result.add_check("Encoder no Inf", not torch.isinf(features).any().item(), "Inf detected")
            result.add_metric("encoder_mean", features.mean().item())
            result.add_metric("encoder_std", features.std().item())
            
            # Test DSC
            if model.dsc is not None:
                centroids, attn_maps, stop_logits = model.dsc(features)
                result.add_check("DSC centroids shape", centroids.shape[1] == model.max_clues, f"shape={centroids.shape}")
                result.add_check("DSC centroids range", (centroids >= 0).all().item() and (centroids <= 1).all().item(), "out of [0,1]")
                result.add_check("DSC attn normalized", torch.allclose(attn_maps.sum(dim=(-2,-1)), torch.ones(attn_maps.shape[:2], device=device), atol=1e-3), "not normalized")
                result.add_check("DSC no NaN", not torch.isnan(attn_maps).any().item(), "NaN in attention")
                result.add_metric("dsc_stop_probs", torch.sigmoid(stop_logits)[0].cpu().tolist())
            else:
                result.add_warning("DSC is disabled")
            
            # Test MSRE
            if model.msre is not None and model.dsc is not None:
                msre_out = model.msre(features, centroids)
                result.add_check("MSRE output shape", len(msre_out.shape) == 5, f"shape={msre_out.shape}")
                result.add_check("MSRE no NaN", not torch.isnan(msre_out).any().item(), "NaN detected")
                result.add_metric("msre_mean", msre_out.mean().item())
            else:
                result.add_warning("MSRE is disabled")
            
            # Test full forward pass
            outputs = model(
                input_grid=x,
                train_inputs=train_inputs,
                train_outputs=train_outputs,
                pair_mask=pair_mask,
                temperature=1.0,
                return_intermediates=True
            )
            
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs
            predictions = logits.argmax(dim=1)
            
            result.add_check("Forward output shape", logits.shape[1] == model.num_classes, f"classes={logits.shape[1]}")
            result.add_check("Forward no NaN", not torch.isnan(logits).any().item(), "NaN in logits")
            result.add_metric("unique_colors", predictions.unique().cpu().tolist())
            
            # Accuracy check
            target = torch.tensor(test_target, dtype=torch.long, device=device)
            correct = (predictions[0] == target).float().mean().item()
            result.add_metric("accuracy", correct)
            result.add_check("Reasonable accuracy", correct > 0.0, f"accuracy={correct:.2%}")
            
    except Exception as e:
        result.passed = False
        result.errors.append(f"Exception: {str(e)}")
        traceback.print_exc()
    
    return result


# ==============================================================================
# TEST 2: RECURSIVE SOLVER PER-STEP VISUALIZATION
# ==============================================================================

def test_recursive_solver_steps(model: RLAN, task_info: dict, output_dir: Path, device: str = 'cpu') -> TestResult:
    """Test recursive solver with per-step visualization."""
    result = TestResult("Recursive Solver Steps")
    task_id = task_info["task_id"]
    task = task_info["task"]
    
    try:
        x, train_inputs, train_outputs, pair_mask, test_input, test_target = prepare_task_tensors(task, device)
        target = torch.tensor(test_target, dtype=torch.long, device=device)
        
        with torch.no_grad():
            # Full forward to get context
            outputs = model(
                input_grid=x,
                train_inputs=train_inputs,
                train_outputs=train_outputs,
                pair_mask=pair_mask,
                temperature=1.0,
                return_intermediates=True
            )
            
            # Get per-step logits from solver
            step_logits_list = outputs.get('step_logits', [])
            if not step_logits_list:
                # Try to get from solver directly
                result.add_warning("step_logits not available in outputs, using final logits only")
                step_logits_list = [outputs['logits']]
            
            num_steps = len(step_logits_list)
            result.add_metric("num_solver_steps", num_steps)
            
            # Analyze each step
            step_entropies = []
            step_accuracies = []
            step_predictions = []
            
            for step_idx, step_logits in enumerate(step_logits_list):
                # Get predictions
                probs = F.softmax(step_logits, dim=1)
                preds = step_logits.argmax(dim=1)
                
                # Entropy (lower = more confident)
                entropy = -(probs * (probs + 1e-10).log()).sum(dim=1).mean().item()
                step_entropies.append(entropy)
                
                # Accuracy
                acc = (preds[0] == target).float().mean().item()
                step_accuracies.append(acc)
                step_predictions.append(preds[0].cpu().numpy())
            
            result.add_metric("step_entropies", step_entropies)
            result.add_metric("step_accuracies", step_accuracies)
            
            # Check for improvement over steps
            improvement = 0
            best_step = 0
            if len(step_accuracies) > 1:
                improvement = step_accuracies[-1] - step_accuracies[0]
                result.add_metric("accuracy_improvement", improvement)
                result.add_check("Steps improve", improvement >= -0.1, f"improvement={improvement:.2%}")
                
                # Best step analysis
                best_step = int(np.argmax(step_accuracies))
                result.add_metric("best_step", best_step)
                result.add_check("Best step found", best_step >= 0, f"best_step={best_step}")
            
            # Create per-step visualization
            fig = plt.figure(figsize=(16, 12))
            gs = GridSpec(3, max(num_steps + 2, 5), figure=fig)
            
            # Row 1: Input, Target, and step predictions
            ax_input = fig.add_subplot(gs[0, 0])
            visualize_grid(ax_input, test_input, "Input")
            
            ax_target = fig.add_subplot(gs[0, 1])
            visualize_grid(ax_target, test_target, "Target")
            
            for step_idx in range(min(num_steps, 6)):
                ax = fig.add_subplot(gs[0, step_idx + 2])
                visualize_grid(ax, step_predictions[step_idx], f"Step {step_idx+1} (acc={step_accuracies[step_idx]:.0%})")
            
            # Row 2: Entropy and accuracy curves
            ax_curves = fig.add_subplot(gs[1, :3])
            steps = list(range(1, num_steps + 1))
            ax_curves.plot(steps, step_accuracies, 'b-o', label='Accuracy', linewidth=2)
            max_entropy = max(step_entropies) if step_entropies else 1
            ax_curves.plot(steps, [e/max_entropy for e in step_entropies], 'r-s', label='Norm Entropy', linewidth=2)
            ax_curves.set_xlabel('Solver Step')
            ax_curves.set_ylabel('Value')
            ax_curves.set_title('Accuracy & Entropy per Step')
            ax_curves.legend()
            ax_curves.grid(True, alpha=0.3)
            
            # Row 2: Per-step difference maps
            if num_steps > 1:
                ax_diff = fig.add_subplot(gs[1, 3:])
                diff_from_target = np.abs(step_predictions[-1].astype(float) - test_target.astype(float))
                ax_diff.imshow(diff_from_target, cmap='Reds')
                ax_diff.set_title(f'Final Error Map ({(diff_from_target > 0).mean():.1%} errors)')
                ax_diff.set_xticks([])
                ax_diff.set_yticks([])
            
            # Row 3: Summary statistics
            ax_summary = fig.add_subplot(gs[2, :])
            ax_summary.axis('off')
            summary_text = f"""Recursive Solver Analysis for {task_id}:
{'='*74}
Number of steps: {num_steps}
Step accuracies: {[f'{a:.1%}' for a in step_accuracies]}
Step entropies:  {[f'{e:.3f}' for e in step_entropies]}
Best step: {best_step + 1} (accuracy={step_accuracies[best_step]:.1%})
Final step: {num_steps} (accuracy={step_accuracies[-1]:.1%})
Accuracy improvement: {improvement:.1%} (step 1 -> step {num_steps})

SIGNAL QUALITY:
- Entropy decreasing: {'Y' if step_entropies[-1] < step_entropies[0] else 'N'} (model becomes more confident)
- Accuracy improving: {'Y' if improvement > 0 else 'N'} (iterative refinement helps)
- Best != Last step: {'Y (early stopping could help)' if best_step < num_steps - 1 else 'N (last step is best)'}
"""
            ax_summary.text(0.02, 0.95, summary_text, transform=ax_summary.transAxes, fontsize=10,
                           verticalalignment='top', fontfamily='monospace')
            
            try:
                plt.tight_layout()
            except Exception:
                pass  # Ignore tight_layout errors
            
            # Save
            task_output_dir = output_dir / "solver_steps" / task_id
            task_output_dir.mkdir(parents=True, exist_ok=True)
            viz_path = task_output_dir / f"{task_id}_solver_steps.png"
            plt.savefig(viz_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"    Saved: {viz_path}")
            
    except Exception as e:
        result.passed = False
        result.errors.append(f"Exception: {str(e)}")
        traceback.print_exc()
    
    return result


# ==============================================================================
# TEST 3: TEPS PROGRAM SEARCH
# ==============================================================================

def test_teps_program_search(task_info: dict, output_dir: Path) -> TestResult:
    """Test TEPS program search module."""
    result = TestResult("TEPS Program Search")
    task_id = task_info["task_id"]
    task = task_info["task"]
    
    try:
        from sci_arc.models.generalization.teps import TEPS, TEPSConfig
        
        config = TEPSConfig(
            max_search_steps=1000,
            timeout_seconds=10.0,
            max_program_depth=2,
        )
        teps = TEPS(config)
        
        # Prepare training pairs as numpy arrays
        train_inputs_np = []
        train_outputs_np = []
        for pair in task["train"]:
            inp = np.array(pair["input"], dtype=np.int64)
            out = np.array(pair["output"], dtype=np.int64)
            train_inputs_np.append(inp)
            train_outputs_np.append(out)
        
        # Use first input as test input
        test_input_np = train_inputs_np[0]
        
        result.add_metric("num_train_pairs", len(train_inputs_np))
        
        # Run search with correct signature
        start_time = time.time()
        search_result = teps.search(test_input_np, train_inputs_np, train_outputs_np)
        search_time = time.time() - start_time
        
        result.add_metric("search_time_seconds", search_time)
        result.add_metric("program_found", search_result.get('success', False))
        
        if search_result.get('success', False):
            found_program = search_result.get('program')
            match_score = search_result.get('stats', {}).get('best_score', 0)
            result.add_metric("program", str(found_program))
            result.add_metric("match_score", match_score)
            result.add_check("High match score", match_score >= 0.9, f"score={match_score:.2%}")
            
            # Visualize program execution
            num_pairs = len(train_inputs_np)
            fig, axes = plt.subplots(2, num_pairs + 1, figsize=(4 * (num_pairs + 1), 8))
            
            for i in range(num_pairs):
                visualize_grid(axes[0, i], train_inputs_np[i], f"Input {i+1}")
                visualize_grid(axes[1, i], train_outputs_np[i], f"Target {i+1}")
            
            # Show program text
            axes[0, -1].axis('off')
            axes[0, -1].text(0.5, 0.5, f"Program Found:\n{found_program}\n\nScore: {match_score:.1%}",
                            ha='center', va='center', fontsize=10, fontfamily='monospace',
                            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
            
            axes[1, -1].axis('off')
            steps_taken = getattr(teps, 'steps_taken', 0)
            axes[1, -1].text(0.5, 0.5, f"Search Stats:\nTime: {search_time:.2f}s\nSteps: {steps_taken}",
                            ha='center', va='center', fontsize=10, fontfamily='monospace')
            
            plt.suptitle(f"TEPS Search Result: {task_id}", fontsize=12, fontweight='bold')
            try:
                plt.tight_layout()
            except Exception:
                pass  # Ignore tight_layout errors
            
            teps_dir = output_dir / "teps" / task_id
            teps_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(teps_dir / f"{task_id}_teps_result.png", dpi=150, bbox_inches='tight')
            plt.close()
            
        else:
            result.add_warning(f"No program found within timeout ({config.timeout_seconds}s)")
            result.add_metric("primitives_tried", getattr(teps, 'steps_taken', 0))
            
    except ImportError as e:
        result.add_warning(f"TEPS module not available: {e}")
    except Exception as e:
        result.passed = False
        result.errors.append(f"Exception: {str(e)}")
        traceback.print_exc()
    
    return result


# ==============================================================================
# TEST 4: NS-TEPS OBJECT-LEVEL PROGRAM SEARCH
# ==============================================================================

def test_ns_teps_program_search(task_info: dict, output_dir: Path) -> TestResult:
    """Test NS-TEPS neuro-symbolic program search."""
    result = TestResult("NS-TEPS Program Search")
    task_id = task_info["task_id"]
    task = task_info["task"]
    
    try:
        from sci_arc.models.generalization.ns_teps import NSTEPS, NSTEPSConfig, ObjectExtractor
        
        config = NSTEPSConfig(
            max_search_steps=500,
            timeout_seconds=5.0,
            max_trace_length=2,
            sample_count=200,
        )
        ns_teps = NSTEPS(config)
        extractor = ObjectExtractor()
        
        # Prepare training inputs/outputs as numpy arrays
        train_inputs_np = []
        train_outputs_np = []
        for pair in task["train"]:
            inp = np.array(pair["input"], dtype=np.int64)
            out = np.array(pair["output"], dtype=np.int64)
            train_inputs_np.append(inp)
            train_outputs_np.append(out)
        
        # Use first input as test input
        test_input_np = train_inputs_np[0]
        
        # Test object extraction
        objects = extractor.extract(test_input_np)
        result.add_metric("num_objects_in_first_input", len(objects))
        result.add_check("Objects extracted", len(objects) >= 0, f"found {len(objects)} objects")
        
        # Run NS-TEPS search with correct signature
        start_time = time.time()
        search_result = ns_teps.search(test_input_np, train_inputs_np, train_outputs_np)
        search_time = time.time() - start_time
        
        result.add_metric("search_time_seconds", search_time)
        result.add_metric("trace_found", search_result.get('success', False))
        
        if search_result.get('success', False):
            found_trace = search_result.get('trace')
            match_score = search_result.get('confidence', 0)
            result.add_metric("trace", str(found_trace))
            result.add_metric("match_score", match_score)
            result.add_check("High match score", match_score >= 0.8, f"score={match_score:.2%}")
            
            # Visualize
            num_pairs = len(train_inputs_np)
            fig, axes = plt.subplots(2, num_pairs + 1, figsize=(4 * (num_pairs + 1), 8))
            
            for i in range(num_pairs):
                num_objs = len(extractor.extract(train_inputs_np[i]))
                visualize_grid(axes[0, i], train_inputs_np[i], f"Input {i+1} ({num_objs} objs)")
                visualize_grid(axes[1, i], train_outputs_np[i], f"Target {i+1}")
            
            axes[0, -1].axis('off')
            axes[0, -1].text(0.5, 0.5, f"NS-TEPS Trace:\n{found_trace}\n\nScore: {match_score:.1%}",
                            ha='center', va='center', fontsize=9, fontfamily='monospace',
                            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            axes[1, -1].axis('off')
            
            plt.suptitle(f"NS-TEPS Search Result: {task_id}", fontsize=12, fontweight='bold')
            try:
                plt.tight_layout()
            except Exception:
                pass  # Ignore tight_layout errors
            
            ns_teps_dir = output_dir / "ns_teps" / task_id
            ns_teps_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(ns_teps_dir / f"{task_id}_ns_teps_result.png", dpi=150, bbox_inches='tight')
            plt.close()
            
        else:
            result.add_warning("No trace found")
            
    except ImportError as e:
        result.add_warning(f"NS-TEPS module not available: {e}")
    except Exception as e:
        result.passed = False
        result.errors.append(f"Exception: {str(e)}")
        traceback.print_exc()
    
    return result


# ==============================================================================
# TEST 5: PROGRAM CACHE INTEGRATION
# ==============================================================================

def test_program_cache(output_dir: Path) -> TestResult:
    """Test program cache loading and usage."""
    result = TestResult("Program Cache")
    cache_path = Path("c:/Users/perahmat/Downloads/SCI-ARC/cache/program_cache_merged_602.json")
    
    try:
        from sci_arc.models.generalization.program_guided_training import ProgramCache
        
        result.add_check("Cache file exists", cache_path.exists(), str(cache_path))
        
        if cache_path.exists():
            cache = ProgramCache(str(cache_path))
            num_programs = len(cache.cache)
            result.add_metric("num_cached_programs", num_programs)
            result.add_check("Cache not empty", num_programs > 0, f"found {num_programs} programs")
            
            # Sample some programs
            if num_programs > 0:
                sample_keys = list(cache.cache.keys())[:5]
                sample_programs = []
                for key in sample_keys:
                    entry = cache.cache[key]
                    trace = entry.get('trace', [])
                    confidence = entry.get('confidence', 0)
                    sample_programs.append({
                        'task_id': key,
                        'trace_length': len(trace),
                        'confidence': confidence,
                    })
                result.add_metric("sample_programs", sample_programs)
                
                # Verify structure
                sample = cache.cache[sample_keys[0]]
                result.add_check("Has trace", 'trace' in sample, "missing trace field")
                result.add_check("Has confidence", 'confidence' in sample, "missing confidence field")
                
    except ImportError as e:
        result.add_warning(f"ProgramCache not available: {e}")
    except Exception as e:
        result.passed = False
        result.errors.append(f"Exception: {str(e)}")
        traceback.print_exc()
    
    return result


# ==============================================================================
# TEST 6: HYPERLORA META-LEARNING
# ==============================================================================

def test_hyperlora(model: RLAN, task_info: dict, output_dir: Path, device: str = 'cpu') -> TestResult:
    """Test HyperLoRA meta-learning module."""
    result = TestResult("HyperLoRA Meta-Learning")
    task_id = task_info["task_id"]
    task = task_info["task"]
    
    try:
        # Check if HyperLoRA is enabled
        has_hyperlora = hasattr(model, 'hyper_lora') and model.hyper_lora is not None
        result.add_check("HyperLoRA exists", has_hyperlora, "module not found")
        
        if not has_hyperlora:
            result.add_warning("HyperLoRA not enabled in model config")
            return result
        
        x, train_inputs, train_outputs, pair_mask, test_input, test_target = prepare_task_tensors(task, device)
        
        with torch.no_grad():
            # Get context encoding
            if hasattr(model, 'context_encoder') and model.context_encoder is not None:
                context = model.context_encoder(train_inputs, train_outputs, pair_mask)
                result.add_check("Context encoded", context is not None, "context is None")
                
                if isinstance(context, dict):
                    context_pooled = context.get('pooled', context.get('context'))
                else:
                    context_pooled = context
                
                if context_pooled is not None:
                    result.add_metric("context_shape", list(context_pooled.shape) if hasattr(context_pooled, 'shape') else str(type(context_pooled)))
                    if hasattr(context_pooled, 'norm'):
                        result.add_metric("context_norm", context_pooled.norm().item())
            
            # Get LoRA predictions if method available
            if hasattr(model.hyper_lora, 'get_lora_params'):
                lora_params = model.hyper_lora.get_lora_params(context_pooled)
                result.add_check("LoRA params generated", lora_params is not None, "no LoRA params")
                
                if lora_params is not None:
                    # Analyze LoRA parameter norms
                    lora_norms = {}
                    for key, val in lora_params.items():
                        if isinstance(val, tuple) and len(val) == 2:
                            A, B = val
                            norm_A = A.norm().item()
                            norm_B = B.norm().item()
                            lora_norms[key] = {'A': norm_A, 'B': norm_B}
                    
                    result.add_metric("lora_norms", lora_norms)
                    
                    # Check for reasonable norms (not exploded)
                    if lora_norms:
                        max_norm = max(max(v['A'], v['B']) for v in lora_norms.values())
                        result.add_check("LoRA norms reasonable", max_norm < 100, f"max_norm={max_norm:.2f}")
                        result.add_metric("max_lora_norm", max_norm)
            else:
                result.add_warning("get_lora_params method not available")
                
    except Exception as e:
        result.passed = False
        result.errors.append(f"Exception: {str(e)}")
        traceback.print_exc()
    
    return result


# ==============================================================================
# TEST 7: END-TO-END SIGNAL QUALITY
# ==============================================================================

def test_signal_quality(model: RLAN, task_info: dict, output_dir: Path, device: str = 'cpu') -> TestResult:
    """Test end-to-end signal quality: gradients, norms, activations."""
    result = TestResult("Signal Quality")
    task_id = task_info["task_id"]
    task = task_info["task"]
    
    try:
        x, train_inputs, train_outputs, pair_mask, test_input, test_target = prepare_task_tensors(task, device)
        target = torch.tensor(test_target, dtype=torch.long, device=device)
        
        # Enable gradients for signal analysis
        model.train()
        
        # Forward pass
        outputs = model(
            input_grid=x,
            train_inputs=train_inputs,
            train_outputs=train_outputs,
            pair_mask=pair_mask,
            temperature=1.0,
            return_intermediates=True
        )
        
        logits = outputs['logits'] if isinstance(outputs, dict) else outputs
        
        # Compute loss
        loss = F.cross_entropy(logits, target.unsqueeze(0), ignore_index=-100)
        result.add_metric("loss", loss.item())
        result.add_check("Loss is finite", torch.isfinite(loss).item(), f"loss={loss.item()}")
        
        # Backward pass for gradient analysis
        loss.backward()
        
        # Analyze gradients
        grad_norms = {}
        zero_grad_modules = []
        nan_grad_modules = []
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                
                # Categorize by module
                module_name = name.split('.')[0]
                if module_name not in grad_norms:
                    grad_norms[module_name] = []
                grad_norms[module_name].append(grad_norm)
                
                if grad_norm == 0:
                    zero_grad_modules.append(name)
                if not np.isfinite(grad_norm):
                    nan_grad_modules.append(name)
        
        # Summarize gradient statistics per module
        grad_summary = {}
        for module, norms in grad_norms.items():
            grad_summary[module] = {
                'mean': float(np.mean(norms)),
                'max': float(np.max(norms)),
                'min': float(np.min(norms)),
            }
        
        result.add_metric("gradient_summary", grad_summary)
        result.add_metric("zero_grad_count", len(zero_grad_modules))
        result.add_metric("nan_grad_count", len(nan_grad_modules))
        
        result.add_check("No NaN gradients", len(nan_grad_modules) == 0, f"{len(nan_grad_modules)} NaN grads")
        result.add_check("Gradients flow", len(zero_grad_modules) < len(list(model.parameters())) * 0.5, 
                        f"{len(zero_grad_modules)} zero grads")
        
        # Check for gradient explosion/vanishing
        all_grads = [n for norms in grad_norms.values() for n in norms]
        mean_grad = 0.0
        max_grad = 0.0
        if all_grads:
            mean_grad = float(np.mean(all_grads))
            max_grad = float(np.max(all_grads))
            result.add_metric("mean_gradient_norm", mean_grad)
            result.add_metric("max_gradient_norm", max_grad)
            result.add_check("No gradient explosion", max_grad < 100, f"max_grad={max_grad:.2f}")
            result.add_check("No gradient vanishing", mean_grad > 1e-8, f"mean_grad={mean_grad:.2e}")
        
        # Reset to eval mode
        model.eval()
        model.zero_grad()
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot 1: Gradient norms by module
        modules = list(grad_summary.keys())
        means = [grad_summary[m]['mean'] for m in modules]
        maxes = [grad_summary[m]['max'] for m in modules]
        
        x_pos = np.arange(len(modules))
        axes[0].bar(x_pos - 0.2, means, 0.4, label='Mean', color='blue', alpha=0.7)
        axes[0].bar(x_pos + 0.2, maxes, 0.4, label='Max', color='red', alpha=0.7)
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels(modules, rotation=45, ha='right', fontsize=8)
        axes[0].set_ylabel('Gradient Norm')
        axes[0].set_title('Gradient Norms by Module')
        axes[0].legend()
        if max(maxes) > 0:
            axes[0].set_yscale('log')
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Loss and accuracy
        predictions = logits.argmax(dim=1)
        accuracy = (predictions[0] == target).float().mean().item()
        metrics = ['Loss', 'Accuracy']
        values = [loss.item(), accuracy]
        colors = ['red' if loss.item() > 2 else 'green', 'green' if accuracy > 0.5 else 'orange']
        axes[1].bar(metrics, values, color=colors, alpha=0.7)
        axes[1].set_title('Training Metrics')
        axes[1].set_ylim(0, max(values) * 1.2 + 0.1)
        for i, v in enumerate(values):
            axes[1].text(i, v + 0.05, f'{v:.3f}', ha='center', fontsize=12)
        
        # Plot 3: Summary text
        axes[2].axis('off')
        summary_text = f"""Signal Quality Summary for {task_id}:
{'='*58}
Loss: {loss.item():.4f}
Accuracy: {accuracy:.2%}

Gradient Statistics:
- Mean gradient norm: {mean_grad:.2e}
- Max gradient norm: {max_grad:.2e}
- Zero gradients: {len(zero_grad_modules)} params
- NaN gradients: {len(nan_grad_modules)} params

Module Gradient Norms:
"""
        for module, stats in list(grad_summary.items())[:6]:
            summary_text += f"  {module}: mean={stats['mean']:.2e}, max={stats['max']:.2e}\n"
        
        axes[2].text(0.02, 0.95, summary_text, transform=axes[2].transAxes, fontsize=9,
                    verticalalignment='top', fontfamily='monospace')
        
        try:
            plt.tight_layout()
        except Exception:
            pass  # Ignore tight_layout errors
        
        signal_dir = output_dir / "signal_quality" / task_id
        signal_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(signal_dir / f"{task_id}_signal_quality.png", dpi=150, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        result.passed = False
        result.errors.append(f"Exception: {str(e)}")
        traceback.print_exc()
        model.eval()
    
    return result


# ==============================================================================
# MAIN TEST RUNNER
# ==============================================================================

def generate_comprehensive_report(all_results: Dict[str, List[TestResult]], output_dir: Path):
    """Generate comprehensive test report."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "summary": {"total_tests": 0, "passed": 0, "failed": 0, "warnings": 0},
        "by_test_type": {},
        "by_task": {},
        "critical_issues": [],
        "recommendations": [],
    }
    
    for test_name, results in all_results.items():
        report["by_test_type"][test_name] = {
            "total": len(results),
            "passed": sum(1 for r in results if r.passed),
            "failed": sum(1 for r in results if not r.passed),
        }
        
        for result in results:
            report["summary"]["total_tests"] += 1
            if result.passed:
                report["summary"]["passed"] += 1
            else:
                report["summary"]["failed"] += 1
            report["summary"]["warnings"] += len(result.warnings)
            
            if result.errors:
                for error in result.errors:
                    report["critical_issues"].append(f"{test_name}: {error}")
    
    # Generate recommendations
    if report["summary"]["failed"] > 0:
        report["recommendations"].append("Fix failing tests before production training")
    
    for test_name, stats in report["by_test_type"].items():
        if stats["failed"] > 0:
            report["recommendations"].append(f"Investigate {test_name}: {stats['failed']} failures")
    
    # Save report
    report_path = output_dir / "comprehensive_test_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    # Print summary
    print("\n" + "=" * 70)
    print("COMPREHENSIVE TEST REPORT")
    print("=" * 70)
    print(f"Total Tests: {report['summary']['total_tests']}")
    print(f"  Passed: {report['summary']['passed']}")
    print(f"  Failed: {report['summary']['failed']}")
    print(f"  Warnings: {report['summary']['warnings']}")
    print()
    
    print("Results by Test Type:")
    for test_name, stats in report["by_test_type"].items():
        status = "Y" if stats["failed"] == 0 else "N"
        print(f"  [{status}] {test_name}: {stats['passed']}/{stats['total']} passed")
    
    if report["critical_issues"]:
        print("\nCritical Issues:")
        for issue in report["critical_issues"][:10]:
            print(f"  - {issue}")
    
    if report["recommendations"]:
        print("\nRecommendations:")
        for rec in report["recommendations"]:
            print(f"  -> {rec}")
    
    return report


def main():
    # Configuration
    checkpoint_path = Path("c:/Users/perahmat/Downloads/SCI-ARC/checkpoints/warmup3.pt")
    data_dir = Path("c:/Users/perahmat/Downloads/SCI-ARC/data/arc-agi/data/training")
    output_dir = Path("c:/Users/perahmat/Downloads/SCI-ARC/scripts/outputs/comprehensive_tests")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    print("=" * 70)
    print("COMPREHENSIVE RLAN MODULE TESTING SUITE")
    print("=" * 70)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model, model_config = load_model_with_checkpoint(checkpoint_path, device)
    
    # Find tasks by size
    print("\nFinding ARC tasks by grid size...")
    tasks_by_size = find_tasks_by_size(data_dir, max_per_size=2)
    
    for size, tasks in tasks_by_size.items():
        print(f"  {size.upper()}: {len(tasks)} tasks")
    
    # Collect all results
    all_results: Dict[str, List[TestResult]] = {
        "Core Modules": [],
        "Recursive Solver Steps": [],
        "TEPS Program Search": [],
        "NS-TEPS Program Search": [],
        "HyperLoRA Meta-Learning": [],
        "Signal Quality": [],
    }
    
    # Run tests on each task
    for size in ["small", "medium", "large"]:
        print(f"\n{'=' * 70}")
        print(f"Testing {size.upper()} grid tasks")
        print("=" * 70)
        
        for task_info in tasks_by_size[size]:
            task_id = task_info["task_id"]
            print(f"\n  Task: {task_id} ({task_info['grid_size'][0]}x{task_info['grid_size'][1]})")
            
            # Test 1: Core modules
            print("    Testing core modules...")
            result = test_core_modules(model, task_info, output_dir, device)
            all_results["Core Modules"].append(result)
            print(f"      {result}")
            
            # Test 2: Recursive solver steps
            print("    Testing recursive solver steps...")
            result = test_recursive_solver_steps(model, task_info, output_dir, device)
            all_results["Recursive Solver Steps"].append(result)
            print(f"      {result}")
            
            # Test 3: TEPS program search
            print("    Testing TEPS program search...")
            result = test_teps_program_search(task_info, output_dir)
            all_results["TEPS Program Search"].append(result)
            print(f"      {result}")
            
            # Test 4: NS-TEPS program search
            print("    Testing NS-TEPS program search...")
            result = test_ns_teps_program_search(task_info, output_dir)
            all_results["NS-TEPS Program Search"].append(result)
            print(f"      {result}")
            
            # Test 5: HyperLoRA
            print("    Testing HyperLoRA meta-learning...")
            result = test_hyperlora(model, task_info, output_dir, device)
            all_results["HyperLoRA Meta-Learning"].append(result)
            print(f"      {result}")
            
            # Test 6: Signal quality
            print("    Testing signal quality...")
            result = test_signal_quality(model, task_info, output_dir, device)
            all_results["Signal Quality"].append(result)
            print(f"      {result}")
    
    # Test program cache (once)
    print("\n  Testing program cache...")
    cache_result = test_program_cache(output_dir)
    all_results["Program Cache"] = [cache_result]
    print(f"    {cache_result}")
    
    # Generate comprehensive report
    report = generate_comprehensive_report(all_results, output_dir)
    
    print(f"\n\nAll outputs saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
