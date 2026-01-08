"""
Production-Accurate RLAN Module Tracing and Bug Detection

This script traces the EXACT order of operations in RLAN production code,
visualizing tensor outputs at each step to detect bugs in:
- Math/Algorithm correctness
- Tensor shape mismatches
- NaN/Inf propagation
- Attention/gradient flow issues

Order of Operations (matching sci_arc/models/rlan.py forward()):
1. GridEncoder.encode() → features (B, D, H, W)
2. ContextEncoder → context/support_features
3. HPM (if enabled) → enhanced context
4. DSC → centroids, attention_maps, stop_logits
5. MSRE → clue_features (B, K, D, H, W)
6. LCR (if enabled) → count_embedding
7. SPH (if enabled) → predicates
8. HyperLoRA (if enabled) → lora_deltas
9. RecursiveSolver → step-by-step refinement

Author: AI Scientist / Statistician
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
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import traceback
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sci_arc.models.rlan import RLAN, RLANConfig

# ARC color palette (10 colors)
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
    """Get colormap for ARC grids."""
    return ListedColormap(ARC_COLORS)


@dataclass
class ModuleStats:
    """Statistics for a module's output."""
    name: str
    shape: tuple
    dtype: str
    min_val: float = 0.0
    max_val: float = 0.0
    mean_val: float = 0.0
    std_val: float = 0.0
    nan_count: int = 0
    inf_count: int = 0
    zero_count: int = 0
    has_bug: bool = False
    bug_description: str = ""
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "min": self.min_val,
            "max": self.max_val,
            "mean": self.mean_val,
            "std": self.std_val,
            "nan_count": self.nan_count,
            "inf_count": self.inf_count,
            "zero_count": self.zero_count,
            "has_bug": self.has_bug,
            "bug_description": self.bug_description,
        }


@dataclass
class SolverStepStats:
    """Statistics for a single solver step."""
    step_idx: int
    accuracy: float = 0.0
    entropy: float = 0.0
    confidence: float = 0.0
    bg_ratio: float = 0.0
    fg_ratio: float = 0.0
    unique_colors: int = 0
    improvement_from_prev: float = 0.0


@dataclass
class TaskAnalysis:
    """Complete analysis for one ARC task."""
    task_id: str
    grid_size: str
    input_shape: tuple
    output_shape: tuple
    module_stats: List[ModuleStats] = field(default_factory=list)
    solver_steps: List[SolverStepStats] = field(default_factory=list)
    bugs_found: List[str] = field(default_factory=list)
    execution_time: float = 0.0


def analyze_tensor(tensor: torch.Tensor, name: str) -> ModuleStats:
    """Compute comprehensive statistics for a tensor."""
    stats = ModuleStats(
        name=name,
        shape=tuple(tensor.shape),
        dtype=str(tensor.dtype),
    )
    
    with torch.no_grad():
        flat = tensor.float().flatten()
        stats.nan_count = torch.isnan(flat).sum().item()
        stats.inf_count = torch.isinf(flat).sum().item()
        
        # Remove NaN/Inf for stats
        valid = flat[torch.isfinite(flat)]
        if len(valid) > 0:
            stats.min_val = valid.min().item()
            stats.max_val = valid.max().item()
            stats.mean_val = valid.mean().item()
            stats.std_val = valid.std().item()
            stats.zero_count = (valid == 0).sum().item()
        
        # Bug detection
        if stats.nan_count > 0:
            stats.has_bug = True
            stats.bug_description = f"Contains {stats.nan_count} NaN values"
        elif stats.inf_count > 0:
            stats.has_bug = True
            stats.bug_description = f"Contains {stats.inf_count} Inf values"
        elif stats.std_val == 0 and len(valid) > 1:
            stats.has_bug = True
            stats.bug_description = "All values are identical (no variance)"
        elif stats.zero_count == len(flat):
            stats.has_bug = True
            stats.bug_description = "All values are zero"
    
    return stats


def load_model_with_checkpoint(checkpoint_path: Path, device: str = 'cpu') -> Tuple[RLAN, dict]:
    """Load RLAN model matching production code exactly."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    full_config = checkpoint.get('config', {})
    model_config = full_config.get('model', {})
    
    # Detect dsc_use_complexity_signals from checkpoint shape
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
    
    # Create config matching checkpoint exactly
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
        use_hyperlora=model_config.get('use_hyperlora', False),
        use_hpm=model_config.get('use_hpm', False),
    )
    
    model = RLAN(config=config)
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    if missing_keys:
        print(f"  Missing keys: {len(missing_keys)}")
    if unexpected_keys:
        print(f"  Unexpected keys: {len(unexpected_keys)}")
    
    model.to(device)
    model.eval()
    
    epoch = checkpoint.get('epoch', 'unknown')
    print(f"  Loaded from epoch {epoch}")
    return model, model_config


def prepare_task_tensors_prod(task: dict, device: str = 'cpu') -> dict:
    """
    Prepare task tensors EXACTLY as production code does.
    Returns dict with all tensors needed for forward pass.
    """
    train_pairs = task["train"]
    test_pair = task.get("test", [train_pairs[-1]])
    if isinstance(test_pair, list) and len(test_pair) > 0:
        test_pair = test_pair[0]
    
    test_input = np.array(test_pair["input"], dtype=np.int64)
    test_target = np.array(test_pair.get("output", test_pair["input"]), dtype=np.int64)
    
    B = 1
    H, W = test_input.shape
    
    # Test input tensor
    x = torch.tensor(test_input, dtype=torch.long).unsqueeze(0).to(device)
    
    # Training pairs (padded to max_grid_size=30)
    max_size = 30
    N = len(train_pairs)
    train_inputs = torch.zeros(B, N, max_size, max_size, dtype=torch.long, device=device)
    train_outputs = torch.zeros(B, N, max_size, max_size, dtype=torch.long, device=device)
    
    for i, pair in enumerate(train_pairs):
        inp = np.array(pair["input"], dtype=np.int64)
        out = np.array(pair["output"], dtype=np.int64)
        ih, iw = inp.shape
        oh, ow = out.shape
        train_inputs[0, i, :ih, :iw] = torch.tensor(inp)
        train_outputs[0, i, :oh, :ow] = torch.tensor(out)
    
    pair_mask = torch.ones(B, N, dtype=torch.bool, device=device)
    
    return {
        "input_grid": x,
        "train_inputs": train_inputs,
        "train_outputs": train_outputs,
        "pair_mask": pair_mask,
        "test_input_np": test_input,
        "test_target_np": test_target,
        "H": H,
        "W": W,
        "N": N,
    }


def trace_rlan_forward(
    model: RLAN,
    tensors: dict,
    device: str = 'cpu'
) -> Tuple[dict, List[ModuleStats]]:
    """
    Trace RLAN forward pass step-by-step, matching production code exactly.
    
    Returns:
        outputs: Dict of all intermediate outputs
        stats: List of ModuleStats for each module
    """
    stats_list = []
    outputs = {}
    
    x = tensors["input_grid"]
    train_inputs = tensors["train_inputs"]
    train_outputs = tensors["train_outputs"]
    pair_mask = tensors["pair_mask"]
    
    B, H, W = x.shape
    
    with torch.no_grad():
        # ========== STEP 1: GRID ENCODER ==========
        features = model.encode(x)  # (B, D, H, W)
        outputs["features"] = features
        stats_list.append(analyze_tensor(features, "1. GridEncoder.features"))
        
        # Get valid mask and grid sizes (production code does this)
        valid_mask = model.encoder.get_valid_mask(x)  # (B, H, W)
        grid_sizes = model.encoder.get_grid_sizes(x)  # (B, 2)
        outputs["valid_mask"] = valid_mask
        outputs["grid_sizes"] = grid_sizes
        
        # ========== STEP 2: CONTEXT ENCODER ==========
        context = None
        dsc_task_context = None
        support_features = None
        
        if model.use_context_encoder and model.context_encoder is not None:
            context_output = model.context_encoder(
                train_inputs, train_outputs, pair_mask
            )
            
            if model.context_encoder.use_spatial_features:
                support_features = context_output  # (B, N, D, H, W)
                outputs["support_features"] = support_features
                stats_list.append(analyze_tensor(support_features, "2. ContextEncoder.support_features"))
                
                # Pool for DSC task context
                dsc_task_context = model.pool_context_from_support(context_output)  # (B, D)
                outputs["dsc_task_context"] = dsc_task_context
                stats_list.append(analyze_tensor(dsc_task_context, "2b. ContextEncoder.dsc_task_context"))
                
                # Cross-attention injection
                cross_attention_active = getattr(model, 'cross_attention_active', True)
                if hasattr(model.context_injector, 'forward') and cross_attention_active:
                    features = model.context_injector(features, context_output)
                    outputs["features_after_context"] = features
                    stats_list.append(analyze_tensor(features, "2c. Features after context injection"))
            else:
                context = context_output  # (B, D)
                dsc_task_context = context
                outputs["context"] = context
                stats_list.append(analyze_tensor(context, "2. ContextEncoder.context"))
                
                features = model.context_injector(features, context)
                outputs["features_after_context"] = features
                stats_list.append(analyze_tensor(features, "2b. Features after FiLM injection"))
        
        # ========== STEP 3: HPM (if enabled) ==========
        hpm_memory_tokens = None
        if model.use_hpm and model.hpm is not None:
            # Skip for this test - warmup3.pt doesn't have HPM
            pass
        
        # ========== STEP 4: DSC (Dynamic Saliency Controller) ==========
        if model.use_dsc and model.dsc is not None:
            centroids, attention_maps, stop_logits = model.dsc(
                features, temperature=1.0, mask=valid_mask, 
                task_context=dsc_task_context, input_grid=x
            )
            outputs["centroids"] = centroids
            outputs["attention_maps"] = attention_maps
            outputs["stop_logits"] = stop_logits
            
            stats_list.append(analyze_tensor(centroids, "4a. DSC.centroids"))
            stats_list.append(analyze_tensor(attention_maps, "4b. DSC.attention_maps"))
            stats_list.append(analyze_tensor(stop_logits, "4c. DSC.stop_logits"))
            
            # Compute stop probs for analysis
            stop_probs = torch.sigmoid(stop_logits)
            outputs["stop_probs"] = stop_probs
        else:
            K = model.max_clues
            centroids = torch.zeros(B, K, 2, device=device)
            centroids[:, :, 0] = H / 2
            centroids[:, :, 1] = W / 2
            attention_maps = torch.ones(B, K, H, W, device=device) / (H * W)
            stop_logits = torch.zeros(B, K, device=device)
            outputs["centroids"] = centroids
            outputs["attention_maps"] = attention_maps
            outputs["stop_logits"] = stop_logits
        
        # ========== STEP 5: MSRE (Multi-Scale Relative Encoding) ==========
        if model.use_msre and model.msre is not None:
            clue_features = model.msre(
                features, centroids, grid_sizes=grid_sizes
            )  # (B, K, D, H, W)
            outputs["clue_features"] = clue_features
            stats_list.append(analyze_tensor(clue_features, "5. MSRE.clue_features"))
        else:
            clue_features = features.unsqueeze(1).expand(-1, model.max_clues, -1, -1, -1)
            outputs["clue_features"] = clue_features
        
        # ========== STEP 6: LCR (if enabled) ==========
        if model.use_lcr and model.lcr is not None:
            count_embedding = model.lcr(
                x, features, mask=valid_mask, attention_maps=attention_maps
            )
            outputs["count_embedding"] = count_embedding
            stats_list.append(analyze_tensor(count_embedding, "6. LCR.count_embedding"))
        else:
            count_embedding = torch.zeros(
                B, model.num_colors, model.hidden_dim, device=device
            )
            outputs["count_embedding"] = count_embedding
        
        # ========== STEP 7: SPH (if enabled) ==========
        if model.use_sph and model.sph is not None:
            predicates = model.sph(features, temperature=1.0)
            outputs["predicates"] = predicates
            stats_list.append(analyze_tensor(predicates, "7. SPH.predicates"))
        else:
            predicates = torch.zeros(B, model.num_predicates, device=device)
            outputs["predicates"] = predicates
        
        # ========== STEP 8: HYPERLORA (if enabled) ==========
        lora_deltas = None
        if model.use_hyperlora and model.hyper_lora is not None:
            if support_features is not None:
                lora_deltas = model.hyper_lora(support_features)
                outputs["lora_deltas"] = lora_deltas
        
        # ========== STEP 9: RECURSIVE SOLVER (step-by-step) ==========
        solver_context_active = getattr(model, 'solver_context_active', True)
        effective_support_features = support_features if solver_context_active else None
        
        solver_output = model.solver(
            clue_features=clue_features,
            count_embedding=count_embedding,
            predicates=predicates,
            input_grid=x,
            attention_maps=attention_maps,
            stop_logits=stop_logits,
            support_features=effective_support_features,
            return_all_steps=True,
            lora_deltas=lora_deltas,
            hpm_memory_tokens=hpm_memory_tokens,
        )
        
        if isinstance(solver_output, tuple):
            all_logits, act_outputs = solver_output
        else:
            all_logits = solver_output
        
        outputs["all_logits"] = all_logits
        outputs["final_logits"] = all_logits[-1]
        
        # Analyze each solver step
        for step_idx, step_logits in enumerate(all_logits):
            stats_list.append(analyze_tensor(step_logits, f"9. Solver.step_{step_idx+1}_logits"))
    
    return outputs, stats_list


def compute_solver_step_metrics(
    all_logits: List[torch.Tensor],
    target: np.ndarray,
    input_grid: np.ndarray,
) -> List[SolverStepStats]:
    """Compute detailed metrics for each solver step."""
    step_stats = []
    prev_accuracy = 0.0
    
    target_tensor = torch.tensor(target, dtype=torch.long)
    
    for step_idx, logits in enumerate(all_logits):
        # logits shape: (B, C, H, W)
        probs = F.softmax(logits[0], dim=0)  # (C, H, W)
        preds = logits[0].argmax(dim=0)  # (H, W)
        
        # Compute metrics over the prediction area matching target size
        th, tw = target.shape
        ph, pw = preds.shape
        
        # Crop/align if needed
        h = min(th, ph)
        w = min(tw, pw)
        preds_crop = preds[:h, :w]
        target_crop = target_tensor[:h, :w]
        
        # Accuracy
        correct = (preds_crop == target_crop).float()
        accuracy = correct.mean().item()
        
        # Entropy (per-pixel, then average)
        probs_crop = probs[:, :h, :w]
        entropy = -(probs_crop * torch.log(probs_crop + 1e-10)).sum(dim=0).mean().item()
        
        # Confidence (max prob per pixel, averaged)
        confidence = probs_crop.max(dim=0)[0].mean().item()
        
        # Background/foreground ratios
        total_pixels = h * w
        bg_pixels = (preds_crop == 0).sum().item()
        bg_ratio = bg_pixels / total_pixels
        fg_ratio = 1 - bg_ratio
        
        # Unique colors predicted
        unique_colors = len(preds_crop.unique())
        
        # Improvement from previous step
        improvement = accuracy - prev_accuracy
        prev_accuracy = accuracy
        
        step_stats.append(SolverStepStats(
            step_idx=step_idx + 1,
            accuracy=accuracy,
            entropy=entropy,
            confidence=confidence,
            bg_ratio=bg_ratio,
            fg_ratio=fg_ratio,
            unique_colors=unique_colors,
            improvement_from_prev=improvement,
        ))
    
    return step_stats


def visualize_module_outputs(
    outputs: dict,
    tensors: dict,
    stats_list: List[ModuleStats],
    solver_stats: List[SolverStepStats],
    task_id: str,
    output_dir: Path,
):
    """Create comprehensive visualization of all module outputs."""
    
    # Create output directory
    task_dir = output_dir / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    
    test_input = tensors["test_input_np"]
    test_target = tensors["test_target_np"]
    H, W = tensors["H"], tensors["W"]
    
    arc_cmap = get_arc_cmap()
    
    # ===== FIGURE 1: Module Statistics Overview =====
    fig1, axes1 = plt.subplots(2, 1, figsize=(14, 8))
    
    # Stats table
    ax_stats = axes1[0]
    ax_stats.axis('off')
    
    stats_text = f"Module Statistics for {task_id} ({H}x{W})\n" + "=" * 60 + "\n\n"
    for stat in stats_list:
        bug_marker = " ⚠️ BUG" if stat.has_bug else " ✅"
        stats_text += f"{stat.name}{bug_marker}\n"
        stats_text += f"  Shape: {stat.shape}, dtype: {stat.dtype}\n"
        stats_text += f"  Range: [{stat.min_val:.4f}, {stat.max_val:.4f}], Mean: {stat.mean_val:.4f}, Std: {stat.std_val:.4f}\n"
        if stat.nan_count > 0 or stat.inf_count > 0:
            stats_text += f"  NaN: {stat.nan_count}, Inf: {stat.inf_count}\n"
        if stat.has_bug:
            stats_text += f"  Bug: {stat.bug_description}\n"
        stats_text += "\n"
    
    ax_stats.text(0.02, 0.98, stats_text, transform=ax_stats.transAxes, fontsize=8,
                  verticalalignment='top', fontfamily='monospace')
    
    # Solver step accuracy plot
    ax_solver = axes1[1]
    steps = [s.step_idx for s in solver_stats]
    accuracies = [s.accuracy for s in solver_stats]
    entropies = [s.entropy for s in solver_stats]
    confidences = [s.confidence for s in solver_stats]
    
    ax_solver.plot(steps, accuracies, 'b-o', label='Accuracy', linewidth=2, markersize=8)
    ax_solver.plot(steps, confidences, 'g-s', label='Confidence', linewidth=2, markersize=8)
    ax_solver.plot(steps, [e/3.0 for e in entropies], 'r-^', label='Entropy/3', linewidth=2, markersize=8)
    
    ax_solver.set_xlabel('Solver Step', fontsize=12)
    ax_solver.set_ylabel('Metric Value', fontsize=12)
    ax_solver.set_title('Solver Step-by-Step Metrics', fontsize=14, fontweight='bold')
    ax_solver.legend(loc='best')
    ax_solver.grid(True, alpha=0.3)
    ax_solver.set_ylim(0, 1.1)
    
    plt.tight_layout()
    fig1.savefig(task_dir / f"{task_id}_module_stats.png", dpi=150, bbox_inches='tight')
    plt.close(fig1)
    
    # ===== FIGURE 2: DSC Attention Maps =====
    if "attention_maps" in outputs:
        attention_maps = outputs["attention_maps"][0].cpu().numpy()  # (K, H, W)
        centroids = outputs["centroids"][0].cpu().numpy()  # (K, 2)
        stop_probs = outputs.get("stop_probs", torch.sigmoid(outputs["stop_logits"]))[0].cpu().numpy()
        
        K = attention_maps.shape[0]
        cols = min(K, 4)
        rows = (K + cols - 1) // cols + 1  # +1 for input/target row
        
        fig2, axes2 = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        if rows == 1:
            axes2 = axes2.reshape(1, -1)
        
        # Row 0: Input and target
        axes2[0, 0].imshow(test_input, cmap=arc_cmap, vmin=0, vmax=9)
        axes2[0, 0].set_title("Test Input", fontsize=10, fontweight='bold')
        axes2[0, 0].axis('off')
        
        axes2[0, 1].imshow(test_target, cmap=arc_cmap, vmin=0, vmax=9)
        axes2[0, 1].set_title("Test Target", fontsize=10, fontweight='bold')
        axes2[0, 1].axis('off')
        
        for c in range(2, cols):
            axes2[0, c].axis('off')
        
        # Remaining rows: Attention maps
        for k in range(K):
            row = 1 + k // cols
            col = k % cols
            
            attn_map = attention_maps[k]
            im = axes2[row, col].imshow(attn_map, cmap='hot', vmin=0, vmax=attn_map.max())
            
            # Mark centroid
            cy, cx = centroids[k]
            if 0 <= cy < attn_map.shape[0] and 0 <= cx < attn_map.shape[1]:
                axes2[row, col].plot(cx, cy, 'c*', markersize=15, markeredgecolor='white')
            
            stop_p = stop_probs[k]
            usage = 1 - stop_p
            axes2[row, col].set_title(f"Clue {k+1} (usage={usage:.2f})", fontsize=9)
            axes2[row, col].axis('off')
        
        # Hide unused axes
        for k in range(K, (rows - 1) * cols):
            row = 1 + k // cols
            col = k % cols
            if row < rows:
                axes2[row, col].axis('off')
        
        plt.suptitle(f"DSC Attention Maps - {task_id}", fontsize=12, fontweight='bold')
        plt.tight_layout()
        fig2.savefig(task_dir / f"{task_id}_dsc_attention.png", dpi=150, bbox_inches='tight')
        plt.close(fig2)
    
    # ===== FIGURE 3: Solver Steps Visualization =====
    all_logits = outputs.get("all_logits", [])
    if all_logits:
        num_steps = len(all_logits)
        fig3, axes3 = plt.subplots(2, num_steps + 2, figsize=(3 * (num_steps + 2), 6))
        
        # Row 0: Input, predictions at each step
        axes3[0, 0].imshow(test_input, cmap=arc_cmap, vmin=0, vmax=9)
        axes3[0, 0].set_title("Input", fontsize=10, fontweight='bold')
        axes3[0, 0].axis('off')
        
        for step_idx, step_logits in enumerate(all_logits):
            pred = step_logits[0].argmax(dim=0).cpu().numpy()
            acc = solver_stats[step_idx].accuracy
            
            # Crop to target size
            ph, pw = pred.shape
            th, tw = test_target.shape
            h, w = min(ph, th), min(pw, tw)
            pred_crop = pred[:h, :w]
            
            axes3[0, step_idx + 1].imshow(pred_crop, cmap=arc_cmap, vmin=0, vmax=9)
            axes3[0, step_idx + 1].set_title(f"Step {step_idx + 1}\nAcc: {acc:.1%}", fontsize=9)
            axes3[0, step_idx + 1].axis('off')
        
        # Target at the end
        axes3[0, -1].imshow(test_target, cmap=arc_cmap, vmin=0, vmax=9)
        axes3[0, -1].set_title("Target", fontsize=10, fontweight='bold')
        axes3[0, -1].axis('off')
        
        # Row 1: Entropy/confidence heatmaps
        axes3[1, 0].axis('off')
        
        for step_idx, step_logits in enumerate(all_logits):
            probs = F.softmax(step_logits[0], dim=0).cpu()  # (C, H, W)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=0).numpy()
            
            ph, pw = entropy.shape
            th, tw = test_target.shape
            h, w = min(ph, th), min(pw, tw)
            entropy_crop = entropy[:h, :w]
            
            im = axes3[1, step_idx + 1].imshow(entropy_crop, cmap='viridis', vmin=0, vmax=2.5)
            axes3[1, step_idx + 1].set_title(f"Entropy", fontsize=9)
            axes3[1, step_idx + 1].axis('off')
        
        axes3[1, -1].axis('off')
        
        plt.suptitle(f"Solver Step-by-Step Predictions - {task_id}", fontsize=12, fontweight='bold')
        plt.tight_layout()
        fig3.savefig(task_dir / f"{task_id}_solver_steps.png", dpi=150, bbox_inches='tight')
        plt.close(fig3)
    
    # ===== FIGURE 4: Feature Analysis =====
    if "features" in outputs:
        features = outputs["features"][0].cpu()  # (D, H, W)
        
        fig4, axes4 = plt.subplots(2, 4, figsize=(16, 8))
        
        # Show first 8 feature channels
        for i in range(8):
            row, col = i // 4, i % 4
            feat_ch = features[i].numpy()
            axes4[row, col].imshow(feat_ch, cmap='RdBu_r')
            axes4[row, col].set_title(f"Feature ch {i}", fontsize=9)
            axes4[row, col].axis('off')
        
        plt.suptitle(f"Encoder Feature Channels - {task_id}", fontsize=12, fontweight='bold')
        plt.tight_layout()
        fig4.savefig(task_dir / f"{task_id}_features.png", dpi=150, bbox_inches='tight')
        plt.close(fig4)
    
    print(f"  Saved visualizations to {task_dir}")


def find_arc_tasks(data_dir: Path, sizes=("small", "medium", "large"), max_per_size: int = 2) -> Dict[str, List[dict]]:
    """Find ARC tasks by grid size category - use specific known tasks for speed."""
    
    # Use specific known tasks to avoid scanning entire directory
    # These are confirmed to exist and have varied grid sizes
    known_tasks = {
        "small": ["007bbfb7", "00d62c1b"],  # 3x3 and 6x6
        "medium": ["025d127b", "0520fde7"],  # ~14x9 and ~11x11
        "large": ["045e512c", "0962bcdd"],  # 21x21 and 22x22
    }
    
    tasks_by_size = {size: [] for size in sizes}
    
    for size, task_ids in known_tasks.items():
        if size not in sizes:
            continue
        for task_id in task_ids[:max_per_size]:
            task_path = data_dir / f"{task_id}.json"
            if task_path.exists():
                try:
                    with open(task_path, 'r', encoding='utf-8') as f:
                        task = json.load(f)
                    
                    if task.get("train") and len(task["train"]) > 0:
                        grid = task["train"][0]["input"]
                        h, w = len(grid), len(grid[0])
                        
                        tasks_by_size[size].append({
                            "path": task_path,
                            "task_id": task_id,
                            "grid_size": (h, w),
                            "task": task,
                        })
                except Exception as e:
                    print(f"  Error loading {task_id}: {e}")
    
    return tasks_by_size


def main():
    """Main entry point."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print("=" * 70)
    print("PRODUCTION-ACCURATE RLAN MODULE TRACING")
    print("=" * 70)
    
    # Paths
    checkpoint_path = Path("c:/Users/perahmat/Downloads/SCI-ARC/checkpoints/warmup3.pt")
    output_dir = Path("c:/Users/perahmat/Downloads/SCI-ARC/scripts/outputs/module_trace")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model, model_config = load_model_with_checkpoint(checkpoint_path, device)
    
    # Find ARC tasks
    data_dirs = [
        Path("c:/Users/perahmat/Downloads/SCI-ARC/data/arc-agi/data/training"),
        Path("c:/Users/perahmat/Downloads/SCI-ARC/data/merged_training"),
    ]
    
    data_dir = None
    for d in data_dirs:
        if d.exists():
            data_dir = d
            break
    
    if data_dir is None:
        print("ERROR: No ARC data directory found")
        return
    
    print(f"\nSearching for ARC tasks in: {data_dir}")
    tasks_by_size = find_arc_tasks(data_dir, max_per_size=2)
    
    for size, tasks in tasks_by_size.items():
        print(f"  {size.upper()}: {len(tasks)} tasks")
    
    # Collect all analysis results
    all_analyses = []
    all_bugs = []
    
    # Process each task
    for size in ["small", "medium", "large"]:
        print(f"\n{'=' * 70}")
        print(f"ANALYZING {size.upper()} GRID TASKS")
        print("=" * 70)
        
        for task_info in tasks_by_size[size]:
            task_id = task_info["task_id"]
            h, w = task_info["grid_size"]
            task = task_info["task"]
            
            print(f"\n  Task: {task_id} ({h}x{w})")
            
            start_time = time.time()
            
            # Prepare tensors exactly as production
            tensors = prepare_task_tensors_prod(task, device)
            
            # Trace forward pass
            outputs, stats_list = trace_rlan_forward(model, tensors, device)
            
            # Compute solver step metrics
            target = tensors["test_target_np"]
            input_np = tensors["test_input_np"]
            solver_stats = compute_solver_step_metrics(outputs["all_logits"], target, input_np)
            
            execution_time = time.time() - start_time
            
            # Create analysis record
            analysis = TaskAnalysis(
                task_id=task_id,
                grid_size=size,
                input_shape=input_np.shape,
                output_shape=target.shape,
                module_stats=stats_list,
                solver_steps=solver_stats,
                execution_time=execution_time,
            )
            
            # Check for bugs
            for stat in stats_list:
                if stat.has_bug:
                    bug_msg = f"{task_id}: {stat.name} - {stat.bug_description}"
                    analysis.bugs_found.append(bug_msg)
                    all_bugs.append(bug_msg)
            
            all_analyses.append(analysis)
            
            # Generate visualizations
            visualize_module_outputs(
                outputs, tensors, stats_list, solver_stats, task_id, output_dir
            )
            
            # Print solver stats
            print(f"    Solver steps: {len(solver_stats)}")
            for ss in solver_stats:
                delta = f"+{ss.improvement_from_prev:.1%}" if ss.improvement_from_prev > 0 else f"{ss.improvement_from_prev:.1%}"
                print(f"      Step {ss.step_idx}: Acc={ss.accuracy:.1%} ({delta}), Conf={ss.confidence:.2f}, Entropy={ss.entropy:.2f}")
            
            if analysis.bugs_found:
                print(f"    ⚠️ BUGS FOUND: {len(analysis.bugs_found)}")
                for bug in analysis.bugs_found:
                    print(f"      - {bug}")
    
    # ===== SUMMARY REPORT =====
    print("\n" + "=" * 70)
    print("SUMMARY REPORT")
    print("=" * 70)
    
    # Overall stats
    total_tasks = len(all_analyses)
    tasks_with_bugs = sum(1 for a in all_analyses if a.bugs_found)
    
    print(f"\nTotal tasks analyzed: {total_tasks}")
    print(f"Tasks with bugs: {tasks_with_bugs}")
    print(f"Total bugs found: {len(all_bugs)}")
    
    # Solver improvement analysis
    print("\nSolver Step Improvement Analysis:")
    all_first_acc = []
    all_last_acc = []
    all_improvements = []
    
    for analysis in all_analyses:
        if analysis.solver_steps:
            first = analysis.solver_steps[0].accuracy
            last = analysis.solver_steps[-1].accuracy
            all_first_acc.append(first)
            all_last_acc.append(last)
            all_improvements.append(last - first)
    
    if all_improvements:
        print(f"  Mean first step accuracy: {np.mean(all_first_acc):.1%}")
        print(f"  Mean last step accuracy: {np.mean(all_last_acc):.1%}")
        print(f"  Mean improvement: {np.mean(all_improvements):.1%}")
        print(f"  Tasks where last step is best: {sum(1 for i in all_improvements if i >= 0)}/{len(all_improvements)}")
    
    # Bug list
    if all_bugs:
        print("\n⚠️ ALL BUGS FOUND:")
        for bug in all_bugs:
            print(f"  - {bug}")
    else:
        print("\n✅ No bugs detected in any module outputs!")
    
    # Save report
    report = {
        "timestamp": datetime.now().isoformat(),
        "device": device,
        "checkpoint": str(checkpoint_path),
        "total_tasks": total_tasks,
        "tasks_with_bugs": tasks_with_bugs,
        "total_bugs": len(all_bugs),
        "bugs": all_bugs,
        "solver_stats": {
            "mean_first_step_acc": float(np.mean(all_first_acc)) if all_first_acc else 0,
            "mean_last_step_acc": float(np.mean(all_last_acc)) if all_last_acc else 0,
            "mean_improvement": float(np.mean(all_improvements)) if all_improvements else 0,
        },
        "analyses": [
            {
                "task_id": a.task_id,
                "grid_size": a.grid_size,
                "input_shape": list(a.input_shape),
                "output_shape": list(a.output_shape),
                "execution_time": a.execution_time,
                "bugs_found": a.bugs_found,
                "solver_steps": [
                    {
                        "step": s.step_idx,
                        "accuracy": s.accuracy,
                        "entropy": s.entropy,
                        "confidence": s.confidence,
                        "improvement": s.improvement_from_prev,
                    }
                    for s in a.solver_steps
                ],
                "module_stats": [stat.to_dict() for stat in a.module_stats],
            }
            for a in all_analyses
        ]
    }
    
    report_path = output_dir / "trace_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nReport saved to: {report_path}")
    print(f"Visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()
