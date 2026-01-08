"""
Production-Accurate RLAN Module Tracing and Bug Detection - NO VISUALIZATIONS

This script traces the EXACT order of operations in RLAN production code
and outputs statistics to console and JSON file.
"""

import os
import sys
import json
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import traceback
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sci_arc.models.rlan import RLAN, RLANConfig


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
    """Prepare task tensors EXACTLY as production code does."""
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
    """Trace RLAN forward pass step-by-step, matching production code exactly."""
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
        
        # Get valid mask and grid sizes
        valid_mask = model.encoder.get_valid_mask(x)
        grid_sizes = model.encoder.get_grid_sizes(x)
        
        # ========== STEP 2: CONTEXT ENCODER ==========
        context = None
        dsc_task_context = None
        support_features = None
        
        if model.use_context_encoder and model.context_encoder is not None:
            context_output = model.context_encoder(
                train_inputs, train_outputs, pair_mask
            )
            
            if model.context_encoder.use_spatial_features:
                support_features = context_output
                stats_list.append(analyze_tensor(support_features, "2a. ContextEncoder.support_features"))
                
                dsc_task_context = model.pool_context_from_support(context_output)
                stats_list.append(analyze_tensor(dsc_task_context, "2b. ContextEncoder.dsc_task_context"))
                
                cross_attention_active = getattr(model, 'cross_attention_active', True)
                if hasattr(model.context_injector, 'forward') and cross_attention_active:
                    features = model.context_injector(features, context_output)
                    stats_list.append(analyze_tensor(features, "2c. Features after context injection"))
            else:
                context = context_output
                dsc_task_context = context
                stats_list.append(analyze_tensor(context, "2. ContextEncoder.context"))
                features = model.context_injector(features, context)
                stats_list.append(analyze_tensor(features, "2b. Features after FiLM injection"))
        
        # ========== STEP 3: HPM (if enabled) ==========
        hpm_memory_tokens = None
        # Skip - warmup3.pt doesn't have HPM
        
        # ========== STEP 4: DSC ==========
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
        
        # ========== STEP 5: MSRE ==========
        if model.use_msre and model.msre is not None:
            clue_features = model.msre(
                features, centroids, grid_sizes=grid_sizes
            )
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
            stats_list.append(analyze_tensor(count_embedding, "6. LCR.count_embedding"))
        else:
            count_embedding = torch.zeros(
                B, model.num_colors, model.hidden_dim, device=device
            )
        outputs["count_embedding"] = count_embedding
        
        # ========== STEP 7: SPH (if enabled) ==========
        if model.use_sph and model.sph is not None:
            predicates = model.sph(features, temperature=1.0)
            stats_list.append(analyze_tensor(predicates, "7. SPH.predicates"))
        else:
            predicates = torch.zeros(B, model.num_predicates, device=device)
        outputs["predicates"] = predicates
        
        # ========== STEP 8: HYPERLORA (if enabled) ==========
        lora_deltas = None
        # Skip - warmup3.pt doesn't have HyperLoRA
        
        # ========== STEP 9: RECURSIVE SOLVER ==========
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
        probs = F.softmax(logits[0], dim=0)  # (C, H, W)
        preds = logits[0].argmax(dim=0)  # (H, W)
        
        # Crop/align if needed
        th, tw = target.shape
        ph, pw = preds.shape
        h, w = min(th, ph), min(pw, tw)
        preds_crop = preds[:h, :w]
        target_crop = target_tensor[:h, :w]
        
        # Accuracy
        correct = (preds_crop == target_crop).float()
        accuracy = correct.mean().item()
        
        # Entropy
        probs_crop = probs[:, :h, :w]
        entropy = -(probs_crop * torch.log(probs_crop + 1e-10)).sum(dim=0).mean().item()
        
        # Confidence
        confidence = probs_crop.max(dim=0)[0].mean().item()
        
        # BG/FG ratios
        total_pixels = h * w
        bg_pixels = (preds_crop == 0).sum().item()
        bg_ratio = bg_pixels / total_pixels
        fg_ratio = 1 - bg_ratio
        
        # Unique colors
        unique_colors = len(preds_crop.unique())
        
        # Improvement
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


def find_arc_tasks(data_dir: Path, sizes=("small", "medium", "large"), max_per_size: int = 2) -> Dict[str, List[dict]]:
    """Find ARC tasks by grid size category."""
    known_tasks = {
        "small": ["007bbfb7", "00d62c1b"],
        "medium": ["025d127b", "0520fde7"],
        "large": ["045e512c", "0962bcdd"],
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
    print("PRODUCTION-ACCURATE RLAN MODULE TRACING (NO VIZ)")
    print("=" * 70)
    
    # Paths
    checkpoint_path = Path("c:/Users/perahmat/Downloads/SCI-ARC/checkpoints/warmup3.pt")
    output_dir = Path("c:/Users/perahmat/Downloads/SCI-ARC/scripts/outputs/module_trace")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model, model_config = load_model_with_checkpoint(checkpoint_path, device)
    
    # Find ARC tasks
    data_dir = Path("c:/Users/perahmat/Downloads/SCI-ARC/data/arc-agi/data/training")
    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
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
            
            # Print module stats
            print(f"    Module Statistics:")
            for stat in stats_list:
                bug_marker = " [BUG]" if stat.has_bug else ""
                print(f"      {stat.name}: shape={stat.shape}, mean={stat.mean_val:.4f}, std={stat.std_val:.4f}{bug_marker}")
            
            # Print solver stats
            print(f"    Solver Steps ({len(solver_stats)} steps):")
            for ss in solver_stats:
                delta = f"+{ss.improvement_from_prev:.1%}" if ss.improvement_from_prev > 0 else f"{ss.improvement_from_prev:.1%}"
                print(f"      Step {ss.step_idx}: Acc={ss.accuracy:.1%} ({delta}), Conf={ss.confidence:.2f}, Entropy={ss.entropy:.2f}, BG={ss.bg_ratio:.1%}")
            
            if analysis.bugs_found:
                print(f"    BUGS FOUND: {len(analysis.bugs_found)}")
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
        print(f"  Tasks where last step >= first: {sum(1 for i in all_improvements if i >= 0)}/{len(all_improvements)}")
    
    # Bug list
    if all_bugs:
        print("\nALL BUGS FOUND:")
        for bug in all_bugs:
            print(f"  - {bug}")
    else:
        print("\nNo bugs detected in any module outputs!")
    
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
                        "bg_ratio": s.bg_ratio,
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


if __name__ == "__main__":
    main()
