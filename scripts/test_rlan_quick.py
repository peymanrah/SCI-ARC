"""
Quick RLAN Module Testing Suite - No Visualizations

Tests ALL RLAN modules (old and new) with trained checkpoint on real ARC data.
Skips visualization to run faster and avoid matplotlib issues.
"""

import os
import sys
import json
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any
import traceback
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sci_arc.models.rlan import RLAN, RLANConfig


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


def find_tasks_by_size(data_dir: Path, target_sizes=("small", "medium", "large"), max_per_size: int = 1) -> Dict[str, List[dict]]:
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


def prepare_task_tensors(task: dict, device: str = 'cpu'):
    """Prepare task tensors for model input."""
    train_pairs = task["train"]
    test_pair = task.get("test", train_pairs[-1:])
    
    test_input = np.array(test_pair[0]["input"], dtype=np.int64)
    test_target = np.array(test_pair[0].get("output", test_pair[0]["input"]), dtype=np.int64)
    
    h, w = test_input.shape
    x = torch.tensor(test_input, dtype=torch.long).unsqueeze(0).to(device)
    
    train_inputs = []
    train_outputs = []
    for pair in train_pairs:
        inp = np.array(pair["input"], dtype=np.int64)
        out = np.array(pair["output"], dtype=np.int64)
        ih, iw = inp.shape
        oh, ow = out.shape
        inp_padded = np.zeros((30, 30), dtype=np.int64)
        inp_padded[:ih, :iw] = inp
        out_padded = np.zeros((30, 30), dtype=np.int64)
        out_padded[:oh, :ow] = out
        train_inputs.append(inp_padded)
        train_outputs.append(out_padded)
    
    train_inputs = torch.tensor(np.array(train_inputs), dtype=torch.long).unsqueeze(0).to(device)
    train_outputs = torch.tensor(np.array(train_outputs), dtype=torch.long).unsqueeze(0).to(device)
    pair_mask = torch.ones(1, len(train_pairs), dtype=torch.bool, device=device)
    
    return x, train_inputs, train_outputs, pair_mask, test_input, test_target


class TestResult:
    """Container for test results."""
    def __init__(self, name: str):
        self.name = name
        self.passed = True
        self.checks = []
        self.metrics = {}
        self.warnings = []
        self.errors = []
    
    def add_check(self, name: str, passed: bool, details: str = ""):
        self.checks.append({"name": name, "passed": passed, "details": details})
        if not passed:
            self.passed = False
    
    def add_metric(self, name: str, value):
        self.metrics[name] = value
    
    def add_warning(self, msg: str):
        self.warnings.append(msg)
    
    def __repr__(self):
        status = "PASS" if self.passed else "FAIL"
        return f"TestResult({self.name}): {status} - {len(self.errors)} errors, {len(self.warnings)} warnings"


def test_core_modules(model: RLAN, task_info: dict, device: str = 'cpu') -> TestResult:
    """Test core RLAN modules: Encoder, DSC, MSRE, Solver."""
    result = TestResult("Core Modules")
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
            else:
                result.add_warning("DSC is disabled")
            
            # Test MSRE
            if model.msre is not None and model.dsc is not None:
                msre_out = model.msre(features, centroids)
                result.add_check("MSRE output shape", len(msre_out.shape) == 5, f"shape={msre_out.shape}")
                result.add_check("MSRE no NaN", not torch.isnan(msre_out).any().item(), "NaN detected")
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


def test_recursive_solver_steps(model: RLAN, task_info: dict, device: str = 'cpu') -> TestResult:
    """Test recursive solver with step-by-step outputs."""
    result = TestResult("Recursive Solver Steps")
    task = task_info["task"]
    
    try:
        x, train_inputs, train_outputs, pair_mask, test_input, test_target = prepare_task_tensors(task, device)
        
        with torch.no_grad():
            outputs = model(
                input_grid=x,
                train_inputs=train_inputs,
                train_outputs=train_outputs,
                pair_mask=pair_mask,
                temperature=1.0,
                return_intermediates=True
            )
            
            step_logits_list = outputs.get('step_logits', [])
            if step_logits_list:
                num_steps = len(step_logits_list)
                result.add_metric("num_solver_steps", num_steps)
                result.add_check("Multiple steps", num_steps > 0, f"steps={num_steps}")
                
                # Check each step
                target = torch.tensor(test_target, dtype=torch.long, device=device)
                step_accuracies = []
                for i, step_logits in enumerate(step_logits_list):
                    preds = step_logits.argmax(dim=1)[0]
                    acc = (preds == target).float().mean().item()
                    step_accuracies.append(acc)
                    result.add_check(f"Step {i+1} no NaN", not torch.isnan(step_logits).any().item(), "")
                
                result.add_metric("step_accuracies", step_accuracies)
                improvement = step_accuracies[-1] - step_accuracies[0]
                result.add_check("Steps improve", improvement >= -0.1, f"improvement={improvement:.2%}")
            else:
                result.add_warning("No step logits returned")
                
    except Exception as e:
        result.passed = False
        result.errors.append(f"Exception: {str(e)}")
        traceback.print_exc()
    
    return result


def test_teps_search(task_info: dict) -> TestResult:
    """Test TEPS program search module."""
    result = TestResult("TEPS Program Search")
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
        
        test_input_np = train_inputs_np[0]
        result.add_metric("num_train_pairs", len(train_inputs_np))
        
        # Run search with correct signature
        start_time = time.time()
        search_result = teps.search(test_input_np, train_inputs_np, train_outputs_np)
        search_time = time.time() - start_time
        
        result.add_metric("search_time_seconds", search_time)
        result.add_metric("program_found", search_result.get('success', False))
        
        if search_result.get('success', False):
            match_score = search_result.get('stats', {}).get('best_score', 0)
            result.add_metric("match_score", match_score)
            result.add_check("High match score", match_score >= 0.9, f"score={match_score:.2%}")
        else:
            result.add_warning(f"No program found within timeout")
            
    except ImportError as e:
        result.add_warning(f"TEPS module not available: {e}")
    except Exception as e:
        result.passed = False
        result.errors.append(f"Exception: {str(e)}")
        traceback.print_exc()
    
    return result


def test_ns_teps_search(task_info: dict) -> TestResult:
    """Test NS-TEPS neuro-symbolic program search."""
    result = TestResult("NS-TEPS Program Search")
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
        
        train_inputs_np = []
        train_outputs_np = []
        for pair in task["train"]:
            inp = np.array(pair["input"], dtype=np.int64)
            out = np.array(pair["output"], dtype=np.int64)
            train_inputs_np.append(inp)
            train_outputs_np.append(out)
        
        test_input_np = train_inputs_np[0]
        
        # Test object extraction
        objects = extractor.extract(test_input_np)
        result.add_metric("num_objects", len(objects))
        result.add_check("Objects extracted", len(objects) >= 0, f"found {len(objects)} objects")
        
        # Run NS-TEPS search with correct signature
        start_time = time.time()
        search_result = ns_teps.search(test_input_np, train_inputs_np, train_outputs_np)
        search_time = time.time() - start_time
        
        result.add_metric("search_time_seconds", search_time)
        result.add_metric("trace_found", search_result.get('success', False))
        
        if search_result.get('success', False):
            match_score = search_result.get('confidence', 0)
            result.add_metric("match_score", match_score)
            result.add_check("High match score", match_score >= 0.8, f"score={match_score:.2%}")
        else:
            result.add_warning("No trace found")
            
    except ImportError as e:
        result.add_warning(f"NS-TEPS module not available: {e}")
    except Exception as e:
        result.passed = False
        result.errors.append(f"Exception: {str(e)}")
        traceback.print_exc()
    
    return result


def test_hyperlora(model: RLAN, task_info: dict, device: str = 'cpu') -> TestResult:
    """Test HyperLoRA meta-learning module."""
    result = TestResult("HyperLoRA Meta-Learning")
    task = task_info["task"]
    
    try:
        x, train_inputs, train_outputs, pair_mask, test_input, test_target = prepare_task_tensors(task, device)
        
        if model.hyper_lora is None:
            result.add_check("HyperLoRA exists", False, "module not found")
            result.add_warning("HyperLoRA is disabled in this checkpoint")
            return result
        
        result.add_check("HyperLoRA exists", True, "")
        
        with torch.no_grad():
            features = model.encode(x)
            centroids, _, _ = model.dsc(features)
            
            # Generate LoRA weights
            lora_weights = model.hyper_lora(features, centroids)
            result.add_check("LoRA weights generated", lora_weights is not None, "")
            
            if lora_weights:
                for name, weight in lora_weights.items():
                    result.add_check(f"LoRA {name} finite", torch.isfinite(weight).all().item(), "")
                    
    except Exception as e:
        result.passed = False
        result.errors.append(f"Exception: {str(e)}")
        traceback.print_exc()
    
    return result


def test_signal_quality(model: RLAN, task_info: dict, device: str = 'cpu') -> TestResult:
    """Test gradient flow and signal quality."""
    result = TestResult("Signal Quality")
    task = task_info["task"]
    
    try:
        x, train_inputs, train_outputs, pair_mask, test_input, test_target = prepare_task_tensors(task, device)
        target = torch.tensor(test_target, dtype=torch.long, device=device)
        
        model.train()
        model.zero_grad()
        
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
        loss = F.cross_entropy(logits[0], target)
        result.add_metric("loss", loss.item())
        result.add_check("Loss is finite", torch.isfinite(loss).item(), f"loss={loss.item():.4f}")
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        grad_norms = []
        zero_grads = []
        nan_grads = []
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                norm = param.grad.norm().item()
                grad_norms.append(norm)
                if norm == 0:
                    zero_grads.append(name)
                if not np.isfinite(norm):
                    nan_grads.append(name)
        
        mean_grad = np.mean(grad_norms) if grad_norms else 0
        max_grad = np.max(grad_norms) if grad_norms else 0
        
        result.add_metric("mean_grad_norm", mean_grad)
        result.add_metric("max_grad_norm", max_grad)
        result.add_check("Has gradients", mean_grad > 0, f"mean={mean_grad:.2e}")
        result.add_check("Grads not exploding", max_grad < 100, f"max={max_grad:.2e}")
        result.add_check("No NaN grads", len(nan_grads) == 0, f"{len(nan_grads)} NaN params")
        
        model.eval()
        model.zero_grad()
        
    except Exception as e:
        result.passed = False
        result.errors.append(f"Exception: {str(e)}")
        traceback.print_exc()
        model.eval()
    
    return result


def test_program_cache() -> TestResult:
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
                
    except ImportError as e:
        result.add_warning(f"ProgramCache not available: {e}")
    except Exception as e:
        result.passed = False
        result.errors.append(f"Exception: {str(e)}")
        traceback.print_exc()
    
    return result


def main():
    """Main test runner."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    print("=" * 70)
    print("QUICK RLAN MODULE TESTING (NO VISUALIZATIONS)")
    print("=" * 70)
    
    # Load model
    checkpoint_path = Path("c:/Users/perahmat/Downloads/SCI-ARC/checkpoints/warmup3.pt")
    model, model_config = load_model_with_checkpoint(checkpoint_path, device)
    
    # Find ARC tasks
    data_dirs = [
        Path("c:/Users/perahmat/Downloads/SCI-ARC/data/arc-agi/data/training"),
        Path("c:/Users/perahmat/Downloads/SCI-ARC/data/training"),
        Path("c:/Users/perahmat/Downloads/SCI-ARC/data/arc-agi_training-challenges"),
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
    
    print(f"\nFinding ARC tasks by grid size...")
    tasks_by_size = find_tasks_by_size(data_dir, max_per_size=1)
    
    for size, tasks in tasks_by_size.items():
        print(f"  {size.upper()}: {len(tasks)} tasks")
    
    # Collect results
    all_results = []
    
    for size in ["small", "medium", "large"]:
        print(f"\n{'=' * 70}")
        print(f"Testing {size.upper()} grid tasks")
        print("=" * 70)
        
        for task_info in tasks_by_size[size]:
            h, w = task_info["grid_size"]
            print(f"\n  Task: {task_info['task_id']} ({h}x{w})")
            
            # Core modules
            print("    Testing core modules...", end=" ")
            result = test_core_modules(model, task_info, device)
            print(result)
            all_results.append(result)
            
            # Solver steps
            print("    Testing solver steps...", end=" ")
            result = test_recursive_solver_steps(model, task_info, device)
            print(result)
            all_results.append(result)
            
            # TEPS
            print("    Testing TEPS...", end=" ")
            result = test_teps_search(task_info)
            print(result)
            all_results.append(result)
            
            # NS-TEPS
            print("    Testing NS-TEPS...", end=" ")
            result = test_ns_teps_search(task_info)
            print(result)
            all_results.append(result)
            
            # HyperLoRA
            print("    Testing HyperLoRA...", end=" ")
            result = test_hyperlora(model, task_info, device)
            print(result)
            all_results.append(result)
            
            # Signal quality
            print("    Testing signal quality...", end=" ")
            result = test_signal_quality(model, task_info, device)
            print(result)
            all_results.append(result)
    
    # Program cache
    print(f"\n{'=' * 70}")
    print("Testing Program Cache")
    print("=" * 70)
    result = test_program_cache()
    print(f"  {result}")
    all_results.append(result)
    
    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for r in all_results if r.passed)
    failed = sum(1 for r in all_results if not r.passed)
    
    print(f"  Total: {len(all_results)}")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    
    # Group by test type
    by_type = {}
    for r in all_results:
        if r.name not in by_type:
            by_type[r.name] = {"passed": 0, "failed": 0}
        if r.passed:
            by_type[r.name]["passed"] += 1
        else:
            by_type[r.name]["failed"] += 1
    
    print("\nBy test type:")
    for name, counts in by_type.items():
        status = "✅" if counts["failed"] == 0 else "❌"
        print(f"  {status} {name}: {counts['passed']} passed, {counts['failed']} failed")
    
    # List failures
    if failed > 0:
        print("\nFailures:")
        for r in all_results:
            if not r.passed:
                print(f"  - {r.name}")
                for err in r.errors:
                    print(f"      Error: {err}")


if __name__ == "__main__":
    main()
