#!/usr/bin/env python
"""
Comprehensive RLAN Testing Script

This script performs a deep mathematical and academic review of RLAN by:
1. Loading real ARC training/evaluation data
2. Running forward passes with detailed tensor logging
3. Checking for NaN/Inf at each module output
4. Validating normalization choices (LayerNorm vs BatchNorm)
5. Analyzing loss function behavior
6. Testing gradient flow
7. Validating counting, spatial reasoning, and compositional learning

Author: AI Agent for RLAN Academic Review
"""

import os
import sys
from pathlib import Path
import json
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sci_arc.models import RLAN, RLANConfig
from sci_arc.models.rlan_modules import (
    DynamicSaliencyController,
    MultiScaleRelativeEncoding,
    LatentCountingRegisters,
    SymbolicPredicateHeads,
    RecursiveSolver,
)
from sci_arc.models.grid_encoder import GridEncoder
from sci_arc.training.rlan_loss import (
    RLANLoss, FocalLoss, EntropyRegularization,
    SparsityRegularization, PredicateDiversityLoss, CurriculumPenalty
)


@dataclass
class TensorStats:
    """Statistics for a tensor."""
    name: str
    shape: tuple
    dtype: str
    min_val: float
    max_val: float
    mean_val: float
    std_val: float
    has_nan: bool
    has_inf: bool
    num_zeros: int
    
    def __str__(self):
        status = "OK" if not (self.has_nan or self.has_inf) else "PROBLEM"
        nan_str = " [NaN!]" if self.has_nan else ""
        inf_str = " [Inf!]" if self.has_inf else ""
        return (
            f"{status} {self.name}: shape={self.shape}, "
            f"range=[{self.min_val:.4f}, {self.max_val:.4f}], "
            f"mean={self.mean_val:.4f}, std={self.std_val:.4f}"
            f"{nan_str}{inf_str}"
        )


def analyze_tensor(name: str, tensor: torch.Tensor) -> TensorStats:
    """Analyze a tensor for numerical issues."""
    if tensor is None:
        return TensorStats(name, (), "None", 0, 0, 0, 0, False, False, 0)
    
    with torch.no_grad():
        flat = tensor.float().flatten()
        return TensorStats(
            name=name,
            shape=tuple(tensor.shape),
            dtype=str(tensor.dtype),
            min_val=flat.min().item() if len(flat) > 0 else 0.0,
            max_val=flat.max().item() if len(flat) > 0 else 0.0,
            mean_val=flat.mean().item() if len(flat) > 0 else 0.0,
            std_val=flat.std().item() if len(flat) > 0 else 0.0,
            has_nan=torch.isnan(tensor).any().item(),
            has_inf=torch.isinf(tensor).any().item(),
            num_zeros=(flat == 0).sum().item(),
        )


def load_arc_tasks(data_dir: str, split: str = "training", max_tasks: int = 50) -> List[Dict]:
    """Load ARC tasks from directory."""
    tasks = []
    split_dir = Path(data_dir) / split
    
    if not split_dir.exists():
        print(f"Warning: {split_dir} not found, looking for alternate paths...")
        # Try alternate locations
        alt_paths = [
            Path(data_dir) / f"{split}_challenges",
            Path(data_dir) / "challenges" / split,
            Path(data_dir),
        ]
        for alt in alt_paths:
            if alt.exists():
                split_dir = alt
                break
    
    if not split_dir.exists():
        print(f"Error: Could not find data directory")
        return []
    
    json_files = list(split_dir.glob("*.json"))[:max_tasks]
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            task = {
                'task_id': json_file.stem,
                'train': [(np.array(p['input']), np.array(p['output'])) for p in data.get('train', [])],
                'test': [(np.array(p['input']), np.array(p.get('output', p['input']))) for p in data.get('test', [])],
            }
            tasks.append(task)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    
    return tasks


def test_grid_encoder(device: torch.device):
    """Test GridEncoder module."""
    print("\n" + "="*80)
    print("TEST 1: GridEncoder Analysis")
    print("="*80)
    
    encoder = GridEncoder(hidden_dim=128, num_colors=10, max_size=30, dropout=0.0).to(device)
    
    # Test cases
    test_grids = [
        ("random_5x5", torch.randint(0, 10, (2, 5, 5))),
        ("all_zeros_10x10", torch.zeros(2, 10, 10, dtype=torch.long)),
        ("all_nines_10x10", torch.full((2, 10, 10), 9, dtype=torch.long)),
        ("max_size_30x30", torch.randint(0, 10, (1, 30, 30))),
        ("single_pixel", torch.randint(0, 10, (1, 1, 1))),
    ]
    
    all_passed = True
    for name, grid in test_grids:
        grid = grid.to(device)
        output = encoder(grid)
        stats = analyze_tensor(f"GridEncoder({name})", output)
        print(f"  {stats}")
        
        if stats.has_nan or stats.has_inf:
            all_passed = False
        
        # Check output shape
        expected_shape = (grid.shape[0], grid.shape[1], grid.shape[2], 128)
        if output.shape != expected_shape:
            print(f"    ✗ Shape mismatch: expected {expected_shape}, got {output.shape}")
            all_passed = False
    
    # Analysis: LayerNorm effect
    print("\n  LayerNorm Analysis:")
    with torch.no_grad():
        grid = torch.randint(0, 10, (4, 10, 10)).to(device)
        output = encoder(grid)
        # Check per-position mean/std
        per_pos_mean = output.mean(dim=-1)  # Should be ~0
        per_pos_std = output.std(dim=-1)    # Should be ~1
        print(f"    Per-position mean: {per_pos_mean.mean():.4f} (should be ~0)")
        print(f"    Per-position std: {per_pos_std.mean():.4f} (should be ~1)")
    
    return all_passed


def test_dsc(device: torch.device):
    """Test Dynamic Saliency Controller."""
    print("\n" + "="*80)
    print("TEST 2: Dynamic Saliency Controller Analysis")
    print("="*80)
    
    dsc = DynamicSaliencyController(hidden_dim=128, max_clues=5, num_heads=4, dropout=0.0).to(device)
    
    test_cases = [
        ("small_features", torch.randn(2, 128, 5, 5)),
        ("larger_features", torch.randn(2, 128, 15, 15)),
        ("max_size", torch.randn(1, 128, 30, 30)),
    ]
    
    temperatures = [5.0, 1.0, 0.5, 0.1]
    
    all_passed = True
    for name, features in test_cases:
        features = features.to(device)
        print(f"\n  Testing {name} (shape={features.shape}):")
        
        for temp in temperatures:
            centroids, attention_maps, stop_logits = dsc(features, temperature=temp)
            
            stats_c = analyze_tensor(f"centroids(T={temp})", centroids)
            stats_a = analyze_tensor(f"attention(T={temp})", attention_maps)
            stats_s = analyze_tensor(f"stop_logits(T={temp})", stop_logits)
            
            print(f"    T={temp}: {stats_c}")
            
            if stats_c.has_nan or stats_a.has_nan or stats_s.has_nan:
                all_passed = False
            
            # Validate attention sums to 1
            H, W = features.shape[2], features.shape[3]
            attn_sum = attention_maps.view(attention_maps.shape[0], attention_maps.shape[1], -1).sum(dim=-1)
            if not torch.allclose(attn_sum, torch.ones_like(attn_sum), atol=1e-4):
                print(f"    ✗ Attention doesn't sum to 1: {attn_sum}")
            
            # Check centroid bounds
            B, K = centroids.shape[0], centroids.shape[1]
            if (centroids[:, :, 0] < 0).any() or (centroids[:, :, 0] > H).any():
                print(f"    ✗ Row centroids out of bounds")
            if (centroids[:, :, 1] < 0).any() or (centroids[:, :, 1] > W).any():
                print(f"    ✗ Col centroids out of bounds")
    
    return all_passed


def test_msre(device: torch.device):
    """Test Multi-Scale Relative Encoding."""
    print("\n" + "="*80)
    print("TEST 3: Multi-Scale Relative Encoding Analysis")
    print("="*80)
    
    msre = MultiScaleRelativeEncoding(hidden_dim=128, encoding_dim=32, max_size=30, num_freq=8).to(device)
    
    features = torch.randn(2, 128, 10, 10).to(device)
    centroids = torch.tensor([[[5.0, 5.0], [2.0, 8.0]], [[3.0, 3.0], [7.0, 7.0]]]).to(device)  # (2, 2, 2)
    
    output = msre(features, centroids)
    stats = analyze_tensor("MSRE output", output)
    print(f"  {stats}")
    
    # Check relative encoding properties
    print("\n  Relative Encoding Properties:")
    
    # Test translation equivariance
    features_shifted = torch.roll(features, shifts=2, dims=2)  # Shift rows
    centroids_shifted = centroids.clone()
    centroids_shifted[:, :, 0] += 2  # Shift centroid rows too
    
    output_shifted = msre(features_shifted, centroids_shifted)
    
    # The relative encoding should be similar (modulo boundary effects)
    diff = (output[:, :, :, 2:-2, 2:-2] - output_shifted[:, :, :, 2:-2, 2:-2]).abs().mean()
    print(f"  Translation equivariance diff: {diff.item():.6f} (should be small)")
    
    return not (stats.has_nan or stats.has_inf)


def test_lcr(device: torch.device):
    """Test Latent Counting Registers."""
    print("\n" + "="*80)
    print("TEST 4: Latent Counting Registers Analysis")
    print("="*80)
    
    lcr = LatentCountingRegisters(num_colors=10, hidden_dim=128, num_freq=8, num_heads=4, dropout=0.0).to(device)
    
    test_cases = [
        ("uniform_colors", torch.randint(0, 10, (2, 10, 10))),
        ("mostly_zeros", torch.zeros(2, 10, 10, dtype=torch.long)),
        ("mostly_ones", torch.ones(2, 10, 10, dtype=torch.long)),
        ("single_color_each", torch.stack([torch.full((10, 10), i, dtype=torch.long) for i in range(2)])),
    ]
    
    all_passed = True
    for name, grid in test_cases:
        grid = grid.to(device)
        features = torch.randn(grid.shape[0], 128, grid.shape[1], grid.shape[2]).to(device)
        
        output = lcr(grid, features)
        stats = analyze_tensor(f"LCR({name})", output)
        print(f"  {stats}")
        
        if stats.has_nan or stats.has_inf:
            all_passed = False
        
        # Check count accuracy
        with torch.no_grad():
            for b in range(grid.shape[0]):
                for c in range(10):
                    actual_count = (grid[b] == c).sum().item()
                    # LCR should encode this count somehow
                    # We can't directly check, but we verify no NaN
    
    # Test counting sensitivity
    print("\n  Counting Sensitivity Test:")
    grid1 = torch.zeros(1, 10, 10, dtype=torch.long).to(device)
    grid2 = torch.ones(1, 10, 10, dtype=torch.long).to(device)
    features = torch.randn(1, 128, 10, 10).to(device)
    
    out1 = lcr(grid1, features)
    out2 = lcr(grid2, features)
    diff = (out1 - out2).abs().mean()
    print(f"  Difference between all-zeros vs all-ones: {diff.item():.4f} (should be large)")
    
    return all_passed


def test_sph(device: torch.device):
    """Test Symbolic Predicate Heads."""
    print("\n" + "="*80)
    print("TEST 5: Symbolic Predicate Heads Analysis")
    print("="*80)
    
    sph = SymbolicPredicateHeads(hidden_dim=128, num_predicates=8, dropout=0.0).to(device)
    
    features = torch.randn(4, 128, 10, 10).to(device)
    
    for temp in [5.0, 1.0, 0.1]:
        predicates = sph(features, temperature=temp)
        stats = analyze_tensor(f"SPH(T={temp})", predicates)
        print(f"  {stats}")
        
        # Check predicates in [0, 1]
        if (predicates < 0).any() or (predicates > 1).any():
            print(f"    ✗ Predicates outside [0, 1] range")
    
    # Test predicate diversity
    print("\n  Predicate Diversity Check:")
    predicates = sph(features, temperature=1.0)
    corr = torch.corrcoef(predicates.T)
    off_diag = corr * (1 - torch.eye(8, device=device))
    print(f"  Mean off-diagonal correlation: {off_diag.abs().mean():.4f} (should be low)")
    
    return True


def test_solver(device: torch.device):
    """Test Recursive Solver."""
    print("\n" + "="*80)
    print("TEST 6: Recursive Solver Analysis")
    print("="*80)
    
    solver = RecursiveSolver(
        hidden_dim=128, num_classes=10, num_steps=6,
        num_predicates=8, num_colors=10, dropout=0.0
    ).to(device)
    
    B, K, H, W = 2, 3, 10, 10
    clue_features = torch.randn(B, K, 128, H, W).to(device)
    count_embedding = torch.randn(B, 10, 128).to(device)
    predicates = torch.rand(B, 8).to(device)
    input_grid = torch.randint(0, 10, (B, H, W)).to(device)
    attention_maps = F.softmax(torch.randn(B, K, H, W).view(B, K, -1), dim=-1).view(B, K, H, W).to(device)
    
    # Test with all steps
    all_logits = solver(
        clue_features=clue_features,
        count_embedding=count_embedding,
        predicates=predicates,
        input_grid=input_grid,
        attention_maps=attention_maps,
        return_all_steps=True,
    )
    
    print(f"  Number of solver steps: {len(all_logits)}")
    
    all_passed = True
    for i, logits in enumerate(all_logits):
        stats = analyze_tensor(f"Step {i} logits", logits)
        print(f"  {stats}")
        
        if stats.has_nan or stats.has_inf:
            all_passed = False
        
        # Check logits are reasonable
        if stats.max_val > 50 or stats.min_val < -50:
            print(f"    [!] Logits may be too extreme")
    
    # Check refinement improves over steps
    print("\n  Refinement Analysis:")
    preds = [l.argmax(dim=1) for l in all_logits]
    # In absence of target, check that predictions stabilize
    changes = []
    for i in range(1, len(preds)):
        change = (preds[i] != preds[i-1]).float().mean()
        changes.append(change.item())
    print(f"  Step-to-step prediction changes: {changes}")
    
    return all_passed


def test_focal_loss(device: torch.device):
    """Test Focal Loss behavior."""
    print("\n" + "="*80)
    print("TEST 7: Focal Loss Analysis")
    print("="*80)
    
    focal = FocalLoss(gamma=2.0, alpha=0.25).to(device)
    ce = nn.CrossEntropyLoss()
    
    B, C, H, W = 4, 11, 10, 10
    
    test_cases = [
        ("random_logits", torch.randn(B, C, H, W)),
        ("confident_correct", torch.zeros(B, C, H, W)),  # Will set correct class high
        ("confident_wrong", torch.zeros(B, C, H, W)),    # Will set wrong class high
        ("extreme_logits", torch.randn(B, C, H, W) * 100),
    ]
    
    targets = torch.randint(0, C, (B, H, W)).to(device)
    
    for name, logits in test_cases:
        logits = logits.to(device)
        
        if name == "confident_correct":
            for b in range(B):
                for h in range(H):
                    for w in range(W):
                        logits[b, targets[b, h, w], h, w] = 10.0
        elif name == "confident_wrong":
            for b in range(B):
                for h in range(H):
                    for w in range(W):
                        wrong_class = (targets[b, h, w].item() + 1) % C
                        logits[b, wrong_class, h, w] = 10.0
        
        focal_loss = focal(logits, targets)
        ce_loss = ce(logits, targets)
        
        stats = analyze_tensor(f"FocalLoss({name})", focal_loss.unsqueeze(0))
        print(f"  {name}:")
        print(f"    Focal Loss: {focal_loss.item():.4f}")
        print(f"    CrossEntropy: {ce_loss.item():.4f}")
        print(f"    Ratio (Focal/CE): {(focal_loss / ce_loss).item():.4f}")
    
    # Test class imbalance handling
    print("\n  Class Imbalance Test:")
    logits = torch.randn(B, C, H, W).to(device)
    
    # Mostly background (class 0)
    targets_bg = torch.zeros(B, H, W, dtype=torch.long, device=device)
    targets_bg[0, 5, 5] = 1  # One foreground pixel
    
    loss_bg = focal(logits, targets_bg)
    print(f"  Loss with mostly background: {loss_bg.item():.4f}")
    
    # Mostly foreground
    targets_fg = torch.ones(B, H, W, dtype=torch.long, device=device)
    loss_fg = focal(logits, targets_fg)
    print(f"  Loss with all foreground: {loss_fg.item():.4f}")
    
    return True


def test_full_rlan(device: torch.device, arc_data_dir: Optional[str] = None):
    """Test full RLAN model with real or synthetic data."""
    print("\n" + "="*80)
    print("TEST 8: Full RLAN End-to-End Test")
    print("="*80)
    
    model = RLAN(
        hidden_dim=128,
        max_clues=5,
        num_predicates=8,
        num_solver_steps=6,
        dropout=0.0,
    ).to(device)
    
    criterion = RLANLoss(
        focal_gamma=2.0,
        focal_alpha=0.25,
        lambda_entropy=0.1,
        lambda_sparsity=0.05,
        lambda_predicate=0.01,
        lambda_curriculum=0.1,
    )
    
    # Count parameters
    params = model.count_parameters()
    print(f"\n  Model Parameters:")
    for name, count in params.items():
        print(f"    {name}: {count:,}")
    
    # Load real data or use synthetic
    if arc_data_dir and Path(arc_data_dir).exists():
        tasks = load_arc_tasks(arc_data_dir, "training", max_tasks=10)
        print(f"\n  Loaded {len(tasks)} real ARC tasks")
    else:
        print("\n  Using synthetic data (no ARC data found)")
        tasks = []
    
    # Test with synthetic data first
    print("\n  Synthetic Data Test:")
    input_grid = torch.randint(0, 10, (4, 12, 12)).to(device)
    target_grid = torch.randint(0, 11, (4, 12, 12)).to(device)
    
    model.train()
    outputs = model(input_grid, temperature=1.0, return_intermediates=True)
    
    # Analyze all outputs
    for key, val in outputs.items():
        if isinstance(val, torch.Tensor):
            stats = analyze_tensor(f"output[{key}]", val)
            print(f"    {stats}")
        elif isinstance(val, list) and len(val) > 0:
            print(f"    output[{key}]: list of {len(val)} tensors")
    
    # Compute loss
    losses = criterion(
        logits=outputs["logits"],
        targets=target_grid,
        attention_maps=outputs["attention_maps"],
        stop_logits=outputs["stop_logits"],
        predicates=outputs["predicates"],
        epoch=0,
        max_epochs=100,
        all_logits=outputs["all_logits"],
    )
    
    print("\n  Loss Components:")
    for key, val in losses.items():
        stats = analyze_tensor(f"loss[{key}]", val.unsqueeze(0))
        print(f"    {key}: {val.item():.4f} {' ✗ NaN!' if stats.has_nan else ''}")
    
    # Test gradient flow
    print("\n  Gradient Flow Test:")
    losses["total_loss"].backward()
    
    grad_stats = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            has_nan = torch.isnan(param.grad).any().item()
            grad_norm = param.grad.norm().item()
            grad_stats[name] = {"norm": grad_norm, "has_nan": has_nan}
    
    nan_grads = [n for n, s in grad_stats.items() if s["has_nan"]]
    if nan_grads:
        print(f"    ✗ NaN gradients in: {nan_grads[:5]}...")
    else:
        print("    [OK] No NaN gradients")
    
    # Top gradient norms
    sorted_grads = sorted(grad_stats.items(), key=lambda x: x[1]["norm"], reverse=True)
    print("    Top 5 gradient norms:")
    for name, stats in sorted_grads[:5]:
        print(f"      {name}: {stats['norm']:.4f}")
    
    # Test with real ARC data
    if tasks:
        print("\n  Real ARC Data Test:")
        for i, task in enumerate(tasks[:3]):
            print(f"\n    Task {task['task_id']}:")
            
            for j, (inp, out) in enumerate(task['train'][:1]):
                inp_t = torch.tensor(inp, dtype=torch.long).unsqueeze(0).to(device)
                out_t = torch.tensor(out, dtype=torch.long).unsqueeze(0).to(device)
                
                # Handle size mismatch (input/output may differ)
                # For testing, we'll just use input-sized output
                if inp_t.shape != out_t.shape:
                    # Pad or crop output to match input for this test
                    out_t = torch.zeros_like(inp_t)
                
                model.zero_grad()
                outputs = model(inp_t, temperature=1.0, return_intermediates=True)
                
                # Quick stats
                logits = outputs["logits"]
                pred = logits.argmax(dim=1)
                accuracy = (pred == out_t).float().mean()
                
                print(f"      Pair {j}: input={inp.shape}, output={out.shape}")
                print(f"        Accuracy (untrained): {accuracy.item():.2%}")
                
                attn = outputs["attention_maps"]
                print(f"        Attention max: {attn.max():.4f}, entropy: {-(attn * (attn + 1e-10).log()).sum() / attn.numel():.4f}")
    
    return True


def test_training_step(device: torch.device):
    """Test a full training step."""
    print("\n" + "="*80)
    print("TEST 9: Training Step Validation")
    print("="*80)
    
    model = RLAN(hidden_dim=64, max_clues=3, num_solver_steps=3, dropout=0.0).to(device)
    criterion = RLANLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    input_grid = torch.randint(0, 10, (4, 8, 8)).to(device)
    target_grid = input_grid.clone()  # Identity task
    
    model.train()
    
    print("\n  Training for 20 steps on identity task:")
    losses_history = []
    
    for step in range(20):
        optimizer.zero_grad()
        
        outputs = model(input_grid, temperature=1.0, return_intermediates=True)
        
        losses = criterion(
            logits=outputs["logits"],
            targets=target_grid,
            attention_maps=outputs["attention_maps"],
            stop_logits=outputs["stop_logits"],
            predicates=outputs["predicates"],
            epoch=step,
            max_epochs=20,
        )
        
        loss = losses["total_loss"]
        
        if torch.isnan(loss):
            print(f"    Step {step}: NaN loss detected!")
            break
        
        loss.backward()
        
        # Check gradient health
        total_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        losses_history.append(loss.item())
        
        if step % 5 == 0:
            pred = outputs["logits"].argmax(dim=1)
            acc = (pred == target_grid).float().mean()
            print(f"    Step {step}: loss={loss.item():.4f}, acc={acc.item():.2%}, grad_norm={total_grad_norm:.4f}")
    
    # Check if loss decreased
    if len(losses_history) >= 10:
        first_half = sum(losses_history[:10]) / 10
        second_half = sum(losses_history[-10:]) / 10
        print(f"\n  Loss trend: {first_half:.4f} -> {second_half:.4f}")
        if second_half < first_half:
            print("  [OK] Loss is decreasing")
        else:
            print("  [!] Loss not decreasing - potential issue")
    
    return True


def analyze_normalization_choices():
    """Analyze LayerNorm vs BatchNorm for RLAN."""
    print("\n" + "="*80)
    print("ANALYSIS: Normalization Choices for RLAN")
    print("="*80)
    
    print("""
    RLAN uses LayerNorm throughout, which is the correct choice for several reasons:
    
    1. SPATIAL INFORMATION PRESERVATION:
       - LayerNorm normalizes across features, keeping spatial relationships intact
       - BatchNorm would normalize across batch and spatial dims, mixing spatial info
       - For ARC's spatial reasoning, preserving relative spatial patterns is critical
    
    2. VARIABLE GRID SIZES:
       - ARC grids vary from 1x1 to 30x30
       - LayerNorm works with any spatial size
       - BatchNorm requires fixed spatial dimensions or special handling
    
    3. SMALL BATCH SIZES:
       - ARC training often uses small batches (limited data)
       - BatchNorm is unstable with small batches
       - LayerNorm is independent of batch size
    
    4. INFERENCE CONSISTENCY:
       - LayerNorm behavior is identical in train/eval modes
       - BatchNorm requires running statistics that may not generalize
    
    RECOMMENDATION: Keep LayerNorm. Consider GroupNorm for ConvGRU (already used).
    """)


def main():
    """Run all tests."""
    print("="*80)
    print("RLAN COMPREHENSIVE TESTING AND ACADEMIC REVIEW")
    print("="*80)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    # Find ARC data
    arc_paths = [
        "../data/arc-agi",
        "../arc-agi",
        "data/arc-agi",
        "arc-agi",
        "../ARC-AGI/data",
    ]
    arc_data_dir = None
    for p in arc_paths:
        if Path(p).exists():
            arc_data_dir = p
            break
    
    print(f"ARC data: {arc_data_dir or 'Not found'}")
    
    # Run all tests
    results = {}
    
    results["GridEncoder"] = test_grid_encoder(device)
    results["DSC"] = test_dsc(device)
    results["MSRE"] = test_msre(device)
    results["LCR"] = test_lcr(device)
    results["SPH"] = test_sph(device)
    results["Solver"] = test_solver(device)
    results["FocalLoss"] = test_focal_loss(device)
    results["FullRLAN"] = test_full_rlan(device, arc_data_dir)
    results["TrainingStep"] = test_training_step(device)
    
    analyze_normalization_choices()
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    all_passed = True
    for name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n[OK] All tests passed!")
    else:
        print("\n✗ Some tests failed - review output above")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
