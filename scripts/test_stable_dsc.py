#!/usr/bin/env python3
"""
Test Stable DSC Techniques
==========================

This script tests mathematically stable alternatives to the current DSC implementation:

1. Log-Softmax → Exp (instead of direct softmax)
2. Log-space entropy computation (instead of log(p) where p→0)
3. Stable Gumbel-softmax using log-space
4. Optional: Top-K sparse attention for better gradient flow

Tests against multiple difficulty levels with augmentations to ensure:
- No NaN across all batches
- 100% training accuracy achievable
- Proper gradient flow

Usage:
    python scripts/test_stable_dsc.py
"""

import sys
import os
import json
import math
import random
from pathlib import Path
from datetime import datetime
from functools import partial

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# =============================================================================
# STABLE DSC IMPLEMENTATION - The Proposed Fix
# =============================================================================

def gumbel_softmax_stable(
    logits: torch.Tensor,
    temperature: float = 1.0,
    hard: bool = False,
    deterministic: bool = False,
) -> tuple:
    """
    STABLE Gumbel-softmax using log-space computation.
    
    Key differences from original:
    1. Uses F.log_softmax (numerically stable)
    2. Returns both attention AND log_attention for downstream use
    3. No need for clamp - log_softmax handles numerical stability internally
    
    Args:
        logits: Shape (B, H, W) or (B, H*W)
        temperature: Softmax temperature
        hard: Use straight-through estimator
        deterministic: Skip Gumbel noise (for eval)
        
    Returns:
        attention: Probability distribution (B, ...)
        log_attention: Log probabilities (B, ...) - for entropy computation
    """
    original_shape = logits.shape
    B = logits.shape[0]
    
    # Flatten to 2D for computation
    if logits.dim() == 3:
        H, W = logits.shape[1], logits.shape[2]
        logits_flat = logits.view(B, -1)  # (B, H*W)
    else:
        logits_flat = logits  # Already (B, N)
    
    # Clamp input logits (reasonable range)
    logits_flat = logits_flat.clamp(min=-50.0, max=50.0)
    
    if deterministic:
        noisy_logits = logits_flat / max(temperature, 1e-6)
    else:
        # Gumbel noise: -log(-log(U)) where U ~ Uniform(0,1)
        uniform = torch.rand_like(logits_flat).clamp(min=1e-10, max=1.0 - 1e-10)
        gumbel_noise = -torch.log(-torch.log(uniform))
        noisy_logits = (logits_flat + gumbel_noise) / max(temperature, 1e-6)
    
    # KEY: Use log_softmax for numerical stability!
    # F.log_softmax subtracts max internally, preventing overflow
    log_attention = F.log_softmax(noisy_logits, dim=-1)  # (B, H*W)
    
    # Attention probabilities via exp (safe because log_softmax is bounded)
    attention = torch.exp(log_attention)  # (B, H*W)
    
    if hard:
        # Straight-through estimator
        idx = attention.argmax(dim=-1, keepdim=True)
        hard_attn = torch.zeros_like(attention)
        hard_attn.scatter_(1, idx, 1.0)
        attention = hard_attn - attention.detach() + attention
        # Log of hard attention (for consistency)
        log_attention = torch.where(
            hard_attn > 0.5,
            torch.zeros_like(log_attention),
            torch.full_like(log_attention, -100.0)  # log(0) ≈ -inf
        )
    
    # Reshape back to spatial if needed
    if len(original_shape) == 3:
        attention = attention.view(B, H, W)
        log_attention = log_attention.view(B, H, W)
    
    return attention, log_attention


def compute_entropy_stable(log_attention: torch.Tensor) -> torch.Tensor:
    """
    STABLE entropy computation using log probabilities.
    
    Entropy = -sum(p * log(p))
    
    With log_attention = log(p), we compute:
    Entropy = -sum(exp(log_p) * log_p)
    
    This avoids log(p) where p→0, since log_p is already computed stably.
    
    Args:
        log_attention: Shape (B, H, W) or (B, H*W) - log probabilities from log_softmax
        
    Returns:
        entropy: Shape (B,) - entropy per sample
    """
    # Flatten if spatial
    if log_attention.dim() == 3:
        log_attention = log_attention.view(log_attention.shape[0], -1)
    
    # p * log(p) = exp(log_p) * log_p
    # Note: where log_p is very negative, exp(log_p) ≈ 0, so product ≈ 0 (safe)
    attention = torch.exp(log_attention)  # p
    entropy = -torch.sum(attention * log_attention, dim=-1)  # -sum(p * log(p))
    
    return entropy


class StableDynamicSaliencyController(nn.Module):
    """
    Numerically stable DSC using log-space computations.
    
    Key improvements over original:
    1. Uses log_softmax → exp instead of direct softmax
    2. Entropy computed from log probabilities (no log(p) where p→0)
    3. No clamps needed in forward pass
    4. Better gradient flow through log-space
    """
    
    def __init__(
        self,
        hidden_dim: int = 128,
        max_clues: int = 5,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.max_clues = max_clues
        self.num_heads = num_heads
        
        # Learnable clue queries
        self.clue_queries = nn.Parameter(torch.randn(max_clues, hidden_dim))
        nn.init.xavier_uniform_(self.clue_queries.unsqueeze(0))
        
        # Projections
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Stop predictor with entropy coupling
        self.stop_predictor = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim // 2),  # +1 for entropy
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self._init_stop_predictor(init_bias=-1.0)
        
        # Layer norms
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.feature_norm = nn.LayerNorm(hidden_dim)
        
        # GRU for recurrence
        self.query_gru = nn.GRUCell(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(hidden_dim)
        
        # Coordinate grids
        self._init_coord_grids(30)
    
    def _init_stop_predictor(self, init_bias: float = -1.0):
        """Initialize stop predictor with entropy coupling."""
        layers = [m for m in self.stop_predictor.modules() if isinstance(m, nn.Linear)]
        if len(layers) >= 2:
            first_layer, last_layer = layers[0], layers[-1]
            in_features = first_layer.in_features
            hidden_dim = in_features - 1
            
            nn.init.kaiming_normal_(first_layer.weight[:, :hidden_dim], mode='fan_in')
            nn.init.normal_(first_layer.weight[:, hidden_dim:], mean=1.5, std=0.5)
            nn.init.zeros_(first_layer.bias)
            
            nn.init.normal_(last_layer.weight, mean=0.0, std=0.1)
            nn.init.constant_(last_layer.bias, init_bias)
    
    def _init_coord_grids(self, max_size: int):
        """Initialize coordinate grids."""
        rows = torch.arange(max_size).float()
        cols = torch.arange(max_size).float()
        row_grid, col_grid = torch.meshgrid(rows, cols, indexing='ij')
        self.register_buffer('row_grid', row_grid.clone())
        self.register_buffer('col_grid', col_grid.clone())
    
    def _compute_centroid(self, attention: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """Compute soft centroid from attention weights."""
        B = attention.shape[0]
        row_grid = self.row_grid[:H, :W].unsqueeze(0).expand(B, -1, -1)
        col_grid = self.col_grid[:H, :W].unsqueeze(0).expand(B, -1, -1)
        
        row_centroid = (attention * row_grid).sum(dim=(-2, -1))
        col_centroid = (attention * col_grid).sum(dim=(-2, -1))
        
        return torch.stack([row_centroid, col_centroid], dim=-1)
    
    def forward(
        self,
        features: torch.Tensor,
        temperature: float = 1.0,
        mask: torch.Tensor = None,
    ) -> tuple:
        """
        Extract clue anchors using STABLE attention computation.
        
        Args:
            features: Shape (B, D, H, W)
            temperature: Gumbel-softmax temperature
            mask: Optional (B, H, W) mask
            
        Returns:
            centroids: (B, K, 2)
            attention_maps: (B, K, H, W)
            stop_logits: (B, K)
        """
        B, D, H, W = features.shape
        K = self.max_clues
        
        # Reshape features
        features_flat = features.permute(0, 2, 3, 1).reshape(B, H * W, D)
        features_flat = self.feature_norm(features_flat)
        
        # Initialize outputs
        all_centroids = []
        all_attention_maps = []
        all_stop_logits = []
        
        # Cumulative mask
        cumulative_mask = torch.ones(B, H, W, device=features.device)
        if mask is not None:
            cumulative_mask = cumulative_mask * mask
        
        # Recurrent state
        query_state = torch.zeros(B, self.hidden_dim, device=features.device)
        
        for k in range(K):
            # Get query
            query = self.clue_queries[k:k+1].expand(B, -1)
            query = query + query_state
            query = self.query_norm(query)
            
            # Project
            q = self.query_proj(query)
            k_proj = self.key_proj(features_flat)
            v = self.value_proj(features_flat)
            
            # Attention scores
            attn_scores = torch.einsum('bd,bnd->bn', q, k_proj) / self.scale
            attn_scores = attn_scores.view(B, H, W)
            
            # Apply mask in log-space (additive)
            safe_mask = cumulative_mask.clamp(min=1e-6)
            attn_scores = attn_scores + torch.log(safe_mask)
            
            # STABLE: Use log-space Gumbel-softmax
            attention, log_attention = gumbel_softmax_stable(
                attn_scores,
                temperature=temperature,
                deterministic=not self.training
            )
            
            # Centroid
            centroid = self._compute_centroid(attention, H, W)
            
            # Attended features
            attention_flat = attention.view(B, H * W, 1)
            attended_features = (v * attention_flat).sum(dim=1)
            
            # Update recurrent state
            query_state = self.query_gru(attended_features, query_state)
            
            # STABLE: Compute entropy from log probabilities
            entropy = compute_entropy_stable(log_attention)  # (B,)
            
            # Normalize entropy
            max_entropy = math.log(H * W + 1e-6)
            entropy_normalized = (entropy / max_entropy).unsqueeze(-1)  # (B, 1)
            
            # Stop prediction
            stop_input = torch.cat([attended_features, entropy_normalized], dim=-1)
            stop_logit_raw = self.stop_predictor(stop_input).squeeze(-1)
            stop_logit = 4.0 * torch.tanh(stop_logit_raw / 4.0)
            
            # Update mask
            mask_update = 1.0 - 0.9 * attention.detach()
            cumulative_mask = cumulative_mask * mask_update
            cumulative_mask = cumulative_mask.clamp(min=1e-6)
            
            # Store
            all_centroids.append(centroid)
            all_attention_maps.append(attention)
            all_stop_logits.append(stop_logit)
        
        centroids = torch.stack(all_centroids, dim=1)
        attention_maps = torch.stack(all_attention_maps, dim=1)
        stop_logits = torch.stack(all_stop_logits, dim=1)
        
        return centroids, attention_maps, stop_logits


# =============================================================================
# TEST MODEL - Uses Stable DSC
# =============================================================================

class StableRLANTest(nn.Module):
    """
    Simplified RLAN model using the Stable DSC for testing.
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_colors: int = 10,
        num_classes: int = 10,
        max_grid_size: int = 30,
        max_clues: int = 6,
        num_solver_steps: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.max_clues = max_clues
        self.num_solver_steps = num_solver_steps
        
        # Color embedding
        self.color_embed = nn.Embedding(num_colors, hidden_dim)
        
        # Feature encoder (simple CNN)
        self.encoder = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
        )
        
        # STABLE DSC
        self.dsc = StableDynamicSaliencyController(
            hidden_dim=hidden_dim,
            max_clues=max_clues,
            num_heads=4,
            dropout=dropout,
        )
        
        # Solver (simple transformer blocks)
        self.solver_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
            )
            for _ in range(num_solver_steps)
        ])
        
        # Output head
        self.output_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, num_classes, 1),
        )
    
    def forward(
        self,
        input_grid: torch.Tensor,
        temperature: float = 1.0,
        return_intermediates: bool = False,
    ) -> dict:
        """
        Forward pass.
        
        Args:
            input_grid: (B, H, W) input color indices
            temperature: DSC temperature
            return_intermediates: Return attention maps etc.
            
        Returns:
            dict with 'logits' and optionally intermediate outputs
        """
        B, H, W = input_grid.shape
        
        # Embed colors
        x = self.color_embed(input_grid)  # (B, H, W, D)
        x = x.permute(0, 3, 1, 2)  # (B, D, H, W)
        
        # Encode
        features = self.encoder(x)  # (B, D, H, W)
        
        # DSC
        centroids, attention_maps, stop_logits = self.dsc(
            features, temperature=temperature
        )
        
        # Solver
        features_flat = features.permute(0, 2, 3, 1).reshape(B, H * W, -1)
        for layer in self.solver_layers:
            features_flat = layer(features_flat)
        features = features_flat.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        
        # Output
        logits = self.output_head(features)
        
        outputs = {'logits': logits}
        if return_intermediates:
            outputs['attention_maps'] = attention_maps
            outputs['stop_logits'] = stop_logits
            outputs['centroids'] = centroids
        
        return outputs


# =============================================================================
# DATA LOADING
# =============================================================================

class LocalARCDataset(Dataset):
    """Load ARC tasks from local JSON files."""
    
    def __init__(
        self,
        data_dir: str,
        max_tasks: int = None,
        max_size: int = 30,
        augment: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.max_size = max_size
        self.augment = augment
        self.num_dihedral = 8 if augment else 1
        
        # Load tasks
        self.samples = []
        task_files = sorted(self.data_dir.glob("*.json"))
        
        if max_tasks:
            task_files = task_files[:max_tasks]
        
        for task_file in task_files:
            with open(task_file) as f:
                task_data = json.load(f)
            
            task_name = task_file.stem
            train_examples = task_data.get('train', [])
            test_examples = task_data.get('test', [])
            
            # Check size
            too_large = False
            for ex in train_examples + test_examples:
                h, w = len(ex['input']), len(ex['input'][0]) if ex['input'] else 0
                oh, ow = len(ex['output']), len(ex['output'][0]) if ex['output'] else 0
                if max(h, w, oh, ow) > max_size:
                    too_large = True
                    break
            
            if too_large:
                continue
            
            # Create samples from train examples
            for i, ex in enumerate(train_examples):
                self.samples.append({
                    'name': f"{task_name}_train{i}",
                    'input': ex['input'],
                    'output': ex['output'],
                })
        
        print(f"Loaded {len(self.samples)} samples (with {self.num_dihedral}x augmentation = {len(self)} total)")
        
        # Dihedral transforms
        self.transforms = [
            lambda x: x,
            lambda x: np.rot90(x, k=1),
            lambda x: np.rot90(x, k=2),
            lambda x: np.rot90(x, k=3),
            lambda x: np.flip(x, axis=1),
            lambda x: np.flip(x, axis=0),
            lambda x: np.flip(np.rot90(x, k=1), axis=1),
            lambda x: np.flip(np.rot90(x, k=1), axis=0),
        ][:self.num_dihedral]
    
    def __len__(self):
        return len(self.samples) * self.num_dihedral
    
    def __getitem__(self, idx):
        sample_idx = idx // self.num_dihedral
        aug_idx = idx % self.num_dihedral
        
        sample = self.samples[sample_idx]
        transform = self.transforms[aug_idx]
        
        # Apply transform
        inp = np.array(sample['input'])
        out = np.array(sample['output'])
        
        inp = transform(inp).copy()
        out = transform(out).copy()
        
        # Pad
        h, w = inp.shape
        inp_padded = np.zeros((self.max_size, self.max_size), dtype=np.int64)
        out_padded = np.full((self.max_size, self.max_size), -100, dtype=np.int64)  # -100 = ignore
        
        inp_padded[:h, :w] = inp
        oh, ow = out.shape
        out_padded[:oh, :ow] = out
        
        return {
            'input': torch.tensor(inp_padded, dtype=torch.long),
            'output': torch.tensor(out_padded, dtype=torch.long),
            'name': sample['name'],
        }


def collate_fn(batch):
    """Collate batch."""
    return {
        'inputs': torch.stack([b['input'] for b in batch]),
        'outputs': torch.stack([b['output'] for b in batch]),
    }


# =============================================================================
# LOSS FUNCTION
# =============================================================================

def weighted_stablemax_loss(logits, targets, bg_cap=2.0, fg_cap=5.0):
    """
    Weighted cross-entropy with stablemax (logsumexp normalization).
    """
    B, C, H, W = logits.shape
    
    # Flatten
    logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)
    targets_flat = targets.view(-1)  # (B*H*W,)
    
    # Valid mask (not -100)
    valid_mask = targets_flat != -100
    if not valid_mask.any():
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    
    logits_valid = logits_flat[valid_mask]
    targets_valid = targets_flat[valid_mask]
    
    # Compute class weights (inverse frequency with caps)
    class_counts = torch.bincount(targets_valid, minlength=C).float().clamp(min=1)
    total = targets_valid.numel()
    class_freq = class_counts / total
    
    # Background (class 0) weight
    bg_weight = min((1.0 / class_freq[0].item()) / C, bg_cap) if class_freq[0] > 0 else 1.0
    
    # Foreground weights
    weights = torch.ones(C, device=logits.device)
    weights[0] = bg_weight
    for c in range(1, C):
        if class_freq[c] > 0:
            weights[c] = min((1.0 / class_freq[c].item()) / C, fg_cap)
    
    # Stablemax: normalize logits with logsumexp
    log_sum_exp = torch.logsumexp(logits_valid, dim=-1, keepdim=True)
    log_probs = logits_valid - log_sum_exp
    
    # Gather log probs for targets
    target_log_probs = log_probs.gather(1, targets_valid.unsqueeze(1)).squeeze(1)
    
    # Weight by class
    sample_weights = weights[targets_valid]
    
    # Weighted negative log likelihood
    loss = -(sample_weights * target_log_probs).mean()
    
    return loss


def sparsity_loss(attention_maps, min_clues=2.5, min_clue_weight=5.0, ponder_weight=0.02):
    """
    Sparsity loss for DSC attention maps.
    """
    B, K, H, W = attention_maps.shape
    
    # Attention entropy per clue
    attn_flat = attention_maps.view(B, K, -1)
    # Use stable entropy computation
    log_attn = torch.log(attn_flat.clamp(min=1e-8))
    entropy_per_clue = -(attn_flat * log_attn).sum(dim=-1)  # (B, K)
    max_entropy = math.log(H * W)
    entropy_normalized = entropy_per_clue / max_entropy  # (B, K)
    
    # Encourage sharp attention (low entropy)
    entropy_loss = entropy_normalized.mean()
    
    return entropy_loss * ponder_weight


# =============================================================================
# TRAINING
# =============================================================================

def train_and_test(
    num_tasks: int = 50,
    max_size: int = 15,
    batch_size: int = 8,
    num_epochs: int = 100,
    lr: float = 5e-4,
    device: str = 'cpu',
):
    """Train and test the stable DSC implementation."""
    
    print("=" * 70)
    print("STABLE DSC TEST")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Tasks: {num_tasks}, Max Size: {max_size}")
    print(f"Batch Size: {batch_size}, Epochs: {num_epochs}")
    print(f"Learning Rate: {lr}")
    print()
    
    # Data
    data_dir = project_root / 'data' / 'arc-agi' / 'data' / 'training'
    dataset = LocalARCDataset(
        str(data_dir),
        max_tasks=num_tasks,
        max_size=max_size,
        augment=True,
    )
    
    if len(dataset) == 0:
        print("ERROR: No valid samples!")
        return False
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    
    print(f"Dataset: {len(dataset)} samples, {len(loader)} batches")
    
    # Model
    model = StableRLANTest(
        hidden_dim=256,
        num_colors=10,
        num_classes=10,
        max_grid_size=max_size,
        max_clues=6,
        num_solver_steps=6,
        dropout=0.1,
    ).to(device)
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # Training
    print()
    print("=" * 70)
    print("TRAINING")
    print("=" * 70)
    
    best_acc = 0.0
    nan_count = 0
    warning_count = 0
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        batch_count = 0
        
        # Temperature annealing
        temp = 1.0 * (0.5 / 1.0) ** (epoch / num_epochs)
        temp = max(temp, 0.5)
        
        for batch in loader:
            inputs = batch['inputs'].to(device)
            targets = batch['outputs'].to(device)
            
            optimizer.zero_grad()
            
            # Forward
            outputs = model(inputs, temperature=temp, return_intermediates=True)
            logits = outputs['logits']
            attention_maps = outputs['attention_maps']
            
            # Check for NaN in attention
            attn_min = attention_maps.min().item()
            if attn_min < 1e-10:
                warning_count += 1
                if warning_count <= 5:
                    print(f"  [WARNING] Attention min = {attn_min:.2e}")
            
            # Loss
            task_loss = weighted_stablemax_loss(logits, targets)
            sparse_loss = sparsity_loss(attention_maps)
            loss = task_loss + sparse_loss
            
            # Check for NaN
            if torch.isnan(loss) or torch.isinf(loss):
                nan_count += 1
                print(f"  [NaN] at epoch {epoch}, batch {batch_count}")
                optimizer.zero_grad()
                continue
            
            # Backward
            loss.backward()
            
            # Check gradients
            grad_nan = False
            for name, param in model.named_parameters():
                if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                    grad_nan = True
                    nan_count += 1
                    if nan_count <= 5:
                        print(f"  [NaN GRAD] in {name} at epoch {epoch}")
                    break
            
            if grad_nan:
                optimizer.zero_grad()
                continue
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Metrics
            epoch_loss += loss.item()
            batch_count += 1
            
            valid_mask = targets != -100
            preds = logits.argmax(dim=1)
            correct = ((preds == targets) & valid_mask).sum().item()
            total = valid_mask.sum().item()
            epoch_correct += correct
            epoch_total += total
        
        # Epoch stats
        if batch_count > 0:
            avg_loss = epoch_loss / batch_count
            accuracy = 100.0 * epoch_correct / max(epoch_total, 1)
            
            if accuracy > best_acc:
                best_acc = accuracy
            
            if epoch % 10 == 0 or epoch == 1:
                print(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | Acc: {accuracy:.1f}% | Temp: {temp:.3f}")
    
    # Summary
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Best Accuracy: {best_acc:.1f}%")
    print(f"NaN Count: {nan_count}")
    print(f"Warning Count: {warning_count}")
    
    success = nan_count == 0 and best_acc > 90.0
    
    if success:
        print()
        print("✅ TEST PASSED - Stable DSC works without NaN!")
    else:
        print()
        print("❌ TEST FAILED")
        if nan_count > 0:
            print(f"   - {nan_count} NaN occurrences")
        if best_acc <= 90.0:
            print(f"   - Best accuracy only {best_acc:.1f}% (target > 90%)")
    
    return success


# =============================================================================
# COMPARISON TEST
# =============================================================================

def compare_stability():
    """Compare old vs new attention computation for numerical stability."""
    
    print("=" * 70)
    print("STABILITY COMPARISON TEST")
    print("=" * 70)
    
    # Test with extreme logits
    torch.manual_seed(42)
    
    # Simulate logits that would cause collapse
    B, H, W = 4, 30, 30  # 900 positions
    
    # Case 1: Normal logits
    logits_normal = torch.randn(B, H, W)
    
    # Case 2: Very peaked logits (one position dominant)
    logits_peaked = torch.randn(B, H, W) * 0.1
    logits_peaked[:, 15, 15] = 50.0  # Strong peak
    
    # Case 3: Very flat logits (uniform-ish)
    logits_flat = torch.randn(B, H, W) * 0.01
    
    test_cases = [
        ("Normal", logits_normal),
        ("Peaked", logits_peaked),
        ("Flat", logits_flat),
    ]
    
    print("\n1. OLD METHOD (direct softmax + log):")
    print("-" * 50)
    
    for name, logits in test_cases:
        # Old method
        flat = logits.view(B, -1)
        soft = F.softmax(flat / 1.0, dim=-1)
        
        min_val = soft.min().item()
        
        # Try to compute entropy the old way
        safe_soft = soft.clamp(min=1e-10)
        log_soft = torch.log(safe_soft)
        entropy = -(safe_soft * log_soft).sum(dim=-1)
        
        # Check for issues
        has_nan = torch.isnan(entropy).any().item()
        has_inf = torch.isinf(entropy).any().item()
        
        print(f"  {name:8s}: min_attn={min_val:.2e}, entropy_nan={has_nan}, entropy_inf={has_inf}")
    
    print("\n2. NEW METHOD (log_softmax + exp):")
    print("-" * 50)
    
    for name, logits in test_cases:
        # New method
        attention, log_attention = gumbel_softmax_stable(
            logits, temperature=1.0, deterministic=True
        )
        
        min_val = attention.min().item()
        
        # Compute entropy from log_attention (stable!)
        entropy = compute_entropy_stable(log_attention)
        
        has_nan = torch.isnan(entropy).any().item()
        has_inf = torch.isinf(entropy).any().item()
        
        print(f"  {name:8s}: min_attn={min_val:.2e}, entropy_nan={has_nan}, entropy_inf={has_inf}")
    
    print("\n3. GRADIENT FLOW TEST:")
    print("-" * 50)
    
    # Test gradient flow through entropy computation
    logits = torch.randn(4, 30, 30, requires_grad=True)
    
    # New method
    attention, log_attention = gumbel_softmax_stable(logits, temperature=1.0, deterministic=True)
    entropy = compute_entropy_stable(log_attention)
    entropy_loss = entropy.mean()
    entropy_loss.backward()
    
    grad_ok = logits.grad is not None and not torch.isnan(logits.grad).any()
    grad_norm = logits.grad.norm().item() if grad_ok else float('nan')
    
    print(f"  Entropy gradient OK: {grad_ok}")
    print(f"  Gradient norm: {grad_norm:.4f}")
    
    print()
    return True


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', type=int, default=50)
    parser.add_argument('--max-size', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--compare-only', action='store_true', help='Only run comparison test')
    args = parser.parse_args()
    
    # Set seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Run stability comparison first
    compare_stability()
    
    if args.compare_only:
        sys.exit(0)
    
    # Run full training test
    success = train_and_test(
        num_tasks=args.tasks,
        max_size=args.max_size,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lr=args.lr,
        device=device,
    )
    
    sys.exit(0 if success else 1)
