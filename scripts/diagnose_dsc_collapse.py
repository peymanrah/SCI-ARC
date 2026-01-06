#!/usr/bin/env python3
"""
DSC and Centroid Collapse Diagnostic Script

This script loads a warmup checkpoint and runs inference/training on a small
batch of ARC tasks to diagnose why:
1. Centroid Spread = 0.00 (all clues at same location)
2. All Clues IDENTICAL (entropy_std = 0.0000)
3. Spatial Attention COLLAPSED (max = 0.001)

We will trace through every DSC-related operation to pinpoint the root cause.

Author: AI Research Assistant
Date: January 2026
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import math
from pathlib import Path
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt

# Force CPU
DEVICE = 'cpu'
torch.set_num_threads(4)  # Limit CPU threads for faster execution

# =============================================================================
# LOGGING UTILITIES
# =============================================================================

class DSCDiagnosticLogger:
    """Logs all DSC-related operations for debugging."""
    
    def __init__(self):
        self.logs = defaultdict(list)
        self.batch_idx = 0
        
    def log(self, key: str, value, batch_idx: int = None):
        """Log a value with optional batch index."""
        if batch_idx is None:
            batch_idx = self.batch_idx
        if torch.is_tensor(value):
            value = value.detach().cpu()
        self.logs[key].append({'batch': batch_idx, 'value': value})
        
    def new_batch(self):
        self.batch_idx += 1
        
    def print_summary(self):
        """Print summary of all logged values."""
        print("\n" + "=" * 70)
        print("DSC DIAGNOSTIC SUMMARY")
        print("=" * 70)
        
        for key in sorted(self.logs.keys()):
            values = self.logs[key]
            print(f"\n{key}:")
            for entry in values[-3:]:  # Last 3 entries
                v = entry['value']
                if torch.is_tensor(v):
                    if v.numel() == 1:
                        print(f"  batch {entry['batch']}: {v.item():.6f}")
                    elif v.numel() <= 20:
                        print(f"  batch {entry['batch']}: {v.tolist()}")
                    else:
                        print(f"  batch {entry['batch']}: shape={v.shape}, mean={v.mean():.6f}, std={v.std():.6f}, min={v.min():.6f}, max={v.max():.6f}")
                else:
                    print(f"  batch {entry['batch']}: {v}")


LOGGER = DSCDiagnosticLogger()


# =============================================================================
# HOOK-BASED DSC TRACING
# =============================================================================

def trace_dsc_forward(dsc_module, features, temperature, task_context=None):
    """
    Manually trace through DSC forward pass with detailed logging.
    This replicates the DSC forward but logs every intermediate value.
    """
    B, D, H, W = features.shape
    K = dsc_module.max_clues
    
    LOGGER.log("input_features_shape", f"B={B}, D={D}, H={H}, W={W}")
    LOGGER.log("input_features_stats", {
        'mean': features.mean().item(),
        'std': features.std().item(),
        'min': features.min().item(),
        'max': features.max().item(),
    })
    
    # Check if features are diverse or collapsed
    feature_per_position = features.permute(0, 2, 3, 1).reshape(B, H*W, D)
    position_variance = feature_per_position.var(dim=1).mean()  # Variance across positions
    LOGGER.log("feature_position_variance", position_variance)
    
    if position_variance < 0.01:
        print(f"  ⚠️ WARNING: Feature position variance = {position_variance:.6f} < 0.01")
        print(f"     Features may be spatially uniform - DSC cannot differentiate positions!")
    
    # Check clue query diversity
    queries = dsc_module.clue_queries  # (K, D)
    query_norms = queries.norm(dim=-1)
    LOGGER.log("clue_query_norms", query_norms)
    
    # Pairwise cosine similarity between queries
    queries_norm = F.normalize(queries, dim=-1)
    query_sim = queries_norm @ queries_norm.t()
    off_diag_mask = ~torch.eye(K, dtype=bool)
    query_sim_off_diag = query_sim[off_diag_mask]
    LOGGER.log("clue_query_cosine_similarity_off_diag", {
        'mean': query_sim_off_diag.mean().item(),
        'max': query_sim_off_diag.max().item(),
        'min': query_sim_off_diag.min().item(),
    })
    
    if query_sim_off_diag.mean() > 0.8:
        print(f"  ⚠️ WARNING: Clue queries too similar! Mean cosine sim = {query_sim_off_diag.mean():.4f}")
        print(f"     Queries should be diverse to attend to different patterns!")
    
    # Prepare task context
    if task_context is None:
        task_context_vec = torch.zeros(B, dsc_module.context_dim, device=features.device)
    elif task_context.dim() == 4:
        task_context_vec = task_context.mean(dim=(2, 3))
    else:
        task_context_vec = task_context
    
    LOGGER.log("task_context_norm", task_context_vec.norm(dim=-1).mean())
    
    # Flatten features
    features_flat = features.permute(0, 2, 3, 1).reshape(B, H * W, D)
    features_flat = dsc_module.feature_norm(features_flat)
    
    LOGGER.log("features_flat_after_norm", {
        'mean': features_flat.mean().item(),
        'std': features_flat.std().item(),
    })
    
    # Process each clue
    cumulative_mask = torch.ones(B, H, W, device=features.device)
    query_state = torch.zeros(B, dsc_module.hidden_dim, device=features.device)
    
    all_centroids = []
    all_attention_maps = []
    all_stop_logits = []
    all_attn_entropies = []
    
    for k in range(K):
        print(f"\n  --- Clue {k} ---")
        
        # Get query
        query = dsc_module.clue_queries[k:k+1].expand(B, -1)
        query = query + query_state
        query = dsc_module.query_norm(query)
        
        LOGGER.log(f"clue_{k}_query_norm", query.norm(dim=-1).mean())
        
        # Project
        q = dsc_module.query_proj(query)
        k_proj = dsc_module.key_proj(features_flat)
        v = dsc_module.value_proj(features_flat)
        
        # Attention scores
        attn_scores = torch.einsum('bd,bnd->bn', q, k_proj) / dsc_module.scale
        attn_scores = attn_scores.view(B, H, W)
        
        LOGGER.log(f"clue_{k}_attn_scores_raw", {
            'mean': attn_scores.mean().item(),
            'std': attn_scores.std().item(),
            'min': attn_scores.min().item(),
            'max': attn_scores.max().item(),
        })
        
        # Check if attention scores are uniform
        attn_score_std = attn_scores.std(dim=(-2, -1)).mean()
        if attn_score_std < 0.1:
            print(f"    ⚠️ PROBLEM: Attention scores are nearly uniform! std={attn_score_std:.6f}")
            print(f"       This means q·k is constant across all positions!")
            print(f"       Possible causes:")
            print(f"         1. Features (k) are spatially uniform")
            print(f"         2. Query (q) is orthogonal to feature variation")
            print(f"         3. Keys/queries not trained to differentiate")
        
        # Apply cumulative mask
        safe_mask = cumulative_mask.clamp(min=1e-6)
        attn_scores_masked = attn_scores + torch.log(safe_mask)
        attn_scores_masked = attn_scores_masked.clamp(min=-50.0, max=50.0)
        
        # Softmax
        attn_flat = attn_scores_masked.view(B, -1)
        attention_flat = F.softmax(attn_flat / max(temperature, 1e-10), dim=-1)
        attention = attention_flat.view(B, H, W)
        
        attn_max = attention.max(dim=-1)[0].max(dim=-1)[0].mean()
        attn_min = attention.min(dim=-1)[0].min(dim=-1)[0].mean()
        
        LOGGER.log(f"clue_{k}_attention_max", attn_max)
        LOGGER.log(f"clue_{k}_attention_min", attn_min)
        
        print(f"    Attention: max={attn_max:.6f}, min={attn_min:.6f}")
        
        if attn_max < 0.01:
            print(f"    ⚠️ ATTENTION COLLAPSED: max={attn_max:.6f} < 0.01")
            print(f"       Attention is nearly uniform over {H*W} positions!")
            expected_uniform = 1.0 / (H * W)
            print(f"       Expected uniform = {expected_uniform:.6f}")
        
        # Compute centroid
        row_grid = dsc_module.row_grid[:H, :W].unsqueeze(0).expand(B, -1, -1)
        col_grid = dsc_module.col_grid[:H, :W].unsqueeze(0).expand(B, -1, -1)
        row_centroid = (attention * row_grid).sum(dim=(-2, -1))
        col_centroid = (attention * col_grid).sum(dim=(-2, -1))
        centroid = torch.stack([row_centroid, col_centroid], dim=-1)
        
        LOGGER.log(f"clue_{k}_centroid", centroid[0])  # First batch
        print(f"    Centroid: ({row_centroid[0].item():.2f}, {col_centroid[0].item():.2f})")
        
        # Compute entropy
        attn_clamped = attention.view(B, -1).clamp(min=1e-6)
        log_attn = torch.log(attn_clamped)
        entropy = -(attn_clamped * log_attn).sum(dim=-1)
        max_entropy = math.log(H * W)
        entropy_normalized = entropy / max_entropy
        
        LOGGER.log(f"clue_{k}_entropy", entropy[0])
        LOGGER.log(f"clue_{k}_entropy_normalized", entropy_normalized[0])
        print(f"    Entropy: {entropy[0].item():.4f} / {max_entropy:.4f} = {entropy_normalized[0].item():.4f}")
        
        all_attn_entropies.append(entropy_normalized[0].item())
        
        # Attended features and stop prediction
        attention_flat_v = attention.view(B, H * W, 1)
        attended_features = (v * attention_flat_v).sum(dim=1)
        
        query_state = dsc_module.query_gru(attended_features, query_state)
        
        confidence = 1.0 - entropy_normalized.unsqueeze(-1)
        stop_input = torch.cat([attended_features, confidence, task_context_vec], dim=-1)
        stop_logit = dsc_module.stop_predictor(stop_input).squeeze(-1)
        stop_prob = torch.sigmoid(stop_logit)
        
        LOGGER.log(f"clue_{k}_stop_logit", stop_logit[0])
        LOGGER.log(f"clue_{k}_stop_prob", stop_prob[0])
        print(f"    Stop: logit={stop_logit[0].item():.4f}, prob={stop_prob[0].item():.4f}")
        
        all_centroids.append(centroid)
        all_attention_maps.append(attention)
        all_stop_logits.append(stop_logit)
        
        # Update cumulative mask (progressive masking)
        # This should make next clue attend to different location
        cumulative_mask = cumulative_mask * (1.0 - attention)
        cumulative_mask = cumulative_mask.clamp(min=1e-6)
    
    # Stack results
    centroids = torch.stack(all_centroids, dim=1)  # (B, K, 2)
    attention_maps = torch.stack(all_attention_maps, dim=1)  # (B, K, H, W)
    stop_logits = torch.stack(all_stop_logits, dim=1)  # (B, K)
    
    # Compute centroid spread
    centroid_mean = centroids.mean(dim=1, keepdim=True)
    spread = ((centroids - centroid_mean) ** 2).sum(dim=-1).sqrt().mean()
    
    print(f"\n  --- Summary ---")
    print(f"  Centroid spread: {spread.item():.4f}")
    print(f"  Entropy std across clues: {np.std(all_attn_entropies):.6f}")
    
    if spread < 0.5:
        print(f"  ⚠️ CENTROID COLLAPSE: spread={spread:.4f} < 0.5")
        print(f"     All clues pointing to same location!")
    
    if np.std(all_attn_entropies) < 0.01:
        print(f"  ⚠️ ALL CLUES IDENTICAL: entropy_std={np.std(all_attn_entropies):.6f}")
        print(f"     DSC is not differentiating between clues!")
    
    LOGGER.log("centroid_spread", spread)
    LOGGER.log("entropy_std_across_clues", np.std(all_attn_entropies))
    
    return centroids, attention_maps, stop_logits


# =============================================================================
# LOAD MODEL AND DATA
# =============================================================================

def load_model_from_checkpoint(checkpoint_path: str, device: str = 'cpu'):
    """Load RLAN model from checkpoint."""
    from sci_arc.models.rlan import RLAN, RLANConfig
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get config from checkpoint - handle nested structure
    if 'config' in checkpoint:
        full_config = checkpoint['config']
        if isinstance(full_config, dict) and 'model' in full_config:
            model_config = full_config['model']
            print(f"  Found nested config under 'model' key")
        else:
            model_config = full_config
    else:
        # Fallback config
        model_config = {
            'hidden_dim': 128,
            'num_colors': 11,
            'num_classes': 10,
            'max_grid_size': 30,
            'num_solver_steps': 7,
            'max_clues': 7,
            'num_predicates': 8,
            'dropout': 0.1,
        }
    
    print(f"  Config keys: {list(model_config.keys())}")
    print(f"Creating model with config...")
    
    # Create RLANConfig from flat model config
    rlan_config = RLANConfig(
        hidden_dim=model_config.get('hidden_dim', 128),
        num_colors=model_config.get('num_colors', 11),
        num_classes=model_config.get('num_classes', 10),
        max_grid_size=model_config.get('max_grid_size', 30),
        max_clues=model_config.get('max_clues', 7),
        num_predicates=model_config.get('num_predicates', 8),
        num_solver_steps=model_config.get('num_solver_steps', 7),
        use_act=model_config.get('use_act', False),
        dropout=model_config.get('dropout', 0.1),
        dsc_num_heads=model_config.get('dsc_num_heads', 4),
        use_context_encoder=model_config.get('use_context_encoder', True),
        use_dsc=model_config.get('use_dsc', True),
        use_msre=model_config.get('use_msre', True),
        use_lcr=model_config.get('use_lcr', True),
        use_sph=model_config.get('use_sph', True),
        use_solver_context=model_config.get('use_solver_context', True),
        use_hyperlora=model_config.get('use_hyperlora', False),
        use_hpm=model_config.get('use_hpm', False),
    )
    
    model = RLAN(config=rlan_config).to(device)
    
    # Load state dict
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
    
    # Handle key remapping if needed
    if any(k.startswith('base_rlan.') for k in state_dict.keys()):
        print("  Found 'base_rlan.' prefix, using as-is")
    elif any(k.startswith('encoder.') for k in state_dict.keys()):
        print("  Loading plain RLAN weights")
    
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"  Missing keys: {len(missing)}")
    print(f"  Unexpected keys: {len(unexpected)}")
    
    model.eval()
    return model, model_config


def load_arc_tasks(data_dir: str, max_tasks: int = 5):
    """Load a few ARC tasks for testing."""
    tasks = []
    
    # Try training data first
    train_dir = Path(data_dir) / "arc-agi" / "data" / "training"
    if not train_dir.exists():
        train_dir = Path(data_dir) / "training"
    if not train_dir.exists():
        # Try finding any JSON files
        train_dir = Path(data_dir)
    
    print(f"Looking for tasks in: {train_dir}")
    
    json_files = list(train_dir.glob("*.json"))[:max_tasks]
    print(f"Found {len(json_files)} task files")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                task = json.load(f)
            task['task_id'] = json_file.stem
            tasks.append(task)
        except Exception as e:
            print(f"  Failed to load {json_file}: {e}")
    
    return tasks


def prepare_batch(tasks, max_size: int = 30, device: str = 'cpu'):
    """Prepare a batch of tasks for model input."""
    batch_size = len(tasks)
    
    # Find max pairs
    max_pairs = max(len(t.get('train', [])) for t in tasks)
    
    # Find actual max grid size
    actual_max = 1
    for task in tasks:
        for pair in task.get('train', []) + task.get('test', []):
            for grid in [pair.get('input', []), pair.get('output', [])]:
                if grid:
                    actual_max = max(actual_max, len(grid), max(len(r) for r in grid) if grid else 1)
    
    effective_max = min(actual_max + 2, max_size)  # Add padding margin
    
    print(f"Batch: {batch_size} tasks, max_pairs={max_pairs}, grid_size={effective_max}")
    
    # Initialize tensors
    PAD = 10
    IGNORE = -100
    
    input_grids = torch.full((batch_size, max_pairs, effective_max, effective_max), PAD, dtype=torch.long)
    output_grids = torch.full((batch_size, max_pairs, effective_max, effective_max), PAD, dtype=torch.long)
    test_inputs = torch.full((batch_size, effective_max, effective_max), PAD, dtype=torch.long)
    test_outputs = torch.full((batch_size, effective_max, effective_max), IGNORE, dtype=torch.long)
    pair_mask = torch.zeros(batch_size, max_pairs, dtype=torch.bool)
    
    task_ids = []
    
    for i, task in enumerate(tasks):
        task_ids.append(task.get('task_id', str(i)))
        
        # Training pairs
        for j, pair in enumerate(task.get('train', [])):
            inp = pair.get('input', [])
            out = pair.get('output', [])
            
            for r, row in enumerate(inp):
                for c, val in enumerate(row):
                    if r < effective_max and c < effective_max:
                        input_grids[i, j, r, c] = val
            
            for r, row in enumerate(out):
                for c, val in enumerate(row):
                    if r < effective_max and c < effective_max:
                        output_grids[i, j, r, c] = val
            
            pair_mask[i, j] = True
        
        # Test pair (first one)
        if task.get('test'):
            test_pair = task['test'][0]
            inp = test_pair.get('input', [])
            out = test_pair.get('output', [])
            
            for r, row in enumerate(inp):
                for c, val in enumerate(row):
                    if r < effective_max and c < effective_max:
                        test_inputs[i, r, c] = val
            
            for r, row in enumerate(out):
                for c, val in enumerate(row):
                    if r < effective_max and c < effective_max:
                        test_outputs[i, r, c] = val
    
    return {
        'input_grids': input_grids.to(device),
        'output_grids': output_grids.to(device),
        'test_inputs': test_inputs.to(device),
        'test_outputs': test_outputs.to(device),
        'pair_mask': pair_mask.to(device),
        'task_ids': task_ids,
    }


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_attention_maps(attention_maps, centroids, save_path: str):
    """Visualize attention maps and centroids."""
    B, K, H, W = attention_maps.shape
    
    fig, axes = plt.subplots(1, K, figsize=(3 * K, 3))
    if K == 1:
        axes = [axes]
    
    for k in range(K):
        attn = attention_maps[0, k].cpu().numpy()
        axes[k].imshow(attn, cmap='hot', vmin=0)
        
        # Mark centroid
        row, col = centroids[0, k].cpu().numpy()
        axes[k].plot(col, row, 'g+', markersize=15, markeredgewidth=2)
        
        axes[k].set_title(f'Clue {k}: max={attn.max():.4f}')
        axes[k].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Saved attention visualization to: {save_path}")


# =============================================================================
# MAIN DIAGNOSTIC
# =============================================================================

def run_diagnostics():
    """Main diagnostic function."""
    print("=" * 70)
    print("DSC AND CENTROID COLLAPSE DIAGNOSTIC")
    print("=" * 70)
    
    # Paths
    checkpoint_path = "checkpoints/rlan_stable_ablation/warmup3.pt"
    data_dir = "data"
    
    # Check paths
    if not Path(checkpoint_path).exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        # Try alternate paths
        alt_paths = [
            "checkpoints/warmup3.pt",
            "checkpoints/rlan_stable_merged/latest.pt",
        ]
        for alt in alt_paths:
            if Path(alt).exists():
                checkpoint_path = alt
                print(f"  Using alternate: {checkpoint_path}")
                break
        else:
            print("  No checkpoint found, creating random model for testing")
            checkpoint_path = None
    
    # Load model
    if checkpoint_path:
        model, config = load_model_from_checkpoint(checkpoint_path, DEVICE)
    else:
        from sci_arc.models.rlan import RLAN
        config = {
            'encoder': {'hidden_dim': 128, 'num_layers': 3},
            'solver': {'hidden_dim': 128, 'num_solver_steps': 7},
            'context_encoder': {'enabled': True, 'hidden_dim': 128},
            'dsc': {'enabled': True, 'hidden_dim': 128, 'num_clues': 7},
            'msre': {'enabled': True, 'hidden_dim': 128},
        }
        model = RLAN(config).to(DEVICE)
        model.eval()
    
    print(f"\nModel type: {type(model).__name__}")
    print(f"DSC enabled: {hasattr(model, 'dsc') and model.dsc is not None}")
    
    # Check DSC module
    if hasattr(model, 'dsc') and model.dsc is not None:
        dsc = model.dsc
        print(f"\nDSC Configuration:")
        print(f"  hidden_dim: {dsc.hidden_dim}")
        print(f"  max_clues: {dsc.max_clues}")
        print(f"  context_dim: {dsc.context_dim}")
        
        # Check clue query initialization
        queries = dsc.clue_queries
        print(f"\nClue Query Analysis:")
        print(f"  Shape: {queries.shape}")
        print(f"  Norm per query: {queries.norm(dim=-1).tolist()}")
        
        # Pairwise similarity
        q_norm = F.normalize(queries, dim=-1)
        sim = q_norm @ q_norm.t()
        off_diag = sim[~torch.eye(sim.shape[0], dtype=bool)]
        print(f"  Pairwise cosine similarity (off-diag):")
        print(f"    mean={off_diag.mean():.4f}, max={off_diag.max():.4f}, min={off_diag.min():.4f}")
        
        if off_diag.mean() > 0.5:
            print(f"  ⚠️ QUERIES TOO SIMILAR - may cause collapse!")
    else:
        print("❌ No DSC module found!")
        return
    
    # Load tasks
    print("\n" + "=" * 70)
    print("LOADING TEST DATA")
    print("=" * 70)
    
    tasks = load_arc_tasks(data_dir, max_tasks=3)
    if not tasks:
        print("❌ No tasks found, creating synthetic test data")
        # Create synthetic tasks
        tasks = []
        for i in range(2):
            h, w = 8, 8
            train_pairs = []
            for _ in range(3):
                inp = [[np.random.randint(0, 10) for _ in range(w)] for _ in range(h)]
                out = [[np.random.randint(0, 10) for _ in range(w)] for _ in range(h)]
                train_pairs.append({'input': inp, 'output': out})
            test_pair = [{'input': inp, 'output': out}]
            tasks.append({
                'task_id': f'synthetic_{i}',
                'train': train_pairs,
                'test': test_pair,
            })
    
    print(f"Loaded {len(tasks)} tasks")
    
    # Prepare batch
    batch = prepare_batch(tasks, max_size=30, device=DEVICE)
    
    # Run inference
    print("\n" + "=" * 70)
    print("RUNNING INFERENCE (FORWARD PASS)")
    print("=" * 70)
    
    with torch.no_grad():
        # Get encoded features
        train_inputs = batch['input_grids']
        train_outputs = batch['output_grids']
        test_input = batch['test_inputs']  # (B, H, W) - already correct for model.forward
        pair_mask = batch['pair_mask']
        
        print(f"\nInput shapes:")
        print(f"  train_inputs: {train_inputs.shape}")
        print(f"  train_outputs: {train_outputs.shape}")
        print(f"  test_input: {test_input.shape}")
        print(f"  pair_mask: {pair_mask.shape}")
        
        # Full forward pass - RLAN.forward signature:
        # forward(input_grid, train_inputs, train_outputs, pair_mask, ...)
        # where input_grid is the test input (B, H, W)
        print("\n--- Full Model Forward ---")
        outputs = model(
            input_grid=test_input,  # (B, H, W)
            train_inputs=train_inputs,  # (B, N, H, W) 
            train_outputs=train_outputs,  # (B, N, H, W)
            pair_mask=pair_mask,
            return_intermediates=True,
        )
        
        print(f"\nOutput keys: {list(outputs.keys())}")
        
        # Check DSC outputs
        if 'centroids' in outputs:
            centroids = outputs['centroids']
            print(f"\nCentroids shape: {centroids.shape}")
            print(f"Centroids (batch 0):")
            for k in range(centroids.shape[1]):
                r, c = centroids[0, k, 0].item(), centroids[0, k, 1].item()
                print(f"  Clue {k}: ({r:.2f}, {c:.2f})")
            
            # Compute spread
            centroid_mean = centroids.mean(dim=1, keepdim=True)
            spread = ((centroids - centroid_mean) ** 2).sum(dim=-1).sqrt().mean()
            print(f"\nCentroid spread: {spread.item():.4f}")
            
            if spread < 0.5:
                print(f"⚠️ CRITICAL: Centroid spread < 0.5 indicates COLLAPSE!")
        
        if 'attention_maps' in outputs:
            attn_maps = outputs['attention_maps']
            print(f"\nAttention maps shape: {attn_maps.shape}")
            for k in range(attn_maps.shape[1]):
                attn_k = attn_maps[:, k]
                print(f"  Clue {k}: max={attn_k.max():.6f}, mean={attn_k.mean():.6f}")
            
            # Save visualization
            if 'centroids' in outputs:
                visualize_attention_maps(
                    attn_maps, outputs['centroids'],
                    "logs/dsc_attention_diagnostic.png"
                )
        
        # Detailed DSC trace
        print("\n" + "=" * 70)
        print("DETAILED DSC TRACE")
        print("=" * 70)
        
        # Get features from encoder
        if hasattr(model, 'encoder'):
            # Get test features
            test_features = model.encoder(test_input)
            print(f"\nTest features shape: {test_features.shape}")
            
            # Get context if available
            task_context = None
            if hasattr(model, 'context_encoder') and model.context_encoder is not None:
                try:
                    ctx_out = model.context_encoder(train_inputs, train_outputs, pair_mask)
                    if isinstance(ctx_out, dict):
                        task_context = ctx_out.get('task_embedding', ctx_out.get('context'))
                    else:
                        task_context = ctx_out
                    print(f"Task context shape: {task_context.shape if task_context is not None else 'None'}")
                except Exception as e:
                    print(f"  Context encoding failed: {e}")
            
            # Trace DSC with detailed logging
            temperature = getattr(model, 'temperature', 1.0)
            print(f"\nDSC temperature: {temperature}")
            
            centroids, attn_maps, stop_logits = trace_dsc_forward(
                model.dsc, test_features, temperature, task_context
            )
            
            # Save traced attention
            visualize_attention_maps(
                attn_maps, centroids,
                "logs/dsc_attention_traced.png"
            )
    
    # Print summary
    LOGGER.print_summary()
    
    # Final diagnosis
    print("\n" + "=" * 70)
    print("DIAGNOSIS")
    print("=" * 70)
    
    print("""
ROOT CAUSE ANALYSIS:

1. SPATIAL FEATURE UNIFORMITY:
   - If encoder outputs spatially uniform features, DSC cannot differentiate positions
   - Check: feature_position_variance should be >> 0.01
   
2. QUERY-KEY ALIGNMENT:
   - If q·k produces constant scores across positions, attention is uniform
   - This happens when:
     a) Keys (features) have no spatial variation
     b) Queries are orthogonal to the dimension of variation
     c) Keys/queries collapse to same representation
   
3. TEMPERATURE TOO HIGH:
   - High temperature (>1.0) makes softmax too smooth
   - At temp=1.0 and grid 10x10: uniform attention = 0.01 per pixel
   - Sharp attention needs temp << 1.0 or strong score variation
   
4. PROGRESSIVE MASKING NOT WORKING:
   - If first clue's attention is uniform, cumulative_mask ≈ 1 everywhere
   - Next clue sees same uniform landscape
   - All clues converge to center-of-mass = grid center
   
5. STOP PREDICTOR COUPLING:
   - If entropy is always high (diffuse attention), confidence is always low
   - Stop predictor never gets signal to differentiate clues
   
RECOMMENDED FIXES:

1. Increase feature spatial variation:
   - Add positional encoding to encoder output
   - Use deeper/wider encoder
   
2. Initialize clue queries with MORE diversity:
   - Current scale 0.3 may not be enough
   - Try orthogonal initialization
   
3. Lower temperature:
   - Try temperature = 0.5 or 0.1
   - Or use learnable temperature per clue
   
4. Add explicit attention sharpening loss:
   - Penalize high entropy attention
   - Reward peaked attention maps
   
5. Pre-train DSC on synthetic anchor tasks:
   - Tasks where ground-truth anchors are known
   - Force DSC to find them before joint training
""")
    
    return True


if __name__ == "__main__":
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    
    success = run_diagnostics()
    sys.exit(0 if success else 1)
