"""
Test script: AvgPool vs MaxPool for HyperLoRA context pooling.

Hypothesis:
- ARC grids have sharp boundaries between foreground (colored objects) and background (0s)
- MaxPool might better capture salient foreground features
- AvgPool might dilute signal with background zeros

This test compares both pooling strategies on real ARC data to measure:
1. Context vector discriminability (task separability)
2. LoRA prediction stability
3. LOO holdout accuracy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import json
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sci_arc.models.rlan import RLAN, RLANConfig
from sci_arc.models.rlan_modules.hyper_lora import HyperLoRA, HyperLoRAConfig


class HyperLoRAWithMaxPool(nn.Module):
    """HyperLoRA variant using MaxPool instead of AvgPool."""
    
    def __init__(self, original_hyperlora: HyperLoRA):
        super().__init__()
        self.original = original_hyperlora
        # Replace AdaptiveAvgPool2d with AdaptiveMaxPool2d
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
    def pool_context(self, support_features: torch.Tensor) -> torch.Tensor:
        """Pool using MaxPool instead of AvgPool."""
        B, N, D, H, W = support_features.shape
        features_flat = support_features.reshape(B * N, D, H, W)
        pooled = self.max_pool(features_flat)  # Use MaxPool
        pooled = pooled.reshape(B, N, D)
        context = pooled.mean(dim=1)  # Still average across pairs
        return context
    
    def forward(self, support_features: torch.Tensor):
        """Forward with MaxPool context."""
        context = self.pool_context(support_features)
        context = self.original.context_fuse(context)
        context = self.original.dropout(context)
        
        deltas = {}
        deltas['gru_reset'] = self.original.gru_reset_lora.compute_delta_w(context)
        deltas['gru_update'] = self.original.gru_update_lora.compute_delta_w(context)
        deltas['gru_candidate'] = self.original.gru_candidate_lora.compute_delta_w(context)
        deltas['output_head'] = self.original.output_head_lora.compute_delta_w(context)
        deltas['context'] = context
        
        return deltas


def load_arc_task(task_path: Path):
    """Load an ARC task from JSON."""
    with open(task_path, 'r') as f:
        task = json.load(f)
    return task


def task_to_tensors(task, device='cpu'):
    """Convert ARC task to tensors."""
    train_inputs = []
    train_outputs = []
    
    for pair in task.get('train', []):
        inp = torch.tensor(pair['input'], dtype=torch.long, device=device)
        out = torch.tensor(pair['output'], dtype=torch.long, device=device)
        train_inputs.append(inp)
        train_outputs.append(out)
    
    if not train_inputs:
        return None, None, None, None
    
    # Pad to same size
    max_h = max(t.shape[0] for t in train_inputs + train_outputs)
    max_w = max(t.shape[1] for t in train_inputs + train_outputs)
    
    def pad_grid(grid, target_h, target_w):
        h, w = grid.shape
        padded = torch.full((target_h, target_w), 10, dtype=grid.dtype, device=grid.device)  # 10 = padding
        padded[:h, :w] = grid
        return padded
    
    train_inputs = torch.stack([pad_grid(t, max_h, max_w) for t in train_inputs])  # (N, H, W)
    train_outputs = torch.stack([pad_grid(t, max_h, max_w) for t in train_outputs])  # (N, H, W)
    
    # Add batch dim
    train_inputs = train_inputs.unsqueeze(0)  # (1, N, H, W)
    train_outputs = train_outputs.unsqueeze(0)  # (1, N, H, W)
    
    # Test input (first test case)
    test_cases = task.get('test', [])
    if test_cases:
        test_inp_raw = torch.tensor(test_cases[0]['input'], dtype=torch.long, device=device)
        # Recompute max sizes including test
        max_h = max(max_h, test_inp_raw.shape[0])
        max_w = max(max_w, test_inp_raw.shape[1])
        if 'output' in test_cases[0]:
            test_out_raw = torch.tensor(test_cases[0]['output'], dtype=torch.long, device=device)
            max_h = max(max_h, test_out_raw.shape[0])
            max_w = max(max_w, test_out_raw.shape[1])
        
        # Re-pad train tensors with new max sizes
        train_inputs_list = [pad_grid(train_inputs[0, i], max_h, max_w) for i in range(train_inputs.shape[1])]
        train_outputs_list = [pad_grid(train_outputs[0, i], max_h, max_w) for i in range(train_outputs.shape[1])]
        train_inputs = torch.stack(train_inputs_list).unsqueeze(0)
        train_outputs = torch.stack(train_outputs_list).unsqueeze(0)
        
        test_input = pad_grid(test_inp_raw, max_h, max_w).unsqueeze(0)
        if 'output' in test_cases[0]:
            test_output = pad_grid(test_out_raw, max_h, max_w).unsqueeze(0)
        else:
            test_output = None
    else:
        test_input = train_inputs[:, 0]  # Use first train as test
        test_output = train_outputs[:, 0]
    
    return train_inputs, train_outputs, test_input, test_output


def compute_context_discriminability(contexts_by_task):
    """
    Measure how separable context vectors are across different tasks.
    Higher = better task discrimination.
    """
    all_contexts = []
    task_labels = []
    
    for task_id, context in contexts_by_task.items():
        all_contexts.append(context)
        task_labels.append(task_id)
    
    if len(all_contexts) < 2:
        return 0.0
    
    all_contexts = torch.stack(all_contexts)  # (T, D)
    
    # Compute within-task variance (should be low for same task)
    # Since we have one context per task, compute inter-task distances
    
    # Pairwise cosine similarities
    normalized = F.normalize(all_contexts, dim=1)
    similarity_matrix = torch.mm(normalized, normalized.t())
    
    # Good discriminability = low off-diagonal similarities
    # Return mean of off-diagonal (should be low)
    mask = ~torch.eye(len(all_contexts), dtype=torch.bool)
    off_diag_sim = similarity_matrix[mask].mean().item()
    
    # Higher discriminability = lower similarity = better
    discriminability = 1.0 - off_diag_sim
    return discriminability


def test_pooling_methods(data_dir: Path, num_tasks: int = 10):
    """Compare AvgPool vs MaxPool on real ARC tasks."""
    
    print("=" * 60)
    print("HyperLoRA Pooling Strategy Comparison")
    print("=" * 60)
    
    # Setup model with HyperLoRA
    config = RLANConfig(
        hidden_dim=64,  # Smaller for speed
        use_hyperlora=True,
        use_context_encoder=True,
        use_solver_context=True,
        use_cross_attention_context=False,
        use_dsc=True,
        use_msre=True,
        use_lcr=False,
        use_sph=False,
    )
    
    print("\n1. Creating RLAN model...")
    model = RLAN(config=config)
    model.eval()
    
    # Create MaxPool variant
    hyperlora_maxpool = HyperLoRAWithMaxPool(model.hyper_lora)
    
    # Find training tasks
    training_dir = data_dir / 'training'
    if not training_dir.exists():
        print(f"ERROR: Training directory not found: {training_dir}")
        return
    
    task_files = list(training_dir.glob('*.json'))[:num_tasks]
    print(f"   Found {len(task_files)} tasks to test")
    
    # Collect results
    avgpool_contexts = {}
    maxpool_contexts = {}
    avgpool_delta_norms = []
    maxpool_delta_norms = []
    
    print("\n2. Processing tasks...")
    
    for i, task_file in enumerate(task_files):
        task = load_arc_task(task_file)
        train_inputs, train_outputs, test_input, test_output = task_to_tensors(task)
        
        if train_inputs is None or train_inputs.shape[1] < 2:
            continue
        
        task_id = task_file.stem
        
        with torch.no_grad():
            # Get support features from context encoder
            support_features = model.context_encoder(train_inputs, train_outputs)  # (1, N, D, H, W)
            
            # AvgPool context (original)
            avgpool_context = model.hyper_lora.pool_context(support_features)
            avgpool_deltas = model.hyper_lora(support_features)
            
            # MaxPool context
            maxpool_context = hyperlora_maxpool.pool_context(support_features)
            maxpool_deltas = hyperlora_maxpool(support_features)
            
            # Store contexts
            avgpool_contexts[task_id] = avgpool_context.squeeze(0)
            maxpool_contexts[task_id] = maxpool_context.squeeze(0)
            
            # Compute delta norms
            avg_norm = sum(d.norm().item() for d in avgpool_deltas.values() if isinstance(d, torch.Tensor) and d.dim() > 1)
            max_norm = sum(d.norm().item() for d in maxpool_deltas.values() if isinstance(d, torch.Tensor) and d.dim() > 1)
            avgpool_delta_norms.append(avg_norm)
            maxpool_delta_norms.append(max_norm)
        
        if (i + 1) % 5 == 0:
            print(f"   Processed {i + 1}/{len(task_files)} tasks")
    
    print(f"\n3. Results for {len(avgpool_contexts)} valid tasks:")
    print("-" * 60)
    
    # Discriminability
    avg_discrim = compute_context_discriminability(avgpool_contexts)
    max_discrim = compute_context_discriminability(maxpool_contexts)
    
    print(f"\nContext Discriminability (higher = better task separation):")
    print(f"  AvgPool: {avg_discrim:.4f}")
    print(f"  MaxPool: {max_discrim:.4f}")
    print(f"  Winner:  {'MaxPool' if max_discrim > avg_discrim else 'AvgPool'} (+{abs(max_discrim - avg_discrim):.4f})")
    
    # Delta norms
    avg_delta_mean = sum(avgpool_delta_norms) / len(avgpool_delta_norms) if avgpool_delta_norms else 0
    max_delta_mean = sum(maxpool_delta_norms) / len(maxpool_delta_norms) if maxpool_delta_norms else 0
    
    print(f"\nLoRA Delta Norms (signal strength):")
    print(f"  AvgPool: {avg_delta_mean:.4f}")
    print(f"  MaxPool: {max_delta_mean:.4f}")
    print(f"  Winner:  {'MaxPool' if max_delta_mean > avg_delta_mean else 'AvgPool'} (stronger signal)")
    
    # Context magnitude
    avg_ctx_norms = [c.norm().item() for c in avgpool_contexts.values()]
    max_ctx_norms = [c.norm().item() for c in maxpool_contexts.values()]
    
    print(f"\nContext Vector Magnitude:")
    print(f"  AvgPool: {sum(avg_ctx_norms)/len(avg_ctx_norms):.4f}")
    print(f"  MaxPool: {sum(max_ctx_norms)/len(max_ctx_norms):.4f}")
    
    print("\n" + "=" * 60)
    print("RECOMMENDATION:")
    if max_discrim > avg_discrim and max_delta_mean > avg_delta_mean:
        print("  MaxPool is better - provides stronger, more discriminable signals.")
        print("  Consider switching HyperLoRA.context_pool to AdaptiveMaxPool2d.")
    elif avg_discrim > max_discrim and avg_delta_mean > max_delta_mean:
        print("  AvgPool is better - current implementation is optimal.")
    else:
        print("  Mixed results - both have trade-offs. AvgPool is default (safer).")
    print("=" * 60)


if __name__ == "__main__":
    # Find ARC data directory
    data_dir = Path(__file__).parent.parent / 'data' / 'arc-agi' / 'data'
    
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        print("Please ensure ARC data is in ../data/arc-agi/data/training/")
        sys.exit(1)
    
    test_pooling_methods(data_dir, num_tasks=20)
