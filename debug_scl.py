#!/usr/bin/env python
"""Debug script to trace why SCL loss is constant.

Run this on the server with: python debug_scl.py
"""

import torch
import torch.nn.functional as F
import yaml
from pathlib import Path

# Ensure we can find the package
import sys
sys.path.insert(0, str(Path(__file__).parent))

print("="*60)
print("SCL Debug Script")
print("="*60)

# Load config
config_path = Path("configs/default.yaml")
with open(config_path) as f:
    config = yaml.safe_load(f)

# Setup
from sci_arc.models import SCIARC
from sci_arc.data import create_dataloader
from sci_arc.config import SCIARCConfig

model_cfg = config['model']
data_cfg = config['data']

# Create model
model_config = SCIARCConfig(
    hidden_dim=model_cfg['hidden_dim'],
    num_colors=model_cfg['num_colors'],
    num_structure_slots=model_cfg['num_structure_slots'],
    num_content_slots=model_cfg['num_content_slots'],
    num_heads=model_cfg['num_heads'],
    H_cycles=model_cfg['H_cycles'],
    L_cycles=model_cfg['L_cycles'],
    dropout=model_cfg['dropout'],
)
model = SCIARC(model_config)
model.eval()  # Use eval mode for determinism

print(f"\nModel created with {sum(p.numel() for p in model.parameters()):,} parameters")

# Create dataloader with same settings as training
print(f"\nCreating dataloader...")
dataloader = create_dataloader(
    data_dir=data_cfg['arc_dir'],
    split='training',
    batch_size=192,
    num_workers=0,  # Single-threaded for debugging
    shuffle=True,  # Same as training
    augment=True,
    max_grid_size=config['model'].get('max_grid_size', 30),
    cache_samples=True,
    cache_augmentations=8,
    use_augment_family=True,
)

print(f"Dataset size: {len(dataloader.dataset)}")

# Analyze first batch
print("\n" + "="*60)
print("Analyzing first batch...")
print("="*60)

for batch in dataloader:
    print(f"\n1. TRANSFORM FAMILIES:")
    transform_families = batch['transform_families']
    print(f"   Shape: {transform_families.shape}")
    print(f"   Values: {transform_families.tolist()[:30]}...")
    unique, counts = torch.unique(transform_families, return_counts=True)
    print(f"   Unique values: {unique.tolist()}")
    print(f"   Counts: {counts.tolist()}")
    
    if len(unique) == 1:
        print("\n   *** WARNING: ALL SAMPLES HAVE SAME TRANSFORM_FAMILY! ***")
        print("   This is the BUG! SCL will be constant because all samples")
        print("   are considered positive pairs!")
    
    # Forward pass
    print(f"\n2. MODEL FORWARD PASS:")
    with torch.no_grad():
        outputs = model.forward_training(
            input_grids=batch['input_grids'],
            output_grids=batch['output_grids'],
            test_input=batch['test_inputs'],
            test_output=batch['test_outputs'],
        )
    
    z_struct = outputs['z_struct']  # [B, K, D]
    print(f"   z_struct shape: {z_struct.shape}")
    print(f"   z_struct mean: {z_struct.mean().item():.6f}")
    print(f"   z_struct std: {z_struct.std().item():.6f}")
    print(f"   z_struct min: {z_struct.min().item():.6f}")
    print(f"   z_struct max: {z_struct.max().item():.6f}")
    
    # Pool and normalize
    z = z_struct.mean(dim=1)  # [B, D]
    z_norm = F.normalize(z, dim=-1)
    
    print(f"\n3. NORMALIZED EMBEDDINGS:")
    print(f"   z_norm shape: {z_norm.shape}")
    print(f"   z_norm mean: {z_norm.mean().item():.6f}")
    print(f"   z_norm std: {z_norm.std().item():.6f}")
    
    # Check if embeddings are all identical
    z_diff = (z_norm - z_norm[0:1]).abs().sum(dim=-1)
    print(f"   Diff from first sample: min={z_diff[1:].min():.6f}, max={z_diff.max():.6f}")
    
    if z_diff.max() < 0.01:
        print("\n   *** WARNING: ALL EMBEDDINGS ARE NEARLY IDENTICAL! ***")
        print("   This explains constant SCL - model produces same output for all inputs!")
    
    # Compute full similarity matrix
    print(f"\n4. SIMILARITY ANALYSIS:")
    sim = torch.mm(z_norm, z_norm.t())  # [B, B]
    print(f"   Self-similarity (diagonal): {sim.diag().mean():.4f} (should be 1.0)")
    
    # Off-diagonal similarities
    mask_diag = torch.eye(192).bool()
    off_diag = sim[~mask_diag]
    print(f"   Off-diagonal similarities:")
    print(f"      Mean: {off_diag.mean():.4f}")
    print(f"      Std: {off_diag.std():.4f}")
    print(f"      Min: {off_diag.min():.4f}")
    print(f"      Max: {off_diag.max():.4f}")
    
    # Check positive pairs (same transform_family)
    labels_equal = transform_families.unsqueeze(0) == transform_families.unsqueeze(1)
    pos_mask = labels_equal & ~mask_diag
    neg_mask = ~labels_equal
    
    if pos_mask.sum() > 0:
        pos_sims = sim[pos_mask]
        print(f"\n   Positive pair similarities (same family):")
        print(f"      Count: {pos_mask.sum().item()}")
        print(f"      Mean: {pos_sims.mean():.4f}")
        print(f"      Std: {pos_sims.std():.4f}")
    else:
        print(f"\n   *** NO POSITIVE PAIRS! ***")
    
    if neg_mask.sum() > 0:
        neg_sims = sim[neg_mask]
        print(f"\n   Negative pair similarities (different family):")
        print(f"      Count: {neg_mask.sum().item()}")
        print(f"      Mean: {neg_sims.mean():.4f}")
        print(f"      Std: {neg_sims.std():.4f}")
    
    # Compute SCL loss manually
    print(f"\n5. SCL LOSS COMPUTATION:")
    temperature = 0.1
    sim_scaled = sim / temperature
    
    sim_masked = sim_scaled.masked_fill(mask_diag, float('-inf'))
    log_sum_exp = torch.logsumexp(sim_masked, dim=1)
    
    print(f"   Temperature: {temperature}")
    print(f"   log_sum_exp mean: {log_sum_exp.mean():.4f}")
    print(f"   log_sum_exp std: {log_sum_exp.std():.4f}")
    print(f"   Expected if random: log(191) = {torch.log(torch.tensor(191.0)):.4f}")
    
    # Compute actual SCL loss
    if pos_mask.sum() > 0:
        loss_matrix = -sim_scaled + log_sum_exp.unsqueeze(1)
        pos_counts = pos_mask.float().sum(dim=1)
        has_positives = pos_counts > 0
        num_valid_anchors = has_positives.float().sum()
        
        pos_loss_sum = (loss_matrix * pos_mask.float()).sum(dim=1)
        per_anchor_loss = torch.where(
            has_positives,
            pos_loss_sum / pos_counts.clamp(min=1),
            torch.zeros_like(pos_loss_sum)
        )
        
        scl_loss = per_anchor_loss.sum() / num_valid_anchors.clamp(min=1)
        
        print(f"\n   Computed SCL loss: {scl_loss.item():.4f}")
        print(f"   Number of valid anchors: {num_valid_anchors.item():.0f}")
    
    # Only process first batch
    break

print("\n" + "="*60)
print("DIAGNOSIS")
print("="*60)
print("""
If SCL = ~5.25 and is constant, check:

1. Transform families: Should be 0-7 with roughly equal distribution
   - If all same value: Bug in cache building
   - If all different: Bug in augmentation or family assignment

2. Embeddings: Should vary across samples
   - If identical: Model architecture issue or initialization
   - If random/orthogonal: Normal at start, should change with training

3. Positive pairs: Should have many positive pairs (same family)
   - If zero: All families different (old bug)
   - If all: All families same (new bug?)

4. For SCL to decrease during training:
   - Positive similarities should INCREASE
   - Negative similarities should stay low or DECREASE
""")

