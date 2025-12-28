"""
Memory Diagnostic Script
========================
Measures GPU memory at each stage of training initialization
to identify where the 7GB memory increase is coming from.

Run: python scripts/diagnose_memory.py
"""

import torch
import gc
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_memory_stats():
    """Get current GPU memory stats in GB."""
    if not torch.cuda.is_available():
        return {"allocated": 0, "reserved": 0, "free": 0}
    
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    free = total - reserved
    
    return {
        "allocated": allocated,
        "reserved": reserved,
        "free": free,
        "total": total
    }

def print_memory(label: str):
    """Print memory stats with label."""
    stats = get_memory_stats()
    print(f"\n{'='*60}")
    print(f"MEMORY @ {label}")
    print(f"{'='*60}")
    print(f"  Allocated: {stats['allocated']:.2f} GB")
    print(f"  Reserved:  {stats['reserved']:.2f} GB")
    print(f"  Free:      {stats['free']:.2f} GB")
    print(f"  Total:     {stats['total']:.2f} GB")
    return stats['reserved']

def main():
    import yaml
    from pathlib import Path
    
    print("="*60)
    print("MEMORY DIAGNOSTIC - Finding the 7GB increase")
    print("="*60)
    
    # Clear everything
    gc.collect()
    torch.cuda.empty_cache()
    
    baseline = print_memory("BASELINE (before anything)")
    
    # Load config
    config_path = Path("configs/rlan_stable.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_cfg = config['model']
    data_cfg = config['data']
    training_cfg = config['training']
    
    print_memory("After loading config")
    
    # Import modules
    print("\n--- Importing modules ---")
    from sci_arc.models.rlan import RLAN
    from sci_arc.data.dataset import ARCDataset
    
    print_memory("After imports")
    
    # Create model
    print("\n--- Creating model ---")
    device = torch.device('cuda')
    model = RLAN(
        num_colors=model_cfg['num_colors'],
        hidden_dim=model_cfg['hidden_dim'],
        num_solver_steps=model_cfg['num_solver_steps'],
        num_heads=model_cfg['num_heads'],
        dropout=model_cfg['dropout'],
        use_hyperlora=model_cfg.get('use_hyperlora', True),
        hyperlora_rank=model_cfg.get('hyperlora_rank', 16),
        use_context_encoding=model_cfg.get('use_context_encoding', True),
        max_pairs=model_cfg.get('max_context_pairs', 10),
        # HPM config
        use_hpm=model_cfg.get('use_hpm', False),
        hpm_num_prototypes=model_cfg.get('hpm_num_prototypes', 128),
        hpm_memory_dim=model_cfg.get('hpm_memory_dim', 256),
    )
    
    print_memory("After model creation (CPU)")
    
    # Move to GPU
    print("\n--- Moving model to GPU ---")
    model = model.to(device)
    
    mem_after_model = print_memory("After model.to(cuda)")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Total params: {total_params:,}")
    print(f"  Trainable:    {trainable_params:,}")
    print(f"  Expected size: ~{trainable_params * 4 / 1024**3:.2f} GB (float32)")
    
    # Create optimizer
    print("\n--- Creating optimizer ---")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_cfg['learning_rate'],
        weight_decay=training_cfg.get('weight_decay', 0.01)
    )
    
    mem_after_optim = print_memory("After optimizer creation")
    
    # Create dummy batch to test forward pass
    print("\n--- Testing forward pass with dummy batch ---")
    batch_size = training_cfg['batch_size']
    max_size = model_cfg['max_grid_size']
    
    # Create dummy tensors
    dummy_input = torch.randint(0, 10, (batch_size, max_size, max_size), device=device)
    dummy_target = torch.randint(0, 10, (batch_size, max_size, max_size), device=device)
    dummy_context_inputs = torch.randint(0, 10, (batch_size, 3, max_size, max_size), device=device)
    dummy_context_outputs = torch.randint(0, 10, (batch_size, 3, max_size, max_size), device=device)
    dummy_context_mask = torch.ones(batch_size, 3, dtype=torch.bool, device=device)
    
    print(f"  Batch size: {batch_size}")
    print(f"  Grid size: {max_size}x{max_size}")
    print(f"  Context pairs: 3")
    
    print_memory("After creating dummy tensors")
    
    # Forward pass
    model.train()
    with torch.amp.autocast('cuda', enabled=True):
        outputs = model(
            test_input=dummy_input,
            context_inputs=dummy_context_inputs,
            context_outputs=dummy_context_outputs,
            context_mask=dummy_context_mask,
        )
    
    mem_after_forward = print_memory("After forward pass (AMP)")
    
    # Compute loss
    import torch.nn.functional as F
    logits = outputs['logits']  # [B, H, W, C]
    logits_flat = logits.reshape(-1, 10)
    target_flat = dummy_target.reshape(-1)
    loss = F.cross_entropy(logits_flat, target_flat)
    
    print_memory("After loss computation")
    
    # Backward pass
    loss.backward()
    
    mem_after_backward = print_memory("After backward pass")
    
    # Print summary
    print("\n" + "="*60)
    print("MEMORY SUMMARY")
    print("="*60)
    print(f"  Baseline:          {baseline:.2f} GB")
    print(f"  Model on GPU:      {mem_after_model:.2f} GB (+{mem_after_model - baseline:.2f} GB)")
    print(f"  + Optimizer:       {mem_after_optim:.2f} GB (+{mem_after_optim - mem_after_model:.2f} GB)")
    print(f"  + Forward pass:    {mem_after_forward:.2f} GB (+{mem_after_forward - mem_after_optim:.2f} GB)")
    print(f"  + Backward pass:   {mem_after_backward:.2f} GB (+{mem_after_backward - mem_after_forward:.2f} GB)")
    print(f"  TOTAL:             {mem_after_backward:.2f} GB")
    print("="*60)
    
    # Check for any surprise allocations
    print("\n--- Checking model submodules ---")
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and module.weight is not None:
            size_mb = module.weight.numel() * 4 / 1024**2
            if size_mb > 50:  # Only show modules > 50MB
                print(f"  {name}: {size_mb:.1f} MB")
    
    print("\nDone! Compare these numbers with the old commit to find the increase.")

if __name__ == "__main__":
    main()
