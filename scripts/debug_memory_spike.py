"""
Memory Spike Diagnostic Script

This script isolates each component to find what's causing the 7GB memory increase
from 19GB to 26GB during epoch 0 training.

Run: python scripts/debug_memory_spike.py
"""

import torch
import gc
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def get_gpu_memory():
    """Get current GPU memory in GB."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        return allocated, reserved
    return 0, 0


def print_memory(label: str):
    """Print current GPU memory usage."""
    allocated, reserved = get_gpu_memory()
    print(f"[{label}] Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")


def clear_memory():
    """Force clear GPU memory."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def main():
    print("=" * 70)
    print("MEMORY SPIKE DIAGNOSTIC")
    print("=" * 70)
    
    # Load config
    import yaml
    config_path = project_root / "configs" / "rlan_stable.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    
    clear_memory()
    print_memory("BASELINE (empty GPU)")
    
    # =========================================================================
    # STEP 1: Load Dataset (CPU only, should not affect GPU)
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 1: Loading Dataset")
    print("=" * 70)
    
    from sci_arc.data.dataset import ARCDataset
    
    data_cfg = config['data']
    dataset = ARCDataset(
        data_path=data_cfg['train_path'],
        max_size=data_cfg.get('max_grid_size', 30),
        augment=data_cfg.get('augment', True),
        color_permutation=data_cfg.get('color_permutation', True),
        color_permutation_prob=data_cfg.get('color_permutation_prob', 0.5),
        translational_augment=data_cfg.get('translational_augment', True),
        cache_samples=data_cfg.get('cache_samples', False),
        num_cached_samples=data_cfg.get('num_cached_samples', 32000),
        cache_path=data_cfg.get('cache_path', None),
        cache_load_percent=data_cfg.get('cache_load_percent', 100.0),
    )
    
    print(f"Dataset loaded: {len(dataset)} samples")
    print_memory("After dataset load")
    
    # =========================================================================
    # STEP 2: Create Model (this is where GPU memory starts)
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: Creating Model")
    print("=" * 70)
    
    from sci_arc.models.rlan import RLAN
    
    model_cfg = config['model']
    model = RLAN(
        hidden_dim=model_cfg['hidden_dim'],
        num_layers=model_cfg['num_layers'],
        num_heads=model_cfg['num_heads'],
        num_colors=model_cfg['num_colors'],
        max_grid_size=model_cfg['max_grid_size'],
        num_predicates=model_cfg['num_predicates'],
        num_solver_steps=model_cfg['num_solver_steps'],
        dropout=model_cfg.get('dropout', 0.1),
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print_memory("After model creation (CPU)")
    
    # Move to GPU
    model = model.to(device)
    print_memory("After model.to(device)")
    
    # =========================================================================
    # STEP 3: Create Optimizer
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: Creating Optimizer")
    print("=" * 70)
    
    train_cfg = config['training']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg['learning_rate'],
        weight_decay=train_cfg.get('weight_decay', 0.01),
    )
    print_memory("After optimizer creation")
    
    # =========================================================================
    # STEP 4: Check for any extra modules being initialized
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 4: Checking Extra Modules")
    print("=" * 70)
    
    # Check if CrossAttentionInjector is being created
    try:
        from sci_arc.models.cross_attention_injector import CrossAttentionInjector
        print("CrossAttentionInjector module available")
    except ImportError:
        print("CrossAttentionInjector not found")
    
    # Check if HyperLoRA is being created
    try:
        from sci_arc.models.hyper_lora import HyperLoRA
        print("HyperLoRA module available")
    except ImportError:
        print("HyperLoRA not found")
    
    # Check if SolverCrossAttention exists
    try:
        from sci_arc.models.solver_cross_attention import SolverCrossAttention
        print("SolverCrossAttention module available")
    except ImportError:
        print("SolverCrossAttention not found")
    
    print_memory("After module checks")
    
    # =========================================================================
    # STEP 5: Create a single batch and do forward pass
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 5: Single Forward Pass")
    print("=" * 70)
    
    from torch.utils.data import DataLoader
    from sci_arc.data.dataset import arc_collate_fn
    
    batch_size = config['training']['batch_size']
    print(f"Batch size: {batch_size}")
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=arc_collate_fn,
        num_workers=0,
        pin_memory=True,
    )
    
    # Get one batch
    batch = next(iter(dataloader))
    
    # Move batch to GPU
    def move_to_device(data, device):
        if isinstance(data, torch.Tensor):
            return data.to(device, non_blocking=True)
        elif isinstance(data, dict):
            return {k: move_to_device(v, device) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return type(data)(move_to_device(x, device) for x in data)
        return data
    
    batch = move_to_device(batch, device)
    print_memory("After batch to GPU")
    
    # Forward pass
    model.train()
    clear_memory()
    print_memory("Before forward pass")
    
    with torch.cuda.amp.autocast(enabled=train_cfg.get('use_amp', True)):
        outputs = model(batch)
    
    print_memory("After forward pass")
    
    # Compute loss
    loss = outputs.get('loss', outputs.get('total_loss', torch.tensor(0.0)))
    print(f"Loss: {loss.item():.4f}")
    print_memory("After loss computation")
    
    # =========================================================================
    # STEP 6: Backward pass
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 6: Backward Pass")
    print("=" * 70)
    
    scaler = torch.cuda.amp.GradScaler(enabled=train_cfg.get('use_amp', True))
    
    scaler.scale(loss).backward()
    print_memory("After backward pass")
    
    scaler.step(optimizer)
    print_memory("After optimizer step")
    
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    print_memory("After zero_grad")
    
    # =========================================================================
    # STEP 7: Clear and check baseline again
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 7: Cleanup Check")
    print("=" * 70)
    
    del outputs, loss, batch
    clear_memory()
    print_memory("After cleanup (model still in memory)")
    
    # =========================================================================
    # STEP 8: Check for any suspicious global state
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 8: Global State Check")
    print("=" * 70)
    
    # Check if there are any registered hooks
    hook_count = 0
    for name, module in model.named_modules():
        if hasattr(module, '_forward_hooks') and module._forward_hooks:
            hook_count += len(module._forward_hooks)
            print(f"  Forward hooks on {name}: {len(module._forward_hooks)}")
        if hasattr(module, '_backward_hooks') and module._backward_hooks:
            hook_count += len(module._backward_hooks)
            print(f"  Backward hooks on {name}: {len(module._backward_hooks)}")
    print(f"Total hooks: {hook_count}")
    
    # Check CUDA memory summary
    print("\n" + "=" * 70)
    print("CUDA Memory Summary")
    print("=" * 70)
    print(torch.cuda.memory_summary(abbreviated=True))
    
    # =========================================================================
    # STEP 9: Compare with actual training script initialization
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 9: Checking train_rlan.py Imports")
    print("=" * 70)
    
    # Check what train_rlan.py imports that we might be missing
    train_script = project_root / "scripts" / "train_rlan.py"
    with open(train_script, 'r') as f:
        content = f.read()
    
    # Look for any extra model initialization
    if 'CrossAttentionInjector' in content:
        print("train_rlan.py uses CrossAttentionInjector")
    if 'HyperLoRA' in content:
        print("train_rlan.py uses HyperLoRA")
    if 'SolverCrossAttention' in content:
        print("train_rlan.py uses SolverCrossAttention")
    if 'HPM' in content or 'HierarchicalPatternMemory' in content:
        print("train_rlan.py uses HPM/HierarchicalPatternMemory")
    if 'create_staged_modules' in content:
        print("train_rlan.py uses create_staged_modules")
    
    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)
    
    # Final memory
    allocated, reserved = get_gpu_memory()
    print(f"\nFinal GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    
    # Recommendations
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    print("""
If memory is lower than 26GB here, the issue is in train_rlan.py initialization.
Check:
1. Are staged modules (CrossAttention, HyperLoRA) being created even when disabled?
2. Is HPM being initialized even when use_hpm=False?
3. Are there any memory leaks from hooks or caching?
4. Check if AMP is disabled (doubles memory without mixed precision)
""")


if __name__ == "__main__":
    main()
