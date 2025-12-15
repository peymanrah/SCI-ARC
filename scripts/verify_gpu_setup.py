#!/usr/bin/env python
"""
GPU/CUDA Setup Verification for SCI-ARC.

Verifies:
1. CUDA availability and version
2. GPU detection and memory
3. PyTorch CUDA compatibility
4. Model fits in VRAM
5. Training loop works on GPU

Optimized for: NVIDIA RTX 3090 (24GB VRAM) with CUDA 12.6
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_cuda_setup():
    """Check CUDA and GPU availability."""
    print("=" * 60)
    print(" GPU/CUDA SETUP VERIFICATION")
    print(" Target: NVIDIA RTX 3090 (24GB VRAM), CUDA 12.6")
    print("=" * 60)
    
    import torch
    
    print("\n1. PyTorch Version")
    print("-" * 40)
    print(f"   PyTorch: {torch.__version__}")
    print(f"   CUDA Available: {torch.cuda.is_available()}")
    
    if not torch.cuda.is_available():
        print("\n   [ERROR] CUDA is not available!")
        print("   Install PyTorch with CUDA support:")
        print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126")
        return False
    
    print(f"   CUDA Version: {torch.version.cuda}")
    print(f"   cuDNN Version: {torch.backends.cudnn.version()}")
    print(f"   cuDNN Enabled: {torch.backends.cudnn.enabled}")
    
    print("\n2. GPU Information")
    print("-" * 40)
    num_gpus = torch.cuda.device_count()
    print(f"   Number of GPUs: {num_gpus}")
    
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        total_mem = props.total_memory / (1024**3)  # GB
        print(f"\n   GPU {i}: {props.name}")
        print(f"   - Compute Capability: {props.major}.{props.minor}")
        print(f"   - Total Memory: {total_mem:.1f} GB")
        print(f"   - Multi-Processor Count: {props.multi_processor_count}")
    
    # Check for RTX 3090
    current_gpu = torch.cuda.get_device_properties(0)
    is_rtx_3090 = "3090" in current_gpu.name
    has_24gb = current_gpu.total_memory >= 20 * (1024**3)  # At least 20GB
    
    if is_rtx_3090:
        print(f"\n   [OK] RTX 3090 detected!")
    elif has_24gb:
        print(f"\n   [OK] GPU with sufficient VRAM detected")
    else:
        print(f"\n   [WARNING] GPU may have limited VRAM")
        print(f"   Consider reducing batch_size if OOM errors occur")
    
    print("\n3. Memory Status")
    print("-" * 40)
    allocated = torch.cuda.memory_allocated(0) / (1024**3)
    reserved = torch.cuda.memory_reserved(0) / (1024**3)
    total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    free = total - reserved
    
    print(f"   Total VRAM: {total:.1f} GB")
    print(f"   Allocated: {allocated:.3f} GB")
    print(f"   Reserved: {reserved:.3f} GB")
    print(f"   Free: {free:.1f} GB")
    
    return True


def check_model_fits():
    """Verify SCI-ARC model fits in VRAM."""
    print("\n4. Model Memory Test")
    print("-" * 40)
    
    import torch
    from sci_arc import SCIARC, SCIARCConfig
    
    # Create model with default config
    config = SCIARCConfig()
    model = SCIARC(config)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    param_memory = num_params * 4 / (1024**3)  # 4 bytes per float32
    
    print(f"   Model Parameters: {num_params/1e6:.2f}M")
    print(f"   Parameter Memory: {param_memory*1000:.1f} MB")
    
    # Move to GPU
    device = torch.device('cuda')
    model = model.to(device)
    
    allocated_after_model = torch.cuda.memory_allocated(0) / (1024**3)
    print(f"   GPU Memory After Model: {allocated_after_model:.3f} GB")
    
    # Test forward pass
    print("\n   Testing forward pass...")
    batch_size = 4
    num_demos = 3
    H, W = 10, 10
    
    demo_inputs = torch.randint(0, 10, (batch_size, num_demos, H, W), device=device)
    demo_outputs = torch.randint(0, 10, (batch_size, num_demos, H, W), device=device)
    test_input = torch.randint(0, 10, (batch_size, H, W), device=device)
    
    # Convert to demo_pairs format (list of tuples per batch)
    demo_pairs = [
        [(demo_inputs[b, i], demo_outputs[b, i]) for i in range(num_demos)]
        for b in range(batch_size)
    ]
    
    with torch.no_grad():
        # Use forward_training for batched interface
        output = model.forward_training(
            input_grids=demo_inputs,
            output_grids=demo_outputs,
            test_input=test_input,
            test_output=test_input  # Use same shape for test_output
        )
    
    allocated_after_forward = torch.cuda.memory_allocated(0) / (1024**3)
    print(f"   GPU Memory After Forward: {allocated_after_forward:.3f} GB")
    print(f"   Output Shape: {output['logits'].shape}")
    
    # Test backward pass with larger batch
    print("\n   Testing training pass (batch_size=32)...")
    model.train()
    
    batch_size = 32
    demo_inputs = torch.randint(0, 10, (batch_size, num_demos, H, W), device=device)
    demo_outputs = torch.randint(0, 10, (batch_size, num_demos, H, W), device=device)
    test_input = torch.randint(0, 10, (batch_size, H, W), device=device)
    test_output = torch.randint(0, 10, (batch_size, H, W), device=device)
    
    # Forward with gradients
    outputs = model.forward_training(
        input_grids=demo_inputs,
        output_grids=demo_outputs,
        test_input=test_input,
        test_output=test_output
    )
    
    # Compute loss
    logits = outputs['logits']
    B, H_out, W_out, C = logits.shape
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, C),
        test_output.view(-1)
    )
    
    # Backward
    loss.backward()
    
    allocated_after_backward = torch.cuda.memory_allocated(0) / (1024**3)
    reserved_after_backward = torch.cuda.memory_reserved(0) / (1024**3)
    
    print(f"   GPU Memory After Backward: {allocated_after_backward:.3f} GB")
    print(f"   GPU Memory Reserved: {reserved_after_backward:.3f} GB")
    print(f"   Loss Value: {loss.item():.4f}")
    
    # Check if we're within limits
    total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    usage_percent = (reserved_after_backward / total_vram) * 100
    
    print(f"\n   VRAM Usage: {usage_percent:.1f}% of {total_vram:.1f} GB")
    
    if usage_percent < 70:
        print("   [OK] Plenty of headroom - can increase batch_size if needed")
    elif usage_percent < 90:
        print("   [OK] Good VRAM usage")
    else:
        print("   [WARNING] High VRAM usage - reduce batch_size if OOM")
    
    # Cleanup
    del model, demo_inputs, demo_outputs, test_input, test_output, logits, loss
    torch.cuda.empty_cache()
    
    return True


def check_amp_support():
    """Check mixed precision (AMP) support."""
    print("\n5. Mixed Precision (AMP) Support")
    print("-" * 40)
    
    import torch
    from torch.amp import autocast, GradScaler
    
    # Check if GPU supports efficient FP16/BF16
    props = torch.cuda.get_device_properties(0)
    
    # RTX 3090 (Ampere) supports BF16
    is_ampere_or_newer = props.major >= 8
    
    print(f"   Compute Capability: {props.major}.{props.minor}")
    print(f"   Ampere or Newer: {is_ampere_or_newer}")
    
    if is_ampere_or_newer:
        print("   [OK] GPU supports efficient BF16 operations")
        print("   [OK] TensorFloat-32 (TF32) available for faster matmul")
    else:
        print("   [OK] GPU supports FP16 mixed precision")
    
    # Test AMP
    print("\n   Testing AMP forward pass...")
    
    from sci_arc import SCIARC, SCIARCConfig
    
    config = SCIARCConfig()
    model = SCIARC(config).cuda()
    scaler = GradScaler('cuda')
    
    batch_size = 8
    demo_inputs = torch.randint(0, 10, (batch_size, 3, 10, 10), device='cuda')
    demo_outputs = torch.randint(0, 10, (batch_size, 3, 10, 10), device='cuda')
    test_input = torch.randint(0, 10, (batch_size, 10, 10), device='cuda')
    test_output = torch.randint(0, 10, (batch_size, 10, 10), device='cuda')
    
    with autocast('cuda', dtype=torch.float16):
        outputs = model.forward_training(
            input_grids=demo_inputs,
            output_grids=demo_outputs,
            test_input=test_input,
            test_output=test_output
        )
        
        logits = outputs['logits']
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            test_output.view(-1)
        )
    
    scaler.scale(loss).backward()
    
    print(f"   AMP Forward+Backward: Success")
    print(f"   Loss (FP16): {loss.item():.4f}")
    
    allocated = torch.cuda.memory_allocated(0) / (1024**3)
    print(f"   GPU Memory with AMP: {allocated:.3f} GB")
    
    # Cleanup
    del model, scaler
    torch.cuda.empty_cache()
    
    return True


def print_summary():
    """Print setup summary."""
    print("\n" + "=" * 60)
    print(" SETUP SUMMARY")
    print("=" * 60)
    print("""
   Your SCI-ARC environment is configured for:
   
   - GPU: NVIDIA RTX 3090 (24GB VRAM)
   - CUDA: 12.6
   - Mixed Precision: Enabled (FP16/BF16)
   - Batch Size: 32 (can adjust based on grid sizes)
   
   To train:
     python scripts/train.py --config configs/default.yaml
   
   To evaluate:
     python scripts/evaluate.py --checkpoint checkpoints/best_model.pt
   
   If you encounter OOM errors:
     1. Reduce batch_size in config (32 -> 16 -> 8)
     2. Increase grad_accumulation_steps (1 -> 2 -> 4)
     3. Ensure use_amp: true is set
""")


def main():
    try:
        if not check_cuda_setup():
            print("\n[FAILED] CUDA setup failed. Please install CUDA support.")
            sys.exit(1)
        
        if not check_model_fits():
            print("\n[FAILED] Model memory test failed.")
            sys.exit(1)
        
        if not check_amp_support():
            print("\n[FAILED] AMP support test failed.")
            sys.exit(1)
        
        print_summary()
        print("\n[SUCCESS] All GPU checks passed!")
        sys.exit(0)
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
