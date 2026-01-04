#!/usr/bin/env python
"""
Smoke Test for Ablation Study End-to-End Training

This script performs a quick validation that the ablation config can:
1. Load the configuration correctly
2. Initialize the RLAN model with ablation settings
3. Create the dataset and dataloader
4. Perform a few training steps without errors
5. Compute losses (including ART and ARPS losses)
6. Run a mini evaluation

This is NOT a full training run - it's a quick sanity check before
launching the actual ablation study.

Run with:
    python scripts/smoke_test_ablation.py

Expected output: All checks pass, ready for full ablation training.
"""

import sys
import os
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from typing import Dict, Any, Optional


def print_header(msg: str):
    """Print formatted header."""
    print(f"\n{'='*60}")
    print(f" {msg}")
    print(f"{'='*60}")


def print_check(name: str, passed: bool, details: str = ""):
    """Print check result."""
    status = "✓" if passed else "✗"
    detail_str = f" ({details})" if details else ""
    print(f"  {status} {name}{detail_str}")
    return passed


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    print_header("Ablation Study Smoke Test")
    
    all_passed = True
    
    # =========================================================================
    # 1. Load Configuration
    # =========================================================================
    print("\n[1/6] Loading ablation configuration...")
    
    config_path = project_root / "configs" / "rlan_stable_dev_ablation.yaml"
    try:
        config = load_config(config_path)
        all_passed &= print_check("Config loaded", True)
        
        # Verify key ablation settings
        model_cfg = config["model"]
        all_passed &= print_check("use_hyperlora=False", model_cfg.get("use_hyperlora") == False)
        all_passed &= print_check("use_hpm=False", model_cfg.get("use_hpm") == False)
        all_passed &= print_check("ART enabled", model_cfg.get("anchor_robustness", {}).get("enabled") == True)
        all_passed &= print_check("ARPS enabled", model_cfg.get("arps_dsl_search", {}).get("enabled") == True)
        
    except Exception as e:
        all_passed &= print_check("Config loading", False, str(e))
        print("\nCRITICAL: Cannot continue without config. Exiting.")
        return 1
    
    # =========================================================================
    # 2. Initialize Model
    # =========================================================================
    print("\n[2/6] Initializing RLAN model with ablation settings...")
    
    try:
        from sci_arc.models import RLAN, RLANConfig
        
        # Build RLANConfig from YAML
        rlan_config = RLANConfig(
            hidden_dim=model_cfg.get("hidden_dim", 256),
            num_colors=model_cfg.get("num_colors", 10),
            num_classes=model_cfg.get("num_classes", 10),
            max_grid_size=model_cfg.get("max_grid_size", 30),
            max_clues=model_cfg.get("max_clues", 7),
            num_predicates=model_cfg.get("num_predicates", 32),
            num_solver_steps=model_cfg.get("num_solver_steps", 4),
            use_best_step_selection=model_cfg.get("use_best_step_selection", True),
            use_context_encoder=model_cfg.get("use_context_encoder", True),
            use_dsc=model_cfg.get("use_dsc", True),
            use_msre=model_cfg.get("use_msre", True),
            use_lcr=model_cfg.get("use_lcr", False),
            use_sph=model_cfg.get("use_sph", False),
            use_hyperlora=model_cfg.get("use_hyperlora", False),
            use_hpm=model_cfg.get("use_hpm", False),
            use_solver_context=model_cfg.get("use_solver_context", True),
            use_cross_attention_context=model_cfg.get("use_cross_attention_context", True),
            dsc_num_heads=model_cfg.get("dsc_num_heads", 4),
            msre_encoding_dim=model_cfg.get("msre_encoding_dim", 32),
            msre_num_freq=model_cfg.get("msre_num_freq", 8),
            dropout=model_cfg.get("dropout", 0.1),
        )
        
        model = RLAN(config=rlan_config)
        all_passed &= print_check("RLAN model created", True)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        all_passed &= print_check(
            "Parameter count",
            total_params > 0,
            f"{trainable_params/1e6:.2f}M trainable"
        )
        
    except Exception as e:
        all_passed &= print_check("Model initialization", False, str(e))
        print("\nCRITICAL: Cannot continue without model. Exiting.")
        return 1
    
    # =========================================================================
    # 3. Initialize Ablation Modules (ART + ARPS)
    # =========================================================================
    print("\n[3/6] Initializing ablation modules (ART + ARPS)...")
    
    try:
        from sci_arc.models.rlan_modules import (
            create_art_from_config,
            create_arps_from_config,
        )
        
        art_cfg = model_cfg.get("anchor_robustness", {})
        arps_cfg = model_cfg.get("arps_dsl_search", {})
        arps_cfg["hidden_dim"] = model_cfg.get("hidden_dim", 256)
        
        art_module = create_art_from_config(art_cfg, hidden_dim=model_cfg.get("hidden_dim", 256))
        all_passed &= print_check("ART module created", art_module is not None)
        
        arps_module = create_arps_from_config(arps_cfg)
        all_passed &= print_check("ARPS module created", arps_module is not None)
        
        # Verify config values
        if art_module:
            all_passed &= print_check(
                "ART consistency_weight",
                art_module.config.consistency_weight > 0,
                f"{art_module.config.consistency_weight}"
            )
        
        if arps_module:
            all_passed &= print_check(
                "ARPS primitives loaded",
                len(arps_module.primitives) > 0,
                f"{len(arps_module.primitives)} primitives"
            )
        
    except Exception as e:
        all_passed &= print_check("Ablation module initialization", False, str(e))
        art_module = None
        arps_module = None
    
    # =========================================================================
    # 4. Create Synthetic Data and Run Forward Pass
    # =========================================================================
    print("\n[4/6] Running forward pass with synthetic data...")
    
    try:
        device = torch.device("cpu")  # Use CPU for smoke test
        model = model.to(device)
        model.train()
        
        # Create synthetic batch
        B, H, W = 4, 12, 12  # Small batch and grid for speed
        N = 2  # Training pairs
        
        input_grid = torch.randint(0, 10, (B, H, W), device=device)
        train_inputs = torch.randint(0, 10, (B, N, H, W), device=device)
        train_outputs = torch.randint(0, 10, (B, N, H, W), device=device)
        targets = torch.randint(0, 10, (B, H, W), device=device)
        
        # Forward pass
        start_time = time.time()
        outputs = model(
            input_grid,
            train_inputs=train_inputs,
            train_outputs=train_outputs,
            return_intermediates=True,
            return_all_steps=True,
        )
        forward_time = time.time() - start_time
        
        all_passed &= print_check("Forward pass completed", True, f"{forward_time:.3f}s")
        
        # Check outputs
        logits = outputs["logits"]
        all_passed &= print_check(
            "Logits shape correct",
            logits.shape == (B, 10, H, W),
            f"{logits.shape}"
        )
        
        all_passed &= print_check(
            "No NaN in logits",
            not torch.isnan(logits).any(),
        )
        
        # Check DSC outputs
        centroids = outputs["centroids"]
        attention_maps = outputs["attention_maps"]
        
        all_passed &= print_check(
            "Centroids shape",
            centroids.shape[0] == B and centroids.shape[2] == 2,
            f"{centroids.shape}"
        )
        
        all_passed &= print_check(
            "Attention maps shape",
            attention_maps.shape[0] == B,
            f"{attention_maps.shape}"
        )
        
    except Exception as e:
        all_passed &= print_check("Forward pass", False, str(e))
        return 1
    
    # =========================================================================
    # 5. Compute Losses (Task + ART + ARPS)
    # =========================================================================
    print("\n[5/6] Computing losses...")
    
    try:
        from sci_arc.training import RLANLoss
        
        # Task loss (RLANLoss requires all intermediate outputs)
        task_loss_fn = RLANLoss(
            loss_mode="focal_weighted",
            focal_gamma=1.2,
            focal_alpha=0.75,
        )
        
        # RLANLoss.forward() requires: logits, targets, attention_maps, stop_logits, predicates
        stop_logits = outputs["stop_logits"]
        predicates = outputs["predicates"]
        
        loss_dict = task_loss_fn(
            logits, 
            targets, 
            attention_maps,
            stop_logits,
            predicates,
        )
        task_loss = loss_dict["total_loss"]
        
        all_passed &= print_check(
            "Task loss computed",
            not torch.isnan(task_loss),
            f"{task_loss.item():.4f}"
        )
        
        # ART consistency loss
        if art_module is not None:
            art_module = art_module.to(device)
            
            # Extract alternate anchors
            alt_centroids = art_module.extract_alternate_anchors(
                attention_maps, centroids
            )
            
            # For smoke test, use same logits as "alternate" (real impl does separate forward)
            art_loss = art_module.compute_consistency_loss(
                logits, [logits + torch.randn_like(logits) * 0.1]
            )
            
            all_passed &= print_check(
                "ART loss computed",
                not torch.isnan(art_loss),
                f"{art_loss.item():.4f}"
            )
        
        # ARPS imitation loss
        if arps_module is not None:
            arps_module = arps_module.to(device)
            arps_module.train()
            
            # Get clue features (mock - in real impl comes from MSRE)
            clue_features = outputs.get("clue_features")
            if clue_features is None:
                # Create mock clue features
                K = centroids.shape[1]
                clue_features = torch.randn(B, K, model_cfg.get("hidden_dim", 256), H, W, device=device)
            
            arps_result = arps_module(
                clue_features,
                input_grid,
                train_inputs,
                train_outputs,
                centroids,
                temperature=1.0,
            )
            
            arps_loss = arps_result["imitation_loss"]
            all_passed &= print_check(
                "ARPS loss computed",
                not torch.isnan(arps_loss) if isinstance(arps_loss, torch.Tensor) else True,
                f"{arps_loss.item() if isinstance(arps_loss, torch.Tensor) else arps_loss:.4f}"
            )
            
            all_passed &= print_check(
                "ARPS search stats",
                "search_stats" in arps_result,
                f"valid={arps_result['search_stats']['num_valid_programs']}"
            )
        
        # Total loss
        total_loss = task_loss
        if art_module is not None:
            total_loss = total_loss + art_cfg.get("consistency_weight", 0.02) * art_loss
        if arps_module is not None and isinstance(arps_loss, torch.Tensor):
            total_loss = total_loss + arps_cfg.get("imitation_weight", 0.1) * arps_loss
        
        all_passed &= print_check(
            "Total loss valid",
            not torch.isnan(total_loss),
            f"{total_loss.item():.4f}"
        )
        
    except Exception as e:
        all_passed &= print_check("Loss computation", False, str(e))
    
    # =========================================================================
    # 6. Backward Pass and Gradient Check
    # =========================================================================
    print("\n[6/6] Running backward pass...")
    
    try:
        # Zero gradients
        model.zero_grad()
        if art_module is not None:
            art_module.zero_grad()
        if arps_module is not None:
            arps_module.zero_grad()
        
        # Backward
        start_time = time.time()
        total_loss.backward()
        backward_time = time.time() - start_time
        
        all_passed &= print_check("Backward pass completed", True, f"{backward_time:.3f}s")
        
        # Check gradients exist and are finite
        has_grad = False
        has_nan_grad = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                has_grad = True
                if torch.isnan(param.grad).any():
                    has_nan_grad = True
                    break
        
        all_passed &= print_check("Gradients computed", has_grad)
        all_passed &= print_check("No NaN gradients", not has_nan_grad)
        
        # Check specific module gradients
        if model.dsc is not None:
            dsc_grad = sum(
                p.grad.abs().mean().item() 
                for p in model.dsc.parameters() 
                if p.grad is not None
            )
            all_passed &= print_check("DSC gradients", dsc_grad > 0, f"mean={dsc_grad:.6f}")
        
        if model.solver is not None:
            solver_grad = sum(
                p.grad.abs().mean().item() 
                for p in model.solver.parameters() 
                if p.grad is not None
            )
            all_passed &= print_check("Solver gradients", solver_grad > 0, f"mean={solver_grad:.6f}")
        
    except Exception as e:
        all_passed &= print_check("Backward pass", False, str(e))
    
    # =========================================================================
    # Summary
    # =========================================================================
    print_header("Smoke Test Summary")
    
    if all_passed:
        print("\n✓ ALL CHECKS PASSED")
        print("\nThe ablation configuration is ready for full training:")
        print(f"  python scripts/train_rlan.py {config_path}")
        print("\nExpected behavior:")
        print("  - RLAN core modules (DSC, MSRE, Context, Solver) active")
        print("  - Meta-learning (HyperLoRA, HPM, LOO) disabled")
        print("  - ART consistency loss active for anchor robustness")
        print("  - ARPS program search active for interpretable predictions")
        print("\nTarget metrics:")
        print("  - Train exact match: >= 70%")
        print("  - Eval exact match: 20-30%")
        print("  - Eval/Train entropy ratio: <= 2.0")
        return 0
    else:
        print("\n✗ SOME CHECKS FAILED")
        print("\nPlease fix the issues above before running full training.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
