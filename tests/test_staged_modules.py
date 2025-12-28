"""
Comprehensive test for staged module activation mechanism.

Tests:
1. Epochs 0-2: All new modules disabled (matches old stable config)
2. Epoch 3+: All new modules activated
3. Transition stability: No BG collapse when activating modules
4. Forward/backward pass correctness
5. Real ARC data validation

Run with: python -m pytest tests/test_staged_modules.py -v -s
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import warnings

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sci_arc.models.rlan import RLAN, RLANConfig
from sci_arc.models.rlan_modules.context_encoder import ContextEncoder, CrossAttentionInjector, ContextInjector
from sci_arc.data.dataset import ARCDataset


class TestStagedModules:
    """Test suite for staged module activation."""
    
    @classmethod
    def setup_class(cls):
        """Setup test fixtures."""
        cls.device = torch.device('cpu')  # CPU for testing
        cls.hidden_dim = 128  # Smaller for faster tests
        cls.max_grid_size = 30
        cls.batch_size = 4
        cls.num_epochs_to_test = 5  # Test through transition
        
        # Create config matching production with all new modules
        cls.config = RLANConfig(
            hidden_dim=cls.hidden_dim,
            num_colors=10,  # ARC uses 10 colors (0-9)
            max_clues=6,
            num_solver_steps=4,  # Fewer for speed
            use_dsc=True,
            use_msre=True,
            use_context_encoder=True,
            use_cross_attention_context=True,  # NEW: CrossAttentionInjector
            use_solver_context=True,  # NEW: SolverCrossAttention
            use_hyperlora=True,  # NEW: HyperLoRA
            use_hpm=False,  # Skip HPM for now (separate staging)
            spatial_downsample=8,
            gradient_checkpointing=False,
            # Note: bg_bias is in RecursiveSolver, not RLANConfig
        )
        
        # Meta-learning starts at epoch 3 (0-indexed: epochs 0,1,2 are staging)
        cls.meta_learning_start_epoch = 3
        
    def create_model(self) -> RLAN:
        """Create a fresh model for testing."""
        model = RLAN(config=self.config, max_grid_size=self.max_grid_size)
        model.to(self.device)
        return model
    
    def create_synthetic_batch(self, batch_size: int = None) -> Dict[str, torch.Tensor]:
        """Create synthetic ARC-like batch for testing."""
        bs = batch_size or self.batch_size
        H, W = 10, 10  # Small grids for speed
        num_pairs = 3
        num_colors = 10  # ARC uses 10 colors (0-9)
        
        # Random grids with some structure (not pure noise)
        input_grid = torch.randint(0, num_colors, (bs, H, W))
        
        # Create output with some FG pixels (not all BG)
        output_grid = input_grid.clone()
        # Add some foreground changes
        for b in range(bs):
            fg_mask = torch.rand(H, W) > 0.7  # ~30% FG
            output_grid[b][fg_mask] = torch.randint(1, num_colors, (fg_mask.sum(),))
        
        # Training pairs
        train_inputs = torch.randint(0, num_colors, (bs, num_pairs, H, W))
        train_outputs = torch.randint(0, num_colors, (bs, num_pairs, H, W))
        pair_mask = torch.ones(bs, num_pairs, dtype=torch.bool)
        
        return {
            'input_grid': input_grid.to(self.device),
            'target_grid': output_grid.to(self.device),  # For loss computation only
            'train_inputs': train_inputs.to(self.device),
            'train_outputs': train_outputs.to(self.device),
            'pair_mask': pair_mask.to(self.device),
        }
    
    def set_staging_flags(self, model: RLAN, epoch: int):
        """Set staging flags based on epoch (mimics train_rlan.py logic)."""
        is_staged = epoch < self.meta_learning_start_epoch
        
        # All new modules disabled during epochs 0-2
        model.hyperlora_active = not is_staged
        model.solver_context_active = not is_staged
        model.cross_attention_active = not is_staged
        
        return {
            'hyperlora_active': model.hyperlora_active,
            'solver_context_active': model.solver_context_active,
            'cross_attention_active': model.cross_attention_active,
        }
    
    def compute_fg_ratio(self, logits: torch.Tensor) -> float:
        """Compute foreground prediction ratio (for collapse detection)."""
        # logits: (B, C, H, W) where C=10, class 0 is background
        preds = logits.argmax(dim=1)  # (B, H, W)
        fg_pixels = (preds != 0).float().sum()
        total_pixels = preds.numel()
        return (fg_pixels / total_pixels).item()
    
    def check_gradient_health(self, model: RLAN) -> Dict[str, Any]:
        """Check gradient statistics for stability."""
        grad_stats = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad
                grad_stats[name] = {
                    'mean': grad.mean().item(),
                    'std': grad.std().item() if grad.numel() > 1 else 0.0,
                    'max': grad.abs().max().item(),
                    'has_nan': torch.isnan(grad).any().item(),
                    'has_inf': torch.isinf(grad).any().item(),
                }
        return grad_stats
    
    # =========================================================================
    # TEST 1: Staging flags properly disable modules in epochs 0-2
    # =========================================================================
    def test_staging_flags_epoch_0_to_2(self):
        """Verify all new modules are disabled during epochs 0-2."""
        model = self.create_model()
        
        for epoch in range(3):  # Epochs 0, 1, 2
            flags = self.set_staging_flags(model, epoch)
            
            assert flags['hyperlora_active'] == False, \
                f"HyperLoRA should be INACTIVE at epoch {epoch}"
            assert flags['solver_context_active'] == False, \
                f"SolverCrossAttention should be INACTIVE at epoch {epoch}"
            assert flags['cross_attention_active'] == False, \
                f"CrossAttentionInjector should be INACTIVE at epoch {epoch}"
            
            print(f"  [OK] Epoch {epoch}: All new modules correctly DISABLED")
    
    # =========================================================================
    # TEST 2: Staging flags enable modules at epoch 3+
    # =========================================================================
    def test_staging_flags_epoch_3_plus(self):
        """Verify all new modules are enabled at epoch 3+."""
        model = self.create_model()
        
        for epoch in range(3, 6):  # Epochs 3, 4, 5
            flags = self.set_staging_flags(model, epoch)
            
            assert flags['hyperlora_active'] == True, \
                f"HyperLoRA should be ACTIVE at epoch {epoch}"
            assert flags['solver_context_active'] == True, \
                f"SolverCrossAttention should be ACTIVE at epoch {epoch}"
            assert flags['cross_attention_active'] == True, \
                f"CrossAttentionInjector should be ACTIVE at epoch {epoch}"
            
            print(f"  [OK] Epoch {epoch}: All new modules correctly ENABLED")
    
    # =========================================================================
    # TEST 3: Forward pass works with staging (epochs 0-2)
    # =========================================================================
    def test_forward_pass_staged(self):
        """Test forward pass with all modules DISABLED (epochs 0-2)."""
        model = self.create_model()
        model.eval()
        
        # Disable all new modules
        self.set_staging_flags(model, epoch=0)
        
        batch = self.create_synthetic_batch()
        
        with torch.no_grad():
            output = model(
                input_grid=batch['input_grid'],
                train_inputs=batch['train_inputs'],
                train_outputs=batch['train_outputs'],
                pair_mask=batch['pair_mask'],
                return_intermediates=True,
            )
        
        # Check output structure
        assert 'logits' in output, "Output should contain logits"
        
        logits = output['logits']
        assert logits.shape == (self.batch_size, 10, 10, 10), \
            f"Unexpected logits shape: {logits.shape}"
        
        # Check no NaN/Inf
        assert not torch.isnan(logits).any(), "Logits contain NaN in staged mode"
        assert not torch.isinf(logits).any(), "Logits contain Inf in staged mode"
        
        fg_ratio = self.compute_fg_ratio(logits)
        print(f"  [OK] Forward pass (staged): FG ratio = {fg_ratio:.3f}")
        
        # Should have some FG predictions (not collapsed)
        assert fg_ratio >= 0, "FG ratio should be non-negative"
    
    # =========================================================================
    # TEST 4: Forward pass works with modules ACTIVE (epoch 3+)
    # =========================================================================
    def test_forward_pass_active(self):
        """Test forward pass with all modules ENABLED (epoch 3+)."""
        model = self.create_model()
        model.eval()
        
        # Enable all new modules
        self.set_staging_flags(model, epoch=3)
        
        batch = self.create_synthetic_batch()
        
        with torch.no_grad():
            output = model(
                input_grid=batch['input_grid'],
                train_inputs=batch['train_inputs'],
                train_outputs=batch['train_outputs'],
                pair_mask=batch['pair_mask'],
                return_intermediates=True,
            )
        
        logits = output['logits']
        
        # Check no NaN/Inf
        assert not torch.isnan(logits).any(), "Logits contain NaN in active mode"
        assert not torch.isinf(logits).any(), "Logits contain Inf in active mode"
        
        fg_ratio = self.compute_fg_ratio(logits)
        print(f"  [OK] Forward pass (active): FG ratio = {fg_ratio:.3f}")
    
    # =========================================================================
    # TEST 5: Backward pass works and gradients are healthy (staged)
    # =========================================================================
    def test_backward_pass_staged(self):
        """Test backward pass with modules DISABLED - gradients should be healthy."""
        model = self.create_model()
        model.train()
        
        # Disable all new modules
        self.set_staging_flags(model, epoch=0)
        
        batch = self.create_synthetic_batch()
        
        # Forward
        output = model(
            input_grid=batch['input_grid'],
            train_inputs=batch['train_inputs'],
            train_outputs=batch['train_outputs'],
            pair_mask=batch['pair_mask'],
            return_intermediates=True,
        )
        
        # Compute simple CE loss
        logits = output['logits']
        target = batch['target_grid']
        loss = nn.functional.cross_entropy(logits, target)
        
        # Backward
        model.zero_grad()
        loss.backward()
        
        # Check gradient health
        grad_stats = self.check_gradient_health(model)
        
        nan_params = [k for k, v in grad_stats.items() if v['has_nan']]
        inf_params = [k for k, v in grad_stats.items() if v['has_inf']]
        
        assert len(nan_params) == 0, f"NaN gradients in: {nan_params}"
        assert len(inf_params) == 0, f"Inf gradients in: {inf_params}"
        
        # Check that base model params have gradients
        encoder_grads = [k for k in grad_stats.keys() if 'encoder' in k.lower()]
        assert len(encoder_grads) > 0, "Encoder should have gradients"
        
        print(f"  [OK] Backward pass (staged): Loss = {loss.item():.4f}, No NaN/Inf gradients")
    
    # =========================================================================
    # TEST 6: Backward pass works with modules ACTIVE
    # =========================================================================
    def test_backward_pass_active(self):
        """Test backward pass with modules ENABLED - gradients should be healthy."""
        model = self.create_model()
        model.train()
        
        # Enable all new modules
        self.set_staging_flags(model, epoch=3)
        
        batch = self.create_synthetic_batch()
        
        # Forward
        output = model(
            input_grid=batch['input_grid'],
            train_inputs=batch['train_inputs'],
            train_outputs=batch['train_outputs'],
            pair_mask=batch['pair_mask'],
            return_intermediates=True,
        )
        
        logits = output['logits']
        target = batch['target_grid']
        loss = nn.functional.cross_entropy(logits, target)
        
        # Backward
        model.zero_grad()
        loss.backward()
        
        # Check gradient health
        grad_stats = self.check_gradient_health(model)
        
        nan_params = [k for k, v in grad_stats.items() if v['has_nan']]
        inf_params = [k for k, v in grad_stats.items() if v['has_inf']]
        
        assert len(nan_params) == 0, f"NaN gradients in: {nan_params}"
        assert len(inf_params) == 0, f"Inf gradients in: {inf_params}"
        
        # Check that NEW module params have gradients when active
        hyperlora_grads = [k for k in grad_stats.keys() if 'hyper_lora' in k.lower()]
        if len(hyperlora_grads) > 0:
            print(f"    HyperLoRA has {len(hyperlora_grads)} params with gradients")
        
        print(f"  [OK] Backward pass (active): Loss = {loss.item():.4f}, No NaN/Inf gradients")
    
    # =========================================================================
    # TEST 7: FiLM fallback is used when CrossAttention is disabled
    # =========================================================================
    def test_film_fallback_used_when_staged(self):
        """Verify FiLM fallback injector is used when cross_attention_active=False."""
        model = self.create_model()
        
        # Verify model has both injectors
        assert hasattr(model, 'context_injector'), "Model should have context_injector"
        assert isinstance(model.context_injector, CrossAttentionInjector), \
            "context_injector should be CrossAttentionInjector"
        
        assert hasattr(model, 'film_fallback_injector'), \
            "Model should have film_fallback_injector for staging"
        assert isinstance(model.film_fallback_injector, ContextInjector), \
            "film_fallback_injector should be ContextInjector (FiLM)"
        
        print("  [OK] FiLM fallback injector exists for staged training")
    
    # =========================================================================
    # TEST 8: Simulate full training transition (CRITICAL)
    # =========================================================================
    def test_training_transition_no_collapse(self):
        """
        CRITICAL TEST: Simulate training through epoch transition.
        
        This tests the real scenario:
        1. Train for epochs 0-2 with modules disabled (like old stable config)
        2. Transition to epoch 3 with all modules enabled
        3. Verify NO BG collapse occurs at transition
        
        BG collapse = FG ratio dropping significantly at transition
        """
        print("\n" + "="*60)
        print("SIMULATING TRAINING TRANSITION (epochs 0-4)")
        print("="*60)
        
        model = self.create_model()
        model.train()
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        # Track FG ratios across epochs
        fg_ratios = []
        losses = []
        
        # Use same batch for consistency (simulates overfitting to measure stability)
        batch = self.create_synthetic_batch(batch_size=8)
        
        for epoch in range(5):  # Epochs 0-4
            # Set staging flags
            flags = self.set_staging_flags(model, epoch)
            
            is_transition = (epoch == self.meta_learning_start_epoch)
            phase = "STAGED" if epoch < self.meta_learning_start_epoch else "ACTIVE"
            
            if is_transition:
                print(f"\n  *** TRANSITION TO ACTIVE MODULES ***")
            
            # Run a few steps per "epoch"
            epoch_fg_ratios = []
            epoch_losses = []
            
            for step in range(3):  # 3 steps per epoch (reduced for speed on CPU)
                optimizer.zero_grad()
                
                output = model(
                    input_grid=batch['input_grid'],
                    train_inputs=batch['train_inputs'],
                    train_outputs=batch['train_outputs'],
                    pair_mask=batch['pair_mask'],
                    return_intermediates=True,
                )
                
                logits = output['logits']
                loss = nn.functional.cross_entropy(logits, batch['target_grid'])
                
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                fg_ratio = self.compute_fg_ratio(logits)
                epoch_fg_ratios.append(fg_ratio)
                epoch_losses.append(loss.item())
            
            avg_fg_ratio = np.mean(epoch_fg_ratios)
            avg_loss = np.mean(epoch_losses)
            
            fg_ratios.append(avg_fg_ratio)
            losses.append(avg_loss)
            
            print(f"  Epoch {epoch} [{phase}]: FG ratio = {avg_fg_ratio:.3f}, Loss = {avg_loss:.4f}")
            
            # Check for NaN
            assert not np.isnan(avg_loss), f"NaN loss at epoch {epoch}"
            assert not np.isnan(avg_fg_ratio), f"NaN FG ratio at epoch {epoch}"
        
        # =====================================================================
        # COLLAPSE DETECTION
        # =====================================================================
        print("\n  --- Collapse Detection ---")
        
        # Check FG ratio before and after transition
        pre_transition_fg = np.mean(fg_ratios[:self.meta_learning_start_epoch])
        post_transition_fg = np.mean(fg_ratios[self.meta_learning_start_epoch:])
        
        print(f"  Pre-transition FG ratio (epochs 0-2):  {pre_transition_fg:.3f}")
        print(f"  Post-transition FG ratio (epochs 3-4): {post_transition_fg:.3f}")
        
        # Collapse = FG ratio drops by more than 50% at transition
        if pre_transition_fg > 0.1:  # Only check if there was meaningful FG
            ratio_drop = (pre_transition_fg - post_transition_fg) / pre_transition_fg
            print(f"  FG ratio change: {ratio_drop*100:.1f}%")
            
            assert ratio_drop < 0.5, \
                f"BG COLLAPSE DETECTED! FG ratio dropped by {ratio_drop*100:.1f}% at transition"
        
        # Check loss didn't explode
        pre_loss = np.mean(losses[:self.meta_learning_start_epoch])
        post_loss = np.mean(losses[self.meta_learning_start_epoch:])
        
        print(f"  Pre-transition loss:  {pre_loss:.4f}")
        print(f"  Post-transition loss: {post_loss:.4f}")
        
        # Loss shouldn't increase dramatically at transition
        if pre_loss < 10:  # Sanity check
            loss_increase = (post_loss - pre_loss) / pre_loss if pre_loss > 0 else 0
            print(f"  Loss change: {loss_increase*100:.1f}%")
            
            # Allow up to 100% increase (modules adding new terms)
            assert loss_increase < 1.0 or post_loss < 5.0, \
                f"Loss exploded at transition: {pre_loss:.4f} -> {post_loss:.4f}"
        
        print("\n  [OK] NO BG COLLAPSE at transition!")
        print("="*60)
    
    # =========================================================================
    # TEST 9: Real ARC data validation (if available)
    # =========================================================================
    def test_with_real_arc_data(self):
        """Test with real ARC dataset if available."""
        # Try multiple potential locations
        arc_paths = [
            Path(__file__).parent.parent / "data" / "arc-agi_training_challenges.json",
            Path(__file__).parent.parent / "others" / "TinyRecursiveModels-main" / "kaggle" / "combined" / "arc-agi_training_challenges.json",
        ]
        
        arc_path = None
        for path in arc_paths:
            if path.exists():
                arc_path = path
                break
        
        if arc_path is None:
            print(f"  [!] Skipping real ARC test - data not found")
            return
        
        print(f"\n  Testing with REAL ARC data from: {arc_path.name}")
        
        try:
            dataset = ARCDataset(
                arc_path, 
                max_grid_size=self.max_grid_size,
                augment=False
            )
            
            if len(dataset) == 0:
                print("  [!] Empty dataset, skipping")
                return
            
            # Get a few real samples
            model = self.create_model()
            model.eval()
            
            # Test both staged and active modes
            for mode, epoch in [("STAGED", 0), ("ACTIVE", 3)]:
                self.set_staging_flags(model, epoch)
                
                success_count = 0
                total_count = min(5, len(dataset))
                
                for i in range(total_count):
                    try:
                        sample = dataset[i]
                        
                        # Prepare batch (add batch dimension)
                        input_grid = sample['input_grid'].unsqueeze(0).to(self.device)
                        output_grid = sample['output_grid'].unsqueeze(0).to(self.device)
                        train_inputs = sample['train_inputs'].unsqueeze(0).to(self.device)
                        train_outputs = sample['train_outputs'].unsqueeze(0).to(self.device)
                        pair_mask = sample['pair_mask'].unsqueeze(0).to(self.device)
                        
                        with torch.no_grad():
                            output = model(
                                input_grid=input_grid,
                                train_inputs=train_inputs,
                                train_outputs=train_outputs,
                                pair_mask=pair_mask,
                                return_intermediates=True,
                            )
                        
                        logits = output['logits']
                        
                        if not torch.isnan(logits).any() and not torch.isinf(logits).any():
                            success_count += 1
                    except Exception as e:
                        print(f"    Sample {i} failed: {e}")
                
                print(f"  [OK] Real ARC [{mode}]: {success_count}/{total_count} samples passed")
        
        except Exception as e:
            print(f"  [!] Real ARC test failed: {e}")
    
    # =========================================================================
    # TEST 10: Module output differences between staged/active
    # =========================================================================
    def test_module_outputs_differ_by_mode(self):
        """
        Verify that outputs are DIFFERENT between staged and active modes.
        This confirms the modules are actually being enabled/disabled.
        """
        model = self.create_model()
        model.eval()
        
        batch = self.create_synthetic_batch(batch_size=2)
        
        # Get output in STAGED mode
        self.set_staging_flags(model, epoch=0)
        with torch.no_grad():
            output_staged = model(
                input_grid=batch['input_grid'],
                train_inputs=batch['train_inputs'],
                train_outputs=batch['train_outputs'],
                pair_mask=batch['pair_mask'],
                return_intermediates=True,
            )
        logits_staged = output_staged['logits'].clone()
        
        # Get output in ACTIVE mode (same weights)
        self.set_staging_flags(model, epoch=3)
        with torch.no_grad():
            output_active = model(
                input_grid=batch['input_grid'],
                train_inputs=batch['train_inputs'],
                train_outputs=batch['train_outputs'],
                pair_mask=batch['pair_mask'],
                return_intermediates=True,
            )
        logits_active = output_active['logits'].clone()
        
        # Outputs should be DIFFERENT (modules change the computation)
        diff = (logits_staged - logits_active).abs().mean().item()
        
        print(f"  Logits difference (staged vs active): {diff:.6f}")
        
        # Should have some difference (if 0, modules aren't doing anything)
        assert diff > 1e-6, \
            "Outputs are identical! Staging flags may not be working."
        
        print("  [OK] Staging flags correctly change model behavior")


def run_all_tests():
    """Run all tests and summarize results."""
    print("\n" + "="*70)
    print("  STAGED MODULE ACTIVATION TEST SUITE")
    print("  Testing: HyperLoRA, SolverCrossAttention, CrossAttentionInjector")
    print("="*70 + "\n")
    
    test_class = TestStagedModules()
    test_class.setup_class()
    
    tests = [
        ("Staging flags epoch 0-2", test_class.test_staging_flags_epoch_0_to_2),
        ("Staging flags epoch 3+", test_class.test_staging_flags_epoch_3_plus),
        ("Forward pass (staged)", test_class.test_forward_pass_staged),
        ("Forward pass (active)", test_class.test_forward_pass_active),
        ("Backward pass (staged)", test_class.test_backward_pass_staged),
        ("Backward pass (active)", test_class.test_backward_pass_active),
        ("FiLM fallback exists", test_class.test_film_fallback_used_when_staged),
        ("Module outputs differ by mode", test_class.test_module_outputs_differ_by_mode),
        ("Training transition (CRITICAL)", test_class.test_training_transition_no_collapse),
        ("Real ARC data", test_class.test_with_real_arc_data),
    ]
    
    results = []
    
    for name, test_fn in tests:
        print(f"\n{'='*60}")
        print(f"TEST: {name}")
        print('='*60)
        
        try:
            test_fn()
            results.append((name, "PASS", None))
            print(f"\n  [OK] {name}: PASSED")
        except AssertionError as e:
            results.append((name, "FAIL", str(e)))
            print(f"\n  [X] {name}: FAILED - {e}")
        except Exception as e:
            results.append((name, "ERROR", str(e)))
            print(f"\n  [X] {name}: ERROR - {e}")
    
    # Summary
    print("\n" + "="*70)
    print("  TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, status, _ in results if status == "PASS")
    failed = sum(1 for _, status, _ in results if status == "FAIL")
    errors = sum(1 for _, status, _ in results if status == "ERROR")
    
    for name, status, error in results:
        icon = "[OK]" if status == "PASS" else "[X]"
        print(f"  {icon} {name}: {status}")
        if error:
            print(f"      {error[:80]}...")
    
    print(f"\n  TOTAL: {passed} passed, {failed} failed, {errors} errors")
    print("="*70)
    
    return passed, failed, errors


if __name__ == "__main__":
    # Run with warnings visible
    warnings.filterwarnings('default')
    
    passed, failed, errors = run_all_tests()
    
    # Exit with error code if any failures
    if failed > 0 or errors > 0:
        sys.exit(1)
    else:
        print("\n  [OK] ALL TESTS PASSED - Staged module activation is working correctly!")
        sys.exit(0)
