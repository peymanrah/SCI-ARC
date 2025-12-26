"""
Tests that verify the bug fixes for metrics, TRM voting, HyperLoRA, LOO, and dataset.

These tests replace the former xfail tests now that the bugs are fixed.
"""
import numpy as np
import pytest
import torch

from sci_arc.evaluation import metrics as eval_metrics


class TestMetricsPaddingAwareness:
    """Verify that evaluation metrics now correctly handle -100 padding."""
    
    def test_task_accuracy_ignores_padding(self):
        """task_accuracy should ignore -100 padding pixels (like trainer validation)."""
        # Simulate ARC padded target (trainer uses ignore_index=-100)
        target = np.full((30, 30), -100, dtype=np.int64)
        target[:2, :2] = np.array([[1, 2], [3, 4]], dtype=np.int64)

        # Prediction matches all VALID pixels; padding values differ
        pred = np.full((30, 30), 0, dtype=np.int64)
        pred[:2, :2] = target[:2, :2]

        # Should be 1.0 because valid pixels match (padding ignored)
        assert eval_metrics.task_accuracy(pred, target) == 1.0
        
    def test_task_accuracy_legacy_mode(self):
        """task_accuracy with ignore_index=None should use legacy behavior."""
        target = np.full((5, 5), -100, dtype=np.int64)
        target[:2, :2] = 1
        pred = np.zeros((5, 5), dtype=np.int64)
        pred[:2, :2] = 1
        
        # Legacy mode: compare all pixels, so should fail
        assert eval_metrics.task_accuracy(pred, target, ignore_index=None) == 0.0
        
    def test_pixel_accuracy_ignores_padding(self):
        """pixel_accuracy should ignore -100 padding pixels."""
        target = np.full((10, 10), -100, dtype=np.int64)
        target[:3, :3] = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int64)
        
        pred = np.zeros((10, 10), dtype=np.int64)
        pred[:3, :3] = target[:3, :3]  # Match valid region
        
        # Should be 1.0 because valid pixels match
        assert eval_metrics.pixel_accuracy(pred, target) == 1.0
        
    def test_pixel_accuracy_strict_shape_mode(self):
        """pixel_accuracy with strict_shape=True returns 0.0 on shape mismatch."""
        target = np.zeros((3, 3), dtype=np.int64)
        pred = np.zeros((2, 2), dtype=np.int64)
        
        # Strict mode: shape mismatch is failure
        assert eval_metrics.pixel_accuracy(pred, target, strict_shape=True) == 0.0
        
    def test_pixel_accuracy_backward_compat_overlap(self):
        """pixel_accuracy default (strict_shape=False) uses overlap for backward compat."""
        target = np.zeros((3, 3), dtype=np.int64)
        pred = np.zeros((2, 2), dtype=np.int64)
        
        # Legacy behavior: overlap crops, both zeros → 1.0
        assert eval_metrics.pixel_accuracy(pred, target, strict_shape=False) == 1.0


class TestTRMVotingRankFix:
    """Verify that TRM voting now ranks by total_confidence (avg * count)."""
    
    def test_voting_prefers_frequent_consensus(self):
        """Voting should prefer frequent predictions with decent confidence."""
        from sci_arc.evaluation.trm_style_evaluator import TRMStyleEvaluator
        
        evaluator = TRMStyleEvaluator(pass_Ks=[1], use_voting=True, pad_value=10)

        # Ground truth should match candidate A
        gt = np.array([[1, 1], [1, 1]], dtype=np.int64)

        # Candidate A: appears 3x with 0.80 confidence each → total = 2.4
        cand_a = gt.copy()
        for _ in range(3):
            evaluator.update(
                task_id="t",
                prediction=cand_a,
                ground_truth=gt,
                aug_info={"dihedral_id": 0, "color_perm": None},
                confidence=0.80,
            )

        # Candidate B: appears 1x with 0.90 confidence → total = 0.9
        cand_b = np.array([[2, 2], [2, 2]], dtype=np.int64)
        evaluator.update(
            task_id="t",
            prediction=cand_b,
            ground_truth=gt,
            aug_info={"dihedral_id": 0, "color_perm": None},
            confidence=0.90,
        )

        # Candidate A has higher total_confidence (2.4 > 0.9), so Pass@1 should be 1.0
        metrics = evaluator.compute_metrics()
        assert metrics["pass@1"] == 1.0


class TestHyperLoRAInitScaleFix:
    """Verify that HyperLoRAConfig.init_scale is now respected."""
    
    def test_init_scale_affects_predictor_weights(self):
        """Different init_scale values should produce different weight initializations."""
        from sci_arc.models.rlan_modules.hyper_lora import HyperLoRA, HyperLoRAConfig

        torch.manual_seed(42)
        h1 = HyperLoRA(
            config=HyperLoRAConfig(
                hidden_dim=32,
                context_dim=32,
                rank=2,
                scaling=0.1,
                dropout=0.0,
                init_scale=0.01,
            )
        )
        w1_std = h1.gru_reset_lora.predict_A[-1].weight.std().item()

        torch.manual_seed(42)
        h2 = HyperLoRA(
            config=HyperLoRAConfig(
                hidden_dim=32,
                context_dim=32,
                rank=2,
                scaling=0.1,
                dropout=0.0,
                init_scale=0.5,
            )
        )
        w2_std = h2.gru_reset_lora.predict_A[-1].weight.std().item()

        # The std of weights should scale roughly with init_scale
        # (0.5 init_scale should give ~50x larger std than 0.01)
        assert w2_std > w1_std * 10, f"Expected w2_std >> w1_std, got {w2_std} vs {w1_std}"


class TestLOOTrainingLossInterfaceFix:
    """Verify that LOOTrainingLoss now uses correct HyperLoRA interface."""
    
    def test_loo_training_loss_callable(self):
        """LOOTrainingLoss should be callable end-to-end with HyperLoRA."""
        from sci_arc.models import RLAN, RLANConfig
        from sci_arc.models.rlan_modules.loo_training import LOOTrainingLoss, LOOConfig

        config = RLANConfig(
            hidden_dim=32,
            num_colors=10,
            num_classes=10,
            max_grid_size=10,
            num_solver_steps=1,
            use_dsc=False,
            use_msre=False,
            use_lcr=False,
            use_sph=False,
            use_context_encoder=True,
            use_cross_attention_context=True,
            spatial_downsample=1,
            dropout=0.0,
            use_hyperlora=True,
            hyperlora_rank=2,
        )
        model = RLAN(config=config)
        hyper_lora = model.hyper_lora

        loo = LOOTrainingLoss(LOOConfig(enabled=True, min_pairs_for_loo=2), hidden_dim=config.hidden_dim)

        B, N, H, W = 2, 3, 8, 8
        support_inputs = torch.randint(0, 10, (B, N, 1, H, W), dtype=torch.long)
        support_targets = torch.randint(0, 10, (B, N, H, W), dtype=torch.long)
        support_features = torch.randn(B, N, config.hidden_dim, H, W)

        # This should NOT raise an error anymore (interface is aligned)
        # LOOTrainingLoss returns a dict, not a tuple
        result = loo(
            hyper_lora=hyper_lora,
            rlan=model,
            support_inputs=support_inputs,
            support_targets=support_targets,
            support_features=support_features,
        )

        assert "loo_loss" in result
        assert torch.is_tensor(result["loo_loss"])


class TestDatasetExpandTestPairsFix:
    """Verify that SCIARCDataset can now deterministically cover all test pairs."""
    
    def test_expand_test_pairs_increases_length(self):
        """With expand_test_pairs=True, dataset length should expand for multi-test tasks."""
        from pathlib import Path
        import json
        from sci_arc.data.dataset import SCIARCDataset
        
        data_root = Path("data/arc-agi/data")
        if not data_root.exists():
            pytest.skip("ARC data not available")
        
        # Count tasks with multiple test pairs
        training_dir = data_root / "training"
        multi_test_count = 0
        for fp in training_dir.glob("*.json"):
            with open(fp) as f:
                task = json.load(f)
            if len(task.get("test", [])) > 1:
                multi_test_count += 1
        
        if multi_test_count == 0:
            pytest.skip("No multi-test tasks found")
        
        ds_normal = SCIARCDataset(str(data_root), split="training", augment=False, expand_test_pairs=False)
        ds_expanded = SCIARCDataset(str(data_root), split="training", augment=False, expand_test_pairs=True)
        
        # Expanded should be larger due to multi-test tasks
        assert len(ds_expanded) > len(ds_normal)
        
    def test_expand_test_pairs_includes_test_idx(self):
        """Samples should include test_idx for tracking."""
        from pathlib import Path
        from sci_arc.data.dataset import SCIARCDataset
        
        data_root = Path("data/arc-agi/data")
        if not data_root.exists():
            pytest.skip("ARC data not available")
        
        ds = SCIARCDataset(str(data_root), split="training", augment=False, expand_test_pairs=True)
        
        if len(ds) == 0:
            pytest.skip("Empty dataset")
        
        sample = ds[0]
        assert "test_idx" in sample, "Sample should include test_idx field"
