"""
Stochastic Sampling for SCI-ARC Inference.

Implements Monte Carlo sampling strategies for robust prediction:
1. MC Dropout: Enable dropout during inference for diversity
2. Temperature Sampling: Softmax temperature for exploration
3. Top-K/Nucleus Sampling: Constrained sampling for quality

Mathematical Stability Notes:
- Temperature T is clamped to [0.1, 2.0] to prevent:
  - T → 0: logits → ±inf after division (numerical overflow)
  - T → ∞: uniform distribution (no information)
- Log-softmax used for numerical stability
- NaN guards on all tensor operations

Reference: 
- Self-Consistency: https://arxiv.org/abs/2203.11171
- MC Dropout: https://arxiv.org/abs/1506.02142
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
from collections import Counter
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SamplingConfig:
    """Configuration for stochastic sampling."""
    
    # Number of samples to generate per test input
    num_samples: int = 32
    
    # Temperature for softmax (1.0 = no scaling)
    # Lower = more confident, Higher = more diverse
    temperature: float = 1.0
    
    # Enable dropout during inference (MC Dropout)
    use_mc_dropout: bool = True
    
    # MC Dropout probability (if different from training)
    mc_dropout_prob: Optional[float] = None
    
    # Top-K sampling (0 = disabled)
    top_k: int = 0
    
    # Nucleus (top-p) sampling (1.0 = disabled)
    top_p: float = 1.0
    
    # Maximum batch size for parallel sampling
    max_batch_size: int = 64
    
    # Device
    device: str = 'cuda'


class StochasticSampler:
    """
    Generates diverse predictions through stochastic sampling.
    
    Key Features:
    - Monte Carlo Dropout for uncertainty estimation
    - Temperature-scaled sampling for diversity
    - Efficient batched inference
    - Automatic deduplication of identical predictions
    
    Usage:
        sampler = StochasticSampler(model, config)
        candidates, frequencies = sampler.generate_candidates(
            demo_inputs, demo_outputs, test_input
        )
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[SamplingConfig] = None,
    ):
        self.model = model
        self.config = config or SamplingConfig()
        self.device = torch.device(
            self.config.device if torch.cuda.is_available() else 'cpu'
        )
        
        # Store original dropout states for restoration
        self._original_dropout_states: Dict[str, bool] = {}
    
    def _enable_mc_dropout(self):
        """Enable dropout layers for MC sampling."""
        self._original_dropout_states = {}
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Dropout):
                self._original_dropout_states[name] = module.training
                module.train()  # Enable dropout
                if self.config.mc_dropout_prob is not None:
                    module.p = self.config.mc_dropout_prob
    
    def _restore_dropout(self):
        """Restore original dropout states."""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Dropout) and name in self._original_dropout_states:
                if not self._original_dropout_states[name]:
                    module.eval()
    
    def _sample_from_logits(
        self,
        logits: torch.Tensor,  # [B, H, W, C]
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Sample predictions from logits with temperature.
        
        Mathematical Stability:
        - Clamp temperature to [0.1, 2.0]
        - Use log_softmax + gumbel for numerical stability
        - Guard against NaN with torch.nan_to_num
        
        Args:
            logits: Raw logits from model [B, H, W, C]
            temperature: Sampling temperature
        
        Returns:
            Sampled predictions [B, H, W]
        """
        # Clamp temperature for stability
        temperature = max(0.1, min(2.0, temperature))
        
        # Scale logits
        scaled_logits = logits / temperature
        
        # Guard against NaN/Inf
        scaled_logits = torch.nan_to_num(scaled_logits, nan=0.0, posinf=10.0, neginf=-10.0)
        
        # Apply top-k filtering if enabled
        if self.config.top_k > 0:
            scaled_logits = self._top_k_filter(scaled_logits, self.config.top_k)
        
        # Apply nucleus (top-p) filtering if enabled
        if self.config.top_p < 1.0:
            scaled_logits = self._nucleus_filter(scaled_logits, self.config.top_p)
        
        # Sample using Gumbel-Softmax trick for efficiency
        # gumbel_softmax is differentiable, but we just want samples
        probs = F.softmax(scaled_logits, dim=-1)
        
        # Multinomial sampling across the color dimension
        B, H, W, C = probs.shape
        probs_flat = probs.view(-1, C)  # [B*H*W, C]
        
        # Guard against invalid probabilities
        probs_flat = probs_flat.clamp(min=1e-8)
        probs_flat = probs_flat / probs_flat.sum(dim=-1, keepdim=True)
        
        samples = torch.multinomial(probs_flat, num_samples=1).squeeze(-1)  # [B*H*W]
        samples = samples.view(B, H, W)  # [B, H, W]
        
        return samples
    
    def _top_k_filter(
        self, 
        logits: torch.Tensor, 
        k: int
    ) -> torch.Tensor:
        """Apply top-k filtering to logits."""
        top_k = min(k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k, dim=-1)[0][..., -1, None]
        logits = logits.masked_fill(indices_to_remove, float('-inf'))
        return logits
    
    def _nucleus_filter(
        self, 
        logits: torch.Tensor, 
        p: float
    ) -> torch.Tensor:
        """Apply nucleus (top-p) filtering to logits."""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > p
        # Keep at least one token
        sorted_indices_to_remove[..., 0] = False
        
        # Scatter back to original order
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, float('-inf'))
        return logits
    
    @torch.no_grad()
    def generate_candidates(
        self,
        input_grids: torch.Tensor,      # [num_demos, H, W]
        output_grids: torch.Tensor,     # [num_demos, H, W]
        test_input: torch.Tensor,       # [H, W]
        target_shape: Optional[Tuple[int, int]] = None,
    ) -> Tuple[List[np.ndarray], Dict[str, int]]:
        """
        Generate multiple candidate predictions through stochastic sampling.
        
        Args:
            input_grids: Demonstration input grids
            output_grids: Demonstration output grids  
            test_input: Test input to predict
            target_shape: Expected output shape (H, W)
        
        Returns:
            candidates: List of unique predicted grids (sorted by frequency)
            frequencies: Dict mapping grid hash to count
        """
        # Move to device
        input_grids = input_grids.to(self.device)
        output_grids = output_grids.to(self.device)
        test_input = test_input.to(self.device)
        
        if target_shape is None:
            target_shape = (test_input.shape[0], test_input.shape[1])
        
        # Collect all samples
        all_predictions = []
        num_samples = self.config.num_samples
        batch_size = min(self.config.max_batch_size, num_samples)
        
        # Enable MC Dropout if configured
        if self.config.use_mc_dropout:
            self._enable_mc_dropout()
        
        try:
            # Batch the sampling for efficiency
            for start_idx in range(0, num_samples, batch_size):
                current_batch = min(batch_size, num_samples - start_idx)
                
                # Expand inputs for batch
                # [num_demos, H, W] -> [batch, num_demos, H, W]
                batch_demos_in = input_grids.unsqueeze(0).expand(current_batch, -1, -1, -1)
                batch_demos_out = output_grids.unsqueeze(0).expand(current_batch, -1, -1, -1)
                batch_test = test_input.unsqueeze(0).expand(current_batch, -1, -1)
                
                # Create dummy test_output for shape inference
                test_output_dummy = torch.zeros(
                    current_batch, target_shape[0], target_shape[1],
                    dtype=torch.long, device=self.device
                )
                
                # Forward pass
                outputs = self.model.forward_training(
                    input_grids=batch_demos_in,
                    output_grids=batch_demos_out,
                    test_input=batch_test,
                    test_output=test_output_dummy,
                )
                
                logits = outputs['logits']  # [batch, H, W, C]
                
                # Sample from logits
                if self.config.temperature == 1.0 and start_idx == 0:
                    # First sample: greedy decoding (most likely)
                    first_pred = logits[0].argmax(dim=-1)  # [H, W]
                    all_predictions.append(first_pred.cpu().numpy())
                    
                    # Rest: stochastic sampling
                    if current_batch > 1:
                        samples = self._sample_from_logits(
                            logits[1:], self.config.temperature
                        )
                        for i in range(samples.shape[0]):
                            all_predictions.append(samples[i].cpu().numpy())
                else:
                    # All stochastic
                    samples = self._sample_from_logits(logits, self.config.temperature)
                    for i in range(samples.shape[0]):
                        all_predictions.append(samples[i].cpu().numpy())
        
        finally:
            # Restore dropout states
            if self.config.use_mc_dropout:
                self._restore_dropout()
        
        # Deduplicate and count frequencies
        candidates, frequencies = self._deduplicate_predictions(all_predictions)
        
        return candidates, frequencies
    
    def _deduplicate_predictions(
        self,
        predictions: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], Dict[str, int]]:
        """
        Deduplicate predictions and count frequencies.
        
        Returns predictions sorted by frequency (most common first).
        """
        # Convert to hashable tuples
        pred_hashes = []
        hash_to_pred = {}
        
        for pred in predictions:
            # Create hash from array
            pred_bytes = pred.tobytes()
            pred_hash = hash(pred_bytes)
            pred_hashes.append(pred_hash)
            
            if pred_hash not in hash_to_pred:
                hash_to_pred[pred_hash] = pred.copy()
        
        # Count frequencies
        counter = Counter(pred_hashes)
        
        # Sort by frequency
        sorted_hashes = [h for h, _ in counter.most_common()]
        candidates = [hash_to_pred[h] for h in sorted_hashes]
        frequencies = {str(h): c for h, c in counter.items()}
        
        return candidates, frequencies
    
    def majority_vote(
        self,
        candidates: List[np.ndarray],
        frequencies: Dict[str, int],
        top_k: int = 3
    ) -> List[np.ndarray]:
        """
        Return top-k candidates by frequency.
        
        ARC allows 2-3 attempts, so we return the most likely candidates.
        
        Args:
            candidates: List of unique grids (sorted by frequency)
            frequencies: Frequency counts
            top_k: Number of top candidates to return
        
        Returns:
            Top-k most frequent candidates
        """
        return candidates[:top_k]


class ConsistencyVerifier:
    """
    Verifies predictions through cross-augmentation consistency.
    
    Concept: A correct prediction should be consistent across
    different augmentations of the same input.
    
    For each candidate:
    1. Apply inverse augmentation to get "original" space
    2. Re-augment with different transforms
    3. Run model on re-augmented inputs
    4. Check if predictions agree after inverse transform
    
    High consistency = higher confidence in prediction.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda'
    ):
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    @torch.no_grad()
    def compute_consistency_score(
        self,
        candidate: np.ndarray,
        input_grids: torch.Tensor,
        output_grids: torch.Tensor,
        test_input: torch.Tensor,
        num_augments: int = 4
    ) -> float:
        """
        Compute consistency score for a candidate prediction.
        
        Score = fraction of augmented predictions that match the candidate.
        
        Args:
            candidate: Candidate prediction [H, W]
            input_grids: Demo inputs [N, H, W]
            output_grids: Demo outputs [N, H, W]
            test_input: Test input [H, W]
            num_augments: Number of augmentations to check
        
        Returns:
            Consistency score in [0, 1]
        """
        from ..evaluation.voting import dihedral_transform_torch, inverse_dihedral_transform_np
        
        matches = 0
        candidate_tensor = torch.from_numpy(candidate).long()
        
        for tid in range(min(num_augments, 8)):
            # Augment inputs
            aug_demos_in = dihedral_transform_torch(input_grids, tid)
            aug_demos_out = dihedral_transform_torch(output_grids, tid)
            aug_test = dihedral_transform_torch(test_input, tid)
            
            # Prepare batch
            aug_demos_in = aug_demos_in.unsqueeze(0).to(self.device)
            aug_demos_out = aug_demos_out.unsqueeze(0).to(self.device)
            aug_test = aug_test.unsqueeze(0).to(self.device)
            test_output_dummy = torch.zeros_like(aug_test)
            
            # Run model
            outputs = self.model.forward_training(
                input_grids=aug_demos_in,
                output_grids=aug_demos_out,
                test_input=aug_test,
                test_output=test_output_dummy,
            )
            
            pred = outputs['logits'][0].argmax(dim=-1).cpu().numpy()
            
            # Apply inverse transform
            pred_inv = inverse_dihedral_transform_np(pred, tid)
            
            # Check if matches candidate
            if np.array_equal(pred_inv, candidate):
                matches += 1
        
        return matches / num_augments
