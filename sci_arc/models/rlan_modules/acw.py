"""
Augmented Confidence Weighting (ACW) Module.

This module implements the inference-time voting mechanism for RLAN.
ACW improves generalization by aggregating predictions from multiple
augmented views of the same task.

Theory:
- Correct predictions are robust to geometric transformations (rotation, flip).
- Incorrect predictions (hallucinations) are often fragile and inconsistent.
- By voting based on consistency across views, we filter out noise.

Usage:
    acw = AugmentedConfidenceWeighting()
    winner, candidates = acw.weighted_vote(predictions)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional, Any

def apply_augmentation(
    tensor: torch.Tensor,
    aug_type: str,
    inverse: bool = False
) -> torch.Tensor:
    """
    Apply or inverse-apply an augmentation.
    
    Args:
        tensor: Input tensor (B, C, H, W) or (B, H, W) or (H, W)
        aug_type: Type of augmentation
        inverse: If True, apply inverse transform
        
    Returns:
        Augmented tensor
    """
    # Handle different input shapes
    ndim = tensor.dim()
    if ndim == 2: # (H, W)
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif ndim == 3: # (B, H, W) or (C, H, W) - ambiguous, assume (B, H, W) if int, (C, H, W) if float?
        # For ARC, grids are usually (H, W) or (B, H, W).
        # Let's standardize to (B, C, H, W) for processing
        tensor = tensor.unsqueeze(1)
    
    # D4 group transforms with proper inverse handling
    # CRITICAL: These must match dataset.py definitions exactly!
    # 
    # Forward transforms (matching TRM/dataset):
    #   0: identity
    #   1: rot90 CCW (k=1)
    #   2: rot180 (k=2)
    #   3: rot270 CCW (k=3)
    #   4: flip_h (horizontal flip)
    #   5: flip_v (vertical flip)
    #   6: transpose (main diagonal)
    #   7: anti-transpose = fliplr(rot90(k=1)) = flip_h after rot90
    #
    # Inverse mapping: [0, 3, 2, 1, 4, 5, 6, 7]
    #   - rot90 inverse is rot270 (and vice versa)
    #   - rot180, flip_h, flip_v, transpose, anti-transpose are self-inverse
    
    if aug_type == 'identity':
        result = tensor
    elif aug_type == 'rotate_90':
        if inverse:
            # Inverse of rot90 CCW is rot270 CCW (or rot90 CW)
            result = torch.rot90(tensor, k=3, dims=(-2, -1))
        else:
            result = torch.rot90(tensor, k=1, dims=(-2, -1))
    elif aug_type == 'rotate_180':
        # rot180 is self-inverse
        result = torch.rot90(tensor, k=2, dims=(-2, -1))
    elif aug_type == 'rotate_270':
        if inverse:
            # Inverse of rot270 CCW is rot90 CCW
            result = torch.rot90(tensor, k=1, dims=(-2, -1))
        else:
            result = torch.rot90(tensor, k=3, dims=(-2, -1))
    elif aug_type == 'flip_h':
        # Horizontal flip is self-inverse
        result = torch.flip(tensor, dims=[-1])
    elif aug_type == 'flip_v':
        # Vertical flip is self-inverse
        result = torch.flip(tensor, dims=[-2])
    elif aug_type == 'transpose':
        # Transpose is self-inverse
        result = tensor.transpose(-1, -2)
    elif aug_type == 'transpose_neg':
        # Anti-transpose = fliplr(rot90(k=1)) - matching TRM/dataset.py exactly!
        # This is self-inverse: applying it twice returns identity
        # Proof: fliplr(rot90(fliplr(rot90(x)))) = x
        result = torch.flip(torch.rot90(tensor, k=1, dims=(-2, -1)), dims=[-1])
    else:
        result = tensor
        
    # Restore original shape
    if ndim == 2:
        result = result.squeeze(0).squeeze(0)
    elif ndim == 3:
        result = result.squeeze(1)
        
    return result

class AugmentedConfidenceWeighting:
    """
    Augmented Confidence Weighting (ACW) for robust voting.
    
    Instead of simple majority voting, ACW weights each augmented
    prediction by its consistency with other augmentations. Predictions
    that agree with more views get higher weights.
    
    This is based on the insight that correct predictions tend to be
    consistent across different augmentations, while errors are often
    random and inconsistent.
    
    Algorithm:
    1. Get predictions from multiple augmented views
    2. For each prediction, compute consistency score:
       consistency = fraction of other views that agree
    3. Weight predictions by consistency in the vote
    4. Return weighted majority prediction
    
    Usage:
        acw = AugmentedConfidenceWeighting()
        predictions = [pred1, pred2, pred3, ...]  # List of (H, W) grids
        winner, ranked = acw.weighted_vote(predictions)
    """
    
    def __init__(self, temperature: float = 1.0):
        """
        Args:
            temperature: Temperature for softening consistency weights.
                         Higher = more uniform weights, lower = sharper.
        """
        self.temperature = temperature
    
    def _compute_hashes(
        self,
        predictions: List[torch.Tensor],
    ) -> List[str]:
        """
        Compute hashes for all predictions (O(N)) with BATCHED CPU transfer.
        
        OPTIMIZED: Stacks all predictions and does a single CPU transfer
        instead of N separate transfers (which caused N GPU syncs).
        
        This is shared between TRM-style counting and ACW consistency.
        """
        if not predictions:
            return []
        
        # CRITICAL FIX: Batch all predictions into single tensor for ONE CPU transfer
        # This eliminates N GPU-to-CPU synchronization points
        stacked = torch.stack(predictions)  # (N, H, W) - still on GPU
        stacked_cpu = stacked.cpu().numpy()  # Single transfer!
        
        # Now compute hashes from CPU array (no more GPU syncs)
        return [stacked_cpu[i].tobytes().hex() for i in range(len(predictions))]
    
    def compute_pairwise_agreement(
        self,
        predictions: List[torch.Tensor],
        hashes: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """
        Compute pairwise agreement matrix between predictions.
        
        OPTIMIZED: Uses hash comparison (O(1) per pair) instead of 
        tensor comparison (O(H*W) per pair).
        
        Args:
            predictions: List of (H, W) prediction tensors
            hashes: Optional pre-computed hashes (for efficiency)
            
        Returns:
            agreement: (N, N) matrix where [i,j] = 1 if predictions match
        """
        n = len(predictions)
        device = predictions[0].device if predictions else 'cpu'
        agreement = torch.zeros(n, n, device=device)
        
        # Compute hashes once (O(N)) if not provided
        if hashes is None:
            hashes = self._compute_hashes(predictions)
        
        # O(N²) hash comparisons (much faster than tensor comparisons)
        for i in range(n):
            for j in range(n):
                agreement[i, j] = float(hashes[i] == hashes[j])
        
        return agreement
    
    def compute_consistency_scores(
        self,
        predictions: List[torch.Tensor],
        hashes: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """
        Compute consistency score for each prediction.
        
        Consistency = average agreement with other predictions.
        
        Args:
            predictions: List of (H, W) prediction tensors
            hashes: Optional pre-computed hashes (for efficiency)
            
        Returns:
            scores: (N,) consistency scores in [0, 1]
        """
        n = len(predictions)
        if n <= 1:
            return torch.ones(n, device=predictions[0].device)
        
        agreement = self.compute_pairwise_agreement(predictions, hashes)
        
        # Consistency = average agreement with OTHER predictions
        # Exclude self-agreement (diagonal)
        mask = 1 - torch.eye(n, device=agreement.device)
        scores = (agreement * mask).sum(dim=1) / (n - 1)
        
        return scores
    
    def weighted_vote(
        self,
        predictions: List[torch.Tensor],
        return_all: bool = False,
    ) -> Tuple[torch.Tensor, List[Dict]]:
        """
        Perform weighted voting using consistency scores.
        
        Args:
            predictions: List of (H, W) prediction tensors
            return_all: If True, return all ranked candidates
            
        Returns:
            winner: Best prediction (H, W)
            ranked_candidates: List of dicts with:
                - "grid": prediction tensor
                - "weighted_count": sum of consistency weights
                - "raw_count": number of identical predictions
                - "avg_consistency": average consistency of this prediction
        """
        if len(predictions) == 0:
            return torch.zeros(1, 1), []
        
        if len(predictions) == 1:
            return predictions[0], [{
                "grid": predictions[0],
                "weighted_count": 1.0,
                "raw_count": 1,
                "avg_consistency": 1.0,
            }]
        
        # Compute hashes ONCE (O(N)) - shared between grouping and consistency
        hashes = self._compute_hashes(predictions)
        
        # Compute consistency scores using pre-computed hashes
        consistency = self.compute_consistency_scores(predictions, hashes)
        
        # Apply temperature scaling
        if self.temperature != 1.0:
            consistency = consistency / self.temperature
            consistency = F.softmax(consistency, dim=0) * len(predictions)
        
        # Group identical predictions using pre-computed hashes
        vote_groups = {}  # {hash: {indices, grids, weights}}
        
        for i, (pred, h) in enumerate(zip(predictions, hashes)):
            if h not in vote_groups:
                vote_groups[h] = {
                    "indices": [],
                    "grid": pred,
                    "weights": [],
                }
            vote_groups[h]["indices"].append(i)
            vote_groups[h]["weights"].append(consistency[i].item())
        
        # Compute aggregated scores
        candidates = []
        for group in vote_groups.values():
            weighted_count = sum(group["weights"])
            raw_count = len(group["indices"])
            avg_consistency = weighted_count / raw_count if raw_count > 0 else 0
            
            candidates.append({
                "grid": group["grid"],
                "weighted_count": weighted_count,
                "raw_count": raw_count,
                "avg_consistency": avg_consistency,
            })
        
        # Sort by weighted count (descending)
        candidates.sort(key=lambda x: x["weighted_count"], reverse=True)
        
        winner = candidates[0]["grid"]
        
        return winner, candidates
    
    def hybrid_vote(
        self,
        predictions: List[torch.Tensor],
        use_temperature: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Hybrid voting that combines TRM-style (count) and ACW (consistency).
        
        Strategy:
        1. Compute TRM winner (highest raw count)
        2. Compute ACW winner (highest consistency-weighted score)
        3. If they agree → high confidence, return that prediction
        4. If they disagree → use ACW winner (more robust to noise)
        
        This gives us the best of both worlds:
        - TRM: Simple, deterministic baseline
        - ACW: Better at filtering out noisy/inconsistent predictions
        
        OPTIMIZED: Computes hashes ONCE (O(N)), then:
        - TRM voting: O(N) using pre-computed hashes
        - ACW consistency: O(N²) hash comparisons (not tensor comparisons)
        - Total: O(N²) which is same as ACW alone
        
        Args:
            predictions: List of (H, W) prediction tensors
            
        Returns:
            winner: Best prediction (H, W)
            info: Dict with voting details and agreement status
        """
        if len(predictions) == 0:
            return torch.zeros(1, 1), {"method": "empty", "agree": True}
        
        if len(predictions) == 1:
            return predictions[0], {"method": "single", "agree": True}
        
        # Compute hashes ONCE (O(N)) - shared by TRM and ACW
        hashes = self._compute_hashes(predictions)
        
        # TRM-style: count-based voting using pre-computed hashes (O(N))
        trm_votes = {}
        for i, (pred, h) in enumerate(zip(predictions, hashes)):
            if h not in trm_votes:
                trm_votes[h] = {"count": 0, "grid": pred}
            trm_votes[h]["count"] += 1
        
        trm_winner_hash = max(trm_votes.keys(), key=lambda h: trm_votes[h]["count"])
        trm_winner = trm_votes[trm_winner_hash]["grid"]
        trm_winner_count = trm_votes[trm_winner_hash]["count"]
        
        # ACW: consistency-weighted voting using pre-computed hashes (O(N²))
        # Pass hashes to avoid recomputing
        consistency = self.compute_consistency_scores(predictions, hashes)
        
        # Apply temperature scaling for consistency with weighted_vote
        if use_temperature and self.temperature != 1.0:
            consistency = consistency / self.temperature
            consistency = F.softmax(consistency, dim=0) * len(predictions)
        
        # Group by hash and sum weighted votes
        acw_groups = {}
        for i, (pred, h) in enumerate(zip(predictions, hashes)):
            if h not in acw_groups:
                acw_groups[h] = {"grid": pred, "weights": [], "indices": []}
            acw_groups[h]["weights"].append(consistency[i].item())
            acw_groups[h]["indices"].append(i)
        
        # Find ACW winner (highest weighted sum)
        acw_scores = {h: sum(g["weights"]) for h, g in acw_groups.items()}
        acw_winner_hash = max(acw_scores.keys(), key=lambda h: acw_scores[h])
        acw_winner = acw_groups[acw_winner_hash]["grid"]
        acw_winner_score = acw_scores[acw_winner_hash]
        acw_winner_consistency = acw_winner_score / len(acw_groups[acw_winner_hash]["weights"])
        
        # Check agreement
        agree = (trm_winner_hash == acw_winner_hash)
        
        # Decision: use ACW when methods disagree (ACW is more robust)
        if agree:
            winner = trm_winner  # Both agree
            method = "consensus"
        else:
            # Methods disagree - use ACW (better at filtering noise)
            winner = acw_winner
            method = "acw_override"
        
        info = {
            "method": method,
            "agree": agree,
            "trm_winner_count": trm_winner_count,
            "acw_winner_score": acw_winner_score,
            "acw_winner_consistency": acw_winner_consistency,
            "num_unique": len(trm_votes),
            "num_predictions": len(predictions),
        }
        
        return winner, info
    
    def evaluate_with_acw(
        self,
        all_predictions: List[torch.Tensor],
        target: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Evaluate predictions using ACW, TRM, and Hybrid - compare all three.
        
        Args:
            all_predictions: List of augmented predictions
            target: Ground truth
            
        Returns:
            Dict with accuracy metrics for each voting method
        """
        # ACW voting
        acw_winner, acw_candidates = self.weighted_vote(all_predictions)
        acw_correct = torch.equal(acw_winner, target)
        
        # Simple majority voting (TRM-style)
        vote_counts = {}
        for pred in all_predictions:
            h = pred.cpu().numpy().tobytes().hex()
            if h not in vote_counts:
                vote_counts[h] = {"count": 0, "grid": pred}
            vote_counts[h]["count"] += 1
        
        simple_winner = max(vote_counts.values(), key=lambda x: x["count"])["grid"]
        simple_correct = torch.equal(simple_winner, target)
        
        # Hybrid voting
        hybrid_winner, hybrid_info = self.hybrid_vote(all_predictions)
        hybrid_correct = torch.equal(hybrid_winner, target)
        
        return {
            # TRM results
            "trm_correct": simple_correct,
            # ACW results
            "acw_correct": acw_correct,
            "acw_improved": acw_correct and not simple_correct,
            "acw_winner_consistency": acw_candidates[0]["avg_consistency"] if acw_candidates else 0,
            # Hybrid results
            "hybrid_correct": hybrid_correct,
            "hybrid_method": hybrid_info["method"],
            "hybrid_agree": hybrid_info["agree"],
            # Stats            "num_unique_predictions": len(vote_counts),
            "num_predictions": len(all_predictions),
        }