"""
ConsistencyVerifier: Self-Consistency Verification for RLAN

Verifies that a predicted transformation is consistent with the 
transformation patterns observed in training pairs.

WHY THIS GENERALIZES:
- A correct prediction should exhibit the SAME transformation pattern
  as the training pairs
- If the test input → predicted output transformation differs from
  train input → train output transformations, the prediction is likely wrong
- This provides a self-consistency check that works on novel rules

Integration with RLAN:
- Takes RLAN's prediction as input
- Provides a confidence score
- Can reject low-confidence predictions
- Does NOT modify any RLAN code - pure wrapper

Author: AI Research Assistant
Date: January 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class ConsistencyConfig:
    """Configuration for ConsistencyVerifier."""
    enabled: bool = True
    hidden_dim: int = 256
    verification_threshold: float = 0.7
    use_structural_features: bool = True  # Compare structural features
    use_diff_features: bool = True  # Compare (output - input) patterns


class TransformationFeatureExtractor(nn.Module):
    """
    Extracts features that characterize the transformation from input to output.
    
    Key insight: The SAME transformation rule should produce SIMILAR features
    when applied to different examples. 
    """
    
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Encode input and output grids
        self.grid_encoder = nn.Sequential(
            nn.Conv2d(10, hidden_dim // 4, 3, padding=1),  # 10 colors one-hot
            nn.GELU(),
            nn.Conv2d(hidden_dim // 4, hidden_dim // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 2, hidden_dim, 3, padding=1),
            nn.AdaptiveAvgPool2d(4),  # Fixed size output
        )
        
        # Transformation encoder (takes concatenated input/output features)
        self.transform_encoder = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        
    def _grid_to_onehot(self, grid: torch.Tensor, num_colors: int = 10) -> torch.Tensor:
        """Convert grid of color indices to one-hot representation."""
        # grid: (H, W) or (B, H, W) with values in [0, num_colors-1]
        if grid.dim() == 2:
            grid = grid.unsqueeze(0)  # Add batch dim
        
        B, H, W = grid.shape
        grid = grid.long().clamp(0, num_colors - 1)
        onehot = torch.zeros(B, num_colors, H, W, device=grid.device)
        
        for c in range(num_colors):
            onehot[:, c] = (grid == c).float()
        
        return onehot
    
    def forward(
        self, 
        input_grid: torch.Tensor,   # (B, H, W) or (H, W)
        output_grid: torch.Tensor,  # (B, H, W) or (H, W)
    ) -> torch.Tensor:
        """
        Extract transformation features.
        
        Returns:
            features: (B, hidden_dim) transformation embedding
        """
        # Ensure batch dimension
        if input_grid.dim() == 2:
            input_grid = input_grid.unsqueeze(0)
        if output_grid.dim() == 2:
            output_grid = output_grid.unsqueeze(0)
        
        # Handle size mismatch by padding to larger size
        H_in, W_in = input_grid.shape[-2:]
        H_out, W_out = output_grid.shape[-2:]
        H_max, W_max = max(H_in, H_out), max(W_in, W_out)
        
        if H_in != H_max or W_in != W_max:
            padded = torch.zeros(input_grid.shape[0], H_max, W_max, device=input_grid.device, dtype=input_grid.dtype)
            padded[:, :H_in, :W_in] = input_grid
            input_grid = padded
            
        if H_out != H_max or W_out != W_max:
            padded = torch.zeros(output_grid.shape[0], H_max, W_max, device=output_grid.device, dtype=output_grid.dtype)
            padded[:, :H_out, :W_out] = output_grid
            output_grid = padded
        
        # Convert to one-hot
        input_onehot = self._grid_to_onehot(input_grid)
        output_onehot = self._grid_to_onehot(output_grid)
        
        # Encode grids
        input_features = self.grid_encoder(input_onehot)
        output_features = self.grid_encoder(output_onehot)
        
        # Concatenate and encode transformation
        combined = torch.cat([input_features, output_features], dim=1)
        transform_features = self.transform_encoder(combined)
        
        return transform_features


class ConsistencyVerifier(nn.Module):
    """
    Verifies that a prediction is consistent with training pair transformations.
    
    The key insight: if we compute the transformation features for
    (train_input → train_output) pairs and compare to
    (test_input → prediction), consistent predictions will have similar features.
    
    Usage:
        verifier = ConsistencyVerifier(ConsistencyConfig())
        
        score = verifier.verify(
            test_input=test_grid,
            prediction=predicted_output,
            train_inputs=[in1, in2, ...],
            train_outputs=[out1, out2, ...],
        )
        
        if score > threshold:
            # Accept prediction
        else:
            # Try alternative or reject
    """
    
    def __init__(self, config: ConsistencyConfig = None):
        super().__init__()
        
        self.config = config or ConsistencyConfig()
        
        if self.config.use_structural_features:
            self.feature_extractor = TransformationFeatureExtractor(
                self.config.hidden_dim
            )
        else:
            self.feature_extractor = None
    
    def _compute_structural_consistency(
        self,
        test_input: torch.Tensor,
        prediction: torch.Tensor,
        train_inputs: List[torch.Tensor],
        train_outputs: List[torch.Tensor],
    ) -> float:
        """Compute consistency using learned structural features."""
        if self.feature_extractor is None:
            return 1.0
        
        device = test_input.device
        
        # Compute test → prediction transformation features
        test_features = self.feature_extractor(test_input, prediction)
        
        # Compute training pair transformation features
        train_features_list = []
        for train_in, train_out in zip(train_inputs, train_outputs):
            features = self.feature_extractor(
                train_in.to(device), 
                train_out.to(device)
            )
            train_features_list.append(features)
        
        if not train_features_list:
            return 0.0
        
        # Stack and compute mean training features
        train_features = torch.stack(train_features_list, dim=0).mean(dim=0)
        
        # Compute cosine similarity
        test_norm = F.normalize(test_features, dim=-1)
        train_norm = F.normalize(train_features, dim=-1)
        
        similarity = (test_norm * train_norm).sum(dim=-1).mean().item()
        
        return similarity
    
    def _compute_diff_consistency(
        self,
        test_input: torch.Tensor,
        prediction: torch.Tensor,
        train_inputs: List[torch.Tensor],
        train_outputs: List[torch.Tensor],
    ) -> float:
        """
        Compute consistency using difference patterns.
        
        Compares what changed (colors added/removed, regions modified)
        between training pairs and test→prediction.
        """
        def to_np(g):
            if isinstance(g, torch.Tensor):
                return g.cpu().numpy()
            return np.array(g)
        
        test_in = to_np(test_input)
        pred_out = to_np(prediction)
        
        # Compute test diff features
        test_diff = self._compute_diff_features(test_in, pred_out)
        
        # Compute training diff features
        train_diffs = []
        for train_in, train_out in zip(train_inputs, train_outputs):
            diff = self._compute_diff_features(to_np(train_in), to_np(train_out))
            train_diffs.append(diff)
        
        if not train_diffs:
            return 0.0
        
        # Compare diff features
        avg_train_diff = np.mean(train_diffs, axis=0)
        
        # Cosine similarity
        test_norm = test_diff / (np.linalg.norm(test_diff) + 1e-8)
        train_norm = avg_train_diff / (np.linalg.norm(avg_train_diff) + 1e-8)
        
        similarity = np.dot(test_norm, train_norm)
        
        return float(similarity)
    
    def _compute_diff_features(
        self, 
        input_grid: np.ndarray, 
        output_grid: np.ndarray,
    ) -> np.ndarray:
        """
        Compute features that characterize the difference.
        
        Features:
        - Size ratio (output/input for H and W)
        - Color change histogram
        - Pixel change ratio
        """
        features = []
        
        # Size ratio
        h_in, w_in = input_grid.shape
        h_out, w_out = output_grid.shape
        features.append(h_out / h_in if h_in > 0 else 1.0)
        features.append(w_out / w_in if w_in > 0 else 1.0)
        
        # Color change histogram (for overlapping region)
        h_min, w_min = min(h_in, h_out), min(w_in, w_out)
        in_crop = input_grid[:h_min, :w_min]
        out_crop = output_grid[:h_min, :w_min]
        
        # What colors were added/removed
        for c in range(10):
            in_count = (in_crop == c).sum()
            out_count = (out_crop == c).sum()
            features.append((out_count - in_count) / max(h_min * w_min, 1))
        
        # Pixel change ratio
        changed = (in_crop != out_crop).sum()
        features.append(changed / max(h_min * w_min, 1))
        
        return np.array(features)
    
    def verify(
        self,
        test_input: torch.Tensor,
        prediction: torch.Tensor,
        train_inputs: List[torch.Tensor],
        train_outputs: List[torch.Tensor],
    ) -> Dict[str, Any]:
        """
        Verify prediction consistency with training pairs.
        
        Args:
            test_input: Test input grid
            prediction: Predicted output
            train_inputs: List of training inputs
            train_outputs: List of training outputs
            
        Returns:
            result: Dict with:
                - score: float - consistency score in [0, 1]
                - is_consistent: bool - whether score > threshold
                - structural_score: float - structural consistency
                - diff_score: float - difference pattern consistency
        """
        if not self.config.enabled:
            return {
                'score': 1.0,
                'is_consistent': True,
                'structural_score': 1.0,
                'diff_score': 1.0,
                'disabled': True,
            }
        
        scores = []
        result = {}
        
        # Structural consistency
        if self.config.use_structural_features:
            struct_score = self._compute_structural_consistency(
                test_input, prediction, train_inputs, train_outputs
            )
            scores.append(struct_score)
            result['structural_score'] = struct_score
        
        # Difference pattern consistency
        if self.config.use_diff_features:
            diff_score = self._compute_diff_consistency(
                test_input, prediction, train_inputs, train_outputs
            )
            scores.append(diff_score)
            result['diff_score'] = diff_score
        
        # Average score
        avg_score = np.mean(scores) if scores else 0.0
        
        result['score'] = avg_score
        result['is_consistent'] = avg_score >= self.config.verification_threshold
        
        return result
    
    def forward(self, *args, **kwargs):
        """Forward calls verify()."""
        return self.verify(*args, **kwargs)
