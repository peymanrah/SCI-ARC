"""
Anchor Robustness Training (ART) for RLAN - Jan 2026 Ablation Study

The ART module addresses the critical eval entropy collapse issue observed in RLAN:
- Train DSC entropy: ~0.21
- Eval DSC entropy: ~4.09 (19x higher!)

Root Cause: The model learns to rely on specific anchor patterns that don't
generalize. When eval data has different spatial distributions, DSC attention
becomes diffuse/random instead of sharp/focused.

Solution: Force the model to produce consistent predictions under ALTERNATE
anchors during training. This teaches DSC to discover anchors that lead to
the same answer, regardless of which specific anchor is chosen.

Theory (RLAN-aligned):
If anchor A and anchor B both cover the relevant spatial feature, then:
    f(input, anchor_A) ≈ f(input, anchor_B) after inverse reprojection

This is the anchor-invariance property that makes RLAN generalizable.

Integration Points:
1. Hook into DSC forward to extract top-K anchor candidates
2. Hook into training loop to compute alternate-anchor predictions
3. Add consistency loss to main loss computation

Usage:
    art_config = ARTConfig(enabled=True, num_alt_anchors=1)
    art_module = AnchorRobustnessTraining(art_config, hidden_dim=256)
    
    # During training:
    primary_logits, alt_logits_list = art_module.forward_with_alternates(
        model, input_grid, train_inputs, train_outputs, primary_centroids
    )
    consistency_loss = art_module.compute_consistency_loss(
        primary_logits, alt_logits_list
    )
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ARTConfig:
    """Configuration for Anchor Robustness Training."""
    enabled: bool = True
    num_alt_anchors: int = 1           # Number of alternate anchors per sample
    anchor_jitter_px: int = 2          # Max pixels to jitter anchor centroid
    use_top_k_anchors: bool = True     # Use top-K attention peaks as alternates
    consistency_loss_type: str = "kl"  # 'kl' or 'l2'
    consistency_weight: float = 0.02   # Weight in total loss
    temperature: float = 1.0           # Temperature for KL divergence
    detach_primary: bool = True        # Detach primary logits (train only alt path)


class AnchorRobustnessTraining(nn.Module):
    """
    Anchor Robustness Training module for RLAN.
    
    This module wraps around the RLAN model to:
    1. Extract alternate anchor candidates from DSC attention
    2. Run forward passes with forced alternate anchors
    3. Compute consistency loss between primary and alternate predictions
    
    The key insight is that anchor choice should not dramatically change
    the model's prediction - if it does, the model is not learning
    generalizable anchor-relative reasoning.
    """
    
    def __init__(self, config: ARTConfig, hidden_dim: int = 256):
        super().__init__()
        self.config = config
        self.hidden_dim = hidden_dim
        
        # Learnable anchor proposal head (optional refinement)
        # Maps attention maps to refined anchor candidates
        self.anchor_refiner = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
        )
    
    def extract_alternate_anchors(
        self,
        attention_maps: torch.Tensor,
        primary_centroids: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extract alternate anchor candidates from attention maps.
        
        Args:
            attention_maps: Shape (B, K, H, W) from DSC
            primary_centroids: Shape (B, K, 2) primary anchor centroids
            valid_mask: Shape (B, H, W) valid grid positions
            
        Returns:
            alt_centroids: Shape (B, num_alt, K, 2) alternate anchor sets
        """
        B, K, H, W = attention_maps.shape
        device = attention_maps.device
        num_alt = self.config.num_alt_anchors
        
        alt_centroids_list = []
        
        for alt_idx in range(num_alt):
            if self.config.use_top_k_anchors:
                # Find secondary peaks in attention maps
                alt_centroids = self._find_secondary_peaks(
                    attention_maps, primary_centroids, valid_mask
                )
            else:
                # Jitter primary centroids
                alt_centroids = self._jitter_centroids(
                    primary_centroids, H, W
                )
            alt_centroids_list.append(alt_centroids)
        
        # Stack: (B, num_alt, K, 2)
        alt_centroids = torch.stack(alt_centroids_list, dim=1)
        return alt_centroids
    
    def _find_secondary_peaks(
        self,
        attention_maps: torch.Tensor,
        primary_centroids: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Find secondary attention peaks by masking out primary regions."""
        B, K, H, W = attention_maps.shape
        device = attention_maps.device
        
        # Create mask around primary centroids
        row_grid = torch.arange(H, device=device).float().view(1, 1, H, 1)
        col_grid = torch.arange(W, device=device).float().view(1, 1, 1, W)
        
        # Primary centroid positions
        p_row = primary_centroids[:, :, 0].view(B, K, 1, 1)
        p_col = primary_centroids[:, :, 1].view(B, K, 1, 1)
        
        # Distance from primary (suppress nearby region)
        dist_sq = (row_grid - p_row)**2 + (col_grid - p_col)**2
        suppress_mask = (dist_sq < 4.0).float()  # Suppress within 2 pixels
        
        # Suppress primary region
        masked_attn = attention_maps * (1.0 - suppress_mask)
        
        # Apply valid mask if provided
        if valid_mask is not None:
            masked_attn = masked_attn * valid_mask.unsqueeze(1)
        
        # Find secondary peaks
        masked_flat = masked_attn.view(B, K, -1)
        peak_indices = masked_flat.argmax(dim=-1)  # (B, K)
        
        # Convert flat indices to (row, col)
        peak_rows = (peak_indices // W).float()
        peak_cols = (peak_indices % W).float()
        
        secondary_centroids = torch.stack([peak_rows, peak_cols], dim=-1)
        return secondary_centroids
    
    def _jitter_centroids(
        self,
        centroids: torch.Tensor,
        H: int,
        W: int,
    ) -> torch.Tensor:
        """Add random jitter to centroids."""
        B, K, _ = centroids.shape
        jitter_range = self.config.anchor_jitter_px
        
        # Random jitter
        jitter = torch.rand(B, K, 2, device=centroids.device) * 2 - 1
        jitter = jitter * jitter_range
        
        jittered = centroids + jitter
        
        # Clamp to valid grid
        jittered[:, :, 0] = jittered[:, :, 0].clamp(0, H - 1)
        jittered[:, :, 1] = jittered[:, :, 1].clamp(0, W - 1)
        
        return jittered
    
    def compute_consistency_loss(
        self,
        primary_logits: torch.Tensor,
        alt_logits_list: List[torch.Tensor],
        target: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute consistency loss between primary and alternate predictions.
        
        Args:
            primary_logits: Shape (B, C, H, W) primary prediction logits
            alt_logits_list: List of (B, C, H, W) alternate prediction logits
            target: Optional (B, H, W) target for weighting
            valid_mask: Optional (B, H, W) valid positions
            
        Returns:
            consistency_loss: Scalar loss
        """
        if not alt_logits_list:
            return torch.tensor(0.0, device=primary_logits.device)
        
        loss_type = self.config.consistency_loss_type
        temperature = self.config.temperature
        
        # Detach primary if configured (trains only alternate path)
        if self.config.detach_primary:
            primary_logits = primary_logits.detach()
        
        total_loss = 0.0
        
        for alt_logits in alt_logits_list:
            if loss_type == "kl":
                # KL divergence between distributions
                primary_probs = F.softmax(primary_logits / temperature, dim=1)
                alt_log_probs = F.log_softmax(alt_logits / temperature, dim=1)
                
                # KL(primary || alt) - per-pixel
                kl_div = F.kl_div(alt_log_probs, primary_probs, reduction='none')
                kl_div = kl_div.sum(dim=1)  # Sum over classes
                
                if valid_mask is not None:
                    kl_div = kl_div * valid_mask
                    loss = kl_div.sum() / (valid_mask.sum() + 1e-6)
                else:
                    loss = kl_div.mean()
                    
            elif loss_type == "l2":
                # L2 distance between logits
                diff_sq = (primary_logits - alt_logits) ** 2
                diff_sq = diff_sq.mean(dim=1)  # Mean over classes
                
                if valid_mask is not None:
                    diff_sq = diff_sq * valid_mask
                    loss = diff_sq.sum() / (valid_mask.sum() + 1e-6)
                else:
                    loss = diff_sq.mean()
            else:
                raise ValueError(f"Unknown consistency loss type: {loss_type}")
            
            total_loss = total_loss + loss
        
        # Average over alternates
        return total_loss / len(alt_logits_list)
    
    def forward(
        self,
        model: nn.Module,
        input_grid: torch.Tensor,
        train_inputs: torch.Tensor,
        train_outputs: torch.Tensor,
        temperature: float = 1.0,
        pair_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Forward pass with anchor robustness training.
        
        This method:
        1. Runs primary forward to get primary predictions and attention
        2. Extracts alternate anchors from attention maps
        3. Runs forward passes with forced alternate anchors
        4. Computes consistency loss
        
        Args:
            model: RLAN model instance
            input_grid: Shape (B, H, W) input grid
            train_inputs: Shape (B, N, H, W) training inputs
            train_outputs: Shape (B, N, H, W) training outputs
            temperature: DSC temperature
            pair_mask: Optional (B, N) valid pair mask
            
        Returns:
            Dictionary with:
                - logits: Primary prediction logits
                - alt_logits: List of alternate prediction logits
                - consistency_loss: ART consistency loss
                - all primary outputs (centroids, attention_maps, etc.)
        """
        # 1. Primary forward pass
        primary_outputs = model(
            input_grid,
            train_inputs=train_inputs,
            train_outputs=train_outputs,
            pair_mask=pair_mask,
            temperature=temperature,
            return_intermediates=True,
        )
        
        primary_logits = primary_outputs["logits"]
        attention_maps = primary_outputs["attention_maps"]
        primary_centroids = primary_outputs["centroids"]
        
        # 2. Extract alternate anchors
        valid_mask = model.encoder.get_valid_mask(input_grid)
        alt_centroids = self.extract_alternate_anchors(
            attention_maps, primary_centroids, valid_mask
        )
        
        # 3. Run forward passes with forced alternate anchors
        alt_logits_list = []
        B, num_alt, K, _ = alt_centroids.shape
        
        for alt_idx in range(num_alt):
            alt_centroid = alt_centroids[:, alt_idx, :, :]  # (B, K, 2)
            
            # Forward with forced centroids (skip DSC, use provided centroids)
            alt_outputs = self._forward_with_forced_centroids(
                model, input_grid, train_inputs, train_outputs,
                alt_centroid, temperature, pair_mask
            )
            alt_logits_list.append(alt_outputs["logits"])
        
        # 4. Compute consistency loss
        consistency_loss = self.compute_consistency_loss(
            primary_logits, alt_logits_list, valid_mask=valid_mask
        )
        
        # Return all outputs
        result = {
            "logits": primary_logits,
            "alt_logits": alt_logits_list,
            "consistency_loss": consistency_loss,
            "alt_centroids": alt_centroids,
        }
        result.update(primary_outputs)
        
        return result
    
    def _forward_with_forced_centroids(
        self,
        model: nn.Module,
        input_grid: torch.Tensor,
        train_inputs: torch.Tensor,
        train_outputs: torch.Tensor,
        forced_centroids: torch.Tensor,
        temperature: float,
        pair_mask: Optional[torch.Tensor],
    ) -> Dict[str, Any]:
        """
        Run RLAN forward pass with forced anchor centroids.
        
        This bypasses DSC's learned attention and uses provided centroids,
        allowing us to test how predictions change with different anchors.
        """
        B, H, W = input_grid.shape
        device = input_grid.device
        K = forced_centroids.shape[1]
        
        # Encode input
        features = model.encode(input_grid)
        valid_mask = model.encoder.get_valid_mask(input_grid)
        grid_sizes = model.encoder.get_grid_sizes(input_grid)
        
        # Context encoding (same as primary)
        context = None
        support_features = None
        dsc_task_context = None
        
        if model.use_context_encoder and model.context_encoder is not None:
            context_output = model.context_encoder(
                train_inputs, train_outputs, pair_mask
            )
            if model.context_encoder.use_spatial_features:
                support_features = context_output
                dsc_task_context = model.pool_context_from_support(context_output)
                # Inject context
                cross_attention_active = getattr(model, 'cross_attention_active', True)
                if hasattr(model.context_injector, 'forward') and cross_attention_active:
                    from sci_arc.models.rlan_modules import CrossAttentionInjector
                    if isinstance(model.context_injector, CrossAttentionInjector):
                        features = model.context_injector(features, context_output)
                    else:
                        features = model.context_injector(features, dsc_task_context)
            else:
                context = context_output
                dsc_task_context = context
                features = model.context_injector(features, context)
        
        # Create uniform attention maps centered on forced centroids
        row_grid = torch.arange(H, device=device).float().view(1, 1, H, 1)
        col_grid = torch.arange(W, device=device).float().view(1, 1, 1, W)
        
        c_row = forced_centroids[:, :, 0].view(B, K, 1, 1)
        c_col = forced_centroids[:, :, 1].view(B, K, 1, 1)
        
        # Gaussian attention around forced centroids
        sigma = 2.0  # Spread of attention
        dist_sq = (row_grid - c_row)**2 + (col_grid - c_col)**2
        attention_maps = torch.exp(-dist_sq / (2 * sigma**2))
        attention_maps = attention_maps / (attention_maps.sum(dim=(2, 3), keepdim=True) + 1e-6)
        
        # Create stop logits (all active)
        stop_logits = torch.zeros(B, K, device=device) - 5.0  # Low = don't stop
        
        # MSRE with forced centroids
        if model.use_msre and model.msre is not None:
            clue_features = model.msre(features, forced_centroids, grid_sizes=grid_sizes)
        else:
            clue_features = features.unsqueeze(1).expand(-1, K, -1, -1, -1)
        
        # LCR/SPH (same as primary)
        if model.use_lcr and model.lcr is not None:
            count_embedding = model.lcr(input_grid, features, mask=valid_mask, attention_maps=attention_maps)
        else:
            count_embedding = torch.zeros(B, model.num_colors, model.hidden_dim, device=device)
        
        if model.use_sph and model.sph is not None:
            predicates = model.sph(features, temperature=temperature)
        else:
            predicates = torch.zeros(B, model.num_predicates, device=device)
        
        # Solver
        solver_context_active = getattr(model, 'solver_context_active', True)
        effective_support_features = support_features if solver_context_active else None
        
        logits = model.solver(
            clue_features=clue_features,
            count_embedding=count_embedding,
            predicates=predicates,
            input_grid=input_grid,
            attention_maps=attention_maps,
            stop_logits=stop_logits,
            support_features=effective_support_features,
            return_all_steps=False,
        )
        
        return {
            "logits": logits,
            "attention_maps": attention_maps,
            "centroids": forced_centroids,
        }


def create_art_from_config(config: dict, hidden_dim: int = 256) -> Optional[AnchorRobustnessTraining]:
    """
    Factory function to create ART module from YAML config.
    
    Args:
        config: Dictionary from YAML config['model']['anchor_robustness']
        hidden_dim: Model hidden dimension
        
    Returns:
        ART module if enabled, None otherwise
    """
    if not config.get('enabled', False):
        return None
    
    art_config = ARTConfig(
        enabled=True,
        num_alt_anchors=config.get('num_alt_anchors', 1),
        anchor_jitter_px=config.get('anchor_jitter_px', 2),
        use_top_k_anchors=config.get('use_top_k_anchors', True),
        consistency_loss_type=config.get('consistency_loss_type', 'kl'),
        consistency_weight=config.get('consistency_weight', 0.02),
    )
    
    return AnchorRobustnessTraining(art_config, hidden_dim)
