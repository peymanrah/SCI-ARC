"""
Causal Binding for SCI-ARC

Binds structural slots to content objects, producing a unified task embedding
z_task that captures both WHAT transformation and WHICH objects.

The binding mechanism:
1. Binding Attention: Structure slots query content objects
2. Causal Intervention: Combine structure with bound content
3. Aggregation: Pool into single task embedding

This is analogous to SCI's Causal Binding Mechanism (CBM), adapted for grids.
"""

from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalBinding2D(nn.Module):
    """
    Bind structural slots to content objects.
    
    This module produces z_task, the task understanding that conditions
    the recursive refinement module.
    
    Architecture:
    1. Binding Attention: Structure queries → Content keys/values
       - Learns which content objects are relevant for each structure slot
       - E.g., "the rotation pattern applies to the red square"
    
    2. Causal Intervention: Combine structure with bound content
       - Creates representation that captures how the transformation
         applies to specific objects
    
    3. Aggregation: Pool all slots into single z_task
       - z_task conditions the recursive refinement
    
    Parameters: ~1M
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_structure_slots: int = 8,
        num_content_slots: int = 16,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_gating: bool = True
    ):
        """
        Args:
            hidden_dim: Hidden dimension
            num_structure_slots: Number of structure slots (K from SE)
            num_content_slots: Number of content slots (M from CE)
            num_heads: Number of attention heads
            dropout: Dropout probability
            use_gating: Whether to use gating mechanism
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_structure_slots = num_structure_slots
        self.num_content_slots = num_content_slots
        self.use_gating = use_gating
        
        # === BINDING ATTENTION ===
        # Structure slots attend to content objects
        self.binding_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # === CAUSAL INTERVENTION ===
        # Process bound structure-content pairs
        self.intervention_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # Optional gating mechanism
        if use_gating:
            self.gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Sigmoid()
            )
        
        # === AGGREGATION ===
        # Pool slots into single task embedding
        self.slot_weights = nn.Linear(hidden_dim, 1)  # Learned attention pooling
        
        self.aggregator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        structure_slots: torch.Tensor,  # [B, K, D]
        content_slots: torch.Tensor     # [B, M, D]
    ) -> torch.Tensor:
        """
        Bind structure to content and produce task embedding.
        
        Args:
            structure_slots: [B, K, D] structural patterns from SE
            content_slots: [B, M, D] content objects from CE
        
        Returns:
            z_task: [B, D] task understanding
        """
        B = structure_slots.size(0)
        
        # Binding: structure queries content
        bound, binding_weights = self.binding_attention(
            query=structure_slots,
            key=content_slots,
            value=content_slots
        )  # bound: [B, K, D]
        
        # Causal intervention: combine structure with bound content
        combined = torch.cat([structure_slots, bound], dim=-1)  # [B, K, 2D]
        
        if self.use_gating:
            # Gated combination
            gate = self.gate(combined)  # [B, K, D]
            intervened = gate * structure_slots + (1 - gate) * self.intervention_mlp(combined)
        else:
            intervened = self.intervention_mlp(combined)
        
        # Aggregate slots into single embedding via attention pooling
        slot_scores = self.slot_weights(intervened)  # [B, K, 1]
        slot_weights = F.softmax(slot_scores, dim=1)
        pooled = (intervened * slot_weights).sum(dim=1)  # [B, D]
        
        # Final processing
        z_task = self.aggregator(pooled)
        z_task = self.norm(z_task)
        
        return z_task
    
    def forward_with_binding_weights(
        self,
        structure_slots: torch.Tensor,
        content_slots: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass that returns binding weights for visualization."""
        B = structure_slots.size(0)
        
        bound, binding_weights = self.binding_attention(
            query=structure_slots,
            key=content_slots,
            value=content_slots
        )
        
        combined = torch.cat([structure_slots, bound], dim=-1)
        
        if self.use_gating:
            gate = self.gate(combined)
            intervened = gate * structure_slots + (1 - gate) * self.intervention_mlp(combined)
        else:
            intervened = self.intervention_mlp(combined)
        
        slot_scores = self.slot_weights(intervened)
        slot_weights_agg = F.softmax(slot_scores, dim=1)
        pooled = (intervened * slot_weights_agg).sum(dim=1)
        
        z_task = self.norm(self.aggregator(pooled))
        
        return z_task, binding_weights, slot_weights_agg.squeeze(-1)


class DemoAggregator(nn.Module):
    """
    Aggregate z_task from multiple demo pairs.
    
    When a task has multiple demos, we need to combine their representations
    into a single task understanding.
    
    Options:
    1. Mean: Simple average
    2. Attention: Learned weighted average
    3. Max: Element-wise max
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        aggregation_type: str = "attention"
    ):
        """
        Args:
            hidden_dim: Hidden dimension
            aggregation_type: "mean", "attention", or "max"
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.aggregation_type = aggregation_type
        
        if aggregation_type == "attention":
            # Learn to weight demos
            self.demo_attention = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1)
            )
        
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, demo_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Aggregate embeddings from multiple demos.
        
        Args:
            demo_embeddings: [B, num_demos, D] embeddings from each demo
        
        Returns:
            aggregated: [B, D] single task embedding
        """
        if self.aggregation_type == "mean":
            aggregated = demo_embeddings.mean(dim=1)
        
        elif self.aggregation_type == "attention":
            weights = self.demo_attention(demo_embeddings)  # [B, num_demos, 1]
            weights = F.softmax(weights, dim=1)
            aggregated = (demo_embeddings * weights).sum(dim=1)
        
        elif self.aggregation_type == "max":
            aggregated, _ = demo_embeddings.max(dim=1)
        
        else:
            raise ValueError(f"Unknown aggregation type: {self.aggregation_type}")
        
        return self.norm(aggregated)


class TaskConditioner(nn.Module):
    """
    Condition computation on z_task.
    
    This module takes z_task and produces conditioning signals that can be
    applied at different layers of the recursive refinement.
    
    Options:
    1. FiLM: Feature-wise linear modulation (scale + shift)
    2. Addition: Simple additive conditioning
    3. Gating: Multiplicative gating
    """
    
    def __init__(
        self,
        task_dim: int = 256,
        target_dim: int = 256,
        conditioning_type: str = "film"
    ):
        """
        Args:
            task_dim: Dimension of z_task
            target_dim: Dimension of features to condition
            conditioning_type: "film", "add", or "gate"
        """
        super().__init__()
        
        self.conditioning_type = conditioning_type
        
        if conditioning_type == "film":
            # FiLM: Feature-wise Linear Modulation
            self.scale_net = nn.Linear(task_dim, target_dim)
            self.shift_net = nn.Linear(task_dim, target_dim)
        
        elif conditioning_type == "add":
            self.proj = nn.Linear(task_dim, target_dim)
        
        elif conditioning_type == "gate":
            self.gate_net = nn.Sequential(
                nn.Linear(task_dim, target_dim),
                nn.Sigmoid()
            )
            self.value_net = nn.Linear(task_dim, target_dim)
    
    def forward(
        self,
        features: torch.Tensor,  # [B, ..., target_dim]
        z_task: torch.Tensor     # [B, task_dim]
    ) -> torch.Tensor:
        """
        Condition features on z_task.
        
        Args:
            features: [B, ..., D] features to condition
            z_task: [B, D] task embedding
        
        Returns:
            conditioned: [B, ..., D] conditioned features
        """
        # Expand z_task to match features shape
        shape = [1] * (features.dim() - 2)  # [B, 1, 1, ..., D]
        
        if self.conditioning_type == "film":
            scale = self.scale_net(z_task).view(z_task.size(0), *shape, -1)
            shift = self.shift_net(z_task).view(z_task.size(0), *shape, -1)
            return features * (1 + scale) + shift
        
        elif self.conditioning_type == "add":
            proj = self.proj(z_task).view(z_task.size(0), *shape, -1)
            return features + proj
        
        elif self.conditioning_type == "gate":
            gate = self.gate_net(z_task).view(z_task.size(0), *shape, -1)
            value = self.value_net(z_task).view(z_task.size(0), *shape, -1)
            return features * gate + value * (1 - gate)
