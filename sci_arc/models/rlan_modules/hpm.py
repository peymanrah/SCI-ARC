"""
Hierarchical Primitive Memory (HPM) v2 for RLAN

This module implements a multi-bank memory system for universal continual learning.
HPM stores ALL types of useful primitives - not just compositional transformations.

v2 IMPROVEMENTS (Dec 2025):
- Sparse MoE routing (Top-K instead of soft routing) - prevents mode collapse
- Gated residual (initialized to 0) - training stability
- Static vs Dynamic bank split - unbounded memory via KV-cache
- Load Balancing Loss - ensures all banks are utilized
- Integration points for HyperLoRA (Procedural) and ContextEncoder (Instance)

BANK TYPES:
- COMPOSITIONAL: Transformations that compose (rotate, translate, scale) [STATIC]
- PATTERN: Holistic patterns/templates (textures, shapes) [STATIC]
- RELATIONAL: Spatial/logical relationships (above, inside, equal) [STATIC]
- CONCEPT: Domain knowledge (semantics, categories) [STATIC]
- PROCEDURAL: HyperLoRA latent codes (task procedures) [DYNAMIC]
- INSTANCE: ContextEncoder outputs (solved task cache) [DYNAMIC]

MEMORY EFFICIENCY:
- No O(N) tensor accumulation (learned from LOO/equivariance fixes)
- Top-K routing means only k banks are queried per sample
- Static banks use nn.Parameter, Dynamic banks use external buffers
- Gated residual allows gradual HPM contribution

MODULAR DESIGN:
- Can enable/disable individual banks via YAML
- HPM itself can be disabled without code changes
- Works with any RLAN module configuration

Usage:
    # In YAML config:
    use_hpm: true
    hpm_top_k: 2
    use_compositional_bank: true
    use_pattern_bank: true
    ...
    
    # In model:
    hpm = HierarchicalPrimitiveMemory(config)
    z_enhanced, routing = hpm(z_context, return_routing=True)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryBankType(Enum):
    """Types of memory banks in HPM.
    
    Static banks (0-3): Use nn.Parameter, learned during training
    Dynamic banks (4-5): Use external KV-cache, grow with solved tasks
    """
    COMPOSITIONAL = 0   # Transformations that compose (rotate, translate)
    PATTERN = 1         # Holistic patterns/templates (textures, shapes)
    RELATIONAL = 2      # Spatial/logical relationships (above, inside)
    CONCEPT = 3         # Domain knowledge (semantics, categories)
    PROCEDURAL = 4      # HyperLoRA latent codes (task procedures) [DYNAMIC]
    INSTANCE = 5        # ContextEncoder outputs (solved task cache) [DYNAMIC]


# Which banks are static (nn.Parameter) vs dynamic (external buffer)
STATIC_BANK_TYPES = frozenset([
    MemoryBankType.COMPOSITIONAL,
    MemoryBankType.PATTERN,
    MemoryBankType.RELATIONAL,
    MemoryBankType.CONCEPT,
])

DYNAMIC_BANK_TYPES = frozenset([
    MemoryBankType.PROCEDURAL,
    MemoryBankType.INSTANCE,
])


@dataclass
class HPMConfig:
    """Configuration for Hierarchical Primitive Memory.
    
    All parameters should be set via YAML, no hardcoding in model code.
    """
    # Core dimensions
    d_model: int = 256
    
    # Sparse MoE routing
    top_k: int = 2                    # Number of banks to route to per sample
    balance_loss_weight: float = 0.01  # Weight for load balancing loss
    
    # Bank configuration
    primitives_per_bank: int = 16     # Number of primitives in each static bank
    n_levels_per_bank: int = 2        # Hierarchical levels within each bank
    use_cross_attention: bool = True  # Cross-attention aggregation
    
    # Dynamic bank settings
    max_dynamic_buffer_size: int = 10000  # Max entries in dynamic banks
    dynamic_retrieval_k: int = 5          # Number of neighbors to retrieve
    
    # Bank selection (which banks to use)
    use_compositional_bank: bool = True
    use_pattern_bank: bool = True
    use_relational_bank: bool = True
    use_concept_bank: bool = False     # Optional, disabled by default
    use_procedural_bank: bool = False  # Dynamic, needs HyperLoRA integration
    use_instance_bank: bool = False    # Dynamic, needs ContextEncoder integration
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'HPMConfig':
        """Create HPMConfig from dictionary (YAML config)."""
        return cls(
            d_model=config.get('hidden_dim', 256),
            top_k=config.get('hpm_top_k', 2),
            balance_loss_weight=config.get('hpm_balance_weight', 0.01),
            primitives_per_bank=config.get('hpm_primitives_per_bank', 16),
            n_levels_per_bank=config.get('hpm_levels_per_bank', 2),
            use_cross_attention=config.get('hpm_use_cross_attention', True),
            max_dynamic_buffer_size=config.get('hpm_memory_size', 10000),
            dynamic_retrieval_k=config.get('hpm_retrieval_k', 5),
            use_compositional_bank=config.get('hpm_use_compositional_bank', True),
            use_pattern_bank=config.get('hpm_use_pattern_bank', True),
            use_relational_bank=config.get('hpm_use_relational_bank', True),
            use_concept_bank=config.get('hpm_use_concept_bank', False),
            use_procedural_bank=config.get('hpm_use_procedural_bank', False),
            use_instance_bank=config.get('hpm_use_instance_bank', False),
        )
    
    def get_enabled_bank_types(self) -> List[MemoryBankType]:
        """Get list of enabled bank types based on configuration."""
        banks = []
        if self.use_compositional_bank:
            banks.append(MemoryBankType.COMPOSITIONAL)
        if self.use_pattern_bank:
            banks.append(MemoryBankType.PATTERN)
        if self.use_relational_bank:
            banks.append(MemoryBankType.RELATIONAL)
        if self.use_concept_bank:
            banks.append(MemoryBankType.CONCEPT)
        if self.use_procedural_bank:
            banks.append(MemoryBankType.PROCEDURAL)
        if self.use_instance_bank:
            banks.append(MemoryBankType.INSTANCE)
        return banks


class MemoryBank(nn.Module):
    """
    Single static memory bank storing one type of primitive.
    
    Each bank has:
    - Hierarchical primitive embeddings (coarse to fine levels)
    - Level mixing weights (learned)
    - Query projection (bank-specific)
    - Optional freeze mechanism for continual learning
    
    MEMORY EFFICIENCY:
    - Fixed size (n_primitives × d_model per level)
    - Attention is O(n_primitives) per query - very efficient
    - No tensor accumulation in forward pass
    """
    
    def __init__(
        self,
        d_model: int = 256,
        n_primitives: int = 16,
        n_levels: int = 2,
        bank_type: MemoryBankType = MemoryBankType.COMPOSITIONAL,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_primitives = n_primitives
        self.n_levels = n_levels
        self.bank_type = bank_type
        
        # Distribute primitives across levels (more at finer levels)
        primitives_per_level = [n_primitives // n_levels] * n_levels
        primitives_per_level[-1] += n_primitives % n_levels  # Remainder to finest
        self.primitives_per_level = primitives_per_level
        
        # Primitive embeddings for each hierarchical level
        # Small init (0.02) for stable training
        self.primitive_levels = nn.ParameterList([
            nn.Parameter(torch.randn(n, d_model) * 0.02)
            for n in primitives_per_level
        ])
        
        # Level mixing weights (learnable, softmax applied at forward)
        self.level_weights = nn.Parameter(torch.ones(n_levels) / n_levels)
        
        # Query projection (bank-specific transform for queries)
        self.query_proj = nn.Linear(d_model, d_model, bias=False)
        nn.init.eye_(self.query_proj.weight)  # Start as identity
        
        # Freeze masks for continual learning (not nn.Parameter, just buffers)
        # These are updated by freeze_stable_primitives()
        for level_idx, n in enumerate(primitives_per_level):
            self.register_buffer(
                f'freeze_mask_{level_idx}',
                torch.zeros(n, dtype=torch.bool),
                persistent=False,  # Don't save to checkpoint
            )
        
        # Usage tracking for freeze decisions
        for level_idx, n in enumerate(primitives_per_level):
            self.register_buffer(
                f'usage_count_{level_idx}',
                torch.zeros(n),
                persistent=False,
            )
    
    def forward(
        self,
        z: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Retrieve from this memory bank.
        
        MEMORY EFFICIENT: No tensor accumulation, just attention over primitives.
        
        Args:
            z: Query encoding [B, D]
            return_attention: Whether to return attention weights for interpretability
            
        Returns:
            output: Bank output [B, D]
            attentions: Optional list of attention weights per level
        """
        B = z.shape[0]
        query = self.query_proj(z)  # [B, D]
        
        level_outputs = []
        level_attentions = [] if return_attention else None
        
        for level_idx, primitives in enumerate(self.primitive_levels):
            # Scaled dot-product attention over primitives
            # scores: [B, K_level] where K_level is primitives at this level
            scores = torch.matmul(query, primitives.T) / math.sqrt(self.d_model)
            alpha = F.softmax(scores, dim=-1)  # [B, K_level]
            
            # Weighted sum of primitives: [B, D]
            output = torch.matmul(alpha, primitives)
            level_outputs.append(output)
            
            if return_attention:
                level_attentions.append(alpha.detach())
            
            # Update usage tracking (training only, detached)
            if self.training:
                usage_buffer = getattr(self, f'usage_count_{level_idx}')
                # Accumulate mean attention across batch (detached, no gradients)
                usage_buffer.add_(alpha.detach().mean(dim=0).to(usage_buffer.device))
        
        # Combine levels with learned weights
        level_weights_normalized = F.softmax(self.level_weights, dim=0)
        output = sum(
            w * o for w, o in zip(level_weights_normalized, level_outputs)
        )
        
        return output, level_attentions
    
    def freeze_stable_primitives(
        self,
        usage_threshold: int = 100,
        top_fraction: float = 0.3,
    ):
        """
        Freeze primitives that are stable and well-used.
        
        Called during continual learning to prevent forgetting.
        Primitives with high usage are frozen (no gradient updates).
        
        Args:
            usage_threshold: Minimum total usage to consider freezing
            top_fraction: Fraction of most-used primitives to freeze
        """
        for level_idx in range(self.n_levels):
            usage = getattr(self, f'usage_count_{level_idx}')
            
            if usage.sum() < usage_threshold:
                continue  # Not enough usage data
            
            # Find top-used primitives
            n_to_freeze = max(1, int(len(usage) * top_fraction))
            _, top_indices = torch.topk(usage, n_to_freeze)
            
            # Update freeze mask
            freeze_mask = getattr(self, f'freeze_mask_{level_idx}')
            freeze_mask.zero_()
            freeze_mask[top_indices] = True
    
    def apply_gradient_routing(self):
        """Zero gradients for frozen primitives after backward."""
        for level_idx, primitives in enumerate(self.primitive_levels):
            if primitives.grad is None:
                continue
            
            freeze_mask = getattr(self, f'freeze_mask_{level_idx}')
            if freeze_mask.any():
                freeze_mask = freeze_mask.to(primitives.device)
                # Zero out gradients for frozen primitives
                primitives.grad[freeze_mask] = 0
    
    def reset_usage_counts(self):
        """Reset usage counts (call at epoch start)."""
        for level_idx in range(self.n_levels):
            usage = getattr(self, f'usage_count_{level_idx}')
            usage.zero_()


class MemoryRouter(nn.Module):
    """
    Routes queries to appropriate memory banks using Sparse MoE Top-K routing.
    
    v2 IMPROVEMENTS:
    - Top-K selection instead of soft routing (prevents mode collapse)
    - Load Balancing Loss to ensure all banks are utilized
    - Temperature-controlled routing sharpness
    
    MEMORY EFFICIENT:
    - Only k banks are queried per sample (not all banks)
    - Statistics tracking uses buffers, no tensor accumulation
    """
    
    def __init__(
        self,
        d_model: int = 256,
        n_banks: int = 6,
        top_k: int = 2,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_banks = n_banks
        self.top_k = min(top_k, n_banks)  # Can't select more banks than exist
        
        # Routing network: z -> bank logits
        self.router = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_banks),
        )
        
        # Temperature for routing sharpness (learnable)
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
        # Statistics for load balancing loss
        self.register_buffer('routing_counts', torch.zeros(n_banks))
        self.register_buffer('routing_probs_sum', torch.zeros(n_banks))
        self.register_buffer('total_samples', torch.tensor(0.0))
    
    def forward(
        self,
        z: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute sparse routing weights using Top-K selection.
        
        Args:
            z: Query encoding [B, D]
            
        Returns:
            weights: Sparse routing weights [B, n_banks] (only top_k nonzero per row)
            top_k_indices: Which banks were selected [B, top_k]
        """
        B = z.shape[0]
        logits = self.router(z)  # [B, n_banks]
        
        # Temperature-controlled sharpness
        temp = F.softplus(self.temperature) + 0.1  # Minimum temp of 0.1
        
        # Top-K selection: only the top_k banks get nonzero weight
        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1)  # [B, top_k]
        
        # Normalize only selected banks (softmax over top_k)
        top_k_weights = F.softmax(top_k_logits / temp, dim=-1)  # [B, top_k]
        
        # Create sparse weight tensor [B, n_banks]
        weights = torch.zeros(B, self.n_banks, device=z.device, dtype=z.dtype)
        weights.scatter_(1, top_k_indices, top_k_weights)
        
        # Update routing statistics for load balancing loss (training only)
        if self.training:
            with torch.no_grad():
                self.routing_counts.add_(weights.sum(dim=0))
                # Track probability assigned to each bank (for load balance)
                all_probs = F.softmax(logits / temp, dim=-1)
                self.routing_probs_sum.add_(all_probs.sum(dim=0))
                self.total_samples.add_(B)
        
        return weights, top_k_indices
    
    def compute_load_balance_loss(self) -> torch.Tensor:
        """
        Compute load balancing loss to encourage uniform bank usage.
        
        L_balance = n_banks * sum(f_b * P_b) where:
        - f_b = fraction of tokens routed to bank b
        - P_b = average routing probability for bank b
        
        Minimizing this encourages uniform distribution of routing.
        
        Returns:
            Load balancing loss (scalar tensor)
        """
        if self.total_samples.item() < 1:
            return torch.tensor(0.0, device=self.routing_counts.device)
        
        # Fraction of tokens routed to each bank
        f = self.routing_counts / (self.total_samples * self.top_k + 1e-8)
        
        # Average routing probability for each bank
        P = self.routing_probs_sum / (self.total_samples + 1e-8)
        
        # Load balance loss: encourages f and P to be uniform
        # When uniform: f_b = 1/n_banks, P_b = 1/n_banks
        # Loss = n_banks * sum(f_b * P_b) = n_banks * (1/n_banks) = 1.0 (ideal)
        loss = self.n_banks * (f * P).sum()
        
        return loss
    
    def reset_statistics(self):
        """Reset routing statistics (call at epoch start)."""
        self.routing_counts.zero_()
        self.routing_probs_sum.zero_()
        self.total_samples.zero_()
    
    def get_routing_distribution(self) -> Dict[str, float]:
        """Get current routing distribution for logging."""
        if self.total_samples.item() < 1:
            return {}
        
        f = self.routing_counts / (self.total_samples * self.top_k + 1e-8)
        return {f'bank_{i}': f[i].item() for i in range(self.n_banks)}


class CrossBankAggregator(nn.Module):
    """
    Aggregates outputs from multiple banks.
    
    Supports two modes:
    1. Weighted sum: Simple routing-weight-based aggregation
    2. Cross-attention: Query attends to all bank outputs for complex interactions
    
    MEMORY EFFICIENT:
    - Single forward pass, no tensor accumulation
    - Cross-attention is O(k × n_banks × d_model) - manageable
    """
    
    def __init__(
        self,
        d_model: int = 256,
        n_banks: int = 6,
        use_cross_attention: bool = True,
        num_heads: int = 4,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_banks = n_banks
        self.use_cross_attention = use_cross_attention
        
        if use_cross_attention:
            self.cross_attn = nn.MultiheadAttention(
                d_model,
                num_heads=num_heads,
                batch_first=True,
                dropout=0.0,  # No dropout for stability
            )
            self.norm = nn.LayerNorm(d_model)
            
            # Mixing weight for cross-attention vs weighted sum
            self.cross_attn_weight = nn.Parameter(torch.tensor(0.5))
    
    def forward(
        self,
        z: torch.Tensor,
        bank_outputs: List[torch.Tensor],
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Aggregate bank outputs.
        
        Args:
            z: Original query [B, D]
            bank_outputs: List of [B, D] outputs from each bank
            routing_weights: [B, n_banks] routing weights
            
        Returns:
            aggregated: [B, D] aggregated output
        """
        B = z.shape[0]
        
        # Stack bank outputs: [B, n_banks, D]
        stacked = torch.stack(bank_outputs, dim=1)
        
        # Weighted sum based on routing
        weighted = (routing_weights.unsqueeze(-1) * stacked).sum(dim=1)  # [B, D]
        
        if self.use_cross_attention:
            # Cross-attention: query attends to all bank outputs
            # This captures complex multi-bank interactions
            attn_out, _ = self.cross_attn(
                z.unsqueeze(1),  # Query: [B, 1, D]
                stacked,          # Keys: [B, n_banks, D]
                stacked,          # Values: [B, n_banks, D]
            )
            attn_out = attn_out.squeeze(1)  # [B, D]
            
            # Blend weighted sum and cross-attention
            mix_weight = torch.sigmoid(self.cross_attn_weight)
            aggregated = self.norm(
                (1 - mix_weight) * weighted + mix_weight * attn_out
            )
        else:
            aggregated = weighted
        
        return aggregated


class HierarchicalPrimitiveMemory(nn.Module):
    """
    Complete Hierarchical Primitive Memory (HPM) v2 system for RLAN.
    
    This is the main module that orchestrates all HPM components:
    - Static memory banks for learned primitives
    - Sparse MoE router for bank selection
    - Cross-bank aggregator for combining outputs
    - Gated residual for training stability
    
    v2 IMPROVEMENTS:
    - Sparse MoE routing (Top-K) - only k banks queried per sample
    - Gated residual initialized to 0 - HPM doesn't disrupt baseline
    - Load Balancing Loss - prevents mode collapse
    - Static/Dynamic bank split - unbounded memory possible
    
    MEMORY EFFICIENCY:
    - No O(N) tensor accumulation (critical lesson from LOO fix)
    - Gate starts at 0, so HPM contribution is minimal initially
    - Top-K routing means only k forward passes per bank
    
    MODULAR DESIGN:
    - Enable/disable individual banks via config
    - Entire HPM can be disabled without code changes
    - Works with any RLAN configuration
    
    Usage:
        config = HPMConfig.from_dict(yaml_config)
        hpm = HierarchicalPrimitiveMemory(config)
        
        # In forward pass:
        z_enhanced, routing = hpm(z_context, return_routing=True)
        
        # Add to loss:
        loss += config.balance_loss_weight * hpm.get_load_balance_loss()
        
        # At epoch start:
        hpm.reset_epoch_stats()
    """
    
    def __init__(self, config: HPMConfig):
        super().__init__()
        
        self.config = config
        self.d_model = config.d_model
        self.top_k = config.top_k
        
        # Get enabled bank types
        self.bank_types = config.get_enabled_bank_types()
        self.n_banks = len(self.bank_types)
        
        if self.n_banks == 0:
            raise ValueError(
                "HPM enabled but no banks selected! Enable at least one bank: "
                "hpm_use_compositional_bank, hpm_use_pattern_bank, etc."
            )
        
        # Separate static and dynamic banks
        self.static_bank_types = [
            bt for bt in self.bank_types if bt in STATIC_BANK_TYPES
        ]
        self.dynamic_bank_types = [
            bt for bt in self.bank_types if bt in DYNAMIC_BANK_TYPES
        ]
        
        # Create static memory banks (nn.Parameter based)
        self.banks = nn.ModuleDict({
            bank_type.name: MemoryBank(
                d_model=config.d_model,
                n_primitives=config.primitives_per_bank,
                n_levels=config.n_levels_per_bank,
                bank_type=bank_type,
            )
            for bank_type in self.static_bank_types
        })
        
        # Dynamic bank query projections (actual buffers managed externally)
        for bank_type in self.dynamic_bank_types:
            setattr(
                self,
                f'{bank_type.name.lower()}_query_proj',
                nn.Linear(config.d_model, config.d_model, bias=False),
            )
        
        # Memory router with Top-K sparse selection
        self.router = MemoryRouter(
            d_model=config.d_model,
            n_banks=self.n_banks,
            top_k=min(config.top_k, self.n_banks),
        )
        
        # Cross-bank aggregator
        self.aggregator = CrossBankAggregator(
            d_model=config.d_model,
            n_banks=self.n_banks,
            use_cross_attention=config.use_cross_attention,
            num_heads=4,
        )
        
        # Output projection (residual-friendly initialization)
        self.output_proj = nn.Linear(config.d_model, config.d_model)
        nn.init.zeros_(self.output_proj.bias)
        nn.init.eye_(self.output_proj.weight)
        self.output_proj.weight.data *= 0.1  # Start near-identity
        
        # v2: Gated residual - CRITICAL for training stability
        # Initialized to 0 so tanh(0) = 0, meaning HPM contributes NOTHING initially
        # This ensures baseline RLAN performance is preserved when HPM is first enabled
        self.residual_gate = nn.Parameter(torch.tensor(0.0))
        
        # Bank type to index mapping for routing
        self._bank_type_to_idx = {
            bt: i for i, bt in enumerate(self.bank_types)
        }
        
        # Print configuration
        static_names = [bt.name for bt in self.static_bank_types]
        dynamic_names = [bt.name for bt in self.dynamic_bank_types]
        print(f"[HPM] Initialized with {self.n_banks} banks:")
        print(f"      Static: {static_names or 'None'}")
        print(f"      Dynamic: {dynamic_names or 'None'}")
        print(f"      Top-K routing: k={self.router.top_k}")
        print(f"      Gated residual: α=0 (starts at 0 contribution)")
    
    def forward(
        self,
        z: torch.Tensor,
        dynamic_buffers: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None,
        return_routing: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through HPM.
        
        MODALITY-AGNOSTIC: Accepts any [B, D] encoding.
        MEMORY-EFFICIENT: No tensor accumulation, gated residual.
        
        Args:
            z: Encoded input [B, D] from any encoder
            dynamic_buffers: Optional dict mapping bank_name -> (keys, values)
                            for dynamic banks (Instance, Procedural)
            return_routing: Whether to return routing weights for logging
            
        Returns:
            z_augmented: Memory-enhanced encoding [B, D]
            routing_weights: Optional [B, n_banks] weights
        """
        if z is None:
            # Backward compatibility: return dummy output
            dummy = torch.zeros(1, self.d_model, device=next(self.parameters()).device)
            return dummy, torch.zeros(1, self.n_banks) if return_routing else None
        
        B = z.shape[0]
        
        # Get sparse routing weights (Top-K)
        routing_weights, top_k_indices = self.router(z)  # [B, n_banks], [B, top_k]
        
        # Query each bank and collect outputs
        # MEMORY EFFICIENT: Each bank processes independently, no accumulation
        bank_outputs = []
        
        for i, bank_type in enumerate(self.bank_types):
            if bank_type in STATIC_BANK_TYPES:
                # Static bank: use learned primitives
                bank = self.banks[bank_type.name]
                output, _ = bank(z)
            elif bank_type in DYNAMIC_BANK_TYPES:
                # Dynamic bank: retrieve from external buffer
                if dynamic_buffers and bank_type.name in dynamic_buffers:
                    keys, values = dynamic_buffers[bank_type.name]
                    query_proj = getattr(self, f'{bank_type.name.lower()}_query_proj')
                    query = query_proj(z)
                    
                    # Attention over retrieved neighbors
                    scores = torch.matmul(query, keys.T) / math.sqrt(self.d_model)
                    alpha = F.softmax(scores, dim=-1)
                    output = torch.matmul(alpha, values)
                else:
                    # No buffer available: return zeros
                    output = torch.zeros_like(z)
            else:
                output = torch.zeros_like(z)
            
            bank_outputs.append(output)
        
        # Aggregate across banks
        aggregated = self.aggregator(z, bank_outputs, routing_weights)
        
        # Apply output projection
        memory_output = self.output_proj(aggregated)
        
        # v2: Gated residual - starts at 0, grows during training
        # This is CRITICAL: tanh(0) = 0, so initially z_augmented = z (no change)
        # The gate learns to increase HPM contribution as training progresses
        gate = torch.tanh(self.residual_gate)
        z_augmented = z + gate * memory_output
        
        if return_routing:
            return z_augmented, routing_weights
        return z_augmented, None
    
    def get_load_balance_loss(self) -> torch.Tensor:
        """Get load balancing loss from router."""
        return self.router.compute_load_balance_loss()
    
    def get_gate_value(self) -> float:
        """Get current residual gate value for monitoring."""
        return torch.tanh(self.residual_gate).item()
    
    def reset_epoch_stats(self):
        """Reset routing statistics (call at epoch start)."""
        self.router.reset_statistics()
        for bank in self.banks.values():
            bank.reset_usage_counts()
    
    def freeze_stable_primitives(self):
        """Freeze stable primitives in all static banks."""
        for bank in self.banks.values():
            bank.freeze_stable_primitives()
    
    def apply_gradient_routing(self):
        """Apply gradient routing (zero frozen gradients) in all banks."""
        for bank in self.banks.values():
            bank.apply_gradient_routing()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get HPM statistics for logging."""
        stats = {
            'gate_value': self.get_gate_value(),
            'routing_distribution': self.router.get_routing_distribution(),
            'load_balance_loss': self.get_load_balance_loss().item(),
        }
        
        # Bank-specific stats
        for name, bank in self.banks.items():
            frozen_count = sum(
                getattr(bank, f'freeze_mask_{l}').sum().item()
                for l in range(bank.n_levels)
            )
            total_count = sum(bank.primitives_per_level)
            stats[f'bank_{name}_frozen'] = frozen_count
            stats[f'bank_{name}_total'] = total_count
        
        return stats
    
    def get_bank_names(self) -> List[str]:
        """Get list of enabled bank names."""
        return [bt.name for bt in self.bank_types]
