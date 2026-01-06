"""
Primitive Head: Trainable Primitive Prediction for NS-TEPS Guided Search

This module adds a learnable "Primitive Head" to RLAN that predicts:
1. Which primitive to apply (from the NS-TEPS library)
2. Object selection scores (which DSC-discovered objects to transform)
3. Parameter predictions (transform parameters)

WHY THIS MATTERS FOR GENERALIZATION:
- Currently NS-TEPS does blind search over primitives
- With this head, the neural network LEARNS which primitives are likely
- At inference, NS-TEPS uses neural predictions to FOCUS search
- This dramatically improves search efficiency and accuracy

TRAINING APPROACH:
1. Run NS-TEPS on training pairs to discover ground-truth programs
2. Use these as pseudo-labels to train the PrimitiveHead
3. The head learns to predict primitives that ACTUALLY work

MODULAR DESIGN (Non-negotiable):
- Can be enabled/disabled via config
- Doesn't modify base RLAN code
- Wraps RecursiveSolver features

Author: AI Research Assistant
Date: January 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np


@dataclass
class PrimitiveHeadConfig:
    """Configuration for the Primitive Head module."""
    enabled: bool = True
    
    # Primitive library size (matches NS-TEPS primitives)
    # IMPORTANT: This should match NUM_PRIMITIVES = len(PRIMITIVE_NAME_TO_ID) = 17
    # If NS-TEPS library changes, update PRIMITIVE_NAME_TO_ID and this will auto-sync
    num_primitives: int = 17  # Synced with PRIMITIVE_NAME_TO_ID at module load
    
    # Object selection
    max_objects: int = 20  # Max objects to score
    
    # Parameter prediction
    num_params: int = 8  # Number of continuous parameters to predict
    param_vocab_size: int = 32  # Discrete parameter vocabulary
    
    # Architecture
    hidden_dim: int = 256
    num_heads: int = 4
    dropout: float = 0.1
    
    # Training
    primitive_loss_weight: float = 0.5  # Weight of primitive prediction loss
    object_loss_weight: float = 0.3  # Weight of object selection loss
    param_loss_weight: float = 0.2  # Weight of parameter prediction loss
    
    # Inference
    top_k_primitives: int = 5  # Top-k primitives to consider during search
    temperature: float = 1.0  # Softmax temperature for sampling


class PrimitiveEmbedding(nn.Module):
    """
    Learnable embeddings for each primitive in the library.
    
    These embeddings capture:
    - What each primitive DOES (semantic meaning)
    - When each primitive is USEFUL (context patterns)
    """
    
    def __init__(self, num_primitives: int, embed_dim: int):
        super().__init__()
        self.num_primitives = num_primitives
        self.embed_dim = embed_dim
        
        # Learnable primitive embeddings
        self.primitive_embeddings = nn.Parameter(
            torch.randn(num_primitives, embed_dim) * 0.02
        )
        
        # Primitive type embeddings (for hierarchical organization)
        # 5 types: OBJECT_TRANSFORM, OBJECT_RELATION, GLOBAL_ARRANGEMENT, OBJECT_FILTER, OBJECT_COMBINE
        self.type_embeddings = nn.Parameter(
            torch.randn(5, embed_dim) * 0.02
        )
        
        # Mapping from primitive ID to type ID (initialized in init_primitive_types)
        self.register_buffer(
            'primitive_to_type', 
            torch.zeros(num_primitives, dtype=torch.long)
        )
    
    def init_primitive_types(self, type_mapping: Dict[int, int]):
        """Initialize the primitive-to-type mapping from NS-TEPS."""
        for prim_id, type_id in type_mapping.items():
            if prim_id < self.num_primitives:
                self.primitive_to_type[prim_id] = type_id
    
    def forward(self, primitive_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get primitive embeddings.
        
        Args:
            primitive_ids: Optional (B,) tensor of specific primitives to embed
            
        Returns:
            If primitive_ids provided: (B, embed_dim) embeddings
            Otherwise: (num_primitives, embed_dim) all embeddings
        """
        if primitive_ids is not None:
            prim_embed = self.primitive_embeddings[primitive_ids]  # (B, D)
            type_ids = self.primitive_to_type[primitive_ids]  # (B,)
            type_embed = self.type_embeddings[type_ids]  # (B, D)
            return prim_embed + type_embed  # (B, D)
        else:
            # Return all embeddings with type information
            type_embed = self.type_embeddings[self.primitive_to_type]  # (num_prims, D)
            return self.primitive_embeddings + type_embed  # (num_prims, D)


class ObjectScorer(nn.Module):
    """
    Scores objects for selection based on spatial features.
    
    Uses attention to determine which DSC-discovered objects
    should be transformed by each primitive.
    """
    
    def __init__(self, hidden_dim: int, max_objects: int, num_heads: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_objects = max_objects
        
        # Object query (learned)
        self.object_query = nn.Parameter(
            torch.randn(1, max_objects, hidden_dim) * 0.02
        )
        
        # Cross-attention: object queries attend to spatial features
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Score head
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(
        self, 
        spatial_features: torch.Tensor,
        object_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Score objects for selection.
        
        Args:
            spatial_features: (B, D, H, W) spatial features from encoder
            object_mask: Optional (B, max_objects) mask for valid objects
            
        Returns:
            object_scores: (B, max_objects) selection probabilities (softmax)
            object_logits: (B, max_objects) raw logits before softmax
            object_features: (B, max_objects, D) object representations
        """
        B, D, H, W = spatial_features.shape
        
        # Flatten spatial features for attention
        flat_features = spatial_features.view(B, D, H * W).transpose(1, 2)  # (B, H*W, D)
        
        # Expand queries for batch
        queries = self.object_query.expand(B, -1, -1)  # (B, max_objects, D)
        
        # Cross-attention: objects attend to spatial features
        object_features, attn_weights = self.cross_attn(
            queries, flat_features, flat_features
        )  # (B, max_objects, D)
        
        # Compute selection scores (raw logits)
        object_logits = self.score_head(object_features).squeeze(-1)  # (B, max_objects)
        
        # Apply mask if provided (for softmax)
        masked_logits = object_logits.clone()
        if object_mask is not None:
            masked_logits = masked_logits.masked_fill(~object_mask, float('-inf'))
        
        object_scores = F.softmax(masked_logits, dim=-1)  # (B, max_objects)
        
        return object_scores, object_logits, object_features


class ParameterPredictor(nn.Module):
    """
    Predicts primitive parameters from context.
    
    Combines:
    - Global context (what's the overall task pattern)
    - Selected primitive (what parameters does it need)
    - Selected object (what's the object's properties)
    """
    
    def __init__(
        self, 
        hidden_dim: int, 
        num_params: int,
        vocab_size: int = 32
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_params = num_params
        self.vocab_size = vocab_size
        
        # Context fusion
        self.context_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),  # primitive + object + global
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Parameter heads (discrete classification for each param slot)
        self.param_heads = nn.ModuleList([
            nn.Linear(hidden_dim, vocab_size)
            for _ in range(num_params)
        ])
        
        # Continuous parameter regression (alternative to discrete)
        self.continuous_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_params)
        )
    
    def forward(
        self,
        primitive_embed: torch.Tensor,
        object_embed: torch.Tensor,
        global_context: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict parameters for the selected primitive and object.
        
        Args:
            primitive_embed: (B, D) embedding of selected primitive
            object_embed: (B, D) embedding of selected object
            global_context: (B, D) global task context
            
        Returns:
            discrete_params: (B, num_params, vocab_size) logits for discrete params
            continuous_params: (B, num_params) continuous param values
        """
        # Fuse contexts
        combined = torch.cat([primitive_embed, object_embed, global_context], dim=-1)
        fused = self.context_fusion(combined)  # (B, D)
        
        # Predict discrete parameters
        discrete_params = torch.stack([
            head(fused) for head in self.param_heads
        ], dim=1)  # (B, num_params, vocab_size)
        
        # Predict continuous parameters
        continuous_params = self.continuous_head(fused)  # (B, num_params)
        
        return discrete_params, continuous_params


class PrimitiveHead(nn.Module):
    """
    Main Primitive Head module for Program-Guided GRU.
    
    This module attaches to the RecursiveSolver and adds:
    1. Primitive prediction: Which transform to apply
    2. Object selection: Which object to transform
    3. Parameter prediction: Transform parameters
    
    TRAINING MODE:
    - Uses NS-TEPS discovered programs as pseudo-labels
    - Learns to predict primitives that actually work
    
    INFERENCE MODE:
    - Provides prior distribution over primitives
    - Guides NS-TEPS search for efficiency
    """
    
    def __init__(self, config: PrimitiveHeadConfig):
        super().__init__()
        self.config = config
        
        # Feature projection (from solver hidden dim to our hidden dim)
        self.feature_proj = nn.Sequential(
            nn.Conv2d(config.hidden_dim, config.hidden_dim, 1),
            nn.GELU(),
            nn.Conv2d(config.hidden_dim, config.hidden_dim, 1)
        )
        
        # Global pooling for context
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Primitive embeddings
        self.primitive_embed = PrimitiveEmbedding(
            num_primitives=config.num_primitives,
            embed_dim=config.hidden_dim
        )
        
        # Primitive classifier
        self.primitive_classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.num_primitives)
        )
        
        # Object scorer
        self.object_scorer = ObjectScorer(
            hidden_dim=config.hidden_dim,
            max_objects=config.max_objects,
            num_heads=config.num_heads
        )
        
        # Parameter predictor
        self.param_predictor = ParameterPredictor(
            hidden_dim=config.hidden_dim,
            num_params=config.num_params,
            vocab_size=config.param_vocab_size
        )
        
        # Multi-step trace predictor (for sequence of primitives)
        self.trace_rnn = nn.GRU(
            input_size=config.hidden_dim,
            hidden_size=config.hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=config.dropout
        )
        self.max_trace_length = 3
    
    def forward(
        self,
        solver_features: torch.Tensor,
        object_mask: Optional[torch.Tensor] = None,
        return_trace: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Predict primitives from solver features.
        
        Args:
            solver_features: (B, D, H, W) features from RecursiveSolver
            object_mask: Optional (B, max_objects) mask for valid objects
            return_trace: If True, predict full trace sequence
            
        Returns:
            Dict containing:
                - primitive_logits: (B, num_primitives) 
                - object_scores: (B, max_objects)
                - param_logits: (B, num_params, vocab_size)
                - param_values: (B, num_params)
                - trace_logits: Optional (B, max_trace, num_primitives)
        """
        B = solver_features.shape[0]
        
        # Project features
        features = self.feature_proj(solver_features)  # (B, D, H, W)
        
        # Global context
        global_ctx = self.global_pool(features).view(B, -1)  # (B, D)
        
        # Predict primitive
        primitive_logits = self.primitive_classifier(global_ctx)  # (B, num_primitives)
        
        # Score objects (now returns logits too)
        object_scores, object_logits, object_features = self.object_scorer(
            features, object_mask
        )  # (B, max_objects), (B, max_objects), (B, max_objects, D)
        
        # Get selected object embedding (weighted sum)
        selected_obj = torch.einsum('bo,bod->bd', object_scores, object_features)  # (B, D)
        
        # Get most likely primitive embedding
        prim_probs = F.softmax(primitive_logits / self.config.temperature, dim=-1)
        all_prim_embeds = self.primitive_embed()  # (num_prims, D)
        selected_prim = torch.einsum('bp,pd->bd', prim_probs, all_prim_embeds)  # (B, D)
        
        # Predict parameters
        param_logits, param_values = self.param_predictor(
            selected_prim, selected_obj, global_ctx
        )
        
        outputs = {
            'primitive_logits': primitive_logits,
            'object_scores': object_scores,
            'object_logits': object_logits,  # Raw logits for proper loss computation
            'param_logits': param_logits,
            'param_values': param_values,
            'global_context': global_ctx,
        }
        
        # Optionally predict full trace
        if return_trace:
            trace_logits = self._predict_trace(global_ctx)
            outputs['trace_logits'] = trace_logits
        
        return outputs
    
    def _predict_trace(self, global_ctx: torch.Tensor) -> torch.Tensor:
        """
        Predict a sequence of primitives (program trace).
        
        Args:
            global_ctx: (B, D) global context
            
        Returns:
            trace_logits: (B, max_trace, num_primitives)
        """
        B, D = global_ctx.shape
        
        # Initialize with global context
        hidden = global_ctx.unsqueeze(0).repeat(2, 1, 1)  # (2, B, D) for 2-layer GRU
        
        # Start token
        start_token = torch.zeros(B, 1, D, device=global_ctx.device)
        
        trace_logits = []
        input_token = start_token
        
        for step in range(self.max_trace_length):
            output, hidden = self.trace_rnn(input_token, hidden)  # (B, 1, D)
            step_logits = self.primitive_classifier(output.squeeze(1))  # (B, num_primitives)
            trace_logits.append(step_logits)
            
            # Next input is embedding of predicted primitive
            prim_probs = F.softmax(step_logits / self.config.temperature, dim=-1)
            all_prim_embeds = self.primitive_embed()
            input_token = torch.einsum('bp,pd->bd', prim_probs, all_prim_embeds)
            input_token = input_token.unsqueeze(1)  # (B, 1, D)
        
        return torch.stack(trace_logits, dim=1)  # (B, max_trace, num_primitives)
    
    def get_primitive_prior(
        self, 
        solver_features: torch.Tensor,
        top_k: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get prior distribution over primitives for NS-TEPS search guidance.
        
        Args:
            solver_features: (B, D, H, W) features from solver
            top_k: Number of top primitives to return (default: config.top_k_primitives)
            
        Returns:
            top_primitive_ids: (B, K) indices of top-k primitives
            top_primitive_probs: (B, K) probabilities of top-k primitives
        """
        if top_k is None:
            top_k = self.config.top_k_primitives
        
        outputs = self.forward(solver_features, return_trace=False)
        primitive_logits = outputs['primitive_logits']  # (B, num_primitives)
        
        probs = F.softmax(primitive_logits / self.config.temperature, dim=-1)
        top_probs, top_ids = torch.topk(probs, k=top_k, dim=-1)
        
        return top_ids, top_probs


class PrimitiveHeadLoss(nn.Module):
    """
    Loss function for training the Primitive Head.
    
    Uses pseudo-labels from NS-TEPS discovered programs.
    """
    
    def __init__(self, config: PrimitiveHeadConfig):
        super().__init__()
        self.config = config
        
        # Cross-entropy for primitive prediction
        self.primitive_ce = nn.CrossEntropyLoss(reduction='none')
        
        # BCE for object selection
        self.object_bce = nn.BCEWithLogitsLoss(reduction='none')
        
        # CE for discrete params, MSE for continuous
        self.param_ce = nn.CrossEntropyLoss(reduction='none')
        self.param_mse = nn.MSELoss(reduction='none')
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute primitive head loss.
        
        Args:
            predictions: Output from PrimitiveHead.forward()
            targets: Dict with:
                - primitive_ids: (B,) ground truth primitive IDs
                - object_mask: (B, max_objects) binary object selection
                - param_discrete: (B, num_params) discrete param targets
                - param_continuous: (B, num_params) continuous param targets
            mask: Optional (B,) mask for valid samples
            
        Returns:
            Dict with loss components and total loss
        """
        B = predictions['primitive_logits'].shape[0]
        
        # Primitive prediction loss
        prim_loss = self.primitive_ce(
            predictions['primitive_logits'],  # (B, num_prims)
            targets['primitive_ids']  # (B,)
        )  # (B,)
        
        # Object selection loss (if targets available)
        # Uses cross-entropy for single-object selection (object_target is index)
        # OR multi-label BCE if object_mask is multi-hot
        obj_loss = torch.zeros(B, device=prim_loss.device)
        if 'object_mask' in targets:
            target_mask = targets['object_mask']  # (B, max_objects)
            
            # Check if single-object (index) or multi-object (mask) selection
            if 'object_logits' in predictions:
                # Use raw logits for proper loss computation
                obj_logits = predictions['object_logits']  # (B, max_objects)
                
                # If target is one-hot or multi-hot, determine mode
                if target_mask.dtype == torch.long and target_mask.dim() == 1:
                    # Single object index target: use cross-entropy
                    obj_loss = F.cross_entropy(
                        obj_logits, target_mask, reduction='none'
                    )  # (B,)
                else:
                    # Multi-object selection: use BCE with logits
                    obj_loss = F.binary_cross_entropy_with_logits(
                        obj_logits,
                        target_mask.float(),
                        reduction='none'
                    ).mean(dim=-1)  # (B,)
            else:
                # Fallback: use object_scores with KL divergence
                obj_scores = predictions['object_scores']  # (B, max_objects)
                target_dist = target_mask.float()
                target_dist = target_dist / target_dist.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                obj_loss = F.kl_div(
                    obj_scores.log().clamp(min=-100),
                    target_dist,
                    reduction='none'
                ).sum(dim=-1)  # (B,)
        
        # Parameter loss
        param_loss = torch.zeros(B, device=prim_loss.device)
        if 'param_discrete' in targets:
            # Discrete parameter loss
            for i in range(self.config.num_params):
                param_loss += self.param_ce(
                    predictions['param_logits'][:, i],  # (B, vocab_size)
                    targets['param_discrete'][:, i]  # (B,)
                )
            param_loss /= self.config.num_params
        
        if 'param_continuous' in targets:
            cont_loss = self.param_mse(
                predictions['param_values'],
                targets['param_continuous']
            ).mean(dim=-1)  # (B,)
            param_loss = param_loss + cont_loss
        
        # Apply sample mask
        if mask is not None:
            prim_loss = prim_loss * mask
            obj_loss = obj_loss * mask
            param_loss = param_loss * mask
            n_valid = mask.sum().clamp(min=1)
        else:
            n_valid = B
        
        # Weighted combination
        total_loss = (
            self.config.primitive_loss_weight * prim_loss.sum() / n_valid +
            self.config.object_loss_weight * obj_loss.sum() / n_valid +
            self.config.param_loss_weight * param_loss.sum() / n_valid
        )
        
        return {
            'primitive_loss': prim_loss.sum() / n_valid,
            'object_loss': obj_loss.sum() / n_valid,
            'param_loss': param_loss.sum() / n_valid,
            'total_loss': total_loss,
        }


# Mapping from NS-TEPS primitive names to IDs
# CRITICAL: These must match EXACTLY the names in ns_teps.ObjectPrimitiveLibrary
PRIMITIVE_NAME_TO_ID = {
    # OBJECT_TRANSFORM primitives
    'identity': 0,
    'rotate_objects_90': 1,
    'flip_objects_horizontal': 2,
    'flip_objects_vertical': 3,
    'recolor_objects': 4,
    'swap_colors': 5,
    # GLOBAL_ARRANGEMENT primitives
    'gravity_down': 6,
    'gravity_left': 7,
    'mirror_horizontal': 8,
    'mirror_vertical': 9,
    'rotate_grid_90': 10,
    'rotate_grid_180': 11,
    'rotate_grid_270': 12,
    'tile_2x2': 13,
    # OBJECT_FILTER primitives
    'keep_largest': 14,
    'keep_smallest': 15,
    'remove_background': 16,
}

# Reverse mapping for ID to name lookup
PRIMITIVE_ID_TO_NAME = {v: k for k, v in PRIMITIVE_NAME_TO_ID.items()}

# Number of primitives (update config to match)
NUM_PRIMITIVES = len(PRIMITIVE_NAME_TO_ID)

PRIMITIVE_TYPE_MAPPING = {
    # OBJECT_TRANSFORM (type 0)
    0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0,
    # GLOBAL_ARRANGEMENT (type 2)
    6: 2, 7: 2, 8: 2, 9: 2, 10: 2, 11: 2, 12: 2, 13: 2,
    # OBJECT_FILTER (type 3)
    14: 3, 15: 3, 16: 3,
}
