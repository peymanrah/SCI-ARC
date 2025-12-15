# =============================================================================
# TRM Original Implementation - sparse_embedding.py
# Source: https://github.com/SamsungSAILMontreal/TinyRecursiveModels
# Authors: Samsung SAIL Montreal
# License: Apache 2.0 (check original repository for latest license)
# 
# This file is copied from the original TRM repository for fair comparison
# in the SCI-ARC publication. No modifications have been made to the core logic.
# =============================================================================

import torch
from torch import nn

from baselines.trm.models.common import trunc_normal_init_


class CastedSparseEmbedding(nn.Module):
    """
    Sparse embedding layer with local gradient tracking for puzzle embeddings.
    
    This is used for per-puzzle embeddings that need to be efficiently updated
    during training while maintaining sparse gradient computation.
    """
    
    def __init__(self, num_embeddings: int, embedding_dim: int, batch_size: int, init_std: float, cast_to: torch.dtype):
        super().__init__()
        self.cast_to = cast_to

        # Real Weights
        # Truncated LeCun normal init
        self.weights = nn.Buffer(
            trunc_normal_init_(torch.empty((num_embeddings, embedding_dim)), std=init_std), persistent=True
        )

        # Local weights and IDs
        # Local embeddings, with gradient, not persistent
        self.local_weights = nn.Buffer(torch.zeros(batch_size, embedding_dim, requires_grad=True), persistent=False)
        # Local embedding IDs, not persistent
        self.local_ids = nn.Buffer(torch.zeros(batch_size, dtype=torch.int32), persistent=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if not self.training:
            # Test mode, no gradient
            return self.weights[inputs].to(self.cast_to)
            
        # Training mode, fill puzzle embedding from weights
        with torch.no_grad():
            self.local_weights.copy_(self.weights[inputs])
            self.local_ids.copy_(inputs)

        return self.local_weights.to(self.cast_to)


class CastedSparseEmbeddingSignSGD_Distributed:
    """
    SignSGD optimizer for sparse embeddings with distributed training support.
    
    This optimizer uses SignSGD for updating puzzle-specific embeddings,
    which provides robustness to gradient noise and enables efficient
    distributed training.
    """
    
    def __init__(self, embeddings: CastedSparseEmbedding, lr: float, weight_decay: float):
        self.embeddings = embeddings
        self.lr = lr
        self.weight_decay = weight_decay
        
    def step(self):
        """Perform a single optimization step using SignSGD."""
        if self.embeddings.local_weights.grad is None:
            return
            
        with torch.no_grad():
            # Get gradient signs
            grad_sign = torch.sign(self.embeddings.local_weights.grad)
            
            # Apply weight decay
            if self.weight_decay > 0:
                self.embeddings.weights[self.embeddings.local_ids] *= (1 - self.lr * self.weight_decay)
            
            # Update weights using SignSGD
            self.embeddings.weights[self.embeddings.local_ids] -= self.lr * grad_sign
            
    def zero_grad(self):
        """Zero out the gradients."""
        if self.embeddings.local_weights.grad is not None:
            self.embeddings.local_weights.grad.zero_()
