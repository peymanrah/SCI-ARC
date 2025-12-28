"""
Dynamic Memory Buffer for HPM v2

This module provides unbounded memory storage for dynamic banks:
- Instance Bank: Stores ContextEncoder outputs from solved tasks
- Procedural Bank: Stores HyperLoRA latent codes from solved tasks

DESIGN PRINCIPLES:
1. CPU-based storage (doesn't consume GPU VRAM)
2. GPU-based retrieval (fast k-NN during forward pass)
3. Optional FAISS index for fast approximate nearest neighbor search
4. FIFO eviction when buffer is full

MEMORY EFFICIENCY:
- Keys/values stored on CPU (not GPU!)
- Only retrieved neighbors are moved to GPU
- FAISS index enables sub-linear retrieval time

CONTINUAL LEARNING:
- Buffer grows as tasks are solved
- Retrieved neighbors augment HPM output
- Enables retrieval-based reasoning for similar tasks

Usage:
    buffer = DynamicMemoryBuffer(d_model=256, max_size=10000)
    
    # After solving a task:
    buffer.add(z_context, z_task, task_id='task_001')
    
    # During forward pass:
    keys, values = buffer.retrieve(query, k=5)
    if keys is not None:
        # Use in HPM dynamic bank
        ...
"""

from __future__ import annotations

import warnings
from collections import deque
from typing import Optional, Tuple, List, Any

import torch
import numpy as np


class DynamicMemoryBuffer:
    """
    Manages dynamic memory buffer for Instance and Procedural banks.
    
    v2 DESIGN:
    - CPU-based storage (doesn't consume GPU VRAM during training)
    - GPU-based retrieval (fast k-NN at inference)
    - Optional FAISS index for O(log N) retrieval instead of O(N)
    - FIFO eviction when max_size is reached
    
    MEMORY EFFICIENCY:
    - Keys/values are stored as CPU tensors
    - Only k retrieved neighbors are moved to GPU per query
    - FAISS index adds ~4 bytes per vector overhead
    
    NOT AN nn.Module:
    - This is a pure Python class (no gradients)
    - Stored separately from model state
    - Can be saved/loaded independently
    """
    
    def __init__(
        self,
        d_model: int = 256,
        max_size: int = 10000,
        use_faiss: bool = True,
    ):
        """
        Initialize dynamic memory buffer.
        
        Args:
            d_model: Embedding dimension
            max_size: Maximum number of entries (FIFO eviction after)
            use_faiss: Whether to use FAISS for fast retrieval
        """
        self.d_model = d_model
        self.max_size = max_size
        self.use_faiss = use_faiss
        
        # Storage: using deque for O(1) FIFO eviction
        self._keys: deque = deque(maxlen=max_size)
        self._values: deque = deque(maxlen=max_size)
        self._task_ids: deque = deque(maxlen=max_size)
        
        # FAISS index (optional, for fast retrieval)
        self._faiss_index = None
        self._faiss_needs_rebuild = False
        
        if use_faiss:
            self._try_init_faiss()
    
    def _try_init_faiss(self):
        """Try to initialize FAISS index."""
        try:
            import faiss
            # Use Inner Product (cosine similarity after L2 normalization)
            self._faiss_index = faiss.IndexFlatIP(self.d_model)
            self.use_faiss = True
        except ImportError:
            warnings.warn(
                "[DynamicMemoryBuffer] FAISS not available, falling back to brute-force retrieval. "
                "Install with: pip install faiss-cpu (or faiss-gpu)"
            )
            self.use_faiss = False
            self._faiss_index = None
    
    def add(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        task_id: Optional[str] = None,
    ):
        """
        Add entry to buffer.
        
        MEMORY EFFICIENT: Tensors are detached and moved to CPU immediately.
        
        Args:
            key: Query key [D] or [B, D]
            value: Value to retrieve [D] or [B, D]
            task_id: Optional identifier for debugging/analysis
        """
        # Detach and move to CPU
        key = key.detach().cpu()
        value = value.detach().cpu()
        
        # Handle batch dimension
        if key.dim() == 1:
            key = key.unsqueeze(0)
            value = value.unsqueeze(0)
        
        B = key.shape[0]
        
        for i in range(B):
            # Add to storage (deque handles maxlen eviction)
            self._keys.append(key[i].clone())
            self._values.append(value[i].clone())
            self._task_ids.append(task_id)
            
            # Add to FAISS index
            if self.use_faiss and self._faiss_index is not None:
                # L2 normalize for cosine similarity via inner product
                key_np = key[i].numpy().astype(np.float32)
                key_np = key_np / (np.linalg.norm(key_np) + 1e-8)
                self._faiss_index.add(key_np.reshape(1, -1))
        
        # Mark FAISS for rebuild if we've evicted entries
        if len(self._keys) >= self.max_size:
            self._faiss_needs_rebuild = True
    
    def retrieve(
        self,
        query: torch.Tensor,
        k: int = 5,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Retrieve k nearest neighbors.
        
        MEMORY EFFICIENT: Only retrieved entries are moved to GPU.
        
        NOTE (Dec 2025): For batch queries [B, D], this returns neighbors based on
        the MEAN query across the batch. This is correct when:
        - All samples in the batch are from the same task (typical in ARC training)
        - You want shared context augmentation across the batch
        
        If you need per-sample retrieval (e.g., mixed-task batches), call this
        function B times with individual queries, or use retrieve_batch().
        
        Args:
            query: Query vector [B, D] or [D]
            k: Number of neighbors to retrieve
            
        Returns:
            keys: [k, D] nearest keys (on same device as query)
            values: [k, D] corresponding values (on same device as query)
            Returns (None, None) if buffer is empty
        """
        if len(self._keys) == 0:
            return None, None
        
        # Handle input shape
        query_device = query.device
        if query.dim() == 1:
            query = query.unsqueeze(0)
        
        # Use MEAN query for batch retrieval (robust to batch variations)
        # This is appropriate when batch contains samples from same task
        query_np = query.mean(dim=0).detach().cpu().numpy().astype(np.float32)
        
        # Limit k to buffer size
        k = min(k, len(self._keys))
        
        if self.use_faiss and self._faiss_index is not None:
            # Rebuild FAISS if needed (after evictions)
            if self._faiss_needs_rebuild:
                self._rebuild_faiss_index()
            
            # L2 normalize query for cosine similarity
            query_np = query_np / (np.linalg.norm(query_np) + 1e-8)
            
            # FAISS retrieval: O(log N) with IVF, O(N) with flat
            distances, indices = self._faiss_index.search(
                query_np.reshape(1, -1), k
            )
            indices = indices[0]  # [k]
            
            # Filter invalid indices (-1 from FAISS)
            valid_mask = indices >= 0
            indices = indices[valid_mask]
        else:
            # Brute-force retrieval: O(N)
            all_keys = torch.stack(list(self._keys), dim=0)  # [N, D]
            # Use mean query for consistency with FAISS path
            mean_query = query.mean(dim=0, keepdim=True).cpu()  # [1, D]
            scores = torch.matmul(
                mean_query, all_keys.T
            ).squeeze(0)  # [N]
            
            _, indices = torch.topk(scores, k)
            indices = indices.tolist()
        
        if len(indices) == 0:
            return None, None
        
        # Gather retrieved entries and move to query device
        ret_keys = torch.stack(
            [self._keys[i] for i in indices], dim=0
        ).to(query_device)
        
        ret_values = torch.stack(
            [self._values[i] for i in indices], dim=0
        ).to(query_device)
        
        return ret_keys, ret_values
    
    def _rebuild_faiss_index(self):
        """Rebuild FAISS index after evictions."""
        if not self.use_faiss or self._faiss_index is None:
            return
        
        import faiss
        
        # Reset index
        self._faiss_index.reset()
        
        # Re-add all current entries
        if len(self._keys) > 0:
            all_keys = torch.stack(list(self._keys), dim=0).numpy()
            # L2 normalize
            norms = np.linalg.norm(all_keys, axis=1, keepdims=True) + 1e-8
            all_keys = all_keys / norms
            self._faiss_index.add(all_keys.astype(np.float32))
        
        self._faiss_needs_rebuild = False
    
    def __len__(self) -> int:
        """Number of entries in buffer."""
        return len(self._keys)
    
    def clear(self):
        """Clear all entries."""
        self._keys.clear()
        self._values.clear()
        self._task_ids.clear()
        
        if self._faiss_index is not None:
            self._faiss_index.reset()
        
        self._faiss_needs_rebuild = False
    
    def get_task_ids(self) -> List[Optional[str]]:
        """Get list of task IDs in buffer."""
        return list(self._task_ids)
    
    def save(self, path: str):
        """Save buffer to disk."""
        state = {
            'd_model': self.d_model,
            'max_size': self.max_size,
            'keys': list(self._keys),
            'values': list(self._values),
            'task_ids': list(self._task_ids),
        }
        torch.save(state, path)
    
    @classmethod
    def load(cls, path: str, use_faiss: bool = True) -> 'DynamicMemoryBuffer':
        """Load buffer from disk."""
        state = torch.load(path)
        
        buffer = cls(
            d_model=state['d_model'],
            max_size=state['max_size'],
            use_faiss=use_faiss,
        )
        
        # Restore entries
        for key, value, task_id in zip(
            state['keys'], state['values'], state['task_ids']
        ):
            buffer._keys.append(key)
            buffer._values.append(value)
            buffer._task_ids.append(task_id)
        
        # Rebuild FAISS index
        if buffer.use_faiss and buffer._faiss_index is not None and len(buffer._keys) > 0:
            buffer._rebuild_faiss_index()
        
        return buffer
    
    def get_stats(self) -> dict:
        """Get buffer statistics."""
        return {
            'size': len(self._keys),
            'max_size': self.max_size,
            'fill_ratio': len(self._keys) / self.max_size,
            'unique_tasks': len(set(tid for tid in self._task_ids if tid is not None)),
            'uses_faiss': self.use_faiss and self._faiss_index is not None,
        }
