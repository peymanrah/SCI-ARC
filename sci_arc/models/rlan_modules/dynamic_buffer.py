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

============================================================================
TODO 9: MEMORY GRANULARITY POLICY (Dec 2025)
============================================================================
This documents the scientific policy decisions for HPM memory storage:

1. SOLVED CRITERION: Store only EXACT MATCH solves (100% pixel accuracy)
   - Rationale: High-quality memories prevent noise from partial solutions
   - Alternative: >=90% accuracy would capture "almost correct" solutions
   - Current: exact_match (set via export_hpm_memory's solved_criterion param)

2. STORAGE GRANULARITY: Per-TASK (not per-sample)
   - Rationale: Multiple augmentations of same task are redundant
   - Implementation: Global dedup via hpm_buffer_contains_task() + task_id
   - Growth: O(unique_tasks), not O(samples)

3. DUPLICATE POLICY: First-seen-wins
   - Rationale: Later epochs may have different training dynamics
   - Alternative: Last-N per task (requires explicit eviction policy)
   - Current: First embedding stored, duplicates skipped

4. MAX CAPACITY: 10,000 entries (configurable via hpm_memory_size)
   - Rationale: Sufficient for 25x ARC training set (400 tasks)
   - Eviction: FIFO when full (oldest tasks evicted first)

5. EPOCH SCOPE: Cross-epoch persistence
   - Rationale: Continual learning requires memory across training
   - Serialization: Saved in checkpoints and standalone exports
============================================================================

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
        self._last_eviction_count = 0  # Track evictions for rebuild logic
        
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
            # Track if eviction will happen (deque is full before adding)
            was_at_max = len(self._keys) >= self.max_size
            
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
            
            # Mark FAISS for rebuild only when actual eviction happens
            # (not just when buffer is full, but when an old entry was pushed out)
            if was_at_max:
                self._last_eviction_count += 1
                self._faiss_needs_rebuild = True
    
    def retrieve(
        self,
        query: torch.Tensor,
        k: int = 5,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[dict]]:
        """
        Retrieve k nearest neighbors using cosine similarity.
        
        MEMORY EFFICIENT: Only retrieved entries are moved to GPU.
        
        SIMILARITY METRIC (Dec 2025 FIX): Both FAISS and brute-force paths now use
        cosine similarity (L2-normalized dot product) for reproducibility across
        environments. Keys are normalized at add-time, queries normalized here.
        
        BATCH HANDLING (Dec 2025 FIX): For batch queries [B, D], uses the FIRST
        sample's query (not mean) to avoid cross-task averaging when batches mix
        tasks. This is safer for bucketed batching by grid size.
        
        Args:
            query: Query vector [B, D] or [D]
            k: Number of neighbors to retrieve
            
        Returns:
            keys: [k, D] nearest keys (on same device as query)
            values: [k, D] corresponding values (on same device as query)
            stats: Optional dict with retrieval statistics for debugging
            Returns (None, None, None) if buffer is empty
        """
        if len(self._keys) == 0:
            return None, None, {'retrieved': 0, 'buffer_size': 0}
        
        # Handle input shape
        query_device = query.device
        if query.dim() == 1:
            query = query.unsqueeze(0)
        
        # FIX (Dec 2025): Use FIRST sample query instead of mean to avoid
        # cross-task averaging when batches contain mixed tasks (bucketed by grid size)
        query_vec = query[0].detach().cpu()  # [D]
        query_np = query_vec.numpy().astype(np.float32)
        
        # L2 normalize query for cosine similarity (BOTH paths use this now)
        query_norm = np.linalg.norm(query_np) + 1e-8
        query_np_normalized = query_np / query_norm
        
        # Limit k to buffer size
        k = min(k, len(self._keys))
        
        # Track retrieval stats for debugging
        stats = {
            'buffer_size': len(self._keys),
            'requested_k': k,
            'query_norm': float(query_norm),
        }
        
        if self.use_faiss and self._faiss_index is not None:
            # Rebuild FAISS if needed (after evictions)
            if self._faiss_needs_rebuild:
                self._rebuild_faiss_index()
            
            # FAISS retrieval: O(log N) with IVF, O(N) with flat
            # Keys are already L2-normalized at add-time
            distances, indices = self._faiss_index.search(
                query_np_normalized.reshape(1, -1), k
            )
            indices = indices[0]  # [k]
            similarities = distances[0]  # [k] - inner product = cosine sim for normalized vectors
            
            # Filter invalid indices (-1 from FAISS)
            valid_mask = indices >= 0
            indices = indices[valid_mask]
            similarities = similarities[valid_mask]
            
            stats['method'] = 'faiss'
            stats['avg_similarity'] = float(np.mean(similarities)) if len(similarities) > 0 else 0.0
        else:
            # Brute-force retrieval: O(N)
            # FIX (Dec 2025): Normalize keys for cosine similarity to match FAISS behavior
            all_keys = torch.stack(list(self._keys), dim=0)  # [N, D]
            all_keys_normalized = all_keys / (all_keys.norm(dim=1, keepdim=True) + 1e-8)
            
            # Compute cosine similarity (normalized dot product)
            query_vec_normalized = query_vec / (query_vec.norm() + 1e-8)
            scores = torch.matmul(
                query_vec_normalized.unsqueeze(0), all_keys_normalized.T
            ).squeeze(0)  # [N]
            
            top_scores, top_indices = torch.topk(scores, k)
            indices = top_indices.tolist()
            
            stats['method'] = 'brute_force'
            stats['avg_similarity'] = float(top_scores.mean().item()) if len(top_scores) > 0 else 0.0
        
        if len(indices) == 0:
            stats['retrieved'] = 0
            return None, None, stats
        
        stats['retrieved'] = len(indices)
        
        # Gather retrieved entries and move to query device
        ret_keys = torch.stack(
            [self._keys[i] for i in indices], dim=0
        ).to(query_device)
        
        ret_values = torch.stack(
            [self._values[i] for i in indices], dim=0
        ).to(query_device)
        
        return ret_keys, ret_values, stats
    
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
    
    def contains_task(self, task_id: str) -> bool:
        """Check if a task_id already exists in the buffer (for global dedup).
        
        Args:
            task_id: Task identifier to check
            
        Returns:
            True if task_id is already in buffer
        """
        return task_id in self._task_ids
    
    def get_unique_task_ids(self) -> set:
        """Get set of unique task IDs in buffer."""
        return set(tid for tid in self._task_ids if tid is not None)
    
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
            'eviction_count': getattr(self, '_last_eviction_count', 0),
        }
    
    def retrieve_batch(
        self,
        queries: torch.Tensor,
        k: int = 5,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[dict]]:
        """
        Retrieve k nearest neighbors for each sample in a batch.
        
        This is the correct approach for mixed-task batches where each sample
        may need different neighbors. Uses batch search when FAISS is available.
        
        Args:
            queries: Query vectors [B, D]
            k: Number of neighbors per query
            
        Returns:
            keys: [B, k, D] nearest keys per sample (on same device as queries)
            values: [B, k, D] corresponding values per sample
            stats: Dict with aggregated retrieval statistics
            Returns (None, None, stats) if buffer is empty
        """
        if len(self._keys) == 0:
            return None, None, {'retrieved': 0, 'buffer_size': 0, 'batch_size': queries.shape[0]}
        
        query_device = queries.device
        if queries.dim() == 1:
            queries = queries.unsqueeze(0)
        
        B = queries.shape[0]
        k = min(k, len(self._keys))
        
        # Move queries to CPU and normalize
        queries_cpu = queries.detach().cpu()  # [B, D]
        queries_np = queries_cpu.numpy().astype(np.float32)
        query_norms = np.linalg.norm(queries_np, axis=1, keepdims=True) + 1e-8
        queries_normalized = queries_np / query_norms
        
        stats = {
            'buffer_size': len(self._keys),
            'batch_size': B,
            'requested_k': k,
        }
        
        all_indices = []
        all_similarities = []
        
        if self.use_faiss and self._faiss_index is not None:
            # Rebuild FAISS if needed
            if self._faiss_needs_rebuild:
                self._rebuild_faiss_index()
            
            # Batch FAISS search: [B, k]
            distances, indices = self._faiss_index.search(queries_normalized, k)
            
            stats['method'] = 'faiss_batch'
            all_indices = indices  # [B, k]
            all_similarities = distances  # [B, k]
        else:
            # Brute-force batch retrieval
            all_keys = torch.stack(list(self._keys), dim=0)  # [N, D]
            all_keys_normalized = all_keys / (all_keys.norm(dim=1, keepdim=True) + 1e-8)
            
            queries_normalized_t = queries_cpu / (queries_cpu.norm(dim=1, keepdim=True) + 1e-8)
            
            # [B, N] similarity matrix
            scores = torch.matmul(queries_normalized_t, all_keys_normalized.T)
            
            # Top-k per query: [B, k]
            top_scores, top_indices = torch.topk(scores, k, dim=1)
            all_indices = top_indices.numpy()
            all_similarities = top_scores.numpy()
            
            stats['method'] = 'brute_force_batch'
        
        # Compute per-sample and aggregate similarity stats (TODO 6: Improved logging)
        valid_sims = all_similarities[all_indices >= 0]
        stats['avg_similarity'] = float(np.mean(valid_sims)) if len(valid_sims) > 0 else 0.0
        stats['retrieved'] = int(np.sum(all_indices >= 0))
        
        # Per-sample stats for debugging mixed-task batches
        if B > 1:
            per_sample_sims = []
            for b in range(B):
                sample_sims = all_similarities[b][all_indices[b] >= 0]
                if len(sample_sims) > 0:
                    per_sample_sims.append(float(np.mean(sample_sims)))
            if per_sample_sims:
                stats['min_similarity'] = float(min(per_sample_sims))
                stats['max_similarity'] = float(max(per_sample_sims))
                stats['median_similarity'] = float(np.median(per_sample_sims))
        
        # Gather results: [B, k, D]
        ret_keys = torch.zeros(B, k, self.d_model, device=query_device)
        ret_values = torch.zeros(B, k, self.d_model, device=query_device)
        
        for b in range(B):
            for j in range(k):
                idx = all_indices[b, j]
                if idx >= 0 and idx < len(self._keys):
                    ret_keys[b, j] = self._keys[idx].to(query_device)
                    ret_values[b, j] = self._values[idx].to(query_device)
        
        return ret_keys, ret_values, stats
    
    def state_dict(self) -> dict:
        """
        Get canonical state dict for serialization.
        
        This is the single source of truth for HPM buffer persistence.
        Use this for both training checkpoints and inference export.
        
        Returns:
            State dict with all buffer data and metadata
        """
        return {
            'd_model': self.d_model,
            'max_size': self.max_size,
            'keys': list(self._keys),
            'values': list(self._values),
            'task_ids': list(self._task_ids),
            'version': '2.0',  # For future format changes
            'eviction_count': getattr(self, '_last_eviction_count', 0),
        }
    
    def load_state_dict(self, state: dict) -> None:
        """
        Load state dict into buffer.
        
        Args:
            state: State dict from state_dict() or checkpoint
        """
        # Clear existing data
        self.clear()
        
        # Validate dimensions
        if state.get('d_model', self.d_model) != self.d_model:
            warnings.warn(
                f"[DynamicMemoryBuffer] d_model mismatch: buffer={self.d_model}, "
                f"state={state.get('d_model')}. Entries may be incompatible."
            )
        
        # Restore entries
        for key, value, task_id in zip(
            state.get('keys', []),
            state.get('values', []),
            state.get('task_ids', [])
        ):
            self._keys.append(key)
            self._values.append(value)
            self._task_ids.append(task_id)
        
        # Restore eviction count if present
        self._last_eviction_count = state.get('eviction_count', 0)
        
        # Rebuild FAISS index
        if self.use_faiss and self._faiss_index is not None and len(self._keys) > 0:
            self._rebuild_faiss_index()
