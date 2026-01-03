#!/usr/bin/env python3
"""
RLAN Visual Debugger - Standalone Script
=========================================

This script is INDEPENDENT from the RLAN codebase and can be run standalone.
It loads a trained RLAN checkpoint and generates an interactive HTML visualization
showing step-by-step refinement and module contributions for debugging.

=============================================================================
COMMAND EXAMPLES - ABLATION TESTS
=============================================================================

# --- BASIC USAGE ---
# Visualize a single task (opens HTML in browser)
python scripts/visualize_rlan_debug.py --task_id 00d62c1b --checkpoint checkpoints/rlan_stable_256/best.pt

# Specify config explicitly (auto-detected from checkpoint if omitted)
python scripts/visualize_rlan_debug.py --task_id 00d62c1b --checkpoint checkpoints/rlan_stable_512/best.pt --config configs/rlan_stable_dev_512.yaml

# --- BATCH MODE ---
# Visualize multiple tasks from a file (one task_id per line)
python scripts/visualize_rlan_debug.py --task_file failing_tasks.txt --checkpoint best.pt --output_dir ./debug_viz

# --- MODULE ABLATION TESTS ---
# Test DSC (Dynamic Saliency Controller) - shows clue selection impact
python scripts/visualize_rlan_debug.py --task_id 00d62c1b --checkpoint best.pt --test_dsc

# Test HyperLoRA - shows task-specific weight adaptation impact
python scripts/visualize_rlan_debug.py --task_id 00d62c1b --checkpoint best.pt --test_hyperlora

# Test Solver iterations - compares different step counts
python scripts/visualize_rlan_debug.py --task_id 00d62c1b --checkpoint best.pt --test_solver

# Run ALL ablations (DSC + HyperLoRA + Solver + Context)
python scripts/visualize_rlan_debug.py --task_id 00d62c1b --checkpoint best.pt --ablations

# --- SOLVER STEP OVERRIDE ---
# Test with more solver iterations than trained (e.g., trained=7, test=10)
python scripts/visualize_rlan_debug.py --task_id 00d62c1b --checkpoint best.pt --num_steps 10

# --- TTA (TEST-TIME AUGMENTATION) ---
# Run with TTA voting across dihedral transforms
python scripts/visualize_rlan_debug.py --task_id 00d62c1b --checkpoint best.pt --use_tta

# --- OUTPUT OPTIONS ---
# Save HTML to specific file instead of auto-opening
python scripts/visualize_rlan_debug.py --task_id 00d62c1b --checkpoint best.pt --output task_debug.html

# --- EVALUATION SET ---
# Debug a task from the evaluation set (not training)
python scripts/visualize_rlan_debug.py --task_id 12345abc --checkpoint best.pt --data_path ./data/arc-agi/data/evaluation

# --- COMBINING FLAGS ---
# Full debug: ablations + TTA + 10 steps + save output
python scripts/visualize_rlan_debug.py --task_id 00d62c1b --checkpoint best.pt --ablations --use_tta --num_steps 10 --output full_debug.html

# Compare 256 vs 512 capacity on same task:
# Terminal 1:
python scripts/visualize_rlan_debug.py --task_id 00d62c1b --checkpoint checkpoints/rlan_stable_256/best.pt --output debug_256.html
# Terminal 2:
python scripts/visualize_rlan_debug.py --task_id 00d62c1b --checkpoint checkpoints/rlan_stable_512/best.pt --output debug_512.html

=============================================================================

Author: RLAN Debug Team
Date: January 2026
"""

import argparse
import json
import os
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import html
import base64
from io import BytesIO

# Add parent to path for imports (but we minimize actual imports)
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn.functional as F
import numpy as np

# Only import what we absolutely need from the codebase
# Everything else is self-contained in this script


# =============================================================================
# CONSTANTS
# =============================================================================

ARC_COLORS = [
    '#000000',  # 0: black
    '#0074D9',  # 1: blue
    '#FF4136',  # 2: red
    '#2ECC40',  # 3: green
    '#FFDC00',  # 4: yellow
    '#AAAAAA',  # 5: grey
    '#F012BE',  # 6: magenta
    '#FF851B',  # 7: orange
    '#7FDBFF',  # 8: cyan
    '#870C25',  # 9: brown
    '#FFFFFF',  # 10: white (padding)
]


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class StepTrace:
    """Trace data for a single refinement step."""
    step_idx: int
    predicted_grid: np.ndarray
    logits_entropy: float
    pixel_accuracy: float
    fg_accuracy: float
    bg_accuracy: float
    diff_mask: Optional[np.ndarray] = None  # Where prediction differs from target
    
    # DSC info (if available)
    attention_maps: Optional[np.ndarray] = None  # (K, H, W)
    centroids: Optional[np.ndarray] = None  # (K, 2)
    stop_probs: Optional[np.ndarray] = None  # (K,)
    estimated_clues: float = 0.0
    centroid_spread: float = 0.0
    
    # HyperLoRA info (if available)
    lora_delta_norms: Optional[Dict[str, float]] = None


@dataclass
class RunTrace:
    """Complete trace for one forward pass configuration."""
    config_name: str
    steps: List[StepTrace] = field(default_factory=list)
    final_prediction: Optional[np.ndarray] = None
    final_accuracy: float = 0.0
    is_exact_match: bool = False
    
    # Module status
    dsc_enabled: bool = False
    hyperlora_enabled: bool = False
    solver_context_enabled: bool = False
    hpm_enabled: bool = False


@dataclass
class TaskVisualization:
    """All data for visualizing one task."""
    task_id: str
    train_inputs: List[np.ndarray]
    train_outputs: List[np.ndarray]
    test_input: np.ndarray
    test_output: np.ndarray
    runs: List[RunTrace] = field(default_factory=list)
    
    # Diagnosis
    diagnosis_notes: List[str] = field(default_factory=list)


@dataclass
class HPMBankAnalysis:
    """Analysis results for a single HPM bank."""
    bank_name: str
    bank_type: str  # 'static' or 'dynamic'
    num_primitives: int
    embedding_dim: int
    
    # Health metrics
    has_nan: bool = False
    has_zero: bool = False
    num_nan_entries: int = 0
    num_zero_entries: int = 0
    
    # Diversity metrics
    mean_pairwise_cosine: float = 0.0  # Lower = more diverse
    std_pairwise_cosine: float = 0.0
    min_pairwise_cosine: float = 0.0
    max_pairwise_cosine: float = 0.0
    
    # Norm statistics
    mean_norm: float = 0.0
    std_norm: float = 0.0
    min_norm: float = 0.0
    max_norm: float = 0.0
    
    # Clustering (for detecting collapse)
    num_unique_clusters: int = 0  # Via k-means
    effective_rank: float = 0.0   # SVD-based dimensionality
    
    # Interpretability
    top_activations: List[Tuple[int, float]] = field(default_factory=list)  # (idx, activation)
    semantic_hints: List[str] = field(default_factory=list)  # Inferred meanings
    
    # Raw embeddings for visualization
    embeddings_2d: Optional[np.ndarray] = None  # t-SNE projection


@dataclass 
class HPMAnalysis:
    """Complete HPM analysis for a model."""
    # Static banks
    compositional_bank: Optional[HPMBankAnalysis] = None
    pattern_bank: Optional[HPMBankAnalysis] = None
    relational_bank: Optional[HPMBankAnalysis] = None
    concept_bank: Optional[HPMBankAnalysis] = None
    
    # Dynamic buffers
    instance_buffer: Optional[HPMBankAnalysis] = None
    procedural_buffer: Optional[HPMBankAnalysis] = None
    
    # Overall health
    total_static_primitives: int = 0
    total_dynamic_entries: int = 0
    overall_health_score: float = 0.0  # 0-1, higher = healthier
    health_notes: List[str] = field(default_factory=list)
    
    # Cross-bank analysis
    inter_bank_similarity: Dict[str, float] = field(default_factory=dict)  # bank_pair -> similarity
    
    @property
    def banks(self) -> Dict[str, HPMBankAnalysis]:
        """Return all non-None banks as a dict."""
        result = {}
        if self.compositional_bank:
            result['Compositional'] = self.compositional_bank
        if self.pattern_bank:
            result['Pattern'] = self.pattern_bank
        if self.relational_bank:
            result['Relational'] = self.relational_bank
        if self.concept_bank:
            result['Concept'] = self.concept_bank
        if self.instance_buffer:
            result['Instance Buffer'] = self.instance_buffer
        if self.procedural_buffer:
            result['Procedural Buffer'] = self.procedural_buffer
        return result
    
    @property
    def overall_diversity(self) -> float:
        """Average diversity across all banks (1 - mean_cosine), clamped to [0, 1]."""
        diversities = []
        for bank in self.banks.values():
            # Clamp each bank's diversity to [0, 1] since cosine can be negative
            div = max(0.0, min(1.0, 1.0 - bank.mean_pairwise_cosine))
            diversities.append(div)
        return sum(diversities) / len(diversities) if diversities else 0.0


# =============================================================================
# GRID RENDERING
# =============================================================================

def grid_to_svg(grid: np.ndarray, cell_size: int = 20, show_values: bool = False) -> str:
    """Convert a grid to SVG string."""
    h, w = grid.shape
    svg_w = w * cell_size
    svg_h = h * cell_size
    
    lines = [f'<svg width="{svg_w}" height="{svg_h}" xmlns="http://www.w3.org/2000/svg">']
    
    for i in range(h):
        for j in range(w):
            val = int(grid[i, j])
            color = ARC_COLORS[min(val, 10)]
            x, y = j * cell_size, i * cell_size
            lines.append(f'<rect x="{x}" y="{y}" width="{cell_size}" height="{cell_size}" '
                        f'fill="{color}" stroke="#333" stroke-width="0.5"/>')
            if show_values:
                text_color = '#fff' if val in [0, 9] else '#000'
                lines.append(f'<text x="{x + cell_size//2}" y="{y + cell_size//2 + 4}" '
                           f'text-anchor="middle" font-size="10" fill="{text_color}">{val}</text>')
    
    lines.append('</svg>')
    return '\n'.join(lines)


def grid_to_base64_png(grid: np.ndarray, cell_size: int = 20) -> str:
    """Convert grid to base64-encoded PNG for embedding in HTML."""
    try:
        from PIL import Image
        h, w = grid.shape
        img = Image.new('RGB', (w * cell_size, h * cell_size))
        
        for i in range(h):
            for j in range(w):
                val = int(grid[i, j])
                color_hex = ARC_COLORS[min(val, 10)]
                r = int(color_hex[1:3], 16)
                g = int(color_hex[3:5], 16)
                b = int(color_hex[5:7], 16)
                
                for di in range(cell_size):
                    for dj in range(cell_size):
                        img.putpixel((j * cell_size + dj, i * cell_size + di), (r, g, b))
        
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except ImportError:
        return None


def attention_heatmap_svg(attention: np.ndarray, cell_size: int = 20) -> str:
    """Render attention map as SVG heatmap."""
    h, w = attention.shape
    svg_w = w * cell_size
    svg_h = h * cell_size
    
    # Normalize attention to [0, 1]
    att_min, att_max = attention.min(), attention.max()
    if att_max > att_min:
        att_norm = (attention - att_min) / (att_max - att_min)
    else:
        att_norm = np.zeros_like(attention)
    
    lines = [f'<svg width="{svg_w}" height="{svg_h}" xmlns="http://www.w3.org/2000/svg">']
    
    for i in range(h):
        for j in range(w):
            val = att_norm[i, j]
            # Blue to red colormap
            r = int(255 * val)
            b = int(255 * (1 - val))
            g = 0
            x, y = j * cell_size, i * cell_size
            lines.append(f'<rect x="{x}" y="{y}" width="{cell_size}" height="{cell_size}" '
                        f'fill="rgb({r},{g},{b})" opacity="0.7"/>')
    
    lines.append('</svg>')
    return '\n'.join(lines)


def diff_mask_svg(diff_mask: np.ndarray, cell_size: int = 20) -> str:
    """Render difference mask as SVG (red where wrong)."""
    h, w = diff_mask.shape
    svg_w = w * cell_size
    svg_h = h * cell_size
    
    lines = [f'<svg width="{svg_w}" height="{svg_h}" xmlns="http://www.w3.org/2000/svg">']
    
    for i in range(h):
        for j in range(w):
            if diff_mask[i, j]:
                x, y = j * cell_size, i * cell_size
                lines.append(f'<rect x="{x}" y="{y}" width="{cell_size}" height="{cell_size}" '
                            f'fill="red" opacity="0.5"/>')
    
    lines.append('</svg>')
    return '\n'.join(lines)


# =============================================================================
# MODEL LOADING (Minimal, standalone)
# =============================================================================

def load_checkpoint(checkpoint_path: str, device: str = 'cuda') -> Tuple[Any, dict]:
    """Load model and config from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get('config', {})
    
    # Import model class only when needed
    from sci_arc.models.rlan import RLAN, RLANConfig
    
    # Build model config
    model_config = config.get('model', {})
    rlan_config = RLANConfig(
        hidden_dim=model_config.get('hidden_dim', 256),
        num_clues=model_config.get('num_clues', 5),
        num_predicates=model_config.get('num_predicates', 8),
        num_steps=model_config.get('num_steps', 5),
        use_dsc=model_config.get('use_dsc', True),
        use_sph=model_config.get('use_sph', False),
        use_lcr=model_config.get('use_lcr', True),
        use_hyperlora=model_config.get('use_hyperlora', True),
        use_context_encoder=model_config.get('use_context_encoder', True),
    )
    
    model = RLAN(rlan_config)
    
    # Load weights
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', {}))
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded: hidden_dim={rlan_config.hidden_dim}, num_clues={rlan_config.num_clues}")
    
    return model, config


def load_task(task_path: str) -> dict:
    """Load task from JSON file."""
    with open(task_path, 'r') as f:
        return json.load(f)


def find_task_file(task_id: str, data_dirs: List[str]) -> Optional[str]:
    """Find task JSON file by ID."""
    for data_dir in data_dirs:
        # Try various patterns
        patterns = [
            f"{task_id}.json",
            f"{task_id[:8]}.json",  # Short ID
        ]
        for pattern in patterns:
            path = Path(data_dir) / pattern
            if path.exists():
                return str(path)
            # Check subdirs
            for subdir in Path(data_dir).glob('**/'):
                path = subdir / pattern
                if path.exists():
                    return str(path)
    return None


# =============================================================================
# HPM BUFFER ANALYSIS
# =============================================================================

def analyze_embedding_bank(
    embeddings: torch.Tensor,
    bank_name: str,
    bank_type: str = 'static'
) -> HPMBankAnalysis:
    """
    Analyze a bank of embeddings for health, diversity, and interpretability.
    
    Args:
        embeddings: Tensor of shape (N, D) where N=num_primitives, D=embedding_dim
        bank_name: Name of the bank (e.g., 'compositional', 'pattern')
        bank_type: 'static' or 'dynamic'
    
    Returns:
        HPMBankAnalysis with comprehensive metrics
    """
    embeddings = embeddings.detach().float()
    N, D = embeddings.shape
    
    analysis = HPMBankAnalysis(
        bank_name=bank_name,
        bank_type=bank_type,
        num_primitives=N,
        embedding_dim=D,
    )
    
    # --- Health Checks ---
    nan_mask = torch.isnan(embeddings).any(dim=1)
    zero_mask = (embeddings.norm(dim=1) < 1e-6)
    
    analysis.has_nan = nan_mask.any().item()
    analysis.has_zero = zero_mask.any().item()
    analysis.num_nan_entries = nan_mask.sum().item()
    analysis.num_zero_entries = zero_mask.sum().item()
    
    # Filter valid embeddings for further analysis
    valid_mask = ~nan_mask & ~zero_mask
    valid_embeddings = embeddings[valid_mask]
    
    if len(valid_embeddings) < 2:
        analysis.semantic_hints.append("⚠️ Too few valid embeddings for analysis")
        return analysis
    
    # --- Norm Statistics ---
    norms = valid_embeddings.norm(dim=1)
    analysis.mean_norm = norms.mean().item()
    analysis.std_norm = norms.std().item()
    analysis.min_norm = norms.min().item()
    analysis.max_norm = norms.max().item()
    
    # --- Diversity via Pairwise Cosine Similarity ---
    # Move to CPU to avoid OOM on large banks, limit to first 1000 embeddings
    MAX_FOR_PAIRWISE = 1000
    if len(valid_embeddings) > MAX_FOR_PAIRWISE:
        # Random sample for large banks
        indices = torch.randperm(len(valid_embeddings))[:MAX_FOR_PAIRWISE]
        sample_embeddings = valid_embeddings[indices].cpu()
    else:
        sample_embeddings = valid_embeddings.cpu()
    
    # Normalize for cosine
    normed = F.normalize(sample_embeddings, dim=1)
    cosine_sim = torch.mm(normed, normed.t())
    
    # Get upper triangle (exclude diagonal)
    mask = torch.triu(torch.ones_like(cosine_sim), diagonal=1).bool()
    pairwise_cosines = cosine_sim[mask]
    
    if len(pairwise_cosines) > 0:
        analysis.mean_pairwise_cosine = pairwise_cosines.mean().item()
        analysis.std_pairwise_cosine = pairwise_cosines.std().item()
        analysis.min_pairwise_cosine = pairwise_cosines.min().item()
        analysis.max_pairwise_cosine = pairwise_cosines.max().item()
    
    # --- Effective Rank via SVD ---
    # Measures dimensionality of the learned embedding space
    try:
        # Center embeddings (use CPU sample if available)
        emb_for_svd = sample_embeddings if 'sample_embeddings' in dir() else valid_embeddings.cpu()
        centered = emb_for_svd - emb_for_svd.mean(dim=0, keepdim=True)
        _, s, _ = torch.svd(centered)
        
        # Guard against division by zero
        s_sum = s.sum()
        if s_sum < 1e-10:
            analysis.effective_rank = 0.0
        else:
            # Normalize singular values
            s_norm = s / s_sum
            # Shannon entropy of singular values
            entropy = -(s_norm * torch.log(s_norm + 1e-10)).sum()
            # Effective rank = exp(entropy)
            eff_rank = torch.exp(entropy).item()
            # Guard against NaN
            analysis.effective_rank = eff_rank if not (eff_rank != eff_rank) else 0.0
    except Exception:
        analysis.effective_rank = 0.0
    
    # --- Cluster Detection (Simple greedy clustering) ---
    # Check if embeddings collapse to few clusters
    try:
        # Use cosine similarity threshold to count "unique" primitives
        SIMILARITY_THRESHOLD = 0.95  # Very similar = same cluster
        n_sample = len(sample_embeddings) if 'sample_embeddings' in dir() else len(valid_embeddings)
        used = [False] * n_sample
        unique_count = 0
        
        for i in range(n_sample):
            if used[i]:
                continue
            # This is a new cluster seed
            unique_count += 1
            used[i] = True
            # Mark all highly similar embeddings as belonging to this cluster
            for j in range(i + 1, n_sample):
                if not used[j] and cosine_sim[i, j] > SIMILARITY_THRESHOLD:
                    used[j] = True
        
        analysis.num_unique_clusters = unique_count
    except Exception:
        analysis.num_unique_clusters = N
    
    # --- Semantic Interpretation ---
    # Generate human-readable insights
    if analysis.mean_pairwise_cosine > 0.9:
        analysis.semantic_hints.append("🔴 COLLAPSED: All primitives nearly identical (cosine > 0.9)")
    elif analysis.mean_pairwise_cosine > 0.7:
        analysis.semantic_hints.append("🟡 LOW DIVERSITY: Primitives too similar (cosine > 0.7)")
    elif analysis.mean_pairwise_cosine < 0.3:
        analysis.semantic_hints.append("🟢 DIVERSE: Primitives well-separated (cosine < 0.3)")
    else:
        analysis.semantic_hints.append("🟡 MODERATE diversity")
    
    if analysis.effective_rank < D * 0.1:
        analysis.semantic_hints.append(f"⚠️ Low effective rank ({analysis.effective_rank:.1f}/{D}) - may be under-utilizing dimensions")
    elif analysis.effective_rank > D * 0.5:
        analysis.semantic_hints.append(f"✓ Good effective rank ({analysis.effective_rank:.1f}/{D})")
    
    if analysis.std_norm > analysis.mean_norm * 0.5:
        analysis.semantic_hints.append("⚠️ High norm variance - some primitives may dominate")
    
    # --- 2D Projection for Visualization ---
    try:
        from sklearn.manifold import TSNE
        # Use sample if available, limit to 500 for t-SNE speed
        emb_for_tsne = sample_embeddings if 'sample_embeddings' in dir() else valid_embeddings.cpu()
        if len(emb_for_tsne) > 500:
            emb_for_tsne = emb_for_tsne[:500]
        if len(emb_for_tsne) >= 5:  # Need enough points for t-SNE
            tsne = TSNE(n_components=2, perplexity=min(5, len(emb_for_tsne)-1), random_state=42)
            analysis.embeddings_2d = tsne.fit_transform(emb_for_tsne.numpy())
    except (ImportError, ValueError, Exception):
        # Fall back to PCA if sklearn not available or t-SNE fails
        try:
            emb_for_pca = sample_embeddings if 'sample_embeddings' in dir() else valid_embeddings.cpu()
            _, _, V = torch.svd(emb_for_pca - emb_for_pca.mean(dim=0))
            analysis.embeddings_2d = (emb_for_pca @ V[:, :2]).numpy()
        except:
            pass
    
    return analysis


def analyze_hpm_buffers(model: Any) -> HPMAnalysis:
    """
    Comprehensive analysis of all HPM buffers in a model.
    
    Returns:
        HPMAnalysis with all bank analyses and overall health metrics
    """
    analysis = HPMAnalysis()
    health_issues = []
    
    # --- Analyze Static Banks ---
    if hasattr(model, 'hpm') and model.hpm is not None:
        hpm = model.hpm
        
        # Compositional bank
        if hasattr(hpm, 'compositional_bank') and hpm.compositional_bank is not None:
            bank = hpm.compositional_bank
            # Collect all primitives from hierarchical levels
            all_primitives = torch.cat([level for level in bank.primitive_levels], dim=0)
            analysis.compositional_bank = analyze_embedding_bank(
                all_primitives, 'Compositional', 'static'
            )
            analysis.total_static_primitives += len(all_primitives)
        
        # Pattern bank
        if hasattr(hpm, 'pattern_bank') and hpm.pattern_bank is not None:
            bank = hpm.pattern_bank
            all_primitives = torch.cat([level for level in bank.primitive_levels], dim=0)
            analysis.pattern_bank = analyze_embedding_bank(
                all_primitives, 'Pattern', 'static'
            )
            analysis.total_static_primitives += len(all_primitives)
        
        # Relational bank
        if hasattr(hpm, 'relational_bank') and hpm.relational_bank is not None:
            bank = hpm.relational_bank
            all_primitives = torch.cat([level for level in bank.primitive_levels], dim=0)
            analysis.relational_bank = analyze_embedding_bank(
                all_primitives, 'Relational', 'static'
            )
            analysis.total_static_primitives += len(all_primitives)
        
        # Concept bank
        if hasattr(hpm, 'concept_bank') and hpm.concept_bank is not None:
            bank = hpm.concept_bank
            all_primitives = torch.cat([level for level in bank.primitive_levels], dim=0)
            analysis.concept_bank = analyze_embedding_bank(
                all_primitives, 'Concept', 'static'
            )
            analysis.total_static_primitives += len(all_primitives)
    
    # --- Analyze Dynamic Buffers ---
    # Instance buffer (ContextEncoder outputs)
    if hasattr(model, 'hpm_instance_buffer') and model.hpm_instance_buffer is not None:
        buffer = model.hpm_instance_buffer
        if hasattr(buffer, 'embeddings') and len(buffer) > 0:
            embeddings = buffer.embeddings[:len(buffer)]
            analysis.instance_buffer = analyze_embedding_bank(
                embeddings, 'Instance (Context)', 'dynamic'
            )
            analysis.total_dynamic_entries += len(buffer)
            
            # Add task diversity info
            if hasattr(buffer, 'task_ids'):
                unique_tasks = len(set(buffer.task_ids[:len(buffer)]))
                analysis.instance_buffer.semantic_hints.append(
                    f"📊 {unique_tasks} unique tasks stored"
                )
    
    # Procedural buffer (HyperLoRA codes)
    if hasattr(model, 'hpm_procedural_buffer') and model.hpm_procedural_buffer is not None:
        buffer = model.hpm_procedural_buffer
        if hasattr(buffer, 'embeddings') and len(buffer) > 0:
            embeddings = buffer.embeddings[:len(buffer)]
            analysis.procedural_buffer = analyze_embedding_bank(
                embeddings, 'Procedural (LoRA)', 'dynamic'
            )
            analysis.total_dynamic_entries += len(buffer)
            
            if hasattr(buffer, 'task_ids'):
                unique_tasks = len(set(buffer.task_ids[:len(buffer)]))
                analysis.procedural_buffer.semantic_hints.append(
                    f"📊 {unique_tasks} unique task procedures stored"
                )
    
    # Note: Cross-bank similarity analysis removed (would require storing raw embeddings)
    
    # --- Overall Health Score ---
    # 0-1 score based on:
    # - No NaN/zero entries
    # - Diverse primitives (low cosine similarity)
    # - Good effective rank
    # - Reasonable norm variance
    
    scores = []
    for bank in [analysis.compositional_bank, analysis.pattern_bank, 
                 analysis.relational_bank, analysis.concept_bank,
                 analysis.instance_buffer, analysis.procedural_buffer]:
        if bank is None:
            continue
        
        bank_score = 1.0
        
        # Penalize NaN/zero
        if bank.has_nan:
            bank_score -= 0.3
            health_issues.append(f"❌ {bank.bank_name}: Contains NaN values")
        if bank.has_zero:
            bank_score -= 0.2
            health_issues.append(f"⚠️ {bank.bank_name}: Contains zero vectors")
        
        # Penalize collapse
        if bank.mean_pairwise_cosine > 0.9:
            bank_score -= 0.4
            health_issues.append(f"🔴 {bank.bank_name}: Collapsed (cosine={bank.mean_pairwise_cosine:.2f})")
        elif bank.mean_pairwise_cosine > 0.7:
            bank_score -= 0.2
            health_issues.append(f"🟡 {bank.bank_name}: Low diversity")
        
        # Reward good effective rank
        if bank.effective_rank > bank.embedding_dim * 0.3:
            bank_score += 0.1
        
        scores.append(max(0, min(1, bank_score)))
    
    if scores:
        analysis.overall_health_score = sum(scores) / len(scores)
    
    analysis.health_notes = health_issues if health_issues else ["✓ All HPM banks appear healthy"]
    
    return analysis


def generate_hpm_html(analysis: HPMAnalysis, output_path: str):
    """Generate standalone HTML visualization for HPM analysis."""
    
    def bank_section(bank: Optional[HPMBankAnalysis], title: str) -> str:
        if bank is None:
            return f'''
            <div class="bank-card disabled">
                <h3>{title}</h3>
                <p class="status">Not configured / Empty</p>
            </div>
            '''
        
        # Determine health color
        if bank.has_nan or bank.mean_pairwise_cosine > 0.9:
            health_color = '#ff4444'
            health_icon = '🔴'
        elif bank.has_zero or bank.mean_pairwise_cosine > 0.7:
            health_color = '#ffaa00'
            health_icon = '🟡'
        else:
            health_color = '#44ff44'
            health_icon = '🟢'
        
        # Generate 2D scatter plot if available
        scatter_html = ''
        if bank.embeddings_2d is not None:
            points = bank.embeddings_2d
            # Normalize to 0-400 range for SVG
            x_min, x_max = points[:, 0].min(), points[:, 0].max()
            y_min, y_max = points[:, 1].min(), points[:, 1].max()
            
            svg_points = []
            for i, (x, y) in enumerate(points):
                sx = 20 + (x - x_min) / (x_max - x_min + 1e-6) * 360
                sy = 20 + (y - y_min) / (y_max - y_min + 1e-6) * 260
                svg_points.append(f'<circle cx="{sx}" cy="{sy}" r="5" fill="{health_color}" opacity="0.7" title="Primitive {i}"/>')
            
            scatter_html = f'''
            <div class="scatter-plot">
                <svg width="400" height="300" style="background: #1a1a2e; border-radius: 8px;">
                    {''.join(svg_points)}
                    <text x="200" y="290" text-anchor="middle" fill="#888" font-size="12">t-SNE Projection</text>
                </svg>
            </div>
            '''
        
        hints_html = '<br>'.join(bank.semantic_hints) if bank.semantic_hints else 'No issues detected'
        
        return f'''
        <div class="bank-card" style="border-left: 4px solid {health_color};">
            <h3>{health_icon} {title}</h3>
            <div class="bank-stats">
                <div class="stat">
                    <span class="label">Primitives</span>
                    <span class="value">{bank.num_primitives}</span>
                </div>
                <div class="stat">
                    <span class="label">Dimensions</span>
                    <span class="value">{bank.embedding_dim}</span>
                </div>
                <div class="stat">
                    <span class="label">Mean Cosine</span>
                    <span class="value" style="color: {health_color}">{bank.mean_pairwise_cosine:.3f}</span>
                </div>
                <div class="stat">
                    <span class="label">Effective Rank</span>
                    <span class="value">{bank.effective_rank:.1f}</span>
                </div>
                <div class="stat">
                    <span class="label">Unique Clusters</span>
                    <span class="value">{bank.num_unique_clusters}</span>
                </div>
                <div class="stat">
                    <span class="label">Norm (μ±σ)</span>
                    <span class="value">{bank.mean_norm:.2f}±{bank.std_norm:.2f}</span>
                </div>
            </div>
            <div class="hints">
                <strong>Analysis:</strong><br>
                {hints_html}
            </div>
            {scatter_html}
        </div>
        '''
    
    # Generate overall health bar
    health_pct = int(analysis.overall_health_score * 100)
    health_color = '#44ff44' if health_pct > 70 else '#ffaa00' if health_pct > 40 else '#ff4444'
    
    html_content = f'''<!DOCTYPE html>
<html>
<head>
    <title>HPM Buffer Analysis</title>
    <style>
        body {{
            font-family: 'Segoe UI', Arial, sans-serif;
            background: #0f0f1a;
            color: #eee;
            margin: 0;
            padding: 20px;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        h1 {{
            color: #7c3aed;
            border-bottom: 2px solid #7c3aed;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #a78bfa;
            margin-top: 30px;
        }}
        .health-bar {{
            background: #1a1a2e;
            border-radius: 10px;
            height: 30px;
            margin: 20px 0;
            overflow: hidden;
        }}
        .health-fill {{
            height: 100%;
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            transition: width 0.5s;
        }}
        .summary-stats {{
            display: flex;
            gap: 30px;
            margin: 20px 0;
        }}
        .summary-stat {{
            background: #1a1a2e;
            padding: 15px 25px;
            border-radius: 8px;
        }}
        .summary-stat .label {{
            color: #888;
            font-size: 12px;
        }}
        .summary-stat .value {{
            font-size: 24px;
            font-weight: bold;
            color: #7c3aed;
        }}
        .banks-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        .bank-card {{
            background: #1a1a2e;
            border-radius: 8px;
            padding: 20px;
        }}
        .bank-card.disabled {{
            opacity: 0.5;
        }}
        .bank-card h3 {{
            margin-top: 0;
            color: #fff;
        }}
        .bank-stats {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
            margin: 15px 0;
        }}
        .stat {{
            background: #0f0f1a;
            padding: 10px;
            border-radius: 4px;
            text-align: center;
        }}
        .stat .label {{
            font-size: 11px;
            color: #888;
            display: block;
        }}
        .stat .value {{
            font-size: 16px;
            font-weight: bold;
        }}
        .hints {{
            background: #0f0f1a;
            padding: 15px;
            border-radius: 4px;
            font-size: 13px;
            line-height: 1.6;
        }}
        .scatter-plot {{
            margin-top: 15px;
            text-align: center;
        }}
        .health-notes {{
            background: #1a1a2e;
            padding: 20px;
            border-radius: 8px;
            margin-top: 20px;
        }}
        .health-notes li {{
            margin: 8px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🧠 HPM Buffer Analysis</h1>
        
        <div class="health-bar">
            <div class="health-fill" style="width: {health_pct}%; background: {health_color};">
                Overall Health: {health_pct}%
            </div>
        </div>
        
        <div class="summary-stats">
            <div class="summary-stat">
                <span class="label">Static Primitives</span>
                <span class="value">{analysis.total_static_primitives}</span>
            </div>
            <div class="summary-stat">
                <span class="label">Dynamic Entries</span>
                <span class="value">{analysis.total_dynamic_entries}</span>
            </div>
            <div class="summary-stat">
                <span class="label">Health Score</span>
                <span class="value" style="color: {health_color}">{analysis.overall_health_score:.2f}</span>
            </div>
        </div>
        
        <h2>📦 Static Banks (Learned Primitives)</h2>
        <div class="banks-grid">
            {bank_section(analysis.compositional_bank, "Compositional Bank")}
            {bank_section(analysis.pattern_bank, "Pattern Bank")}
            {bank_section(analysis.relational_bank, "Relational Bank")}
            {bank_section(analysis.concept_bank, "Concept Bank")}
        </div>
        
        <h2>📝 Dynamic Buffers (Runtime Memory)</h2>
        <div class="banks-grid">
            {bank_section(analysis.instance_buffer, "Instance Buffer (Context Encodings)")}
            {bank_section(analysis.procedural_buffer, "Procedural Buffer (LoRA Codes)")}
        </div>
        
        <div class="health-notes">
            <h3>🩺 Health Notes</h3>
            <ul>
                {''.join(f'<li>{note}</li>' for note in analysis.health_notes)}
            </ul>
        </div>
        
        <div style="margin-top: 40px; color: #666; font-size: 12px;">
            Generated by RLAN Visual Debugger | HPM Analysis Module
        </div>
    </div>
</body>
</html>'''
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"📊 HPM analysis saved to: {output_path}")


# =============================================================================
# INFERENCE AND TRACING
# =============================================================================

def run_inference_with_trace(
    model: Any,
    task: dict,
    device: str,
    config_overrides: dict = None,
) -> RunTrace:
    """
    Run model inference and collect step-by-step trace.
    
    Args:
        model: RLAN model
        task: Task dict with train/test pairs
        device: Device string
        config_overrides: Dict of runtime attribute overrides for ablation
    
    Returns:
        RunTrace with all step data
    """
    config_name = config_overrides.get('name', 'default') if config_overrides else 'default'
    trace = RunTrace(config_name=config_name)
    
    # Apply runtime overrides (for ablation studies)
    original_attrs = {}
    if config_overrides:
        for attr, value in config_overrides.items():
            if attr == 'name':
                continue
            if hasattr(model, attr):
                original_attrs[attr] = getattr(model, attr)
                setattr(model, attr, value)
    
    try:
        # Parse task
        train_inputs = [np.array(p['input'], dtype=np.int64) for p in task['train']]
        train_outputs = [np.array(p['output'], dtype=np.int64) for p in task['train']]
        test_input = np.array(task['test'][0]['input'], dtype=np.int64)
        
        # Handle evaluation challenges that may not have test output
        # Use first train output shape as fallback for cropping
        if 'output' in task['test'][0]:
            test_output = np.array(task['test'][0]['output'], dtype=np.int64)
            has_test_output = True
        else:
            # No ground truth - use train output shape for cropping, zeros for comparison
            test_output = np.zeros_like(train_outputs[0])
            has_test_output = False
            print(f"    ⚠️ No test output in task - metrics will show prediction only")
        
        # Record module status
        trace.dsc_enabled = getattr(model, 'use_dsc', False)
        trace.hyperlora_enabled = getattr(model, 'use_hyperlora', False) and getattr(model, 'hyperlora_active', True)
        trace.solver_context_enabled = getattr(model, 'solver_context_active', False)
        trace.hpm_enabled = getattr(model, 'use_hpm', False)
        
        # Prepare inputs
        max_size = 30
        
        def pad_grid(g, is_target=False):
            h, w = g.shape
            padded = np.full((max_size, max_size), -100 if is_target else 10, dtype=np.int64)
            padded[:h, :w] = g
            return padded
        
        train_in_t = torch.stack([torch.from_numpy(pad_grid(g)) for g in train_inputs]).unsqueeze(0).to(device)
        train_out_t = torch.stack([torch.from_numpy(pad_grid(g, True)) for g in train_outputs]).unsqueeze(0).to(device)
        test_in_t = torch.from_numpy(pad_grid(test_input)).unsqueeze(0).to(device)
        
        # FIX: pair_mask shape must be (B, N) where N = actual number of pairs
        # Previously hardcoded to (1, 10) which caused shape mismatch with train_inputs
        num_pairs = len(train_inputs)
        pair_mask = torch.ones(1, num_pairs, dtype=torch.bool, device=device)
        
        # Check for num_steps_override in config (for --test_solver)
        num_steps_override = None
        if config_overrides and '_num_steps_override' in config_overrides:
            num_steps_override = config_overrides['_num_steps_override']
        
        # Forward pass with intermediates
        with torch.no_grad():
            outputs = model(
                test_in_t,
                train_inputs=train_in_t,
                train_outputs=train_out_t,
                pair_mask=pair_mask,
                temperature=0.5,
                return_intermediates=True,
                num_steps_override=num_steps_override,  # FIX: Pass to model for --test_solver
            )
        
        # Extract step-by-step predictions
        all_logits = outputs.get('all_logits', [outputs['logits']])
        attention_maps = outputs.get('attention_maps')  # (B, K, H, W)
        centroids = outputs.get('centroids')  # (B, K, 2)
        stop_logits = outputs.get('stop_logits')  # (B, K)
        lora_deltas = outputs.get('lora_deltas')
        
        # Get test output shape for cropping
        out_h, out_w = test_output.shape
        
        for t, step_logits in enumerate(all_logits):
            pred = step_logits.argmax(dim=1)[0].cpu().numpy()
            pred_cropped = pred[:out_h, :out_w]
            
            # Compute metrics
            diff_mask = (pred_cropped != test_output)
            total_pixels = out_h * out_w
            correct_pixels = (~diff_mask).sum()
            pixel_acc = correct_pixels / total_pixels
            
            # FG/BG accuracy
            fg_mask = test_output != 0
            bg_mask = test_output == 0
            fg_acc = (pred_cropped[fg_mask] == test_output[fg_mask]).mean() if fg_mask.any() else 1.0
            bg_acc = (pred_cropped[bg_mask] == test_output[bg_mask]).mean() if bg_mask.any() else 1.0
            
            # Entropy
            probs = F.softmax(step_logits, dim=1)[0]  # (C, H, W)
            entropy = -(probs * probs.clamp(min=1e-8).log()).sum(dim=0).mean().item()
            
            step_trace = StepTrace(
                step_idx=t,
                predicted_grid=pred_cropped,
                logits_entropy=entropy,
                pixel_accuracy=pixel_acc,
                fg_accuracy=float(fg_acc),
                bg_accuracy=float(bg_acc),
                diff_mask=diff_mask,
            )
            
            # Add DSC info for last step
            if t == len(all_logits) - 1:
                if attention_maps is not None:
                    step_trace.attention_maps = attention_maps[0].cpu().numpy()[:, :out_h, :out_w]
                if centroids is not None:
                    step_trace.centroids = centroids[0].cpu().numpy()
                    # Compute centroid spread
                    cents = step_trace.centroids
                    if len(cents) > 1:
                        from scipy.spatial.distance import pdist
                        try:
                            step_trace.centroid_spread = pdist(cents).mean()
                        except:
                            step_trace.centroid_spread = 0.0
                if stop_logits is not None:
                    stop_probs = torch.sigmoid(stop_logits[0]).cpu().numpy()
                    step_trace.stop_probs = stop_probs
                    step_trace.estimated_clues = (1 - stop_probs).sum()
                
                # HyperLoRA info
                if lora_deltas is not None:
                    step_trace.lora_delta_norms = {}
                    for key in ['gru_reset', 'gru_update', 'gru_candidate', 'output_head']:
                        if key in lora_deltas and lora_deltas[key] is not None:
                            norm = lora_deltas[key][0].norm().item()
                            step_trace.lora_delta_norms[key] = norm
            
            trace.steps.append(step_trace)
        
        # Final results
        trace.final_prediction = trace.steps[-1].predicted_grid
        trace.final_accuracy = trace.steps[-1].pixel_accuracy
        trace.is_exact_match = np.array_equal(trace.final_prediction, test_output)
        
    finally:
        # Restore original attributes
        for attr, value in original_attrs.items():
            setattr(model, attr, value)
    
    return trace


# =============================================================================
# DIAGNOSIS
# =============================================================================

def diagnose_task(task_viz: TaskVisualization) -> List[str]:
    """Generate diagnosis notes based on traces."""
    notes = []
    
    for run in task_viz.runs:
        if len(run.steps) < 2:
            continue
        
        # Check for accuracy degradation
        accs = [s.pixel_accuracy for s in run.steps]
        if len(accs) >= 2:
            if accs[-1] < accs[0] - 0.05:
                notes.append(f"[{run.config_name}] ⚠️ Accuracy DEGRADED: {accs[0]:.1%} → {accs[-1]:.1%} (model drifts)")
            elif accs[-1] > accs[0] + 0.1:
                notes.append(f"[{run.config_name}] ✓ Accuracy IMPROVED: {accs[0]:.1%} → {accs[-1]:.1%}")
        
        # Check for clue collapse
        last_step = run.steps[-1]
        if last_step.centroid_spread is not None and last_step.centroid_spread < 1.0:
            notes.append(f"[{run.config_name}] ⚠️ Centroid COLLAPSE: spread={last_step.centroid_spread:.2f} (clues overlap)")
        
        # Check for stop saturation
        if last_step.stop_probs is not None:
            mean_stop = last_step.stop_probs.mean()
            if mean_stop > 0.95:
                notes.append(f"[{run.config_name}] ⚠️ Stop probs SATURATED high: mean={mean_stop:.3f} (always stopping)")
            elif mean_stop < 0.05:
                notes.append(f"[{run.config_name}] ⚠️ Stop probs SATURATED low: mean={mean_stop:.3f} (never stopping)")
        
        # Check HyperLoRA contribution
        if last_step.lora_delta_norms:
            total_norm = sum(last_step.lora_delta_norms.values())
            if total_norm < 0.001:
                notes.append(f"[{run.config_name}] ⚠️ HyperLoRA near-zero: Σnorm={total_norm:.4f} (not adapting)")
            elif total_norm > 5.0:
                notes.append(f"[{run.config_name}] ⚠️ HyperLoRA very large: Σnorm={total_norm:.2f} (may saturate)")
        
        # Check FG/BG balance
        if last_step.fg_accuracy < 0.3 and last_step.bg_accuracy > 0.9:
            notes.append(f"[{run.config_name}] ⚠️ FG COLLAPSE: fg_acc={last_step.fg_accuracy:.1%}, bg_acc={last_step.bg_accuracy:.1%}")
    
    return notes


# =============================================================================
# HTML GENERATION
# =============================================================================

def generate_html(task_viz: TaskVisualization, output_path: str):
    """Generate interactive HTML visualization."""
    
    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RLAN Debug: {task_viz.task_id}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #1a1a2e; color: #eee; padding: 20px; }}
        h1, h2, h3 {{ margin-bottom: 15px; }}
        .container {{ max-width: 1800px; margin: 0 auto; }}
        .header {{ background: #16213e; padding: 20px; border-radius: 10px; margin-bottom: 20px; }}
        .header h1 {{ color: #00d9ff; }}
        
        .panels {{ display: flex; gap: 20px; flex-wrap: wrap; }}
        .panel {{ background: #16213e; padding: 20px; border-radius: 10px; flex: 1; min-width: 300px; }}
        .panel h2 {{ color: #ffd700; border-bottom: 1px solid #333; padding-bottom: 10px; }}
        
        .pair {{ display: flex; gap: 10px; margin: 10px 0; align-items: flex-start; }}
        .pair-label {{ min-width: 60px; color: #888; }}
        
        .run-selector {{ margin: 15px 0; }}
        .run-selector select {{ padding: 8px 12px; font-size: 14px; border-radius: 5px; background: #0f3460; color: #fff; border: 1px solid #00d9ff; }}
        
        .step-slider {{ margin: 15px 0; }}
        .step-slider input {{ width: 100%; }}
        .step-info {{ display: flex; gap: 20px; flex-wrap: wrap; margin-top: 10px; }}
        .metric {{ background: #0f3460; padding: 8px 15px; border-radius: 5px; }}
        .metric-label {{ color: #888; font-size: 12px; }}
        .metric-value {{ font-size: 18px; font-weight: bold; }}
        .metric-value.good {{ color: #2ecc40; }}
        .metric-value.bad {{ color: #ff4136; }}
        .metric-value.neutral {{ color: #ffdc00; }}
        
        .grid-container {{ display: flex; gap: 20px; align-items: flex-start; flex-wrap: wrap; }}
        .grid-box {{ text-align: center; }}
        .grid-box label {{ display: block; margin-bottom: 5px; color: #888; font-size: 12px; }}
        
        .overlay-controls {{ margin: 15px 0; }}
        .overlay-controls label {{ margin-right: 15px; cursor: pointer; }}
        .overlay-controls input {{ margin-right: 5px; }}
        
        .attention-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(120px, 1fr)); gap: 10px; margin-top: 15px; }}
        .attention-item {{ text-align: center; }}
        .attention-item label {{ display: block; font-size: 11px; color: #888; margin-bottom: 3px; }}
        
        .diagnosis {{ background: #2d132c; padding: 15px; border-radius: 10px; margin-top: 20px; }}
        .diagnosis h3 {{ color: #ff6b6b; }}
        .diagnosis ul {{ list-style: none; margin-top: 10px; }}
        .diagnosis li {{ padding: 5px 0; border-bottom: 1px solid #333; }}
        .diagnosis li:last-child {{ border-bottom: none; }}
        
        .clue-stats {{ display: flex; gap: 10px; flex-wrap: wrap; margin-top: 10px; }}
        .clue-stat {{ background: #0f3460; padding: 5px 10px; border-radius: 5px; font-size: 12px; }}
        
        .charts {{ margin-top: 20px; }}
        .chart {{ background: #0f3460; padding: 15px; border-radius: 10px; margin-bottom: 15px; }}
        .bar-chart {{ display: flex; align-items: flex-end; gap: 5px; height: 100px; }}
        .bar {{ background: #00d9ff; min-width: 30px; text-align: center; font-size: 10px; color: #000; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔬 RLAN Debug Visualization</h1>
            <p>Task ID: <strong>{task_viz.task_id}</strong></p>
        </div>
        
        <div class="panels">
            <!-- Training Examples -->
            <div class="panel" style="flex: 0.8;">
                <h2>📚 Training Examples</h2>
                {generate_training_pairs_html(task_viz)}
            </div>
            
            <!-- Test I/O -->
            <div class="panel" style="flex: 0.6;">
                <h2>🎯 Test Case</h2>
                <div class="grid-container">
                    <div class="grid-box">
                        <label>Input</label>
                        {grid_to_svg(task_viz.test_input, cell_size=15)}
                    </div>
                    <div class="grid-box">
                        <label>Expected Output</label>
                        {grid_to_svg(task_viz.test_output, cell_size=15)}
                    </div>
                </div>
            </div>
            
            <!-- Main Prediction Panel -->
            <div class="panel" style="flex: 1.5;">
                <h2>🔄 Step-by-Step Refinement</h2>
                
                <div class="run-selector">
                    <label>Run Configuration: </label>
                    <select id="runSelect" onchange="updateVisualization()">
                        {generate_run_options(task_viz)}
                    </select>
                </div>
                
                <div class="step-slider">
                    <label>Step: <span id="stepLabel">0</span></label>
                    <input type="range" id="stepSlider" min="0" max="{len(task_viz.runs[0].steps)-1 if task_viz.runs else 0}" value="0" onchange="updateStep(this.value)">
                </div>
                
                <div class="step-info" id="stepInfo">
                    <!-- Filled by JS -->
                </div>
                
                <div class="grid-container" style="margin-top: 15px;">
                    <div class="grid-box">
                        <label>Prediction at Step</label>
                        <div id="predictionGrid"></div>
                    </div>
                    <div class="grid-box">
                        <label>Difference (red=wrong)</label>
                        <div id="diffGrid"></div>
                    </div>
                </div>
                
                <div class="overlay-controls">
                    <label><input type="checkbox" id="showDiff" checked onchange="updateVisualization()"> Show Differences</label>
                    <label><input type="checkbox" id="showAttention" onchange="updateVisualization()"> Show Attention</label>
                </div>
            </div>
        </div>
        
        <!-- DSC Attention Panel -->
        <div class="panel" style="margin-top: 20px;">
            <h2>🎯 DSC Attention Maps (Final Step)</h2>
            <div id="attentionMaps" class="attention-grid">
                <!-- Filled by JS -->
            </div>
            <div class="clue-stats" id="clueStats">
                <!-- Filled by JS -->
            </div>
        </div>
        
        <!-- Diagnosis Panel -->
        <div class="diagnosis">
            <h3>🔍 Automatic Diagnosis</h3>
            <ul>
                {generate_diagnosis_html(task_viz)}
            </ul>
        </div>
        
        <!-- Accuracy Chart -->
        <div class="charts">
            <div class="chart">
                <h3>📈 Accuracy Over Steps</h3>
                <div id="accuracyChart" class="bar-chart"></div>
            </div>
        </div>
    </div>
    
    <script>
        // Embed trace data as JSON
        const traceData = {json.dumps(serialize_task_viz(task_viz))};
        
        let currentRun = 0;
        let currentStep = 0;
        
        function updateVisualization() {{
            currentRun = document.getElementById('runSelect').selectedIndex;
            updateStep(currentStep);
            updateAttentionMaps();
            updateAccuracyChart();
        }}
        
        function updateStep(step) {{
            currentStep = parseInt(step);
            document.getElementById('stepLabel').textContent = currentStep;
            document.getElementById('stepSlider').value = currentStep;
            
            const run = traceData.runs[currentRun];
            if (!run || !run.steps[currentStep]) return;
            
            const stepData = run.steps[currentStep];
            
            // Update metrics
            const accClass = stepData.pixel_accuracy > 0.9 ? 'good' : (stepData.pixel_accuracy > 0.5 ? 'neutral' : 'bad');
            const fgClass = stepData.fg_accuracy > 0.7 ? 'good' : 'bad';
            
            document.getElementById('stepInfo').innerHTML = `
                <div class="metric">
                    <div class="metric-label">Pixel Accuracy</div>
                    <div class="metric-value ${{accClass}}">${{(stepData.pixel_accuracy * 100).toFixed(1)}}%</div>
                </div>
                <div class="metric">
                    <div class="metric-label">FG Accuracy</div>
                    <div class="metric-value ${{fgClass}}">${{(stepData.fg_accuracy * 100).toFixed(1)}}%</div>
                </div>
                <div class="metric">
                    <div class="metric-label">BG Accuracy</div>
                    <div class="metric-value">${{(stepData.bg_accuracy * 100).toFixed(1)}}%</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Entropy</div>
                    <div class="metric-value">${{stepData.logits_entropy.toFixed(3)}}</div>
                </div>
            `;
            
            // Update grids
            document.getElementById('predictionGrid').innerHTML = stepData.predicted_svg;
            
            if (document.getElementById('showDiff').checked && stepData.diff_svg) {{
                document.getElementById('diffGrid').innerHTML = stepData.diff_svg;
            }} else {{
                document.getElementById('diffGrid').innerHTML = '<p style="color:#888">No diff overlay</p>';
            }}
        }}
        
        function updateAttentionMaps() {{
            const run = traceData.runs[currentRun];
            if (!run) return;
            
            const lastStep = run.steps[run.steps.length - 1];
            if (!lastStep || !lastStep.attention_svgs) {{
                document.getElementById('attentionMaps').innerHTML = '<p style="color:#888">No attention data</p>';
                return;
            }}
            
            let html = '';
            for (let k = 0; k < lastStep.attention_svgs.length; k++) {{
                const stopProb = lastStep.stop_probs ? lastStep.stop_probs[k] : 0;
                const centroid = lastStep.centroids ? lastStep.centroids[k] : null;
                html += `
                    <div class="attention-item">
                        <label>Clue ${{k+1}} (stop=${{stopProb.toFixed(2)}})</label>
                        ${{lastStep.attention_svgs[k]}}
                        ${{centroid ? `<br><small>centroid: (${{centroid[0].toFixed(1)}}, ${{centroid[1].toFixed(1)}})</small>` : ''}}
                    </div>
                `;
            }}
            document.getElementById('attentionMaps').innerHTML = html;
            
            // Clue stats
            let statsHtml = '';
            if (lastStep.estimated_clues !== undefined) {{
                statsHtml += `<div class="clue-stat">Est. Clues: ${{lastStep.estimated_clues.toFixed(2)}}</div>`;
            }}
            if (lastStep.centroid_spread !== undefined) {{
                const spreadClass = lastStep.centroid_spread < 1 ? 'bad' : 'good';
                statsHtml += `<div class="clue-stat" style="color: ${{spreadClass === 'bad' ? '#ff4136' : '#2ecc40'}}">Centroid Spread: ${{lastStep.centroid_spread.toFixed(2)}}</div>`;
            }}
            if (lastStep.lora_delta_norms) {{
                const totalNorm = Object.values(lastStep.lora_delta_norms).reduce((a, b) => a + b, 0);
                statsHtml += `<div class="clue-stat">LoRA Σnorm: ${{totalNorm.toFixed(3)}}</div>`;
            }}
            document.getElementById('clueStats').innerHTML = statsHtml;
        }}
        
        function updateAccuracyChart() {{
            const run = traceData.runs[currentRun];
            if (!run) return;
            
            let html = '';
            for (let i = 0; i < run.steps.length; i++) {{
                const acc = run.steps[i].pixel_accuracy;
                const height = Math.max(5, acc * 100);
                const color = acc > 0.9 ? '#2ecc40' : (acc > 0.5 ? '#ffdc00' : '#ff4136');
                html += `<div class="bar" style="height: ${{height}}px; background: ${{color}};">${{(acc * 100).toFixed(0)}}%</div>`;
            }}
            document.getElementById('accuracyChart').innerHTML = html;
        }}
        
        // Initialize
        updateVisualization();
    </script>
</body>
</html>
'''
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ Visualization saved to: {output_path}")


def generate_training_pairs_html(task_viz: TaskVisualization) -> str:
    """Generate HTML for training pairs."""
    html = ''
    for i, (inp, out) in enumerate(zip(task_viz.train_inputs, task_viz.train_outputs)):
        html += f'''
        <div class="pair">
            <span class="pair-label">Pair {i+1}</span>
            {grid_to_svg(inp, cell_size=12)}
            <span style="margin: 0 10px; color: #888;">→</span>
            {grid_to_svg(out, cell_size=12)}
        </div>
        '''
    return html


def generate_run_options(task_viz: TaskVisualization) -> str:
    """Generate HTML select options for runs."""
    html = ''
    for i, run in enumerate(task_viz.runs):
        status = '✓' if run.is_exact_match else f'{run.final_accuracy*100:.0f}%'
        modules = []
        if run.dsc_enabled: modules.append('DSC')
        if run.hyperlora_enabled: modules.append('HyperLoRA')
        if run.solver_context_enabled: modules.append('SolverCtx')
        if run.hpm_enabled: modules.append('HPM')
        module_str = ', '.join(modules) if modules else 'Base'
        html += f'<option value="{i}">{run.config_name} [{module_str}] - {status}</option>'
    return html


def generate_diagnosis_html(task_viz: TaskVisualization) -> str:
    """Generate HTML for diagnosis notes."""
    if not task_viz.diagnosis_notes:
        return '<li style="color: #2ecc40;">No issues detected</li>'
    return '\n'.join([f'<li>{html.escape(note)}</li>' for note in task_viz.diagnosis_notes])


def serialize_task_viz(task_viz: TaskVisualization) -> dict:
    """Serialize TaskVisualization to JSON-safe dict with SVG strings."""
    runs_data = []
    for run in task_viz.runs:
        steps_data = []
        for step in run.steps:
            step_dict = {
                'step_idx': step.step_idx,
                'pixel_accuracy': step.pixel_accuracy,
                'fg_accuracy': step.fg_accuracy,
                'bg_accuracy': step.bg_accuracy,
                'logits_entropy': step.logits_entropy,
                'predicted_svg': grid_to_svg(step.predicted_grid, cell_size=15),
                'diff_svg': diff_mask_svg(step.diff_mask, cell_size=15) if step.diff_mask is not None else None,
                'estimated_clues': step.estimated_clues,
                'centroid_spread': step.centroid_spread,
                'lora_delta_norms': step.lora_delta_norms,
            }
            
            # Attention maps
            if step.attention_maps is not None:
                step_dict['attention_svgs'] = [
                    attention_heatmap_svg(step.attention_maps[k], cell_size=10)
                    for k in range(step.attention_maps.shape[0])
                ]
            
            # Centroids
            if step.centroids is not None:
                step_dict['centroids'] = step.centroids.tolist()
            
            # Stop probs
            if step.stop_probs is not None:
                step_dict['stop_probs'] = step.stop_probs.tolist()
            
            steps_data.append(step_dict)
        
        runs_data.append({
            'config_name': run.config_name,
            'steps': steps_data,
            'final_accuracy': run.final_accuracy,
            'is_exact_match': run.is_exact_match,
            'dsc_enabled': run.dsc_enabled,
            'hyperlora_enabled': run.hyperlora_enabled,
            'solver_context_enabled': run.solver_context_enabled,
            'hpm_enabled': run.hpm_enabled,
        })
    
    return {
        'task_id': task_viz.task_id,
        'runs': runs_data,
    }


# =============================================================================
# MAIN CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='RLAN Visual Debugger - Analyze model behavior on ARC tasks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Single task visualization
  python scripts/visualize_rlan_debug.py --task_id 00d62c1b --checkpoint checkpoints/rlan_256_40pct.pt
  
  # With ablation studies (compare module contributions)
  python scripts/visualize_rlan_debug.py --task_id 00d62c1b --checkpoint checkpoints/rlan_256_40pct.pt --ablations
  
  # Test specific module behavior
  python scripts/visualize_rlan_debug.py --task_id 00d62c1b --checkpoint checkpoints/rlan_256_40pct.pt --test_dsc
  python scripts/visualize_rlan_debug.py --task_id 00d62c1b --checkpoint checkpoints/rlan_256_40pct.pt --test_hyperlora
  
  # Batch mode (multiple tasks)
  python scripts/visualize_rlan_debug.py --task_file failing_tasks.txt --checkpoint checkpoints/rlan_256_40pct.pt --output_dir ./debug_viz
  
  # HPM Buffer Analysis (no task needed - just checkpoint)
  python scripts/visualize_rlan_debug.py --analyze_hpm --checkpoint checkpoints/rlan_256_40pct.pt
  python scripts/visualize_rlan_debug.py --analyze_hpm --checkpoint checkpoints/rlan_256_40pct.pt --output_dir ./hpm_analysis
'''
    )
    
    # Input options (task_id/task_file/task_json OR analyze_hpm)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--task_id', type=str, help='Single task ID to visualize')
    input_group.add_argument('--task_file', type=str, help='File with task IDs (one per line)')
    input_group.add_argument('--task_json', type=str, help='Direct path to task JSON file')
    input_group.add_argument('--analyze_hpm', action='store_true', help='Analyze HPM buffers (static banks & dynamic buffers) - no task needed')
    
    # Model options
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    
    # Data paths
    parser.add_argument('--data_dir', type=str, default='data/arc-agi/data', help='ARC data directory')
    
    # Output options
    parser.add_argument('--output_dir', type=str, default='./debug_viz', help='Output directory for HTML files')
    parser.add_argument('--output_file', type=str, help='Output HTML file (for single task)')
    
    # Test modes
    parser.add_argument('--ablations', action='store_true', help='Run ablation study (compare with/without modules)')
    parser.add_argument('--test_dsc', action='store_true', help='Focus on DSC behavior')
    parser.add_argument('--test_hyperlora', action='store_true', help='Focus on HyperLoRA behavior')
    parser.add_argument('--test_solver', action='store_true', help='Focus on solver refinement')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    device = args.device if torch.cuda.is_available() else 'cpu'
    model, config = load_checkpoint(args.checkpoint, device)
    
    # ============================================
    # HPM Analysis Mode (no task required)
    # ============================================
    if args.analyze_hpm:
        print("\n" + "="*60)
        print("🧠 HPM Buffer Analysis Mode")
        print("="*60)
        
        # Analyze HPM buffers
        hpm_analysis = analyze_hpm_buffers(model)
        
        if not hpm_analysis.banks:
            print("❌ No HPM banks found in model (no static banks or dynamic buffers)")
            return
        
        # Print summary
        print(f"\n📊 HPM Analysis Summary:")
        print(f"  Overall Health Score: {hpm_analysis.overall_health_score:.2%}")
        print(f"  Overall Diversity: {hpm_analysis.overall_diversity:.2%}")
        print(f"  Static Primitives: {hpm_analysis.total_static_primitives}")
        print(f"  Dynamic Entries: {hpm_analysis.total_dynamic_entries}")
        print(f"  Banks Analyzed: {len(hpm_analysis.banks)}")
        
        # Per-bank summary
        for bank_name, bank_analysis in hpm_analysis.banks.items():
            # Health: based on NaN/zero entries
            nan_rate = bank_analysis.num_nan_entries / max(1, bank_analysis.num_primitives)
            zero_rate = bank_analysis.num_zero_entries / max(1, bank_analysis.num_primitives)
            health_score = max(0.0, min(1.0, 1.0 - nan_rate - zero_rate))
            
            # Diversity: inverse of mean cosine similarity (lower cosine = more diverse), clamped
            diversity_score = max(0.0, min(1.0, 1.0 - bank_analysis.mean_pairwise_cosine))
            
            health_icon = "✅" if health_score > 0.9 else ("⚠️" if health_score > 0.5 else "❌")
            diversity_icon = "✅" if diversity_score > 0.5 else ("⚠️" if diversity_score > 0.3 else "❌")
            
            print(f"\n  📦 {bank_name}:")
            print(f"     Shape: ({bank_analysis.num_primitives}, {bank_analysis.embedding_dim})")
            print(f"     {health_icon} Health: {health_score:.2%} (NaN: {bank_analysis.num_nan_entries}, Zero: {bank_analysis.num_zero_entries})")
            print(f"     {diversity_icon} Diversity: {diversity_score:.2%} (mean cosine: {bank_analysis.mean_pairwise_cosine:.3f})")
            print(f"     Effective Rank: {bank_analysis.effective_rank:.1f}")
            print(f"     Clusters: {bank_analysis.num_unique_clusters}")
            if bank_analysis.semantic_hints:
                for hint in bank_analysis.semantic_hints:
                    print(f"     💡 {hint}")
        
        # Generate HTML report
        output_path = os.path.join(args.output_dir, "hpm_analysis.html")
        generate_hpm_html(hpm_analysis, output_path)
        print(f"\n📁 HTML report saved to: {output_path}")
        print("\n✅ HPM Analysis Complete!")
        return
    
    # ============================================
    # Task Visualization Mode (original flow)
    # ============================================
    # Determine task(s) to process
    if args.task_id:
        task_ids = [args.task_id]
    elif args.task_file:
        with open(args.task_file, 'r') as f:
            task_ids = [line.strip() for line in f if line.strip()]
    else:
        task_ids = [Path(args.task_json).stem]
    
    # Data directories to search
    data_dirs = [
        args.data_dir,
        Path(args.data_dir) / 'training',
        Path(args.data_dir) / 'evaluation',
        'data/arc-agi_training_challenges',
        'data/arc-agi_evaluation_challenges',
    ]
    
    # Process each task
    for task_id in task_ids:
        print(f"\n{'='*60}")
        print(f"Processing task: {task_id}")
        print('='*60)
        
        # Find task file
        if args.task_json:
            task_path = args.task_json
        else:
            task_path = find_task_file(task_id, [str(d) for d in data_dirs])
        
        if task_path is None:
            print(f"❌ Task file not found for: {task_id}")
            continue
        
        # Load task
        task = load_task(task_path)
        
        # Handle evaluation challenges that may not have test output
        train_inputs = [np.array(p['input'], dtype=np.int64) for p in task['train']]
        train_outputs = [np.array(p['output'], dtype=np.int64) for p in task['train']]
        test_input = np.array(task['test'][0]['input'], dtype=np.int64)
        
        if 'output' in task['test'][0]:
            test_output = np.array(task['test'][0]['output'], dtype=np.int64)
        else:
            # No ground truth - use zeros with train output shape for display
            test_output = np.zeros_like(train_outputs[0])
            print(f"  ⚠️ No test output available - showing predictions only")
        
        # Create visualization object
        task_viz = TaskVisualization(
            task_id=task_id,
            train_inputs=train_inputs,
            train_outputs=train_outputs,
            test_input=test_input,
            test_output=test_output,
        )
        
        # Define run configurations
        run_configs = [{'name': 'Full Model'}]  # Default: all modules enabled
        
        if args.ablations:
            # Add ablation configurations
            run_configs.extend([
                {'name': 'No HyperLoRA', 'hyperlora_active': False},
                {'name': 'No Solver Context', 'solver_context_active': False},
                {'name': 'Base Only', 'hyperlora_active': False, 'solver_context_active': False},
            ])
        
        if args.test_dsc:
            run_configs = [
                {'name': 'DSC Enabled', 'use_dsc': True},
                {'name': 'DSC Disabled', 'use_dsc': False},
            ]
        
        if args.test_hyperlora:
            run_configs = [
                {'name': 'HyperLoRA ON', 'hyperlora_active': True},
                {'name': 'HyperLoRA OFF', 'hyperlora_active': False},
            ]
        
        if args.test_solver:
            run_configs = [
                {'name': '3 Steps', '_num_steps_override': 3},
                {'name': '5 Steps', '_num_steps_override': 5},
                {'name': '7 Steps', '_num_steps_override': 7},
            ]
        
        # Run inference for each configuration
        for cfg in run_configs:
            print(f"  Running: {cfg['name']}...")
            try:
                trace = run_inference_with_trace(model, task, device, cfg)
                task_viz.runs.append(trace)
                print(f"    → Accuracy: {trace.final_accuracy*100:.1f}%, Exact Match: {trace.is_exact_match}")
            except Exception as e:
                print(f"    ❌ Error: {e}")
        
        # Generate diagnosis
        task_viz.diagnosis_notes = diagnose_task(task_viz)
        
        # Generate HTML
        if args.output_file and len(task_ids) == 1:
            output_path = args.output_file
        else:
            output_path = os.path.join(args.output_dir, f"{task_id}_debug.html")
        
        generate_html(task_viz, output_path)
        
        # Print diagnosis summary
        if task_viz.diagnosis_notes:
            print("\n📋 Diagnosis:")
            for note in task_viz.diagnosis_notes:
                print(f"  {note}")
    
    # Generate index file for batch mode
    if len(task_ids) > 1:
        index_path = os.path.join(args.output_dir, 'index.html')
        with open(index_path, 'w') as f:
            f.write(f'''<!DOCTYPE html>
<html>
<head><title>RLAN Debug Index</title></head>
<body style="font-family: sans-serif; padding: 20px;">
<h1>RLAN Debug Visualizations</h1>
<ul>
{''.join([f'<li><a href="{tid}_debug.html">{tid}</a></li>' for tid in task_ids])}
</ul>
</body>
</html>''')
        print(f"\n📁 Index saved to: {index_path}")
    
    print("\n✅ Done!")


if __name__ == '__main__':
    main()
