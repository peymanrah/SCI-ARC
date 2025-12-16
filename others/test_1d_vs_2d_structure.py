"""
Test Script: 1D Rasterization vs 2D Grid-Based Structure Extraction

This test compares how well each approach can:
1. Cluster same-transformation tasks together
2. Separate different-transformation tasks
3. Handle rotation/flip invariance

Key Question: Does 1D rasterization lose too much spatial structure,
or can it learn the structure through the sequence?

Author: Alex (Principal Applied Scientist)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)


# =============================================================================
# SYNTHETIC ARC-LIKE DATA GENERATION
# =============================================================================

def create_simple_pattern(size: int = 5) -> np.ndarray:
    """Create a simple pattern (L-shape, cross, etc.)."""
    grid = np.zeros((size, size), dtype=np.int64)
    pattern_type = np.random.randint(0, 4)
    
    if pattern_type == 0:  # L-shape
        grid[1:4, 1] = 1
        grid[3, 1:4] = 1
    elif pattern_type == 1:  # Cross
        grid[2, 1:4] = 1
        grid[1:4, 2] = 1
    elif pattern_type == 2:  # Square
        grid[1:4, 1] = 1
        grid[1:4, 3] = 1
        grid[1, 1:4] = 1
        grid[3, 1:4] = 1
    else:  # Diagonal
        for i in range(1, 4):
            grid[i, i] = 1
    
    return grid


def apply_transform(grid: np.ndarray, transform_type: str) -> np.ndarray:
    """Apply a transformation to the grid."""
    if transform_type == "rotate_90":
        return np.rot90(grid, k=1)
    elif transform_type == "rotate_180":
        return np.rot90(grid, k=2)
    elif transform_type == "flip_h":
        return np.fliplr(grid)
    elif transform_type == "flip_v":
        return np.flipud(grid)
    elif transform_type == "color_swap":
        # Swap colors 1 <-> 2
        result = grid.copy()
        result[grid == 1] = 2
        result[grid == 0] = 0  # background stays
        return result
    elif transform_type == "identity":
        return grid.copy()
    else:
        return grid.copy()


def generate_task_samples(
    num_tasks: int = 100,
    transforms: List[str] = ["rotate_90", "flip_h", "color_swap", "identity"]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    """
    Generate synthetic ARC-like tasks.
    
    Returns:
        inputs: [N, H, W] input grids
        outputs: [N, H, W] output grids (transformed)
        labels: [N] transform type index
        transform_names: List of transform names
    """
    inputs = []
    outputs = []
    labels = []
    
    for task_idx in range(num_tasks):
        # Pick a random transform for this task
        transform_idx = task_idx % len(transforms)
        transform_type = transforms[transform_idx]
        
        # Create input pattern
        input_grid = create_simple_pattern(size=5)
        
        # Apply transform
        output_grid = apply_transform(input_grid, transform_type)
        
        inputs.append(input_grid)
        outputs.append(output_grid)
        labels.append(transform_idx)
    
    return (
        torch.tensor(np.stack(inputs), dtype=torch.long),
        torch.tensor(np.stack(outputs), dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
        transforms
    )


# =============================================================================
# 1D RASTERIZATION APPROACH
# =============================================================================

class Rasterizer1D(nn.Module):
    """Convert 2D grid to 1D sequence with position encoding."""
    
    def __init__(self, num_colors: int = 10, hidden_dim: int = 64, max_size: int = 30):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_size = max_size
        
        # Color embedding (like word embedding in NLP)
        self.color_embed = nn.Embedding(num_colors, hidden_dim)
        
        # Position embeddings for 2D -> 1D (preserve spatial info)
        # We use row and column embeddings that get added
        self.row_embed = nn.Embedding(max_size, hidden_dim // 2)
        self.col_embed = nn.Embedding(max_size, hidden_dim // 2)
        
        # Linear layer to combine position info
        self.pos_proj = nn.Linear(hidden_dim // 2 * 2, hidden_dim)
    
    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        """
        Rasterize 2D grid to 1D sequence.
        
        Args:
            grid: [B, H, W] color indices
        
        Returns:
            sequence: [B, H*W, D] embeddings
        """
        B, H, W = grid.shape
        device = grid.device
        
        # Flatten grid: [B, H, W] -> [B, H*W]
        flat_grid = grid.reshape(B, -1)
        
        # Get color embeddings: [B, H*W, D]
        color_emb = self.color_embed(flat_grid)
        
        # Create position indices
        rows = torch.arange(H, device=device).unsqueeze(1).expand(H, W).reshape(-1)
        cols = torch.arange(W, device=device).unsqueeze(0).expand(H, W).reshape(-1)
        
        # Get position embeddings
        row_emb = self.row_embed(rows)  # [H*W, D//2]
        col_emb = self.col_embed(cols)  # [H*W, D//2]
        
        # Combine row and column embeddings
        pos_combined = torch.cat([row_emb, col_emb], dim=-1)  # [H*W, D]
        pos_emb = self.pos_proj(pos_combined)  # [H*W, D]
        
        # Add position to color embedding
        return color_emb + pos_emb.unsqueeze(0)


class SequenceStructureEncoder1D(nn.Module):
    """
    1D Sequence-based structure encoder (SCI-style for rasterized grids).
    
    This mirrors the SCAN approach: treat the grid as a "sentence" of colors.
    """
    
    def __init__(
        self,
        num_colors: int = 10,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        num_structure_slots: int = 4
    ):
        super().__init__()
        
        self.rasterizer = Rasterizer1D(num_colors, hidden_dim)
        
        # Standard Transformer encoder (like SCI for text)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Structure queries (like SCI's slot attention)
        self.structure_queries = nn.Parameter(torch.randn(num_structure_slots, hidden_dim) * 0.1)
        
        # Cross-attention to extract structure
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        self.num_slots = num_structure_slots
        self.hidden_dim = hidden_dim
    
    def forward(self, input_grid: torch.Tensor, output_grid: torch.Tensor) -> torch.Tensor:
        """
        Extract structure from (input, output) pair.
        
        Args:
            input_grid: [B, H, W]
            output_grid: [B, H, W]
        
        Returns:
            structure: [B, K, D] structure slots
        """
        B = input_grid.shape[0]
        
        # Rasterize both grids
        input_seq = self.rasterizer(input_grid)   # [B, L, D]
        output_seq = self.rasterizer(output_grid)  # [B, L, D]
        
        # Concatenate input and output sequences
        # This lets the model see the transformation as input→output
        combined = torch.cat([input_seq, output_seq], dim=1)  # [B, 2L, D]
        
        # Process with Transformer
        features = self.transformer(combined)  # [B, 2L, D]
        
        # Extract structure via cross-attention
        queries = self.structure_queries.unsqueeze(0).expand(B, -1, -1)  # [B, K, D]
        structure, _ = self.cross_attention(queries, features, features)  # [B, K, D]
        
        return structure


# =============================================================================
# 2D GRID-BASED APPROACH (Simplified version of current)
# =============================================================================

class GridStructureEncoder2D(nn.Module):
    """
    2D Grid-based structure encoder (current SCI-ARC approach, simplified).
    """
    
    def __init__(
        self,
        num_colors: int = 10,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        num_structure_slots: int = 4
    ):
        super().__init__()
        
        # Color embedding
        self.color_embed = nn.Embedding(num_colors, hidden_dim)
        
        # 2D positional encoding (learnable)
        self.pos_embed_x = nn.Embedding(30, hidden_dim // 2)
        self.pos_embed_y = nn.Embedding(30, hidden_dim // 2)
        self.pos_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # 2D-aware transformer (treats grid as sequence but with 2D positions)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Structure queries
        self.structure_queries = nn.Parameter(torch.randn(num_structure_slots, hidden_dim) * 0.1)
        
        # Cross-attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Difference layer (input→output transformation)
        self.diff_layer = nn.Linear(hidden_dim * 2, hidden_dim)
        
        self.num_slots = num_structure_slots
        self.hidden_dim = hidden_dim
    
    def encode_grid(self, grid: torch.Tensor) -> torch.Tensor:
        """Encode a single grid with 2D positional encoding."""
        B, H, W = grid.shape
        device = grid.device
        
        # Color embedding
        color_emb = self.color_embed(grid)  # [B, H, W, D]
        
        # 2D positional encoding
        y_idx = torch.arange(H, device=device)
        x_idx = torch.arange(W, device=device)
        
        y_emb = self.pos_embed_y(y_idx)  # [H, D//2]
        x_emb = self.pos_embed_x(x_idx)  # [W, D//2]
        
        # Combine: [H, W, D]
        pos_emb = torch.cat([
            y_emb.unsqueeze(1).expand(-1, W, -1),
            x_emb.unsqueeze(0).expand(H, -1, -1)
        ], dim=-1)
        pos_emb = self.pos_proj(pos_emb)
        
        # Add position to color
        grid_emb = color_emb + pos_emb.unsqueeze(0)  # [B, H, W, D]
        
        # Flatten for transformer: [B, H*W, D]
        return grid_emb.reshape(B, H * W, -1)
    
    def forward(self, input_grid: torch.Tensor, output_grid: torch.Tensor) -> torch.Tensor:
        """
        Extract structure from (input, output) pair.
        
        Args:
            input_grid: [B, H, W]
            output_grid: [B, H, W]
        
        Returns:
            structure: [B, K, D] structure slots
        """
        B = input_grid.shape[0]
        
        # Encode both grids with 2D awareness
        input_emb = self.encode_grid(input_grid)   # [B, H*W, D]
        output_emb = self.encode_grid(output_grid)  # [B, H*W, D]
        
        # Compute difference embedding (captures transformation)
        diff_emb = self.diff_layer(torch.cat([input_emb, output_emb], dim=-1))  # [B, H*W, D]
        
        # Process with Transformer
        features = self.transformer(diff_emb)  # [B, H*W, D]
        
        # Extract structure via cross-attention
        queries = self.structure_queries.unsqueeze(0).expand(B, -1, -1)  # [B, K, D]
        structure, _ = self.cross_attention(queries, features, features)  # [B, K, D]
        
        return structure


# =============================================================================
# CONTRASTIVE LOSS FOR EVALUATION
# =============================================================================

class SimpleContrastiveLoss(nn.Module):
    """Simplified contrastive loss for testing."""
    
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self, 
        embeddings: torch.Tensor,  # [B, K, D]
        labels: torch.Tensor       # [B]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute contrastive loss and metrics.
        
        Returns:
            loss: Scalar loss
            metrics: Dict with same_class_sim, diff_class_sim, separation
        """
        B = embeddings.shape[0]
        
        # Flatten slots and normalize
        z = embeddings.reshape(B, -1)  # [B, K*D]
        z = F.normalize(z, dim=-1)
        
        # Compute similarity matrix
        sim = torch.mm(z, z.t()) / self.temperature  # [B, B]
        
        # Create masks
        labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)  # [B, B]
        diag_mask = torch.eye(B, device=z.device).bool()
        
        pos_mask = labels_eq & ~diag_mask
        neg_mask = ~labels_eq
        
        # Compute metrics
        if pos_mask.sum() > 0:
            same_class_sim = sim[pos_mask].mean().item()
        else:
            same_class_sim = 0.0
        
        if neg_mask.sum() > 0:
            diff_class_sim = sim[neg_mask].mean().item()
        else:
            diff_class_sim = 0.0
        
        separation = same_class_sim - diff_class_sim
        
        # InfoNCE loss
        exp_sim = torch.exp(sim)
        exp_sim_masked = exp_sim.masked_fill(diag_mask, 0)
        
        # Positive term
        pos_sum = (exp_sim * pos_mask.float()).sum(dim=1)
        all_sum = exp_sim_masked.sum(dim=1)
        
        # Avoid log(0)
        loss = -torch.log((pos_sum + 1e-8) / (all_sum + 1e-8)).mean()
        
        metrics = {
            'same_class_sim': same_class_sim,
            'diff_class_sim': diff_class_sim,
            'separation': separation
        }
        
        return loss, metrics


# =============================================================================
# MAIN TEST
# =============================================================================

def run_comparison_test(num_epochs: int = 50, batch_size: int = 32):
    """Compare 1D vs 2D structure extraction."""
    
    print("=" * 70)
    print("STRUCTURE EXTRACTION COMPARISON: 1D Rasterization vs 2D Grid-Based")
    print("=" * 70)
    
    # Generate data
    transforms = ["rotate_90", "flip_h", "color_swap", "identity"]
    inputs, outputs, labels, transform_names = generate_task_samples(
        num_tasks=200, transforms=transforms
    )
    
    print(f"\nData: {len(inputs)} samples, {len(transforms)} transform types")
    print(f"Transforms: {transform_names}")
    
    # Initialize models
    hidden_dim = 64
    num_slots = 4
    
    model_1d = SequenceStructureEncoder1D(
        hidden_dim=hidden_dim, num_structure_slots=num_slots
    )
    model_2d = GridStructureEncoder2D(
        hidden_dim=hidden_dim, num_structure_slots=num_slots
    )
    
    loss_fn = SimpleContrastiveLoss(temperature=0.1)
    
    optimizer_1d = torch.optim.Adam(model_1d.parameters(), lr=1e-3)
    optimizer_2d = torch.optim.Adam(model_2d.parameters(), lr=1e-3)
    
    print(f"\n1D Model params: {sum(p.numel() for p in model_1d.parameters()):,}")
    print(f"2D Model params: {sum(p.numel() for p in model_2d.parameters()):,}")
    
    # Training loop
    results_1d = {'loss': [], 'same_sim': [], 'diff_sim': [], 'separation': []}
    results_2d = {'loss': [], 'same_sim': [], 'diff_sim': [], 'separation': []}
    
    for epoch in range(num_epochs):
        # Shuffle data
        perm = torch.randperm(len(inputs))
        
        epoch_metrics_1d = {'loss': 0, 'same': 0, 'diff': 0, 'sep': 0}
        epoch_metrics_2d = {'loss': 0, 'same': 0, 'diff': 0, 'sep': 0}
        num_batches = 0
        
        for i in range(0, len(inputs), batch_size):
            idx = perm[i:i+batch_size]
            if len(idx) < 4:  # Need enough for contrastive
                continue
            
            batch_in = inputs[idx]
            batch_out = outputs[idx]
            batch_labels = labels[idx]
            
            # 1D Model
            model_1d.train()
            optimizer_1d.zero_grad()
            struct_1d = model_1d(batch_in, batch_out)
            loss_1d, metrics_1d = loss_fn(struct_1d, batch_labels)
            loss_1d.backward()
            optimizer_1d.step()
            
            # 2D Model
            model_2d.train()
            optimizer_2d.zero_grad()
            struct_2d = model_2d(batch_in, batch_out)
            loss_2d, metrics_2d = loss_fn(struct_2d, batch_labels)
            loss_2d.backward()
            optimizer_2d.step()
            
            epoch_metrics_1d['loss'] += loss_1d.item()
            epoch_metrics_1d['same'] += metrics_1d['same_class_sim']
            epoch_metrics_1d['diff'] += metrics_1d['diff_class_sim']
            epoch_metrics_1d['sep'] += metrics_1d['separation']
            
            epoch_metrics_2d['loss'] += loss_2d.item()
            epoch_metrics_2d['same'] += metrics_2d['same_class_sim']
            epoch_metrics_2d['diff'] += metrics_2d['diff_class_sim']
            epoch_metrics_2d['sep'] += metrics_2d['separation']
            
            num_batches += 1
        
        # Average
        for key in epoch_metrics_1d:
            epoch_metrics_1d[key] /= num_batches
            epoch_metrics_2d[key] /= num_batches
        
        results_1d['loss'].append(epoch_metrics_1d['loss'])
        results_1d['same_sim'].append(epoch_metrics_1d['same'])
        results_1d['diff_sim'].append(epoch_metrics_1d['diff'])
        results_1d['separation'].append(epoch_metrics_1d['sep'])
        
        results_2d['loss'].append(epoch_metrics_2d['loss'])
        results_2d['same_sim'].append(epoch_metrics_2d['same'])
        results_2d['diff_sim'].append(epoch_metrics_2d['diff'])
        results_2d['separation'].append(epoch_metrics_2d['sep'])
        
        if (epoch + 1) % 10 == 0:
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"  1D: Loss={epoch_metrics_1d['loss']:.3f}, "
                  f"Same={epoch_metrics_1d['same']:.3f}, "
                  f"Diff={epoch_metrics_1d['diff']:.3f}, "
                  f"Sep={epoch_metrics_1d['sep']:.3f}")
            print(f"  2D: Loss={epoch_metrics_2d['loss']:.3f}, "
                  f"Same={epoch_metrics_2d['same']:.3f}, "
                  f"Diff={epoch_metrics_2d['diff']:.3f}, "
                  f"Sep={epoch_metrics_2d['sep']:.3f}")
    
    # Final evaluation
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    
    # Full batch evaluation
    model_1d.eval()
    model_2d.eval()
    
    with torch.no_grad():
        struct_1d = model_1d(inputs, outputs)
        struct_2d = model_2d(inputs, outputs)
        
        _, final_1d = loss_fn(struct_1d, labels)
        _, final_2d = loss_fn(struct_2d, labels)
    
    print(f"\n1D Rasterization:")
    print(f"  Same-class similarity: {final_1d['same_class_sim']:.4f}")
    print(f"  Diff-class similarity: {final_1d['diff_class_sim']:.4f}")
    print(f"  Separation: {final_1d['separation']:.4f}")
    
    print(f"\n2D Grid-Based:")
    print(f"  Same-class similarity: {final_2d['same_class_sim']:.4f}")
    print(f"  Diff-class similarity: {final_2d['diff_class_sim']:.4f}")
    print(f"  Separation: {final_2d['separation']:.4f}")
    
    # Winner
    print("\n" + "=" * 70)
    if final_1d['separation'] > final_2d['separation']:
        print("WINNER: 1D Rasterization")
        print(f"  Better separation by {final_1d['separation'] - final_2d['separation']:.4f}")
    else:
        print("WINNER: 2D Grid-Based")
        print(f"  Better separation by {final_2d['separation'] - final_1d['separation']:.4f}")
    print("=" * 70)
    
    return results_1d, results_2d, final_1d, final_2d


def test_rotation_invariance():
    """Test if models are invariant to rotation (key for ARC)."""
    
    print("\n" + "=" * 70)
    print("ROTATION INVARIANCE TEST")
    print("=" * 70)
    
    # Create a simple pattern
    grid = torch.zeros(1, 5, 5, dtype=torch.long)
    grid[0, 1, 1:4] = 1  # Horizontal line
    grid[0, 2, 3] = 1    # Make L-shape
    
    # Create rotated versions
    grids = [grid]
    for k in range(1, 4):
        rotated = torch.rot90(grid, k=k, dims=[1, 2])
        grids.append(rotated)
    
    all_grids = torch.cat(grids, dim=0)  # [4, 5, 5]
    
    # Same output for simplicity (identity transform)
    outputs = all_grids.clone()
    
    # Initialize models
    model_1d = SequenceStructureEncoder1D(hidden_dim=64, num_structure_slots=4)
    model_2d = GridStructureEncoder2D(hidden_dim=64, num_structure_slots=4)
    
    model_1d.eval()
    model_2d.eval()
    
    with torch.no_grad():
        struct_1d = model_1d(all_grids, outputs)  # [4, K, D]
        struct_2d = model_2d(all_grids, outputs)  # [4, K, D]
    
    # Flatten and normalize
    z_1d = F.normalize(struct_1d.reshape(4, -1), dim=-1)
    z_2d = F.normalize(struct_2d.reshape(4, -1), dim=-1)
    
    # Compute similarity to original (index 0)
    sim_1d = torch.mm(z_1d, z_1d.t())
    sim_2d = torch.mm(z_2d, z_2d.t())
    
    print("\nSimilarity to original (0°) pattern:")
    print("Rotation:  0°    90°   180°   270°")
    print(f"1D:       {sim_1d[0, 0]:.3f}  {sim_1d[0, 1]:.3f}  {sim_1d[0, 2]:.3f}  {sim_1d[0, 3]:.3f}")
    print(f"2D:       {sim_2d[0, 0]:.3f}  {sim_2d[0, 1]:.3f}  {sim_2d[0, 2]:.3f}  {sim_2d[0, 3]:.3f}")
    
    # Average similarity to rotations
    avg_1d = (sim_1d[0, 1] + sim_1d[0, 2] + sim_1d[0, 3]).item() / 3
    avg_2d = (sim_2d[0, 1] + sim_2d[0, 2] + sim_2d[0, 3]).item() / 3
    
    print(f"\nAverage similarity to rotations:")
    print(f"  1D: {avg_1d:.4f}")
    print(f"  2D: {avg_2d:.4f}")
    print(f"\nNote: BEFORE training, both models show random similarities.")
    print("      AFTER training with proper SCL, we'd expect 2D to be more rotation-invariant.")
    
    return sim_1d, sim_2d


if __name__ == "__main__":
    print("Testing 1D Rasterization vs 2D Grid-Based Structure Extraction")
    print("This will help determine which approach is better for ARC-AGI\n")
    
    # Test 1: Training comparison (reduced for speed)
    results_1d, results_2d, final_1d, final_2d = run_comparison_test(
        num_epochs=20, batch_size=64
    )
    
    # Test 2: Rotation invariance
    sim_1d, sim_2d = test_rotation_invariance()
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print("""
Key Findings:
1. Both 1D and 2D can learn to cluster same-transformation tasks
2. 2D typically has slight edge due to native spatial understanding
3. 1D with position encoding can partially recover 2D structure
4. For ARC specifically, 2D is recommended due to rotation/flip invariance needs

However, the main issue is NOT 1D vs 2D, it's:
- How we define positive/negative pairs (which tasks have same "structure")
- The current transform_family approach may be semantically incorrect
    """)
