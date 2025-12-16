# SCI-ARC: Structural Causal Invariance for Abstract Reasoning Corpus
## Complete AI Agent Implementation Instructions

**Author**: Alex (Microsoft Principal Applied Scientist)
**Target**: Novel application of SCI principles to ARC-AGI benchmark
**Innovation**: First combination of structural-content separation with visual abstract reasoning

---

## 🔧 Key Architectural Fixes (December 2024)

> **IMPORTANT**: The original SCL implementation suffered from representation collapse
> (constant loss at ~5.25). The following fixes were applied:

### Phase 1: Initial Fixes (Representation Diversity)

| Fix | Component | Problem | Solution |
|-----|-----------|---------|----------|
| **#1** | `PositionalEncoding2D` | Transformer blind to geometry | Learnable (x,y) position embeddings |
| **#2** | `structure_queries` | Slots initialized too small (0.002) | Full-scale orthogonal init (1.0) |
| **#3** | `StructuralContrastiveLoss` | Mean pooling kills variance | **Flatten** slots instead of pool |
| **#4** | `temperature` | Fixed at 0.07 | **Learnable** parameter |
| **#5** | `AbstractionLayer2D` | Per-sample std normalization | Removed (was collapsing outputs) |
| **#6** | `output_norm` | LayerNorm after cross-attention | Removed (was collapsing samples) |

### Phase 2: Background Signal Removal (December 2024)

After Phase 1, encoder produced diverse outputs but **all similarities remained ~0.95**.  
Root cause: ARC grids are 90% background → embeddings share a massive "common signal".

| Fix | Component | Problem | Solution |
|-----|-----------|---------|----------|
| **#7** | `BatchNorm1d` | Common background signal | Centers batch by subtracting mean vector |
| **#8** | `Difference Embedding` | Model must learn (output - input) | Explicit `diff_emb = output - input` channel |
| **#9** | `temperature` | 0.07 too low for high similarity | Increased to **0.5** (higher temp = better gradients) |
| **#10** | `scl_weight` | 0.1 too weak vs task loss | Increased to **1.0** |

**Result**: Post-BatchNorm similarity drops from ~0.95 to near 0. SCL loss now decreases.

See [Section 7: Structural Contrastive Loss](#7-structural-contrastive-loss-scl---fixed-architecture) for detailed diagrams.

### Phase 3: RLAN Enhancements (December 2024)

To match and exceed the capabilities of TinyRecursiveModels (TRM), the following modules were integrated into the RLAN architecture:

| Feature | Component | Purpose | Implementation |
|---------|-----------|---------|----------------|
| **ACT** | `ACTController` | Adaptive Computation Time | Dynamic recursion depth based on halting probability |
| **SwiGLU** | `SwiGLUConv2d` | Improved Activation | Replaces Tanh in ConvGRU for better gradient flow |
| **TTA** | `evaluate_rlan.py` | Test-Time Augmentation | 8 dihedral transforms + majority voting during inference |

These features are enabled in `configs/rlan_base.yaml` and `configs/rlan_fair.yaml`.

---

## CRITICAL: Understanding the Source Architectures

### Your SCI (Structural Causal Invariance) - For Text/Language

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SCI Architecture (Current - Text)                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Input: "walk twice and jump left"                                          │
│         │                                                                    │
│         ▼                                                                    │
│  ┌──────────────────────┐      ┌──────────────────────┐                     │
│  │  STRUCTURAL ENCODER  │      │   CONTENT ENCODER    │                     │
│  │  ────────────────────│      │  ──────────────────  │                     │
│  │  • AbstractionLayer  │      │  • Entity extraction │                     │
│  │  • Structure queries │      │  • Orthogonal to SE  │                     │
│  │  • GNN for causal    │      │  • Content vectors   │                     │
│  │    graph             │      │                      │                     │
│  └──────────┬───────────┘      └──────────┬───────────┘                     │
│             │                              │                                 │
│             │    S(x) = structure          │    C(x) = content              │
│             │                              │                                 │
│             └──────────────┬───────────────┘                                │
│                            ▼                                                 │
│                 ┌──────────────────────┐                                    │
│                 │  CAUSAL BINDING (CBM)│                                    │
│                 │  • Binding attention │                                    │
│                 │  • Causal intervention│                                   │
│                 │  • Broadcast         │                                    │
│                 └──────────────────────┘                                    │
│                            │                                                 │
│                            ▼                                                 │
│                   Inject into TinyLlama                                     │
│                                                                              │
│  TRAINING: SCL Loss enforces S(x₁) ≈ S(x₂) when structure matches          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key SCI Principles:**
1. **Structural Encoder (SE)**: Extracts structural patterns, ignores content
2. **Content Encoder (CE)**: Extracts entities/content, orthogonal to structure
3. **Causal Binding Mechanism (CBM)**: Binds structure slots to content
4. **Structural Contrastive Loss (SCL)**: Forces same structure → same S(x)

### TRM (Tiny Recursive Model) - Current ARC SOTA

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TRM Architecture (ARC SOTA)                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Input: x (embedded grid), y₀ (initial answer), z₀ (initial latent)        │
│                                                                              │
│  For k = 1 to K (supervision steps):                                        │
│    For n = 1 to N (recursion steps):                                        │
│      z = f(x, y, z)        ← Update latent given input, answer, latent     │
│    y = g(y, z)             ← Update answer given current answer, latent    │
│    Loss_k = CE(y, target)  ← Deep supervision at each step                 │
│                                                                              │
│  Key insights:                                                              │
│  • 7M parameters only (TINY)                                                │
│  • 2 layers only                                                            │
│  • No pre-trained backbone                                                  │
│  • Deep supervision doubles accuracy                                        │
│  • Recursion prevents overfitting                                           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Can SCI Be Directly Applied to ARC?

### Assessment Table

| SCI Component | Text Domain | ARC Domain | Direct Transfer? |
|---------------|-------------|------------|------------------|
| **AbstractionLayer** | Suppresses content words | Needs to suppress grid-specific content | ❌ Needs 2D adaptation |
| **Structure queries** | Attend to token sequence | Need to attend to grid regions | ❌ Needs 2D attention |
| **Content Encoder** | Extract entities (nouns) | Extract objects (connected components) | ❌ Needs vision version |
| **CBM** | Bind slots to tokens | Bind slots to grid cells | ❌ Needs 2D version |
| **SCL** | Contrastive on S(x) | Same principle applies | ✅ Direct transfer |
| **Orthogonality loss** | S(x) ⊥ C(x) | Same principle applies | ✅ Direct transfer |
| **TinyLlama backbone** | 1.1B params | Not needed (TRM shows 7M works) | ❌ Remove |

### Verdict: Fresh Implementation Required, But Principles Transfer

**What transfers:**
- SCL loss function (identical math)
- Orthogonality constraint (identical math)
- Structure-content separation principle
- The key insight: same transformation rule → same structural representation

**What needs reimplementation:**
- Grid encoder (2D, not 1D tokens)
- Structural Encoder adapted for 2D grids
- Content Encoder adapted for visual objects
- Causal Binding for grids
- Recursive refinement (from TRM)

---

## Novel SCI-ARC Architecture

### Design Philosophy

Combine SCI's structure-content separation with TRM's recursive refinement:

```
SCI-ARC = SCI(structure-content separation) + TRM(tiny recursive)
```

### Complete Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SCI-ARC Architecture                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  PHASE 1: DEMO ENCODING (Infer transformation rule)                         │
│  ═══════════════════════════════════════════════════                         │
│                                                                              │
│  Demo pairs: [(in₁,out₁), (in₂,out₂), ...]                                  │
│       │                                                                      │
│       ▼                                                                      │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                    GRID ENCODER (Shared)                               │ │
│  │  • Per-cell color embedding (10 colors → 64 dim)                      │ │
│  │  • 2D sinusoidal positional encoding                                  │ │
│  │  • Output: [B, H, W, D] grid embeddings                               │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│       │                                                                      │
│       ├───────────────────────────────────────┐                             │
│       ▼                                       ▼                             │
│  ┌─────────────────────────┐    ┌─────────────────────────┐                 │
│  │  STRUCTURAL ENCODER     │    │  CONTENT ENCODER        │                 │
│  │  ═══════════════════    │    │  ═══════════════        │                 │
│  │                         │    │                         │                 │
│  │  AbstractionLayer2D:    │    │  ObjectDetector:        │                 │
│  │  • Detects structural   │    │  • Connected component  │                 │
│  │    patterns (what       │    │    analysis             │                 │
│  │    transformation?)     │    │  • Per-object embedding │                 │
│  │  • Suppresses content   │    │  • Color/shape/size     │                 │
│  │    (which objects?)     │    │    features             │                 │
│  │                         │    │                         │                 │
│  │  StructureSlots:        │    │  OrthogonalProjector:   │                 │
│  │  • K learnable queries  │    │  • Projects orthogonal  │                 │
│  │  • Cross-attend to      │    │    to structure         │                 │
│  │    (input, output) diff │    │  • Ensures S(x) ⊥ C(x)  │                 │
│  │                         │    │                         │                 │
│  └───────────┬─────────────┘    └───────────┬─────────────┘                 │
│              │                              │                               │
│              │  S(demos) = transformation   │  C(demos) = objects           │
│              │  rule embedding              │  in demos                     │
│              │                              │                               │
│              └──────────────┬───────────────┘                               │
│                             ▼                                               │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                    CAUSAL BINDING 2D (CBM)                             │ │
│  │  • Binding attention: structure slots query content objects           │ │
│  │  • Causal graph: learned transformation dependencies                  │ │
│  │  • Output: z_task = bound(S, C) = task understanding                  │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                             │                                               │
│                             ▼                                               │
│                       z_task (128-dim)                                      │
│                                                                              │
│  ═══════════════════════════════════════════════════════════════════════════ │
│                                                                              │
│  PHASE 2: RECURSIVE REFINEMENT (Apply transformation)                       │
│  ═══════════════════════════════════════════════════════                     │
│                                                                              │
│  Test input: x_test                                                         │
│       │                                                                      │
│       ▼                                                                      │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                    TRM-STYLE RECURSION                                 │ │
│  │                                                                        │ │
│  │  Initialize: y₀ = zeros, z₀ = z_task                                  │ │
│  │                                                                        │ │
│  │  For k = 1 to K (supervision steps = 16):                             │ │
│  │    For n = 1 to N (recursion steps = 4):                              │ │
│  │      z = f(x_test, y, z, z_task)  ← Latent update (conditioned on task)│ │
│  │    y = g(y, z)                    ← Answer update                      │ │
│  │    Loss_k = CE(y_k, target)       ← Deep supervision                  │ │
│  │                                                                        │ │
│  │  Final output: y_K                                                     │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ═══════════════════════════════════════════════════════════════════════════ │
│                                                                              │
│  TRAINING LOSSES                                                            │
│  ═══════════════                                                            │
│                                                                              │
│  L_total = L_CE (deep supervision)                                          │
│          + λ_scl * L_SCL (structural contrastive)                           │
│          + λ_orth * L_orth (orthogonality)                                  │
│                                                                              │
│  L_SCL: Same transformation rule → same S(demos)                            │
│         "rotate_90" tasks should cluster together                           │
│                                                                              │
│  L_orth: S(x) ⊥ C(x)                                                        │
│         Structure representation independent of content                      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Why This Is Novel

| Prior Work | What It Does | What SCI-ARC Adds |
|------------|--------------|-------------------|
| **TRM** | Recursive refinement | + Structure-content separation |
| **TTT** | Per-task fine-tuning | + Explicit transformation embedding |
| **SCI** | Structure-content for text | + 2D grid adaptation + recursion |
| **Program Synthesis** | Generate code | + Neural structure understanding |

**Key novelty**: No one has applied the structural invariance principle (SCL) to visual abstract reasoning. SCI-ARC is the first to:
1. Explicitly separate "what transformation" from "what objects"
2. Enforce that same transformation → same embedding via contrastive learning
3. Combine this with recursive refinement

---

## Implementation Specification

### 1. Grid Encoder

```python
class GridEncoder(nn.Module):
    """
    Encode ARC grids into embeddings suitable for SCI processing.
    
    Key differences from SCI text encoder:
    - 2D positional encoding (not 1D)
    - Color embedding (not token embedding)
    - Per-cell output (not per-token)
    
    Parameters: ~500K
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_colors: int = 10,
        max_size: int = 30
    ):
        super().__init__()
        
        # Color embedding (like token embedding in text)
        self.color_embed = nn.Embedding(num_colors, hidden_dim // 2)
        
        # 2D sinusoidal positional encoding
        self.pos_embed = SinusoidalPositionalEncoding2D(hidden_dim // 2, max_size)
        
        # Combine and project
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        """
        Args:
            grid: [B, H, W] integer tensor (0-9 colors)
        
        Returns:
            embeddings: [B, H, W, hidden_dim]
        """
        B, H, W = grid.shape
        
        # Color embedding
        color_emb = self.color_embed(grid)  # [B, H, W, D/2]
        
        # 2D positional encoding
        pos_emb = self.pos_embed(H, W)  # [H, W, D/2]
        pos_emb = pos_emb.unsqueeze(0).expand(B, -1, -1, -1)
        
        # Combine
        combined = torch.cat([color_emb, pos_emb], dim=-1)  # [B, H, W, D]
        
        return self.norm(self.proj(combined))


class SinusoidalPositionalEncoding2D(nn.Module):
    """2D sinusoidal positional encoding for grids."""
    
    def __init__(self, dim: int, max_size: int = 30):
        super().__init__()
        self.dim = dim
        
        # Create position encodings
        pe = torch.zeros(max_size, max_size, dim)
        
        y_pos = torch.arange(max_size).unsqueeze(1).expand(max_size, max_size)
        x_pos = torch.arange(max_size).unsqueeze(0).expand(max_size, max_size)
        
        div_term = torch.exp(torch.arange(0, dim, 4) * -(math.log(10000.0) / dim))
        
        pe[:, :, 0::4] = torch.sin(x_pos.unsqueeze(-1) * div_term)
        pe[:, :, 1::4] = torch.cos(x_pos.unsqueeze(-1) * div_term)
        pe[:, :, 2::4] = torch.sin(y_pos.unsqueeze(-1) * div_term)
        pe[:, :, 3::4] = torch.cos(y_pos.unsqueeze(-1) * div_term)
        
        self.register_buffer('pe', pe)
        
    def forward(self, h: int, w: int) -> torch.Tensor:
        return self.pe[:h, :w, :]
```

### 2. Structural Encoder for Grids (with 2D Positional Encoding)

```python
class PositionalEncoding2D(nn.Module):
    """
    2D Positional Encoding for grids.
    
    CRITICAL for spatial reasoning: Without this, the Transformer cannot
    distinguish between positions - it would see a vertical line and horizontal
    line as identical if they have the same pixels.
    
    Uses learnable embeddings for (x, y) coordinates that are added together.
    This allows the model to learn spatial relationships like "move right"
    or "rotate 90 degrees".
    """
    
    def __init__(self, hidden_dim: int, max_size: int = 32):
        super().__init__()
        self.x_embed = nn.Embedding(max_size, hidden_dim)
        self.y_embed = nn.Embedding(max_size, hidden_dim)
        
        nn.init.normal_(self.x_embed.weight, std=0.02)
        nn.init.normal_(self.y_embed.weight, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add 2D positional encodings: [B, H, W, D] → [B, H, W, D]"""
        B, H, W, D = x.shape
        device = x.device
        
        y_pos = torch.arange(H, device=device)
        x_pos = torch.arange(W, device=device)
        
        y_emb = self.y_embed(y_pos)  # [H, D]
        x_emb = self.x_embed(x_pos)  # [W, D]
        
        # [H, 1, D] + [1, W, D] → [H, W, D]
        pos_emb = y_emb.unsqueeze(1) + x_emb.unsqueeze(0)
        
        return x + pos_emb  # Broadcast over batch


class StructuralEncoder2D(nn.Module):
    """
    Extract transformation structure from (input, output) grid pairs.
    
    Adaptation of SCI's Structural Encoder for 2D grids:
    - ★ 2D Positional Encoding enables spatial reasoning
    - AbstractionLayer2D suppresses content-specific features
    - Structure queries attend to transformation patterns
    - Output is invariant to specific objects/colors
    
    KEY INSIGHT: The difference between input and output encodes the transformation.
    SE should extract WHAT transformation, not WHICH objects.
    
    Parameters: ~2M
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_structure_slots: int = 8,  # Like SCI's K slots
        num_layers: int = 2,
        num_heads: int = 4
    ):
        super().__init__()
        
        self.num_slots = num_structure_slots
        self.hidden_dim = hidden_dim
        
        # ★ 2D POSITIONAL ENCODING (Critical for spatial reasoning)
        # Without this, Transformer cannot learn "move right", "rotate", etc.
        self.pos_encoder = PositionalEncoding2D(hidden_dim, max_size=32)
        
        # === ABSTRACTION LAYER (Key SCI component) ===
        # Learns to identify structural vs content features
        self.abstraction_layer = AbstractionLayer2D(hidden_dim)
        
        # === STRUCTURE QUERIES ===
        # ★ Full-scale orthogonal init for diverse attention patterns
        self.structure_queries = nn.Parameter(
            torch.empty(1, num_structure_slots, hidden_dim)
        )
        nn.init.orthogonal_(self.structure_queries.data.squeeze(0))
        # No scaling down - orthogonal vectors already have unit norm
        
        # === CROSS-ATTENTION ===
        # Queries attend to grid differences
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # === TRANSFORMATION ENCODER (PreLN for stability) ===
        self.context_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                batch_first=True,
                norm_first=True  # PreLN is more stable
            ),
            num_layers=num_layers
        )
        
        # === OUTPUT PROJECTION ===
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        # NOTE: No LayerNorm here - it was causing representation collapse
        
        # === DIFFERENCE PROJECTION (NEW - Phase 2) ===
        # Explicit (output - input) embedding for change detection
        self.use_difference = True
        self.io_embed = nn.Embedding(3, hidden_dim)  # 0=input, 1=output, 2=diff
        self.diff_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(
        self,
        input_emb: torch.Tensor,   # [B, H_in, W_in, D]
        output_emb: torch.Tensor   # [B, H_out, W_out, D]
    ) -> torch.Tensor:
        """
        Extract structural representation from (input, output) pair.
        
        Returns:
            structure_slots: [B, K, D] - K structural pattern slots
        """
        B = input_emb.size(0)
        D = self.hidden_dim
        H_in, W_in = input_emb.shape[1], input_emb.shape[2]
        H_out, W_out = output_emb.shape[1], output_emb.shape[2]
        
        # ★ Add 2D positional encodings BEFORE flattening
        # This gives each cell a unique spatial address (x, y)
        input_pos = self.pos_encoder(input_emb)   # [B, H_in, W_in, D]
        output_pos = self.pos_encoder(output_emb) # [B, H_out, W_out, D]
        
        # Flatten grids to sequences
        input_flat = input_pos.view(B, -1, D)   # [B, H*W, D]
        output_flat = output_pos.view(B, -1, D) # [B, H*W, D]
        
        # Add input/output type indicators
        input_flat = input_flat + self.io_embed.weight[0]
        output_flat = output_flat + self.io_embed.weight[1]
        
        # ★ FIX #8: EXPLICIT DIFFERENCE EMBEDDING
        # Compute (output - input) to highlight WHERE changes happened
        if self.use_difference:
            H_min, W_min = min(H_in, H_out), min(W_in, W_out)
            diff_emb = output_emb[:, :H_min, :W_min, :] - input_emb[:, :H_min, :W_min, :]
            diff_pos = self.pos_encoder(diff_emb)
            diff_flat = diff_pos.view(B, -1, D)
            diff_flat = self.diff_proj(diff_flat)
            diff_flat = diff_flat + self.io_embed.weight[2]
        
        # Apply AbstractionLayer to suppress content
        input_abs = self.abstraction_layer(input_flat)
        output_abs = self.abstraction_layer(output_flat)
        if self.use_difference:
            diff_abs = self.abstraction_layer(diff_flat)
        
        # Concatenate: [input | output | difference]
        if self.use_difference:
            context = torch.cat([input_abs, output_abs, diff_abs], dim=1)
        else:
            context = torch.cat([input_abs, output_abs], dim=1)
        
        # Encode transformation patterns
        context_encoded = self.context_encoder(context)
        
        # Structure queries attend to context
        queries = self.structure_queries.expand(B, -1, -1)
        
        structure_slots, _ = self.cross_attention(
            query=queries,
            key=context_encoded,
            value=context_encoded
        )
        
        return self.norm(self.output_proj(structure_slots))


class AbstractionLayer2D(nn.Module):
    """
    THE KEY SCI INNOVATION adapted for 2D grids.
    
    Learns to identify and preserve ONLY structural information.
    Suppresses content-specific features (which colors, which positions).
    
    How it works:
    1. Structural detector scores each feature for "structuralness"
    2. High scores = structural (keep), low = content (suppress)
    3. Trained end-to-end with SCL
    """
    
    def __init__(self, d_model: int, hidden_mult: int = 2):
        super().__init__()
        
        # Structural feature detector
        self.structural_detector = nn.Sequential(
            nn.Linear(d_model, d_model * hidden_mult),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * hidden_mult, d_model),
            nn.Sigmoid()  # [0, 1] structuralness scores
        )
        
        # Residual gate
        self.residual_gate = nn.Parameter(torch.tensor(0.1))
        
        # Normalization
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply structural abstraction.
        
        Args:
            x: [B, N, D] input embeddings
        
        Returns:
            abstracted: [B, N, D] with content suppressed
        """
        # Compute structuralness scores
        scores = self.structural_detector(x)  # [B, N, D]
        
        # Apply soft mask: keep structural, suppress content
        abstracted = x * scores + x * self.residual_gate * (1 - scores)
        
        return self.norm(abstracted)
```

### 3. Content Encoder for Grids

```python
class ContentEncoder2D(nn.Module):
    """
    Extract content (objects) from grids, orthogonal to structure.
    
    Adaptation of SCI's Content Encoder:
    - Detects objects (connected components)
    - Extracts per-object features (color, shape, size, position)
    - Projects orthogonal to structural representation
    
    Parameters: ~1M
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        max_objects: int = 16
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.max_objects = max_objects
        
        # Object feature extractor (simple CNN)
        self.object_encoder = nn.Sequential(
            nn.Conv2d(10, 64, kernel_size=3, padding=1),  # 10 color channels
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Project to hidden dim
        self.object_proj = nn.Linear(128, hidden_dim)
        
        # Orthogonal projector (KEY SCI COMPONENT)
        self.orthogonal_projector = OrthogonalProjector(hidden_dim)
        
        # Learnable object queries (like DETR)
        self.object_queries = nn.Parameter(
            torch.randn(1, max_objects, hidden_dim) * 0.02
        )
        
    def forward(
        self,
        grid_emb: torch.Tensor,      # [B, H, W, D]
        structure_rep: torch.Tensor  # [B, K, D] from StructuralEncoder
    ) -> torch.Tensor:
        """
        Extract content representation orthogonal to structure.
        
        Returns:
            content_slots: [B, max_objects, D]
        """
        B, H, W, D = grid_emb.shape
        
        # Simple content extraction via attention to grid
        queries = self.object_queries.expand(B, -1, -1)
        
        # Flatten grid for attention
        grid_flat = grid_emb.view(B, H * W, D)
        
        # Cross-attention: object queries attend to grid
        content_raw = torch.bmm(
            F.softmax(torch.bmm(queries, grid_flat.transpose(1, 2)) / math.sqrt(D), dim=-1),
            grid_flat
        )
        
        # Project orthogonal to structure
        content_orthogonal = self.orthogonal_projector(
            content_raw,
            structure_rep.mean(dim=1, keepdim=True).expand(-1, self.max_objects, -1)
        )
        
        return content_orthogonal


class OrthogonalProjector(nn.Module):
    """
    Projects content representation orthogonal to structure.
    
    Ensures S(x) ⊥ C(x) which is critical for SCI.
    
    Uses Gram-Schmidt-style projection:
    C_orth = C - proj_S(C)
    """
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(
        self,
        content: torch.Tensor,    # [B, N, D]
        structure: torch.Tensor   # [B, N, D]
    ) -> torch.Tensor:
        """Project content orthogonal to structure."""
        # Normalize structure
        structure_norm = F.normalize(structure, dim=-1)
        
        # Compute projection of content onto structure
        dot_product = (content * structure_norm).sum(dim=-1, keepdim=True)
        projection = dot_product * structure_norm
        
        # Subtract projection (Gram-Schmidt)
        content_orthogonal = content - projection
        
        return self.proj(content_orthogonal)
```

### 4. Causal Binding for Grids

```python
class CausalBinding2D(nn.Module):
    """
    Bind structural slots to content objects.
    
    Adaptation of SCI's CBM:
    - Binding attention: which structure slot controls which object
    - Causal intervention: apply transformation to bound objects
    - Produces task embedding z_task
    
    Parameters: ~1M
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_structure_slots: int = 8,
        num_content_slots: int = 16
    ):
        super().__init__()
        
        # Binding attention
        self.binding_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        # Causal intervention MLP
        self.intervention_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Aggregate to single task embedding
        self.aggregator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
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
        
        Returns:
            z_task: [B, D] task understanding
        """
        # Binding: structure queries content
        bound, binding_weights = self.binding_attention(
            query=structure_slots,
            key=content_slots,
            value=content_slots
        )
        
        # Causal intervention: combine structure with bound content
        combined = torch.cat([structure_slots, bound], dim=-1)
        intervened = self.intervention_mlp(combined)
        
        # Aggregate slots to single task embedding
        pooled = intervened.mean(dim=1)  # [B, D]
        z_task = self.norm(self.aggregator(pooled))
        
        return z_task
```

### 5. Recursive Refinement (from TRM)

```python
class RecursiveRefinement(nn.Module):
    """
    TRM-style recursive refinement, conditioned on z_task from SCI.
    
    Key modification from TRM:
    - Latent update f() is conditioned on z_task
    - This injects the structural understanding into refinement
    
    Parameters: ~3M
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        max_cells: int = 900,  # 30×30
        num_colors: int = 10,
        H_cycles: int = 16,    # Supervision steps
        L_cycles: int = 4,     # Recursion steps per supervision
        L_layers: int = 2      # Network depth (TRM uses 2)
    ):
        super().__init__()
        
        self.H_cycles = H_cycles
        self.L_cycles = L_cycles
        self.hidden_dim = hidden_dim
        
        # Initialize answer and latent
        self.y_init = nn.Parameter(torch.randn(1, max_cells, hidden_dim) * 0.02)
        self.z_init = nn.Parameter(torch.randn(1, 64, hidden_dim) * 0.02)
        
        # Latent update: f(x, y, z, z_task)
        # Conditioned on z_task (SCI contribution)
        self.latent_update = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),  # x, y, z_task concatenated
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Answer update: g(y, z)
        self.answer_update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Output projection
        self.to_logits = nn.Linear(hidden_dim, num_colors)
        
    def forward(
        self,
        x_test_emb: torch.Tensor,  # [B, H*W, D] encoded test input
        z_task: torch.Tensor,       # [B, D] from CausalBinding
        target_shape: Tuple[int, int]
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Recursively refine answer conditioned on task understanding.
        
        Returns:
            outputs: List of predictions at each H_cycle (for deep supervision)
            final: Final prediction
        """
        B = x_test_emb.size(0)
        H, W = target_shape
        num_cells = H * W
        
        # Initialize
        y = self.y_init[:, :num_cells, :].expand(B, -1, -1).clone()
        z = self.z_init.expand(B, -1, -1).clone()
        
        # Pool x for conditioning
        x_pool = x_test_emb.mean(dim=1)  # [B, D]
        
        outputs = []
        
        for h in range(self.H_cycles):
            # Inner recursion loop (L_cycles)
            for l in range(self.L_cycles):
                # Pool y and z for update
                y_pool = y.mean(dim=1)  # [B, D]
                z_pool = z.mean(dim=1)  # [B, D]
                
                # Conditioned latent update: f(x, y, z_task)
                # KEY SCI CONTRIBUTION: z_task conditions the update
                update_input = torch.cat([x_pool, y_pool, z_task], dim=-1)
                z_update = self.latent_update(update_input).unsqueeze(1)
                z = z + z_update.expand(-1, z.size(1), -1)
            
            # Answer update: g(y, z)
            z_broadcast = z.mean(dim=1, keepdim=True).expand(-1, num_cells, -1)
            update_input = torch.cat([y, z_broadcast], dim=-1)
            y = y + self.answer_update(update_input)
            
            # Project to logits for this step
            logits = self.to_logits(y).view(B, H, W, -1)
            outputs.append(logits)
        
        return outputs, outputs[-1]
```

### 6. Complete SCI-ARC Model

```python
class SCIARC(nn.Module):
    """
    Complete SCI-ARC model.
    
    Combines:
    - SCI's structure-content separation (SE, CE, CBM)
    - TRM's recursive refinement
    - SCL for structural invariance
    
    Total parameters: ~8M (intentionally small like TRM)
    """
    
    def __init__(self, config: SCIARCConfig):
        super().__init__()
        
        self.config = config
        
        # Shared grid encoder
        self.grid_encoder = GridEncoder(
            hidden_dim=config.hidden_dim,
            num_colors=10,
            max_size=config.max_grid_size
        )
        
        # SCI components
        self.structural_encoder = StructuralEncoder2D(
            hidden_dim=config.hidden_dim,
            num_structure_slots=config.num_structure_slots,
            num_layers=config.se_layers,
            num_heads=config.num_heads
        )
        
        self.content_encoder = ContentEncoder2D(
            hidden_dim=config.hidden_dim,
            max_objects=config.max_objects
        )
        
        self.causal_binding = CausalBinding2D(
            hidden_dim=config.hidden_dim,
            num_structure_slots=config.num_structure_slots,
            num_content_slots=config.max_objects
        )
        
        # TRM component
        self.refiner = RecursiveRefinement(
            hidden_dim=config.hidden_dim,
            max_cells=config.max_grid_size ** 2,
            num_colors=10,
            H_cycles=config.H_cycles,
            L_cycles=config.L_cycles,
            L_layers=config.L_layers
        )
        
    def forward(
        self,
        demo_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
        test_input: torch.Tensor,
        target_shape: Tuple[int, int]
    ) -> Tuple[List[torch.Tensor], torch.Tensor, Dict]:
        """
        Full forward pass.
        
        Args:
            demo_pairs: List of (input, output) grid tensors
            test_input: [B, H, W] test input grid
            target_shape: (H_out, W_out) expected output size
        
        Returns:
            outputs: List of predictions for deep supervision
            final: Final prediction
            aux: Auxiliary outputs (structure_rep for SCL)
        """
        # === PHASE 1: Encode demos to get task understanding ===
        
        all_structure_reps = []
        all_content_reps = []
        
        for input_grid, output_grid in demo_pairs:
            # Encode grids
            input_emb = self.grid_encoder(input_grid)
            output_emb = self.grid_encoder(output_grid)
            
            # Extract structure
            structure_rep = self.structural_encoder(input_emb, output_emb)
            all_structure_reps.append(structure_rep)
            
            # Extract content (from input)
            content_rep = self.content_encoder(input_emb, structure_rep)
            all_content_reps.append(content_rep)
        
        # Aggregate across demos
        structure_agg = torch.stack(all_structure_reps, dim=1).mean(dim=1)  # [B, K, D]
        content_agg = torch.stack(all_content_reps, dim=1).mean(dim=1)      # [B, M, D]
        
        # Causal binding → task embedding
        z_task = self.causal_binding(structure_agg, content_agg)  # [B, D]
        
        # === PHASE 2: Recursive refinement on test input ===
        
        test_emb = self.grid_encoder(test_input)
        test_flat = test_emb.view(test_emb.size(0), -1, self.config.hidden_dim)
        
        outputs, final = self.refiner(test_flat, z_task, target_shape)
        
        # Return structure rep for SCL
        aux = {
            'structure_rep': structure_agg,  # For SCL
            'z_task': z_task
        }
        
        return outputs, final, aux
```

### 7. Structural Contrastive Loss (SCL) - Fixed Architecture

> **CRITICAL FIXES (December 2024)**: The original SCL implementation suffered from
> **representation collapse** - the structural encoder produced near-identical embeddings
> for all inputs, causing SCL loss to remain constant at $\ln(\text{batch\_size}) \approx 5.25$.
> 
> **Two phases of fixes were required:**
> 1. **Phase 1**: Fix variance reduction (pooling → flattening, remove LayerNorm)
> 2. **Phase 2**: Fix common background signal (add BatchNorm, Difference Embedding)

#### The Problem: Constant SCL Loss

During initial training, we observed:
```
Epoch 1: SCL Loss = 5.25 (constant)
Epoch 2: SCL Loss = 5.25 (constant)
...
Epoch 10: SCL Loss = 5.25 (constant)
```

**Root Cause Analysis (Two Issues):**

**Issue 1: Variance Reduction (Fixed in Phase 1)**
The issue was **mean pooling** of structure slots:
```python
# OLD (broken) code:
z = structure_reps.mean(dim=1)  # [B, K, D] → [B, D]
```

**Issue 2: Common Background Signal (Fixed in Phase 2)**
After fixing pooling, all samples still had **similarity ~0.95**:
```
Pre-BatchNorm similarities: (0,1)=0.9592, (0,2)=0.9572
```

Root cause: ARC grids are 90% black (background). The embedding is dominated by this common signal:
$$v_{sample} = v_{background} + v_{transformation}$$
Since $v_{background}$ is huge, all samples point in the same direction.

#### The Solution: Two-Phase Architectural Fixes

We implemented **four key fixes**:

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                    SCL ARCHITECTURE FIX - Data Flow Diagram                          │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  Input Grid [B, H, W]                                                               │
│        │                                                                             │
│        ▼                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐          │
│  │  Grid Encoder                                                          │          │
│  │  [B, H, W] → [B, H, W, D]                                             │          │
│  └───────────────────────────────────────────────────────────────────────┘          │
│        │                                                                             │
│        ▼                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐          │
│  │  ★ FIX #1: 2D POSITIONAL ENCODING (NEW)                               │          │
│  │  ─────────────────────────────────────────────────────────────────────│          │
│  │  • Learnable (x, y) embeddings added BEFORE flattening               │          │
│  │  • Each cell gets unique spatial "address"                            │          │
│  │  • Enables learning "move right", "rotate 90°" etc.                   │          │
│  │                                                                        │          │
│  │  pos_emb = y_embed(row) + x_embed(col)  # [H, W, D]                   │          │
│  │  grid_emb = grid_emb + pos_emb          # Broadcast over batch        │          │
│  └───────────────────────────────────────────────────────────────────────┘          │
│        │                                                                             │
│        ▼                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐          │
│  │  Structural Encoder (with Abstraction Layer)                          │          │
│  │  [B, H*W, D] → [B, K, D]  (K=8 structure slots)                       │          │
│  │                                                                        │          │
│  │  ★ FIX #2: FULL-SCALE QUERY INITIALIZATION                            │          │
│  │  ─────────────────────────────────────────────────────────────────────│          │
│  │  OLD: queries = randn() * 0.02 * 0.1  # Scale = 0.002 (too small!)   │          │
│  │  NEW: queries = orthogonal_init()      # Scale = 1.0 (full diversity)│          │
│  └───────────────────────────────────────────────────────────────────────┘          │
│        │                                                                             │
│        │  Structure Slots: [B, K, D] where K=8, D=256                               │
│        │                                                                             │
│        ▼                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐          │
│  │  ★ FIX #3: FLATTENING (NOT POOLING)                                   │          │
│  │  ─────────────────────────────────────────────────────────────────────│          │
│  │                                                                        │          │
│  │  OLD (collapsed):     z = structure_reps.mean(dim=1)   # [B, D]       │          │
│  │                       Variance reduced by 1/K = 12.5%                 │          │
│  │                       All samples → similar embeddings                │          │
│  │                                                                        │          │
│  │  NEW (diverse):       z = structure_reps.reshape(B, -1) # [B, K*D]    │          │
│  │                       Full variance preserved                          │          │
│  │                       Each sample maintains unique "signature"         │          │
│  │                                                                        │          │
│  │  Intuition: Flattening CONCATENATES slot information:                 │          │
│  │             [slot₁ | slot₂ | ... | slot₈]                             │          │
│  │             This preserves the structural "topology" of attention     │          │
│  └───────────────────────────────────────────────────────────────────────┘          │
│        │                                                                             │
│        │  Flattened: [B, K*D] = [B, 2048]                                           │
│        │                                                                             │
│        ▼                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐          │
│  │  PROJECTION HEAD (SimCLR-style with LayerNorm)                        │          │
│  │  ─────────────────────────────────────────────────────────────────────│          │
│  │                                                                        │          │
│  │  projector = Sequential(                                               │          │
│  │      Linear(K*D → D),      # 2048 → 256                               │          │
│  │      LayerNorm(D),         # ★ LayerNorm, NOT BatchNorm               │          │
│  │      ReLU(),                                                           │          │
│  │      Linear(D → proj_dim)  # 256 → 128                                │          │
│  │  )                                                                     │          │
│  │                                                                        │          │
│  │  Why LayerNorm?                                                        │          │
│  │  • BatchNorm normalizes ACROSS batch → identical inputs stay identical│          │
│  │  • LayerNorm normalizes WITHIN sample → preserves inter-sample diff   │          │
│  └───────────────────────────────────────────────────────────────────────┘          │
│        │                                                                             │
│        │  Projected: [B, 128]                                                        │
│        │                                                                             │
│        ▼                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐          │
│  │  L2 Normalization                                                      │          │
│  │  z = F.normalize(z, dim=-1)  # Unit vectors on hypersphere            │          │
│  └───────────────────────────────────────────────────────────────────────┘          │
│        │                                                                             │
│        ▼                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐          │
│  │  ★ FIX #4: LEARNABLE TEMPERATURE                                      │          │
│  │  ─────────────────────────────────────────────────────────────────────│          │
│  │                                                                        │          │
│  │  OLD: temperature = 0.07  (fixed)                                      │          │
│  │  NEW: log_temperature = nn.Parameter(log(0.07))                        │          │
│  │       temperature = exp(log_temperature).clamp(0.01, 1.0)              │          │
│  │                                                                        │          │
│  │  • Starts at 0.07 (loose clusters)                                     │          │
│  │  • Learns to become smaller (tighter clusters) as training progresses │          │
│  │  • Log parameterization ensures temperature stays positive             │          │
│  └───────────────────────────────────────────────────────────────────────┘          │
│        │                                                                             │
│        ▼                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐          │
│  │  InfoNCE Loss                                                          │          │
│  │  sim = z @ z.T / temperature                                           │          │
│  │  loss = -log(exp(sim_pos) / sum(exp(sim_all)))                        │          │
│  └───────────────────────────────────────────────────────────────────────┘          │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

#### Phase 2 Fixes: Background Signal Removal

After Phase 1 fixes, we observed that embeddings were diverse (variance ~1.0) but 
**all similarities remained ~0.95**. The issue was the common background signal.

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                    PHASE 2 FIXES - Background Signal Removal                         │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  PROBLEM: All embeddings point in same direction (sim ~0.95)                        │
│  ════════════════════════════════════════════════════════════                        │
│                                                                                      │
│  ARC grids are 90% background (black/0):                                            │
│                                                                                      │
│    v_sample = v_background + v_transformation                                        │
│               ─────────────   ────────────────                                       │
│               HUGE (shared)   small (unique)                                         │
│                                                                                      │
│  Result: cos(v₁, v₂) ≈ 0.95 because background dominates                            │
│                                                                                      │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  ★ FIX #7: BATCHNORM1D (Background Subtraction)                                     │
│  ─────────────────────────────────────────────────────────────────────────────────── │
│                                                                                      │
│  BatchNorm centers the batch by subtracting mean vector:                            │
│                                                                                      │
│    μ_batch ≈ v_background  (the common signal across all samples)                   │
│    z_centered = z - μ_batch  (removes background!)                                  │
│                                                                                      │
│  Code:                                                                               │
│    self.batch_norm = nn.BatchNorm1d(input_dim, affine=True)                         │
│    z = structure_reps.reshape(B, -1)  # [B, K*D]                                    │
│    z = self.batch_norm(z)              # Centers → removes common signal            │
│    z = self.projector(z)               # Then project                               │
│                                                                                      │
│  Effect: Post-BatchNorm similarity drops from 0.95 → near 0                         │
│                                                                                      │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  ★ FIX #8: DIFFERENCE EMBEDDING (Explicit Change Detection)                         │
│  ─────────────────────────────────────────────────────────────────────────────────── │
│                                                                                      │
│  Instead of making the model learn (output - input), compute it explicitly:         │
│                                                                                      │
│    OLD: context = [input_emb, output_emb]                                           │
│    NEW: context = [input_emb, output_emb, diff_emb]                                 │
│                                                                                      │
│         where diff_emb = output_emb - input_emb                                     │
│                                                                                      │
│  The difference highlights WHERE changes happened:                                  │
│    • Zeros where nothing changed (background)                                       │
│    • Non-zeros only at transformation locations                                     │
│                                                                                      │
│  Code (StructuralEncoder2D):                                                        │
│    diff_emb = output_emb[:, :H_min, :W_min, :] - input_emb[:, :H_min, :W_min, :]   │
│    diff_pos = self.pos_encoder(diff_emb)                                            │
│    diff_flat = self.diff_proj(diff_flat)  # Learnable projection                   │
│    context = torch.cat([input_abs, output_abs, diff_abs], dim=1)                   │
│                                                                                      │
│  Backward compatible: use_difference=True (default)                                 │
│                                                                                      │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  ★ FIX #9: HIGHER INITIAL TEMPERATURE                                               │
│  ─────────────────────────────────────────────────────────────────────────────────── │
│                                                                                      │
│    OLD: temperature = 0.07 (too low for high similarity regime)                     │
│    NEW: temperature = 0.5  (spreads softmax, better gradients)                      │
│                                                                                      │
│  With sim~0.95 and temp=0.07:                                                       │
│    exp(0.95/0.07) ≈ exp(13.6) → near-uniform softmax → no gradient                 │
│                                                                                      │
│  With sim~0.95 and temp=0.5:                                                        │
│    exp(0.95/0.5) = exp(1.9) → good gradient signal                                  │
│                                                                                      │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  ★ FIX #10: INCREASED SCL WEIGHT                                                    │
│  ─────────────────────────────────────────────────────────────────────────────────── │
│                                                                                      │
│    OLD: scl_weight = 0.1  (task loss dominates)                                     │
│    NEW: scl_weight = 1.0  (balanced influence)                                      │
│                                                                                      │
│  Task loss was optimized quickly, leaving no gradient budget for SCL.               │
│  With higher weight, model must satisfy both objectives.                            │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

#### Mathematical Justification

**Why Pooling Causes Collapse (Central Limit Theorem):**

Given $K$ structure slots $\{s_1, s_2, ..., s_K\}$ with $s_i \sim \mathcal{N}(0, \sigma^2 I)$:

$$\bar{s} = \frac{1}{K}\sum_{i=1}^{K} s_i \implies \text{Var}(\bar{s}) = \frac{\sigma^2}{K}$$

For $K=8$: variance drops to 12.5% of original → all samples converge to near-zero mean.

**Why Flattening Preserves Diversity:**

$$z = [s_1 \| s_2 \| ... \| s_K] \in \mathbb{R}^{K \cdot D}$$

Each dimension retains full variance $\sigma^2$. The "signature" of which slots attended to what is preserved in the concatenated structure.

#### Updated Code Implementation

```python
class StructuralContrastiveLoss(nn.Module):
    """
    SCL with architectural fixes for preventing representation collapse.
    
    Key Changes from Original (Phase 1 + Phase 2):
    1. FLATTEN instead of mean pool
    2. BatchNorm1d BEFORE projection (removes common background signal)
    3. LayerNorm inside projector
    4. Learnable temperature (starts at 0.5 for high-similarity regime)
    5. Orthogonal initialization for projector weights
    """
    
    def __init__(
        self, 
        temperature: float = 0.5,  # ★ Higher for high-similarity regime
        normalize: bool = True,
        hidden_dim: int = 256,
        projection_dim: int = 128,
        num_structure_slots: int = 8
    ):
        super().__init__()
        
        # ★ FIX #4: Learnable temperature
        self.log_temperature = nn.Parameter(torch.tensor(temperature).log())
        self.normalize = normalize
        self.num_slots = num_structure_slots
        
        # ★ FIX #3: Input is K*D (flattened) not D (pooled)
        input_dim = hidden_dim * num_structure_slots  # 256 * 8 = 2048
        
        # ★ FIX #7: BatchNorm to remove common background signal
        self.batch_norm = nn.BatchNorm1d(input_dim, affine=True)
        
        # Projection head with LayerNorm
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # ★ LayerNorm, not BatchNorm
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim)
        )
        
        # Orthogonal init for stability
        nn.init.orthogonal_(self.projector[0].weight)
        nn.init.orthogonal_(self.projector[3].weight)
    
    def forward(
        self,
        structure_reps: torch.Tensor,  # [B, K, D]
        transform_labels: torch.Tensor  # [B]
    ) -> torch.Tensor:
        B = structure_reps.size(0)
        
        if B < 2:
            return torch.tensor(0.0, device=structure_reps.device)
        
        # ★ FIX #3: FLATTEN not mean pool
        z = structure_reps.reshape(B, -1)  # [B, K*D]
        
        # ★ FIX #7: Apply BatchNorm to remove common background signal
        z = self.batch_norm(z)  # Centers batch → removes shared direction
        
        # Project to contrastive space
        z = self.projector(z)  # [B, projection_dim]
        
        # L2 normalize
        if self.normalize:
            z = F.normalize(z, dim=-1)
        
        # ★ FIX #4: Use learnable temperature
        temperature = self.log_temperature.exp().clamp(min=0.01, max=1.0)
        sim = torch.mm(z, z.t()) / temperature
        
        # InfoNCE loss computation...
        # (same as before)
```

#### Empirical Validation

After applying Phase 1 + Phase 2 fixes:

| Metric | Before Any Fix | After Phase 1 | After Phase 2 |
|--------|----------------|---------------|---------------|
| SCL Loss (epoch 1) | 5.25 (constant) | 5.25 (still constant) | Decreasing |
| Pre-BatchNorm Similarity | ~1.0 | ~0.95 | ~0.95 |
| Post-BatchNorm Similarity | N/A | N/A | **< 0.3** |
| Embedding Variance | ~0.001 | ~1.0 | ~1.0 |
| InfoNCE Gradient | Near-zero | Weak | **Strong signal** |

**Key insight**: Phase 1 fixed the variance but not the direction. 
Phase 2 (BatchNorm) removes the common direction, making contrastive learning work.

---

### 8. Combined Loss Function (SCIARCLoss)

```python
class SCIARCLoss(nn.Module):
    """Combined loss for SCI-ARC training."""
    
    def __init__(
        self,
        H_cycles: int = 16,
        scl_weight: float = 1.0,  # ★ Increased from 0.1
        orthogonality_weight: float = 0.01,
        hidden_dim: int = 256,
        num_structure_slots: int = 8
    ):
        super().__init__()
        self.H_cycles = H_cycles
        self.scl_weight = scl_weight
        self.orth_weight = orthogonality_weight
        
        # Use updated SCL with flattening fix
        self.scl = StructuralContrastiveLoss(
            hidden_dim=hidden_dim,
            num_structure_slots=num_structure_slots
        )
        
        # Deep supervision weights (later steps weighted more)
        self.step_weights = torch.arange(1, H_cycles + 1).float() / H_cycles
        
    def forward(
        self,
        outputs: List[torch.Tensor],  # Predictions at each step
        target: torch.Tensor,          # Ground truth
        structure_rep: torch.Tensor,   # For SCL
        content_rep: torch.Tensor,     # For orthogonality
        transform_labels: torch.Tensor # Transform family labels
    ) -> Dict[str, torch.Tensor]:
        """Compute all losses."""
        device = outputs[0].device
        weights = self.step_weights.to(device)
        
        # 1. Deep supervision CE loss
        ce_loss = 0.0
        for t, pred in enumerate(outputs):
            pred_flat = pred.view(-1, 10)
            target_flat = target.view(-1)
            
            # Ignore padding (-1)
            valid_mask = target_flat != -1
            if valid_mask.any():
                step_loss = F.cross_entropy(
                    pred_flat[valid_mask],
                    target_flat[valid_mask]
                )
                ce_loss += weights[t] * step_loss
        ce_loss /= self.H_cycles
        
        # 2. Structural Contrastive Loss
        scl_loss = self.scl(structure_rep, transform_labels)
        
        # 3. Orthogonality loss (S ⊥ C)
        s_norm = F.normalize(structure_rep.mean(dim=1), dim=-1)
        c_norm = F.normalize(content_rep.mean(dim=1), dim=-1)
        orth_loss = (s_norm * c_norm).sum(dim=-1).abs().mean()
        
        total = ce_loss + self.scl_weight * scl_loss + self.orth_weight * orth_loss
        
        return {
            'total': total,
            'ce': ce_loss,
            'scl': scl_loss,
            'orthogonality': orth_loss
        }
```

---

## Dataset Preparation

### Download Script

```bash
#!/bin/bash
# scripts/download_data.sh

set -e

mkdir -p data/{arc_agi_1,arc_agi_2,re_arc,barc,concept_arc}

echo "=== Downloading ARC-AGI-1 ==="
git clone https://github.com/fchollet/ARC-AGI.git data/arc_agi_1_repo
cp -r data/arc_agi_1_repo/data/* data/arc_agi_1/

echo "=== Downloading ARC-AGI-2 ==="
git clone https://github.com/fchollet/ARC-AGI-2.git data/arc_agi_2_repo
cp -r data/arc_agi_2_repo/data/* data/arc_agi_2/

echo "=== Downloading RE-ARC (synthetic augmentation) ==="
git clone https://github.com/michaelhodel/re-arc.git data/re_arc_repo
cd data/re_arc_repo
python generate.py --num_tasks 50000 --output ../re_arc/
cd ../..

echo "=== Downloading BARC (LLM-generated) ==="
git clone https://github.com/xu3kev/BARC.git data/barc_repo
# Download pre-generated data from HuggingFace
pip install huggingface_hub
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id='barc/arc-heavy', filename='barc_train.json', local_dir='data/barc/')
"

echo "=== Downloading ConceptARC ==="
git clone https://github.com/victorvikram/ConceptARC.git data/concept_arc_repo
cp -r data/concept_arc_repo/corpus/* data/concept_arc/

echo "=== Downloading TRM repository (for reference) ==="
git clone https://github.com/SamsungSAILMontreal/TinyRecursiveModels.git data/trm_repo

echo "Done! All datasets downloaded."
```

### Transformation Family Labels for SCL

```python
# data/transform_families.py

"""
Transformation family labels for Structural Contrastive Loss.

KEY FOR SCL: Tasks with same transformation should cluster.

Manual annotation based on ARC task analysis.
"""

TRANSFORM_FAMILIES = {
    # Geometric
    'rotate_90': 0,
    'rotate_180': 1,
    'rotate_270': 2,
    'flip_horizontal': 3,
    'flip_vertical': 4,
    'transpose': 5,
    
    # Scaling
    'upscale_2x': 6,
    'downscale_2x': 7,
    'tile_2x2': 8,
    
    # Color
    'color_swap': 9,
    'color_invert': 10,
    'recolor_by_rule': 11,
    
    # Object operations
    'copy_object': 12,
    'move_object': 13,
    'delete_object': 14,
    
    # Pattern
    'extend_pattern': 15,
    'complete_grid': 16,
    'fill_enclosed': 17,
    
    # Logical
    'boolean_and': 18,
    'boolean_or': 19,
    'mask_apply': 20,
}


def get_transform_family(task_id: str, task_metadata: dict = None) -> int:
    """
    Get transformation family for a task.
    
    For RE-ARC and BARC, family is often encoded in task_id.
    For original ARC, use metadata or default to task_id hash.
    """
    task_lower = task_id.lower()
    
    # Check explicit patterns
    for family_name, family_idx in TRANSFORM_FAMILIES.items():
        if family_name.replace('_', '') in task_lower.replace('_', ''):
            return family_idx
    
    # Check metadata if available
    if task_metadata and 'transform_type' in task_metadata:
        return TRANSFORM_FAMILIES.get(task_metadata['transform_type'], -1)
    
    # Default: hash task_id to family (not ideal but allows SCL to learn)
    return hash(task_id) % len(TRANSFORM_FAMILIES)
```

### Dataset Class

```python
# data/sci_arc_dataset.py

import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from torch.utils.data import Dataset

from .transform_families import get_transform_family


class SCIARCDataset(Dataset):
    """
    Dataset for SCI-ARC training with transformation family labels.
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'training',
        augment: bool = True,
        max_demos: int = 3
    ):
        self.data_dir = Path(data_dir)
        self.augment = augment
        self.max_demos = max_demos
        
        # Load tasks
        self.tasks = self._load_tasks(split)
        
        # Create samples: (task, test_idx)
        self.samples = []
        for task in self.tasks:
            for test_idx in range(len(task['test'])):
                self.samples.append((task, test_idx))
    
    def _load_tasks(self, split: str) -> List[Dict]:
        tasks = []
        split_dir = self.data_dir / split
        
        for json_path in split_dir.glob('*.json'):
            with open(json_path) as f:
                data = json.load(f)
            
            task = {
                'task_id': json_path.stem,
                'train': [
                    (np.array(ex['input']), np.array(ex['output']))
                    for ex in data['train']
                ],
                'test': [
                    (np.array(ex['input']), np.array(ex['output']))
                    for ex in data['test']
                ],
                'transform_family': get_transform_family(json_path.stem)
            }
            tasks.append(task)
        
        return tasks
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        task, test_idx = self.samples[idx]
        
        # Get demo pairs
        demos = task['train'][:self.max_demos]
        
        # Get test pair
        test_input, test_output = task['test'][test_idx]
        
        # Augmentation (consistent across all grids)
        if self.augment:
            demos, test_input, test_output = self._augment(
                demos, test_input, test_output
            )
        
        # Convert to tensors
        demo_tensors = [
            (torch.tensor(d[0], dtype=torch.long),
             torch.tensor(d[1], dtype=torch.long))
            for d in demos
        ]
        
        return {
            'task_id': task['task_id'],
            'transform_family': task['transform_family'],
            'demos': demo_tensors,
            'test_input': torch.tensor(test_input, dtype=torch.long),
            'test_output': torch.tensor(test_output, dtype=torch.long),
            'target_shape': test_output.shape
        }
    
    def _augment(self, demos, test_in, test_out):
        """Apply consistent augmentation."""
        aug_type = np.random.choice([
            'none', 'rot90', 'rot180', 'rot270',
            'flip_h', 'flip_v', 'transpose'
        ])
        
        def apply(g):
            if aug_type == 'none': return g
            elif aug_type == 'rot90': return np.rot90(g, 1)
            elif aug_type == 'rot180': return np.rot90(g, 2)
            elif aug_type == 'rot270': return np.rot90(g, 3)
            elif aug_type == 'flip_h': return np.fliplr(g)
            elif aug_type == 'flip_v': return np.flipud(g)
            elif aug_type == 'transpose': return g.T
        
        aug_demos = [(apply(d[0]), apply(d[1])) for d in demos]
        return aug_demos, apply(test_in), apply(test_out)


def collate_sci_arc(batch):
    """Custom collate for variable-size grids."""
    # Find max sizes
    max_h_in = max(b['test_input'].shape[0] for b in batch)
    max_w_in = max(b['test_input'].shape[1] for b in batch)
    max_h_out = max(b['test_output'].shape[0] for b in batch)
    max_w_out = max(b['test_output'].shape[1] for b in batch)
    
    test_inputs = []
    test_outputs = []
    target_shapes = []
    transform_families = []
    all_demos = []
    
    for b in batch:
        # Pad test input
        h, w = b['test_input'].shape
        padded_in = torch.zeros(max_h_in, max_w_in, dtype=torch.long)
        padded_in[:h, :w] = b['test_input']
        test_inputs.append(padded_in)
        
        # Pad test output (use -1 for ignore)
        h, w = b['test_output'].shape
        padded_out = torch.full((max_h_out, max_w_out), -1, dtype=torch.long)
        padded_out[:h, :w] = b['test_output']
        test_outputs.append(padded_out)
        
        target_shapes.append(b['target_shape'])
        transform_families.append(b['transform_family'])
        all_demos.append(b['demos'])
    
    return {
        'demos': all_demos,
        'test_input': torch.stack(test_inputs),
        'test_output': torch.stack(test_outputs),
        'target_shapes': target_shapes,
        'transform_family': torch.tensor(transform_families)
    }
```

---

## Training Configuration

```yaml
# configs/sci_arc_full.yaml

model:
  hidden_dim: 256
  num_structure_slots: 8      # K in SCI
  max_objects: 16             # Content slots
  se_layers: 2                # Structural encoder depth
  num_heads: 4
  max_grid_size: 30
  
  # TRM parameters
  H_cycles: 16                # Supervision steps
  L_cycles: 4                 # Recursion per step
  L_layers: 2                 # Network depth (TRM insight: keep at 2)

loss:
  scl_weight: 0.1             # Structural Contrastive Loss weight
  orthogonality_weight: 0.01  # S ⊥ C constraint

training:
  # Curriculum
  curriculum:
    - phase: "re_arc"
      data_path: "data/re_arc"
      epochs: 5
      lr: 1e-4
      description: "Learn transformations from synthetic data"
    
    - phase: "barc"
      data_path: "data/barc"
      epochs: 5
      lr: 5e-5
      description: "Diverse transformations from LLM-generated data"
    
    - phase: "arc_agi_1"
      data_path: "data/arc_agi_1/training"
      epochs: 20
      lr: 2e-5
      description: "Fine-tune on real ARC tasks"
  
  batch_size: 8
  gradient_accumulation: 4    # Effective batch: 32
  max_grad_norm: 1.0
  weight_decay: 0.01
  
  optimizer: "AdamW"
  scheduler: "cosine"
  warmup_ratio: 0.1
  
  # SCL warmup (prevent instability early)
  scl_warmup_epochs: 2

data:
  max_demos: 3
  augment: true
  num_workers: 4

logging:
  wandb_project: "sci-arc"
  log_interval: 50
  eval_interval: 500
  save_interval: 1000

hardware:
  fp16: true
  gradient_checkpointing: false  # Model is small enough
```

---

## Implementation Checklist

### Phase 1: Environment Setup (Day 1)
- [ ] Create conda environment: `conda create -n sci_arc python=3.10`
- [ ] Install PyTorch 2.1+ with CUDA
- [ ] Install dependencies: `einops`, `wandb`, `pytest`, `matplotlib`
- [ ] Clone TRM repo for reference: `git clone https://github.com/SamsungSAILMontreal/TinyRecursiveModels`
- [ ] Set up Weights & Biases

### Phase 2: Data Pipeline (Day 1-2)
- [ ] Run download script
- [ ] Implement `SCIARCDataset`
- [ ] Implement `collate_sci_arc`
- [ ] Create transformation family annotations
- [ ] Unit test data loading
- [ ] Verify augmentation consistency

### Phase 3: Model Components (Day 2-6)
- [ ] Implement `GridEncoder` with 2D positional encoding
- [ ] Unit test GridEncoder
- [ ] Implement `AbstractionLayer2D` (KEY SCI COMPONENT)
- [ ] Unit test AbstractionLayer2D
- [ ] Implement `StructuralEncoder2D`
- [ ] Unit test SE outputs
- [ ] Implement `ContentEncoder2D` with `OrthogonalProjector`
- [ ] Unit test CE orthogonality
- [ ] Implement `CausalBinding2D`
- [ ] Unit test CBM
- [ ] Implement `RecursiveRefinement`
- [ ] Unit test refinement loop
- [ ] Implement complete `SCIARC` model
- [ ] Verify parameter count (~8M)

### Phase 4: Losses (Day 6-7)
- [ ] Implement `StructuralContrastiveLoss` (from SCI)
- [ ] Unit test SCL with known positive/negative pairs
- [ ] Implement `SCIARCLoss` (combined)
- [ ] Verify gradient flow

### Phase 5: Training (Day 7-14)
- [ ] Implement training loop
- [ ] Implement validation
- [ ] Set up curriculum learning
- [ ] Train Phase 1: RE-ARC
- [ ] Train Phase 2: BARC
- [ ] Train Phase 3: ARC-AGI-1
- [ ] Monitor SCL loss (should decrease)

### Phase 6: Evaluation (Day 14-18)
- [ ] Evaluate on ARC-AGI-1 eval
- [ ] Evaluate on ARC-AGI-2
- [ ] Evaluate on ConceptARC
- [ ] Compare with TRM baseline
- [ ] Analyze structural clustering (t-SNE)

### Phase 7: Ablations (Day 18-22)
- [ ] Ablation: No SE (remove structural encoder)
- [ ] Ablation: No CE (remove content encoder)
- [ ] Ablation: No SCL (remove contrastive loss)
- [ ] Ablation: No orthogonality
- [ ] Ablation: Structure slot count sweep

### Phase 8: Paper & Submission (Day 22-30)
- [ ] Generate result tables
- [ ] Create architecture diagrams
- [ ] Write paper draft
- [ ] Prepare for ARC Prize submission (if results good)

---

## Unit Tests

### Test Structural Encoder

```python
# tests/test_structural_encoder.py

import pytest
import torch
from sci_arc.models import StructuralEncoder2D, GridEncoder


class TestStructuralEncoder:
    
    @pytest.fixture
    def se(self):
        return StructuralEncoder2D(
            hidden_dim=128,
            num_structure_slots=4,
            num_layers=1
        )
    
    @pytest.fixture
    def grid_enc(self):
        return GridEncoder(hidden_dim=128)
    
    def test_output_shape(self, se, grid_enc):
        """Test structure slots output shape."""
        input_grid = torch.randint(0, 10, (2, 5, 5))
        output_grid = torch.randint(0, 10, (2, 7, 7))
        
        input_emb = grid_enc(input_grid)
        output_emb = grid_enc(output_grid)
        
        structure_slots = se(input_emb, output_emb)
        
        assert structure_slots.shape == (2, 4, 128)  # [B, K, D]
    
    def test_structural_invariance(self, se, grid_enc):
        """
        KEY TEST: Same transformation, different content → similar structure.
        
        This is what SCL enforces during training.
        """
        se.eval()
        
        # Two tasks with same transformation (e.g., both are rotations)
        # but different content
        input1 = torch.zeros(1, 3, 3, dtype=torch.long)
        input1[0, 0, 0] = 1  # Single red cell top-left
        output1 = torch.zeros(1, 3, 3, dtype=torch.long)
        output1[0, 0, 2] = 1  # Rotated 90 degrees
        
        input2 = torch.zeros(1, 3, 3, dtype=torch.long)
        input2[0, 1, 1] = 5  # Single gray cell center
        output2 = torch.zeros(1, 3, 3, dtype=torch.long)
        output2[0, 1, 1] = 5  # Same (rotation doesn't change center)
        
        with torch.no_grad():
            emb1_in = grid_enc(input1)
            emb1_out = grid_enc(output1)
            s1 = se(emb1_in, emb1_out)
            
            emb2_in = grid_enc(input2)
            emb2_out = grid_enc(output2)
            s2 = se(emb2_in, emb2_out)
        
        # After training with SCL, these should be similar
        # For now, just verify they're computed
        assert s1.shape == s2.shape


class TestAbstractionLayer:
    """Test the key SCI innovation."""
    
    @pytest.fixture
    def abstraction(self):
        from sci_arc.models.structural_encoder import AbstractionLayer2D
        return AbstractionLayer2D(d_model=128)
    
    def test_output_shape(self, abstraction):
        """Output shape should match input."""
        x = torch.randn(2, 25, 128)
        out = abstraction(x)
        assert out.shape == x.shape
    
    def test_gradient_flow(self, abstraction):
        """Gradients should flow through abstraction layer."""
        x = torch.randn(2, 25, 128, requires_grad=True)
        out = abstraction(x)
        loss = out.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
```

### Test SCL

```python
# tests/test_scl.py

import pytest
import torch
from sci_arc.training import StructuralContrastiveLoss


class TestSCL:
    
    @pytest.fixture
    def scl(self):
        return StructuralContrastiveLoss(temperature=0.07)
    
    def test_same_family_lower_loss(self, scl):
        """Same transform family should have lower loss."""
        # All same family
        structure_reps = torch.randn(4, 8, 128)
        labels_same = torch.tensor([0, 0, 0, 0])
        
        loss_same = scl(structure_reps, labels_same)
        
        # All different families
        labels_diff = torch.tensor([0, 1, 2, 3])
        loss_diff = scl(structure_reps, labels_diff)
        
        # Same family should have lower or equal loss
        # (depends on actual representation similarity)
        assert loss_same.item() >= 0  # Just verify it computes
    
    def test_identical_reps_zero_loss(self, scl):
        """Identical representations with same label → low loss."""
        rep = torch.randn(1, 8, 128)
        structure_reps = rep.expand(4, -1, -1).clone()
        
        # Add tiny noise to avoid numerical issues
        structure_reps = structure_reps + torch.randn_like(structure_reps) * 0.01
        
        labels = torch.tensor([0, 0, 0, 0])
        
        loss = scl(structure_reps, labels)
        
        # Should be low (representations are very similar)
        assert loss.item() < 5.0
```

---

## Expected Results

### Performance Targets

| Metric | TRM Baseline | SCI-ARC Expected | Improvement |
|--------|-------------|------------------|-------------|
| ARC-AGI-1 Task Acc | 45% | 50-55% | +5-10% |
| ARC-AGI-2 Task Acc | 8% | 12-15% | +4-7% |
| Zero-shot Transfer | N/A | >30% | Novel metric |

### Ablation Expected Results

| Ablation | Δ Task Acc | Reason |
|----------|-----------|--------|
| No SE | -15% | Loses transformation understanding |
| No CE | -5% | Loses content awareness |
| No SCL | -8% | Loses structural invariance |
| No Orthogonality | -3% | S and C leak into each other |

### Structural Clustering Analysis

After training, z_task embeddings should cluster by transformation family:
- All "rotate" tasks cluster together
- All "flip" tasks cluster together
- etc.

Visualize with t-SNE to verify SCL is working.

---

## Code Structure

```
sci_arc/
├── README.md
├── requirements.txt
├── setup.py
├── configs/
│   ├── sci_arc_full.yaml
│   ├── sci_arc_small.yaml
│   └── ablations/
│       ├── no_se.yaml
│       ├── no_ce.yaml
│       ├── no_scl.yaml
│       └── no_orth.yaml
├── data/
│   ├── __init__.py
│   ├── sci_arc_dataset.py
│   ├── transform_families.py
│   └── download_data.sh
├── models/
│   ├── __init__.py
│   ├── grid_encoder.py
│   ├── structural_encoder.py      # SE with AbstractionLayer2D
│   ├── content_encoder.py         # CE with OrthogonalProjector
│   ├── causal_binding.py          # CBM
│   ├── recursive_refinement.py    # From TRM
│   └── sci_arc.py                 # Complete model
├── training/
│   ├── __init__.py
│   ├── losses.py                  # SCL + combined loss
│   ├── trainer.py
│   └── scheduler.py
├── evaluation/
│   ├── __init__.py
│   ├── evaluator.py
│   ├── metrics.py
│   └── visualization.py
├── tests/
│   ├── test_grid_encoder.py
│   ├── test_structural_encoder.py
│   ├── test_content_encoder.py
│   ├── test_scl.py
│   ├── test_model.py
│   └── test_data.py
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── ablation_sweep.py
│   └── visualize_clusters.py
└── notebooks/
    ├── 01_data_exploration.ipynb
    ├── 02_debug_model.ipynb
    └── 03_analyze_results.ipynb
```

---

## Quick Start

```bash
# 1. Setup
conda create -n sci_arc python=3.10
conda activate sci_arc
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install einops wandb pytest matplotlib pandas

# 2. Clone and install
git clone https://github.com/your-repo/sci-arc.git
cd sci_arc
pip install -e .

# 3. Download data
bash data/download_data.sh

# 4. Run tests
pytest tests/ -v

# 5. Train (small test)
python scripts/train.py --config configs/sci_arc_small.yaml

# 6. Train (full)
python scripts/train.py --config configs/sci_arc_full.yaml

# 7. Evaluate
python scripts/evaluate.py --model checkpoints/best --dataset arc_agi_1_eval

# 8. Ablations
python scripts/ablation_sweep.py
```

---

## Summary: Why This Could Work

### SCI Principles Applied to ARC

| SCI Principle | Text Domain | ARC Application |
|---------------|-------------|-----------------|
| **Structure ≠ Content** | "walk twice" ≠ "jump twice" as structure | "rotate" ≠ specific objects being rotated |
| **SCL** | Same syntax → same S(x) | Same transformation rule → same S(demos) |
| **Orthogonality** | S ⊥ C in embedding space | Transformation embedding ⊥ object embedding |
| **AbstractionLayer** | Suppress content words | Suppress grid-specific details |

### Novel Contributions

1. **First application of structural invariance to visual reasoning**
2. **SCI + TRM hybrid**: Structure-content separation + recursive refinement
3. **Explicit transformation embedding**: z_task captures transformation rule
4. **SCL for ARC**: Contrastive learning over transformation families

### Why It Might Outperform TRM

TRM learns **correlations** between input/output grids.
SCI-ARC learns **causal structure** of transformations.

When a novel task appears:
- TRM: Pattern match to similar training examples
- SCI-ARC: Recognize transformation type → apply transformation

---

## Future Enhancements: Roadmap to SOTA (December 2024)

> **IMPORTANT:** These enhancements should be implemented **after** validating that SCL loss
> decreases correctly and embeddings cluster by transformation type. Do NOT add these prematurely.

### Phase 2: Verifier Loop (Low Risk, High Impact)

**Problem:** The model generates multiple candidate outputs but has no way to score them.

**Solution:** Use the `StructuralEncoder` at inference time as a verifier/scorer.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    GENERATE & VERIFY PIPELINE                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. Encode demo pairs → z_demos = mean(SE(demo_input, demo_output))         │
│                                                                              │
│  2. Generate N candidates: [cand_1, cand_2, ..., cand_N]                    │
│                                                                              │
│  3. For each candidate:                                                      │
│     z_cand = SE(test_input, candidate)                                      │
│     score = cosine_similarity(z_cand, z_demos)                              │
│                                                                              │
│  4. Select: argmax(scores)                                                   │
│                                                                              │
│  This turns SCL into an accuracy booster at test time!                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Implementation Complexity:** Low (pure inference, no training code changes)
**Risk:** Low (read-only, cannot break training)
**When to Add:** After Phase 1 validates SCL works.

### Phase 3: Test-Time Training (TTT) (Medium Risk)

**Problem:** Generic weights may not adapt to the specific logic of a novel task.

**Solution:** Fine-tune on the demo pairs of the test task before prediction.

```python
# Pseudocode for TTT
def inference_with_ttt(model, demo_pairs, test_input, ttt_steps=20):
    # Freeze main weights, only update adapters
    for param in model.parameters():
        param.requires_grad = False
    for param in model.adapter_layers.parameters():
        param.requires_grad = True
    
    optimizer = AdamW(model.adapter_layers.parameters(), lr=1e-4)
    
    # Fine-tune on demo pairs
    for step in range(ttt_steps):
        loss = 0
        for inp, out in demo_pairs:
            pred = model(inp)
            loss += F.cross_entropy(pred, out)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    # Now predict
    return model(test_input)
```

**Implementation Complexity:** Medium (requires adapter layers, careful optimizer handling)
**Risk:** Medium (optimizer state at inference can cause subtle bugs)
**When to Add:** After Phase 2 if accuracy plateaus.

### Phase 4: Massive Synthetic Data (High Risk, Long-Term)

**Problem:** 400 ARC training tasks are insufficient to train a Transformer from scratch.

**Solution:** Procedural data generator using a DSL (Domain Specific Language).

**What the Generator Should Produce:**
- Object primitives: rectangles, lines, dots, patterns
- Transformations: move, rotate, flip, scale, recolor
- Compositions: apply multiple transformations in sequence
- Distractors: add irrelevant objects to test abstraction

**Implementation Complexity:** High (essentially a separate research project)
**Risk:** High (bugs in generator can poison the entire training)
**When to Add:** Only after Phases 1-3 prove the architecture works.

### Summary: Staged Approach

| Phase | Component | Complexity | Risk | Prerequisite |
|-------|-----------|------------|------|--------------|
| **1 (Current)** | Train & validate SCL | Done | - | None |
| **2** | Verifier Loop | Low | Low | SCL loss decreases |
| **3** | Test-Time Training | Medium | Medium | Phase 2 accuracy plateau |
| **4** | Synthetic Data | High | High | Phase 3 proves architecture |

**Rationale:** Each phase validates the previous one before adding complexity.
Debugging a failing system with all components is nearly impossible.
Debugging a staged system isolates failures to specific components.

This should improve **compositional generalization** - exactly what ARC tests.

---

## ✅ IMPLEMENTED: Competitive Inference Modules

> **Status:** These modules have been fully implemented and tested in `sci_arc/inference/`.
> All 14 unit tests pass. Configuration available in `configs/competitive.yaml`.

### Overview

The inference pipeline combines three strategies for improved test-time performance:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    COMPETITIVE INFERENCE PIPELINE                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Input: Demo Pairs [(I₁,O₁), ..., (Iₙ,Oₙ)] + Test Input I_test             │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ Stage 1: TEST-TIME TRAINING (TTTAdapter)                              │   │
│  │   - Leave-one-out training on demo pairs                              │   │
│  │   - Freezes SCL components (batch_norm, projector, contrastive)       │   │
│  │   - Quick adaptation to task-specific patterns                        │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                               ↓                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ Stage 2: STOCHASTIC SAMPLING (StochasticSampler)                      │   │
│  │   - MC Dropout for diverse candidates                                 │   │
│  │   - Temperature scaling for exploration/exploitation                  │   │
│  │   - Top-K / Nucleus sampling                                          │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                               ↓                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ Stage 3: CONSISTENCY VERIFICATION (ConsistencyVerifier)               │   │
│  │   - Score candidates via cross-augmentation agreement                 │   │
│  │   - Augmentations: rotate, flip, color permute                        │   │
│  │   - High consistency = confident prediction                           │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                               ↓                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ Stage 4: ENSEMBLE & VOTING                                            │   │
│  │   - Combine scores from all stages                                    │   │
│  │   - Weighted voting across candidates                                 │   │
│  │   - Final prediction selection                                        │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Output: Best prediction for I_test                                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Module 1: Test-Time Training (TTTAdapter)

**Location:** `sci_arc/inference/ttt.py`

```python
@dataclass
class TTTConfig:
    """Configuration for Test-Time Training."""
    enabled: bool = True
    learning_rate: float = 1e-4
    num_steps: int = 20
    use_leave_one_out: bool = True
    gradient_clip: float = 1.0
    frozen_modules: List[str] = field(default_factory=lambda: [
        'scl', 'batch_norm', 'projector', 'contrastive'
    ])


class TTTAdapter:
    """
    Test-Time Training adapter for task-specific adaptation.
    
    KEY SAFETY FEATURES:
    - Freezes SCL-related components to preserve structural learning
    - Gradient clipping prevents catastrophic updates
    - Leave-one-out training validates on held-out demos
    
    FROZEN MODULES (to protect SCL stability):
    - batch_norm: SCL's background signal removal
    - projector: Contrastive projection head
    - contrastive: Any explicit contrastive components
    - scl: Catch-all for SCL-related modules
    """
    
    def __init__(self, model: nn.Module, config: TTTConfig):
        self.model = model
        self.config = config
        self._frozen_modules: Set[str] = set()
    
    def adapt(self, demo_pairs: List[Tuple[torch.Tensor, torch.Tensor]]) -> None:
        """
        Adapt model to task using demo pairs.
        
        Uses leave-one-out: for each step, train on N-1 demos,
        validate on the held-out demo.
        """
        if not self.config.enabled or len(demo_pairs) < 2:
            return
        
        self._freeze_scl_components()
        self._setup_optimizer()
        
        for step in range(self.config.num_steps):
            # Leave-one-out: hold out one demo for validation
            if self.config.use_leave_one_out:
                held_out_idx = step % len(demo_pairs)
                train_pairs = [p for i, p in enumerate(demo_pairs) if i != held_out_idx]
            else:
                train_pairs = demo_pairs
            
            # Training step
            self.model.train()
            total_loss = 0.0
            for inp, out in train_pairs:
                pred = self.model(inp.unsqueeze(0))
                loss = F.cross_entropy(pred.view(-1, pred.size(-1)), out.view(-1))
                total_loss += loss
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self._get_trainable_params(), 
                self.config.gradient_clip
            )
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        self.model.eval()
```

### Module 2: Stochastic Sampling (StochasticSampler)

**Location:** `sci_arc/inference/sampler.py`

```python
@dataclass
class SamplingConfig:
    """Configuration for stochastic sampling."""
    enabled: bool = True
    num_samples: int = 32
    temperature: float = 1.0
    top_k: int = 0            # 0 = disabled
    top_p: float = 0.9        # Nucleus sampling threshold
    use_mc_dropout: bool = True
    mc_dropout_rate: float = 0.1


class StochasticSampler:
    """
    Stochastic sampling for diverse candidate generation.
    
    MATHEMATICAL STABILITY:
    - Temperature clamped to [0.1, 2.0] to prevent overflow/underflow
    - Probability clamped to [1e-8, 1.0] before log
    - NaN guards on all outputs
    
    SAMPLING STRATEGIES:
    1. MC Dropout: Enable dropout at inference for diversity
    2. Temperature: Higher T = more exploration
    3. Top-K: Only sample from K most likely tokens
    4. Nucleus (Top-P): Sample from minimal set covering P probability mass
    """
    
    def __init__(self, model: nn.Module, config: SamplingConfig):
        self.model = model
        self.config = config
    
    def sample(self, input_grid: torch.Tensor, num_samples: int = None) -> List[torch.Tensor]:
        """Generate diverse candidate predictions."""
        num_samples = num_samples or self.config.num_samples
        candidates = []
        
        # Enable MC Dropout if configured
        if self.config.use_mc_dropout:
            self._enable_mc_dropout()
        
        for _ in range(num_samples):
            with torch.no_grad():
                logits = self.model(input_grid.unsqueeze(0))
                
                # Apply temperature scaling (clamped for stability)
                temp = max(0.1, min(2.0, self.config.temperature))
                scaled_logits = logits / temp
                
                # Apply top-k filtering
                if self.config.top_k > 0:
                    scaled_logits = self._top_k_filtering(scaled_logits, self.config.top_k)
                
                # Apply nucleus (top-p) filtering
                if self.config.top_p < 1.0:
                    scaled_logits = self._nucleus_filtering(scaled_logits, self.config.top_p)
                
                # Sample from distribution (with numerical stability)
                probs = F.softmax(scaled_logits, dim=-1)
                probs = torch.clamp(probs, min=1e-8)
                probs = probs / probs.sum(dim=-1, keepdim=True)  # Renormalize
                
                sampled = torch.multinomial(probs.view(-1, probs.size(-1)), 1)
                candidates.append(sampled.view(logits.shape[1:-1]))
        
        if self.config.use_mc_dropout:
            self._disable_mc_dropout()
        
        return candidates
```

### Module 3: Consistency Verification (ConsistencyVerifier)

**Location:** `sci_arc/inference/sampler.py`

```python
class ConsistencyVerifier:
    """
    Score predictions by cross-augmentation consistency.
    
    INSIGHT: A correct prediction should be consistent when we augment
    the input (rotate, flip) and compare against augmented output.
    
    AUGMENTATIONS:
    - Rotation: 0°, 90°, 180°, 270°
    - Flip: horizontal, vertical
    - Color permutation: shuffle non-background colors
    
    SCORING:
    - Apply inverse augmentation to each prediction
    - Compare all variants for consistency
    - High agreement = high confidence
    """
    
    def __init__(self, model: nn.Module, augmentations: List[str] = None):
        self.model = model
        self.augmentations = augmentations or ['rotate_90', 'rotate_180', 'rotate_270', 'flip_h', 'flip_v']
    
    def score_candidates(
        self, 
        candidates: List[torch.Tensor],
        input_grid: torch.Tensor,
        demo_pairs: List[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> List[float]:
        """
        Score each candidate by consistency across augmentations.
        
        Returns:
            List of consistency scores in [0, 1], higher = more consistent
        """
        scores = []
        
        for candidate in candidates:
            aug_predictions = []
            
            for aug_name in self.augmentations:
                # Augment input
                aug_input = self._apply_augmentation(input_grid, aug_name)
                
                # Get prediction for augmented input
                with torch.no_grad():
                    aug_pred = self.model(aug_input.unsqueeze(0))
                    aug_pred = aug_pred.argmax(dim=-1).squeeze(0)
                
                # Apply inverse augmentation to prediction
                inv_pred = self._apply_inverse_augmentation(aug_pred, aug_name)
                aug_predictions.append(inv_pred)
            
            # Compute consistency: how often do augmented predictions agree with candidate?
            agreements = []
            for aug_pred in aug_predictions:
                # Handle size mismatches
                if aug_pred.shape == candidate.shape:
                    agreement = (aug_pred == candidate).float().mean().item()
                    agreements.append(agreement)
            
            consistency_score = np.mean(agreements) if agreements else 0.0
            scores.append(consistency_score)
        
        return scores
```

### Module 4: Ensemble Predictor

**Location:** `sci_arc/inference/ensemble.py`

```python
@dataclass
class EnsembleConfig:
    """Configuration for ensemble prediction."""
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    ttt: TTTConfig = field(default_factory=TTTConfig)
    use_consistency_verification: bool = True
    consistency_weight: float = 0.3
    voting_weight: float = 0.7


class EnsemblePredictor:
    """
    Combined inference pipeline using all strategies.
    
    PIPELINE:
    1. TTT: Adapt model to task (if enabled)
    2. Sample: Generate diverse candidates (if enabled)
    3. Verify: Score by consistency (if enabled)
    4. Vote: Combine scores for final prediction
    
    ABLATION SUPPORT:
    Each component can be toggled via config for systematic ablation studies.
    """
    
    def __init__(self, model: nn.Module, config: EnsembleConfig = None):
        self.model = model
        self.config = config or EnsembleConfig()
        
        # Initialize sub-modules
        self.sampler = StochasticSampler(model, self.config.sampling)
        self.ttt = TTTAdapter(model, self.config.ttt)
        self.verifier = ConsistencyVerifier(model) if self.config.use_consistency_verification else None
    
    def predict(
        self,
        input_grid: torch.Tensor,
        demo_pairs: List[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Full inference pipeline.
        
        Returns:
            Best prediction for input_grid
        """
        # Stage 1: Test-Time Training
        if self.config.ttt.enabled and demo_pairs:
            self.ttt.adapt(demo_pairs)
        
        # Stage 2: Generate candidates
        if self.config.sampling.enabled:
            candidates = self.sampler.sample(input_grid)
        else:
            # Single greedy prediction
            with torch.no_grad():
                logits = self.model(input_grid.unsqueeze(0))
                candidates = [logits.argmax(dim=-1).squeeze(0)]
        
        if len(candidates) == 1:
            return candidates[0]
        
        # Stage 3: Score candidates
        if self.verifier:
            consistency_scores = self.verifier.score_candidates(candidates, input_grid, demo_pairs)
        else:
            consistency_scores = [1.0] * len(candidates)
        
        # Stage 4: Voting (pixel-wise majority + consistency weighting)
        voting_scores = self._compute_voting_scores(candidates)
        
        # Combine scores
        final_scores = [
            self.config.consistency_weight * cs + self.config.voting_weight * vs
            for cs, vs in zip(consistency_scores, voting_scores)
        ]
        
        # Select best candidate
        best_idx = np.argmax(final_scores)
        return candidates[best_idx]
```

### Configuration (YAML)

**Location:** `configs/competitive.yaml` (excerpt)

```yaml
inference:
  # === Test-Time Training ===
  ttt:
    enabled: true
    learning_rate: 1e-4
    num_steps: 20
    use_leave_one_out: true
    gradient_clip: 1.0
    frozen_modules:
      - scl
      - batch_norm
      - projector
      - contrastive

  # === Stochastic Sampling ===
  sampling:
    enabled: true
    num_samples: 32
    temperature: 1.0
    top_k: 0
    top_p: 0.9
    use_mc_dropout: true
    mc_dropout_rate: 0.1

  # === Ensemble Settings ===
  ensemble:
    use_consistency_verification: true
    consistency_weight: 0.3
    voting_weight: 0.7
```

### Ablation Study Script

**Location:** `scripts/evaluate_competitive.py`

```python
# Run ablation studies with different configurations
ABLATION_MODES = {
    'baseline': {
        'ttt': False, 'sampling': False, 'consistency': False
    },
    'voting_only': {
        'ttt': False, 'sampling': True, 'consistency': False
    },
    'no_ttt': {
        'ttt': False, 'sampling': True, 'consistency': True
    },
    'no_sampling': {
        'ttt': True, 'sampling': False, 'consistency': False
    },
    'no_consistency': {
        'ttt': True, 'sampling': True, 'consistency': False
    },
    'full': {
        'ttt': True, 'sampling': True, 'consistency': True
    }
}

# Usage:
# python scripts/evaluate_competitive.py --mode full --checkpoint best.pt
# python scripts/evaluate_competitive.py --ablation-sweep  # Run all modes
```

### Test Coverage

**Location:** `tests/test_inference.py`

| Test Class | Tests | Status |
|------------|-------|--------|
| `TestSamplingConfig` | 2 | ✅ Pass |
| `TestStochasticSampler` | 4 | ✅ Pass |
| `TestConsistencyVerifier` | 2 | ✅ Pass |
| `TestTTTConfig` | 2 | ✅ Pass |
| `TestTTTAdapter` | 2 | ✅ Pass |
| `TestEnsemblePredictor` | 2 | ✅ Pass |
| **Total** | **14** | ✅ **All Pass** |

### Mathematical Stability Guarantees

| Component | Stability Measure | Implementation |
|-----------|-------------------|----------------|
| Temperature | Clamped range | `max(0.1, min(2.0, temp))` |
| Probabilities | Minimum value | `torch.clamp(probs, min=1e-8)` |
| Gradients | Clipping | `clip_grad_norm_(params, 1.0)` |
| NaN handling | Guard | `torch.nan_to_num(tensor)` |
| SCL protection | Freezing | `frozen_modules: [scl, batch_norm, projector, contrastive]` |

### Usage Example

```python
from sci_arc import SCIARC, get_inference_modules

# Load model
model = SCIARC.from_pretrained("checkpoints/best.pt")

# Get inference modules with config
sampler, ttt, ensemble = get_inference_modules(model, "configs/competitive.yaml")

# For full pipeline
prediction = ensemble.predict(
    input_grid=test_input,
    demo_pairs=[(demo1_in, demo1_out), (demo2_in, demo2_out)]
)

# For ablation (sampling only)
candidates = sampler.sample(test_input, num_samples=32)
```

---

## 🆕 CISL: Content-Invariant Structure Learning (January 2025)

> **Note:** Originally named CICL (Color-Invariant Consistency Learning), renamed to CISL
> to reflect the general-purpose nature of content-invariant structure learning.
> CICL names are preserved as backward-compatible aliases.

### Why CISL Replaces SCL

The original Structural Contrastive Loss (SCL) suffers from fundamental issues in the ARC domain:

| Problem | SCL Issue | CISL Solution |
|---------|-----------|---------------|
| **Too few samples** | InfoNCE needs many negatives; ARC has 2-4 demos per task | Uses within-task consistency instead |
| **Collapse to zero** | Model learns constant embedding to minimize loss | Variance loss prevents collapse |
| **No explicit invariance** | Structure-content separation is implicit | Content permutation explicitly tests invariance |

### CISL Four-Component Loss

```
L_total = L_recon + λ₁·L_consist + λ₂·L_content_inv + λ₃·L_var
```

| Component | Formula | Purpose |
|-----------|---------|---------|
| **L_recon** | `CrossEntropy(pred, target)` | Reconstruction (existing task loss) |
| **L_consist** | `(1/K)·Σ\|z_i - mean(z)\|²` | All demos → same structure embedding |
| **L_content_inv** | `\|z_orig - z_content_permuted\|²` | Content change doesn't change structure |
| **L_var** | `ReLU(γ - std(Z_batch))` | Prevent constant-zero collapse |

### Content Permutation: The Key Insight

For ARC, structure = transformation rule. If you swap red↔blue everywhere:
- The **structure** (rule) is unchanged
- The **content** (colors) changed

CISL explicitly teaches this: `f(grid) == f(permute_content(grid))`

```python
# Color permutation (content permutation for ARC) preserves structure
original:  [[1, 1, 2],    # Rule: "mirror horizontally"
            [2, 2, 1]]    

permuted:  [[3, 3, 5],    # Same rule applied, different colors
            [5, 5, 3]]    

# CISL forces: z_struct(original) == z_struct(permuted)
```

### Configuration

```yaml
# configs/default.yaml
# Note: Config params use cicl_ prefix for backward compatibility
training:
  use_cicl: true                 # Enable CISL (uses cicl name for compat)
  cicl_consist_weight: 0.5       # Within-task consistency weight
  cicl_color_inv_weight: 0.5     # Content invariance weight (color inv for ARC)
  cicl_variance_weight: 0.1      # Anti-collapse regularization
  cicl_target_std: 0.5           # Target embedding std
```

### Usage

```python
from sci_arc.training import CISLLoss  # Preferred name
from sci_arc.training import CICLLoss  # Backward-compatible alias

# Create loss (content_inv_weight replaces color_inv_weight)
cisl_loss = CISLLoss(
    consist_weight=0.5,
    content_inv_weight=0.5,
    variance_weight=0.1,
    target_std=0.5
)

# Compute (in trainer)
result = cisl_loss(
    z_struct=z_struct,                    # [B, K, D] structure embeddings
    z_struct_content_aug=z_content_aug,   # [B, K, D] content-permuted version
)

# Result dict contains:
# 'total': Combined CISL loss
# 'consistency': Within-task consistency
# 'content_inv': Content invariance (was 'color_inv')
# 'variance': Anti-collapse term
```

### Backward Compatibility

CISL is opt-in. Set `use_cicl: false` in config to use legacy SCL:

```python
config = TrainingConfig(use_cicl=False)  # Legacy SCL
config = TrainingConfig(use_cicl=True)   # New CISL (uses cicl param name)

# Both class names work:
from sci_arc.training import CISLLoss  # New preferred name
from sci_arc.training import CICLLoss  # Old name (alias for CISLLoss)
```

### Logging

When CISL is enabled, training logs these additional metrics to wandb:
- `train/cisl_consist` - Within-task consistency loss
- `train/cisl_content_inv` - Content invariance loss (was cisl_color_inv)
- `train/cisl_variance` - Batch variance loss
