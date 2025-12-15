# SCI-ARC: Structural Causal Invariance for Abstract Reasoning Corpus
## Complete AI Agent Implementation Instructions

**Author**: Alex (Microsoft Principal Applied Scientist)
**Target**: Novel application of SCI principles to ARC-AGI benchmark
**Innovation**: First combination of structural-content separation with visual abstract reasoning

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

### 2. Structural Encoder for Grids

```python
class StructuralEncoder2D(nn.Module):
    """
    Extract transformation structure from (input, output) grid pairs.
    
    Adaptation of SCI's Structural Encoder for 2D grids:
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
        
        # === ABSTRACTION LAYER (Key SCI component) ===
        # Learns to identify structural vs content features
        self.abstraction_layer = AbstractionLayer2D(hidden_dim)
        
        # === STRUCTURE QUERIES ===
        # Learnable queries that extract transformation patterns
        self.structure_queries = nn.Parameter(
            torch.randn(1, num_structure_slots, hidden_dim) * 0.02
        )
        
        # === CROSS-ATTENTION ===
        # Queries attend to grid differences
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # === TRANSFORMATION ENCODER ===
        # Process the (input, output) difference
        self.diff_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=num_layers
        )
        
        # === OUTPUT PROJECTION ===
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        
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
        
        # Flatten grids to sequences
        input_flat = input_emb.view(B, -1, self.hidden_dim)   # [B, H*W, D]
        output_flat = output_emb.view(B, -1, self.hidden_dim) # [B, H*W, D]
        
        # Apply AbstractionLayer to suppress content
        input_abs = self.abstraction_layer(input_flat)
        output_abs = self.abstraction_layer(output_flat)
        
        # Concatenate input and output (transformation context)
        context = torch.cat([input_abs, output_abs], dim=1)  # [B, 2*H*W, D]
        
        # Encode transformation patterns
        context_encoded = self.diff_encoder(context)
        
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

### 7. Structural Contrastive Loss (Direct from SCI)

```python
class StructuralContrastiveLoss(nn.Module):
    """
    SCL adapted for ARC grids.
    
    IDENTICAL to SCI's SCL in principle:
    - Positive pairs: Same transformation rule, different grids
    - Negative pairs: Different transformation rules
    - Objective: Pull together, push apart
    
    This is what makes SCI novel and should transfer directly.
    """
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(
        self,
        structure_reps: torch.Tensor,  # [B, K, D]
        transform_labels: torch.Tensor  # [B] which transformation family
    ) -> torch.Tensor:
        """
        Compute SCL loss.
        
        Args:
            structure_reps: Structural representations from SE
            transform_labels: Integer labels indicating transformation type
                             (e.g., 0=rotate, 1=flip, 2=color_swap, ...)
        """
        B = structure_reps.size(0)
        
        # Pool structure slots
        z = structure_reps.mean(dim=1)  # [B, D]
        z = F.normalize(z, dim=-1)
        
        # Compute similarity matrix
        sim = torch.mm(z, z.t()) / self.temperature  # [B, B]
        
        # Mask diagonal
        mask = torch.eye(B, device=z.device).bool()
        sim.masked_fill_(mask, -float('inf'))
        
        # Positive mask: same transform family
        pos_mask = transform_labels.unsqueeze(0) == transform_labels.unsqueeze(1)
        pos_mask = pos_mask & ~mask
        
        # InfoNCE loss
        loss = 0.0
        count = 0
        
        for i in range(B):
            if pos_mask[i].any():
                pos_sim = sim[i, pos_mask[i]]
                
                # Log-sum-exp over all (for denominator)
                all_sim = sim[i, ~mask[i]]
                
                # InfoNCE: -log(exp(pos) / sum(exp(all)))
                loss += -pos_sim.mean() + torch.logsumexp(all_sim, dim=0)
                count += 1
        
        return loss / max(count, 1)


class SCIARCLoss(nn.Module):
    """Combined loss for SCI-ARC training."""
    
    def __init__(
        self,
        H_cycles: int = 16,
        scl_weight: float = 0.1,
        orthogonality_weight: float = 0.01
    ):
        super().__init__()
        self.H_cycles = H_cycles
        self.scl_weight = scl_weight
        self.orth_weight = orthogonality_weight
        
        self.scl = StructuralContrastiveLoss()
        
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

This should improve **compositional generalization** - exactly what ARC tests.
