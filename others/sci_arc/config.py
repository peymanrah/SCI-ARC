"""
SCI-ARC Configuration

Centralized configuration for SCI-ARC model, training, and evaluation.
Uses Pydantic for validation and type safety.
"""

from typing import Optional, List, Literal
from dataclasses import dataclass, field


@dataclass
class SCIARCConfig:
    """
    Main configuration for SCI-ARC model.
    
    Architecture Parameters:
    - hidden_dim: Core hidden dimension (embedding size)
    - num_heads: Number of attention heads
    - dropout: Dropout probability
    
    SCI Parameters:
    - num_structure_slots: Number of structural pattern slots
    - num_content_slots: Number of content/object slots
    - abstraction_layers: Depth of the abstraction module
    
    TRM Parameters:
    - H_cycles: Number of outer supervision cycles (TRM's K)
    - L_cycles: Number of inner recursion steps per H-cycle
    - L_layers: Depth of reasoning module
    
    Training Parameters:
    - learning_rate: Base learning rate
    - weight_decay: L2 regularization
    - warmup_steps: Linear warmup steps
    - max_epochs: Maximum training epochs
    
    Loss Weights:
    - scl_weight: Weight for Structural Contrastive Loss
    - orthogonality_weight: Weight for orthogonality constraint
    - deep_supervision_weight: Weight for intermediate supervision
    """
    
    # === MODEL ARCHITECTURE ===
    hidden_dim: int = 256
    num_heads: int = 8
    dropout: float = 0.1
    num_colors: int = 10
    max_grid_size: int = 30
    
    # === SCI ENCODER PARAMETERS ===
    num_structure_slots: int = 16
    num_content_slots: int = 32
    max_objects: int = 32  # Alias for content slots (used in content encoder)
    abstraction_layers: int = 2
    se_layers: int = 3  # Structural encoder layers
    ce_layers: int = 3  # Content encoder layers  
    structure_encoder_layers: int = 3  # Alias for se_layers
    content_encoder_layers: int = 3    # Alias for ce_layers
    use_abstraction: bool = True
    demo_aggregation: str = "attention"  # "attention", "mean", "last"
    
    # === TRM RECURSIVE PARAMETERS ===
    H_cycles: int = 16  # Outer supervision loops
    L_cycles: int = 4   # Inner recursion per H-cycle
    L_layers: int = 2   # Reasoning network depth
    latent_size: int = 64  # Latent sequence length
    
    # === TRAINING ===
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    warmup_ratio: float = 0.1
    max_epochs: int = 100
    batch_size: int = 16  # Reduced from 32 to prevent VRAM overflow
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # === LOSS WEIGHTS ===
    scl_weight: float = 0.1
    orthogonality_weight: float = 0.01
    deep_supervision_weight: float = 0.1
    deep_supervision_gamma: float = 0.8  # Decay factor for intermediate losses
    
    # === MEMORY OPTIMIZATION ===
    use_memory_efficient_training: bool = True
    use_amp: bool = True  # Automatic Mixed Precision
    amp_dtype: str = "bfloat16"  # "float16" or "bfloat16"
    
    # === FEATURE FLAGS ===
    use_task_conditioning: bool = True
    use_deep_supervision: bool = True
    deep_supervision: bool = True  # Alias for use_deep_supervision
    use_film_conditioning: bool = True
    use_embed_scaling: bool = True  # TRM-style sqrt(d) scaling
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        assert self.hidden_dim > 0, "hidden_dim must be positive"
        assert self.num_heads > 0, "num_heads must be positive"
        assert self.hidden_dim % self.num_heads == 0, "hidden_dim must be divisible by num_heads"
        assert 0 <= self.dropout < 1, "dropout must be in [0, 1)"
        assert self.H_cycles > 0, "H_cycles must be positive"
        assert self.L_cycles > 0, "L_cycles must be positive"
        assert self.learning_rate > 0, "learning_rate must be positive"

    @classmethod
    def small(cls) -> "SCIARCConfig":
        """Small configuration for testing and debugging."""
        return cls(
            hidden_dim=64,
            num_heads=4,
            H_cycles=4,
            L_cycles=2,
            L_layers=1,
            abstraction_layers=1,
            structure_encoder_layers=2,
            content_encoder_layers=2,
        )
    
    @classmethod
    def base(cls) -> "SCIARCConfig":
        """Base configuration (~8M parameters)."""
        return cls(
            hidden_dim=256,
            num_heads=8,
            H_cycles=16,
            L_cycles=4,
            L_layers=2,
        )
    
    @classmethod
    def large(cls) -> "SCIARCConfig":
        """Large configuration (~20M parameters)."""
        return cls(
            hidden_dim=384,
            num_heads=12,
            H_cycles=24,
            L_cycles=6,
            L_layers=3,
            abstraction_layers=3,
        )
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "SCIARCConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class TrainingConfig:
    """Training-specific configuration."""
    
    # Dataset
    data_dir: str = "./data/arc"
    train_file: str = "training"
    eval_file: str = "evaluation"
    
    # Training loop
    num_epochs: int = 100
    batch_size: int = 16  # Reduced from 32 to prevent VRAM overflow
    eval_batch_size: int = 16  # Reduced from 64 to prevent VRAM overflow
    num_workers: int = 8
    
    # Optimization
    learning_rate: float = 1e-4
    min_learning_rate: float = 1e-6
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    gradient_clip: float = 1.0
    gradient_accumulation_steps: int = 1
    
    # Curriculum learning
    use_curriculum: bool = True
    curriculum_stages: int = 5
    curriculum_metric: str = "accuracy"
    
    # Checkpointing
    save_dir: str = "./checkpoints"
    save_every: int = 5  # epochs
    keep_last_n: int = 3
    
    # Logging
    log_every: int = 100  # steps
    use_wandb: bool = False
    wandb_project: str = "sci-arc"
    wandb_run_name: Optional[str] = None
    
    # Early stopping
    early_stopping: bool = True
    patience: int = 10
    min_delta: float = 0.001


@dataclass  
class EvaluationConfig:
    """Evaluation-specific configuration."""
    
    # Data
    eval_file: str = "evaluation"
    
    # Search
    num_attempts: int = 3
    temperature: float = 0.0  # 0 = greedy
    beam_size: int = 1
    
    # Metrics
    compute_per_task_metrics: bool = True
    save_predictions: bool = True
    output_dir: str = "./outputs"


def get_default_config() -> SCIARCConfig:
    """Get default SCI-ARC configuration."""
    return SCIARCConfig.base()
