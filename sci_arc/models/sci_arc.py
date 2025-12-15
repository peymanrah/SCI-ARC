"""
Complete SCI-ARC Model

Integrates all components:
1. GridEncoder: Encode input/output grids
2. StructuralEncoder2D: Extract transformation patterns
3. ContentEncoder2D: Extract object content
4. CausalBinding2D: Bind structure to content → z_task
5. RecursiveRefinement: TRM-style iterative answer improvement

This is the main model class for SCI-ARC training and inference.
"""

from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel

from sci_arc.models.grid_encoder import GridEncoder
from sci_arc.models.structural_encoder import StructuralEncoder2D
from sci_arc.models.content_encoder import ContentEncoder2D
from sci_arc.models.causal_binding import CausalBinding2D, DemoAggregator
from sci_arc.models.recursive_refinement import RecursiveRefinement


class SCIARCConfig(BaseModel):
    """Configuration for SCI-ARC model."""
    
    # Dimensions
    hidden_dim: int = 256
    num_colors: int = 10
    max_grid_size: int = 30
    
    # Structural Encoder
    num_structure_slots: int = 8
    se_layers: int = 2
    use_abstraction: bool = True
    
    # Content Encoder
    max_objects: int = 16
    
    # Attention
    num_heads: int = 4
    dropout: float = 0.1
    
    # Recursive Refinement (TRM parameters)
    H_cycles: int = 16          # Supervision steps
    L_cycles: int = 4           # Recursion per step
    L_layers: int = 2           # Network depth
    latent_size: int = 64       # Latent sequence length
    
    # Training
    deep_supervision: bool = True
    use_task_conditioning: bool = True
    
    # Demo aggregation
    demo_aggregation: str = "attention"  # "mean", "attention", or "max"
    
    class Config:
        extra = "allow"


@dataclass
class SCIARCOutput:
    """Output container for SCI-ARC forward pass."""
    
    # Predictions
    predictions: List[torch.Tensor]  # Predictions at each H-step
    final_prediction: torch.Tensor    # Final prediction [B, H, W, C]
    
    # For losses
    structure_rep: torch.Tensor       # [B, K, D] for SCL
    content_rep: torch.Tensor         # [B, M, D] for orthogonality loss
    z_task: torch.Tensor              # [B, D] task embedding
    
    # Optional (for analysis)
    binding_weights: Optional[torch.Tensor] = None
    structural_scores: Optional[torch.Tensor] = None


class SCIARC(nn.Module):
    """
    Complete SCI-ARC model.
    
    Combines:
    - SCI's structure-content separation (SE, CE, CBM)
    - TRM's recursive refinement
    - SCL for structural invariance
    
    Key Innovation:
    - First application of structural invariance to visual reasoning
    - Explicit separation of "what transformation" from "what objects"
    - z_task conditions recursive refinement with structural understanding
    
    Architecture Flow:
    1. DEMO ENCODING:
       - Encode demo (input, output) pairs
       - Extract structure (SE) and content (CE)
       - Bind to get z_task per demo
       - Aggregate across demos
    
    2. TEST PROCESSING:
       - Encode test input
       - Run recursive refinement conditioned on z_task
       - Produce output at each step (deep supervision)
    
    Total parameters: ~8M (intentionally small like TRM philosophy)
    """
    
    def __init__(self, config: SCIARCConfig):
        """
        Args:
            config: Model configuration
        """
        super().__init__()
        
        self.config = config
        
        # === SHARED GRID ENCODER ===
        self.grid_encoder = GridEncoder(
            hidden_dim=config.hidden_dim,
            num_colors=config.num_colors,
            max_size=config.max_grid_size,
            dropout=config.dropout
        )
        
        # === SCI COMPONENTS ===
        
        # Structural Encoder: Extract transformation patterns
        self.structural_encoder = StructuralEncoder2D(
            hidden_dim=config.hidden_dim,
            num_structure_slots=config.num_structure_slots,
            num_layers=config.se_layers,
            num_heads=config.num_heads,
            dropout=config.dropout,
            use_abstraction=config.use_abstraction
        )
        
        # Content Encoder: Extract objects (orthogonal to structure)
        self.content_encoder = ContentEncoder2D(
            hidden_dim=config.hidden_dim,
            max_objects=config.max_objects,
            num_heads=config.num_heads,
            dropout=config.dropout
        )
        
        # Causal Binding: Bind structure to content → z_task
        self.causal_binding = CausalBinding2D(
            hidden_dim=config.hidden_dim,
            num_structure_slots=config.num_structure_slots,
            num_content_slots=config.max_objects,
            num_heads=config.num_heads,
            dropout=config.dropout
        )
        
        # Demo Aggregator: Combine z_task from multiple demos
        self.demo_aggregator = DemoAggregator(
            hidden_dim=config.hidden_dim,
            aggregation_type=config.demo_aggregation
        )
        
        # === TRM COMPONENT ===
        
        # Recursive Refinement: Iterative answer improvement
        self.refiner = RecursiveRefinement(
            hidden_dim=config.hidden_dim,
            max_cells=config.max_grid_size ** 2,
            num_colors=config.num_colors,
            H_cycles=config.H_cycles,
            L_cycles=config.L_cycles,
            L_layers=config.L_layers,
            latent_size=config.latent_size,
            num_heads=config.num_heads,
            dropout=config.dropout,
            deep_supervision=config.deep_supervision,
            use_task_conditioning=config.use_task_conditioning
        )
    
    def encode_demos(
        self,
        demo_pairs: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode demo pairs to get task understanding.
        
        Args:
            demo_pairs: List of (input_grid, output_grid) tensors
                       Each grid is [B, H, W] integers
        
        Returns:
            z_task: [B, D] task embedding
            structure_agg: [B, K, D] aggregated structure
            content_agg: [B, M, D] aggregated content
        """
        all_structure_reps = []
        all_content_reps = []
        all_z_tasks = []
        
        for input_grid, output_grid in demo_pairs:
            # Encode grids
            input_emb = self.grid_encoder(input_grid)   # [B, H_in, W_in, D]
            output_emb = self.grid_encoder(output_grid) # [B, H_out, W_out, D]
            
            # Extract structure from (input, output) transformation
            structure_rep = self.structural_encoder(input_emb, output_emb)  # [B, K, D]
            all_structure_reps.append(structure_rep)
            
            # Extract content from input (orthogonal to structure)
            content_rep = self.content_encoder(input_emb, structure_rep)  # [B, M, D]
            all_content_reps.append(content_rep)
            
            # Bind structure to content
            z_task_demo = self.causal_binding(structure_rep, content_rep)  # [B, D]
            all_z_tasks.append(z_task_demo)
        
        # Aggregate structure and content across demos
        structure_agg = torch.stack(all_structure_reps, dim=1).mean(dim=1)  # [B, K, D]
        content_agg = torch.stack(all_content_reps, dim=1).mean(dim=1)      # [B, M, D]
        
        # Aggregate z_task across demos
        z_tasks_stacked = torch.stack(all_z_tasks, dim=1)  # [B, num_demos, D]
        z_task = self.demo_aggregator(z_tasks_stacked)      # [B, D]
        
        return z_task, structure_agg, content_agg
    
    def forward(
        self,
        input_grids: torch.Tensor = None,
        output_grids: torch.Tensor = None,
        test_input: torch.Tensor = None,
        test_output: torch.Tensor = None,
        grid_mask: torch.Tensor = None,
        # Legacy interface for inference
        demo_pairs: List[Tuple[torch.Tensor, torch.Tensor]] = None,
        target_shape: Tuple[int, int] = None,
        **kwargs
    ):
        """
        Unified forward pass supporting both training and inference interfaces.
        
        Training interface (batched):
            input_grids: [B, num_pairs, H, W] batched input grids
            output_grids: [B, num_pairs, H, W] batched output grids
            test_input: [B, H, W] test input grid
            test_output: [B, H_out, W_out] target (for shape inference)
            grid_mask: Optional [B, num_pairs] mask for valid demos
        
        Inference interface (demo_pairs):
            demo_pairs: List of (input, output) grid tensors
            test_input: [B, H, W] test input grid
            target_shape: (H_out, W_out) expected output size
        
        Returns:
            Dict with logits, z_struct, z_content, z_task, intermediate_logits
        """
        # Dispatch to appropriate method based on arguments
        if demo_pairs is not None:
            # Legacy inference interface
            return self._forward_demo_pairs(demo_pairs, test_input, target_shape)
        elif input_grids is not None:
            # Training interface - delegate to forward_training
            # Infer test_output if not provided
            if test_output is None:
                test_output = test_input  # Use same shape as input
            return self.forward_training(
                input_grids=input_grids,
                output_grids=output_grids,
                test_input=test_input,
                test_output=test_output,
                grid_mask=grid_mask,
                **kwargs
            )
        else:
            raise ValueError("Must provide either (input_grids, output_grids, test_input) or (demo_pairs, test_input, target_shape)")
    
    def _forward_demo_pairs(
        self,
        demo_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
        test_input: torch.Tensor,
        target_shape: Tuple[int, int]
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with demo_pairs interface (for inference).
        
        Args:
            demo_pairs: List of (input, output) grid tensors
                       Each grid: [B, H, W] integers (0-9)
            test_input: [B, H, W] test input grid
            target_shape: (H_out, W_out) expected output size
        
        Returns:
            Dict with predictions and intermediate representations
        """
        # === PHASE 1: Encode demos to get task understanding ===
        z_task, structure_rep, content_rep = self.encode_demos(demo_pairs)
        
        # === PHASE 2: Recursive refinement on test input ===
        
        # Encode test input
        test_emb = self.grid_encoder(test_input)  # [B, H, W, D]
        test_flat = test_emb.view(test_emb.size(0), -1, self.config.hidden_dim)  # [B, N, D]
        
        # Run recursive refinement
        predictions, final = self.refiner(test_flat, z_task, target_shape)
        
        return {
            'logits': final,
            'intermediate_logits': predictions,
            'z_struct': structure_rep,
            'z_content': content_rep,
            'z_task': z_task
        }
    
    def forward_training(
        self,
        input_grids: torch.Tensor,      # [B, num_pairs, H, W]
        output_grids: torch.Tensor,     # [B, num_pairs, H, W]
        test_input: torch.Tensor,       # [B, H, W]
        test_output: torch.Tensor,      # [B, H_out, W_out] (for shape inference)
        grid_mask: Optional[torch.Tensor] = None,  # [B, num_pairs] valid demo mask
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with training-compatible interface.
        
        This method bridges the gap between the trainer's expected interface
        and the model's internal architecture.
        
        Args:
            input_grids: [B, num_pairs, H, W] batched input grids
            output_grids: [B, num_pairs, H, W] batched output grids
            test_input: [B, H, W] test input grid
            test_output: [B, H_out, W_out] target (used for shape)
            grid_mask: Optional [B, num_pairs] mask for valid demos
        
        Returns:
            Dict with:
                - 'logits': [B, H, W, C] final prediction
                - 'intermediate_logits': list of intermediate predictions
                - 'z_struct': [B, K, D] structural representation
                - 'z_content': [B, M, D] content representation  
                - 'z_task': [B, D] task embedding
        """
        B = input_grids.size(0)
        num_pairs = input_grids.size(1)
        
        # Get target shape from test_output
        target_shape = (test_output.size(1), test_output.size(2))
        
        # Process demo pairs sequentially to minimize peak VRAM usage
        # (Processing B*num_pairs at once causes VRAM overflow to shared memory)
        all_structure_reps = []
        all_content_reps = []
        all_z_tasks = []
        
        for p in range(num_pairs):
            # Get pair p for all batches
            inp = input_grids[:, p, :, :]   # [B, H, W]
            out = output_grids[:, p, :, :]  # [B, H, W]
            
            # Encode grids
            input_emb = self.grid_encoder(inp)   # [B, H, W, D]
            output_emb = self.grid_encoder(out)  # [B, H, W, D]
            
            # Extract structure
            structure_rep = self.structural_encoder(input_emb, output_emb)  # [B, K, D]
            all_structure_reps.append(structure_rep)
            
            # Extract content
            content_rep = self.content_encoder(input_emb, structure_rep)  # [B, M, D]
            all_content_reps.append(content_rep)
            
            # Bind
            z_task_demo = self.causal_binding(structure_rep, content_rep)  # [B, D]
            all_z_tasks.append(z_task_demo)
        
        # Aggregate across demos
        structure_agg = torch.stack(all_structure_reps, dim=1).mean(dim=1)  # [B, K, D]
        content_agg = torch.stack(all_content_reps, dim=1).mean(dim=1)      # [B, M, D]
        z_tasks_stacked = torch.stack(all_z_tasks, dim=1)  # [B, num_pairs, D]
        z_task = self.demo_aggregator(z_tasks_stacked)      # [B, D]
        
        # Encode test input
        test_emb = self.grid_encoder(test_input)  # [B, H, W, D]
        test_flat = test_emb.view(B, -1, self.config.hidden_dim)  # [B, N, D]
        
        # Run recursive refinement
        predictions, final = self.refiner(test_flat, z_task, target_shape)
        
        return {
            'logits': final,                    # [B, H, W, C]
            'intermediate_logits': predictions, # List of [B, H, W, C]
            'z_struct': structure_agg,          # [B, K, D]
            'z_content': content_agg,           # [B, M, D]
            'z_task': z_task,                   # [B, D]
        }
    
    def forward_with_analysis(
        self,
        demo_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
        test_input: torch.Tensor,
        target_shape: Tuple[int, int]
    ) -> SCIARCOutput:
        """
        Forward pass with additional outputs for analysis.
        
        Returns binding weights, structural scores, etc.
        """
        all_structure_reps = []
        all_content_reps = []
        all_z_tasks = []
        all_binding_weights = []
        all_structural_scores = []
        
        for input_grid, output_grid in demo_pairs:
            input_emb = self.grid_encoder(input_grid)
            output_emb = self.grid_encoder(output_grid)
            
            # Get structure with attention
            structure_rep, attn_weights, structural_scores = \
                self.structural_encoder.forward_with_attention(input_emb, output_emb)
            all_structure_reps.append(structure_rep)
            all_structural_scores.append(structural_scores)
            
            # Get content with attention
            content_rep, content_attn = \
                self.content_encoder.forward_with_attention(input_emb, structure_rep)
            all_content_reps.append(content_rep)
            
            # Bind with weights
            z_task_demo, binding_weights, slot_weights = \
                self.causal_binding.forward_with_binding_weights(structure_rep, content_rep)
            all_z_tasks.append(z_task_demo)
            all_binding_weights.append(binding_weights)
        
        structure_agg = torch.stack(all_structure_reps, dim=1).mean(dim=1)
        content_agg = torch.stack(all_content_reps, dim=1).mean(dim=1)
        
        z_tasks_stacked = torch.stack(all_z_tasks, dim=1)
        z_task = self.demo_aggregator(z_tasks_stacked)
        
        # Run refinement
        test_emb = self.grid_encoder(test_input)
        test_flat = test_emb.view(test_emb.size(0), -1, self.config.hidden_dim)
        predictions, final = self.refiner(test_flat, z_task, target_shape)
        
        return SCIARCOutput(
            predictions=predictions,
            final_prediction=final,
            structure_rep=structure_agg,
            content_rep=content_agg,
            z_task=z_task,
            binding_weights=torch.stack(all_binding_weights, dim=1) if all_binding_weights else None,
            structural_scores=torch.stack(all_structural_scores, dim=1) if all_structural_scores[0] is not None else None
        )
    
    def count_parameters(self) -> Dict[str, int]:
        """Count parameters per component."""
        counts = {
            "grid_encoder": sum(p.numel() for p in self.grid_encoder.parameters()),
            "structural_encoder": sum(p.numel() for p in self.structural_encoder.parameters()),
            "content_encoder": sum(p.numel() for p in self.content_encoder.parameters()),
            "causal_binding": sum(p.numel() for p in self.causal_binding.parameters()),
            "demo_aggregator": sum(p.numel() for p in self.demo_aggregator.parameters()),
            "refiner": sum(p.numel() for p in self.refiner.parameters()),
        }
        counts["total"] = sum(counts.values())
        return counts


class SCIARCForTRMComparison(SCIARC):
    """
    SCI-ARC variant designed for fair comparison with TRM.
    
    Uses TRM-compatible data format and evaluation.
    """
    
    def forward_trm_format(
        self,
        inputs: torch.Tensor,           # [B, seq_len] flattened grids
        labels: torch.Tensor,            # [B, seq_len] target grids
        puzzle_identifiers: torch.Tensor # [B] puzzle IDs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass using TRM data format.
        
        Args:
            inputs: Flattened input grids (30*30 = 900 tokens)
            labels: Flattened target grids
            puzzle_identifiers: Puzzle IDs for task-specific embeddings
        
        Returns:
            Dict with logits and predictions
        """
        B = inputs.size(0)
        
        # Reshape to 2D grids
        # TRM uses: PAD=0, EOS=1, digits=2-11
        # We use: 0-9 directly, so need to shift
        inputs_2d = (inputs.clamp(2, 11) - 2).view(B, 30, 30)  # Shift from TRM encoding
        
        # For this simplified version, use input as single demo
        # In full implementation, would use puzzle_identifiers to load demos
        demo_pairs = [(inputs_2d, inputs_2d)]  # Self-reference (will be replaced)
        
        z_task, structure_rep, content_rep = self.encode_demos(demo_pairs)
        
        # Encode test
        test_emb = self.grid_encoder(inputs_2d)
        test_flat = test_emb.view(B, -1, self.config.hidden_dim)
        
        # Get output shape from labels
        # For TRM, output is same shape as input (30x30)
        predictions, final = self.refiner(test_flat, z_task, (30, 30))
        
        # Convert to TRM format
        logits = final.view(B, 900, -1)  # [B, seq_len, num_colors]
        preds = logits.argmax(dim=-1)
        
        return {
            "logits": logits,
            "preds": preds,
            "structure_rep": structure_rep,
            "content_rep": content_rep,
            "z_task": z_task
        }
