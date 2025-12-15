"""
Test-Time Training (TTT) for SCI-ARC.

TTT adapts the model to each specific test task by fine-tuning
on the demonstration pairs before inference.

Mathematical Stability Considerations:
1. NEVER update SCL projection head - would destroy learned contrastive space
2. Gradient clipping is CRITICAL (small data = high variance gradients)
3. Learning rate must be low (1e-5 to 1e-4) to prevent catastrophic forgetting
4. State restoration MUST happen between tasks

Key Design Decisions:
1. Only fine-tune specific layers (grid_encoder, structural_encoder)
2. Freeze SCL-related components (batch_norm, projector)
3. Use task loss only (no SCL during TTT)
4. Short adaptation (10-50 steps)

Reference:
- TTT Original: https://arxiv.org/abs/1909.13231
- MARC (ARC-specific TTT): https://github.com/ekinakyurek/marc
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
import copy
import io

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW, SGD


@dataclass  
class TTTConfig:
    """Configuration for Test-Time Training."""
    
    # Enable/disable TTT
    enabled: bool = True
    
    # Number of adaptation steps per task
    num_steps: int = 20
    
    # Learning rate for adaptation
    # CRITICAL: Must be low to prevent catastrophic forgetting
    learning_rate: float = 1e-4
    
    # Weight decay (regularization)
    weight_decay: float = 0.01
    
    # Gradient clipping (CRITICAL for small batch stability)
    grad_clip: float = 1.0
    
    # Which modules to fine-tune
    # Options: 'grid_encoder', 'structural_encoder', 'content_encoder',
    #          'causal_binding', 'refiner'
    # NOTE: NEVER include 'scl' - would destroy contrastive learning
    finetune_modules: List[str] = field(default_factory=lambda: [
        'grid_encoder', 
        'structural_encoder'
    ])
    
    # Whether to use data augmentation during TTT
    use_augmentation: bool = True
    
    # Number of augmented versions per demo
    num_augmentations: int = 4
    
    # Use mixed precision during TTT
    use_amp: bool = True
    
    # Optimizer type ('adamw' or 'sgd')
    optimizer: str = 'adamw'
    
    # Device
    device: str = 'cuda'
    
    # Verbose logging
    verbose: bool = False


class TTTAdapter:
    """
    Test-Time Training Adapter for SCI-ARC.
    
    Adapts the model to each specific task by fine-tuning on the
    demonstration pairs before making predictions.
    
    Critical Safety Features:
    1. State backup/restore to prevent cross-task contamination
    2. Frozen SCL components to preserve contrastive space
    3. Gradient clipping for stability with small data
    4. Short training to prevent overfitting
    
    Usage:
        ttt = TTTAdapter(model, config)
        
        # Adapt and predict for one task
        adapted_model = ttt.adapt(demo_inputs, demo_outputs)
        prediction = adapted_model(test_input)
        
        # CRITICAL: Reset before next task
        ttt.reset()
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[TTTConfig] = None,
    ):
        self.model = model
        self.config = config or TTTConfig()
        self.device = torch.device(
            self.config.device if torch.cuda.is_available() else 'cpu'
        )
        
        # Store original state for restoration
        self._original_state: Optional[bytes] = None
        self._save_state()
        
        # Identify which parameters to optimize
        self._setup_param_groups()
        
        # Setup for AMP
        self.scaler = torch.amp.GradScaler('cuda') if (
            self.config.use_amp and self.device.type == 'cuda'
        ) else None
    
    def _save_state(self):
        """Save model state for later restoration."""
        # Use in-memory buffer for speed
        buffer = io.BytesIO()
        torch.save(self.model.state_dict(), buffer)
        self._original_state = buffer.getvalue()
    
    def _restore_state(self):
        """Restore model to original state."""
        if self._original_state is not None:
            buffer = io.BytesIO(self._original_state)
            state_dict = torch.load(buffer, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)
    
    def _setup_param_groups(self):
        """Identify parameters to fine-tune based on config."""
        self._trainable_params: List[nn.Parameter] = []
        self._frozen_modules: Set[str] = set()
        
        # Use a set to track parameter ids to avoid duplicates
        seen_param_ids: Set[int] = set()
        
        for name, module in self.model.named_modules():
            # Check if this module should be fine-tuned
            should_finetune = any(
                ft_name in name for ft_name in self.config.finetune_modules
            )
            
            # SAFETY: Never fine-tune SCL-related components
            is_scl_related = any(x in name.lower() for x in [
                'scl', 'batch_norm', 'projector', 'contrastive'
            ])
            
            if is_scl_related:
                # Freeze SCL components
                for param in module.parameters():
                    param.requires_grad = False
                self._frozen_modules.add(name)
            elif should_finetune:
                # Mark for fine-tuning (avoid duplicates)
                for param in module.parameters():
                    param_id = id(param)
                    if param.requires_grad and param_id not in seen_param_ids:
                        self._trainable_params.append(param)
                        seen_param_ids.add(param_id)
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer for TTT."""
        if self.config.optimizer == 'sgd':
            return SGD(
                self._trainable_params,
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        else:
            return AdamW(
                self._trainable_params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
    
    def _augment_demos(
        self,
        input_grids: torch.Tensor,   # [N, H, W]
        output_grids: torch.Tensor,  # [N, H, W]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Augment demonstrations for more training signal.
        
        Uses dihedral transforms (rotations + flips).
        """
        from ..evaluation.voting import dihedral_transform_torch
        
        all_inputs = [input_grids]
        all_outputs = [output_grids]
        
        # Apply random dihedral transforms
        for _ in range(self.config.num_augmentations - 1):
            tid = torch.randint(1, 8, (1,)).item()
            aug_in = dihedral_transform_torch(input_grids, tid)
            aug_out = dihedral_transform_torch(output_grids, tid)
            all_inputs.append(aug_in)
            all_outputs.append(aug_out)
        
        # Stack along demo dimension
        # [num_aug, N, H, W] -> [num_aug * N, H, W]
        aug_inputs = torch.cat(all_inputs, dim=0)
        aug_outputs = torch.cat(all_outputs, dim=0)
        
        return aug_inputs, aug_outputs
    
    def _compute_ttt_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute TTT loss (task loss only, no SCL).
        
        CRITICAL: We do NOT compute SCL during TTT because:
        1. SCL requires multiple tasks with different transforms
        2. A single task has only one transform type
        3. Updating on one transform would bias the contrastive space
        
        Args:
            outputs: Model outputs dict
            targets: Target grids [B, H, W]
        
        Returns:
            Task loss (cross entropy)
        """
        logits = outputs['logits']  # [B, H, W, C]
        
        B, H, W, C = logits.shape
        logits_flat = logits.view(B * H * W, C)
        targets_flat = targets.view(B * H * W)
        
        # Use label smoothing for regularization
        loss = F.cross_entropy(
            logits_flat, 
            targets_flat,
            label_smoothing=0.1  # Mild regularization
        )
        
        # Add deep supervision if available
        if 'intermediate_logits' in outputs and outputs['intermediate_logits']:
            deep_loss = 0.0
            for inter_logits in outputs['intermediate_logits']:
                inter_flat = inter_logits.view(B * H * W, C)
                deep_loss += F.cross_entropy(
                    inter_flat, 
                    targets_flat,
                    label_smoothing=0.1
                )
            deep_loss /= len(outputs['intermediate_logits'])
            loss = loss + 0.5 * deep_loss
        
        return loss
    
    @torch.enable_grad()
    def adapt(
        self,
        input_grids: torch.Tensor,   # [N, H, W] demo inputs
        output_grids: torch.Tensor,  # [N, H, W] demo outputs
    ) -> nn.Module:
        """
        Adapt model to a specific task.
        
        Fine-tunes on the demo pairs for a few steps.
        
        Args:
            input_grids: Demonstration input grids [N, H, W]
            output_grids: Demonstration output grids [N, H, W]
        
        Returns:
            Adapted model (same object, modified in-place)
        """
        if not self.config.enabled:
            return self.model
        
        # Move to device
        input_grids = input_grids.to(self.device)
        output_grids = output_grids.to(self.device)
        
        # Augment if configured
        if self.config.use_augmentation:
            input_grids, output_grids = self._augment_demos(input_grids, output_grids)
        
        N = input_grids.shape[0]
        
        # Set model to training mode for layers we're updating
        self.model.train()
        
        # Re-freeze SCL components (safety check)
        for name, module in self.model.named_modules():
            if name in self._frozen_modules:
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False
        
        # Create optimizer
        optimizer = self._create_optimizer()
        
        # Training loop
        losses = []
        for step in range(self.config.num_steps):
            optimizer.zero_grad()
            
            # Use leave-one-out: predict each demo output from others
            total_loss = torch.tensor(0.0, device=self.device)
            
            for i in range(N):
                # Leave one out
                mask = torch.ones(N, dtype=torch.bool)
                mask[i] = False
                
                train_inputs = input_grids[mask]    # [N-1, H, W]
                train_outputs = output_grids[mask]  # [N-1, H, W]
                test_input = input_grids[i]         # [H, W]
                test_output = output_grids[i]       # [H, W]
                
                # Batch dimensions
                train_inputs = train_inputs.unsqueeze(0)   # [1, N-1, H, W]
                train_outputs = train_outputs.unsqueeze(0)
                test_input = test_input.unsqueeze(0)       # [1, H, W]
                test_output = test_output.unsqueeze(0)     # [1, H, W]
                
                # Forward pass
                if self.scaler:
                    with torch.amp.autocast('cuda'):
                        outputs = self.model.forward_training(
                            input_grids=train_inputs,
                            output_grids=train_outputs,
                            test_input=test_input,
                            test_output=test_output,
                        )
                        loss = self._compute_ttt_loss(outputs, test_output)
                else:
                    outputs = self.model.forward_training(
                        input_grids=train_inputs,
                        output_grids=train_outputs,
                        test_input=test_input,
                        test_output=test_output,
                    )
                    loss = self._compute_ttt_loss(outputs, test_output)
                
                total_loss = total_loss + loss / N
            
            # Backward pass
            if self.scaler:
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self._trainable_params, 
                    self.config.grad_clip
                )
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self._trainable_params,
                    self.config.grad_clip
                )
                optimizer.step()
            
            losses.append(total_loss.item())
            
            if self.config.verbose:
                print(f"  TTT Step {step+1}/{self.config.num_steps}: loss={total_loss.item():.4f}")
        
        # Set back to eval mode
        self.model.eval()
        
        if self.config.verbose:
            print(f"  TTT Complete: loss {losses[0]:.4f} -> {losses[-1]:.4f}")
        
        return self.model
    
    def reset(self):
        """
        Reset model to original state.
        
        CRITICAL: Must be called between tasks to prevent
        cross-task contamination.
        """
        self._restore_state()
        # Re-setup param groups after restoration
        self._setup_param_groups()
    
    def predict_with_ttt(
        self,
        input_grids: torch.Tensor,   # [N, H, W]
        output_grids: torch.Tensor,  # [N, H, W]
        test_input: torch.Tensor,    # [H, W]
        target_shape: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """
        Adapt model and make prediction.
        
        Convenience method that combines adapt() and predict().
        Automatically resets after prediction.
        
        Args:
            input_grids: Demo inputs [N, H, W]
            output_grids: Demo outputs [N, H, W]
            test_input: Test input [H, W]
            target_shape: Expected output shape
        
        Returns:
            Predicted grid [H, W]
        """
        try:
            # Adapt
            self.adapt(input_grids, output_grids)
            
            # Predict
            test_input = test_input.to(self.device)
            input_grids = input_grids.to(self.device)
            output_grids = output_grids.to(self.device)
            
            if target_shape is None:
                target_shape = (test_input.shape[0], test_input.shape[1])
            
            with torch.no_grad():
                # Prepare batch
                batch_demos_in = input_grids.unsqueeze(0)
                batch_demos_out = output_grids.unsqueeze(0)
                batch_test = test_input.unsqueeze(0)
                test_output_dummy = torch.zeros(
                    1, target_shape[0], target_shape[1],
                    dtype=torch.long, device=self.device
                )
                
                outputs = self.model.forward_training(
                    input_grids=batch_demos_in,
                    output_grids=batch_demos_out,
                    test_input=batch_test,
                    test_output=test_output_dummy,
                )
                
                prediction = outputs['logits'][0].argmax(dim=-1)  # [H, W]
            
            return prediction.cpu()
        
        finally:
            # ALWAYS reset after prediction
            self.reset()
