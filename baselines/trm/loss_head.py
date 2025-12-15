# =============================================================================
# TRM Original Implementation - loss_head.py (Loss computation wrapper)
# Source: https://github.com/SamsungSAILMontreal/TinyRecursiveModels
# Authors: Samsung SAIL Montreal
# License: Apache 2.0 (check original repository for latest license)
# 
# This file implements the loss computation wrapper for TRM training.
# =============================================================================

from typing import Dict, Tuple, Any
import torch
import torch.nn.functional as F
from torch import nn


IGNORE_LABEL_ID = -100


class TRMLossHead(nn.Module):
    """
    Loss computation wrapper for TRM.
    
    Computes:
    1. Cross-entropy loss for next-token prediction
    2. Q-learning loss for ACT halting decisions
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        q_loss_weight: float = 0.1,
        use_halt_only: bool = True,
    ):
        super().__init__()
        self.model = model
        self.q_loss_weight = q_loss_weight
        self.use_halt_only = use_halt_only
        
    def forward(
        self, 
        carry: Any, 
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[Any, Dict[str, torch.Tensor]]:
        """
        Forward pass with loss computation.
        
        Args:
            carry: TRM carry state
            batch: Input batch with 'inputs', 'puzzle_identifiers', 'labels'
        
        Returns:
            Updated carry and dict with 'loss', 'logits', 'preds', etc.
        """
        carry, outputs = self.model(carry, batch)
        
        logits = outputs['logits']
        
        # Compute CE loss if labels provided
        if 'labels' in batch:
            labels = batch['labels']
            
            # Flatten for cross-entropy
            # logits: [B, seq_len, vocab_size]
            # labels: [B, seq_len]
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=IGNORE_LABEL_ID,
            )
            outputs['ce_loss'] = loss
            
            # Predictions
            preds = logits.argmax(dim=-1)
            outputs['preds'] = preds
            
            # Accuracy (ignoring padding)
            mask = (labels != IGNORE_LABEL_ID)
            if mask.any():
                correct = (preds == labels) & mask
                outputs['accuracy'] = correct.sum().float() / mask.sum().float()
            else:
                outputs['accuracy'] = torch.tensor(0.0, device=logits.device)
        else:
            loss = torch.tensor(0.0, device=logits.device)
            outputs['ce_loss'] = loss
        
        # Q-learning loss for ACT
        if self.q_loss_weight > 0:
            q_halt = outputs.get('q_halt_logits')
            q_continue = outputs.get('q_continue_logits')
            
            if q_halt is not None:
                if self.use_halt_only:
                    # Sigmoid loss for halt logits
                    # Target is whether the current prediction is correct
                    if 'accuracy' in outputs:
                        target_q = outputs['accuracy'].detach()
                        q_loss = F.binary_cross_entropy_with_logits(
                            q_halt,
                            target_q.expand_as(q_halt),
                        )
                    else:
                        q_loss = torch.tensor(0.0, device=logits.device)
                else:
                    # Full Q-learning with continue logits
                    target_q = outputs.get('target_q_continue')
                    if target_q is not None:
                        q_loss = F.mse_loss(torch.sigmoid(q_continue), target_q)
                    else:
                        q_loss = torch.tensor(0.0, device=logits.device)
                
                outputs['q_loss'] = q_loss
                loss = loss + self.q_loss_weight * q_loss
        
        outputs['loss'] = loss
        
        return carry, outputs
    
    @property
    def puzzle_emb(self):
        return self.model.puzzle_emb
    
    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        return self.model.initial_carry(batch)


class TRMWithLoss(nn.Module):
    """
    Complete TRM model with integrated loss computation for training.
    
    This wraps TRM + TRMLossHead for end-to-end training.
    """
    
    def __init__(
        self,
        config_dict: Dict,
        q_loss_weight: float = 0.1,
    ):
        super().__init__()
        from baselines.trm.models.recursive_reasoning.trm import TRM
        
        self.trm = TRM(config_dict)
        self.loss_head = TRMLossHead(self.trm, q_loss_weight=q_loss_weight)
        
    def forward(self, carry, batch):
        return self.loss_head(carry, batch)
    
    @property
    def puzzle_emb(self):
        return self.trm.puzzle_emb
    
    def initial_carry(self, batch):
        return self.trm.initial_carry(batch)
