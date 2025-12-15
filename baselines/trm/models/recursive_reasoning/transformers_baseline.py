# =============================================================================
# TRM Original Implementation - transformers_baseline.py
# Source: https://github.com/SamsungSAILMontreal/TinyRecursiveModels
# Authors: Samsung SAIL Montreal
# License: Apache 2.0 (check original repository for latest license)
# 
# HRM ACT V2: Transformer Baseline for Architecture Ablation
#
# This is an architecture ablation of the Hierarchical Reasoning Model (HRM).
# Key changes from V1:
# 1. REMOVED hierarchical split (no separate H and L levels)
# 2. REMOVED inner cycles (no H_cycles/L_cycles loops within reasoning)
# 3. KEPT ACT outer loop structure intact
# 4. KEPT all data preprocessing, embeddings, and evaluation infrastructure
#
# Architecture: Single-level transformer that processes the full 30x30 grid as a
# 900-token sequence, with the same positional encodings and sparse embeddings as V1.
# =============================================================================

from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel

from baselines.trm.models.common import trunc_normal_init_
from baselines.trm.models.layers import (
    rms_norm, SwiGLU, Attention, RotaryEmbedding, CosSin, 
    CastedEmbedding, CastedLinear
)
from baselines.trm.models.sparse_embedding import CastedSparseEmbedding


@dataclass
class Model_ACTV2InnerCarry:
    """Inner carry state for transformer baseline."""
    z_H: torch.Tensor


@dataclass
class Model_ACTV2Carry:
    """Complete carry state for the transformer baseline ACT wrapper."""
    inner_carry: Model_ACTV2InnerCarry
    steps: torch.Tensor
    halted: torch.Tensor
    current_data: Dict[str, torch.Tensor]


class Model_ACTV2Config(BaseModel):
    """Configuration for transformer baseline model."""
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int

    H_cycles: int  # Used for ACT outer loop

    H_layers: int  # Number of transformer layers

    # Transformer config
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    # Halting Q-learning config
    halt_max_steps: int
    halt_exploration_prob: float
    act_enabled: bool = True  # If False, always run halt_max_steps (no early stopping during training)
    act_inference: bool = False  # If True, use adaptive computation during inference

    forward_dtype: str = "bfloat16"
    
    class Config:
        extra = "allow"


class Model_ACTV2Block(nn.Module):
    """Single transformer block for baseline."""
    
    def __init__(self, config: Model_ACTV2Config) -> None:
        super().__init__()

        self.self_attn = Attention(
            hidden_size=config.hidden_size,
            head_dim=config.hidden_size // config.num_heads,
            num_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            causal=False,
        )
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        # Post Norm
        # Self Attention
        hidden_states = rms_norm(
            hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states),
            variance_epsilon=self.norm_eps,
        )
        # Fully Connected
        hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
        return hidden_states


class Model_ACTV2ReasoningModule(nn.Module):
    """Reasoning module with input injection."""
    
    def __init__(self, layers: List[Model_ACTV2Block]):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> torch.Tensor:
        # Input injection (add)
        hidden_states = hidden_states + input_injection
        # Layers
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)
        return hidden_states


class Model_ACTV2_Inner(nn.Module):
    """Inner model for transformer baseline."""
    
    def __init__(self, config: Model_ACTV2Config) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        # I/O
        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            init_std=embed_init_std,
            cast_to=self.forward_dtype,
        )
        self.lm_head = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.q_head = CastedLinear(self.config.hidden_size, 2, bias=True)

        self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size)
        if self.config.puzzle_emb_ndim > 0:
            self.puzzle_emb = CastedSparseEmbedding(
                self.config.num_puzzle_identifiers,
                self.config.puzzle_emb_ndim,
                batch_size=self.config.batch_size,
                init_std=0,
                cast_to=self.forward_dtype,
            )

        # LM Blocks
        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(
                dim=self.config.hidden_size // self.config.num_heads,
                max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                base=self.config.rope_theta,
            )
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(
                self.config.seq_len + self.puzzle_emb_len,
                self.config.hidden_size,
                init_std=embed_init_std,
                cast_to=self.forward_dtype,
            )
        else:
            raise NotImplementedError()

        # Reasoning Layers
        self.H_level = Model_ACTV2ReasoningModule(
            layers=[Model_ACTV2Block(self.config) for _ in range(self.config.H_layers)]
        )

        # Initial states
        self.H_init = nn.Buffer(
            trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1),
            persistent=True,
        )

        # Q head special init
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        """Generate input embeddings."""
        embedding = self.embed_tokens(input.to(torch.int32))

        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)

            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))

            embedding = torch.cat(
                (puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding), 
                dim=-2
            )

        if self.config.pos_encodings == "learned":
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))

        return self.embed_scale * embedding

    def empty_carry(self, batch_size: int):
        return Model_ACTV2InnerCarry(
            z_H=torch.empty(
                batch_size,
                self.config.seq_len + self.puzzle_emb_len,
                self.config.hidden_size,
                dtype=self.forward_dtype,
            ),
        )

    def reset_carry(self, reset_flag: torch.Tensor, carry: Model_ACTV2InnerCarry):
        return Model_ACTV2InnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H),
        )

    def forward(
        self, carry: Model_ACTV2InnerCarry, batch: Dict[str, torch.Tensor]
    ) -> Tuple[Model_ACTV2InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )

        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        # 1-step grad (no inner cycles in baseline)
        z_H = self.H_level(carry.z_H, input_embeddings, **seq_info)

        # LM Outputs
        new_carry = Model_ACTV2InnerCarry(z_H=z_H.detach())
        output = self.lm_head(z_H)[:, self.puzzle_emb_len:]

        # Q head
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])


class Model_ACTV2(nn.Module):
    """ACT wrapper for transformer baseline."""

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = Model_ACTV2Config(**config_dict)
        self.inner = Model_ACTV2_Inner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]

        return Model_ACTV2Carry(
            inner_carry=self.inner.empty_carry(batch_size),
            steps=torch.zeros((batch_size,), dtype=torch.int32),
            halted=torch.ones((batch_size,), dtype=torch.bool),
            current_data={k: torch.empty_like(v) for k, v in batch.items()},
        )

    def forward(
        self,
        carry: Model_ACTV2Carry,
        batch: Dict[str, torch.Tensor],
        compute_target_q: bool = False,
    ) -> Tuple[Model_ACTV2Carry, Dict[str, torch.Tensor]]:
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)

        new_steps = torch.where(carry.halted, 0, carry.steps)

        new_current_data = {
            k: torch.where(carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)), batch[k], v)
            for k, v in carry.current_data.items()
        }

        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(
            new_inner_carry, new_current_data
        )

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits,
        }

        with torch.no_grad():
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps

            halted = is_last_step

            act_enabled = self.config.act_enabled and (
                self.training or self.config.act_inference
            )

            if act_enabled and (self.config.halt_max_steps > 1):
                halted = halted | (q_halt_logits > q_continue_logits)

                if self.training:
                    min_halt_steps = (
                        torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob
                    ) * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
                    halted = halted & (new_steps >= min_halt_steps)

                if compute_target_q:
                    _, _, (next_q_halt_logits, next_q_continue_logits) = self.inner(
                        new_inner_carry, new_current_data
                    )
                    outputs["target_q_continue"] = torch.sigmoid(
                        torch.where(
                            is_last_step,
                            next_q_halt_logits,
                            torch.maximum(next_q_halt_logits, next_q_continue_logits),
                        )
                    )

        return Model_ACTV2Carry(new_inner_carry, new_steps, halted, new_current_data), outputs


# Aliases
TransformerBaseline = Model_ACTV2
TransformerBaselineConfig = Model_ACTV2Config
