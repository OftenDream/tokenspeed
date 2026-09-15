# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""LongCat DSA indexer projections shared by model and backend paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from tokenspeed_kernel.ops.attention.dsa import dsa_interleave_rope
from torch import nn
from transformers import PretrainedConfig

from tokenspeed.runtime.layers.layernorm import RMSNorm
from tokenspeed.runtime.layers.linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
)
from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig
from tokenspeed.runtime.layers.rotary_embedding import get_rope
from tokenspeed.runtime.utils import add_prefix


def normalize_longcat_rope_scaling(
    rope_scaling: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not rope_scaling or "factor" not in rope_scaling:
        return None
    normalized = dict(rope_scaling)
    normalized["rope_type"] = "deepseek_yarn"
    return normalized


@dataclass
class LongCatDSAIndexerOutput:
    query: torch.Tensor
    key: torch.Tensor
    weights: torch.Tensor


class _LongCatDSAIndexerBase(nn.Module):
    """Shared LongCat index-query and rotary projection contract."""

    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        q_lora_rank: int,
        qk_rope_head_dim: int,
        rope_theta: float,
        rope_scaling: dict[str, Any] | None,
        max_position_embeddings: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        self.index_topk = config.index_topk
        self.index_n_heads = config.index_n_heads
        self.index_head_dim = config.index_head_dim
        self.rope_head_dim = int(qk_rope_head_dim)
        self.weights_softmax_scale = self.index_head_dim**-0.5 * (
            self.index_n_heads**-0.5
        )
        if not 0 < self.rope_head_dim <= self.index_head_dim:
            raise ValueError(
                "LongCat Indexer requires 0 < qk_rope_head_dim <= "
                f"index_head_dim, got {self.rope_head_dim} and "
                f"{self.index_head_dim}."
            )

        self.wq_b = ReplicatedLinear(
            q_lora_rank,
            self.index_n_heads * self.index_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("wq_b", prefix),
        )
        self.k_norm = RMSNorm(self.index_head_dim, eps=1e-6)
        self.rotary_emb = get_rope(
            self.rope_head_dim,
            rotary_dim=self.rope_head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=normalize_longcat_rope_scaling(rope_scaling),
            is_neox_style=False,
        )

    def project_key_weights(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def project_key(self, hidden_states: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def project_weights(
        self,
        hidden_states: torch.Tensor,
        *,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        raise NotImplementedError

    def interleave_rope(
        self,
        tensor: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return dsa_interleave_rope(
            tensor,
            positions,
            self.rotary_emb.cos_sin_cache,
            rope_dim=self.rope_head_dim,
        )

    def project_query(self, q_lora: torch.Tensor) -> torch.Tensor:
        return self.wq_b(q_lora)[0].view(-1, self.index_n_heads, self.index_head_dim)

    def finish_projection(
        self,
        index_q: torch.Tensor,
        index_k: torch.Tensor,
        weights: torch.Tensor,
        positions: torch.Tensor,
    ) -> LongCatDSAIndexerOutput:
        q_rope, k_rope = self.rotary_emb(
            positions,
            index_q[..., : self.rope_head_dim],
            index_k[:, None, : self.rope_head_dim],
        )
        index_q[..., : self.rope_head_dim] = q_rope
        index_k[:, : self.rope_head_dim] = k_rope.squeeze(1)
        return LongCatDSAIndexerOutput(index_q, index_k, weights)

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_lora: torch.Tensor,
        positions: torch.Tensor,
    ) -> LongCatDSAIndexerOutput:
        index_k, weights = self.project_key_weights(hidden_states)
        index_q = self.project_query(q_lora)
        return self.finish_projection(index_q, index_k, weights, positions)


class LongCatDSAIndexer(_LongCatDSAIndexerBase):
    """LongCat indexer with BF16 keys and an FP32 score projection."""

    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        q_lora_rank: int,
        qk_rope_head_dim: int,
        rope_theta: float,
        rope_scaling: dict[str, Any] | None,
        max_position_embeddings: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            config=config,
            hidden_size=hidden_size,
            q_lora_rank=q_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            prefix=prefix,
        )
        # The Ascend interleaved-RoPE operator consumes BF16 trigonometric
        # tables. Keep a private buffer instead of mutating get_rope()'s shared
        # module, which would change the legacy paired GPU indexer to BF16 too.
        self.register_buffer(
            "_interleaved_rope_cache",
            self.rotary_emb.cos_sin_cache.to(torch.bfloat16),
            persistent=False,
        )
        self.wk = ReplicatedLinear(
            hidden_size,
            self.index_head_dim,
            bias=False,
            quant_config=None,
            prefix=add_prefix("wk", prefix),
        )
        self.weights_proj = ReplicatedLinear(
            hidden_size,
            self.index_n_heads,
            bias=False,
            params_dtype=torch.float32,
            quant_config=None,
            prefix=add_prefix("weights_proj", prefix),
        )

    def interleave_rope(
        self,
        tensor: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return dsa_interleave_rope(
            tensor,
            positions,
            self._interleaved_rope_cache,
            rope_dim=self.rope_head_dim,
        )

    def project_key_weights(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        index_k = self.project_key(hidden_states)
        weights = self.project_weights(hidden_states, output_dtype=index_k.dtype)
        return index_k, weights

    def project_key(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.k_norm(self.wk(hidden_states)[0])

    def project_weights(
        self,
        hidden_states: torch.Tensor,
        *,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        return self.weights_proj(hidden_states.float())[0].to(output_dtype)


class _PackedLongCatDSAIndexer(_LongCatDSAIndexerBase):
    """Legacy paired-layer indexer retaining its packed checkpoint layout."""

    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        q_lora_rank: int,
        qk_rope_head_dim: int,
        rope_theta: float,
        rope_scaling: dict[str, Any] | None,
        max_position_embeddings: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            config=config,
            hidden_size=hidden_size,
            q_lora_rank=q_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            prefix=prefix,
        )
        self.wk_weights_proj = MergedColumnParallelLinear(
            hidden_size,
            [self.index_head_dim, self.index_n_heads],
            bias=False,
            quant_config=None,
            prefix=add_prefix("wk_weights_proj", prefix),
        )
        self._packed_projection_loaded = False

    def set_packed_projection_loaded(self) -> None:
        self._packed_projection_loaded = True

    def project_key_weights(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self._packed_projection_loaded:
            raise RuntimeError("LongCat Indexer packed projection was not loaded")
        key_weights = self.wk_weights_proj(hidden_states)[0]
        index_k, weights = key_weights.split(
            [self.index_head_dim, self.index_n_heads], dim=-1
        )
        return self.k_norm(index_k), weights


__all__ = ["LongCatDSAIndexer", "LongCatDSAIndexerOutput"]
