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

from __future__ import annotations

from dataclasses import dataclass

import torch
from tokenspeed_kernel.platform import current_platform

from tokenspeed.runtime.configs.model_config import ModelConfig
from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
from tokenspeed.runtime.layers.attention.configs.mla import MLAConfig
from tokenspeed.runtime.utils.server_args import ServerArgs

_INDEX_K_FP8_GROUP_SIZE = 128
_INDEX_K_SCALE_BYTES = torch._utils._element_size(torch.float32)


def dsa_index_k_row_bytes(index_head_dim: int) -> int:
    if index_head_dim <= 0 or index_head_dim % _INDEX_K_FP8_GROUP_SIZE != 0:
        raise ValueError(
            f"DSA index_head_dim must be a positive multiple of {_INDEX_K_FP8_GROUP_SIZE}, got {index_head_dim}"
        )
    return (
        index_head_dim
        + index_head_dim // _INDEX_K_FP8_GROUP_SIZE * _INDEX_K_SCALE_BYTES
    )


@dataclass(kw_only=True)
class DSAConfig(MLAConfig):
    is_dsa = True

    index_topk: int
    index_head_dim: int
    index_n_heads: int
    indexer_layer_ids: frozenset[int] | None = None
    index_kpool: int | None = None
    # Index-cache and selection contracts vary independently of the model
    # family. Existing GPU DSA keeps the packed cache defaults; Lite selects
    # independent selection from checkpoint facts. Index-K uses FP8 on GPU
    # and BF16 on Ascend. Storage placement is
    # resolved separately by AttnConfig from the execution device.
    index_init_tokens: int = 0
    index_local_tokens: int = 0

    def __post_init__(self) -> None:
        if self.index_init_tokens < 0 or self.index_local_tokens < 0:
            raise ValueError("DSA initial/local token counts must be nonnegative")
        if self.index_init_tokens + self.index_local_tokens > self.index_topk:
            raise ValueError("DSA initial/local candidates must fit inside index_topk")

    @classmethod
    def _spec_kwargs(
        cls, server_args: ServerArgs, model_config: ModelConfig, is_draft: bool
    ) -> dict:
        text_config = model_config.hf_text_config
        independent_selection = bool(
            getattr(text_config, "uses_independent_dsa_selection", False)
        )
        return dict(
            **super()._spec_kwargs(server_args, model_config, is_draft),
            index_topk=model_config.index_topk,
            index_head_dim=model_config.index_head_dim,
            index_n_heads=model_config.index_n_heads,
            index_kpool=getattr(model_config, "index_kpool", None),
            indexer_layer_ids=getattr(model_config, "indexer_layer_ids", None),
            uses_independent_index_cache=independent_selection,
            index_init_tokens=getattr(model_config, "index_init_tokens", 0),
            index_local_tokens=getattr(model_config, "index_local_tokens", 0),
        )

    @classmethod
    def generate(
        cls,
        server_args: ServerArgs,
        model_config: ModelConfig,
        is_draft: bool = False,
    ) -> AttnConfig:
        text_config = model_config.hf_text_config
        independent_selection = bool(
            getattr(text_config, "uses_independent_dsa_selection", False)
        )
        if independent_selection and (
            is_draft or server_args.speculative_algorithm is not None
        ):
            raise ValueError("Independent DSA selection does not support MTP")
        config = super().generate(server_args, model_config, is_draft)
        spec = config.component(DSAConfig)
        if spec.uses_independent_index_cache and (
            config.kv_cache_dtype != torch.bfloat16
        ):
            raise ValueError(
                "Independent DSA selection currently requires BF16 KV cache"
            )
        if config.kv_cache_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            platform = current_platform()
            if not (platform.is_blackwell_plus or platform.is_cdna4_plus):
                raise ValueError(
                    "GLM DSA FP8 KV cache currently requires NVIDIA Blackwell "
                    "or AMD CDNA4 sparse attention support; use --kv-cache-dtype "
                    "auto or bfloat16 on this platform, got "
                    f"{server_args.kv_cache_dtype}."
                )
        return config

    def has_indexer(self, layer_id: int) -> bool:
        """Return whether a physical attention layer owns an Index-K plane."""

        return self.indexer_layer_ids is None or layer_id in self.indexer_layer_ids

    def cache_cell_size(self, config: AttnConfig) -> int:
        if self.uses_independent_index_cache:
            element_size = torch._utils._element_size(torch.bfloat16)
            return element_size * (self.kv_lora_rank + self.qk_rope_head_dim) + (
                element_size * self.index_head_dim
                if str(config.device).split(":", 1)[0] == "npu"
                else dsa_index_k_row_bytes(self.index_head_dim)
            )
        index_k_cell_size = dsa_index_k_row_bytes(
            self.index_head_dim,
        )
        return super().cache_cell_size(config) + index_k_cell_size
