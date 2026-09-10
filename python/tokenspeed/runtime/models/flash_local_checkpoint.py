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

"""Strict source checkpoint ledger for the shared Flash-Lite model."""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

import torch

from tokenspeed.runtime.configs.flash_kda_config import FLASHLocalConfig

_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")
_EXPERT_RE = re.compile(
    r"^mlp\.experts\.(\d+)\.(gate|up|down)_proj\."
    r"(weight|weight_scale|smooth_scale)$"
)
_ROUTER_RE = re.compile(
    r"^mlp\.expert_groups\.(\d+)\.router\."
    r"(classifier\.weight|e_score_correction_bias)$"
)
_OE_RE = re.compile(r"^model\.ngram_embeddings\.(embedders|post_projs)\.(\d+)\.weight$")
_Parallel = Literal["dense", "linear"]


@dataclass(frozen=True)
class FLASHLocalWeightSpec:
    source_name: str
    target_name: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    category: str
    parallel: _Parallel | None = None
    shard_axis: int | None = None
    expert_id: int | None = None
    component_id: int | None = None


class FLASHLocalCheckpointLayout:
    """Single source of truth for Lite source keys, shapes and placement."""

    def __init__(
        self,
        config: FLASHLocalConfig,
        *,
        moe_quant_kind: str = "unquant",
        moe_smooth_quant: bool = False,
    ) -> None:
        self.config = config
        self.moe_quant_kind = moe_quant_kind
        self.moe_smooth_quant = moe_smooth_quant

    def iter_source_names(self) -> Iterator[str]:
        yield "model.embed_tokens.weight"
        yield "lm_head.weight"
        yield "model.norm.weight"
        for layer_id in range(self.config.num_hidden_layers):
            prefix = f"model.layers.{layer_id}"
            yield f"{prefix}.input_layernorm.weight"
            yield f"{prefix}.post_attention_layernorm.weight"
            for group_id in range(self.config.moe_group_size):
                router = f"{prefix}.mlp.expert_groups.{group_id}.router"
                yield f"{router}.classifier.weight"
                yield f"{router}.e_score_correction_bias"
            for expert_id in range(
                self.config.n_routed_experts * self.config.moe_group_size
            ):
                expert = f"{prefix}.mlp.experts.{expert_id}"
                yield f"{expert}.gate_proj.weight"
                yield f"{expert}.up_proj.weight"
                yield f"{expert}.down_proj.weight"
                if self.moe_quant_kind == "int8":
                    for projection in ("gate_proj", "up_proj", "down_proj"):
                        yield f"{expert}.{projection}.weight_scale"
                        if self.moe_smooth_quant:
                            yield f"{expert}.{projection}.smooth_scale"
            yield f"{prefix}.mlp.norm.weight"
            yield f"{prefix}.mlp.proj_input.weight"
            yield f"{prefix}.mlp.proj_output.weight"
            for projection in ("gate_proj", "up_proj", "down_proj"):
                yield f"{prefix}.mlp.shared_experts.{projection}.weight"

            attention = f"{prefix}.self_attn"
            if self.config.is_kda_layer(layer_id):
                core = f"{attention}.linear_core"
                for projection in ("q_proj", "k_proj", "v_proj", "g_proj.0"):
                    yield f"{core}.{projection}.weight"
                for projection in ("q_conv1d", "k_conv1d", "v_conv1d"):
                    yield f"{core}.{projection}.weight"
                for projection in ("b_proj", "f_proj"):
                    yield f"{core}.{projection}.0.weight"
                    yield f"{core}.{projection}.1.weight"
                yield f"{core}.A_log"
                yield f"{core}.dt_bias"
                yield f"{core}.o_norm.weight"
                yield f"{core}.o_proj.weight"
            else:
                if self.config.mla_use_output_gate:
                    yield f"{attention}.g_proj.weight"
                for suffix in (
                    "q_a_proj.weight",
                    "q_a_layernorm.weight",
                    "q_b_proj.weight",
                    "kv_a_proj_with_mqa.weight",
                    "kv_a_layernorm.weight",
                    "kv_b_proj.weight",
                    "o_proj.weight",
                ):
                    yield f"{attention}.{suffix}"

        for table_id in range(self.config.oe_component_count):
            yield f"model.ngram_embeddings.embedders.{table_id}.weight"
            yield f"model.ngram_embeddings.post_projs.{table_id}.weight"

    def spec(self, name: str) -> FLASHLocalWeightSpec:
        config = self.config
        if name in {"model.embed_tokens.weight", "lm_head.weight"}:
            return FLASHLocalWeightSpec(
                name,
                name,
                (config.vocab_size, config.hidden_size),
                torch.bfloat16,
                "dense-shard",
                "dense",
                0,
            )
        if name == "model.norm.weight":
            return self._replicated(name, (config.hidden_size,))

        oe_match = _OE_RE.fullmatch(name)
        if oe_match:
            family, raw_id = oe_match.groups()
            table_id = int(raw_id)
            if table_id >= config.oe_component_count:
                raise ValueError(f"Unexpected Lite checkpoint weight {name!r}.")
            shape = (
                (config.oe_table_rows(table_id), config.oe_hidden_size)
                if family == "embedders"
                else (config.hidden_size, config.oe_hidden_size)
            )
            return FLASHLocalWeightSpec(
                name,
                (
                    name
                    if family == "embedders"
                    else "model.ngram_embeddings.projection"
                ),
                shape,
                torch.bfloat16,
                "host-oe" if family == "embedders" else "oe-projection",
                component_id=table_id,
            )

        layer_match = _LAYER_RE.fullmatch(name)
        if layer_match is None:
            raise ValueError(f"Unexpected Lite checkpoint weight {name!r}.")
        layer_id = int(layer_match.group(1))
        suffix = layer_match.group(2)
        if layer_id >= config.num_hidden_layers:
            raise ValueError(f"Unexpected Lite checkpoint weight {name!r}.")

        if suffix in {
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "mlp.norm.weight",
        }:
            return self._replicated(name, (config.hidden_size,))

        router_match = _ROUTER_RE.fullmatch(suffix)
        if router_match:
            group_id, leaf = router_match.groups()
            if int(group_id) >= config.moe_group_size:
                raise ValueError(f"Unexpected Lite checkpoint weight {name!r}.")
            experts = config.n_routed_experts + config.zero_expert_num
            shape = (
                (experts, config.hidden_size // config.moe_group_size)
                if leaf == "classifier.weight"
                else (experts,)
            )
            return FLASHLocalWeightSpec(name, name, shape, torch.float32, "router")

        expert_match = _EXPERT_RE.fullmatch(suffix)
        if expert_match:
            expert_id = int(expert_match.group(1))
            projection = expert_match.group(2)
            tensor_kind = expert_match.group(3)
            if expert_id >= config.n_routed_experts * config.moe_group_size:
                raise ValueError(f"Unexpected Lite checkpoint weight {name!r}.")
            group_hidden = config.hidden_size // config.moe_group_size
            if tensor_kind == "weight":
                shape = (
                    (group_hidden, config.expert_ffn_hidden_size)
                    if projection == "down"
                    else (config.expert_ffn_hidden_size, group_hidden)
                )
                dtype = torch.int8 if self.moe_quant_kind == "int8" else torch.bfloat16
            elif self.moe_quant_kind != "int8":
                raise ValueError(f"Unexpected Lite checkpoint weight {name!r}.")
            elif tensor_kind == "weight_scale":
                shape = (
                    (group_hidden, 1)
                    if projection == "down"
                    else (config.expert_ffn_hidden_size, 1)
                )
                dtype = torch.bfloat16
            else:
                if not self.moe_smooth_quant:
                    raise ValueError(f"Unexpected Lite checkpoint weight {name!r}.")
                shape = (
                    (config.expert_ffn_hidden_size,)
                    if projection == "down"
                    else (group_hidden,)
                )
                dtype = torch.bfloat16
            packed_prefix = "w2_" if projection == "down" else "w13_"
            packed_name = packed_prefix + tensor_kind
            target = name.rsplit(".experts.", 1)[0] + f".experts.{packed_name}"
            return FLASHLocalWeightSpec(
                name,
                target,
                shape,
                dtype,
                "expert-ep",
                expert_id=expert_id,
            )

        dense_shapes = {
            "mlp.proj_input.weight": (config.hidden_size, config.hidden_size),
            "mlp.proj_output.weight": (config.hidden_size, config.hidden_size),
            "mlp.shared_experts.gate_proj.weight": (
                config.ffn_hidden_size,
                config.hidden_size,
            ),
            "mlp.shared_experts.up_proj.weight": (
                config.ffn_hidden_size,
                config.hidden_size,
            ),
            "mlp.shared_experts.down_proj.weight": (
                config.hidden_size,
                config.ffn_hidden_size,
            ),
        }
        if suffix in dense_shapes:
            shard_axis = (
                1 if suffix.endswith(("proj_output.weight", "down_proj.weight")) else 0
            )
            return FLASHLocalWeightSpec(
                name,
                name,
                dense_shapes[suffix],
                torch.bfloat16,
                "dense-shard",
                "dense",
                shard_axis,
            )

        attention_prefix = "self_attn."
        if not suffix.startswith(attention_prefix):
            raise ValueError(f"Unexpected Lite checkpoint weight {name!r}.")
        attention_suffix = suffix[len(attention_prefix) :]
        if config.is_kda_layer(layer_id):
            return self._kda_spec(name, attention_suffix)
        return self._mla_spec(name, attention_suffix)

    def _kda_spec(self, name: str, suffix: str) -> FLASHLocalWeightSpec:
        config = self.config
        core = "linear_core."
        if not suffix.startswith(core):
            raise ValueError(f"Unexpected Lite KDA weight {name!r}.")
        leaf = suffix[len(core) :]
        projection = config.linear_num_heads * config.linear_head_dim
        shapes = {
            "q_proj.weight": (projection, config.hidden_size),
            "k_proj.weight": (projection, config.hidden_size),
            "v_proj.weight": (projection, config.hidden_size),
            "g_proj.0.weight": (projection, config.hidden_size),
            "q_conv1d.weight": (projection, 1, config.linear_conv_size),
            "k_conv1d.weight": (projection, 1, config.linear_conv_size),
            "v_conv1d.weight": (projection, 1, config.linear_conv_size),
            "b_proj.0.weight": (config.linear_head_dim, config.hidden_size),
            "b_proj.1.weight": (projection, config.linear_head_dim),
            "f_proj.0.weight": (config.linear_head_dim, config.hidden_size),
            "f_proj.1.weight": (projection, config.linear_head_dim),
            "A_log": (config.linear_num_heads,),
            "dt_bias": (projection,),
            "o_norm.weight": (config.linear_head_dim,),
            "o_proj.weight": (config.hidden_size, projection),
        }
        if leaf not in shapes:
            raise ValueError(f"Unexpected Lite KDA weight {name!r}.")
        packed_components = {
            "q_proj.weight": 0,
            "k_proj.weight": 1,
            "v_proj.weight": 2,
            "g_proj.0.weight": 3,
            "f_proj.0.weight": 4,
            "b_proj.0.weight": 5,
        }
        if leaf in packed_components:
            target = name.replace(f"{core}{leaf}", "input_projection.weight")
            replicated = leaf in {"f_proj.0.weight", "b_proj.0.weight"}
            return FLASHLocalWeightSpec(
                name,
                target,
                shapes[leaf],
                torch.bfloat16,
                "kda-packed-projection",
                None if replicated else "linear",
                None if replicated else 0,
                component_id=packed_components[leaf],
            )
        target_leaf = {
            "b_proj.1.weight": "beta_b_proj.weight",
            "f_proj.1.weight": "f_b_proj.weight",
        }.get(leaf, leaf)
        target = name.replace(f"{core}{leaf}", target_leaf)
        replicated = {"o_norm.weight"}
        shard_axis = 1 if leaf == "o_proj.weight" else 0
        return FLASHLocalWeightSpec(
            name,
            target,
            shapes[leaf],
            torch.float32 if leaf in {"A_log", "dt_bias"} else torch.bfloat16,
            "rename" if target != name else "linear-shard",
            None if leaf in replicated else "linear",
            None if leaf in replicated else shard_axis,
        )

    def _mla_spec(self, name: str, suffix: str) -> FLASHLocalWeightSpec:
        config = self.config
        qk_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        shapes = {
            "g_proj.weight": (
                config.num_attention_heads * config.v_head_dim,
                config.hidden_size,
            ),
            "q_a_proj.weight": (config.q_lora_rank, config.hidden_size),
            "q_a_layernorm.weight": (config.q_lora_rank,),
            "q_b_proj.weight": (
                config.num_attention_heads * qk_dim,
                config.q_lora_rank,
            ),
            "kv_a_proj_with_mqa.weight": (
                config.kv_lora_rank + config.qk_rope_head_dim,
                config.hidden_size,
            ),
            "kv_a_layernorm.weight": (config.kv_lora_rank,),
            "kv_b_proj.weight": (
                config.num_attention_heads
                * (config.qk_nope_head_dim + config.v_head_dim),
                config.kv_lora_rank,
            ),
            "o_proj.weight": (
                config.hidden_size,
                config.num_attention_heads * config.v_head_dim,
            ),
        }
        if not config.mla_use_output_gate:
            shapes.pop("g_proj.weight")
        if suffix not in shapes:
            raise ValueError(f"Unexpected Lite MLA weight {name!r}.")
        return self._replicated(name, shapes[suffix])

    @staticmethod
    def _replicated(name: str, shape: tuple[int, ...]) -> FLASHLocalWeightSpec:
        return FLASHLocalWeightSpec(name, name, shape, torch.bfloat16, "replicated")
