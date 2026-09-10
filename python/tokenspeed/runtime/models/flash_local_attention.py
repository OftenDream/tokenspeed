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

"""Physical attention leaves for FLASHLocal checkpoints."""

from __future__ import annotations

import math
from typing import Any, Literal

import torch
from tokenspeed_kernel.ops.activation import sigmoid_mul
from tokenspeed_kernel.ops.attention.mla import (
    mla_normalize_project_query,
    mla_prolog,
    mla_prolog_available,
)
from tokenspeed_kernel.platform import current_platform, pdl_enabled
from torch import nn
from torch.nn import functional as F

from tokenspeed.runtime.layers.layernorm import FusedRMSNorm, RMSNorm
from tokenspeed.runtime.layers.linear import ReplicatedLinear
from tokenspeed.runtime.models.deepseek_v3 import _prepare_mla_kv_b_proj_weights
from tokenspeed.runtime.models.kimi_k3 import KimiLinearMLAAttention
from tokenspeed.runtime.utils import add_prefix, set_weight_attrs
from tokenspeed.runtime.utils.env import global_server_args_dict


def flash_local_prefers_packed_projections() -> bool:
    """Select a physical projection layout once during model construction.

    Ascend uses the packed KDA and separate Weight-NZ MLA leaves; other
    backends keep the existing separate KDA and fused Kimi MLA leaves. The
    checkpoint schema and attention math are unchanged.
    """
    return current_platform().is_npu


class WeightNZReplicatedLinear(ReplicatedLinear):
    """Replicated BF16 linear with optional Ascend Weight-NZ preparation."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        weight_nz: Literal["standard", "transposed"],
        prefix: str,
    ) -> None:
        super().__init__(
            input_size,
            output_size,
            bias=False,
            params_dtype=torch.bfloat16,
            prefix=prefix,
        )
        self.weight_nz = weight_nz
        self._weight_nz_prepared = False
        self._weight_nz_transposed = False

    def process_weights_after_loading(self, _module: nn.Module | None = None) -> None:
        if self._weight_nz_prepared or self.weight.device.type != "npu":
            return
        if (
            not global_server_args_dict.get("npu_enable_weight_nz", False)
            or global_server_args_dict.get("disaggregation_mode") != "decode"
        ):
            return

        import tokenspeed_kernel

        transposed = self.weight_nz == "transposed"
        self.weight.data = tokenspeed_kernel.prepare_weight_nz(
            self.weight.data, transpose=transposed
        )
        self._weight_nz_transposed = transposed
        self._weight_nz_prepared = True

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, None]:
        if self._weight_nz_transposed:
            return torch.matmul(hidden_states, self.weight), None
        if self._weight_nz_prepared:
            return F.linear(hidden_states, self.weight), None
        return super().forward(hidden_states)


class PackedKDAProjection(nn.Module):
    """Pack six separate checkpoint projections into one runtime GEMM weight.

    The component-aware loader writes q/k/v/g/f-a/b-a into fixed row slices;
    ``forward`` performs one projection and returns the same semantic tensors.
    """

    component_count = 6

    def __init__(self, hidden_size: int, local_projection: int, head_dim: int):
        super().__init__()
        self.local_projection = local_projection
        self.head_dim = head_dim
        self._rows = (
            local_projection,
            local_projection,
            local_projection,
            local_projection,
            head_dim,
            head_dim,
        )
        self._offsets = (
            0,
            local_projection,
            2 * local_projection,
            3 * local_projection,
            4 * local_projection,
            4 * local_projection + head_dim,
        )
        self.weight = nn.Parameter(
            torch.empty(sum(self._rows), hidden_size, dtype=torch.bfloat16),
            requires_grad=False,
        )
        set_weight_attrs(self.weight, {"weight_loader": self.weight_loader})

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        component_id: int,
    ) -> None:
        if param is not self.weight:
            raise ValueError("Packed KDA loader received an unrelated parameter.")
        self.load_component(component_id, loaded_weight)

    def load_component(self, component_id: int, local_weight: torch.Tensor) -> None:
        if component_id < 0 or component_id >= self.component_count:
            raise ValueError(f"Invalid packed KDA projection component {component_id}.")
        rows = self._rows[component_id]
        expected = (rows, self.weight.shape[1])
        if (
            tuple(local_weight.shape) != expected
            or local_weight.dtype != self.weight.dtype
        ):
            raise ValueError(
                f"Packed KDA projection component {component_id} has "
                f"shape/dtype {tuple(local_weight.shape)}/{local_weight.dtype}; "
                f"expected {expected}/{self.weight.dtype}."
            )
        if not self.weight.is_meta:
            start = self._offsets[component_id]
            self.weight.data[start : start + rows].copy_(
                local_weight.to(device=self.weight.device)
            )

    def forward(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        projected = F.linear(hidden_states, self.weight)
        projection = self.local_projection
        low_rank = self.head_dim
        return (
            projected[:, : 3 * projection].contiguous(),
            projected[:, 3 * projection : 4 * projection],
            projected[:, 4 * projection : 4 * projection + low_rank],
            projected[:, 4 * projection + low_rank :],
        )


class PackedFLASHLocalKDA(nn.Module):
    """FLASHLocal KDA using the packed Ascend projection layout."""

    def __init__(self, config: Any, mapping: Any, layer_id: int) -> None:
        super().__init__()
        tp = mapping.linear_attn.tp_size
        projection = config.linear_num_heads * config.linear_head_dim
        local_projection = projection // tp
        self.config = config
        self.mapping = mapping
        self.layer_id = layer_id
        self.local_num_heads = config.linear_num_heads // tp
        self.head_dim = config.linear_head_dim
        self.projection = projection
        self.input_projection = PackedKDAProjection(
            config.hidden_size, local_projection, config.linear_head_dim
        )
        self.q_conv1d = nn.Conv1d(
            local_projection,
            local_projection,
            config.linear_conv_size,
            groups=local_projection,
            bias=False,
            dtype=torch.bfloat16,
        )
        self.k_conv1d = nn.Conv1d(
            local_projection,
            local_projection,
            config.linear_conv_size,
            groups=local_projection,
            bias=False,
            dtype=torch.bfloat16,
        )
        self.v_conv1d = nn.Conv1d(
            local_projection,
            local_projection,
            config.linear_conv_size,
            groups=local_projection,
            bias=False,
            dtype=torch.bfloat16,
        )
        self.conv_weights: torch.Tensor | None = None
        self.beta_b_proj = ReplicatedLinear(
            config.linear_head_dim,
            local_projection,
            bias=False,
            params_dtype=torch.bfloat16,
            prefix="beta_b_proj",
        )
        self.f_b_proj = ReplicatedLinear(
            config.linear_head_dim,
            local_projection,
            bias=False,
            params_dtype=torch.bfloat16,
            prefix="f_b_proj",
        )
        self.A_log = nn.Parameter(
            torch.empty(config.linear_num_heads // tp, dtype=torch.float32),
            requires_grad=False,
        )
        self.dt_bias = nn.Parameter(
            torch.empty(local_projection, dtype=torch.float32), requires_grad=False
        )
        self.o_norm = RMSNorm(config.linear_head_dim, eps=config.rms_norm_eps).to(
            dtype=torch.bfloat16
        )
        self.o_proj = WeightNZReplicatedLinear(
            local_projection,
            config.hidden_size,
            weight_nz="standard",
            prefix="o_proj",
        )

    def process_weights_after_loading(self, _module: nn.Module | None = None) -> None:
        if self.conv_weights is not None:
            return
        self.conv_weights = torch.cat(
            [
                conv.weight.squeeze(1)
                for conv in (self.q_conv1d, self.k_conv1d, self.v_conv1d)
            ],
            dim=0,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: Any,
        comm_manager: Any = None,
        **_kwargs: Any,
    ) -> torch.Tensor:
        del positions, comm_manager
        if hidden_states.shape[0] == 0:
            return hidden_states

        qkv, output_gate, f_a_out, beta_down = self.input_projection(hidden_states)
        beta_logits = F.linear(beta_down, self.beta_b_proj.weight)
        conv_weights = self.conv_weights
        if conv_weights is None:
            self.process_weights_after_loading()
            conv_weights = self.conv_weights
        assert conv_weights is not None

        core_output = ctx.attn_backend.forward(
            q=None,
            k=None,
            v=None,
            layer=None,
            token_to_kv_pool=ctx.token_to_kv_pool,
            forward_mode=ctx.forward_mode,
            bs=ctx.bs,
            mixed_qkv=qkv,
            conv_weights=conv_weights,
            bias=None,
            activation="silu",
            key_dim=self.projection,
            value_dim=self.projection,
            attention_tp_size=self.mapping.linear_attn.tp_size,
            head_k_dim=self.head_dim,
            head_v_dim=self.head_dim,
            f_a_out=f_a_out,
            f_b_weight=self.f_b_proj.weight,
            beta_raw=beta_logits,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            lower_bound=self.config.linear_attn_config["gate_lower_bound"],
            output_gate=None,
            norm_weight=None,
            norm_eps=None,
            layer_id=self.layer_id,
            seq_len=hidden_states.shape[0],
        ).reshape(-1, self.local_num_heads, self.head_dim)

        if core_output.device.type in ("cuda", "npu"):
            from tokenspeed_kernel.ops.activation.triton import (
                rmsnorm_gated_sigmoid,
            )

            gated = rmsnorm_gated_sigmoid(
                core_output.flatten(1).contiguous(),
                output_gate,
                self.o_norm.weight,
                self.config.rms_norm_eps,
                self.local_num_heads,
                self.head_dim,
                enable_pdl=pdl_enabled(),
            )
        else:
            core_fp32 = core_output.float()
            normalized = core_fp32 * torch.rsqrt(
                core_fp32.square().mean(dim=-1, keepdim=True) + self.config.rms_norm_eps
            )
            normalized = (normalized * self.o_norm.weight.float()).to(core_output.dtype)
            output_gate = output_gate.reshape_as(normalized)
            gated = normalized * torch.sigmoid(output_gate.float()).to(normalized.dtype)
            gated = gated.flatten(1)
        return self.o_proj(gated)[0]


class SeparateProjectionKimiLinearMLAAttention(KimiLinearMLAAttention):
    """Shared Kimi MLA dataflow with separate Weight-NZ-capable projections.

    Here ``separate`` describes the Ascend runtime parameters: q-a, kv-a, and
    output-gate weights remain distinct instead of using Kimi's fused input
    parameter. All cache, Prefill, absorbed-Decode, and output-gate math stays in
    the inherited implementation.
    """

    def __init__(
        self,
        config: Any,
        mapping: Any,
        *,
        layer_id: int,
        prefix: str = "",
    ) -> None:
        component_mapping = mapping.mla_weight
        super().__init__(
            config=config,
            mapping=mapping,
            component_mapping=component_mapping,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            q_lora_rank=config.q_lora_rank,
            kv_lora_rank=config.kv_lora_rank,
            max_position_embeddings=config.max_position_embeddings,
            layer_id=layer_id,
            prefix=prefix,
            reduce_attn_results=False,
        )
        del self.fused_qkv_a_proj_with_mqa
        qk_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.q_a_layernorm = RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps).to(
            dtype=torch.bfloat16
        )
        self.kv_a_layernorm = RMSNorm(config.kv_lora_rank, eps=config.rms_norm_eps).to(
            dtype=torch.bfloat16
        )
        self.fused_qk_layernorm = FusedRMSNorm(self.q_a_layernorm, self.kv_a_layernorm)
        self.q_a_proj = WeightNZReplicatedLinear(
            config.hidden_size,
            config.q_lora_rank,
            weight_nz="transposed",
            prefix=add_prefix("q_a_proj", prefix),
        )
        self.kv_a_proj_with_mqa = WeightNZReplicatedLinear(
            config.hidden_size,
            config.kv_lora_rank + config.qk_rope_head_dim,
            weight_nz="transposed",
            prefix=add_prefix("kv_a_proj_with_mqa", prefix),
        )
        if self.use_output_gate:
            self.g_proj = ReplicatedLinear(
                config.hidden_size,
                config.num_attention_heads * config.v_head_dim,
                bias=False,
                params_dtype=torch.bfloat16,
                prefix=add_prefix("g_proj", prefix),
            )
        self.q_b_proj = WeightNZReplicatedLinear(
            config.q_lora_rank,
            config.num_attention_heads * qk_dim,
            # The prolog consumes logical [in, out] NZ weights. Preparation
            # happens once on load under the existing Decode Weight-NZ switch.
            weight_nz="transposed",
            prefix=add_prefix("q_b_proj", prefix),
        )
        self.kv_b_proj = ReplicatedLinear(
            config.kv_lora_rank,
            config.num_attention_heads * (config.qk_nope_head_dim + config.v_head_dim),
            bias=False,
            params_dtype=torch.bfloat16,
            prefix=add_prefix("kv_b_proj", prefix),
        )
        self.o_proj = WeightNZReplicatedLinear(
            config.num_attention_heads * config.v_head_dim,
            config.hidden_size,
            weight_nz="standard",
            prefix=add_prefix("o_proj", prefix),
        )
        self._mla_scales_prepared = False

    def process_weights_after_loading(self, _module: nn.Module | None = None) -> None:
        if not self._mla_scales_prepared:
            if self.config.mla_scale_q_lora:
                self.q_a_layernorm.weight.data.mul_(
                    math.sqrt(self.config.hidden_size / self.config.q_lora_rank)
                )
            if self.config.mla_scale_kv_lora:
                self.kv_a_layernorm.weight.data.mul_(
                    math.sqrt(self.config.hidden_size / self.config.kv_lora_rank)
                )
            self._mla_scales_prepared = True
        if self.w_kc is None:
            self.w_kc, self.w_vc = _prepare_mla_kv_b_proj_weights(
                self.kv_b_proj.weight, self
            )

    def _project_q_latent_gated(
        self,
        hidden_states: torch.Tensor,
        ctx: Any,
        comm_manager: Any,
        block_scale: torch.Tensor | None,
        attnres_partial_args: tuple | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, None]:
        if block_scale is not None or attnres_partial_args is not None:
            raise ValueError(
                "Separate FLASHLocal MLA projections do not accept quantized "
                "block scales or AttnRes partials."
            )
        q_a = self.q_a_proj(hidden_states)[0]
        latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]
        gate = self.g_proj(hidden_states)[0] if self.use_output_gate else None
        if comm_manager is not None:
            pieces = [q_a, latent_cache]
            if gate is not None:
                pieces.append(gate)
            widths = [piece.shape[-1] for piece in pieces]
            projected = comm_manager.pre_attn_comm(torch.cat(pieces, dim=-1), ctx)
            pieces = projected.split(widths, dim=-1)
            q_a, latent_cache = pieces[:2]
            if gate is not None:
                gate = pieces[2]
        kv_a = latent_cache[..., : self.kv_lora_rank]
        if self.q_b_proj._weight_nz_transposed:
            # Primitive fallback for the same [in, out] weight orientation;
            # this is not the fused prolog (which bypasses this method).
            q_norm = torch.empty_like(q_a)
            if q_a.size(0) > 0:
                self.fused_qk_layernorm(
                    input_q_a=q_a, input_kv_a=kv_a, output_q_a=q_norm
                )
            return self.q_b_proj(q_norm)[0], latent_cache, gate, None
        query, _ = mla_normalize_project_query(
            q_a,
            kv_a,
            self.q_a_layernorm.weight,
            self.kv_a_layernorm.weight,
            self.q_b_proj.weight,
            eps=self.q_a_layernorm.variance_epsilon,
            prepare_absorbed_query=False,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
        )
        return query, latent_cache, gate, None

    def _project_q_latent(
        self,
        hidden_states: torch.Tensor,
        ctx: Any,
        comm_manager: Any,
        block_scale: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query, latent_cache, _, _ = self._project_q_latent_gated(
            hidden_states, ctx, comm_manager, block_scale, None
        )
        return query, latent_cache

    def _mla_prolog_inputs(
        self,
        hidden_states: torch.Tensor,
        ctx: Any,
        comm_manager: Any,
        block_scale: torch.Tensor | None,
        attnres_partial_args: tuple | None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if (
            block_scale is not None
            or attnres_partial_args is not None
            or hidden_states.ndim != 2
            or not ctx.forward_mode.is_decode()
            or ctx.bs == 0
            or ctx.num_extends != 0
            or ctx.input_num_tokens != ctx.bs
            or hidden_states.shape[0] != ctx.bs
            or (ctx.attn_backend.spec_num_tokens or 1) != 1
            or self.component_mapping.tp_size != 1
            or self.attention_backend not in self._MLA_KERNEL_BACKENDS
            or self.w_kc is None
            or not self.q_a_proj._weight_nz_transposed
            or not self.kv_a_proj_with_mqa._weight_nz_transposed
            or not self.q_b_proj._weight_nz_transposed
        ):
            return None
        # Fusing projection with cache writes cannot bypass a token all-gather.
        # Keep the existing projection/communication path when rows are scattered.
        if (
            comm_manager is not None
            and comm_manager.layer_id != 0
            and comm_manager.attn_mapping.has_tp
            and not comm_manager.use_all_reduce(comm_manager.prev_is_moe)
        ):
            return None
        pool = ctx.token_to_kv_pool
        if getattr(pool, "quant_method", None) == "per_token_head":
            return None
        cache = pool.get_key_buffer(self.attn_mqa.layer_id)
        page_size = pool.arena.kv_page_size
        cache_width = self.kv_lora_rank + self.qk_rope_head_dim
        # Only construct a view of backend-owned storage here. Hardware shape,
        # dtype and NZ-format eligibility belong to the kernel adapter.
        if (
            not isinstance(cache, torch.Tensor)
            or not cache.is_contiguous()
            or page_size <= 0
            or cache.numel() % (page_size * cache_width) != 0
            or not mla_prolog_available()
        ):
            return None
        selected = ctx.attn_backend.write_locations(self.attn_mqa, ctx.forward_mode)
        if selected.numel() != ctx.bs:
            return None
        return cache.view(-1, page_size, 1, cache_width), selected

    def _project_q_with_mla_prolog(
        self,
        hidden_states: torch.Tensor,
        cache: torch.Tensor,
        cache_index: torch.Tensor,
    ) -> torch.Tensor | None:
        """Project the absorbed query and write KV cache, or decline before writes."""
        result = mla_prolog(
            hidden_states,
            self.q_a_proj.weight,
            self.q_b_proj.weight,
            self.w_kc,
            self.kv_a_proj_with_mqa.weight,
            self.q_a_layernorm.weight,
            self.kv_a_layernorm.weight,
            cache,
            cache_index,
            rmsnorm_epsilon_cq=self.q_a_layernorm.variance_epsilon,
            rmsnorm_epsilon_ckv=self.kv_a_layernorm.variance_epsilon,
        )
        if result is None:
            return None
        query_nope, query_rope = result
        return torch.cat((query_nope, query_rope), dim=-1)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: Any,
        comm_manager: Any = None,
        block_scale: torch.Tensor | None = None,
        attnres_partial_args: tuple | None = None,
    ) -> torch.Tensor:
        prolog_inputs = self._mla_prolog_inputs(
            hidden_states, ctx, comm_manager, block_scale, attnres_partial_args
        )
        if prolog_inputs is not None:
            query = self._project_q_with_mla_prolog(hidden_states, *prolog_inputs)
            if query is not None:
                # Prolog has written KV; attention and output projection remain
                # explicit model stages, outside the projection helper.
                _, cache_index = prolog_inputs
                gate = self.g_proj(hidden_states)[0] if self.use_output_gate else None
                fuse_value_gate = ctx.attn_backend.supports_mla_projected_value_decode
                attn_output = hidden_states.new_empty(
                    (hidden_states.shape[0], self.num_heads * self.v_head_dim)
                )
                self.forward_absorb_attn_v_proj(
                    query,
                    None,
                    ctx,
                    cache_index,
                    attn_output,
                    record_kv_cache=None,
                    output_gate=gate if fuse_value_gate else None,
                )
                if gate is not None and not fuse_value_gate:
                    attn_output = sigmoid_mul(attn_output, gate)
                return self.o_proj(attn_output)[0]
        return super().forward(
            positions,
            hidden_states,
            ctx,
            comm_manager,
            block_scale,
            attnres_partial_args,
        )


__all__ = [
    "PackedFLASHLocalKDA",
    "PackedKDAProjection",
    "SeparateProjectionKimiLinearMLAAttention",
    "WeightNZReplicatedLinear",
    "flash_local_prefers_packed_projections",
]
