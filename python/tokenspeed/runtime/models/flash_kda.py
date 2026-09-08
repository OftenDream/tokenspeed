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

import re
from collections.abc import Callable, Iterable

import torch
from tokenspeed_kernel.ops.activation.triton import rmsnorm_gated_sigmoid
from tokenspeed_kernel.ops.moe.cuda import (
    moe_finalize_fuse_shared as _moe_finalize_fuse_shared,
)
from tokenspeed_kernel.ops.moe.triton.group_moe import (
    group_moe_proj_output as _group_moe_proj_output,
)
from tokenspeed_kernel.ops.moe.triton.group_moe import (
    group_moe_router as _group_moe_router,
)
from tokenspeed_kernel.ops.moe.triton.group_moe import rmsnorm_scale as _rmsnorm_scale
from tokenspeed_kernel.platform import pdl_enabled as _pdl_enabled
from torch import nn
from transformers import PretrainedConfig as _PretrainedConfig

from tokenspeed.runtime.configs.flash_kda_config import FLASHLocalConfig
from tokenspeed.runtime.configs.utils import get_rope_theta as _get_rope_theta
from tokenspeed.runtime.distributed.comm_manager import CommManager as _CommManager
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.execution.forward_step import (
    get_is_capture_mode as _get_is_capture_mode,
)
from tokenspeed.runtime.layers.layernorm import RMSNorm
from tokenspeed.runtime.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from tokenspeed.runtime.layers.logits_processor import LogitsProcessor
from tokenspeed.runtime.layers.moe import (
    ExpertCheckpointSchema,
    build_moe_checkpoint_loader,
)
from tokenspeed.runtime.layers.moe.expert import MoELayer
from tokenspeed.runtime.layers.moe.topk import StandardTopKOutput
from tokenspeed.runtime.layers.moe.utils import RoutingMethodType
from tokenspeed.runtime.layers.over_embedding import (
    HostLongCatOverEmbedding,
    LongCatOverEmbedding,
    resolve_longcat_oe_hyperparameters,
)
from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig
from tokenspeed.runtime.layers.utils import get_layer_id as _get_layer_id
from tokenspeed.runtime.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from tokenspeed.runtime.model_loader.weight_utils import (
    default_weight_loader,
    sharded_weight_loader,
)
from tokenspeed.runtime.models.base.causal_lm import BaseCausalLM
from tokenspeed.runtime.models.deepseek_v3 import (
    DeepseekV3MLP,
    _prepare_mla_kv_b_proj_weights,
)
from tokenspeed.runtime.models.flash_local_attention import (
    PackedFLASHLocalKDA,
    SeparateProjectionKimiLinearMLAAttention,
    flash_local_prefers_packed_projections,
)
from tokenspeed.runtime.models.flash_local_checkpoint import (
    FLASHLocalCheckpointLayout,
)
from tokenspeed.runtime.models.flash_local_moe import (
    PackedFLASHLocalMoE,
    flash_local_prefers_packed_moe,
)
from tokenspeed.runtime.models.flash_local_moe import (
    grouped_moe_topk as _grouped_moe_topk_local,
)
from tokenspeed.runtime.models.kimi_k3 import (
    KimiLinearMLAAttention,
)
from tokenspeed.runtime.moe.distribution_recorder import (
    get_global_expert_distribution_recorder,
)
from tokenspeed.runtime.moe.expert_location import (
    ModelConfigForExpertLocation as _ModelConfigForExpertLocation,
)
from tokenspeed.runtime.utils import LazyValue, add_prefix, get_colorful_logger
from tokenspeed.runtime.utils.cuda_stream import StreamFork as _StreamFork
from tokenspeed.runtime.utils.env import global_server_args_dict

logger = get_colorful_logger(__name__)
_OPTIONAL_MISSING_WEIGHT_SUFFIXES = (
    ".k_scale",
    ".v_scale",
)


def _canonical_flash_kda_weight_name(name: str) -> str:
    """Map Megatron / legacy Flash-KDA checkpoint names to runtime names.

    A checkpoint key may need both a top-level Megatron rewrite and a nested
    Flash-KDA rewrite, so all matching replacements are applied in sequence.
    """

    replacements = [
        (
            "language_model.embedding.word_embeddings.weight",
            "model.embed_tokens.weight",
        ),
        ("language_model.output_layer.weight", "lm_head.weight"),
        ("language_model.decoder.final_layernorm.weight", "model.norm.weight"),
        ("language_model.encoder.final_layernorm.weight", "model.norm.weight"),
        ("language_model.decoder.layers.", "model.layers."),
        ("language_model.encoder.layers.", "model.layers."),
        (".input_layernorm.m.", ".input_layernorm."),
        (".post_attention_layernorm.m.", ".post_attention_layernorm."),
        (".self_attention.", ".self_attn."),
        (".mlp.", ".moe."),
        (".self_attn.linear_core.f_proj.0.", ".self_attn.f_proj.fc1."),
        (".self_attn.linear_core.f_proj.1.", ".self_attn.f_proj.fc2."),
        (".self_attn.linear_core.g_proj.0.", ".self_attn.g_proj."),
        (".self_attn.linear_core.g_proj.1.", ".self_attn.g_proj.fc2."),
        (".self_attn.linear_core.b_proj.0.", ".self_attn.b_proj.fc1."),
        (".self_attn.linear_core.b_proj.1.", ".self_attn.b_proj.fc2."),
        (".self_attn.f_proj.0.", ".self_attn.f_proj.fc1."),
        (".self_attn.f_proj.1.", ".self_attn.f_proj.fc2."),
        (".self_attn.g_proj.0.", ".self_attn.g_proj."),
        (".self_attn.g_proj.1.", ".self_attn.g_proj.fc2."),
        (".self_attn.b_proj.0.", ".self_attn.b_proj.fc1."),
        (".self_attn.b_proj.1.", ".self_attn.b_proj.fc2."),
        (".self_attn.linear_core.q_proj.", ".self_attn.q_proj."),
        (".self_attn.linear_core.k_proj.", ".self_attn.k_proj."),
        (".self_attn.linear_core.v_proj.", ".self_attn.v_proj."),
        (".self_attn.linear_core.o_norm.", ".self_attn.o_norm."),
        (".self_attn.linear_core.o_proj.", ".self_attn.o_proj."),
        (".self_attn.linear_core.A_log", ".self_attn.A_log"),
        (".self_attn.linear_core.dt_bias", ".self_attn.dt_bias"),
        (".self_attn.linear_core.q_conv1d.", ".self_attn.q_conv1d."),
        (".self_attn.linear_core.k_conv1d.", ".self_attn.k_conv1d."),
        (".self_attn.linear_core.v_conv1d.", ".self_attn.v_conv1d."),
        (".self_attn.linear_core.b_proj.", ".self_attn.b_proj."),
        (".self_attn.linear_core.f_proj.", ".self_attn.f_proj."),
        (".self_attn.linear_core.g_proj.", ".self_attn.g_proj."),
    ]
    for src, dst in replacements:
        name = name.replace(src, dst)
    return name


def _get_shared_expert_intermediate_size(config) -> int:
    """Return the checkpoint's dense shared-expert width."""

    dense_width = getattr(config, "ffn_hidden_size", None)
    if dense_width is None:
        dense_width = config.intermediate_size
    return int(dense_width) * int(config.n_shared_experts)


def _validate_shared_expert_weight_shape(
    config, name: str, loaded_weight: torch.Tensor
) -> None:
    """Reject silent truncation of Flash-Lite shared-expert weights."""

    if ".shared_experts." not in name or not name.endswith(".weight"):
        return
    shared_width = _get_shared_expert_intermediate_size(config)
    if name.endswith((".gate_proj.weight", ".up_proj.weight")):
        expected = (shared_width, config.hidden_size)
    elif name.endswith(".down_proj.weight"):
        expected = (config.hidden_size, shared_width)
    else:
        return
    if tuple(loaded_weight.shape) != expected:
        raise ValueError(
            f"Flash-KDA shared-expert weight {name!r} has shape "
            f"{tuple(loaded_weight.shape)}, expected {expected}."
        )


def _reshape_fgbkda_beta_logits(
    beta_channel_logits: torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Restore FGBKDA's channel gate to ``[..., heads, head_dim]``."""

    expected_width = num_heads * head_dim
    if beta_channel_logits.shape[-1] != expected_width:
        raise ValueError(
            "FGBKDA beta projection width mismatch: "
            f"got {beta_channel_logits.shape[-1]}, expected {expected_width}"
        )
    return beta_channel_logits.reshape(
        *beta_channel_logits.shape[:-1], num_heads, head_dim
    )


def _parse_flash_kda_expert_group_router_name(name: str) -> tuple[int, int, str] | None:
    """Parse ``...moe.expert_groups.{g}.router.{field}`` checkpoint names.

    Returns ``(layer_id, group_id, field)`` for the per-group router tensors
    that the Flash-KDA group-MoE checkpoint stores separately, or ``None`` for
    unrelated names.
    """

    marker = ".moe.expert_groups."
    if marker not in name or ".router." not in name:
        return None
    try:
        layer_id = _get_layer_id(name)
        if layer_id is None:
            return None
        tail = name.split(marker, 1)[1]
        group_str, tail = tail.split(".router.", 1)
        group_id = int(group_str)
        return layer_id, group_id, tail
    except (ValueError, IndexError):
        return None


def _parse_flash_kda_router_name(name: str) -> tuple[int, str] | None:
    """Parse top-level ``...moe.router.{field}`` checkpoint names."""

    marker = ".moe.router."
    if marker not in name:
        return None
    layer_id = _get_layer_id(name)
    if layer_id is None:
        return None
    return layer_id, name.split(marker, 1)[1]


def _load_flash_kda_router_correction_bias(
    moe: "FLASHLocalMoE | PackedFLASHLocalMoE",
    loaded_weight: torch.Tensor,
) -> None:
    """Load a full-width or shared per-group router correction bias."""

    groups = moe.moe_group_size
    logits_per_group = moe.n_routed_experts + moe.zero_expert_num
    full_shape = (groups * logits_per_group,)
    loaded_shape = tuple(loaded_weight.shape)
    if loaded_shape == (logits_per_group,):
        loaded_weight = loaded_weight.repeat(groups)
    elif loaded_shape != full_shape:
        raise ValueError(
            "Flash-KDA router correction bias checkpoint tensor has shape "
            f"{loaded_shape}, expected {(logits_per_group,)} (shared per group) "
            f"or {full_shape} (full width)."
        )
    if isinstance(moe, PackedFLASHLocalMoE):
        for group_id, group_weight in enumerate(
            loaded_weight.view(groups, logits_per_group)
        ):
            moe.load_group_router_weight(
                group_id, "e_score_correction_bias", group_weight
            )
        return
    bias = moe.router.e_score_correction_bias
    if tuple(bias.shape) != full_shape:
        raise ValueError(
            "Flash-KDA router correction bias target has shape "
            f"{tuple(bias.shape)}, expected {full_shape}."
        )
    with torch.no_grad():
        bias.copy_(loaded_weight)


def _get_flash_kda_moe_quant_config(
    config: _PretrainedConfig,
    quant_config: QuantizationConfig | None,
    prefix: str,
):
    if quant_config is None:
        return None

    ignored_layers = quant_config.ignored_layers
    if not ignored_layers:
        return quant_config

    expert_proj_names = ("gate_proj", "up_proj", "down_proj")
    num_experts = config.n_routed_experts * config.moe_group_size
    num_expected = num_experts * len(expert_proj_names)
    num_ignored = 0
    for expert_id in range(num_experts):
        expert_prefix = add_prefix(f"experts.{expert_id}", prefix)
        for proj_name in expert_proj_names:
            from tokenspeed.runtime.layers.quantization.utils import (
                should_ignore_quant_layer as _should_ignore_quant_layer,
            )

            if _should_ignore_quant_layer(
                prefix=add_prefix(proj_name, expert_prefix),
                ignored_layers=ignored_layers,
            ):
                num_ignored += 1

    if num_ignored == 0:
        return quant_config
    if num_ignored == num_expected:
        return None

    raise ValueError(
        f"Flash-KDA MoE layer {prefix} has partially ignored expert "
        f"quantization ({num_ignored}/{num_expected} expert projections). "
        "TokenSpeed requires all experts in one fused MoE layer to use the "
        "same weight format."
    )


# ===----------------------------------------------------------------------=== #
# Group-MoE router + block (EveryLayer-MoE with
# per-group routing, shared expert, and zero experts)
# ===----------------------------------------------------------------------=== #


class _FLASHLocalRouter(nn.Module):
    """Softmax / noaux_tc per-group router for Flash-KDA Multi-Group-Head MoE.

    Mirrors ``MultiGroupHeadLongcatRouter`` but is tuned for the Flash-KDA
    expert count (384 routed + 32 zero experts per group). The classifier
    outputs ``n_routed_experts + zero_expert_num`` per-group logits; the fused
    ``group_moe_router`` kernel scatters them group-major. The correction
    bias is kept full width (``G * E_per_group``) for the per-group TopK.
    """

    def __init__(self, config: FLASHLocalConfig, prefix: str = ""):
        super().__init__()
        if getattr(config, "router_bias", False):
            raise ValueError("Flash-KDA router bias is not supported.")

        self.moe_group_size = config.moe_group_size
        n_logits_per_group = config.n_routed_experts + config.zero_expert_num
        params_dtype = (
            torch.bfloat16 if config.router_dtype == "bfloat16" else torch.float32
        )
        self.classifier = ReplicatedLinear(
            config.hidden_size,
            n_logits_per_group,
            bias=False,
            params_dtype=params_dtype,
            quant_config=None,
            prefix=add_prefix("classifier", prefix),
        )
        # Full-width bias (G * E_per_group) consumed by the per-group TopK.
        self.e_score_correction_bias = nn.Parameter(
            torch.zeros(
                self.moe_group_size * n_logits_per_group,
                dtype=torch.float32,
            )
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute per-group router logits via the fused group-moe router kernel.

        Args:
            hidden_states: ``[T * G, H_g]`` group-split hidden states.

        Returns:
            Router logits, ``[T * G, E_per_group]``, dtype float32.
        """
        if hidden_states.shape[0] > 0:
            return _group_moe_router(
                hidden_states,
                self.classifier.weight,
                self.moe_group_size,
                out_dtype=torch.float32,
            )
        # Empty-token fallback: the input is group-split ([0, H_g]) so a plain
        # classifier matmul ([0, H_g] @ [hidden, E]^T) would mismatch on the
        # hidden dim. Return an empty [0, E_per_group] float32 tensor instead —
        # the downstream TopK empty path only needs the shape contract.
        n_logits_per_group = self.classifier.weight.shape[0]
        return torch.empty(
            (0, n_logits_per_group),
            dtype=torch.float32,
            device=hidden_states.device,
        )


def _group_moe_topk(
    router_logits: torch.Tensor,
    correction_bias: torch.Tensor,
    *,
    top_k: int,
    moe_group_size: int,
    zero_expert_num: int,
    renormalize: bool,
    routed_scaling_factor: float,
) -> StandardTopKOutput:
    """Select experts independently per Flash group and globalize their ids."""

    num_experts_per_group = router_logits.shape[-1]
    num_real_experts_per_group = num_experts_per_group - zero_expert_num
    weights, local_ids = _grouped_moe_topk_local(
        router_logits.float().view(moe_group_size, -1, num_experts_per_group),
        correction_bias,
        top_k=top_k,
        renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor,
    )

    group_offsets = torch.arange(
        moe_group_size, device=router_logits.device, dtype=local_ids.dtype
    ).view(-1, 1, 1)
    global_ids = local_ids + group_offsets * num_real_experts_per_group
    global_ids = torch.where(local_ids < num_real_experts_per_group, global_ids, -1)
    return StandardTopKOutput(
        weights.flatten(0, 1),
        global_ids.flatten(0, 1).to(torch.int32),
        router_logits,
    )


class FLASHLocalMoE(nn.Module):
    """Flash-KDA Multi-Group-Head Group-MoE block.

    Structure (mirrors ``MultiGroupHeadLongcatMoe`` with the Flash-KDA
    expert counts, plus a shared expert):

    * **proj_input** — ``nn.Linear(hidden, hidden)`` projecting into the
      per-group space before normalization.
    * **norm** — RMSNorm fused with ``grouped_moe_norm_scale`` and a
      group-split via the ``rmsnorm_scale`` kernel.
    * **Router** ``_FLASHLocalRouter`` — per-group softmax / noaux_tc over
      ``n_routed_experts + zero_expert_num`` logits.
    * **TopK** — per-group bias-corrected top-k (``moe_group_size=4``) with
      global expert offset.
    * **Routed experts** — ``MoELayer`` at ``hidden_size // G`` (per-group
      hidden) with ``n_routed_experts * G`` total experts.
    * **proj_output** — fused ``group_moe_proj_output`` kernel gathering the
      per-group outputs back to the full hidden dim.
    * **Shared expert** — a dense SwiGLU MLP at the full hidden size; its
      TP-partial result joins the layer-end reduction with the routed output.
    """

    def __init__(
        self,
        config: FLASHLocalConfig,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        layer_index: int = -1,
        prefix: str = "",
        alt_stream: torch.cuda.Stream | None = None,
    ):
        super().__init__()
        self.mapping = mapping
        self.layer_index = layer_index
        self.hidden_size = config.hidden_size
        self.moe_group_size = config.moe_group_size
        self.hidden_size_per_group = config.hidden_size // config.moe_group_size
        self.n_routed_experts = config.n_routed_experts
        self.zero_expert_num = config.zero_expert_num
        self.zero_expert_type = config.zero_expert_type
        self.routed_scaling_factor = config.routed_scaling_factor
        self.top_k = config.moe_topk
        self.renormalize_topk = config.norm_topk_prob
        self.grouped_moe_norm_scale = config.grouped_moe_norm_scale
        self.stream_fork = _StreamFork(alt_stream)

        if config.hidden_size % config.moe_group_size != 0:
            raise ValueError(
                f"hidden_size {config.hidden_size} must be divisible by "
                f"moe_group_size {config.moe_group_size}."
            )
        total_routed_experts = config.n_routed_experts * config.moe_group_size
        if self.mapping.moe.ep_size > total_routed_experts:
            raise ValueError(
                f"EP size {self.mapping.moe.ep_size} is greater than the total "
                f"number of Flash-KDA routed experts {total_routed_experts}."
            )
        if config.hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {config.hidden_act}. "
                "Only silu is supported for Flash-KDA."
            )

        params_dtype = (
            torch.bfloat16
            if getattr(config, "router_dtype", "float32") == "bfloat16"
            else torch.get_default_dtype()
        )

        # --- per-group input projection + norm + output projection ---
        # Plain nn.Linear (replicated) — the group-moe kernels read slices of
        # the full [hidden, hidden] weight directly, so no TP sharding here.
        self.proj_input = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.proj_output = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.router = _FLASHLocalRouter(
            config=config,
            prefix=add_prefix("router", prefix),
        )
        # Experts operate at per-group hidden width. Total expert pool is
        # per-group routed * G (zero experts are folded into the top-k weights
        # by the per-group TopK, not stored as real expert rows).
        self.experts = MoELayer(
            top_k=config.moe_topk,
            num_experts=(
                total_routed_experts
                + global_server_args_dict["ep_num_redundant_experts"]
            ),
            hidden_size=self.hidden_size_per_group,
            intermediate_size=config.moe_intermediate_size,
            quant_config=quant_config,
            layer_index=layer_index,
            prefix=prefix,
            tp_rank=self.mapping.moe.tp_rank,
            tp_size=self.mapping.moe.tp_size,
            ep_rank=self.mapping.moe.ep_rank,
            ep_size=self.mapping.moe.ep_size,
            zero_expert_type=config.zero_expert_type,
            activation="silu",
            routing_mode=("precomputed_topk" if config.zero_expert_num > 0 else None),
            routing_config={
                "routed_scaling_factor": self.routed_scaling_factor,
                "normalize_topk_weights": config.norm_topk_prob,
                "correction_bias": self.router.e_score_correction_bias,
                "routing_method_type": RoutingMethodType.DeepSeekV3,
            },
        )
        # Shared expert: a dense SwiGLU MLP at the full hidden width. Its
        # row-parallel output remains partial until the layer-end reduction.
        self.shared_experts = DeepseekV3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=_get_shared_expert_intermediate_size(config),
            hidden_act="silu",
            mapping=self.mapping,
            quant_config=quant_config,
            prefix=add_prefix("shared_experts", prefix),
            is_shared_expert=True,
        )

        # Keep proj weights in the runtime dtype so the fused kernels match.
        self.proj_input.weight.data = self.proj_input.weight.data.to(params_dtype)
        self.proj_output.weight.data = self.proj_output.weight.data.to(params_dtype)

    def get_moe_routed_weights(self):
        return [
            param.data
            for name, param in self.experts.named_parameters()
            if name not in ["correction_bias"] and "shared_experts" not in name
        ]

    def load_group_router_weight(
        self, group_id: int, field: str, loaded_weight: torch.Tensor
    ) -> None:
        """Load one separately stored checkpoint router into CUDA fused storage.

        A ``[E, H/G]`` classifier is copied into group ``g``'s column slice of
        the runtime ``[E, H]`` classifier; correction-bias slices are placed
        analogously. This reconstructs router storage only. Expert placement is
        handled separately by the EP loader and is not group-colocated here.
        """
        if not 0 <= group_id < self.moe_group_size:
            raise ValueError(f"Invalid FLASHLocal router group {group_id}.")
        if field == "classifier.weight":
            start = group_id * self.hidden_size_per_group
            target = self.router.classifier.weight[
                :, start : start + self.hidden_size_per_group
            ]
        elif field == "e_score_correction_bias":
            width = self.n_routed_experts + self.zero_expert_num
            start = group_id * width
            target = self.router.e_score_correction_bias[start : start + width]
        else:
            raise ValueError(f"Invalid FLASHLocal router tensor {field!r}.")
        if tuple(target.shape) != tuple(loaded_weight.shape):
            raise ValueError(
                f"Invalid FLASHLocal router tensor {field!r} for group {group_id}."
            )
        target.data.copy_(loaded_weight.to(device=target.device))

    def _apply_zero_experts(self, hidden_states: torch.Tensor, topk_output):
        """Extract the zero-expert contribution from the per-group top-k.

        ``fused_group_moe_topk_bias`` marks zero-expert slots with id -1 (see
        ``topk.py``); the routed expert ids are global (offset by
        ``g * n_routed_experts``). The identity zero expert contributes its
        input scaled by the zero-expert weight sum.
        """
        if self.zero_expert_num <= 0:
            return None

        zero_expert_mask = topk_output.topk_ids < 0
        zero_expert_weights = torch.where(
            zero_expert_mask,
            topk_output.topk_weights,
            torch.zeros_like(topk_output.topk_weights),
        )
        # Fused MoE kernels still read every selected expert id while building
        # the dispatch plan, so zero-expert slots must keep a valid id.
        topk_output.topk_ids[zero_expert_mask] = 0
        topk_output.topk_weights[zero_expert_mask] = 0.0

        if self.zero_expert_type in ("identity", "copy"):
            zero_weight = zero_expert_weights.sum(dim=-1, keepdim=True).to(
                hidden_states.dtype
            )
            return hidden_states * (zero_weight / self.mapping.moe.tp_ep_size)
        if self.zero_expert_type in ("", "drop"):
            return None
        raise ValueError(
            f"Unsupported Flash-KDA zero expert type: {self.zero_expert_type}"
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        num_global_tokens: int,
        max_num_tokens_per_gpu: int,
        ctx: ForwardContext | None = None,
    ) -> torch.Tensor:
        """Multi-Group-Head MoE forward.

        Flow::

            shared_out = shared_experts(hidden_states)   # full-hidden MLP
            h = proj_input(hidden_states)                  # [T, H] -> [T, H]
            h = rmsnorm_scale(h, norm, G, norm_scale)     # -> [T*G, H//G]
            logits = router(h)                            # [T*G, E_per_group]
            topk = TopK(h, logits)                        # per-group top-k
            zero_out = _apply_zero_experts(h, topk)        # identity zero expert
            routed = experts(h, topk, ...)                 # [T*G, H//G]
            routed = group_moe_proj_output(routed, proj_output, G)  # [T, H]
            return routed + shared_out + zero_out
        """
        del ctx
        real_num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        with self.stream_fork.scope(enable=_get_is_capture_mode()):
            # Shared expert runs on the full hidden states (pre-group-split).
            shared_out = self.shared_experts(hidden_states)

            # Per-group input projection + fused RMSNorm + scale + group-split.
            h = self.proj_input(hidden_states)
            h = _rmsnorm_scale(
                h,
                self.norm.weight.data,
                self.norm.variance_epsilon,
                self.moe_group_size,
                self.grouped_moe_norm_scale,
            )

            router_logits = self.router(h)
            topk_output = _group_moe_topk(
                router_logits,
                self.router.e_score_correction_bias,
                top_k=self.top_k,
                moe_group_size=self.moe_group_size,
                zero_expert_num=self.zero_expert_num,
                renormalize=self.renormalize_topk,
                routed_scaling_factor=self.routed_scaling_factor,
            )

            zero_expert_output = self._apply_zero_experts(h, topk_output)
            deferred_finalize = self.experts.supports_deferred_finalize

            # Routed experts at per-group hidden width; num_global_tokens and
            # max_num_tokens_per_gpu scale by G because the token count after
            # group-split is T*G.
            routed_expert_output = self.experts(
                hidden_states=h,
                topk_output=topk_output,
                num_global_tokens=num_global_tokens * self.moe_group_size,
                max_num_tokens_per_gpu=max_num_tokens_per_gpu * self.moe_group_size,
                do_finalize=not deferred_finalize,
            )

        # Gather per-group outputs back to [T, H] via the fused projection.
        if deferred_finalize:
            gemm2_out, expert_weights, expanded_idx = routed_expert_output
            routed_expert_output = _moe_finalize_fuse_shared(
                gemm2_out,
                expanded_idx,
                expert_weights,
                zero_expert_output,
                top_k=self.top_k,
                enable_pdl=_pdl_enabled(),
            )
        else:
            if zero_expert_output is not None:
                routed_expert_output = routed_expert_output + zero_expert_output

        # routed_expert_output is [T*G, H_g]; project back to [T, H].
        routed_expert_output = _group_moe_proj_output(
            routed_expert_output,
            self.proj_output.weight,
            self.moe_group_size,
        )

        if shared_out is not None:
            routed_expert_output = routed_expert_output + shared_out
        return routed_expert_output.view(real_num_tokens, hidden_dim)


class _FLASHLocalTwoLayerProj(nn.Module):
    """Megatron-style 2-layer bottleneck projection used by Flash-KDA KDA."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        *,
        tp_rank: int,
        tp_size: int,
        tp_group,
        quant_config: QuantizationConfig | None,
        prefix: str,
        out_parallel: bool,
        second_bias: bool = False,
    ) -> None:
        super().__init__()
        self.fc1 = ReplicatedLinear(
            in_features,
            hidden_features,
            bias=False,
            quant_config=None,
            prefix=add_prefix("0", prefix),
        )
        if out_parallel:
            self.fc2 = ColumnParallelLinear(
                hidden_features,
                out_features,
                bias=second_bias,
                quant_config=quant_config,
                prefix=add_prefix("1", prefix),
                tp_rank=tp_rank,
                tp_size=tp_size,
                tp_group=tp_group,
            )
        else:
            self.fc2 = ReplicatedLinear(
                hidden_features,
                out_features,
                bias=second_bias,
                quant_config=None,
                prefix=add_prefix("1", prefix),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.fc1(x)
        x, _ = self.fc2(x)
        return x


class SeperateFLASHLocal(nn.Module):
    """Megatron-style separate-proj Flash-KDA block backed by MambaAttnBackend.

    Keeps the Megatron-visible module layout (q/k/v, f_proj, g_proj, b_proj,
    q/k/v_conv1d, A_log, dt_bias, o_norm, o_proj), but packs the forward-time
    intermediates into the existing tokenspeed KDA backend contract.
    """

    def __init__(
        self,
        config: FLASHLocalConfig,
        mapping: Mapping,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.mapping = mapping
        self.layer_id = layer_id

        la = config.linear_attn_config
        if not la.get("kda_use_full_rank_gate", la.get("use_full_rank_gate", False)):
            raise NotImplementedError(
                "Flash-KDA separate KDA only implements kda_use_full_rank_gate=True."
            )

        self.hidden_size = config.hidden_size
        self.num_heads = la["linear_num_heads"]
        self.head_dim = la["linear_head_dim"]
        self.conv_size = la["linear_conv_size"]
        self.gate_lower_bound = la.get("gate_lower_bound")
        self.linear_method = str(getattr(config, "linear_method", "FGBKDA")).upper()
        self.is_fgbkda = self.linear_method == "FGBKDA"

        tp_rank = mapping.linear_attn.tp_rank
        tp_size = mapping.linear_attn.tp_size
        tp_group = mapping.linear_attn.tp_group

        self.local_num_heads = self.num_heads // tp_size
        self.proj = self.num_heads * self.head_dim
        self.proj_local = self.proj // tp_size

        self.q_proj = ColumnParallelLinear(
            self.hidden_size,
            self.proj,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("q_proj", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
            tp_group=tp_group,
        )
        self.k_proj = ColumnParallelLinear(
            self.hidden_size,
            self.proj,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("k_proj", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
            tp_group=tp_group,
        )
        self.v_proj = ColumnParallelLinear(
            self.hidden_size,
            self.proj,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("v_proj", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
            tp_group=tp_group,
        )

        self.f_proj = _FLASHLocalTwoLayerProj(
            self.hidden_size,
            self.head_dim,
            self.proj,
            tp_rank=tp_rank,
            tp_size=tp_size,
            tp_group=tp_group,
            quant_config=quant_config,
            prefix=add_prefix("f_proj", prefix),
            out_parallel=True,
        )
        if self.is_fgbkda:
            self.g_proj = ColumnParallelLinear(
                self.hidden_size,
                self.proj,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("g_proj", prefix),
                tp_rank=tp_rank,
                tp_size=tp_size,
                tp_group=tp_group,
            )
            self.b_proj = _FLASHLocalTwoLayerProj(
                self.hidden_size,
                self.head_dim,
                self.proj,
                tp_rank=tp_rank,
                tp_size=tp_size,
                tp_group=tp_group,
                quant_config=quant_config,
                prefix=add_prefix("b_proj", prefix),
                out_parallel=True,
                second_bias=False,
            )
        else:
            self.g_proj = _FLASHLocalTwoLayerProj(
                self.hidden_size,
                self.proj,
                self.proj,
                tp_rank=tp_rank,
                tp_size=tp_size,
                tp_group=tp_group,
                quant_config=quant_config,
                prefix=add_prefix("g_proj", prefix),
                out_parallel=True,
                second_bias=True,
            )
            self.b_proj = ColumnParallelLinear(
                self.hidden_size,
                self.num_heads,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("b_proj", prefix),
                tp_rank=tp_rank,
                tp_size=tp_size,
                tp_group=tp_group,
            )

        self.q_conv1d_weight = nn.Parameter(
            torch.zeros(self.proj_local, 1, self.conv_size)
        )
        self.k_conv1d_weight = nn.Parameter(
            torch.zeros(self.proj_local, 1, self.conv_size)
        )
        self.v_conv1d_weight = nn.Parameter(
            torch.zeros(self.proj_local, 1, self.conv_size)
        )
        for w in (self.q_conv1d_weight, self.k_conv1d_weight, self.v_conv1d_weight):
            w.weight_loader = sharded_weight_loader(0, tp_rank)

        self.A_log = nn.Parameter(
            torch.zeros(self.local_num_heads, dtype=torch.float32)
        )
        _alog_start = self.local_num_heads * tp_rank

        def _a_log_head_loader(param, loaded_weight):
            param.data.copy_(loaded_weight.narrow(0, _alog_start, self.local_num_heads))

        self.A_log.weight_loader = _a_log_head_loader
        self.dt_bias = nn.Parameter(torch.zeros(self.proj_local, dtype=torch.float32))
        self.dt_bias.weight_loader = sharded_weight_loader(0, tp_rank)
        self.conv_weights: torch.Tensor | None = None

        self.o_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.o_proj = RowParallelLinear(
            self.proj,
            self.hidden_size,
            bias=False,
            reduce_results=False,
            tp_rank=tp_rank,
            tp_size=tp_size,
            tp_group=tp_group,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )

    def fuse_conv_weights(self) -> None:
        self.conv_weights = torch.cat(
            (self.q_conv1d_weight, self.k_conv1d_weight, self.v_conv1d_weight), dim=0
        ).squeeze(1)

    def _project_qkv(self, hidden_states: torch.Tensor) -> torch.Tensor:
        q, _ = self.q_proj(hidden_states)
        k, _ = self.k_proj(hidden_states)
        v, _ = self.v_proj(hidden_states)
        return torch.cat((q, k, v), dim=-1).contiguous()

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: ForwardContext,
        comm_manager,
        block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del positions, comm_manager, block_scale
        if hidden_states.shape[0] == 0:
            return hidden_states

        num_tokens = hidden_states.shape[0]
        hn, hd = self.local_num_heads, self.head_dim

        mixed_qkv = self._project_qkv(hidden_states)
        beta_channel_raw = None
        beta_is_logit = True
        if self.is_fgbkda:
            out_gate, _ = self.g_proj(hidden_states)
            beta_channel_raw = _reshape_fgbkda_beta_logits(
                self.b_proj(hidden_states), num_heads=hn, head_dim=hd
            )
            beta_raw = torch.ones(
                (num_tokens, hn),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            beta_is_logit = False
        else:
            out_gate = self.g_proj(hidden_states)
            beta_raw, _ = self.b_proj(hidden_states)
        f_a_out, _ = self.f_proj.fc1(hidden_states)
        conv_weights = self.conv_weights

        core_out = ctx.attn_backend.forward(
            q=None,
            k=None,
            v=None,
            layer=None,
            token_to_kv_pool=ctx.token_to_kv_pool,
            forward_mode=ctx.forward_mode,
            bs=ctx.bs,
            mixed_qkv=mixed_qkv,
            conv_weights=conv_weights,
            bias=None,
            activation="silu",
            key_dim=self.proj,
            value_dim=self.proj,
            attention_tp_size=self.mapping.linear_attn.tp_size,
            head_k_dim=hd,
            head_v_dim=hd,
            f_a_out=f_a_out,
            f_b_weight=self.f_proj.fc2.weight,
            beta_raw=beta_raw,
            beta_channel_raw=beta_channel_raw,
            beta_is_logit=beta_is_logit,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            lower_bound=self.gate_lower_bound,
            layer_id=self.layer_id,
            seq_len=num_tokens,
        )

        core_out = rmsnorm_gated_sigmoid(
            core_out.reshape(num_tokens, hn * hd).contiguous(),
            out_gate.contiguous(),
            self.o_norm.weight,
            self.o_norm.variance_epsilon,
            hn,
            hd,
            enable_pdl=_pdl_enabled(),
        )
        output, _ = self.o_proj(core_out)
        return output


# ===----------------------------------------------------------------------=== #
# Decoder layer
# ===----------------------------------------------------------------------=== #


class FLASHLocalDecoderLayer(nn.Module):
    """Flash-KDA decoder layer: MLA/KDA dispatch + Group-MoE.

    The residual flow is the plain Flash-style add (no AttnRes block-residual
    mixing, no dense shortcut MLP)::

        h = input_layernorm(hidden_states)
        h = self_attn(h)                     # MLA or KDA (per layer)
        hidden_states = hidden_states + h    # attention residual
        h = post_attention_layernorm(hidden_states)
        h_moe = moe(h)                       # Group-MoE + shared + zero
        hidden_states = hidden_states + h_moe

    The MLA/KDA dispatch follows ``KimiLinearDecoderLayer``: KDA layers route
    through ``SeperateFLASHLocal`` (hybrid ``MambaAttnBackend`` KDA branch), MLA
    layers through ``KimiLinearMLAAttention`` (gated NoPE MLA).
    """

    def __init__(
        self,
        config: FLASHLocalConfig,
        layer_id: int,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        alt_stream: torch.cuda.Stream | None = None,
    ) -> None:
        super().__init__()
        self.mapping = mapping
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        self.config = config

        rope_theta = _get_rope_theta(config)
        rope_scaling = getattr(config, "rope_scaling", None)
        if rope_scaling and "factor" not in rope_scaling:
            rope_scaling = None
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)

        # --- attention: KDA (linear) or gated NoPE-MLA (full); under "self_attn" ---
        attn_prefix = add_prefix("self_attn", prefix)
        if config.is_kda_layer(layer_id):
            if flash_local_prefers_packed_projections():
                self.self_attn = PackedFLASHLocalKDA(config, mapping, layer_id)
            else:
                self.self_attn = SeperateFLASHLocal(
                    config, mapping, layer_id, quant_config, attn_prefix
                )
        else:
            # reduce_attn_results=False: the o_proj returns a TP partial and
            # CommManager.post_attn_reduce_norm owns the all-reduce / RSAG,
            # matching LongCat / DeepSeek-V3. With reduce_attn_results=True the
            # o_proj would all-reduce internally AND CommManager would reduce
            # again (double all-reduce in AR mode, or a reduce over an already-
            # full tensor in RSAG mode producing wrong partial sums).
            #
            # skip_rope=not mla_use_nope: Flash-KDA MLA carries partial RoPE on
            # the qk_rope_head_dim slice (DeepSeek-V3 convention) when
            # mla_use_nope=False. rope_theta / rope_scaling are honored only on
            # this branch (KDA layers own their own positional encoding); the
            # parent builds rotary_emb iff skip_rope=False, and every rope
            # application in the absorbed-decode / chunked-prefill paths is
            # already guarded by ``self.rotary_emb is not None``.
            if flash_local_prefers_packed_projections():
                self.self_attn = SeparateProjectionKimiLinearMLAAttention(
                    config,
                    mapping,
                    layer_id=layer_id,
                    prefix=attn_prefix,
                )
            else:
                self.self_attn = KimiLinearMLAAttention(
                    config=config,
                    mapping=mapping,
                    component_mapping=mapping.mla_weight,
                    hidden_size=config.hidden_size,
                    num_heads=config.num_attention_heads,
                    qk_nope_head_dim=config.qk_nope_head_dim,
                    qk_rope_head_dim=config.qk_rope_head_dim,
                    v_head_dim=config.v_head_dim,
                    q_lora_rank=config.q_lora_rank,
                    kv_lora_rank=config.kv_lora_rank,
                    rope_theta=rope_theta,
                    rope_scaling=rope_scaling,
                    max_position_embeddings=max_position_embeddings,
                    quant_config=quant_config,
                    layer_id=layer_id,
                    prefix=attn_prefix,
                    reduce_attn_results=False,
                    alt_stream=alt_stream,
                    skip_rope=config.mla_use_nope,
                )

        # --- FFN: EveryLayer Group-MoE (no dense shortcut) ---
        # The Group-MoE block (``moe``) runs every layer; Flash-KDA's
        # ``first_k_dense_replace=0`` makes every layer a MoE layer. There is
        # no parallel dense shortcut MLP — the routed + shared + zero experts
        # are the only FFN path.
        self.is_moe_layer = (
            config.num_experts is not None
            and layer_id >= config.first_k_dense_replace
            and layer_id % config.moe_layer_freq == 0
        )
        if not self.is_moe_layer:
            raise ValueError(
                "Flash-KDA requires EveryLayer-MoE (first_k_dense_replace=0); "
                f"got layer {layer_id} not flagged as MoE."
            )
        # Both leaves implement the same four-group routing math. ``packed``
        # denotes Ascend's group-major EP-local w13/w2 runtime storage; CUDA
        # retains the existing MoELayer layout and kernels.
        self.moe = (
            PackedFLASHLocalMoE(config, mapping)
            if flash_local_prefers_packed_moe()
            else FLASHLocalMoE(
                config=config,
                mapping=mapping,
                quant_config=_get_flash_kda_moe_quant_config(
                    config,
                    quant_config,
                    add_prefix("moe", prefix),
                ),
                layer_index=layer_id,
                prefix=add_prefix("moe", prefix),
                alt_stream=alt_stream,
            )
        )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        if flash_local_prefers_packed_projections():
            self.input_layernorm.to(dtype=torch.bfloat16)
            self.post_attention_layernorm.to(dtype=torch.bfloat16)

        # CommManager: plain pre-attn / post-attn / pre-mlp / post-mlp fusion.
        # The MoE block uses its own CommManager scope (is_moe=True) for the
        # routed expert all-reduce.
        self.moe_comm = _CommManager(
            mapping=mapping,
            layer_id=self.layer_id,
            is_moe=True,
            prev_is_moe=False,
            input_layernorm=self.input_layernorm,
            post_attn_layernorm=self.post_attention_layernorm,
            attn_mapping=(
                mapping.linear_attn if config.is_kda_layer(layer_id) else mapping.attn
            ),
        )

    def _forward_attn(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: ForwardContext,
    ) -> torch.Tensor:
        """Attention forward (MLA or KDA) with pre-attn comm fusion."""
        hidden_states = self.moe_comm.pre_attn_comm(hidden_states, ctx)
        attn_out = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            ctx=ctx,
            comm_manager=self.moe_comm,
        )
        return attn_out

    def _forward_moe(
        self,
        hidden_states: torch.Tensor,
        ctx: ForwardContext,
        num_global_tokens: int,
        max_num_tokens_per_gpu: int,
    ) -> torch.Tensor:
        """MoE forward (routed + shared + zero) with pre-mlp comm fusion."""
        hidden_states = self.moe_comm.pre_mlp_comm(hidden_states, ctx)
        return self.moe(
            hidden_states,
            num_global_tokens=num_global_tokens,
            max_num_tokens_per_gpu=max_num_tokens_per_gpu,
            ctx=ctx,
        )

    def _post_attn(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        ctx: ForwardContext,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(self.self_attn, SeparateProjectionKimiLinearMLAAttention):
            return self.post_attention_layernorm(hidden_states, residual)
        return self.moe_comm.post_attn_reduce_norm(hidden_states, residual, ctx)

    def _post_moe(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        ctx: ForwardContext,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(self.moe, PackedFLASHLocalMoE):
            return hidden_states, residual
        return self.moe_comm.post_mlp_fused(hidden_states, residual, ctx)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: ForwardContext,
        residual: torch.Tensor | None,
        capture_hidden_state: Callable[[torch.Tensor], None] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        num_global_tokens, max_num_tokens_per_gpu = self.moe_comm.get_num_tokens(ctx)

        if ctx.forward_mode.is_idle():
            # Idle: just run the MoE (keeps the graph shape stable).
            hidden_states, residual = self.moe_comm.input_reduce_norm(
                hidden_states, residual
            )
            moe_out = self._forward_moe(
                hidden_states, ctx, num_global_tokens, max_num_tokens_per_gpu
            )
            hidden_states, residual = self._post_moe(moe_out, residual, ctx)
            return hidden_states, residual

        # --- attention ---
        hidden_states, residual = self.moe_comm.input_reduce_norm(
            hidden_states, residual
        )
        if capture_hidden_state is not None:
            capture_hidden_state(self.moe_comm.gather_residual(residual, ctx).clone())
        attn_out = self._forward_attn(positions, hidden_states, ctx)
        hidden_states, residual = self._post_attn(attn_out, residual, ctx)

        # --- FFN: Group-MoE (EveryLayer-MoE, no dense shortcut) ---
        moe_out = self._forward_moe(
            hidden_states, ctx, num_global_tokens, max_num_tokens_per_gpu
        )
        hidden_states, residual = self._post_moe(moe_out, residual, ctx)
        return hidden_states, residual


# ===----------------------------------------------------------------------=== #
# Model
# ===----------------------------------------------------------------------=== #


class FLASHLocalModel(nn.Module):
    """Flash-KDA text transformer: embedding + hybrid decoder layers.

    Plain residual flow (no AttnRes block-residual mixing); the per-layer
    ``FLASHLocalDecoderLayer`` threads ``residual`` through ``CommManager``.
    """

    fall_back_to_pt_during_load = False

    def __init__(
        self,
        config: FLASHLocalConfig,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        oe_table_placement: str | None = "device",
    ) -> None:
        super().__init__()
        self.config = config
        self.mapping = mapping
        self.padding_id = getattr(config, "pad_token_id", None)
        self.vocab_size = config.vocab_size

        self.ngram_embeddings: HostLongCatOverEmbedding | None = None
        self.embed_tokens = self._build_embed_tokens(
            config,
            quant_config,
            oe_table_placement,
        )
        self.alt_stream = torch.cuda.Stream() if torch.cuda.is_available() else None

        def get_layer(idx: int, prefix: str):
            return FLASHLocalDecoderLayer(
                config=config,
                layer_id=idx,
                mapping=self.mapping,
                quant_config=quant_config,
                prefix=prefix,
                alt_stream=self.alt_stream,
            )

        from tokenspeed.runtime.utils import make_layers

        self.layers = make_layers(
            config.num_hidden_layers,
            get_layer,
            prefix=add_prefix("layers", prefix),
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if flash_local_prefers_packed_projections():
            self.norm.to(dtype=torch.bfloat16)
        self.layers_to_capture: set[int] = set()

    def _build_embed_tokens(self, config, quant_config, oe_table_placement):
        """Create the plain, Device-OE, or Host-OE embedding components.

        OE is enabled when ``config.use_over_embedding`` is True (i.e. the
        checkpoint's ``ngram_vocab_size_ratio`` / legacy ``oe_vocab_size_ratio``
        is set). The OE layer embeds the regular word table plus ``n_grams``
        n-gram over-embedding tables and merges them via a per-gram projection.
        """
        if not getattr(config, "use_over_embedding", False):
            return VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                tp_rank=self.mapping.attn.tp_rank,
                tp_size=self.mapping.attn.tp_size,
                tp_group=self.mapping.attn.tp_group,
            )

        if oe_table_placement == "host":
            self.ngram_embeddings = HostLongCatOverEmbedding(config)
            return VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                params_dtype=torch.bfloat16,
                tp_rank=self.mapping.dense.tp_rank,
                tp_size=self.mapping.dense.tp_size,
                tp_group=self.mapping.dense.tp_group,
            )
        if oe_table_placement != "device":
            raise ValueError(
                "OE checkpoints require a resolved host or device table placement."
            )

        # OE hyperparameters: neighbor_num is the maximum n-gram order, while
        # split_num is the number of hash branches per order. LongCat-2.0 uses
        # neighbor=5 and split=4, producing (5-1)*4 = 16 branches.
        max_ngram_order, hashes_per_order = resolve_longcat_oe_hyperparameters(config)
        exclude_special_tokens = getattr(config, "ngram_exclude_sp_token", False)
        return LongCatOverEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            over_embedding_m=config.over_embedding_m,
            hashes_per_order=hashes_per_order,
            max_ngram_order=max_ngram_order,
            tp_rank=self.mapping.attn.tp_rank,
            tp_size=self.mapping.attn.tp_size,
            tp_group=self.mapping.attn.tp_group,
            ignored_token_ids=(
                tuple(config.oe_ignore_tokens) if exclude_special_tokens else ()
            ),
            eos_token_id=getattr(config, "eos_token_id", None),
            segment_ignored_tokens=exclude_special_tokens,
            fix_normalize_factor=getattr(config, "ngram_fix_normalize_factor", False),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        ctx: ForwardContext,
        input_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        if input_embeds is not None:
            hidden_states = input_embeds
        elif isinstance(self.embed_tokens, LongCatOverEmbedding):
            hidden_states = self.embed_tokens(input_ids, ctx)
        else:
            hidden_states = self.embed_tokens(input_ids)
            if self.ngram_embeddings is not None:
                hidden_states = self.ngram_embeddings.project_and_merge(
                    hidden_states,
                    self.ngram_embeddings.prepared_raw_oe(input_ids.numel()),
                    input_ids,
                )

        residual = None
        aux_hidden_states = [] if self.layers_to_capture else None
        layer = None
        for layer_id, layer in enumerate(self.layers):
            capture_hidden_state = None
            if aux_hidden_states is not None and layer_id in self.layers_to_capture:
                capture_hidden_state = aux_hidden_states.append
            with get_global_expert_distribution_recorder().with_current_layer(layer_id):
                hidden_states, residual = layer(
                    positions,
                    hidden_states,
                    ctx,
                    residual,
                    capture_hidden_state=capture_hidden_state,
                )

        if not ctx.forward_mode.is_idle() and layer is not None:
            hidden_states, _ = layer.moe_comm.final_norm(
                hidden_states,
                residual,
                ctx,
                self.norm,
            )
        return hidden_states, aux_hidden_states


# ===----------------------------------------------------------------------=== #
# Causal LM wrapper
# ===----------------------------------------------------------------------=== #


class FLASHLocalForCausalLM(BaseCausalLM):
    """Flash-KDA 3B text backbone: ``FLASHLocalModel`` + lm head + logits processor.

    Inherits ``BaseCausalLM`` so the ``model.*`` / ``lm_head.*`` weight
    hierarchy matches the checkpoint (``model.*`` / ``lm_head.*`` after the
    wrapper strips any ``language_model.`` prefix).
    """

    model_cls = FLASHLocalModel
    supports_oe_table_placement = True
    oe_runtime_capabilities = {
        "cuda": {"device": "runtime-full-history"},
        "npu": {"host": "cache-checkpointed-tail"},
    }

    def __init__(
        self,
        config: FLASHLocalConfig,
        mapping: Mapping,
        model: FLASHLocalModel | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        oe_table_placement: str | None = "device",
    ) -> None:
        self._model_override = model
        self.oe_table_placement = oe_table_placement
        self.checkpoint_layout = (
            FLASHLocalCheckpointLayout(config)
            if getattr(config, "strict_checkpoint_layout", False)
            else None
        )
        super().__init__(
            config=config,
            mapping=mapping,
            quant_config=quant_config,
            prefix=prefix,
        )

    def resolve_model(
        self,
        config: FLASHLocalConfig,
        mapping: Mapping,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> FLASHLocalModel:
        if self._model_override is not None:
            return self._model_override
        return self.model_cls(
            config,
            mapping=mapping,
            quant_config=quant_config,
            prefix=add_prefix("model", prefix),
            oe_table_placement=self.oe_table_placement,
        )

    def resolve_lm_head(self, config, quant_config, prefix):
        """Construct the head in the domain that owns the final hidden state.

        The migrated Host-OE/NPU leaf follows the dense TP/DP domain used by the
        former Lite entry. Device-OE/GPU keeps BaseCausalLM's existing attention
        domain. Process-group initialization cannot own this choice because it
        has no model, vocabulary, quantization, or layer-construction context.
        """
        if self.oe_table_placement != "host":
            return super().resolve_lm_head(config, quant_config, prefix)
        if self.mapping.dense.has_dp:
            return ReplicatedLinear(
                config.hidden_size,
                config.vocab_size,
                bias=False,
                params_dtype=torch.bfloat16,
                prefix=add_prefix("lm_head", prefix),
            )
        return ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            params_dtype=torch.bfloat16,
            quant_config=quant_config,
            prefix=add_prefix("lm_head", prefix),
            tp_rank=self.mapping.dense.tp_rank,
            tp_size=self.mapping.dense.tp_size,
            tp_group=self.mapping.dense.tp_group,
        )

    def resolve_logits_processor(self, config):
        if self.oe_table_placement != "host":
            return super().resolve_logits_processor(config)
        return LogitsProcessor(
            config,
            skip_all_gather=self.mapping.dense.has_dp,
            tp_rank=self.mapping.dense.tp_rank,
            tp_size=self.mapping.dense.tp_size,
            tp_group=self.mapping.dense.tp_group,
        )

    def initialize_external_inputs(
        self,
        *,
        token_to_kv_pool,
        max_request_slots: int,
        max_graph_tokens: int,
        device: str,
    ) -> None:
        host_oe = getattr(self.model, "ngram_embeddings", None)
        if host_oe is None:
            return
        host_oe.initialize_runtime(
            context_pages=token_to_kv_pool.arena.field("layer.0.lite.oe.context"),
            checkpoint_granularity=token_to_kv_pool.arena.plan.prefix_granularity,
            max_request_slots=max_request_slots,
            max_graph_tokens=max_graph_tokens,
            device=device,
        )

    def prepare_external_inputs(
        self,
        forward_op,
        *,
        resolved_input_ids: torch.Tensor,
        graph_tokens: int | None,
    ) -> torch.Tensor | None:
        host_oe = getattr(self.model, "ngram_embeddings", None)
        if host_oe is None:
            return None
        return host_oe.prepare_forward_op(
            forward_op,
            resolved_input_ids=resolved_input_ids,
            graph_tokens=graph_tokens,
        )

    def post_init(self) -> None:
        self._routed_experts_weights_of_layer = LazyValue(
            lambda: {
                layer_id: layer.moe.get_moe_routed_weights()
                for layer_id, layer in enumerate(self.model.layers)
                if isinstance(layer.moe, (FLASHLocalMoE, PackedFLASHLocalMoE))
            }
        )

    @property
    def routed_experts_weights_of_layer(self):
        return self._routed_experts_weights_of_layer.value

    def set_eagle3_layers_to_capture(self, layer_ids: list[int] | None = None):
        self.capture_aux_hidden_states = True
        if layer_ids is None:
            num_layers = self.config.num_hidden_layers
            self.model.layers_to_capture = {2, num_layers // 2, num_layers - 3}
        else:
            self.model.layers_to_capture = {val + 1 for val in layer_ids}

    def get_param(self, params_dict, name):
        if name in params_dict:
            return params_dict[name]
        if "language_model." in name:
            name = name.replace("language_model.", "")
            if name in params_dict:
                return params_dict[name]
        if name.endswith(_OPTIONAL_MISSING_WEIGHT_SUFFIXES):
            return None
        logger.warning("The %s is not in the model.", name)
        return None

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        """Load the ``model.*`` / ``lm_head.*`` text weights.

        Model-owned ``load_weights`` is the repository convention for mapping a
        checkpoint schema onto model-specific parameters. This implementation
        keeps source keys stable, validates strict Flash-Lite streams, and lets
        the selected KDA/MLA/MoE/OE target leaf own packing and sharding. Legacy
        Flash-KDA still follows its separate Megatron-style KDA layout.
        """
        config = self.config
        stacked_params_mapping = [
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        fuse_qkv_a_proj = config.q_lora_rank is not None

        params_dict = dict(self.named_parameters())
        packed_moe = next(
            (
                layer.moe
                for layer in self.model.layers
                if isinstance(getattr(layer, "moe", None), PackedFLASHLocalMoE)
            ),
            None,
        )
        moe_loader = build_moe_checkpoint_loader(
            params_dict=params_dict,
            expert_schema=ExpertCheckpointSchema(
                gate_proj_name="gate_proj",
                down_proj_name="down_proj",
                up_proj_name="up_proj",
            ),
            num_experts=config.n_routed_experts * config.moe_group_size,
            ep_rank=self.mapping.moe.ep_rank,
            ep_size=self.mapping.moe.ep_size,
            local_expert_ids=(
                packed_moe.local_expert_ids if packed_moe is not None else None
            ),
        )

        expected_sources = (
            set(self.checkpoint_layout.iter_source_names())
            if self.checkpoint_layout is not None
            else None
        )
        seen_sources: set[str] = set()

        for source_name, loaded_weight in weights:
            checkpoint_spec = None
            if self.checkpoint_layout is not None:
                if source_name in seen_sources:
                    raise ValueError(
                        f"Duplicate Flash-Lite checkpoint weight {source_name!r}."
                    )
                checkpoint_spec = self.checkpoint_layout.spec(source_name)
                if (
                    tuple(loaded_weight.shape) != checkpoint_spec.shape
                    or loaded_weight.dtype != checkpoint_spec.dtype
                ):
                    raise ValueError(
                        f"Flash-Lite checkpoint weight {source_name!r} has "
                        f"shape/dtype {tuple(loaded_weight.shape)}/"
                        f"{loaded_weight.dtype}; expected "
                        f"{checkpoint_spec.shape}/{checkpoint_spec.dtype}."
                    )
                seen_sources.add(source_name)

            name = source_name
            name = _canonical_flash_kda_weight_name(name)
            _validate_shared_expert_weight_shape(config, name, loaded_weight)
            layer_id = _get_layer_id(name)
            if (
                layer_id is not None
                and hasattr(self.model, "start_layer")
                and (
                    layer_id < self.model.start_layer
                    or layer_id >= self.model.end_layer
                )
            ):
                continue
            if "rotary_emb.inv_freq" in name:
                continue

            if checkpoint_spec is not None and checkpoint_spec.expert_id is not None:
                if moe_loader.matches(name):
                    moe_loader.load(name, loaded_weight)
                continue

            if (
                checkpoint_spec is not None
                and layer_id is not None
                and ".self_attn.linear_core." in source_name
                and isinstance(
                    self.model.layers[layer_id].self_attn, PackedFLASHLocalKDA
                )
            ):
                param = params_dict.get(checkpoint_spec.target_name)
                if param is None:
                    raise ValueError(
                        "Missing packed KDA target parameter "
                        f"{checkpoint_spec.target_name!r}."
                    )
                local_weight = loaded_weight
                if checkpoint_spec.parallel is not None:
                    parallel = (
                        self.mapping.dense
                        if checkpoint_spec.parallel == "dense"
                        else self.mapping.linear_attn
                    )
                    assert checkpoint_spec.shard_axis is not None
                    shard_size = (
                        checkpoint_spec.shape[checkpoint_spec.shard_axis]
                        // parallel.tp_size
                    )
                    local_weight = loaded_weight.narrow(
                        checkpoint_spec.shard_axis,
                        parallel.tp_rank * shard_size,
                        shard_size,
                    )
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                if checkpoint_spec.component_id is None:
                    weight_loader(param, local_weight)
                else:
                    weight_loader(param, local_weight, checkpoint_spec.component_id)
                continue

            if (
                checkpoint_spec is not None
                and layer_id is not None
                and ".self_attn." in source_name
                and isinstance(
                    self.model.layers[layer_id].self_attn,
                    SeparateProjectionKimiLinearMLAAttention,
                )
            ):
                param = params_dict.get(checkpoint_spec.target_name)
                if param is None:
                    raise ValueError(
                        "Missing separate MLA target parameter "
                        f"{checkpoint_spec.target_name!r}."
                    )
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                continue

            group_router_match = _parse_flash_kda_expert_group_router_name(name)
            if group_router_match is not None:
                layer_id, group_id, field = group_router_match
                layer = self.model.layers[layer_id]
                moe = getattr(layer, "moe", None)
                if moe is None:
                    continue
                moe.load_group_router_weight(group_id, field, loaded_weight)
                continue

            router_match = _parse_flash_kda_router_name(name)
            if router_match is not None:
                layer_id, field = router_match
                layer = self.model.layers[layer_id]
                moe = getattr(layer, "moe", None)
                if moe is None:
                    continue
                if field == "classifier.weight":
                    param = moe.router.classifier.weight
                    if param.shape == loaded_weight.shape:
                        with torch.no_grad():
                            param.copy_(loaded_weight)
                        continue
                if field == "e_score_correction_bias":
                    _load_flash_kda_router_correction_bias(moe, loaded_weight)
                    continue

            if name.endswith((".g_proj.fc2.weight", ".g_proj.fc2.bias")) and (
                name not in params_dict
            ):
                # FGBKDA uses the first g projection directly; checkpoints may
                # still carry the unused second projection from the generic KDA
                # module layout.
                continue

            # OE (n-gram over-embedding) weights: route to the branch-local layer
            # when present. These weights are NOT in ``params_dict`` as named
            # parameters (LongCatOverEmbedding owns them and loads via its own
            # load_weight), so intercept them before the stacked-mapping /
            # get_param path would skip them. Supported HF key conventions:
            #   * ``model.embed_tokens.weight``                       -> word
            #   * ``model.oe_embed_tokens{N}.weight`` / ``model.oe_embed_proj{N}.weight``
            #     (legacy oe_* naming)
            #   * ``model.ngram_embeddings.embedders.{N}.weight``      -> oe_embeder
            #   * ``model.ngram_embeddings.post_projs.{N}.weight``    -> oe_projection
            #     (longcat-flash-lite open naming — the canonical Flash-KDA ckpt)
            embed = getattr(self.model, "embed_tokens", None)
            host_oe = getattr(self.model, "ngram_embeddings", None)
            if isinstance(host_oe, HostLongCatOverEmbedding) and (
                ".oe_embed_tokens" in name
                or ".oe_embed_proj" in name
                or ".ngram_embeddings" in name
            ):
                if not host_oe.load_weight(name, loaded_weight):
                    raise ValueError(f"Unexpected Host OE weight {name!r}.")
                continue
            if isinstance(embed, LongCatOverEmbedding) and (
                ".embed_tokens" in name
                or ".oe_embed_tokens" in name
                or ".oe_embed_proj" in name
                or ".ngram_embeddings" in name
            ):
                # OE weights are owned by LongCatOverEmbedding rather than the
                # generic parameter-name path.
                embed.load_weight(name, loaded_weight)
                continue

            # Compressed-tensors MXFP4 routed experts ship the packed weight
            # as ``...w{1,2,3}.weight_packed``; the mxfp4 MoE param is
            # ``w13_weight`` / ``w2_weight`` (packed uint8), so drop the
            # ``_packed`` suffix for the expert loader.
            if "experts." in name and name.endswith(".weight_packed"):
                name = name[: -len(".weight_packed")] + ".weight"
            # KDA conv weights are plain params named ``<qkv>_conv1d_weight``.
            if "_conv1d.weight" in name:
                name = name.replace("_conv1d.weight", "_conv1d_weight")

            if ".shared_experts." in name and name in params_dict:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if ".experts." in name and name not in params_dict:
                    continue  # routed-expert weights handled by moe_loader below
                mapped_name = name.replace(weight_name, param_name)
                if mapped_name.endswith(".bias") and mapped_name not in params_dict:
                    continue
                param = self.get_param(params_dict, mapped_name)
                if param is None:
                    break
                param.weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if moe_loader.matches(name):
                    moe_loader.load(name, loaded_weight)
                    continue

                if fuse_qkv_a_proj and ".g_proj" in name:
                    # MLA output gate (KDA g_proj stacked into qkvgb above):
                    # the per-rank shard sits after [q_a | kv_a+rope] in the
                    # widened fused a-projection.
                    mapped = name.replace("g_proj", "fused_qkv_a_proj_with_mqa")
                    param = params_dict.get(mapped)
                    if param is not None:
                        gate_offset = (
                            config.q_lora_rank
                            + config.kv_lora_rank
                            + config.qk_rope_head_dim
                        )
                        gate_rows = loaded_weight.shape[0] // self.mapping.attn.tp_size
                        gate_start = self.mapping.attn.tp_rank * gate_rows
                        gate_shard = loaded_weight[gate_start : gate_start + gate_rows]
                        param.weight_loader(param, gate_shard, begin_size=gate_offset)
                        continue

                if fuse_qkv_a_proj and (
                    "q_a_proj" in name or "kv_a_proj_with_mqa" in name
                ):
                    # Single targeted replace: chaining ``.replace`` corrupts the
                    # q_a case because ``fused_qkv_a_proj_with_mqa`` (the q_a
                    # result) itself contains ``kv_a_proj_with_mqa`` as a
                    # substring.
                    if "q_a_proj" in name:
                        begin_size = 0
                        mapped = name.replace("q_a_proj", "fused_qkv_a_proj_with_mqa")
                    else:
                        begin_size = config.q_lora_rank
                        mapped = name.replace(
                            "kv_a_proj_with_mqa", "fused_qkv_a_proj_with_mqa"
                        )
                    param = self.get_param(params_dict, mapped)
                    if param is None:
                        continue
                    param.weight_loader(param, loaded_weight, begin_size=begin_size)
                    continue

                if ".shared_experts." in name:
                    if ".gate_proj." in name:
                        shared_name = name.replace(".gate_proj.", ".gate_up_proj.")
                        param = self.get_param(params_dict, shared_name)
                        if param is not None:
                            param.weight_loader(param, loaded_weight, 0)
                            continue
                    if ".up_proj." in name:
                        shared_name = name.replace(".up_proj.", ".gate_up_proj.")
                        param = self.get_param(params_dict, shared_name)
                        if param is not None:
                            param.weight_loader(param, loaded_weight, 1)
                            continue

                if "q_a_proj" in name and name not in params_dict:
                    name = name.replace("q_a_proj", "q_proj")
                param = self.get_param(params_dict, name)
                if param is None:
                    continue
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                try:
                    weight_loader(param, loaded_weight)
                except Exception as exc:
                    raise RuntimeError(
                        "Flash-KDA fallback weight load failed for "
                        f"checkpoint tensor {name!r}: target_param_shape="
                        f"{tuple(param.shape)}, loaded_weight_shape="
                        f"{tuple(loaded_weight.shape)}"
                    ) from exc

        if expected_sources is not None and seen_sources != expected_sources:
            missing = expected_sources - seen_sources
            unexpected = seen_sources - expected_sources
            raise ValueError(
                "Flash-Lite checkpoint stream mismatch: "
                f"missing {len(missing)} source(s), "
                f"unexpected {len(unexpected)} source(s)."
            )
        self.post_load_weights()

    def post_load_weights(self):
        """Prepare the absorbed MLA weights and KDA convolution weight banks.

        Flash-KDA's attention is unquantized (MXFP4 ignores ``self_attn.*``), so
        ``kv_b_proj.weight`` is bf16 and no block dequant is needed — mirrors
        Kimi-K3's post_load_weights. KDA layers have no ``kv_b_proj`` (not
        ``KimiLinearMLAAttention``) and are skipped.

        """
        from tokenspeed.runtime.models.kimi_k3 import KimiLinearMLAAttention

        embed = getattr(self.model, "embed_tokens", None)
        if isinstance(embed, LongCatOverEmbedding):
            embed.validate_loaded_weights()
        host_oe = getattr(self.model, "ngram_embeddings", None)
        if isinstance(host_oe, HostLongCatOverEmbedding):
            host_oe.validate_loaded_weights()

        for layer in self.model.layers:
            self_attn = layer.self_attn
            if isinstance(self_attn, SeparateProjectionKimiLinearMLAAttention):
                self_attn.process_weights_after_loading()
            elif isinstance(self_attn, KimiLinearMLAAttention):
                self_attn.w_kc, self_attn.w_vc = _prepare_mla_kv_b_proj_weights(
                    self_attn.kv_b_proj.weight, self_attn
                )
                if getattr(self.config, "mla_scale_q_lora", False) and hasattr(
                    self_attn, "q_a_layernorm"
                ):
                    self_attn.q_a_layernorm.weight.data *= (
                        self.config.hidden_size / self.config.q_lora_rank
                    ) ** 0.5
                if getattr(self.config, "mla_scale_kv_lora", False):
                    self_attn.kv_a_layernorm.weight.data *= (
                        self.config.hidden_size / self.config.kv_lora_rank
                    ) ** 0.5
            elif isinstance(self_attn, PackedFLASHLocalKDA):
                self_attn.process_weights_after_loading()
            elif isinstance(self_attn, SeperateFLASHLocal):
                self_attn.fuse_conv_weights()
            if isinstance(layer.moe, PackedFLASHLocalMoE):
                layer.moe.process_weights_after_loading()

    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        tp_size = self.mapping.attn.tp_size
        tp_rank = self.mapping.attn.tp_rank
        from tokenspeed.runtime.model_loader.weight_utils import (
            kv_cache_scales_loader as _kv_cache_scales_loader,
        )

        for attn_idx, scaling_factor in _kv_cache_scales_loader(
            quantization_param_path,
            tp_rank,
            tp_size,
            self.config.num_hidden_layers,
            self.config.__class__.model_type,
        ):
            layer_idx = attn_idx
            if not isinstance(self.model.layers[layer_idx], nn.Identity):
                self_attn = self.model.layers[layer_idx].self_attn
                for attn in (
                    getattr(self_attn, "attn_mha", None),
                    getattr(self_attn, "attn_mqa", None),
                ):
                    if attn is not None and hasattr(attn, "k_scale"):
                        attn.k_scale = scaling_factor
                        attn.k_scale_float = scaling_factor

    def get_embed_and_head(self):
        embed = self.model.embed_tokens
        # Only the regular word table participates in weight tying; the OE
        # tables and projection remain model-local.
        if isinstance(embed, LongCatOverEmbedding):
            return embed.word_embedding.weight, self.lm_head.weight
        return embed.weight, self.lm_head.weight

    def get_input_embeddings(self) -> nn.Module:
        """Return the word-only embedding used by DFlash's draft model."""
        embed = self.model.embed_tokens
        if isinstance(embed, LongCatOverEmbedding):
            return embed.word_embedding
        return embed

    def checkpoint_weight_name_filter(self, name: str) -> bool:
        """Skip safetensors shards that contain only non-local OE branches."""
        embed = getattr(self.model, "embed_tokens", None)
        if not isinstance(embed, LongCatOverEmbedding):
            return True
        branch_match = re.search(
            r"(?:oe_embed_(?:tokens|proj)|(?:embedders|post_projs)\.)(\d+)",
            name,
        )
        if branch_match is None:
            return True
        branch = int(branch_match.group(1))
        return any(fragment.branch_id == branch for fragment in embed.spec.fragments)

    def set_embed_and_head(self, embed, head):
        embed_module = self.model.embed_tokens
        if isinstance(embed_module, LongCatOverEmbedding):
            embed_module.word_embedding.weight = embed
        else:
            embed_module.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        return _ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=config.n_routed_experts,
            num_groups=config.moe_group_size,
        )


EntryClass = FLASHLocalForCausalLM
