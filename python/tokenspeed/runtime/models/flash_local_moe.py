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

"""Group-aware FLASHLocal MoE execution strategy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.layers.layernorm import RMSNorm
from tokenspeed.runtime.layers.linear import ReplicatedLinear
from tokenspeed.runtime.layers.moe.types import MoELayerSpec
from tokenspeed.runtime.layers.moe.weights import create_layer_weights
from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig
from tokenspeed.runtime.models.deepseek_v3 import DeepseekV3MLP
from tokenspeed.runtime.utils import add_prefix


@dataclass(frozen=True)
class GMoETopology:
    """Rank decomposition for group-first distributed MoE."""

    ep_group: tuple[int, ...]
    egp_group: tuple[int, ...]
    exchange_group: tuple[int, ...]
    group_id: int
    ep_rank: int
    ep_size: int
    egp_rank: int
    egp_size: int
    num_groups: int


def gmoe_topology(
    *, ep_group: tuple[int, ...], rank: int, num_groups: int
) -> GMoETopology:
    """Split an EP domain into logical expert-group and exchange groups.

    ``EP`` is the complete expert-parallel domain. Ranks are group-major:
    ``ep_rank = group_id * EGP + egp_rank``. A contiguous ``EGP`` group owns
    one logical MoE group. Ranks at the same EGP offset form a strided exchange
    group that moves hidden shards before and after MoE execution.
    """
    ep_group = tuple(ep_group)
    ep_size = len(ep_group)
    if ep_size <= 0 or num_groups <= 0 or ep_size % num_groups or rank not in ep_group:
        raise ValueError("Grouped MoE requires G to divide its valid EP group")
    egp_size = ep_size // num_groups
    ep_rank = ep_group.index(rank)
    group_id, egp_rank = divmod(ep_rank, egp_size)
    egp_start = group_id * egp_size
    egp_group = ep_group[egp_start : egp_start + egp_size]
    exchange_group = tuple(
        ep_group[other_group * egp_size + egp_rank] for other_group in range(num_groups)
    )
    return GMoETopology(
        ep_group=ep_group,
        egp_group=egp_group,
        exchange_group=exchange_group,
        group_id=group_id,
        ep_rank=ep_rank,
        ep_size=ep_size,
        egp_rank=egp_rank,
        egp_size=egp_size,
        num_groups=num_groups,
    )


def gmoe_local_expert_ids(
    *,
    topology: GMoETopology,
    num_experts_per_group: int,
) -> tuple[int, ...]:
    """Return flat checkpoint IDs owned by a group-first rank."""
    if num_experts_per_group <= 0 or num_experts_per_group % topology.egp_size:
        raise ValueError("One MoE group's experts must divide across its EGP ranks")
    local_count = num_experts_per_group // topology.egp_size
    start = topology.group_id * num_experts_per_group + topology.egp_rank * local_count
    return tuple(range(start, start + local_count))


class GroupAwareFlashLocalMoE(nn.Module):
    """Group-aware FLASHLocal MoE execution strategy.

    This strategy assigns each rank to one logical MoE group and exchanges
    group-hidden shards before and after expert execution. Selection is an
    explicit model strategy and is independent of the accelerator type.
    """

    def __init__(
        self,
        config: Any,
        mapping: Any,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        group_hidden = config.hidden_size // config.moe_group_size
        intermediate = config.expert_ffn_hidden_size
        self.config = config
        self.mapping = mapping
        self.group_hidden = group_hidden
        self.moe_group_size = config.moe_group_size
        self.n_routed_experts = config.n_routed_experts
        self.zero_expert_num = config.zero_expert_num
        self.quant_config = quant_config
        self.prefix = prefix
        self.renormalize_topk = bool(getattr(config, "norm_topk_prob", False))
        ep_group = tuple(mapping.moe.ep_group)
        if len(ep_group) < config.moe_group_size:
            raise ValueError("Group-aware MoE requires EP >= G")
        if int(getattr(mapping.moe, "tp_size", 1)) != 1:
            raise ValueError("Group-first Lite MoE requires moe_tp_size=1")
        self.topology = gmoe_topology(
            ep_group=ep_group,
            rank=int(getattr(mapping, "rank", mapping.moe.ep_rank)),
            num_groups=config.moe_group_size,
        )
        self.local_expert_ids = gmoe_local_expert_ids(
            topology=self.topology,
            num_experts_per_group=config.n_routed_experts,
        )
        experts_per_router = config.n_routed_experts + config.zero_expert_num
        self.router = nn.Module()
        self.router.classifier = ReplicatedLinear(
            group_hidden,
            experts_per_router,
            bias=False,
            params_dtype=torch.float32,
            quant_config=None,
        )
        self.router.e_score_correction_bias = nn.Parameter(
            torch.empty(experts_per_router, dtype=torch.float32),
            requires_grad=False,
        )

        self.experts = nn.Module()
        expert_spec = MoELayerSpec(
            top_k=config.moe_topk,
            num_experts=config.n_routed_experts,
            num_local_experts=len(self.local_expert_ids),
            hidden_size=group_hidden,
            intermediate_size=intermediate,
            activation="silu",
            tp_rank=0,
            tp_size=1,
            # MoELayerSpec calls the expert-owning leaf group EP. In the
            # group-aware model topology that leaf group is EGP.
            ep_rank=self.topology.egp_rank,
            ep_size=self.topology.egp_size,
        )
        for field in (
            "num_experts",
            "num_local_experts",
            "hidden_size",
            "intermediate_size",
            "ep_rank",
            "ep_size",
        ):
            setattr(self.experts, field, getattr(expert_spec, field))
        self._quant_kind = (
            "unquant" if quant_config is None else quant_config.moe_weight_dtype(prefix)
        )
        create_layer_weights(
            expert_spec,
            self.experts,
            self._quant_kind,
            quant_config,
        )

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.proj_input = nn.Linear(
            config.hidden_size, config.hidden_size, bias=False, dtype=torch.bfloat16
        )
        self.proj_output = nn.Linear(
            config.hidden_size, config.hidden_size, bias=False, dtype=torch.bfloat16
        )
        shared_count = getattr(
            config, "n_shared_experts", getattr(config, "num_shared_experts", 1)
        )
        self.shared_experts = DeepseekV3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=int(config.ffn_hidden_size) * int(shared_count),
            hidden_act="silu",
            mapping=Mapping(rank=0),
            quant_config=quant_config if self._quant_kind == "int8" else None,
            prefix=add_prefix("shared_experts", prefix),
            is_shared_expert=False,
            params_dtype=torch.bfloat16,
        )
        self._moe_plan: dict[str, Any] | None = None
        self._moe_stages = None
        self._moe_stage_context = None

    def get_moe_routed_weights(self) -> list[torch.Tensor]:
        return [self.experts.w13_weight.data, self.experts.w2_weight.data]

    def load_group_router_weight(
        self, group_id: int, field: str, loaded_weight: torch.Tensor
    ) -> None:
        """Load the checkpoint router owned by this rank's expert group."""
        if not 0 <= group_id < self.moe_group_size:
            raise ValueError(f"Invalid FLASHLocal router group {group_id}.")
        if group_id != self.topology.group_id:
            return
        if field == "classifier.weight":
            target = self.router.classifier.weight
        elif field == "e_score_correction_bias":
            target = self.router.e_score_correction_bias
        else:
            target = None
        if target is None or tuple(target.shape) != tuple(loaded_weight.shape):
            raise ValueError(
                f"Invalid FLASHLocal router tensor {field!r} for group {group_id}."
            )
        target.data.copy_(loaded_weight.to(device=target.device))

    def process_weights_after_loading(self, _module: nn.Module | None = None) -> None:
        if self._moe_plan is not None:
            return

        import tokenspeed_kernel

        try:
            self._moe_plan = tokenspeed_kernel.moe_plan(
                self._quant_kind,
                input_dtype=torch.bfloat16,
                activation="silu",
                routing_mode="precomputed_topk",
                ep_size=self.experts.ep_size,
                num_zero_experts=self.zero_expert_num,
                ispp=self.config.expert_ffn_hidden_size,
                internal_activation_dtype=(
                    "int8" if self._quant_kind == "int8" else "input"
                ),
                solution=getattr(self.config, "gmoe_expert_solution", None),
            )
        except tokenspeed_kernel.NoKernelFoundError:
            if getattr(self.config, "gmoe_expert_solution", None) is not None:
                raise
            return
        tokenspeed_kernel.moe_process_weights(self._moe_plan, self.experts)
        topology = self.topology
        from tokenspeed.runtime.distributed.process_group_manager import (
            process_group_manager as pg_manager,
        )

        pg_manager.init_process_group(topology.exchange_group)
        if topology.egp_size > 1:
            pg_manager.init_process_group(topology.egp_group)

        from tokenspeed_kernel.ops.moe.gmoe import (
            GMoEContext,
            select_gmoe_stages,
        )

        from tokenspeed.runtime.distributed.comm_ops import (
            all_to_all_single,
            token_all_gather,
            token_reduce_scatter,
        )

        self._moe_stages = select_gmoe_stages(
            input_dtype=torch.bfloat16,
            traits={
                "hidden_size": self.config.hidden_size,
                "num_groups": topology.num_groups,
                "egp_size": topology.egp_size,
                "weight_dtype": self._quant_kind,
                "shared_gate_weight_dtype": self.shared_experts.gate_up_proj.weight.dtype,
                "shared_down_weight_dtype": self.shared_experts.down_proj.weight.dtype,
                "top_k": self.config.moe_topk,
                "num_experts": self.n_routed_experts,
                "renormalize_topk": self.renormalize_topk,
                "gmoe_exchange_enabled": bool(
                    getattr(self.config, "gmoe_exchange_options", None)
                ),
            },
            pre_solution=getattr(self.config, "gmoe_pre_solution", None),
            post_solution=getattr(self.config, "gmoe_post_solution", None),
        )
        self._moe_stage_context = GMoEContext(
            num_groups=topology.num_groups,
            egp_group=topology.egp_group,
            egp_rank=topology.egp_rank,
            exchange_group=topology.exchange_group,
            num_experts=self.n_routed_experts,
            norm_scale=self.config.grouped_moe_norm_scale,
            proj_input=self.proj_input,
            norm=self.norm,
            shared_experts=self.shared_experts,
            proj_output=self.proj_output,
            route=self._route,
            router=self.router,
            top_k=self.config.moe_topk,
            routed_scaling_factor=self.config.routed_scaling_factor,
            renormalize_topk=self.renormalize_topk,
            all_to_all=all_to_all_single,
            all_gather=token_all_gather,
            reduce_scatter=token_reduce_scatter,
            ep_group=tuple(self.mapping.moe.ep_group),
            create_exchange_group=pg_manager.get_dedicated_device_group,
        )
        self._moe_stage_context = self._moe_stages.prepare(
            self._moe_stage_context,
            device=self.proj_input.weight.device,
            options=getattr(self.config, "gmoe_exchange_options", None),
        )

    def release_moe_resources(self) -> None:
        """Release exchange resources collectively after all graphs are destroyed.

        Layers in this EP domain share the resource; calling on one layer closes
        it for all of them. No subsequent forward/replay is permitted.
        """
        context = self._moe_stage_context
        if context is not None and context.exchange is not None:
            context.exchange.close()

    def _route(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Route the single logical group owned by this rank."""
        logits, _ = self.router.classifier(hidden_states.float())
        import tokenspeed_kernel

        weights, ids = tokenspeed_kernel.moe_softmax_bias_topk(
            logits,
            self.router.e_score_correction_bias,
            self.config.moe_topk,
            routed_scaling_factor=self.config.routed_scaling_factor,
        )
        if self.renormalize_topk:
            weights = (
                weights
                / weights.sum(dim=-1, keepdim=True)
                * self.config.routed_scaling_factor
            )
        return weights, ids.to(torch.int32)

    def forward(
        self,
        hidden_states: torch.Tensor,
        num_global_tokens: int = 0,
        max_num_tokens_per_gpu: int = 0,
        ctx: Any | None = None,
    ) -> torch.Tensor:
        if hidden_states.shape[0] == 0:
            return hidden_states
        if ctx is None or ctx.forward_mode is None:
            raise ValueError("Distributed group-aware MoE requires ForwardContext")

        import tokenspeed_kernel

        topology = self.topology
        plan = self._moe_plan
        if plan is None:
            raise RuntimeError("FLASHLocal group-first MoE has no kernel plan")
        if self._moe_stages is None or self._moe_stage_context is None:
            raise RuntimeError("FLASHLocal group-first MoE stages are not prepared")
        inputs = self._moe_stages.pre(
            hidden_states=hidden_states, context=self._moe_stage_context
        )
        received = inputs.received
        topk_weights, topk_ids = inputs.topk_weights, inputs.topk_ids

        # The init-routing -> finalize-routing kernel plan and leaf are unchanged.
        routed = tokenspeed_kernel.moe_apply(
            plan,
            received,
            self.experts,
            topk_weights,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            num_tokens_global=num_global_tokens,
            max_num_tokens_per_gpu=(
                max_num_tokens_per_gpu * topology.num_groups
                if max_num_tokens_per_gpu
                else 0
            ),
        )
        return self._moe_stages.post(
            routed=routed, inputs=inputs, context=self._moe_stage_context
        )


__all__ = [
    "GroupAwareFlashLocalMoE",
    "gmoe_local_expert_ids",
    "gmoe_topology",
]
