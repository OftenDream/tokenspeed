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

"""Selectable group-first stages using optional 910B fused communication."""

import math
from dataclasses import replace

import torch
from tokenspeed_kernel.ops.moe.gmoe import GMoEInputs
from tokenspeed_kernel.platform import CapabilityRequirement, current_platform
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

if current_platform().is_npu:
    from tokenspeed_kernel_npu.ops.fused_ffn import prepare_shared_ffn
    from tokenspeed_kernel_npu.ops.gmoe_comm import (
        prepare_gmoe_exchange,
    )

    def prepare(context, *, device, options):
        if context.exchange is not None:
            return context
        exchange = prepare_gmoe_exchange(
            ep_group=context.ep_group,
            num_groups=context.num_groups,
            num_experts=context.num_experts,
            top_k=context.top_k,
            device=device,
            create_group=context.create_exchange_group,
            options=options,
        )
        return replace(context, exchange=exchange)

    def prepare_router(context, *, device, options):
        weight = context.router.classifier.weight
        bias = context.router.e_score_correction_bias
        if (
            weight.ndim != 2
            or not 1 <= weight.shape[0] <= 1024
            or not 64 <= weight.shape[1] <= 8192
            or weight.shape[1] % 64
            or weight.dtype != torch.float32
            or not weight.is_contiguous()
            or bias.shape != (weight.shape[0],)
            or bias.dtype != torch.float32
            or not bias.is_contiguous()
            or weight.device != bias.device
            or weight.device != torch.device(device)
        ):
            raise ValueError(
                "Fused router requires contiguous FP32 classifier [N,H] and bias [N], N<=1024, H=64..8192 aligned to 64"
            )
        if not 1 <= context.top_k <= min(weight.shape[0], 64):
            raise ValueError("Fused router requires top_k in [1,min(N,64)]")
        scale = context.routed_scaling_factor
        if not math.isfinite(scale) or not 0 < scale <= torch.finfo(torch.float32).max:
            raise ValueError("Fused router scale must be finite positive FP32")
        options = {"router_cores": 32, **options}
        if not options["router_cores"]:
            raise ValueError("Fused router requires nonzero router_cores")
        return prepare(context, device=device, options=options)

    def prepare_router_ffn(context, *, device, options):
        options = dict(options)
        binding = options.pop("ffn_binding", None)
        shared = prepare_shared_ffn(context.shared_experts, binding=binding)
        context = prepare_router(context, device=device, options=options)
        return replace(context, prepared_shared=shared)

    _REGISTRATION = dict(
        solution="flash_npu",
        capability=CapabilityRequirement(vendors=frozenset({"ascend"})),
        signatures=format_signatures("hidden_states", "dense", {torch.bfloat16}),
        traits={"gmoe_exchange_enabled": frozenset({True})},
        priority=Priority.SPECIALIZED,
        tags={"ascend", "npu_graph"},
    )

    @register_kernel("moe", "gmoe_pre", name="flash_npu_gmoe_pre", **_REGISTRATION)
    def gmoe_pre(*, hidden_states, context):
        """Retain shared/projection/norm/router; fuse only layout + A2A + AG."""
        return _pre(hidden_states, context, fused_router=False)

    @register_kernel(
        "moe",
        "gmoe_pre",
        name="flash_npu_router_gmoe_pre",
        **{**_REGISTRATION, "solution": "flash_npu_router", "priority": 0},
    )
    def gmoe_router_pre(*, hidden_states, context):
        """Overlap local FP32 routing with hidden AG and shared-expert compute."""
        return _pre(hidden_states, context, fused_router=True)

    @register_kernel(
        "moe",
        "gmoe_pre",
        name="flash_npu_router_ffn_gmoe_pre",
        **{**_REGISTRATION, "solution": "flash_npu_router_ffn", "priority": 0},
    )
    def gmoe_router_ffn_pre(*, hidden_states, context):
        """Use the fused shared FFN on the existing dispatch-overlap stream."""
        return _pre(hidden_states, context, fused_router=True, fused_shared=True)

    def _pre(hidden_states, context, *, fused_router, fused_shared=False):
        exchange = context.exchange
        exchange.check_execution_stream()
        tokens = hidden_states.shape[0]
        hidden = hidden_states.shape[1] // context.num_groups
        if fused_router:
            exchange.check_router_shape(tokens, hidden, context.top_k)
        else:
            exchange.check_shape(tokens, hidden)
        projected = context.proj_input(hidden_states)
        # Fork before norm/scale, but after projection: overlapping both GEMMs
        # regresses the limited shared stream on the validated decode workload.
        shared_output, shared_wait = exchange.run_shared(
            context.prepared_shared if fused_shared else context.shared_experts,
            hidden_states,
        )
        projected = context.norm(projected)
        grouped = (projected * context.norm_scale).view(tokens, context.num_groups, -1)
        if fused_router:
            received, weights, ids = exchange.exchange_router(
                grouped,
                context.router,
                context.top_k,
                context.routed_scaling_factor,
                context.renormalize_topk,
            )
        else:
            received = exchange.exchange(grouped)
            weights, ids = context.route(received)
        local_rows = tokens * context.num_groups
        local_received = received.narrow(0, context.egp_rank * local_rows, local_rows)
        # Join dispatch/shared before the unchanged init-routing expert leaf.
        # Shared work never overlaps the later zero-expert/combine kernel.
        if shared_wait is not None:
            shared_wait()
        return GMoEInputs(received, local_received, weights, ids, shared_output)

    @register_kernel("moe", "gmoe_post", name="flash_npu_gmoe_post", **_REGISTRATION)
    def gmoe_post(*, routed, inputs, context):
        """Fuse RS + identity + A2A + layout, then retain projection/shared add."""
        rows = inputs.local_received.shape[0]
        start = context.egp_rank * rows
        restored = context.exchange.restore_zero(
            routed,
            inputs.local_received,
            inputs.topk_weights.narrow(0, start, rows),
            inputs.topk_ids.narrow(0, start, rows),
        )
        return context.proj_output(restored) + inputs.shared_output

    gmoe_pre.prepare = prepare
    gmoe_router_pre.prepare = prepare_router
    gmoe_router_ffn_pre.prepare = prepare_router_ffn
    gmoe_post.prepare = prepare
