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
"""Existing operator composition for the group-first pre/post boundaries."""

import torch
from tokenspeed_kernel.ops.moe.gmoe import (
    GMoEContext,
    GMoEInputs,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures


@register_kernel(
    "moe",
    "gmoe_pre",
    name="composed_gmoe_pre",
    solution="composed",
    signatures=format_signatures("hidden_states", "dense", {torch.bfloat16}),
    priority=Priority.PORTABLE,
)
def gmoe_pre(*, hidden_states: torch.Tensor, context: GMoEContext) -> GMoEInputs:
    """Run shared MLP and all input preparation up to (not including) init routing."""
    shared_output = context.shared_experts(hidden_states)
    tokens = hidden_states.shape[0]
    projected = context.norm(context.proj_input(hidden_states))
    grouped = (projected * context.norm_scale).view(tokens, context.num_groups, -1)
    exchange_input = grouped.permute(1, 0, 2).contiguous().flatten(0, 1)
    received = torch.empty_like(exchange_input)
    context.all_to_all(received, exchange_input, context.exchange_group)
    local_received = received
    token_splits = [received.shape[0]] * len(context.egp_group)
    if len(context.egp_group) > 1:
        received = context.all_gather(received, context.egp_group, token_splits)
    topk_weights, topk_ids = context.route(received)
    return GMoEInputs(received, local_received, topk_weights, topk_ids, shared_output)


@register_kernel(
    "moe",
    "gmoe_post",
    name="composed_gmoe_post",
    solution="composed",
    signatures=format_signatures("hidden_states", "dense", {torch.bfloat16}),
    priority=Priority.PORTABLE,
)
def gmoe_post(
    *,
    routed: torch.Tensor,
    inputs: GMoEInputs,
    context: GMoEContext,
) -> torch.Tensor:
    """Run return communication, identity experts, output projection and shared add."""
    topk_weights, topk_ids = inputs.topk_weights, inputs.topk_ids
    token_splits = [inputs.local_received.shape[0]] * len(context.egp_group)
    if len(context.egp_group) > 1:
        routed = context.reduce_scatter(routed, context.egp_group, token_splits)
        local_start = context.egp_rank * token_splits[0]
        topk_weights = topk_weights.narrow(0, local_start, token_splits[0])
        topk_ids = topk_ids.narrow(0, local_start, token_splits[0])
    identity_weight = torch.where(
        topk_ids >= context.num_experts,
        topk_weights,
        torch.zeros_like(topk_weights),
    ).sum(dim=-1, keepdim=True)
    routed = routed + inputs.local_received * identity_weight.to(
        inputs.local_received.dtype
    )
    restored = torch.empty_like(routed)
    context.all_to_all(restored, routed.contiguous(), context.exchange_group)
    restored = (
        restored.view(context.num_groups, inputs.shared_output.shape[0], -1)
        .permute(1, 0, 2)
        .flatten(1)
    )
    return context.proj_output(restored) + inputs.shared_output
