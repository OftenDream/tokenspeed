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

"""Profile Lite Group-Aware BF16 or W8A8 Decode on one Ascend host.

Run with ``torchrun --nproc-per-node=8|16``. The script profiles the production
group-first exchange/EGP schedule and imports the matching production expert
leaf. It emits both an eager trace with recorded tensor shapes and an ACL Graph
replay trace for each rank.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch_npu
from tokenspeed_kernel_npu.ops.moe import (
    ascend_bf16_precomputed_moe_apply,
    ascend_bf16_process_moe_weights,
    ascend_int8_precomputed_moe_apply,
    ascend_int8_process_moe_weights,
)

GROUPS = 8
TOKENS_PER_RANK = 32
MODEL_HIDDEN = 4096
GROUP_HIDDEN = MODEL_HIDDEN // GROUPS
INTERMEDIATE = 1024
EXPERTS = 384
ZERO_EXPERTS = 32
TOP_K = 16
REAL_TOP_K = 10


def _new_groups(ep_size: int, rank: int):
    egp_size = ep_size // GROUPS
    egp_groups = []
    for group_id in range(GROUPS):
        ranks = list(range(group_id * egp_size, (group_id + 1) * egp_size))
        group = dist.new_group(ranks)
        if rank in ranks:
            egp_groups.append((group_id, ranks, group))
    exchange_groups = []
    for egp_rank in range(egp_size):
        ranks = [group_id * egp_size + egp_rank for group_id in range(GROUPS)]
        group = dist.new_group(ranks)
        if rank in ranks:
            exchange_groups.append((egp_rank, ranks, group))
    assert len(egp_groups) == len(exchange_groups) == 1
    group_id, egp_ranks, egp_group = egp_groups[0]
    egp_rank, exchange_ranks, exchange_group = exchange_groups[0]
    return group_id, egp_rank, egp_ranks, egp_group, exchange_ranks, exchange_group


def _make_experts(
    local_experts: int,
    egp_rank: int,
    device: torch.device,
    weight_dtype: str,
):
    torch.manual_seed(13 + egp_rank)
    experts = SimpleNamespace(
        num_local_experts=local_experts,
        num_experts=EXPERTS,
        hidden_size=GROUP_HIDDEN,
        intermediate_size=INTERMEDIATE,
        ep_rank=egp_rank,
        ep_size=EXPERTS // local_experts,
    )
    weight_shape = (local_experts, 2 * INTERMEDIATE, GROUP_HIDDEN)
    w2_shape = (local_experts, GROUP_HIDDEN, INTERMEDIATE)
    if weight_dtype == "bf16":
        experts.w13_weight = torch.nn.Parameter(
            torch.randn(weight_shape, dtype=torch.bfloat16, device=device) * 0.01,
            requires_grad=False,
        )
        experts.w2_weight = torch.nn.Parameter(
            torch.randn(w2_shape, dtype=torch.bfloat16, device=device) * 0.01,
            requires_grad=False,
        )
        ascend_bf16_process_moe_weights(plan={}, w=experts)
    else:
        experts.w13_weight = torch.nn.Parameter(
            torch.randint(-8, 8, weight_shape, dtype=torch.int8, device=device),
            requires_grad=False,
        )
        experts.w2_weight = torch.nn.Parameter(
            torch.randint(-8, 8, w2_shape, dtype=torch.int8, device=device),
            requires_grad=False,
        )
        experts.w13_weight_scale = torch.nn.Parameter(
            torch.full(
                (local_experts, 2 * INTERMEDIATE),
                1.0 / 128.0,
                dtype=torch.float32,
                device=device,
            ),
            requires_grad=False,
        )
        experts.w2_weight_scale = torch.nn.Parameter(
            torch.full(
                (local_experts, GROUP_HIDDEN),
                1.0 / 128.0,
                dtype=torch.bfloat16,
                device=device,
            ),
            requires_grad=False,
        )
        ascend_int8_process_moe_weights(plan={}, w=experts)
    return experts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ep-size", type=int, choices=(8, 16), required=True)
    parser.add_argument("--weight-dtype", choices=("bf16", "w8a8"), default="w8a8")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--graph-iterations", type=int, default=20)
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != args.ep_size or args.ep_size % GROUPS:
        raise ValueError("WORLD_SIZE must equal EP and be divisible by G=8")
    torch.npu.set_device(local_rank)
    device = torch.device(f"npu:{local_rank}")
    dist.init_process_group("hccl")
    control_group = dist.new_group(list(range(world_size)))
    (
        group_id,
        egp_rank,
        egp_ranks,
        egp_group,
        exchange_ranks,
        exchange_group,
    ) = _new_groups(args.ep_size, rank)
    egp_size = args.ep_size // GROUPS
    local_experts = EXPERTS // egp_size

    experts = _make_experts(local_experts, egp_rank, device, args.weight_dtype)
    torch.manual_seed(17 + group_id)
    router_weight = (
        torch.randn(
            EXPERTS + ZERO_EXPERTS,
            GROUP_HIDDEN,
            dtype=torch.float32,
            device=device,
        )
        * 0.01
    )
    correction_bias = torch.zeros(
        EXPERTS + ZERO_EXPERTS, dtype=torch.float32, device=device
    )
    # Reproduce target_topk=10 inside moe_topk=16: six identity routes.
    correction_bias[EXPERTS : EXPERTS + TOP_K - REAL_TOP_K] = 100.0
    torch.manual_seed(23 + rank)
    hidden_states = torch.randn(
        TOKENS_PER_RANK, MODEL_HIDDEN, dtype=torch.bfloat16, device=device
    )
    plan = {"activation": "silu", "num_zero_experts": ZERO_EXPERTS}
    moe_apply = (
        ascend_bf16_precomputed_moe_apply
        if args.weight_dtype == "bf16"
        else ascend_int8_precomputed_moe_apply
    )

    def run():
        grouped = hidden_states.view(TOKENS_PER_RANK, GROUPS, GROUP_HIDDEN)
        exchange_input = grouped.permute(1, 0, 2).contiguous().flatten(0, 1)
        received = torch.empty_like(exchange_input)
        dist.all_to_all_single(received, exchange_input, group=exchange_group)
        local_received = received
        if egp_size > 1:
            gathered = torch.empty(
                egp_size * received.shape[0],
                GROUP_HIDDEN,
                dtype=received.dtype,
                device=device,
            )
            dist.all_gather_into_tensor(gathered, received, group=egp_group)
            received = gathered

        logits = F.linear(received.float(), router_weight)
        topk_weights, topk_ids, _ = torch_npu.npu_moe_gating_top_k(
            logits,
            TOP_K,
            bias=correction_bias,
            k_group=1,
            group_count=1,
            group_select_mode=0,
            renorm=0,
            norm_type=0,
            out_flag=False,
            routed_scaling_factor=6.0,
            eps=1e-20,
        )
        topk_ids = topk_ids.to(torch.int32)
        routed = moe_apply(
            plan,
            received,
            experts,
            topk_weights,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
        )
        if egp_size > 1:
            reduced = torch.empty_like(local_received)
            dist.reduce_scatter_tensor(reduced, routed, group=egp_group)
            routed = reduced
            start = egp_rank * local_received.shape[0]
            topk_weights = topk_weights.narrow(0, start, local_received.shape[0])
            topk_ids = topk_ids.narrow(0, start, local_received.shape[0])
        identity_weight = torch.where(
            topk_ids >= EXPERTS,
            topk_weights,
            torch.zeros_like(topk_weights),
        ).sum(dim=-1, keepdim=True)
        routed = routed + local_received * identity_weight.to(local_received.dtype)

        restored = torch.empty_like(routed)
        dist.all_to_all_single(restored, routed.contiguous(), group=exchange_group)
        return (
            restored.view(GROUPS, TOKENS_PER_RANK, GROUP_HIDDEN)
            .permute(1, 0, 2)
            .flatten(1)
        )

    for _ in range(args.warmups):
        output = run()
    torch.npu.synchronize()
    dist.barrier(group=control_group)
    if not torch.isfinite(output).all():
        raise RuntimeError("non-finite Group-Aware output")

    output_dir = args.output.resolve()
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier(group=control_group)
    raw_dir = output_dir / f"rank{rank:02d}-raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(raw_dir)
    activities = [
        torch_npu.profiler.ProfilerActivity.CPU,
        torch_npu.profiler.ProfilerActivity.NPU,
    ]

    eager_trace = output_dir / f"rank{rank:02d}-eager-shapes.json"
    dist.barrier(group=control_group)
    with torch_npu.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=False,
        with_stack=False,
    ) as profiler:
        with torch.profiler.record_function(
            f"lite_gmoe_{args.weight_dtype}_decode_eager"
        ):
            output = run()
            torch.npu.synchronize()
    profiler.export_chrome_trace(str(eager_trace))

    dist.barrier(group=control_group)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(
        graph,
        stream=torch.npu.Stream(device=device),
        auto_dispatch_capture=True,
    ):
        graph_output = run()
    for _ in range(args.warmups):
        graph.replay()
    torch.npu.synchronize()
    dist.barrier(group=control_group)

    graph_trace = output_dir / f"rank{rank:02d}-acl-graph.json"
    with torch_npu.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=False,
        with_stack=False,
    ) as profiler:
        dist.barrier(group=control_group)
        with torch.profiler.record_function(
            f"lite_gmoe_{args.weight_dtype}_decode_graph"
        ):
            for _ in range(args.graph_iterations):
                graph.replay()
            torch.npu.synchronize()
    profiler.export_chrome_trace(str(graph_trace))
    if not torch.isfinite(graph_output).all():
        raise RuntimeError("non-finite ACL Graph output")

    metadata = {
        "rank": rank,
        "weight_dtype": args.weight_dtype,
        "ep_size": args.ep_size,
        "groups": GROUPS,
        "egp_size": egp_size,
        "group_id": group_id,
        "egp_rank": egp_rank,
        "egp_group": egp_ranks,
        "exchange_group": exchange_ranks,
        "tokens_per_rank": TOKENS_PER_RANK,
        "model_hidden": MODEL_HIDDEN,
        "group_hidden": GROUP_HIDDEN,
        "tokens_after_exchange": GROUPS * TOKENS_PER_RANK,
        "tokens_after_egp_gather": egp_size * GROUPS * TOKENS_PER_RANK,
        "experts": EXPERTS,
        "local_experts": local_experts,
        "zero_experts": ZERO_EXPERTS,
        "top_k": TOP_K,
        "real_top_k": REAL_TOP_K,
        "intermediate": INTERMEDIATE,
        "w13_shape": list(experts.w13_weight.shape),
        "w2_shape": list(experts.w2_weight.shape),
        "w13_format": str(torch_npu.get_npu_format(experts.w13_weight)),
        "w2_format": str(torch_npu.get_npu_format(experts.w2_weight)),
        "output_shape": list(graph_output.shape),
        "graph_iterations": args.graph_iterations,
        "eager_trace": eager_trace.name,
        "graph_trace": graph_trace.name,
    }
    (output_dir / f"rank{rank:02d}.metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    dist.barrier(group=control_group)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
