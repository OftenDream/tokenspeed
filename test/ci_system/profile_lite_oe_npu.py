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

"""Capture the Lite OE decode path as one ACL Graph on 16 NPUs.

Run with::

    torchrun --standalone --nproc-per-node=16 \
      test/ci_system/profile_lite_oe_npu.py --table-placement host \
      --output /tmp/lite-oe-profile

The two DP replicas use independent TP8 HCCL groups. One graph starts with a
16-rank HCCL synchronization and then contains 50 consecutive iterations of
fused append/hash/lookup, the portable Torch projection/merge, and TP8
all-reduce. Each TP rank contributes 32 tokens, so the post-gather OE input on
every rank has 256 rows. The in-graph synchronization absorbs process-side
launch skew before either replica starts touching the shared Host tables.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path

import torch
import torch.distributed as dist
import torch_npu
from tokenspeed_kernel.ops.over_embedding import (
    append_packed_lookup_,
    project_add_word_,
    register_host_tables_,
)

from tokenspeed.runtime.layers.over_embedding import resolve_longcat_oe_spec

WORLD_SIZE = 16
TP_SIZE = 8
LOCAL_TOKENS_PER_RANK = 32
TOKEN_COUNT = TP_SIZE * LOCAL_TOKENS_PER_RANK
PROFILE_HIDDEN_SIZE = 4096
HISTORY_CAPACITY = 32


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--graph-iterations", type=int, default=50)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--table-placement", choices=("host", "device"), required=True)
    parser.add_argument(
        "--table-rows",
        type=int,
        default=0,
        help="Override both moduli for a quick smoke test; zero uses checkpoint sizes.",
    )
    return parser.parse_args()


def _create_tp_groups(rank: int) -> tuple[dist.ProcessGroup, tuple[int, ...]]:
    groups: list[tuple[dist.ProcessGroup, tuple[int, ...]]] = []
    for dp_rank in range(2):
        ranks = tuple(range(dp_rank * TP_SIZE, (dp_rank + 1) * TP_SIZE))
        groups.append((dist.new_group(ranks=list(ranks), backend="hccl"), ranks))
    return groups[rank // TP_SIZE]


def _tables(spec, rank: int, placement: str) -> tuple[torch.Tensor, ...]:
    tables = []
    for fragment in spec.fragments:
        # Keep the real row-major table layout used by the checkpoint mapping.
        storage = torch.empty(
            (fragment.modulus, spec.branch_width), dtype=torch.bfloat16
        )
        storage.fill_((rank + fragment.branch_id + 1) / 256.0)
        fragment_table = storage[:, fragment.feature_begin : fragment.feature_end]
        if placement == "device":
            fragment_table = fragment_table.npu().contiguous()
        tables.append(fragment_table)
    return tuple(tables)


def main() -> None:
    args = _arguments()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != WORLD_SIZE:
        raise RuntimeError(f"expected WORLD_SIZE={WORLD_SIZE}, got {world_size}")
    if args.graph_iterations != 50:
        raise ValueError("the contention profile must capture exactly 50 iterations")

    device = torch.device("npu", local_rank)
    torch.npu.set_device(device)
    dist.init_process_group(backend="hccl")
    tp_group, tp_ranks = _create_tp_groups(rank)
    control_group = dist.new_group(ranks=list(range(WORLD_SIZE)), backend="gloo")
    tp_rank = rank % TP_SIZE

    spec = resolve_longcat_oe_spec(
        vocab_size=163840,
        hidden_size=PROFILE_HIDDEN_SIZE,
        max_ngram_order=5,
        hashes_per_order=4,
        modulus0=9765520,
        tp_size=TP_SIZE,
        tp_rank=tp_rank,
    )
    if args.table_rows:
        spec = replace(
            spec,
            fragments=tuple(
                replace(fragment, modulus=args.table_rows + index * 2)
                for index, fragment in enumerate(spec.fragments)
            ),
        )
    tables = _tables(spec, rank, args.table_placement)
    if args.table_placement == "host":
        register_host_tables_(tables, device=device, solution="flash_npu_kernel")

    input_ids = torch.arange(
        1000 + rank * TOKEN_COUNT,
        1000 + (rank + 1) * TOKEN_COUNT,
        dtype=torch.int32,
        device=device,
    )
    offsets = torch.arange(TOKEN_COUNT + 1, dtype=torch.int32, device=device)
    slots = torch.arange(TOKEN_COUNT, dtype=torch.int64, device=device)
    active = torch.ones(TOKEN_COUNT, dtype=torch.bool, device=device)
    history = torch.zeros(
        (TOKEN_COUNT, HISTORY_CAPACITY), dtype=torch.int32, device=device
    )
    history[:, :3] = torch.arange(11, 14, dtype=torch.int32, device=device)
    committed = torch.full((TOKEN_COUNT,), 3, dtype=torch.int32, device=device)
    activation = torch.empty(
        (TOKEN_COUNT, spec.local_width), dtype=torch.bfloat16, device=device
    )
    projection = (
        torch.randn(
            (spec.local_width, spec.hidden_size), dtype=torch.bfloat16, device=device
        )
        / spec.local_width**0.5
    )
    word = torch.randn(
        (TOKEN_COUNT, spec.hidden_size), dtype=torch.bfloat16, device=device
    )
    bypass_mask = torch.zeros(TOKEN_COUNT, dtype=torch.bool, device=device)
    graph_start_signal = torch.ones((), dtype=torch.int32, device=device)

    def run() -> None:
        append_packed_lookup_(
            input_ids,
            offsets,
            slots,
            active,
            history,
            committed,
            tables,
            spec=spec,
            out=activation,
            solution="flash_npu_kernel",
            enable_pdl=False,
        )
        project_add_word_(
            word,
            activation,
            projection,
            scale=float(spec.scale),
            bypass_mask=bypass_mask,
            solution="torch",
        )
        dist.all_reduce(word, group=tp_group)

    for _ in range(args.warmups):
        run()
    torch.npu.synchronize()
    dist.barrier(group=control_group)

    graph = torch.npu.NPUGraph()
    capture_stream = torch.npu.Stream(device=device)
    with torch.npu.graph(
        graph,
        stream=capture_stream,
        auto_dispatch_capture=True,
    ):
        # A CPU barrier cannot prevent rank skew between returning from the
        # barrier and submitting this graph. Put the rendezvous on device so
        # the first Host-table lookup starts only after all 16 ranks arrive.
        dist.all_reduce(graph_start_signal)
        for _ in range(args.graph_iterations):
            run()
    for _ in range(args.warmups):
        graph.replay()
    torch.npu.synchronize()
    dist.barrier(group=control_group)

    output_dir = args.output.resolve()
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier(group=control_group)

    # torch_npu stages raw CANN data below the current working directory when
    # export_chrome_trace is used.  Isolate that staging area per process;
    # otherwise concurrent ranks can accidentally export another rank's data.
    raw_dir = output_dir / f"rank{rank:02d}-raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(raw_dir)

    activities = [
        torch_npu.profiler.ProfilerActivity.CPU,
        torch_npu.profiler.ProfilerActivity.NPU,
    ]
    trace_path = output_dir / f"rank{rank:02d}.json"
    torch.npu.synchronize()
    dist.barrier(group=control_group)
    with torch_npu.profiler.profile(
        activities=activities,
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as profiler:
        # Keep host launch times close. The in-graph HCCL rendezvous below
        # absorbs the remaining process scheduling skew on device.
        dist.barrier(group=control_group)
        with torch.profiler.record_function("lite_oe_acl_graph_50_iterations"):
            graph.replay()
            torch.npu.synchronize()
    profiler.export_chrome_trace(str(trace_path))

    metadata = {
        "rank": rank,
        "local_rank": local_rank,
        "dp_rank": rank // TP_SIZE,
        "tp_rank": tp_rank,
        "tp_group": tp_ranks,
        "local_tokens_per_rank": LOCAL_TOKENS_PER_RANK,
        "tokens_after_tp8_gather": TOKEN_COUNT,
        "graph_iterations": args.graph_iterations,
        "measured_graph_launches": 1,
        "graph_start_sync": "16-rank HCCL all-reduce of one int32 scalar",
        "project_solution": "torch",
        "table_placement": args.table_placement,
        "table_shapes": [list(table.shape) for table in tables],
        "table_strides": [list(table.stride()) for table in tables],
        "activation_shape": list(activation.shape),
        "projection_shape": list(projection.shape),
        "output_shape": list(word.shape),
        "trace": trace_path.name,
    }
    (output_dir / f"rank{rank:02d}.metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    dist.barrier(group=control_group)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
