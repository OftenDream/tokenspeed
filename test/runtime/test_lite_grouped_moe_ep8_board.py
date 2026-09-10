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

"""Eight-rank admission board for Lite Grouped MoE placements.

Launch the hardware case explicitly with::

    LITE_EP8_BOARD=1 torchrun --standalone --nproc-per-node=8 \
      -m pytest -q -s test/runtime/test_lite_grouped_moe_ep8_board.py
"""

from __future__ import annotations

import gc
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Literal

import pytest
import torch

Layout = Literal["A", "B", "C"]
GROUPS = 4
REAL_EXPERTS = 384
IDENTITY_EXPERTS = 32
TOPK = 12
HIDDEN = 768
INTERMEDIATE = 512
EP_SIZE = 8
FLAT_EXPERTS = GROUPS * REAL_EXPERTS
PREFILL_SPLITS = {
    32: (1, 3, 4, 5, 2, 6, 7, 4),
    128: (13, 15, 17, 19, 11, 21, 14, 18),
    1024: (125, 127, 129, 131, 123, 133, 126, 130),
}


def _production_grouped(hidden: torch.Tensor) -> torch.Tensor:
    """Model the unit-weight RMSNorm and x2 scale before the Grouped MoE."""
    flat = hidden.view(-1, GROUPS * HIDDEN)
    value = flat.float()
    value = value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + 1e-5)
    return (value * 2.0).to(hidden.dtype).view_as(hidden)


def _owner(layout: Layout, group: int, expert: int) -> tuple[int, int, int]:
    if layout == "A":
        rank, local = divmod(expert, 48)
        return rank, local, expert
    if layout == "B":
        rank, within_group = divmod(expert, 48)
        local = group * 48 + within_group
        return rank, local, rank * 192 + local
    physical = group * REAL_EXPERTS + expert
    rank, local = divmod(physical, 192)
    return rank, local, physical


def _logical_experts(layout: Layout, rank: int) -> list[int]:
    owned = []
    for group in range(GROUPS):
        for expert in range(REAL_EXPERTS):
            owner, local, _ = _owner(layout, group, expert)
            if owner == rank:
                owned.append((local, group * REAL_EXPERTS + expert))
    return [logical for _, logical in sorted(owned)]


@pytest.mark.parametrize("layout", ["A", "B", "C"])
def test_lite_ep8_placement_is_a_bijection(layout: Layout):
    physical = set()
    owners = [0] * EP_SIZE
    for group in range(GROUPS):
        for expert in range(REAL_EXPERTS):
            rank, local, mapped = _owner(layout, group, expert)
            assert 0 <= rank < EP_SIZE
            assert 0 <= local < 192
            key = (group, mapped) if layout == "A" else mapped
            assert key not in physical
            physical.add(key)
            owners[rank] += 1
    assert owners == [192] * EP_SIZE
    assert len(physical) == FLAT_EXPERTS
    for rank in range(EP_SIZE):
        assert len(_logical_experts(layout, rank)) == 192


class _Weights(torch.nn.Module):
    def __init__(
        self,
        num_experts: int,
        ep_rank: int,
        w13: torch.Tensor,
        w2: torch.Tensor,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.num_local_experts = w13.shape[0]
        self.ep_rank = ep_rank
        self.ep_size = EP_SIZE
        self.hidden_size = HIDDEN
        self.intermediate_size = INTERMEDIATE
        self.w13_weight = torch.nn.Parameter(w13, requires_grad=False)
        self.w2_weight = torch.nn.Parameter(w2, requires_grad=False)


class _Checkpoint:
    def __init__(self, root: Path, layer: int) -> None:
        index = json.loads((root / "model.safetensors.index.json").read_text())
        self.root = root
        self.layer = layer
        self.weight_map: dict[str, str] = index["weight_map"]

    def read(self, names: list[str]) -> dict[str, torch.Tensor]:
        from safetensors import safe_open

        by_file: dict[str, list[str]] = defaultdict(list)
        for name in names:
            if name not in self.weight_map:
                raise KeyError(f"checkpoint tensor is missing: {name}")
            by_file[self.weight_map[name]].append(name)
        result = {}
        for filename, keys in by_file.items():
            with safe_open(self.root / filename, framework="pt", device="cpu") as file:
                result.update((key, file.get_tensor(key)) for key in keys)
        return result

    def routers(self, device: torch.device) -> list[tuple[torch.Tensor, torch.Tensor]]:
        prefix = f"model.layers.{self.layer}.mlp.expert_groups"
        names = []
        for group in range(GROUPS):
            names.extend(
                (
                    f"{prefix}.{group}.router.classifier.weight",
                    f"{prefix}.{group}.router.e_score_correction_bias",
                )
            )
        tensors = self.read(names)
        return [
            (
                tensors[f"{prefix}.{group}.router.classifier.weight"]
                .float()
                .to(device),
                tensors[f"{prefix}.{group}.router.e_score_correction_bias"]
                .float()
                .to(device),
            )
            for group in range(GROUPS)
        ]

    def experts(
        self, logical_ids: list[int], device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prefix = f"model.layers.{self.layer}.mlp.experts"
        names = [
            f"{prefix}.{expert}.{projection}_proj.weight"
            for expert in logical_ids
            for projection in ("gate", "up", "down")
        ]
        tensors = self.read(names)
        w13, w2 = [], []
        for expert in logical_ids:
            base = f"{prefix}.{expert}"
            w13.append(
                torch.cat(
                    (
                        tensors[f"{base}.gate_proj.weight"],
                        tensors[f"{base}.up_proj.weight"],
                    )
                )
            )
            w2.append(tensors[f"{base}.down_proj.weight"])
        return torch.stack(w13).to(device), torch.stack(w2).to(device)


def _synthetic_experts(
    logical_ids: list[int], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(6601)
    base13 = torch.randn(
        2 * INTERMEDIATE, HIDDEN, dtype=torch.bfloat16, generator=generator
    ).to(device)
    base2 = torch.randn(
        HIDDEN, INTERMEDIATE, dtype=torch.bfloat16, generator=generator
    ).to(device)
    logical = torch.tensor(logical_ids, device=device, dtype=torch.float32)
    scale13 = (0.004 + logical.remainder(23) * 0.00001).to(torch.bfloat16)
    scale2 = (0.004 + logical.remainder(29) * 0.00001).to(torch.bfloat16)
    return (
        (base13.unsqueeze(0) * scale13[:, None, None]).contiguous(),
        (base2.unsqueeze(0) * scale2[:, None, None]).contiguous(),
    )


def _make_weight(
    logical_ids: list[int],
    num_experts: int,
    rank: int,
    device: torch.device,
    checkpoint: _Checkpoint | None,
) -> _Weights:
    w13, w2 = (
        checkpoint.experts(logical_ids, device)
        if checkpoint is not None
        else _synthetic_experts(logical_ids, device)
    )
    return _Weights(num_experts, rank, w13, w2)


def _plan(weight: _Weights) -> dict:
    import tokenspeed_kernel

    plan = tokenspeed_kernel.moe_plan(
        "unquant",
        input_dtype=torch.bfloat16,
        activation="silu",
        routing_mode="precomputed_topk",
        ep_size=EP_SIZE,
        ispp=INTERMEDIATE,
    )
    tokenspeed_kernel.moe_process_weights(plan, weight)
    return plan


def _contexts(
    layout: Layout,
    rank: int,
    device: torch.device,
    checkpoint: _Checkpoint | None,
) -> list[tuple[dict, _Weights]]:
    if layout != "A":
        weight = _make_weight(
            _logical_experts(layout, rank), FLAT_EXPERTS, rank, device, checkpoint
        )
        return [(_plan(weight), weight)]

    contexts = []
    for group in range(GROUPS):
        logical = [group * REAL_EXPERTS + rank * 48 + expert for expert in range(48)]
        weight = _make_weight(logical, REAL_EXPERTS, rank, device, checkpoint)
        contexts.append((_plan(weight), weight))
    return contexts


def _routes(
    grouped: torch.Tensor,
    checkpoint_routers: list[tuple[torch.Tensor, torch.Tensor]] | None,
    kind: str = "mixed",
) -> tuple[torch.Tensor, torch.Tensor]:
    if checkpoint_routers is not None:
        import tokenspeed_kernel

        weights, ids = [], []
        for group, (router_weight, bias) in enumerate(checkpoint_routers):
            logits = torch.nn.functional.linear(
                grouped[:, group].float(), router_weight
            )
            route_weight, route_id = tokenspeed_kernel.moe_softmax_bias_topk(
                logits, bias, TOPK, routed_scaling_factor=6.0
            )
            weights.append(route_weight)
            ids.append(route_id)
        return torch.stack(weights), torch.stack(ids)

    tokens = grouped.shape[0]
    token = torch.arange(tokens, device=grouped.device)[:, None]
    route = torch.arange(TOPK, device=grouped.device)[None, :]
    ids, weights = [], []
    for group in range(GROUPS):
        if kind == "empty":
            group_ids = (token + route).remainder(12)
        else:
            group_ids = (token * 17 + group * 97 + route * 13).remainder(REAL_EXPERTS)
        if kind == "mixed":
            identity = (token + group + route).remainder(17) == 0
            group_ids = torch.where(
                identity,
                REAL_EXPERTS + (token + route).remainder(IDENTITY_EXPERTS),
                group_ids,
            )
        raw = torch.cos((token + 1) * (route + group + 1) * 0.03125)
        weights.append(torch.softmax(raw.float(), dim=-1) * 6.0)
        ids.append(group_ids.to(torch.int32))
    return torch.stack(weights), torch.stack(ids)


def _physical_ids(layout: Layout, ids: torch.Tensor) -> torch.Tensor:
    if layout == "A":
        return ids
    group = torch.arange(GROUPS, device=ids.device, dtype=torch.int32)[:, None, None]
    real = ids < REAL_EXPERTS
    if layout == "B":
        rank = torch.div(ids, 48, rounding_mode="floor")
        local = group * 48 + ids.remainder(48)
        mapped = rank * 192 + local
    else:
        mapped = group * REAL_EXPERTS + ids
    return torch.where(real, mapped, torch.full_like(mapped, FLAT_EXPERTS))


def _apply_leaf(
    context: tuple[dict, _Weights],
    hidden: torch.Tensor,
    weights: torch.Tensor,
    ids: torch.Tensor,
) -> torch.Tensor:
    import tokenspeed_kernel

    plan, layer_weight = context
    return tokenspeed_kernel.moe_apply(
        plan,
        hidden,
        layer_weight,
        torch.empty(hidden.shape[0], 1, device=hidden.device),
        topk_weights=weights,
        topk_ids=ids,
    )


def _local_partial(
    layout: Layout,
    contexts: list[tuple[dict, _Weights]],
    grouped: torch.Tensor,
    route_weights: torch.Tensor,
    route_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    identity_weights = torch.where(
        route_ids >= REAL_EXPERTS,
        route_weights,
        torch.zeros_like(route_weights),
    ).sum(dim=-1)
    identity = (grouped * identity_weights.permute(1, 0).unsqueeze(-1)).reshape(
        grouped.shape[0], GROUPS * HIDDEN
    )
    if layout == "A":
        partial = [
            _apply_leaf(
                contexts[group],
                grouped[:, group],
                route_weights[group],
                route_ids[group],
            )
            for group in range(GROUPS)
        ]
        return torch.cat(partial, dim=-1), identity

    flattened = grouped.permute(1, 0, 2).reshape(-1, HIDDEN)
    output = _apply_leaf(
        contexts[0],
        flattened,
        route_weights.reshape(-1, TOPK),
        _physical_ids(layout, route_ids).reshape(-1, TOPK),
    )
    output = output.view(GROUPS, grouped.shape[0], HIDDEN).permute(1, 0, 2)
    return output.reshape(grouped.shape[0], GROUPS * HIDDEN), identity


def _torch_local_partial_a(
    contexts: list[tuple[dict, _Weights]],
    grouped: torch.Tensor,
    route_weights: torch.Tensor,
    route_ids: torch.Tensor,
    rank: int,
) -> torch.Tensor:
    """Independent FP32 accumulation oracle for placement A."""
    output = torch.zeros_like(grouped, dtype=torch.float32)
    for group in range(GROUPS):
        layer_weight = contexts[group][1]
        group_ids = route_ids[group]
        for local in range(48):
            expert = rank * 48 + local
            token_ids, route_ids_for_expert = torch.where(group_ids == expert)
            if token_ids.numel() == 0:
                continue
            gate_up = (
                grouped[token_ids, group].float()
                @ layer_weight.w13_weight[local].float()
            )
            gate, up = gate_up.chunk(2, dim=-1)
            expert_output = torch.nn.functional.silu(gate) * up
            expert_output = expert_output @ layer_weight.w2_weight[local].float()
            output[:, group].index_add_(
                0,
                token_ids,
                expert_output
                * route_weights[group, token_ids, route_ids_for_expert].unsqueeze(-1),
            )
    return output.reshape(grouped.shape[0], GROUPS * HIDDEN)


def _max_rank(value: float, device: torch.device) -> float:
    import torch.distributed as dist

    tensor = torch.tensor(value, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor)


def _measure(fn, device: torch.device, warmup: int, repeats: int) -> float:
    import torch.distributed as dist

    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    dist.barrier()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    torch.npu.synchronize()
    elapsed = (time.perf_counter() - start) * 1e6 / repeats
    dist.barrier()
    return _max_rank(elapsed, device)


def _comparison(
    actual: torch.Tensor, expected: torch.Tensor
) -> tuple[float, float, bool]:
    assert torch.isfinite(actual).all()
    actual_float = actual.float()
    expected_float = expected.float()
    difference = (actual_float - expected_float).abs()
    rel = torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(
        expected_float
    ).clamp_min(1e-12)
    close = torch.isclose(actual_float, expected_float, atol=2e-2, rtol=2e-2).all()
    return float(rel), float(difference.max()), bool(close)


def _assert_close(
    actual: torch.Tensor, expected: torch.Tensor, max_rel: float = 1e-2
) -> float:
    rel, _, close = _comparison(actual, expected)
    assert close
    assert rel <= max_rel
    return rel


def _route_stats(layout: Layout, ids: torch.Tensor) -> dict:
    counts = [0] * EP_SIZE
    identity = int((ids >= REAL_EXPERTS).sum())
    for group in range(GROUPS):
        for expert in ids[group][ids[group] < REAL_EXPERTS].tolist():
            rank, _, _ = _owner(layout, group, int(expert))
            counts[rank] += 1
    mean = sum(counts) / EP_SIZE
    variance = sum((count - mean) ** 2 for count in counts) / EP_SIZE
    return {
        "rank_real_routes": counts,
        "pair_real_routes": [
            counts[index] + counts[index + 1] for index in range(0, 8, 2)
        ],
        "rank_max_mean": max(counts) / mean if mean else 0.0,
        "rank_cv": math.sqrt(variance) / mean if mean else 0.0,
        "zero_route_ranks": sum(count == 0 for count in counts),
        "identity_routes": identity,
    }


def _graph_replay(
    layout: Layout,
    contexts: list[tuple[dict, _Weights]],
    grouped: torch.Tensor,
    route_weights: torch.Tensor,
    route_ids: torch.Tensor,
    group: tuple[int, ...],
) -> float:
    from tokenspeed.runtime.distributed.comm_ops import all_reduce

    hidden_buffer = grouped.clone()
    weight_buffer = route_weights.clone()
    id_buffer = route_ids.clone()

    def run():
        partial, identity = _local_partial(
            layout, contexts, hidden_buffer, weight_buffer, id_buffer
        )
        return all_reduce(partial, group) + identity

    for _ in range(4):
        output = run()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=torch.npu.Stream(), auto_dispatch_capture=True):
        output = run()
    torch.npu.synchronize()
    graph.replay()
    torch.npu.synchronize()
    first = output.clone()
    hidden_buffer.neg_()
    real = id_buffer < REAL_EXPERTS
    id_buffer.copy_(
        torch.where(real, (id_buffer + 1).remainder(REAL_EXPERTS), id_buffer)
    )
    graph.replay()
    torch.npu.synchronize()
    actual = output.clone()
    expected = run()
    torch.npu.synchronize()
    assert not torch.equal(first, actual)
    return _assert_close(actual, expected)


@pytest.mark.skipif(
    os.getenv("LITE_EP8_BOARD") != "1",
    reason="set LITE_EP8_BOARD=1 under an eight-rank torchrun",
)
def test_lite_grouped_moe_ep8_board():
    import torch.distributed as dist

    if int(os.environ.get("WORLD_SIZE", "1")) != EP_SIZE:
        pytest.fail("Lite EP8 board requires WORLD_SIZE=8")
    import torch_npu  # noqa: F401

    from tokenspeed.runtime.distributed.comm_backend import initialize_comm_backend
    from tokenspeed.runtime.distributed.comm_ops import (
        all_reduce,
        token_all_gather,
        token_reduce_scatter,
    )
    from tokenspeed.runtime.distributed.mapping import Mapping
    from tokenspeed.runtime.distributed.process_group_manager import (
        process_group_manager as pg_manager,
    )

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"npu:{local_rank}")
    torch.npu.set_device(device)
    mapping = Mapping(
        rank=rank,
        world_size=EP_SIZE,
        attn_tp_size=1,
        attn_cp_size=EP_SIZE,
        dense_tp_size=EP_SIZE,
        moe_tp_size=1,
        moe_ep_size=EP_SIZE,
        linear_attn_tp_size=EP_SIZE,
        nprocs_per_node=EP_SIZE,
        nnodes=1,
    )
    pg_manager.init_distributed(mapping, backend="hccl", timeout=300, device_id=device)
    group = mapping.moe.ep_group
    pg_manager.init_process_group(group, backend="hccl")
    initialize_comm_backend()

    source = os.getenv("LITE_EP8_BOARD_WEIGHTS", "synthetic")
    if source not in {"synthetic", "checkpoint"}:
        pytest.fail("LITE_EP8_BOARD_WEIGHTS must be synthetic or checkpoint")
    checkpoint = None
    routers = None
    if source == "checkpoint":
        root = os.getenv("LITE_EP8_BOARD_CHECKPOINT")
        if not root:
            pytest.fail("checkpoint replay requires LITE_EP8_BOARD_CHECKPOINT")
        checkpoint = _Checkpoint(
            Path(root), int(os.getenv("LITE_EP8_BOARD_LAYER", "0"))
        )
        routers = checkpoint.routers(device)

    warmup = int(os.getenv("LITE_EP8_BOARD_WARMUP", "20"))
    repeats = int(os.getenv("LITE_EP8_BOARD_REPEATS", "50"))
    records: list[dict] = []
    references: dict[str, torch.Tensor] = {}
    oracle_rel = None

    torch.manual_seed(6701)
    full_hidden = {
        tokens: _production_grouped(
            torch.randn(tokens, GROUPS * HIDDEN, device=device, dtype=torch.bfloat16)
        )
        for tokens in PREFILL_SPLITS
    }
    decode_hidden = {
        batch: _production_grouped(
            torch.randn(batch, GROUPS, HIDDEN, device=device, dtype=torch.bfloat16)
        )
        for batch in (1, 2)
    }

    for layout in ("A", "B", "C"):
        contexts = _contexts(layout, rank, device, checkpoint)
        torch.npu.reset_peak_memory_stats(device)

        empty_ids = _routes(decode_hidden[1], None, kind="empty")[1]
        empty_weights = _routes(decode_hidden[1], None, kind="empty")[0]
        empty_partial, empty_identity = _local_partial(
            layout, contexts, decode_hidden[1], empty_weights, empty_ids
        )
        empty_output = all_reduce(empty_partial, group) + empty_identity
        torch.npu.synchronize()
        if layout == "A":
            if source == "synthetic":
                oracle = all_reduce(
                    _torch_local_partial_a(
                        contexts,
                        decode_hidden[1],
                        empty_weights,
                        empty_ids,
                        rank,
                    ),
                    group,
                )
                oracle_rel = _assert_close(empty_output.float(), oracle, max_rel=5e-3)
            references["empty"] = empty_output.detach().cpu()
        else:
            _assert_close(empty_output, references["empty"].to(device))

        for tokens, split in PREFILL_SPLITS.items():
            offset = sum(split[:rank])
            local_hidden = full_hidden[tokens][offset : offset + split[rank]]
            gathered = token_all_gather(local_hidden, group, list(split))
            grouped = gathered.view(tokens, GROUPS, HIDDEN)
            route_weights, route_ids = _routes(grouped, routers)

            def prefill_once():
                current = token_all_gather(local_hidden, group, list(split))
                current = current.view(tokens, GROUPS, HIDDEN)
                partial, identity = _local_partial(
                    layout, contexts, current, route_weights, route_ids
                )
                scattered = token_reduce_scatter(partial, group, list(split))
                return scattered + identity[offset : offset + split[rank]]

            output = prefill_once()
            torch.npu.synchronize()
            key = f"P{tokens}"
            rel = 0.0
            max_abs = 0.0
            numeric_gate = True
            if layout == "A":
                references[key] = output.detach().cpu()
            else:
                rel, max_abs, elementwise = _comparison(
                    output, references[key].to(device)
                )
                numeric_gate = elementwise and rel <= 1e-2
            latency = _measure(prefill_once, device, warmup, repeats)
            if rank == 0:
                records.append(
                    {
                        "layout": layout,
                        "case": key,
                        "latency_us": latency,
                        "rel_l2_vs_a": rel,
                        "max_abs_vs_a": max_abs,
                        "numeric_gate_pass": numeric_gate,
                        "collectives": {
                            "token_all_gather": 1,
                            "token_reduce_scatter": 1,
                        },
                        "leaf_calls": 4 if layout == "A" else 1,
                        "route": _route_stats(layout, route_ids),
                    }
                )

        for batch, grouped in decode_hidden.items():
            route_weights, route_ids = _routes(grouped, routers)

            def decode_once():
                partial, identity = _local_partial(
                    layout, contexts, grouped, route_weights, route_ids
                )
                return all_reduce(partial, group) + identity

            output = decode_once()
            torch.npu.synchronize()
            key = f"D{batch}"
            rel = 0.0
            max_abs = 0.0
            numeric_gate = True
            if layout == "A":
                references[key] = output.detach().cpu()
            else:
                rel, max_abs, elementwise = _comparison(
                    output, references[key].to(device)
                )
                numeric_gate = elementwise and rel <= 1e-2
            latency = _measure(decode_once, device, warmup, repeats)
            graph_rel = _graph_replay(
                layout, contexts, grouped, route_weights, route_ids, group
            )
            if rank == 0:
                records.append(
                    {
                        "layout": layout,
                        "case": key,
                        "latency_us": latency,
                        "rel_l2_vs_a": rel,
                        "max_abs_vs_a": max_abs,
                        "numeric_gate_pass": numeric_gate,
                        "graph_rel_l2": graph_rel,
                        "collectives": {"all_reduce": 1},
                        "leaf_calls": 4 if layout == "A" else 1,
                        "route": _route_stats(layout, route_ids),
                    }
                )

        allocated = _max_rank(torch.npu.max_memory_allocated(device), device)
        reserved = _max_rank(torch.npu.max_memory_reserved(device), device)
        if rank == 0:
            records.append(
                {
                    "layout": layout,
                    "case": "memory",
                    "allocated_bytes": int(allocated),
                    "reserved_bytes": int(reserved),
                }
            )
        del contexts
        gc.collect()
        torch.npu.empty_cache()
        dist.barrier()

    if rank == 0:
        payload = {
            "weights": source,
            "warmup": warmup,
            "repeats": repeats,
            "synthetic_oracle_rel_l2": oracle_rel,
            "records": records,
        }
        output_path = os.getenv("LITE_EP8_BOARD_JSON")
        if output_path:
            Path(output_path).write_text(json.dumps(payload, indent=2, sort_keys=True))
        print(
            "LITE_EP8_BOARD_RESULT=" + json.dumps(payload, sort_keys=True), flush=True
        )

    dist.barrier()
    dist.destroy_process_group()
