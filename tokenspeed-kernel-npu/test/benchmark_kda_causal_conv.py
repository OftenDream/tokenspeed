"""Compare default fused causal-conv with solution=ref on real TS arenas.

Run from the repository root with its runtime/test dependencies installed:
    python tokenspeed-kernel-npu/test/benchmark_kda_causal_conv.py \
        --cases tokenspeed-kernel-npu/test/benchmark_kda_causal_conv_cases.jsonl \
        --output results --profile
Each JSONL case has id and inputs=[{name, value}, ...]; see the performance doc.
RESULT lines contain paired wall samples, device time, strides and correctness.
"""

import argparse
import csv
import gc
import json
import runpy
import statistics
import time
from pathlib import Path

import torch
import torch_npu
from tokenspeed_kernel.ops.attention import kda_causal_conv1d
from tokenspeed_kernel_npu.ops.kda import _load_flash_causal_conv_ops


def profile_call(invoke, trace_root, fused):
    if trace_root.exists():
        raise RuntimeError(f"Refusing to mix profiler captures: {trace_root}")
    with torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        schedule=torch_npu.profiler.schedule(wait=0, warmup=5, active=5, repeat=1),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(trace_root)),
        experimental_config=torch_npu.profiler._ExperimentalConfig(
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1
        ),
    ) as prof:
        for _ in range(10):
            invoke()
            torch.npu.synchronize()
            prof.step()
    return read_profile(trace_root, fused)


def read_profile(trace_root, fused):
    """Include AI_CPU ViewCopy, not just AI Core/Vector kernels."""
    csvs = list(trace_root.rglob("kernel_details.csv"))
    assert len(csvs) == 1, csvs
    with csvs[0].open(encoding="utf-8-sig") as file:
        kernels = [
            row
            for row in csv.DictReader(file)
            if row.get("Accelerator Core")
            in ("AI_VECTOR_CORE", "AI_CORE", "MIX_AIC", "MIX_AIV", "AI_CPU")
        ]
    assert kernels, csvs[0]
    causal = [
        row
        for row in kernels
        if "CausalConv1d" in row["Name"] or "causal_conv1d" in row["Name"]
    ]
    assert len(causal) == (5 if fused else 0), causal
    return dict(
        device_us=sum(float(row["Duration(us)"]) for row in kernels) / 5,
        aicpu_us=sum(
            float(row["Duration(us)"])
            for row in kernels
            if row["Accelerator Core"] == "AI_CPU"
        )
        / 5,
        causal_us=sum(float(row["Duration(us)"]) for row in causal) / 5,
        kernel_count=len(kernels) / 5,
        kernel_names=sorted({row["Name"] for row in kernels}),
    )


def run_case(case, args, pool_factory):
    values = {item["name"]: item["value"] for item in case["inputs"]}
    channels, batch, length = (values[key] for key in ("channels", "batch", "length"))
    mode, dtype = values["mode"], getattr(torch, values["dtype"])
    assert mode in ("prefill_eager", "decode_eager", "decode_graph")
    decode = mode != "prefill_eager"
    assert not decode or length == 1
    heads, tp = (64, 1) if channels == 24576 else (32, 12288 // channels)
    _, pool = pool_factory(
        "npu", num_lcm_blocks=2 * batch + 2, tp_size=tp, linear_num_heads=heads
    )
    arena_state = pool.get_state_buffers(0)[0]
    state = pool.arena.buffer.view(dtype).as_strided(
        arena_state.shape, arena_state.stride(), arena_state.storage_offset()
    )
    assert state.shape[1:] == (3, channels)
    assert state.stride()[1:] == (channels, 1)
    assert state.stride(0) > 3 * channels
    assert state.data_ptr() % 256 == 0
    assert all(state.stride(axis) * state.element_size() % 256 == 0 for axis in (0, 1))
    torch.manual_seed(1901 + channels + batch + length)
    initial = torch.randn(state.shape, dtype=dtype, device="npu") * 0.02
    x = torch.randn(batch * length, channels, dtype=dtype, device="npu") * 0.05
    weight = torch.randn(channels, 4, dtype=dtype, device="npu") * 0.03
    starts_cpu = torch.arange(batch + 1, dtype=torch.int64) * length
    starts = starts_cpu.to(device="npu", dtype=torch.int32)
    reads = torch.arange(1, batch + 1, device="npu", dtype=torch.int32)
    writes = reads + batch
    modes = None if decode else torch.ones(batch, device="npu", dtype=torch.bool)

    def call(solution):
        return kda_causal_conv1d(
            x,
            weight,
            state,
            reads,
            writes,
            starts,
            cu_seqlens_cpu=starts_cpu,
            has_initial_state=modes,
            decode=decode,
            solution=solution,
        )

    def verify(out):
        active = (reads >= 0) & (writes >= 0)
        hist = initial.index_select(0, reads.clamp_min(0).long())
        seq = torch.cat((hist, x.view(batch, length, channels)), dim=1)
        expected = sum(
            seq[:, tap : tap + length].float() * weight[:, tap].float()
            for tap in range(4)
        )
        expected = torch.nn.functional.silu(expected).to(dtype)
        expected = torch.where(active[:, None, None], expected, 0).reshape_as(out)
        torch.testing.assert_close(out.float(), expected.float(), atol=1e-4, rtol=0.03)
        expected_state = initial.clone()
        safe_writes = writes.clamp_min(0).long()
        updated = torch.where(
            active[:, None, None], seq[:, -3:], initial.index_select(0, safe_writes)
        )
        expected_state.index_copy_(0, safe_writes, updated)
        torch.testing.assert_close(state, expected_state, atol=0, rtol=0)

    invocations, outputs, graphs = {}, {}, []
    for name, solution in (("fused", None), ("ref", "ref")):
        state.copy_(initial)
        invoke = lambda solution=solution: call(solution)
        out = invoke()
        verify(out)
        if mode == "decode_graph":
            for _ in range(3):
                invoke()
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(
                graph, stream=torch.npu.Stream(), auto_dispatch_capture=True
            ):
                out = invoke()
            graphs.append(graph)
            invoke = graph.replay
        invocations[name], outputs[name] = invoke, out
    # Changed inputs/pages must work without recapture, including padding.
    x.mul_(0.7)
    reads.copy_(reads.flip(0))
    if batch > 1:
        reads[-1], writes[-1] = -1, -1
    for name, invoke in invocations.items():
        state.copy_(initial)
        out = invoke()
        verify(outputs[name] if mode == "decode_graph" else out)
    reads.copy_(torch.arange(1, batch + 1, device="npu", dtype=torch.int32))
    writes.copy_(reads + batch)
    state.copy_(initial)
    for invoke in invocations.values():
        for _ in range(20):
            invoke()
    samples = {name: [] for name in invocations}
    repeats = 200 if mode == "decode_graph" else 20
    # Both paths use the same arena/input; read and write pages are disjoint.
    # Alternate order each round, with no state reset inside timed regions.
    for round_index in range(7):
        for name in (
            list(invocations) if round_index % 2 == 0 else list(reversed(invocations))
        ):
            invoke = invocations[name]
            torch.npu.synchronize()
            start = time.perf_counter()
            for _ in range(repeats):
                invoke()
            torch.npu.synchronize()
            samples[name].append((time.perf_counter() - start) * 1e6 / repeats)
    record = dict(
        case=case["id"],
        **values,
        correctness=True,
        state_shape=list(state.shape),
        state_stride=list(state.stride()),
        state_offset=state.storage_offset(),
        base_alignment_bytes=256,
    )
    for name, invoke in invocations.items():
        record[name] = dict(
            wall_us=statistics.median(samples[name]), samples_us=samples[name]
        )
        if args.profile:
            record[name].update(
                profile_call(
                    invoke, Path(args.output) / case["id"] / name, name == "fused"
                )
            )
    print("RESULT " + json.dumps(record), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    torch.npu.set_device(0)
    assert (
        _load_flash_causal_conv_ops() is not None
    ), "A stride-capable flash_ops package is required"
    root = Path(__file__).resolve().parents[2]
    helpers = runpy.run_path(str(root / "test/runtime/test_lite_hybrid_cache.py"))
    with open(args.cases) as file:
        cases = [json.loads(line) for line in file if line.strip()]
    for case in cases[: args.limit]:
        run_case(case, args, helpers["_pool"])
        gc.collect()
        torch.npu.empty_cache()
