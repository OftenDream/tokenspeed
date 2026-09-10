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

"""Full-layer composed/fused A/B: torchrun with GMOE_EXCHANGE_OPTIONS JSON.\n\nSystem CANN/custom OPP environment must already be configured. Set OUTPUT_DIR,\nWEIGHT_DTYPE=int8|bf16, GROUPS=8, TOKENS=32 and optionally PROFILE=1.\n"""

import itertools
import json
import math
import os
import random
import statistics
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.distributed as dist
import torch_npu

from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.distributed.process_group_manager import process_group_manager
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from tokenspeed.runtime.models.flash_local_moe import GroupAwareFlashLocalMoE


def finite_bitwise_equal(left, right):
    """Compare shape, dtype and all storage bits, rejecting NaN/Inf on either side."""
    left, right = left.cpu().contiguous(), right.cpu().contiguous()
    return (
        left.shape == right.shape
        and left.dtype == right.dtype
        and bool(torch.isfinite(left).all() & torch.isfinite(right).all())
        and torch.equal(left.view(torch.uint8), right.view(torch.uint8))
    )


def precision_token_lengths(spec):
    """Explicit ordered lengths; dense means powers through 8192 then every tail length."""
    if spec == "dense":
        return [2**power for power in range(14)] + list(range(8193, 16385))
    values = [int(value) for value in spec.split(",")]
    if (
        not values
        or any(value <= 0 for value in values)
        or len(set(values)) != len(values)
    ):
        raise ValueError("PRECISION_LENGTHS requires distinct positive token lengths")
    return values


def error_metrics(actual, reference):
    """Diagnostic magnitudes, not a replacement for strict acceptance checks."""
    a, b = actual.detach().cpu().contiguous(), reference.detach().cpu().contiguous()
    if a.shape != b.shape or a.dtype != b.dtype:
        raise ValueError("Error metrics require matching shape/dtype")
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    af, bf = a.double(), b.double()
    delta = af - bf
    different = (
        a.view(torch.uint8)
        .reshape(a.numel(), a.element_size())
        .ne(b.view(torch.uint8).reshape(b.numel(), b.element_size()))
        .any(dim=1)
    )
    return dict(
        elements=a.numel(),
        different=int(different.sum()),
        finite=finite,
        max_abs=float(delta.abs().max()) if finite and a.numel() else None,
        mean_abs=float(delta.abs().mean()) if finite and a.numel() else None,
        relative_l2=(
            float(delta.norm() / bf.norm().clamp_min(1e-30)) if finite else None
        ),
    )


def random_token_lengths(iterations, seed, max_tokens, *, unbounded=False):
    """Reproducible boundary coverage followed by balanced decode/prefill sampling."""
    if iterations <= 0 or max_tokens < 129:
        raise ValueError("Require positive iterations and RANDOM_MAX_TOKENS >= 129")
    rng = random.Random(seed)
    boundaries = sorted(
        {
            1,
            7,
            31,
            32,
            33,
            64,
            65,
            127,
            128,
            129,
            255,
            256,
            257,
            511,
            512,
            513,
            1023,
            1024,
            1025,
            2048,
            2049,
            4095,
            4096,
            4097,
            8191,
            8192,
            8193,
            16383,
            16384,
            16385,
            max_tokens - 1,
            max_tokens,
        }
    )
    boundaries = [value for value in boundaries if value <= max_tokens]
    rng.shuffle(boundaries)
    for iteration in (itertools.count() if unbounded else range(iterations)):
        if iteration < len(boundaries):
            yield boundaries[iteration]
        elif rng.randrange(2):
            yield rng.randint(1, 128)
        else:
            yield rng.randint(129, max_tokens)


def stability_finished(completed, target, elapsed, minimum_seconds):
    return completed >= target and elapsed >= minimum_seconds


def prepare_graph_streams(device, shared_stream):
    """Reserve two unrestricted streams once, without changing any core quota.

    Stream() rotates through a finite pool; even a still-live limited shared
    stream can be returned again. Recreating capture streams per shape can
    therefore change the reference's arithmetic or serialize the overlap.
    """
    limits = torch.npu.get_device_limit(device)
    streams, seen = {}, set()
    while len(streams) < 2:
        stream = torch.npu.Stream(device=device)
        if stream in seen:
            raise RuntimeError(
                "Cannot reserve two independent unrestricted graph streams"
            )
        seen.add(stream)
        if stream == shared_stream or torch.npu.get_stream_limit(stream) != limits:
            continue
        streams[("composed", "fused")[len(streams)]] = stream
    return streams


def chunk_reference_experts(apply, plan, x, w, router_logits, *, chunk_rows, **kwargs):
    """Bound only the test oracle's local expert intermediates, not fused inputs."""
    if chunk_rows <= 0:
        raise ValueError("reference chunk_rows must be positive")
    if x.shape[0] <= chunk_rows:
        return apply(plan, x, w, router_logits, **kwargs)
    output = torch.empty_like(x)
    for start in range(0, x.shape[0], chunk_rows):
        stop = min(start + chunk_rows, x.shape[0])
        part_kwargs = {
            key: (
                value[start:stop]
                if key in {"topk_weights", "topk_ids"} and value is not None
                else value
            )
            for key, value in kwargs.items()
        }
        logits = None if router_logits is None else router_logits[start:stop]
        output[start:stop].copy_(apply(plan, x[start:stop], w, logits, **part_kwargs))
    return output


@torch.inference_mode()
def main():
    root = Path(os.environ["OUTPUT_DIR"])
    root.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ["RANK"])
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("npu", int(os.environ["LOCAL_RANK"]))
    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(123)
    world = int(os.environ["WORLD_SIZE"])
    # Bash reserves GROUPS as an array; use a task-specific environment key.
    groups = int(os.environ.get("GMOE_NUM_GROUPS", os.environ.get("GROUPS", "8")))
    tokens = int(os.environ.get("TOKENS", "32"))
    mapping = Mapping(
        rank=rank,
        world_size=world,
        attn_tp_size=1,
        dense_tp_size=1,
        moe_tp_size=1,
        moe_ep_size=world,
    )
    process_group_manager.init_distributed(mapping, backend="hccl")
    config = SimpleNamespace(
        hidden_size=4096,
        moe_group_size=groups,
        expert_ffn_hidden_size=1024,
        n_routed_experts=384,
        zero_expert_num=32,
        moe_topk=16,
        rms_norm_eps=1e-5,
        ffn_hidden_size=2048,
        n_shared_experts=1,
        norm_topk_prob=False,
        routed_scaling_factor=6.0,
        grouped_moe_norm_scale=2.0,
    )
    config.gmoe_exchange_options = json.loads(os.environ["GMOE_EXCHANGE_OPTIONS"])
    if "GMOE_RDMA_LIBRARY" in os.environ:
        config.gmoe_exchange_options["rdma_library"] = os.environ["GMOE_RDMA_LIBRARY"]
    if "SHARED_OVERLAP" in os.environ:
        config.gmoe_exchange_options["shared_overlap"] = (
            os.environ["SHARED_OVERLAP"] == "1"
        )
    config.gmoe_pre_solution = config.gmoe_post_solution = "flash_npu"
    router_fusion = os.environ.get("ROUTER_FUSION") == "1"
    shared_early_test = os.environ.get("SHARED_EARLY_TEST") == "1"
    full_reference = os.environ.get("FULL_REFERENCE") == "1"
    shared_ffn = os.environ.get("SHARED_FFN") == "1"
    if shared_ffn and not full_reference:
        raise ValueError(
            "SHARED_FFN requires FULL_REFERENCE: keep the old MLP unrestricted"
        )
    strict_bits = os.environ.get("STRICT_BITS") == "1"
    role = os.environ.get("FORWARD_ROLE", "decode")
    if role not in {"decode", "prefill"}:
        raise ValueError("FORWARD_ROLE must be decode or prefill")
    if full_reference and (not router_fusion or shared_early_test):
        raise ValueError(
            "FULL_REFERENCE requires ROUTER_FUSION=1 and SHARED_EARLY_TEST=0"
        )
    stability_seconds = float(os.environ.get("STABILITY_SECONDS", "0"))
    if stability_seconds < 0 or not math.isfinite(stability_seconds):
        raise ValueError("STABILITY_SECONDS must be finite and nonnegative")
    if stability_seconds and not (full_reference and strict_bits):
        raise ValueError("Stability requires FULL_REFERENCE=1 STRICT_BITS=1")
    if shared_early_test and not router_fusion:
        raise ValueError("SHARED_EARLY_TEST requires ROUTER_FUSION=1")
    if router_fusion:
        config.gmoe_pre_solution = "flash_npu_router"
    if shared_ffn:
        config.gmoe_pre_solution = "flash_npu_router_ffn"
        config.gmoe_exchange_options["ffn_binding"] = os.environ["FUSED_FFN_BINDING"]
    routed_fusion = os.environ.get("ROUTED_FUSION") == "1"
    routed_single_kernel = (
        os.environ.get("ROUTED_SINGLE_KERNEL", "1" if routed_fusion else "0") == "1"
    )
    if routed_single_kernel and not routed_fusion:
        raise ValueError("ROUTED_SINGLE_KERNEL requires ROUTED_FUSION=1")
    mm2_fusion = os.environ.get("MM2_FUSION") == "1"
    if full_reference and not (routed_fusion and mm2_fusion):
        raise ValueError("FULL_REFERENCE requires both fused expert segments")
    if mm2_fusion and not routed_fusion:
        raise ValueError("MM2_FUSION requires ROUTED_FUSION=1")
    if routed_fusion:
        config.gmoe_expert_solution = (
            "flash_npu_routed_full" if mm2_fusion else "flash_npu_routed"
        )
    quant_kind = os.environ.get("WEIGHT_DTYPE", "int8")
    smooth_quant = os.environ.get("SMOOTH_QUANT", "none")
    if smooth_quant not in {"none", "w13", "w2", "both"}:
        raise ValueError("SMOOTH_QUANT must be none, w13, w2, or both")
    quant = CompressedTensorsConfig.from_config(
        {
            "format": "int-quantized",
            "quant_method": "compressed-tensors",
            "config_groups": {
                "group_0": {
                    "targets": ["Linear"],
                    "weights": {
                        "num_bits": 8,
                        "type": "int",
                        "strategy": "channel",
                        "symmetric": True,
                        "dynamic": False,
                    },
                    "input_activations": {
                        "num_bits": 8,
                        "type": "int",
                        "strategy": "token",
                        "symmetric": True,
                        "dynamic": True,
                    },
                }
            },
        }
    )
    if quant_kind == "bf16":
        quant = None
    with torch.device(device):
        model = GroupAwareFlashLocalMoE(config, mapping, quant_config=quant).eval()
    # This fixture supplies routed-expert smooth weights only. The global
    # enable_smooth_quant flag also targets shared dense linears, whose existing
    # scheme explicitly rejects SmoothQuant; this test does not change that.
    if smooth_quant != "none" and quant_kind != "int8":
        raise ValueError("Routed-expert SmoothQuant requires WEIGHT_DTYPE=int8")
    for name, width, enabled in (
        (
            "w13_smooth_scale",
            model.experts.hidden_size,
            smooth_quant in {"w13", "both"},
        ),
        (
            "w2_smooth_scale",
            model.experts.intermediate_size,
            smooth_quant in {"w2", "both"},
        ),
    ):
        if enabled:
            model.experts.register_parameter(
                name,
                torch.nn.Parameter(
                    torch.ones(
                        model.experts.num_local_experts,
                        width,
                        device=device,
                        dtype=torch.float32,
                    ),
                    requires_grad=False,
                ),
            )
    # Common projections/shared expert are replicated, so use the same seed on
    # all ranks. Dummy real-expert slices and routers receive independent seeds.
    for name, param in model.named_parameters():
        if param.dtype == torch.int8:
            param.copy_(
                torch.randint(-8, 8, param.shape, dtype=torch.int8, device=device)
            )
        elif "weight_scale" in name:
            param.fill_(1 / 128)
        elif "smooth_scale" in name:
            param.uniform_(0.25, 1.75)
        elif name == "norm.weight":
            param.fill_(1)
        elif "correction_bias" in name:
            param.zero_()
        else:
            param.normal_(0, 0.01)
    torch.manual_seed(13 + rank)
    for param in model.experts.parameters():
        if param.dtype == torch.int8:
            param.copy_(
                torch.randint(-8, 8, param.shape, dtype=torch.int8, device=device)
            )
    torch.manual_seed(17 + model.topology.group_id)
    model.router.classifier.weight.normal_(0, 0.01)
    route_case = os.environ.get("ROUTE_CASE", "mixed")
    bias = model.router.e_score_correction_bias
    if route_case == "mixed":
        # Match the prior synthetic real-top-k-10 workload.
        bias[384:390] = 100
    elif route_case == "real_only":
        bias[384:] = -100
    elif route_case == "zero_only":
        bias[384:] = 100
    elif route_case == "hot_real":
        bias[:16] = 100
    elif route_case != "random":
        raise ValueError("Unknown ROUTE_CASE")
    model.process_weights_after_loading()
    if routed_fusion and not routed_single_kernel:
        # Explicit test-only two-launch control. Normal fused execution uses the
        # canonical single-launch binding selected by production preparation.
        if not hasattr(torch.ops.custom, "lite_routed_gmm13_swiglu"):
            raise RuntimeError(
                "Loaded binding does not provide the two-launch reference"
            )
        torch.ops.custom.fused_init_routing_mm13_swiglu = (
            torch.ops.custom.lite_routed_gmm13_swiglu
        )
    assert model._moe_plan is not None, "No production MoE kernel selected"
    assert model._quant_kind == ("int8" if quant_kind == "int8" else "unquant")
    for projection in (
        model.shared_experts.gate_up_proj,
        model.shared_experts.down_proj,
    ):
        projection.quant_method.process_weights_after_loading(projection)
        if quant_kind == "int8":
            assert projection.weight.dtype == torch.int8
            assert torch_npu.get_npu_format(projection._int8_plan.weight) == 29
    assert model.shared_experts._use_int8_swiglu == (quant_kind == "int8")
    torch.manual_seed(23 + rank)
    hidden = torch.randn(tokens, 4096, device=device, dtype=torch.bfloat16)
    context = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND if role == "prefill" else ForwardMode.DECODE
    )

    def run():
        return model(
            hidden,
            num_global_tokens=world * tokens,
            max_num_tokens_per_gpu=tokens,
            ctx=context,
        )

    from tokenspeed_kernel.ops.moe.gmoe import select_gmoe_stages

    fused = model._moe_stages
    assert fused.pre.name == (
        "flash_npu_router_ffn_gmoe_pre"
        if shared_ffn
        else ("flash_npu_router_gmoe_pre" if router_fusion else "flash_npu_gmoe_pre")
    )
    assert fused.post.name == "flash_npu_gmoe_post"
    baseline = select_gmoe_stages(
        input_dtype=torch.bfloat16,
        traits={},
        pre_solution="composed",
        post_solution="composed",
    )
    fused_plan = model._moe_plan
    reference_plan = fused_plan
    if routed_fusion:
        import tokenspeed_kernel

        assert world in (8, 16) and world % groups == 0 and tokens > 0
        assert quant_kind in ("int8", "bf16")
        assert fused_plan["solution"] == config.gmoe_expert_solution
        reference_plan = tokenspeed_kernel.moe_plan(
            quant_kind,
            input_dtype=torch.bfloat16,
            activation="silu",
            routing_mode="precomputed_topk",
            ep_size=world // groups,
            num_zero_experts=config.zero_expert_num,
            ispp=1024,
            internal_activation_dtype="int8" if quant_kind == "int8" else "input",
            solution="flash_npu_routed" if mm2_fusion else "torch_npu",
        )
        # Isolate expert input fusion; both controls retain identical exchange/shared work.
        baseline = fused
    if router_fusion:
        # Change only dispatch/router. Expert kernels, weights, combine and
        # shared-stream quota are identical on both sides of this A/B test.
        baseline = select_gmoe_stages(
            input_dtype=torch.bfloat16,
            traits={"gmoe_exchange_enabled": True},
            pre_solution="flash_npu",
            post_solution="flash_npu",
        )
        reference_plan = fused_plan
    if full_reference:
        # Direct whole-layer control: no custom fused exchange/router/MM13/MM2.
        # Shared dense MLP/projections and canonical weights are still shared.
        baseline = select_gmoe_stages(
            input_dtype=torch.bfloat16,
            traits={},
            pre_solution="composed",
            post_solution="composed",
        )
        reference_plan = tokenspeed_kernel.moe_plan(
            quant_kind,
            input_dtype=torch.bfloat16,
            activation="silu",
            routing_mode="precomputed_topk",
            ep_size=world // groups,
            num_zero_experts=config.zero_expert_num,
            ispp=1024,
            internal_activation_dtype="int8" if quant_kind == "int8" else "input",
            solution="torch_npu",
        )
    stage_context = model._moe_stage_context
    capture_streams = prepare_graph_streams(
        device, stage_context.exchange.shared_stream
    )
    (root / f"capture-streams-rank{rank:02d}.json").write_text(
        json.dumps(
            {
                name: {
                    "stream_id": stream.stream_id,
                    "limits": torch.npu.get_stream_limit(stream),
                }
                for name, stream in {
                    **capture_streams,
                    **(
                        {"shared": stage_context.exchange.shared_stream}
                        if stage_context.exchange.shared_stream is not None
                        else {}
                    ),
                }.items()
            },
            indent=2,
        )
    )
    debug_tails, debug_inputs = {}, {}
    debug_name = "eager"
    debug_pre = {}
    debug_shared = {}
    projection_trace_active = False
    debug_projection, debug_projection_eager = {}, {}
    if os.environ.get("TRACE_PROJECTION") == "1":

        def projection_hook(name):
            def hook(module, args, output):
                if projection_trace_active:
                    debug_projection.setdefault(debug_name, {})[name] = output

            return hook

        model.proj_input.register_forward_hook(projection_hook("projection"))
        model.norm.register_forward_hook(projection_hook("norm"))
    if os.environ.get("SHARED_DEBUG") == "1":

        def save_shared(name):
            def hook(module, args, output):
                debug_shared[f"{debug_name}-{name}"] = output

            return hook

        for name in ("gate_up_proj", "act_fn", "down_proj"):
            getattr(model.shared_experts, name).register_forward_hook(save_shared(name))
    if router_fusion:
        import tokenspeed_kernel.ops.moe.gmoe_ascend as pre_backend

        original_pre = pre_backend._pre

        def capture_pre(*args, **kwargs):
            result = original_pre(*args, **kwargs)
            debug_pre[debug_name] = result
            return result

        pre_backend._pre = capture_pre
        if full_reference:
            from dataclasses import replace

            from tokenspeed_kernel.selection import SelectedKernel

            reference_pre = baseline.pre

            def capture_reference_pre(**kwargs):
                result = reference_pre(**kwargs)
                debug_pre[debug_name] = result
                return result

            baseline = replace(
                baseline, pre=SelectedKernel(reference_pre.name, capture_reference_pre)
            )

    def check_bits(actual, expected, label):
        # Actual storage comparison includes signed zero; numerical equality
        # and rtol=atol=0 alone do not prove bitwise equality.
        left, right = actual.cpu().contiguous(), expected.cpu().contiguous()
        equal = finite_bitwise_equal(left, right)
        mismatch = torch.tensor([not equal], device=device, dtype=torch.int32)
        dist.all_reduce(mismatch, op=dist.ReduceOp.MAX)
        if mismatch.item():

            def cpu_snapshot(value):
                if isinstance(value, torch.Tensor):
                    return value.cpu()
                if isinstance(value, (list, tuple)):
                    return [cpu_snapshot(item) for item in value]
                return value

            torch.save(
                {
                    "label": label,
                    "hidden": hidden.cpu(),
                    "actual": left,
                    "expected": right,
                    "projection_eager": debug_projection_eager,
                    "projection_current": {
                        variant: {key: value.cpu() for key, value in fields.items()}
                        for variant, fields in debug_projection.items()
                    },
                    "shared": {
                        key: cpu_snapshot(value) for key, value in debug_shared.items()
                    },
                    "shared_stream_limit": (
                        torch.npu.get_stream_limit(stage_context.exchange.shared_stream)
                        if stage_context.exchange.shared_stream is not None
                        else None
                    ),
                    "pre": {
                        key: {
                            field: getattr(value, field).cpu()
                            for field in (
                                "received",
                                "local_received",
                                "topk_weights",
                                "topk_ids",
                                "shared_output",
                            )
                        }
                        for key, value in debug_pre.items()
                    },
                },
                root / f"bit-mismatch-rank{rank:02d}.pt",
            )
            raise AssertionError(f"Bitwise mismatch: {label}; all-rank snapshots saved")

    if os.environ.get("REPRO_INPUT_DIR"):
        fixture = torch.load(
            Path(os.environ["REPRO_INPUT_DIR"]) / f"bit-mismatch-rank{rank:02d}.pt",
            map_location="cpu",
            weights_only=True,
        )
        hidden = fixture["hidden"].to(device)
        tokens = hidden.shape[0]
        if rank == 0:
            torch.save(
                {
                    "projection": model.proj_input.weight.cpu(),
                    "norm": model.norm.weight.cpu(),
                },
                root / "repro-weights.pt",
            )
        observed = {}

        def remember(name):
            def hook(module, args, output):
                observed[name] = output

            return hook

        for name, module in (
            ("projection", model.proj_input),
            ("norm", model.norm),
            ("output_projection", model.proj_output),
            ("shared_mm13", model.shared_experts.gate_up_proj),
            ("shared_act", model.shared_experts.act_fn),
            ("shared_mm2", model.shared_experts.down_proj),
        ):
            module.register_forward_hook(remember(name))

        def stage_values(output):
            values = {}

            def visit(name, item):
                if isinstance(item, torch.Tensor):
                    values[name] = item
                elif isinstance(item, (tuple, list)):
                    for index, part in enumerate(item):
                        visit(f"{name}.{index}", part)

            for key, value in observed.items():
                visit(key, value)
            for field in ("received", "topk_weights", "topk_ids", "shared_output"):
                visit(field, getattr(debug_pre[debug_name], field))
            visit("output", output)
            return values

        reports = {}
        for name, stages, plan in (
            ("composed", baseline, reference_plan),
            ("fused", fused, fused_plan),
        ):
            debug_name = name
            model._moe_stages, model._moe_plan = stages, plan
            observed.clear()
            value = run()
            eager = {
                key: item.cpu().clone() for key, item in stage_values(value).items()
            }
            stream = capture_streams[name]
            stream.wait_stream(torch.npu.current_stream())
            with torch.npu.stream(stream):
                for _ in range(5):
                    run()
            torch.npu.synchronize()
            dist.barrier()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph, stream=stream, auto_dispatch_capture=True):
                value = run()
            captured = stage_values(value)
            graph.replay()
            replay = {key: item.cpu().clone() for key, item in captured.items()}
            reports[name] = {
                key: {
                    "exact": finite_bitwise_equal(replay[key], item),
                    "max_abs": float((replay[key].float() - item.float()).abs().max()),
                }
                for key, item in eager.items()
            }
            reports[name]["saved_eager"] = {
                "exact": finite_bitwise_equal(eager["output"], fixture["expected"])
            }
            torch.save(
                {"eager": eager, "graph": replay},
                root / f"repro-{name}-rank{rank:02d}.pt",
            )
            graph.reset()
            captured.clear()
        (root / f"repro-stages-rank{rank:02d}.json").write_text(
            json.dumps(reports, indent=2)
        )
        print("REPRO " + json.dumps({"rank": rank, "stages": reports}), flush=True)
        torch.npu.synchronize()
        dist.barrier()
        model.release_moe_resources()
        dist.destroy_process_group()
        return

    length_spec = os.environ.get("PRECISION_LENGTHS")
    explicit_lengths = precision_token_lengths(length_spec) if length_spec else None
    random_iterations = int(os.environ.get("RANDOM_LENGTH_ITERATIONS", "0"))
    if explicit_lengths is not None:
        if random_iterations or float(os.environ.get("RANDOM_MIN_SECONDS", "0")):
            raise ValueError(
                "Explicit precision matrix must not implicitly resume random stability"
            )
        random_iterations = len(explicit_lengths)
    if random_iterations:
        if not (full_reference and strict_bits):
            raise ValueError("Random stability requires FULL_REFERENCE=1 STRICT_BITS=1")
        seed = int(os.environ.get("RANDOM_LENGTH_SEED", "20260909"))
        max_tokens = (
            max(explicit_lengths)
            if explicit_lengths is not None
            else int(os.environ.get("RANDOM_MAX_TOKENS", "32684"))
        )
        minimum_seconds = float(os.environ.get("RANDOM_MIN_SECONDS", "0"))
        if minimum_seconds < 0 or not math.isfinite(minimum_seconds):
            raise ValueError("RANDOM_MIN_SECONDS must be finite and nonnegative")
        reference_chunk_rows = int(os.environ.get("REFERENCE_EXPERT_CHUNK_ROWS", "0"))
        if reference_chunk_rows < 0:
            raise ValueError("REFERENCE_EXPERT_CHUNK_ROWS must be nonnegative")
        original_apply = tokenspeed_kernel.moe_apply
        if reference_chunk_rows:

            def test_apply(plan, x, w, router_logits, **kwargs):
                if plan is reference_plan:
                    return chunk_reference_experts(
                        original_apply,
                        plan,
                        x,
                        w,
                        router_logits,
                        chunk_rows=reference_chunk_rows,
                        **kwargs,
                    )
                return original_apply(plan, x, w, router_logits, **kwargs)

            tokenspeed_kernel.moe_apply = test_apply
            if os.environ.get("REFERENCE_CHECK_UNCHUNKED") == "1":
                # One small-enough independent oracle check before any capture.
                inputs = baseline.pre(hidden_states=hidden, context=stage_context)
                assert inputs.received.shape[0] > reference_chunk_rows
                kwargs = dict(
                    topk_weights=inputs.topk_weights, topk_ids=inputs.topk_ids
                )
                whole = original_apply(
                    reference_plan,
                    inputs.received,
                    model.experts,
                    inputs.topk_weights,
                    **kwargs,
                )
                chunked = test_apply(
                    reference_plan,
                    inputs.received,
                    model.experts,
                    inputs.topk_weights,
                    **kwargs,
                )
                check_bits(chunked, whole, "chunked-unfused-vs-original-unfused")
                del whole, chunked, inputs
                debug_pre.clear()
        graph_every = int(os.environ.get("RANDOM_GRAPH_EVERY", "100"))
        if graph_every <= 0:
            raise ValueError("RANDOM_GRAPH_EVERY must be positive")
        lengths = (
            explicit_lengths
            if explicit_lengths is not None
            else random_token_lengths(
                random_iterations, seed, max_tokens, unbounded=minimum_seconds > 0
            )
        )
        choice = torch.zeros(2, dtype=torch.int64, device=device)
        histogram, graph_histogram = {}, {}
        started = time.monotonic()
        initial_memory = torch.npu.memory_allocated()
        sequence_kind = "precision" if explicit_lengths is not None else "random"
        progress_path = root / f"{sequence_kind}-progress-rank{rank:02d}.jsonl"

        def select_variant(name):
            nonlocal debug_name
            debug_name = name
            model._moe_stages = baseline if name == "composed" else fused
            model._moe_plan = reference_plan if name == "composed" else fused_plan

        for iteration, selected_tokens in enumerate(lengths, 1):
            if rank == 0:
                stop = stability_finished(
                    iteration - 1,
                    random_iterations,
                    time.monotonic() - started,
                    minimum_seconds,
                )
                choice.copy_(torch.tensor([selected_tokens, int(stop)], device=device))
            dist.broadcast(choice, src=0)
            if choice[1].item():
                break
            tokens = int(choice[0].item())
            projection_trace_active = (
                os.environ.get("TRACE_PROJECTION") == "1"
                and tokens == 70
                and (iteration - 1) % graph_every == 0
            )
            debug_projection.clear()
            debug_projection_eager.clear()
            role = "decode" if tokens <= 128 else "prefill"
            context.forward_mode = (
                ForwardMode.DECODE if role == "decode" else ForwardMode.EXTEND
            )
            # Per-iteration/per-rank seeds make a failure reproducible without
            # depending on the number of warmups or graph captures before it.
            input_seed = seed + iteration * world + rank
            torch.manual_seed(input_seed)
            hidden = torch.randn(tokens, 4096, device=device, dtype=torch.bfloat16)
            label = f"iteration={iteration},T={tokens},role={role},seed={input_seed}"
            if rank == 0:
                with (root / f"{sequence_kind}-sequence.jsonl").open("a") as log:
                    log.write(
                        json.dumps(
                            dict(
                                iteration=iteration,
                                tokens=tokens,
                                role=role,
                                input_seed_rank0=input_seed,
                            )
                        )
                        + "\n"
                    )
            saved = {}
            for repeat in range(2):
                for name in ("composed", "fused"):
                    select_variant(name)
                    # CPU snapshots cannot alias mutable device buffers reused
                    # by the next invocation of either implementation.
                    if (
                        os.environ.get("RANDOM_PROFILE_ITERATION") == str(iteration)
                        and name == "fused"
                        and repeat == 0
                        and rank == 0
                    ):
                        with torch_npu.profiler.profile(
                            activities=[
                                torch_npu.profiler.ProfilerActivity.CPU,
                                torch_npu.profiler.ProfilerActivity.NPU,
                            ],
                            record_shapes=True,
                            experimental_config=torch_npu.profiler._ExperimentalConfig(
                                profiler_level=torch_npu.profiler.ProfilerLevel.Level1
                            ),
                            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                                str(root / "failure-profile")
                            ),
                        ):
                            result = run().cpu().clone()
                    else:
                        result = run().cpu().clone()
                    if repeat:
                        check_bits(result, saved[name], f"{label},{name},self-repeat")
                    else:
                        saved[name] = result
                        if projection_trace_active:
                            debug_projection_eager[name] = {
                                key: value.cpu().clone()
                                for key, value in debug_projection[name].items()
                            }
                if not repeat:
                    check_bits(saved["fused"], saved["composed"], f"{label},eager-A/B")

            if (iteration - 1) % graph_every == 0 or (
                tokens == max_tokens and tokens not in graph_histogram
            ):
                # Capture only the current shape, replay it, then release it.
                # Persistent model/weights/communication survive all iterations;
                # arbitrary dynamic shapes are not mislabeled as one ACL Graph.
                graphs, graph_outputs = {}, {}
                try:
                    for name in ("composed", "fused"):
                        select_variant(name)
                        stream = capture_streams[name]
                        if torch.npu.get_stream_limit(
                            stream
                        ) != torch.npu.get_device_limit(device):
                            raise RuntimeError(
                                "Reserved capture stream core quota changed"
                            )
                        stream.wait_stream(torch.npu.current_stream())
                        with torch.npu.stream(stream):
                            for _ in range(5):
                                run()
                        torch.npu.synchronize()
                        dist.barrier()
                        graph = torch.npu.NPUGraph()
                        graphs[name] = graph
                        with torch.npu.graph(
                            graph, stream=stream, auto_dispatch_capture=True
                        ):
                            graph_outputs[name] = run()
                    assert (
                        graph_outputs["fused"].data_ptr()
                        != graph_outputs["composed"].data_ptr()
                    )
                    for repeat in range(2):
                        for name in ("composed", "fused"):
                            graphs[name].replay()
                            check_bits(
                                graph_outputs[name],
                                saved[name],
                                f"{label},{name},graph-repeat={repeat}",
                            )
                finally:
                    torch.npu.synchronize()
                    for graph in graphs.values():
                        graph.reset()
                    graphs.clear()
                    graph_outputs.clear()
                graph_histogram[tokens] = graph_histogram.get(tokens, 0) + 1

            histogram[tokens] = histogram.get(tokens, 0) + 1
            debug_pre.clear()
            del saved, result
            if (
                explicit_lengths is not None
                or iteration == 1
                or iteration % 100 == 0
                or iteration == random_iterations
            ):
                record = dict(
                    rank=rank,
                    completed=iteration,
                    requested=random_iterations,
                    test_kind=sequence_kind,
                    tokens=tokens,
                    case_passed=True,
                    passed=False,
                    seed=seed,
                    max_tokens=max_tokens,
                    unique_lengths=len(histogram),
                    graph_iterations=sum(graph_histogram.values()),
                    elapsed_seconds=time.monotonic() - started,
                    memory_allocated_bytes=torch.npu.memory_allocated(),
                    memory_reserved_bytes=torch.npu.memory_reserved(),
                    initial_memory_bytes=initial_memory,
                    full_reference=True,
                    strict_bits=True,
                    reference_expert_solution=reference_plan["solution"],
                    fused_expert_solution=fused_plan["solution"],
                    fused_pre=fused.pre.name,
                    fused_post=fused.post.name,
                    baseline_pre=baseline.pre.name,
                    baseline_post=baseline.post.name,
                    weight_dtype=quant_kind,
                    smooth_quant=smooth_quant,
                    world_size=world,
                    groups=groups,
                    hidden_size=4096,
                    shared_overlap=stage_context.exchange.shared_stream is not None,
                    shared_ffn=shared_ffn,
                    shared_core_limit=(
                        torch.npu.get_stream_limit(stage_context.exchange.shared_stream)
                        if stage_context.exchange.shared_stream is not None
                        else None
                    ),
                    communication_window_bytes=stage_context.exchange.window_bytes,
                    token_chunking=stage_context.exchange.token_chunking,
                    reference_expert_chunk_rows=reference_chunk_rows,
                    minimum_seconds=minimum_seconds,
                    route_case=os.environ.get("ROUTE_CASE", "mixed"),
                    bitwise_mismatches=0,
                    nonfinite_outputs=0,
                )
                with progress_path.open("a") as log:
                    log.write(json.dumps(record) + "\n")
                if rank == 0:
                    print(json.dumps(record), flush=True)
        record.update(
            completed=sum(histogram.values()),
            passed=True,
            elapsed_seconds=time.monotonic() - started,
            unique_lengths=len(histogram),
            graph_iterations=sum(graph_histogram.values()),
            token_histogram=histogram,
            graph_token_histogram=graph_histogram,
        )
        if explicit_lengths is not None and histogram != {
            t: 1 for t in explicit_lengths
        }:
            raise AssertionError(
                "Explicit precision matrix did not cover every requested length exactly once"
            )
        (root / f"{sequence_kind}-result-rank{rank:02d}.json").write_text(
            json.dumps(record, indent=2)
        )
        torch.npu.synchronize()
        dist.barrier()
        model.release_moe_resources()
        tokenspeed_kernel.moe_apply = original_apply
        dist.destroy_process_group()
        return

    if shared_early_test:
        from dataclasses import replace

        from tokenspeed_kernel.ops.moe.gmoe import GMoEInputs
        from tokenspeed_kernel.selection import SelectedKernel

        def late_shared_pre(*, hidden_states, context):
            # Test-only prior schedule: same fused router/experts/quotas, but
            # shared waits for projection/norm/scale before it can start.
            exchange = context.exchange
            tokens = hidden_states.shape[0]
            exchange.check_router_shape(
                tokens, hidden_states.shape[1] // context.num_groups, context.top_k
            )
            projected = context.norm(context.proj_input(hidden_states))
            grouped = (projected * context.norm_scale).view(
                tokens, context.num_groups, -1
            )
            shared_output, shared_wait = exchange.run_shared(
                context.shared_experts, hidden_states
            )
            received, weights, ids = exchange.exchange_router(
                grouped,
                context.router,
                context.top_k,
                context.routed_scaling_factor,
                context.renormalize_topk,
            )
            rows = tokens * context.num_groups
            local = received.narrow(0, context.egp_rank * rows, rows)
            if shared_wait is not None:
                shared_wait()
            result = GMoEInputs(received, local, weights, ids, shared_output)
            debug_pre[debug_name] = result
            return result

        baseline = replace(
            fused, pre=SelectedKernel("late_shared_router_pre", late_shared_pre)
        )
    if routed_fusion and not router_fusion:
        import tokenspeed_kernel.ops.moe.ascend as moe_registry
        import tokenspeed_kernel_npu.ops.moe as moe_backend

        original_tail = moe_backend._ascend_int8_gmm2_finalize

        def capture_tail(h, s, c, r, i, weights, w):
            debug_tails[debug_name] = (h, s, c, r, i)
            return original_tail(h, s, c, r, i, weights, w)

        moe_backend._ascend_int8_gmm2_finalize = capture_tail
        if mm2_fusion:
            original_fused_tail = fused_plan["mm2_finalize"]

            def capture_fused_tail(h, s, c, r, i, weights, w):
                debug_tails[debug_name] = (h, s, c, r, i)
                return original_fused_tail(h, s, c, r, i, weights, w)

            fused_plan["mm2_finalize"] = capture_fused_tail
        for attr in ("_int8_moe_apply", "_routed_int8_moe_apply"):
            original_apply = getattr(moe_registry, attr)

            def capture_input(_apply=original_apply, **kwargs):
                debug_inputs[debug_name] = kwargs["x"]
                return _apply(**kwargs)

            setattr(moe_registry, attr, capture_input)
    if os.environ.get("ERROR_REPORT") == "1":
        if not full_reference or strict_bits:
            raise ValueError(
                "ERROR_REPORT needs FULL_REFERENCE=1 STRICT_BITS=0; it is diagnostic only"
            )
        reports = []

        def measure(label, a, b):
            reports.append(dict(label=label, **error_metrics(a, b)))

        def variant(name):
            nonlocal debug_name
            debug_name = name
            model._moe_stages = baseline if name == "composed" else fused
            model._moe_plan = reference_plan if name == "composed" else fused_plan

        # Identical expert input/routes isolate the expert leaf from communication.
        before = baseline.pre(hidden_states=hidden, context=stage_context)
        after = fused.pre(hidden_states=hidden, context=stage_context)
        for field in ("received", "topk_ids", "topk_weights", "shared_output"):
            measure(f"pre.{field}", getattr(after, field), getattr(before, field))
        kwargs = dict(topk_weights=before.topk_weights, topk_ids=before.topk_ids)
        ref_leaf = tokenspeed_kernel.moe_apply(
            reference_plan,
            before.received,
            model.experts,
            before.topk_weights,
            **kwargs,
        )
        fused_leaf = tokenspeed_kernel.moe_apply(
            fused_plan, before.received, model.experts, before.topk_weights, **kwargs
        )
        measure("expert_leaf.same_input", fused_leaf, ref_leaf)
        measure(
            "post.same_input",
            fused.post(routed=ref_leaf, inputs=before, context=stage_context),
            baseline.post(routed=ref_leaf, inputs=before, context=stage_context),
        )
        if os.environ.get("REDUCE_REPORT") == "1":
            from dataclasses import replace

            rows = before.local_received.shape[0]
            splits = [rows] * len(stage_context.egp_group)
            raw_bf16 = stage_context.reduce_scatter(
                ref_leaf, stage_context.egp_group, splits
            )
            raw_fp32 = stage_context.reduce_scatter(
                ref_leaf.float(), stage_context.egp_group, splits
            ).to(ref_leaf.dtype)
            # Independent high-precision oracle: gather each rank's original
            # BF16 contribution, sum on CPU in FP64, then round once to BF16.
            gathered = stage_context.all_gather(
                ref_leaf,
                stage_context.egp_group,
                [ref_leaf.shape[0]] * len(stage_context.egp_group),
            )
            start = stage_context.egp_rank * rows
            parts = gathered.cpu().reshape(
                len(stage_context.egp_group), *ref_leaf.shape
            )
            gold = (
                parts[:, start : start + rows]
                .double()
                .sum(0)
                .to(ref_leaf.dtype)
                .to(device)
            )
            measure("reduce.bf16_vs_fp64_oracle", raw_bf16, gold)
            measure("reduce.fp32_vs_fp64_oracle", raw_fp32, gold)
            measure("reduce.bf16_vs_fp32", raw_bf16, raw_fp32)
            # Substitute only the reduction result; all later math/weights
            # remain the original composed post-stage implementation.
            fp32_context = replace(stage_context, reduce_scatter=lambda *args: raw_fp32)
            gold_context = replace(stage_context, reduce_scatter=lambda *args: gold)
            fused_post = fused.post(
                routed=ref_leaf, inputs=before, context=stage_context
            )
            fp32_post = baseline.post(
                routed=ref_leaf, inputs=before, context=fp32_context
            )
            gold_post = baseline.post(
                routed=ref_leaf, inputs=before, context=gold_context
            )
            measure("post.fused_vs_fp32_reduce_only", fused_post, fp32_post)
            measure("post.fused_vs_fp64_reduce_only", fused_post, gold_post)
            record = dict(
                rank=rank,
                ep=world,
                groups=groups,
                egp=world // groups,
                dtype=quant_kind,
                tokens=tokens,
                diagnostic_only=True,
                measurement_complete=True,
                group_hidden=model.experts.hidden_size,
                local_experts=model.experts.num_local_experts,
                hccl_deterministic=os.environ.get("HCCL_DETERMINISTIC", "unset"),
                reports=reports,
            )
            (root / f"reduce-report-rank{rank:02d}.json").write_text(
                json.dumps(record, indent=2)
            )
            torch.npu.synchronize()
            dist.barrier()
            model.release_moe_resources()
            dist.destroy_process_group()
            return
        for name in ("composed", "fused"):
            variant(name)
            value = run().clone()
            if name == "composed":
                ref_eager = value
            else:
                measure("full.eager", value, ref_eager)
        graphs, results = {}, {}
        for name in ("composed", "fused"):
            variant(name)
            stream = capture_streams[name]
            stream.wait_stream(torch.npu.current_stream())
            with torch.npu.stream(stream):
                for _ in range(3):
                    run()
            torch.npu.synchronize()
            dist.barrier()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph, stream=stream, auto_dispatch_capture=True):
                results[name] = run()
            graphs[name] = graph
        for step in range(int(os.environ.get("GRAPH_REPLAYS", "20"))):
            hidden.copy_(torch.randn_like(hidden))
            for name in ("composed", "fused"):
                graphs[name].replay()
            torch.npu.synchronize()
            measure(f"full.graph.{step}", results["fused"], results["composed"])
            for name in ("composed", "fused"):
                saved = results[name].clone()
                graphs[name].replay()
                # Diagnose reference nondeterminism as well as fusion error;
                # do not turn a self-repeat discrepancy into a tolerance pass.
                measure(f"self.{name}.{step}", results[name], saved)
        record = dict(
            rank=rank,
            ep=world,
            groups=groups,
            egp=world // groups,
            dtype=quant_kind,
            tokens=tokens,
            diagnostic_only=True,
            measurement_complete=True,
            hccl_deterministic=os.environ.get("HCCL_DETERMINISTIC", "unset"),
            torch_deterministic=torch.are_deterministic_algorithms_enabled(),
            group_hidden=model.experts.hidden_size,
            local_experts=model.experts.num_local_experts,
            shared_overlap=stage_context.exchange.shared_stream is not None,
            reports=reports,
        )
        (root / f"error-report-rank{rank:02d}.json").write_text(
            json.dumps(record, indent=2)
        )
        torch.npu.synchronize()
        for graph in graphs.values():
            graph.reset()
        graphs.clear()
        results.clear()
        model.release_moe_resources()
        dist.barrier()
        dist.destroy_process_group()
        return

    before = baseline.pre(hidden_states=hidden, context=stage_context)
    after = fused.pre(hidden_states=hidden, context=stage_context)
    torch.npu.synchronize()
    for field in ("received", "local_received", "topk_ids"):
        torch.testing.assert_close(
            getattr(before, field), getattr(after, field), rtol=0, atol=0
        )
    torch.testing.assert_close(
        before.topk_weights,
        after.topk_weights,
        rtol=1e-5 if router_fusion else 0,
        atol=2e-6 if router_fusion else 0,
    )
    torch.testing.assert_close(
        before.shared_output, after.shared_output, rtol=0.01, atol=1e-3
    )
    if shared_ffn and strict_bits:
        check_bits(after.shared_output, before.shared_output, "shared-ffn")
    shared_error = (
        (before.shared_output.float() - after.shared_output.float()).abs().max().item()
    )
    # Isolate return communication from router and expert-leaf computation.
    synthetic_routed = torch.randn_like(after.received) * 0.01
    expected_post = baseline.post(
        routed=synthetic_routed, inputs=before, context=stage_context
    )
    actual_post = fused.post(
        routed=synthetic_routed, inputs=after, context=stage_context
    )
    if strict_bits:
        check_bits(actual_post, expected_post, "isolated-post")
    torch.testing.assert_close(actual_post, expected_post, rtol=0.02, atol=2e-3)

    debug_name = "composed"
    model._moe_stages = baseline
    model._moe_plan = reference_plan
    eager_reference = run()
    debug_name = "fused"
    model._moe_stages = fused
    model._moe_plan = fused_plan
    eager_fused = run()
    torch.npu.synchronize()
    if strict_bits:
        check_bits(eager_fused, eager_reference, "eager")
    torch.testing.assert_close(eager_fused, eager_reference, rtol=0.02, atol=2e-3)
    eager_exact = torch.equal(eager_fused, eager_reference)
    if routed_fusion and not router_fusion:
        torch.testing.assert_close(eager_fused, eager_reference, rtol=0, atol=0)
    if shared_early_test:
        torch.testing.assert_close(eager_fused, eager_reference, rtol=0, atol=0)

    graphs, outputs = {}, {}
    for name, stages in (("composed", baseline), ("fused", fused)):
        debug_name = name
        model._moe_stages = stages
        model._moe_plan = reference_plan if name == "composed" else fused_plan
        stream = capture_streams[name]
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            for _ in range(5):
                run()
        torch.npu.synchronize()
        dist.barrier()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream, auto_dispatch_capture=True):
            outputs[name] = run()
        graphs[name] = graph
    if strict_bits:
        assert (
            outputs["fused"].data_ptr() != outputs["composed"].data_ptr()
        ), "A/B graph outputs must not alias"
    max_error = 0.0
    exact_replays = 0
    max_route_weight_error = 0.0
    graph_replays = int(os.environ.get("GRAPH_REPLAYS", "20"))
    fixed_replays = int(os.environ.get("FIXED_REPLAYS", "0"))
    bitwise_replays = 0
    for iteration in range(graph_replays):
        hidden.copy_(torch.randn_like(hidden))
        graphs["composed"].replay()
        graphs["fused"].replay()
        torch.npu.synchronize()
        a, b = outputs["fused"], outputs["composed"]
        if strict_bits:
            check_bits(a, b, f"changed-replay-{iteration}")
            bitwise_replays += 1
        assert torch.isfinite(a).all()
        if router_fusion:
            left, right = debug_pre["composed"], debug_pre["fused"]
            valid = (
                torch.eq(left.received, right.received).all()
                & torch.eq(left.topk_ids, right.topk_ids).all()
                & torch.isclose(
                    left.topk_weights, right.topk_weights, rtol=1e-5, atol=2e-6
                ).all()
                & torch.isclose(a, b, rtol=0.02, atol=2e-3).all()
            )
            mismatch = (~valid).to(torch.int32).reshape(1)
            if shared_early_test:
                mismatch |= (~torch.eq(a, b).all()).to(torch.int32).reshape(1)
            dist.all_reduce(mismatch, op=dist.ReduceOp.MAX)
            if mismatch.item():
                torch.save(
                    {
                        "hidden": hidden.cpu(),
                        "pre": {
                            key: {
                                field: getattr(value, field).cpu()
                                for field in ("received", "topk_ids", "topk_weights")
                            }
                            for key, value in debug_pre.items()
                        },
                        "outputs": {key: value.cpu() for key, value in outputs.items()},
                    },
                    root / f"router-mismatch-rank{rank:02d}.pt",
                )
                raise AssertionError(
                    f"Router fusion mismatch on replay {iteration}, rank {rank}"
                )
            max_route_weight_error = max(
                max_route_weight_error,
                (left.topk_weights - right.topk_weights).abs().max().item(),
            )
        elif not routed_fusion:
            torch.testing.assert_close(a, b, rtol=0.02, atol=2e-3)
        if routed_fusion and not router_fusion:
            # All ranks participate in failure detection, so a failed assertion
            # cannot leave peers blocked inside the next exchange.
            mismatch = torch.tensor(
                [not torch.equal(a, b)], device=device, dtype=torch.int32
            )
            dist.all_reduce(mismatch, op=dist.ReduceOp.MAX)
            if mismatch.item():
                torch.save(
                    {
                        "iteration": iteration,
                        "input": {k: v.cpu() for k, v in debug_inputs.items()},
                        "tail": {
                            k: [v.cpu() for v in vals]
                            for k, vals in debug_tails.items()
                        },
                        "output": {k: v.cpu() for k, v in outputs.items()},
                        "weight_scale": model.experts.w13_weight_scale.cpu(),
                    },
                    root / f"mismatch-rank{rank:02d}.pt",
                )
                raise AssertionError(
                    f"Routed fusion mismatch on iteration {iteration}; saved rank {rank}"
                )
        max_error = max(max_error, (a.float() - b.float()).abs().max().item())
        exact_replays += int(torch.equal(a, b))

    fixed_expected = outputs["fused"].clone()
    for iteration in range(fixed_replays):
        graphs["composed"].replay()
        graphs["fused"].replay()
        torch.npu.synchronize()
        check_bits(outputs["fused"], outputs["composed"], f"fixed-ab-{iteration}")
        check_bits(outputs["fused"], fixed_expected, f"fixed-stability-{iteration}")

    stability_cycles = 0
    stability_started = time.monotonic()
    stability_deadline = stability_started + stability_seconds
    last_report = stability_started
    memory_start = torch.npu.memory_allocated(device)
    memory_peak = memory_start
    while stability_seconds:
        now = time.monotonic()
        # Rank zero owns the deadline/log cadence. Independent rank clocks must
        # not cause one participant to leave a collective ahead of its peers.
        control = torch.tensor(
            [int(now < stability_deadline), int(now - last_report >= 60)],
            dtype=torch.int32,
            device=device,
        )
        dist.broadcast(control, src=0)
        keep_running, report = control.cpu().tolist()
        if not keep_running:
            break
        hidden.copy_(torch.randn_like(hidden))
        for graph in graphs.values():
            graph.replay()
        torch.npu.synchronize()
        check_bits(
            outputs["fused"],
            outputs["composed"],
            f"stability-changed-{stability_cycles}",
        )
        fixed_expected.copy_(outputs["fused"])
        for graph in graphs.values():
            graph.replay()
        torch.npu.synchronize()
        check_bits(
            outputs["fused"],
            outputs["composed"],
            f"stability-fixed-ab-{stability_cycles}",
        )
        check_bits(
            outputs["fused"],
            fixed_expected,
            f"stability-fixed-repeat-{stability_cycles}",
        )
        stability_cycles += 1
        memory_peak = max(memory_peak, torch.npu.memory_allocated(device))
        if report:
            progress = dict(
                rank=rank,
                elapsed_seconds=time.monotonic() - stability_started,
                cycles=stability_cycles,
                bitwise_errors=0,
                role=role,
                tokens=tokens,
                allocated_bytes=torch.npu.memory_allocated(device),
                allocated_start_bytes=memory_start,
                allocated_peak_bytes=memory_peak,
            )
            with (root / f"stability-rank{rank:02d}.jsonl").open("a") as log:
                log.write(json.dumps(progress) + "\n")
            if rank == 0:
                print("STABILITY " + json.dumps(progress), flush=True)
            last_report = now
    stability_elapsed = time.monotonic() - stability_started if stability_seconds else 0

    timings = {}
    for name, graph in graphs.items():
        if os.environ.get("MEASURE", "1") == "0":
            continue
        for _ in range(20):
            graph.replay()
        dist.barrier()
        torch.npu.synchronize()
        samples = []
        for _ in range(5):
            begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(
                enable_timing=True
            )
            begin.record()
            for _ in range(100):
                graph.replay()
            end.record()
            end.synchronize()
            samples.append(begin.elapsed_time(end) * 10)
        timings[name] = statistics.median(samples)
    # Finish all event timing before starting any profiler or parser processes.
    for name, graph in graphs.items():
        if os.environ.get("PROFILE") == "1":
            for _ in range(20):
                graph.replay()
            torch.npu.synchronize()
            dist.barrier()
            trace = root / name / f"rank{rank:02d}"
            with torch_npu.profiler.profile(
                activities=[
                    torch_npu.profiler.ProfilerActivity.CPU,
                    torch_npu.profiler.ProfilerActivity.NPU,
                ],
                record_shapes=True,
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(trace)),
                experimental_config=torch_npu.profiler._ExperimentalConfig(
                    profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                ),
            ) as prof:
                for _ in range(10):
                    graph.replay()
                    prof.step()
                torch.npu.synchronize()
    if os.environ.get("PROFILE") == "1":
        # Graph replay traces may not retain input-shape metadata from capture.
        # Keep a separate eager shape trace; never mix it into graph statistics.
        for name, stages in (("composed", baseline), ("fused", fused)):
            model._moe_stages = stages
            model._moe_plan = reference_plan if name == "composed" else fused_plan
            torch.npu.synchronize()
            dist.barrier()
            with torch_npu.profiler.profile(
                activities=[
                    torch_npu.profiler.ProfilerActivity.CPU,
                    torch_npu.profiler.ProfilerActivity.NPU,
                ],
                record_shapes=True,
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                    str(root / "eager_shapes" / name / f"rank{rank:02d}")
                ),
                experimental_config=torch_npu.profiler._ExperimentalConfig(
                    profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                ),
            ):
                run()
                torch.npu.synchronize()
    record = dict(
        rank=rank,
        quant_kind=quant_kind,
        smooth_quant=smooth_quant,
        smooth_scope="routed_experts",
        ep_size=world,
        groups=groups,
        tokens_per_rank=tokens,
        hidden_size=4096,
        pre=fused.pre.name,
        post=fused.post.name,
        routed_fusion=routed_fusion,
        mm2_fusion=mm2_fusion,
        router_fusion=router_fusion,
        shared_early_test=shared_early_test,
        shared_overlap=model._moe_stage_context.exchange.shared_stream is not None,
        shared_ffn=shared_ffn,
        full_reference=full_reference,
        reference_expert_solution=reference_plan["solution"],
        baseline_post=baseline.post.name,
        forward_role=role,
        route_case=route_case,
        strict_bits=strict_bits,
        token_chunking=model._moe_stage_context.exchange.token_chunking,
        communication_window_bytes=model._moe_stage_context.exchange.window_bytes,
        stability_requested_seconds=stability_seconds,
        stability_elapsed_seconds=stability_elapsed,
        stability_cycles=stability_cycles,
        stability_memory_start_bytes=memory_start,
        stability_memory_peak_bytes=memory_peak,
        bitwise_changed_replays=bitwise_replays,
        bitwise_fixed_replays=fixed_replays,
        baseline_pre=baseline.pre.name,
        max_route_weight_error=max_route_weight_error,
        shared_cube_cores=config.gmoe_exchange_options.get(
            "shared_cube_cores", 4 if router_fusion else 8
        ),
        shared_vector_cores=config.gmoe_exchange_options.get("shared_vector_cores", 8),
        router_cores=model._moe_stage_context.exchange.router_cores,
        routed_single_kernel=routed_single_kernel,
        routed_binding=(
            "fused_init_routing_mm13_swiglu"
            if routed_single_kernel
            else "lite_routed_gmm13_swiglu" if routed_fusion else None
        ),
        expert_kernel=fused_plan["apply_kernel_name"],
        exchange_options=config.gmoe_exchange_options,
        eager_exact=eager_exact,
        pre_shared_max_abs_error=shared_error,
        changed_input_graph_replays=graph_replays,
        exact_replays=exact_replays,
        max_abs_error=max_error,
        graph_us=timings,
    )
    (root / f"{quant_kind}-rank{rank:02d}.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )
    print(json.dumps(record), flush=True)
    torch.npu.synchronize()
    for graph in graphs.values():
        graph.reset()
    graphs.clear()
    outputs.clear()
    model.release_moe_resources()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
