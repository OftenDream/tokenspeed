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

import os
from types import SimpleNamespace

import pytest
import tokenspeed_kernel
import torch
import torch_npu

from tokenspeed.runtime.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)


@pytest.fixture(autouse=True, scope="module")
def _requires_npu():
    if not torch.npu.is_available():
        pytest.skip("requires an Ascend NPU")
    torch.npu.set_device(0)


def test_compressed_tensors_dynamic_token_int8_selects_int8_moe() -> None:
    quant_config = CompressedTensorsConfig.from_config(
        {
            "format": "int-quantized",
            "enable_smooth_quant": True,
            "config_groups": {
                "group_0": {
                    "targets": ["Linear"],
                    "weights": {
                        "num_bits": 8,
                        "type": "int",
                        "symmetric": True,
                        "strategy": "channel",
                        "dynamic": False,
                    },
                    "input_activations": {
                        "num_bits": 8,
                        "type": "int",
                        "symmetric": True,
                        "strategy": "token",
                        "dynamic": True,
                    },
                }
            },
        }
    )
    assert quant_config.moe_weight_dtype("model.layers.0.moe") == "int8"


@pytest.mark.parametrize("smooth_quant", ("none", "w13", "w2", "both"))
def test_w8a8_moe_uses_nz_weights_and_runs(smooth_quant: str) -> None:
    tokens, hidden, intermediate, experts, top_k = 16, 64, 128, 8, 4
    device = torch.device("npu:0")
    weights = SimpleNamespace(
        num_local_experts=experts,
        num_experts=experts,
        hidden_size=hidden,
        intermediate_size=intermediate,
        ep_rank=0,
        ep_size=1,
    )
    weights.w13_weight = torch.nn.Parameter(
        torch.randint(
            -8,
            8,
            (experts, 2 * intermediate, hidden),
            dtype=torch.int8,
            device=device,
        ),
        requires_grad=False,
    )
    weights.w2_weight = torch.nn.Parameter(
        torch.randint(
            -8,
            8,
            (experts, hidden, intermediate),
            dtype=torch.int8,
            device=device,
        ),
        requires_grad=False,
    )
    weights.w13_weight_scale = torch.nn.Parameter(
        torch.full(
            (experts, 2 * intermediate),
            1.0 / 128.0,
            dtype=torch.float32,
            device=device,
        ),
        requires_grad=False,
    )
    weights.w2_weight_scale = torch.nn.Parameter(
        torch.full(
            (experts, hidden),
            1.0 / 128.0,
            dtype=torch.bfloat16,
            device=device,
        ),
        requires_grad=False,
    )
    if smooth_quant in {"w13", "both"}:
        weights.w13_smooth_scale = torch.nn.Parameter(
            torch.ones(experts, hidden, dtype=torch.float32, device=device),
            requires_grad=False,
        )
    if smooth_quant in {"w2", "both"}:
        weights.w2_smooth_scale = torch.nn.Parameter(
            torch.ones(experts, intermediate, dtype=torch.float32, device=device),
            requires_grad=False,
        )

    plan = tokenspeed_kernel.moe_plan(
        "int8",
        input_dtype=torch.bfloat16,
        activation="silu",
        routing_mode="precomputed_topk",
        ep_size=1,
        ispp=intermediate,
        internal_activation_dtype="int8",
        num_zero_experts=2,
    )
    assert plan["apply_kernel_name"] == "ascend_int8_precomputed_moe_apply"
    tokenspeed_kernel.moe_process_weights(plan, weights)
    assert torch_npu.get_npu_format(weights.w13_weight) == 29
    assert torch_npu.get_npu_format(weights.w2_weight) == 29

    x = torch.randn(tokens, hidden, dtype=torch.bfloat16, device=device)
    ids = torch.arange(tokens * top_k, dtype=torch.int32, device=device).view(
        tokens, top_k
    )
    ids = ids.remainder(experts + 2)
    route_weights = torch.rand(tokens, top_k, dtype=torch.float32, device=device)
    output = tokenspeed_kernel.moe_apply(
        plan,
        x,
        weights,
        route_weights,
        topk_weights=route_weights,
        topk_ids=ids,
    )
    torch.npu.synchronize()
    assert output.shape == x.shape
    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("smooth_quant", ("none", "w13", "w2", "both"))
@pytest.mark.parametrize("smooth_layout", ("expert", "shared", "vector"))
@pytest.mark.parametrize("solution", ("flash_npu_routed", "flash_npu_routed_full"))
def test_routed_w8a8_optional_smooth_matches_composed_graph(
    smooth_quant: str, smooth_layout: str, solution: str
) -> None:
    library = os.environ.get("TOKENSPEED_LITE_GMM13_LIBRARY")
    if not library:
        pytest.skip("set TOKENSPEED_LITE_GMM13_LIBRARY for routed fusion tests")
    if solution == "flash_npu_routed_full" and not os.environ.get(
        "TOKENSPEED_FUSED_MM2_LIBRARY"
    ):
        pytest.skip("set TOKENSPEED_FUSED_MM2_LIBRARY for full routed fusion tests")
    torch.ops.load_library(library)
    device = torch.device("npu:0")
    torch.manual_seed(981)
    experts, tokens, hidden, intermediate = 4, 17, 512, 1024
    w = torch.nn.Module()
    w.num_local_experts, w.num_experts = experts, experts * 2
    w.hidden_size, w.intermediate_size = hidden, intermediate
    w.ep_rank, w.ep_size = 1, 2
    for name, shape, dtype in (
        ("w13_weight", (experts, 2 * intermediate, hidden), torch.int8),
        ("w2_weight", (experts, hidden, intermediate), torch.int8),
        ("w13_weight_scale", (experts, 2 * intermediate), torch.float32),
        ("w2_weight_scale", (experts, hidden), torch.bfloat16),
    ):
        data = (
            torch.randint(-8, 8, shape, device=device, dtype=dtype)
            if dtype == torch.int8
            else torch.full(shape, 1 / 128, device=device, dtype=dtype)
        )
        w.register_parameter(name, torch.nn.Parameter(data, requires_grad=False))
    for name, width, enabled in (
        ("w13_smooth_scale", hidden, smooth_quant in {"w13", "both"}),
        ("w2_smooth_scale", intermediate, smooth_quant in {"w2", "both"}),
    ):
        data = (torch.rand(experts, width, device=device) * 1.5 + 0.25).bfloat16()
        if smooth_layout != "expert":
            data = data[:1]
        # Exercise independent None parameters, BF16 conversion, and strided
        # checkpoint-style storage. Preparation must persist FP32 contiguous data.
        data = data.transpose(0, 1).contiguous().transpose(0, 1)
        if smooth_layout == "vector":
            data = data.squeeze(0)
        w.register_parameter(
            name, torch.nn.Parameter(data, requires_grad=False) if enabled else None
        )
    kwargs = dict(
        input_dtype=torch.bfloat16,
        activation="silu",
        routing_mode="precomputed_topk",
        ep_size=2,
        ispp=intermediate,
        internal_activation_dtype="int8",
    )
    fused = tokenspeed_kernel.moe_plan("int8", solution=solution, **kwargs)
    reference = tokenspeed_kernel.moe_plan("int8", solution="torch_npu", **kwargs)
    tokenspeed_kernel.moe_process_weights(fused, w)
    for name in ("w13_smooth_scale", "w2_smooth_scale"):
        smooth = getattr(w, name)
        if smooth is not None:
            assert smooth.dtype == torch.float32 and smooth.is_contiguous()
    assert torch_npu.get_npu_format(w.w13_weight) == 29
    assert torch_npu.get_npu_format(w.w2_weight) == 29
    x = torch.randn(tokens, hidden, dtype=torch.bfloat16, device=device)
    ids = torch.randint(experts * 2 + 2, (tokens, 16), dtype=torch.int32, device=device)
    weights = torch.rand(tokens, 16, dtype=torch.float32, device=device)

    def run(plan):
        return tokenspeed_kernel.moe_apply(
            plan, x, w, weights, topk_ids=ids, topk_weights=weights
        )

    torch.testing.assert_close(run(fused), run(reference), rtol=0, atol=0)
    for _ in range(3):
        run(fused)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=torch.npu.Stream(), auto_dispatch_capture=True):
        output = run(fused)
    for step in range(20):
        x.copy_(torch.randn_like(x))
        ids.copy_(torch.randint_like(ids, experts * 2 + 2))
        if step == 10:
            ids.fill_(experts * 2)
        graph.replay()
        torch.testing.assert_close(output, run(reference), rtol=0, atol=0)
    torch.npu.synchronize()
    graph.reset()
