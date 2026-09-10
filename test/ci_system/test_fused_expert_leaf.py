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

"""Selected BF16/W8A8 leaf against the original init/GMM/activation/finalize.

Requires the two BF16-capable bindings via TOKENSPEED_LITE_GMM13_LIBRARY and
TOKENSPEED_FUSED_MM2_LIBRARY. This is single-device expert-shard coverage, not
an EP8/EP16 communication or full-layer acceptance result.
"""

import os

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")

from tokenspeed_kernel_npu.ops.moe import (
    ascend_bf16_precomputed_moe_apply,
    ascend_int8_precomputed_moe_apply,
    ascend_routed_bf16_precomputed_moe_apply,
    ascend_routed_full_bf16_process_moe_weights,
    ascend_routed_full_int8_process_moe_weights,
    ascend_routed_int8_precomputed_moe_apply,
)

pytestmark = pytest.mark.skipif(
    not os.environ.get("TOKENSPEED_LITE_GMM13_LIBRARY")
    or not os.environ.get("TOKENSPEED_FUSED_MM2_LIBRARY"),
    reason="explicit BF16-capable experimental bindings required",
)


_COMMON_SHAPES = [
    (32, 512, 1024, 192, 16, 2, "mlp_split"),
    (1, 512, 1024, 384, 16, 1, "mlp_split"),
    (7, 8192, 32, 4, 3, 2, "mlp_split"),
    (33, 1024, 512, 17, 7, 4, "mlp_split"),
    (1, 32, 32, 1024, 64, 1, "mlp_split"),
    (32684, 32, 64, 17, 3, 2, "mlp_split"),
    (7, 8192, 4096, 4, 3, 2, "mlp_split"),
]


@pytest.mark.parametrize(
    "weight_dtype,tokens,hidden,intermediate,experts,topk,egp,activation_oracle",
    [(dtype, *shape) for dtype in ("bf16", "int8") for shape in _COMMON_SHAPES]
    + [
        # Stock mlp_split_swiglu rejects I8192 during tiling. This explicitly
        # distinct oracle uses stock npu_swiglu, not a runtime fallback or an
        # assertion that the unsupported original chain passed.
        ("bf16", 1, 8192, 8192, 1, 1, 1, "npu_swiglu"),
    ],
)
def test_fused_leaf(
    weight_dtype, tokens, hidden, intermediate, experts, topk, egp, activation_oracle
):
    torch.npu.set_device(0)
    torch.manual_seed(311)
    w = torch.nn.Module()
    w.ep_size, w.ep_rank = egp, egp - 1
    w.num_local_experts, w.num_experts = experts, experts * egp
    w.hidden_size, w.intermediate_size = hidden, intermediate
    w.w13_weight = torch.nn.Parameter(
        (torch.randn(experts, 2 * intermediate, hidden) / hidden**0.5).bfloat16().npu(),
        requires_grad=False,
    )
    w.w2_weight = torch.nn.Parameter(
        (torch.randn(experts, hidden, intermediate) / intermediate**0.5)
        .bfloat16()
        .npu(),
        requires_grad=False,
    )
    plan = dict(activation="silu", num_zero_experts=32)
    if weight_dtype == "int8":
        for name, shape in (
            ("w13", (experts, 2 * intermediate, hidden)),
            ("w2", (experts, hidden, intermediate)),
        ):
            setattr(
                w,
                f"{name}_weight",
                torch.nn.Parameter(
                    torch.randint(-8, 8, shape, dtype=torch.int8, device="npu"),
                    requires_grad=False,
                ),
            )
            scales = torch.rand(shape[:2], device="npu") * 0.01 + 0.001
            setattr(
                w,
                f"{name}_weight_scale",
                torch.nn.Parameter(
                    scales.bfloat16() if name == "w2" else scales, requires_grad=False
                ),
            )
        for name, width in (("w13", hidden), ("w2", intermediate)):
            setattr(
                w,
                f"{name}_smooth_scale",
                torch.nn.Parameter(
                    torch.rand(experts, width, device="npu") * 1.5 + 0.25,
                    requires_grad=False,
                ),
            )
        ascend_routed_full_int8_process_moe_weights(plan=plan, w=w)
    else:
        ascend_routed_full_bf16_process_moe_weights(plan=plan, w=w)
    x = torch.randn(tokens, hidden).bfloat16().npu()
    ids = torch.randint(
        w.num_experts + 32, (tokens, topk), dtype=torch.int32, device="npu"
    )
    ids[:, 0] = w.ep_rank * experts  # Every shape must execute real expert math.
    weights = torch.rand(tokens, topk, device="npu") / topk

    def run(fused):
        if not fused and activation_oracle == "npu_swiglu":
            expanded, rows, counts, _ = torch_npu.npu_moe_init_routing_v2(
                x,
                ids,
                active_num=ids.numel(),
                expert_num=w.num_experts + 32,
                quant_mode=-1,
                active_expert_range=[w.ep_rank * experts, (w.ep_rank + 1) * experts],
                expert_tokens_num_type=1,
                expert_tokens_num_flag=True,
                row_idx_type=0,
                drop_pad_mode=0,
            )
            z = torch_npu.npu_grouped_matmul(
                [expanded],
                [w.w13_weight],
                group_list=counts,
                split_item=3,
                output_dtype=torch.bfloat16,
                group_type=0,
                group_list_type=1,
            )[0]
            z = torch_npu.npu_swiglu(z, dim=-1)
            z = torch_npu.npu_grouped_matmul(
                [z],
                [w.w2_weight],
                group_list=counts,
                split_item=3,
                output_dtype=torch.bfloat16,
                group_type=0,
                group_list_type=1,
            )[0]
            return torch_npu.npu_moe_finalize_routing(
                z.unsqueeze(1), None, None, None, weights.bfloat16(), rows, ids, 3
            )
        if weight_dtype == "int8":
            fn = (
                ascend_routed_int8_precomputed_moe_apply
                if fused
                else ascend_int8_precomputed_moe_apply
            )
        else:
            fn = (
                ascend_routed_bf16_precomputed_moe_apply
                if fused
                else ascend_bf16_precomputed_moe_apply
            )
        return fn(
            plan=plan, x=x, w=w, router_logits=None, topk_weights=weights, topk_ids=ids
        )

    def exact(a, b):
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        torch.testing.assert_close(
            a.cpu().view(torch.int16), b.cpu().view(torch.int16), rtol=0, atol=0
        )

    exact(run(True), run(False))
    stream = torch.npu.Stream()
    with torch.npu.graph(
        graph := torch.npu.NPUGraph(), stream=stream, auto_dispatch_capture=True
    ):
        actual = run(True)
    for _ in range(3):
        x.normal_()
        ids.random_(0, w.num_experts + 32)
        ids[:, 0] = w.ep_rank * experts
        weights.uniform_(0, 1 / topk)
        graph.replay()
        expected = run(False)
        torch.npu.synchronize()
        exact(actual, expected)
    graph.reset()
