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

"""INT8 dense/shared expert checkpoint layout and NPU execution tests."""

import os
from types import SimpleNamespace

import pytest
import torch

from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from tokenspeed.runtime.models.deepseek_v3 import DeepseekV3MLP
from tokenspeed.runtime.models.flash_local_moe import GroupAwareFlashLocalMoE


def _config(ignore=()):
    return CompressedTensorsConfig.from_config(
        {
            "format": "int-quantized",
            "ignore": list(ignore),
            "config_groups": {
                "g": {
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


def _mlp(device, hidden=256, intermediate=128, dtype=torch.bfloat16):
    with torch.device(device):
        return DeepseekV3MLP(
            hidden,
            intermediate,
            "silu",
            Mapping(rank=0),
            quant_config=_config(),
            params_dtype=dtype,
        ).eval()


def test_shared_checkpoint_loaders():
    model = _mlp("cpu")
    gate = torch.randint(-8, 8, (128, 256), dtype=torch.int8)
    up = torch.randint(-8, 8, (128, 256), dtype=torch.int8)
    gs, us = torch.rand(128, 1), torch.rand(128, 1)
    for name, values in (("weight", (gate, up)), ("weight_scale", (gs, us))):
        param = getattr(model.gate_up_proj, name)
        for shard, value in enumerate(values):
            param.weight_loader(param, value, shard)
        torch.testing.assert_close(param, torch.cat(values), rtol=0, atol=0)
    for name, value in (
        ("weight", torch.randint(-8, 8, (256, 128), dtype=torch.int8)),
        ("weight_scale", torch.rand(256, 1)),
    ):
        param = getattr(model.down_proj, name)
        param.weight_loader(param, value)
        torch.testing.assert_close(param, value, rtol=0, atol=0)


@pytest.mark.parametrize("ignored", [False, True])
def test_group_aware_shared_respects_quant_config(ignored):
    config = SimpleNamespace(
        hidden_size=128,
        moe_group_size=8,
        expert_ffn_hidden_size=32,
        n_routed_experts=4,
        zero_expert_num=2,
        moe_topk=2,
        rms_norm_eps=1e-5,
        ffn_hidden_size=64,
        n_shared_experts=1,
    )
    prefix = "model.layers.0.moe"
    quant = _config(
        [
            prefix + ".shared_experts.gate_up_proj",
            prefix + ".shared_experts.down_proj",
        ]
        if ignored
        else []
    )
    model = GroupAwareFlashLocalMoE(
        config,
        Mapping(rank=0, world_size=16, moe_ep_size=16),
        quant_config=quant,
        prefix=prefix,
    )
    expected = torch.bfloat16 if ignored else torch.int8
    assert model.shared_experts.gate_up_proj.weight.dtype == expected
    assert model.shared_experts.down_proj.weight.dtype == expected
    assert model.shared_experts._use_int8_swiglu == (not ignored)


def _quantize(x):
    scale = x.abs().amax(-1).clamp_min(1e-10) / 127
    return (x / scale[:, None]).round().clamp(-128, 127), scale


@pytest.mark.parametrize(
    "tokens,hidden,intermediate",
    [(0, 256, 128), (1, 256, 128), (32, 4096, 2048), (128, 256, 128)],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_shared_w8a8_nz_and_graph(tokens, hidden, intermediate, dtype):
    torch_npu = pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend required")
    torch.npu.set_device(0)
    model = _mlp("npu", hidden, intermediate, dtype)
    with torch.no_grad():
        for proj in (model.gate_up_proj, model.down_proj):
            proj.weight.random_(-8, 8)
            proj.weight_scale.fill_(1 / 128)
            proj.quant_method.process_weights_after_loading(proj)
            assert proj.weight.dtype == torch.int8
            assert torch_npu.get_npu_format(proj._int8_plan.weight) == 29
    x = torch.randn(tokens, hidden, device="npu", dtype=dtype)
    with torch.no_grad():
        result = model(x)
    assert result.shape == x.shape and result.dtype == dtype
    if not tokens:
        return
    # Independent CPU reference: dynamic INT8 input, INT32-equivalent matmul,
    # FP32 SwiGLU, dynamic requant, then a second integer matmul.
    q, s = _quantize(x.cpu().float())
    a = (q @ model.gate_up_proj.weight.cpu().float().t()) * s[:, None] / 128
    gate, up = a.chunk(2, -1)
    q2, s2 = _quantize(torch.nn.functional.silu(gate) * up)
    ref = (q2 @ model.down_proj.weight.cpu().float().t()) * s2[:, None] / 128
    error = (result.cpu().float() - ref).abs()
    assert (error <= ref.abs().amax(-1, keepdim=True) * 0.015 + 1e-4).all()

    for _ in range(3):
        model(x)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = model(x)
    x.copy_(torch.randn_like(x))
    graph.replay()
    torch.npu.synchronize()
    torch.testing.assert_close(output, model(x), rtol=0, atol=0)


@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("tokens", [0, 1, 32, 64, 257, 512])
def test_fused_shared_ffn_bitwise(quantized, tokens):
    """Real runtime MLP oracle, post-load NZ reuse and limited graph replay."""
    binding = os.environ.get("FUSED_FFN_BINDING")
    if not binding:
        pytest.skip("Set FUSED_FFN_BINDING to test the optional FFN")
    pytest.importorskip("torch_npu")
    from tokenspeed_kernel_npu.ops.fused_ffn import prepare_shared_ffn

    torch.npu.set_device(0)
    torch.manual_seed(901 + tokens)
    with torch.device("npu"):
        model = DeepseekV3MLP(
            4096,
            2048,
            "silu",
            Mapping(rank=0),
            quant_config=_config() if quantized else None,
            params_dtype=torch.bfloat16,
        ).eval()
    # Production traverses parent before child post-load hooks. Preparation
    # must not freeze stale plans or allocate another checkpoint weight copy.
    fused = prepare_shared_ffn(model, binding=binding)
    with torch.no_grad():
        for proj in (model.gate_up_proj, model.down_proj):
            if quantized:
                proj.weight.random_(-128, 128)
                proj.weight_scale.uniform_(0.0001, 0.001)
            else:
                proj.weight.normal_(std=0.01)
            proj.quant_method.process_weights_after_loading(proj)
    x = torch.randn(tokens, 4096, device="npu", dtype=torch.bfloat16)

    def assert_bits(actual, expected):
        actual, expected = actual.cpu(), expected.cpu()
        assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
        assert actual.shape == expected.shape and actual.dtype == expected.dtype
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))

    stream = torch.npu.Stream()
    torch.npu.set_stream_limit(stream, cube_num=4, vector_num=8)
    with torch.inference_mode():
        expected = model(x)
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            actual = fused(x)
        torch.npu.current_stream().wait_stream(stream)
        assert_bits(actual, expected)
        if not tokens:
            return
        for _ in range(3):
            with torch.npu.stream(stream):
                fused(x)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream):
            output = fused(x)
        for _ in range(10):
            x.normal_()
            expected = model(x)
            torch.npu.synchronize()
            graph.replay()
            assert_bits(output, expected)
            snapshot = output.clone()
            graph.replay()
            assert_bits(output, snapshot)
        graph.reset()
