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

"""SwiGLU runtime dispatch, INT8 scale semantics and Ascend graph regression."""

import pytest
import torch

from tokenspeed.runtime.layers.activation import SiluAndMul


def _npu():
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend required")
    torch.npu.set_device(0)
    return "npu"


def _reference(x, limit=None):
    gate, up = x.float().chunk(2, -1)
    if limit is not None:
        gate = gate.clamp_max(limit)
        up = up.clamp(-limit, limit)
    return torch.nn.functional.silu(gate) * up


@pytest.mark.parametrize("limit", [None, 1.0])
def test_cpu_swiglu(limit):
    x = torch.randn(3, 128, dtype=torch.bfloat16)
    torch.testing.assert_close(
        SiluAndMul(limit)(x), _reference(x, limit).to(x.dtype), rtol=0, atol=0
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize(
    "shape", [(1, 1024), (32, 4096), (128, 4096), (2, 3, 512), (0, 4096)]
)
@pytest.mark.parametrize("int8_out", [False, True])
def test_npu_swiglu(dtype, shape, int8_out):
    x = torch.randn(shape, device=_npu(), dtype=dtype)
    result = SiluAndMul()(x, int8_out=int8_out)
    ref = _reference(x)
    if int8_out:
        values, scales = result
        assert values.dtype == torch.int8 and scales.dtype == torch.float32
        assert scales.shape == x.shape[:-1]
        actual = values.float() * scales.unsqueeze(-1)
        if x.numel():
            # Half a quantization bin, plus floating evaluation rounding.
            bound = ref.abs().amax(-1, keepdim=True) / 127 * 0.51 + 1e-5
            assert ((actual - ref).abs() <= bound).all()
    else:
        torch.testing.assert_close(result, ref.to(dtype), atol=1e-5, rtol=0.008)


def test_npu_clamp_and_output_buffer():
    from tokenspeed_kernel.ops.activation import silu_and_mul

    x = torch.randn(32, 4096, device=_npu(), dtype=torch.bfloat16)
    out = torch.empty(32, 2048, device=x.device, dtype=x.dtype)
    assert silu_and_mul(x, out).data_ptr() == out.data_ptr()
    torch.testing.assert_close(out, _reference(x).to(x.dtype))
    torch.testing.assert_close(SiluAndMul(1.0)(x), _reference(x, 1.0).to(x.dtype))
    with pytest.raises(ValueError, match="out shape"):
        silu_and_mul(x, out[:, :1])


@pytest.mark.parametrize("int8_out", [False, True])
def test_npu_graph_changes_input(int8_out):
    x = torch.randn(32, 4096, device=_npu(), dtype=torch.bfloat16)
    layer = SiluAndMul()
    for _ in range(3):
        layer(x, int8_out=int8_out)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        result = layer(x, int8_out=int8_out)
    for zero in (False, True):
        x.copy_(torch.zeros_like(x) if zero else torch.randn_like(x))
        graph.replay()
        torch.npu.synchronize()
        expected = layer(x, int8_out=int8_out)
        outputs = zip(result, expected) if int8_out else [(result, expected)]
        for actual, ref in outputs:
            torch.testing.assert_close(actual, ref, rtol=0, atol=0)
        if int8_out:
            assert torch.isfinite(result[1]).all()
            if zero:
                assert (result[0].float() * result[1][:, None] == 0).all()


def test_npu_int32_dequant_swiglu():
    x = torch.randint(-10000, 10000, (32, 4096), device=_npu(), dtype=torch.int32)
    w_scale = torch.rand(4096, device=x.device) * 0.01
    a_scale = torch.rand(32, device=x.device) * 0.01
    values, scales = SiluAndMul()(
        x, int8_out=True, weight_scale=w_scale, activation_scale=a_scale
    )
    ref = _reference(x.float() * w_scale * a_scale[:, None])
    actual = values.float() * scales[:, None]
    assert (
        (actual - ref).abs() <= ref.abs().amax(-1, keepdim=True) / 127 * 0.51 + 1e-5
    ).all()


def test_quantized_swiglu_rejects_ambiguous_flags():
    x = torch.ones(2, 64)
    with pytest.raises(ValueError, match="mutually exclusive"):
        SiluAndMul()(x, fp8_out=True, int8_out=True)
    with pytest.raises(ValueError, match="requires weight"):
        SiluAndMul()(x.int(), int8_out=True)
    with pytest.raises(NotImplementedError, match="Clamped"):
        SiluAndMul(1.0)(x, int8_out=True)
