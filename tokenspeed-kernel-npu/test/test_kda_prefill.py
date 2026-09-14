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

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from tokenspeed_kernel.ops.attention.kda import (
    _attention_format_signature,
    kda_paged_prefill,
)
from tokenspeed_kernel.ops.attention.kda_reference import torch_kda_paged_prefill
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.selection import select_kernel
from tokenspeed_kernel_npu.ops import kda as kda_ops


def _case(lengths=(2, 0, 2), dim=16):
    torch.manual_seed(41)
    tokens = sum(lengths)
    boundaries = [0]
    for length in lengths:
        boundaries.append(boundaries[-1] + length)
    shape = (1, tokens, 1, dim)
    return {
        "q": torch.randn(shape, dtype=torch.bfloat16),
        "k": torch.randn(shape, dtype=torch.bfloat16),
        "v": torch.randn(shape, dtype=torch.bfloat16),
        "g_raw": torch.randn(shape, dtype=torch.bfloat16),
        "beta_logits": torch.randn(shape, dtype=torch.bfloat16),
        "A_log": torch.zeros(1, dtype=torch.float32),
        "dt_bias": torch.zeros(1, dim, dtype=torch.float32),
        "initial_state": torch.randn(len(lengths), 1, dim, dim, dtype=torch.float32),
        "cu_seqlens": torch.tensor(boundaries, dtype=torch.int32),
        "cu_seqlens_cpu": torch.tensor(boundaries, dtype=torch.int64),
        "lower_bound": -5.0,
    }


def test_ascend_prefill_registration_only_specializes_featurewise_beta():
    if not current_platform().is_npu:
        pytest.skip("Ascend registration")
    probe = torch.empty(0, dtype=torch.bfloat16, device="meta")
    signature = _attention_format_signature(q=probe, k=probe, v=probe)
    selected = {
        mode: select_kernel(
            "attention",
            "kda_paged_prefill",
            signature,
            traits={"beta_mode": mode},
        ).name
        for mode in ("scalar", "featurewise")
    }
    assert selected == {
        "scalar": "torch_ascend_kda_paged_prefill",
        "featurewise": "public_ascend_kda_paged_prefill",
    }


def test_missing_public_prefill_falls_back_unless_explicit(monkeypatch):
    case = _case(lengths=(2,))
    monkeypatch.setattr(kda_ops, "is_available", lambda _name: False)
    monkeypatch.setattr(
        kda_ops,
        "load_public_kda_ops",
        lambda: SimpleNamespace(reason="missing test artifact"),
    )
    expected = torch_kda_paged_prefill(**case)
    actual = kda_ops.public_kda_paged_prefill(**case)
    torch.testing.assert_close(actual.out, expected.out)
    torch.testing.assert_close(actual.final_state, expected.final_state)
    with pytest.raises(RuntimeError, match="missing test artifact"):
        kda_ops.public_kda_paged_prefill(**case, require_public=True)


def test_public_prefill_compacts_empty_segments_and_scatter(monkeypatch):
    case = _case()
    captured = {}

    def gate_cumsum(gate, chunk_size, **kwargs):
        captured["gate"] = (gate, chunk_size, kwargs)
        return torch.zeros_like(gate, dtype=torch.float32)

    def chunk_kda_fwd(q, k, v, gate, beta, scale, chunk_size, layout, **kwargs):
        captured["chunk"] = (
            q,
            k,
            v,
            gate,
            beta,
            scale,
            chunk_size,
            layout,
            kwargs,
        )
        return v.clone(), kwargs["initial_state"] + 10

    monkeypatch.setattr(kda_ops, "is_available", lambda _name: True)
    monkeypatch.setattr(
        torch.ops,
        "tokenspeed_npu_public_kda",
        SimpleNamespace(
            kda_gate_cumsum=gate_cumsum,
            chunk_kda_fwd=chunk_kda_fwd,
        ),
    )
    initial = case["initial_state"].clone()
    result = kda_ops.public_kda_paged_prefill(**case, require_public=True)

    gate, chunk_size, gate_kwargs = captured["gate"]
    q, k, v, _, beta, scale, _, layout, chunk_kwargs = captured["chunk"]
    beta_scale = torch.sqrt(torch.sigmoid(case["beta_logits"].float()) + 1e-10)
    torch.testing.assert_close(q, F.normalize(case["q"].float(), dim=-1).to(q.dtype))
    torch.testing.assert_close(
        k,
        (F.normalize(case["k"].float(), dim=-1) * beta_scale).to(k.dtype),
    )
    torch.testing.assert_close(v, (case["v"].float() * beta_scale).to(v.dtype))
    torch.testing.assert_close(beta, torch.ones_like(beta))
    assert gate is case["g_raw"]
    assert chunk_size == 64
    assert gate_kwargs["cu_seqlens"] == (0, 2, 4)
    assert chunk_kwargs["cu_seqlens"] == (0, 2, 4)
    assert chunk_kwargs["chunk_indices"] == (0, 0, 1, 0)
    assert scale == 16**-0.5 and layout == "BSND"
    torch.testing.assert_close(result.out, v)
    torch.testing.assert_close(result.final_state[0], initial[0] + 10)
    torch.testing.assert_close(result.final_state[1], initial[1])
    torch.testing.assert_close(result.final_state[2], initial[2] + 10)
    torch.testing.assert_close(case["initial_state"], initial)


def test_public_prefill_all_empty_skips_both_ops(monkeypatch):
    case = _case(lengths=(0, 0))
    monkeypatch.setattr(kda_ops, "is_available", lambda _name: True)

    def unexpected(*_args, **_kwargs):
        raise AssertionError("empty Prefill must not launch a public op")

    monkeypatch.setattr(
        torch.ops,
        "tokenspeed_npu_public_kda",
        SimpleNamespace(kda_gate_cumsum=unexpected, chunk_kda_fwd=unexpected),
    )
    result = kda_ops.public_kda_paged_prefill(**case, require_public=True)
    assert result.out.shape == case["v"].shape
    assert result.out.numel() == 0
    assert result.final_state is not case["initial_state"]
    torch.testing.assert_close(result.final_state, case["initial_state"])


def test_public_prefill_nonempty_returns_kernel_state_without_scatter(monkeypatch):
    case = _case(lengths=(32,))
    final_state = case["initial_state"] + 1
    monkeypatch.setattr(kda_ops, "is_available", lambda _name: True)
    monkeypatch.setattr(
        torch.ops,
        "tokenspeed_npu_public_kda",
        SimpleNamespace(
            kda_gate_cumsum=lambda gate, *_args, **_kwargs: gate.float(),
            chunk_kda_fwd=lambda *_args, **_kwargs: (case["v"], final_state),
        ),
    )
    result = kda_ops.public_kda_paged_prefill(**case)
    assert result.final_state is final_state


def test_public_prefill_rejects_invalid_host_boundaries_before_launch(monkeypatch):
    case = _case(lengths=(2,))
    case["cu_seqlens_cpu"] = torch.tensor([0, 1], dtype=torch.int64)
    monkeypatch.setattr(kda_ops, "is_available", lambda _name: True)
    with pytest.raises(ValueError, match="cover the packed input"):
        kda_ops.public_kda_paged_prefill(**case, require_public=True)


def test_public_prefill_short_auto_selection_uses_reference(monkeypatch):
    case = _case(lengths=(2,))
    monkeypatch.setattr(kda_ops, "is_available", lambda _name: True)

    def unexpected(*_args, **_kwargs):
        raise AssertionError("short automatic Prefill must use the reference")

    monkeypatch.setattr(
        torch.ops,
        "tokenspeed_npu_public_kda",
        SimpleNamespace(kda_gate_cumsum=unexpected, chunk_kda_fwd=unexpected),
    )
    expected = torch_kda_paged_prefill(**case)
    actual = kda_ops.public_kda_paged_prefill(**case)
    torch.testing.assert_close(actual.out, expected.out)
    torch.testing.assert_close(actual.final_state, expected.final_state)


@pytest.mark.parametrize("lengths", [(4,), (1, 63), (32, 0, 32), (65, 127)])
def test_npu_public_prefill_matches_reference(lengths):
    if not current_platform().is_npu:
        pytest.skip("requires an Ascend NPU")
    torch.npu.set_device(0)
    device = torch.device("npu:0")
    torch.manual_seed(sum(lengths) + 53)
    tokens, heads, dim = sum(lengths), 4, 128
    shape = (1, tokens, heads, dim)
    q = torch.randn(shape, device=device, dtype=torch.bfloat16)
    k = torch.randn(shape, device=device, dtype=torch.bfloat16)
    v = torch.randn(shape, device=device, dtype=torch.bfloat16)
    gate = torch.randn(shape, device=device, dtype=torch.bfloat16)
    beta = torch.randn(shape, device=device, dtype=torch.bfloat16)
    A_log = torch.linspace(-0.2, 0.2, heads, device=device)
    dt_bias = torch.randn(heads, dim, device=device) * 0.1
    initial = (
        torch.randn(len(lengths), heads, dim, dim, device=device, dtype=torch.float32)
        * 0.01
    )
    initial_copy = initial.clone()
    boundaries = [0]
    for length in lengths:
        boundaries.append(boundaries[-1] + length)
    cu_cpu = torch.tensor(boundaries, dtype=torch.int64)
    cu = cu_cpu.to(device=device, dtype=torch.int32)
    kwargs = dict(
        initial_state=initial,
        cu_seqlens=cu,
        cu_seqlens_cpu=cu_cpu,
        lower_bound=-5.0,
    )
    expected = kda_paged_prefill(
        q, k, v, gate, beta, A_log, dt_bias, solution="torch", **kwargs
    )
    actual = kda_paged_prefill(
        q, k, v, gate, beta, A_log, dt_bias, solution="public_kda", **kwargs
    )
    torch.npu.synchronize()

    output_rel = (
        actual.out.float() - expected.out.float()
    ).norm() / expected.out.float().norm().clamp_min(1e-12)
    state_rel = (
        actual.final_state - expected.final_state
    ).norm() / expected.final_state.norm().clamp_min(1e-12)
    assert float(output_rel.cpu()) < 0.01
    assert float(state_rel.cpu()) < 0.005
    assert torch.isfinite(actual.out).all()
    assert torch.isfinite(actual.final_state).all()
    torch.testing.assert_close(initial, initial_copy, atol=0, rtol=0)
    for index, length in enumerate(lengths):
        if length == 0:
            torch.testing.assert_close(
                actual.final_state[index], initial[index], atol=0, rtol=0
            )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("tokens", [32, 63, 65, 256, 4096])
@pytest.mark.parametrize("seed", [3, 11, 29, 97])
def test_npu_fused_prefill_prepare_matches_torch_with_strides(dtype, tokens, seed):
    if not current_platform().is_npu:
        pytest.skip("requires an Ascend NPU")
    from tokenspeed_kernel_npu.ops.kda_prepare import prepare_kda_inputs

    torch.npu.set_device(0)
    torch.manual_seed(seed)
    packed = torch.randn(1, tokens, 3, 4, 128, device="npu:0", dtype=dtype)
    q, k, v = packed.unbind(2)
    beta = torch.randn(1, tokens, 4, 256, device="npu:0", dtype=dtype)[..., :128]
    # Include zero-norm rows and saturated sigmoid at both ends.
    q[:, 0].zero_()
    k[:, 1].zero_()
    beta[:, 0].fill_(-80)
    beta[:, 1].fill_(80)
    scale = torch.sqrt(torch.sigmoid(beta.float()) + 1e-10)
    expected = (
        F.normalize(q.float(), p=2, dim=-1).to(dtype),
        (F.normalize(k.float(), p=2, dim=-1) * scale).to(dtype),
        (v.float() * scale).to(dtype),
        torch.ones(q.shape[:-1], device="npu:0", dtype=dtype),
    )
    for _ in range(3):
        actual = prepare_kda_inputs(q, k, v, beta)
        for result, reference in zip(actual, expected):
            assert result.is_contiguous() and torch.isfinite(result).all()
            torch.testing.assert_close(result, reference, atol=0, rtol=0)
