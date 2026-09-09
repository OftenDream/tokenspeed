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

import importlib.util
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from tokenspeed_kernel_npu.ops import kda as kda_ops
from tokenspeed_kernel_npu.ops.kda import ref_kda_causal_conv1d


@pytest.mark.parametrize("supports_stride", [False, True])
def test_flash_causal_conv_loader_rejects_old_package(monkeypatch, supports_stride):
    package = SimpleNamespace()
    if supports_stride:
        package.CAUSAL_CONV1D_STATE_SLOT_STRIDE = True
    monkeypatch.setitem(sys.modules, "flash_ops", package)
    # Bypass the production import cache without changing its cached value.
    loaded = kda_ops._load_flash_causal_conv_ops.__wrapped__()
    assert loaded is (torch.ops.custom if supports_stride else None)


def _npu_available() -> bool:
    if importlib.util.find_spec("torch_npu") is None:
        return False
    import torch_npu  # noqa: F401

    return torch.npu.is_available()


def _oracle(projected, weight, state, reads, writes, boundaries, initial):
    output = torch.zeros_like(projected)
    final = state.clone()
    for row, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        read, write = reads[row], writes[row]
        if start == end or read < 0 or write < 0:
            continue
        history = state[read].clone() if initial[row] else torch.zeros_like(state[read])
        for token in range(start, end):
            window = torch.cat((history, projected[token, :, None]), dim=-1)
            output[token] = F.silu((window.float() * weight.float()).sum(-1)).to(
                output.dtype
            )
            history = window[:, 1:]
        final[write] = history
    return output, final


def test_ref_prefill_matches_oracle_and_preserves_padding_page():
    torch.manual_seed(17)
    projected = torch.randn(4, 3)
    weight = torch.randn(3, 4)
    state = torch.randn(5, 3, 3)
    reads = torch.tensor([1, 2, -1], dtype=torch.int32)
    writes = torch.tensor([3, 4, -1], dtype=torch.int32)
    boundaries = torch.tensor([0, 1, 4, 4], dtype=torch.int64)
    initial = torch.tensor([True, False, False])
    expected_output, expected_state = _oracle(
        projected,
        weight,
        state,
        reads.tolist(),
        writes.tolist(),
        boundaries.tolist(),
        initial.tolist(),
    )

    actual_state = state.clone()
    actual_output = ref_kda_causal_conv1d(
        projected=projected,
        weight=weight,
        conv_state=actual_state,
        read_indices=reads,
        write_indices=writes,
        cu_seqlens=boundaries,
        cu_seqlens_cpu=boundaries,
        bias=None,
        has_initial_state=initial,
        activation="silu",
        decode=False,
    )

    torch.testing.assert_close(actual_output, expected_output)
    torch.testing.assert_close(actual_state, expected_state)


def test_ref_decode_matches_oracle_with_independent_pages():
    torch.manual_seed(23)
    projected = torch.randn(3, 3)
    weight = torch.randn(3, 4)
    state = torch.randn(7, 3, 3)
    reads = torch.tensor([1, -1, 2], dtype=torch.int32)
    writes = torch.tensor([5, -1, 6], dtype=torch.int32)
    boundaries = torch.arange(4, dtype=torch.int64)
    initial = [True, False, True]
    expected_output, expected_state = _oracle(
        projected,
        weight,
        state,
        reads.tolist(),
        writes.tolist(),
        boundaries.tolist(),
        initial,
    )

    actual_state = state.clone()
    actual_output = ref_kda_causal_conv1d(
        projected=projected,
        weight=weight,
        conv_state=actual_state,
        read_indices=reads,
        write_indices=writes,
        cu_seqlens=boundaries,
        cu_seqlens_cpu=None,
        bias=None,
        has_initial_state=None,
        activation="silu",
        decode=True,
    )

    torch.testing.assert_close(actual_output, expected_output)
    torch.testing.assert_close(actual_state, expected_state)


@pytest.mark.parametrize("decode", [False, True])
def test_ref_updates_width_major_state_storage(decode):
    torch.manual_seed(24)
    projected = torch.randn(2, 4)
    weight = torch.randn(4, 4)
    channel_major = torch.randn(6, 4, 3)
    width_major_state = channel_major.transpose(1, 2).contiguous()
    reads = torch.tensor([1, 2], dtype=torch.int32)
    writes = torch.tensor([4, 5], dtype=torch.int32)
    boundaries = torch.arange(3, dtype=torch.int64)
    expected_output, expected_state = _oracle(
        projected,
        weight,
        channel_major,
        reads.tolist(),
        writes.tolist(),
        boundaries.tolist(),
        [True, True],
    )

    actual_output = ref_kda_causal_conv1d(
        projected=projected,
        weight=weight,
        conv_state=width_major_state,
        read_indices=reads,
        write_indices=writes,
        cu_seqlens=boundaries,
        cu_seqlens_cpu=boundaries,
        bias=None,
        has_initial_state=None,
        activation="silu",
        decode=decode,
    )

    torch.testing.assert_close(actual_output, expected_output)
    torch.testing.assert_close(width_major_state, expected_state.transpose(1, 2))


@pytest.mark.parametrize("layout", ["width_major", "strided_width_major"])
def test_public_prefill_preserves_width_major_storage(monkeypatch, layout):
    def fake_public(x, weight, state, **kwargs):
        assert kwargs["run_mode"] == 0
        assert state is original_view
        assert state.shape == (5, 3, 4)
        assert kwargs["cache_indices"].tolist() == [1, -1]
        assert kwargs["write_indices"].tolist() == [3, -1]
        state[3].fill_(7)
        return x.clone()

    monkeypatch.setattr(
        kda_ops,
        "_load_flash_causal_conv_ops",
        lambda: SimpleNamespace(npu_causal_conv1d=fake_public),
    )
    state = torch.randn(5, 3, 4)
    if layout == "strided_width_major":
        backing = torch.full((5, 40), 9.0)
        view = backing[:, :12].view(5, 3, 4)
        view.copy_(state)
        state = view
        assert not state.is_contiguous()
    original_view = state
    original = state.clone()
    kda_ops.public_kda_causal_conv1d(
        projected=torch.randn(2, 4),
        weight=torch.randn(4, 4),
        conv_state=state,
        read_indices=torch.tensor([1, -1], dtype=torch.int32),
        write_indices=torch.tensor([3, -1], dtype=torch.int32),
        cu_seqlens=torch.tensor([0, 1, 2], dtype=torch.int64),
        cu_seqlens_cpu=torch.tensor([0, 1, 2], dtype=torch.int64),
        bias=None,
        has_initial_state=None,
        activation="silu",
        decode=False,
        require_public=True,
    )
    assert torch.all(state[3] == 7)
    torch.testing.assert_close(state[[0, 1, 2, 4]], original[[0, 1, 2, 4]])
    if layout == "strided_width_major":
        assert torch.all(backing[:, 12:] == 9)


def test_flash_ops_decode_passes_width_major_state_and_independent_pages(monkeypatch):
    captured = {}

    def fake_public(x, weight, state, **kwargs):
        captured.update(
            calls=captured.get("calls", 0) + 1,
            x=x,
            weight=weight,
            state=state,
            kwargs=kwargs,
        )
        return x.clone()

    monkeypatch.setattr(
        kda_ops,
        "_load_flash_causal_conv_ops",
        lambda: SimpleNamespace(npu_causal_conv1d=fake_public),
    )
    projected = torch.randn(2, 4, dtype=torch.bfloat16)
    weight = torch.randn(4, 4, dtype=torch.bfloat16)
    state = torch.randn(6, 3, 4, dtype=torch.bfloat16)
    reads = torch.tensor([1, 2], dtype=torch.int32)
    writes = torch.tensor([4, 5], dtype=torch.int32)

    output = kda_ops.public_kda_causal_conv1d(
        projected=projected,
        weight=weight,
        conv_state=state,
        read_indices=reads,
        write_indices=writes,
        cu_seqlens=torch.arange(3, dtype=torch.int32),
        cu_seqlens_cpu=None,
        bias=None,
        has_initial_state=None,
        activation="silu",
        decode=True,
        require_public=True,
    )

    assert captured["state"] is state
    assert captured["weight"].shape == (4, 4)
    assert captured["kwargs"] == {
        "bias": None,
        "cache_indices": reads,
        "write_indices": writes,
        "activation_mode": 1,
        "pad_slot_id": -1,
        "run_mode": 1,
    }
    torch.testing.assert_close(output, projected)

    monkeypatch.setattr(
        kda_ops.torch,
        "npu",
        SimpleNamespace(is_current_stream_capturing=lambda: False),
        raising=False,
    )
    automatic = kda_ops.public_kda_causal_conv1d(
        projected=projected,
        weight=weight,
        conv_state=state,
        read_indices=reads,
        write_indices=writes,
        cu_seqlens=torch.arange(3, dtype=torch.int32),
        cu_seqlens_cpu=None,
        bias=None,
        has_initial_state=None,
        activation="silu",
        decode=True,
    )
    assert captured["calls"] == 2
    assert captured["state"] is state
    torch.testing.assert_close(automatic, projected)


@pytest.mark.parametrize("decode", [False, True])
@pytest.mark.parametrize("invalid_layout", ["channel_major", "strided_inner"])
def test_public_rejects_non_width_major_state(monkeypatch, decode, invalid_layout):
    monkeypatch.setattr(
        kda_ops,
        "_load_flash_causal_conv_ops",
        lambda: pytest.fail("invalid state must be rejected before loading the op"),
    )
    state = (
        torch.zeros(5, 4, 3)
        if invalid_layout == "channel_major"
        else torch.zeros(5, 3, 8)[:, :, ::2]
    )
    with pytest.raises(ValueError, match="public causal-conv requires state"):
        kda_ops.public_kda_causal_conv1d(
            projected=torch.zeros(1, 4),
            weight=torch.zeros(4, 4),
            conv_state=state,
            read_indices=torch.tensor([1], dtype=torch.int32),
            write_indices=torch.tensor([2], dtype=torch.int32),
            cu_seqlens=torch.tensor([0, 1], dtype=torch.int64),
            cu_seqlens_cpu=torch.tensor([0, 1], dtype=torch.int64),
            bias=None,
            has_initial_state=None,
            activation="silu",
            decode=decode,
            require_public=True,
        )


def test_missing_flash_ops_falls_back_unless_explicit(monkeypatch):
    monkeypatch.setattr(kda_ops, "_load_flash_causal_conv_ops", lambda: None)
    projected = torch.randn(2, 2)
    weight = torch.randn(2, 4)
    state = torch.randn(3, 3, 2)
    indices = torch.tensor([1], dtype=torch.int32)
    boundaries = torch.tensor([0, 2], dtype=torch.int64)

    expected_state = state.clone()
    expected = ref_kda_causal_conv1d(
        projected=projected,
        weight=weight,
        conv_state=expected_state,
        read_indices=indices,
        write_indices=torch.tensor([2], dtype=torch.int32),
        cu_seqlens=boundaries,
        cu_seqlens_cpu=boundaries,
        bias=None,
        has_initial_state=None,
        activation="silu",
        decode=False,
    )
    actual_state = state.clone()
    actual = kda_ops.public_kda_causal_conv1d(
        projected=projected,
        weight=weight,
        conv_state=actual_state,
        read_indices=indices,
        write_indices=torch.tensor([2], dtype=torch.int32),
        cu_seqlens=boundaries,
        cu_seqlens_cpu=boundaries,
        bias=None,
        has_initial_state=None,
        activation="silu",
        decode=False,
    )

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_state, expected_state)
    with pytest.raises(
        RuntimeError, match="flash_ops schema custom::npu_causal_conv1d"
    ):
        kda_ops.public_kda_causal_conv1d(
            projected=projected,
            weight=weight,
            conv_state=state,
            read_indices=indices,
            write_indices=torch.tensor([2], dtype=torch.int32),
            cu_seqlens=boundaries,
            cu_seqlens_cpu=boundaries,
            bias=None,
            has_initial_state=None,
            activation="silu",
            decode=False,
            require_public=True,
        )


@pytest.mark.skipif(not _npu_available(), reason="requires an Ascend NPU")
@pytest.mark.parametrize("batch", [1, 16])
@pytest.mark.parametrize("decode", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_npu_default_fused_matches_ref(monkeypatch, batch, decode, dtype):
    from tokenspeed_kernel.ops.attention.kda import (
        _attention_format_signature,
        kda_causal_conv1d,
    )
    from tokenspeed_kernel.selection import select_kernel

    torch.npu.set_device(0)
    device = torch.device("npu:0")
    probe = torch.empty(1, 1536, dtype=torch.bfloat16, device=device)
    signature = _attention_format_signature(projected=probe, weight=probe)
    expected = {
        ("decode", "small"): "public_ascend_kda_causal_conv1d_decode",
        ("prefill", "small"): "public_ascend_kda_causal_conv1d",
        ("prefill", "large"): "public_ascend_kda_causal_conv1d",
    }
    for (mode, batch_class), name in expected.items():
        selected = select_kernel(
            "attention",
            "kda_causal_conv1d",
            signature,
            traits={
                "forward_mode": mode,
                "batch_class": batch_class,
                "activation": "silu",
                "width": 4,
            },
        )
        assert selected.name == name

    torch.manual_seed(29)
    lengths = [1 if decode else index % 4 + 1 for index in range(batch)]
    boundaries_cpu = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int64
    )
    boundaries = boundaries_cpu.to(device=device, dtype=torch.int32)
    channels = 1536
    projected = (torch.randn(sum(lengths), channels, device=device) * 0.05).to(dtype)
    weight = (torch.randn(channels, 4, device=device) * 0.03).to(dtype)
    initial_pool = (torch.randn(2 * batch + 2, 3, channels, device=device) * 0.02).to(
        dtype
    )
    initial_pool[0].zero_()
    reads = torch.arange(1, batch + 1, dtype=torch.int32, device=device)
    writes = reads + batch
    initial = torch.tensor([index % 3 != 0 for index in range(batch)], device=device)
    public_pool = initial_pool.clone()
    ref_pool = initial_pool.clone()

    # Prove auto dispatch calls the packaged op, rather than a numerically
    # equivalent fallback. The explicit ref debug path must not call it.
    flash = kda_ops._load_flash_causal_conv_ops()
    assert flash is not None
    calls = []

    def counted_op(*args, **kwargs):
        calls.append(kwargs["run_mode"])
        return flash.npu_causal_conv1d(*args, **kwargs)

    monkeypatch.setattr(
        kda_ops,
        "_load_flash_causal_conv_ops",
        lambda: SimpleNamespace(npu_causal_conv1d=counted_op),
    )

    public_output = kda_causal_conv1d(
        projected,
        weight,
        public_pool,
        reads,
        writes,
        boundaries,
        cu_seqlens_cpu=boundaries_cpu,
        has_initial_state=initial,
        decode=decode,
    )
    ref_output = kda_causal_conv1d(
        projected,
        weight,
        ref_pool,
        reads,
        writes,
        boundaries,
        cu_seqlens_cpu=boundaries_cpu,
        has_initial_state=initial,
        solution="ref",
        decode=decode,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(
        public_output.float(), ref_output.float(), atol=3e-2, rtol=3e-2
    )
    assert calls == [int(decode)]
    torch.testing.assert_close(public_pool, ref_pool, atol=0, rtol=0)
    ref_pool.copy_(initial_pool)
    override_output = kda_causal_conv1d(
        projected,
        weight,
        ref_pool,
        reads,
        writes,
        boundaries,
        cu_seqlens_cpu=boundaries_cpu,
        has_initial_state=initial,
        override="ref_ascend_kda_causal_conv1d",
        decode=decode,
    )
    torch.testing.assert_close(override_output, ref_output, atol=0, rtol=0)
    torch.testing.assert_close(public_pool, ref_pool, atol=0, rtol=0)
    assert calls == [int(decode)]
    torch.testing.assert_close(
        public_pool[[0, -1]], initial_pool[[0, -1]], atol=0, rtol=0
    )
    assert torch.isfinite(public_output).all()


@pytest.mark.skipif(not _npu_available(), reason="requires an Ascend NPU")
def test_npu_decode_graph_replays_updated_input_and_pages():
    from tokenspeed_kernel.ops.attention.kda import kda_causal_conv1d

    torch.npu.set_device(0)
    device = torch.device("npu:0")
    torch.manual_seed(37)
    channels = 1536
    weight = (torch.randn(channels, 4, device=device) * 0.03).to(torch.bfloat16)
    initial_pool = (torch.randn(8, 3, channels, device=device) * 0.02).to(
        torch.bfloat16
    )
    initial_pool[0].zero_()
    graph_pool = initial_pool.clone()
    eager_pool = initial_pool.clone()
    projected = (torch.randn(2, channels, device=device) * 0.05).to(torch.bfloat16)
    reads = torch.tensor([1, 2], dtype=torch.int32, device=device)
    writes = torch.tensor([3, 4], dtype=torch.int32, device=device)
    boundaries = torch.arange(3, dtype=torch.int32, device=device)

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=torch.npu.Stream(), auto_dispatch_capture=True):
        graph_output = kda_causal_conv1d(
            projected,
            weight,
            graph_pool,
            reads,
            writes,
            boundaries,
            decode=True,
        )
    graph.replay()
    eager_output = ref_kda_causal_conv1d(
        projected=projected,
        weight=weight,
        conv_state=eager_pool,
        read_indices=reads,
        write_indices=writes,
        cu_seqlens=boundaries,
        cu_seqlens_cpu=None,
        bias=None,
        has_initial_state=None,
        activation="silu",
        decode=True,
    )
    torch.npu.synchronize()
    torch.testing.assert_close(graph_output, eager_output, atol=0, rtol=0)
    torch.testing.assert_close(graph_pool, eager_pool, atol=0, rtol=0)

    projected.copy_((torch.randn(2, channels, device=device) * 0.05).to(torch.bfloat16))
    reads.copy_(torch.tensor([3, 4], dtype=torch.int32, device=device))
    writes.copy_(torch.tensor([5, 6], dtype=torch.int32, device=device))
    eager_output = ref_kda_causal_conv1d(
        projected=projected,
        weight=weight,
        conv_state=eager_pool,
        read_indices=reads,
        write_indices=writes,
        cu_seqlens=boundaries,
        cu_seqlens_cpu=None,
        bias=None,
        has_initial_state=None,
        activation="silu",
        decode=True,
    )
    graph.replay()
    torch.npu.synchronize()

    torch.testing.assert_close(graph_output, eager_output, atol=0, rtol=0)
    torch.testing.assert_close(graph_pool, eager_pool, atol=0, rtol=0)
    torch.testing.assert_close(graph_pool[0], initial_pool[0], atol=0, rtol=0)
    torch.testing.assert_close(graph_pool[7], initial_pool[7], atol=0, rtol=0)
    assert torch.isfinite(graph_output).all()
