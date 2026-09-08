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

import pytest
import torch
import torch.nn.functional as F
from tokenspeed_kernel_npu.ops import kda as kda_ops
from tokenspeed_kernel_npu.ops.kda import torch_kda_causal_conv1d


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


def test_torch_prefill_matches_oracle_and_preserves_padding_page():
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
    actual_output = torch_kda_causal_conv1d(
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


def test_torch_decode_matches_oracle_with_independent_pages():
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
    actual_output = torch_kda_causal_conv1d(
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


def test_missing_public_artifact_falls_back_unless_explicit(monkeypatch):
    monkeypatch.setattr(kda_ops, "is_available", lambda _name: False)
    monkeypatch.setattr(
        kda_ops,
        "load_public_kda_ops",
        lambda: type("Status", (), {"reason": "missing test artifact"})(),
    )
    projected = torch.randn(2, 2)
    weight = torch.randn(2, 4)
    state = torch.randn(3, 2, 3)
    indices = torch.tensor([1], dtype=torch.int32)
    boundaries = torch.tensor([0, 2], dtype=torch.int64)

    expected_state = state.clone()
    expected = torch_kda_causal_conv1d(
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
    with pytest.raises(RuntimeError, match="missing test artifact"):
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
def test_npu_selection_and_large_prefill_match_torch():
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
        ("decode", "small"): "torch_ascend_kda_causal_conv1d",
        ("prefill", "small"): "torch_ascend_kda_causal_conv1d",
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
    lengths = (1, 2, 3, 4) * 4
    boundaries_cpu = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int64
    )
    boundaries = boundaries_cpu.to(device=device, dtype=torch.int32)
    channels = 1536
    projected = (torch.randn(sum(lengths), channels, device=device) * 0.05).to(
        torch.bfloat16
    )
    weight = (torch.randn(channels, 4, device=device) * 0.03).to(torch.bfloat16)
    initial_pool = (torch.randn(34, channels, 3, device=device) * 0.02).to(
        torch.bfloat16
    )
    initial_pool[0].zero_()
    reads = torch.arange(1, 17, dtype=torch.int32, device=device)
    writes = torch.arange(17, 33, dtype=torch.int32, device=device)
    initial = torch.tensor([index % 3 != 0 for index in range(16)], device=device)
    public_pool = initial_pool.clone()
    torch_pool = initial_pool.clone()

    public_output = kda_causal_conv1d(
        projected,
        weight,
        public_pool,
        reads,
        writes,
        boundaries,
        cu_seqlens_cpu=boundaries_cpu,
        has_initial_state=initial,
    )
    torch_output = kda_causal_conv1d(
        projected,
        weight,
        torch_pool,
        reads,
        writes,
        boundaries,
        cu_seqlens_cpu=boundaries_cpu,
        has_initial_state=initial,
        solution="torch",
    )
    torch.npu.synchronize()

    torch.testing.assert_close(
        public_output.float(), torch_output.float(), atol=3e-2, rtol=3e-2
    )
    torch.testing.assert_close(public_pool[17:33], torch_pool[17:33], atol=0, rtol=0)
    torch.testing.assert_close(public_pool[0], initial_pool[0], atol=0, rtol=0)
    torch.testing.assert_close(public_pool[33], initial_pool[33], atol=0, rtol=0)
    assert torch.isfinite(public_output).all()


@pytest.mark.skipif(not _npu_available(), reason="requires an Ascend NPU")
def test_npu_decode_graph_replays_updated_input_and_pages():
    from tokenspeed_kernel.ops.attention.kda import kda_causal_conv1d

    torch.npu.set_device(0)
    device = torch.device("npu:0")
    torch.manual_seed(37)
    channels = 1536
    weight = (torch.randn(channels, 4, device=device) * 0.03).to(torch.bfloat16)
    initial_pool = (torch.randn(8, channels, 3, device=device) * 0.02).to(
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
    eager_output = torch_kda_causal_conv1d(
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
    eager_output = torch_kda_causal_conv1d(
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
