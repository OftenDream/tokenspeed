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
from tokenspeed_kernel_npu.ops.kda import torch_kda_paged_decode


def _npu_available() -> bool:
    if importlib.util.find_spec("torch_npu") is None:
        return False
    import torch_npu  # noqa: F401

    return torch.npu.is_available()


def _inputs(beta_mode: str, *, batch: int = 3, device: str = "cpu"):
    torch.manual_seed(47)
    heads, key_dim = 2, 4
    value_dim = key_dim if beta_mode == "featurewise" else 3
    q = torch.randn(1, batch, heads, key_dim, device=device, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(1, batch, heads, value_dim, device=device, dtype=torch.bfloat16)
    g_raw = torch.randn_like(q)
    beta_shape = q.shape if beta_mode == "featurewise" else q.shape[:-1]
    beta = torch.randn(beta_shape, device=device, dtype=torch.bfloat16)
    A_log = torch.randn(heads, device=device, dtype=torch.float32)
    dt_bias = torch.randn(heads, key_dim, device=device, dtype=torch.float32)
    state = torch.randn(
        8, heads, key_dim, value_dim, device=device, dtype=torch.float32
    )
    return q, k, v, g_raw, beta, A_log, dt_bias, state


def _lite_inputs(seed: int, *, batch: int = 2, device: str = "npu:0"):
    torch.manual_seed(seed)
    heads, dim = 4, 128
    packed = torch.randn(1, batch, heads, 5, dim, device=device, dtype=torch.bfloat16)
    q, k, v, g_raw, beta = packed.unbind(dim=-2)
    A_log = torch.linspace(-0.2, 0.2, heads, device=device)
    dt_bias = torch.randn(heads, dim, device=device) * 0.1
    state = torch.randn(7, heads, dim, dim, device=device, dtype=torch.float32) * 0.01
    assert not q.is_contiguous()
    return q, k, v, g_raw, beta, A_log, dt_bias, state


def _noncontiguous_copy(tensor: torch.Tensor) -> torch.Tensor:
    result = torch.stack((tensor, tensor), dim=-1)[..., 0]
    assert not result.is_contiguous()
    return result


def _oracle(args, reads, writes, boundaries):
    q, k, v, g_raw, beta_logits, A_log, dt_bias, state = args
    q = F.normalize(q.float(), p=2, dim=-1)
    k = F.normalize(k.float(), p=2, dim=-1)
    v = v.float()
    scalar_beta = None
    if beta_logits.shape == q.shape:
        beta_scale = torch.sqrt(torch.sigmoid(beta_logits.float()) + 1e-10)
        k = k * beta_scale
        v = v * beta_scale
    else:
        scalar_beta = torch.sigmoid(beta_logits.float())
    gate = -5.0 * torch.sigmoid(
        A_log.reshape(-1, 1).exp()
        * (g_raw.float() + dt_bias.reshape(A_log.numel(), -1))
    )
    output = torch.zeros_like(v)
    final_state = state.clone()
    updates = []
    for row, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        if start == end:
            continue
        memory = state[reads[row]].clone() * gate[0, row].exp().unsqueeze(-1)
        delta = v[0, row] - torch.matmul(k[0, row].unsqueeze(-2), memory).squeeze(-2)
        if scalar_beta is not None:
            delta = delta * scalar_beta[0, row].unsqueeze(-1)
        memory = memory + k[0, row].unsqueeze(-1) * delta.unsqueeze(-2)
        output[0, row] = torch.matmul(
            (q[0, row] * (q.shape[-1] ** -0.5)).unsqueeze(-2), memory
        ).squeeze(-2)
        updates.append((writes[row], memory))
    for write, memory in updates:
        final_state[write] = memory
    return output.to(args[2].dtype), final_state


@pytest.mark.parametrize("beta_mode", ["scalar", "featurewise"])
def test_tensor_decode_matches_reference_with_independent_pages(beta_mode: str):
    args = _inputs(beta_mode)
    reads = torch.tensor([1, 2, 3], dtype=torch.int32)
    writes = torch.tensor([4, 2, 5], dtype=torch.int32)
    boundaries = torch.arange(4, dtype=torch.int32)
    actual_state = args[-1].clone()
    expected, expected_state = _oracle(
        args, reads.tolist(), writes.tolist(), boundaries.tolist()
    )
    actual = torch_kda_paged_decode(
        *args[:-1],
        state_pool=actual_state,
        read_indices=reads,
        write_indices=writes,
        cu_seqlens=boundaries,
        lower_bound=-5.0,
    )

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(actual_state, expected_state, atol=2e-6, rtol=2e-6)


def test_tensor_decode_masks_graph_padding_and_preserves_other_pages():
    args = _inputs("featurewise", batch=4)
    reads = torch.tensor([1, 2, -1, -1], dtype=torch.int32)
    writes = torch.tensor([4, 5, -1, -1], dtype=torch.int32)
    boundaries = torch.tensor([0, 1, 2, 2, 2], dtype=torch.int32)
    initial = args[-1].clone()
    expected, expected_state = _oracle(
        args, reads.tolist(), writes.tolist(), boundaries.tolist()
    )
    actual_state = args[-1].clone()
    actual = torch_kda_paged_decode(
        *args[:-1],
        state_pool=actual_state,
        read_indices=reads,
        write_indices=writes,
        cu_seqlens=boundaries,
        lower_bound=-5.0,
    )

    torch.testing.assert_close(actual[:, :2], expected[:, :2], atol=0, rtol=0)
    torch.testing.assert_close(actual[:, 2:], torch.zeros_like(actual[:, 2:]))
    torch.testing.assert_close(actual_state, expected_state, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(actual_state[0], initial[0], atol=0, rtol=0)
    torch.testing.assert_close(actual_state[6:], initial[6:], atol=0, rtol=0)


@pytest.mark.parametrize(
    ("boundaries", "reads", "writes", "message"),
    [
        ([0, 1, 2], [1, 2], [4, 4], "distinct write"),
        ([0, 1, 2], [1, -1], [4, -1], "active rows"),
        ([0, 0, 1], [-1, 1], [-1, 4], "batch tail"),
        ([0, 2, 2], [1, -1], [4, -1], "zero/one-token"),
    ],
)
def test_tensor_decode_rejects_invalid_cpu_metadata(boundaries, reads, writes, message):
    args = _inputs("scalar", batch=2)
    with pytest.raises(ValueError, match=message):
        torch_kda_paged_decode(
            *args[:-1],
            state_pool=args[-1],
            read_indices=torch.tensor(reads, dtype=torch.int32),
            write_indices=torch.tensor(writes, dtype=torch.int32),
            cu_seqlens=torch.tensor(boundaries, dtype=torch.int32),
            lower_bound=-5.0,
        )


def test_tensor_decode_rejects_nonnegative_lower_bound():
    args = _inputs("scalar", batch=1)
    with pytest.raises(ValueError, match="finite negative"):
        torch_kda_paged_decode(
            *args[:-1],
            state_pool=args[-1],
            read_indices=torch.tensor([1], dtype=torch.int32),
            write_indices=torch.tensor([4], dtype=torch.int32),
            cu_seqlens=torch.tensor([0, 1], dtype=torch.int32),
            lower_bound=0.0,
        )


@pytest.mark.skipif(not _npu_available(), reason="requires an Ascend NPU")
def test_npu_decode_selection_and_graph_replay_updated_values():
    from tokenspeed_kernel.ops.attention.kda import (
        _attention_format_signature,
        kda_paged_decode,
    )
    from tokenspeed_kernel.selection import NoKernelFoundError, select_kernel

    torch.npu.set_device(0)
    device = "npu:0"
    args = _inputs("featurewise", batch=2, device=device)
    reads = torch.tensor([1, -1], dtype=torch.int32, device=device)
    writes = torch.tensor([4, -1], dtype=torch.int32, device=device)
    boundaries = torch.tensor([0, 1, 1], dtype=torch.int32, device=device)
    signature = _attention_format_signature(q=args[0], k=args[1], v=args[2])
    selected = select_kernel(
        "attention",
        "kda_paged_decode",
        signature,
        traits={
            "indexed_state": True,
            "single_token": True,
            "recurrent_layout": "k_major",
            "beta_mode": "featurewise",
        },
    )
    assert selected.name == "torch_ascend_kda_paged_decode"
    with pytest.raises(NoKernelFoundError):
        select_kernel(
            "attention",
            "kda_paged_decode",
            signature,
            traits={
                "indexed_state": True,
                "single_token": True,
                "recurrent_layout": "k_major",
                "beta_mode": "featurewise",
            },
            solution="public_kda",
        )

    graph_state = args[-1].clone()
    eager_state = args[-1].clone()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=torch.npu.Stream(), auto_dispatch_capture=True):
        graph_output = kda_paged_decode(
            *args[:-1],
            state_pool=graph_state,
            read_indices=reads,
            write_indices=writes,
            cu_seqlens=boundaries,
            lower_bound=-5.0,
        )
    graph.replay()
    eager_output = torch_kda_paged_decode(
        *args[:-1],
        state_pool=eager_state,
        read_indices=reads,
        write_indices=writes,
        cu_seqlens=boundaries,
        lower_bound=-5.0,
    )
    torch.npu.synchronize()
    torch.testing.assert_close(graph_output, eager_output, atol=0, rtol=0)
    torch.testing.assert_close(graph_state, eager_state, atol=0, rtol=0)

    for tensor in args[:-3]:
        tensor.copy_(torch.randn_like(tensor))
    reads.copy_(torch.tensor([4, 2], dtype=torch.int32, device=device))
    writes.copy_(torch.tensor([5, 6], dtype=torch.int32, device=device))
    boundaries.copy_(torch.arange(3, dtype=torch.int32, device=device))
    eager_output = torch_kda_paged_decode(
        *args[:-1],
        state_pool=eager_state,
        read_indices=reads,
        write_indices=writes,
        cu_seqlens=boundaries,
        lower_bound=-5.0,
    )
    graph.replay()
    torch.npu.synchronize()

    torch.testing.assert_close(graph_output, eager_output, atol=0, rtol=0)
    torch.testing.assert_close(graph_state, eager_state, atol=0, rtol=0)
    assert torch.isfinite(graph_output).all()


@pytest.mark.skipif(not _npu_available(), reason="requires an Ascend NPU")
def test_npu_lite_decode_triton_handles_strided_graph_inputs_and_padding():
    torch.npu.set_device(0)
    args = _lite_inputs(71)
    reads = torch.tensor([1, -1], dtype=torch.int32, device="npu:0")
    writes = torch.tensor([4, -1], dtype=torch.int32, device="npu:0")
    boundaries = torch.tensor([0, 1, 1], dtype=torch.int32, device="npu:0")
    initial = args[-1].clone()
    oracle_a_log = _noncontiguous_copy(args[5])

    eager_state = initial.clone()
    expected_state = initial.clone()
    actual = torch_kda_paged_decode(
        *args[:-1],
        state_pool=eager_state,
        read_indices=reads,
        write_indices=writes,
        cu_seqlens=boundaries,
        lower_bound=-5.0,
    )
    expected = torch_kda_paged_decode(
        *args[:5],
        oracle_a_log,
        args[6],
        state_pool=expected_state,
        read_indices=reads,
        write_indices=writes,
        cu_seqlens=boundaries,
        lower_bound=-5.0,
    )
    torch.npu.synchronize()
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-5)
    torch.testing.assert_close(eager_state, expected_state, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(actual[:, 1], torch.zeros_like(actual[:, 1]))
    torch.testing.assert_close(eager_state[0], initial[0], atol=0, rtol=0)
    torch.testing.assert_close(eager_state[1:4], initial[1:4], atol=0, rtol=0)
    torch.testing.assert_close(eager_state[5:], initial[5:], atol=0, rtol=0)

    graph_state = initial.clone()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=torch.npu.Stream(), auto_dispatch_capture=True):
        graph_output = torch_kda_paged_decode(
            *args[:-1],
            state_pool=graph_state,
            read_indices=reads,
            write_indices=writes,
            cu_seqlens=boundaries,
            lower_bound=-5.0,
        )
    graph_state.copy_(initial)
    graph.replay()
    torch.npu.synchronize()
    torch.testing.assert_close(graph_output, expected, atol=1e-4, rtol=1e-5)
    torch.testing.assert_close(graph_state, expected_state, atol=1e-6, rtol=1e-6)

    for tensor in args[:5]:
        tensor.copy_(torch.randn_like(tensor))
    reads.copy_(torch.tensor([1, 2], dtype=torch.int32, device="npu:0"))
    writes.copy_(torch.tensor([4, 5], dtype=torch.int32, device="npu:0"))
    boundaries.copy_(torch.arange(3, dtype=torch.int32, device="npu:0"))
    graph_state.copy_(initial)
    expected_state.copy_(initial)
    expected = torch_kda_paged_decode(
        *args[:5],
        oracle_a_log,
        args[6],
        state_pool=expected_state,
        read_indices=reads,
        write_indices=writes,
        cu_seqlens=boundaries,
        lower_bound=-5.0,
    )
    graph.replay()
    torch.npu.synchronize()
    torch.testing.assert_close(graph_output, expected, atol=1e-4, rtol=1e-5)
    torch.testing.assert_close(graph_state, expected_state, atol=1e-6, rtol=1e-6)
    assert torch.isfinite(graph_output).all()
    assert torch.isfinite(graph_state).all()


@pytest.mark.skipif(not _npu_available(), reason="requires an Ascend NPU")
@pytest.mark.parametrize("seed", [81, 82, 83, 84])
def test_npu_lite_decode_triton_128_step_trajectory(seed: int):
    torch.npu.set_device(0)
    args = _lite_inputs(seed)
    initial = args[-1].clone()
    actual_state = initial.clone()
    expected_state = initial.clone()
    reads = torch.tensor([1, 2], dtype=torch.int32, device="npu:0")
    boundaries = torch.arange(3, dtype=torch.int32, device="npu:0")
    oracle_a_log = _noncontiguous_copy(args[5])
    actual_outputs = []
    expected_outputs = []
    for _ in range(128):
        for tensor in args[:5]:
            tensor.copy_(torch.randn_like(tensor))
        expected_outputs.append(
            torch_kda_paged_decode(
                *args[:5],
                oracle_a_log,
                args[6],
                state_pool=expected_state,
                read_indices=reads,
                write_indices=reads,
                cu_seqlens=boundaries,
                lower_bound=-5.0,
            )
        )
        actual_outputs.append(
            torch_kda_paged_decode(
                *args[:-1],
                state_pool=actual_state,
                read_indices=reads,
                write_indices=reads,
                cu_seqlens=boundaries,
                lower_bound=-5.0,
            )
        )
    actual = torch.stack(actual_outputs)
    expected = torch.stack(expected_outputs)
    torch.npu.synchronize()
    difference = (actual.float() - expected.float()).flatten(1)
    denominator = expected.float().flatten(1).norm(dim=1).clamp_min(1e-12)
    assert float(difference.abs().max().cpu()) <= 1e-4
    assert float((difference.norm(dim=1) / denominator).max().cpu()) <= 1e-3
    state_difference = actual_state - expected_state
    state_denominator = expected_state.norm().clamp_min(1e-12)
    assert float(state_difference.abs().max().cpu()) <= 1e-6
    assert float((state_difference.norm() / state_denominator).cpu()) <= 1e-6
    assert torch.isfinite(actual).all()
    assert torch.isfinite(actual_state).all()
    torch.testing.assert_close(actual_state[0], initial[0], atol=0, rtol=0)
    torch.testing.assert_close(actual_state[3:], initial[3:], atol=0, rtol=0)
