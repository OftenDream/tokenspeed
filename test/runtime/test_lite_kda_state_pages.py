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

"""Real-arena Prefill state publication and Decode graph handoff."""

from unittest.mock import patch

import pytest
import torch
from tokenspeed_kernel.ops.attention import kda_paged_decode, kda_paged_prefill
from tokenspeed_kernel.ops.copy.state import gather_state_rows, scatter_state_rows_

from tokenspeed.runtime.layers.attention.backends.state.mamba import (
    _prepare_cache_prefill_state_inputs,
)


def _npu_pool(pages: int):
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("requires an Ascend NPU")
    from test.runtime.test_lite_hybrid_cache import _pool

    torch.npu.set_device(0)
    return _pool(device="npu:0", num_lcm_blocks=pages, tp_size=8, linear_num_heads=32)[
        1
    ]


def test_npu_prefill_state_real_arena_fresh_crossed_pages_and_guards():
    pool = _npu_pool(8)
    conv, state = pool.get_state_buffers(0)
    assert state.shape == (9, 4, 128, 128)
    assert state.stride() == (73728, 16384, 128, 1)
    assert conv.shape == (9, 3, 1536)
    assert conv.stride() == (147456, 1536, 1)
    assert state.dtype == torch.float32
    state[0].fill_(float("nan"))
    state[2].fill_(0.25)
    state[3].fill_(float("nan"))
    conv[2].fill_(0.5)
    before = pool.arena.buffer.cpu()
    read = torch.tensor([0, 2], device="npu:0", dtype=torch.int32)
    write = torch.tensor([3, 4], device="npu:0", dtype=torch.int32)
    initial, has_initial, conv_reads = _prepare_cache_prefill_state_inputs(
        state, read, write
    )
    assert has_initial.tolist() == [False, True]
    assert conv_reads.tolist() == [3, 2]
    torch.testing.assert_close(pool.arena.buffer.cpu(), before, atol=0, rtol=0)
    torch.testing.assert_close(initial[0], torch.zeros_like(initial[0]), atol=0, rtol=0)
    torch.testing.assert_close(
        initial[1], torch.full_like(initial[1], 0.25), atol=0, rtol=0
    )
    final = initial + 0.125
    scatter_state_rows_(state, write, final)
    expected = before.clone()
    expected_state = expected.view(torch.float32).as_strided(
        state.shape, state.stride(), state.storage_offset()
    )
    expected_state[[3, 4]] = final.cpu()
    # Compares every arena byte: null page, neighboring fields/layers and page gaps.
    torch.testing.assert_close(pool.arena.buffer.cpu(), expected, atol=0, rtol=0)
    # Both sources must be captured before crossed destinations are published.
    next_read = torch.tensor([4, 3], device="npu:0", dtype=torch.int32)
    snapshot = gather_state_rows(state, next_read)
    scatter_state_rows_(state, write, snapshot)
    torch.testing.assert_close(
        gather_state_rows(state, write), final.flip(0), atol=0, rtol=0
    )


def test_npu_two_4096_chunks_then_bs32_changed_input_decode_graph():
    from tokenspeed_kernel_npu.ops import kda as npu_kda

    pool = _npu_pool(80)
    _, state = pool.get_state_buffers(0)
    assert npu_kda.is_available("chunk_kda_fwd")
    assert npu_kda._load_flash_recurrent_kda() is not None
    torch.manual_seed(97)
    # The convolution returns packed Q/K/V; splitting retains the token stride.
    qkv = torch.randn(1, 8192, 3, 4, 128, device="npu:0", dtype=torch.bfloat16)
    q, k, v = qkv.unbind(2)
    gate = torch.randn_like(q)
    beta = torch.randn_like(q)
    a = torch.linspace(-0.2, 0.2, 4, device="npu:0")
    dt = torch.randn(4, 128, device="npu:0") * 0.1
    state[3].fill_(float("nan"))
    outputs = []
    for chunk, (read_page, write_page) in enumerate(((0, 3), (3, 7))):
        read = torch.tensor([read_page], device="npu:0", dtype=torch.int32)
        write = torch.tensor([write_page], device="npu:0", dtype=torch.int32)
        initial, _, _ = _prepare_cache_prefill_state_inputs(state, read, write)
        cu_cpu = torch.tensor([0, 4096], dtype=torch.int64)
        start = chunk * 4096
        result = kda_paged_prefill(
            *(x[:, start : start + 4096] for x in (q, k, v, gate, beta)),
            a,
            dt,
            initial_state=initial,
            cu_seqlens=cu_cpu.to(device="npu:0", dtype=torch.int32),
            cu_seqlens_cpu=cu_cpu,
            lower_bound=-5.0,
            solution="public_kda",
            recurrent_layout="k_major",
        )
        scatter_state_rows_(state, write, result.final_state)
        outputs.append(result.out)
    cu_cpu = torch.tensor([0, 8192], dtype=torch.int64)
    whole = kda_paged_prefill(
        q,
        k,
        v,
        gate,
        beta,
        a,
        dt,
        initial_state=torch.zeros(1, 4, 128, 128, device="npu:0"),
        cu_seqlens=cu_cpu.to(device="npu:0", dtype=torch.int32),
        cu_seqlens_cpu=cu_cpu,
        lower_bound=-5.0,
        solution="public_kda",
        recurrent_layout="k_major",
    )
    torch.testing.assert_close(
        torch.cat(outputs, dim=1), whole.out, atol=1e-5, rtol=1e-4
    )
    torch.testing.assert_close(state[7], whole.final_state[0], atol=1e-6, rtol=1e-5)
    assert torch.isfinite(whole.out).all() and torch.isfinite(state[7]).all()
    # One of the 32 live rows resumes the actual Prefill final state.
    reads = torch.tensor([7, *range(9, 40)], device="npu:0", dtype=torch.int32)
    writes = torch.arange(41, 73, device="npu:0", dtype=torch.int32)
    boundaries = torch.arange(33, device="npu:0", dtype=torch.int32)
    inputs = [x[:, :32].clone() for x in (q, k, v, gate, beta)]
    initial_arena = pool.arena.buffer.clone()

    def decode(memory):
        return kda_paged_decode(
            *inputs,
            a,
            dt,
            state_pool=memory,
            read_indices=reads,
            write_indices=writes,
            cu_seqlens=boundaries,
            lower_bound=-5.0,
        )

    decode(state)
    pool.arena.buffer.copy_(initial_arena)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=torch.npu.Stream(), auto_dispatch_capture=True):
        actual = decode(state)
    for replay in range(2):
        pool.arena.buffer.copy_(initial_arena)
        if replay:
            for value in inputs:
                value.add_(0.125)
            reads.copy_(
                torch.tensor([7, *range(10, 41)], device="npu:0", dtype=torch.int32)
            )
            writes.copy_(torch.arange(42, 74, device="npu:0", dtype=torch.int32))
        dense = state.clone()
        with patch.object(npu_kda, "_load_flash_recurrent_kda", return_value=None):
            expected = decode(dense)
        before = pool.arena.buffer.cpu()
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-5)
        torch.testing.assert_close(
            gather_state_rows(state, writes),
            dense.index_select(0, writes),
            atol=1e-6,
            rtol=1e-6,
        )
        # Mask just published recurrent bytes, and compare all remaining storage.
        mask = torch.ones(before.numel(), dtype=torch.bool)
        for page in writes.cpu().tolist():
            begin = (state.storage_offset() + page * state.stride(0)) * 4
            mask[begin : begin + 4 * 4 * 128 * 128] = False
        torch.testing.assert_close(
            pool.arena.buffer.cpu()[mask], before[mask], atol=0, rtol=0
        )
