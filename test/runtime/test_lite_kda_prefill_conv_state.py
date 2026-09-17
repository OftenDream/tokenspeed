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

"""Checkpoint ownership across chunked Prefill and native Decode graph replay."""

from test.runtime.test_lite_hybrid_cache import _pool
from types import SimpleNamespace

import pytest
import torch

from tokenspeed.runtime.layers.attention.backends.state.kda import KdaAttnBackend
from tokenspeed.runtime.layers.attention.backends.state.mamba import (
    MambaAttnBackend,
    _prepare_cache_prefill_state_inputs,
)


def _oracle(x, weight, history):
    signal = torch.cat((history, x), dim=0)
    output = sum(
        signal[tap : tap + len(x)].float() * weight[:, tap].float() for tap in range(4)
    )
    return torch.nn.functional.silu(output).to(x.dtype), signal[-3:]


@pytest.mark.parametrize("backend_type", [MambaAttnBackend, KdaAttnBackend])
def test_single_index_leaf_stages_only_its_destination(monkeypatch, backend_type):
    from tokenspeed.runtime.layers.attention.backends.state import mamba

    state = torch.arange(5 * 4 * 3, dtype=torch.float32).view(5, 4, 3)
    before = state.clone()
    reads = torch.tensor([1, 2], dtype=torch.int32)
    writes = torch.tensor([3, 4], dtype=torch.int32)
    x = torch.zeros(2, 4)

    def conv(x, weight, bias, **kwargs):
        assert kwargs["conv_states"] is state
        assert kwargs["cache_indices"] is writes
        torch.testing.assert_close(state[writes], before[reads], atol=0, rtol=0)
        torch.testing.assert_close(state[:3], before[:3], atol=0, rtol=0)
        return x

    monkeypatch.setattr(mamba, "causal_conv1d_fn", conv)
    backend = object.__new__(backend_type)
    backend._causal_conv_prefill(
        x,
        state,
        torch.ones(4, 4),
        None,
        "silu",
        reads,
        writes,
        torch.tensor([0, 1, 2]),
        torch.tensor([True, True]),
        torch.tensor([1, 1]),
        beta_raw=torch.zeros(2, 4),
        num_heads=4,
        head_dim=128,
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "lengths", [(1, 2, 4096, 4096), (4096,) * 16], ids=["short-and-long", "64k"]
)
def test_npu_chunked_prefill_uses_checkpoint_directly(monkeypatch, dtype, lengths):
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("requires an Ascend NPU and a stride-capable flash package")
    from tokenspeed_kernel_npu.ops import kda as kda_ops

    assert kda_ops._load_flash_causal_conv_ops() is not None
    torch.npu.set_device(0)
    _, pool = _pool("npu", num_lcm_blocks=4, tp_size=8, linear_num_heads=32)
    original_state, ssm = pool.get_state_buffers(1)
    state = pool.arena.buffer.view(dtype).as_strided(
        original_state.shape, original_state.stride(), original_state.storage_offset()
    )
    assert state.shape[1:] == (3, 1536)
    assert state.stride()[1:] == (1536, 1)
    assert state.stride(0) > 3 * 1536
    assert state.storage_offset() > 0
    pool.arena.buffer.fill_(7)
    # Both the null slot and fresh destination can contain stale NaNs.
    state.fill_(float("nan"))
    ssm.fill_(float("nan"))
    expected_arena = pool.arena.buffer.cpu()
    expected_state = expected_arena.view(dtype).as_strided(
        state.shape, state.stride(), state.storage_offset()
    )
    torch.manual_seed(819)
    weight_cpu = (torch.randn(1536, 4) * 0.03).to(dtype)
    weight = weight_cpu.to("npu")
    backend = object.__new__(KdaAttnBackend)
    flash = kda_ops._load_flash_causal_conv_ops()
    assert flash is not None
    calls = []

    def counted(x, weight, conv_state, **kwargs):
        assert conv_state is state
        assert kwargs["run_mode"] == 0
        calls.append(
            (
                kwargs["cache_indices"].cpu().tolist(),
                kwargs["write_indices"].cpu().tolist(),
            )
        )
        return flash.npu_causal_conv1d(x, weight, conv_state, **kwargs)

    monkeypatch.setattr(
        kda_ops,
        "_load_flash_causal_conv_ops",
        lambda: SimpleNamespace(npu_causal_conv1d=counted),
    )
    previous = 0
    history = torch.zeros(3, 1536, dtype=dtype)
    for chunk, length in enumerate(lengths):
        destination = chunk % 2 + 1
        before = pool.arena.buffer.cpu()
        reads = torch.tensor([previous], dtype=torch.int32, device="npu")
        writes = torch.tensor([destination], dtype=torch.int32, device="npu")
        recurrent, initial, conv_reads = _prepare_cache_prefill_state_inputs(
            ssm, reads, writes
        )
        assert torch.equal(pool.arena.buffer.cpu(), before)
        if chunk == 0:
            assert torch.count_nonzero(recurrent).item() == 0
        assert initial.cpu().tolist() == [chunk != 0]
        x_cpu = (torch.randn(length, 1536) * 0.05).to(dtype)
        expected, history = _oracle(x_cpu, weight_cpu, history)
        x = x_cpu.to("npu")
        output = backend._causal_conv_prefill(
            x,
            state,
            weight,
            None,
            "silu",
            conv_reads,
            writes,
            torch.tensor([0, length], dtype=torch.int32, device="npu"),
            initial,
            torch.tensor([length], dtype=torch.int64),
            beta_raw=torch.empty(length, 512, dtype=dtype, device="npu"),
            num_heads=4,
            head_dim=128,
        )
        torch.testing.assert_close(
            output.cpu().float(), expected.float(), atol=1e-4, rtol=0.03
        )
        expected_state[destination].copy_(history)
        assert torch.equal(pool.arena.buffer.cpu(), expected_arena)
        assert calls[-1] == ([previous if previous else destination], [destination])
        previous = destination
    assert len(calls) == len(lengths)


def test_npu_bs32_decode_graph_preserves_arena_with_changed_pages():
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("requires an Ascend NPU and a stride-capable flash package")
    from tokenspeed_kernel_npu.ops import kda as kda_ops

    assert kda_ops._load_flash_causal_conv_ops() is not None
    torch.npu.set_device(0)
    _, pool = _pool("npu", num_lcm_blocks=66, tp_size=8, linear_num_heads=32)
    state, _ = pool.get_state_buffers(1)
    assert state.stride(0) > 3 * 1536 and state.storage_offset() > 0
    pool.arena.buffer.fill_(7)
    torch.manual_seed(827)
    state.copy_((torch.randn(state.shape, device="npu") * 0.02).to(state.dtype))
    expected_arena = pool.arena.buffer.cpu()
    expected_state = expected_arena.view(state.dtype).as_strided(
        state.shape, state.stride(), state.storage_offset()
    )
    weight = (torch.randn(1536, 4, device="npu") * 0.03).to(state.dtype)
    x = torch.zeros(32, 1536, device="npu", dtype=state.dtype)
    reads = torch.arange(1, 33, dtype=torch.int32, device="npu")
    writes = reads + 32
    backend = object.__new__(KdaAttnBackend)
    backend.forward_metadata = SimpleNamespace(
        query_start_loc=torch.arange(33, device="npu", dtype=torch.int32)
    )
    beta = torch.zeros(32, 512, device="npu", dtype=state.dtype)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=torch.npu.Stream(), auto_dispatch_capture=True):
        output = backend._causal_conv_decode(
            x,
            state,
            weight,
            None,
            "silu",
            reads,
            writes,
            beta_raw=beta,
            num_heads=4,
            head_dim=128,
        )
    pool.arena.buffer.copy_(expected_arena.to("npu"))
    pointer = output.data_ptr()
    for replay in range(3):
        x.copy_((torch.randn_like(x.float()) * 0.05).to(x.dtype))
        if replay:
            reads.copy_(writes.flip(0))
            writes.copy_(
                torch.arange(1, 33, device="npu", dtype=torch.int32)
                + (0 if replay == 1 else 32)
            )
        read_list, write_list = reads.cpu().tolist(), writes.cpu().tolist()
        x_cpu, weight_cpu = x.cpu(), weight.cpu()
        expected_rows, next_states = [], []
        for row, read in enumerate(read_list):
            expected, history = _oracle(
                x_cpu[row : row + 1], weight_cpu, expected_state[read]
            )
            expected_rows.append(expected)
            next_states.append(history)
        for write, history in zip(write_list, next_states):
            expected_state[write].copy_(history)
        graph.replay()
        torch.npu.synchronize()
        assert output.data_ptr() == pointer
        torch.testing.assert_close(
            output.cpu().float(), torch.cat(expected_rows).float(), atol=1e-4, rtol=0.03
        )
        assert torch.equal(pool.arena.buffer.cpu(), expected_arena)
