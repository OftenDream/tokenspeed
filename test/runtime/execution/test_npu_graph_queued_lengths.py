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
"""Queued FIA graph updates retain every replay's attention lengths."""

import pytest
import torch


def test_queued_fia_graph_updates_match_each_eager_length():
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("NPU unavailable")
    # Import the dispatcher before creating or capturing any NPU work.
    from tokenspeed_kernel_npu.ops import mla_packed as adapter
    from tokenspeed_kernel_npu.ops.mla import mla_decode_with_kvcache

    from tokenspeed.runtime.execution.forward_step import replay_graph_then_update

    if adapter._packed_op() is None:
        pytest.skip("packed FIA dispatcher unavailable")
    torch.manual_seed(1729)
    batch, heads, page = 2, 4, 128
    q = torch.randn(batch, 1, heads, 576, device="npu", dtype=torch.bfloat16) * 0.1
    cache = (
        torch.randn(batch * 3, page, 1, 576, device="npu", dtype=torch.bfloat16) * 0.1
    )
    table = torch.arange(batch * 3, device="npu", dtype=torch.int32).reshape(batch, 3)

    def attention(lengths):
        return mla_decode_with_kvcache(
            q,
            cache,
            table,
            lengths,
            3 * page,
            128,
            512,
            64,
            192**-0.5,
            0.0,
            False,
            None,
        )

    lengths = [[1, 3], [129, 127], [256, 255], [383, 382]] * 4
    expected = [attention(n).clone() for n in lengths]
    delay = torch.randn(1024, 1024, device="npu", dtype=torch.bfloat16)
    scratch = torch.empty_like(delay)
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(3):
            torch.mm(delay, delay, out=scratch)
            attention(lengths[0])
    stream.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream, auto_dispatch_capture=True):
        # Keep the graph busy before FIA consumes its updated task group.
        for _ in range(16):
            torch.mm(delay, delay, out=scratch)
        captured = attention(lengths[0])
    actual = []
    previous = None
    with torch.npu.stream(stream):
        for n in lengths:
            done = torch.npu.Event()
            replay_graph_then_update(
                graph, [{"actual_seq_lengths_kv": n}], previous, done
            )
            previous = done
            actual.append(captured.clone())
    stream.synchronize()
    for i, (got, want) in enumerate(zip(actual, expected)):
        torch.testing.assert_close(got, want, rtol=0, atol=0, msg=f"queued replay {i}")
