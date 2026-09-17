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
"""NPU decode slot parity, including holes, padding, overflow and graph replay."""

from unittest.mock import patch

import pytest
import torch

from tokenspeed.runtime.layers.attention.backends.paged.write_locations import (
    decode_write_locations,
)


@pytest.mark.parametrize("width", [1, 3])
def test_npu_decode_slots_match_reference_without_host_scalar_reads(width):
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("NPU unavailable")
    # The live 64K window and its 128-token page boundary, plus null/overflow.
    pages = torch.tensor([128, 64], dtype=torch.int32)
    tables = torch.arange(1, 2 * 32 * 640 + 1, dtype=torch.int32).reshape(2, 32, 640)
    tables[:, 0, :3] = 0
    tables[:, 1, :3] = -1
    lengths = torch.tensor(
        [1, 2, 127, 128, 129, 65535, 65536, 65537] * 4, dtype=torch.int32
    )
    expected = torch.full((2, 32 * width), -9, dtype=torch.int32)
    decode_write_locations(tables, pages, lengths, expected, 32, width)
    inputs = [t.to("npu") for t in (tables, pages, lengths)]
    actual = torch.full_like(expected, -9, device="npu")
    original_int = torch.Tensor.__int__

    def forbid_npu_scalar(tensor):
        assert tensor.device.type != "npu", "decode metadata read an NPU scalar"
        return original_int(tensor)

    with patch.object(torch.Tensor, "__int__", forbid_npu_scalar):
        decode_write_locations(*inputs, actual, 32, width)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        decode_write_locations(*inputs, actual, 32, width)
    for delta in (1, 128, 256):
        next_lengths = lengths + delta
        decode_write_locations(tables, pages, next_lengths, expected, 32, width)
        with torch.npu.stream(stream):
            inputs[2].copy_(next_lengths)
            graph.replay()
        stream.synchronize()
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
