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


"""Zero-expert routing must keep static shapes during graph replay."""

from types import SimpleNamespace

import pytest
import torch

from tokenspeed.runtime.models.flash_kda import FLASHLocalMoE


def _module(kind):
    return SimpleNamespace(
        zero_expert_num=4,
        zero_expert_type=kind,
        mapping=SimpleNamespace(moe=SimpleNamespace(tp_ep_size=8)),
    )


def _expected(hidden, ids, weights):
    sums = [
        sum(weight for expert, weight in zip(row_ids, row_weights) if expert < 0)
        for row_ids, row_weights in zip(ids.tolist(), weights.tolist())
    ]
    zero_weights = torch.tensor(sums, dtype=hidden.dtype).unsqueeze(1)
    return hidden * (zero_weights / 8)


@pytest.mark.parametrize("kind", ["identity", "copy", "drop", ""])
def test_zero_expert_routing_preserves_buffers_and_contribution(kind):
    hidden = torch.arange(16, dtype=torch.bfloat16).reshape(4, 4)
    ids = torch.tensor([[-1, 2], [0, 3], [-1, -1], [4, -1]], dtype=torch.int32)
    weights = torch.tensor([[0.25, 0.75], [0.5, 0.5], [0.125, 0.875], [0.75, 0.25]])
    expected = _expected(hidden, ids, weights)
    expected_ids = ids.clamp_min(0)
    expected_weights = weights * (ids >= 0)
    topk = SimpleNamespace(topk_ids=ids, topk_weights=weights)
    pointers = (ids.data_ptr(), weights.data_ptr())
    result = FLASHLocalMoE._apply_zero_experts(_module(kind), hidden, topk)
    assert (ids.data_ptr(), weights.data_ptr()) == pointers
    torch.testing.assert_close(ids, expected_ids, rtol=0, atol=0)
    torch.testing.assert_close(weights, expected_weights, rtol=0, atol=0)
    if kind in ("identity", "copy"):
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
    else:
        assert result is None


def test_npu_zero_expert_graph_replays_changed_routes():
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("requires NPU")
    hidden_cpu = torch.arange(256 * 16, dtype=torch.float32).reshape(256, 16)
    hidden_cpu = (hidden_cpu / 256).to(torch.bfloat16)
    ids_cpu = torch.arange(256 * 8, dtype=torch.int32).reshape(256, 8) % 16
    weights_cpu = torch.full((256, 8), 0.125)
    hidden = hidden_cpu.npu()
    ids = ids_cpu.npu()
    weights = weights_cpu.npu()
    topk = SimpleNamespace(topk_ids=ids, topk_weights=weights)
    module = _module("identity")
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(3):
            FLASHLocalMoE._apply_zero_experts(module, hidden, topk)
    torch.npu.current_stream().wait_stream(stream)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        output = FLASHLocalMoE._apply_zero_experts(module, hidden, topk)
    pointers = (ids.data_ptr(), weights.data_ptr())
    for pattern in (0, 1, 2):
        changed_ids = ids_cpu.clone()
        if pattern == 1:
            changed_ids[::2, 1::2] = -1
        elif pattern == 2:
            changed_ids.fill_(-1)
        changed_weights = weights_cpu * (pattern + 1)
        changed_hidden = hidden_cpu + pattern
        expected = _expected(changed_hidden, changed_ids, changed_weights)
        hidden.copy_(changed_hidden)
        ids.copy_(changed_ids)
        weights.copy_(changed_weights)
        graph.replay()
        torch.npu.synchronize()
        assert (ids.data_ptr(), weights.data_ptr()) == pointers
        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
        torch.testing.assert_close(ids.cpu(), changed_ids.clamp_min(0), rtol=0, atol=0)
        torch.testing.assert_close(
            weights.cpu(), changed_weights * (changed_ids >= 0), rtol=0, atol=0
        )
