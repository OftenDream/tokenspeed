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

import pytest
import torch
from tokenspeed_kernel.ops.over_embedding import (
    OverEmbeddingSpec,
    TableFragmentSpec,
    append_packed_lookup_,
)


@pytest.mark.parametrize("enable_pdl", [False, True])
@pytest.mark.parametrize("fragment_count", [3, 4])
def test_fragment_lookup_with_prefix_padding_and_graph(enable_pdl, fragment_count):
    spec = OverEmbeddingSpec(
        profile="longcat-lite-tp4-test",
        tp_size=4,
        rank=0,
        vocab_size=32,
        branch_count=16,
        branch_width=256,
        hidden_size=4096,
        max_ngram_order=5,
        fragments=tuple(
            TableFragmentSpec(i, i + 2, modulus, 0, 256)
            for i, modulus in enumerate((11, 13, 17, 19)[:fragment_count])
        ),
        ignored_token_ids=(3,),
        eos_token_id=None,
        segment_ignored_tokens=True,
    )
    tables = tuple(
        torch.arange(f.modulus, device="cuda", dtype=torch.bfloat16)
        .unsqueeze(1)
        .expand(-1, 256)
        .contiguous()
        for f in spec.fragments
    )
    ids = torch.tensor([7, 3, 9, 11], device="cuda", dtype=torch.int32)
    offsets = torch.tensor([0, 3, 4], device="cuda", dtype=torch.int32)
    slots = torch.tensor([0, 1], device="cuda", dtype=torch.int64)
    active = torch.tensor([True, False], device="cuda")
    history = torch.zeros((2, 16), device="cuda", dtype=torch.int32)
    history[0, :2] = torch.tensor([4, 6], device="cuda", dtype=torch.int32)
    lengths = torch.tensor([2, 0], device="cuda", dtype=torch.int32)

    def run():
        return append_packed_lookup_(
            ids,
            offsets,
            slots,
            active,
            history,
            lengths,
            tables,
            spec=spec,
            out=None,
            solution=None,
            enable_pdl=enable_pdl,
        )

    actual = run()
    expected = torch.zeros_like(actual)
    for i, fragment in enumerate(spec.fragments):
        value = (
            sum(
                token * 32**j
                for j, token in enumerate([7, 6, 4][: fragment.ngram_order])
            )
            % fragment.modulus
        )
        expected[0, i * 256 : (i + 1) * 256] = value
        expected[2, i * 256 : (i + 1) * 256] = 9 % fragment.modulus
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert history[0, :5].tolist() == [4, 6, 7, 3, 9]
    assert (history[1] == 0).all()
    assert lengths.tolist() == [2, 0]
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        run()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()
    ids[0] = 8
    graph.replay()
    torch.testing.assert_close(captured, run(), rtol=0, atol=0)
