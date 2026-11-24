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

pytest.importorskip("torch_npu")

from tokenspeed_kernel.ops.sampling.ascend import apply_penalties_logit_bias

pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="Ascend required")


def test_strided_penalty_pools_and_logits():
    vocab = 259
    base = torch.arange(3 * 2 * vocab, device="npu").reshape(3, 2 * vocab)
    counts = base.to(torch.int32)[:, ::2]
    bias = base.float()[:, 1::2]
    freq = base.float()[0, 1:6:2]
    slots = torch.tensor([2, 0], device="npu", dtype=torch.int32)
    logits = base[:2].float()[:, ::2]
    zeros = torch.zeros_like(freq)
    ones = torch.ones_like(freq)
    result = apply_penalties_logit_bias(
        logits, slots, counts, bias, freq, zeros, ones, num_tokens_per_req=1
    )
    expected = logits - counts[slots] * freq[slots, None] + bias[slots]
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    for short_counts, short_bias in ((counts[:, :-1], bias), (counts, bias[:, :-1])):
        with pytest.raises(ValueError, match="vocabulary"):
            apply_penalties_logit_bias(
                logits,
                slots,
                short_counts,
                short_bias,
                freq,
                zeros,
                ones,
                num_tokens_per_req=1,
            )


def test_logits_column_offset_above_int32():
    # Three logical values, with the last one more than 2**31 elements away.
    stride = (1 << 30) + 32
    if torch.npu.mem_get_info()[0] < (2 * stride + 1) * 4 + (1 << 30):
        pytest.skip("large-stride test requires 9 GiB free")
    storage = torch.empty(2 * stride + 1, dtype=torch.float32, device="npu")
    for col in range(3):
        storage[col * stride : col * stride + 1].fill_(col + 1)
    logits = storage.as_strided((1, 3), (1, stride))
    slot = torch.zeros(1, dtype=torch.int32, device="npu")
    counts = torch.zeros((1, 3), dtype=torch.int32, device="npu")
    bias = torch.zeros_like(counts, dtype=torch.float32)
    penalty = torch.ones(1, device="npu")
    result = apply_penalties_logit_bias(
        logits, slot, counts, bias, penalty, penalty, penalty, num_tokens_per_req=1
    )
    assert result.tolist() == [[1.0, 2.0, 3.0]]
