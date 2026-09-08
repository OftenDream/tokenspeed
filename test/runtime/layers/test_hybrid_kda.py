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

import torch


def test_fgbkda_channel_beta_matches_training_order() -> None:
    from tokenspeed.runtime.layers.attention.backends.state.mamba import (
        apply_fgbkda_channel_beta,
    )

    query = torch.tensor([[[[3.0, 4.0]]]])
    key = torch.tensor([[[[5.0, 12.0]]]])
    value = torch.tensor([[[[2.0, 8.0]]]])
    logits = torch.tensor([[[[0.0, 2.0]]]])

    actual_q, actual_k, actual_v = apply_fgbkda_channel_beta(query, key, value, logits)

    beta = torch.sqrt(torch.sigmoid(logits) + 1e-10)
    torch.testing.assert_close(actual_q, torch.nn.functional.normalize(query, dim=-1))
    torch.testing.assert_close(
        actual_k, torch.nn.functional.normalize(key, dim=-1) * beta
    )
    torch.testing.assert_close(actual_v, value * beta)
