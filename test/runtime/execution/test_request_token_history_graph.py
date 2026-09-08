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

from types import SimpleNamespace

import torch

from tokenspeed.runtime.execution.forward_step import ForwardStepRunner


def test_decode_graph_history_layout_masks_padding_rows() -> None:
    wrapper = ForwardStepRunner.__new__(ForwardStepRunner)
    wrapper.runtime_states = SimpleNamespace(has_request_token_history=True)
    wrapper.input_buffers = SimpleNamespace(
        input_start_offsets_buf=torch.empty(4, dtype=torch.int32),
        active_request_mask_buf=torch.empty(3, dtype=torch.bool),
    )
    wrapper.max_tokens_per_req = 8
    wrapper.device = "cpu"

    wrapper._prepare_request_token_history_graph_inputs(
        active_bs=2,
        padded_bs=3,
    )

    assert wrapper.input_buffers.input_start_offsets_buf.tolist() == [0, 8, 16, 24]
    assert wrapper.input_buffers.active_request_mask_buf.tolist() == [True, True, False]
