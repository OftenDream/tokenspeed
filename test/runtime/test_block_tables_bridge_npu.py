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
"""Packed page uploads must survive asynchronous staging reuse on NPU."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from tokenspeed.runtime.engine.scheduler_utils import block_tables_from_forward_op


def test_npu_packed_upload_owns_pinned_staging_until_dma_finishes():
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("requires an Ascend NPU")
    stream = torch.npu.Stream()
    empty = torch.empty
    stages = []

    def allocate(*args, **kwargs):
        tensor = empty(*args, **kwargs)
        if kwargs.get("pin_memory"):
            stages.append(tensor.is_pinned())
        return tensor

    outputs = []
    with torch.npu.stream(stream), patch.object(torch, "empty", allocate):
        for step in range(32):
            # The next call releases its source immediately; host allocator
            # reuse must wait for the stream's outstanding copies.
            arrays = {
                "full": np.full((32, 2049), step + 1, dtype=np.int32),
                "state": np.full((32, 33), step + 101, dtype=np.int32),
            }
            arrays["full"][0, :2] = (-1, 0)
            for value in arrays.values():
                value.flags.writeable = False
            op = SimpleNamespace(block_tables_arrays=lambda: arrays)
            tables = block_tables_from_forward_op(op, "npu", num_reqs=32)
            assert (
                tables["full"].untyped_storage().data_ptr()
                == tables["state"].untyped_storage().data_ptr()
            )
            outputs.append(tables)
    stream.synchronize()
    assert stages == [True] * 32
    for step, tables in enumerate(outputs):
        full = tables["full"].cpu().numpy()
        assert full[0, :2].tolist() == [-1, 0]
        assert np.all(full.reshape(-1)[2:] == step + 1)
        assert np.all(tables["state"].cpu().numpy() == step + 101)
