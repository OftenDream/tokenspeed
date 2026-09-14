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

"""Page copies must preserve strided storage, guards and crossed reads/writes."""

import pytest
import torch
from tokenspeed_kernel.ops.copy.state import gather_state_rows, scatter_state_rows_


@pytest.mark.parametrize("device", ["cpu", "npu"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("width", [2053, 65536])
def test_state_rows_preserve_gaps_and_read_snapshot(device, dtype, width):
    if device == "npu" and (not hasattr(torch, "npu") or not torch.npu.is_available()):
        pytest.skip("requires Ascend")
    storage = torch.arange(7 * (width + 256) + 32, dtype=torch.float32).to(dtype)
    expected = storage.clone()
    backing = storage.to(device)
    pool = backing.as_strided((7, 1, 1, width), (width + 256, width, width, 1), 16)
    host = expected.as_strided(pool.shape, pool.stride(), 16)
    # Crossed and repeated reads are snapshotted before any destination changes.
    indices = torch.tensor([1, 99, 3, 99, 1, 99], device=device, dtype=torch.int64)[::2]
    writes = torch.tensor([3, 1, 5], device=device, dtype=torch.int32)
    gathered = gather_state_rows(pool, indices)
    torch.testing.assert_close(gathered.cpu(), host[[1, 3, 1]], atol=0, rtol=0)
    assert gathered.is_contiguous()
    assert gathered.untyped_storage().data_ptr() != pool.untyped_storage().data_ptr()
    final = gathered + 2
    host[[3, 1, 5]] = final.cpu()
    scatter_state_rows_(pool, writes, final)
    torch.testing.assert_close(backing.cpu(), expected, atol=0, rtol=0)
    empty = indices[:0]
    assert gather_state_rows(pool, empty).shape == (0, 1, 1, width)
    scatter_state_rows_(pool, empty, final[:0])
    torch.testing.assert_close(backing.cpu(), expected, atol=0, rtol=0)


def test_state_rows_preserve_non_dense_inner_views():
    storage = torch.randn(5, 2, 3, 7)
    pool = storage.transpose(2, 3)
    expected = storage.clone().transpose(2, 3)
    ids = torch.tensor([3, 1], dtype=torch.int64)
    copied = gather_state_rows(pool, ids)
    torch.testing.assert_close(copied, expected[ids])
    values = copied + 1
    scatter_state_rows_(pool, ids, values)
    expected[ids] = values
    torch.testing.assert_close(pool, expected)
