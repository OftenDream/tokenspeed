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

"""Row-scoped access to recurrent state views in a cache arena."""

from __future__ import annotations

import torch


def _dense_row_view(tensor: torch.Tensor) -> torch.Tensor | None:
    stride = 1
    for size, actual_stride in zip(
        reversed(tensor.shape[1:]), reversed(tensor.stride()[1:])
    ):
        if size > 1 and actual_stride != stride:
            return None
        stride *= size
    return tensor.view(tensor.shape[0], stride)


def gather_state_rows(pool: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Copy selected state pages to a compact, independently owned tensor.

    Args:
        pool: State view ``[pages, ...]``; its page stride may include arena gaps.
        indices: Device integer vector of valid page IDs, one per request.
            Repeated reads are allowed. The caller supplies safe IDs for fresh rows.

    Returns:
        A contiguous ``[len(indices), ...]`` tensor with no alias to the pool.
    """
    rows = _dense_row_view(pool) if pool.device.type == "npu" else None
    if rows is None:
        return pool[indices]
    from tokenspeed_kernel.ops.copy.triton import gather_rows

    return gather_rows(rows, indices).view(indices.numel(), *pool.shape[1:])


def scatter_state_rows_(
    pool: torch.Tensor, indices: torch.Tensor, values: torch.Tensor
) -> None:
    """Publish compact final states to selected pages of the existing pool.

    Args:
        pool: State view ``[pages, ...]``; its page stride may include arena gaps.
        indices: Device integer vector of distinct valid destination page IDs.
        values: ``[len(indices), ...]`` final states, same dtype/device as pool.
            Storage must be independent of the pool so crossed reads/writes are safe.

    Returns:
        None. Only the named pool rows are changed, including when it is a view.
    """
    rows = _dense_row_view(pool) if pool.device.type == "npu" else None
    value_rows = _dense_row_view(values) if rows is not None else None
    if rows is None or value_rows is None:
        pool[indices] = values
        return
    from tokenspeed_kernel.ops.copy.triton import scatter_rows

    scatter_rows(rows, indices, value_rows)
