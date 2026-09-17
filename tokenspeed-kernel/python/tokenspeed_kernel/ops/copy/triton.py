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

"""Tiled copies between strided pool rows and compact request state."""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton


@triton.jit
def _state_rows_kernel(
    pool,
    indices,
    values,
    width: tl.constexpr,
    pool_stride: tl.constexpr,
    index_stride: tl.constexpr,
    value_stride: tl.constexpr,
    SCATTER: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offset = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    # Large cache arenas can exceed the int32 element-address range.
    page = tl.load(indices + row * index_stride).to(tl.int64)
    pool_ptr = pool + page * pool_stride + offset
    value_ptr = values + row.to(tl.int64) * value_stride + offset
    if SCATTER:
        value = tl.load(value_ptr, mask=offset < width, other=0)
        tl.store(pool_ptr, value, mask=offset < width)
    else:
        value = tl.load(pool_ptr, mask=offset < width, other=0)
        tl.store(value_ptr, value, mask=offset < width)


def gather_rows(pool: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    output = torch.empty(
        (indices.numel(), pool.shape[1]), device=pool.device, dtype=pool.dtype
    )
    if output.numel():
        _state_rows_kernel[(indices.numel(), triton.cdiv(pool.shape[1], 2048))](
            pool,
            indices,
            output,
            pool.shape[1],
            pool.stride(0),
            indices.stride(0),
            output.stride(0),
            SCATTER=False,
            BLOCK=2048,
        )
    return output


def scatter_rows(
    pool: torch.Tensor, indices: torch.Tensor, values: torch.Tensor
) -> None:
    if values.numel():
        _state_rows_kernel[(indices.numel(), triton.cdiv(pool.shape[1], 2048))](
            pool,
            indices,
            values,
            pool.shape[1],
            pool.stride(0),
            indices.stride(0),
            values.stride(0),
            SCATTER=True,
            BLOCK=2048,
        )
