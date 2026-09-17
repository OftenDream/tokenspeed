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

"""Shared DCP construction over scheduler virtual cache blocks."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from tokenspeed_kernel.ops.kvcache.triton_virtual_blocks import (
    virtual_slots_to_local,
)


@dataclass(frozen=True)
class DCPPageTableMetadata:
    """One DCP rank's views of a scheduler virtual page table.

    ``virtual_page_table`` is the scheduler-owned input. ``local_page_table``
    uses the rank-local physical IDs produced by the cache arena's cyclic
    placement and marks null/foreign pages as ``-1``. ``owner_mask`` records
    the exact same translation decision for consumers that keep replicated
    cache storage but shard their compute.
    """

    virtual_page_table: torch.Tensor
    local_page_table: torch.Tensor
    owner_mask: torch.Tensor
    virtual_block_count: int
    degree: int
    rank: int

    def slice_requests(self, start: int, end: int) -> "DCPPageTableMetadata":
        """Return row views without rebuilding virtual-block placement."""

        return DCPPageTableMetadata(
            virtual_page_table=self.virtual_page_table[start:end],
            local_page_table=self.local_page_table[start:end],
            owner_mask=self.owner_mask[start:end],
            virtual_block_count=self.virtual_block_count,
            degree=self.degree,
            rank=self.rank,
        )


def refresh_dcp_page_table_metadata(
    *,
    page_table: torch.Tensor,
    virtual_block_count: int,
    degree: int,
    rank: int,
    previous: DCPPageTableMetadata | None,
) -> DCPPageTableMetadata:
    """Construct or refresh the canonical DCP page-table views.

    This is the single virtual-to-local metadata path shared by DeepSeek V4
    and DSA. Output storage is reused when its geometry matches so graph
    captures retain stable pointers.
    """
    if page_table.ndim != 2:
        raise ValueError("DCP page_table must have shape [batch, max_pages]")
    if page_table.shape[1] == 0:
        raise ValueError("DCP page_table must contain at least one column")

    local_out = None
    owner_out = None
    if previous is not None:
        if previous.degree != degree or previous.rank != rank:
            raise ValueError("DCP metadata topology changed during refresh")
        if previous.virtual_block_count != virtual_block_count:
            raise ValueError("DCP virtual block capacity changed during refresh")
        local = previous.local_page_table
        owner = previous.owner_mask
        if (
            local.shape == page_table.shape
            and local.dtype == page_table.dtype
            and local.device == page_table.device
            and local.is_contiguous()
            and owner.shape == page_table.shape
            and owner.dtype == torch.bool
            and owner.device == page_table.device
            and owner.is_contiguous()
        ):
            local_out = local
            owner_out = owner

    if local_out is None:
        # Replay setup may run outside the warmup inference context.
        with torch.inference_mode(False):
            local_out = torch.empty_like(
                page_table, memory_format=torch.contiguous_format
            )
            owner_out = torch.empty(
                page_table.shape,
                dtype=torch.bool,
                device=page_table.device,
            )

    local, owned = virtual_slots_to_local(
        page_table,
        rows_per_page=1,
        virtual_block_count=virtual_block_count,
        degree=degree,
        rank=rank,
        out=local_out,
        owner_mask=owner_out,
    )
    local.masked_fill_(~owned, -1)
    return DCPPageTableMetadata(
        virtual_page_table=page_table,
        local_page_table=local,
        owner_mask=owned,
        virtual_block_count=virtual_block_count,
        degree=degree,
        rank=rank,
    )


__all__ = [
    "DCPPageTableMetadata",
    "refresh_dcp_page_table_metadata",
]
