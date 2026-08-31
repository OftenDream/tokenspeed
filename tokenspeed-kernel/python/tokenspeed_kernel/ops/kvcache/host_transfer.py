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

"""Vendor-neutral byte transfer boundary for compact Host cache storage."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Literal

import torch
from tokenspeed_kernel.ops.kvcache.triton import (
    transfer_cache_ranges as _transfer_cache_ranges_triton,
)
from tokenspeed_kernel.platform import current_platform

_mapped_host_triton_available: bool | None = None


class HostTransferWorkspace:
    """Reusable address/range tables for one Host-cache transfer stream."""

    def __init__(self) -> None:
        self._address_key: tuple[int, ...] | None = None
        self._address_table: torch.Tensor | None = None
        self._range_host: torch.Tensor | None = None
        self._range_device: torch.Tensor | None = None

    def ensure_range_host(self, num_ranges: int) -> torch.Tensor:
        if num_ranges <= 0:
            raise ValueError("num_ranges must be positive")
        if self._range_host is None or self._range_host.shape[0] < num_ranges:
            capacity = num_ranges
            if self._range_host is not None:
                capacity = max(capacity, self._range_host.shape[0] * 2)
            self._range_host = torch.empty(
                (capacity, 4), dtype=torch.int64, pin_memory=True
            )
        return self._range_host

    def bind_addresses(
        self,
        device_buffers: Sequence[torch.Tensor],
        host_buffer: torch.Tensor,
    ) -> torch.Tensor:
        device = device_buffers[0].device
        key = tuple(int(buffer.data_ptr()) for buffer in device_buffers) + (
            int(current_platform().device_visible_data_ptr(host_buffer)),
            int(device.index if device.index is not None else 0),
        )
        if self._address_table is None or self._address_key != key:
            addresses = [int(buffer.data_ptr()) for buffer in device_buffers]
            addresses.append(
                int(current_platform().device_visible_data_ptr(host_buffer))
            )
            self._address_table = torch.tensor(
                addresses, dtype=torch.uint64, device=device
            )
            self._address_key = key
        return self._address_table

    def commit_ranges(self, num_ranges: int, device: torch.device) -> torch.Tensor:
        if self._range_host is None or self._range_host.shape[0] < num_ranges:
            raise ValueError("range host table is smaller than num_ranges")
        if self._range_device is None or self._range_device.shape[0] < num_ranges:
            capacity = num_ranges
            if self._range_device is not None:
                capacity = max(capacity, self._range_device.shape[0] * 2)
            self._range_device = torch.empty(
                (capacity, 4), dtype=torch.int64, device=device
            )
        # Sync: layerwise load rewrites this pinned table as soon as the call
        # returns; async HtoD would race and corrupt later small KDA restores.
        self._range_device[:num_ranges].copy_(
            self._range_host[:num_ranges], non_blocking=False
        )
        return self._range_device

    def load_ranges(
        self,
        ranges: Sequence[tuple[int, int, int, int]],
    ) -> tuple[int, int]:
        num_ranges = len(ranges)
        if num_ranges <= 0:
            return 0, 0
        host = self.ensure_range_host(num_ranges)
        host[:num_ranges].copy_(torch.as_tensor(ranges, dtype=torch.int64))
        return num_ranges, max(int(row[3]) for row in ranges)

    def host_rows(self, num_ranges: int) -> torch.Tensor:
        if self._range_host is None:
            raise ValueError("range host table is empty")
        return self._range_host[:num_ranges]


def _triton_is_unavailable(error: Exception) -> bool:
    message = str(error).lower()
    return isinstance(error, AttributeError) or any(
        marker in message
        for marker in (
            "triton is not available",
            "hostgetdevicepointer",
            "mapped host access is not available",
            "has no attribute 'transfer_cache_ranges'",
        )
    )


def _validate_ranges(
    device_buffers: Sequence[torch.Tensor],
    host_buffer: torch.Tensor,
    ranges: Sequence[tuple[int, int, int, int]],
) -> None:
    if (
        host_buffer.device.type != "cpu"
        or host_buffer.dtype != torch.uint8
        or not host_buffer.is_contiguous()
        or not host_buffer.is_pinned()
    ):
        raise ValueError("host_buffer must be contiguous pinned CPU uint8")
    for buffer in device_buffers:
        if buffer.device.type == "cpu" or not buffer.is_contiguous():
            raise ValueError("device cache buffers must be contiguous device tensors")
    devices = {buffer.device for buffer in device_buffers}
    if len(devices) > 1:
        raise ValueError("device cache buffers must be on one device")
    for device_buffer_index, device_offset, host_offset, num_bytes in ranges:
        if not 0 <= device_buffer_index < len(device_buffers):
            raise IndexError(f"unknown device buffer {device_buffer_index}")
        if device_offset < 0 or host_offset < 0 or num_bytes <= 0:
            raise ValueError(
                "cache transfer offsets must be non-negative and non-empty"
            )
        device_bytes = (
            device_buffers[device_buffer_index].numel()
            * device_buffers[device_buffer_index].element_size()
        )
        if device_offset + num_bytes > device_bytes:
            raise IndexError("cache transfer range lies outside its device buffer")
        if host_offset + num_bytes > host_buffer.numel():
            raise IndexError("cache transfer range lies outside Host buffer")


def _transfer_dma(
    direction: Literal["d2h", "h2d"],
    device_buffers: Sequence[torch.Tensor],
    host_buffer: torch.Tensor,
    ranges: Sequence[tuple[int, int, int, int]],
) -> None:
    byte_buffers = tuple(
        buffer.view(torch.uint8).reshape(-1) for buffer in device_buffers
    )
    for device_buffer_index, device_offset, host_offset, num_bytes in ranges:
        device_slice = byte_buffers[device_buffer_index][
            device_offset : device_offset + num_bytes
        ]
        host_slice = host_buffer[host_offset : host_offset + num_bytes]
        destination, source = (
            (host_slice, device_slice)
            if direction == "d2h"
            else (device_slice, host_slice)
        )
        destination.copy_(source, non_blocking=True)


def transfer_cache_ranges(
    direction: Literal["d2h", "h2d"],
    device_buffers: Sequence[torch.Tensor],
    host_buffer: torch.Tensor,
    ranges: Sequence[tuple[int, int, int, int]],
    stream,
    *,
    backend: Literal["auto", "triton", "dma"] = "auto",
    workspace: HostTransferWorkspace | None = None,
    num_ranges: int | None = None,
    max_bytes: int | None = None,
    grid_cap: int | None = None,
) -> None:
    """Copy byte ranges between cache buffers and compact pinned Host memory.

    Args:
        direction: ``"d2h"`` for snapshot/store or ``"h2d"`` for load/recover.
        device_buffers: Device tensors referenced by range buffer indices.
        host_buffer: Contiguous pinned uint8 Host allocation.
        ranges: ``(device_buffer_index, device_offset, host_offset, num_bytes)``
            rows. Ignored when ``num_ranges`` is set on a pre-filled workspace.
        stream: Device stream that orders the asynchronous copies.
        backend: Prefer one mapped-Host Triton launch or use asynchronous DMA.
        workspace: Reused address/range tables for this stream. Required when
            ``num_ranges`` is set.
        num_ranges: Valid leading rows already written to
            ``workspace.ensure_range_host``.
        max_bytes: Largest ``num_bytes`` among those rows.
        grid_cap: Max Triton CTAs; defaults to ``TOKENSPEED_HOST_CACHE_GRID_CAP``.

    Returns:
        None. Completion is observed by recording an event on ``stream``.
    """

    if direction not in ("d2h", "h2d"):
        raise ValueError(f"unknown cache transfer direction {direction!r}")
    if backend not in ("auto", "triton", "dma"):
        raise ValueError(f"unknown cache transfer backend {backend!r}")
    if num_ranges is None:
        _validate_ranges(device_buffers, host_buffer, ranges)
        if not ranges:
            return
        prepared_ranges = ranges
        prepared_count = len(ranges)
        prepared_max_bytes = max(row[3] for row in ranges)
    else:
        if workspace is None or max_bytes is None:
            raise ValueError("pre-filled ranges require workspace and max_bytes")
        if num_ranges <= 0:
            return
        prepared_ranges = None
        prepared_count = num_ranges
        prepared_max_bytes = max_bytes

    global _mapped_host_triton_available
    with torch.cuda.stream(stream):
        if backend != "dma" and _mapped_host_triton_available is not False:
            try:
                tables = workspace or HostTransferWorkspace()
                if prepared_ranges is not None:
                    tables.load_ranges(prepared_ranges)
                address_table = tables.bind_addresses(device_buffers, host_buffer)
                range_table = tables.commit_ranges(
                    prepared_count, device_buffers[0].device
                )
                _transfer_cache_ranges_triton(
                    address_table,
                    range_table,
                    0 if direction == "d2h" else 1,
                    num_ranges=prepared_count,
                    max_bytes=prepared_max_bytes,
                    num_device_buffers=len(device_buffers),
                    grid_cap=grid_cap,
                )
                _mapped_host_triton_available = True
                return
            except (AttributeError, RuntimeError) as error:
                if backend == "triton" or not _triton_is_unavailable(error):
                    raise
                _mapped_host_triton_available = False
                warnings.warn(
                    "Mapped Host Triton transfer is unavailable; falling back to DMA",
                    RuntimeWarning,
                    stacklevel=2,
                )
        if backend == "triton":
            raise RuntimeError("mapped Host Triton transfer is unavailable")
        if prepared_ranges is None:
            prepared_ranges = [
                (int(row[0]), int(row[1]), int(row[2]), int(row[3]))
                for row in workspace.host_rows(prepared_count)
            ]
        _transfer_dma(
            direction,
            device_buffers,
            host_buffer,
            prepared_ranges,
        )
