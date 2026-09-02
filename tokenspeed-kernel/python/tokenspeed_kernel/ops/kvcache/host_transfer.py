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
from contextlib import nullcontext
from dataclasses import dataclass, replace
from typing import Literal

import torch
from tokenspeed_kernel.ops.kvcache.triton import (
    HOST_CACHE_TRANSFER_CHUNK_BYTES,
    transfer_cache_blocks as _transfer_cache_blocks_triton,
    transfer_cache_ranges as _transfer_cache_ranges_triton,
)
from tokenspeed_kernel.platform import current_platform

_mapped_host_triton_available: bool | None = None

_GEOMETRY_ROW_WIDTH = 8
GeometryRow = tuple[int, int, int, int, int, int, int, int]
LayerSlice = tuple[int, int, int]


def _pinned_host_int64(shape: tuple[int, ...]) -> torch.Tensor:
    if torch.cuda.is_available():
        return torch.empty(shape, dtype=torch.int64, pin_memory=True)
    return torch.empty(shape, dtype=torch.int64, device="cpu")


@dataclass(frozen=True, slots=True)
class HostTransferGeometry:
    """Immutable field geometry for one executor lifetime."""

    host_rows: torch.Tensor
    device_rows: torch.Tensor | None
    layer_slices: tuple[LayerSlice, ...]
    group_packing: tuple[int, ...]
    host_lcm_block_bytes: int
    num_host_lcm_blocks: int
    num_device_lcm_blocks: int

    @property
    def num_groups(self) -> int:
        return len(self.group_packing)

    @property
    def num_field_rows(self) -> int:
        return int(self.host_rows.shape[0])

    def max_device_block_id(self, group_index: int) -> int:
        return self.num_device_lcm_blocks * self.group_packing[group_index]

    def max_host_block_id(self, group_index: int) -> int:
        return self.num_host_lcm_blocks * self.group_packing[group_index]

    def bind(
        self,
        device: torch.device,
        *,
        non_blocking: bool = False,
    ) -> HostTransferGeometry:
        if self.device_rows is not None:
            if self.device_rows.device != device:
                raise ValueError("geometry is already bound to another device")
            return self
        device_rows = torch.empty_like(self.host_rows, device=device)
        device_rows.copy_(self.host_rows, non_blocking=non_blocking)
        return replace(self, device_rows=device_rows)


def _validate_geometry_row(
    row: Sequence[int],
    *,
    row_index: int,
    group_packing: Sequence[int],
    num_device_buffers: int,
    host_lcm_block_bytes: int,
) -> None:
    if len(row) != _GEOMETRY_ROW_WIDTH:
        raise ValueError(
            f"geometry row {row_index} must have {_GEOMETRY_ROW_WIDTH} columns, "
            f"got {len(row)}"
        )
    (
        group_index,
        device_buffer_index,
        device_block_zero_offset_bytes,
        device_block_stride_bytes,
        host_cache_block_bytes,
        host_field_offset_bytes,
        row_packing,
        payload_bytes,
    ) = (int(value) for value in row)
    if not 0 <= group_index < len(group_packing):
        raise IndexError(
            f"geometry row {row_index} group {group_index} outside "
            f"[0, {len(group_packing)})"
        )
    if not 0 <= device_buffer_index < num_device_buffers:
        raise IndexError(
            f"geometry row {row_index} device_buffer_index {device_buffer_index} "
            f"outside [0, {num_device_buffers})"
        )
    if device_block_zero_offset_bytes < 0:
        raise ValueError(
            f"geometry row {row_index} device_block_zero_offset_bytes must be "
            "non-negative"
        )
    if device_block_stride_bytes <= 0:
        raise ValueError(
            f"geometry row {row_index} device_block_stride_bytes must be positive"
        )
    if host_cache_block_bytes <= 0:
        raise ValueError(
            f"geometry row {row_index} host_cache_block_bytes must be positive"
        )
    if host_field_offset_bytes < 0:
        raise ValueError(
            f"geometry row {row_index} host_field_offset_bytes must be non-negative"
        )
    expected_packing = int(group_packing[group_index])
    if row_packing != expected_packing:
        raise ValueError(
            f"geometry row {row_index} cache_blocks_per_lcm_block {row_packing} "
            f"!= group packing {expected_packing}"
        )
    if payload_bytes <= 0:
        raise ValueError(f"geometry row {row_index} payload_bytes must be positive")
    if payload_bytes > device_block_stride_bytes:
        raise ValueError(
            f"geometry row {row_index} payload_bytes cannot exceed "
            "device_block_stride_bytes"
        )
    if host_field_offset_bytes + payload_bytes > host_cache_block_bytes:
        raise ValueError(
            f"geometry row {row_index} field payload lies outside host cache block"
        )
    if row_packing * host_cache_block_bytes > host_lcm_block_bytes:
        raise ValueError(
            f"geometry row {row_index} packed group lies outside host LCM block"
        )


def build_host_transfer_geometry(
    *,
    rows: Sequence[GeometryRow],
    layer_slices: Sequence[LayerSlice],
    group_packing: Sequence[int],
    host_lcm_block_bytes: int,
    num_host_lcm_blocks: int,
    num_device_lcm_blocks: int,
    num_device_buffers: int,
) -> HostTransferGeometry:
    if not rows:
        raise ValueError("geometry must contain at least one field row")
    if not group_packing:
        raise ValueError("geometry must contain at least one cache group")
    if host_lcm_block_bytes <= 0:
        raise ValueError("host_lcm_block_bytes must be positive")
    if num_host_lcm_blocks <= 0:
        raise ValueError("num_host_lcm_blocks must be positive")
    if num_device_lcm_blocks <= 0:
        raise ValueError("num_device_lcm_blocks must be positive")
    if num_device_buffers <= 0:
        raise ValueError("num_device_buffers must be positive")
    if any(packing <= 0 for packing in group_packing):
        raise ValueError("group_packing entries must be positive")

    for row_index, row in enumerate(rows):
        _validate_geometry_row(
            row,
            row_index=row_index,
            group_packing=group_packing,
            num_device_buffers=num_device_buffers,
            host_lcm_block_bytes=host_lcm_block_bytes,
        )

    row_offset = 0
    for layer_index, (slice_offset, num_rows, max_payload_bytes) in enumerate(
        layer_slices
    ):
        if slice_offset != row_offset:
            raise ValueError(
                f"layer slice {layer_index} offset {slice_offset} != {row_offset}"
            )
        if num_rows < 0:
            raise ValueError("layer slice row counts must be non-negative")
        expected_max_payload = max(
            (int(row[7]) for row in rows[slice_offset : slice_offset + num_rows]),
            default=0,
        )
        if max_payload_bytes != expected_max_payload:
            raise ValueError(
                f"layer slice {layer_index} max payload {max_payload_bytes} "
                f"!= {expected_max_payload}"
            )
        row_offset += num_rows
    if row_offset != len(rows):
        raise ValueError("layer slices do not cover all geometry rows")

    host_rows = _pinned_host_int64((len(rows), _GEOMETRY_ROW_WIDTH))
    host_rows.copy_(torch.tensor(rows, dtype=torch.int64, device="cpu"))

    return HostTransferGeometry(
        host_rows=host_rows,
        device_rows=None,
        layer_slices=tuple(layer_slices),
        group_packing=tuple(group_packing),
        host_lcm_block_bytes=int(host_lcm_block_bytes),
        num_host_lcm_blocks=int(num_host_lcm_blocks),
        num_device_lcm_blocks=int(num_device_lcm_blocks),
    )


class HostTransferWorkspace:
    """Reusable address/range tables for one Host-cache transfer stream."""

    def __init__(self) -> None:
        self._address_key: tuple[int, ...] | None = None
        self._address_table: torch.Tensor | None = None
        self._range_host: torch.Tensor | None = None
        self._range_device: torch.Tensor | None = None
        self._num_loaded_ranges = 0
        self._num_committed_ranges = 0
        self._block_host: torch.Tensor | None = None
        self._block_device: torch.Tensor | None = None
        self._block_group_offsets_host: torch.Tensor | None = None
        self._block_group_offsets_device: torch.Tensor | None = None
        self._num_loaded_blocks = 0
        self._num_committed_blocks = 0
        self._block_group_offsets_count = 0
        self._block_load_generation = 0
        self._block_commit_generation = -1

    def _invalidate_block_commit(self) -> None:
        self._block_load_generation += 1
        self._num_committed_blocks = 0
        self._block_group_offsets_count = 0
        self._block_commit_generation = -1

    def _require_committed_block_state(self) -> None:
        if self._block_commit_generation != self._block_load_generation:
            raise ValueError("block transfers are not committed for the current load")

    def _require_loaded_block_state(self, num_blocks: int) -> None:
        if self._block_commit_generation == self._block_load_generation:
            raise ValueError("block transfers already committed for this load")
        if num_blocks != self._num_loaded_blocks:
            raise ValueError(
                "num_blocks must equal the number of rows loaded for this generation"
            )

    def ensure_range_host(self, num_ranges: int) -> torch.Tensor:
        if num_ranges <= 0:
            raise ValueError("num_ranges must be positive")
        if self._range_host is None or self._range_host.shape[0] < num_ranges:
            capacity = num_ranges
            if self._range_host is not None:
                capacity = max(capacity, self._range_host.shape[0] * 2)
            self._range_host = _pinned_host_int64((capacity, 4))
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

    def commit_ranges(
        self,
        num_ranges: int,
        device: torch.device,
        *,
        non_blocking: bool = False,
    ) -> torch.Tensor:
        if self._range_host is None or self._num_loaded_ranges < num_ranges:
            raise ValueError("range host table is smaller than num_ranges")
        if self._range_device is None or self._range_device.shape[0] < num_ranges:
            capacity = num_ranges
            if self._range_device is not None:
                capacity = max(capacity, self._range_device.shape[0] * 2)
            self._range_device = torch.empty(
                (capacity, 4), dtype=torch.int64, device=device
            )
        self._range_device[:num_ranges].copy_(
            self._range_host[:num_ranges], non_blocking=non_blocking
        )
        self._num_committed_ranges = num_ranges
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
        self._num_loaded_ranges = num_ranges
        return num_ranges, max(int(row[3]) for row in ranges)

    def load_range_batches(
        self,
        batches: Sequence[Sequence[tuple[int, int, int, int]]],
    ) -> tuple[tuple[int, int, int], ...]:
        """Load immutable range batches into one pinned table.

        Args:
            batches: Per-launch range rows. Empty launches remain represented.

        Returns:
            One ``(range_offset, num_ranges, max_bytes)`` descriptor per batch.
        """

        descriptors = []
        flat_ranges = []
        range_offset = 0
        for ranges in batches:
            num_ranges = len(ranges)
            max_bytes = max((int(row[3]) for row in ranges), default=0)
            descriptors.append((range_offset, num_ranges, max_bytes))
            flat_ranges.extend(ranges)
            range_offset += num_ranges
        if flat_ranges:
            host = self.ensure_range_host(len(flat_ranges))
            host[: len(flat_ranges)].copy_(
                torch.as_tensor(flat_ranges, dtype=torch.int64)
            )
        self._num_loaded_ranges = len(flat_ranges)
        return tuple(descriptors)

    def host_rows(self, num_ranges: int, range_offset: int = 0) -> torch.Tensor:
        if self._range_host is None:
            raise ValueError("range host table is empty")
        if range_offset < 0 or range_offset + num_ranges > self._num_loaded_ranges:
            raise ValueError("range host slice lies outside the table")
        return self._range_host[range_offset : range_offset + num_ranges]

    def device_rows(self, range_offset: int, num_ranges: int) -> torch.Tensor:
        if self._range_device is None:
            raise ValueError("range device table is empty")
        if range_offset < 0 or range_offset + num_ranges > self._num_committed_ranges:
            raise ValueError("range device slice lies outside the table")
        return self._range_device[range_offset : range_offset + num_ranges]

    def _ensure_block_host(self, num_blocks: int) -> torch.Tensor:
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        if self._block_host is None or self._block_host.shape[0] < num_blocks:
            capacity = num_blocks
            if self._block_host is not None:
                capacity = max(capacity, self._block_host.shape[0] * 2)
            self._block_host = _pinned_host_int64((capacity, 2))
        return self._block_host

    def _ensure_block_group_offsets_host(self, num_groups: int) -> torch.Tensor:
        needed = num_groups + 1
        if (
            self._block_group_offsets_host is None
            or self._block_group_offsets_host.shape[0] < needed
        ):
            capacity = needed
            if self._block_group_offsets_host is not None:
                capacity = max(capacity, self._block_group_offsets_host.shape[0] * 2)
            self._block_group_offsets_host = _pinned_host_int64((capacity,))
        return self._block_group_offsets_host

    def _validate_block_transfer(
        self,
        group_index: int,
        device_block_id: int,
        host_block_id: int,
        *,
        geometry: HostTransferGeometry,
    ) -> None:
        if not 0 <= group_index < geometry.num_groups:
            raise IndexError(f"group {group_index} outside [0, {geometry.num_groups})")
        if device_block_id <= 0 or host_block_id <= 0:
            raise ValueError(
                "device_block_id and host_block_id must be 1-based positive integers"
            )
        if device_block_id > geometry.max_device_block_id(group_index):
            raise IndexError(
                f"device_block_id {device_block_id} outside "
                f"[1, {geometry.max_device_block_id(group_index)}] "
                f"for group {group_index}"
            )
        max_host_block_id = geometry.max_host_block_id(group_index)
        if host_block_id > max_host_block_id:
            raise IndexError(
                f"host_block_id {host_block_id} outside [1, {max_host_block_id}] "
                f"for group {group_index}"
            )

    def load_block_transfers(
        self,
        transfers: Sequence[tuple[int, int, int]],
        *,
        geometry: HostTransferGeometry,
    ) -> tuple[int, tuple[int, ...]]:
        """Load dynamic block mappings bucketed stably by cache group.

        Args:
            transfers: ``(group_index, device_block_id, host_block_id)`` rows.
            geometry: Static bounds used to validate 1-based block IDs.

        Returns:
            ``(num_blocks, group_offsets)`` where ``group_offsets`` has length
            ``num_groups + 1`` and indexes the flattened host/device tables.
        """

        self._invalidate_block_commit()
        group_offset_count = geometry.num_groups + 1

        if not transfers:
            offsets = tuple(0 for _ in range(group_offset_count))
            self._num_loaded_blocks = 0
            self._block_group_offsets_count = group_offset_count
            host_offsets = self._ensure_block_group_offsets_host(geometry.num_groups)
            host_offsets[:group_offset_count].copy_(
                torch.tensor(offsets, dtype=torch.int64, device="cpu")
            )
            return 0, offsets

        buckets: list[list[tuple[int, int]]] = [[] for _ in range(geometry.num_groups)]
        for group_index, device_block_id, host_block_id in transfers:
            self._validate_block_transfer(
                int(group_index),
                int(device_block_id),
                int(host_block_id),
                geometry=geometry,
            )
            buckets[int(group_index)].append((int(device_block_id), int(host_block_id)))

        flat_rows: list[tuple[int, int]] = []
        group_offsets: list[int] = [0]
        for bucket in buckets:
            flat_rows.extend(bucket)
            group_offsets.append(len(flat_rows))

        num_blocks = len(flat_rows)
        host = self._ensure_block_host(num_blocks)
        host[:num_blocks].copy_(
            torch.tensor(flat_rows, dtype=torch.int64, device="cpu")
        )
        offsets = self._ensure_block_group_offsets_host(geometry.num_groups)
        offsets[:group_offset_count].copy_(
            torch.tensor(group_offsets, dtype=torch.int64, device="cpu")
        )
        self._num_loaded_blocks = num_blocks
        self._block_group_offsets_count = group_offset_count
        return num_blocks, tuple(group_offsets)

    def commit_block_transfers(
        self,
        num_blocks: int,
        device: torch.device,
        *,
        non_blocking: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._require_loaded_block_state(num_blocks)
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        if self._block_host is None:
            raise ValueError("block host table is empty")
        if self._block_group_offsets_host is None:
            raise ValueError("block group offsets are empty")
        if self._block_group_offsets_count <= 0:
            raise ValueError("block group offsets are empty")

        if self._block_device is None or self._block_device.shape[0] < num_blocks:
            capacity = num_blocks
            if self._block_device is not None:
                capacity = max(capacity, self._block_device.shape[0] * 2)
            self._block_device = torch.empty(
                (capacity, 2), dtype=torch.int64, device=device
            )

        offset_count = self._block_group_offsets_count
        if (
            self._block_group_offsets_device is None
            or self._block_group_offsets_device.shape[0] < offset_count
        ):
            capacity = offset_count
            if self._block_group_offsets_device is not None:
                capacity = max(capacity, self._block_group_offsets_device.shape[0] * 2)
            self._block_group_offsets_device = torch.empty(
                (capacity,), dtype=torch.int64, device=device
            )

        self._block_device[:num_blocks].copy_(
            self._block_host[:num_blocks], non_blocking=non_blocking
        )
        self._block_group_offsets_device[:offset_count].copy_(
            self._block_group_offsets_host[:offset_count],
            non_blocking=non_blocking,
        )
        self._num_committed_blocks = num_blocks
        self._block_commit_generation = self._block_load_generation
        return (
            self._block_device[:num_blocks],
            self._block_group_offsets_device[:offset_count],
        )

    def device_block_rows(self, block_offset: int, num_blocks: int) -> torch.Tensor:
        self._require_committed_block_state()
        if self._block_device is None:
            raise ValueError("block device table is empty")
        if block_offset < 0 or block_offset + num_blocks > self._num_committed_blocks:
            raise ValueError("block device slice lies outside the table")
        return self._block_device[block_offset : block_offset + num_blocks]

    def block_group_offsets_device(self) -> torch.Tensor:
        self._require_committed_block_state()
        if self._block_group_offsets_device is None:
            raise ValueError("block group offsets are empty")
        if self._block_group_offsets_count <= 0:
            raise ValueError("block group offsets are empty")
        return self._block_group_offsets_device[: self._block_group_offsets_count]

    def committed_block_tables(
        self,
        num_blocks: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one complete committed generation's Device metadata."""

        self._require_committed_block_state()
        if num_blocks != self._num_committed_blocks:
            raise ValueError(
                f"num_blocks {num_blocks} must equal committed block count "
                f"{self._num_committed_blocks}"
            )
        return (
            self.device_block_rows(0, num_blocks),
            self.block_group_offsets_device(),
        )

    def host_block_rows(self, num_blocks: int) -> torch.Tensor:
        """Return valid Host block-pair rows for the current load."""

        if num_blocks != self._num_loaded_blocks:
            raise ValueError(
                "num_blocks must equal the number of rows loaded for this generation"
            )
        if self._block_host is None:
            raise ValueError("block host table is empty")
        return self._block_host[:num_blocks]

    def block_group_offsets_host(self) -> torch.Tensor:
        """Return only valid Host group offsets for the current load."""

        if self._block_group_offsets_host is None:
            raise ValueError("block group offsets are empty")
        if self._block_group_offsets_count <= 0:
            raise ValueError("block group offsets are empty")
        return self._block_group_offsets_host[: self._block_group_offsets_count]


def _triton_is_unavailable(error: Exception) -> bool:
    message = str(error).lower()
    return isinstance(error, AttributeError) or any(
        marker in message
        for marker in (
            "triton is not available",
            "hostgetdevicepointer",
            "mapped host access is not available",
            "has no attribute 'transfer_cache_blocks'",
            "has no attribute 'transfer_cache_ranges'",
        )
    )


def _make_block_ranges(
    geometry: HostTransferGeometry,
    workspace: HostTransferWorkspace,
    *,
    num_blocks: int,
    geometry_offset: int,
    num_geometry_rows: int,
) -> tuple[tuple[int, int, int, int], ...]:
    geometry_rows = geometry.host_rows[
        geometry_offset : geometry_offset + num_geometry_rows
    ].tolist()
    block_rows = workspace.host_block_rows(num_blocks)
    group_offsets = workspace.block_group_offsets_host()
    group_block_rows = {}
    ranges = []
    for row in geometry_rows:
        (
            group_index,
            device_buffer_index,
            device_zero,
            device_stride,
            host_block_bytes,
            host_field_offset,
            packing,
            payload_bytes,
        ) = (int(value) for value in row)
        if group_index not in group_block_rows:
            group_start = int(group_offsets[group_index].item())
            group_end = int(group_offsets[group_index + 1].item())
            group_block_rows[group_index] = block_rows[group_start:group_end].tolist()
        for device_block_id, host_block_id in group_block_rows[group_index]:
            device_block_id = int(device_block_id)
            host_block_id = int(host_block_id)
            host_zero_based = host_block_id - 1
            host_parent, host_child = divmod(host_zero_based, packing)
            ranges.append(
                (
                    device_buffer_index,
                    device_zero + device_block_id * device_stride,
                    host_parent * geometry.host_lcm_block_bytes
                    + host_child * host_block_bytes
                    + host_field_offset,
                    payload_bytes,
                )
            )
    return tuple(ranges)


def _block_work_items(
    geometry: HostTransferGeometry,
    workspace: HostTransferWorkspace,
    *,
    geometry_offset: int,
    num_geometry_rows: int,
) -> int:
    """Return the largest real block/chunk count in a geometry slice."""

    geometry_rows = geometry.host_rows[
        geometry_offset : geometry_offset + num_geometry_rows
    ].tolist()
    group_offsets = workspace.block_group_offsets_host().tolist()
    work_items = 0
    for row in geometry_rows:
        group_index = int(row[0])
        payload_bytes = int(row[7])
        group_blocks = int(group_offsets[group_index + 1] - group_offsets[group_index])
        num_chunks = (
            payload_bytes + HOST_CACHE_TRANSFER_CHUNK_BYTES - 1
        ) // HOST_CACHE_TRANSFER_CHUNK_BYTES
        work_items = max(work_items, group_blocks * num_chunks)
    return work_items


def transfer_cache_blocks(
    direction: Literal["d2h", "h2d"],
    device_buffers: Sequence[torch.Tensor],
    host_buffer: torch.Tensor,
    geometry: HostTransferGeometry,
    workspace: HostTransferWorkspace,
    stream,
    *,
    num_blocks: int,
    geometry_offset: int,
    num_geometry_rows: int,
    max_payload_bytes: int,
    backend: Literal["auto", "triton", "dma"] = "auto",
    grid_cap: int | None = None,
) -> None:
    """Copy compact Host blocks using static geometry and dynamic block IDs.

    Args:
        direction: ``"d2h"`` for writeback or ``"h2d"`` for restore.
        device_buffers: Device tensors referenced by static geometry rows.
        host_buffer: Contiguous pinned uint8 compact Host allocation.
        geometry: Immutable transfer geometry, Device-bound when using Triton.
        workspace: Current dynamic block mapping; Triton requires it committed.
        stream: Device stream ordering the asynchronous copies.
        num_blocks: Number of valid dynamic block-pair rows.
        geometry_offset: First static field row for this layer.
        num_geometry_rows: Number of static field rows for this layer.
        max_payload_bytes: Largest field payload in this geometry slice.
        backend: Prefer mapped-Host Triton or lazily expand ranges for DMA.
        grid_cap: Max Triton CTAs.

    Returns:
        None. Completion is observed by recording an event on ``stream``.
    """

    if direction not in ("d2h", "h2d"):
        raise ValueError(f"unknown cache transfer direction {direction!r}")
    if backend not in ("auto", "triton", "dma"):
        raise ValueError(f"unknown cache transfer backend {backend!r}")
    if geometry_offset < 0 or num_geometry_rows < 0 or max_payload_bytes < 0:
        raise ValueError("geometry slice values must be non-negative")
    if geometry_offset + num_geometry_rows > geometry.num_field_rows:
        raise ValueError("geometry slice lies outside the static table")
    if num_blocks < 0:
        raise ValueError("num_blocks must be non-negative")
    if num_geometry_rows > 0 and max_payload_bytes <= 0:
        raise ValueError("max_payload_bytes must be positive for a non-empty slice")
    if num_blocks == 0 or num_geometry_rows == 0:
        return

    def transfer_dma() -> None:
        ranges = _make_block_ranges(
            geometry,
            workspace,
            num_blocks=num_blocks,
            geometry_offset=geometry_offset,
            num_geometry_rows=num_geometry_rows,
        )
        transfer_cache_ranges(
            direction,
            device_buffers,
            host_buffer,
            ranges,
            stream,
            backend="dma",
            grid_cap=grid_cap,
        )

    if backend == "dma":
        transfer_dma()
        return

    global _mapped_host_triton_available
    mapped_host_candidate = device_buffers[0].device.type != "npu"
    if mapped_host_candidate and _mapped_host_triton_available is not False:
        if geometry.device_rows is None:
            raise ValueError("geometry must be bound before a Triton block transfer")
        device_module = torch.get_device_module(device_buffers[0].device)
        stream_context = (
            device_module.stream(stream) if stream is not None else nullcontext()
        )
        try:
            with stream_context:
                block_rows, group_offsets = workspace.committed_block_tables(num_blocks)
                work_items = _block_work_items(
                    geometry,
                    workspace,
                    geometry_offset=geometry_offset,
                    num_geometry_rows=num_geometry_rows,
                )
                if work_items == 0:
                    return
                address_table = workspace.bind_addresses(device_buffers, host_buffer)
                _transfer_cache_blocks_triton(
                    address_table,
                    geometry.device_rows,
                    block_rows,
                    group_offsets,
                    0 if direction == "d2h" else 1,
                    geometry_offset=geometry_offset,
                    num_geometry_rows=num_geometry_rows,
                    host_lcm_block_bytes=geometry.host_lcm_block_bytes,
                    work_items=work_items,
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
                "Mapped Host Triton block transfer is unavailable; falling back to DMA",
                RuntimeWarning,
                stacklevel=2,
            )
    if backend == "triton":
        raise RuntimeError("mapped Host Triton transfer is unavailable")
    transfer_dma()


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
    range_offset: int = 0,
    ranges_committed: bool = False,
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
        range_offset: First row of an immutable, pre-filled workspace batch.
        ranges_committed: Read the Device table slice without uploading metadata.

    Returns:
        None. Completion is observed by recording an event on ``stream``.
    """

    if direction not in ("d2h", "h2d"):
        raise ValueError(f"unknown cache transfer direction {direction!r}")
    if backend not in ("auto", "triton", "dma"):
        raise ValueError(f"unknown cache transfer backend {backend!r}")
    if range_offset < 0:
        raise ValueError("range_offset must be non-negative")
    if ranges_committed and (workspace is None or num_ranges is None):
        raise ValueError("committed ranges require workspace and num_ranges")
    if num_ranges is None:
        if range_offset != 0:
            raise ValueError("range_offset requires pre-filled ranges")
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
    device_module = torch.get_device_module(device_buffers[0].device)
    with device_module.stream(stream):
        mapped_host_candidate = device_buffers[0].device.type != "npu"
        if (
            backend != "dma"
            and mapped_host_candidate
            and _mapped_host_triton_available is not False
        ):
            try:
                tables = workspace or HostTransferWorkspace()
                if prepared_ranges is not None:
                    tables.load_ranges(prepared_ranges)
                address_table = tables.bind_addresses(device_buffers, host_buffer)
                range_table = (
                    tables.device_rows(range_offset, prepared_count)
                    if ranges_committed
                    else tables.commit_ranges(prepared_count, device_buffers[0].device)
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
                for row in workspace.host_rows(prepared_count, range_offset)
            ]
        _transfer_dma(
            direction,
            device_buffers,
            host_buffer,
            prepared_ranges,
        )
