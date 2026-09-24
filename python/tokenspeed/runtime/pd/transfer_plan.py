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

from __future__ import annotations

import math
from dataclasses import dataclass

from tokenspeed.runtime.pd.cache_protocol import (
    CacheTransferContract,
    validate_cache_peer_layout,
)


class UnsupportedPDLayoutError(ValueError):
    pass


@dataclass(frozen=True)
class CacheTransferFragment:
    """One field-relative row fragment copied for every selected cache page.

    Arena bases, segment page-zero offsets, page bases, and page strides are
    deliberately resolved from the validated source/destination cache layouts
    at execution time. Keeping those peer-local addresses out of the route
    plan prevents the wire fragment from becoming a second, independently
    trusted cache ABI.
    """

    group_id: str
    field_id: str
    src_byte_offset: int
    dst_byte_offset: int
    src_row_stride_bytes: int
    dst_row_stride_bytes: int
    bytes_per_row: int
    rows_per_page: int


def local_transfer_pages(
    source_blocks: tuple[int, ...],
    destination_blocks: tuple[int, ...],
    *,
    group_id: str,
    source_layout: CacheTransferContract,
    destination_layout: CacheTransferContract,
    source_tp_rank: int,
    destination_tp_rank: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Resolve paired logical positions to pages owned by this transfer edge.

    Block IDs are independently allocated by P and D, not comparable between
    peers. The input order pairs the same logical positions. Rank coordinates
    are attention TP ranks, containing consecutive DCP replica subgroups.
    Returns equally sized source/destination physical-page tuples; an edge
    with no owned pairs returns empty tuples and still participates in ACKs.
    """
    source_degree = source_layout.shard_count(group_id)
    destination_degree = destination_layout.shard_count(group_id)
    source_bound = source_layout.virtual_block_count(group_id)
    destination_bound = destination_layout.virtual_block_count(group_id)
    source_pages, destination_pages = [], []
    for source, destination in zip(source_blocks, destination_blocks, strict=True):
        if not 0 < source < source_bound or not 0 < destination < destination_bound:
            raise UnsupportedPDLayoutError(
                "Cache transfer block is outside the virtual address space"
            )
        source_page, source_owner = divmod(source - 1, source_degree)
        destination_page, destination_owner = divmod(
            destination - 1, destination_degree
        )
        if (
            source_owner != source_tp_rank % source_degree
            or destination_owner != destination_tp_rank % destination_degree
        ):
            continue
        source_pages.append(source_page + 1)
        destination_pages.append(destination_page + 1)
    return tuple(source_pages), tuple(destination_pages)


MAX_CACHE_TP_SIZE = 1024


@dataclass(frozen=True)
class RankTransferPlan:
    fragments_by_prefill_rank: dict[int, tuple[CacheTransferFragment, ...]]

    @property
    def target_prefill_ranks(self) -> tuple[int, ...]:
        return tuple(self.fragments_by_prefill_rank)


@dataclass(frozen=True)
class _Interval:
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start

    def intersect(self, other: "_Interval") -> "_Interval | None":
        start = max(self.start, other.start)
        end = min(self.end, other.end)
        if start >= end:
            return None
        return _Interval(start, end)


@dataclass(frozen=True)
class _RankPartition:
    interval: _Interval
    local_offset: int


class CacheTransferPlanner:
    """Plan model-neutral dense cache fields across unequal TP sizes."""

    def __init__(
        self,
        *,
        prefill_tp_size: int,
        decode_tp_size: int,
        prefill_layout: CacheTransferContract,
        decode_layout: CacheTransferContract,
        prefill_field_ids: frozenset[str] | None,
    ):
        """Plan fragments between one Prefill rank set and one Decode rank set.

        Args:
            prefill_field_ids: Explicit resident fields to transfer, or None
                for the complete plan. Model/cache setup determines placement.
        """
        if prefill_tp_size <= 0 or decode_tp_size <= 0:
            raise UnsupportedPDLayoutError("Cache TP sizes must be positive")
        if prefill_tp_size > MAX_CACHE_TP_SIZE or decode_tp_size > MAX_CACHE_TP_SIZE:
            raise UnsupportedPDLayoutError(
                f"Cache TP sizes cannot exceed {MAX_CACHE_TP_SIZE}"
            )
        self._prefill_shards = {
            spec.group_id: spec.shard_count for spec in prefill_layout.group_specs
        }
        self._decode_shards = {
            spec.group_id: spec.shard_count for spec in decode_layout.group_specs
        }
        for layout, tp_size in (
            (prefill_layout, prefill_tp_size),
            (decode_layout, decode_tp_size),
        ):
            for spec in layout.group_specs:
                if tp_size % spec.shard_count:
                    raise UnsupportedPDLayoutError(
                        "Cache shard count must divide attention TP size"
                    )
                if spec.shard_count > 1 and any(
                    layout.transfer_schema.partition_for(field.field_id) is not None
                    for field in layout.fields_for_group(spec.group_id)
                ):
                    raise UnsupportedPDLayoutError(
                        "DCP transfer requires TP-replicated fields within sharded cache groups"
                    )
        self._has_sharded_cache = any(
            count > 1
            for count in (*self._prefill_shards.values(), *self._decode_shards.values())
        )
        self.prefill_tp_size = prefill_tp_size
        self.decode_tp_size = decode_tp_size
        all_fields = frozenset(field.field_id for field in prefill_layout.plan.fields)
        if prefill_field_ids is not None and not prefill_field_ids <= all_fields:
            raise UnsupportedPDLayoutError(
                "stage placement contains unknown cache fields"
            )
        self._field_ids = None if prefill_field_ids == all_fields else prefill_field_ids
        validate_cache_peer_layout(prefill_layout, decode_layout)

        self._partitions = {
            field.field_id: prefill_layout.transfer_schema.partition_for(field.field_id)
            for field in prefill_layout.plan.fields
        }
        self._segment_pairs = tuple(
            (prefill_spec.group_id, prefill_segment, decode_segment)
            for prefill_spec, decode_spec in zip(
                prefill_layout.group_specs,
                decode_layout.group_specs,
                strict=True,
            )
            for prefill_segment, decode_segment in zip(
                prefill_layout.fields_for_group(prefill_spec.group_id),
                decode_layout.fields_for_group(decode_spec.group_id),
                strict=True,
            )
            if prefill_field_ids is None
            or prefill_segment.field_id in prefill_field_ids
        )
        for _, prefill_segment, decode_segment in self._segment_pairs:
            self._validate_tp_mapping(prefill_segment, decode_segment)
        self._decode_ranks_by_prefill_rank = self._calc_source_decode_ranks()

    @property
    def decode_ranks_by_prefill_rank(self) -> dict[int, frozenset[int]]:
        """Decode ranks served by each Prefill rank."""
        return dict(self._decode_ranks_by_prefill_rank)

    def plan_for_decode_rank(self, decode_tp_rank: int) -> RankTransferPlan:
        if not 0 <= decode_tp_rank < self.decode_tp_size:
            raise UnsupportedPDLayoutError(
                f"decode_tp_rank={decode_tp_rank} is out of range"
            )
        # Equal-TP fast path: empty fragments mean "copy every field whole".
        # A stage owning only a subset must keep its explicit fragment route.
        if (
            self.prefill_tp_size == self.decode_tp_size
            and self._field_ids is None
            and not self._has_sharded_cache
        ):
            return RankTransferPlan(
                fragments_by_prefill_rank={decode_tp_rank: ()},
            )

        fragments_by_rank = self._fragments_for_decode_rank(decode_tp_rank)
        target_ranks = tuple(fragments_by_rank)
        if not target_ranks:
            raise UnsupportedPDLayoutError(
                f"Cache-transfer decode TP rank {decode_tp_rank} has no source fragments"
            )
        return RankTransferPlan(
            fragments_by_prefill_rank=fragments_by_rank,
        )

    def _validate_tp_mapping(self, prefill_segment, decode_segment) -> None:
        field = prefill_segment.field_id
        if self.prefill_tp_size == self.decode_tp_size and (
            prefill_segment.shape != decode_segment.shape
            or prefill_segment.payload_bytes != decode_segment.payload_bytes
        ):
            raise UnsupportedPDLayoutError(
                f"equal-TP cache field {field!r} rank-local geometry differs"
            )
        partition = self._partitions[prefill_segment.field_id]
        if partition is None:
            return
        self._rank_partitions(prefill_segment, partition, self.prefill_tp_size, 0)
        self._rank_partitions(decode_segment, partition, self.decode_tp_size, 0)

    def _fragments_for_decode_rank(
        self, decode_tp_rank: int
    ) -> dict[int, tuple[CacheTransferFragment, ...]]:
        fragments: dict[int, list[CacheTransferFragment]] = {}
        for group_id, prefill_segment, decode_segment in self._segment_pairs:
            partition = self._partitions[prefill_segment.field_id]
            if partition is None:
                prefill_rank = self._replicated_source_tp_rank(
                    self.prefill_tp_size,
                    self.decode_tp_size,
                    decode_tp_rank,
                )
                fragment = self._make_fragment(
                    group_id=group_id,
                    prefill_segment=prefill_segment,
                    decode_segment=decode_segment,
                    partition=None,
                    intersection=None,
                    prefill_interval=None,
                    decode_interval=None,
                )
                # Pick one replica subgroup, then contact every possible owner.
                # Request-specific source/destination IDs determine which pages
                # each edge actually moves, even when both DCP sizes are equal.
                degree = self._prefill_shards[group_id]
                base_rank = prefill_rank // degree * degree
                for owner in range(base_rank, base_rank + degree):
                    fragments.setdefault(owner, []).append(fragment)
                continue

            decode_partitions = self._rank_partitions(
                decode_segment, partition, self.decode_tp_size, decode_tp_rank
            )
            for prefill_rank in range(self.prefill_tp_size):
                if not self._is_representative_rank(
                    prefill_segment,
                    partition,
                    self.prefill_tp_size,
                    prefill_rank,
                ):
                    continue
                prefill_partitions = self._rank_partitions(
                    prefill_segment,
                    partition,
                    self.prefill_tp_size,
                    prefill_rank,
                )
                for prefill_partition, decode_partition in zip(
                    prefill_partitions, decode_partitions, strict=True
                ):
                    intersection = prefill_partition.interval.intersect(
                        decode_partition.interval
                    )
                    if intersection is None:
                        continue
                    fragment = self._make_fragment(
                        group_id=group_id,
                        prefill_segment=prefill_segment,
                        decode_segment=decode_segment,
                        partition=partition,
                        intersection=intersection,
                        prefill_interval=prefill_partition.interval,
                        decode_interval=decode_partition.interval,
                        prefill_local_offset=prefill_partition.local_offset,
                        decode_local_offset=decode_partition.local_offset,
                    )
                    fragments.setdefault(prefill_rank, []).append(fragment)
        return {
            rank: tuple(rank_fragments)
            for rank, rank_fragments in sorted(fragments.items())
        }

    @staticmethod
    def _make_fragment(
        *,
        group_id,
        prefill_segment,
        decode_segment,
        partition,
        intersection,
        prefill_interval,
        decode_interval,
        prefill_local_offset=0,
        decode_local_offset=0,
    ) -> CacheTransferFragment:
        if partition is None:
            rows_per_page = 1
            src_row_stride = prefill_segment.payload_bytes
            dst_row_stride = decode_segment.payload_bytes
            bytes_per_row = prefill_segment.payload_bytes
            src_byte_offset = 0
            dst_byte_offset = 0
        else:
            axis = partition.axis
            inner_bytes = (
                math.prod(prefill_segment.shape[axis + 1 :])
                * prefill_segment.element_size
            )
            rows_per_page = math.prod(prefill_segment.shape[:axis])
            src_row_stride = prefill_segment.shape[axis] * inner_bytes
            dst_row_stride = decode_segment.shape[axis] * inner_bytes
            bytes_per_row = intersection.length * inner_bytes
            src_byte_offset = (
                prefill_local_offset + intersection.start - prefill_interval.start
            ) * inner_bytes
            dst_byte_offset = (
                decode_local_offset + intersection.start - decode_interval.start
            ) * inner_bytes

        if (
            rows_per_page > 1
            and src_row_stride == bytes_per_row
            and dst_row_stride == bytes_per_row
        ):
            bytes_per_row *= rows_per_page
            src_row_stride = bytes_per_row
            dst_row_stride = bytes_per_row
            rows_per_page = 1

        return CacheTransferFragment(
            group_id=group_id,
            field_id=prefill_segment.field_id,
            src_byte_offset=src_byte_offset,
            dst_byte_offset=dst_byte_offset,
            src_row_stride_bytes=src_row_stride,
            dst_row_stride_bytes=dst_row_stride,
            bytes_per_row=bytes_per_row,
            rows_per_page=rows_per_page,
        )

    @staticmethod
    def _rank_partitions(
        segment, partition, tp_size: int, tp_rank: int
    ) -> tuple[_RankPartition, ...]:
        axis = partition.axis
        local_extent = segment.shape[axis]
        global_extent = partition.global_extent
        distinct_shards = global_extent // local_extent
        if distinct_shards > tp_size or tp_size % distinct_shards:
            raise UnsupportedPDLayoutError(
                f"Cache field {segment.field_id!r} cannot map global "
                f"extent {global_extent} and local extent {local_extent} to TP={tp_size}"
            )
        replica_group_size = tp_size // distinct_shards
        shard_rank = tp_rank // replica_group_size
        global_parts = partition.global_parts or (global_extent,)
        partitions = []
        global_offset = 0
        local_offset = 0
        for global_part_extent in global_parts:
            local_part_extent = global_part_extent // distinct_shards
            start = global_offset + shard_rank * local_part_extent
            partitions.append(
                _RankPartition(
                    interval=_Interval(start, start + local_part_extent),
                    local_offset=local_offset,
                )
            )
            global_offset += global_part_extent
            local_offset += local_part_extent
        return tuple(partitions)

    @staticmethod
    def _is_representative_rank(segment, partition, tp_size: int, tp_rank: int) -> bool:
        local_extent = segment.shape[partition.axis]
        distinct_shards = partition.global_extent // local_extent
        replica_group_size = tp_size // distinct_shards
        return tp_rank % replica_group_size == 0

    @staticmethod
    def _replicated_source_tp_rank(
        prefill_tp_size: int, decode_tp_size: int, decode_tp_rank: int
    ) -> int:
        return (decode_tp_rank * prefill_tp_size) // decode_tp_size

    def _calc_source_decode_ranks(self) -> dict[int, frozenset[int]]:
        if self.prefill_tp_size == self.decode_tp_size and not self._has_sharded_cache:
            return {rank: frozenset({rank}) for rank in range(self.prefill_tp_size)}
        decode_ranks = {rank: set() for rank in range(self.prefill_tp_size)}
        for decode_tp_rank in range(self.decode_tp_size):
            for prefill_rank in self._fragments_for_decode_rank(decode_tp_rank):
                decode_ranks[prefill_rank].add(decode_tp_rank)
        return {rank: frozenset(ranks) for rank, ranks in decode_ranks.items()}


def build_pipeline_transfer_plan(
    *,
    prefill_tp_size: int,
    decode_tp_size: int,
    decode_tp_rank: int,
    prefill_layout: CacheTransferContract,
    decode_layout: CacheTransferContract,
    cache_fields_by_stage: tuple[tuple[str, ...], ...],
) -> tuple[RankTransferPlan, tuple[int, ...]]:
    """Plan every Prefill stage's resident fields for one Decode TP rank.

    Args:
        prefill_tp_size: Attention TP width inside one Prefill stage.
        decode_tp_size: Attention TP width inside one Decode replica.
        decode_tp_rank: Receiving TP coordinate inside that replica.
        prefill_layout: Complete logical source cache contract.
        decode_layout: Complete destination cache contract.
        cache_fields_by_stage: Explicit resident field IDs for every Prefill stage.

    Returns:
        The stage-major source-rank plan and ranks joining completion with no data.
    """
    validate_cache_stage_fields(prefill_layout, cache_fields_by_stage)
    fragments: dict[int, tuple[CacheTransferFragment, ...]] = {}
    dummy_ranks: list[int] = []
    for stage, field_ids in enumerate(cache_fields_by_stage):
        planner = CacheTransferPlanner(
            prefill_tp_size=prefill_tp_size,
            decode_tp_size=decode_tp_size,
            prefill_layout=prefill_layout,
            decode_layout=decode_layout,
            prefill_field_ids=frozenset(field_ids),
        )
        stage_plan = planner.plan_for_decode_rank(decode_tp_rank)
        base = stage * prefill_tp_size
        for rank, stage_fragments in stage_plan.fragments_by_prefill_rank.items():
            fragments[base + rank] = stage_fragments
        if decode_tp_rank == 0:
            dummy_ranks.extend(
                base + rank
                for rank, decode_ranks in planner.decode_ranks_by_prefill_rank.items()
                if not decode_ranks
            )
    return RankTransferPlan(fragments_by_prefill_rank=fragments), tuple(
        sorted(dummy_ranks)
    )


def validate_cache_stage_fields(
    layout: CacheTransferContract, cache_fields_by_stage: tuple[tuple[str, ...], ...]
) -> None:
    """Require a nonempty stage list covering each logical field exactly once."""
    fields = [field for stage in cache_fields_by_stage for field in stage]
    if (
        not cache_fields_by_stage
        or any(not isinstance(field, str) for field in fields)
        or len(fields) != len(set(fields))
        or set(fields) != {field.field_id for field in layout.plan.fields}
    ):
        raise ValueError(
            "cache stage placement must cover every logical field exactly once"
        )
