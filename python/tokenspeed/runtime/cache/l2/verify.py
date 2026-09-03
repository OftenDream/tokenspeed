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

"""Hash Host L2 pages at the store/load boundaries.

The durable baseline is the Host page **after D2H finishes**, not the Device
snapshot taken when writeback is submitted. That earlier Device hash is only
used to classify a torn store (Device still zero, or Device != Host).

Steps:

1. ``snapshot_store_device`` — Device after waiting on the producer stream
2. ``commit_store_host`` — Host after D2H; this becomes the baseline
3. ``snapshot_load_host`` — Host before H2D; must equal the baseline
4. ``check_load_device`` (``LOAD device post-h2d``) — Device right after H2D
   completes and before the model forward can mutate pages; this is the
   authoritative H2D mapping check
5. ``check_load_device`` (``LOAD device at-poll``) — Device when the load ack
   is drained; may diverge for KDA when the same checkpoint page is updated
   in-place during a non-crossing extend that overlaps the poll

Enable with ``TOKENSPEED_L2_VERIFY=1``. ``TOKENSPEED_L2_VERIFY_LOG`` writes
one json object per block per step to ``<path>.rank<N>``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field

import torch

from tokenspeed.runtime.utils import get_colorful_logger
from tokenspeed.runtime.utils.env import envs

logger = get_colorful_logger(__name__)

_DIGEST_BYTES = 16


def l2_verify_enabled() -> bool:
    """Return whether L2 store/load byte verification is on."""

    return bool(envs.TOKENSPEED_L2_VERIFY.get())


def l2_verify_log_path() -> str:
    """Return the jsonl prefix for per-block verify records, or empty."""

    return str(envs.TOKENSPEED_L2_VERIFY_LOG.get() or "")


def payload_is_zero(payload: bytes) -> bool:
    """Return whether every byte in ``payload`` is zero."""

    return not any(payload)


def _digest(payload: bytes) -> str:
    return hashlib.blake2b(payload, digest_size=_DIGEST_BYTES).hexdigest()


def _tensor_bytes(tensor, offset: int, nbytes: int) -> bytes:
    flat = tensor.detach().view(torch.uint8).reshape(-1)
    view = flat[offset : offset + nbytes]
    if view.numel() != nbytes:
        raise IndexError(
            f"cache slice [{offset}:{offset + nbytes}] is outside buffer "
            f"of {flat.numel()} bytes"
        )
    if view.is_cuda:
        view = view.cpu()
    return bytes(view.contiguous().numpy())


def _group_id(layout, group_index: int) -> str:
    return layout.groups[group_index].group_id


def _append_jsonl(path: str | None, record: dict) -> None:
    if not path:
        return
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


@dataclass
class _FieldDigest:
    digest: str
    payload: bytes
    zero: bool


@dataclass
class _BlockSnap:
    digest: str
    fields: dict[str, _FieldDigest]
    zero_fields: int
    device_block: int


@dataclass
class L2VerifyStats:
    stores: int = 0
    store_mismatches: int = 0
    store_device_zero_blocks: int = 0
    store_host_zero_blocks: int = 0
    loads: int = 0
    load_host_mismatches: int = 0
    load_device_mismatches: int = 0
    load_device_post_h2d_mismatches: int = 0
    load_device_at_poll_mismatches: int = 0
    loads_without_baseline: int = 0


@dataclass
class L2TransferVerifier:
    """Compare store/load copies against the Host page that D2H persisted."""

    layout: object
    host_storage: object
    attn_tp_rank: int = 0
    _device_at_store: dict[tuple[int, int], _BlockSnap] = field(default_factory=dict)
    _baselines: dict[tuple[int, int], _BlockSnap] = field(default_factory=dict)
    _store_seq: int = 0
    stats: L2VerifyStats = field(default_factory=L2VerifyStats)
    _jsonl_path: str | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        prefix = l2_verify_log_path()
        self._jsonl_path = f"{prefix}.rank{self.attn_tp_rank}" if prefix else None

    def snapshot_store_device(
        self, transfers: Sequence[tuple[int, int, int]]
    ) -> None:
        """Hash Device pages after the producer stream has been waited on."""

        zero_fields = 0
        zero_blocks = 0
        for group_index, device_block, host_block in transfers:
            snap = self._hash_device_fields(group_index, device_block)
            self._device_at_store[(group_index, host_block)] = snap
            zero_fields += snap.zero_fields
            if snap.fields and snap.zero_fields == len(snap.fields):
                zero_blocks += 1
        if self.attn_tp_rank == 0 or zero_fields:
            logger.info(
                "[L2 verify] STORE device snapshot rank=%s blocks=%s "
                "zero_fields=%s zero_blocks=%s",
                self.attn_tp_rank,
                len(transfers),
                zero_fields,
                zero_blocks,
            )

    def commit_store_host(self, transfers: Sequence[tuple[int, int, int]]) -> int:
        """Hash Host after D2H and keep that page as the load baseline."""

        self.stats.stores += len(transfers)
        mismatches = 0
        mismatch_fields = 0
        device_zero_blocks = 0
        host_zero_blocks = 0
        for group_index, device_block, host_block in transfers:
            self._store_seq += 1
            host_snap = self._hash_host_fields(group_index, host_block)
            host_snap.device_block = device_block
            self._baselines[(group_index, host_block)] = host_snap
            if host_snap.zero_fields == len(host_snap.fields):
                host_zero_blocks += 1
                self.stats.store_host_zero_blocks += 1
            device_snap = self._device_at_store.get((group_index, host_block))
            device_zero = 0
            if device_snap is not None and device_snap.zero_fields == len(
                device_snap.fields
            ):
                device_zero = 1
                device_zero_blocks += 1
                self.stats.store_device_zero_blocks += 1
            diffs = []
            if device_snap is not None:
                diffs = self._compare_fields(
                    stage="STORE device-vs-host",
                    group_index=group_index,
                    host_block=host_block,
                    device_block=device_block,
                    expected=device_snap.fields,
                    actual=host_snap.fields,
                )
            if diffs:
                mismatches += 1
                mismatch_fields += len(diffs)
                self.stats.store_mismatches += 1
            self._write_record(
                {
                    "stage": "STORE",
                    "rank": self.attn_tp_rank,
                    "group": group_index,
                    "group_id": _group_id(self.layout, group_index),
                    "host": host_block,
                    "device": device_block,
                    "store_seq": self._store_seq,
                    "device_digest": device_snap.digest if device_snap else None,
                    "host_digest": host_snap.digest,
                    "device_zero_fields": (
                        device_snap.zero_fields if device_snap else None
                    ),
                    "host_zero_fields": host_snap.zero_fields,
                    "device_all_zero": bool(device_zero),
                    "host_all_zero": host_snap.zero_fields == len(host_snap.fields),
                    "mismatch_fields": len(diffs),
                    "fields": diffs,
                }
            )
        logger.info(
            "[L2 verify] STORE done rank=%s blocks=%s device_vs_host=%s "
            "mismatch_fields=%s device_zero_blocks=%s host_zero_blocks=%s "
            "totals_blocks=%s totals_mismatch_blocks=%s",
            self.attn_tp_rank,
            len(transfers),
            mismatches,
            mismatch_fields,
            device_zero_blocks,
            host_zero_blocks,
            self.stats.stores,
            self.stats.store_mismatches,
        )
        return mismatches

    def snapshot_load_host(self, transfers: Sequence[tuple[int, int, int]]) -> int:
        """Hash Host before H2D and compare it to the persisted baseline."""

        mismatches = 0
        mismatch_fields = 0
        missing = 0
        for group_index, device_block, host_block in transfers:
            baseline = self._baselines.get((group_index, host_block))
            host_snap = self._hash_host_fields(group_index, host_block)
            if baseline is None:
                missing += 1
                self.stats.loads_without_baseline += 1
                self._write_record(
                    {
                        "stage": "LOAD host",
                        "rank": self.attn_tp_rank,
                        "group": group_index,
                        "group_id": _group_id(self.layout, group_index),
                        "host": host_block,
                        "device": device_block,
                        "baseline": None,
                        "host_digest": host_snap.digest,
                        "no_baseline": True,
                    }
                )
                continue
            diffs = self._compare_fields(
                stage="LOAD host-vs-baseline",
                group_index=group_index,
                host_block=host_block,
                device_block=device_block,
                expected=baseline.fields,
                actual=host_snap.fields,
            )
            if diffs:
                mismatches += 1
                mismatch_fields += len(diffs)
                self.stats.load_host_mismatches += 1
            self._write_record(
                {
                    "stage": "LOAD host",
                    "rank": self.attn_tp_rank,
                    "group": group_index,
                    "group_id": _group_id(self.layout, group_index),
                    "host": host_block,
                    "device": device_block,
                    "baseline": baseline.digest,
                    "host_digest": host_snap.digest,
                    "host_all_zero": host_snap.zero_fields == len(host_snap.fields),
                    "baseline_all_zero": baseline.zero_fields == len(baseline.fields),
                    "mismatch_fields": len(diffs),
                    "fields": diffs,
                }
            )
        logger.info(
            "[L2 verify] LOAD host rank=%s blocks=%s host_vs_baseline=%s "
            "mismatch_fields=%s no_baseline=%s",
            self.attn_tp_rank,
            len(transfers),
            mismatches,
            mismatch_fields,
            missing,
        )
        return mismatches

    def check_load_device(
        self,
        transfers: Sequence[tuple[int, int, int]],
        *,
        stage: str = "LOAD device post-h2d",
    ) -> int:
        """Hash Device against the Host baseline at one load boundary.

        Args:
            transfers: ``(group_index, device_block, host_block)`` rows.
            stage: Jsonl/log label. Use ``LOAD device post-h2d`` immediately
                after H2D (authoritative mapping check) and
                ``LOAD device at-poll`` when the load ack is drained (may see
                in-place KDA updates from an overlapping forward).

        Returns:
            Number of blocks whose Device digest disagrees with the baseline.
        """

        if stage == "LOAD device post-h2d":
            self.stats.loads += len(transfers)
        mismatches = 0
        mismatch_fields = 0
        missing = 0
        compare_stage = f"{stage}-vs-baseline"
        for group_index, device_block, host_block in transfers:
            baseline = self._baselines.get((group_index, host_block))
            device_snap = self._hash_device_fields(group_index, device_block)
            if baseline is None:
                missing += 1
                self._write_record(
                    {
                        "stage": stage,
                        "rank": self.attn_tp_rank,
                        "group": group_index,
                        "group_id": _group_id(self.layout, group_index),
                        "host": host_block,
                        "device": device_block,
                        "baseline": None,
                        "device_digest": device_snap.digest,
                        "no_baseline": True,
                    }
                )
                continue
            diffs = self._compare_fields(
                stage=compare_stage,
                group_index=group_index,
                host_block=host_block,
                device_block=device_block,
                expected=baseline.fields,
                actual=device_snap.fields,
            )
            if diffs:
                mismatches += 1
                mismatch_fields += len(diffs)
                self.stats.load_device_mismatches += 1
                if stage == "LOAD device post-h2d":
                    self.stats.load_device_post_h2d_mismatches += 1
                elif stage == "LOAD device at-poll":
                    self.stats.load_device_at_poll_mismatches += 1
            self._write_record(
                {
                    "stage": stage,
                    "rank": self.attn_tp_rank,
                    "group": group_index,
                    "group_id": _group_id(self.layout, group_index),
                    "host": host_block,
                    "device": device_block,
                    "baseline": baseline.digest,
                    "device_digest": device_snap.digest,
                    "device_all_zero": device_snap.zero_fields == len(device_snap.fields),
                    "baseline_all_zero": baseline.zero_fields == len(baseline.fields),
                    "mismatch_fields": len(diffs),
                    "fields": diffs,
                }
            )
        logger.info(
            "[L2 verify] %s rank=%s blocks=%s device_vs_baseline=%s "
            "mismatch_fields=%s no_baseline=%s totals_loads=%s "
            "post_h2d_mismatch_blocks=%s at_poll_mismatch_blocks=%s",
            stage,
            self.attn_tp_rank,
            len(transfers),
            mismatches,
            mismatch_fields,
            missing,
            self.stats.loads,
            self.stats.load_device_post_h2d_mismatches,
            self.stats.load_device_at_poll_mismatches,
        )
        return mismatches

    def reset(self) -> None:
        self._device_at_store.clear()
        self._baselines.clear()
        self._store_seq = 0
        self.stats = L2VerifyStats()

    def _write_record(self, record: dict) -> None:
        try:
            _append_jsonl(self._jsonl_path, record)
        except OSError:
            logger.exception("[L2 verify] failed to append jsonl %s", self._jsonl_path)

    def _hash_device_fields(self, group_index: int, device_block: int) -> _BlockSnap:
        group = self.layout.groups[group_index]
        fields: dict[str, _FieldDigest] = {}
        for cache_field in group.fields:
            offset = (
                cache_field.device_block_zero_offset_bytes
                + device_block * cache_field.block_stride_bytes
            )
            payload = _tensor_bytes(
                self.layout.buffers[cache_field.device_buffer_index],
                offset,
                cache_field.payload_bytes,
            )
            fields[cache_field.field_id] = _FieldDigest(
                _digest(payload), payload, payload_is_zero(payload)
            )
        digest = _digest(b"".join(item.payload for item in fields.values()))
        return _BlockSnap(
            digest,
            fields,
            sum(1 for item in fields.values() if item.zero),
            device_block,
        )

    def _hash_host_fields(self, group_index: int, host_block: int) -> _BlockSnap:
        group = self.layout.groups[group_index]
        fields: dict[str, _FieldDigest] = {}
        for field_index, cache_field in enumerate(group.fields):
            offset = self.host_storage.host_field_offset(
                group_index, host_block, field_index
            )
            payload = _tensor_bytes(
                self.host_storage.host_buffer, offset, cache_field.payload_bytes
            )
            fields[cache_field.field_id] = _FieldDigest(
                _digest(payload), payload, payload_is_zero(payload)
            )
        digest = _digest(b"".join(item.payload for item in fields.values()))
        return _BlockSnap(
            digest,
            fields,
            sum(1 for item in fields.values() if item.zero),
            0,
        )

    def _compare_fields(
        self,
        *,
        stage: str,
        group_index: int,
        host_block: int,
        device_block: int,
        expected: dict[str, _FieldDigest],
        actual: dict[str, _FieldDigest],
    ) -> list[dict]:
        diffs: list[dict] = []
        group_id = _group_id(self.layout, group_index)
        for field_id, expected_field in expected.items():
            actual_field = actual.get(field_id)
            if actual_field is None:
                diffs.append(
                    {
                        "field": field_id,
                        "missing": True,
                        "expected_zero": expected_field.zero,
                    }
                )
                logger.error(
                    "[L2 verify] %s MISSING field rank=%s group_id=%s host=%s "
                    "device=%s field=%s",
                    stage,
                    self.attn_tp_rank,
                    group_id,
                    host_block,
                    device_block,
                    field_id,
                )
                continue
            if actual_field.digest == expected_field.digest:
                continue
            diffs.append(
                {
                    "field": field_id,
                    "expected": expected_field.digest,
                    "actual": actual_field.digest,
                    "expected_zero": expected_field.zero,
                    "actual_zero": actual_field.zero,
                    "nbytes": len(expected_field.payload),
                }
            )
            logger.error(
                "[L2 verify] %s MISMATCH rank=%s group_id=%s host=%s device=%s "
                "field=%s expected=%s actual=%s expected_zero=%s actual_zero=%s "
                "nbytes=%s",
                stage,
                self.attn_tp_rank,
                group_id,
                host_block,
                device_block,
                field_id,
                expected_field.digest,
                actual_field.digest,
                int(expected_field.zero),
                int(actual_field.zero),
                len(expected_field.payload),
            )
        return diffs
