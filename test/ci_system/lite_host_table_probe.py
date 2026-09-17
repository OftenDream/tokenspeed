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

"""Linux mmap accounting helpers for Lite host-resident OE tables."""

from __future__ import annotations

import ctypes
import math
import os
import re
from collections.abc import Iterable
from typing import Any

import torch

_SMAPS_FIELDS = (
    "Size",
    "Rss",
    "Pss",
    "Shared_Clean",
    "Private_Clean",
    "Private_Dirty",
    "Anonymous",
)
_SMAPS_HEADER = re.compile(r"^([0-9a-f]+)-([0-9a-f]+)\s+(\S+)\s+")


def _smaps_entries(pid: int) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    with open(f"/proc/{pid}/smaps") as source:
        for line in source:
            match = _SMAPS_HEADER.match(line)
            if match:
                current = {
                    "start": int(match.group(1), 16),
                    "end": int(match.group(2), 16),
                    "file_backed": len(line.split(maxsplit=5)) == 6
                    and not line.split(maxsplit=5)[5].startswith("["),
                    **{field: 0 for field in _SMAPS_FIELDS},
                }
                entries.append(current)
                continue
            if current is None or ":" not in line:
                continue
            field, raw_value = line.split(":", 1)
            if field in _SMAPS_FIELDS:
                current[field] = int(raw_value.split()[0])
    return entries


def mapping_metrics(pid: int, pointers: list[int]) -> dict[str, Any]:
    """Aggregate each unique mapping containing a supplied tensor pointer."""
    entries = _smaps_entries(pid)
    selected: dict[tuple[int, int], dict[str, Any]] = {}
    for pointer in pointers:
        entry = next(
            (
                candidate
                for candidate in entries
                if candidate["start"] <= pointer < candidate["end"]
            ),
            None,
        )
        if entry is None:
            raise RuntimeError("A Lite OE tensor pointer has no /proc mapping.")
        selected[(entry["start"], entry["end"])] = entry
    return {
        "mapping_count": len(selected),
        "all_file_backed": all(entry["file_backed"] for entry in selected.values()),
        **{
            f"{field.lower()}_kib": sum(entry[field] for entry in selected.values())
            for field in _SMAPS_FIELDS
        },
    }


def rollup_metrics(pid: int) -> dict[str, int]:
    """Read process-wide Linux proportional-set accounting."""
    result = {f"{field.lower()}_kib": 0 for field in _SMAPS_FIELDS}
    with open(f"/proc/{pid}/smaps_rollup") as source:
        for line in source:
            if ":" not in line:
                continue
            field, raw_value = line.split(":", 1)
            if field in _SMAPS_FIELDS:
                result[f"{field.lower()}_kib"] = int(raw_value.split()[0])
    return result


def normalize_file_mappings(pointers: list[int]) -> None:
    """Drop incidental PTEs and disable read-ahead for measured mappings."""
    entries = _smaps_entries(os.getpid())
    selected = {
        (entry["start"], entry["end"]): entry
        for pointer in pointers
        for entry in entries
        if entry["start"] <= pointer < entry["end"]
    }
    if not selected or not all(entry["file_backed"] for entry in selected.values()):
        raise RuntimeError("Lite OE PTE normalization requires file-backed mappings.")
    libc = ctypes.CDLL(None, use_errno=True)
    libc.madvise.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int)
    for entry in selected.values():
        for advice in (4, 1):  # MADV_DONTNEED, then MADV_RANDOM.
            if libc.madvise(entry["start"], entry["end"] - entry["start"], advice):
                error = ctypes.get_errno()
                raise OSError(error, os.strerror(error))


def touch_tables(tables: Iterable[torch.Tensor], touch_mib: int) -> float:
    """Read a bounded, evenly distributed sample from host OE table pages."""
    tables = tuple(tables)
    total_pages = max(1, touch_mib * 256)
    pages_per_table = max(1, total_pages // len(tables))
    checksum = 0.0
    for table in tables:
        row_width = table.shape[1]
        rows_per_page = max(
            1,
            os.sysconf("SC_PAGE_SIZE") // (row_width * table.element_size()),
        )
        rows = min(pages_per_table, math.ceil(table.shape[0] / rows_per_page))
        indices = torch.arange(rows) * rows_per_page
        checksum += float(table[indices, 0].float().sum())
    return checksum


__all__ = [
    "mapping_metrics",
    "normalize_file_mappings",
    "rollup_metrics",
    "touch_tables",
]
