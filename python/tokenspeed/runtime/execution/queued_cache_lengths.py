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
"""Host lengths for forwards issued in FIFO order, independent of commits."""

from collections.abc import Sequence


class QueuedCacheLengths:
    """Track ordinary one-token execution by request identity and pool slot.

    Only the forward thread accesses this state. A reset begins a new slot
    lifetime, including PD landing and re-admission of the same request ID.
    Every advance returns a new immutable attention-length snapshot. The
    snapshot describes the forward being issued, not the last committed one.
    """

    def __init__(self) -> None:
        self._slots: dict[int, tuple[str, int]] = {}

    def reset(
        self,
        request_ids: Sequence[str],
        pool_indices: Sequence[int],
        lengths: Sequence[int],
    ) -> None:
        """Seed a new slot lifetime from the same values sent to the device."""
        if not len(request_ids) == len(pool_indices) == len(lengths):
            raise ValueError("host cache reset fields must have matching lengths")
        for rid, slot, length in zip(request_ids, pool_indices, lengths):
            self._slots[slot] = (rid, int(length))

    def advance(
        self,
        request_ids: Sequence[str],
        pool_indices: Sequence[int],
        input_lengths: Sequence[int],
        num_extends: int,
        single_token_decode: bool,
    ) -> tuple[int, ...] | None:
        """Advance issued work; return None if a device readback is needed.

        Prefill consumes its input length; ordinary decode consumes one.
        Unknown identities and variable acceptance invalidate their slots.
        Other known rows still advance when one row requires the fallback.
        """
        if not len(request_ids) == len(pool_indices) == len(input_lengths):
            raise ValueError("host cache advance fields must have matching lengths")
        result = []
        known = True
        for i, (rid, slot, width) in enumerate(
            zip(request_ids, pool_indices, input_lengths)
        ):
            previous = self._slots.get(slot)
            if (
                previous is None
                or previous[0] != rid
                or (i >= num_extends and (not single_token_decode or width != 1))
            ):
                self._slots.pop(slot, None)
                known = False
                continue
            length = previous[1] + width
            self._slots[slot] = (rid, length)
            result.append(length)
        return tuple(result) if known else None
