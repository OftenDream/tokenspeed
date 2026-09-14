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
import pytest

from tokenspeed.runtime.execution.queued_cache_lengths import QueuedCacheLengths


def test_pd_lengths_follow_issued_forwards_without_commits():
    state = QueuedCacheLengths()
    state.reset(("a", "b"), (7, 2), (65535, 65536))
    first = state.advance(("a", "b"), (7, 2), (1, 1), 0, True)
    second = state.advance(("b", "a"), (2, 7), (1, 1), 0, True)
    assert first == (65536, 65537)
    assert second == (65538, 65537)
    assert first == (65536, 65537)  # an in-flight snapshot is immutable
    for _ in range(128):
        last = state.advance(("a",), (7,), (1,), 0, True)
    assert last == (65665,)


def test_chunked_prefill_and_mixed_batch_advance_only_issued_rows():
    state = QueuedCacheLengths()
    state.reset(("a",), (3,), (0,))
    assert state.advance(("a",), (3,), (4096,), 1, True) == (4096,)
    state.reset(("a",), (3,), (4096,))
    assert state.advance(("a",), (3,), (4096,), 1, True) == (8192,)
    state.reset(("b",), (1,), (127,))
    assert state.advance(("b", "a"), (1, 3), (17, 1), 1, True) == (144, 8193)
    assert state.advance((), (), (), 0, True) == ()
    assert state.advance(("a",), (3,), (1,), 0, True) == (8194,)


def test_slot_reuse_and_readmission_invalidate_old_lifetime():
    state = QueuedCacheLengths()
    state.reset(("a", "b"), (0, 1), (500, 12))
    assert state.advance(("c", "b"), (0, 1), (1, 1), 0, True) is None
    assert state.advance(("b",), (1,), (1,), 0, True) == (14,)
    # A missing reset must never reuse a former owner's length.
    assert state.advance(("a",), (0,), (1,), 0, True) is None
    state.reset(("c",), (0,), (65536,))
    assert state.advance(("c",), (0,), (1,), 0, True) == (65537,)
    # Re-admission of the very same ID also starts a new slot lifetime.
    state.reset(("c",), (0,), (128,))
    assert state.advance(("c",), (0,), (1,), 0, True) == (129,)


@pytest.mark.parametrize("width,single", [(4, True), (1, False)])
def test_variable_acceptance_requires_readback_until_reset(width, single):
    state = QueuedCacheLengths()
    state.reset(("a",), (0,), (1024,))
    assert state.advance(("a",), (0,), (width,), 0, single) is None
    assert state.advance(("a",), (0,), (1,), 0, True) is None
    state.reset(("a",), (0,), (37,))
    assert state.advance(("a",), (0,), (1,), 0, True) == (38,)


def test_misaligned_host_fields_fail_before_mutation():
    state = QueuedCacheLengths()
    with pytest.raises(ValueError, match="matching lengths"):
        state.reset(("a", "b"), (0,), (10,))
    state.reset(("a",), (0,), (10,))
    with pytest.raises(ValueError, match="matching lengths"):
        state.advance(("a",), (), (1,), 0, True)
    assert state.advance(("a",), (0,), (1,), 0, True) == (11,)
