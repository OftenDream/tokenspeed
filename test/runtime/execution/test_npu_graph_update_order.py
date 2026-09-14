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

"""The NPU decode graph must be replayed before its FIA tasks are refreshed."""

from types import SimpleNamespace

from tokenspeed.runtime.execution.forward_step import replay_graph_then_update


class _RecordingGraph:
    def __init__(self) -> None:
        self.calls: list = []
        self.graph_dispatch_mode = SimpleNamespace(
            update_stream=SimpleNamespace(
                wait_event=lambda event: self.calls.append(("wait", event))
            )
        )

    def replay(self) -> None:
        self.calls.append(("replay", None))

    def update(self, *, cpu_update_input) -> None:
        self.calls.append(("update", cpu_update_input))


def test_replay_is_queued_before_the_task_update() -> None:
    graph = _RecordingGraph()
    payload = [{"actual_seq_lengths_kv": [4, 5]}]

    done = SimpleNamespace(record=lambda: graph.calls.append(("record", None)))
    replay_graph_then_update(graph, payload, None, done)

    assert graph.calls == [("replay", None), ("update", payload), ("record", None)]


def test_replay_without_update_input_never_touches_update() -> None:
    graph = _RecordingGraph()

    replay_graph_then_update(graph, None, None, None)

    assert graph.calls == [("replay", None)]


def test_next_update_waits_for_previous_replay_without_host_synchronization():
    graph = _RecordingGraph()
    previous = object()
    done = SimpleNamespace(record=lambda: graph.calls.append(("record", None)))
    payload = [{"actual_seq_lengths_kv": [128, 129]}]
    replay_graph_then_update(graph, payload, previous, done)
    assert graph.calls == [
        ("replay", None),
        ("wait", previous),
        ("update", payload),
        ("record", None),
    ]
