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

from tokenspeed.runtime.execution.forward_step import replay_graph_then_update


class _RecordingGraph:
    def __init__(self) -> None:
        self.calls: list = []

    def replay(self) -> None:
        self.calls.append(("replay", None))

    def update(self, *, cpu_update_input) -> None:
        self.calls.append(("update", cpu_update_input))


def test_replay_is_queued_before_the_task_update() -> None:
    graph = _RecordingGraph()
    payload = [{"actual_seq_lengths_kv": [4, 5]}]

    replay_graph_then_update(graph, payload)

    assert graph.calls == [("replay", None), ("update", payload)]


def test_replay_without_update_input_never_touches_update() -> None:
    graph = _RecordingGraph()

    replay_graph_then_update(graph, None)

    assert graph.calls == [("replay", None)]
