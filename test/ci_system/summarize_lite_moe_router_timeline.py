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

"""Full-layer dispatch/router A/B summary, with shared-stream overlap evidence."""

import argparse
import json
import statistics
from decimal import Decimal
from pathlib import Path


def summarize(root: Path) -> dict:
    records = [json.loads(p.read_text()) for p in sorted(root.glob("int8-rank*.json"))]
    assert len(records) == 16
    shared_early_test = records[0].get("shared_early_test", False)
    result = {"excluded_initial_calls": 2}
    for mode in ("composed", "fused"):
        ranks = []
        for rank in sorted((root / mode).glob("rank*")):
            traces = list(rank.glob("**/trace_view.json"))
            assert len(traces) == 1, traces
            data = json.loads(traces[0].read_text())
            events = data if isinstance(data, list) else data["traceEvents"]
            device = sorted(
                (
                    e
                    for e in events
                    if e.get("ph") == "X"
                    and e.get("args", {}).get("Task Type")
                    in {"AI_CORE", "AI_VECTOR_CORE", "MIX_AIC", "MIX_AIV"}
                ),
                key=lambda e: Decimal(str(e["ts"])),
            )

            def start(e):
                return Decimal(str(e["ts"]))

            def end(e):
                return start(e) + Decimal(str(e["dur"]))

            dispatches = [e for e in device if "gmoe_dispatch" in e["name"]]
            assert len(dispatches) == 10, (rank, len(dispatches))
            assert all(
                ("gmoe_dispatch_router" in e["name"])
                == (mode == "fused" or shared_early_test)
                for e in dispatches
            )
            standalone_topk = [e for e in device if "MoeGatingTopK" in e["name"]]
            assert len(standalone_topk) == (
                0 if mode == "fused" or shared_early_test else 10
            )
            all_shared = [
                e
                for e in device
                if e["tid"] != dispatches[0]["tid"]
                and (
                    "DynamicQuant" in e["name"]
                    or "QuantMatmul" in e["name"]
                    or "DequantSwigluQuant" in e["name"]
                )
            ]
            assert len(all_shared) == len(dispatches) * 4
            measurements = []
            for index, dispatch in enumerate(dispatches):
                main = [e for e in device if e["tid"] == dispatch["tid"]]
                pos = main.index(dispatch)
                mm13 = next(
                    e
                    for e in main[pos:]
                    if "fused_init_routing_mm13_swiglu" in e["name"]
                )
                mm2 = next(
                    e for e in main[pos:] if "fused_mm2_fin_routing" in e["name"]
                )
                combine = next(e for e in main[pos:] if "gmoe_combine" in e["name"])
                final = next(e for e in main[pos:] if "aclnnAdd_" in e["name"])
                projection = main[pos - 3]
                assert "MatMul" in projection["name"]
                shared = all_shared[index * 4 : (index + 1) * 4]
                assert len(shared) == 4, (rank, index, shared)
                overlap = sum(
                    max(
                        Decimal(0),
                        min(end(e), end(dispatch)) - max(start(e), start(dispatch)),
                    )
                    for e in shared
                )
                measurements.append(
                    dict(
                        dispatch_us=float(dispatch["dur"]),
                        dispatch_to_experts_us=float(start(mm13) - start(dispatch)),
                        shared_span_us=float(end(shared[-1]) - start(shared[0])),
                        shared_gate_up_us=float(shared[1]["dur"]),
                        projection_us=float(projection["dur"]),
                        norm_us=float(main[pos - 2]["dur"]),
                        shared_lead_before_dispatch_us=float(
                            start(dispatch) - start(shared[0])
                        ),
                        shared_tail_after_dispatch_us=float(
                            max(Decimal(0), end(shared[-1]) - end(dispatch))
                        ),
                        shared_dispatch_overlap_us=float(overlap),
                        mm13_us=float(mm13["dur"]),
                        mm2_us=float(mm2["dur"]),
                        combine_us=float(combine["dur"]),
                        layer_device_span_us=float(
                            end(final) - min(start(projection), start(shared[0]))
                        ),
                    )
                )
            steady = measurements[2:]
            ranks.append(
                dict(
                    rank=rank.name,
                    calls=len(measurements),
                    medians={
                        key: statistics.median(m[key] for m in steady)
                        for key in steady[0]
                    },
                    samples=measurements,
                )
            )
        assert ranks, mode
        result[mode] = dict(
            ranks=ranks,
            medians={
                key: statistics.median(r["medians"][key] for r in ranks)
                for key in ranks[0]["medians"]
            },
        )
    result["shared_early_test"] = shared_early_test
    result["correctness"] = dict(
        ranks=len(records),
        eager_exact=all(r["eager_exact"] for r in records),
        changed_replays=[r["changed_input_graph_replays"] for r in records],
        exact_replays=[r["exact_replays"] for r in records],
        max_output_error=max(r["max_abs_error"] for r in records),
        max_route_weight_error=max(r["max_route_weight_error"] for r in records),
    )
    result["whole_layer_graph_us"] = {
        mode: statistics.median(r["graph_us"][mode] for r in records)
        for mode in ("composed", "fused")
    }
    result["core_quotas"] = {
        key: records[0][key]
        for key in ("router_cores", "shared_cube_cores", "shared_vector_cores")
    }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    print(json.dumps(summarize(parser.parse_args().root), indent=2))
