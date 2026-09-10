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

"""Persistent EP8/16 x BF16/W8A8 strict matrix; never starts a random soak."""

import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--lengths", default="dense", help="dense or comma-separated positive lengths"
    )
    parser.add_argument("--graph-every", type=int, default=1)
    args = parser.parse_args()
    lengths = (
        [2**p for p in range(14)] + list(range(8193, 16385))
        if args.lengths == "dense"
        else [int(v) for v in args.lengths.split(",")]
    )
    if (
        not lengths
        or min(lengths) <= 0
        or len(set(lengths)) != len(lengths)
        or args.graph_every <= 0
    ):
        parser.error("require distinct positive lengths and positive graph interval")
    for name in (
        "GMOE_EXCHANGE_OPTIONS",
        "TOKENSPEED_LITE_GMM13_LIBRARY",
        "TOKENSPEED_FUSED_MM2_LIBRARY",
    ):
        if not os.environ.get(name):
            parser.error(f"configure {name} in the system CANN environment first")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    combos = [(ep, dtype) for ep in (8, 16) for dtype in ("int8", "bf16")]
    manifest = dict(
        lengths=lengths,
        combinations=combos,
        groups=8,
        hidden=4096,
        graph_every=args.graph_every,
        strict_bits=True,
        random_soak=False,
        libraries={
            name: os.environ[name]
            for name in (
                "GMOE_EXCHANGE_OPTIONS",
                "TOKENSPEED_LITE_GMM13_LIBRARY",
                "TOKENSPEED_FUSED_MM2_LIBRARY",
            )
        },
    )
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    script = Path(__file__).with_name("validate_lite_moe_fused_exchange.py")
    for ep, dtype in combos:
        label = f"ep{ep}-{dtype}"
        output = root / label
        env = dict(
            os.environ,
            OUTPUT_DIR=str(output),
            WEIGHT_DTYPE=dtype,
            SMOOTH_QUANT="both" if dtype == "int8" else "none",
            GMOE_NUM_GROUPS="8",
            TOKENS="1025",
            FULL_REFERENCE="1",
            STRICT_BITS="1",
            ROUTER_FUSION="1",
            ROUTED_FUSION="1",
            MM2_FUSION="1",
            ROUTED_SINGLE_KERNEL="1",
            SHARED_EARLY_TEST="0",
            SHARED_OVERLAP="0",
            SHARED_FFN="0",
            RANDOM_LENGTH_ITERATIONS="0",
            RANDOM_MIN_SECONDS="0",
            STABILITY_SECONDS="0",
            PRECISION_LENGTHS=args.lengths,
            RANDOM_GRAPH_EVERY=str(args.graph_every),
            REFERENCE_EXPERT_CHUNK_ROWS="8192",
            REFERENCE_CHECK_UNCHUNKED="1",
            ROUTE_CASE="mixed",
            PROFILE="0",
            MEASURE="0",
        )
        # Debug overrides must not silently bypass the requested matrix.
        for key in (
            "REPRO_INPUT_DIR",
            "TRACE_PROJECTION",
            "RANDOM_PROFILE_ITERATION",
            "SHARED_DEBUG",
        ):
            env.pop(key, None)
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={ep}",
            str(script),
        ]
        print(f"START {label}: {len(lengths)} lengths", flush=True)
        with (root / f"{label}.log").open("w") as log:
            process = subprocess.Popen(
                command,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                status = process.wait()
            except BaseException:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise
        if status:
            raise SystemExit(
                f'FAIL {label}: exit {status}; inspect {root / (label + ".log")}'
            )
        for rank in range(ep):
            record = json.loads(
                (output / f"precision-result-rank{rank:02d}.json").read_text()
            )
            assert record["passed"] and record["completed"] == len(lengths)
            assert record["token_histogram"] == {str(t): 1 for t in lengths}
            assert record["graph_token_histogram"] == {
                str(t): 1
                for index, t in enumerate(lengths)
                if index % args.graph_every == 0 or t == max(lengths)
            }
            assert record["world_size"] == ep and record["weight_dtype"] == dtype
            assert record["strict_bits"] and record["full_reference"]
            assert record["fused_expert_solution"] == "flash_npu_routed_full"
            assert record["reference_expert_solution"] == "torch_npu"
            assert record["baseline_pre"] == "composed_gmoe_pre"
            assert record["baseline_post"] == "composed_gmoe_post"
            assert record["bitwise_mismatches"] == record["nonfinite_outputs"] == 0
        print(f"PASS {label}: all {ep} ranks", flush=True)
    (root / "PASS.json").write_text(
        json.dumps(dict(combinations=4, lengths_per_combination=len(lengths)))
    )
    print("PASS: all four combinations, no random soak started", flush=True)


if __name__ == "__main__":
    main()
