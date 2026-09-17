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

"""Strict EP16/G8 W8A8 shape matrix and optional persistent decode/prefill soak.

Run in the configured system CANN/OPP/PYTHONPATH environment. This driver never
builds kernels, changes device quotas, or creates a virtual environment.
"""

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def cases():
    result = [
        dict(role="decode", tokens=t, route="mixed", smooth="both")
        for t in (1, 7, 31, 32, 33, 64, 65)
    ] + [
        dict(role="prefill", tokens=t, route="mixed", smooth="both")
        for t in (128, 257, 512, 1024, 2049, 4096)
    ]
    result += [
        dict(role=role, tokens=tokens, route=route, smooth="both")
        for role, tokens in (("decode", 64), ("prefill", 257))
        for route in ("real_only", "zero_only", "hot_real")
    ]
    result += [
        dict(role="decode", tokens=64, route="mixed", smooth=smooth)
        for smooth in ("none", "w13", "w2")
    ]
    return result


def case_name(case):
    return "{role}-t{tokens}-{route}-sq-{smooth}".format(**case)


def validate_records(root, replays, fixed, seconds):
    paths = sorted(root.glob("int8-rank*.json"))
    if len(paths) != 16:
        raise AssertionError(f"expected 16 rank records, got {len(paths)}")
    records = [json.loads(path.read_text()) for path in paths]
    assert {r["rank"] for r in records} == set(range(16))
    for r in records:
        assert r["full_reference"] and r["strict_bits"]
        assert r["reference_expert_solution"] == "torch_npu"
        assert r["baseline_pre"] == "composed_gmoe_pre"
        assert r["baseline_post"] == "composed_gmoe_post"
        assert r["bitwise_changed_replays"] == replays
        assert r["bitwise_fixed_replays"] == fixed
        assert r["max_abs_error"] == 0
        if seconds:
            assert r["stability_cycles"] > 0
            # Rank-zero deadline, not a per-rank independently rounded clock.
            assert records[0]["stability_elapsed_seconds"] >= seconds
    return dict(
        ranks=16,
        changed_replays=replays,
        fixed_replays=fixed,
        max_abs_error=max(r["max_abs_error"] for r in records),
        stability_cycles=min(r["stability_cycles"] for r in records),
        stability_elapsed_seconds=records[0]["stability_elapsed_seconds"],
        memory_growth_bytes=max(
            r["stability_memory_peak_bytes"] - r["stability_memory_start_bytes"]
            for r in records
        ),
    )


def run_case(root, case, replays, fixed, seconds, timeout):
    label = ("soak-" if seconds else "") + case_name(case)
    output = root / label
    output.mkdir()  # Never overwrite a previous run or mistake old records for new.
    env = {
        **os.environ,
        "OUTPUT_DIR": str(output),
        "TOKENS": str(case["tokens"]),
        "FORWARD_ROLE": case["role"],
        "ROUTE_CASE": case["route"],
        "SMOOTH_QUANT": case["smooth"],
        "WEIGHT_DTYPE": "int8",
        "GROUPS": "8",
        "ROUTER_FUSION": "1",
        "ROUTED_FUSION": "1",
        "MM2_FUSION": "1",
        "ROUTED_SINGLE_KERNEL": "1",
        "FULL_REFERENCE": "1",
        "STRICT_BITS": "1",
        "SHARED_EARLY_TEST": "0",
        "GRAPH_REPLAYS": str(replays),
        "FIXED_REPLAYS": str(fixed),
        "STABILITY_SECONDS": str(seconds),
        "MEASURE": "0",
        "PROFILE": "0",
    }
    script = Path(__file__).with_name("validate_lite_moe_fused_exchange.py")
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=16",
        str(script),
    ]
    start = time.monotonic()
    record = dict(name=label, case=case, requested_seconds=seconds)
    print("START " + json.dumps(record), flush=True)
    with (output / "run.log").open("w") as log:
        try:
            with subprocess.Popen(
                command,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            ) as process:
                try:
                    returncode = process.wait(timeout=timeout + seconds)
                except subprocess.TimeoutExpired:
                    # The session belongs only to this torchrun and its ranks.
                    # Reap every rank before the next case can acquire devices.
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    raise
                if returncode:
                    raise subprocess.CalledProcessError(returncode, command)
            record.update(
                status="passed", **validate_records(output, replays, fixed, seconds)
            )
        except (subprocess.SubprocessError, AssertionError) as error:
            record.update(status="failed", error=str(error))
    record["wall_seconds"] = time.monotonic() - start
    with (root / "results.jsonl").open("a") as log:
        log.write(json.dumps(record) + "\n")
    print("RESULT " + json.dumps(record), flush=True)
    return record["status"] == "passed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--fixed-replays", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--soak-hours", type=float, default=0)
    parser.add_argument(
        "--only", nargs="*", help="Exact matrix case names; omitted runs all"
    )
    args = parser.parse_args()
    if (
        args.replays < 1
        or args.fixed_replays < 1
        or args.timeout < 1
        or not math.isfinite(args.soak_hours)
        or args.soak_hours < 0
    ):
        parser.error(
            "replay counts/timeout must be positive; soak hours finite and nonnegative"
        )
    for name in (
        "GMOE_EXCHANGE_OPTIONS",
        "TOKENSPEED_LITE_GMM13_LIBRARY",
        "TOKENSPEED_FUSED_MM2_LIBRARY",
    ):
        if not os.environ.get(name):
            parser.error(f"configure {name} in the system environment first")
    matrix = cases()
    if args.only is not None:
        known = {case_name(c) for c in matrix}
        if not set(args.only) <= known:
            parser.error(f"unknown case names: {set(args.only) - known}")
        matrix = [c for c in matrix if case_name(c) in args.only]
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    manifest = dict(
        ep=16,
        groups=8,
        hidden=4096,
        local_experts=192,
        topk=16,
        weight_dtype="int8",
        weight_layout="NZ",
        matrix=matrix,
        replays=args.replays,
        fixed_replays=args.fixed_replays,
        requested_soak_hours=args.soak_hours,
    )
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    passed = True
    for case in matrix:
        passed = (
            run_case(root, case, args.replays, args.fixed_replays, 0, args.timeout)
            and passed
        )
    if not passed:
        raise SystemExit(
            "Shape matrix failed; soak was NOT started. Inspect results.jsonl and rank snapshots."
        )
    if args.soak_hours:
        seconds = args.soak_hours * 3600 / 2
        for role, tokens in (("decode", 64), ("prefill", 2049)):
            case = dict(role=role, tokens=tokens, route="mixed", smooth="both")
            if not run_case(
                root, case, args.replays, args.fixed_replays, seconds, args.timeout
            ):
                raise SystemExit(
                    "Stability mismatch/failure; remaining soak was NOT started."
                )
    print("PASS: all requested shapes and soak phases completed", flush=True)


if __name__ == "__main__":
    main()
