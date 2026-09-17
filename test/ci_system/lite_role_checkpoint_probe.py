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

"""Load the exact Lite checkpoint in concurrent P8/D8 NPU role worlds."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from lite_host_table_probe import (
    mapping_metrics,
    normalize_file_mappings,
    rollup_metrics,
    touch_tables,
)

ROLE_WORLD_SIZE = 8
EXPECTED_PARAMETER_BYTES = {
    "prefill": 17_607_814_736,
    "decode": 14_414_900_816,
}
_CROSS_ROLE_CATEGORIES = ("kda", "mla", "moe_experts", "oe_projection")


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def role_mapping_kwargs(role: str) -> dict[str, int]:
    """Return the fixed parallel ownership inside one eight-rank role."""
    if role == "prefill":
        return {
            "attn_tp_size": 1,
            "attn_cp_size": 8,
            "attn_dp_size": 1,
            "dense_tp_size": 1,
            "moe_tp_size": 1,
            "moe_ep_size": 8,
            "linear_attn_tp_size": 8,
            "mla_weight_tp_size": 1,
        }
    if role == "decode":
        return {
            "attn_tp_size": 1,
            "attn_cp_size": 1,
            "attn_dp_size": 8,
            "dense_tp_size": 8,
            "moe_tp_size": 1,
            "moe_ep_size": 8,
            "linear_attn_tp_size": 8,
            "mla_weight_tp_size": 1,
        }
    raise ValueError(f"Unknown Lite role {role!r}.")


def _parameter_category(name: str, config: Any) -> str:
    if name == "model.embed_tokens.projection":
        return "oe_projection"
    if ".self_attn." in name:
        layer_id = int(name.split(".layers.", 1)[1].split(".", 1)[0])
        return "kda" if config.is_kda_layer(layer_id) else "mla"
    if ".moe.experts." in name:
        return "moe_experts"
    if ".moe." in name:
        return "grouped_moe"
    if ".embed_tokens.oe_tables." in name:
        return "host_oe"
    return "outer"


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def parameter_ledger(model: torch.nn.Module) -> dict[str, Any]:
    """Describe local parameter ownership without copying full tensors to host."""
    by_category: dict[str, dict[str, Any]] = {}
    tensors: dict[str, list[tuple[str, torch.Tensor]]] = {}
    for name, parameter in model.named_parameters():
        category = _parameter_category(name, model.config)
        entry = by_category.setdefault(
            category,
            {"parameter_count": 0, "bytes": 0, "devices": set()},
        )
        entry["parameter_count"] += 1
        entry["bytes"] += _tensor_bytes(parameter)
        entry["devices"].add(parameter.device.type)
        tensors.setdefault(category, []).append((name, parameter))

    fingerprints = {
        category: _category_fingerprint(category_tensors)
        for category, category_tensors in tensors.items()
        if category != "host_oe"
    }
    for entry in by_category.values():
        entry["devices"] = sorted(entry["devices"])
    npu_bytes = sum(
        entry["bytes"]
        for category, entry in by_category.items()
        if category != "host_oe"
    )
    return {
        "by_category": by_category,
        "npu_parameter_bytes": npu_bytes,
        "host_oe_bytes": by_category.get("host_oe", {}).get("bytes", 0),
        "fingerprints": fingerprints,
    }


def _category_fingerprint(
    named_tensors: list[tuple[str, torch.Tensor]],
) -> str:
    """Hash the full manifest and sparse values from representative tensors."""
    ordered = sorted(named_tensors)
    digest = hashlib.sha256()
    for name, tensor in ordered:
        digest.update(
            json.dumps(
                [name, list(tensor.shape), str(tensor.dtype), _tensor_bytes(tensor)],
                separators=(",", ":"),
            ).encode()
        )
    if not ordered or ordered[0][1].is_meta:
        return digest.hexdigest()
    selected = sorted({0, len(ordered) // 2, len(ordered) - 1})
    for index in selected:
        name, tensor = ordered[index]
        flat = tensor.detach().reshape(-1)
        offsets = sorted({0, flat.numel() // 2, flat.numel() - 1})
        values = flat[offsets].float().cpu().tolist()
        digest.update(json.dumps([name, values], separators=(",", ":")).encode())
    return digest.hexdigest()


def derived_buffer_bytes(model: torch.nn.Module) -> int:
    total = 0
    for layer in model.model.layers:
        attention = layer.self_attn
        conv_weights = getattr(attention, "conv_weights", None)
        if conv_weights is not None:
            total += _tensor_bytes(conv_weights)
        for name in ("w_kc", "w_vc"):
            tensor = getattr(attention, name, None)
            if tensor is not None:
                total += _tensor_bytes(tensor)
    return total


def expected_derived_buffer_bytes(config: Any, linear_tp_size: int) -> int:
    element_bytes = torch.empty((), dtype=torch.bfloat16).element_size()
    local_projection = (
        config.linear_num_heads * config.linear_head_dim // linear_tp_size
    )
    packed_conv = (
        len(config.linear_layer_ids)
        * 3
        * local_projection
        * config.linear_conv_size
        * element_bytes
    )
    mla = (
        len(config.full_attention_layer_ids)
        * config.num_attention_heads
        * config.kv_lora_rank
        * (config.qk_nope_head_dim + config.v_head_dim)
        * element_bytes
    )
    return packed_conv + mla


def first_nonfinite_parameter(model: torch.nn.Module) -> dict[str, Any] | None:
    """Return the first non-finite NPU parameter using bounded temporary storage."""
    chunk_elements = 4 * 1024 * 1024
    for name, parameter in model.named_parameters():
        if parameter.device.type != "npu":
            continue
        flat = parameter.detach().reshape(-1)
        for start in range(0, flat.numel(), chunk_elements):
            chunk = flat[start : start + chunk_elements]
            if not bool(torch.isfinite(chunk).all().item()):
                return {
                    "stage": "checkpoint.parameters",
                    "name": name,
                    "shape": list(parameter.shape),
                    "dtype": str(parameter.dtype),
                    "chunk_start": start,
                    "nonfinite": int((~torch.isfinite(chunk)).sum().item()),
                }
    return None


def _npu_snapshot(device: int) -> dict[str, int]:
    torch.npu.synchronize(device)
    return {
        "allocated_bytes": int(torch.npu.memory_allocated(device)),
        "reserved_bytes": int(torch.npu.memory_reserved(device)),
        "max_allocated_bytes": int(torch.npu.max_memory_allocated(device)),
    }


def _wait_for(path: Path, timeout: int) -> None:
    deadline = time.monotonic() + timeout
    while not path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for {path.name}.")
        time.sleep(0.1)


def _run_worker(args: argparse.Namespace) -> None:
    import torch.distributed as dist
    import torch_npu  # noqa: F401

    from tokenspeed.runtime.configs.device_config import DeviceConfig
    from tokenspeed.runtime.configs.load_config import LoadConfig
    from tokenspeed.runtime.configs.model_config import ModelConfig
    from tokenspeed.runtime.distributed.comm_backend import initialize_comm_backend
    from tokenspeed.runtime.distributed.mapping import Mapping
    from tokenspeed.runtime.distributed.process_group_manager import (
        process_group_manager as pg_manager,
    )
    from tokenspeed.runtime.model_loader import get_model

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != ROLE_WORLD_SIZE:
        raise ValueError(f"Lite role probe requires WORLD_SIZE={ROLE_WORLD_SIZE}.")
    device_id = args.device_offset + int(os.environ["LOCAL_RANK"])
    device = torch.device("npu", device_id)
    torch.npu.set_device(device)
    mapping = Mapping(
        rank=rank,
        world_size=world_size,
        nprocs_per_node=world_size,
        nnodes=1,
        base_gpu_id=args.device_offset,
        **role_mapping_kwargs(args.role),
    )
    pg_manager.init_distributed(
        mapping,
        backend="hccl",
        timeout=args.timeout,
        # HCCL does not expose a c10d backend for torch's eager device split.
        device_id=None,
    )
    for group in (
        mapping.world_group,
        mapping.attn.tp_group,
        mapping.attn.dp_group,
        mapping.linear_attn.tp_group,
        mapping.mla_weight.tp_group,
        mapping.dense.tp_group,
        mapping.moe.tp_ep_group,
    ):
        pg_manager.init_process_group(group)
    initialize_comm_backend()
    torch.npu.synchronize(device)
    torch.npu.reset_peak_memory_stats(device)
    hbm = {"distributed": _npu_snapshot(device_id)}

    server_args = SimpleNamespace(
        mapping=mapping,
        load_format="auto",
        ext_yaml=None,
    )
    model_config = ModelConfig(
        args.checkpoint,
        trust_remote_code=False,
        model_override_args="{}",
        dtype="bfloat16",
        server_args=server_args,
    )
    model = get_model(
        model_config=model_config,
        load_config=LoadConfig(load_format="auto"),
        device_config=DeviceConfig("npu"),
    )
    hbm["checkpoint_loaded"] = _npu_snapshot(device_id)
    ledger = parameter_ledger(model)
    derived_bytes = derived_buffer_bytes(model)
    expected_derived_bytes = expected_derived_buffer_bytes(
        model.config, mapping.linear_attn.tp_size
    )
    finite_failure = first_nonfinite_parameter(model) if args.check_finite else None

    tables = tuple(model.model.embed_tokens.oe_tables)
    table_pointers = [table.data_ptr() for table in tables]
    normalize_file_mappings(table_pointers)
    checksum = touch_tables(tables, args.touch_mib)
    ready = Path(args.barrier_root) / f"ready-{args.role}-{rank}"
    ready.touch()
    _wait_for(Path(args.barrier_root) / "measure", args.timeout)
    table_mapping = mapping_metrics(os.getpid(), table_pointers)
    process_rollup = rollup_metrics(os.getpid())
    result_path = Path(args.output_root) / f"{args.role}-rank-{rank}.json"
    expected_parameters = EXPECTED_PARAMETER_BYTES[args.role]
    expected_host_oe_bytes = sum(
        fragment.modulus * fragment.feature_width * 2
        for fragment in model.model.embed_tokens.spec.fragments
    )
    checks = {
        "parameter_bytes": ledger["npu_parameter_bytes"] == expected_parameters,
        "derived_bytes": derived_bytes == expected_derived_bytes,
        "host_oe_bytes": ledger["host_oe_bytes"] == expected_host_oe_bytes,
        "parameter_residency": all(
            entry["devices"] == (["cpu"] if category == "host_oe" else ["npu"])
            for category, entry in ledger["by_category"].items()
        ),
        "host_oe_file_backed": table_mapping["all_file_backed"],
        "host_oe_anonymous_zero": table_mapping["anonymous_kib"] == 0,
        "touch_finite": math.isfinite(checksum),
    }
    if args.check_finite:
        checks["parameters_finite"] = finite_failure is None
    _write_json(
        result_path,
        {
            "schema_version": 1,
            "role": args.role,
            "rank": rank,
            "device": device_id,
            "mapping": role_mapping_kwargs(args.role),
            "ledger": ledger,
            "derived_bytes": derived_bytes,
            "expected_derived_bytes": expected_derived_bytes,
            "hbm": hbm,
            "host_oe_mapping": table_mapping,
            "process_rollup": process_rollup,
            "finite_failure": finite_failure,
            "checks": checks,
            "passed": all(checks.values()),
        },
    )
    _wait_for(Path(args.barrier_root) / "release", args.timeout)
    dist.barrier()
    dist.destroy_process_group()


def role_command(
    *,
    role: str,
    checkpoint: str,
    output_root: str,
    barrier_root: str,
    port: int,
    device_offset: int,
    timeout: int,
    touch_mib: int,
    check_finite: bool,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc-per-node",
        str(ROLE_WORLD_SIZE),
        "--max-restarts",
        "0",
        "--master-port",
        str(port),
        str(Path(__file__).resolve()),
        "worker",
        "--role",
        role,
        "--checkpoint",
        checkpoint,
        "--output-root",
        output_root,
        "--barrier-root",
        barrier_root,
        "--device-offset",
        str(device_offset),
        "--timeout",
        str(timeout),
        "--touch-mib",
        str(touch_mib),
    ]
    if check_finite:
        command.append("--check-finite")
    return command


def _port_is_free(port: int) -> bool:
    with socket.socket() as listener:
        try:
            listener.bind(("", port))
        except OSError:
            return False
    return True


def _validate_controller_args(args: argparse.Namespace) -> tuple[Path, Path]:
    checkpoint = Path(args.checkpoint)
    if (
        not (checkpoint / "config.json").is_file()
        or not (checkpoint / "model.safetensors.index.json").is_file()
    ):
        raise ValueError("checkpoint must contain config and safetensors index files.")
    devices = set(range(args.prefill_device_offset, args.prefill_device_offset + 8))
    decode_devices = set(
        range(args.decode_device_offset, args.decode_device_offset + 8)
    )
    if devices & decode_devices:
        raise ValueError("Prefill and Decode device ranges must not overlap.")
    if args.prefill_port == args.decode_port or not all(
        _port_is_free(port) for port in (args.prefill_port, args.decode_port)
    ):
        raise ValueError("Role rendezvous ports must be distinct and free.")
    output_root = Path(args.output_root)
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError("output-root must be absent or empty.")
    output_root.mkdir(parents=True, exist_ok=True)
    barrier_root = output_root / ".barrier"
    barrier_root.mkdir()
    return output_root, barrier_root


def _terminate(processes: list[subprocess.Popen[Any]]) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()
    deadline = time.monotonic() + 30
    for process in processes:
        remaining = max(0.0, deadline - time.monotonic())
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def _wait_for_paths(
    paths: list[Path], processes: list[subprocess.Popen[Any]], timeout: int
) -> None:
    deadline = time.monotonic() + timeout
    while not all(path.exists() for path in paths):
        failed = [
            process.returncode for process in processes if process.poll() is not None
        ]
        if failed:
            raise RuntimeError(f"Lite role world exited before barrier: {failed}.")
        if time.monotonic() >= deadline:
            raise TimeoutError("Lite role worlds timed out at the global barrier.")
        time.sleep(0.2)


def _aggregate(workers: list[dict[str, Any]]) -> dict[str, Any]:
    indexed = {(worker["role"], worker["rank"]): worker for worker in workers}
    checks = {
        "worker_count": len(indexed) == 2 * ROLE_WORLD_SIZE,
        "worker_checks": all(worker["passed"] for worker in workers),
        "role_parameter_balance": all(
            len(
                {
                    worker["ledger"]["npu_parameter_bytes"]
                    for worker in workers
                    if worker["role"] == role
                }
            )
            == 1
            for role in ("prefill", "decode")
        ),
        "cross_role_fingerprints": (
            all(
                indexed[("prefill", rank)]["ledger"]["fingerprints"][category]
                == indexed[("decode", rank)]["ledger"]["fingerprints"][category]
                for rank in range(ROLE_WORLD_SIZE)
                for category in _CROSS_ROLE_CATEGORIES
            )
            if len(indexed) == 2 * ROLE_WORLD_SIZE
            else False
        ),
    }
    mappings = [worker["host_oe_mapping"] for worker in workers]
    sum_pss = sum(mapping["pss_kib"] for mapping in mappings)
    max_rss = max((mapping["rss_kib"] for mapping in mappings), default=0)
    checks["host_oe_pss_one_copy"] = sum_pss <= max_rss * 1.25 + 32 * 1024
    checks["host_oe_not_copied_to_hbm"] = all(
        worker["hbm"]["checkpoint_loaded"]["allocated_bytes"]
        - worker["hbm"]["distributed"]["allocated_bytes"]
        <= worker["ledger"]["npu_parameter_bytes"] + worker["derived_bytes"] + 1024**3
        for worker in workers
    )
    return {
        "schema_version": 1,
        "workers": workers,
        "totals": {
            "host_oe_pss_kib": sum_pss,
            "max_host_oe_rss_kib": max_rss,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def _run_controller(args: argparse.Namespace) -> None:
    output_root, barrier_root = _validate_controller_args(args)
    processes: list[subprocess.Popen[Any]] = []
    logs = []
    try:
        for role, port, offset in (
            ("prefill", args.prefill_port, args.prefill_device_offset),
            ("decode", args.decode_port, args.decode_device_offset),
        ):
            log = open(output_root / f"{role}.log", "w")
            logs.append(log)
            processes.append(
                subprocess.Popen(
                    role_command(
                        role=role,
                        checkpoint=args.checkpoint,
                        output_root=str(output_root),
                        barrier_root=str(barrier_root),
                        port=port,
                        device_offset=offset,
                        timeout=args.timeout,
                        touch_mib=args.touch_mib,
                        check_finite=args.check_finite,
                    ),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            )
        ready = [
            barrier_root / f"ready-{role}-{rank}"
            for role in ("prefill", "decode")
            for rank in range(ROLE_WORLD_SIZE)
        ]
        _wait_for_paths(ready, processes, args.timeout)
        (barrier_root / "measure").touch()
        result_paths = [
            output_root / f"{role}-rank-{rank}.json"
            for role in ("prefill", "decode")
            for rank in range(ROLE_WORLD_SIZE)
        ]
        _wait_for_paths(result_paths, processes, args.timeout)
        workers = [json.loads(path.read_text()) for path in result_paths]
        result = _aggregate(workers)
        _write_json(output_root / "summary.json", result)
        (barrier_root / "release").touch()
        return_codes = [process.wait(timeout=120) for process in processes]
        if any(return_codes) or not result["passed"]:
            raise SystemExit("Lite 8P8D checkpoint probe failed a hard gate.")
    finally:
        (barrier_root / "release").touch()
        _terminate(processes)
        for log in logs:
            log.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    controller = commands.add_parser("controller")
    controller.add_argument("--checkpoint", required=True)
    controller.add_argument("--output-root", required=True)
    controller.add_argument("--prefill-port", type=int, required=True)
    controller.add_argument("--decode-port", type=int, required=True)
    controller.add_argument("--prefill-device-offset", type=int, default=0)
    controller.add_argument("--decode-device-offset", type=int, default=8)
    controller.add_argument("--touch-mib", type=int, default=64)
    controller.add_argument("--timeout", type=int, default=3600)
    controller.add_argument("--check-finite", action="store_true")

    worker = commands.add_parser("worker")
    worker.add_argument("--role", choices=("prefill", "decode"), required=True)
    worker.add_argument("--checkpoint", required=True)
    worker.add_argument("--output-root", required=True)
    worker.add_argument("--barrier-root", required=True)
    worker.add_argument("--device-offset", type=int, required=True)
    worker.add_argument("--touch-mib", type=int, required=True)
    worker.add_argument("--timeout", type=int, required=True)
    worker.add_argument("--check-finite", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "worker":
        _run_worker(args)
    else:
        _run_controller(args)


if __name__ == "__main__":
    main()
