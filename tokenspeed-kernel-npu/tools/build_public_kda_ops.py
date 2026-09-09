#!/usr/bin/env python3
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

"""Build the pinned public 910B KDA OPP subset and TokenSpeed binding."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

NPU_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = NPU_ROOT / "python"
sys.path.insert(0, str(PACKAGE_PARENT))

from tokenspeed_kernel_npu.public_kda_ops import (  # noqa: E402
    OP_NAMES,
    load_lock,
    resolve_ops,
)

DEFAULT_ARTIFACT_ROOT = PACKAGE_PARENT / "tokenspeed_kernel_npu" / "_public_kda_ops"
BINDING_SOURCE = NPU_ROOT / "csrc" / "public_kda_ops_binding.cpp"


def git_head(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


def validate_source(source: Path, lock: dict) -> None:
    if git_head(source) != lock["commit"]:
        raise ValueError(f"source checkout must be exact {lock['commit']}")
    catlass = source / "csrc" / "third_party" / "catlass"
    if (
        not (catlass / "include").is_dir()
        or git_head(catlass) != lock["catlass_commit"]
    ):
        raise ValueError(f"CATLASS checkout must be exact {lock['catlass_commit']}")
    required = (source / "csrc" / "build.sh", BINDING_SOURCE)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError(f"required public KDA build inputs are missing: {missing}")


def cann_toolkit_paths(home: Path) -> tuple[Path, Path]:
    for root in (home / f"{platform.machine()}-linux", home):
        include = root / "include"
        library = root / "lib64"
        if (include / "acl" / "acl_base.h").is_file() and all(
            (library / name).is_file() for name in ("libascendcl.so", "libopapi.so")
        ):
            return include, library
    raise RuntimeError(f"CANN headers and libraries are incomplete under {home}")


def build_opp(source: Path, staging: Path, lock: dict, ops: tuple[str, ...]) -> Path:
    command = [
        "bash",
        "build.sh",
        "--make_clean",
        "--pkg",
        f"--ops={','.join(ops)}",
        f"--soc={lock['soc']}",
    ]
    with tempfile.TemporaryDirectory(prefix="python-bin-", dir=staging) as python_bin:
        for name in ("python", "python3"):
            Path(python_bin, name).symlink_to(Path(sys.executable).resolve())
        env = os.environ.copy()
        env["PATH"] = os.pathsep.join((python_bin, env.get("PATH", "")))
        subprocess.run(command, cwd=source / "csrc", check=True, env=env)
    installers = list((source / "csrc" / "build").glob("cann-ops-transformer*.run"))
    if len(installers) != 1:
        raise RuntimeError(f"expected one OPP installer, found {len(installers)}")
    install_root = staging / "opp"
    install_root.mkdir()
    subprocess.run([str(installers[0]), f"--install-path={install_root}"], check=True)
    vendor = install_root / "vendors" / "custom_transformer"
    required = (
        vendor / "op_api" / "lib" / "libcust_opapi.so",
        vendor / "op_impl" / "ai_core" / "tbe" / "op_tiling" / "liboptiling.so",
    )
    if not all(path.is_file() for path in required):
        raise RuntimeError(f"custom OPP install is incomplete: {vendor}")
    validate_vendor_api(required[0], ops)
    return vendor


def validate_vendor_api(library: Path, ops: tuple[str, ...]) -> None:
    """Reject partial OPP packages before their manifest is published."""

    output = subprocess.run(
        ["nm", "-D", "--defined-only", str(library)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    symbols = {line.split()[-1] for line in output.splitlines() if line.split()}
    expected = {
        f"aclnn{''.join(part.capitalize() for part in op.split('_'))}{suffix}"
        for op in ops
        for suffix in ("", "GetWorkspaceSize")
    }
    missing = sorted(expected - symbols)
    if missing:
        raise RuntimeError(f"custom OPP API is missing symbols: {missing}")


def build_binding(staging: Path) -> Path:
    import torch
    import torch_npu
    from torch.utils.cpp_extension import load

    torch_npu_root = Path(torch_npu.__path__[0])
    cann_include, cann_library = cann_toolkit_paths(
        Path(os.environ["ASCEND_HOME_PATH"])
    )
    build_dir = staging / "binding"
    build_dir.mkdir()
    library_path = load(
        name="tokenspeed_npu_public_kda_ops",
        sources=[str(BINDING_SOURCE)],
        extra_include_paths=[
            str(torch_npu_root / "include"),
            str(torch_npu_root / "include" / "third_party" / "op-plugin"),
            str(cann_include),
        ],
        extra_cflags=["-O2"],
        extra_ldflags=[
            f"-L{torch_npu_root / 'lib'}",
            "-ltorch_npu",
            f"-Wl,-rpath,{torch_npu_root / 'lib'}",
            f"-L{cann_library}",
            f"-Wl,-rpath,{cann_library}",
            "-lascendcl",
            "-lopapi",
        ],
        build_directory=str(build_dir),
        is_python_module=False,
        verbose=True,
    )
    library = Path(library_path).resolve()
    namespace = getattr(torch.ops, "tokenspeed_npu_public_kda")
    missing = [name for name in OP_NAMES if not hasattr(namespace, name)]
    if missing:
        raise RuntimeError(f"Torch binding schemas are missing: {missing}")
    for name in OP_NAMES:
        qualified = f"tokenspeed_npu_public_kda::{name}"
        for dispatch_key in ("PrivateUse1", "Meta"):
            if not torch._C._dispatch_has_kernel_for_dispatch_key(
                qualified, dispatch_key
            ):
                raise RuntimeError(f"{qualified} has no {dispatch_key} implementation")
    return library


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download_package(package: dict, target: Path) -> None:
    if target.is_file():
        if sha256(target) != package["sha256"]:
            raise ValueError(f"third-party package digest mismatch: {target.name}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.download")
    try:
        with urllib.request.urlopen(
            package["url"], timeout=60
        ) as response, temporary.open("wb") as stream:
            shutil.copyfileobj(response, stream)
        if sha256(temporary) != package["sha256"]:
            raise ValueError(f"downloaded package digest mismatch: {target.name}")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def _prepare_json_headers(archive_path: Path, third_party: Path) -> None:
    target = third_party / "json"
    if (target / "include" / "nlohmann" / "json.hpp").is_file():
        return
    if target.exists():
        raise ValueError(f"incomplete JSON headers require inspection: {target}")
    staging = Path(tempfile.mkdtemp(prefix=".json-", dir=third_party))
    try:
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                path = PurePosixPath(member.filename)
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError("JSON archive contains an unsafe path")
            archive.extractall(staging)
        if not (staging / "include" / "nlohmann" / "json.hpp").is_file():
            raise ValueError("JSON archive does not contain nlohmann/json.hpp")
        staging.rename(target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def prepare_third_party(source: Path, lock: dict) -> None:
    third_party = source / "csrc" / "third_party"
    package_root = third_party / "pkg"
    for package in lock["third_party_packages"]:
        archive_path = package_root / package["filename"]
        _download_package(package, archive_path)
        if package.get("unpack") == "json_headers":
            _prepare_json_headers(archive_path, third_party)


def write_manifest(
    staging: Path,
    binding: Path,
    vendor: Path,
    lock: dict,
    ops: tuple[str, ...],
) -> None:
    vendor_api = vendor / "op_api" / "lib" / "libcust_opapi.so"
    vendor_tiling = (
        vendor / "op_impl" / "ai_core" / "tbe" / "op_tiling" / "liboptiling.so"
    )
    manifest = {
        "schema_version": 2,
        "source_commit": lock["commit"],
        "catlass_commit": lock["catlass_commit"],
        "soc": lock["soc"],
        "ops": list(ops),
        "binding": str(binding.relative_to(staging)),
        "vendor_op_api": str(vendor_api.relative_to(staging)),
        "vendor_op_tiling": str(vendor_tiling.relative_to(staging)),
        "sha256": {
            "binding": sha256(binding),
            "vendor_op_api": sha256(vendor_api),
            "vendor_op_tiling": sha256(vendor_tiling),
        },
    }
    (staging / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def publish(staging: Path, target: Path) -> None:
    backup = target.with_name(f".{target.name}.previous")
    if backup.exists():
        raise RuntimeError(f"stale artifact backup requires inspection: {backup}")
    if target.exists():
        target.rename(backup)
    try:
        staging.rename(target)
    except Exception:
        if backup.exists():
            backup.rename(target)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    args = parser.parse_args()

    if "ASCEND_HOME_PATH" not in os.environ:
        raise RuntimeError("source the target CANN environment before building")
    source = args.source_dir.resolve()
    target = args.artifact_root.resolve()
    lock = load_lock()
    ops = resolve_ops(lock)
    validate_source(source, lock)
    prepare_third_party(source, lock)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
    try:
        vendor = build_opp(source, staging, lock, ops)
        binding = build_binding(staging)
        write_manifest(staging, binding, vendor, lock, ops)
        publish(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print(target / "manifest.json")


if __name__ == "__main__":
    main()
