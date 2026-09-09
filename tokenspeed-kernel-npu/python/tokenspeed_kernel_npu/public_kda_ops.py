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

"""Lazy loader for the optional public Ascend KDA operator artifact."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any

import torch

OP_NAMES = (
    "recurrent_kda",
    "kda_gate_cumsum",
    "chunk_kda_fwd",
)
NAMESPACE = "tokenspeed_npu_public_kda"
PACKAGE_ROOT = Path(__file__).resolve().parent
LOCK_PATH = PACKAGE_ROOT / "thirdparty" / "public_kda_ops.lock.json"
ARTIFACT_ROOT = PACKAGE_ROOT / "_public_kda_ops"
_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")
_DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}")
_TILING_HANDLES: list[Any] = []


@dataclass(frozen=True)
class PublicKdaOpsStatus:
    """Immutable result of one artifact load attempt."""

    available: frozenset[str]
    reason: str | None = None


def load_lock(path: Path = LOCK_PATH) -> dict[str, Any]:
    """Read and validate the pinned public source lock."""

    data = json.loads(path.read_text())
    required = {
        "repository",
        "commit",
        "catlass_repository",
        "catlass_commit",
        "soc",
        "third_party_packages",
        "root_ops",
        "dependencies",
    }
    if data.get("schema_version") != 1 or required - data.keys():
        raise ValueError("invalid public KDA lock schema")
    if data["soc"] != "ascend910b":
        raise ValueError("public KDA lock must target ascend910b")
    if not _SHA_PATTERN.fullmatch(data["commit"]) or not _SHA_PATTERN.fullmatch(
        data["catlass_commit"]
    ):
        raise ValueError("public KDA revisions must be full lowercase Git SHAs")
    packages = data["third_party_packages"]
    if not isinstance(packages, list) or not packages:
        raise ValueError("public KDA third-party packages must be a non-empty list")
    filenames = set()
    for package in packages:
        if not isinstance(package, dict):
            raise ValueError("public KDA third-party package must be an object")
        filename = package.get("filename")
        if (
            not isinstance(filename, str)
            or PurePosixPath(filename).name != filename
            or filename in filenames
        ):
            raise ValueError("public KDA third-party filename must be unique and safe")
        if not str(package.get("url", "")).startswith("https://"):
            raise ValueError("public KDA third-party URL must use HTTPS")
        if not _DIGEST_PATTERN.fullmatch(str(package.get("sha256", ""))):
            raise ValueError("public KDA third-party digest must be SHA-256")
        if package.get("unpack") not in (None, "json_headers"):
            raise ValueError("unknown public KDA third-party unpack mode")
        filenames.add(filename)
    resolve_ops(data)
    return data


def resolve_ops(lock: dict[str, Any]) -> tuple[str, ...]:
    """Return the dependency closure in deterministic dependency-first order."""

    dependencies = lock["dependencies"]
    resolved: list[str] = []
    visiting: set[str] = set()

    def visit(name: str) -> None:
        if name in resolved:
            return
        if name in visiting:
            raise ValueError(f"public KDA dependency cycle at {name}")
        if name not in dependencies:
            raise ValueError(f"public KDA operator {name} has no dependency entry")
        visiting.add(name)
        for dependency in dependencies[name]:
            visit(dependency)
        visiting.remove(name)
        resolved.append(name)

    for root in lock["root_ops"]:
        visit(root)
    return tuple(resolved)


def _relative_file(root: Path, value: object) -> Path:
    if not isinstance(value, str):
        raise ValueError("artifact path must be a string")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError("artifact path must stay below its package root")
    resolved = root.joinpath(*path.parts).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("artifact path must stay below its package root") from error
    if not resolved.is_file():
        raise ValueError(f"artifact file is missing: {value}")
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validated_artifact(root: Path) -> tuple[Path, Path, Path]:
    lock = load_lock()
    manifest = json.loads((root / "manifest.json").read_text())
    expected = {
        "schema_version": 2,
        "source_commit": lock["commit"],
        "catlass_commit": lock["catlass_commit"],
        "soc": lock["soc"],
        "ops": list(resolve_ops(lock)),
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"artifact manifest {key} does not match the source lock")
    binding = _relative_file(root, manifest.get("binding"))
    vendor_api = _relative_file(root, manifest.get("vendor_op_api"))
    vendor_tiling = _relative_file(root, manifest.get("vendor_op_tiling"))
    digests = manifest.get("sha256")
    if not isinstance(digests, dict):
        raise ValueError("artifact manifest sha256 must be an object")
    for key, path in (
        ("binding", binding),
        ("vendor_op_api", vendor_api),
        ("vendor_op_tiling", vendor_tiling),
    ):
        expected_digest = digests.get(key)
        if not isinstance(expected_digest, str) or not _DIGEST_PATTERN.fullmatch(
            expected_digest
        ):
            raise ValueError(f"artifact manifest has an invalid {key} digest")
        if _sha256(path) != expected_digest:
            raise ValueError(f"artifact {key} digest mismatch")
    return binding, vendor_api.parents[2], vendor_tiling


def _prepend_custom_opp(vendor: Path) -> str | None:
    old = os.environ.get("ASCEND_CUSTOM_OPP_PATH")
    values = [value for value in (old or "").split(os.pathsep) if value]
    value = str(vendor)
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = os.pathsep.join(
        [value, *(item for item in values if item != value)]
    )
    return old


def _restore_custom_opp(old: str | None) -> None:
    if old is None:
        os.environ.pop("ASCEND_CUSTOM_OPP_PATH", None)
    else:
        os.environ["ASCEND_CUSTOM_OPP_PATH"] = old


@lru_cache(maxsize=1)
def load_public_kda_ops() -> PublicKdaOpsStatus:
    """Load the optional artifact once and report schemas that are present."""

    if not (ARTIFACT_ROOT / "manifest.json").is_file():
        return PublicKdaOpsStatus(frozenset(), "public KDA artifact is not installed")
    try:
        binding, vendor, vendor_tiling = _validated_artifact(ARTIFACT_ROOT)
    except (OSError, ValueError, json.JSONDecodeError, KeyError, TypeError) as error:
        return PublicKdaOpsStatus(frozenset(), str(error))

    old_opp = _prepend_custom_opp(vendor)
    try:
        tiling_handle = ctypes.CDLL(str(vendor_tiling), mode=ctypes.RTLD_GLOBAL)
        _TILING_HANDLES.append(tiling_handle)
        torch.ops.load_library(str(binding))
    except (OSError, RuntimeError) as error:
        _restore_custom_opp(old_opp)
        return PublicKdaOpsStatus(
            frozenset(), f"cannot load public KDA artifact: {error}"
        )

    namespace = getattr(torch.ops, NAMESPACE)
    available = frozenset(name for name in OP_NAMES if hasattr(namespace, name))
    missing = sorted(set(OP_NAMES) - available)
    reason = f"public KDA binding is missing schemas: {missing}" if missing else None
    return PublicKdaOpsStatus(available, reason)


def is_available(name: str) -> bool:
    """Whether one named public KDA schema was loaded."""

    if name not in OP_NAMES:
        raise ValueError(f"unknown public KDA operator: {name}")
    return name in load_public_kda_ops().available


__all__ = [
    "OP_NAMES",
    "PublicKdaOpsStatus",
    "is_available",
    "load_lock",
    "load_public_kda_ops",
    "resolve_ops",
]
