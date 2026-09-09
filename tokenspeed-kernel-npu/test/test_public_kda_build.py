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

import hashlib
import importlib.util
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from tokenspeed_kernel_npu.public_kda_ops import load_lock, resolve_ops

BUILD_SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "build_public_kda_ops.py"
SPEC = importlib.util.spec_from_file_location("build_public_kda_ops", BUILD_SCRIPT)
assert SPEC and SPEC.loader
build = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build)


def test_source_validation_requires_exact_parent_and_submodule(tmp_path, monkeypatch):
    source = tmp_path / "source"
    catlass = source / "csrc" / "third_party" / "catlass"
    catlass.joinpath("include").mkdir(parents=True)
    source.joinpath("csrc", "build.sh").write_text("#!/usr/bin/env bash\n")
    lock = load_lock()

    def exact_head(path):
        return lock["catlass_commit"] if path == catlass else lock["commit"]

    monkeypatch.setattr(build, "git_head", exact_head)
    build.validate_source(source, lock)
    monkeypatch.setattr(build, "git_head", lambda _: "0" * 40)
    with pytest.raises(ValueError, match="source checkout must be exact"):
        build.validate_source(source, lock)


def test_manifest_is_relative_and_reproducible(tmp_path):
    staging = tmp_path / "staging"
    binding = staging / "binding" / "ops.so"
    vendor = staging / "opp" / "vendors" / "custom_transformer"
    vendor_api = vendor / "op_api" / "lib" / "libcust_opapi.so"
    vendor_tiling = (
        vendor / "op_impl" / "ai_core" / "tbe" / "op_tiling" / "liboptiling.so"
    )
    binding.parent.mkdir(parents=True)
    vendor_api.parent.mkdir(parents=True)
    vendor_tiling.parent.mkdir(parents=True)
    binding.write_bytes(b"binding")
    vendor_api.write_bytes(b"vendor")
    vendor_tiling.write_bytes(b"tiling")
    lock = load_lock()
    build.write_manifest(staging, binding, vendor, lock, resolve_ops(lock))
    manifest = json.loads((staging / "manifest.json").read_text())
    assert manifest["schema_version"] == 2
    assert manifest["binding"] == "binding/ops.so"
    assert manifest["vendor_op_tiling"].endswith("op_tiling/liboptiling.so")
    assert not any(str(tmp_path) in str(value) for value in manifest.values())
    assert len(manifest["sha256"]["binding"]) == 64
    assert len(manifest["sha256"]["vendor_op_tiling"]) == 64


def test_vendor_api_rejects_partial_package(tmp_path, monkeypatch):
    library = tmp_path / "libcust_opapi.so"
    library.touch()
    monkeypatch.setattr(
        build.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            stdout="000 T aclnnRecurrentKda\n"
            "000 T aclnnRecurrentKdaGetWorkspaceSize\n"
        ),
    )
    with pytest.raises(RuntimeError, match="aclnnChunkKdaFwd"):
        build.validate_vendor_api(library, ("recurrent_kda", "chunk_kda_fwd"))


def test_publish_replaces_only_after_staging_is_complete(tmp_path):
    target = tmp_path / "artifact"
    target.mkdir()
    target.joinpath("old").write_text("old")
    staging = tmp_path / "staging"
    staging.mkdir()
    staging.joinpath("manifest.json").write_text("{}")
    build.publish(staging, target)
    assert not staging.exists()
    assert target.joinpath("manifest.json").is_file()
    assert not target.with_name(".artifact.previous").exists()


def test_prepare_third_party_verifies_preseeded_packages(tmp_path):
    source = tmp_path / "source"
    package_root = source / "csrc" / "third_party" / "pkg"
    package_root.mkdir(parents=True)
    archive = package_root / "include.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("include/nlohmann/json.hpp", "json")
    payload = b"dependency"
    package_root.joinpath("dependency.tar.gz").write_bytes(payload)
    lock = {
        "third_party_packages": [
            {
                "filename": "include.zip",
                "url": "https://example.invalid/include.zip",
                "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                "unpack": "json_headers",
            },
            {
                "filename": "dependency.tar.gz",
                "url": "https://example.invalid/dependency.tar.gz",
                "sha256": hashlib.sha256(payload).hexdigest(),
            },
        ]
    }
    build.prepare_third_party(source, lock)
    assert (
        source.joinpath(
            "csrc", "third_party", "json", "include", "nlohmann", "json.hpp"
        ).read_text()
        == "json"
    )

    package_root.joinpath("dependency.tar.gz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="digest mismatch"):
        build.prepare_third_party(source, lock)
