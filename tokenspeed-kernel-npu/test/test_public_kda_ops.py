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
import json
import os
from types import SimpleNamespace

import pytest
from tokenspeed_kernel_npu import public_kda_ops


@pytest.fixture(autouse=True)
def _clear_loader_cache(monkeypatch):
    # Keep any real process-global tiling handle pinned while unit tests use an
    # isolated handle list for their fake artifacts.
    monkeypatch.setattr(public_kda_ops, "_TILING_HANDLES", [])
    public_kda_ops.load_public_kda_ops.cache_clear()
    yield
    public_kda_ops.load_public_kda_ops.cache_clear()


def _artifact(tmp_path, *, binding="binding/ops.so"):
    root = tmp_path / "artifact"
    binding_path = root / binding
    vendor_api = (
        root
        / "opp"
        / "vendors"
        / "custom_transformer"
        / "op_api"
        / "lib"
        / "libcust_opapi.so"
    )
    vendor_tiling = (
        root
        / "opp"
        / "vendors"
        / "custom_transformer"
        / "op_impl"
        / "ai_core"
        / "tbe"
        / "op_tiling"
        / "liboptiling.so"
    )
    binding_path.parent.mkdir(parents=True)
    vendor_api.parent.mkdir(parents=True)
    vendor_tiling.parent.mkdir(parents=True)
    binding_path.write_bytes(b"binding")
    vendor_api.write_bytes(b"vendor")
    vendor_tiling.write_bytes(b"tiling")
    lock = public_kda_ops.load_lock()
    manifest = {
        "schema_version": 2,
        "source_commit": lock["commit"],
        "catlass_commit": lock["catlass_commit"],
        "soc": lock["soc"],
        "ops": list(public_kda_ops.resolve_ops(lock)),
        "binding": binding,
        "vendor_op_api": str(vendor_api.relative_to(root)),
        "vendor_op_tiling": str(vendor_tiling.relative_to(root)),
        "sha256": {
            "binding": hashlib.sha256(binding_path.read_bytes()).hexdigest(),
            "vendor_op_api": hashlib.sha256(vendor_api.read_bytes()).hexdigest(),
            "vendor_op_tiling": hashlib.sha256(vendor_tiling.read_bytes()).hexdigest(),
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root, manifest


def test_lock_resolves_dependencies_before_roots():
    lock = public_kda_ops.load_lock()
    resolved = public_kda_ops.resolve_ops(lock)
    assert set(resolved) == set(lock["dependencies"])
    assert resolved.index("chunk_gated_delta_rule_fwd_h") < resolved.index(
        "chunk_kda_fwd"
    )
    cyclic = {"root_ops": ["a"], "dependencies": {"a": ["b"], "b": ["a"]}}
    with pytest.raises(ValueError, match="dependency cycle"):
        public_kda_ops.resolve_ops(cyclic)


def test_missing_artifact_does_not_change_custom_opp(tmp_path, monkeypatch):
    monkeypatch.setattr(public_kda_ops, "ARTIFACT_ROOT", tmp_path / "missing")
    monkeypatch.setenv("ASCEND_CUSTOM_OPP_PATH", "/existing")
    status = public_kda_ops.load_public_kda_ops()
    assert not status.available
    assert os.environ["ASCEND_CUSTOM_OPP_PATH"] == "/existing"


@pytest.mark.parametrize(
    "corruption", ["source_commit", "binding", "digest", "tiling", "tiling_digest"]
)
def test_invalid_artifact_fails_before_environment_mutation(
    tmp_path, monkeypatch, corruption
):
    root, manifest = _artifact(tmp_path)
    if corruption == "source_commit":
        manifest["source_commit"] = "0" * 40
    elif corruption == "binding":
        manifest["binding"] = "../ops.so"
    elif corruption == "digest":
        manifest["sha256"]["binding"] = "0" * 64
    elif corruption == "tiling":
        manifest["vendor_op_tiling"] = "../liboptiling.so"
    else:
        manifest["sha256"]["vendor_op_tiling"] = "0" * 64
    (root / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(public_kda_ops, "ARTIFACT_ROOT", root)
    monkeypatch.setenv("ASCEND_CUSTOM_OPP_PATH", "/existing")
    status = public_kda_ops.load_public_kda_ops()
    assert not status.available
    assert os.environ["ASCEND_CUSTOM_OPP_PATH"] == "/existing"


def test_artifact_symlink_cannot_escape_package_root(tmp_path, monkeypatch):
    root, manifest = _artifact(tmp_path)
    tiling = root / manifest["vendor_op_tiling"]
    outside = tmp_path / "outside.so"
    outside.write_bytes(tiling.read_bytes())
    tiling.unlink()
    tiling.symlink_to(outside)
    monkeypatch.setattr(public_kda_ops, "ARTIFACT_ROOT", root)
    status = public_kda_ops.load_public_kda_ops()
    assert not status.available
    assert "stay below" in status.reason


def test_loader_reports_schemas_independently_and_is_idempotent(tmp_path, monkeypatch):
    root, _ = _artifact(tmp_path)
    calls = []
    namespace = SimpleNamespace(recurrent_kda=object())
    fake_ops = SimpleNamespace(
        load_library=lambda path: calls.append(("binding", path)),
        tokenspeed_npu_public_kda=namespace,
    )
    monkeypatch.setattr(public_kda_ops, "ARTIFACT_ROOT", root)
    monkeypatch.setattr(public_kda_ops, "torch", SimpleNamespace(ops=fake_ops))
    monkeypatch.setattr(
        public_kda_ops.ctypes,
        "CDLL",
        lambda path, mode: calls.append(("tiling", path, mode)) or object(),
    )
    monkeypatch.setenv("ASCEND_CUSTOM_OPP_PATH", "/existing:/existing")

    first = public_kda_ops.load_public_kda_ops()
    second = public_kda_ops.load_public_kda_ops()
    assert first is second
    assert first.available == {"recurrent_kda"}
    assert "chunk_kda_fwd" in first.reason
    assert [call[0] for call in calls] == ["tiling", "binding"]
    assert calls[0][2] == public_kda_ops.ctypes.RTLD_GLOBAL
    assert len(public_kda_ops._TILING_HANDLES) == 1
    values = os.environ["ASCEND_CUSTOM_OPP_PATH"].split(os.pathsep)
    assert values[0].endswith("opp/vendors/custom_transformer")
    assert values.count("/existing") == 2


def test_load_failure_restores_custom_opp(tmp_path, monkeypatch):
    root, _ = _artifact(tmp_path)

    def fail(_):
        raise RuntimeError("bad ABI")

    monkeypatch.setattr(public_kda_ops, "ARTIFACT_ROOT", root)
    handle = object()
    monkeypatch.setattr(public_kda_ops.ctypes, "CDLL", lambda *_args, **_kwargs: handle)
    monkeypatch.setattr(
        public_kda_ops, "torch", SimpleNamespace(ops=SimpleNamespace(load_library=fail))
    )
    monkeypatch.setenv("ASCEND_CUSTOM_OPP_PATH", "/existing")
    status = public_kda_ops.load_public_kda_ops()
    assert not status.available
    assert "bad ABI" in status.reason
    assert os.environ["ASCEND_CUSTOM_OPP_PATH"] == "/existing"
    assert public_kda_ops._TILING_HANDLES == [handle]


def test_tiling_load_failure_restores_custom_opp(tmp_path, monkeypatch):
    root, _ = _artifact(tmp_path)

    def fail(*_args, **_kwargs):
        raise OSError("bad tiling ABI")

    monkeypatch.setattr(public_kda_ops, "ARTIFACT_ROOT", root)
    monkeypatch.setattr(public_kda_ops.ctypes, "CDLL", fail)
    monkeypatch.setenv("ASCEND_CUSTOM_OPP_PATH", "/existing")
    status = public_kda_ops.load_public_kda_ops()
    assert not status.available
    assert "bad tiling ABI" in status.reason
    assert os.environ["ASCEND_CUSTOM_OPP_PATH"] == "/existing"
    assert not public_kda_ops._TILING_HANDLES
