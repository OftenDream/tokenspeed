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

"""Unit tests for raw L2 page dumps at store/load boundaries."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="runtime-1gpu")

_PAYLOAD = 8


class _HostStorage:
    def __init__(self, buffer):
        self.host_buffer = buffer

    def host_field_offset(self, group_index: int, block_id: int, field_index: int):
        return (block_id - 1) * _PAYLOAD + field_index * 0


def _layout(device: torch.Tensor) -> SimpleNamespace:
    field = SimpleNamespace(
        field_id="layer.0.latent_kv",
        device_buffer_index=0,
        device_block_zero_offset_bytes=0,
        block_stride_bytes=_PAYLOAD,
        payload_bytes=_PAYLOAD,
    )
    group = SimpleNamespace(group_id="full_attention", fields=(field,))
    return SimpleNamespace(groups=(group,), buffers=(device,))


class L2VerifyDumpTest(unittest.TestCase):
    def setUp(self):
        try:
            from tokenspeed.runtime.cache.l2.verify import L2TransferVerifier
            from tokenspeed.runtime.utils.env import envs
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs runtime dependencies: {exc}")
        self.L2TransferVerifier = L2TransferVerifier
        self.envs = envs
        self.tmp = tempfile.TemporaryDirectory()
        self.dump = Path(self.tmp.name) / "dump"

    def tearDown(self):
        self.tmp.cleanup()

    def test_host_mode_checks_host_corruption_without_device_access(self):
        from unittest.mock import patch

        device = torch.zeros(2 * _PAYLOAD, dtype=torch.uint8)
        host = torch.arange(_PAYLOAD, dtype=torch.uint8)
        transfers = [(0, 1, 1)]
        with self.envs.TOKENSPEED_L2_VERIFY_MODE.override("host"):
            verifier = self.L2TransferVerifier(_layout(device), _HostStorage(host))
        self.assertFalse(verifier.requires_device_sync)
        with patch.object(
            verifier, "_hash_device_fields", side_effect=AssertionError("Device read")
        ):
            self.assertEqual(verifier.commit_store_host(transfers), 0)
            self.assertEqual(verifier.snapshot_load_host(transfers), 0)
            host[0] = 99
            self.assertEqual(verifier.snapshot_load_host(transfers), 1)
            with self.assertRaises(RuntimeError):
                verifier.snapshot_store_device(transfers)
            with self.assertRaises(RuntimeError):
                verifier.check_load_device(transfers)

    def test_invalid_verify_mode_is_rejected(self):
        with self.envs.TOKENSPEED_L2_VERIFY_MODE.override("async-typo"):
            with self.assertRaisesRegex(ValueError, "sync or host"):
                self.L2TransferVerifier(None, None)

    def test_dump_store_and_load_roundtrip(self):
        # Host pages are 1-based (logical 0 is null). Device block 1 lives at
        # stride offset _PAYLOAD, so the Device buffer must hold two pages.
        device = torch.zeros(2 * _PAYLOAD, dtype=torch.uint8)
        device[_PAYLOAD:] = torch.arange(_PAYLOAD, dtype=torch.uint8)
        host = torch.arange(_PAYLOAD, dtype=torch.uint8)
        transfers = [(0, 1, 1)]
        with (
            self.envs.TOKENSPEED_L2_VERIFY.override(True),
            self.envs.TOKENSPEED_L2_VERIFY_DUMP.override(str(self.dump)),
        ):
            verifier = self.L2TransferVerifier(
                _layout(device), _HostStorage(host), attn_tp_rank=0
            )
            verifier.snapshot_store_device(transfers)
            verifier.commit_store_host(transfers)
            verifier.snapshot_load_host(transfers)
            verifier.check_load_device(transfers, stage="LOAD device post-h2d")

        pages = list((self.dump / "rank0" / "pages.jsonl").read_text().splitlines())
        recs = [json.loads(line) for line in pages if line]
        stages = {rec["stage"] for rec in recs}
        self.assertEqual(
            stages, {"STORE device", "STORE host", "LOAD host", "LOAD device"}
        )
        digest = recs[0]["digest"]
        blob = self.dump / "rank0" / "blobs" / f"{digest}.bin"
        self.assertTrue(blob.exists())
        self.assertEqual(blob.read_bytes(), bytes(range(_PAYLOAD)))
        self.assertTrue(all(rec["digest"] == digest for rec in recs))

    def test_dump_dir_uses_attn_tp_rank(self):
        device = torch.zeros(2 * _PAYLOAD, dtype=torch.uint8)
        device[_PAYLOAD:] = torch.arange(_PAYLOAD, dtype=torch.uint8)
        host = torch.arange(_PAYLOAD, dtype=torch.uint8)
        transfers = [(0, 1, 1)]
        with (
            self.envs.TOKENSPEED_L2_VERIFY.override(True),
            self.envs.TOKENSPEED_L2_VERIFY_DUMP.override(str(self.dump)),
        ):
            verifier = self.L2TransferVerifier(
                _layout(device), _HostStorage(host), attn_tp_rank=5
            )
            verifier.snapshot_store_device(transfers)
            verifier.commit_store_host(transfers)

        rank5 = self.dump / "rank5" / "pages.jsonl"
        self.assertTrue(rank5.exists())
        self.assertFalse((self.dump / "rank0" / "pages.jsonl").exists())
        recs = [json.loads(line) for line in rank5.read_text().splitlines() if line]
        self.assertTrue(recs)
        self.assertTrue(all(rec["rank"] == 5 for rec in recs))

    def test_device_side_forwards_attn_tp_rank_to_l2_executor(self):
        import inspect

        from tokenspeed.runtime.execution import device as device_mod

        source = inspect.getsource(device_mod.build_device_side)
        self.assertIn("attn_tp_rank=attn_tp_rank", source)


if __name__ == "__main__":
    unittest.main()
