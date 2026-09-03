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

"""Unit tests for Host-persisted L2 store/load hashing."""

from __future__ import annotations

import os
import sys
import unittest
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


class L2VerifyEnvTest(unittest.TestCase):
    def setUp(self):
        try:
            from tokenspeed.runtime.cache.l2.verify import l2_verify_enabled
            from tokenspeed.runtime.utils.env import envs
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs runtime dependencies: {exc}")
        self.envs = envs
        self.l2_verify_enabled = l2_verify_enabled

    def test_flag_defaults_off(self):
        with self.envs.TOKENSPEED_L2_VERIFY.override(False):
            self.assertFalse(self.l2_verify_enabled())

    def test_flag_can_enable(self):
        with self.envs.TOKENSPEED_L2_VERIFY.override(True):
            self.assertTrue(self.l2_verify_enabled())


class L2TransferVerifierTest(unittest.TestCase):
    def setUp(self):
        try:
            from tokenspeed.runtime.cache.l2.verify import L2TransferVerifier
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs runtime dependencies: {exc}")
        self.device = torch.zeros(32, dtype=torch.uint8)
        self.host = torch.zeros(32, dtype=torch.uint8)
        self.verifier = L2TransferVerifier(
            _layout(self.device), _HostStorage(self.host), attn_tp_rank=0
        )

    def test_matching_store_uses_host_as_baseline(self):
        self.device[8:16] = torch.arange(8, dtype=torch.uint8)
        self.host[0:8] = torch.arange(8, dtype=torch.uint8)
        self.verifier.snapshot_store_device([(0, 1, 1)])
        mismatches = self.verifier.commit_store_host([(0, 1, 1)])
        self.assertEqual(mismatches, 0)
        baseline = self.verifier._baselines[(0, 1)]
        self.assertEqual(baseline.digest, self.verifier._hash_host_fields(0, 1).digest)

    def test_early_zero_device_vs_later_host_is_store_mismatch(self):
        self.verifier.snapshot_store_device([(0, 1, 1)])
        self.host[0:8] = torch.arange(8, dtype=torch.uint8) + 1
        mismatches = self.verifier.commit_store_host([(0, 1, 1)])
        self.assertEqual(mismatches, 1)
        baseline = self.verifier._baselines[(0, 1)]
        self.assertFalse(all(item.zero for item in baseline.fields.values()))

    def test_load_matches_persisted_host_not_early_device(self):
        self.verifier.snapshot_store_device([(0, 1, 1)])
        self.host[0:8] = torch.arange(8, dtype=torch.uint8) + 3
        self.verifier.commit_store_host([(0, 1, 1)])
        self.device[8:16] = torch.arange(8, dtype=torch.uint8) + 3
        self.assertEqual(self.verifier.snapshot_load_host([(0, 1, 1)]), 0)
        self.assertEqual(
            self.verifier.check_load_device(
                [(0, 1, 1)], stage="LOAD device post-h2d"
            ),
            0,
        )
        self.assertEqual(
            self.verifier.check_load_device(
                [(0, 1, 1)], stage="LOAD device at-poll"
            ),
            0,
        )
        self.assertEqual(self.verifier.stats.load_device_post_h2d_mismatches, 0)
        self.assertEqual(self.verifier.stats.load_device_at_poll_mismatches, 0)

    def test_load_device_mismatch_when_h2d_does_not_match_host(self):
        self.device[8:16] = torch.arange(8, dtype=torch.uint8) + 2
        self.host[0:8] = torch.arange(8, dtype=torch.uint8) + 2
        self.verifier.snapshot_store_device([(0, 1, 1)])
        self.verifier.commit_store_host([(0, 1, 1)])
        self.device[8:16] = 0
        self.assertEqual(self.verifier.snapshot_load_host([(0, 1, 1)]), 0)
        self.assertEqual(
            self.verifier.check_load_device(
                [(0, 1, 1)], stage="LOAD device post-h2d"
            ),
            1,
        )
        self.assertEqual(self.verifier.stats.load_device_post_h2d_mismatches, 1)

    def test_at_poll_mismatch_is_tracked_separately_from_post_h2d(self):
        self.host[0:8] = torch.arange(8, dtype=torch.uint8) + 4
        self.device[8:16] = torch.arange(8, dtype=torch.uint8) + 4
        self.verifier.snapshot_store_device([(0, 1, 1)])
        self.verifier.commit_store_host([(0, 1, 1)])
        self.assertEqual(
            self.verifier.check_load_device(
                [(0, 1, 1)], stage="LOAD device post-h2d"
            ),
            0,
        )
        self.device[8:16] = torch.arange(8, dtype=torch.uint8) + 9
        self.assertEqual(
            self.verifier.check_load_device(
                [(0, 1, 1)], stage="LOAD device at-poll"
            ),
            1,
        )
        self.assertEqual(self.verifier.stats.load_device_post_h2d_mismatches, 0)
        self.assertEqual(self.verifier.stats.load_device_at_poll_mismatches, 1)
