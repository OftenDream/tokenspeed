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

import copy
from test.runtime.test_lite_hybrid_cache import _pool
from test.runtime.test_lite_mla_eager import _Context
from types import SimpleNamespace

import pytest
import torch

from tokenspeed.runtime.configs.flash_kda_config import FLASHLocalConfig
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.models.flash_local_attention import (
    SeparateProjectionKimiLinearMLAAttention,
)
from tokenspeed.runtime.utils.env import global_server_args_dict


class _Backend:
    spec_num_tokens = 1
    supports_mla_projected_value_decode = False

    def __init__(self, pool):
        self.pool = pool

    def write_locations(self, layer, mode):
        return self.locations

    def forward(self, q, k, v, layer, pool, mode, bs, save_kv_cache, **kwargs):
        assert not save_kv_cache
        c = self.chunked_prefill_metadata
        from tokenspeed_kernel_npu.ops.mla import mla_extend_with_kvcache

        return mla_extend_with_kvcache(
            q.view(-1, 32, 576),
            pool.get_key_buffer(3).view(-1, 64, 1, 576),
            self.table,
            self.lengths,
            c.cum_extend_seq_lens,
            self.cukv,
            4096,
            self.prefix + 4096,
            128,
            512,
            64,
            192**-0.5,
            True,
            0.0,
            False,
            None,
        )

    def forward_extend_chunked(self, q, k, v, scale, cap, **kw):
        from tokenspeed_kernel_npu.ops.mla import mla_prefill

        return mla_prefill(
            q,
            k,
            v,
            kw["cum_seq_lens_q"],
            kw["cum_seq_lens_kv"],
            kw["max_q_len"],
            kw["max_kv_len"],
            scale,
            kw["seq_lens"],
            kw["causal"],
            cap,
            True,
            kw["out"],
        )

    def chunk(self, index):
        self.prefix = index * 4096
        positions = torch.arange(
            self.prefix, self.prefix + 4096, device="npu", dtype=torch.int32
        )
        self.locations = (
            self.table[0, positions // 64] * 64 + positions % 64
        ).contiguous()
        self.lengths = torch.tensor(
            [self.prefix + 4096], device="npu", dtype=torch.int32
        )
        self.cukv = torch.tensor(
            [0, self.prefix + 4096], device="npu", dtype=torch.int32
        )
        self.chunked_prefill_metadata = SimpleNamespace(
            use_absorbed_cached_extend=index > 0,
            extend_seq_lens_cpu=[4096],
            extend_seq_lens=torch.tensor([4096], device="npu", dtype=torch.int32),
            cum_extend_seq_lens=torch.tensor(
                [0, 4096], device="npu", dtype=torch.int32
            ),
            max_extend_seq_len=4096,
            chunked_loop_num=0,
        )
        return positions


@pytest.mark.skipif(
    not hasattr(torch, "npu") or not torch.npu.is_available(),
    reason="requires an Ascend NPU and Lite-capable flash_ops",
)
@torch.inference_mode()
def test_lite_mla_prolog_prefill_continuation(monkeypatch):
    import torch_npu

    torch.npu.set_device(0)
    torch.npu.config.allow_internal_format = True
    torch.manual_seed(20260911)
    monkeypatch.setitem(global_server_args_dict, "npu_enable_weight_nz", False)
    monkeypatch.setitem(global_server_args_dict, "disaggregation_mode", "prefill")
    cfg = FLASHLocalConfig()
    component = SimpleNamespace(tp_size=1, tp_rank=0, tp_group=(0,))
    mapping = SimpleNamespace(
        attn=SimpleNamespace(tp_size=8, tp_rank=0, tp_group=tuple(range(8))),
        mla_weight=component,
    )
    baseline = SeparateProjectionKimiLinearMLAAttention(
        cfg, mapping, layer_id=3, prefix=""
    ).to("npu")
    for name, p in baseline.named_parameters():
        p.normal_(mean=0.0, std=0.01)
        if "layernorm" in name:
            p.fill_(1.0)
    baseline.process_weights_after_loading(None)
    candidate = copy.deepcopy(baseline)
    monkeypatch.setitem(global_server_args_dict, "npu_enable_weight_nz", True)
    monkeypatch.setitem(global_server_args_dict, "disaggregation_mode", "prefill")
    for module in candidate.modules():
        if module is not candidate and hasattr(module, "process_weights_after_loading"):
            module.process_weights_after_loading(None)
    _, pool = _pool("npu", num_lcm_blocks=520, tp_size=8, linear_num_heads=32)
    cache = pool.get_key_buffer(3)
    cache.fill_(-0.5)
    untouched = pool.get_key_buffer(7)
    untouched.fill_(-0.75)
    backend = _Backend(pool)
    # Independent non-monotonic read pages and append write slots; page zero and
    # every page after the logical sequence are guard regions.
    backend.table = (torch.randperm(1024, device="npu", dtype=torch.int32) + 1).view(
        1, -1
    )
    ctx = _Context(ForwardMode.EXTEND, backend, pool, 1, 1, 4096)
    hits = []
    prolog = candidate._project_q_with_mla_prolog

    def record(*args):
        result = prolog(*args)
        hits.append(result is not None)
        return result

    monkeypatch.setattr(candidate, "_project_q_with_mla_prolog", record)
    reference_cache = cache.clone()
    for index in range(16):
        pos = backend.chunk(index)
        x = (torch.randn(4096, 3072, device="npu") * 0.01).to(torch.bfloat16)
        before = cache.clone()
        # Each path continues from its own previous result, so numerical drift
        # across all 16 chunks is included in the comparison.
        cache.copy_(reference_cache)
        a = baseline(pos, x, ctx, None, None, None)
        reference_cache = cache.clone()
        cache.copy_(before)
        b = candidate(pos, x, ctx, None, None, None)
        torch.npu.synchronize()
        torch.testing.assert_close(a, b, rtol=0.03, atol=0.004)
        assert (a.float() - b.float()).norm() < 0.003 * a.float().norm()
        torch.testing.assert_close(cache, reference_cache, rtol=0.02, atol=0.004)
        written = backend.locations.long()
        actual = cache[written].float()
        expected = reference_cache[written].float()
        assert (actual - expected).norm() < 0.003 * expected.norm()
        # Only the supplied write indices may change, including on continuation.
        mask = torch.ones(cache.shape[0], device="npu", dtype=torch.bool)
        mask[backend.locations.long()] = False
        assert torch.equal(cache[mask], before[mask])
        assert torch.all(untouched == -0.75)
        assert len(hits) == index
    assert len(hits) == 15 and all(hits)
    assert cache.stride() == (576, 576, 1)
    assert cache.storage_offset() > 0
    assert cache.dtype == torch.bfloat16
