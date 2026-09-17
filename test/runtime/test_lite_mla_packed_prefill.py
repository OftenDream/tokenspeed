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

"""Synthetic full-size Lite MLA layer through actual cached-Extend backend.

Run with pytest on one NPU or torchrun --nproc-per-node=8 on an allocated
8-card node. This is a layer integration test, not a serving benchmark.
"""

import json
import os
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch_npu
from tokenspeed_kernel_npu.ops import mla_packed

from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.paged.mla import MLAAttnBackend
from tokenspeed.runtime.models.flash_local_attention import (
    SeparateProjectionKimiLinearMLAAttention,
)


class Pool:
    def __init__(self, page, pages):
        self.arena = SimpleNamespace(kv_page_size=page)
        self.quant_method = None
        self.backing = torch.full(
            (pages + 4, page, 1, 576), 0.125, device="npu", dtype=torch.bfloat16
        )
        self.cache = self.backing[2:-2]
        self.writes = 0

    def get_key_buffer(self, layer_id):
        return self.cache

    def set_mla_kv_buffer(self, layer, loc, *, cache_k_nope, cache_k_rope):
        self.writes += 1
        rows = torch.cat((cache_k_nope, cache_k_rope), dim=-1).view(-1, 576)
        self.cache.view(-1, 576).index_copy_(0, loc.long(), rows)


class Backend(MLAAttnBackend):
    def __init__(self, page, table):
        # Exercise the real forward_extend; the fixture supplies scheduler-owned
        # metadata and write locations without starting a serving scheduler.
        self.kernel_page_size = page
        self.kv_lora_rank, self.qk_nope_head_dim, self.qk_rope_head_dim = 512, 128, 64
        self.v_head_dim = 128
        self.kv_cache_dim, self.max_context_len = 576, 81920
        self.data_type, self.q_data_type = torch.bfloat16, torch.bfloat16
        self.num_local_heads = 32
        self.kernel_solution = "torch_npu"
        self.spec_num_tokens = 1
        self.table = table

    def set_chunk(self, prefix, count):
        device = self.table.device
        cu = torch.tensor([0, count], dtype=torch.int32, device=device)
        self.forward_prefill_metadata = SimpleNamespace(
            use_absorbed_cached_extend=self._should_use_absorbed_cached_extend(
                max_extend_seq_len=count, max_extend_prefix_len=prefix
            ),
            extend_seq_lens_cpu=[count],
            extend_seq_lens=torch.tensor([count], dtype=torch.int32, device=device),
            cum_extend_seq_lens=cu,
            cum_seq_lens_kv=torch.tensor(
                [0, prefix + count], dtype=torch.int32, device=device
            ),
            seq_lens=torch.tensor([prefix + count], dtype=torch.int64, device=device),
            max_extend_seq_len=count,
            max_extend_prefix_len=prefix,
            page_table=self.table,
            chunked_loop_num=0,
        )
        self.chunked_prefill_metadata = self.forward_prefill_metadata
        positions = torch.arange(prefix, prefix + count, device=device)
        self.locations = (
            self.table[0, positions // self.kernel_page_size].long()
            * self.kernel_page_size
            + positions % self.kernel_page_size
        )
        assert self.forward_prefill_metadata.use_absorbed_cached_extend == (prefix > 0)

    def write_locations(self, layer, mode):
        return self.locations

    def forward(self, q, k, v, layer, pool, mode, bs, save_kv_cache, **kwargs):
        assert mode == ForwardMode.EXTEND
        return self.forward_extend(
            q,
            k,
            v,
            layer,
            self.locations,
            pool,
            bs,
            save_kv_cache=save_kv_cache,
            **kwargs
        )


@dataclass
class Context:
    forward_mode: ForwardMode
    attn_backend: Backend
    token_to_kv_pool: Pool
    bs: int
    num_extends: int
    input_num_tokens: int


@torch.inference_mode()
def run_layer(chunks, count, attention_tp):
    torch.manual_seed(1729)
    config = SimpleNamespace(
        hidden_size=3072,
        num_attention_heads=32,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        rms_norm_eps=1e-5,
        mla_scale_q_lora=True,
        mla_scale_kv_lora=True,
        mla_use_output_gate=True,
        max_position_embeddings=81920,
    )
    component = SimpleNamespace(tp_size=1, tp_rank=0, tp_group=(0,))
    rank = dist.get_rank() if dist.is_initialized() else 0
    mapping = SimpleNamespace(
        mla_weight=component,
        attn=SimpleNamespace(
            tp_size=attention_tp, tp_rank=rank, tp_group=tuple(range(attention_tp))
        ),
    )
    layer = SeparateProjectionKimiLinearMLAAttention(
        config, mapping, layer_id=0, prefix=""
    ).to(device="npu", dtype=torch.bfloat16)
    for parameter in layer.parameters():
        parameter.normal_(0, 0.01)
    layer.process_weights_after_loading()
    assert layer.num_local_heads == 32 and layer.component_mapping.tp_size == 1
    assert layer.rotary_emb is None
    page = 64
    pool = Pool(page, 81920 // page)
    table = torch.randperm(81920 // page, dtype=torch.int32, device="npu").view(1, -1)
    backend = Backend(page, table)
    op_loader = mla_packed._packed_prefill_op
    op = op_loader()
    assert op is not None
    calls = []

    def tracked(*args, **kwargs):
        assert args[2].data_ptr() == pool.cache.data_ptr()
        calls.append(args[4])
        return op(*args, **kwargs)

    records = []
    try:
        for chunk in range(chunks):
            prefix = chunk * count
            backend.set_chunk(prefix, count)
            ctx = Context(ForwardMode.EXTEND, backend, pool, 1, 1, count)
            hidden = torch.randn(count, 3072, dtype=torch.bfloat16, device="npu") * 0.1
            positions = torch.arange(prefix, prefix + count, device="npu")
            before = pool.cache.clone()
            writes = pool.writes
            mla_packed._packed_prefill_op = lambda: tracked
            actual = layer(
                positions,
                hidden,
                ctx,
                comm_manager=None,
                block_scale=None,
                attnres_partial_args=None,
            )
            torch.npu.synchronize()
            assert pool.writes == writes + 1
            saved = actual.clone()
            cache_after = pool.cache.clone()
            # A distinct reference invocation over exactly the same prefix and
            # current input. Cache writes are compared, not retried on failure.
            pool.cache.copy_(before)
            mla_packed._packed_prefill_op = lambda: None
            expected = layer(
                positions,
                hidden,
                ctx,
                comm_manager=None,
                block_scale=None,
                attnres_partial_args=None,
            )
            torch.npu.synchronize()
            assert pool.writes == writes + 2
            torch.testing.assert_close(saved, expected, rtol=0.015, atol=0.002)
            assert torch.equal(pool.cache, cache_after)
            touched = torch.zeros(
                pool.cache.numel() // 576, dtype=torch.bool, device="npu"
            )
            touched[backend.locations] = True
            assert torch.equal(
                pool.cache.view(-1, 576)[~touched], before.view(-1, 576)[~touched]
            )
            assert torch.all(pool.backing[[0, 1, -2, -1]] == 0.125).item()
            diff = saved.float() - expected.float()
            record = dict(
                rank=rank,
                chunk=chunk,
                prefix=prefix,
                queries=count,
                heads=layer.num_local_heads,
                weight_tp=1,
                attention_tp=attention_tp,
                relative_l2=(diff.norm() / expected.float().norm()).item(),
                max_abs=diff.abs().max().item(),
            )
            records.append(record)
            print(json.dumps(record), flush=True)
        assert (
            len(calls) == chunks - 1
        ), "only continuation chunks may use packed Prefill"
    finally:
        mla_packed._packed_prefill_op = op_loader
    return records


@pytest.mark.skipif(
    not hasattr(torch, "npu") or not torch.npu.is_available(), reason="requires NPU"
)
def test_lite_layer_sixteen_chunks():
    run_layer(16, 4096, 1)


if __name__ == "__main__":
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    torch.ops.load_library(os.environ["MLA_FIA_PACKED_LIBRARY"])
    dist.init_process_group(backend="hccl")
    try:
        results = run_layer(16, 4096, dist.get_world_size())
        passed = torch.tensor([len(results)], dtype=torch.int32, device="npu")
        dist.all_reduce(passed)
        assert passed.item() == 16 * dist.get_world_size()
        print("TP8_LAYER_PASS", dist.get_rank(), passed.item(), flush=True)
    finally:
        dist.destroy_process_group()
