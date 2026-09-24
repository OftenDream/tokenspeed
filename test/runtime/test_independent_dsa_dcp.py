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


"""Run with torchrun --standalone --nproc-per-node=4 -m pytest <this file>."""

import dataclasses
import os
from test.runtime.conftest import kimi_recipe
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from tokenspeed_kernel.ops.kvcache.triton import set_mla_kv_buffer_triton
from tokenspeed_kernel.ops.quantization import quantize_fp8_with_scale

from tokenspeed.runtime.distributed.process_group_manager import process_group_manager
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.paged import dsa as dsa_module
from tokenspeed.runtime.layers.attention.configs.dsa import DSAConfig
from tokenspeed.runtime.layers.attention.configs.mla import MLAConfig
from tokenspeed.runtime.layers.attention.dcp.placement import (
    CachePlacement,
    resolve_cache_slots,
)
from tokenspeed.runtime.layers.attention.kv_cache.dsa import write_index_k_cache

pytestmark = pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) != 4,
    reason="requires torchrun with four CUDA ranks",
)


@pytest.fixture(scope="module")
def rank():
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    process_group_manager.register_process_group("nccl", (0, 1, 2, 3), dist.group.WORLD)
    yield rank
    dist.destroy_process_group()


class _Pool:
    def __init__(self, kv, index):
        self.kv, self.index = kv, index

    def get_key_buffer(self, layer_id):
        return self.kv

    def get_component(self, layer_id, name):
        assert name == "dsa_index_k"
        return self.index

    def set_mla_kv_buffer(self, layer, slots, nope, rope, *, write_mask):
        set_mla_kv_buffer_triton(
            self.kv,
            slots,
            nope,
            rope,
            enable_pdl=False,
            sanitize=False,
            write_mask=write_mask,
        )


@pytest.mark.parametrize("topk", [512, 2048])
@pytest.mark.parametrize("phase", ["prefill", "decode", "mixed"])
def test_independent_dsa_matches_replicated_cache(rank, monkeypatch, phase, topk):
    prefill = phase != "decode"
    mixed = phase == "mixed"
    from tokenspeed.runtime.utils.env import global_server_args_dict

    monkeypatch.setitem(global_server_args_dict, "chunked_prefill_size", 512)
    monkeypatch.setitem(global_server_args_dict, "force_deterministic_rsag", True)
    torch.manual_seed(135)
    torch.cuda.manual_seed(135)
    device = torch.device("cuda", rank)
    original_kv = (
        torch.randn(34 * 64, 1, 576, dtype=torch.bfloat16, device=device) * 0.3
    )
    original_index = torch.randn(34 * 64, 128, dtype=torch.bfloat16, device=device)
    lengths = torch.tensor([755, 543, 0], dtype=torch.int32, device=device)
    extends = torch.tensor(
        [5, 1, 1] if mixed else ([5, 3, 0] if prefill else [1, 1, 1]),
        device=device,
        dtype=torch.int32,
    )
    tokens = 7 if mixed else (8 if prefill else 3)
    query = torch.randn(tokens, 32, 576, dtype=torch.bfloat16, device=device) * 0.3
    query = query[:, rank * 8 : (rank + 1) * 8].contiguous()
    index_query = torch.randn(tokens, 16, 128, dtype=torch.bfloat16, device=device)
    weights = torch.randn(tokens, 16, dtype=torch.bfloat16, device=device)
    config = kimi_recipe(
        kv_cache_dtype=torch.bfloat16, context_len=1024, max_bs=3
    ).attn_config
    spec = DSAConfig(
        **{
            **dataclasses.asdict(config.component(MLAConfig)),
            "backend_name": "dsa",
            "num_attention_heads": 32,
            "attn_tp_size": 4,
            "uses_independent_index_cache": True,
        },
        index_topk=topk,
        index_head_dim=128,
        index_n_heads=16,
        index_init_tokens=2,
        index_local_tokens=3
    )
    layer = SimpleNamespace(
        layer_id=0,
        tp_q_head_num=8,
        head_dim=576,
        v_head_dim=512,
        logit_cap=0.0,
        scaling=192**-0.5,
    )
    outputs, selections, selected = [], [], []
    select = dsa_module.select_dsa_topk

    def record(*args, **kwargs):
        indices, counts = select(*args, **kwargs)
        selections.append((indices.clone(), counts.clone()))
        return indices, counts

    monkeypatch.setattr(dsa_module, "select_dsa_topk", record)
    for degree in (1, 4):
        selection_start = len(selections)
        group = (rank,) if degree == 1 else (0, 1, 2, 3)
        owner = 0 if degree == 1 else rank
        side = dataclasses.replace(
            config,
            device=str(device),
            components=(spec,),
            dcp_size=degree,
            dcp_group=group,
            dcp_rank=owner,
        )
        backend = dsa_module.DSABackend(side, spec, kernel_page_size=64)
        backend.init_cuda_graph_state(3)
        backend.configure_runtime(
            block_granularity=128, virtual_block_count=17, shard_count=degree
        )
        table = torch.zeros(
            (3, backend.max_num_pages), dtype=torch.int32, device=device
        )
        table[0, :16] = torch.tensor(
            [10, 11, 4, 5, 16, 17, 2, 3, 12, 13, 8, 9, 14, 15, 6, 7], device=device
        )
        table[1, :16] = torch.tensor(
            [30, 31, 18, 19, 24, 25, 26, 27, 20, 21, 22, 23, 28, 29, 32, 33],
            device=device,
        )
        if prefill:
            backend.init_forward_metadata(
                3,
                1 if mixed else 3,
                lengths,
                table,
                ForwardMode.MIXED if mixed else ForwardMode.EXTEND,
                extend_seq_lens=extends,
                extend_seq_lens_cpu=extends.cpu(),
                extend_prefix_lens=lengths - extends,
                extend_prefix_lens_cpu=(lengths - extends).cpu(),
                extend_with_prefix=True,
            )
            req = torch.repeat_interleave(
                torch.arange(3, device=device), extends.long()
            )
            ends = extends.cumsum(0)
            pos = lengths[req] - (ends[req] - torch.arange(tokens, device=device))
        else:
            backend.refresh_decode_metadata(3, 2, lengths, table)
            req = torch.arange(3, device=device)
            pos = lengths - 1
        safe_pos = pos.clamp_min(0)
        slots = table[req, safe_pos // 64].long() * 64 + safe_pos % 64
        slots = torch.where(pos >= 0, slots, 0)
        key, index_key = original_kv[slots], original_index[slots].unsqueeze(1)
        placement = CachePlacement(128, 17, group, owner)
        all_slots = torch.arange(34 * 64, device=device)
        physical, owned = resolve_cache_slots(all_slots, placement)
        kv = torch.zeros(
            ((16 // degree + 1) * 128, 1, 576), dtype=torch.bfloat16, device=device
        )
        index = torch.zeros(
            (kv.shape[0] // 64, 64, 132), dtype=torch.uint8, device=device
        )
        kv[physical[owned]] = original_kv[owned]
        write_index_k_cache(
            index,
            original_index,
            physical,
            page_size=64,
            head_dim=128,
            write_mask=owned,
        )
        destinations, mask = resolve_cache_slots(slots, placement)
        kv[destinations[mask]] = torch.nan
        pages = index.view(-1, 64 * 132)
        packed_keys = pages[:, : 64 * 128].reshape(-1, 64, 128)
        packed_scales = pages[:, 64 * 128 :].view(torch.float32)
        page, offset = destinations[mask] // 64, destinations[mask] % 64
        packed_keys[page, offset] = 255
        packed_scales[page, offset] = torch.nan
        pool = _Pool(kv, index)
        from tokenspeed.runtime.layers.attention.longcat_dsa import (
            LongCatDSAIndexerOutput,
        )
        from tokenspeed.runtime.models.longcat_dsa import LongCatDSAAttention

        model = SimpleNamespace(attn_mqa=layer, num_local_heads=8, kv_lora_rank=512)
        ctx = SimpleNamespace(
            attn_backend=SimpleNamespace(
                write_locations=lambda layer, mode: slots,
                chunked_prefill_metadata=backend.chunked_prefill_metadata,
                max_context_len=backend.max_context_len,
                prepare_sparse_selection=backend.prepare_sparse_selection,
                forward_sparse_prefill=backend.forward_sparse_prefill,
                forward_sparse_decode=backend.forward_sparse_decode,
            ),
            token_to_kv_pool=pool,
            bs=3,
            forward_mode=(
                ForwardMode.MIXED
                if mixed
                else (ForwardMode.EXTEND if prefill else ForwardMode.DECODE)
            ),
        )

        def forbidden(*args, **kwargs):
            raise AssertionError(
                "Independent GPU DSA must use sparse attention directly"
            )

        monkeypatch.setattr(backend, "forward_extend", forbidden)
        monkeypatch.setattr(backend, "forward_decode", forbidden)

        def forward():
            return LongCatDSAAttention._forward_independent_sparse(
                model,
                query,
                key,
                LongCatDSAIndexerOutput(index_query, index_key, weights),
                ctx,
                True,
            )

        result = forward()
        assert result.shape == (8, tokens, 512)
        assert torch.isfinite(result).all()
        torch.testing.assert_close(kv[destinations[mask]], key[mask], rtol=0, atol=0)
        quantized, scales = quantize_fp8_with_scale(
            index_key.view(-1, 128),
            granularity="token_group",
            group_size=128,
            scale_encoding="float32",
        )
        torch.testing.assert_close(
            packed_keys[page, offset], quantized.view(torch.uint8)[mask], rtol=0, atol=0
        )
        torch.testing.assert_close(
            packed_scales[page, offset],
            scales.reshape(-1)[:tokens][mask],
            rtol=0,
            atol=0,
        )
        assert not kv[:128].any() and not index[:2].any()
        outputs.append(result)
        selected.append(
            (
                torch.cat([x[0] for x in selections[selection_start:]]),
                torch.cat([x[1] for x in selections[selection_start:]]),
            )
        )
        if degree == 4 and phase == "decode":
            # Exercise the same cache geometry/metadata with graph replay,
            # changing query values and logical lengths after capture.
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                forward()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = forward()
            index_query.add_(0.125)
            query.add_(0.0625)
            shorter = lengths.clone()
            shorter[0] -= 7
            backend.refresh_decode_metadata(3, 2, shorter, table, for_graph_replay=True)
            graph.replay()
            replayed = captured.clone()
            expected = forward()
            torch.testing.assert_close(replayed, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        selected[0][0].sort(1).values, selected[1][0].sort(1).values, rtol=0, atol=0
    )
    torch.testing.assert_close(selected[0][1], selected[1][1], rtol=0, atol=0)
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0.02, atol=0.002)
    print(
        "phase=",
        phase,
        "rank=",
        rank,
        "max_abs=",
        (outputs[0].float() - outputs[1].float()).abs().max().item(),
    )
