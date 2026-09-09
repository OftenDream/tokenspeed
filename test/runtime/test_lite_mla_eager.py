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

import math
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.models.flash_local_attention import (
    SeparateProjectionKimiLinearMLAAttention,
)

_NPU_AVAILABLE = hasattr(torch, "npu") and torch.npu.is_available()


def _config():
    return SimpleNamespace(
        hidden_size=8,
        num_attention_heads=2,
        q_lora_rank=4,
        kv_lora_rank=3,
        qk_nope_head_dim=2,
        qk_rope_head_dim=1,
        v_head_dim=2,
        rms_norm_eps=1e-5,
        mla_scale_q_lora=True,
        mla_scale_kv_lora=True,
        mla_use_output_gate=True,
        max_position_embeddings=128,
    )


class _Pool:
    def __init__(self, events):
        self.events = events
        self.rows = None

    def set_mla_kv_buffer(
        self,
        layer,
        loc,
        kv=None,
        auxiliary=None,
        *,
        cache_k_nope=None,
        cache_k_rope=None,
    ):
        self.events.append("write")
        kv = kv if kv is not None else cache_k_nope
        auxiliary = auxiliary if auxiliary is not None else cache_k_rope
        self.rows = (layer, loc.clone(), kv.clone(), auxiliary.clone())


class _Backend:
    def __init__(self, events, output):
        self.events = events
        self.output = output
        self.chunked_prefill_metadata = SimpleNamespace(
            use_absorbed_cached_extend=False,
            extend_seq_lens_cpu=[output.shape[0]],
            extend_seq_lens=torch.tensor(
                [output.shape[0]], dtype=torch.int32, device=output.device
            ),
            cum_extend_seq_lens=torch.tensor(
                [0, output.shape[0]], dtype=torch.int32, device=output.device
            ),
            max_extend_seq_len=output.shape[0],
            chunked_loop_num=0,
        )
        self.spec_num_tokens = 1
        self.supports_mla_projected_value_decode = False

    def select_out_cache_loc(self, _layer, loc, _mode):
        return loc

    def forward(self, _q, _k, v, layer, *_args, **_kwargs):
        self.events.append("attention")
        assert self.events[-2:] == ["write", "attention"]
        if v is not None:
            assert _q.shape[-1] == layer.qk_head_dim
        return self.output

    def forward_extend_chunked(self, _q, _k, _v, *_args, out=None, **_kwargs):
        self.events.append("attention")
        assert self.events[-2:] == ["write", "attention"]
        out.copy_(self.output)
        return out, torch.zeros(1, device=out.device)


@dataclass
class _Context:
    forward_mode: ForwardMode
    attn_backend: _Backend
    token_to_kv_pool: _Pool
    bs: int
    num_extends: int
    input_num_tokens: int


def _layer():
    torch.manual_seed(5501)
    component = SimpleNamespace(tp_size=1, tp_rank=0, tp_group=(0,))
    layer = SeparateProjectionKimiLinearMLAAttention(
        _config(),
        SimpleNamespace(attn=component, mla_weight=component),
        layer_id=3,
    ).to(torch.bfloat16)
    for parameter in layer.parameters():
        parameter.data.normal_(mean=0.0, std=0.2)
    return layer


def _ctx(mode, output, events):
    num_extends = output.shape[0] if mode.is_extend() else 0
    return _Context(
        forward_mode=mode,
        attn_backend=_Backend(events, output),
        token_to_kv_pool=_Pool(events),
        bs=output.shape[0],
        num_extends=num_extends,
        input_num_tokens=output.shape[0],
    )


def test_lite_mla_post_load_scales_once_and_prepares_absorbed_weights():
    layer = _layer()
    q_weight = layer.q_a_layernorm.weight.detach().clone()
    kv_weight = layer.kv_a_layernorm.weight.detach().clone()

    layer.process_weights_after_loading()
    first_q = layer.q_a_layernorm.weight.detach().clone()
    first_kv = layer.kv_a_layernorm.weight.detach().clone()
    layer.process_weights_after_loading()

    torch.testing.assert_close(first_q, q_weight * math.sqrt(2), rtol=0, atol=0)
    torch.testing.assert_close(first_kv, kv_weight * math.sqrt(8 / 3), rtol=0, atol=0)
    torch.testing.assert_close(layer.q_a_layernorm.weight, first_q, rtol=0, atol=0)
    torch.testing.assert_close(layer.kv_a_layernorm.weight, first_kv, rtol=0, atol=0)
    assert layer.w_kc.shape == (2, 2, 3)
    assert layer.w_vc.shape == (2, 3, 2)


@pytest.mark.parametrize("mode", [ForwardMode.EXTEND, ForwardMode.DECODE])
def test_lite_mla_writes_one_live_cache_before_attention_and_applies_gate(mode):
    layer = _layer()
    layer.process_weights_after_loading()
    hidden = torch.randn(2, 8, dtype=torch.bfloat16)
    events = []
    attention = (
        torch.randn(2, 2, 2, dtype=torch.bfloat16)
        if mode.is_extend()
        else torch.randn(2, 2, 3, dtype=torch.bfloat16)
    )
    ctx = _ctx(mode, attention, events)
    locations = torch.tensor([3, 5], dtype=torch.int64)

    with torch.no_grad():
        output = layer(torch.arange(2), hidden, ctx, locations, comm_manager=None)

    assert output.shape == hidden.shape
    assert events == ["write", "attention"]
    _, actual_locations, kv, auxiliary = ctx.token_to_kv_pool.rows
    assert torch.equal(actual_locations, locations)
    assert kv.shape == (2, 1, 3)
    assert auxiliary.shape == (2, 1, 1)
    assert torch.isfinite(output).all()


def test_lite_mla_nope_auxiliary_is_preserved_without_rotation():
    layer = _layer()
    layer.process_weights_after_loading()
    hidden = torch.randn(3, 8, dtype=torch.bfloat16)
    with torch.no_grad():
        q, latent, _, _ = layer._project_q_latent_gated(hidden, None, None, None)
    q_auxiliary = q.view(3, 2, 3)[..., -1].clone()
    latent_auxiliary = latent[..., -1].clone()
    events = []
    ctx = _ctx(
        ForwardMode.DECODE,
        torch.zeros(3, 2, 3, dtype=torch.bfloat16),
        events,
    )

    with torch.no_grad():
        absorbed, _ = layer.forward_absorb_qkv_proj(
            q,
            latent,
            torch.arange(3),
            ctx,
            torch.arange(3),
        )

    torch.testing.assert_close(ctx.token_to_kv_pool.rows[3][:, 0, 0], latent_auxiliary)
    assert torch.equal(absorbed[..., -1], q_auxiliary)


def test_lite_mla_cached_extend_uses_absorbed_attention():
    layer = _layer()
    layer.process_weights_after_loading()
    hidden = torch.randn(2, 8, dtype=torch.bfloat16)
    events = []
    ctx = _ctx(
        ForwardMode.EXTEND,
        torch.randn(2, 2, 3, dtype=torch.bfloat16),
        events,
    )
    ctx.attn_backend.chunked_prefill_metadata.use_absorbed_cached_extend = True

    with torch.no_grad():
        output = layer(torch.arange(2), hidden, ctx, torch.tensor([3, 5]))

    assert output.shape == hidden.shape
    assert events == ["write", "attention"]
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("mode", [ForwardMode.EXTEND, ForwardMode.DECODE])
@pytest.mark.skipif(not _NPU_AVAILABLE, reason="Ascend NPU is unavailable")
def test_ascend_lite_mla_model_path_uses_registered_projection(mode):
    layer = _layer().to("npu")
    layer.process_weights_after_loading()
    hidden = torch.randn(2, 8, dtype=torch.bfloat16, device="npu")
    events = []
    attention = (
        torch.randn(2, 2, 2, dtype=torch.bfloat16, device="npu")
        if mode.is_extend()
        else torch.randn(2, 2, 3, dtype=torch.bfloat16, device="npu")
    )
    ctx = _ctx(mode, attention, events)

    with torch.no_grad():
        output = layer(
            torch.arange(2, device="npu"),
            hidden,
            ctx,
            torch.tensor([3, 5], dtype=torch.int64, device="npu"),
        )

    assert output.shape == hidden.shape
    assert events == ["write", "attention"]
    assert torch.isfinite(output).all()


@pytest.mark.skipif(not _NPU_AVAILABLE, reason="Ascend NPU is unavailable")
def test_ascend_lite_mla_prefill_and_lse_match_oracle():
    from tokenspeed_kernel_npu.ops.mla import mla_prefill

    torch.manual_seed(5502)
    q = torch.randn(4, 32, 192, dtype=torch.bfloat16, device="npu")
    k = torch.randn_like(q)
    v = torch.randn(4, 32, 128, dtype=torch.bfloat16, device="npu")
    cu = torch.tensor([0, 4], dtype=torch.int32, device="npu")
    output, lse = mla_prefill(
        q,
        k,
        v,
        cu,
        cu,
        4,
        4,
        192**-0.5,
        None,
        True,
        0.0,
        True,
        None,
    )
    logits = torch.einsum("thd,shd->hts", q.float(), k.float()) * 192**-0.5
    logits.masked_fill_(
        torch.triu(
            torch.ones(4, 4, dtype=torch.bool, device="npu"), diagonal=1
        ).unsqueeze(0),
        -torch.inf,
    )
    expected = torch.einsum(
        "hts,shv->thv", torch.softmax(logits, dim=-1), v.float()
    ).to(v.dtype)
    expected_lse = torch.logsumexp(logits, dim=-1).transpose(0, 1)

    torch.testing.assert_close(output, expected, rtol=0.02, atol=0.02)
    torch.testing.assert_close(lse, expected_lse, rtol=2e-5, atol=2e-5)


@pytest.mark.skipif(not _NPU_AVAILABLE, reason="Ascend NPU is unavailable")
def test_ascend_lite_mla_absorbed_extend_crosses_page_boundary():
    from tokenspeed_kernel import (
        mla_extend_with_kvcache,
        mla_use_absorbed_extend,
    )

    torch.manual_seed(5505)
    q_lengths = (2, 3)
    cache_lengths = (5, 129)
    q = torch.randn(sum(q_lengths), 32, 576, dtype=torch.bfloat16, device="npu")
    cache = torch.randn(5, 64, 1, 576, dtype=torch.bfloat16, device="npu")
    table = torch.tensor([[1, -1, -1], [2, 3, 4]], dtype=torch.int32, device="npu")
    cu_q = torch.tensor([0, 2, 5], dtype=torch.int32, device="npu")
    cu_kv = torch.tensor([0, 5, 134], dtype=torch.int32, device="npu")
    lengths = torch.tensor(cache_lengths, dtype=torch.int64, device="npu")

    assert mla_use_absorbed_extend(
        q_dtype=q.dtype,
        kv_dtype=cache.dtype,
        num_q_heads=q.shape[1],
        page_size=cache.shape[1],
        qk_nope_head_dim=128,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        max_seqlen_q=max(q_lengths),
    )
    output, lse = mla_extend_with_kvcache(
        q=q,
        kv_cache=cache,
        page_table=table,
        cache_seqlens=lengths,
        cu_seqlens_q=cu_q,
        cu_seqlens_kv=cu_kv,
        max_seqlen_q=max(q_lengths),
        max_seqlen_k=max(cache_lengths),
        qk_nope_head_dim=128,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        softmax_scale=192**-0.5,
        is_causal=True,
        return_lse=True,
        solution="torch_npu",
    )

    expected_output = []
    expected_lse = []
    start = 0
    for row, (q_length, cache_length) in enumerate(
        zip(q_lengths, cache_lengths, strict=True)
    ):
        page_count = math.ceil(cache_length / cache.shape[1])
        key = cache[table[row, :page_count].long()].reshape(-1, 576)[:cache_length]
        query = q[start : start + q_length]
        logits = torch.einsum("thd,sd->hts", query.float(), key.float()) * 192**-0.5
        query_positions = cache_length - q_length + torch.arange(q_length, device="npu")
        key_positions = torch.arange(cache_length, device="npu")
        logits.masked_fill_(
            (key_positions[None, :] > query_positions[:, None]).unsqueeze(0),
            -torch.inf,
        )
        probabilities = torch.softmax(logits, dim=-1)
        expected_output.append(
            torch.einsum("hts,sv->thv", probabilities, key[:, :512].float()).to(q.dtype)
        )
        expected_lse.append(torch.logsumexp(logits, dim=-1).transpose(0, 1))
        start += q_length

    torch.testing.assert_close(output, torch.cat(expected_output), rtol=0.02, atol=0.02)
    torch.testing.assert_close(lse, torch.cat(expected_lse), rtol=2e-5, atol=2e-5)


@pytest.mark.skipif(not _NPU_AVAILABLE, reason="Ascend NPU is unavailable")
@pytest.mark.parametrize("lengths", [(73,), (73, 91)])
def test_ascend_lite_mla_decode_value_gate_and_merge(lengths):
    from tokenspeed_kernel.ops.attention.mla import mla_decode_with_kvcache
    from tokenspeed_kernel_npu.ops.mla import (
        attn_merge_state,
        mla_project_value,
    )

    torch.manual_seed(5503)
    batch = len(lengths)
    q = torch.randn(batch, 1, 32, 576, dtype=torch.bfloat16, device="npu")
    cache = torch.randn(6, 64, 1, 576, dtype=torch.bfloat16, device="npu")
    table = torch.tensor([[1, 2], [3, 4]][:batch], dtype=torch.int32, device="npu")
    cache_seqlens = torch.tensor(lengths, dtype=torch.int64, device="npu")
    latent, lse = mla_decode_with_kvcache(
        q,
        cache,
        table,
        cache_seqlens,
        max(lengths),
        128,
        512,
        64,
        192**-0.5,
        0.0,
        True,
        None,
        solution="torch_npu",
    )
    expected_rows = []
    for row, length in enumerate(lengths):
        key = cache[table[row].long()].reshape(-1, 576)[:length]
        logits = q[row, 0].float() @ key.float().transpose(0, 1) * 192**-0.5
        expected_rows.append(
            (torch.softmax(logits, dim=-1) @ key[:, :512].float()).to(q.dtype)
        )
    expected_latent = torch.stack(expected_rows).unsqueeze(1)
    torch.testing.assert_close(latent, expected_latent, rtol=0.02, atol=0.02)

    weight = torch.randn(32, 512, 128, dtype=torch.bfloat16, device="npu")
    gate = torch.randn(batch, 4096, dtype=torch.bfloat16, device="npu")
    projected = torch.empty_like(gate)
    mla_project_value(latent[:, 0], weight, gate, projected)
    expected = torch.bmm(latent[:, 0].transpose(0, 1), weight).transpose(0, 1)
    expected = expected.reshape_as(gate)
    expected.mul_(torch.sigmoid(gate).to(expected.dtype))
    torch.testing.assert_close(projected, expected, rtol=0, atol=0)

    merged, merged_lse = attn_merge_state(
        latent[:, 0], lse, latent[:, 0], lse, math.log2(math.e)
    )
    torch.testing.assert_close(merged, latent[:, 0], rtol=0.02, atol=0.02)
    torch.testing.assert_close(merged_lse, lse + math.log(2), rtol=2e-5, atol=2e-5)
    assert torch.isfinite(projected).all()


@pytest.mark.skipif(not _NPU_AVAILABLE, reason="Ascend NPU is unavailable")
def test_ascend_lite_mla_decode_graph_updates_live_lengths():
    from tokenspeed_kernel_npu.ops.mla import mla_decode_with_kvcache

    torch.manual_seed(5504)
    q = torch.randn(2, 1, 32, 576, dtype=torch.bfloat16, device="npu")
    cache = torch.randn(4, 128, 1, 576, dtype=torch.bfloat16, device="npu")
    table = torch.tensor([[1, 0], [2, 0]], dtype=torch.int32, device="npu")
    capture_lengths = torch.ones(2, dtype=torch.int64, device="npu")

    def decode():
        return mla_decode_with_kvcache(
            q,
            cache,
            table,
            capture_lengths,
            256,
            128,
            512,
            64,
            192**-0.5,
            0.0,
            False,
            None,
        )

    for _ in range(3):
        decode()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=torch.npu.Stream(), auto_dispatch_capture=True):
        output = decode()
    torch.npu.synchronize()

    previous = None
    for lengths, pages in (([3, 4], [1, 2]), ([5, 7], [3, 1])):
        q.copy_(torch.randn_like(q))
        cache.copy_(torch.randn_like(cache))
        table[:, 0].copy_(torch.tensor(pages, dtype=table.dtype, device="npu"))
        torch.npu.synchronize()
        graph.update(cpu_update_input=[{"actual_seq_lengths_kv": lengths}])
        graph.replay()
        torch.npu.synchronize()
        expected = []
        for row, length in enumerate(lengths):
            key = cache[pages[row], :length, 0]
            logits = q[row, 0].float() @ key.float().transpose(0, 1) * 192**-0.5
            expected.append(
                (torch.softmax(logits, dim=-1) @ key[:, :512].float()).to(q.dtype)
            )
        expected = torch.stack(expected).unsqueeze(1)
        torch.testing.assert_close(output, expected, rtol=0.02, atol=0.02)
        if previous is not None:
            assert not torch.equal(output, previous)
        previous = output.clone()
