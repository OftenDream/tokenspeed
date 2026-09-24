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

"""Shared LongCat model contracts; physical NPU kernels are tested separately."""

import dataclasses
import os
from test.runtime.conftest import kimi_recipe
from types import SimpleNamespace
from unittest.mock import Mock, PropertyMock

import pytest
import torch
import torch.nn.functional as F
from tokenspeed_kernel.platform import current_platform

from tokenspeed.runtime.configs.model_config import AttentionArch
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention import registry
from tokenspeed.runtime.layers.attention.backends.paged import (
    ascend_dsa as ascend_backend,
)
from tokenspeed.runtime.layers.attention.backends.paged import dsa as dsa_backend
from tokenspeed.runtime.layers.attention.backends.paged import mla as mla_backend
from tokenspeed.runtime.layers.attention.configs.dsa import DSAConfig
from tokenspeed.runtime.layers.attention.configs.mla import MLAConfig
from tokenspeed.runtime.layers.attention.kv_cache import hybrid_kda
from tokenspeed.runtime.layers.attention.kv_cache.factory import create_cache_pool
from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import pack
from tokenspeed.runtime.layers.attention.longcat_dsa import _PackedLongCatDSAIndexer
from tokenspeed.runtime.models.longcat_dsa import LongCatDSAAttention, LongCatDSAIndexer
from tokenspeed.runtime.utils.cuda_stream import limit_stream_cores


def _lite_dsa_config(**kwargs) -> DSAConfig:
    kwargs.update(
        uses_independent_index_cache=True,
    )
    return DSAConfig(**kwargs)


def test_lite_reuses_standard_dsa_config_with_distinct_cache_layout():
    assert DSAConfig.is_dsa
    assert not DSAConfig.uses_independent_index_cache


@pytest.mark.parametrize("independent", [False, True])
def test_dsa_selection_policy_is_generated_by_standard_config(monkeypatch, independent):
    monkeypatch.setattr(
        MLAConfig,
        "_spec_kwargs",
        classmethod(lambda cls, server_args, model_config, is_draft: {}),
    )
    model_config = SimpleNamespace(
        hf_text_config=SimpleNamespace(
            uses_independent_dsa_selection=independent,
        ),
        index_topk=2048,
        index_head_dim=128,
        index_n_heads=16,
        index_init_tokens=4,
        index_local_tokens=1024,
    )

    values = DSAConfig._spec_kwargs(SimpleNamespace(), model_config, is_draft=False)

    assert values["index_init_tokens"] == 4
    assert values["index_local_tokens"] == 1024
    assert values["uses_independent_index_cache"] is independent
    assert "uses_dsa_dcp_partials" not in values


@pytest.mark.parametrize("device", ["cuda", "npu", "npu:0"])
@pytest.mark.parametrize("independent", [False, True])
def test_dsa_cache_placement_depends_on_device_and_layout(device, independent):
    recipe = kimi_recipe(kv_cache_dtype=torch.bfloat16)
    mla = recipe.attn_config.component(MLAConfig)
    dsa = DSAConfig(
        **{
            **dataclasses.asdict(mla),
            "backend_name": "dsa",
            "uses_independent_index_cache": independent,
        },
        index_n_heads=16,
        index_head_dim=128,
        index_topk=2048,
    )
    config = dataclasses.replace(
        recipe.attn_config,
        device=device,
        components=(dsa,),
    )
    replicated = independent and device.startswith("npu")
    assert config.uses_replicated_dcp_cache is replicated
    assert config.dcp_cache_shard_count == 1
    if device == "cuda" or replicated:
        config = dataclasses.replace(config, dcp_size=4, dcp_group=(0, 1, 2, 3))
        assert config.dcp_cache_shard_count == (1 if replicated else 4)


def test_longcat_dsa_inheritance_keeps_hybrid_kda_cache_pool(monkeypatch):
    recipe = kimi_recipe(kv_cache_dtype=torch.bfloat16)
    mla = recipe.attn_config.component(MLAConfig)
    dsa = _lite_dsa_config(
        **dataclasses.asdict(mla),
        index_n_heads=16,
        index_head_dim=128,
        index_topk=2048,
        index_init_tokens=4,
        index_local_tokens=1024,
    )
    config = dataclasses.replace(recipe.attn_config, components=(dsa,))
    expected = object()
    constructor = Mock(return_value=expected)
    monkeypatch.setattr(hybrid_kda, "HybridKDATokenToKVPool", constructor)

    result = create_cache_pool(
        SimpleNamespace(family="kimi_k3", layer_types=recipe.layer_types),
        config,
        object(),
        num_layers=48,
        rank=0,
    )

    assert result is expected
    constructor.assert_called_once()


def test_stream_core_limit_restores_outer_budget(monkeypatch):
    stream = SimpleNamespace(device="npu:0")
    calls = []
    device_module = SimpleNamespace(
        get_stream_limit=lambda received: {
            "cube_core_num": 20,
            "vector_core_num": 40,
        },
        set_stream_limit=lambda received, **limits: calls.append((received, limits)),
    )
    monkeypatch.setattr(torch, "get_device_module", lambda device: device_module)

    with limit_stream_cores(
        stream,
        cube_num=12,
        vector_num=24,
        enable=True,
    ):
        assert calls[-1] == (
            stream,
            {"cube_num": 12, "vector_num": 24},
        )

    assert calls[-1] == (
        stream,
        {"cube_num": 20, "vector_num": 40},
    )


def test_stream_core_limit_is_noop_when_disabled(monkeypatch):
    monkeypatch.setattr(
        torch,
        "get_device_module",
        Mock(side_effect=AssertionError("device module must not be queried")),
    )
    with limit_stream_cores(
        SimpleNamespace(device="npu:0"),
        cube_num=12,
        vector_num=24,
        enable=False,
    ):
        pass


def test_cp8_dp2_ep16_kda_tp8_groups_are_independent():
    for rank in range(16):
        mapping = Mapping(
            rank=rank,
            world_size=16,
            attn_tp_size=8,
            attn_dp_size=2,
            attn_dcp_size=8,
            dense_tp_size=8,
            moe_tp_size=1,
            moe_ep_size=16,
            linear_attn_tp_size=8,
            mla_weight_tp_size=8,
        )
        replica_start = rank // 8 * 8
        assert mapping.attn.dcp_group == tuple(range(replica_start, replica_start + 8))
        assert mapping.attn.dp_group == (rank % 8, rank % 8 + 8)
        assert mapping.linear_attn.tp_group == mapping.attn.dcp_group
        assert mapping.mla_weight.tp_group == mapping.attn.dcp_group
        assert mapping.moe.ep_group == tuple(range(16))


@pytest.mark.parametrize("rows", [1, 2, 32, 129])
def test_shared_indexer_separate_score_is_fp32(monkeypatch, rows):
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        indexer = LongCatDSAIndexer(
            SimpleNamespace(index_topk=12, index_n_heads=2, index_head_dim=8),
            hidden_size=16,
            q_lora_rank=8,
            qk_rope_head_dim=4,
            rope_theta=10000,
            rope_scaling=None,
            max_position_embeddings=256,
        )
    finally:
        torch.set_default_dtype(previous)
    assert not hasattr(indexer, "wk_weights_proj")
    assert indexer.weights_proj.weight.dtype == torch.float32
    assert indexer.wk.weight.dtype == torch.bfloat16
    assert indexer.rotary_emb.cos_sin_cache.dtype == torch.float32
    assert indexer._interleaved_rope_cache.dtype == torch.bfloat16
    generator = torch.Generator().manual_seed(19)
    for projection in (indexer.wq_b, indexer.wk, indexer.weights_proj):
        projection.weight.data.copy_(
            torch.randn(
                projection.weight.shape, generator=generator, dtype=torch.float32
            ).to(torch.bfloat16)
        )
        monkeypatch.setattr(
            projection, "forward", lambda x, p=projection: (F.linear(x, p.weight), None)
        )
    monkeypatch.setattr(indexer.k_norm, "forward", lambda x: x)

    # Observe the shared first-channel RoPE contract without a device kernel.
    def rope(positions, q, k):
        assert q.shape == (rows, 2, 4)
        assert k.shape == (rows, 1, 4)
        return q + 1, k + 2

    monkeypatch.setattr(indexer.rotary_emb, "forward", rope)
    hidden = torch.randn(rows, 16, generator=generator).to(torch.bfloat16)
    q_lora = torch.randn(rows, 8, generator=generator).to(torch.bfloat16)
    result = indexer(hidden, q_lora, torch.arange(rows))
    expected = F.linear(hidden.float(), indexer.weights_proj.weight).to(torch.bfloat16)
    assert torch.equal(result.weights, expected)
    q = F.linear(q_lora, indexer.wq_b.weight).view(rows, 2, 8)
    assert torch.equal(result.query[..., :4], q[..., :4] + 1)
    assert torch.equal(result.query[..., 4:], q[..., 4:])


def test_independent_indexer_does_not_mutate_paired_gpu_rope_cache():
    config = SimpleNamespace(index_topk=12, index_n_heads=2, index_head_dim=8)
    common = dict(
        config=config,
        hidden_size=16,
        q_lora_rank=8,
        qk_rope_head_dim=4,
        rope_theta=10000,
        rope_scaling=None,
        max_position_embeddings=257,
    )

    independent = LongCatDSAIndexer(**common)
    paired = _PackedLongCatDSAIndexer(**common)

    # get_rope() caches and shares modules. The Ascend-only BF16 table must be
    # private, otherwise constructing the independent indexer silently changes
    # the legacy paired GPU path's RoPE precision.
    assert independent.rotary_emb is paired.rotary_emb
    assert paired.rotary_emb.cos_sin_cache.dtype == torch.float32
    assert independent._interleaved_rope_cache.dtype == torch.bfloat16


def test_lite_reuses_longcat_owner_for_every_attention_layer(monkeypatch):
    config = SimpleNamespace(
        hidden_size=16,
        q_lora_rank=8,
        kv_lora_rank=8,
        num_attention_heads=8,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=4,
        rope_theta=10000,
        rope_scaling=None,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        cli_factor=1,
        index_topk=12,
        index_n_heads=2,
        index_head_dim=8,
        index_init_tokens=2,
        index_local_tokens=4,
        mla_use_nope=False,
        mla_use_output_gate=True,
        mla_scale_q_lora=True,
        mla_scale_kv_lora=True,
        is_longcat_dsa=True,
        uses_independent_dsa_selection=True,
    )
    tp = SimpleNamespace(tp_size=2, tp_rank=0, tp_group=(0, 1))
    mapping = SimpleNamespace(attn=tp, mla_weight=tp)
    layers = [
        LongCatDSAAttention(
            config=config,
            mapping=mapping,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            q_lora_rank=config.q_lora_rank,
            kv_lora_rank=config.kv_lora_rank,
            rope_theta=config.rope_theta,
            rope_scaling=config.rope_scaling,
            max_position_embeddings=config.max_position_embeddings,
            layer_id=i,
            prefix=f"layer.{i}",
            reduce_attn_results=False,
            computes_selection=True,
            selection_owner_layer_id=i,
            lora_norm_eps=config.rms_norm_eps,
            component_mapping=tp,
        )
        for i in (3, 7, 11, 15, 19, 23, 27)
    ]
    assert len({id(layer.indexer) for layer in layers}) == 7
    assert len({id(layer.indexer.wq_b.weight) for layer in layers}) == 7
    for layer in layers:
        assert type(layer) is LongCatDSAAttention
        assert layer.independent_selection
        assert layer._selection is None
        assert isinstance(layer.indexer, LongCatDSAIndexer)
        assert layer.computes_selection
        assert layer.selection_owner_layer_id == layer.layer_id
        assert layer.num_local_heads == 4
        assert layer.indexer.weights_proj.weight.dtype == torch.float32
        assert not layer.o_proj.reduce_results
        assert not hasattr(layer, "dsa_stream_fork")
    monkeypatch.setattr(
        "tokenspeed.runtime.models.longcat_dsa._prepare_mla_kv_b_proj_weights",
        lambda weight, layer: (weight, weight),
    )
    layer = layers[0]
    layer.q_a_layernorm.weight.data.fill_(1)
    layer.kv_a_layernorm.weight.data.fill_(1)
    layer.process_weights_after_loading()
    q_weight = layer.q_a_layernorm.weight.clone()
    kv_weight = layer.kv_a_layernorm.weight.clone()
    layer.process_weights_after_loading(layer)
    assert torch.equal(q_weight, layer.q_a_layernorm.weight)
    assert torch.equal(kv_weight, layer.kv_a_layernorm.weight)


@pytest.mark.parametrize("tp_size", [4, 8, 16])
@pytest.mark.parametrize("degree", [1, 4])
@pytest.mark.parametrize("device", ["cuda", "npu"])
def test_independent_cache_planes_have_exact_page_strides(tp_size, degree, device):
    recipe = kimi_recipe(tp_size=tp_size, kv_cache_dtype=torch.bfloat16)
    mla = recipe.attn_config.component(MLAConfig)
    dsa = _lite_dsa_config(
        **dataclasses.asdict(mla),
        index_n_heads=16,
        index_head_dim=128,
        index_topk=2048,
        index_init_tokens=4,
        index_local_tokens=1024,
    )
    dsa = dataclasses.replace(dsa, backend_name="dsa")
    recipe.attn_config = dataclasses.replace(
        recipe.attn_config,
        device=device,
        dcp_size=degree,
        dcp_group=tuple(range(degree)),
        components=tuple(
            dsa if isinstance(c, MLAConfig) else c
            for c in recipe.attn_config.components
        ),
    )
    groups = recipe.groups()
    assert all(
        spec.shard_count
        == (degree if spec.group_id == "full_attention" and device == "cuda" else 1)
        for spec, _ in groups
    )
    layout = pack(
        groups,
        prefix_granularity=recipe.prefix_granularity,
        cache_blocks_per_lcm_block=recipe.packing(groups),
        alignment=recipe.alignment,
        max_padding_fraction=recipe.max_padding_fraction,
    )
    recipe.check_layout(layout)
    assert len(layout.plane_bytes) == 24 * 2
    full_fields = next(
        fields for spec, fields in groups if spec.group_id == "full_attention"
    )
    assert len(full_fields) == 24 * 2
    index_fields = [
        field for field in full_fields if field.field_id.endswith("dsa_index_k")
    ]
    assert len(index_fields) == 24
    assert all(
        field.dtype == ("uint8" if device == "cuda" else "bfloat16")
        and field.shape == ((128, 132) if device == "cuda" else (128, 1, 128))
        for field in index_fields
    )
    assert all(field.exact_page_stride for field in full_fields)
    assert len({field.plane_id for field in full_fields}) == len(full_fields)


def test_paired_selection_keeps_existing_post_load_and_forward_contract():
    # The original model owns its weight preparation. The newly shared hook
    # must not touch paired-mode weights or require independent-mode state.
    layer = LongCatDSAAttention.__new__(LongCatDSAAttention)
    torch.nn.Module.__init__(layer)
    layer.independent_selection = False
    layer.process_weights_after_loading(layer)
    assert not hasattr(layer, "_scales_prepared")
    with pytest.raises(RuntimeError, match="forward_with_selection"):
        layer.forward(None, None, None, None)


@pytest.mark.parametrize("rows", [0, 1, 32])
def test_npu_rope_adapter_preserves_strided_head_views(monkeypatch, rows):
    pytest.importorskip("torch_npu")
    from tokenspeed_kernel_npu.ops import rotary_embedding as leaf

    q_full = torch.zeros(rows, 16, 128)
    k_full = torch.zeros(rows, 1, 128)
    q, k = q_full[..., :64], k_full[..., :64]
    calls = []

    def mrope(positions, query, key, cache, head_size, **kwargs):
        assert query.shape == (rows, 16 * 64)
        assert key.shape == (rows, 64)
        assert query.is_contiguous() and key.is_contiguous()
        calls.append(head_size)
        return query + 1, key + 2

    monkeypatch.setattr(leaf.torch_npu, "npu_mrope", mrope)
    leaf.apply_rope(
        positions=torch.arange(rows),
        q=q,
        k=k,
        head_size=64,
        cos_sin_cache=torch.zeros(max(rows, 1), 64),
        is_neox=False,
    )
    assert calls == ([64] if rows else [])
    assert torch.equal(q, torch.ones_like(q))
    assert torch.equal(k, torch.full_like(k, 2))
    assert torch.count_nonzero(q_full[..., 64:]) == 0
    assert torch.count_nonzero(k_full[..., 64:]) == 0


def test_longcat_indexer_rope_preserves_unrotated_tail(monkeypatch):
    pytest.importorskip("torch_npu")
    from tokenspeed_kernel_npu.ops import longcat_dsa as leaf

    tensor = torch.arange(2 * 3 * 8, dtype=torch.float32).view(2, 3, 8)
    positions = torch.tensor([1, 3], dtype=torch.int64)
    cache = torch.arange(5 * 4, dtype=torch.float32).view(5, 4)
    calls = []

    def interleave(rotary, cos, sin):
        calls.append((rotary.shape, cos.clone(), sin.clone()))
        return rotary + 7

    monkeypatch.setattr(leaf.torch_npu, "npu_interleave_rope", interleave)
    result = leaf.interleave_rope(
        tensor,
        positions,
        cache,
        rope_dim=4,
    )

    assert calls[0][0] == (2, 3, 1, 4)
    torch.testing.assert_close(result[..., :4], tensor[..., :4] + 7)
    assert torch.equal(result[..., 4:], tensor[..., 4:])


@pytest.fixture
def indexed_backend(monkeypatch):
    """Real DSA/MLA metadata with only the physical NPU kernels substituted."""
    config = kimi_recipe(kv_cache_dtype=torch.bfloat16, max_bs=8).attn_config
    # Metadata is allocated on CPU while this fixture simulates the Ascend
    # backend, including its replicated storage policy. Device-policy tests
    # above exercise the real property independently of this fixture.
    monkeypatch.setattr(
        type(config), "uses_replicated_dcp_cache", PropertyMock(return_value=True)
    )
    # ServerArgs uses the canonical string "none" for an unquantized cache.
    config = dataclasses.replace(config, kv_cache_quant_method="none")
    spec = _lite_dsa_config(
        **dataclasses.asdict(config.component(MLAConfig)),
        index_n_heads=16,
        index_head_dim=128,
        index_topk=2048,
        index_init_tokens=4,
        index_local_tokens=1024,
    )
    spec = dataclasses.replace(spec, backend_name="dsa")
    config = dataclasses.replace(config, components=(spec,))
    kernels = Mock()
    monkeypatch.setattr(
        ascend_backend,
        "current_platform",
        lambda: SimpleNamespace(is_nvidia=False, is_amd=False, is_npu=True),
    )
    monkeypatch.setattr(ascend_backend, "ascend_dsa_kernels", lambda: kernels)
    monkeypatch.setattr(
        dsa_backend, "dsa_plan", Mock(side_effect=AssertionError("Unexpected FP8 plan"))
    )
    # Exercise the normal MLA metadata builder, including its no-prefix arm
    # which omits the page table until the DSA wrapper publishes it.
    monkeypatch.setattr(mla_backend, "mla_use_absorbed_extend", lambda **kw: False)
    monkeypatch.setattr(
        mla_backend,
        "build_chunked_prefill_metadata_arrays",
        lambda *args: (0, [], [], [], []),
    )
    backend = ascend_backend.AscendDSABackend(config, spec, kernel_page_size=128)
    backend.init_cuda_graph_state(8)
    return backend, config, spec, kernels


def test_longcat_uses_registered_dsa_backend(indexed_backend):
    backend, config, spec, _ = indexed_backend
    assert type(backend) is ascend_backend.AscendDSABackend
    assert backend.kernel_page_size == 128
    assert type(backend._dense_backend) is mla_backend.MLAAttnBackend
    assert backend.child_backends() == (backend._dense_backend,)
    assert backend.kpool_runtime is None
    assert backend._stream_fork is not None
    assert backend.dsa_selection_policy == (4, 1024)
    for name in ("dsa", "longcat_dsa", "trtllm_mla"):
        assert registry._resolve_full_attn_backend_name(None, spec, name) == "dsa"
    if current_platform().is_npu:
        for name in ("dsa", "longcat_dsa"):
            assert (
                registry._get_backend_cls(name, AttentionArch.MLA)
                is ascend_backend.AscendDSABackend
            )
        registered_type = ascend_backend.AscendDSABackend
    else:
        # Independent Lite MLA-DSA is an Ascend implementation. Do not let a
        # GPU process silently instantiate the generic paired DSA backend.
        with pytest.raises(ValueError, match="does not support"):
            registry._get_backend_cls("dsa", AttentionArch.MLA)
        registered_type = dsa_backend.DSABackend
    assert registry._get_backend_cls("dsa", AttentionArch.DSA) is registered_type
    pool = object()
    backend.set_cache_pool(pool)
    assert backend.cache_pool is pool and backend._dense_backend.cache_pool is pool


def test_ascend_backend_owns_projection_branch_scheduling(indexed_backend, monkeypatch):
    backend, _, _, _ = indexed_backend
    calls = []
    monkeypatch.setattr(ascend_backend, "get_is_cuda_graph_phase", lambda: False)

    result = backend.run_projection_branches(
        None,
        lambda: calls.append("indexer") or "index-result",
        lambda: calls.append("mla") or "mla-result",
    )

    assert calls == ["indexer", "mla"]
    assert result == ("index-result", "mla-result")


def test_ascend_dsa_rejects_unimplemented_forward_variants(indexed_backend):
    backend, _, _, _ = indexed_backend
    unsupported = (
        "forward_extend_chunked",
        "forward_sparse_prefill",
        "forward_sparse_decode",
    )
    for name in unsupported:
        assert name in ascend_backend.AscendDSABackend.__dict__
        with pytest.raises(NotImplementedError, match="Ascend DSA does not"):
            getattr(backend, name)()


def test_longcat_dcp_reuses_one_auxiliary_communicator(indexed_backend, monkeypatch):
    _, config, spec, _ = indexed_backend
    dcp_group = tuple(range(8))
    config = dataclasses.replace(
        config,
        dcp_size=8,
        dcp_rank=0,
        dcp_group=dcp_group,
    )
    primary_group = object()
    auxiliary_group = object()
    created = []

    monkeypatch.setattr(ascend_backend.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(ascend_backend.dist, "get_world_size", lambda: 16)

    def get_dedicated_device_group(group, namespace):
        created.append((group, namespace))
        return auxiliary_group

    monkeypatch.setattr(
        ascend_backend.pg_manager,
        "get_dedicated_device_group",
        get_dedicated_device_group,
    )
    monkeypatch.setattr(
        ascend_backend.pg_manager,
        "get_device_process_group",
        lambda group: primary_group,
    )

    backend = ascend_backend.AscendDSABackend(config, spec, kernel_page_size=128)

    assert created == [(dcp_group, "dsa_cp_aux")]
    assert backend._dcp.primary_process_group is primary_group
    assert backend._dcp.auxiliary_process_group is auxiliary_group
    assert not hasattr(backend, "_dcp_query_process_group")
    assert not hasattr(backend, "_dcp_lse_process_group")


@pytest.mark.parametrize("bs", [1, 2, 8])
def test_bf16_dsa_refresh_is_pointer_stable_without_fp8_plan(indexed_backend, bs):
    backend, _, _, _ = indexed_backend
    table = torch.ones(bs, backend.max_num_pages, dtype=torch.int32)
    lengths = torch.arange(1, bs + 1, dtype=torch.int32)
    backend.refresh_decode_metadata(bs, bs, lengths, table, for_graph_replay=False)
    metadata = backend.forward_decode_metadata
    pointers = metadata.seq_lens.data_ptr(), metadata.page_table.data_ptr()
    for replay in (True, False, True):
        lengths += 1
        table += 1
        backend.refresh_decode_metadata(bs, bs, lengths, table, for_graph_replay=replay)
        assert backend.forward_decode_metadata is metadata
        assert (
            metadata.seq_lens.data_ptr(),
            metadata.page_table.data_ptr(),
        ) == pointers
        assert torch.equal(metadata.seq_lens, lengths)
        assert torch.equal(metadata.page_table, table)
        assert not hasattr(metadata, "_dsa_plan")


@pytest.mark.parametrize("rank", [0, 1, 2])
def test_longcat_dcp_metadata_is_owned_by_backend(indexed_backend, rank):
    _, config, spec, kernels = indexed_backend
    config = dataclasses.replace(
        config,
        context_len=512,
        dcp_size=8,
        dcp_rank=rank,
        dcp_group=tuple(range(8)),
    )
    backend = ascend_backend.AscendDSABackend(config, spec, kernel_page_size=128)
    backend._dcp.bind_virtual_block_count(64)
    backend.init_cuda_graph_state(2)
    table = torch.zeros(2, backend.max_num_pages, dtype=torch.int32)
    table[0, :2] = torch.tensor([1, 2])
    table[1, :3] = torch.tensor([9, 10, 11])
    lengths = torch.tensor([129, 257], dtype=torch.int32)

    backend.refresh_decode_metadata(2, 2, lengths, table)
    metadata = backend.forward_decode_metadata
    expected = {
        0: ([128, 128], [[1], [9]], [4, 4], [128, 128]),
        1: ([1, 128], [[2], [10]], [0, 0], [1, 128]),
        2: ([0, 1], [[], [11]], [0, 0], [0, 1]),
    }
    expected_lengths, expected_pages, expected_init, expected_local = expected[rank]
    assert backend._dcp.seq_lens.tolist() == expected_lengths
    for row, pages in zip(backend._dcp.page_table, expected_pages):
        assert row[: len(pages)].tolist() == pages
        assert not row[len(pages) :].any()
    assert backend._dcp.init_counts.tolist() == expected_init
    assert backend._dcp.local_counts.tolist() == expected_local
    dcp_pointers = (
        backend._dcp.page_table.data_ptr(),
        backend._dcp.seq_lens.data_ptr(),
        backend._dcp.init_counts.data_ptr(),
        backend._dcp.local_counts.data_ptr(),
    )
    pointers = metadata.seq_lens.data_ptr(), metadata.page_table.data_ptr()
    backend.refresh_decode_metadata(2, 2, lengths, table, for_graph_replay=True)
    assert pointers == (
        metadata.seq_lens.data_ptr(),
        metadata.page_table.data_ptr(),
    )
    assert dcp_pointers == (
        backend._dcp.page_table.data_ptr(),
        backend._dcp.seq_lens.data_ptr(),
        backend._dcp.init_counts.data_ptr(),
        backend._dcp.local_counts.data_ptr(),
    )


@pytest.mark.parametrize("prefix", [0, 128])
@pytest.mark.parametrize("mode", [ForwardMode.EXTEND, ForwardMode.MIXED])
def test_bf16_dsa_prefill_retains_full_history_table(indexed_backend, prefix, mode):
    backend, _, _, _ = indexed_backend
    extends = torch.tensor([3, 7], dtype=torch.int32)
    prefixes = torch.tensor([prefix, prefix], dtype=torch.int32)
    lengths = extends + prefixes
    bs = 2
    if mode == ForwardMode.MIXED:
        lengths = torch.cat((lengths, torch.tensor([200], dtype=torch.int32)))
        bs = 3
    table = torch.arange(bs * backend.max_num_pages, dtype=torch.int32).view(bs, -1)
    backend.init_forward_metadata(
        bs,
        2,
        lengths,
        table,
        mode,
        extend_seq_lens=extends,
        extend_seq_lens_cpu=extends,
        extend_prefix_lens=prefixes,
        extend_prefix_lens_cpu=prefixes,
        extend_with_prefix=bool(prefix),
    )
    metadata = backend.forward_prefill_metadata
    assert metadata is backend.chunked_prefill_metadata
    assert torch.equal(metadata.page_table, table[:2])
    assert metadata.page_table.data_ptr() == table.data_ptr()
    assert metadata.cum_extend_seq_lens.tolist() == [0, 3, 10]
    assert torch.equal(metadata.seq_lens, lengths[:2])
    if mode == ForwardMode.MIXED:
        assert backend.forward_decode_metadata.num_extends == 2
        assert not hasattr(backend.forward_decode_metadata, "_dsa_plan")


@pytest.mark.parametrize("prefill", [False, True])
def test_bf16_dsa_writes_packed_kv_then_indexes_and_attends(indexed_backend, prefill):
    backend, _, spec, kernels = indexed_backend
    rows, bs = (5, 2) if prefill else (2, 2)
    q = torch.randn(rows, 2, 576, dtype=torch.bfloat16)
    k = torch.randn(rows, 1, 576, dtype=torch.bfloat16)
    locations = torch.arange(64, 64 + rows)
    planes = {
        "latent_kv": torch.zeros(3, 128, 1, 576, dtype=torch.bfloat16),
        "dsa_index_k": torch.zeros(3, 128, 1, 128, dtype=torch.bfloat16),
    }
    pool = SimpleNamespace(
        get_key_buffer=lambda layer_id: planes["latent_kv"],
        get_component=lambda layer_id, name: planes[name],
    )
    layer = SimpleNamespace(layer_id=3, scaling=0.125, logit_cap=0)
    projections = dict(
        index_query=torch.randn(rows, 16, 128, dtype=torch.bfloat16),
        index_key=torch.randn(rows, 1, 128, dtype=torch.bfloat16),
        index_weights=torch.randn(rows, 16, dtype=torch.bfloat16),
    )
    table = torch.ones(3, backend.max_num_pages, dtype=torch.int32)
    lengths = torch.tensor([111, 68, 75], dtype=torch.int32)
    # The first request is a prefill in a mixed batch. Decode must skip it.
    backend.refresh_decode_metadata(3, 3, lengths, table, num_extends=1)
    if prefill:
        backend._dense_backend.forward_prefill_metadata = SimpleNamespace(
            cum_extend_seq_lens=torch.tensor([0, 2, 5], dtype=torch.int32),
            seq_lens=lengths[1:],
            page_table=table[1:],
        )
    expected = torch.randn(rows, 2, 512, dtype=torch.bfloat16)
    indices = torch.zeros(rows, 1, spec.index_topk, dtype=torch.int32)
    chunks = torch.full((rows,), 16, dtype=torch.int32)
    kernels.index.return_value = indices, chunks
    kernels.attention.return_value = expected
    backend.register_step_counter(SimpleNamespace(record_cache=kernels.ready))
    forward = backend.forward_extend if prefill else backend.forward_decode
    actual = forward(q, k, None, layer, locations, pool, bs, **projections)
    assert torch.equal(actual, expected.flatten(1))
    assert [c[0] for c in kernels.mock_calls] == [
        "scatter",
        "scatter",
        "ready",
        "index",
        "attention",
    ]
    for call, name, payload in zip(
        kernels.scatter.call_args_list,
        ("latent_kv", "dsa_index_k"),
        (k, projections["index_key"]),
    ):
        src, cache, loc = call.args
        assert src.is_contiguous() and torch.equal(src, payload)
        assert cache.data_ptr() == planes[name].data_ptr()
        assert loc is locations
    selected = kernels.index.call_args.args
    assert selected[0] is projections["index_query"]
    assert selected[2] is projections["index_weights"]
    assert selected[3].tolist() == ([2, 5] if prefill else [1, 2])
    assert selected[4].tolist() == [68, 75]
    assert torch.equal(selected[5], table[1:])
    assert selected[6:] == (
        spec.index_topk,
        spec.index_init_tokens,
        spec.index_local_tokens,
    )
    attended = kernels.attention.call_args.args
    assert torch.equal(attended[0], q[..., : spec.kv_lora_rank])
    assert torch.equal(attended[1], q[..., spec.kv_lora_rank :])
    assert attended[-1] == layer.scaling


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"kv_cache_dtype": torch.float32}, "BF16 KV"),
        ({"speculative_num_draft_tokens": 2}, "MTP"),
        ({"is_draft": True}, "MTP"),
        ({"kv_cache_quant_method": "per_token_head"}, "unquantized"),
    ],
)
def test_bf16_dsa_rejects_unsupported_contracts(indexed_backend, changes, match):
    _, config, spec, _ = indexed_backend
    with pytest.raises(ValueError, match=match):
        ascend_backend.AscendDSABackend(
            dataclasses.replace(config, **changes), spec, kernel_page_size=128
        )


def test_existing_dsa_keeps_gpu_plan_and_rejects_unadapted_npu_layout(
    indexed_backend, monkeypatch
):
    _, config, spec, _ = indexed_backend
    values = dataclasses.asdict(spec)
    values.update(
        uses_independent_index_cache=False,
    )
    gpu_spec = DSAConfig(**values)
    with pytest.raises(NotImplementedError, match="BF16 indexer/cache contract"):
        ascend_backend.AscendDSABackend(config, gpu_spec, kernel_page_size=64)
    monkeypatch.setattr(
        dsa_backend,
        "current_platform",
        lambda: SimpleNamespace(is_nvidia=False, is_amd=True, is_npu=False),
    )
    plan = Mock(return_value=torch.zeros(1))
    monkeypatch.setattr(dsa_backend, "dsa_plan", plan)
    backend = dsa_backend.DSABackend(config, gpu_spec, kernel_page_size=64)
    assert backend.dsa_selection_policy == (4, 1024)
    backend.init_cuda_graph_state(2)
    table = torch.ones(2, backend.max_num_pages, dtype=torch.int32)
    lengths = torch.tensor([7, 9], dtype=torch.int32)
    for _ in range(2):
        backend.refresh_decode_metadata(2, 2, lengths, table)
    assert plan.call_count == 2
    assert backend.forward_decode_metadata._dsa_plan is plan.return_value


@pytest.mark.skipif(
    os.environ.get("TOKENSPEED_TEST_LONGCAT_DSA_NPU") != "1",
    reason="requires an allocated NPU and matching sparse-attention operators",
)
@pytest.mark.parametrize("mode", ["decode", "prefill", "cached_prefill"])
def test_npu_dsa_matches_explicit_paged_pipeline(mode, monkeypatch):
    """Compare the shared backend with the original explicit operator sequence."""
    import torch_npu

    from tokenspeed.runtime.utils.env import global_server_args_dict

    monkeypatch.setitem(global_server_args_dict, "chunked_prefill_size", 4096)
    torch.npu.set_device(0)
    config = kimi_recipe(kv_cache_dtype=torch.bfloat16, max_bs=2).attn_config
    spec = _lite_dsa_config(
        **dataclasses.asdict(config.component(MLAConfig)),
        index_n_heads=16,
        index_head_dim=128,
        index_topk=2048,
        index_init_tokens=4,
        index_local_tokens=32,
    )
    spec = dataclasses.replace(spec, backend_name="dsa")
    config = dataclasses.replace(
        config, device="npu:0", context_len=256, components=(spec,)
    )
    backend = ascend_backend.AscendDSABackend(config, spec, kernel_page_size=128)
    backend.init_cuda_graph_state(2)
    kernels = backend._indexer_kernels
    counts = [1, 1] if mode == "decode" else [3, 5]
    prefixes = [64, 128] if mode != "prefill" else [0, 0]
    lengths_cpu = torch.tensor(
        [p + n for p, n in zip(prefixes, counts)], dtype=torch.int32
    )
    lengths = lengths_cpu.npu()
    q_ends = torch.tensor(counts, dtype=torch.int32).cumsum(0).to(torch.int32).npu()
    table = torch.tensor([[1, 2], [5, 6]], dtype=torch.int32).npu()
    slots = [
        (1 + request * 4) * 128 + position
        for request, (prefix, count) in enumerate(zip(prefixes, counts))
        for position in range(prefix, prefix + count)
    ]
    locations = torch.tensor(slots, dtype=torch.int64).npu()
    rows = sum(counts)
    generator = torch.Generator().manual_seed(317)

    def rand(shape):
        return (torch.randn(shape, generator=generator) * 0.1).to(torch.bfloat16).npu()

    q, k = rand((rows, 32, 576)), rand((rows, 1, 576))
    projections = dict(
        index_query=rand((rows, 16, 128)),
        index_key=rand((rows, 1, 128)),
        index_weights=rand((rows, 16)),
    )
    planes = {
        "latent_kv": rand((9, 128, 1, 576)),
        "dsa_index_k": rand((9, 128, 1, 128)),
    }
    reference_planes = {name: value.clone() for name, value in planes.items()}
    pool = SimpleNamespace(
        get_key_buffer=lambda layer_id: planes["latent_kv"],
        get_component=lambda layer_id, name: planes[name],
    )
    layer = SimpleNamespace(layer_id=0, scaling=192**-0.5, logit_cap=0)

    def reference():
        key, index_k = reference_planes.values()
        kernels.scatter(k.contiguous(), key, locations)
        kernels.scatter(projections["index_key"], index_k, locations)
        indices, valid_chunks = kernels.index(
            projections["index_query"],
            index_k,
            projections["index_weights"],
            q_ends,
            lengths,
            table,
            2048,
            4,
            32,
        )
        output = kernels.attention(
            q[..., :512].contiguous(),
            q[..., 512:].contiguous(),
            key,
            indices,
            valid_chunks,
            q_ends,
            lengths,
            table,
            layer.scaling,
        )
        return output.flatten(1)

    if mode == "decode":
        backend.refresh_decode_metadata(2, 2, lengths, table)
        forward = backend.forward_decode
    else:
        extends_cpu = torch.tensor(counts, dtype=torch.int32)
        prefix_cpu = torch.tensor(prefixes, dtype=torch.int32)
        backend.init_forward_metadata(
            2,
            2,
            lengths,
            table,
            ForwardMode.EXTEND,
            extend_seq_lens=extends_cpu.npu(),
            extend_seq_lens_cpu=extends_cpu,
            extend_prefix_lens=prefix_cpu.npu(),
            extend_prefix_lens_cpu=prefix_cpu,
            extend_with_prefix=bool(max(prefixes)),
        )
        forward = backend.forward_extend

    def run():
        return forward(q, k, None, layer, locations, pool, 2, **projections)

    expected = reference()
    for _ in range(3):
        actual = run()
    torch.npu.synchronize()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for name in planes:
        torch.testing.assert_close(planes[name], reference_planes[name], rtol=0, atol=0)
    if mode == "decode":
        graph = torch_npu.npu.NPUGraph()
        with torch_npu.npu.graph(graph):
            captured = run()
        # Same addresses, changing lengths and payload: replay must not use
        # captured Python-side lengths or an FP8 indexer's stale workspace.
        for _ in range(3):
            lengths.add_(1)
            q.add_(0.01)
            backend.refresh_decode_metadata(2, 2, lengths, table, for_graph_replay=True)
            expected = reference()
            graph.replay()
            torch.npu.synchronize()
            torch.testing.assert_close(captured, expected, rtol=0, atol=0)


def test_gpu_backend_implements_projection_branch_contract():
    backend = dsa_backend.DSABackend.__new__(dsa_backend.DSABackend)
    calls = []
    result = backend.run_projection_branches(
        None,
        lambda: calls.append("indexer") or "index-result",
        lambda: calls.append("mla") or "mla-result",
    )
    assert calls == ["indexer", "mla"]
    assert result == ("index-result", "mla-result")
