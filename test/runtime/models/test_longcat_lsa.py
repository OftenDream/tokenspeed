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

"""LongCat LSA ownership, Indexer, and checkpoint-loading contracts."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tokenspeed.runtime.configs.model_config import AttentionArch
from tokenspeed.runtime.layers.attention import registry
from tokenspeed.runtime.layers.layernorm import RMSNorm
from tokenspeed.runtime.models.longcat_dsa import (
    LongCatDSAAttention,
    LongCatDSAIndexer,
    LongCatDSAIndexerWeightLoaderMixin,
    LongCatDSASelection,
)


def test_sparse_decode_uses_backend_owned_write_locations() -> None:
    query = torch.ones(1, 1, 2)
    locations = torch.tensor([7], dtype=torch.int32)
    topk = torch.tensor([[0]], dtype=torch.int32)
    lengths = torch.tensor([1], dtype=torch.int32)
    ctx = SimpleNamespace()
    calls = []

    def attention(q, k, v, context, save_kv_cache=True, **kwargs):
        calls.append((context, save_kv_cache, kwargs))
        return q

    def project(q, latent, positions, context, write_locations):
        assert write_locations is locations
        return q, latent

    layer = SimpleNamespace(
        forward_absorb_qkv_proj=project,
        attention_backend="mla",
        _MLA_KERNEL_BACKENDS=("mla",),
        attn_mqa=attention,
        kv_lora_rank=2,
        num_local_heads=1,
        v_head_dim=2,
        w_vc=torch.eye(2).unsqueeze(0),
    )
    output = torch.empty(1, 2)
    LongCatDSAAttention._forward_sparse_decode(
        layer, torch.tensor([0]), query, query, ctx, locations, output, topk, lengths
    )
    torch.testing.assert_close(output, torch.ones_like(output))
    assert calls == [(ctx, False, {"topk_indices": topk, "topk_lens": lengths})]


def test_dsa_outer_backend_wraps_explicit_dense_backend(monkeypatch) -> None:
    spec = SimpleNamespace(
        backend_name="trtllm_mla",
        indexer_layer_ids=frozenset({0}),
    )
    config = SimpleNamespace(component=lambda _cls: spec)
    selections = []

    class FakeDSABackend:
        def __init__(self, received_config) -> None:
            self.config = received_config

    def create_backend(name, arch, received_config):
        selections.append((name, arch))
        return FakeDSABackend(received_config)

    monkeypatch.setattr(registry, "_create_attn_backend_with_name", create_backend)

    backend = registry._create_attn_backend(AttentionArch.DSA, config)

    assert selections == [("dsa", AttentionArch.DSA)]
    assert backend.config is config
    assert spec.backend_name == "trtllm_mla"

    unrelated_spec = SimpleNamespace(
        backend_name="trtllm_mla",
        indexer_layer_ids=None,
    )
    unrelated_config = SimpleNamespace(component=lambda _cls: unrelated_spec)
    registry._create_attn_backend(AttentionArch.DSA, unrelated_config)
    assert selections[-1] == ("trtllm_mla", AttentionArch.DSA)


def _indexer() -> LongCatDSAIndexer:
    return LongCatDSAIndexer(
        config=SimpleNamespace(
            index_topk=2048,
            index_n_heads=32,
            index_head_dim=128,
        ),
        hidden_size=16,
        q_lora_rank=8,
        qk_rope_head_dim=4,
        rope_theta=10_000,
        rope_scaling=None,
        max_position_embeddings=128,
        quant_config=None,
    )


def test_selection_is_consumed_only_by_the_paired_physical_layer() -> None:
    selection = LongCatDSASelection(owner_layer_id=4)

    selection.require_consumer(5)

    with pytest.raises(RuntimeError, match=r"owner layer 4.*consumer layer 7"):
        selection.require_consumer(7)


def test_indexer_uses_rms_norm_and_interleaved_rope() -> None:
    indexer = _indexer()

    assert isinstance(indexer.k_norm, RMSNorm)
    assert not hasattr(indexer.k_norm, "bias")
    assert indexer.rotary_emb.is_neox_style is False


def test_sparse_decode_accepts_lite_verify_width_and_rejects_larger_width() -> None:
    LongCatDSAAttention.check_decode_width(1)
    LongCatDSAAttention.check_decode_width(8)

    with pytest.raises(NotImplementedError, match=r"1-8.*got 9"):
        LongCatDSAAttention.check_decode_width(9)


def test_separate_indexer_weights_load_into_packed_projection() -> None:
    module_name = "model.layers.0.self_attn.0.indexer"
    packed_weight = nn.Parameter(torch.zeros(5, 4))

    def load_shard(param, loaded_weight, shard_id=None):
        if shard_id == 0:
            param.data[:4].copy_(loaded_weight)
        elif shard_id == 1:
            param.data[4:].copy_(loaded_weight)
        else:
            param.data.copy_(loaded_weight)

    packed_weight.weight_loader = load_shard

    class IndexerOwner(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.packed_projection_loaded = False

        def set_packed_projection_loaded(self) -> None:
            self.packed_projection_loaded = True

    owner = IndexerOwner()
    loader = LongCatDSAIndexerWeightLoaderMixin()
    params = {f"{module_name}.wk_weights_proj.weight": packed_weight}
    modules = {module_name: owner}
    loaded_shards = {}

    assert loader.try_load_indexer_projection(
        name=f"{module_name}.wk.weight",
        loaded_weight=torch.full((4, 4), 2.0),
        params=params,
        modules=modules,
        pending_fp8={},
        loaded_shards=loaded_shards,
        weight_block_size=None,
    )
    assert loader.try_load_indexer_projection(
        name=f"{module_name}.weights_proj.weight",
        loaded_weight=torch.full((1, 4), 3.0),
        params=params,
        modules=modules,
        pending_fp8={},
        loaded_shards=loaded_shards,
        weight_block_size=None,
    )
    loader.validate_indexer_projections(
        modules=modules,
        pending_fp8={},
        loaded_shards=loaded_shards,
    )

    torch.testing.assert_close(packed_weight[:4], torch.full((4, 4), 2.0))
    torch.testing.assert_close(packed_weight[4:], torch.full((1, 4), 3.0))
    assert owner.packed_projection_loaded


def test_missing_indexer_projection_shard_is_rejected() -> None:
    module_name = "model.layers.0.self_attn.0.indexer"

    class IndexerOwner(nn.Module):
        def set_packed_projection_loaded(self) -> None:
            pass

    with pytest.raises(RuntimeError, match=r"packed projections.*incomplete"):
        LongCatDSAIndexerWeightLoaderMixin().validate_indexer_projections(
            modules={module_name: IndexerOwner()},
            pending_fp8={},
            loaded_shards={module_name: {0}},
        )
