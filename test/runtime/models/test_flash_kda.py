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

import json
from types import SimpleNamespace
from unittest import mock

import pytest
import torch


def test_flash_kda_config_resolves_hybrid_layer_pattern() -> None:
    from tokenspeed.runtime.configs.flash_kda_config import FLASHLocalConfig

    config = FLASHLocalConfig(num_hidden_layers=8, fa_interval=4, use_mla=True)

    assert config.linear_layer_ids == [0, 1, 2, 4, 5, 6]
    assert config.full_attention_layer_ids == [3, 7]


def test_flash_kda_config_rejects_unimplemented_sigmoid_router() -> None:
    from tokenspeed.runtime.configs.flash_kda_config import FLASHLocalConfig

    with pytest.raises(ValueError, match="only supports softmax"):
        FLASHLocalConfig(moe_router_activation_func="sigmoid")


def test_get_config_loads_flash_lite_without_model_type(tmp_path) -> None:
    from tokenspeed.runtime.configs.flash_kda_config import FLASHLocalConfig
    from tokenspeed.runtime.utils.hf_transformers_utils import get_config

    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "architectures": ["FLASHLocalForCausalLM"],
                "vocab_size": 163840,
                "hidden_size": 3072,
                "num_layers": 28,
                "num_attention_heads": 32,
                "n_routed_experts": 384,
                "moe_topk": 12,
                "ngram_vocab_size_ratio": 28.8,
                "emb_split_num": 4,
                "emb_neighbor_num": 4,
                "ngram_exclude_sp_token": True,
                "special_token_scope": "0:4,36:55",
                "fa_interval": 4,
                "use_mla": 1,
            }
        )
    )

    config = get_config(str(tmp_path), trust_remote_code=False)

    assert isinstance(config, FLASHLocalConfig)
    assert config.num_hidden_layers == 28
    assert config.hidden_act == "silu"
    assert config.num_experts == 384
    assert config.num_experts_per_token == 12
    assert config.over_embedding_m == 4_718_592
    assert config.oe_ignore_tokens == list(range(4)) + list(range(36, 55))


def test_flash_kda_model_entry_is_registered() -> None:
    from tokenspeed.runtime.models.flash_kda import FLASHLocalForCausalLM
    from tokenspeed.runtime.models.registry import ModelRegistry

    model_cls, architecture = ModelRegistry.resolve_model_cls(["FLASHLocalForCausalLM"])

    assert model_cls is FLASHLocalForCausalLM
    assert architecture == "FLASHLocalForCausalLM"


def test_flash_kda_registers_hybrid_mla_kda_attention() -> None:
    from tokenspeed.runtime.configs import model_config
    from tokenspeed.runtime.layers.attention import registry

    architecture = "FLASHLocalForCausalLM"

    assert architecture in model_config._MLA_ARCHITECTURES
    assert architecture in registry._HYBRID_MLA_KDA_ARCHITECTURES


def test_fgbkda_backend_disables_incompatible_verify_replay(monkeypatch) -> None:
    from tokenspeed.runtime.layers.attention import registry
    from tokenspeed.runtime.layers.attention.backends.state import kda as hybrid_kda
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
        LINEAR_ATTENTION,
    )

    config = SimpleNamespace(
        device=torch.device("cpu"),
        num_attention_heads=2,
        num_kv_heads=2,
        attn_tp_size=1,
        dtype=torch.bfloat16,
        head_dim=8,
        is_draft=False,
        speculative_num_draft_tokens=1,
        max_bs=1,
    )
    text_config = SimpleNamespace(
        full_attention_layer_ids=(1,),
        mamba2_cache_params=(None, None, None, None, (0,)),
        linear_method="FGBKDA",
    )
    model_config = SimpleNamespace(hf_config=text_config, attention_arch=object())
    state_group = SimpleNamespace(
        group_id=LINEAR_ATTENTION,
        family="state",
        checkpoint_granularity=128,
    )
    pool = SimpleNamespace(
        arena=SimpleNamespace(
            runtime_contract=SimpleNamespace(group_specs=(state_group,))
        ),
        state_group_by_layer={0: LINEAR_ATTENTION},
        get_component=lambda _layer_id, _name: None,
    )
    monkeypatch.setattr(
        registry,
        "_create_attn_backend_with_name",
        lambda *_args, **_kwargs: SimpleNamespace(device=torch.device("cpu")),
    )
    monkeypatch.setattr(registry, "_resolve_kda_backend", lambda _name: "auto")
    monkeypatch.setattr(
        hybrid_kda, "kda_replay_commit_supported", lambda *_args, **_kwargs: True
    )

    backend = registry._create_hybrid_linear_attn_backend(
        SimpleNamespace(speculative_algorithm=None, kda_backend="auto"),
        model_config,
        config,
        pool=pool,
        is_kda=True,
    )

    assert not backend.linear_attn_backend._replay_active


def test_flash_kda_group_topk_uses_independent_global_expert_ranges() -> None:
    from tokenspeed.runtime.models.flash_kda import _group_moe_topk

    logits = torch.tensor(
        [
            [0.0, 2.0, 1.0],
            [2.0, 0.0, 1.0],
            [0.0, 1.0, 2.0],
            [1.0, 2.0, 0.0],
        ]
    )
    bias = torch.zeros(6)
    output = _group_moe_topk(
        logits,
        bias,
        top_k=1,
        moe_group_size=2,
        zero_expert_num=1,
        renormalize=False,
        routed_scaling_factor=2.0,
    )

    assert output.topk_ids.tolist() == [[1], [0], [-1], [3]]
    expected = logits.softmax(dim=-1).amax(dim=-1, keepdim=True) * 2.0
    torch.testing.assert_close(output.topk_weights, expected)


def test_flash_kda_loads_single_group_router_bias_for_every_group() -> None:
    from tokenspeed.runtime.models.flash_kda import (
        _load_flash_kda_router_correction_bias,
    )

    moe = SimpleNamespace(
        moe_group_size=2,
        n_routed_experts=2,
        zero_expert_num=1,
        router=SimpleNamespace(
            e_score_correction_bias=torch.nn.Parameter(torch.zeros(6))
        ),
    )

    _load_flash_kda_router_correction_bias(moe, torch.tensor([1.0, 2.0, 3.0]))

    torch.testing.assert_close(
        moe.router.e_score_correction_bias,
        torch.tensor([1.0, 2.0, 3.0, 1.0, 2.0, 3.0]),
    )


def test_flash_kda_quant_ignore_checks_experts_in_every_group() -> None:
    from tokenspeed.runtime.models.flash_kda import (
        _get_flash_kda_moe_quant_config,
    )

    config = SimpleNamespace(n_routed_experts=2, moe_group_size=2)
    quant_config = SimpleNamespace(
        ignored_layers=[
            f"model.layers.0.moe.experts.{expert_id}.{proj_name}"
            for expert_id in (2, 3)
            for proj_name in ("gate_proj", "up_proj", "down_proj")
        ]
    )

    with pytest.raises(ValueError, match="partially ignored"):
        _get_flash_kda_moe_quant_config(
            config,
            quant_config,
            "model.layers.0.moe",
        )


def test_flash_kda_identity_zero_expert_is_partitioned_across_moe_ranks() -> None:
    from tokenspeed.runtime.models.flash_kda import FLASHLocalMoE

    moe = object.__new__(FLASHLocalMoE)
    moe.__dict__.update(
        zero_expert_num=1,
        zero_expert_type="identity",
        mapping=SimpleNamespace(moe=SimpleNamespace(tp_ep_size=4)),
    )
    hidden_states = torch.tensor([[2.0, 4.0], [6.0, 8.0]])
    topk_output = SimpleNamespace(
        topk_ids=torch.tensor([[-1, 3], [2, -1]], dtype=torch.int32),
        topk_weights=torch.tensor([[0.5, 0.25], [0.75, 0.125]]),
    )

    output = moe._apply_zero_experts(hidden_states, topk_output)

    expected = hidden_states * torch.tensor([[0.5], [0.125]]) / 4
    torch.testing.assert_close(output, expected)
    assert topk_output.topk_ids.tolist() == [[0, 3], [2, 0]]
    assert topk_output.topk_weights.tolist() == [[0.0, 0.25], [0.75, 0.0]]


def test_flash_local_decoder_selects_packed_moe_by_capability(monkeypatch) -> None:
    from test.runtime.test_lite_model_loader import lite_config_dict

    from tokenspeed.runtime.configs.flash_kda_config import FLASHLocalConfig
    from tokenspeed.runtime.distributed.mapping import Mapping
    from tokenspeed.runtime.models import flash_kda
    from tokenspeed.runtime.models.flash_local_moe import PackedFLASHLocalMoE

    monkeypatch.setattr(flash_kda, "flash_local_prefers_packed_moe", lambda: True)
    config = FLASHLocalConfig.from_dict(lite_config_dict())
    mapping = Mapping(rank=0, world_size=1)

    with torch.device("meta"):
        layer = flash_kda.FLASHLocalDecoderLayer(config, 0, mapping)

    assert isinstance(layer.moe, PackedFLASHLocalMoE)
    assert layer.moe.local_expert_ids == tuple(range(32))


def test_flash_kda_maps_fgbkda_projection_weights_to_checkpoint_structure() -> None:
    from tokenspeed.runtime.models.flash_kda import (
        _canonical_flash_kda_weight_name,
    )

    assert (
        _canonical_flash_kda_weight_name(
            "model.layers.0.self_attn.linear_core.g_proj.0.weight"
        )
        == "model.layers.0.self_attn.g_proj.weight"
    )
    assert (
        _canonical_flash_kda_weight_name(
            "model.layers.0.self_attn.linear_core.b_proj.1.weight"
        )
        == "model.layers.0.self_attn.b_proj.fc2.weight"
    )
    assert (
        _canonical_flash_kda_weight_name(
            "language_model.embedding.word_embeddings.weight"
        )
        == "model.embed_tokens.weight"
    )
    assert (
        _canonical_flash_kda_weight_name(
            "language_model.decoder.layers.3.input_layernorm.m.weight"
        )
        == "model.layers.3.input_layernorm.weight"
    )
    assert (
        _canonical_flash_kda_weight_name(
            "language_model.encoder.layers.4.self_attention.linear_core.q_proj.weight"
        )
        == "model.layers.4.self_attn.q_proj.weight"
    )


def test_separate_kda_geometry_uses_linear_attention_mapping() -> None:
    from tokenspeed.runtime.configs.flash_kda_config import FLASHLocalConfig
    from tokenspeed.runtime.models.flash_kda import SeperateFLASHLocal

    config = FLASHLocalConfig(
        hidden_size=16,
        num_attention_heads=4,
        qk_nope_head_dim=4,
        linear_head_dim=4,
        linear_num_heads=4,
        num_hidden_layers=4,
        fa_interval=4,
    )
    mapping = SimpleNamespace(
        attn=SimpleNamespace(tp_rank=0, tp_size=1, tp_group=(0,)),
        linear_attn=SimpleNamespace(tp_rank=1, tp_size=2, tp_group=(0, 1)),
    )

    layer = SeperateFLASHLocal(config, mapping, layer_id=0)

    assert layer.local_num_heads == 2
    assert layer.q_proj.weight.shape == (8, 16)
    assert layer.A_log.shape == (2,)


def test_flash_lite_cache_allows_its_wider_recurrent_state() -> None:
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.kimi_k3 import (
        KimiK3Recipe,
    )

    def padding_bound(model_type: str) -> float:
        recipe = object.__new__(KimiK3Recipe)
        recipe.draft_attn_config = None
        recipe.model_config = SimpleNamespace(
            hf_text_config=SimpleNamespace(model_type=model_type)
        )
        return recipe.max_padding_fraction

    assert padding_bound("flash_kda") == 0.75
    assert padding_bound("kimi_k3") == 0.25


@pytest.mark.parametrize("exclude_special_tokens", [False, True])
def test_flash_lite_oe_keeps_its_special_token_policy(
    exclude_special_tokens: bool,
) -> None:
    from tokenspeed.runtime.models.flash_kda import FLASHLocalModel

    model = object.__new__(FLASHLocalModel)
    model.mapping = SimpleNamespace(
        attn=SimpleNamespace(tp_rank=0, tp_size=4, tp_group=(0, 1, 2, 3))
    )
    config = SimpleNamespace(
        use_over_embedding=True,
        vocab_size=163840,
        hidden_size=3072,
        over_embedding_m=4718592,
        emb_neighbor_num=4,
        emb_split_num=4,
        oe_ignore_tokens=[2, 3],
        ngram_exclude_sp_token=exclude_special_tokens,
        ngram_fix_normalize_factor=True,
        eos_token_id=2,
    )

    with mock.patch(
        "tokenspeed.runtime.models.flash_kda.LongCatOverEmbedding"
    ) as over_embedding:
        model._build_embed_tokens(
            config, quant_config=None, oe_table_placement="device"
        )

    kwargs = over_embedding.call_args.kwargs
    assert kwargs["ignored_token_ids"] == ((2, 3) if exclude_special_tokens else ())
    assert kwargs["segment_ignored_tokens"] is exclude_special_tokens


class _CaptureFinalNorm:
    def final_norm(self, hidden_states, residual, ctx, norm):
        return hidden_states + residual + 1000, None


class _CaptureLayer(torch.nn.Module):
    def __init__(
        self,
        hidden_delta: int,
        residual_value: int,
        captured_value: int,
    ) -> None:
        super().__init__()
        self.hidden_delta = hidden_delta
        self.residual_value = residual_value
        self.captured_value = captured_value
        self.moe_comm = _CaptureFinalNorm()

    def forward(
        self,
        positions,
        hidden_states,
        ctx,
        residual,
        capture_hidden_state=None,
    ):
        if capture_hidden_state is not None:
            capture_hidden_state(torch.full_like(hidden_states, self.captured_value))
        return (
            hidden_states + self.hidden_delta,
            torch.full_like(hidden_states, self.residual_value),
        )


def test_flash_lite_eagle3_captures_materialized_completed_layer_residual() -> None:
    from tokenspeed.runtime.models.flash_kda import FLASHLocalModel

    model = object.__new__(FLASHLocalModel)
    torch.nn.Module.__init__(model)
    model.layers = torch.nn.ModuleList(
        [
            _CaptureLayer(1, 10, 110),
            _CaptureLayer(1, 20, 120),
        ]
    )
    model.layers_to_capture = {1}
    model.norm = torch.nn.Identity()
    ctx = SimpleNamespace(forward_mode=SimpleNamespace(is_idle=lambda: False))

    output, captures = model(
        input_ids=torch.empty(1, dtype=torch.int64),
        positions=torch.empty(1, dtype=torch.int64),
        ctx=ctx,
        input_embeds=torch.ones(1, 1),
    )

    assert captures is not None
    torch.testing.assert_close(captures[0], torch.tensor([[120.0]]))
    torch.testing.assert_close(output, torch.tensor([[1023.0]]))
