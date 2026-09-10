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

import importlib
import json
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from tokenspeed.runtime.configs.flash_kda_config import FLASHLocalConfig
from tokenspeed.runtime.models.flash_kda import (
    FLASHLocalForCausalLM,
)
from tokenspeed.runtime.models.flash_local_attention import (
    PackedFLASHLocalKDA,
    SeparateProjectionKimiLinearMLAAttention,
)
from tokenspeed.runtime.models.flash_local_checkpoint import (
    FLASHLocalCheckpointLayout,
)


def lite_config_dict(**overrides):
    config = {
        "architectures": ["FLASHLocalForCausalLM"],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "vocab_size": 120,
        "hidden_size": 96,
        "ffn_hidden_size": 96,
        "expert_ffn_hidden_size": 16,
        "num_layers": 4,
        "num_attention_heads": 8,
        "kv_lora_rank": 12,
        "q_lora_rank": 24,
        "qk_rope_head_dim": 2,
        "v_head_dim": 4,
        "qk_nope_head_dim": 4,
        "mla_scale_q_lora": True,
        "mla_scale_kv_lora": True,
        "routed_scaling_factor": 6.0,
        "n_routed_experts": 8,
        "max_position_embeddings": 128,
        "rms_norm_eps": 1e-5,
        "use_cache": True,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "zero_expert_num": 4,
        "zero_expert_type": "identity",
        "moe_topk": 2,
        "moe_switch_token_num": 16,
        "moe_impl": "mix",
        "gmoe_strategy": "gmoe_aware",
        "ngram_vocab_size_ratio": 0.1,
        "emb_neighbor_num": 5,
        "emb_split_num": 4,
        "ngram_exclude_sp_token": True,
        "special_token_scope": "0:4,36:55",
        "ngram_fix_normalize_factor": True,
        "moe_group_size": 4,
        "grouped_moe_norm_scale": 2,
        "use_mla": 1,
        "attention_method": "MLA",
        "fa_interval": 4,
        "linear_method": "FGBKDA",
        "kda_nope": True,
        "linear_hidden_size": 96,
        "linear_head_dim": 4,
        "linear_num_heads": 8,
        "linear_conv_size": 4,
        "mla_use_output_gate": True,
        "kda_use_full_rank_gate": True,
    }
    config.update(overrides)
    return config


def mapping(world_size=1, rank=0, role="prefill"):
    size = world_size
    replicated = role == "replicated"
    dense_size = size if role in {"decode", "replicated"} else 1
    world_group = tuple(range(world_size))
    return SimpleNamespace(
        world_size=world_size,
        rank=rank,
        pp_size=1,
        is_first_pp_rank=True,
        is_last_pp_rank=True,
        dense=SimpleNamespace(
            tp_size=dense_size,
            tp_rank=rank if role in {"decode", "replicated"} else 0,
            tp_group=(world_group if role in {"decode", "replicated"} else (rank,)),
            has_dp=dense_size < world_size,
        ),
        linear_attn=SimpleNamespace(tp_size=size, tp_rank=rank, tp_group=world_group),
        mla_weight=SimpleNamespace(tp_size=1, tp_rank=0, tp_group=(rank,)),
        moe=SimpleNamespace(
            tp_size=1, ep_size=size, ep_rank=rank, ep_group=world_group
        ),
        attn=SimpleNamespace(
            tp_size=size if replicated else 1,
            tp_rank=rank if replicated else 0,
            tp_group=world_group if replicated else (rank,),
            cp_size=size if role == "prefill" else 1,
            dp_size=size if role == "decode" else 1,
            cp_rank=rank if role == "prefill" else 0,
            dp_rank=rank if role == "decode" else 0,
            cp_group=world_group if role == "prefill" else (rank,),
            dp_group=world_group if role == "decode" else (rank,),
            has_dp=role == "decode",
        ),
    )


def weights(layout, mutate=None):
    for index, name in enumerate(layout.iter_source_names()):
        spec = layout.spec(name)
        tensor = torch.zeros(spec.shape, dtype=spec.dtype)
        if mutate is not None:
            tensor = mutate(index, name, tensor)
        yield name, tensor


def no_accelerator_error(exc):
    while exc is not None:
        if "requires an NVIDIA CUDA, AMD ROCm, or Ascend NPU device" in str(exc):
            return True
        exc = exc.__cause__
    return False


def test_lite_config_derives_hybrid_and_oe_geometry():
    config = FLASHLocalConfig.from_dict(lite_config_dict())

    assert config.linear_layer_ids == [0, 1, 2]
    assert config.full_attention_layer_ids == [3]
    assert config.layers_block_type == [
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "attention",
    ]
    assert config.oe_component_count == 16
    assert config.oe_hidden_size == 6
    assert config.oe_table_rows(0) == 13
    assert config.oe_table_rows(15) == 43
    assert config.special_token_ids == tuple(range(4)) + tuple(range(36, 55))


def test_lite_strict_layout_describes_w8a8_expert_sidecars():
    config = FLASHLocalConfig.from_dict(lite_config_dict())
    layout = FLASHLocalCheckpointLayout(
        config,
        moe_quant_kind="int8",
        moe_smooth_quant=True,
    )
    prefix = "model.layers.0.mlp.experts.0"
    names = set(layout.iter_source_names())

    assert f"{prefix}.gate_proj.weight_scale" in names
    assert f"{prefix}.down_proj.smooth_scale" in names
    gate = layout.spec(f"{prefix}.gate_proj.weight")
    gate_scale = layout.spec(f"{prefix}.gate_proj.weight_scale")
    gate_smooth = layout.spec(f"{prefix}.gate_proj.smooth_scale")
    down_scale = layout.spec(f"{prefix}.down_proj.weight_scale")
    down_smooth = layout.spec(f"{prefix}.down_proj.smooth_scale")

    assert gate.shape == (16, 24)
    assert gate.dtype == torch.int8
    assert gate_scale.shape == (16, 1)
    assert gate_scale.target_name.endswith("experts.w13_weight_scale")
    assert gate_smooth.shape == (24,)
    assert gate_smooth.target_name.endswith("experts.w13_smooth_scale")
    assert down_scale.shape == (24, 1)
    assert down_scale.target_name.endswith("experts.w2_weight_scale")
    assert down_smooth.shape == (16,)
    assert down_smooth.target_name.endswith("experts.w2_smooth_scale")


def test_lite_config_exposes_linear_tp_cache_geometry():
    import tokenspeed.runtime.utils.env as env_mod

    config = FLASHLocalConfig.from_dict(lite_config_dict())
    with mock.patch.dict(
        env_mod.global_server_args_dict,
        {"mapping": mapping(8)},
    ):
        conv, recurrent, conv_dtype, state_dtype, layer_ids = config.mamba2_cache_params

    assert conv == (12, 3)
    assert recurrent == (1, 4, 4)
    assert conv_dtype == torch.bfloat16
    assert state_dtype == torch.float32
    assert layer_ids == config.linear_layer_ids


def test_lite_config_fails_closed_on_raw_checkpoint_fields():
    raw = lite_config_dict()
    raw.pop("linear_method")
    with pytest.raises(ValueError, match="linear_method"):
        FLASHLocalConfig.from_dict(raw)

    with pytest.raises(ValueError, match="divisible by fa_interval"):
        FLASHLocalConfig.from_dict(lite_config_dict(num_layers=5))
    with pytest.raises(ValueError, match="full-rank"):
        FLASHLocalConfig.from_dict(lite_config_dict(kda_use_full_rank_gate=False))
    with pytest.raises(ValueError, match="architectures"):
        FLASHLocalConfig.from_dict(lite_config_dict(architectures=["LlamaForCausalLM"]))


def test_get_config_selects_lite_by_architecture_without_model_type(tmp_path):
    try:
        module = importlib.import_module(
            "tokenspeed.runtime.utils.hf_transformers_utils"
        )
    except RuntimeError as exc:
        if no_accelerator_error(exc):
            pytest.skip("Full runtime import requires an accelerator platform.")
        raise
    (tmp_path / "config.json").write_text(json.dumps(lite_config_dict()))

    config = module.get_config(str(tmp_path), trust_remote_code=False)

    assert isinstance(config, FLASHLocalConfig)
    assert config.architectures == ["FLASHLocalForCausalLM"]


def test_model_registry_resolves_real_lite_entry_class():
    try:
        module = importlib.import_module("tokenspeed.runtime.models.registry")
    except RuntimeError as exc:
        if no_accelerator_error(exc):
            pytest.skip("Full runtime import requires an accelerator platform.")
        raise

    model_class, architecture = module.ModelRegistry.resolve_model_cls(
        ["FLASHLocalForCausalLM"]
    )

    assert model_class is FLASHLocalForCausalLM
    assert architecture == "FLASHLocalForCausalLM"


def test_real_layout_has_exact_source_count_and_shapes():
    config = FLASHLocalConfig.from_dict(
        lite_config_dict(
            vocab_size=163840,
            hidden_size=4096,
            ffn_hidden_size=3072,
            expert_ffn_hidden_size=512,
            num_layers=28,
            num_attention_heads=32,
            kv_lora_rank=512,
            q_lora_rank=1536,
            qk_rope_head_dim=64,
            v_head_dim=128,
            qk_nope_head_dim=128,
            n_routed_experts=384,
            zero_expert_num=32,
            moe_topk=12,
            linear_hidden_size=4096,
            linear_head_dim=128,
            linear_num_heads=32,
            ngram_vocab_size_ratio=59.604,
        )
    )
    layout = FLASHLocalCheckpointLayout(config)
    names = list(layout.iter_source_names())

    assert len(names) == len(set(names)) == 129_878
    assert layout.spec(
        "model.layers.0.self_attn.linear_core.b_proj.1.weight"
    ).shape == (
        4096,
        128,
    )
    q_spec = layout.spec("model.layers.0.self_attn.linear_core.q_proj.weight")
    forget_spec = layout.spec("model.layers.0.self_attn.linear_core.f_proj.0.weight")
    beta_spec = layout.spec("model.layers.0.self_attn.linear_core.b_proj.0.weight")
    assert q_spec.target_name.endswith("self_attn.input_projection.weight")
    assert q_spec.category == "kda-packed-projection"
    assert (q_spec.component_id, q_spec.parallel, q_spec.shard_axis) == (
        0,
        "linear",
        0,
    )
    assert (forget_spec.component_id, forget_spec.parallel) == (4, None)
    assert (beta_spec.component_id, beta_spec.parallel) == (5, None)
    assert (
        layout.spec("model.layers.0.self_attn.linear_core.b_proj.1.weight").target_name
        == "model.layers.0.self_attn.beta_b_proj.weight"
    )
    assert layout.spec("model.layers.3.self_attn.kv_b_proj.weight").shape == (
        8192,
        512,
    )
    assert layout.spec("model.layers.0.mlp.experts.1535.down_proj.weight").shape == (
        1024,
        512,
    )
    assert layout.spec("model.ngram_embeddings.embedders.11.weight").shape == (
        9_765_542,
        256,
    )
    assert (
        layout.spec("model.ngram_embeddings.post_projs.11.weight").target_name
        == "model.ngram_embeddings.projection"
    )


@pytest.mark.parametrize("role", ["prefill", "decode"])
def test_model_skeleton_uses_kda_tp8_moe_ep8_and_replicated_mla(role):
    config = FLASHLocalConfig.from_dict(lite_config_dict())
    model = FLASHLocalForCausalLM(
        config, mapping(8, rank=3, role=role), oe_table_placement="host"
    )

    assert isinstance(model.model.layers[0].self_attn, PackedFLASHLocalKDA)
    assert isinstance(
        model.model.layers[3].self_attn,
        SeparateProjectionKimiLinearMLAAttention,
    )
    kda = model.model.layers[0].self_attn
    assert kda.input_projection.weight.shape == (24, 96)
    assert kda.input_projection.weight.numel() == (4 * 4 + 2 * 4) * 96
    assert not any(
        name.startswith(("q_proj.", "k_proj.", "v_proj.", "g_proj.", "f_a_proj."))
        for name, _ in kda.named_parameters()
    )
    assert model.model.layers[3].self_attn.q_b_proj.weight.shape == (48, 24)
    assert model.model.layers[0].moe.experts.w13_weight.shape == (4, 32, 24)
    assert model.model.layers[0].moe.experts.w2_weight.shape == (4, 24, 16)
    assert model.model.layers[0].moe.proj_input.weight.shape == (96, 96)
    assert model.model.layers[0].moe.proj_output.weight.shape == (96, 96)
    assert model.model.layers[0].moe.router.classifier.weight.shape == (12, 24)
    assert len(model.model.embed_tokens.oe_tables) == 2
    assert model.model.embed_tokens.oe_tables[0].numel() == 0
    assert model.model.embed_tokens.oe_tables[0].device.type == "cpu"
    assert model.model.embed_tokens.projection.shape == (12, 96)


@pytest.mark.parametrize(("is_npu", "expected"), [(False, False), (True, True)])
def test_flash_local_projection_layout_resolver_follows_platform(is_npu, expected):
    import tokenspeed.runtime.models.flash_local_attention as attention

    with mock.patch.object(
        attention, "current_platform", return_value=SimpleNamespace(is_npu=is_npu)
    ):
        assert attention.flash_local_prefers_packed_projections() is expected


def test_kimi_mla_default_geometry_still_follows_attention_mapping():
    from tokenspeed.runtime.models.kimi_k3 import KimiLinearMLAAttention

    config = FLASHLocalConfig.from_dict(lite_config_dict())
    parallel = mapping(8, rank=3, role="replicated")
    layer = KimiLinearMLAAttention(
        config=config,
        mapping=parallel,
        hidden_size=config.hidden_size,
        num_heads=config.num_attention_heads,
        qk_nope_head_dim=config.qk_nope_head_dim,
        qk_rope_head_dim=config.qk_rope_head_dim,
        v_head_dim=config.v_head_dim,
        q_lora_rank=config.q_lora_rank,
        kv_lora_rank=config.kv_lora_rank,
        max_position_embeddings=config.max_position_embeddings,
        layer_id=3,
        skip_rope=True,
    )

    assert layer.component_mapping is parallel.attn
    assert layer.num_local_heads == 1
    assert layer.q_b_proj.weight.shape == (6, 24)
    assert layer.fused_qkv_a_proj_with_mqa.weight.shape == (42, 96)


def test_kda_merged_projection_splits_one_gemm_and_materializes_qkv():
    config = FLASHLocalConfig.from_dict(lite_config_dict())
    layer = PackedFLASHLocalKDA(config, mapping(), layer_id=0)
    torch.manual_seed(1201)
    with torch.no_grad():
        layer.input_projection.weight.copy_(
            torch.randn_like(layer.input_projection.weight).mul_(0.05)
        )
    hidden = torch.randn(3, config.hidden_size, dtype=torch.bfloat16)

    qkv, gate, forget_a, beta_a = layer.input_projection(hidden)
    projected = torch.nn.functional.linear(hidden, layer.input_projection.weight)
    projection = layer.input_projection.local_projection
    low_rank = layer.input_projection.head_dim

    assert qkv.is_contiguous()
    torch.testing.assert_close(qkv, projected[:, : 3 * projection])
    torch.testing.assert_close(gate, projected[:, 3 * projection : 4 * projection])
    torch.testing.assert_close(
        forget_a, projected[:, 4 * projection : 4 * projection + low_rank]
    )
    torch.testing.assert_close(beta_a, projected[:, 4 * projection + low_rank :])


def test_kda_merged_projection_rejects_invalid_component_shape_and_id():
    layer = PackedFLASHLocalKDA(
        FLASHLocalConfig.from_dict(lite_config_dict()), mapping(8), layer_id=0
    )
    projection = layer.input_projection

    with pytest.raises(ValueError, match="component 6"):
        projection.load_component(6, torch.empty(4, 96, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="shape/dtype"):
        projection.load_component(0, torch.empty(3, 96, dtype=torch.bfloat16))


def test_kda_merged_projection_meta_load_validates_without_copy():
    config = FLASHLocalConfig.from_dict(lite_config_dict())
    with torch.device("meta"):
        layer = PackedFLASHLocalKDA(config, mapping(8), layer_id=0)

    layer.input_projection.load_component(0, torch.empty(4, 96, dtype=torch.bfloat16))
    assert layer.input_projection.weight.is_meta


def test_model_accepts_bounded_replicated_mla_topology():
    config = FLASHLocalConfig.from_dict(lite_config_dict())
    config.use_over_embedding = False
    model = FLASHLocalForCausalLM(
        config,
        mapping(8, rank=3, role="replicated"),
        oe_table_placement="device",
    )

    assert model.mapping.attn.tp_size == 8
    assert model.model.layers[3].self_attn.component_mapping.tp_size == 1
    assert model.model.layers[3].self_attn.q_b_proj.weight.shape == (48, 24)
    assert model.model.layers[0].moe.proj_input.weight.shape == (96, 96)
    assert model.model.embed_tokens.weight.shape == (16, 96)


def test_strict_loader_covers_rename_shards_experts_and_host_oe():
    config = FLASHLocalConfig.from_dict(lite_config_dict())
    model = FLASHLocalForCausalLM(config, mapping(8, rank=1), oe_table_placement="host")
    layout = model.checkpoint_layout

    def sentinel(_index, name, tensor):
        if name == "model.layers.0.self_attn.linear_core.q_proj.weight":
            return torch.arange(tensor.numel(), dtype=tensor.dtype).view(tensor.shape)
        packed_values = {
            "model.layers.0.self_attn.linear_core.k_proj.weight": 2,
            "model.layers.0.self_attn.linear_core.v_proj.weight": 3,
            "model.layers.0.self_attn.linear_core.g_proj.0.weight": 4,
            "model.layers.0.self_attn.linear_core.f_proj.0.weight": 5,
            "model.layers.0.self_attn.linear_core.b_proj.0.weight": 6,
        }
        if name in packed_values:
            return torch.full_like(tensor, packed_values[name])
        if name == "model.ngram_embeddings.post_projs.1.weight":
            return torch.arange(tensor.numel(), dtype=tensor.dtype).view(tensor.shape)
        mla_values = {
            "model.layers.3.self_attn.q_a_proj.weight": 7,
            "model.layers.3.self_attn.kv_a_proj_with_mqa.weight": 8,
            "model.layers.3.self_attn.g_proj.weight": 9,
        }
        if name in mla_values:
            return torch.full_like(tensor, mla_values[name])
        expert_values = {
            "model.layers.0.mlp.experts.1.gate_proj.weight": 1,
            "model.layers.0.mlp.experts.1.up_proj.weight": 2,
            "model.layers.0.mlp.experts.1.down_proj.weight": 3,
        }
        if name in expert_values:
            return torch.full_like(tensor, expert_values[name])
        return tensor

    model.load_weights(weights(layout, sentinel))

    source = sentinel(
        0,
        "model.layers.0.self_attn.linear_core.q_proj.weight",
        torch.empty((32, 96), dtype=torch.bfloat16),
    )
    merged = model.model.layers[0].self_attn.input_projection.weight
    assert torch.equal(merged[:4], source[4:8])
    for start, end, value in (
        (4, 8, 2),
        (8, 12, 3),
        (12, 16, 4),
        (16, 20, 5),
        (20, 24, 6),
    ):
        assert torch.equal(merged[start:end], torch.full_like(merged[start:end], value))
    experts = model.model.layers[0].moe.experts
    assert torch.equal(
        experts.w13_weight[0, :16], torch.ones_like(experts.w13_weight[0, :16])
    )
    assert torch.equal(
        experts.w13_weight[0, 16:], torch.full_like(experts.w13_weight[0, 16:], 2)
    )
    assert torch.equal(experts.w2_weight[0], torch.full_like(experts.w2_weight[0], 3))
    assert model.model.layers[0].moe.local_expert_ids == (1, 9, 17, 25)
    assert model.model.embed_tokens.oe_tables[0].device.type == "cpu"
    assert model.model.embed_tokens.oe_tables[0].shape == (15, 6)
    projection_source = sentinel(
        0,
        "model.ngram_embeddings.post_projs.1.weight",
        torch.empty((96, 6), dtype=torch.bfloat16),
    )
    assert torch.equal(model.model.embed_tokens.projection[:6], projection_source.t())
    mla = model.model.layers[3].self_attn
    for parameter, value in (
        (mla.q_a_proj.weight, 7),
        (mla.kv_a_proj_with_mqa.weight, 8),
        (mla.g_proj.weight, 9),
    ):
        assert torch.equal(parameter, torch.full_like(parameter, value))


@pytest.mark.parametrize("rank", [0, 7])
def test_strict_loader_places_kda_packed_edge_rank_shards(rank):
    config = FLASHLocalConfig.from_dict(lite_config_dict())
    model = FLASHLocalForCausalLM(
        config, mapping(8, rank=rank), oe_table_placement="host"
    )
    source = torch.arange(32 * 96, dtype=torch.bfloat16).reshape(32, 96)

    def sentinel(_index, name, tensor):
        if name == "model.layers.0.self_attn.linear_core.q_proj.weight":
            return source
        return tensor

    model.load_weights(weights(model.checkpoint_layout, sentinel))

    expected = source[rank * 4 : (rank + 1) * 4]
    torch.testing.assert_close(
        model.model.layers[0].self_attn.input_projection.weight[:4], expected
    )


def test_lite_grouped_expert_placement_is_a_bijection() -> None:
    models = [
        FLASHLocalForCausalLM(
            FLASHLocalConfig.from_dict(lite_config_dict()),
            mapping(8, rank=rank),
            oe_table_placement="host",
        )
        for rank in range(8)
    ]

    assert sorted(
        expert
        for model in models
        for expert in model.model.layers[0].moe.local_expert_ids
    ) == list(range(32))


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("missing", "missing 1 source"),
        ("duplicate", "Duplicate Flash-Lite checkpoint weight"),
        ("unexpected", "Unexpected Lite checkpoint weight"),
        ("shape", "shape/dtype"),
    ],
)
def test_strict_loader_rejects_incomplete_or_ambiguous_stream(case, message):
    config = FLASHLocalConfig.from_dict(
        lite_config_dict(gmoe_strategy="global_expert_id")
    )
    model = FLASHLocalForCausalLM(config, mapping(), oe_table_placement="host")
    checkpoint = list(weights(model.checkpoint_layout))
    if case == "missing":
        checkpoint.pop()
    elif case == "duplicate":
        checkpoint.insert(1, checkpoint[0])
    elif case == "unexpected":
        checkpoint.insert(0, ("model.ghost.weight", torch.zeros(1)))
    else:
        name, tensor = checkpoint[0]
        checkpoint[0] = (name, tensor[:-1])

    with pytest.raises(ValueError, match=message):
        model.load_weights(checkpoint)
