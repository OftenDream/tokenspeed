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

"""Behavior guards for the unified Flash-Lite runtime entry."""

from test.runtime.test_lite_model_loader import lite_config_dict

import pytest

from tokenspeed.runtime.configs.flash_kda_config import FLASHLocalConfig
from tokenspeed.runtime.models.flash_kda import _canonical_flash_kda_weight_name
from tokenspeed.runtime.models.flash_local_checkpoint import FLASHLocalCheckpointLayout


def test_single_config_preserves_flash_lite_semantics() -> None:
    config = FLASHLocalConfig.from_dict(lite_config_dict())

    assert config.architectures == ["FLASHLocalForCausalLM"]
    assert config.strict_checkpoint_layout
    assert config.intermediate_size == config.ffn_hidden_size == 96
    assert config.moe_intermediate_size == config.expert_ffn_hidden_size == 16
    assert config.n_shared_experts == config.num_shared_experts == 1
    assert config.linear_layer_ids == [0, 1, 2]
    assert config.full_attention_layer_ids == [3]
    assert config.oe_ignore_tokens == list(config.special_token_ids)
    assert config.over_embedding_m == config.oe_table_base_rows


@pytest.mark.parametrize(
    ("source_name", "flash_target", "lite_target"),
    [
        (
            "model.layers.0.self_attn.linear_core.q_proj.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.input_projection.weight",
        ),
        (
            "model.layers.0.self_attn.linear_core.b_proj.1.weight",
            "model.layers.0.self_attn.b_proj.fc2.weight",
            "model.layers.0.self_attn.beta_b_proj.weight",
        ),
        (
            "model.layers.3.self_attn.kv_b_proj.weight",
            "model.layers.3.self_attn.kv_b_proj.weight",
            "model.layers.3.self_attn.kv_b_proj.weight",
        ),
        (
            "model.layers.0.mlp.experts.1.down_proj.weight",
            "model.layers.0.moe.experts.1.down_proj.weight",
            "model.layers.0.mlp.experts.w2_weight",
        ),
        (
            "model.ngram_embeddings.embedders.0.weight",
            "model.ngram_embeddings.embedders.0.weight",
            "model.ngram_embeddings.embedders.0.weight",
        ),
    ],
)
def test_checkpoint_sources_preserve_current_physical_layouts(
    source_name: str,
    flash_target: str,
    lite_target: str,
) -> None:
    layout = FLASHLocalCheckpointLayout(FLASHLocalConfig.from_dict(lite_config_dict()))

    assert _canonical_flash_kda_weight_name(source_name) == flash_target
    assert layout.spec(source_name).target_name == lite_target
