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

"""Cheap LongCat-Flash model wiring tests."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn

from tokenspeed.runtime.configs.model_config import (
    AttentionArch,
    _resolve_attention_family,
    configure_longcat_lsa_attention,
)
from tokenspeed.runtime.layers.moe.topk import StandardTopKOutput
from tokenspeed.runtime.models.longcat_flash import (
    LongcatFlashForCausalLM,
    _ensure_longcat_config,
    _get_longcat_moe_quant_config,
    _LongCatOverEmbedding,
    _RuntimeLongcatModel,
    _RuntimeLongcatMoE,
)


class TestLongcatFlashRegistry(unittest.TestCase):
    def test_registered(self):
        from tokenspeed.runtime.models.registry import ModelRegistry

        cls, arch = ModelRegistry.resolve_model_cls(["LongcatFlashForCausalLM"])
        self.assertIs(cls, LongcatFlashForCausalLM)
        self.assertEqual(arch, "LongcatFlashForCausalLM")

    def test_longcat_2_entry_registered(self):
        from tokenspeed.runtime.models.longcat_flash import LongcatCausalLM
        from tokenspeed.runtime.models.registry import ModelRegistry

        cls, arch = ModelRegistry.resolve_model_cls(["LongcatCausalLM"])
        self.assertIs(cls, LongcatCausalLM)
        self.assertEqual(arch, "LongcatCausalLM")

    def test_attention_and_double_attention_metadata_registered(self):
        from tokenspeed.runtime.configs import model_config

        self.assertIn("LongcatFlashForCausalLM", model_config._MLA_ARCHITECTURES)
        self.assertIn(
            "LongcatFlashForCausalLM",
            model_config._DOUBLE_ATTENTION_LAYER_ARCHITECTURES,
        )
        self.assertIn("LongcatCausalLM", model_config._DSA_ARCHITECTURES)
        self.assertNotIn("LongcatCausalLM", model_config._MLA_ARCHITECTURES)
        self.assertIn(
            "LongcatCausalLM",
            model_config._DOUBLE_ATTENTION_LAYER_ARCHITECTURES,
        )


class TestLongcatFlashConfig(unittest.TestCase):
    def test_longcat_2_geometry_selects_dsa_with_pair_owners(self):
        hf_config = SimpleNamespace(
            architectures=["LongcatCausalLM"],
            num_layers=38,
            kv_lora_rank=512,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
            index_topk=2048,
            index_head_dim=128,
            index_n_heads=32,
            index_init_tokens=16,
            index_local_tokens=1024,
            cli_factor=2,
            index_k_norm_type="rms",
        )
        runtime_config = SimpleNamespace(
            hf_config=hf_config,
            hf_text_config=hf_config,
            num_attention_layers=76,
        )

        attention_family = _resolve_attention_family(hf_config, hf_config)
        self.assertIs(attention_family.configure, configure_longcat_lsa_attention)

        attention_family.configure(runtime_config)

        self.assertEqual(runtime_config.attention_arch, AttentionArch.DSA)
        self.assertEqual(runtime_config.index_topk, 2048)
        self.assertEqual(
            runtime_config.indexer_layer_ids,
            frozenset(range(0, 76, 2)),
        )

    def test_config_aliases_are_normalized(self):
        config = SimpleNamespace(
            num_layers=28,
            ffn_hidden_size=14336,
            expert_ffn_hidden_size=2048,
            moe_topk=8,
            hidden_size=6144,
            n_routed_experts=512,
        )

        _ensure_longcat_config(config)

        self.assertEqual(config.num_hidden_layers, 28)
        self.assertEqual(config.intermediate_size, 14336)
        self.assertEqual(config.moe_intermediate_size, 2048)
        self.assertEqual(config.num_experts_per_tok, 8)
        self.assertEqual(config.hidden_act, "silu")
        self.assertEqual(config.zero_expert_num, 0)
        self.assertFalse(config.router_bias)

    def test_over_embedding_aliases_are_normalized(self):
        config = SimpleNamespace(
            num_layers=38,
            ffn_hidden_size=12288,
            expert_ffn_hidden_size=2048,
            moe_topk=12,
            hidden_size=8192,
            vocab_size=163840,
            n_routed_experts=768,
            oe_vocab_size_ratio=100.567,
            oe_neighbor_num=5,
            oe_split_num=4,
            oe_ignored_token_ids=[2, 3],
        )

        _ensure_longcat_config(config)

        self.assertTrue(config.use_over_embedding)
        self.assertEqual(config.over_embedding_m, 16476897)
        self.assertEqual(config.oe_ignore_tokens, [2, 3])

    def test_longcat_2_builds_shared_over_embedding_layer(self):
        model = object.__new__(_RuntimeLongcatModel)
        model.mapping = SimpleNamespace(
            attn=SimpleNamespace(tp_rank=3, tp_size=8, tp_group=tuple(range(8)))
        )
        config = SimpleNamespace(
            use_over_embedding=True,
            vocab_size=163840,
            hidden_size=8192,
            over_embedding_m=16476897,
            oe_neighbor_num=5,
            oe_split_num=4,
            oe_ignore_tokens=[2, 3],
            eos_token_id=2,
        )

        with mock.patch(
            "tokenspeed.runtime.models.longcat_flash._LongCatOverEmbedding"
        ) as over_embedding:
            embedding = _RuntimeLongcatModel._build_embed_tokens(model, config)

        self.assertIs(embedding, over_embedding.return_value)
        over_embedding.assert_called_once_with(
            num_embeddings=163840,
            embedding_dim=8192,
            over_embedding_m=16476897,
            hashes_per_order=4,
            max_ngram_order=5,
            tp_rank=3,
            tp_size=8,
            tp_group=tuple(range(8)),
            ignored_token_ids=(2, 3),
            eos_token_id=2,
            fix_normalize_factor=False,
        )

    def test_shared_embed_replaces_only_the_oe_word_table(self):
        model = object.__new__(LongcatFlashForCausalLM)
        nn.Module.__init__(model)
        over_embedding = object.__new__(_LongCatOverEmbedding)
        nn.Module.__init__(over_embedding)
        over_embedding.word_embedding = nn.Embedding(2, 3)
        model.model = nn.Module()
        model.model.embed_tokens = over_embedding
        model.lm_head = nn.Linear(3, 2, bias=False)
        shared_embed = nn.Parameter(torch.empty(2, 3))
        shared_head = nn.Parameter(torch.empty(2, 3))

        with (
            mock.patch("torch.cuda.empty_cache"),
            mock.patch("torch.cuda.synchronize"),
        ):
            model.set_embed_and_head(shared_embed, shared_head)

        self.assertIs(over_embedding.word_embedding.weight, shared_embed)
        self.assertIs(model.lm_head.weight, shared_head)


class TestLongcatMixedFp8Config(unittest.TestCase):
    def test_moe_layer_uses_unquantized_backend_when_all_experts_are_ignored(self):
        config = SimpleNamespace(n_routed_experts=2)
        quant_config = SimpleNamespace(
            ignored_layers=[
                f"model.layers.0.mlp.experts.{expert_id}.{proj_name}"
                for expert_id in range(2)
                for proj_name in ("gate_proj", "up_proj", "down_proj")
            ]
        )

        self.assertIsNone(
            _get_longcat_moe_quant_config(
                config,
                quant_config,
                "model.layers.0.mlp",
            )
        )

    def test_moe_layer_keeps_quantization_when_no_experts_are_ignored(self):
        config = SimpleNamespace(n_routed_experts=2)
        quant_config = SimpleNamespace(ignored_layers=[])

        self.assertIs(
            _get_longcat_moe_quant_config(
                config,
                quant_config,
                "model.layers.0.mlp",
            ),
            quant_config,
        )

    def test_moe_layer_rejects_partially_ignored_experts(self):
        config = SimpleNamespace(n_routed_experts=2)
        quant_config = SimpleNamespace(
            ignored_layers=[
                "model.layers.0.mlp.experts.0.gate_proj",
            ]
        )

        with self.assertRaisesRegex(ValueError, "partially ignored"):
            _get_longcat_moe_quant_config(
                config,
                quant_config,
                "model.layers.0.mlp",
            )


class TestLongcatZeroExpert(unittest.TestCase):
    def test_zero_experts_select_precomputed_topk_moe_plan(self):
        config = SimpleNamespace(
            n_routed_experts=3,
            zero_expert_num=1,
            zero_expert_type="identity",
            routed_scaling_factor=1.0,
            hidden_act="silu",
            moe_topk=2,
            hidden_size=8,
            moe_intermediate_size=16,
            norm_topk_prob=False,
        )
        mapping = SimpleNamespace(
            moe=SimpleNamespace(
                tp_rank=0,
                tp_size=1,
                ep_rank=0,
                ep_size=1,
            )
        )
        router = SimpleNamespace(e_score_correction_bias=torch.zeros(4))

        with (
            mock.patch.dict(
                "tokenspeed.runtime.models.longcat_flash.global_server_args_dict",
                {"ep_num_redundant_experts": 0, "enable_deep_ep": False},
            ),
            mock.patch(
                "tokenspeed.runtime.models.longcat_flash._RuntimeLongcatRouter",
                return_value=router,
            ),
            mock.patch(
                "tokenspeed.runtime.models.longcat_flash._MoELayer"
            ) as moe_layer,
        ):
            _RuntimeLongcatMoE(config, mapping)

        self.assertEqual(
            moe_layer.call_args.kwargs["routing_mode"],
            "precomputed_topk",
        )

    def test_identity_zero_expert_masks_and_adds_hidden_state(self):
        moe = object.__new__(_RuntimeLongcatMoE)
        moe.zero_expert_num = 1
        moe.n_routed_experts = 3
        moe.zero_expert_type = "identity"
        moe.mapping = SimpleNamespace(moe=SimpleNamespace(tp_ep_size=2))
        hidden_states = torch.tensor(
            [[2.0, 4.0], [6.0, 8.0]],
            dtype=torch.float32,
        )
        topk_output = StandardTopKOutput(
            topk_weights=torch.tensor([[0.25, 0.75], [0.5, 0.5]]),
            topk_ids=torch.tensor([[0, -1], [3, 1]]),
            router_logits=torch.zeros(2, 4),
        )

        zero_output = _RuntimeLongcatMoE._apply_zero_experts(
            moe,
            hidden_states,
            topk_output,
        )

        torch.testing.assert_close(
            zero_output,
            torch.tensor([[0.75, 1.5], [1.5, 2.0]]),
        )
        torch.testing.assert_close(
            topk_output.topk_weights,
            torch.tensor([[0.25, 0.0], [0.0, 0.5]]),
        )
        torch.testing.assert_close(
            topk_output.topk_ids,
            torch.tensor([[0, 0], [0, 1]]),
        )


class _CaptureFinalNorm:
    def final_norm(self, hidden_states, residual, ctx, norm):
        return hidden_states + residual + 1000, torch.full_like(hidden_states, 30)

    def gather_residual(self, residual, ctx):
        return residual + 100


class _CaptureLayer(nn.Module):
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
        self.final_norm_comm = _CaptureFinalNorm()

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


class TestLongcatDFlashCapture(unittest.TestCase):
    @staticmethod
    def _make_target():
        model = object.__new__(_RuntimeLongcatModel)
        nn.Module.__init__(model)
        model.layers = nn.ModuleList(
            [
                _CaptureLayer(
                    hidden_delta=1,
                    residual_value=10,
                    captured_value=110,
                ),
                _CaptureLayer(
                    hidden_delta=1,
                    residual_value=20,
                    captured_value=120,
                ),
            ]
        )
        model.layers_to_capture = set()
        model.norm = nn.Identity()

        target = object.__new__(LongcatFlashForCausalLM)
        nn.Module.__init__(target)
        target.model = model
        target.capture_aux_hidden_states = False
        return target

    def test_captures_materialized_completed_layer_residuals(self):
        target = self._make_target()
        slots = [torch.empty(1, 1), torch.empty(1, 1)]
        callbacks = []
        target.set_dflash_layers_to_capture(
            [0, 1],
            incremental_callback=lambda capture_idx, num_tokens: callbacks.append(
                (capture_idx, num_tokens)
            ),
            slot_bufs=slots,
        )
        target.model._dflash_incr_active = True
        ctx = SimpleNamespace(forward_mode=SimpleNamespace(is_idle=lambda: False))

        output, captures = target.model(
            input_ids=torch.empty(1, dtype=torch.int64),
            positions=torch.empty(1, dtype=torch.int64),
            ctx=ctx,
            input_embeds=torch.ones(1, 1),
        )

        self.assertEqual(target.model.layers_to_capture, {1, 2})
        torch.testing.assert_close(captures[0], torch.tensor([[120.0]]))
        torch.testing.assert_close(captures[1], torch.tensor([[130.0]]))
        torch.testing.assert_close(output, torch.tensor([[1023.0]]))
        torch.testing.assert_close(slots[0], torch.tensor([[120.0]]))
        torch.testing.assert_close(slots[1], torch.tensor([[130.0]]))
        self.assertEqual(callbacks, [(0, 1), (1, 1)])

    def test_rejects_capture_ids_outside_target_layers(self):
        target = self._make_target()

        with self.assertRaisesRegex(ValueError, r"valid range is \[0, 1\]"):
            target.set_dflash_layers_to_capture([2])


class TestLongcatCheckpointLoading(unittest.TestCase):
    def test_missing_kv_scale_params_are_silent(self):
        model = object.__new__(LongcatFlashForCausalLM)
        with mock.patch(
            "tokenspeed.runtime.models.longcat_flash._longcat_logger.warning"
        ) as warning:
            self.assertIsNone(model.get_param({}, "model.layers.0.self_attn.0.k_scale"))
            self.assertIsNone(model.get_param({}, "model.layers.0.self_attn.1.v_scale"))
        warning.assert_not_called()

    def test_missing_mtp_params_are_silent(self):
        model = object.__new__(LongcatFlashForCausalLM)
        with mock.patch(
            "tokenspeed.runtime.models.longcat_flash._longcat_logger.warning"
        ) as warning:
            self.assertIsNone(
                model.get_param({}, "model.mtp.layers.0.self_attn.q_proj.weight")
            )
        warning.assert_not_called()


if __name__ == "__main__":
    unittest.main()
