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


"""Selection and numerical contracts for group-first MoE stage boundaries."""

from types import SimpleNamespace

import pytest
import tokenspeed_kernel
import torch
from tokenspeed_kernel.ops.moe.gmoe import (
    GMoEContext,
    GMoEInputs,
    GMoEStages,
    select_gmoe_stages,
)
from tokenspeed_kernel.registry import KernelRegistry, Priority, register_kernel
from tokenspeed_kernel.selection import NoKernelFoundError, SelectedKernel
from tokenspeed_kernel.signature import format_signatures

from tokenspeed.runtime.configs.flash_kda_config import FLASHLocalConfig
from tokenspeed.runtime.models.flash_local_moe import GroupAwareFlashLocalMoE


def test_stage_selection_and_independent_override(monkeypatch):
    default = select_gmoe_stages(
        input_dtype=torch.bfloat16,
        traits={"weight_dtype": "int8", "egp_size": 2},
    )
    assert default.pre.name == "composed_gmoe_pre"
    assert default.post.name == "composed_gmoe_post"
    # Isolate test-only registrations from other tests and production selection.
    registry = KernelRegistry()
    original = KernelRegistry.get()
    for mode in ("gmoe_pre", "gmoe_post"):
        for spec in original.get_for_operator("moe", mode):
            registry.register(spec, original.get_impl(spec.name))
    monkeypatch.setattr(KernelRegistry, "_instance", registry)

    @register_kernel(
        "moe",
        "gmoe_pre",
        name="test_gmoe_pre",
        solution="test_stage",
        priority=Priority.PORTABLE,
        signatures=format_signatures("hidden_states", "dense", {torch.bfloat16}),
        traits={"weight_dtype": frozenset({"int8"}), "egp_size": frozenset({2})},
    )
    def replacement(**kwargs):
        return kwargs

    selected = select_gmoe_stages(
        input_dtype=torch.bfloat16,
        traits={"weight_dtype": "int8", "egp_size": 2},
        pre_solution="test_stage",
        post_solution="composed",
    )
    assert selected.pre.impl is replacement
    assert selected.post.name == default.post.name
    with pytest.raises(NoKernelFoundError):
        select_gmoe_stages(
            input_dtype=torch.bfloat16,
            traits={"weight_dtype": "unquant", "egp_size": 2},
            pre_solution="test_stage",
        )
    with pytest.raises(NoKernelFoundError):
        select_gmoe_stages(
            input_dtype=torch.bfloat16,
            traits={},
            post_solution="missing_fused",
        )


@pytest.mark.parametrize(
    "groups,egp_rank,egp_size", [(4, 0, 2), (4, 1, 2), (8, 0, 1), (8, 0, 2), (8, 1, 2)]
)
@pytest.mark.parametrize("tokens", [1, 3, 32])
def test_composed_stages_identity_and_communication(groups, egp_rank, egp_size, tokens):
    x = torch.arange(tokens * 32).reshape(tokens, 32).to(torch.bfloat16) / 32
    events = []

    def a2a(out, value, group):
        events.append(("a2a", group))
        out.copy_(value)

    def ag(value, group, splits):
        events.append(("ag", group))
        assert splits == [tokens * groups] * egp_size
        # Distinct inputs across EGP ranks, not copies of the same tokens.
        return torch.cat([value + i for i in range(egp_size)])

    def rs(value, group, splits):
        events.append(("rs", group))
        assert splits == [tokens * groups] * egp_size
        return value.narrow(0, egp_rank * splits[0], splits[0]) * egp_size

    def route(received):
        events.append(("route",))
        ids = torch.tensor([0, 4], dtype=torch.int32).expand(received.shape[0], 2)
        weights = torch.tensor([0.75, 0.25]).expand_as(ids).clone()
        return weights, ids

    def shared(value):
        events.append(("shared",))
        return value * 0.5

    ctx = GMoEContext(
        num_groups=groups,
        egp_group=tuple(range(egp_size)),
        egp_rank=egp_rank,
        exchange_group=tuple(range(groups)),
        num_experts=4,
        norm_scale=2,
        proj_input=torch.nn.Identity(),
        norm=torch.nn.Identity(),
        shared_experts=shared,
        proj_output=torch.nn.Identity(),
        route=route,
        router=torch.nn.Module(),
        top_k=2,
        routed_scaling_factor=1,
        renormalize_topk=False,
        all_to_all=a2a,
        all_gather=ag,
        reduce_scatter=rs,
    )
    stages = select_gmoe_stages(input_dtype=torch.bfloat16, traits={})
    inputs = stages.pre(hidden_states=x, context=ctx)
    assert events[0] == ("shared",)
    assert inputs.received.shape == (tokens * groups * egp_size, 32 // groups)
    # The local identity source is the pre-AG buffer, even on EGP rank 1.
    grouped = (
        (x * 2)
        .reshape(tokens, groups, -1)
        .permute(1, 0, 2)
        .reshape(tokens * groups, -1)
    )
    torch.testing.assert_close(inputs.local_received, grouped, rtol=0, atol=0)
    leaf_result = inputs.received * 0.75
    output = stages.post(routed=leaf_result, inputs=inputs, context=ctx)
    local_routed = (
        (grouped + egp_rank) * 0.75 * egp_size if egp_size > 1 else grouped * 0.75
    )
    expected = local_routed + grouped * 0.25
    expected = (
        expected.reshape(groups, tokens, -1).permute(1, 0, 2).flatten(1) + x * 0.5
    )
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert [e[0] for e in events] == (
        ["shared", "a2a", "ag", "route", "rs", "a2a"]
        if egp_size > 1
        else ["shared", "a2a", "route", "a2a"]
    )
    assert inputs.topk_ids.shape[0] == tokens * groups * egp_size


def test_model_forward_keeps_middle_call_and_uses_stages(monkeypatch):
    x = torch.ones(3, 32, dtype=torch.bfloat16)
    received = torch.ones(48, 4, dtype=torch.bfloat16)
    weights = torch.ones(48, 2)
    ids = torch.zeros(48, 2, dtype=torch.int32)
    prepared = GMoEInputs(received, received[:24], weights, ids, x)
    context, plan, experts = object(), object(), object()
    events = []

    def pre(**kwargs):
        assert kwargs == {"hidden_states": x, "context": context}
        events.append("pre")
        return prepared

    def middle(p, value, w, logits, **kwargs):
        assert p is plan and value is received and w is experts and logits is weights
        assert kwargs["topk_weights"] is weights and kwargs["topk_ids"] is ids
        assert kwargs["num_tokens_global"] == 48
        assert kwargs["max_num_tokens_per_gpu"] == 24
        events.append("middle")
        return received

    def post(**kwargs):
        assert kwargs["inputs"] is prepared and kwargs["context"] is context
        assert kwargs["routed"] is received
        events.append("post")
        return x

    monkeypatch.setattr(tokenspeed_kernel, "moe_apply", middle)
    model = SimpleNamespace(
        topology=SimpleNamespace(num_groups=8),
        experts=experts,
        _moe_plan=plan,
        _moe_stages=GMoEStages(
            SelectedKernel("pre", pre), SelectedKernel("post", post)
        ),
        _moe_stage_context=context,
    )
    output = GroupAwareFlashLocalMoE.forward(
        model,
        x,
        num_global_tokens=48,
        max_num_tokens_per_gpu=3,
        ctx=SimpleNamespace(forward_mode="decode"),
    )
    assert output is x and events == ["pre", "middle", "post"]
    assert GroupAwareFlashLocalMoE.forward(model, x[:0]) is not None
    assert events == ["pre", "middle", "post"]


def test_stage_solutions_roundtrip_config():
    from test.runtime.test_lite_model_loader import lite_config_dict

    config = FLASHLocalConfig.from_dict(
        lite_config_dict(
            gmoe_strategy="gmoe_aware",
            gmoe_pre_solution="composed",
            gmoe_post_solution="composed",
            gmoe_expert_solution="flash_npu_routed",
        )
    )
    saved = config.to_dict()
    saved["num_layers"] = config.num_hidden_layers
    saved["architectures"] = ["FLASHLocalForCausalLM"]
    restored = FLASHLocalConfig.from_dict(saved)
    assert restored.gmoe_pre_solution == restored.gmoe_post_solution == "composed"
    assert restored.gmoe_expert_solution == "flash_npu_routed"


def test_routed_expert_selection_is_explicit():
    from tokenspeed_kernel.platform import current_platform

    if not current_platform().is_npu:
        pytest.skip("Ascend registry contract")
    kwargs = dict(
        input_dtype=torch.bfloat16,
        activation="silu",
        routing_mode="precomputed_topk",
        ep_size=2,
        ispp=1024,
        internal_activation_dtype="int8",
    )
    old = tokenspeed_kernel.moe_plan("int8", **kwargs)
    new = tokenspeed_kernel.moe_plan("int8", solution="flash_npu_routed", **kwargs)
    assert old["apply_kernel_name"] == "ascend_int8_precomputed_moe_apply"
    assert new["apply_kernel_name"] == "ascend_routed_int8_precomputed_moe_apply"
    full = tokenspeed_kernel.moe_plan(
        "int8", solution="flash_npu_routed_full", **kwargs
    )
    assert full["apply_kernel_name"] == "ascend_routed_full_int8_precomputed_moe_apply"
    for ep_size in (1, 2, 4, 8):
        kwargs["ep_size"] = ep_size
        assert (
            tokenspeed_kernel.moe_plan(
                "int8", solution="flash_npu_routed_full", **kwargs
            )["apply_kernel_name"]
            == "ascend_routed_full_int8_precomputed_moe_apply"
        )
        bf16_kwargs = {**kwargs, "internal_activation_dtype": "input"}
        assert (
            tokenspeed_kernel.moe_plan(
                "unquant", solution="flash_npu_routed_full", **bf16_kwargs
            )["apply_kernel_name"]
            == "ascend_routed_full_bf16_precomputed_moe_apply"
        )
        assert (
            tokenspeed_kernel.moe_plan("unquant", **bf16_kwargs)["apply_kernel_name"]
            == "ascend_bf16_precomputed_moe_apply"
        )


def test_explicit_expert_solution_failure_is_not_suppressed(monkeypatch):
    def missing(*args, **kwargs):
        raise NoKernelFoundError("test unavailable expert solution")

    monkeypatch.setattr(tokenspeed_kernel, "moe_plan", missing)
    fake = SimpleNamespace(
        _moe_plan=None,
        _quant_kind="int8",
        zero_expert_num=32,
        experts=SimpleNamespace(ep_size=2),
        config=SimpleNamespace(
            expert_ffn_hidden_size=1024, gmoe_expert_solution="missing"
        ),
    )
    with pytest.raises(NoKernelFoundError):
        GroupAwareFlashLocalMoE.process_weights_after_loading(fake)


def test_routed_expert_preparation_rejects_unsupported_layout():
    from tokenspeed_kernel.platform import current_platform

    if not current_platform().is_npu:
        pytest.skip("Ascend backend contract")
    from tokenspeed_kernel_npu.ops.moe import ascend_routed_int8_process_moe_weights

    w = SimpleNamespace(
        ep_size=1,
        num_experts=384,
        num_local_experts=384,
        hidden_size=31,
        intermediate_size=1024,
    )
    with pytest.raises(ValueError, match="H/I aligned to 32"):
        ascend_routed_int8_process_moe_weights(plan={}, w=w)
    w.hidden_size = 512
    w.ep_size, w.num_local_experts, w.w13_smooth_scale = 2, 192, torch.ones(1)
    with pytest.raises(ValueError, match="w13_smooth_scale.*shape"):
        ascend_routed_int8_process_moe_weights(plan={}, w=w)
    w.w13_smooth_scale = None
    w.num_experts, w.num_local_experts = 2050, 1025
    with pytest.raises(ValueError, match="at most 1024 local experts"):
        ascend_routed_int8_process_moe_weights(plan={}, w=w)
    w.num_experts, w.num_local_experts = 385, 192
    with pytest.raises(ValueError, match="equal nonempty expert shards"):
        ascend_routed_int8_process_moe_weights(plan={}, w=w)


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("hidden_size", 31, "H/I aligned"),
        ("intermediate_size", 8193, "H/I aligned"),
        ("num_local_experts", 1025, "1..1024"),
        ("ep_size", 0, "equal expert shards"),
        ("w13_smooth_scale", torch.ones(32), "does not consume"),
    ],
)
def test_bf16_routed_preparation_rejects_unsupported_contract(field, value, message):
    from tokenspeed_kernel.platform import current_platform

    if not current_platform().is_npu:
        pytest.skip("Ascend backend contract")
    from tokenspeed_kernel_npu.ops.moe import ascend_routed_bf16_process_moe_weights

    w = SimpleNamespace(
        ep_size=1,
        num_experts=1,
        num_local_experts=1,
        hidden_size=32,
        intermediate_size=32,
    )
    setattr(w, field, value)
    with pytest.raises(ValueError, match=message):
        ascend_routed_bf16_process_moe_weights(plan={}, w=w)


@pytest.mark.parametrize("local_experts", [1, 47, 193, 384, 512, 513, 1023, 1024])
@pytest.mark.parametrize("smooth_quant", ["none", "w13", "w2", "both"])
def test_routed_expert_count_and_start_follow_weights(
    monkeypatch, local_experts, smooth_quant
):
    from tokenspeed_kernel.platform import current_platform

    if not current_platform().is_npu:
        pytest.skip("Ascend backend contract")
    import tokenspeed_kernel_npu.ops.moe as backend

    w = SimpleNamespace(
        ep_size=2,
        ep_rank=1,
        num_experts=2 * local_experts,
        num_local_experts=local_experts,
        hidden_size=512,
        intermediate_size=1024,
        w13_weight=object(),
        w13_weight_scale=object(),
        _ascend_int8_moe_weights_processed=True,
    )
    if smooth_quant in {"w13", "both"}:
        w.w13_smooth_scale = torch.ones(local_experts, 512)
    if smooth_quant in {"w2", "both"}:
        w.w2_smooth_scale = torch.ones(local_experts, 1024)
    prepared = []
    monkeypatch.setattr(
        backend,
        "ascend_int8_process_moe_weights",
        lambda **kw: prepared.append(kw["w"]),
    )
    result = torch.zeros(1)
    calls = []

    def fused(x, ids, weight, scale, expert_start, *, smooth13=None, smooth2=None):
        assert smooth13 is getattr(w, "w13_smooth_scale", None)
        assert smooth2 is getattr(w, "w2_smooth_scale", None)
        calls.append((weight, scale, expert_start))
        return result, None, None, None

    monkeypatch.setattr(
        torch.ops.custom, "fused_init_routing_mm13_swiglu", fused, raising=False
    )
    monkeypatch.setattr(backend, "_ascend_int8_gmm2_finalize", lambda *args: args[0])
    plan = {"activation": "swiglu"}
    backend.ascend_routed_int8_process_moe_weights(plan=plan, w=w)
    assert prepared == [w]
    out = backend.ascend_routed_int8_precomputed_moe_apply(
        plan=plan,
        x=torch.empty(1),
        w=w,
        router_logits=None,
        topk_ids=torch.zeros(1, 16, dtype=torch.int32),
        topk_weights=torch.ones(1, 16),
    )
    assert out is result
    assert calls == [(w.w13_weight, w.w13_weight_scale, local_experts)]
