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

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from test.runtime.test_lite_model_loader import lite_config_dict, mapping, weights
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from tokenspeed.runtime.configs.flash_kda_config import FLASHLocalConfig
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.models.flash_kda import FLASHLocalForCausalLM
from tokenspeed.runtime.models.flash_local_moe import (
    GroupAwareFlashLocalMoE,
    gmoe_local_expert_ids,
    gmoe_topology,
)


def test_lite_group_aware_moe_import_does_not_require_accelerator() -> None:
    root = Path(__file__).parents[2]
    env = os.environ.copy()
    source_paths = [root / "python", root / "tokenspeed-kernel" / "python"]
    env["PYTHONPATH"] = os.pathsep.join(
        [*(str(path) for path in source_paths), env.get("PYTHONPATH", "")]
    )
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; "
            "import tokenspeed.runtime.layers.moe.loader; "
            "import tokenspeed.runtime.layers.moe.weights.unquant; "
            "print('tokenspeed_kernel' in sys.modules)",
        ],
        cwd=root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert probe.stdout.strip() == "False"


def test_lite_router_bias_only_changes_selection() -> None:
    config = FLASHLocalConfig.from_dict(lite_config_dict(routed_scaling_factor=5.0))
    layer = GroupAwareFlashLocalMoE(config, mapping(8, rank=0))
    group_input = torch.randn(3, layer.group_hidden, dtype=torch.bfloat16)
    layer.router.classifier.weight.data.zero_()
    bias = layer.router.e_score_correction_bias
    bias.data.copy_(torch.arange(12, dtype=torch.float32))

    topk_weights, topk_ids = layer._route(group_input)

    probabilities = torch.softmax(
        F.linear(group_input.float(), layer.router.classifier.weight.float()), dim=-1
    )
    expected_ids = torch.topk(
        probabilities + bias,
        2,
        dim=-1,
        sorted=False,
    ).indices
    assert torch.equal(topk_ids, expected_ids.to(torch.int32))
    assert torch.allclose(
        topk_weights,
        torch.full_like(topk_weights, 5.0 / 12.0),
        atol=1e-6,
        rtol=1e-5,
    )
    assert torch.allclose(
        topk_weights.sum(dim=-1),
        torch.full_like(topk_weights.sum(dim=-1), 10.0 / 12.0),
    )


@pytest.mark.parametrize(
    (
        "ep_size",
        "groups",
        "rank",
        "egp_size",
        "exchange_group",
        "egp_group",
        "owned",
    ),
    [
        (8, 4, 0, 2, (0, 2, 4, 6), (0, 1), tuple(range(0, 192))),
        (8, 4, 5, 2, (1, 3, 5, 7), (4, 5), tuple(range(960, 1152))),
        (8, 8, 6, 1, tuple(range(8)), (6,), tuple(range(2304, 2688))),
        (
            16,
            4,
            13,
            4,
            (1, 5, 9, 13),
            (12, 13, 14, 15),
            tuple(range(1248, 1344)),
        ),
    ],
)
def test_gmoe_topology_and_expert_ownership(
    ep_size, groups, rank, egp_size, exchange_group, egp_group, owned
) -> None:
    ep_group = tuple(range(ep_size))
    topology = gmoe_topology(ep_group=ep_group, rank=rank, num_groups=groups)

    assert topology.ep_group == ep_group
    assert topology.ep_size == ep_size
    assert topology.ep_rank == rank
    assert topology.egp_size == egp_size
    assert topology.egp_rank == egp_group.index(rank)
    assert topology.num_groups == groups
    assert topology.exchange_group == exchange_group
    assert topology.egp_group == egp_group
    assert gmoe_local_expert_ids(topology=topology, num_experts_per_group=384) == owned


def test_shared_flash_local_loader_uses_group_aware_ownership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tokenspeed.runtime.configs.flash_kda_config import FLASHLocalConfig
    from tokenspeed.runtime.distributed.mapping import Mapping
    from tokenspeed.runtime.models import flash_kda

    monkeypatch.setattr(
        flash_kda, "flash_local_prefers_packed_projections", lambda: True
    )
    config = FLASHLocalConfig.from_dict(lite_config_dict())
    parallel = Mapping(
        rank=1,
        world_size=8,
        attn_tp_size=1,
        attn_cp_size=8,
        dense_tp_size=1,
        moe_tp_size=1,
        moe_ep_size=8,
        linear_attn_tp_size=8,
        mla_weight_tp_size=1,
    )
    config.strict_checkpoint_layout = False
    config.use_over_embedding = False
    model = flash_kda.FLASHLocalForCausalLM(config, parallel)
    prefix = "model.layers.0.mlp.experts.5"

    model.load_weights(
        [
            (
                f"{prefix}.gate_proj.weight",
                torch.full((16, 24), 1, dtype=torch.bfloat16),
            ),
            (
                f"{prefix}.up_proj.weight",
                torch.full((16, 24), 2, dtype=torch.bfloat16),
            ),
            (
                f"{prefix}.down_proj.weight",
                torch.full((24, 16), 3, dtype=torch.bfloat16),
            ),
        ]
    )

    experts = model.model.layers[0].moe.experts
    assert model.model.layers[0].moe.local_expert_ids == (4, 5, 6, 7)
    assert torch.equal(
        experts.w13_weight[0, :16], torch.ones_like(experts.w13_weight[0, :16])
    )
    assert torch.equal(
        experts.w13_weight[0, 16:], torch.full_like(experts.w13_weight[0, 16:], 2)
    )
    assert torch.equal(experts.w2_weight[0], torch.full_like(experts.w2_weight[0], 3))


def test_lite_grouped_moe_empty_and_distributed_context_contract() -> None:
    config = FLASHLocalConfig.from_dict(lite_config_dict())
    with pytest.raises(ValueError, match="EP >= G"):
        GroupAwareFlashLocalMoE(config, mapping())

    distributed = GroupAwareFlashLocalMoE(config, mapping(8, rank=3))
    empty = torch.empty(0, 96, dtype=torch.bfloat16)
    assert distributed(empty).shape == empty.shape

    with pytest.raises(ValueError, match="ForwardContext"):
        distributed(torch.zeros(1, 96, dtype=torch.bfloat16))


def _stub_comm_ops(monkeypatch: pytest.MonkeyPatch, **operations):
    distributed = ModuleType("tokenspeed.runtime.distributed")
    comm_ops = ModuleType("tokenspeed.runtime.distributed.comm_ops")
    for name, operation in operations.items():
        setattr(comm_ops, name, operation)
    distributed.comm_ops = comm_ops
    monkeypatch.setitem(sys.modules, distributed.__name__, distributed)
    monkeypatch.setitem(sys.modules, comm_ops.__name__, comm_ops)


def _local_expert_oracle(
    layer: GroupAwareFlashLocalMoE,
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    output = torch.zeros_like(hidden_states)
    local_count = layer.experts.num_local_experts
    expert_lo = layer.topology.egp_rank * local_count
    expert_hi = expert_lo + local_count
    for token_id in range(hidden_states.shape[0]):
        for route_id in range(topk_ids.shape[1]):
            expert_id = int(topk_ids[token_id, route_id])
            if not expert_lo <= expert_id < expert_hi:
                continue
            local_id = expert_id - expert_lo
            gate_up = F.linear(
                hidden_states[token_id], layer.experts.w13_weight[local_id]
            )
            gate, up = gate_up.chunk(2)
            expert_output = F.linear(
                F.silu(gate) * up, layer.experts.w2_weight[local_id]
            )
            output[token_id] += expert_output * topk_weights[token_id, route_id].to(
                output.dtype
            )
    return output


@pytest.mark.parametrize(
    ("role", "mode"),
    [("prefill", ForwardMode.EXTEND), ("decode", ForwardMode.DECODE)],
)
def test_lite_gmoe_collectives_and_identity_math(
    monkeypatch: pytest.MonkeyPatch,
    role: str,
    mode: ForwardMode,
) -> None:
    torch.manual_seed(13)
    config = FLASHLocalConfig.from_dict(lite_config_dict())
    layer = GroupAwareFlashLocalMoE(config, mapping(8, rank=0, role=role))
    for parameter in layer.parameters():
        parameter.data.uniform_(-0.05, 0.05)
    hidden = torch.randn(2, 96, dtype=torch.bfloat16)
    route_weights = torch.tensor([0.25, 0.5], dtype=torch.float32).repeat(16, 1)
    route_ids = torch.tensor([0, 8], dtype=torch.int32).repeat(16, 1)
    monkeypatch.setattr(
        layer,
        "_route",
        lambda _received: (route_weights, route_ids),
    )
    layer._moe_plan = {"test": True}
    monkeypatch.setattr(layer.shared_experts, "forward", lambda x: torch.zeros_like(x))
    calls = []

    def all_to_all(output, tensor, group):
        calls.append(("a2a", group, tuple(tensor.shape)))
        output.copy_(tensor)

    def token_gather(tensor, group, split):
        calls.append(("tag", group, tuple(tensor.shape), tuple(split)))
        return torch.cat((tensor, tensor), dim=0)

    def token_scatter(tensor, group, split):
        calls.append(("trs", group, tuple(tensor.shape), tuple(split)))
        return tensor[: split[0]]

    _stub_comm_ops(
        monkeypatch,
        all_to_all_single=all_to_all,
        token_all_gather=token_gather,
        token_reduce_scatter=token_scatter,
    )
    import tokenspeed_kernel

    def moe_apply(_plan, x, _experts, _router_logits, **kwargs):
        return _local_expert_oracle(
            layer, x, kwargs["topk_weights"], kwargs["topk_ids"]
        )

    monkeypatch.setattr(tokenspeed_kernel, "moe_apply", moe_apply)

    actual = layer(
        hidden,
        ctx=SimpleNamespace(
            forward_mode=mode,
            global_num_tokens=[2] * 8,
        ),
    )
    projected = F.linear(hidden, layer.proj_input.weight)
    projected_fp32 = projected.float()
    grouped = (
        (
            projected_fp32
            * torch.rsqrt(
                projected_fp32.square().mean(dim=-1, keepdim=True) + config.rms_norm_eps
            )
            * layer.norm.weight.float()
        ).to(projected.dtype)
        * config.grouped_moe_norm_scale
    ).view(2, 4, 24)
    received = grouped.permute(1, 0, 2).contiguous().flatten(0, 1)
    routed = (
        _local_expert_oracle(layer, received, route_weights[:8], route_ids[:8])
        + received * 0.5
    )
    restored = routed.view(4, 2, 24).permute(1, 0, 2).flatten(1)
    expected = F.linear(restored, layer.proj_output.weight)

    assert torch.allclose(actual, expected, atol=2e-2, rtol=2e-2)
    assert calls == [
        ("a2a", (0, 2, 4, 6), (8, 24)),
        ("tag", (0, 1), (8, 24), (8, 8)),
        ("trs", (0, 1), (16, 24), (8, 8)),
        ("a2a", (0, 2, 4, 6), (8, 24)),
    ]


def test_lite_grouped_moe_meta_shapes_and_loader() -> None:
    config = FLASHLocalConfig.from_dict(lite_config_dict())
    world_size = 8
    parallel = Mapping(
        rank=0,
        world_size=world_size,
        attn_tp_size=1,
        attn_cp_size=world_size,
        attn_dp_size=1,
        dense_tp_size=1,
        dense_dp_size=world_size,
        moe_tp_size=1,
        moe_ep_size=world_size,
        moe_dp_size=1,
        linear_attn_tp_size=world_size,
        mla_weight_tp_size=1,
    )
    with torch.device("meta"):
        model = FLASHLocalForCausalLM(
            config,
            parallel,
            oe_table_placement="host",
        )
    experts = model.model.layers[0].moe.experts
    local_experts = (
        config.n_routed_experts // model.model.layers[0].moe.topology.egp_size
    )
    assert experts.w13_weight.shape == (local_experts, 32, 24)
    assert experts.w2_weight.shape == (local_experts, 24, 16)
    assert experts.w13_weight.is_meta and experts.w2_weight.is_meta

    model.load_weights(weights(model.checkpoint_layout))
    assert experts.w13_weight.is_meta and experts.w2_weight.is_meta
