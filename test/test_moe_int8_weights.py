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

import pytest
import torch
from torch import nn

from tokenspeed.runtime.layers.moe import (
    ExpertCheckpointSchema,
    build_moe_checkpoint_loader,
)
from tokenspeed.runtime.layers.moe.types import MoELayerSpec
from tokenspeed.runtime.layers.moe.weights.int8 import create_int8_weight_pair


def test_int8_moe_weight_loaders_pack_projections_scales_and_smooth_scales() -> None:
    spec = MoELayerSpec(
        top_k=2,
        num_experts=2,
        num_local_experts=2,
        hidden_size=4,
        intermediate_size=3,
        activation="silu",
        tp_rank=0,
        tp_size=1,
        ep_rank=0,
        ep_size=1,
    )
    layer = nn.Module()
    create_int8_weight_pair(spec, layer, smooth_quant=True)
    prefix = "model.layers.0.moe."
    loader = build_moe_checkpoint_loader(
        params_dict={
            prefix + "experts." + name: param
            for name, param in layer.named_parameters()
        },
        expert_schema=ExpertCheckpointSchema(),
        num_experts=2,
    )

    gate = torch.arange(12, dtype=torch.int8).view(3, 4)
    up = gate + 20
    down = torch.arange(12, dtype=torch.int8).view(4, 3)
    loader.load(
        f"{prefix}experts.0.gate_proj.weight",
        gate,
    )
    loader.load(
        f"{prefix}experts.0.up_proj.weight",
        up,
    )
    loader.load(
        f"{prefix}experts.0.down_proj.weight",
        down,
    )

    gate_scale = torch.arange(1, 4, dtype=torch.bfloat16).view(3, 1)
    up_scale = gate_scale + 3
    down_scale = torch.arange(1, 5, dtype=torch.bfloat16).view(4, 1)
    loader.load(
        f"{prefix}experts.0.gate_proj.weight_scale",
        gate_scale,
    )
    loader.load(
        f"{prefix}experts.0.up_proj.weight_scale",
        up_scale,
    )
    loader.load(
        f"{prefix}experts.0.down_proj.weight_scale",
        down_scale,
    )

    smooth13 = torch.arange(1, 5, dtype=torch.bfloat16)
    smooth2 = torch.arange(1, 4, dtype=torch.bfloat16)
    loader.load(
        f"{prefix}experts.0.gate_proj.smooth_scale",
        smooth13,
    )
    loader.load(
        f"{prefix}experts.0.down_proj.smooth_scale",
        smooth2,
    )

    assert torch.equal(layer.w13_weight[0, :3], gate)
    assert torch.equal(layer.w13_weight[0, 3:], up)
    assert torch.equal(layer.w2_weight[0], down)
    assert torch.equal(layer.w13_weight_scale[0, :3], gate_scale.flatten())
    assert torch.equal(layer.w13_weight_scale[0, 3:], up_scale.flatten())
    assert torch.equal(layer.w2_weight_scale[0], down_scale.flatten())
    assert torch.equal(layer.w13_smooth_scale[0], smooth13.float())
    assert torch.equal(layer.w2_smooth_scale[0], smooth2.float())


@pytest.mark.parametrize("enabled", (False, True))
def test_int8_moe_optional_smooth_parameters_keep_identity_defaults(
    enabled: bool,
) -> None:
    spec = MoELayerSpec(
        top_k=2,
        num_experts=2,
        num_local_experts=2,
        hidden_size=4,
        intermediate_size=3,
        activation="silu",
        tp_rank=0,
        tp_size=1,
        ep_rank=0,
        ep_size=1,
    )
    layer = nn.Module()
    create_int8_weight_pair(spec, layer, smooth_quant=enabled)
    if not enabled:
        assert not hasattr(layer, "w13_smooth_scale")
        assert not hasattr(layer, "w2_smooth_scale")
        return
    torch.testing.assert_close(layer.w13_smooth_scale, torch.ones(2, 4))
    torch.testing.assert_close(layer.w2_smooth_scale, torch.ones(2, 3))
    layer.w13_smooth_scale.weight_loader(
        layer.w13_smooth_scale, torch.full((4,), 2.0), "w1", 1
    )
    torch.testing.assert_close(layer.w13_smooth_scale[1], torch.full((4,), 2.0))
    # A missing other projection/expert retains identity, not uninitialized data.
    torch.testing.assert_close(layer.w13_smooth_scale[0], torch.ones(4))
    torch.testing.assert_close(layer.w2_smooth_scale, torch.ones(2, 3))
