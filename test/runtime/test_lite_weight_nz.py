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

from test.runtime.test_lite_model_loader import lite_config_dict, mapping
from unittest import mock

import torch
import torch.nn.functional as F

from tokenspeed.runtime.configs.flash_kda_config import FLASHLocalConfig
from tokenspeed.runtime.models.flash_kda import FLASHLocalForCausalLM
from tokenspeed.runtime.models.flash_local_attention import WeightNZReplicatedLinear
from tokenspeed.runtime.models.flash_local_moe import PackedWeight as _Weight


def test_weight_nz_server_arg_defaults_off_and_propagates_decode_role():
    try:
        from tokenspeed.runtime.utils.env import (
            global_server_args_dict,
            global_server_args_dict_update,
        )
        from tokenspeed.runtime.utils.server_args import prepare_server_args
    except RuntimeError as error:
        if "requires an NVIDIA CUDA, AMD ROCm, or Ascend NPU device" in str(error):
            import pytest

            pytest.skip("full runtime arguments require an accelerator platform")
        raise

    assert not prepare_server_args(["--model", "x"]).npu_enable_weight_nz
    args = prepare_server_args(
        ["--model", "x", "--npu-enable-weight-nz", "--disaggregation-mode", "decode"]
    )
    snapshot = dict(global_server_args_dict)
    try:
        with mock.patch("tokenspeed.runtime.utils.env.pdl_enabled"):
            global_server_args_dict_update(args)
        assert global_server_args_dict["npu_enable_weight_nz"] is True
        assert global_server_args_dict["disaggregation_mode"] == "decode"
    finally:
        global_server_args_dict.clear()
        global_server_args_dict.update(snapshot)


def test_lite_weight_nz_whitelist_is_exact():
    config = FLASHLocalConfig.from_dict(lite_config_dict())
    model = FLASHLocalForCausalLM(
        config,
        mapping(8, rank=0, role="decode"),
        oe_table_placement="host",
    )
    marked = {
        name: module.weight_nz
        for name, module in model.named_modules()
        if isinstance(module, (_Weight, WeightNZReplicatedLinear))
        and module.weight_nz is not None
    }
    expected = {}
    for layer_id in range(4):
        prefix = f"model.layers.{layer_id}"
        expected[f"{prefix}.moe.proj_output"] = "standard"
        expected[f"{prefix}.moe.shared_experts.down_proj"] = "standard"
        if layer_id < 3:
            expected[f"{prefix}.self_attn.o_proj"] = "standard"
        else:
            expected.update(
                {
                    f"{prefix}.self_attn.q_a_proj": "transposed",
                    f"{prefix}.self_attn.q_b_proj": "transposed",
                    f"{prefix}.self_attn.kv_a_proj_with_mqa": "transposed",
                    f"{prefix}.self_attn.o_proj": "standard",
                }
            )
    assert marked == expected

    target = FLASHLocalConfig()
    assert (
        len(target.linear_layer_ids)
        + 4 * len(target.full_attention_layer_ids)
        + 2 * target.num_hidden_layers
        == 105
    )


def test_weight_nz_cpu_path_preserves_canonical_linear():
    module = _Weight((3, 4), torch.bfloat16, weight_nz="transposed")
    module.weight.data.copy_(
        torch.arange(12, dtype=torch.bfloat16).reshape_as(module.weight)
    )
    source = module.weight.detach().clone()
    hidden = torch.randn(2, 4, dtype=torch.bfloat16)
    module.process_weights_after_loading()

    assert not module._weight_nz_prepared
    assert not module._weight_nz_transposed
    assert torch.equal(module.weight, source)
    torch.testing.assert_close(module(hidden), F.linear(hidden, source), rtol=0, atol=0)
