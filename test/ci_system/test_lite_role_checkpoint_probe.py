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

import argparse
import socket
import sys
from test.runtime.test_lite_model_loader import mapping

import pytest
import torch
from lite_role_checkpoint_probe import (
    EXPECTED_PARAMETER_BYTES,
    expected_derived_buffer_bytes,
    parameter_ledger,
    role_command,
    role_mapping_kwargs,
)

from tokenspeed.runtime.configs.flash_kda_config import FLASHLocalConfig
from tokenspeed.runtime.models.flash_kda import FLASHLocalForCausalLM


def _real_config() -> FLASHLocalConfig:
    return FLASHLocalConfig(
        mla_scale_q_lora=True,
        mla_scale_kv_lora=True,
        ngram_vocab_size_ratio=28.8,
        emb_neighbor_num=4,
        emb_split_num=4,
        ngram_exclude_sp_token=True,
        special_token_scope="0:4,36:55",
        ngram_fix_normalize_factor=True,
    )


@pytest.mark.parametrize("role", ["prefill", "decode"])
def test_role_mapping_and_exact_static_parameter_bytes(role) -> None:
    kwargs = role_mapping_kwargs(role)
    assert kwargs["linear_attn_tp_size"] == 8
    assert kwargs["mla_weight_tp_size"] == 1
    assert kwargs["moe_ep_size"] == 8
    assert (kwargs["attn_cp_size"], kwargs["attn_dp_size"]) == (
        (8, 1) if role == "prefill" else (1, 8)
    )

    with torch.device("meta"):
        model = FLASHLocalForCausalLM(
            _real_config(),
            mapping(8, role=role),
            oe_table_placement="host",
        )
    ledger = parameter_ledger(model)

    assert ledger["npu_parameter_bytes"] == EXPECTED_PARAMETER_BYTES[role]
    assert ledger["host_oe_bytes"] == 0
    assert ledger["by_category"]["kda"]["bytes"] == 369_143_376
    assert ledger["by_category"]["mla"]["bytes"] == 634_023_936
    assert ledger["by_category"]["oe_projection"]["bytes"] == 2_359_296
    assert ledger["by_category"]["moe_experts"]["bytes"] == 12_683_575_296
    assert ledger["by_category"]["moe_experts"]["bytes"] + ledger["by_category"][
        "grouped_moe"
    ]["bytes"] == (15_469_475_840 if role == "prefill" else 13_157_365_760)
    assert expected_derived_buffer_bytes(model.config, 8) == 58_978_304


def test_tiny_host_oe_is_excluded_from_accelerator_parameter_total() -> None:
    config = _real_config()
    with torch.device("meta"):
        model = FLASHLocalForCausalLM(
            config,
            mapping(8, role="prefill"),
            oe_table_placement="host",
        )
    model.model.embed_tokens.oe_tables[0].data = torch.empty(
        (13, 256), dtype=torch.bfloat16
    )

    ledger = parameter_ledger(model)

    assert ledger["host_oe_bytes"] == 13 * 256 * 2
    assert ledger["by_category"]["host_oe"]["devices"] == ["cpu"]


def test_role_command_uses_two_independent_eight_rank_worlds() -> None:
    command = role_command(
        role="decode",
        checkpoint="/checkpoint",
        output_root="/output",
        barrier_root="/output/.barrier",
        port=29702,
        device_offset=8,
        timeout=10,
        touch_mib=1,
        check_finite=True,
    )

    assert command[:3] == [sys.executable, "-m", "torch.distributed.run"]
    assert command[command.index("--nproc-per-node") + 1] == "8"
    assert command[command.index("--device-offset") + 1] == "8"
    assert command[command.index("--role") + 1] == "decode"
    assert command[-1] == "--check-finite"


def test_unknown_role_fails_before_hardware_initialization() -> None:
    with pytest.raises(ValueError, match="Unknown Lite role"):
        role_mapping_kwargs("aggregate")


def test_port_check_matches_tcpstore_all_interface_listener() -> None:
    from lite_role_checkpoint_probe import _port_is_free

    with socket.socket() as listener:
        listener.bind(("", 0))
        port = listener.getsockname()[1]
        assert not _port_is_free(port)


def test_controller_rejects_overlapping_device_ranges(tmp_path) -> None:
    from lite_role_checkpoint_probe import _validate_controller_args

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    (checkpoint / "model.safetensors.index.json").write_text("{}")
    args = argparse.Namespace(
        checkpoint=str(checkpoint),
        output_root=str(tmp_path / "output"),
        prefill_device_offset=0,
        decode_device_offset=7,
        prefill_port=29701,
        decode_port=29702,
    )

    with pytest.raises(ValueError, match="must not overlap"):
        _validate_controller_args(args)
