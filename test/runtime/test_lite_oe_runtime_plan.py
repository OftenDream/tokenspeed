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
from types import SimpleNamespace

import pytest

from tokenspeed.runtime.configs.model_config import resolve_oe_runtime_plan
from tokenspeed.runtime.utils.server_args import ServerArgs

_CAPABILITIES = {
    "cuda": {"device": "runtime-full-history"},
    "npu": {
        "device": "runtime-full-history",
        "host": "runtime-full-history",
    },
}


@pytest.mark.parametrize(
    ("device", "requested", "expected"),
    [
        ("cuda", "auto", ("device", "runtime-full-history")),
        ("cuda", "device", ("device", "runtime-full-history")),
        ("npu", "auto", ("host", "runtime-full-history")),
        ("npu", "host", ("host", "runtime-full-history")),
        ("npu", "device", ("device", "runtime-full-history")),
    ],
)
def test_oe_runtime_plan_resolves_declared_pairs(
    device: str,
    requested: str,
    expected: tuple[str, str],
) -> None:
    assert (
        resolve_oe_runtime_plan(
            use_over_embedding=True,
            requested_placement=requested,
            device=device,
            capabilities=_CAPABILITIES,
        )
        == expected
    )


@pytest.mark.parametrize(("device", "requested"), [("cuda", "host")])
def test_oe_runtime_plan_never_falls_back_explicit_placement(
    device: str,
    requested: str,
) -> None:
    with pytest.raises(ValueError, match="not supported"):
        resolve_oe_runtime_plan(
            use_over_embedding=True,
            requested_placement=requested,
            device=device,
            capabilities=_CAPABILITIES,
        )


def test_non_oe_auto_is_noop_and_explicit_is_rejected() -> None:
    assert resolve_oe_runtime_plan(
        use_over_embedding=False,
        requested_placement="auto",
        device="cuda",
        capabilities=_CAPABILITIES,
    ) == (None, None)
    with pytest.raises(ValueError, match="only valid for an OE checkpoint"):
        resolve_oe_runtime_plan(
            use_over_embedding=False,
            requested_placement="device",
            device="cuda",
            capabilities=_CAPABILITIES,
        )


def test_oe_table_placement_cli_contract() -> None:
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    assert parser.parse_args(["--model", "x"]).oe_table_placement == "auto"
    for placement in ("host", "device"):
        assert (
            parser.parse_args(
                ["--model", "x", "--oe-table-placement", placement]
            ).oe_table_placement
            == placement
        )
    with pytest.raises(SystemExit):
        parser.parse_args(["--model", "x", "--oe-table-placement", "invalid"])


def test_model_loader_resolves_and_passes_class_declared_plan(monkeypatch) -> None:
    from tokenspeed.runtime.model_loader import loader

    class Model:
        supports_oe_table_placement = True
        oe_runtime_capabilities = _CAPABILITIES

        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    model_config = SimpleNamespace(
        hf_config=object(),
        hf_text_config=SimpleNamespace(use_over_embedding=True),
        mapping=object(),
        device="npu",
        _oe_table_placement_request="auto",
        quantization=None,
        is_multimodal=False,
    )
    monkeypatch.setattr(
        loader,
        "get_model_architecture",
        lambda _config: (Model, "Model"),
    )
    monkeypatch.setattr(loader, "_get_quantization_config", lambda *_args: None)

    model = loader._initialize_model(model_config, SimpleNamespace())

    assert model_config.oe_table_placement == "host"
    assert model_config.oe_state_provider == "runtime-full-history"
    assert model.kwargs["oe_table_placement"] == "host"
