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

import importlib.util
import sys
from collections import namedtuple
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

import pytest
import torch
import torch.nn.functional as F

from tokenspeed.runtime.models.flash_local_attention import PackedFLASHLocalKDA


def _load_reference():
    """Load the pure-Torch leaf without weakening the package's device guard."""
    path = (
        Path(__file__).parents[2]
        / "tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/kda_reference.py"
    )
    result_module = ModuleType("tokenspeed_kernel.ops.attention.kda_utils")
    result_module.KdaPrefillResult = namedtuple(
        "KdaPrefillResult", ("out", "final_state")
    )
    spec = importlib.util.spec_from_file_location("lite_kda_reference_test", path)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(
        sys.modules,
        {"tokenspeed_kernel.ops.attention.kda_utils": result_module},
    ):
        spec.loader.exec_module(module)
    return module


REFERENCE = _load_reference()


def _oracle_conv(projected, weight, state, reads, writes, boundaries):
    output = torch.zeros_like(projected)
    final = state.clone()
    updates = []
    for start, end, read, write in zip(boundaries, boundaries[1:], reads, writes):
        if start == end or read < 0:
            continue
        history = state[read].clone()
        for token in range(start, end):
            window = torch.cat((history, projected[token, :, None]), dim=-1)
            output[token] = F.silu((window * weight).sum(-1))
            history = window[:, 1:]
        updates.append((write, history))
    for write, history in updates:
        final[write] = history
    return output, final


def _oracle_kda(q, k, v, raw_gate, beta_logits, a_log, dt_bias, state, bounds):
    q = F.normalize(q.float(), p=2, dim=-1)
    k = F.normalize(k.float(), p=2, dim=-1)
    v = v.float()
    scalar_beta = beta_logits.ndim == q.ndim - 1
    if scalar_beta:
        beta = torch.sigmoid(beta_logits.float())
    else:
        beta_scale = torch.sqrt(torch.sigmoid(beta_logits.float()) + 1e-10)
        k = k * beta_scale
        v = v * beta_scale
        beta = None
    gate = -5.0 * torch.sigmoid(
        a_log.float().exp().reshape(-1, 1)
        * (raw_gate.float() + dt_bias.float().reshape(q.shape[-2:]))
    )

    output = torch.zeros_like(v)
    final = state.clone()
    for request, (start, end) in enumerate(zip(bounds, bounds[1:])):
        current = state[request].clone()
        for token in range(start, end):
            current = current * gate[token].exp().unsqueeze(-1)
            delta = v[token] - torch.matmul(k[token].unsqueeze(-2), current).squeeze(-2)
            if beta is not None:
                delta = delta * beta[token].unsqueeze(-1)
            current = current + k[token].unsqueeze(-1) * delta.unsqueeze(-2)
            output[token] = torch.matmul(
                (q[token] * q.shape[-1] ** -0.5).unsqueeze(-2), current
            ).squeeze(-2)
        final[request] = current
    return output, final


def _kda_inputs(tokens=4, heads=2, dim=4, dtype=torch.float32, device="cpu"):
    generator = torch.Generator(device=device).manual_seed(31)
    shape = (1, tokens, heads, dim)
    return {
        "q": torch.randn(shape, generator=generator, dtype=dtype, device=device),
        "k": torch.randn(shape, generator=generator, dtype=dtype, device=device),
        "v": torch.randn(shape, generator=generator, dtype=dtype, device=device),
        "g_raw": torch.randn(shape, generator=generator, dtype=dtype, device=device),
        "beta_logits": torch.randn(
            shape, generator=generator, dtype=dtype, device=device
        ),
        "A_log": torch.randn(heads, generator=generator, device=device),
        "dt_bias": torch.randn(heads * dim, generator=generator, device=device),
    }


def test_packed_causal_conv_publishes_independent_output_slots():
    torch.manual_seed(7)
    projected = torch.randn(4, 3)
    weight = torch.randn(3, 3)
    state = torch.randn(3, 3, 2)
    reads = torch.tensor([0, 1], dtype=torch.int32)
    writes = torch.tensor([2, 0], dtype=torch.int32)
    boundaries = torch.tensor([0, 1, 4], dtype=torch.int64)
    expected_output, expected_state = _oracle_conv(
        projected, weight, state, reads.tolist(), writes.tolist(), boundaries.tolist()
    )

    actual_state = state.clone()
    actual_output = REFERENCE.torch_kda_causal_conv1d(
        projected,
        weight,
        actual_state,
        reads,
        writes,
        boundaries,
    )

    torch.testing.assert_close(actual_output, expected_output)
    torch.testing.assert_close(actual_state, expected_state)


@pytest.mark.parametrize("featurewise", [False, True])
def test_packed_kda_matches_independent_scalar_and_featurewise_oracle(featurewise):
    inputs = _kda_inputs()
    if not featurewise:
        inputs["beta_logits"] = inputs["beta_logits"][..., 0]
    initial_state = torch.randn(3, 2, 4, 4)
    boundaries = torch.tensor([0, 2, 2, 4], dtype=torch.int64)
    expected_output, expected_state = _oracle_kda(
        *(inputs[name][0] for name in ("q", "k", "v", "g_raw", "beta_logits")),
        inputs["A_log"],
        inputs["dt_bias"],
        initial_state,
        boundaries.tolist(),
    )

    result = REFERENCE.torch_kda_paged_prefill(
        **inputs,
        initial_state=initial_state,
        cu_seqlens=boundaries,
        cu_seqlens_cpu=boundaries,
        lower_bound=-5.0,
    )

    torch.testing.assert_close(result.out[0], expected_output)
    torch.testing.assert_close(result.final_state, expected_state)


def test_four_token_prefill_matches_sequential_decode_state_reuse():
    inputs = _kda_inputs()
    initial_state = torch.randn(1, 2, 4, 4)
    boundaries = torch.tensor([0, 4], dtype=torch.int64)
    prefill = REFERENCE.torch_kda_paged_prefill(
        **inputs,
        initial_state=initial_state,
        cu_seqlens=boundaries,
        cu_seqlens_cpu=boundaries,
        lower_bound=-5.0,
    )

    state_pool = initial_state.clone()
    decoded = []
    for token in range(4):
        step = {
            name: value[:, token : token + 1]
            for name, value in inputs.items()
            if value.ndim == 4
        }
        step.update({name: inputs[name] for name in ("A_log", "dt_bias")})
        decoded.append(
            REFERENCE.torch_kda_paged_decode(
                **step,
                state_pool=state_pool,
                read_indices=torch.tensor([0]),
                write_indices=torch.tensor([0]),
                cu_seqlens=torch.tensor([0, 1]),
                lower_bound=-5.0,
            )
        )

    torch.testing.assert_close(torch.cat(decoded, dim=1), prefill.out)
    torch.testing.assert_close(state_pool, prefill.final_state)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("q", float("inf")),
        ("beta_logits", float("nan")),
        ("A_log", float("inf")),
        ("dt_bias", float("nan")),
    ],
)
def test_kda_reference_rejects_nonfinite_inputs(name, value):
    inputs = _kda_inputs()
    inputs[name].reshape(-1)[0] = value
    boundaries = torch.tensor([0, 4], dtype=torch.int64)

    with pytest.raises(ValueError, match="NaN or Inf"):
        REFERENCE.torch_kda_paged_prefill(
            **inputs,
            initial_state=torch.zeros(1, 2, 4, 4),
            cu_seqlens=boundaries,
            cu_seqlens_cpu=boundaries,
            lower_bound=-5.0,
        )


def test_lite_kda_projects_featurewise_beta_and_applies_output_gate():
    config = SimpleNamespace(
        linear_num_heads=2,
        linear_head_dim=3,
        linear_conv_size=3,
        hidden_size=6,
        rms_norm_eps=1e-5,
        linear_attn_config={"gate_lower_bound": -5.0},
    )
    mapping = SimpleNamespace(linear_attn=SimpleNamespace(tp_size=1))
    layer = PackedFLASHLocalKDA(config, mapping, layer_id=2)
    with torch.no_grad():
        for index, parameter in enumerate(layer.parameters(), start=1):
            values = torch.arange(parameter.numel(), dtype=torch.float32).reshape(
                parameter.shape
            )
            parameter.copy_(((values % 7) - 3).mul_(0.05 * index))

    hidden = torch.tensor(
        [[-1.0, 0.5, 1.0, -0.5, 0.25, 0.75], [0.2, -0.4, 0.6, -0.8, 1.0, -1.2]],
        dtype=torch.bfloat16,
    )
    core = torch.linspace(-0.8, 0.9, 12, dtype=torch.bfloat16).reshape(2, 2, 3)

    class Backend:
        kwargs = None

        def forward(self, **kwargs):
            self.kwargs = kwargs
            return core.clone()

    backend = Backend()
    ctx = SimpleNamespace(
        attn_backend=backend,
        token_to_kv_pool=object(),
        forward_mode=object(),
        bs=2,
    )
    output = layer(torch.arange(2), hidden, ctx, torch.tensor([0, 1]))

    assert layer.conv_weights is backend.kwargs["conv_weights"]
    cached_conv_weights = layer.conv_weights
    layer.process_weights_after_loading()
    assert layer.conv_weights is cached_conv_weights
    projection = layer.input_projection.local_projection
    low_rank = layer.input_projection.head_dim
    merged_weight = layer.input_projection.weight
    beta_down = F.linear(
        hidden,
        merged_weight[4 * projection + low_rank : 4 * projection + 2 * low_rank],
    )
    beta_expected = F.linear(beta_down, layer.beta_b_proj.weight)
    torch.testing.assert_close(backend.kwargs["beta_raw"], beta_expected)
    assert backend.kwargs["beta_raw"].shape == (2, 2 * 3)
    assert backend.kwargs["output_gate"] is None

    gate = torch.sigmoid(
        F.linear(hidden, merged_weight[3 * projection : 4 * projection]).float()
    ).to(torch.bfloat16)
    normalized = core.float() * torch.rsqrt(
        core.float().square().mean(-1, keepdim=True) + config.rms_norm_eps
    )
    normalized = (normalized * layer.o_norm.weight.float()).to(torch.bfloat16)
    expected = F.linear(
        (normalized * gate.reshape_as(normalized)).flatten(1), layer.o_proj.weight
    )
    torch.testing.assert_close(output, expected)


def _npu_available():
    if importlib.util.find_spec("torch_npu") is None:
        return False
    import torch_npu  # noqa: F401

    return torch.npu.is_available()


@pytest.mark.parametrize("requested", ["auto", "fla", "flashkda", "cutedsl_kda"])
@pytest.mark.skipif(not _npu_available(), reason="requires an Ascend NPU")
def test_ascend_kda_backend_policy_stays_registry_driven(monkeypatch, requested):
    from tokenspeed.runtime.layers.attention import registry

    monkeypatch.setattr(
        registry,
        "current_platform",
        lambda: SimpleNamespace(is_amd=False, is_npu=True),
    )

    assert registry._resolve_kda_backend(requested) == "auto"


def _lite_output_epilogue_oracle(core, gate, weight, eps, heads, dim):
    shaped = core.reshape(-1, heads, dim).float()
    output = shaped * torch.rsqrt(shaped.square().mean(dim=-1, keepdim=True) + eps)
    return (
        (output * weight.float() * torch.sigmoid(gate.reshape_as(shaped).float()))
        .flatten(1)
        .to(core.dtype)
    )


@pytest.mark.parametrize("enable_pdl", [False, True])
def test_separate_lite_kda_passes_output_epilogue_pdl_policy(enable_pdl):
    from tokenspeed.runtime.models import flash_kda

    hidden = torch.zeros(2, 4)
    core = torch.ones(2, 1, 4)
    projection = SimpleNamespace(weight=torch.ones(4, 4))
    layer = SimpleNamespace(
        local_num_heads=1,
        head_dim=4,
        _project_qkv=lambda x: torch.cat((x, x, x), dim=-1),
        is_fgbkda=False,
        g_proj=lambda x: x,
        b_proj=lambda x: (x[:, :1], None),
        f_proj=SimpleNamespace(fc1=lambda x: (x, None), fc2=projection),
        conv_weights=torch.ones(12, 4),
        proj=4,
        mapping=SimpleNamespace(linear_attn=SimpleNamespace(tp_size=1)),
        A_log=torch.zeros(1),
        dt_bias=torch.zeros(4),
        gate_lower_bound=-5.0,
        layer_id=0,
        o_norm=SimpleNamespace(weight=torch.ones(4), variance_epsilon=1e-6),
        o_proj=lambda x: (x, None),
    )
    ctx = SimpleNamespace(
        attn_backend=SimpleNamespace(forward=lambda **kwargs: core),
        token_to_kv_pool=None,
        forward_mode=None,
        bs=2,
    )
    with (
        mock.patch.object(flash_kda, "_pdl_enabled", return_value=enable_pdl),
        mock.patch.object(
            flash_kda,
            "rmsnorm_gated_sigmoid",
            autospec=True,
            return_value=core.flatten(1),
        ) as epilogue,
    ):
        flash_kda.SeperateFLASHLocal.forward(layer, None, hidden, ctx, None)
    assert epilogue.call_args.kwargs["enable_pdl"] is enable_pdl


@pytest.mark.parametrize("tokens", [1, 2, 8, 32, 256, 1024])
@pytest.mark.skipif(not _npu_available(), reason="requires an Ascend NPU")
def test_ascend_lite_output_epilogue_matches_single_rounding_oracle(tokens):
    from tokenspeed_kernel.ops.activation.triton import rmsnorm_gated_sigmoid

    torch.npu.set_device(0)
    torch.manual_seed(1307 + tokens)
    heads, dim, width, eps = 4, 128, 512, 1e-6
    core = (torch.randn(tokens, width, device="npu") * 0.3).to(torch.bfloat16)
    packed_gate = (torch.randn(tokens, width + 192, device="npu") * 0.4).to(
        torch.bfloat16
    )
    gate = packed_gate[:, 64 : 64 + width]
    weight = (torch.randn(dim, device="npu") * 0.2 + 1).to(torch.bfloat16)

    actual = rmsnorm_gated_sigmoid(
        core, gate, weight, eps, heads, dim, enable_pdl=False
    )
    expected = _lite_output_epilogue_oracle(core, gate, weight, eps, heads, dim)
    torch.npu.synchronize()

    difference = actual.float() - expected.float()
    relative_l2 = difference.norm() / expected.float().norm().clamp_min(1e-12)
    assert difference.abs().max().item() <= 0.004
    assert relative_l2.item() <= 1e-4
    assert torch.isfinite(actual).all()
    assert gate.stride() == (width + 192, 1)


@pytest.mark.skipif(not _npu_available(), reason="requires an Ascend NPU")
def test_ascend_lite_model_uses_fused_output_epilogue():
    torch.npu.set_device(0)
    torch.manual_seed(1511)
    heads, dim, width, eps = 4, 128, 512, 1e-6
    config = SimpleNamespace(
        linear_num_heads=heads,
        linear_head_dim=dim,
        linear_conv_size=4,
        hidden_size=width,
        rms_norm_eps=eps,
        linear_attn_config={"gate_lower_bound": -5.0},
    )
    mapping = SimpleNamespace(linear_attn=SimpleNamespace(tp_size=1))
    layer = PackedFLASHLocalKDA(config, mapping, layer_id=2).to("npu")
    with torch.no_grad():
        for parameter in layer.parameters():
            parameter.zero_()
        identity = torch.eye(width, dtype=torch.bfloat16, device="npu")
        gate_start = 3 * width
        layer.input_projection.weight[gate_start : gate_start + width].copy_(identity)
        layer.o_norm.weight.fill_(1)
        layer.o_proj.weight.copy_(identity)

    hidden = (torch.randn(2, width, device="npu") * 0.2).to(torch.bfloat16)
    core = (torch.randn(2, heads, dim, device="npu") * 0.3).to(torch.bfloat16)

    class Backend:
        kwargs = None

        def forward(self, **kwargs):
            self.kwargs = kwargs
            return core.clone()

    backend = Backend()
    ctx = SimpleNamespace(
        attn_backend=backend,
        token_to_kv_pool=object(),
        forward_mode=object(),
        bs=2,
    )
    output = layer(
        torch.arange(2, device="npu"),
        hidden,
        ctx,
        torch.arange(2, dtype=torch.int32, device="npu"),
    )
    gate = F.linear(
        hidden,
        layer.input_projection.weight[gate_start : gate_start + width],
    )
    expected = _lite_output_epilogue_oracle(
        core, gate, layer.o_norm.weight, eps, heads, dim
    )
    torch.npu.synchronize()

    torch.testing.assert_close(output.float(), expected.float(), atol=0.004, rtol=1e-4)
    assert backend.kwargs["output_gate"] is None
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.skipif(not _npu_available(), reason="requires an Ascend NPU")
def test_ascend_lite_output_epilogue_graph_replays_updated_values(batch):
    from tokenspeed_kernel.ops.activation.triton import rmsnorm_gated_sigmoid

    torch.npu.set_device(0)
    torch.manual_seed(1709 + batch)
    heads, dim, width, eps = 4, 128, 512, 1e-6
    core = (torch.randn(batch, width, device="npu") * 0.3).to(torch.bfloat16)
    packed_gate = (torch.randn(batch, width + 192, device="npu") * 0.4).to(
        torch.bfloat16
    )
    gate = packed_gate[:, 64 : 64 + width]
    weight = (torch.randn(dim, device="npu") * 0.2 + 1).to(torch.bfloat16)
    rmsnorm_gated_sigmoid(core, gate, weight, eps, heads, dim, enable_pdl=False)
    torch.npu.synchronize()

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=torch.npu.Stream(), auto_dispatch_capture=True):
        graph_output = rmsnorm_gated_sigmoid(
            core, gate, weight, eps, heads, dim, enable_pdl=False
        )
    graph.replay()
    torch.npu.synchronize()
    expected = _lite_output_epilogue_oracle(core, gate, weight, eps, heads, dim)
    torch.testing.assert_close(
        graph_output.float(), expected.float(), atol=0.004, rtol=1e-4
    )
    first_output = graph_output.clone()

    core.copy_((torch.randn_like(core.float()) * 0.25).to(torch.bfloat16))
    packed_gate.copy_((torch.randn_like(packed_gate.float()) * 0.35).to(torch.bfloat16))
    weight.copy_((torch.randn_like(weight.float()) * 0.15 + 1).to(torch.bfloat16))
    graph.replay()
    torch.npu.synchronize()
    expected = _lite_output_epilogue_oracle(core, gate, weight, eps, heads, dim)
    torch.testing.assert_close(
        graph_output.float(), expected.float(), atol=0.004, rtol=1e-4
    )
    assert not torch.equal(graph_output, first_output)
    assert torch.isfinite(graph_output).all()


@pytest.mark.parametrize("featurewise", [False, True])
@pytest.mark.skipif(not _npu_available(), reason="requires an Ascend NPU")
def test_ascend_registry_runs_scalar_and_featurewise_kda_reference(featurewise):
    from tokenspeed_kernel.ops.attention.kda import kda_paged_prefill

    inputs = _kda_inputs(dtype=torch.bfloat16, device="npu")
    if not featurewise:
        inputs["beta_logits"] = inputs["beta_logits"][..., 0]
    boundaries = torch.tensor([0, 2, 4], dtype=torch.int32, device="npu")
    initial_state = torch.randn(2, 2, 4, 4, device="npu")
    expected_output, expected_state = _oracle_kda(
        *(inputs[name][0].cpu() for name in ("q", "k", "v", "g_raw", "beta_logits")),
        inputs["A_log"].cpu(),
        inputs["dt_bias"].cpu(),
        initial_state.cpu(),
        [0, 2, 4],
    )

    result = kda_paged_prefill(
        **inputs,
        initial_state=initial_state,
        cu_seqlens=boundaries,
        cu_seqlens_cpu=torch.tensor([0, 2, 4], dtype=torch.int64),
        lower_bound=-5.0,
        solution="torch",
        recurrent_layout="k_major",
    )

    torch.testing.assert_close(
        result.out[0].cpu().float(), expected_output, atol=2e-2, rtol=2e-2
    )
    torch.testing.assert_close(
        result.final_state.cpu(), expected_state, atol=2e-2, rtol=2e-2
    )


@pytest.mark.skipif(not _npu_available(), reason="requires an Ascend NPU")
def test_lite_kda_production_shape_prefill_matches_continuous_decode():
    from tokenspeed.runtime.layers.attention.backends.state.kda import (
        KdaAttnBackend,
    )

    torch.manual_seed(43)
    heads, dim, tokens = 4, 128, 4
    channels = 3 * heads * dim
    projected = torch.randn(tokens, channels, dtype=torch.bfloat16, device="npu")
    weight = torch.randn(channels, 4, dtype=torch.bfloat16, device="npu") * 0.05
    initial_conv = (
        torch.randn(2, channels, 3, dtype=torch.bfloat16, device="npu") * 0.05
    )
    initial_state = (
        torch.randn(2, heads, dim, dim, dtype=torch.float32, device="npu") * 0.01
    )
    raw_gate = torch.randn(tokens, heads * dim, dtype=torch.bfloat16, device="npu")
    beta_logits = torch.randn(tokens, heads * dim, dtype=torch.bfloat16, device="npu")
    a_log = torch.randn(heads, dtype=torch.float32, device="npu") * 0.1
    dt_bias = torch.randn(heads * dim, dtype=torch.float32, device="npu") * 0.1

    backend = object.__new__(KdaAttnBackend)
    backend.kda_backend = "auto"
    backend.kda_recurrent_layout = "k_major"
    boundaries = torch.tensor([0, 1, 4], dtype=torch.int32, device="npu")
    pages = torch.tensor([2, 3], dtype=torch.int32, device="npu")
    prefill_conv_pool = torch.zeros(4, channels, 3, dtype=torch.bfloat16, device="npu")
    prefill_conv_pool[2:4].copy_(initial_conv)
    backend.forward_metadata = SimpleNamespace(query_start_loc=boundaries)
    conv_output = backend._causal_conv_prefill(
        projected,
        prefill_conv_pool,
        weight,
        None,
        "silu",
        pages,
        boundaries,
        torch.ones(2, dtype=torch.bool, device="npu"),
        torch.tensor([1, 3]),
        beta_raw=beta_logits,
        num_heads=heads,
        head_dim=dim,
    )
    query, key, value = (
        part.reshape(1, tokens, heads, dim)
        for part in torch.split(conv_output, heads * dim, dim=-1)
    )
    prefill_output, prefill_state = backend._prefill_scan(
        query,
        key,
        value,
        initial_state.clone(),
        boundaries,
        A_log=a_log,
        dt_bias=dt_bias,
        a=None,
        b=None,
        g_raw=raw_gate,
        f_a_out=None,
        f_b_weight=None,
        beta_raw=beta_logits,
        beta_is_logit=True,
        qk_l2norm_in_kernel=True,
        seq_len=tokens,
        num_real_tokens=tokens,
        lower_bound=-5.0,
        cu_seqlens_cpu=torch.tensor([0, 1, 4], dtype=torch.int64),
    )

    expected_conv, _ = _oracle_conv(
        projected.cpu().float(),
        weight.cpu().float(),
        initial_conv.cpu().float(),
        [0, 1],
        [0, 1],
        [0, 1, 4],
    )
    expected_output, expected_state = _oracle_kda(
        *(
            part.reshape(tokens, heads, dim)
            for part in torch.split(expected_conv, heads * dim, dim=-1)
        ),
        raw_gate.cpu().reshape(tokens, heads, dim),
        beta_logits.cpu().reshape(tokens, heads, dim),
        a_log.cpu(),
        dt_bias.cpu(),
        initial_state.cpu(),
        [0, 1, 4],
    )
    torch.testing.assert_close(
        conv_output.cpu().float(), expected_conv, atol=3e-2, rtol=3e-2
    )
    torch.testing.assert_close(
        prefill_output.cpu().float(), expected_output, atol=3e-2, rtol=3e-2
    )
    torch.testing.assert_close(
        prefill_state.cpu(), expected_state, atol=3e-2, rtol=3e-2
    )

    decode_conv_pool = torch.cat(
        (
            initial_conv.clone(),
            torch.full((1, channels, 3), 7, dtype=torch.bfloat16, device="npu"),
        )
    )
    decode_state_pool = torch.cat(
        (
            initial_state.clone(),
            torch.full((1, heads, dim, dim), 9, device="npu"),
        )
    )
    neighbor_conv = decode_conv_pool[2].clone()
    neighbor_state = decode_state_pool[2].clone()
    decoded = []
    for request, token_ids in enumerate(([0], [1, 2, 3])):
        slot = torch.tensor([request], dtype=torch.int32, device="npu")
        for token in token_ids:
            one_boundary = torch.tensor([0, 1], dtype=torch.int32, device="npu")
            backend.forward_metadata = SimpleNamespace(query_start_loc=one_boundary)
            conv_token = backend._causal_conv_decode(
                projected[token : token + 1],
                decode_conv_pool,
                weight,
                None,
                "silu",
                slot,
                slot,
                beta_raw=beta_logits[token : token + 1],
                num_heads=heads,
                head_dim=dim,
            )
            q_token, k_token, v_token = (
                part.reshape(1, 1, heads, dim)
                for part in torch.split(conv_token, heads * dim, dim=-1)
            )
            decoded.append(
                backend._decode_scan(
                    q_token,
                    k_token,
                    v_token,
                    decode_state_pool,
                    slot,
                    slot,
                    A_log=a_log,
                    dt_bias=dt_bias,
                    a=None,
                    b=None,
                    g_raw=raw_gate[token : token + 1],
                    f_a_out=None,
                    f_b_weight=None,
                    beta_raw=beta_logits[token : token + 1],
                    beta_is_logit=True,
                    qk_l2norm_in_kernel=True,
                    lower_bound=-5.0,
                    output_gate=None,
                    norm_weight=None,
                    norm_eps=None,
                )
            )

    decode_output = torch.cat(decoded, dim=0)
    torch.testing.assert_close(decode_output, prefill_output, atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(
        decode_conv_pool[:2], prefill_conv_pool[2:4], atol=0, rtol=0
    )
    torch.testing.assert_close(
        decode_state_pool[:2], prefill_state, atol=5e-4, rtol=5e-2
    )
    torch.testing.assert_close(
        decode_state_pool[:2].cpu(), expected_state, atol=3e-2, rtol=3e-2
    )
    torch.testing.assert_close(decode_conv_pool[2], neighbor_conv)
    torch.testing.assert_close(decode_state_pool[2], neighbor_state)
    assert torch.isfinite(decode_output).all()
    assert torch.isfinite(decode_state_pool[:2]).all()
