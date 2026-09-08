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

"""Portable PyTorch references for Lite's featurewise-beta KDA path."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from tokenspeed_kernel.ops.attention.kda import KdaPrefillResult


def _require_finite(name: str, tensor: torch.Tensor) -> None:
    if tensor.numel() and not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} contains NaN or Inf")


def _host_values(tensor: torch.Tensor) -> list[int]:
    return [int(value) for value in tensor.detach().cpu().tolist()]


def _segments(
    token_count: int,
    cu_seqlens: torch.Tensor,
    read_indices: torch.Tensor,
    write_indices: torch.Tensor,
    pool_size: int,
) -> list[tuple[int, int, int, int]]:
    if cu_seqlens.ndim != 1 or read_indices.ndim != 1:
        raise ValueError("KDA boundaries and state indices must be one-dimensional")
    if write_indices.shape != read_indices.shape:
        raise ValueError("KDA read/write state indices must have identical shapes")
    if cu_seqlens.numel() != read_indices.numel() + 1:
        raise ValueError("KDA boundaries must contain one more item than state indices")
    if cu_seqlens.dtype not in (torch.int32, torch.int64):
        raise ValueError("KDA boundaries must use int32 or int64")
    if read_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("KDA state indices must use int32 or int64")

    boundaries = _host_values(cu_seqlens)
    reads = _host_values(read_indices)
    writes = _host_values(write_indices)
    if not boundaries or boundaries[0] != 0 or boundaries[-1] != token_count:
        raise ValueError("KDA boundaries must cover the packed token range")
    if any(left > right for left, right in zip(boundaries, boundaries[1:])):
        raise ValueError("KDA boundaries must be monotonic")

    active_writes = []
    segments = []
    for start, end, read, write in zip(boundaries, boundaries[1:], reads, writes):
        if read < -1 or read >= pool_size or write < -1 or write >= pool_size:
            raise ValueError("KDA state index is outside the state pool")
        if end > start:
            if (read < 0) != (write < 0):
                raise ValueError("KDA active segments must pad both state indices")
            if write >= 0:
                active_writes.append(write)
        segments.append((start, end, read, write))
    if len(active_writes) != len(set(active_writes)):
        raise ValueError("KDA active segments must use distinct output state slots")
    return segments


def torch_kda_causal_conv1d(
    projected: torch.Tensor,
    weight: torch.Tensor,
    conv_state: torch.Tensor,
    read_indices: torch.Tensor,
    write_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
    activation: str | None = "silu",
) -> torch.Tensor:
    """Run packed depthwise causal-conv and publish its final raw-input window."""
    if projected.ndim != 2 or weight.ndim != 2 or conv_state.ndim != 3:
        raise ValueError("KDA causal-conv expects 2D input/weight and 3D state")
    channels, width = weight.shape
    if width < 1 or projected.shape[1] != channels:
        raise ValueError("KDA causal-conv input and weight shapes do not match")
    if conv_state.shape[1:] != (channels, width - 1):
        raise ValueError("KDA causal-conv state shape does not match its weight")
    if bias is not None and bias.shape != (channels,):
        raise ValueError("KDA causal-conv bias must match the channel count")
    if activation not in (None, "silu", "swish"):
        raise ValueError(f"Unsupported KDA causal-conv activation {activation!r}")
    if has_initial_state is not None and has_initial_state.shape != read_indices.shape:
        raise ValueError("KDA initial-state mask must match state indices")

    segments = _segments(
        projected.shape[0],
        cu_seqlens,
        read_indices,
        write_indices,
        conv_state.shape[0],
    )
    _require_finite("KDA causal-conv input", projected)
    _require_finite("KDA causal-conv weight", weight)
    if bias is not None:
        _require_finite("KDA causal-conv bias", bias)

    initial = (
        [True] * read_indices.numel()
        if has_initial_state is None
        else [bool(value) for value in has_initial_state.detach().cpu().tolist()]
    )
    output = torch.zeros_like(projected)
    updates: list[tuple[int, torch.Tensor]] = []
    weight_fp32 = weight.float()
    bias_fp32 = None if bias is None else bias.float()
    for request, (start, end, read, write) in enumerate(segments):
        if start == end or read < 0:
            continue
        history = (
            conv_state[read].clone()
            if initial[request]
            else conv_state.new_zeros((channels, width - 1))
        )
        _require_finite("KDA causal-conv initial state", history)
        request_output = []
        for token in projected[start:end]:
            window = torch.cat((history, token.unsqueeze(-1)), dim=-1)
            value = (window.float() * weight_fp32).sum(dim=-1)
            if bias_fp32 is not None:
                value = value + bias_fp32
            if activation is not None:
                value = F.silu(value)
            request_output.append(value.to(projected.dtype))
            history = window[:, 1:]
        output[start:end] = torch.stack(request_output)
        updates.append((write, history))

    _require_finite("KDA causal-conv output", output)
    for write, state in updates:
        _require_finite("KDA causal-conv final state", state)
        conv_state[write].copy_(state)
    return output


def _prepare_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    _require_finite("KDA beta logits", beta_logits)
    q_fp32 = F.normalize(q.float(), p=2, dim=-1)
    k_fp32 = F.normalize(k.float(), p=2, dim=-1)
    if beta_logits.shape == q.shape:
        if v.shape[-1] != q.shape[-1]:
            raise ValueError("Featurewise beta requires equal KDA key/value dims")
        beta_scale = torch.sqrt(torch.sigmoid(beta_logits.float()) + 1e-10)
        return q_fp32, k_fp32 * beta_scale, v.float() * beta_scale, None
    if beta_logits.shape == q.shape[:-1]:
        return q_fp32, k_fp32, v.float(), torch.sigmoid(beta_logits.float())
    raise ValueError("KDA beta logits must be scalar-per-head or featurewise")


def _safe_gate(
    g_raw: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float | None,
) -> torch.Tensor:
    if lower_bound is None or not math.isfinite(lower_bound) or lower_bound >= 0:
        raise ValueError("KDA reference requires a finite negative lower bound")
    heads, key_dim = g_raw.shape[-2:]
    if A_log.numel() != heads or dt_bias.numel() != heads * key_dim:
        raise ValueError("KDA gate parameters do not match the local head shape")
    _require_finite("KDA A_log", A_log)
    _require_finite("KDA dt_bias", dt_bias)
    logits = g_raw.float() + dt_bias.float().reshape(heads, key_dim)
    return lower_bound * torch.sigmoid(A_log.float().reshape(heads, 1).exp() * logits)


def _scan_segment(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor | None,
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    scale = q.shape[-1] ** -0.5
    outputs = []
    for token in range(q.shape[0]):
        state = state * gate[token].exp().unsqueeze(-1)
        delta = v[token] - torch.matmul(k[token].unsqueeze(-2), state).squeeze(-2)
        if beta is not None:
            delta = delta * beta[token].unsqueeze(-1)
        state = state + k[token].unsqueeze(-1) * delta.unsqueeze(-2)
        outputs.append(
            torch.matmul((q[token] * scale).unsqueeze(-2), state).squeeze(-2)
        )
    return torch.stack(outputs), state


def _validate_kda_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_raw: torch.Tensor,
    state: torch.Tensor,
) -> None:
    if q.ndim != 4 or q.shape[0] != 1 or k.shape != q.shape:
        raise ValueError("KDA reference requires q/k shaped [1,T,H,K]")
    if g_raw.shape != q.shape or v.ndim != 4 or v.shape[:3] != q.shape[:3]:
        raise ValueError("KDA reference v/g shapes do not match q")
    if state.ndim != 4 or state.shape[1:] != (
        q.shape[2],
        q.shape[3],
        v.shape[3],
    ):
        raise ValueError("KDA reference requires key-major [N,H,K,V] state")
    if state.dtype != torch.float32:
        raise ValueError("KDA reference recurrent state must use FP32")
    for name, tensor in (("q", q), ("k", k), ("v", v), ("gate", g_raw)):
        _require_finite(f"KDA {name}", tensor)


def torch_kda_paged_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_raw: torch.Tensor,
    beta_logits: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_cpu: torch.Tensor,
    lower_bound: float | None,
) -> KdaPrefillResult:
    """Reference packed KDA Prefill with one initial/final state per segment."""
    _validate_kda_inputs(q, k, v, g_raw, initial_state)
    if beta_logits.shape not in (q.shape, q.shape[:-1]):
        raise ValueError("KDA beta logits have an invalid shape")
    if (
        not isinstance(cu_seqlens_cpu, torch.Tensor)
        or cu_seqlens_cpu.device.type != "cpu"
    ):
        raise ValueError("KDA Prefill requires host sequence boundaries")
    if _host_values(cu_seqlens) != _host_values(cu_seqlens_cpu):
        raise ValueError("KDA device and host sequence boundaries differ")
    indices = torch.arange(initial_state.shape[0], dtype=torch.int64)
    segments = _segments(
        q.shape[1], cu_seqlens_cpu, indices, indices, initial_state.shape[0]
    )
    q_fp32, k_fp32, v_fp32, beta = _prepare_inputs(q, k, v, beta_logits)
    gate = _safe_gate(g_raw, A_log, dt_bias, lower_bound)
    _require_finite("KDA prepared query", q_fp32)
    _require_finite("KDA prepared key", k_fp32)
    _require_finite("KDA prepared value", v_fp32)
    _require_finite("KDA safe gate", gate)

    output = torch.zeros_like(v)
    final_state = initial_state.clone()
    for request, (start, end, _, _) in enumerate(segments):
        if start == end:
            continue
        request_output, request_state = _scan_segment(
            q_fp32[0, start:end],
            k_fp32[0, start:end],
            v_fp32[0, start:end],
            gate[0, start:end],
            None if beta is None else beta[0, start:end],
            initial_state[request],
        )
        output[0, start:end] = request_output.to(output.dtype)
        final_state[request].copy_(request_state)
    _require_finite("KDA Prefill output", output)
    _require_finite("KDA Prefill final state", final_state)
    return KdaPrefillResult(output, final_state)


def torch_kda_paged_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_raw: torch.Tensor,
    beta_logits: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    state_pool: torch.Tensor,
    read_indices: torch.Tensor,
    write_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    lower_bound: float | None,
) -> torch.Tensor:
    """Reference single-token Decode against independently indexed state pages."""
    _validate_kda_inputs(q, k, v, g_raw, state_pool)
    segments = _segments(
        q.shape[1], cu_seqlens, read_indices, write_indices, state_pool.shape[0]
    )
    if any(end - start > 1 for start, end, _, _ in segments):
        raise ValueError("KDA Decode accepts at most one token per request")
    q_fp32, k_fp32, v_fp32, beta = _prepare_inputs(q, k, v, beta_logits)
    gate = _safe_gate(g_raw, A_log, dt_bias, lower_bound)
    output = torch.zeros_like(v)
    updates: list[tuple[int, torch.Tensor]] = []
    for start, end, read, write in segments:
        if start == end or read < 0:
            continue
        _require_finite("KDA Decode initial state", state_pool[read])
        request_output, request_state = _scan_segment(
            q_fp32[0, start:end],
            k_fp32[0, start:end],
            v_fp32[0, start:end],
            gate[0, start:end],
            None if beta is None else beta[0, start:end],
            state_pool[read].clone(),
        )
        output[0, start:end] = request_output.to(output.dtype)
        updates.append((write, request_state))
    _require_finite("KDA Decode output", output)
    for write, state in updates:
        _require_finite("KDA Decode final state", state)
        state_pool[write].copy_(state)
    return output


__all__ = [
    "torch_kda_causal_conv1d",
    "torch_kda_paged_decode",
    "torch_kda_paged_prefill",
]
