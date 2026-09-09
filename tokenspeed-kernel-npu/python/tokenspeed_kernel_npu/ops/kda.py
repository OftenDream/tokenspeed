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

"""Ascend implementations of Lite KDA operators."""

from __future__ import annotations

import math
from functools import cache
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from tokenspeed_kernel_npu._triton import tl, triton
from tokenspeed_kernel_npu.public_kda_ops import is_available, load_public_kda_ops

if TYPE_CHECKING:
    from tokenspeed_kernel.ops.attention.kda import KdaPrefillResult

_PREFILL_CHUNK_SIZE = 64
_PREFILL_MIN_PUBLIC_TOKENS = 32
_LITE_KDA_HEADS = 4
_LITE_KDA_DIM = 128
_DECODE_VALUE_BLOCK = 32


@cache
def _load_flash_causal_conv_ops():
    try:
        import flash_ops
    except ModuleNotFoundError as error:
        if error.name != "flash_ops":
            raise
        return None
    # Older packages expose the same schema but assume dense slots and cannot
    # write Prefill results to independent destination pages.
    if not getattr(flash_ops, "CAUSAL_CONV1D_STATE_SLOT_STRIDE", False):
        return None
    return torch.ops.custom


# CANN discovers custom OPP metadata when the device is first initialized.  The
# NPU registry imports this module before model tensors are allocated, so load
# the optional artifacts here instead of waiting for the first kernel call.
# Keep this order: loading flash_ops first corrupts CANN teardown when both
# vendor registries are present.
load_public_kda_ops()
_load_flash_causal_conv_ops()


@triton.jit
def _featurewise_decode_kernel(
    q,
    k,
    v,
    g,
    beta,
    a_log,
    dt_bias,
    output,
    state_pool,
    read_indices,
    write_indices,
    cu_seqlens,
    stride_q_batch: tl.constexpr,
    stride_q_head: tl.constexpr,
    stride_k_batch: tl.constexpr,
    stride_k_head: tl.constexpr,
    stride_v_batch: tl.constexpr,
    stride_v_head: tl.constexpr,
    stride_g_batch: tl.constexpr,
    stride_g_head: tl.constexpr,
    stride_beta_batch: tl.constexpr,
    stride_beta_head: tl.constexpr,
    stride_state_page: tl.constexpr,
    stride_state_head: tl.constexpr,
    stride_state_key: tl.constexpr,
    stride_state_value: tl.constexpr,
    lower_bound: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
):
    pid = tl.program_id(0)
    value_tiles = tl.cdiv(V, BV)
    value_tile = pid % value_tiles
    row = pid // value_tiles
    batch = row // H
    head = row % H

    key_offsets = tl.arange(0, K)
    value_offsets = value_tile * BV + tl.arange(0, BV)
    value_mask = value_offsets < V
    begin = tl.load(cu_seqlens + batch).to(tl.int64)
    end = tl.load(cu_seqlens + batch + 1).to(tl.int64)
    read_page = tl.load(read_indices + batch).to(tl.int64)
    write_page = tl.load(write_indices + batch).to(tl.int64)
    active = (end > begin) & (read_page >= 0) & (write_page >= 0)
    safe_read = tl.where(active, read_page, 0)
    safe_write = tl.where(active, write_page, 0)

    q_row = tl.load(q + batch * stride_q_batch + head * stride_q_head + key_offsets).to(
        tl.float32
    )
    k_row = tl.load(k + batch * stride_k_batch + head * stride_k_head + key_offsets).to(
        tl.float32
    )
    g_row = tl.load(g + batch * stride_g_batch + head * stride_g_head + key_offsets).to(
        tl.float32
    )
    beta_key = tl.load(
        beta + batch * stride_beta_batch + head * stride_beta_head + key_offsets
    ).to(tl.float32)
    value = tl.load(
        v + batch * stride_v_batch + head * stride_v_head + value_offsets,
        mask=value_mask,
        other=0.0,
    ).to(tl.float32)
    beta_value = tl.load(
        beta + batch * stride_beta_batch + head * stride_beta_head + value_offsets,
        mask=value_mask,
        other=0.0,
    ).to(tl.float32)

    q_norm = tl.maximum(tl.sqrt(tl.sum(q_row * q_row, axis=0)), 1.0e-12)
    k_norm = tl.maximum(tl.sqrt(tl.sum(k_row * k_row, axis=0)), 1.0e-12)
    q_row = q_row / q_norm
    key = (k_row / k_norm) * tl.sqrt(tl.sigmoid(beta_key) + 1.0e-10)
    value = value * tl.sqrt(tl.sigmoid(beta_value) + 1.0e-10)
    a = tl.load(a_log + head).to(tl.float32)
    bias = tl.load(dt_bias + head * K + key_offsets).to(tl.float32)
    gate = lower_bound * tl.sigmoid(tl.exp(a) * (g_row + bias))

    state_offsets = (
        safe_read * stride_state_page
        + head * stride_state_head
        + key_offsets[:, None] * stride_state_key
        + value_offsets[None, :] * stride_state_value
    )
    state = tl.load(
        state_pool + state_offsets,
        mask=active & value_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    state = state * tl.exp(gate)[:, None]
    delta = value - tl.sum(state * key[:, None], axis=0)
    next_state = state + key[:, None] * delta[None, :]
    result = tl.sum(next_state * q_row[:, None], axis=0) * (K**-0.5)

    output_offsets = row * V + value_offsets
    tl.store(output + output_offsets, result, mask=active & value_mask)
    tl.store(output + output_offsets, 0.0, mask=(~active) & value_mask)
    write_offsets = (
        safe_write * stride_state_page
        + head * stride_state_head
        + key_offsets[:, None] * stride_state_key
        + value_offsets[None, :] * stride_state_value
    )
    tl.store(
        state_pool + write_offsets,
        next_state,
        mask=active & value_mask[None, :],
    )


def _triton_kda_paged_decode(
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
    lower_bound: float,
) -> torch.Tensor:
    output = torch.empty(v.shape, dtype=v.dtype, device=v.device)
    batch = q.shape[1]
    grid = (batch * _LITE_KDA_HEADS * triton.cdiv(_LITE_KDA_DIM, _DECODE_VALUE_BLOCK),)
    _featurewise_decode_kernel[grid](
        q,
        k,
        v,
        g_raw,
        beta_logits,
        A_log,
        dt_bias,
        output,
        state_pool,
        read_indices,
        write_indices,
        cu_seqlens,
        stride_q_batch=q.stride(1),
        stride_q_head=q.stride(2),
        stride_k_batch=k.stride(1),
        stride_k_head=k.stride(2),
        stride_v_batch=v.stride(1),
        stride_v_head=v.stride(2),
        stride_g_batch=g_raw.stride(1),
        stride_g_head=g_raw.stride(2),
        stride_beta_batch=beta_logits.stride(1),
        stride_beta_head=beta_logits.stride(2),
        stride_state_page=state_pool.stride(0),
        stride_state_head=state_pool.stride(1),
        stride_state_key=state_pool.stride(2),
        stride_state_value=state_pool.stride(3),
        lower_bound=lower_bound,
        H=_LITE_KDA_HEADS,
        K=_LITE_KDA_DIM,
        V=_LITE_KDA_DIM,
        BV=_DECODE_VALUE_BLOCK,
    )
    return output


def _activate(value: torch.Tensor, activation: str | None) -> torch.Tensor:
    return value if activation is None else F.silu(value)


def _safe_indices(
    read_indices: torch.Tensor,
    write_indices: torch.Tensor,
    active: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    indices_active = (read_indices >= 0) & (write_indices >= 0)
    active = indices_active if active is None else active & indices_active
    zero = torch.zeros((), dtype=read_indices.dtype, device=read_indices.device)
    return (
        active,
        torch.where(active, read_indices, zero).to(torch.int64),
        torch.where(active, write_indices, zero).to(torch.int64),
    )


def _channel_major_conv_state_view(
    conv_state: torch.Tensor, channels: int
) -> torch.Tensor:
    """Share storage as [slots, channels, history] for stride-aware Torch ops.

    This is not a physical relayout for the packaged kernel. Keeping a view
    ensures index_copy_ publishes updates to the original cache allocation.
    """
    if conv_state.shape[1] == channels:
        return conv_state
    return conv_state.transpose(1, 2)


def _decode(
    projected: torch.Tensor,
    weight: torch.Tensor,
    conv_state: torch.Tensor,
    read_indices: torch.Tensor,
    write_indices: torch.Tensor,
    bias: torch.Tensor | None,
    activation: str | None,
) -> torch.Tensor:
    conv_state = _channel_major_conv_state_view(conv_state, weight.shape[0])
    active, safe_reads, safe_writes = _safe_indices(read_indices, write_indices)
    history = conv_state.index_select(0, safe_reads)
    window = torch.cat((history, projected.unsqueeze(-1)), dim=-1)
    output = (window.float() * weight.float()).sum(dim=-1)
    if bias is not None:
        output = output + bias.float()
    output = _activate(output, activation).to(projected.dtype)

    destination = conv_state.index_select(0, safe_writes)
    next_state = torch.where(active[:, None, None], window[:, :, 1:], destination)
    conv_state.index_copy_(0, safe_writes, next_state)
    return torch.where(active[:, None], output, torch.zeros_like(output))


def _prefill(
    projected: torch.Tensor,
    weight: torch.Tensor,
    conv_state: torch.Tensor,
    read_indices: torch.Tensor,
    write_indices: torch.Tensor,
    cu_seqlens_cpu: torch.Tensor,
    bias: torch.Tensor | None,
    has_initial_state: torch.Tensor | None,
    activation: str | None,
) -> torch.Tensor:
    conv_state = _channel_major_conv_state_view(conv_state, weight.shape[0])
    boundaries = [int(value) for value in cu_seqlens_cpu.tolist()]
    if (
        not boundaries
        or boundaries[0] != 0
        or boundaries[-1] != projected.shape[0]
        or any(left > right for left, right in zip(boundaries, boundaries[1:]))
    ):
        raise ValueError("KDA causal-conv boundaries must cover packed input")

    active, safe_reads, safe_writes = _safe_indices(read_indices, write_indices)
    initial = (
        torch.ones_like(active) if has_initial_state is None else has_initial_state
    )
    output = torch.zeros_like(projected)
    for row, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        if start == end:
            continue
        read = safe_reads[row : row + 1]
        write = safe_writes[row : row + 1]
        history = conv_state.index_select(0, read)[0]
        history = torch.where(initial[row], history, torch.zeros_like(history))
        signal = torch.cat((history, projected[start:end].transpose(0, 1)), dim=-1)
        convolved = F.conv1d(
            signal.unsqueeze(0),
            weight.unsqueeze(1),
            bias=bias,
            groups=weight.shape[0],
        )[0].transpose(0, 1)
        output[start:end] = torch.where(
            active[row],
            _activate(convolved, activation),
            torch.zeros_like(convolved),
        )
        destination = conv_state.index_select(0, write)[0]
        next_state = torch.where(active[row], signal[:, -3:], destination)
        conv_state.index_copy_(0, write, next_state.unsqueeze(0))
    return output


def _validate_decode_metadata_cpu(
    cu_seqlens: torch.Tensor,
    read_indices: torch.Tensor,
    write_indices: torch.Tensor,
    pool_size: int,
) -> None:
    if cu_seqlens.device.type != "cpu":
        return
    boundaries = cu_seqlens.tolist()
    reads = read_indices.tolist()
    writes = write_indices.tolist()
    increments = [right - left for left, right in zip(boundaries, boundaries[1:])]
    if (
        not boundaries
        or boundaries[0] != 0
        or any(step not in (0, 1) for step in increments)
    ):
        raise ValueError("KDA Decode boundaries must contain zero/one-token segments")
    if 0 in increments and any(step for step in increments[increments.index(0) :]):
        raise ValueError("KDA Decode graph padding must be a batch tail")

    active_writes = []
    for step, read, write in zip(increments, reads, writes):
        if step:
            if read < 0 or write < 0:
                raise ValueError("KDA Decode active rows require read/write pages")
            if read >= pool_size or write >= pool_size:
                raise ValueError("KDA Decode state index is outside the state pool")
            active_writes.append(write)
        elif read != -1 or write != -1:
            raise ValueError("KDA Decode padded rows require -1 read/write pages")
    if len(active_writes) != len(set(active_writes)):
        raise ValueError("KDA Decode active rows require distinct write pages")


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
    """Run graph-safe single-token KDA Decode against K-major state pages."""
    if q.ndim != 4 or q.shape[0] != 1 or k.shape != q.shape or g_raw.shape != q.shape:
        raise ValueError("KDA Decode requires q/k/g shaped [1,batch,heads,key_dim]")
    if v.ndim != 4 or v.shape[:3] != q.shape[:3]:
        raise ValueError("KDA Decode value must match q through the head dimension")
    batch, heads, key_dim, value_dim = q.shape[1], q.shape[2], q.shape[3], v.shape[3]
    if read_indices.shape != (batch,) or write_indices.shape != (batch,):
        raise ValueError("KDA Decode requires one read/write page per batch row")
    if cu_seqlens.shape != (batch + 1,):
        raise ValueError("KDA Decode requires one more boundary than batch rows")
    if beta_logits.shape not in (q.shape, q.shape[:-1]):
        raise ValueError("KDA beta logits must be scalar-per-head or featurewise")
    if beta_logits.shape == q.shape and value_dim != key_dim:
        raise ValueError("Featurewise beta requires equal KDA key/value dims")
    if state_pool.ndim != 4 or state_pool.shape[1:] != (
        heads,
        key_dim,
        value_dim,
    ):
        raise ValueError(
            "KDA Decode requires key-major [pages,heads,key_dim,value_dim] state"
        )
    if state_pool.shape[0] < 1 or state_pool.dtype != torch.float32:
        raise ValueError("KDA Decode recurrent state must be a non-empty FP32 pool")
    if A_log.shape != (heads,) or dt_bias.numel() != heads * key_dim:
        raise ValueError("KDA Decode gate parameters do not match the local head shape")
    if lower_bound is None or not math.isfinite(lower_bound) or lower_bound >= 0:
        raise ValueError("KDA Decode requires a finite negative lower bound")
    if read_indices.dtype not in (
        torch.int32,
        torch.int64,
    ) or write_indices.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("KDA Decode state indices must use int32 or int64")
    if cu_seqlens.dtype not in (torch.int32, torch.int64):
        raise ValueError("KDA Decode boundaries must use int32 or int64")
    tensors = (
        k,
        v,
        g_raw,
        beta_logits,
        A_log,
        dt_bias,
        state_pool,
        read_indices,
        write_indices,
        cu_seqlens,
    )
    if any(tensor.device != q.device for tensor in tensors):
        raise ValueError("KDA Decode inputs and state must share one device")
    _validate_decode_metadata_cpu(
        cu_seqlens, read_indices, write_indices, state_pool.shape[0]
    )
    if (
        q.device.type == "npu"
        and beta_logits.shape == q.shape
        and (heads, key_dim, value_dim)
        == (_LITE_KDA_HEADS, _LITE_KDA_DIM, _LITE_KDA_DIM)
        and all(
            tensor.stride(-1) == 1
            for tensor in (q, k, v, g_raw, beta_logits, state_pool)
        )
        and A_log.is_contiguous()
        and dt_bias.is_contiguous()
        and read_indices.is_contiguous()
        and write_indices.is_contiguous()
        and cu_seqlens.is_contiguous()
    ):
        return _triton_kda_paged_decode(
            q,
            k,
            v,
            g_raw,
            beta_logits,
            A_log,
            dt_bias,
            state_pool=state_pool,
            read_indices=read_indices,
            write_indices=write_indices,
            cu_seqlens=cu_seqlens,
            lower_bound=lower_bound,
        )

    segment_active = cu_seqlens[1:] > cu_seqlens[:-1]
    active, safe_reads, safe_writes = _safe_indices(
        read_indices, write_indices, segment_active
    )
    q_fp32 = F.normalize(q.float(), p=2, dim=-1)
    k_fp32 = F.normalize(k.float(), p=2, dim=-1)
    v_fp32 = v.float()
    scalar_beta = None
    if beta_logits.shape == q.shape:
        beta_scale = torch.sqrt(torch.sigmoid(beta_logits.float()) + 1e-10)
        k_fp32 = k_fp32 * beta_scale
        v_fp32 = v_fp32 * beta_scale
    else:
        scalar_beta = torch.sigmoid(beta_logits.float())

    gate_logits = g_raw.float() + dt_bias.float().reshape(heads, key_dim)
    gate = lower_bound * torch.sigmoid(
        A_log.float().reshape(heads, 1).exp() * gate_logits
    )
    state = state_pool.index_select(0, safe_reads)
    state = state * gate[0].exp().unsqueeze(-1)
    key = k_fp32[0]
    delta = v_fp32[0] - torch.matmul(key.unsqueeze(-2), state).squeeze(-2)
    if scalar_beta is not None:
        delta = delta * scalar_beta[0].unsqueeze(-1)
    next_state = state + key.unsqueeze(-1) * delta.unsqueeze(-2)
    output = torch.matmul(
        (q_fp32[0] * (key_dim**-0.5)).unsqueeze(-2), next_state
    ).squeeze(-2)

    destination = state_pool.index_select(0, safe_writes)
    published = torch.where(active[:, None, None, None], next_state, destination)
    state_pool.index_copy_(0, safe_writes, published)
    output = torch.where(active[:, None, None], output, torch.zeros_like(output))
    return output.unsqueeze(0).to(v.dtype)


def ref_kda_causal_conv1d(
    *,
    projected: torch.Tensor,
    weight: torch.Tensor,
    conv_state: torch.Tensor,
    read_indices: torch.Tensor,
    write_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_cpu: torch.Tensor | None,
    bias: torch.Tensor | None,
    has_initial_state: torch.Tensor | None,
    activation: str | None,
    decode: bool,
    require_public: bool = False,
) -> torch.Tensor:
    """Ref composition: graph-safe Decode or per-request depthwise-conv Prefill."""
    del cu_seqlens, require_public
    if decode:
        return _decode(
            projected,
            weight,
            conv_state,
            read_indices,
            write_indices,
            bias,
            activation,
        )
    assert cu_seqlens_cpu is not None
    return _prefill(
        projected,
        weight,
        conv_state,
        read_indices,
        write_indices,
        cu_seqlens_cpu,
        bias,
        has_initial_state,
        activation,
    )


def public_kda_causal_conv1d(
    *,
    projected: torch.Tensor,
    weight: torch.Tensor,
    conv_state: torch.Tensor,
    read_indices: torch.Tensor,
    write_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_cpu: torch.Tensor | None,
    bias: torch.Tensor | None,
    has_initial_state: torch.Tensor | None,
    activation: str | None,
    decode: bool,
    require_public: bool = False,
) -> torch.Tensor:
    """Use the packaged op on width-major state, including strided arena slots."""
    channels, width = weight.shape
    if conv_state.ndim != 3 or conv_state.shape[1:] != (width - 1, channels):
        raise ValueError(
            "public causal-conv requires state [slots, width - 1, channels]"
        )
    if (
        conv_state.stride(2) != 1
        or conv_state.stride(1) != channels
        or conv_state.stride(0) < (width - 1) * channels
    ):
        raise ValueError("public causal-conv requires state strides [S, channels, 1]")
    op_name = "npu_causal_conv1d"
    flash_ops = _load_flash_causal_conv_ops()
    op = getattr(flash_ops, op_name, None) if flash_ops is not None else None
    if op is None:
        if require_public:
            raise RuntimeError(
                f"flash_ops schema custom::{op_name} with strided-state support is unavailable; "
                "install the matching updated wheel and custom OPP"
            )
        return ref_kda_causal_conv1d(
            projected=projected,
            weight=weight,
            conv_state=conv_state,
            read_indices=read_indices,
            write_indices=write_indices,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            bias=bias,
            has_initial_state=has_initial_state,
            activation=activation,
            decode=decode,
        )
    if decode:
        return op(
            projected,
            weight.transpose(0, 1).contiguous(),
            conv_state,
            bias=bias,
            cache_indices=read_indices,
            write_indices=write_indices,
            activation_mode=0 if activation is None else 1,
            pad_slot_id=-1,
            run_mode=1,
        )

    # Keep the original arena view: the op reads its slot stride and writes
    # directly to independent destination slots, without a compact pool.
    return op(
        projected,
        weight.transpose(0, 1).contiguous(),
        conv_state,
        bias=bias,
        query_start_loc=cu_seqlens,
        cache_indices=read_indices,
        write_indices=write_indices,
        initial_state_mode=has_initial_state,
        activation_mode=0 if activation is None else 1,
        pad_slot_id=-1,
        run_mode=0,
    )


def _prefill_chunk_plan(
    cu_seqlens_cpu: torch.Tensor,
    token_count: int,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    boundaries = tuple(int(value) for value in cu_seqlens_cpu.tolist())
    if (
        not boundaries
        or boundaries[0] != 0
        or boundaries[-1] != token_count
        or any(left > right for left, right in zip(boundaries, boundaries[1:]))
    ):
        raise ValueError("KDA Prefill boundaries must cover the packed input")
    keep = tuple(
        index
        for index, (left, right) in enumerate(zip(boundaries, boundaries[1:]))
        if right > left
    )
    compact = [0]
    chunks: list[int] = []
    for sequence, request in enumerate(keep):
        length = boundaries[request + 1] - boundaries[request]
        compact.append(compact[-1] + length)
        for chunk in range(math.ceil(length / _PREFILL_CHUNK_SIZE)):
            chunks.extend((sequence, chunk))
    return tuple(compact), keep, tuple(chunks)


def _public_prefill_unsupported(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_raw: torch.Tensor,
    beta_logits: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    initial_state: torch.Tensor,
    lower_bound: float | None,
) -> str | None:
    heads, key_dim, value_dim = q.shape[2], q.shape[3], v.shape[3]
    if q.dtype not in (torch.bfloat16, torch.float16) or any(
        tensor.dtype != q.dtype for tensor in (k, v, g_raw, beta_logits)
    ):
        return "public KDA Prefill requires matching BF16 or FP16 activations"
    if any(
        tensor.device != q.device
        for tensor in (
            k,
            v,
            g_raw,
            beta_logits,
            A_log,
            dt_bias,
            initial_state,
        )
    ):
        return "public KDA Prefill tensors must share one device"
    if key_dim % 16 or value_dim % 16 or value_dim > 256:
        return "public KDA Prefill requires aligned K/V dimensions and V <= 256"
    if value_dim != key_dim or beta_logits.shape != q.shape:
        return "public KDA Prefill requires Lite featurewise beta with K == V"
    if initial_state.shape[1:] != (heads, key_dim, value_dim) or (
        initial_state.dtype != torch.float32
    ):
        return "public KDA Prefill requires K-major FP32 recurrent state"
    if A_log.shape != (heads,) or A_log.dtype != torch.float32:
        return "public KDA Prefill requires FP32 A_log matching local heads"
    if dt_bias.numel() != heads * key_dim or dt_bias.dtype != torch.float32:
        return "public KDA Prefill requires FP32 dt_bias matching local heads"
    if (
        lower_bound is None
        or not math.isfinite(lower_bound)
        or not -5.0 <= lower_bound < 0.0
    ):
        return "public KDA Prefill requires a safe-gate lower bound in [-5, 0)"
    return None


def public_kda_paged_prefill(
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
    require_public: bool = False,
) -> KdaPrefillResult:
    """Run Lite featurewise-beta Prefill through the public split KDA ops."""
    from tokenspeed_kernel.ops.attention.kda import KdaPrefillResult

    required = ("kda_gate_cumsum", "chunk_kda_fwd")
    missing = [name for name in required if not is_available(name)]
    unsupported = _public_prefill_unsupported(
        q,
        k,
        v,
        g_raw,
        beta_logits,
        A_log,
        dt_bias,
        initial_state,
        lower_bound,
    )
    prefer_reference = q.shape[1] < _PREFILL_MIN_PUBLIC_TOKENS and not require_public
    if missing or unsupported or prefer_reference:
        if require_public:
            if unsupported:
                raise RuntimeError(unsupported)
            status = load_public_kda_ops()
            reason = status.reason or f"missing public ops: {missing}"
            raise RuntimeError(reason)
        from tokenspeed_kernel.ops.attention.kda_reference import (
            torch_kda_paged_prefill,
        )

        return torch_kda_paged_prefill(
            q,
            k,
            v,
            g_raw,
            beta_logits,
            A_log,
            dt_bias,
            initial_state=initial_state,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            lower_bound=lower_bound,
        )

    compact_cu, keep, chunk_indices = _prefill_chunk_plan(cu_seqlens_cpu, q.shape[1])
    if not keep:
        return KdaPrefillResult(torch.zeros_like(v), initial_state.clone())

    q_prepared = F.normalize(q.float(), p=2, dim=-1).to(q.dtype)
    key_dim = q.shape[-1]
    beta_scale = torch.sqrt(torch.sigmoid(beta_logits.float()) + 1e-10)
    k_prepared = (F.normalize(k.float(), p=2, dim=-1) * beta_scale).to(k.dtype)
    v_prepared = (v.float() * beta_scale).to(v.dtype)
    beta = torch.ones(q.shape[:-1], dtype=q.dtype, device=q.device)

    gate_cumsum = torch.ops.tokenspeed_npu_public_kda.kda_gate_cumsum(
        g_raw.contiguous(),
        _PREFILL_CHUNK_SIZE,
        a_log=A_log.contiguous(),
        dt_bias=dt_bias.contiguous(),
        cu_seqlens=compact_cu,
        use_gate_in_kernel=True,
        safe_gate=True,
        lower_bound=lower_bound,
        layout="BSND",
    )
    compacted = len(keep) != initial_state.shape[0]
    if compacted:
        keep_tensor = torch.tensor(keep, dtype=torch.int64, device=q.device)
        compact_initial = initial_state.index_select(0, keep_tensor).contiguous()
    else:
        compact_initial = initial_state.contiguous()
    output, compact_final = torch.ops.tokenspeed_npu_public_kda.chunk_kda_fwd(
        q_prepared,
        k_prepared,
        v_prepared,
        gate_cumsum,
        beta,
        key_dim**-0.5,
        _PREFILL_CHUNK_SIZE,
        "BSND",
        initial_state=compact_initial,
        cu_seqlens=compact_cu,
        chunk_indices=chunk_indices,
    )
    if not compacted:
        return KdaPrefillResult(output, compact_final)
    final_state = initial_state.clone()
    final_state.index_copy_(0, keep_tensor, compact_final)
    return KdaPrefillResult(output, final_state)


__all__ = [
    "public_kda_causal_conv1d",
    "public_kda_paged_prefill",
    "ref_kda_causal_conv1d",
    "torch_kda_paged_decode",
]
