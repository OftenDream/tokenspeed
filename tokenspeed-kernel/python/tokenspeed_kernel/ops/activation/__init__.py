# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Activation kernel entry points."""

from __future__ import annotations

import torch
from tokenspeed_kernel.ops.activation.flashinfer import (
    silu_and_mul as flashinfer_silu_and_mul,
)
from tokenspeed_kernel.ops.activation.triton import (
    add3,
)
from tokenspeed_kernel.ops.activation.triton import sigmoid_mul as triton_sigmoid_mul
from tokenspeed_kernel.ops.activation.triton import silu_and_mul as triton_silu_and_mul
from tokenspeed_kernel.ops.activation.triton import situ_and_mul as triton_situ_and_mul
from tokenspeed_kernel.ops.gemm import _fp8_linear_activation
from tokenspeed_kernel.platform import current_platform, pdl_enabled
from tokenspeed_kernel.registry import error_fn


def silu_and_mul(
    x: torch.Tensor,
    out: torch.Tensor | None = None,
    limit: float | None = None,
) -> torch.Tensor:
    """Apply SwiGLU through the platform implementation.

    ``x`` contains concatenated gate/up values on the last dimension. Return
    ``SiLU(gate) * up`` in the input dtype, writing ``out`` when provided.
    Positive ``limit`` clamps gate from above and up on both sides before SiLU.
    CPU and clamped Ascend calls preserve the FP32 native calculation; ordinary
    Ascend calls use fused SwiGLU. CUDA/AMD selection is unchanged.
    """
    if x.ndim == 0 or x.shape[-1] % 2:
        raise ValueError("SwiGLU expects an even [gate, up] width")
    if out is not None:
        if out.shape != x.shape[:-1] + (x.shape[-1] // 2,):
            raise ValueError("out shape must match the SwiGLU output")
        if out.dtype != x.dtype or out.device != x.device:
            raise ValueError("out dtype and device must match x")
    if limit is not None and limit <= 0:
        limit = None
    if x.device.type == "cpu" or (current_platform().is_npu and limit is not None):
        gate, up = x.float().chunk(2, dim=-1)
        if limit is not None:
            gate = gate.clamp_max(limit)
            up = up.clamp(-limit, limit)
        result = (torch.nn.functional.silu(gate) * up).to(x.dtype)
        if out is not None:
            out.copy_(result)
            return out
        return result
    if current_platform().is_npu:
        from tokenspeed_kernel.ops.activation.ascend import silu_and_mul

        return silu_and_mul(x, out)
    if (
        limit is not None
        or current_platform().is_amd
        or flashinfer_silu_and_mul is error_fn
    ):
        return triton_silu_and_mul(x, out, enable_pdl=pdl_enabled(), limit=limit)
    return flashinfer_silu_and_mul(x, out, enable_pdl=pdl_enabled())


def silu_and_mul_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return INT8 SwiGLU values and FP32 per-row dequantization scales.

    ``x`` is floating [gate,up], with at least two dimensions and an even last
    dimension of at most 8192. Reconstruct with ``values.float() * scale[...,None]``.
    This entry currently supports Ascend; INT32 matmul accumulators instead use
    :func:`dequant_silu_and_mul_quant` with weight and activation scales.
    """
    if x.ndim < 2 or x.shape[-1] % 2 or x.shape[-1] > 8192:
        raise ValueError("Quantized SwiGLU requires even-width rows of at most 8192")
    if x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("SwiGLU quantization requires floating input")
    if not current_platform().is_npu or x.device.type == "cpu":
        raise NotImplementedError("INT8-output SwiGLU currently requires Ascend")
    from tokenspeed_kernel.ops.activation.ascend import silu_and_mul_quant

    return silu_and_mul_quant(x)


def dequant_silu_and_mul_quant(
    x: torch.Tensor, weight_scale: torch.Tensor, activation_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply SwiGLU to scaled INT32 accumulators and return INT8 rows + scales.

    ``x`` stores [gate,up] in its last dimension. ``weight_scale`` is FP32
    with one value per column; ``activation_scale`` is FP32 per input row.
    Output scales have shape ``x.shape[:-1]`` and reconstruct values as
    ``quantized.float() * scales.unsqueeze(-1)``. Currently supported on Ascend.
    """
    if x.ndim < 2 or x.shape[-1] % 2 or x.dtype != torch.int32:
        raise ValueError("Quantized SwiGLU requires even-width INT32 [gate,up] rows")
    if weight_scale.shape != (x.shape[-1],) or activation_scale.shape != x.shape[:-1]:
        raise ValueError("SwiGLU scales must match input columns and rows")
    if any(
        s.dtype != torch.float32 or s.device != x.device
        for s in (weight_scale, activation_scale)
    ):
        raise ValueError("SwiGLU scales must be FP32 on the input device")
    if not current_platform().is_npu or x.device.type == "cpu":
        raise NotImplementedError("INT8-output SwiGLU currently requires Ascend")
    from tokenspeed_kernel.ops.activation.ascend import dequant_silu_and_mul_quant

    return dequant_silu_and_mul_quant(x, weight_scale, activation_scale)


def sigmoid_mul(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Apply ``x *= sigmoid(gate)`` through a portable backend boundary."""
    if current_platform().is_npu or x.device.type == "cpu":
        if x.ndim != 2 or not x.is_contiguous():
            raise ValueError("x must be contiguous 2D")
        if x.dtype != gate.dtype:
            raise ValueError(f"dtype mismatch: x={x.dtype} gate={gate.dtype}")
        if gate.ndim == 3:
            if (
                gate.shape[0] != x.shape[0]
                or gate.shape[1] * gate.shape[2] != x.shape[1]
            ):
                raise ValueError(f"shape mismatch: x={x.shape} gate={gate.shape}")
            gate = gate.reshape_as(x)
        elif gate.ndim != 2 or gate.shape != x.shape:
            raise ValueError(f"shape mismatch: x={x.shape} gate={gate.shape}")
        x.mul_(torch.sigmoid(gate.float()).to(x.dtype))
        return x
    return triton_sigmoid_mul(x, gate)


def prepare_fp8_linear_activation(
    plan: object,
    x: torch.Tensor,
    *,
    activation: str,
    limit: float | None = None,
    alpha: float = 1.0,
    beta: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Prepare an activation for a compatible block-FP8 linear plan.

    The prepared linear implementation decides whether it can fuse activation
    and quantization. ``None`` means the caller must evaluate the activation
    normally and invoke the linear operation through its ordinary path.

    Args:
        plan: Opaque plan returned by the GEMM layer's ``prepare_fp8_linear``.
        x: Input to the activation.
        activation: Semantic activation name, currently ``"swiglu"``.
        limit: Optional activation clamp limit.
        alpha: Sigmoid multiplier for SwiGLU.
        beta: Value added to SwiGLU's up branch.

    Returns:
        Prepared FP8 values and scales, or ``None`` when no fused contract is
        available.
    """
    return _fp8_linear_activation(
        plan,
        x,
        activation=activation,
        limit=limit,
        alpha=alpha,
        beta=beta,
        enable_pdl=pdl_enabled(),
    )


def situ_and_mul(
    x: torch.Tensor,
    out: torch.Tensor | None = None,
    *,
    beta: float = 1.0,
    linear_beta: float | None = None,
) -> torch.Tensor:
    """Apply SiTU through the portable Triton implementation."""

    return triton_situ_and_mul(
        x,
        out,
        beta=beta,
        linear_beta=linear_beta,
        enable_pdl=pdl_enabled(),
    )


__all__ = [
    "add3",
    "dequant_silu_and_mul_quant",
    "prepare_fp8_linear_activation",
    "sigmoid_mul",
    "silu_and_mul",
    "silu_and_mul_quant",
    "situ_and_mul",
]
