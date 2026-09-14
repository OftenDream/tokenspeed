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

"""Fused Q/K normalization and featurewise-beta preparation for Prefill."""

from __future__ import annotations

import torch
from tokenspeed_kernel_npu._triton import tl, triton


@triton.jit
def _prepare_kernel(
    Q,
    K,
    V,
    B,
    QO,
    KO,
    VO,
    T: tl.constexpr,
    H: tl.constexpr,
    QS: tl.constexpr,
    QH: tl.constexpr,
    KS: tl.constexpr,
    KH: tl.constexpr,
    VS: tl.constexpr,
    VH: tl.constexpr,
    BS: tl.constexpr,
    BH: tl.constexpr,
    ROWS: tl.constexpr,
):
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    t = row // H
    h = row % H
    c = tl.arange(0, 128)
    q = tl.load(
        Q + t[:, None] * QS + h[:, None] * QH + c[None, :], t[:, None] < T, 0
    ).to(tl.float32)
    k = tl.load(
        K + t[:, None] * KS + h[:, None] * KH + c[None, :], t[:, None] < T, 0
    ).to(tl.float32)
    v = tl.load(
        V + t[:, None] * VS + h[:, None] * VH + c[None, :], t[:, None] < T, 0
    ).to(tl.float32)
    b = tl.load(
        B + t[:, None] * BS + h[:, None] * BH + c[None, :], t[:, None] < T, 0
    ).to(tl.float32)
    qnorm = tl.maximum(tl.sqrt(tl.sum(q * q, 1)), 1.0e-12)
    knorm = tl.maximum(tl.sqrt(tl.sum(k * k, 1)), 1.0e-12)
    scale = tl.sqrt(1.0 / (1.0 + tl.exp(-b)) + 1.0e-10)
    off = row[:, None] * 128 + c[None, :]
    tl.store(QO + off, q / qnorm[:, None], t[:, None] < T)
    tl.store(KO + off, k / knorm[:, None] * scale, t[:, None] < T)
    tl.store(VO + off, v * scale, t[:, None] < T)


def prepare_kda_inputs(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare rank-local H4/K128 Q/K/V and scalar unit beta for ChunkKda.

    Args:
        q: BF16/FP16 query [1, tokens, 4, 128], with contiguous key channels.
        k: Key with the same shape/dtype; token and head strides may differ.
        v: Value with the same shape/dtype and contiguous channels.
        b: Featurewise beta logits with the same shape/dtype.

    Returns:
        Contiguous L2-normalized query, normalized/scaled key, scaled value,
        and unit beta [1, tokens, 4], all in the input dtype. Normalization
        and sqrt(sigmoid(beta) + 1e-10) scaling are computed in FP32.
    """
    out = [torch.empty(q.shape, device=q.device, dtype=q.dtype) for _ in range(3)]
    # Keep scalar beta initialization separate from the vector row stores.
    beta = torch.ones(q.shape[:-1], device=q.device, dtype=q.dtype)
    # Every program owns complete rows, including short final Prefill chunks.
    rows = 16 if q.shape[1] % 4 == 0 else 4
    _prepare_kernel[(triton.cdiv(q.shape[1] * q.shape[2], rows),)](
        q,
        k,
        v,
        b,
        *out,
        q.shape[1],
        q.shape[2],
        q.stride(1),
        q.stride(2),
        k.stride(1),
        k.stride(2),
        v.stride(1),
        v.stride(2),
        b.stride(1),
        b.stride(2),
        ROWS=rows,
        enable_fp_fusion=False,
    )
    return *out, beta
