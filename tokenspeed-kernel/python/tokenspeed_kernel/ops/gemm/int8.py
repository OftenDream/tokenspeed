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

"""Dynamic-token, symmetric per-channel INT8 linear kernel boundary."""

import torch
from tokenspeed_kernel.platform import current_platform


def prepare_int8_linear(
    weight: torch.Tensor, weight_scale: torch.Tensor, *, output_dtype: torch.dtype
) -> torch.nn.Module:
    """Prepare canonical [N,K] INT8 weights and [N,1] scales once after loading.

    Return a callable plan accepting floating input or (INT8 values, FP32 token
    scales). With defer_dequant=True it returns (INT32 accumulators, token scales);
    otherwise it returns dequantized values in output_dtype.
    """
    if not current_platform().is_npu:
        raise NotImplementedError("Dynamic INT8 linear currently requires Ascend")
    from tokenspeed_kernel_npu.ops.gemm import PreparedInt8Linear

    return PreparedInt8Linear(weight, weight_scale, output_dtype=output_dtype)
