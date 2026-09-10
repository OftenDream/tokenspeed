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

"""Optional shared FFN binding; checkpoint and NZ ownership stay in the MLP."""

from pathlib import Path

import torch


def prepare_shared_ffn(module, *, binding=None):
    """Return a callable BF16/W8A8 FFN using the original TP1 MLP's weights.

    module supplies gate_up_proj, down_proj and an unclamped SwiGLU. binding
    optionally names the deployed custom-op library; it is loaded only here,
    outside forward/capture. No weights are copied or post-load hooks repeated.
    INT8 plans are resolved when invoked because parent preparation can precede
    child Linear post-load processing. The returned callable produces [T,H].
    """
    gate, down = module.gate_up_proj, module.down_proj
    for projection in (gate, down):
        if projection.tp_size != 1 or projection.bias is not None:
            raise ValueError("Fused shared FFN requires bias-free TP1 projections")
        if projection.params_dtype != torch.bfloat16:
            raise ValueError("Fused shared FFN requires BF16 activations/output")
    if getattr(module.act_fn, "swiglu_limit", None) is not None:
        raise ValueError("Fused shared FFN does not support clamped SwiGLU")
    if down.reduce_results or getattr(gate, "interleave_linear_and_gate", False):
        raise ValueError("Fused shared FFN requires local, non-interleaved weights")
    dtype = gate.weight.dtype
    if dtype != down.weight.dtype or dtype not in (torch.int8, torch.bfloat16):
        raise ValueError("Fused shared FFN requires two BF16 or two INT8 weights")
    if dtype == torch.int8 and not module._use_int8_swiglu:
        raise ValueError("Fused shared FFN requires dynamic-token symmetric W8A8")
    h, i = gate.weight.shape[1], down.weight.shape[1]
    if gate.weight.shape != (2 * i, h) or down.weight.shape[0] != h:
        raise ValueError("Fused shared FFN requires [2I,H] and [H,I] weights")
    if h % 32 or i % 32:
        raise ValueError("Fused shared FFN requires H/I aligned to 32")
    if not hasattr(torch.ops.custom, "fused_ffn"):
        if binding is None or not Path(binding).is_file():
            raise ValueError("Set ffn_binding to the deployed fused_ffn binding")
        torch.ops.load_library(str(Path(binding).resolve()))
    op = torch.ops.custom.fused_ffn

    if dtype == torch.int8:

        def forward(x):
            # Read the final child plans, never cache a pre-load NZ allocation.
            first, second = gate._int8_plan, down._int8_plan
            return op(
                x, first.weight, second.weight, first.weight_scale, second.weight_scale
            )

    else:

        def forward(x):
            return op(x, gate.weight, down.weight)

    return forward
