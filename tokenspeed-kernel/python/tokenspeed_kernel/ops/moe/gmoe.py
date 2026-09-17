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
"""Selectable boundaries around the unchanged group-first MoE expert leaf."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from tokenspeed_kernel.selection import SelectedKernel, select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature


@dataclass(frozen=True)
class GMoEContext:
    """Model components and communication callbacks owned by the runtime.

    Callbacks keep this package independent of runtime modules/process-group
    managers. They must enqueue work on the current stream without host tensor
    reads. Modules retain their existing weights and post-load kernel plans;
    selecting a stage must not create another copy of checkpoint weights.
    The composed pre-stage reuses route; fused backends can instead consume
    router weights and the explicit routing policy without inspecting a model.
    """

    num_groups: int
    egp_group: tuple[int, ...]
    egp_rank: int
    exchange_group: tuple[int, ...]
    num_experts: int
    norm_scale: float
    proj_input: torch.nn.Module
    norm: torch.nn.Module
    shared_experts: torch.nn.Module
    proj_output: torch.nn.Module
    route: Callable
    router: torch.nn.Module
    top_k: int
    routed_scaling_factor: float
    renormalize_topk: bool
    all_to_all: Callable
    all_gather: Callable
    reduce_scatter: Callable
    ep_group: tuple[int, ...] = ()
    create_exchange_group: Callable | None = None
    exchange: Any = None
    # Optional selected shared implementation. Keep shared_experts unchanged so
    # composed and older pre-stage solutions remain independent A/B controls.
    prepared_shared: Callable | None = None


@dataclass(frozen=True)
class GMoEInputs:
    """Pre-stage outputs consumed by the expert leaf and post-stage.

    received: [EGP*G*T,H/G] inputs to init routing.
    local_received: [G*T,H/G] original local slice for identity experts.
    topk_weights/topk_ids: global-within-group routes for received.
    shared_output: [T,H], computed from the original full-hidden input.
    These tensors belong to this invocation; no mutable per-replay state is kept
    in a model/plan. Post-stage adds identity and shared contributions once.
    """

    received: torch.Tensor
    local_received: torch.Tensor
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    shared_output: torch.Tensor


@dataclass(frozen=True)
class GMoEStages:
    """Independently selected pre/post callables; the MoE leaf is not included."""

    pre: SelectedKernel
    post: SelectedKernel

    def prepare(self, context, *, device, options=None) -> GMoEContext:
        """Prepare selected backend resources outside capture, once per EP domain.

        Each distinct implementation prepare hook receives the current context,
        bound device and deployment options, and returns an updated context.
        Composed stages have no hook and allocate no communication resources.
        """
        prepared = set()
        for kernel in (self.pre, self.post):
            prepare = getattr(kernel.impl, "prepare", None)
            if prepare is not None and prepare not in prepared:
                context = prepare(context, device=device, options=options or {})
                prepared.add(prepare)
        return context


def select_gmoe_stages(
    *,
    input_dtype: torch.dtype,
    traits: dict[str, Any],
    pre_solution: str | None = None,
    post_solution: str | None = None,
) -> GMoEStages:
    """Select pre/post kernels once during model preparation, outside forward.

    The pre kernel accepts hidden_states/context and returns GMoEInputs.
    The post kernel accepts routed/inputs/context and returns [T,H] output.
    traits describe placement, dimensions and quantization for backend filtering.
    A solution restricts selection without bypassing capability/signature/trait
    checks. None selects automatically. The optional exchange backend is only
    eligible when deployment options explicitly enable its resource setup.
    """
    traits = {"gmoe_exchange_enabled": False, **traits}
    signature = format_signature(hidden_states=dense_tensor_format(input_dtype))
    return GMoEStages(
        pre=select_kernel(
            "moe",
            "gmoe_pre",
            signature,
            traits=traits,
            solution=pre_solution,
        ),
        post=select_kernel(
            "moe",
            "gmoe_post",
            signature,
            traits=traits,
            solution=post_solution,
        ),
    )
