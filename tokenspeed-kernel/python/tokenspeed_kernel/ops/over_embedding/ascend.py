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

"""Ascend LongCat over-embedding registrations."""

from __future__ import annotations

import torch
from tokenspeed_kernel.ops.over_embedding.spec import OverEmbeddingSpec
from tokenspeed_kernel.platform import CapabilityRequirement, current_platform
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

if current_platform().is_npu:
    from tokenspeed_kernel_npu.ops.over_embedding import (
        append_packed_lookup as _append_packed_lookup,
    )
    from tokenspeed_kernel_npu.ops.over_embedding import (
        register_host_tables as _register_host_tables,
    )

    _CAPABILITY = CapabilityRequirement(vendors=frozenset({"ascend"}))
    _TRAITS = {
        "fragment_count": frozenset({2, 3}),
        "fragment_widths": frozenset(
            {
                (256, 128),
                (256, 256),
                (256, 256, 256),
            }
        ),
    }

    @register_kernel(
        "over_embedding",
        "append_packed_lookup",
        name="ascend_longcat_oe_append_packed_lookup",
        solution="flash_npu_kernel",
        capability=_CAPABILITY,
        signatures=format_signatures(("table",), "dense", {torch.bfloat16}),
        priority=Priority.SPECIALIZED,
        traits=_TRAITS,
        tags={
            "ascend",
            "host-or-device-table",
            "variable-length",
            "deterministic",
        },
    )
    def ascend_append_packed_lookup(
        *,
        input_ids: torch.Tensor,
        input_start_offsets: torch.Tensor,
        req_pool_indices: torch.Tensor,
        active_request_mask: torch.Tensor,
        history_token_ids: torch.Tensor,
        committed_lengths: torch.Tensor,
        oe_tables: tuple[torch.Tensor, ...],
        out: torch.Tensor,
        spec: OverEmbeddingSpec,
        enable_pdl: bool,
    ) -> None:
        del enable_pdl
        _append_packed_lookup(
            input_ids=input_ids,
            input_start_offsets=input_start_offsets,
            req_pool_indices=req_pool_indices,
            active_request_mask=active_request_mask,
            history_token_ids=history_token_ids,
            committed_lengths=committed_lengths,
            oe_tables=oe_tables,
            out=out,
            spec=spec,
        )

    @register_kernel(
        "over_embedding",
        "register_host_tables",
        name="ascend_longcat_oe_register_host_tables",
        solution="flash_npu_kernel",
        capability=_CAPABILITY,
        signatures=format_signatures(("table",), "dense", {torch.bfloat16}),
        priority=Priority.SPECIALIZED,
        traits=_TRAITS,
        tags={"ascend", "host-table", "setup"},
    )
    def ascend_register_host_tables(
        *, oe_tables: tuple[torch.Tensor, ...], device: torch.device | str | int
    ) -> None:
        _register_host_tables(oe_tables=oe_tables, device=device)


__all__ = [
    "ascend_append_packed_lookup",
    "ascend_register_host_tables",
]
