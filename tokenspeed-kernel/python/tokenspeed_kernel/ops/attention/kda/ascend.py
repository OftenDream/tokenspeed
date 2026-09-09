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

"""Ascend KDA kernel registrations."""

import torch
from tokenspeed_kernel.platform import CapabilityRequirement, current_platform
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

if current_platform().is_npu:
    from tokenspeed_kernel.ops.attention.kda_reference import (
        torch_kda_paged_prefill as _torch_kda_paged_prefill,
    )
    from tokenspeed_kernel_npu.ops.kda import (
        public_kda_causal_conv1d as _public_kda_causal_conv1d,
    )
    from tokenspeed_kernel_npu.ops.kda import (
        public_kda_paged_prefill as _public_kda_paged_prefill,
    )
    from tokenspeed_kernel_npu.ops.kda import (
        ref_kda_causal_conv1d as _ref_kda_causal_conv1d,
    )
    from tokenspeed_kernel_npu.ops.kda import (
        torch_kda_paged_decode as _torch_kda_paged_decode,
    )

    _CAPABILITY = CapabilityRequirement(vendors=frozenset({"ascend"}))
    _DTYPES = {torch.float16, torch.bfloat16}

    @register_kernel(
        "attention",
        "kda_paged_prefill",
        name="torch_ascend_kda_paged_prefill",
        solution="torch",
        capability=_CAPABILITY,
        signatures=format_signatures(("q", "k", "v"), "dense", _DTYPES),
        priority=Priority.REFERENCE,
        traits={
            "beta_mode": frozenset({"scalar", "featurewise"}),
            "recurrent_layout": frozenset({"k_major"}),
        },
        tags={"ascend", "reference"},
    )
    def kda_paged_prefill(**kwargs):
        return _torch_kda_paged_prefill(**kwargs)

    @register_kernel(
        "attention",
        "kda_paged_prefill",
        name="public_ascend_kda_paged_prefill",
        solution="public_kda",
        capability=_CAPABILITY,
        signatures=format_signatures(("q", "k", "v"), "dense", _DTYPES),
        priority=Priority.SPECIALIZED,
        traits={
            "beta_mode": frozenset({"featurewise"}),
            "recurrent_layout": frozenset({"k_major"}),
        },
        tags={"ascend", "throughput"},
    )
    def public_kda_paged_prefill(**kwargs):
        return _public_kda_paged_prefill(**kwargs)

    @register_kernel(
        "attention",
        "kda_paged_decode",
        name="torch_ascend_kda_paged_decode",
        solution="torch",
        capability=_CAPABILITY,
        signatures=format_signatures(("q", "k", "v"), "dense", _DTYPES),
        priority=Priority.PORTABLE,
        traits={
            "indexed_state": frozenset({True}),
            "single_token": frozenset({True}),
            "beta_mode": frozenset({"scalar", "featurewise"}),
            "recurrent_layout": frozenset({"k_major"}),
        },
        tags={"ascend", "portability"},
    )
    def kda_paged_decode(**kwargs):
        return _torch_kda_paged_decode(**kwargs)

    @register_kernel(
        "attention",
        "kda_causal_conv1d",
        name="ref_ascend_kda_causal_conv1d",
        solution="ref",
        capability=_CAPABILITY,
        signatures=format_signatures(("projected", "weight"), "dense", _DTYPES),
        priority=Priority.PORTABLE,
        traits={
            "forward_mode": frozenset({"prefill", "decode"}),
            "batch_class": frozenset({"small", "large"}),
            "activation": frozenset({"none", "silu"}),
            "width": frozenset({4}),
        },
        tags={"ascend", "portability"},
    )
    def kda_causal_conv1d(**kwargs):
        return _ref_kda_causal_conv1d(**kwargs)

    @register_kernel(
        "attention",
        "kda_causal_conv1d",
        name="public_ascend_kda_causal_conv1d",
        solution="public_kda",
        capability=_CAPABILITY,
        signatures=format_signatures(("projected", "weight"), "dense", _DTYPES),
        priority=Priority.SPECIALIZED,
        traits={
            "forward_mode": frozenset({"prefill"}),
            "batch_class": frozenset({"small", "large"}),
            "activation": frozenset({"none", "silu"}),
            "width": frozenset({4}),
        },
        tags={"ascend", "throughput"},
    )
    def public_kda_causal_conv1d(**kwargs):
        return _public_kda_causal_conv1d(**kwargs)

    @register_kernel(
        "attention",
        "kda_causal_conv1d",
        name="public_ascend_kda_causal_conv1d_decode",
        solution="public_kda",
        capability=_CAPABILITY,
        signatures=format_signatures(("projected", "weight"), "dense", _DTYPES),
        priority=Priority.SPECIALIZED,
        traits={
            "forward_mode": frozenset({"decode"}),
            "batch_class": frozenset({"small", "large"}),
            "activation": frozenset({"none", "silu"}),
            "width": frozenset({4}),
        },
        tags={"ascend", "latency"},
    )
    def public_kda_causal_conv1d_decode(**kwargs):
        return _public_kda_causal_conv1d(**kwargs)


__all__ = [
    "kda_causal_conv1d",
    "kda_paged_decode",
    "kda_paged_prefill",
    "public_kda_causal_conv1d",
    "public_kda_paged_prefill",
]
