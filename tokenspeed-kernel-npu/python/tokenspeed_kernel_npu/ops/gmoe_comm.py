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

"""Optional 910B group-first exchange binding and process-scoped SHMEM resource."""

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import torch
import torch.distributed as dist


@dataclass
class GMoEExchange:
    """Prepared, sequential-only EP resource; no initialization in the hot path."""

    key: tuple
    process_group: object
    comm_name: str
    world_size: int
    egp_size: int
    window_bytes: int
    zero_cores: int
    num_experts: int
    shared_stream: object = None
    closed: bool = False
    router_cores: int = 0
    token_chunking: bool = False

    def check_execution_stream(self) -> None:
        """Reject pool aliasing before projection or any other main-stream work."""
        if (
            self.shared_stream is not None
            and torch.npu.current_stream(device=self.shared_stream.device)
            == self.shared_stream
        ):
            raise RuntimeError(
                "Main/graph stream aliases the limited shared stream. "
                "Reserve and reuse independent capture streams; Stream() may "
                "return a still-live stream from the pool. Do not reset the "
                "shared core quota to bypass this error."
            )

    def check_shape(self, tokens: int, hidden: int) -> None:
        if self.closed:
            raise RuntimeError("Group-first exchange resource has been closed")
        if tokens < 1 or not 16 <= hidden <= 32768 or hidden % 16:
            raise ValueError(
                "Group-first requires positive T and H in [16,32768], aligned to 16"
            )
        if not self.token_chunking and tokens > 65535:
            raise ValueError(
                "Large T requires an updated token-chunking group-first binding"
            )
        # New bindings stream logical tensors through a bounded transport
        # window. Older bindings retain the whole-tensor capacity guard.
        window_tokens = 1 if self.token_chunking else tokens
        local_bytes = (self.world_size // self.egp_size) * window_tokens * hidden * 2
        if 32768 + (self.egp_size + 2) * local_bytes > self.window_bytes:
            raise ValueError(
                "Group-first input exceeds the configured zero-restore window"
            )

    def exchange(self, grouped: torch.Tensor) -> torch.Tensor:
        self.check_shape(grouped.shape[0], grouped.shape[2])
        return torch.ops.custom.gmoe_dispatch(
            grouped, self.comm_name, self.world_size, self.egp_size
        )

    def check_router_shape(self, tokens: int, hidden: int, top_k: int) -> None:
        self.check_shape(tokens, hidden)
        if not 64 <= hidden <= 8192 or hidden % 64:
            raise ValueError("Fused router requires H in [64,8192], aligned to 64")
        local_rows = (self.world_size // self.egp_size) * (
            1 if self.token_chunking else tokens
        )
        route_bytes = local_rows * ((top_k + 7) // 8 * 8) * 8
        required = 32768 + (self.egp_size + 1) * local_rows * hidden * 2
        if required + self.egp_size * route_bytes > self.window_bytes:
            raise ValueError("Hidden and router metadata exceed the exchange window")

    def exchange_router(self, grouped, router, top_k, scaling_factor, renormalize):
        """Return gathered hidden and routes, computed only on pre-AG local rows."""
        return torch.ops.custom.gmoe_dispatch.router(
            grouped,
            router.classifier.weight,
            router.e_score_correction_bias,
            self.comm_name,
            self.world_size,
            self.egp_size,
            top_k,
            routed_scaling_factor=scaling_factor,
            renormalize=renormalize,
            router_cores=self.router_cores,
            overlap=True,
        )

    def restore_zero(self, routed, local_received, weights, ids):
        self.check_shape(
            local_received.shape[0] // (self.world_size // self.egp_size),
            local_received.shape[1],
        )
        return torch.ops.custom.gmoe_combine(
            routed,
            local_received,
            weights,
            ids,
            self.comm_name,
            self.world_size,
            self.egp_size,
            self.num_experts,
            zero_cores=self.zero_cores,
            overlap=True,
        )

    def run_shared(self, module, hidden_states):
        """Fork the limited shared stream; return output and a deferred join."""
        if self.shared_stream is None:
            return module(hidden_states), None
        main_stream = torch.npu.current_stream()
        self.shared_stream.wait_stream(main_stream)
        hidden_states.record_stream(self.shared_stream)
        with torch.npu.stream(self.shared_stream):
            output = module(hidden_states)
            ready = torch.npu.Event()
            ready.record(self.shared_stream)
        output.record_stream(main_stream)
        return output, lambda: main_stream.wait_event(ready)

    def close(self) -> None:
        """Collectively close after graphs are released; never call from __del__."""
        global _active
        if self.closed:
            return
        torch.npu.synchronize()
        torch.ops.custom.finalize_group_first_rdma(self.comm_name)
        self.closed = True
        if _active is self:
            _active = None


_active: GMoEExchange | None = None


def prepare_gmoe_exchange(
    *, ep_group, num_groups, num_experts, top_k, device, create_group, options
) -> GMoEExchange:
    """Initialize/reuse one dedicated EP runtime outside capture.

    options requires rdma_library and rendezvous (a TCP URL, or URLs keyed by
    the first global rank of each EP domain). binding optionally names a shared
    library to load if custom ops are not already registered. window_bytes and
    zero_cores tune capacity and the zero partition. All rank octets must map
    to complete physical HCCS domains; the caller owns that deployment contract.
    """
    global _active
    allowed = {
        "binding",
        "rdma_library",
        "rendezvous",
        "window_bytes",
        "zero_cores",
        "shared_overlap",
        "shared_cube_cores",
        "shared_vector_cores",
        "router_cores",
    }
    if options.keys() - allowed:
        raise ValueError(
            f"Unknown group-first exchange options: {options.keys() - allowed}"
        )
    world = len(ep_group)
    if (
        not 8 <= world <= 128
        or world % 8
        or num_groups <= 1
        or world % num_groups
        or ep_group[0] % 8
        or tuple(ep_group) != tuple(range(ep_group[0], ep_group[0] + world))
    ):
        raise ValueError(
            "Group-first requires a contiguous, octet-aligned EP domain, W=8..128, G>1 dividing W"
        )
    if not 1 <= top_k <= 64:
        raise ValueError("Group-first zero restore requires top_k in [1,64]")
    if create_group is None:
        raise ValueError("Group-first requires a dedicated process-group factory")
    if "910B" not in torch.npu.get_device_name(device).upper():
        raise ValueError("The group-first exchange backend supports Ascend 910B only")
    rendezvous = options.get("rendezvous")
    if isinstance(rendezvous, dict):
        rendezvous = rendezvous.get(str(ep_group[0]))
    if not isinstance(rendezvous, str):
        raise ValueError(
            "Set an independent group-first TCP rendezvous for each EP domain"
        )
    parsed = urlparse(rendezvous)
    if parsed.scheme != "tcp" or not parsed.hostname or not parsed.port:
        raise ValueError("Group-first rendezvous must be tcp://host:port")
    library = options.get("rdma_library")
    if not library or not Path(library).is_file():
        raise ValueError("Set rdma_library to the matching libgmoe_comm.so")
    library = str(Path(library).resolve())
    window = int(options.get("window_bytes", 32 * 1024 * 1024))
    if not 32768 <= window <= 1024 * 1024 * 1024:
        raise ValueError("Group-first window_bytes must be in [32768,1GiB]")
    egp_size = world // num_groups
    # The communication partition is fixed at eight AIV cores. Give the zero
    # partition the other half of the sixteen-core launch for every EGP size.
    zero_cores = int(options.get("zero_cores", 8))
    if not 1 <= zero_cores <= 8:
        raise ValueError("Group-first zero_cores must be in [1,8]")
    # Keep shared work on the unrestricted main stream until the stock NZ
    # QuantMatmul per-token tiling is safe across shape/core-limit changes.
    # Explicit opt-in retains the old overlap path for regression experiments.
    overlap = options.get("shared_overlap", False)
    if not isinstance(overlap, bool):
        raise ValueError("shared_overlap must be a boolean")
    router_cores = int(options.get("router_cores", 0))
    if router_cores and (not 2 <= router_cores <= 32 or router_cores % 2):
        raise ValueError("router_cores must be zero or even in [2,32]")
    shared_cube = int(options.get("shared_cube_cores", 4 if router_cores else 8))
    shared_vector = int(options.get("shared_vector_cores", 8))
    limits = torch.npu.get_device_limit(device)
    main_vector = 8 + router_cores
    main_cube = main_vector // 2 if router_cores else 1
    if main_cube > limits["cube_core_num"] or main_vector > limits["vector_core_num"]:
        raise ValueError("Insufficient cores for fused dispatch/router")
    if overlap:
        if not 1 <= shared_cube <= limits["cube_core_num"] - main_cube:
            raise ValueError("Shared stream must leave cube cores for dispatch/router")
        if not 1 <= shared_vector <= limits["vector_core_num"] - main_vector:
            raise ValueError("Shared stream must leave AIV cores for dispatch/router")
    # This binding has one process-wide runtime. Do not silently reinitialize it
    # for a different model, device, EP domain or window configuration.
    key = (
        tuple(ep_group),
        str(device),
        num_groups,
        library,
        rendezvous,
        window,
        zero_cores,
        num_experts,
        overlap,
        shared_cube,
        shared_vector,
        router_cores,
    )
    if _active is not None:
        if _active.key != key:
            raise RuntimeError(
                "Only one compatible group-first EP runtime may be active per process"
            )
        return _active
    if not hasattr(torch.ops.custom, "gmoe_dispatch"):
        binding = options.get("binding")
        if not binding or not Path(binding).is_file():
            raise ValueError(
                "Load the group-first custom binding or set its binding path"
            )
        torch.ops.load_library(str(Path(binding).resolve()))
    for name in (
        "gmoe_dispatch",
        "gmoe_combine",
        "initialize_group_first_rdma",
        "finalize_group_first_rdma",
    ):
        if not hasattr(torch.ops.custom, name):
            raise RuntimeError(
                f"Group-first binding is missing {name}; use the matching split-core build"
            )
    if router_cores and not hasattr(torch.ops.custom.gmoe_dispatch, "router"):
        raise RuntimeError("Group-first binding is missing gmoe_dispatch.router")
    process_group = create_group(tuple(ep_group), "gmoe_exchange")
    rank = dist.get_rank(process_group)
    if dist.get_world_size(process_group) != world or ep_group[rank] != dist.get_rank():
        raise RuntimeError(
            "Dedicated communicator does not match the group-first EP rank order"
        )
    comm_name = process_group._get_backend(device).get_hccl_comm_name(rank)
    torch.ops.custom.initialize_group_first_rdma(
        comm_name,
        rank,
        world,
        rendezvous,
        library,
        window_bytes=window,
    )
    shared_stream = None
    if overlap:
        shared_stream = torch.npu.Stream(device=device)
        torch.npu.set_stream_limit(
            shared_stream, cube_num=shared_cube, vector_num=shared_vector
        )
    _active = GMoEExchange(
        key,
        process_group,
        comm_name,
        world,
        egp_size,
        window,
        zero_cores,
        num_experts,
        shared_stream,
        router_cores=router_cores,
        token_chunking=(
            hasattr(torch.ops.custom, "gmoe_comm_capabilities")
            and bool(torch.ops.custom.gmoe_comm_capabilities() & 1)
        ),
    )
    return _active
