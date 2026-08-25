from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.kvcache.host_transfer import (
    _triton_is_unavailable,
    transfer_cache_ranges,
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA device"
)


def test_workspace_reuses_host_range_storage():
    from tokenspeed_kernel.ops.kvcache.host_transfer import HostTransferWorkspace

    workspace = HostTransferWorkspace()
    first = ((0, 0, 0, 8), (1, 16, 32, 24))
    count, max_bytes = workspace.load_ranges(first)
    assert count == 2
    assert max_bytes == 24
    host_ptr = workspace._range_host.data_ptr()
    count, max_bytes = workspace.load_ranges(((0, 8, 8, 16),))
    assert count == 1
    assert max_bytes == 16
    assert workspace._range_host.data_ptr() == host_ptr


def test_unrelated_triton_runtime_error_does_not_fall_back_to_dma():
    assert not _triton_is_unavailable(
        RuntimeError("requested kernel specialization is not available")
    )


@requires_cuda
@pytest.mark.parametrize("backend", ["dma", "auto", "triton"])
def test_cache_ranges_round_trip_across_multiple_device_buffers(backend):
    first = torch.arange(64, dtype=torch.uint8, device="cuda")
    second = torch.arange(48, dtype=torch.bfloat16, device="cuda")
    second_bytes = second.view(torch.uint8)
    host = torch.zeros(96, dtype=torch.uint8, pin_memory=True)
    ranges = ((0, 8, 0, 24), (1, 16, 48, 32))
    stream = torch.cuda.Stream()

    try:
        transfer_cache_ranges(
            "d2h", (first, second), host, ranges, stream, backend=backend
        )
    except RuntimeError as error:
        message = str(error).lower()
        if backend == "triton" and (
            "unavailable" in message or "not device-mapped" in message
        ):
            pytest.skip(str(error))
        raise
    stream.synchronize()
    assert torch.equal(host[0:24], first[8:32].cpu())
    assert torch.equal(host[48:80], second_bytes[16:48].cpu())

    host[0:24].fill_(7)
    host[48:80].fill_(9)
    transfer_cache_ranges("h2d", (first, second), host, ranges, stream, backend=backend)
    stream.synchronize()
    assert torch.equal(first[8:32].cpu(), torch.full((24,), 7, dtype=torch.uint8))
    assert torch.equal(
        second_bytes[16:48].cpu(), torch.full((32,), 9, dtype=torch.uint8)
    )


@requires_cuda
def test_cache_ranges_grid_stride_and_workspace_reuse():
    from tokenspeed_kernel.ops.kvcache.host_transfer import HostTransferWorkspace

    device = torch.arange(256, dtype=torch.uint8, device="cuda")
    host = torch.zeros(256, dtype=torch.uint8, pin_memory=True)
    first = tuple((0, offset, offset, 8) for offset in range(0, 128, 8))
    second = tuple((0, offset, offset, 8) for offset in range(128, 256, 8))
    workspace = HostTransferWorkspace()
    stream = torch.cuda.Stream()

    try:
        transfer_cache_ranges(
            "d2h",
            (device,),
            host,
            first,
            stream,
            backend="triton",
            workspace=workspace,
            grid_cap=3,
        )
    except RuntimeError as error:
        message = str(error).lower()
        if "unavailable" in message or "not device-mapped" in message:
            pytest.skip(str(error))
        raise
    stream.synchronize()
    assert torch.equal(host[0:128], device[0:128].cpu())
    address_ptr = workspace._address_table.data_ptr()
    range_ptr = workspace._range_device.data_ptr()

    transfer_cache_ranges(
        "d2h",
        (device,),
        host,
        second,
        stream,
        backend="triton",
        workspace=workspace,
        grid_cap=3,
    )
    stream.synchronize()
    assert torch.equal(host[128:256], device[128:256].cpu())
    assert workspace._address_table.data_ptr() == address_ptr
    assert workspace._range_device.data_ptr() == range_ptr
