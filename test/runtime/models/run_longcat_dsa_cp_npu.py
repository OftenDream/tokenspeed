"""Eight-rank LongCat DSA context-parallel operator-chain validation.

Run with ``torchrun --nproc-per-node=8`` after loading the matching flash_ops
Torch extension and OPP package.  This is intentionally a standalone hardware
test: ordinary pytest collection must not initialize HCCL.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401
from tokenspeed_kernel_npu.ops.longcat_dsa import AscendDSAKernels

from tokenspeed.runtime.layers.attention.dcp.metadata import (
    PositionPreservingDCPLayout,
    refresh_dcp_page_table_metadata,
)

DEGREE = 8
HEADS = 64
HEADS_PER_RANK = HEADS // DEGREE
TOKENS = 8
PAGE_SIZE = 128
PAGES_PER_REQUEST = 32
SEQUENCE_LENGTH = PAGE_SIZE * PAGES_PER_REQUEST
TOPK = 2048


def _all_gather(tensor: torch.Tensor) -> torch.Tensor:
    gathered = [torch.empty_like(tensor) for _ in range(DEGREE)]
    dist.all_gather(gathered, tensor)
    return torch.stack(gathered)


def _owned_metadata(rank: int, table: torch.Tensor):
    placement = refresh_dcp_page_table_metadata(
        layout=PositionPreservingDCPLayout(),
        page_table=table,
        virtual_block_count=1 + TOKENS * PAGES_PER_REQUEST,
        degree=DEGREE,
        rank=rank,
        previous=None,
    )
    owned = placement.owner_mask
    mapped = torch.where(owned, table, 0)
    permutation = (~owned).to(torch.int32).argsort(dim=1, stable=True)
    return (
        mapped.gather(1, permutation),
        owned.sum(dim=1, dtype=torch.int32) * PAGE_SIZE,
        owned[:, :1].sum(dim=1, dtype=torch.int32) * 4,
        owned[:, -8:].sum(dim=1, dtype=torch.int32) * PAGE_SIZE,
    )


def main() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != DEGREE:
        raise RuntimeError(f"expected {DEGREE} ranks, got {world_size}")
    torch.npu.set_device(local_rank)
    dist.init_process_group("hccl")
    try:
        torch.manual_seed(20260916)
        kernels = AscendDSAKernels()
        kernels.require_context_parallel()

        total_pages = 1 + TOKENS * PAGES_PER_REQUEST
        kv_cache = (
            torch.randn(
                total_pages,
                PAGE_SIZE,
                1,
                576,
                dtype=torch.bfloat16,
                device="npu",
            )
            * 0.02
        )
        index_cache = (
            torch.randn(
                total_pages,
                PAGE_SIZE,
                1,
                128,
                dtype=torch.bfloat16,
                device="npu",
            )
            * 0.02
        )
        full_query = (
            torch.randn(TOKENS, HEADS, 576, dtype=torch.bfloat16, device="npu") * 0.02
        )
        index_query = (
            torch.randn(TOKENS, 16, 128, dtype=torch.bfloat16, device="npu") * 0.02
        )
        index_weights = (
            torch.randn(TOKENS, 16, dtype=torch.bfloat16, device="npu") * 0.02
        )
        table = (
            torch.arange(
                1,
                total_pages,
                dtype=torch.int32,
                device="npu",
            )
            .view(TOKENS, PAGES_PER_REQUEST)
            .contiguous()
        )
        lengths = torch.full(
            (TOKENS,), SEQUENCE_LENGTH, dtype=torch.int32, device="npu"
        )
        query_ends = torch.arange(1, TOKENS + 1, dtype=torch.int32, device="npu")
        local_table, local_lengths, init_counts, local_counts = _owned_metadata(
            rank, table
        )

        local_indices, local_values = kernels.index_partial(
            index_query,
            index_cache,
            index_weights,
            query_ends,
            local_lengths,
            local_table,
            TOPK,
            init_counts,
            local_counts,
            sparse_mode=3,
        )

        received_values = torch.empty_like(local_values.squeeze(1).reshape(-1))
        dist.all_to_all_single(
            received_values, local_values.squeeze(1).contiguous().reshape(-1)
        )
        global_values = (
            received_values.view(DEGREE, TOKENS // DEGREE, TOPK)
            .transpose(0, 1)
            .contiguous()
            .view(TOKENS // DEGREE, DEGREE * TOPK)
        )
        global_positions = global_values.topk(TOPK, dim=1).indices.to(torch.int32)
        global_positions = _all_gather(global_positions).flatten(0, 1)
        sparse_indices, valid_chunks = kernels.select_local(
            local_indices, global_positions, rank
        )

        local_query = full_query[
            :, rank * HEADS_PER_RANK : (rank + 1) * HEADS_PER_RANK
        ].contiguous()
        packed_query = _all_gather(local_query)
        partial, softmax_max, softmax_sum = kernels.attention_partial(
            packed_query,
            kv_cache,
            sparse_indices,
            valid_chunks,
            query_ends,
            local_lengths,
            local_table,
            192**-0.5,
            sparse_mode=3,
        )

        output_send = partial.view(DEGREE, HEADS_PER_RANK, TOKENS, 512).contiguous()
        output_recv = torch.empty_like(output_send)
        dist.all_to_all_single(output_recv, output_send)
        lse_send = (
            (softmax_max + torch.log(softmax_sum))
            .squeeze(0)
            .transpose(0, 1)
            .reshape(DEGREE, HEADS_PER_RANK, TOKENS)
            .contiguous()
        )
        lse_recv = torch.empty_like(lse_send)
        dist.all_to_all_single(lse_recv, lse_send)
        actual = (
            kernels.merge_partials(
                output_recv.reshape(DEGREE, HEADS_PER_RANK * TOKENS, 512),
                lse_recv.reshape(DEGREE, HEADS_PER_RANK * TOKENS),
            )
            .view(HEADS_PER_RANK, TOKENS, 512)
            .transpose(0, 1)
            .contiguous()
        )

        full_indices, _ = kernels.index_partial(
            index_query,
            index_cache,
            index_weights,
            query_ends,
            lengths,
            table,
            TOPK,
            torch.full_like(lengths, 4),
            torch.full_like(lengths, 1024),
            sparse_mode=3,
        )
        full_chunks = (
            (full_indices >= 0).sum(dim=(1, 2), dtype=torch.int32) + 127
        ) // 128
        expected = kernels.attention(
            full_query[..., :512].contiguous(),
            full_query[..., 512:].contiguous(),
            kv_cache,
            full_indices,
            full_chunks,
            query_ends,
            lengths,
            table,
            192**-0.5,
        )[:, rank * HEADS_PER_RANK : (rank + 1) * HEADS_PER_RANK]

        error = (actual.float() - expected.float()).abs()
        metrics = torch.tensor(
            [error.max().item(), error.mean().item()],
            dtype=torch.float32,
            device="npu",
        )
        dist.all_reduce(metrics, op=dist.ReduceOp.MAX)
        if rank == 0:
            print(
                "LongCat DSA CP8 operator chain passed: "
                f"max_abs={metrics[0].item():.6f}, "
                f"max_rank_mean_abs={metrics[1].item():.6f}",
                flush=True,
            )
        if metrics[0].item() > 0.08 or metrics[1].item() > 0.01:
            raise AssertionError(f"context-parallel error is too large: {metrics}")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
