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

"""Ascend implementation of the paged LongCat DSA backend."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import torch
from tokenspeed_kernel.ops.attention.dsa.ascend import ascend_dsa_kernels
from tokenspeed_kernel.platform import current_platform

from tokenspeed.runtime.configs.model_config import AttentionArch
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.execution.forward_step import (
    get_is_capture_mode,
    get_is_cuda_graph_phase,
)
from tokenspeed.runtime.layers.attention.backends.paged.dsa import DSABackend
from tokenspeed.runtime.layers.attention.backends.paged.mla import MLAAttnBackend
from tokenspeed.runtime.layers.attention.kernel_page_sizes import ASCEND_SFAD_PAGE_SIZE
from tokenspeed.runtime.layers.attention.registry import register_backend
from tokenspeed.runtime.utils.cuda_stream import (
    StreamFork,
    limit_stream_cores,
    new_device_stream,
)


@dataclass(frozen=True)
class _AscendSparseSelection:
    query: torch.Tensor
    indices: torch.Tensor
    valid_chunks: torch.Tensor
    q_ends: torch.Tensor
    kv_lengths: torch.Tensor
    table: torch.Tensor
    sparse_mode: int
    context_parallel: bool


class AscendDSABackend(DSABackend):
    """LongCat DSA execution backed by Ascend indexer and sparse-attention ops."""

    default_kernel_page_size = ASCEND_SFAD_PAGE_SIZE

    @classmethod
    def resolve_kernel_page_size(cls, config, block_granularity: int) -> int:
        del block_granularity
        if config.kernel_page_size not in (None, ASCEND_SFAD_PAGE_SIZE):
            raise ValueError(
                "Ascend SparseFlashAttentionDecode requires "
                f"kernel_page_size={ASCEND_SFAD_PAGE_SIZE}, got "
                f"{config.kernel_page_size}"
            )
        return ASCEND_SFAD_PAGE_SIZE

    def _create_dense_leaf(
        self,
        config,
        spec,
        platform,
        kernel_page_size: int,
    ):
        del platform
        dense_spec = dataclasses.replace(spec, backend_name=None)
        return MLAAttnBackend(config, dense_spec, kernel_page_size=kernel_page_size)

    def __init__(
        self,
        config,
        spec,
        *,
        kernel_page_size: int,
    ) -> None:
        if not (spec.is_dsa and spec.uses_separate_bf16_index_cache):
            raise NotImplementedError(
                "Ascend DSA currently requires the LongCat BF16 indexer/cache contract"
            )
        if kernel_page_size != ASCEND_SFAD_PAGE_SIZE:
            raise ValueError(
                "Ascend SparseFlashAttentionDecode requires "
                f"kernel_page_size={ASCEND_SFAD_PAGE_SIZE}, got {kernel_page_size}"
            )
        if config.kv_cache_dtype != torch.bfloat16:
            raise ValueError("LongCat DSA currently supports BF16 KV only")
        if config.kv_cache_quant_method not in (None, "none"):
            raise ValueError("LongCat DSA requires unquantized BF16 cache planes")
        super().__init__(config, spec, kernel_page_size=kernel_page_size)
        if self.spec_num_tokens != 1 or self.is_draft:
            raise ValueError("LongCat DSA MTP is disabled")
        self._indexer_spec = spec
        self._indexer_kernels = ascend_dsa_kernels()
        self._stream_fork = StreamFork(new_device_stream())
        self._init_context_parallel(config, spec, self._indexer_kernels)

    def run_projection_branches(self, layer, primary, secondary):
        del layer
        graph_phase = get_is_cuda_graph_phase()
        stream_fork = self._stream_fork
        device_module = stream_fork.device_module
        main_stream = (
            device_module.current_stream()
            if graph_phase and device_module is not None
            else None
        )
        with (
            limit_stream_cores(
                main_stream,
                cube_num=12,
                vector_num=24,
                enable=graph_phase,
            ),
            limit_stream_cores(
                stream_fork.aux_stream,
                cube_num=12,
                vector_num=24,
                enable=graph_phase,
            ),
            stream_fork.scope(
                enable=graph_phase,
                overlap=get_is_capture_mode(),
            ) as fork,
        ):
            primary_result = primary()
            with fork.branch():
                secondary_result = secondary()
        return primary_result, secondary_result

    def refresh_decode_metadata(
        self,
        bs: int,
        actual_bs: int,
        seq_lens: torch.Tensor,
        page_table: torch.Tensor,
        *,
        num_extends: int = 0,
        for_graph_replay: bool = False,
    ) -> None:
        self._dense_backend.refresh_decode_metadata(
            bs,
            actual_bs,
            seq_lens,
            page_table,
            num_extends=num_extends,
            for_graph_replay=for_graph_replay,
        )
        if self.dcp_size > 1:
            metadata = self.forward_decode_metadata
            self._refresh_dcp_metadata(
                lengths=metadata.seq_lens,
                table=metadata.page_table,
                page_size=self.kernel_page_size,
                initial_tokens=self.index_init_tokens,
                local_tokens=self.index_local_tokens,
            )

    def init_forward_metadata(
        self,
        bs: int,
        num_extends: int,
        seq_lens: torch.Tensor,
        page_table: torch.Tensor,
        forward_mode: ForwardMode,
        *,
        extend_seq_lens: torch.Tensor,
        extend_seq_lens_cpu: torch.Tensor,
        extend_prefix_lens: torch.Tensor,
        extend_prefix_lens_cpu: torch.Tensor,
        extend_with_prefix: bool,
        **kwargs,
    ):
        if not (forward_mode.is_extend_or_mixed() or forward_mode.is_idle()):
            raise RuntimeError(
                "DSA decode metadata goes through refresh_decode_metadata; "
                f"init_forward_metadata only serves extend/mixed ({forward_mode})"
            )
        self._dense_backend.init_forward_metadata(
            bs,
            num_extends,
            seq_lens,
            page_table,
            forward_mode,
            extend_seq_lens=extend_seq_lens,
            extend_seq_lens_cpu=extend_seq_lens_cpu,
            extend_prefix_lens=extend_prefix_lens,
            extend_prefix_lens_cpu=extend_prefix_lens_cpu,
            extend_with_prefix=extend_with_prefix,
            **kwargs,
        )
        self._prefill_page_table = None
        if num_extends > 0 and forward_mode.is_extend_or_mixed():
            cmeta = self._dense_backend.chunked_prefill_metadata
            if cmeta is not None:
                self._prefill_page_table = page_table[:num_extends]
                cmeta.page_table = self._prefill_page_table

    def _select_indexed(
        self,
        q,
        k,
        layer,
        out_cache_loc,
        pool,
        q_ends,
        kv_lengths,
        table,
        kwargs,
        *,
        save_kv_cache=True,
        context_parallel=False,
        context_parallel_row_start=0,
    ):
        """Write BF16 cache planes and select request-local sparse indices.

        Query ends delimit the TND request spans; page identities and write
        slots belong to the router. Sparse attention remains in the existing
        forward_sparse_prefill / forward_sparse_decode entry points.
        """
        spec = self._indexer_spec
        kernels = self._indexer_kernels
        kv_cache = pool.get_key_buffer(layer.layer_id)
        index_cache = pool.get_component(layer.layer_id, "dsa_index_k")
        page = self.kernel_page_size
        kv_cache = kv_cache.view(-1, page, 1, spec.kv_cache_dim)
        index_cache = index_cache.view(-1, page, 1, spec.index_head_dim)
        if save_kv_cache:
            kernels.scatter(k.contiguous(), kv_cache, out_cache_loc)
        kernels.scatter(kwargs["index_key"].contiguous(), index_cache, out_cache_loc)
        if self.step_counter is not None:
            self.step_counter.record_cache()
        if not context_parallel or self.dcp_size == 1:
            indices, valid_chunks = kernels.index(
                kwargs["index_query"],
                index_cache,
                kwargs["index_weights"],
                q_ends,
                kv_lengths,
                table,
                spec.index_topk,
                self.index_init_tokens,
                self.index_local_tokens,
            )
            return _AscendSparseSelection(
                q,
                indices,
                valid_chunks,
                q_ends,
                kv_lengths,
                table,
                3,
                False,
            )

        if self._dcp_process_group is None:
            raise RuntimeError("LongCat DSA CP process group was not configured")
        row_end = context_parallel_row_start + table.shape[0]
        local_table = self._dcp_page_table[context_parallel_row_start:row_end]
        local_lengths = self._dcp_seq_lens[context_parallel_row_start:row_end]
        init_counts = self._dcp_init_counts[context_parallel_row_start:row_end]
        local_counts = self._dcp_local_counts[context_parallel_row_start:row_end]
        sparse_mode = 3 if self.dcp_rank == 0 else 0
        local_indices, local_values = kernels.index_partial(
            kwargs["index_query"],
            index_cache,
            kwargs["index_weights"],
            q_ends,
            local_lengths,
            local_table,
            spec.index_topk,
            init_counts,
            local_counts,
            sparse_mode=sparse_mode,
        )

        local_query = q.contiguous()
        tokens = local_values.shape[0]
        candidates = local_values.shape[2]
        padded_tokens = (tokens + self.dcp_size - 1) // self.dcp_size * self.dcp_size
        if padded_tokens != tokens:
            local_values = torch.cat(
                (
                    local_values,
                    torch.full(
                        (padded_tokens - tokens, 1, candidates),
                        float("-inf"),
                        dtype=local_values.dtype,
                        device=local_values.device,
                    ),
                )
            )
        tokens_per_rank = padded_tokens // self.dcp_size
        received_values = torch.empty(
            self.dcp_size * tokens_per_rank * candidates,
            dtype=local_values.dtype,
            device=local_values.device,
        )
        graph_phase = get_is_cuda_graph_phase()
        capture_mode = get_is_capture_mode()
        overlap_fork = self._stream_fork
        device_module = overlap_fork.device_module
        main_stream = (
            device_module.current_stream()
            if graph_phase and device_module is not None
            else None
        )

        def gather_query():
            group = self._dcp_aux_process_group or self._dcp_process_group
            return self._dcp_all_gather(local_query, group).view(
                self.dcp_size, *local_query.shape
            )

        can_overlap_query = graph_phase and self._dcp_aux_process_group is not None
        if can_overlap_query:
            with (
                limit_stream_cores(
                    main_stream, cube_num=12, vector_num=24, enable=True
                ),
                limit_stream_cores(
                    overlap_fork.aux_stream,
                    cube_num=12,
                    vector_num=24,
                    enable=True,
                ),
                overlap_fork.scope(enable=True, overlap=capture_mode) as fork,
            ):
                with fork.branch():
                    packed_query = gather_query()
                self._dcp_all_to_all(
                    received_values,
                    local_values.squeeze(1).contiguous().view(-1),
                    self._dcp_process_group,
                )
        else:
            self._dcp_all_to_all(
                received_values,
                local_values.squeeze(1).contiguous().view(-1),
                self._dcp_process_group,
            )
            packed_query = gather_query()

        with limit_stream_cores(
            main_stream, cube_num=16, vector_num=32, enable=graph_phase
        ):
            global_values = (
                received_values.view(self.dcp_size, tokens_per_rank, candidates)
                .transpose(0, 1)
                .contiguous()
                .view(tokens_per_rank, self.dcp_size * candidates)
            )
            _, global_positions = global_values.topk(spec.index_topk, dim=1)
            global_positions = self._dcp_all_gather(
                global_positions.to(torch.int32).contiguous(),
                self._dcp_process_group,
            )[:tokens]
            indices, valid_chunks = kernels.select_local(
                local_indices,
                global_positions,
                self.dcp_rank,
            )

        return _AscendSparseSelection(
            packed_query,
            indices,
            valid_chunks,
            q_ends,
            local_lengths,
            local_table,
            sparse_mode,
            True,
        )

    def _run_indexed_sparse_attention(
        self,
        selection,
        layer,
        token_to_kv_pool,
        *,
        head_major_output,
    ):
        spec = self._indexer_spec
        kv_cache = token_to_kv_pool.get_key_buffer(layer.layer_id).view(
            -1,
            self.kernel_page_size,
            1,
            spec.kv_cache_dim,
        )
        expected_width = spec.kv_lora_rank + spec.qk_rope_head_dim
        if selection.query.shape[-1] != expected_width:
            raise ValueError(
                f"LongCat DSA query width must be {expected_width}, got "
                f"{selection.query.shape[-1]}"
            )
        kernels = self._indexer_kernels
        if not selection.context_parallel:
            result = kernels.attention(
                selection.query[..., : spec.kv_lora_rank].contiguous(),
                selection.query[..., spec.kv_lora_rank :].contiguous(),
                kv_cache,
                selection.indices,
                selection.valid_chunks,
                selection.q_ends,
                selection.kv_lengths,
                selection.table,
                layer.scaling,
            )
            return (
                result.transpose(0, 1).contiguous()
                if head_major_output
                else result.flatten(1)
            )

        output, softmax_max, softmax_sum = kernels.attention_partial(
            selection.query,
            kv_cache,
            selection.indices,
            selection.valid_chunks,
            selection.q_ends,
            selection.kv_lengths,
            selection.table,
            layer.scaling,
            sparse_mode=selection.sparse_mode,
        )
        tokens = selection.indices.shape[0]
        heads = spec.num_attention_heads
        heads_per_rank = heads // self.dcp_size
        graph_phase = get_is_cuda_graph_phase()
        capture_mode = get_is_capture_mode()
        overlap_fork = self._stream_fork
        device_module = overlap_fork.device_module
        main_stream = (
            device_module.current_stream()
            if graph_phase and device_module is not None
            else None
        )

        def update_output():
            output_send = output.view(
                self.dcp_size, heads_per_rank, tokens, spec.kv_lora_rank
            ).contiguous()
            output_recv = torch.empty_like(output_send)
            self._dcp_all_to_all(output_recv, output_send, self._dcp_process_group)
            return output_recv

        def update_lse():
            lse_send = (
                (softmax_max + torch.log(softmax_sum))
                .squeeze(0)
                .transpose(0, 1)
                .reshape(self.dcp_size, heads_per_rank, tokens)
                .contiguous()
            )
            lse_recv = torch.empty_like(lse_send)
            group = self._dcp_aux_process_group or self._dcp_process_group
            self._dcp_all_to_all(lse_recv, lse_send, group)
            return lse_recv

        if self._dcp_aux_process_group is not None:
            with (
                limit_stream_cores(
                    main_stream,
                    cube_num=12,
                    vector_num=24,
                    enable=graph_phase,
                ),
                limit_stream_cores(
                    overlap_fork.aux_stream,
                    cube_num=12,
                    vector_num=24,
                    enable=graph_phase,
                ),
                overlap_fork.scope(
                    enable=graph_phase,
                    overlap=capture_mode,
                ) as fork,
            ):
                with fork.branch():
                    lse_recv = update_lse()
                output_recv = update_output()
        else:
            output_recv = update_output()
            lse_recv = update_lse()

        result = kernels.merge_partials(
            output_recv.reshape(
                self.dcp_size,
                heads_per_rank * tokens,
                spec.kv_lora_rank,
            ).contiguous(),
            lse_recv.reshape(self.dcp_size, heads_per_rank * tokens).contiguous(),
        ).view(heads_per_rank, tokens, spec.kv_lora_rank)
        return (
            result
            if head_major_output
            else result.transpose(0, 1).contiguous().flatten(1)
        )

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool,
        bs: int,
        save_kv_cache: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        if self._indexer_kernels is not None:
            self._validate_logit_cap(layer.logit_cap)
            if not save_kv_cache:
                raise ValueError("LongCat DSA prefill requires cache writes")
            metadata = self.forward_prefill_metadata
            head_major_output = kwargs.pop("head_major_output", False)
            selection = self._select_indexed(
                q,
                k,
                layer,
                out_cache_loc,
                token_to_kv_pool,
                metadata.cum_extend_seq_lens[1:].to(torch.int32),
                metadata.seq_lens.to(torch.int32),
                metadata.page_table,
                kwargs,
            )
            return self._run_indexed_sparse_attention(
                selection,
                layer,
                token_to_kv_pool,
                head_major_output=head_major_output,
            )
        # The model drives DSA prefill through forward_extend_chunked /
        # forward_sparse_prefill directly.
        raise NotImplementedError(
            "DSA prefill runs through forward_extend_chunked / forward_sparse_prefill"
        )

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool,
        bs: int,
        save_kv_cache: bool = True,
        topk_indices: torch.Tensor | None = None,
        topk_lens: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        self._validate_logit_cap(layer.logit_cap)
        if self._indexer_kernels is not None:
            if q.shape[0] != bs:
                raise ValueError("LongCat DSA decode requires one token per request")
            if save_kv_cache and k is None:
                raise ValueError("LongCat DSA decode cache write requires k")
            if topk_indices is not None or topk_lens is not None:
                raise ValueError(
                    "LongCat DSA expects indexer projections, not global-slot TopK"
                )
            metadata = self.forward_decode_metadata
            start = metadata.num_extends
            head_major_output = kwargs.pop("head_major_output", False)
            selection = self._select_indexed(
                q,
                k,
                layer,
                out_cache_loc,
                token_to_kv_pool,
                torch.arange(1, bs + 1, dtype=torch.int32, device=q.device),
                metadata.seq_lens[start : start + bs],
                metadata.page_table[start : start + bs],
                kwargs,
                save_kv_cache=save_kv_cache,
                context_parallel=True,
                context_parallel_row_start=start,
            )
            return self._run_indexed_sparse_attention(
                selection,
                layer,
                token_to_kv_pool,
                head_major_output=head_major_output,
            )
        if topk_indices is not None:
            return self.forward_sparse_decode(
                q=q,
                k=k,
                v=v,
                layer=layer,
                out_cache_loc=out_cache_loc,
                token_to_kv_pool=token_to_kv_pool,
                bs=bs,
                save_kv_cache=save_kv_cache,
                topk_indices=topk_indices,
                topk_lens=topk_lens,
            )
        metadata = self.forward_decode_metadata
        if metadata is not None and metadata.seq_lens_k is not None:
            num_extends = int(metadata.num_extends or 0)
            self._validate_dense_context(metadata.seq_lens_k[num_extends:], bs)
        return self._dense_backend.forward_decode(
            q=q,
            k=k,
            v=v,
            layer=layer,
            out_cache_loc=out_cache_loc,
            token_to_kv_pool=token_to_kv_pool,
            bs=bs,
            save_kv_cache=save_kv_cache,
            **kwargs,
        )


if current_platform().is_npu:
    register_backend(
        "dsa",
        {AttentionArch.DSA, AttentionArch.MLA},
        AscendDSABackend,
    )
    register_backend("longcat_dsa", {AttentionArch.MLA}, AscendDSABackend)
