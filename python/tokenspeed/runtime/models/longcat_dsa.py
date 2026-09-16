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

"""LongCat-2.0 LSA Indexer and pair-local sparse-attention ownership."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import torch
from tokenspeed_kernel.ops.attention.dsa import dsa_decode_topk, dsa_prefill_topk
from torch import nn
from transformers import PretrainedConfig

from tokenspeed.runtime.distributed import Mapping
from tokenspeed.runtime.distributed.comm_manager import CommManager
from tokenspeed.runtime.execution.breakable_cuda_graph import (
    break_point,
    current_forward_ctx,
    slice_to_real_tokens,
)
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.page_table import (
    build_prefill_kv_workspace_slots,
)
from tokenspeed.runtime.layers.layernorm import FusedRMSNorm, RMSNorm
from tokenspeed.runtime.layers.linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
)
from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig
from tokenspeed.runtime.layers.quantization.utils import block_dequant
from tokenspeed.runtime.layers.rotary_embedding import get_rope
from tokenspeed.runtime.model_loader.weight_utils import default_weight_loader
from tokenspeed.runtime.models.deepseek_v3 import DeepseekV3AttentionMLA
from tokenspeed.runtime.utils import add_prefix
from tokenspeed.runtime.utils.env import global_server_args_dict

_INDEXER_PREFILL_MAX_LOGITS_MB_ARG = "deepseek_v4_indexer_prefill_max_logits_mb"


@dataclass
class LongCatDSAIndexerOutput:
    query: torch.Tensor
    key: torch.Tensor
    weights: torch.Tensor


@dataclass
class LongCatDSAPrefillSelection:
    workspace_indices: torch.Tensor
    topk_lens: torch.Tensor
    page_table: torch.Tensor
    seq_lens: torch.Tensor
    max_seq_len: int
    kv_workspace_slots: torch.Tensor


@dataclass
class LongCatDSADecodeSelection:
    topk_indices: torch.Tensor
    topk_lens: torch.Tensor


@dataclass(frozen=True)
class LongCatDSADecodeWindow:
    start: int
    end: int
    num_tokens: int
    num_reqs: int
    q_len_per_req: int


@dataclass
class LongCatDSASelection:
    """Forward-local selection produced by one owner for its paired consumer."""

    owner_layer_id: int
    prefill: LongCatDSAPrefillSelection | None = None
    decode: LongCatDSADecodeSelection | None = None

    def require_consumer(self, consumer_layer_id: int) -> None:
        expected_consumer = self.owner_layer_id + 1
        if consumer_layer_id != expected_consumer:
            raise RuntimeError(
                f"LongCat DSA selection from owner layer {self.owner_layer_id} "
                f"requires consumer layer {expected_consumer}, got consumer "
                f"layer {consumer_layer_id}."
            )


def _slice_indexer_rows(
    hidden_states: torch.Tensor,
    *,
    expected_rows: int,
) -> torch.Tensor:
    actual_rows = int(hidden_states.shape[0])
    if actual_rows < expected_rows:
        raise RuntimeError(
            f"LongCat Indexer expected {expected_rows} rows, but input only has "
            f"{actual_rows}."
        )
    return (
        hidden_states if actual_rows == expected_rows else hidden_states[:expected_rows]
    )


def _longcat_rope_scaling(
    rope_scaling: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not rope_scaling or "factor" not in rope_scaling:
        return None
    normalized = dict(rope_scaling)
    normalized["rope_type"] = "deepseek_yarn"
    return normalized


class LongCatDSAIndexer(nn.Module):
    """LongCat Indexer with RMSNorm and interleaved rotary layout."""

    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        q_lora_rank: int,
        qk_rope_head_dim: int,
        rope_theta: float,
        rope_scaling: dict[str, Any] | None,
        max_position_embeddings: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.index_topk = config.index_topk
        self.index_n_heads = config.index_n_heads
        self.index_head_dim = config.index_head_dim
        self.rope_head_dim = int(qk_rope_head_dim)
        self.weights_softmax_scale = self.index_head_dim**-0.5 * (
            self.index_n_heads**-0.5
        )
        if not 0 < self.rope_head_dim <= self.index_head_dim:
            raise ValueError(
                "LongCat Indexer requires 0 < qk_rope_head_dim <= "
                f"index_head_dim, got {self.rope_head_dim} and "
                f"{self.index_head_dim}."
            )

        self.wq_b = ReplicatedLinear(
            q_lora_rank,
            self.index_n_heads * self.index_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("wq_b", prefix),
        )
        self.wk_weights_proj = MergedColumnParallelLinear(
            hidden_size,
            [self.index_head_dim, self.index_n_heads],
            bias=False,
            quant_config=None,
            prefix=add_prefix("wk_weights_proj", prefix),
        )
        self._packed_projection_loaded = False
        self.k_norm = RMSNorm(self.index_head_dim, eps=1e-6)
        self.rotary_emb = get_rope(
            self.rope_head_dim,
            rotary_dim=self.rope_head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=_longcat_rope_scaling(rope_scaling),
            is_neox_style=False,
        )

    def set_packed_projection_loaded(self) -> None:
        self._packed_projection_loaded = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_lora: torch.Tensor,
        positions: torch.Tensor,
    ) -> LongCatDSAIndexerOutput:
        if not self._packed_projection_loaded:
            raise RuntimeError("LongCat Indexer packed projection was not loaded")
        index_q = self.wq_b(q_lora)[0].view(-1, self.index_n_heads, self.index_head_dim)
        key_weights = self.wk_weights_proj(hidden_states)[0]
        index_k, weights = key_weights.split(
            [self.index_head_dim, self.index_n_heads], dim=-1
        )
        index_k = self.k_norm(index_k)
        q_rope, k_rope = self.rotary_emb(
            positions,
            index_q[..., : self.rope_head_dim],
            index_k[:, None, : self.rope_head_dim],
        )
        index_q[..., : self.rope_head_dim] = q_rope
        index_k[:, : self.rope_head_dim] = k_rope.squeeze(1)
        return LongCatDSAIndexerOutput(index_q, index_k, weights)


class LongCatDSAIndexerWeightLoaderMixin:
    """Load LongCat's separate Index-K and score projections into one GEMM."""

    @staticmethod
    def _record_projection_shard(
        *,
        module_name: str,
        shard_id: int,
        loaded_shards: dict[str, set[int]],
        modules: dict[str, nn.Module],
    ) -> None:
        shards = loaded_shards.setdefault(module_name, set())
        shards.add(int(shard_id))
        if shards == {0, 1}:
            module = modules.get(module_name)
            if module is not None and hasattr(module, "set_packed_projection_loaded"):
                module.set_packed_projection_loaded()

    def _load_projection_shard(
        self,
        *,
        module_name: str,
        shard_id: int,
        loaded_weight: torch.Tensor,
        params: dict[str, torch.Tensor],
        modules: dict[str, nn.Module],
        loaded_shards: dict[str, set[int]],
    ) -> bool:
        param = params.get(f"{module_name}.wk_weights_proj.weight")
        if param is None:
            return False
        loader = getattr(param, "weight_loader", default_weight_loader)
        loader(param, loaded_weight, shard_id)
        self._record_projection_shard(
            module_name=module_name,
            shard_id=shard_id,
            loaded_shards=loaded_shards,
            modules=modules,
        )
        return True

    def _flush_fp8_index_k(
        self,
        *,
        module_name: str,
        pending_fp8: dict[str, dict[str, torch.Tensor]],
        params: dict[str, torch.Tensor],
        modules: dict[str, nn.Module],
        loaded_shards: dict[str, set[int]],
        weight_block_size: list[int] | tuple[int, ...] | None,
    ) -> None:
        entry = pending_fp8.get(module_name)
        if (
            not entry
            or "weight" not in entry
            or "scale" not in entry
            or weight_block_size is None
        ):
            return
        weight = block_dequant(
            entry["weight"], entry["scale"], list(weight_block_size)
        ).to(torch.bfloat16)
        if self._load_projection_shard(
            module_name=module_name,
            shard_id=0,
            loaded_weight=weight,
            params=params,
            modules=modules,
            loaded_shards=loaded_shards,
        ):
            del pending_fp8[module_name]

    def try_load_indexer_projection(
        self,
        *,
        name: str,
        loaded_weight: torch.Tensor,
        params: dict[str, torch.Tensor],
        modules: dict[str, nn.Module],
        pending_fp8: dict[str, dict[str, torch.Tensor]],
        loaded_shards: dict[str, set[int]],
        weight_block_size: list[int] | tuple[int, ...] | None,
    ) -> bool:
        if name.endswith(".indexer.wk_weights_proj.weight"):
            module_name = name.rsplit(".wk_weights_proj.weight", 1)[0]
            param = params.get(f"{module_name}.wk_weights_proj.weight")
            if param is None:
                raise RuntimeError(
                    f"LongCat packed Indexer projection is missing: {module_name}"
                )
            getattr(param, "weight_loader", default_weight_loader)(param, loaded_weight)
            for shard_id in (0, 1):
                self._record_projection_shard(
                    module_name=module_name,
                    shard_id=shard_id,
                    loaded_shards=loaded_shards,
                    modules=modules,
                )
            return True
        if name.endswith(".indexer.weights_proj.weight"):
            module_name = name.rsplit(".weights_proj.weight", 1)[0]
            if not self._load_projection_shard(
                module_name=module_name,
                shard_id=1,
                loaded_weight=loaded_weight,
                params=params,
                modules=modules,
                loaded_shards=loaded_shards,
            ):
                raise RuntimeError(
                    f"LongCat packed Indexer projection is missing: {module_name}"
                )
            return True
        if ".indexer.wk." not in name:
            return False

        module_name = name.rsplit(".wk.", 1)[0]
        fp8_dtypes = tuple(
            dtype
            for dtype in (
                getattr(torch, "float8_e4m3fn", None),
                getattr(torch, "float8_e4m3fnuz", None),
            )
            if dtype is not None
        )
        if name.endswith(".weight") and loaded_weight.dtype in fp8_dtypes:
            pending_fp8.setdefault(module_name, {})["weight"] = loaded_weight
            self._flush_fp8_index_k(
                module_name=module_name,
                pending_fp8=pending_fp8,
                params=params,
                modules=modules,
                loaded_shards=loaded_shards,
                weight_block_size=weight_block_size,
            )
            return True
        if name.endswith(".weight"):
            if not self._load_projection_shard(
                module_name=module_name,
                shard_id=0,
                loaded_weight=loaded_weight,
                params=params,
                modules=modules,
                loaded_shards=loaded_shards,
            ):
                raise RuntimeError(
                    f"LongCat packed Indexer projection is missing: {module_name}"
                )
            return True
        if "weight_scale_inv" in name:
            pending_fp8.setdefault(module_name, {})["scale"] = loaded_weight
            self._flush_fp8_index_k(
                module_name=module_name,
                pending_fp8=pending_fp8,
                params=params,
                modules=modules,
                loaded_shards=loaded_shards,
                weight_block_size=weight_block_size,
            )
            return True
        return False

    @staticmethod
    def validate_indexer_projections(
        *,
        modules: dict[str, nn.Module],
        pending_fp8: dict[str, dict[str, torch.Tensor]],
        loaded_shards: dict[str, set[int]],
    ) -> None:
        expected = {
            name
            for name, module in modules.items()
            if name.endswith(".indexer")
            and hasattr(module, "set_packed_projection_loaded")
        }
        incomplete = {
            name for name in expected if loaded_shards.get(name, set()) != {0, 1}
        }
        if pending_fp8 or incomplete:
            raise RuntimeError(
                "LongCat Indexer packed projections are incomplete: "
                f"pending_fp8={sorted(pending_fp8)}, missing={sorted(incomplete)}"
            )


class LongCatDSAAttention(DeepseekV3AttentionMLA):
    """One physical LongCat LSA attention, either selection owner or consumer."""

    _MLA_KERNEL_BACKENDS = ("trtllm_mla", "tokenspeed_mla", "dsa")
    _RAGGED_PREFILL_BACKENDS = ("trtllm_mla", "tokenspeed_mla", "dsa")
    rope_is_neox_style = False

    def __init__(
        self,
        config: PretrainedConfig,
        mapping: Mapping,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        rope_theta: float = 10000,
        rope_scaling: dict[str, Any] | None = None,
        max_position_embeddings: int = 8192,
        quant_config: QuantizationConfig | None = None,
        layer_id: int | None = None,
        prefix: str = "",
        reduce_attn_results: bool = True,
        alt_stream: torch.cuda.Stream | None = None,
        *,
        computes_selection: bool,
        selection_owner_layer_id: int,
        initial_tokens: int,
        local_tokens: int,
        lora_norm_eps: float,
    ) -> None:
        rope_scaling = _longcat_rope_scaling(rope_scaling)
        super().__init__(
            config=config,
            mapping=mapping,
            hidden_size=hidden_size,
            num_heads=num_heads,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            layer_id=layer_id,
            prefix=prefix,
            reduce_attn_results=reduce_attn_results,
            alt_stream=alt_stream,
        )
        if q_lora_rank is None:
            raise ValueError("LongCat LSA requires q_lora_rank")
        expected_layer_id = (
            selection_owner_layer_id
            if computes_selection
            else selection_owner_layer_id + 1
        )
        if layer_id != expected_layer_id:
            role = "owner" if computes_selection else "consumer"
            raise ValueError(
                f"LongCat DSA {role} layer id must be {expected_layer_id}, "
                f"got {layer_id}"
            )
        self.q_a_layernorm = RMSNorm(q_lora_rank, eps=lora_norm_eps)
        self.kv_a_layernorm = RMSNorm(kv_lora_rank, eps=lora_norm_eps)
        self.fused_qk_layernorm = FusedRMSNorm(self.q_a_layernorm, self.kv_a_layernorm)
        self.index_topk = int(config.index_topk)
        self.initial_tokens = int(initial_tokens)
        self.local_tokens = int(local_tokens)
        if self.initial_tokens + self.local_tokens > self.index_topk:
            raise ValueError(
                "LongCat initial/local candidates must fit inside index_topk"
            )
        self.computes_selection = bool(computes_selection)
        self.selection_owner_layer_id = int(selection_owner_layer_id)
        self._selection = (
            LongCatDSASelection(owner_layer_id=self.selection_owner_layer_id)
            if self.computes_selection
            else None
        )
        self.indexer = (
            LongCatDSAIndexer(
                config=config,
                hidden_size=hidden_size,
                q_lora_rank=q_lora_rank,
                qk_rope_head_dim=qk_rope_head_dim,
                rope_theta=rope_theta,
                rope_scaling=rope_scaling,
                max_position_embeddings=max_position_embeddings,
                quant_config=quant_config,
                prefix=add_prefix("indexer", prefix),
            )
            if self.computes_selection
            else None
        )
        self._decode_topk_indices_buffer: torch.Tensor | None = None
        self._decode_topk_lens_buffer: torch.Tensor | None = None

    def _require_indexer(self) -> LongCatDSAIndexer:
        if self.indexer is None:
            raise RuntimeError("LongCat DSA consumer does not own an Indexer")
        return self.indexer

    def _get_decode_topk_workspace(
        self,
        rows: int,
        cols: int,
        device: torch.device,
        *,
        fill_value: int | None,
    ) -> torch.Tensor:
        buffer = self._decode_topk_indices_buffer
        if (
            buffer is None
            or buffer.device != device
            or buffer.shape[0] < rows
            or buffer.shape[1] != cols
        ):
            # A captured CUDA graph may still reference the old buffer. Keep
            # it alive when the workspace grows or moves to another device.
            if buffer is not None:
                self._retire_decode_workspace(buffer)
            buffer = torch.empty((rows, cols), dtype=torch.int32, device=device)
            self._decode_topk_indices_buffer = buffer
        workspace = buffer[:rows]
        if fill_value is not None:
            workspace.fill_(fill_value)
        return workspace

    def _get_decode_topk_lens_workspace(
        self,
        rows: int,
        device: torch.device,
        *,
        fill: bool,
    ) -> torch.Tensor:
        buffer = self._decode_topk_lens_buffer
        if buffer is None or buffer.device != device or buffer.numel() < rows:
            if buffer is not None:
                self._retire_decode_workspace(buffer)
            buffer = torch.empty(rows, dtype=torch.int32, device=device)
            self._decode_topk_lens_buffer = buffer
        workspace = buffer[:rows]
        if fill:
            workspace.zero_()
        return workspace

    def _retire_decode_workspace(self, buffer: torch.Tensor) -> None:
        retired = getattr(self, "_retired_decode_workspaces", None)
        if retired is None:
            retired = []
            self._retired_decode_workspaces = retired
        retired.append(buffer)

    @staticmethod
    def _resolve_decode_req_count(ctx: ForwardContext, metadata: Any) -> int:
        num_extends = int(getattr(metadata, "num_extends", 0) or 0)
        limits = [max(0, int(ctx.bs) - int(ctx.num_extends))]
        seq_lens = getattr(metadata, "seq_lens_k", None)
        if seq_lens is not None:
            limits.append(max(0, int(seq_lens.shape[0]) - num_extends))
        block_tables = getattr(metadata, "page_table", None)
        if block_tables is not None:
            limits.append(max(0, int(block_tables.shape[0]) - num_extends))
        return min(limits)

    @classmethod
    def _resolve_decode_window(
        cls,
        ctx: ForwardContext,
        metadata: Any,
        *,
        total_tokens: int,
    ) -> LongCatDSADecodeWindow:
        num_reqs = cls._resolve_decode_req_count(ctx, metadata)
        spec_width = int(getattr(ctx.attn_backend, "spec_num_tokens", 1) or 1)
        num_tokens = min(total_tokens, num_reqs * spec_width) if num_reqs else 0
        q_len_per_req = 1
        if num_reqs and num_tokens:
            q_len_per_req, remainder = divmod(num_tokens, num_reqs)
            if remainder or q_len_per_req <= 0:
                q_len_per_req = 1
        start = total_tokens - num_tokens
        return LongCatDSADecodeWindow(
            start=start,
            end=start + num_tokens,
            num_tokens=num_tokens,
            num_reqs=num_reqs,
            q_len_per_req=q_len_per_req,
        )

    @staticmethod
    def check_decode_width(q_len_per_req: int) -> None:
        if not 1 <= q_len_per_req <= 8:
            raise NotImplementedError(
                "LongCat DSA sparse decode supports 1-8 query tokens per "
                f"request, got {q_len_per_req}."
            )

    def _compute_decode_selection(
        self,
        indexer_output: LongCatDSAIndexerOutput,
        ctx: ForwardContext,
    ) -> LongCatDSADecodeSelection | None:
        metadata = getattr(ctx.attn_backend, "forward_decode_metadata", None)
        if metadata is None or metadata.page_table is None:
            return None
        num_tokens = int(indexer_output.query.shape[0])
        window = self._resolve_decode_window(ctx, metadata, total_tokens=num_tokens)
        if window.num_reqs <= 0 or window.num_tokens == 0:
            return None
        self.check_decode_width(window.q_len_per_req)
        num_extends = int(metadata.num_extends or 0)
        seq_lens = metadata.seq_lens_k[num_extends : num_extends + window.num_reqs]
        page_table = metadata.page_table[num_extends : num_extends + window.num_reqs]
        q = indexer_output.query[window.start : window.end]
        weights = indexer_output.weights[window.start : window.end]
        index_k_cache = ctx.token_to_kv_pool.get_index_k_buffer(
            self.selection_owner_layer_id
        )
        writes_all = window.start == 0 and window.num_tokens == num_tokens
        indices = self._get_decode_topk_workspace(
            num_tokens,
            self.index_topk,
            q.device,
            fill_value=None if writes_all else -1,
        )
        index_slice = indices[window.start : window.end]
        lengths = self._get_decode_topk_lens_workspace(
            num_tokens, q.device, fill=not writes_all
        )
        length_slice = lengths[window.start : window.end]
        seq_lens_2d = (
            metadata._dsa_seq_lens_2d[ctx.num_extends * window.q_len_per_req :]
            if window.q_len_per_req > 1
            else seq_lens.unsqueeze(1)
        )
        dsa_decode_topk(
            q,
            weights,
            seq_lens,
            page_table,
            page_size=ctx.token_to_kv_pool.arena.kv_page_size,
            topk=self.index_topk,
            softmax_scale=self._require_indexer().weights_softmax_scale,
            q_len_per_req=window.q_len_per_req,
            index_k_cache=index_k_cache,
            seq_lens_2d=seq_lens_2d,
            plan=metadata._dsa_plan,
            initial_tokens=self.initial_tokens,
            local_tokens=self.local_tokens,
            out=index_slice,
            lens_out=length_slice,
        )
        return LongCatDSADecodeSelection(indices, lengths)

    def _compute_prefill_selection(
        self,
        indexer_output: LongCatDSAIndexerOutput,
        ctx: ForwardContext,
        num_prefill_tokens: int,
    ) -> LongCatDSAPrefillSelection | None:
        chunk_meta = ctx.attn_backend.chunked_prefill_metadata
        prefix_lens = getattr(chunk_meta, "extend_prefix_lens_cpu", None)
        if prefix_lens is None:
            prefix_lens = chunk_meta.extend_prefix_lens
        prefix_lens_cpu = torch.as_tensor(
            prefix_lens[: ctx.num_extends], dtype=torch.int64, device="cpu"
        )
        extend_lens_cpu = torch.as_tensor(
            chunk_meta.extend_seq_lens_cpu[: ctx.num_extends],
            dtype=torch.int64,
            device="cpu",
        )
        if extend_lens_cpu.numel() == 0:
            return None
        metadata_tokens = int(extend_lens_cpu.sum().item())
        if metadata_tokens != num_prefill_tokens:
            raise RuntimeError(
                "LongCat DSA prefill token count mismatch: "
                f"metadata={metadata_tokens}, tokens={num_prefill_tokens}"
            )
        if chunk_meta.page_table is None:
            raise RuntimeError("LongCat DSA prefill requires a page table")

        device = indexer_output.query.device
        seq_lens_cpu = prefix_lens_cpu + extend_lens_cpu
        seq_lens = seq_lens_cpu.to(device=device, dtype=torch.int32)
        max_seq_len = int(seq_lens_cpu.max().item())
        page_size = ctx.token_to_kv_pool.arena.kv_page_size
        max_pages = (max_seq_len + page_size - 1) // page_size
        page_table = chunk_meta.page_table[:, :max_pages].to(
            device=device, dtype=torch.int32
        )
        kv_workspace_slots = build_prefill_kv_workspace_slots(
            page_table=page_table,
            seq_lens=seq_lens,
            max_seq_len=max_seq_len,
            page_size=page_size,
            device=device,
            num_tokens=int(seq_lens_cpu.sum().item()),
        )
        req_ids = torch.arange(seq_lens_cpu.numel(), dtype=torch.int64)
        token_req = torch.repeat_interleave(
            req_ids, extend_lens_cpu, output_size=num_prefill_tokens
        )
        extend_cu = torch.zeros(extend_lens_cpu.numel() + 1, dtype=torch.int64)
        torch.cumsum(extend_lens_cpu, dim=0, out=extend_cu[1:])
        token_offsets = torch.arange(num_prefill_tokens, dtype=torch.int64)
        token_offsets -= extend_cu.index_select(0, token_req)
        candidate_lens = prefix_lens_cpu.index_select(0, token_req) + token_offsets + 1
        seq_cu = torch.zeros(seq_lens_cpu.numel() + 1, dtype=torch.int64)
        torch.cumsum(seq_lens_cpu, dim=0, out=seq_cu[1:])
        row_starts = seq_cu.index_select(0, token_req)
        row_ends = row_starts + candidate_lens
        max_logits_mb = int(global_server_args_dict[_INDEXER_PREFILL_MAX_LOGITS_MB_ARG])
        workspace_indices, topk_lens = dsa_prefill_topk(
            indexer_output.query[:num_prefill_tokens].contiguous(),
            indexer_output.weights[:num_prefill_tokens],
            kv_workspace_slots,
            row_starts.to(device=device, dtype=torch.int32),
            row_ends.to(device=device, dtype=torch.int32),
            topk=self.index_topk,
            softmax_scale=self._require_indexer().weights_softmax_scale,
            index_k_cache=ctx.token_to_kv_pool.get_index_k_buffer(
                self.selection_owner_layer_id
            ),
            page_size=page_size,
            max_logits_bytes=max(1, max_logits_mb) * 1024 * 1024,
            candidate_lens_cpu=candidate_lens,
            initial_tokens=self.initial_tokens,
            local_tokens=self.local_tokens,
        )
        return LongCatDSAPrefillSelection(
            workspace_indices=workspace_indices,
            topk_lens=topk_lens,
            page_table=page_table,
            seq_lens=seq_lens,
            max_seq_len=max_seq_len,
            kv_workspace_slots=kv_workspace_slots,
        )

    def forward(self, *args, **kwargs):
        raise RuntimeError("LongCat DSA layers require forward_with_selection()")

    def forward_with_selection(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: ForwardContext,
        comm_manager: CommManager,
        *,
        selection: LongCatDSASelection | None,
        block_scale: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, LongCatDSASelection]:
        hidden_states = self._forward_with_selection_output(
            positions,
            hidden_states,
            ctx,
            comm_manager,
            block_scale,
            selection,
        )
        result = self._selection if self.computes_selection else selection
        if result is None:
            raise RuntimeError("LongCat DSA selection was not produced")
        return hidden_states, result

    @break_point
    def _forward_with_selection_output(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: ForwardContext,
        comm_manager: CommManager,
        block_scale: torch.Tensor | None,
        selection: LongCatDSASelection | None,
    ) -> torch.Tensor:
        if self.computes_selection:
            if selection is not None:
                raise RuntimeError("LongCat DSA owner requires a fresh selection")
            self._selection.prefill = None
            self._selection.decode = None
        else:
            if selection is None:
                raise RuntimeError("LongCat DSA consumer requires owner selection")
            selection.require_consumer(self.attn_mqa.layer_id)
        if hidden_states.shape[0] == 0:
            return hidden_states

        qkv = self.fused_qkv_a_proj_with_mqa(hidden_states, block_scale, torch.bfloat16)
        logical_width = self.q_lora_rank + self.kv_lora_rank + self.qk_rope_head_dim
        qkv = qkv[..., :logical_width]
        qkv = comm_manager.pre_attn_comm(qkv, ctx)
        metadata = getattr(ctx.attn_backend, "forward_metadata", None)
        token_to_req = getattr(metadata, "token_to_req_indices", None)
        if current_forward_ctx() is not None and token_to_req is not None:
            positions, qkv = slice_to_real_tokens(token_to_req.numel(), positions, qkv)
        q_a, latent_cache = qkv.split(
            [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim], dim=-1
        )
        q_norm = torch.empty_like(q_a)
        if q_a.shape[0] > 0:
            self.fused_qk_layernorm(
                input_q_a=q_a,
                input_kv_a=latent_cache[..., : self.kv_lora_rank],
                output_q_a=q_norm,
            )

        window = self._resolve_decode_window(
            ctx,
            getattr(ctx.attn_backend, "forward_decode_metadata", None),
            total_tokens=int(q_norm.shape[0]),
        )
        num_prefill_tokens = window.start
        write_locations = []
        if ctx.num_extends > 0:
            write_locations.append(
                ctx.attn_backend.write_locations(self.attn_mqa, ForwardMode.EXTEND)
            )
        if window.num_tokens > 0:
            write_locations.append(
                ctx.attn_backend.write_locations(self.attn_mqa, ForwardMode.DECODE)
            )
        out_cache_loc = (
            write_locations[0]
            if len(write_locations) == 1
            else torch.cat(write_locations)
        )
        if self.computes_selection:
            indexer_hidden = comm_manager.pre_attn_comm(hidden_states, ctx)
            indexer_hidden = _slice_indexer_rows(
                indexer_hidden, expected_rows=int(q_norm.shape[0])
            )
            indexer_output = self._require_indexer()(indexer_hidden, q_norm, positions)
            ctx.token_to_kv_pool.set_index_k_buffer(
                self.selection_owner_layer_id,
                out_cache_loc,
                indexer_output.key,
            )
            if ctx.num_extends > 0:
                self._selection.prefill = self._compute_prefill_selection(
                    indexer_output, ctx, num_prefill_tokens
                )
            if ctx.num_extends < ctx.bs:
                self._selection.decode = self._compute_decode_selection(
                    indexer_output, ctx
                )
            active_selection = self._selection
        else:
            active_selection = selection

        q = self.q_b_proj(q_norm)[0]
        output = torch.empty(
            q.shape[0],
            self.num_local_heads * self.v_head_dim,
            dtype=q.dtype,
            device=q.device,
        )
        if ctx.num_extends > 0:
            if active_selection.prefill is None:
                raise RuntimeError("LongCat DSA prefill selection is missing")
            prefill_ctx = replace(
                ctx,
                bs=ctx.num_extends,
                input_num_tokens=num_prefill_tokens,
                forward_mode=ForwardMode.EXTEND,
            )
            self._forward_sparse_prefill(
                positions[:num_prefill_tokens],
                q[:num_prefill_tokens],
                latent_cache[:num_prefill_tokens],
                prefill_ctx,
                out_cache_loc[:num_prefill_tokens],
                output[:num_prefill_tokens],
                active_selection.prefill,
            )
        if window.num_tokens > 0:
            if active_selection.decode is None:
                raise RuntimeError("LongCat DSA decode selection is missing")
            decode_ctx = replace(
                ctx,
                bs=window.num_reqs,
                num_extends=0,
                input_num_tokens=window.num_tokens,
                forward_mode=ForwardMode.DECODE,
            )
            topk_indices = active_selection.decode.topk_indices[
                window.start : window.end
            ]
            topk_lens = active_selection.decode.topk_lens[window.start : window.end]
            self._forward_sparse_decode(
                positions[window.start : window.end],
                q[window.start : window.end],
                latent_cache[window.start : window.end],
                decode_ctx,
                out_cache_loc[window.start : window.end],
                output[window.start : window.end],
                topk_indices,
                topk_lens,
            )
        if ctx.draft_narrowing is not None:
            output = output.index_select(0, ctx.gather_ids)
        return self.o_proj(output)[0]

    def _forward_sparse_prefill(
        self,
        positions: torch.Tensor,
        q: torch.Tensor,
        latent_cache: torch.Tensor,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
        output: torch.Tensor,
        selection: LongCatDSAPrefillSelection,
    ) -> None:
        query, _ = self.forward_absorb_qkv_proj(
            q, latent_cache, positions, ctx, out_cache_loc
        )
        attention = ctx.attn_backend.forward_sparse_prefill(
            q=query,
            layer=self.attn_mqa,
            token_to_kv_pool=ctx.token_to_kv_pool,
            page_table=selection.page_table,
            seq_lens=selection.seq_lens,
            workspace_indices=selection.workspace_indices,
            topk_lens=selection.topk_lens,
            kv_workspace_slots=selection.kv_workspace_slots,
            max_seq_len=selection.max_seq_len,
        )
        attention = attention.view(-1, self.num_local_heads, self.kv_lora_rank)
        output_view = output.view(-1, self.num_local_heads, self.v_head_dim)
        torch.bmm(
            attention.transpose(0, 1),
            self.w_vc,
            out=output_view.transpose(0, 1),
        )

    def _forward_sparse_decode(
        self,
        positions: torch.Tensor,
        q: torch.Tensor,
        latent_cache: torch.Tensor,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
        output: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_lens: torch.Tensor,
    ) -> None:
        query, key = self.forward_absorb_qkv_proj(
            q, latent_cache, positions, ctx, out_cache_loc
        )
        need_save_kv = False
        if self.attention_backend not in self._MLA_KERNEL_BACKENDS:
            need_save_kv = not self.use_fused_set_kv_buffer
        attention = self.attn_mqa(
            query,
            key,
            key[..., : self.kv_lora_rank] if key is not None else None,
            ctx,
            save_kv_cache=need_save_kv,
            topk_indices=topk_indices,
            topk_lens=topk_lens,
        )
        attention = attention.view(-1, self.num_local_heads, self.kv_lora_rank)
        output_view = output.view(-1, self.num_local_heads, self.v_head_dim)
        torch.bmm(
            attention.transpose(0, 1),
            self.w_vc,
            out=output_view.transpose(0, 1),
        )


__all__ = [
    "LongCatDSAAttention",
    "LongCatDSAIndexer",
    "LongCatDSAIndexerWeightLoaderMixin",
    "LongCatDSASelection",
]
