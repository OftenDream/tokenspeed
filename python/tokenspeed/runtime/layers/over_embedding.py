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
"""LongCat branch-local over-embedding runtime layer."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from dataclasses import replace
from typing import Any

import torch
from tokenspeed_kernel.ops.over_embedding import (
    OverEmbeddingSpec,
    TableFragmentSpec,
    append_packed_lookup_,
    project_add_word_,
)
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter

from tokenspeed.runtime.distributed.comm_ops import all_reduce
from tokenspeed.runtime.layers.vocab_parallel_embedding import VocabParallelEmbedding


def resolve_longcat_oe_hyperparameters(config) -> tuple[int, int]:
    """Return ``(max_ngram_order, hashes_per_order)`` from HF field aliases."""
    max_ngram_order = getattr(config, "emb_neighbor_num", None)
    if max_ngram_order is None:
        max_ngram_order = getattr(config, "oe_neighbor_num", None)
    hashes_per_order = getattr(config, "emb_split_num", None)
    if hashes_per_order is None:
        hashes_per_order = getattr(config, "oe_split_num", None)
    if max_ngram_order is None or hashes_per_order is None:
        raise ValueError(
            "LongCat OE requires neighbor_num (maximum n-gram order) and "
            "split_num (hash branches per order)"
        )
    return int(max_ngram_order), int(hashes_per_order)


def resolve_longcat_oe_spec(
    *,
    vocab_size: int,
    hidden_size: int,
    max_ngram_order: int,
    hashes_per_order: int,
    modulus0: int,
    tp_size: int,
    tp_rank: int,
) -> OverEmbeddingSpec:
    """Derive rank-local table feature slices from checkpoint geometry.

    Branch widths follow the checkpoint's floor division. Complete branches
    are assigned first; the remaining branch features are partitioned over
    ranks without duplication. Two-whole-branch ownership stays contiguous
    to preserve Pro's layout; other full rounds are interleaved, as in Lite.
    Kernel selection still checks executable fragment shapes independently.
    """
    for name, value, minimum in (
        ("vocab_size", vocab_size, 1),
        ("hidden_size", hidden_size, 1),
        ("max_ngram_order", max_ngram_order, 2),
        ("hashes_per_order", hashes_per_order, 1),
        ("modulus0", modulus0, 1),
        ("tp_size", tp_size, 1),
        ("tp_rank", tp_rank, 0),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    if tp_rank >= tp_size:
        raise ValueError(f"tp_rank must be less than tp_size, got {tp_rank}/{tp_size}")
    branch_count = (max_ngram_order - 1) * hashes_per_order
    branch_width = hidden_size // branch_count
    if branch_width == 0 or branch_count * branch_width < tp_size:
        raise ValueError("OE geometry must provide at least one feature per TP rank")
    full_rounds, tail_branches = divmod(branch_count, tp_size)
    fragments = []

    def append_fragment(branch: int, begin: int, width: int) -> None:
        fragments.append(
            TableFragmentSpec(
                branch_id=branch,
                ngram_order=branch // hashes_per_order + 2,
                modulus=modulus0 + 2 * branch,
                feature_begin=begin,
                feature_width=width,
            )
        )

    for index in range(full_rounds):
        branch = (
            tp_rank * full_rounds + index
            if full_rounds == 2 and tail_branches == 0
            else index * tp_size + tp_rank
        )
        append_fragment(branch, 0, branch_width)
    tail_width = tail_branches * branch_width
    begin = tail_width * tp_rank // tp_size
    end = tail_width * (tp_rank + 1) // tp_size
    while begin < end:
        tail_branch, feature_begin = divmod(begin, branch_width)
        width = min(end - begin, branch_width - feature_begin)
        append_fragment(full_rounds * tp_size + tail_branch, feature_begin, width)
        begin += width
    return OverEmbeddingSpec(
        profile="longcat-configured",
        tp_size=tp_size,
        rank=tp_rank,
        vocab_size=vocab_size,
        branch_count=branch_count,
        branch_width=branch_width,
        hidden_size=hidden_size,
        max_ngram_order=max_ngram_order,
        fragments=tuple(fragments),
    )


def _branch_index(weight_name: str, *, projection: bool) -> int | None:
    patterns = (
        (
            r"(?:^|\.)oe_embed_proj(\d+)\.weight$",
            r"(?:^|\.)ngram_embeddings\.post_projs\.(\d+)\.weight$",
        )
        if projection
        else (
            r"(?:^|\.)oe_embed_tokens(\d+)\.weight$",
            r"(?:^|\.)ngram_embeddings\.embedders\.(\d+)\.weight$",
        )
    )
    for pattern in patterns:
        match = re.search(pattern, weight_name)
        if match is not None:
            return int(match.group(1))
    return None


def _add_megatron_special_token_oe(
    merged: torch.Tensor,
    projected_oe: torch.Tensor,
    special_mask: torch.Tensor,
) -> torch.Tensor:
    """Restore learned OE rows at special positions without normalization.

    Args:
        merged: ``[tokens, hidden]`` merge result; special positions must still
            hold the unscaled word embedding (or its TP-local contribution).
        projected_oe: OE projection, either ``[tokens, hidden]`` from Host lookup
            or ``[1, hidden]`` from Device TP-local learned row-zero fragments.
        special_mask: Device-local bool ``[tokens]`` marking excluded tokens.

    Returns:
        A tensor with unscaled word-plus-OE at special positions and unchanged
        ordinary rows. The caller retains ownership of any TP reduction.

    This compatibility boundary can be removed only once both physical leaves
    implement the same learned-row semantics themselves.
    """
    return torch.where(special_mask.unsqueeze(-1), merged + projected_oe, merged)


class LongCatOverEmbedding(nn.Module):
    """Word embedding plus TP-local OE branches and one final TP reduction."""

    def __init__(
        self,
        *,
        num_embeddings: int,
        embedding_dim: int,
        over_embedding_m: int,
        hashes_per_order: int,
        max_ngram_order: int,
        tp_rank: int,
        tp_size: int,
        tp_group: tuple[int, ...] | None,
        params_dtype: torch.dtype | None = None,
        ignored_token_ids: tuple[int, ...] = (),
        eos_token_id: int | None = None,
        segment_ignored_tokens: bool = False,
        fix_normalize_factor: bool = False,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.tp_group = tp_group
        dtype = params_dtype or torch.get_default_dtype()
        self.spec = resolve_longcat_oe_spec(
            vocab_size=num_embeddings,
            hidden_size=embedding_dim,
            max_ngram_order=max_ngram_order,
            hashes_per_order=hashes_per_order,
            modulus0=over_embedding_m + 1,
            tp_size=tp_size,
            tp_rank=tp_rank,
        )
        self.spec = replace(
            self.spec,
            ignored_token_ids=tuple(dict.fromkeys(ignored_token_ids)),
            eos_token_id=eos_token_id,
            segment_ignored_tokens=segment_ignored_tokens,
        )
        self.normalize_scale = (
            math.sqrt(self.spec.scale)
            if fix_normalize_factor
            else float(self.spec.scale)
        )

        self.word_embedding = VocabParallelEmbedding(
            num_embeddings,
            embedding_dim,
            tp_rank=tp_rank,
            tp_size=tp_size,
            tp_group=tp_group,
            params_dtype=params_dtype,
        )
        self.oe_tables = nn.ParameterList(
            [
                Parameter(
                    torch.empty(
                        (fragment.modulus, fragment.feature_width),
                        dtype=dtype,
                    ),
                    requires_grad=False,
                )
                for fragment in self.spec.fragments
            ]
        )
        self.projection = Parameter(
            torch.empty(
                (self.spec.local_width, embedding_dim),
                dtype=dtype,
            ),
            requires_grad=False,
        )
        self._table_loaded = [False] * len(self.spec.fragments)
        self._projection_loaded = [False] * len(self.spec.fragments)
        self.register_buffer(
            "_ignored_token_ids",
            torch.tensor(self.spec.ignored_token_ids, dtype=torch.int32),
            persistent=False,
        )

    def load_weight(self, weight_name: str, loaded_weight: torch.Tensor) -> bool:
        """Load one word, OE table, or projection checkpoint tensor.

        Returns:
            ``True`` when this layer owns the weight name. Non-local OE branches
            also return ``True`` because skipping them is intentional.
        """
        if weight_name.endswith("embed_tokens.weight") and not (
            "oe_embed_tokens" in weight_name
            or "ngram_embeddings.embedders" in weight_name
        ):
            self.word_embedding.weight_loader(
                self.word_embedding.weight,
                loaded_weight,
            )
            return True

        table_branch = _branch_index(weight_name, projection=False)
        if table_branch is not None:
            for local_index, fragment in enumerate(self.spec.fragments):
                if fragment.branch_id != table_branch:
                    continue
                expected = (fragment.modulus, self.spec.branch_width)
                if tuple(loaded_weight.shape) != expected:
                    raise ValueError(
                        f"OE table branch {table_branch} has shape "
                        f"{tuple(loaded_weight.shape)}, expected {expected}"
                    )
                self.oe_tables[local_index].data.copy_(
                    loaded_weight[
                        :,
                        fragment.feature_begin : fragment.feature_end,
                    ]
                )
                self._table_loaded[local_index] = True
            return True

        projection_branch = _branch_index(weight_name, projection=True)
        if projection_branch is not None:
            if tuple(loaded_weight.shape) == (
                self.embedding_dim,
                self.spec.branch_width,
            ):
                branch_projection = loaded_weight.t()
            elif tuple(loaded_weight.shape) == (
                self.spec.branch_width,
                self.embedding_dim,
            ):
                branch_projection = loaded_weight
            else:
                raise ValueError(
                    f"OE projection branch {projection_branch} has shape "
                    f"{tuple(loaded_weight.shape)}, expected "
                    f"({self.embedding_dim}, {self.spec.branch_width}) or "
                    f"({self.spec.branch_width}, {self.embedding_dim})"
                )
            output_begin = 0
            for local_index, fragment in enumerate(self.spec.fragments):
                output_end = output_begin + fragment.feature_width
                if fragment.branch_id == projection_branch:
                    self.projection.data[output_begin:output_end].copy_(
                        branch_projection[fragment.feature_begin : fragment.feature_end]
                    )
                    self._projection_loaded[local_index] = True
                output_begin = output_end
            return True
        return False

    def validate_loaded_weights(self) -> None:
        """Fail startup when any rank-local projection fragment is missing."""
        missing_tables = [
            fragment.branch_id
            for fragment, loaded in zip(
                self.spec.fragments,
                self._table_loaded,
            )
            if not loaded
        ]
        missing_projections = [
            fragment.branch_id
            for fragment, loaded in zip(
                self.spec.fragments,
                self._projection_loaded,
            )
            if not loaded
        ]
        if missing_tables or missing_projections:
            raise RuntimeError(
                "missing LongCat OE weights: "
                f"tables={missing_tables}, projections={missing_projections}"
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        ctx,
    ) -> torch.Tensor:
        if input_ids.numel() == 0:
            return self.word_embedding(input_ids)
        view = ctx.request_token_history
        if view is None:
            raise RuntimeError("LongCat OE forward requires request token history")
        word_partial = self.word_embedding(input_ids, reduce_results=False)
        bypass_mask = None
        if self.spec.segment_ignored_tokens and self.spec.ignored_token_ids:
            bypass_mask = (input_ids.unsqueeze(-1) == self._ignored_token_ids).any(
                dim=-1
            )
        activation = append_packed_lookup_(
            input_ids,
            view.input_start_offsets,
            view.req_pool_indices,
            view.active_request_mask,
            view.history_token_ids,
            view.committed_lengths,
            tuple(self.oe_tables),
            spec=self.spec,
            enable_pdl=True,
        )
        special_projection = None
        if bypass_mask is not None:
            # Lite checkpoints retain the learned row-zero OE projection at
            # special positions. Only normalization is bypassed; a zero hash
            # index is not a zero embedding. Sum TP-local contributions below.
            zero_lookup = torch.cat([table[0] for table in self.oe_tables]).unsqueeze(0)
            special_projection = F.linear(zero_lookup, self.projection.t())
        local_partial = project_add_word_(
            word_partial,
            activation,
            self.projection,
            scale=self.normalize_scale,
            bypass_mask=bypass_mask,
        )
        if special_projection is not None:
            local_partial = _add_megatron_special_token_oe(
                local_partial, special_projection, bypass_mask
            )
        if self.tp_size > 1:
            local_partial = all_reduce(local_partial, self.tp_group)
        return local_partial

    @property
    def weight(self) -> Parameter:
        """Expose the tied regular word-embedding weight."""
        return self.word_embedding.weight

    @weight.setter
    def weight(self, value: Parameter) -> None:
        self.word_embedding.weight = value


class HostLongCatOverEmbedding(nn.Module):
    """Host-table OE leaf migrated from the former Lite model entry.

    It stays separate from ``LongCatOverEmbedding`` because its physical ABI is
    different: table weights remain on CPU, raw activations are staged to the
    accelerator, and request context comes from checkpointed three-token tails.
    Both leaves implement the same word-plus-n-gram embedding semantics.
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = config
        self._special_token_ids = (
            config.special_token_ids
            if getattr(config, "ngram_exclude_sp_token", False)
            else ()
        )
        self.normalize_scale = (
            math.sqrt(config.oe_component_count + 1)
            if getattr(config, "ngram_fix_normalize_factor", False)
            else float(config.oe_component_count + 1)
        )
        self.embedders = nn.ModuleList()
        for _ in range(config.oe_component_count):
            table = nn.Module()
            table.register_parameter(
                "weight",
                nn.Parameter(
                    torch.empty(0, device="cpu", dtype=torch.bfloat16),
                    requires_grad=False,
                ),
            )
            self.embedders.append(table)
        self.projection = nn.Parameter(
            torch.empty(
                config.oe_component_count,
                config.oe_hidden_size,
                config.hidden_size,
                dtype=torch.bfloat16,
            ),
            requires_grad=False,
        )
        self.register_buffer(
            "ignore_tokens",
            torch.tensor(self._special_token_ids, dtype=torch.int64),
            persistent=False,
        )
        self._runtime: CheckpointedTailOEStatePreparer | None = None
        self._table_loaded = [False] * config.oe_component_count
        self._projection_loaded = [False] * config.oe_component_count

    def load_weight(self, weight_name: str, loaded_weight: torch.Tensor) -> bool:
        table_id = _branch_index(weight_name, projection=False)
        if table_id is not None:
            if not 0 <= table_id < self.config.oe_component_count:
                raise ValueError(f"Unexpected Host OE table branch {table_id}.")
            expected = (
                self.config.oe_table_rows(table_id),
                self.config.oe_hidden_size,
            )
            if tuple(loaded_weight.shape) != expected:
                raise ValueError(
                    f"Host OE table branch {table_id} has shape "
                    f"{tuple(loaded_weight.shape)}, expected {expected}"
                )
            if loaded_weight.device.type != "cpu":
                raise ValueError("Host OE embedding tables must load from CPU.")
            self.embedders[table_id].weight.data = loaded_weight
            self._table_loaded[table_id] = True
            return True

        projection_id = _branch_index(weight_name, projection=True)
        if projection_id is None:
            return False
        if not 0 <= projection_id < self.config.oe_component_count:
            raise ValueError(f"Unexpected Host OE projection branch {projection_id}.")
        expected = (self.config.hidden_size, self.config.oe_hidden_size)
        if tuple(loaded_weight.shape) != expected:
            raise ValueError(
                f"Host OE projection branch {projection_id} has shape "
                f"{tuple(loaded_weight.shape)}, expected {expected}"
            )
        self.projection.data[projection_id].copy_(
            loaded_weight.t().to(device=self.projection.device)
        )
        self._projection_loaded[projection_id] = True
        return True

    def validate_loaded_weights(self) -> None:
        missing_tables = [
            branch for branch, loaded in enumerate(self._table_loaded) if not loaded
        ]
        missing_projections = [
            branch
            for branch, loaded in enumerate(self._projection_loaded)
            if not loaded
        ]
        if missing_tables or missing_projections:
            raise RuntimeError(
                "missing Host OE weights: "
                f"tables={missing_tables}, projections={missing_projections}"
            )

    def _host_special_mask(self, tokens: torch.Tensor) -> torch.Tensor:
        ignore = tokens.new_tensor(self._special_token_ids)
        return (tokens.unsqueeze(-1) == ignore).any(dim=-1)

    def ngram_ids(
        self,
        input_ids: torch.Tensor,
        initial_context: torch.Tensor,
        lengths: Iterable[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return local table IDs, special mask, and each request's raw tail."""
        if input_ids.device.type != "cpu" or initial_context.device.type != "cpu":
            raise ValueError("Lite OE host IDs and context must be CPU tensors.")
        if input_ids.ndim != 1 or input_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("Lite OE input_ids must be a flat CPU integer tensor.")
        if (
            initial_context.ndim != 2
            or initial_context.shape[1] != self.config.emb_neighbor_num - 1
            or initial_context.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("Lite OE initial_context must have shape [batch, 3].")

        lengths_tensor = torch.as_tensor(tuple(lengths), dtype=torch.int64)
        if (
            lengths_tensor.ndim != 1
            or lengths_tensor.shape[0] != initial_context.shape[0]
            or bool((lengths_tensor < 0).any())
            or int(lengths_tensor.sum()) != input_ids.numel()
        ):
            raise ValueError("Lite OE ragged lengths must match the flat input.")
        tokens = input_ids.to(torch.int64)
        context = initial_context.to(torch.int64)
        if bool((tokens < 0).any()) or bool((tokens >= self.config.vocab_size).any()):
            raise ValueError(
                f"Lite OE input IDs must be inside [0,{self.config.vocab_size})."
            )
        if tokens.numel() == 0:
            return (
                torch.empty((0, self.config.oe_component_count), dtype=torch.int64),
                torch.empty(0, dtype=torch.bool),
                context.clone(),
            )

        starts = torch.cumsum(lengths_tensor, dim=0) - lengths_tensor
        request_ids = torch.repeat_interleave(
            torch.arange(len(lengths_tensor)), lengths_tensor
        )
        columns = torch.arange(tokens.numel()) - starts[request_ids]
        relative = torch.arange(-3, 1)
        virtual = columns.unsqueeze(1) + relative.unsqueeze(0)
        from_context = virtual < 0
        context_values = context[request_ids].gather(1, (virtual + 3).clamp(0, 2))
        token_indices = (starts[request_ids].unsqueeze(1) + virtual).clamp(
            0, tokens.numel() - 1
        )
        windows = torch.where(from_context, context_values, tokens[token_indices])
        special = self._host_special_mask(windows)
        clean = windows.masked_fill(special, 0)

        shifted = {
            distance: clean[:, 3 - distance]
            * (clean[:, 3 - distance :].ne(0).all(dim=1))
            for distance in range(1, self.config.emb_neighbor_num)
        }
        table_ids = []
        current = clean[:, -1]
        for order in range(2, self.config.emb_neighbor_num + 1):
            for split_id in range(self.config.emb_split_num):
                table_id = (order - 2) * self.config.emb_split_num + split_id
                rows = self.config.oe_table_rows(table_id)
                local_ids = current.clone()
                for distance in range(1, order):
                    local_ids.add_(
                        shifted[distance] * pow(self.config.vocab_size, distance, rows)
                    )
                table_ids.append(local_ids.remainder(rows))

        tail_virtual = lengths_tensor.unsqueeze(1) + torch.arange(-3, 0)
        tail_from_context = tail_virtual < 0
        tail_context = context.gather(1, (tail_virtual + 3).clamp(0, 2))
        tail_token_indices = (starts.unsqueeze(1) + tail_virtual).clamp(
            0, tokens.numel() - 1
        )
        final_context = torch.where(
            tail_from_context, tail_context, tokens[tail_token_indices]
        )
        return torch.stack(table_ids, dim=1), special[:, -1], final_context

    def lookup_host(self, local_ids: torch.Tensor) -> torch.Tensor:
        if (
            local_ids.device.type != "cpu"
            or local_ids.dtype != torch.int64
            or local_ids.ndim != 2
            or local_ids.shape[1] != self.config.oe_component_count
        ):
            raise ValueError("Lite OE lookup IDs must be CPU int64 [tokens, 12].")
        if local_ids.shape[0] == 0:
            return torch.empty(
                (0, self.config.oe_component_count, self.config.oe_hidden_size),
                dtype=torch.bfloat16,
            )
        outputs = []
        for table_id, table in enumerate(self.embedders):
            expected = (self.config.oe_table_rows(table_id), self.config.oe_hidden_size)
            if tuple(table.weight.shape) != expected:
                raise RuntimeError(
                    f"Lite OE table {table_id} is not loaded: "
                    f"got {tuple(table.weight.shape)}, expected {expected}."
                )
            ids = local_ids[:, table_id]
            if bool((ids < 0).any()) or bool((ids >= expected[0]).any()):
                raise ValueError(f"Lite OE table {table_id} received an invalid ID.")
            outputs.append(F.embedding(ids, table.weight))
        return torch.stack(outputs, dim=1)

    def project_and_merge(
        self,
        word_hidden_states: torch.Tensor,
        raw_oe: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        tokens = word_hidden_states.shape[0]
        expected_raw = (
            tokens,
            self.config.oe_component_count,
            self.config.oe_hidden_size,
        )
        if (
            word_hidden_states.ndim != 2
            or word_hidden_states.shape[1] != self.config.hidden_size
            or tuple(raw_oe.shape) != expected_raw
            or input_ids.ndim != 1
            or input_ids.shape[0] != tokens
        ):
            raise ValueError("Lite OE projection inputs have incompatible shapes.")
        if tokens == 0:
            return word_hidden_states
        projected = raw_oe.reshape(tokens, self.config.hidden_size) @ (
            self.projection.reshape(self.config.hidden_size, self.config.hidden_size)
        )
        merged = (word_hidden_states + projected) / self.normalize_scale
        special = (input_ids.unsqueeze(-1) == self.ignore_tokens).any(dim=-1)
        merged = torch.where(special.unsqueeze(-1), word_hidden_states, merged)
        return _add_megatron_special_token_oe(merged, projected, special)

    def initialize_runtime(
        self,
        *,
        context_pages: torch.Tensor,
        checkpoint_granularity: int,
        max_request_slots: int,
        max_graph_tokens: int,
        device: str,
    ) -> None:
        self._runtime = CheckpointedTailOEStatePreparer(
            self,
            context_pages=context_pages,
            checkpoint_granularity=checkpoint_granularity,
            max_request_slots=max_request_slots,
            max_graph_tokens=max_graph_tokens,
            device=device,
        )

    def prepare_forward_op(
        self,
        forward_op: Any,
        *,
        resolved_input_ids: torch.Tensor,
        graph_tokens: int | None,
    ) -> torch.Tensor:
        if self._runtime is None:
            raise RuntimeError("Lite OE runtime is not initialized.")
        return self._runtime.prepare_forward_op(
            forward_op,
            resolved_input_ids=resolved_input_ids,
            graph_tokens=graph_tokens,
        )

    def prepared_raw_oe(self, num_tokens: int) -> torch.Tensor:
        if self._runtime is None:
            raise RuntimeError("Lite OE runtime is not initialized.")
        return self._runtime.prepared_raw_oe(num_tokens)


class CheckpointedTailOEStatePreparer:
    """Slot-safe host OE lookup and fixed-address Decode staging."""

    def __init__(
        self,
        ngram: HostLongCatOverEmbedding,
        *,
        context_pages: torch.Tensor,
        checkpoint_granularity: int,
        max_request_slots: int,
        max_graph_tokens: int,
        device: str,
    ) -> None:
        if (
            context_pages.ndim != 2
            or context_pages.shape[1] != ngram.config.emb_neighbor_num - 1
            or context_pages.dtype != torch.int32
        ):
            raise ValueError("Lite OE context pages must be int32 [pages, 3].")
        if (
            checkpoint_granularity <= 0
            or max_request_slots <= 0
            or max_graph_tokens <= 0
        ):
            raise ValueError("Lite OE runtime sizes must be positive.")
        self.ngram = ngram
        self.context_pages = context_pages
        self.checkpoint_granularity = checkpoint_granularity
        self.max_request_slots = max_request_slots
        self.max_graph_tokens = max_graph_tokens
        self._owners: list[str | None] = [None] * max_request_slots
        self._lengths = [0] * max_request_slots
        self._contexts = torch.full(
            (max_request_slots, ngram.config.emb_neighbor_num - 1),
            ngram.config.eos_token_id,
            dtype=torch.int64,
        )
        self.staging = torch.zeros(
            (
                max_graph_tokens,
                ngram.config.oe_component_count,
                ngram.config.oe_hidden_size,
            ),
            dtype=torch.bfloat16,
            device=device,
        )
        self._prepared = self.staging
        self.restore_count = 0

    @staticmethod
    def _page_id(table: Any, row: int, slot: int) -> int:
        if slot < 0 or getattr(table, "ndim", None) != 2:
            raise ValueError("Lite OE block table must be two-dimensional.")
        if row >= table.shape[0] or slot >= table.shape[1]:
            raise ValueError("Lite OE block table does not cover the state slot.")
        page_id = int(table[row, slot])
        if page_id <= 0:
            raise ValueError("Lite OE active state must not use page 0 or padding.")
        return page_id

    def _initial_context(
        self,
        *,
        request_id: str,
        request_pool_index: int,
        before_length: int,
        table: Any,
        row: int,
    ) -> torch.Tensor:
        if not 0 <= request_pool_index < self.max_request_slots:
            raise ValueError("Lite OE request-pool index is out of range.")
        if before_length < 0:
            raise ValueError("Lite OE prefix length must be non-negative.")
        if (
            self._owners[request_pool_index] == request_id
            and self._lengths[request_pool_index] == before_length
        ):
            return self._contexts[request_pool_index]
        if before_length == 0:
            return torch.full(
                (self.ngram.config.emb_neighbor_num - 1,),
                self.ngram.config.eos_token_id,
                dtype=torch.int64,
            )
        page_id = self._page_id(
            table,
            row,
            (before_length - 1) // self.checkpoint_granularity,
        )
        if page_id >= self.context_pages.shape[0]:
            raise ValueError("Lite OE input state page is out of range.")
        self.restore_count += 1
        return self.context_pages[page_id].to(device="cpu", dtype=torch.int64)

    def prepare(
        self,
        *,
        request_ids: Iterable[str],
        request_pool_indices: Iterable[int],
        input_ids: torch.Tensor,
        lengths: Iterable[int],
        before_lengths: Iterable[int],
        block_table: Any,
        graph_tokens: int | None = None,
    ) -> torch.Tensor:
        request_ids = tuple(request_ids)
        request_pool_indices = tuple(request_pool_indices)
        lengths = tuple(lengths)
        before_lengths = tuple(before_lengths)
        batch = len(request_ids)
        if not (
            len(request_pool_indices)
            == len(lengths)
            == len(before_lengths)
            == batch
            == block_table.shape[0]
        ):
            raise ValueError("Lite OE request metadata lengths do not match.")
        if input_ids.device.type != "cpu":
            raise ValueError("Lite OE input IDs must remain on CPU for host lookup.")
        if sum(lengths) != input_ids.numel() or any(length < 0 for length in lengths):
            raise ValueError("Lite OE ragged lengths do not match input IDs.")

        contexts = torch.stack(
            [
                self._initial_context(
                    request_id=request_ids[row],
                    request_pool_index=request_pool_indices[row],
                    before_length=before_lengths[row],
                    table=block_table,
                    row=row,
                )
                for row in range(batch)
            ]
        )
        ids, _, final_contexts = self.ngram.ngram_ids(input_ids, contexts, lengths)

        output_pages: list[int | None] = []
        written_pages: dict[int, tuple[str, int]] = {}
        for row, length in enumerate(lengths):
            request_pool_index = request_pool_indices[row]
            request_id = request_ids[row]
            after_length = before_lengths[row] + length
            page_id = (
                self._page_id(
                    block_table,
                    row,
                    (after_length - 1) // self.checkpoint_granularity,
                )
                if length
                else None
            )
            if page_id is not None:
                owner = (request_id, request_pool_index)
                if page_id in written_pages and written_pages[page_id] != owner:
                    raise ValueError("Lite OE output state page has multiple owners.")
                if page_id >= self.context_pages.shape[0]:
                    raise ValueError("Lite OE output state page is out of range.")
                written_pages[page_id] = owner
            output_pages.append(page_id)

        for row, page_id in enumerate(output_pages):
            request_pool_index = request_pool_indices[row]
            request_id = request_ids[row]
            after_length = before_lengths[row] + lengths[row]
            if page_id is not None:
                self.context_pages[page_id].copy_(
                    final_contexts[row].to(
                        device=self.context_pages.device,
                        dtype=torch.int32,
                    ),
                    non_blocking=True,
                )
            self._owners[request_pool_index] = request_id
            self._lengths[request_pool_index] = after_length
            self._contexts[request_pool_index].copy_(final_contexts[row])

        raw_oe = self.ngram.lookup_host(ids)
        if graph_tokens is None:
            self._prepared = raw_oe.to(self.staging.device, non_blocking=True)
            return self._prepared
        if not input_ids.numel() <= graph_tokens <= self.max_graph_tokens:
            raise ValueError("Lite OE graph token count is outside the staging range.")
        self.staging[input_ids.numel() :].zero_()
        if input_ids.numel():
            self.staging[: input_ids.numel()].copy_(raw_oe, non_blocking=True)
        self._prepared = self.staging[:graph_tokens]
        return self._prepared

    def prepare_forward_op(
        self,
        forward_op: Any,
        *,
        resolved_input_ids: torch.Tensor,
        graph_tokens: int | None,
    ) -> torch.Tensor:
        request_ids = tuple(forward_op.request_ids)
        request_pool_indices = tuple(forward_op.request_pool_indices)
        lengths = tuple(forward_op.input_lengths)
        num_extends = forward_op.num_extends()
        decode_ids = tuple(forward_op.decode_input_ids)
        if len(decode_ids) != len(request_ids) - num_extends:
            raise ValueError("Lite OE decode IDs do not match the decode rows.")
        if any(not 0 <= slot < self.max_request_slots for slot in request_pool_indices):
            raise ValueError("Lite OE request-pool index is out of range.")
        if resolved_input_ids.ndim != 1 or resolved_input_ids.numel() != sum(lengths):
            raise ValueError("Lite OE resolved input IDs do not match ragged lengths.")
        if any(length != 1 for length in lengths[num_extends:]):
            raise RuntimeError("Lite OE speculative Decode is not supported yet.")
        tokens = resolved_input_ids.to(device="cpu", dtype=torch.int64)
        before_lengths = tuple(forward_op.extend_prefix_lens) + tuple(
            (
                self._lengths[slot]
                if self._owners[slot] == request_id
                else forward_op.prefill_lengths[row]
            )
            for row, (request_id, slot) in enumerate(
                zip(
                    request_ids[num_extends:],
                    request_pool_indices[num_extends:],
                    strict=True,
                ),
                start=num_extends,
            )
        )
        return self.prepare(
            request_ids=request_ids,
            request_pool_indices=request_pool_indices,
            input_ids=tokens,
            lengths=lengths,
            before_lengths=before_lengths,
            block_table=forward_op.block_tables_arrays()["lite_oe"],
            graph_tokens=graph_tokens,
        )

    def prepared_raw_oe(self, num_tokens: int) -> torch.Tensor:
        if not 0 <= num_tokens <= self._prepared.shape[0]:
            raise ValueError("Lite OE prepared tensor does not cover model input.")
        return self._prepared[:num_tokens]


__all__ = [
    "CheckpointedTailOEStatePreparer",
    "HostLongCatOverEmbedding",
    "LongCatOverEmbedding",
    "resolve_longcat_oe_hyperparameters",
    "resolve_longcat_oe_spec",
]
