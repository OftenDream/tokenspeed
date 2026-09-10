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
from dataclasses import replace

import torch
import torch.nn.functional as F
from tokenspeed_kernel.ops.over_embedding import (
    OverEmbeddingSpec,
    TableFragmentSpec,
    append_packed_lookup_,
    project_add_word_,
    register_host_tables_,
)
from torch import nn
from torch.nn.parameter import Parameter

from tokenspeed.runtime.distributed.comm_ops import all_reduce
from tokenspeed.runtime.layers.vocab_parallel_embedding import VocabParallelEmbedding

_LongCatOeConfig = tuple[int, int, int, int, int, int, int]


def _fragment(
    branch_id: int,
    *,
    hashes_per_order: int,
    modulus0: int,
    feature_begin: int = 0,
    feature_width: int,
) -> TableFragmentSpec:
    return TableFragmentSpec(
        branch_id=branch_id,
        ngram_order=branch_id // hashes_per_order + 2,
        modulus=modulus0 + 2 * branch_id,
        feature_begin=feature_begin,
        feature_width=feature_width,
    )


def _spec(
    *,
    profile: str,
    vocab_size: int,
    hidden_size: int,
    max_ngram_order: int,
    hashes_per_order: int,
    modulus0: int,
    tp_size: int,
    tp_rank: int,
    branch_width: int,
    fragments: tuple[TableFragmentSpec, ...],
) -> OverEmbeddingSpec:
    branch_count = (max_ngram_order - 1) * hashes_per_order
    expected_hidden_size = branch_count * branch_width
    if hidden_size != expected_hidden_size:
        raise ValueError(
            f"invalid OE geometry for {profile}: hidden_size={hidden_size}, "
            f"but (max_ngram_order - 1) * hashes_per_order * branch_width "
            f"= {expected_hidden_size}"
        )
    return OverEmbeddingSpec(
        profile=profile,
        tp_size=tp_size,
        rank=tp_rank,
        vocab_size=vocab_size,
        branch_count=branch_count,
        branch_width=branch_width,
        hidden_size=hidden_size,
        max_ngram_order=max_ngram_order,
        fragments=fragments,
    )


# Concrete checkpoint geometry indexed by every field that affects OE layout.
# The rank is part of the key, so resolution returns an OverEmbeddingSpec
# directly rather than dispatching through model-specific factory functions.
_LONGCAT_OE_SPEC_BY_CONFIG: dict[_LongCatOeConfig, OverEmbeddingSpec] = {
    **{
        (163840, 8192, 5, 4, 16476898, 8, rank): _spec(
            profile="longcat-2.0",
            vocab_size=163840,
            hidden_size=8192,
            max_ngram_order=5,
            hashes_per_order=4,
            modulus0=16476898,
            tp_size=8,
            tp_rank=rank,
            branch_width=512,
            fragments=tuple(
                _fragment(
                    branch_id,
                    hashes_per_order=4,
                    modulus0=16476898,
                    feature_width=512,
                )
                for branch_id in (2 * rank, 2 * rank + 1)
            ),
        )
        for rank in range(8)
    },
    **{
        (163840, 3072, 4, 4, 4718593, 4, rank): _spec(
            profile="longcat-lite",
            vocab_size=163840,
            hidden_size=3072,
            max_ngram_order=4,
            hashes_per_order=4,
            modulus0=4718593,
            tp_size=4,
            tp_rank=rank,
            branch_width=256,
            fragments=tuple(
                _fragment(
                    branch_id,
                    hashes_per_order=4,
                    modulus0=4718593,
                    feature_width=256,
                )
                for branch_id in (rank, rank + 4, rank + 8)
            ),
        )
        for rank in range(4)
    },
    **{
        (163840, 3072, 4, 4, 4718593, 8, rank): _spec(
            profile="longcat-lite",
            vocab_size=163840,
            hidden_size=3072,
            max_ngram_order=4,
            hashes_per_order=4,
            modulus0=4718593,
            tp_size=8,
            tp_rank=rank,
            branch_width=256,
            fragments=(
                _fragment(
                    rank,
                    hashes_per_order=4,
                    modulus0=4718593,
                    feature_width=256,
                ),
                _fragment(
                    8 + rank // 2,
                    hashes_per_order=4,
                    modulus0=4718593,
                    feature_begin=(rank % 2) * 128,
                    feature_width=128,
                ),
            ),
        )
        for rank in range(8)
    },
    **{
        (163840, 4096, 5, 4, 9765520, 8, rank): _spec(
            profile="longcat-lite",
            vocab_size=163840,
            hidden_size=4096,
            max_ngram_order=5,
            hashes_per_order=4,
            modulus0=9765520,
            tp_size=8,
            tp_rank=rank,
            branch_width=256,
            fragments=tuple(
                _fragment(
                    branch_id,
                    hashes_per_order=4,
                    modulus0=9765520,
                    feature_width=256,
                )
                for branch_id in (rank, rank + 8)
            ),
        )
        for rank in range(8)
    },
}


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
    ranks without duplication. Pro's 8192-wide two-branch ownership stays
    contiguous; Lite checkpoints retain interleaved full rounds, including
    their 4096-wide two-round case.
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

    is_pro_contiguous = (
        hidden_size == 8192
        and branch_count == 16
        and tp_size == 8
        and full_rounds == 2
        and tail_branches == 0
    )
    for index in range(full_rounds):
        branch = (
            tp_rank * full_rounds + index
            if is_pro_contiguous
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
        table_placement: str = "device",
    ) -> None:
        super().__init__()
        if table_placement not in ("device", "host"):
            raise ValueError("table_placement must be 'device' or 'host'")
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.tp_group = tp_group
        self.table_placement = table_placement
        dtype = params_dtype or torch.get_default_dtype()
        self.oe_modulus0 = over_embedding_m + 1
        self.spec = resolve_longcat_oe_spec(
            vocab_size=num_embeddings,
            hidden_size=embedding_dim,
            max_ngram_order=max_ngram_order,
            hashes_per_order=hashes_per_order,
            modulus0=self.oe_modulus0,
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
                    (
                        torch.empty(0, dtype=dtype, device="cpu")
                        if table_placement == "host"
                        else torch.empty(
                            (fragment.modulus, fragment.feature_width),
                            dtype=dtype,
                        )
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
        self._host_tables_registered = False
        self.register_buffer("_host_zero_lookup", None, persistent=False)

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
            if not 0 <= table_branch < self.spec.branch_count:
                raise ValueError(
                    f"OE table branch {table_branch} is outside the configured "
                    f"range [0, {self.spec.branch_count})"
                )
            expected = (
                self.oe_modulus0 + 2 * table_branch,
                self.spec.branch_width,
            )
            expected_dtype = self.projection.dtype
            if (
                tuple(loaded_weight.shape) != expected
                or loaded_weight.dtype != expected_dtype
            ):
                raise ValueError(
                    f"OE table branch {table_branch} has shape/dtype "
                    f"{tuple(loaded_weight.shape)}/{loaded_weight.dtype}, "
                    f"expected {expected}/{expected_dtype} from the configured "
                    "OE modulus and branch width"
                )
            for local_index, fragment in enumerate(self.spec.fragments):
                if fragment.branch_id != table_branch:
                    continue
                table_fragment = loaded_weight[
                    :, fragment.feature_begin : fragment.feature_end
                ]
                if self.table_placement == "host":
                    if loaded_weight.device.type != "cpu":
                        raise ValueError("Host OE tables must load from CPU storage")
                    # Preserve safetensors mmap storage instead of allocating
                    # an anonymous multi-GiB compact copy.
                    self.oe_tables[local_index].data = table_fragment
                else:
                    self.oe_tables[local_index].data.copy_(table_fragment)
                self._table_loaded[local_index] = True
            return True

        projection_branch = _branch_index(weight_name, projection=True)
        if projection_branch is not None:
            if not 0 <= projection_branch < self.spec.branch_count:
                raise ValueError(
                    f"OE projection branch {projection_branch} is outside the "
                    f"configured range [0, {self.spec.branch_count})"
                )
            if loaded_weight.dtype != self.projection.dtype:
                raise ValueError(
                    f"OE projection branch {projection_branch} has dtype "
                    f"{loaded_weight.dtype}, expected {self.projection.dtype}"
                )
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
            enable_pdl=self.table_placement == "device",
        )
        special_projection = None
        if bypass_mask is not None:
            # Lite checkpoints retain the learned row-zero OE projection at
            # special positions. Only normalization is bypassed; a zero hash
            # index is not a zero embedding. Sum TP-local contributions below.
            if self.table_placement == "host":
                zero_lookup = self._host_zero_lookup
                if zero_lookup is None:
                    raise RuntimeError(
                        "Host OE requires initialize_host_runtime() before forward"
                    )
            else:
                zero_lookup = torch.cat(
                    [table[0] for table in self.oe_tables]
                ).unsqueeze(0)
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

    def initialize_host_runtime(self) -> None:
        """Register Host tables and stage local row-zero fragments before capture."""
        if self.table_placement != "host":
            return
        if not self._host_tables_registered:
            if (
                self.spec.profile == "longcat-lite"
                and self.spec.segment_ignored_tokens
                and self.spec.ignored_token_ids
            ):
                self._host_zero_lookup = (
                    torch.cat([table[0] for table in self.oe_tables])
                    .unsqueeze(0)
                    .to(device=self.projection.device)
                )
            register_host_tables_(
                tuple(self.oe_tables),
                device=self.projection.device,
            )
            self._host_tables_registered = True

    @property
    def weight(self) -> Parameter:
        """Expose the tied regular word-embedding weight."""
        return self.word_embedding.weight

    @weight.setter
    def weight(self, value: Parameter) -> None:
        self.word_embedding.weight = value


__all__ = [
    "LongCatOverEmbedding",
    "resolve_longcat_oe_hyperparameters",
    "resolve_longcat_oe_spec",
]
