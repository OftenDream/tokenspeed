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

"""CPU oracle copied from the retired Host OE runtime implementation.

This module is test-only. Keep the deliberately straightforward CPU operations
independent of the fused kernels so it can detect hash, lookup, normalization,
and special-token regressions.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch
from torch.nn import functional as F


class LegacyHostOEReference:
    """Reference-only form of the former ``HostLongCatOverEmbedding`` math."""

    def __init__(
        self,
        config,
        tables: Sequence[torch.Tensor],
        projection: torch.Tensor,
    ) -> None:
        self.config = config
        self.tables = tuple(tables)
        self.projection = projection
        self._special_token_ids = (
            tuple(config.special_token_ids)
            if getattr(config, "ngram_exclude_sp_token", False)
            else ()
        )
        self.normalize_scale = (
            (config.oe_component_count + 1) ** 0.5
            if getattr(config, "ngram_fix_normalize_factor", False)
            else float(config.oe_component_count + 1)
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
        """Return table row IDs, current-token mask, and final request tails."""
        lengths_tensor = torch.as_tensor(tuple(lengths), dtype=torch.int64)
        tokens = input_ids.to(torch.int64)
        context = initial_context.to(torch.int64)
        if tokens.numel() == 0:
            return (
                torch.empty((0, self.config.oe_component_count), dtype=torch.int64),
                torch.empty(0, dtype=torch.bool),
                context.clone(),
            )

        lookback = self.config.emb_neighbor_num - 1
        starts = torch.cumsum(lengths_tensor, dim=0) - lengths_tensor
        request_ids = torch.repeat_interleave(
            torch.arange(len(lengths_tensor)), lengths_tensor
        )
        columns = torch.arange(tokens.numel()) - starts[request_ids]
        relative = torch.arange(-lookback, 1)
        virtual = columns.unsqueeze(1) + relative.unsqueeze(0)
        from_context = virtual < 0
        context_values = context[request_ids].gather(
            1, (virtual + lookback).clamp(0, lookback - 1)
        )
        token_indices = (starts[request_ids].unsqueeze(1) + virtual).clamp(
            0, tokens.numel() - 1
        )
        windows = torch.where(from_context, context_values, tokens[token_indices])
        special = self._host_special_mask(windows)
        clean = windows.masked_fill(special, 0)

        shifted = {
            distance: clean[:, lookback - distance]
            * (clean[:, lookback - distance :].ne(0).all(dim=1))
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

        tail_virtual = lengths_tensor.unsqueeze(1) + torch.arange(-lookback, 0)
        tail_from_context = tail_virtual < 0
        tail_context = context.gather(
            1, (tail_virtual + lookback).clamp(0, lookback - 1)
        )
        tail_token_indices = (starts.unsqueeze(1) + tail_virtual).clamp(
            0, tokens.numel() - 1
        )
        final_context = torch.where(
            tail_from_context, tail_context, tokens[tail_token_indices]
        )
        return torch.stack(table_ids, dim=1), special[:, -1], final_context

    def lookup_host(self, local_ids: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [
                F.embedding(local_ids[:, table_id], table)
                for table_id, table in enumerate(self.tables)
            ],
            dim=1,
        )

    def project_and_merge(
        self,
        word_hidden_states: torch.Tensor,
        raw_oe: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        tokens = word_hidden_states.shape[0]
        projected = raw_oe.reshape(tokens, self.config.hidden_size) @ (
            self.projection.reshape(self.config.hidden_size, self.config.hidden_size)
        )
        merged = (word_hidden_states + projected) / self.normalize_scale
        special = self._host_special_mask(input_ids)
        return torch.where(special.unsqueeze(-1), word_hidden_states, merged)


__all__ = ["LegacyHostOEReference"]
