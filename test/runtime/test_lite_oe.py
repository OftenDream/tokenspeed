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

from __future__ import annotations

from test.runtime.legacy_host_oe_reference import LegacyHostOEReference
from types import SimpleNamespace

import torch
from tokenspeed_kernel.ops.over_embedding import project_add_word_


def _config() -> SimpleNamespace:
    rows = (11, 13, 17, 19)
    return SimpleNamespace(
        vocab_size=32,
        emb_neighbor_num=3,
        emb_split_num=2,
        oe_component_count=4,
        oe_hidden_size=2,
        hidden_size=8,
        special_token_ids=(3,),
        ngram_exclude_sp_token=True,
        ngram_fix_normalize_factor=True,
        oe_table_rows=lambda table_id: rows[table_id],
    )


def _brute_ids(config, flat_tokens, initial_context, lengths):
    outputs = []
    tails = []
    special = set(config.special_token_ids)
    offset = 0
    lookback = config.emb_neighbor_num - 1
    for request_id, length in enumerate(lengths):
        history = initial_context[request_id].tolist()
        for token in flat_tokens[offset : offset + length].tolist():
            window = history[-lookback:] + [token]
            clean = [0 if value in special else value for value in window]
            row = []
            for order in range(2, config.emb_neighbor_num + 1):
                for split_id in range(config.emb_split_num):
                    table_id = (order - 2) * config.emb_split_num + split_id
                    rows = config.oe_table_rows(table_id)
                    value = clean[-1]
                    for distance in range(1, order):
                        if all(clean[-1 - distance :]):
                            value += clean[-1 - distance] * pow(
                                config.vocab_size, distance, rows
                            )
                    row.append(value % rows)
            outputs.append(row)
            history.append(token)
        tails.append(history[-lookback:])
        offset += length
    return (
        torch.tensor(outputs, dtype=torch.int64).reshape(-1, config.oe_component_count),
        torch.tensor(tails, dtype=torch.int64),
    )


def test_legacy_host_reference_hashes_match_scalar_oracle() -> None:
    config = _config()
    reference = LegacyHostOEReference(config, (), torch.empty(0))
    initial = torch.tensor([[2, 5], [2, 2], [7, 8]])
    tokens = torch.tensor([6, 3, 10, 11, 12, 3, 13, 14], dtype=torch.int64)
    lengths = [3, 5, 0]

    ids, special, tail = reference.ngram_ids(tokens, initial, lengths)
    expected_ids, expected_tail = _brute_ids(config, tokens, initial, lengths)

    torch.testing.assert_close(ids, expected_ids, atol=0, rtol=0)
    torch.testing.assert_close(tail, expected_tail, atol=0, rtol=0)
    torch.testing.assert_close(
        special,
        torch.tensor([False, True, False, False, False, True, False, False]),
        atol=0,
        rtol=0,
    )


def test_legacy_host_reference_lookup_projection_and_special_bypass() -> None:
    torch.manual_seed(19)
    config = _config()
    tables = tuple(
        torch.randn(config.oe_table_rows(i), config.oe_hidden_size).to(torch.bfloat16)
        for i in range(config.oe_component_count)
    )
    projection = torch.randn(
        config.oe_component_count,
        config.oe_hidden_size,
        config.hidden_size,
        dtype=torch.bfloat16,
    )
    reference = LegacyHostOEReference(config, tables, projection)
    ids = torch.stack(
        [
            torch.arange(3).remainder(config.oe_table_rows(table_id))
            for table_id in range(config.oe_component_count)
        ],
        dim=1,
    ).to(torch.int64)
    ids[1].zero_()

    raw = reference.lookup_host(ids)
    expected_raw = torch.stack(
        [tables[i][ids[:, i]] for i in range(config.oe_component_count)], dim=1
    )
    torch.testing.assert_close(raw, expected_raw, atol=0, rtol=0)

    word = torch.randn(3, config.hidden_size, dtype=torch.bfloat16)
    input_ids = torch.tensor([5, 3, 8])
    expected = reference.project_and_merge(word, raw, input_ids)
    actual = project_add_word_(
        word.clone(),
        raw.reshape(3, config.hidden_size),
        projection.reshape(config.hidden_size, config.hidden_size),
        scale=reference.normalize_scale,
        bypass_mask=input_ids == 3,
        solution="torch",
    )
    torch.testing.assert_close(actual[1], word[1], atol=0, rtol=0)
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
