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

"""Regression coverage for Lite checkpoint special-token OE semantics."""

from types import SimpleNamespace

import pytest
import torch
from tokenspeed_kernel.ops.over_embedding import (
    OverEmbeddingSpec,
    TableFragmentSpec,
    project_add_word_,
)

import tokenspeed.runtime.layers.over_embedding as over_embedding


@pytest.mark.parametrize("fix_normalize_factor", [True, False])
@pytest.mark.parametrize("profile", ["longcat-lite", "unit-test"])
@pytest.mark.parametrize("segment_ignored_tokens", [True, False])
def test_special_token_row_zero_projection(
    monkeypatch, fix_normalize_factor, profile, segment_ignored_tokens
):
    dtype = torch.bfloat16
    spec = OverEmbeddingSpec(
        profile=profile,
        tp_size=1,
        rank=0,
        vocab_size=31,
        branch_count=2,
        branch_width=4,
        hidden_size=8,
        max_ngram_order=3,
        fragments=(
            TableFragmentSpec(0, 2, 7, 0, 4),
            TableFragmentSpec(1, 3, 9, 2, 2),
        ),
    )
    monkeypatch.setattr(over_embedding, "resolve_longcat_oe_spec", lambda **_: spec)
    layer = over_embedding.LongCatOverEmbedding(
        num_embeddings=31,
        embedding_dim=8,
        over_embedding_m=6,
        hashes_per_order=1,
        max_ngram_order=3,
        tp_rank=0,
        tp_size=1,
        tp_group=(0,),
        params_dtype=dtype,
        ignored_token_ids=(2,),
        segment_ignored_tokens=segment_ignored_tokens,
        fix_normalize_factor=fix_normalize_factor,
    )
    with torch.no_grad():
        for branch, table in enumerate(layer.oe_tables):
            table.fill_(9)
            table[0].fill_(branch + 2)
        layer.projection.fill_(0.125)

    input_ids = torch.tensor([2, 30], dtype=torch.int32)
    word = torch.ones((2, 8), dtype=dtype)
    activation = torch.full((2, 6), 0.5, dtype=dtype)
    if segment_ignored_tokens:
        activation[0].zero_()

    monkeypatch.setattr(
        layer.word_embedding, "forward", lambda *_args, **_kwargs: word.clone()
    )
    monkeypatch.setattr(
        over_embedding,
        "append_packed_lookup_",
        lambda *_args, **_kwargs: activation.clone(),
    )
    monkeypatch.setattr(
        over_embedding,
        "project_add_word_",
        lambda *args, **kwargs: project_add_word_(*args, **kwargs, solution="torch"),
    )
    view = SimpleNamespace(
        input_start_offsets=None,
        req_pool_indices=None,
        active_request_mask=None,
        history_token_ids=None,
        committed_lengths=None,
    )
    mask = torch.tensor([True, False]) if segment_ignored_tokens else None
    expected = project_add_word_(
        word.clone(),
        activation,
        layer.projection,
        scale=layer.normalize_scale,
        bypass_mask=mask,
        solution="torch",
    )
    if segment_ignored_tokens:
        # Row zero is a learned, nonzero embedding, NOT a zero activation.
        # The special-token sum bypasses sqrt(1 + branch_count) normalization.
        expected[0].fill_(2.75)  # 1 + (4 * 2 + 2 * 3) * 0.125

    actual = layer(input_ids, SimpleNamespace(request_token_history=view))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    # A local TP contribution must be combined only after adding special OE.
    layer.tp_size = 2
    seen = []

    def fake_all_reduce(value, group):
        seen.append(value.clone())
        return value * 2

    monkeypatch.setattr(over_embedding, "all_reduce", fake_all_reduce)
    reduced = layer(input_ids, SimpleNamespace(request_token_history=view))
    assert len(seen) == 1
    torch.testing.assert_close(seen[0], expected, rtol=0, atol=0)
    torch.testing.assert_close(reduced, expected * 2, rtol=0, atol=0)


@pytest.mark.parametrize("device", ["cpu", "npu"])
@pytest.mark.parametrize("fix_normalize_factor", [True, False])
@pytest.mark.parametrize("exclude_special_tokens", [True, False])
def test_host_special_token_row_zero_projection(
    device, fix_normalize_factor, exclude_special_tokens
):
    if device == "npu":
        pytest.importorskip("torch_npu")
        if not torch.npu.is_available():
            pytest.skip("Ascend NPU is unavailable")
    config = SimpleNamespace(
        hidden_size=8,
        oe_hidden_size=4,
        oe_component_count=2,
        special_token_ids=(2, 3),
        ngram_exclude_sp_token=exclude_special_tokens,
        ngram_fix_normalize_factor=fix_normalize_factor,
        oe_table_rows=lambda table_id: 7 + 2 * table_id,
    )
    layer = over_embedding.HostLongCatOverEmbedding(config)
    with torch.no_grad():
        for table_id, table in enumerate(layer.embedders):
            table.weight.data = torch.full(
                (config.oe_table_rows(table_id), 4), 9.0, dtype=torch.bfloat16
            )
            table.weight[0].fill_(table_id + 2)
        layer.projection.fill_(0.125)
    ids = torch.tensor([[0, 0], [1, 1], [0, 0]], dtype=torch.int64)
    raw = layer.lookup_host(ids).to(device)
    layer.projection.data = layer.projection.data.to(device)
    layer.ignore_tokens = layer.ignore_tokens.to(device)
    word = torch.ones((3, 8), dtype=torch.bfloat16, device=device)
    input_ids = torch.tensor([2, 5, 3], dtype=torch.int64, device=device)
    projected = raw.reshape(3, 8) @ layer.projection.reshape(8, 8)
    expected = (word + projected) / layer.normalize_scale
    if exclude_special_tokens:
        expected[0].fill_(3.5)  # 1 + (4 * 2 + 4 * 3) * 0.125
        expected[2].fill_(3.5)
    actual = layer.project_and_merge(word, raw, input_ids)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert all(table.weight.device.type == "cpu" for table in layer.embedders)


def test_shared_special_merge_keeps_ordinary_rows_and_inputs():
    merged = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16)
    original = merged.clone()
    projected = torch.tensor([[0.5, 0.25]], dtype=torch.bfloat16)
    mask = torch.tensor([True, False])
    result = over_embedding._add_megatron_special_token_oe(merged, projected, mask)
    torch.testing.assert_close(merged, original, rtol=0, atol=0)
    torch.testing.assert_close(result[0], original[0] + projected[0], rtol=0, atol=0)
    torch.testing.assert_close(result[1], original[1], rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("num_tokens", [1, 17, 1024])
@pytest.mark.parametrize("mask_mode", ["mixed", "all", "none"])
@pytest.mark.parametrize("scale", [13**0.5, 13.0])
def test_device_shared_merge_preserves_cuda_formula(num_tokens, mask_mode, scale):
    """The shared helper must preserve the previous Device merge bit for bit."""
    generator = torch.Generator(device="cuda").manual_seed(37)
    dtype = torch.bfloat16
    word = torch.randn(
        num_tokens, 3072, device="cuda", dtype=dtype, generator=generator
    )
    activation = torch.randn(
        num_tokens, 768, device="cuda", dtype=dtype, generator=generator
    )
    projection = torch.randn(768, 3072, device="cuda", dtype=dtype, generator=generator)
    zero_row = torch.randn(1, 768, device="cuda", dtype=dtype, generator=generator)
    mask = torch.arange(num_tokens, device="cuda") % 3 == 0
    if mask_mode != "mixed":
        mask.fill_(mask_mode == "all")
    activation[mask] = 0

    # The old path computes the special result before the in-place kernel.
    special_before = word + torch.nn.functional.linear(zero_row, projection.t())
    old_partial = project_add_word_(
        word.clone(), activation, projection, scale=scale, bypass_mask=mask
    )
    expected = torch.where(mask.unsqueeze(-1), special_before, old_partial)

    special_projection = torch.nn.functional.linear(zero_row, projection.t())
    new_partial = project_add_word_(
        word.clone(), activation, projection, scale=scale, bypass_mask=mask
    )
    actual = over_embedding._add_megatron_special_token_oe(
        new_partial, special_projection, mask
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
