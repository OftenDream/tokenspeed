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

import json
from itertools import cycle, islice
from types import SimpleNamespace

import pytest
import torch
from run_lite_moe_precision import (
    case_name,
    cases,
    validate_records,
)
from validate_lite_moe_fused_exchange import (
    chunk_reference_experts,
    error_metrics,
    finite_bitwise_equal,
    precision_token_lengths,
    prepare_graph_streams,
    random_token_lengths,
    stability_finished,
)


def test_error_metrics_report_signed_zero_and_magnitudes():
    a = torch.tensor([-0.0, 2.0])
    b = torch.tensor([0.0, 1.0])
    r = error_metrics(a, b)
    assert r["finite"] and r["elements"] == r["different"] == 2
    assert r["max_abs"] == 1 and r["mean_abs"] == 0.5 and r["relative_l2"] == 1


def test_error_metrics_do_not_report_nonfinite_as_exact():
    r = error_metrics(torch.tensor([float("nan")]), torch.tensor([float("nan")]))
    assert not r["finite"] and r["max_abs"] is None and r["relative_l2"] is None


def test_dense_precision_lengths_include_every_tail_value():
    lengths = precision_token_lengths("dense")
    assert lengths[:14] == [2**power for power in range(14)]
    assert lengths[14:] == list(range(8193, 16385))
    assert len(lengths) == len(set(lengths)) == 8206


@pytest.mark.parametrize("spec", ["0", "1,1", "1,-2", "", "1,,2"])
def test_precision_lengths_reject_invalid_sequence(spec):
    with pytest.raises(ValueError):
        precision_token_lengths(spec)


def test_precision_lengths_keep_explicit_order():
    assert precision_token_lengths("32,1,8193") == [32, 1, 8193]


def test_graph_stream_reservation_skips_live_limited_pool_entries(monkeypatch):
    shared, other_limited, composed, fused = (object() for _ in range(4))
    pool = cycle([shared, other_limited, composed, fused])
    limits = {"cube_core_num": 24, "vector_core_num": 48}
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(
            Stream=lambda **kw: next(pool),
            get_device_limit=lambda device: limits,
            get_stream_limit=lambda stream: (
                {"cube_core_num": 4, "vector_core_num": 8}
                if stream in (shared, other_limited)
                else limits
            ),
        ),
    )
    streams = prepare_graph_streams("npu:0", shared)
    assert streams == {"composed": composed, "fused": fused}
    # Cycling the allocator does not change the reserved streams or any quota.
    for _ in range(100):
        next(pool)
        assert streams["composed"] is composed and streams["fused"] is fused


def test_graph_stream_reservation_rejects_exhausted_pool(monkeypatch):
    shared, only_free = object(), object()
    pool = cycle([shared, only_free])
    limits = {"cube_core_num": 24, "vector_core_num": 48}
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(
            Stream=lambda **kw: next(pool),
            get_device_limit=lambda device: limits,
            get_stream_limit=lambda stream: limits,
        ),
    )
    with pytest.raises(RuntimeError, match="independent unrestricted"):
        prepare_graph_streams("npu:0", shared)


def test_random_length_soak_is_reproducible_and_covers_boundaries():
    lengths = list(random_token_lengths(10000, 20260909, 4096))
    assert lengths == list(random_token_lengths(10000, 20260909, 4096))
    assert lengths != list(random_token_lengths(10000, 123, 4096))
    assert len(lengths) == 10000
    assert min(lengths) == 1 and max(lengths) == 4096
    assert {31, 32, 33, 64, 65, 1023, 1024, 1025, 2049, 4096} <= set(lengths[:32])
    assert len(set(lengths)) > 2000
    assert 4000 < sum(value <= 128 for value in lengths) < 6000


def test_large_length_boundaries_and_duration_extension():
    lengths = list(
        islice(random_token_lengths(1, 20260909, 32684, unbounded=True), 10001)
    )
    assert {4097, 8191, 8192, 8193, 16383, 16384, 16385, 32683, 32684} <= set(
        lengths[:32]
    )
    assert len(lengths) == 10001 and max(lengths) == 32684


@pytest.mark.parametrize(
    "count,elapsed,done",
    [(9999, 10801, False), (10000, 10799, False), (10000, 10800, True)],
)
def test_soak_requires_both_time_and_iterations(count, elapsed, done):
    assert stability_finished(count, 10000, elapsed, 10800) is done


@pytest.mark.parametrize("chunk_rows", [1, 3, 8, 16])
def test_reference_chunking_aligns_routes_and_logits(chunk_rows):
    x = torch.arange(16, dtype=torch.float32).reshape(8, 2)
    ids, weights, logits = x + 10, x + 20, x + 30
    calls = []
    plan, w = object(), object()

    def apply(p, part, weight, scores, **kwargs):
        assert p is plan and weight is w and kwargs["flag"] == "unchanged"
        assert torch.equal(kwargs["topk_ids"], part + 10)
        assert torch.equal(kwargs["topk_weights"], part + 20)
        assert torch.equal(scores, part + 30)
        calls.append(part.shape[0])
        return part + 1

    output = chunk_reference_experts(
        apply,
        plan,
        x,
        w,
        logits,
        chunk_rows=chunk_rows,
        topk_ids=ids,
        topk_weights=weights,
        flag="unchanged",
    )
    assert torch.equal(output, x + 1) and output.data_ptr() != x.data_ptr()
    assert sum(calls) == 8 and max(calls) <= chunk_rows


@pytest.mark.parametrize("iterations,maximum", [(0, 4096), (-1, 4096), (10, 128)])
def test_random_length_soak_rejects_invalid_arguments(iterations, maximum):
    with pytest.raises(ValueError):
        list(random_token_lengths(iterations, 1, maximum))


def test_matrix_covers_decode_prefill_boundaries_and_smooth():
    matrix = cases()
    assert len({case_name(case) for case in matrix}) == len(matrix)
    assert {1, 7, 31, 32, 33, 64, 65} <= {
        c["tokens"] for c in matrix if c["role"] == "decode"
    }
    assert {128, 257, 512, 1024, 2049, 4096} <= {
        c["tokens"] for c in matrix if c["role"] == "prefill"
    }
    assert {c["smooth"] for c in matrix} == {"none", "w13", "w2", "both"}
    assert {c["route"] for c in matrix} >= {
        "mixed",
        "real_only",
        "zero_only",
        "hot_real",
    }


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_never_passes_even_if_bits_match(value):
    tensor = torch.tensor([value])
    assert not finite_bitwise_equal(tensor, tensor.clone())


def test_signed_zero_is_not_bitwise_equal():
    positive, negative = torch.tensor([0.0]), torch.tensor([-0.0])
    assert torch.equal(positive, negative)
    assert not finite_bitwise_equal(positive, negative)


def test_dtype_shape_and_noncontiguous_comparison():
    tensor = torch.arange(8, dtype=torch.bfloat16).reshape(2, 4).t()
    assert finite_bitwise_equal(tensor, tensor.contiguous())
    assert not finite_bitwise_equal(tensor, tensor.float())
    assert not finite_bitwise_equal(tensor, tensor.reshape(-1))


def test_missing_rank_cannot_pass(tmp_path):
    with pytest.raises(AssertionError, match="expected 16"):
        validate_records(tmp_path, 20, 20, 0)


def test_tolerant_or_short_soak_cannot_pass(tmp_path):
    for rank in range(16):
        record = dict(
            rank=rank,
            full_reference=True,
            strict_bits=True,
            reference_expert_solution="torch_npu",
            baseline_pre="composed_gmoe_pre",
            baseline_post="composed_gmoe_post",
            bitwise_changed_replays=20,
            bitwise_fixed_replays=20,
            max_abs_error=0,
            stability_cycles=1,
            stability_elapsed_seconds=10,
            stability_memory_peak_bytes=1024,
            stability_memory_start_bytes=1024,
        )
        (tmp_path / f"int8-rank{rank:02d}.json").write_text(json.dumps(record))
    assert validate_records(tmp_path, 20, 20, 0)["ranks"] == 16
    with pytest.raises(AssertionError):
        validate_records(tmp_path, 20, 20, 60)
    record["strict_bits"] = False
    (tmp_path / "int8-rank15.json").write_text(json.dumps(record))
    with pytest.raises(AssertionError):
        validate_records(tmp_path, 20, 20, 0)
