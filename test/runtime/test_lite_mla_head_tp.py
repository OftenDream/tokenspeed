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

"""Actual Lite geometry, full-checkpoint loaders and eight-rank head TP.

Run with torchrun --nproc-per-node=8 and --tp-size 2, 4 or 8. Both paths use
identical BF16 inputs, weights, pages, gate and optional Weight-NZ/prolog.
"""

import argparse
import json
import os
from pathlib import Path
from test.runtime.test_lite_mla_packed_prefill import Backend, Context, Pool
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch_npu

from tokenspeed.runtime.distributed.process_group_manager import process_group_manager
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.layernorm import RMSNorm
from tokenspeed.runtime.models.flash_kda import FLASHLocalDecoderLayer
from tokenspeed.runtime.models.flash_local_attention import (
    SeparateProjectionKimiLinearMLAAttention,
)
from tokenspeed.runtime.utils.env import global_server_args_dict


def layer_config():
    return SimpleNamespace(
        hidden_size=3072,
        num_attention_heads=32,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        rms_norm_eps=1e-5,
        mla_scale_q_lora=True,
        mla_scale_kv_lora=True,
        mla_use_output_gate=True,
        max_position_embeddings=81920,
    )


def make_layer(tp_size, rank):
    group_start = rank // tp_size * tp_size
    group = tuple(range(group_start, group_start + tp_size))
    component = SimpleNamespace(tp_size=tp_size, tp_rank=rank % tp_size, tp_group=group)
    mapping = SimpleNamespace(
        mla_weight=component,
        attn=SimpleNamespace(tp_size=8, tp_rank=rank, tp_group=tuple(range(8))),
    )
    return SeparateProjectionKimiLinearMLAAttention(
        layer_config(), mapping, layer_id=0, prefix=""
    ).to(device="npu", dtype=torch.bfloat16)


def error_stats(actual, expected):
    delta = actual.float() - expected.float()
    relative_l2 = (delta.norm() / expected.float().norm()).item()
    return dict(
        relative_l2=relative_l2,
        max_abs=delta.abs().max().item(),
        pointwise_exceedances=(delta.abs() > 0.004 + 0.03 * expected.float().abs())
        .sum()
        .item(),
    )


def compare(actual, expected, label):
    # Existing full-sequence prolog gate; no tolerance relaxation for TP.
    stats = error_stats(actual, expected)
    torch.testing.assert_close(
        actual, expected, rtol=0.03, atol=0.004, msg=f"{label}: {stats}"
    )
    assert stats["relative_l2"] < 0.003, (label, stats)
    return stats


@torch.inference_mode()
def run_head_tp(tp_size):
    rank = dist.get_rank()
    assert dist.get_world_size() == 8
    assert tp_size in (2, 4, 8)
    local_rank, local_heads = rank % tp_size, 32 // tp_size
    torch.manual_seed(7303)
    torch.npu.manual_seed_all(7303)
    torch.npu.config.allow_internal_format = True
    global_server_args_dict.update(
        attention_backend="mla",
        npu_enable_weight_nz=True,
        disaggregation_mode="prefill",
    )
    baseline, candidate = make_layer(1, rank), make_layer(tp_size, rank)
    for name, p in baseline.named_parameters():
        p.normal_(0, 0.01)
        if "layernorm" in name:
            p.fill_(1)
        target = dict(candidate.named_parameters())[name]
        loader = getattr(target, "weight_loader", None)
        if loader is None:
            target.copy_(p)
        else:
            loader(target, p)
        axis = (
            0
            if any(key in name for key in ("q_b_proj", "kv_b_proj", "g_proj"))
            else 1 if "o_proj" in name else None
        )
        expected = p if axis is None else p.chunk(tp_size, dim=axis)[local_rank]
        torch.testing.assert_close(target, expected, rtol=0, atol=0)
    # Verify separate projections before NZ preparation as well.
    x = torch.randn(65, 3072, device="npu", dtype=torch.bfloat16) * 0.1
    for name in ("q_a_proj", "kv_a_proj_with_mqa", "g_proj"):
        full = getattr(baseline, name)(x)[0]
        local = getattr(candidate, name)(x)[0]
        expected = full.chunk(tp_size, dim=1)[local_rank] if name == "g_proj" else full
        compare(local, expected, name)
    for layer in (baseline, candidate):
        layer.process_weights_after_loading(None)
        for name in ("q_a_proj", "kv_a_proj_with_mqa", "q_b_proj", "o_proj"):
            getattr(layer, name).process_weights_after_loading(None)
    assert candidate.q_b_proj.weight.shape == (1536, local_heads * 192)
    assert candidate.w_kc.shape == (local_heads, 128, 512)
    assert candidate.w_vc.shape == (local_heads, 512, 128)
    head_slice = slice(local_rank * local_heads, (local_rank + 1) * local_heads)
    compare(candidate.w_kc, baseline.w_kc[head_slice], "w_kc")
    compare(candidate.w_vc, baseline.w_vc[head_slice], "w_vc")
    norm = RMSNorm(3072, eps=1e-5).to(device="npu", dtype=torch.bfloat16)
    norm.weight.fill_(1)
    reduced = []

    def record_norm(hidden, residual):
        reduced.append(hidden.clone())
        return norm(hidden, residual)

    post = SimpleNamespace(self_attn=candidate, post_attention_layernorm=record_norm)
    hits = []
    original = candidate._project_q_with_mla_prolog

    def tracked(*args):
        result = original(*args)
        hits.append(result is not None)
        return result

    candidate._project_q_with_mla_prolog = tracked
    records = []
    for case, counts in (("boundary", (63, 2, 65)), ("64k", (4096,) * 16)):
        pools = [Pool(64, 81920 // 64), Pool(64, 81920 // 64)]
        table = torch.randperm(81920 // 64, dtype=torch.int32, device="npu").view(1, -1)
        backends = [Backend(64, table), Backend(64, table)]
        for backend, heads in zip(backends, (32, local_heads)):
            backend.num_local_heads = heads
            backend.step_counter = None
        prefix = 0
        for index, count in enumerate(counts):
            x = torch.randn(count, 3072, device="npu", dtype=torch.bfloat16) * 0.1
            residual = torch.randn_like(x) * 0.01
            positions = torch.arange(prefix, prefix + count, device="npu")
            outputs = []
            snapshots = [p.cache.clone() for p in pools]
            for layer, pool, backend in zip((baseline, candidate), pools, backends):
                backend.set_chunk(prefix, count)
                ctx = Context(ForwardMode.EXTEND, backend, pool, 1, 1, count)
                outputs.append(layer(positions, x, ctx, None, None, None))
            assert outputs[0].dtype == torch.bfloat16
            assert outputs[1].dtype == torch.float32
            expected_norm, expected_residual = norm(outputs[0], residual.clone())
            actual_norm, actual_residual = FLASHLocalDecoderLayer._post_attn(
                post, outputs[1], residual.clone(), ctx
            )
            torch.npu.synchronize()
            reduced_output = reduced.pop()
            # Validate norm against the FP64 formula at its actual input. The
            # fixed output tolerance above is in attention-output units; near
            # residual cancellation RMSNorm can amplify a valid BF16 difference.
            # Retain that cross-TP pointwise discrepancy explicitly in results.
            norm_input = reduced_output.cpu().double() + residual.cpu().double()
            reference_norm = (
                norm_input
                * torch.rsqrt(norm_input.square().mean(-1, keepdim=True) + 1e-5)
            ).to(torch.bfloat16)
            norm_tp1 = error_stats(actual_norm, expected_norm)
            assert norm_tp1["relative_l2"] < 0.003, norm_tp1
            row = dict(
                rank=rank,
                case=case,
                chunk=index,
                prefix=prefix,
                count=count,
                heads=local_heads,
                tp_size=tp_size,
                gate=True,
                weight_nz=True,
                output=compare(reduced_output, outputs[0], "reduced output"),
                residual=compare(actual_residual, expected_residual, "residual"),
                norm=compare(actual_norm.cpu(), reference_norm, "norm FP64 reference"),
                norm_tp1=norm_tp1,
            )
            torch.testing.assert_close(
                pools[1].cache, pools[0].cache, rtol=0.02, atol=0.004
            )
            row["cache"] = compare(pools[1].cache, pools[0].cache, "cache")
            for pool, backend, before in zip(pools, backends, snapshots):
                untouched = torch.ones(
                    pool.cache.numel() // 576, dtype=torch.bool, device="npu"
                )
                untouched[backend.locations.long()] = False
                assert torch.equal(
                    pool.cache.view(-1, 576)[untouched], before.view(-1, 576)[untouched]
                )
                assert torch.all(pool.backing[[0, 1, -2, -1]] == 0.125).item()
            records.append(row)
            print(json.dumps(row), flush=True)
            prefix += count
    assert len(hits) == 17 and all(hits), hits
    # Keep the original cross-TP norm pointwise gate. Collect all chunks before
    # failing so the independent FP64 diagnostic cannot mask a TP discrepancy.
    passed = all(row["norm_tp1"]["pointwise_exceedances"] == 0 for row in records)
    result = dict(rank=rank, passed=passed, prolog_hits=len(hits), records=records)
    output = Path(os.environ["LITE_TEST_OUTPUT"])
    output.mkdir(parents=True, exist_ok=True)
    (output / f"head-tp-rank{rank}.json").write_text(json.dumps(result, indent=2))
    dist.barrier()
    assert passed, "Cross-TP normalized output exceeds the original pointwise gate"
    return result


@pytest.mark.skipif(
    not dist.is_initialized() or dist.get_world_size() != 8,
    reason="requires an allocated eight-rank torchrun job",
)
def test_head_tp():
    run_head_tp(8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp-size", type=int, choices=(2, 4, 8), required=True)
    args = parser.parse_args()
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(backend="hccl")
    process_group_manager.register_process_group(
        "hccl", tuple(range(8)), dist.group.WORLD
    )
    if args.tp_size < 8:
        for start in range(0, 8, args.tp_size):
            ranks = tuple(range(start, start + args.tp_size))
            group = dist.new_group(ranks=list(ranks), backend="hccl")
            if dist.get_rank() in ranks:
                process_group_manager.register_process_group("hccl", ranks, group)
    try:
        run_head_tp(args.tp_size)
    finally:
        dist.destroy_process_group()
