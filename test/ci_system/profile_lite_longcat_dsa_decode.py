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

"""Capture a stable full-model Lite LongCat DSA decode timeline.

The validated fixture is W8A8, EP16/G8, attention TP8, DCP8, DP2, per-replica
batch 8, and 64K history. Historical KV/SSM is synthetic and MoE routes are
generated deterministically outside graph capture. This is a performance and
execution harness, not a prefill or model-quality test.
"""

import gc
import json
import os
import statistics
import time
from dataclasses import replace
from pathlib import Path

import torch
import torch.distributed as dist
import torch_npu

from tokenspeed.runtime.configs.model_config import ModelConfig
from tokenspeed.runtime.distributed.comm_backend import initialize_comm_backend
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.distributed.process_group_manager import (
    process_group_manager as pg,
)
from tokenspeed.runtime.engine.scheduler_utils import scheduler_cache_geometry_from_pool
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.execution.factory import (
    ModelExecutorConfig,
    create_model_executor,
    create_model_runner,
)
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.registry import create_attn_components
from tokenspeed.runtime.utils import get_available_gpu_memory
from tokenspeed.runtime.utils.server_args import prepare_server_args

ROOT = Path(os.environ["RUN_DIR"]).resolve()
MODE = os.environ["GMOE_PROFILE_MODE"]
assert MODE in ("unfused", "fused")
RANK = int(os.environ["RANK"])
LOCAL = int(os.environ["LOCAL_RANK"])
WORLD = int(os.environ["WORLD_SIZE"])
TP = 8
BS = 8
KV = 65536
CHECKPOINT = os.environ["MODEL"]
GMOE_RUNTIME = os.environ["FLASH_OPS_GMOE_RUNTIME"]
GMOE_RENDEZVOUS = os.environ["GMOE_RENDEZVOUS"]
RANDOM_ROUTE_SEED = 2027

if WORLD != 16:
    raise RuntimeError(f"this profile requires WORLD_SIZE=16, got {WORLD}")


class FixedRandomRouterExchange:
    """Profile-only exchange proxy with routes generated before graph capture."""

    def __init__(self, exchange, weights, ids):
        self._exchange = exchange
        self._weights = weights
        self._ids = ids

    def __getattr__(self, name):
        return getattr(self._exchange, name)

    def exchange_router(
        self, grouped, router, top_k, scaling_factor, renormalize, *, input_scale
    ):
        del router, top_k, scaling_factor, renormalize
        received = self._exchange.exchange(grouped, input_scale=input_scale)
        if received.shape[0] != self._ids.shape[0]:
            raise RuntimeError(
                f"fixed random route rows {self._ids.shape[0]} != received rows "
                f"{received.shape[0]}"
            )
        return received, self._weights, self._ids


def install_fixed_random_routes(runner, config):
    """Generate distinct per-rank/per-layer routes outside NPUGraph capture."""
    layers = runner.model.model.layers
    summaries = []
    for layer_index, layer in enumerate(layers):
        moe = layer.moe
        route_rows = BS * moe.moe_group_size * moe.topology.egp_size
        route_experts = moe.n_routed_experts + moe.zero_expert_num
        top_k = moe.config.moe_topk
        seed = RANDOM_ROUTE_SEED + RANK * len(layers) + layer_index
        generator = torch.Generator(device="cpu").manual_seed(seed)
        # topk over independent random scores gives unique expert IDs per row.
        ids_cpu = (
            torch.rand(
                (route_rows, route_experts), generator=generator, dtype=torch.float32
            )
            .topk(top_k, dim=-1, sorted=False)
            .indices.to(torch.int32)
        )
        weights_cpu = torch.rand(
            (route_rows, top_k), generator=generator, dtype=torch.float32
        )
        weights_cpu = (
            weights_cpu
            / weights_cpu.sum(dim=-1, keepdim=True)
            * float(moe.config.routed_scaling_factor)
        )
        ids = ids_cpu.to(device="npu")
        weights = weights_cpu.to(device="npu")
        proxy = FixedRandomRouterExchange(moe._moe_stage_context.exchange, weights, ids)
        moe._moe_stage_context = replace(moe._moe_stage_context, exchange=proxy)
        summaries.append(
            dict(
                layer=layer_index,
                seed=seed,
                first_ids=ids_cpu[0].tolist(),
                zero_routes=int((ids_cpu >= moe.n_routed_experts).sum().item()),
            )
        )
    return summaries


def log(stage, **kwargs):
    entry = dict(
        time=time.strftime("%Y-%m-%d %H:%M:%S"), rank=RANK, stage=stage, **kwargs
    )
    with (ROOT / f"rank-{RANK}.jsonl").open("a") as stream:
        stream.write(json.dumps(entry) + "\n")
    print(json.dumps(entry), flush=True)


@torch.inference_mode()
def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    torch.npu.set_device(LOCAL)
    torch.manual_seed(2027 + RANK // TP)
    args = prepare_server_args(
        [
            "--model",
            CHECKPOINT,
            "--device",
            "npu",
            "--dtype",
            "bfloat16",
            "--kv-cache-dtype",
            "bfloat16",
            "--world-size",
            "16",
            "--nprocs-per-node",
            "16",
            "--attn-tp-size",
            "8",
            "--decode-context-parallel-size",
            "8",
            "--linear-attn-tp-size",
            "8",
            "--mla-weight-tp-size",
            "8",
            "--dense-tp-size",
            "8",
            "--ep-size",
            "16",
            "--moe-tp-size",
            "1",
            "--oe-table-placement",
            "host",
            "--max-num-seqs",
            "16",
            "--max-model-len",
            "81920",
            "--max-total-tokens",
            "600000",
            "--chunked-prefill-size",
            "4096",
            "--prefix-granularity",
            "128",
            "--gpu-memory-utilization",
            "0.9",
            "--attention-backend",
            "longcat_dsa",
            "--kda-backend",
            "auto",
            "--sampling-backend",
            "greedy",
            "--disable-prefill-graph",
            "--disable-autotune",
            "--disable-pdl",
            "--disable-kvstore",
            "--disable-prefix-caching",
            "--disable-overlap-schedule",
            "--disable-cuda-graph-padding",
            "--cudagraph-capture-sizes",
            "8",
            "--max-cudagraph-capture-size",
            "8",
            "--npu-enable-weight-nz",
            "--disaggregation-mode",
            "null",
            "--grammar-backend",
            "none",
        ]
    )
    args.mapping = Mapping(
        rank=RANK,
        world_size=WORLD,
        nprocs_per_node=WORLD,
        nnodes=1,
        attn_tp_size=8,
        attn_dp_size=2,
        attn_dcp_size=8,
        linear_attn_tp_size=8,
        mla_weight_tp_size=8,
        dense_tp_size=8,
        moe_tp_size=1,
        moe_ep_size=16,
    )
    args.enable_nan_detection = True
    override = dict(gmoe_strategy="gmoe_aware")
    if MODE == "unfused":
        override.update(
            gmoe_pre_solution="composed",
            gmoe_post_solution="composed",
            gmoe_expert_solution="torch_npu",
        )
    else:
        override.update(
            gmoe_pre_solution="flash_npu_router_ffn",
            gmoe_post_solution="flash_npu",
            gmoe_expert_solution="flash_npu_routed_full",
            gmoe_exchange_options=dict(
                rdma_library=GMOE_RUNTIME,
                rendezvous=GMOE_RENDEZVOUS,
                shared_overlap=True,
                router_cores=24,
                shared_cube_cores=8,
                shared_vector_cores=16,
            ),
        )
    mapping = args.mapping
    pg.init_distributed(mapping, backend="hccl", timeout=1800, device_id=None)
    for group in (
        mapping.world_group,
        mapping.attn.tp_group,
        mapping.attn.dp_group,
        mapping.attn.dcp_group,
        mapping.linear_attn.tp_group,
        mapping.mla_weight.tp_group,
        mapping.dense.tp_group,
        mapping.moe.tp_ep_group,
    ):
        pg.init_process_group(group)
    initialize_comm_backend()
    free_memory = get_available_gpu_memory(
        "npu",
        LOCAL,
        distributed=True,
        cpu_group=pg.get_process_group("gloo", mapping.world_group),
    )
    config = ModelConfig(
        CHECKPOINT,
        trust_remote_code=False,
        model_override_args=json.dumps(override),
        dtype="bfloat16",
        context_length=args.max_model_len,
        server_args=args,
    )
    required_config = {
        "num_layers": 48,
        "hidden_size": 4096,
        "num_attention_heads": 64,
        "num_key_value_heads": 64,
        "moe_group_size": 8,
        "n_routed_experts": 384,
        "zero_expert_num": 32,
        "moe_topk": 18,
        "target_topk": 12,
        "index_n_heads": 16,
        "index_head_dim": 128,
        "index_topk": 2048,
    }
    hf_text_config = config.hf_text_config
    mismatches = {
        name: (getattr(hf_text_config, name, None), expected)
        for name, expected in required_config.items()
        if getattr(hf_text_config, name, None) != expected
    }
    if mismatches:
        raise RuntimeError(
            f"checkpoint does not match the validated fixture: {mismatches}"
        )
    log(
        "load_start",
        free_gib=free_memory,
        config=override,
        parallelism=str(mapping),
        checkpoint=CHECKPOINT,
    )
    runner, _ = create_model_runner(args, config, None, LOCAL, RANK)
    torch.npu.synchronize()
    random_route_summaries = install_fixed_random_routes(runner, config)
    first_moe = runner.model.model.layers[0].moe
    log(
        "fixed_random_routes_installed",
        generated_outside_graph=True,
        rank_distinct=True,
        rows=BS * first_moe.moe_group_size * first_moe.topology.egp_size,
        experts=first_moe.n_routed_experts + first_moe.zero_expert_num,
        top_k=first_moe.config.moe_topk,
        routes=random_route_summaries,
    )
    for index, layer in enumerate(runner.model.model.layers):
        if not layer.config.is_kda_layer(index):
            attn = layer.self_attn
            assert attn.num_local_heads == config.num_attention_heads // 8
            assert type(attn).__name__ == "LongCatDSAAttention"
            log(
                "mla_head_tp",
                layer=index,
                tp_size=mapping.mla_weight.tp_size,
                local_heads=attn.num_local_heads,
                q_b_shape=list(attn.q_b_proj.weight.shape),
                kv_b_shape=list(attn.kv_b_proj.weight.shape),
                o_shape=list(attn.o_proj.weight.shape),
                separate_mla_projections=attn.separate_mla_projections,
                q_a_nz_transposed=attn.q_a_proj._weight_nz_transposed,
                q_b_nz_transposed=attn.q_b_proj._weight_nz_transposed,
                kv_a_nz_transposed=attn.kv_a_proj_with_mqa._weight_nz_transposed,
                o_prepared_dtype=str(attn.o_proj.weight.dtype),
            )
        stages = layer.moe._moe_stages
        log(
            "stage_selection",
            layer=index,
            pre=stages.pre.name,
            post=stages.post.name,
            expert_plan=str(layer.moe._moe_plan),
            hidden=config.hidden_size,
            local_experts=layer.moe.experts.num_local_experts,
        )
    log(
        "load_complete",
        layers=len(runner.model.model.layers),
        allocated_gib=torch.npu.memory_allocated() / 2**30,
    )
    backend, pool, _, _, _ = create_attn_components(
        args,
        config,
        LOCAL,
        RANK,
        free_memory,
        args.enable_memory_saver,
        None,
        decode_input_tokens=1,
        overlap_schedule_depth=0,
    )
    geometry = scheduler_cache_geometry_from_pool(pool)
    log(
        "cache_created",
        groups=[str(s) for s in pool.arena.cache_group_specs],
        counts=dict(pool.arena.cache_group_page_counts),
    )
    observed = []
    hooks = []
    for index, layer in enumerate(runner.model.model.layers):

        def observe(module, values, index=index):
            if len(observed) < config.num_hidden_layers:
                observed.append(
                    dict(
                        layer=index,
                        module=type(module).__name__,
                        shapes=[
                            list(x.shape) for x in values if isinstance(x, torch.Tensor)
                        ],
                    )
                )

        hooks.append(layer.moe.register_forward_pre_hook(observe))
    executor = create_model_executor(
        server_args=args,
        config=ModelExecutorConfig.from_server_args(
            args,
            config,
            BS + 1,
            LOCAL,
            RANK,
            geometry.prefix_granularity,
            overlap_schedule_depth=0,
            requires_request_token_history=config.oe_state_provider
            == "runtime-full-history",
        ),
        model_runner=runner,
        attn_backend=backend,
        token_to_kv_pool=pool,
    )
    executor.capture_graphs()
    for hook in hooks:
        hook.remove()
    log(
        "graph_captured",
        graph_keys=[str(k) for k in executor.forward_step.graphs],
        observed_moe=observed,
    )
    inputs = executor.input_buffers
    rows = torch.arange(1, BS + 1, device="npu", dtype=torch.int64)
    inputs.req_pool_indices_buf[:BS].copy_(rows)
    inputs.state_write_req_pool_indices_buf[:BS].copy_(rows)
    inputs.seq_lens_buf[:BS].fill_(KV)
    inputs.positions_buf[:BS].fill_(KV - 1)
    inputs.input_lengths_buf[:BS].fill_(1)
    inputs.input_ids_buf[:BS].copy_(
        torch.arange(BS, device="npu", dtype=torch.int32) + (RANK // TP) * BS + 100
    )
    inputs.prepare_request_token_history_inputs(
        batch_size=BS, num_extends=0, decode_width=1
    )
    state = executor.runtime_states
    state.valid_cache_lengths[1 : BS + 1].fill_(KV - 1)
    history_generator = torch.Generator().manual_seed(3027 + RANK // TP)
    history = torch.randint(
        1,
        config.vocab_size,
        (BS, KV - 1),
        dtype=torch.int32,
        generator=history_generator,
    )
    state.seed_request_token_history(
        req_pool_indices=list(range(1, BS + 1)),
        prefix_lengths=[KV - 1] * BS,
        request_token_ids=history.tolist(),
    )
    tables = {}
    contract = pool.arena.runtime_contract
    next_parent = 1
    allocations = []
    for spec in pool.arena.cache_group_specs:
        width = (
            executor.config.physical_context_len + spec.block_granularity - 1
        ) // spec.block_granularity
        capacity = pool.arena.cache_group_page_counts[spec.group_id]
        packing = contract.group_packing[spec.group_id]
        live_cols = (KV + spec.block_granularity - 1) // spec.block_granularity
        block_count = BS * live_cols if spec.family == "history" else BS
        parents = (block_count + packing - 1) // packing
        assert next_parent + parents - 1 <= contract.num_lcm_blocks, (
            spec.group_id,
            next_parent,
            parents,
            contract.num_lcm_blocks,
        )
        first_block = 1 + (next_parent - 1) * packing
        assert first_block + block_count <= capacity
        block_ids = torch.arange(
            first_block, first_block + block_count, device="npu", dtype=torch.int32
        )
        if spec.family == "history":
            table = torch.zeros((BS, width), dtype=torch.int32, device="npu")
            table[:, :live_cols] = block_ids.reshape(BS, live_cols)
        else:
            table = block_ids[:, None].expand(BS, width).contiguous()
        allocations.append(
            dict(
                group=str(spec.group_id),
                packing=packing,
                first_parent=next_parent,
                parent_count=parents,
                first_block=first_block,
                block_count=block_count,
            )
        )
        next_parent += parents
        tables[str(spec.group_id)] = table
    log(
        "disjoint_cache_allocation",
        groups=allocations,
        used_parents=next_parent - 1,
        total_parents=contract.num_lcm_blocks,
    )
    ctx = ForwardContext(
        attn_backend=backend,
        token_to_kv_pool=pool,
        bs=BS,
        num_extends=0,
        input_num_tokens=BS,
        forward_mode=ForwardMode.DECODE,
        global_num_tokens=[BS] * WORLD,
        global_bs=[BS] * WORLD,
        all_decode_or_idle=True,
        request_token_history=executor._request_token_history_view(BS),
    )
    sampling = executor._build_sampling_info(BS)
    assert executor.forward_step._can_use_graph(BS, ctx)

    def step():
        return executor.forward_step(
            BS,
            ctx,
            sampling,
            extend_with_prefix=False,
            extend_prefix_lens=inputs.extend_prefix_lens_buf[:0],
            extend_prefix_lens_cpu=inputs.extend_prefix_lens_cpu[:0],
            extend_seq_lens=inputs.extend_seq_lens_buf[:0],
            extend_seq_lens_cpu=inputs.extend_seq_lens_cpu[:0],
            seq_lens_cpu=(KV,) * BS,
            block_tables=tables,
        )

    assert len(observed) == config.num_hidden_layers and all(
        entry["module"] == "GroupAwareFlashLocalMoE"
        and entry["shapes"][0] == [BS, config.hidden_size]
        for entry in observed
    ), observed
    log("live_warmup_start")
    # Capture warmups may write recurrent states; start the live fixture clean.
    pool.arena.buffer.zero_()
    executor.nan_guard.reset(BS)
    for iteration in range(20):
        output = step()
        torch.npu.synchronize()
        if executor.nan_guard.flags[:BS].any().item():
            log(
                "nan_detected",
                iteration=iteration,
                flags=executor.nan_guard.flags[:BS].cpu().tolist(),
            )
            raise AssertionError("NaN or invalid token detected before profiling")
    torch.npu.synchronize()
    assert (
        not executor.nan_guard.flags[:BS].any().item()
    ), "NaN or invalid sampled token in full-model decode"
    log(
        "live_warmup_complete",
        output_ids=output[0].cpu().tolist(),
        allocated_gib=torch.npu.memory_allocated() / 2**30,
    )
    elapsed = []
    for _ in range(5):
        dist.barrier()
        start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(
            enable_timing=True
        )
        start.record()
        for _ in range(10):
            step()
        end.record()
        end.synchronize()
        elapsed.append(start.elapsed_time(end) / 10)
    log(
        "profiler_off_timing", per_step_ms=elapsed, median_ms=statistics.median(elapsed)
    )
    dist.barrier()
    profile_dir = str(ROOT / "prof" / "rank-0")

    def profile_steps(prof=None):
        step()
        torch.npu.synchronize()
        if prof is not None:
            prof.step()
        # Allow asynchronous device profiling setup to settle before replay.
        time.sleep(2)
        dist.barrier()
        for iteration in range(5):
            with torch.autograd.profiler.record_function(
                f"FULL_DECODE_{MODE}_EP16_G8_W8A8_T8_KV65536_STEP_{iteration}"
            ):
                step()
            torch.npu.synchronize()
            if iteration == 4:
                # Let device profiling buffers drain before RECORD_AND_SAVE.
                time.sleep(2)
            if prof is not None:
                prof.step()

    if RANK == 0:
        with torch_npu.profiler.profile(
            schedule=torch_npu.profiler.schedule(wait=0, warmup=1, active=5, repeat=1),
            activities=[
                torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU,
            ],
            record_shapes=True,
            with_stack=False,
            profile_memory=False,
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(profile_dir),
            experimental_config=torch_npu.profiler._ExperimentalConfig(
                profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
            ),
        ) as prof:
            profile_steps(prof)
    else:
        profile_steps()
    torch.npu.synchronize()
    log(
        "profile_complete",
        directory=profile_dir if RANK == 0 else None,
        measured_steps=5,
    )
    dist.barrier()
    # Drop captured graphs before collectively releasing the dedicated SHMEM
    # resource, while HCCL/RDMA and the NPU runtime are still alive.
    executor.forward_step.graphs.clear()
    gc.collect()
    torch.npu.synchronize()
    runner.model.model.layers[0].moe.release_moe_resources()
    log("moe_resources_released")
    dist.barrier()
    dist.destroy_process_group()
    log("shutdown_complete")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        log("ERROR", exception=repr(error))
        raise
