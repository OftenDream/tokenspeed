# Lite LongCat DSA：16 卡启动与 Decode Timeline

本文给出当前已验证的整网 profiling 路径。入口
`profile_lite_longcat_dsa_decode.py` 直接构造模型执行器、捕获 NPUGraph，
运行稳定的合成 64K Decode workload，并只采集 rank 0 的完整 CPU + NPU
timeline。它不会启动 HTTP 服务，也不会执行真实 Prefill。

## 1. 已验证配置

| 项目 | 配置 |
| --- | --- |
| 设备 | 单机 16 张 Ascend 910B |
| 模型 | Lite LongCat DSA，48 层，hidden size 4096 |
| 权重 | W8A8 MoE；Attention/KDA/OE 保持 checkpoint 定义 |
| 并行 | Attention TP8、DCP8、DP2；KDA TP8；MoE EP16、TP1 |
| GMoE | group size 8，384 routed experts + 32 zero experts |
| DSA | 64 heads，16 index heads，index TopK 2048，CLI=1 |
| Decode batch | 每个 DP replica 8 条，全实例 16 条 |
| KV 长度 | 每条 65,536 tokens |
| 图 | 原生 NPUGraph，capture size 8 |
| Timeline | rank 0，5 个 Decode step，Level1 + PipeUtilization，记录 shape |

这个入口使用合成历史 KV/SSM，并在图外为每个 rank、每一层生成不同的固定
随机 MoE route。这样 dummy checkpoint 不会因为 router 权重退化而得到失真的
专家负载。该结果只用于执行和性能分析，不能用于模型质量或 Prefill 精度结论。

## 2. 准备环境

使用系统 Python，不创建虚拟环境。先准备三个路径：

```bash
export TS_ROOT=/path/to/tokenspeed
export MODEL=/path/to/lite-dsa-w8a8-format-hf
export FLASH_OPS_BUNDLE=/path/to/matching-flash-ops-bundle
```

`MODEL` 至少应满足下面的关键配置：

```text
hidden_size=4096
num_attention_heads=64
num_key_value_heads=64
moe_group_size=8
n_routed_experts=384
zero_expert_num=32
moe_topk=18
target_topk=12
index_n_heads=16
index_head_dim=128
index_topk=2048
```

算子 bundle 必须来自与当前 TokenSpeed 匹配的 flash-npu-kernel revision，且同时
包含 custom OPP、Python/C++ binding、`libgmoe_comm.so` 和它固定版本的 SHMEM
依赖。不要混用旧 wheel、另一个 OPP 目录和新通信库。

```bash
source /usr/local/Ascend/cann/set_env.sh
source "$FLASH_OPS_BUNDLE/env.sh"

export PYTHONPATH="$TS_ROOT/python:$TS_ROOT/tokenspeed-kernel/python:$TS_ROOT/tokenspeed-kernel-npu/python${PYTHONPATH:+:$PYTHONPATH}"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export HCCL_CONNECT_TIMEOUT=600
export HCCL_RDMA_TIMEOUT=20
export OMP_NUM_THREADS=1
```

`env.sh` 应设置 `FLASH_OPS_GMOE_RUNTIME`。启动前做一次 fail-fast 检查：

```bash
cd "$TS_ROOT"
python - <<'PY'
import os

import torch
import torch_npu
import flash_ops

assert torch.npu.is_available()
assert torch.npu.device_count() >= 16
assert os.path.isfile(os.environ["FLASH_OPS_GMOE_RUNTIME"])

required = (
    "gmoe_dispatch",
    "gmoe_combine",
    "fused_init_routing_mm13_swiglu",
    "fused_mm2_fin_routing",
    "npu_mla_prolog_v3",
    "npu_sparse_flash_attention_decode",
)
missing = [name for name in required if not hasattr(torch.ops.custom, name)]
assert not missing, f"missing custom operators: {missing}"
print("environment OK")
PY
```

## 3. 抓取融合路径的 Decode Timeline

为本次运行创建全新的输出目录，并为 GMoE SHMEM 选择未占用的 rendezvous
端口。不要让两个 16 卡作业复用同一个端口。

```bash
cd "$TS_ROOT"

export RUN_DIR=/path/to/output/lite-dsa-decode-fused
export GMOE_PROFILE_MODE=fused
export GMOE_RENDEZVOUS=tcp://127.0.0.1:25816

test ! -e "$RUN_DIR"
mkdir -p "$RUN_DIR"

torchrun \
  --standalone \
  --nproc_per_node=16 \
  test/ci_system/profile_lite_longcat_dsa_decode.py \
  2>&1 | tee "$RUN_DIR/torchrun.log"
```

`fused` 模式显式选择：

```text
gmoe_pre_solution=flash_npu_router_ffn
gmoe_post_solution=flash_npu
gmoe_expert_solution=flash_npu_routed_full
shared_overlap=true
router_cores=24
shared_cube_cores=8
shared_vector_cores=16
```

脚本执行顺序为：加载模型 → 安装图外固定随机 route → 创建 64K synthetic
cache → 捕获 BS8 NPUGraph → 20 次 warmup → profiler-off 计时 → 采集 5 步
rank0 timeline → 清图 → 集体释放 GMoE SHMEM → 销毁进程组。

成功时，所有 16 个 `rank-*.jsonl` 的最后阶段都应是
`shutdown_complete`：

```bash
grep -h '"stage": "shutdown_complete"' "$RUN_DIR"/rank-*.jsonl | wc -l
grep -h '"stage": "profiler_off_timing"' "$RUN_DIR"/rank-0.jsonl
grep -h '"stage": "profile_complete"' "$RUN_DIR"/rank-0.jsonl
```

第一条命令应输出 `16`。不要使用 `kill -9` 正常结束作业；GMoE runtime 必须在
HCCL 和 NPU runtime 仍存活时完成集体释放。

## 4. 找到并打开 Timeline

Profiler 仅在 rank 0 创建原始数据和解析结果：

```bash
export PROFILE_OUTPUT=$(find "$RUN_DIR/prof/rank-0" \
  -type d -name ASCEND_PROFILER_OUTPUT -print -quit)

test -n "$PROFILE_OUTPUT"
test -s "$PROFILE_OUTPUT/trace_view.json"
test -s "$PROFILE_OUTPUT/operator_details.csv"
test -s "$PROFILE_OUTPUT/kernel_details.csv"

echo "$PROFILE_OUTPUT/trace_view.json"
```

用 MindStudio Insight 打开 `trace_view.json`；也可以把它拖入 Perfetto Web UI。
`operator_details.csv` 用于查看算子输入 shape，`kernel_details.csv` 用于统计设备
kernel 时间，`communication.json` 用于检查 HCCL collective。

timeline 中每一步都有如下标记：

```text
FULL_DECODE_fused_EP16_G8_W8A8_T8_KV65536_STEP_0
...
FULL_DECODE_fused_EP16_G8_W8A8_T8_KV65536_STEP_4
```

分析稳态时优先看后四步；第一步可能包含 profiler 激活后的冷启动扰动。

## 5. 抓取未融合对照

环境、checkpoint 和 16 张卡保持不变，只切换输出目录和模式：

```bash
cd "$TS_ROOT"

export RUN_DIR=/path/to/output/lite-dsa-decode-unfused
export GMOE_PROFILE_MODE=unfused
export GMOE_RENDEZVOUS=tcp://127.0.0.1:25817

test ! -e "$RUN_DIR"
mkdir -p "$RUN_DIR"

torchrun \
  --standalone \
  --nproc_per_node=16 \
  test/ci_system/profile_lite_longcat_dsa_decode.py \
  2>&1 | tee "$RUN_DIR/torchrun.log"
```

未融合模式使用 `composed` pre/post 和 `torch_npu` experts。比较 A/B 时必须使用
同一个 checkpoint、可见卡顺序、CANN、算子 bundle 和随机种子，且不要把
profiler-on 时间当作 TPOT。

## 6. 常见问题

- **缺少 `gmoe_dispatch.router` 或 DSA 算子**：Python schema、binding 与 OPP
  不是同一 revision。重新部署完整 bundle，重新 source `env.sh`。
- **16 rank 在初始化阶段卡住**：检查可见设备确实为 16 张卡、没有残留进程，
  并为 `GMOE_RENDEZVOUS` 换一个空闲端口。
- **有 timeline 但没有 shape**：确认运行的是本脚本；它固定设置
  `record_shapes=True`。不要只保留 profiler 原始目录而丢掉解析输出。
- **没有 `profile_complete`**：先查看所有 `rank-*.jsonl` 的 `ERROR`，再查看
  `torchrun.log` 中最早出现异常的 rank；不要只看最后的 elastic 汇总错误。
- **退出时 SHMEM double free/崩溃**：不要重复加载通信 binding，不要并发复用同一
  GMoE runtime，也不要在图仍存活时销毁进程组。脚本已经按正确顺序释放资源。
- **想改 batch 或 KV 长度**：当前脚本刻意固定 BS8/KV64K。修改时必须同步
  capture size、cache 容量、请求历史和 timeline 标记，不能只改一个常量后宣称
  配置已验证。
