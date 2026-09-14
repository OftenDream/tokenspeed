# Lite NPU 阶段 10A：Decode Graph BS1/BS2

## 1. 目标与边界

本阶段在已准入的 8P8D bounded eager 服务上，仅将 Decode role 切换为固定
BS1/BS2 的 NPU graph：

- Prefill 仍为 eager，关闭 Prefill graph 和 overlap；
- Decode capture 且只 capture BS1 与 BS2；
- Decode overlap 本阶段仍关闭，留到独立阶段 10B；
- 不修改 Lite 数学、checkpoint ownership、PD wire、cache recipe、Grouped MoE 布局或
  采样语义；
- 不增加更大 graph batch，因为当前 launcher 只准入 `max_num_seqs <= 2`。

阶段 10A 只回答“Decode graph 是否与 eager 等价”，不用 overlap 的时序变化干扰
根因定位，也不提前宣称最终性能或容量准入。

## 2. 复用边界

仓库已有的 `CudaGraphWrapper` 是唯一 graph runner。NPU 路径使用 device module 提供的
`NPUGraph` 和 `graph(..., auto_dispatch_capture=True)`，因此不新增 Lite 专用 runner、
`torch.compile` 分支或第二套 graph cache。

现有契约已覆盖本次需要的动态数据：

| 数据 | capture 后的 publication 方式 | 要求 |
| --- | --- | --- |
| request/sequence metadata | executor 在 replay 前原地刷新 input buffer | 地址不变 |
| KDA group state page | `HybridLinearAttnBackend` 按 BS 持有固定 `state_in/state_out` buffer | 只原地写 page ID，padding 写 null slot |
| MLA latent page/length | `TokenSpeedMLABackend` 原地刷新 page table、write location 和 sequence length | 不重绑 metadata tensor |
| OE host lookup | model forward 前在 graph 外解析 token，写入固定 device staging | graph 内只读 staging |
| greedy sampling | 已有 sampling backend 只有 default graph variant | 每个 BS 只 capture 一次 |

NPU `graph.update` 所需的 sequence-length CPU 参数更新发生在 replay 之前，不是被 capture
的模型体。本阶段不将这个已有 NPU graph ABI 误判为模型内动态 host 分支。

## 3. Launcher 策略

当前 launcher 将 eager/overlap 开关放在 P/D 共享参数中。实现时只拆分这组 role
policy，其余拓扑、端口、容量和 PD 参数继续共享。

Prefill 命令保留：

```text
--enforce-eager
--disable-prefill-graph
--disable-pdl
--disable-overlap-schedule
```

Decode 命令使用：

```text
--cudagraph-capture-sizes 1 2
--max-cudagraph-capture-size 2
--disable-prefill-graph
--disable-pdl
--disable-overlap-schedule
```

Decode 不得出现 `--enforce-eager`。显式的 size list 与 cap 共同保证每个 Decode rank
只建立 default-variant BS1/BS2 两个 graph；不依赖通用默认 bucket 列表。

## 4. Graph 安全契约

capture 与 replay 必须保持：

1. 模型体内无 `.item()`、`.tolist()`、tensor 值驱动的 Python 分支或新 tensor 地址；
2. KDA 三个 state group 的 input/output page-index buffer 在多次 replay 间 `data_ptr` 不变；
3. MLA page table、sequence-length 和 latent write-location buffer 在多次 replay 间
   `data_ptr` 不变；
4. OE staging 地址不变，BS1 使用 BS1 graph，BS2 使用 BS2 graph；padding row 不得
   写 live request 的 OE/KDA/MLA state；
5. graph capture warmup 不得污染 request slot、null page、KDA recurrent state 或 OE context；
6. 服务 ready 后不得再出现 capture，不允许未记录的 eager fallback。

Zero-expert routing keeps the top-k ID and weight buffers in place with
`masked_fill_`. Boolean-index assignment lowers to `NonZero` on NPU and
requires a host synchronization that capture rejects. The identity/copy
contribution uses the original zero-expert weights; routed slots then carry
ID 0 and weight 0. The changed-route replay regression is
`test/runtime/test_lite_zero_expert_graph.py`.

同步 finite probe 包含 host scalar read，不放入 graph 体也不用它证明 replay 数值。graph
正确性由组件 pointer/state 对齐、executor 的 post-replay NaN guard 和真实服务输出共同
证明；如这些证据不足以覆盖某个实际失败，只在 graph 外增加最小的 test-only
snapshot，不把诊断放进 production graph。

## 5. 实现与测试拆分

第一个实现提交只修改 launcher 和它的 dry-run test：

- P 命令继续明确 eager/overlap-off；
- D 命令明确 graph BS1/BS2/overlap-off；
- 负例防止 D 重新出现 `--enforce-eager`，或 P 误获得 Decode capture size。

实现前后都运行下列累计 focused tests：

- launcher command 构造和所有已有端口/拓扑负例；
- Lite graph state-index 原地刷新；
- Lite MLA Decode graph live-length 刷新；
- Lite KDA output epilogue graph replay；
- Lite OE fixed staging 多次更新；
- Hybrid cache/OE/MLA/KDA 累计回归；
- `pre-commit run --all-files`。

若 launcher-only 版本在上板 capture 或首次 replay 失败，必须以八个 Decode rank 的共同
最早错误为边界继续拆分。只修改所有 caller 共享的 metadata/kernel 根因，不在
launcher 中通过扩大 batch、关闭 Lite 子模块或默认回退 eager 使请求假通过。根修复与测试
将使用独立提交，不与 launcher policy 提交混合。

## 6. Exact NPU16 验收

验证使用 exact tracked source 和单节点 16 张 Ascend NPU，不改动系统环境或基础依赖。

### 6.1 Capture 和 replay 证据

- P 无 Decode graph capture 日志；
- D 启动只报告 capture sizes `[1, 2]`，greedy 只产生 default variant；
- 通过现有 timing 记录确认 sequential BS1 为 `graph=true,padded_bs=1`，steady/late
  BS2 为 `graph=true,padded_bs=2`；
- readiness 后 capture 记录不增加，没有 eager fallback。

### 6.2 数值与 state 对齐

使用同一 exact source 的 eager reference 和 graph candidate，固定 greedy 请求、slot 分配和
admission 顺序，分别覆盖：

- sequential BS1、steady BS2、真实 late-admission BS2 和 slot reuse；
- token IDs、completion length、finish reason 和可用的 token logprob；
- post-replay NaN guard 为零，日志无独立词 NaN/Inf、OOM、HCCL 或 transfer 错误；
- focused NPU graph tests 对齐 KDA output/state、MLA latent/output-gate/page 和 OE staging；
- 若 token 或 logprob 在低 margin 位置受已知 BF16 collective 非 bitwise 影响，先比较同一
  服务内的 eager/graph 组件 oracle 和逐元素 state，不修改精度门槛掩盖。

Phase 10A 不以“请求能返回”代替 state 门槛；任一 live KDA/MLA/OE slot 污染、
BS2 row 串扰、graph 外回退或非有限值都是准入失败。

### 6.3 资源与性能记录

- 记录 graph 启动额外 HBM、capture 时间和 BS1/BS2 稳态 latency/throughput；
- 性能在本阶段只是记录，不为了更好数字改动数值路径；
- 服务结束后按精确 PGID 清理，确认 listener、worker marker 和 NPU context 为零。

## 7. 提交与回退

阶段边界为：

1. 本设计文档单独 signed-off 提交并推送；
2. launcher policy 与 dry-run tests 独立 signed-off 提交并推送；
3. 若需共享 graph 根修复，每个真实根因独立设计/实现/验证；
4. exact NPU16 结果作为独立验证记录提交并推送；
5. 只有阶段 10A 通过后才开始阶段 10B overlap。

回退只需让 Decode 恢复 `--enforce-eager`，不改变 P、checkpoint、cache 或模型实现。
不保留“部分 rank graph、部分 rank eager”或“BS1 graph、BS2 默认 eager”的半启用状态。
