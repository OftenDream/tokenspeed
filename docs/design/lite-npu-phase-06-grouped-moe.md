# Lite NPU 阶段 6：Grouped MoE 总体设计

## 1. 目标与边界

本阶段实现 Lite EveryLayer Grouped MoE 的正确性 baseline、Ascend leaf 和 EP8 执行，并通过三组
受控实验选择唯一生产布局。模型固定参数为：

| 参数 | 值 |
| --- | ---: |
| group 数 `G` | 4 |
| 每组 real expert `E_g` | 384 |
| 每组 identity expert `Z_g` | 32 |
| 每组 TopK `K` | 12 |
| hidden `H` | 3072 |
| group hidden `D=H/G` | 768 |
| expert intermediate `I` | 512 |
| routed scale | 6.0 |
| EP | 8 |
| 每 rank real expert | 192 |

本阶段覆盖：

- 四组独立 softmax、selection-only correction bias 和 TopK；
- real/identity expert 分流、shared expert 只累加一次；
- packed expert 权重加载、NPU GMM/SwiGLU/finalize；
- Prefill eager 和 Decode eager/graph 的 EP8 通信；
- A/B/C executor 与 placement 实验、负载和性能准入。

本阶段不实现 OE、完整 decoder、CP/KVP attention、PD cache 传输、服务启动或 GSM8K；这些在 MoE
winner 固定后继续接入。量化、EPLB、冗余 expert、MC2 和自定义端到端 Grouped-MoE kernel 也不在
本阶段范围内。

## 2. 复用审计

### 2.1 直接复用

| 能力 | 复用对象 | 决策 |
| --- | --- | --- |
| EP8 topology | `Mapping.moe.ep_group` 与既有 process-group manager | 不增加通信组抽象 |
| packed expert owner | 通用 `MoELayer` 的 `w13_weight/w2_weight` | 替换当前逐 expert `ModuleDict` |
| checkpoint packing | `build_moe_checkpoint_loader`、`ExpertCheckpointSchema` | 复用 gate/up/down 到 W13/W2 的 loader |
| dense/shared TP8 | `Mapping.dense` 与通用 collective | 不给 routed expert 叠加 TP |
| RMSNorm | 通用 `RMSNorm`/kernel facade | 不实现模型专用 norm |
| HCCL | 既有 `all_gather`、`reduce_scatter`、`all_reduce` | runtime 不直接创建第二套 group |
| Torch oracle | portable precomputed-TopK expert loop | 作为 CPU/meta 和 NPU 对照 |

现有 `MoELayer` 不支持 mixed TP+EP，但目标 routed expert 是 `TP=1,EP=8`，不需要扩展该限制。

### 2.2 必须新增

- 通用 softmax + correction-bias TopK facade；Torch reference 保持 selection bias 不进入输出权重，
  Ascend registration 调用公开 `npu_moe_gating_top_k`；
- Ascend BF16 precomputed-TopK expert leaf，复用 routing、GMM、SwiGLU 和 finalize 原语；
- Lite 薄 wrapper：四组 router、logical ID 映射、identity contribution、shared expert 和外围 projection；
- 测试专用 A/B/C placement planner 与分布式 harness。

Runtime 只调用统一 kernel facade。`torch_npu` 和 NPU weight layout 只存在于
`tokenspeed-kernel-npu`/Ascend registration，不进入 Lite runtime 模块顶层。

### 2.3 明确拒绝

- BF16 expert 激活使用目标 Torch-NPU 已公开注册的
  `torch_npu.mlp_split_swiglu`，按 `expert_counts` 只处理有效 route 前缀；不直接依赖
  `custom::` namespace；
- 不使用 `npu_moe_distribute_dispatch_v2/combine_v2`。目标为 `1536/EP8=192` experts/rank，超过
  A2 常规 MC2 每 rank 24 expert 的已知约束；
- 不把四组 router 合成一次全局 TopK。模型语义要求每个 token 在每组独立选 12 条 route；
- 不让 identity ID 进入 GMM，不为 identity expert 分配权重；
- 不长期保留三个 production 模式。A/B/C 只存在于 board/准入代码，生产只保留 winner。

## 3. 数学与数据流

### 3.1 外围 projection 与分组

输入 `x` 为 `[T,3072]`：

```text
p = x @ W_in^T
n = RMSNorm(p) * 2
u = transpose(view(n, [T,4,768]), 0, 1)   # [4,T,768]
```

Dense TP8 下 `W_in` 按输出维切分。每 rank 先得到 `[T,384]`，沿 hidden 维 AllGather 后才执行完整
RMSNorm 和分组；不能分别归一化八个局部 shard。

### 3.2 四组独立路由

对组 `g`：

```text
logits[g] = u[g] @ W_router[g]^T                  # [T,416]
prob[g]   = softmax(logits[g], dim=-1)            # FP32
ids[g]    = topk(prob[g] + correction_bias[g], 12)
weight[g] = gather(prob[g], ids[g]) * 6.0         # bias 不进入 weight
```

不做 TopK 内重新归一化。非 tie 输入要求 ID 逐元素一致，权重始终为 FP32，进入 BF16 finalize 前才转换。

### 3.3 Real 与 identity expert

组内 ID `e<384` 为 real expert；`384<=e<416` 为 identity expert：

```text
real_id(g,e) = g * 384 + e
identity[g,t] = u[g,t] * sum(weight[g,t,j] for selected identity routes j)
```

real route 输入 expert MLP：

```text
expert(x) = W_down @ (SiLU(W_gate @ x) * (W_up @ x))
```

NPU expert executor 把 identity route ID 改为 `-1` 且对应 routed weight 置零；identity contribution 在
executor 外按上式只加一次。这样 GMM token count、负载统计和通信字节都只计 real route。

### 3.4 输出与 shared expert

四组输出恢复为 `[T,3072]` 后：

```text
routed = W_out @ concat(group_output[0:4])
shared = W_shared_down @ (SiLU(W_shared_gate @ x) * (W_shared_up @ x))
output = routed + shared
```

`W_out` 与 shared expert 继续按 dense TP8 切分并 AllReduce。shared expert 消费原始 `x`，不消费
`proj_input` 或分组后的 `u`，且无论 A/B/C 都只执行、累加一次。

## 4. Packed 权重与加载

### 4.1 Canonical 参数

生产参数使用通用 MoE canonical layout：

| 参数 | 单 rank canonical shape | checkpoint source |
| --- | --- | --- |
| `experts.w13_weight` | `[192,1024,768]` | gate/up 各 `[512,768]` |
| `experts.w2_weight` | `[192,768,512]` | down `[768,512]` |

加载期保持 canonical layout；Ascend post-load 再派生或原地转换为 GMM 消费的 `[E,768,1024]`、
`[E,512,768]`。CPU/meta reference 始终使用 canonical layout。若 Weight-NZ 没有独立 A/B 收益证据，
本阶段允许 ND 权重继续运行，不把 format cast 当作正确性前提。

严格 loader 仍遍历并验证全部 source key/dtype/shape。expert key 交给通用 MoE loader；非本 rank expert
只做 source coverage，不分配目标参数。一个 packed target 接收多个 source shard是合法行为，不能沿用
普通参数的“target 只能写一次”断言。

### 4.2 Placement 公式

令逻辑 real expert 为 `(g,e)`，`0<=g<4,0<=e<384`。

**A：4 个独立 EP8 executor**

```text
rank_A(g,e)  = floor(e / 48)
local_A(g,e) = e mod 48       # 位于第 g 个 executor
```

每 rank 在每组持有 48 个，共 192 个；每组独立 dispatch/GMM/finalize/collective。

**B：一个 flattened-interleaved EP8 executor**

```text
rank_B(g,e)     = floor(e / 48)
local_B(g,e)    = g * 48 + (e mod 48)
physical_B(g,e) = rank_B * 192 + local_B
```

每 rank 仍为每组 48 个，但四组共享一次 executor。router 输出的 logical real ID 需按上述公式 remap，
loader 使用同一公式放置权重。

**C：一个 flattened-group-major EP8 executor**

```text
physical_C(g,e) = g * 384 + e
rank_C(g,e)     = g * 2 + floor(e / 192)
local_C(g,e)    = e mod 192
```

每 rank 只持有一个 group 的 192 个 expert；相邻两个 rank 合成完整一组。这是当前 checkpoint loader 的
自然连续布局，也是理论目标候选，但最终仍由 B/C 实测决定。

## 5. Ascend leaf

### 5.1 TopK

统一 API 输入 `[T,416]` FP16/BF16/FP32 logits、FP32 bias、`K=12`，输出 FP32 weight 和 INT32 ID。
Ascend leaf 固定：

```text
npu_moe_gating_top_k(
  k_group=1, group_count=1, group_select_mode=0,
  renorm=0, norm_type=0, routed_scaling_factor=6.0
)
```

四组调用四次该 leaf；本阶段不增加一个覆盖四组的专用 kernel。

### 5.2 Local expert

每个 executor 的 native 链为：

```text
npu_moe_init_routing_v2
  -> npu_grouped_matmul(W13)
  -> npu_swiglu
  -> npu_grouped_matmul(W2)
  -> npu_moe_finalize_routing
```

`expert_tokens_num_type=1` 与 GMM `group_list_type=1` 配套。eager 路径可以按
`sum(local_expert_counts)` 截断 active rows；graph 路径禁止 `.item()`/D2H，使用固定 capacity、device
valid-row mask 和稳定 workspace。

`init_routing_v2` 的 inactive/padding row 不保证初始化。任何 finite 检查、GMM 或 finalize 都不能读取
未 mask 的尾部。identity、非本地和 padding route 统一不贡献 local expert output。

### 5.3 NPU0 discovery 证据

当前目标环境的公开 TopK、routing、re-routing、GMM、SwiGLU、finalize 和 dispatch/combine API 均存在。
目标 shape discovery 得到：

| case | 结果 |
| --- | --- |
| TopK T1/T2/T32, E416, K12 | ID 与 Torch exact；weight max abs `<=5.96e-8` |
| H768/I512、48 local experts、48 active routes | GMM 链 rel-L2 `6.97e-5` |
| native finalize 对 independent scatter oracle | rel-L2 `2.25e-3`、全部 finite |
| 同一 local chain T4 warm median | `127.52 us` |
| private split-SwiGLU | 未注册；公开 `npu_swiglu` shape/finite 通过 |

这些仅证明 leaf ABI 可实施，不等于 EP8、graph 或 A/B/C 已准入。

## 6. EP8 Prefill 与 Decode

### 6.1 Prefill：AG/RS

Prefill 保持 eager。CP/SP8 每 rank 持有自己的 token rows：

1. 按真实 token split AllGather group input、route ID 和 route weight；
2. 每 rank 只执行自己的 192 个 local expert；
3. local finalize 得到全局 token 顺序的 partial output；
4. ReduceScatter 回原 token split；
5. identity 与 shared 分支按各自正确的 replicated/sharded边界合并。

A 执行四次 group executor/collective；B/C 执行一次 flattened executor/collective。split 不能假设八个
rank 等长，使用既有 token collective 的 explicit split metadata。

### 6.2 Decode：local expert + AllReduce

Decode 的 KVP8 ranks 消费相同当前 token rows和路由：

1. 每 rank 固定 shape routing 到本地 192 experts；
2. local GMM/finalize 产生 partial `[BS,group_hidden]` 或 `[4*BS,group_hidden]`；
3. AllReduce 得到完整 routed output；
4. BS1/BS2 graph 复用固定 workspace 和稳定地址。

该路径没有动态 AllToAll split，也不受 MC2 24 experts/rank 限制。PA2A 仅作为 eager 诊断对照，不是
graph production 路径。

所有 rank 即使 local active route 为 0，也必须执行相同 collective 序列；local tensor 为零而不是提前
return。A/B/C 的 collective 次数必须被 harness 记录，不能只从源码推断。

## 7. A/B/C 实验

### 7.1 固定变量

三组实验使用相同 checkpoint layer、输入、router result、BF16 math、EP8 group、每 rank 192 real
experts、identity/shared contribution 和数值阈值。只改变 executor 边界或 real expert physical ID。

### 7.2 实验顺序

1. A 对单 rank dense oracle，证明当前四 executor 组合正确；
2. B 对 A，隔离“4 次执行合为 1 次”的收益；
3. C 对 B，隔离 group-to-rank placement 的负载/通信影响；
4. winner 才接入 production model，A/B/C harness 保留为离线准入测试。

### 7.3 观测项

- route ID、weight、每 token 四组各 12 条 route；
- 每 rank/group real-route 数、pair 聚合数、`max/mean`、变异系数、空 expert/rank；
- identity route 数与权重、GMM active rows；
- TopK、dispatch、GMM1、SwiGLU、GMM2、finalize、collective 的次数和 NPU Event 时间；
- collective bytes、workspace/HBM peak；
- P T32/T128/T1024，D eager/graph BS1/BS2 的稳态 latency。

### 7.4 准入门槛

- 非 tie route ID exact；weight `atol=1e-6,rtol=1e-5`；
- native routed output 对 BF16 Torch oracle `atol=2e-2,rtol=2e-2`，rel-L2 `<=5e-3`；
- 完整 Grouped MoE output rel-L2 `<=1e-2`，所有中间值和输出 finite；
- identity-only、real-only、mixed、duplicate route、boundary ID、empty local rank 全通过；
- B 必须把四次 expert executor/collective 合为一次，稳态 P/D 均不慢于 A，HBM 不增加超过 5%；
- C 的 pair 级 real-route 总数满足组级约束，且相对 B 的稳态 P/D 回退不超过 3%、HBM 不增加；
- Decode BS1/BS2 capture/replay 更新输入和 route 后输出同步变化，无额外 specialization。

若 B 不通过，保留 A；若 B 通过而 C 不通过，生产保留 B；只有 C 全部门槛通过才保留 C。

## 8. 测试矩阵

### 8.1 CPU/meta

- 四组 reshape/restore、router bmm 和 selection-only bias；
- logical/physical/local ID 的 A/B/C 正反映射；
- packed loader 的 gate/up/down 顺序、EP owner、未拥有 expert skip；
- real/identity/shared 分支独立 oracle；
- T0/T1/T4、all-identity、all-real、重复 ID、tie 排除。

### 8.2 NPU0

- E416/K12 fused TopK T1/T2/T32/T1024；
- 48/192 local expert、H768/I512、0/部分/全部 active route；
- ND/NZ weight、active-row mask、finite、数值和 leaf A/B；
- BS1/BS2 NPUGraph capture/update/replay。

### 8.3 NPU8

- A/B/C synthetic deterministic weights；
- checkpoint 单层真实 weight/router replay；
- P T32/T128/T1024 不等长 token split；
- D eager/graph BS1/BS2；
- 每 rank route/GMM/collective/HBM 日志完整，所有进程 collective 顺序一致。

## 9. 实施与提交顺序

每个子阶段都按“设计文档 -> 实现与测试 -> exact-source 验证记录”独立 signed-off 提交并立即推送：

1. **6A Torch 语义与 packed owner**：portable oracle、packed loader、Lite wrapper 的 N=1 路径；
2. **6B Ascend local leaf**：fused TopK、GMM/SwiGLU/finalize、weight post-load 和 NPU0 graph；
3. **6C EP8 A/B/C board**：三种 layout、P/D collective、负载/性能/HBM，选择 winner；
4. **6D production 接线**：只接入 winner、dense/shared TP8、P/D mode、最终 exact-source 回归。

失败候选不进入 production code；其负结果、shape、错误阶段和回退决策写入对应验证记录。

## 10. 完成条件

- winner 的 checkpoint packed ownership 在八个 rank 上无重无漏；
- 四组 TopK、real/identity/shared 数学与统一 oracle 对齐；
- NPU0 leaf、NPU8 P eager、D eager/graph 全部通过并有性能/HBM证据；
- production 只剩一个 placement/executor 和 portable Torch fallback；
- 不支持的 MC2/EPLB/量化明确 fail closed；
- local、tracking 与远端提交 SHA 一致，工作树只保留本地 planning artifacts。
