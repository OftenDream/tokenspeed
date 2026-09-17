# Lite NPU 阶段 8C：Decode 官方 Dispatch/Combine 准入设计

> Follow-up: the repository also retains an experimental V2 adapter for
> the global-expert (scheme 2) layout.  It gives every logical expert a global
> ID, maps identity experts to V2 copy experts, and internally chunks expert
> groups to respect the A2 limit of 1024 real experts per invocation.  Direct
> A2 validation accepted the small shape, rejected 3072 real experts during
> tiling, and did not complete the larger 768-expert full-mesh case.  Therefore
> this adapter is preserved as experimental code and is not the production
> decode path.  The next production candidate is the group-first (scheme 1)
> layout: an inter-group AllToAll followed by an EP-local MoE invocation.

## 1. 目标与结论边界

Lite Decode 的当前正确性基线是四个 Grouped MoE local leaf 加一次 EP8
AllReduce。它没有执行 token dispatch/combine：八个 Decode rank 使用相同的
current-token 输入和路由，各 rank 只计算本地 expert partial，最后归约完整 routed
output。

阶段 8C 增加一个独立优化候选：使用华为公开的非 V2 接口
`torch_npu.npu_moe_distribute_dispatch` 和
`torch_npu.npu_moe_distribute_combine`，将 token 重排、EP AllToAllV、返回通信、
route 权重乘加和原 token 顺序恢复交给 CANN。

本阶段明确不使用
`npu_moe_distribute_dispatch_v2/npu_moe_distribute_combine_v2`。此前 FluentLLM
中的 V2 调用和失败 board 只保留为历史对照，不作为 TokenSpeed 的官方 production
候选，也不能用于证明本阶段非 V2 接口可用。

本设计只覆盖 Decode Grouped MoE 通信，不改变 Prefill AG/RS、模型 expert placement、
P/D 拓扑、KDA/MLA、OE 或最后 TODO 中的分布式 KV Cache。

## 2. 来源与目标环境证据

生产候选的来源是 CANN/Torch-NPU，而不是 vLLM-Ascend：

- 华为公开的
  [`npu_moe_distribute_dispatch`](https://www.hiascend.com/document/detail/zh/Pytorch/710/apiref/torchnpuCustomsapi/context/torch_npu-npu_moe_distribute_dispatch.md)
  负责按 expert ID 做 EP AllToAllV，并生成 Combine 所需的 opaque metadata；
- 华为公开的
  [`npu_moe_distribute_combine`](https://www.hiascend.com/document/detail/zh/Pytorch/710/apiref/torchnpuCustomsapi/context/torch_npu-npu_moe_distribute_combine.md)
  沿原通信路径返回 expert output，并完成 route 权重乘加和 token 顺序恢复；
- CANN 对
  [`MoeDistributeDispatch/Combine`](https://www.hiascend.com/developer/techArticles/20250726-1?envFlag=1)
  的公开说明给出了对应 ACLNN 接口和 device-side 通算融合边界；
- vLLM-Ascend 的 `TokenDispatcherWithMC2` 只作为公开调用方式参考，不是算子
  provenance，也不成为 TokenSpeed runtime 依赖。

目标 910B 镜像的只读探针确认：

| 项目 | 结果 |
| --- | --- |
| CANN | `9.0.0` |
| PyTorch | `2.9.0` |
| Torch-NPU | `2.9.0.post2+git912882d` |
| 非 V2 dispatch/combine | Python binding 与 `torch.ops.npu` schema 均存在 |
| V2 | 同时存在，但不属于本阶段候选 |

接口“存在”只证明 binding 可见，不等于 Lite 的 EP8 shape 已获支持。

## 3. Lite Decode 映射

### 3.1 四组分别调用

每层有四个独立 group，每组参数为：

| 参数 | 值 |
| --- | ---: |
| input hidden | `768` |
| real experts | `384` |
| local real experts / EP8 rank | `48` |
| TopK | `12` |
| Decode BS | `1/2` |
| dtype | BF16 activation，FP32 route weight |

官方接口的 `moe_expert_num` 公开范围上限为 `512`，因此不能把四组 flatten 成
`1536` experts 后调用一次。本阶段复用阶段 6C 已准入的 winner A：每 rank 每组持有
48 个 expert，四组分别调用一对 Dispatch/Combine。静态 expert 权重和 checkpoint
placement 不变。

每组的数据流为：

```text
[BS,768] group input + [BS,12] real expert IDs/weights
  -> official MoeDistributeDispatch
  -> local 48-expert GMM1
  -> SwiGLU
  -> local 48-expert GMM2
  -> official MoeDistributeCombine
  -> [BS,768] weighted group output
```

四组 output 在 hidden 维拼回 `[BS,3072]`。只删除当前 routed partial 的 EP8
AllReduce；output projection、shared expert 之后已有的 dense TP8 AllReduce 仍保留。

### 3.2 Identity 与 shared expert

Lite 每组还有 32 个 identity expert，而非 V2 官方接口要求 `expert_ids` 全部位于
`[0, moe_expert_num)`，且同一 token 的 ID 不能重复。因此 identity route 不进入官方
Dispatch/GMM：

1. 先按现有数学求出 identity route weight sum；
2. real route 保留原 ID/weight；
3. identity slot 使用未被当前 token 选中的不同 real ID 补齐，weight 固定为零；
4. Combine 后在本 rank 加一次 `group_input * identity_weight_sum`。

dummy route 会产生无数学贡献的 expert 计算，但保持固定 `K=12`、合法且无重复的 ID，
不依赖公开文档中仍标为预留的 `x_active_mask`。board 必须统计 dummy route 比例和额外
GMM active rows；若这项成本抵消通信收益，候选不进入 production。

shared expert 继续沿现有 dense TP8 分支独立计算并只累加一次。A2 非 V2 接口不负责
Lite shared expert，也不改变其权重切分。

### 3.3 Opaque metadata

Dispatch 返回的 `expand_idx`、`ep_recv_counts`、`tp_recv_counts` 和
`expand_scales` 只能原样传给同层、同次调用的 Combine。模型、kernel adapter 和诊断代码
都不得读取、重排或跨 step 缓存其值。`expert_token_nums` 只作为本次 local GMM 的
group list 使用。

## 4. EP8 是硬门禁，不是已知支持项

华为当前公开的 A2 文档把非 V2 `ep_world_size` 列为 `16/32/64`，没有包含目标
`EP8`；普通通信模式还要求 `local_expert_num <= 24`，而 Lite 每组为 `48`。文档同时
说明，A2 在 `HCCL_INTRA_PCIE_ENABLE=1`、`HCCL_INTRA_ROCE_ENABLE=0` 时可取消
`local_expert_num <= 24` 限制，但没有据此声明 EP8 合法。

所以阶段 8C 的第一个硬件步骤必须是非 V2 的 NPU8 direct board。不得因为：

- Python binding 存在；
- V2 曾进入过另一个框架；
- vLLM-Ascend 有 MC2 adapter；
- H/K/BS/E 的其余维度满足；

就提前把非 V2 标成 production 可用。

对 `E=384, H=768, K=12, BS=2`，官方优化模式的 buffer 下界约为：

```text
384 * 2 * (768 * 2 + 4 * 16) + 4 MiB + 100 MiB
= 105.172 MiB
```

board 使用独立进程和独立 HCCL communicator，`HCCL_BUFFSIZE=128`（MiB 口径），
并对当前通信环境与上述官方推荐环境做隔离 A/B。不得直接修改服务的全局通信环境。

## 5. 通信域与 graph 约束

官方文档要求一对 Dispatch/Combine 使用相同 EP 通信域，且该通信域不混用其它算子。
因此 implementation 不能复用当前还承载 routed AllReduce、dense collective 或其它
模型通信的 process group。需要由既有 `ProcessGroupManager` 为相同八个 Decode rank
创建一个专用于官方 MoE 通信的 communicator；不增加新的 rank 拓扑或跨角色通信。

Decode graph 固定捕获 BS1 和 BS2：

- group name、EP rank/size、expert count 和 `global_bs` 在所有 rank/layer 保持一致；
- route ID/weight、hidden 和 Dispatch metadata 都是 device tensor；
- 图内无 `.item()`、D2H、动态 split list 或 host 分支；
- A2 graph board 按官方接口要求验证 `group_tp` 的传值，不把 eager 成功外推到 graph；
- graph replay 原地更新输入后，output 必须变化且不产生额外 specialization。

## 6. 实施拆分

### 8C.1：ABI 与 direct board

- 独立设计提交：本文；
- 先修正 public-kernel manifest：provider/API 指向华为公开的非 V2 接口，vLLM-Ascend
  只标作 caller reference，不再作为算子 provenance；
- 独立 harness 提交：只调用非 V2 官方 pair，不复制 Fluent V2 adapter；
- 覆盖 real-only、mixed identity、identity-only、hot expert 和 empty-owner；
- 覆盖 BS1/BS2 eager，再覆盖 BS1/BS2 graph；
- 对当前 local-leaf+AllReduce baseline 做同输入 output、finite、HBM 和延迟 A/B；
- 保存准确的参数校验、tiling、AICORE 或 HCCL 首错，不用异常字符串降级成“可用”。

停止条件：EP8 eager 失败即结束该候选，记录目标 CANN/910B 不支持；不尝试 V2，不把
Decode 扩到 16 rank，也不借用 Prefill rank。

### 8C.2：最小 kernel adapter

仅当 8C.1 全部硬门禁通过后实施：

- 在 `tokenspeed-kernel-npu` 复用现有 MoE plan/weight view，新增一个官方非 V2 solution；
- TokenSpeed runtime 不直接 import `torch_npu`；
- 每组仍调用同一个 `moe_apply` facade，solution 内拥有 Dispatch/GMM/Combine；
- Lite wrapper 只负责 identity 分离、四组拼接和选择已准入 solution；
- 保留当前 local-leaf+AllReduce solution 作为 rollback，不新增第二套模型类。

### 8C.3：8P8D service admission

- Prefill 完全不变；
- Decode 完成 eager/graph BS1、已热 BS2、late-admission BS2 和 overlap off/on；
- 对同一次输入比较 group output、完整 MoE output、layer output 和 logits；
- 运行服务联通性与 GSM8K 前 100 条，要求 100/100 完成、零 API error/空输出；
- 记录 TPOT、四组 Dispatch/GMM/Combine 时间、dummy route 比例、HCCL buffer 和 HBM peak；
- candidate 必须无精度回退且 TPOT/HBM 不回退，才替换默认 solution。

每个小阶段均使用“设计/实现与测试/上板结果”独立 signed-off commit，并立即推送。

## 7. 决策表

| 结果 | 决策 |
| --- | --- |
| EP8 参数或 tiling 拒绝 | 保留 local leaf + AllReduce；记录官方非 V2 在目标环境不支持 |
| eager 通过、graph 失败 | 不进入目标 Decode production；保留 board 证据 |
| 数值失败 | 先在 Dispatch output、GMM2 output、Combine output 二分；不放宽门槛 |
| 正确但慢于 baseline | 不接入 production；不为“用了官方算子”牺牲 TPOT |
| eager/graph、精度、HBM、性能均通过 | 经 kernel facade 接入，并在 8P8D 累计验收 |

无论上述哪一分支成立，V2 都不会成为 fallback；最后 TODO 中的 CP/KVP、分布式 KV
page 和 P/D fragment 工作也不会被提前。
