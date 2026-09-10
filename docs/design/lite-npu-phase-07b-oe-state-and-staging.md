# Lite NPU 阶段 7B：OE 请求状态与 Decode Graph Staging

> 历史设计：checkpointed-tail 生产路径已删除，仅保留测试用 CPU 精度 oracle；
> 当前实现与约束见 `lite-npu-oe-device-hash-host-table.md`。

## 1. 目标与边界

本阶段把阶段 7A 已完成的 OE host lookup 接入统一 cache/PD 契约，并冻结 Decode graph 的固定地址
输入。交付内容只有四项：

1. Lite 专属 cache recipe 增加最近 3 个原始 token 的 snapshot state；
2. KDA backend 只管理 KDA layer 真正拥有的 state group，不误消费模型自有 OE state；
3. slot-safe CPU mirror 在 steady Decode 不做逐 token D2H，并能从 prefix/PD snapshot 恢复；
4. Decode 使用 capture 前分配的固定 raw-OE staging，host lookup 保持在 graph 外。

本阶段不连接完整 Lite decoder，不新增 cache family、PD 协议、异步 lookup 线程或 OE 融合算子；完整
模型接线与 8P8D 服务在后续阶段完成。投机 Decode 的 accepted-tail commit 也不在本阶段实现；普通
Decode 固定为每请求 1 个输入 token。

## 2. Cache 契约

### 2.1 Lite recipe 选择

Lite 与 Kimi K3 继续对 scheduler 发布 `family="kimi_k3"`，复用同一 hybrid pool、MLA/KDA backend
和 PD transport。`prepare_cache_setup()` 已获得 `model_config`，因此只在 architecture 包含
`FLASHLocalForCausalLM` 时选择 `LiteRecipe(KimiK3Recipe)`；Kimi K3 仍选择原 recipe，几何和门禁不变。

Lite recipe 在 Kimi 的四个 group 后追加一个完整 group declaration：

| 属性 | 值 |
| --- | --- |
| `group_id` | `lite_oe` |
| `family` | `state` |
| `retention` | `full_history` |
| `checkpoint_granularity` | `128` |
| P/D transfer policy | `latest_snapshot` |
| `field_id` | `layer.0.lite.oe.context` |
| `plane_id` | `slot.0` |
| shape / dtype | `[3]` / `int32` |
| payload | `3 * 4 = 12 bytes` |

该 state 表示 OE 最大 4-gram 所需的最近 3 个**原始** token。它不按 TP/CP/KVP 切分；P/D 的每个
rank 保存同一份 12-byte snapshot。

### 2.2 Packing 与物理内存

TP8 Lite 现有 `slot.0` plane 宽度为 294,912 bytes，LCM parent 为 2,064,384 bytes，共 7 个 plane。
OE 的 packing 为：

```text
oe_packing = 294,912 / 12 = 24,576 child pages / parent
```

因此 `lite_oe` 与既有 `slot.0` plane 共用物理 parent，不增加第八个 plane，且 parent 大小不变。
每个 state group 的 rolling snapshot 需求为：

```text
child_pages = 2 * max_live_requests
oe_parents = ceil(child_pages / 24,576)
```

P64 和 D256 分别只需 128/512 个 child page，均占 1 个共享 parent。每个 group 的 parent demand
按现有 Kimi closed form 相加；三个 KDA group 的 packing、offset、stride 不变。

### 2.3 Capacity 修正

通用 `CacheRecipe.num_lcm_blocks()` 用“最大 packing × 128”把 token limit 直接换算成 parent。新增
`oe_packing=24,576` 后，这个上界不再代表 full-attention token 容量，不能沿用 Kimi 先调用 `super()`
的实现。Lite recipe 直接计算：

```text
budgeted = (cache_budget_bytes - workspace_bytes) // lcm_block_bytes - 1
allocated = budgeted                                      # 无 token limit
allocated = min(budgeted, parents_needed(layout, limit))  # 有 token limit
```

`token_capacity()` 继续复用 Kimi 的单调二分反解；上界仍由 full-attention packing 得出。Lite 的
`max_padding_fraction` 设为无穷，因为 12-byte side state 被有意装入宽 plane；只对 Lite 生效。

## 3. KDA state 与模型自有 state 的所有权

当前 `MambaAttnBackend.set_kv_pool()` 会收集 contract 中所有 `family="state"` 的 group；若直接追加
`lite_oe`，KDA graph metadata、dual-index buffer 和 replay 都会错误地把 OE 当成 KDA state。

根因修复是让 backend 只从下列映射取得 state group：

```text
kv_pool.state_group_by_layer.values()
```

该映射由 `conv_state/recurrent_state` field 派生，天然只含三个 KDA group。backend 仍验证这些
layer-owned group 都存在于 contract、都是 state family 且 checkpoint grain 相同。`lite_oe` 由模型
preparer 直接通过 `arena.field("layer.0.lite.oe.context")` 管理，不进入 KDA 的 capture/replay 循环。
这是一处共享根因修复，Kimi 与其它 hybrid-linear 模型行为不变。

## 4. CPU mirror 生命周期

### 4.1 权威状态与镜像键

NPU arena field 是 prefix cache、slot restore 和 PD transfer 的权威状态；CPU mirror 只是 host hash 的
热路径副本。mirror 固定为 `[max_request_slots,3] int32` 和同长度 owner 列表，避免无界 request 字典。

owner 使用 `(request_pool_index, request_id)`：

- 同 slot、同 request ID：继续消费 mirror；
- slot 第一次出现或 request ID 改变：旧 mirror 失效；
- `before_tokens == 0`：以边界 context 初始化；
- `before_tokens > 0`：从本轮 input page 只恢复 12 bytes；
- batch index、padding row 和 page 0 永远不是 owner。

请求 ID 在 scheduler 活跃期唯一；slot 复用时新的 request ID 强制恢复/初始化，因此不需要另造
generation counter。若未来 scheduler 允许同一 request ID 跨释放后复用，必须先给 ForwardOp 增加原生
generation，再替换 owner tuple，不能在 OE 内猜测。

### 4.2 Token 来源

host preparer 复用 `fill_input_buffers()` 的 CPU 语义，不从 device input buffer D2H：

- Prefill/mixed 的 extend token 来自 flat `forward_op.input_ids` 和 `input_lengths[:num_extends]`；
- Decode token 使用 scheduler 的显式 `decode_input_ids` override；无 override 时，普通非投机场景必须
  从 CPU `forward_op` 暴露的 effective decode token 取得；
- 任何路径都要与 `InputBuffers` 的 vocab clamp、mixed-batch 顺序和 ragged length 完全一致。

实现阶段优先让 `fill_input_buffers()` 返回/保留一份已经归一化的 CPU effective token list，供 optional
model preparer复用；不复制 `future_input_map` 的 override 状态机。若当前 runtime 不能无 D2H 得到非
override Decode token，7B 应在 sampler→ForwardOp 的既有 CPU token 通路补齐这一项，而不是读取 device
buffer。

### 4.3 Restore、更新和 publication

设当前请求 forward 前长度为 `b`、本轮真实输入长度为 `q`，forward 后长度为 `a=b+q`。grain 为 128：

```text
input_slot  = floor((b - 1) / 128), b > 0
output_slot = floor((a - 1) / 128)
```

相应 page ID 从 `forward_op.block_tables_arrays()["lite_oe"]` 的 request row 读取。preparer 执行顺序为：

1. 根据 owner 和 `b` 判断初始化、mirror hit 或 12-byte snapshot restore；
2. 对本轮 ragged token 调用 7A `ngram_ids()`，得到 ID、special mask 与 final context；
3. host lookup 生成 `BF16[T_real,12,256]`；
4. final context 写入 output page，page 0、负 page、重复/越界 page fail loud；
5. 更新 slot owner/mirror；
6. H2D staging 完成后才允许 eager forward 或 graph replay。

同 stream 的 H2D/state publication 与后续 forward 保持顺序。prefix hit 和 Decode PD admission 都只在
owner miss 时读取一次 12 bytes；steady Decode 不做 D2H。PD 的 `latest_snapshot` 已由统一 manifest 和
Mooncake transfer 覆盖，不增加 OE 专用 sender/receiver。

## 5. Decode NPUGraph 固定 staging

### 5.1 地址与形状

Decode capture 前由 Lite 模型分配：

```text
raw_oe_staging: BF16[max_graph_tokens, 12, 256]
```

地址在 capture/replay 生命周期内不变。每轮 host lookup 后：

1. 将 `[:T_real]` 异步 H2D copy 到 staging；
2. 将 `[T_real:T_bucket]` 清零；
3. model forward 只读取 `[:T_bucket]` 并执行既有 packed projection/merge。

host hash、mmap lookup、page-table 解析和 CPU mirror 均在 graph 外；graph 内只有固定 shape/address 的
projection/merge。P eager 使用本轮精确 shape tensor，不为 16K chunk 常驻最大 staging。

### 5.2 最小 executor seam

`ModelExecutor` 在 graph wrapper 构造/capture 前仅对实现了 Lite hook 的模型调用一次 runtime 初始化；
每步在 `fill_input_buffers()` 和 operation-bound cache metadata 发布后、`forward_step()`/graph replay 前
调用一次 preparer。非 Lite 模型无 hook，完全不变。

不把 tensor 塞进 `ForwardContext`，也不新增 callback registry。Lite 模型内部持有 fixed staging；未来
完整 embedding forward 直接读取它。阶段 7B 的 hook 允许在 decoder 尚未接线时通过 focused test 验证
地址稳定、padding 清零、state restore/publication 和 projection 数值。

## 6. Fail-closed 条件

以下情况立即抛错，不静默退化：

- 非 Lite architecture 选择 Lite recipe，或 Lite 未得到 `lite_oe` field；
- OE group 改变 Kimi 的 7-plane/parent 几何，或三个 KDA field offset/stride 变化；
- KDA backend 发现 layer-owned group 缺失、非 state 或 checkpoint grain 不一致；
- request/length/token/page-table 数量不一致，输入 ID 越界；
- active row 指向 page 0、负 page、越界 page，或多个不同 owner 写同一 output page；
- graph replay 更换 staging data pointer、tail 未清零或真实 token 超过 bucket；
- 普通 Decode 无法在 CPU 取得与模型一致的 effective token；
- 投机 Decode 宽度大于 1。

## 7. 验证矩阵

### 7.1 Recipe 与 KDA 隔离

- Lite 自动选择专属 recipe，Kimi 仍选择原 recipe；
- TP8 plane 数仍为 7，plane/parent bytes 不变，OE stride=12、packing=24,576；
- P64/D256 的 2M token capacity 精确可反解，OE 额外需求不超过 1 parent；
- 三个 KDA group 的 field offset、stride、packing 与 7A 前 bitwise 相同；
- KDA backend 只建立三个 group 的 dual-index buffer，不含 `lite_oe`。

### 7.2 Mirror、边界与 PD

- 新请求、空 prompt、chunk continuation、两请求交错；
- 127/128/129 token grain 边界；
- prefix-hit owner miss 从 12-byte snapshot 恢复，steady Decode 不读 device；
- slot reuse 更换 request ID 后旧 context 不泄漏；
- page 0、padding row、邻接页不被覆盖；
- P snapshot 经现有 `latest_snapshot` manifest/transfer 后，D continuation 与 uninterrupted oracle exact。

### 7.3 Graph staging

- BS1/BS2 capture/replay 前后 data pointer 不变；
- 从大 bucket 切回小 batch 后 padding 全零；
- eager exact tensor 与 fixed staging projection/merge 在 BF16 门槛内一致；
- NPU replay 多次 finite，无 NaN/Inf，无额外 host lookup 进入 capture；
- KDA/MLA/Grouped MoE/7A OE 累计 focused tests 全部通过。

## 8. 提交边界

1. 本文档单独提交并推送；
2. recipe/KDA ownership、mirror/staging 实现和 focused tests 单独提交并推送；
3. Phase 7C exact-source NPU、graph、PSS/HBM 结果写独立验证记录再提交并推送。

只有第 2 步所有 CPU/meta/focused 门禁通过，才部署到从资源文件动态解析出的 CANN 9 目标机；每轮测试
结束都精确清理进程并确认 NPU 无残留。
