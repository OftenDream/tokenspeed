# Lite NPU 阶段 7：OE 正确性与 Host Residency

> 历史设计：checkpointed-tail 生产路径已删除，仅保留测试用 CPU 精度 oracle；
> 当前实现与约束见 `lite-npu-oe-device-hash-host-table.md`。

## 1. 目标与阶段边界

本阶段实现 Lite 的 Over Embedding（OE）最小完整闭环：严格加载目标 checkpoint 的 12 张 BF16
n-gram 表且让主表常驻 host；按 checkpoint 语义生成 n-gram ID、查表、一次 block projection、与普通
word embedding 合并；最近 3 个 token 进入统一 cache/PD 契约；Decode NPUGraph 只捕获固定地址的
projection/merge，host lookup 在 replay 外完成。

阶段拆成三个独立实现与验证提交：

1. **7A：语义与存储**——mmap host loader、CPU oracle、packed projection、special merge；
2. **7B：历史与 graph**——最近 3 token cache/PD snapshot、slot-safe CPU mirror、固定 staging；
3. **7C：实卡验收**——NPU 数值、graph replay、exact checkpoint、PSS/HBM ledger。

本阶段不连接完整 decoder，不实现 CP8/KVP8 attention 或 8P8D HTTP 服务；这些仍由后续集成阶段完成。
不引入私有运行时依赖，不为未来量化、投机推理或尚不可执行的融合算子预建配置。

## 2. Checkpoint 几何与内存

目标配置固定为：

| 符号 | 含义 | 值 |
| --- | --- | ---: |
| `V` | 普通词表大小 | 163,840 |
| `H` | 模型 hidden size | 3,072 |
| `N` | 最大 n-gram 阶数 | 4 |
| `S` | 每阶 split 数 | 4 |
| `G=(N-1)S` | OE component 数 | 12 |
| `K=H/G` | 每路 OE hidden | 256 |
| `M` | 基础表行数 `int(V*28.8)` | 4,718,592 |

第 `i` 张表（`i=0..11`）的形状和行数为：

```text
rows_i = M + 2*i + 1
table_i = BF16[rows_i, K]
post_i  = BF16[H, K]
```

因此：

```text
sum_rows = sum(M + 2*i + 1, i=0..11)
         = 12*M + 12^2
         = 56,623,248

host_table_bytes = sum_rows * K * sizeof(BF16)
                 = 56,623,248 * 256 * 2
                 = 28,991,102,976 bytes
                 ~= 27.00007 GiB

projection_bytes = G * H * K * sizeof(BF16)
                 = 12 * 3,072 * 256 * 2
                 = 18,874,368 bytes
                 = 18 MiB/rank
```

27 GiB 主表不得出现在 HBM，也不得在每个 rank 创建 anonymous CPU 副本。18 MiB projection 在 P/D
每个 rank 复制；它很小，且 P/D 都需要完整 12 路输出。

## 3. 数学语义

### 3.1 Component 顺序与 n-gram ID

component 顺序固定为 `order=2..4` 外层、`split=0..3` 内层：

```text
i = (order - 2) * S + split
```

令 `x_t` 为当前 token，`M_i=rows_i`。在一张独立表中的 local ID 为：

```text
id_i(t) = (
    x_t
    + sum(x_(t-d) * pow(V, d, M_i), d=1..order-1)
) mod M_i
```

每个乘数预先计算为 int32 常量；运行时不做大整数幂。TokenSpeed 保持 12 张分表，因此 ID 不加融合
大表的 prefix offset。输入 ID 必须位于 `[0,V)`，越界立即失败。

当前 token 之前最多只读取 3 个 token。请求起点、EOS 和 special token 都形成边界；边界之外的历史值
按 0 处理，不能串到相邻请求。变长 batch 由每个请求的 length/offset 决定，不能把 flat batch 当成一条
连续序列。

### 3.2 Special token

配置 `special_token_scope="0:4,36:55"` 对应 23 个 ID。`ngram_exclude_sp_token=true` 时：

- hash 前把 special token 映射为 0，并阻断更早历史；
- 当前 token 为 special 时，最终输出严格等于原始 word embedding；
- special token 后的普通 token 从新的边界重新累计 n-gram。

目标 checkpoint 配置与当前已验证 reference 都采用上述 base-only epilogue。历史 Megatron replay
记录中存在 special epilogue 差异；用户已把 Megatron 精度对齐留到后续阶段，因此本阶段冻结
checkpoint/reference 语义，不增加一个无法同时正确的运行时开关。最终 Megatron 对齐若证明目标训练
分支不同，应以独立精度变更提交修正并重跑本阶段全部门禁。

### 3.3 Lookup、projection 与 merge

12 路 lookup 得到：

```text
E[t,g,k] = table_g[id_g(t), k]                 [T,12,256]
P[g,k,h] = transpose(post_g)[k,h]              [12,256,3072]
oe[t,h]  = reshape(E,[T,3072]) @ reshape(P,[3072,3072])
```

普通 token：

```text
output = (word_embedding + oe) / sqrt(G + 1)
       = (word_embedding + oe) / sqrt(13)
```

special token：

```text
output = word_embedding
```

projection 直接装载成一份连续 `[12,256,3072]` Parameter。loader 把 checkpoint 的每个 `[3072,256]`
source 转置后写入对应 slice；forward 不再 stack/transpose 12 份权重，也不物化 `[12,T,3072]` BMM
结果。当前 block matmul 已是可接受 baseline；没有证据支持新增专用 OE projection kernel。

## 4. Host 主表加载

### 4.1 safetensors mmap 所有权

现有 safetensors iterator 使用 `load_file(path, device="cpu")`。返回 tensor 的 Storage 指向 checkpoint
文件 mmap；只要 Parameter 继续持有该 Storage，即使 iterator 的临时字典释放，mapping 仍然有效。

host-OE source 使用以下最小所有权转换：

```text
loaded mmap tensor -> existing Parameter adopts loaded storage
```

不能先 materialize 再 `copy_`。后者会把 27 GiB 变成每进程一份 anonymous private memory，并失去
OS page-cache 共享。普通权重、expert packed loader 和非 OE 参数继续沿用现有 copy/shard 逻辑。

目标 16 rank 进程映射同一组 checkpoint 文件：

- 每个进程各有约 27 GiB virtual mapping；
- 同一宿主机实际文件页由 OS page cache 共享；
- 只有被当前请求命中的页按需 fault；
- 不创建额外 27 GiB `multiprocessing.shared_memory`，也不 pin 整张表。

非 safetensors source 仍可保持 host-only，但不承诺跨进程物理共享；完整目标 checkpoint 的验收要求
safetensors file-backed mapping。

### 4.2 与通用 post-load 的边界

host table module 不定义 `process_weights_after_loading`。通用 loader 会把带该方法的模块所有 CPU
Parameter 临时搬到目标 device，再复制回 CPU；对 27 GiB 表这既可能 OOM，也会再次产生 anonymous
副本。packed projection 本身是 device Parameter，不需要 host table 参与 post-load。

loader 必须继续严格检查 12 张表的行数、`K=256`、BF16、component 无重无漏；“为了 mmap”不能放宽
checkpoint coverage 或 shape/dtype 门禁。

## 5. 当前 batch staging

host lookup 后只搬运命中行：

```text
raw_oe = BF16[T_real,12,256]
H2D bytes = T_real * 12 * 256 * 2 = T_real * 6 KiB
```

典型大小：

| 场景 | `T_real` | 单 rank H2D |
| --- | ---: | ---: |
| Decode BS1 | 1 | 6 KiB |
| Decode BS2 | 2 | 12 KiB |
| Prefill local T=16,384 | 16,384 | 96 MiB |

P eager 使用当轮 tensor，生命周期到 embedding merge 结束；不常驻一份最大 Prefill buffer。D graph 在
capture 前分配固定地址 staging，大小只覆盖实际捕获的 Decode bucket。replay 前：

1. host preparer 计算真实行；
2. copy 到 staging 前 `T_real` 行；
3. padded bucket 剩余行清零；
4. captured matmul/merge 读取同一 data pointer。

host lookup 和 H2D copy 不进入 NPUGraph。graph 内不读取 Python 容器、host pointer 或动态 tensor
shape，也不按 live BS 重新分配 buffer。

## 6. 最近 3 token 的 cache/PD 设计

### 6.1 为什么不保留完整 token table

OE 最大阶数为 4，所以状态充分统计量只有每请求最近 3 个原始 token。完整
`[max_requests,max_context_len]` int32 token table 会在 2M 上下文下按最大并发线性增长，并且与主表
host residency 的内存目标冲突；本阶段不复制该实现。

### 6.2 Lite cache recipe

新增 `LiteRecipe(KimiK3Recipe)`，只扩展一个 state group；对外 cache family 仍为 `kimi_k3`，已有
hybrid pool/backend/factory 不增加第二套实现。只在 architecture 为 Lite 时选择该 recipe。
由于 OE 是刻意塞入既有 parent 的窄 side state，Lite recipe 只对自身 layout 放宽 padding fraction；
Kimi recipe 和其它模型的 padding 门禁不变。

```text
group_id                = "lite_oe"
family                  = "state"
retention               = "full_history"
checkpoint_granularity  = 128
transfer_policy (PD)    = "latest_snapshot"
field_id                = "layer.0.lite.oe.context"
shape                   = [3]
dtype                   = int32
payload                 = 12 bytes/request snapshot
```

field 复用现有 `slot.0` physical plane，不增加第八个 plane。目标 KDA TP8 下该 plane 为 294,912 bytes，
OE packing 为 `294,912/12=24,576` blocks/parent；LCM parent 仍为 2,064,384 bytes。P64/D256 的双页
snapshot 分别只有 128/512 个 child blocks，因而最多增加一个现有大小的 parent，而不是按 2M token
扩张。layout test 必须证明：

- plane 数仍为 7；
- plane bytes 与 LCM parent bytes 不变；
- OE page stride 为 12 bytes；
- target concurrency 下新增 parent 数不超过 1；
- 三个 KDA group 的既有 field offset/stride 完全不变。

OE field 不做 TP/CP partition。P/D 的每个 rank 都保存完整 3-token context，PD 使用既有
`latest_snapshot` 传输；不新增 OE 专用网络协议。

### 6.3 CPU mirror 与 slot 生命周期

host hash/lookup 需要 CPU context。steady Decode 直接消费 CPU `forward_op` 中的输入 token，并更新
slot-keyed mirror，不做每 token D2H。NPU cache field是 prefix-cache/PD/restore 的权威 snapshot：

- 新请求：mirror 初始化为边界值；
- 普通延续：从 mirror 读取并以 CPU 输入更新；
- prefix hit、PD admission：首次从 12-byte NPU snapshot 恢复 mirror；
- 请求结束、abort、slot generation 变化：立即使旧 mirror 失效；
- output snapshot 在 graph replay 前由 host preparer写入 cache 的 output page。

mirror key 至少包含 request-pool slot 与其当前 generation/identity，不能只按 batch index。padding row、
idle row 和已释放 slot 不得写入 page 0 或覆盖活跃请求。

## 7. P/D 角色行为

| 角色 | ID/history | lookup/projection | 并行语义 |
| --- | --- | --- | --- |
| P | 对完整 ragged prompt/chunk先生成 ID，再按 SP owner 取本 rank 行 | host lookup 本 rank 行，eager H2D + block matmul | CP/SP8 不切 OE weight；每 rank 完整 mmap 主表和 projection |
| D | KVP8 rank 消费同一 current-token batch与相同 3-token snapshot | 每 rank独立 host lookup，固定 staging + captured matmul | 不做 OE collective；结果在进入后续 TP8 projection 前保持 rank 间一致 |

P 的最终 input ownership 由 CP/SP 集成阶段提供；Phase 7 API 显式消费 ragged lengths 和 token-owner
indices，不在 OE 内部猜测均匀切分。D 不是 request-DP，不能让八个 rank 以不同请求子集更新 mirror。

## 8. Runtime 接线边界

复用现有执行次序：`fill_input_buffers -> CacheBatchMetadata -> model forward/graph replay`。增加一个可选
model external-input preparer，调用点位于 input/cache metadata 已就绪之后、任何 eager forward 或
graph replay 之前。非 Lite 模型没有该 hook，行为不变。

初始化时 preparer只做两件事：

1. 按已配置 Decode capture bucket 预分配固定 staging；
2. 绑定 OE context field view，不另建 cache manager。

每步调用只传已有对象：CPU `forward_op`、input buffers、cache metadata、真实/padded token 数。hook
不拥有 scheduler 状态，不复制 page table，不发 collective。模型 forward 只消费已经准备好的 device
raw-OE tensor并执行 packed projection/merge。

阶段 7B 若现有执行对象已能直接承载这一步，则只加一处可选调用；不会新增通用 callback registry、
线程池或 OE 专用 executor。异步 CPU lookup/H2D overlap 只有在后续 profile 证明同步 baseline 是瓶颈时
才设计。

## 9. NPU 算子审计与回退

目标 TorchNPU 暴露：

```text
npu::compute_n_gram_ids(
  oe_weights, oe_mods, exclusive_oe_embeder_size_sums,
  tokens, exclusive_req_len_sums, oe_token_table,
  row_indices, column_starts,
  batch_size, oe_n, oe_k, max_context_len, eos_token_id
) -> Tensor
```

但当前 CANN 9 环境实际最小调用失败：`libopapi.so` 不包含
`aclnnComputeNGramIds/GetWorkspaceSize`；私有 `custom::npu_compute_n_gram_ids` 也未注册。因此能力门禁
不能只检查 `hasattr` 或 dispatcher schema。

本阶段选择：

- CPU hash 是 correctness oracle 和生产 baseline；
- host lookup 天然要求 CPU ID，避免 native ID 再 D2H；
- 在统一 embedding kernel 边界保留一处替换点；
- 只有公开、可执行的 compact-context op 在目标镜像通过数值、graph、内存和性能门禁后才能替换；
- TokenSpeed 不依赖 reference 工程的私有 `flash_ops`。

projection 使用普通 NPU matmul。未来融合 kernel 只允许替换
`raw lookup [T,12,256] -> projected [T,3072]` 这一行，不能改变 host ownership、ID、cache 或 merge ABI。

## 10. 测试矩阵

### 10.1 Phase 7A：CPU/meta

- 手算 2/3/4-gram、12 路顺序、模数、请求边界、EOS/special、ragged batch；
- prefill 一次输入与逐 token decode continuation 的 ID/output exact 一致；
- ordinary `/sqrt(13)`、23 个 special base-only、零 token；
- 12 路独立 projection 对一个显式 sum oracle；
- loader source 无重无漏、slice 转置 exact、错误 shape/dtype fail loud；
- mmap storage data pointer/file mapping 保持，iterator 临时对象释放后仍可读；
- 非 OE loader、Grouped MoE、KDA/MLA focused tests不回退。

### 10.2 Phase 7B：cache/graph contract

- 新请求、chunk continuation、prefix boundary 127/128/129、slot reuse、abort、两请求交错；
- context page input/output、page 0、padding row、neighbor page exact；
- PD policy 为 `latest_snapshot`，field不分片且 P/D restore 后 continuation exact；
- cache layout plane/stride/parent 上界；
- Decode BS1/BS2 capture/replay，更新 token/context/table row 后输出随之变化；
- padded BS2 的第二行清零，不复用上一步 OE；idle 不更新 mirror/page。

### 10.3 Phase 7C：目标 NPU/exact checkpoint

- synthetic T1/T2/T4/T32 与 Prefill T1/T128/T1024；
- exact checkpoint 随机固定 token、special、EOS、两请求 continuation；
- projection/merge 对 CPU FP32 oracle：全部 finite，BF16 `atol=0.02,rtol=0.02`；
- Decode BS1/BS2 graph replay至少两组 live 输入；
- 两个独立 rank 进程的 `/proc/<pid>/smaps`/PSS 证明表为 file-backed、无 27 GiB anonymous private
  copy；NPU memory 证明主表不在 HBM；
- 记录 table virtual/RSS/PSS、projection/staging/cache parent、peak allocated/reserved；
- exact source SHA、本地/tracking/远端 SHA 和测试后 NPU 进程清理写入独立验证记录。

## 11. 准入、失败与回退

Phase 7 只有同时满足以下条件才闭环：

1. CPU 语义、cache/slot、loader 和 mmap tests 全部通过；
2. exact checkpoint 主表保持 file-backed，HBM 只看到 projection/staging/context；
3. eager 与 graph 对同一输入满足门槛且无 NaN/Inf；
4. prefix/PD restore 后与不拆分序列 exact 一致；
5. full pre-commit、signed-off commit、立即推送和独立验证记录完成。

若 mmap adoption 失败，拒绝完整模型加载，不回退到 16 份 anonymous 27 GiB copy。若 graph external
staging 失败，保留 eager OE 并禁止 Decode graph，不在 graph 内执行 host lookup。若 projection 性能
不达标，先保留同数学的一次 matmul并记录 profile，不以未经验证的融合 kernel 替换正确性路径。

## 12. 本阶段明确不做

- 不实现完整 decoder/residual/lm-head 或 8P8D launcher；
- 不实现 CP8/KVP8 attention、PD transport 网络层或 overlap；
- 不保存 2M token 完整 OE table；
- 不接入私有 `flash_ops`，不复制 reference runtime；
- 不启用 Prefill graph；
- 不增加 OE quantization、投机 verify rollback 或异步 host lookup 线程池；
- 不提前裁决后续 Megatron golden 的 special epilogue 差异。
