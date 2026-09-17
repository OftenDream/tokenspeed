# Lite NPU 阶段 7C：Exact OE Checkpoint 与 8P8D 内存账本设计

> 历史设计：checkpointed-tail 生产路径已删除，仅保留测试用 CPU 精度 oracle；
> 当前实现与约束见 `lite-npu-oe-device-hash-host-table.md`。

## 1. 目标与边界

阶段 7C 对 Phase 7A/7B 已完成的 OE 数学、mmap adoption、12-byte state 和 fixed graph staging 做真实
checkpoint 验收，回答四个问题：

1. 12 张 OE 主表是否始终保持 CPU file-backed storage，没有形成逐进程 27 GiB anonymous copy；
2. 8P8D 共 16 个独立 rank 同时映射和访问相同表页时，host resident page 是否由 OS page cache 共享；
3. NPU HBM 是否只增加 18 MiB packed projection、KB 级 graph staging 和既有 cache arena，而没有
   27 GiB 主表；
4. exact checkpoint 的 lookup/projection/special、state snapshot 和 graph replay 是否继续满足数值与
   生命周期门禁。

本阶段仍不连接完整 decoder、HTTP 服务或正常 Decode scheduler token 通路；这些属于后续模型集成。
因此 7C 的 8P8D 指 16 个真实角色进程同时加载 OE 子模块并建立角色对应内存账本，不提前宣称完整
8P8D 服务可用。所有 checkpoint 路径仅由运行环境传入，不进入源码、文档或日志提交。

## 2. 固定几何与账本口径

目标 OE 几何为：

```text
V = 163,840
H = 3,072
G = 12
K = 256
rows_i = 4,718,592 + 2*i + 1, i=0..11
```

由此得到：

```text
OE payload     = 28,991,102,976 bytes = 27.00007 GiB
projection     = 12 * 256 * 3,072 * 2 = 18,874,368 bytes = 18 MiB/rank
staging/token  = 12 * 256 * 2 = 6,144 bytes
context/page   = 3 * sizeof(int32) = 12 bytes
```

必须区分三种量：

- **tensor payload**：12 个 OE tensor 的逻辑 byte interval，总和必须精确为 28,991,102,976 bytes；
- **file mapping Size**：safetensors 按 shard mmap，保留一个 tensor 可能保留包含它的整个 shard；该值可
  大于 27 GiB，不能当作 resident table 大小；
- **RSS/PSS**：实际被 fault-in 的文件页。只有这一层能回答 host physical memory 是否共享。

HBM 同样区分 `torch.npu.memory_allocated/reserved`、`npu-smi` 的 process memory 和模型逻辑 payload，
不从单一数字反推其它层。

## 3. 最小复用实现

新增一个仅用于 exact-checkpoint 验收的通用 probe，不引入生产 runtime manager：

- 从 `model.safetensors.index.json` 找到 12 个 embedder 与 12 个 post projection key；
- 复用 safetensors CPU mmap 和 Phase 7A 的 table storage adoption 语义；
- 复用 `LiteConfig`、`LiteNgramParameters.ngram_ids/lookup_host/project_and_merge`；
- 复用 Phase 7B `LiteOEStatePreparer` 和 fixed staging；
- 通过命令行或环境传入 checkpoint、rank、role、device 和结果 JSON 路径；
- 只输出 shape、byte、PSS/HBM、数值误差和匿名化 rank 指标，不输出 checkpoint 路径。

probe 使用标准库解析 `/proc` 和同步 16 个 worker；不增加 psutil、共享内存服务或新的运行时依赖。
生产代码只在上板发现真实缺陷时修改；纯测量逻辑留在 `test/ci_system` 或 focused test 中。

## 4. Exact checkpoint storage adoption

每个 worker 按 index 只打开 OE key 所在 shard。对每张表执行：

1. 校验 source shape/dtype 与 `LiteCheckpointLayout.spec()` 完全一致；
2. 将 loaded CPU tensor storage 交给对应空 Parameter；
3. 校验 Parameter 与 source 的 storage data pointer 相同；
4. 释放 shard 结果字典并执行 GC；
5. 再次读取固定行，证明 mapping 生命周期由 Parameter 保持；
6. 从 `/proc/<pid>/maps` 找到覆盖 tensor pointer 的 file-backed mapping；
7. 校验 12 个 tensor payload interval 总和精确为 27.00007 GiB。

不允许通过 `clone()`、`contiguous()`、anonymous shared memory 或 host tensor cache 规避该门禁。任何表
未落在 file-backed mapping，或 adoption 后 pointer 改变，立即失败且不进入 16 进程实验。

## 5. Shared page 实验

### 5.1 两进程根证据

先启动两个独立 worker，映射同一 checkpoint，并在每张表读取相同、确定性的稀疏行集合。触页集合按
4 KiB OS page 对齐，目标总量至少 256 MiB，但不扫描完整 27 GiB。两个 worker 在 barrier 后保持存活，
controller 同时读取：

为排除 safetensors 加载和文件系统预读产生的非确定 PTE，worker 在正式触页前仅对自身 file mapping
执行 `MADV_DONTNEED`，随后用 `MADV_RANDOM` 关闭预读，再逐 4 KiB 连续覆盖固定 256 MiB。该操作不
修改 checkpoint，也不执行系统级 page-cache 清理；测量只包含两个 worker 明确访问的相同文件页。

- checkpoint file mappings 的 `Size/Rss/Pss/Shared_Clean/Private_Clean/Private_Dirty/Anonymous`；
- 整进程 `smaps_rollup`；
- 每个 table pointer、payload bytes 和所属 file mapping；
- 触页前后 `MemAvailable/Cached` 差值。

硬门禁：

- table mappings 的 `Anonymous == 0`；
- 两个 worker 均出现非零 file-backed RSS；
- 相同页同时驻留后 `Shared_Clean > 0`；
- 两进程 checkpoint mapping 的 `sum(Pss)` 接近一份实际 resident set，而不是两份 RSS 之和；允许
  25% 页面对齐、header 和异步回收余量；
- 进程 anonymous 增量远小于 27 GiB，默认上限 512 MiB/worker。

### 5.2 16 进程 8P8D 账本

两进程通过后才启动 16 个 worker：rank 0--7 标记为 P，rank 8--15 标记为 D。所有 worker 触摸同一
稀疏页集合并在同一 barrier 采样。记录每 rank 和全局：

```text
checkpoint RSS/PSS/Shared_Clean
process RSS/PSS/Anonymous
sum(checkpoint PSS)
max/sum(checkpoint RSS)
MemAvailable/Cached delta
```

`sum(checkpoint PSS)` 应仍接近一份实际 resident page set；若随 rank 近似线性增长，说明页并未共享，
Phase 7 失败。16 进程只增加页表和各自小型 Python/runtime 内存，不允许出现 16 份 OE anonymous payload。

## 6. NPU HBM 账本

每个 worker 绑定唯一 NPU，在四个 publication point 同步采样：

1. NPU runtime 初始化后 baseline；
2. 12 张 host table mmap/adoption 后；
3. 12 个 post projection 打包到 NPU 后；
4. role staging、context view 和一次 projection/merge 后。

记录 `memory_allocated`、`memory_reserved`、`max_memory_allocated` 和 `npu-smi` process memory。硬门禁：

- 第 2 步相对 baseline 的 `memory_allocated` 增量不超过 16 MiB，证明 host table 未进入 HBM；
- 第 3 步逻辑增量为 18 MiB，allocator/format 余量单独记录；
- fixed staging 的逻辑增量严格为 `bucket_tokens * 6,144` bytes；P eager 只保留最小 runtime bucket，
  D 验证 BS1/BS2 fixed bucket；
- `lite_oe` 复用 Kimi 既有 7-plane arena，parent bytes 和 parent count不变，因此 OE context 的增量
  cache arena HBM 为 0；只记录 12-byte child view，不重复归因完整 parent；
- 任一 rank 出现接近 27 GiB 的 HBM 增量立即失败。

本阶段不把 Python/NPU runtime、通信库或 allocator reserve 混入“模型权重”列；它们分别放在 runtime
和 reserved 列。

## 7. Exact 数值矩阵

固定 token 序列覆盖普通 token、23 个 special、EOS、请求边界与两请求 continuation：

| 路径 | token 数 |
| --- | --- |
| Decode-like eager/graph | 1、2、4、32 |
| Prefill eager | 1、128、1,024 |

验证内容：

- 12 路 local ID 在表范围内，lookup 行与 source file exact；
- packed projection/merge 对逐路 CPU FP32 oracle；
- ordinary token 使用 `/sqrt(13)`，special 行与 word bitwise exact；
- 全部输出 finite，无 NaN/Inf；
- BF16 门禁为 `atol=0.02, rtol=0.02`，同时记录 max-abs 和 relative-L2；
- D BS1/BS2 graph replay 至少更新两轮 word、token、context 和 lookup row，输出必须变化并与 eager 对齐；
- Prefill T1024 的 CPU oracle单独运行，不在 16 个 worker 中重复 16 次。

## 8. State 与 PD manifest

本阶段不实现新的 PD transport。使用统一 cache contract 做一条可执行闭环：

1. P preparer 处理跨 128 boundary 的 token，发布 `lite_oe` output snapshot；
2. 统一 `latest_snapshot` manifest 选择该 group 的唯一最新 child page；
3. 将 manifest 指定的 12 bytes 复制到独立 D arena 的 destination page；
4. D slot owner miss 从 snapshot 恢复，并用显式 CPU decode token继续；
5. P→D continuation 与不拆分 CPU oracle exact；
6. manifest 不包含旧 OE page，不增加第二份 transfer cache，KDA/MLA group 仍按原 policy 处理。

正常 Decode 的 device-only token 仍 fail closed；该 manifest 测试使用明确 CPU token，只证明 state/PD
selection 和 restore，不掩盖 scheduler publication 缺口。

## 9. 执行顺序与停止条件

固定顺序：

1. generic probe 单进程 manifest/shape/mmap smoke；
2. NPU0 exact数值与 BS1/BS2 graph；
3. 两进程 shared-page/PSS 根证据；
4. 16 进程 8P8D OE-only PSS/HBM ledger；
5. PD manifest/restore；
6. 累计 Lite/KDA/MLA/Grouped MoE focused 回归；
7. 清理、验证记录、signed-off 提交并推送。

任一步出现 anonymous 27 GiB copy、主表 HBM copy、pointer 不别名、数值门禁失败或无法精确清理时，
停止扩大并发；先定位根因，不通过降低触页量或只报 virtual Size 隐藏问题。

## 10. 提交边界

1. 本设计文档独立提交并推送；
2. generic probe、必要的最小生产修复和 focused tests独立提交并推送；
3. exact-source 结果、PSS/HBM 表、SHA 与清理证据写独立验证记录，再提交并推送。

Phase 7C 通过后，Phase 7 才算闭环。后续按总体计划进入完整 decoder、8P8D role execution、正常
Decode CPU token publication、服务和 GSM8K first100，不在本阶段交叉实现。
