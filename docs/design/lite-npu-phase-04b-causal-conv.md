# Lite NPU 阶段 4B：Causal-Conv

## 1. 目的

本阶段为 Lite width-4 packed QKV causal-conv 建立统一 kernel 入口，并替换阶段 3 逐 token Python
reference。目标同时覆盖 Prefill eager、Decode fixed-BS NPU Graph、独立 `state_in/state_out` page 和
padding；不改变阶段 2 的 cache 几何或 PD 协议。当前生产算子由可选的 `flash_ops`
包提供；阶段 4A public artifact 仅继续承载其余 KDA 算子。

当前合同以第 10.7、11 节为准。第 2--9 节保留早期 channel-major 适配与性能实验背景，
其中 compact Prefill 和连续 state 的测试结果不能证明真实 arena 的首轴 stride 兼容性。

阶段 4B 分成三个独立提交：

1. 本文：冻结布局适配、kernel selection、状态发布和上板门槛；
2. 统一 kernel API、Ascend 实现、Lite 调用和 focused tests；
3. exact-source NPU 数值、graph、性能 A/B 和回退验证记录。

本阶段不接入 recurrent/chunk KDA，不改变 beta/gate 数学，也不启动完整模型服务。

## 2. 复用边界

### 2.1 保持不变的部分

- Lite 继续把 Q/K/V projection 拼成一次 `[T,C]` 输入，其中 TP8 的 `C=1536`；
- checkpoint 权重的逻辑布局保持 `[C,4]`；
- cache 中 conv state 保持 `[pages,C,3]`；
- `KdaAttnBackend` 继续从统一 metadata 取得独立的 `read_indices/write_indices`；
- Prefill 继续由 `_prepare_cache_prefill_state_inputs` 将输入 checkpoint 复制到工作输出页；
- page 0 保持只读逻辑零页，padding index 保持 `-1`；
- portable reference 继续作为 CPU、非 Ascend 和显式 debug oracle。

不得为了迁就一个算子，把整个 cache 改成 `[pages,3,C]`。该改动会同时扩散到 allocator、PD schema、
Kimi/GDN fallback 和既有 Triton ABI，而本阶段只需搬运 active rows 的 `B*C*3` 个元素。

### 2.2 新增的唯一公共入口

`tokenspeed_kernel.ops.attention.kda_causal_conv1d` 接收标准布局：

```text
projected       [T,C]             BF16/FP16
weight          [C,W]             BF16/FP16，W=4
conv_state      [pages,C,W-1]     与 projected 同 dtype
read_indices    [B]               int32/int64
write_indices   [B]               int32/int64
cu_seqlens      [B+1]             device int32/int64
cu_seqlens_cpu  [B+1] or None     host int64 Prefill 边界
has_initial     [B] or None       Prefill 初态掩码
bias            [C] or None
decode          bool
```

返回 `[T,C]`，并只向 `write_indices` 发布最后 `W-1` 个原始 projection。runtime 不导入
`torch_npu`、公开 artifact loader 或 vendor op；这些内容只存在于 kernel package 的 Ascend 注册中。

## 3. 公开算子 ABI 审计

阶段 4A 固定的 `causal_conv1d` schema 使用：

```text
x             [T,C] or [B,S,C]
weight        [W,C]
conv_state    [slots,W-1,C]
cache_indices [B]
run_mode      0=Prefill, 1=Decode
```

它只有一个 `cache_indices`：同一 slot 既是输入又是输出；`initial_state_mode` 仅在 Prefill 有效；Decode
始终从该 slot 读取已有 history。公开 kernel 支持 width 2--4、BF16/FP16、2D varlen Prefill、2D/3D
Decode、SiLU、pad sentinel 和可选 bias。

这与 TokenSpeed 有两个不能忽略的差异：

1. state 的最后两维相反；
2. TokenSpeed 允许 `read_page != write_page`，公开 ABI 不允许。

把 `[pages,C,3]` 直接转置成非连续 `[pages,3,C]` view 虽然不会报错，但公开 op 的 contiguous
materialization 不会把 mutation 写回原 pool。这是静默 state 丢失，禁止作为生产路径。

## 4. State 适配

公开 Prefill 的 large-batch 路径只搬运 active batch rows：

```text
active       = (read >= 0) & (write >= 0)
safe_read    = active ? read  : 0
safe_write   = active ? write : 0
compact      = state[safe_read].transpose(1,2).contiguous()   # [B,3,C]
local_index  = active ? arange(B) : -1
output       = public_op(..., compact, cache_indices=local_index)
state.index_copy_(0, safe_write, compact.transpose(1,2))
```

该 compact buffer 同时解决布局和双页语义。padding row 从 page 0 读取，公开 op 因 `-1` 跳过，回写值
仍等于原 page 0；即使多个 padding row 映射到 page 0，也不得改变其内容。公开 binding 的输出必须
zero-init，保证 varlen 尾部和跳过行不会把未初始化值送入后续 split/gate。

Prefill 现有准备逻辑通常已令 `read==write==state_out`，但公共 API 仍保留双索引，避免在调用点制造
第二套状态规则。Decode 的 ref 路径直接 gather `read`、计算、再 scatter `write`，不需要 compact
布局转换。

## 5. 数值实现

### 5.1 Decode graph-safe ref 路径

Decode 每个 request 只有一个 token：

```text
history = state[safe_read]                         # [B,C,3]
window  = cat(history, projected[...,None])        # [B,C,4]
value   = sum(fp32(window) * fp32(weight), dim=-1)
output  = silu(value + bias).to(projected.dtype)
next    = window[...,1:]
state.index_copy_(0, safe_write, next)
```

padding row 的 `next` 强制保持原 page-0 history，output 强制为零。整条路径只使用 tensor op，没有
`.item()`、device-to-host copy 或 Python token 循环，可被 fixed-BS NPU Graph 捕获。

### 5.2 Prefill small-batch ref 路径

Prefill 从 scheduler 已有的 host `cu_seqlens_cpu` 遍历 request，而不是读取 device boundary。每个
非空 request 将 `[C,3]` history 与 `[C,L]` 输入拼接，然后调用一次标准 depthwise `conv1d`：

```text
signal = cat(history, projected_segment.T, dim=-1)
value  = conv1d(signal[None], weight[:,None,:], groups=C)
output = silu(value).T
next   = signal[:,-3:]
```

这是 eager Prefill，不要求 graph；循环次数等于 request 数，不再按 token 循环。fresh request 在
conv 前用 `has_initial=false` 将 history 置零。

### 5.3 Prefill large-batch 公开路径

公开 op 一次消费 packed varlen batch，并执行 depthwise width-4 convolution、SiLU 与 compact state
更新。标准 `[C,4]` 权重只在该分支转为 contiguous `[4,C]`；不为此引入第二种 checkpoint 参数或
platform-specific model 字段。

Lite 当前每次 forward 都重新 `cat` 三个 conv weight。阶段 4B 复用模型 loader 已有的
`process_weights_after_loading` hook，把标准 `[C,4]` packed weight 缓存一次；未经过 loader 的小型
单测仍允许按原表达式构造 fallback。该缓存是公共 causal-conv 输入，不包含 NPU 类型或 vendor 状态。

## 6. Kernel selection

历史阶段 4B 曾使用以下阈值（已由第 10.7 节的默认融合策略替代）：

| 模式 | request 数 | 默认实现 | 原因 |
| --- | ---: | --- | --- |
| Decode | 任意 fixed BS | graph-safe ref | 当前 cache 布局下全范围更稳定，BS1/2/8/256 更快 |
| Prefill | `<16` | depthwise `conv1d` | 避免公开 op 的固定 host/tiling 开销 |
| Prefill | `>=16` | 公开 fused op | 多 request 时一次 varlen 调用明显减少 launch 数 |

当时 selection trait 固定为 `forward_mode={prefill,decode}` 和
`batch_class={small,large}`，large 定义为 `B>=16`。公开 artifact 不可用或 schema 校验失败时，large
Prefill 必须自动回退同一个 ref 实现；不能让可选 artifact 成为功能依赖。显式 kernel override
仍需 fail loud，便于测试指定实现。

阈值只依据阶段 4B 同口径 NPU0 A/B，不推导为其他 SoC 的通用结论。若后续完整服务 profile 显示
交叉点改变，应改 registration/selection 测试和本文，而不是在模型 forward 中散落条件分支。

## 7. 上板证据与决策

在 910B、TP8 rank-local `C=1536,W=4` 上得到：

### 7.1 正确性

- compact 公开 Prefill/Decode 对 FP32 oracle 的 output/state 最大绝对误差均为 0；
- `read!=write`、fresh history、empty segment、padding 和 page-0 isolation 通过；
- 非连续 state view 调用不报错但不修改源 pool，证明必须 compact/scatter；
- 公开 op eager 接受无 bias；Prefill 生产路径无需为 graph 创建零 bias 参数。

### 7.2 Decode NPU Graph（微秒/次）

| BS | ref | 公开 op + compact | 决策 |
| ---: | ---: | ---: | --- |
| 1 | 55--59 | 76--81 | ref |
| 2 | 77 | 93--95 | ref |
| 8 | 101--106 | 110--115 | ref |
| 64 | 166--176 | 152--159 | 不为单一交叉点增加生产分支 |
| 256 | 270--274 | 282--295 | ref |

公开 op eager 本体约 1.21--1.30 ms，compact 端到端约 1.35--1.54 ms；ref eager 为
0.16--0.27 ms。graph 后公开 op 只在 BS64 的孤立点领先，不能抵消双实现、布局搬运和边界测试成本。

### 7.3 Prefill eager（微秒/次）

| requests | tokens | depthwise `conv1d` | 公开 op + compact |
| ---: | ---: | ---: | ---: |
| 1 | 64 | 196 | 1417 |
| 4 | 250 | 732 | 1435 |
| 8 | 500 | 1449 | 1432 |
| 16 | 1000 | 2900 | 1441 |

单 request 下 token 数从 64 增至 1024 时，ref 路径仅从约 0.17 ms 增至 0.21 ms，公开 op 保持约
1.35--1.44 ms。阶段 3 的逐 token reference 在 T64 约 40--41 ms。因此阶段 4B 默认阈值取保守的
16，而不是在误差范围内的 8。

## 8. 测试矩阵

### 8.1 CPU/meta

- 统一 API 对 4-token、`[1,3]`、`[2,0,2]`、fresh/resume、`read!=write` 与 padding 对齐独立 oracle；
- 错误 width、channel、dtype、boundary、index shape 和重复 active write fail loud；
- 非 Ascend 或无注册 kernel 时保持阶段 3 portable reference；
- Lite post-load 只缓存一次标准 packed weight，参数名和 strict loader coverage 不变；
- selection 测试冻结 Decode=ref、Prefill B15=ref、B16=public，以及 artifact-missing fallback。

### 8.2 NPU0

- Prefill B1/B4/B16，包含变长、empty、fresh/resume、双页与 page-0 sentinel；
- Decode BS1/2/8/64/256 eager 与 NPU Graph 两次 replay；第二次更新 projected/index 内容，证明 graph
  不冻结 value；
- output、目标 state 与 FP32 oracle 在 BF16 阈值内，未选 state/page0/neighbor bitwise 不变；
- 所有 output/state finite；
- 强制 public/ref override 各自运行，默认 selection 与阈值一致；
- artifact 缺失时同一 focused case 自动回退且结果不变。

性能 A/B 只比较同步后的 steady-state；graph 单独报告 replay latency，不拿 eager host/tiling 开销冒充
设备 kernel 时间。

## 9. 准入与回退

实现提交准入要求：本地 focused、全量 `pre-commit run --all-files`、NPU0 exact-source focused 全部
通过；Prefill 大小 batch 和 Decode graph selection 命中本文；没有 NaN/Inf、page0 mutation 或
neighbor write；公开 op 缺失仍能运行。

若公开 large-Prefill 任一 correctness/availability 门禁失败，只删除其 performant registration，保留
统一 API 和 ref 路径；不得回退到阶段 3 逐 token NPU reference。若后续完整服务 profile 证明
large-Prefill 公开 op 无收益，再按同样方式移除 registration，不改 runtime 或 cache。

阶段 4B 明确不包含：

- Decode conv+gate+recurrent 的新融合 kernel；
- Prefill gate-cumsum/chunk KDA；
- output epilogue 融合；
- speculative verify/replay；
- Decode overlap、CP8/KVP8、PD one-copy 或 8P8D 服务。

## 10. 2026-09-02 Decode follow-up

本节取代本文对 Lite NPU Decode state layout 和默认实现的旧结论；Prefill 结论不变。旧结论对未修改
的公开 ABI 仍然正确：`[page,C,3]` 适配到 `[page,3,C]` 需要 gather、transpose、contiguous、公开算子和
scatter，BS32 profile 共 37 个小 kernel、289.052 us，不能成为优化。

### 10.1 最小 ABI 扩展

打包的 `CausalConv1d` 在 Prefill/Decode 共用独立读写索引，不再复用初态 mask 表示写页：

```text
cache_indices       [B] int32/int64   # read page
write_indices       [B] int32/int64   # write page, Prefill and Decode
initial_state_mode  [B] bool/int      # initial-history mask, Prefill only
```

Torch 使用统一的 `custom::npu_causal_conv1d` schema。底层新增独立 `writeIndices` 输入和
由 binding 从 tensor 推导的 `stateSlotStride` 属性；wheel 与 custom OPP 必须配套升级。
Prefill 可以同时传初态 mask 和写页，Decode 不接受初态 mask。TokenSpeed 不再 patch 或编译算子源码。

### 10.2 原生 state layout

共享 cache recipe 不按设备选择布局，而是在构造期查询通用 attention operator facade 的
`kda_causal_conv_state_layout()`。Ascend operator leaf 为 `flash_ops` kernel 返回 `[page,3,C]`；其他 backend
返回原有 `[page,C,3]`。因此 model/runtime forward 没有 device 分叉，物理布局与融合选择由最小算子
seam 持有。cache 字节数、arena packing、PD one-copy 生命周期都不变；width-major 布局的 PD partition
axis 为 1，global 三段仍是 Q/K/V 各 4096 channel。

缓存维度与 TP 分片继续取自 `LinearAttnConfig`；recipe 只按 operator 布局交换卷积状态轴。
KDA prefill 的 kernel selection 仍使用独立的 `kda_paged_prefill` mode。
TP logits 的 NVIDIA 自定义通信初始化只在 NVIDIA 上查询 CUDA capture；NPU 使用通用 gather。

ref fallback 接受两种 layout，并用 transpose view 复用原数学路径。未安装 `flash_ops` 或缺少
`custom::npu_causal_conv1d` schema 时自动回退 ref，因此正式算子包仍是可选依赖。

### 10.3 准入结果

910B、BF16、`C=1536,W=4` 的真实 `kda_causal_conv1d` 入口结果如下。时间为七轮 steady-state 中位数，
单位 us；当时默认 eager 因 host dispatch 开销走 ref，捕获期才走公开算子。
这份历史数据不代表当前默认策略；第 10.7 节已统一 eager/graph 默认融合。

| BS | ref eager | Public eager | ref graph | Public graph | Graph speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 182.93 | 1294.90 | 111.20 | 32.41 | 3.43x |
| 2 | 184.75 | 1294.00 | 145.25 | 34.07 | 4.26x |
| 8 | 179.07 | 1306.18 | 174.49 | 42.35 | 4.12x |
| 32 | 204.39 | 1315.17 | 210.79 | 48.45 | 4.35x |
| 64 | 354.37 | 1289.98 | 357.32 | 50.39 | 7.09x |
| 256 | 496.96 | 1294.05 | 512.75 | 64.29 | 7.98x |

六档 output、完整 state、独立 source/destination page 均 bitwise exact；padding output 为零，source、
page 0 和 neighbor 未改；BS2 graph replay 的 changed input/state 也 exact。`bias=None` 可直接捕获，
无需增加零 bias buffer。

备选 Triton single-kernel 已拒绝：BLOCK_C=1024 的六档 graph geomean 仅为 ref 的 0.37x，且 BS256
慢 6.61x。生产实现不保留 Triton 分支，也不保留 compact Decode 分支。

### 10.4 BS32 profiler 与 8P8D 服务验收

BS32 独立 profiler 的 11 轮 wall-time 中位数为 ref 209.929 us、公开入口 50.845 us，提升 4.129x。
设备侧每 replay 为 213.877 us/30 kernels 对 47.250 us/5 kernels，提升 4.527x；公开路径的主要 kernel
为 CausalConv1d 16.274 us、Transpose/InplaceCopy 18.085 us、ZerosLike 8.854 us 和两个 FillScalar
合计 4.036 us。它证明收益来自减少 graph replay 内的小算子，而不是 host 计时误差。

exact-source `ae846bee` 在 8P+8D 上完成 Prefill eager、Decode BS1/2/32 graph capture。32 个并发请求
各强制生成 32 token，全部返回 HTTP 200 和非空结果，共 1024 completion tokens；请求耗时范围
7.662--8.042 s。日志扫描无 Traceback、RuntimeError、HTTP 500 或 NaN/Inf，结束后相关端口、进程和
NPU holder 全部释放。

### 10.5 Artifact 完整性门禁与算子拆包

首次服务尝试暴露了一个独立于 causal-conv 数学的打包问题：artifact manifest 宣称包含六个公开算子，
但 `libcust_opapi.so` 实际只有 CausalConv1d 符号，Prefill 因缺少 `aclnnKdaGateCumsum` 在首个请求失败。
重新从锁定的上游与 CATLASS commit 全量构建后，所需 ACLNN API 及其 GetWorkspaceSize 符号均存在，
同一 8P8D 验收通过。

当前 causal-conv 已从 TokenSpeed 的 public source lock、patch、binding 和 artifact manifest 中移出，
由算子提供方独立发布 run 包与 `flash_ops` wheel。TokenSpeed 的 public KDA 构建脚本仍在发布
manifest 前用 `nm -D --defined-only` 校验 recurrent/prefill KDA 依赖闭包中每个算子的两类符号；
causal-conv 的 schema、eager 和 TorchAir graph 门禁则由正式算子仓负责。两个包互不冒充对方的完整性。

### 10.6 PR #2 重采数据（2026-09-03）

在 PR #2 单一共享模型入口上重新执行真实 `kda_causal_conv1d` A/B。910B、BF16、`C=1536,W=4`、
Decode NPU Graph 的七轮中位数如下：

| BS | ref graph (us) | Public graph (us) | Speedup |
| ---: | ---: | ---: | ---: |
| 1 | 122.41 | 37.66 | 3.25x |
| 2 | 154.03 | 39.44 | 3.91x |
| 8 | 184.56 | 47.46 | 3.89x |
| 32 | 222.05 | 53.42 | 4.16x |
| 64 | 374.89 | 54.95 | 6.82x |
| 256 | 521.52 | 70.50 | 7.40x |

六档 output/state 最大绝对误差均为 0；独立 read/write、padding、page 0、neighbor sentinel 和 changed
input replay 全部通过，实际选择为 `public_ascend_kda_causal_conv1d_decode`。

完整模型 profile 使用 8P8D、Decode graph BS32、NPU activity only、Level1 + PipeUtilization。八个 Decode
rank 都包含 31 个完整 forward 和 651 个 `CausalConv1d`；固定取前 30 个 forward 作为分析窗口，即每 rank
630 次调用。八份 `kernel_details.csv`、`op_statistic.csv`、`api_statistic.csv` 和 `trace_view.json` 均完整，
所有 trace JSON 可解析。

无 profiler TPOT 使用同一 8P8D 服务、32 并发、每请求固定生成 256 token、temperature 0、ignore EOS；
一次 warm-up 后测两轮。两轮 32/32 请求均为 HTTP 200 且每请求恰好返回 256 completion tokens：

| Variant | Round 1 TPOT (ms) | Round 2 TPOT (ms) | Mean TPOT (ms) | Mean output tok/s |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 192.344 | 191.783 | 192.063 | 163.677 |
| Optimized | 188.717 | 188.777 | 188.747 | 166.122 |

优化降低 3.316 ms/token（1.73%），output throughput 提升 1.49%。原 PR head `1928bfea` 的 cache-zero
Triton grid 在 BS32 会超出 device limit；为隔离 causal-conv 变量，baseline 只带同一 PR 中后续的
cache-zero grid 拆分修复，仍保留旧 causal-conv 与 channel-major state。profile-only 采集改动不进入
生产 PR。

### 10.7 布局与选核的调用合同

`C = 3 * local_heads * head_dim` 中的 3 表示 Q/K/V 三段通道；历史轴长度为 `W-1`，当前
`W=4` 时也恰好为 3。单个 cache field 不含 slot 轴，shape 为 `[C,W-1]` 或 `[W-1,C]`。
PD partition 从模型配置的历史长度确定 channel 轴，按 Q/K/V 三段分别分片，历史轴不分片。
replay workspace 按实际 QKV 通道数计费，与物理布局无关。

`ref_kda_causal_conv1d` 是 PyTorch 组合实现。它使用共享存储的 channel-major view；
`transpose` 仅交换 shape/stride，ref 的索引和写回按 stride 访问。融合 Decode 直接消费原生
width-major pool；融合 Prefill 同样直接消费该 pool，支持首轴跨槽间隔。只有 channel-major
兼容路径仍创建 compact buffer、转换内部布局并 scatter。仅对 view 做 contiguous 而不 scatter
会丢失状态更新。

| 调用条件 | 默认行为 |
| --- | --- |
| Prefill 任意 B，融合算子可用 | native pool 直接调用；channel-major 兼容路径 compact/scatter |
| Decode eager/capture，原生 layout 且算子可用 | 默认融合；capture 记录同一路径，replay 不重新执行 Python 选核 |
| 算子缺失，或 Decode layout 不匹配 | 自动选择时 fallback |
| 显式 `solution="ref"` / ref override | 调试对照，跳过融合实现 |

正常调用不需要 enable 开关：安装配套 wheel/OPP、能力标记和输入合同满足即可。
不再依据 Prefill batch 大小或 Decode 是否 capture 改变策略；保留版本、layout 和参数校验。
缺失可选包可 fallback，但任意算子执行错误不会被吞掉后改走 ref。
此简化统一启用行为，不承诺所有小 batch/eager 用例更快；历史性能表不作新策略性能证据。

显式 `solution="public_kda"` 或有效 override 会令 `require_public=True`，禁止上述静默 fallback；
缺算子或 Decode layout 不匹配时直接报错。override 优先级统一为环境变量、`kernel_override`
context、调用参数。环境变量为 `TOKENSPEED_KERNEL_OVERRIDE_ATTENTION_KDA_CAUSAL_CONV1D`，
可设为注册名 `public_ascend_kda_causal_conv1d_decode`。同一 override 解析也用于 KDA Prefill core，
因此选核和强制执行策略不会分离。`require_public` 本身是 kernel facade 下传的参数，不是模型配置。

cache-zero 是独立的初始化操作：`CacheArena.zero_blocks()` 将新分配块转为字节区间，再交给
`_zero_byte_ranges_kernel` 清零。NPU 分批控制 launch grid，`BLOCK_OFFSET` 指定当前批次在区间内的
块偏移；该操作不参与 causal-conv 的数值计算，也不改变 cache layout。

## 11. 首轴 stride 合同

Ascend 原生 state 的 shape 为 `[slots,W-1,C]`，stride 为 `[S,C,1]`，其中
`S >= (W-1)*C`。真实 `CacheArena` 同一 slot 还包含其他字段，因此通常严格大于；
页内连续不等于整个 state tensor 连续。算子原地寻址为 `slot*S + history*C + channel`，
storage offset 由 tensor 指针处理。首轴间隙、其他层 state 与 recurrent state 不可修改。

原生 Prefill 删除仅为算子服务的 gather/compact/scatter；Decode 直接传原 arena view。
权重 `[C,W] -> [W,C]` 连续化、channel-major 兼容转换，以及 runtime checkpoint/work-page
准备仍保留，它们不是首轴 stride 问题。活跃写页须唯一；允许同一请求 `read==write`，
不允许一条请求覆盖同批其他请求仍需读取的页。

loader 检查 `flash_ops.CAUSAL_CONV1D_STATE_SLOT_STRIDE`，旧包自动走 ref；显式指定融合算子
则报错提示配套更新 wheel/OPP，避免旧同名 schema 静默按 dense 页步长写错位置。

非连续首轴支持 eager 与 TS 原生 `NPUGraph`。当前 TorchAir GE 在图输入阶段会执行
`.contiguous()`，因此 converter 对非连续 state 明确报错，不会用原 stride 访问复制后的存储；
连续 state 的 GE 路径保留。此限制不能解释成 TorchAir 支持非连续 state。

回归入口 `test_npu_causal_conv_real_arena` 使用实际 recipe/pool 的 Lite 3B/Large × TP1/2/4/8，
覆盖 Prefill fresh/resume/empty/padding、独立读写及同页更新、Decode changed-input/page graph
replay，并逐字节检查整个 arena。单算子额外覆盖非零 storage offset 和非法页内 stride。

### 10.9 默认融合与 ref 补测（2026-09-08）

小算子拼接模式统一为 `solution="ref"`，注册名为 `ref_ascend_kda_causal_conv1d`。
默认启用策略不变。补测覆盖 BF16/FP16、两 Lite 尺寸×TP1/2/4/8 去重后的5档C、
B1/B16、Prefill eager / Decode eager / Decode graph，共60组；配套122项回归通过。
完整口径、真实arena stride及结果见[本轮 fused/ref 报告](lite-npu-causal-conv-fused-ref-perf.md)。
本轮设备统计包含 ref 的 AI_CPU ViewCopy；不能把调用段收益等同于纯卷积核或整模型收益。
