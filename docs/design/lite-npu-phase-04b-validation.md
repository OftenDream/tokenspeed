# Lite NPU 阶段 4B：Causal-Conv 验证记录

此文件是 `0bbab6d5` 的历史实验记录，不代表当前选核策略或性能。
小算子拼接模式统一称为 ref；当前策略见
[设计文档 10.7 节](lite-npu-phase-04b-causal-conv.md#107-布局与选核的调用合同)。

## 1. 验证对象

本记录验证阶段 4B 实现提交 `0bbab6d5`。范围限定为 Lite width-4 packed QKV causal-conv：统一
kernel API、Ascend selection、Decode graph-safe ref 路径、Prefill depthwise-conv/公开融合路径、
双 page state 发布、可选 artifact 回退和 Lite conv weight post-load packing。

本阶段不验证 recurrent/chunk KDA、output epilogue、完整模型服务、CP8/KVP8 或 PD one-copy；这些
能力仍按总计划在后续阶段独立准入。

目标环境为单张 Ascend 910B、CANN 9.0.0、PyTorch 2.9.0 和 `torch_npu` 2.9.0.post2。所有算子与
graph 测试只使用 NPU0，未启动模型服务，也未占用其余设备。

## 2. Exact-source 与 artifact

从 `0bbab6d5` 生成全新 `git archive`，并对 archive 内全部 tracked 文件逐项执行 SHA-256 对比，结果
全部一致。公开 OPP/binding 在阶段 4B 候选源码上重新 clean build；其中本阶段修改的 binding 源码与
exact archive 字节一致。构建完成后再把 package-local artifact 搬入 exact archive，没有复制候选
Python/runtime 源码。

artifact 重新加载后的 schema 集合严格为：

```text
causal_conv1d
recurrent_kda
kda_gate_cumsum
chunk_kda_fwd
```

manifest 中 binding 与 vendor op-api digest 复验通过。`causal_conv1d` binding 的 output 改为
zero-init 后重新编译，避免 varlen 跳过区域向后续算子暴露未初始化值。

## 3. 功能与数值

### 3.1 Selection 与回退

真实 Ascend registry 的默认选择为：

| 模式 | batch class | 默认 kernel |
| --- | --- | --- |
| Decode | small/large | ref（历史拼接路径） |
| Prefill | small，B<16 | ref（历史拼接路径） |
| Prefill | large，B>=16 | `public_ascend_kda_causal_conv1d` |

无 artifact 时，large Prefill 自动执行同一 ref 实现；显式指定 `public_kda` 时则直接报错，未出现
静默降级。统一 API 同时拒绝错误 width、channel、dtype、index/mask shape 和跨设备 tensor。

### 3.2 Prefill

生产 TP8 rank-local shape 固定为 `C=1536,W=4`。NPU focused 覆盖 B1、B4、B16、变长、empty、
fresh/resume、`read!=write`、page 0 与未选邻页。B16 默认公开路径和显式 ref 路径满足
`atol=3e-2,rtol=3e-2`；两条路径写出的最后三个 raw projection state bitwise 一致，page 0 与邻页
bitwise 不变，output/state 均 finite。

Lite 的三路 checkpoint conv 权重在 `process_weights_after_loading` 中只打包一次为标准 `[C,4]`；
未经过 loader 的小型测试会首次 forward 时建立同一缓存。参数名、strict loader coverage 和 state
dict 均未改变。

### 3.3 Decode graph

Decode 使用 tensor-only gather/window/FP32 reduce/scatter，没有 `.item()`、D2H 或 token Python 循环。
BS2 NPUGraph 完成 capture 和两次 replay：第二次 replay 前同时更新 projection、read page 和 write
page，graph output 与 eager ref bitwise 一致，目标 state pool bitwise 一致，page 0 与邻页不变。

NPU Graph capture 只记录、不执行首轮计算；测试在 capture 后先 replay，再比较结果。该行为是测试
协议修正，不是 kernel fallback，也没有在生产路径加入同步。

### 3.4 跨 Prefill/Decode 数值

阶段 3 的逐 token reference 使 Prefill/Decode 使用完全相同的算术顺序，因此旧测试要求 bitwise
相等。阶段 4B 的 small-Prefill 标准 `conv1d` 与 Decode FP32 reduce 使用不同 kernel，实测最终 KDA
output 最大绝对差为 `7.63e-6`；recurrent state 最大绝对差为 `3.83e-4`，只有约 0.7% 元素超过
`3e-5`。

门禁因此调整为：

- raw conv state 和未选页继续要求 bitwise；
- Prefill/Decode output 要求 `atol=3e-5,rtol=3e-5`；
- 两条 recurrent state 要求 `atol=5e-4,rtol=5e-2`；
- Prefill 和 Decode state 仍分别对独立 FP32 oracle 满足 `atol=3e-2,rtol=3e-2`。

这保留了独立正确性检查，没有用跨 kernel bitwise 假设替代 oracle。

## 4. 性能 A/B

最终 exact-source 代码使用同步后的 steady-state 平均值，单位为微秒：

| 场景 | ref | 公开融合 | 默认选择 |
| --- | ---: | ---: | --- |
| Prefill B8/T500 | 1611.6 | 1364.7 | ref |
| Prefill B16/T1000 | 3278.4 | 1357.2 | 公开融合 |

同一最终代码的 Decode BS2 NPUGraph replay 为 `80.48 us`。它与阶段 4B 设计阶段测得的约
`77 us` 同量级；没有为了 BS64 的孤立交叉点增加第二条 Decode 生产分支。

B8 本次采样中公开路径领先，但设计阶段的重复采样为近似持平，说明交叉点对负载和测量波动敏感。
本阶段仍保守保持 B16 阈值；若完整服务 profile 稳定证明 B8 获益，再单独调整 registration trait 和
对应门禁，不在模型 forward 中加入额外分支。

## 5. 测试汇总

| 环境 | 结果 |
| --- | --- |
| 本地 causal-conv/Lite focused | `12 passed, 5 skipped`；skip 均为 NPU-only |
| 910B exact-source causal-conv/Lite | `17 passed` |
| 910B exact-source selection boundary | `3 passed, 66 deselected` |
| 全仓 `pre-commit run --all-files` | 全部通过 |

NPU focused 包含真实公开 artifact、B16 public、missing-artifact fallback、显式 override、production
shape、NaN/Inf 回归、Decode graph replay、page sentinel、neighbor isolation 和 post-load packing。

## 6. 结论与回退

阶段 4B 准入通过。Lite featurewise-beta KDA 已从逐 token causal-conv reference 切换到统一 kernel
边界：Decode 固定使用 graph-safe ref，small-Prefill 使用标准 depthwise conv，large-Prefill 使用
公开融合 op；可选 artifact 缺失不影响正确性 baseline。

若后续服务验证发现公开 large-Prefill 路径不稳定，只需删除其 performant registration；runtime、
cache、Lite model 和 ref fallback 均无需修改。下一阶段按既定顺序准入公开 recurrent Decode。
