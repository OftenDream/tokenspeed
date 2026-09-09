# Lite causal-conv 默认融合与 ref 性能补测

2026-09-08，TS 默认策略基于 `b2406595`（本次仅统一 ref 命名，不改变启用策略），
配套 flash-npu-kernel PR21 `16fe26f0` 的 910B OPP 与同 ABI wheel。
ref 指生产小算子拼接实现，不是 CPU oracle，也不是另一版融合算子。

## 测量口径

- 单张 Ascend 910B2C，物理 NPU0；CANN 9.0.0、PyTorch 2.9.0、torch-npu 2.9.0.post2。
- 默认 facade（未传 solution）对照 `solution="ref"`；width=4、SiLU、无 bias。
- BF16/FP16 × 5 档本地 packed C × B1/B16 × Prefill eager / Decode eager / Decode NPUGraph，共 60 组。
  Prefill 每请求64 token，总 T=B×64；Decode 每请求1 token。性能输入均为 resume、无 padding。
- 双路径共享相同输入/权重和真实 `CacheArena` view，独立读页1..B、写页B+1..2B。
  每路径先对四 tap FP32 oracle 验证输出（atol=1e-4、rtol=0.03）与完整 conv_state（bitwise），
  再改变输入/读页并加入 padding 检查；graph 不重新 capture。正确性后恢复无 padding 才计时。
- 调用段 wall：各路径20次 warmup；7轮交替先后顺序，每轮 eager20次或 graph200次，轮前后同步，
  取每轮单次均值的中位数。包括 facade/host 调度（eager）、权重转置、输出 zero-init、
  ref 的 gather/scatter 等，不含输入生成、state reset、graph capture。
- 设备 kernel 总时间：每个 case/路径独立 profiler Level1，warmup=5、active=5；
  `kernel_details.csv` 的 AI Core/Vector **及 AI_CPU** duration 总和除以5。不是 wall，也不只取卷积核。
  每份 fused trace 恰有5次 CausalConv1d，ref 为0次，证明默认路径实际融合。
  原始采集完成后统一以包含 AI_CPU 的解析函数重新汇总，wall 样本未改。
- 这是单卡本地 TP 几何矩阵，不是多卡通信、完整 KDA 或整模型吞吐测试；
  不推断更长 Prefill、其他 batch/arena 容量都具有相同收益。

## 两个 Lite 尺寸与本地 C 映射

相同本地几何去重；1536/3072 等是 TP 后的 packed channel，并非模型尺寸超参数。

| 模型 | TP1 | TP2 | TP4 | TP8 |
| --- | ---: | ---: | ---: | ---: |
| Lite 3B，32 heads | 12288 | 6144 | 3072 | 1536 |
| Lite Large，64 heads | 24576 | 12288 | 6144 | 3072 |

## State stride 与对齐

没有人工指定 padding 倍数；stride 直接来自 recipe/pool。每个 case 都断言指针、slot 起点与 history 行起点按256B对齐。
state 为 `[2*B+3,3,C]`、stride `[S,C,1]`；本轮 state storage offset=0。
BF16/FP16 每元素2字节，页内连续；slot 间隙中存在 recurrent/其他层字段，不是可任意覆盖的 padding。

| 本地 C | slot stride（字节） | 单页 conv 有效数据（字节） |
| ---: | ---: | ---: |
| 1536 | 294912 | 9216 |
| 3072 | 589824 | 18432 |
| 6144 | 1179648 | 36864 |
| 12288 | 2211840 | 73728 |
| 24576 | 4423680 | 147456 |

NPU Prefill 与 Decode 的 cache recipe 统一分配 `[slots, W-1, C]`，不按 P/D 角色切换布局。
融合入口统一校验这一形状和 stride `[S, C, 1]`；slot 之间允许间隙，slot 内连续。
已移除 channel-major Prefill 的 gather/transpose/contiguous/scatter 兼容分支，
P/D 均直接传原 arena view。缺少融合包时仍可使用支持该布局的 ref 实现。
权重从 `[C, W]` 转为算子要求的连续 `[W, C]` 是另一处转换，予以保留。
P/D transfer schema 对 conv 字段沿 channel axis 1 切分；本次未改变 arena 或传输格式。

## 汇总与结论

60/60 组 output/state/changed-input 检查通过；调用段 fused 更快60组、ref更快0组。
全量调用段 ref/fused 几何平均为 **33.00x**。
以下加速比均为 ref/fused，>1 表示 fused 更快；每行覆盖5档C×B1/B16。

| DType | 模式 | 调用段加速比（几何平均） | 最小–最大 | fused 更快 |
| --- | --- | ---: | ---: | ---: |
| BF16 | Prefill eager | 64.85x | 6.52–670.95x | 10/10 |
| BF16 | Decode eager | 20.78x | 4.45–151.52x | 10/10 |
| BF16 | Decode graph | 27.47x | 7.28–185.51x | 10/10 |
| FP16 | Prefill eager | 63.45x | 6.55–659.03x | 10/10 |
| FP16 | Decode eager | 21.15x | 4.41–160.04x | 10/10 |
| FP16 | Decode graph | 25.98x | 7.39–171.39x | 10/10 |

按 dtype 汇总：BF16 30/30 更快，几何平均 **33.33x**；FP16 30/30 更快，几何平均 **32.67x**。

- 本矩阵未见小 batch Prefill 或 eager Decode 回退，支持当前统一默认启用策略；不再增加 batch/capture 门槛。
- BF16/FP16 趋势一致。大C/B16的高比值包括消除 ref 的 AI_CPU `aclnnInplaceIndexCopy_ViewCopyAiCpu_ViewCopy`，不是纯卷积核算力提升：例如 FP16/C24576/B16/Decode eager 中该项占 ref 设备 kernel 总时间96.3%。
- 融合直接寻址 strided state；ref 经 channel-major view 做索引写回，profiler 记录额外 ViewCopy。Prefill 逐请求执行该写回，开销随B增加；因此结果依赖真实 arena 的形状/stride/容量，不能拿连续小pool的旧数据混比。
- wall 与 profiler 为独立采样；profile有采集扰动，kernel duration之和也不等于端到端延迟。未重跑整模型服务，不能把上述比值解释成模型吞吐提升。

## 全量性能矩阵

单位 μs；设备列含 AI_CPU。Prefill T=B×64；Decode T=B。所有行正确性 PASS。

| DType | C | B | 模式 | fused 调用段 | ref 调用段 | 加速比 | fused 设备总计 | ref 设备总计 |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| BF16 | 1536 | 1 | Prefill eager | 65.04 | 424.03 | 6.52x | 41.24 | 429.95 |
| BF16 | 1536 | 1 | Decode eager | 55.74 | 248.04 | 4.45x | 20.97 | 250.41 |
| BF16 | 1536 | 1 | Decode graph | 34.30 | 249.62 | 7.28x | 29.91 | 270.32 |
| BF16 | 1536 | 16 | Prefill eager | 91.92 | 13302.51 | 144.71x | 76.62 | 13088.91 |
| BF16 | 1536 | 16 | Decode eager | 58.21 | 767.53 | 13.19x | 31.66 | 779.62 |
| BF16 | 1536 | 16 | Decode graph | 45.95 | 770.97 | 16.78x | 41.47 | 715.38 |
| BF16 | 3072 | 1 | Prefill eager | 63.83 | 549.33 | 8.61x | 41.70 | 532.27 |
| BF16 | 3072 | 1 | Decode eager | 56.89 | 340.76 | 5.99x | 22.12 | 369.29 |
| BF16 | 3072 | 1 | Decode graph | 36.31 | 345.11 | 9.50x | 31.63 | 363.81 |
| BF16 | 3072 | 16 | Prefill eager | 104.50 | 24742.27 | 236.77x | 86.45 | 24872.97 |
| BF16 | 3072 | 16 | Decode eager | 57.46 | 1409.79 | 24.53x | 39.97 | 1460.58 |
| BF16 | 3072 | 16 | Decode graph | 50.66 | 1424.75 | 28.12x | 46.18 | 1332.36 |
| BF16 | 6144 | 1 | Prefill eager | 64.94 | 777.75 | 11.98x | 56.37 | 765.68 |
| BF16 | 6144 | 1 | Decode eager | 56.95 | 504.69 | 8.86x | 25.94 | 481.94 |
| BF16 | 6144 | 1 | Decode graph | 40.63 | 507.29 | 12.48x | 36.15 | 509.50 |
| BF16 | 6144 | 16 | Prefill eager | 138.70 | 42945.95 | 309.63x | 126.10 | 43542.59 |
| BF16 | 6144 | 16 | Decode eager | 69.21 | 2500.43 | 36.13x | 40.18 | 2349.34 |
| BF16 | 6144 | 16 | Decode graph | 52.35 | 2488.86 | 47.54x | 47.75 | 2526.86 |
| BF16 | 12288 | 1 | Prefill eager | 70.08 | 1288.59 | 18.39x | 65.86 | 1372.03 |
| BF16 | 12288 | 1 | Decode eager | 56.73 | 907.09 | 15.99x | 35.89 | 951.51 |
| BF16 | 12288 | 1 | Decode graph | 45.39 | 892.05 | 19.65x | 40.84 | 1053.07 |
| BF16 | 12288 | 16 | Prefill eager | 176.87 | 83627.36 | 472.82x | 156.15 | 84391.61 |
| BF16 | 12288 | 16 | Decode eager | 64.61 | 4871.32 | 75.39x | 41.90 | 4911.22 |
| BF16 | 12288 | 16 | Decode graph | 51.82 | 4909.45 | 94.75x | 47.35 | 4354.69 |
| BF16 | 24576 | 1 | Prefill eager | 75.59 | 2391.43 | 31.63x | 71.68 | 2464.69 |
| BF16 | 24576 | 1 | Decode eager | 57.14 | 1697.73 | 29.71x | 34.61 | 1786.90 |
| BF16 | 24576 | 1 | Decode graph | 47.26 | 1728.66 | 36.58x | 42.79 | 1689.76 |
| BF16 | 24576 | 16 | Prefill eager | 262.21 | 175929.36 | 670.95x | 231.73 | 182311.84 |
| BF16 | 24576 | 16 | Decode eager | 69.65 | 10553.36 | 151.52x | 47.71 | 9840.82 |
| BF16 | 24576 | 16 | Decode graph | 56.18 | 10422.60 | 185.51x | 51.65 | 11069.30 |
| FP16 | 1536 | 1 | Prefill eager | 64.36 | 421.25 | 6.55x | 42.02 | 426.08 |
| FP16 | 1536 | 1 | Decode eager | 57.65 | 254.40 | 4.41x | 19.87 | 253.18 |
| FP16 | 1536 | 1 | Decode graph | 34.15 | 252.42 | 7.39x | 29.81 | 255.34 |
| FP16 | 1536 | 16 | Prefill eager | 94.65 | 13353.38 | 141.08x | 80.26 | 13336.34 |
| FP16 | 1536 | 16 | Decode eager | 57.59 | 765.78 | 13.30x | 32.04 | 801.61 |
| FP16 | 1536 | 16 | Decode graph | 49.82 | 758.88 | 15.23x | 45.49 | 898.32 |
| FP16 | 3072 | 1 | Prefill eager | 63.91 | 545.29 | 8.53x | 46.06 | 562.62 |
| FP16 | 3072 | 1 | Decode eager | 58.00 | 339.74 | 5.86x | 22.37 | 345.91 |
| FP16 | 3072 | 1 | Decode graph | 36.36 | 345.26 | 9.50x | 32.06 | 345.97 |
| FP16 | 3072 | 16 | Prefill eager | 107.47 | 24427.36 | 227.30x | 89.47 | 24580.35 |
| FP16 | 3072 | 16 | Decode eager | 56.75 | 1429.67 | 25.19x | 33.96 | 1414.80 |
| FP16 | 3072 | 16 | Decode graph | 52.11 | 1411.02 | 27.08x | 47.61 | 1467.64 |
| FP16 | 6144 | 1 | Prefill eager | 65.57 | 764.13 | 11.65x | 57.16 | 759.68 |
| FP16 | 6144 | 1 | Decode eager | 57.03 | 506.46 | 8.88x | 26.36 | 534.69 |
| FP16 | 6144 | 1 | Decode graph | 41.57 | 503.55 | 12.11x | 37.19 | 490.62 |
| FP16 | 6144 | 16 | Prefill eager | 140.28 | 42160.01 | 300.55x | 129.02 | 42867.71 |
| FP16 | 6144 | 16 | Decode eager | 59.50 | 2504.07 | 42.09x | 45.18 | 2665.28 |
| FP16 | 6144 | 16 | Decode graph | 55.65 | 2525.76 | 45.38x | 50.79 | 2544.95 |
| FP16 | 12288 | 1 | Prefill eager | 68.30 | 1252.91 | 18.35x | 62.54 | 1310.65 |
| FP16 | 12288 | 1 | Decode eager | 57.56 | 893.25 | 15.52x | 35.01 | 848.41 |
| FP16 | 12288 | 1 | Decode graph | 50.21 | 893.97 | 17.80x | 45.83 | 826.71 |
| FP16 | 12288 | 16 | Prefill eager | 176.86 | 83329.62 | 471.16x | 157.52 | 84217.34 |
| FP16 | 12288 | 16 | Decode eager | 59.30 | 4784.38 | 80.68x | 45.37 | 4531.31 |
| FP16 | 12288 | 16 | Decode graph | 55.87 | 4869.40 | 87.16x | 51.57 | 4711.85 |
| FP16 | 24576 | 1 | Prefill eager | 78.59 | 2326.68 | 29.60x | 76.83 | 2300.74 |
| FP16 | 24576 | 1 | Decode eager | 58.65 | 1619.58 | 27.61x | 32.80 | 1736.19 |
| FP16 | 24576 | 1 | Decode graph | 49.90 | 1654.25 | 33.15x | 45.49 | 1606.35 |
| FP16 | 24576 | 16 | Prefill eager | 261.11 | 172080.00 | 659.03x | 232.56 | 168833.41 |
| FP16 | 24576 | 16 | Decode eager | 62.69 | 10032.99 | 160.04x | 50.42 | 10520.47 |
| FP16 | 24576 | 16 | Decode graph | 59.67 | 10226.39 | 171.39x | 55.54 | 9448.88 |

## 复现与回归

`benchmark_kda_causal_conv_cases.jsonl` 是本 NPU benchmark 的 60 组输入清单，
由相邻的 `benchmark_kda_causal_conv.py --cases` 显式读取，不是 pytest 自动收集格式，
也不是 TS 通用 benchmark 的既有 case 规范。保留它用于复现上表的固定矩阵；
正确性回归仍由 `test_kda_causal_conv.py` 承担。

安装匹配的 stride-capable flash_ops wheel/OPP，并加载 CANN/OPP 环境及 TS 运行时测试依赖。
从仓库根目录运行（`perf-out` 必须为新输出目录）：

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python tokenspeed-kernel-npu/test/benchmark_kda_causal_conv.py \
  --cases tokenspeed-kernel-npu/test/benchmark_kda_causal_conv_cases.jsonl \
  --output perf-out --profile

ASCEND_RT_VISIBLE_DEVICES=0 python -m pytest --import-mode=importlib -q \
  tokenspeed-kernel-npu/test/test_kda_causal_conv.py \
  test/runtime/test_lite_hybrid_cache.py \
  tokenspeed-kernel/test/test_selection.py
```

最终回归122 passed，包括两 Lite×TP1/2/4/8 的16项真实 arena测试、原生图回放，以及默认真实调用计数。
显式 `solution="ref"` 和 `override="ref_ascend_kda_causal_conv1d"` 均验证不调用融合算子。
本次只改 causal-conv 的 ref 命名，其他 KDA core 的 solution 未改；启用策略与算子二进制未改。
Black/isort/diff 检查通过；未声称本轮全量 pre-commit 或完整模型服务验证。
