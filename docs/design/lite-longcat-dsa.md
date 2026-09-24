# Lite LongCat DSA

Lite directly instantiates the existing `LongCatDSAAttention`; the checkpoint's
`is_longcat_dsa` architecture fact selects independent per-layer indexing,
rather than defining another model class. It reuses
the packed Q/KV down projection, head-parallel Q/KV/output projections,
interleaved YaRN RoPE, `LongCatDSAIndexer`, and absorbed KV weight preparation.
The Lite decoder owns the input communication and output reduction: the
independent forward must not repeat the paired path's input gather.

The single attention class and shared indexer live in `models/longcat_dsa.py`;
there is no Lite-specific attention class, model file, or attention config.
Lite reuses `DSAConfig`; `uses_independent_index_cache` selects independent
per-layer indexing. GPU quantizes Index-K to packed FP8 with FP32 scales;
Ascend retains BF16 Index-K. Projection dtype does not select cache dtype.
Execution keeps the existing `DSABackend` contract and graph lifecycle. The
common/CUDA implementation remains in
`backends/paged/dsa.py`; the Ascend implementation is the
`AscendDSABackend` subclass in `backends/paged/ascend_dsa.py`. The canonical
backend name is `dsa`; `longcat_dsa` remains a compatibility alias selected by
the same registry.

## Selection and numerical contract

- Each full-attention layer owns an independent indexer (`cli_factor=1`).
  There is no cross-layer TopK reuse. KDA layers are unchanged.
- Checkpoints without `is_longcat_dsa` keep the original owner/consumer API
  and default behavior. Output gating and Q/KV low-rank scale flags come from
  the Lite architecture config; post-load preparation is a no-op for paired
  models.
- Existing paired LongCat models retain their private packed indexer class.
  Lite's canonical indexer uses BF16 Q/K projections and FP32 inputs and
  weights for its score projection, then rounds that result to BF16 for
  LightningIndexer. No runtime projection-mode string or extra head/dimension
  scale is involved.
- Index K uses RMSNorm and rotates the first rotary channels. Attention
  rotates the tail rotary channels. YaRN allocation follows the model's
  device context; it must not hard-code CUDA.
- Initial and local candidates are included in TopK. Sparse indices are
  request-local token positions, not arena-global token slots.
- `DSAConfig` is the single source for `index_init_tokens` and
  `index_local_tokens`. `DSABackend` publishes that policy through paged and
  hybrid wrappers; both the paired GPU selector and Ascend selector consume
  those same values at runtime.
- Lite's output gate and Q/KV low-rank normalization scales are retained.
  Post-load normalization scale preparation is idempotent.

## Cache and kernel boundaries

The existing hybrid KDA recipe owns all persistent state. Full-attention
pages carry BF16 `latent_kv` (NoPE and RoPE) and `dsa_index_k` (packed FP8
plus FP32 scales on GPU, BF16 on Ascend). They share scheduler page identities but occupy separate
physical planes, because the sparse kernels require exact contiguous page
strides. There is no backend-private persistent cache.

KDA state can share latent planes, not the separate index planes. For this
DSA layout the recipe bounds parent padding at 1.0 rather than MLA's 0.25;
this is an explicit memory reservation, not additional live cache tokens.
Existing MLA/draft packing policies are unchanged. Runtime capacity is still
computed from the actual arena layout and byte budget.

Runtime calls the `tokenspeed-kernel` facade. As with `dsa/cuda.py`,
`dsa/ascend.py` registers the Ascend solutions and adapts the optional NPU
extension. The NPU package owns only device primitives: cache scatter,
LightningIndexer, local-index selection, SparseFlashAttention and partial
merge. The model-independent `attention/dcp` package owns cyclic page
placement and virtual-to-local translation. DeepSeek V4 and DSA call the same
`refresh_dcp_page_table_metadata` constructor; V4 consumes its local physical
table, while DSA derives LightningIndexer's compact replicated-cache ABI from
the constructor's owner mask. Neither backend computes page ownership itself.
The common `DSABackend` remains free of device-specific CP scheduling. The
Ascend backend owns its pointer-stable compact indexer metadata, collective
state, A2A/AG, global TopK, O/LSE exchange and stream overlap. It also owns
the auxiliary stream, graph/capture checks, and per-stream core budgets for
overlapping Indexer and MLA projections. The model submits two logical
projection callables through the backend contract and carries no device
scheduling state. The existing
`forward_sparse_prefill` and `forward_sparse_decode` interfaces are unchanged;
the independent Ascend path enters its private CP scheduler directly instead
of adding an opaque selection argument to either interface. Missing operator
bindings fail at construction. A matching Python extension and custom OPP
vendor containing both operators must be installed; finding a Python schema
alone does not verify the vendor binaries.

Decode CP preserves the scheduler page's natural cyclic owner; it does not
move every live tail page to rank 0. The current independent Ascend path admits
one query token per request, so every local partial uses causal mode 3: at
`q_len == 1` its visible length is the complete rank-local KV length. A future
multi-token draft path must represent the per-request tail owner explicitly
instead of introducing a rank-0 convention.

The common backend continues to receive global-slot TopK indices and use the
existing `forward_sparse_prefill` / `forward_sparse_decode` path. The Ascend
subclass receives LongCat indexer projections, selects request-local indices,
and exposes only its native `forward_extend` / `forward_decode` execution.
Inherited chunked-prefill and externally selected sparse entry points fail
explicitly instead of falling back to a common dense or GPU implementation.
The Ascend subclass reuses the same dense-leaf metadata contract. Only the original FP8
indexer requires `dsa_plan`; the BF16 Ascend indexer consumes lengths and page
tables directly. Prefill retains the full-history table even with an empty
prefix. NVIDIA/AMD dense delegates and precomputed-TopK execution are
unchanged. The Ascend adapter currently admits only the LongCat BF16 contract,
not every DSA checkpoint/cache format.

## Admission and validation

Select `--attention-backend dsa`, or let the indexer checkpoint fields
select it automatically. The current integration requires BF16 cache, CLI=1,
16 index heads of width 128, TopK=2048, and equal attention/head-projection TP.
MTP is explicitly rejected. Initial plus local candidates must fit in TopK.
The existing LongCat GPU path keeps its default packed projection and
pair-local selection behavior.

Focused tests:

```bash
python -m pytest test/runtime/models/test_lite_longcat_dsa.py \
  test/runtime/models/test_longcat_lsa.py \
  test/runtime/test_kimi_k3_cache_spec.py -q
```

Coverage includes independent indexer ownership, FP32 score projection,
rotary slice handling, TP4/8/16 cache-plane packing, and existing LongCat/MLA
contracts. Backend regressions also cover the shared DSA leaf's registration,
pointer-stable decode metadata, mixed-batch request offsets, no-prefix and
cached-prefill tables, and the unchanged GPU indexer-plan contract.
`tokenspeed-kernel-npu/test/test_longcat_dsa.py` separately validates the
device-only adapter. Runtime tests validate the Ascend backend's CP page
compaction and pointer-stable buffers separately from device kernels.

On an allocated NPU with matching operator bindings and vendor binaries:

```bash
TOKENSPEED_TEST_LONGCAT_DSA_NPU=1 python -m pytest \
  test/runtime/models/test_lite_longcat_dsa.py \
  -k npu_dsa_matches_explicit_paged_pipeline -q
```

These three physical-kernel tests compare `AscendDSABackend` with the
original explicit scatter/index/attention sequence: decode, no-prefix prefill,
and cached prefill. Decode additionally captures an NPUGraph and replays with
changed lengths and query payloads. Outputs and written cache planes match
bit for bit in these small fixtures. This is backend-wiring regression
coverage, not an independent mathematical accuracy or full-model benchmark.

A 16-rank W8A8 full-model decode fixture completed native graph capture,
warmup, profiler-off timing, and a shape-recorded rank-0 profile at attention
TP8, DCP8, DP2, MoE EP16/G8, per-replica batch 8, and 64K synthetic KV. This
is execution and performance validation, not trained-model quality validation.
Prefill and mixed prefill/decode still require separate long-context numerical
qualification.

See `test/ci_system/README_LITE_LONGCAT_DSA_DECODE.md` for the reproducible
environment, launch, shutdown, and profile-inspection commands.
