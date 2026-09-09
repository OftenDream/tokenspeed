# Lite NPU Phase 12: KDA fusion plan

> The 2026-09-02 Decode follow-up in Phase 04B supersedes this plan's old
> `[page,C,3]` Lite NPU conv-state assumption; the Phase 12 fusion boundaries are unchanged.

## 1. Scope

Phase 12 optimizes the Lite KDA path after the functional NPU, graph, overlap, cache, and service
contracts have already been admitted. It is split into three independently designed, implemented,
validated, committed, and reversible subphases:

1. 12A merges the six hidden-state projections into one GEMM;
2. 12B measures and, only if beneficial, removes the remaining packed-QKV materialization before
   causal convolution;
3. 12C evaluates a fused featurewise-beta prepare for Prefill and a featurewise-beta recurrent
   kernel for Decode.

This phase does not change KDA mathematics, KDA/MLA layer placement, TP8, cache layout, PD one-copy,
graph ownership, Grouped MoE, OE host residency, Weight-NZ, or the supported 8P8D topology. It does
not introduce a second cache, graph executor, state layout, operator registry, or dependency.

Each subphase has its own design and validation record. A candidate that fails correctness, graph,
or performance admission leaves no partial production implementation; its negative result is
recorded instead.

## 2. Current production data flow

For the target TP8 rank, a Lite KDA layer receives BF16 hidden states `[T,3072]` and currently runs:

```text
q          = linear(hidden, q_weight)       # [T,512]
k          = linear(hidden, k_weight)       # [T,512]
v          = linear(hidden, v_weight)       # [T,512]
mixed_qkv  = cat(q,k,v)                     # [T,1536], contiguous
gate       = linear(hidden, gate_weight)    # [T,512]
forget_a   = linear(hidden, forget_a_weight)# [T,128]
beta_a     = linear(hidden, beta_a_weight)  # [T,128]
forget     = linear(forget_a, forget_b)     # [T,512]
beta       = linear(beta_a, beta_b)          # [T,512]
packed causal conv(mixed_qkv, packed weights, paged conv state)
featurewise-beta prepare / KDA recurrence
per-head norm + output gate + output projection + TP reduction
```

The three convolution weights are already concatenated after checkpoint loading. Both Prefill and
Decode therefore use one packed-QKV convolution call and one packed conv-state allocation. There is
no per-channel convolution loop and no three-way conv-state cache to merge. The remaining packing
cost is the Q/K/V projection output materialization.

Lite differs from the existing Kimi merged projection in one important way. Kimi projects a
head-scalar beta directly from hidden states. Lite projects a replicated 128-dimensional `beta_a`
from hidden states, then applies a separate row-sharded `beta_b` projection. The Kimi class is a
loader/layout blueprint, not a shape-compatible implementation.

## 3. Frozen mathematical contract

The featurewise-beta order remains:

```text
q0    = L2Norm(q)
k0    = L2Norm(k)
scale = sqrt(sigmoid(beta_logits) + 1e-10)
k1    = k0 * scale
v1    = v * scale
```

The recurrent core consumes prepared `q0/k1/v1` with an effective beta of one. Forget-gate
construction, FP32 state accumulation, readout scale, output normalization, sigmoid output gate,
and output projection remain unchanged. In particular:

- K must be normalized before featurewise scaling;
- beta must not be applied again inside the recurrence;
- `forget_b` and `beta_b` cannot join the hidden-state GEMM because their inputs are the 128-wide
  low-rank outputs, not hidden states;
- Decode state publication stays in the existing live slot and must remain graph-safe.

## 4. Phase 12A: merged hidden-state projection

### 4.1 Rank-local layout

Phase 12A allocates one BF16 rank-local weight `[2304,3072]` with fixed row ranges:

| Component | Row range | Local shape | Checkpoint shard rule |
| --- | ---: | ---: | --- |
| Q | `[0,512)` | `[512,3072]` | row shard over KDA TP8 |
| K | `[512,1024)` | `[512,3072]` | row shard over KDA TP8 |
| V | `[1024,1536)` | `[512,3072]` | row shard over KDA TP8 |
| output gate | `[1536,2048)` | `[512,3072]` | row shard over KDA TP8 |
| forget-a | `[2048,2176)` | `[128,3072]` | replicated on every rank |
| beta-a | `[2176,2304)` | `[128,3072]` | replicated on every rank |

The merged row count is already aligned to 128. No padding, quantization, or Weight-NZ conversion is
needed. The old six registered parameters are removed; retaining them and building a post-load copy
would duplicate approximately 13.5 MiB per KDA layer and is not acceptable.

One GEMM produces `[T,2304]`. Fixed views recover `mixed_qkv`, output gate, forget-a, and beta-a.
`forget_b[512,128]` and `beta_b[512,128]` remain separate parameters and GEMMs.

### 4.2 Loader contract

The strict Lite checkpoint layout continues to enumerate and validate all six original source
weights. Each source maps to one component of the merged target while preserving its existing
source shape, dtype, and TP/replication rule.

The loader must:

- reject duplicate, missing, unexpected, wrong-shape, and wrong-dtype source tensors exactly as it
  does today;
- copy four rank-local row shards and two full replicated tensors into the fixed ranges above;
- count the merged target as loaded only after all six distinct components are present;
- support meta construction without copying data;
- expose only one registered merged parameter, so `expected_targets` and static-weight accounting
  remain exact.

This is a model-local extension of the existing strict loader. It does not change the general model
loader or add a one-implementation abstraction.

### 4.3 QKV materialization boundary

The merged output's first 1536 columns have row stride 2304 and are not row-contiguous when `T>1`.
The public Ascend `CausalConv1d` declares `AutoContiguous()` for every input, so passing that view
directly merely hides a copy inside the operator adapter.

Phase 12A keeps one explicit packed-QKV materialization before the existing convolution. It replaces
the current three-way `torch.cat`; it is not presented as the Phase 12B optimization. This preserves
the current convolution ABI and makes the copy visible to profiling. At the projection boundary,
the worst-case live activation increases by at most 768 BF16 values per token, or 1.5 KiB/token, and
must be checked in the full service.

### 4.4 Direct-board evidence and admission

The current six-GEMM path, including the current QKV concatenation, was compared with one merged
BF16 GEMM at the exact rank-local shape on an Atlas 910B NPU:

| Tokens | Current six GEMMs (us) | Merged GEMM (us) | Speedup | Max abs | Relative L2 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 37.435 | 14.480 | 2.585x | 0 | 0 |
| 2 | 48.542 | 16.587 | 2.927x | 0 | 0 |
| 32 | 48.072 | 16.676 | 2.883x | 0.125 | `3.54e-5` |
| 256 | 59.335 | 19.637 | 3.022x | 0.5 | `2.46e-5` |
| 1024 | 89.345 | 47.861 | 1.867x | 0.5 | `4.40e-5` |

The multi-token differences are expected BF16 GEMM tiling differences. Phase 12A therefore requires
elementwise equality at T1/T2 and relative-L2 agreement at larger T, followed by final hidden/logit,
service, and GSM8K checks. Leaf speedup alone is not sufficient.

The independent 12A design must additionally freeze the implementation flag/A-B mechanism,
focused test list, full projection-plus-materialization timing, graph BS1/BS2 checks, and exact 8P8D
rollback boundary before production code is changed.

## 5. Phase 12B: packed causal-convolution experiment

### 5.1 What is already packed

Checkpoint post-load already creates one QKV convolution weight, the cache already stores one packed
conv state, and the backend already issues one convolution call. Rewriting those pieces would add
code without removing work. Phase 12B is limited to the remaining `[T,2304] -> [T,1536]` packed-QKV
materialization introduced by the merged projection view.

### 5.2 Required experiment

The independent 12B design begins with an operator profile of:

1. explicit `qkv_view.contiguous()` plus the current public convolution;
2. the strided view passed to the current `AutoContiguous` adapter;
3. a stride/offset-aware direct kernel candidate, if the current kernel indexing can be extended
   without a second state/output layout.

The profile covers Decode BS1/BS2 graph and Prefill T32/T256/T1024. It records copy/operator count,
temporary bytes, output/state precision, and end-to-end layer latency.

Production changes are admitted only if they provide one convolution launch, no hidden contiguous
copy, unchanged packed state, graph capture/replay, and a measurable Decode benefit. Otherwise 12B
ends with a negative validation record and retains the explicit one-copy boundary. No generic
strided-tensor framework is added for this single use.

## 6. Phase 12C: featurewise-beta fusion experiment

### 6.1 Public implementation boundary

The pinned public ChunkKda and RecurrentKda operators accept head-scalar beta. An audit of a newer
upstream revision found the same scalar-beta ABI; no public operator directly accepts Lite's
`[T,H,K]` featurewise beta.

The existing public RecurrentKda was already rejected for production Decode: its K-major path was
about 2 ms, while the current graph-safe tensor implementation was approximately 130/172 us at
BS1/BS2. A V-major compiled adapter remained slower and could not be nested inside TokenSpeed's NPU
graph executor. Merely adding a featurewise-beta argument to that slow call is not an optimization.

### 6.2 Prefill candidate

The smallest Prefill candidate is one unified kernel boundary that performs:

- Q and K L2 normalization;
- `sqrt(sigmoid(beta_logits)+1e-10)`;
- featurewise K/V scaling.

It feeds the existing public gate-cumsum and chunk KDA calls. Unit-beta allocation must either be
eliminated by an explicit implicit-one core mode or reused as graph-stable storage; allocating a new
large ones tensor per layer is not acceptable. The candidate is connected to production only if the
full prepare-plus-chunk chain is faster than the existing Torch operations and does not add a large
temporary tensor.

### 6.3 Decode candidate

The Decode candidate folds normalization, featurewise scaling, forget-gate application, recurrence,
readout, and live-slot publication into one graph-capturable call. It must consume the existing
K-major FP32 state directly and preserve current padding/page-zero semantics.

The first board is an incremental featurewise-beta prototype at BS1/BS2 against the current
approximately 128/166 us tensor baseline. It is rejected before runtime integration unless it is
faster, finite, graph-capturable, and numerically aligned. If the public recurrent tiling cannot meet
that gate, a new minimal kernel may reuse its recurrence math, but no second executor or state layout
is allowed.

## 7. Validation hierarchy

Every admitted subphase must pass, in order:

1. CPU/meta construction, loader coverage, formulas, empty-token behavior, and negative checks;
2. exact-source single-NPU eager and graph leaf tests at target shapes;
3. cumulative Lite kernel/model tests and full pre-commit;
4. exact-source 8P8D with P eager/overlap-off and D fixed-BS graph/overlap-on;
5. four-token smoke, 1,100-token Prefill continuation, stable BS1, late-admission BS2, retraction,
   finite/fatal scans, graph fallback scan, HBM, TTFT, TPOT, throughput, and route-load checks;
6. the frozen EvalScope GSM8K first 100, including completion count, API errors, empty outputs,
   per-sample results, and comparison with the preceding exact baseline;
7. exact service, listener, process, and NPU-holder cleanup.

T1/T2 projection and graph Decode comparisons require exact output where the direct board proved it.
Longer Prefill uses relative-L2 plus downstream hidden/logit and service gates because a wider GEMM
can select a different valid BF16 accumulation tiling. Any new NaN/Inf, state/page mutation, graph
fallback, attributable score regression, or material end-to-end slowdown rejects the candidate.

## 8. Commit and rollback policy

The intended sequence is:

```text
Phase 12 overview design
12A design -> 12A implementation/tests -> 12A exact validation
12B design -> 12B implementation/tests or negative board -> 12B validation
12C design -> 12C implementation/tests or negative board -> 12C validation
Phase 13 cumulative acceptance
```

Each successful step uses explicit file-list, signed-off commits and is pushed immediately. Failed
experiments commit only public-safe design/negative-validation evidence, not dead production code,
private artifacts, machine addresses, credentials, checkpoint paths, or benchmark harness output.
The previous exact commit remains the rollback point for each subphase.
