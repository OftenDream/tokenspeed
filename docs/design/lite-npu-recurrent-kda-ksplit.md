# Lite NPU recurrent KDA Ksplit64 integration

## Scope and invariants

This adaptation is based on `hz/fix-lite-fia-contiguous` at `5d517586`.
The base's FIA layout copies outside task update and native NPU graph path
are preserved unchanged. Only the recurrent KDA implementation is added here.
On this base, all 43 focused recurrent KDA and FIA tests pass on NPU,
including changed-input graph replay. A candidate-only full-model run is
reported below, separately from historical operator and old-base A/B results.

The Ascend `torch_ascend_kda_paged_decode` implementation can use an optional
`flash_ops` recurrent kernel automatically when the installed package exposes
a callable `npu_recurrent_kda`. No enable flag or `RECURRENT_KDA_KSPLIT64`
marker is required. The registry name, runtime kernel boundary, cache planner
and K-major FP32 state remain unchanged. A missing package or callable uses
the Torch reference composition; package dependency/import failures and execution
errors are not silently converted into fallback. Tiling is owned by the
installed package: the measured package uses Ksplit64, but symbol availability
alone does not guarantee that optimization in an older package.

The fast path accepts BF16 featurewise-beta single-token Decode, B=1–256,
local H=4/8/16/32/64, K=V=128 and lower bound -5. This covers Lite's two head
geometries at TP1/2/4/8. Other dtypes, dimensions, scalar beta and gate bounds
use the Torch reference composition. Prefill is unchanged. The TS-local
K/BV32 Triton kernel and its launch/dispatch branch have been removed; the
optimized recurrent kernel is maintained only in the optional operator package.
Without that package, the reference path is not a performance-equivalent fallback.

Q/K/V/g/beta retain their independent batch/head strides. State matrices are
K-major with dense inner [128,128] storage, but page and head strides may
include gaps. No activation packing or cache transpose/copy is introduced.
Independent source/destination pages, same-row in-place updates and zero
output for graph padding remain required. Destination uniqueness and absence
of cross-row source/write hazards remain cache allocator obligations.

Ksplit64 assigns one program to a batch/head pair and processes two [64K,128V]
state halves. Both prediction halves contribute before either state half is
published. This reduces the per-program buffer requirement relative to an
unsplit K-major [128K,128V] tile without changing the physical cache layout.

## Verification and performance boundary

Metadata-only tests cover callable discovery without a capability marker,
missing-package/entry-point fallback, dependency errors, all deduplicated Lite/TP head
counts, strided inputs/state and fallback exclusions. Existing recurrent
Decode tests cover mathematical correctness, independent pages, padding and
changed-input native NPU graph replay. Fast-path NPU tests require the packaged
entry point so a missing package cannot silently test reference versus reference.
Deployment must use matching tested
kernel and extension artifacts.

The adaptation performance comparison is the parent implementation versus this
fast path with all other operators/configuration fixed. Operator timing is
reported separately from measured model TPOT; operator speedups are not an
estimate of full-model speedup.

The rebuilt public API was measured on Ascend910B2C with CANN 9.0,
PyTorch/torch_npu 2.9 and Triton 3.2. Inputs were BF16 independent views into
packed Q/K/V/g/beta, with FP32 interleaved K-major state. Seven paired rounds
used 16-unrolled native graph replay; the median NPU Event time excludes
packing, state reset, host dispatch and communication. The controls are the
unchanged K/BV32 kernel and a V/BV128 alternative. All 35 shapes passed eager
and graph pre-timing correctness checks.

| Batch | Local heads | K/BV32 (us) | Public Ksplit64 (us) | Speedup |
|---:|---:|---:|---:|---:|
| 1 | 4 | 6.230 | 6.902 | 0.90x |
| 32 | 4 | 59.254 | 25.764 | 2.30x |
| 32 | 8 | 112.782 | 41.167 | 2.74x |
| 32 | 16 | 219.382 | 71.845 | 3.05x |
| 32 | 32 | 433.322 | 132.088 | 3.28x |
| 32 | 64 | 861.111 | 252.325 | 3.41x |

Across B=1/2/4/8/32/64/256 and H=4/8/16/32/64, geometric mean speedup
over K/BV32 is 2.25x. The B1H4 case regresses 10.78%; the remaining 34
cases improve by more than 5%. All comparisons to V/BV128 are within 5%,
with geometric mean 1.00x. These are public-API measurements, not timings
of the earlier packed-pointer experimental kernel. The fallback paths for
larger local head counts may differ from this K/BV32 microbenchmark control.

## Current-base candidate-only full-model result

The adaptation on FIA-contiguous base `5d517586` completed one unprofiled
Lite3B run: input 65536, output cap 1024, BS32, 8 Prefill and 8 Decode ranks,
TP8/EP8 on both sides, chunked Prefill 4096, context 81920 and token capacity
2621440. Prefill uses eager, Decode native graph, and both disable prefix
caching. MLA weight TP1 replication, Decode weight-NZ, Prefill overlap off
and the existing Decode overlap configuration match the historical protocol.

| Metric | FIA-contiguous + Ksplit64 |
|---|---:|
| BS32 steady TPOT (ms/token) | 137.005 |
| Completed requests / output tokens each | 32 / 1024 |
| BS32 graph steps per Decode rank | 968 |
| Actual graph replays per Decode rank | 1080 |

All eight Decode ranks agree on the batch histogram and replay counts. The
same experiment-only preparation gate waits until all 32 requests are ready,
then releases permanently. TPOT excludes that preparation wait, the first 32
steady tokens and the final two boundary tokens; each request contributes
933–935 intervals. This is a steady-BS32 metric, not whole-request TPOT that
includes the artificial preparation gate. Requests are greedy and respect EOS;
all finished at the output cap.

**No new baseline was run.** The matching FIA-only baseline is to be measured
separately, so no KDA full-model speedup is claimed. Neither the old-base A/B
below nor measurements with different input/output/capacity or timing rules
can substitute for that baseline. This is one performance run, not repeated
statistical validation or full-model accuracy acceptance.

## Historical full-model performance observation (old base)

The following run used `hz/optimize-lite-kda-causal-conv` at `f982c1d9`,
not the current FIA-contiguous base. Neither its TPOT nor its output-difference
observations establish behavior of this rebased adaptation.

The full-model comparison uses Lite3B, input 65536, output cap 1024,
BS32, 8 Prefill ranks and 8 Decode ranks, TP8/EP8 on each side, chunked Prefill
4096, maximum context 81920, total token capacity 2621440 (2.5M), Prefill eager,
Decode native graph, and prefix caching disabled on both sides. Both variants
use the same checkpoint, 32 seeded synthetic prompts, greedy EOS-respecting
requests, dependency artifacts, MLA weight TP1 replication, Decode weight-NZ
setting and disabled Prefill overlap scheduling. Decode retains its existing
overlap configuration. Only recurrent KDA dispatch differs
from parent commit `f982c1d9`.

One unprofiled A/B pair completed all 32 requests with 1024 output tokens each.
Identical experiment-only hooks wait for BS32 during preparation and then
permanently release the batch. TPOT below excludes this waiting, the first
32 steady-state tokens and the final two boundary tokens. All eight Decode
ranks agree on 969 BS32 graph steps and 1079 actual graph replays per run;
each request contributes 934–936 steady token intervals. These hooks are not
part of the implementation change.

| Metric | Parent A | Ksplit64 B |
|---|---:|---:|
| BS32 steady TPOT (ms/token) | 255.585 | 259.922 |
| Complete requests / output tokens each | 32 / 1024 | 32 / 1024 |
| Observed speedup A/B | — | 0.9833x |

**No full-model TPOT improvement was demonstrated.** B is 1.70% slower in
this single pair. This is not evidence of a statistically stable regression
or a causal attribution to KDA; repeated A/B and component profiling would
be needed for that. The operator-level improvement must not be presented as
an end-to-end gain.

This is performance evidence, not full-model accuracy acceptance: the unchanged
Prefill paths produced different first tokens for 2/32 requests, and the last
emitted Decode token differs for 10/32. The client recorded token arrival times
and the final emitted token, not the complete output token streams, so full
sequence equivalence was not checked. The requested context override also
exceeds the checkpoint's derived context length. Operator correctness and
changed-input graph tests passed separately; neither the broader Phase 12
acceptance suite nor full-model quality is claimed here.
