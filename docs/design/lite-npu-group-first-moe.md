# Lite Group-First MoE

This path is selected explicitly with `gmoe_strategy="gmoe_aware"`.
`gmoe_strategy="global_expert_id"` retains the standard `FLASHLocalMoE`
implementation. Strategy selection does not inspect the accelerator type.

## Contract

The group-aware path uses one group-first layout for both Prefill and Decode.
For `W` model ranks and `G` logical MoE groups:

```text
EP  = W
EGP = EP / G
```

This strategy requires `EP >= G` and `G` to divide `EP`. It has no single-rank
runtime fallback; numerical reference implementations belong in tests.

Ranks are group-major. A contiguous `EGP` rank group owns one logical expert
group, while ranks at the same EGP offset form a strided `exchange_group`.
`exchange_group` is a communication group, not another parallel dimension.
The route is:

```text
[EP,T,H] -> [EP,G,T,H/G]
         -> AllToAll(exchange_group)
         -> [EGP,G*T,H/G] per logical group
         -> route / dispatch / GMM13 / GMM2 / combine within EGP
         -> AllToAll(exchange_group)
         -> [EP,T,H]
```

The two AllToAll operations use equal first-dimension splits. Both roles must
therefore enter this path with the same padded token capacity on every rank;
their actual token count and graph policy may differ, but the MoE algorithm does
not.

### Precision matrix and persistent stability tests

`test/ci_system/run_lite_moe_shape_matrix.py --output <new-directory>` runs
EP8/EP16 x W8A8/BF16 with G8/H4096. Each combination retains one process set,
weights and communication resources. Default lengths are 1,2,4,...8192 followed
by **every integer** 8193..16384 (8206 lengths per combination). `--lengths`
accepts an explicit comma-separated preflight list. Every length checks eager
fusion/reference bytes and fixed-input repeats; by default every shape is
also captured and replayed twice for both paths. `--graph-every` explicitly
allows sampled graph coverage and is recorded. It never resumes random soak.
The harness uses `PRECISION_LENGTHS=dense` or an explicit list and rejects
combining it with random stability iterations/duration. Per-rank JSONL records
each completed shape; final records require all requested lengths exactly once.
The driver requires successful process exit and all EP rank records before
continuing to the next combination. Partial output is not a completed pass.

The composed reference must receive the same `num_zero_experts` as the model.
This is essential at EGP1: init routing must include zero IDs in its declared
expert count while excluding them from the active real-expert interval. Omitting
this test-plan argument produced nonfinite reference output at EP8; it is not
fixed by changing the fused operator or relaxing the comparison.

The generalized bindings passed the full-layer T32 preflight in all four
EP8/EP16 x W8A8/BF16 combinations: every rank matched finite output bytes in
eager, three changed-input Graph replays and two fixed-input replays. Shared
overlap was disabled, W8A8 used both routed SmoothQuant scales and NZ weights,
and BF16 used unquantized weights. This preflight does not establish the full
8206-length matrix, random stability, or a performance result.

The subsequent persistent-process boundary matrix also passed every rank in
all four combinations at T=1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,
8193,16383,16384. All 17 shapes were captured for each combination, with eager
A/B and same-input repeats plus two Graph replays against eager snapshots.
Both BF16 and W8A8 first checked the chunked reference against its original
unchunked expert leaf at per-rank T1025. These 816 rank/shape cases establish
boundary coverage only; dense-tail completion still requires its own records.

For error characterization of other group counts, the full-layer harness has
an explicit `ERROR_REPORT=1 FULL_REFERENCE=1 STRICT_BITS=0` diagnostic mode.
Set `GMOE_NUM_GROUPS=2` or `4`; do not assign Bash's reserved `GROUPS` array.
The diagnostic record includes actual group count, group hidden width and local
expert count, which must match the requested configuration before reporting it.
It records finite status, maximum/mean absolute error, relative L2 error and
storage-bit mismatch counts for the pre stage, the expert leaf with identical
input/routes, the post stage with identical expert output, eager full output,
and changed-input Graph outputs. Both graph variants also report same-input
repeat discrepancies separately, including reference nondeterminism. This mode reports observed discrepancies instead
of declaring a tolerance pass; it does not change strict acceptance thresholds
or runtime kernels. Post-stage metrics include return communication, identity
experts, output projection and shared addition, not just ReduceScatter.
For a deterministic HCCL control, set `HCCL_DETERMINISTIC=true` before launching
new workers/initializing communicators. Diagnostic records retain that setting
and the separate PyTorch deterministic-algorithms flag. Deterministic repetition
does not by itself establish identical arithmetic/rounding between HCCL and
the fused return kernel. This test setting does not change production defaults.

`REDUCE_REPORT=1` within diagnostic mode isolates one identical expert-output
fixture before Graph tests. It compares native BF16 ReduceScatter with FP32
ReduceScatter rounded once to BF16. An independent oracle AllGathers the
original BF16 contributions, selects the destination shard and sums on CPU in
FP64 before one BF16 rounding. The composed post stage is then rerun with only
its reduction result substituted; identity handling, exchange, projection and
shared addition remain unchanged. This CPU oracle is test-only and never
enters runtime execution. Per-rank `reduce-report` records distinguish raw
reduction errors from the final post-stage error.

`test/ci_system/run_lite_moe_precision.py` runs a strict W8A8 EP16/G8 matrix
in the configured system CANN environment. It never changes production
selection, core quotas, weights, or the virtual environment.
The default matrix covers per-rank decode token counts 1, 7, 31, 32, 33, 64,
65 and prefill counts 128, 257, 512, 1024, 2049, 4096, plus real-only, zero-only and
concentrated real-expert routing and all four routed SmoothQuant settings.
Prefill explicitly uses `ForwardMode.EXTEND`, with the same group-first
algorithm, rather than relabeling a decode run.

The full-layer reference pins composed pre/post communication and the
`torch_npu` split expert plan. It does not reuse fused dispatch/router,
MM13 or MM2/finalize from the candidate. Shared dense MLP, BF16 projections
and underlying individual vendor operators remain common; this is a fusion
equivalence test, not an independent whole-model/quantization-quality oracle.
Canonical INT8 weights are prepared as NZ in both paths.

`FULL_REFERENCE=1 STRICT_BITS=1` requires identical dtype, shape and bytes
and rejects nonfinite output even if both paths produce the same NaN/Inf.
It tests isolated return-stage output, eager full-layer output, changed-input
graph replay, fixed-input A/B and fixed-input self-repeat. A failed rank
notifies all ranks before snapshots are saved. No tolerance is widened to
make the strict path pass; existing tolerant tests retain their old policy.

```sh
python test/ci_system/run_lite_moe_precision.py \
  --output <new-results-directory> --soak-hours 3
```

The driver requires `GMOE_EXCHANGE_OPTIONS`,
`TOKENSPEED_LITE_GMM13_LIBRARY`, `TOKENSPEED_FUSED_MM2_LIBRARY`, and the
existing CANN/custom-OPP/PYTHONPATH setup. Results directories must be new.
It records one result per shape and requires records from all 16 ranks.
Any matrix failure prevents the soak from starting.
The driver retains the configured window for every shape, including the
default 32 MiB for large prefill. Updated bindings expose
`gmoe_comm_capabilities() & 1` and chunk communication internally. The C++
operator entry point derives a per-submission token tile from the registered
window and metadata size, packs expert-major return slices, and restores the
original [EGP,G,T,H] order after dispatch. The model forward has no chunk loop.
Small inputs retain the original single-launch path and all core quotas.
Large inputs currently incur additional device layout copies and multiple
communication launches; this is a capacity fix, not a fusion-speedup claim.
Old bindings retain explicit whole-tensor capacity guards rather than silently
accepting inputs their kernels cannot handle.

The logical communication input has no fixed 65535-token cutoff. Individual
transport/DMA submissions remain bounded, and all normal tensor-allocation and
integer-overflow checks remain necessary. MM13/MM2 still use INT32 expanded
route indices: index overflow is rejected explicitly, not truncated. Their
workspace/output memory can still exhaust HBM for sufficiently large batches.

The optional soak splits its requested wall time equally between decode T64
and prefill T2049 (cross-window chunks). Each phase retains the same processes, weights, graph
buffers and communication resource throughout. Every cycle changes input,
compares fused/split output bytes, repeats the same input and checks both
A/B and temporal stability. Rank zero owns the deadline; all participants
follow its stop decision. Per-rank minute logs report completed cycles,
allocated memory and elapsed time. A mismatch terminates the phase and saves
input/output/pre-stage snapshots; a short/failed run cannot count as a passed
soak. There is no performance claim from these instrumented accuracy runs.

For persistent random-length stability, run the full-layer harness directly:

```sh
FULL_REFERENCE=1 STRICT_BITS=1 ROUTER_FUSION=1 ROUTED_FUSION=1 MM2_FUSION=1 \
WEIGHT_DTYPE=int8 SMOOTH_QUANT=both SHARED_OVERLAP=0 RANDOM_LENGTH_ITERATIONS=10000 \
RANDOM_MAX_TOKENS=32684 RANDOM_MIN_SECONDS=10800 RANDOM_GRAPH_EVERY=100 \
TOKENS=1024 REFERENCE_EXPERT_CHUNK_ROWS=8192 REFERENCE_CHECK_UNCHUNKED=1 \
OUTPUT_DIR=<new-results-directory> \
torchrun --standalone --nproc-per-node=16 test/ci_system/validate_lite_moe_fused_exchange.py
```

One set of processes, weights and communication resources survives all 10000
iterations. Rank zero broadcasts each length; rank-specific deterministic seeds
generate independent inputs. A shuffled boundary prefix covers tile/window
edges, followed by balanced uniform sampling of decode T1..128 and prefill
T129..32684. The maximum is a configurable test budget, not an operator limit.
The run ends only after both 10000 iterations and 10800 seconds have elapsed.
The maximum length is also captured/replayed at least once, independently of
the periodic graph sampling. All lengths are per rank.

For these large inputs, only the unfused test oracle's local expert leaf is
split into 8192 received-row chunks to bound its INT32 GMM1 intermediates.
The candidate fused leaf still receives the full input; communication, router,
shared experts and projections are not shortened. Before the random sequence,
the initial T1024 fixture checks this chunked oracle against its original
unchunked leaf bit for bit. This is a test memory policy, not a runtime fallback.
Every iteration executes both implementations twice, checks finite output bytes
against the unfused control and each implementation's first output. Every 100
iterations, the current actual shape is also captured and replayed twice against
its eager snapshot; graphs are released afterward, not cached without bound.
This is random-shape eager coverage plus sampled ACL Graph coverage, not 10000
replays of one graph. The JSONL sequence is written before each iteration;
per-rank progress, final length histograms and mismatch snapshots make incomplete
runs distinguishable from a full pass. Correctness here means fusion equivalence
for this fixture, not independently validated checkpoint/model quality.

Capture uses two persistent, distinct, unrestricted streams reserved once per
process, never a new `torch.npu.Stream()` per shape. The installed stream pool
rotates through 32 handles and can return a still-live shared stream with its
4 Cube / 8 Vector quota intact. Capturing the reference on that handle changes
BF16 projection tiling/accumulation order; it is not a valid full-core oracle.
The T70 failure's normalized projection was reproduced bitwise using four
cores, while 24 cores reproduced the eager reference. Stream reservation skips
limited handles without resetting their quotas; all rank logs record capture
and shared stream IDs/limits. The fused pre-stage also rejects main/shared
aliasing before input projection. Shared/dispatch overlap remains enabled when
requested. Fresh short repros do not replace the full random-sequence soak.

Full-fusion selection now supports BF16 as well as W8A8, with no EGP2-only
registration constraint. `flash_npu_routed` fuses the first expert segment;
`flash_npu_routed_full` additionally fuses MM2/finalize. The BF16 path uses ND
weights and real BF16 Cube/Vector kernels, not quantization or a composed
fallback. BF16 H/I are 32-aligned in [32,8192], TopK is [1,64], and local
expert count is [1,1024]. Host tiling checks actual UB and index capacities;
global metadata synchronization requires a full-core stream. Weight loading
and canonical expert placement are shared with the existing BF16 leaf.

The W8A8 MM13 pipeline accepts 32-aligned H/I in [32,8192] and TopK in [1,64],
subject to actual host-tiling UB/DMA checks. It selects a UB row batch from
8/4/2/1. If a complete row cannot fit, tiling throws before allocating output
or workspace or launching the kernel, reporting H/I and required/available UB.
There is no fallback or silent truncation. On the tested 196352-byte UB device,
H8192/I4096 fits; H8192/I6144 requires 236800 bytes and is rejected. BF16 has
a different UB footprint and does not inherit this W8A8 restriction. NZ
weights, optional SmoothQuant and exact quantization rounding remain unchanged.
Removing the EGP-only selection restriction does not remove resource checks. The existing
full-layer matrix records above validate only their recorded W8A8 EP16/G8
configuration; new BF16/EGP/shape combinations need their own full-layer
acceptance. Standalone kernel passes do not imply a completed distributed
matrix or stability run. Unsupported cases must raise, never count as passes.
`GroupAwareFlashLocalMoE` owns the input exchange as well as the output exchange;
the decoder must not run the generic pre-MoE token AllGather first. Thus every
rank enters this path with its own (potentially different) `[T,H]` tokens.

## Expert placement

Checkpoint expert IDs remain global and group-major. For both roles, rank index
`ep_rank = group_id * EGP + egp_rank` owns one contiguous `E / EGP` slice of
group `group_id`.

For an eight-rank stage:

| EP | G | EGP | exchange group size | local experts when E=384 |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 4 | 2 | 4 | 192 |
| 8 | 8 | 1 | 8 | 384 |
| 16 | 4 | 4 | 4 | 96 |
| 16 | 8 | 2 | 8 | 192 |

## EGP execution

The composed / pure-communication `EGP > 1` path first AllGathers group input, then every EGP rank recomputes
the same router TopK locally. Only hidden states are gathered; route weights and
IDs are not separate collectives. Local-expert GMM output is ReduceScattered on
the contiguous expert group. This supports distinct input tokens without
depending on the A2 V2 dispatch/combine pair, which does not complete for the
required EGP2 execution. The communication boundary is isolated so a corrected
dispatch/combine leaf can later replace AG/RS without changing placement or the
surrounding group exchanges.

## W8A8 expert leaf

Compressed-tensors checkpoints with static symmetric per-channel INT8 weights
and dynamic symmetric per-token INT8 activations select the Ascend W8A8 leaf.
Canonical checkpoint weights and scales remain vendor-neutral in the runtime;
the selected kernel preprocessor transposes W13/W2 once after loading and stores
both tensors in Ascend format 29 (FRACTAL_NZ). The forward path is:

```text
BF16 init-routing + dynamic quant -> INT8 expanded rows + FP32 row scales
-> INT8 NZ GMM13 (INT32 output)
-> dequant + SwiGLU + dynamic requant
-> INT8 NZ GMM2 (BF16 output)
-> finalize routing
```

Optional expert smooth scales are consumed by init-routing for GMM13 and by
the fused SwiGLU requantization for GMM2. NZ conversion is a post-load
operation; no format conversion or device-specific policy appears in model
forward.

`EGP = 1` must not initialize MC2 or HCCL for expert routing. It uses local
`npu_moe_init_routing_v2`, two grouped matmuls with SwiGLU, and
`npu_moe_finalize_routing`. This is the semantic EGP1 form of the same backend,
not a communication fallback.

The BF16 leaf uses `mlp_split_swiglu` between its grouped matmuls. Unlike a
plain SwiGLU over the fixed init-routing output capacity, it consumes the
per-expert counts and computes only the valid dispatch-ordered route prefix.
The W8A8 leaf retains `npu_dequant_swiglu_quant` because GMM13 produces INT32
and the activation step must also dequantize and requantize.

The shared dense expert uses the common runtime `SiluAndMul` entry point.
Ordinary activation dispatch belongs to `tokenspeed-kernel`, with Ascend using
`npu_swiglu` on concatenated `[gate, up]` values. This avoids separate slices,
FP32 casts, SiLU, and multiplication kernels. An explicit clamp limit retains
the native FP32 gate/up clamp semantics.

For symmetric channel-INT8 / dynamic-token-INT8 checkpoints, the shared expert
now passes the quantization config and its fully qualified prefix to the common
dense MLP. Ignored projections remain unquantized; BF16 checkpoints are unchanged.
Canonical INT8 weights and per-channel scales use the existing merged-column
and row-parallel checkpoint loaders. After loading, each dense linear prepares
a transposed NZ weight and FP32 scales through the kernel boundary. The shared
W8A8 path is:

```text
dynamic token quant -> INT8 NZ gate/up matmul (INT32 accumulators)
-> DequantSwigluQuant (INT8 values + FP32 dequantization scales)
-> INT8 NZ down matmul (BF16 output)
```

This preserves the shared expert's TP1 placement and communication strategy.
The first version supports symmetric channel weights and dynamic token scales;
it does not add asymmetric, static-activation or SmoothQuant checkpoint support.

`SiluAndMul(..., int8_out=True)` also accepts floating input. Its Ascend
`SwiGluQuant` wrapper supplies a cached identity smooth-scale tensor because
the CANN 9.0 kernel reads this optional input unconditionally. It converts the
operator's `127/amax` output to the common dequantization scale `amax/127`.
INT32 input requires explicit FP32 weight and activation scales and uses
`DequantSwigluQuant` instead. Clamped INT8 output is rejected explicitly.
Regression tests cover all three floating dtypes, both quantized entry points,
empty/zero inputs, changed-input graph replay, checkpoint loading and NZ MLP
execution, including `[32,4096]` shared input and intermediate width 2048.

## Selectable stage boundaries

The group-aware forward has three sequential stages. This is an execution
boundary, not a claim that each stage currently runs as one device kernel:

| Stage | Contract | Current implementation |
| --- | --- | --- |
| `moe.gmoe_pre` | Shared MLP from original input; input projection, norm, scaling, group exchange, EGP AllGather, router and TopK | `composed_gmoe_pre` |
| existing `moe_apply` | Init routing through finalize routing | Existing selected BF16/W8A8 expert leaf, unchanged |
| `moe.gmoe_post` | EGP ReduceScatter, local identity contribution, reverse group exchange, output projection and shared addition | `composed_gmoe_post` |

Input AllToAll, AllGather and routing stay in the pre stage. The composed path
routes after AllGather; the optional router fusion below routes local rows
concurrently with AllGather. Only return communication is in the post stage. Shared computation
moves before init routing, but no overlapping stream or fused implementation
is added. The existing individual math and communication operators are reused.

`select_gmoe_stages` uses the common registry/`select_kernel`
mechanism during model preparation, never during forward or graph replay.
`gmoe_pre_solution` and `gmoe_post_solution` independently constrain selection;
`None` means automatic and `"composed"` explicitly pins the existing composition.
An unavailable solution fails selection instead of silently falling back.
Future fused kernels register under the same modes with their own capability,
signature, trait and priority filters. Traits include group placement, hidden
size, routed/shared weight dtypes, TopK and normalization policy. The middle
MoE plan is not selected or modified by this framework.

The context retains references to the existing model components and runtime
communication callbacks. Optional stage preparation hooks request dedicated
process groups through the runtime's group factory; the kernel package does not
import the runtime. No checkpoint weights or quantization/NZ preparation
are duplicated. Pre returns received input, its local pre-AllGather slice,
TopK weights/IDs and shared output. These values are per invocation, not mutable
state stored on a plan. Post consumes them and adds identity/shared exactly
once, preserving EGP1/EGP2 behavior and distinct tokens across ranks. Empty
inputs keep the runtime's existing early return. Raw router weights and routing
policy are also exposed for future fusion, alongside the composed route callback.

The decomposition was checked on EP16/G8 with distinct `[32,4096]` input per
rank in both BF16 and W8A8 configurations. All 16 ranks matched the original
forward bitwise in eager and in three changed-input graph replays. Selection
and composition unit tests cover EGP1/EGP2, local identity ownership, independent
overrides and unchanged middle-call arguments. No fusion speedup is claimed.

### Optional fused exchange and shared/dispatch overlap

Shared overlap now defaults to **disabled** (`shared_overlap=false`). Shared
experts execute on the unrestricted main stream before dispatch, using the same
NZ weights and activation kernels. Random-shape testing on CANN 9.0 exposed a
stock per-token BF16-output QuantMatmul tiling mismatch under stream limits:
the launch uses four Cube blocks but the column partition can retain sixteen
pieces, leaving output unwritten. Serial shared execution avoids this trigger;
it is not an operator fix. Explicit `shared_overlap=true` retains the previous
experimental schedule below for diagnosis. The dispatch/router and combine/zero
operators, their internal overlap and their core allocations are unchanged.
Previously measured overlap performance below does not describe this serial
default. Random stability records include the actual shared-stream state.
The serial preflight passed 100 iterations on all 16 ranks, covering 93 distinct
lengths and ten sampled graph shapes, with finite bitwise-equal outputs and
same-input repeats against the composed/split reference. This is a preflight
result, not a completed 10000-iteration stability claim.

The `flash_npu` solution replaces the pre-stage layout/A2A/AG with
`gmoe_dispatch`, and the post-stage RS/identity/A2A/layout with
`gmoe_combine`. The original `composed` implementations are
retained for A/B testing; the init-routing through finalize-routing leaf is
unchanged. Projection, normalization, scaling, router and shared MLP still use
their existing operators and weights, including W8A8 NZ preparation.

Shared computation starts after input projection, before norm/scale, and overlaps
**forward dispatch**, not zero-expert combine:

```text
projection
  main stream:   norm / scale -> fused A2A+AG -> router -> wait shared
  shared stream: limited shared MLP -> event --------------^
-> unchanged init-routing ... finalize-routing
-> fused RS + zero expert + A2A -> output projection + shared add
```

The shared stream is created and limited with `torch.npu.set_stream_limit`
outside capture (default 8 cube / 8 vector cores). It waits for input production;
the main stream waits on its completion event before returning from pre. There
is no host synchronization in forward. Eager cross-stream allocation lifetimes
are protected with `record_stream`. `shared_overlap=false` runs this same fused
exchange solution with serial shared computation for a second A/B control.
The main/communication stream is not restricted by the shared limit; in
particular the later split-core restore must have at least `8+zero_cores` AIV
cores available. Shared work has already joined before that kernel launches.

Enable through model configuration, after deploying the matching custom OPP,
PyTorch binding, SHMEM runtime and its pinned SHMEM dependencies:

```json
{
  "gmoe_strategy": "gmoe_aware",
  "gmoe_pre_solution": "flash_npu",
  "gmoe_post_solution": "flash_npu",
  "gmoe_exchange_options": {
    "binding": "/path/to/gmoe_comm_binding.so",
    "rdma_library": "/path/to/libgmoe_comm.so",
    "rendezvous": "tcp://bootstrap-host:43033",
    "window_bytes": 33554432,
    "shared_overlap": false,
    "shared_cube_cores": 8,
    "shared_vector_cores": 8
  }
}
```

An already registered binding is reused, never loaded a second time. Dynamic
libraries and OPP paths are deployment settings, not globally installed or
hardcoded by the model. With nonempty exchange options, automatic stage
selection may choose `flash_npu`; without them the existing `composed` path
remains the default. Explicitly pin both solutions to `composed` to run the old
path even when deployment options are present. Pre/post can also be selected
independently to isolate numerical or performance effects.

The runtime creates an isolated full-EP communicator in deterministic global
group order, separate from ordinary A2A/AG/RS groups. It initializes one SHMEM
resource per process/EP domain before capture, shared by all compatible MoE
layers. Incompatible second-domain initialization fails instead of silently
reusing windows. For multiple DP EP domains, `rendezvous` must be a dictionary
mapping each EP domain's first global rank (string key) to an independent TCP
URL. The same EP runtime and its graphs must be used sequentially, never from
concurrent model invocations or graph replays.

Requirements: Ascend 910B; contiguous octet-aligned EP ranks, EP in [8,128] and
divisible by 8; G>1 dividing EP; every physical rank octet must be one complete
HCCS domain. Arbitrary device/rank permutations and physical multi-server RDMA
are not validated. Every rank must have the same token capacity/dtype/call order.
The empty fast path is valid only if all EP ranks are empty. Nonempty T must be
positive, group hidden 16..32768 aligned to 16, and TopK 1..64. Legacy bindings
require T<=65535 and window capacity `32768+(EGP+2)*G*T*group_hidden*2`.
Chunk-capable bindings tile internally using the fixed window as described above.
EP8/G8 automatically omits AG/RS inside the operators; its forward
uses OPP/HCCL windows, while its restore still requires SHMEM. Default zero
partition: 8 cores for every EGP size, matching the 8 communication cores
(half of the 16-core launch, not half of all physical AIV cores).
Override with `zero_cores` (1..8) to reproduce the previous 4-core baseline.
Invalid deployment/shape contracts raise, with no rank-local fallback.

After completing all work and destroying all graphs, call
`release_moe_resources()` collectively on one group-aware layer per process,
before destroying process groups. This closes the shared resource for all
layers; calls on other layers are idempotent. Forward/replay after release is
invalid. Do not run collective finalization from destructors/atexit.

`test/ci_system/validate_lite_moe_fused_exchange.py` exercises the actual full
group-aware layer, including shared experts. It compares composed/fused received
inputs, route IDs/weights, local identity sources, return stages, eager outputs
and changed-input ACL Graph replays using the same weights and distinct inputs
per rank. Set `GMOE_EXCHANGE_OPTIONS` to the options JSON, `OUTPUT_DIR`,
`WEIGHT_DTYPE=int8|bf16`, optionally `GROUPS`, `TOKENS`, and `PROFILE=1`, then
run through torchrun with 8 or 16 processes. Graph event latency includes host
submission/rank arrival overhead; profiler kernel durations are separate metrics.
SHMEM RS uses rank-ordered FP32 accumulation before casting, so larger EGP
reductions are tolerance-checked against HCCL, not assumed bitwise identical.

### Experimental fused shared FFN

`gmoe_pre_solution="flash_npu_router_ffn"` explicitly selects the custom shared
FFN together with dispatch/router fusion. Keep `gmoe_post_solution="flash_npu"`
and set `gmoe_exchange_options.ffn_binding` to the deployed FFN binding. This is
opt-in: automatic selection and the serial shared default remain unchanged.
Set `shared_overlap=true` to test the existing limited shared stream with this
FFN. Core quotas are unchanged: shared 4 Cube / 8 Vector, dispatch/router
20 Cube / 40 Vector. Input/event waits and the join before init routing remain.

The FFN supports bias-free, unclamped TP1 BF16 and symmetric dynamic-token W8A8.
W8A8 fuses input dynamic quant, MM13, dequant/SwiGLU/requant and MM2/dequant in
one mixed launch. BF16 preserves the intermediate BF16 rounding. The binding
derives its task partition from the actual stream quota. Current scratch is
proportional to token count; H/I must be 32-aligned and the binding explicitly
checks its row UB budget. It does not impose a fixed token cutoff.

Preparation keeps the original MLP and checkpoint loaders intact. The callable
reads the final Linear INT8 plans after child post-load processing, sharing
their NZ weights/scales without an additional conversion. BF16 reads the
original ND weights. No library load or weight conversion occurs in forward.
`context.prepared_shared` belongs only to the new selected pre-stage; composed
and older fused pre-stages continue to invoke `context.shared_experts`.

For acceptance use `SHARED_FFN=1 FUSED_FFN_BINDING=<library> FULL_REFERENCE=1`
with the strict full-layer harness above. It requires the independent serial
unfused reference, avoiding the known limited stock QuantMatmul failure and
avoiding accidental reuse of the fused FFN by the oracle. Standalone equality
or successful graph capture does not establish concurrent full-layer stability.

Initial EP16/G8 W8A8 acceptance at per-rank T32 and T64/H4096 passed on all
16 ranks: finite bitwise-equal shared/eager outputs and 100 changed-input graph
replays per rank against the composed/split reference. Both routed expert
fusions and routed SmoothQuant were enabled, with shared 4 Cube / 8 Vector
overlapping dispatch/router. This is short-run coverage, not a completed
10000-iteration or multi-hour random-prefill stability claim.

A subsequent persistent-process preflight passed 100 randomized lengths on
each of 16 ranks, covering 94 distinct T values and 11 captured shapes,
including a T32684 graph. Each iteration compares A/B bytes and same-input
self-repeats; sampled graphs replay twice against eager snapshots. The shared
stream remains limited to 4 Cube / 8 Vector and overlaps dispatch throughout.

For the T32 integrated fixture, the median across rank medians after excluding
the first two profiler replays was 90.68 us for shared FFN, with 75.32 us of
dispatch overlap, 14.83 us launch lead and zero median shared tail. Whole-layer
ACL Graph event latency was 496.14 us versus 964.11 us for the fully composed
unfused control. This includes all MoE fusions, not FFN-only speedup. The FFN
is slower under dispatch concurrency than its standalone four-Cube timing;
the schedule hides that duration in the measured decode workload.

### Dispatch/router fusion contract

Set `gmoe_pre_solution="flash_npu_router"` and keep
`gmoe_post_solution="flash_npu"` to use `gmoe_dispatch.router`. The existing
`flash_npu` pre-stage remains the pure-dispatch plus separate-router control;
automatic selection is unchanged. No device policy is added to model forward.
The existing FP32 classifier/bias and routing policy are passed directly to the
overload, without copying weights or rerunning `context.route` afterward.
Routers must be fully replicated within each EGP group, as in the existing
group-aware checkpoint loader. Expert weights remain sharded, NZ and unchanged.

The fused kernel performs input A2A, then computes router on `G*T` local rows
while gathering hidden, followed by gathering its TopK weights/IDs. It returns
hidden `[EP*T,H/G]` and group-local routes `[EP*T,K]` in the existing order.
The local hidden narrow-view remains the source for zero experts in combine.
The shared expert forks after projection but before norm/scale, overlaps dispatch, and joins before init-routing;
it does not overlap the later combine. FP32 routing avoids a BF16 classifier
conversion, but floating reduction order can differ: route IDs are tested
exactly and weights numerically, not assumed bitwise identical.

The framework explicitly defaults `router_cores` to **32 AIV** for this
solution; the operator API default is **unchanged**. Including eight communication
AIVs, the launch uses **40 Vector / 20 Cube blocks**. On the validated 24-Cube,
48-Vector device, shared-stream defaults therefore change from **8 Cube / 8
Vector** in the old solution to **4 Cube / 8 Vector** in this solution. This is
a framework stream quota, not a kernel source change. Both use stream limits
and event waits outside/inside capture respectively. Preparation rejects core
overcommit instead of silently reducing the router quota. `router_cores` may be
an even value in [2,32]; explicit shared quotas must leave room for the entire
mixed launch, including its non-computing paired Cube blocks. The main stream
must remain unrestricted, and runtime window use must stay sequential.

Preparation checks contiguous FP32 classifier `[N,H/G]` and correction bias
`[N]`, with 1..1024 total routing experts (including zero experts), 64-aligned
group hidden 64..1024, K in [1,min(N,64)] and a finite positive FP32 scale.
Before shared work starts, each invocation checks both the old combine window
bound and the router bound:
`32768 + (EGP+1)*G*T*(H/G)*2 + EGP*G*T*ceil(K/8)*8*8`.
T retains the communication limit; there is no decode-only token assumption.
New binding and SHMEM runtime must be deployed together. Unlike the old EP8
pure-dispatch entry, the router overload uses SHMEM for EP8 as well as EP16.

Set `ROUTER_FUSION=1` in the full-layer validation harness. It compares the new
pre-stage to `flash_npu`, using identical expert plans, shared-stream quotas and
combine. It checks hidden and IDs exactly, route weights with FP32 tolerances,
and complete eager/changed-input graph outputs. With
`ROUTED_FUSION=1 MM2_FUSION=1`, both controls retain both expert fusions.
Records include actual framework core quotas, route-weight error and baseline
selection, so a quota change is not hidden inside an unrelated optimization.

Before moving shared launch ahead of projection/norm, full EP16/G8 W8A8 validation with distinct T32/H4096 inputs and optional routed
smooth scales on both projections passed 100 changed-input graph replays per
rank. Hidden, IDs, route weights and layer outputs were bitwise equal in this
fixture; this does not establish bitwise equality for arbitrary routing scores.
At the same shared quota (4 Cube / 8 Vector), full-layer Graph event medians
were 538.69 us for pure dispatch plus router and 519.46 us for router fusion.
Across all 16 ranks, excluding the first two of ten profiled replays, median
dispatch-to-expert spans were 119.76 and 101.98 us; the new dispatch/router
kernel alone was 91.51 us. Shared-expert device work overlaps that kernel by
89.92 us, but its full span is 98.05 us, so the pre-stage still has an exposed
shared tail. These are full-layer measurements with both expert fusions, not
the communication-only microbenchmark or a comparison against the old 8-Cube
shared quota. Separate eager traces retain input shapes when graph replay
does not. `summarize_lite_moe_router_timeline.py` verifies ten fused calls per
rank, absence of separate TopK in the new trace, and shared-stream overlap.

The current pre-stage forks shared after input projection but before norm and
scale. Shared consumes the original hidden states; its stream wait deliberately
includes input projection to avoid overlapping the two input GEMMs. Moving it
ahead of projection was also tested: the shared gate/up GEMM increased from
about 49 us to 115 us in one steady-state trace and whole-layer Graph medians
regressed from 522 to 544 us. Those negative results are retained, not claimed
as an optimization. The main stream joins at the same
pre-stage boundary before init routing. No kernel, core quota, weight layout or
quantization changes are involved. The serial `shared_overlap=false` control
remains serial. `SHARED_EARLY_TEST=1 ROUTER_FUSION=1` in the validation harness
compares this schedule to a test-only copy of the previous late-shared schedule;
both controls use the same fused dispatch/router and expert implementations.
The summary records shared lead time and shared tail beyond dispatch explicitly.

For the validated EP16/G8 W8A8 T32/H4096 fixture, moving shared between projection
and norm advances its start relative to dispatch by 18.67 us. The steady-state
shared tail after dispatch drops from 10.23 to 0.04 us, and dispatch-to-expert
span from 102.98 to 93.19 us. Paired whole-layer Graph event medians are 523.60
us (late launch) versus 512.63 us (before norm). All 16 ranks pass eager and
100 changed-input replays bitwise. Both variants retain 32 router AIV cores,
shared 4 Cube / 8 Vector, identical NZ/SmoothQuant weights, both expert fusions
and combine. This is a scheduling improvement without an operator source or
quota change. The shared gate/up GEMM still grows from 47.72 to 62.46 us under
this overlap, but the earlier start more than offsets that growth for this
fixture; the result is not a claim for all batch sizes.

### Experimental routed W8A8 expert fusion

`gmoe_expert_solution="flash_npu_routed"` explicitly selects a replacement for
the middle leaf's **init routing + GMM13 + SwiGLUQuant** segment. It does not
change pre/post selection, shared/dispatch overlap, weight layout, GMM2,
finalize, or identity handling. `None` remains the existing automatic selection;
`"torch_npu"` explicitly pins the unfused reference. A requested unavailable
solution raises instead of leaving the layer unprepared.

The original validated fixture is EP16/G8 (EGP2), model H4096,
group H512, I1024, TopK16, BF16 input and symmetric W8A8 NZ weights. The
current generalized H/I/TopK and UB rejection contract is described above. Each
rank can own 1..1024 local experts in an equal contiguous shard; expert counts
and starts come from the loaded weights, not E384/E192 constants. The kernel
supports partial expert buckets and arbitrary local-range start alignment.
The explicit 1024-local-expert limit (matching downstream NZ GMM) and UB budget are checked
before launch; unsupported input shapes are rejected;
there is no implicit runtime fallback. Preparation loads the standalone binding
from `TOKENSPEED_LITE_GMM13_LIBRARY`, unless already registered. No library load,
CPU count readback, or NZ conversion occurs during forward/capture.
The full-layer harness accepts a positive per-rank `TOKENS` value rather than
assuming T32; communication and kernel bindings retain their own shape/capacity
checks. T64 produces received `[1024,512]`, routes `[1024,16]` and expanded
capacity 16384 in EP16. It uses the same kernels and core quotas, not a
shape-specific implementation. Full T64 W8A8 early/late-shared A/B passed
100 bitwise-equal changed-input graph replays per rank. The measured current
early-shared Graph event median was 632.67 us (late-shared control: 630.34 us);
both had zero median shared tail after dispatch, so no early-shared speedup is
claimed for T64. Separate eager traces retain the larger operator input shapes.

The routed leaf accepts optional `w13_smooth_scale` and `w2_smooth_scale`
independently. Existing `enable_smooth_quant` checkpoint allocation/loaders are
reused; absent parameters are passed as `None`, and unfilled allocated smooth
parameters retain their identity initialization. Preparation converts each
present tensor to contiguous FP32, without assuming the other exists. Supported
layouts are `[width]`, `[1,width]`, and `[local_experts,width]`, where widths are
group hidden H and intermediate I respectively. Expert-dependent rows use local shard indices.
Preparation normalizes a shared W13 vector to `[1,H]` and materializes shared
W2 scales as `[local_experts,I]` once for compatibility with official DSQ.
The standalone binding and library must both be rebuilt for the optional inputs.

Input smoothing multiplies FP32 gathered activations before amax/INT8 conversion;
output smoothing multiplies SwiGLU before output amax/requantization. This matches
the existing init-routing `scale` and DSQ `quant_scale` conventions. Already
prepared quantized weights are not smoothed again. Missing scales do not cause
identity allocations or additional operators in forward. Both Vector subblocks
retain identical row-parallel work and reuse existing UB storage. GMM2/finalize
and shared-expert execution are unchanged.

The replacement calls `fused_init_routing_mm13_swiglu`, whose public binding and
timeline name match. Expert-parallel filtering, device prefix/task generation,
and the mixed Cube/Vector pipeline run in one launch. AIVs join before prefix
construction and again before signaling their paired Cube; the subsequent
pipeline uses pair-local synchronization. The two-launch binding remains as a
test control (`ROUTED_SINGLE_KERNEL=0` in the full-layer harness). Metadata contains INT64 counts,
stable source-route maps, and 32-row expert/row/count tiles. Nonlocal and identity
routes have finalize row index -1. The two Vector subblocks execute the **same
algorithm on disjoint rows**: gather BF16, dynamically quantize, publish input,
consume GMM13, and apply dequant/SwiGLU/requant. Both participate in every
pair-local ready/ack phase, including a subblock with no rows. They are not
split into input-only and output-only workers.

Two slots per Cube bound INT8 input and INT32 accumulator workspaces; full
expanded activation and accumulator tensors are not materialized. The scratch
buffers still traverse GM, so this is not an all-on-chip implementation.
Scalar FP32 scale division matches the existing init-routing quantizer exactly.
Explicit Vector dependencies and TPipe-managed event IDs are necessary when
compiling without inferred synchronization.
Input scales have separate 32-byte blocks per Vector, independent of the
possibly odd row split, to avoid concurrent partial updates to one GM block.
The current implementation reserves a 16-FP32, 32-byte-aligned scale slice per
Vector and processes eight rows per UB batch. Cube streams NZ expert weights
without L2 allocation, matching the official GMM single-M-block cache policy.
Cold-cache tests are necessary: repeated standalone calls can otherwise hide
the cost of the 192 MiB weight tensor behind L2 reuse.

For the full-layer A/B harness above, set `ROUTED_FUSION=1` and the binding
environment variable. Both variants then retain identical fused exchange and
shared work; only the expert segment differs. Captures record both complete
layers, and graph correctness compares outputs exactly. Failure fixtures retain
per-rank inputs, IDs, counts, row maps and intermediate quantized results for
independent replay. Standalone tests/source and build instructions live in the
kernel repository's `cann-flash-ops/src/fused_init_routing_mm13_swiglu/` directory.

The integer-filter / compile-time-specialized version measured 177.348 us for
the complete fused segment versus 250.544 us unfused, including metadata and
inter-kernel gaps (median of steady-state rank medians across EP16). Whole-layer
ACL Graph event latency was 611.156 us versus 543.483 us with shared experts and
communication enabled in both variants. All 16 ranks passed eager bitwise
equality and 100 changed-input graph replays. These are fixed-shape measurements,
not a claim for other shapes or a 170 us latency guarantee.

### Optional MM2/finalize fusion

`gmoe_expert_solution="flash_npu_routed_full"` additionally replaces the second
expert segment with `fused_mm2_fin_routing`. The existing `flash_npu_routed`
solution keeps its original GMM2/finalize tail for A/B testing; automatic
selection and communication/shared-expert paths are unchanged. Preparation
loads `TOKENSPEED_FUSED_MM2_LIBRARY` and installs the selected tail in the plan.
Forward does not inspect a device type or environment variable.

The new kernel takes the existing INT8 SwiGLU output, FP32 row scales, INT64
counts, token-major INT32 row map, NZ W2 weights, BF16 W2/channel scales and
FP32/BF16 route weights. FP32 route weights are rounded to BF16 inside the
Vector stage, eliminating the external cast. It retains the intermediate
BF16 rounding and ordered FP32 TopK sum.
SmoothQuant is already applied before SwiGLU output quantization, so it must
not be applied again by MM2. Missing/independent smooth weights remain supported.

The standalone MM2 kernel has no EP-specific shape cases: 1..1024 experts,
arbitrary token/TopK counts within INT32 indexing, 32-aligned K/N dimensions,
bounded task batches and explicit M/N tails. The combined two-segment model
solution still inherits the existing MM13 kernel's documented shape limits.
Two identical row-parallel Vector workers consume each Cube's double-buffered
INT32 tiles. A final all-AIV join precedes deterministic token reduction; this
first implementation retains BF16 expanded-row scratch, not FP32 atomic sums.

Set `MM2_FUSION=1 ROUTED_FUSION=1` in the full-layer validation harness to
compare both-segment fusion against MM13-only fusion with identical exchange,
shared experts and weights. Standalone shape/graph/performance tests are in
the kernel repository's `cann-flash-ops/src/fused_mm2_fin_routing/tests/`.

## Identity experts

Identity experts are initially kept outside the expert leaf so their cost and
placement remain visible in profiling:

- EGP2 keeps group-local route IDs unchanged; the active expert range drops every
  non-local route, including identity IDs, before the local GMM;
- EGP1 keeps the `E..E+Z` IDs while setting `expert_num=E+Z` and
  `active_expert_range=[0,E)`, which lets `npu_moe_init_routing_v2` compact them
  out without a fake expert or a custom kernel;
- the local GMM consumes only the prefix described by `expert_counts`, and
  finalize consumes `row_indices`; inactive buffer tails are not masked or read;
- the model adds `group_input * sum(identity route weights)` exactly once
  before the reverse group exchange.

The composed identity epilogue remains separate as the correctness/performance
control. The optional fused post-stage applies the same identity contribution
once, after its RS, in the split-core restore operator described above.

## Decode timeline

The implementation was profiled on eight NPUs with 32 distinct tokens per
rank.  To model the configured `target_topk=10` with `moe_topk=16`, the A/B
profile forces six identity-zero routes per token.  Rank 0's second eager
iteration showed:

| EGP1 zero handling | init routing | GMM13 | GMM2 | group-first span |
| --- | ---: | ---: | ---: | ---: |
| dummy real expert | 98.2 us | 650.8 us | 278.4 us | 3494.6 us |
| inactive `E..E+Z` range | 103.4 us | 539.8 us | 269.9 us | 3328.0 us |

Filtering saves about 166.6 us (4.8%) for this distribution, mainly by keeping
the six zero routes out of GMM13.  It is therefore preferable to a finalize-only
fusion: the latter cannot recover wasted expert matmul work.

ACL Graph replay was also validated for both layouts.  Measuring from the first
group exchange through the reverse group exchange on rank 0 gives about 1.58 ms
for `EP=8, G=8, EGP=1` and 1.62 ms for `EP=8, G=4, EGP=2`. The separate
zero/identity bookkeeping is about 42 us and 36 us respectively, including
mask/reduce plus multiply/add.
That is the upper bound for a later identity-epilogue fusion and is intentionally
left unfused for now.

The W8A8 path was profiled separately on 910B2C with `G=8`, 32 distinct Decode
tokens per rank, model hidden size 4096, group hidden size 512, 384 real plus 32
identity experts, top-k 16 (10 real routes), and intermediate size 1024. Both
W13 and W2 reported `FRACTAL_NZ` and executed as
`aclnnGroupedMatmulWeightNz`. Ten graph replays on rank 0 averaged 0.822 ms for
EP8/EGP1 and 0.857 ms for EP16/EGP2. The EP16 trace additionally contains a
256x512-to-512x512 AllGather and a 512x512-to-256x512 ReduceScatter. Eager
shape traces and graph traces are emitted by
`test/ci_system/profile_lite_gmoe_w8a8.py`.

The production EP16 W8A8 path uses dynamic-token init routing
(`quant_mode=1`). With the Lite counting-route kernel, each source token is
dynamically quantized once and its INT8 row and scale are reused by all local
expert routes. Across 16 ranks and 20 ACL Graph replays, the rank medians are
58.3 us for init routing, 143.1 us for GMM13, 167.8 us for
`DequantSwigluQuant`, and 63.2 us for GMM2. The median device span from the
first group-exchange transpose through the final transpose is 0.867 ms. In an
immediately preceding trace with the stock dynamic init-routing path, init
routing was 95.6 us and the corresponding span was 0.895 ms.

The same profiler now supports `--weight-dtype bf16`. For EP16/G8/EGP2 with
32 Decode tokens per rank, the BF16 leaf uses `MlpSplitSwiglu` (about 29 us in
ACL Graph) between GMM13 (about 283 us) and GMM2 (about 117 us). The optimized
EP16 init-routing is about 56 us across ranks, compared with 70--71 us for the
stock generic multi-core sort. The median graph device span from the first
group-exchange transpose through the final transpose is about 0.916 ms over 20
replays.
