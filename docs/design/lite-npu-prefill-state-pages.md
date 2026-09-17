# Lite NPU Prefill recurrent-state page access

The KDA Prefill scan gathers one initial recurrent matrix per request and
publishes one final matrix per request. On Ascend, indexing a noncontiguous
CacheArena field with Torch can materialize the entire field before the gather
and around the scatter. This makes a single request's state traffic grow with
configured cache capacity.

## Layout and ownership

Decode determines the persistent layout. At TP8, Lite uses FP32 K-major
`[pages, 4, 128, 128]` recurrent state, with element strides
`[73728, 16384, 128, 1]` in the current CacheArena plan. Each page contains
65536 state elements; the remaining page stride belongs to the existing arena
packing. Conv state is BF16 `[pages, 3, 1536]` with strides
`[147456, 1536, 1]`. MLA and state fields continue to share the same allocation.
No Prefill persistent cache, transpose, or alternative page table is introduced.

The MLA reader reshapes the same arena's BF16 latent field into
`[pages, 64, 1, 576]`, with strides `[36864, 576, 576, 1]`. Its writer uses
token-slot indices, while Decode reads an int32 page table and sequence lengths.
The packed FIA reader consumes that field directly when its operator and input
capabilities match; the native fallback materializes separate 512/64-channel
inputs. These interfaces and lifetimes are preserved.

The scheduler still owns allocation, checkpoint selection, separate input and
output page IDs, fresh-page zeroing, prefix matching, transfer and reclamation.
The backend continues to derive state indices once per cache group. Kernel
helpers consume those validated indices; they do not allocate or choose pages.

## Request-scoped copies

`tokenspeed_kernel.ops.copy.state` provides gather/scatter helpers. For Ascend
views with contiguous elements inside a page, a tiled Triton copy addresses
`page_id * page_stride + offset` directly. Only selected pages are accessed.
Page offsets use int64 arithmetic, independent of the total arena size.
Other devices and views with noncontiguous page payloads retain Torch indexing.
The runtime continues through the same Prefill scan and final-state publication.

The compact `[requests, heads, key_dim, value_dim]` initial/final state tensors
are transient inputs and outputs of the existing chunk algorithm. Gathering
finishes before any state publication, so repeated reads and crossed input/output
pages preserve snapshot semantics. Destination IDs must be distinct and valid;
source values must not alias the persistent pool.

Fresh rows still select a safe working page instead of physical page zero, then
mask the gathered recurrent matrix to logical zero. A resumed row retains its
selected input matrix. The next 4096-token chunk reads the preceding chunk's
published state through the scheduler's indices. Decode consumes the same bytes
and retains its graph-stable metadata and page-stride-aware recurrent kernel.

Preparation returns the same safe input page IDs to the convolution leaf without
copying convolution state. KDA reads its checkpoint directly and publishes to the
separate destination; the single-index Mamba leaf stages its input at its own call.

## Validation

`test_state_rows.py` checks FP32/BF16, irregular row tails, page gaps, storage
offsets, repeated reads, crossed writes, strided indices, empty batches, and the
Torch fallback. `test_lite_kda_state_pages.py` uses actual CacheArena fields to
check fresh and resumed rows, poisoned null/working pages, byte-exact protection
of all unselected storage, two consecutive 4096-token Prefill chunks, and the
handoff to BS32 Decode graph with changed inputs and read/write pages.

Performance admission uses alternating profiler-off trials against the pinned
baseline in one process, at the same physical geometry and with identical public
KDA artifacts. Separate profiles verify that whole-pool AsStrided/ViewCopy work
is eliminated. Public KDA gate/cumsum and chunk stages remain required algorithm
steps and must not be counted as redundant launches. Full-model timings and
operator timings are reported separately.

## Q/K and featurewise-beta preparation

After fixing state traffic, the TP8 Prefill preparation is combined into one
Triton launch: FP32 Q/K L2 normalization with epsilon `1e-12`,
`sqrt(sigmoid(beta_logits) + 1e-10)`, K/V scaling, conversion to the input dtype,
while Torch retains the small scalar unit-beta fill for the public ChunkKda ABI.
Input token/head strides are read directly; the output tensors remain the existing
compact BF16/FP16 public-op inputs. No persistent storage is added. Launches use
16 rows when the token count is divisible by four, otherwise four rows, so every
program stores complete rows. Repeated tail tests exposed incorrect output with
a masked final 16-row program on Ascend.

The specialized path accepts `H=4`, `K=V=128` and contiguous channels on NPU.
Other shapes/strides retain the existing Torch preparation. Gate preprocessing
remains inside `KdaGateCumsum`; its chunk-local cumulative sum and the subsequent
chunk algorithm retain their original interfaces and stages. Decode preparation
and recurrent update continue to use the installed flash package.

The preparation tests cover BF16/FP16, four seeds, strided packed Q/K/V and beta,
zero norms, saturated beta, and token counts 32, 63, 65, 256 and 4096, with three
consecutive calls per case. The full public Prefill reference tests and the
two-chunk Decode handoff exercise the integrated
path. The preparation candidate is admitted using the complete prepare/gate/chunk
chain, rather than just its isolated elementwise kernel time.
