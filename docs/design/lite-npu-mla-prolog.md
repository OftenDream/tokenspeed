# Lite NPU MLA Decode prolog

## Fusion and package boundary

The optional `flash_ops` MLA prolog fuses Q/KV down-projection, Q/KV RMSNorm,
Q up-projection, query absorption and current-token packed-cache writes.
Lite uses NoPE: the auxiliary 64 dimensions remain present but are not rotated.
Gate projection, attention (including FIA's external contiguous copies),
value projection, output gating and output projection stay outside the prolog.
The model's `_project_q_with_mla_prolog` helper only invokes the prolog and
concatenates its query outputs. Attention, gate handling and output projection
are explicit stages in `forward`, not hidden inside the projection helper.

The runtime calls the vendor-neutral `tokenspeed_kernel.mla_prolog` facade.
The NPU leaf adapter owns optional package discovery, device/dtype/shape/NZ
admission and the call to `torch.ops.custom.npu_mla_prolog_v3`.
TS adds no MLA C++ binding, source patch, OPP build or GE converter. Registration
and the converter belong to the installed package. The existing public KDA
loader/build machinery is unchanged.

A package must expose the V3 schema with `enable_rope` and
`ckvkr_repo_mode`, and contain the Lite packed NoPE implementation. Schema
discovery alone cannot verify the installed device binary. The adapter passes
`enable_rope=False`, `ckvkr_repo_mode=1`, `cache_mode="PA_BSND"` and
non-quantized modes explicitly. V3 still requires rope-shaped placeholders for
output-shape inference; their values are unused for NoPE. The ignored separate
KR-cache argument receives a small placeholder, never another managed cache.

## Runtime ownership and fallback

The model retains only orchestration: pure-Decode eligibility, communication
ordering, existing cache storage and backend-owned write locations. This follows
the existing model-side MLA prewrite boundary exposed by
`AttentionBackend.write_locations`; it does not introduce a cache manager.
The adapter owns hardware-specific constraints: BF16, hidden/heads 3072/32 or
4096/64, ranks 1536/512, head dimensions 128/64, NZ projection weights, and
contiguous packed cache `[pages, page_size, 1, 576]`.

Backend-owned write locations may be int32 or int64. The V3 operator requires
int64, so the NPU adapter widens int32 locations on-device after admission,
without modifying the backend buffer. Native graph capture includes this cast;
replay uses the refreshed locations. Its cost is included in performance tests.

Replicated MLA weights may run with attention TP8. A prolog that writes cache
cannot bypass a required pre-attention token AllGather, so that case preserves
the primitive projection/communication path. Prefill, mixed/speculative/empty
batches, unsupported inputs and an absent or incompatible package also fall
back. Admission returns `None` before any cache write; execution errors from
an eligible op propagate rather than retrying after a possible mutation.

## Weight orientation and normalization fallback

The prolog requires logical `[in, out]` NZ projection weights. Q-A/KV-A already
use this orientation; Q-B changes from `[out, in]` to `[in, out]`. Both
orientations are NZ; this is not a new Weight-NZ switch. Conversion happens
once during weight loading under the existing Decode Weight-NZ configuration,
not every forward.

When the prolog is ineligible but Q-B was prepared transposed, the primitive
path normalizes the Q/KV latents and calls the transposed linear projection.
The portable branch of `FusedRMSNorm` invokes two separate RMSNorm operations:
it is a fallback, not part of the prolog fusion. The untransposed path retains
the existing normalize/project helper.

## Validation boundary

CPU-only tests cover package/schema discovery, metadata admission, exact V3
arguments, no-write fallback, error propagation, backend-owned cache locations,
communication ordering, and primitive fallback. The operator golden and native
NPU-graph replay tests remain available but require an installed Lite-capable
package and hardware.

NPU golden/replay tests cover both Lite geometries, including BS32 with the
backend's int32 indices and updated cross-page write locations on replay.
These tests require the installed Lite-capable device binary. End-to-end
performance comparisons must also verify prolog kernel hits in the profile;
successful fallback execution alone is not evidence that fusion was enabled.

This extends the primitive baseline in `lite-npu-phase-05-mla.md`.
