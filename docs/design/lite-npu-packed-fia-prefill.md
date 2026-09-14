# Packed MLA cached Prefill on Ascend

Cached Extend can read the existing BF16 `[pages,page_size,1,576]` latent
cache directly through `custom.npu_mla_fia_packed_prefill.out`. The adapter
selects it from package capabilities and input metadata. No environment flag
enables this path. Missing packages, older packages without the new operator,
and unsupported geometry use the existing native TND FIA implementation.
Errors after dispatch propagate to the caller; attention does not repeat a
cache write or retry a failing operator.

The query is absorbed TND `[T,H,576]`: latent width 512 and auxiliary width 64.
Each request supplies its cumulative query end and its noncumulative live KV
length. For request `b`, `q_len[b] = q_end[b] - q_end[b-1]` (the initial end is
zero), and `prefix[b] = kv_len[b] - q_len[b]`. Query `i` can see KV positions
`0..prefix[b]+i`, inclusive. The packed operator uses the vendored FIA multiquery
tiles, paged readers, and right-down causal mask. It does not loop over Decode
calls. Query lengths are 1–4096, page sizes 64/128, heads 1–64, batch 1–1024,
and live KV lengths at most 1048576. Every query length must be positive and
no greater than its KV length. Page IDs remain the caller's valid device data.

The adapter materializes only the two query components. It passes a view of
the persistent cache with the original page stride and storage offset, plus
the existing page table and reusable 2048-square compressed causal mask.
Host length conversion, when needed, belongs to the measured complete call.
No storage format, page-table ownership, PD payload, cache lifetime, scheduler,
or cache write location changes.

The first chunk remains explicit Q/K/V attention with logical value width 128.
That interface cannot consume a 512-wide latent result directly. Algebraically,
absorption is possible when explicit K/V are the matching linear projections
of the same latent cache: move the K projection into Q and apply the V projection
after attention. BF16 reassociation can change rounding, so mathematical
equivalence alone is insufficient to switch the model's first-chunk algorithm.
The runtime continues selecting absorbed Extend only for a positive prefix;
the operator's prefix-zero cases validate its own absorbed semantics.
Auxiliary channels contribute to attention but this NoPE path applies no RoPE.

The existing BSH single-query `npu_mla_fia_packed` ABI and native graph task-update
handler remain unchanged. Prefill is validated in eager execution; adding its
operator does not register a second Decode handler or change live-length updates.

Tests cover capability fallback, execution errors, explicit first-chunk value
width, actual Extend calls, and Decode graph updates. NPU performance and build
provenance are version-bound and recorded with the operator's validation artifacts;
single-layer results do not establish serving TTFT or TPOT improvements.
