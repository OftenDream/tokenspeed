# Lite NPU OE: device hash and host table lookup

## Runtime contract

Lite NPU uses the same request-history and TP-fragment contract as the GPU OE
path. The only physical difference is table placement:

| Stage | GPU Device tables | NPU Host tables |
| --- | --- | --- |
| append current packed tokens | Device | NPU |
| n-gram hash | Device | NPU |
| TP-local table lookup | Device | AI Core over registered Host mapping |
| packed activation | Device | NPU |
| local projection and reduction | Device | NPU |

Both placements use `runtime-full-history`. `committed_lengths` is a read-only
publication pointer: the append/hash operator may write speculative tokens
after that pointer, but only verification publishes a longer committed prefix.
The checkpointed three-token-tail implementation remains available as an
explicit compatibility path. It is not used by the default full-history NPU
path and therefore adds no cache group or PD wire state to that path.

## Ownership and memory

With `oe_neighbor_num=5` and `oe_split_num=4`, the checkpoint has
`(5 - 1) * 4 = 16` branches. Under TP8, rank `r` owns the two complete 256-wide
branches `r` and `r + 8`, for a local lookup/projection width of 512 and one
eighth of the table payload per rank. AI Core reads only the selected BF16 rows
through the Host mapping. This new 4096-hidden layout is currently registered
for TP8 only. Its `oe_vocab_size_ratio=59.604` gives
`modulus0=int(163840*59.604)+1=9765520`.

The existing 3072-hidden checkpoint remains supported. Its
`oe_neighbor_num=4`, `oe_split_num=4` layout has 12 branches: TP4 owns three
complete branches `r/r+4/r+8` (local width 768), while TP8 owns complete branch
`r` plus one 128-wide half of branch `8+r//2` (local width 384).

Host loading adopts each local safetensors tensor without a copy. Both local
branches are row-contiguous `[rows, 256]` mappings, preserving file-backed mmap
storage and avoiding a multi-GiB anonymous copy.

The local projection is `[512, 4096]`. The portable Torch `project_add_word_`
path combines the local word and OE partials with the checkpoint normalization.
Ignored current tokens have zero fused-lookup activation, but Lite restores the
learned row-zero OE projection alongside the unscaled word embedding. Only
normalization is bypassed. One TP all-reduce follows the local projection.

After loading weights and placing the projection on its execution device,
`initialize_host_runtime()` copies each TP-local table fragment's row zero into
one Device buffer `[1, local_width]`. The full tables remain on Host, including
their mmap views. This non-persistent buffer is initialized once and reused for
special-token projection in eager and Graph execution; forward performs no
Host row-zero lookup or Host-to-Device copy. Device-table execution is unchanged.

## Operator boundary

The generic `append_packed_lookup_` API is unchanged. Ascend lowers it to one
public Flash operator, `npu_append_packed_oe_lookup`, which accepts the local
`oe_tables` directly. All tables in one call must use the same placement:

- CPU mmap tables: model setup registers each row-contiguous file mapping once
  with `aclrtHostRegister` and caches its Device-visible pointer; the AscendC
  kernel gathers directly from Host memory on every inference launch;
- NPU tables: the same AscendC kernel gathers from the HBM pointer.

The caller only supplies the history, tables, and caller-owned NPU output.
There is no standalone row-ID tensor or CPU lookup API in the runtime.

The setup-only `register_host_tables_` interface has an Ascend implementation;
NVIDIA intentionally leaves it unimplemented. GPU Device-table lookup
continues to use its fused CuTe implementation.

The fused lookup follows the GPU boundary rules exactly: request
intervals are ragged, inactive graph-padding requests output `-1`, EOS in the
look-back stops the hash, and Lite ignored tokens segment history. The current
ignored token is still appended as raw history but emits no OE row. Hashing
reads earlier tokens from the packed input and only reads positions before the
current round from committed history, so execution does not depend on core
scheduling order.

Host table registration accepts row-contiguous mmap tensors. Lookup copies BF16
bits exactly and writes zero for a `-1` row.

## Graph execution

The fused append/hash/lookup kernel runs directly inside model forward for both
Host and Device tables. Its Host pointers are registered once at startup and
remain stable across NPUGraph capture/replay; graph inputs provide the current
tokens and full-history metadata. No external activation staging, row-ID D2H,
or packed-activation H2D transfer exists.

The production decode chain can be captured as one ACL Graph containing the
fused Host lookup, Torch local projection/merge, and the TP8 HCCL all-reduce. The
contention profile runs both DP replicas concurrently on 16 ranks. The 32
tokens contributed by each TP rank are gathered across TP8 before OE, so every
rank runs OE on 256 tokens. Each rank owns two real-size Host fragments,
produces a `[256, 512]` local activation, projects with `[512, 4096]`, and
all-reduces a `[256, 4096]` BF16 buffer. A world synchronization precedes the
measured one graph launch containing a 16-rank HCCL start synchronization and
50 consecutive iterations. This puts process-side launch skew before the OE
work so both DP replicas exercise Host memory concurrently.

## Validation

The flash-kernel suite compares 100,000 deterministic random cases against an
independent scalar CPU oracle, including history writes, EOS, ignored-token
boundaries, TP fragments, and a non-contiguous half-table view. A second test
repeats the same fused NPU operation 10,000 times and requires bitwise identical
BF16 activations and final history. TokenSpeed unit tests
guard mmap-view adoption, full-history wiring, direct Host lookup during model
forward, runtime-plan selection, and absence of the retired checkpoint-tail
cache group.
