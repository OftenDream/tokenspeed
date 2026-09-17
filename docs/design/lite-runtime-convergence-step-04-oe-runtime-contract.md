# Flash-Lite GPU/NPU runtime convergence: Step 4 OE runtime contract

## Scope

This document described the original convergence step. The active Ascend
contract has since been superseded by
`lite-npu-oe-device-hash-host-table.md`. The checkpointed-tail runtime was
subsequently removed; only its CPU math remains as a test oracle.

| Runtime plan | Table lookup | Context owner |
| --- | --- | --- |
| `device` + `runtime-full-history` | existing Device OE kernel | `RuntimeStates` request-token history |
| `host` + `runtime-full-history` | CPU mmap table lookup and Device activation staging | `RuntimeStates` request-token history |

Table placement and context ownership remain separate resolved fields. Only the
two rows above are admitted in this step; unimplemented cross-products fail at
startup rather than silently changing placement.

## Resolution boundary

`ServerArgs.oe_table_placement` stores the deployment request and defaults to
`auto`. The selected model class declares the placements and state providers it
can construct on each backend. The model loader resolves that declaration before
allocating model weights and writes the two final values to runtime
`ModelConfig`:

- `oe_table_placement`: `device`, `host`, or `None` for a non-OE model;
- `oe_state_provider`: `runtime-full-history`,
  `cache-checkpointed-tail`, or `None`.

An explicit placement is accepted only when the selected class declares the
exact backend pair. `auto` prefers `device`, then `host`, among declared pairs.
For a non-OE checkpoint, `auto` is a no-op and an explicit placement is an
error. The HF config and global runtime arguments are not rewritten.

The temporary Lite entry declares only its existing Ascend Host plan. The shared
FLASHLocal entry declares the existing NVIDIA Device plan and the migrated
Ascend Host plan. LongCat keeps its existing Device/full-history construction.

## Shared OE layer ownership

The Host table implementation and its checkpointed-tail preparer move from the
temporary model file to `layers/over_embedding.py`. Their n-gram hash, special
token handling, normalization, CPU storage adoption, three-token restoration,
page publication, and graph-stable staging are unchanged.

The shared FLASHLocal model selects one physical OE leaf at construction:

- Device placement keeps `LongCatOverEmbedding`, including TP-local tables and
  Device full-history lookup;
- Host placement keeps the ordinary word embedding plus the migrated Host OE
  component. The external-input hook prepares Host lookup activations and the
  model consumes them in the same word-plus-OE projection equation.

The model forward branches on the constructed component, never on a device or
architecture string. The checkpoint loader delegates OE tensors to that
component, so the model-level loader does not encode Host versus Device tensor
layout.

## Runtime consumers

- request-token history allocation reads only
  `oe_state_provider == "runtime-full-history"`;
- the OE cache recipe is selected only for
  `oe_state_provider == "cache-checkpointed-tail"`;
- both currently admitted providers keep Prefill graph disabled, while their
  validated Decode graph paths remain unchanged;
- Host external-input hooks bind the existing stable protocol IDs `lite_oe` and
  `layer.0.lite.oe.context`; Device/full-history does not allocate that cache
  group.

This prevents the Ascend Host path from allocating a second full-history tensor
and prevents the NVIDIA Device path from allocating tail pages.

## Deliberately unchanged

- Device OE kernels, TP fragments, full-history publication, and prefix seeding;
- Host lookup math, tail restore/publication, staging, and PD snapshot protocol;
- Prefill graph support (still disabled for both current providers);
- checkpoint source names and the temporary strict whole-checkpoint loader;
- KDA, MLA, Grouped MoE, Weight-NZ, and backend fusion choices.

The temporary `LiteForCausalLM` entry remains during this step. Step 5 switches
the runtime registry to the shared entry and removes the temporary config/model
and strict-layout duplicate.

## Validation

The focused suite must prove:

- CLI default/choices and explicit no-fallback behavior;
- non-OE `auto` no-op and explicit placement rejection;
- class-declared CUDA Device and Ascend Host plan resolution before model
  construction;
- cache recipe, Prefill graph, and history allocation depend on the resolved
  provider rather than `LiteForCausalLM`, model type, or embed class;
- Host n-gram IDs, special-token behavior, projection, storage aliasing, slot
  reuse, checkpoint restore, and graph staging remain unchanged after moving;
- the shared FLASHLocal Host leaf consumes the same Host OE tensors and external
  state hook as the temporary entry;
- the existing Device OE and LongCat behavior remains covered without changing
  its physical implementation.

NPU validation runs the cumulative Lite suite plus the Host OE graph test on
NPU0. A real NVIDIA run is required only if the Device physical path changes;
otherwise its existing construction, loader, and semantic contract tests are
the gate when the configured GPU host is unavailable.

## Validation result

- Ascend NPU0 cumulative Lite/runtime suite: `176 passed, 3 skipped`;
- focused OE capability, loader, cache, graph, and launcher suite:
  `68 passed`;
- exact repository gate: `.venv/bin/pre-commit run --all-files` passed;
- the configured NVIDIA host remained unreachable, so this step makes no new
  real-GPU claim. The Device OE implementation itself was not changed.
