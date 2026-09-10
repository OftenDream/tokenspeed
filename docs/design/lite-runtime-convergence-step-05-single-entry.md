# Flash-Lite GPU/NPU runtime convergence: Step 5 single entry

## Goal

Both CUDA and Ascend load the released Flash-Lite checkpoint as
`FLASHLocalConfig` and instantiate `FLASHLocalForCausalLM`. This step removes
the temporary `LiteConfig` / `LiteForCausalLM` identity after Steps 1--4 moved
all required physical leaves and runtime contracts behind the shared entry.

## Kept differences

The single model entry does not make the backends physically identical:

- CUDA keeps separate KDA projections, its existing MLA/MoE kernels, and
  Device OE with runtime full history;
- Ascend keeps packed KDA, packed EP-local Grouped MoE, Weight-NZ leaves,
  Host OE with NPU full-history hashing, and its P/D collective schedule;
- mapping, kernel and cache capability selection remains construction-time or
  backend-registry policy. Decoder/model forward contains no device branch.

The shared decoder also preserves physical-leaf collective ownership. Packed
Ascend Grouped MoE returns an EP- and dense-reduced output, while replicated
Ascend MLA returns a complete output after its KVP merge. Those two leaves skip
the generic post-stage collective but retain the shared residual and norm
sequence. CUDA leaves keep the existing deferred collective path. This choice
is based on the selected leaf contract, not a runtime device check.

## Config and registry cutover

`FLASHLocalConfig.from_dict()` detects the released strict Flash-Lite schema
from its feature markers, checks the complete raw field set before constructor
defaults can hide omissions, and then preserves the checkpoint architecture as
`FLASHLocalForCausalLM`. Older Flash-KDA configs which do not declare that
schema keep the existing permissive compatibility path.

The runtime config registry, attention registry, model registry and tokenizer
gate retain only `FLASHLocalForCausalLM`. `LiteConfig`, `LiteForCausalLM` and
their architecture rewrite are deleted rather than retained as aliases.

## Checkpoint ledger

The existing strict source-key/shape/dtype ledger is retained as
`FLASHLocalCheckpointLayout`. It validates the released final-HF stream before
the shared loader canonicalizes names and delegates physical packing/sharding
to the selected KDA, MLA, MoE and OE leaves. The ledger is enabled only for the
strict released schema, so legacy Flash-KDA aliases keep their prior loader
compatibility.

For strict streams the loader rejects unexpected, duplicate, wrong-shape,
wrong-dtype and missing source tensors. It does not prescribe Device versus
Host OE storage or separate versus packed KDA targets.

## OE state protocol

Ascend OE now defaults to the same runtime full-token history as the GPU path.
The former `CheckpointedTailOERecipe`, `lite_oe` cache group and
`layer.0.lite.oe.context` field have been removed from production; verification
alone publishes the committed pointer on the full-history path. The retired CPU
implementation survives only as an independent precision oracle under tests.

## Deletion boundary

Delete:

- `configs/lite_config.py` and its exports/registry entries;
- `models/lite.py` and its `EntryClass`;
- `LiteForCausalLM` architecture membership and model-name gates;
- Lite-named production recipe/config/model helpers which have shared owners.

Historical design documents and test filenames may retain “Lite” as the model
product name. Active tests must import only the shared runtime entry and shared
physical leaves.

## Validation

- static scan: no active runtime reference to `LiteConfig`,
  `LiteForCausalLM`, `models.lite`, or `model_type == "lite"`;
- config: missing raw strict fields fail, architecture remains
  `FLASHLocalForCausalLM`, legacy non-strict Flash-KDA still parses;
- checkpoint: complete strict stream loads through both selected physical
  layouts; duplicate, unexpected, missing, shape and dtype failures remain;
- NPU: cumulative unit suite, NPU leaf/graph tests, 8P8D service smoke and
  GSM8K first 100;
- GPU: platform-neutral separate-layout tests and, when reachable, the first
  two GPUs for smoke and GSM8K first 100;
- exact pre-commit, signed-off commit, push, and local/tracking/remote SHA
  equality.

The 8P8D service acceptance keeps the bounded topology already admitted by the
NPU integration: both roles run an eight-rank lockstep attention group, dense
TP8, KDA TP8, replicated MLA weights/cache, and MoE EP8. This deliberately does
not enable Prefill CP8, Decode attention-DP/KVP8, distributed page ownership,
or fragment transfer; those remain the final distributed-cache TODO. In this
lockstep topology an Extend step uses the same feature all-gather, expert
all-reduce and dense projection schedule as Decode. The CP token-gather
schedule remains selected only when `mapping.attn.cp_size > 1`.

The bounded CANN 9.0 acceptance requires exclusive ownership of all sixteen
NPUs. Mooncake 0.3.9 generates an ADXL rank table without `device_port`, so its
device-side HIXL listener uses port 16666 on each selected NPU independently of
the host-side `HCCL_IF_BASE_PORT`. A concurrent process on any selected NPU is
therefore a resource-admission failure, not a model/runtime fallback case.

## Bring-up findings

The first 8P8D service reached all three serving roles but generated repeated
tokens. The source-key ledger was complete, yet the generic fused-MLA mapping
did not target the selected separate q/kv/g projections. Direct target-owned
loading removed every checkpoint warning; an all-zero strict stream also
proved that every non-empty target parameter was covered.

Output remained wrong until the old temporary entry and shared decoder were
compared at the collective boundary. The shared decoder reduced both the
already-complete packed MoE output and replicated MLA output a second time.
All tensors stayed finite, so NaN/Inf checks could not detect this error. The
minimal leaf-ownership guards above restore the old mathematical schedule and
have dedicated regression tests that fail if either outer collective runs.

## Final validation results

- The cumulative locked-environment suite on the second 16-card 910B node
  passed: `215 passed, 3 skipped` in 31.59 seconds. The skips are explicit
  board-test admissions, not failures.
- The final 8P8D service completed model load, Decode BS1/BS2 graph capture,
  Prefill, AscendDirect cache transfer and Decode. The frozen smoke request
  returned a coherent 83-token answer and all service logs contain zero
  checkpoint `not in model` warnings, traceback, fatal/error, NaN or Inf
  records.
- EvalScope 1.9.1 completed GSM8K `main/test` first 100 with 4-shot,
  batch-size 2, temperature 0 and max 512: 100/100 requests completed,
  zero API errors, zero empty outputs, 15 max-length outputs and `15/100`.
  The previous temporary-entry run scored `16/100`; the two runs share all 15
  new correct samples and differ only at index 59 in correctness. A standalone
  replay reproduced the new `67 - 120 = -53` branch, so this is a stable
  generation choice rather than a batch/graph transport failure. There is no
  Megatron golden claim in this step.
- The requested CUDA host remained unreachable (`No route to host`). No real
  two-GPU result is claimed. Platform-neutral tests cover the unchanged CUDA
  separate KDA/MLA/MoE and Device/full-history OE contracts; the production
  CUDA physical leaves were not modified by the single-entry cutover.
- The r18 service was stopped by its recorded launcher PID and its complete
  logs were copied to the local validation-artifact directory before cleanup.
