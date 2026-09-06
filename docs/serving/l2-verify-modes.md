# L2 verification modes

Enable verification with `TOKENSPEED_L2_VERIFY=1`.

`TOKENSPEED_L2_VERIFY_MODE=sync` (default) preserves the original diagnostic:
wait for Device producers, hash Device before D2H, hash Host after D2H and
before H2D, then wait for all H2D layers and hash Device before forward.
These extra waits can suppress asynchronous failures.

To investigate a failure that disappears with synchronous verification:

```sh
export TOKENSPEED_L2_VERIFY=1
export TOKENSPEED_L2_VERIFY_MODE=host
export TOKENSPEED_L2_VERIFY_LOG=/tmp/l2-host-verify.jsonl
export TOKENSPEED_L2_VERIFY_DUMP=""
```

Host mode adds no Device snapshots, `.cpu()` copies, or CUDA waits. It hashes
Host pages after the ordinary D2H completion event queries successfully,
before publishing the write ACK. At load submission it checks the Host page
against that baseline, while scheduler load tickets still pin its lifetime.
It preserves the normal per-layer readiness gates and asynchronous H2D/forward
overlap. Workspace retirement and other normal execution fences remain.

JSONL records carry `verify_mode`. A missing Device digest in a STORE record
means that Device verification was skipped, not that Device bytes matched.
No LOAD Device records are produced in host mode. A Host mismatch localizes
corruption to the interval between the stored Host baseline and the load
snapshot; it does not establish a particular writer as responsible.

Passing Host checks does not prove that D2H captured correct Device state,
that H2D wrote the correct destinations, or that consumers waited correctly.
In particular, do not add a Device check at ACK polling: an overlapping KDA
forward can legitimately have updated those pages by then.

Compare disabled verification, host mode, and sync mode using the same workload
and seeds. Keep dumps disabled initially. CPU hashing, logging, and disk I/O
still change scheduling, so host mode cannot guarantee reproduction. The
layer-0 stall branch should keep the stall enabled for all three comparisons.
