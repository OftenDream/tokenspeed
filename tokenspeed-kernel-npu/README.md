# TokenSpeed-Kernel-NPU

TokenSpeed-Kernel-NPU contains the Ascend-specific operators used by
TokenSpeed. Keeping these implementations in a standalone package mirrors the
AMD package layout and keeps the TokenSpeed runtime vendor-neutral.

The initial Ascend path provides paged MHA, RMSNorm, Q/K RMSNorm, rotary
embedding, and the Triton-Ascend import adapter required by CANN 9.0.0.

For development from this repository:

```bash
test/ci_system/install_triton_ascend.sh
```

The validated stack is CANN 9.0.0, PyTorch 2.9.0, `torch_npu` 2.9.0.post2,
Transformers 5.12.0, Triton 3.2.0, and Triton-Ascend 3.2.1. The setup script
records every Python package mutation needed by this Ascend path, including the
`apache-tvm-ffi==0.1.13` build dependency and editable install of this package.

TokenSpeed applications should continue to import operators from
`tokenspeed-kernel`; it owns registration and dispatch to this package.

MLA Decode can use the optional `custom.npu_mla_fia_packed.out` operator to
read the persistent packed cache directly, without a full-cache layout copy.
The fast path requires BF16, a single query token, latent/auxiliary widths
512/64, 1–64 query heads, batch 1–1024, contiguous packed pages of 64 or 128
tokens, an int32 contiguous page table, and a context bound at most 1M.
The installed operator must expose Host `int[]` lengths and the active
`torch_npu` must provide the NPUGraph handler registration interface. Missing
optional capabilities or unsupported geometry retain native FIA; broken
installed-package imports and execution errors are not silently swallowed.
Use a matching extension and custom OPP bundle: registering a Python schema
alone does not install its ACLNN implementation. Keep the Lite-capable MLA
prolog and causal-conv implementations available when adding the packed reader.

The packed handler participates in the existing `graph.update()` flow:
Q layout conversion and output allocation happen outside task-update, and
each replay updates Host lengths while reusing the captured output addresses.
Cache writes, page-table refresh, Prefill, and MLA prolog are unchanged.
Set `TOKENSPEED_NPU_PACKED_FIA=0` before starting the process to force native
FIA for A/B; the default is enabled when all capabilities match. Do not change
the selection after graphs have been captured.

The native fallback retains FIA with native NPUGraph. Its split Q/cache inputs are
made contiguous before FIA dispatch, so layout copies execute on graph replay
rather than inside FIA task-update. K and V share the same dense latent copy;
the persistent cache remains packed. This copies the full allocated cache,
not just live pages, so cache-capacity costs must be included in benchmarks.

The changed-input, cache, page-table and sequence-length replay check is:

```bash
pytest -q test/runtime/test_lite_mla_eager.py -k decode_graph_updates_live_lengths
pytest -q tokenspeed-kernel-npu/test/test_mla_packed.py
```

Lite KDA can additionally build the pinned public AscendC operator subset:

```bash
test/ci_system/install_public_kda_ops.sh
```

This command builds against the active CANN/PyTorch ABI and installs the
generated artifact below this package. It does not install vLLM-Ascend as a
Python dependency. Set `TOKENSPEED_PUBLIC_KDA_SOURCE_DIR` to an exact prepared
public checkout when the build host has no network access.

The optimized Lite causal-conv path is supplied separately by the `flash_ops`
run package and wheel built from `flash-npu-kernel`. If that package or its
schemas are unavailable, TokenSpeed keeps the existing Torch fallback.

Run the operator correctness suite on a visible NPU with:

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
PYTHONPATH="${PWD}/python:${PWD}/tokenspeed-kernel/python:${PWD}/tokenspeed-kernel-npu/python:${PYTHONPATH:-}" \
    pytest -q tokenspeed-kernel-npu/test
```

For the complete Qwen3-0.6B launch command, ACL Graph capture sizes, serving
limits, and a request example, see the
[Ascend model recipe](../docs/recipes/models.md#qwen3-06b-on-ascend-npu).

## Serving Dependencies in a Source Checkout

When running TokenSpeed directly from a checkout through `PYTHONPATH`, install
the SMG serving packages explicitly. Run these commands from the repository
root with the same Python interpreter used to launch TokenSpeed:

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh

python -m pip install \
    "tokenspeed-smg==1.9.0.post20260823" \
    "tokenspeed-smg-grpc-proto==0.4.14.post20260823" \
    "tokenspeed-smg-grpc-servicer==0.8.0.post20260823" \
    "grpcio==1.81.1" \
    "grpcio-health-checking==1.81.1" \
    "grpcio-reflection==1.81.1" \
    "protobuf>=5.26.0,<7" \
    "viztracer"
```

If the host requires an outbound HTTP proxy, configure it for the installation
without committing credentials:

```bash
export PROXY_URL="http://<username>:<password>@<proxy-host>:<port>"
export HTTP_PROXY="${PROXY_URL}"
export HTTPS_PROXY="${PROXY_URL}"
export http_proxy="${PROXY_URL}"
export https_proxy="${PROXY_URL}"
```

Verify the gRPC servicer and the source paths together:

```bash
PYTHONPATH="${PWD}/python:${PWD}/tokenspeed-kernel/python:${PWD}/tokenspeed-kernel-npu/python:${PYTHONPATH:-}" \
    python -m smg_grpc_servicer.tokenspeed --help
```
