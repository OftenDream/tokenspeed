# Experimental Hopper / CUDA 12.9 installation

This source-install recipe targets H100/H200, CUDA Toolkit 12.9, and Python
3.11 on Linux. It installs the kernel package, scheduler, and Python runtime
from this checkout, including its Lite model support. GPU build and serving
validation must be performed on the target host.

With the virtual environment activated and CUDA Toolkit 12.9 available,
run this single installation command from the repository root:

```bash
CUDA_VARIANT=cu129 bash test/ci_system/install_deps.sh
```

For the first installation, prepare a fresh environment to avoid carrying
over CUDA 13 packages from a previous installation:

```bash
uv venv .venv-cu129 --python 3.11 --seed
source .venv-cu129/bin/activate
export CUDA_HOME=/usr/local/cuda
CUDA_VARIANT=cu129 bash test/ci_system/install_deps.sh
```

Set `CUDA_HOME` to the CUDA 12.9 toolkit location when it is installed
elsewhere. The host supplies its compiler, OpenSSL development files, and
required runtime libraries. This recipe does not run system package checks,
apt, or sudo. Each pip call uses the active Python interpreter; proxy
environment variables are inherited.

If a temporary DeepEP build is needed, install its wheel manually into this
environment before running the recipe. The wheel must be built for cu129 and
Python 3.11, and its public version must satisfy the existing pin in
`tokenspeed-kernel/python/requirements/cuda-thirdparty.txt`. Compatible local
version suffixes are preserved. The installer checks dependency versions,
not the CUDA ABI of an already installed DeepEP wheel.

The explicit `CUDA_VARIANT=cu129` value dispatches to
`test/ci_system/install_deps_cu129.py`. The recipe pins Torch/torchvision to
cu129 and obtains native TokenSpeed wheels from the LightSeek cu129 index.
It uses the existing checkout's kernel versions, switches the CuTe runtime
to cu12, and omits the CUDA-13-only CuteDSL KDA AOT package.
`requirements/nvidia-cu129-constraints.txt` contains the additional version
constraints. Each installation step selects its package index explicitly;
global pip configuration and extra package indexes are excluded so native
wheels with identical public versions cannot be chosen from a cu130 source.

The kernel build uses that adjusted dependency metadata, skips its default
build-time pip install, and rebuilds the in-tree CUDA kernels for `sm90a`.
The scheduler builds in a temporary directory. After updating this checkout,
rerun the same installation command in the activated environment to rebuild
the kernel and scheduler and refresh the editable runtime installation.
Without the cu129 opt-in, the existing NVIDIA and ROCm installers and package
metadata are unchanged.

The installer finishes with `python -m pip check`. On the target host, also
verify imports and the CLI before running model-specific kernel and serving
tests:

```bash
tokenspeed env
python -c 'import torch; import tokenspeed_kernel; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name())'
tokenspeed serve --help
```

Dependency checks alone do not validate binary ABI compatibility or model
serving correctness.
