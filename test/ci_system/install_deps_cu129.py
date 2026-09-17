# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Experimental H100/H200 + CUDA 12.9 source install into an active Python 3.11 venv."""

import os
import runpy
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

PYPI = "https://pypi.org/simple"
PYTORCH = "https://download.pytorch.org/whl/cu129"
WHEELS = "https://lightseek.org/whl/cu129/"
NATIVE_PACKAGES = {
    "tokenspeed-deepgemm",
    "tokenspeed-flashmla",
    "tokenspeed-fast-hadamard-transform",
    "tokenspeed-fa3",
    "tokenspeed-trtllm-kernel",
    "tokenspeed-flashkda",
}


def kernel_requirements(workspace: Path) -> list[str]:
    # Evaluate metadata from the checkout, without running any build command.
    import setuptools

    captured = {}
    original = setuptools.setup
    try:
        setuptools.setup = lambda **kwargs: captured.update(kwargs)
        runpy.run_path(str(workspace / "tokenspeed-kernel" / "python" / "setup.py"))
    finally:
        setuptools.setup = original
    return captured["install_requires"]


def pip_install(arguments: list[str], index: str) -> None:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--index-url", index, *arguments],
        check=True,
    )


def main() -> None:
    workspace = Path(os.environ.get("WORKSPACE", Path(__file__).resolve().parents[2]))
    workspace = workspace.resolve()
    os.environ.update(
        TOKENSPEED_KERNEL_BACKEND="cuda",
        TOKENSPEED_KERNEL_CUDA_VARIANT="cu129",
        FLASHINFER_CUDA_ARCH_LIST="9.0a",
        MAX_JOBS=os.environ.get("BUILD_AND_DOWNLOAD_PARALLEL", "16"),
        PIP_CONSTRAINT=str(workspace / "requirements" / "nvidia-cu129-constraints.txt"),
        PIP_CONFIG_FILE=os.devnull,
        PIP_INDEX_URL=PYPI,
    )
    # Identical native package versions exist on PyPI with a different CUDA
    # ABI. Each install below uses one source, including in configured hosts.
    os.environ.pop("PIP_EXTRA_INDEX_URL", None)
    os.environ.pop("PIP_FIND_LINKS", None)
    os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
    print(
        f"Experimental Hopper/cu129 install: {workspace}, Python: {sys.executable}",
        flush=True,
    )

    pip_install(["--upgrade", "pip", "setuptools==83.0.0", "wheel"], PYPI)
    pip_install(["torch", "torchvision"], PYTORCH)

    requirements = kernel_requirements(workspace)
    native = [req for req in requirements if req.split("==")[0] in NATIVE_PACKAGES]
    deepep = [req for req in requirements if req.startswith("tokenspeed-deepep==")]
    runtime = tomllib.loads((workspace / "python" / "pyproject.toml").read_text())
    mooncake = [
        req
        for req in runtime["project"]["dependencies"]
        if req.startswith("tokenspeed-mooncake>=")
    ]
    pip_install(["--force-reinstall", "--no-deps", *native, *mooncake], WHEELS)
    # Preserve a manually installed development wheel satisfying the main pin.
    pip_install(["--no-deps", *deepep], WHEELS)

    flashinfer = next(
        req.split("==")[1]
        for req in requirements
        if req.startswith("flashinfer-python==")
    )
    cubin = (
        "https://github.com/flashinfer-ai/flashinfer/releases/download/"
        f"v{flashinfer}/flashinfer_cubin-{flashinfer}-py3-none-any.whl"
    )
    # Resolve native wheels' transitive dependencies after the right binaries
    # are installed. Matching public versions (including local suffixes) stay.
    pip_install([*requirements, *mooncake, cubin], PYPI)
    pip_install(
        [
            "--no-build-isolation",
            "--no-deps",
            "-e",
            str(workspace / "tokenspeed-kernel" / "python"),
        ],
        PYPI,
    )
    with tempfile.TemporaryDirectory(prefix="tokenspeed-cu129-scheduler-") as build_dir:
        pip_install(
            [
                str(workspace / "tokenspeed-scheduler"),
                f"--config-settings=build-dir={build_dir}",
            ],
            PYPI,
        )
    pip_install(["--no-build-isolation", "-e", str(workspace / "python")], PYPI)
    subprocess.run([sys.executable, "-m", "pip", "check"], check=True)


if __name__ == "__main__":
    main()
