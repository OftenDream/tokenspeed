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

import os
import shutil
import subprocess
from pathlib import Path

import install_deps_cu129 as installer


def test_shell_entrypoint_dispatches_cu129_from_another_directory(tmp_path) -> None:
    script_dir = Path(__file__).resolve().parent
    python = tmp_path / "python3"
    invocation = tmp_path / "invocation"
    python.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$INVOCATION"\nexit 37\n')
    python.chmod(0o755)
    # Only expose the path helper and fake Python: a broken dispatch must not
    # reach a real package manager or perform any installation on the host.
    (tmp_path / "dirname").symlink_to(shutil.which("dirname"))

    result = subprocess.run(
        ["/bin/bash", str(script_dir / "install_deps.sh")],
        cwd=tmp_path,
        env={
            "PATH": str(tmp_path),
            "CUDA_VARIANT": "cu129",
            "INVOCATION": str(invocation),
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=5,
    )

    assert result.returncode == 37, result.stdout + result.stderr
    assert invocation.read_text().splitlines() == [
        str(script_dir / "install_deps_cu129.py")
    ]


def test_pip_uses_active_interpreter_and_explicit_index(monkeypatch) -> None:
    commands = []
    monkeypatch.setattr(installer.sys, "executable", "/active-venv/bin/python")
    monkeypatch.setattr(
        installer.subprocess,
        "run",
        lambda args, *, check: commands.append((args, check)),
    )

    installer.pip_install(["--no-deps", "some-package==1.0"], installer.WHEELS)

    assert commands == [
        (
            [
                "/active-venv/bin/python",
                "-m",
                "pip",
                "install",
                "--index-url",
                installer.WHEELS,
                "--no-deps",
                "some-package==1.0",
            ],
            True,
        )
    ]


def test_recipe_uses_cu129_sources_and_preserves_manual_deepep(monkeypatch) -> None:
    workspace = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("TOKENSPEED_KERNEL_BACKEND", "cuda")
    monkeypatch.setenv("TOKENSPEED_KERNEL_CUDA_VARIANT", "cu129")
    requirements = installer.kernel_requirements(workspace)
    monkeypatch.setattr(installer, "kernel_requirements", lambda _: requirements)
    monkeypatch.setattr(
        os,
        "environ",
        {
            "WORKSPACE": str(workspace),
            "CUDA_HOME": "/opt/cuda-12.9",
            "BUILD_AND_DOWNLOAD_PARALLEL": "4",
            "PIP_EXTRA_INDEX_URL": "https://example.invalid/cu130",
            "PIP_FIND_LINKS": "/old-cu130-wheels",
            "PIP_CONFIG_FILE": "/old-pip.conf",
        },
    )
    installs = []
    checks = []
    monkeypatch.setattr(
        installer, "pip_install", lambda args, index: installs.append((args, index))
    )
    monkeypatch.setattr(
        installer.subprocess, "run", lambda args, *, check: checks.append(args)
    )

    installer.main()

    torch_args = [
        args
        for args, index in installs
        if index == "https://download.pytorch.org/whl/cu129"
    ]
    assert torch_args == [["torch", "torchvision"]]
    native_calls = [
        args for args, index in installs if index == "https://lightseek.org/whl/cu129/"
    ]
    assert len(native_calls) == 2
    native_args, deepep_args = native_calls
    assert native_args[:2] == ["--force-reinstall", "--no-deps"]
    assert (
        next(req for req in requirements if req.startswith("tokenspeed-fa3=="))
        in native_args
    )
    assert any(req.startswith("tokenspeed-mooncake>=") for req in native_args)
    assert not any("tokenspeed-deepep" in arg for arg in native_args)
    deepep_pin = next(
        req for req in requirements if req.startswith("tokenspeed-deepep==")
    )
    assert deepep_args == ["--no-deps", deepep_pin]

    dependency_args = next(
        args for args, _ in installs if deepep_pin in args and args != deepep_args
    )
    assert "--force-reinstall" not in dependency_args
    assert "[cu13]" not in " ".join(dependency_args)
    assert not any("tokenspeed-cutedsl-kda" in arg for arg in dependency_args)
    flashinfer_version = next(
        req.split("==")[1]
        for req in requirements
        if req.startswith("flashinfer-python==")
    )
    assert any(
        f"flashinfer_cubin-{flashinfer_version}-" in arg for arg in dependency_args
    )

    kernel_install, scheduler_install, runtime_install = installs[-3:]
    assert kernel_install == (
        [
            "--no-build-isolation",
            "--no-deps",
            "-e",
            str(workspace / "tokenspeed-kernel" / "python"),
        ],
        installer.PYPI,
    )
    scheduler_args, scheduler_index = scheduler_install
    assert scheduler_index == installer.PYPI
    assert scheduler_args[0] == str(workspace / "tokenspeed-scheduler")
    build_setting = next(
        arg for arg in scheduler_args if arg.startswith("--config-settings=build-dir=")
    )
    build_dir = Path(build_setting.split("=", 2)[2])
    assert build_dir.name.startswith("tokenspeed-cu129-scheduler-")
    assert not build_dir.exists()
    assert runtime_install == (
        ["--no-build-isolation", "-e", str(workspace / "python")],
        installer.PYPI,
    )

    assert os.environ["FLASHINFER_CUDA_ARCH_LIST"] == "9.0a"
    assert os.environ["TOKENSPEED_KERNEL_CUDA_VARIANT"] == "cu129"
    assert os.environ["PIP_CONSTRAINT"] == str(
        workspace / "requirements" / "nvidia-cu129-constraints.txt"
    )
    assert os.environ["PIP_CONFIG_FILE"] == os.devnull
    assert os.environ["PIP_INDEX_URL"] == installer.PYPI
    assert "PIP_EXTRA_INDEX_URL" not in os.environ
    assert "PIP_FIND_LINKS" not in os.environ
    assert os.environ["CUDA_HOME"] == "/opt/cuda-12.9"
    assert os.environ["MAX_JOBS"] == "4"
    assert checks == [[installer.sys.executable, "-m", "pip", "check"]]
