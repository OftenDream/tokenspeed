from __future__ import annotations

import os
import runpy
import shutil
import tarfile
from collections import Counter
from pathlib import Path

import pytest
import setuptools
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from setuptools import build_meta

SETUP_PY = Path(__file__).parents[1] / "python" / "setup.py"
REQUIREMENTS_DIR = SETUP_PY.parent / "requirements"


def _capture_install_requires(monkeypatch, backend: str) -> list[str]:
    setup_kwargs = {}
    monkeypatch.setenv("TOKENSPEED_KERNEL_BACKEND", backend)
    monkeypatch.setattr(
        setuptools, "setup", lambda **kwargs: setup_kwargs.update(kwargs)
    )
    runpy.run_path(str(SETUP_PY))
    return setup_kwargs["install_requires"]


def _direct_requirements(path: Path) -> list[str]:
    requirements = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        requirements.append(line)
    return requirements


def _expected_install_requires(backend: str) -> list[str]:
    requirements = _direct_requirements(REQUIREMENTS_DIR / f"{backend}.txt")
    requirements.extend(
        _direct_requirements(REQUIREMENTS_DIR / f"{backend}-thirdparty.txt")
    )
    return list(dict.fromkeys(requirements))


def _requirements_by_name(requirements: list[str]) -> dict[str, Requirement]:
    assert all(not requirement.startswith("-") for requirement in requirements)
    parsed = [Requirement(requirement) for requirement in requirements]
    names = [canonicalize_name(requirement.name) for requirement in parsed]
    duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
    assert not duplicates, f"duplicate dependency names: {duplicates}"
    return dict(zip(names, parsed, strict=True))


def test_cuda_install_requires_include_runtime_dependencies(monkeypatch) -> None:
    install_requires = _capture_install_requires(monkeypatch, "cuda")

    assert install_requires == _expected_install_requires("cuda")
    requirements = _requirements_by_name(install_requires)
    assert {
        "tokenspeed-proton",
        "tokenspeed-triton",
        "flashinfer-python",
        "nvidia-cudnn-frontend",
        "nvidia-ml-py",
        "nvtx",
        "torch",
        "tokenspeed-deepgemm",
    } <= requirements.keys()
    assert {"tokenspeed-kernel-amd", "tokenspeed-iris"}.isdisjoint(requirements)
    assert "tokenspeed-triton-kernels" not in requirements
    assert requirements["nvidia-cutlass-dsl"].extras == {"cu13"}
    assert str(requirements["nvidia-cudnn-frontend"].specifier) == "==1.26.0"


def test_opt_in_cu129_metadata_preserves_other_cuda_pins(monkeypatch) -> None:
    monkeypatch.setenv("TOKENSPEED_KERNEL_CUDA_VARIANT", "cu129")
    actual = _requirements_by_name(_capture_install_requires(monkeypatch, "cuda"))
    expected = _requirements_by_name(_expected_install_requires("cuda"))
    expected["nvidia-cutlass-dsl"] = Requirement("nvidia-cutlass-dsl==4.7.1")
    expected.pop("nvidia-cutlass-dsl-libs-cu13")
    expected["nvidia-cutlass-dsl-libs-cu12"] = Requirement(
        "nvidia-cutlass-dsl-libs-cu12==4.7.1"
    )
    expected.pop("tokenspeed-cutedsl-kda")

    assert actual == expected


@pytest.mark.parametrize("variant", ["", "cu129"])
def test_cu129_variant_preserves_rocm_metadata(monkeypatch, variant: str) -> None:
    monkeypatch.setenv("TOKENSPEED_KERNEL_CUDA_VARIANT", variant)

    assert _capture_install_requires(monkeypatch, "rocm") == _expected_install_requires(
        "rocm"
    )


@pytest.mark.parametrize(
    ("backend", "variant", "installs_requirements"),
    [("cuda", "", True), ("rocm", "cu129", True), ("cuda", "cu129", False)],
)
def test_only_cu129_recipe_skips_nested_pip(
    monkeypatch, backend: str, variant: str, installs_requirements: bool
) -> None:
    monkeypatch.setenv("TOKENSPEED_KERNEL_BACKEND", backend)
    monkeypatch.setenv("TOKENSPEED_KERNEL_CUDA_VARIANT", variant)
    monkeypatch.setattr(setuptools, "setup", lambda **_kwargs: None)
    namespace = runpy.run_path(str(SETUP_PY))
    install = namespace["_install_backend_build_requirements"]
    calls = []
    monkeypatch.setattr(namespace["subprocess"], "check_call", calls.append)
    monkeypatch.setitem(
        install.__globals__, "_refresh_python_install_paths", lambda: None
    )

    install(False)

    assert bool(calls) == installs_requirements
    if calls:
        assert calls[0][1:4] == ["-m", "pip", "install"]
        assert str(REQUIREMENTS_DIR / f"{backend}.txt") in calls[0]


@pytest.mark.parametrize(
    ("variant", "source_newer", "rebuilds"),
    [
        ("", False, False),
        ("cu130", False, False),
        ("cu129", False, True),
        ("", True, True),
    ],
)
def test_cu129_rebuilds_cached_cuda_kernels(
    tmp_path, monkeypatch, variant: str, source_newer: bool, rebuilds: bool
) -> None:
    monkeypatch.setenv("TOKENSPEED_KERNEL_BACKEND", "cuda")
    monkeypatch.setenv("TOKENSPEED_KERNEL_CUDA_VARIANT", variant)
    monkeypatch.setenv("TOKENSPEED_CUDA_ARCH", "90")
    monkeypatch.delenv("FLASHINFER_CUDA_ARCH_LIST", raising=False)
    monkeypatch.setenv("MAX_JOBS", "1")
    monkeypatch.setattr(setuptools, "setup", lambda **_kwargs: None)
    namespace = runpy.run_path(str(SETUP_PY))

    source = tmp_path / "cached.cu"
    source.touch()
    output_dir = tmp_path / "objs"
    library = output_dir / "cached" / "cached.so"
    library.parent.mkdir(parents=True)
    library.touch()
    source_mtime = 3000 if source_newer else 1000
    os.utime(source, (source_mtime, source_mtime))
    os.utime(library, (2000, 2000))

    builder = namespace["CudaKernelBuilder"]([("cached", [source], [])], verbose=False)
    monkeypatch.setitem(builder.run.__globals__, "CUDA_OBJS_DIR", output_dir)
    monkeypatch.setattr(builder, "_prepare_cuda_toolchain_env", lambda: None)
    monkeypatch.setattr(builder, "_resolve_include_dirs", lambda: [])
    monkeypatch.setattr(builder, "_resolve_cuda_lib_flags", lambda: [])
    calls = []
    monkeypatch.setattr(namespace["subprocess"], "check_call", calls.append)

    builder.run()

    if rebuilds:
        assert len(calls) == 2
        assert "-gencode=arch=compute_90a,code=sm_90a" in calls[0]
        assert calls[0][-4:-2] == ["-c", str(source)]
        assert calls[1][-2:] == ["-o", str(library)]
    else:
        assert not calls


def test_cuda_thirdparty_requirements_are_exactly_pinned() -> None:
    requirements = _requirements_by_name(
        _direct_requirements(REQUIREMENTS_DIR / "cuda-thirdparty.txt")
    )

    not_exactly_pinned = sorted(
        name
        for name, requirement in requirements.items()
        if {specifier.operator for specifier in requirement.specifier} != {"=="}
    )
    assert (
        not not_exactly_pinned
    ), f"CUDA third-party requirements must use ==: {not_exactly_pinned}"


def test_rocm_install_requires_exclude_cuda_dependencies(monkeypatch) -> None:
    install_requires = _capture_install_requires(monkeypatch, "rocm")

    assert install_requires == _expected_install_requires("rocm")
    requirements = _requirements_by_name(install_requires)
    assert {
        "tokenspeed-proton",
        "tokenspeed-triton",
        "tokenspeed-kernel-amd",
        "tokenspeed-iris",
        "torch",
    } <= requirements.keys()
    assert {
        specifier.operator
        for specifier in requirements["tokenspeed-kernel-amd"].specifier
    } == {">="}
    assert {
        "flashinfer-python",
        "nvidia-cutlass-dsl",
        "nvidia-cutlass-dsl-libs-cu13",
        "nvidia-ml-py",
        "nvtx",
        "tokenspeed-deepep",
        "tokenspeed-deepgemm",
        "tokenspeed-fa3",
        "tokenspeed-fa4",
        "tokenspeed-fast-hadamard-transform",
        "tokenspeed-flashmla",
        "tokenspeed-mla",
        "tokenspeed-trtllm-kernel",
        "tokenspeed-triton-kernels",
    }.isdisjoint(requirements)


def test_read_requirements_skips_installer_options_and_cycles(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("TOKENSPEED_KERNEL_BACKEND", "cuda")
    monkeypatch.setattr(setuptools, "setup", lambda **_kwargs: None)
    setup_namespace = runpy.run_path(str(SETUP_PY))

    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text(
        "-rsecond.txt\n--extra-index-url https://example.invalid/simple\n"
        "first-package==1\n",
        encoding="utf-8",
    )
    second.write_text(
        "--requirement=first.txt\n--find-links https://example.invalid/wheels\n"
        "second-package>=2\n",
        encoding="utf-8",
    )

    assert setup_namespace["_read_requirements"](first) == [
        "second-package>=2",
        "first-package==1",
    ]


def test_sdist_includes_requirements_and_python_sources(tmp_path, monkeypatch) -> None:
    source = tmp_path / "python"
    dist_dir = tmp_path / "dist"
    shutil.copytree(SETUP_PY.parent, source)
    dist_dir.mkdir()
    monkeypatch.setenv("TOKENSPEED_KERNEL_BACKEND", "cuda")
    monkeypatch.setenv("TOKENSPEED_KERNEL_GIT_SHA", "test")
    monkeypatch.chdir(source)

    archive_name = build_meta.build_sdist(str(dist_dir))

    with tarfile.open(dist_dir / archive_name) as archive:
        archived_files = {
            name.split("/", maxsplit=1)[1] for name in archive.getnames() if "/" in name
        }
    expected_files = {
        f"requirements/{path.name}" for path in REQUIREMENTS_DIR.glob("*.txt")
    }
    expected_files.update(
        path.relative_to(source).as_posix()
        for path in (source / "tokenspeed_kernel").rglob("*.py")
    )
    assert expected_files <= archived_files


def test_cuda_include_dirs_prefer_complete_toolkit_headers(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("TOKENSPEED_KERNEL_BACKEND", "cuda")
    monkeypatch.setattr(setuptools, "setup", lambda **_kwargs: None)
    setup_namespace = runpy.run_path(str(SETUP_PY))

    cuda_root = tmp_path / "cuda"
    cuda_include = cuda_root / "include"
    cuda_include.mkdir(parents=True)
    (cuda_include / "cuda_runtime.h").touch()
    (cuda_include / "cublas_v2.h").touch()

    site_packages = tmp_path / "site-packages"
    wheel_include = site_packages / "nvidia" / "cu13" / "include"
    wheel_include.mkdir(parents=True)
    (wheel_include / "cuda_runtime.h").touch()
    (wheel_include / "cublas_v2.h").touch()

    builder = setup_namespace["CudaKernelBuilder"]([], verbose=False)
    monkeypatch.setattr(builder, "_cuda_toolkit_roots", lambda: iter([cuda_root]))
    monkeypatch.setattr(builder, "_site_paths", lambda: iter([site_packages]))

    include_dirs = builder._resolve_include_dirs()

    assert str(cuda_include) in include_dirs
    assert str(wheel_include) not in include_dirs


def test_cuda_include_dirs_fall_back_from_partial_toolkit(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("TOKENSPEED_KERNEL_BACKEND", "cuda")
    monkeypatch.setattr(setuptools, "setup", lambda **_kwargs: None)
    setup_namespace = runpy.run_path(str(SETUP_PY))

    cuda_root = tmp_path / "cuda"
    cuda_include = cuda_root / "include"
    cuda_include.mkdir(parents=True)
    (cuda_include / "cuda_runtime.h").touch()

    site_packages = tmp_path / "site-packages"
    wheel_include = site_packages / "nvidia" / "cu13" / "include"
    wheel_include.mkdir(parents=True)
    (wheel_include / "cuda_runtime.h").touch()
    (wheel_include / "cublas_v2.h").touch()

    builder = setup_namespace["CudaKernelBuilder"]([], verbose=False)
    monkeypatch.setattr(builder, "_cuda_toolkit_roots", lambda: iter([cuda_root]))
    monkeypatch.setattr(builder, "_site_paths", lambda: iter([site_packages]))

    include_dirs = builder._resolve_include_dirs()

    assert str(cuda_include) not in include_dirs
    assert str(wheel_include) in include_dirs
