# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).with_name("install-pytorch.py")
SDK_DOCKERFILE = Path(__file__).resolve().parents[1] / "Dockerfile"
SPEC = importlib.util.spec_from_file_location("install_pytorch", SCRIPT)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules["install_pytorch"] = MODULE
SPEC.loader.exec_module(MODULE)

# Requires-Dist of torch==2.13.0+cu130 (manylinux x86_64 and aarch64).
TORCH_REQUIRES_DIST = [
    "filelock",
    "typing-extensions>=4.10.0",
    "setuptools>=77.0.3",
    "sympy>=1.13.3",
    "networkx>=2.5.1",
    "jinja2",
    "fsspec>=0.8.5",
    "cuda-toolkit[cublas,cudart,cufft,cufile,cupti,curand,cusolver,cusparse,nvjitlink,nvrtc,nvtx]"
    '==13.0.3; platform_system == "Linux"',
    'cuda-bindings<14,>=13.0.3; platform_system == "Linux" and python_version < "3.15"',
    'nvidia-cudnn-cu13==9.20.0.48; platform_system == "Linux"',
    'nvidia-cusparselt-cu13==0.8.1; platform_system == "Linux"',
    'nvidia-nccl-cu13==2.29.7; platform_system == "Linux"',
    'nvidia-nvshmem-cu13==3.4.5; platform_system == "Linux"',
    'triton==3.7.1; platform_system == "Linux" and python_version < "3.15"',
    'optree>=0.13.0; extra == "optree"',
]
LINUX_PY312 = {
    "platform_system": "Linux",
    "python_version": "3.12",
    "sys_platform": "linux",
}
CUDA_13_0 = MODULE.Platform("x86_64", 13, 0)
CUDA_13_2 = MODULE.Platform("aarch64", 13, 2)


def classify(requires_dist=TORCH_REQUIRES_DIST, target=CUDA_13_0, **kwargs):
    return MODULE.classify_requirements(requires_dist, target, environment=LINUX_PY312, **kwargs)


class FakeProbe:
    def __init__(self, packages=None, libraries=None):
        self.packages = packages or {}
        self.libraries = libraries or {}

    def dpkg_version(self, package):
        return self.packages.get(package)

    def find_library(self, soname):
        return self.libraries.get(soname)


def test_default_version_matches_build_image_pin() -> None:
    match = re.search(r"^ARG PYTORCH_VERSION=(\S+)$", SDK_DOCKERFILE.read_text(), re.MULTILINE)
    assert match is not None
    assert match.group(1) == MODULE.DEFAULT_VERSION


def test_only_pure_python_dependencies_come_from_pip() -> None:
    resolution = classify()

    assert resolution.pip == [
        "filelock",
        "typing-extensions>=4.10.0",
        "setuptools>=77.0.3",
        "sympy>=1.13.3",
        "networkx>=2.5.1",
        "jinja2",
        "fsspec>=0.8.5",
        "cuda-bindings<14,>=13.0.3",
    ]
    assert resolution.skipped == ["triton==3.7.1"]
    assert not any(r.startswith(("nvidia-", "cuda-toolkit")) for r in resolution.pip)


def test_cuda_dependencies_map_to_system_packages() -> None:
    packages = {r.apt_package: r for r in classify(target=CUDA_13_2).system}

    assert packages["libcublas-13-2"].soname == "libcublas.so.13"
    assert packages["cuda-cudart-13-2"].soname == "libcudart.so.13"
    assert packages["cuda-nvtx-13-2"].soname is None
    assert packages["libcudnn9-cuda-13"].pinned_version == "9.20.0.48"
    assert packages["libnccl2"].pinned_version == "2.29.7"
    assert packages["libcusparselt0-cuda-13"].pinned_version == "0.8.1"
    assert packages["libnvshmem3-cuda-13"].soname == "libnvshmem_host.so.3"
    assert len(packages) == 11 + 4


def test_markers_for_other_platforms_are_ignored() -> None:
    resolution = MODULE.classify_requirements(
        TORCH_REQUIRES_DIST,
        CUDA_13_0,
        environment={**LINUX_PY312, "platform_system": "Windows"},
    )

    assert resolution.system == []
    assert "cuda-bindings<14,>=13.0.3" not in resolution.pip


@pytest.mark.parametrize(
    "requirement",
    [
        "nvidia-cufile-cu13==1.0",
        "nvidia-cuda-runtime==13.0.96",
        "cuda-cccl==13.0",
        "libfoo-cu13==1.0",
        "cuda-toolkit[cublas,cudss]==13.0.3",
    ],
)
def test_unknown_cuda_dependencies_are_rejected(requirement: str) -> None:
    with pytest.raises(MODULE.InstallError, match="Unknown"):
        classify([requirement])


def test_toolkit_newer_than_system_cuda_is_rejected() -> None:
    with pytest.raises(MODULE.InstallError, match=r"install the \+cu130 wheel"):
        classify(['cuda-toolkit[cublas]==13.2.1; platform_system == "Linux"'])


@pytest.mark.parametrize(
    ("major", "minor", "tag"),
    [(13, 0, "cu130"), (13, 1, "cu130"), (13, 2, "cu132"), (13, 4, "cu132")],
)
def test_wheel_tag_is_newest_supported_minor(major: int, minor: int, tag: str) -> None:
    assert MODULE.select_wheel_tag(major, minor) == tag


@pytest.mark.parametrize(("major", "minor"), [(12, 8), (14, 0)])
def test_unsupported_cuda_major_is_rejected(major: int, minor: int) -> None:
    with pytest.raises(MODULE.InstallError, match="not supported"):
        MODULE.select_wheel_tag(major, minor)


def test_cuda_version_is_read_from_environment_or_version_file(tmp_path: Path) -> None:
    assert MODULE.detect_cuda_version({"CUDA_VERSION": "13.2.1"}, tmp_path) == (13, 2)

    version_file = tmp_path / "usr/local/cuda/version.json"
    version_file.parent.mkdir(parents=True)
    version_file.write_text(json.dumps({"cuda": {"version": "13.0.2"}}))
    assert MODULE.detect_cuda_version({}, tmp_path) == (13, 0)

    with pytest.raises(MODULE.InstallError, match="Could not detect"):
        MODULE.detect_cuda_version({}, tmp_path / "missing")


def make_sbsa_root(path: Path) -> Path:
    (path / "usr/local/cuda/targets/sbsa-linux").mkdir(parents=True)
    return path


def test_sbsa_platform_is_accepted(tmp_path: Path) -> None:
    target = MODULE.detect_platform(make_sbsa_root(tmp_path), {"CUDA_VERSION": "13.2.0"}, "aarch64")

    assert target == CUDA_13_2
    assert target.wheel_tag == "cu132"


def test_jetpack_6_cuda_layout_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "usr/local/cuda/targets/aarch64-linux").mkdir(parents=True)

    with pytest.raises(MODULE.InstallError, match="JetPack 6"):
        MODULE.detect_platform(tmp_path, {"CUDA_VERSION": "13.0.2"}, "aarch64")


@pytest.mark.parametrize(
    ("setup", "machine", "message"),
    [
        (lambda root: (root / "usr/local/cuda/targets/sbsa-linux").rmdir(), "aarch64", "SBSA"),
        (lambda _root: None, "ppc64le", "Unsupported architecture"),
    ],
)
def test_unsupported_platforms_are_rejected(tmp_path, setup, machine, message) -> None:
    root = make_sbsa_root(tmp_path)
    (root / "usr/local/cuda/targets/aarch64-linux").mkdir()
    setup(root)

    with pytest.raises(MODULE.InstallError, match=message):
        MODULE.detect_platform(root, {"CUDA_VERSION": "13.0.2"}, machine)


def test_cuda_12_platform_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(MODULE.InstallError, match="CUDA 12.8 is not supported"):
        MODULE.detect_platform(tmp_path, {"CUDA_VERSION": "12.8.1"}, "x86_64")


@pytest.mark.parametrize(
    ("debian", "upstream"),
    [("2.29.7-1+cuda13.2", "2.29.7"), ("9.20.0.48-1", "9.20.0.48"), ("1:0.9.0.3-1", "0.9.0.3")],
)
def test_upstream_version_strips_epoch_and_revision(debian: str, upstream: str) -> None:
    assert MODULE.upstream_version(debian) == upstream


def system_requirement(name: str, target=CUDA_13_2):
    return next(r for r in classify(target=target).system if r.apt_package == name)


def test_same_or_newer_apt_versions_meet_pins() -> None:
    probe = FakeProbe(
        packages={"libnccl2": "2.29.7-1+cuda13.2", "libcudnn9-cuda-13": "9.27.0.42-1"}
    )

    assert MODULE.system_problem(system_requirement("libnccl2"), probe) is None
    assert MODULE.system_problem(system_requirement("libcudnn9-cuda-13"), probe) is None


def test_older_apt_version_is_a_problem() -> None:
    probe = FakeProbe(packages={"libnccl2": "2.28.9-1+cuda13.0"})

    problem = MODULE.system_problem(system_requirement("libnccl2"), probe)

    assert problem == "libnccl2 2.28.9-1+cuda13.0 is older than 2.29.7"


def test_libraries_installed_outside_apt_are_accepted() -> None:
    probe = FakeProbe(libraries={"libcusparseLt.so.0": "/opt/cusparselt/lib/libcusparseLt.so.0"})

    assert MODULE.system_problem(system_requirement("libcusparselt0-cuda-13"), probe) is None


def test_missing_libraries_fail_with_apt_hint() -> None:
    requirements = [system_requirement("libcublas-13-2"), system_requirement("libnccl2")]

    with pytest.raises(MODULE.InstallError) as error:
        MODULE.require_system_libraries(requirements, FakeProbe())

    assert "libcublas-13-2 is not installed (cuda-toolkit[cublas])" in str(error.value)
    assert "apt-get install libcublas-13-2 libnccl2" in str(error.value)


def test_bundled_cuda_libraries_are_removed_and_torch_libraries_kept(tmp_path: Path) -> None:
    names = [
        "libtorch_cuda.so",
        "libc10_cuda.so",
        "libcaffe2_nvrtc.so",
        "libtorch_nvshmem.so",
        "libgomp.so.1",
        "libcudart.so.13",
        "libcudnn_graph.so.9",
        "libnccl.so.2",
    ]
    for name in names:
        (tmp_path / name).touch()

    removed = MODULE.remove_bundled_libraries(tmp_path, lambda _name: None)

    assert sorted(path.name for path in removed) == [
        "libcudart.so.13",
        "libcudnn_graph.so.9",
        "libnccl.so.2",
    ]


def test_bundled_nvpl_is_kept_unless_apt_provides_it(tmp_path: Path) -> None:
    torch_lib = tmp_path / "torch/lib"
    torch_lib.mkdir(parents=True)
    for name in ("libnvpl_blas_lp64_gomp.so.0", "libnvpl_lapack_lp64_gomp.so.0"):
        (torch_lib / name).touch()
    system = {
        "libnvpl_blas_lp64_gomp.so.0": "/usr/lib/aarch64-linux-gnu/libnvpl_blas_lp64_gomp.so.0"
    }

    removed = MODULE.remove_bundled_libraries(torch_lib, system.get)

    assert [path.name for path in removed] == ["libnvpl_blas_lp64_gomp.so.0"]
    assert (torch_lib / "libnvpl_lapack_lp64_gomp.so.0").exists()


def test_bundled_nvpl_registered_with_ldconfig_is_kept(tmp_path: Path) -> None:
    (tmp_path / "libnvpl_blas_core.so.0").touch()

    removed = MODULE.remove_bundled_libraries(tmp_path, lambda name: str(tmp_path / name))

    assert removed == []


def test_kineto_lines_are_stripped(tmp_path: Path) -> None:
    config = tmp_path / "share/cmake/Caffe2/Caffe2Config.cmake"
    config.parent.mkdir(parents=True)
    config.write_text("set(A 1)\nfind_library(kineto_LIBRARY kineto)\nset(B 2)\n")

    MODULE.strip_kineto(tmp_path)

    assert config.read_text() == "set(A 1)\nset(B 2)\n"


def test_pip_cuda_libraries_are_reported(tmp_path: Path) -> None:
    site = tmp_path / "site-packages"
    maps = [
        "/usr/local/cuda-13.0/targets/x86_64-linux/lib/libcublas.so.13",
        f"{site}/nvidia/cu13/lib/libcudart.so.13",
        f"{site}/torch/lib/libtorch_cuda.so",
        f"{site}/torch/lib/libnvpl_blas_core.so.0",
        f"{site}/nvidia/nvpl/lib/libnvpl_lapack_core.so.0",
    ]

    assert MODULE.pip_locations(maps, site) == [maps[1], maps[4]]


def test_pip_cuda_distributions_are_forbidden() -> None:
    names = [
        "nvidia-cublas",
        "nvidia_cudnn_cu13",
        "nvidia-cuda-runtime",
        "cuda-toolkit",
        "cuda-bindings",
        "nvidia-ml-py",
        "nvidia-nvpl-blas",
    ]

    assert MODULE.forbidden_distributions(names) == [
        "cuda-toolkit",
        "nvidia-cublas",
        "nvidia-cuda-runtime",
        "nvidia-cudnn-cu13",
    ]


def test_pip_cuda_library_files_are_reported(tmp_path: Path) -> None:
    lib = tmp_path / "nvidia/cu13/lib"
    lib.mkdir(parents=True)
    for name in ("libcublas.so.13", "libcudart.so.13", "README"):
        (lib / name).touch()

    assert MODULE.pip_cuda_library_files(tmp_path) == [
        str(lib / "libcublas.so.13"),
        str(lib / "libcudart.so.13"),
    ]
    assert MODULE.pip_cuda_library_files(tmp_path / "missing") == []


def test_help_lists_commands() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"], check=False, capture_output=True, text=True
    )

    assert result.returncode == 0, result.stderr
    assert "install" in result.stdout
    assert "check" in result.stdout


def test_pip_locations_handle_dist_packages_and_relative_runpath() -> None:
    site = Path("/usr/local/lib/python3.12/dist-packages")
    paths = [
        f"{site}/torch/lib/../../nvidia/cu13/lib/libcublas.so.13",
        "/usr/local/cuda-13.0/lib64/libcublas.so.13",
        f"{site}/torch/lib/libnvpl_blas_core.so.0",
        f"{site}/nvidia/nvpl/lib/libnvpl_blas_core.so.0",
    ]

    assert MODULE.pip_locations(paths, site) == [paths[0], paths[3]]
