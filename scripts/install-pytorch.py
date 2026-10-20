#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Install and validate PyTorch against the system CUDA libraries used by Holoscan.

PyTorch wheels declare their CUDA runtime as pip wheels (``cuda-toolkit[...]`` and
``nvidia-*-cu13``). Installing those next to the system CUDA stack loads two copies of
libraries such as cuBLAS into one process. This helper installs ``torch`` without its
dependencies, installs only the pure-Python dependencies with pip, and requires every
CUDA dependency to be met by a system library.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path

try:
    from packaging.markers import default_environment
    from packaging.requirements import Requirement
    from packaging.version import InvalidVersion, Version
except ImportError:  # pragma: no cover - fall back to the copy vendored by pip
    from pip._vendor.packaging.markers import default_environment
    from pip._vendor.packaging.requirements import Requirement
    from pip._vendor.packaging.version import InvalidVersion, Version

DEFAULT_VERSION = "2.13.0"
DEFAULT_INDEX_URL = "https://download.pytorch.org/whl"
SUPPORTED_ARCHITECTURES = ("x86_64", "aarch64")
SUPPORTED_CUDA_MAJOR = 13
# CUDA minor versions with a ``+cu13Y`` build of DEFAULT_VERSION on download.pytorch.org.
WHEEL_CUDA_MINORS = (0, 2)

# CUDA-named pip packages that ship no CUDA libraries and are safe to install with pip.
PIP_CUDA_HELPERS = frozenset({"cuda-bindings", "cuda-pathfinder", "cuda-python"})
# Only torch.compile needs triton; install it separately when required.
SKIPPED_PACKAGES = frozenset({"triton"})
DRIVER_LIBRARIES = frozenset({"libcuda.so.1", "libnvidia-ml.so.1"})


@dataclass(frozen=True)
class SystemComponent:
    """A system APT package and shared library that replace a pip CUDA dependency."""
    apt_prefix: str
    soname: str | None
    versioned_package: bool = False

    def apt_package(self, cuda_major: int, cuda_minor: int) -> str:
        """Return the APT package name for the given system CUDA version."""
        if self.versioned_package:
            return self.apt_prefix.format(major=cuda_major)
        return f"{self.apt_prefix}-{cuda_major}-{cuda_minor}"


# ``cuda-toolkit`` extras, met by the CUDA toolkit packages matching the system CUDA.
CUDA_TOOLKIT_COMPONENTS = {
    "cublas": SystemComponent("libcublas", "libcublas.so.{major}"),
    "cudart": SystemComponent("cuda-cudart", "libcudart.so.{major}"),
    "cufft": SystemComponent("libcufft", "libcufft.so.12"),
    "cufile": SystemComponent("libcufile", "libcufile.so.0"),
    "cupti": SystemComponent("cuda-cupti", "libcupti.so.{major}"),
    "curand": SystemComponent("libcurand", "libcurand.so.10"),
    "cusolver": SystemComponent("libcusolver", "libcusolver.so.12"),
    "cusparse": SystemComponent("libcusparse", "libcusparse.so.12"),
    "nvjitlink": SystemComponent("libnvjitlink", "libnvJitLink.so.{major}"),
    "nvrtc": SystemComponent("cuda-nvrtc", "libnvrtc.so.{major}"),
    # NVTX 3 is header-only; the headers are needed by Torch's CMake configuration.
    "nvtx": SystemComponent("cuda-nvtx", None),
}

# Versioned CUDA libraries released outside the toolkit, keyed by pip name template.
VERSIONED_COMPONENTS = {
    "nvidia-cudnn-cu{major}": SystemComponent("libcudnn9-cuda-{major}", "libcudnn.so.9", True),
    "nvidia-cusparselt-cu{major}": SystemComponent(
        "libcusparselt0-cuda-{major}", "libcusparseLt.so.0", True
    ),
    "nvidia-nccl-cu{major}": SystemComponent("libnccl2", "libnccl.so.2", True),
    "nvidia-nvshmem-cu{major}": SystemComponent(
        "libnvshmem3-cuda-{major}", "libnvshmem_host.so.3", True
    ),
}

# CUDA runtime libraries that must never be loaded from a pip location.
CUDA_LIBRARY_PATTERN = re.compile(
    r"^lib(cudart|cublas|cublasLt|cudnn\w*|cufft\w*|cufile\w*|cupti|curand|cusolver\w*|"
    r"cusparse|cusparseLt|nccl|nvJitLink|nvrtc[\w-]*|nvToolsExt|nvshmem\w*)\.so(\.|$)"
)
NVPL_LIBRARY_PATTERN = re.compile(r"^libnvpl_\w+\.so(\.|$)")
PIP_CUDA_DISTRIBUTION_PATTERN = re.compile(
    r"^nvidia-(cublas|cuda-[\w-]+|cudnn|cufft|cufile|cupti|curand|cusolver|cusparse|cusparselt|"
    r"nccl|nvjitlink|nvrtc|nvshmem|nvtx)(-cu\d+)?$|^cuda-toolkit$"
)
PIP_DIRECTORIES = {"site-packages", "dist-packages"}


class InstallError(RuntimeError):
    """Raised when PyTorch cannot be installed or validated on this system."""


@dataclass(frozen=True)
class Platform:
    """The detected architecture and system CUDA version."""
    architecture: str
    cuda_major: int
    cuda_minor: int

    @property
    def wheel_tag(self) -> str:
        """Return the PyTorch wheel CUDA tag for this platform, e.g. ``cu130``."""
        return select_wheel_tag(self.cuda_major, self.cuda_minor)

    def __str__(self) -> str:
        return f"{self.architecture}, CUDA {self.cuda_major}.{self.cuda_minor}"


@dataclass(frozen=True)
class SystemRequirement:
    """A torch dependency that must be met by a system package or library."""
    requirement: str
    apt_package: str
    soname: str | None
    pinned_version: str | None = None


@dataclass
class Resolution:
    """Torch dependencies split into pip, system, and skipped requirements."""
    pip: list[str] = field(default_factory=list)
    system: list[SystemRequirement] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


def normalize(name: str) -> str:
    """Return the PEP 503 normalized form of a distribution name."""
    return re.sub(r"[-_.]+", "-", name).lower()


def upstream_version(debian_version: str) -> str:
    """Return the upstream part of a Debian version, e.g. ``2.29.7`` for ``2.29.7-1+cuda13.2``."""
    version = debian_version.split(":", 1)[-1]
    return version.rsplit("-", 1)[0] if "-" in version else version


def version_at_least(installed: str, required: str) -> bool:
    """Return whether ``installed`` is a valid version no older than ``required``."""
    try:
        return Version(installed) >= Version(required)
    except InvalidVersion:
        return False


def pinned_version(requirement: Requirement) -> str | None:
    """Return the exact version pinned by ``requirement``, or ``None``."""
    specifiers = list(requirement.specifier)
    if len(specifiers) == 1 and specifiers[0].operator in ("==", "==="):
        return specifiers[0].version
    return None


def run(command: Sequence[str], **kwargs) -> subprocess.CompletedProcess[str]:
    """Run ``command`` and capture its text output without raising on failure."""
    return subprocess.run(command, check=False, capture_output=True, text=True, **kwargs)


# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------


def detect_cuda_version(
    environment: dict[str, str] | None = None, root: Path = Path("/")
) -> tuple[int, int]:
    """Return the system CUDA ``(major, minor)`` from ``CUDA_VERSION`` or ``version.json``."""
    environment = os.environ if environment is None else environment
    version = environment.get("CUDA_VERSION", "")
    if not version:
        version_file = root / "usr/local/cuda/version.json"
        if version_file.is_file():
            version = json.loads(version_file.read_text())["cuda"]["version"]
    match = re.match(r"^(\d+)\.(\d+)", version)
    if not match:
        raise InstallError(
            "Could not detect the system CUDA version from CUDA_VERSION or "
            "/usr/local/cuda/version.json."
        )
    return int(match.group(1)), int(match.group(2))


def select_wheel_tag(
    cuda_major: int, cuda_minor: int, wheel_minors: Sequence[int] = WHEEL_CUDA_MINORS
) -> str:
    """Return the newest PyTorch wheel tag that is not newer than the system CUDA.

    Raises:
        InstallError: If the CUDA major version is unsupported or no wheel matches.
    """
    if cuda_major != SUPPORTED_CUDA_MAJOR:
        raise InstallError(
            f"CUDA {cuda_major}.{cuda_minor} is not supported; this helper supports CUDA "
            f"{SUPPORTED_CUDA_MAJOR}.x only."
        )
    candidates = [minor for minor in wheel_minors if minor <= cuda_minor]
    if not candidates:
        raise InstallError(f"No PyTorch wheel is available for CUDA {cuda_major}.{cuda_minor}.")
    return f"cu{cuda_major}{max(candidates)}"


def detect_platform(
    root: Path = Path("/"),
    environment: dict[str, str] | None = None,
    machine: str | None = None,
) -> Platform:
    """Accept x86_64 and SBSA, which includes Tegra iGPU systems on JetPack 7 and later."""
    architecture = machine or platform.machine()
    if architecture not in SUPPORTED_ARCHITECTURES:
        raise InstallError(
            f"Unsupported architecture '{architecture}'; supported: "
            f"{', '.join(SUPPORTED_ARCHITECTURES)}."
        )
    targets = root / "usr/local/cuda/targets"
    if architecture == "aarch64" and targets.is_dir() and not (targets / "sbsa-linux").is_dir():
        raise InstallError(
            "The aarch64 CUDA installation is not an SBSA installation (JetPack 6 and earlier "
            "are not supported)."
        )
    cuda_major, cuda_minor = detect_cuda_version(environment, root)
    select_wheel_tag(cuda_major, cuda_minor)
    return Platform(architecture, cuda_major, cuda_minor)


# ---------------------------------------------------------------------------
# Dependency classification
# ---------------------------------------------------------------------------


def is_cuda_like(name: str) -> bool:
    """Return whether a normalized distribution name looks like a CUDA package."""
    return name.startswith(("nvidia-", "cuda-")) or re.search(r"-cu\d+$", name) is not None


def classify_requirements(
    requires_dist: Iterable[str],
    target: Platform,
    *,
    environment: dict[str, str] | None = None,
) -> Resolution:
    """Split torch's ``Requires-Dist`` entries into pip and system requirements."""
    marker_environment = dict(default_environment() if environment is None else environment)
    marker_environment.setdefault("extra", "")
    versioned = {
        key.format(major=target.cuda_major): component
        for key, component in VERSIONED_COMPONENTS.items()
    }
    resolution = Resolution()
    for line in requires_dist:
        requirement = Requirement(line)
        if requirement.marker is not None and not requirement.marker.evaluate(marker_environment):
            continue
        name = normalize(requirement.name)
        extras = f"[{','.join(sorted(requirement.extras))}]" if requirement.extras else ""
        spec = f"{requirement.name}{extras}{requirement.specifier}"
        if name in SKIPPED_PACKAGES:
            resolution.skipped.append(spec)
        elif name in PIP_CUDA_HELPERS or not is_cuda_like(name):
            resolution.pip.append(spec)
        elif name == "cuda-toolkit":
            resolution.system.extend(_toolkit_requirements(requirement, spec, target))
        elif name in versioned:
            component = versioned[name]
            resolution.system.append(
                SystemRequirement(
                    spec,
                    component.apt_package(target.cuda_major, target.cuda_minor),
                    component.soname,
                    pinned_version(requirement),
                )
            )
        else:
            raise InstallError(
                f"Unknown CUDA dependency '{line}'. Add a system package mapping to this helper "
                "instead of installing CUDA libraries from pip."
            )
    return resolution


def _toolkit_requirements(
    requirement: Requirement, spec: str, target: Platform
) -> list[SystemRequirement]:
    """Map the extras of a ``cuda-toolkit[...]`` requirement to system requirements."""
    version = pinned_version(requirement)
    if version is not None:
        major, minor = Version(version).release[:2]
        if major != target.cuda_major or minor > target.cuda_minor:
            raise InstallError(
                f"The PyTorch wheel requires CUDA {version} but the system has CUDA "
                f"{target.cuda_major}.{target.cuda_minor}; install the "
                f"+{target.wheel_tag} wheel instead."
            )
    unknown = sorted(set(requirement.extras) - CUDA_TOOLKIT_COMPONENTS.keys())
    if unknown:
        raise InstallError(f"Unknown cuda-toolkit components in '{spec}': {', '.join(unknown)}.")
    result = []
    for extra in sorted(requirement.extras):
        component = CUDA_TOOLKIT_COMPONENTS[extra]
        soname = component.soname.format(major=target.cuda_major) if component.soname else None
        result.append(
            SystemRequirement(
                f"cuda-toolkit[{extra}]",
                component.apt_package(target.cuda_major, target.cuda_minor),
                soname,
            )
        )
    return result


# ---------------------------------------------------------------------------
# System libraries
# ---------------------------------------------------------------------------


class SystemProbe:
    """Query installed APT packages and shared libraries."""

    def __init__(self) -> None:
        self._ldconfig: dict[str, str] | None = None

    def dpkg_version(self, package: str) -> str | None:
        """Return the installed version of an APT package, or ``None``."""
        result = run(["dpkg-query", "-W", "-f=${db:Status-Status} ${Version}", package])
        if result.returncode != 0:
            return None
        status, _, version = result.stdout.strip().partition(" ")
        return version if status == "installed" and version else None

    def find_library(self, soname: str) -> str | None:
        """Return the system path of ``soname`` from the ldconfig cache or CUDA paths."""
        if self._ldconfig is None:
            self._ldconfig = {}
            for line in run(["ldconfig", "-p"]).stdout.splitlines():
                match = re.match(r"^\s*(\S+) \(([^)]*)\) => (\S+)$", line)
                if match and match.group(1) not in self._ldconfig:
                    self._ldconfig[match.group(1)] = match.group(3)
        if soname in self._ldconfig:
            return self._ldconfig[soname]
        search_dirs = [*os.environ.get("LD_LIBRARY_PATH", "").split(":"), "/usr/local/cuda/lib64"]
        search_dirs += [str(p) for p in Path("/usr/local/cuda/targets").glob("*/lib")]
        search_dirs.append("/usr/local/cuda/extras/CUPTI/lib64")
        for directory in filter(None, search_dirs):
            candidate = Path(directory) / soname
            if candidate.exists():
                return str(candidate)
        return None


def system_problem(requirement: SystemRequirement, probe: SystemProbe) -> str | None:
    """Describe why ``requirement`` is not met by the system, or return ``None``."""
    installed = probe.dpkg_version(requirement.apt_package)
    if installed is not None:
        pin = requirement.pinned_version
        if pin and not version_at_least(upstream_version(installed), pin):
            return f"{requirement.apt_package} {installed} is older than {pin}"
        return None
    if requirement.soname is not None:
        if probe.find_library(requirement.soname) is not None:
            return None
    elif Path("/usr/local/cuda/include/nvtx3").is_dir():
        return None
    return f"{requirement.apt_package} is not installed ({requirement.requirement})"


def require_system_libraries(requirements: Iterable[SystemRequirement], probe: SystemProbe) -> None:
    """Raise an ``InstallError`` listing APT packages that are missing or too old."""
    problems = {}
    for requirement in requirements:
        problem = system_problem(requirement, probe)
        if problem is not None:
            problems[requirement.apt_package] = problem
    if problems:
        details = "\n".join(f"  - {problem}" for problem in problems.values())
        raise InstallError(
            f"System CUDA libraries required by PyTorch are missing or too old:\n{details}\n"
            f"Install them first, for example: apt-get install {' '.join(problems)}"
        )


# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------


def pip_source_args(args: argparse.Namespace) -> list[str]:
    """Return the pip arguments that select the torch wheel source."""
    if args.wheel_dir:
        return ["--no-index", "--find-links", str(args.wheel_dir)]
    return ["--index-url", args.index_url]


def resolve_torch_wheel(args: argparse.Namespace, target: Platform) -> tuple[str, list[str]]:
    """Return the torch requirement and its ``Requires-Dist`` without installing it."""
    spec = f"torch=={args.version}+{target.wheel_tag}"
    with tempfile.TemporaryDirectory() as tmp:
        report = Path(tmp) / "report.json"
        command = [
            *(sys.executable, "-m", "pip", "install", "--dry-run", "--quiet"),
            *("--no-deps", "--ignore-installed", "--report", str(report)),
            *pip_source_args(args),
            spec,
        ]
        result = run(command)
        if result.returncode != 0:
            raise InstallError(f"Could not find {spec}:\n{result.stderr.strip()}")
        install = json.loads(report.read_text())["install"]
    return spec, install[0]["metadata"].get("requires_dist", [])


def torch_directory() -> Path:
    """Return the installed ``torch`` package directory for this interpreter."""
    result = run(
        [sys.executable, "-c", "import importlib.util as u; print(u.find_spec('torch').origin)"]
    )
    if result.returncode != 0:
        raise InstallError("PyTorch is not installed for this Python interpreter.")
    return Path(result.stdout.strip()).parent


def remove_bundled_libraries(
    torch_lib: Path, find_system_library: Callable[[str], str | None]
) -> list[Path]:
    """Remove bundled CUDA libraries, and bundled NVPL libraries that the system provides."""
    removed = []
    for library in sorted(torch_lib.glob("lib*.so*")):
        if CUDA_LIBRARY_PATTERN.match(library.name):
            reason = "CUDA library provided by the system"
        elif NVPL_LIBRARY_PATTERN.match(library.name):
            system_copy = find_system_library(library.name)
            if system_copy is None or Path(system_copy).parent == torch_lib:
                continue
            reason = f"system copy at {system_copy}"
        else:
            continue
        print(f"-- Removing bundled {library.name} ({reason})")
        library.unlink()
        removed.append(library)
    return removed


def strip_kineto(torch_dir: Path) -> None:
    """Remove kineto references from torch's CMake configs so ``find_package(Torch)`` works."""
    for config in (torch_dir / "share/cmake").rglob("*Config.cmake"):
        lines = config.read_text().splitlines(keepends=True)
        kept = [line for line in lines if "kineto" not in line]
        if kept != lines:
            config.write_text("".join(kept))


def install(args: argparse.Namespace) -> int:
    """Install torch without pip CUDA dependencies, clean it up, then run ``check``.

    Fails before installing anything when a system CUDA library is missing or too old.
    """
    target = detect_platform()
    spec, requires_dist = resolve_torch_wheel(args, target)
    resolution = classify_requirements(requires_dist, target)
    print(f"Installing {spec} on {target}")
    probe = SystemProbe()
    require_system_libraries(resolution.system, probe)
    for requirement in resolution.skipped:
        print(f"-- Skipping {requirement}; install it separately if needed")

    pip = [sys.executable, "-m", "pip", "install"]
    subprocess.run([*pip, "--no-deps", *pip_source_args(args), spec], check=True)
    if resolution.pip:
        subprocess.run([*pip, *resolution.pip], check=True)

    torch_dir = torch_directory()
    remove_bundled_libraries(torch_dir / "lib", probe.find_library)
    strip_kineto(torch_dir)
    return 0 if args.skip_check else check(args)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

CHECK_SCRIPT = r"""
import json, torch
result = {"version": torch.__version__, "cuda": torch.version.cuda, "gpu": None}
if torch.cuda.is_available():
    a = torch.randn(256, 256, device="cuda")
    b = torch.nn.functional.conv2d(
        torch.randn(1, 3, 32, 32, device="cuda"), torch.randn(8, 3, 3, 3, device="cuda")
    )
    torch.cuda.synchronize()
    result["gpu"] = {"device": torch.cuda.get_device_name(), "ok": bool(torch.isfinite(a @ a).all())}
with open("/proc/self/maps") as maps:
    result["maps"] = sorted({line.split()[-1] for line in maps if ".so" in line})
print(json.dumps(result))
"""

CMAKE_PROJECT = """cmake_minimum_required(VERSION 3.24)
project(holoscan_torch_check CXX)
find_package(Torch REQUIRED)
add_executable(torch_check main.cpp)
target_link_libraries(torch_check PRIVATE torch)
"""

CMAKE_MAIN = """#include <torch/version.h>
#include <iostream>
int main() { std::cout << TORCH_VERSION << std::endl; return 0; }
"""


def pip_locations(paths: Iterable[str], site_packages: Path) -> list[str]:
    """Return CUDA or NVPL libraries loaded from pip locations instead of the system."""
    torch_lib = Path(os.path.normpath(site_packages / "torch/lib"))
    problems = []
    for path in paths:
        library = Path(os.path.normpath(path))
        if not (library.is_relative_to(site_packages) or PIP_DIRECTORIES & set(library.parts)):
            continue
        if CUDA_LIBRARY_PATTERN.match(library.name) or (
            NVPL_LIBRARY_PATTERN.match(library.name) and library.parent != torch_lib
        ):
            problems.append(path)
    return problems


def ldd_problems(libraries: Iterable[Path], site_packages: Path) -> list[str]:
    """Return ``ldd`` dependencies of ``libraries`` that are missing or resolve to pip copies."""
    problems = []
    for library in libraries:
        for line in run(["ldd", str(library)]).stdout.splitlines():
            name, _, target = (part.strip() for part in line.partition("=>"))
            if target == "not found" and name not in DRIVER_LIBRARIES:
                problems.append(f"{library.name}: {name} not found")
            elif target and pip_locations([target.split()[0]], site_packages):
                problems.append(f"{library.name}: {name} resolves to {target.split()[0]}")
    return problems


def forbidden_distributions(names: Iterable[str]) -> list[str]:
    """Return the installed pip distributions that ship CUDA libraries."""
    return sorted(n for n in set(map(normalize, names)) if PIP_CUDA_DISTRIBUTION_PATTERN.match(n))


def pip_cuda_library_files(site_packages: Path) -> list[str]:
    """Return CUDA libraries installed under ``site-packages/nvidia``."""
    return sorted(
        str(path)
        for path in (site_packages / "nvidia").rglob("lib*.so*")
        if CUDA_LIBRARY_PATTERN.match(path.name)
    )


def check_cmake(torch_dir: Path) -> list[str]:
    """Build and run a small ``find_package(Torch)`` consumer; return any failures."""
    if shutil.which("cmake") is None:
        return ["cmake is not installed; cannot validate find_package(Torch)"]
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp)
        build = source / "build"
        (source / "CMakeLists.txt").write_text(CMAKE_PROJECT)
        (source / "main.cpp").write_text(CMAKE_MAIN)
        prefix = f"-DCMAKE_PREFIX_PATH={torch_dir / 'share/cmake'}"
        for step, command in (
            ("find_package(Torch)", ["cmake", "-S", str(source), "-B", str(build), prefix]),
            ("Building a Torch consumer", ["cmake", "--build", str(build)]),
            ("Running a Torch consumer", [str(build / "torch_check")]),
        ):
            result = run(command)
            if result.returncode != 0:
                return [f"{step} failed:\n{result.stdout}{result.stderr}"]
    print("CMake find_package(Torch): PASS")
    return []


def check(args: argparse.Namespace) -> int:
    """Validate that the installed torch loads CUDA only from the system; return an exit code."""
    target = detect_platform()
    torch_dir = torch_directory()
    site_packages = torch_dir.parent
    problems: list[str] = []

    version = metadata.version("torch")
    print(f"PyTorch {version} at {torch_dir}")
    expected = f"{args.version}+{target.wheel_tag}" if args.version else None
    if expected and version != expected:
        problems.append(f"PyTorch {version} is installed; expected {expected}")
    elif "+cu" not in version:
        problems.append(f"PyTorch {version} is not a +cu{target.cuda_major}Y build")

    names = [d.metadata["Name"] for d in metadata.distributions()]
    problems += [f"pip CUDA package {name} is installed" for name in forbidden_distributions(names)]
    problems += [f"pip CUDA library {path}" for path in pip_cuda_library_files(site_packages)]
    problems += ldd_problems(sorted(torch_dir.glob("lib/lib*.so*")), site_packages)

    child = run([sys.executable, "-c", CHECK_SCRIPT])
    if child.returncode != 0:
        problems.append(f"import torch failed:\n{child.stderr.strip()}")
    else:
        result = json.loads(child.stdout.strip().splitlines()[-1])
        problems += [f"loaded from pip: {p}" for p in pip_locations(result["maps"], site_packages)]
        if result["gpu"] is not None:
            if not result["gpu"]["ok"]:
                problems.append(f"GPU smoke test failed on {result['gpu']['device']}")
            print(f"GPU smoke test (matmul, cuDNN conv2d) ran on {result['gpu']['device']}")
        elif args.require_gpu:
            problems.append("No GPU available for the GPU smoke test")
        else:
            print("No GPU available; skipped the GPU smoke test")

    if args.cmake:
        problems += check_cmake(torch_dir)

    if problems:
        print("PyTorch check: FAIL", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print("PyTorch check: PASS")
    return 0


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the ``install`` and ``check`` command lines."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)

    install_parser = commands.add_parser("install", help="install PyTorch, then run 'check'")
    install_parser.add_argument("--version", default=DEFAULT_VERSION, help="PyTorch version")
    source = install_parser.add_mutually_exclusive_group()
    source.add_argument("--index-url", default=DEFAULT_INDEX_URL, help="PyTorch wheel index")
    source.add_argument("--wheel-dir", type=Path, help="install the torch wheel from DIR only")
    install_parser.add_argument("--skip-check", action="store_true", help="do not run 'check'")

    check_parser = commands.add_parser("check", help="validate an installed PyTorch")
    check_parser.add_argument("--version", help="fail unless this PyTorch version is installed")

    for sub in (install_parser, check_parser):
        sub.add_argument(
            "--cmake", action="store_true", help="also build a find_package(Torch) consumer"
        )
        sub.add_argument("--require-gpu", action="store_true", help="fail if no GPU is available")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the requested command and return a process exit code."""
    args = parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)
    try:
        return install(args) if args.command == "install" else check(args)
    except InstallError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as error:
        print(f"error: command failed: {' '.join(map(str, error.cmd))}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
