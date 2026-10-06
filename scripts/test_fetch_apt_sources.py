#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import re
import subprocess
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = REPOSITORY_ROOT / "public/Dockerfile.dev"
POLICY_EXCLUSIONS = REPOSITORY_ROOT / "public/dev-container/scripts/osrb-apt-exclusions.txt"
NVSHMEM_PROVENANCE = REPOSITORY_ROOT / "public/dev-container/scripts/nvshmem-source-provenance.md"
SCRIPT_CANDIDATES = (
    REPOSITORY_ROOT / "public/dev-container/scripts/fetch-apt-sources.sh",
    REPOSITORY_ROOT / "docker/release/scripts/fetch-apt-sources.sh",
)


def source_script() -> Path:
    """Return the OSRB APT helper at its current repository location."""
    return next(path for path in SCRIPT_CANDIDATES if path.is_file())


def docker_arg(dockerfile: str, name: str) -> str:
    """Return the default value of a global Docker build argument."""
    match = re.search(rf"^ARG {re.escape(name)}=(\S+)$", dockerfile, re.MULTILINE)
    assert match is not None
    return match.group(1)


def write_command(path: Path, name: str, body: str) -> None:
    """Create an executable command stub."""
    command = path / name
    command.write_text(f"#!/usr/bin/env bash\n{body}\n", encoding="utf-8")
    command.chmod(0o755)


def run_script(tmp_path: Path, apt_get_body: str) -> subprocess.CompletedProcess[str]:
    """Run the helper with deterministic package-manager command stubs."""
    command_path = tmp_path / "bin"
    command_path.mkdir()
    write_command(command_path, "dpkg-query", "printf 'example==1.0\\n'")
    write_command(command_path, "apt-get", apt_get_body)

    sources_path = tmp_path / "sources"
    temporary_path = tmp_path / "temporary"
    temporary_path.mkdir()
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{command_path}:{environment['PATH']}",
            "SOURCES_PATH": str(sources_path),
            "TMPDIR": str(temporary_path),
        }
    )
    return subprocess.run(
        ["/usr/bin/env", "bash", str(source_script())],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )


def test_uses_and_removes_a_private_temporary_directory(tmp_path: Path) -> None:
    result = run_script(tmp_path, "exit 0")

    assert result.returncode == 0, result.stderr
    assert list((tmp_path / "temporary").iterdir()) == []


def test_uses_unversioned_source_when_versioned_lookup_fails(tmp_path: Path) -> None:
    result = run_script(tmp_path, 'if [[ "$2" == "example=1.0" ]]; then exit 1; fi')

    assert result.returncode == 0, result.stderr


def test_fails_when_both_apt_source_lookups_fail(tmp_path: Path) -> None:
    result = run_script(tmp_path, "exit 1")

    assert result.returncode == 1
    assert "Failed to fetch APT sources for: example==1.0" in result.stderr


def test_nsight_systems_is_excluded_from_apt_source_fetching() -> None:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    exclusions = {
        line.strip()
        for line in POLICY_EXCLUSIONS.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    nsys_version = re.search(r"^ARG NSYS_VERSION=(\S+)$", dockerfile, re.MULTILINE)

    assert nsys_version is not None
    assert f"nsight-systems-cli-{nsys_version.group(1)}" in exclusions
    assert "${SOURCES_PATH}/scripts/osrb-apt-exclusions.txt" in dockerfile


def test_nvshmem_packages_and_commit_pinned_source_stay_in_sync() -> None:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    exclusions = {
        line.strip()
        for line in POLICY_EXCLUSIONS.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    provenance = NVSHMEM_PROVENANCE.read_text(encoding="utf-8")

    cuda_major = docker_arg(dockerfile, "CUDA_MAJOR")
    package_version = docker_arg(dockerfile, "NVSHMEM_DEBIAN_VERSION")
    source_commit = docker_arg(dockerfile, "NVSHMEM_SOURCE_COMMIT")
    source_sha256 = docker_arg(dockerfile, "NVSHMEM_SOURCE_SHA256")
    package_stems = {"libnvshmem3-cuda", "libnvshmem3-dev-cuda"}

    assert package_version == "3.7.2-1"
    assert source_commit == "3f4c6d81225f45f9c42ed9af09bdb3c61eb356d4"
    assert source_sha256 == "16ffdc0dad774e1048e0ff648464262d95d6584ace8528da320b72db696b3ba2"
    for package_stem in package_stems:
        package = f"{package_stem}-{cuda_major}"
        assert f"{package_stem}-${{CUDA_MAJOR}}=${{NVSHMEM_DEBIAN_VERSION}}" in dockerfile
        assert package in exclusions
        assert package in provenance

    osrb_stage = dockerfile.split("FROM deps-common AS osrb-sources", maxsplit=1)[1]
    assert "ARG NVSHMEM_SOURCE_COMMIT" in osrb_stage
    assert "ARG NVSHMEM_SOURCE_SHA256" in osrb_stage
    assert dockerfile.count("github.com/NVIDIA/nvshmem/archive/") == 1
    assert "${NVSHMEM_SOURCE_COMMIT}.tar.gz" in osrb_stage
    assert "${NVSHMEM_SOURCE_SHA256}  ${NVSHMEM_SOURCE_ARCHIVE}" in osrb_stage
    assert "sha256sum --check --strict" in osrb_stage
    assert "grep -q '/License.txt$'" in osrb_stage
    assert "grep -q '/NOTICE$'" in osrb_stage
    assert "${SOURCES_PATH}/nvshmem" in osrb_stage

    assert "v3.7.2-0" in provenance
    assert package_version in provenance
    assert source_commit in provenance
    assert source_sha256 in provenance


def test_proprietary_nvpl_packages_are_excluded_from_source_fetching() -> None:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    exclusions = {
        line.strip()
        for line in POLICY_EXCLUSIONS.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    nvpl_packages = {
        "libnvpl-blas-dev",
        "libnvpl-blas0",
        "libnvpl-lapack-dev",
        "libnvpl-lapack0",
        "nvpl-blas",
        "nvpl-lapack",
    }

    assert 'CONDITIONAL_LIBS="nvpl-blas nvpl-lapack"' in dockerfile
    assert nvpl_packages <= exclusions
