#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import re
import subprocess
from pathlib import Path

SDK_ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = SDK_ROOT / "Dockerfile.dev"
POLICY_EXCLUSIONS = SDK_ROOT / "dev-container/scripts/source-apt-exclusions.txt"
SOURCE_SCRIPT = SDK_ROOT / "dev-container/scripts/fetch-apt-sources.sh"


def source_script() -> Path:
    """Return the APT source archive helper."""
    return SOURCE_SCRIPT


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
    assert "${SOURCES_PATH}/scripts/source-apt-exclusions.txt" in dockerfile


def test_nvshmem_source_is_not_embedded() -> None:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    exclusions = {
        line.strip()
        for line in POLICY_EXCLUSIONS.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    assert "NVSHMEM_SOURCE_COMMIT" not in dockerfile
    assert "github.com/NVIDIA/nvshmem/archive/" not in dockerfile
    assert not any(name.startswith("libnvshmem3") for name in exclusions)


def test_removed_nvpl_packages_are_not_excluded_from_source_fetching() -> None:
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

    assert 'CONDITIONAL_LIBS="nvpl-blas nvpl-lapack"' not in dockerfile
    assert nvpl_packages.isdisjoint(exclusions)
