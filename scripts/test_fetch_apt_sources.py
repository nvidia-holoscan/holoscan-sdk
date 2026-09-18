#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_CANDIDATES = (
    REPOSITORY_ROOT / "public/dev-container/scripts/fetch-apt-sources.sh",
    REPOSITORY_ROOT / "docker/release/scripts/fetch-apt-sources.sh",
)


def source_script() -> Path:
    """Return the OSRB APT helper at its current repository location."""
    return next(path for path in SCRIPT_CANDIDATES if path.is_file())


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
