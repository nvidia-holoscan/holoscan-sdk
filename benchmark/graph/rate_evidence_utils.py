#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private low-level evidence helpers shared by the rate benchmark runners."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

DEFAULT_COMPILER_VERSION_TIMEOUT_SEC = 5.0
_CMAKE_METADATA_KEYS = {
    "CMAKE_BUILD_TYPE",
    "CMAKE_CXX_COMPILER",
    "CMAKE_CXX_COMPILER_ID",
    "CMAKE_CXX_COMPILER_VERSION",
    "HOLOSCAN_ENABLE_CLANG_TIDY",
}


def valid_sha256(value: object) -> bool:
    """Return whether *value* is a canonical lowercase SHA-256 digest."""

    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def file_sha256(path: Path) -> str:
    """Hash one evidence input or output without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file_sha256(path: Path, expected_sha256: str, context: str) -> None:
    """Reject evidence collection if a frozen file's bytes have changed."""

    observed_sha256 = file_sha256(path)
    if observed_sha256 != expected_sha256:
        raise RuntimeError(
            f"{context}: expected {expected_sha256}, observed {observed_sha256}: {path}"
        )


def nearest_rank_percentile(values: list[int | float], quantile: float) -> int | float:
    """Return the bounded nearest-rank percentile used by both runners."""

    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(len(ordered) * quantile + 0.999999) - 1))
    return ordered[index]


def cmake_build_metadata(
    binary: Path,
    *,
    include_compiler_version_output: bool,
    compiler_version_timeout_sec: float = DEFAULT_COMPILER_VERSION_TIMEOUT_SEC,
) -> dict[str, object]:
    """Read stable CMake metadata and optionally probe a bounded compiler version."""

    cache = next(
        (
            directory / "CMakeCache.txt"
            for directory in binary.parents
            if (directory / "CMakeCache.txt").is_file()
        ),
        None,
    )
    if cache is None:
        return {"cmake_cache": None}
    result: dict[str, object] = {"cmake_cache": str(cache)}
    for line in cache.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(("//", "#")) or "=" not in line or ":" not in line:
            continue
        key_and_type, value = line.split("=", 1)
        key = key_and_type.split(":", 1)[0]
        if key in _CMAKE_METADATA_KEYS:
            result[key] = value

    compiler = result.get("CMAKE_CXX_COMPILER")
    if include_compiler_version_output and isinstance(compiler, str):
        try:
            completed = subprocess.run(
                [compiler, "--version"],
                capture_output=True,
                text=True,
                check=False,
                timeout=compiler_version_timeout_sec,
            )
        except (OSError, subprocess.TimeoutExpired):
            result["compiler_version_output"] = None
        else:
            result["compiler_version_output"] = (
                completed.stdout.splitlines()[0]
                if completed.returncode == 0 and completed.stdout
                else None
            )
    return result


def _captured_text(value: str | bytes | None) -> str:
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value or ""


def run_logged_child(
    command: list[str], case_dir: Path, timeout: float
) -> subprocess.CompletedProcess[str]:
    """Run one bounded child and replace its command/stdout/stderr evidence."""

    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "command.json").write_text(json.dumps(command, indent=2) + "\n", encoding="utf-8")
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        (case_dir / "stdout.log").write_text(_captured_text(error.stdout), encoding="utf-8")
        (case_dir / "stderr.log").write_text(_captured_text(error.stderr), encoding="utf-8")
        raise
    (case_dir / "stdout.log").write_text(completed.stdout, encoding="utf-8")
    (case_dir / "stderr.log").write_text(completed.stderr, encoding="utf-8")
    return completed
