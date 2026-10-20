# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).with_name("development-tools.sh")
INVENTORY_COMMANDS = {
    "cc",
    "c++",
    "make",
    "cmake",
    "ctest",
    "cpack",
    "ninja",
    "pkg-config",
    "patchelf",
    "python3",
    "git",
    "curl",
    "rsync",
    "unzip",
    "gpg",
    "gdb",
    "gdbserver",
    "valgrind",
    "nsys",
    "nvcc",
    "cuda-gdb",
}


def run_tool(path: Path, mode: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PATH"] = str(path)
    return subprocess.run(
        ["/bin/bash", str(SCRIPT), mode],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )


def write_command(path: Path, name: str, body: str = "exit 0") -> None:
    command = path / name
    command.write_text(f"#!/bin/bash\n{body}\n", encoding="utf-8")
    command.chmod(0o755)


def create_tool_path(path: Path, *, cmake_major: int = 4, omit: str = "") -> None:
    path.mkdir(exist_ok=True)
    for command in INVENTORY_COMMANDS - {omit, "cmake", "python3"}:
        write_command(path, command)

    if omit != "cmake":
        write_command(path, "cmake", f"echo 'cmake version {cmake_major}.0.0'")

    if omit != "python3":
        write_command(
            path,
            "python3",
            """if [[ \"${1:-}\" == \"-m\" && \"${2:-}\" == \"venv\" ]]; then
    mkdir -p \"$3/bin\"
    printf '#!/bin/bash\\nexit 0\\n' > \"$3/bin/python\"
    chmod +x \"$3/bin/python\"
fi
exit 0""",
        )

    for utility in ("chmod", "mkdir", "mktemp", "rm"):
        if not (path / utility).exists():
            (path / utility).symlink_to(f"/usr/bin/{utility}")


def test_report_lists_the_complete_inventory_without_failing(tmp_path: Path) -> None:
    result = run_tool(tmp_path, "report")

    assert result.returncode == 0
    reported_commands = {line.split()[0] for line in result.stdout.splitlines()[1:]}
    assert reported_commands == INVENTORY_COMMANDS
    assert "missing" in result.stdout


def test_check_accepts_complete_core_capabilities(tmp_path: Path) -> None:
    create_tool_path(tmp_path)

    result = run_tool(tmp_path, "check")

    assert result.returncode == 0, result.stderr
    assert "capabilities: PASS" in result.stdout


def test_check_does_not_gate_on_diagnostic_inventory_tool(tmp_path: Path) -> None:
    create_tool_path(tmp_path, omit="gdbserver")

    result = run_tool(tmp_path, "check")

    assert result.returncode == 0, result.stderr


def test_check_reports_missing_core_capability(tmp_path: Path) -> None:
    create_tool_path(tmp_path, omit="ninja")

    result = run_tool(tmp_path, "check")

    assert result.returncode == 1
    assert "Missing required development capability: ninja" in result.stderr


def test_check_rejects_wrong_cmake_major(tmp_path: Path) -> None:
    create_tool_path(tmp_path, cmake_major=3)

    result = run_tool(tmp_path, "check")

    assert result.returncode == 1
    assert "CMake 4.x is required" in result.stderr
