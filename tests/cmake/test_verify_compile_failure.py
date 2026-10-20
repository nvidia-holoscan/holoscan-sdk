# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contracts for the negative-compiler runner, including false-pass prevention."""

import subprocess
from pathlib import Path

import pytest
from verify_compile_failure import isolated_command, verify_result


def test_preserves_quoted_flags_and_redirects_outputs():
    entry = {
        "directory": "/sdk build",
        "output": "tests/CMakeFiles/probe.dir/source.cpp.o",
        "command": "c++ -O3 -DNDEBUG '-DVALUE=two words' -std=c++23 "
        "-isystem '/sdk headers' -MD -MF shared.d -o shared.o -c '/src/probe.cpp'",
    }
    command, directory = isolated_command([entry], "probe", Path("/private output"))
    assert directory == "/sdk build"
    assert command == [
        "c++",
        "-O3",
        "-DNDEBUG",
        "-DVALUE=two words",
        "-std=c++23",
        "-isystem",
        "/sdk headers",
        "-MD",
        "-MF",
        "/private output/probe.d",
        "-o",
        "/private output/probe.o",
        "-c",
        "/src/probe.cpp",
    ]


def test_argument_array_and_exact_target():
    entries = [
        {
            "output": f"CMakeFiles/{target}.dir/probe.cpp.o",
            "directory": "/sdk",
            "arguments": ["c++", "-o", "shared.o", "-c", f"{target}.cpp"],
        }
        for target in ("probe", "probe_extra")
    ]
    command, _ = isolated_command(entries, "probe", Path("/private"))
    assert command[-1] == "probe.cpp"
    with pytest.raises(ValueError, match="found 0"):
        isolated_command(entries, "missing", Path("/private"))
    with pytest.raises(ValueError, match="found 2"):
        isolated_command(entries + [entries[0]], "probe", Path("/private"))


@pytest.mark.parametrize("command", [["c++", "@response"], ["c++", "-o", "binary", "probe.o"]])
def test_rejects_unsafe_or_unrecognized_commands(command):
    with pytest.raises(ValueError, match="response file|compile-only command"):
        isolated_command(
            [{"output": "CMakeFiles/probe.dir/probe.o", "directory": "/sdk", "arguments": command}],
            "probe",
            Path("/private"),
        )


def test_accepts_expected_rejection():
    verify_result(subprocess.CompletedProcess([], 1, "static assertion: expected"), "expected")
    verify_result(
        subprocess.CompletedProcess([], 1, "some error"), "HOLOSCAN_ACCEPT_ANY_DIAGNOSTIC"
    )


@pytest.mark.parametrize(
    "status,output,diagnostic,message",
    [
        (0, "expected", "expected", "unexpectedly compiled successfully"),
        (1, "unrelated error", "expected", "without the expected diagnostic"),
        (-9, "expected", "expected", "terminated by signal"),
        (-9, "", "HOLOSCAN_ACCEPT_ANY_DIAGNOSTIC", "terminated by signal"),
    ],
)
def test_requires_expected_rejection(status, output, diagnostic, message):
    with pytest.raises(ValueError, match=message):
        verify_result(subprocess.CompletedProcess([], status, output), diagnostic)
