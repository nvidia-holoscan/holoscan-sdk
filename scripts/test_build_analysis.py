#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check CLI failure reporting with a controlled compiler subprocess."""

import json
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).with_name("build_analysis.py")


@pytest.fixture
def build_dir(tmp_path):
    compiler = tmp_path / "compiler.py"
    compiler.write_text(
        """import pathlib, sys
args = sys.argv[1:]
source = pathlib.Path(args[-1]).read_text()
if ('bad' in source or ('stageE' in source and '-E' in args)
        or ('stageM' in source and '-M' in args)):
    print('fixture compiler: rejected measurement', file=sys.stderr)
    sys.exit(1)
print('int valid;')
"""
    )
    entries = []
    for name in ("good", "bad"):
        source = tmp_path / f"{name}.cpp"
        source.write_text(f"// {name}\n")
        entries.append(
            {
                "file": str(source),
                "directory": str(tmp_path),
                "command": shlex.join(
                    [sys.executable, str(compiler), "-c", str(source), "-o", "fixture.o"]
                ),
            }
        )
    (tmp_path / "compile_commands.json").write_text(json.dumps(entries))
    return tmp_path


def invoke(build_dir, command, names, *, compiler=None):
    output = build_dir / "result.json"
    arguments = [sys.executable, str(SCRIPT), "--json", str(output), command, str(build_dir)]
    if command == "header-cost":
        arguments.extend(f"{name}.hpp" for name in names)
        arguments.extend(["--reference-source", "good.cpp", "--repeats", "1"])
    else:
        arguments.extend(f"{name}.cpp" for name in names)
    if command == "ftime-trace":
        arguments.extend(
            ["--clang", compiler or sys.executable, "--out", str(build_dir / "traces")]
        )
    process = subprocess.run(arguments, capture_output=True, text=True, check=False)
    return process, json.loads(output.read_text())


@pytest.mark.parametrize("command", ["header-cost", "ftime-trace", "ftime-report"])
@pytest.mark.parametrize("names", [("good",), ("bad",), ("good", "bad")])
def test_compiler_measurement_status_and_partial_results(build_dir, command, names):
    process, result = invoke(build_dir, command, names)
    rows_key = {"header-cost": "includes", "ftime-trace": "runs", "ftime-report": "units"}[command]
    rows = result[rows_key]
    assert len(rows) == len(names)
    assert process.returncode == int("bad" in names)
    assert [row.get("returncode", 0) for row in rows] == [int(name == "bad") for name in names]
    if "bad" in names:
        assert "bad" in process.stderr
        assert "measurement failed" in process.stderr
        assert "fixture compiler: rejected measurement" in process.stderr
        assert "fixture compiler: rejected measurement" in rows[-1]["error"]
        if command != "ftime-trace":
            assert "error" in process.stdout
    else:
        assert process.stderr == ""


@pytest.mark.parametrize("stage", ["stageE", "stageM"])
def test_header_auxiliary_compiler_failure_is_not_a_success(build_dir, stage):
    process, result = invoke(build_dir, "header-cost", ("good", stage))
    assert process.returncode != 0
    assert len(result["includes"]) == 2
    assert "seconds" in result["includes"][0]
    assert result["includes"][1]["returncode"] == 1
    assert f"{stage}.hpp: measurement failed" in process.stderr


def test_missing_compiler_preserves_partial_results(build_dir):
    process, result = invoke(
        build_dir, "ftime-trace", ("good", "bad"), compiler=str(build_dir / "missing-compiler")
    )
    assert process.returncode != 0
    assert len(result["runs"]) == 2
    assert all(
        row["returncode"] != 0 and "missing-compiler" in row["error"] for row in result["runs"]
    )
    assert "measurement failed" in process.stderr


@pytest.mark.parametrize("command", ["closure", "duplicates", "trace-report", "deps", "ninja-log"])
def test_analysis_only_command_succeeds(build_dir, command):
    (build_dir / "build.ninja").write_text("build fixture.o: CXX_COMPILER good.cpp\n")
    (build_dir / ".ninja_log").write_text("# ninja log v5\n0\t1\t0\tfixture.o\t0\n")
    deps = build_dir / "deps.txt"
    deps.write_text("")
    if command == "closure":
        arguments = [command, "good.hpp", "--include-root", str(build_dir)]
        (build_dir / "good.hpp").write_text("#pragma once\n")
    elif command == "deps":
        arguments = [command, str(build_dir), "--deps-file", str(deps)]
    elif command == "trace-report":
        traces = build_dir / "traces"
        traces.mkdir()
        arguments = [command, str(traces)]
    else:
        arguments = [command, str(build_dir)]
    process = subprocess.run(
        [sys.executable, str(SCRIPT), *arguments], capture_output=True, text=True, check=False
    )
    assert process.returncode == 0, process.stderr
    assert process.stderr == ""
