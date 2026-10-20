#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run Google Benchmark executables with optional arguments from the environment."""

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def benchmark_alias(test_name: str) -> str:
    return test_name.removeprefix("BENCHMARK_").lower()


def test_has_benchmark_label(test: dict) -> bool:
    for prop in test.get("properties", []):
        if prop.get("name") == "LABELS" and "benchmark" in prop.get("value", []):
            return True
    return False


def test_working_directory(test: dict) -> str | None:
    for prop in test.get("properties", []):
        if prop.get("name") == "WORKING_DIRECTORY":
            return prop.get("value")
    return None


def test_matches(test_name: str, regex: str) -> bool:
    if not regex:
        return True
    alias = benchmark_alias(test_name)
    return re.search(regex, test_name) is not None or re.search(regex, alias) is not None


def ctest_json() -> dict:
    result = subprocess.run(
        ["ctest", ".", "-L", "benchmark", "--show-only=json-v1"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr, end="")
        raise SystemExit(result.returncode)
    return json.loads(result.stdout)


def print_benchmark_list(tests: list[dict]) -> int:
    print("Available benchmark aliases")
    if not tests:
        print("  No benchmark aliases found.")
        return 1
    for test in tests:
        name = test["name"]
        alias = benchmark_alias(name)
        print(f"  {alias} ({name})")
        print(f"    Run:  ./run bench --name {alias}")
        print(f"    JSON: ./run bench --name {alias} --output-json benchmark-{alias}.json")
    return 0


def run_ctest_benchmarks(regex: str, list_only: bool) -> int:
    tests = [
        test
        for test in ctest_json().get("tests", [])
        if test_has_benchmark_label(test) and test_matches(test["name"], regex)
    ]
    if list_only:
        return print_benchmark_list(tests)
    if not tests:
        print("No benchmark tests matched the requested filter.", file=sys.stderr)
        return 1

    exit_code = 0
    for test in tests:
        name = test["name"]
        print(f"\n== {benchmark_alias(name)} ({name}) ==", flush=True)
        result = subprocess.run(test["command"], check=False, cwd=test_working_directory(test))
        if result.returncode != 0 and exit_code == 0:
            exit_code = result.returncode
    return exit_code


def argument_value(arguments: list[str], option: str) -> str | None:
    """Return the final value supplied for a Google Benchmark option."""
    value = None
    for index, argument in enumerate(arguments):
        if argument.startswith(f"{option}="):
            value = argument.removeprefix(f"{option}=")
        elif argument == option and index + 1 < len(arguments):
            value = arguments[index + 1]
    return value


def benchmark_errors(output_path: Path) -> list[str] | None:
    """Return failed benchmark names, or None when the JSON report is invalid."""
    try:
        report = json.loads(output_path.read_text(encoding="utf-8"))
        benchmarks = report["benchmarks"]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError):
        return None

    if not isinstance(benchmarks, list):
        return None
    return [
        str(benchmark.get("name", "<unnamed benchmark>"))
        for benchmark in benchmarks
        if isinstance(benchmark, dict) and benchmark.get("error_occurred")
    ]


def run_benchmark(command: list[str], output_path: Path, caller_provided_output: bool) -> int:
    """Run a benchmark and validate its newly produced JSON report."""
    backup_directory = None
    backup_path = None
    if caller_provided_output and output_path.is_file():
        backup_directory = tempfile.TemporaryDirectory(
            prefix=".holoscan-benchmark-", dir=output_path.parent
        )
        backup_path = Path(backup_directory.name) / output_path.name
        output_path.replace(backup_path)

    def restore_backup() -> None:
        if backup_path is not None:
            backup_path.replace(output_path)

    try:
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            restore_backup()
            return result.returncode

        errors = benchmark_errors(output_path)
        if errors is None:
            restore_backup()
            print(
                f"error: benchmark did not produce a valid JSON report: {output_path}",
                file=sys.stderr,
            )
            return 1
        if errors:
            print(
                "error: Google Benchmark reported failed case(s): " + ", ".join(errors),
                file=sys.stderr,
            )
            return 1
        return 0
    finally:
        if backup_directory is not None:
            backup_directory.cleanup()


def run_single_benchmark(argv: list[str]) -> int:
    if not argv:
        print("usage: run_benchmark.py <benchmark_executable> [default_args...]", file=sys.stderr)
        return 2

    executable = argv[0]
    args = argv[1:]
    extra_args = shlex.split(os.environ.get("HOLOSCAN_BENCHMARK_ARGS", ""))
    command = [executable, *args, *extra_args]

    output_argument = argument_value(command[1:], "--benchmark_out")
    if output_argument is None:
        output_argument = os.environ.get("BENCHMARK_OUT") or None

    output_format = argument_value(command[1:], "--benchmark_out_format")
    if output_format is None:
        output_format = os.environ.get("BENCHMARK_OUT_FORMAT") or None
    if output_format not in (None, "json"):
        print(
            "error: benchmark runner requires --benchmark_out_format=json to validate results",
            file=sys.stderr,
        )
        return 2

    temporary_directory = None
    caller_provided_output = output_argument is not None
    if output_argument is None:
        temporary_directory = tempfile.TemporaryDirectory(prefix="holoscan-benchmark-")
        output_path = Path(temporary_directory.name) / "benchmark.json"
        command.extend((f"--benchmark_out={output_path}", "--benchmark_out_format=json"))
    else:
        output_path = Path(output_argument)
        if output_format is None:
            command.append("--benchmark_out_format=json")

    if os.environ.get("HOLOSCAN_BENCHMARK_DISABLE_ASLR") == "1":
        arch = subprocess.run(["uname", "-m"], check=False, stdout=subprocess.PIPE, text=True)
        if arch.returncode == 0 and shutil.which("setarch"):
            command = ["setarch", arch.stdout.strip(), "-R", *command]
        else:
            print("warning: setarch is unavailable; ASLR may remain enabled", file=sys.stderr)
    try:
        return run_benchmark(command, output_path, caller_provided_output)
    finally:
        if temporary_directory is not None:
            temporary_directory.cleanup()


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--run-ctest-benchmarks", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--regex", default=os.environ.get("HOLOSCAN_BENCHMARK_REGEX", ""))
    args, remaining = parser.parse_known_args()

    if args.run_ctest_benchmarks:
        return run_ctest_benchmarks(args.regex, args.list)
    return run_single_benchmark(remaining)


if __name__ == "__main__":
    raise SystemExit(main())
