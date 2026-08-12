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


def run_single_benchmark(argv: list[str]) -> int:
    if not argv:
        print("usage: run_benchmark.py <benchmark_executable> [default_args...]", file=sys.stderr)
        return 2

    executable = argv[0]
    args = argv[1:]
    extra_args = shlex.split(os.environ.get("HOLOSCAN_BENCHMARK_ARGS", ""))
    command = [executable, *args, *extra_args]
    if os.environ.get("HOLOSCAN_BENCHMARK_DISABLE_ASLR") == "1":
        arch = subprocess.run(["uname", "-m"], check=False, stdout=subprocess.PIPE, text=True)
        if arch.returncode == 0 and shutil.which("setarch"):
            command = ["setarch", arch.stdout.strip(), "-R", *command]
        else:
            print("warning: setarch is unavailable; ASLR may remain enabled", file=sys.stderr)
    return subprocess.run(command, check=False).returncode


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
