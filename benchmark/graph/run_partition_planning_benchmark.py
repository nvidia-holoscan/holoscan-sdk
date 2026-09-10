#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run isolated cold partition-planning probes and retain raw evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

SCHEMA = "holoscan.ea2.partition-planning-benchmark/v1"
SUMMARY_SCHEMA = "holoscan.ea2.partition-planning-summary/v1"
TOPOLOGIES = (
    "linear",
    "fanout",
    "binary_fanout",
    "fanin",
    "diamond",
    "multi_root",
    "embedded",
    "hybrid",
)
MAX_OPERATOR_COUNT = (1 << 32) - 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark_binary", type=Path)
    parser.add_argument("--topology", choices=TOPOLOGIES, action="append", required=True)
    parser.add_argument("--operators", type=int, action="append", required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--process-timeout-sec", type=float, default=120.0)
    parser.add_argument("--git-revision", default=os.environ.get("HOLOSCAN_GIT_REVISION"))
    parser.add_argument("--source-state", default="unknown")
    parser.add_argument("--source-snapshot-sha256", default=None)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if any(size < 0 or size > MAX_OPERATOR_COUNT for size in args.operators):
        raise ValueError("--operators must be within the uint32 admission domain")
    if "hybrid" in args.topology and any(0 < size < 4 for size in args.operators):
        raise ValueError("hybrid topology requires zero or at least four operators")
    if args.repetitions <= 0 or args.process_timeout_sec <= 0:
        raise ValueError("repetitions and process timeout must be positive")
    if args.source_snapshot_sha256 is not None and not valid_sha256(args.source_snapshot_sha256):
        raise ValueError("--source-snapshot-sha256 must be a lowercase SHA-256 digest")


def percentile(values: list[int], quantile: float) -> int:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(len(ordered) * quantile + 0.999999) - 1))
    return ordered[index]


def summarize(values: list[int]) -> dict[str, int | float]:
    return {
        "minimum": min(values),
        "median": statistics.median(values),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "maximum": max(values),
    }


def build_metadata(binary: Path) -> dict[str, object]:
    requested = {
        "CMAKE_BUILD_TYPE",
        "CMAKE_CXX_COMPILER",
        "CMAKE_CXX_COMPILER_ID",
        "CMAKE_CXX_COMPILER_VERSION",
        "HOLOSCAN_ENABLE_CLANG_TIDY",
    }
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
        if line.startswith("//") or line.startswith("#") or "=" not in line or ":" not in line:
            continue
        key_and_type, value = line.split("=", 1)
        key = key_and_type.split(":", 1)[0]
        if key in requested:
            result[key] = value
    compiler = result.get("CMAKE_CXX_COMPILER")
    if isinstance(compiler, str):
        try:
            completed = subprocess.run(
                [compiler, "--version"],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def checked_result(path: Path, topology: str, operators: int) -> dict[str, object]:
    result = json.loads(path.read_text(encoding="utf-8"))
    expected_mode = (
        "constrained_automatic" if topology == "hybrid" and operators != 0 else "automatic"
    )
    numeric = {
        "compile_ns",
        "allocation_count",
        "cumulative_requested_bytes",
        "peak_live_requested_bytes",
        "live_requested_bytes_after_compile",
        "live_requested_bytes_after_plan_release",
        "process_max_rss_kib",
        "edge_count",
        "partition_count",
        "authored_partition_count",
        "automatic_partition_count",
        "authored_member_count",
        "automatic_member_count",
    }
    if (
        result.get("schema") != SCHEMA
        or result.get("topology") != topology
        or result.get("operator_count") != operators
        or result.get("warmup_compile") != "accepted_zero_work"
        or result.get("partitioning_mode") != expected_mode
        or any(not isinstance(result.get(key), int) or result[key] < 0 for key in numeric)
        or not isinstance(result.get("plan_digest"), str)
        or len(result["plan_digest"]) != 64
    ):
        raise RuntimeError(f"invalid partition-planning result: {path}")
    return result


def run_case(
    binary: Path, topology: str, operators: int, repetition: int, output_dir: Path, timeout: float
) -> dict[str, object]:
    case_dir = output_dir / f"{topology}_{operators}" / f"repetition_{repetition:03d}"
    case_dir.mkdir(parents=True, exist_ok=True)
    result_path = case_dir / "result.json"
    command = [
        str(binary),
        "--topology",
        topology,
        "--operators",
        str(operators),
        "--output",
        str(result_path),
    ]
    (case_dir / "command.json").write_text(json.dumps(command, indent=2) + "\n", encoding="utf-8")
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout, check=False
        )
    except subprocess.TimeoutExpired as error:
        stdout = (
            error.stdout.decode(errors="replace")
            if isinstance(error.stdout, bytes)
            else error.stdout
        )
        stderr = (
            error.stderr.decode(errors="replace")
            if isinstance(error.stderr, bytes)
            else error.stderr
        )
        (case_dir / "stdout.log").write_text(stdout or "", encoding="utf-8")
        (case_dir / "stderr.log").write_text(stderr or "", encoding="utf-8")
        raise RuntimeError(f"partition-planning probe timed out: {case_dir}") from error
    (case_dir / "stdout.log").write_text(completed.stdout, encoding="utf-8")
    (case_dir / "stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"partition-planning probe exited {completed.returncode}: {case_dir}")
    return checked_result(result_path, topology, operators)


def main() -> int:
    args = parse_args()
    try:
        validate_args(args)
        binary = args.benchmark_binary.resolve(strict=True)
        if args.output_dir is None:
            output_dir = Path(tempfile.mkdtemp(prefix="holoscan-partition-planning-"))
        else:
            output_dir = args.output_dir.resolve()
            output_dir.mkdir(parents=True, exist_ok=True)
        cases = []
        for topology in args.topology:
            for operators in args.operators:
                results = [
                    run_case(
                        binary,
                        topology,
                        operators,
                        repetition,
                        output_dir,
                        args.process_timeout_sec,
                    )
                    for repetition in range(1, args.repetitions + 1)
                ]
                digests = {str(result["plan_digest"]) for result in results}
                covers = {
                    (
                        result["partitioning_mode"],
                        result["partition_count"],
                        result["authored_partition_count"],
                        result["automatic_partition_count"],
                        result["edge_count"],
                        result["authored_member_count"],
                        result["automatic_member_count"],
                    )
                    for result in results
                }
                cases.append(
                    {
                        "topology": topology,
                        "operators": operators,
                        "repetitions": args.repetitions,
                        "deterministic_plan": len(digests) == 1 and len(covers) == 1,
                        **{
                            key: summarize([int(result[key]) for result in results])
                            for key in (
                                "compile_ns",
                                "allocation_count",
                                "cumulative_requested_bytes",
                                "peak_live_requested_bytes",
                                "live_requested_bytes_after_compile",
                                "live_requested_bytes_after_plan_release",
                                "process_max_rss_kib",
                            )
                        },
                        "plan_digest": next(iter(digests)) if len(digests) == 1 else None,
                        "partitioning_mode": results[0]["partitioning_mode"],
                        "partition_count": results[0]["partition_count"],
                        "authored_partition_count": results[0]["authored_partition_count"],
                        "automatic_partition_count": results[0]["automatic_partition_count"],
                        "edge_count": results[0]["edge_count"],
                        "authored_member_count": results[0]["authored_member_count"],
                        "automatic_member_count": results[0]["automatic_member_count"],
                    }
                )
        summary = {
            "schema": SUMMARY_SCHEMA,
            "qualification": {
                "status": "characterization_only",
                "issues": [
                    "practical_size_points_are_not_the_uint32_admission_ceiling",
                    "process_max_rss_is_whole_process_supporting_evidence",
                ],
            },
            "host": {
                "hostname": platform.node(),
                "platform": platform.platform(),
                "processor": platform.processor(),
                "affinity": sorted(os.sched_getaffinity(0))
                if hasattr(os, "sched_getaffinity")
                else None,
            },
            "binary": str(binary),
            "binary_sha256": file_sha256(binary),
            "git_revision": args.git_revision,
            "source_state": args.source_state,
            "source_snapshot_sha256": args.source_snapshot_sha256,
            "build": build_metadata(binary),
            "cases": cases,
        }
        summary_path = output_dir / "summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(summary_path)
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
