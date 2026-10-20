#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the matched CUDA Green Context phase-1 qualification matrix."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import platform
import secrets
import shutil
import stat
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA = "holoscan.ea2.green-context-qualification/v1"
ATTEMPT_SCHEMA = "holoscan.ea2.green-context-qualification-attempt/v1"
REALTIME_PROFILE_DISPOSITION = (
    "admitted: preallocated hot-path contract; terminal stream quiescence remains potentially "
    "unbounded"
)
SKIPPED = 77


@dataclass(frozen=True)
class MatrixCell:
    name: str
    layout: str
    operator_lanes: str
    baseline: str | None


MATRIX = (
    MatrixCell("baseline_shared", "none", "shared", None),
    MatrixCell("same_partition", "same_partition", "shared", "baseline_shared"),
    MatrixCell("baseline_separate", "none", "separate", None),
    MatrixCell(
        "dedicated_to_dedicated",
        "dedicated_to_dedicated",
        "separate",
        "baseline_separate",
    ),
    MatrixCell(
        "dedicated_to_residual",
        "dedicated_to_residual",
        "separate",
        "baseline_separate",
    ),
)


@dataclass
class QualificationAttempt:
    """Own one fresh qualification output directory without replacing prior evidence."""

    output_dir: Path
    revision: str
    directory_fd: int
    directory_identity: tuple[int, int]
    attempt_schema: str = ATTEMPT_SCHEMA

    @classmethod
    def begin(
        cls, output_dir: Path, revision: str, attempt_schema: str = ATTEMPT_SCHEMA
    ) -> QualificationAttempt:
        if output_dir == output_dir.parent:
            raise ValueError("--output-dir must not name a filesystem root")
        if output_dir.is_symlink() or any(parent.is_symlink() for parent in output_dir.parents):
            raise ValueError("--output-dir must not be a symbolic link")
        if output_dir.exists():
            if not output_dir.is_dir():
                raise ValueError("--output-dir must name a directory")
            if any(output_dir.iterdir()):
                raise ValueError("--output-dir must be absent or empty")
        else:
            output_dir.mkdir(parents=True)

        directory_fd = os.open(
            output_dir,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
        try:
            claimed = os.fstat(directory_fd)
            visible = os.stat(output_dir, follow_symlinks=False)
            if not stat.S_ISDIR(visible.st_mode) or (
                visible.st_dev,
                visible.st_ino,
            ) != (claimed.st_dev, claimed.st_ino):
                raise RuntimeError("qualification output directory identity changed")

            # Claim the opened directory before publishing status. Exclusive handle-relative
            # creation makes concurrent attempts fail without following a replaced pathname.
            lock_fd = os.open(
                ".qualification-attempt.lock",
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                0o600,
                dir_fd=directory_fd,
            )
            with os.fdopen(lock_fd, "w", encoding="utf-8") as lock:
                lock.write(f"revision={revision}\npid={os.getpid()}\n")
            attempt = cls(
                output_dir,
                revision,
                directory_fd,
                (claimed.st_dev, claimed.st_ino),
                attempt_schema,
            )
            attempt.write_status("running")
            return attempt
        except BaseException:
            os.close(directory_fd)
            raise

    def close(self) -> None:
        """Close the retained output-directory descriptor."""
        if self.directory_fd >= 0:
            os.close(self.directory_fd)
            self.directory_fd = -1

    def verify_directory_identity(self) -> None:
        """Verify that the authored path still identifies the claimed directory."""
        claimed = os.fstat(self.directory_fd)
        try:
            visible = os.stat(self.output_dir, follow_symlinks=False)
        except FileNotFoundError as error:
            raise RuntimeError("qualification output directory identity changed") from error
        if (
            not stat.S_ISDIR(visible.st_mode)
            or (claimed.st_dev, claimed.st_ino) != self.directory_identity
            or (visible.st_dev, visible.st_ino) != self.directory_identity
        ):
            raise RuntimeError("qualification output directory identity changed")

    def open_temporary_status(self) -> tuple[int, str]:
        """Create one owned status temporary file relative to the claimed directory."""
        for _ in range(16):
            name = f".attempt-{os.getpid()}-{secrets.token_hex(8)}.json.tmp"
            try:
                descriptor = os.open(
                    name,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                    0o600,
                    dir_fd=self.directory_fd,
                )
                return descriptor, name
            except FileExistsError:
                continue
        raise FileExistsError("could not reserve qualification status temporary file")

    def write_status(self, status: str, detail: str | None = None) -> None:
        self.verify_directory_identity()
        record = {
            "schema": self.attempt_schema,
            "revision": self.revision,
            "status": status,
            "summary": "summary.json" if status == "succeeded" else None,
            "detail": detail,
        }
        contents = json.dumps(record, indent=2, sort_keys=True) + "\n"
        descriptor, temporary_name = self.open_temporary_status()
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as output:
                output.write(contents)
                output.flush()
                os.fsync(output.fileno())
            os.replace(
                temporary_name,
                "attempt.json",
                src_dir_fd=self.directory_fd,
                dst_dir_fd=self.directory_fd,
            )
            os.fsync(self.directory_fd)
        finally:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(temporary_name, dir_fd=self.directory_fd)

    def finish(self, status: str, detail: str | None = None) -> None:
        self.verify_directory_identity()
        try:
            summary = os.stat("summary.json", dir_fd=self.directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            summary = None
        if status == "succeeded" and (summary is None or not stat.S_ISREG(summary.st_mode)):
            raise RuntimeError("successful qualification attempt did not publish summary.json")
        if status != "succeeded":
            with contextlib.suppress(FileNotFoundError):
                os.unlink("summary.json", dir_fd=self.directory_fd)
        self.write_status(status, detail)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark_binary", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--revision",
        required=True,
        help="exact source revision represented by the benchmark binary",
    )
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--warmup", type=int, default=1_000)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--tensor-bytes", type=int, default=1 << 20)
    parser.add_argument("--tensor-device", type=int, default=0)
    parser.add_argument("--publisher-period-us", type=int, default=1_000)
    parser.add_argument("--session-timeout-ms", type=int, default=60_000)
    parser.add_argument("--process-timeout-sec", type=float, default=120.0)
    parser.add_argument(
        "--workloads", nargs="+", choices=("latency", "saturated"), default=["latency", "saturated"]
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.samples <= 0:
        raise ValueError("--samples must be positive")
    if args.warmup < 0:
        raise ValueError("--warmup must be nonnegative")
    if args.repetitions < 2:
        raise ValueError("--repetitions must be at least two")
    if args.tensor_bytes <= 0:
        raise ValueError("--tensor-bytes must be positive")
    if args.tensor_device < 0:
        raise ValueError("--tensor-device must be nonnegative")
    if args.publisher_period_us <= 0:
        raise ValueError("--publisher-period-us must be positive")
    if args.session_timeout_ms <= 0 or args.process_timeout_sec <= 0:
        raise ValueError("qualification timeouts must be positive")
    if not args.revision.strip():
        raise ValueError("--revision must be nonempty")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def command_output(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            command, check=True, capture_output=True, text=True, timeout=10.0
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None


def gpu_metadata() -> list[dict[str, str]]:
    if shutil.which("nvidia-smi") is None:
        return []
    fields = (
        "name,uuid,pci.bus_id,driver_version,persistence_mode,power.limit,"
        "clocks.current.sm,clocks.max.sm"
    )
    output = command_output(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"]
    )
    if output is None:
        return []
    keys = (
        "name",
        "uuid",
        "pci_bus_id",
        "driver_version",
        "persistence_mode",
        "power_limit_w",
        "current_sm_clock_mhz",
        "maximum_sm_clock_mhz",
    )
    return [
        dict(zip(keys, (field.strip() for field in line.split(",")), strict=True))
        for line in output.splitlines()
        if len(line.split(",")) == len(keys)
    ]


def native_command(
    args: argparse.Namespace,
    cell: MatrixCell,
    workload: str,
    result_path: Path,
    plan_path: Path,
    instrumented: bool,
) -> list[str]:
    queue_depth = 1 if workload == "latency" else 8
    notification_capacity = 1 if workload == "latency" else queue_depth
    command = [
        str(args.benchmark_binary),
        "--scenario",
        "tensor",
        "--tensor-storage",
        "device",
        "--tensor-device",
        str(args.tensor_device),
        "--tensor-bytes",
        str(args.tensor_bytes),
        "--operator-lanes",
        cell.operator_lanes,
        "--green-context-layout",
        cell.layout,
        "--workload",
        workload,
        "--samples",
        str(args.samples if not instrumented else min(args.samples, 1_000)),
        "--warmup",
        str(args.warmup if not instrumented else min(args.warmup, 100)),
        "--queue-depth",
        str(queue_depth),
        "--notification-capacity",
        str(notification_capacity),
        "--publisher-period-us",
        str(
            max(args.publisher_period_us, 1_000)
            if workload == "latency" and instrumented
            else (args.publisher_period_us if workload == "latency" else 0)
        ),
        "--payload-validation",
        "final",
        "--timeout-ms",
        str(args.session_timeout_ms),
        "--output",
        str(result_path),
        "--plan-output",
        str(plan_path),
    ]
    if instrumented:
        command.append("--cuda-resource-counters")
        if workload == "latency":
            command.append("--runtime-phase-counters")
    return command


def checked_result(
    path: Path, cell: MatrixCell, workload: str, instrumented: bool
) -> dict[str, Any]:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid qualification result {path}: {error}") from error
    green = result.get("green_context")
    validation = result.get("validation")
    session = result.get("session")
    allocation = result.get("allocation_probe")
    counters = result.get("cuda_resource_counters")
    timing = result.get("timing")
    cold = result.get("cold_timing_ns")
    thread_usage = result.get("runtime_thread_usage")
    if (
        result.get("scenario") != "tensor"
        or result.get("workload") != workload
        or not isinstance(green, dict)
        or green.get("layout") != cell.layout
        or green.get("enabled") is not (cell.layout != "none")
        or not isinstance(validation, dict)
        or validation.get("passed") is not True
        or not isinstance(session, dict)
        or session.get("termination") != "stopped"
        or session.get("diagnostics") != []
        or not isinstance(allocation, dict)
        or allocation.get("source_compute_allocations") != 0
        or allocation.get("sink_compute_allocations") != 0
        or not isinstance(counters, dict)
        or counters.get("enabled") is not instrumented
        or not isinstance(timing, dict)
        or not isinstance(cold, dict)
    ):
        raise RuntimeError(f"{path} failed the Green Context qualification contract")
    latency = timing.get("latency_ns")
    if workload == "latency" and (
        not isinstance(latency, dict)
        or any(not isinstance(latency.get(key), int) for key in ("p50", "p95", "p99"))
    ):
        raise RuntimeError(f"{path} lacks p50/p95/p99 latency")
    if workload == "saturated" and not isinstance(
        timing.get("logical_publications_per_second"), (int, float)
    ):
        raise RuntimeError(f"{path} lacks saturated throughput")
    if (
        instrumented
        and workload == "latency"
        and (
            not isinstance(thread_usage, dict)
            or thread_usage.get("enabled") is not True
            or not isinstance(thread_usage.get("aggregate"), dict)
            or not isinstance(thread_usage["aggregate"].get("samples"), int)
            or thread_usage["aggregate"]["samples"] <= 0
        )
    ):
        raise RuntimeError(f"{path} lacks Runtime CPU usage samples")
    if instrumented and (
        counters.get("streams_created") != counters.get("streams_destroyed")
        or counters.get("events_created") != counters.get("events_destroyed")
        or not isinstance(counters.get("stream_quiesce_ns"), int)
    ):
        raise RuntimeError(f"{path} reports incomplete CUDA resource retirement")
    return result


def run_one(command: list[str], timeout: float) -> int:
    try:
        completed = subprocess.run(command, check=False, timeout=timeout)
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"qualification command timed out: {' '.join(command)}") from error
    if completed.returncode not in (0, SKIPPED):
        raise RuntimeError(
            f"qualification command failed ({completed.returncode}): {' '.join(command)}"
        )
    return completed.returncode


def numeric_path(value: dict[str, Any], *keys: str) -> float:
    current: object = value
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            raise RuntimeError(f"missing numeric field {'.'.join(keys)}")
        current = current[key]
    if isinstance(current, bool) or not isinstance(current, (int, float)):
        raise RuntimeError(f"invalid numeric field {'.'.join(keys)}")
    return float(current)


def summarize(values: list[float]) -> dict[str, float | int]:
    mean = statistics.fmean(values)
    return {
        "count": len(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "mean": mean,
        "coefficient_of_variation": statistics.pstdev(values) / mean if mean else 0.0,
    }


def metric_summary(results: list[dict[str, Any]], workload: str) -> dict[str, object]:
    metrics: dict[str, object] = {
        "compile_ns": summarize(
            [numeric_path(row, "cold_timing_ns", "compile") for row in results]
        ),
        "run_async_return_ns": summarize(
            [numeric_path(row, "cold_timing_ns", "run_async_return") for row in results]
        ),
        "run_async_to_first_callback_ns": summarize(
            [numeric_path(row, "cold_timing_ns", "run_async_to_first_callback") for row in results]
        ),
        "terminal_wait_ns": summarize(
            [numeric_path(row, "cold_timing_ns", "terminal_wait") for row in results]
        ),
    }
    if workload == "latency":
        for percentile in ("p50", "p95", "p99"):
            metrics[f"latency_{percentile}_ns"] = summarize(
                [numeric_path(row, "timing", "latency_ns", percentile) for row in results]
            )
    else:
        metrics["logical_publications_per_second"] = summarize(
            [numeric_path(row, "timing", "logical_publications_per_second") for row in results]
        )
    return metrics


def median_metric(summary: dict[str, object], workload: str) -> float:
    key = "latency_p50_ns" if workload == "latency" else "logical_publications_per_second"
    metric = summary[key]
    if not isinstance(metric, dict) or not isinstance(metric.get("median"), (int, float)):
        raise RuntimeError("matrix summary lacks a comparison median")
    return float(metric["median"])


def compare(value: float, baseline: float, workload: str) -> float:
    if baseline == 0:
        return math.inf
    raw = 100.0 * (value - baseline) / baseline
    return -raw if workload == "latency" else raw


def run_qualification(args: argparse.Namespace) -> int:
    commands: list[list[str]] = []
    artifacts: list[dict[str, str]] = []
    cells: dict[str, dict[str, object]] = {}
    for workload in args.workloads:
        workload_cells: dict[str, object] = {}
        for cell in MATRIX:
            cell_dir = args.output_dir / workload / cell.name
            cell_dir.mkdir(parents=True, exist_ok=True)
            results: list[dict[str, Any]] = []
            for repetition in range(args.repetitions):
                result_path = cell_dir / f"run-{repetition + 1}.json"
                plan_path = cell_dir / f"run-{repetition + 1}-plan.json"
                command = native_command(args, cell, workload, result_path, plan_path, False)
                commands.append(command)
                if run_one(command, args.process_timeout_sec) == SKIPPED:
                    print("SKIP: CUDA Green Context qualification is unavailable", file=sys.stderr)
                    return SKIPPED
                result = checked_result(result_path, cell, workload, False)
                results.append(result)
                artifacts.extend(
                    {"path": str(artifact), "sha256": sha256_file(artifact)}
                    for artifact in (result_path, plan_path)
                )

            instrumentation_result = cell_dir / "instrumentation.json"
            instrumentation_plan = cell_dir / "instrumentation-plan.json"
            instrumentation_command = native_command(
                args, cell, workload, instrumentation_result, instrumentation_plan, True
            )
            commands.append(instrumentation_command)
            if run_one(instrumentation_command, args.process_timeout_sec) == SKIPPED:
                return SKIPPED
            instrumentation = checked_result(instrumentation_result, cell, workload, True)
            artifacts.extend(
                {"path": str(artifact), "sha256": sha256_file(artifact)}
                for artifact in (instrumentation_result, instrumentation_plan)
            )
            workload_cells[cell.name] = {
                "layout": cell.layout,
                "operator_lanes": cell.operator_lanes,
                "baseline": cell.baseline,
                "metrics": metric_summary(results, workload),
                "device_memory_pool_capacity_bytes": numeric_path(
                    results[0], "green_context", "device_memory_pool_capacity_bytes"
                ),
                "instrumentation": {
                    "cuda_resource_counters": instrumentation["cuda_resource_counters"],
                    "runtime_thread_usage": instrumentation.get("runtime_thread_usage"),
                    "runtime_phase_trace": instrumentation.get("runtime_phase_trace"),
                },
            }
        cells[workload] = workload_cells

    comparisons: list[dict[str, object]] = []
    for workload, workload_cells in cells.items():
        for cell in MATRIX:
            if cell.baseline is None:
                continue
            current = workload_cells[cell.name]
            baseline = workload_cells[cell.baseline]
            if not isinstance(current, dict) or not isinstance(baseline, dict):
                raise RuntimeError("invalid matrix summary")
            current_metrics = current["metrics"]
            baseline_metrics = baseline["metrics"]
            if not isinstance(current_metrics, dict) or not isinstance(baseline_metrics, dict):
                raise RuntimeError("invalid matrix metrics")
            comparisons.append(
                {
                    "workload": workload,
                    "cell": cell.name,
                    "baseline": cell.baseline,
                    "favorable_delta_percent": compare(
                        median_metric(current_metrics, workload),
                        median_metric(baseline_metrics, workload),
                        workload,
                    ),
                    "criterion": (
                        "negative_is_latency_regression"
                        if workload == "latency"
                        else "negative_is_throughput_regression"
                    ),
                }
            )

    summary = {
        "schema": SCHEMA,
        "revision": args.revision,
        "host": platform.node(),
        "kernel": platform.release(),
        "python": platform.python_version(),
        "cuda_toolkit": command_output(["nvcc", "--version"]),
        "gpus": gpu_metadata(),
        "geometry": {
            "samples": args.samples,
            "warmup": args.warmup,
            "repetitions": args.repetitions,
            "tensor_bytes": args.tensor_bytes,
            "tensor_device": args.tensor_device,
            "publisher_period_us": args.publisher_period_us,
        },
        "cells": cells,
        "comparisons": comparisons,
        "commands": commands,
        "artifacts": artifacts,
        "realtime_profile_disposition": REALTIME_PROFILE_DISPOSITION,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def main() -> int:
    args = parse_args()
    validate_args(args)
    args.benchmark_binary = args.benchmark_binary.resolve()
    args.output_dir = args.output_dir.absolute()
    if not args.benchmark_binary.is_file():
        raise FileNotFoundError(f"benchmark binary does not exist: {args.benchmark_binary}")

    attempt = QualificationAttempt.begin(args.output_dir, args.revision)
    try:
        try:
            result = run_qualification(args)
            if result == SKIPPED:
                attempt.finish("skipped", "CUDA Green Context qualification is unavailable")
            else:
                attempt.finish("succeeded")
            return result
        except BaseException as error:
            try:
                attempt.finish("failed", str(error))
            except BaseException as status_error:
                # Preserve the qualification failure as the raised exception. The atomic status
                # writer leaves the last complete attempt.json if terminal publication also fails.
                raise error from status_error
            raise
    finally:
        attempt.close()


if __name__ == "__main__":
    raise SystemExit(main())
