#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a matched CUDA compute-stream policy characterization matrix."""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from run_green_context_qualification import (
    QualificationAttempt,
    command_output,
    gpu_metadata,
    run_one,
    sha256_file,
)

SCHEMA = "holoscan.ga.cuda-stream-policy-qualification/v1"
ATTEMPT_SCHEMA = "holoscan.ga.cuda-stream-policy-qualification-attempt/v1"
SKIPPED = 77


@dataclass(frozen=True)
class Regime:
    """One GPU-work geometry compared under both stream policies."""

    name: str
    profile: str
    branch_work_us: tuple[int, int] | None


@dataclass(frozen=True)
class Cell:
    """One policy/regime matrix cell."""

    policy: str
    regime: Regime

    @property
    def name(self) -> str:
        return f"{self.regime.name}_{self.policy}"


def regimes(args: argparse.Namespace) -> tuple[Regime, ...]:
    """Return the fixed control, overlap, saturation, and asymmetry regimes."""
    return (
        Regime("no_device_work", "none", None),
        Regime("low_occupancy", "low_occupancy", (args.low_work_us, args.low_work_us)),
        Regime("saturating", "saturating", (args.saturating_work_us, args.saturating_work_us)),
        Regime("asymmetric", "low_occupancy", tuple(args.asymmetric_work_us)),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark_binary", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--samples", type=int, default=1_000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--tensor-bytes", type=int, default=1 << 20)
    parser.add_argument("--tensor-device", type=int, default=0)
    parser.add_argument("--publisher-period-us", type=int, default=5_000)
    parser.add_argument("--low-work-us", type=int, default=1_000)
    parser.add_argument("--saturating-work-us", type=int, default=1_000)
    parser.add_argument("--asymmetric-work-us", type=int, nargs=2, default=(500, 2_500))
    parser.add_argument("--overlap-samples", type=int, default=100)
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
    if args.tensor_bytes <= 0 or args.tensor_device < 0:
        raise ValueError("Tensor bytes must be positive and the device must be nonnegative")
    if args.publisher_period_us <= 0:
        raise ValueError("--publisher-period-us must be positive")
    durations = [args.low_work_us, args.saturating_work_us, *args.asymmetric_work_us]
    if any(duration <= 0 or duration > 1_000_000 for duration in durations):
        raise ValueError("CUDA work durations must be between 1 and 1000000 us")
    if args.overlap_samples <= 0:
        raise ValueError("--overlap-samples must be positive")
    if args.session_timeout_ms <= 0 or args.process_timeout_sec <= 0:
        raise ValueError("qualification timeouts must be positive")
    if not args.revision.strip():
        raise ValueError("--revision must be nonempty")


def native_command(
    args: argparse.Namespace,
    cell: Cell,
    workload: str,
    result_path: Path,
    plan_path: Path,
) -> list[str]:
    queue_depth = 1 if workload == "latency" else 8
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
        "--fanout",
        "2",
        "--operator-lanes",
        "separate",
        "--stream-policy",
        cell.policy,
        "--cuda-work-profile",
        cell.regime.profile,
        "--cuda-overlap-samples",
        str(min(args.samples, args.overlap_samples)),
        "--workload",
        workload,
        "--samples",
        str(args.samples),
        "--warmup",
        str(args.warmup),
        "--queue-depth",
        str(queue_depth),
        "--notification-capacity",
        str(1 if workload == "latency" else queue_depth),
        "--publisher-period-us",
        str(args.publisher_period_us if workload == "latency" else 0),
        "--payload-validation",
        "final",
        "--cuda-resource-counters",
        "--timeout-ms",
        str(args.session_timeout_ms),
        "--output",
        str(result_path),
        "--plan-output",
        str(plan_path),
    ]
    if cell.regime.branch_work_us is not None:
        command.extend(
            (
                "--cuda-branch-work-us",
                f"{cell.regime.branch_work_us[0]},{cell.regime.branch_work_us[1]}",
            )
        )
    return command


def checked_result(path: Path, cell: Cell, workload: str, overlap_samples: int) -> dict[str, Any]:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid qualification result {path}: {error}") from error
    if not isinstance(result, dict):
        raise RuntimeError(f"{path} failed the CUDA stream-policy qualification contract")
    execution = result.get("cuda_execution")
    work = result.get("cuda_work")
    counters = result.get("cuda_resource_counters")
    timing = result.get("timing")
    validation = result.get("validation")
    session = result.get("session")
    if (
        result.get("scenario") != "tensor"
        or result.get("workload") != workload
        or not isinstance(execution, dict)
        or execution.get("stream_policy") != cell.policy
        or execution.get("plan_stream_policy") != cell.policy
        or execution.get("stream_records") != (1 if cell.policy == "shared" else 3)
        or execution.get("operator_binding_records") != 3
        or not isinstance(work, dict)
        or work.get("profile") != cell.regime.profile
        or not isinstance(counters, dict)
        or counters.get("enabled") is not True
        or counters.get("streams_created") != counters.get("streams_destroyed")
        or counters.get("events_created") != counters.get("events_destroyed")
        or not isinstance(timing, dict)
        or not isinstance(validation, dict)
        or validation.get("passed") is not True
        or not isinstance(session, dict)
        or session.get("termination") != "stopped"
        or session.get("diagnostics") != []
    ):
        raise RuntimeError(f"{path} failed the CUDA stream-policy qualification contract")
    latency = timing.get("latency_ns")
    if workload == "latency" and (
        not isinstance(latency, dict)
        or any(not isinstance(latency.get(key), int) for key in ("p50", "p95", "p99", "max"))
    ):
        raise RuntimeError(f"{path} lacks end-to-end latency percentiles")
    if workload == "saturated" and not isinstance(
        timing.get("logical_publications_per_second"), (int, float)
    ):
        raise RuntimeError(f"{path} lacks saturated throughput")
    branch_latency = timing.get("source_to_sink_by_sink_ns")
    if workload == "latency" and (
        not isinstance(branch_latency, list)
        or len(branch_latency) != 2
        or any(
            not isinstance(item, dict) or not isinstance(item.get("latency_ns"), dict)
            for item in branch_latency
        )
    ):
        raise RuntimeError(f"{path} lacks per-branch latency")
    if cell.regime.profile == "none":
        if work.get("overlap_ns") is not None or work.get("branch_duration_ns") != [None, None]:
            raise RuntimeError(f"{path} reports CUDA timing for the no-device-work control")
    else:
        overlap = work.get("overlap_ns")
        durations = work.get("branch_duration_ns")
        completion = work.get("end_to_end_completion_latency_ns")
        branch_completion = work.get("branch_completion_latency_ns")
        if (
            not isinstance(overlap, dict)
            or overlap.get("samples") != overlap_samples
            or not isinstance(durations, list)
            or len(durations) != 2
            or any(
                not isinstance(item, dict) or item.get("samples") != overlap_samples
                for item in durations
            )
            or (
                workload == "latency"
                and (
                    not isinstance(completion, dict)
                    or completion.get("samples") != overlap_samples
                    or not isinstance(branch_completion, list)
                    or len(branch_completion) != 2
                    or any(
                        not isinstance(item, dict) or item.get("samples") != overlap_samples
                        for item in branch_completion
                    )
                )
            )
            or (cell.policy == "shared" and work.get("overlapping_samples") != 0)
        ):
            raise RuntimeError(f"{path} lacks complete CUDA branch/overlap timing")
    return result


def numeric_path(value: dict[str, Any], *keys: str | int) -> float:
    current: object = value
    for key in keys:
        if isinstance(key, int):
            if not isinstance(current, list) or key >= len(current):
                raise RuntimeError(f"missing numeric field {'.'.join(map(str, keys))}")
            current = current[key]
            continue
        if not isinstance(current, dict) or key not in current:
            raise RuntimeError(f"missing numeric field {'.'.join(map(str, keys))}")
        current = current[key]
    if isinstance(current, bool) or not isinstance(current, (int, float)):
        raise RuntimeError(f"invalid numeric field {'.'.join(map(str, keys))}")
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


def metric_summary(
    results: list[dict[str, Any]], workload: str, device_work: bool
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    if workload == "latency":
        for percentile in ("p50", "p95", "p99", "max"):
            metrics[f"latency_{percentile}_ns"] = summarize(
                [numeric_path(row, "timing", "latency_ns", percentile) for row in results]
            )
            for branch in range(2):
                metrics[f"branch_{branch}_{percentile}_ns"] = summarize(
                    [
                        numeric_path(
                            row,
                            "timing",
                            "source_to_sink_by_sink_ns",
                            branch,
                            "latency_ns",
                            percentile,
                        )
                        for row in results
                    ]
                )
    else:
        metrics["logical_publications_per_second"] = summarize(
            [numeric_path(row, "timing", "logical_publications_per_second") for row in results]
        )
    if device_work:
        metrics["overlap_p50_ns"] = summarize(
            [numeric_path(row, "cuda_work", "overlap_ns", "p50") for row in results]
        )
        metrics["overlapping_samples"] = summarize(
            [numeric_path(row, "cuda_work", "overlapping_samples") for row in results]
        )
        for branch in range(2):
            metrics[f"cuda_branch_{branch}_p50_ns"] = summarize(
                [
                    numeric_path(row, "cuda_work", "branch_duration_ns", branch, "p50")
                    for row in results
                ]
            )
        if workload == "latency":
            for percentile in ("p50", "p95", "p99", "max"):
                metrics[f"gpu_completion_latency_{percentile}_ns"] = summarize(
                    [
                        numeric_path(
                            row,
                            "cuda_work",
                            "end_to_end_completion_latency_ns",
                            percentile,
                        )
                        for row in results
                    ]
                )
                for branch in range(2):
                    metrics[f"gpu_branch_{branch}_completion_{percentile}_ns"] = summarize(
                        [
                            numeric_path(
                                row,
                                "cuda_work",
                                "branch_completion_latency_ns",
                                branch,
                                percentile,
                            )
                            for row in results
                        ]
                    )
    return metrics


def favorable_delta(value: float, baseline: float, lower_is_better: bool) -> float:
    if baseline == 0.0:
        return math.inf
    delta = 100.0 * (value - baseline) / baseline
    return -delta if lower_is_better else delta


def run_qualification(args: argparse.Namespace) -> int:
    commands: list[list[str]] = []
    artifacts: list[dict[str, str]] = [
        {"path": str(args.benchmark_binary), "sha256": sha256_file(args.benchmark_binary)}
    ]
    matrix: dict[str, dict[str, Any]] = {}
    for workload in args.workloads:
        workload_cells: dict[str, Any] = {}
        for regime in regimes(args):
            for policy in ("shared", "per_partition"):
                cell = Cell(policy, regime)
                cell_dir = args.output_dir / workload / cell.name
                cell_dir.mkdir(parents=True, exist_ok=True)
                results: list[dict[str, Any]] = []
                for repetition in range(args.repetitions):
                    result_path = cell_dir / f"run-{repetition + 1}.json"
                    plan_path = cell_dir / f"run-{repetition + 1}-plan.json"
                    command = native_command(args, cell, workload, result_path, plan_path)
                    commands.append(command)
                    if run_one(command, args.process_timeout_sec) == SKIPPED:
                        return SKIPPED
                    results.append(
                        checked_result(
                            result_path, cell, workload, min(args.samples, args.overlap_samples)
                        )
                    )
                    artifacts.extend(
                        {"path": str(path), "sha256": sha256_file(path)}
                        for path in (result_path, plan_path)
                    )
                workload_cells[cell.name] = {
                    "policy": policy,
                    "regime": regime.name,
                    "profile": regime.profile,
                    "branch_work_us": regime.branch_work_us,
                    "baseline": f"{regime.name}_shared" if policy == "per_partition" else None,
                    "metrics": metric_summary(results, workload, regime.profile != "none"),
                    "cuda_resource_counters": results[0]["cuda_resource_counters"],
                }
        matrix[workload] = workload_cells

    comparisons: list[dict[str, Any]] = []
    for workload, cells in matrix.items():
        for regime in regimes(args):
            metric = (
                "gpu_completion_latency_p50_ns"
                if workload == "latency" and regime.profile != "none"
                else (
                    "latency_p50_ns" if workload == "latency" else "logical_publications_per_second"
                )
            )
            current = cells[f"{regime.name}_per_partition"]["metrics"][metric]["median"]
            baseline = cells[f"{regime.name}_shared"]["metrics"][metric]["median"]
            comparisons.append(
                {
                    "workload": workload,
                    "regime": regime.name,
                    "cell": f"{regime.name}_per_partition",
                    "baseline": f"{regime.name}_shared",
                    "metric": metric,
                    "favorable_delta_percent": favorable_delta(
                        float(current), float(baseline), workload == "latency"
                    ),
                    "disposition": "characterization_only_no_pass_fail_threshold",
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
            "overlap_samples": args.overlap_samples,
        },
        "matrix": matrix,
        "comparisons": comparisons,
        "commands": commands,
        "artifacts": artifacts,
        "decision_policy": (
            "record evidence only; this runner defines no pass/fail performance threshold"
        ),
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
    attempt = QualificationAttempt.begin(args.output_dir, args.revision, ATTEMPT_SCHEMA)
    try:
        try:
            result = run_qualification(args)
            attempt.finish("skipped" if result == SKIPPED else "succeeded")
            return result
        except BaseException as error:
            try:
                attempt.finish("failed", str(error))
            except BaseException as status_error:
                raise error from status_error
            raise
    finally:
        attempt.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=__import__("sys").stderr)
        raise SystemExit(2) from error
