#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the maintained HoloMQ, Channel, and Runtime Graph transport benchmarks."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any

from transport_benchmark_identity import scenario_identity_contract

BENCHMARK_DIR = Path(__file__).resolve().parent
PUBLIC_DIR = BENCHMARK_DIR.parent
HOLOMQ_RUNNER = PUBLIC_DIR / "benchmark/holomq/run_holomq_benchmark.py"
CHANNEL_MATRIX_RUNNER = BENCHMARK_DIR / "channel_interprocess/run_channel_benchmark_matrix.py"
GRAPH_RUNNER = BENCHMARK_DIR / "runtime/run_runtime_graph_benchmark.py"
GRAPH_INTERPROCESS_RUNNER = BENCHMARK_DIR / "runtime/run_runtime_graph_interprocess_benchmark.py"
SUMMARIZER = BENCHMARK_DIR / "summarize_transport_benchmark_suite.py"
SUITE_SCHEMA = "holoscan.transport-benchmark-suite/v3"
GROUPS = (
    "holomq_intraprocess",
    "holomq_interprocess",
    "channel",
    "graph_intraprocess",
    "graph_interprocess",
)
DOCS_CORE_RUNTIME_GROUPS = ("graph_intraprocess", "graph_interprocess")
DOCS_CORE_RUNTIME_PRESET = "docs-core-runtime"
GRAPH_INTERPROCESS_TENSOR_PAYLOADS = frozenset(
    {"tensor", "image", "multi_tensor_loan", "multi_tensor_staging"}
)
MINIMUM_P99_SAMPLES = 1_000
MINIMUM_P999_SAMPLES = 10_000
THROUGHPUT_QUEUE_DEPTHS = (1, 2, 8, 64)
DEFAULT_GRAPH_SESSION_TIMEOUT_MS = 30_000
DEFAULT_GRAPH_PROCESS_TIMEOUT_SEC = 90
PRODUCTION_GRAPH_INTRAPROCESS_SESSION_TIMEOUT_MS = 120_000
PRODUCTION_GRAPH_INTRAPROCESS_PROCESS_TIMEOUT_SEC = 150
WORKLOAD_GEOMETRY_POLICY = {
    "matched_scenario": "pod64_1to1_saturated_throughput",
    "throughput_queue_depths": list(THROUGHPUT_QUEUE_DEPTHS),
    "graph_notification_capacity": "matches_queue_depth_for_notification_driven_sweeps",
    "graph_fanout_latency_queue_depth": 2,
    "exclusions": {
        "holomq_intraprocess_google_benchmark": (
            "kernel-owned queue geometry is not exposed by these diagnostic rows"
        ),
        "bulk_payloads": (
            "large Tensor and Image rows retain one representative depth to bound suite duration"
        ),
    },
}
SMOKE_MEASUREMENT_POLICY = {
    "repetitions": 1,
    "inline": {
        "throughput_samples": 1_000,
        "latency_samples": 1_000,
        "warmup_samples": 100,
        "minimum_elapsed_seconds": 0.01,
    },
    "bulk_host": {
        "throughput_samples": 200,
        "latency_samples": 200,
        "warmup_samples": 100,
        "minimum_elapsed_seconds": 0.01,
    },
    "bulk_device": {
        "throughput_samples": 100,
        "latency_samples": 100,
        "warmup_samples": 100,
        "minimum_elapsed_seconds": 0.01,
    },
    "percentiles": {
        "p99_minimum_samples": MINIMUM_P99_SAMPLES,
        "p999_minimum_samples": MINIMUM_P999_SAMPLES,
    },
}
PRODUCTION_MEASUREMENT_POLICY = {
    "repetitions": 7,
    "inline": {
        "throughput_samples": 100_000,
        "latency_samples": 10_000,
        "warmup_samples": 1_000,
        "minimum_elapsed_seconds": 0.1,
    },
    "bulk_host": {
        "throughput_samples": 20_000,
        "latency_samples": 20_000,
        "warmup_samples": 1_000,
        "minimum_elapsed_seconds": 0.25,
    },
    "bulk_device": {
        "throughput_samples": 10_000,
        "latency_samples": 10_000,
        "warmup_samples": 1_000,
        "minimum_elapsed_seconds": 0.25,
    },
    "percentiles": {
        "p99_minimum_samples": MINIMUM_P99_SAMPLES,
        "p999_minimum_samples": MINIMUM_P999_SAMPLES,
    },
}
DOCS_CORE_RUNTIME_MEASUREMENT_POLICY = {
    **PRODUCTION_MEASUREMENT_POLICY,
    "exceptions": {
        "interprocess_latency_samples": 1_000,
        "interprocess_latency_warmup_samples": 1_000,
        "interprocess_latency_percentiles": ["p50", "p99"],
    },
}
BALANCED_MEASUREMENT_POLICY = {
    "repetitions": 3,
    "inline": {
        "throughput_samples": 50_000,
        "latency_samples": 5_000,
        "warmup_samples": 500,
        "minimum_elapsed_seconds": 0.05,
    },
    "bulk_host": {
        "throughput_samples": 10_000,
        "latency_samples": 5_000,
        "warmup_samples": 500,
        "minimum_elapsed_seconds": 0.1,
    },
    "bulk_device": {
        "throughput_samples": 5_000,
        "latency_samples": 5_000,
        "warmup_samples": 500,
        "minimum_elapsed_seconds": 0.1,
    },
    "percentiles": {
        "p99_minimum_samples": MINIMUM_P99_SAMPLES,
        "p999_minimum_samples": MINIMUM_P999_SAMPLES,
    },
    "exceptions": {
        "distributed_graph_1to2_latency_samples": 2_000,
        "distributed_graph_1to2_warmup_samples": 200,
    },
}


@dataclass
class BenchmarkCase:
    case_id: str
    layer: str
    domain: str
    workload: str
    result_kind: str
    output: Path
    command: list[str]
    optional: bool = False
    status: str = "planned"
    returncode: int | None = None
    required_cpus: int | None = None
    skip_reason: str | None = None

    def manifest_record(self, root: Path) -> dict[str, Any]:
        record = asdict(self)
        record["output"] = str(self.output.relative_to(root))
        return record


def parse_cpu_list(text: str) -> list[int]:
    try:
        cpus = [int(item.strip()) for item in text.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("--cpus must contain integer ordinals") from error
    if not cpus or any(cpu < 0 for cpu in cpus) or len(cpus) != len(set(cpus)):
        raise argparse.ArgumentTypeError("--cpus must contain unique nonnegative ordinals")
    return cpus


def validate_distinct_physical_cpus(
    cpus: list[int],
    sysfs_root: Path | None = None,
) -> None:
    root = sysfs_root or Path(os.environ.get("HOLOSCAN_CPU_SYSFS_ROOT", "/sys/devices/system/cpu"))
    selected_cores: dict[tuple[int, str], int] = {}
    for cpu in cpus:
        cpu_dir = root / f"cpu{cpu}"
        if not cpu_dir.is_dir():
            raise ValueError(f"CPU {cpu} is absent from {root}")
        online_path = cpu_dir / "online"
        if online_path.is_file() and online_path.read_text(encoding="utf-8").strip() != "1":
            raise ValueError(f"CPU {cpu} is offline")
        package_path = cpu_dir / "topology/physical_package_id"
        core_path = cpu_dir / "topology/core_id"
        siblings_path = cpu_dir / "topology/thread_siblings_list"
        package = (
            int(package_path.read_text(encoding="utf-8").strip()) if package_path.is_file() else 0
        )
        core = int(core_path.read_text(encoding="utf-8").strip()) if core_path.is_file() else cpu
        siblings = (
            siblings_path.read_text(encoding="utf-8").strip() if siblings_path.is_file() else ""
        )
        physical_core = (package, f"siblings:{siblings}" if siblings else f"core:{core}")
        sibling = selected_cores.get(physical_core)
        if sibling is not None:
            raise ValueError(f"CPUs {sibling} and {cpu} are sibling threads on socket {package}")
        selected_cores[physical_core] = cpu


def parse_groups(text: str) -> list[str]:
    groups = [item.strip() for item in text.split(",") if item.strip()]
    if not groups or any(group not in GROUPS for group in groups):
        raise argparse.ArgumentTypeError(f"--groups must contain values from {','.join(GROUPS)}")
    return list(dict.fromkeys(groups))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--preset",
        choices=("smoke", "balanced", "production", DOCS_CORE_RUNTIME_PRESET),
        default="smoke",
    )
    parser.add_argument("--groups", type=parse_groups, default=list(GROUPS))
    parser.add_argument(
        "--cpus",
        type=parse_cpu_list,
        help=(
            "Comma-separated CPUs on distinct physical cores. Most presets require eight; "
            "docs-core-runtime accepts four or more and skips rows that need additional cores. "
            "Omit to retain unpinned developer measurements."
        ),
    )
    parser.add_argument("--latency-period-us", type=int, default=200)
    parser.add_argument("--tensor-bytes", type=int, default=1 << 20)
    parser.add_argument("--repetitions", type=int)
    parser.add_argument(
        "--skip-gpu",
        action="store_true",
        help=(
            "Skip direct HoloMQ and Graph device rows. The existing Channel "
            "matrix still includes its maintained device-storage cases."
        ),
    )
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume failed and planned cases from an existing compatible suite directory.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--build",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Build the five benchmark binaries before running the suite.",
    )
    args = parser.parse_args()
    if args.preset == DOCS_CORE_RUNTIME_PRESET:
        args.groups = list(DOCS_CORE_RUNTIME_GROUPS)
        if args.skip_gpu:
            parser.error(f"--skip-gpu is incompatible with --preset {DOCS_CORE_RUNTIME_PRESET}")
    if args.latency_period_us <= 0:
        parser.error("--latency-period-us must be positive")
    if args.tensor_bytes <= 0:
        parser.error("--tensor-bytes must be positive")
    if args.repetitions is None:
        if args.preset == "smoke":
            args.repetitions = 1
        else:
            args.repetitions = int(measurement_policy(args)["repetitions"])
    if args.preset == DOCS_CORE_RUNTIME_PRESET:
        if args.cpus is None:
            parser.error(f"--preset {DOCS_CORE_RUNTIME_PRESET} requires --cpus")
        if args.repetitions != PRODUCTION_MEASUREMENT_POLICY["repetitions"]:
            parser.error(
                f"--preset {DOCS_CORE_RUNTIME_PRESET} requires "
                f"--repetitions {PRODUCTION_MEASUREMENT_POLICY['repetitions']}"
            )
        if args.latency_period_us != 200:
            parser.error(f"--preset {DOCS_CORE_RUNTIME_PRESET} requires --latency-period-us 200")
        if args.tensor_bytes != 1 << 20:
            parser.error(f"--preset {DOCS_CORE_RUNTIME_PRESET} requires --tensor-bytes 1048576")
    if args.repetitions <= 0:
        parser.error("--repetitions must be positive")
    if args.cpus is not None:
        minimum_cpus = 4 if args.preset == DOCS_CORE_RUNTIME_PRESET else 8
        if len(args.cpus) < minimum_cpus:
            parser.error(
                f"--preset {args.preset} requires at least {minimum_cpus} distinct physical CPUs"
            )
        try:
            validate_distinct_physical_cpus(args.cpus)
        except (OSError, ValueError) as error:
            parser.error(f"--cpus must select distinct online physical cores: {error}")
    return args


def cpu(args: argparse.Namespace, index: int) -> int:
    return args.cpus[index] if args.cpus is not None else -1


def docs_intraprocess_required_cpus(
    workload: str,
    scenario: str,
    fanout: int,
    fanout_operator_lanes: str | None,
) -> int:
    """Return the distinct CPU roles needed by one docs intraprocess row."""
    notification_driver = 1 if workload == "throughput" else 0
    if scenario == "sample_forward":
        return 4 + notification_driver
    if fanout > 1 and fanout_operator_lanes == "shared":
        return 2 + notification_driver
    if fanout > 1:
        return fanout + 3 + notification_driver
    return 3 + notification_driver


def docs_interprocess_required_cpus(topology: str, fanout: int) -> int:
    """Return the distinct process-lane and support-thread roles for a docs row."""
    return 8 if topology == "chain" or fanout > 1 else 5


def preset_counts(
    args: argparse.Namespace, workload: str, payload_class: str = "inline"
) -> tuple[int, int]:
    policy = measurement_policy(args)[payload_class]
    sample_key = "latency_samples" if workload == "latency" else "throughput_samples"
    return int(policy[sample_key]), int(policy["warmup_samples"])


def measurement_policy(args: argparse.Namespace) -> dict[str, Any]:
    if args.preset == "smoke":
        return SMOKE_MEASUREMENT_POLICY
    if args.preset == "balanced":
        return BALANCED_MEASUREMENT_POLICY
    if args.preset == DOCS_CORE_RUNTIME_PRESET:
        return DOCS_CORE_RUNTIME_MEASUREMENT_POLICY
    return PRODUCTION_MEASUREMENT_POLICY


def child_preset(args: argparse.Namespace) -> str:
    return "smoke" if args.preset == "smoke" else "production"


def binary_paths(build_dir: Path) -> dict[str, Path]:
    return {
        "holomq_intraprocess": (build_dir / "benchmark/holomq/holomq_pubsub_benchmarks"),
        "holomq_interprocess": (build_dir / "benchmark/holomq/holomq_transport_benchmark"),
        "channel": (build_dir / "benchmark/channel_interprocess/channel_interprocess_benchmark"),
        "graph_intraprocess": (build_dir / "benchmark/runtime/runtime_graph_benchmark"),
        "graph_interprocess": (
            build_dir / "benchmark/runtime/runtime_graph_interprocess_benchmark"
        ),
    }


def add_holomq_intraprocess_cases(
    cases: list[BenchmarkCase],
    args: argparse.Namespace,
    binaries: dict[str, Path],
    output_dir: Path,
) -> None:
    result = output_dir / "holomq/intraprocess/results.json"
    minimum_time = {
        "smoke": "0.01s",
        "balanced": "0.25s",
        "production": "1s",
    }[args.preset]
    command = [
        str(binaries["holomq_intraprocess"]),
        (
            "--benchmark_filter=^(BM_PubSubPOD_OneToOne_Blocking|"
            "BM_PubSubPOD_Fanout_Blocking<.*>|"
            "BM_PubSubFixed_PollingRoundTrip<.*>)/real_time$"
        ),
        f"--benchmark_min_time={minimum_time}",
        f"--benchmark_repetitions={args.repetitions}",
        "--benchmark_report_aggregates_only=false",
        f"--benchmark_out={result}",
        "--benchmark_out_format=json",
    ]
    cases.append(
        BenchmarkCase(
            "holomq_intraprocess_transport",
            "holomq",
            "intraprocess",
            "mixed",
            "google_benchmark",
            result,
            command,
        )
    )
    concurrent_result = output_dir / "holomq/intraprocess/concurrent-results.json"
    concurrent_command = [
        str(binaries["holomq_intraprocess"]),
        "--benchmark_filter=^BM_PubSubPOD_Concurrent.*",
        f"--benchmark_min_time={minimum_time}",
        f"--benchmark_repetitions={args.repetitions}",
        "--benchmark_report_aggregates_only=false",
        f"--benchmark_out={concurrent_result}",
        "--benchmark_out_format=json",
    ]
    if args.cpus is not None:
        concurrent_command = [
            "cmake",
            "-E",
            "env",
            f"HOLOMQ_BENCHMARK_PUBLISHER_CPU={cpu(args, 0)}",
            f"HOLOMQ_BENCHMARK_SUBSCRIBER_CPU={cpu(args, 1)}",
            *concurrent_command,
        ]
    cases.append(
        BenchmarkCase(
            "holomq_intraprocess_concurrent_1to1",
            "holomq",
            "intraprocess",
            "mixed",
            "google_benchmark",
            concurrent_result,
            concurrent_command,
        )
    )


def holomq_interprocess_command(
    args: argparse.Namespace,
    binary: Path,
    output: Path,
    workload: str,
    *,
    payload_bytes: int = 64,
    subscribers: int = 1,
    buffer: bool = False,
    storage: str = "pageable_host",
    queue_depth: int = 64,
    device_sync: str | None = None,
) -> list[str]:
    device = storage == "device"
    # Measured presets reuse device contents initialized before timing. Smoke retains the complete
    # ready/release protocol to exercise stream-ordered slot recycling. Measured presets add that
    # profile as a separate case so the transport-only and fully ordered costs remain visible.
    resolved_device_sync = (
        device_sync
        if device_sync is not None
        else ("ready" if device and args.preset == "smoke" else "none")
    )
    payload_class = "bulk_device" if device else ("bulk_host" if buffer else "inline")
    samples, warmup = preset_counts(args, workload, payload_class)
    command = [
        sys.executable,
        str(HOLOMQ_RUNNER),
        "--binary",
        str(binary),
        "--output-dir",
        str(output),
        "--preset",
        child_preset(args),
        "--workload",
        workload,
        "--repetitions",
        str(args.repetitions),
        "--samples",
        str(samples),
        "--warmup",
        str(warmup),
        "--publisher-cpu",
        str(cpu(args, 0)),
        "--subscriber-count",
        str(subscribers),
        "--payload-validation",
        "all"
        if args.preset == "smoke" and not device
        else ("final" if args.preset == "smoke" else "none"),
        "--queue-depth",
        str(1 if workload == "latency" else queue_depth),
    ]
    subscriber_cpus = ",".join(str(cpu(args, index + 1)) for index in range(subscribers))
    if subscribers == 1:
        command.extend(["--subscriber-cpu", str(cpu(args, 1))])
    elif args.cpus is not None:
        command.extend(["--subscriber-cpus", subscriber_cpus])
    if workload == "latency":
        command.extend(
            [
                "--publisher-period-us",
                str(
                    max(500, args.latency_period_us)
                    if resolved_device_sync == "ready"
                    else args.latency_period_us
                ),
                "--publisher-pacing",
                "spin",
            ]
        )
    if buffer:
        command.extend(
            [
                "--payload",
                "buffer_handle",
                "--buffer-bytes",
                str(args.tensor_bytes),
                "--storage",
                storage,
            ]
        )
        if device:
            command.extend(
                [
                    "--device-sync",
                    resolved_device_sync,
                    "--skip-if-no-cuda",
                ]
            )
    else:
        command.extend(["--payload-bytes", str(payload_bytes)])
    return command


def add_holomq_interprocess_cases(
    cases: list[BenchmarkCase],
    args: argparse.Namespace,
    binaries: dict[str, Path],
    output_dir: Path,
) -> None:
    binary = binaries["holomq_interprocess"]
    for payload_bytes in (64, 128, 440):
        for workload in ("throughput", "latency"):
            queue_depths = (
                THROUGHPUT_QUEUE_DEPTHS
                if payload_bytes == 64 and workload == "throughput"
                else (64,)
            )
            for queue_depth in queue_depths:
                suffix = f"_q{queue_depth}" if len(queue_depths) > 1 else ""
                case_id = f"holomq_interprocess_inline_{payload_bytes}_{workload}{suffix}"
                output = output_dir / f"holomq/interprocess/{case_id}"
                cases.append(
                    BenchmarkCase(
                        case_id,
                        "holomq",
                        "interprocess",
                        workload,
                        "holomq_interprocess",
                        output / "summary.json",
                        holomq_interprocess_command(
                            args,
                            binary,
                            output,
                            workload,
                            payload_bytes=payload_bytes,
                            queue_depth=queue_depth,
                        ),
                    )
                )
        fanout_id = f"holomq_interprocess_inline_{payload_bytes}_1to2_throughput"
        fanout_output = output_dir / f"holomq/interprocess/{fanout_id}"
        cases.append(
            BenchmarkCase(
                fanout_id,
                "holomq",
                "interprocess",
                "throughput",
                "holomq_interprocess",
                fanout_output / "summary.json",
                holomq_interprocess_command(
                    args,
                    binary,
                    fanout_output,
                    "throughput",
                    payload_bytes=payload_bytes,
                    subscribers=2,
                    queue_depth=64,
                ),
            )
        )
    for storage in ("pageable_host", "pinned_host", "device"):
        device = storage == "device"
        if device and args.skip_gpu:
            continue
        for workload in ("throughput", "latency"):
            case_id = f"holomq_interprocess_buffer_{storage}_{workload}"
            output = output_dir / f"holomq/interprocess/{case_id}"
            cases.append(
                BenchmarkCase(
                    case_id,
                    "holomq",
                    "interprocess",
                    workload,
                    "holomq_interprocess",
                    output / "summary.json",
                    holomq_interprocess_command(
                        args,
                        binary,
                        output,
                        workload,
                        buffer=True,
                        storage=storage,
                        queue_depth=64,
                    ),
                    optional=device,
                )
            )
            if device and args.preset != "smoke":
                ready_case_id = f"holomq_interprocess_buffer_{storage}_ready_{workload}"
                ready_output = output_dir / f"holomq/interprocess/{ready_case_id}"
                cases.append(
                    BenchmarkCase(
                        ready_case_id,
                        "holomq",
                        "interprocess",
                        workload,
                        "holomq_interprocess",
                        ready_output / "summary.json",
                        holomq_interprocess_command(
                            args,
                            binary,
                            ready_output,
                            workload,
                            buffer=True,
                            storage=storage,
                            queue_depth=64,
                            device_sync="ready",
                        ),
                        optional=True,
                    )
                )


def channel_matrix_command(
    args: argparse.Namespace,
    binary: Path,
    output: Path,
    workload: str,
    *,
    queue_depth: int,
    payloads: str = "pod,tensor,image",
    fanouts: str | None = None,
) -> list[str]:
    smoke = args.preset == "smoke"
    pod_samples, warmup = preset_counts(args, workload, "inline")
    host_samples, host_warmup = preset_counts(args, workload, "bulk_host")
    device_samples, device_warmup = preset_counts(args, workload, "bulk_device")
    if len({warmup, host_warmup, device_warmup}) != 1:
        raise RuntimeError("Channel matrix payload classes require one common warmup count")
    command = [
        sys.executable,
        str(CHANNEL_MATRIX_RUNNER),
        str(binary),
        "--output-dir",
        str(output),
        "--domains",
        "intraprocess,intraprocess_sidecar,interprocess",
        "--workload",
        workload,
        "--fanouts",
        fanouts or ("1,2" if workload == "throughput" else "1"),
        "--payloads",
        payloads,
        "--pod-samples",
        str(pod_samples),
        "--host-samples",
        str(host_samples),
        "--device-samples",
        str(device_samples),
        "--warmup",
        str(warmup),
        "--tensor-bytes",
        str(args.tensor_bytes),
        "--subscriber-queue-depth",
        str(queue_depth),
        "--publisher-period-us",
        "0" if workload == "throughput" else str(args.latency_period_us),
        "--publisher-pacing",
        "sleep" if workload == "throughput" else "spin",
        "--publisher-cpu",
        str(cpu(args, 0)),
        "--subscriber-cpu",
        str(cpu(args, 1)),
        "--payload-validation",
        "all" if smoke else "none",
    ]
    if args.cpus is not None:
        command.extend(["--subscriber-cpus", f"{cpu(args, 1)},{cpu(args, 2)}"])
    return command


def add_channel_cases(
    cases: list[BenchmarkCase],
    args: argparse.Namespace,
    binaries: dict[str, Path],
    output_dir: Path,
) -> None:
    for repetition in range(args.repetitions):
        for workload in ("throughput", "latency"):
            case_id = f"channel_matrix_{workload}_rep{repetition:03d}"
            output = output_dir / f"channel/{workload}/repetition_{repetition:03d}"
            cases.append(
                BenchmarkCase(
                    case_id,
                    "channel",
                    "mixed",
                    workload,
                    "channel_matrix",
                    output,
                    channel_matrix_command(
                        args,
                        binaries["channel"],
                        output,
                        workload,
                        queue_depth=64 if workload == "throughput" else 1,
                    ),
                )
            )
        for queue_depth in THROUGHPUT_QUEUE_DEPTHS[:-1]:
            case_id = f"channel_geometry_pod_q{queue_depth}_rep{repetition:03d}"
            output = output_dir / f"channel/geometry/q{queue_depth}/repetition_{repetition:03d}"
            cases.append(
                BenchmarkCase(
                    case_id,
                    "channel",
                    "mixed",
                    "throughput",
                    "channel_matrix",
                    output,
                    channel_matrix_command(
                        args,
                        binaries["channel"],
                        output,
                        "throughput",
                        queue_depth=queue_depth,
                        payloads="pod",
                        fanouts="1",
                    ),
                )
            )


def graph_intraprocess_command(
    args: argparse.Namespace,
    binary: Path,
    output: Path,
    workload: str,
    scenario: str,
    *,
    storage: str = "pageable_host",
    bulk_work_profile: str = "full_write",
    fanout: int = 1,
    relays: int = 1,
    force_holomq: bool = False,
    busy_poll: bool = False,
    queue_depth: int | None = None,
    fanout_operator_lanes: str | None = None,
) -> list[str]:
    effective_fanout_operator_lanes = fanout_operator_lanes or (
        "shared" if args.preset == DOCS_CORE_RUNTIME_PRESET else "per_sink"
    )
    if effective_fanout_operator_lanes not in {"shared", "per_sink"}:
        raise ValueError("fanout operator lanes must be shared or per_sink")
    shared_fanout = fanout > 1 and effective_fanout_operator_lanes == "shared"
    payload_class = (
        "bulk_device"
        if scenario == "tensor" and storage == "device"
        else ("bulk_host" if scenario in {"tensor", "image", "multi_tensor"} else "inline")
    )
    samples, warmup = preset_counts(
        args, "latency" if workload == "latency" else "throughput", payload_class
    )
    if queue_depth is not None:
        effective_queue_depth = queue_depth
    elif workload == "latency":
        effective_queue_depth = 2 if fanout > 1 else 1
    else:
        effective_queue_depth = 2
    command = [
        sys.executable,
        str(GRAPH_RUNNER),
        str(binary),
        "--output-dir",
        str(output),
        "--workload",
        "latency" if workload == "latency" else "saturated",
        "--scenario",
        scenario,
        "--samples",
        str(samples),
        "--warmup",
        str(warmup),
        "--repetitions",
        str(args.repetitions),
        "--queue-depth",
        str(effective_queue_depth),
        "--notification-capacity",
        str(1 if workload == "latency" else effective_queue_depth),
        "--publisher-period-us",
        str(args.latency_period_us if workload == "latency" else 0),
        "--payload-validation",
        "all" if args.preset == "smoke" else "none",
    ]
    if args.preset in {"production", DOCS_CORE_RUNTIME_PRESET}:
        command.extend(
            [
                "--session-timeout-ms",
                str(PRODUCTION_GRAPH_INTRAPROCESS_SESSION_TIMEOUT_MS),
                "--process-timeout-sec",
                str(PRODUCTION_GRAPH_INTRAPROCESS_PROCESS_TIMEOUT_SEC),
            ]
        )
    if args.cpus is not None:
        compact_docs_1to1 = (
            args.preset == DOCS_CORE_RUNTIME_PRESET
            and len(args.cpus) < 8
            and fanout == 1
            and scenario != "sample_forward"
        )
        if shared_fanout:
            controller_cpu_index = 1
        elif fanout > 1:
            controller_cpu_index = fanout + 1
        else:
            controller_cpu_index = 2 if compact_docs_1to1 else 3
        command.extend(
            [
                "--source-cpu",
                str(cpu(args, 0)),
                "--controller-cpu",
                str(cpu(args, controller_cpu_index)),
            ]
        )
        if workload == "throughput":
            if shared_fanout:
                notification_driver_cpu_index = 2
            elif fanout > 1:
                notification_driver_cpu_index = fanout + 3
            else:
                notification_driver_cpu_index = 3 if compact_docs_1to1 else 4
            command.extend(
                ["--notification-driver-cpu", str(cpu(args, notification_driver_cpu_index))]
            )
    if scenario == "sample_forward":
        command.extend(["--relay-count", str(relays)])
        if args.cpus is not None:
            command.extend(
                [
                    "--relay-cpu",
                    str(cpu(args, 2)),
                    "--sink-cpu",
                    str(cpu(args, 1)),
                ]
            )
    elif fanout > 1:
        command.extend(
            [
                "--fanout",
                str(fanout),
                "--operator-lanes",
                effective_fanout_operator_lanes,
            ]
        )
        if args.cpus is not None and not shared_fanout:
            sink_cpus = ",".join(str(cpu(args, index)) for index in range(1, fanout + 1))
            command.extend(
                [
                    "--sink-cpus",
                    sink_cpus,
                    "--channel-pump-cpu",
                    str(cpu(args, fanout + 2)),
                ]
            )
    elif scenario != "source_only" and args.cpus is not None:
        command.extend(["--sink-cpu", str(cpu(args, 1))])
    if scenario in {"tensor", "image", "multi_tensor"}:
        command.extend(
            [
                "--tensor-storage",
                storage,
                "--tensor-bytes",
                str(args.tensor_bytes),
                "--bulk-work-profile",
                bulk_work_profile,
            ]
        )
    if force_holomq:
        command.append("--force-holomq-process-local-inline")
        if args.cpus is not None:
            command.extend(["--channel-pump-cpu", str(cpu(args, 5))])
    if busy_poll:
        command.append("--busy-poll")
    return command


def add_graph_intraprocess_cases(
    cases: list[BenchmarkCase],
    args: argparse.Namespace,
    binaries: dict[str, Path],
    output_dir: Path,
) -> None:
    binary = binaries["graph_intraprocess"]
    scenarios = [
        ("pod", "pod", "pageable_host", False),
        ("pod_holomq", "pod", "pageable_host", True),
        ("signal", "signal", "pageable_host", False),
        ("schema", "schema", "pageable_host", False),
        ("tensor_pageable_host", "tensor", "pageable_host", False),
        ("tensor_pinned_host", "tensor", "pinned_host", False),
        ("image_pageable_host", "image", "pageable_host", False),
    ]
    if not args.skip_gpu:
        scenarios.append(("tensor_device", "tensor", "device", False))
    for name, scenario, storage, force_holomq in scenarios:
        for workload in ("throughput", "latency"):
            if name == "pod" and workload == "throughput":
                continue
            case_id = f"graph_intraprocess_{name}_{workload}"
            output = output_dir / f"graph/intraprocess/{case_id}"
            bulk_work_profile = (
                "full_write"
                if scenario == "tensor" and workload == "latency" and args.preset == "smoke"
                else "descriptor_only"
            )
            cases.append(
                BenchmarkCase(
                    case_id,
                    "graph",
                    "intraprocess",
                    workload,
                    "graph_intraprocess",
                    output / "summary.json",
                    graph_intraprocess_command(
                        args,
                        binary,
                        output,
                        workload,
                        scenario,
                        storage=storage,
                        bulk_work_profile=bulk_work_profile,
                        force_holomq=force_holomq,
                    ),
                    optional=storage == "device",
                )
            )
    for queue_depth in THROUGHPUT_QUEUE_DEPTHS:
        case_id = f"graph_intraprocess_pod_throughput_q{queue_depth}"
        output = output_dir / f"graph/intraprocess/{case_id}"
        cases.append(
            BenchmarkCase(
                case_id,
                "graph",
                "intraprocess",
                "throughput",
                "graph_intraprocess",
                output / "summary.json",
                graph_intraprocess_command(
                    args,
                    binary,
                    output,
                    "throughput",
                    "pod",
                    queue_depth=queue_depth,
                ),
            )
        )
    for scenario in ("pod", "schema"):
        for workload in ("throughput", "latency"):
            case_id = f"graph_intraprocess_{scenario}_1to2_{workload}"
            output = output_dir / f"graph/intraprocess/{case_id}"
            cases.append(
                BenchmarkCase(
                    case_id,
                    "graph",
                    "intraprocess",
                    workload,
                    "graph_intraprocess",
                    output / "summary.json",
                    graph_intraprocess_command(args, binary, output, workload, scenario, fanout=2),
                )
            )
    for relays in (1, 2, 4, 8):
        for workload in ("throughput", "latency"):
            case_id = f"graph_intraprocess_chain_{relays}_{workload}"
            output = output_dir / f"graph/intraprocess/{case_id}"
            cases.append(
                BenchmarkCase(
                    case_id,
                    "graph",
                    "intraprocess",
                    workload,
                    "graph_intraprocess",
                    output / "summary.json",
                    graph_intraprocess_command(
                        args,
                        binary,
                        output,
                        workload,
                        "sample_forward",
                        relays=relays,
                    ),
                )
            )
    busy_poll_output = output_dir / "graph/intraprocess/graph_intraprocess_pod_busy_poll_latency"
    cases.append(
        BenchmarkCase(
            "graph_intraprocess_pod_busy_poll_latency",
            "graph",
            "intraprocess",
            "latency",
            "graph_intraprocess",
            busy_poll_output / "summary.json",
            graph_intraprocess_command(
                args, binary, busy_poll_output, "latency", "pod", busy_poll=True
            ),
        )
    )


def graph_interprocess_command(
    args: argparse.Namespace,
    binary: Path,
    output: Path,
    topology: str,
    workload: str,
    fanout: int,
    *,
    payload: str = "pod",
    storage: str = "pageable_host",
    bulk_work_profile: str = "full_write",
    busy_poll: bool = False,
    queue_depth: int | None = None,
) -> list[str]:
    tensor_backed = payload in GRAPH_INTERPROCESS_TENSOR_PAYLOADS
    payload_class = (
        "bulk_device"
        if tensor_backed and storage == "device"
        else ("bulk_host" if tensor_backed else "inline")
    )
    samples, warmup = preset_counts(
        args, "latency" if workload == "latency" else "throughput", payload_class
    )
    if args.preset == DOCS_CORE_RUNTIME_PRESET and workload == "latency":
        exceptions = DOCS_CORE_RUNTIME_MEASUREMENT_POLICY["exceptions"]
        samples = int(exceptions["interprocess_latency_samples"])
        warmup = int(exceptions["interprocess_latency_warmup_samples"])
    latency_period_us = args.latency_period_us
    if topology == "fanout" and fanout > 1:
        latency_period_us = max(latency_period_us, 5_000)
        if workload == "latency" and args.preset == "balanced":
            exceptions = BALANCED_MEASUREMENT_POLICY["exceptions"]
            samples = int(exceptions["distributed_graph_1to2_latency_samples"])
            warmup = int(exceptions["distributed_graph_1to2_warmup_samples"])
    session_timeout_ms = DEFAULT_GRAPH_SESSION_TIMEOUT_MS
    process_timeout_sec = DEFAULT_GRAPH_PROCESS_TIMEOUT_SEC
    if workload == "latency":
        nominal_duration_ms = ((samples + warmup) * latency_period_us + 999) // 1_000
        session_timeout_ms = max(session_timeout_ms, 2 * nominal_duration_ms)
        process_timeout_sec = max(
            process_timeout_sec,
            (3 * session_timeout_ms + 999) // 1_000,
        )
    if queue_depth is not None:
        effective_queue_depth = queue_depth
    elif workload == "latency":
        # A paced multi-recipient row still needs one slot beyond the sample being consumed.
        # Otherwise a transiently delayed sleeping sink can make a two-recipient send cross the
        # post-submit visibility boundary, where retrying would risk duplicating the sample at the
        # other sink. This matches the intraprocess fanout latency geometry above.
        effective_queue_depth = 2 if topology == "fanout" and fanout > 1 else 1
    else:
        effective_queue_depth = 2
    command = [
        sys.executable,
        str(GRAPH_INTERPROCESS_RUNNER),
        str(binary),
        "--output-dir",
        str(output),
        "--topology",
        topology,
        "--workload",
        "latency" if workload == "latency" else "saturated",
        "--fanout",
        str(fanout),
        "--payload",
        payload,
        "--samples",
        str(samples),
        "--warmup",
        str(warmup),
        "--repetitions",
        str(args.repetitions),
        "--queue-depth",
        str(effective_queue_depth),
        "--publisher-period-us",
        str(latency_period_us if workload == "latency" else 0),
        "--payload-validation",
        "all" if args.preset == "smoke" else "none",
    ]
    if workload == "latency":
        command.extend(
            [
                "--session-timeout-ms",
                str(session_timeout_ms),
                "--process-timeout-sec",
                str(process_timeout_sec),
            ]
        )
    if tensor_backed:
        command.extend(
            [
                "--tensor-storage",
                storage,
                "--tensor-bytes",
                str(args.tensor_bytes),
                "--bulk-work-profile",
                bulk_work_profile,
            ]
        )
    if busy_poll:
        command.append("--busy-poll")
    if args.cpus is None:
        return command
    if topology == "chain":
        command.extend(
            [
                "--producer-cpu",
                str(cpu(args, 0)),
                "--relay-cpu",
                str(cpu(args, 1)),
                "--consumer-cpu",
                str(cpu(args, 2)),
                "--producer-controller-cpu",
                str(cpu(args, 3)),
                "--relay-controller-cpu",
                str(cpu(args, 4)),
                "--relay-pump-cpu",
                str(cpu(args, 5)),
                "--consumer-controller-cpu",
                str(cpu(args, 6)),
                "--consumer-pump-cpu",
                str(cpu(args, 7)),
            ]
        )
    elif fanout == 2:
        command.extend(
            [
                "--producer-cpu",
                str(cpu(args, 0)),
                "--consumer-cpus",
                f"{cpu(args, 1)},{cpu(args, 2)}",
                "--producer-controller-cpu",
                str(cpu(args, 3)),
                "--consumer-controller-cpus",
                f"{cpu(args, 4)},{cpu(args, 5)}",
                "--consumer-pump-cpus",
                f"{cpu(args, 6)},{cpu(args, 7)}",
            ]
        )
    else:
        command.extend(
            [
                "--producer-cpu",
                str(cpu(args, 0)),
                "--consumer-cpu",
                str(cpu(args, 1)),
                "--producer-controller-cpu",
                str(cpu(args, 2)),
                "--consumer-controller-cpu",
                str(cpu(args, 3)),
                "--consumer-pump-cpu",
                str(cpu(args, 4)),
            ]
        )
    return command


def docs_benchmark_case(
    args: argparse.Namespace,
    *,
    case_id: str,
    domain: str,
    workload: str,
    result_kind: str,
    output: Path,
    required_cpus: int,
    command_factory: Callable[[], list[str]],
) -> BenchmarkCase:
    """Build a docs case or record a topology-based omission before indexing CPU roles."""
    selected_cpus = len(args.cpus) if args.cpus is not None else 0
    supported = selected_cpus >= required_cpus
    return BenchmarkCase(
        case_id=case_id,
        layer="graph",
        domain=domain,
        workload=workload,
        result_kind=result_kind,
        output=output,
        command=command_factory() if supported else [],
        status="planned" if supported else "skipped",
        returncode=None if supported else 77,
        required_cpus=required_cpus,
        skip_reason=(
            None
            if supported
            else (
                f"topology requires {required_cpus} distinct physical CPUs; "
                f"only {selected_cpus} selected"
            )
        ),
    )


def add_docs_core_runtime_cases(
    cases: list[BenchmarkCase],
    args: argparse.Namespace,
    binaries: dict[str, Path],
    output_dir: Path,
) -> None:
    """Add only the production Graph rows rendered by benchmarks_core_runtime.mdx.

    Single-process fanout stops at 1:4 and covers both one shared operator partition and one
    partition per subscriber. Interprocess fanout stops at 1:2 so every process and support-thread
    role fits on the preset's eight distinct CPUs.
    """
    intraprocess_binary = binaries["graph_intraprocess"]
    intraprocess_definitions = (
        ("pod_1to1", "pod", "pageable_host", "full_write", 1, 1, None),
        ("schema_1to1", "schema", "pageable_host", "full_write", 1, 1, None),
        ("tensor_pinned_host_1to1", "tensor", "pinned_host", "descriptor_only", 1, 1, None),
        ("tensor_device_1to1", "tensor", "device", "descriptor_only", 1, 1, None),
        (
            "multi_tensor_pageable_host_1to1",
            "multi_tensor",
            "pageable_host",
            "descriptor_only",
            1,
            1,
            None,
        ),
        (
            "image_pageable_host_1to1",
            "image",
            "pageable_host",
            "descriptor_only",
            1,
            1,
            None,
        ),
        ("pod_1to2", "pod", "pageable_host", "full_write", 2, 1, "shared"),
        ("pod_1to2_per_sink", "pod", "pageable_host", "full_write", 2, 1, "per_sink"),
        ("pod_1to4", "pod", "pageable_host", "full_write", 4, 1, "shared"),
        ("pod_1to4_per_sink", "pod", "pageable_host", "full_write", 4, 1, "per_sink"),
        ("chain_10_operators", "sample_forward", "pageable_host", "full_write", 1, 8, None),
    )
    for (
        name,
        scenario,
        storage,
        bulk_work_profile,
        fanout,
        relays,
        fanout_operator_lanes,
    ) in intraprocess_definitions:
        for workload in ("latency", "throughput"):
            case_id = f"graph_intraprocess_docs_{name}_{workload}"
            output = output_dir / f"graph/intraprocess/{case_id}"
            cases.append(
                docs_benchmark_case(
                    args,
                    case_id=case_id,
                    domain="intraprocess",
                    workload=workload,
                    result_kind="graph_intraprocess",
                    output=output / "summary.json",
                    required_cpus=docs_intraprocess_required_cpus(
                        workload, scenario, fanout, fanout_operator_lanes
                    ),
                    command_factory=partial(
                        graph_intraprocess_command,
                        args,
                        intraprocess_binary,
                        output,
                        workload,
                        scenario,
                        storage=storage,
                        bulk_work_profile=bulk_work_profile,
                        fanout=fanout,
                        relays=relays,
                        fanout_operator_lanes=fanout_operator_lanes,
                    ),
                )
            )

    interprocess_binary = binaries["graph_interprocess"]
    interprocess_definitions = (
        ("pod_1to1", "fanout", 1, "pod", "pageable_host"),
        ("schema_1to1", "fanout", 1, "schema", "pageable_host"),
        ("tensor_pinned_host_1to1", "fanout", 1, "tensor", "pinned_host"),
        ("tensor_device_1to1", "fanout", 1, "tensor", "device"),
        ("image_pageable_host_1to1", "fanout", 1, "image", "pageable_host"),
        ("image_pinned_host_1to1", "fanout", 1, "image", "pinned_host"),
        ("image_device_1to1", "fanout", 1, "image", "device"),
        (
            "multi_tensor_loan_pageable_host_1to1",
            "fanout",
            1,
            "multi_tensor_loan",
            "pageable_host",
        ),
        ("pod_1to2", "fanout", 2, "pod", "pageable_host"),
        ("chain_3_operators", "chain", 1, "pod", "pageable_host"),
    )
    for name, topology, fanout, payload, storage in interprocess_definitions:
        for workload in ("latency", "throughput"):
            case_id = f"graph_interprocess_docs_{name}_{workload}"
            output = output_dir / f"graph/interprocess/{case_id}"
            cases.append(
                docs_benchmark_case(
                    args,
                    case_id=case_id,
                    domain="interprocess",
                    workload=workload,
                    result_kind="graph_interprocess",
                    output=output / "summary.json",
                    required_cpus=docs_interprocess_required_cpus(topology, fanout),
                    command_factory=partial(
                        graph_interprocess_command,
                        args,
                        interprocess_binary,
                        output,
                        topology,
                        workload,
                        fanout,
                        payload=payload,
                        storage=storage,
                        bulk_work_profile=(
                            "descriptor_only"
                            if payload in GRAPH_INTERPROCESS_TENSOR_PAYLOADS
                            else "full_write"
                        ),
                    ),
                )
            )


def add_graph_interprocess_cases(
    cases: list[BenchmarkCase],
    args: argparse.Namespace,
    binaries: dict[str, Path],
    output_dir: Path,
) -> None:
    binary = binaries["graph_interprocess"]
    definitions = (
        ("1to1", "fanout", "latency", 1),
        ("1to2", "fanout", "latency", 2),
        ("1to2", "fanout", "throughput", 2),
        ("chain", "chain", "throughput", 1),
        ("chain", "chain", "latency", 1),
    )
    for name, topology, workload, fanout in definitions:
        case_id = f"graph_interprocess_{name}_{workload}"
        output = output_dir / f"graph/interprocess/{case_id}"
        cases.append(
            BenchmarkCase(
                case_id,
                "graph",
                "interprocess",
                workload,
                "graph_interprocess",
                output / "summary.json",
                graph_interprocess_command(args, binary, output, topology, workload, fanout),
            )
        )
    for workload in ("throughput", "latency"):
        case_id = f"graph_interprocess_schema_1to1_{workload}"
        output = output_dir / f"graph/interprocess/{case_id}"
        cases.append(
            BenchmarkCase(
                case_id,
                "graph",
                "interprocess",
                workload,
                "graph_interprocess",
                output / "summary.json",
                graph_interprocess_command(
                    args,
                    binary,
                    output,
                    "fanout",
                    workload,
                    1,
                    payload="schema",
                ),
            )
        )
    for payload in ("tensor", "image"):
        for storage in ("pageable_host", "pinned_host", "device"):
            if storage == "device" and args.skip_gpu:
                continue
            for workload in ("throughput", "latency"):
                bulk_work_profile = (
                    "full_write"
                    if workload == "latency" and args.preset == "smoke"
                    else "descriptor_only"
                )
                case_id = f"graph_interprocess_{payload}_{storage}_1to1_{workload}"
                output = output_dir / f"graph/interprocess/{case_id}"
                cases.append(
                    BenchmarkCase(
                        case_id,
                        "graph",
                        "interprocess",
                        workload,
                        "graph_interprocess",
                        output / "summary.json",
                        graph_interprocess_command(
                            args,
                            binary,
                            output,
                            "fanout",
                            workload,
                            1,
                            payload=payload,
                            storage=storage,
                            bulk_work_profile=bulk_work_profile,
                        ),
                        optional=storage == "device",
                    )
                )
    for queue_depth in THROUGHPUT_QUEUE_DEPTHS:
        case_id = f"graph_interprocess_1to1_throughput_q{queue_depth}"
        output = output_dir / f"graph/interprocess/{case_id}"
        cases.append(
            BenchmarkCase(
                case_id,
                "graph",
                "interprocess",
                "throughput",
                "graph_interprocess",
                output / "summary.json",
                graph_interprocess_command(
                    args,
                    binary,
                    output,
                    "fanout",
                    "throughput",
                    1,
                    queue_depth=queue_depth,
                ),
            )
        )
    busy_poll_output = output_dir / "graph/interprocess/graph_interprocess_1to1_busy_poll_latency"
    cases.append(
        BenchmarkCase(
            "graph_interprocess_1to1_busy_poll_latency",
            "graph",
            "interprocess",
            "latency",
            "graph_interprocess",
            busy_poll_output / "summary.json",
            graph_interprocess_command(
                args,
                binary,
                busy_poll_output,
                "fanout",
                "latency",
                1,
                busy_poll=True,
            ),
        )
    )


def build_cases(args: argparse.Namespace) -> list[BenchmarkCase]:
    build_dir = args.build_dir.resolve()
    output_dir = args.output_dir.resolve()
    binaries = binary_paths(build_dir)
    cases: list[BenchmarkCase] = []
    if args.preset == DOCS_CORE_RUNTIME_PRESET:
        add_docs_core_runtime_cases(cases, args, binaries, output_dir)
    else:
        builders = {
            "holomq_intraprocess": add_holomq_intraprocess_cases,
            "holomq_interprocess": add_holomq_interprocess_cases,
            "channel": add_channel_cases,
            "graph_intraprocess": add_graph_intraprocess_cases,
            "graph_interprocess": add_graph_interprocess_cases,
        }
        for group in args.groups:
            builders[group](cases, args, binaries, output_dir)
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise RuntimeError("benchmark suite generated duplicate case identifiers")
    return cases


def write_manifest(
    args: argparse.Namespace,
    cases: list[BenchmarkCase],
    output_dir: Path,
    build_command: list[str] | None,
    summarizer_returncode: int | None = None,
) -> None:
    manifest = {
        "schema": SUITE_SCHEMA,
        "created_unix_seconds": args.created_unix_seconds,
        "preset": args.preset,
        "repetitions": args.repetitions,
        "latency_period_us": args.latency_period_us,
        "tensor_bytes": args.tensor_bytes,
        "measurement_policy": measurement_policy(args),
        "workload_geometry_policy": WORKLOAD_GEOMETRY_POLICY,
        "scenario_identity_contract": scenario_identity_contract(),
        "groups": args.groups,
        "cpus": args.cpus,
        "build_dir": str(args.build_dir.resolve()),
        "build_command": build_command,
        "result_summary": suite_result_summary(cases, summarizer_returncode),
        "cases": [case.manifest_record(output_dir) for case in cases],
    }
    temporary = output_dir / "suite_manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output_dir / "suite_manifest.json")


def suite_result_summary(
    cases: list[BenchmarkCase], summarizer_returncode: int | None = None
) -> dict[str, Any]:
    """Summarize progress and preserve actionable terminal-case evidence."""
    status_counts = {
        status: sum(case.status == status for case in cases)
        for status in ("planned", "passed", "failed", "skipped")
    }
    if status_counts["failed"] or summarizer_returncode not in (None, 0):
        outcome = "failed"
    elif status_counts["planned"] or summarizer_returncode is None:
        outcome = "incomplete"
    else:
        outcome = "passed"
    return {
        "outcome": outcome,
        "status_counts": status_counts,
        "summarizer_returncode": summarizer_returncode,
        "failed_cases": [
            {
                "case_id": case.case_id,
                "returncode": case.returncode,
                "stdout_log": f"logs/{case.case_id}.stdout.log",
                "stderr_log": f"logs/{case.case_id}.stderr.log",
            }
            for case in cases
            if case.status == "failed"
        ],
        "skipped_cases": [
            {
                "case_id": case.case_id,
                "returncode": case.returncode,
                "required_cpus": case.required_cpus,
                "reason": case.skip_reason,
            }
            for case in cases
            if case.status == "skipped"
        ],
    }


def failure_message(case: BenchmarkCase, output_dir: Path) -> str:
    stderr_path = output_dir / "logs" / f"{case.case_id}.stderr.log"
    return f"{case.case_id}: exit code {case.returncode}; stderr: {stderr_path}"


def restore_completed_cases(
    args: argparse.Namespace,
    cases: list[BenchmarkCase],
    manifest: dict[str, Any],
) -> None:
    expected_identity = {
        "schema": SUITE_SCHEMA,
        "preset": args.preset,
        "repetitions": args.repetitions,
        "latency_period_us": args.latency_period_us,
        "tensor_bytes": args.tensor_bytes,
        "groups": args.groups,
        "cpus": args.cpus,
        "build_dir": str(args.build_dir.resolve()),
    }
    for field, expected in expected_identity.items():
        if manifest.get(field) != expected:
            raise RuntimeError(
                f"--resume suite identity mismatch for {field}: "
                f"expected {expected!r}, found {manifest.get(field)!r}"
            )
    prior_records = manifest.get("cases")
    if not isinstance(prior_records, list):
        raise RuntimeError("--resume suite manifest is missing cases")
    prior_by_id = {
        record.get("case_id"): record for record in prior_records if isinstance(record, dict)
    }
    if set(prior_by_id) != {case.case_id for case in cases}:
        raise RuntimeError("--resume suite case inventory differs from the current plan")
    for case in cases:
        prior = prior_by_id[case.case_id]
        if prior.get("status") not in {"passed", "skipped"}:
            continue
        prior_command = prior.get("command")
        if not isinstance(prior_command, list) or not all(
            isinstance(argument, str) for argument in prior_command
        ):
            raise RuntimeError(f"--resume case {case.case_id} has an invalid recorded command")
        case.command = prior_command
        case.status = prior["status"]
        prior_returncode = prior.get("returncode")
        case.returncode = prior_returncode if isinstance(prior_returncode, int) else None


def append_command(path: Path, command: list[str]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(shlex.join(command))
        stream.write("\n")


def required_targets(groups: list[str]) -> list[str]:
    targets = {
        "holomq_intraprocess": "holomq_pubsub_benchmarks",
        "holomq_interprocess": "holomq_transport_benchmark",
        "channel": "channel_interprocess_benchmark",
        "graph_intraprocess": "runtime_graph_benchmark",
        "graph_interprocess": "runtime_graph_interprocess_benchmark",
    }
    return [targets[group] for group in groups]


def case_group(case: BenchmarkCase) -> str:
    if case.layer == "channel":
        return "channel"
    if case.layer not in {"holomq", "graph"}:
        raise ValueError(f"unsupported benchmark layer: {case.layer}")
    if case.domain not in {"intraprocess", "interprocess"}:
        raise ValueError(f"unsupported benchmark domain: {case.domain}")
    return f"{case.layer}_{case.domain}"


def pending_case_groups(cases: list[BenchmarkCase]) -> set[str]:
    return {case_group(case) for case in cases if case.status not in {"passed", "skipped"}}


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    manifest_path = output_dir / "suite_manifest.json"
    prior_manifest = None
    if args.resume:
        if not manifest_path.is_file():
            raise RuntimeError(f"--resume requires an existing suite manifest: {manifest_path}")
        prior_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        created = prior_manifest.get("created_unix_seconds")
        if not isinstance(created, (int, float)):
            raise RuntimeError("--resume suite manifest is missing its creation time")
        args.created_unix_seconds = created
    else:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise RuntimeError(f"--output-dir must be absent or empty: {output_dir}")
        args.created_unix_seconds = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(exist_ok=args.resume)
    commands_path = output_dir / ("resume_commands.sh" if args.resume else "commands.sh")
    commands_path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n\n", encoding="utf-8")

    cases = build_cases(args)
    if prior_manifest is not None:
        restore_completed_cases(args, cases, prior_manifest)
    build_command = None
    if args.build:
        build_command = [
            "cmake",
            "--build",
            str(args.build_dir.resolve()),
            "--target",
            *required_targets(args.groups),
        ]
        append_command(commands_path, build_command)
    pending_cases = [case for case in cases if case.status not in {"passed", "skipped"}]
    for case in pending_cases:
        append_command(commands_path, case.command)
    write_manifest(args, cases, output_dir, build_command)
    omitted_cases = [case for case in cases if case.status == "skipped" and case.skip_reason]
    if omitted_cases:
        print(
            f"omitting {len(omitted_cases)} of {len(cases)} docs rows that exceed "
            f"the {len(args.cpus)} selected physical CPUs",
            flush=True,
        )
    if args.dry_run:
        print(f"planned {len(pending_cases)} cases in {output_dir}")
        return 0

    if build_command is not None:
        subprocess.run(build_command, check=True)
    binaries = binary_paths(args.build_dir.resolve())
    pending_groups = pending_case_groups(cases)
    for group in pending_groups:
        binary = binaries[group]
        if not binary.is_file():
            raise FileNotFoundError(f"required benchmark binary does not exist: {binary}")

    for index, case in enumerate(cases, start=1):
        if case.status in {"passed", "skipped"}:
            continue
        print(f"[{index}/{len(cases)}] {case.case_id}", flush=True)
        case.output.parent.mkdir(parents=True, exist_ok=True)
        stdout_path = logs_dir / f"{case.case_id}.stdout.log"
        stderr_path = logs_dir / f"{case.case_id}.stderr.log"
        with (
            stdout_path.open("w", encoding="utf-8") as stdout,
            stderr_path.open("w", encoding="utf-8") as stderr,
        ):
            completed = subprocess.run(case.command, stdout=stdout, stderr=stderr, text=True)
        case.returncode = completed.returncode
        if completed.returncode == 0:
            case.status = "passed"
        elif completed.returncode == 77 and case.optional:
            case.status = "skipped"
        else:
            case.status = "failed"
        write_manifest(args, cases, output_dir, build_command)
        if case.status == "failed" and not args.keep_going:
            raise RuntimeError(failure_message(case, output_dir))
        if case.status == "failed":
            print(f"FAILED: {failure_message(case, output_dir)}", file=sys.stderr, flush=True)

    summary_command = [sys.executable, str(SUMMARIZER), str(output_dir)]
    append_command(commands_path, summary_command)
    completed = subprocess.run(summary_command, check=False)
    write_manifest(args, cases, output_dir, build_command, completed.returncode)
    failed = [case for case in cases if case.status == "failed"]
    if failed:
        print(f"suite completed with {len(failed)} failed case(s):", file=sys.stderr)
        for case in failed:
            print(f"  - {failure_message(case, output_dir)}", file=sys.stderr)
    if completed.returncode != 0:
        print(
            f"suite summarizer failed with exit code {completed.returncode}",
            file=sys.stderr,
        )
        return completed.returncode
    if failed:
        return 1
    print(f"suite results: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
