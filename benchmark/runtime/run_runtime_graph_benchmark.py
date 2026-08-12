#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run repeated production-Graph benchmarks and preserve measurement metadata."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from runtime_graph_runner_common import parse_integer_list
from runtime_graph_runner_common import summarize_values as summarize_common_values

RESULT_SCHEMA = "holoscan.ea1.runtime-benchmark/v1"
SUMMARY_SCHEMA = "holoscan.ea1.runtime-benchmark-summary/v1"
MANIFEST_SCHEMA = "holoscan.ea1.runtime-benchmark-manifest/v1"
PARTITION_ENVIRONMENT = "HOLOSCAN_EXPERIMENTAL_PARTITION_ALGORITHM"
PARTITION_ALGORITHMS = {
    "core_default": ("holoscan.partition.serial_baseline", "core_default"),
}
BENCHMARK_NAMES = {
    "source_only": "runtime_graph_source_activation",
    "pod": "runtime_graph_pod_1to1",
    "signal": "runtime_graph_signal_1to1",
    "sample_forward": "runtime_graph_pod_chain_1to1",
    "schema": "runtime_graph_schema_time_1to1",
    "tensor": "runtime_graph_tensor_1to1",
    "image": "runtime_graph_image_1to1",
    "multi_tensor": "runtime_graph_multi_tensor_1to1",
}


def direct_retained_scenario(scenario: str) -> bool:
    return scenario in {
        "pod",
        "signal",
        "sample_forward",
        "schema",
        "tensor",
        "image",
        "multi_tensor",
    }


def fanout_scenario(scenario: str) -> bool:
    return scenario in {"pod", "schema"}


def lossy_queue_policy(queue_policy: str) -> bool:
    return queue_policy == "drop_oldest"


def lossy_scenario_supported(queue_policy: str, scenario: str) -> bool:
    return queue_policy == "drop_oldest" and scenario in {"pod", "schema", "tensor", "image"}


def uses_channel_pump(args: argparse.Namespace) -> bool:
    return direct_retained_scenario(args.scenario) and (
        args.force_holomq_process_local_inline or args.fanout > 1
    )


def process_local_delivery(args: argparse.Namespace) -> str:
    if not direct_retained_scenario(args.scenario):
        return "not_applicable"
    return "holomq" if uses_channel_pump(args) else "direct_retained"


def benchmark_name(
    scenario: str,
    workload: str,
    force_holomq: bool = False,
    relay_count: int = 1,
    fanout: int = 1,
    queue_policy: str = "blocking",
) -> str:
    name = BENCHMARK_NAMES[scenario]
    if fanout_scenario(scenario):
        name = name.removesuffix("1to1") + f"1to{fanout}"
    if scenario == "sample_forward":
        name += f"_relays_{relay_count}"
    if force_holomq:
        name += "_holomq_fallback"
    if queue_policy != "blocking":
        name += f"_{queue_policy}"
    return f"{name}_{workload}" if workload != "latency" else name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark_binary", type=Path)
    parser.add_argument(
        "--workload",
        choices=("latency", "notification_latency", "saturated"),
        default="latency",
    )
    parser.add_argument("--scenario", choices=tuple(BENCHMARK_NAMES), default="pod")
    parser.add_argument(
        "--operator-lanes", choices=("separate", "shared", "per_sink"), default="separate"
    )
    parser.add_argument(
        "--partition-algorithm",
        choices=tuple(PARTITION_ALGORITHMS),
        default="core_default",
    )
    parser.add_argument("--samples", type=int, default=100_000)
    parser.add_argument("--warmup", type=int, default=1_000)
    parser.add_argument(
        "--queue-policy",
        choices=("blocking", "drop_oldest"),
        default="blocking",
    )
    parser.add_argument("--queue-depth", type=int, default=1)
    parser.add_argument("--relay-count", type=int, default=1)
    parser.add_argument("--fanout", type=int, default=1)
    parser.add_argument("--notification-capacity", type=int, default=1)
    parser.add_argument("--publisher-period-us", type=int, default=1_000)
    parser.add_argument("--payload-validation", choices=("none", "final", "all"), default="final")
    parser.add_argument(
        "--tensor-storage",
        choices=("pageable_host", "pinned_host", "device"),
        default="pageable_host",
    )
    parser.add_argument("--tensor-bytes", type=int, default=1 << 20)
    parser.add_argument("--tensor-device", type=int, default=0)
    parser.add_argument(
        "--bulk-work-profile",
        choices=("full_write", "descriptor_only"),
        default="full_write",
    )
    parser.add_argument("--source-cpu", type=int, default=-1)
    parser.add_argument("--relay-cpu", type=int, default=-1)
    parser.add_argument("--sink-cpu", type=int, default=-1)
    parser.add_argument("--sink-cpus", type=parse_cpu_list, default=[])
    parser.add_argument("--controller-cpu", type=int, default=-1)
    parser.add_argument("--channel-pump-cpu", type=int, default=-1)
    parser.add_argument("--notification-driver-cpu", type=int, default=-1)
    parser.add_argument("--busy-poll", action="store_true")
    parser.add_argument("--runtime-phase-counters", action="store_true")
    parser.add_argument("--disable-direct-continuation-elision", action="store_true")
    parser.add_argument("--disable-closed-island-composite", action="store_true")
    parser.add_argument("--force-holomq-process-local-inline", action="store_true")
    parser.add_argument("--session-timeout-ms", type=int, default=30_000)
    parser.add_argument("--process-timeout-sec", type=float, default=60.0)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--max-median-cv", type=float, default=0.10)
    parser.add_argument("--git-revision", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.samples <= 0:
        raise ValueError("--samples must be positive")
    if args.warmup < 0:
        raise ValueError("--warmup must be nonnegative")
    if args.workload in {"latency", "notification_latency"} and args.queue_depth not in {1, 2}:
        raise ValueError("latency workloads require --queue-depth 1 or 2")
    if args.workload == "saturated" and args.queue_depth <= 0:
        raise ValueError("saturated throughput requires a positive --queue-depth")
    if lossy_queue_policy(args.queue_policy) and not lossy_scenario_supported(
        args.queue_policy, args.scenario
    ):
        raise ValueError("--queue-policy drop_oldest supports pod, schema, tensor, or image")
    if lossy_queue_policy(args.queue_policy) and args.fanout != 1:
        raise ValueError("--queue-policy drop_oldest currently requires --fanout 1")
    if args.relay_count <= 0 or args.relay_count > 64:
        raise ValueError("--relay-count must be between 1 and 64")
    if args.scenario != "sample_forward" and args.relay_count != 1:
        raise ValueError("--relay-count requires --scenario sample_forward")
    if args.fanout <= 0 or args.fanout > 8:
        raise ValueError("--fanout must be between 1 and 8")
    if not fanout_scenario(args.scenario) and args.fanout != 1:
        raise ValueError("--fanout greater than one requires --scenario pod or schema")
    if args.notification_capacity <= 0:
        raise ValueError("--notification-capacity must be positive")
    if args.workload == "latency" and args.notification_capacity != 1:
        raise ValueError("--notification-capacity requires a notification workload")
    if args.workload == "notification_latency" and args.notification_capacity != 1:
        raise ValueError("notification latency requires --notification-capacity 1")
    if args.workload == "saturated" and args.notification_capacity > args.queue_depth:
        raise ValueError("--notification-capacity must not exceed --queue-depth")
    if args.workload in {"latency", "notification_latency"} and args.publisher_period_us <= 0:
        raise ValueError("--publisher-period-us must be positive")
    if args.tensor_bytes <= 0 or args.tensor_bytes > 1 << 30:
        raise ValueError("--tensor-bytes must be between 1 and 1073741824")
    if args.tensor_device < 0:
        raise ValueError("--tensor-device must be nonnegative")
    if (
        args.source_cpu < -1
        or args.relay_cpu < -1
        or args.sink_cpu < -1
        or args.controller_cpu < -1
        or args.channel_pump_cpu < -1
        or args.notification_driver_cpu < -1
        or any(cpu < -1 for cpu in args.sink_cpus)
    ):
        raise ValueError("CPU values must be -1 or a nonnegative ordinal")
    if args.scenario != "sample_forward" and args.relay_cpu >= 0:
        raise ValueError("--relay-cpu requires --scenario sample_forward")
    if args.partition_algorithm != "core_default":
        raise ValueError(
            "--partition-algorithm must remain core_default for benchmark-authored partitions"
        )
    if args.scenario == "source_only" and args.sink_cpu >= 0:
        raise ValueError("--sink-cpu is not used by --scenario source_only")
    if not uses_channel_pump(args) and args.channel_pump_cpu >= 0:
        raise ValueError("--channel-pump-cpu requires a transport-backed scenario")
    if args.force_holomq_process_local_inline and not direct_retained_scenario(args.scenario):
        raise ValueError(
            "--force-holomq-process-local-inline requires pod, signal, sample_forward, schema, "
            "tensor, image, or multi_tensor"
        )
    if args.force_holomq_process_local_inline and args.fanout != 1:
        raise ValueError("--force-holomq-process-local-inline requires --fanout 1")
    if args.disable_direct_continuation_elision and args.force_holomq_process_local_inline:
        raise ValueError(
            "--disable-direct-continuation-elision and --force-holomq-process-local-inline "
            "are mutually exclusive"
        )
    if args.scenario in {"image", "multi_tensor"} and args.tensor_storage != "pageable_host":
        raise ValueError(
            "--scenario image and multi_tensor currently require --tensor-storage pageable_host"
        )
    if args.scenario in {"image", "multi_tensor"} and args.tensor_device != 0:
        raise ValueError("--scenario image and multi_tensor currently require --tensor-device 0")
    if args.bulk_work_profile == "descriptor_only" and args.scenario not in {
        "tensor",
        "multi_tensor",
    }:
        raise ValueError(
            "--bulk-work-profile descriptor_only requires --scenario tensor or multi_tensor"
        )
    if args.operator_lanes == "shared" and args.sink_cpu >= 0:
        raise ValueError("--sink-cpu is not used with --operator-lanes shared")
    if args.operator_lanes == "shared" and args.relay_cpu >= 0:
        raise ValueError("--relay-cpu is not used with --operator-lanes shared")
    if args.operator_lanes == "per_sink" and (
        not fanout_scenario(args.scenario) or args.fanout <= 1
    ):
        raise ValueError("--operator-lanes per_sink requires pod or schema fanout greater than one")
    if args.operator_lanes == "per_sink" and args.sink_cpu >= 0:
        raise ValueError("--sink-cpu is not used with --operator-lanes per_sink")
    if args.operator_lanes == "separate" and args.fanout > 1 and args.sink_cpu >= 0:
        raise ValueError(
            "--sink-cpu with fanout greater than one requires "
            "--operator-lanes per_sink and --sink-cpus"
        )
    if args.operator_lanes != "per_sink" and args.sink_cpus:
        raise ValueError("--sink-cpus requires --operator-lanes per_sink")
    if args.sink_cpus and len(args.sink_cpus) != args.fanout:
        raise ValueError("--sink-cpus must provide exactly one CPU per fanout sink")
    if args.scenario == "source_only" and args.workload == "latency":
        raise ValueError("--scenario source_only requires a notification workload")
    if args.workload == "latency" and args.notification_driver_cpu >= 0:
        raise ValueError("--notification-driver-cpu requires a notification workload")
    assigned_cpus = [
        args.source_cpu,
        (
            -1
            if args.scenario == "source_only" or args.operator_lanes != "separate"
            else args.sink_cpu
        ),
        args.controller_cpu,
        args.channel_pump_cpu if uses_channel_pump(args) else -1,
    ]
    if args.scenario == "sample_forward" and args.operator_lanes == "separate":
        assigned_cpus.append(args.relay_cpu)
    assigned_cpus.extend(args.sink_cpus)
    if args.workload != "latency":
        assigned_cpus.append(args.notification_driver_cpu)
    pinned_cpus = [cpu for cpu in assigned_cpus if cpu >= 0]
    if len(pinned_cpus) != len(set(pinned_cpus)):
        raise ValueError("Runtime benchmark roles must not share one logical CPU")
    if args.session_timeout_ms <= 0 or args.process_timeout_sec <= 0.0:
        raise ValueError("benchmark timeouts must be positive")
    if args.repetitions <= 0:
        raise ValueError("--repetitions must be positive")
    if args.max_median_cv < 0.0:
        raise ValueError("--max-median-cv must be nonnegative")


def read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def read_cpu_topology(role: str, cpu: int) -> dict[str, object]:
    result: dict[str, object] = {"role": role, "cpu": cpu, "pinned": cpu >= 0}
    if cpu < 0:
        return result
    if not sys.platform.startswith("linux"):
        result.update({"topology_available": False, "governor": None})
        return result
    if hasattr(os, "sched_getaffinity") and cpu not in os.sched_getaffinity(0):
        raise RuntimeError(f"{role} CPU {cpu} is outside this process's allowed CPU affinity")

    cpu_path = Path(f"/sys/devices/system/cpu/cpu{cpu}")
    package_id = read_text(cpu_path / "topology/physical_package_id")
    core_id = read_text(cpu_path / "topology/core_id")
    try:
        result["package_id"] = int(package_id) if package_id is not None else None
        result["core_id"] = int(core_id) if core_id is not None else None
        result["topology_available"] = package_id is not None and core_id is not None
    except ValueError:
        result["topology_available"] = False
    result["thread_siblings"] = read_text(cpu_path / "topology/thread_siblings_list")
    result["core_type"] = read_text(cpu_path / "topology/core_type")
    result["governor"] = read_text(cpu_path / "cpufreq/scaling_governor")
    return result


def optional_cpu(cpu: int | None) -> int | None:
    return cpu if cpu is not None and cpu >= 0 else None


def is_nonnegative_integer(value: object) -> bool:
    """Return whether value is an integer count rather than bool or a negative sentinel."""
    return not isinstance(value, bool) and isinstance(value, int) and value >= 0


def complete_overlap_interval(latency: object, overlaps: object, expected_samples: int) -> bool:
    """Return whether timed intervals and valid concurrent overlaps form a complete set."""
    if not is_nonnegative_integer(overlaps):
        return False
    if latency is None:
        return overlaps == expected_samples
    samples = latency.get("samples") if isinstance(latency, dict) else None
    return (
        is_nonnegative_integer(samples)
        and isinstance(samples, int)
        and isinstance(overlaps, int)
        and samples + overlaps == expected_samples
    )


def expected_operator_lane_count(args: argparse.Namespace) -> int:
    if args.operator_lanes == "shared" or args.scenario == "source_only":
        return 1
    if args.scenario == "sample_forward":
        return 3
    if args.operator_lanes == "per_sink" or args.fanout > 1:
        return args.fanout + 1
    return 2


def expected_execution_lane_worker_count(args: argparse.Namespace) -> int:
    """Return the number of execution-lane worker threads in the accepted plan."""
    if args.operator_lanes == "shared" or args.scenario == "source_only":
        return 0
    return expected_operator_lane_count(args)


def parse_cpu_list(value: str) -> list[int]:
    return parse_integer_list(
        value,
        invalid_message="CPU list must contain integers",
        empty_message="CPU list must not contain empty entries",
    )


def collect_cpu_topology(args: argparse.Namespace) -> list[dict[str, object]]:
    if args.operator_lanes == "shared":
        placements = [read_cpu_topology("shared_operator_lane", args.source_cpu)]
    else:
        placements = [read_cpu_topology("source", args.source_cpu)]
        if args.scenario == "sample_forward":
            placements.append(read_cpu_topology("relay", args.relay_cpu))
        if args.operator_lanes == "per_sink":
            sink_cpus = args.sink_cpus or [-1] * args.fanout
            placements.extend(
                read_cpu_topology(f"sink_{index + 1}", cpu) for index, cpu in enumerate(sink_cpus)
            )
        elif args.fanout > 1:
            placements.extend(
                read_cpu_topology(f"sink_{index + 1}", -1) for index in range(args.fanout)
            )
        elif args.scenario != "source_only":
            placements.append(read_cpu_topology("sink", args.sink_cpu))
    placements.append(read_cpu_topology("controller", args.controller_cpu))
    if uses_channel_pump(args):
        placements.append(read_cpu_topology("channel_pump", args.channel_pump_cpu))
    if args.workload != "latency":
        placements.append(read_cpu_topology("notification_driver", args.notification_driver_cpu))
    pinned = [item for item in placements if item["pinned"]]
    physical_cores: set[tuple[object, object]] = set()
    for item in pinned:
        if not item.get("topology_available"):
            continue
        physical_core = (item.get("package_id"), item.get("core_id"))
        if physical_core in physical_cores:
            raise RuntimeError("Runtime benchmark CPUs share one physical core")
        physical_cores.add(physical_core)
    return placements


def find_cmake_cache(binary: Path) -> Path | None:
    for directory in (binary.parent, *binary.parents):
        candidate = directory / "CMakeCache.txt"
        if candidate.is_file():
            return candidate
    return None


def cmake_metadata(binary: Path) -> dict[str, object]:
    cache = find_cmake_cache(binary)
    if cache is None:
        return {"cache": None}
    requested = {
        "CMAKE_BUILD_TYPE",
        "CMAKE_CXX_COMPILER",
        "CMAKE_CUDA_COMPILER",
        "CMAKE_CUDA_COMPILER_VERSION",
    }
    values: dict[str, object] = {"cache": str(cache)}
    for line in cache.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("//") or "=" not in line or ":" not in line:
            continue
        key_and_type, value = line.split("=", 1)
        key = key_and_type.split(":", 1)[0]
        if key in requested:
            values[key.lower()] = value
    compiler = values.get("cmake_cxx_compiler")
    if isinstance(compiler, str):
        compiler_version = command_output([compiler, "--version"])
        values["cxx_compiler_version"] = (
            compiler_version.splitlines()[0] if compiler_version is not None else None
        )
    cuda_compiler = values.get("cmake_cuda_compiler")
    if isinstance(cuda_compiler, str):
        cuda_version = command_output([cuda_compiler, "--version"])
        values["cuda_compiler_version_output"] = (
            cuda_version.splitlines()[-1] if cuda_version is not None else None
        )
    return values


def command_output(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None


def discovered_git_revision() -> str | None:
    repository = Path(__file__).resolve().parents[3]
    return command_output(["git", "-C", str(repository), "rev-parse", "HEAD"])


def git_revision(explicit_revision: str | None) -> tuple[str | None, str]:
    if explicit_revision:
        return explicit_revision, "command_line"
    if environment_revision := os.environ.get("HOLOSCAN_GIT_REVISION"):
        return environment_revision, "environment"
    return discovered_git_revision(), "source_tree"


def gpu_metadata() -> list[dict[str, str]]:
    if shutil.which("nvidia-smi") is None:
        return []
    output = command_output(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,pci.bus_id",
            "--format=csv,noheader,nounits",
        ]
    )
    if output is None:
        return []
    result: list[dict[str, str]] = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 3:
            result.append(
                {
                    "name": fields[0],
                    "driver_version": fields[1],
                    "pci_bus_id": fields[2],
                }
            )
    return result


def validate_runtime_phase_counters(
    phase_counters: object, counts: dict[str, Any], args: argparse.Namespace
) -> None:
    if not args.runtime_phase_counters:
        if phase_counters not in (None, {"enabled": False}):
            raise RuntimeError("result unexpectedly enabled Runtime phase counters")
        return
    if not isinstance(phase_counters, dict) or phase_counters.get("enabled") is not True:
        raise RuntimeError("result is missing enabled Runtime phase counters")
    required = (
        "readiness_scans",
        "no_progress_scans",
        "wait_calls",
        "wake_signals",
        "wake_returns",
        "timeout_returns",
        "stop_signals",
        "candidates_frozen",
        "candidates_selected",
        "readiness_poll_dispatches",
        "direct_continuation_receipts_accepted",
        "direct_continuation_poll_elisions",
        "direct_composite_attempts",
        "direct_composite_completions",
        "direct_composite_fallbacks",
        "direct_composite_failures",
        "direct_composite_callbacks_attempted",
        "direct_composite_callbacks_completed",
        "compute_dispatches",
        "activations_completed",
        "channel_prepare_calls",
        "channel_emit_attempts",
        "channel_emit_accepted",
        "channel_emit_backpressured",
        "channel_samples_received",
        "channel_readiness_published",
        "channel_input_reads",
        "channel_witness_retired",
        "channel_retire_calls",
        "channel_activation_cuts",
        "channel_activation_cut_drains",
        "channel_activation_cut_skips",
        "channel_receive_backpressured",
        "lane_dispatches",
        "lane_compute_dispatches",
        "lane_dispatcher_dispatches",
        "lane_lifecycle_dispatches",
        "lane_unknown_dispatches",
        "lane_busy_task_publications",
        "lane_sleeping_wake_signals",
        "lane_callbacks_entered",
        "lane_callbacks_exited",
        "lane_completions_observed",
        "lane_serialization_contentions",
        "lane_inline_owner_invocations",
    )
    if any(not isinstance(phase_counters.get(name), int) for name in required):
        raise RuntimeError("result has incomplete Runtime phase counters")
    expected_compute = sum(
        counts.get(name, 0)
        for name in ("source_compute_entries", "relay_callbacks", "sink_callbacks")
    )
    frozen = phase_counters["candidates_frozen"]
    expected_emit_attempts = counts.get("source_callbacks", 0) + counts.get("relay_callbacks", 0)
    expected_accepted_emits = counts.get("publications", 0) + counts.get("relay_publications", 0)
    expected_input_operations = counts.get("relay_callbacks", 0) + counts.get("sink_callbacks", 0)
    channel_pump_used = uses_channel_pump(args)
    activation_cut_consistent = (
        phase_counters["channel_activation_cut_drains"]
        + phase_counters["channel_activation_cut_skips"]
        == frozen
        if channel_pump_used
        else phase_counters["channel_activation_cut_drains"] == 0
        and phase_counters["channel_activation_cut_skips"] == 0
    )
    expected_activation_cuts = frozen if channel_pump_used else 0
    channel_consistent = (
        phase_counters["channel_prepare_calls"] == frozen
        and phase_counters["channel_emit_attempts"] == expected_emit_attempts
        and phase_counters["channel_emit_accepted"] == expected_accepted_emits
        and phase_counters["channel_emit_backpressured"] == counts.get("backpressured_emits", 0)
        and phase_counters["channel_samples_received"] == expected_input_operations
        and phase_counters["channel_readiness_published"] == expected_input_operations
        and phase_counters["channel_input_reads"] == expected_input_operations
        and phase_counters["channel_witness_retired"] == expected_input_operations
        and phase_counters["channel_retire_calls"] == frozen
        and phase_counters["channel_activation_cuts"] == expected_activation_cuts
        and activation_cut_consistent
    )
    if args.scenario == "source_only":
        channel_consistent = (
            phase_counters["channel_prepare_calls"] == 0
            and phase_counters["channel_emit_attempts"] == 0
            and phase_counters["channel_samples_received"] == 0
        )
    composite_attempts = phase_counters["direct_composite_attempts"]
    composite_callbacks_attempted = phase_counters["direct_composite_callbacks_attempted"]
    composite_consistent = (
        composite_callbacks_attempted <= expected_compute
        and phase_counters["direct_composite_callbacks_completed"] <= composite_callbacks_attempted
        and composite_attempts <= composite_callbacks_attempted <= composite_attempts * 2
        and phase_counters["direct_composite_completions"]
        + phase_counters["direct_composite_fallbacks"]
        + phase_counters["direct_composite_failures"]
        <= composite_attempts
        and (
            not args.disable_closed_island_composite
            or all(
                phase_counters[name] == 0
                for name in (
                    "direct_composite_attempts",
                    "direct_composite_completions",
                    "direct_composite_fallbacks",
                    "direct_composite_failures",
                    "direct_composite_callbacks_attempted",
                    "direct_composite_callbacks_completed",
                )
            )
        )
    )
    lane_dispatches = phase_counters["lane_dispatches"]
    inline_owner_invocations = phase_counters["lane_inline_owner_invocations"]
    expected_lane_compute_calls = frozen - composite_callbacks_attempted
    expected_lane_dispatcher_calls = (
        phase_counters["readiness_poll_dispatches"] + composite_attempts
    )
    ordinary_lane_consistent = (
        inline_owner_invocations == 0
        and phase_counters["lane_compute_dispatches"] == expected_lane_compute_calls
        and phase_counters["lane_dispatcher_dispatches"] == expected_lane_dispatcher_calls
        and lane_dispatches
        == phase_counters["lane_compute_dispatches"]
        + phase_counters["lane_dispatcher_dispatches"]
        + phase_counters["lane_lifecycle_dispatches"]
        and phase_counters["lane_unknown_dispatches"] == 0
        and phase_counters["lane_callbacks_entered"] == lane_dispatches
        and phase_counters["lane_callbacks_exited"] == lane_dispatches
        and phase_counters["lane_completions_observed"] == lane_dispatches
        and (
            (
                args.busy_poll
                and phase_counters["lane_busy_task_publications"] == lane_dispatches
                and phase_counters["lane_sleeping_wake_signals"] == 0
            )
            or (
                not args.busy_poll
                and phase_counters["lane_sleeping_wake_signals"] == lane_dispatches
                and phase_counters["lane_busy_task_publications"] == 0
            )
        )
    )
    inline_lane_consistent = (
        inline_owner_invocations > 0
        and phase_counters["lane_compute_dispatches"] == 0
        and phase_counters["lane_dispatcher_dispatches"] == 0
        and lane_dispatches == phase_counters["lane_lifecycle_dispatches"]
        and phase_counters["lane_unknown_dispatches"] == 0
        and (
            (
                args.busy_poll
                and phase_counters["lane_busy_task_publications"] == lane_dispatches
                and phase_counters["lane_sleeping_wake_signals"] == 0
            )
            or (
                not args.busy_poll
                and phase_counters["lane_sleeping_wake_signals"] == lane_dispatches
                and phase_counters["lane_busy_task_publications"] == 0
            )
        )
        and phase_counters["lane_callbacks_entered"] == inline_owner_invocations + lane_dispatches
        and phase_counters["lane_callbacks_exited"] == inline_owner_invocations + lane_dispatches
        and phase_counters["lane_completions_observed"] == lane_dispatches
        and inline_owner_invocations >= expected_lane_compute_calls + expected_lane_dispatcher_calls
    )
    lane_consistent = ordinary_lane_consistent or inline_lane_consistent
    if (
        frozen != expected_compute
        or phase_counters["candidates_selected"] != frozen
        or phase_counters["compute_dispatches"] != frozen
        or phase_counters["activations_completed"] != frozen
        or phase_counters["direct_continuation_receipts_accepted"]
        != phase_counters["direct_continuation_poll_elisions"]
        or (
            args.disable_direct_continuation_elision
            and phase_counters["direct_continuation_receipts_accepted"] != 0
        )
        or phase_counters["wake_returns"] + phase_counters["timeout_returns"]
        != phase_counters["wait_calls"]
        or phase_counters["no_progress_scans"] > phase_counters["readiness_scans"]
        or not composite_consistent
        or not channel_consistent
        or not lane_consistent
    ):
        raise RuntimeError("result has inconsistent Runtime phase counters")


def validate_runtime_phase_trace(
    phase_trace: object, counts: dict[str, Any], args: argparse.Namespace
) -> None:
    if not args.runtime_phase_counters:
        if phase_trace not in (None, {"enabled": False}):
            raise RuntimeError("result unexpectedly enabled Runtime phase trace")
        return
    if not isinstance(phase_trace, dict) or phase_trace.get("enabled") is not True:
        raise RuntimeError("result is missing enabled Runtime phase trace")
    activation_trace_applicable = phase_trace.get("activation_trace_applicable", True)
    expected_activation_trace_applicable = (
        args.operator_lanes == "shared" or args.scenario == "source_only"
    )
    if not isinstance(activation_trace_applicable, bool) or (
        "activation_trace_applicable" in phase_trace
        and activation_trace_applicable is not expected_activation_trace_applicable
    ):
        raise RuntimeError("result has inconsistent Runtime phase-trace applicability")
    expected_activations = sum(
        counts.get(name, 0)
        for name in ("source_compute_entries", "relay_callbacks", "sink_callbacks")
    )
    receivers = phase_trace.get("receivers")
    scalar_fields = (
        "records",
        "dropped",
        "readiness_polls",
        "unmatched_readiness_polls",
        "complete_activations",
        "source_activation_runs",
        "maximum_consecutive_source_activations",
        "source_to_source_transitions",
        "source_to_downstream_transitions",
        "downstream_to_source_transitions",
        "downstream_to_downstream_transitions",
        "complete_tensor_emits",
        "complete_inline_payload_emits",
    )
    transition_fields = (
        "source_to_source_transitions",
        "source_to_downstream_transitions",
        "downstream_to_source_transitions",
        "downstream_to_downstream_transitions",
    )
    expected_tensor_emits = counts.get("publications", 0) if args.scenario == "tensor" else 0
    expected_tensor_transport_emits = (
        expected_tensor_emits if args.force_holomq_process_local_inline else 0
    )
    expected_inline_payload_emits = (
        counts.get("publications", 0) + counts.get("relay_publications", 0)
        if args.scenario not in ("source_only", "tensor", "image", "multi_tensor")
        else 0
    )
    expected_inline_transport_emits = (
        expected_inline_payload_emits
        if args.force_holomq_process_local_inline or args.fanout > 1
        else 0
    )
    if not activation_trace_applicable:
        if (
            phase_trace.get("valid") is not True
            or phase_trace.get("capacity_exhausted") is not False
            or phase_trace.get("storage_invariants_valid") is not True
            or not is_nonnegative_integer(phase_trace.get("capacity"))
            or not is_nonnegative_integer(phase_trace.get("attempted_records"))
            or not is_nonnegative_integer(phase_trace.get("records"))
            or phase_trace["records"] > phase_trace["capacity"]
            or phase_trace["attempted_records"] != phase_trace["records"]
            or phase_trace.get("dropped") != 0
            or phase_trace.get("unmatched_readiness_polls") != 0
            or phase_trace.get("complete_activations") != 0
            or phase_trace.get("receivers") != []
        ):
            raise RuntimeError("result has inconsistent bounded concurrent-partition trace")
        return
    if (
        phase_trace.get("valid") is not True
        or any(not isinstance(phase_trace.get(name), int) for name in scalar_fields)
        or phase_trace.get("dropped") != 0
        or phase_trace.get("unmatched_readiness_polls") != 0
        or phase_trace.get("complete_activations") != expected_activations
        or phase_trace.get("source_activation_runs", 0) <= 0
        or phase_trace.get("maximum_consecutive_source_activations", 0) <= 0
        or phase_trace.get("complete_tensor_emits") != expected_tensor_emits
        or phase_trace.get("complete_inline_payload_emits") != expected_inline_payload_emits
        or not isinstance(receivers, list)
        or not receivers
        or sum(
            receiver.get("activations", -1) if isinstance(receiver, dict) else -1
            for receiver in receivers
        )
        != expected_activations
    ):
        raise RuntimeError("result has inconsistent Runtime phase trace")
    if sum(phase_trace[name] for name in transition_fields) != max(expected_activations - 1, 0):
        raise RuntimeError("result has inconsistent Runtime phase trace")

    def validate_distribution(name: str, expected_samples: int, phase_kind: str) -> None:
        timing = phase_trace.get(name)
        if expected_samples == 0:
            if timing is not None:
                raise RuntimeError(f"result has incomplete Runtime {phase_kind} phase trace")
            return
        if not isinstance(timing, dict) or timing.get("samples") != expected_samples:
            raise RuntimeError(f"result has incomplete Runtime {phase_kind} phase trace")

    validate_distribution("tensor_emit_total_ns", expected_tensor_emits, "Tensor")
    validate_distribution("tensor_emit_validation_ns", expected_tensor_emits, "Tensor")
    validate_distribution("tensor_pre_transport_ns", expected_tensor_transport_emits, "Tensor")
    validate_distribution(
        "tensor_transport_publication_ns", expected_tensor_transport_emits, "Tensor"
    )
    validate_distribution("tensor_post_transport_ns", expected_tensor_transport_emits, "Tensor")
    validate_distribution(
        "inline_payload_emit_total_ns", expected_inline_payload_emits, "inline-payload"
    )
    validate_distribution(
        "inline_payload_encoding_ns", expected_inline_payload_emits, "inline-payload"
    )
    validate_distribution(
        "inline_payload_pre_transport_ns",
        expected_inline_transport_emits,
        "inline-payload",
    )
    validate_distribution(
        "inline_payload_transport_publication_ns",
        expected_inline_transport_emits,
        "inline-payload",
    )
    validate_distribution(
        "inline_payload_post_transport_ns",
        expected_inline_transport_emits,
        "inline-payload",
    )

    for receiver in receivers:
        if not isinstance(receiver, dict) or not isinstance(receiver.get("activations"), int):
            raise RuntimeError("result has incomplete Runtime phase receiver trace")
        activations = receiver["activations"]
        for name in (
            "candidate_to_dispatch_start_ns",
            "compute_dispatch_ns",
            "dispatch_return_to_retirement_ns",
            "retirement_to_cut_ns",
            "dispatch_return_to_cut_ns",
        ):
            timing = receiver.get(name)
            if not isinstance(timing, dict) or timing.get("samples") != activations:
                raise RuntimeError("result has incomplete Runtime phase receiver trace")
        readiness = receiver.get("readiness_poll_dispatch_source_fifo_proxy_ns")
        if readiness is not None and (
            not isinstance(readiness, dict)
            or not is_nonnegative_integer(readiness.get("samples"))
            or readiness["samples"] > activations
        ):
            raise RuntimeError("result has invalid candidate-attributed readiness-poll trace")


def validate_runtime_thread_usage(usage: object, args: argparse.Namespace) -> None:
    if not args.runtime_phase_counters:
        if usage not in (None, {"enabled": False}):
            raise RuntimeError("result unexpectedly enabled Runtime thread-usage evidence")
        return
    if not isinstance(usage, dict) or usage.get("enabled") is not True:
        raise RuntimeError("result is missing Runtime thread-usage evidence")
    supported = sys.platform.startswith("linux")
    if usage.get("supported") is not supported:
        raise RuntimeError("result has inconsistent Runtime thread-usage platform evidence")
    roles = usage.get("roles")
    aggregate = usage.get("aggregate")
    if (
        usage.get("scope") != "whole_runtime_thread_lifetime_including_startup_and_teardown"
        or not isinstance(roles, list)
        or len(roles) != 4
        or not isinstance(aggregate, dict)
        or aggregate.get("scope") != "sum_of_runtime_roles"
    ):
        raise RuntimeError("result has malformed Runtime thread-usage evidence")
    expected_samples = {
        "controller": 1,
        "execution_lanes": expected_execution_lane_worker_count(args),
        "channel_pump": 1 if uses_channel_pump(args) else 0,
        "notification_driver": 1 if args.workload != "latency" else 0,
    }
    fields = (
        "samples",
        "user_cpu_ns",
        "system_cpu_ns",
        "voluntary_context_switches",
        "involuntary_context_switches",
        "minor_page_faults",
        "major_page_faults",
    )
    by_role = {
        role.get("role"): role
        for role in roles
        if isinstance(role, dict) and isinstance(role.get("role"), str)
    }
    if set(by_role) != set(expected_samples):
        raise RuntimeError("result has incomplete Runtime thread-role evidence")
    for role_name, expected in expected_samples.items():
        role = by_role[role_name]
        if role.get("supported") is not supported or any(
            isinstance(role.get(field), bool)
            or not isinstance(role.get(field), int)
            or role[field] < 0
            for field in fields
        ):
            raise RuntimeError("result has invalid Runtime thread-role evidence")
        if role["samples"] != (expected if supported else 0):
            raise RuntimeError("result has unexpected Runtime thread-role population")
    for field in fields:
        value = aggregate.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            or value != sum(by_role[role][field] for role in expected_samples)
        ):
            raise RuntimeError("result has inconsistent aggregate Runtime thread usage")


def valid_thread_identity_evidence(
    identity_kind: object, native_tid_syscall_unavailable: object
) -> bool:
    """Return whether thread-identity provenance is valid on the current platform."""
    if not isinstance(native_tid_syscall_unavailable, bool):
        return False
    if sys.platform.startswith("linux"):
        return (identity_kind == "native_tid" and not native_tid_syscall_unavailable) or (
            identity_kind == "std_thread_id_hash" and native_tid_syscall_unavailable
        )
    return identity_kind == "std_thread_id_hash" and not native_tid_syscall_unavailable


def validate_callback_thread_evidence(
    evidence: object, counts: dict[str, Any], args: argparse.Namespace
) -> None:
    if not args.runtime_phase_counters:
        if evidence not in (None, {"enabled": False}):
            raise RuntimeError("result unexpectedly enabled callback-thread evidence")
        return
    if not isinstance(evidence, dict):
        raise RuntimeError("result is missing callback-thread evidence")
    controller_tid = evidence.get("controller_native_tid")
    roles = evidence.get("roles")
    callbacks_on_partition_owner = evidence.get("callbacks_on_partition_owner")
    if (
        evidence.get("enabled") is not True
        or evidence.get("valid") is not True
        or not valid_thread_identity_evidence(
            evidence.get("thread_identity_kind"),
            evidence.get("native_tid_syscall_unavailable"),
        )
        or not isinstance(controller_tid, int)
        or isinstance(controller_tid, bool)
        or controller_tid <= 0
        or not isinstance(callbacks_on_partition_owner, bool)
        or not isinstance(roles, list)
        or len(roles) != 3
    ):
        raise RuntimeError("result has invalid callback-thread evidence")
    expected_callbacks = {
        "source": counts.get("source_compute_entries", counts.get("source_callbacks")),
        "relay": counts.get("relay_callbacks"),
        "sink": counts.get("sink_callbacks"),
    }
    active_tids: set[int] = set()
    for expected_role, role in zip(("source", "relay", "sink"), roles, strict=True):
        if not isinstance(role, dict) or role.get("role") != expected_role:
            raise RuntimeError("result has invalid callback-thread evidence")
        callbacks = role.get("callbacks")
        native_tid = role.get("native_tid")
        multiple_threads = role.get("multiple_threads")
        if (
            not isinstance(callbacks, int)
            or isinstance(callbacks, bool)
            or callbacks < 0
            or callbacks != expected_callbacks[expected_role]
            or not isinstance(multiple_threads, bool)
            or (
                callbacks > 0
                and (
                    not isinstance(native_tid, int)
                    or isinstance(native_tid, bool)
                    or native_tid <= 0
                )
            )
            or (callbacks == 0 and native_tid is not None)
        ):
            raise RuntimeError("result has invalid callback-thread evidence")
        if callbacks > 0:
            active_tids.add(native_tid)
    if args.operator_lanes == "shared" and (
        evidence.get("shared_lane_single_thread") is not True
        or len(active_tids) != 1
        or (controller_tid in active_tids) is not callbacks_on_partition_owner
        or any(role["multiple_threads"] for role in roles)
    ):
        raise RuntimeError("result has inconsistent shared-lane callback-thread evidence")


def validate_callback_counts(
    path: Path,
    counts: dict[str, Any],
    args: argparse.Namespace,
    expected_total: int,
    expected_sink: int,
) -> None:
    backpressure = counts.get("backpressured_emits")
    source_backpressure = counts.get("source_backpressured_emits")
    expected_relay = expected_total * args.relay_count if args.scenario == "sample_forward" else 0
    backpressure_valid = (
        isinstance(backpressure, int) and not isinstance(backpressure, bool) and backpressure >= 0
    )
    source_backpressure_valid = (
        isinstance(source_backpressure, int)
        and not isinstance(source_backpressure, bool)
        and source_backpressure >= 0
        and backpressure_valid
        and source_backpressure <= backpressure
    )
    expected_source = expected_total + source_backpressure if source_backpressure_valid else None
    # Aggregate backpressure also includes relay retries, which do not re-run the source.
    if (
        not backpressure_valid
        or not source_backpressure_valid
        or counts.get("source_callbacks") != expected_source
        or counts.get("relay_callbacks") != expected_relay
        or counts.get("relay_publications") != expected_relay
        or counts.get("sink_callbacks") != expected_sink
    ):
        raise RuntimeError(
            f"{path} reported inconsistent callback counts: "
            f"source_callbacks={counts.get('source_callbacks')}, "
            f"expected_source_callbacks={expected_source}, "
            f"relay_callbacks={counts.get('relay_callbacks')}, "
            f"expected_relay_callbacks={expected_relay}, "
            f"relay_publications={counts.get('relay_publications')}, "
            f"expected_relay_publications={expected_relay}, "
            f"sink_callbacks={counts.get('sink_callbacks')}, "
            f"expected_sink_callbacks={expected_sink}, "
            f"backpressured_emits={backpressure}, "
            f"source_backpressured_emits={source_backpressure}"
        )


def checked_result(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid benchmark result {path}: {error}") from error
    if result.get("schema") != RESULT_SCHEMA:
        raise RuntimeError(f"{path} has an unsupported result schema")
    if result.get("benchmark") != benchmark_name(
        args.scenario,
        args.workload,
        args.force_holomq_process_local_inline,
        args.relay_count,
        args.fanout,
        args.queue_policy,
    ):
        raise RuntimeError(f"{path} reported an unexpected benchmark")
    if result.get("scenario") != args.scenario:
        raise RuntimeError(f"{path} reported an unexpected scenario")
    if result.get("workload") != args.workload:
        raise RuntimeError(f"{path} reported an unexpected workload")
    if result.get("process_local_inline_delivery") != process_local_delivery(args):
        raise RuntimeError(f"{path} reported an unexpected process-local delivery")
    expected_runtime_controls = {
        "direct_continuation_elision_disabled": args.disable_direct_continuation_elision,
        "closed_island_composite_disabled": args.disable_closed_island_composite,
    }
    if result.get("runtime_controls") != expected_runtime_controls:
        raise RuntimeError(f"{path} reported unexpected private Runtime controls")
    if result.get("queue_policy") != args.queue_policy:
        raise RuntimeError(f"{path} reported an unexpected queue policy")
    geometry = result.get("geometry")
    counts = result.get("counts")
    validation = result.get("validation")
    session = result.get("session")
    timing = result.get("timing")
    allocation_probe = result.get("allocation_probe")
    wake = result.get("wake")
    placement = result.get("placement")
    partitioning = result.get("partitioning")
    timestamp_instrumentation = result.get("timestamp_instrumentation")
    if not all(
        isinstance(item, dict)
        for item in (
            geometry,
            counts,
            validation,
            session,
            timing,
            allocation_probe,
            wake,
            placement,
            partitioning,
            timestamp_instrumentation,
        )
    ):
        raise RuntimeError(f"{path} is missing structured result sections")
    pair_overhead = timestamp_instrumentation.get("pair_overhead_ns")
    if (
        result.get("clock")
        != {
            "source": "std::chrono::steady_clock",
            "steady": True,
            "scope": "single_process",
        }
        or timestamp_instrumentation.get("calibration") != "post_measurement_back_to_back_pair"
        or timestamp_instrumentation.get("clock_reads_per_sample") != 2
        or not isinstance(pair_overhead, dict)
        or pair_overhead.get("samples") != 4_096
        or not isinstance(pair_overhead.get("p50"), int)
    ):
        raise RuntimeError(f"{path} reported invalid clock or timestamp calibration evidence")
    expected_wake = {
        "event_wake": uses_channel_pump(args),
        "busy_poll": args.busy_poll,
        "receive_polling": False,
    }
    if wake != expected_wake:
        raise RuntimeError(f"{path} reported unexpected wake or polling evidence")
    expected_allocation_limits = {
        "source_compute_allocation_limit": 0,
        "relay_compute_allocation_limit": 0,
        "sink_compute_allocation_limit": (
            6 if args.scenario == "image" else (4 if args.scenario == "multi_tensor" else 0)
        ),
    }
    allocations_valid = allocation_probe.get("excluded_from_measurement") is True
    for limit_name, expected_limit in expected_allocation_limits.items():
        count_name = limit_name.replace("_limit", "s")
        limit = allocation_probe.get(limit_name)
        count = allocation_probe.get(count_name)
        allocations_valid = (
            allocations_valid
            and limit == expected_limit
            and isinstance(count, int)
            and not isinstance(count, bool)
            and 0 <= count <= expected_limit
        )
    if not allocations_valid:
        raise RuntimeError(f"{path} reported compute-path allocations")
    validate_runtime_phase_counters(result.get("runtime_phase_counters"), counts, args)
    validate_runtime_phase_trace(result.get("runtime_phase_trace"), counts, args)
    validate_runtime_thread_usage(result.get("runtime_thread_usage"), args)
    validate_callback_thread_evidence(result.get("callback_thread_evidence"), counts, args)
    expected_algorithm, expected_source = PARTITION_ALGORITHMS[args.partition_algorithm]
    requested = partitioning.get("requested_algorithm")
    accepted = partitioning.get("accepted_algorithm")
    if (
        not isinstance(requested, dict)
        or not isinstance(accepted, dict)
        or requested.get("id") != expected_algorithm
        or accepted.get("id") != expected_algorithm
        or requested.get("version") != 1
        or accepted.get("version") != 1
        or partitioning.get("selection_source") != expected_source
    ):
        raise RuntimeError(f"{path} reported unexpected partitioning provenance")
    support_threads = placement.get("support_threads")
    if not isinstance(support_threads, list):
        raise RuntimeError(f"{path} is missing support-thread evidence")
    expected_lane_count = expected_operator_lane_count(args)
    expected_sink_cpu = (
        None
        if args.scenario == "source_only"
        else (
            args.source_cpu
            if args.operator_lanes == "shared"
            else (None if args.operator_lanes == "per_sink" else args.sink_cpu)
        )
    )
    expected_relay_cpu = (
        (args.source_cpu if args.operator_lanes == "shared" else args.relay_cpu)
        if args.scenario == "sample_forward"
        else None
    )
    expected_sink_cpus = (
        []
        if args.scenario == "source_only"
        else (
            [optional_cpu(args.source_cpu)] * args.fanout
            if args.operator_lanes == "shared"
            else (
                [optional_cpu(cpu) for cpu in (args.sink_cpus or [-1] * args.fanout)]
                if args.operator_lanes == "per_sink"
                else [optional_cpu(args.sink_cpu)] * args.fanout
            )
        )
    )
    partition_uses = placement.get("operator_partition_uses")
    if (
        placement.get("operator_lanes") != args.operator_lanes
        or placement.get("operator_lane_count") != expected_lane_count
        or placement.get("source_compile_accepted_cpu") != optional_cpu(args.source_cpu)
        or placement.get("sink_compile_accepted_cpu") != optional_cpu(expected_sink_cpu)
        or placement.get("sink_compile_accepted_cpus") != expected_sink_cpus
        or placement.get("relay_compile_accepted_cpu") != optional_cpu(expected_relay_cpu)
        or not isinstance(partition_uses, list)
        or len(partition_uses) != expected_lane_count
    ):
        raise RuntimeError(f"{path} reported unexpected Operator lane placement")
    partition_names = {
        item.get("partition")
        for item in partition_uses
        if isinstance(item, dict) and isinstance(item.get("partition"), str)
    }
    lane_ids = {
        item.get("lane")
        for item in partition_uses
        if isinstance(item, dict) and isinstance(item.get("lane"), int) and item.get("lane") >= 0
    }
    expected_partitions = {"runtime-graph-source"}
    if args.operator_lanes == "shared":
        expected_partitions = {"runtime-graph-shared"}
    elif args.scenario == "sample_forward":
        expected_partitions.update(("runtime-graph-relay", "runtime-graph-sink"))
    elif args.operator_lanes == "per_sink" or args.fanout > 1:
        expected_partitions.update(
            f"runtime-graph-sink-{index + 1}" for index in range(args.fanout)
        )
    elif args.scenario != "source_only":
        expected_partitions.add("runtime-graph-sink")
    if partition_names != expected_partitions or len(lane_ids) != expected_lane_count:
        raise RuntimeError(f"{path} reported unexpected Operator partition/lane inventory")
    by_role = {item.get("role"): item for item in support_threads if isinstance(item, dict)}
    expected_support_threads = {
        "controller": args.controller_cpu,
    }
    if uses_channel_pump(args):
        expected_support_threads["channel_pump"] = args.channel_pump_cpu
    if args.workload != "latency":
        expected_support_threads["notification_driver"] = args.notification_driver_cpu
    known_support_thread_roles = {"controller", "channel_pump", "notification_driver"}
    if (
        len(support_threads) != len(by_role)
        or not set(expected_support_threads).issubset(by_role)
        or not set(by_role).issubset(known_support_thread_roles)
    ):
        raise RuntimeError(f"{path} reported unexpected support-thread inventory")
    for role, requested_cpu in expected_support_threads.items():
        evidence = by_role.get(role)
        if (
            not isinstance(evidence, dict)
            or evidence.get("entered") is not True
            or evidence.get("error") is not None
            or not valid_thread_identity_evidence(
                evidence.get("thread_identity_kind"),
                evidence.get("native_tid_syscall_unavailable"),
            )
            or evidence.get("requested_cpu") != (requested_cpu if requested_cpu >= 0 else None)
        ):
            raise RuntimeError(f"{path} reported invalid {role} thread evidence")
        if requested_cpu >= 0 and evidence.get("observed_affinity") != [requested_cpu]:
            raise RuntimeError(f"{path} did not apply the requested {role} affinity")
    for role in set(by_role) - set(expected_support_threads):
        evidence = by_role[role]
        if (
            evidence.get("entered") is not False
            or evidence.get("error") is not None
            or evidence.get("thread_identity_kind") != "none"
            or evidence.get("native_tid_syscall_unavailable") is not False
            or evidence.get("requested_cpu") is not None
            or evidence.get("native_tid") is not None
            or evidence.get("observed_affinity") != []
        ):
            raise RuntimeError(f"{path} reported unexpected support-thread inventory")
    expected_total = args.samples + args.warmup + 1
    source_only = args.scenario == "source_only"
    if geometry.get("samples") != args.samples or geometry.get("warmup") != args.warmup:
        raise RuntimeError(f"{path} reported unexpected workload geometry")
    expected_period = (
        args.publisher_period_us if args.workload in {"latency", "notification_latency"} else None
    )
    expected_pipeline_edges = (
        0
        if source_only
        else (
            args.relay_count + 1
            if args.scenario == "sample_forward"
            else (args.fanout if fanout_scenario(args.scenario) else 1)
        )
    )
    if (
        geometry.get("queue_depth") != args.queue_depth
        or geometry.get("fanout") != (0 if source_only else args.fanout)
        or geometry.get("relay_count")
        != (args.relay_count if args.scenario == "sample_forward" else 0)
        or geometry.get("pipeline_edges") != expected_pipeline_edges
        or geometry.get("publisher_period_us") != expected_period
        or geometry.get("notification_capacity")
        != (args.notification_capacity if args.workload != "latency" else 0)
        or geometry.get("publication_backpressure_capacity") != args.queue_depth
    ):
        raise RuntimeError(f"{path} reported unexpected queue, notification, or pacing geometry")
    bulk_payload = args.scenario in {"tensor", "image", "multi_tensor"}
    expected_bulk_work_profile = {
        "tensor": args.bulk_work_profile,
        "image": "descriptor_only",
        "multi_tensor": args.bulk_work_profile,
    }.get(args.scenario, "not_applicable")
    pool_slots = geometry.get("pool_slots")
    release_capacity = geometry.get("release_capacity")
    expected_release_streams = args.fanout if bulk_payload else None
    if (
        geometry.get("max_fanout_release_streams") != expected_release_streams
        or geometry.get("bulk_work_profile") != expected_bulk_work_profile
        or (
            bulk_payload
            and (
                not isinstance(pool_slots, int)
                or pool_slots <= 0
                or not isinstance(release_capacity, int)
                or release_capacity <= 0
            )
        )
        or (not bulk_payload and (pool_slots is not None or release_capacity is not None))
    ):
        raise RuntimeError(f"{path} reported unexpected bulk-resource geometry")
    expected_publications = 0 if source_only else expected_total
    lossy = lossy_queue_policy(args.queue_policy)
    reported_deliveries = counts.get("deliveries")
    expected_deliveries = (
        0
        if source_only
        else (
            reported_deliveries
            if lossy and isinstance(reported_deliveries, int)
            else expected_total * args.fanout
        )
    )
    expected_sink = expected_deliveries
    expected_deliveries_per_sink = (
        expected_total if source_only or not lossy else expected_deliveries
    )
    expected_completions = expected_total if source_only or not lossy else expected_deliveries
    if (
        counts.get("publications") != expected_publications
        or counts.get("deliveries") != expected_deliveries
        or counts.get("activation_completions") != expected_completions
    ):
        raise RuntimeError(f"{path} reported incomplete publication or delivery counts")
    replacements = counts.get("consumer_visible_replacements")
    replacement_rate = counts.get("replacement_rate")
    if (
        not isinstance(replacements, int)
        or isinstance(replacement_rate, bool)
        or not isinstance(replacement_rate, (int, float))
        or (
            lossy
            and (
                expected_deliveries <= 0
                or expected_deliveries > expected_publications
                or replacements != expected_publications - expected_deliveries
                or not math.isclose(
                    float(replacement_rate),
                    replacements / expected_publications,
                    rel_tol=1.0e-12,
                    abs_tol=1.0e-12,
                )
                or (args.workload != "saturated" and replacements != 0)
            )
        )
        or (not lossy and (replacements != 0 or replacement_rate != 0))
    ):
        raise RuntimeError(f"{path} reported inconsistent replacement counts")
    validate_callback_counts(path, counts, args, expected_total, expected_sink)
    descriptor_initializations = counts.get("descriptor_backing_initializations")
    measured_descriptor_initializations = counts.get(
        "descriptor_backing_initializations_during_measurement"
    )
    descriptor_only = (
        args.scenario in {"tensor", "multi_tensor"} and args.bulk_work_profile == "descriptor_only"
    )
    invalid_descriptor_initializations = False
    if descriptor_only:
        invalid_descriptor_initializations = (
            not isinstance(descriptor_initializations, int)
            or not isinstance(measured_descriptor_initializations, int)
            or descriptor_initializations < (2 if args.scenario == "multi_tensor" else 1)
            or measured_descriptor_initializations != 0
        )
    elif descriptor_initializations is not None or measured_descriptor_initializations is not None:
        invalid_descriptor_initializations = (
            descriptor_initializations != 0 or measured_descriptor_initializations != 0
        )
    if invalid_descriptor_initializations:
        raise RuntimeError(f"{path} reported invalid descriptor-backing initialization evidence")
    per_sink = counts.get("per_sink")
    if source_only:
        if per_sink != []:
            raise RuntimeError(f"{path} reported unexpected sink evidence")
    elif (
        not isinstance(per_sink, list)
        or len(per_sink) != args.fanout
        or any(
            not isinstance(item, dict)
            or item.get("index") != index
            or item.get("deliveries") != expected_deliveries_per_sink
            or item.get("final_sequence") != expected_total
            for index, item in enumerate(per_sink)
        )
    ):
        raise RuntimeError(f"{path} reported incomplete per-sink evidence")
    if counts.get("validation_errors") != 0 or validation.get("passed") is not True:
        raise RuntimeError(f"{path} reported payload validation failures")
    if validation.get("mode") != args.payload_validation:
        raise RuntimeError(f"{path} reported an unexpected validation mode")
    expected_validation_timing = {
        "none": "none",
        "final": "after_timed_operations",
        "all": "between_timed_operations",
    }[args.payload_validation]
    if validation.get("timing") != expected_validation_timing:
        raise RuntimeError(f"{path} reported unexpected validation timing")
    final_validation_ns = validation.get("final_validation_ns")
    final_validation_expected = (
        args.scenario in {"tensor", "image", "multi_tensor"} and args.payload_validation == "final"
    )
    if (
        final_validation_expected
        and (
            not isinstance(final_validation_ns, int)
            or isinstance(final_validation_ns, bool)
            or final_validation_ns < 0
        )
    ) or (not final_validation_expected and final_validation_ns is not None):
        raise RuntimeError(f"{path} reported invalid final-validation timing")
    tensor = result.get("tensor")
    if args.scenario == "tensor":
        if not isinstance(tensor, dict):
            raise RuntimeError(f"{path} is missing Tensor evidence")
        if (
            geometry.get("payload_bytes") != args.tensor_bytes
            or tensor.get("storage") != args.tensor_storage
            or tensor.get("device") != args.tensor_device
            or tensor.get("direct_allocation_records") != 1
            or tensor.get("pointer_identity_checked") is not True
        ):
            raise RuntimeError(f"{path} reported unexpected Tensor evidence")
    elif args.scenario == "image":
        if not isinstance(tensor, dict):
            raise RuntimeError(f"{path} is missing Image Tensor evidence")
        if (
            geometry.get("payload_bytes") != args.tensor_bytes
            or tensor.get("storage") != "pageable_host"
            or tensor.get("device") != 0
            or tensor.get("direct_allocation_records") != 0
            or tensor.get("pointer_identity_checked") is not True
            or tensor.get("descriptor_schema") is not True
        ):
            raise RuntimeError(f"{path} reported unexpected Image Tensor evidence")
    elif args.scenario == "multi_tensor":
        if not isinstance(tensor, dict):
            raise RuntimeError(f"{path} is missing multi-Tensor evidence")
        expected_fields = [
            {"name": "first", "dtype": "uint8", "rank": 1, "bytes": args.tensor_bytes},
            {"name": "second", "dtype": "uint8", "rank": 1, "bytes": args.tensor_bytes},
        ]
        if (
            geometry.get("payload_bytes") != 2 * args.tensor_bytes
            or geometry.get("application_payload_bytes") != 2 * args.tensor_bytes
            or tensor.get("storage") != "pageable_host"
            or tensor.get("device") != 0
            or tensor.get("direct_allocation_records") != 2
            or tensor.get("pointer_identity_checked") is not True
            or tensor.get("descriptor_schema") is not True
            or tensor.get("native_type") != "runtime_benchmark::MultiTensorMessageT"
            or tensor.get("field_count") != 2
            or tensor.get("fields") != expected_fields
        ):
            raise RuntimeError(f"{path} reported unexpected multi-Tensor evidence")
    if session.get("termination") != "stopped" or session.get("diagnostics") != []:
        raise RuntimeError(f"{path} reported an unsuccessful Runtime session")
    throughput_fields = (
        "logical_publications_per_second",
        "aggregate_deliveries_per_second",
        "logical_payload_bytes_per_second",
        "aggregate_payload_bytes_per_second",
    )
    configured_rate = timing.get("configured_publication_rate_per_second")
    if args.workload == "saturated":
        if configured_rate is not None or any(
            isinstance(timing.get(name), bool) or not isinstance(timing.get(name), (int, float))
            for name in throughput_fields
        ):
            raise RuntimeError(f"{path} reported invalid saturated throughput metrics")
    elif (
        isinstance(configured_rate, bool)
        or not isinstance(configured_rate, (int, float))
        or configured_rate <= 0
        or any(timing.get(name) is not None for name in throughput_fields)
    ):
        raise RuntimeError(f"{path} reported throughput metrics for a paced workload")
    expected_timing_fanout = 0 if source_only else args.fanout
    causal_per_sink_timing_fields = (
        timing.get("source_to_sink_by_sink_ns"),
        timing.get("sink_callback_to_post_receive_by_sink_ns"),
    )
    overlap_per_sink_timing_fields = (
        timing.get("source_post_emit_to_sink_by_sink_ns"),
        timing.get("source_pre_return_to_sink_callback_by_sink_ns"),
    )
    if any(
        not isinstance(per_sink_timing, list)
        or len(per_sink_timing) != expected_timing_fanout
        or any(
            not isinstance(item, dict)
            or item.get("index") != index
            or (
                args.workload in {"latency", "notification_latency"}
                and (
                    not isinstance(item.get("latency_ns"), dict)
                    or item["latency_ns"].get("samples") != args.samples
                )
            )
            or (args.workload == "saturated" and item.get("latency_ns") is not None)
            for index, item in enumerate(per_sink_timing)
        )
        for per_sink_timing in causal_per_sink_timing_fields
    ) or any(
        not isinstance(per_sink_timing, list)
        or len(per_sink_timing) != expected_timing_fanout
        or any(
            not isinstance(item, dict)
            or item.get("index") != index
            or (
                args.workload in {"latency", "notification_latency"}
                and not complete_overlap_interval(
                    item.get("latency_ns"), item.get("overlaps"), args.samples
                )
            )
            or (
                args.workload == "saturated"
                and (item.get("latency_ns") is not None or item.get("overlaps") != 0)
            )
            for index, item in enumerate(per_sink_timing)
        )
        for per_sink_timing in overlap_per_sink_timing_fields
    ):
        raise RuntimeError(f"{path} reported incomplete per-sink timing evidence")
    latency = timing.get("latency_ns")
    publication_to_receive = timing.get("publication_to_receive_ns")
    source_emit = timing.get("source_emit_ns")
    source_publication_call = timing.get("source_publication_call_ns")
    source_post_emit_to_pre_return = timing.get("source_post_emit_to_pre_return_ns")
    source_post_emit_to_sink = timing.get("source_post_emit_to_sink_ns")
    source_post_emit_to_sink_overlaps = timing.get("source_post_emit_to_sink_overlaps")
    tensor_phase_fields = (
        timing.get("tensor_loan_acquire_ns"),
        timing.get("tensor_write_commit_ns"),
        timing.get("tensor_receive_call_ns"),
    )
    tensor_phase_expected = args.scenario == "tensor" and args.workload in {
        "latency",
        "notification_latency",
    }
    if tensor_phase_expected:
        if any(
            not isinstance(field, dict) or field.get("samples") != args.samples
            for field in tensor_phase_fields
        ):
            raise RuntimeError(f"{path} reported incomplete Tensor phase timing")
    elif any(field is not None for field in tensor_phase_fields):
        raise RuntimeError(f"{path} reported unexpected Tensor phase timing")
    if args.workload == "latency":
        if (
            not isinstance(latency, dict)
            or latency.get("samples") != args.samples
            or (
                not source_only
                and (
                    not isinstance(publication_to_receive, dict)
                    or publication_to_receive.get("samples") != args.samples
                    or not isinstance(source_emit, dict)
                    or source_emit.get("samples") != args.samples
                    or not isinstance(source_publication_call, dict)
                    or source_publication_call.get("samples") != args.samples
                    or not isinstance(source_post_emit_to_pre_return, dict)
                    or source_post_emit_to_pre_return.get("samples") != args.samples
                    or not complete_overlap_interval(
                        source_post_emit_to_sink,
                        source_post_emit_to_sink_overlaps,
                        args.samples,
                    )
                )
            )
        ):
            raise RuntimeError(f"{path} reported incomplete latency samples")
        if counts.get("notification_post_attempts") != 0:
            raise RuntimeError(f"{path} reported notification posts for paced latency")
    elif args.workload == "notification_latency":
        accepted = counts.get("notifications_accepted")
        attempts = counts.get("notification_post_attempts")
        notification_to_source = timing.get("notification_to_source_ns")
        source_callback_entry_to_pre_emit = timing.get("source_callback_entry_to_pre_emit_ns")
        source_to_sink = timing.get("source_to_sink_ns")
        complete_components = (
            not isinstance(latency, dict)
            or latency.get("samples") != args.samples
            or not isinstance(notification_to_source, dict)
            or notification_to_source.get("samples") != args.samples
        )
        if source_only:
            complete_components = (
                complete_components
                or source_callback_entry_to_pre_emit is not None
                or publication_to_receive is not None
                or source_emit is not None
                or source_publication_call is not None
                or source_post_emit_to_pre_return is not None
                or source_post_emit_to_sink is not None
                or source_post_emit_to_sink_overlaps != 0
                or source_to_sink is not None
            )
        else:
            complete_components = (
                complete_components
                or not isinstance(publication_to_receive, dict)
                or publication_to_receive.get("samples") != args.samples
                or not isinstance(source_callback_entry_to_pre_emit, dict)
                or source_callback_entry_to_pre_emit.get("samples") != args.samples
                or not isinstance(source_emit, dict)
                or source_emit.get("samples") != args.samples
                or not isinstance(source_publication_call, dict)
                or source_publication_call.get("samples") != args.samples
                or not isinstance(source_post_emit_to_pre_return, dict)
                or source_post_emit_to_pre_return.get("samples") != args.samples
                or not complete_overlap_interval(
                    source_post_emit_to_sink,
                    source_post_emit_to_sink_overlaps,
                    args.samples,
                )
                or not isinstance(source_to_sink, dict)
                or source_to_sink.get("samples") != args.samples
            )
        if complete_components:
            raise RuntimeError(f"{path} reported incomplete notification latency samples")
        if accepted != expected_total or not isinstance(attempts, int) or attempts < accepted:
            raise RuntimeError(f"{path} reported inconsistent notification counts")
    else:
        accepted = counts.get("notifications_accepted")
        attempts = counts.get("notification_post_attempts")
        if (
            latency is not None
            or publication_to_receive is not None
            or source_emit is not None
            or source_publication_call is not None
            or source_post_emit_to_pre_return is not None
            or source_post_emit_to_sink is not None
            or source_post_emit_to_sink_overlaps != 0
        ):
            raise RuntimeError(f"{path} reported latency samples for saturated throughput")
        if (
            not isinstance(accepted, int)
            or not isinstance(attempts, int)
            or accepted < expected_total
            or attempts < accepted
        ):
            raise RuntimeError(f"{path} reported inconsistent notification counts")
    if isinstance(latency, dict):
        p999 = latency.get("p999")
        if (
            args.samples >= 10_000
            and (isinstance(p999, bool) or not isinstance(p999, (int, float)))
        ) or (args.samples < 10_000 and p999 is not None):
            raise RuntimeError(f"{path} reported inconsistent latency p999 evidence")
    return result


def numeric_path(result: dict[str, Any], *keys: str) -> float:
    value: object = result
    for key in keys:
        if not isinstance(value, dict):
            raise RuntimeError(f"result field {'.'.join(keys)} is not numeric")
        value = value.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"result field {'.'.join(keys)} is not numeric")
    return float(value)


def optional_numeric_path(result: dict[str, Any], *keys: str) -> float | None:
    value: object = result
    for key in keys:
        if value is None:
            return None
        if not isinstance(value, dict):
            raise RuntimeError(f"result field {'.'.join(keys)} is not numeric or null")
        if key not in value:
            raise RuntimeError(f"result field {'.'.join(keys)} is not numeric or null")
        value = value[key]
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"result field {'.'.join(keys)} is not numeric or null")
    return float(value)


def per_sink_numeric_path(
    result: dict[str, Any], timing_field: str, sink_index: int, *keys: str
) -> float:
    timing = result.get("timing")
    values = timing.get(timing_field) if isinstance(timing, dict) else None
    if not isinstance(values, list) or sink_index >= len(values):
        raise RuntimeError(f"result field timing.{timing_field} is incomplete")
    item = values[sink_index]
    if not isinstance(item, dict) or item.get("index") != sink_index:
        raise RuntimeError("result per-sink timing index is invalid")
    return numeric_path(item, *keys)


def per_sink_optional_numeric_path(
    result: dict[str, Any], timing_field: str, sink_index: int, *keys: str
) -> float | None:
    timing = result.get("timing")
    values = timing.get(timing_field) if isinstance(timing, dict) else None
    if not isinstance(values, list) or sink_index >= len(values):
        raise RuntimeError(f"result field timing.{timing_field} is incomplete")
    item = values[sink_index]
    if not isinstance(item, dict) or item.get("index") != sink_index:
        raise RuntimeError("result per-sink timing index is invalid")
    return optional_numeric_path(item, *keys)


def summarize_values(values: list[float]) -> dict[str, float | int | None]:
    return summarize_common_values(values, median_key="median")


def summarize_optional_values(
    values: list[float | None],
) -> dict[str, float | int | None]:
    numeric_values = [value for value in values if value is not None]
    if numeric_values:
        return summarize_values(numeric_values)
    return {
        "count": 0,
        "median": None,
        "min": None,
        "max": None,
        "mean": None,
        "coefficient_of_variation": None,
    }


def quality_gate(
    args: argparse.Namespace,
    cpu_topology: list[dict[str, object]],
    latency_medians: dict[str, float | int | None],
) -> dict[str, object]:
    issues: list[str] = []
    if args.repetitions < 2:
        issues.append("insufficient_repetitions_for_dispersion")
    if any(not item["pinned"] for item in cpu_topology):
        issues.append("runtime_thread_cpu_affinity_unpinned")
    if any(item["pinned"] and not item.get("topology_available") for item in cpu_topology):
        issues.append("physical_cpu_topology_unavailable")
    governors = [item.get("governor") for item in cpu_topology if item["pinned"]]
    if governors and any(governor != "performance" for governor in governors):
        issues.append("operator_cpu_governor_not_performance")
    median_cv = latency_medians["coefficient_of_variation"]
    if isinstance(median_cv, float) and median_cv > args.max_median_cv:
        issues.append("run_to_run_median_cv_exceeded")
    return {
        "publishable": not issues,
        "issues": issues,
        "thresholds": {"maximum_run_to_run_median_cv": args.max_median_cv},
    }


def benchmark_command(args: argparse.Namespace, binary: Path, output: Path) -> list[str]:
    command = [
        str(binary),
        "--workload",
        args.workload,
        "--scenario",
        args.scenario,
        "--operator-lanes",
        args.operator_lanes,
        "--samples",
        str(args.samples),
        "--warmup",
        str(args.warmup),
        "--queue-policy",
        args.queue_policy,
        "--queue-depth",
        str(args.queue_depth),
        "--relay-count",
        str(args.relay_count),
        "--fanout",
        str(args.fanout),
        "--payload-validation",
        args.payload_validation,
        "--tensor-storage",
        args.tensor_storage,
        "--tensor-bytes",
        str(args.tensor_bytes),
        "--tensor-device",
        str(args.tensor_device),
        "--bulk-work-profile",
        args.bulk_work_profile,
        "--source-cpu",
        str(args.source_cpu),
        "--relay-cpu",
        str(args.relay_cpu),
        "--sink-cpu",
        str(args.sink_cpu),
        "--controller-cpu",
        str(args.controller_cpu),
        "--channel-pump-cpu",
        str(args.channel_pump_cpu),
        "--notification-driver-cpu",
        str(args.notification_driver_cpu),
        "--timeout-ms",
        str(args.session_timeout_ms),
        "--output",
        str(output),
    ]
    if args.sink_cpus:
        command.extend(["--sink-cpus", ",".join(str(cpu) for cpu in args.sink_cpus)])
    if args.workload in {"latency", "notification_latency"}:
        command.extend(["--publisher-period-us", str(args.publisher_period_us)])
    if args.workload != "latency":
        command.extend(["--notification-capacity", str(args.notification_capacity)])
    if args.busy_poll:
        command.append("--busy-poll")
    if args.runtime_phase_counters:
        command.append("--runtime-phase-counters")
    if args.disable_direct_continuation_elision:
        command.append("--disable-direct-continuation-elision")
    if args.disable_closed_island_composite:
        command.append("--disable-closed-island-composite")
    if args.force_holomq_process_local_inline:
        command.append("--force-holomq-process-local-inline")
    return command


def benchmark_environment(_args: argparse.Namespace) -> dict[str, str]:
    environment = os.environ.copy()
    environment.pop(PARTITION_ENVIRONMENT, None)
    return environment


def run_repetition(
    args: argparse.Namespace, binary: Path, repetition: int, output_dir: Path
) -> tuple[dict[str, Any], list[str]]:
    repetition_dir = output_dir / f"repetition_{repetition:03d}"
    repetition_dir.mkdir(parents=True, exist_ok=True)
    run_token = f"{os.getpid()}_{time.monotonic_ns()}"
    result_path = repetition_dir / f"result_{run_token}.json"
    command = benchmark_command(args, binary, result_path)
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            env=benchmark_environment(args),
            text=True,
            timeout=args.process_timeout_sec,
        )
    except subprocess.TimeoutExpired as error:
        stdout = error.stdout.decode() if isinstance(error.stdout, bytes) else error.stdout or ""
        stderr = error.stderr.decode() if isinstance(error.stderr, bytes) else error.stderr or ""
        (repetition_dir / f"stdout_{run_token}.log").write_text(stdout, encoding="utf-8")
        (repetition_dir / f"stderr_{run_token}.log").write_text(stderr, encoding="utf-8")
        raise TimeoutError(
            f"benchmark repetition {repetition} timed out after {args.process_timeout_sec} s"
        ) from error
    (repetition_dir / f"stdout_{run_token}.log").write_text(completed.stdout, encoding="utf-8")
    (repetition_dir / f"stderr_{run_token}.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"benchmark repetition {repetition} exited with {completed.returncode}; "
            f"logs are in {repetition_dir}"
        )
    return checked_result(result_path, args), command


def summarize_results(
    args: argparse.Namespace,
    results: list[dict[str, Any]],
    cpu_topology: list[dict[str, object]],
) -> dict[str, object]:
    activation_rates = summarize_values(
        [numeric_path(result, "timing", "source_activations_per_second") for result in results]
    )
    metrics: dict[str, object] = {
        "source_activations_per_second_across_runs": activation_rates,
        "consumer_visible_replacements_across_runs": summarize_values(
            [numeric_path(result, "counts", "consumer_visible_replacements") for result in results]
        ),
        "replacement_rate_across_runs": summarize_values(
            [numeric_path(result, "counts", "replacement_rate") for result in results]
        ),
    }
    if args.scenario in {"tensor", "image", "multi_tensor"} and args.payload_validation == "final":
        metrics["final_validation_ns_across_runs"] = summarize_values(
            [numeric_path(result, "validation", "final_validation_ns") for result in results]
        )
    publication_rates: dict[str, float | int | None] | None = None
    if args.workload == "saturated":
        publication_rates = summarize_values(
            [
                numeric_path(result, "timing", "logical_publications_per_second")
                for result in results
            ]
        )
        metrics["logical_publications_per_second_across_runs"] = publication_rates
        metrics["aggregate_deliveries_per_second_across_runs"] = summarize_values(
            [
                numeric_path(result, "timing", "aggregate_deliveries_per_second")
                for result in results
            ]
        )
    else:
        metrics["configured_publication_rate_per_second"] = summarize_values(
            [
                numeric_path(result, "timing", "configured_publication_rate_per_second")
                for result in results
            ]
        )
        metrics["logical_publications_per_second_across_runs"] = None
        metrics["aggregate_deliveries_per_second_across_runs"] = None
    if args.workload in {"latency", "notification_latency"}:
        stability_values = summarize_values(
            [numeric_path(result, "timing", "latency_ns", "p50") for result in results]
        )
        metrics["latency_p50_ns_across_runs"] = stability_values
        metrics["latency_p99_ns_across_runs"] = summarize_values(
            [numeric_path(result, "timing", "latency_ns", "p99") for result in results]
        )
        if args.scenario != "source_only":
            metrics["publication_to_receive_p50_ns_across_runs"] = summarize_values(
                [
                    numeric_path(result, "timing", "publication_to_receive_ns", "p50")
                    for result in results
                ]
            )
            metrics["publication_to_receive_p99_ns_across_runs"] = summarize_values(
                [
                    numeric_path(result, "timing", "publication_to_receive_ns", "p99")
                    for result in results
                ]
            )
        if args.samples >= 10_000:
            metrics["latency_p999_ns_across_runs"] = summarize_values(
                [numeric_path(result, "timing", "latency_ns", "p999") for result in results]
            )
            if args.scenario != "source_only":
                metrics["publication_to_receive_p999_ns_across_runs"] = summarize_values(
                    [
                        numeric_path(result, "timing", "publication_to_receive_ns", "p999")
                        for result in results
                    ]
                )
        if args.scenario == "tensor":
            for field in (
                "tensor_loan_acquire_ns",
                "tensor_write_commit_ns",
                "tensor_receive_call_ns",
            ):
                metrics[f"{field.removesuffix('_ns')}_p50_ns_across_runs"] = summarize_values(
                    [numeric_path(result, "timing", field, "p50") for result in results]
                )
        if args.workload == "notification_latency":
            metrics["notification_to_source_p50_ns_across_runs"] = summarize_values(
                [
                    numeric_path(result, "timing", "notification_to_source_ns", "p50")
                    for result in results
                ]
            )
            if args.scenario != "source_only":
                metrics["source_callback_entry_to_pre_emit_p50_ns_across_runs"] = summarize_values(
                    [
                        numeric_path(
                            result,
                            "timing",
                            "source_callback_entry_to_pre_emit_ns",
                            "p50",
                        )
                        for result in results
                    ]
                )
                metrics["source_to_sink_p50_ns_across_runs"] = summarize_values(
                    [
                        numeric_path(result, "timing", "source_to_sink_ns", "p50")
                        for result in results
                    ]
                )
                metrics["source_emit_p50_ns_across_runs"] = summarize_values(
                    [numeric_path(result, "timing", "source_emit_ns", "p50") for result in results]
                )
                metrics["source_publication_call_p50_ns_across_runs"] = summarize_values(
                    [
                        numeric_path(
                            result,
                            "timing",
                            "source_publication_call_ns",
                            "p50",
                        )
                        for result in results
                    ]
                )
                metrics["source_post_emit_to_pre_return_p50_ns_across_runs"] = summarize_values(
                    [
                        numeric_path(
                            result,
                            "timing",
                            "source_post_emit_to_pre_return_ns",
                            "p50",
                        )
                        for result in results
                    ]
                )
                metrics["source_post_emit_to_sink_p50_ns_across_runs"] = summarize_optional_values(
                    [
                        optional_numeric_path(
                            result, "timing", "source_post_emit_to_sink_ns", "p50"
                        )
                        for result in results
                    ]
                )
                metrics["source_post_emit_to_sink_overlaps_across_runs"] = summarize_values(
                    [
                        numeric_path(result, "timing", "source_post_emit_to_sink_overlaps")
                        for result in results
                    ]
                )
                metrics["source_to_sink_by_sink_p50_ns_across_runs"] = [
                    {
                        "index": sink_index,
                        **summarize_values(
                            [
                                per_sink_numeric_path(
                                    result,
                                    "source_to_sink_by_sink_ns",
                                    sink_index,
                                    "latency_ns",
                                    "p50",
                                )
                                for result in results
                            ]
                        ),
                    }
                    for sink_index in range(args.fanout)
                ]
                metrics["source_post_emit_to_sink_by_sink_p50_ns_across_runs"] = [
                    {
                        "index": sink_index,
                        **summarize_optional_values(
                            [
                                per_sink_optional_numeric_path(
                                    result,
                                    "source_post_emit_to_sink_by_sink_ns",
                                    sink_index,
                                    "latency_ns",
                                    "p50",
                                )
                                for result in results
                            ]
                        ),
                    }
                    for sink_index in range(args.fanout)
                ]
                metrics["source_post_emit_to_sink_by_sink_overlaps_across_runs"] = [
                    {
                        "index": sink_index,
                        **summarize_values(
                            [
                                per_sink_numeric_path(
                                    result,
                                    "source_post_emit_to_sink_by_sink_ns",
                                    sink_index,
                                    "overlaps",
                                )
                                for result in results
                            ]
                        ),
                    }
                    for sink_index in range(args.fanout)
                ]
                metrics["source_pre_return_to_sink_callback_by_sink_p50_ns_across_runs"] = [
                    {
                        "index": sink_index,
                        **summarize_optional_values(
                            [
                                per_sink_optional_numeric_path(
                                    result,
                                    "source_pre_return_to_sink_callback_by_sink_ns",
                                    sink_index,
                                    "latency_ns",
                                    "p50",
                                )
                                for result in results
                            ]
                        ),
                    }
                    for sink_index in range(args.fanout)
                ]
                metrics["source_pre_return_to_sink_callback_by_sink_overlaps_across_runs"] = [
                    {
                        "index": sink_index,
                        **summarize_values(
                            [
                                per_sink_numeric_path(
                                    result,
                                    "source_pre_return_to_sink_callback_by_sink_ns",
                                    sink_index,
                                    "overlaps",
                                )
                                for result in results
                            ]
                        ),
                    }
                    for sink_index in range(args.fanout)
                ]
                metrics["sink_callback_to_post_receive_by_sink_p50_ns_across_runs"] = [
                    {
                        "index": sink_index,
                        **summarize_values(
                            [
                                per_sink_numeric_path(
                                    result,
                                    "sink_callback_to_post_receive_by_sink_ns",
                                    sink_index,
                                    "latency_ns",
                                    "p50",
                                )
                                for result in results
                            ]
                        ),
                    }
                    for sink_index in range(args.fanout)
                ]
        runs = [
            {
                "repetition": index,
                "latency_p50_ns": numeric_path(result, "timing", "latency_ns", "p50"),
                "latency_p90_ns": numeric_path(result, "timing", "latency_ns", "p90"),
                "latency_p99_ns": numeric_path(result, "timing", "latency_ns", "p99"),
                **(
                    {
                        "publication_to_receive_p50_ns": numeric_path(
                            result, "timing", "publication_to_receive_ns", "p50"
                        ),
                        "publication_to_receive_p99_ns": numeric_path(
                            result, "timing", "publication_to_receive_ns", "p99"
                        ),
                        **(
                            {
                                "publication_to_receive_p999_ns": numeric_path(
                                    result,
                                    "timing",
                                    "publication_to_receive_ns",
                                    "p999",
                                )
                            }
                            if args.samples >= 10_000
                            else {}
                        ),
                    }
                    if args.scenario != "source_only"
                    else {}
                ),
                **(
                    {"latency_p999_ns": numeric_path(result, "timing", "latency_ns", "p999")}
                    if args.samples >= 10_000
                    else {}
                ),
                **(
                    {
                        f"{field.removesuffix('_ns')}_p50_ns": numeric_path(
                            result, "timing", field, "p50"
                        )
                        for field in (
                            "tensor_loan_acquire_ns",
                            "tensor_write_commit_ns",
                            "tensor_receive_call_ns",
                        )
                    }
                    if args.scenario == "tensor"
                    else {}
                ),
                **(
                    {
                        "notification_to_source_p50_ns": numeric_path(
                            result, "timing", "notification_to_source_ns", "p50"
                        ),
                        **(
                            {
                                "source_callback_entry_to_pre_emit_p50_ns": numeric_path(
                                    result,
                                    "timing",
                                    "source_callback_entry_to_pre_emit_ns",
                                    "p50",
                                ),
                                "source_to_sink_p50_ns": numeric_path(
                                    result, "timing", "source_to_sink_ns", "p50"
                                ),
                                "source_emit_p50_ns": numeric_path(
                                    result, "timing", "source_emit_ns", "p50"
                                ),
                                "source_publication_call_p50_ns": numeric_path(
                                    result,
                                    "timing",
                                    "source_publication_call_ns",
                                    "p50",
                                ),
                                "source_post_emit_to_pre_return_p50_ns": numeric_path(
                                    result,
                                    "timing",
                                    "source_post_emit_to_pre_return_ns",
                                    "p50",
                                ),
                                "source_post_emit_to_sink_p50_ns": optional_numeric_path(
                                    result, "timing", "source_post_emit_to_sink_ns", "p50"
                                ),
                                "source_post_emit_to_sink_overlaps": numeric_path(
                                    result, "timing", "source_post_emit_to_sink_overlaps"
                                ),
                                "source_to_sink_by_sink_p50_ns": [
                                    per_sink_numeric_path(
                                        result,
                                        "source_to_sink_by_sink_ns",
                                        sink_index,
                                        "latency_ns",
                                        "p50",
                                    )
                                    for sink_index in range(args.fanout)
                                ],
                                "source_post_emit_to_sink_by_sink_p50_ns": [
                                    per_sink_optional_numeric_path(
                                        result,
                                        "source_post_emit_to_sink_by_sink_ns",
                                        sink_index,
                                        "latency_ns",
                                        "p50",
                                    )
                                    for sink_index in range(args.fanout)
                                ],
                                "source_post_emit_to_sink_by_sink_overlaps": [
                                    per_sink_numeric_path(
                                        result,
                                        "source_post_emit_to_sink_by_sink_ns",
                                        sink_index,
                                        "overlaps",
                                    )
                                    for sink_index in range(args.fanout)
                                ],
                                "source_pre_return_to_sink_callback_by_sink_p50_ns": [
                                    per_sink_optional_numeric_path(
                                        result,
                                        "source_pre_return_to_sink_callback_by_sink_ns",
                                        sink_index,
                                        "latency_ns",
                                        "p50",
                                    )
                                    for sink_index in range(args.fanout)
                                ],
                                "source_pre_return_to_sink_callback_by_sink_overlaps": [
                                    per_sink_numeric_path(
                                        result,
                                        "source_pre_return_to_sink_callback_by_sink_ns",
                                        sink_index,
                                        "overlaps",
                                    )
                                    for sink_index in range(args.fanout)
                                ],
                                "sink_callback_to_post_receive_by_sink_p50_ns": [
                                    per_sink_numeric_path(
                                        result,
                                        "sink_callback_to_post_receive_by_sink_ns",
                                        sink_index,
                                        "latency_ns",
                                        "p50",
                                    )
                                    for sink_index in range(args.fanout)
                                ],
                            }
                            if args.scenario != "source_only"
                            else {}
                        ),
                    }
                    if args.workload == "notification_latency"
                    else {}
                ),
                "configured_publication_rate_per_second": numeric_path(
                    result, "timing", "configured_publication_rate_per_second"
                ),
                "logical_publications_per_second": None,
                "source_activations_per_second": numeric_path(
                    result, "timing", "source_activations_per_second"
                ),
                "aggregate_deliveries_per_second": None,
                "counts": result.get("counts", {}),
            }
            for index, result in enumerate(results)
        ]
    else:
        stability_values = activation_rates if args.scenario == "source_only" else publication_rates
        if stability_values is None:
            raise RuntimeError("saturated summary is missing publication rates")
        runs = [
            {
                "repetition": index,
                "source_activations_per_second": numeric_path(
                    result, "timing", "source_activations_per_second"
                ),
                "logical_publications_per_second": numeric_path(
                    result, "timing", "logical_publications_per_second"
                ),
                "aggregate_deliveries_per_second": numeric_path(
                    result, "timing", "aggregate_deliveries_per_second"
                ),
                "counts": result.get("counts", {}),
            }
            for index, result in enumerate(results)
        ]
    summary = {
        "schema": SUMMARY_SCHEMA,
        "benchmark": results[0]["benchmark"],
        "answers": results[0]["answers"],
        "scenario": args.scenario,
        "process_local_inline_delivery": process_local_delivery(args),
        "runtime_controls": {
            "direct_continuation_elision_disabled": args.disable_direct_continuation_elision,
            "closed_island_composite_disabled": args.disable_closed_island_composite,
        },
        "tensor": results[0].get("tensor", {}),
        "workload": args.workload,
        "source_trigger": results[0]["source_trigger"],
        "repetitions": len(results),
        "geometry": results[0]["geometry"],
        "clock": results[0]["clock"],
        "timestamp_instrumentation": {
            "calibration": "post_measurement_back_to_back_pair",
            "clock_reads_per_sample": 2,
            "pair_overhead_p50_ns_across_runs": summarize_values(
                [
                    numeric_path(
                        result,
                        "timestamp_instrumentation",
                        "pair_overhead_ns",
                        "p50",
                    )
                    for result in results
                ]
            ),
        },
        "validation": results[0].get(
            "validation",
            {
                "mode": args.payload_validation,
                "timing": "unknown",
                "passed": True,
            },
        ),
        "allocation_probe": results[0].get("allocation_probe"),
        "partitioning": results[0]["partitioning"],
        "wake": results[0]["wake"],
        "placement": results[0]["placement"],
        "runtime_support_threads": ",".join(
            role
            for role in ("controller", "channel_pump", "notification_driver")
            if any(
                isinstance(item, dict) and item.get("role") == role and item.get("entered") is True
                for item in results[0]["placement"]["support_threads"]
            )
        )
        or "none",
        "metrics": metrics,
        "quality_gate": quality_gate(args, cpu_topology, stability_values),
        "runs": runs,
    }
    if args.runtime_phase_counters:
        summary["runtime_phase_counter_runs"] = [
            result["runtime_phase_counters"] for result in results
        ]
        summary["runtime_phase_trace_runs"] = [result["runtime_phase_trace"] for result in results]
    return summary


def main() -> int:
    args = parse_args()
    validate_args(args)
    binary = args.benchmark_binary.resolve()
    if not binary.is_file():
        raise FileNotFoundError(f"benchmark binary does not exist: {binary}")
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="hsdk-runtime-graph-bench-"))
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"benchmark artifacts: {output_dir}", file=sys.stderr)

    cpu_topology = collect_cpu_topology(args)
    results: list[dict[str, Any]] = []
    commands: list[list[str]] = []
    for repetition in range(args.repetitions):
        result, command = run_repetition(args, binary, repetition, output_dir)
        results.append(result)
        commands.append(command)

    summary = summarize_results(args, results, cpu_topology)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    revision, revision_source = git_revision(args.git_revision)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "command": sys.argv,
        "native_commands": commands,
        "hostname": platform.node(),
        "kernel": platform.release(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "git_revision": revision,
        "git_revision_source": revision_source,
        "binary": str(binary),
        "partition_environment": benchmark_environment(args).get(PARTITION_ENVIRONMENT),
        "runtime_controls": {
            "direct_continuation_elision_disabled": args.disable_direct_continuation_elision,
            "closed_island_composite_disabled": args.disable_closed_island_composite,
        },
        "build": cmake_metadata(binary),
        "cpu_topology": cpu_topology,
        "gpus": gpu_metadata(),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
