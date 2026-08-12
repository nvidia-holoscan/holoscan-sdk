#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run one interprocess HoloMQ inline-payload benchmark case."""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import os
import platform
import selectors
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import IO, Any

PRESET_DEFAULTS = {
    "none": {"samples": 100_000, "warmup": 10_000, "queue_depth": 64, "repetitions": 1},
    "smoke": {"samples": 1_000, "warmup": 100, "queue_depth": 8, "repetitions": 1},
    "production": {
        "samples": 100_000,
        "warmup": 10_000,
        "queue_depth": 64,
        "repetitions": 7,
    },
}
DEFAULT_TIMEOUT_SEC = 30
PACED_TIMEOUT_GRACE_SEC = 10
CUDA_UNAVAILABLE_EXIT_CODE = 77
MINIMUM_P999_SAMPLES = 10_000


def apply_preset(args: argparse.Namespace) -> None:
    defaults = PRESET_DEFAULTS[args.preset]
    for name, value in defaults.items():
        if getattr(args, name) is None:
            if name == "queue_depth" and args.workload == "latency":
                value = 1
            setattr(args, name, value)
    if args.publisher_period_us is None:
        if args.workload == "latency":
            if getattr(args, "payload", "inline") == "buffer_handle":
                args.publisher_period_us = (
                    500 if getattr(args, "storage", "pageable_host") == "device" else 250
                )
            else:
                args.publisher_period_us = 10
        else:
            args.publisher_period_us = 0
    if args.pool_slots is None:
        minimum_pool_slots = args.queue_depth + 2
        if (
            args.workload == "latency"
            and getattr(args, "payload", "inline") == "buffer_handle"
            and getattr(args, "storage", "pageable_host") == "device"
            and getattr(args, "device_sync", "none") == "ready"
        ):
            minimum_pool_slots = max(minimum_pool_slots, 10)
        args.pool_slots = minimum_pool_slots


def default_timeout_sec(args: argparse.Namespace) -> int:
    pacing_us = (args.samples + args.warmup) * args.publisher_period_us
    pacing_sec = (pacing_us + 999_999) // 1_000_000
    return max(DEFAULT_TIMEOUT_SEC, pacing_sec + PACED_TIMEOUT_GRACE_SEC)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preset", choices=tuple(PRESET_DEFAULTS), default="none")
    parser.add_argument("--workload", choices=("throughput", "latency"), default="throughput")
    parser.add_argument("--payload", choices=("inline", "buffer_handle"), default="inline")
    parser.add_argument(
        "--storage",
        choices=("pageable_host", "pinned_host", "device"),
        default="pageable_host",
    )
    parser.add_argument(
        "--device-sync",
        choices=("none", "publisher_ready", "subscriber_release", "ready"),
        default="none",
    )
    parser.add_argument("--payload-bytes", type=int, choices=(64, 128, 440), default=64)
    parser.add_argument("--buffer-bytes", type=int, default=1 << 20)
    parser.add_argument("--pool-slots", type=int)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--queue-depth", type=int)
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--subscriber-count", type=int, default=1)
    parser.add_argument("--publisher-period-us", type=int)
    parser.add_argument("--publisher-pacing", choices=("sleep", "spin"), default="spin")
    parser.add_argument("--payload-validation", choices=("none", "final", "all"), default="final")
    parser.add_argument("--publisher-cpu", type=int, default=-1)
    parser.add_argument("--subscriber-cpu", type=int, default=-1)
    parser.add_argument("--skip-if-no-cuda", action="store_true")
    parser.add_argument(
        "--subscriber-cpus",
        default="",
        help="comma-separated subscriber CPUs; required for pinned fanout",
    )
    parser.add_argument(
        "--timeout-sec",
        type=int,
        help="per-run timeout; defaults to at least the paced workload duration plus 10 seconds",
    )
    args = parser.parse_args()
    apply_preset(args)
    if args.timeout_sec is None:
        args.timeout_sec = default_timeout_sec(args)
    if (
        args.samples <= 0
        or args.warmup < 0
        or args.queue_depth <= 0
        or args.subscriber_count <= 0
        or args.repetitions <= 0
        or args.timeout_sec <= 0
        or args.publisher_period_us < 0
        or args.buffer_bytes <= 0
        or args.pool_slots <= 0
    ):
        parser.error(
            "samples, queue depth, subscriber count, repetitions, and timeout must be positive; "
            "warmup must be nonnegative"
        )
    if args.workload == "throughput" and args.publisher_period_us != 0:
        parser.error("throughput requires --publisher-period-us 0")
    if args.workload == "latency" and (
        args.publisher_period_us <= 0 or args.queue_depth != 1 or args.subscriber_count != 1
    ):
        parser.error(
            "latency requires one subscriber, queue depth 1, and a positive publisher period"
        )
    if args.payload == "buffer_handle" and (
        args.subscriber_count != 1
        or args.buffer_bytes < 16
        or args.pool_slots < args.queue_depth + 2
    ):
        parser.error(
            "buffer_handle currently requires 1:1 operation, at least 16 buffer bytes, and "
            "at least queue depth plus two pool slots"
        )
    if args.payload == "inline" and (args.storage != "pageable_host" or args.device_sync != "none"):
        parser.error("inline payload requires default --storage and --device-sync values")
    if args.storage != "device" and args.device_sync != "none":
        parser.error("non-default --device-sync requires --storage device")
    if args.storage == "device" and args.payload_validation == "all":
        parser.error("device storage supports none or final payload validation")
    if args.publisher_cpu < -1 or args.subscriber_cpu < -1:
        parser.error("CPU values must be -1 (unpinned) or nonnegative")
    if args.subscriber_cpus:
        try:
            subscriber_cpus = [int(value) for value in args.subscriber_cpus.split(",")]
        except ValueError as error:
            parser.error(f"invalid --subscriber-cpus list: {error}")
        if len(subscriber_cpus) != args.subscriber_count:
            parser.error("--subscriber-cpus must contain one CPU per subscriber")
        if any(cpu < 0 for cpu in subscriber_cpus):
            parser.error("--subscriber-cpus values must be nonnegative")
    elif args.subscriber_count == 1:
        subscriber_cpus = [args.subscriber_cpu]
    elif args.subscriber_cpu >= 0:
        parser.error("use --subscriber-cpus when --subscriber-count is greater than one")
    else:
        subscriber_cpus = [-1] * args.subscriber_count
    args.subscriber_cpu_values = subscriber_cpus
    return args


def read_cpu_topology(role: str, cpu: int) -> dict[str, Any]:
    result: dict[str, Any] = {"role": role, "cpu": cpu, "pinned": cpu >= 0}
    if cpu < 0:
        return result
    if not sys.platform.startswith("linux"):
        result["topology_available"] = False
        result["topology_reason"] = "unsupported_platform"
        return result
    if hasattr(os, "sched_getaffinity") and cpu not in os.sched_getaffinity(0):
        raise RuntimeError(f"{role} CPU {cpu} is outside this process's allowed CPU affinity")
    topology_dir = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
    try:
        result["package_id"] = int(
            (topology_dir / "physical_package_id").read_text(encoding="utf-8").strip()
        )
        result["core_id"] = int((topology_dir / "core_id").read_text(encoding="utf-8").strip())
        result["topology_available"] = True
    except (OSError, ValueError):
        result["topology_available"] = False
    return result


def validate_distinct_physical_cores(cpu_topology: list[dict[str, Any]]) -> None:
    physical_cores: dict[tuple[int, int], dict[str, Any]] = {}
    for placement in cpu_topology:
        if not placement.get("pinned") or not placement.get("topology_available"):
            continue
        physical_core = (placement["package_id"], placement["core_id"])
        if previous := physical_cores.get(physical_core):
            raise RuntimeError(
                f"{previous['role']} CPU {previous['cpu']} and {placement['role']} CPU "
                f"{placement['cpu']} share package/core {physical_core}; use distinct physical "
                "cores"
            )
        physical_cores[physical_core] = placement


def collect_cpu_topology(args: argparse.Namespace) -> list[dict[str, Any]]:
    placements = [read_cpu_topology("publisher", args.publisher_cpu)]
    placements.extend(
        read_cpu_topology(f"subscriber_{index}", cpu)
        for index, cpu in enumerate(args.subscriber_cpu_values)
    )
    validate_distinct_physical_cores(placements)
    missing = [
        entry["role"] for entry in placements if entry["pinned"] and not entry["topology_available"]
    ]
    if missing and sys.platform.startswith("linux"):
        print(
            f"warning: physical CPU topology unavailable for {', '.join(missing)}",
            file=sys.stderr,
        )
    return placements


def command_for(
    args: argparse.Namespace,
    role: str,
    service_name: str,
    cpu: int,
    subscriber_index: int = 0,
    ready_fd: int | None = None,
) -> list[str]:
    command = [
        str(args.binary),
        "--role",
        role,
        "--service-name",
        service_name,
        "--workload",
        args.workload,
        "--payload",
        args.payload,
        "--storage",
        args.storage,
        "--device-sync",
        args.device_sync,
        "--payload-bytes",
        str(args.payload_bytes),
        "--buffer-bytes",
        str(args.buffer_bytes),
        "--pool-slots",
        str(args.pool_slots),
        "--samples",
        str(args.samples),
        "--warmup",
        str(args.warmup),
        "--queue-depth",
        str(args.queue_depth),
        "--subscriber-count",
        str(args.subscriber_count),
        "--subscriber-index",
        str(subscriber_index),
        "--publisher-period-us",
        str(args.publisher_period_us),
        "--publisher-pacing",
        args.publisher_pacing,
        "--payload-validation",
        args.payload_validation,
        "--timeout-seconds",
        str(args.timeout_sec),
    ]
    if cpu >= 0:
        command.extend(("--cpu", str(cpu)))
    if ready_fd is not None:
        command.extend(("--ready-fd", str(ready_fd)))
    return command


def terminate(processes: list[subprocess.Popen[str]]) -> None:
    for process in processes:
        if process.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
    deadline = time.monotonic() + 2.0
    for process in processes:
        if process.poll() is None:
            try:
                process.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
    for process in processes:
        if process.poll() is None:
            process.wait()


def wait_ready(
    subscribers: list[subprocess.Popen[str]],
    ready_fds: list[int],
    timeout_sec: int,
    skip_if_no_cuda: bool,
) -> None:
    if len(subscribers) != len(ready_fds):
        raise RuntimeError("subscriber and readiness descriptor counts differ")
    pending = set(range(len(subscribers)))
    deadline = time.monotonic() + timeout_sec
    with selectors.DefaultSelector() as selector:
        for index, ready_fd in enumerate(ready_fds):
            selector.register(ready_fd, selectors.EVENT_READ, index)
        while pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"subscribers {sorted(pending)} did not signal readiness within "
                    f"{timeout_sec} seconds"
                )
            events = selector.select(remaining)
            if not events:
                continue
            for key, _ in events:
                index = key.data
                marker = os.read(key.fd, 1)
                if marker != b"R":
                    returncode = subscribers[index].poll()
                    if marker == b"" and returncode is None:
                        with contextlib.suppress(subprocess.TimeoutExpired):
                            returncode = subscribers[index].wait(timeout=1.0)
                    if skip_if_no_cuda and returncode == CUDA_UNAVAILABLE_EXIT_CODE:
                        print("CUDA is unavailable; skipping benchmark", file=sys.stderr)
                        raise SystemExit(CUDA_UNAVAILABLE_EXIT_CODE)
                    raise RuntimeError(
                        f"invalid readiness marker from subscriber {index}; returncode={returncode}"
                    )
                selector.unregister(key.fd)
                pending.remove(index)


def checked_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid role result {path}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"role result {path} is not a JSON object")
    return value


def start_process(
    command: list[str], output: IO[str], error: IO[str], pass_fds: tuple[int, ...] = ()
) -> subprocess.Popen[str]:
    return subprocess.Popen(
        command,
        stdout=output,
        stderr=error,
        text=True,
        pass_fds=pass_fds,
        start_new_session=True,
    )


def wait_processes(processes: list[subprocess.Popen[str]], timeout_sec: int) -> None:
    deadline = time.monotonic() + timeout_sec
    while any(process.poll() is None for process in processes):
        failures = [process.returncode for process in processes if process.poll() not in (None, 0)]
        if failures:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(f"benchmark timed out after {timeout_sec} seconds")
        time.sleep(0.01)


def run_case(
    args: argparse.Namespace, service_name: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    publisher_json = args.output_dir / "publisher.json"
    publisher_log = args.output_dir / "publisher.log"
    processes: list[subprocess.Popen[str]] = []
    subscribers: list[subprocess.Popen[str]] = []
    ready_read_fds: list[int] = []
    ready_write_fds: list[int] = []
    try:
        with contextlib.ExitStack() as stack:
            for index, cpu in enumerate(args.subscriber_cpu_values):
                subscriber_output = stack.enter_context(
                    (args.output_dir / f"subscriber_{index}.json").open("w", encoding="utf-8")
                )
                subscriber_error = stack.enter_context(
                    (args.output_dir / f"subscriber_{index}.log").open("w", encoding="utf-8")
                )
                ready_read_fd, ready_write_fd = os.pipe()
                ready_read_fds.append(ready_read_fd)
                ready_write_fds.append(ready_write_fd)
                subscriber = start_process(
                    command_for(
                        args,
                        "subscriber",
                        service_name,
                        cpu,
                        subscriber_index=index,
                        ready_fd=ready_write_fd,
                    ),
                    subscriber_output,
                    subscriber_error,
                    (ready_write_fd,),
                )
                subscribers.append(subscriber)
                processes.append(subscriber)
                os.close(ready_write_fd)
                ready_write_fds[-1] = -1
            wait_ready(subscribers, ready_read_fds, args.timeout_sec, args.skip_if_no_cuda)
            with (
                publisher_json.open("w", encoding="utf-8") as publisher_output,
                publisher_log.open("w", encoding="utf-8") as publisher_error,
            ):
                publisher = start_process(
                    command_for(args, "publisher", service_name, args.publisher_cpu),
                    publisher_output,
                    publisher_error,
                )
                processes.append(publisher)
                wait_processes([publisher, *subscribers], args.timeout_sec)
    finally:
        for ready_write_fd in ready_write_fds:
            if ready_write_fd >= 0:
                os.close(ready_write_fd)
        for ready_read_fd in ready_read_fds:
            os.close(ready_read_fd)
        terminate(processes)

    failures = [process.returncode for process in processes if process.returncode != 0]
    if failures:
        if (
            args.skip_if_no_cuda
            and CUDA_UNAVAILABLE_EXIT_CODE in failures
            and all(
                returncode in (CUDA_UNAVAILABLE_EXIT_CODE, -signal.SIGTERM)
                for returncode in failures
            )
        ):
            print("CUDA is unavailable; skipping benchmark", file=sys.stderr)
            raise SystemExit(CUDA_UNAVAILABLE_EXIT_CODE)
        raise RuntimeError(f"benchmark process failures: {failures}; logs in {args.output_dir}")
    return checked_json(publisher_json), [
        checked_json(args.output_dir / f"subscriber_{index}.json")
        for index in range(args.subscriber_count)
    ]


def validate_results(
    args: argparse.Namespace, publisher: dict[str, Any], subscribers: list[dict[str, Any]]
) -> dict[str, Any]:
    if len(subscribers) != args.subscriber_count:
        raise RuntimeError(
            f"received {len(subscribers)} subscriber results; expected {args.subscriber_count}"
        )
    expected_payload_bytes = args.payload_bytes
    expected_benchmark = "holomq_inline_transport"
    expected_storage = "inline"
    expected_clock = {
        "source": (
            "clock_gettime(CLOCK_MONOTONIC)"
            if sys.platform == "linux"
            else "std::chrono::steady_clock"
        ),
        "steady": True,
        "scope": "same_host_cross_process" if sys.platform == "linux" else "process_local",
    }
    if args.payload == "buffer_handle":
        expected_payload_bytes = publisher.get("payload_bytes")
        expected_benchmark = "holomq_buffer_handle_transport"
        expected_storage = args.storage
        if not isinstance(expected_payload_bytes, int) or expected_payload_bytes <= 0:
            raise RuntimeError("publisher is missing a valid BufferHandle control payload size")
    for role, result in (
        ("publisher", publisher),
        *(("subscriber", subscriber) for subscriber in subscribers),
    ):
        if result.get("role") != role:
            raise RuntimeError(f"expected {role} result, got {result.get('role')!r}")
        for key, expected in (
            ("schema_version", 1),
            ("benchmark", expected_benchmark),
            ("payload", args.payload),
            ("storage", expected_storage),
            ("payload_bytes", expected_payload_bytes),
            ("samples", args.samples),
            ("warmup", args.warmup),
            ("queue_depth", args.queue_depth),
            ("subscriber_count", args.subscriber_count),
            ("workload", args.workload),
            ("publisher_period_us", args.publisher_period_us),
            ("publisher_pacing", args.publisher_pacing),
            ("rate_limited", args.workload == "latency"),
            ("event_wake", False),
            ("clock", expected_clock),
            ("payload_validation", args.payload_validation),
            (
                "validation_timing",
                {
                    "none": "none",
                    "final": "after_timed_operations",
                    "all": "between_timed_operations",
                }[args.payload_validation],
            ),
        ):
            if result.get(key) != expected:
                raise RuntimeError(f"{role} {key} mismatch: {result.get(key)!r} != {expected!r}")
        if args.payload == "buffer_handle":
            for key, expected in (
                ("buffer_bytes", args.buffer_bytes),
                ("pool_slots", args.pool_slots),
                (
                    "device_data_work",
                    "preinitialized" if args.storage == "device" else "none",
                ),
                ("device_sync", args.device_sync),
            ):
                if result.get(key) != expected:
                    raise RuntimeError(
                        f"{role} {key} mismatch: {result.get(key)!r} != {expected!r}"
                    )
        if (
            role == "subscriber"
            and args.payload == "buffer_handle"
            and result.get("validation_errors") != 0
        ):
            raise RuntimeError("buffer subscriber reported validation errors")
        if role == "subscriber" and args.payload == "buffer_handle":
            final_validation_ns = result.get("final_validation_ns")
            if (
                args.payload_validation == "final"
                and (
                    not isinstance(final_validation_ns, int)
                    or isinstance(final_validation_ns, bool)
                    or final_validation_ns < 0
                )
            ) or (args.payload_validation != "final" and final_validation_ns is not None):
                raise RuntimeError("buffer subscriber reported invalid final-validation timing")
    for index, subscriber in enumerate(subscribers):
        if subscriber.get("subscriber_index") != index:
            raise RuntimeError(f"subscriber {index} result has the wrong index")
        if args.payload_validation != "none" and publisher.get("checksum") != subscriber.get(
            "checksum"
        ):
            raise RuntimeError(f"publisher and subscriber {index} checksums differ")
    backpressured_loans = publisher.get("backpressured_loans")
    if not isinstance(backpressured_loans, int) or backpressured_loans < 0:
        raise RuntimeError("publisher is missing a valid backpressured-loan count")
    connection_probe_backpressured_loans = publisher.get("connection_probe_backpressured_loans")
    if (
        not isinstance(connection_probe_backpressured_loans, int)
        or connection_probe_backpressured_loans < 0
    ):
        raise RuntimeError("publisher is missing a valid connection-probe backpressured-loan count")

    measured_start_ns = publisher.get("measured_start_ns")
    publisher_end_ns = publisher.get("measured_end_ns")
    subscriber_end_ns = [subscriber.get("measured_end_ns") for subscriber in subscribers]
    if not all(
        isinstance(value, int) and value > 0
        for value in (measured_start_ns, publisher_end_ns, *subscriber_end_ns)
    ):
        raise RuntimeError("missing valid measured timestamps")
    publisher_seconds = (publisher_end_ns - measured_start_ns) / 1_000_000_000
    completion_seconds = (max(subscriber_end_ns) - measured_start_ns) / 1_000_000_000
    if publisher_seconds <= 0 or completion_seconds <= 0:
        raise RuntimeError("nonpositive measured duration")

    completed_messages_per_second = args.samples / completion_seconds
    buffer_handle = args.payload == "buffer_handle"
    payload_write_profile = "full_payload_per_sample"
    bulk_work_profile = "not_applicable"
    if buffer_handle:
        bulk_work_profile = "descriptor_only"
        payload_write_profile = (
            "reused_preinitialized_contents"
            if args.storage == "device"
            else "boundary_markers_per_sample"
        )
    combined = {
        "schema_version": 1,
        "benchmark": expected_benchmark,
        "domain": "interprocess",
        "workload": args.workload,
        "payload": args.payload,
        "storage": expected_storage,
        "device_sync": args.device_sync,
        "payload_bytes": expected_payload_bytes,
        "application_payload_bytes": (
            args.buffer_bytes if buffer_handle else expected_payload_bytes
        ),
        "inline_transport_bytes": expected_payload_bytes,
        "external_buffer_bytes": args.buffer_bytes if buffer_handle else 0,
        "payload_write_profile": payload_write_profile,
        "bulk_work_profile": bulk_work_profile,
        "wake": {
            "event_wake": False,
            "busy_poll": False,
            "receive_polling": True,
        },
        "runtime_support_threads": "none",
        "payload_validation": args.payload_validation,
        "validation_timing": {
            "none": "none",
            "final": "after_timed_operations",
            "all": "between_timed_operations",
        }[args.payload_validation],
        "samples": args.samples,
        "warmup": args.warmup,
        "queue_policy": "blocking",
        "queue_depth": args.queue_depth,
        "subscriber_count": args.subscriber_count,
        "publisher_cpu": args.publisher_cpu,
        "subscriber_cpus": args.subscriber_cpu_values,
        "publisher_period_us": args.publisher_period_us,
        "publisher_pacing": args.publisher_pacing,
        "rate_limited": args.workload == "latency",
        "clock": expected_clock,
        "publisher_samples_per_second": args.samples / publisher_seconds,
        "completed_messages_per_second": completed_messages_per_second,
        "completed_deliveries_per_second": completed_messages_per_second * args.subscriber_count,
        "worst_subscriber_latency_p50_ns": max(
            subscriber["latency_p50_ns"] for subscriber in subscribers
        ),
        "worst_subscriber_latency_p99_ns": max(
            subscriber["latency_p99_ns"] for subscriber in subscribers
        ),
        "worst_subscriber_latency_p999_ns": (
            max(subscriber["latency_p999_ns"] for subscriber in subscribers)
            if args.samples >= MINIMUM_P999_SAMPLES
            else None
        ),
        "worst_subscriber_latency_max_ns": max(
            subscriber["latency_max_ns"] for subscriber in subscribers
        ),
        "publisher": publisher,
        "subscribers": subscribers,
        "publication_counts": {
            "scope": "measured_plus_warmup",
            "attempted": args.samples + args.warmup + backpressured_loans,
            "accepted": args.samples + args.warmup,
            "backpressured": backpressured_loans,
            "indeterminate": 0,
            "completed": args.samples + args.warmup,
        },
        "connection_probe_backpressured_loans": connection_probe_backpressured_loans,
    }
    if args.payload == "buffer_handle":
        combined.update(
            {
                "buffer_bytes": args.buffer_bytes,
                "pool_slots": args.pool_slots,
                "cold_first_import_ns": subscribers[0]["cold_first_import_ns"],
                "buffer_acquire_p50_ns": publisher["buffer_acquire_p50_ns"],
                "buffer_acquire_p99_ns": publisher["buffer_acquire_p99_ns"],
                "buffer_export_p50_ns": publisher["buffer_export_p50_ns"],
                "buffer_export_p99_ns": publisher["buffer_export_p99_ns"],
                "buffer_import_p50_ns": subscribers[0]["buffer_import_p50_ns"],
                "buffer_import_p99_ns": subscribers[0]["buffer_import_p99_ns"],
                "final_validation_ns": subscribers[0]["final_validation_ns"],
            }
        )
    return combined


def summarize_results(
    args: argparse.Namespace, combined_results: list[dict[str, Any]]
) -> dict[str, Any]:
    if len(combined_results) != args.repetitions:
        raise RuntimeError(
            f"received {len(combined_results)} combined results; expected {args.repetitions}"
        )
    metric_names = (
        "publisher_samples_per_second",
        "completed_messages_per_second",
        "completed_deliveries_per_second",
        "worst_subscriber_latency_p50_ns",
        "worst_subscriber_latency_p99_ns",
        "worst_subscriber_latency_max_ns",
    )
    if args.samples >= MINIMUM_P999_SAMPLES:
        metric_names += ("worst_subscriber_latency_p999_ns",)
    if args.payload == "buffer_handle":
        metric_names += (
            "cold_first_import_ns",
            "buffer_acquire_p50_ns",
            "buffer_acquire_p99_ns",
            "buffer_export_p50_ns",
            "buffer_export_p99_ns",
            "buffer_import_p50_ns",
            "buffer_import_p99_ns",
        )
        if args.payload_validation == "final":
            metric_names += ("final_validation_ns",)
    metrics: dict[str, dict[str, float | int | None]] = {}
    for name in metric_names:
        values = [float(result[name]) for result in combined_results]
        mean = statistics.fmean(values)
        metrics[name] = {
            "count": len(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
            "mean": mean,
            "coefficient_of_variation": (
                statistics.stdev(values) / mean if len(values) > 1 and mean != 0.0 else None
            ),
        }
    summary = {
        "schema_version": 1,
        "benchmark": combined_results[0]["benchmark"],
        "domain": "interprocess",
        "workload": args.workload,
        "payload": args.payload,
        "storage": combined_results[0]["storage"],
        "device_sync": args.device_sync,
        "preset": args.preset,
        "repetitions": args.repetitions,
        "payload_bytes": combined_results[0]["payload_bytes"],
        "application_payload_bytes": combined_results[0]["application_payload_bytes"],
        "inline_transport_bytes": combined_results[0]["inline_transport_bytes"],
        "external_buffer_bytes": combined_results[0]["external_buffer_bytes"],
        "payload_write_profile": combined_results[0]["payload_write_profile"],
        "bulk_work_profile": combined_results[0]["bulk_work_profile"],
        "wake": combined_results[0]["wake"],
        "runtime_support_threads": combined_results[0]["runtime_support_threads"],
        "payload_validation": args.payload_validation,
        "validation_timing": {
            "none": "none",
            "final": "after_timed_operations",
            "all": "between_timed_operations",
        }[args.payload_validation],
        "samples": args.samples,
        "warmup": args.warmup,
        "queue_policy": "blocking",
        "queue_depth": args.queue_depth,
        "subscriber_count": args.subscriber_count,
        "publisher_cpu": args.publisher_cpu,
        "subscriber_cpus": args.subscriber_cpu_values,
        "publisher_period_us": args.publisher_period_us,
        "publisher_pacing": args.publisher_pacing,
        "rate_limited": args.workload == "latency",
        "clock": combined_results[0]["clock"],
        "connection_probe_backpressured_loans": sum(
            result["connection_probe_backpressured_loans"] for result in combined_results
        ),
        "metrics": metrics,
        "runs": [
            {
                "repetition": result["repetition"],
                "publication_counts": result["publication_counts"],
                "connection_probe_backpressured_loans": result[
                    "connection_probe_backpressured_loans"
                ],
                **{name: result[name] for name in metric_names},
            }
            for result in combined_results
        ],
    }
    if args.payload == "buffer_handle":
        summary["buffer_bytes"] = args.buffer_bytes
        summary["pool_slots"] = args.pool_slots
    return summary


def main() -> int:
    args = parse_args()
    cpu_topology = collect_cpu_topology(args)
    args.binary = args.binary.resolve()
    args.output_dir = args.output_dir.resolve()
    if not args.binary.is_file():
        raise FileNotFoundError(f"benchmark binary does not exist: {args.binary}")
    combined_results: list[dict[str, Any]] = []
    for repetition in range(args.repetitions):
        repetition_args = copy.copy(args)
        if args.repetitions > 1:
            repetition_args.output_dir = args.output_dir / f"repetition_{repetition:03d}"
        service_name = f"holomq_transport_{os.getpid()}_{time.monotonic_ns()}_{repetition}"
        publisher, subscribers = run_case(repetition_args, service_name)
        combined = validate_results(repetition_args, publisher, subscribers)
        combined["repetition"] = repetition
        (repetition_args.output_dir / "combined.json").write_text(
            json.dumps(combined, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        combined_results.append(combined)
    summary = summarize_results(args, combined_results)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": 1,
        "command": sys.argv,
        "hostname": platform.node(),
        "kernel": platform.release(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "binary": str(args.binary),
        "preset": args.preset,
        "repetitions": args.repetitions,
        "cpu_topology": cpu_topology,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
