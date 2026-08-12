#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run matched role-aware Channel execution-domain and transport benchmarks."""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark_binary", type=Path)
    parser.add_argument(
        "--domain",
        choices=(
            "intraprocess",
            "intraprocess_sidecar",
            "intraprocess_serial",
            "interprocess",
        ),
        default="interprocess",
    )
    parser.add_argument("--payload", choices=("pod", "tensor", "image"), default="pod")
    parser.add_argument("--workload", choices=("throughput", "latency"), default="throughput")
    parser.add_argument(
        "--tensor-storage",
        choices=("pageable_host", "pinned_host", "device"),
        default="pinned_host",
    )
    parser.add_argument("--samples", type=int, default=100_000)
    parser.add_argument("--warmup", type=int, default=1_000)
    parser.add_argument("--tensor-bytes", type=int, default=1 << 20)
    parser.add_argument("--subscriber-queue-depth", type=int, default=64)
    parser.add_argument("--publisher-period-us", type=int, default=0)
    parser.add_argument("--publisher-pacing", choices=("sleep", "spin"), default="sleep")
    parser.add_argument("--async-cuda-work", action="store_true")
    parser.add_argument("--payload-validation", choices=("none", "final", "all"), default="final")
    parser.add_argument("--expected-subscribers", type=int, default=1)
    parser.add_argument("--publisher-cpu", type=int, default=-1)
    parser.add_argument("--subscriber-cpu", type=int, default=-1)
    parser.add_argument("--skip-if-no-cuda", action="store_true")
    parser.add_argument(
        "--subscriber-cpus",
        default=None,
        help=(
            "Comma-separated subscriber CPU list for fanout runs. "
            "If omitted, --subscriber-cpu is used for every subscriber."
        ),
    )
    parser.add_argument("--timeout-sec", type=float, default=120.0)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--channel-name", default=None)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.domain == "intraprocess_serial" and args.expected_subscribers != 1:
        raise ValueError("intraprocess_serial requires --expected-subscribers 1")
    if args.domain == "intraprocess_serial" and args.payload != "tensor":
        raise ValueError("intraprocess_serial currently requires --payload tensor")
    if args.domain == "intraprocess_serial" and args.subscriber_cpu >= 0:
        raise ValueError(
            "intraprocess_serial uses --publisher-cpu and does not accept --subscriber-cpu"
        )
    if args.domain == "intraprocess_serial" and getattr(args, "subscriber_cpus", None) is not None:
        raise ValueError(
            "intraprocess_serial uses --publisher-cpu and does not accept --subscriber-cpus"
        )
    if args.workload == "latency" and args.publisher_period_us <= 0:
        raise ValueError("latency workload requires a nonzero --publisher-period-us")
    if args.workload == "latency" and args.subscriber_queue_depth not in {1, 2}:
        raise ValueError("latency workload requires --subscriber-queue-depth 1 or 2")
    if args.workload == "throughput" and args.publisher_period_us != 0:
        raise ValueError("--publisher-period-us requires --workload latency")
    if args.workload != "latency" and args.publisher_pacing == "spin":
        raise ValueError("--publisher-pacing spin requires --workload latency")
    if args.async_cuda_work and (args.payload != "tensor" or args.tensor_storage != "device"):
        raise ValueError("--async-cuda-work requires --payload tensor --tensor-storage device")
    if args.async_cuda_work and args.payload_validation == "all":
        raise ValueError(
            "--payload-validation all is incompatible with --async-cuda-work; "
            "use its deferred CUDA validation"
        )
    if args.async_cuda_work and args.domain in {
        "intraprocess_sidecar",
        "intraprocess_serial",
    }:
        raise ValueError("--async-cuda-work is not yet supported by sidecar benchmark domains")


def parse_cpu_list(text: str | None, count: int, fallback: int) -> list[int]:
    if text is None:
        return [fallback] * count
    cpus = [int(item.strip()) for item in text.split(",") if item.strip()]
    if len(cpus) < count:
        raise ValueError(
            f"--subscriber-cpus provided {len(cpus)} entries for {count} subscriber(s)"
        )
    return cpus[:count]


def read_cpu_topology(role: str, cpu: int) -> dict[str, object]:
    result: dict[str, object] = {"role": role, "cpu": cpu, "pinned": cpu >= 0}
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


def validate_distinct_physical_cores(cpu_topology: list[dict[str, object]]) -> None:
    physical_cores: dict[tuple[object, object], dict[str, object]] = {}
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


def collect_cpu_topology(
    args: argparse.Namespace, subscriber_cpus: list[int]
) -> list[dict[str, object]]:
    placements = [read_cpu_topology("publisher", args.publisher_cpu)]
    if args.domain != "intraprocess_serial":
        placements.extend(
            read_cpu_topology(f"subscriber_{index}", cpu)
            for index, cpu in enumerate(subscriber_cpus)
        )
    validate_distinct_physical_cores(placements)
    missing = [
        str(entry["role"])
        for entry in placements
        if entry["pinned"] and not entry.get("topology_available")
    ]
    if missing and sys.platform.startswith("linux"):
        print(
            f"warning: physical CPU topology unavailable for {', '.join(missing)}",
            file=sys.stderr,
        )
    return placements


def checked_json(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise RuntimeError(f"{path} was empty")
    return json.loads(text)


def expected_transport(args: argparse.Namespace) -> str:
    if args.payload == "pod":
        return "inline"
    return (
        "sidecar"
        if args.domain in {"intraprocess_sidecar", "intraprocess_serial"}
        else "buffer_handle"
    )


def expected_bulk_work_profile(args: argparse.Namespace) -> str:
    if args.payload == "pod":
        return "not_applicable"
    return "full_write" if args.async_cuda_work else "descriptor_only"


def validate_results(
    args: argparse.Namespace,
    publisher: dict[str, object],
    subscribers: list[dict[str, object]],
) -> None:
    expected_total = args.samples + args.warmup
    expected_deliveries = expected_total * args.expected_subscribers
    if len(subscribers) != args.expected_subscribers:
        raise RuntimeError(
            f"expected {args.expected_subscribers} subscriber result(s), got {len(subscribers)}"
        )
    if publisher.get("expected_subscribers") != args.expected_subscribers:
        raise RuntimeError("publisher reported an unexpected subscriber count")
    if publisher.get("domain") != args.domain:
        raise RuntimeError("publisher reported an unexpected execution domain")
    if publisher.get("transport") != expected_transport(args):
        raise RuntimeError("publisher reported an unexpected transport")
    if publisher.get("workload") != args.workload:
        raise RuntimeError("publisher reported an unexpected workload")
    if publisher.get("publisher_pacing") != args.publisher_pacing:
        raise RuntimeError("publisher reported an unexpected pacing mode")
    if publisher.get("bulk_work_profile") != expected_bulk_work_profile(args):
        raise RuntimeError("publisher reported an unexpected bulk work profile")
    expected_validation_timing = {
        "none": "none",
        "final": "after_timed_operations",
        "all": "between_timed_operations",
    }[args.payload_validation]
    expected_clock = {
        "source": (
            "clock_gettime(CLOCK_MONOTONIC)"
            if sys.platform == "linux"
            else "std::chrono::steady_clock"
        ),
        "steady": True,
        "scope": "same_host_cross_process" if sys.platform == "linux" else "process_local",
    }
    if (
        publisher.get("payload_validation") != args.payload_validation
        or publisher.get("validation_timing") != expected_validation_timing
        or publisher.get("clock") != expected_clock
    ):
        raise RuntimeError("publisher reported unexpected validation or clock evidence")
    expected_wake = {
        "event_wake": False,
        "busy_poll": False,
        "receive_polling": True,
    }
    if publisher.get("wake") != expected_wake:
        raise RuntimeError("publisher reported unexpected wake or polling evidence")
    expected_pool_slots = (
        None if args.payload == "pod" else max(256, args.subscriber_queue_depth + 8)
    )
    expected_release_capacity = expected_pool_slots
    expected_release_streams = None if args.payload == "pod" else args.expected_subscribers
    for role, result in (
        ("publisher", publisher),
        *[(f"subscriber {index}", subscriber) for index, subscriber in enumerate(subscribers)],
    ):
        if (
            result.get("pool_slots") != expected_pool_slots
            or result.get("release_capacity") != expected_release_capacity
            or result.get("publication_backpressure_capacity") != args.subscriber_queue_depth
            or result.get("max_fanout_release_streams") != expected_release_streams
        ):
            raise RuntimeError(f"{role} reported unexpected bounded-resource geometry")
        if result.get("bulk_work_profile") != expected_bulk_work_profile(args):
            raise RuntimeError(f"{role} reported an unexpected bulk work profile")
    if publisher.get("reached_total") != expected_deliveries:
        raise RuntimeError(
            f"publisher reached {publisher.get('reached_total')} total recipients; "
            f"expected {expected_deliveries}"
        )
    async_lease_acquisitions = publisher.get("async_lease_acquisitions")
    if args.async_cuda_work and (
        not isinstance(async_lease_acquisitions, int) or async_lease_acquisitions < expected_total
    ):
        raise RuntimeError(
            "publisher acquired "
            f"{async_lease_acquisitions} async Tensor leases; "
            f"expected at least one fresh lease for each of {expected_total} samples"
        )

    error_counter = "checksum_errors" if args.payload == "pod" else "validation_errors"
    for index, subscriber in enumerate(subscribers):
        if subscriber.get("domain") != args.domain:
            raise RuntimeError(f"subscriber {index} reported an unexpected execution domain")
        if subscriber.get("transport") != expected_transport(args):
            raise RuntimeError(f"subscriber {index} reported an unexpected transport")
        if subscriber.get("expected_subscribers") != args.expected_subscribers:
            raise RuntimeError(f"subscriber {index} reported an unexpected subscriber count")
        if subscriber.get("wake") != expected_wake:
            raise RuntimeError(f"subscriber {index} reported unexpected wake or polling evidence")
        if (
            subscriber.get("payload_validation") != args.payload_validation
            or subscriber.get("validation_timing") != expected_validation_timing
            or subscriber.get("clock") != expected_clock
        ):
            raise RuntimeError(
                f"subscriber {index} reported unexpected validation or clock evidence"
            )
        if subscriber.get("received_count") != expected_total:
            raise RuntimeError(
                f"subscriber {index} received {subscriber.get('received_count')} samples; "
                f"expected {expected_total}"
            )
        if subscriber.get("sequence_gaps") != 0:
            raise RuntimeError(
                f"subscriber {index} reported {subscriber.get('sequence_gaps')} sequence gaps"
            )
        if subscriber.get(error_counter) != 0:
            raise RuntimeError(
                f"subscriber {index} reported {subscriber.get(error_counter)} "
                f"{error_counter.replace('_', ' ')}"
            )
        if args.payload in {"tensor", "image"}:
            final_validation_ns = subscriber.get("final_validation_ns")
            if (
                args.payload_validation == "final"
                and (
                    not isinstance(final_validation_ns, int)
                    or isinstance(final_validation_ns, bool)
                    or final_validation_ns < 0
                )
            ) or (args.payload_validation != "final" and final_validation_ns is not None):
                raise RuntimeError(f"subscriber {index} reported invalid final-validation timing")


def numeric_values(results: list[dict[str, object]], key: str) -> list[float]:
    values = [result.get(key) for result in results]
    return [float(value) for value in values if isinstance(value, (int, float))]


def fanout_summary(
    publisher: dict[str, object], subscribers: list[dict[str, object]]
) -> dict[str, float | int]:
    subscriber_count = len(subscribers)
    logical_rate_value = publisher.get("messages_per_second", 0.0)
    logical_rate = (
        float(logical_rate_value) if isinstance(logical_rate_value, (int, float)) else 0.0
    )
    summary: dict[str, float | int] = {
        "subscriber_count": subscriber_count,
        "logical_messages_per_second": logical_rate,
        "publisher_deliveries_per_second": logical_rate * subscriber_count,
    }
    measured_start = publisher.get("measured_start_ns")
    measured_ends = numeric_values(subscribers, "measured_end_ns")
    sample_count = publisher.get("samples")
    if (
        isinstance(measured_start, int)
        and subscriber_count > 0
        and len(measured_ends) == subscriber_count
        and isinstance(sample_count, int)
        and max(measured_ends) > measured_start
    ):
        completion_window_ns = int(max(measured_ends)) - measured_start
        completed_logical_rate = sample_count * 1.0e9 / completion_window_ns
        summary.update(
            {
                "completion_window_ns": completion_window_ns,
                "completed_logical_messages_per_second": completed_logical_rate,
                "completed_deliveries_per_second": completed_logical_rate * subscriber_count,
            }
        )
    metrics = {
        "subscriber_messages_per_second_min": "messages_per_second",
        "subscriber_messages_per_second_max": "messages_per_second",
        "worst_subscriber_channel_receive_p50_ns": "channel_receive_p50_ns",
        "worst_subscriber_channel_receive_p99_ns": "channel_receive_p99_ns",
        "worst_subscriber_channel_receive_p999_ns": "channel_receive_p999_ns",
        "worst_subscriber_end_to_end_p50_ns": "end_to_end_p50_ns",
        "worst_subscriber_end_to_end_p99_ns": "end_to_end_p99_ns",
        "worst_subscriber_end_to_end_p999_ns": "end_to_end_p999_ns",
    }
    for output_key, input_key in metrics.items():
        values = numeric_values(subscribers, input_key)
        if values:
            summary[output_key] = (
                min(values) if output_key == "subscriber_messages_per_second_min" else max(values)
            )
    return summary


def terminate(processes: list[subprocess.Popen[str]]) -> None:
    for proc in processes:
        if proc.poll() is None:
            proc.terminate()
    deadline = time.monotonic() + 2.0
    for proc in processes:
        while proc.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
    for proc in processes:
        if proc.poll() is None:
            proc.kill()


def run_intraprocess(
    args: argparse.Namespace,
    common: list[str],
    subscriber_cpus: list[int],
    output_dir: Path,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    command = [
        *common,
        "--role",
        "coordinator",
        "--publisher-cpu",
        str(args.publisher_cpu),
    ]
    if args.domain != "intraprocess_serial":
        for cpu in subscriber_cpus:
            command.extend(["--subscriber-cpu", str(cpu)])

    output_path = output_dir / "coordinator.jsonl"
    log_path = output_dir / "coordinator.log"
    try:
        with (
            output_path.open("w", encoding="utf-8") as output,
            log_path.open("w", encoding="utf-8") as log,
        ):
            completed = subprocess.run(
                command,
                stdout=output,
                stderr=log,
                text=True,
                timeout=args.timeout_sec,
                check=False,
            )
    except subprocess.TimeoutExpired as error:
        raise TimeoutError(f"benchmark timed out after {args.timeout_sec} s") from error

    if completed.returncode != 0:
        if args.skip_if_no_cuda and completed.returncode == 2:
            print("CUDA is unavailable; skipping benchmark", file=sys.stderr)
            raise SystemExit(77)
        raise RuntimeError(
            f"benchmark coordinator failed with {completed.returncode}; logs in {output_dir}"
        )

    records = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    publishers = [record for record in records if record.get("role") == "publisher"]
    subscribers = [record for record in records if record.get("role") == "subscriber"]
    if len(publishers) != 1:
        raise RuntimeError(f"coordinator returned {len(publishers)} publisher result(s)")
    if len(subscribers) != args.expected_subscribers:
        raise RuntimeError(
            f"coordinator returned {len(subscribers)} subscriber result(s); "
            f"expected {args.expected_subscribers}"
        )

    publisher = publishers[0]
    (output_dir / "publisher.json").write_text(
        json.dumps(publisher, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for index, subscriber in enumerate(subscribers):
        (output_dir / f"subscriber_{index}.json").write_text(
            json.dumps(subscriber, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return publisher, subscribers


def run_interprocess(
    args: argparse,
    common: list[str],
    subscriber_cpus: list[int],
    output_dir: Path,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    processes: list[subprocess.Popen[str]] = []
    try:
        for index in range(args.expected_subscribers):
            out = (output_dir / f"subscriber_{index}.json").open("w", encoding="utf-8")
            err = (output_dir / f"subscriber_{index}.log").open("w", encoding="utf-8")
            command = [*common, "--role", "subscriber", "--cpu", str(subscriber_cpus[index])]
            processes.append(subprocess.Popen(command, stdout=out, stderr=err, text=True))

        time.sleep(0.2)

        publisher_out = (output_dir / "publisher.json").open("w", encoding="utf-8")
        publisher_err = (output_dir / "publisher.log").open("w", encoding="utf-8")
        publisher_command = [*common, "--role", "publisher", "--cpu", str(args.publisher_cpu)]
        publisher = subprocess.Popen(
            publisher_command, stdout=publisher_out, stderr=publisher_err, text=True
        )
        processes.append(publisher)

        deadline = time.monotonic() + args.timeout_sec
        while any(process.poll() is None for process in processes):
            failed = [
                process.returncode for process in processes if process.poll() not in (None, 0)
            ]
            if failed:
                break
            if time.monotonic() > deadline:
                raise TimeoutError(f"benchmark timed out after {args.timeout_sec} s")
            time.sleep(0.05)
    finally:
        terminate(processes)

    failed = [process.returncode for process in processes if process.returncode != 0]
    if failed:
        if (
            args.skip_if_no_cuda
            and 2 in failed
            and all(returncode in (2, -signal.SIGTERM) for returncode in failed)
        ):
            print("CUDA is unavailable; skipping benchmark", file=sys.stderr)
            raise SystemExit(77)
        raise RuntimeError(f"benchmark process failures: {failed}; logs in {output_dir}")

    publisher_json = checked_json(output_dir / "publisher.json")
    subscriber_json = [
        checked_json(output_dir / f"subscriber_{index}.json")
        for index in range(args.expected_subscribers)
    ]
    return publisher_json, subscriber_json


def main() -> int:
    args = parse_args()
    validate_args(args)
    binary = args.benchmark_binary.resolve()
    if not binary.exists():
        raise FileNotFoundError(binary)

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix=f"hsdk-channel-{args.domain}-bench-"))
    output_dir.mkdir(parents=True, exist_ok=True)

    channel_name = args.channel_name
    if channel_name is None:
        channel_name = f"hsdk_channel_bench_{os.getpid()}_{time.monotonic_ns()}"
    subscriber_cpus = parse_cpu_list(
        args.subscriber_cpus, args.expected_subscribers, args.subscriber_cpu
    )
    cpu_topology = collect_cpu_topology(args, subscriber_cpus)

    common = [
        str(binary),
        "--domain",
        args.domain,
        "--channel-name",
        channel_name,
        "--payload",
        args.payload,
        "--workload",
        args.workload,
        "--tensor-storage",
        args.tensor_storage,
        "--samples",
        str(args.samples),
        "--warmup",
        str(args.warmup),
        "--tensor-bytes",
        str(args.tensor_bytes),
        "--subscriber-queue-depth",
        str(args.subscriber_queue_depth),
        "--publisher-period-us",
        str(args.publisher_period_us),
        "--publisher-pacing",
        args.publisher_pacing,
        "--expected-subscribers",
        str(args.expected_subscribers),
        "--payload-validation",
        args.payload_validation,
    ]
    if args.async_cuda_work:
        common.append("--async-cuda-work")

    if args.domain in {"intraprocess", "intraprocess_sidecar", "intraprocess_serial"}:
        publisher_json, subscriber_json = run_intraprocess(
            args, common, subscriber_cpus, output_dir
        )
    else:
        publisher_json, subscriber_json = run_interprocess(
            args, common, subscriber_cpus, output_dir
        )
    validate_results(args, publisher_json, subscriber_json)

    fanout = f"1to{args.expected_subscribers}"
    benchmark_name = f"channel_{args.domain}_{args.payload}_{fanout}"
    if args.payload in {"tensor", "image"}:
        benchmark_name = f"channel_{args.domain}_{args.payload}_{args.tensor_storage}_{fanout}"
    if args.async_cuda_work:
        benchmark_name += "_async_cuda"
    if args.workload == "latency":
        benchmark_name += "_latency"
        if args.publisher_pacing == "spin":
            benchmark_name += "_spin"

    result = {
        "benchmark": benchmark_name,
        "domain": args.domain,
        "workload": args.workload,
        "publisher_pacing": args.publisher_pacing,
        "output_dir": str(output_dir),
        "channel_name": channel_name,
        "hostname": socket.gethostname(),
        "publisher_cpu": args.publisher_cpu,
        "subscriber_cpus": subscriber_cpus,
        "cpu_topology": cpu_topology,
        "subscriber_count": args.expected_subscribers,
        "fanout": fanout_summary(publisher_json, subscriber_json),
        "publisher": publisher_json,
        "subscribers": subscriber_json,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
