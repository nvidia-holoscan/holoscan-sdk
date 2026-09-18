# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run matched producer/consumer workers for a distributed Runtime Graph."""

from __future__ import annotations

import argparse
import json
import math
import os
import select
import signal
import subprocess
import sys
import tempfile
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from runtime_graph_runner_common import parse_integer_list
from runtime_graph_runner_common import summarize_values as summarize_common_values

MAXIMUM_PHASE_TRACE_SAMPLES = 100_000
WORKER_FAILURE_LOG_LIMIT = 8_192
DEFAULT_TENSOR_BYTES = 1 << 20
MAXIMUM_TENSOR_BYTES = 1 << 30
MULTI_TENSOR_PAYLOADS = frozenset({"multi_tensor_loan", "multi_tensor_staging"})
TENSOR_PAYLOADS = frozenset({"tensor", "image", *MULTI_TENSOR_PAYLOADS})
SCHEMA_PAYLOADS = frozenset({"schema", "image", *MULTI_TENSOR_PAYLOADS})
MAXIMUM_DESCRIPTOR_WARMUP_ATTEMPTS = 3


class DescriptorWarmupIncompleteError(RuntimeError):
    """Signal that an otherwise valid measurement needs a fresh warmup attempt."""


def parse_cpu_list(text: str) -> list[int]:
    return parse_integer_list(
        text,
        invalid_message="CPU lists must contain integer ordinals",
        empty_message="CPU lists must not be empty",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark_binary", type=Path)
    parser.add_argument("--topology", choices=("fanout", "chain"), default="fanout")
    parser.add_argument("--workload", choices=("latency", "saturated"), default="latency")
    parser.add_argument(
        "--payload",
        choices=("pod", "schema", "tensor", "image", *sorted(MULTI_TENSOR_PAYLOADS)),
        default="pod",
    )
    parser.add_argument(
        "--tensor-storage",
        choices=("pageable_host", "pinned_host", "device"),
        default="pageable_host",
    )
    parser.add_argument("--tensor-bytes", type=int, default=DEFAULT_TENSOR_BYTES)
    parser.add_argument("--tensor-device", type=int, default=0)
    parser.add_argument(
        "--bulk-work-profile",
        choices=("full_write", "descriptor_only"),
        default="full_write",
    )
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--warmup", type=int, default=1_000)
    parser.add_argument("--fanout", type=int, choices=range(1, 4), default=1)
    parser.add_argument("--queue-depth", type=int, default=1)
    parser.add_argument("--publisher-period-us", type=int, default=1_000)
    parser.add_argument("--payload-validation", choices=("none", "final", "all"), default="final")
    parser.add_argument("--producer-cpu", type=int, default=-1)
    parser.add_argument("--relay-cpu", type=int, default=-1)
    parser.add_argument("--consumer-cpu", type=int, default=-1)
    parser.add_argument("--consumer-cpus", type=parse_cpu_list)
    parser.add_argument("--producer-controller-cpu", type=int, default=-1)
    parser.add_argument("--relay-controller-cpu", type=int, default=-1)
    parser.add_argument("--relay-pump-cpu", type=int, default=-1)
    parser.add_argument("--consumer-controller-cpu", type=int, default=-1)
    parser.add_argument("--consumer-controller-cpus", type=parse_cpu_list)
    parser.add_argument("--consumer-pump-cpu", type=int, default=-1)
    parser.add_argument("--consumer-pump-cpus", type=parse_cpu_list)
    parser.add_argument("--busy-poll", action="store_true")
    parser.add_argument("--runtime-phase-trace", action="store_true")
    parser.add_argument("--session-timeout-ms", type=int, default=30_000)
    parser.add_argument("--process-timeout-sec", type=float, default=90.0)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args(argv)


def normalize_cpu_list(args: argparse.Namespace, singular_name: str, plural_name: str) -> list[int]:
    singular = getattr(args, singular_name)
    plural = getattr(args, plural_name, None)
    if plural is not None:
        if singular != -1:
            raise ValueError(
                f"--{singular_name.replace('_', '-')} cannot be combined with "
                f"--{plural_name.replace('_', '-')}"
            )
        if len(plural) != args.fanout:
            raise ValueError(
                f"--{plural_name.replace('_', '-')} must provide exactly one CPU per consumer"
            )
        return plural
    if args.fanout > 1 and singular >= 0:
        raise ValueError(f"--{plural_name.replace('_', '-')} is required to pin fanout consumers")
    return [singular] * args.fanout


def validate_args(args: argparse.Namespace) -> None:
    if os.name != "posix":
        raise ValueError("distributed Runtime Graph benchmark requires a POSIX process host")
    if args.samples <= 0:
        raise ValueError("--samples must be positive")
    if args.warmup < 0:
        raise ValueError("--warmup must be nonnegative")
    if args.runtime_phase_trace and args.samples + args.warmup > MAXIMUM_PHASE_TRACE_SAMPLES:
        raise ValueError("runtime phase trace exceeds the bounded diagnostic sample count")
    if args.queue_depth <= 0:
        raise ValueError("--queue-depth must be positive")
    if args.workload == "latency" and args.queue_depth not in {1, 2}:
        raise ValueError("latency workload requires --queue-depth 1 or 2")
    if args.workload == "latency" and args.publisher_period_us <= 0:
        raise ValueError("latency workload requires a positive --publisher-period-us")
    if args.workload == "saturated" and args.publisher_period_us != 0:
        raise ValueError("saturated workload requires --publisher-period-us 0")
    if args.topology == "chain" and args.fanout != 1:
        raise ValueError("chain topology requires --fanout 1")
    if args.payload in TENSOR_PAYLOADS and (args.topology != "fanout" or args.fanout != 1):
        raise ValueError(
            "Tensor-backed payloads currently require --topology fanout and --fanout 1"
        )
    if args.payload == "schema" and (args.topology != "fanout" or args.fanout != 1):
        raise ValueError("schema payload currently requires --topology fanout and --fanout 1")
    if args.payload in TENSOR_PAYLOADS and not 0 < args.tensor_bytes <= MAXIMUM_TENSOR_BYTES:
        raise ValueError("--tensor-bytes exceeds the bounded Tensor range")
    if args.tensor_device < 0:
        raise ValueError("--tensor-device must be nonnegative")
    if args.payload not in TENSOR_PAYLOADS and (
        args.tensor_storage != "pageable_host"
        or args.tensor_bytes != DEFAULT_TENSOR_BYTES
        or args.tensor_device != 0
        or args.bulk_work_profile != "full_write"
    ):
        raise ValueError("Tensor storage options require a Tensor-backed payload")
    if args.payload in MULTI_TENSOR_PAYLOADS and args.warmup == 0:
        raise ValueError("multi-Tensor benchmark payloads require warmup for allocation evidence")
    if args.payload == "multi_tensor_staging" and args.bulk_work_profile == "full_write":
        raise ValueError("multi-Tensor staging requires --bulk-work-profile descriptor_only")
    args.consumer_cpus = normalize_cpu_list(args, "consumer_cpu", "consumer_cpus")
    args.consumer_controller_cpus = normalize_cpu_list(
        args, "consumer_controller_cpu", "consumer_controller_cpus"
    )
    args.consumer_pump_cpus = normalize_cpu_list(args, "consumer_pump_cpu", "consumer_pump_cpus")
    requested_cpus = [
        args.producer_cpu,
        args.producer_controller_cpu,
        *args.consumer_cpus,
        *args.consumer_controller_cpus,
        *args.consumer_pump_cpus,
    ]
    if args.topology == "chain":
        requested_cpus.extend((args.relay_cpu, args.relay_controller_cpu, args.relay_pump_cpu))
    elif any(cpu >= 0 for cpu in (args.relay_cpu, args.relay_controller_cpu, args.relay_pump_cpu)):
        raise ValueError("relay CPU placement requires --topology chain")
    if any(cpu < -1 for cpu in requested_cpus):
        raise ValueError("CPU values must be -1 or nonnegative")
    pinned_cpus = [cpu for cpu in requested_cpus if cpu >= 0]
    if len(pinned_cpus) != len(set(pinned_cpus)):
        raise ValueError("all requested lane and support-thread CPUs must be distinct")
    if args.session_timeout_ms <= 0 or args.process_timeout_sec <= 0:
        raise ValueError("timeouts must be positive")
    if args.workload == "latency":
        nominal_paced_duration_ms = (
            (args.samples + args.warmup) * args.publisher_period_us + 999
        ) // 1000
        if args.session_timeout_ms < 2 * nominal_paced_duration_ms:
            raise ValueError(
                "--session-timeout-ms must provide at least twice the nominal paced duration"
            )
    if args.process_timeout_sec * 1000 < 3 * args.session_timeout_ms:
        raise ValueError("--process-timeout-sec must cover three worker session-timeout intervals")
    if args.repetitions <= 0:
        raise ValueError("--repetitions must be positive")


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(1, int(len(ordered) * quantile + 0.999999)) - 1
    return ordered[min(index, len(ordered) - 1)]


def summarize_values(values: list[float]) -> dict[str, float | int | None]:
    return summarize_common_values(values, median_key="p50")


def checked_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"failed to read benchmark result {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"benchmark result {path} is not a JSON object")
    return value


def expected_benchmark_name(args: argparse.Namespace) -> str:
    if args.payload == "tensor":
        return (
            f"runtime_graph_interprocess_tensor_1to1_{args.tensor_storage}_{args.bulk_work_profile}"
        )
    if args.payload == "image":
        return (
            f"runtime_graph_interprocess_image_1to1_{args.tensor_storage}_{args.bulk_work_profile}"
        )
    if args.payload == "schema":
        return "runtime_graph_interprocess_schema_time_1to1"
    if args.payload in MULTI_TENSOR_PAYLOADS:
        return (
            f"runtime_graph_interprocess_{args.payload}_1to1_"
            f"{args.tensor_storage}_{args.bulk_work_profile}"
        )
    if args.topology == "chain":
        return "runtime_graph_interprocess_pod_source_relay_sink"
    return f"runtime_graph_interprocess_pod_1to{args.fanout}"


def valid_ingress_latency_bucket(value: object, required: bool) -> bool:
    """Return whether one emitted ingress latency bucket matches its applicability."""
    if not isinstance(value, dict):
        return False
    percentiles = ("pMin", "p50", "p99", "pMax")
    if required:
        return all(
            isinstance(value.get(percentile), int)
            and not isinstance(value.get(percentile), bool)
            and value[percentile] >= 0
            for percentile in percentiles
        )
    return all(value.get(percentile) is None for percentile in (*percentiles, "p999"))


def valid_optional_ingress_latency_bucket(value: object) -> bool:
    """Return whether one conditionally populated ingress latency bucket is well formed."""
    return valid_ingress_latency_bucket(value, True) or valid_ingress_latency_bucket(value, False)


INTERPROCESS_RUNTIME_PHASE_COUNTER_NAMES = (
    "readiness_scans",
    "no_progress_scans",
    "wait_calls",
    "wake_signals",
    "wake_returns",
    "timeout_returns",
    "channel_samples_received",
    "channel_readiness_published",
    "readiness_poll_dispatches",
    "compute_dispatches",
)


def validate_runtime_phases(
    role: str, value: object, enabled: bool, counts: dict[str, Any]
) -> None:
    """Validate the bounded interprocess Runtime phase-counter payload."""
    if not enabled:
        if value != {"enabled": False}:
            raise RuntimeError(f"{role} reported invalid Runtime phase-counter state")
        return
    if not isinstance(value, dict) or value.get("enabled") is not True:
        raise RuntimeError(f"{role} reported invalid Runtime phase-counter state")
    if any(type(value.get(name)) is not int for name in INTERPROCESS_RUNTIME_PHASE_COUNTER_NAMES):
        raise RuntimeError(f"{role} reported incomplete Runtime phase counters")
    if any(value[name] < 0 for name in INTERPROCESS_RUNTIME_PHASE_COUNTER_NAMES):
        raise RuntimeError(f"{role} reported negative Runtime phase counters")
    expected_compute = (
        counts.get("source_callbacks")
        if role == "producer"
        else counts.get("relay_callbacks")
        if role == "relay"
        else counts.get("deliveries")
    )
    expected_inputs = (
        0
        if role == "producer"
        else counts.get("relay_inputs")
        if role == "relay"
        else counts.get("deliveries")
    )
    if (
        type(expected_compute) is not int
        or type(expected_inputs) is not int
        or value["compute_dispatches"] != expected_compute
        or value["channel_samples_received"] != expected_inputs
        or value["channel_readiness_published"] != expected_inputs
        or value["wake_returns"] + value["timeout_returns"] != value["wait_calls"]
        or value["no_progress_scans"] > value["readiness_scans"]
    ):
        raise RuntimeError(f"{role} reported inconsistent Runtime phase counters")


def validate_runtime_thread_usage(role: str, value: object, enabled: bool) -> None:
    """Validate opt-in Runtime-role CPU and context-switch evidence."""
    if not isinstance(value, dict) or value.get("enabled") is not enabled:
        raise RuntimeError(f"{role} reported invalid Runtime thread-usage state")
    if not enabled:
        return
    records = value.get("roles")
    if not isinstance(records, list) or len(records) != 4:
        raise RuntimeError(f"{role} reported invalid Runtime thread-usage roles")
    by_role = {record.get("role"): record for record in records if isinstance(record, dict)}
    expected_roles = {"controller", "execution_lanes", "channel_pump", "notification_driver"}
    if set(by_role) != expected_roles:
        raise RuntimeError(f"{role} reported invalid Runtime thread-usage roles")
    fields = (
        "samples",
        "user_cpu_ns",
        "system_cpu_ns",
        "voluntary_context_switches",
        "involuntary_context_switches",
        "minor_page_faults",
        "major_page_faults",
    )
    supported = value.get("supported")
    if not isinstance(supported, bool):
        raise RuntimeError(f"{role} reported invalid Runtime thread-usage support")
    for record in by_role.values():
        if record.get("supported") is not supported or any(
            not isinstance(record.get(field), int)
            or isinstance(record.get(field), bool)
            or record[field] < 0
            for field in fields
        ):
            raise RuntimeError(f"{role} reported invalid Runtime thread-usage counters")
    expected_samples = {
        "controller": 1,
        "execution_lanes": 0,
        "channel_pump": 0 if role == "producer" else 1,
        "notification_driver": 0,
    }
    if any(
        by_role[name]["samples"] != (count if supported else 0)
        for name, count in expected_samples.items()
    ):
        raise RuntimeError(f"{role} reported invalid Runtime thread-usage samples")
    aggregate = value.get("aggregate")
    if value.get("scope") != "whole_runtime_thread_lifetime_including_startup_and_teardown" or (
        not isinstance(aggregate, dict) or aggregate.get("scope") != "sum_of_runtime_roles"
    ):
        raise RuntimeError(f"{role} reported inconsistent Runtime thread-usage aggregate")
    for field in fields:
        aggregate_value = aggregate.get(field)
        if (
            type(aggregate_value) is not int
            or aggregate_value < 0
            or aggregate_value != sum(record[field] for record in by_role.values())
        ):
            raise RuntimeError(f"{role} reported inconsistent Runtime thread-usage aggregate")


def validate_results(
    args: argparse.Namespace,
    producer: dict[str, Any],
    relay: dict[str, Any] | None,
    consumers: list[dict[str, Any]],
) -> None:
    expected_total = args.samples + args.warmup
    if len(consumers) != args.fanout:
        raise RuntimeError("consumer result count does not match fanout")
    role_records = [("producer", producer)]
    if args.topology == "chain":
        if relay is None:
            raise RuntimeError("chain result is missing its relay")
        role_records.append(("relay", relay))
    elif relay is not None:
        raise RuntimeError("fanout result unexpectedly contains a relay")
    role_records.extend(
        (
            "consumer" if args.fanout == 1 else f"consumer-{index + 1}",
            consumer,
        )
        for index, consumer in enumerate(consumers)
    )
    producer_geometry = producer.get("geometry")
    if not isinstance(producer_geometry, dict):
        raise RuntimeError("producer is missing benchmark geometry")
    pool_slots = producer_geometry.get("pool_slots")
    expected_benchmark = expected_benchmark_name(args)
    for role, record in role_records:
        if record.get("schema") != "holoscan.test.runtime-interprocess-benchmark/v1":
            raise RuntimeError(f"{role} reported an unexpected schema")
        if record.get("role") != role:
            raise RuntimeError(f"{role} result reported the wrong role")
        if record.get("workload") != args.workload:
            raise RuntimeError(f"{role} reported the wrong workload")
        if record.get("payload") != args.payload:
            raise RuntimeError(f"{role} reported the wrong payload")
        if record.get("benchmark") != expected_benchmark:
            raise RuntimeError(f"{role} reported the wrong benchmark identity")
        if record.get("termination") != "stopped" or record.get("diagnostic_count") != 0:
            raise RuntimeError(f"{role} did not stop without diagnostics")
    for key in ("run_id", "graph_hash", "plan_digest", "service_ids"):
        expected = producer.get(key)
        if not expected or any(record.get(key) != expected for _, record in role_records[1:]):
            raise RuntimeError(f"distributed workers disagree on {key}")
    service_ids = producer["service_ids"]
    # Compatible fanout arms share one compiler-owned broadcast service. A chain retains one
    # service per cross-process hop.
    expected_service_count = 2 if args.topology == "chain" else 1
    if (
        not isinstance(service_ids, list)
        or len(service_ids) != expected_service_count
        or len(set(service_ids)) != expected_service_count
    ):
        raise RuntimeError("compiled service identities do not match fanout")
    expected_clock = {
        "source": "clock_gettime(CLOCK_MONOTONIC)",
        "steady": True,
        "scope": "same_host_cross_process",
    }
    if any(record.get("clock") != expected_clock for _, record in role_records):
        raise RuntimeError("workers did not prove one same-host steady clock domain")
    for role, record in role_records:
        instrumentation = record.get("timestamp_instrumentation")
        pair_overhead = (
            instrumentation.get("pair_overhead_ns") if isinstance(instrumentation, dict) else None
        )
        if (
            not isinstance(instrumentation, dict)
            or instrumentation.get("calibration") != "post_measurement_back_to_back_pair"
            or instrumentation.get("clock_reads_per_sample") != 2
            or not isinstance(pair_overhead, dict)
            or not isinstance(pair_overhead.get("p50"), int)
        ):
            raise RuntimeError(f"{role} reported invalid timestamp calibration evidence")
    process_ids = [record.get("process_id") for _, record in role_records]
    process_groups = [record.get("process_group") for _, record in role_records]
    if len(set(process_ids)) != len(process_ids):
        raise RuntimeError("workers did not execute in distinct processes")
    if len(set(process_groups)) != len(process_groups):
        raise RuntimeError("workers selected duplicate process groups")

    for role, record in role_records:
        if role == "producer":
            expected_support_cpus = {
                "controller": args.producer_controller_cpu,
                "channel_pump": -1,
            }
        elif role == "relay":
            expected_support_cpus = {
                "controller": args.relay_controller_cpu,
                "channel_pump": args.relay_pump_cpu,
            }
        else:
            consumer_index = int(record.get("consumer_index", 0)) - 1
            if consumer_index < 0 or consumer_index >= args.fanout:
                raise RuntimeError(f"{role} reported an invalid consumer index")
            expected_support_cpus = {
                "controller": args.consumer_controller_cpus[consumer_index],
                "channel_pump": args.consumer_pump_cpus[consumer_index],
            }
        evidence = record.get("support_threads")
        if not isinstance(evidence, list) or len(evidence) != 2:
            raise RuntimeError(f"{role} reported invalid support-thread evidence")
        by_role = {item.get("role"): item for item in evidence if isinstance(item, dict)}
        if set(by_role) != {"controller", "channel_pump"}:
            raise RuntimeError(f"{role} reported invalid support-thread roles")
        for support_role, item in by_role.items():
            expected_entered = support_role == "controller" or role != "producer"
            requested_cpu = expected_support_cpus[support_role]
            if (
                item.get("requested_cpu") != (requested_cpu if requested_cpu >= 0 else None)
                or item.get("entered") is not expected_entered
                or item.get("error") is not None
            ):
                raise RuntimeError(f"{role} reported invalid {support_role} evidence")
            if expected_entered and (
                item.get("affinity_supported") is not True
                or not isinstance(item.get("native_tid"), int)
                or item["native_tid"] <= 0
            ):
                raise RuntimeError(f"{role} did not observe its {support_role} thread")
            if (
                expected_entered
                and requested_cpu >= 0
                and item.get("observed_affinity") != [requested_cpu]
            ):
                raise RuntimeError(f"{role} did not apply its {support_role} CPU")
        validate_runtime_thread_usage(
            role, record.get("runtime_thread_usage"), args.runtime_phase_trace
        )
        counts = record.get("counts", {})
        validate_runtime_phases(
            role, record.get("runtime_phases"), args.runtime_phase_trace, counts
        )
        callback_evidence = record.get("callback_thread_evidence")
        if (
            not isinstance(callback_evidence, dict)
            or callback_evidence.get("enabled") is not args.runtime_phase_trace
        ):
            raise RuntimeError(f"{role} reported invalid callback-owner evidence state")
        if args.runtime_phase_trace:
            expected_callback_count = (
                counts.get("source_callbacks")
                if role == "producer"
                else counts.get("relay_callbacks")
                if role == "relay"
                else counts.get("deliveries")
            )
            if (
                callback_evidence.get("valid") is not True
                or callback_evidence.get("thread_identity_kind") != "native_tid"
                or callback_evidence.get("native_tid_syscall_unavailable") is not False
                or callback_evidence.get("controller_native_tid")
                != by_role["controller"].get("native_tid")
                or callback_evidence.get("callback_role") != record.get("role")
                or callback_evidence.get("callback_count") != expected_callback_count
                or callback_evidence.get("callback_native_tid")
                != by_role["controller"].get("native_tid")
                or callback_evidence.get("multiple_threads") is not False
                or callback_evidence.get("callbacks_on_partition_owner") is not True
            ):
                raise RuntimeError(f"{role} reported invalid callback-owner evidence")
        trace = record.get("ingress_wake_trace")
        expected_trace = args.runtime_phase_trace and role != "producer"
        if not isinstance(trace, dict) or trace.get("enabled") is not expected_trace:
            raise RuntimeError(f"{role} reported invalid ingress-wake trace state")
        if expected_trace and (
            trace.get("valid") is not True
            or not isinstance(trace.get("readiness_poll_elided"), bool)
            or trace.get("dropped") != 0
            or trace.get("samples") != args.samples
            or trace.get("transport_wait_samples") != args.samples
            or not isinstance(trace.get("pump_wake_samples"), int)
            or trace.get("pump_wake_samples") < 0
            or trace.get("pump_wake_samples") > args.samples
            or trace.get("parked_wake_samples", 0) + trace.get("active_controller_samples", 0)
            != args.samples
        ):
            raise RuntimeError("consumer reported incomplete ingress-wake trace evidence")
        if expected_trace:
            readiness_poll_elided = trace["readiness_poll_elided"]
            readiness_latency_buckets_valid = (
                valid_ingress_latency_bucket(trace.get("pump_wake_to_readiness_poll_ns"), False)
                and valid_ingress_latency_bucket(
                    trace.get("controller_return_to_readiness_poll_ns"), False
                )
                and valid_ingress_latency_bucket(trace.get("readiness_poll_to_candidate_ns"), False)
                and valid_ingress_latency_bucket(
                    trace.get("readiness_poll_to_compute_dispatch_ns"), False
                )
                if readiness_poll_elided
                else valid_optional_ingress_latency_bucket(
                    trace.get("pump_wake_to_readiness_poll_ns")
                )
                and valid_ingress_latency_bucket(
                    trace.get("controller_return_to_readiness_poll_ns"),
                    trace["parked_wake_samples"] > 0,
                )
                and valid_ingress_latency_bucket(trace.get("readiness_poll_to_candidate_ns"), True)
                and valid_ingress_latency_bucket(
                    trace.get("readiness_poll_to_compute_dispatch_ns"), True
                )
            )
            if (
                not valid_ingress_latency_bucket(trace.get("transport_wait_to_ingress_ns"), True)
                or not valid_ingress_latency_bucket(
                    trace.get("compute_dispatch_to_callback_entry_ns"), True
                )
                or not valid_ingress_latency_bucket(
                    trace.get("transport_wait_to_callback_entry_ns"), True
                )
                or not valid_ingress_latency_bucket(
                    trace.get("ingress_to_pump_wake_ns"), trace["pump_wake_samples"] > 0
                )
                or not valid_ingress_latency_bucket(
                    trace.get("pump_wake_to_controller_return_ns"), trace["parked_wake_samples"] > 0
                )
                or not valid_ingress_latency_bucket(
                    trace.get("controller_return_to_candidate_ns"),
                    trace["parked_wake_samples"] > 0,
                )
                or not valid_ingress_latency_bucket(
                    trace.get("candidate_to_compute_dispatch_ns"), True
                )
                or not readiness_latency_buckets_valid
            ):
                raise RuntimeError("consumer reported invalid ingress-wake latency buckets")
        expected_wake_mode = "busy_poll" if args.busy_poll else "sleeping_waitset"
        if record.get("placement", {}).get("lane_wake_mode") != expected_wake_mode:
            raise RuntimeError(f"{role} reported the wrong lane wake mode")
        expected_wake = {
            "event_wake": role != "producer",
            "busy_poll": args.busy_poll,
            "receive_polling": False,
        }
        if record.get("wake") != expected_wake:
            raise RuntimeError(f"{role} reported unexpected wake or polling evidence")
        expected_validation_timing = {
            "none": "none",
            "final": "after_timed_operations",
            "all": "between_timed_operations",
        }[args.payload_validation]
        if (
            record.get("payload_validation") != args.payload_validation
            or record.get("validation_timing") != expected_validation_timing
        ):
            raise RuntimeError(f"{role} reported unexpected payload-validation evidence")

    producer_counts = producer.get("counts", {})
    if (
        producer_counts.get("source_starts") != 1
        or producer_counts.get("relay_starts", 0) != 0
        or producer_counts.get("consumer_starts") != 0
        or producer_counts.get("publications") != expected_total
        or producer_counts.get("relay_inputs", 0) != 0
        or producer_counts.get("relay_publications", 0) != 0
        or producer_counts.get("deliveries") != 0
    ):
        raise RuntimeError("producer role-isolation or publication evidence is invalid")
    if relay is not None:
        relay_counts = relay.get("counts", {})
        if (
            relay_counts.get("source_starts") != 0
            or relay_counts.get("relay_starts") != 1
            or relay_counts.get("consumer_starts") != 0
            or relay_counts.get("publications") != 0
            or relay_counts.get("relay_inputs") != expected_total
            or relay_counts.get("relay_publications") != expected_total
            or relay_counts.get("deliveries") != 0
            or relay_counts.get("final_sequence") != expected_total
        ):
            raise RuntimeError("relay role-isolation or forwarding evidence is invalid")
    for index, consumer in enumerate(consumers):
        consumer_counts = consumer.get("counts", {})
        if (
            consumer.get("consumer_index") != index + 1
            or consumer_counts.get("source_starts") != 0
            or consumer_counts.get("relay_starts", 0) != 0
            or consumer_counts.get("consumer_starts") != 1
            or consumer_counts.get("publications") != 0
            or consumer_counts.get("relay_inputs", 0) != 0
            or consumer_counts.get("relay_publications", 0) != 0
            or consumer_counts.get("deliveries") != expected_total
            or consumer_counts.get("final_sequence") != expected_total
        ):
            raise RuntimeError(
                f"consumer-{index + 1} role-isolation or delivery evidence is invalid"
            )

    expected_processes = 3 if args.topology == "chain" else args.fanout + 1
    expected_transport_hops = 2 if args.topology == "chain" else 1
    expected_write_profile = (
        (
            "full_buffer_write_per_sample"
            if args.bulk_work_profile == "full_write"
            else "preinitialized_immutable_backing"
        )
        if args.payload in TENSOR_PAYLOADS
        else "full_payload_per_sample"
        if args.payload == "schema"
        else "value_fill"
    )
    if (
        producer_geometry.get("samples") != args.samples
        or producer_geometry.get("warmup") != args.warmup
        or producer_geometry.get("topology") != args.topology
        or producer_geometry.get("fanout") != args.fanout
        or producer_geometry.get("processes") != expected_processes
        or producer_geometry.get("transport_hops") != expected_transport_hops
        or producer_geometry.get("notification_capacity") != args.queue_depth
        or producer_geometry.get("publication_backpressure_capacity") != args.queue_depth
        or producer_geometry.get("payload_write_profile") != expected_write_profile
        or producer_geometry.get("bulk_work_profile")
        != (args.bulk_work_profile if args.payload in TENSOR_PAYLOADS else "not_applicable")
        or any(consumer.get("geometry") != producer_geometry for consumer in consumers)
        or (relay is not None and relay.get("geometry") != producer_geometry)
    ):
        raise RuntimeError("workers disagree on benchmark geometry")
    application_bytes = producer_geometry.get("application_payload_bytes")
    inline_bytes = producer_geometry.get("inline_transport_bytes")
    external_bytes = producer_geometry.get("external_buffer_bytes")
    release_capacity = producer_geometry.get("release_capacity")
    if (
        not isinstance(application_bytes, int)
        or isinstance(application_bytes, bool)
        or application_bytes <= 0
        or producer_geometry.get("payload_bytes") != application_bytes
        or not isinstance(external_bytes, int)
        or isinstance(external_bytes, bool)
        or external_bytes < 0
    ):
        raise RuntimeError("workers reported invalid payload geometry")
    if args.payload in TENSOR_PAYLOADS:
        expected_application_bytes = (
            args.tensor_bytes * 2 if args.payload in MULTI_TENSOR_PAYLOADS else args.tensor_bytes
        )
        if (
            application_bytes != expected_application_bytes
            or external_bytes != application_bytes
            or (
                args.payload not in MULTI_TENSOR_PAYLOADS
                and (
                    not isinstance(inline_bytes, int)
                    or isinstance(inline_bytes, bool)
                    or inline_bytes <= 0
                )
            )
            or (args.payload in MULTI_TENSOR_PAYLOADS and inline_bytes is not None)
            or not isinstance(pool_slots, int)
            or isinstance(pool_slots, bool)
            or pool_slots <= 0
            or not isinstance(release_capacity, int)
            or isinstance(release_capacity, bool)
            or release_capacity < pool_slots
        ):
            raise RuntimeError("workers reported invalid Tensor-backed geometry")
    elif external_bytes != 0 or pool_slots is not None or release_capacity is not None:
        raise RuntimeError("inline workers unexpectedly reported external buffer bytes")
    elif args.payload == "pod" and inline_bytes != application_bytes:
        raise RuntimeError("POD workers reported inconsistent inline payload bytes")
    elif args.payload == "schema" and inline_bytes is not None:
        raise RuntimeError("schema workers unexpectedly reported a fixed inline payload size")
    schema_evidence = producer.get("generated_schema")
    if args.payload in SCHEMA_PAYLOADS:
        expected_native_type = {
            "schema": "schema::TimeT",
            "image": "schema::ImageT",
            "multi_tensor_loan": "runtime_benchmark::MultiTensorMessageT",
            "multi_tensor_staging": "runtime_benchmark::MultiTensorMessageT",
        }[args.payload]
        if (
            not isinstance(schema_evidence, dict)
            or schema_evidence.get("native_type") != expected_native_type
            or not isinstance(schema_evidence.get("wire_capacity"), int)
            or schema_evidence["wire_capacity"] <= 0
            or (args.payload == "image" and schema_evidence["wire_capacity"] != inline_bytes)
            or any(
                record.get("generated_schema") != schema_evidence for _, record in role_records[1:]
            )
        ):
            raise RuntimeError("workers reported invalid generated-schema evidence")
    elif any(record.get("generated_schema") != {} for _, record in role_records):
        raise RuntimeError("non-schema workers unexpectedly reported generated-schema evidence")
    if args.payload in TENSOR_PAYLOADS:
        expected_tensor = {
            "storage": args.tensor_storage,
            "device": args.tensor_device,
            "descriptor_schema": args.payload != "tensor",
            "read_only_import_checked": True,
            "capture_timestamp_in_sample_metadata": True,
            "direct_allocation_records": 2 if args.payload in MULTI_TENSOR_PAYLOADS else 1,
            "memory_kind_plan_match": True,
            "service_pool_bound": True,
        }
        tensor_evidence_valid = True
        for _, record in role_records:
            tensor = record.get("tensor")
            tensor_evidence_valid = (
                tensor_evidence_valid
                and isinstance(tensor, dict)
                and all(tensor.get(key) == value for key, value in expected_tensor.items())
            )
            if args.payload in MULTI_TENSOR_PAYLOADS and isinstance(tensor, dict):
                fields = tensor.get("fields")
                pool_ids: list[str] = []
                fields_valid = isinstance(fields, list) and len(fields) == 2
                if fields_valid:
                    for index, field in enumerate(fields):
                        pool_id = field.get("pool_id") if isinstance(field, dict) else None
                        fields_valid = fields_valid and (
                            isinstance(field, dict)
                            and field.get("name") == ("first" if index == 0 else "second")
                            and field.get("dtype") == "uint8"
                            and field.get("rank") == 1
                            and field.get("bytes") == args.tensor_bytes
                            and field.get("tensor_index") == index
                            and isinstance(pool_id, str)
                            and bool(pool_id)
                            and isinstance(field.get("pool_slots"), int)
                            and not isinstance(field.get("pool_slots"), bool)
                            and field["pool_slots"] > 0
                            and isinstance(field.get("release_capacity"), int)
                            and not isinstance(field.get("release_capacity"), bool)
                            and field["release_capacity"] > 0
                        )
                        if isinstance(pool_id, str):
                            pool_ids.append(pool_id)
                tensor_evidence_valid = tensor_evidence_valid and (
                    tensor.get("field_pools_distinct") is True
                    and fields_valid
                    and len(set(pool_ids)) == 2
                )
        if not tensor_evidence_valid:
            raise RuntimeError("workers reported invalid Tensor plan or import evidence")
        descriptor_initializations = producer_counts.get("descriptor_backing_initializations", 0)
        measured_initializations = producer_counts.get(
            "descriptor_backing_initializations_during_measurement", 0
        )
        if args.bulk_work_profile == "descriptor_only":
            if descriptor_initializations <= 0:
                raise RuntimeError(
                    "descriptor-only Tensor producer reported no backing initialization "
                    f"(total={descriptor_initializations}, "
                    f"during_measurement={measured_initializations}, "
                    f"warmup={args.warmup}, pool_slots={pool_slots})"
                )
            # A first-touch callback initializes and cancels its unpublished loan. It therefore
            # contributes no latency sample, but it would consume time inside a saturated
            # throughput interval and must remain disallowed there.
            if measured_initializations != 0 and args.workload == "saturated":
                raise DescriptorWarmupIncompleteError(
                    "descriptor-only Tensor producer initialized backing during measurement "
                    f"(total={descriptor_initializations}, "
                    f"during_measurement={measured_initializations}, "
                    f"warmup={args.warmup}, pool_slots={pool_slots})"
                )
            # The compiled pool also contains bounded failure-recovery reserves. Qualification
            # warms only the producer loan plus the queued and borrowed steady-state retention
            # sets; forcing every reserve into circulation can exhaust readiness capacity.
            # Paced latency may first-touch and cancel the retirement slot outside any published
            # sample. Saturated throughput must condition that slot before starting its wall-clock
            # interval, in addition to the active loan and queued/borrowed retention sets.
            retirement_slots = 1 if args.workload == "saturated" else 0
            expected_hot_slots = min(pool_slots, 1 + 2 * args.queue_depth + retirement_slots)
            if (
                args.payload in {"tensor", "image"}
                and descriptor_initializations < expected_hot_slots
            ):
                raise DescriptorWarmupIncompleteError(
                    "descriptor-only Tensor producer did not initialize the hot-path inventory "
                    f"(total={descriptor_initializations}, "
                    f"during_measurement={measured_initializations}, "
                    f"warmup={args.warmup}, hot_slots={expected_hot_slots}, "
                    f"pool_slots={pool_slots})"
                )
            if args.payload in {"tensor", "image"} and descriptor_initializations > pool_slots:
                raise RuntimeError(
                    "descriptor-only Tensor producer initialized more than the declared pool "
                    f"inventory (total={descriptor_initializations}, pool_slots={pool_slots})"
                )
            if (
                args.payload == "multi_tensor_loan"
                and args.workload == "saturated"
                and descriptor_initializations != pool_slots
            ):
                raise DescriptorWarmupIncompleteError(
                    "descriptor-only multi-Tensor producer did not initialize the saturated "
                    "field-pool inventory "
                    f"(total={descriptor_initializations}, "
                    f"during_measurement={measured_initializations}, "
                    f"warmup={args.warmup}, pool_slots={pool_slots})"
                )
        elif descriptor_initializations != 0 or measured_initializations != 0:
            raise RuntimeError("full-write Tensor row reported descriptor-only initialization")
    elif any(record.get("tensor") != {} for _, record in role_records):
        raise RuntimeError("non-Tensor workers unexpectedly reported Tensor evidence")
    if args.payload in SCHEMA_PAYLOADS:
        probe_enabled = args.warmup != 0
        expected_access = "selected_batch" if args.payload in MULTI_TENSOR_PAYLOADS else "receive"
        expected_source_limit = (
            (1 if probe_enabled and args.payload == "multi_tensor_staging" else 0)
            if probe_enabled
            else None
        )
        expected_sink_limit = (
            6 if probe_enabled and args.payload == "image" else 0 if probe_enabled else None
        )
        for role, record in role_records:
            probe = record.get("allocation_probe")
            source_allocations = probe.get("source_compute_allocations") if probe else None
            sink_allocations = probe.get("sink_compute_allocations") if probe else None
            source_valid = (
                isinstance(source_allocations, int)
                and not isinstance(source_allocations, bool)
                and source_allocations >= 0
                if role == "producer" and probe_enabled
                else source_allocations is None
            )
            sink_valid = (
                isinstance(sink_allocations, int)
                and not isinstance(sink_allocations, bool)
                and sink_allocations >= 0
                if role == "consumer" and probe_enabled
                else sink_allocations is None
            )
            if (
                not isinstance(probe, dict)
                or probe.get("sequence") != (args.warmup if probe_enabled else None)
                or probe.get("excluded_from_measurement") is not probe_enabled
                or probe.get("sink_access") != expected_access
                or probe.get("source_compute_allocation_limit") != expected_source_limit
                or probe.get("sink_compute_allocation_limit") != expected_sink_limit
                or not source_valid
                or not sink_valid
            ):
                raise RuntimeError(f"{role} reported invalid schema allocation evidence")
            if (source_allocations is not None and source_allocations > expected_source_limit) or (
                sink_allocations is not None and sink_allocations > expected_sink_limit
            ):
                raise RuntimeError(f"{role} exceeded its schema callback allocation limit")
    if relay is not None:
        relay_timing = relay.get("timing", {})
        if (
            relay_timing.get("measured_start_ns", 0) <= 0
            or relay_timing.get("measured_end_ns", 0) <= relay_timing.get("measured_start_ns", 0)
            or relay_timing.get("messages_per_second", 0.0) <= 0.0
        ):
            raise RuntimeError("relay reported invalid forwarding timing")
    for index, consumer in enumerate(consumers):
        consumer_timing = consumer.get("timing", {})
        if (
            consumer_timing.get("measured_start_ns", 0) <= 0
            or consumer_timing.get("measured_end_ns", 0)
            <= consumer_timing.get("measured_start_ns", 0)
            or consumer_timing.get("messages_per_second", 0.0) <= 0.0
        ):
            raise RuntimeError(f"consumer-{index + 1} reported invalid end-to-end timing")


def terminate(processes: list[subprocess.Popen[str]]) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()
    deadline = time.monotonic() + 2.0
    for process in processes:
        while process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
    for process in processes:
        if process.poll() is None:
            process.kill()


def worker_failure_details(log_paths: list[Path]) -> str:
    details: list[str] = []
    for path in log_paths:
        try:
            with path.open("rb") as stream:
                stream.seek(0, os.SEEK_END)
                size = stream.tell()
                start = max(0, size - WORKER_FAILURE_LOG_LIMIT)
                stream.seek(start)
                text = stream.read(WORKER_FAILURE_LOG_LIMIT).decode("utf-8", errors="replace")
        except OSError as error:
            details.append(f"--- {path.name} unavailable: {error}")
            continue
        marker = "[... earlier output truncated ...]\n" if start != 0 else ""
        details.append(f"--- {path.name}\n{marker}{text.rstrip()}")
    return "\n".join(details)


def worker_result_failure_details(result_paths: list[Path]) -> str:
    details: list[str] = []
    for path in result_paths:
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, RecursionError):
            continue
        if not isinstance(result, dict):
            continue
        summary = {
            "role": result.get("role"),
            "termination": result.get("termination"),
            "diagnostic_count": result.get("diagnostic_count"),
            "diagnostics": result.get("diagnostics"),
            "counts": result.get("counts"),
            "runtime_thread_usage": result.get("runtime_thread_usage"),
            "support_threads": result.get("support_threads"),
        }
        details.append(
            f"--- {path.name} diagnostic result\n"
            + json.dumps(summary, separators=(",", ":"), sort_keys=True)
        )
    return "\n".join(details)


def common_command(args: argparse.Namespace, run_id: str) -> list[str]:
    start_timeout_ms = min((1 << 32) - 1, max(1, math.ceil(args.process_timeout_sec * 1000)))
    return [
        str(args.benchmark_binary),
        "--run-id",
        run_id,
        "--topology",
        args.topology,
        "--fanout",
        str(args.fanout),
        "--workload",
        args.workload,
        "--payload",
        args.payload,
        "--tensor-storage",
        args.tensor_storage,
        "--tensor-bytes",
        str(args.tensor_bytes),
        "--tensor-device",
        str(args.tensor_device),
        "--bulk-work-profile",
        args.bulk_work_profile,
        "--samples",
        str(args.samples),
        "--warmup",
        str(args.warmup),
        "--queue-depth",
        str(args.queue_depth),
        "--publisher-period-us",
        str(args.publisher_period_us),
        "--payload-validation",
        args.payload_validation,
        "--producer-cpu",
        str(args.producer_cpu),
        "--relay-cpu",
        str(args.relay_cpu),
        "--consumer-cpus",
        ",".join(str(cpu) for cpu in args.consumer_cpus),
        "--producer-controller-cpu",
        str(args.producer_controller_cpu),
        "--relay-controller-cpu",
        str(args.relay_controller_cpu),
        "--relay-pump-cpu",
        str(args.relay_pump_cpu),
        "--consumer-controller-cpus",
        ",".join(str(cpu) for cpu in args.consumer_controller_cpus),
        "--consumer-pump-cpus",
        ",".join(str(cpu) for cpu in args.consumer_pump_cpus),
        "--busy-poll",
        "1" if args.busy_poll else "0",
        "--runtime-phase-trace",
        "1" if args.runtime_phase_trace else "0",
        "--timeout-ms",
        str(args.session_timeout_ms),
        "--start-timeout-ms",
        str(start_timeout_ms),
    ]


def release_producer_start(write_fd: int) -> None:
    try:
        written = os.write(write_fd, b"\x01")
    except BrokenPipeError as error:
        raise RuntimeError("producer exited before the start barrier was released") from error
    if written != 1:
        raise RuntimeError("producer start barrier accepted an incomplete release signal")


def run_repetition(
    args: argparse.Namespace, repetition: int, output_dir: Path, attempt: int = 0
) -> tuple[dict[str, Any], dict[str, Any] | None, list[dict[str, Any]]]:
    suffix = "" if attempt == 0 else f"_warmup_retry_{attempt:02d}"
    repetition_dir = output_dir / f"repetition_{repetition:03d}{suffix}"
    repetition_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"runtime_graph_interprocess_{os.getpid()}_{time.monotonic_ns()}"
    producer_path = repetition_dir / "producer.json"
    producer_log_path = repetition_dir / "producer.log"
    relay_path = repetition_dir / "relay.json"
    relay_log_path = repetition_dir / "relay.log"
    consumer_paths = [
        repetition_dir / f"consumer_{index + 1:03d}.json" for index in range(args.fanout)
    ]
    consumer_log_paths = [
        repetition_dir / f"consumer_{index + 1:03d}.log" for index in range(args.fanout)
    ]
    processes: list[subprocess.Popen[str]] = []
    open_fds: set[int] = set()

    try:
        with ExitStack() as stack:
            producer_log = stack.enter_context(producer_log_path.open("w", encoding="utf-8"))
            producer_ready_read, producer_ready_write = os.pipe()
            producer_start_read, producer_start_write = os.pipe()
            open_fds.update(
                (
                    producer_ready_read,
                    producer_ready_write,
                    producer_start_read,
                    producer_start_write,
                )
            )
            producer_command = [
                *common_command(args, run_id),
                "--role",
                "producer",
                "--ready-fd",
                str(producer_ready_write),
                "--start-fd",
                str(producer_start_read),
                "--output",
                str(producer_path),
            ]
            producer = subprocess.Popen(
                producer_command,
                stdout=producer_log,
                stderr=producer_log,
                text=True,
                pass_fds=(producer_ready_write, producer_start_read),
            )
            processes.append(producer)
            os.close(producer_ready_write)
            open_fds.remove(producer_ready_write)
            os.close(producer_start_read)
            open_fds.remove(producer_start_read)
            ready, _, _ = select.select([producer_ready_read], [], [], args.process_timeout_sec)
            if not ready or os.read(producer_ready_read, 1) != b"\x01":
                raise TimeoutError("producer did not report process-group readiness")
            os.close(producer_ready_read)
            open_fds.remove(producer_ready_read)

            if args.topology == "chain":
                relay_log = stack.enter_context(relay_log_path.open("w", encoding="utf-8"))
                read_fd, write_fd = os.pipe()
                open_fds.update((read_fd, write_fd))
                relay_command = [
                    *common_command(args, run_id),
                    "--role",
                    "relay",
                    "--ready-fd",
                    str(write_fd),
                    "--output",
                    str(relay_path),
                ]
                relay = subprocess.Popen(
                    relay_command,
                    stdout=relay_log,
                    stderr=relay_log,
                    text=True,
                    pass_fds=(write_fd,),
                )
                processes.append(relay)
                os.close(write_fd)
                open_fds.remove(write_fd)

                ready, _, _ = select.select([read_fd], [], [], args.process_timeout_sec)
                if not ready or os.read(read_fd, 1) != b"\x01":
                    raise TimeoutError("relay did not report process-group readiness")
                os.close(read_fd)
                open_fds.remove(read_fd)

            for index, (consumer_path, consumer_log_path) in enumerate(
                zip(consumer_paths, consumer_log_paths, strict=True)
            ):
                consumer_log = stack.enter_context(consumer_log_path.open("w", encoding="utf-8"))
                read_fd, write_fd = os.pipe()
                open_fds.update((read_fd, write_fd))
                consumer_command = [
                    *common_command(args, run_id),
                    "--role",
                    "consumer",
                    "--consumer-index",
                    str(index + 1),
                    "--ready-fd",
                    str(write_fd),
                    "--output",
                    str(consumer_path),
                ]
                consumer = subprocess.Popen(
                    consumer_command,
                    stdout=consumer_log,
                    stderr=consumer_log,
                    text=True,
                    pass_fds=(write_fd,),
                )
                processes.append(consumer)
                os.close(write_fd)
                open_fds.remove(write_fd)

                ready, _, _ = select.select([read_fd], [], [], args.process_timeout_sec)
                if not ready or os.read(read_fd, 1) != b"\x01":
                    raise TimeoutError(
                        f"consumer-{index + 1} did not report process-group readiness"
                    )
                os.close(read_fd)
                open_fds.remove(read_fd)

            release_producer_start(producer_start_write)
            os.close(producer_start_write)
            open_fds.remove(producer_start_write)

            deadline = time.monotonic() + args.process_timeout_sec
            while any(process.poll() is None for process in processes):
                failures = [
                    process.returncode for process in processes if process.poll() not in (None, 0)
                ]
                if failures:
                    break
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        f"benchmark repetition {repetition} timed out after "
                        f"{args.process_timeout_sec} s"
                    )
                time.sleep(0.02)
    finally:
        for fd in open_fds:
            os.close(fd)
        terminate(processes)

    failures = [process.returncode for process in processes if process.returncode != 0]
    if failures:
        if 77 in failures and all(code in (77, -signal.SIGTERM) for code in failures):
            raise SystemExit(77)
        worker_logs = [producer_log_path, *consumer_log_paths]
        worker_results = [producer_path, *consumer_paths]
        if args.topology == "chain":
            worker_logs.append(relay_log_path)
            worker_results.append(relay_path)
        failure_details = worker_failure_details(worker_logs)
        result_details = worker_result_failure_details(worker_results)
        if result_details:
            failure_details += "\n" + result_details
        raise RuntimeError(
            f"benchmark process failures {failures}; logs in {repetition_dir}\n{failure_details}"
        )

    producer = checked_json(producer_path)
    relay_result = checked_json(relay_path) if args.topology == "chain" else None
    consumers = [checked_json(path) for path in consumer_paths]
    validate_results(args, producer, relay_result, consumers)
    return producer, relay_result, consumers


def run_qualified_repetition(
    args: argparse.Namespace, repetition: int, output_dir: Path
) -> tuple[dict[str, Any], dict[str, Any] | None, list[dict[str, Any]]]:
    def run_attempt(
        attempt: int,
    ) -> tuple[dict[str, Any], dict[str, Any] | None, list[dict[str, Any]]]:
        try:
            return run_repetition(args, repetition, output_dir, attempt)
        except DescriptorWarmupIncompleteError as error:
            if attempt + 1 == MAXIMUM_DESCRIPTOR_WARMUP_ATTEMPTS:
                raise
            print(
                f"discarding benchmark repetition {repetition} after incomplete descriptor "
                f"warmup attempt {attempt + 1}: {error}",
                file=sys.stderr,
            )
            return run_attempt(attempt + 1)

    return run_attempt(0)


def write_summary(
    args: argparse,
    output_dir: Path,
    records: list[tuple[dict[str, Any], dict[str, Any] | None, list[dict[str, Any]]]],
) -> None:
    logical_publication_rates = [
        float(producer["timing"]["messages_per_second"]) for producer, _, _ in records
    ]
    slowest_completion_rates = [
        min(float(consumer["timing"]["messages_per_second"]) for consumer in consumers)
        for _, _, consumers in records
    ]
    aggregate_delivery_rates: list[float] = []
    slowest_latency_p50: list[float] = []
    slowest_latency_p99: list[float] = []
    slowest_latency_p999: list[float] = []
    for _, _, consumers in records:
        measured_start_ns = min(consumer["timing"]["measured_start_ns"] for consumer in consumers)
        measured_end_ns = max(consumer["timing"]["measured_end_ns"] for consumer in consumers)
        aggregate_delivery_rates.append(
            args.samples * args.fanout * 1.0e9 / (measured_end_ns - measured_start_ns)
        )
        if args.workload != "saturated":
            slowest_latency_p50.append(
                max(float(consumer["timing"]["end_to_end_ns"]["p50"]) for consumer in consumers)
            )
            slowest_latency_p99.append(
                max(float(consumer["timing"]["end_to_end_ns"]["p99"]) for consumer in consumers)
            )
            if args.samples >= 10_000:
                slowest_latency_p999.append(
                    max(
                        float(consumer["timing"]["end_to_end_ns"]["p999"]) for consumer in consumers
                    )
                )
    relay_records = [relay for _, relay, _ in records if relay is not None]
    result_records = []
    for producer, relay, consumers in records:
        result_records.append(producer)
        if relay is not None:
            result_records.append(relay)
        result_records.extend(consumers)
    runtime_support_threads = ",".join(
        role
        for role in ("controller", "channel_pump", "notification_driver")
        if any(
            isinstance(item, dict) and item.get("role") == role and item.get("entered") is True
            for result in result_records
            for item in result.get("support_threads", [])
        )
    )
    timestamp_pair_p50_values = [
        float(result["timestamp_instrumentation"]["pair_overhead_ns"]["p50"])
        for result in result_records
    ]
    summary = {
        "schema": "holoscan.test.runtime-interprocess-summary/v1",
        "benchmark": expected_benchmark_name(args),
        "payload": args.payload,
        "tensor_storage": (args.tensor_storage if args.payload in TENSOR_PAYLOADS else None),
        "tensor_bytes": args.tensor_bytes if args.payload in TENSOR_PAYLOADS else None,
        "tensor_device": args.tensor_device if args.payload in TENSOR_PAYLOADS else None,
        "bulk_work_profile": (
            args.bulk_work_profile if args.payload in TENSOR_PAYLOADS else "not_applicable"
        ),
        "topology": args.topology,
        "workload": args.workload,
        "fanout": args.fanout,
        "repetitions": len(records),
        "clock": result_records[0]["clock"],
        "timestamp_instrumentation": {
            "calibration": "post_measurement_back_to_back_pair",
            "clock_reads_per_sample": 2,
            "pair_overhead_p50_ns_across_processes_and_runs": summarize_values(
                timestamp_pair_p50_values
            ),
        },
        "wake": {
            "event_wake": True,
            "busy_poll": args.busy_poll,
            "receive_polling": False,
        },
        "runtime_support_threads": runtime_support_threads or "none",
        "payload_validation": args.payload_validation,
        "validation_timing": {
            "none": "none",
            "final": "after_timed_operations",
            "all": "between_timed_operations",
        }[args.payload_validation],
        "logical_publications_per_second": summarize_values(logical_publication_rates),
        "slowest_sink_messages_per_second": summarize_values(slowest_completion_rates),
        "aggregate_deliveries_per_second": summarize_values(aggregate_delivery_rates),
        "slowest_sink_end_to_end_p50_ns": (
            summarize_values(slowest_latency_p50) if slowest_latency_p50 else None
        ),
        "slowest_sink_end_to_end_p99_ns": (
            summarize_values(slowest_latency_p99) if slowest_latency_p99 else None
        ),
        "slowest_sink_end_to_end_p999_ns": (
            summarize_values(slowest_latency_p999) if slowest_latency_p999 else None
        ),
        "producer": [producer for producer, _, _ in records],
        "relay": relay_records,
        "consumers": [consumers for _, _, consumers in records],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    if sys.platform != "linux":
        print("SKIP: distributed Runtime Graph benchmark currently requires Linux", file=sys.stderr)
        return 77
    validate_args(args)
    args.benchmark_binary = args.benchmark_binary.resolve()
    if not args.benchmark_binary.exists():
        raise FileNotFoundError(args.benchmark_binary)

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="hsdk-runtime-interprocess-bench-"))
    output_dir.mkdir(parents=True, exist_ok=True)

    records = [
        run_qualified_repetition(args, repetition, output_dir)
        for repetition in range(args.repetitions)
    ]
    write_summary(args, output_dir, records)
    print(output_dir / "summary.json")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, TimeoutError, ValueError) as error:
        print(f"runtime interprocess benchmark failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
