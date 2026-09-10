#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Normalize one transport benchmark suite into concise JSON, CSV, and Markdown tables."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from collections.abc import Iterable
from contextlib import suppress
from pathlib import Path
from typing import Any

from transport_benchmark_identity import (
    make_scenario_identity,
    scenario_identity_contract,
    scenario_identity_key,
)

SUMMARY_SCHEMA = "holoscan.transport-benchmark-summary/v8"
METRICS = (
    "logical_messages_per_second",
    "aggregate_deliveries_per_second",
    "latency_p50_ns",
    "latency_p99_ns",
    "latency_p999_ns",
    "mean_iteration_ns",
    "timestamp_read_pair_p50_ns",
    "pool_acquisition_p50_ns",
    "import_materialization_p50_ns",
    "cuda_enqueue_p50_ns",
    "deferred_release_p50_ns",
    "final_validation_ns",
)
COUNT_FIELDS = (
    "publication_attempted_count",
    "publication_accepted_count",
    "publication_backpressured_count",
    "publication_indeterminate_count",
    "publication_completed_count",
)
IDENTITY_FIELDS = (
    "layer",
    "scope",
    "domain",
    "scenario_domain",
    "benchmark",
    "native_benchmark",
    "topology",
    "workload",
    "workload_class",
    "trigger_mechanism",
    "payload",
    "scenario_payload",
    "storage",
    "delivery_path",
    "execution_model",
    "synchronization",
    "wake_mode",
    "fanout",
    "relay_count",
    "queue_depth",
    "notification_capacity",
    "pool_slots",
    "release_capacity",
    "publication_backpressure_capacity",
    "max_fanout_release_streams",
    "process_count",
    "application_payload_bytes",
    "inline_transport_bytes",
    "external_buffer_bytes",
    "payload_write_profile",
    "bulk_work_profile",
    "event_wake",
    "busy_poll",
    "receive_polling",
    "runtime_support_threads",
    "measured_samples",
    "warmup_samples",
    "setup_samples",
    "publication_count_scope",
    "latency_anchor",
    "publisher_pacing",
    "comparison_period_us",
    "effective_period_us",
    "pacing_period_policy",
    "pacing_period_reason",
    "clock_source",
    "clock_scope",
    "clock_steady",
    "payload_validation",
    "validation_timing",
    "allocation_state",
    "materialization_state",
    "notes",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite_dir", type=Path)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} is not a JSON object")
    return value


def nested(mapping: dict[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def optional_nonnegative_int(value: Any) -> int | None:
    return (
        int(value)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0
        else None
    )


def required_nonnegative_int(value: Any, context: str) -> int:
    result = optional_nonnegative_int(value)
    if result is None:
        raise RuntimeError(f"{context} is missing nonnegative integer evidence")
    return result


def median_nonnegative_int(records: list[dict[str, Any]], key: str, context: str) -> int:
    values = [required_nonnegative_int(record.get(key), context) for record in records]
    if not values:
        raise RuntimeError(f"{context} has no run evidence")
    return int(statistics.median(values))


def normalized_workload(workload: str) -> str:
    if workload in {"throughput", "saturated"}:
        return "saturated_throughput"
    if workload in {"latency", "notification_latency"}:
        return "rate_limited_latency"
    raise RuntimeError(f"unsupported native workload: {workload}")


def wake_evidence(mapping: dict[str, Any]) -> tuple[bool, bool, bool]:
    wake = mapping.get("wake")
    if not isinstance(wake, dict):
        raise RuntimeError("benchmark result is missing wake evidence")
    values = tuple(wake.get(name) for name in ("event_wake", "busy_poll", "receive_polling"))
    if not all(isinstance(value, bool) for value in values):
        raise RuntimeError("benchmark result has invalid wake evidence")
    return values


def clock_evidence(mapping: dict[str, Any]) -> tuple[str, str, bool]:
    clock = mapping.get("clock")
    if not isinstance(clock, dict):
        raise RuntimeError("benchmark result is missing clock evidence")
    source = clock.get("source")
    scope = clock.get("scope")
    steady = clock.get("steady")
    if (
        not isinstance(source, str)
        or not source
        or not isinstance(scope, str)
        or not isinstance(steady, bool)
    ):
        raise RuntimeError("benchmark result has invalid clock evidence")
    return source, scope, steady


def google_bool_counter(mapping: dict[str, Any], name: str) -> bool:
    value = number(mapping.get(name))
    if value not in (0.0, 1.0):
        raise RuntimeError(f"Google Benchmark row has invalid {name} evidence")
    return value == 1.0


def runtime_support_threads(mapping: dict[str, Any]) -> str:
    value = mapping.get("runtime_support_threads")
    if not isinstance(value, str) or not value:
        raise RuntimeError("benchmark result is missing Runtime support-thread evidence")
    return value


def normalized_domain(domain: str) -> str:
    if domain.startswith("intraprocess"):
        return "intraprocess"
    if domain == "interprocess":
        return domain
    raise RuntimeError(f"unsupported benchmark domain: {domain}")


def normalized_payload(row: dict[str, Any]) -> str:
    payload = str(row["payload"])
    payload_bytes = int(row.get("application_payload_bytes", 0))
    benchmark = str(row["benchmark"])
    if payload == "pod" or "PubSubPOD" in benchmark:
        return "pod64"
    if payload == "inline":
        return f"fixed_bytes_{payload_bytes}"
    if payload == "buffer_handle":
        return f"raw_buffer_{payload_bytes}"
    if payload == "signal":
        return "trigger_token"
    return payload


def normalized_topology(row: dict[str, Any]) -> str:
    explicit = row.get("_topology")
    relay_count = int(row.get("relay_count", 0))
    if explicit == "chain" or relay_count > 0:
        return f"linear_chain_{relay_count + 2}_operators"
    return f"1to{int(row['fanout'])}"


def normalized_wake_mode(row: dict[str, Any]) -> str:
    explicit = row.get("_wake_mode")
    if isinstance(explicit, str) and explicit:
        return explicit
    if row["busy_poll"]:
        return "busy_poll"
    if row["event_wake"]:
        return "event_waitset"
    if row["receive_polling"]:
        return "receive_polling"
    return "none"


def complete_scenario_dimensions(row: dict[str, Any]) -> dict[str, Any]:
    row["scope"] = {
        "holomq": "holomq",
        "channel": "channel",
        "graph": "operator_graph",
    }[str(row["layer"])]
    row["scenario_domain"] = normalized_domain(str(row["domain"]))
    row["native_benchmark"] = str(row["benchmark"])
    row["workload_class"] = normalized_workload(str(row["workload"]))
    row["scenario_payload"] = normalized_payload(row)
    row["topology"] = normalized_topology(row)
    row["wake_mode"] = normalized_wake_mode(row)
    validate_workload_geometry(row)
    return row


def validate_workload_geometry(row: dict[str, Any]) -> None:
    counts = (
        "publication_attempted_count",
        "publication_accepted_count",
        "publication_backpressured_count",
        "publication_indeterminate_count",
        "publication_completed_count",
    )
    if any(optional_nonnegative_int(row.get(field)) is None for field in counts):
        raise RuntimeError("benchmark row is missing complete publication accounting")
    if row["publication_attempted_count"] != (
        row["publication_accepted_count"]
        + row["publication_backpressured_count"]
        + row["publication_indeterminate_count"]
    ):
        raise RuntimeError("benchmark row has inconsistent publication accounting")
    if row["publication_completed_count"] > row["publication_accepted_count"]:
        raise RuntimeError("completed publications exceed accepted publications")
    if int(row.get("external_buffer_bytes", 0)) > 0 and (
        optional_nonnegative_int(row.get("pool_slots")) is None
        or optional_nonnegative_int(row.get("release_capacity")) is None
        or optional_nonnegative_int(row.get("publication_backpressure_capacity")) is None
    ):
        raise RuntimeError("bulk benchmark row is missing pool or pressure bounds")
    if (
        row.get("layer") == "graph"
        and optional_nonnegative_int(row.get("notification_capacity")) is None
    ):
        raise RuntimeError("Runtime Graph row is missing notification capacity")


def summary_number(mapping: dict[str, Any], *keys: str) -> float | None:
    value = nested(mapping, *keys)
    if isinstance(value, dict):
        for statistic in ("median", "p50"):
            result = number(value.get(statistic))
            if result is not None:
                return result
        return None
    return number(value)


def time_to_ns(value: float, unit: str) -> float:
    scale = {"ns": 1.0, "us": 1_000.0, "ms": 1_000_000.0, "s": 1_000_000_000.0}
    if unit not in scale:
        raise RuntimeError(f"unsupported Google Benchmark time unit: {unit}")
    return value * scale[unit]


def google_benchmark_rows(path: Path) -> list[dict[str, Any]]:
    result = read_json(path)
    candidates = result.get("benchmarks")
    if not isinstance(candidates, list):
        raise RuntimeError(f"{path} is missing Google Benchmark rows")
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        aggregate_name = candidate.get("aggregate_name")
        if aggregate_name is not None:
            continue
        name = str(candidate.get("run_name") or candidate.get("name") or "")
        if not name:
            continue
        payload_match = re.search(r"<(\d+)>", name)
        payload_bytes = int(payload_match.group(1)) if payload_match else 64
        fanout = int(candidate.get("subscribers", 1))
        items_per_second = number(candidate.get("items_per_second"))
        is_latency = (
            "PollingRoundTrip" in name
            or "_Latency" in name
            or number(candidate.get("latency_workload")) == 1.0
        )
        event_wake = google_bool_counter(candidate, "event_wake")
        busy_poll = google_bool_counter(candidate, "busy_poll")
        receive_polling = google_bool_counter(candidate, "receive_polling")
        concurrent = number(candidate.get("concurrent")) == 1.0 or "Concurrent" in name
        row = {
            "layer": "holomq",
            "domain": "intraprocess",
            "benchmark": name.removesuffix("/real_time").removesuffix("/manual_time"),
            "workload": "latency" if is_latency else "throughput",
            "trigger_mechanism": "serialized_caller" if not concurrent else "publisher_thread",
            "payload": "inline",
            "storage": "inline",
            "fanout": fanout,
            "application_payload_bytes": payload_bytes,
            "inline_transport_bytes": payload_bytes,
            "external_buffer_bytes": 0,
            "payload_write_profile": "full_payload_per_sample",
            "bulk_work_profile": "not_applicable",
            "delivery_path": "holomq_inline",
            "execution_model": "concurrent" if concurrent else "serialized",
            "synchronization": "none",
            "event_wake": event_wake,
            "busy_poll": busy_poll,
            "receive_polling": receive_polling,
            "relay_count": 0,
            "queue_depth": None,
            "notification_capacity": None,
            "pool_slots": None,
            "release_capacity": None,
            "publication_backpressure_capacity": None,
            "max_fanout_release_streams": None,
            "process_count": 1,
            "runtime_support_threads": "none",
            "measured_samples": None,
            "warmup_samples": None,
            "setup_samples": 0,
            "publication_count_scope": "google_benchmark_iterations",
            "publication_attempted_count": optional_nonnegative_int(candidate.get("iterations")),
            "publication_accepted_count": optional_nonnegative_int(candidate.get("iterations")),
            "publication_backpressured_count": 0,
            "publication_indeterminate_count": 0,
            "publication_completed_count": optional_nonnegative_int(candidate.get("iterations")),
            "latency_anchor": "serialized_iteration",
            "publisher_pacing": "serialized",
            "effective_period_us": None,
            "clock_source": "google_benchmark",
            "clock_scope": "single_process",
            "clock_steady": None,
            "payload_validation": "all",
            "validation_timing": "inside_measurement",
            "allocation_state": "not_applicable",
            "materialization_state": "not_applicable",
            "logical_messages_per_second": (
                None if is_latency or items_per_second is None else items_per_second / fanout
            ),
            "aggregate_deliveries_per_second": (None if is_latency else items_per_second),
            "latency_p50_ns": None,
            "latency_p99_ns": None,
            "latency_p999_ns": None,
            "mean_iteration_ns": (
                time_to_ns(float(candidate["real_time"]), str(candidate["time_unit"]))
                if is_latency
                else None
            ),
            "timestamp_read_pair_p50_ns": None,
            "pool_acquisition_p50_ns": None,
            "import_materialization_p50_ns": None,
            "cuda_enqueue_p50_ns": None,
            "deferred_release_p50_ns": None,
            "final_validation_ns": None,
            "notes": (
                "Google Benchmark mean iteration time; per-message percentiles unavailable"
                if is_latency
                else ""
            ),
            "_primary_values": [
                (
                    time_to_ns(float(candidate["real_time"]), str(candidate["time_unit"]))
                    if is_latency
                    else (items_per_second / fanout if items_per_second is not None else 0.0)
                )
            ],
        }
        rows.append(row)
    return rows


def holomq_interprocess_row(path: Path) -> dict[str, Any]:
    result = read_json(path)
    metrics = result.get("metrics")
    if not isinstance(metrics, dict):
        raise RuntimeError(f"{path} is missing HoloMQ metrics")
    workload = str(result.get("workload"))
    event_wake, busy_poll, receive_polling = wake_evidence(result)
    clock_source, clock_scope, clock_steady = clock_evidence(result)
    runs = result.get("runs")
    primary_name = (
        "worst_subscriber_latency_p50_ns"
        if workload == "latency"
        else "completed_messages_per_second"
    )
    primary_values = (
        [
            float(run[primary_name])
            for run in runs
            if isinstance(run, dict) and isinstance(run.get(primary_name), (int, float))
        ]
        if isinstance(runs, list)
        else []
    )
    publication_records = (
        [
            run["publication_counts"]
            for run in runs
            if isinstance(run, dict) and isinstance(run.get("publication_counts"), dict)
        ]
        if isinstance(runs, list)
        else []
    )
    if not publication_records:
        raise RuntimeError(f"{path} is missing HoloMQ publication accounting")
    publication_attempted = median_nonnegative_int(
        publication_records, "attempted", "HoloMQ attempted publications"
    )
    publication_accepted = median_nonnegative_int(
        publication_records, "accepted", "HoloMQ accepted publications"
    )
    publication_backpressure = median_nonnegative_int(
        publication_records, "backpressured", "HoloMQ backpressured publications"
    )
    publication_indeterminate = median_nonnegative_int(
        publication_records, "indeterminate", "HoloMQ indeterminate publications"
    )
    publication_completed = median_nonnegative_int(
        publication_records, "completed", "HoloMQ completed publications"
    )
    pool_slots = optional_nonnegative_int(result.get("pool_slots"))
    return {
        "layer": "holomq",
        "domain": "interprocess",
        "benchmark": str(result.get("benchmark")),
        "workload": workload,
        "trigger_mechanism": "publisher_thread",
        "payload": str(result.get("payload")),
        "storage": str(result.get("storage")),
        "fanout": int(result.get("subscriber_count", 1)),
        "application_payload_bytes": int(result.get("application_payload_bytes", 0)),
        "inline_transport_bytes": int(result.get("inline_transport_bytes", 0)),
        "external_buffer_bytes": int(result.get("external_buffer_bytes", 0)),
        "payload_write_profile": str(result.get("payload_write_profile")),
        "bulk_work_profile": str(result.get("bulk_work_profile")),
        "delivery_path": (
            "holomq_buffer_handle" if result.get("payload") == "buffer_handle" else "holomq_inline"
        ),
        "execution_model": "concurrent",
        "synchronization": ("cuda_ready_event" if result.get("device_sync") == "ready" else "none"),
        "event_wake": event_wake,
        "busy_poll": busy_poll,
        "receive_polling": receive_polling,
        "relay_count": 0,
        "queue_depth": optional_nonnegative_int(result.get("queue_depth")),
        "notification_capacity": None,
        "pool_slots": pool_slots,
        "release_capacity": pool_slots,
        "publication_backpressure_capacity": optional_nonnegative_int(result.get("queue_depth")),
        "max_fanout_release_streams": (
            int(result.get("subscriber_count", 1))
            if result.get("payload") == "buffer_handle"
            else None
        ),
        "process_count": int(result.get("subscriber_count", 1)) + 1,
        "runtime_support_threads": runtime_support_threads(result),
        "measured_samples": int(result.get("samples", 0)),
        "warmup_samples": int(result.get("warmup", 0)),
        "setup_samples": 0,
        "publication_count_scope": "measured_plus_warmup",
        "publication_attempted_count": publication_attempted,
        "publication_accepted_count": publication_accepted,
        "publication_backpressured_count": publication_backpressure,
        "publication_indeterminate_count": publication_indeterminate,
        "publication_completed_count": publication_completed,
        "latency_anchor": "publication_to_receive_completion",
        "publisher_pacing": (
            str(result.get("publisher_pacing")) if workload == "latency" else "unpaced"
        ),
        "effective_period_us": (
            int(result.get("publisher_period_us", 0)) if workload == "latency" else None
        ),
        "clock_source": clock_source,
        "clock_scope": clock_scope,
        "clock_steady": clock_steady,
        "payload_validation": str(result.get("payload_validation", "all")),
        "validation_timing": str(result.get("validation_timing", "inside_measurement")),
        "allocation_state": (
            "preallocated_pool" if result.get("payload") == "buffer_handle" else "not_applicable"
        ),
        "materialization_state": (
            "cold_first_then_warmed"
            if result.get("payload") == "buffer_handle"
            else "not_applicable"
        ),
        "logical_messages_per_second": (
            summary_number(metrics, "completed_messages_per_second")
            if workload == "throughput"
            else None
        ),
        "aggregate_deliveries_per_second": (
            summary_number(metrics, "completed_deliveries_per_second")
            if workload == "throughput"
            else None
        ),
        "latency_p50_ns": (
            summary_number(metrics, "worst_subscriber_latency_p50_ns")
            if workload == "latency"
            else None
        ),
        "latency_p99_ns": (
            summary_number(metrics, "worst_subscriber_latency_p99_ns")
            if workload == "latency"
            else None
        ),
        "latency_p999_ns": (
            summary_number(metrics, "worst_subscriber_latency_p999_ns")
            if workload == "latency"
            else None
        ),
        "mean_iteration_ns": None,
        "timestamp_read_pair_p50_ns": None,
        "pool_acquisition_p50_ns": (
            summary_number(metrics, "buffer_acquire_p50_ns")
            if result.get("payload") == "buffer_handle"
            else None
        ),
        "import_materialization_p50_ns": (
            summary_number(metrics, "buffer_import_p50_ns")
            if result.get("payload") == "buffer_handle"
            else None
        ),
        "cuda_enqueue_p50_ns": None,
        "deferred_release_p50_ns": None,
        "final_validation_ns": (
            summary_number(metrics, "final_validation_ns")
            if result.get("payload") == "buffer_handle"
            else None
        ),
        "notes": "low_latency_ceiling" if busy_poll else "",
        "_primary_values": primary_values,
    }


def channel_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result_path in sorted(path.glob("*.json")):
        result = read_json(result_path)
        publisher = result.get("publisher")
        subscribers = result.get("subscribers")
        fanout_summary = result.get("fanout")
        if (
            not isinstance(publisher, dict)
            or not isinstance(subscribers, list)
            or not isinstance(fanout_summary, dict)
        ):
            continue
        subscriber_records = [item for item in subscribers if isinstance(item, dict)]
        workload = str(publisher.get("workload"))
        event_wake, busy_poll, receive_polling = wake_evidence(publisher)
        clock_source, clock_scope, clock_steady = clock_evidence(publisher)
        payload = str(publisher.get("payload"))
        native_domain = str(result.get("domain"))
        native_transport = str(publisher.get("transport"))
        delivery_path = {
            "inline": "holomq_inline",
            "sidecar": "holomq_sidecar",
            "buffer_handle": "holomq_buffer_handle",
        }.get(native_transport)
        if delivery_path is None:
            raise RuntimeError(
                f"{result_path} has an unsupported Channel transport: {native_transport}"
            )
        storage = (
            str(publisher.get("tensor_storage", "unknown"))
            if payload in {"tensor", "image"}
            else "inline"
        )
        latency_p50 = [number(item.get("end_to_end_p50_ns")) for item in subscriber_records]
        latency_p99 = [number(item.get("end_to_end_p99_ns")) for item in subscriber_records]
        measured_samples = required_nonnegative_int(
            publisher.get("samples"), "Channel publisher samples"
        )
        warmup_samples = required_nonnegative_int(
            publisher.get("warmup"), "Channel publisher warmup"
        )
        total_samples = measured_samples + warmup_samples
        publication_backpressure = required_nonnegative_int(
            publisher.get("failed_emits"), "Channel publisher backpressure"
        )
        pool_slots = optional_nonnegative_int(publisher.get("pool_slots"))
        release_capacity = optional_nonnegative_int(publisher.get("release_capacity"))
        rows.append(
            {
                "layer": "channel",
                "domain": native_domain,
                "benchmark": str(result.get("benchmark") or result_path.stem),
                "workload": workload,
                "trigger_mechanism": "publisher_thread",
                "payload": payload,
                "storage": storage,
                "fanout": len(subscriber_records),
                "application_payload_bytes": int(publisher.get("application_payload_bytes", 0)),
                "inline_transport_bytes": int(publisher.get("inline_transport_bytes", 0)),
                "external_buffer_bytes": int(publisher.get("external_buffer_bytes", 0)),
                "payload_write_profile": str(publisher.get("payload_write_profile")),
                "bulk_work_profile": str(publisher.get("bulk_work_profile")),
                "delivery_path": delivery_path,
                "execution_model": (
                    "serialized" if native_domain == "intraprocess_serial" else "concurrent"
                ),
                "synchronization": (
                    "cuda_stream_ordered" if publisher.get("async_cuda_work") is True else "none"
                ),
                "event_wake": event_wake,
                "busy_poll": busy_poll,
                "receive_polling": receive_polling,
                "relay_count": 0,
                "queue_depth": optional_nonnegative_int(publisher.get("subscriber_queue_depth")),
                "notification_capacity": None,
                "pool_slots": pool_slots,
                "release_capacity": release_capacity,
                "publication_backpressure_capacity": optional_nonnegative_int(
                    publisher.get("subscriber_queue_depth")
                ),
                "max_fanout_release_streams": optional_nonnegative_int(
                    publisher.get("max_fanout_release_streams")
                ),
                "process_count": (
                    len(subscriber_records) + 1 if native_domain == "interprocess" else 1
                ),
                "runtime_support_threads": "none",
                "measured_samples": measured_samples,
                "warmup_samples": warmup_samples,
                "setup_samples": 0,
                "publication_count_scope": "measured_plus_warmup",
                "publication_attempted_count": total_samples + publication_backpressure,
                "publication_accepted_count": total_samples,
                "publication_backpressured_count": publication_backpressure,
                "publication_indeterminate_count": 0,
                "publication_completed_count": total_samples,
                "latency_anchor": "publication_to_receive_completion",
                "publisher_pacing": (
                    str(publisher.get("publisher_pacing")) if workload == "latency" else "unpaced"
                ),
                "effective_period_us": (
                    int(publisher.get("publisher_period_us", 0)) if workload == "latency" else None
                ),
                "clock_source": clock_source,
                "clock_scope": clock_scope,
                "clock_steady": clock_steady,
                "payload_validation": str(publisher.get("payload_validation", "unknown")),
                "validation_timing": str(publisher.get("validation_timing", "unknown")),
                "allocation_state": (
                    "preallocated_pool" if payload in {"tensor", "image"} else "not_applicable"
                ),
                "materialization_state": (
                    "warmed_before_measurement"
                    if payload in {"tensor", "image"} and int(publisher.get("warmup", 0)) > 0
                    else "not_applicable"
                ),
                "logical_messages_per_second": (
                    number(fanout_summary.get("completed_logical_messages_per_second"))
                    if workload == "throughput"
                    else None
                ),
                "aggregate_deliveries_per_second": (
                    number(fanout_summary.get("completed_deliveries_per_second"))
                    if workload == "throughput"
                    else None
                ),
                "latency_p50_ns": (
                    max(value for value in latency_p50 if value is not None)
                    if workload == "latency" and any(value is not None for value in latency_p50)
                    else None
                ),
                "latency_p99_ns": (
                    max(value for value in latency_p99 if value is not None)
                    if workload == "latency" and any(value is not None for value in latency_p99)
                    else None
                ),
                "latency_p999_ns": (
                    max(
                        value
                        for value in (
                            number(item.get("end_to_end_p999_ns")) for item in subscriber_records
                        )
                        if value is not None
                    )
                    if workload == "latency" and int(publisher.get("samples", 0)) >= 10_000
                    else None
                ),
                "mean_iteration_ns": None,
                "timestamp_read_pair_p50_ns": None,
                "pool_acquisition_p50_ns": None,
                "import_materialization_p50_ns": (
                    max(
                        value
                        for value in (
                            number(item.get("channel_receive_p50_ns"))
                            for item in subscriber_records
                        )
                        if value is not None
                    )
                    if payload in {"tensor", "image"}
                    and any(
                        number(item.get("channel_receive_p50_ns")) is not None
                        for item in subscriber_records
                    )
                    else None
                ),
                "cuda_enqueue_p50_ns": (
                    max(
                        value
                        for value in (
                            number(publisher.get("cuda_work_enqueue_p50_ns")),
                            *(
                                number(item.get("cuda_work_enqueue_p50_ns"))
                                for item in subscriber_records
                            ),
                        )
                        if value is not None
                    )
                    if publisher.get("async_cuda_work") is True
                    else None
                ),
                "deferred_release_p50_ns": None,
                "final_validation_ns": (
                    max(
                        value
                        for value in (
                            number(item.get("final_validation_ns")) for item in subscriber_records
                        )
                        if value is not None
                    )
                    if payload in {"tensor", "image"}
                    and any(
                        number(item.get("final_validation_ns")) is not None
                        for item in subscriber_records
                    )
                    else None
                ),
                "notes": "",
                "_native_artifact": result_path,
                "_primary_values": [
                    (
                        max(value for value in latency_p50 if value is not None)
                        if workload == "latency"
                        else number(fanout_summary.get("completed_logical_messages_per_second"))
                    )
                ],
            }
        )
    return rows


def graph_intraprocess_row(path: Path, case_id: str) -> dict[str, Any]:
    result = read_json(path)
    metrics = result.get("metrics")
    geometry = result.get("geometry")
    if not isinstance(metrics, dict) or not isinstance(geometry, dict):
        raise RuntimeError(f"{path} is missing Runtime Graph metrics")
    workload = str(result.get("workload"))
    event_wake, busy_poll, receive_polling = wake_evidence(result)
    clock_source, clock_scope, clock_steady = clock_evidence(result)
    scenario = str(result.get("scenario"))
    payload = {
        "pod": "pod",
        "sample_forward": "pod",
        "signal": "signal",
        "schema": "schema",
        "tensor": "tensor",
        "image": "image",
        "multi_tensor": "multi_tensor",
    }.get(scenario)
    if payload is None:
        payload = next(
            (
                candidate
                for candidate in ("multi_tensor", "tensor", "image", "schema", "signal", "pod")
                if candidate in case_id
            ),
            "unknown",
        )
    tensor = result.get("tensor")
    storage = (
        str(tensor.get("storage"))
        if payload in {"tensor", "image", "multi_tensor"} and isinstance(tensor, dict)
        else "inline"
    )
    if storage == "None":
        storage = "inline"
    if payload in {"tensor", "image", "multi_tensor"} and storage == "inline":
        storage = next(
            (
                candidate
                for candidate in ("pageable_host", "pinned_host", "device")
                if f"_{candidate}_" in case_id
            ),
            "pageable_host" if payload in {"image", "multi_tensor"} else "unknown",
        )
    runs = result.get("runs")
    primary_name = (
        "publication_to_receive_p50_ns"
        if workload != "saturated" and payload != "unknown"
        else ("latency_p50_ns" if workload != "saturated" else "logical_publications_per_second")
    )
    primary_values = (
        [
            float(run[primary_name])
            for run in runs
            if isinstance(run, dict) and isinstance(run.get(primary_name), (int, float))
        ]
        if isinstance(runs, list)
        else []
    )
    validation = result.get("validation")
    validation_mode = str(validation.get("mode")) if isinstance(validation, dict) else "unknown"
    placement = result.get("placement")
    if not isinstance(placement, dict):
        raise RuntimeError(f"{path} is missing Runtime Graph placement evidence")
    lane_mode = placement.get("requested_operator_lanes")
    if not isinstance(lane_mode, str):
        raise RuntimeError(f"{path} is missing valid requested_operator_lanes placement evidence")
    execution_model = {
        "shared": "shared_lane",
        "separate": "separate_lane",
        "per_sink": "per_subscriber_lane",
    }.get(lane_mode)
    if execution_model is None:
        raise RuntimeError(f"{path} has an unsupported Operator lane mode: {lane_mode}")
    native_delivery = str(result.get("process_local_inline_delivery"))
    if native_delivery not in {"direct_retained", "holomq", "not_applicable"}:
        raise RuntimeError(
            f"{path} has an unsupported process-local delivery mode: {native_delivery}"
        )
    delivery_path = native_delivery
    if native_delivery == "holomq":
        delivery_path = (
            "holomq_sidecar" if payload in {"tensor", "image", "multi_tensor"} else "holomq_inline"
        )
    run_records = [run for run in runs if isinstance(run, dict)] if isinstance(runs, list) else []
    count_records = [run["counts"] for run in run_records if isinstance(run.get("counts"), dict)]
    publication_accepted = median_nonnegative_int(
        count_records, "publications", "Runtime Graph accepted publications"
    )
    publication_backpressured = median_nonnegative_int(
        count_records, "backpressured_emits", "Runtime Graph backpressured publications"
    )
    return {
        "layer": "graph",
        "domain": "intraprocess",
        "benchmark": str(result.get("benchmark")),
        "workload": "throughput" if workload == "saturated" else "latency",
        "trigger_mechanism": (
            "runtime_on_clock"
            if result.get("source_trigger") == "clock"
            else "runtime_notification_source"
        ),
        "payload": payload,
        "storage": storage,
        "fanout": int(geometry.get("fanout", 1)),
        "application_payload_bytes": int(geometry.get("application_payload_bytes", 0)),
        "inline_transport_bytes": optional_nonnegative_int(geometry.get("inline_transport_bytes")),
        "external_buffer_bytes": int(geometry.get("external_buffer_bytes", 0)),
        "payload_write_profile": str(geometry.get("payload_write_profile")),
        "bulk_work_profile": str(geometry.get("bulk_work_profile")),
        "delivery_path": delivery_path,
        "execution_model": execution_model,
        "synchronization": (
            "cuda_stream_ordered" if payload == "tensor" and storage == "device" else "none"
        ),
        "event_wake": event_wake,
        "busy_poll": busy_poll,
        "receive_polling": receive_polling,
        "_wake_mode": str(placement.get("wake_mode", "sleeping_waitset")),
        "relay_count": int(geometry.get("relay_count", 0)),
        "queue_depth": optional_nonnegative_int(geometry.get("queue_depth")),
        "notification_capacity": optional_nonnegative_int(geometry.get("notification_capacity")),
        "pool_slots": optional_nonnegative_int(geometry.get("pool_slots")),
        "release_capacity": optional_nonnegative_int(geometry.get("release_capacity")),
        "publication_backpressure_capacity": optional_nonnegative_int(
            geometry.get("publication_backpressure_capacity")
        ),
        "max_fanout_release_streams": optional_nonnegative_int(
            geometry.get("max_fanout_release_streams")
        ),
        "process_count": 1,
        "runtime_support_threads": runtime_support_threads(result),
        "measured_samples": int(geometry.get("samples", 0)),
        "warmup_samples": int(geometry.get("warmup", 0)),
        "setup_samples": int(geometry.get("allocation_probe_samples", 0)),
        "publication_count_scope": "measured_plus_warmup_plus_setup",
        "publication_attempted_count": publication_accepted + publication_backpressured,
        "publication_accepted_count": publication_accepted,
        "publication_backpressured_count": publication_backpressured,
        "publication_indeterminate_count": 0,
        "publication_completed_count": publication_accepted,
        "latency_anchor": "publication_to_receive_completion",
        "publisher_pacing": (
            (
                "runtime_on_clock"
                if result.get("source_trigger") == "clock"
                else "notification_driver_sleep"
            )
            if workload != "saturated"
            else "unpaced"
        ),
        "effective_period_us": (
            int(geometry.get("publisher_period_us", 0)) if workload != "saturated" else None
        ),
        "clock_source": clock_source,
        "clock_scope": clock_scope,
        "clock_steady": clock_steady,
        "payload_validation": validation_mode,
        "validation_timing": (
            str(validation.get("timing")) if isinstance(validation, dict) else "unknown"
        ),
        "allocation_state": (
            "preallocated_pool"
            if payload in {"tensor", "image", "multi_tensor"}
            else "not_applicable"
        ),
        "materialization_state": (
            "warmed_before_measurement"
            if payload in {"tensor", "image", "multi_tensor"} and int(geometry.get("warmup", 0)) > 0
            else "not_applicable"
        ),
        "logical_messages_per_second": (
            summary_number(metrics, "logical_publications_per_second_across_runs")
            if workload == "saturated"
            else None
        ),
        "aggregate_deliveries_per_second": (
            summary_number(metrics, "aggregate_deliveries_per_second_across_runs")
            if workload == "saturated"
            else None
        ),
        "latency_p50_ns": (
            summary_number(metrics, "publication_to_receive_p50_ns_across_runs")
            if workload != "saturated"
            else None
        ),
        "latency_p99_ns": (
            summary_number(metrics, "publication_to_receive_p99_ns_across_runs")
            if workload != "saturated"
            else None
        ),
        "latency_p999_ns": (
            summary_number(metrics, "publication_to_receive_p999_ns_across_runs")
            if workload != "saturated"
            else None
        ),
        "mean_iteration_ns": None,
        "timestamp_read_pair_p50_ns": summary_number(
            result, "timestamp_instrumentation", "pair_overhead_p50_ns_across_runs"
        ),
        "pool_acquisition_p50_ns": None,
        "import_materialization_p50_ns": None,
        "cuda_enqueue_p50_ns": None,
        "deferred_release_p50_ns": None,
        "final_validation_ns": summary_number(metrics, "final_validation_ns_across_runs"),
        "notes": "low_latency_ceiling" if busy_poll else "",
        "_primary_values": primary_values,
    }


def graph_interprocess_row(path: Path) -> dict[str, Any]:
    result = read_json(path)
    workload = str(result.get("workload"))
    latency = result.get("slowest_sink_end_to_end_p50_ns")
    event_wake, busy_poll, receive_polling = wake_evidence(result)
    clock_source, clock_scope, clock_steady = clock_evidence(result)
    producer_runs = result.get("producer")
    consumer_runs = result.get("consumers")
    if workload == "saturated":
        primary_values = (
            [
                float(producer["timing"]["messages_per_second"])
                for producer in producer_runs
                if isinstance(producer, dict)
                and isinstance(nested(producer, "timing", "messages_per_second"), (int, float))
            ]
            if isinstance(producer_runs, list)
            else []
        )
    else:
        primary_values = (
            [
                max(
                    float(consumer["timing"]["end_to_end_ns"]["p50"])
                    for consumer in consumers
                    if isinstance(consumer, dict)
                )
                for consumers in consumer_runs
                if isinstance(consumers, list) and consumers
            ]
            if isinstance(consumer_runs, list)
            else []
        )
    measured_samples = 0
    warmup_samples = 0
    geometry: dict[str, Any] = {}
    if isinstance(producer_runs, list) and producer_runs and isinstance(producer_runs[0], dict):
        emitted_geometry = producer_runs[0].get("geometry")
        if isinstance(emitted_geometry, dict):
            geometry = emitted_geometry
            measured_samples = int(geometry.get("samples", 0))
            warmup_samples = int(geometry.get("warmup", 0))
    producer_records = (
        [producer for producer in producer_runs if isinstance(producer, dict)]
        if isinstance(producer_runs, list)
        else []
    )
    count_records = [
        producer["counts"]
        for producer in producer_records
        if isinstance(producer.get("counts"), dict)
    ]
    publication_accepted = median_nonnegative_int(
        count_records, "publications", "distributed Graph accepted publications"
    )
    publication_backpressured = median_nonnegative_int(
        count_records, "backpressured_emits", "distributed Graph backpressured publications"
    )
    payload = str(result.get("payload", "pod"))
    tensor_backed = payload in {
        "tensor",
        "image",
        "multi_tensor_loan",
        "multi_tensor_staging",
    }
    storage = str(result.get("tensor_storage")) if tensor_backed else "inline"
    return {
        "layer": "graph",
        "domain": "interprocess",
        "benchmark": str(result.get("benchmark")),
        "workload": "throughput" if workload == "saturated" else "latency",
        "trigger_mechanism": "runtime_notification_source",
        "payload": payload,
        "storage": storage,
        "fanout": int(result.get("fanout", 1)),
        "application_payload_bytes": int(geometry.get("application_payload_bytes", 0)),
        "inline_transport_bytes": optional_nonnegative_int(geometry.get("inline_transport_bytes")),
        "external_buffer_bytes": int(geometry.get("external_buffer_bytes", 0)),
        "payload_write_profile": str(geometry.get("payload_write_profile")),
        "bulk_work_profile": str(geometry.get("bulk_work_profile")),
        "delivery_path": "holomq_buffer_handle" if tensor_backed else "holomq_inline",
        "execution_model": "separate_lane",
        "synchronization": (
            "cuda_stream_ordered" if tensor_backed and storage == "device" else "none"
        ),
        "event_wake": event_wake,
        "busy_poll": busy_poll,
        "receive_polling": receive_polling,
        "_wake_mode": "busy_poll" if busy_poll else "sleeping_waitset",
        "_topology": str(result.get("topology")),
        "relay_count": 1 if result.get("topology") == "chain" else 0,
        "queue_depth": optional_nonnegative_int(geometry.get("queue_depth")),
        "notification_capacity": optional_nonnegative_int(geometry.get("notification_capacity")),
        "pool_slots": optional_nonnegative_int(geometry.get("pool_slots")),
        "release_capacity": optional_nonnegative_int(geometry.get("release_capacity")),
        "publication_backpressure_capacity": optional_nonnegative_int(
            geometry.get("publication_backpressure_capacity")
        ),
        "max_fanout_release_streams": None,
        "process_count": (
            3 if result.get("topology") == "chain" else int(result.get("fanout", 1)) + 1
        ),
        "runtime_support_threads": runtime_support_threads(result),
        "measured_samples": measured_samples,
        "warmup_samples": warmup_samples,
        "setup_samples": 0,
        "publication_count_scope": "measured_plus_warmup",
        "publication_attempted_count": publication_accepted + publication_backpressured,
        "publication_accepted_count": publication_accepted,
        "publication_backpressured_count": publication_backpressured,
        "publication_indeterminate_count": 0,
        "publication_completed_count": publication_accepted,
        "latency_anchor": "publication_to_receive_completion",
        "publisher_pacing": ("notification_driver_sleep" if workload != "saturated" else "unpaced"),
        "effective_period_us": (
            int(geometry.get("publisher_period_us", 0)) if workload != "saturated" else None
        ),
        "clock_source": clock_source,
        "clock_scope": clock_scope,
        "clock_steady": clock_steady,
        "payload_validation": str(result.get("payload_validation", "all")),
        "validation_timing": str(result.get("validation_timing", "inside_measurement")),
        "allocation_state": "preallocated_pool" if tensor_backed else "not_applicable",
        "materialization_state": (
            "warmed_before_measurement"
            if tensor_backed and warmup_samples > 0
            else "not_applicable"
        ),
        "logical_messages_per_second": (
            summary_number(result, "logical_publications_per_second")
            if workload == "saturated"
            else None
        ),
        "aggregate_deliveries_per_second": (
            summary_number(result, "aggregate_deliveries_per_second")
            if workload == "saturated"
            else None
        ),
        "latency_p50_ns": (
            summary_number(latency)
            if workload != "saturated" and isinstance(latency, dict)
            else None
        ),
        "latency_p99_ns": (
            summary_number(result, "slowest_sink_end_to_end_p99_ns")
            if workload != "saturated"
            else None
        ),
        "latency_p999_ns": (
            summary_number(result, "slowest_sink_end_to_end_p999_ns")
            if workload != "saturated"
            else None
        ),
        "mean_iteration_ns": None,
        "timestamp_read_pair_p50_ns": summary_number(
            result,
            "timestamp_instrumentation",
            "pair_overhead_p50_ns_across_processes_and_runs",
        ),
        "pool_acquisition_p50_ns": None,
        "import_materialization_p50_ns": None,
        "cuda_enqueue_p50_ns": None,
        "deferred_release_p50_ns": None,
        "final_validation_ns": None,
        "notes": "low_latency_ceiling" if busy_poll else "",
        "_primary_values": primary_values,
    }


def apply_latency_contract(
    row: dict[str, Any], case: dict[str, Any], comparison_period_us: int
) -> dict[str, Any]:
    row["comparison_period_us"] = None
    row["pacing_period_policy"] = "not_applicable"
    row["pacing_period_reason"] = ""
    if row.get("workload") != "latency":
        return row
    if row.get("publisher_pacing") == "serialized":
        row["pacing_period_policy"] = "native_serialized"
        row["pacing_period_reason"] = "serialized_round_trip_has_no_external_pacer"
        return row

    effective_period_us = row.get("effective_period_us")
    if not isinstance(effective_period_us, int) or effective_period_us <= 0:
        raise RuntimeError("paced latency row is missing an effective period")
    row["comparison_period_us"] = comparison_period_us
    if effective_period_us == comparison_period_us:
        row["pacing_period_policy"] = "common_requested_period"
        return row

    row["pacing_period_policy"] = "minimum_period_override"
    case_id = str(case.get("case_id", ""))
    if case_id == "graph_interprocess_1to2_latency":
        row["pacing_period_reason"] = "distributed_graph_1to2_flow_control_minimum"
    elif (
        row.get("layer") == "holomq"
        and row.get("payload") == "buffer_handle"
        and row.get("storage") == "device"
    ):
        row["pacing_period_reason"] = "device_ready_release_headroom"
    else:
        row["pacing_period_reason"] = "native_row_minimum"
    return row


def rows_for_case(
    case: dict[str, Any], suite_dir: Path, comparison_period_us: int
) -> list[dict[str, Any]]:
    path = suite_dir / str(case["output"])
    kind = case.get("result_kind")
    if kind == "google_benchmark":
        rows = google_benchmark_rows(path)
    elif kind == "holomq_interprocess":
        rows = [holomq_interprocess_row(path)]
    elif kind == "channel_matrix":
        rows = channel_rows(path)
    elif kind == "graph_intraprocess":
        rows = [graph_intraprocess_row(path, str(case.get("case_id", "")))]
    elif kind == "graph_interprocess":
        rows = [graph_interprocess_row(path)]
    else:
        raise RuntimeError(f"unknown benchmark result kind: {kind}")
    completed: list[dict[str, Any]] = []
    for row in rows:
        row = complete_scenario_dimensions(apply_latency_contract(row, case, comparison_period_us))
        artifact = Path(str(row.get("_native_artifact", path)))
        with suppress(ValueError):
            artifact = artifact.relative_to(suite_dir)
        row["_native_run"] = {
            "manifest_case_id": str(case.get("case_id", "")),
            "artifact": str(artifact),
            "command": case.get("command"),
        }
        completed.append(row)
    return completed


def aggregate_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(field) for field in IDENTITY_FIELDS)].append(row)
    aggregated: list[dict[str, Any]] = []
    for identity, values in grouped.items():
        row = dict(zip(IDENTITY_FIELDS, identity, strict=False))
        row["repetitions"] = sum(max(1, len(value.get("_primary_values", []))) for value in values)
        for metric in METRICS:
            numeric = [number(value.get(metric)) for value in values]
            row[metric] = (
                statistics.median(value for value in numeric if value is not None)
                if any(value is not None for value in numeric)
                else None
            )
        for field in COUNT_FIELDS:
            numeric = [optional_nonnegative_int(value.get(field)) for value in values]
            if any(value is None for value in numeric):
                raise RuntimeError(f"aggregated row is missing {field}")
            row[field] = int(statistics.median(value for value in numeric if value is not None))
        primary_metric = (
            "latency_p50_ns"
            if row["workload"] == "latency" and row["latency_p50_ns"] is not None
            else (
                "mean_iteration_ns"
                if row["workload"] == "latency"
                else "logical_messages_per_second"
            )
        )
        primary_values = [
            float(item)
            for value in values
            for item in value.get("_primary_values", [])
            if isinstance(item, (int, float))
        ]
        if not primary_values:
            fallback = number(row.get(primary_metric))
            primary_values = [] if fallback is None else [fallback]
        mean = statistics.fmean(primary_values) if primary_values else 0.0
        row["primary_metric"] = primary_metric
        row["primary_metric_median"] = statistics.median(primary_values) if primary_values else None
        row["primary_metric_min"] = min(primary_values) if primary_values else None
        row["primary_metric_max"] = max(primary_values) if primary_values else None
        row["primary_metric_coefficient_of_variation"] = (
            statistics.stdev(primary_values) / mean
            if len(primary_values) > 1 and mean != 0.0
            else None
        )
        native_runs = {
            json.dumps(value["_native_run"], separators=(",", ":"), sort_keys=True)
            for value in values
            if isinstance(value.get("_native_run"), dict)
        }
        row["native_runs"] = [json.loads(value) for value in sorted(native_runs)]
        row["scenario_identity"] = make_scenario_identity(row)
        row["scenario_key"] = scenario_identity_key(row["scenario_identity"])
        aggregated.append(row)
    return sorted(
        aggregated,
        key=lambda row: (
            str(row["layer"]),
            str(row["domain"]),
            str(row["workload"]),
            str(row["benchmark"]),
        ),
    )


def collect_rows(suite_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = read_json(suite_dir / "suite_manifest.json")
    if manifest.get("scenario_identity_contract") != scenario_identity_contract():
        raise RuntimeError("suite manifest has an unsupported scenario identity contract")
    cases = manifest.get("cases")
    if not isinstance(cases, list):
        raise RuntimeError("suite manifest is missing cases")
    comparison_period_us = manifest.get("latency_period_us")
    if not isinstance(comparison_period_us, int) or comparison_period_us <= 0:
        raise RuntimeError("suite manifest is missing its comparison latency period")
    rows: list[dict[str, Any]] = []
    for case in cases:
        if not isinstance(case, dict) or case.get("status") != "passed":
            continue
        rows.extend(rows_for_case(case, suite_dir, comparison_period_us))
    return manifest, aggregate_rows(rows)


def display_number(value: Any, *, rate: bool = False) -> str:
    numeric = number(value)
    if numeric is None:
        return ""
    return f"{numeric:,.0f}" if rate or numeric >= 100 else f"{numeric:,.2f}"


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    headings = (
        "Layer",
        "Scope",
        "Domain",
        "Topology",
        "Benchmark",
        "Workload",
        "Payload",
        "Scenario payload",
        "Storage",
        "Delivery",
        "Execution",
        "Synchronization",
        "Wake mode",
        "Fanout",
        "Application bytes",
        "Inline bytes",
        "External bytes",
        "Write profile",
        "Bulk work",
        "Event wake",
        "Busy poll",
        "Receive polling",
        "Runtime support threads",
        "Samples/run",
        "Warmup",
        "Latency anchor",
        "Publisher pacing",
        "Comparison period (us)",
        "Effective period (us)",
        "Period policy",
        "Period reason",
        "Clock source",
        "Clock scope",
        "Steady clock",
        "Payload validation",
        "Validation timing",
        "Allocation",
        "Materialization",
        "Messages/s",
        "Deliveries/s",
        "p50 (ns)",
        "p99 (ns)",
        "Mean iteration (ns)",
        "Clock-read pair p50 (ns)",
        "Pool acquire p50 (ns)",
        "Import/materialize p50 (ns)",
        "CUDA enqueue p50 (ns)",
        "Deferred release p50 (ns)",
        "Final validation (ns)",
        "Repetitions",
        "Primary metric",
        "Primary min",
        "Primary median",
        "Primary max",
        "Primary CV",
        "Notes",
        "Manifest cases",
    )
    lines = [
        "# Transport Benchmark Summary",
        "",
        "| " + " | ".join(headings) + " |",
        "|" + "|".join("---" for _ in headings) + "|",
    ]
    for row in rows:
        values = (
            row["layer"],
            row["scope"],
            row["domain"],
            row["topology"],
            row["benchmark"],
            row["workload"],
            row["payload"],
            row["scenario_payload"],
            row["storage"],
            row["delivery_path"],
            row["execution_model"],
            row["synchronization"],
            row["wake_mode"],
            row["fanout"],
            display_number(row["application_payload_bytes"], rate=True),
            display_number(row["inline_transport_bytes"], rate=True),
            display_number(row["external_buffer_bytes"], rate=True),
            row["payload_write_profile"],
            row["bulk_work_profile"],
            row["event_wake"],
            row["busy_poll"],
            row["receive_polling"],
            row["runtime_support_threads"],
            row["measured_samples"],
            row["warmup_samples"],
            row["latency_anchor"],
            row["publisher_pacing"],
            row["comparison_period_us"],
            row["effective_period_us"],
            row["pacing_period_policy"],
            row["pacing_period_reason"],
            row["clock_source"],
            row["clock_scope"],
            row["clock_steady"],
            row["payload_validation"],
            row["validation_timing"],
            row["allocation_state"],
            row["materialization_state"],
            display_number(row["logical_messages_per_second"], rate=True),
            display_number(row["aggregate_deliveries_per_second"], rate=True),
            display_number(row["latency_p50_ns"], rate=True),
            display_number(row["latency_p99_ns"], rate=True),
            display_number(row["mean_iteration_ns"], rate=True),
            display_number(row["timestamp_read_pair_p50_ns"], rate=True),
            display_number(row["pool_acquisition_p50_ns"], rate=True),
            display_number(row["import_materialization_p50_ns"], rate=True),
            display_number(row["cuda_enqueue_p50_ns"], rate=True),
            display_number(row["deferred_release_p50_ns"], rate=True),
            display_number(row["final_validation_ns"], rate=True),
            row["repetitions"],
            row["primary_metric"],
            display_number(row["primary_metric_min"], rate=True),
            display_number(row["primary_metric_median"], rate=True),
            display_number(row["primary_metric_max"], rate=True),
            display_number(row["primary_metric_coefficient_of_variation"]),
            row["notes"],
            "<br>".join(str(native_run["manifest_case_id"]) for native_run in row["native_runs"]),
        )
        lines.append("| " + " | ".join(str(value) for value in values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        *IDENTITY_FIELDS,
        *COUNT_FIELDS,
        "repetitions",
        *METRICS,
        "primary_metric",
        "primary_metric_min",
        "primary_metric_median",
        "primary_metric_max",
        "primary_metric_coefficient_of_variation",
        "scenario_key",
        "scenario_identity",
        "native_runs",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            serialized = dict(row)
            serialized["scenario_identity"] = json.dumps(
                row["scenario_identity"], separators=(",", ":"), sort_keys=True
            )
            serialized["native_runs"] = json.dumps(
                row["native_runs"], separators=(",", ":"), sort_keys=True
            )
            writer.writerow(serialized)


def main() -> int:
    args = parse_args()
    suite_dir = args.suite_dir.resolve()
    manifest, rows = collect_rows(suite_dir)
    summary = {
        "schema": SUMMARY_SCHEMA,
        "suite_schema": manifest.get("schema"),
        "scenario_identity_contract": scenario_identity_contract(),
        "workload_geometry_policy": manifest.get("workload_geometry_policy"),
        "preset": manifest.get("preset"),
        "rows": rows,
    }
    (suite_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_csv(suite_dir / "summary.csv", rows)
    write_markdown(suite_dir / "summary.md", rows)
    print(suite_dir / "summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
