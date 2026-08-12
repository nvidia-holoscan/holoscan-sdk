#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Summarize matched Channel execution-domain and transport benchmark results."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Runner combined JSON file(s), per-run output dir(s), or a parent directory.",
    )
    parser.add_argument("--format", choices=("markdown", "csv"), default="markdown")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def transport_name(domain: object, payload: object, publisher: dict[str, Any]) -> str:
    transport = publisher.get("transport")
    if isinstance(transport, str):
        return transport
    if payload == "pod":
        return "inline"
    return (
        "sidecar" if domain in {"intraprocess_sidecar", "intraprocess_serial"} else "buffer_handle"
    )


def result_from_output_dir(path: Path) -> dict[str, Any] | None:
    publisher = path / "publisher.json"
    if not publisher.exists():
        return None
    subscribers = sorted(path.glob("subscriber_*.json"))
    if not subscribers:
        return None
    publisher_json = read_json(publisher)
    subscriber_json = [read_json(item) for item in subscribers]
    domain = publisher_json.get("domain", "interprocess")
    payload = publisher_json.get("payload", "unknown")
    transport = transport_name(domain, payload, publisher_json)
    fanout = f"1to{len(subscriber_json)}"
    benchmark = f"channel_{domain}_{payload}_{fanout}"
    if payload in {"tensor", "image"}:
        benchmark = (
            f"channel_{domain}_{payload}_{publisher_json.get('tensor_storage', 'unknown')}_{fanout}"
        )
    if publisher_json.get("async_cuda_work") is True:
        benchmark += "_async_cuda"
    if publisher_json.get("workload") == "latency":
        benchmark += "_latency"
        if publisher_json.get("publisher_pacing") == "spin":
            benchmark += "_spin"
    return {
        "benchmark": benchmark,
        "domain": domain,
        "transport": transport,
        "output_dir": str(path),
        "channel_name": publisher_json.get("channel_name", ""),
        "publisher": publisher_json,
        "subscribers": subscriber_json,
    }


def result_identity(result: dict[str, Any]) -> str:
    channel_name = result.get("channel_name", "")
    if isinstance(channel_name, str) and channel_name:
        return channel_name
    output_dir = result.get("output_dir", "")
    if isinstance(output_dir, str) and output_dir:
        return output_dir
    return json.dumps(result, sort_keys=True)


def prefer_runner_result(current: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    current_has_runner_metadata = "hostname" in current or "publisher_cpu" in current
    candidate_has_runner_metadata = "hostname" in candidate or "publisher_cpu" in candidate
    if candidate_has_runner_metadata and not current_has_runner_metadata:
        return candidate
    return current


def deduplicate_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered_keys: list[str] = []
    deduped: dict[str, dict[str, Any]] = {}
    for result in results:
        key = result_identity(result)
        if key not in deduped:
            ordered_keys.append(key)
            deduped[key] = result
            continue
        deduped[key] = prefer_runner_result(deduped[key], result)
    return [deduped[key] for key in ordered_keys]


def discover_results(path: Path) -> list[dict[str, Any]]:
    if path.is_file():
        return [read_json(path)]
    direct = result_from_output_dir(path)
    if direct is not None:
        return [direct]
    results: list[dict[str, Any]] = []
    for child in sorted(path.iterdir()):
        if child.is_file() and child.suffix == ".json":
            try:
                result = read_json(child)
            except json.JSONDecodeError:
                continue
            if "publisher" in result and "subscribers" in result:
                results.append(result)
        elif child.is_dir():
            nested = result_from_output_dir(child)
            if nested is not None:
                results.append(nested)
    return deduplicate_results(results)


def get_number(mapping: dict[str, Any], key: str) -> float | int | str:
    value = mapping.get(key, "")
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float, str)):
        return value
    return ""


def subscriber_results(result: dict[str, Any]) -> list[dict[str, Any]]:
    subscribers = result.get("subscribers", [])
    if not isinstance(subscribers, list):
        return []
    return [subscriber for subscriber in subscribers if isinstance(subscriber, dict)]


def subscriber_metric(
    subscribers: list[dict[str, Any]], key: str, operation: str
) -> float | int | str:
    values = [subscriber.get(key) for subscriber in subscribers]
    numbers = [value for value in values if isinstance(value, (int, float))]
    if not numbers:
        return ""
    if operation == "min":
        return min(numbers)
    if operation == "sum":
        return sum(numbers)
    return max(numbers)


def comparison_key(result: dict[str, Any]) -> tuple[Any, ...]:
    publisher = result.get("publisher", {})
    if not isinstance(publisher, dict):
        publisher = {}
    subscriber_cpus = result.get("subscriber_cpus", [])
    if not isinstance(subscriber_cpus, list):
        subscriber_cpus = []
    fields = (
        "payload",
        "workload",
        "tensor_storage",
        "tensor_bytes",
        "samples",
        "warmup",
        "subscriber_queue_depth",
        "max_subscriber_queue_depth",
        "expected_subscribers",
        "publisher_period_us",
        "publisher_pacing",
        "async_cuda_work",
        "payload_validation",
    )
    return (
        *(publisher.get(field) for field in fields),
        result.get("hostname"),
        result.get("publisher_cpu"),
        tuple(subscriber_cpus),
    )


def result_completed_delivery_rate(result: dict[str, Any]) -> float | None:
    publisher = result.get("publisher", {})
    if not isinstance(publisher, dict):
        return None
    subscribers = subscriber_results(result)
    measured_start = publisher.get("measured_start_ns")
    measured_ends = [
        subscriber.get("measured_end_ns")
        for subscriber in subscribers
        if isinstance(subscriber.get("measured_end_ns"), int)
    ]
    if not subscribers or len(measured_ends) != len(subscribers):
        return None

    fanout = result.get("fanout", {})
    if isinstance(fanout, dict):
        rate = fanout.get("completed_deliveries_per_second")
        if isinstance(rate, (int, float)):
            return float(rate)

    samples = publisher.get("samples")
    if (
        not isinstance(measured_start, int)
        or not isinstance(samples, int)
        or max(measured_ends) <= measured_start
    ):
        return None
    completion_window = max(measured_ends) - measured_start
    return samples * len(subscribers) * 1.0e9 / completion_window


def rows(results: list[dict[str, Any]]) -> list[dict[str, float | int | str]]:
    matched_rates: dict[tuple[Any, ...], dict[str, float]] = {}
    for result in results:
        domain = result.get("domain")
        publisher = result.get("publisher", {})
        if not isinstance(domain, str):
            domain = publisher.get("domain") if isinstance(publisher, dict) else None
        workload = publisher.get("workload", "throughput") if isinstance(publisher, dict) else ""
        rate = result_completed_delivery_rate(result)
        if (
            workload == "throughput"
            and domain
            in {
                "intraprocess",
                "intraprocess_sidecar",
                "intraprocess_serial",
                "interprocess",
            }
            and rate is not None
        ):
            matched_rates.setdefault(comparison_key(result), {})[domain] = rate

    table: list[dict[str, float | int | str]] = []
    for result in results:
        publisher = result.get("publisher", {})
        if not isinstance(publisher, dict):
            publisher = {}
        subscribers = subscriber_results(result)
        subscriber_count = len(subscribers)
        logical_rate = get_number(publisher, "messages_per_second")
        publisher_delivery_rate = (
            logical_rate * subscriber_count if isinstance(logical_rate, (int, float)) else ""
        )
        fanout = result.get("fanout", {})
        if not isinstance(fanout, dict):
            fanout = {}
        measured_ends = [
            subscriber.get("measured_end_ns")
            for subscriber in subscribers
            if isinstance(subscriber.get("measured_end_ns"), int)
        ]
        has_complete_timing = subscriber_count > 0 and len(measured_ends) == subscriber_count
        completion_window: float | int | str = ""
        completed_logical_rate: float | int | str = ""
        completed_delivery_rate: float | int | str = ""
        if has_complete_timing:
            completion_window = get_number(fanout, "completion_window_ns")
            completed_logical_rate = get_number(fanout, "completed_logical_messages_per_second")
            completed_delivery_rate = get_number(fanout, "completed_deliveries_per_second")
        if has_complete_timing and completed_logical_rate == "":
            measured_start = publisher.get("measured_start_ns")
            samples = publisher.get("samples")
            if (
                isinstance(measured_start, int)
                and isinstance(samples, int)
                and max(measured_ends) > measured_start
            ):
                completion_window = max(measured_ends) - measured_start
                completed_logical_rate = samples * 1.0e9 / completion_window
                completed_delivery_rate = completed_logical_rate * subscriber_count
        rates = matched_rates.get(comparison_key(result), {})
        intraprocess_rate = rates.get("intraprocess")
        intraprocess_sidecar_rate = rates.get("intraprocess_sidecar")
        intraprocess_serial_rate = rates.get("intraprocess_serial")
        interprocess_rate = rates.get("interprocess")
        domain_rate_ratio: float | str = ""
        interprocess_over_sidecar_rate: float | str = ""
        sidecar_over_buffer_handle_rate: float | str = ""
        sidecar_over_serial_rate: float | str = ""
        if (
            intraprocess_rate is not None
            and intraprocess_rate > 0.0
            and interprocess_rate is not None
        ):
            domain_rate_ratio = interprocess_rate / intraprocess_rate
        if (
            intraprocess_sidecar_rate is not None
            and intraprocess_sidecar_rate > 0.0
            and interprocess_rate is not None
        ):
            interprocess_over_sidecar_rate = interprocess_rate / intraprocess_sidecar_rate
        if (
            intraprocess_rate is not None
            and intraprocess_rate > 0.0
            and intraprocess_sidecar_rate is not None
        ):
            sidecar_over_buffer_handle_rate = intraprocess_sidecar_rate / intraprocess_rate
        if (
            intraprocess_serial_rate is not None
            and intraprocess_serial_rate > 0.0
            and intraprocess_sidecar_rate is not None
        ):
            sidecar_over_serial_rate = intraprocess_sidecar_rate / intraprocess_serial_rate
        domain = result.get("domain", publisher.get("domain", ""))
        payload = publisher.get("payload", "")
        table.append(
            {
                "benchmark": get_number(result, "benchmark"),
                "domain": domain if isinstance(domain, str) else "",
                "transport": transport_name(domain, payload, publisher),
                "payload": payload if isinstance(payload, str) else "",
                "workload": get_number(publisher, "workload"),
                "rate_limited": get_number(publisher, "rate_limited"),
                "storage": get_number(publisher, "tensor_storage"),
                "tensor_bytes": get_number(publisher, "tensor_bytes"),
                "samples": get_number(publisher, "samples"),
                "queue_depth": get_number(publisher, "subscriber_queue_depth"),
                "period_us": get_number(publisher, "publisher_period_us"),
                "publisher_pacing": get_number(publisher, "publisher_pacing"),
                "subscribers": subscriber_count,
                "publisher_msgs_s": logical_rate,
                "publisher_deliveries_s": publisher_delivery_rate,
                "completion_window_ns": completion_window,
                "completed_logical_msgs_s": completed_logical_rate,
                "completed_deliveries_s": completed_delivery_rate,
                "interprocess_over_intraprocess_delivery_rate": domain_rate_ratio,
                "interprocess_over_sidecar_delivery_rate": interprocess_over_sidecar_rate,
                "sidecar_over_intraprocess_delivery_rate": sidecar_over_buffer_handle_rate,
                "concurrent_sidecar_over_serial_delivery_rate": sidecar_over_serial_rate,
                "sub_min_msgs_s": subscriber_metric(subscribers, "messages_per_second", "min"),
                "sub_max_msgs_s": subscriber_metric(subscribers, "messages_per_second", "max"),
                "emit_p50_ns": get_number(publisher, "channel_emit_p50_ns"),
                "emit_p99_ns": get_number(publisher, "channel_emit_p99_ns"),
                "emit_p999_ns": get_number(publisher, "channel_emit_p999_ns"),
                "worst_recv_p50_ns": subscriber_metric(
                    subscribers, "channel_receive_p50_ns", "max"
                ),
                "worst_recv_p99_ns": subscriber_metric(
                    subscribers, "channel_receive_p99_ns", "max"
                ),
                "worst_recv_p999_ns": subscriber_metric(
                    subscribers, "channel_receive_p999_ns", "max"
                ),
                "worst_e2e_p50_ns": subscriber_metric(subscribers, "end_to_end_p50_ns", "max"),
                "worst_e2e_p99_ns": subscriber_metric(subscribers, "end_to_end_p99_ns", "max"),
                "worst_e2e_p999_ns": subscriber_metric(subscribers, "end_to_end_p999_ns", "max"),
                "sequence_gaps": subscriber_metric(subscribers, "sequence_gaps", "sum"),
                "output_dir": get_number(result, "output_dir"),
            }
        )
    return table


def emit_markdown(table: list[dict[str, float | int | str]]) -> None:
    headers = list(table[0].keys()) if table else []
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join("---" for _ in headers) + " |")
    for row in table:
        print("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")


def emit_csv(table: list[dict[str, float | int | str]]) -> None:
    if not table:
        return
    writer = csv.DictWriter(sys.stdout, fieldnames=list(table[0].keys()))
    writer.writeheader()
    writer.writerows(table)


def main() -> int:
    args = parse_args()
    results: list[dict[str, Any]] = []
    for path in args.inputs:
        results.extend(discover_results(path))
    if not results:
        raise RuntimeError("no Channel benchmark results found")
    table = rows(results)
    if args.format == "csv":
        emit_csv(table)
    else:
        emit_markdown(table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
