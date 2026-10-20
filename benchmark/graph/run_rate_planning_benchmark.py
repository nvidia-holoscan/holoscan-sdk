#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run isolated cold rate-planning probes and retain raw evidence."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import tempfile
from pathlib import Path

import rate_evidence_utils as evidence

SCHEMA = "holoscan.ea2.rate-planning-benchmark/v1"
SUMMARY_SCHEMA = "holoscan.ea2.rate-planning-summary/v1"
COMPILE_TOPOLOGIES = (
    "linear_fixed",
    "fanout_fixed",
    "dual_rate_fusion",
    "multi_follows",
    "embedded_follows",
    "lifecycle_bridge",
    "lifecycle_cross_process_rejected",
)
HEADROOM_PAYLOADS = ("small", "large")
HEADROOM_MEMORY_KINDS = ("host", "pinned_host", "cuda_device")
HEADROOM_RATIOS = ("1/1", "2/1")
HEADROOM_QUEUE_POLICIES = ("blocking", "drop_oldest")
MAX_OPERATOR_COUNT = (1 << 32) - 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark_binary", type=Path)
    parser.add_argument(
        "--mode",
        action="append",
        choices=("compile_scaling", "headroom"),
        required=True,
        help="One or both benchmark families to execute",
    )
    parser.add_argument("--topology", action="append", choices=COMPILE_TOPOLOGIES, default=[])
    parser.add_argument("--operators", type=int, action="append", default=[])
    parser.add_argument("--payload", action="append", choices=HEADROOM_PAYLOADS, default=[])
    parser.add_argument("--memory", action="append", choices=HEADROOM_MEMORY_KINDS, default=[])
    parser.add_argument("--headroom", action="append", choices=HEADROOM_RATIOS, default=[])
    parser.add_argument(
        "--queue-policy", action="append", choices=HEADROOM_QUEUE_POLICIES, default=[]
    )
    parser.add_argument("--traffic-warmup-samples", type=int, default=34)
    parser.add_argument("--traffic-samples", type=int, default=64)
    parser.add_argument("--traffic-timeout-sec", type=float, default=10.0)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--process-timeout-sec", type=float, default=120.0)
    parser.add_argument("--git-revision", default=os.environ.get("HOLOSCAN_GIT_REVISION"))
    parser.add_argument("--source-state", default="unknown")
    parser.add_argument("--source-snapshot-sha256", default=None)
    return parser.parse_args()


def valid_sha256(value: object) -> bool:
    return evidence.valid_sha256(value)


def validate_args(args: argparse.Namespace) -> None:
    if any(size < 0 or size > MAX_OPERATOR_COUNT for size in args.operators):
        raise ValueError("--operators must be within the uint32 admission domain")
    if (
        args.repetitions <= 0
        or args.process_timeout_sec <= 0
        or args.traffic_warmup_samples < 0
        or args.traffic_samples <= 0
        or args.traffic_timeout_sec <= 0
    ):
        raise ValueError(
            "repetitions, timeouts, and traffic samples must be positive; warmup is nonnegative"
        )
    if args.traffic_warmup_samples + args.traffic_samples > MAX_OPERATOR_COUNT:
        raise ValueError("traffic samples must fit the uint32 admission domain")
    if args.source_snapshot_sha256 is not None and not valid_sha256(args.source_snapshot_sha256):
        raise ValueError("--source-snapshot-sha256 must be a lowercase SHA-256 digest")
    selected_modes = set(args.mode)
    if "compile_scaling" in selected_modes and (not args.topology or not args.operators):
        raise ValueError("compile_scaling requires at least one --topology and one --operators")
    if "headroom" in selected_modes and (not args.payload or not args.memory or not args.headroom):
        raise ValueError("headroom requires --payload, --memory, and --headroom")


def percentile(values: list[int | float], quantile: float) -> int | float:
    return evidence.nearest_rank_percentile(values, quantile)


def summarize(values: list[int | float]) -> dict[str, int | float]:
    return {
        "minimum": min(values),
        "median": statistics.median(values),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "maximum": max(values),
    }


def file_sha256(path: Path) -> str:
    return evidence.file_sha256(path)


def verify_binary_sha256(binary: Path, expected_sha256: str, stage: str) -> None:
    evidence.require_file_sha256(
        binary, expected_sha256, f"rate-planning benchmark binary changed {stage}"
    )


def build_metadata(binary: Path) -> dict[str, object]:
    return evidence.cmake_build_metadata(binary, include_compiler_version_output=True)


def nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def valid_traffic_outcomes(value: object) -> bool:
    if not isinstance(value, dict) or value.get("availability") not in {
        "available",
        "unavailable",
    }:
        return False
    common = (
        value.get("observation_scope") == "finite_realtime_1khz_to_60hz_drain_boundaries"
        and value.get("latency_scope") == "last_source_attempt_to_drain_boundary"
        and value.get("throughput_scope")
        == "last_attempt_of_first_measured_sample_to_final_sample_drain"
        and nonnegative_int(value.get("warmup_samples"))
        and nonnegative_int(value.get("requested_samples"))
        and value["requested_samples"] > 0
    )
    measured_fields = (
        "accepted_publications",
        "delivered_samples",
        "sampled_occupancy_high_water",
        "backpressured_emit_attempts",
        "resource_exhausted_source_attempts",
        "not_ready_source_attempts",
        "consumer_visible_replacements",
        "measurement_duration_ns",
    )
    if value.get("availability") == "unavailable":
        return (
            common
            and isinstance(value.get("unavailable_reason"), str)
            and all(value.get(field) is None for field in measured_fields)
            and value.get("finite_pipeline_accepted_publications_per_second") is None
            and value.get("finite_pipeline_deliveries_per_second") is None
            and value.get("latency_ns") is None
        )

    latency = value.get("latency_ns")
    throughput = value.get("finite_pipeline_accepted_publications_per_second")
    delivery_rate = value.get("finite_pipeline_deliveries_per_second")
    return (
        common
        and value.get("unavailable_reason") is None
        and all(nonnegative_int(value.get(field)) for field in measured_fields)
        and value["accepted_publications"] == value["requested_samples"]
        and value["delivered_samples"] + value["consumer_visible_replacements"]
        == value["accepted_publications"]
        and value["measurement_duration_ns"] > 0
        and isinstance(throughput, (int, float))
        and not isinstance(throughput, bool)
        and math.isfinite(throughput)
        and throughput > 0
        and isinstance(delivery_rate, (int, float))
        and not isinstance(delivery_rate, bool)
        and math.isfinite(delivery_rate)
        and delivery_rate > 0
        and isinstance(latency, dict)
        and all(
            nonnegative_int(latency.get(field))
            for field in ("samples", "minimum", "p50", "p95", "p99", "maximum")
        )
        and latency["samples"] == value["delivered_samples"]
        and latency["samples"] > 0
        and latency["minimum"] <= latency["p50"] <= latency["p95"]
        and latency["p95"] <= latency["p99"] <= latency["maximum"]
    )


def checked_result(path: Path) -> dict[str, object]:
    result = json.loads(path.read_text(encoding="utf-8"))
    common_numeric = {
        "compile_ns",
        "allocation_count",
        "cumulative_requested_bytes",
        "peak_live_requested_bytes",
        "live_requested_bytes_after_compile",
        "live_requested_bytes_at_probe_end",
        "process_max_rss_kib",
    }
    plan_retained = result.get("plan_retained_at_probe_end")
    compile_disposition = result.get("compile_disposition")
    common_valid = (
        result.get("schema") == SCHEMA
        and result.get("warmup_compile") == "accepted_zero_work"
        and all(isinstance(result.get(key), int) and result[key] >= 0 for key in common_numeric)
        and isinstance(plan_retained, bool)
        and compile_disposition in {"accepted", "rejected"}
        and (
            (
                compile_disposition == "accepted"
                and isinstance(result.get("plan_digest"), str)
                and len(result["plan_digest"]) == 64
            )
            or (compile_disposition == "rejected" and result.get("plan_digest") is None)
        )
    )
    if result.get("mode") == "compile_scaling":
        relations = result.get("rate_dependency_relations")
        valid = (
            common_valid
            and result.get("topology") in COMPILE_TOPOLOGIES
            and isinstance(result.get("operator_count"), int)
            and result["operator_count"] >= 0
            and isinstance(result.get("edge_count"), int)
            and isinstance(result.get("rate_port_count"), int)
            and isinstance(result.get("rate_dependency_count"), int)
            and isinstance(relations, dict)
            and isinstance(relations.get("follows_input_count"), int)
            and isinstance(relations.get("inverse_rate_horizon_count"), int)
            and isinstance(result.get("queue_recipe_count"), int)
            and isinstance(result.get("automatic_depth_edge_count"), int)
            and isinstance(result.get("lifecycle_candidate_recipe_count"), int)
            and isinstance(result.get("compile_concrete_provider_count"), int)
            and isinstance(result.get("diagnostic_tokens"), list)
            and all(isinstance(token, str) for token in result["diagnostic_tokens"])
        )
    elif result.get("mode") == "headroom":
        available = result.get("availability") == "available"
        headroom = result.get("headroom_ratio")
        queue_headroom = result.get("queue_recipe_headroom_ratio")
        valid = (
            common_valid
            and result.get("payload_profile") in HEADROOM_PAYLOADS
            and result.get("memory_kind") in HEADROOM_MEMORY_KINDS
            and result.get("queue_policy") in HEADROOM_QUEUE_POLICIES
            and isinstance(headroom, dict)
            and isinstance(headroom.get("numerator"), int)
            and isinstance(headroom.get("denominator"), int)
            and f"{headroom.get('numerator')}/{headroom.get('denominator')}" in HEADROOM_RATIOS
            and result.get("availability") in {"available", "unavailable"}
            and isinstance(result.get("source_nominal_hz"), int)
            and isinstance(result.get("source_period_ns"), int)
            and result["source_period_ns"] > 0
            and isinstance(result.get("sink_period_ns"), int)
            and isinstance(result.get("queue_recipe_count"), int)
            and isinstance(queue_headroom, dict)
            and queue_headroom == headroom
            and (
                result.get("startup_to_published_generation_ns") is None
                or isinstance(result.get("startup_to_published_generation_ns"), int)
            )
            and result.get("restart_latency_ns") is None
            and isinstance(result.get("restart_latency_unavailable_reason"), str)
            and valid_traffic_outcomes(result.get("traffic_outcomes"))
        )
        if valid and available:
            unique = result.get("unique_pool_summary")
            valid = (
                result.get("unavailable_reason") is None
                and isinstance(result.get("queue_effective_logical_depth"), int)
                and isinstance(result.get("queue_rate_recommendation"), int)
                and isinstance(result.get("queue_nominal_arrivals"), int)
                and isinstance(result.get("pool_capacity_generation_id"), int)
                and isinstance(result.get("pool_required_entries"), int)
                and (
                    result.get("pool_logical_payload_capacity_bytes") is None
                    or isinstance(result.get("pool_logical_payload_capacity_bytes"), int)
                )
                and (
                    result.get("pool_max_payload_byte_span_per_slot") is None
                    or isinstance(result.get("pool_max_payload_byte_span_per_slot"), int)
                )
                and isinstance(result.get("provider_action"), str)
                and isinstance(result.get("provider_decision_id"), int)
                and isinstance(result.get("provider_plan_id"), int)
                and (
                    result.get("allocation_generation_id") is None
                    or isinstance(result.get("allocation_generation_id"), int)
                )
                and (
                    result.get("storage_binding_generation_id") is None
                    or isinstance(result.get("storage_binding_generation_id"), int)
                )
                and isinstance(result.get("provider_materialized_entries"), int)
                and (
                    result.get("provider_stride_bytes_per_entry") is None
                    or isinstance(result.get("provider_stride_bytes_per_entry"), int)
                )
                and (
                    result.get("provider_reserved_bytes_total") is None
                    or isinstance(result.get("provider_reserved_bytes_total"), int)
                )
                and (
                    (
                        result.get("provider_reserved_bytes_total") is None
                        and isinstance(
                            result.get("provider_reserved_bytes_unavailable_reason"), str
                        )
                    )
                    or (
                        isinstance(result.get("provider_reserved_bytes_total"), int)
                        and result.get("provider_reserved_bytes_unavailable_reason") is None
                    )
                )
                and result.get("physical_queue_entries") is None
                and result.get("physical_ring_entries") is None
                and isinstance(result.get("physical_queue_geometry_unavailable_reason"), str)
                and isinstance(result.get("inspection_json"), str)
                and isinstance(unique, dict)
                and isinstance(unique.get("pool_count"), int)
                and isinstance(unique.get("slot_count"), int)
                and isinstance(unique.get("aggregate_byte_capacity"), int)
                and isinstance(unique.get("memory_kinds"), list)
                and isinstance(unique.get("devices"), list)
                and (
                    result["traffic_outcomes"]["availability"] != "available"
                    or result["traffic_outcomes"]["sampled_occupancy_high_water"]
                    <= result["queue_effective_logical_depth"]
                )
                and (
                    result.get("queue_policy") != "blocking"
                    or result["traffic_outcomes"]["availability"] != "available"
                    or result["traffic_outcomes"]["consumer_visible_replacements"] == 0
                )
            )
        elif valid:
            valid = (
                isinstance(result.get("unavailable_reason"), str)
                and result.get("queue_effective_logical_depth") is None
                and result.get("queue_rate_recommendation") is None
                and result.get("queue_nominal_arrivals") is None
                and result.get("pool_capacity_generation_id") is None
                and result.get("pool_required_entries") is None
                and result.get("pool_logical_payload_capacity_bytes") is None
                and result.get("pool_max_payload_byte_span_per_slot") is None
                and result.get("provider_action") is None
                and result.get("provider_decision_id") is None
                and result.get("provider_plan_id") is None
                and result.get("allocation_generation_id") is None
                and result.get("storage_binding_generation_id") is None
                and result.get("provider_materialized_entries") is None
                and result.get("provider_stride_bytes_per_entry") is None
                and result.get("provider_reserved_bytes_total") is None
                and isinstance(result.get("provider_reserved_bytes_unavailable_reason"), str)
                and result.get("physical_queue_entries") is None
                and result.get("physical_ring_entries") is None
                and isinstance(result.get("physical_queue_geometry_unavailable_reason"), str)
                and result.get("unique_pool_summary") is None
                and result.get("inspection_json") is None
                and result["traffic_outcomes"]["availability"] == "unavailable"
            )
    else:
        valid = False
    if not valid:
        raise RuntimeError(f"invalid rate-planning result: {path}")
    return result


def aggregate_unique_pool_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    pools: dict[str, dict[str, object]] = {}
    for row in rows:
        pool_id = str(row["pool_id"])
        pools.setdefault(pool_id, row)
    memory_kinds = sorted({str(row["memory_kind"]) for row in pools.values()})
    devices = sorted(
        {
            int(device)
            for device in (row.get("device") for row in pools.values())
            if isinstance(device, int)
        }
    )
    return {
        "pool_count": len(pools),
        "slot_count": sum(int(row["slot_count"]) for row in pools.values()),
        "aggregate_byte_capacity": sum(
            int(row["aggregate_byte_capacity"]) for row in pools.values()
        ),
        "memory_kinds": memory_kinds,
        "devices": devices,
    }


def headroom_key(result: dict[str, object]) -> tuple[str, str, str, str]:
    ratio = result["headroom_ratio"]
    return (
        str(result["payload_profile"]),
        str(result["memory_kind"]),
        str(result["queue_policy"]),
        f"{ratio['numerator']}/{ratio['denominator']}",
    )


def summarize_compile_case(
    topology: str, operators: int, results: list[dict[str, object]]
) -> dict[str, object]:
    digests = {result["plan_digest"] for result in results}
    dispositions = {str(result["compile_disposition"]) for result in results}
    identity = {
        (
            int(result["edge_count"]),
            int(result["rate_port_count"]),
            int(result["rate_dependency_count"]),
            int(result["rate_dependency_relations"]["follows_input_count"]),
            int(result["rate_dependency_relations"]["inverse_rate_horizon_count"]),
            int(result["queue_recipe_count"]),
            int(result["automatic_depth_edge_count"]),
            int(result["lifecycle_candidate_recipe_count"]),
            int(result["compile_concrete_provider_count"]),
            tuple(str(token) for token in result["diagnostic_tokens"]),
        )
        for result in results
    }
    return {
        "topology": topology,
        "operators": operators,
        "repetitions": len(results),
        "status": "characterization_only",
        "deterministic_plan": len(digests) == 1 and len(identity) == 1 and len(dispositions) == 1,
        **{
            key: summarize([int(result[key]) for result in results])
            for key in (
                "compile_ns",
                "allocation_count",
                "cumulative_requested_bytes",
                "peak_live_requested_bytes",
                "live_requested_bytes_after_compile",
                "live_requested_bytes_at_probe_end",
                "process_max_rss_kib",
            )
        },
        "plan_digest": next(iter(digests)) if len(digests) == 1 else None,
        "plan_retained_at_probe_end": results[0]["plan_retained_at_probe_end"],
        "compile_disposition": next(iter(dispositions)) if len(dispositions) == 1 else None,
        "edge_count": results[0]["edge_count"],
        "rate_port_count": results[0]["rate_port_count"],
        "rate_dependency_count": results[0]["rate_dependency_count"],
        "rate_dependency_relations": results[0]["rate_dependency_relations"],
        "queue_recipe_count": results[0]["queue_recipe_count"],
        "automatic_depth_edge_count": results[0]["automatic_depth_edge_count"],
        "lifecycle_candidate_recipe_count": results[0]["lifecycle_candidate_recipe_count"],
        "compile_concrete_provider_count": results[0]["compile_concrete_provider_count"],
        "diagnostic_tokens": results[0]["diagnostic_tokens"],
    }


def summarize_traffic_outcomes(results: list[dict[str, object]]) -> dict[str, object]:
    outcomes = [result.get("traffic_outcomes") for result in results]
    available = bool(outcomes) and all(
        isinstance(outcome, dict) and outcome.get("availability") == "available"
        for outcome in outcomes
    )
    if not available:
        return {
            "availability": "unavailable",
            "repetitions": len(results),
            "unavailable_reasons": sorted(
                {
                    str(outcome.get("unavailable_reason"))
                    for outcome in outcomes
                    if isinstance(outcome, dict) and outcome.get("unavailable_reason") is not None
                }
            ),
            "observation_scope": (
                outcomes[0].get("observation_scope") if isinstance(outcomes[0], dict) else None
            )
            if outcomes
            else None,
            "latency_scope": (
                outcomes[0].get("latency_scope") if isinstance(outcomes[0], dict) else None
            )
            if outcomes
            else None,
            "throughput_scope": (
                outcomes[0].get("throughput_scope") if isinstance(outcomes[0], dict) else None
            )
            if outcomes
            else None,
            "accepted_publications": None,
            "delivered_samples": None,
            "sampled_occupancy_high_water": None,
            "backpressured_emit_attempts": None,
            "resource_exhausted_source_attempts": None,
            "not_ready_source_attempts": None,
            "consumer_visible_replacements": None,
            "measurement_duration_ns": None,
            "finite_pipeline_accepted_publications_per_second": None,
            "finite_pipeline_deliveries_per_second": None,
            "latency_p50_ns": None,
            "latency_p95_ns": None,
            "latency_p99_ns": None,
        }

    typed_outcomes = [outcome for outcome in outcomes if isinstance(outcome, dict)]
    return {
        "availability": "available",
        "repetitions": len(typed_outcomes),
        "unavailable_reasons": [],
        "observation_scope": typed_outcomes[0]["observation_scope"],
        "latency_scope": typed_outcomes[0]["latency_scope"],
        "throughput_scope": typed_outcomes[0]["throughput_scope"],
        **{
            field: summarize([outcome[field] for outcome in typed_outcomes])
            for field in (
                "accepted_publications",
                "delivered_samples",
                "sampled_occupancy_high_water",
                "backpressured_emit_attempts",
                "resource_exhausted_source_attempts",
                "not_ready_source_attempts",
                "consumer_visible_replacements",
                "measurement_duration_ns",
                "finite_pipeline_accepted_publications_per_second",
                "finite_pipeline_deliveries_per_second",
            )
        },
        "latency_p50_ns": summarize([outcome["latency_ns"]["p50"] for outcome in typed_outcomes]),
        "latency_p95_ns": summarize([outcome["latency_ns"]["p95"] for outcome in typed_outcomes]),
        "latency_p99_ns": summarize([outcome["latency_ns"]["p99"] for outcome in typed_outcomes]),
    }


def summarize_headroom_pair(
    payload: str,
    memory_kind: str,
    queue_policy: str,
    one_one_results: list[dict[str, object]],
    two_one_results: list[dict[str, object]],
) -> dict[str, object]:
    def expected_ratio(value: str) -> dict[str, int]:
        numerator, denominator = value.split("/", 1)
        return {"numerator": int(numerator), "denominator": int(denominator)}

    def stable_identity(row: dict[str, object]) -> dict[str, object]:
        traffic = row["traffic_outcomes"]
        return {
            "mode": row["mode"],
            "payload_profile": row["payload_profile"],
            "memory_kind": row["memory_kind"],
            "queue_policy": row["queue_policy"],
            "availability": row["availability"],
            "source_nominal_hz": row["source_nominal_hz"],
            "source_period_ns": row["source_period_ns"],
            "sink_period_ns": row["sink_period_ns"],
            "traffic_observation_scope": traffic["observation_scope"],
            "traffic_latency_scope": traffic["latency_scope"],
            "traffic_throughput_scope": traffic["throughput_scope"],
            "traffic_warmup_samples": traffic["warmup_samples"],
            "traffic_requested_samples": traffic["requested_samples"],
        }

    def static_ratio_identity(row: dict[str, object]) -> dict[str, object]:
        return {
            "headroom_ratio": row.get("headroom_ratio"),
            "queue_recipe_headroom_ratio": row.get("queue_recipe_headroom_ratio"),
            "availability": row.get("availability"),
            "unavailable_reason": row.get("unavailable_reason"),
            "plan_digest": row.get("plan_digest"),
            "queue_effective_logical_depth": row.get("queue_effective_logical_depth"),
            "queue_rate_recommendation": row.get("queue_rate_recommendation"),
            "queue_nominal_arrivals": row.get("queue_nominal_arrivals"),
            "pool_capacity_generation_id": row.get("pool_capacity_generation_id"),
            "pool_required_entries": row.get("pool_required_entries"),
            "pool_logical_payload_capacity_bytes": row.get("pool_logical_payload_capacity_bytes"),
            "pool_max_payload_byte_span_per_slot": row.get("pool_max_payload_byte_span_per_slot"),
            "provider_action": row.get("provider_action"),
            "provider_decision_id": row.get("provider_decision_id"),
            "provider_plan_id": row.get("provider_plan_id"),
            "allocation_generation_id": row.get("allocation_generation_id"),
            "storage_binding_generation_id": row.get("storage_binding_generation_id"),
            "provider_materialized_entries": row.get("provider_materialized_entries"),
            "provider_stride_bytes_per_entry": row.get("provider_stride_bytes_per_entry"),
            "provider_reserved_bytes_total": row.get("provider_reserved_bytes_total"),
            "provider_reserved_bytes_unavailable_reason": row.get(
                "provider_reserved_bytes_unavailable_reason"
            ),
            "physical_queue_entries": row.get("physical_queue_entries"),
            "physical_ring_entries": row.get("physical_ring_entries"),
            "physical_queue_geometry_unavailable_reason": row.get(
                "physical_queue_geometry_unavailable_reason"
            ),
            "unique_pool_summary": row.get("unique_pool_summary"),
            "inspection_json": row.get("inspection_json"),
        }

    def summarize_optional_numeric(
        results: list[dict[str, object]], field: str
    ) -> dict[str, object] | None:
        values = [row.get(field) for row in results]
        if not values or any(value is None for value in values):
            return None
        return summarize(values)

    def validate_ratio_group(results: list[dict[str, object]], ratio: str) -> None:
        if not results:
            return
        ratio_value = expected_ratio(ratio)
        reference_workload = stable_identity(results[0])
        reference_static = static_ratio_identity(results[0])
        for row in results:
            if (
                stable_identity(row) != reference_workload
                or row.get("payload_profile") != payload
                or row.get("memory_kind") != memory_kind
                or row.get("queue_policy") != queue_policy
                or row.get("headroom_ratio") != ratio_value
                or row.get("queue_recipe_headroom_ratio") != ratio_value
                or static_ratio_identity(row) != reference_static
            ):
                raise RuntimeError(
                    f"headroom {ratio} repetitions disagree for "
                    f"{payload}/{memory_kind}/{queue_policy}"
                )

    validate_ratio_group(one_one_results, "1/1")
    validate_ratio_group(two_one_results, "2/1")
    paired_identity_match = bool(one_one_results and two_one_results) and stable_identity(
        one_one_results[0]
    ) == stable_identity(two_one_results[0])
    availability = (
        "available"
        if one_one_results
        and two_one_results
        and all(row["availability"] == "available" for row in [*one_one_results, *two_one_results])
        else "unavailable"
    )
    summary = {
        "payload_profile": payload,
        "memory_kind": memory_kind,
        "queue_policy": queue_policy,
        "repetitions": {"1/1": len(one_one_results), "2/1": len(two_one_results)},
        "status": "characterization_only",
        "paired_identity_match": paired_identity_match,
        "static_identity_stable_within_ratio": {
            "1/1": bool(one_one_results),
            "2/1": bool(two_one_results),
        },
        "availability": availability,
        "source_nominal_hz": one_one_results[0]["source_nominal_hz"] if one_one_results else None,
        "source_period_ns": one_one_results[0]["source_period_ns"] if one_one_results else None,
        "sink_period_ns": one_one_results[0]["sink_period_ns"] if one_one_results else None,
        "queue_effective_logical_depth": {
            "1/1": one_one_results[0].get("queue_effective_logical_depth")
            if one_one_results
            else None,
            "2/1": two_one_results[0].get("queue_effective_logical_depth")
            if two_one_results
            else None,
        },
        "queue_rate_recommendation": {
            "1/1": one_one_results[0].get("queue_rate_recommendation") if one_one_results else None,
            "2/1": two_one_results[0].get("queue_rate_recommendation") if two_one_results else None,
        },
        "startup_to_published_generation_ns": {
            "1/1": summarize_optional_numeric(
                one_one_results, "startup_to_published_generation_ns"
            ),
            "2/1": summarize_optional_numeric(
                two_one_results, "startup_to_published_generation_ns"
            ),
        },
        "restart_latency_ns": {
            "1/1": one_one_results[0].get("restart_latency_ns") if one_one_results else None,
            "2/1": two_one_results[0].get("restart_latency_ns") if two_one_results else None,
        },
        "restart_latency_unavailable_reason": {
            "1/1": one_one_results[0].get("restart_latency_unavailable_reason")
            if one_one_results
            else None,
            "2/1": two_one_results[0].get("restart_latency_unavailable_reason")
            if two_one_results
            else None,
        },
        "pool_capacity_generation_id": {
            "1/1": one_one_results[0].get("pool_capacity_generation_id")
            if one_one_results
            else None,
            "2/1": two_one_results[0].get("pool_capacity_generation_id")
            if two_one_results
            else None,
        },
        "pool_required_entries": {
            "1/1": one_one_results[0].get("pool_required_entries") if one_one_results else None,
            "2/1": two_one_results[0].get("pool_required_entries") if two_one_results else None,
        },
        "pool_logical_payload_capacity_bytes": {
            "1/1": one_one_results[0].get("pool_logical_payload_capacity_bytes")
            if one_one_results
            else None,
            "2/1": two_one_results[0].get("pool_logical_payload_capacity_bytes")
            if two_one_results
            else None,
        },
        "pool_max_payload_byte_span_per_slot": {
            "1/1": one_one_results[0].get("pool_max_payload_byte_span_per_slot")
            if one_one_results
            else None,
            "2/1": two_one_results[0].get("pool_max_payload_byte_span_per_slot")
            if two_one_results
            else None,
        },
        "provider_action": {
            "1/1": one_one_results[0].get("provider_action") if one_one_results else None,
            "2/1": two_one_results[0].get("provider_action") if two_one_results else None,
        },
        "provider_decision_id": {
            "1/1": one_one_results[0].get("provider_decision_id") if one_one_results else None,
            "2/1": two_one_results[0].get("provider_decision_id") if two_one_results else None,
        },
        "provider_plan_id": {
            "1/1": one_one_results[0].get("provider_plan_id") if one_one_results else None,
            "2/1": two_one_results[0].get("provider_plan_id") if two_one_results else None,
        },
        "allocation_generation_id": {
            "1/1": one_one_results[0].get("allocation_generation_id") if one_one_results else None,
            "2/1": two_one_results[0].get("allocation_generation_id") if two_one_results else None,
        },
        "storage_binding_generation_id": {
            "1/1": one_one_results[0].get("storage_binding_generation_id")
            if one_one_results
            else None,
            "2/1": two_one_results[0].get("storage_binding_generation_id")
            if two_one_results
            else None,
        },
        "provider_materialized_entries": {
            "1/1": one_one_results[0].get("provider_materialized_entries")
            if one_one_results
            else None,
            "2/1": two_one_results[0].get("provider_materialized_entries")
            if two_one_results
            else None,
        },
        "provider_stride_bytes_per_entry": {
            "1/1": one_one_results[0].get("provider_stride_bytes_per_entry")
            if one_one_results
            else None,
            "2/1": two_one_results[0].get("provider_stride_bytes_per_entry")
            if two_one_results
            else None,
        },
        "provider_reserved_bytes_total": {
            "1/1": one_one_results[0].get("provider_reserved_bytes_total")
            if one_one_results
            else None,
            "2/1": two_one_results[0].get("provider_reserved_bytes_total")
            if two_one_results
            else None,
        },
        "provider_reserved_bytes_unavailable_reason": {
            "1/1": one_one_results[0].get("provider_reserved_bytes_unavailable_reason")
            if one_one_results
            else None,
            "2/1": two_one_results[0].get("provider_reserved_bytes_unavailable_reason")
            if two_one_results
            else None,
        },
        "physical_queue_entries": {
            "1/1": one_one_results[0].get("physical_queue_entries") if one_one_results else None,
            "2/1": two_one_results[0].get("physical_queue_entries") if two_one_results else None,
        },
        "physical_ring_entries": {
            "1/1": one_one_results[0].get("physical_ring_entries") if one_one_results else None,
            "2/1": two_one_results[0].get("physical_ring_entries") if two_one_results else None,
        },
        "physical_queue_geometry_unavailable_reason": {
            "1/1": one_one_results[0].get("physical_queue_geometry_unavailable_reason")
            if one_one_results
            else None,
            "2/1": two_one_results[0].get("physical_queue_geometry_unavailable_reason")
            if two_one_results
            else None,
        },
        "unique_pool_summary": {
            "1/1": one_one_results[0].get("unique_pool_summary") if one_one_results else None,
            "2/1": two_one_results[0].get("unique_pool_summary") if two_one_results else None,
        },
        "plan_digest": {
            "1/1": one_one_results[0].get("plan_digest") if one_one_results else None,
            "2/1": two_one_results[0].get("plan_digest") if two_one_results else None,
        },
        "unavailable_reason": {
            "1/1": one_one_results[0].get("unavailable_reason") if one_one_results else None,
            "2/1": two_one_results[0].get("unavailable_reason") if two_one_results else None,
        },
        "traffic_outcomes": {
            "1/1": summarize_traffic_outcomes(one_one_results),
            "2/1": summarize_traffic_outcomes(two_one_results),
        },
    }
    return summary


def run_case(
    command: list[str], case_dir: Path, timeout: float, expected_binary_sha256: str
) -> dict[str, object]:
    case_dir.mkdir(parents=True, exist_ok=True)
    result_path = case_dir / "result.json"
    result_path.unlink(missing_ok=True)
    binary = Path(command[0]).resolve(strict=True)
    verify_binary_sha256(binary, expected_binary_sha256, "before child launch")
    try:
        completed = evidence.run_logged_child(
            [*command, "--output", str(result_path)], case_dir, timeout
        )
    finally:
        verify_binary_sha256(binary, expected_binary_sha256, "after child completion")
    if completed.returncode != 0:
        raise RuntimeError(f"rate-planning probe exited {completed.returncode}: {case_dir}")
    if not result_path.is_file():
        raise RuntimeError(f"rate-planning probe did not create a fresh result: {case_dir}")
    return checked_result(result_path)


def main() -> int:
    args = parse_args()
    validate_args(args)
    binary = args.benchmark_binary.resolve(strict=True)
    benchmark_binary_sha256 = file_sha256(binary)
    if args.output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="holoscan-rate-planning-"))
    else:
        output_dir = args.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

    compile_cases: list[dict[str, object]] = []
    headroom_rows: list[dict[str, object]] = []

    if "compile_scaling" in set(args.mode):
        for topology in args.topology:
            for operators in args.operators:
                results = [
                    run_case(
                        [
                            str(binary),
                            "--mode",
                            "compile_scaling",
                            "--topology",
                            topology,
                            "--operators",
                            str(operators),
                        ],
                        output_dir
                        / "compile_scaling"
                        / topology
                        / f"operators_{operators}"
                        / f"repetition_{repetition:03d}",
                        args.process_timeout_sec,
                        benchmark_binary_sha256,
                    )
                    for repetition in range(1, args.repetitions + 1)
                ]
                compile_cases.append(summarize_compile_case(topology, operators, results))

    if "headroom" in set(args.mode):
        raw_rows: dict[tuple[str, str, str, str], list[dict[str, object]]] = {}
        queue_policies = args.queue_policy or list(HEADROOM_QUEUE_POLICIES)
        for payload in args.payload:
            for memory in args.memory:
                for queue_policy in queue_policies:
                    for headroom in args.headroom:
                        results = [
                            run_case(
                                [
                                    str(binary),
                                    "--mode",
                                    "headroom",
                                    "--payload",
                                    payload,
                                    "--memory",
                                    memory,
                                    "--queue-policy",
                                    queue_policy,
                                    "--headroom",
                                    headroom,
                                    "--traffic-warmup-samples",
                                    str(args.traffic_warmup_samples),
                                    "--traffic-samples",
                                    str(args.traffic_samples),
                                    "--traffic-timeout-ms",
                                    str(max(1, round(args.traffic_timeout_sec * 1000))),
                                ],
                                output_dir
                                / "headroom"
                                / payload
                                / memory
                                / queue_policy
                                / headroom.replace("/", "_")
                                / f"repetition_{repetition:03d}",
                                args.process_timeout_sec,
                                benchmark_binary_sha256,
                            )
                            for repetition in range(1, args.repetitions + 1)
                        ]
                        raw_rows[(payload, memory, queue_policy, headroom)] = results
        headroom_rows.extend(
            summarize_headroom_pair(
                payload,
                memory,
                queue_policy,
                raw_rows.get((payload, memory, queue_policy, "1/1"), []),
                raw_rows.get((payload, memory, queue_policy, "2/1"), []),
            )
            for payload in args.payload
            for memory in args.memory
            for queue_policy in queue_policies
        )

    verify_binary_sha256(binary, benchmark_binary_sha256, "before summary publication")
    binary_build_metadata = build_metadata(binary)
    verify_binary_sha256(binary, benchmark_binary_sha256, "after build metadata capture")
    summary = {
        "schema": SUMMARY_SCHEMA,
        "status": "characterization_only",
        "benchmark_binary": str(binary),
        "benchmark_binary_sha256": benchmark_binary_sha256,
        "git_revision": args.git_revision,
        "source_state": args.source_state,
        "source_snapshot_sha256": args.source_snapshot_sha256,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "build_metadata": binary_build_metadata,
        "compile_scaling_cases": compile_cases,
        "headroom_pairs": headroom_rows,
        "artifact_root": str(output_dir),
    }
    verify_binary_sha256(binary, benchmark_binary_sha256, "immediately before summary write")
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    try:
        verify_binary_sha256(binary, benchmark_binary_sha256, "after summary publication")
    except RuntimeError:
        summary_path.unlink(missing_ok=True)
        raise
    print(str(summary_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
