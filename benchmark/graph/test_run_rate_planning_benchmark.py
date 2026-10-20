# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).with_name("run_rate_planning_benchmark.py")
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("rate_planning_runner", MODULE_PATH)
assert SPEC
assert SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def available_traffic_outcomes(*, high_water: int = 17, replacements: int = 0) -> dict[str, object]:
    delivered = 64 - replacements
    return {
        "availability": "available",
        "unavailable_reason": None,
        "observation_scope": "finite_realtime_1khz_to_60hz_drain_boundaries",
        "latency_scope": "last_source_attempt_to_drain_boundary",
        "throughput_scope": "last_attempt_of_first_measured_sample_to_final_sample_drain",
        "warmup_samples": 34,
        "requested_samples": 64,
        "accepted_publications": 64,
        "delivered_samples": delivered,
        "sampled_occupancy_high_water": high_water,
        "backpressured_emit_attempts": 0,
        "resource_exhausted_source_attempts": 0,
        "not_ready_source_attempts": 0,
        "consumer_visible_replacements": replacements,
        "measurement_duration_ns": 64_000_000,
        "finite_pipeline_accepted_publications_per_second": 1000.0,
        "finite_pipeline_deliveries_per_second": delivered / 0.064,
        "latency_ns": {
            "samples": delivered,
            "minimum": 1,
            "p50": 8,
            "p95": 16,
            "p99": 17,
            "maximum": 17,
        },
    }


def unavailable_traffic_outcomes() -> dict[str, object]:
    return {
        "availability": "unavailable",
        "unavailable_reason": "CUDA device 0 unavailable",
        "observation_scope": "finite_realtime_1khz_to_60hz_drain_boundaries",
        "latency_scope": "last_source_attempt_to_drain_boundary",
        "throughput_scope": "last_attempt_of_first_measured_sample_to_final_sample_drain",
        "warmup_samples": 34,
        "requested_samples": 64,
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
        "latency_ns": None,
    }


def test_checked_compile_scaling_result_accepts_complete_probe(tmp_path: Path) -> None:
    path = tmp_path / "compile.json"
    path.write_text(
        json.dumps(
            {
                "schema": runner.SCHEMA,
                "mode": "compile_scaling",
                "topology": "dual_rate_fusion",
                "operator_count": 16,
                "warmup_compile": "accepted_zero_work",
                "compile_ns": 10,
                "allocation_count": 2,
                "cumulative_requested_bytes": 30,
                "peak_live_requested_bytes": 20,
                "live_requested_bytes_after_compile": 12,
                "live_requested_bytes_at_probe_end": 12,
                "plan_retained_at_probe_end": True,
                "process_max_rss_kib": 100,
                "compile_disposition": "accepted",
                "plan_digest": "a" * 64,
                "edge_count": 24,
                "rate_port_count": 8,
                "rate_dependency_count": 4,
                "rate_dependency_relations": {
                    "follows_input_count": 2,
                    "inverse_rate_horizon_count": 2,
                },
                "queue_recipe_count": 4,
                "automatic_depth_edge_count": 4,
                "lifecycle_candidate_recipe_count": 1,
                "compile_concrete_provider_count": 1,
                "diagnostic_tokens": [],
            }
        ),
        encoding="utf-8",
    )
    result = runner.checked_result(path)
    assert result["mode"] == "compile_scaling"
    assert result["queue_recipe_count"] == 4
    assert result["plan_retained_at_probe_end"] is True
    assert result["live_requested_bytes_at_probe_end"] == 12


def test_checked_compile_scaling_result_accepts_rejected_cross_process_probe(
    tmp_path: Path,
) -> None:
    path = tmp_path / "compile-rejected.json"
    path.write_text(
        json.dumps(
            {
                "schema": runner.SCHEMA,
                "mode": "compile_scaling",
                "topology": "lifecycle_cross_process_rejected",
                "operator_count": 2,
                "warmup_compile": "accepted_zero_work",
                "compile_ns": 10,
                "allocation_count": 2,
                "cumulative_requested_bytes": 30,
                "peak_live_requested_bytes": 20,
                "live_requested_bytes_after_compile": 12,
                "live_requested_bytes_at_probe_end": 12,
                "plan_retained_at_probe_end": True,
                "process_max_rss_kib": 100,
                "compile_disposition": "rejected",
                "plan_digest": None,
                "edge_count": 2,
                "rate_port_count": 2,
                "rate_dependency_count": 2,
                "rate_dependency_relations": {
                    "follows_input_count": 1,
                    "inverse_rate_horizon_count": 1,
                },
                "queue_recipe_count": 0,
                "automatic_depth_edge_count": 0,
                "lifecycle_candidate_recipe_count": 0,
                "compile_concrete_provider_count": 0,
                "diagnostic_tokens": ["RATE_LIFECYCLE_CROSS_PROCESS_UNSUPPORTED"],
            }
        ),
        encoding="utf-8",
    )
    result = runner.checked_result(path)
    assert result["compile_disposition"] == "rejected"
    assert result["plan_digest"] is None
    assert result["diagnostic_tokens"] == ["RATE_LIFECYCLE_CROSS_PROCESS_UNSUPPORTED"]


def test_checked_headroom_result_accepts_available_probe(tmp_path: Path) -> None:
    path = tmp_path / "headroom.json"
    path.write_text(
        json.dumps(
            {
                "schema": runner.SCHEMA,
                "mode": "headroom",
                "payload_profile": "small",
                "memory_kind": "host",
                "queue_policy": "blocking",
                "headroom_ratio": {"numerator": 2, "denominator": 1},
                "availability": "available",
                "unavailable_reason": None,
                "source_nominal_hz": 1000,
                "source_period_ns": 1_000_000,
                "sink_period_ns": 16666667,
                "warmup_compile": "accepted_zero_work",
                "compile_ns": 10,
                "allocation_count": 2,
                "cumulative_requested_bytes": 30,
                "peak_live_requested_bytes": 20,
                "live_requested_bytes_after_compile": 12,
                "live_requested_bytes_at_probe_end": 12,
                "plan_retained_at_probe_end": True,
                "process_max_rss_kib": 100,
                "compile_disposition": "accepted",
                "plan_digest": "a" * 64,
                "queue_recipe_count": 1,
                "queue_recipe_headroom_ratio": {"numerator": 2, "denominator": 1},
                "startup_to_published_generation_ns": 321,
                "restart_latency_ns": None,
                "restart_latency_unavailable_reason": "not_measured_in_task041_headroom_probe",
                "queue_effective_logical_depth": 34,
                "queue_rate_recommendation": 34,
                "queue_nominal_arrivals": 17,
                "pool_capacity_generation_id": 7,
                "pool_required_entries": 34,
                "pool_logical_payload_capacity_bytes": 272,
                "pool_max_payload_byte_span_per_slot": 8,
                "provider_action": "fresh",
                "provider_decision_id": 8,
                "provider_plan_id": 9,
                "allocation_generation_id": 10,
                "storage_binding_generation_id": 11,
                "provider_materialized_entries": 34,
                "provider_stride_bytes_per_entry": 8,
                "provider_reserved_bytes_total": 272,
                "provider_reserved_bytes_unavailable_reason": None,
                "physical_queue_entries": None,
                "physical_ring_entries": None,
                "physical_queue_geometry_unavailable_reason": (
                    "runtime_did_not_publish_physical_queue_geometry"
                ),
                "unique_pool_summary": {
                    "pool_count": 1,
                    "slot_count": 34,
                    "aggregate_byte_capacity": 272,
                    "memory_kinds": ["host"],
                    "devices": [],
                },
                "traffic_outcomes": {
                    "availability": "available",
                    "unavailable_reason": None,
                    "observation_scope": "finite_realtime_1khz_to_60hz_drain_boundaries",
                    "latency_scope": "last_source_attempt_to_drain_boundary",
                    "throughput_scope": (
                        "last_attempt_of_first_measured_sample_to_final_sample_drain"
                    ),
                    "warmup_samples": 34,
                    "requested_samples": 64,
                    "accepted_publications": 64,
                    "delivered_samples": 64,
                    "sampled_occupancy_high_water": 17,
                    "backpressured_emit_attempts": 0,
                    "resource_exhausted_source_attempts": 0,
                    "not_ready_source_attempts": 0,
                    "consumer_visible_replacements": 0,
                    "measurement_duration_ns": 64_000_000,
                    "finite_pipeline_accepted_publications_per_second": 1000.0,
                    "finite_pipeline_deliveries_per_second": 1000.0,
                    "latency_ns": {
                        "samples": 64,
                        "minimum": 1,
                        "p50": 8,
                        "p95": 16,
                        "p99": 17,
                        "maximum": 17,
                    },
                },
                "inspection_json": '{"schema":"holoscan.rate_queue_inspection.canonical_json.v1"}',
            }
        ),
        encoding="utf-8",
    )
    result = runner.checked_result(path)
    assert result["availability"] == "available"
    assert result["unique_pool_summary"]["slot_count"] == 34
    assert result["startup_to_published_generation_ns"] == 321
    assert result["provider_decision_id"] == 8


def test_checked_headroom_result_accepts_unavailable_semantics(tmp_path: Path) -> None:
    path = tmp_path / "headroom-unavailable.json"
    path.write_text(
        json.dumps(
            {
                "schema": runner.SCHEMA,
                "mode": "headroom",
                "payload_profile": "large",
                "memory_kind": "cuda_device",
                "queue_policy": "drop_oldest",
                "headroom_ratio": {"numerator": 1, "denominator": 1},
                "availability": "unavailable",
                "unavailable_reason": "CUDA device 0 unavailable",
                "source_nominal_hz": 1000,
                "source_period_ns": 1_000_000,
                "sink_period_ns": 16666667,
                "warmup_compile": "accepted_zero_work",
                "compile_ns": 10,
                "allocation_count": 2,
                "cumulative_requested_bytes": 30,
                "peak_live_requested_bytes": 20,
                "live_requested_bytes_after_compile": 12,
                "live_requested_bytes_at_probe_end": 12,
                "plan_retained_at_probe_end": True,
                "process_max_rss_kib": 100,
                "compile_disposition": "accepted",
                "plan_digest": "a" * 64,
                "queue_recipe_count": 1,
                "queue_recipe_headroom_ratio": {"numerator": 1, "denominator": 1},
                "startup_to_published_generation_ns": None,
                "restart_latency_ns": None,
                "restart_latency_unavailable_reason": "not_measured_in_task041_headroom_probe",
                "queue_effective_logical_depth": None,
                "queue_rate_recommendation": None,
                "queue_nominal_arrivals": None,
                "pool_capacity_generation_id": None,
                "pool_required_entries": None,
                "pool_logical_payload_capacity_bytes": None,
                "pool_max_payload_byte_span_per_slot": None,
                "provider_action": None,
                "provider_decision_id": None,
                "provider_plan_id": None,
                "allocation_generation_id": None,
                "storage_binding_generation_id": None,
                "provider_materialized_entries": None,
                "provider_stride_bytes_per_entry": None,
                "provider_reserved_bytes_total": None,
                "provider_reserved_bytes_unavailable_reason": "runtime_generation_unavailable",
                "physical_queue_entries": None,
                "physical_ring_entries": None,
                "physical_queue_geometry_unavailable_reason": "runtime_generation_unavailable",
                "unique_pool_summary": None,
                "traffic_outcomes": {
                    "availability": "unavailable",
                    "unavailable_reason": "CUDA device 0 unavailable",
                    "observation_scope": "finite_realtime_1khz_to_60hz_drain_boundaries",
                    "latency_scope": "last_source_attempt_to_drain_boundary",
                    "throughput_scope": (
                        "last_attempt_of_first_measured_sample_to_final_sample_drain"
                    ),
                    "warmup_samples": 34,
                    "requested_samples": 64,
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
                    "latency_ns": None,
                },
                "inspection_json": None,
            }
        ),
        encoding="utf-8",
    )
    result = runner.checked_result(path)
    assert result["availability"] == "unavailable"
    assert result["queue_effective_logical_depth"] is None
    assert result["provider_materialized_entries"] is None
    assert result["provider_decision_id"] is None


def test_checked_result_rejects_wrong_identity(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="invalid rate-planning result"):
        runner.checked_result(path)


def test_traffic_outcome_validation_preserves_measured_zero_and_unavailable_null() -> None:
    measured = available_traffic_outcomes()
    assert runner.valid_traffic_outcomes(measured)
    assert measured["backpressured_emit_attempts"] == 0

    mismatched = {**measured, "consumer_visible_replacements": 1}
    assert not runner.valid_traffic_outcomes(mismatched)

    unavailable = unavailable_traffic_outcomes()
    assert runner.valid_traffic_outcomes(unavailable)
    invalid_unavailable = {**unavailable, "backpressured_emit_attempts": 0}
    assert not runner.valid_traffic_outcomes(invalid_unavailable)


def test_run_case_removes_stale_result_and_rejects_missing_fresh_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary = tmp_path / "rate_planning_benchmark"
    binary.write_bytes(b"stable-binary")
    expected_sha256 = runner.file_sha256(binary)
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    result_path = case_dir / "result.json"
    result_path.write_text('{"stale":true}\n', encoding="utf-8")

    completed = type("Completed", (), {"returncode": 0, "stdout": "", "stderr": ""})()
    monkeypatch.setattr(runner.evidence.subprocess, "run", lambda *_args, **_kwargs: completed)

    with pytest.raises(RuntimeError, match="did not create a fresh result"):
        runner.run_case([str(binary)], case_dir, 1.0, expected_sha256)
    assert not result_path.exists()


def test_binary_hash_guard_rejects_relinked_bytes(tmp_path: Path) -> None:
    binary = tmp_path / "rate_planning_benchmark"
    binary.write_bytes(b"before")
    expected_sha256 = runner.file_sha256(binary)
    binary.write_bytes(b"after")
    with pytest.raises(RuntimeError, match="binary changed before test"):
        runner.verify_binary_sha256(binary, expected_sha256, "before test")


def test_run_case_rejects_binary_drift_after_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary = tmp_path / "rate_planning_benchmark"
    binary.write_bytes(b"before")
    expected_sha256 = runner.file_sha256(binary)

    def drift_binary(*_args: object, **_kwargs: object) -> object:
        binary.write_bytes(b"after")
        return type("Completed", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(runner.evidence.subprocess, "run", drift_binary)
    with pytest.raises(RuntimeError, match="binary changed after child completion"):
        runner.run_case([str(binary)], tmp_path / "case", 1.0, expected_sha256)


def test_build_metadata_bounds_compiler_version_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    build_dir = tmp_path / "build"
    binary = build_dir / "bin" / "rate_planning_benchmark"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"binary")
    compiler = tmp_path / "compiler-wrapper"
    (build_dir / "CMakeCache.txt").write_text(
        f"CMAKE_CXX_COMPILER:FILEPATH={compiler}\n", encoding="utf-8"
    )
    observed_timeout: float | None = None

    def time_out(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal observed_timeout
        observed_timeout = float(kwargs["timeout"])
        raise subprocess.TimeoutExpired(command, observed_timeout)

    monkeypatch.setattr(runner.evidence.subprocess, "run", time_out)
    metadata = runner.build_metadata(binary)

    assert observed_timeout == runner.evidence.DEFAULT_COMPILER_VERSION_TIMEOUT_SEC
    assert metadata["compiler_version_output"] is None


def test_aggregate_unique_pool_summary_deduplicates_pool_ids() -> None:
    summary = runner.aggregate_unique_pool_summary(
        [
            {
                "pool_id": "tensor.output",
                "slot_count": 17,
                "aggregate_byte_capacity": 136,
                "memory_kind": "host",
                "device": None,
            },
            {
                "pool_id": "tensor.output",
                "slot_count": 17,
                "aggregate_byte_capacity": 136,
                "memory_kind": "host",
                "device": None,
            },
            {
                "pool_id": "tensor.shadow",
                "slot_count": 4,
                "aggregate_byte_capacity": 32,
                "memory_kind": "pinned_host",
                "device": None,
            },
        ]
    )
    assert summary == {
        "pool_count": 2,
        "slot_count": 21,
        "aggregate_byte_capacity": 168,
        "memory_kinds": ["host", "pinned_host"],
        "devices": [],
    }


def test_summarize_compile_case_preserves_rejected_schema_and_probe_end_semantics() -> None:
    summary = runner.summarize_compile_case(
        "lifecycle_cross_process_rejected",
        2,
        [
            {
                "compile_ns": 10,
                "allocation_count": 2,
                "cumulative_requested_bytes": 30,
                "peak_live_requested_bytes": 20,
                "live_requested_bytes_after_compile": 12,
                "live_requested_bytes_at_probe_end": 12,
                "process_max_rss_kib": 100,
                "plan_digest": None,
                "plan_retained_at_probe_end": True,
                "compile_disposition": "rejected",
                "edge_count": 2,
                "rate_port_count": 2,
                "rate_dependency_count": 2,
                "rate_dependency_relations": {
                    "follows_input_count": 1,
                    "inverse_rate_horizon_count": 1,
                },
                "queue_recipe_count": 0,
                "automatic_depth_edge_count": 0,
                "lifecycle_candidate_recipe_count": 0,
                "compile_concrete_provider_count": 0,
                "diagnostic_tokens": ["RATE_LIFECYCLE_CROSS_PROCESS_UNSUPPORTED"],
            }
        ],
    )
    assert summary["compile_disposition"] == "rejected"
    assert summary["plan_digest"] is None
    assert summary["plan_retained_at_probe_end"] is True
    assert summary["live_requested_bytes_at_probe_end"]["minimum"] == 12
    assert summary["diagnostic_tokens"] == ["RATE_LIFECYCLE_CROSS_PROCESS_UNSUPPORTED"]


def test_summarize_headroom_pair_tracks_identity_except_headroom() -> None:
    one_one = {
        "mode": "headroom",
        "payload_profile": "small",
        "memory_kind": "host",
        "queue_policy": "blocking",
        "headroom_ratio": {"numerator": 1, "denominator": 1},
        "queue_recipe_headroom_ratio": {"numerator": 1, "denominator": 1},
        "availability": "available",
        "startup_to_published_generation_ns": 210,
        "restart_latency_ns": None,
        "restart_latency_unavailable_reason": "not_measured_in_task041_headroom_probe",
        "queue_effective_logical_depth": 17,
        "queue_rate_recommendation": 17,
        "pool_capacity_generation_id": 7,
        "pool_required_entries": 17,
        "pool_logical_payload_capacity_bytes": 136,
        "pool_max_payload_byte_span_per_slot": 8,
        "provider_action": "fresh",
        "provider_decision_id": 8,
        "provider_plan_id": 9,
        "allocation_generation_id": 10,
        "storage_binding_generation_id": 11,
        "provider_materialized_entries": 17,
        "provider_stride_bytes_per_entry": 8,
        "provider_reserved_bytes_total": 136,
        "plan_digest": "a" * 64,
        "source_nominal_hz": 1000,
        "source_period_ns": 1_000_000,
        "sink_period_ns": 16666667,
        "unique_pool_summary": {
            "pool_count": 1,
            "slot_count": 17,
            "aggregate_byte_capacity": 136,
            "memory_kinds": ["host"],
            "devices": [],
        },
        "traffic_outcomes": available_traffic_outcomes(),
    }
    two_one = {
        **one_one,
        "headroom_ratio": {"numerator": 2, "denominator": 1},
        "queue_recipe_headroom_ratio": {"numerator": 2, "denominator": 1},
        "startup_to_published_generation_ns": 420,
        "queue_effective_logical_depth": 34,
        "queue_rate_recommendation": 34,
        "pool_required_entries": 34,
        "pool_logical_payload_capacity_bytes": 272,
        "provider_materialized_entries": 34,
        "provider_reserved_bytes_total": 272,
        "unique_pool_summary": {
            "pool_count": 1,
            "slot_count": 34,
            "aggregate_byte_capacity": 272,
            "memory_kinds": ["host"],
            "devices": [],
        },
    }
    one_one_repeat = {**one_one, "startup_to_published_generation_ns": 230}
    two_one_repeat = {**two_one, "startup_to_published_generation_ns": 460}
    summary = runner.summarize_headroom_pair(
        "small", "host", "blocking", [one_one, one_one_repeat], [two_one, two_one_repeat]
    )
    assert summary["paired_identity_match"] is True
    assert summary["status"] == "characterization_only"
    assert summary["availability"] == "available"
    assert summary["queue_effective_logical_depth"] == {"1/1": 17, "2/1": 34}
    assert summary["provider_materialized_entries"] == {"1/1": 17, "2/1": 34}
    assert summary["startup_to_published_generation_ns"]["1/1"] == {
        "minimum": 210,
        "median": 220.0,
        "p95": 230,
        "p99": 230,
        "maximum": 230,
    }
    assert summary["startup_to_published_generation_ns"]["2/1"] == {
        "minimum": 420,
        "median": 440.0,
        "p95": 460,
        "p99": 460,
        "maximum": 460,
    }
    assert summary["restart_latency_ns"] == {"1/1": None, "2/1": None}
    assert summary["provider_decision_id"] == {"1/1": 8, "2/1": 8}
    assert summary["traffic_outcomes"]["1/1"]["sampled_occupancy_high_water"]["median"] == 17
    assert summary["traffic_outcomes"]["1/1"]["accepted_publications"]["median"] == 64
    assert summary["traffic_outcomes"]["1/1"]["delivered_samples"]["median"] == 64


def test_summarize_headroom_pair_preserves_unavailable_instead_of_zero() -> None:
    unavailable = {
        "mode": "headroom",
        "payload_profile": "large",
        "memory_kind": "cuda_device",
        "queue_policy": "drop_oldest",
        "headroom_ratio": {"numerator": 2, "denominator": 1},
        "queue_recipe_headroom_ratio": {"numerator": 2, "denominator": 1},
        "availability": "unavailable",
        "unavailable_reason": "CUDA device 0 unavailable",
        "source_nominal_hz": 1000,
        "source_period_ns": 1_000_000,
        "sink_period_ns": 16666667,
        "startup_to_published_generation_ns": None,
        "restart_latency_ns": None,
        "restart_latency_unavailable_reason": "not_measured_in_task041_headroom_probe",
        "queue_effective_logical_depth": None,
        "queue_rate_recommendation": None,
        "pool_capacity_generation_id": None,
        "pool_required_entries": None,
        "pool_logical_payload_capacity_bytes": None,
        "pool_max_payload_byte_span_per_slot": None,
        "provider_action": None,
        "provider_decision_id": None,
        "provider_plan_id": None,
        "allocation_generation_id": None,
        "storage_binding_generation_id": None,
        "provider_materialized_entries": None,
        "provider_stride_bytes_per_entry": None,
        "provider_reserved_bytes_total": None,
        "plan_digest": "b" * 64,
        "unique_pool_summary": None,
        "traffic_outcomes": unavailable_traffic_outcomes(),
    }
    one_one_unavailable = {
        **unavailable,
        "headroom_ratio": {"numerator": 1, "denominator": 1},
        "queue_recipe_headroom_ratio": {"numerator": 1, "denominator": 1},
    }
    summary = runner.summarize_headroom_pair(
        "large", "cuda_device", "drop_oldest", [one_one_unavailable], [unavailable]
    )
    assert summary["availability"] == "unavailable"
    assert summary["queue_effective_logical_depth"] == {"1/1": None, "2/1": None}
    assert summary["provider_materialized_entries"] == {"1/1": None, "2/1": None}
    assert summary["unique_pool_summary"] == {"1/1": None, "2/1": None}
    assert summary["provider_decision_id"] == {"1/1": None, "2/1": None}
    assert summary["traffic_outcomes"]["1/1"]["sampled_occupancy_high_water"] is None


def test_summarize_headroom_pair_rejects_wrong_ratio_and_repetition_drift() -> None:
    row = {
        "mode": "headroom",
        "payload_profile": "small",
        "memory_kind": "host",
        "queue_policy": "blocking",
        "headroom_ratio": {"numerator": 2, "denominator": 1},
        "queue_recipe_headroom_ratio": {"numerator": 2, "denominator": 1},
        "availability": "available",
        "unavailable_reason": None,
        "source_nominal_hz": 1000,
        "source_period_ns": 1_000_000,
        "sink_period_ns": 16_666_667,
        "queue_nominal_arrivals": 17,
        "inspection_json": "canonical-a",
        "traffic_outcomes": available_traffic_outcomes(),
    }
    with pytest.raises(RuntimeError, match="1/1 repetitions disagree"):
        runner.summarize_headroom_pair("small", "host", "blocking", [row], [row])

    one_one = {
        **row,
        "headroom_ratio": {"numerator": 1, "denominator": 1},
        "queue_recipe_headroom_ratio": {"numerator": 1, "denominator": 1},
    }
    availability_drift = {**one_one, "availability": "unavailable"}
    with pytest.raises(RuntimeError, match="1/1 repetitions disagree"):
        runner.summarize_headroom_pair(
            "small", "host", "blocking", [one_one, availability_drift], [row]
        )

    static_drift = {**one_one, "queue_nominal_arrivals": 18}
    with pytest.raises(RuntimeError, match="1/1 repetitions disagree"):
        runner.summarize_headroom_pair("small", "host", "blocking", [one_one, static_drift], [row])

    inspection_drift = {**one_one, "inspection_json": "canonical-b"}
    with pytest.raises(RuntimeError, match="1/1 repetitions disagree"):
        runner.summarize_headroom_pair(
            "small", "host", "blocking", [one_one, inspection_drift], [row]
        )


def test_validate_args_rejects_invalid_source_snapshot_digest() -> None:
    with pytest.raises(ValueError, match="source-snapshot-sha256"):
        runner.validate_args(
            type(
                "Args",
                (),
                {
                    "mode": ["compile_scaling"],
                    "topology": ["linear_fixed"],
                    "operators": [16],
                    "payload": ["small"],
                    "memory": ["host"],
                    "headroom": ["1/1"],
                    "repetitions": 1,
                    "process_timeout_sec": 1.0,
                    "traffic_warmup_samples": 34,
                    "traffic_samples": 64,
                    "traffic_timeout_sec": 1.0,
                    "source_snapshot_sha256": "not-a-digest",
                },
            )()
        )
