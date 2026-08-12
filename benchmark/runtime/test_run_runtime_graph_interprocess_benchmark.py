# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for distributed Runtime Graph benchmark orchestration."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

MODULE_PATH = Path(__file__).with_name("run_runtime_graph_interprocess_benchmark.py")
SPEC = importlib.util.spec_from_file_location("runtime_graph_interprocess_runner", MODULE_PATH)
assert SPEC is not None
assert SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def make_args(**overrides: Any) -> argparse.Namespace:
    values = {
        "topology": "fanout",
        "workload": "latency",
        "payload": "pod",
        "tensor_storage": "pageable_host",
        "tensor_bytes": 1 << 20,
        "tensor_device": 0,
        "bulk_work_profile": "full_write",
        "samples": 8,
        "warmup": 2,
        "fanout": 1,
        "queue_depth": 1,
        "publisher_period_us": 1000,
        "payload_validation": "final",
        "producer_cpu": -1,
        "relay_cpu": -1,
        "consumer_cpu": -1,
        "consumer_cpus": None,
        "producer_controller_cpu": -1,
        "relay_controller_cpu": -1,
        "relay_pump_cpu": -1,
        "consumer_controller_cpu": -1,
        "consumer_controller_cpus": None,
        "consumer_pump_cpu": -1,
        "consumer_pump_cpus": None,
        "busy_poll": False,
        "runtime_phase_trace": False,
        "session_timeout_ms": 10_000,
        "process_timeout_sec": 30.0,
        "repetitions": 1,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def validated_args(**overrides: Any) -> argparse.Namespace:
    args = make_args(**overrides)
    runner.validate_args(args)
    return args


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"samples": 0}, "--samples must be positive"),
        ({"warmup": -1}, "--warmup must be nonnegative"),
        ({"queue_depth": 0}, "--queue-depth must be positive"),
        (
            {"workload": "latency", "queue_depth": 3},
            "latency workload requires --queue-depth 1 or 2",
        ),
        (
            {"workload": "latency", "publisher_period_us": 0},
            "latency workload requires",
        ),
        (
            {"workload": "saturated", "publisher_period_us": 1},
            "saturated workload requires",
        ),
        (
            {"topology": "chain", "fanout": 2},
            "chain topology requires --fanout 1",
        ),
        (
            {"payload": "tensor", "fanout": 2},
            "Tensor-backed payloads currently require",
        ),
        (
            {"payload": "tensor", "topology": "chain"},
            "Tensor-backed payloads currently require",
        ),
        (
            {"payload": "image", "fanout": 2},
            "Tensor-backed payloads currently require",
        ),
        (
            {"payload": "schema", "fanout": 2},
            "schema payload currently requires",
        ),
        (
            {"payload": "schema", "topology": "chain"},
            "schema payload currently requires",
        ),
        (
            {"payload": "tensor", "tensor_bytes": 0},
            "bounded Tensor range",
        ),
        (
            {"payload": "tensor", "tensor_device": -1},
            "--tensor-device must be nonnegative",
        ),
        (
            {"tensor_storage": "device"},
            "Tensor storage options require a Tensor-backed payload",
        ),
        (
            {"bulk_work_profile": "descriptor_only"},
            "Tensor storage options require a Tensor-backed payload",
        ),
        (
            {"payload": "multi_tensor_loan", "tensor_storage": "device"},
            "multi-Tensor benchmark payloads currently require pageable_host",
        ),
        (
            {"payload": "multi_tensor_staging", "warmup": 0},
            "multi-Tensor benchmark payloads require warmup",
        ),
        (
            {"relay_cpu": 4},
            "relay CPU placement requires --topology chain",
        ),
        (
            {"producer_cpu": 4, "consumer_cpu": 4},
            "all requested lane and support-thread CPUs must be distinct",
        ),
        (
            {"consumer_cpu": 4, "consumer_pump_cpu": 4},
            "all requested lane and support-thread CPUs must be distinct",
        ),
        (
            {"fanout": 2, "consumer_cpu": 4},
            "--consumer-cpus is required to pin fanout consumers",
        ),
        (
            {"fanout": 2, "consumer_cpus": [4]},
            "--consumer-cpus must provide exactly one CPU per consumer",
        ),
        ({"consumer_controller_cpu": -2}, "CPU values must be -1 or nonnegative"),
        (
            {"runtime_phase_trace": True, "samples": 100_000, "warmup": 1},
            "runtime phase trace exceeds",
        ),
        (
            {
                "samples": 10_000,
                "warmup": 1_000,
                "publisher_period_us": 1_000,
                "session_timeout_ms": 20_000,
            },
            "twice the nominal paced duration",
        ),
        (
            {"session_timeout_ms": 10_000, "process_timeout_sec": 29.0},
            "three worker session-timeout intervals",
        ),
        ({"repetitions": 0}, "--repetitions must be positive"),
    ],
)
def test_validate_args_rejects_invalid_geometry(overrides: dict[str, Any], message: str) -> None:
    # Key setup: change one bounded orchestration input at a time.
    args = make_args(**overrides)

    # Key assertion: invalid measurement geometry fails before any worker is launched.
    with pytest.raises(ValueError, match=message):
        runner.validate_args(args)


def test_default_invocation_has_self_consistent_time_budgets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Key setup: parse the bare documented invocation without test-specific timeout overrides.
    monkeypatch.setattr(sys, "argv", ["runner", "runtime_graph_interprocess_benchmark"])
    args = runner.parse_args()

    # Key assertion: the default paced schedule fits both worker and coordinator budgets.
    runner.validate_args(args)
    assert args.samples == 10_000
    assert args.session_timeout_ms == 30_000
    assert args.process_timeout_sec == 90.0


def test_main_skips_before_launching_workers_on_non_linux(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Key setup: present a valid bare invocation on a POSIX platform without Linux HoloMQ support.
    monkeypatch.setattr(sys, "argv", ["runner", "runtime_graph_interprocess_benchmark"])
    monkeypatch.setattr(runner.sys, "platform", "freebsd")

    # Key assertion: the coordinator returns CTest's skip code before probing the binary or pipes.
    assert runner.main() == 77


def valid_records() -> tuple[dict[str, Any], dict[str, Any]]:
    common = {
        "schema": "holoscan.ea1.runtime-interprocess-benchmark/v1",
        "benchmark": "runtime_graph_interprocess_pod_1to1",
        "workload": "latency",
        "payload": "pod",
        "run_id": "run",
        "graph_hash": "graph",
        "plan_digest": "plan",
        "service_id": "service",
        "service_ids": ["service"],
        "termination": "stopped",
        "diagnostic_count": 0,
        "clock": {
            "source": "clock_gettime(CLOCK_MONOTONIC)",
            "steady": True,
            "scope": "same_host_cross_process",
        },
        "timestamp_instrumentation": {
            "calibration": "post_measurement_back_to_back_pair",
            "clock_reads_per_sample": 2,
            "pair_overhead_ns": {"p50": 20},
        },
        "geometry": {
            "samples": 8,
            "warmup": 2,
            "queue_depth": 1,
            "notification_capacity": 1,
            "publication_backpressure_capacity": 1,
            "pool_slots": None,
            "release_capacity": None,
            "payload_bytes": 64,
            "application_payload_bytes": 64,
            "inline_transport_bytes": 64,
            "external_buffer_bytes": 0,
            "payload_write_profile": "value_fill",
            "bulk_work_profile": "not_applicable",
            "topology": "fanout",
            "fanout": 1,
            "processes": 2,
            "transport_hops": 1,
            "publisher_period_us": 1000,
        },
        "placement": {"lane_wake_mode": "sleeping_waitset"},
        "generated_schema": {},
        "tensor": {},
        "payload_validation": "final",
        "validation_timing": "after_timed_operations",
    }
    producer = {
        **common,
        "wake": {
            "event_wake": False,
            "busy_poll": False,
            "receive_polling": False,
        },
        "role": "producer",
        "consumer_index": None,
        "process_id": 100,
        "process_group": 1,
        "counts": {
            "source_starts": 1,
            "consumer_starts": 0,
            "publications": 10,
            "deliveries": 0,
        },
        "timing": {},
        "support_threads": [
            {
                "role": "controller",
                "requested_cpu": None,
                "entered": True,
                "affinity_supported": True,
                "native_tid": 1001,
                "observed_affinity": [0, 1],
                "error": None,
            },
            {
                "role": "channel_pump",
                "requested_cpu": None,
                "entered": False,
                "affinity_supported": False,
                "native_tid": None,
                "observed_affinity": [],
                "error": None,
            },
        ],
        "ingress_wake_trace": {"enabled": False},
    }
    consumer = {
        **common,
        "wake": {
            "event_wake": True,
            "busy_poll": False,
            "receive_polling": False,
        },
        "role": "consumer",
        "consumer_index": 1,
        "process_id": 101,
        "process_group": 2,
        "counts": {
            "source_starts": 0,
            "consumer_starts": 1,
            "publications": 0,
            "deliveries": 10,
            "final_sequence": 10,
        },
        "timing": {
            "measured_start_ns": 100,
            "measured_end_ns": 200,
            "messages_per_second": 1.0,
        },
        "support_threads": [
            {
                "role": "controller",
                "requested_cpu": None,
                "entered": True,
                "affinity_supported": True,
                "native_tid": 1002,
                "observed_affinity": [0, 1],
                "error": None,
            },
            {
                "role": "channel_pump",
                "requested_cpu": None,
                "entered": True,
                "affinity_supported": True,
                "native_tid": 1003,
                "observed_affinity": [0, 1],
                "error": None,
            },
        ],
        "ingress_wake_trace": {"enabled": False},
    }
    return producer, consumer


def ingress_latency_bucket() -> dict[str, int | None]:
    return {"pMin": 1, "p50": 2, "p99": 3, "p999": None, "pMax": 4}


def empty_ingress_latency_bucket() -> dict[str, None]:
    return {"pMin": None, "p50": None, "p99": None, "p999": None, "pMax": None}


def complete_ingress_wake_trace(*, readiness_poll_elided: bool) -> dict[str, Any]:
    readiness_bucket = (
        empty_ingress_latency_bucket() if readiness_poll_elided else ingress_latency_bucket()
    )
    return {
        "enabled": True,
        "valid": True,
        "readiness_poll_elided": readiness_poll_elided,
        "dropped": 0,
        "samples": 8,
        "transport_wait_samples": 8,
        "pump_wake_samples": 3,
        "parked_wake_samples": 3,
        "active_controller_samples": 5,
        "transport_wait_to_ingress_ns": ingress_latency_bucket(),
        "ingress_to_pump_wake_ns": ingress_latency_bucket(),
        "pump_wake_to_readiness_poll_ns": readiness_bucket,
        "pump_wake_to_controller_return_ns": ingress_latency_bucket(),
        "controller_return_to_readiness_poll_ns": readiness_bucket,
        "readiness_poll_to_compute_dispatch_ns": readiness_bucket,
        "compute_dispatch_to_callback_entry_ns": ingress_latency_bucket(),
        "transport_wait_to_callback_entry_ns": ingress_latency_bucket(),
    }


def test_validate_results_proves_independent_worker_roles() -> None:
    # Key setup: provide matching plan/service identities with distinct process groups.
    producer, consumer = valid_records()

    # Key assertion: exact publication/delivery counts and one-sided lifecycle starts are accepted.
    runner.validate_results(validated_args(), producer, None, [consumer])


def test_validate_results_accepts_tensor_plan_and_metadata_evidence() -> None:
    # Key setup: both isolated workers report one exportable 1 MiB pinned-host Tensor pool.
    producer, consumer = valid_records()
    for record in (producer, consumer):
        record["payload"] = "tensor"
        record["benchmark"] = "runtime_graph_interprocess_tensor_1to1_pinned_host_full_write"
        record["geometry"].update(
            payload_bytes=1 << 20,
            application_payload_bytes=1 << 20,
            inline_transport_bytes=464,
            external_buffer_bytes=1 << 20,
            payload_write_profile="full_buffer_write_per_sample",
            bulk_work_profile="full_write",
            pool_slots=6,
            release_capacity=6,
        )
        record["tensor"] = {
            "storage": "pinned_host",
            "device": 0,
            "descriptor_schema": False,
            "read_only_import_checked": True,
            "capture_timestamp_in_sample_metadata": True,
            "direct_allocation_records": 1,
            "memory_kind_plan_match": True,
            "service_pool_bound": True,
        }

    # Key assertion: plan-bound storage and receive-side metadata evidence qualify the Tensor row.
    args = validated_args(payload="tensor", tensor_storage="pinned_host")
    assert runner.expected_benchmark_name(args) == (
        "runtime_graph_interprocess_tensor_1to1_pinned_host_full_write"
    )
    runner.validate_results(args, producer, None, [consumer])


def test_validate_results_accepts_generated_time_schema_evidence() -> None:
    # Key setup: both workers report the fixed-size owning Time schema and one wire-capacity bound.
    producer, consumer = valid_records()
    for role, record in (("producer", producer), ("consumer", consumer)):
        record["payload"] = "schema"
        record["benchmark"] = "runtime_graph_interprocess_schema_time_1to1"
        record["geometry"].update(
            payload_bytes=16,
            application_payload_bytes=16,
            inline_transport_bytes=None,
            payload_write_profile="full_payload_per_sample",
        )
        record["generated_schema"] = {
            "native_type": "schema::TimeT",
            "wire_capacity": 40,
        }
        record["allocation_probe"] = {
            "sequence": 2,
            "excluded_from_measurement": True,
            "sink_access": "receive",
            "source_compute_allocation_limit": 0,
            "sink_compute_allocation_limit": 0,
            "source_compute_allocations": 0 if role == "producer" else None,
            "sink_compute_allocations": 0 if role == "consumer" else None,
        }

    # Key assertion: matching evidence is accepted, while one owning-copy allocation is rejected.
    args = validated_args(payload="schema")
    assert runner.expected_benchmark_name(args) == "runtime_graph_interprocess_schema_time_1to1"
    runner.validate_results(args, producer, None, [consumer])
    consumer["allocation_probe"]["sink_compute_allocations"] = 1
    with pytest.raises(RuntimeError, match="exceeded its schema callback allocation limit"):
        runner.validate_results(args, producer, None, [consumer])


def test_validate_results_accepts_device_image_schema_evidence() -> None:
    # Key setup: both workers report Image metadata plus one 1 MiB device Tensor sidecar.
    producer, consumer = valid_records()
    for role, record in (("producer", producer), ("consumer", consumer)):
        record["payload"] = "image"
        record["benchmark"] = "runtime_graph_interprocess_image_1to1_device_full_write"
        record["geometry"].update(
            payload_bytes=1 << 20,
            application_payload_bytes=1 << 20,
            inline_transport_bytes=856,
            external_buffer_bytes=1 << 20,
            payload_write_profile="full_buffer_write_per_sample",
            bulk_work_profile="full_write",
            pool_slots=6,
            release_capacity=6,
        )
        record["generated_schema"] = {
            "native_type": "schema::ImageT",
            "wire_capacity": 856,
        }
        record["tensor"] = {
            "storage": "device",
            "device": 0,
            "descriptor_schema": True,
            "read_only_import_checked": True,
            "capture_timestamp_in_sample_metadata": True,
            "direct_allocation_records": 1,
            "memory_kind_plan_match": True,
            "service_pool_bound": True,
        }
        record["allocation_probe"] = {
            "sequence": 2,
            "excluded_from_measurement": True,
            "sink_access": "receive",
            "source_compute_allocation_limit": 0,
            "sink_compute_allocation_limit": 6,
            "source_compute_allocations": 0 if role == "producer" else None,
            "sink_compute_allocations": 6 if role == "consumer" else None,
        }

    # Key assertion: the runner accepts Image only when its final device provider is retained.
    args = validated_args(payload="image", tensor_storage="device")
    assert runner.expected_benchmark_name(args) == (
        "runtime_graph_interprocess_image_1to1_device_full_write"
    )
    runner.validate_results(args, producer, None, [consumer])


@pytest.mark.parametrize("payload", sorted(runner.MULTI_TENSOR_PAYLOADS))
def test_validate_results_accepts_allocation_free_multi_tensor_paths(payload: str) -> None:
    # Key setup: both workers report two 1 MiB fields and one excluded zero-allocation probe.
    producer, consumer = valid_records()
    for role, record in (("producer", producer), ("consumer", consumer)):
        record["payload"] = payload
        record["benchmark"] = f"runtime_graph_interprocess_{payload}"
        record["geometry"].update(
            payload_bytes=2 << 20,
            application_payload_bytes=2 << 20,
            inline_transport_bytes=None,
            external_buffer_bytes=2 << 20,
            payload_write_profile="full_buffer_write_per_sample",
            bulk_work_profile="full_write",
            pool_slots=6,
            release_capacity=6,
        )
        record["generated_schema"] = {
            "native_type": "runtime_benchmark::MultiTensorMessageT",
            "wire_capacity": 1536,
        }
        record["tensor"] = {
            "storage": "pageable_host",
            "device": 0,
            "descriptor_schema": True,
            "read_only_import_checked": True,
            "capture_timestamp_in_sample_metadata": True,
            "direct_allocation_records": 2,
            "memory_kind_plan_match": True,
            "service_pool_bound": True,
        }
        record["allocation_probe"] = {
            "sequence": 2,
            "excluded_from_measurement": True,
            "sink_access": "selected_batch",
            "source_compute_allocation_limit": 0,
            "sink_compute_allocation_limit": 0,
            "source_compute_allocations": 0 if role == "producer" else None,
            "sink_compute_allocations": 0 if role == "consumer" else None,
        }

    # Key assertion: loan and staged-copy rows both prove allocation-free steady-state callbacks.
    args = validated_args(payload=payload)
    runner.validate_results(args, producer, None, [consumer])


def descriptor_only_tensor_records() -> tuple[dict[str, Any], dict[str, Any]]:
    producer, consumer = valid_records()
    for record in (producer, consumer):
        record["payload"] = "tensor"
        record["benchmark"] = "runtime_graph_interprocess_tensor_1to1_device_descriptor_only"
        record["geometry"].update(
            payload_bytes=1 << 20,
            application_payload_bytes=1 << 20,
            inline_transport_bytes=464,
            external_buffer_bytes=1 << 20,
            payload_write_profile="preinitialized_immutable_backing",
            bulk_work_profile="descriptor_only",
            pool_slots=6,
            release_capacity=6,
        )
        record["tensor"] = {
            "storage": "device",
            "device": 0,
            "descriptor_schema": False,
            "read_only_import_checked": True,
            "capture_timestamp_in_sample_metadata": True,
            "direct_allocation_records": 1,
            "memory_kind_plan_match": True,
            "service_pool_bound": True,
        }
    producer["counts"]["descriptor_backing_initializations"] = 1
    producer["counts"]["descriptor_backing_initializations_during_measurement"] = 0
    return producer, consumer


def test_validate_results_accepts_preinitialized_descriptor_only_tensor() -> None:
    # Key setup: both workers report immutable backing and the producer initialized it pre-measure.
    producer, consumer = descriptor_only_tensor_records()

    # Key assertion: the transport-only row is accepted only with pre-measure initialization.
    args = validated_args(
        payload="tensor",
        tensor_storage="device",
        bulk_work_profile="descriptor_only",
    )
    assert runner.expected_benchmark_name(args) == (
        "runtime_graph_interprocess_tensor_1to1_device_descriptor_only"
    )
    runner.validate_results(args, producer, None, [consumer])


@pytest.mark.parametrize(
    ("initializations", "measured_initializations", "error_type", "message"),
    [
        (
            0,
            0,
            RuntimeError,
            r"reported no backing initialization .*total=0.*during_measurement=0",
        ),
        (
            2,
            1,
            runner.DescriptorWarmupIncompleteError,
            r"initialized backing during measurement .*total=2.*during_measurement=1",
        ),
    ],
)
def test_validate_results_diagnoses_descriptor_only_initialization_phase(
    initializations: int,
    measured_initializations: int,
    error_type: type[RuntimeError],
    message: str,
) -> None:
    # Key setup: preserve valid Tensor evidence while corrupting one initialization-phase invariant.
    producer, consumer = descriptor_only_tensor_records()
    producer["counts"]["descriptor_backing_initializations"] = initializations
    producer["counts"]["descriptor_backing_initializations_during_measurement"] = (
        measured_initializations
    )

    # Key assertion: the diagnostic distinguishes missing initialization from late initialization.
    args = validated_args(
        payload="tensor",
        tensor_storage="device",
        bulk_work_profile="descriptor_only",
    )
    with pytest.raises(error_type, match=message):
        runner.validate_results(args, producer, None, [consumer])


def test_run_qualified_repetition_retries_incomplete_descriptor_warmup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Key setup: make the first otherwise valid attempt discover backing during measurement.
    records = ({"role": "producer"}, None, [{"role": "consumer"}])
    attempts: list[int] = []

    def run_repetition(
        _args: argparse.Namespace, _repetition: int, _output_dir: Path, attempt: int
    ) -> tuple[dict[str, Any], dict[str, Any] | None, list[dict[str, Any]]]:
        attempts.append(attempt)
        if attempt == 0:
            raise runner.DescriptorWarmupIncompleteError("late backing")
        return records

    monkeypatch.setattr(runner, "run_repetition", run_repetition)

    # Key assertion: the invalid measurement is discarded and a fresh attempt supplies the row.
    assert runner.run_qualified_repetition(make_args(), 2, tmp_path) == records
    assert attempts == [0, 1]
    assert "discarding benchmark repetition 2" in capsys.readouterr().err


def test_run_qualified_repetition_bounds_descriptor_warmup_retries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Key setup: make every fresh worker attempt discover backing during measurement.
    attempts: list[int] = []

    def run_repetition(
        _args: argparse.Namespace, _repetition: int, _output_dir: Path, attempt: int
    ) -> tuple[dict[str, Any], dict[str, Any] | None, list[dict[str, Any]]]:
        attempts.append(attempt)
        raise runner.DescriptorWarmupIncompleteError("late backing")

    monkeypatch.setattr(runner, "run_repetition", run_repetition)

    # Key assertion: qualification remains fail-closed after the bounded retry budget.
    with pytest.raises(runner.DescriptorWarmupIncompleteError, match="late backing"):
        runner.run_qualified_repetition(make_args(), 0, tmp_path)
    assert attempts == list(range(runner.MAXIMUM_DESCRIPTOR_WARMUP_ATTEMPTS))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("process_id", 100, "distinct processes"),
        ("process_group", 1, "duplicate process groups"),
        ("service_ids", ["other"], "disagree on service_ids"),
        ("plan_digest", "other", "disagree on plan_digest"),
        ("clock", {"source": "other"}, "same-host steady clock domain"),
    ],
)
def test_validate_results_rejects_mismatched_worker_evidence(
    field: str, value: Any, message: str
) -> None:
    # Key setup: corrupt one consumer-side isolation or identity field.
    producer, consumer = valid_records()
    consumer[field] = value

    # Key assertion: orchestration never summarizes a mismatched worker pair.
    with pytest.raises(RuntimeError, match=message):
        runner.validate_results(validated_args(), producer, None, [consumer])


def test_validate_results_rejects_unapplied_support_thread_cpu() -> None:
    # Key setup: request a dedicated consumer-pump CPU but report another affinity.
    producer, consumer = valid_records()
    consumer["support_threads"][1]["requested_cpu"] = 6
    consumer["support_threads"][1]["observed_affinity"] = [7]

    # Key assertion: worker output cannot qualify a run with unverified support-thread placement.
    with pytest.raises(RuntimeError, match="did not apply its channel_pump CPU"):
        runner.validate_results(validated_args(consumer_pump_cpu=6), producer, None, [consumer])


def test_validate_results_requires_complete_ingress_wake_trace() -> None:
    # Key setup: request phase attribution but report one missing measured sample.
    producer, consumer = valid_records()
    consumer["ingress_wake_trace"] = {
        "enabled": True,
        "valid": True,
        "readiness_poll_elided": True,
        "dropped": 0,
        "samples": 7,
    }

    # Key assertion: partial phase evidence cannot qualify a controlled diagnostic run.
    with pytest.raises(RuntimeError, match="incomplete ingress-wake trace evidence"):
        runner.validate_results(
            validated_args(runtime_phase_trace=True), producer, None, [consumer]
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("transport_wait_samples", 7),
        ("pump_wake_samples", 9),
    ],
)
def test_validate_results_rejects_incomplete_ingress_phase_counts(field: str, value: int) -> None:
    producer, consumer = valid_records()
    consumer["ingress_wake_trace"] = {
        "enabled": True,
        "valid": True,
        "readiness_poll_elided": True,
        "dropped": 0,
        "samples": 8,
        "transport_wait_samples": 8,
        "pump_wake_samples": 8,
        "parked_wake_samples": 8,
        "active_controller_samples": 0,
    }
    consumer["ingress_wake_trace"][field] = value

    with pytest.raises(RuntimeError, match="incomplete ingress-wake trace evidence"):
        runner.validate_results(
            validated_args(runtime_phase_trace=True), producer, None, [consumer]
        )


def test_validate_results_accepts_complete_ingress_phase_counts() -> None:
    producer, consumer = valid_records()
    consumer["ingress_wake_trace"] = complete_ingress_wake_trace(readiness_poll_elided=True)

    runner.validate_results(validated_args(runtime_phase_trace=True), producer, None, [consumer])


def test_validate_results_accepts_explicit_readiness_poll_latency_buckets() -> None:
    producer, consumer = valid_records()
    consumer["ingress_wake_trace"] = complete_ingress_wake_trace(readiness_poll_elided=False)

    runner.validate_results(validated_args(runtime_phase_trace=True), producer, None, [consumer])


def test_validate_results_rejects_wrong_readiness_poll_latency_bucket_shape() -> None:
    producer, consumer = valid_records()
    consumer["ingress_wake_trace"] = complete_ingress_wake_trace(readiness_poll_elided=False)
    consumer["ingress_wake_trace"]["readiness_poll_to_compute_dispatch_ns"] = (
        empty_ingress_latency_bucket()
    )

    with pytest.raises(RuntimeError, match="invalid ingress-wake latency buckets"):
        runner.validate_results(
            validated_args(runtime_phase_trace=True), producer, None, [consumer]
        )


def test_validate_results_rejects_wrong_lane_wake_mode() -> None:
    # Key setup: request busy lanes but retain the sleeping mode in both worker records.
    producer, consumer = valid_records()

    # Key assertion: a mislabeled scheduling result cannot qualify the A/B comparison.
    with pytest.raises(RuntimeError, match="wrong lane wake mode"):
        runner.validate_results(validated_args(busy_poll=True), producer, None, [consumer])


def test_validate_results_accepts_two_independent_consumers() -> None:
    # Key setup: two indexed consumers independently select distinct process groups on one
    # compiler-coalesced broadcast service.
    producer, first = valid_records()
    second = copy.deepcopy(first)
    for record in (producer, first, second):
        record["benchmark"] = "runtime_graph_interprocess_pod_1to2"
        record["geometry"]["fanout"] = 2
        record["geometry"]["processes"] = 3
    first["role"] = "consumer-1"
    second["role"] = "consumer-2"
    second["consumer_index"] = 2
    second["process_id"] = 102
    second["process_group"] = 3
    second["support_threads"][0]["native_tid"] = 1004
    second["support_threads"][1]["native_tid"] = 1005
    args = validated_args(
        fanout=2,
        consumer_cpus=[-1, -1],
        consumer_controller_cpus=[-1, -1],
        consumer_pump_cpus=[-1, -1],
    )

    # Key assertion: one shared service plus exact per-sink delivery accounting qualifies 1:2.
    runner.validate_results(args, producer, None, [first, second])


def test_validate_results_accepts_three_process_chain() -> None:
    # Key setup: insert one independently scheduled relay between producer and consumer.
    producer, consumer = valid_records()
    relay = copy.deepcopy(consumer)
    for record in (producer, relay, consumer):
        record["benchmark"] = "runtime_graph_interprocess_pod_source_relay_sink"
        record["service_id"] = ""
        record["service_ids"] = ["source-relay", "relay-sink"]
        record["geometry"].update(topology="chain", processes=3, transport_hops=2)
    relay["role"] = "relay"
    relay["consumer_index"] = None
    relay["process_id"] = 102
    relay["process_group"] = 3
    relay["counts"] = {
        "source_starts": 0,
        "relay_starts": 1,
        "consumer_starts": 0,
        "publications": 0,
        "relay_inputs": 10,
        "relay_publications": 10,
        "deliveries": 0,
        "final_sequence": 10,
    }
    relay["support_threads"][0]["native_tid"] = 1004
    relay["support_threads"][1]["native_tid"] = 1005
    args = validated_args(topology="chain")

    # Key assertion: both transport hops and all three isolated roles qualify together.
    runner.validate_results(args, producer, relay, [consumer])


def test_common_command_carries_complete_fanout_placement() -> None:
    # Key setup: independently pin both consumer lanes, controllers, and pumps.
    args = validated_args(
        fanout=2,
        consumer_cpus=[2, 3],
        consumer_controller_cpus=[4, 5],
        consumer_pump_cpus=[6, 7],
    )
    args.benchmark_binary = Path("benchmark")

    # Key assertion: every worker receives identical compile-time and support-thread geometry.
    command = runner.common_command(args, "run")
    assert command[command.index("--fanout") + 1] == "2"
    assert command[command.index("--consumer-cpus") + 1] == "2,3"
    assert command[command.index("--consumer-controller-cpus") + 1] == "4,5"
    assert command[command.index("--consumer-pump-cpus") + 1] == "6,7"
    assert command[command.index("--start-timeout-ms") + 1] == "30000"


def test_release_producer_start_reports_early_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken_write(_: int, __: bytes) -> int:
        raise BrokenPipeError

    monkeypatch.setattr(runner.os, "write", broken_write)

    with pytest.raises(RuntimeError, match="producer exited before the start barrier"):
        runner.release_producer_start(17)


def test_worker_failure_details_reports_bounded_role_logs(tmp_path: Path) -> None:
    # Key setup: retain one short role log and one log larger than the diagnostic bound.
    producer_log = tmp_path / "producer.log"
    consumer_log = tmp_path / "consumer.log"
    producer_log.write_text("producer failed\n", encoding="utf-8")
    consumer_log.write_text(
        "discarded-prefix\n" + "x" * runner.WORKER_FAILURE_LOG_LIMIT,
        encoding="utf-8",
    )

    details = runner.worker_failure_details([producer_log, consumer_log])

    # Key assertion: CTest receives each role's failure and a visible truncation marker.
    assert "--- producer.log\nproducer failed" in details
    assert "--- consumer.log\n[... earlier output truncated ...]" in details
    assert "discarded-prefix" not in details


def test_common_command_carries_tensor_identity() -> None:
    # Key setup: select a non-default device-backed 1 MiB Tensor scenario.
    args = validated_args(
        payload="tensor",
        tensor_storage="device",
        tensor_device=1,
        bulk_work_profile="descriptor_only",
    )
    args.benchmark_binary = Path("benchmark")

    # Key assertion: every independently compiling worker receives identical Tensor geometry.
    command = runner.common_command(args, "run")
    assert command[command.index("--payload") + 1] == "tensor"
    assert command[command.index("--tensor-storage") + 1] == "device"
    assert command[command.index("--tensor-bytes") + 1] == str(1 << 20)
    assert command[command.index("--tensor-device") + 1] == "1"
    assert command[command.index("--bulk-work-profile") + 1] == "descriptor_only"


def test_validate_args_accepts_saturated_distributed_fanout() -> None:
    # Key setup: select the compiler-coalesced 1:2 topology with unpaced publication.
    args = make_args(
        workload="saturated",
        publisher_period_us=0,
        fanout=2,
        queue_depth=2,
        consumer_cpus=[-1, -1],
        consumer_controller_cpus=[-1, -1],
        consumer_pump_cpus=[-1, -1],
    )

    # Key assertion: bounded graph pressure now admits the saturated fanout workload.
    runner.validate_args(args)


def test_write_summary_separates_logical_and_aggregate_rates(tmp_path: Path) -> None:
    # Key setup: the second sink completes later than the first for one 1:2 repetition.
    producer, first = valid_records()
    second = copy.deepcopy(first)
    producer["timing"]["messages_per_second"] = 500.0
    first["timing"].update(
        measured_start_ns=100,
        measured_end_ns=200,
        messages_per_second=400.0,
    )
    second["timing"].update(
        measured_start_ns=100,
        measured_end_ns=300,
        messages_per_second=300.0,
    )
    first["timing"]["end_to_end_ns"] = {"p50": 70, "p99": 100}
    second["timing"]["end_to_end_ns"] = {"p50": 90, "p99": 120}
    args = validated_args(
        fanout=2,
        consumer_cpus=[-1, -1],
        consumer_controller_cpus=[-1, -1],
        consumer_pump_cpus=[-1, -1],
    )

    second_run_producer = copy.deepcopy(producer)
    second_run_first = copy.deepcopy(first)
    second_run_second = copy.deepcopy(second)
    second_run_producer["timing"]["messages_per_second"] = 600.0
    second_run_first["timing"]["end_to_end_ns"]["p99"] = 110
    second_run_second["timing"]["end_to_end_ns"]["p99"] = 132

    runner.write_summary(
        args,
        tmp_path,
        [
            (producer, None, [first, second]),
            (second_run_producer, None, [second_run_first, second_run_second]),
        ],
    )
    summary = runner.checked_json(tmp_path / "summary.json")

    # Key assertion: publication, delivery, slowest-sink p99, and run-to-run CV remain distinct.
    assert summary["logical_publications_per_second"]["p50"] == 550.0
    assert summary["logical_publications_per_second"]["coefficient_of_variation"] == pytest.approx(
        0.128564869306645
    )
    assert summary["slowest_sink_messages_per_second"]["p50"] == 300.0
    assert summary["aggregate_deliveries_per_second"]["p50"] == 80_000_000.0
    assert summary["slowest_sink_end_to_end_p50_ns"]["p50"] == 90.0
    assert summary["slowest_sink_end_to_end_p99_ns"]["p50"] == 126.0
    assert summary["slowest_sink_end_to_end_p99_ns"]["coefficient_of_variation"] == pytest.approx(
        0.06734350297014738
    )
    assert summary["slowest_sink_end_to_end_p999_ns"] is None
    assert summary["wake"] == {
        "event_wake": True,
        "busy_poll": False,
        "receive_polling": False,
    }
    assert summary["runtime_support_threads"] == "controller,channel_pump"


def test_write_summary_omits_latency_for_saturated_workload(tmp_path: Path) -> None:
    # Key setup: saturated worker evidence reports rates but no paced-latency distribution.
    producer, consumer = valid_records()
    producer["timing"]["messages_per_second"] = 500.0
    consumer["timing"]["messages_per_second"] = 400.0
    args = validated_args(
        workload="saturated",
        publisher_period_us=0,
    )

    runner.write_summary(args, tmp_path, [(producer, None, [consumer])])
    summary = runner.checked_json(tmp_path / "summary.json")

    # Key assertion: saturated summaries retain throughput without reading absent latency data.
    assert summary["logical_publications_per_second"]["p50"] == 500.0
    assert summary["slowest_sink_end_to_end_p50_ns"] is None
