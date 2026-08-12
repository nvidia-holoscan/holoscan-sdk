# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os
from types import SimpleNamespace
from typing import Any

import pytest
import run_holomq_benchmark as runner


def make_args() -> SimpleNamespace:
    return SimpleNamespace(
        payload_bytes=64,
        samples=1_000,
        warmup=100,
        queue_depth=8,
        subscriber_count=2,
        repetitions=2,
        preset="production",
        workload="throughput",
        payload="inline",
        storage="pageable_host",
        device_sync="none",
        buffer_bytes=4096,
        pool_slots=10,
        publisher_period_us=0,
        publisher_pacing="spin",
        payload_validation="final",
        publisher_cpu=2,
        subscriber_cpu_values=[3, 4],
    )


def make_result(role: str, **updates: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": 1,
        "benchmark": "holomq_inline_transport",
        "role": role,
        "payload": "inline",
        "storage": "inline",
        "payload_bytes": 64,
        "samples": 1_000,
        "warmup": 100,
        "queue_depth": 8,
        "subscriber_count": 2,
        "workload": "throughput",
        "publisher_period_us": 0,
        "publisher_pacing": "spin",
        "rate_limited": False,
        "event_wake": False,
        "clock": {
            "source": "clock_gettime(CLOCK_MONOTONIC)",
            "steady": True,
            "scope": "same_host_cross_process",
        },
        "payload_validation": "final",
        "validation_timing": "after_timed_operations",
        "checksum": 1234,
        "backpressured_loans": 0,
        "connection_probe_backpressured_loans": 0,
    }
    result.update(updates)
    return result


class TestValidateResults:
    @pytest.mark.parametrize(
        "options", [("--storage", "device"), ("--device-sync", "publisher_ready")]
    )
    def test_inline_payload_rejects_device_only_options(
        self, options: tuple[str, str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        common = [
            "run_holomq_benchmark.py",
            "--binary",
            "holomq_transport_benchmark",
            "--output-dir",
            "benchmark-output",
        ]
        monkeypatch.setattr(runner.sys, "argv", [*common, *options])
        with pytest.raises(SystemExit):
            runner.parse_args()

    def test_cuda_unavailable_before_readiness_is_skipped(self) -> None:
        ready_read_fd, ready_write_fd = os.pipe()
        os.close(ready_write_fd)
        subscriber = SimpleNamespace(poll=lambda: runner.CUDA_UNAVAILABLE_EXIT_CODE)

        try:
            with pytest.raises(SystemExit, match=str(runner.CUDA_UNAVAILABLE_EXIT_CODE)):
                runner.wait_ready([subscriber], [ready_read_fd], 1, True)
        finally:
            os.close(ready_read_fd)

    def test_cpu_topology_skips_non_linux_platforms(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(runner.sys, "platform", "win32")
        placement = runner.read_cpu_topology("publisher", 2)

        assert not placement["topology_available"]
        assert placement["topology_reason"] == "unsupported_platform"

    def test_cpu_placement_rejects_smt_siblings(self) -> None:
        placements = [
            {
                "role": "publisher",
                "cpu": 2,
                "pinned": True,
                "topology_available": True,
                "package_id": 0,
                "core_id": 2,
            },
            {
                "role": "subscriber_0",
                "cpu": 20,
                "pinned": True,
                "topology_available": True,
                "package_id": 0,
                "core_id": 2,
            },
        ]

        with pytest.raises(RuntimeError, match="share package/core"):
            runner.validate_distinct_physical_cores(placements)

    def test_cpu_placement_accepts_distinct_cores(self) -> None:
        placements = [
            {
                "role": "publisher",
                "cpu": 2,
                "pinned": True,
                "topology_available": True,
                "package_id": 0,
                "core_id": 2,
            },
            {
                "role": "subscriber_0",
                "cpu": 3,
                "pinned": True,
                "topology_available": True,
                "package_id": 0,
                "core_id": 3,
            },
        ]

        runner.validate_distinct_physical_cores(placements)

    def test_production_preset_respects_explicit_overrides(self) -> None:
        args = SimpleNamespace(
            preset="production",
            workload="throughput",
            samples=None,
            warmup=500,
            queue_depth=None,
            repetitions=None,
            publisher_period_us=None,
            pool_slots=None,
        )

        runner.apply_preset(args)

        assert args.samples == 100_000
        assert args.warmup == 500
        assert args.queue_depth == 64
        assert args.repetitions == 7
        assert args.publisher_period_us == 0
        assert args.pool_slots == 66

    def test_latency_preset_uses_depth_one_and_default_period(self) -> None:
        args = SimpleNamespace(
            preset="production",
            workload="latency",
            samples=None,
            warmup=None,
            queue_depth=None,
            repetitions=None,
            publisher_period_us=None,
            pool_slots=None,
        )

        runner.apply_preset(args)

        assert args.queue_depth == 1
        assert args.publisher_period_us == 10
        assert args.pool_slots == 3

    def test_buffer_handle_latency_uses_release_headroom(self) -> None:
        args = SimpleNamespace(
            preset="production",
            workload="latency",
            payload="buffer_handle",
            storage="device",
            device_sync="ready",
            samples=None,
            warmup=None,
            queue_depth=None,
            repetitions=None,
            publisher_period_us=None,
            pool_slots=None,
        )

        runner.apply_preset(args)

        assert args.queue_depth == 1
        assert args.publisher_period_us == 500
        assert args.pool_slots == 10

    def test_paced_workload_timeout_covers_default_duration(self) -> None:
        args = SimpleNamespace(
            preset="production",
            workload="latency",
            payload="buffer_handle",
            storage="device",
            device_sync="ready",
            samples=None,
            warmup=None,
            queue_depth=None,
            repetitions=None,
            publisher_period_us=None,
            pool_slots=None,
        )

        runner.apply_preset(args)

        assert runner.default_timeout_sec(args) == 65

    def test_buffer_handle_result_keeps_cold_and_warm_import_metrics(self) -> None:
        args = make_args()
        args.payload = "buffer_handle"
        args.subscriber_count = 1
        args.subscriber_cpu_values = [3]
        publisher = make_result(
            "publisher",
            benchmark="holomq_buffer_handle_transport",
            payload="buffer_handle",
            storage="pageable_host",
            payload_bytes=424,
            subscriber_count=1,
            buffer_bytes=4096,
            pool_slots=10,
            device_data_work="none",
            device_sync="none",
            measured_start_ns=1_000_000_000,
            measured_end_ns=1_002_000_000,
            buffer_acquire_p50_ns=100,
            buffer_acquire_p99_ns=200,
            buffer_export_p50_ns=50,
            buffer_export_p99_ns=80,
        )
        subscriber = make_result(
            "subscriber",
            benchmark="holomq_buffer_handle_transport",
            payload="buffer_handle",
            storage="pageable_host",
            payload_bytes=424,
            subscriber_count=1,
            subscriber_index=0,
            buffer_bytes=4096,
            pool_slots=10,
            device_data_work="none",
            device_sync="none",
            validation_errors=0,
            measured_end_ns=1_004_000_000,
            cold_first_import_ns=50_000,
            buffer_import_p50_ns=120,
            buffer_import_p99_ns=250,
            final_validation_ns=500,
            latency_p50_ns=800,
            latency_p99_ns=1_000,
            latency_p999_ns=1_200,
            latency_max_ns=1_400,
        )

        combined = runner.validate_results(args, publisher, [subscriber])

        assert combined["cold_first_import_ns"] == 50_000
        assert combined["buffer_import_p50_ns"] == 120
        assert combined["final_validation_ns"] == 500
        assert combined["application_payload_bytes"] == 4096
        assert combined["inline_transport_bytes"] == 424
        assert combined["external_buffer_bytes"] == 4096
        assert combined["payload_write_profile"] == "boundary_markers_per_sample"
        assert combined["bulk_work_profile"] == "descriptor_only"
        assert combined["wake"] == {
            "event_wake": False,
            "busy_poll": False,
            "receive_polling": True,
        }
        assert combined["runtime_support_threads"] == "none"

        args.storage = "device"
        publisher["storage"] = "device"
        publisher["device_data_work"] = "preinitialized"
        subscriber["storage"] = "device"
        subscriber["device_data_work"] = "preinitialized"

        for device_sync in ("none", "publisher_ready", "subscriber_release", "ready"):
            args.device_sync = device_sync
            publisher["device_sync"] = device_sync
            subscriber["device_sync"] = device_sync
            device_combined = runner.validate_results(args, publisher, [subscriber])

            assert device_combined["storage"] == "device"
            assert device_combined["device_sync"] == device_sync
            assert device_combined["payload_write_profile"] == "reused_preinitialized_contents"
            assert device_combined["bulk_work_profile"] == "descriptor_only"

    def test_fanout_completion_uses_slowest_subscriber(self) -> None:
        publisher = make_result(
            "publisher", measured_start_ns=1_000_000_000, measured_end_ns=1_001_000_000
        )
        subscribers = [
            make_result(
                "subscriber",
                subscriber_index=0,
                measured_end_ns=1_002_000_000,
                latency_p50_ns=700,
                latency_p99_ns=900,
                latency_p999_ns=1_100,
                latency_max_ns=1_300,
            ),
            make_result(
                "subscriber",
                subscriber_index=1,
                measured_end_ns=1_004_000_000,
                latency_p50_ns=800,
                latency_p99_ns=1_000,
                latency_p999_ns=1_200,
                latency_max_ns=1_400,
            ),
        ]

        combined = runner.validate_results(make_args(), publisher, subscribers)

        assert combined["completed_messages_per_second"] == pytest.approx(250_000)
        assert combined["completed_deliveries_per_second"] == pytest.approx(500_000)
        assert combined["worst_subscriber_latency_p50_ns"] == 800

    def test_fanout_completion_rejects_missing_subscriber_timestamp(self) -> None:
        publisher = make_result(
            "publisher", measured_start_ns=1_000_000_000, measured_end_ns=1_001_000_000
        )
        subscribers = [
            make_result(
                "subscriber",
                subscriber_index=index,
                measured_end_ns=1_002_000_000 if index == 0 else None,
                latency_p50_ns=700,
                latency_p99_ns=900,
                latency_p999_ns=1_100,
                latency_max_ns=1_300,
            )
            for index in range(2)
        ]

        with pytest.raises(RuntimeError, match="missing valid measured timestamps"):
            runner.validate_results(make_args(), publisher, subscribers)

    def test_repetition_summary_uses_metric_medians(self) -> None:
        metric_names = (
            "publisher_samples_per_second",
            "completed_messages_per_second",
            "completed_deliveries_per_second",
            "worst_subscriber_latency_p50_ns",
            "worst_subscriber_latency_p99_ns",
            "worst_subscriber_latency_p999_ns",
            "worst_subscriber_latency_max_ns",
        )
        results = []
        for repetition, value in enumerate((100.0, 300.0)):
            result = {name: value for name in metric_names}
            result["repetition"] = repetition
            result["benchmark"] = "holomq_inline_transport"
            result["storage"] = "inline"
            result["payload_bytes"] = 64
            result["application_payload_bytes"] = 64
            result["inline_transport_bytes"] = 64
            result["external_buffer_bytes"] = 0
            result["payload_write_profile"] = "full_payload_per_sample"
            result["bulk_work_profile"] = "not_applicable"
            result["wake"] = {
                "event_wake": False,
                "busy_poll": False,
                "receive_polling": True,
            }
            result["clock"] = {
                "source": "clock_gettime(CLOCK_MONOTONIC)",
                "steady": True,
                "scope": "same_host_cross_process",
            }
            result["runtime_support_threads"] = "none"
            result["publication_counts"] = {
                "scope": "measured_plus_warmup",
                "attempted": 1_100,
                "accepted": 1_100,
                "backpressured": 0,
                "indeterminate": 0,
                "completed": 1_100,
            }
            result["connection_probe_backpressured_loans"] = 0
            results.append(result)

        summary = runner.summarize_results(make_args(), results)

        assert summary["repetitions"] == 2
        assert summary["application_payload_bytes"] == 64
        assert summary["inline_transport_bytes"] == 64
        assert summary["external_buffer_bytes"] == 0
        assert summary["wake"]["receive_polling"]
        assert summary["runtime_support_threads"] == "none"
        assert summary["payload_write_profile"] == "full_payload_per_sample"
        assert summary["bulk_work_profile"] == "not_applicable"
        assert summary["connection_probe_backpressured_loans"] == 0
        for name in metric_names:
            if name == "worst_subscriber_latency_p999_ns":
                continue
            assert summary["metrics"][name]["median"] == 200.0
        assert "worst_subscriber_latency_p999_ns" not in summary["metrics"]
