# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import run_runtime_graph_benchmark as runner


def clock_evidence() -> dict[str, object]:
    return {
        "source": "std::chrono::steady_clock",
        "steady": True,
        "scope": "single_process",
    }


def timestamp_instrumentation() -> dict[str, object]:
    return {
        "calibration": "post_measurement_back_to_back_pair",
        "clock_reads_per_sample": 2,
        "pair_overhead_ns": {"samples": 4_096, "p50": 20},
    }


def summary_evidence(source_trigger: str) -> dict[str, object]:
    return {
        "source_trigger": source_trigger,
        "clock": clock_evidence(),
        "timestamp_instrumentation": timestamp_instrumentation(),
        "counts": {
            "consumer_visible_replacements": 0,
            "replacement_rate": 0.0,
        },
    }


def partitioning_evidence(args: SimpleNamespace) -> dict[str, object]:
    algorithm, mode = runner.expected_partitioning(args)
    lane_count = runner.expected_operator_lane_count(args)
    authored_count = {
        "automatic": 0,
        "include_all": 1,
        "hybrid": 1,
    }.get(args.partition_cover, lane_count)
    automatic_count = (
        lane_count
        if args.partition_cover == "automatic"
        else lane_count - 1
        if args.partition_cover == "hybrid"
        else 0
    )
    return {
        "requested_algorithm": {"id": algorithm, "version": 1},
        "accepted_algorithm": {"id": algorithm, "version": 1},
        "selection_source": "core_default",
        "authoring": args.partition_cover,
        "mode": mode,
        "partition_count": lane_count,
        "authored_partition_count": authored_count,
        "automatic_partition_count": automatic_count,
        "cover_signature": "test-cover",
    }


def make_args(**updates: object) -> SimpleNamespace:
    values = {
        "samples": 100,
        "workload": "latency",
        "scenario": "pod",
        "operator_lanes": "separate",
        "partition_cover": "legacy",
        "warmup": 10,
        "queue_policy": "blocking",
        "queue_depth": 1,
        "sink_queue_depths": [],
        "relay_count": 1,
        "fanout": 1,
        "multi_output": False,
        "multi_input": False,
        "borrow_image_input": False,
        "mixed_policy_fanout": False,
        "notification_capacity": 1,
        "offered_rate_hz": 0,
        "offered_burst_size": 1,
        "publisher_period_us": 1_000,
        "callback_work_us": 0,
        "experimental_fanout": "disabled",
        "fanout_attribution": False,
        "measure_source_period": False,
        "disable_blocked_head_wake": False,
        "source_work_us": None,
        "sink_work_us": None,
        "emit_then_work": False,
        "burst_size": 1,
        "lane_fifo_priority": 0,
        "sink_fifo_priorities": [],
        "allow_pump_lane_cpu_sharing": False,
        "source_cpu": -1,
        "relay_cpu": -1,
        "sink_cpu": -1,
        "sink_cpus": [],
        "controller_cpu": -1,
        "channel_pump_cpu": -1,
        "notification_driver_cpu": -1,
        "session_timeout_ms": 10_000,
        "process_timeout_sec": 15.0,
        "repetitions": 3,
        "max_median_cv": 0.10,
        "payload_validation": "final",
        "tensor_storage": "pageable_host",
        "tensor_bytes": 1 << 20,
        "tensor_device": 0,
        "bulk_work_profile": "full_write",
        "busy_poll": False,
        "runtime_phase_counters": False,
        "disable_direct_continuation_elision": False,
        "disable_closed_island_composite": False,
        "force_holomq_process_local_inline": False,
        "force_holomq_fanout": False,
        "force_holomq_event_wake": False,
        "force_process_local_channel_pump": False,
        "force_process_local_pump_delivery": False,
        "test_gate_pod_sink_for_pressure": False,
        "source_state": "unknown",
        "source_snapshot_sha256": None,
    }
    values.update(updates)
    return SimpleNamespace(**values)


def test_read_cpu_topology_keeps_siblings_when_core_id_is_malformed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    topology_values = {
        "physical_package_id": "0",
        "core_id": "unknown",
        "thread_siblings_list": "7",
        "core_type": None,
        "scaling_governor": "performance",
    }
    monkeypatch.setattr(runner.sys, "platform", "linux")
    monkeypatch.setattr(runner.os, "sched_getaffinity", lambda _pid: {7}, raising=False)
    monkeypatch.setattr(runner, "read_text", lambda path: topology_values[path.name])

    topology = runner.read_cpu_topology("source", 7)

    assert topology["package_id"] == 0
    assert topology["core_id"] is None
    assert topology["thread_siblings"] == "7"
    assert topology["topology_available"] is True


def test_expected_operator_lane_count_keeps_relay_topology_distinct_from_fanout() -> None:
    # sample_forward authors source, relay, and one sink; argument validation rejects fanout there.
    assert runner.expected_operator_lane_count(make_args(scenario="sample_forward", fanout=4)) == 3
    assert runner.expected_operator_lane_count(make_args(scenario="pod", fanout=4)) == 5
    assert runner.expected_sink_operator_count(make_args(multi_input=True, fanout=2)) == 1
    assert (
        runner.expected_operator_lane_count(
            make_args(partition_cover="automatic", multi_input=True, fanout=2)
        )
        == 1
    )


def test_automatic_lane_expectation_is_topology_derived_not_request_derived() -> None:
    automatic = make_args(partition_cover="automatic", scenario="pod", fanout=3)

    assert runner.expected_operator_lane_count(automatic) == 4


def test_effective_automatic_lane_count_comes_from_the_accepted_plan() -> None:
    # The compiler selects automatic partitions; a requested lane policy cannot redefine them.
    automatic = make_args(partition_cover="automatic", operator_lanes="shared", fanout=3)

    assert runner.effective_operator_lane_count({"operator_lane_count": 4}, automatic) == 4
    assert (
        runner.effective_operator_lane_count(
            {"operator_lane_count": 1},
            make_args(
                partition_cover="automatic",
                scenario="source_only",
                workload="notification_latency",
            ),
        )
        == 1
    )
    with pytest.raises(RuntimeError, match="invalid effective Operator lane count"):
        runner.effective_operator_lane_count({"operator_lane_count": 0}, automatic)


def test_automatic_source_only_keeps_only_controller_topology_evidence() -> None:
    automatic = make_args(
        partition_cover="automatic",
        scenario="source_only",
        workload="notification_latency",
    )

    runner.validate_args(automatic)
    assert [item["role"] for item in runner.collect_cpu_topology(automatic)] == [
        "controller",
        "notification_driver",
    ]


@pytest.mark.parametrize(
    ("partition_cover", "operator_lanes", "scenario", "fanout", "lanes", "algorithm", "mode"),
    [
        ("legacy", "separate", "pod", 1, 2, "holoscan.partition.serial_baseline", "manual"),
        ("include_all", "shared", "pod", 1, 1, "holoscan.partition.serial_baseline", "manual"),
        ("automatic", "separate", "pod", 1, 1, "holoscan.partition.source_chain", "automatic"),
        (
            "automatic",
            "separate",
            "pod",
            3,
            4,
            "holoscan.partition.source_chain",
            "automatic",
        ),
        (
            "hybrid",
            "separate",
            "pod",
            1,
            2,
            "holoscan.partition.source_chain",
            "constrained_automatic",
        ),
    ],
)
def test_partition_cover_predicts_accepted_plan(
    partition_cover: str,
    operator_lanes: str,
    scenario: str,
    fanout: int,
    lanes: int,
    algorithm: str,
    mode: str,
) -> None:
    args = make_args(
        partition_cover=partition_cover,
        operator_lanes=operator_lanes,
        scenario=scenario,
        fanout=fanout,
    )
    assert runner.expected_operator_lane_count(args) == lanes
    assert runner.expected_partitioning(args) == (algorithm, mode)


def test_benchmark_command_preserves_legacy_default_and_forwards_cover(tmp_path: Path) -> None:
    args = make_args(partition_cover="automatic")

    command = runner.benchmark_command(args, Path("benchmark"), tmp_path / "result.json")

    assert command[command.index("--partition-cover") + 1] == "automatic"


def test_blocked_head_wake_control_is_forwarded_named_and_reported(tmp_path: Path) -> None:
    # Key setup: the private A/B control that leaves a blocked head visible to the Channel pump.
    args = make_args(force_holomq_event_wake=True, disable_blocked_head_wake=True)

    command = runner.benchmark_command(args, Path("benchmark"), tmp_path / "result.json")

    # Key assertion: the control reaches the binary and renames the result, so an A/B pair can
    # never be mistaken for one configuration.
    assert "--disable-blocked-head-wake" in command
    assert "--disable-blocked-head-wake" not in runner.benchmark_command(
        make_args(force_holomq_event_wake=True), Path("benchmark"), tmp_path / "result.json"
    )
    assert runner.benchmark_name(
        "pod", "latency", force_holomq_event_wake=True, disable_blocked_head_wake=True
    ).endswith("_event_wake_control_without_blocked_head_wake")


def test_expected_execution_lane_worker_count_excludes_inline_partition_owners() -> None:
    assert (
        runner.expected_execution_lane_worker_count(
            make_args(scenario="pod", operator_lanes="shared")
        )
        == 0
    )
    assert runner.expected_execution_lane_worker_count(make_args(scenario="source_only")) == 0
    assert (
        runner.expected_execution_lane_worker_count(
            make_args(scenario="pod", operator_lanes="separate")
        )
        == 2
    )
    assert runner.expected_execution_lane_worker_count(make_args(scenario="pod", fanout=4)) == 5
    assert runner.expected_execution_lane_worker_count(make_args(partition_cover="automatic")) == 0
    assert (
        runner.expected_execution_lane_worker_count(
            make_args(partition_cover="automatic", fanout=4)
        )
        == 5
    )


@pytest.mark.parametrize(
    ("latency", "overlaps", "expected", "complete"),
    [
        ({"samples": 95}, 5, 100, True),
        (None, 100, 100, True),
        ({"samples": 95}, 4, 100, False),
        ({"samples": 100}, -1, 100, False),
        (None, 99, 100, False),
    ],
)
def test_overlap_timing_requires_every_sample_to_be_classified(
    latency: object, overlaps: object, expected: int, complete: bool
) -> None:
    # Concurrent consumers can complete before the producer records its post-call boundary.
    assert runner.complete_overlap_interval(latency, overlaps, expected) is complete


def test_optional_overlap_summary_preserves_all_overlap_runs() -> None:
    # An interval can be undefined for every sample when the consumer always wins the race.
    result = {
        "timing": {
            "source_post_emit_to_sink_ns": None,
            "source_post_emit_to_sink_by_sink_ns": [
                {"index": 0, "overlaps": 100, "latency_ns": None}
            ],
        }
    }

    assert (
        runner.optional_numeric_path(result, "timing", "source_post_emit_to_sink_ns", "p50") is None
    )
    assert (
        runner.per_sink_optional_numeric_path(
            result,
            "source_post_emit_to_sink_by_sink_ns",
            0,
            "latency_ns",
            "p50",
        )
        is None
    )
    assert runner.summarize_optional_values([None, None]) == {
        "count": 0,
        "median": None,
        "min": None,
        "max": None,
        "mean": None,
        "coefficient_of_variation": None,
    }


class TestArgumentValidation:
    @pytest.fixture(autouse=True)
    def _use_synthetic_cpu_topology(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Keep role/placement unit tests independent of the host or container cpuset.
        def synthetic_topology(role: str, cpu: int) -> dict[str, object]:
            result: dict[str, object] = {"role": role, "cpu": cpu, "pinned": cpu >= 0}
            if cpu >= 0:
                result.update(
                    {
                        "package_id": 0,
                        "core_id": cpu,
                        "topology_available": True,
                        "thread_siblings": str(cpu),
                        "core_type": None,
                        "governor": "performance",
                    }
                )
            return result

        monkeypatch.setattr(runner, "read_cpu_topology", synthetic_topology)

    def test_singleton_sibling_sets_override_ambiguous_core_ids(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def thor_topology(role: str, cpu: int) -> dict[str, object]:
            return {
                "role": role,
                "cpu": cpu,
                "pinned": cpu >= 0,
                "package_id": 0,
                "core_id": 0,
                "topology_available": True,
                "thread_siblings": str(cpu),
            }

        monkeypatch.setattr(runner, "read_cpu_topology", thor_topology)
        args = make_args(source_cpu=0, sink_cpu=1, controller_cpu=2)

        topology = runner.collect_cpu_topology(args)

        assert len({runner.physical_core_identity(item) for item in topology}) == len(topology)

    # Each vector isolates one CLI contract so an invalid workload cannot produce mislabeled data.
    @pytest.mark.parametrize(
        ("updates", "message"),
        [
            ({"samples": 0}, "--samples must be positive"),
            ({"warmup": -1}, "--warmup must be nonnegative"),
            ({"callback_work_us": 2}, "--callback-work-us must be 0, 1, 5, 20, or 100"),
            ({"queue_depth": 3}, "latency workloads require --queue-depth 1 or 2"),
            (
                {"scenario": "multi_tensor", "queue_policy": "drop_oldest"},
                "--queue-policy drop_oldest supports pod, schema, third_party_schema, tensor, "
                "image, multi_tensor_loan, or multi_tensor_staging",
            ),
            (
                {"queue_policy": "drop_oldest", "fanout": 2},
                "--queue-policy drop_oldest fanout requires --offered-rate-hz",
            ),
            (
                {"offered_rate_hz": -1},
                "--offered-rate-hz must be between 1 and 1000000 when enabled",
            ),
            (
                {"offered_rate_hz": 1_000_001},
                "--offered-rate-hz must be between 1 and 1000000 when enabled",
            ),
            (
                {"offered_rate_hz": 100_000},
                "--offered-rate-hz requires shared-lane saturated POD drop_oldest fanout",
            ),
            (
                {"offered_burst_size": 2},
                "--offered-burst-size requires --offered-rate-hz",
            ),
            (
                {
                    "workload": "saturated",
                    "scenario": "pod",
                    "operator_lanes": "shared",
                    "queue_policy": "drop_oldest",
                    "queue_depth": 2,
                    "fanout": 2,
                    "offered_rate_hz": 100_000,
                    "offered_burst_size": 4,
                    "samples": 10,
                },
                "--offered-rate-hz requires --samples to contain only complete bursts",
            ),
            ({"relay_count": 0}, "--relay-count must be between 1 and 64"),
            ({"relay_count": 65}, "--relay-count must be between 1 and 64"),
            ({"relay_count": 2}, "--relay-count requires --scenario sample_forward"),
            ({"fanout": 0}, "--fanout must be between 1 and 32"),
            ({"fanout": 33}, "--fanout must be between 1 and 32"),
            (
                {"mixed_policy_fanout": True},
                "--mixed-policy-fanout requires shared-lane blocking POD --fanout 2",
            ),
            (
                {"scenario": "tensor", "operator_lanes": "shared", "fanout": 16},
                "--fanout greater than 8 requires --scenario pod, schema, or third_party_schema",
            ),
            (
                {"scenario": "signal", "fanout": 2},
                "--fanout greater than one requires --scenario pod, schema, third_party_schema, "
                "tensor, image, multi_tensor_loan, or multi_tensor_staging",
            ),
            (
                {
                    "workload": "saturated",
                    "queue_depth": 4,
                    "fanout": 1,
                    "sink_queue_depths": [4],
                },
                "--sink-queue-depths requires supported --fanout greater than one",
            ),
            (
                {
                    "workload": "saturated",
                    "queue_depth": 4,
                    "fanout": 2,
                    "sink_queue_depths": [4],
                },
                "--sink-queue-depths must provide exactly one depth per fanout sink",
            ),
            (
                {
                    "workload": "saturated",
                    "queue_depth": 4,
                    "fanout": 2,
                    "sink_queue_depths": [4, 0],
                },
                "--sink-queue-depths values must be positive",
            ),
            (
                {
                    "workload": "saturated",
                    "queue_depth": 4,
                    "fanout": 2,
                    "sink_queue_depths": [3, 2],
                },
                "maximum --sink-queue-depths value must equal --queue-depth",
            ),
            (
                {
                    "workload": "saturated",
                    "scenario": "pod",
                    "operator_lanes": "shared",
                    "queue_depth": 4,
                    "fanout": 2,
                    "sink_queue_depths": [4, 2],
                    "multi_output": True,
                    "force_holomq_process_local_inline": True,
                },
                "--multi-output/--multi-input requires equal sink queue depths",
            ),
            (
                {"fanout": 2, "force_holomq_process_local_inline": True},
                "--force-holomq-process-local-inline requires --fanout 1, --multi-output, or "
                "--multi-input",
            ),
            (
                {"force_holomq_event_wake": True},
                "--force-holomq-event-wake requires a shared-lane",
            ),
            (
                {"force_process_local_channel_pump": True},
                "--force-process-local-channel-pump requires an otherwise eventless shared-lane",
            ),
            (
                {
                    "operator_lanes": "shared",
                    "force_holomq_process_local_inline": True,
                    "force_holomq_event_wake": True,
                    "force_process_local_channel_pump": True,
                },
                "--force-process-local-channel-pump requires an otherwise eventless shared-lane",
            ),
            (
                {"workload": "saturated", "queue_depth": 0},
                "saturated throughput requires a positive --queue-depth",
            ),
            ({"notification_capacity": 0}, "--notification-capacity must be positive"),
            (
                {"notification_capacity": 2},
                "--notification-capacity requires a notification workload",
            ),
            (
                {"workload": "notification_latency", "notification_capacity": 2},
                "notification latency requires --notification-capacity 1",
            ),
            (
                {"workload": "saturated", "queue_depth": 2, "notification_capacity": 8},
                "--notification-capacity must not exceed --queue-depth",
            ),
            ({"publisher_period_us": 0}, "--publisher-period-us must be positive"),
            ({"tensor_bytes": 0}, "--tensor-bytes must be between"),
            ({"tensor_device": -1}, "--tensor-device must be nonnegative"),
            ({"source_cpu": 2, "sink_cpu": 2}, "must not share one logical CPU"),
            (
                {
                    "controller_cpu": 2,
                    "channel_pump_cpu": 2,
                    "force_holomq_process_local_inline": True,
                    "force_process_local_pump_delivery": True,
                },
                "must not share one logical CPU",
            ),
            (
                {"operator_lanes": "shared", "source_cpu": 2, "controller_cpu": 4},
                "must match when the sole Operator lane shares the controller thread",
            ),
            ({"relay_cpu": 3}, "requires --scenario sample_forward"),
            (
                {"partition_cover": "automatic", "source_cpu": 2},
                "cannot bind generated partition names",
            ),
            (
                {"partition_cover": "automatic", "operator_lanes": "shared"},
                "requires --operator-lanes separate",
            ),
            (
                {"partition_cover": "automatic", "operator_lanes": "per_sink", "fanout": 2},
                "requires --operator-lanes separate",
            ),
            (
                {
                    "partition_cover": "automatic",
                    "force_holomq_process_local_inline": True,
                },
                "cannot force HoloMQ because compiler-derived ownership is authoritative",
            ),
            (
                {"partition_cover": "automatic", "runtime_phase_counters": True},
                "without explicit fan-out cannot use --runtime-phase-counters",
            ),
            (
                {"operator_lanes": "shared", "sink_cpu": 3},
                "--sink-cpu is not used with --operator-lanes shared",
            ),
            (
                {
                    "scenario": "sample_forward",
                    "operator_lanes": "shared",
                    "relay_cpu": 3,
                },
                "--relay-cpu is not used with --operator-lanes shared",
            ),
            (
                {"operator_lanes": "per_sink", "fanout": 1},
                "--operator-lanes per_sink requires supported fanout greater than one",
            ),
            (
                {
                    "scenario": "multi_tensor_loan",
                    "operator_lanes": "separate",
                    "fanout": 2,
                    "bulk_work_profile": "descriptor_only",
                    "payload_validation": "all",
                },
                (
                    "multi-Tensor fanout with --payload-validation all requires "
                    "--operator-lanes shared"
                ),
            ),
            (
                {
                    "scenario": "multi_tensor_staging",
                    "operator_lanes": "separate",
                    "fanout": 2,
                    "bulk_work_profile": "descriptor_only",
                    "payload_validation": "all",
                },
                (
                    "multi-Tensor fanout with --payload-validation all requires "
                    "--operator-lanes shared"
                ),
            ),
            (
                {"operator_lanes": "per_sink", "fanout": 2, "sink_cpu": 3},
                "--sink-cpu is not used with --operator-lanes per_sink",
            ),
            (
                {"operator_lanes": "separate", "fanout": 2, "sink_cpu": 3},
                (
                    "--sink-cpu with fanout greater than one requires "
                    "--operator-lanes per_sink and --sink-cpus"
                ),
            ),
            (
                {"operator_lanes": "separate", "fanout": 2, "sink_cpus": [3, 7]},
                "--sink-cpus requires --operator-lanes per_sink",
            ),
            (
                {"operator_lanes": "per_sink", "fanout": 4, "sink_cpus": [3, 7]},
                "--sink-cpus must provide exactly one CPU per fanout sink",
            ),
            (
                {"scenario": "source_only"},
                "--scenario source_only requires a notification workload",
            ),
            (
                {
                    "scenario": "source_only",
                    "workload": "notification_latency",
                    "sink_cpu": 3,
                },
                "--sink-cpu is not used by --scenario source_only",
            ),
            (
                {
                    "scenario": "source_only",
                    "workload": "notification_latency",
                    "channel_pump_cpu": 5,
                },
                "--channel-pump-cpu requires an active channel pump",
            ),
            (
                {
                    "scenario": "source_only",
                    "force_holomq_process_local_inline": True,
                },
                (
                    "--force-holomq-process-local-inline requires pod, signal, "
                    "sample_forward, schema, third_party_schema, tensor, image, "
                    "multi_tensor, multi_tensor_loan, or multi_tensor_staging"
                ),
            ),
            (
                {"bulk_work_profile": "descriptor_only"},
                "--bulk-work-profile descriptor_only requires --scenario tensor, image, "
                "multi_tensor, multi_tensor_loan, or multi_tensor_staging",
            ),
            (
                {"scenario": "image"},
                "--scenario image requires --bulk-work-profile descriptor_only",
            ),
            (
                {"scenario": "multi_tensor_loan"},
                "--scenario multi_tensor_loan requires --bulk-work-profile descriptor_only",
            ),
            (
                {"scenario": "multi_tensor_staging"},
                "--scenario multi_tensor_staging requires --bulk-work-profile descriptor_only",
            ),
            ({"notification_driver_cpu": 3}, "requires a notification workload"),
            ({"repetitions": 0}, "--repetitions must be positive"),
            ({"max_median_cv": -0.1}, "--max-median-cv must be nonnegative"),
            ({"source_state": ""}, "--source-state must be nonempty"),
            (
                {"source_snapshot_sha256": "not-a-sha256"},
                "--source-snapshot-sha256 must be a lowercase SHA-256 digest",
            ),
        ],
    )
    def test_invalid_options_are_rejected(self, updates: dict[str, object], message: str) -> None:
        with pytest.raises(ValueError, match=message):
            runner.validate_args(make_args(**updates))

    def test_saturated_workload_accepts_deep_queue_without_pacing(self) -> None:
        # Key setup: saturated mode admits a deep bounded queue and zero publication period.
        args = make_args(
            workload="saturated",
            queue_depth=8,
            notification_capacity=8,
            publisher_period_us=0,
        )
        runner.validate_args(args)

        command = runner.benchmark_command(
            args,
            Path("/tmp/runtime_graph_benchmark"),
            Path("/tmp/result.json"),
        )

        # Key assertion: command construction preserves depth and omits latency-only pacing.
        assert command[command.index("--workload") + 1] == "saturated"
        assert command[command.index("--notification-capacity") + 1] == "8"
        assert "--publisher-period-us" not in command

    @pytest.mark.parametrize(
        ("scenario", "tensor_storage"),
        [
            ("image", "pinned_host"),
            ("image", "device"),
            ("multi_tensor", "pinned_host"),
            ("multi_tensor", "device"),
            ("multi_tensor_loan", "pinned_host"),
            ("multi_tensor_loan", "device"),
            ("multi_tensor_staging", "pinned_host"),
            ("multi_tensor_staging", "device"),
        ],
    )
    def test_descriptor_schema_scenarios_accept_nonpageable_storage(
        self, scenario: str, tensor_storage: str
    ) -> None:
        # Key setup: select an authored one- or two-field descriptor schema with an explicit
        # non-pageable storage contract and non-default CUDA device.
        args = make_args(
            scenario=scenario,
            tensor_storage=tensor_storage,
            tensor_device=1,
            bulk_work_profile=(
                "descriptor_only"
                if scenario in {"image", "multi_tensor", *runner.MULTI_TENSOR_SCENARIOS}
                else "full_write"
            ),
        )

        runner.validate_args(args)
        command = runner.benchmark_command(
            args,
            Path("/tmp/runtime_graph_benchmark"),
            Path("/tmp/result.json"),
        )

        # Key assertion: storage and device reach the native benchmark without relabeling the
        # scenario or changing its established publication mode.
        assert command[command.index("--tensor-storage") + 1] == tensor_storage
        assert command[command.index("--tensor-device") + 1] == "1"
        assert command[command.index("--scenario") + 1] == scenario
        if scenario == "multi_tensor":
            assert command[command.index("--bulk-work-profile") + 1] == "descriptor_only"
        else:
            assert "--bulk-work-profile" not in command

    def test_saturated_workload_accepts_diagnostic_fanout_scaling(self) -> None:
        # Key setup: request the maximum diagnostic POD fanout without changing Runtime eligibility.
        args = make_args(
            workload="saturated",
            scenario="pod",
            operator_lanes="shared",
            fanout=32,
        )
        runner.validate_args(args)
        command = runner.benchmark_command(
            args,
            Path("/tmp/runtime_graph_benchmark"),
            Path("/tmp/result.json"),
        )

        # Key assertion: the runner forwards the full benchmark-only scaling width.
        assert command[command.index("--fanout") + 1] == "32"

    def test_saturated_fanout_accepts_explicit_per_sink_queue_depths(self) -> None:
        # Key setup: author two shared-lane POD subscribers at depths four and two while retaining
        # the service maximum and publication-pressure capacity at four.
        args = make_args(
            workload="saturated",
            scenario="pod",
            operator_lanes="shared",
            queue_depth=4,
            sink_queue_depths=[4, 2],
            fanout=2,
            notification_capacity=4,
        )
        runner.validate_args(args)

        command = runner.benchmark_command(
            args,
            Path("/tmp/runtime_graph_benchmark"),
            Path("/tmp/result.json"),
        )

        # Key assertion: the native command receives ordered subscriber depths and the expected
        # result geometry preserves that exact topology rather than flattening it to the maximum.
        assert command[command.index("--sink-queue-depths") + 1] == "4,2"
        assert runner.subscriber_queue_depths(args) == [4, 2]

    @pytest.mark.parametrize("topology", ["multi_output", "multi_input"])
    def test_multi_output_topologies_accept_only_uniform_explicit_depths(
        self, topology: str
    ) -> None:
        # Key setup: author either paired-output topology with the same explicit depth on both
        # independently published services.
        args = make_args(
            workload="saturated",
            scenario="pod",
            operator_lanes="shared",
            queue_depth=2,
            sink_queue_depths=[2, 2],
            fanout=2,
            notification_capacity=2,
            force_holomq_process_local_inline=True,
            **{topology: True},
        )
        runner.validate_args(args)

        # Key assertion: the uniform topology reaches the native parser with both depths intact.
        command = runner.benchmark_command(
            args, Path("/tmp/runtime_graph_benchmark"), Path("/tmp/result.json")
        )
        assert command[command.index("--sink-queue-depths") + 1] == "2,2"

    def test_drop_oldest_fanout_accepts_fixed_publication_offer_rate(self) -> None:
        # Key setup: pace a shared-lane two-subscriber drop-oldest service from the Runtime clock
        # instead of advancing notification credit behind source callbacks.
        args = make_args(
            workload="saturated",
            scenario="pod",
            operator_lanes="shared",
            queue_policy="drop_oldest",
            queue_depth=2,
            fanout=2,
            notification_capacity=1,
            offered_rate_hz=100_000,
            offered_burst_size=4,
        )
        runner.validate_args(args)

        command = runner.benchmark_command(
            args,
            Path("/tmp/runtime_graph_benchmark"),
            Path("/tmp/result.json"),
        )

        # Key assertion: command and result identity retain the configured rate so eventless and
        # forced-wake arms cannot silently use source-completion-limited pressure.
        assert command[command.index("--offered-rate-hz") + 1] == "100000"
        assert command[command.index("--offered-burst-size") + 1] == "4"
        assert runner.benchmark_name(
            "pod",
            "saturated",
            fanout=2,
            queue_policy="drop_oldest",
            offered_rate_hz=100_000,
            offered_burst_size=4,
        ).endswith("_drop_oldest_offered_100000hz_burst4_saturated")

    def test_offered_bursts_require_every_fired_tick_and_bound_lost_ticks(self) -> None:
        # Key setup: a 20 kHz, four-publication-burst row whose source woke too late for 150 clock
        # ticks, as an IGX Thor run did. Its accepted rate is 3.5% below the clock rate.
        args = make_args(
            workload="saturated",
            scenario="pod",
            operator_lanes="shared",
            queue_policy="drop_oldest",
            queue_depth=2,
            fanout=2,
            offered_rate_hz=20_000,
            offered_burst_size=4,
            samples=16_384,
        )
        timing = {
            "accepted_publication_offer_rate_per_second": 19_296.9,
            "publication_offer_start_ns": 1_000,
            "publication_offer_end_ns": 848_867_913,
            "maximum_publication_offer_lateness_ns": 30_000_000,
            "offered_bursts": 4_096,
            "lost_offer_ticks": 150,
        }

        # Key assertion: every tick that fired carried an accepted burst, so the row passes even
        # though late wakes lowered the rate; a missing burst or too many lost ticks still fail.
        runner.validate_offered_bursts(timing, args, "result.json")
        runner.validate_offered_bursts({**timing, "lost_offer_ticks": 1_365}, args, "result.json")
        with pytest.raises(RuntimeError, match="incomplete fixed publication-offer burst count"):
            runner.validate_offered_bursts({**timing, "offered_bursts": 4_095}, args, "result.json")
        with pytest.raises(RuntimeError, match="lost too many fixed publication-offer clock ticks"):
            runner.validate_offered_bursts(
                {**timing, "lost_offer_ticks": 1_366}, args, "result.json"
            )
        for invalid in (
            {"lost_offer_ticks": None},
            {"offered_bursts": True},
            {"lost_offer_ticks": -1},
        ):
            with pytest.raises(RuntimeError, match="invalid fixed publication-offer schedule"):
                runner.validate_offered_bursts({**timing, **invalid}, args, "result.json")
        unpaced = make_args()
        runner.validate_offered_bursts(dict.fromkeys(timing), unpaced, "result.json")
        with pytest.raises(RuntimeError, match="unexpected fixed publication-offer timing"):
            runner.validate_offered_bursts(timing, unpaced, "result.json")

    def test_drop_oldest_pod_fanout_is_direct_unless_holomq_is_forced(self) -> None:
        # Key setup: change only the compiler-private delivery controls on one clocked
        # drop-oldest POD fanout row.
        row = {
            "workload": "saturated",
            "scenario": "pod",
            "operator_lanes": "shared",
            "queue_policy": "drop_oldest",
            "queue_depth": 2,
            "fanout": 2,
            "offered_rate_hz": 20_000,
            "offered_burst_size": 4,
        }
        direct = make_args(**row)
        forced = make_args(**row, force_holomq_fanout=True)
        wake_control = make_args(**row, force_holomq_event_wake=True)
        for args in (direct, forced, wake_control):
            runner.validate_args(args)
        command = runner.benchmark_command(
            forced, Path("/tmp/runtime_graph_benchmark"), Path("/tmp/result.json")
        )

        # Key assertion: compile() delivers the drop-oldest POD family by direct retention without
        # an event wake or continuation receipts. Both HoloMQ arms keep the service, and only the
        # forced-fanout arm passes and names the seam.
        assert runner.direct_fanout_family(direct)
        assert runner.process_local_delivery(direct) == "direct_retained"
        assert not runner.event_wake_required(direct)
        assert runner.continuation_receipt_ineligible(direct)
        assert runner.process_local_delivery(forced) == "holomq"
        assert not runner.event_wake_required(forced)
        assert runner.process_local_delivery(wake_control) == "holomq"
        assert runner.event_wake_required(wake_control)
        assert "--force-holomq-fanout" in command
        assert runner.benchmark_name(
            "pod",
            "saturated",
            fanout=2,
            queue_policy="drop_oldest",
            offered_rate_hz=20_000,
            offered_burst_size=4,
            force_holomq_fanout=True,
        ).endswith("_holomq_fanout_drop_oldest_offered_20000hz_burst4_saturated")

    def test_blocking_and_mixed_policy_fanout_rows_are_direct_families(self):
        row = {
            "workload": "latency",
            "scenario": "pod",
            "operator_lanes": "shared",
            "queue_policy": "blocking",
            "fanout": 2,
        }
        blocking = make_args(**row)
        mixed = make_args(**row, mixed_policy_fanout=True)
        forced = make_args(**row, force_holomq_fanout=True)

        # Key assertion: compile() delivers blocking and mixed-policy POD families by direct
        # retention (CHAN-167); the forced-fanout control keeps HoloMQ.
        assert runner.direct_fanout_family(blocking)
        assert runner.process_local_delivery(blocking) == "direct_retained"
        assert runner.direct_fanout_family(mixed)
        assert runner.process_local_delivery(forced) == "holomq"

    @pytest.mark.parametrize(
        ("updates", "message"),
        [
            (
                {"fanout": 1, "offered_rate_hz": 0, "offered_burst_size": 1},
                "--force-holomq-fanout requires",
            ),
            ({"force_holomq_process_local_inline": True}, "--force-holomq-fanout requires"),
        ],
    )
    def test_force_holomq_fanout_requires_a_pod_or_schema_fanout_row(
        self, updates: dict[str, object], message: str
    ) -> None:
        args = make_args(
            workload="saturated",
            scenario="pod",
            operator_lanes="shared",
            queue_policy="drop_oldest",
            queue_depth=2,
            fanout=2,
            offered_rate_hz=20_000,
            offered_burst_size=4,
            force_holomq_fanout=True,
        )
        for name, value in updates.items():
            setattr(args, name, value)

        with pytest.raises(ValueError, match=message):
            runner.validate_args(args)

    def test_shared_operator_lane_accepts_paced_latency(self) -> None:
        args = make_args(operator_lanes="shared")
        runner.validate_args(args)

        command = runner.benchmark_command(
            args,
            Path("/tmp/runtime_graph_benchmark"),
            Path("/tmp/result.json"),
        )

        # Key assertion: the shared lane is preserved for the paced latency workload.
        assert command[command.index("--operator-lanes") + 1] == "shared"
        assert command[command.index("--workload") + 1] == "latency"
        assert command[command.index("--publisher-period-us") + 1] == "1000"

    @pytest.mark.parametrize("callback_work_us", runner.SUPPORTED_CALLBACK_WORK_US)
    def test_callback_work_sweep_is_forwarded_explicitly(self, callback_work_us: int) -> None:
        args = make_args(callback_work_us=callback_work_us)
        runner.validate_args(args)

        command = runner.benchmark_command(
            args, Path("/tmp/runtime_graph_benchmark"), Path("/tmp/result.json")
        )

        assert command[command.index("--callback-work-us") + 1] == str(callback_work_us)

    def test_partition_environment_is_explicit_and_clears_inherited_selection(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Key setup: simulate an inherited selector that could silently relabel benchmark results.
        monkeypatch.setenv(runner.PARTITION_ENVIRONMENT, "source_chain/v1")

        # Key assertion: benchmark-authored partitions always clear inherited planner selection.
        assert runner.PARTITION_ENVIRONMENT not in runner.benchmark_environment(make_args())

    def test_saturated_command_carries_every_support_thread_cpu(self) -> None:
        # Key setup: give every Runtime-owned support thread a distinct requested CPU.
        args = make_args(
            workload="saturated",
            source_cpu=2,
            sink_cpu=3,
            controller_cpu=4,
            channel_pump_cpu=5,
            notification_driver_cpu=6,
            force_holomq_process_local_inline=True,
            force_process_local_pump_delivery=True,
        )

        runner.validate_args(args)
        command = runner.benchmark_command(
            args, Path("/tmp/runtime_graph_benchmark"), Path("/tmp/result.json")
        )

        # Key assertion: all three support-thread placements reach the benchmark executable.
        assert command[command.index("--controller-cpu") + 1] == "4"
        assert command[command.index("--channel-pump-cpu") + 1] == "5"
        assert command[command.index("--notification-driver-cpu") + 1] == "6"
        assert "--force-holomq-process-local-inline" in command

    def test_runtime_phase_counter_mode_is_explicit(self) -> None:
        # Key setup: opt into source-tree Runtime phase cardinality evidence.
        args = make_args(runtime_phase_counters=True)

        command = runner.benchmark_command(
            args, Path("/tmp/runtime_graph_benchmark"), Path("/tmp/result.json")
        )

        # Key assertion: ordinary commands omit the instrumentation; requested runs preserve it.
        assert "--runtime-phase-counters" in command
        assert "--runtime-phase-counters" not in runner.benchmark_command(
            make_args(), Path("/tmp/runtime_graph_benchmark"), Path("/tmp/result.json")
        )

    def test_image_sink_borrow_is_explicit_and_image_only(self) -> None:
        # Key setup: the image sink may borrow each sample instead of receiving an owning copy.
        borrowed = make_args(
            scenario="image", bulk_work_profile="descriptor_only", borrow_image_input=True
        )
        runner.validate_args(borrowed)

        command = runner.benchmark_command(
            borrowed, Path("/tmp/runtime_graph_benchmark"), Path("/tmp/result.json")
        )

        # Key assertion: only requested rows borrow, and every image row names its accessor.
        assert "--borrow-image-input" in command
        assert "--borrow-image-input" not in runner.benchmark_command(
            make_args(scenario="image"),
            Path("/tmp/runtime_graph_benchmark"),
            Path("/tmp/result.json"),
        )
        assert runner.image_input_accessor(borrowed) == "selected_sample"
        assert runner.image_input_accessor(make_args(scenario="image")) == "receive"
        assert runner.image_input_accessor(make_args()) is None
        with pytest.raises(ValueError, match="borrow-image-input requires --scenario image"):
            runner.validate_args(make_args(borrow_image_input=True))

    def test_pod_sink_pressure_gate_is_exact_and_opt_in(self) -> None:
        # Key setup: select only the counter-enabled, separate-lane, forced-HoloMQ depth-one POD
        # diagnostic whose independent notification gate leaves source activations runnable.
        args = make_args(
            workload="saturated",
            queue_policy="drop_oldest",
            runtime_phase_counters=True,
            force_holomq_process_local_inline=True,
            test_gate_pod_sink_for_pressure=True,
        )

        runner.validate_args(args)
        command = runner.benchmark_command(
            args, Path("/tmp/runtime_graph_benchmark"), Path("/tmp/result.json")
        )

        # Key assertion: the test seam reaches only its exact diagnostic command and remains
        # absent from the ordinary counters-off benchmark command used for A/C measurement.
        assert "--test-gate-pod-sink-for-pressure" in command
        assert "--runtime-phase-counters" in command
        assert "--test-gate-pod-sink-for-pressure" not in runner.benchmark_command(
            make_args(
                workload="saturated",
                queue_policy="drop_oldest",
                force_holomq_process_local_inline=True,
            ),
            Path("/tmp/runtime_graph_benchmark"),
            Path("/tmp/result.json"),
        )

    def test_pod_sink_pressure_gate_requires_ordered_bounded_acknowledgements(self) -> None:
        # Key setup: encode the source/witness/first-delivery barriers and bounded one-at-a-time
        # gate posts used to drain a final survivor after 21 finite source publications.
        args = make_args(
            workload="saturated",
            queue_policy="drop_oldest",
            runtime_phase_counters=True,
            force_holomq_process_local_inline=True,
            test_gate_pod_sink_for_pressure=True,
        )
        counts = {"publications": 21}
        pressure_gate = {
            "first_readiness_published": 1,
            "first_witness_retired": 1,
            "first_delivery_publications": 1,
            "source_completion_publications": 21,
            "source_retirement_activations_completed": 22,
            "notification_attempts": 3,
            "notifications_accepted": 3,
            "notification_backpressure": 0,
            "notification_not_ready": 0,
        }

        runner.validate_pod_sink_pressure_gate(
            pressure_gate, counts, args, Path("/tmp/result.json")
        )

        for name in (
            "first_readiness_published",
            "first_witness_retired",
            "source_completion_publications",
            "source_retirement_activations_completed",
            "notifications_accepted",
        ):
            invalid = dict(pressure_gate)
            invalid[name] = 0
            with pytest.raises(RuntimeError, match="pressure-gate|pressure gate"):
                runner.validate_pod_sink_pressure_gate(
                    invalid, counts, args, Path("/tmp/result.json")
                )

        with pytest.raises(RuntimeError, match="unexpectedly enabled"):
            runner.validate_pod_sink_pressure_gate(
                pressure_gate, counts, make_args(), Path("/tmp/result.json")
            )

    @pytest.mark.parametrize(
        "updates",
        [
            {"partition_cover": "hybrid"},
            {"runtime_phase_counters": False},
            {"workload": "latency"},
            {"scenario": "schema"},
            {"operator_lanes": "shared"},
            {"queue_policy": "blocking"},
            {"queue_depth": 2},
            {"sink_queue_depths": [1]},
            {"fanout": 2},
            {"multi_output": True},
            {"multi_input": True},
            {"mixed_policy_fanout": True},
            {"notification_capacity": 2},
            {"offered_rate_hz": 100_000},
            {"offered_burst_size": 2},
            {"callback_work_us": 1},
            {"force_holomq_process_local_inline": False},
            {"force_holomq_event_wake": True},
            {"force_process_local_channel_pump": True},
        ],
    )
    def test_pod_sink_pressure_gate_rejects_nearby_rows(self, updates: dict[str, object]) -> None:
        # Key setup: start from the sole supported diagnostic identity, then drift one field.
        values: dict[str, object] = {
            "workload": "saturated",
            "queue_policy": "drop_oldest",
            "runtime_phase_counters": True,
            "force_holomq_process_local_inline": True,
            "test_gate_pod_sink_for_pressure": True,
        }
        values.update(updates)

        # Key assertion: the seam cannot leak into ordinary performance or nearby graph rows.
        with pytest.raises(ValueError, match="counter-enabled saturated"):
            runner.validate_args(make_args(**values))

    def test_closed_island_composite_control_is_explicit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Key setup: request only the closed-island composite A/B control.
        control = make_args(disable_closed_island_composite=True)

        command = runner.benchmark_command(
            control, Path("/tmp/runtime_graph_benchmark"), Path("/tmp/result.json")
        )

        # Key assertion: the control is independently forwarded and remains absent by default.
        assert "--disable-closed-island-composite" in command
        assert "--disable-closed-island-composite" not in runner.benchmark_command(
            make_args(), Path("/tmp/runtime_graph_benchmark"), Path("/tmp/result.json")
        )

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_runtime_graph_benchmark.py",
                "/tmp/runtime_graph_benchmark",
                "--disable-closed-island-composite",
            ],
        )
        parsed = runner.parse_args()
        assert parsed.disable_closed_island_composite
        assert not parsed.disable_direct_continuation_elision

    def test_direct_continuation_elision_control_is_explicit_and_preserves_delivery(self) -> None:
        # Key setup: disable only the continuation optimization for the exact direct-retained
        # ordinary-dispatch control; do not select the separate HoloMQ fallback mechanism.
        control = make_args(disable_direct_continuation_elision=True)

        command = runner.benchmark_command(
            control, Path("/tmp/runtime_graph_benchmark"), Path("/tmp/result.json")
        )

        # Key assertion: the private control is independently forwarded and remains absent from
        # ordinary runs, while process-local delivery is still direct retained.
        assert "--disable-direct-continuation-elision" in command
        assert "--force-holomq-process-local-inline" not in command
        assert runner.process_local_delivery(control) == "direct_retained"
        assert "--disable-direct-continuation-elision" not in runner.benchmark_command(
            make_args(), Path("/tmp/runtime_graph_benchmark"), Path("/tmp/result.json")
        )

    def test_direct_continuation_elision_control_parses_explicitly(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Key setup: opt into the private A/B control through the public runner command line.
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_runtime_graph_benchmark.py",
                "/tmp/runtime_graph_benchmark",
                "--disable-direct-continuation-elision",
            ],
        )

        # Key assertion: parsing changes only the requested private control.
        args = runner.parse_args()
        assert args.disable_direct_continuation_elision
        assert not args.force_holomq_process_local_inline

    def test_holomq_event_wake_control_composes_with_ordinary_dispatch(self) -> None:
        # Key setup: force the one-deep shared POD edge through HoloMQ while independently
        # retaining event wake and selecting the ordinary-dispatch control.
        args = make_args(
            scenario="pod",
            operator_lanes="shared",
            force_holomq_process_local_inline=True,
            force_holomq_event_wake=True,
            disable_direct_continuation_elision=True,
        )

        runner.validate_args(args)
        command = runner.benchmark_command(
            args, Path("/tmp/runtime_graph_benchmark"), Path("/tmp/result.json")
        )

        # Key assertion: all three controls remain explicit, event wake is required, and the
        # result identity prevents the forced-wake row from merging with the optimized row.
        assert "--force-holomq-process-local-inline" in command
        assert "--force-holomq-event-wake" in command
        assert "--disable-direct-continuation-elision" in command
        assert runner.event_wake_required(args) is True
        assert runner.benchmark_name("pod", "latency", True, force_holomq_event_wake=True) == (
            "runtime_graph_pod_1to1_holomq_fallback_event_wake_control"
        )

    def test_shared_eventless_holomq_does_not_require_channel_pump(self) -> None:
        # Key setup: compare a shared-lane eventless POD HoloMQ row with its forced-event-wake
        # control while preserving topology, transport, and queue policy.
        eventless = make_args(
            operator_lanes="shared",
            force_holomq_process_local_inline=True,
        )
        wake_control = make_args(
            operator_lanes="shared",
            force_holomq_process_local_inline=True,
            force_holomq_event_wake=True,
            channel_pump_cpu=5,
        )

        runner.validate_args(eventless)
        runner.validate_args(wake_control)

        # Key assertion: both rows use HoloMQ, but only the event-wake control requires and
        # reports a Channel-pump support thread.
        assert runner.uses_holomq(eventless)
        assert runner.process_local_delivery(eventless) == "holomq"
        assert not runner.event_wake_required(eventless)
        assert not runner.uses_channel_pump(eventless)
        assert all(
            item["role"] != "channel_pump" for item in runner.collect_cpu_topology(eventless)
        )
        assert runner.uses_holomq(wake_control)
        assert runner.event_wake_required(wake_control)
        assert runner.uses_channel_pump(wake_control)
        assert "channel_pump" in {
            item["role"] for item in runner.collect_cpu_topology(wake_control)
        }

    def test_eventless_holomq_forced_pump_control_changes_only_cold_thread_selection(self) -> None:
        # Key setup: pair one shared-lane eventless POD HoloMQ row with the private forced-pump
        # control while preserving its topology, transport, queue, and wake state.
        baseline = make_args(
            scenario="pod",
            operator_lanes="shared",
            force_holomq_process_local_inline=True,
        )
        control = make_args(
            scenario="pod",
            operator_lanes="shared",
            force_holomq_process_local_inline=True,
            force_process_local_channel_pump=True,
            channel_pump_cpu=5,
        )

        runner.validate_args(baseline)
        runner.validate_args(control)
        command = runner.benchmark_command(
            control, Path("/tmp/runtime_graph_benchmark"), Path("/tmp/result.json")
        )

        # Key assertion: both rows remain eventless HoloMQ; only the control owns and names a pump.
        assert runner.uses_holomq(baseline)
        assert not runner.event_wake_required(baseline)
        assert not runner.uses_channel_pump(baseline)
        assert runner.uses_holomq(control)
        assert not runner.event_wake_required(control)
        assert runner.uses_channel_pump(control)
        assert "--force-process-local-channel-pump" in command
        assert (
            runner.benchmark_name("pod", "latency", True, force_process_local_channel_pump=True)
            == "runtime_graph_pod_1to1_holomq_fallback_channel_pump_control"
        )

    @pytest.mark.parametrize("force_event_wake", [False, True])
    def test_multi_output_event_wake_pair_preserves_exact_topology(
        self, force_event_wake: bool
    ) -> None:
        # Key setup: author two POD outputs with one subscriber each on one shared operator lane.
        args = make_args(
            scenario="pod",
            operator_lanes="shared",
            fanout=2,
            multi_output=True,
            force_holomq_process_local_inline=True,
            force_holomq_event_wake=force_event_wake,
        )

        runner.validate_args(args)
        command = runner.benchmark_command(
            args, Path("/tmp/runtime_graph_benchmark"), Path("/tmp/result.json")
        )

        # Key assertion: the paired rows retain identical 2x1 geometry and differ only in wake.
        assert "--multi-output" in command
        assert command[command.index("--fanout") + 1] == "2"
        assert ("--force-holomq-event-wake" in command) is force_event_wake
        assert runner.benchmark_name(
            "pod",
            "latency",
            True,
            fanout=2,
            force_holomq_event_wake=force_event_wake,
            multi_output=True,
        ) == (
            "runtime_graph_pod_2x1_holomq_fallback"
            + ("_event_wake_control" if force_event_wake else "")
        )

    @pytest.mark.parametrize("force_event_wake", [False, True])
    def test_multi_input_event_wake_pair_preserves_exact_topology(
        self, force_event_wake: bool
    ) -> None:
        # Key setup: author two POD outputs feeding two paired inputs on one shared-lane sink.
        args = make_args(
            scenario="pod",
            operator_lanes="shared",
            fanout=2,
            multi_input=True,
            force_holomq_process_local_inline=True,
            force_holomq_event_wake=force_event_wake,
        )

        runner.validate_args(args)
        command = runner.benchmark_command(
            args, Path("/tmp/runtime_graph_benchmark"), Path("/tmp/result.json")
        )

        # Key assertion: the paired rows retain two outputs, one two-input sink, and differ only
        # in the exact compiler-selected wake control.
        assert "--multi-input" in command
        assert command[command.index("--fanout") + 1] == "2"
        assert ("--force-holomq-event-wake" in command) is force_event_wake
        assert runner.source_output_count(args) == 2
        assert runner.sink_operator_count(args) == 1
        assert runner.inputs_per_sink(args) == 2
        assert runner.benchmark_name(
            "pod",
            "latency",
            True,
            fanout=2,
            force_holomq_event_wake=force_event_wake,
            multi_input=True,
        ) == (
            "runtime_graph_pod_2to1_holomq_fallback"
            + ("_event_wake_control" if force_event_wake else "")
        )

    @pytest.mark.parametrize("force_event_wake", [False, True])
    def test_mixed_policy_event_wake_pair_preserves_split_service_geometry(
        self, force_event_wake: bool
    ) -> None:
        # Key setup: author one POD output with blocking and drop-oldest subscribers on the same
        # shared lane, optionally retaining event wake through the existing compiler control.
        args = make_args(
            workload="saturated",
            scenario="pod",
            operator_lanes="shared",
            fanout=2,
            mixed_policy_fanout=True,
            force_holomq_event_wake=force_event_wake,
        )

        runner.validate_args(args)
        command = runner.benchmark_command(
            args, Path("/tmp/runtime_graph_benchmark"), Path("/tmp/result.json")
        )

        # Key assertion: the pair retains one source and two policy-specific subscribers; only
        # the control enables the pump-backed event wake and changes benchmark identity.
        assert "--mixed-policy-fanout" in command
        assert runner.subscriber_queue_policies(args) == ["blocking", "drop_oldest"]
        assert runner.has_lossy_subscriber(args)
        assert runner.event_wake_required(args) is force_event_wake
        assert runner.uses_channel_pump(args) is force_event_wake
        assert runner.benchmark_name(
            "pod",
            "saturated",
            fanout=2,
            force_holomq_event_wake=force_event_wake,
            mixed_policy_fanout=True,
        ) == (
            "runtime_graph_pod_1to2"
            + ("_event_wake_control" if force_event_wake else "")
            + "_mixed_policy_saturated"
        )

    @pytest.mark.parametrize("fanout", [1, 2])
    def test_holomq_event_wake_control_accepts_deeper_queues(self, fanout: int) -> None:
        # Key setup: retain event wake on a shared-lane depth-eight POD service for both exact
        # 1:1 forced-HoloMQ delivery and native two-subscriber fanout.
        args = make_args(
            workload="saturated",
            scenario="pod",
            operator_lanes="shared",
            queue_depth=8,
            notification_capacity=8,
            fanout=fanout,
            force_holomq_process_local_inline=fanout == 1,
            force_holomq_event_wake=True,
        )

        runner.validate_args(args)
        command = runner.benchmark_command(
            args, Path("/tmp/runtime_graph_benchmark"), Path("/tmp/result.json")
        )

        # Key assertion: queue capacity does not change control eligibility, active fanout, or
        # the explicit forced-wake provenance passed to the native benchmark.
        assert "--force-holomq-event-wake" in command
        assert command[command.index("--queue-depth") + 1] == "8"
        assert command[command.index("--fanout") + 1] == str(fanout)
        assert runner.event_wake_required(args) is True

    @pytest.mark.parametrize("scenario", sorted(runner.TENSOR_BACKED_SCENARIOS))
    def test_tensor_holomq_event_wake_control_is_explicit(self, scenario: str) -> None:
        # Key setup: retain event wake on an otherwise eventless exact 1:1 Tensor HoloMQ row.
        args = make_args(
            scenario=scenario,
            operator_lanes="shared",
            force_holomq_process_local_inline=True,
            force_holomq_event_wake=True,
            bulk_work_profile=(
                "descriptor_only" if scenario in runner.DESCRIPTOR_ONLY_SCENARIOS else "full_write"
            ),
        )

        runner.validate_args(args)
        command = runner.benchmark_command(
            args, Path("/tmp/runtime_graph_benchmark"), Path("/tmp/result.json")
        )

        # Key assertion: the command and result identity preserve the forced-wake provenance.
        assert "--force-holomq-event-wake" in command
        assert runner.event_wake_required(args) is True
        assert runner.benchmark_name(
            scenario, "latency", True, force_holomq_event_wake=True
        ).endswith("_holomq_fallback_event_wake_control")

    def test_direct_pod_omits_channel_pump_and_names_holomq_fallback(self) -> None:
        # Key setup: compare the compiler-selected direct POD path with its private control mode.
        direct = make_args()
        fallback = make_args(force_holomq_process_local_inline=True)

        runner.validate_args(direct)
        runner.validate_args(fallback)

        # Key assertion: direct delivery has no pump role; the fallback is explicit in command and
        # result identity so measurements from the two mechanisms cannot be combined accidentally.
        assert all(item["role"] != "channel_pump" for item in runner.collect_cpu_topology(direct))
        fallback_command = runner.benchmark_command(
            fallback, Path("/tmp/runtime_graph_benchmark"), Path("/tmp/result.json")
        )
        assert "--force-holomq-process-local-inline" in fallback_command
        assert runner.benchmark_name("pod", "latency", True) == (
            "runtime_graph_pod_1to1_holomq_fallback"
        )

    def test_linear_chain_depth_is_explicit_in_command_name_and_topology(self) -> None:
        # Key setup: four forwarding Operators share the existing relay role between one source
        # and one sink, isolating chain depth from the number of execution lanes.
        args = make_args(
            scenario="sample_forward",
            relay_count=4,
        )

        runner.validate_args(args)
        command = runner.benchmark_command(
            args, Path("/tmp/runtime_graph_benchmark"), Path("/tmp/result.json")
        )
        topology = runner.collect_cpu_topology(args)

        # Key assertion: the native command and result identity retain the exact chain depth,
        # while separate placement still has one source, relay, and sink role.
        assert command[command.index("--relay-count") + 1] == "4"
        assert runner.benchmark_name("sample_forward", "latency", False, 4) == (
            "runtime_graph_pod_chain_1to1_relays_4"
        )
        assert [item["role"] for item in topology] == ["source", "relay", "sink", "controller"]

    def test_pod_fanout_is_direct_and_records_each_unpinned_sink_lane(self) -> None:
        # Key setup: one source publishes each logical POD sample to four process-local sinks.
        args = make_args(scenario="pod", fanout=4)

        runner.validate_args(args)
        command = runner.benchmark_command(
            args, Path("/tmp/runtime_graph_benchmark"), Path("/tmp/result.json")
        )
        topology = runner.collect_cpu_topology(args)

        # Key assertion: fanout is explicit, is delivered by direct retention without a transport
        # pump, and records every independently compiled sink lane without inventing CPU placement.
        assert command[command.index("--fanout") + 1] == "4"
        assert runner.process_local_delivery(args) == "direct_retained"
        assert runner.uses_channel_pump(args) is False
        assert runner.benchmark_name("pod", "latency", fanout=4) == "runtime_graph_pod_1to4"
        assert [item["role"] for item in topology] == [
            "source",
            "sink_1",
            "sink_2",
            "sink_3",
            "sink_4",
            "controller",
        ]
        assert all(not item["pinned"] for item in topology if item["role"].startswith("sink_"))

    def test_pod_fanout_can_assign_one_lane_per_sink(self) -> None:
        # Key setup: four fanout subscribers receive distinct role names and physical-core pins.
        args = make_args(
            workload="notification_latency",
            scenario="pod",
            operator_lanes="per_sink",
            fanout=4,
            source_cpu=2,
            sink_cpus=[8, 10, 12, 14],
            controller_cpu=4,
            notification_driver_cpu=6,
        )

        runner.validate_args(args)
        command = runner.benchmark_command(
            args, Path("/tmp/runtime_graph_benchmark"), Path("/tmp/result.json")
        )
        topology = runner.collect_cpu_topology(args)

        # Key assertion: every sink pin reaches the native process and remains separately labeled.
        assert command[command.index("--sink-cpus") + 1] == "8,10,12,14"
        assert [item["role"] for item in topology] == [
            "source",
            "sink_1",
            "sink_2",
            "sink_3",
            "sink_4",
            "controller",
            "notification_driver",
        ]

    def test_direct_tensor_omits_channel_pump_and_admits_holomq_fallback(self) -> None:
        # Key setup: compare compiler-retained Tensor delivery with the same graph forced through
        # its process-local HoloMQ sidecar endpoint.
        direct = make_args(scenario="tensor")
        fallback = make_args(
            scenario="tensor",
            force_holomq_process_local_inline=True,
            channel_pump_cpu=5,
        )

        runner.validate_args(direct)
        runner.validate_args(fallback)

        # Key assertion: only the forced transport control owns a Channel pump, and both result
        # identities preserve which mechanism was measured.
        assert runner.uses_channel_pump(direct) is False
        assert runner.process_local_delivery(direct) == "direct_retained"
        assert runner.uses_channel_pump(fallback) is True
        assert runner.process_local_delivery(fallback) == "holomq"
        assert runner.benchmark_name("tensor", "latency", True) == (
            "runtime_graph_tensor_1to1_holomq_fallback"
        )

    def test_lossy_direct_retention_admits_matched_holomq_control(self) -> None:
        # Key setup: keep graph topology and queue policy identical while selecting only the
        # compiler-private forced-HoloMQ control for the second row.
        queue_policy = "drop_oldest"
        direct = make_args(queue_policy=queue_policy)
        fallback = make_args(queue_policy=queue_policy, force_holomq_process_local_inline=True)

        runner.validate_args(direct)
        runner.validate_args(fallback)
        direct_command = runner.benchmark_command(
            direct, Path("/tmp/runtime_graph_benchmark"), Path("/tmp/direct.json")
        )

        # Key assertion: drop-oldest is direct-eligible, preserves the authored policy on the
        # native command, and names the A/B rows distinctly.
        assert runner.process_local_delivery(direct) == "direct_retained"
        assert runner.process_local_delivery(fallback) == "holomq"
        assert direct_command[direct_command.index("--queue-policy") + 1] == queue_policy
        assert (
            runner.benchmark_name("pod", "latency", queue_policy=queue_policy)
            == f"runtime_graph_pod_1to1_{queue_policy}"
        )
        assert (
            runner.benchmark_name("pod", "saturated", True, queue_policy=queue_policy)
            == f"runtime_graph_pod_1to1_holomq_fallback_{queue_policy}_saturated"
        )

    @pytest.mark.parametrize(
        "scenario",
        [
            "schema",
            "third_party_schema",
            "tensor",
            "image",
            "multi_tensor_loan",
            "multi_tensor_staging",
        ],
    )
    def test_drop_oldest_payloads_admit_matched_holomq_control(self, scenario: str) -> None:
        # Key setup: apply the same depth-two drop-oldest policy to each direct-retained payload
        # family and change only the compiler-private delivery control.
        direct = make_args(
            scenario=scenario,
            queue_policy="drop_oldest",
            queue_depth=2,
            tensor_bytes=64,
            bulk_work_profile=(
                "descriptor_only" if scenario in runner.DESCRIPTOR_ONLY_SCENARIOS else "full_write"
            ),
        )
        fallback = make_args(
            scenario=scenario,
            queue_policy="drop_oldest",
            queue_depth=2,
            tensor_bytes=64,
            bulk_work_profile=(
                "descriptor_only" if scenario in runner.DESCRIPTOR_ONLY_SCENARIOS else "full_write"
            ),
            force_holomq_process_local_inline=True,
            channel_pump_cpu=5 if scenario in runner.TENSOR_BACKED_SCENARIOS else -1,
        )

        runner.validate_args(direct)
        runner.validate_args(fallback)
        direct_command = runner.benchmark_command(
            direct, Path("/tmp/runtime_graph_benchmark"), Path("/tmp/direct.json")
        )

        # Key assertion: generated schema, raw Tensor, and descriptor-backed Image retain the
        # authored lossy policy. Only the Tensor-backed matched control requires a Channel pump;
        # compile() delivers the schema control inside emit().
        assert runner.process_local_delivery(direct) == "direct_retained"
        assert runner.process_local_delivery(fallback) == "holomq"
        assert runner.uses_channel_pump(fallback) is (scenario in runner.TENSOR_BACKED_SCENARIOS)
        assert direct_command[direct_command.index("--queue-policy") + 1] == "drop_oldest"
        assert (
            runner.benchmark_name(scenario, "latency", queue_policy="drop_oldest")
            == f"{runner.BENCHMARK_NAMES[scenario]}_drop_oldest"
        )
        assert (
            runner.benchmark_name(
                scenario,
                "saturated",
                True,
                queue_policy="drop_oldest",
            )
            == f"{runner.BENCHMARK_NAMES[scenario]}_holomq_fallback_drop_oldest_saturated"
        )

    @pytest.mark.parametrize("scenario", ["multi_tensor_loan", "multi_tensor_staging"])
    def test_direct_multi_tensor_admits_matched_holomq_control(self, scenario: str) -> None:
        direct = make_args(scenario=scenario, tensor_bytes=64, bulk_work_profile="descriptor_only")
        fallback = make_args(
            scenario=scenario,
            tensor_bytes=64,
            bulk_work_profile="descriptor_only",
            force_holomq_process_local_inline=True,
            channel_pump_cpu=5,
        )

        runner.validate_args(direct)
        runner.validate_args(fallback)

        assert runner.process_local_delivery(direct) == "direct_retained"
        assert runner.process_local_delivery(fallback) == "holomq"
        assert runner.benchmark_name(scenario, "latency", True) == (
            f"{runner.BENCHMARK_NAMES[scenario]}_holomq_fallback"
        )

    @pytest.mark.parametrize(
        ("scenario", "native_name"),
        [
            ("schema", "runtime_graph_schema_time_1to1_holomq_fallback"),
            (
                "third_party_schema",
                "runtime_graph_third_party_telemetry_1to1_holomq_fallback",
            ),
        ],
    )
    def test_direct_schema_omits_channel_pump_and_admits_holomq_fallback(
        self, scenario: str, native_name: str
    ) -> None:
        # Key setup: compare compiler-retained generated-schema delivery with the same graph
        # forced through its process-local HoloMQ serialized endpoint.
        direct = make_args(scenario=scenario)
        fallback = make_args(scenario=scenario, force_holomq_process_local_inline=True)

        runner.validate_args(direct)
        runner.validate_args(fallback)
        with pytest.raises(ValueError, match="requires an active channel pump"):
            runner.validate_args(
                make_args(
                    scenario=scenario,
                    force_holomq_process_local_inline=True,
                    channel_pump_cpu=5,
                )
            )

        # Key assertion: result identity distinguishes the two delivery paths, and compile()
        # delivers the split HoloMQ row inside emit(), so neither owns a Channel pump.
        assert runner.uses_channel_pump(direct) is False
        assert runner.process_local_delivery(direct) == "direct_retained"
        assert runner.default_emit_delivery(fallback) is True
        assert runner.uses_channel_pump(fallback) is False
        assert runner.process_local_delivery(fallback) == "holomq"
        assert runner.benchmark_name(scenario, "latency", True) == native_name

    def test_pump_delivery_control_restores_the_channel_pump(self) -> None:
        # Key setup: the same split POD HoloMQ row under compile() and under the pump-delivery
        # control.
        default = make_args(scenario="pod", force_holomq_process_local_inline=True)
        pump = make_args(
            scenario="pod",
            force_holomq_process_local_inline=True,
            force_process_local_pump_delivery=True,
            channel_pump_cpu=5,
        )

        runner.validate_args(default)
        runner.validate_args(pump)
        command = runner.benchmark_command(
            pump, Path("/tmp/runtime_graph_benchmark"), Path("/tmp/result.json")
        )

        # Key assertion: only the control needs event wake and a pump, and its command and
        # result identity record the control.
        assert runner.default_emit_delivery(default) is True
        assert runner.event_wake_required(default) is False
        assert runner.default_emit_delivery(pump) is False
        assert runner.event_wake_required(pump) is True
        assert runner.uses_channel_pump(pump) is True
        assert "--force-process-local-pump-delivery" in command
        assert (
            runner.benchmark_name("pod", "latency", True, force_process_local_pump_delivery=True)
            == "runtime_graph_pod_1to1_holomq_fallback_pump_delivery"
        )
        with pytest.raises(ValueError, match="requires a split-lane HoloMQ row"):
            runner.validate_args(
                make_args(
                    scenario="pod",
                    operator_lanes="shared",
                    force_holomq_process_local_inline=True,
                    force_process_local_pump_delivery=True,
                )
            )

    def test_event_wake_is_suppressed_only_for_certified_holomq_rows(self) -> None:
        # Key setup: compare both certified queue policies and Tensor shape with retained-wake
        # controls.
        blocking = make_args(
            scenario="pod", operator_lanes="shared", force_holomq_process_local_inline=True
        )
        drop_oldest = make_args(
            scenario="pod",
            operator_lanes="shared",
            queue_policy="drop_oldest",
            queue_depth=1,
            force_holomq_process_local_inline=True,
        )
        split = make_args(scenario="pod", force_holomq_process_local_inline=True)
        split_tensor = make_args(scenario="tensor", force_holomq_process_local_inline=True)
        tensor = make_args(
            scenario="tensor", operator_lanes="shared", force_holomq_process_local_inline=True
        )
        tensor_forced_wake = make_args(
            scenario="tensor",
            operator_lanes="shared",
            force_holomq_process_local_inline=True,
            force_holomq_event_wake=True,
        )
        pod_fanout = make_args(scenario="pod", operator_lanes="shared", fanout=2)
        pod_fanout_forced_wake = make_args(
            scenario="pod",
            operator_lanes="shared",
            fanout=2,
            force_holomq_event_wake=True,
        )
        schema_fanout = make_args(scenario="schema", operator_lanes="shared", fanout=2)
        third_party_fanout = make_args(
            scenario="third_party_schema", operator_lanes="shared", fanout=2
        )
        tensor_fanout = make_args(scenario="tensor", operator_lanes="shared", fanout=2)
        tensor_fanout_forced_wake = make_args(
            scenario="tensor",
            operator_lanes="shared",
            fanout=2,
            force_holomq_event_wake=True,
        )
        image_fanout = make_args(scenario="image", operator_lanes="shared", fanout=2)
        image_fanout_forced_wake = make_args(
            scenario="image",
            operator_lanes="shared",
            fanout=2,
            force_holomq_event_wake=True,
        )
        multi_tensor_loan_fanout = make_args(
            scenario="multi_tensor_loan", operator_lanes="shared", fanout=2
        )
        multi_tensor_loan_fanout_forced_wake = make_args(
            scenario="multi_tensor_loan",
            operator_lanes="shared",
            fanout=2,
            force_holomq_event_wake=True,
        )
        multi_tensor_staging_fanout = make_args(
            scenario="multi_tensor_staging", operator_lanes="shared", fanout=2
        )
        multi_tensor_staging_fanout_forced_wake = make_args(
            scenario="multi_tensor_staging",
            operator_lanes="shared",
            fanout=2,
            force_holomq_event_wake=True,
        )

        # Key assertion: exact POD/Tensor and complete POD/schema/Tensor descriptor fanout
        # suppress wake; split Tensor and forced controls retain it.
        assert runner.event_wake_required(blocking) is False
        assert runner.event_wake_required(drop_oldest) is False
        # compile() delivers a split inline row inside emit(); a split Tensor row keeps the wake.
        assert runner.event_wake_required(split) is False
        assert runner.event_wake_required(split_tensor) is True
        assert runner.event_wake_required(tensor) is False
        assert runner.event_wake_required(tensor_forced_wake) is True
        assert runner.event_wake_required(pod_fanout) is False
        assert runner.event_wake_required(pod_fanout_forced_wake) is True
        assert runner.event_wake_required(schema_fanout) is False
        assert runner.event_wake_required(third_party_fanout) is False
        assert runner.event_wake_required(tensor_fanout) is False
        assert runner.event_wake_required(tensor_fanout_forced_wake) is True
        assert runner.event_wake_required(image_fanout) is False
        assert runner.event_wake_required(image_fanout_forced_wake) is True
        assert runner.event_wake_required(multi_tensor_loan_fanout) is False
        assert runner.event_wake_required(multi_tensor_loan_fanout_forced_wake) is True
        assert runner.event_wake_required(multi_tensor_staging_fanout) is False
        assert runner.event_wake_required(multi_tensor_staging_fanout_forced_wake) is True

    @pytest.mark.parametrize(
        "scenario",
        [
            "pod",
            "schema",
            "third_party_schema",
            "tensor",
            "image",
            "multi_tensor_loan",
            "multi_tensor_staging",
        ],
    )
    def test_fanout_event_wake_control_preserves_active_geometry(self, scenario: str) -> None:
        # Key setup: retain event wake on an otherwise eventless two-subscriber shared-lane
        # POD/schema/Tensor descriptor service without requesting the redundant 1:1 HoloMQ
        # fallback control.
        args = make_args(
            scenario=scenario,
            operator_lanes="shared",
            fanout=2,
            force_holomq_event_wake=True,
            bulk_work_profile=(
                "descriptor_only"
                if scenario in {"image", *runner.MULTI_TENSOR_SCENARIOS}
                else "full_write"
            ),
        )

        runner.validate_args(args)
        command = runner.benchmark_command(
            args, Path("/tmp/runtime_graph_benchmark"), Path("/tmp/result.json")
        )

        # Key assertion: the control changes only wake eligibility and result identity; the two
        # active subscribers remain the native fanout geometry.
        assert "--fanout" in command
        assert "--force-holomq-event-wake" in command
        assert "--force-holomq-process-local-inline" not in command
        assert runner.process_local_delivery(args) == "holomq"
        assert runner.event_wake_required(args) is True
        assert (
            runner.benchmark_name(scenario, "latency", fanout=2, force_holomq_event_wake=True)
            == runner.BENCHMARK_NAMES[scenario].removesuffix("1to1") + "1to2_event_wake_control"
        )

    def test_direct_image_omits_channel_pump_and_admits_holomq_fallback(self) -> None:
        # Key setup: compare direct retention of Image metadata plus its embedded Tensor handle
        # with the same graph forced through the HoloMQ descriptor-sidecar path.
        direct = make_args(
            scenario="image", tensor_bytes=1 << 20, bulk_work_profile="descriptor_only"
        )
        fallback = make_args(
            scenario="image",
            tensor_bytes=1 << 20,
            bulk_work_profile="descriptor_only",
            force_holomq_process_local_inline=True,
            channel_pump_cpu=5,
        )

        runner.validate_args(direct)
        runner.validate_args(fallback)

        # Key assertion: both measurements retain their exact delivery identity and only the
        # transport-backed control owns a Channel pump.
        assert runner.uses_channel_pump(direct) is False
        assert runner.process_local_delivery(direct) == "direct_retained"
        assert runner.uses_channel_pump(fallback) is True
        assert runner.process_local_delivery(fallback) == "holomq"
        assert runner.benchmark_name("image", "latency", True) == (
            "runtime_graph_image_1to1_holomq_fallback"
        )

    def test_notification_latency_carries_idle_period_and_driver_placement(self) -> None:
        # Key setup: configure the paced notification workload and all participating CPUs.
        args = make_args(
            workload="notification_latency",
            publisher_period_us=5_000,
            source_cpu=2,
            sink_cpu=3,
            controller_cpu=4,
            channel_pump_cpu=5,
            notification_driver_cpu=6,
            force_holomq_process_local_inline=True,
            force_process_local_pump_delivery=True,
        )

        runner.validate_args(args)
        command = runner.benchmark_command(
            args, Path("/tmp/runtime_graph_benchmark"), Path("/tmp/result.json")
        )

        # Key assertion: the result identity and command retain the notification geometry.
        assert runner.benchmark_name("pod", args.workload).endswith("_notification_latency")
        assert command[command.index("--publisher-period-us") + 1] == "5000"
        assert command[command.index("--notification-capacity") + 1] == "1"
        assert command[command.index("--notification-driver-cpu") + 1] == "6"

    @pytest.mark.parametrize(
        ("source_cpu", "controller_cpu"),
        [(2, -1), (-1, 2), (2, 2)],
    )
    def test_shared_operator_lane_uses_one_controller_owner(
        self, source_cpu: int, controller_cpu: int
    ) -> None:
        # Key setup: source and sink share the controller thread while the exact pump control
        # retains its independently pinned support thread.
        args = make_args(
            operator_lanes="shared",
            workload="notification_latency",
            source_cpu=source_cpu,
            controller_cpu=controller_cpu,
            channel_pump_cpu=5,
            notification_driver_cpu=6,
            force_holomq_process_local_inline=True,
            force_process_local_channel_pump=True,
        )

        runner.validate_args(args)
        command = runner.benchmark_command(
            args, Path("/tmp/runtime_graph_benchmark"), Path("/tmp/result.json")
        )
        topology = runner.collect_cpu_topology(args)

        # Key assertion: topology reports the controller-owned lane once and keeps the explicit
        # pump control without inventing a separate source or sink thread.
        assert command[command.index("--operator-lanes") + 1] == "shared"
        assert command[command.index("--channel-pump-cpu") + 1] == "5"
        assert [item["role"] for item in topology] == [
            "controller",
            "channel_pump",
            "notification_driver",
        ]
        assert topology[0]["cpu"] == 2
        quality = runner.quality_gate(
            args,
            topology,
            {"coefficient_of_variation": 0.0},
        )
        assert quality["publishable"] is True
        assert quality["issues"] == []

    def test_source_only_notification_latency_omits_sink_placement(self) -> None:
        # Key setup: isolate notification-to-source activation without a Channel edge or sink.
        args = make_args(
            workload="notification_latency",
            scenario="source_only",
            source_cpu=2,
            controller_cpu=2,
            notification_driver_cpu=6,
        )

        runner.validate_args(args)
        topology = runner.collect_cpu_topology(args)

        # Key assertion: result naming and topology expose the reduced measurement boundary.
        assert runner.benchmark_name(args.scenario, args.workload) == (
            "runtime_graph_source_activation_notification_latency"
        )
        assert [item["role"] for item in topology] == [
            "controller",
            "notification_driver",
        ]
        assert topology[0]["cpu"] == 2

    def test_source_only_saturated_uses_only_activation_roles(self) -> None:
        # Key setup: saturate one portless Operator with bounded notification credits.
        args = make_args(
            workload="saturated",
            scenario="source_only",
            queue_depth=8,
            notification_capacity=8,
            publisher_period_us=0,
            source_cpu=2,
            controller_cpu=2,
            notification_driver_cpu=6,
        )

        runner.validate_args(args)
        topology = runner.collect_cpu_topology(args)

        # Key assertion: the row retains one controller-owned lane and the independent driver.
        assert runner.benchmark_name(args.scenario, args.workload) == (
            "runtime_graph_source_activation_saturated"
        )
        assert [item["role"] for item in topology] == [
            "controller",
            "notification_driver",
        ]


class TestSummaries:
    def test_summarize_values_reports_run_to_run_cv(self) -> None:
        # Key assertion: the runner reports sample CV alongside central tendency.
        summary = runner.summarize_values([90.0, 100.0, 110.0])

        assert summary["count"] == 3
        assert summary["median"] == 100.0
        assert summary["mean"] == 100.0
        assert summary["coefficient_of_variation"] == pytest.approx(0.1)

    def test_quality_gate_flags_environment_without_failing_measurement(self) -> None:
        # Key setup: one unpinned repetition is measurable but not publication-quality evidence.
        quality = runner.quality_gate(
            make_args(repetitions=1),
            [
                {"role": "source", "cpu": -1, "pinned": False},
                {"role": "sink", "cpu": -1, "pinned": False},
            ],
            {"coefficient_of_variation": None},
        )

        # Key assertion: quality metadata records both independent deficiencies.
        assert not quality["publishable"]
        assert quality["issues"] == [
            "insufficient_repetitions_for_dispersion",
            "runtime_thread_cpu_affinity_unpinned",
        ]

    def test_source_only_saturated_stability_uses_activation_rate(self) -> None:
        # Key setup: source-only runs report zero publications but differing activation rates.
        args = make_args(
            workload="saturated",
            scenario="source_only",
            repetitions=2,
            max_median_cv=0.01,
        )
        results = [
            {
                **summary_evidence("notification"),
                "benchmark": "runtime_graph_source_activation_saturated",
                "answers": "source activation ceiling",
                "geometry": {},
                "partitioning": {
                    "accepted_algorithm": {
                        "id": "holoscan.partition.serial_baseline",
                        "version": 1,
                    }
                },
                "wake": {
                    "event_wake": False,
                    "busy_poll": False,
                    "receive_polling": False,
                },
                "placement": {"support_threads": []},
                "timing": {
                    "logical_publications_per_second": 0.0,
                    "source_activations_per_second": rate,
                    "aggregate_deliveries_per_second": 0.0,
                },
            }
            for rate in (100.0, 110.0)
        ]

        summary = runner.summarize_results(args, results, [])

        # Key assertion: activation dispersion, not the constant zero rate, gates the row.
        assert "run_to_run_median_cv_exceeded" in summary["quality_gate"]["issues"]
        assert summary["wake"] == {
            "event_wake": False,
            "busy_poll": False,
            "receive_polling": False,
        }
        assert summary["runtime_support_threads"] == "none"

    def test_notification_summary_keeps_component_latencies(self) -> None:
        # Key setup: end-to-end latency includes notification delivery, while the native result
        # separately records source callback setup and pre-emit through sink completion.
        args = make_args(workload="notification_latency", repetitions=2, fanout=2)
        results = [
            {
                **summary_evidence("notification"),
                "benchmark": "runtime_graph_pod_1to2_notification_latency",
                "answers": "typed process-local delivery",
                "geometry": {},
                "partitioning": {},
                "wake": {
                    "event_wake": False,
                    "busy_poll": False,
                    "receive_polling": False,
                },
                "placement": {"support_threads": []},
                "timing": {
                    "configured_publication_rate_per_second": 1_000.0,
                    "logical_publications_per_second": None,
                    "source_activations_per_second": 100.0,
                    "aggregate_deliveries_per_second": None,
                    "latency_ns": {"p50": end_to_end, "p90": end_to_end, "p99": end_to_end},
                    "publication_to_receive_ns": {
                        "p50": source_to_sink,
                        "p90": source_to_sink,
                        "p99": source_to_sink,
                    },
                    "notification_to_source_ns": {"p50": 2_000.0},
                    "source_callback_entry_to_pre_emit_ns": {"p50": callback_setup},
                    "source_emit_ns": {"p50": source_emit},
                    "source_publication_call_ns": {"p50": publication_call},
                    "source_post_emit_to_pre_return_ns": {"p50": source_residual},
                    "source_post_emit_to_sink_ns": {"p50": post_emit_to_sink},
                    "source_post_emit_to_sink_overlaps": 0,
                    "source_to_sink_ns": {"p50": source_to_sink},
                    "source_to_sink_by_sink_ns": [
                        {"index": 0, "latency_ns": {"p50": first_sink}},
                        {"index": 1, "latency_ns": {"p50": source_to_sink}},
                    ],
                    "source_post_emit_to_sink_by_sink_ns": [
                        {
                            "index": 0,
                            "overlaps": 0,
                            "latency_ns": {"p50": first_sink - source_emit},
                        },
                        {
                            "index": 1,
                            "overlaps": 0,
                            "latency_ns": {"p50": post_emit_to_sink},
                        },
                    ],
                    "source_pre_return_to_sink_callback_by_sink_ns": [
                        {
                            "index": 0,
                            "overlaps": 0,
                            "latency_ns": {
                                "p50": first_sink - source_emit - source_residual - sink_receive
                            },
                        },
                        {
                            "index": 1,
                            "overlaps": 0,
                            "latency_ns": {
                                "p50": post_emit_to_sink - source_residual - sink_receive
                            },
                        },
                    ],
                    "sink_callback_to_post_receive_by_sink_ns": [
                        {"index": 0, "latency_ns": {"p50": sink_receive}},
                        {"index": 1, "latency_ns": {"p50": sink_receive}},
                    ],
                },
            }
            for (
                end_to_end,
                callback_setup,
                source_emit,
                publication_call,
                first_sink,
                source_to_sink,
                post_emit_to_sink,
                source_residual,
                sink_receive,
            ) in (
                (
                    12_000.0,
                    1_000.0,
                    2_500.0,
                    500.0,
                    4_500.0,
                    9_000.0,
                    6_500.0,
                    100.0,
                    200.0,
                ),
                (
                    14_000.0,
                    1_200.0,
                    3_000.0,
                    600.0,
                    5_500.0,
                    11_000.0,
                    8_000.0,
                    120.0,
                    220.0,
                ),
            )
        ]

        summary = runner.summarize_results(args, results, [])

        # Key assertion: repeated-run output retains both boundaries instead of presenting only
        # notification-to-sink latency as the data-path cost.
        assert summary["metrics"]["latency_p50_ns_across_runs"]["median"] == 13_000.0
        assert summary["metrics"]["logical_publications_per_second_across_runs"] is None
        assert summary["metrics"]["configured_publication_rate_per_second"]["median"] == 1_000.0
        assert (
            summary["metrics"]["source_callback_entry_to_pre_emit_p50_ns_across_runs"]["median"]
            == 1_100.0
        )
        assert summary["metrics"]["source_to_sink_p50_ns_across_runs"]["median"] == 10_000.0
        assert [run["source_callback_entry_to_pre_emit_p50_ns"] for run in summary["runs"]] == [
            1_000.0,
            1_200.0,
        ]
        assert [run["source_to_sink_p50_ns"] for run in summary["runs"]] == [9_000.0, 11_000.0]
        assert [
            item["median"]
            for item in summary["metrics"]["source_to_sink_by_sink_p50_ns_across_runs"]
        ] == [5_000.0, 10_000.0]
        assert [run["source_to_sink_by_sink_p50_ns"] for run in summary["runs"]] == [
            [4_500.0, 9_000.0],
            [5_500.0, 11_000.0],
        ]
        assert summary["metrics"]["source_emit_p50_ns_across_runs"]["median"] == 2_750.0
        assert summary["metrics"]["source_publication_call_p50_ns_across_runs"]["median"] == 550.0
        assert [run["source_publication_call_p50_ns"] for run in summary["runs"]] == [
            500.0,
            600.0,
        ]
        assert (
            summary["metrics"]["source_post_emit_to_pre_return_p50_ns_across_runs"]["median"]
            == 110.0
        )
        assert (
            summary["metrics"]["source_post_emit_to_sink_p50_ns_across_runs"]["median"] == 7_250.0
        )
        assert summary["metrics"]["source_post_emit_to_sink_overlaps_across_runs"]["median"] == 0.0
        assert [
            item["median"]
            for item in summary["metrics"]["source_post_emit_to_sink_by_sink_p50_ns_across_runs"]
        ] == [2_250.0, 7_250.0]
        assert [
            item["median"]
            for item in summary["metrics"]["source_post_emit_to_sink_by_sink_overlaps_across_runs"]
        ] == [0.0, 0.0]
        assert [
            item["median"]
            for item in summary["metrics"][
                "source_pre_return_to_sink_callback_by_sink_p50_ns_across_runs"
            ]
        ] == [1_930.0, 6_930.0]
        assert [
            item["median"]
            for item in summary["metrics"][
                "source_pre_return_to_sink_callback_by_sink_overlaps_across_runs"
            ]
        ] == [0.0, 0.0]
        assert [
            item["median"]
            for item in summary["metrics"][
                "sink_callback_to_post_receive_by_sink_p50_ns_across_runs"
            ]
        ] == [210.0, 210.0]
        assert summary["runs"][0]["source_post_emit_to_sink_overlaps"] == 0.0
        assert summary["runs"][0]["source_post_emit_to_sink_by_sink_overlaps"] == [0.0, 0.0]
        assert summary["runs"][0]["source_pre_return_to_sink_callback_by_sink_overlaps"] == [
            0.0,
            0.0,
        ]

    def test_tensor_summary_keeps_loan_write_and_receive_phases(self) -> None:
        # Key setup: two repeated Tensor rows expose the pool-loan, producer-write, and public
        # receive intervals independently of the complete publication call.
        args = make_args(
            workload="notification_latency",
            scenario="tensor",
            payload_validation="none",
            repetitions=2,
        )
        results = []
        for loan, write, receive in ((400.0, 300.0, 350.0), (500.0, 340.0, 450.0)):
            results.append(
                {
                    **summary_evidence("notification"),
                    "benchmark": "runtime_graph_tensor_1to1_notification_latency",
                    "answers": "Tensor phase attribution",
                    "tensor": {},
                    "geometry": {},
                    "partitioning": {},
                    "wake": {
                        "event_wake": False,
                        "busy_poll": True,
                        "receive_polling": False,
                    },
                    "placement": {"support_threads": []},
                    "validation": {
                        "mode": "none",
                        "timing": "not_performed",
                        "passed": True,
                    },
                    "timing": {
                        "configured_publication_rate_per_second": 5_000.0,
                        "logical_publications_per_second": None,
                        "source_activations_per_second": 3_700.0,
                        "aggregate_deliveries_per_second": None,
                        "latency_ns": {"p50": 12_000.0, "p90": 13_000.0, "p99": 14_000.0},
                        "publication_to_receive_ns": {"p50": 6_000.0, "p99": 7_000.0},
                        "notification_to_source_ns": {"p50": 5_000.0},
                        "source_callback_entry_to_pre_emit_ns": {"p50": 900.0},
                        "source_emit_ns": {"p50": 1_600.0},
                        "source_publication_call_ns": {"p50": 700.0},
                        "source_post_emit_to_pre_return_ns": {"p50": 30.0},
                        "source_post_emit_to_sink_ns": {"p50": 4_500.0},
                        "source_post_emit_to_sink_overlaps": 0,
                        "source_to_sink_ns": {"p50": 6_100.0},
                        "tensor_loan_acquire_ns": {"p50": loan},
                        "tensor_write_commit_ns": {"p50": write},
                        "tensor_receive_call_ns": {"p50": receive},
                        "source_to_sink_by_sink_ns": [{"index": 0, "latency_ns": {"p50": 6_100.0}}],
                        "source_post_emit_to_sink_by_sink_ns": [
                            {"index": 0, "overlaps": 0, "latency_ns": {"p50": 4_500.0}}
                        ],
                        "source_pre_return_to_sink_callback_by_sink_ns": [
                            {"index": 0, "overlaps": 0, "latency_ns": {"p50": 4_100.0}}
                        ],
                        "sink_callback_to_post_receive_by_sink_ns": [
                            {"index": 0, "latency_ns": {"p50": receive}}
                        ],
                    },
                }
            )

        summary = runner.summarize_results(args, results, [])

        # Key assertion: aggregation preserves each Tensor-specific interval for comparisons
        # between direct retention and the transport-backed sidecar path.
        assert summary["metrics"]["tensor_loan_acquire_p50_ns_across_runs"]["median"] == 450.0
        assert summary["metrics"]["tensor_write_commit_p50_ns_across_runs"]["median"] == 320.0
        assert summary["metrics"]["tensor_receive_call_p50_ns_across_runs"]["median"] == 400.0
        assert [run["tensor_receive_call_p50_ns"] for run in summary["runs"]] == [350.0, 450.0]

    def test_latency_summary_retains_p99_and_available_p999(self) -> None:
        # Key setup: a 10,000-sample paced run carries the native p999 tail.
        args = make_args(samples=10_000, repetitions=2)
        results = [
            {
                **summary_evidence("clock"),
                "benchmark": "runtime_graph_pod_1to1",
                "answers": "typed process-local delivery",
                "geometry": {},
                "partitioning": {},
                "wake": {
                    "event_wake": False,
                    "busy_poll": False,
                    "receive_polling": False,
                },
                "placement": {"support_threads": []},
                "timing": {
                    "configured_publication_rate_per_second": 1_000.0,
                    "logical_publications_per_second": None,
                    "source_activations_per_second": 1_000.0,
                    "aggregate_deliveries_per_second": None,
                    "latency_ns": {
                        "p50": 10_000.0,
                        "p90": 20_000.0,
                        "p99": p99,
                        "p999": p999,
                    },
                    "publication_to_receive_ns": {
                        "p50": 8_000.0,
                        "p90": 16_000.0,
                        "p99": p99 - 2_000.0,
                        "p999": p999 - 2_000.0,
                    },
                },
            }
            for p99, p999 in ((30_000.0, 40_000.0), (32_000.0, 44_000.0))
        ]

        summary = runner.summarize_results(args, results, [])

        # Key assertion: aggregation and per-run output preserve both reported tail levels.
        assert summary["metrics"]["latency_p99_ns_across_runs"]["median"] == 31_000.0
        assert summary["metrics"]["latency_p999_ns_across_runs"]["median"] == 42_000.0
        assert [run["latency_p999_ns"] for run in summary["runs"]] == [40_000.0, 44_000.0]

    def test_quality_gate_accepts_stable_performance_governor_run(self) -> None:
        # Key setup: repeated, stable measurements run on pinned performance-governor CPUs.
        quality = runner.quality_gate(
            make_args(),
            [
                {
                    "role": "source",
                    "cpu": 2,
                    "pinned": True,
                    "topology_available": True,
                    "governor": "performance",
                },
                {
                    "role": "sink",
                    "cpu": 3,
                    "pinned": True,
                    "topology_available": True,
                    "governor": "performance",
                },
            ],
            {"coefficient_of_variation": 0.02},
        )

        # Key assertion: a controlled environment clears the advisory publication gate.
        assert quality["publishable"]
        assert quality["issues"] == []


class TestResultValidation:
    @pytest.mark.parametrize("fanout", [1, 2, 4, 8])
    def test_tensor_phase_timing_counts_receive_samples_per_sink(self, fanout: int) -> None:
        # Key setup: loan and write are producer phases recorded once per publication, while each
        # fanout recipient contributes one independently paired public receive-call interval.
        args = make_args(samples=8, scenario="tensor", fanout=fanout)
        timing = {
            "tensor_loan_acquire_ns": {"samples": 8},
            "tensor_write_commit_ns": {"samples": 8},
            "tensor_receive_call_ns": {"samples": 8 * fanout},
        }

        runner.validate_tensor_phase_timing(timing, args, Path("result.json"))

        # Key assertion: dropping even one sink interval fails the aggregate receive evidence.
        timing["tensor_receive_call_ns"]["samples"] -= 1
        with pytest.raises(RuntimeError, match="incomplete Tensor phase timing"):
            runner.validate_tensor_phase_timing(timing, args, Path("result.json"))

    @pytest.mark.parametrize("field", ["tensor_loan_acquire_ns", "tensor_write_commit_ns"])
    def test_tensor_phase_timing_does_not_multiply_producer_samples(self, field: str) -> None:
        args = make_args(samples=8, scenario="tensor", fanout=2)
        timing = {
            "tensor_loan_acquire_ns": {"samples": 8},
            "tensor_write_commit_ns": {"samples": 8},
            "tensor_receive_call_ns": {"samples": 16},
        }
        timing[field]["samples"] = 16

        # Key assertion: only the recipient-side phase scales with fanout.
        with pytest.raises(RuntimeError, match="incomplete Tensor phase timing"):
            runner.validate_tensor_phase_timing(timing, args, Path("result.json"))

    def test_callback_counts_accept_reported_clocked_terminal_drain_entries(self) -> None:
        # Key setup: reproduce the aarch64 fanout control result where final Tensor retirement
        # took long enough for three more periodic source candidates to enter compute.
        counts = {
            "source_callbacks": 11,
            "source_compute_entries": 14,
            "source_terminal_compute_entries": 3,
            "source_tensor_pool_retries": 0,
            "source_tensor_pool_in_activation_retries": 0,
            "relay_callbacks": 0,
            "sink_callbacks": 22,
            "relay_publications": 0,
            "backpressured_emits": 0,
            "source_backpressured_emits": 0,
        }
        args = make_args(
            samples=8,
            warmup=2,
            workload="latency",
            scenario="tensor",
            fanout=2,
        )

        runner.validate_callback_counts(Path("result.json"), counts, args, 11, 22)

        # Key assertion: the explicit terminal count must account for every otherwise extra
        # compute entry; it is evidence, not an arbitrary scheduler allowance.
        counts["source_terminal_compute_entries"] = 2
        with pytest.raises(RuntimeError, match="inconsistent callback counts"):
            runner.validate_callback_counts(Path("result.json"), counts, args, 11, 22)

    @pytest.mark.parametrize(
        ("scenario", "accepted"), [("multi_tensor_loan", True), ("multi_tensor_staging", False)]
    )
    def test_callback_counts_accept_loan_field_pool_retries(
        self, scenario: str, accepted: bool
    ) -> None:
        # Key setup: the Jetson AGX Thor fanout event-wake control counts, with the activation
        # whose bundle allocation found a field pool exhausted reported as a pool retry.
        counts = {
            "source_callbacks": 19,
            "source_compute_entries": 20,
            "source_tensor_pool_retries": 1,
            "source_tensor_pool_in_activation_retries": 0,
            "relay_callbacks": 0,
            "sink_callbacks": 38,
            "relay_publications": 0,
            "backpressured_emits": 0,
            "source_backpressured_emits": 0,
        }
        args = make_args(samples=8, warmup=10, workload="latency", scenario=scenario, fanout=2)

        # Key assertion: only the loan source allocates from a field pool before it emits.
        if accepted:
            runner.validate_callback_counts(Path("result.json"), counts, args, 19, 38)
        else:
            with pytest.raises(RuntimeError, match="inconsistent callback counts"):
                runner.validate_callback_counts(Path("result.json"), counts, args, 19, 38)

    def test_callback_counts_distinguish_relay_backpressure_from_source_retries(self) -> None:
        # This is the exact count pattern from the partitioned-chain backpressure CI failure:
        # all nine retries occurred in relays, so the source still ran exactly once per sample.
        counts = {
            "source_callbacks": 41,
            "source_compute_entries": 41,
            "source_tensor_pool_retries": 0,
            "source_tensor_pool_in_activation_retries": 0,
            "relay_callbacks": 164,
            "sink_callbacks": 41,
            "relay_publications": 164,
            "backpressured_emits": 9,
            "source_backpressured_emits": 0,
        }
        args = make_args(samples=40, warmup=0, scenario="sample_forward", relay_count=4)

        runner.validate_callback_counts(Path("result.json"), counts, args, 41, 41)

        # Source retries add source callbacks independently of relay-side backpressure.
        counts.update(
            {
                "source_callbacks": 43,
                "source_compute_entries": 43,
                "backpressured_emits": 11,
                "source_backpressured_emits": 2,
            }
        )
        runner.validate_callback_counts(Path("result.json"), counts, args, 41, 41)

    def test_callback_count_failure_reports_actual_and_expected_counts(self) -> None:
        counts = {
            "source_callbacks": 42,
            "source_compute_entries": 42,
            "source_tensor_pool_retries": 0,
            "source_tensor_pool_in_activation_retries": 0,
            "relay_callbacks": 164,
            "sink_callbacks": 41,
            "relay_publications": 164,
            "backpressured_emits": 9,
            "source_backpressured_emits": 0,
        }
        args = make_args(samples=40, warmup=0, scenario="sample_forward", relay_count=4)

        with pytest.raises(RuntimeError) as error:
            runner.validate_callback_counts(Path("result.json"), counts, args, 41, 41)

        message = str(error.value)
        assert "source_callbacks=42, expected_source_callbacks=41" in message
        assert "relay_callbacks=164, expected_relay_callbacks=164" in message
        assert "relay_publications=164, expected_relay_publications=164" in message
        assert "sink_callbacks=41, expected_sink_callbacks=41" in message
        assert "backpressured_emits=9, source_backpressured_emits=0" in message

    @staticmethod
    def burst_counts(
        source: int,
        measured: int,
        empty: int = 0,
        short: int = 0,
        measured_empty: int | None = None,
        measured_short: int | None = None,
        sinks: int = 5,
    ) -> dict[str, int]:
        # Retries start in measured callbacks unless a test says otherwise.
        measured_empty = empty if measured_empty is None else measured_empty
        measured_short = short if measured_short is None else measured_short
        return {
            "source_callbacks": source,
            "source_compute_entries": source,
            "measured_source_callbacks": measured,
            "source_empty_retry_callbacks": empty,
            "source_short_retry_callbacks": short,
            "measured_source_empty_retry_callbacks": measured_empty,
            "measured_source_short_retry_callbacks": measured_short,
            "source_tensor_pool_retries": 0,
            "source_tensor_pool_in_activation_retries": 0,
            "relay_callbacks": 0,
            "sink_callbacks": sinks,
            "relay_publications": 0,
            "backpressured_emits": empty + short,
            "source_backpressured_emits": empty + short,
        }

    def test_burst_callback_counts_account_for_short_and_empty_retries(self) -> None:
        # Key setup: blocking POD bursts of two after the allocation probe (sequence 1). With four
        # samples the rejection-free callbacks are: probe; 2-3; 4-5.
        args = make_args(samples=4, warmup=0, burst_size=2, queue_depth=2)
        runner.validate_callback_counts(Path("result.json"), self.burst_counts(3, 2), args, 5, 5)

        # A short retry: publish 2 and reject 3; publish 3-4; publish 5. One more callback.
        runner.validate_callback_counts(
            Path("result.json"), self.burst_counts(4, 3, short=1), args, 5, 5
        )
        # An empty retry: reject 2 at once; publish 2-3; publish 4-5. Exactly one more callback.
        runner.validate_callback_counts(
            Path("result.json"), self.burst_counts(4, 3, empty=1), args, 5, 5
        )

        # Key assertion: an empty retry always costs a callback, a short retry at most one, and
        # every source rejection is classified once.
        for counts in (
            self.burst_counts(3, 2, empty=1),
            self.burst_counts(5, 4, short=1),
            {**self.burst_counts(4, 3, short=1), "source_backpressured_emits": 2},
            self.burst_counts(4, 3, short=1, measured_short=2),
        ):
            with pytest.raises(RuntimeError, match="inconsistent callback counts"):
                runner.validate_callback_counts(Path("result.json"), counts, args, 5, 5)

    def test_short_burst_retry_can_leave_the_callback_count_at_its_minimum(self) -> None:
        # Key setup: five samples after the probe: probe; 2-3; 4-5; 6. Publishing 2 and rejecting
        # 3 realigns the bursts to 3-4 and 5-6, so the run still takes four callbacks.
        args = make_args(samples=5, warmup=0, burst_size=2, queue_depth=2)

        runner.validate_callback_counts(
            Path("result.json"), self.burst_counts(4, 3, short=1, sinks=6), args, 6, 6
        )

    def test_notification_count_caps_source_compute_entries_after_source_retries(self) -> None:
        # Key setup: two source backpressure retries account for the only notifications accepted
        # beyond the three productive source callbacks.
        args = make_args(
            samples=2,
            warmup=0,
            workload="saturated",
            notification_capacity=4,
        )
        counts = {
            "source_callbacks": 5,
            "source_compute_entries": 5,
            "source_tensor_pool_retries": 0,
            "source_tensor_pool_in_activation_retries": 0,
            "relay_callbacks": 0,
            "sink_callbacks": 3,
            "relay_publications": 0,
            "backpressured_emits": 2,
            "source_backpressured_emits": 2,
            "notifications_accepted": 5,
        }
        runner.validate_callback_counts(Path("result.json"), counts, args, 3, 3)

        # Key assertion: accepted notifications are the absolute compute-entry ceiling; source
        # backpressure cannot be added to that same surplus a second time.
        counts["source_compute_entries"] = 6
        with pytest.raises(RuntimeError, match="inconsistent callback counts"):
            runner.validate_callback_counts(Path("result.json"), counts, args, 3, 3)

    @pytest.mark.parametrize("notifications_accepted", [None, True, 5.0, "5"])
    def test_notification_count_requires_non_boolean_integer(
        self, notifications_accepted: object
    ) -> None:
        # Key setup: make every callback count valid in a notification-driven run while supplying
        # evidence that is absent, boolean, or another non-integer type.
        args = make_args(samples=2, warmup=0, workload="saturated")
        counts = {
            "source_callbacks": 3,
            "source_compute_entries": 3,
            "source_tensor_pool_retries": 0,
            "source_tensor_pool_in_activation_retries": 0,
            "relay_callbacks": 0,
            "sink_callbacks": 3,
            "relay_publications": 0,
            "backpressured_emits": 0,
            "source_backpressured_emits": 0,
            "notifications_accepted": notifications_accepted,
        }

        # Key assertion: malformed notification evidence cannot authorize the compute count.
        with pytest.raises(RuntimeError, match="inconsistent callback counts"):
            runner.validate_callback_counts(Path("result.json"), counts, args, 3, 3)

    def test_callback_counts_do_not_double_count_tensor_pool_retries(self) -> None:
        # Key setup: one Tensor pool retry consumes the only notification admitted beyond the
        # expected publication count in a saturated run.
        args = make_args(
            samples=2,
            warmup=0,
            workload="saturated",
            scenario="tensor",
            notification_capacity=1,
        )
        counts = {
            "source_callbacks": 3,
            "source_compute_entries": 4,
            "source_tensor_pool_retries": 1,
            "source_tensor_pool_in_activation_retries": 0,
            "relay_callbacks": 0,
            "sink_callbacks": 3,
            "relay_publications": 0,
            "backpressured_emits": 0,
            "source_backpressured_emits": 0,
            "notifications_accepted": 4,
        }
        runner.validate_callback_counts(Path("result.json"), counts, args, 3, 3)

        # Key assertion: the retry cannot also be added above the notification-derived terminal
        # allowance, so a fifth source compute entry is rejected.
        counts["source_compute_entries"] = 5
        with pytest.raises(RuntimeError, match="inconsistent callback counts"):
            runner.validate_callback_counts(Path("result.json"), counts, args, 3, 3)

    @pytest.mark.parametrize("scenario", ["tensor", "multi_tensor"])
    def test_in_activation_tensor_pool_retry_is_not_emit_backpressure(self, scenario: str) -> None:
        # Key setup: one allocation retry succeeds inside the same notification-driven Tensor
        # callback without attempting a Channel emit for the failed allocation.
        args = make_args(
            samples=2,
            warmup=0,
            workload="saturated",
            scenario=scenario,
            notification_capacity=1,
        )
        counts = {
            "source_callbacks": 3,
            "source_compute_entries": 3,
            "source_tensor_pool_retries": 0,
            "source_tensor_pool_in_activation_retries": 1,
            "relay_callbacks": 0,
            "sink_callbacks": 3,
            "relay_publications": 0,
            "backpressured_emits": 0,
            "source_backpressured_emits": 0,
            "notifications_accepted": 3,
        }

        # Key assertion: the distinct retry evidence does not add a callback, notification, or
        # Channel backpressure expectation.
        runner.validate_callback_counts(Path("result.json"), counts, args, 3, 3)

    # REQ: RUNTIME-041
    @pytest.mark.parametrize(
        "args",
        [
            make_args(scenario="tensor", operator_lanes="shared", runtime_phase_counters=True),
            make_args(
                scenario="pod",
                operator_lanes="shared",
                force_holomq_process_local_inline=True,
                runtime_phase_counters=True,
            ),
        ],
        ids=("direct-retained-tensor", "holomq-pod"),
    )
    def test_direct_tensor_allows_and_holomq_rejects_continuation_receipt_evidence(
        self, args: SimpleNamespace
    ) -> None:
        # Key setup: model a complete direct Tensor or HoloMQ shared-lane cycle.
        counts = {
            "source_callbacks": 1,
            "source_compute_entries": 2,
            "relay_callbacks": 0,
            "sink_callbacks": 1,
            "publications": 1,
            "relay_publications": 0,
            "backpressured_emits": 0,
        }
        counters = {
            "enabled": True,
            "readiness_scans": 3,
            "no_progress_scans": 1,
            "wait_calls": 1,
            "wake_signals": 1,
            "wake_returns": 1,
            "timeout_returns": 0,
            "stop_signals": 1,
            "candidates_frozen": 3,
            "candidates_selected": 3,
            "readiness_poll_dispatches": 1,
            "readiness_poll_contention_retries": 0,
            "direct_continuation_receipts_accepted": 0,
            "direct_continuation_poll_elisions": 0,
            "direct_composite_attempts": 0,
            "direct_composite_completions": 0,
            "direct_composite_fallbacks": 0,
            "direct_composite_failures": 0,
            "direct_composite_callbacks_attempted": 0,
            "direct_composite_callbacks_completed": 0,
            "compute_dispatches": 3,
            "compute_admissions_run_closing": 0,
            "activations_completed": 3,
            "channel_prepare_calls": 3,
            "channel_emit_attempts": 1,
            "channel_emit_accepted": 1,
            "channel_emit_backpressured": 0,
            "channel_samples_received": 1,
            "channel_readiness_published": 1,
            "channel_input_reads": 1,
            "channel_witness_retired": 1,
            "channel_retire_calls": 3,
            "channel_activation_cuts": 0,
            "channel_activation_cut_drains": 0,
            "channel_activation_cut_skips": 0,
            "channel_activation_cut_scoped_retries": 0,
            "channel_receive_backpressured": 0,
            "channel_poll_incomplete": 0,
            "channel_activation_cut_inventory_scans": 0,
            "channel_activation_cut_subscriber_probes": 0,
            "channel_activation_cut_transport_probes": 0,
            "channel_activation_cut_materializations": 0,
            "channel_pump_waits": 0,
            "channel_pump_waits_ready": 0,
            "channel_pump_wait_timeouts": 0,
            "channel_pump_wait_timeouts_with_blocked_head": 0,
            "channel_scan_lock_contended_cut": 0,
            "channel_scan_lock_wait_ns_cut": 0,
            "channel_blocked_heads_masked": 0,
            "channel_pump_wakes_signaled": 0,
            "candidate_source_lookup_probes": 0,
            "candidate_order_comparisons": 0,
            "candidate_order_moves": 0,
            "candidate_tag_time_reservations": 0,
            "lane_dispatches": 6,
            "lane_compute_dispatches": 3,
            "lane_dispatcher_dispatches": 1,
            "lane_lifecycle_dispatches": 2,
            "lane_unknown_dispatches": 0,
            "lane_busy_task_publications": 0,
            "lane_sleeping_wake_signals": 6,
            "lane_callbacks_entered": 6,
            "lane_callbacks_exited": 6,
            "lane_completions_observed": 6,
            "lane_serialization_contentions": 0,
            "lane_inline_owner_invocations": 0,
        }
        if runner.uses_holomq(args):
            counters["channel_activation_cuts"] = 3
            counters["channel_activation_cut_skips"] = 3

        # Key assertion: zero receipts remain valid for both delivery paths.
        runner.validate_runtime_phase_counters(counters, counts, args)

        if runner.uses_holomq(args):
            # Key setup: a pump-free blocked head adds one retirement-owned activation cut and
            # records the receive backpressure that authorized it.
            retry_counters = {
                **counters,
                "channel_activation_cuts": 4,
                "channel_activation_cut_drains": 1,
                "channel_receive_backpressured": 1,
            }

            # Key assertion: the bounded retry cut is admitted, but a second unaccounted cut is
            # rejected.
            runner.validate_runtime_phase_counters(retry_counters, counts, args)
            retry_counters["channel_activation_cuts"] = 5
            retry_counters["channel_activation_cut_drains"] = 2
            with pytest.raises(RuntimeError, match="inconsistent Runtime phase counters"):
                runner.validate_runtime_phase_counters(retry_counters, counts, args)

            # Key setup: the same retry taken for its one subscriber is a scoped retry, not a
            # full drain.
            scoped_counters = {
                **counters,
                "channel_activation_cuts": 4,
                "channel_activation_cut_scoped_retries": 1,
                "channel_receive_backpressured": 1,
            }

            # Key assertion: a valid scoped retry is admitted; a cut no bucket accounts for, and a
            # scoped retry beyond the receive-backpressure budget, are rejected.
            runner.validate_runtime_phase_counters(scoped_counters, counts, args)
            for invalid in (
                {**scoped_counters, "channel_activation_cut_scoped_retries": 0},
                {
                    **scoped_counters,
                    "channel_activation_cuts": 5,
                    "channel_activation_cut_scoped_retries": 2,
                },
            ):
                with pytest.raises(RuntimeError, match="inconsistent Runtime phase counters"):
                    runner.validate_runtime_phase_counters(invalid, counts, args)

        # Key setup: pump waits that returned ready or timed out, one timeout with a blocked head.
        pump_counters = {
            **counters,
            "channel_pump_waits": 5,
            "channel_pump_waits_ready": 3,
            "channel_pump_wait_timeouts": 2,
            "channel_pump_wait_timeouts_with_blocked_head": 1,
        }

        # Key assertion: outcomes within the waits are admitted; more outcomes than waits, or more
        # blocked-head timeouts than timeouts, are rejected.
        runner.validate_runtime_phase_counters(pump_counters, counts, args)
        for invalid in (
            {**pump_counters, "channel_pump_waits": 4},
            {**pump_counters, "channel_pump_wait_timeouts_with_blocked_head": 3},
        ):
            with pytest.raises(RuntimeError, match="inconsistent Runtime phase counters"):
                runner.validate_runtime_phase_counters(invalid, counts, args)

        # Key setup: clean stop can leave one scheduler candidate frozen after every expected
        # compute transaction has already completed.
        terminal_frozen = {**counters, "candidates_frozen": 4}

        # Key assertion: exactly one frozen-but-unselected terminal candidate is admitted, while
        # two remain inconsistent with the bounded scheduler state.
        runner.validate_runtime_phase_counters(terminal_frozen, counts, args)
        terminal_frozen["candidates_frozen"] = 5
        with pytest.raises(RuntimeError, match="inconsistent Runtime phase counters"):
            runner.validate_runtime_phase_counters(terminal_frozen, counts, args)
        counters["direct_continuation_receipts_accepted"] = 1
        counters["direct_continuation_poll_elisions"] = 1
        if runner.uses_holomq(args):
            # Key assertion: transported delivery rejects direct continuation evidence.
            with pytest.raises(RuntimeError, match="inconsistent Runtime phase counters"):
                runner.validate_runtime_phase_counters(counters, counts, args)
        else:
            # Key assertion: direct-retained Tensor delivery admits matched receipt and poll-elision
            # evidence under the same cardinality bounds as other process-local payloads.
            runner.validate_runtime_phase_counters(counters, counts, args)

    def test_runtime_phase_counters_require_exact_activation_cardinality(self) -> None:
        # Key setup: one productive source callback, one final empty source entry, and one sink
        # callback require three complete candidate transactions.
        counts = {
            "source_callbacks": 1,
            "source_compute_entries": 2,
            "relay_callbacks": 0,
            "sink_callbacks": 1,
            "publications": 1,
            "relay_publications": 0,
            "backpressured_emits": 0,
        }
        counters = {
            "enabled": True,
            "readiness_scans": 3,
            "no_progress_scans": 1,
            "wait_calls": 1,
            "wake_signals": 1,
            "wake_returns": 1,
            "timeout_returns": 0,
            "stop_signals": 1,
            "candidates_frozen": 3,
            "candidates_selected": 3,
            "readiness_poll_dispatches": 1,
            "readiness_poll_contention_retries": 0,
            "direct_continuation_receipts_accepted": 0,
            "direct_continuation_poll_elisions": 0,
            "direct_composite_attempts": 0,
            "direct_composite_completions": 0,
            "direct_composite_fallbacks": 0,
            "direct_composite_failures": 0,
            "direct_composite_callbacks_attempted": 0,
            "direct_composite_callbacks_completed": 0,
            "compute_dispatches": 3,
            "compute_admissions_run_closing": 0,
            "activations_completed": 3,
            "channel_prepare_calls": 3,
            "channel_emit_attempts": 1,
            "channel_emit_accepted": 1,
            "channel_emit_backpressured": 0,
            "channel_samples_received": 1,
            "channel_readiness_published": 1,
            "channel_input_reads": 1,
            "channel_witness_retired": 1,
            "channel_retire_calls": 3,
            "channel_activation_cuts": 0,
            "channel_activation_cut_drains": 0,
            "channel_activation_cut_skips": 0,
            "channel_activation_cut_scoped_retries": 0,
            "channel_receive_backpressured": 0,
            "channel_poll_incomplete": 0,
            "channel_activation_cut_inventory_scans": 0,
            "channel_activation_cut_subscriber_probes": 0,
            "channel_activation_cut_transport_probes": 0,
            "channel_activation_cut_materializations": 0,
            "channel_pump_waits": 0,
            "channel_pump_waits_ready": 0,
            "channel_pump_wait_timeouts": 0,
            "channel_pump_wait_timeouts_with_blocked_head": 0,
            "channel_scan_lock_contended_cut": 0,
            "channel_scan_lock_wait_ns_cut": 0,
            "channel_blocked_heads_masked": 0,
            "channel_pump_wakes_signaled": 0,
            "candidate_source_lookup_probes": 0,
            "candidate_order_comparisons": 0,
            "candidate_order_moves": 0,
            "candidate_tag_time_reservations": 0,
            "lane_dispatches": 6,
            "lane_compute_dispatches": 3,
            "lane_dispatcher_dispatches": 1,
            "lane_lifecycle_dispatches": 2,
            "lane_unknown_dispatches": 0,
            "lane_busy_task_publications": 0,
            "lane_sleeping_wake_signals": 6,
            "lane_callbacks_entered": 6,
            "lane_callbacks_exited": 6,
            "lane_completions_observed": 6,
            "lane_serialization_contentions": 0,
            "lane_inline_owner_invocations": 0,
        }
        args = make_args(runtime_phase_counters=True)
        runner.validate_runtime_phase_counters(counters, counts, args)

        # Stop may reject a dispatched candidate before its Operator callback claims admission.
        # Its prepare/retire and lane dispatch are real, but no callback or activation completes.
        stopped = {
            **counters,
            "candidates_frozen": 4,
            "candidates_selected": 4,
            "compute_dispatches": 4,
            "compute_admissions_run_closing": 1,
            "channel_prepare_calls": 4,
            "channel_retire_calls": 4,
            "lane_dispatches": 7,
            "lane_compute_dispatches": 4,
            "lane_sleeping_wake_signals": 7,
            "lane_callbacks_entered": 7,
            "lane_callbacks_exited": 7,
            "lane_completions_observed": 7,
        }
        runner.validate_runtime_phase_counters(stopped, counts, args)
        with pytest.raises(RuntimeError, match="inconsistent Runtime phase counters"):
            runner.validate_runtime_phase_counters(
                {**stopped, "compute_admissions_run_closing": 0}, counts, args
            )
        with pytest.raises(RuntimeError, match="inconsistent Runtime phase counters"):
            runner.validate_runtime_phase_counters(
                {**stopped, "compute_admissions_run_closing": 2}, counts, args
            )

        with pytest.raises(RuntimeError, match="inconsistent Runtime phase counters"):
            runner.validate_runtime_phase_counters({**stopped, "stop_signals": 0}, counts, args)

        # Key setup: a cross-partition drop-oldest source can publish two retained revisions while
        # the sink reads only the surviving sample. Retention/readiness therefore remain bounded
        # by accepted emits and reads rather than equaling the read count.
        lossy_counts = {
            **counts,
            "source_callbacks": 2,
            "source_compute_entries": 3,
            "sink_callbacks": 1,
            "publications": 2,
        }
        lossy_counters = {
            **counters,
            "readiness_scans": 4,
            "candidates_frozen": 4,
            "candidates_selected": 4,
            "compute_dispatches": 4,
            "compute_admissions_run_closing": 0,
            "activations_completed": 4,
            "channel_prepare_calls": 4,
            "channel_emit_attempts": 2,
            "channel_emit_accepted": 2,
            "channel_samples_received": 2,
            "channel_readiness_published": 2,
            "channel_retire_calls": 4,
            "lane_dispatches": 7,
            "lane_compute_dispatches": 4,
            "lane_sleeping_wake_signals": 7,
            "lane_callbacks_entered": 7,
            "lane_callbacks_exited": 7,
            "lane_completions_observed": 7,
        }
        runner.validate_runtime_phase_counters(
            lossy_counters,
            lossy_counts,
            make_args(runtime_phase_counters=True, queue_policy="drop_oldest"),
        )

        # Key assertion: lossy retention cannot fall below reads or exceed accepted publications.
        lossy_counters["channel_samples_received"] = 0
        with pytest.raises(RuntimeError, match="inconsistent Runtime phase counters"):
            runner.validate_runtime_phase_counters(
                lossy_counters,
                lossy_counts,
                make_args(runtime_phase_counters=True, queue_policy="drop_oldest"),
            )

        inline_counters = dict(counters)
        inline_counters.update(
            {
                "lane_dispatches": 0,
                "lane_compute_dispatches": 0,
                "lane_dispatcher_dispatches": 0,
                "lane_lifecycle_dispatches": 0,
                "lane_sleeping_wake_signals": 0,
                "lane_callbacks_entered": 6,
                "lane_callbacks_exited": 6,
                "lane_completions_observed": 0,
                "lane_inline_owner_invocations": 6,
            }
        )
        runner.validate_runtime_phase_counters(inline_counters, counts, args)

        # Key setup: a source-only row retains the same scheduler and lane evidence but performs
        # no channel operation of any kind.
        source_only_counts = {
            **counts,
            "source_compute_entries": 3,
            "sink_callbacks": 0,
            "publications": 0,
        }
        source_only_counters = {
            **counters,
            **{name: 0 for name in counters if name.startswith("channel_")},
            "lane_compute_dispatches": 3,
        }
        source_only_args = make_args(
            scenario="source_only", workload="saturated", runtime_phase_counters=True
        )
        runner.validate_runtime_phase_counters(
            source_only_counters, source_only_counts, source_only_args
        )

        # Key assertion: every channel counter, including accepted emits, retirement, and
        # activation cuts, fails closed when fabricated in source-only evidence.
        for name in (counter for counter in counters if counter.startswith("channel_")):
            invalid_source_only_counters = {**source_only_counters, name: 1}
            with pytest.raises(RuntimeError, match="inconsistent Runtime phase counters"):
                runner.validate_runtime_phase_counters(
                    invalid_source_only_counters, source_only_counts, source_only_args
                )

        # Key assertion: a missing completion cannot enter an aggregate benchmark summary.
        counters["activations_completed"] = 2
        with pytest.raises(RuntimeError, match="inconsistent Runtime phase counters"):
            runner.validate_runtime_phase_counters(counters, counts, args)

        counters["activations_completed"] = 3
        counters["direct_continuation_receipts_accepted"] = 1
        with pytest.raises(RuntimeError, match="inconsistent Runtime phase counters"):
            runner.validate_runtime_phase_counters(counters, counts, args)

    def test_pressured_staged_diagnostics_require_attributed_depth_one_drain(self) -> None:
        # Key setup: model the one forced-HoloMQ depth-one POD diagnostic after its first gated
        # delivery, two later staged replacements, and the final gated drain.
        counts = {
            "source_compute_entries": 4,
            "relay_callbacks": 0,
            "sink_callbacks": 2,
            "publications": 4,
            "relay_publications": 0,
            "backpressured_emits": 0,
            "consumer_visible_replacements": 2,
            "deliveries": 2,
            "activation_completions": 2,
        }
        counters = {name: 0 for name in runner.RUNTIME_PHASE_COUNTER_NAMES}
        counters.update(
            {
                "enabled": True,
                "readiness_scans": 6,
                "no_progress_scans": 1,
                "wait_calls": 1,
                "wake_signals": 1,
                "wake_returns": 1,
                "stop_signals": 1,
                "candidates_frozen": 6,
                "candidates_selected": 6,
                "readiness_poll_dispatches": 2,
                "compute_dispatches": 6,
                "compute_admissions_run_closing": 0,
                "activations_completed": 6,
                "channel_prepare_calls": 6,
                "channel_emit_attempts": 4,
                "channel_emit_accepted": 4,
                "channel_samples_received": 3,
                "channel_readiness_published": 3,
                "channel_input_reads": 2,
                "channel_witness_retired": 2,
                "channel_retire_calls": 6,
                "channel_activation_cuts": 6,
                "channel_activation_cut_skips": 6,
                "channel_activation_cut_scoped_retries": 0,
                "lane_dispatches": 10,
                "lane_compute_dispatches": 6,
                "lane_dispatcher_dispatches": 2,
                "lane_lifecycle_dispatches": 2,
                "lane_sleeping_wake_signals": 10,
                "lane_callbacks_entered": 10,
                "lane_callbacks_exited": 10,
                "lane_completions_observed": 10,
            }
        )
        counters.update(
            {
                name: 0
                for name in (
                    runner.PRESSURED_STAGED_DIAGNOSTIC_COUNTER_NAMES
                    + runner.PRESSURED_STAGED_EXISTING_ATTRIBUTION_COUNTER_NAMES
                )
            }
        )
        counters.update(
            {
                "channel_direct_ingress_promotion_passes": 8,
                "channel_direct_ingress_promotions": 3,
                "channel_staged_evidence_sweep_visits": 12,
                "channel_pre_witness_replacements": 1,
                "temporal_retention_candidates": 2,
                "temporal_retention_evictions": 1,
                "temporal_reuse_last_retained_record_queries": 2,
            }
        )
        args = make_args(
            workload="saturated",
            queue_policy="drop_oldest",
            runtime_phase_counters=True,
            force_holomq_process_local_inline=True,
            test_gate_pod_sink_for_pressure=True,
        )

        # Key assertion: the diagnostic schema preserves positive staged traversal, ReuseLast
        # handoff, replacement accounting, and final-data delivery. The protected-terminal probe
        # and depth-one comparator remain required but nonnegative because exact scheduling and
        # claim cardinality decide whether this row observes them.
        assert runner.pressured_staged_diagnostics_required(args)
        runner.validate_runtime_phase_counters(counters, counts, args)
        # A private fallback can let all three revisions become retired witnesses, although only
        # two payloads are read. Otherwise the extra revision is replaced before publication.
        runner.validate_runtime_phase_counters(
            {**counters, "channel_witness_retired": 3, "channel_pre_witness_replacements": 0},
            counts,
            args,
        )
        for retired in (1, 3):
            with pytest.raises(RuntimeError, match="inconsistent Runtime phase counters"):
                runner.validate_runtime_phase_counters(
                    {**counters, "channel_witness_retired": retired}, counts, args
                )
        runner.validate_runtime_phase_counters(
            {**counters, "channel_protected_terminal_probes_with_pending_data": 1}, counts, args
        )
        runner.validate_runtime_phase_counters(
            {**counters, "temporal_reuse_last_comparator_calls": 1}, counts, args
        )

        for name in runner.PRESSURED_STAGED_DIAGNOSTIC_COUNTER_NAMES:
            incomplete = dict(counters)
            incomplete.pop(name)
            with pytest.raises(RuntimeError, match="incomplete Runtime phase counters"):
                runner.validate_runtime_phase_counters(incomplete, counts, args)

        for name in (
            "channel_direct_ingress_promotion_passes",
            "channel_direct_ingress_promotions",
            "channel_staged_evidence_sweep_visits",
            "temporal_retention_candidates",
            "temporal_retention_evictions",
            "temporal_reuse_last_retained_record_queries",
        ):
            unexercised = {**counters, name: 0}
            with pytest.raises(RuntimeError, match="inconsistent Runtime phase counters"):
                runner.validate_runtime_phase_counters(unexercised, counts, args)

        without_replacement = {**counts, "consumer_visible_replacements": 0}
        with pytest.raises(RuntimeError, match="inconsistent Runtime phase counters"):
            runner.validate_runtime_phase_counters(counters, without_replacement, args)

        without_final_data_delivery = {**counts, "activation_completions": 0}
        with pytest.raises(RuntimeError, match="inconsistent Runtime phase counters"):
            runner.validate_runtime_phase_counters(counters, without_final_data_delivery, args)

    @pytest.mark.parametrize("name", runner.RUNTIME_PHASE_COUNTER_NAMES)
    @pytest.mark.parametrize("invalid_kind", ["missing", "negative", "boolean"])
    def test_runtime_phase_counters_require_nonnegative_integer_fields(
        self, name: str, invalid_kind: str
    ) -> None:
        # Key setup: start from one complete ordinary-dispatch source/sink cycle, then corrupt one
        # required field using each JSON shape that previously evaded the evidence contract.
        counts = {
            "source_compute_entries": 2,
            "relay_callbacks": 0,
            "sink_callbacks": 1,
            "publications": 1,
            "relay_publications": 0,
            "backpressured_emits": 0,
        }
        counters = {counter: 0 for counter in runner.RUNTIME_PHASE_COUNTER_NAMES}
        counters.update(
            {
                "enabled": True,
                "readiness_scans": 3,
                "no_progress_scans": 1,
                "wait_calls": 1,
                "wake_signals": 1,
                "wake_returns": 1,
                "stop_signals": 1,
                "candidates_frozen": 3,
                "candidates_selected": 3,
                "readiness_poll_dispatches": 1,
                "compute_dispatches": 3,
                "compute_admissions_run_closing": 0,
                "activations_completed": 3,
                "channel_prepare_calls": 3,
                "channel_emit_attempts": 1,
                "channel_emit_accepted": 1,
                "channel_samples_received": 1,
                "channel_readiness_published": 1,
                "channel_input_reads": 1,
                "channel_witness_retired": 1,
                "channel_retire_calls": 3,
                "lane_dispatches": 6,
                "lane_compute_dispatches": 3,
                "lane_dispatcher_dispatches": 1,
                "lane_lifecycle_dispatches": 2,
                "lane_sleeping_wake_signals": 6,
                "lane_callbacks_entered": 6,
                "lane_callbacks_exited": 6,
                "lane_completions_observed": 6,
            }
        )
        if invalid_kind == "missing":
            counters.pop(name)
        else:
            counters[name] = -1 if invalid_kind == "negative" else False

        # Key assertion: every required counter rejects omission, negative values, and booleans.
        with pytest.raises(RuntimeError, match="Runtime phase counters"):
            runner.validate_runtime_phase_counters(
                counters, counts, make_args(runtime_phase_counters=True)
            )

    def test_runtime_phase_trace_requires_complete_bounded_activations(self) -> None:
        # Key setup: source and sink receivers retain all four controller timestamps for one
        # complete notification-driven publication.
        counts = {
            "source_compute_entries": 1,
            "relay_callbacks": 0,
            "sink_callbacks": 1,
        }
        complete_timing = {"samples": 1, "p50": 100}
        phase_trace = {
            "enabled": True,
            "valid": True,
            "records": 9,
            "dropped": 0,
            "readiness_polls": 1,
            "unmatched_readiness_polls": 0,
            "complete_activations": 2,
            "terminal_frozen_candidates": 0,
            "terminal_run_closing_dispatches": 0,
            "source_activation_runs": 1,
            "maximum_consecutive_source_activations": 1,
            "source_to_source_transitions": 0,
            "source_to_downstream_transitions": 1,
            "downstream_to_source_transitions": 0,
            "downstream_to_downstream_transitions": 0,
            "complete_tensor_emits": 0,
            "tensor_emit_total_ns": None,
            "tensor_emit_validation_ns": None,
            "tensor_pre_transport_ns": None,
            "tensor_transport_publication_ns": None,
            "tensor_post_transport_ns": None,
            "complete_inline_payload_emits": 0,
            "inline_payload_emit_total_ns": None,
            "inline_payload_encoding_ns": None,
            "inline_payload_pre_transport_ns": None,
            "inline_payload_transport_publication_ns": None,
            "inline_payload_post_transport_ns": None,
            "receivers": [
                {
                    "activations": 1,
                    "candidate_to_dispatch_start_ns": complete_timing,
                    "compute_dispatch_ns": complete_timing,
                    "dispatch_return_to_retirement_ns": complete_timing,
                    "retirement_to_cut_ns": complete_timing,
                    "dispatch_return_to_cut_ns": complete_timing,
                },
                {
                    "activations": 1,
                    "candidate_to_dispatch_start_ns": complete_timing,
                    "compute_dispatch_ns": complete_timing,
                    "dispatch_return_to_retirement_ns": complete_timing,
                    "retirement_to_cut_ns": complete_timing,
                    "dispatch_return_to_cut_ns": complete_timing,
                },
            ],
        }
        args = make_args(runtime_phase_counters=True)
        runner.validate_runtime_phase_trace(phase_trace, counts, args)

        # Key setup: the matching counters identify one final candidate freeze after both complete
        # source and sink activations.
        phase_trace["terminal_frozen_candidates"] = 1
        terminal_counters = {"candidates_frozen": 3, "candidates_selected": 2}
        runner.validate_runtime_phase_trace(phase_trace, counts, args, terminal_counters)

        # Key assertion: the trace allowance must match the exact counter surplus and never exceed
        # one candidate.
        terminal_counters["candidates_frozen"] = 2
        with pytest.raises(RuntimeError, match="inconsistent terminal Runtime phase state"):
            runner.validate_runtime_phase_trace(phase_trace, counts, args, terminal_counters)
        phase_trace["terminal_frozen_candidates"] = 2
        with pytest.raises(RuntimeError, match="inconsistent Runtime phase trace"):
            runner.validate_runtime_phase_trace(phase_trace, counts, args)
        phase_trace["terminal_frozen_candidates"] = 0

        # A selected terminal dispatch rejected by stop is distinct from an unselected freeze.
        phase_trace["terminal_run_closing_dispatches"] = 1
        stopped_counters = {
            "candidates_frozen": 3,
            "candidates_selected": 3,
            "compute_admissions_run_closing": 1,
            "stop_signals": 1,
        }
        runner.validate_runtime_phase_trace(phase_trace, counts, args, stopped_counters)
        for bad_counters in (
            {**stopped_counters, "compute_admissions_run_closing": 0},
            {**stopped_counters, "stop_signals": 0},
        ):
            with pytest.raises(RuntimeError, match="inconsistent terminal Runtime phase state"):
                runner.validate_runtime_phase_trace(phase_trace, counts, args, bad_counters)
        with pytest.raises(RuntimeError, match="missing stop-admission evidence"):
            runner.validate_runtime_phase_trace(phase_trace, counts, args)
        for bad_count in (2, -1, True):
            phase_trace["terminal_run_closing_dispatches"] = bad_count
            with pytest.raises(RuntimeError, match="inconsistent Runtime phase trace"):
                runner.validate_runtime_phase_trace(phase_trace, counts, args, stopped_counters)
        phase_trace["terminal_run_closing_dispatches"] = 1
        phase_trace["terminal_frozen_candidates"] = 1
        with pytest.raises(RuntimeError, match="inconsistent Runtime phase trace"):
            runner.validate_runtime_phase_trace(phase_trace, counts, args, stopped_counters)
        phase_trace["terminal_frozen_candidates"] = 0
        phase_trace["terminal_run_closing_dispatches"] = 0

        # Key assertion: storage overflow cannot silently yield a partial diagnostic breakdown.
        phase_trace["dropped"] = 1
        with pytest.raises(RuntimeError, match="inconsistent Runtime phase trace"):
            runner.validate_runtime_phase_trace(phase_trace, counts, args)

    def test_runtime_phase_trace_keeps_bounded_storage_for_concurrent_partitions(self) -> None:
        counts = {
            "source_compute_entries": 1,
            "relay_callbacks": 0,
            "sink_callbacks": 1,
        }
        phase_trace = {
            "enabled": True,
            "activation_trace_applicable": False,
            "valid": True,
            "capacity_exhausted": False,
            "storage_invariants_valid": True,
            "capacity": 100,
            "attempted_records": 12,
            "records": 12,
            "dropped": 0,
            "unmatched_readiness_polls": 0,
            "complete_activations": 0,
            "terminal_frozen_candidates": 0,
            "terminal_run_closing_dispatches": 0,
            "receivers": [],
        }
        runner.validate_runtime_phase_trace(
            phase_trace, counts, make_args(runtime_phase_counters=True)
        )

        phase_trace["dropped"] = 1
        with pytest.raises(RuntimeError, match="bounded concurrent-partition trace"):
            runner.validate_runtime_phase_trace(
                phase_trace, counts, make_args(runtime_phase_counters=True)
            )

    def test_runtime_phase_trace_requires_complete_tensor_publication(self) -> None:
        # Key setup: a forced-HoloMQ Tensor publication has all five paired publication phases.
        counts = {
            "source_compute_entries": 1,
            "relay_callbacks": 0,
            "sink_callbacks": 1,
            "publications": 1,
        }
        complete_timing = {"samples": 1, "p50": 100}
        phase_trace = {
            "enabled": True,
            "valid": True,
            "records": 14,
            "dropped": 0,
            "readiness_polls": 1,
            "unmatched_readiness_polls": 0,
            "complete_activations": 2,
            "terminal_frozen_candidates": 0,
            "terminal_run_closing_dispatches": 0,
            "source_activation_runs": 1,
            "maximum_consecutive_source_activations": 1,
            "source_to_source_transitions": 0,
            "source_to_downstream_transitions": 1,
            "downstream_to_source_transitions": 0,
            "downstream_to_downstream_transitions": 0,
            "complete_tensor_emits": 1,
            "tensor_emit_total_ns": complete_timing,
            "tensor_emit_validation_ns": complete_timing,
            "tensor_pre_transport_ns": complete_timing,
            "tensor_transport_publication_ns": complete_timing,
            "tensor_post_transport_ns": complete_timing,
            "complete_inline_payload_emits": 0,
            "inline_payload_emit_total_ns": None,
            "inline_payload_encoding_ns": None,
            "inline_payload_pre_transport_ns": None,
            "inline_payload_transport_publication_ns": None,
            "inline_payload_post_transport_ns": None,
            "receivers": [
                {
                    "activations": 1,
                    "candidate_to_dispatch_start_ns": complete_timing,
                    "compute_dispatch_ns": complete_timing,
                    "dispatch_return_to_retirement_ns": complete_timing,
                    "retirement_to_cut_ns": complete_timing,
                    "dispatch_return_to_cut_ns": complete_timing,
                },
                {
                    "activations": 1,
                    "candidate_to_dispatch_start_ns": complete_timing,
                    "compute_dispatch_ns": complete_timing,
                    "dispatch_return_to_retirement_ns": complete_timing,
                    "retirement_to_cut_ns": complete_timing,
                    "dispatch_return_to_cut_ns": complete_timing,
                },
            ],
        }
        args = make_args(
            scenario="tensor",
            runtime_phase_counters=True,
            force_holomq_process_local_inline=True,
        )
        runner.validate_runtime_phase_trace(phase_trace, counts, args)

        # Key assertion: an unpaired transport phase cannot enter an aggregate summary.
        phase_trace["tensor_transport_publication_ns"] = None
        with pytest.raises(RuntimeError, match="incomplete Runtime Tensor phase trace"):
            runner.validate_runtime_phase_trace(phase_trace, counts, args)

        # Key assertion: an unpaired batched readiness poll also invalidates the breakdown.
        phase_trace["tensor_transport_publication_ns"] = complete_timing
        phase_trace["dropped"] = 0
        phase_trace["unmatched_readiness_polls"] = 1
        with pytest.raises(RuntimeError, match="inconsistent Runtime phase trace"):
            runner.validate_runtime_phase_trace(phase_trace, counts, args)

    def test_runtime_phase_trace_requires_complete_inline_publication(self) -> None:
        # Key setup: a forced-HoloMQ schema publication has all five paired inline phases.
        counts = {
            "source_compute_entries": 1,
            "relay_callbacks": 0,
            "relay_publications": 0,
            "sink_callbacks": 1,
            "publications": 1,
        }
        complete_timing = {"samples": 1, "p50": 100}
        phase_trace = {
            "enabled": True,
            "valid": True,
            "records": 14,
            "dropped": 0,
            "readiness_polls": 1,
            "unmatched_readiness_polls": 0,
            "complete_activations": 2,
            "terminal_frozen_candidates": 0,
            "terminal_run_closing_dispatches": 0,
            "source_activation_runs": 1,
            "maximum_consecutive_source_activations": 1,
            "source_to_source_transitions": 0,
            "source_to_downstream_transitions": 1,
            "downstream_to_source_transitions": 0,
            "downstream_to_downstream_transitions": 0,
            "complete_tensor_emits": 0,
            "tensor_emit_total_ns": None,
            "tensor_emit_validation_ns": None,
            "tensor_pre_transport_ns": None,
            "tensor_transport_publication_ns": None,
            "tensor_post_transport_ns": None,
            "complete_inline_payload_emits": 1,
            "inline_payload_emit_total_ns": complete_timing,
            "inline_payload_encoding_ns": complete_timing,
            "inline_payload_pre_transport_ns": complete_timing,
            "inline_payload_transport_publication_ns": complete_timing,
            "inline_payload_post_transport_ns": complete_timing,
            "receivers": [
                {
                    "activations": 1,
                    "candidate_to_dispatch_start_ns": complete_timing,
                    "compute_dispatch_ns": complete_timing,
                    "dispatch_return_to_retirement_ns": complete_timing,
                    "retirement_to_cut_ns": complete_timing,
                    "dispatch_return_to_cut_ns": complete_timing,
                },
                {
                    "activations": 1,
                    "candidate_to_dispatch_start_ns": complete_timing,
                    "compute_dispatch_ns": complete_timing,
                    "dispatch_return_to_retirement_ns": complete_timing,
                    "retirement_to_cut_ns": complete_timing,
                    "dispatch_return_to_cut_ns": complete_timing,
                },
            ],
        }
        args = make_args(
            scenario="schema",
            runtime_phase_counters=True,
            force_holomq_process_local_inline=True,
        )
        runner.validate_runtime_phase_trace(phase_trace, counts, args)

        # Key assertion: an unpaired inline transport phase invalidates the breakdown.
        phase_trace["inline_payload_transport_publication_ns"] = None
        with pytest.raises(RuntimeError, match="incomplete Runtime inline-payload phase trace"):
            runner.validate_runtime_phase_trace(phase_trace, counts, args)

    def test_runtime_phase_trace_rejects_missing_activation_transition(self) -> None:
        # Key setup: three activations encode one source run followed by two downstream callbacks.
        counts = {
            "source_compute_entries": 1,
            "relay_callbacks": 1,
            "sink_callbacks": 1,
            "publications": 1,
        }
        complete_timing = {"samples": 1, "p50": 100}
        phase_trace = {
            "enabled": True,
            "valid": True,
            "records": 15,
            "dropped": 0,
            "readiness_polls": 2,
            "unmatched_readiness_polls": 0,
            "complete_activations": 3,
            "terminal_frozen_candidates": 0,
            "terminal_run_closing_dispatches": 0,
            "source_activation_runs": 1,
            "maximum_consecutive_source_activations": 1,
            "source_to_source_transitions": 0,
            "source_to_downstream_transitions": 1,
            "downstream_to_source_transitions": 0,
            "downstream_to_downstream_transitions": 1,
            "complete_tensor_emits": 0,
            "tensor_emit_total_ns": None,
            "tensor_emit_validation_ns": None,
            "tensor_pre_transport_ns": None,
            "tensor_transport_publication_ns": None,
            "tensor_post_transport_ns": None,
            "complete_inline_payload_emits": 1,
            "inline_payload_emit_total_ns": complete_timing,
            "inline_payload_encoding_ns": complete_timing,
            "inline_payload_pre_transport_ns": None,
            "inline_payload_transport_publication_ns": None,
            "inline_payload_post_transport_ns": None,
            "receivers": [
                {
                    "activations": 1,
                    "candidate_to_selection_ns": complete_timing,
                    "selection_to_input_preparation_ns": complete_timing,
                    "input_preparation_to_activation_ns": complete_timing,
                    "activation_to_dispatch_start_ns": complete_timing,
                    "candidate_to_dispatch_start_ns": complete_timing,
                    "dispatch_start_to_callback_entry_ns": complete_timing,
                    "callback_entry_to_return_ns": complete_timing,
                    "callback_return_to_dispatch_return_ns": complete_timing,
                    "compute_dispatch_ns": complete_timing,
                    "dispatch_return_to_retirement_ns": complete_timing,
                    "retirement_to_cut_ns": complete_timing,
                    "dispatch_return_to_cut_ns": complete_timing,
                },
                {
                    "activations": 2,
                    "candidate_to_selection_ns": {"samples": 2, "p50": 100},
                    "selection_to_input_preparation_ns": {"samples": 2, "p50": 100},
                    "input_preparation_to_activation_ns": {"samples": 2, "p50": 100},
                    "activation_to_dispatch_start_ns": {"samples": 2, "p50": 100},
                    "candidate_to_dispatch_start_ns": {"samples": 2, "p50": 100},
                    "dispatch_start_to_callback_entry_ns": {"samples": 2, "p50": 100},
                    "callback_entry_to_return_ns": {"samples": 2, "p50": 100},
                    "callback_return_to_dispatch_return_ns": {"samples": 2, "p50": 100},
                    "compute_dispatch_ns": {"samples": 2, "p50": 100},
                    "dispatch_return_to_retirement_ns": {"samples": 2, "p50": 100},
                    "retirement_to_cut_ns": {"samples": 2, "p50": 100},
                    "dispatch_return_to_cut_ns": {"samples": 2, "p50": 100},
                },
            ],
        }
        args = make_args(scenario="sample_forward", runtime_phase_counters=True)
        runner.validate_runtime_phase_trace(phase_trace, counts, args)

        # Key assertion: every general-chain activation must retain a complete preparation
        # breakdown before transition cardinality is considered.
        phase_trace["receivers"][1]["input_preparation_to_activation_ns"] = None
        with pytest.raises(RuntimeError, match="incomplete Runtime activation-preparation trace"):
            runner.validate_runtime_phase_trace(phase_trace, counts, args)
        phase_trace["receivers"][1]["input_preparation_to_activation_ns"] = {
            "samples": 2,
            "p50": 100,
        }
        phase_trace["receivers"][1]["callback_entry_to_return_ns"] = None
        with pytest.raises(RuntimeError, match="incomplete Runtime callback phase trace"):
            runner.validate_runtime_phase_trace(phase_trace, counts, args)
        phase_trace["receivers"][1]["callback_entry_to_return_ns"] = {
            "samples": 2,
            "p50": 100,
        }

        # Key assertion: dropping one transition makes the activation sequence incomplete.
        phase_trace["downstream_to_downstream_transitions"] = 0
        with pytest.raises(RuntimeError, match="inconsistent Runtime phase trace"):
            runner.validate_runtime_phase_trace(phase_trace, counts, args)

    def test_shared_lane_callback_threads_report_worker_or_partition_owner(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Key setup: ordinary source and sink callbacks share one lane worker while the controller
        # is independently identified.
        counts = {
            "source_compute_entries": 11,
            "relay_callbacks": 0,
            "sink_callbacks": 11,
            "per_sink": [{"index": 0, "deliveries": 11}],
        }
        evidence = {
            "enabled": True,
            "valid": True,
            "thread_identity_kind": (
                "native_tid" if sys.platform.startswith("linux") else "std_thread_id_hash"
            ),
            "native_tid_syscall_unavailable": False,
            "controller_native_tid": 17,
            "shared_lane_single_thread": True,
            "callbacks_on_partition_owner": False,
            "roles": [
                {
                    "role": "source",
                    "callbacks": 11,
                    "native_tid": 23,
                    "multiple_threads": False,
                },
                {
                    "role": "relay",
                    "callbacks": 0,
                    "native_tid": None,
                    "multiple_threads": False,
                },
                {
                    "role": "sink",
                    "callbacks": 11,
                    "native_tid": 23,
                    "multiple_threads": False,
                },
            ],
            "sink_roles": [
                {
                    "index": 0,
                    "callbacks": 11,
                    "native_tid": 23,
                    "multiple_threads": False,
                }
            ],
        }
        args = make_args(runtime_phase_counters=True, operator_lanes="shared")
        runner.validate_callback_thread_evidence(evidence, counts, args)

        # Key assertion: a controller/lane identity collision or role migration fails closed.
        evidence["controller_native_tid"] = 23
        with pytest.raises(RuntimeError, match="shared-lane"):
            runner.validate_callback_thread_evidence(evidence, counts, args)
        evidence["controller_native_tid"] = 17
        evidence["roles"][2]["native_tid"] = 29
        with pytest.raises(RuntimeError, match="shared-lane"):
            runner.validate_callback_thread_evidence(evidence, counts, args)

        # Inline partition execution deliberately runs the same shared callback lane on its owner.
        evidence["roles"][2]["native_tid"] = 23
        evidence["controller_native_tid"] = 23
        evidence["callbacks_on_partition_owner"] = True
        runner.validate_callback_thread_evidence(evidence, counts, args)
        evidence["callbacks_on_partition_owner"] = False
        with pytest.raises(RuntimeError, match="shared-lane"):
            runner.validate_callback_thread_evidence(evidence, counts, args)

        evidence["callbacks_on_partition_owner"] = True
        evidence["thread_identity_kind"] = "std_thread_id_hash"
        evidence["native_tid_syscall_unavailable"] = False
        monkeypatch.setattr(runner.sys, "platform", "darwin")
        runner.validate_callback_thread_evidence(evidence, counts, args)
        evidence["controller_native_tid"] = 17
        with pytest.raises(RuntimeError, match="shared-lane"):
            runner.validate_callback_thread_evidence(evidence, counts, args)

        # Linux accepts a process-local hash only with explicit syscall-unavailable evidence.
        evidence["controller_native_tid"] = 23
        monkeypatch.setattr(runner.sys, "platform", "linux")
        with pytest.raises(RuntimeError, match="callback-thread"):
            runner.validate_callback_thread_evidence(evidence, counts, args)
        evidence["native_tid_syscall_unavailable"] = True
        runner.validate_callback_thread_evidence(evidence, counts, args)

        # Disabled instrumentation cannot silently carry callback-thread evidence.
        disabled_args = make_args(runtime_phase_counters=False, operator_lanes="shared")
        with pytest.raises(RuntimeError, match="unexpectedly enabled"):
            runner.validate_callback_thread_evidence(evidence, counts, disabled_args)
        runner.validate_callback_thread_evidence({"enabled": False}, counts, disabled_args)

    def test_fanout_callback_threads_require_stable_distinct_partition_owners(self) -> None:
        counts = {
            "source_compute_entries": 11,
            "relay_callbacks": 0,
            "sink_callbacks": 22,
            "per_sink": [
                {"index": 0, "deliveries": 11},
                {"index": 1, "deliveries": 11},
            ],
        }
        evidence = {
            "enabled": True,
            "valid": True,
            "thread_identity_kind": (
                "native_tid" if sys.platform.startswith("linux") else "std_thread_id_hash"
            ),
            "native_tid_syscall_unavailable": False,
            "controller_native_tid": 17,
            "shared_lane_single_thread": False,
            "callbacks_on_partition_owner": False,
            "roles": [
                {
                    "role": "source",
                    "callbacks": 11,
                    "native_tid": 23,
                    "multiple_threads": False,
                },
                {
                    "role": "relay",
                    "callbacks": 0,
                    "native_tid": None,
                    "multiple_threads": False,
                },
                {
                    "role": "sink",
                    "callbacks": 22,
                    "native_tid": 29,
                    "multiple_threads": True,
                },
            ],
            "sink_roles": [
                {"index": 0, "callbacks": 11, "native_tid": 29, "multiple_threads": False},
                {"index": 1, "callbacks": 11, "native_tid": 31, "multiple_threads": False},
            ],
        }
        args = make_args(runtime_phase_counters=True, fanout=2, operator_lanes="separate")
        runner.validate_callback_thread_evidence(evidence, counts, args)

        evidence["sink_roles"][1]["native_tid"] = 29
        with pytest.raises(RuntimeError, match="per-partition"):
            runner.validate_callback_thread_evidence(evidence, counts, args)
        evidence["sink_roles"][1]["native_tid"] = 31
        evidence["sink_roles"][1]["multiple_threads"] = True
        with pytest.raises(RuntimeError, match="per-sink"):
            runner.validate_callback_thread_evidence(evidence, counts, args)

    def test_source_only_callback_threads_require_no_sink_records(self) -> None:
        counts = {
            "source_compute_entries": 11,
            "relay_callbacks": 0,
            "sink_callbacks": 0,
            "per_sink": [],
        }
        evidence = {
            "enabled": True,
            "valid": True,
            "thread_identity_kind": (
                "native_tid" if sys.platform.startswith("linux") else "std_thread_id_hash"
            ),
            "native_tid_syscall_unavailable": False,
            "controller_native_tid": 17,
            "shared_lane_single_thread": True,
            "callbacks_on_partition_owner": True,
            "roles": [
                {
                    "role": "source",
                    "callbacks": 11,
                    "native_tid": 17,
                    "multiple_threads": False,
                },
                {
                    "role": "relay",
                    "callbacks": 0,
                    "native_tid": None,
                    "multiple_threads": False,
                },
                {
                    "role": "sink",
                    "callbacks": 0,
                    "native_tid": None,
                    "multiple_threads": False,
                },
            ],
            "sink_roles": [],
        }
        args = make_args(
            scenario="source_only",
            workload="notification_latency",
            runtime_phase_counters=True,
        )

        runner.validate_callback_thread_evidence(evidence, counts, args)

    def test_multi_input_callback_threads_describe_one_logical_sink_not_two_input_legs(
        self,
    ) -> None:
        counts = {
            "source_compute_entries": 11,
            "relay_callbacks": 0,
            "sink_callbacks": 11,
            "per_sink": [
                {"index": 0, "deliveries": 11},
                {"index": 1, "deliveries": 11},
            ],
        }
        evidence = {
            "enabled": True,
            "valid": True,
            "thread_identity_kind": (
                "native_tid" if sys.platform.startswith("linux") else "std_thread_id_hash"
            ),
            "native_tid_syscall_unavailable": False,
            "controller_native_tid": 23,
            "shared_lane_single_thread": True,
            "callbacks_on_partition_owner": True,
            "roles": [
                {
                    "role": "source",
                    "callbacks": 11,
                    "native_tid": 23,
                    "multiple_threads": False,
                },
                {
                    "role": "relay",
                    "callbacks": 0,
                    "native_tid": None,
                    "multiple_threads": False,
                },
                {
                    "role": "sink",
                    "callbacks": 11,
                    "native_tid": 23,
                    "multiple_threads": False,
                },
            ],
            "sink_roles": [
                {"index": 0, "callbacks": 11, "native_tid": 23, "multiple_threads": False}
            ],
        }
        args = make_args(
            runtime_phase_counters=True,
            operator_lanes="shared",
            multi_input=True,
            fanout=2,
        )

        runner.validate_callback_thread_evidence(evidence, counts, args)

        evidence["sink_roles"].append(
            {"index": 1, "callbacks": 11, "native_tid": 23, "multiple_threads": False}
        )
        with pytest.raises(RuntimeError, match="callback-thread"):
            runner.validate_callback_thread_evidence(evidence, counts, args)

    def test_runtime_thread_usage_requires_exact_role_population_and_aggregate(self) -> None:
        supported = sys.platform.startswith("linux")
        expected_samples = {
            "controller": 1,
            "execution_lanes": 0,
            "channel_pump": 0,
            "notification_driver": 1,
        }
        roles = [
            {
                "role": role,
                "supported": supported,
                "samples": samples if supported else 0,
                "user_cpu_ns": samples * 100 if supported else 0,
                "system_cpu_ns": samples * 10 if supported else 0,
                "voluntary_context_switches": samples if supported else 0,
                "involuntary_context_switches": 0,
                "minor_page_faults": samples if supported else 0,
                "major_page_faults": 0,
            }
            for role, samples in expected_samples.items()
        ]
        fields = (
            "samples",
            "user_cpu_ns",
            "system_cpu_ns",
            "voluntary_context_switches",
            "involuntary_context_switches",
            "minor_page_faults",
            "major_page_faults",
        )
        evidence = {
            "enabled": True,
            "supported": supported,
            "scope": "whole_runtime_thread_lifetime_including_startup_and_teardown",
            "roles": roles,
            "aggregate": {
                "scope": "sum_of_runtime_roles",
                **{field: sum(role[field] for role in roles) for field in fields},
            },
        }
        args = make_args(
            runtime_phase_counters=True,
            operator_lanes="shared",
            workload="notification_latency",
        )
        runner.validate_runtime_thread_usage(evidence, args)

        evidence["aggregate"]["user_cpu_ns"] += 1
        with pytest.raises(RuntimeError, match="aggregate"):
            runner.validate_runtime_thread_usage(evidence, args)

    def test_checked_result_rejects_incomplete_deliveries(self, tmp_path: Path) -> None:
        # Key setup: encode one fewer delivery than the requested latency sample count.
        args = make_args()
        result = {
            "schema": runner.RESULT_SCHEMA,
            "runtime_controls": {
                "direct_continuation_elision_disabled": False,
                "closed_island_composite_disabled": False,
                "holomq_event_wake_forced": False,
                "process_local_channel_pump_forced": False,
                "process_local_pump_delivery_forced": False,
            },
            "clock": clock_evidence(),
            "timestamp_instrumentation": timestamp_instrumentation(),
            "benchmark": "runtime_graph_pod_1to1",
            "scenario": "pod",
            "process_local_inline_delivery": "direct_retained",
            "queue_policy": "blocking",
            "workload": "latency",
            "source_trigger": "clock",
            "partitioning": partitioning_evidence(args),
            "geometry": {
                "concurrent_admission": False,
                "samples": 100,
                "warmup": 10,
                "callback_work_us": 0,
                "experimental_fanout": "disabled",
                "source_work_us": None,
                "sink_work_us": None,
                "emit_then_work": False,
                "burst_size": 1,
                "lane_fifo_priority": 0,
                "allow_pump_lane_cpu_sharing": False,
                "queue_depth": 1,
                "subscriber_queue_depths": [1],
                "subscriber_queue_policies": ["blocking"],
                "offered_burst_size": 1,
                "fanout": 1,
                "source_outputs": 1,
                "subscribers_per_output": 1,
                "sink_operators": 1,
                "inputs_per_sink": 1,
                "relay_count": 0,
                "pipeline_edges": 1,
                "notification_capacity": 0,
                "publication_backpressure_capacity": 1,
                "pool_slots": None,
                "release_capacity": None,
                "max_fanout_release_streams": None,
                "bulk_work_profile": "not_applicable",
                "publisher_period_us": 1_000,
            },
            "counts": {
                "source_callbacks": 111,
                "source_compute_entries": 111,
                "source_tensor_pool_retries": 0,
                "source_tensor_pool_in_activation_retries": 0,
                "measured_source_callbacks": 100,
                "relay_callbacks": 0,
                "sink_callbacks": 110,
                "publications": 111,
                "relay_publications": 0,
                "deliveries": 110,
                "consumer_visible_replacements": 0,
                "replacement_rate": 0.0,
                "activation_completions": 110,
                "backpressured_emits": 0,
                "source_backpressured_emits": 0,
                "notification_post_attempts": 0,
                "validation_errors": 0,
                "per_sink": [
                    {
                        "index": 0,
                        "deliveries": 110,
                        "final_sequence": 110,
                        "replacements": 0,
                        "replacement_rate": 0.0,
                    }
                ],
            },
            "validation": {
                "mode": "final",
                "timing": "after_timed_operations",
                "passed": True,
            },
            "allocation_probe": {
                "excluded_from_measurement": True,
                "source_compute_allocations": 0,
                "source_compute_allocation_limit": 0,
                "relay_compute_allocations": 0,
                "relay_compute_allocation_limit": 0,
                "sink_compute_allocations": 0,
                "sink_compute_allocation_limit": 0,
            },
            "wake": {
                "event_wake": False,
                "busy_poll": False,
                "receive_polling": False,
            },
            "placement": {
                "operator_lanes": "separate",
                "requested_operator_lanes": "separate",
                "operator_lane_count": 2,
                "source_compile_accepted_cpu": None,
                "relay_compile_accepted_cpu": None,
                "sink_compile_accepted_cpu": None,
                "sink_compile_accepted_cpus": [None],
                "operator_partition_uses": [
                    {"partition": "runtime-graph-source", "lane": 0, "operator_count": 1},
                    {"partition": "runtime-graph-sink", "lane": 2, "operator_count": 1},
                ],
                "support_threads": [
                    {
                        "role": "controller",
                        "requested_cpu": None,
                        "entered": True,
                        "thread_identity_kind": (
                            "native_tid"
                            if sys.platform.startswith("linux")
                            else "std_thread_id_hash"
                        ),
                        "native_tid_syscall_unavailable": False,
                        "observed_affinity": [],
                        "error": None,
                    },
                ],
            },
            "session": {"termination": "stopped", "diagnostics": []},
            "timing": {
                "latency_ns": {"samples": 100},
                "publication_to_receive_ns": {"samples": 100},
            },
        }
        path = tmp_path / "result.json"
        path.write_text(json.dumps(result), encoding="utf-8")

        # Key assertion: incomplete evidence fails before it can enter aggregate summaries.
        with pytest.raises(RuntimeError, match="incomplete publication or delivery"):
            runner.checked_result(path, args)

        # Key setup: make aggregate counts complete while leaving the sole sink one sample short.
        result["counts"].update(
            {"sink_callbacks": 111, "deliveries": 111, "activation_completions": 111}
        )
        path.write_text(json.dumps(result), encoding="utf-8")

        # Key assertion: per-sink evidence cannot be hidden by a correct aggregate count.
        with pytest.raises(RuntimeError, match="incomplete per-sink evidence"):
            runner.checked_result(path, make_args())

    def test_checked_result_accepts_source_only_activation_counts(self, tmp_path: Path) -> None:
        # Key setup: source-only evidence has activations but intentionally no publications.
        args = make_args(
            samples=2,
            warmup=0,
            workload="notification_latency",
            scenario="source_only",
        )
        support_threads = [
            {
                "role": role,
                "requested_cpu": None,
                "entered": True,
                "thread_identity_kind": (
                    "native_tid" if sys.platform.startswith("linux") else "std_thread_id_hash"
                ),
                "native_tid_syscall_unavailable": False,
                "observed_affinity": [],
                "error": None,
            }
            for role in ("controller", "notification_driver")
        ]
        result = {
            "schema": runner.RESULT_SCHEMA,
            "runtime_controls": {
                "direct_continuation_elision_disabled": False,
                "closed_island_composite_disabled": False,
                "holomq_event_wake_forced": False,
                "process_local_channel_pump_forced": False,
                "process_local_pump_delivery_forced": False,
            },
            "clock": clock_evidence(),
            "timestamp_instrumentation": timestamp_instrumentation(),
            "benchmark": "runtime_graph_source_activation_notification_latency",
            "scenario": "source_only",
            "process_local_inline_delivery": "not_applicable",
            "queue_policy": "blocking",
            "workload": "notification_latency",
            "source_trigger": "notification",
            "partitioning": partitioning_evidence(args),
            "geometry": {
                "concurrent_admission": False,
                "samples": 2,
                "warmup": 0,
                "callback_work_us": 0,
                "experimental_fanout": "disabled",
                "source_work_us": None,
                "sink_work_us": None,
                "emit_then_work": False,
                "burst_size": 1,
                "lane_fifo_priority": 0,
                "allow_pump_lane_cpu_sharing": False,
                "queue_depth": 1,
                "subscriber_queue_depths": [],
                "subscriber_queue_policies": [],
                "offered_burst_size": 1,
                "fanout": 0,
                "source_outputs": 0,
                "subscribers_per_output": 0,
                "sink_operators": 0,
                "inputs_per_sink": 0,
                "relay_count": 0,
                "pipeline_edges": 0,
                "notification_capacity": 1,
                "publication_backpressure_capacity": 1,
                "pool_slots": None,
                "release_capacity": None,
                "max_fanout_release_streams": None,
                "bulk_work_profile": "not_applicable",
                "publisher_period_us": 1_000,
            },
            "counts": {
                "source_callbacks": 3,
                "source_compute_entries": 3,
                "source_tensor_pool_retries": 0,
                "source_tensor_pool_in_activation_retries": 0,
                "measured_source_callbacks": 2,
                "relay_callbacks": 0,
                "sink_callbacks": 0,
                "publications": 0,
                "relay_publications": 0,
                "deliveries": 0,
                "consumer_visible_replacements": 0,
                "replacement_rate": 0.0,
                "activation_completions": 3,
                "backpressured_emits": 0,
                "source_backpressured_emits": 0,
                "notification_post_attempts": 3,
                "notifications_accepted": 3,
                "validation_errors": 0,
                "per_sink": [],
            },
            "validation": {
                "mode": "final",
                "timing": "after_timed_operations",
                "passed": True,
            },
            "allocation_probe": {
                "excluded_from_measurement": True,
                "source_compute_allocations": 0,
                "source_compute_allocation_limit": 0,
                "relay_compute_allocations": 0,
                "relay_compute_allocation_limit": 0,
                "sink_compute_allocations": 0,
                "sink_compute_allocation_limit": 0,
            },
            "wake": {
                "event_wake": False,
                "busy_poll": False,
                "receive_polling": False,
            },
            "placement": {
                "operator_lanes": "separate",
                "requested_operator_lanes": "separate",
                "operator_lane_count": 1,
                "source_compile_accepted_cpu": None,
                "relay_compile_accepted_cpu": None,
                "sink_compile_accepted_cpu": None,
                "sink_compile_accepted_cpus": [],
                "operator_partition_uses": [
                    {"partition": "runtime-graph-source", "lane": 1, "operator_count": 1}
                ],
                "support_threads": support_threads,
            },
            "session": {"termination": "stopped", "diagnostics": []},
            "timing": {
                "configured_publication_rate_per_second": 1_000.0,
                "logical_publications_per_second": None,
                "aggregate_deliveries_per_second": None,
                "logical_payload_bytes_per_second": None,
                "aggregate_payload_bytes_per_second": None,
                "latency_ns": {"samples": 2},
                "notification_to_source_ns": {"samples": 2},
                "source_callback_entry_to_pre_emit_ns": None,
                "source_emit_ns": None,
                "source_publication_call_ns": None,
                "source_post_emit_to_pre_return_ns": None,
                "source_post_emit_to_sink_ns": None,
                "source_post_emit_to_sink_overlaps": 0,
                "source_to_sink_ns": None,
                "source_to_sink_by_sink_ns": [],
                "source_post_emit_to_sink_by_sink_ns": [],
                "source_pre_return_to_sink_callback_by_sink_ns": [],
                "sink_callback_to_post_receive_by_sink_ns": [],
            },
        }
        path = tmp_path / "source_only.json"
        path.write_text(json.dumps(result), encoding="utf-8")

        # Key assertion: validation applies scenario-specific count invariants.
        assert runner.checked_result(path, args) == result

        # Key assertion: native partition-cover evidence is mandatory for every v1 result.
        result["partitioning"].pop("authoring")
        path.write_text(json.dumps(result), encoding="utf-8")
        with pytest.raises(RuntimeError, match="partition-cover evidence"):
            runner.checked_result(path, args)
        result["partitioning"]["authoring"] = "legacy"

        # Key setup: the versioned v1 compatibility alias must remain present and agree with the
        # precise requested-policy field.
        result["placement"].pop("operator_lanes")
        path.write_text(json.dumps(result), encoding="utf-8")
        with pytest.raises(RuntimeError, match="unexpected Operator lane placement"):
            runner.checked_result(path, args)
        result["placement"]["operator_lanes"] = "shared"
        path.write_text(json.dumps(result), encoding="utf-8")
        with pytest.raises(RuntimeError, match="unexpected Operator lane placement"):
            runner.checked_result(path, args)
        result["placement"]["operator_lanes"] = "separate"

        # Key assertion: the private A/B provenance is mandatory rather than inferred from the
        # requested command, so a stale binary cannot enter a matched summary silently.
        result.pop("runtime_controls")
        path.write_text(json.dumps(result), encoding="utf-8")
        with pytest.raises(RuntimeError, match="private Runtime controls"):
            runner.checked_result(path, args)
        result["runtime_controls"] = {
            "direct_continuation_elision_disabled": False,
            "closed_island_composite_disabled": False,
            "holomq_event_wake_forced": False,
            "process_local_channel_pump_forced": False,
            "process_local_pump_delivery_forced": False,
        }

        # Key setup: native evidence retains a dormant record for each bounded support role.
        result["placement"]["support_threads"].append(
            {
                "role": "channel_pump",
                "requested_cpu": None,
                "entered": False,
                "thread_identity_kind": "none",
                "native_tid_syscall_unavailable": False,
                "native_tid": None,
                "observed_affinity": [],
                "error": None,
            }
        )
        path.write_text(json.dumps(result), encoding="utf-8")

        # Key assertion: a known dormant slot is inventory, not unexpected Runtime work.
        assert runner.checked_result(path, args) == result

        # Key setup: mark the unrequested pump slot as having entered Runtime work.
        result["placement"]["support_threads"][-1]["entered"] = True
        path.write_text(json.dumps(result), encoding="utf-8")

        # Key assertion: unexpected active Runtime work cannot enter the aggregate summary.
        with pytest.raises(RuntimeError, match="unexpected support-thread inventory"):
            runner.checked_result(path, args)

        # Key setup: unknown role names fail even when their evidence claims to be dormant.
        result["placement"]["support_threads"][-1].update({"role": "source_only", "entered": False})
        path.write_text(json.dumps(result), encoding="utf-8")

        # Key assertion: the result schema cannot grow unrecognized support roles silently.
        with pytest.raises(RuntimeError, match="unexpected support-thread inventory"):
            runner.checked_result(path, args)

        result["placement"]["support_threads"].pop()

        # Key assertion: even an excluded warm-up probe fails the alloc-free compute contract.
        result["allocation_probe"]["source_compute_allocations"] = 1
        path.write_text(json.dumps(result), encoding="utf-8")
        with pytest.raises(RuntimeError, match="compute-path allocations"):
            runner.checked_result(path, args)

    def test_checked_result_accepts_source_only_saturated_counts(self, tmp_path: Path) -> None:
        # Key setup: source-only saturation retires credits by callback completion, not delivery.
        args = make_args(
            samples=2,
            warmup=0,
            workload="saturated",
            scenario="source_only",
            queue_depth=2,
            notification_capacity=2,
            publisher_period_us=0,
        )
        result = {
            "schema": runner.RESULT_SCHEMA,
            "runtime_controls": {
                "direct_continuation_elision_disabled": False,
                "closed_island_composite_disabled": False,
                "holomq_event_wake_forced": False,
                "process_local_channel_pump_forced": False,
                "process_local_pump_delivery_forced": False,
            },
            "clock": clock_evidence(),
            "timestamp_instrumentation": timestamp_instrumentation(),
            "benchmark": "runtime_graph_source_activation_saturated",
            "scenario": "source_only",
            "process_local_inline_delivery": "not_applicable",
            "queue_policy": "blocking",
            "workload": "saturated",
            "source_trigger": "notification",
            "partitioning": partitioning_evidence(args),
            "geometry": {
                "concurrent_admission": False,
                "samples": 2,
                "warmup": 0,
                "callback_work_us": 0,
                "experimental_fanout": "disabled",
                "source_work_us": None,
                "sink_work_us": None,
                "emit_then_work": False,
                "burst_size": 1,
                "lane_fifo_priority": 0,
                "allow_pump_lane_cpu_sharing": False,
                "queue_depth": 2,
                "subscriber_queue_depths": [],
                "subscriber_queue_policies": [],
                "offered_burst_size": 1,
                "fanout": 0,
                "source_outputs": 0,
                "subscribers_per_output": 0,
                "sink_operators": 0,
                "inputs_per_sink": 0,
                "relay_count": 0,
                "pipeline_edges": 0,
                "notification_capacity": 2,
                "publication_backpressure_capacity": 2,
                "pool_slots": None,
                "release_capacity": None,
                "max_fanout_release_streams": None,
                "bulk_work_profile": "not_applicable",
                "publisher_period_us": None,
            },
            "counts": {
                "source_callbacks": 3,
                "source_compute_entries": 3,
                "source_tensor_pool_retries": 0,
                "source_tensor_pool_in_activation_retries": 0,
                "measured_source_callbacks": 2,
                "relay_callbacks": 0,
                "sink_callbacks": 0,
                "publications": 0,
                "relay_publications": 0,
                "deliveries": 0,
                "consumer_visible_replacements": 0,
                "replacement_rate": 0.0,
                "activation_completions": 3,
                "backpressured_emits": 0,
                "source_backpressured_emits": 0,
                "notification_post_attempts": 4,
                "notifications_accepted": 4,
                "validation_errors": 0,
                "per_sink": [],
            },
            "validation": {
                "mode": "final",
                "timing": "after_timed_operations",
                "passed": True,
            },
            "allocation_probe": {
                "excluded_from_measurement": True,
                "source_compute_allocations": 0,
                "source_compute_allocation_limit": 0,
                "relay_compute_allocations": 0,
                "relay_compute_allocation_limit": 0,
                "sink_compute_allocations": 0,
                "sink_compute_allocation_limit": 0,
            },
            "wake": {
                "event_wake": False,
                "busy_poll": False,
                "receive_polling": False,
            },
            "placement": {
                "operator_lanes": "separate",
                "requested_operator_lanes": "separate",
                "operator_lane_count": 1,
                "source_compile_accepted_cpu": None,
                "relay_compile_accepted_cpu": None,
                "sink_compile_accepted_cpu": None,
                "sink_compile_accepted_cpus": [],
                "operator_partition_uses": [
                    {"partition": "runtime-graph-source", "lane": 1, "operator_count": 1}
                ],
                "support_threads": [
                    {
                        "role": role,
                        "requested_cpu": None,
                        "entered": True,
                        "thread_identity_kind": (
                            "native_tid"
                            if sys.platform.startswith("linux")
                            else "std_thread_id_hash"
                        ),
                        "native_tid_syscall_unavailable": False,
                        "observed_affinity": [],
                        "error": None,
                    }
                    for role in ("controller", "notification_driver")
                ],
            },
            "session": {"termination": "stopped", "diagnostics": []},
            "timing": {
                "configured_publication_rate_per_second": None,
                "logical_publications_per_second": 0.0,
                "aggregate_deliveries_per_second": 0.0,
                "logical_payload_bytes_per_second": 0.0,
                "aggregate_payload_bytes_per_second": 0.0,
                "latency_ns": None,
                "source_emit_ns": None,
                "source_publication_call_ns": None,
                "source_post_emit_to_pre_return_ns": None,
                "source_post_emit_to_sink_ns": None,
                "source_post_emit_to_sink_overlaps": 0,
                "source_to_sink_by_sink_ns": [],
                "source_post_emit_to_sink_by_sink_ns": [],
                "source_pre_return_to_sink_callback_by_sink_ns": [],
                "sink_callback_to_post_receive_by_sink_ns": [],
            },
        }
        path = tmp_path / "source_only_saturated.json"
        path.write_text(json.dumps(result), encoding="utf-8")

        # Key assertion: no Channel publication or delivery is invented for the ceiling row.
        assert runner.checked_result(path, args) == result


class TestRunRepetitionFailures:
    def test_native_skip_exit_is_preserved_with_logs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Key setup: the CUDA-sensitive native benchmark reports the CTest skip return code.
        monkeypatch.setattr(
            runner.subprocess,
            "run",
            lambda *_args, **_kwargs: SimpleNamespace(
                returncode=runner.NATIVE_SKIP_RETURN_CODE,
                stdout="native skip",
                stderr="CUDA unavailable",
            ),
        )

        # Key assertion: the Python runner preserves code 77 and retains both diagnostic streams.
        with pytest.raises(SystemExit) as error:
            runner.run_repetition(make_args(), Path("/tmp/benchmark"), 0, tmp_path)
        assert error.value.code == runner.NATIVE_SKIP_RETURN_CODE
        repetition_dir = tmp_path / "repetition_000"
        assert next(repetition_dir.glob("stdout_*.log")).read_text() == "native skip"
        assert next(repetition_dir.glob("stderr_*.log")).read_text() == "CUDA unavailable"

    def test_nonzero_exit_retains_logs_and_fails_closed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Key setup: the native process exits after writing diagnostic output.
        monkeypatch.setattr(
            runner.subprocess,
            "run",
            lambda *_args, **_kwargs: SimpleNamespace(
                returncode=9, stdout="native stdout", stderr="native stderr"
            ),
        )

        # Key assertion: no result is accepted and both streams remain available for diagnosis.
        with pytest.raises(RuntimeError, match="exited with 9") as error:
            runner.run_repetition(make_args(), Path("/tmp/benchmark"), 0, tmp_path)
        assert "--- native stdout\nnative stdout" in str(error.value)
        assert "--- native stderr\nnative stderr" in str(error.value)
        repetition_dir = tmp_path / "repetition_000"
        assert next(repetition_dir.glob("stdout_*.log")).read_text() == "native stdout"
        assert next(repetition_dir.glob("stderr_*.log")).read_text() == "native stderr"

    def test_native_failure_details_reports_bounded_stream_tails(self) -> None:
        # Key setup: stderr exceeds the diagnostic bound while stdout remains short.
        stderr = "discarded-prefix\n" + "x" * runner.NATIVE_FAILURE_LOG_LIMIT

        details = runner.native_failure_details("native stdout", stderr)

        # Key assertion: CTest receives both streams and an explicit truncation marker.
        assert "--- native stdout\nnative stdout" in details
        assert "--- native stderr\n[... earlier output truncated ...]" in details
        assert "discarded-prefix" not in details

    def test_native_result_failure_details_fails_closed_when_file_is_missing(
        self, tmp_path: Path
    ) -> None:
        assert runner.native_result_failure_details(tmp_path / "missing.json") == ""

    @pytest.mark.parametrize(
        "payload",
        ["{", "[]"],
        ids=("malformed-json", "non-object-json"),
    )
    def test_native_result_failure_details_fails_closed_for_invalid_inputs(
        self, tmp_path: Path, payload: str
    ) -> None:
        result_path = tmp_path / "result.json"
        result_path.write_text(payload, encoding="utf-8")

        assert runner.native_result_failure_details(result_path) == ""

    def test_native_result_failure_details_reports_runtime_startup_evidence(
        self, tmp_path: Path
    ) -> None:
        result_path = tmp_path / "result.json"
        result_path.write_text(
            json.dumps(
                {
                    "cold_timing_ns": {
                        "compile": 100,
                        "run_async_return": 200,
                        "run_async_to_first_callback": 0,
                        "terminal_wait": 16_000_000_000,
                    },
                    "counts": {"source_compute_entries": 0},
                    "placement": {"support_threads": [{"role": "controller", "entered": True}]},
                    "runtime_phase_counters": {
                        "enabled": True,
                        "readiness_scans": 0,
                        "lane_lifecycle_dispatches": 2,
                    },
                    "runtime_phase_trace": {
                        "enabled": True,
                        "valid": False,
                        "terminal_run_closing_dispatches": 1,
                    },
                    "runtime_thread_usage": {"enabled": True, "supported": True},
                    "session": {"termination": "stopped", "diagnostics": []},
                }
            ),
            encoding="utf-8",
        )

        details = runner.native_result_failure_details(result_path)

        assert details.startswith("--- native result diagnostics\n")
        evidence = json.loads(details.split("\n", 1)[1])
        assert evidence["cold_timing_ns"]["run_async_to_first_callback"] == 0
        assert evidence["runtime_phase_counters"] == {
            "enabled": True,
            "lane_lifecycle_dispatches": 2,
        }
        assert evidence["support_threads"] == [{"entered": True, "role": "controller"}]
        assert evidence["runtime_phase_trace"] == {
            "enabled": True,
            "valid": False,
            "terminal_run_closing_dispatches": 1,
        }

    def test_timeout_retains_partial_logs_and_fails_closed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Key setup: emulate a timed-out native process with partial byte-stream output.
        def time_out(*_args: object, **_kwargs: object) -> None:
            raise subprocess.TimeoutExpired(
                cmd=["benchmark"], timeout=15.0, output=b"partial stdout", stderr=b"partial stderr"
            )

        monkeypatch.setattr(runner.subprocess, "run", time_out)

        # Key assertion: timeout remains distinct from a native exit and preserves partial logs.
        with pytest.raises(TimeoutError, match="timed out") as error:
            runner.run_repetition(make_args(), Path("/tmp/benchmark"), 0, tmp_path)
        assert "--- native stdout\npartial stdout" in str(error.value)
        assert "--- native stderr\npartial stderr" in str(error.value)
        repetition_dir = tmp_path / "repetition_000"
        assert next(repetition_dir.glob("stdout_*.log")).read_text() == "partial stdout"
        assert next(repetition_dir.glob("stderr_*.log")).read_text() == "partial stderr"

    @pytest.mark.parametrize(
        ("payload", "message"),
        [
            ("{", "invalid benchmark result"),
            (json.dumps({"schema": "unknown"}), "unsupported result schema"),
        ],
    )
    def test_invalid_native_json_never_reaches_aggregation(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        payload: str,
        message: str,
    ) -> None:
        # Key setup: the native process exits successfully but leaves unusable result evidence.
        def write_result(command: list[str], **_kwargs: object) -> SimpleNamespace:
            result_path = Path(command[command.index("--output") + 1])
            result_path.write_text(payload, encoding="utf-8")
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(runner.subprocess, "run", write_result)

        # Key assertion: malformed or incompatible evidence fails before summary generation.
        with pytest.raises(RuntimeError, match=message):
            runner.run_repetition(make_args(), Path("/tmp/benchmark"), 0, tmp_path)


class TestArtifactReporting:
    def test_main_reports_default_artifact_directory(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # Key setup: use the default temporary artifact directory with controlled runner seams.
        binary = tmp_path / "runtime_graph_benchmark"
        binary.touch()
        output_dir = tmp_path / "generated-artifacts"
        args = make_args(disable_direct_continuation_elision=True)
        args.benchmark_binary = binary
        args.output_dir = None
        args.git_revision = "abc123"
        args.source_state = "dirty-task040-worktree-snapshot"
        args.source_snapshot_sha256 = "b" * 64
        monkeypatch.setattr(runner, "parse_args", lambda: args)
        monkeypatch.setattr(runner.tempfile, "mkdtemp", lambda **_kwargs: str(output_dir))
        monkeypatch.setattr(runner, "collect_cpu_topology", lambda _args: [])
        monkeypatch.setattr(
            runner,
            "run_repetition",
            lambda *_args: (
                {"benchmark": "row"},
                [str(binary), "--disable-direct-continuation-elision"],
            ),
        )
        monkeypatch.setattr(
            runner,
            "summarize_results",
            lambda *_args: {"schema": runner.SUMMARY_SCHEMA, "benchmark": "row"},
        )
        monkeypatch.setattr(runner, "cmake_metadata", lambda _binary: {})
        monkeypatch.setattr(runner, "gpu_metadata", lambda: [])

        assert runner.main() == 0

        # Key assertion: an implicit temporary location is visible and contains both durable files.
        assert f"benchmark artifacts: {output_dir}" in capsys.readouterr().err
        assert (output_dir / "summary.json").is_file()
        assert (output_dir / "manifest.json").is_file()
        manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["binary_sha256"] == (
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        )
        assert manifest["source_state"] == "dirty-task040-worktree-snapshot"
        assert manifest["source_snapshot_sha256"] == "b" * 64
        assert manifest["runtime_controls"] == {
            "direct_continuation_elision_disabled": True,
            "closed_island_composite_disabled": False,
            "holomq_event_wake_forced": False,
            "process_local_channel_pump_forced": False,
            "process_local_pump_delivery_forced": False,
        }
        assert "--disable-direct-continuation-elision" in manifest["native_commands"][0]


class TestGitRevision:
    def test_explicit_revision_survives_container_without_git_tree(self) -> None:
        # Key assertion: an injected revision avoids dependence on container Git metadata.
        assert runner.git_revision("abc123") == ("abc123", "command_line")


class TestFanoutExperimentControls:
    @pytest.mark.parametrize("fanout", [2, 4, 8])
    def test_separate_outputs_use_direct_delivery_with_matching_partitions(self, fanout):
        args = make_args(multi_output=True, fanout=fanout, operator_lanes="per_sink")
        runner.validate_args(args)
        assert runner.source_output_count(args) == fanout
        assert runner.source_emit_count(args) == fanout
        assert runner.sink_operator_count(args) == fanout
        assert not runner.uses_holomq(args)
        assert not runner.uses_channel_pump(args)
        assert runner.benchmark_name("pod", "latency", multi_output=True, fanout=fanout).endswith(
            f"{fanout}x1"
        )

    def test_pump_sharing_opt_in_does_not_allow_other_role_collisions(self):
        args = make_args(
            fanout=2,
            operator_lanes="per_sink",
            source_cpu=2,
            sink_cpus=[3, 4],
            controller_cpu=5,
            channel_pump_cpu=3,
            force_process_local_pump_delivery=True,
        )
        with pytest.raises(ValueError, match="share one logical CPU"):
            runner.validate_args(args)
        args.allow_pump_lane_cpu_sharing = True
        runner.validate_args(args)
        args.controller_cpu = 3
        with pytest.raises(ValueError, match="only with operator lanes"):
            runner.validate_args(args)
        args.controller_cpu = 5
        args.sink_cpus = [2, 4]
        with pytest.raises(ValueError, match="share one logical CPU"):
            runner.validate_args(args)

    def test_physical_core_sharing_is_limited_to_pump_lane_pair(self, monkeypatch):
        args = make_args(
            fanout=2,
            operator_lanes="per_sink",
            source_cpu=2,
            sink_cpus=[3, 4],
            controller_cpu=5,
            channel_pump_cpu=13,
            allow_pump_lane_cpu_sharing=True,
            force_process_local_pump_delivery=True,
        )

        def topology(role, cpu):
            return {
                "role": role,
                "cpu": cpu,
                "pinned": True,
                "topology_available": True,
                "package_id": 0,
                "core_id": cpu % 10,
            }

        monkeypatch.setattr(runner, "read_cpu_topology", topology)
        runner.collect_cpu_topology(args)
        args.channel_pump_cpu = 15
        with pytest.raises(RuntimeError, match="share one physical core"):
            runner.collect_cpu_topology(args)

    def test_work_burst_and_fifo_are_forwarded(self):
        args = make_args(
            fanout=2,
            operator_lanes="per_sink",
            source_work_us=500,
            sink_work_us=0,
            emit_then_work=True,
            burst_size=4,
            queue_depth=4,
            lane_fifo_priority=1,
        )
        runner.validate_args(args)
        command = runner.benchmark_command(args, Path("/tmp/benchmark"), Path("/tmp/result.json"))
        for name, value in (
            ("source-work-us", "500"),
            ("sink-work-us", "0"),
            ("burst-size", "4"),
            ("lane-fifo-priority", "1"),
        ):
            assert command[command.index("--" + name) + 1] == value
        assert "--emit-then-work" in command

    @pytest.mark.parametrize(
        "updates",
        [
            {"source_work_us": -1},
            {"sink_work_us": 1_000_001},
            {"burst_size": 2, "queue_depth": 1},
            {"burst_size": 65, "queue_depth": 65},
            {"lane_fifo_priority": 100},
            {"lane_fifo_priority": 1, "operator_lanes": "shared"},
            {"emit_then_work": True, "scenario": "tensor"},
        ],
    )
    def test_unsupported_controls_fail_before_launch(self, updates):
        with pytest.raises(ValueError, match="work|burst|FIFO|priority"):
            runner.validate_args(make_args(**updates))


class TestPrivateFanoutControls:
    @pytest.mark.parametrize("mode", ["cut", "emit"])
    def test_experiment_has_explicit_identity_and_no_pump(self, mode):
        args = make_args(fanout=2, operator_lanes="per_sink", experimental_fanout=mode)
        runner.validate_args(args)
        assert runner.uses_holomq(args)
        assert not runner.event_wake_required(args)
        assert not runner.uses_channel_pump(args)
        command = runner.benchmark_command(args, Path("/tmp/benchmark"), Path("/tmp/result.json"))
        assert command[command.index("--experimental-fanout") + 1] == mode
        assert runner.benchmark_name("pod", "latency", fanout=2, experimental_fanout=mode).endswith(
            "_experimental_" + mode
        )

    def test_experiment_admits_raw_tensor_per_sink_fanout(self):
        args = make_args(
            scenario="tensor", fanout=3, operator_lanes="per_sink", experimental_fanout="emit"
        )
        runner.validate_args(args)
        assert not runner.event_wake_required(args)
        assert not runner.uses_channel_pump(args)

    @pytest.mark.parametrize("scenario", ["schema", "third_party_schema"])
    def test_experiment_admits_generated_schema_per_sink_fanout(self, scenario):
        args = make_args(
            scenario=scenario, fanout=2, operator_lanes="per_sink", experimental_fanout="emit"
        )
        runner.validate_args(args)
        assert not runner.event_wake_required(args)

    def test_raw_tensor_fanout_accepts_separate_sink_lanes(self):
        runner.validate_args(make_args(scenario="tensor", fanout=2, operator_lanes="per_sink"))

    @pytest.mark.parametrize(
        "changes",
        [
            {"multi_output": True},
            {"queue_policy": "drop_oldest"},
            {"operator_lanes": "shared"},
            {"scenario": "sample_forward"},
            {"channel_pump_cpu": 7},
            {"force_holomq_event_wake": True},
        ],
    )
    def test_experiment_rejects_unsupported_controls(self, changes):
        args = make_args(fanout=2, operator_lanes="per_sink", experimental_fanout="cut")
        for name, value in changes.items():
            setattr(args, name, value)
        with pytest.raises(ValueError, match="private fanout controls"):
            runner.validate_args(args)


class TestFanoutAttributionControls:
    def test_sink_fifo_priorities_are_explicit(self):
        args = make_args(
            fanout=2,
            operator_lanes="per_sink",
            experimental_fanout="emit",
            lane_fifo_priority=3,
            sink_fifo_priorities=[1, 2],
        )
        runner.validate_args(args)
        command = runner.benchmark_command(args, Path("/tmp/benchmark"), Path("/tmp/result.json"))
        assert command[command.index("--sink-fifo-priorities") + 1] == "1,2"

    def test_concurrent_admission_is_not_a_requested_option(self):
        # Runtime admits concurrently whenever delivery inside emit() serves a service, so no
        # option selects it; results report the effective mode instead.
        with pytest.raises(SystemExit):
            runner.parse_args(
                [
                    "/tmp/benchmark",
                    "--experimental-fanout",
                    "emit",
                    "--experimental-concurrent-admission",
                ]
            )

    @pytest.mark.parametrize(
        "updates",
        [
            {"sink_fifo_priorities": [1]},
            {"sink_fifo_priorities": [1, 100]},
        ],
    )
    def test_refinements_fail_closed(self, updates):
        with pytest.raises(ValueError, match="fanout refinements|sink FIFO"):
            runner.validate_args(make_args(fanout=2, operator_lanes="per_sink", **updates))

    def test_attribution_is_explicit_and_separate_from_phase_trace(self):
        args = make_args(
            fanout=2, fanout_attribution=True, measure_source_period=True, force_holomq_fanout=True
        )
        runner.validate_args(args)
        command = runner.benchmark_command(args, Path("/tmp/benchmark"), Path("/tmp/result.json"))
        assert "--fanout-attribution" in command
        assert "--measure-source-period" in command
        assert "--runtime-phase-counters" not in command
        assert runner.benchmark_name(
            "pod", "latency", fanout=2, fanout_attribution=True, measure_source_period=True
        ).endswith("_fanout_attribution_source_period")

    @pytest.mark.parametrize(
        "changes", [{"multi_output": True}, {"scenario": "tensor"}, {"queue_policy": "drop_oldest"}]
    )
    def test_unsupported_attribution_fails_before_launch(self, changes):
        args = make_args(fanout=2, fanout_attribution=True, **changes)
        with pytest.raises(ValueError, match="fanout attribution"):
            runner.validate_args(args)

    def test_attribution_rejects_a_direct_family(self):
        # A blocking POD family is delivered by direct retention unless HoloMQ is forced.
        args = make_args(fanout=2, fanout_attribution=True)
        assert runner.direct_fanout_family(args)
        with pytest.raises(ValueError, match="fanout attribution observes HoloMQ delivery"):
            runner.validate_args(args)

    def test_period_requires_clocked_blocking_payload(self):
        with pytest.raises(ValueError, match="source period"):
            runner.validate_args(make_args(measure_source_period=True, workload="saturated"))


class TestFanoutTimingCompleteness:
    @staticmethod
    def evidence():
        sink = {
            "index": 0,
            **{
                metric: {"samples": 4}
                for metric in (
                    "emit_to_admission_ns",
                    "admission_to_hint_ns",
                    "hint_call_ns",
                    "hint_to_callback_ns",
                )
            },
            "hint_callback_overlaps": 0,
        }
        return {
            "fanout_attribution": {"enabled": True, "valid": True, "sinks": [sink]},
            "source_period_ns": {"samples": 3},
        }

    def test_complete_burst_period_evidence(self):
        args = make_args(samples=4, fanout=1, fanout_attribution=True, measure_source_period=True)
        timing = self.evidence()
        # Four publications in two callbacks produce one interval, not three.
        timing["source_period_ns"] = {"samples": 1}
        runner.validate_fanout_timing(timing, {"measured_source_callbacks": 2}, args, Path("row"))

    @pytest.mark.parametrize(
        "metric",
        [
            "emit_to_admission_ns",
            "admission_to_hint_ns",
            "hint_call_ns",
            "hint_to_callback_ns",
        ],
    )
    def test_missing_recipient_observation_fails(self, metric):
        args = make_args(samples=4, fanout=1, fanout_attribution=True)
        timing = self.evidence()
        timing["fanout_attribution"]["sinks"][0][metric] = None
        with pytest.raises(RuntimeError, match="incomplete|omitted"):
            runner.validate_fanout_timing(timing, {}, args, Path("row"))

    @pytest.mark.parametrize(
        "corrupt",
        [
            lambda timing, _counts: timing.update(fanout_attribution=[]),
            lambda timing, _counts: timing["fanout_attribution"].update(sinks={}),
            lambda timing, _counts: timing["fanout_attribution"]["sinks"].append(4),
            lambda timing, _counts: timing["fanout_attribution"]["sinks"][0].update(
                hint_call_ns=[4]
            ),
            lambda timing, _counts: timing["fanout_attribution"]["sinks"][0].update(
                hint_callback_overlaps="0"
            ),
            lambda _timing, counts: counts.update(measured_source_callbacks="4"),
            lambda _timing, counts: counts.clear(),
        ],
    )
    def test_malformed_evidence_fails_the_contract(self, corrupt):
        args = make_args(samples=4, fanout=1, fanout_attribution=True, measure_source_period=True)
        timing, counts = self.evidence(), {"measured_source_callbacks": 4}
        runner.validate_fanout_timing(timing, counts, args, Path("row"))
        corrupt(timing, counts)
        with pytest.raises(RuntimeError):
            runner.validate_fanout_timing(timing, counts, args, Path("row"))

    @pytest.mark.parametrize("value", [[], 0, False, ""])
    def test_only_null_reports_an_empty_distribution(self, value):
        # Every callback overlapped its hint and one callback gave no period, so null passes.
        args = make_args(samples=4, fanout=1, fanout_attribution=True, measure_source_period=True)
        timing, counts = self.evidence(), {"measured_source_callbacks": 1}
        sink = timing["fanout_attribution"]["sinks"][0]
        sink.update(hint_to_callback_ns=None, hint_callback_overlaps=4)
        timing["source_period_ns"] = None
        runner.validate_fanout_timing(timing, counts, args, Path("row"))
        sink["hint_to_callback_ns"] = value
        with pytest.raises(RuntimeError, match="omitted"):
            runner.validate_fanout_timing(timing, counts, args, Path("row"))
        sink["hint_to_callback_ns"] = None
        timing["source_period_ns"] = value
        with pytest.raises(RuntimeError, match="incomplete source periods"):
            runner.validate_fanout_timing(timing, counts, args, Path("row"))

    def test_period_count_is_not_publication_count(self):
        args = make_args(samples=4, measure_source_period=True)
        with pytest.raises(RuntimeError, match="incomplete source periods"):
            runner.validate_fanout_timing(
                self.evidence(), {"measured_source_callbacks": 2}, args, Path("row")
            )
