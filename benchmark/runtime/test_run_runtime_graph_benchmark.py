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


def make_args(**updates: object) -> SimpleNamespace:
    values = {
        "samples": 100,
        "workload": "latency",
        "scenario": "pod",
        "operator_lanes": "separate",
        "partition_algorithm": "core_default",
        "warmup": 10,
        "queue_policy": "blocking",
        "queue_depth": 1,
        "relay_count": 1,
        "fanout": 1,
        "notification_capacity": 1,
        "publisher_period_us": 1_000,
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
    }
    values.update(updates)
    return SimpleNamespace(**values)


def test_expected_operator_lane_count_keeps_relay_topology_distinct_from_fanout() -> None:
    # sample_forward authors source, relay, and one sink; argument validation rejects fanout there.
    assert runner.expected_operator_lane_count(make_args(scenario="sample_forward", fanout=4)) == 3
    assert runner.expected_operator_lane_count(make_args(scenario="pod", fanout=4)) == 5


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

    # Each vector isolates one CLI contract so an invalid workload cannot produce mislabeled data.
    @pytest.mark.parametrize(
        ("updates", "message"),
        [
            ({"samples": 0}, "--samples must be positive"),
            ({"warmup": -1}, "--warmup must be nonnegative"),
            ({"queue_depth": 3}, "latency workloads require --queue-depth 1 or 2"),
            (
                {"queue_policy": "drop_oldest", "fanout": 2},
                "--queue-policy drop_oldest currently requires --fanout 1",
            ),
            ({"relay_count": 0}, "--relay-count must be between 1 and 64"),
            ({"relay_count": 65}, "--relay-count must be between 1 and 64"),
            ({"relay_count": 2}, "--relay-count requires --scenario sample_forward"),
            ({"fanout": 0}, "--fanout must be between 1 and 8"),
            ({"fanout": 9}, "--fanout must be between 1 and 8"),
            (
                {"scenario": "tensor", "fanout": 2},
                "--fanout greater than one requires --scenario pod or schema",
            ),
            (
                {"fanout": 2, "force_holomq_process_local_inline": True},
                "--force-holomq-process-local-inline requires --fanout 1",
            ),
            (
                {
                    "disable_direct_continuation_elision": True,
                    "force_holomq_process_local_inline": True,
                },
                "are mutually exclusive",
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
                },
                "must not share one logical CPU",
            ),
            ({"relay_cpu": 3}, "requires --scenario sample_forward"),
            (
                {"partition_algorithm": "source_chain"},
                "must remain core_default for benchmark-authored partitions",
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
                "--operator-lanes per_sink requires pod or schema fanout greater than one",
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
                "--channel-pump-cpu requires a transport-backed scenario",
            ),
            (
                {
                    "scenario": "source_only",
                    "force_holomq_process_local_inline": True,
                },
                (
                    "--force-holomq-process-local-inline requires pod, signal, "
                    "sample_forward, schema, tensor, image, or multi_tensor"
                ),
            ),
            (
                {"scenario": "image", "tensor_storage": "device"},
                "--scenario image and multi_tensor currently require --tensor-storage "
                "pageable_host",
            ),
            (
                {"scenario": "image", "tensor_device": 1},
                "--scenario image and multi_tensor currently require --tensor-device 0",
            ),
            (
                {"scenario": "multi_tensor", "tensor_storage": "pinned_host"},
                "--scenario image and multi_tensor currently require --tensor-storage "
                "pageable_host",
            ),
            (
                {"scenario": "multi_tensor", "tensor_device": 1},
                "--scenario image and multi_tensor currently require --tensor-device 0",
            ),
            (
                {"bulk_work_profile": "descriptor_only"},
                "--bulk-work-profile descriptor_only requires --scenario tensor or multi_tensor",
            ),
            ({"notification_driver_cpu": 3}, "requires a notification workload"),
            ({"repetitions": 0}, "--repetitions must be positive"),
            ({"max_median_cv": -0.1}, "--max-median-cv must be nonnegative"),
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

    def test_pod_fanout_uses_holomq_and_records_each_unpinned_sink_lane(self) -> None:
        # Key setup: one source publishes each logical POD sample to four process-local sinks.
        args = make_args(scenario="pod", fanout=4, channel_pump_cpu=5)

        runner.validate_args(args)
        command = runner.benchmark_command(
            args, Path("/tmp/runtime_graph_benchmark"), Path("/tmp/result.json")
        )
        topology = runner.collect_cpu_topology(args)

        # Key assertion: fanout is explicit, uses the required transport pump, and records every
        # independently compiled sink lane without inventing CPU placement.
        assert command[command.index("--fanout") + 1] == "4"
        assert runner.process_local_delivery(args) == "holomq"
        assert runner.benchmark_name("pod", "latency", fanout=4) == "runtime_graph_pod_1to4"
        assert [item["role"] for item in topology] == [
            "source",
            "sink_1",
            "sink_2",
            "sink_3",
            "sink_4",
            "controller",
            "channel_pump",
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
            channel_pump_cpu=5,
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
            "channel_pump",
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
        fallback = make_args(
            queue_policy=queue_policy,
            force_holomq_process_local_inline=True,
            channel_pump_cpu=5,
        )

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

    @pytest.mark.parametrize("scenario", ["schema", "tensor", "image"])
    def test_drop_oldest_payloads_admit_matched_holomq_control(self, scenario: str) -> None:
        # Key setup: apply the same depth-two drop-oldest policy to each direct-retained payload
        # family and change only the compiler-private delivery control.
        direct = make_args(
            scenario=scenario,
            queue_policy="drop_oldest",
            queue_depth=2,
            tensor_bytes=64,
        )
        fallback = make_args(
            scenario=scenario,
            queue_policy="drop_oldest",
            queue_depth=2,
            tensor_bytes=64,
            force_holomq_process_local_inline=True,
            channel_pump_cpu=5,
        )

        runner.validate_args(direct)
        runner.validate_args(fallback)
        direct_command = runner.benchmark_command(
            direct, Path("/tmp/runtime_graph_benchmark"), Path("/tmp/direct.json")
        )

        # Key assertion: generated schema, raw Tensor, and descriptor-backed Image retain the
        # authored lossy policy while the matched control alone requires a Channel pump.
        assert runner.process_local_delivery(direct) == "direct_retained"
        assert runner.process_local_delivery(fallback) == "holomq"
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

    def test_direct_schema_omits_channel_pump_and_admits_holomq_fallback(self) -> None:
        # Key setup: compare compiler-retained generated-schema delivery with the same graph
        # forced through its process-local HoloMQ serialized endpoint.
        direct = make_args(scenario="schema")
        fallback = make_args(
            scenario="schema",
            force_holomq_process_local_inline=True,
            channel_pump_cpu=5,
        )

        runner.validate_args(direct)
        runner.validate_args(fallback)

        # Key assertion: result identity and pump ownership distinguish the two delivery paths.
        assert runner.uses_channel_pump(direct) is False
        assert runner.process_local_delivery(direct) == "direct_retained"
        assert runner.uses_channel_pump(fallback) is True
        assert runner.process_local_delivery(fallback) == "holomq"
        assert runner.benchmark_name("schema", "latency", True) == (
            "runtime_graph_schema_time_1to1_holomq_fallback"
        )

    def test_direct_image_omits_channel_pump_and_admits_holomq_fallback(self) -> None:
        # Key setup: compare direct retention of Image metadata plus its embedded Tensor handle
        # with the same graph forced through the HoloMQ descriptor-sidecar path.
        direct = make_args(scenario="image", tensor_bytes=1 << 20)
        fallback = make_args(
            scenario="image",
            tensor_bytes=1 << 20,
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

    def test_shared_operator_lane_uses_only_source_cpu(self) -> None:
        # Key setup: source and sink share one lane while support threads remain independent.
        args = make_args(
            operator_lanes="shared",
            workload="notification_latency",
            source_cpu=2,
            controller_cpu=4,
            channel_pump_cpu=5,
            force_holomq_process_local_inline=True,
        )

        runner.validate_args(args)
        command = runner.benchmark_command(
            args, Path("/tmp/runtime_graph_benchmark"), Path("/tmp/result.json")
        )
        topology = runner.collect_cpu_topology(args)

        # Key assertion: topology reports one shared operator lane and no separate sink lane.
        assert command[command.index("--operator-lanes") + 1] == "shared"
        assert [item["role"] for item in topology] == [
            "shared_operator_lane",
            "controller",
            "channel_pump",
            "notification_driver",
        ]

    def test_source_only_notification_latency_omits_sink_placement(self) -> None:
        # Key setup: isolate notification-to-source activation without a Channel edge or sink.
        args = make_args(
            workload="notification_latency",
            scenario="source_only",
            source_cpu=2,
            controller_cpu=4,
            notification_driver_cpu=6,
        )

        runner.validate_args(args)
        topology = runner.collect_cpu_topology(
            make_args(workload="notification_latency", scenario="source_only")
        )

        # Key assertion: result naming and topology expose the reduced measurement boundary.
        assert runner.benchmark_name(args.scenario, args.workload) == (
            "runtime_graph_source_activation_notification_latency"
        )
        assert all(item["role"] != "sink" for item in topology)
        assert all(item["role"] != "channel_pump" for item in topology)

    def test_source_only_saturated_uses_only_activation_roles(self) -> None:
        # Key setup: saturate one portless Operator with bounded notification credits.
        args = make_args(
            workload="saturated",
            scenario="source_only",
            queue_depth=8,
            notification_capacity=8,
            publisher_period_us=0,
            source_cpu=2,
            controller_cpu=4,
            notification_driver_cpu=6,
        )

        runner.validate_args(args)
        topology = runner.collect_cpu_topology(args)

        # Key assertion: the row retains only the source lane, controller, and driver.
        assert runner.benchmark_name(args.scenario, args.workload) == (
            "runtime_graph_source_activation_saturated"
        )
        assert [item["role"] for item in topology] == [
            "source",
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
    def test_callback_counts_distinguish_relay_backpressure_from_source_retries(self) -> None:
        # This is the exact count pattern from the partitioned-chain backpressure CI failure:
        # all nine retries occurred in relays, so the source still ran exactly once per sample.
        counts = {
            "source_callbacks": 41,
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
                "backpressured_emits": 11,
                "source_backpressured_emits": 2,
            }
        )
        runner.validate_callback_counts(Path("result.json"), counts, args, 41, 41)

    def test_callback_count_failure_reports_actual_and_expected_counts(self) -> None:
        counts = {
            "source_callbacks": 42,
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
            "direct_continuation_receipts_accepted": 0,
            "direct_continuation_poll_elisions": 0,
            "direct_composite_attempts": 0,
            "direct_composite_completions": 0,
            "direct_composite_fallbacks": 0,
            "direct_composite_failures": 0,
            "direct_composite_callbacks_attempted": 0,
            "direct_composite_callbacks_completed": 0,
            "compute_dispatches": 3,
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
            "channel_receive_backpressured": 0,
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

        # Key assertion: a missing completion cannot enter an aggregate benchmark summary.
        counters["activations_completed"] = 2
        with pytest.raises(RuntimeError, match="inconsistent Runtime phase counters"):
            runner.validate_runtime_phase_counters(counters, counts, args)

        counters["activations_completed"] = 3
        counters["direct_continuation_receipts_accepted"] = 1
        with pytest.raises(RuntimeError, match="inconsistent Runtime phase counters"):
            runner.validate_runtime_phase_counters(counters, counts, args)

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
                    "candidate_to_dispatch_start_ns": complete_timing,
                    "compute_dispatch_ns": complete_timing,
                    "dispatch_return_to_retirement_ns": complete_timing,
                    "retirement_to_cut_ns": complete_timing,
                    "dispatch_return_to_cut_ns": complete_timing,
                },
                {
                    "activations": 2,
                    "candidate_to_dispatch_start_ns": {"samples": 2, "p50": 100},
                    "compute_dispatch_ns": {"samples": 2, "p50": 100},
                    "dispatch_return_to_retirement_ns": {"samples": 2, "p50": 100},
                    "retirement_to_cut_ns": {"samples": 2, "p50": 100},
                    "dispatch_return_to_cut_ns": {"samples": 2, "p50": 100},
                },
            ],
        }
        args = make_args(runtime_phase_counters=True)
        runner.validate_runtime_phase_trace(phase_trace, counts, args)

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
        result = {
            "schema": runner.RESULT_SCHEMA,
            "runtime_controls": {
                "direct_continuation_elision_disabled": False,
                "closed_island_composite_disabled": False,
            },
            "clock": clock_evidence(),
            "timestamp_instrumentation": timestamp_instrumentation(),
            "benchmark": "runtime_graph_pod_1to1",
            "scenario": "pod",
            "process_local_inline_delivery": "direct_retained",
            "queue_policy": "blocking",
            "workload": "latency",
            "partitioning": {
                "requested_algorithm": {
                    "id": "holoscan.partition.serial_baseline",
                    "version": 1,
                },
                "accepted_algorithm": {
                    "id": "holoscan.partition.serial_baseline",
                    "version": 1,
                },
                "selection_source": "core_default",
            },
            "geometry": {
                "samples": 100,
                "warmup": 10,
                "queue_depth": 1,
                "fanout": 1,
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
                "per_sink": [{"index": 0, "deliveries": 110, "final_sequence": 110}],
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
            runner.checked_result(path, make_args())

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
            },
            "clock": clock_evidence(),
            "timestamp_instrumentation": timestamp_instrumentation(),
            "benchmark": "runtime_graph_source_activation_notification_latency",
            "scenario": "source_only",
            "process_local_inline_delivery": "not_applicable",
            "queue_policy": "blocking",
            "workload": "notification_latency",
            "partitioning": {
                "requested_algorithm": {
                    "id": "holoscan.partition.serial_baseline",
                    "version": 1,
                },
                "accepted_algorithm": {
                    "id": "holoscan.partition.serial_baseline",
                    "version": 1,
                },
                "selection_source": "core_default",
            },
            "geometry": {
                "samples": 2,
                "warmup": 0,
                "queue_depth": 1,
                "fanout": 0,
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

        # Key assertion: the private A/B provenance is mandatory rather than inferred from the
        # requested command, so a stale binary cannot enter a matched summary silently.
        result.pop("runtime_controls")
        path.write_text(json.dumps(result), encoding="utf-8")
        with pytest.raises(RuntimeError, match="private Runtime controls"):
            runner.checked_result(path, args)
        result["runtime_controls"] = {
            "direct_continuation_elision_disabled": False,
            "closed_island_composite_disabled": False,
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
            },
            "clock": clock_evidence(),
            "timestamp_instrumentation": timestamp_instrumentation(),
            "benchmark": "runtime_graph_source_activation_saturated",
            "scenario": "source_only",
            "process_local_inline_delivery": "not_applicable",
            "queue_policy": "blocking",
            "workload": "saturated",
            "partitioning": {
                "requested_algorithm": {
                    "id": "holoscan.partition.serial_baseline",
                    "version": 1,
                },
                "accepted_algorithm": {
                    "id": "holoscan.partition.serial_baseline",
                    "version": 1,
                },
                "selection_source": "core_default",
            },
            "geometry": {
                "samples": 2,
                "warmup": 0,
                "queue_depth": 2,
                "fanout": 0,
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
        with pytest.raises(RuntimeError, match="exited with 9"):
            runner.run_repetition(make_args(), Path("/tmp/benchmark"), 0, tmp_path)
        repetition_dir = tmp_path / "repetition_000"
        assert next(repetition_dir.glob("stdout_*.log")).read_text() == "native stdout"
        assert next(repetition_dir.glob("stderr_*.log")).read_text() == "native stderr"

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
        with pytest.raises(TimeoutError, match="timed out"):
            runner.run_repetition(make_args(), Path("/tmp/benchmark"), 0, tmp_path)
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
        assert manifest["runtime_controls"] == {
            "direct_continuation_elision_disabled": True,
            "closed_island_composite_disabled": False,
        }
        assert "--disable-direct-continuation-elision" in manifest["native_commands"][0]


class TestGitRevision:
    def test_explicit_revision_survives_container_without_git_tree(self) -> None:
        # Key assertion: an injected revision avoids dependence on container Git metadata.
        assert runner.git_revision("abc123") == ("abc123", "command_line")
