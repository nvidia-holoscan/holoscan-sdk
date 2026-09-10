# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from types import SimpleNamespace

import pytest
import run_channel_benchmark as runner


def make_args(**updates: object) -> SimpleNamespace:
    values = {
        "domain": "interprocess",
        "expected_subscribers": 1,
        "payload": "tensor",
        "workload": "throughput",
        "publisher_period_us": 0,
        "subscriber_queue_depth": 64,
        "publisher_pacing": "sleep",
        "async_cuda_work": True,
        "tensor_storage": "device",
        "payload_validation": "final",
        "subscriber_cpu": -1,
    }
    values.update(updates)
    return SimpleNamespace(**values)


class TestRunnerValidation:
    @pytest.mark.parametrize("updates", [{"payload": "pod"}, {"tensor_storage": "pinned_host"}])
    def test_async_cuda_requires_device_tensor_payload(self, updates: dict[str, object]) -> None:
        with pytest.raises(ValueError, match="requires --payload tensor"):
            runner.validate_args(make_args(**updates))

    def test_async_cuda_rejects_all_payload_validation(self) -> None:
        with pytest.raises(ValueError, match="incompatible with --async-cuda-work"):
            runner.validate_args(make_args(payload_validation="all"))

    @pytest.mark.parametrize("domain", ["intraprocess_sidecar", "intraprocess_serial"])
    def test_async_cuda_rejects_sidecar_domains(self, domain: str) -> None:
        with pytest.raises(ValueError, match="not yet supported"):
            runner.validate_args(make_args(domain=domain))

    @pytest.mark.parametrize(
        ("updates", "message"),
        [
            (
                {"workload": "latency", "publisher_period_us": 0},
                "requires a nonzero --publisher-period-us",
            ),
            (
                {"workload": "latency", "publisher_period_us": 1, "subscriber_queue_depth": 0},
                "requires --subscriber-queue-depth 1 or 2",
            ),
            (
                {"workload": "latency", "publisher_period_us": 1, "subscriber_queue_depth": 3},
                "requires --subscriber-queue-depth 1 or 2",
            ),
            (
                {"workload": "throughput", "publisher_period_us": 1},
                "requires --workload latency",
            ),
            (
                {"workload": "throughput", "publisher_pacing": "spin"},
                "requires --workload latency",
            ),
        ],
    )
    def test_workload_and_pacing_constraints(
        self, updates: dict[str, object], message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            runner.validate_args(make_args(**updates))

    def test_serial_domain_rejects_subscriber_cpu(self) -> None:
        with pytest.raises(ValueError, match="does not accept --subscriber-cpu"):
            runner.validate_args(
                make_args(domain="intraprocess_serial", subscriber_cpu=2, async_cuda_work=False)
            )

    def test_serial_domain_rejects_subscriber_cpu_list(self) -> None:
        # Key setup: provide a subscriber CPU list to the single-thread serial domain.
        # Key assertion: validation rejects an option the serial coordinator would ignore.
        with pytest.raises(ValueError, match="does not accept --subscriber-cpus"):
            runner.validate_args(
                make_args(
                    domain="intraprocess_serial",
                    subscriber_cpus="2",
                    async_cuda_work=False,
                )
            )

    def test_serial_coordinator_omits_subscriber_cpu(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        observed_command: list[str] = []

        def run(command: list[str], **kwargs: object) -> SimpleNamespace:
            observed_command.extend(command)
            output = kwargs["stdout"]
            output.write('{"role":"publisher"}\n{"role":"subscriber"}\n')
            return SimpleNamespace(returncode=0)

        monkeypatch.setattr(runner.subprocess, "run", run)
        publisher, subscribers = runner.run_intraprocess(
            SimpleNamespace(
                domain="intraprocess_serial",
                publisher_cpu=2,
                skip_if_no_cuda=False,
                timeout_sec=1.0,
                expected_subscribers=1,
            ),
            ["benchmark"],
            [-1],
            tmp_path,
        )

        assert publisher["role"] == "publisher"
        assert subscribers == [{"role": "subscriber"}]
        assert "--subscriber-cpu" not in observed_command


class TestCPUTopology:
    def test_non_linux_topology_is_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(runner.sys, "platform", "win32")
        placement = runner.read_cpu_topology("publisher", 2)

        assert not placement["topology_available"]
        assert placement["topology_reason"] == "unsupported_platform"

    def test_smt_siblings_are_rejected(self) -> None:
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

        with pytest.raises(RuntimeError, match="share physical core"):
            runner.validate_distinct_physical_cores(placements)

    def test_distinct_physical_cores_are_accepted(self) -> None:
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

    def test_singleton_sibling_sets_override_ambiguous_core_ids(self) -> None:
        placements = [
            {
                "role": f"subscriber_{cpu}",
                "cpu": cpu,
                "pinned": True,
                "topology_available": True,
                "package_id": 0,
                "core_id": 0,
                "thread_siblings": str(cpu),
            }
            for cpu in range(8)
        ]

        runner.validate_distinct_physical_cores(placements)
