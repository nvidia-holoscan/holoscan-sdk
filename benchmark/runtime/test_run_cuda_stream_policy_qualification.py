# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import run_cuda_stream_policy_qualification as qualification


def make_args(**updates: object) -> SimpleNamespace:
    values = {
        "benchmark_binary": Path("/tmp/runtime_graph_benchmark"),
        "output_dir": Path("/tmp/cuda-stream-policy-qualification"),
        "revision": "0123456789abcdef",
        "samples": 100,
        "warmup": 10,
        "repetitions": 3,
        "tensor_bytes": 4096,
        "tensor_device": 0,
        "publisher_period_us": 5_000,
        "low_work_us": 1_000,
        "saturating_work_us": 1_000,
        "asymmetric_work_us": (500, 2_500),
        "overlap_samples": 20,
        "session_timeout_ms": 60_000,
        "process_timeout_sec": 120.0,
        "workloads": ["latency", "saturated"],
    }
    values.update(updates)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"samples": 0}, "--samples must be positive"),
        ({"repetitions": 1}, "--repetitions must be at least two"),
        ({"low_work_us": 0}, "CUDA work durations"),
        ({"overlap_samples": 0}, "--overlap-samples must be positive"),
        ({"revision": ""}, "--revision must be nonempty"),
    ],
)
def test_invalid_geometry_is_rejected(updates: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        qualification.validate_args(make_args(**updates))


def test_matrix_has_matched_control_overlap_saturation_and_asymmetry() -> None:
    matrix = qualification.regimes(make_args())
    assert [regime.name for regime in matrix] == [
        "no_device_work",
        "low_occupancy",
        "saturating",
        "asymmetric",
    ]
    assert matrix[0].branch_work_us is None
    assert matrix[1].branch_work_us == (1_000, 1_000)
    assert matrix[3].branch_work_us == (500, 2_500)


def test_native_command_changes_only_policy_for_a_matched_pair() -> None:
    args = make_args()
    regime = qualification.regimes(args)[1]
    shared = qualification.native_command(
        args,
        qualification.Cell("shared", regime),
        "latency",
        Path("/tmp/shared.json"),
        Path("/tmp/shared-plan.json"),
    )
    partitioned = qualification.native_command(
        args,
        qualification.Cell("per_partition", regime),
        "latency",
        Path("/tmp/partitioned.json"),
        Path("/tmp/partitioned-plan.json"),
    )
    ignored = {"--output", "--plan-output", "--stream-policy"}

    def normalized(command: list[str]) -> list[str]:
        result: list[str] = []
        index = 0
        while index < len(command):
            if command[index] in ignored:
                index += 2
            else:
                result.append(command[index])
                index += 1
        return result

    assert normalized(shared) == normalized(partitioned)
    assert shared[shared.index("--stream-policy") + 1] == "shared"
    assert partitioned[partitioned.index("--stream-policy") + 1] == "per_partition"


def test_checked_result_requires_exact_policy_and_cuda_timing(tmp_path: Path) -> None:
    cell = qualification.Cell("shared", qualification.regimes(make_args())[1])
    path = tmp_path / "result.json"
    result = {
        "scenario": "tensor",
        "workload": "latency",
        "cuda_execution": {
            "stream_policy": "shared",
            "plan_stream_policy": "shared",
            "stream_records": 1,
            "operator_binding_records": 3,
        },
        "cuda_work": {
            "profile": "low_occupancy",
            "branch_duration_ns": [{"samples": 20}, {"samples": 20}],
            "overlap_ns": {"samples": 20},
            "overlapping_samples": 0,
            "branch_completion_latency_ns": [{"samples": 20}, {"samples": 20}],
            "end_to_end_completion_latency_ns": {"samples": 20},
        },
        "cuda_resource_counters": {
            "enabled": True,
            "streams_created": 1,
            "streams_destroyed": 1,
            "events_created": 4,
            "events_destroyed": 4,
        },
        "timing": {
            "latency_ns": {"p50": 1, "p95": 2, "p99": 3, "max": 4},
            "source_to_sink_by_sink_ns": [
                {"latency_ns": {"p50": 1}},
                {"latency_ns": {"p50": 1}},
            ],
        },
        "validation": {"passed": True},
        "session": {"termination": "stopped", "diagnostics": []},
    }
    path.write_text(json.dumps(result), encoding="utf-8")
    assert qualification.checked_result(path, cell, "latency", 20) == result

    result["cuda_work"]["overlapping_samples"] = 1
    path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(RuntimeError, match="complete CUDA branch/overlap timing"):
        qualification.checked_result(path, cell, "latency", 20)

    result["cuda_work"]["overlapping_samples"] = 0
    invalid_branch = json.loads(json.dumps(result))
    invalid_branch["timing"]["source_to_sink_by_sink_ns"][0] = None
    invalid_results: list[object] = [
        [],
        {**result, "validation": []},
        {**result, "session": "invalid"},
        invalid_branch,
    ]
    for index, invalid in enumerate(invalid_results):
        invalid_path = tmp_path / f"invalid-{index}.json"
        invalid_path.write_text(json.dumps(invalid), encoding="utf-8")
        with pytest.raises(RuntimeError):
            qualification.checked_result(invalid_path, cell, "latency", 20)


def test_comparison_direction_is_favorable_without_defining_a_threshold() -> None:
    assert qualification.favorable_delta(90.0, 100.0, True) == pytest.approx(10.0)
    assert qualification.favorable_delta(110.0, 100.0, False) == pytest.approx(10.0)


def test_terminal_status_failure_preserves_qualification_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary = tmp_path / "runtime_graph_benchmark"
    binary.touch()
    output_dir = tmp_path / "qualification"
    args = make_args(benchmark_binary=binary, output_dir=output_dir, revision="current")
    monkeypatch.setattr(qualification, "parse_args", lambda: args)

    def fail(_args: object) -> int:
        raise RuntimeError("injected qualification failure")

    original_write_status = qualification.QualificationAttempt.write_status

    def fail_terminal_status(
        attempt: qualification.QualificationAttempt, status: str, detail: str | None = None
    ) -> None:
        if status != "running":
            raise OSError("injected status failure")
        original_write_status(attempt, status, detail)

    monkeypatch.setattr(qualification, "run_qualification", fail)
    monkeypatch.setattr(qualification.QualificationAttempt, "write_status", fail_terminal_status)

    with pytest.raises(RuntimeError, match="injected qualification failure") as caught:
        qualification.main()
    assert isinstance(caught.value.__cause__, OSError)
    attempt = json.loads((output_dir / "attempt.json").read_text(encoding="utf-8"))
    assert attempt["status"] == "running"
